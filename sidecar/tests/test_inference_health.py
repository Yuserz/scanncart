"""The monitor that watches a remote backend's server, and the two ways it could lie.

`app/inference_health.py` exists because `local_api` points at a process nobody supervises, so the
interesting cases are all about the verdict being *wrong*: reporting the old endpoint's health under
a new one's name, reporting a server that is merely busy, or dying quietly and watching nothing at
all. The verdicts are asserted through `probe_once` (the loop's whole body, called directly) so these
tests are about the state machine rather than about sleeping; the threading is tested separately, and
`probe_url` against a real socket pair, because "any HTTP response counts as answering" is the rule
the whole design rests on and a fake cannot show it.
"""

from __future__ import annotations

import http.server
import socket
import threading
import time

from app.inference_health import (
    OK,
    UNKNOWN,
    UNRESPONSIVE,
    InferenceHealthMonitor,
    InferenceStatus,
    probe_url,
)


class _Recorder:
    """The monitor's `on_change`, in list form: what a client would have been told, in order."""

    def __init__(self) -> None:
        self.statuses: list[InferenceStatus] = []

    def __call__(self, status: InferenceStatus) -> None:
        self.statuses.append(status)

    @property
    def states(self) -> list[str]:
        return [s.state for s in self.statuses]


def _monitor(probe, *, failures: int = 3, interval_s: float = 0.01):
    seen = _Recorder()
    return InferenceHealthMonitor(probe, seen, interval_s=interval_s, failures=failures), seen


def _answering(answers):
    """A probe that walks a list of answers (the last one repeating) and counts its calls."""
    calls: list[str] = []

    def probe(url: str):
        calls.append(url)
        answer = answers[min(len(calls) - 1, len(answers) - 1)]
        return answer if isinstance(answer, tuple) else (answer, "" if answer else "ConnectError: refused")

    return probe, calls


# --- the age of the reading ---------------------------------------------------
#
# `checked_at` is re-stamped on every probe, not only when the verdict moves, and the difference is
# the whole point of the field: this monitor re-confirms a standing verdict every interval, so a
# server that has been down for an hour was still checked five seconds ago. A number that only moved
# on a *transition* would describe a monitor nobody is running - and it is the number the Admin
# Panel shows beside the backend picker, where the operator decides whether to trust the verdict.


def test_age_is_computed_at_read_time_rather_than_carried():
    """A value computed when the verdict was reached is wrong the moment it is read."""
    assert InferenceStatus(checked_at=1000.0).age_seconds(now=1002.5) == 2.5
    # An unprobed target is not "checked a very long time ago".
    assert InferenceStatus().age_seconds(now=1000.0) is None


def test_a_verdict_carries_when_its_probe_completed():
    probe, _ = _answering([True])
    monitor, _ = _monitor(probe, failures=1)
    monitor.retarget("local_api", "http://127.0.0.1:9001")
    monitor.probe_once("local_api", "http://127.0.0.1:9001")

    age = monitor.status.age_seconds()
    assert age is not None and 0.0 <= age < 5.0


def test_the_reading_is_re_stamped_even_when_the_verdict_does_not_change():
    """Both halves at once: the stored reading moves, the broadcast does not.

    A second success is not news to a client - it already knows the server answers - but it *is*
    fresh evidence for the age, and dropping it would make a standing verdict look un-checked.
    """
    probe, _ = _answering([True])
    monitor, seen = _monitor(probe, failures=1)
    monitor.retarget("local_api", "http://127.0.0.1:9001")
    monitor.probe_once("local_api", "http://127.0.0.1:9001")
    first = monitor.status.checked_at

    time.sleep(0.02)
    monitor.probe_once("local_api", "http://127.0.0.1:9001")

    assert monitor.status.checked_at > first
    assert monitor.status.state == OK
    assert seen.states == [UNKNOWN, OK], "a re-confirmation was broadcast as a change"


def test_a_new_target_forgets_when_the_old_one_was_checked():
    """The same rule as the state, in the time dimension: the old endpoint's last check says nothing
    about the new one, and an age carried across would date the new verdict from the old server."""
    probe, _ = _answering([True])
    monitor, _ = _monitor(probe, failures=1)
    monitor.retarget("local_api", "http://127.0.0.1:9001")
    monitor.probe_once("local_api", "http://127.0.0.1:9001")
    assert monitor.status.age_seconds() is not None

    monitor.retarget("local_api", "http://127.0.0.1:9002")

    assert monitor.status.age_seconds() is None


# --- the verdict -------------------------------------------------------------


def test_a_server_that_stops_answering_is_reported_after_the_threshold():
    probe, _ = _answering([False])
    monitor, seen = _monitor(probe, failures=3)
    monitor.retarget("local_api", "http://127.0.0.1:9001")

    monitor.probe_once("local_api", "http://127.0.0.1:9001")
    monitor.probe_once("local_api", "http://127.0.0.1:9001")
    assert seen.states == [UNKNOWN], "two misses are not a verdict"

    monitor.probe_once("local_api", "http://127.0.0.1:9001")
    assert seen.states == [UNKNOWN, UNRESPONSIVE]
    assert seen.statuses[-1].detail, "the verdict has to carry why"
    assert seen.statuses[-1].url == "http://127.0.0.1:9001"


def test_further_misses_after_the_verdict_are_not_reported():
    """Transitions, not readings: the callers are a broadcast and a log line."""
    probe, calls = _answering([False])
    monitor, seen = _monitor(probe, failures=1)
    monitor.retarget("local_api", "u")

    for _ in range(4):
        monitor.probe_once("local_api", "u")
    assert len(calls) == 4
    assert seen.states == [UNKNOWN, UNRESPONSIVE]


def test_a_server_coming_back_is_reported_and_resets_the_count():
    probe, _ = _answering([False, False, True, False])
    monitor, seen = _monitor(probe, failures=2)
    monitor.retarget("local_api", "u")

    monitor.probe_once("local_api", "u")
    monitor.probe_once("local_api", "u")
    assert seen.states == [UNKNOWN, UNRESPONSIVE]

    monitor.probe_once("local_api", "u")
    assert seen.states == [UNKNOWN, UNRESPONSIVE, OK]
    assert seen.statuses[-1].detail == "", "a server that answers has no reason to give"

    # The miss count restarted with the recovery, so one more miss is again not a verdict.
    monitor.probe_once("local_api", "u")
    assert seen.states == [UNKNOWN, UNRESPONSIVE, OK]


def test_a_probe_that_raises_is_a_miss_and_not_a_dead_monitor():
    def probe(_url: str):
        raise RuntimeError("the endpoint's client blew up")

    monitor, seen = _monitor(probe, failures=1)
    monitor.retarget("cloud_api", "https://serverless.roboflow.com")
    monitor.probe_once("cloud_api", "https://serverless.roboflow.com")

    assert seen.states == [UNKNOWN, UNRESPONSIVE]
    assert "RuntimeError" in seen.statuses[-1].detail


# --- what is watched ---------------------------------------------------------


def test_native_is_never_probed():
    """The app's promise: no request to a URL nobody configured."""
    probe, calls = _answering([True])
    monitor, seen = _monitor(probe)
    monitor.retarget("native", "http://127.0.0.1:9001")
    monitor.start()
    try:
        time.sleep(0.05)
    finally:
        monitor.stop()

    assert calls == [], "a native backend has no server to watch"
    assert seen.states == [], "nothing changed, so there is nothing to report"
    assert monitor.status.state == UNKNOWN
    assert monitor.status.url == ""


def test_switching_to_native_withdraws_the_endpoint_and_reports_nothing_to_expect():
    """A backend that stops calling a server takes its URL with it.

    `backend_url` answers with the *cloud* URL for a `native` backend (it has to answer something),
    so a status that carried the argument through would name an endpoint this app does not contact
    in the log line and in a message a client renders.
    """
    probe, _ = _answering([True])
    monitor, seen = _monitor(probe, failures=1)
    monitor.retarget("local_api", "http://127.0.0.1:9001")
    monitor.probe_once("local_api", "http://127.0.0.1:9001")
    assert seen.states == [UNKNOWN, OK]

    monitor.retarget("native", "https://serverless.roboflow.com")
    assert seen.states == [UNKNOWN, OK, UNKNOWN]
    assert seen.statuses[-1].url == "", "a native backend calls no server"


def test_a_remote_backend_with_no_url_is_never_probed():
    probe, calls = _answering([True])
    monitor, _ = _monitor(probe)
    monitor.retarget("local_api", "")
    monitor.start()
    try:
        time.sleep(0.05)
    finally:
        monitor.stop()
    assert calls == []


def test_a_new_target_goes_back_to_unknown():
    """The old endpoint's health is not evidence about the new one."""
    probe, _ = _answering([True])
    monitor, seen = _monitor(probe, failures=1)
    monitor.retarget("local_api", "http://127.0.0.1:9001")
    monitor.probe_once("local_api", "http://127.0.0.1:9001")
    assert seen.states == [UNKNOWN, OK]

    monitor.retarget("cloud_api", "https://serverless.roboflow.com")
    assert seen.states == [UNKNOWN, OK, UNKNOWN]
    assert seen.statuses[-1].url == "https://serverless.roboflow.com"


def test_retargeting_to_the_same_endpoint_reports_nothing():
    probe, _ = _answering([True])
    monitor, seen = _monitor(probe)
    monitor.retarget("local_api", "u")
    monitor.retarget("local_api", "u")
    assert seen.states == [UNKNOWN], "a settings patch that did not move the target is not news"


def test_an_answer_for_the_old_target_is_discarded():
    """The one failure this module could introduce: the old server's verdict under the new name.

    Reached here by having the probe itself move the target while its request is 'in the air',
    which is what a settings change during a request looks like from in here.
    """
    monitor: InferenceHealthMonitor
    seen = _Recorder()

    def probe(_url: str):
        monitor.retarget("cloud_api", "https://elsewhere")
        return False, "ConnectError: refused"

    monitor = InferenceHealthMonitor(probe, seen, interval_s=0.01, failures=1)
    monitor.retarget("local_api", "http://127.0.0.1:9001")
    monitor.probe_once("local_api", "http://127.0.0.1:9001")

    assert seen.states == [UNKNOWN, UNKNOWN], (
        "the answer described the endpoint this app had already stopped watching"
    )
    assert seen.statuses[-1].url == "https://elsewhere"
    assert monitor.status.state == UNKNOWN


# --- the thread --------------------------------------------------------------


def test_a_verdict_that_lands_after_stop_is_not_reported():
    """Shutdown is not a verdict: no store, no broadcast, no log line for a state nobody reads."""
    flying = threading.Event()

    def probe(_url: str):
        flying.wait(5)
        return False, "ConnectError: refused"

    monitor, seen = _monitor(probe, failures=1)
    monitor.retarget("local_api", "u")
    worker = threading.Thread(target=monitor.probe_once, args=("local_api", "u"))
    worker.start()
    time.sleep(0.05)  # let the probe get in the air

    monitor.stop()
    flying.set()
    worker.join(timeout=5)

    assert seen.states == [UNKNOWN]


def test_start_runs_the_loop_and_stop_does_not_wait_out_the_interval():
    """`stop` runs on shutdown, so it cannot be the thing that holds it up."""
    probe, calls = _answering([True])
    # The first probe happens as soon as the loop starts (`retarget` wakes it), which is what makes
    # this deterministic with an interval the test would never wait out.
    monitor, seen = _monitor(probe, interval_s=30.0)
    monitor.retarget("local_api", "u")

    monitor.start()
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        time.sleep(0.005)
    assert calls, "the monitor started but never probed"
    assert seen.states == [UNKNOWN, OK]

    started = time.monotonic()
    monitor.stop()
    assert time.monotonic() - started < 1.0, "stop waited out the interval instead of waking it"


# --- what "answering" means --------------------------------------------------


def _http_server(status: int) -> tuple[str, http.server.ThreadingHTTPServer]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):  # keep pytest's output clean
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def test_any_http_response_counts_as_answering():
    """401, 404 and 500 all prove something is listening - which is the whole question.

    A 500 is the awkward case and the reason this is asserted rather than argued: a server that is
    *failing* is not a server that is missing, and the deeper question (does the workflow work?) is
    what `POST /api/detector/probe` answers, on a real frame, when somebody asks it.
    """
    for status in (200, 401, 404, 500, 503):
        url, server = _http_server(status)
        try:
            answering, detail = probe_url(url, timeout_s=2.0)
        finally:
            server.shutdown()
            server.server_close()
        assert answering is True, f"a {status} is a server that answered"
        assert detail == ""


def test_a_port_with_nothing_on_it_is_not_answering():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # Closed, so the connect is refused rather than accepted; the port is not reused between the
    # bind above and this probe in practice, and a refused connect is the failure being measured.
    answering, detail = probe_url(f"http://127.0.0.1:{port}", timeout_s=2.0)
    assert answering is False
    assert detail, "a miss has to carry a reason for the operator"
