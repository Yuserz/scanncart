"""What the app *does* with the monitor: when it starts, where it points, and who gets told.

`tests/test_inference_health.py` covers the verdict; this covers the wiring, which is the half that
can be wrong while every unit test passes - a monitor nobody starts, a startup that watches a URL
the settings do not name, or a transition that reaches the log but not a connected client. The probe
is injected everywhere (`AppState.inference_probe`), so nothing here opens a socket, and the monitor
comes from `AppState.inference_monitor_factory` with the real timings replaced: the app owns *which*
probe and *which* callback, the test owns the clock.

`TestClient(app)` without a context manager does not run the lifespan, which is why the handshake
test below can assert a stored verdict with nothing running, and why the ones that are about the
watch say `with` explicitly.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from app.inference_health import (
    OK,
    UNKNOWN,
    UNRESPONSIVE,
    InferenceHealthMonitor,
    InferenceStatus,
)
from app.main import AppState, build_app
from app.settings import Settings

LOCAL = "http://127.0.0.1:9001"
#: What `local_server_command` answers on a machine that has the local backend set up. The app's
#: default reads this checkout's disk, so every test here supplies its own: a payload assertion that
#: depended on whether someone had run `uv venv .venv-inference` would fail on a bare checkout and
#: pass on a working one, which is the one difference these tests must not be measuring.
START_COMMAND = ".venv-inference/Scripts/python.exe local_inference_server.py"


def settings(**over) -> Settings:
    s = Settings()
    for key, value in over.items():
        setattr(s, key, value)
    return s


class _Probe:
    """A probe that answers the same thing every time and remembers what it was asked.

    `release` is the seam that makes a transition land when the test wants it to: left unset, the
    probe blocks until the test sets it, so a verdict can be made to arrive *after* a client has
    connected rather than during startup, where no client would be there to receive it.
    """

    def __init__(self, answer: tuple[bool, str] = (True, "")) -> None:
        self.answer = answer
        self.urls: list[str] = []
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, url: str) -> tuple[bool, str]:
        with self._lock:
            self.urls.append(url)
        self.release.wait(5)
        return self.answer

    def wait_for(self, url: str, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if url in self.urls:
                    return True
            time.sleep(0.01)
        return False


def _factory(*, interval_s: float = 0.01, failures: int = 1) -> Callable[..., InferenceHealthMonitor]:
    """The app's monitor with a clock a test can wait out (`monitor_for`'s shape, other timings)."""

    def build(probe, on_change):
        return InferenceHealthMonitor(
            probe, on_change, interval_s=interval_s, failures=failures
        )

    return build


def _state(tmp_path, probe: _Probe, **over) -> AppState:
    return AppState(
        settings=settings(**over),
        settings_path=str(tmp_path / "settings.json"),
        inference_probe=probe,
        inference_monitor_factory=_factory(),
        local_server_command_factory=lambda: START_COMMAND,
    )


def test_the_handshake_carries_the_verdict_a_late_client_missed(tmp_path):
    """The monitor reports transitions, so a client that connects afterwards has only this."""
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)
    state.inference_health = InferenceStatus(
        backend="local_api", url=LOCAL, state=UNRESPONSIVE, detail="ConnectError: refused"
    )

    client = TestClient(build_app(lambda: state))
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "status"
        message = ws.receive_json()

    assert message["type"] == "inference"
    assert message["state"] == UNRESPONSIVE
    assert message["backend"] == "local_api"
    assert message["url"] == LOCAL
    assert message["detail"] == "ConnectError: refused"


def test_the_handshake_says_unknown_for_a_native_backend(tmp_path):
    """`unknown` is what a client renders nothing for, and it is the ordinary answer here."""
    probe = _Probe()
    state = _state(tmp_path, probe)

    client = TestClient(build_app(lambda: state))
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        message = ws.receive_json()

    assert message["type"] == "inference"
    assert message["state"] == UNKNOWN
    assert message["url"] == ""
    assert probe.urls == []


def test_the_health_read_carries_the_verdict_and_when_it_was_checked(tmp_path):
    """The Admin Panel's half of the verdict.

    The push reports a *change*; this is the standing reading beside the backend picker, and the age
    is the part that has to come from here: the endpoint is polled, so the number is recomputed each
    time, where a pushed copy would age from the moment of the change.
    """
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)
    state.inference_health = InferenceStatus(
        backend="local_api",
        url=LOCAL,
        state=UNRESPONSIVE,
        detail="ConnectError: refused",
        checked_at=time.time() - 4.0,
    )

    body = TestClient(build_app(lambda: state)).get("/api/health").json()

    assert body["inference"]["url"] == LOCAL
    assert body["inference"]["state"] == UNRESPONSIVE
    assert body["inference"]["detail"] == "ConnectError: refused"
    assert 4.0 <= body["inference"]["age_seconds"] < 60.0


def test_a_native_backend_reports_no_server_to_watch(tmp_path):
    """`null` rather than a verdict: the panel needs "there is no endpoint" told apart from "there is
    one and nobody has asked it yet", and an empty-url payload would be neither."""
    probe = _Probe()
    state = _state(tmp_path, probe)

    body = TestClient(build_app(lambda: state)).get("/api/health").json()

    assert body["inference"] is None


def test_the_age_is_recomputed_on_every_read(tmp_path):
    """The reading stays fresh while nothing is broadcast.

    Once the verdict is `ok` the monitor stops telling clients about it - that is the transitions-only
    design - so if this number were the one from the push, a server confirmed seconds ago would show
    the age of the change. It is instead recomputed from the last probe, which the loop keeps
    refreshing, so the age stays near zero however long the verdict stands.
    """
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)

    with TestClient(build_app(lambda: state)) as client:
        assert probe.wait_for(LOCAL)
        probe.release.set()
        deadline = time.monotonic() + 5.0
        while state.inference_health.state != OK and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state.inference_health.state == OK

        first = client.get("/api/health").json()["inference"]["age_seconds"]
        # Several probe intervals (the factory uses 0.01 s), during which nothing is broadcast.
        time.sleep(0.2)
        second = client.get("/api/health").json()["inference"]["age_seconds"]

    assert first is not None and second is not None
    # If the age came from the transition it would now read ~0.2 s plus however long the verdict had
    # already stood. Re-stamped every probe, it is under the probe timeout and does not grow.
    assert second < 0.15, f"the age of a standing verdict grew between reads: {first} -> {second}"


def test_the_lifespan_starts_the_watch_and_stops_it(tmp_path):
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)

    assert state.inference_monitor is None
    with TestClient(build_app(lambda: state)):
        assert state.inference_monitor is not None, "the app started without watching anything"
        assert probe.wait_for(LOCAL), "the watch never probed the configured URL"
    assert state.inference_monitor is None, "the monitor outlived the app"


def test_a_native_backend_watches_nothing_at_startup(tmp_path):
    probe = _Probe()
    state = _state(tmp_path, probe)

    with TestClient(build_app(lambda: state)):
        time.sleep(0.1)
        assert probe.urls == [], "a native backend must not call a URL nobody configured"


def test_a_verdict_reaches_a_connected_client(tmp_path):
    """The broadcast half: the monitor is what notices, the client is what needs telling."""
    probe = _Probe(answer=(False, "ConnectError: refused"))
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)

    with TestClient(build_app(lambda: state)) as client:
        with client.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["type"] == "status"
            assert ws.receive_json()["state"] == UNKNOWN, "the probe is still in the air"
            # ...which is what `release` is for: the answer lands now, with a client watching.
            probe.release.set()
            message = ws.receive_json()

    assert message["type"] == "inference"
    assert message["state"] == UNRESPONSIVE
    assert message["detail"] == "ConnectError: refused"
    assert state.inference_health.state == UNRESPONSIVE


def test_the_verdict_carries_the_command_that_starts_the_server(tmp_path):
    """The remedy rides on the verdict, because only the sidecar can write it.

    The renderer knows no filesystem and no platform, and the command is not `python
    local_inference_server.py`: that server runs in its own venv, which the interpreter running the
    sidecar cannot import. Both surfaces carry it off this one payload, so the pushed copy is what
    the Live view renders.
    """
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)
    state.inference_health = InferenceStatus(
        backend="local_api", url=LOCAL, state=UNRESPONSIVE, detail="ConnectError: refused"
    )

    body = TestClient(build_app(lambda: state)).get("/api/health").json()
    assert body["inference"]["local_server_command"] == START_COMMAND

    client = TestClient(build_app(lambda: state))
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        message = ws.receive_json()
    assert message["local_server_command"] == START_COMMAND


def test_a_cloud_verdict_carries_no_local_command(tmp_path):
    """Gated on `local_api`, like the URL is gated on having a server to watch at all.

    A command for a backend that does not use it would be a notice telling the operator to start a
    server this configuration never calls.
    """
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="cloud_api")
    state.inference_health = InferenceStatus(
        backend="cloud_api", url="https://detect.roboflow.com", state=UNRESPONSIVE, detail="Timeout"
    )

    body = TestClient(build_app(lambda: state)).get("/api/health").json()
    assert body["inference"]["local_server_command"] is None


def test_the_remedy_is_read_again_on_every_payload(tmp_path):
    """A venv created while the app is running has to stop the notice pointing at the setup step.

    Which is the whole reason the app state holds a factory rather than the answer: the operator
    reads the notice, follows it, and between those two moments the thing it said was missing appears.
    """
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)
    # No venv yet, then one - read in order, one payload at a time.
    answers: list[str | None] = [None, START_COMMAND]
    state.local_server_command_factory = lambda: answers.pop(0)
    state.inference_health = InferenceStatus(
        backend="local_api", url=LOCAL, state=UNRESPONSIVE, detail="ConnectError: refused"
    )

    client = TestClient(build_app(lambda: state))
    assert client.get("/api/health").json()["inference"]["local_server_command"] is None

    # The operator did what that notice said and ran `uv venv .venv-inference`.
    assert client.get("/api/health").json()["inference"]["local_server_command"] == START_COMMAND


def test_changing_the_backend_moves_what_is_watched(tmp_path):
    """A patch that moves the backend or its URL has to move the watch, or the notice is about the
    server the app no longer calls."""
    probe = _Probe()
    state = _state(tmp_path, probe)

    other = "http://127.0.0.1:9333"
    with TestClient(build_app(lambda: state)) as client:
        assert probe.urls == []
        response = client.patch(
            "/api/settings?persist=false",
            json={"detector_backend": "local_api", "local_api_url": other},
        )
        assert response.status_code == 200, response.text
        assert probe.wait_for(other), f"the watch never moved to {other}"
        assert state.inference_health.url == other

        # ...and back to native, which calls nothing: no further probes.
        assert client.patch("/api/settings?persist=false", json={"detector_backend": "native"}).status_code == 200
        seen = len(probe.urls)
        time.sleep(0.1)
        assert len(probe.urls) == seen, "a native backend kept probing an endpoint"
        assert state.inference_health.state == UNKNOWN
        assert state.inference_health.url == ""


def test_a_recovering_server_is_reported_to_clients_too(tmp_path):
    """Both directions, and the count restarted in between - `retarget` is not the only path."""
    probe = _Probe(answer=(False, "ConnectError: refused"))
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)

    with TestClient(build_app(lambda: state)) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            assert ws.receive_json()["state"] == UNKNOWN
            probe.release.set()
            assert ws.receive_json()["state"] == UNRESPONSIVE

            probe.answer = (True, "")
            probe.release.clear()
            probe.release.set()
            message = ws.receive_json()

    assert message["state"] == OK
    assert state.inference_health.state == OK
