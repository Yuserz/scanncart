"""Whether the server a remote detector backend calls is answering, watched continuously.

`detector_backend` has three values and two of them call an HTTP server that this app does not
supervise: `local_api` points at `local_inference_server.py` (a *separate process*, in its own venv,
started by hand) and `cloud_api` at Roboflow's serverless endpoint. Neither has anything watching it.
The failure that costs the most is the quiet one, and it is the shape this module exists for: the
capture starts, the camera works, the preview streams, and every `predictions` call fails — or, with
`local_api`, the server was simply never started, which is the ordinary case rather than a rare one,
since it is one command nobody in the app can run for you. What the operator sees is a live preview
with nothing on it.

The desktop answers the same question one level up and for the same reason: with nothing answering
the sidecar, every panel fails while the process stays alive (`desktop/src/main/sidecarHealth.ts`).
This is that monitor pointed at the other end of the line — an interval rather than one request per
call, transitions rather than a reading per probe, and the same three refinements, each of which is
about not *lying* rather than about efficiency:

* **Every probe is scheduled from the end of the last one**, so a server that accepts a connection
  and then never answers cannot stack up requests behind itself; the interval bounds the requests in
  flight at one, whatever the endpoint does.
* **An answer for a target the monitor has moved on from is discarded**, because an in-flight request
  outliving the setting that sent it is the ordinary case on a settings change — and reporting the
  old server's health under the new one's name is the one failure this whole module could introduce.
* **The verdict is deliberately slow** (`FAILURES_BEFORE_UNRESPONSIVE` misses), because telling an
  operator their inference server is down while it is merely busy is worse than noticing a dead one
  a few seconds later.

Two things it deliberately does not do. It does not probe at all when the backend is `native` — there
is no server to watch, and a request to a URL nobody configured is exactly the kind of thing this app
promises not to do — which is why `retarget` takes the *backend* and not only a URL. And it does not
ask whether the workflow works: any HTTP response counts as answering, including 401, 404 and 500,
because those all prove something is listening and the deeper check already exists in
`POST /api/detector/probe`, which sends a real frame through the workflow on demand. A monitor that
called the workflow every five seconds would be a workflow call every five seconds, which for
`cloud_api` is a paid request and for `local_api` is a model load.

What a verdict carries, and to whom, is the last thing worth knowing before touching this. The
*states* are transition-only, because a client renders the moment a server goes away and its coming
back, and a message saying "still down" every five seconds is a message nobody reads by the second
one. The *age* is the opposite: a standing verdict is re-confirmed every interval, so "how long ago
was this last checked" is a fresh number on every read and belongs on a read (`GET /api/health`,
where the Admin Panel polls it) rather than on the one push that reported the change. Both carry the
same `InferenceStatus`, so the two surfaces cannot disagree about the state, the endpoint or the
failure - only about how recently either was confirmed, which is the question each is positioned to
answer.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

#: The three things a client can be told. `unknown` is not "fine" and not "broken": it is a backend
#: that has not been asked yet (`native`, or the moment before the first probe lands), and a client
#: must render nothing for it rather than a verdict it does not have.
InferenceState = Literal["unknown", "ok", "unresponsive"]

UNKNOWN: InferenceState = "unknown"
OK: InferenceState = "ok"
UNRESPONSIVE: InferenceState = "unresponsive"

#: A reachability check, not a workflow call, so it gets a reachability deadline: the 5 s default of
#: `remote_timeout_s` is sized for a frame going through a model, and a probe that slow would spend
#: its whole interval waiting on a server that is not there.
PROBE_TIMEOUT_S = 2.0
PROBE_INTERVAL_S = 5.0
#: Three consecutive misses, matching the desktop's monitor. One miss is a blip; a server that is
#: genuinely down misses every probe, so the cost of the third is a few seconds of latency on the
#: notice, against never telling an operator to restart a server that is mid-request.
FAILURES_BEFORE_UNRESPONSIVE = 3

#: The local backend's server, and the venv it has to run in. `inference` pins numpy/opencv versions
#: that conflict with this process's ultralytics stack (`docs/DETECTOR_BACKENDS.md` §7a), so the
#: interpreter running *this* module is the one interpreter that cannot import it - which is why the
#: command below is not `python local_inference_server.py`.
LOCAL_SERVER_SCRIPT = "local_inference_server.py"
LOCAL_SERVER_VENV = ".venv-inference"

#: This module's own package directory: `sidecar/`. Both halves of that command live under it, and it
#: is the directory every documented command for this server is written from.
SIDECAR_DIR = Path(__file__).resolve().parents[1]


def local_server_command(
    sidecar_dir: Path | str | None = None, windows: bool | None = None
) -> str | None:
    """The command that starts the local inference server *on this machine*, or None when it cannot.

    An unresponsive `local_api` leaves the operator with a live preview that detects nothing, and the
    one fix no button in the app can perform is starting a second process - so the notice has to name
    a command to run. Only this side can write that sentence: the renderer sees no filesystem and no
    platform, and `python local_inference_server.py` - what it said before this existed - is a
    command that fails on the machine it is printed on, because the interpreter running the sidecar
    is precisely the one that cannot import `inference` (see `LOCAL_SERVER_VENV` above).

    `None` is the other honest answer, and the reason this looks at the disk rather than formatting a
    string: a checkout that has never set the local backend up has no venv to name, and handing an
    operator a command that cannot run is worse than sending them to the one step that comes first.
    Both inputs are parameters - they default to where this module actually is and to what this
    machine actually is - so each half can be tested without either.
    """
    root = SIDECAR_DIR if sidecar_dir is None else Path(sidecar_dir)
    on_windows = os.name == "nt" if windows is None else windows
    interpreter = root / LOCAL_SERVER_VENV / ("Scripts/python.exe" if on_windows else "bin/python")
    if not interpreter.is_file():
        return None
    # Relative to `sidecar/`, with forward slashes: the form the docs print it in (`§7a`) and the one
    # a POSIX shell runs - Git Bash included, which is the shell this repo assumes on Windows. The
    # backslash form is the one that breaks, since bash eats it as an escape. A reader in cmd or
    # PowerShell adds a `.\`; the notice cannot say that per shell, and this is the shell the rest of
    # the project's commands are written for.
    return f"{interpreter.relative_to(root).as_posix()} {LOCAL_SERVER_SCRIPT}"


@dataclass(frozen=True)
class InferenceStatus:
    """What is known about the server a remote backend calls, and why it is not answering.

    Frozen and copied rather than mutated in place: this is the value that gets stored on the app
    state, sent to every connected client and replayed on connect, so an update arriving between
    those three is the one way they could disagree.
    """

    backend: str = "native"
    url: str = ""
    state: InferenceState = UNKNOWN
    # The failure in the endpoint's own words, e.g. `ConnectError: ...`. Empty unless the state is
    # `unresponsive`, and it is a *reason* rather than the operator-facing sentence: the sentence
    # belongs wherever it is shown, and this travels with the verdict to whoever has to write one.
    detail: str = ""
    #: Epoch seconds of the most recent completed probe, refreshed on **every** probe rather than
    #: only when the verdict moves. That distinction is the whole point of the field: this monitor
    #: re-confirms a standing verdict every interval, and a reading that could only say when it last
    #: *changed* would describe a server that has been down for an hour as though nobody had looked
    #: since, when in fact it was checked five seconds ago. `0.0` means never probed — a target that
    #: was just configured — which is a different fact from an old probe, and `age_seconds` keeps
    #: them apart.
    checked_at: float = 0.0

    def age_seconds(self, now: float | None = None) -> float | None:
        """How long ago the last probe completed, or None when there has not been one.

        Computed at read time rather than carried as a number, because a value computed when the
        verdict was reached is wrong the moment it is read: the verdict goes on being re-confirmed
        every `PROBE_INTERVAL_S`, so the honest age stays near zero, and only a fresh computation
        can say that. Callers on different surfaces — the WebSocket push and the health read —
        therefore get their own answer instead of one number they would have to age themselves.
        """
        if self.checked_at <= 0.0:
            return None
        return max(0.0, (time.time() if now is None else now) - self.checked_at)


def probe_url(url: str, timeout_s: float = PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Does `url` answer? Returns `(answering, detail)` — the status code is deliberately ignored.

    A 401, a 404 and a 500 all mean a server is there, which is the whole question this asks; only a
    connect failure, a refused connection or a timeout means it is not. `raise_for_status` is
    therefore never called, and no API key is sent: the key belongs to the workflow call, and a probe
    that needed one would report a working server as broken whenever the key was missing.

    `httpx` is imported here rather than at module scope for the reason `app/roboflow.py` does it —
    this module is imported by the app's own startup, and the client library is only ever needed
    once a probe actually runs.
    """
    import httpx

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
            client.get(url)
    except Exception as exc:  # noqa: BLE001 - every failure here means the same thing
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


class InferenceHealthMonitor:
    """Probes one endpoint on an interval and reports only when the verdict changes.

    Transitions rather than every reading, because the callers are a WebSocket broadcast and a log
    line: a probe every five seconds that reported every time would be a message every five seconds
    saying nothing, and the two interesting moments are the server going away and its coming back.

    `probe` and `on_change` are injected, so the tests drive this with a synthetic endpoint and a
    recording callback rather than with sockets; `app/main.py` supplies the real ones, and
    `AppState.inference_monitor_factory` is the seam that lets a test keep this class's *timing* out
    of the picture.
    """

    def __init__(
        self,
        probe: Callable[[str], tuple[bool, str]],
        on_change: Callable[[InferenceStatus], None],
        *,
        interval_s: float = PROBE_INTERVAL_S,
        failures: int = FAILURES_BEFORE_UNRESPONSIVE,
    ) -> None:
        self._probe = probe
        self._on_change = on_change
        self._interval_s = interval_s
        self._failures = max(1, failures)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Woken by a stop or a retarget, so neither waits out an interval that can be seconds long.
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._target: tuple[str, str] | None = None
        self._misses = 0
        self._status = InferenceStatus()

    # -- what is watched ---------------------------------------------------------

    @property
    def status(self) -> InferenceStatus:
        with self._lock:
            return self._status

    def retarget(self, backend: str, url: str) -> None:
        """Point the monitor at the server `backend` calls, or at nothing.

        `native` — and a remote backend with no URL configured — means there is nothing to watch: the
        target becomes None, no request is ever made, and the state goes back to `unknown` so a
        client renders nothing rather than the last verdict about a server this app is no longer
        using. A *new* remote target resets to `unknown` for the same reason, in the other direction:
        the old endpoint's health says nothing about the new one, and a notice that arrived before
        the operator finished choosing would be about the wrong server.

        Called on startup and on every settings patch (it is a no-op when the target has not moved),
        so it is also what tells the loop to probe the new endpoint now instead of at the end of the
        old one's interval.
        """
        target = (backend, url) if backend != "native" and url else None
        with self._lock:
            if target == self._target:
                return
            self._target = target
            self._misses = 0
            # The URL is blanked when there is nothing to watch, rather than carried from the
            # argument: `backend_url` still answers with `cloud_api_url` for a `native` backend, and
            # a status that named a URL for a backend that does not call one would put an endpoint
            # into the log line and the client message for a server this app never contacts.
            # `checked_at` stays at its default here, which is what makes the new target's age read
            # as "never probed" rather than as the old endpoint's last check.
            status = InferenceStatus(
                backend=backend, url=url if target is not None else "", state=UNKNOWN
            )
            self._status = status
        # Outside the lock: this reaches the WebSocket manager and a print, and a callback that
        # called back in here would deadlock on a lock `threading` does not make reentrant.
        self._on_change(status)
        self._wake.set()

    # -- lifetime ----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="inference-health", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the loop, and do not wait out an interval to do it (this runs on shutdown).

        A thread rather than an asyncio task, and that is the same choice `Pipeline` and
        `CameraCapture` make: the work is a blocking HTTP call, `WSManager.submit` exists precisely
        to hand a result from a worker thread back to the event loop, and nothing here needs to be
        awaited.
        """
        thread, self._thread = self._thread, None
        self._stop.set()
        self._wake.set()
        if thread is not None:
            thread.join(timeout=2.0)

    # -- the loop ----------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            target = self._target
            if target is not None:
                self.probe_once(*target)
            self._wake.wait(self._interval_s)
            self._wake.clear()

    def probe_once(self, backend: str, url: str) -> None:
        """One probe, counted, and reported only if it settles the verdict.

        Public because it is the loop's whole body: a test can call it against a synthetic probe and
        assert the verdict without owning a clock, and the threaded behaviour is then tested on its
        own (that `start` runs it, that `stop` does not wait out an interval).

        The `except` is not redundant with the probe's own: `probe_url` catches for `httpx`-shaped
        failures, but this class takes *any* callable, and a probe that raised would otherwise kill
        the thread - leaving the app watching nothing, which is the state this module exists to
        prevent, arrived at silently.
        """
        try:
            answering, detail = self._probe(url)
        except Exception as exc:  # noqa: BLE001 - a raising probe is a miss, not a crash
            answering, detail = False, f"{type(exc).__name__}: {exc}"

        now = time.time()
        with self._lock:
            if self._stop.is_set() or (backend, url) != self._target:
                # Either the settings moved while this request was in the air - so its answer
                # describes the endpoint this app is no longer pointed at - or the app is shutting
                # down, where a verdict would be a broadcast into a closing server and a log line
                # for a state nobody will read.
                return
            # The reading is re-stamped here, before the transition tests below, because *every*
            # probe confirms or corrects the verdict and the age of the reading has to say so - a
            # server that has been down for an hour is checked every five seconds, and reporting
            # the hour would describe a monitor nobody is running. What stays transition-only is
            # the *broadcast*: `changed` below decides whether this reaches the WebSocket, so a
            # standing verdict costs one local update per interval and no traffic.
            changed = False
            if not answering:
                self._misses += 1
                if self._status.state != UNRESPONSIVE and self._misses >= self._failures:
                    status = InferenceStatus(
                        backend=backend, url=url, state=UNRESPONSIVE, detail=detail, checked_at=now
                    )
                    changed = True
                else:
                    status = replace(self._status, checked_at=now)
            else:
                self._misses = 0
                if self._status.state != OK:
                    status = InferenceStatus(backend=backend, url=url, state=OK, checked_at=now)
                    changed = True
                else:
                    status = replace(self._status, checked_at=now)
            self._status = status
        if changed:
            self._on_change(status)


def monitor_for(
    probe: Callable[[str], tuple[bool, str]],
    on_change: Callable[[InferenceStatus], None],
) -> InferenceHealthMonitor:
    """The monitor the app runs, as a function so a test can substitute one with other timings.

    `AppState.inference_monitor_factory` points here, which is the same seam as `source_factory` and
    `detector_factory`: the app owns the wiring (which probe, which callback), and a test owns the
    clock it would otherwise have to wait out.
    """
    return InferenceHealthMonitor(probe, on_change)


__all__ = [
    "FAILURES_BEFORE_UNRESPONSIVE",
    "LOCAL_SERVER_SCRIPT",
    "LOCAL_SERVER_VENV",
    "OK",
    "PROBE_INTERVAL_S",
    "PROBE_TIMEOUT_S",
    "SIDECAR_DIR",
    "UNKNOWN",
    "UNRESPONSIVE",
    "InferenceHealthMonitor",
    "InferenceState",
    "InferenceStatus",
    "local_server_command",
    "monitor_for",
    "probe_url",
]
