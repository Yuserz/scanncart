"""Capture failure and teardown paths.

These cover four bugs that were unreachable with the native detector and became
routine with the remote backends: a detector raising mid-capture, a detector
raising at start, and the resources start acquires never being freed.
"""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.camera import FakeFrameSource
from app.main import AppState, build_app
from app.pipeline import Pipeline
from app.roboflow import RoboflowAuthError, RoboflowUnavailable
from app.schemas import Detection
from app.settings import Settings


class TrackingSource(FakeFrameSource):
    """Frame source that records whether it was released."""

    def __init__(self, frames=None):
        super().__init__(frames if frames is not None else [np.zeros((4, 4, 3), np.uint8)] * 500)
        self.released = False
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def release(self) -> None:
        self.released = True

    def latest(self):
        return (0, np.zeros((4, 4, 3), dtype=np.uint8))


class ExplodingDetector:
    """Raises on the Nth infer, like a remote backend losing its server."""

    names: dict = {}

    def __init__(self, fail_after: int = 0, exc=None, gate: threading.Event | None = None):
        self._fail_after = fail_after
        self._exc = exc or RoboflowUnavailable("inference server went away")
        # When set, infer blocks here until the test opens the gate, so
        # "capture is running" can be asserted without racing the failure.
        self._gate = gate
        self.calls = 0
        self.closed = False

    def infer(self, frame):
        self.calls += 1
        if self.calls > self._fail_after:
            if self._gate is not None:
                self._gate.wait(timeout=5)
            raise self._exc
        return []

    def close(self) -> None:
        self.closed = True


class OkDetector:
    names: dict = {}

    def __init__(self):
        self.closed = False

    def infer(self, frame):
        return [Detection(track_id=1, cls="apple", conf=0.9, box=(0.1, 0.1, 0.2, 0.2))]

    def close(self) -> None:
        self.closed = True


# --- Pipeline._loop must not die silently -------------------------------


def test_pipeline_loop_reports_a_detector_failure_instead_of_dying_silently():
    seen: list[Exception] = []
    done = threading.Event()

    def on_error(exc):
        seen.append(exc)
        done.set()

    pipe = Pipeline(
        TrackingSource(), ExplodingDetector(), Settings(),
        on_message=lambda m: None, on_error=on_error,
    )
    pipe.start()
    assert done.wait(timeout=5), "on_error was never called - the thread died silently"
    assert isinstance(seen[0], RoboflowUnavailable)
    # is_running must not stay True with a dead thread, or capture looks alive.
    assert pipe.is_running is False


def test_pipeline_thread_actually_exits_after_a_failure():
    pipe = Pipeline(
        TrackingSource(), ExplodingDetector(), Settings(),
        on_message=lambda m: None, on_error=lambda e: None,
    )
    pipe.start()
    thread = pipe._thread
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_pipeline_survives_an_on_error_handler_that_itself_raises():
    def bad_handler(exc):
        raise RuntimeError("handler is broken")

    pipe = Pipeline(
        TrackingSource(), ExplodingDetector(), Settings(),
        on_message=lambda m: None, on_error=bad_handler,
    )
    pipe.start()
    pipe._thread.join(timeout=5)
    assert not pipe._thread.is_alive()


def test_pipeline_without_an_error_handler_still_stops_cleanly():
    pipe = Pipeline(
        TrackingSource(), ExplodingDetector(), Settings(), on_message=lambda m: None
    )
    pipe.start()
    pipe._thread.join(timeout=5)
    assert pipe.is_running is False


# --- The app-level failure path ------------------------------------------


@pytest.fixture
def state():
    source = TrackingSource()
    gate = threading.Event()
    detector = ExplodingDetector(fail_after=1, gate=gate)
    st = AppState(
        source_factory=lambda s: source,
        detector_factory=lambda s, d: detector,
        db_path=":memory:",
    )
    st._source = source
    st._detector = detector
    st._gate = gate
    return st


def test_a_mid_capture_failure_returns_the_sidecar_to_idle(state):
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        assert client.get("/api/health").json()["state"] == "running"
        state._gate.set()

        deadline = time.time() + 5
        while time.time() < deadline and state.state == "running":
            time.sleep(0.05)

        # Before the fix this stayed "running" forever: the thread was dead but
        # nothing updated the state, so Start became a silent no-op and the
        # Stop button that would recover was never rendered.
        assert state.state == "idle"
        assert client.get("/api/health").json()["state"] == "idle"


def test_a_mid_capture_failure_frees_the_camera_and_the_detector(state):
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        state._gate.set()
        deadline = time.time() + 5
        while time.time() < deadline and state.state == "running":
            time.sleep(0.05)

    assert state._source.released is True
    assert state._detector.closed is True


def test_start_works_again_after_a_mid_capture_failure(state):
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        state._gate.set()
        deadline = time.time() + 5
        while time.time() < deadline and state.state == "running":
            time.sleep(0.05)

        state.detector_factory = lambda s, d: OkDetector()
        state.source_factory = lambda s: TrackingSource()
        assert client.post("/api/capture/start").json() == {"state": "running"}
        # Stop it: the fakes never block, so a pipeline left running spins a
        # daemon thread at full speed for the rest of the session and skews
        # every timing-sensitive test that follows.
        client.post("/api/capture/stop")


# --- Start-time failure ---------------------------------------------------


def test_a_detector_that_fails_at_start_releases_the_camera():
    source = TrackingSource()

    def boom(settings, device):
        raise RoboflowAuthError("no api key")

    st = AppState(source_factory=lambda s: source, detector_factory=boom, db_path=":memory:")
    with TestClient(build_app(lambda: st)) as client:
        assert client.post("/api/capture/start").status_code == 401

    # source.open() had already started the capture thread. The old cleanup
    # looked for close(), which no frame source defines, so this leaked one
    # camera device and thread per failed attempt.
    assert source.opened is True
    assert source.released is True
    assert st.state == "idle"


# --- Normal stop ----------------------------------------------------------


def test_stop_releases_the_camera_and_the_detector():
    source = TrackingSource()
    detector = OkDetector()
    st = AppState(
        source_factory=lambda s: source,
        detector_factory=lambda s, d: detector,
        db_path=":memory:",
    )
    with TestClient(build_app(lambda: st)) as client:
        client.post("/api/capture/start")
        client.post("/api/capture/stop")

    # Each start/stop cycle used to leak an httpx pool and the camera.
    assert detector.closed is True
    assert source.released is True


def test_stop_is_idempotent():
    st = AppState(
        source_factory=lambda s: TrackingSource(),
        detector_factory=lambda s, d: OkDetector(),
        db_path=":memory:",
    )
    with TestClient(build_app(lambda: st)) as client:
        client.post("/api/capture/start")
        assert client.post("/api/capture/stop").json() == {"state": "idle"}
        assert client.post("/api/capture/stop").json() == {"state": "idle"}


def test_stop_after_a_mid_capture_failure_does_not_error(state):
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        state._gate.set()
        deadline = time.time() + 5
        while time.time() < deadline and state.state == "running":
            time.sleep(0.05)
        assert client.post("/api/capture/stop").json() == {"state": "idle"}


# --- teardown races -------------------------------------------------------


def test_stop_races_the_pipelines_own_error_teardown(state):
    """Both callers tear down; the loser must no-op, not crash.

    The pipeline thread's error handler and POST /api/capture/stop both run
    teardown. Without the lock, one cleared `state.pipeline` between the
    other's `is not None` check and its use, raising
    `AttributeError: 'NoneType' object has no attribute 'resolve_open_tracks'`
    out of the stop endpoint as a 500.
    """
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        # Fire Stop and the failure at the same moment.
        stop_result: list = []

        def do_stop():
            stop_result.append(client.post("/api/capture/stop"))

        t = threading.Thread(target=do_stop)
        state._gate.set()
        t.start()
        t.join(timeout=20)

    assert stop_result and stop_result[0].status_code == 200
    assert state.state == "idle"


def test_teardown_is_safe_to_call_twice_concurrently():
    from app.main import _teardown_capture

    st = AppState(
        source_factory=lambda s: TrackingSource(),
        detector_factory=lambda s, d: OkDetector(),
        db_path=":memory:",
    )
    with TestClient(build_app(lambda: st)) as client:
        client.post("/api/capture/start")
        threads = [threading.Thread(target=_teardown_capture, args=(st,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert st.state == "idle"
    assert st.pipeline is None


def test_start_does_not_block_the_event_loop_on_a_slow_camera():
    """A slow camera open must not freeze the rest of the sidecar.

    A Logitech StreamCam takes ~28 s to open. Inline on the event loop that
    stalled /api/health and the WebSocket handshake for the whole duration.
    """
    opening = threading.Event()
    release_open = threading.Event()

    class SlowSource(TrackingSource):
        def open(self) -> None:
            self.opened = True
            opening.set()
            release_open.wait(timeout=10)

    st = AppState(
        source_factory=lambda s: SlowSource(),
        detector_factory=lambda s, d: OkDetector(),
        db_path=":memory:",
    )
    with TestClient(build_app(lambda: st)) as client:
        started: list = []
        t = threading.Thread(target=lambda: started.append(client.post("/api/capture/start")))
        t.start()
        assert opening.wait(timeout=5), "start never reached source.open()"

        # The camera is still opening; health must still answer.
        assert client.get("/api/health").status_code == 200

        release_open.set()
        t.join(timeout=15)
        assert started and started[0].status_code == 200
        client.post("/api/capture/stop")
