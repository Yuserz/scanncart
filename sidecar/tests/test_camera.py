import threading
import time

import cv2
import numpy as np
import pytest

from app.camera import CameraCapture, LatestFrameBuffer, FakeFrameSource


def _frame(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val, dtype=np.uint8)


def test_buffer_returns_none_when_empty():
    buf = LatestFrameBuffer()
    assert buf.get() is None


def test_buffer_newest_wins():
    buf = LatestFrameBuffer()
    buf.put(1, _frame(10))
    buf.put(2, _frame(20))
    seq, frame = buf.get()
    assert seq == 2
    assert frame[0, 0, 0] == 20


def test_fake_frame_source_yields_then_none():
    src = FakeFrameSource([_frame(1), _frame(2)], fps=30.0)
    src.open()
    assert src.read()[0, 0, 0] == 1
    assert src.read()[0, 0, 0] == 2
    assert src.read() is None
    assert src.fps == 30.0
    src.release()


# --- a device that goes away ---------------------------------------------


class _FailingCap:
    """Opens fine, then never yields a frame — an invalidated device."""

    def __init__(self, fail_after=0):
        self.reads = 0
        self._fail_after = fail_after

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        self.reads += 1
        if self.reads <= self._fail_after:
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return False, None

    def release(self):
        pass


def test_capture_gives_up_on_a_device_that_stopped_delivering():
    """It used to retry ~200x/second forever, burning a core and writing
    ~23,000 OpenCV warnings while the app looked like it was running."""
    cap = _FailingCap()
    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: cap)
    c.FAILURE_TIMEOUT_S = 0.3  # the real 3 s deadline, shortened for the test
    c.open()
    deadline = time.time() + 5
    while time.time() < deadline and not c.failure:
        time.sleep(0.02)
    c.release()

    assert c.failure is not None
    assert "stopped delivering frames" in c.failure
    # Backed off rather than spinning: ~200/s unthrottled would be far more.
    assert cap.reads < 100


def test_a_transient_read_failure_does_not_kill_capture():
    # One bad read among good ones must not take the camera down.
    class _Flaky(_FailingCap):
        def read(self):
            self.reads += 1
            if self.reads % 10 == 0:
                return False, None
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    cap = _Flaky()
    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: cap)
    c.open()
    time.sleep(0.3)
    failure = c.failure
    c.release()

    assert failure is None


def test_pipeline_reports_a_dead_camera_instead_of_freezing():
    """Otherwise capture stays 'running' with a frozen image and no reason."""
    from app.pipeline import Pipeline
    from app.settings import Settings

    class _DeadSource:
        width, height, fps = 128, 96, 30.0
        failure = "Camera 0 stopped delivering frames after 150 attempts."

        def latest(self):
            return None

    pipe = Pipeline(_DeadSource(), None, Settings(), on_message=lambda m: None)
    with pytest.raises(RuntimeError, match="stopped delivering frames"):
        pipe.process_once()


# --- measured delivery rate ------------------------------------------------


def test_capture_reports_the_rate_it_actually_delivers():
    """capture_fps used to report the *requested* value: the UI showed 60
    while the camera delivered 12, hiding a 5x shortfall all session."""
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self):
            time.sleep(0.01)  # ~100 fps ceiling
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open()
    time.sleep(1.2)
    rate = c.measured_fps
    c.release()

    assert rate > 10.0          # it is measuring something real
    assert rate < 300.0         # and not nonsense


def test_measured_fps_does_not_inflate_on_burst_then_stall():
    """A burst of frames close together followed by a stall used to report
    an inflated rate: (n-1)/span blows up when span is tiny. Two frames 1ms
    apart within the last second, with nothing more recent (i.e. a stall
    right after the burst), must report a plain count over the window (2.0),
    not ~1000."""
    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: object())
    now = time.monotonic()
    c._read_times.append(now - 0.5)
    c._read_times.append(now - 0.499)  # 1ms after the previous sample

    rate = c.measured_fps

    assert rate == pytest.approx(2.0)


def test_measured_fps_is_zero_before_any_frame():
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self): return False, None
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    assert c.measured_fps == 0.0


def test_open_applies_configured_controls():
    """Auto exposure and face-tracking autofocus are wrong for a counter:
    the StreamCam's smart AF/AE follows faces, and there is no face here."""
    sets = []

    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value):
            sets.append((prop, value)); return True
        def get(self, prop): return 0
        def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap(),
                      brightness=180.0, exposure=-6.0, autofocus=False, focus=30.0)
    c.open(); c.release()

    assert (cv2.CAP_PROP_BRIGHTNESS, 180.0) in sets
    assert (cv2.CAP_PROP_EXPOSURE, -6.0) in sets
    assert (cv2.CAP_PROP_AUTOFOCUS, 0) in sets
    assert (cv2.CAP_PROP_FOCUS, 30.0) in sets


def test_unset_controls_are_left_alone():
    """None means 'do not touch', so existing behaviour is unchanged."""
    sets = []

    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value):
            sets.append(prop); return True
        def get(self, prop): return 0
        def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open(); c.release()

    assert cv2.CAP_PROP_BRIGHTNESS not in sets
    assert cv2.CAP_PROP_EXPOSURE not in sets


def test_measured_fps_survives_concurrent_reads():
    """measured_fps used to iterate _read_times directly while the capture
    thread appended to it, which can raise 'deque mutated during iteration'.
    Hammer both from separate threads and confirm no exception surfaces."""
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open()

    errors: list[BaseException] = []

    def _hammer():
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                c.measured_fps
            except BaseException as exc:  # noqa: BLE001 - pin the race, not a specific type
                errors.append(exc)
                return

    readers = [threading.Thread(target=_hammer) for _ in range(4)]
    for t in readers:
        t.start()
    for t in readers:
        t.join()
    c.release()

    assert errors == []


# --- live control changes -------------------------------------------------


class _RecordingCap:
    """Opens fine, yields frames forever, and records every set() with the
    name of the thread that made it."""

    def __init__(self):
        self.sets: list[tuple[int, object, str]] = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value, threading.current_thread().name))
        return True

    def read(self):
        time.sleep(0.001)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _wrote(cap, prop, value) -> bool:
    return any(p == prop and v == value for p, v, _ in cap.sets)


def _wait_for(predicate, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_control_writes_happen_on_the_capture_thread():
    """cv2.VideoCapture is not thread-safe and _loop is calling read() on a
    background thread, so a set() issued from the caller's thread would race
    it. The write must be deferred to the thread that owns the handle."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=140.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 140.0))
        writers = {t for p, _, t in cap.sets if p == cv2.CAP_PROP_BRIGHTNESS}
        assert threading.current_thread().name not in writers
    finally:
        src.release()


def test_set_controls_coalesces_a_fast_drag():
    """A slider drag emits dozens of values. Only the newest matters, and
    applying every one would stall reads behind a queue of set() calls."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=100.0)
        src.set_controls(brightness=110.0)
        src.set_controls(brightness=120.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 120.0))
        written = [v for p, v, _ in cap.sets if p == cv2.CAP_PROP_BRIGHTNESS]
        assert 110.0 not in written
    finally:
        src.release()


def test_controls_set_live_survive_a_reopen():
    """A restart (resolution change, say) rebuilds the handle. Values tuned
    live must come back with it, or a restart silently reverts them."""
    caps = []

    def factory(index):
        cap = _RecordingCap()
        caps.append(cap)
        return cap

    src = CameraCapture(0, 4, 4, 30, cap_factory=factory)
    src.open()
    src.set_controls(brightness=140.0)
    assert _wait_for(lambda: _wrote(caps[0], cv2.CAP_PROP_BRIGHTNESS, 140.0))
    src.release()

    src.open()
    src.release()
    assert _wrote(caps[1], cv2.CAP_PROP_BRIGHTNESS, 140.0)


def test_autofocus_is_written_before_focus_when_set_live():
    """Same ordering open() has always used: a focus value written while
    autofocus is on is immediately hunted away from."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(focus=30.0, autofocus=False)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_FOCUS, 30.0))
        props = [p for p, _, _ in cap.sets]
        assert props.index(cv2.CAP_PROP_AUTOFOCUS) < props.index(cv2.CAP_PROP_FOCUS)
    finally:
        src.release()


def test_a_rejected_control_write_does_not_kill_the_capture_thread():
    """_loop has no handler above it. A backend that raises on set() would
    otherwise end the thread mid-loop with `failure` unset and `_running`
    still true — the feed freezes and nothing says why. Losing one control
    is not worth losing the stream."""

    class _RefusingCap(_RecordingCap):
        def set(self, prop, value):
            if prop == cv2.CAP_PROP_BRIGHTNESS and value == 999.0:
                raise RuntimeError("backend refused brightness")
            return super().set(prop, value)

    cap = _RefusingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=999.0)
        assert _wait_for(lambda: src.control_error is not None)
        assert "brightness" in src.control_error
        assert src.failure is None

        # Still delivering frames, and still accepting later writes.
        src.set_controls(exposure=-6.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_EXPOSURE, -6.0))
        assert src._thread.is_alive()
    finally:
        src.release()
