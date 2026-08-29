import threading
import time

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


def test_measured_fps_is_zero_before_any_frame():
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self): return False, None
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    assert c.measured_fps == 0.0


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
