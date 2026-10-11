import time
import numpy as np
from app.camera import CameraCapture


class FakeCap:
    """Mimics cv2.VideoCapture: isOpened/read/set/release."""

    def __init__(self, index):
        self.index = index
        self._opened = True
        self._n = 0

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        return True

    def read(self):
        self._n += 1
        return True, np.full((4, 4, 3), self._n % 256, dtype=np.uint8)

    def release(self):
        self._opened = False


def test_capture_thread_populates_latest_frame():
    cam = CameraCapture(0, 640, 480, 30, cap_factory=FakeCap)
    cam.open()
    assert cam.is_open
    # Give the thread a moment to produce at least one frame.
    deadline = time.time() + 2.0
    got = None
    while time.time() < deadline:
        got = cam.latest()
        if got is not None:
            break
        time.sleep(0.01)
    cam.release()
    assert got is not None
    seq, frame = got
    assert seq >= 1
    assert frame.shape == (4, 4, 3)
    assert not cam.is_open


def test_capture_open_failure_sets_not_open():
    class DeadCap(FakeCap):
        def isOpened(self):
            return False

    cam = CameraCapture(0, 640, 480, 30, cap_factory=DeadCap)
    opened = cam.open()
    assert opened is False
    assert cam.is_open is False


def test_stalled_delivery_is_failure_even_when_reads_succeed():
    """The unplug shape MSMF actually produces: read() keeps succeeding with one
    stale frame at ~1 fps, so a failed-read deadline never fires and capture
    looks running while the preview is a frozen black image. The delivered-rate
    stall guard is what must catch it."""

    class ZombieCap(FakeCap):
        """Succeeds at ~1 fps with the same black frame, forever."""

        def read(self):
            time.sleep(1.0)
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    cam = CameraCapture(0, 640, 480, 30, cap_factory=ZombieCap)
    cam.open()
    assert cam.is_open
    deadline = time.time() + 15.0
    while time.time() < deadline and cam.failure is None:
        time.sleep(0.25)
    cam.release()
    assert cam.failure is not None
    assert "frozen" in cam.failure
    assert "unplugged" in cam.failure


def test_healthy_delivery_never_trips_the_stall_guard():
    """The guard judges delivery rate over a window; an ordinary camera at half
    its target (the fps latch from DEFENSE_READINESS) sits far above the stall
    threshold and must never be declared failed."""

    class SlowCap(FakeCap):
        """Delivers ~30 fps, i.e. half the requested 60."""

        def read(self):
            time.sleep(0.033)
            return True, np.full((4, 4, 3), 200, dtype=np.uint8)

    cam = CameraCapture(0, 640, 480, 60, cap_factory=SlowCap)
    cam.open()
    assert cam.is_open
    deadline = time.time() + 8.0
    while time.time() < deadline and cam.failure is None:
        time.sleep(0.25)
    cam.release()
    assert cam.failure is None
