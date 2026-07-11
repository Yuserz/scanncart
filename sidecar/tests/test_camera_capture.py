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
