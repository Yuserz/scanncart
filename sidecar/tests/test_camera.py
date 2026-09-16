import numpy as np
from app.camera import (
    FALLBACK_MODES,
    LatestFrameBuffer,
    FakeFrameSource,
    open_verified,
)


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


class _FakeOpenCVCap:
    """Mimics the slice of cv2.VideoCapture that open_verified() touches."""

    last_instance = None

    def __init__(self, delivers: bool):
        self.delivers = delivers
        self.released = False
        self.props: dict[int, float] = {}
        _FakeOpenCVCap.last_instance = self

    def isOpened(self):
        return not self.released

    def set(self, prop, value):
        self.props[prop] = float(value)
        return True

    def read(self):
        if not self.delivers:
            return False, None
        return True, _frame(7)

    def release(self):
        self.released = True


def test_open_verified_returns_first_delivering_mode(monkeypatch):
    monkeypatch.setattr(
        "app.camera._default_capture", lambda index: _FakeOpenCVCap(delivers=True)
    )
    cap, mode = open_verified(0, 640, 480, 30)
    assert mode == (640, 480, 30)  # requested mode works -> tried first
    cap.release()


def test_open_verified_falls_back_when_requested_mode_delivers_nothing(monkeypatch):
    # Every cap from _default_capture refuses to deliver -> no mode works.
    monkeypatch.setattr(
        "app.camera._default_capture", lambda index: _FakeOpenCVCap(delivers=False)
    )
    cap, mode = open_verified(0, 1920, 1080, 60)
    assert cap is None and mode is None


def test_open_verified_sets_mjpg_fourcc(monkeypatch):
    import cv2

    cap_box = {}

    def factory(_index):
        cap = _FakeOpenCVCap(delivers=True)
        cap_box["cap"] = cap
        return cap

    monkeypatch.setattr("app.camera._default_capture", factory)
    open_verified(0, 640, 480, 30)
    fourcc = int(cap_box["cap"].props[cv2.CAP_PROP_FOURCC])
    # 'MJPG' packed OpenCV-style: c1 | c2<<8 | c3<<16 | c4<<24
    assert fourcc == ord("M") | ord("J") << 8 | ord("P") << 16 | ord("G") << 24


def test_fallback_modes_are_distinct_and_end_small():
    assert len(set(FALLBACK_MODES)) == len(FALLBACK_MODES)
    assert FALLBACK_MODES[0] == (1920, 1080, 60)
    assert FALLBACK_MODES[-1] == (640, 480, 30)
