import numpy as np
from app.camera import LatestFrameBuffer, FakeFrameSource


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
