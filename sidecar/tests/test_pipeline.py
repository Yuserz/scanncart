import numpy as np
from app.pipeline import Pipeline, encode_preview_jpeg
from app.camera import CameraCapture
from app.settings import Settings
from app.schemas import Detection


def _frame(h=96, w=128, val=100):
    return np.full((h, w, 3), val, dtype=np.uint8)


class _StubDetector:
    names = {0: "banana"}

    def __init__(self):
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return [Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]


class _StubSource:
    """Always returns the same latest frame."""
    width = 128
    height = 96
    fps = 30.0

    def latest(self):
        return (5, _frame())

    def read(self):
        return _frame()


def test_encode_preview_jpeg_returns_base64():
    s = encode_preview_jpeg(_frame(720, 1280), target_height=360)
    assert isinstance(s, str)
    assert len(s) > 0


def test_process_once_builds_frame_message():
    msgs = []
    pipe = Pipeline(_StubSource(), _StubDetector(), Settings(), on_message=msgs.append)
    out = pipe.process_once()
    assert out is not None
    assert out["type"] == "frame"
    assert out["seq"] == 5
    assert out["detections"][0]["cls"] == "banana"
    assert out["jpeg"]
    assert "infer_fps" in out["stats"]
    assert msgs and msgs[0] is out


def test_process_once_returns_none_without_frame():
    class Empty:
        width = 1
        height = 1
        fps = 1.0

        def latest(self):
            return None

    pipe = Pipeline(Empty(), _StubDetector(), Settings(), on_message=lambda m: None)
    assert pipe.process_once() is None


def test_frame_skip_skips_inference():
    det = _StubDetector()
    settings = Settings(infer_frame_skip=1)  # process 1, skip 1, ...
    pipe = Pipeline(_StubSource(), det, settings, on_message=lambda m: None)
    pipe.process_once()  # processed (infer called)
    pipe.process_once()  # skipped (no infer)
    assert det.calls == 1
