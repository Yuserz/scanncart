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


# --- preview decoupled from inference ------------------------------------


class _EmptySource:
    """Never has a frame ready."""
    width, height, fps = 128, 96, 30.0

    def latest(self):
        return None


def _preview_pipe(settings=None):
    sent = []
    det = _StubDetector()
    pipe = Pipeline(_StubSource(), det, settings or Settings(), on_message=sent.append)
    return pipe, sent, det


def test_emit_preview_sends_a_frame_without_running_inference():
    """The whole point: fill the gap between inferences so the image stays
    smooth. Preview used to be emitted only after inference, delivering 9 fps
    from a 60 fps camera with gaps ranging 13-431 ms."""
    pipe, sent, det = _preview_pipe()

    msg = pipe.emit_preview()

    assert msg is not None and msg["type"] == "frame"
    assert det.calls == 0
    assert len(sent) == 1


def test_emit_preview_reuses_the_most_recent_detections():
    pipe, sent, _ = _preview_pipe()
    pipe.process_once()
    # process_once just emitted, so wind the rate-limit clock back rather than
    # sleeping for the interval.
    pipe._last_emit_ts = 0.0

    msg = pipe.emit_preview()

    # Boxes are one inference old — the trade for a smooth image.
    assert msg is not None
    assert [d["track_id"] for d in msg["detections"]] == [1]


def test_emit_preview_is_rate_limited():
    pipe, _, _ = _preview_pipe(Settings(preview_max_fps=30))

    assert pipe.emit_preview() is not None
    # A second call in the same instant is not due yet.
    assert pipe.emit_preview() is None


def test_preview_can_be_disabled_with_zero():
    """Restores emit-only-on-inference, for a machine where the extra JPEG
    encode competes with inference."""
    pipe, sent, _ = _preview_pipe(Settings(preview_max_fps=0))

    assert pipe.emit_preview() is None
    assert sent == []


def test_process_once_still_emits_and_resets_the_preview_clock():
    pipe, sent, det = _preview_pipe()

    assert pipe.process_once() is not None
    assert det.calls == 1
    # An inference just emitted, so a preview is not immediately due.
    assert pipe.emit_preview() is None
    assert len(sent) == 1


def test_emit_preview_returns_none_without_a_frame():
    sent = []
    pipe = Pipeline(_EmptySource(), _StubDetector(), Settings(), on_message=sent.append)

    assert pipe.emit_preview() is None
    assert sent == []


def test_stats_report_the_measured_capture_rate():
    """Not the requested one: the UI showed 60 fps while 12 arrived."""
    class _Source(_StubSource):
        measured_fps = 12.5

    sent = []
    pipe = Pipeline(_Source(), _StubDetector(), Settings(capture_fps=60),
                    on_message=sent.append)
    msg = pipe.process_once()

    assert msg["stats"]["capture_fps"] == 12.5


def test_stats_fall_back_to_the_configured_rate_for_a_source_that_cannot_measure():
    # FakeFrameSource and other test doubles have no measured_fps.
    sent = []
    pipe = Pipeline(_StubSource(), _StubDetector(), Settings(), on_message=sent.append)
    msg = pipe.process_once()

    assert msg["stats"]["capture_fps"] == _StubSource.fps
