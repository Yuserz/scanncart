import numpy as np
from app.pipeline import Pipeline, encode_preview_jpeg, mirrored_detections
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


class _DetectorThatLearnsItsClasses:
    """The realistic native shape: `names` is empty until the first inference fills it.

    Not a contrivance - `YoloDetector` starts that way deliberately, because reading the names off
    the model at construction builds a second ONNX session (on the default device) that the first
    `infer()` then discards.
    """

    def __init__(self, names: dict | None = None):
        self.names: dict = {}
        self._after_infer = names or {}

    def infer(self, frame):
        self.names = self._after_infer
        return [Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]


def test_the_class_list_is_reported_once_the_detector_knows_it():
    """The runtime half of the roster guard: the app has to be able to say what the model it is
    actually running predicts. It cannot be asked at start - the names do not exist yet - so the
    report happens on the first inference that has them, and rides a callback into the WS status.
    """
    seen: list[list[str]] = []
    detector = _DetectorThatLearnsItsClasses({0: "milo", 1: "milo close", 2: "milo mid"})
    pipe = Pipeline(
        _StubSource(), detector, Settings(), on_message=lambda _m: None,
        on_class_list=seen.append,
    )

    pipe.process_once()
    assert seen == [["milo", "milo close", "milo mid"]]

    # Once, not per frame: the vocabulary cannot change mid-run, so a second report would only be
    # the same sentence arriving 30 times a second.
    pipe.process_once()
    assert len(seen) == 1


def test_nothing_is_reported_while_the_detector_does_not_know_its_classes():
    """The cry-wolf case, and the reason the report waits rather than being asked for at start.
    Reporting an empty name list would say "this model predicts none of the 8 roster classes" on
    every healthy capture - a warning that fires always is a warning nobody reads.
    """
    seen: list[list[str]] = []
    pipe = Pipeline(
        _StubSource(), _DetectorThatLearnsItsClasses(), Settings(),
        on_message=lambda _m: None, on_class_list=seen.append,
    )

    for _ in range(3):
        pipe.process_once()

    assert seen == []


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


def test_stats_report_zero_when_camera_stalls():
    """A stalled camera has measured_fps == 0.0 (attribute present, value is zero).
    We must report 0.0, NOT the requested rate — that is exactly the fabricated
    number this task exists to remove."""
    class _StallSource(_StubSource):
        measured_fps = 0.0

    sent = []
    pipe = Pipeline(_StallSource(), _StubDetector(), Settings(capture_fps=60),
                    on_message=sent.append)
    msg = pipe.process_once()

    assert msg["stats"]["capture_fps"] == 0.0


# --- the preview is mirrored; inference is not ---------------------------


def _left_lit_source(w=64, h=48):
    """A frame whose only bright area is on the left, so a mirror shows up in the bytes.

    Uniform frames are the trap for this: one encodes byte-identically mirrored or not, so a
    test built on `_frame()` passes whether or not anything was reflected. This one cannot.
    """

    class _Source:
        width, height, fps = w, h, 30.0

        def latest(self):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, : w // 4] = 255
            return (7, frame)

        def read(self):
            return self.latest()[1]

    return _Source()


def _mirrored(frame):
    """Left-to-right reflection, written independently of the implementation."""
    return np.ascontiguousarray(frame[:, ::-1])


#: `_StubDetector` returns a box on the left: the overlay has to move to the right of a
#: mirrored preview for it to still describe the item it was detected on.
_STUB_BOX_MIRRORED = (0.7, 0.2, 0.9, 0.4)


def _rounded(box):
    return tuple(round(v, 6) for v in box)


def test_the_preview_is_mirrored_and_the_boxes_reflect_with_it():
    """The operator asked for a mirror image; the overlay has to come with it.

    Both halves are asserted against independent expressions - a numpy reverse of the x axis for
    the image, and the reflected coordinates for the box - so this fails if either the flip is
    dropped or it is applied to only one of the two. A mirrored image with unmirrored boxes is
    worse than either alone: the boxes sit on the wrong items.
    """
    sent = []
    source = _left_lit_source()
    pipe = Pipeline(source, _StubDetector(), Settings(), on_message=sent.append)

    msg = pipe.process_once()
    _, true_frame = source.latest()
    height = Settings().preview_height

    assert msg["jpeg"] == encode_preview_jpeg(_mirrored(true_frame), height)
    # And not the true frame either, which is what makes the line above mean something.
    assert msg["jpeg"] != encode_preview_jpeg(true_frame, height)
    assert _rounded(msg["detections"][0]["box"]) == _STUB_BOX_MIRRORED


def test_the_gap_frames_are_mirrored_by_the_same_rule():
    """`emit_preview` renders most of the frames on screen, so it has to agree with the other path.

    Reflecting in only one of the two would flicker the overlay between mirrored and true at the
    inference rate - which is more distracting than not mirroring at all, and is the reason the
    mirror is applied at both message builds rather than once when the frame is read.
    """
    sent = []
    source = _left_lit_source()
    pipe = Pipeline(source, _StubDetector(), Settings(), on_message=sent.append)
    pipe.process_once()
    pipe._last_emit_ts = 0.0

    msg = pipe.emit_preview()
    _, true_frame = source.latest()

    assert msg is not None
    assert msg["jpeg"] == encode_preview_jpeg(
        _mirrored(true_frame), Settings().preview_height
    )
    assert _rounded(msg["detections"][0]["box"]) == _STUB_BOX_MIRRORED


def test_the_detector_is_given_the_true_frame_and_never_the_mirror():
    """The load-bearing half of the decision to mirror in the preview rather than the capture.

    These weights were trained on ordinary photographs, so a mirrored input asks the model to read
    reversed text and mirrored brand marks - exactly what identifies a sachet. The operator wants
    the mirror; the model must not pay for it. If the flip ever moves into `CameraCapture`, this
    is the test that fails.
    """
    seen = []

    class _RecordingDetector(_StubDetector):
        def infer(self, frame):
            seen.append(np.array(frame, copy=True))
            return super().infer(frame)

    source = _left_lit_source()
    pipe = Pipeline(source, _RecordingDetector(), Settings(), on_message=lambda _m: None)
    pipe.process_once()
    _, true_frame = source.latest()

    assert len(seen) == 1
    assert np.array_equal(seen[0], true_frame)
    assert not np.array_equal(seen[0], _mirrored(true_frame))
    # Stated as the thing that would actually hurt: the bright region is still on the left
    # in what the model saw, and on the right in what the operator saw.
    assert seen[0][:, 0].mean() > seen[0][:, -1].mean()


def test_mirrored_detections_swaps_the_edges_rather_than_negating_them():
    """Pure, so the one easy-to-get-wrong step is pinned without needing a frame at all.

    A reflection is `x' = 1 - x`, so the two edges trade places. Negating both is the tempting
    version and it is wrong twice: it leaves the box on the side it started on, and it reverses
    which edge is which, so the overlay would draw inside out.
    """
    (out,) = mirrored_detections([Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))])

    assert _rounded(out.box) == _STUB_BOX_MIRRORED
    # y is untouched and nothing else about the detection changes.
    assert (out.track_id, out.cls, out.conf) == (1, "banana", 0.9)

    # A centred box is its own mirror - which the negate-both version also passes, and the reason
    # the assertion above uses an off-centre one.
    centred = Detection(track_id=2, cls="banana", conf=0.9, box=(0.25, 0.1, 0.75, 0.9))
    (same,) = mirrored_detections([centred])
    assert _rounded(same.box) == (0.25, 0.1, 0.75, 0.9)

    # Reflecting twice is the identity: a box lands back where it started.
    (back,) = mirrored_detections([out])
    assert _rounded(back.box) == (0.1, 0.2, 0.3, 0.4)

    assert mirrored_detections([]) == []
