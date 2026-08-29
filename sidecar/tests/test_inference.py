import numpy as np
from app.inference import normalize_detections, YoloDetector


def test_normalize_scales_and_labels():
    dets = normalize_detections(
        xyxy=[[64, 48, 128, 96]],
        confs=[0.9],
        clss=[0],
        ids=[7],
        names={0: "banana"},
        width=128,
        height=96,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "banana"
    assert d.track_id == 7
    assert d.conf == 0.9
    assert d.box == (0.5, 0.5, 1.0, 1.0)


def test_normalize_handles_missing_ids():
    dets = normalize_detections(
        xyxy=[[0, 0, 64, 48]], confs=[0.3], clss=[1], ids=None,
        names={1: "apple"}, width=64, height=48,
    )
    assert dets[0].track_id is None


def test_normalize_clamps_out_of_bounds():
    dets = normalize_detections(
        xyxy=[[-10, -10, 200, 200]], confs=[0.5], clss=[0], ids=[1],
        names={0: "x"}, width=100, height=100,
    )
    assert dets[0].box == (0.0, 0.0, 1.0, 1.0)


class _FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]])
        self.conf = np.array([0.8])
        self.cls = np.array([0.0])
        self.id = np.array([3.0])

    def __len__(self):
        return len(self.xyxy)


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()
        self.names = {0: "bottle"}


class _FakeModel:
    def __init__(self, path):
        self.path = path
        self.names = {0: "bottle"}
        self.track_kwargs = None

    def track(self, frame, **kwargs):
        self.track_kwargs = kwargs
        return [_FakeResult()]


def test_yolo_detector_infer_converts_results():
    det = YoloDetector("yolo11n.pt", device="cpu", conf=0.5, model_factory=_FakeModel)
    frame = np.zeros((40, 30, 3), dtype=np.uint8)
    out = det.infer(frame)
    assert len(out) == 1
    assert out[0].cls == "bottle"
    assert out[0].track_id == 3
    assert out[0].box[0] == 10.0 / 30.0


def test_yolo_detector_forwards_imgsz_to_track():
    det = YoloDetector(
        "yolo11n.pt", device="cpu", conf=0.5, imgsz=960, model_factory=_FakeModel
    )
    det.infer(np.zeros((40, 30, 3), dtype=np.uint8))
    assert det._model.track_kwargs["imgsz"] == 960


def test_yolo_detector_uses_bytetrack_not_the_botsort_default():
    """Ultralytics defaults to BoT-SORT, whose GMC runs optical flow over the
    full frame to cancel *camera* motion. This camera is fixed to a counter, so
    it bought nothing and doubled per-frame cost (~83 ms vs ~39 ms measured on
    a GTX 1050 Ti at 720p). The configs are otherwise identical.
    """
    calls = {}

    class FakeModel:
        names = {0: "apple"}

        def track(self, frame, **kw):
            calls.update(kw)
            return []

    from app.inference import DEFAULT_TRACKER, YoloDetector

    d = YoloDetector("m.pt", device="cpu", conf=0.5, model_factory=lambda p: FakeModel())
    d.infer(np.zeros((8, 8, 3), dtype=np.uint8))

    assert DEFAULT_TRACKER == "bytetrack.yaml"
    assert calls["tracker"] == "bytetrack.yaml"
    assert calls["persist"] is True


def test_yolo_detector_tracker_is_overridable():
    class FakeModel:
        names: dict = {}

        def __init__(self):
            self.kw = {}

        def track(self, frame, **kw):
            self.kw = kw
            return []

    from app.inference import YoloDetector

    model = FakeModel()
    d = YoloDetector("m.pt", device="cpu", conf=0.5, model_factory=lambda p: model,
                     tracker="botsort.yaml")
    d.infer(np.zeros((8, 8, 3), dtype=np.uint8))
    assert model.kw["tracker"] == "botsort.yaml"


def test_onnx_model_is_given_its_device():
    """`device` is what selects the CUDA execution provider for an ONNX model.

    It was briefly omitted as a ~10 ms saving, back when onnxruntime here could
    only ever run on CPU. With a CUDA-12 onnxruntime-gpu and torch's bundled
    DLLs the GPU works (19.7 ms vs 66.3 ms), and omitting device would pin the
    model to CPU and throw that away.
    """
    seen = {}

    class FakeModel:
        names = {0: "a"}

        def track(self, frame, **kw):
            seen.update(kw)
            return []

    from app.inference import YoloDetector

    d = YoloDetector("models/x.onnx", device="cuda", conf=0.5, model_factory=lambda p: FakeModel())
    d.infer(np.zeros((8, 8, 3), dtype=np.uint8))
    assert seen["device"] == "cuda"


def test_pt_model_still_gets_its_device():
    seen = {}

    class FakeModel:
        names = {0: "a"}

        def track(self, frame, **kw):
            seen.update(kw)
            return []

    from app.inference import YoloDetector

    d = YoloDetector("yolo11n.pt", device="cuda", conf=0.5, model_factory=lambda p: FakeModel())
    d.infer(np.zeros((8, 8, 3), dtype=np.uint8))
    assert seen["device"] == "cuda"


def test_stretch_mode_feeds_a_square_frame():
    """Reproduces Roboflow's "Stretch to" training preprocessing. Letterboxing
    a 1280x720 frame fills only 56% of the 640x640 canvas, presenting every
    object well under its training scale."""
    seen = {}

    class FakeModel:
        names = {0: "a"}

        def track(self, frame, **kw):
            seen["shape"] = frame.shape
            return []

    from app.inference import YoloDetector

    d = YoloDetector("models/x.onnx", device="cpu", conf=0.5, imgsz=640,
                     model_factory=lambda p: FakeModel(), resize_mode="stretch")
    d.infer(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert seen["shape"][:2] == (640, 640)


def test_letterbox_mode_leaves_the_frame_alone():
    seen = {}

    class FakeModel:
        names = {0: "a"}

        def track(self, frame, **kw):
            seen["shape"] = frame.shape
            return []

    from app.inference import YoloDetector

    d = YoloDetector("yolo11n.pt", device="cpu", conf=0.5, imgsz=640,
                     model_factory=lambda p: FakeModel(), resize_mode="letterbox")
    d.infer(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert seen["shape"][:2] == (720, 1280)


def test_stretching_preserves_normalized_boxes():
    """The reason no un-warping is needed: a per-axis scale cancels out of a
    normalized coordinate, so a box normalized against the square frame equals
    the same box normalized against the original."""
    from app.inference import normalize_detections

    # A box covering the right half, full height, in each coordinate space.
    orig = normalize_detections([[640.0, 0.0, 1280.0, 720.0]], [0.9], [0], None,
                                {0: "a"}, 1280, 720)
    squished = normalize_detections([[320.0, 0.0, 640.0, 640.0]], [0.9], [0], None,
                                    {0: "a"}, 640, 640)
    assert orig[0].box == squished[0].box
