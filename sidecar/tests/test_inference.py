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
