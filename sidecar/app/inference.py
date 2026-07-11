from typing import Protocol
import numpy as np
from app.schemas import Detection


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalize_detections(xyxy, confs, clss, ids, names, width, height) -> list[Detection]:
    out: list[Detection] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        box = (
            _clamp01(x1 / width),
            _clamp01(y1 / height),
            _clamp01(x2 / width),
            _clamp01(y2 / height),
        )
        track_id = None if ids is None or ids[i] is None else int(ids[i])
        out.append(
            Detection(
                track_id=track_id,
                cls=names[int(clss[i])],
                conf=float(confs[i]),
                box=box,
            )
        )
    return out


class Detector(Protocol):
    names: dict

    def infer(self, frame: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    def __init__(self, model_path, device, conf, model_factory=None):
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        self._model = model_factory(model_path)
        self._device = device
        self._conf = conf
        self.names = self._model.names

    def infer(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame, persist=True, conf=self._conf, device=self._device, verbose=False
        )
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else boxes.xyxy
        confs = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else boxes.conf
        clss = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else boxes.cls
        ids = None
        if boxes.id is not None:
            ids = boxes.id.tolist() if hasattr(boxes.id, "tolist") else boxes.id
        h, w = frame.shape[0], frame.shape[1]
        return normalize_detections(xyxy, confs, clss, ids, r.names, w, h)
