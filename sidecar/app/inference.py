from typing import Protocol
import numpy as np
from app.roboflow import RoboflowError, find_image_size, find_predictions
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


# Ultralytics defaults `.track()` to BoT-SORT, whose `gmc_method:
# sparseOptFlow` runs global motion compensation — optical flow over the *full*
# frame — to cancel camera movement. This camera is bolted to a counter, so
# that buys nothing and costs a great deal: measured on a GTX 1050 Ti at
# 1280x720, predict alone is ~42 ms, BoT-SORT ~83 ms, ByteTrack ~39 ms. GMC
# was doubling the cost of every frame, and it scales with capture resolution
# (+47 ms at 720p, +5 ms at 480p) rather than with `imgsz`.
#
# The two shipped configs are otherwise identical — same track_high_thresh,
# track_low_thresh, new_track_thresh, track_buffer, match_thresh, fuse_score —
# and botsort.yaml has `with_reid: False`, so GMC is the *only* behavioural
# difference. Dropping it is free here.
DEFAULT_TRACKER = "bytetrack.yaml"


class YoloDetector:
    def __init__(self, model_path, device, conf, imgsz=640, model_factory=None,
                 tracker=DEFAULT_TRACKER, resize_mode="letterbox"):
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        self._model = model_factory(model_path)
        self._device = device
        self._conf = conf
        self._imgsz = imgsz
        self._tracker = tracker
        self._is_onnx = str(model_path).lower().endswith(".onnx")
        self._stretch = resize_mode == "stretch"
        self.names = self._model.names

    def infer(self, frame: np.ndarray) -> list[Detection]:
        kwargs = dict(
            persist=True, conf=self._conf, imgsz=self._imgsz,
            verbose=False, tracker=self._tracker,
        )
        # An ONNX session fixes its execution provider at load time, so `device`
        # cannot move it and only costs ultralytics a per-call device
        # resolution — measured ~10 ms/frame on the custom grocery model.
        if not self._is_onnx:
            kwargs["device"] = self._device
        if self._stretch:
            # Feed an already-square frame so ultralytics' letterbox adds no
            # padding — reproducing the "Stretch to" preprocessing the model was
            # trained with. Letterboxing a 1280x720 frame fills only 56% of the
            # 640x640 canvas, presenting every object well below its training
            # scale.
            #
            # Boxes stay correct without any un-warping: normalize_detections
            # divides by the dimensions actually fed to the model, and a
            # per-axis scale cancels out of a normalized coordinate.
            import cv2

            frame = cv2.resize(frame, (self._imgsz, self._imgsz))
        results = self._model.track(frame, **kwargs)
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


class RoboflowRemoteDetector:
    """Runs detection through a Roboflow Workflow — cloud or self-hosted.

    Satisfies the same `Detector` protocol as `YoloDetector`, so `Pipeline`
    cannot tell the difference. See docs/DETECTOR_BACKENDS.md.

    Three things this has to do that the native detector gets for free:

    * **Downscale before transmit.** YOLO11 infers at 640 regardless, so
      sending a full 1080p frame is wasted bandwidth. Quality 80 matches
      `pipeline.encode_preview_jpeg`.
    * **Filter by confidence client-side.** The workflow declares no
      parameters (verified in Phase 0), so `conf_threshold` cannot be pushed
      to the server and must be applied here.
    * **Assign track ids.** The workflow has no tracking block, so responses
      carry no `tracker_id` and the injected tracker is the only source of
      stable ids. A response that *does* carry one wins, so adding a tracking
      block upstream later needs no code change here.
    """

    def __init__(
        self,
        client,
        infer_size: int = 640,
        conf: float = 0.5,
        tracker=None,
        jpeg_quality: int = 80,
    ):
        self._client = client
        self._infer_size = infer_size
        self._conf = conf
        self._tracker = tracker
        self._jpeg_quality = jpeg_quality
        # No model manifest to read over HTTP; filled in as classes are seen.
        self.names: dict = {}

    def _encode(self, frame: np.ndarray) -> tuple[str, int, int]:
        import base64

        import cv2

        h, w = frame.shape[0], frame.shape[1]
        longest = max(h, w)
        if longest > self._infer_size:
            scale = self._infer_size / longest
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            h, w = frame.shape[0], frame.shape[1]
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            raise RoboflowError("Failed to JPEG-encode the frame for transmit.")
        return base64.b64encode(buf).decode("ascii"), w, h

    def infer(self, frame: np.ndarray) -> list[Detection]:
        image_b64, sent_w, sent_h = self._encode(frame)
        result = self._client.run(image_b64)
        predictions = find_predictions(result)
        if not predictions:
            # Still age out tracks, or an item that leaves frame keeps its slot
            # until something else happens to be detected.
            return self._tracker.assign([]) if self._tracker is not None else []

        # Coordinates are relative to the frame the server saw. It echoes those
        # dimensions back; trust that over our own when present.
        ref = find_image_size(result) or (sent_w, sent_h)
        ref_w, ref_h = ref

        out: list[Detection] = []
        for p in predictions:
            detection = self._to_detection(p, ref_w, ref_h)
            if detection is not None:
                out.append(detection)

        if self._tracker is None:
            return out

        # Track only what the server did not. Assigning over the whole list
        # would overwrite a server-supplied `tracker_id`, so a workflow with a
        # tracking block that reports ids for confirmed tracks only (ByteTrack
        # does exactly this) would see its stable ids replaced on any frame
        # containing one new detection — the same item flipping id between
        # frames, and duplicate rows in the session log.
        untracked = [i for i, d in enumerate(out) if d.track_id is None]
        for d in out:
            if d.track_id is not None:
                self._tracker.reserve_id(d.track_id)
        # Called even when `untracked` is empty: that is what ages out local
        # tracks for an item that has left frame.
        for i, assigned in zip(untracked, self._tracker.assign([out[i] for i in untracked])):
            out[i] = assigned
        return out

    def _to_detection(self, p: dict, ref_w: int, ref_h: int) -> Detection | None:
        try:
            conf = float(p["confidence"])
            cx, cy = float(p["x"]), float(p["y"])
            bw, bh = float(p["width"]), float(p["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if conf < self._conf:
            return None
        # Phase 0 verified this workflow emits `class`; `class_name` is the
        # spelling used elsewhere in Roboflow's stack.
        label = p.get("class") or p.get("class_name")
        if not label:
            return None
        class_id = p.get("class_id")
        if isinstance(class_id, int):
            self.names.setdefault(class_id, label)

        box = (
            _clamp01((cx - bw / 2) / ref_w),
            _clamp01((cy - bh / 2) / ref_h),
            _clamp01((cx + bw / 2) / ref_w),
            _clamp01((cy + bh / 2) / ref_h),
        )
        tracker_id = p.get("tracker_id")
        return Detection(
            track_id=int(tracker_id) if isinstance(tracker_id, int) else None,
            cls=str(label),
            conf=conf,
            box=box,
        )

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()
