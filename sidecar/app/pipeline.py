import base64
import threading
import time
from typing import Callable
import cv2
import numpy as np
from app.schemas import Detection, Stats, FrameMessage


def encode_preview_jpeg(frame: np.ndarray, target_height: int) -> str:
    h, w = frame.shape[0], frame.shape[1]
    if h > target_height:
        scale = target_height / h
        frame = cv2.resize(frame, (int(w * scale), target_height))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Pipeline:
    def __init__(self, source, detector, settings, on_message: Callable[[dict], None]):
        self._source = source
        self._detector = detector
        self._settings = settings
        self._on_message = on_message
        self._thread = None
        self.is_running = False
        self._frame_counter = 0
        self._last_infer_ts = None
        self._infer_fps = 0.0

    def process_once(self) -> dict | None:
        got = self._source.latest()
        if got is None:
            return None
        seq, frame = got

        skip = self._settings.infer_frame_skip
        self._frame_counter += 1
        if skip > 0 and (self._frame_counter - 1) % (skip + 1) != 0:
            return None

        t0 = time.time()
        detections = self._detector.infer(frame)
        t1 = time.time()

        if self._last_infer_ts is not None:
            dt = t1 - self._last_infer_ts
            if dt > 0:
                self._infer_fps = 1.0 / dt
        self._last_infer_ts = t1

        jpeg = encode_preview_jpeg(frame, self._settings.preview_height)
        stats = Stats(
            infer_fps=round(self._infer_fps, 1),
            capture_fps=float(getattr(self._source, "fps", 0.0)),
            latency_ms=round((t1 - t0) * 1000.0, 1),
        )
        msg = FrameMessage(
            type="frame", ts=t1, seq=seq, jpeg=jpeg,
            detections=detections, stats=stats,
        ).model_dump()
        self._on_message(msg)
        return msg

    def _loop(self) -> None:
        while self.is_running:
            produced = self.process_once()
            if produced is None:
                time.sleep(0.005)

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
