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
    def __init__(
        self,
        source,
        detector,
        settings,
        on_message: Callable[[dict], None],
        logging_store=None,
        session_id=None,
        clock: Callable[[], float] = time.time,
    ):
        self._source = source
        self._detector = detector
        self._settings = settings
        self._on_message = on_message
        self._logging_store = logging_store
        self._session_id = session_id
        self._clock = clock
        self._open: dict[int, float] = {}   # track_id -> last-seen timestamp
        # Debounce: tracks not yet confirmed. count is consecutive inferences
        # seen; cls/conf are kept from the first sighting so the eventual log
        # row reports the entry moment. Cleared on removal.
        self._pending: dict[int, dict] = {}
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
        # Class allowlist (hot-reloaded): drop classes not on the list before
        # tracking/logging/streaming, so overlay, item log, and DB all agree.
        allow = self._settings.class_allowlist
        if allow:
            allowed = set(allow)
            detections = [d for d in detections if d.cls in allowed]
        t1 = time.time()

        if self._last_infer_ts is not None:
            dt = t1 - self._last_infer_ts
            if dt > 0:
                self._infer_fps = 1.0 / dt
        self._last_infer_ts = t1

        self._log_detections(detections)

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

    def _log_detections(self, detections: list[Detection]) -> None:
        if self._logging_store is None or self._session_id is None:
            return
        now = self._clock()
        seen_now = set()
        for d in detections:
            if d.track_id is None:
                continue
            seen_now.add(d.track_id)
            if d.track_id not in self._open:
                pending = self._pending.get(d.track_id)
                if pending is None:
                    pending = {"count": 0, "cls": d.cls, "conf": d.conf, "ts": now}
                    self._pending[d.track_id] = pending
                pending["count"] += 1
                if pending["count"] < self._settings.track_confirm_hits:
                    continue
                # Confirmed: promote with the FIRST sighting as entered_at.
                del self._pending[d.track_id]
            self._logging_store.record_detection(
                self._session_id, d.track_id, d.cls, d.conf, now
            )
            self._open[d.track_id] = now
        for track_id, last_seen in list(self._open.items()):
            if now - last_seen > self._settings.track_expiry_s:
                self._logging_store.resolve_left(self._session_id, track_id, last_seen)
                del self._open[track_id]
        # Pending tracks that vanished before confirming never get logged:
        # resolve_left for a never-recorded track_id would corrupt the log.
        for track_id in list(self._pending):
            if track_id not in seen_now:
                del self._pending[track_id]

    def resolve_open_tracks(self) -> None:
        if self._logging_store is None or self._session_id is None:
            return
        for track_id, last_seen in list(self._open.items()):
            self._logging_store.resolve_left(self._session_id, track_id, last_seen)
        self._open.clear()
        self._pending.clear()

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
            self._thread.join()
            self._thread = None
