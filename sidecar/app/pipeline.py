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
        on_error: Callable[[Exception], None] | None = None,
    ):
        self._source = source
        self._detector = detector
        self._settings = settings
        self._on_message = on_message
        self._logging_store = logging_store
        self._session_id = session_id
        self._clock = clock
        self._on_error = on_error
        self._open: dict[int, float] = {}   # track_id -> last-seen timestamp
        self._thread = None
        self.is_running = False
        self._frame_counter = 0
        self._last_infer_ts = None
        self._infer_fps = 0.0
        # Preview is decoupled from inference. Inference blocks its thread for
        # as long as a frame takes (~90-210 ms here), so emitting only after it
        # delivered the preview at the *inference* rate: measured 9 fps from a
        # 60 fps camera, with gaps from 13 ms to 431 ms. That jitter is what
        # reads as stutter. A second thread now emits frames at a steady rate
        # in between, reusing the most recent detections.
        self._preview_thread = None
        self._state_lock = threading.Lock()
        self._latest_detections: list[Detection] = []
        self._latest_stats = None
        self._last_emit_ts = 0.0

    def _capture_fps(self) -> float:
        """What the camera actually delivers, falling back to its nominal rate
        for sources that cannot measure (test doubles)."""
        measured = getattr(self._source, "measured_fps", None)
        if measured:
            return float(measured)
        return float(getattr(self._source, "fps", 0.0))

    def process_once(self) -> dict | None:
        # A source that has given up is reported, not waited on: otherwise
        # capture sits in "running" with a frozen image and no explanation.
        failure = getattr(self._source, "failure", None)
        if failure:
            raise RuntimeError(failure)

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

        self._log_detections(detections)

        jpeg = encode_preview_jpeg(frame, self._settings.preview_height)
        stats = Stats(
            infer_fps=round(self._infer_fps, 1),
            capture_fps=self._capture_fps(),
            latency_ms=round((t1 - t0) * 1000.0, 1),
        )
        with self._state_lock:
            self._latest_detections = detections
            self._latest_stats = stats
            self._last_emit_ts = t1

        msg = FrameMessage(
            type="frame", ts=t1, seq=seq, jpeg=jpeg,
            detections=detections, stats=stats,
        ).model_dump()
        self._on_message(msg)
        return msg

    def emit_preview(self) -> dict | None:
        """Send one preview frame using the most recent detections.

        The boxes are up to one inference old, which is the trade: a smooth
        image with slightly trailing boxes beats a sharp-but-stuttering one,
        and checkout items sit still anyway. Returns None when there is no
        frame yet, or when an emit is not due.
        """
        max_fps = self._settings.preview_max_fps
        if max_fps <= 0:
            return None
        now = time.time()
        with self._state_lock:
            if now - self._last_emit_ts < 1.0 / max_fps:
                return None
            detections = list(self._latest_detections)
            stats = self._latest_stats
            self._last_emit_ts = now

        got = self._source.latest()
        if got is None:
            return None
        seq, frame = got
        jpeg = encode_preview_jpeg(frame, self._settings.preview_height)
        msg = FrameMessage(
            type="frame", ts=now, seq=seq, jpeg=jpeg,
            detections=detections,
            stats=stats
            or Stats(infer_fps=0.0, capture_fps=self._capture_fps(), latency_ms=0.0),
        ).model_dump()
        self._on_message(msg)
        return msg

    def _log_detections(self, detections: list[Detection]) -> None:
        if self._logging_store is None or self._session_id is None:
            return
        now = self._clock()
        for d in detections:
            if d.track_id is None:
                continue
            self._logging_store.record_detection(
                self._session_id, d.track_id, d.cls, d.conf, now
            )
            self._open[d.track_id] = now
        for track_id, last_seen in list(self._open.items()):
            if now - last_seen > self._settings.track_expiry_s:
                self._logging_store.resolve_left(self._session_id, track_id, last_seen)
                del self._open[track_id]

    def resolve_open_tracks(self) -> None:
        if self._logging_store is None or self._session_id is None:
            return
        for track_id, last_seen in list(self._open.items()):
            self._logging_store.resolve_left(self._session_id, track_id, last_seen)
        self._open.clear()

    def _loop(self) -> None:
        while self.is_running:
            try:
                produced = self.process_once()
            except Exception as exc:  # noqa: BLE001 - the thread must not die silently
                # With a native detector `infer` essentially never raised, so an
                # uncaught exception here used to be unreachable. The remote
                # backends make it routine (server stopped, network drop, 5xx),
                # and an unhandled one killed this daemon thread while
                # `is_running` stayed True — capture appeared to run forever with
                # a frozen preview and no error anywhere. Report and shut down
                # instead.
                self.is_running = False
                if self._on_error is not None:
                    # Runs ON this thread, so the handler must never join it.
                    try:
                        self._on_error(exc)
                    except Exception:  # noqa: BLE001 - nothing useful left to do
                        pass
                return
            if produced is None:
                time.sleep(0.005)

    def _preview_loop(self) -> None:
        """Fills the gaps between inferences so the image stays smooth.

        Deliberately swallows its errors: this thread is cosmetic, and the
        inference loop is the one that owns reporting failure and shutting
        capture down.
        """
        while self.is_running:
            try:
                self.emit_preview()
            except Exception:  # noqa: BLE001 - cosmetic thread, never kills capture
                pass
            time.sleep(0.005)

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def signal_stop(self) -> None:
        """Ask the loop to finish without waiting for it.

        Separate from `stop()` so a caller on the event loop can end inference
        immediately and then join from a worker thread — the join can take as
        long as one remote round trip plus its retries.
        """
        self.is_running = False

    def stop(self) -> None:
        self.is_running = False
        if self._preview_thread is not None:
            self._preview_thread.join()
            self._preview_thread = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None
