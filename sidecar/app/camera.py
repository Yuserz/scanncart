import threading
import time
from collections import deque
from typing import Protocol
import cv2
import numpy as np


class LatestFrameBuffer:
    """Thread-safe size-1 buffer where the newest frame always wins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: tuple[int, np.ndarray] | None = None

    def put(self, seq: int, frame: np.ndarray) -> None:
        with self._lock:
            self._item = (seq, frame)

    def get(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            return self._item


def _default_capture(index):
    # Pin the Media Foundation backend on Windows: it delivers the StreamCam's
    # full 60 fps at 1080p, whereas OpenCV's DirectShow path caps around 15 fps
    # for the same mode. Fall back to OpenCV's auto backend if MSMF can't open
    # the device (e.g. a non-Windows host or a camera with no MSMF driver).
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


class FrameSource(Protocol):
    width: int
    height: int
    fps: float

    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def release(self) -> None: ...


class FakeFrameSource:
    """Test double that yields the provided frames in order, then None."""

    def __init__(self, frames: list[np.ndarray], fps: float = 30.0) -> None:
        self._frames = frames
        self._i = 0
        h, w = (frames[0].shape[0], frames[0].shape[1]) if frames else (0, 0)
        self.width = w
        self.height = h
        self.fps = fps

    def open(self) -> None:
        self._i = 0

    def read(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def release(self) -> None:
        pass


class CameraCapture:
    """Owns an OpenCV device and runs a background capture thread."""

    def __init__(
        self, index, width, height, fps, cap_factory=_default_capture,
        brightness: float | None = None, exposure: float | None = None,
        autofocus: bool | None = None, focus: float | None = None,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = float(fps)
        self._cap_factory = cap_factory
        # None means "leave the camera alone" — see Settings.camera_brightness
        # et al. The StreamCam's automatic focus/exposure track faces, which a
        # checkout counter never has, so locked manual values suit this app.
        self._brightness = brightness
        self._exposure = exposure
        self._autofocus = autofocus
        self._focus = focus
        # Control changes queued by another thread, drained by _loop between
        # reads. cv2.VideoCapture is not thread-safe, so set() must never be
        # called from the FastAPI request thread while read() is in flight.
        self._controls_lock = threading.Lock()
        self._pending_controls: dict[str, float | bool | None] = {}
        # Set when a live control write is refused by the device. Not a
        # capture failure — the stream keeps running — so it stays separate
        # from `failure`.
        self.control_error: str | None = None
        self._cap = None
        self._buffer = LatestFrameBuffer()
        self._thread = None
        self._running = False
        self._seq = 0
        self.is_open = False
        # A device can be invalidated while open — unplugged, taken by another
        # process, or suspended by USB power management. read() then fails
        # instantly and forever. Retrying flat out burned a core and wrote
        # ~23,000 OpenCV warnings to stderr in one session while the app sat
        # there looking like it was running.
        self.failure: str | None = None
        self._consecutive_failures = 0
        self._failing_since: float | None = None
        # Timestamps of recent successful reads, for the measured rate. The
        # requested fps is a request; this is what arrived.
        self._read_times: deque[float] = deque(maxlen=120)

    def _current_controls(self) -> dict:
        return {
            "autofocus": self._autofocus,
            "focus": self._focus,
            "brightness": self._brightness,
            "exposure": self._exposure,
        }

    @staticmethod
    def _write_controls(cap, controls: dict) -> None:
        """Write device controls in dependency order.

        Autofocus goes first: a focus value written while autofocus is on is
        immediately hunted away from. Keys absent or None mean "leave the
        camera alone" — see Settings.camera_brightness et al.
        """
        if controls.get("autofocus") is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if controls["autofocus"] else 0)
        if controls.get("focus") is not None:
            cap.set(cv2.CAP_PROP_FOCUS, controls["focus"])
        if controls.get("brightness") is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, controls["brightness"])
        if controls.get("exposure") is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, controls["exposure"])

    def set_controls(self, **changes) -> None:
        """Queue control changes for the capture thread.

        Accepts brightness/exposure/autofocus/focus. Updating a dict rather
        than appending to a queue coalesces a fast slider drag to its newest
        value, so the capture thread never works through a backlog.
        """
        with self._controls_lock:
            self._pending_controls.update(changes)

    def _drain_controls(self) -> None:
        with self._controls_lock:
            if not self._pending_controls:
                return
            changes = self._pending_controls
            self._pending_controls = {}
        # Merge onto the instance fields so a later reopen replays them.
        for name, value in changes.items():
            setattr(self, f"_{name}", value)
        # Deliberately outside the lock: cap.set() can block on some
        # backends, and holding the lock across it would stall the caller.
        #
        # Swallowed rather than raised: this runs on the capture thread, which
        # has no handler above it. A backend that rejects one value would
        # otherwise kill the thread mid-loop, leaving `failure` unset and
        # `_running` true — the feed freezes with nothing to explain it. A
        # control that will not take is not worth the stream.
        try:
            self._write_controls(self._cap, changes)
        except Exception as exc:  # noqa: BLE001 - see above
            self.control_error = f"Camera {self.index} rejected {sorted(changes)}: {exc}"

    def open(self) -> bool:
        self._cap = self._cap_factory(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._write_controls(self._cap, self._current_controls())
        if not self._cap.isOpened():
            self.is_open = False
            return False
        self.is_open = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    # A stalled read is normal for a frame or two; a device that has gone away
    # never recovers. Give up on a deadline rather than a retry count, so the
    # time a user stares at a frozen image does not depend on the backoff.
    FAILURE_TIMEOUT_S = 3.0
    MAX_FAILURE_BACKOFF_S = 0.2

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                now = time.monotonic()
                if self._failing_since is None:
                    self._failing_since = now
                self._consecutive_failures += 1
                if now - self._failing_since >= self.FAILURE_TIMEOUT_S:
                    self.failure = (
                        f"Camera {self.index} stopped delivering frames for "
                        f"{self.FAILURE_TIMEOUT_S:.0f}s ({self._consecutive_failures} "
                        "attempts) — it may have been unplugged, suspended, or taken "
                        "by another program."
                    )
                    self._running = False
                    return
                # Back off instead of spinning: the first few failures retry
                # promptly, a dead device settles at 5 reads/second.
                time.sleep(min(0.005 * self._consecutive_failures, self.MAX_FAILURE_BACKOFF_S))
                continue
            self._consecutive_failures = 0
            self._failing_since = None
            self._drain_controls()
            self._read_times.append(time.monotonic())
            self._seq += 1
            self._buffer.put(self._seq, frame)

    MEASURED_FPS_WINDOW_S = 1.0

    @property
    def measured_fps(self) -> float:
        """Frames delivered per second over the last second, 0.0 until known.

        A plain count over a fixed window — NOT (count-1)/span between the
        first and last sample. That span-based formula blows up whenever
        samples land close together (startup, or a burst right before a
        stall): two frames 1ms apart reports ~1000 fps for a full second,
        which is exactly wrong for a readout whose job is to catch a
        starved camera. A fixed-window count has no such edge case — it
        ramps up honestly from 0 and decays to 0 during a stall.
        """
        now = time.monotonic()
        # Snapshot before iterating: the capture thread appends to
        # _read_times concurrently, and deque raises "deque mutated during
        # iteration" if a mutation lands mid-comprehension.
        snapshot = list(self._read_times)
        recent = [t for t in snapshot if now - t <= self.MEASURED_FPS_WINDOW_S]
        return len(recent) / self.MEASURED_FPS_WINDOW_S

    def latest(self):
        return self._buffer.get()

    def read(self):
        got = self._buffer.get()
        return None if got is None else got[1]

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self.is_open = False
