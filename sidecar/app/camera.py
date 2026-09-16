import threading
import time
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


# Modes tried in order by open_verified(). The first entry is the preferred
# full-quality mode; the rest are degraded fallbacks so a capture start still
# yields frames when the camera/driver refuses the primary mode. The settings
# API never auto-edits capture_* settings — the user sees the exact mode that
# produced their frames.
FALLBACK_MODES: list[tuple[int, int, int]] = [
    (1920, 1080, 60),
    (1920, 1080, 30),
    (1280, 720, 30),
    (640, 480, 30),
]

# MSMF "opened but zero frames" wedge: the OpenCV/MSMF capture reports
# is_opened=True but read() fails indefinitely (cap_msmf 'can't grab frame'
# errors). It can hit ANY resolution after repeated open/close cycles, and
# often needs a physical unplug/replug to clear. Always verify with real
# frames; never trust isOpened() alone.
FRAME_WARMUP_ATTEMPTS = 30


def _warmup_read(cap) -> tuple[bool, np.ndarray | None]:
    """Pull up to FRAME_WARMUP_ATTEMPTS frames, succeeding on the first read."""
    for _ in range(FRAME_WARMUP_ATTEMPTS):
        ok, frame = cap.read()
        if ok:
            return True, frame
    return False, None


def open_verified(index: int, width: int, height: int, fps: int) -> tuple[object, tuple[int, int, int] | None]:
    """Open the camera and confirm it actually delivers frames.

    Tries the requested mode first, then FALLBACK_MODES, and returns
    (cap, mode_used) where mode_used is the (w, h, fps) that produced a real
    frame — None if every attempt failed. Caller owns cap (including
    release() on failure, which we handle here by releasing before returning
    None). Sets FOURCC to MJPG for high-fps modes, the format StreamCam
    delivers 1080p60 in.
    """
    modes = [(width, height, fps)] + [m for m in FALLBACK_MODES if m != (width, height, fps)]
    for w, h, f in modes:
        cap = _default_capture(index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, f)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _frame = _warmup_read(cap)
        if ok:
            return cap, (w, h, f)
        cap.release()
    return None, None


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

    def __init__(self, index, width, height, fps, cap_factory=_default_capture):
        self.index = index
        self.width = width
        self.height = height
        self.fps = float(fps)
        self._cap_factory = cap_factory
        self._cap = None
        self._buffer = LatestFrameBuffer()
        self._thread = None
        self._running = False
        self._seq = 0
        self.is_open = False

    def open(self) -> bool:
        self._cap = self._cap_factory(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            self.is_open = False
            return False
        self.is_open = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            self._seq += 1
            self._buffer.put(self._seq, frame)

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
