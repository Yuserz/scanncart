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

    def __init__(self, index, width, height, fps, cap_factory=cv2.VideoCapture):
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
