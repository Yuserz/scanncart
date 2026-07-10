import threading
from typing import Protocol
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
