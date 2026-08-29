"""Frame metrics and the thresholds that judge them.

Pure functions over arrays: no device handling, so tests run on synthetic
frames. The thresholds live here so the API, the wizard and the tests share one
source rather than three drifting copies.
"""

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

# A frame this dark detected nothing on 2026-08-29 (measured 23/255).
BRIGHTNESS_MIN = 110.0
BRIGHTNESS_TARGET = 130.0
BRIGHTNESS_MAX = 160.0
# Laplacian variance. A soft StreamCam frame measured ~5; a focused one is
# in the hundreds.
SHARPNESS_MIN = 60.0
# stdev/mean of sharpness over a still scene. Autofocus hunting showed 6->31.
FOCUS_DRIFT_MAX = 0.25
FPS_MIN = 25.0


@dataclass
class FrameQuality:
    brightness: float
    contrast: float
    sharpness: float


def frame_quality(frame: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return FrameQuality(
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def focus_drift(samples: list[float]) -> float:
    """Relative spread of sharpness over a still scene.

    Autofocus hunting shows up as a large spread with nothing moving. Fewer
    than two samples cannot show drift, so they report none.
    """
    if len(samples) < 2:
        return 0.0
    mean = statistics.fmean(samples)
    if mean <= 0.0:
        return 0.0
    return statistics.pstdev(samples) / mean


def measure_fps(
    read: Callable[[], bool],
    seconds: float = 3.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Frames actually delivered per second.

    The only trustworthy capability signal: a camera reports 60 while
    delivering 12, and setting exposure changes the rate without changing what
    any getter returns. `clock`/`sleep` are injected so tests run instantly.
    """
    start = clock()
    frames = 0
    while clock() - start < seconds:
        if read():
            frames += 1
        else:
            sleep(0.005)
    elapsed = clock() - start
    return frames / elapsed if elapsed > 0 else 0.0
