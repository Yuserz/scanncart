"""Frame metrics and the thresholds that judge them.

Pure functions over arrays: no device handling, so tests run on synthetic
frames. The thresholds live here so the API, the wizard and the tests share one
source rather than three drifting copies.
"""

from dataclasses import dataclass

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
