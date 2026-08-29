"""Measured camera capability.

Nothing here trusts `cap.get()`. Setting CAP_PROP_EXPOSURE on MSMF took
delivered framerate from 12.3 to 30.3 fps while the getter kept returning the
old value, so a control counts as supported only when the image or the frame
rate visibly changes. That is also what keeps this brand-independent.
"""

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from app.camera_quality import frame_quality

# A control must move mean brightness by at least this much to count. Below it
# we cannot distinguish a real effect from sensor noise.
EFFECT_THRESHOLD = 6.0


@dataclass
class ControlSupport:
    brightness: bool = False
    exposure: bool = False
    gain: bool = False
    focus: bool = False


@dataclass
class CameraProfile:
    device_key: str
    backend: str
    width: int
    height: int
    fps_auto_exposure: float
    fps_capped_exposure: float
    controls: ControlSupport
    recommended: dict = field(default_factory=dict)
    measured_at: float = 0.0


def _brightness(read_frame: Callable[[], np.ndarray], samples: int = 5) -> float:
    vals = []
    for _ in range(samples):
        frame = read_frame()
        if frame is not None:
            vals.append(frame_quality(frame).brightness)
    return sum(vals) / len(vals) if vals else 0.0


def probe_controls(cap, read_frame: Callable[[], np.ndarray]) -> ControlSupport:
    """Which controls actually do something on this device."""
    support = ControlSupport()
    baseline = _brightness(read_frame)
    for prop, name, value in (
        (cv2.CAP_PROP_BRIGHTNESS, "brightness", 255),
        (cv2.CAP_PROP_EXPOSURE, "exposure", -3),
        (cv2.CAP_PROP_GAIN, "gain", 255),
    ):
        before = _brightness(read_frame)
        cap.set(prop, value)
        after = _brightness(read_frame)
        setattr(support, name, abs(after - before) >= EFFECT_THRESHOLD)
        cap.set(prop, cap.get(prop))
    # Focus does not change mean brightness, so judge it on sharpness instead.
    before_sharp = frame_quality(read_frame()).sharpness
    cap.set(cv2.CAP_PROP_FOCUS, 30)
    after_sharp = frame_quality(read_frame()).sharpness
    support.focus = abs(after_sharp - before_sharp) >= 1.0
    _ = baseline
    return support
