"""Measured camera capability.

Nothing here trusts `cap.get()`. Setting CAP_PROP_EXPOSURE on MSMF took
delivered framerate from 12.3 to 30.3 fps while the getter kept returning the
old value, so a control counts as supported only when the image or the frame
rate visibly changes. That is also what keeps this brand-independent.

`probe_controls` is DESTRUCTIVE: it leaves the device holding whatever values
it set (brightness/exposure/gain pinned to their probe value, focus left at
its probe position) rather than "restoring" them, because restoring a value
by reading it back through `cap.get()` would mean trusting the very getter
this module treats as a liar — incoherent by construction. The caller is
responsible for reopening the device afterwards to get a clean state; Task
11's `calibrate()` does exactly that.
"""

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from app.camera_quality import frame_quality

# A control must move mean brightness by at least this much to count. Below it
# we cannot distinguish a real effect from sensor noise.
EFFECT_THRESHOLD = 6.0

# Focus is judged on Laplacian variance (camera_quality.frame_quality), which
# ranges from ~5 on a soft frame to hundreds on a sharp one — a fixed absolute
# delta would be noise-level at the low end and resolution-dependent besides.
# A relative threshold stays meaningful across that whole range.
FOCUS_RELATIVE_THRESHOLD = 0.2


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
    """Which controls actually do something on this device.

    Destructive: see module docstring. Each probed control is left set to
    its probe value; nothing here attempts to restore prior state.
    """
    support = ControlSupport()
    for prop, name, value in (
        (cv2.CAP_PROP_BRIGHTNESS, "brightness", 255),
        (cv2.CAP_PROP_EXPOSURE, "exposure", -3),
        (cv2.CAP_PROP_GAIN, "gain", 255),
    ):
        before = _brightness(read_frame)
        cap.set(prop, value)
        after = _brightness(read_frame)
        setattr(support, name, abs(after - before) >= EFFECT_THRESHOLD)
    # Focus does not change mean brightness, so judge it on sharpness instead.
    before_sharp = frame_quality(read_frame()).sharpness
    cap.set(cv2.CAP_PROP_FOCUS, 30)
    after_sharp = frame_quality(read_frame()).sharpness
    support.focus = abs(after_sharp - before_sharp) >= max(before_sharp, 1.0) * FOCUS_RELATIVE_THRESHOLD
    return support
