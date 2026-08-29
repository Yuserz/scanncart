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

import time
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
    # The focus probe reads immediately after cap.set(CAP_PROP_FOCUS, ...) —
    # exactly when a refocusing UVC camera is most likely to drop a frame —
    # so both reads are guarded the same way _brightness() guards its own:
    # an unavailable frame makes the control unmeasurable, and an
    # unmeasurable control is exactly an unsupported one, not a crash.
    before_sharp = _sharpness(read_frame)
    cap.set(cv2.CAP_PROP_FOCUS, 30)
    after_sharp = _sharpness(read_frame)
    if before_sharp is None or after_sharp is None:
        support.focus = False
    else:
        support.focus = abs(after_sharp - before_sharp) >= max(before_sharp, 1.0) * FOCUS_RELATIVE_THRESHOLD
    return support


def _sharpness(read_frame: Callable[[], np.ndarray]) -> float | None:
    """Sharpness of one fresh read, or None when the read itself dropped —
    `cv2.cvtColor(None, ...)` raises `cv2.error`, so this is the guard that
    keeps a single flaky read from escaping as an HTTP 500 mid-calibration."""
    frame = read_frame()
    return frame_quality(frame).sharpness if frame is not None else None


def _default_open(index: int, backend: int):
    return cv2.VideoCapture(index, backend)


def calibrate(
    index: int,
    width: int,
    height: int,
    open_device: Callable[[int, int], object] = _default_open,
    device_name: str = "",
    sample_seconds: float = 3.0,
) -> CameraProfile:
    """Measure one camera and recommend settings.

    Opens the device exclusively, so the caller must have stopped capture.

    `probe_controls` is destructive (see its docstring): it leaves
    brightness/exposure/gain/focus pinned to whatever probe value last
    stuck, and names this function as the caller responsible for reopening
    the device afterward. This function does exactly that — release the
    probed handle and open a fresh one — both so the exposure-capped fps
    measurement below reflects only the exposure cap (not leftover probe
    state) and so the device is left clean for whoever uses it next.
    """
    from app.camera_derive import derive_camera_settings
    from app.camera_quality import measure_fps

    def _open_and_prime():
        # Mirrors camera.py's _default_capture: MSMF delivers the full
        # framerate on Windows but some devices refuse to open under it;
        # fall back to OpenCV's auto backend rather than handing back an
        # unopened handle that then fails far away, inside a getter/setter,
        # with no indication the device was never open in the first place.
        opened = open_device(index, cv2.CAP_MSMF)
        if not opened.isOpened():
            release = getattr(opened, "release", None)
            if callable(release):
                release()
            opened = open_device(index, cv2.CAP_ANY)
        if not opened.isOpened():
            return None
        opened.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        opened.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(10):
            opened.read()
        return opened

    cap = _open_and_prime()
    if cap is None:
        # Skip probing gracefully: no device to measure means no profile,
        # not a crash from calling .set()/.read() on a handle that was never
        # open.
        raise RuntimeError(
            f"Could not open camera {index} for calibration (tried MSMF and "
            "the auto backend)."
        )
    try:
        def read_ok() -> bool:
            ok, _ = cap.read()
            return bool(ok)

        def read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        fps_auto = measure_fps(read_ok, seconds=sample_seconds)
        brightness = _brightness(read_frame)
        controls = probe_controls(cap, read_frame)

        # Reopen: probe_controls just dirtied brightness/exposure/gain/focus.
        # Measuring the exposure-capped rate on the same handle would
        # confound the exposure effect with whatever probing left behind,
        # and skipping this reopen entirely would silently leave the device
        # in that dirtied state for good.
        release = getattr(cap, "release", None)
        if callable(release):
            release()
        cap = _open_and_prime()
        if cap is None:
            raise RuntimeError(
                f"Could not reopen camera {index} after probing controls."
            )

        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        fps_capped = measure_fps(read_ok, seconds=sample_seconds)

        profile = CameraProfile(
            device_key=f"{device_name}:{index}:{width}x{height}",
            backend="msmf", width=width, height=height,
            fps_auto_exposure=round(fps_auto, 1),
            fps_capped_exposure=round(fps_capped, 1),
            controls=controls, measured_at=time.time(),
        )
        profile.recommended = derive_camera_settings(profile, brightness)
        return profile
    finally:
        release = getattr(cap, "release", None)
        if callable(release):
            release()
