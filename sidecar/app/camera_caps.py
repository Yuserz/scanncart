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

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from app.camera_quality import BRIGHTNESS_TARGET, FOCUS_DRIFT_MAX, FPS_MIN, focus_drift, frame_quality
from app.camera_search import SearchResult, search_for_peak, search_to_target

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
    # Measured, not inferred. camera_derive used to conclude "some control
    # worked, so autofocus is probably lockable" — a proxy it documented as
    # such. A swept focus value is worthless if the lock does not take.
    autofocus: bool = False


# Bumped when the sweep's algorithm or thresholds change, so older profiles
# are known-stale without re-deriving anything. 0 means the profile predates
# level measurement entirely.
SWEEP_VERSION = 1


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
    # Evidence: what the device actually did, per control. Kept separate from
    # `recommended`, which is policy — that split is what lets
    # derive_camera_settings stay pure and be re-run against a stored profile.
    measured: dict = field(default_factory=dict)
    sweep_version: int = 0


def _brightness(read_frame: Callable[[], np.ndarray], samples: int = 5) -> float:
    vals = []
    for _ in range(samples):
        frame = read_frame()
        if frame is not None:
            vals.append(frame_quality(frame).brightness)
    return sum(vals) / len(vals) if vals else 0.0


# Pin focus somewhere mid-range for the lock test. The exact distance does
# not matter — we are watching whether the lens stays put, not whether it is
# sharp.
AUTOFOCUS_PROBE_FOCUS = 300


def probe_autofocus(cap, read_frame: Callable[[], np.ndarray], samples: int = 10) -> bool:
    """Whether the device honours CAP_PROP_AUTOFOCUS=0.

    Measured, never asked: turn autofocus off, pin a focus value, then watch
    sharpness over a still scene. A lens still hunting moves; one that
    accepted the lock holds. `focus_drift`/`FOCUS_DRIFT_MAX` are the same
    stability test the calibration design uses for its focus step.

    A fixed-focus camera with no autofocus at all also holds still and so
    reports True. That is harmless — the recommendation it earns is a write
    to a property the device ignores — and the alternative would be
    distinguishing "no autofocus" from "locked autofocus" through a getter
    this module treats as a liar.
    """
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, AUTOFOCUS_PROBE_FOCUS)
    seen = [s for s in (_sharpness(read_frame) for _ in range(samples)) if s is not None]
    if len(seen) < 3:
        # Too few readings to call it stable. Unmeasurable is unsupported.
        return False
    # A degenerate signal (no gradient at all, e.g. a uniform frame from a
    # device that ignores every write) makes focus_drift's mean <= 0.0, and
    # focus_drift guards that by returning 0.0 — which would then read as
    # "perfectly stable" and pass the drift check. That is a false positive,
    # not stability: a flat frame proves nothing about whether the lens is
    # holding position, only that nothing was measurable. probe_controls
    # already treats an unmeasurable focus control as unsupported for the
    # same reason; apply that rule here too.
    if statistics.fmean(seen) <= 0.0:
        return False
    return focus_drift(seen) <= FOCUS_DRIFT_MAX


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
    # Last: it writes CAP_PROP_FOCUS itself, and the focus check above needs
    # the device's own focus behaviour undisturbed by a lock.
    support.autofocus = probe_autofocus(cap, read_frame)
    return support


def _sharpness(read_frame: Callable[[], np.ndarray]) -> float | None:
    """Sharpness of one fresh read, or None when the read itself dropped —
    `cv2.cvtColor(None, ...)` raises `cv2.error`, so this is the guard that
    keeps a single flaky read from escaping as an HTTP 500 mid-calibration."""
    frame = read_frame()
    return frame_quality(frame).sharpness if frame is not None else None


# Bounds mirror SettingsUpdateRequest's ge/le — a value outside them is
# rejected by the API that would have to apply it.
EXPOSURE_RANGE = (-13.0, 0.0)
BRIGHTNESS_RANGE = (0.0, 255.0)
FOCUS_RANGE = (0.0, 1023.0)
# Focus to +/- one step of this. Finer buys nothing: depth of field on a
# fixed counter camera is far wider than 16 units.
FOCUS_STEP = 16.0
# Half the distance between BRIGHTNESS_MIN and BRIGHTNESS_MAX: anywhere in
# the band the quality readout calls "ok" is a hit.
BRIGHTNESS_TOLERANCE = 25.0
# Sharpness must move at least this much across a focus sweep for the peak to
# be real rather than sensor noise. A soft StreamCam frame measures ~5.
MIN_SHARPNESS_SPAN = 10.0
# Frames discarded after a write, then averaged, per probe. A UVC device
# applies a control within a frame or two; averaging three rides out noise.
SETTLE_FRAMES = 3
AVERAGE_FRAMES = 3


class ProbeUnavailable(RuntimeError):
    """The device stopped delivering frames mid-probe, so this control cannot
    be measured. Raised rather than scored, because any score would be a
    fabricated data point the search would then optimise against."""


def exposure_ceiling(fps_floor: float) -> int:
    """The longest exposure, in log2 seconds, that still permits `fps_floor`.

    Windows exposure is log2 seconds: a value e holds the shutter open 2^e
    seconds and so caps delivery at 1/2^e fps. That makes the ceiling
    arithmetic. Measuring it would mean walking the search off the framerate
    cliff to discover where the edge was.
    """
    lo, hi = EXPOSURE_RANGE
    if fps_floor <= 0:
        return int(hi)
    return int(max(lo, min(hi, math.floor(-math.log2(fps_floor)))))


def _metric_probe(cap, read_frame, prop, metric_of):
    """A probe closure: write the control, let it settle, average the metric."""

    def probe(value: float) -> float:
        cap.set(prop, value)
        for _ in range(SETTLE_FRAMES):
            read_frame()
        seen = [m for m in (metric_of(read_frame) for _ in range(AVERAGE_FRAMES)) if m is not None]
        if not seen:
            # One retry: a single dropped read is normal right after a write.
            seen = [m for m in (metric_of(read_frame) for _ in range(AVERAGE_FRAMES)) if m is not None]
        if not seen:
            raise ProbeUnavailable(f"no frames while probing property {prop}")
        return sum(seen) / len(seen)

    return probe


def _mean_brightness(read_frame) -> float | None:
    frame = read_frame()
    return frame_quality(frame).brightness if frame is not None else None


def _record(result: SearchResult, baseline: float) -> dict:
    return {
        "value": result.value,
        "metric": round(result.metric, 1),
        "baseline": round(baseline, 1),
        "reached": result.reached,
        "probes": result.probes,
    }


def sweep_controls(
    cap,
    read_frame: Callable[[], np.ndarray],
    support: ControlSupport,
    *,
    fps_floor: float,
) -> dict[str, dict]:
    """Search each supported control for its best value.

    Order is a dependency chain, not a preference:

    1. Autofocus off, or the device hunts away from any focus value written.
    2. Exposure onto the brightness target — sharpness is unmeasurable on a
       badly exposed frame, because a blown-out or black image has no
       gradient for the Laplacian to find.
    3. Focus for the sharpness peak, now that the image is exposed properly.
    4. Brightness, only to trim what exposure could not reach. It amplifies
       noise along with the picture, so it goes last and does the least.

    Returns evidence keyed by `Settings` field name. A control that cannot be
    measured is omitted rather than guessed at; the caller distinguishes
    "no recommendation" from "recommendation of zero".
    """
    measured: dict[str, dict] = {}

    if support.autofocus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if support.exposure:
        baseline = _mean_brightness(read_frame) or 0.0
        try:
            result = search_to_target(
                _metric_probe(cap, read_frame, cv2.CAP_PROP_EXPOSURE, _mean_brightness),
                lo=EXPOSURE_RANGE[0],
                hi=float(exposure_ceiling(fps_floor)),
                target=BRIGHTNESS_TARGET,
                tolerance=BRIGHTNESS_TOLERANCE,
            )
        except ProbeUnavailable:
            pass
        else:
            measured["camera_exposure"] = _record(result, baseline)

    # Focus only when the lock will actually take. Otherwise the lens wanders
    # off whatever we find, and shipping it anyway would be the worst
    # outcome: confident and wrong.
    if support.focus and support.autofocus:
        baseline = _sharpness(read_frame) or 0.0
        try:
            result = search_for_peak(
                _metric_probe(cap, read_frame, cv2.CAP_PROP_FOCUS, _sharpness),
                lo=FOCUS_RANGE[0],
                hi=FOCUS_RANGE[1],
                step=FOCUS_STEP,
                min_span=MIN_SHARPNESS_SPAN,
            )
        except ProbeUnavailable:
            pass
        else:
            if result.reached:
                measured["camera_focus"] = _record(result, baseline)

    if support.brightness:
        baseline = _mean_brightness(read_frame) or 0.0
        try:
            result = search_to_target(
                _metric_probe(cap, read_frame, cv2.CAP_PROP_BRIGHTNESS, _mean_brightness),
                lo=BRIGHTNESS_RANGE[0],
                hi=BRIGHTNESS_RANGE[1],
                target=BRIGHTNESS_TARGET,
                tolerance=BRIGHTNESS_TOLERANCE,
            )
        except ProbeUnavailable:
            pass
        else:
            measured["camera_brightness"] = _record(result, baseline)

    return measured


def _default_open(index: int, backend: int):
    return cv2.VideoCapture(index, backend)


def device_key_for(device_name: str, index: int, width: int, height: int) -> str:
    """The identity a CameraProfile is stored under.

    Includes the resolution because control support is measured at one: a
    device can accept exposure at 720p and ignore it at 1080p. Used by both
    calibrate() and GET /api/camera/profile, so the format lives here once.
    """
    return f"{device_name}:{index}:{width}x{height}"


def calibrate(
    index: int,
    width: int,
    height: int,
    open_device: Callable[[int, int], object] = _default_open,
    device_name: str = "",
    sample_seconds: float = 3.0,
    target_fps: float | None = None,
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

        # The sweep runs on the freshly reopened handle: probe_controls left
        # every control pinned to its probe value, and searching from there
        # would confound the search with leftover probe state.
        fps_floor = target_fps * 0.8 if target_fps is not None else FPS_MIN
        measured = sweep_controls(cap, read_frame, controls, fps_floor=fps_floor)

        # Measure the capped rate at the exposure we will actually recommend,
        # not at a hardcoded -6. The old number described a setting nobody was
        # going to apply.
        chosen_exposure = measured.get("camera_exposure", {}).get("value", -6)
        cap.set(cv2.CAP_PROP_EXPOSURE, chosen_exposure)
        fps_capped = measure_fps(read_ok, seconds=sample_seconds)

        profile = CameraProfile(
            device_key=device_key_for(device_name, index, width, height),
            backend="msmf", width=width, height=height,
            fps_auto_exposure=round(fps_auto, 1),
            fps_capped_exposure=round(fps_capped, 1),
            controls=controls, measured_at=time.time(),
            measured=measured, sweep_version=SWEEP_VERSION,
        )
        profile.recommended = derive_camera_settings(profile, brightness, target_fps=target_fps)
        return profile
    finally:
        release = getattr(cap, "release", None)
        if callable(release):
            release()
