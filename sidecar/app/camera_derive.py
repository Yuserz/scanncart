"""Turn a measured CameraProfile into a settings patch.

Pure and total: no I/O, no device. All the judgement lives here so it can be
tested against hand-written profiles.

The objective, once: recommend the value the sweep measured. Where no sweep
has run — a profile written before camera_search existed, or a control the
device ignores — fall back to a flat boost below BRIGHTNESS_MIN, which is a
guess and so is only risked on an image that is already clearly broken.
Exposure is preferred over brightness because it is real light rather than
amplification, but it is also what costs frames, so the fps floor gates it.
"""

from app.camera_caps import CameraProfile
from app.camera_quality import BRIGHTNESS_MIN, FPS_MIN

# Shorter than this and the image goes dark faster than brightness can rescue.
EXPOSURE_CAPPED = -6.0
BRIGHTNESS_BOOST = 180.0
# A far confidence this far below near means the model is losing small objects.
FAR_CONF_GAP = 0.25
IMGSZ_STEP = 320
IMGSZ_MAX = 1280


def _measured_value(measured: dict, key: str) -> float | int | None:
    """The numeric `value` of one measured entry, or None if the entry is
    missing or malformed.

    `measured` is not always sweep_controls' own output: it can arrive from
    `load_profiles` deserializing a hand-edited or truncated
    data/camera_profiles.json with no validation of its interior. A
    malformed entry there must read as "not measured" — falling through to
    the constant-based branch exactly as if the control had never been
    swept — rather than raising and taking every other recommendation in
    the patch down with it.
    """
    entry = measured.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def derive_camera_settings(
    profile: CameraProfile,
    measured_brightness: float,
    near_conf: float | None = None,
    far_conf: float | None = None,
    imgsz: int = 640,
    target_fps: float | None = None,
) -> dict:
    patch: dict = {}
    measured = profile.measured or {}

    # Measured, not inferred. This used to fire whenever any control worked,
    # as a proxy for "the camera is settled and controllable" — the code said
    # so. probe_autofocus now answers it directly, and proposing the lock on a
    # device that ignores the property is noise.
    if profile.controls.autofocus:
        patch["camera_autofocus"] = False

    fps_floor = target_fps * 0.8 if target_fps is not None else FPS_MIN
    too_dark = measured_brightness < BRIGHTNESS_MIN

    # A swept value is an observation, so it applies whenever it exists. The
    # constants below are guesses, and a guess is only worth risking on an
    # image that is already clearly broken — hence the `too_dark` gate on
    # those and not on these.
    exposure_value = _measured_value(measured, "camera_exposure")
    if exposure_value is not None and profile.fps_capped_exposure >= fps_floor:
        patch["camera_exposure"] = exposure_value
    elif too_dark and profile.controls.exposure and profile.fps_capped_exposure >= fps_floor:
        patch["camera_exposure"] = EXPOSURE_CAPPED

    brightness_value = _measured_value(measured, "camera_brightness")
    if brightness_value is not None:
        patch["camera_brightness"] = brightness_value
    elif too_dark and profile.controls.brightness:
        patch["camera_brightness"] = BRIGHTNESS_BOOST

    # Focus needs the lock to hold, or the lens wanders off the value within
    # seconds of applying it. Withhold the lock proposal too: on a device
    # that ignores CAP_PROP_AUTOFOCUS it buys nothing.
    focus_value = _measured_value(measured, "camera_focus")
    if focus_value is not None and profile.controls.autofocus:
        patch["camera_focus"] = focus_value

    # A distant item is a small item. Raising imgsz keeps more of it, and CUDA
    # made that affordable (~20 ms/frame on the custom model).
    if near_conf is not None and far_conf is not None:
        if near_conf - far_conf >= FAR_CONF_GAP and imgsz < IMGSZ_MAX:
            patch["imgsz"] = min(imgsz + IMGSZ_STEP, IMGSZ_MAX)

    return patch
