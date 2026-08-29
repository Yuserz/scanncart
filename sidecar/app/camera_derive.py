"""Turn a measured CameraProfile into a settings patch.

Pure and total: no I/O, no device. All the judgement lives here so it can be
tested against hand-written profiles.

The objective, once: when the measured frame is below BRIGHTNESS_MIN, propose
a fixed brightness boost and (fps permitting) a capped exposure for the
operator to review on the calibration card. This is a flat boost, not a value
computed to land on BRIGHTNESS_TARGET: the control's units are device-specific
and its transfer function is unknown without a dedicated brightness sweep,
which is out of scope here. Exposure is preferred over brightness because it
is real light rather than amplification, but it is also what costs frames, so
the fps floor gates it.
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


def derive_camera_settings(
    profile: CameraProfile,
    measured_brightness: float,
    near_conf: float | None = None,
    far_conf: float | None = None,
    imgsz: int = 640,
    target_fps: float | None = None,
) -> dict:
    patch: dict = {}

    # The StreamCam's smart AF/AE follows faces. A counter has none, so auto
    # has nothing to lock onto and hunts. Lock it whenever any control that
    # implies a settled, controllable camera is present. NOTE: this is a
    # proxy inference, not a measured capability — ControlSupport has no
    # `autofocus` field, and CAP_PROP_AUTOFOCUS (a distinct UVC property) is
    # never probed by camera_caps.probe_controls.
    if profile.controls.focus or profile.controls.exposure or profile.controls.brightness:
        patch["camera_autofocus"] = False

    # Relative to what the operator actually configured, exactly like the
    # capture-fps VERDICT in main.py's /api/camera/quality — the shipped
    # low_end preset asks for capture_fps=15, so gating on the absolute
    # FPS_MIN=25 silently withheld camera_exposure on that hardware even
    # when the capped-exposure rate comfortably cleared its own target.
    # FPS_MIN remains the fallback floor when no target is supplied (e.g.
    # existing callers/tests that predate this parameter).
    fps_floor = target_fps * 0.8 if target_fps is not None else FPS_MIN
    too_dark = measured_brightness < BRIGHTNESS_MIN
    if too_dark and profile.controls.exposure and profile.fps_capped_exposure >= fps_floor:
        patch["camera_exposure"] = EXPOSURE_CAPPED
    if too_dark and profile.controls.brightness:
        patch["camera_brightness"] = BRIGHTNESS_BOOST

    # A distant item is a small item. Raising imgsz keeps more of it, and CUDA
    # made that affordable (~20 ms/frame on the custom model).
    if near_conf is not None and far_conf is not None:
        if near_conf - far_conf >= FAR_CONF_GAP and imgsz < IMGSZ_MAX:
            patch["imgsz"] = min(imgsz + IMGSZ_STEP, IMGSZ_MAX)

    return patch
