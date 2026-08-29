from app.camera_caps import CameraProfile, ControlSupport
from app.camera_derive import derive_camera_settings


def _profile(**over):
    base = dict(
        device_key="Logitech StreamCam:1:1920x1080", backend="msmf",
        width=1280, height=720,
        fps_auto_exposure=12.3, fps_capped_exposure=30.3,
        controls=ControlSupport(brightness=True, exposure=True, gain=False, focus=False),
        recommended={}, measured_at=0.0,
    )
    base.update(over)
    return CameraProfile(**base)


def test_a_dark_streamcam_gets_exposure_capped_and_brightness_raised():
    """The measured case: 12 fps on auto exposure, 30 with it capped."""
    patch = derive_camera_settings(_profile(), measured_brightness=23.0)

    assert patch["camera_exposure"] is not None
    assert patch["camera_brightness"] is not None
    assert patch["camera_autofocus"] is False   # face-tracking AF is wrong here


def test_nothing_is_proposed_for_a_camera_that_is_already_good():
    patch = derive_camera_settings(
        _profile(fps_auto_exposure=30.0), measured_brightness=130.0
    )
    assert "camera_brightness" not in patch
    assert "camera_exposure" not in patch


def test_no_settings_are_invented_for_a_device_that_supports_nothing():
    patch = derive_camera_settings(
        _profile(controls=ControlSupport()), measured_brightness=23.0
    )
    assert patch == {}


def test_exposure_is_not_lengthened_below_the_fps_floor():
    """Longer exposure buys brightness and costs frames; 1/4 s capped the
    camera at 4 fps. Brightness must not be bought below FPS_MIN."""
    patch = derive_camera_settings(
        _profile(fps_capped_exposure=10.0, fps_auto_exposure=10.0),
        measured_brightness=23.0,
    )
    assert "camera_exposure" not in patch


def test_a_weak_far_confidence_proposes_a_larger_inference_size():
    """A distant SKU is a small SKU; imgsz is the cheapest lever, affordable
    now that CUDA runs the model at ~20 ms."""
    patch = derive_camera_settings(
        _profile(), measured_brightness=130.0, near_conf=0.85, far_conf=0.35, imgsz=640
    )
    assert patch["imgsz"] > 640
    assert patch["imgsz"] % 32 == 0


def test_exposure_gate_is_relative_to_the_configured_capture_rate():
    """The shipped low_end preset sets capture_fps=15 (presets.py), so a
    camera delivering 20 fps with exposure capped is comfortably above its
    own target even though it is below the absolute FPS_MIN=25 floor.
    Gating on the absolute floor silently withheld camera_exposure on
    exactly that hardware; the gate must match main.py's capture-fps VERDICT
    (fps < target_fps * 0.8), not a fixed floor, whenever a target is given."""
    profile = _profile(fps_capped_exposure=20.0, fps_auto_exposure=10.0)

    patch_low_target = derive_camera_settings(profile, measured_brightness=23.0, target_fps=15.0)
    assert patch_low_target["camera_exposure"] is not None

    patch_high_target = derive_camera_settings(profile, measured_brightness=23.0, target_fps=30.0)
    assert "camera_exposure" not in patch_high_target


def test_exposure_gate_falls_back_to_the_absolute_floor_without_a_target():
    """Existing callers that never pass target_fps must keep the old
    behavior exactly — the absolute FPS_MIN floor."""
    patch = derive_camera_settings(
        _profile(fps_capped_exposure=10.0, fps_auto_exposure=10.0),
        measured_brightness=23.0,
    )
    assert "camera_exposure" not in patch


def test_a_consistent_near_and_far_confidence_leaves_imgsz_alone():
    patch = derive_camera_settings(
        _profile(), measured_brightness=130.0, near_conf=0.85, far_conf=0.80, imgsz=640
    )
    assert "imgsz" not in patch
