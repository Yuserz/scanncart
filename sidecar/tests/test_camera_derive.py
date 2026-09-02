from app.camera_caps import CameraProfile, ControlSupport
from app.camera_derive import EXPOSURE_CAPPED, derive_camera_settings


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
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(brightness=True, exposure=True, gain=False,
                                     focus=False, autofocus=True)
        ),
        measured_brightness=23.0,
    )

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


def test_a_measured_exposure_is_preferred_over_the_hardcoded_constant():
    """EXPOSURE_CAPPED was a guess standing in for a sweep that did not
    exist. Where one has run, its answer wins."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(exposure=True, autofocus=True),
            measured={"camera_exposure": {"value": -8.0, "metric": 128.0,
                                          "baseline": 23.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=23.0,
    )

    assert patch["camera_exposure"] == -8.0


def test_a_measured_focus_is_recommended():
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(focus=True, autofocus=True),
            measured={"camera_focus": {"value": 400.0, "metric": 92.0,
                                       "baseline": 31.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=130.0,
    )

    assert patch["camera_focus"] == 400.0


def test_focus_is_withheld_when_the_autofocus_lock_will_not_take():
    """The lens hunts straight off it. Recommending it anyway would be
    confidently wrong, which is worse than recommending nothing."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(focus=True, autofocus=False),
            measured={"camera_focus": {"value": 400.0, "metric": 92.0,
                                       "baseline": 31.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=130.0,
    )

    assert "camera_focus" not in patch
    assert "camera_autofocus" not in patch


def test_autofocus_is_only_locked_on_a_device_that_honours_the_lock():
    """It used to be proposed off a proxy — 'some other control worked'."""
    patch = derive_camera_settings(
        _profile(controls=ControlSupport(brightness=True, autofocus=True)),
        measured_brightness=130.0,
    )
    assert patch["camera_autofocus"] is False

    patch = derive_camera_settings(
        _profile(controls=ControlSupport(brightness=True, autofocus=False)),
        measured_brightness=130.0,
    )
    assert "camera_autofocus" not in patch


def test_an_old_profile_still_yields_the_constant_based_recommendation():
    """sweep_version 0 profiles predate the sweep. derive stays total: they
    get the old behaviour rather than nothing."""
    patch = derive_camera_settings(
        _profile(controls=ControlSupport(brightness=True, exposure=True, autofocus=True)),
        measured_brightness=23.0,
    )

    assert patch["camera_exposure"] == -6.0
    assert patch["camera_brightness"] == 180.0


def test_a_measured_value_is_recommended_even_on_a_frame_that_is_not_dark():
    """The constants only fired below BRIGHTNESS_MIN because a guess is only
    worth risking on a clearly broken image. A measured optimum is not a
    guess, so it applies whenever it exists."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(exposure=True, autofocus=True),
            measured={"camera_exposure": {"value": -8.0, "metric": 128.0,
                                          "baseline": 118.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=118.0,
    )

    assert patch["camera_exposure"] == -8.0


def test_a_non_dict_measured_entry_falls_back_to_the_constant_without_crashing():
    """measured can arrive from load_profiles deserializing a hand-edited or
    truncated camera_profiles.json, with no validation of its interior. A
    malformed entry must read as 'not measured', not raise."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(exposure=True, brightness=True, autofocus=True),
            measured={
                "camera_exposure": "oops",
                "camera_brightness": {"value": 200.0, "metric": 100.0,
                                      "baseline": 20.0, "reached": True},
            },
            sweep_version=1,
        ),
        measured_brightness=23.0,
    )

    assert patch["camera_exposure"] == EXPOSURE_CAPPED
    assert patch["camera_brightness"] == 200.0


def test_a_measured_entry_missing_value_falls_back_to_the_constant_without_crashing():
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(exposure=True, brightness=True, autofocus=True),
            measured={
                "camera_exposure": {"metric": 1.0, "baseline": 1.0, "reached": True},
                "camera_brightness": {"value": 200.0, "metric": 100.0,
                                      "baseline": 20.0, "reached": True},
            },
            sweep_version=1,
        ),
        measured_brightness=23.0,
    )

    assert patch["camera_exposure"] == EXPOSURE_CAPPED
    assert patch["camera_brightness"] == 200.0


def test_a_malformed_focus_entry_is_simply_absent_where_too_dark_does_not_apply():
    """camera_focus has no constant fallback, so a malformed entry just
    yields no recommendation rather than a crash."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(focus=True, autofocus=True),
            measured={"camera_focus": {"metric": 92.0, "baseline": 31.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=130.0,
    )

    assert "camera_focus" not in patch


def test_a_measured_exposure_below_the_fps_floor_is_withheld():
    """The guard stopping a measured optimum from silently capping the
    camera's framerate: fps_capped_exposure below the floor must withhold
    camera_exposure even though a measured value exists."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(exposure=True, autofocus=True),
            fps_capped_exposure=10.0, fps_auto_exposure=10.0,
            measured={"camera_exposure": {"value": -8.0, "metric": 128.0,
                                          "baseline": 23.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=23.0,
    )

    assert "camera_exposure" not in patch
