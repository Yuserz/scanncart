"""Round-trip tests for camera_profiles.py — the only module on this branch
whose correctness previously rested on a reviewer's manual execution rather
than a committed test.

Every test uses pytest's tmp_path so nothing here ever touches the real
data/camera_profiles.json.
"""

import json
from dataclasses import asdict

from app.camera_caps import CameraProfile, ControlSupport
from app.camera_profiles import load_profiles, save_profile


def _profile(device_key: str = "Logitech StreamCam:0:1280x720", **over) -> CameraProfile:
    base = dict(
        device_key=device_key, backend="msmf", width=1280, height=720,
        fps_auto_exposure=12.3, fps_capped_exposure=30.3,
        controls=ControlSupport(brightness=True, exposure=True, gain=False, focus=False),
        recommended={"camera_exposure": -6.0, "camera_brightness": 180.0},
        measured_at=1234.5,
    )
    base.update(over)
    return CameraProfile(**base)


def test_save_then_load_reconstructs_a_real_control_support_dataclass(tmp_path):
    path = tmp_path / "camera_profiles.json"
    profile = _profile()

    save_profile(profile, str(path))
    loaded = load_profiles(str(path))

    round_tripped = loaded[profile.device_key]
    assert round_tripped == profile
    # json.dump/load round-trips the dataclass through a plain dict — the
    # whole point of load_profiles's `ControlSupport(**...)` reconstruction
    # step is that this must come back as the dataclass, not that dict.
    assert isinstance(round_tripped.controls, ControlSupport)
    assert not isinstance(round_tripped.controls, dict)


def test_a_missing_file_yields_no_profiles(tmp_path):
    path = tmp_path / "does_not_exist.json"

    assert load_profiles(str(path)) == {}


def test_a_corrupt_file_yields_no_profiles_rather_than_raising(tmp_path):
    path = tmp_path / "camera_profiles.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    assert load_profiles(str(path)) == {}  # must not raise


def test_a_malformed_entry_is_skipped_while_its_valid_sibling_still_loads(tmp_path):
    path = tmp_path / "camera_profiles.json"
    good = _profile(device_key="Good Cam:0:1280x720")
    raw = {
        good.device_key: asdict(good),
        # Missing every required field (device_key, backend, height,
        # fps_auto_exposure, fps_capped_exposure) — CameraProfile(**value)
        # raises TypeError, which load_profiles must catch per-entry rather
        # than losing the whole file over one bad row.
        "Bad Cam:1:640x480": {"width": "not-an-int"},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_profiles(str(path))

    assert set(loaded) == {good.device_key}
    assert loaded[good.device_key] == good


def test_saving_a_second_profile_preserves_the_first(tmp_path):
    path = tmp_path / "camera_profiles.json"
    first = _profile(device_key="Cam A:0:1280x720")
    second = _profile(device_key="Cam B:1:1920x1080", width=1920, height=1080)

    save_profile(first, str(path))
    save_profile(second, str(path))

    loaded = load_profiles(str(path))

    assert set(loaded) == {first.device_key, second.device_key}
    assert loaded[first.device_key] == first
    assert loaded[second.device_key] == second


# --- reading a profile back ----------------------------------------------

from app.camera_caps import CameraProfile, ControlSupport, device_key_for
from app.camera_profiles import save_profile
from app.main import AppState, build_app
from fastapi.testclient import TestClient


def test_device_key_for_matches_what_calibration_writes():
    """One format string, used by both the writer and the reader — a second
    copy would drift and silently orphan every stored profile."""
    assert device_key_for("Logitech StreamCam", 1, 1280, 720) == (
        "Logitech StreamCam:1:1280x720"
    )


def _stored_profile(device_key: str) -> CameraProfile:
    return CameraProfile(
        device_key=device_key,
        backend="MSMF",
        width=1280,
        height=720,
        fps_auto_exposure=29.9,
        fps_capped_exposure=30.8,
        controls=ControlSupport(brightness=True, exposure=True, gain=False, focus=False),
        recommended={"camera_autofocus": False},
        measured_at=1.0,
    )


def test_a_stored_profile_is_returned_for_the_current_camera(tmp_path):
    """Profiles were written and never read back, so a calibration did not
    survive an app restart. This is that round trip."""
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        camera_namer=lambda: ["Logitech StreamCam"],
    )
    state.settings.camera_index = 0
    state.settings.capture_width = 1280
    state.settings.capture_height = 720
    save_profile(_stored_profile("Logitech StreamCam:0:1280x720"), str(tmp_path / "camera_profiles.json"))

    with TestClient(build_app(lambda: state)) as client:
        body = client.get("/api/camera/profile").json()

    assert body["profile"]["controls"]["brightness"] is True
    assert body["profile"]["controls"]["focus"] is False


def test_an_uncalibrated_camera_returns_a_null_profile(tmp_path):
    """A normal state the card renders, not an error it handles — hence 200
    with a null field rather than the 404 its /apply sibling uses."""
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        camera_namer=lambda: ["Some Other Camera"],
    )
    with TestClient(build_app(lambda: state)) as client:
        r = client.get("/api/camera/profile")

    assert r.status_code == 200
    assert r.json()["profile"] is None


def test_a_profile_for_a_different_resolution_does_not_match(tmp_path):
    """Control support is measured at a resolution; the key includes it."""
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        camera_namer=lambda: ["Logitech StreamCam"],
    )
    state.settings.camera_index = 0
    state.settings.capture_width = 640
    state.settings.capture_height = 480
    save_profile(_stored_profile("Logitech StreamCam:0:1280x720"), str(tmp_path / "camera_profiles.json"))

    with TestClient(build_app(lambda: state)) as client:
        assert client.get("/api/camera/profile").json()["profile"] is None
