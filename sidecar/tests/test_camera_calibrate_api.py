import pytest
from fastapi.testclient import TestClient

from app.camera_caps import CameraProfile, ControlSupport
from app.main import AppState, build_app, _resolve_camera_name


def _profile():
    return CameraProfile(
        device_key="cam:1", backend="msmf", width=1280, height=720,
        fps_auto_exposure=12.3, fps_capped_exposure=30.3,
        controls=ControlSupport(brightness=True, exposure=True),
        recommended={"camera_exposure": -6.0, "camera_brightness": 180.0},
        measured_at=1.0,
    )


def _client(tmp_path):
    state = AppState(settings_path=str(tmp_path / "s.json"), db_path=":memory:",
                     calibrator=lambda: _profile())
    return TestClient(build_app(lambda: state)), state


def test_calibrate_returns_a_profile_without_applying_it(tmp_path):
    """Review-first: the operator sees the evidence before anything changes."""
    client, state = _client(tmp_path)
    body = client.post("/api/camera/calibrate").json()

    assert body["recommended"]["camera_exposure"] == -6.0
    assert body["fps_capped_exposure"] == 30.3
    assert state.settings.camera_exposure is None   # untouched


def test_apply_writes_the_recommendation(tmp_path):
    client, state = _client(tmp_path)
    client.post("/api/camera/calibrate")
    r = client.post("/api/camera/profile/apply")

    assert r.status_code == 200
    assert state.settings.camera_exposure == -6.0
    assert state.settings.camera_brightness == 180.0


def test_calibrate_is_refused_while_capture_is_running(tmp_path):
    """The device is exclusive; calibrating would fight the pipeline."""
    client, state = _client(tmp_path)
    state.state = "running"
    r = client.post("/api/camera/calibrate")

    assert r.status_code == 409


def test_apply_without_a_profile_is_a_404(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/api/camera/profile/apply").status_code == 404


# --- camera exclusivity is bidirectional ----------------------------------
#
# state.state == "running" already refused calibration while capture held the
# camera. Nothing refused the reverse: /api/capture/start and /api/cameras
# could open or probe the same device a calibration is mid-measurement on,
# because AppState had no flag saying calibration owns the camera.


def test_capture_start_is_refused_while_calibration_is_in_progress(tmp_path):
    """Real scenario: operator starts calibration in Admin, switches to Live
    view (the component unmounts but the server request keeps running),
    presses Start. Without this refusal the device gets opened twice."""
    client, state = _client(tmp_path)
    state.calibrating = True

    r = client.post("/api/capture/start")

    assert r.status_code == 409
    assert "calibrat" in r.json()["detail"].lower()
    assert state.state != "running"  # never touched source_factory/detector_factory


def test_calibrate_is_refused_while_another_calibration_is_in_progress(tmp_path):
    """The UI's `calibrating` flag is per-component, not per-device — a
    second concurrent request (two tabs, a stuck component) must still be
    refused server-side."""
    client, state = _client(tmp_path)
    state.calibrating = True

    r = client.post("/api/camera/calibrate")

    assert r.status_code == 409
    assert "already in progress" in r.json()["detail"].lower()


def test_calibrating_flag_is_set_during_and_cleared_after_a_successful_calibration(tmp_path):
    seen_during = {}

    def calibrator():
        seen_during["calibrating"] = state.calibrating
        return _profile()

    state = AppState(settings_path=str(tmp_path / "s.json"), db_path=":memory:",
                     calibrator=calibrator)
    client = TestClient(build_app(lambda: state))

    assert state.calibrating is False
    client.post("/api/camera/calibrate")

    assert seen_during["calibrating"] is True  # held for the duration of the request
    assert state.calibrating is False  # cleared once it completes


def test_calibrating_flag_is_cleared_even_when_the_calibrator_raises(tmp_path):
    """Set True before the calibrator runs, cleared in a finally — an
    exception (a real camera going away mid-measurement, say) must not
    strand the camera permanently exclusive."""
    def boom():
        raise RuntimeError("camera unplugged mid-calibration")

    state = AppState(settings_path=str(tmp_path / "s.json"), db_path=":memory:",
                     calibrator=boom)
    client = TestClient(build_app(lambda: state))

    with pytest.raises(RuntimeError):
        client.post("/api/camera/calibrate")

    assert state.calibrating is False


# --- default calibrator wiring (the device_name regression) --------------
#
# On a real StreamCam, /api/camera/calibrate used to return
# device_key=":1:1280x720" — the device name was never passed into
# calibrate() at all. device_key is the persistence key camera_profiles.py
# stores profiles under, so an empty name let two physically different
# cameras at the same index/resolution collide on one saved profile.


def test_default_calibrator_passes_a_nonempty_device_name(tmp_path, monkeypatch):
    """AppState.__post_init__ wires its own calibrator when none is injected.
    Patch the calibrate() function main.py calls (never a real device, and
    never the state.calibrator seam, which would just bypass the wiring
    under test) to capture what that default wiring actually passes it."""
    captured = {}

    def fake_calibrate(index, width, height, **kwargs):
        captured.update(kwargs)
        return _profile()

    monkeypatch.setattr("app.main.calibrate", fake_calibrate)

    state = AppState(
        settings_path=str(tmp_path / "s.json"),
        db_path=":memory:",
        camera_namer=lambda: ["Logitech StreamCam"],
    )
    assert state.calibrator is not None  # built by __post_init__, not injected
    state.calibrator()

    assert captured.get("device_name") == "Logitech StreamCam"
    # The exposure gate in derive_camera_settings needs the operator's
    # configured capture rate to be relative rather than an absolute floor;
    # the default settings' capture_fps is the source of truth for it.
    assert captured.get("target_fps") == state.settings.capture_fps


def test_resolve_camera_name_survives_a_failing_camera_namer(tmp_path):
    """list_device_names() shells out to PowerShell and is documented to
    never raise, but the fallback here must hold even if an injected namer
    (or a future implementation) breaks that contract — a naming failure
    must never turn into a failed calibration."""
    def boom():
        raise OSError("no powershell")

    state = AppState(settings_path=str(tmp_path / "s.json"), db_path=":memory:",
                      camera_namer=boom, calibrator=lambda: _profile())

    assert _resolve_camera_name(state) == f"Camera {state.settings.camera_index}"


def test_resolve_camera_name_falls_back_when_fewer_names_than_index(tmp_path):
    """Same convention as list_cameras: Windows naming fewer devices than
    opened degrades to 'Camera N', not an IndexError."""
    state = AppState(settings_path=str(tmp_path / "s.json"), db_path=":memory:",
                      camera_namer=lambda: [], calibrator=lambda: _profile())
    state.settings.camera_index = 1

    assert _resolve_camera_name(state) == "Camera 1"


def test_the_calibrate_response_carries_the_measured_evidence():
    """CameraProfileResponse(**asdict(profile)) silently drops fields the
    model does not declare, so a schema that lags the dataclass loses data
    with no error anywhere."""
    profile = CameraProfile(
        device_key="StreamCam:0:1280x720", backend="msmf", width=1280, height=720,
        fps_auto_exposure=30.0, fps_capped_exposure=29.0,
        controls=ControlSupport(exposure=True, autofocus=True),
        measured={"camera_exposure": {"value": -7.0, "metric": 129.0,
                                      "baseline": 23.0, "reached": True, "probes": 4}},
        sweep_version=1,
    )
    state = AppState(calibrator=lambda: profile, db_path=":memory:")

    with TestClient(build_app(lambda: state)) as client:
        body = client.post("/api/camera/calibrate").json()

    assert body["sweep_version"] == 1
    assert body["measured"]["camera_exposure"]["value"] == -7.0
    assert body["controls"]["autofocus"] is True
