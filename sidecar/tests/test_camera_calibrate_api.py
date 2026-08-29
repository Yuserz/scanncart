from fastapi.testclient import TestClient

from app.camera_caps import CameraProfile, ControlSupport
from app.main import AppState, build_app


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
