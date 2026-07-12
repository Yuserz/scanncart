import numpy as np
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.schemas import Detection
from app.settings import Settings


class _StubSource:
    width, height, fps = 128, 96, 30.0

    def open(self):
        return True

    def latest(self):
        return (1, np.full((96, 128, 3), 50, dtype=np.uint8))

    def read(self):
        return np.full((96, 128, 3), 50, dtype=np.uint8)

    def release(self):
        pass


class _StubDetector:
    names = {0: "banana"}

    def infer(self, frame):
        return [Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]


def _make_client(tmp_path, **kwargs):
    state = AppState(
        settings=kwargs.pop("settings", Settings()),
        settings_path=str(tmp_path / "settings.json"),
        source_factory=lambda settings: _StubSource(),
        detector_factory=lambda settings, device: _StubDetector(),
        db_path=":memory:",
        **kwargs,
    )
    return TestClient(build_app(lambda: state)), state


def test_get_settings_returns_current_values_and_field_classification(tmp_path):
    client, _ = _make_client(tmp_path)
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["active_model"] == "yolo11n.pt"
    assert "infer_frame_skip" in body["hot_reloadable_fields"]
    assert "active_model" in body["restart_required_fields"]
    assert isinstance(body["warnings"], list)


def test_patch_hot_reloadable_field_while_idle(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"infer_frame_skip": 3})
    assert r.status_code == 200
    assert r.json()["infer_frame_skip"] == 3
    assert state.settings.infer_frame_skip == 3


def test_patch_hot_reloadable_field_while_running_succeeds(tmp_path):
    client, state = _make_client(tmp_path)
    client.post("/api/capture/start")
    r = client.patch("/api/settings", json={"infer_frame_skip": 4})
    assert r.status_code == 200
    assert state.settings.infer_frame_skip == 4
    client.post("/api/capture/stop")


def test_patch_restart_required_field_while_running_is_rejected(tmp_path):
    client, state = _make_client(tmp_path)
    client.post("/api/capture/start")
    r = client.patch("/api/settings", json={"active_model": "yolo11s.pt"})
    assert r.status_code == 409
    assert state.settings.active_model == "yolo11n.pt"  # unchanged
    client.post("/api/capture/stop")


def test_patch_restart_required_field_while_idle_succeeds(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"active_model": "yolo11s.pt"})
    assert r.status_code == 200
    assert state.settings.active_model == "yolo11s.pt"


def test_patch_out_of_range_value_is_rejected(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"conf_threshold": 5.0})
    assert r.status_code == 422
    assert state.settings.conf_threshold == 0.5  # unchanged


def test_patch_unknown_model_is_rejected(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"active_model": "not_a_real_model.pt"})
    assert r.status_code == 422
    assert state.settings.active_model == "yolo11n.pt"


def test_patch_device_reresolves_app_state_device(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"device": "cpu"})
    assert r.status_code == 200
    assert state.device == "cpu"


def test_patch_persists_to_disk_and_survives_new_app_state(tmp_path):
    client, state = _make_client(tmp_path)
    client.patch("/api/settings", json={"infer_frame_skip": 2})

    reloaded = AppState(settings_path=str(tmp_path / "settings.json"), db_path=":memory:")
    assert reloaded.settings.infer_frame_skip == 2


def test_system_info_reports_hardware_and_recommendation(tmp_path):
    client, _ = _make_client(tmp_path)
    r = client.get("/api/system-info")
    assert r.status_code == 200
    body = r.json()
    assert body["cpu_count"] > 0
    assert body["ram_gb"] > 0
    assert body["recommended_preset"] in ("low_end", "mid_range", "high_end")


def test_presets_lists_three_and_a_recommendation(tmp_path):
    client, _ = _make_client(tmp_path)
    r = client.get("/api/presets")
    assert r.status_code == 200
    body = r.json()
    names = {p["name"] for p in body["presets"]}
    assert names == {"low_end", "mid_range", "high_end"}
    assert body["recommended"] in names


def test_apply_preset_while_idle_applies_settings(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.post("/api/settings/preset", json={"name": "low_end"})
    assert r.status_code == 200
    assert state.settings.active_model == "yolo11n.pt"
    assert state.settings.device == "cpu"
    assert state.settings.capture_width == 640


def test_apply_preset_while_running_is_rejected(tmp_path):
    client, state = _make_client(tmp_path)
    client.post("/api/capture/start")
    r = client.post("/api/settings/preset", json={"name": "high_end"})
    assert r.status_code == 409
    client.post("/api/capture/stop")


def test_apply_unknown_preset_returns_404(tmp_path):
    client, _ = _make_client(tmp_path)
    r = client.post("/api/settings/preset", json={"name": "nonexistent"})
    assert r.status_code == 404
