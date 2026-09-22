import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.presets import PRESETS
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


def _fake_hardware():
    from app.hardware import HardwareInfo

    return HardwareInfo(cpu_count=8, ram_gb=16.0, cuda_available=False, accelerator="cpu")


def _make_client(tmp_path, **kwargs):
    state = AppState(
        settings=kwargs.pop("settings", Settings()),
        settings_path=str(tmp_path / "settings.json"),
        source_factory=lambda settings: _StubSource(),
        detector_factory=lambda settings, device: _StubDetector(),
        db_path=":memory:",
        hardware_prober=kwargs.pop("hardware_prober", _fake_hardware),
        **kwargs,
    )
    return TestClient(build_app(lambda: state)), state


def test_get_settings_returns_current_values_and_field_classification(tmp_path):
    client, _ = _make_client(tmp_path)
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["active_model"] == Settings().active_model
    assert "infer_frame_skip" in body["hot_reloadable_fields"]
    assert "active_model" in body["restart_required_fields"]
    assert isinstance(body["warnings"], list)


def test_the_settings_route_honours_a_recorded_resize_requirement(tmp_path, monkeypatch):
    """The record reaches the warning check too, or the app would flag the configuration it
    just told the operator to use. `requirement_for` is patched rather than written to
    `sidecar/models/`, so this stays off the real directory the app ships with."""
    import app.main as main

    monkeypatch.setattr(main, "requirement_for", lambda name: "stretch")
    client, _ = _make_client(
        tmp_path,
        settings=Settings(active_model="models/scanncart-grocery-v2.pt", resize_mode="stretch"),
    )
    warnings = client.get("/api/settings").json()["warnings"]
    assert not any("letterbox-trained" in w for w in warnings)


def test_the_settings_route_reports_the_geometry_the_detector_will_use(tmp_path, monkeypatch):
    """`resize_mode: auto` is the default and the documented answer, so the setting alone does not
    say what geometry runs. The response carries the resolved mode — literally the value
    `_default_detector_factory` passes to the detector — because the Live view's readout has no way
    to derive it: `auto` depends on the weights' record, and on the file format for weights nobody
    recorded anything about."""
    import app.main as main

    monkeypatch.setattr(main, "requirement_for", lambda name: "stretch")
    client, _ = _make_client(
        tmp_path,
        settings=Settings(active_model="models/scanncart-grocery-v2.pt", resize_mode="auto"),
    )
    assert client.get("/api/settings").json()["resize_mode_resolved"] == "stretch"


def test_the_reported_geometry_is_the_unrecorded_heuristic_when_nothing_was_recorded(tmp_path):
    # The other half of the same field: for weights with no record, `auto` is still the format
    # heuristic, and the readout has to print what that actually falls back to rather than nothing.
    client, _ = _make_client(tmp_path, settings=Settings(active_model="yolo11n.pt"))
    assert client.get("/api/settings").json()["resize_mode_resolved"] == "letterbox"


def test_an_explicit_mode_is_reported_as_itself(tmp_path, monkeypatch):
    # The panel's warning is about a mismatch, so the readout must not quietly report what the
    # operator "should" have chosen: an explicit letterbox over a record saying stretch is reported
    # as letterbox, which is the geometry that will actually run.
    import app.main as main

    monkeypatch.setattr(main, "requirement_for", lambda name: "stretch")
    client, _ = _make_client(
        tmp_path,
        settings=Settings(active_model="models/scanncart-grocery-v2.pt", resize_mode="letterbox"),
    )
    assert client.get("/api/settings").json()["resize_mode_resolved"] == "letterbox"


def test_an_unrecorded_weight_comes_back_with_the_geometry_to_record(tmp_path):
    """The panel's button needs three things and gets all three here: which weight, which mode to
    write, and the prose to render beside it. The mode is `auto`'s answer for these weights, so
    recording it cannot change what the detector does — it converts the fallback into a fact.

    Both sentences are on the response because both views render the entry, and a view without the
    button still has to say what to do about it."""
    client, _ = _make_client(
        tmp_path,
        settings=Settings(active_model="models/hand-copied.pt", resize_mode="auto"),
    )
    entry = client.get("/api/settings").json()["unrecorded_resize_mode"]
    assert entry["model"] == "models/hand-copied.pt"
    assert entry["resize_mode"] == "letterbox"
    assert "no record of the geometry" in entry["warning"]
    assert "--install" in entry["remedy"]
    # And it is not *also* in the flat warning list, or the panel would render the same situation
    # twice — once with the remedy beside it and once without.
    assert not any(
        "no record of the geometry" in w for w in client.get("/api/settings").json()["warnings"]
    )


def test_a_recorded_weight_comes_back_with_nothing_to_record(tmp_path, monkeypatch):
    """The entry's absence is the whole trigger for the button, so it has to clear the moment the
    requirement exists — otherwise the panel offers to record a fact it already has."""
    import app.main as main

    monkeypatch.setattr(main, "requirement_for", lambda name: "letterbox")
    client, _ = _make_client(
        tmp_path,
        settings=Settings(active_model="models/hand-copied.pt", resize_mode="auto"),
    )
    assert client.get("/api/settings").json()["unrecorded_resize_mode"] is None


@pytest.mark.parametrize("backend", ["local_api", "cloud_api"])
def test_a_remote_backend_reports_no_local_geometry(tmp_path, backend):
    """Only the native branch resizes with these weights. A remote backend hands the frame to a
    workflow that holds its own model and resizes server-side, so `active_model` and `resize_mode`
    are out of the inference path — and reporting a geometry there would be a claim about a resize
    that never happens. `None` is what the Live view reads as "no local weights to describe"."""
    client, _ = _make_client(
        tmp_path,
        settings=Settings(active_model="models/scanncart-grocery-v2.pt", detector_backend=backend),
    )
    assert client.get("/api/settings").json()["resize_mode_resolved"] is None


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
    assert state.settings.active_model == Settings().active_model  # unchanged
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
    assert state.settings.active_model == Settings().active_model


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


def test_hardware_probed_once_across_endpoints(tmp_path):
    from app.hardware import HardwareInfo

    calls = {"n": 0}

    def counting_prober():
        calls["n"] += 1
        return HardwareInfo(cpu_count=8, ram_gb=16.0, cuda_available=False, accelerator="cpu")

    client, _ = _make_client(tmp_path, hardware_prober=counting_prober)
    client.get("/api/system-info")
    client.get("/api/presets")
    # Memoized: both endpoints share one probe rather than re-running it.
    assert calls["n"] == 1


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
    # Start from a stock model: the default is now a custom one, which presets
    # deliberately leave alone (see test_apply_preset_keeps_a_custom_model).
    state.settings.active_model = "yolo11m.pt"

    r = client.post("/api/settings/preset", json={"name": "low_end"})
    assert r.status_code == 200
    # The preset's own value, not the default: presets pick a model size.
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


def test_patch_experimental_yolo26_model_is_accepted(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"active_model": "yolo26n.pt"})
    assert r.status_code == 200
    assert state.settings.active_model == "yolo26n.pt"


def test_apply_preset_keeps_a_custom_model(tmp_path):
    """Presets pick a stock model size, which would silently swap a custom
    model for generic COCO weights — the whole point of the app."""
    client, state = _make_client(tmp_path)
    state.settings.active_model = "models/scanncart-grocery.onnx"

    r = client.post("/api/settings/preset", json={"name": "low_end"})

    assert r.status_code == 200
    assert state.settings.active_model == "models/scanncart-grocery.onnx"
    # ...while still tuning everything else the preset carries.
    assert state.settings.imgsz == PRESETS["low_end"].settings.get(
        "imgsz", state.settings.imgsz
    )


def test_apply_preset_still_sets_the_model_for_a_stock_one(tmp_path):
    client, state = _make_client(tmp_path)
    state.settings.active_model = "yolo11m.pt"

    client.post("/api/settings/preset", json={"name": "low_end"})

    assert state.settings.active_model == "yolo11n.pt"


def test_patch_class_allowlist_hot_reloadable(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"class_allowlist": ["bottle", "cup"]})
    assert r.status_code == 200
    assert r.json()["class_allowlist"] == ["bottle", "cup"]
    assert state.settings.class_allowlist == ["bottle", "cup"]


def test_patch_invalid_class_allowlist_is_rejected(tmp_path):
    client, state = _make_client(tmp_path)
    r = client.patch("/api/settings", json={"class_allowlist": ["bottle", ""]})
    assert r.status_code == 422
    assert state.settings.class_allowlist == []  # unchanged
