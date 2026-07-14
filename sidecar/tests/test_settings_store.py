import json

from app.settings import Settings
from app.settings_store import compute_warnings, load_settings, save_settings


def test_load_settings_missing_file_returns_defaults(tmp_path):
    settings = load_settings(str(tmp_path / "missing.json"))
    assert settings == Settings()


def test_load_settings_corrupt_json_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json", encoding="utf-8")
    settings = load_settings(str(path))
    assert settings == Settings()


def test_load_settings_overlays_valid_fields(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"active_model": "yolo11s.pt", "conf_threshold": 0.7}), encoding="utf-8")
    settings = load_settings(str(path))
    assert settings.active_model == "yolo11s.pt"
    assert settings.conf_threshold == 0.7
    assert settings.capture_width == 1280  # untouched fields keep defaults


def test_load_settings_falls_back_per_field_on_invalid_value(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"conf_threshold": "oops", "active_model": "yolo11m.pt"}), encoding="utf-8"
    )
    settings = load_settings(str(path))
    assert settings.conf_threshold == 0.5  # invalid -> default, not a crash
    assert settings.active_model == "yolo11m.pt"  # valid field still applied


def test_load_settings_ignores_unknown_keys(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"not_a_real_field": 123}), encoding="utf-8")
    settings = load_settings(str(path))
    assert settings == Settings()


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "settings.json"
    original = Settings(active_model="yolo11l.pt", device="cuda", infer_frame_skip=2)
    save_settings(original, str(path))
    loaded = load_settings(str(path))
    assert loaded == original


def test_save_settings_creates_parent_dir(tmp_path):
    path = tmp_path / "a" / "b" / "settings.json"
    save_settings(Settings(), str(path))
    assert path.exists()


def test_compute_warnings_running_locks_restart_fields():
    warnings = compute_warnings(Settings(), "running")
    assert any("stopping capture" in w for w in warnings)


def test_compute_warnings_idle_no_running_warning():
    warnings = compute_warnings(Settings(), "idle")
    assert not any("stopping capture" in w for w in warnings)


def test_compute_warnings_uncommon_resolution():
    warnings = compute_warnings(Settings(capture_width=800, capture_height=600), "idle")
    assert any("resolution" in w for w in warnings)


def test_compute_warnings_common_resolution_no_warning():
    warnings = compute_warnings(Settings(capture_width=1280, capture_height=720), "idle")
    assert not any("resolution" in w for w in warnings)


def test_compute_warnings_frame_skip_vs_expiry():
    settings = Settings(infer_frame_skip=10, capture_fps=15, track_expiry_s=1.0)
    warnings = compute_warnings(settings, "idle")
    assert any("infer_frame_skip" in w for w in warnings)


def test_compute_warnings_low_frame_skip_no_expiry_warning():
    settings = Settings(infer_frame_skip=0, capture_fps=60, track_expiry_s=1.5)
    warnings = compute_warnings(settings, "idle")
    assert not any("infer_frame_skip" in w for w in warnings)


def test_compute_warnings_experimental_model():
    warnings = compute_warnings(Settings(active_model="yolo26n.pt"), "idle")
    assert any("experimental" in w for w in warnings)


def test_compute_warnings_supported_model_no_experimental_warning():
    warnings = compute_warnings(Settings(), "idle")
    assert not any("experimental" in w for w in warnings)


def test_save_then_load_round_trips_experimental_model(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(Settings(active_model="yolo26m.pt"), str(path))
    assert load_settings(str(path)).active_model == "yolo26m.pt"
