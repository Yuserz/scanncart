import json

import pytest

from app.settings import Settings
from app.settings_store import (
    RESTART_REQUIRED_FIELDS,
    _valid_field,
    compute_warnings,
    load_settings,
    resolve_resize_mode,
    save_settings,
)


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


def test_imgsz_is_restart_required():
    assert "imgsz" in RESTART_REQUIRED_FIELDS


def test_valid_field_imgsz_accepts_stride_multiples():
    assert _valid_field("imgsz", 640)
    assert _valid_field("imgsz", 960)


def test_valid_field_imgsz_rejects_non_stride_and_out_of_range():
    assert not _valid_field("imgsz", 641)  # not a multiple of 32
    assert not _valid_field("imgsz", 160)  # below the 320 floor
    assert not _valid_field("imgsz", 2048)  # above the 1920 ceiling
    assert not _valid_field("imgsz", 640.0)  # must be an int


def test_compute_warnings_high_imgsz():
    warnings = compute_warnings(Settings(imgsz=1280), "idle")
    assert any("imgsz above 960" in w for w in warnings)


def test_compute_warnings_default_imgsz_no_warning():
    warnings = compute_warnings(Settings(imgsz=640), "idle")
    assert not any("imgsz" in w for w in warnings)


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


# --- resolve_resize_mode -------------------------------------------------


def test_auto_resolves_custom_onnx_to_stretch():
    """A Roboflow export records "Stretch to" preprocessing."""
    assert resolve_resize_mode("auto", "models/scanncart-grocery.onnx") == "stretch"


def test_auto_resolves_custom_pt_to_letterbox():
    """A locally trained .pt is letterbox-trained; stretching it would shrink
    objects below their training scale. Design doc 2026-09-04 §C."""
    assert resolve_resize_mode("auto", "models/scanncart-grocery.pt") == "letterbox"


def test_auto_resolves_stock_weights_to_letterbox():
    assert resolve_resize_mode("auto", "yolo11n.pt") == "letterbox"


def test_explicit_modes_win_over_auto():
    assert resolve_resize_mode("letterbox", "models/scanncart-grocery.onnx") == "letterbox"
    assert resolve_resize_mode("stretch", "yolo11n.pt") == "stretch"


# --- native onnx CPU-fallback warning ------------------------------------


def test_onnx_on_cuda_without_the_gpu_runtime_warns(monkeypatch):
    """device resolves via torch, but a .onnx runs through onnxruntime; with
    the CPU wheel installed (the requirements.txt default) inference silently
    falls back to CPU with only an ultralytics log line."""
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: False)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.onnx",
                 device="cuda"),
        "idle",
    )
    assert any("onnxruntime-gpu" in w for w in warnings)


def test_onnx_on_cuda_with_the_gpu_runtime_does_not_warn(monkeypatch):
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: True)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.onnx",
                 device="cuda"),
        "idle",
    )
    assert not any("onnxruntime-gpu" in w for w in warnings)


def test_onnx_on_cpu_never_warns(monkeypatch):
    """CPU is a deliberate choice, not a fallback."""
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: False)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.onnx",
                 device="cpu"),
        "idle",
    )
    assert not any("onnxruntime-gpu" in w for w in warnings)


def test_a_pt_model_never_triggers_the_onnx_warning(monkeypatch):
    """A .pt runs on torch, whose CUDA support is independent of onnxruntime."""
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: False)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.pt",
                 device="cuda"),
        "idle",
    )
    assert not any("onnxruntime-gpu" in w for w in warnings)


# --- custom .pt + stretch warning ----------------------------------------


def test_custom_pt_with_explicit_stretch_warns():
    warnings = compute_warnings(
        Settings(active_model="models/scanncart-grocery.pt", resize_mode="stretch"), "idle"
    )
    assert any("letterbox-trained" in w for w in warnings)


def test_custom_pt_with_auto_does_not_warn():
    warnings = compute_warnings(
        Settings(active_model="models/scanncart-grocery.pt", resize_mode="auto"), "idle"
    )
    assert not any("letterbox-trained" in w for w in warnings)


def test_custom_onnx_with_stretch_does_not_warn():
    """Stretch is exactly right for a Roboflow export."""
    warnings = compute_warnings(
        Settings(active_model="models/scanncart-grocery.onnx", resize_mode="stretch"),
        "idle",
    )
    assert not any("letterbox-trained" in w for w in warnings)


def test_stock_weights_never_warn_about_stretch():
    warnings = compute_warnings(
        Settings(active_model="yolo11n.pt", resize_mode="stretch"), "idle"
    )
    assert not any("letterbox-trained" in w for w in warnings)
