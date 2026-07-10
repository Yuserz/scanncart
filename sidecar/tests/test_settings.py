from app.settings import Settings, resolve_device


def test_settings_defaults():
    s = Settings()
    assert s.active_model == "yolo11n.pt"
    assert s.capture_width == 1280
    assert s.capture_height == 720
    assert s.capture_fps == 60
    assert s.conf_threshold == 0.5
    assert s.infer_frame_skip == 0
    assert s.device == "auto"
    assert s.preview_height == 720


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    # Simulate torch missing -> must fall back to cpu, never crash.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_device("auto") == "cpu"
