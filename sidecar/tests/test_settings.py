from app.settings import Settings, resolve_device


def test_settings_defaults():
    s = Settings()
    assert s.active_model == "yolo11n.pt"
    assert s.capture_width == 1280
    assert s.capture_height == 720
    assert s.capture_fps == 60
    assert s.conf_threshold == 0.5
    assert s.imgsz == 640
    assert s.infer_frame_skip == 0
    assert s.device == "auto"
    assert s.preview_height == 720


def _fake_torch(monkeypatch, cuda_available: bool):
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_resolve_device_cpu_always_passthrough(monkeypatch):
    # "cpu" is honored even when a CUDA GPU is available.
    _fake_torch(monkeypatch, cuda_available=True)
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_and_auto_use_gpu_when_available(monkeypatch):
    _fake_torch(monkeypatch, cuda_available=True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") == "cuda"


def test_resolve_device_forced_cuda_falls_back_when_unavailable(monkeypatch):
    # A stale/forced "cuda" must NOT be returned when torch can't use it,
    # otherwise capture crashes at start on a CPU-only torch install.
    _fake_torch(monkeypatch, cuda_available=False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("auto") == "cpu"


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
    # A forced "cuda" with no torch at all must also degrade, not crash.
    assert resolve_device("cuda") == "cpu"
