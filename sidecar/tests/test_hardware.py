from app.hardware import probe_hardware


def test_probe_hardware_reports_positive_cpu_and_ram():
    hw = probe_hardware()
    assert hw.cpu_count > 0
    assert hw.ram_gb > 0


def test_probe_hardware_falls_back_when_torch_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    hw = probe_hardware()
    assert hw.cuda_available is False
    assert hw.gpu_name is None
    assert hw.gpu_vram_gb is None


def test_probe_hardware_reports_gpu_when_cuda_available(monkeypatch):
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "Fake GPU 3000",
        get_device_properties=lambda idx: types.SimpleNamespace(total_memory=8 * 1024**3),
    )
    fake_torch.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    hw = probe_hardware()
    assert hw.cuda_available is True
    assert hw.gpu_name == "Fake GPU 3000"
    assert hw.gpu_vram_gb is not None and hw.gpu_vram_gb > 0
