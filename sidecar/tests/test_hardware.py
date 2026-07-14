from app.hardware import probe_hardware


def test_probe_hardware_reports_positive_cpu_and_ram():
    hw = probe_hardware()
    assert hw.cpu_count > 0
    assert hw.ram_gb > 0


def test_ram_gb_uses_binary_gib_not_decimal_gb(monkeypatch):
    # A "16GB" RAM stick is physically 16 * 1024**3 bytes. Reporting it via
    # decimal (bytes / 1e9) yields ~17.18 -- the "16GB" a user paid for
    # shouldn't display as 17GB. OS tools (Task Manager, Settings) report
    # binary GiB, so we must match: bytes / 1024**3 == 16.0 exactly.
    import types

    import app.hardware as hardware_module

    fake_total = 16 * 1024**3
    monkeypatch.setattr(
        hardware_module.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total=fake_total),
    )
    hw = probe_hardware(adapter_lister=lambda: [])
    assert hw.ram_gb == 16.0


def test_probe_hardware_falls_back_when_torch_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    hw = probe_hardware(adapter_lister=lambda: [])
    assert hw.cuda_available is False
    assert hw.accelerator == "cpu"
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
    # Same binary-GiB requirement as RAM: an 8GB card is 8 * 1024**3 bytes,
    # and should report as 8.0, not ~8.59 (decimal GB).
    assert hw.gpu_vram_gb == 8.0
    assert hw.accelerator == "cuda"


def _fake_torch_no_cuda(monkeypatch):
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_accelerator_cuda_when_gpu_available(monkeypatch):
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "Fake GPU 3000",
        get_device_properties=lambda idx: types.SimpleNamespace(total_memory=8 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    hw = probe_hardware(adapter_lister=lambda: ["NVIDIA GeForce RTX 4060"])
    assert hw.accelerator == "cuda"
    assert hw.cuda_available is True


def test_accelerator_integrated_for_non_cuda_adapter(monkeypatch):
    _fake_torch_no_cuda(monkeypatch)
    hw = probe_hardware(adapter_lister=lambda: ["AMD Radeon(TM) Graphics"])
    assert hw.accelerator == "integrated"
    assert hw.cuda_available is False
    assert hw.gpu_name == "AMD Radeon(TM) Graphics"
    assert hw.gpu_vram_gb is None


def test_accelerator_cpu_when_only_microsoft_basic_adapter(monkeypatch):
    _fake_torch_no_cuda(monkeypatch)
    hw = probe_hardware(adapter_lister=lambda: ["Microsoft Basic Display Adapter"])
    assert hw.accelerator == "cpu"
    assert hw.gpu_name is None


def test_accelerator_cpu_when_no_adapters(monkeypatch):
    _fake_torch_no_cuda(monkeypatch)
    hw = probe_hardware(adapter_lister=lambda: [])
    assert hw.accelerator == "cpu"


def test_accelerator_cpu_when_lister_raises(monkeypatch):
    _fake_torch_no_cuda(monkeypatch)

    def boom():
        raise RuntimeError("wmi query failed")

    hw = probe_hardware(adapter_lister=boom)
    assert hw.accelerator == "cpu"
