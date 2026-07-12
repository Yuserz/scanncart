import pytest

from app.hardware import HardwareInfo
from app.presets import PRESETS, recommend_preset
from app.settings_store import RESTART_REQUIRED_FIELDS, HOT_RELOADABLE_FIELDS, _valid_field


@pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
def test_preset_settings_pass_field_validation(preset):
    for key, value in preset.settings.items():
        assert _valid_field(key, value), f"{preset.name}.{key}={value!r} fails validation"


@pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
def test_preset_never_touches_camera_index(preset):
    assert "camera_index" not in preset.settings


@pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
def test_preset_only_touches_known_fields(preset):
    known = RESTART_REQUIRED_FIELDS | HOT_RELOADABLE_FIELDS
    assert set(preset.settings) <= known


def test_recommend_preset_high_end_for_strong_gpu():
    hw = HardwareInfo(cpu_count=4, ram_gb=8, cuda_available=True, gpu_name="X", gpu_vram_gb=6.0)
    assert recommend_preset(hw) == "high_end"


def test_recommend_preset_ignores_weak_gpu():
    hw = HardwareInfo(cpu_count=4, ram_gb=4, cuda_available=True, gpu_name="X", gpu_vram_gb=2.0)
    assert recommend_preset(hw) != "high_end"


def test_recommend_preset_mid_range_for_strong_cpu_no_gpu():
    hw = HardwareInfo(cpu_count=8, ram_gb=16, cuda_available=False)
    assert recommend_preset(hw) == "mid_range"


def test_recommend_preset_low_end_fallback():
    hw = HardwareInfo(cpu_count=2, ram_gb=4, cuda_available=False)
    assert recommend_preset(hw) == "low_end"
