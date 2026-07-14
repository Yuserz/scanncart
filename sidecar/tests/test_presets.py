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


def test_recommend_preset_mid_range_for_realistic_8gb_machine():
    # hardware.py reports ram_gb in binary GiB. A genuine "8GB" machine's
    # OS-visible RAM is commonly a bit under 8 GiB after firmware/iGPU
    # reservation (e.g. 7.6 GiB) -- it must still classify as mid_range,
    # not get bumped down to low_end just because it isn't exactly >= 8.
    hw = HardwareInfo(cpu_count=6, ram_gb=7.6, cuda_available=False)
    assert recommend_preset(hw) == "mid_range"


def test_recommend_preset_low_end_for_genuinely_weak_ram():
    # Guards against overcorrecting: a machine well below the "8GB class"
    # (not just a little under due to reservation) must still fall back.
    hw = HardwareInfo(cpu_count=6, ram_gb=6.0, cuda_available=False)
    assert recommend_preset(hw) == "low_end"


def test_recommend_preset_high_end_for_realistic_4gb_gpu():
    # Same reasoning for VRAM: a "4GB" GPU's driver/OS-visible VRAM is
    # commonly a bit under 4 GiB (e.g. 3.6 GiB) after reservation.
    hw = HardwareInfo(cpu_count=4, ram_gb=8, cuda_available=True, gpu_name="X", gpu_vram_gb=3.6)
    assert recommend_preset(hw) == "high_end"


def test_recommend_preset_ignores_genuinely_weak_gpu():
    # Guards against overcorrecting: a GPU well below the "4GB class" must
    # still be ignored for the high_end recommendation.
    hw = HardwareInfo(cpu_count=4, ram_gb=8, cuda_available=True, gpu_name="X", gpu_vram_gb=3.0)
    assert recommend_preset(hw) != "high_end"
