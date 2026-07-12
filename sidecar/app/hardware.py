from dataclasses import dataclass

import psutil


@dataclass
class HardwareInfo:
    cpu_count: int
    ram_gb: float
    cuda_available: bool
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None


def probe_hardware() -> HardwareInfo:
    cpu_count = psutil.cpu_count(logical=True) or 1
    ram_gb = psutil.virtual_memory().total / 1e9
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return HardwareInfo(
                cpu_count=cpu_count,
                ram_gb=ram_gb,
                cuda_available=True,
                gpu_name=torch.cuda.get_device_name(0),
                gpu_vram_gb=props.total_memory / 1e9,
            )
    except ImportError:
        pass
    return HardwareInfo(cpu_count=cpu_count, ram_gb=ram_gb, cuda_available=False)
