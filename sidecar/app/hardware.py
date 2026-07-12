import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Literal

import psutil


@dataclass
class HardwareInfo:
    cpu_count: int
    ram_gb: float
    cuda_available: bool
    accelerator: Literal["cuda", "integrated", "cpu"] = "cpu"
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None


def _list_display_adapters() -> list[str]:
    """Best-effort enumeration of display adapter names via Windows
    Win32_VideoController. Returns [] on any failure (non-Windows, no
    PowerShell, timeout) — never raises."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def probe_hardware(
    adapter_lister: Callable[[], list[str]] = _list_display_adapters,
) -> HardwareInfo:
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
                accelerator="cuda",
                gpu_name=torch.cuda.get_device_name(0),
                gpu_vram_gb=props.total_memory / 1e9,
            )
    except ImportError:
        pass

    # No usable CUDA GPU: classify integrated vs cpu-only from display adapters.
    # A "real" GPU is any non-empty adapter name not starting with "Microsoft"
    # (excludes the software adapters — real GPUs are branded NVIDIA/AMD/Intel).
    try:
        adapters = adapter_lister()
    except Exception:
        adapters = []
    integrated = next(
        (a for a in adapters if a and not a.lower().startswith("microsoft")),
        None,
    )
    if integrated is not None:
        return HardwareInfo(
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            cuda_available=False,
            accelerator="integrated",
            gpu_name=integrated,
        )
    return HardwareInfo(
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        cuda_available=False,
        accelerator="cpu",
    )
