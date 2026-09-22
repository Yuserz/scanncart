#!/usr/bin/env python
"""Keep the dataset tools inside an explicit resource budget.

The machine this runs on: 34 GB RAM, 12 logical cores, an 8.6 GB RTX 4060, and a
C: drive sitting at 86% full (about 68 GB free of 476). Everything here shares
that box with the Electron app, a browser and the sidecar, so a dataset pass
that just uses "everything available" is the reason an earlier audit felt like
it had taken the machine over.

The rule: this tool may use at most `USE_PERCENT` of each *shareable* resource
(CPU threads, RAM, VRAM), leaving the rest for whatever else is running. Disk is
handled differently and deliberately - see DISK below.

Import this BEFORE numpy/torch. OpenMP/MKL read their thread limits when the
native libs initialise, so setting them afterwards is a silent no-op.

    import resources          # first
    import numpy as np        # now
    resources.apply()         # torch/CUDA limits, once torch is importable

DISK: a percentage cap does not work here. "Leave 80% of a 476 GB drive free"
would mean keeping 381 GB free, and the drive is already 86% used, so no rule of
that shape can be satisfied - it would forbid the tool from writing anything.
Disk instead gets an absolute floor (DISK_RESERVE_GB): the tool refuses a write
that would drop free space below it. Override with --disk-reserve-gb.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

# Overridable so nobody has to edit code to loosen or tighten this.
USE_PERCENT = int(os.environ.get("SCANNCART_MAX_USE_PERCENT", "20"))
DISK_RESERVE_GB = float(os.environ.get("SCANNCART_DISK_RESERVE_GB", "20"))

# VRAM gets its own, looser number, and the reason is measured rather than
# assumed. A 20% cap on the 8.6 GB card is 1.7 GB, and yolo11m at 640 with batch
# 8 *reserves 1.67 GB* - essentially the entire budget - so at 20% the GPU is
# unusable, every forward OOMs, and the work silently falls back to CPU, which is
# slower and busier for every other app on the box. 35% (~3.0 GB) leaves two
# thirds of the card free and runs the same batch comfortably. CPU and RAM stay
# at 20%, because those are what other applications actually contend for.
VRAM_PERCENT = int(os.environ.get("SCANNCART_MAX_VRAM_PERCENT", "35"))

# The CUDA OOM message recommends this, and it is right: the default allocator
# fragments, and fragmentation is what turns a 0.67 GB working set into a
# failed 1.58 GB allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _psutil():
    try:
        import psutil  # already a sidecar dependency

        return psutil
    except ImportError:
        return None


def _cpu_threads(use_percent: int) -> int:
    logical = os.cpu_count() or 4
    # Round down and always leave at least one core idle, so the box stays
    # responsive even on a 2-core machine.
    return max(1, min(logical - 1, int(logical * use_percent / 100)))


# Set at import time, before the native libs load.
CPU_THREADS = _cpu_threads(USE_PERCENT)
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
# Ultralytics spawns per-worker dataloaders by default; that is where the RAM
# spikes come from on 3060x4080 inputs. Load in-process instead.
os.environ.setdefault("YOLO_VERBOSE", "false")


@dataclass
class Budget:
    """The caps actually in force for one run."""

    use_percent: int = USE_PERCENT
    vram_percent: int = VRAM_PERCENT
    disk_reserve_gb: float = DISK_RESERVE_GB
    cpu_threads: int = CPU_THREADS
    ram_total_gb: float = 0.0
    ram_cap_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_cap_gb: float = 0.0
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"resource budget: <= {self.use_percent}% of CPU/RAM, <= {self.vram_percent}% VRAM, "
            f"disk reserve {self.disk_reserve_gb:.0f} GB",
            f"  CPU   {self.cpu_threads} threads",
        ]
        if self.ram_total_gb:
            lines.append(f"  RAM   cap {self.ram_cap_gb:.1f} GB of {self.ram_total_gb:.1f} GB")
        if self.vram_total_gb:
            lines.append(f"  VRAM  cap {self.vram_cap_gb:.1f} GB of {self.vram_total_gb:.1f} GB")
        lines += [f"  note  {n}" for n in self.notes]
        return "\n".join(lines)

    # -- derived sizes -----------------------------------------------------

    def batch_size(self, imgsz: int, per_image_mb: float = 260.0, overhead_mb: float = 600.0) -> int:
        """Largest batch that fits the VRAM cap, and never more than 8.

        Deliberately conservative: measured, yolo11m at 640 OOMs even at batch 4
        inside a 1.7 GB cap (the CUDA context and fragmentation are not free),
        so the estimate is padded and any residual OOM halves the batch.
        """
        if not self.vram_cap_gb:
            return 4
        budget = self.vram_cap_gb * 1000.0 - overhead_mb
        if budget <= 0:
            return 1
        scale = (imgsz / 640.0) ** 2
        return max(1, min(8, int(budget / (per_image_mb * scale))))

    def check_disk(self, path: str, extra_bytes: int = 0) -> float:
        """Refuse to write if it would breach the reserve. Returns free GB."""
        free_gb = shutil.disk_usage(path).free / 1e9
        if extra_bytes and (free_gb - extra_bytes / 1e9) < self.disk_reserve_gb:
            raise SystemExit(
                f"refusing to write {extra_bytes / 1e9:.1f} GB: free space on {path} is "
                f"{free_gb:.1f} GB and the reserve is {self.disk_reserve_gb:.0f} GB. "
                f"Free space, lower --disk-reserve-gb, or skip this step."
            )
        return free_gb

    def check_ram(self, needed_gb: float) -> None:
        ps = _psutil()
        if ps is None:
            return
        vm = ps.virtual_memory()
        available_gb = vm.available / 1e9
        if needed_gb > available_gb:
            raise SystemExit(
                f"needs ~{needed_gb:.1f} GB RAM but only {available_gb:.1f} GB is available "
                f"({vm.percent:.0f}% in use). Close something, or lower the batch size."
            )


def measure(
    use_percent: int = USE_PERCENT,
    disk_reserve_gb: float = DISK_RESERVE_GB,
    vram_percent: int = VRAM_PERCENT,
) -> Budget:
    """Read the machine and work out the caps, without initialising torch."""
    b = Budget(use_percent=use_percent, vram_percent=vram_percent, disk_reserve_gb=disk_reserve_gb)

    ps = _psutil()
    if ps is None:
        b.notes.append("psutil unavailable - RAM/VRAM operating uncapped")
        return b
    vm = ps.virtual_memory()
    b.ram_total_gb = vm.total / 1e9
    b.ram_cap_gb = b.ram_total_gb * use_percent / 100
    if vm.percent > 80:
        b.notes.append(f"RAM already {vm.percent:.0f}% used before we start")
    return b


def apply(budget: Budget, device: str | None = None, hard_vram_cap: bool = False) -> Budget:
    """Apply torch/CUDA limits. Call once, after torch is importable.

    VRAM is bounded by *batch size*, not by `set_per_process_memory_fraction`, and
    that is a measured decision. The allocator cap looked like the honest way to
    keep the card free, but it fights ultralytics: with a 1.7 GB cap a 640px
    forward asked for a single 1.58 GiB block, and raising the cap to 3.0 GB made
    it ask for 3.12 GiB - the requested block scales with the cap, while the
    real working set for yolo11s at batch 8 is only 0.60 GB. A cap strict enough
    to matter therefore forces OOMs and drops the work onto the CPU, which is
    slower *and* busier for everything else. So the cap is opt-in, and the
    default is a batch size derived from the VRAM budget plus a measured peak in
    the report.
    """
    try:
        import torch
    except ImportError:
        budget.notes.append("torch unavailable - no GPU caps applied")
        return budget

    torch.set_num_threads(budget.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, budget.cpu_threads // 2))
    except RuntimeError:
        pass  # already set, or too late in process start; harmless

    if device and device != "cpu" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        budget.vram_total_gb = total / 1e9
        budget.vram_cap_gb = budget.vram_total_gb * budget.vram_percent / 100
        if not hard_vram_cap:
            budget.notes.append(
                f"VRAM bounded by batch size, not the allocator (measured need ~0.6 GB; "
                f"use --hard-vram-cap to force the {budget.vram_cap_gb:.1f} GB limit)"
            )
            return budget
        # Must happen before the CUDA context allocates, hence "apply early".
        try:
            torch.cuda.set_per_process_memory_fraction(budget.vram_percent / 100.0, 0)
            budget.notes.append(f"hard VRAM cap {budget.vram_cap_gb:.1f} GB enforced by the allocator")
        except Exception as exc:  # noqa: BLE001 - cap is best-effort, never fatal
            budget.notes.append(f"VRAM cap not applied ({type(exc).__name__})")
    elif device and device != "cpu":
        budget.notes.append("CUDA unavailable - falling back to CPU, expect ~17x slower")
    return budget


def resolve_device(requested: str = "auto") -> str:
    """'auto' -> '0' when CUDA is real, else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
