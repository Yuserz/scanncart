# SCANnCART — Hardware-class detection + CPU/GPU choice — Design

**Date:** 2026-07-12
**Status:** Approved (pending spec review)
**Depends on:** existing `hardware.py`, `settings.py`, `settings_store.py`, `schemas.py` (sidecar); `AdminPanel.tsx`, `settingsFields.ts`, `api.ts` (desktop).

## Problem

The sidecar only detects NVIDIA/CUDA GPUs (via `torch.cuda`). A machine with an
APU or integrated GPU (AMD/Intel) reports "No GPU detected," indistinguishable
from a truly GPU-less machine. Users also want an explicit, clearly-labelled
CPU-vs-GPU choice that **defaults to GPU** for performance.

## Key constraint

Inference acceleration in this app goes through **CUDA, which is NVIDIA-only**.
An APU / integrated GPU can be *detected and labelled* but **cannot** run CUDA
inference without a different backend (DirectML/OpenVINO/ROCm), which is out of
scope. Therefore "GPU" in the UI means "the NVIDIA CUDA GPU"; on an APU/CPU-only
machine only CPU is offered. This design adds detection + labelling + a
CPU/GPU toggle; it does **not** add a new inference backend.

## Goals

- Detect and report three hardware classes: **`cuda`** (CPU + usable NVIDIA GPU),
  **`integrated`** (APU / integrated GPU, no CUDA), **`cpu`** (no GPU).
- Give the user a clear **GPU (recommended) / CPU only** choice, defaulting to
  GPU when a CUDA GPU is present, and crash-safe on any machine.
- Keep sidecar tests hardware-free (fakes/injection), per existing conventions.

## Non-goals

- Accelerating inference on non-NVIDIA GPUs (APU/iGPU) — explicitly out of scope.
- Changing capture/model/logging behaviour.
- Precise AMD-integrated-vs-discrete disambiguation by adapter name (heuristic is
  acceptable; see Detection).

## Design

### 1. Backend detection — `sidecar/app/hardware.py`

Add adapter enumeration so the three classes can be told apart, and a new
classification field.

- **New field** on `HardwareInfo`: `accelerator: Literal["cuda", "integrated", "cpu"]`.
  Existing fields (`cpu_count`, `ram_gb`, `cuda_available`, `gpu_name`,
  `gpu_vram_gb`) are unchanged and stay populated as today.
- **Adapter enumeration** via a module-level default helper
  `_list_display_adapters() -> list[str]` that queries Windows
  `Win32_VideoController` (PowerShell `Get-CimInstance Win32_VideoController`,
  parse the `Name` column). Failures (non-Windows, PowerShell missing, timeout)
  return `[]` and never raise.
- **`probe_hardware(adapter_lister: Callable[[], list[str]] = _list_display_adapters)`**
  — the lister is injected so tests supply fakes (no real hardware), matching the
  `source_factory`/`detector_factory` convention.
- **Classification:**
  - `cuda_available` = `torch.cuda.is_available()` (unchanged; wrapped in the
    existing `try/except ImportError`).
  - If `cuda_available` → `accelerator = "cuda"`; `gpu_name`/`gpu_vram_gb` from
    torch as today.
  - Else if any enumerated adapter is a "real" GPU — i.e. a non-empty name that
    does **not** start with `"Microsoft"` (case-insensitive; excludes the
    software adapters "Microsoft Basic Display Adapter", "Microsoft Remote
    Display Adapter", "Microsoft Basic Render Driver" — real GPUs are branded
    NVIDIA/AMD/Intel, never Microsoft) — → `accelerator = "integrated"`;
    `gpu_name` = that adapter name; `gpu_vram_gb` stays `None` (VRAM not probed
    for non-CUDA adapters).
  - Else → `accelerator = "cpu"`; `gpu_name = None`.

### 2. Device default semantics — `sidecar/app/settings.py`

No functional change. `resolve_device("auto")` already returns `cuda` when
available else `cpu`, and the default `device = "auto"` remains. `"auto"` is the
"GPU-when-present, safe fallback" behaviour: on an `integrated`/`cpu` machine it
resolves to `cpu` with no error. `ALLOWED_DEVICES = {"auto", "cpu", "cuda"}` is
unchanged; `"cuda"` remains a valid stored value even though the UI no longer
offers it as a distinct choice.

### 3. Schemas — `sidecar/app/schemas.py`

Add `accelerator: Literal["cuda", "integrated", "cpu"]` to `HardwareInfo`
(inherited by `SystemInfoResponse`). No other schema changes.

### 4. UI — desktop

- **`api.ts`** (hand-synced contract): add `accelerator: 'cuda' | 'integrated' | 'cpu'`
  to the `SystemInfo` type.
- **`AdminPanel.tsx`**:
  - Hardware label ("This machine") is driven by `accelerator`:
    - `cuda` → `"<gpu_name> (<vram> GB VRAM) — GPU acceleration available"`
    - `integrated` → `"Integrated graphics: <gpu_name> (APU) — no CUDA acceleration, runs on CPU"`
    - `cpu` → `"No GPU detected — CPU only"`
  - The **Device** field becomes a 2-way toggle instead of the raw dropdown:
    - **"GPU (recommended)"** → stores `"auto"`.
    - **"CPU only"** → stores `"cpu"`.
    - When `accelerator !== "cuda"`, the GPU option is **disabled** with an inline
      note ("no CUDA GPU on this machine — integrated/APU can't accelerate"), and
      CPU is the only selectable option — the panel shows **"CPU only" as the
      effective selection regardless of the stored value** (a stored `"auto"`/
      `"cuda"` already resolves to `cpu` on such a machine, so this matches
      runtime behaviour; saving persists `"cpu"`).
    - Display mapping on a `cuda` machine: stored `"auto"` and `"cuda"` both show
      as "GPU (recommended)"; `"cpu"` shows as "CPU only".
- **`settingsFields.ts`** (hand-synced contract): the `device` field metadata
  changes from the raw `auto/cpu/cuda` select to the GPU/CPU toggle model above.
  `ALLOWED_DEVICES` stays as the mirror of the server's stored values.

## Data flow

`GET /api/system-info` → `probe_hardware()` returns `HardwareInfo` incl.
`accelerator` → `SystemInfoResponse` → `useSidecarSettings` → `AdminPanel`
renders the label and gates the Device toggle. Device changes still go through
the existing `PATCH /api/settings` path (device is a restart-required field —
unchanged).

## Error handling

- Adapter enumeration never raises; any failure yields `[]`, degrading to
  `cuda` (if torch sees a GPU) or `cpu`. Worst case an integrated GPU is
  mislabelled as "CPU only" — cosmetic, no functional impact.
- No new failure modes in the device path: `resolve_device` is unchanged and
  `"auto"` is safe on every class.

## Testing

- **Sidecar** `tests/test_hardware.py`: inject a fake `adapter_lister` and
  monkeypatch `torch.cuda.is_available` to cover all three classes:
  - GPU present → `accelerator == "cuda"`.
  - No CUDA + `["AMD Radeon(TM) Graphics"]` → `"integrated"`, `gpu_name` set.
  - No CUDA + `["Microsoft Basic Display Adapter"]` / `[]` → `"cpu"`.
  - `adapter_lister` raising → treated as `[]` (no crash).
- **Sidecar** schema test: `accelerator` present and one of the three literals.
- **Desktop** `AdminPanel` tests: render with `systemInfo` of each class; assert
  the hardware label text and that the GPU toggle option is enabled only for
  `cuda`.

## Manual verification

On this dev machine (RTX 4060): `GET /api/system-info` reports
`accelerator: "cuda"`, AdminPanel shows the RTX 4060 label with GPU selectable
and selected by default.

## Out of scope / future

- Non-CUDA acceleration backends (DirectML/OpenVINO/ROCm) for APU/iGPU inference.
- VRAM probing for integrated GPUs.
