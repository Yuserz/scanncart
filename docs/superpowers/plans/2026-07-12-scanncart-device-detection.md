# SCANnCART Device Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect and report three hardware classes (CPU-only / APU-integrated GPU / CPU+NVIDIA GPU) and give the user a clear GPU-vs-CPU device choice that defaults to GPU when a CUDA GPU is present.

**Architecture:** The sidecar's `probe_hardware()` gains display-adapter enumeration (Windows `Win32_VideoController`, injected for tests) and a new `accelerator` classification field, surfaced through the existing `GET /api/system-info`. The desktop AdminPanel renders a class-aware hardware label and replaces the raw `auto/cpu/cuda` dropdown with a GPU/CPU toggle that disables GPU when no CUDA GPU exists. Device default stays `"auto"` (GPU-when-present, crash-safe).

**Tech Stack:** Python 3.12, FastAPI, pydantic, psutil, pytest (sidecar). TypeScript, React, Vitest, @testing-library (desktop).

## Global Constraints

- Adapter detection is Windows-only (`Win32_VideoController`); any failure returns `[]` and never raises — matches the PRD's single-Windows-PC scope.
- Sidecar tests stay hardware-free: the adapter lister and torch are injected/monkeypatched, never a real probe. (`FakeFrameSource`-style convention.)
- The WS/settings/system-info contracts are hand-synced between Python (`schemas.py`) and TS (`api.ts`, `settingsFields.ts`) — update both sides in lockstep.
- `device` remains a **restart-required** field (unchanged); stored values stay `{"auto", "cpu", "cuda"}`. `resolve_device` is unchanged.
- `accelerator` is one of exactly `"cuda" | "integrated" | "cpu"`.
- Do not break existing tests: the sidecar torch-missing test and the desktop AdminPanel `getSystemInfo` mock both need updating in-plan.

---

### Task 1: Sidecar — adapter enumeration + `accelerator` classification

**Files:**
- Modify: `sidecar/app/hardware.py`
- Test: `sidecar/tests/test_hardware.py`

**Interfaces:**
- Consumes: nothing new (stdlib `subprocess`, `sys`; existing `psutil`, lazy `torch`).
- Produces:
  - `HardwareInfo.accelerator: Literal["cuda", "integrated", "cpu"]` (new field, default `"cpu"`).
  - `probe_hardware(adapter_lister: Callable[[], list[str]] = _list_display_adapters) -> HardwareInfo`.
  - `_list_display_adapters() -> list[str]` (module-level default lister).

- [ ] **Step 1: Update the existing torch-missing test to inject an empty lister**

In `sidecar/tests/test_hardware.py`, the torch-missing test must pin the lister so it stays deterministic (otherwise the real probe could now classify the test machine as `integrated`). Replace `test_probe_hardware_falls_back_when_torch_missing` with:

```python
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
```

- [ ] **Step 2: Add the new classification tests**

Append to `sidecar/tests/test_hardware.py`:

```python
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
```

Also add `assert hw.accelerator == "cuda"` to the existing `test_probe_hardware_reports_gpu_when_cuda_available` (after its other asserts).

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_hardware.py -v`
Expected: FAIL — `TypeError: probe_hardware() got an unexpected keyword argument 'adapter_lister'` (and `accelerator` attribute errors).

- [ ] **Step 4: Implement the detection in `hardware.py`**

Replace the entire contents of `sidecar/app/hardware.py` with:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_hardware.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add sidecar/app/hardware.py sidecar/tests/test_hardware.py
git commit -m "feat(sidecar): classify accelerator (cuda/integrated/cpu) via display-adapter probe"
```

---

### Task 2: Sidecar — expose `accelerator` on `SystemInfoResponse`

**Files:**
- Modify: `sidecar/app/schemas.py`
- Modify: `sidecar/app/main.py:184-191` (the `/api/system-info` handler)
- Test: `sidecar/tests/test_schemas.py`, `sidecar/tests/test_main.py`

**Interfaces:**
- Consumes: `HardwareInfo.accelerator` (Task 1).
- Produces: `schemas.HardwareInfo.accelerator` (pydantic) → inherited by `SystemInfoResponse`; `GET /api/system-info` returns `accelerator`.

- [ ] **Step 1: Write the failing tests**

In `sidecar/tests/test_schemas.py`, append:

```python
def test_hardware_info_accepts_accelerator():
    from app.schemas import HardwareInfo

    hw = HardwareInfo(
        cpu_count=4, ram_gb=8.0, cuda_available=False, accelerator="integrated"
    )
    assert hw.accelerator == "integrated"
```

In `sidecar/tests/test_main.py`, append (the module already defines `_make_client()` returning `(client, state)`):

```python
def test_system_info_reports_accelerator():
    client, _ = _make_client()
    r = client.get("/api/system-info")
    assert r.status_code == 200
    assert r.json()["accelerator"] in {"cuda", "integrated", "cpu"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_schemas.py::test_hardware_info_accepts_accelerator tests/test_main.py::test_system_info_reports_accelerator -v`
Expected: FAIL — `HardwareInfo` has no `accelerator` field / response missing `accelerator`.

- [ ] **Step 3: Add the schema field**

In `sidecar/app/schemas.py`, ensure `Literal` is imported (add to the existing `typing` import if absent):

```python
from typing import Literal
```

Then add the field to `HardwareInfo` (the pydantic `BaseModel`, currently near line 105):

```python
class HardwareInfo(BaseModel):
    cpu_count: int
    ram_gb: float
    cuda_available: bool
    accelerator: Literal["cuda", "integrated", "cpu"] = "cpu"
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
```

- [ ] **Step 4: Wire it through the route**

In `sidecar/app/main.py`, the `/api/system-info` handler builds `SystemInfoResponse` explicitly. Add the `accelerator` line:

```python
    @app.get("/api/system-info", response_model=SystemInfoResponse)
    async def system_info():
        hw = probe_hardware()
        return SystemInfoResponse(
            cpu_count=hw.cpu_count,
            ram_gb=hw.ram_gb,
            cuda_available=hw.cuda_available,
            accelerator=hw.accelerator,
            gpu_name=hw.gpu_name,
            gpu_vram_gb=hw.gpu_vram_gb,
            recommended_preset=recommend_preset(hw),
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_schemas.py tests/test_main.py -v`
Expected: PASS (all).

- [ ] **Step 6: Full sidecar regression + commit**

Run: `cd sidecar && python -m pytest -q`
Expected: PASS (all). Then:

```bash
git add sidecar/app/schemas.py sidecar/app/main.py sidecar/tests/test_schemas.py sidecar/tests/test_main.py
git commit -m "feat(sidecar): surface accelerator class on GET /api/system-info"
```

---

### Task 3: Desktop — `accelerator` type + class-aware hardware label

**Files:**
- Modify: `desktop/src/renderer/src/lib/api.ts:51-57` (`SystemInfoResponse`)
- Modify: `desktop/src/renderer/src/views/AdminPanel.tsx:110-115` (GPU label)
- Modify: `desktop/src/renderer/src/views/AdminPanel.test.tsx` (mock + new tests)

**Interfaces:**
- Consumes: `GET /api/system-info` `accelerator` field (Task 2).
- Produces: `SystemInfoResponse.accelerator: 'cuda' | 'integrated' | 'cpu'`; class-aware label text in the hardware section.

- [ ] **Step 1: Add `accelerator` to the shared mock and write failing label tests**

In `desktop/src/renderer/src/views/AdminPanel.test.tsx`, add `accelerator: 'cpu'` to the default `getSystemInfo` mock (inside `makeDeps`, the object returned around lines 46–53):

```typescript
    getSystemInfo: vi.fn(async () => ({
      cpu_count: 8,
      ram_gb: 16,
      cuda_available: false,
      accelerator: 'cpu' as const,
      gpu_name: null,
      gpu_vram_gb: null,
      recommended_preset: 'mid_range'
    })),
```

Then append these tests inside the `describe('AdminPanel', ...)` block:

```typescript
  it('labels a CUDA machine as GPU-acceleration-available', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: true,
        accelerator: 'cuda' as const,
        gpu_name: 'NVIDIA GeForce RTX 4060',
        gpu_vram_gb: 8,
        recommended_preset: 'high_end'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)
    await waitFor(() =>
      expect(screen.getByTestId('hardware-info')).toHaveTextContent(
        /NVIDIA GeForce RTX 4060.*GPU acceleration available/
      )
    )
  })

  it('labels an integrated GPU as APU without CUDA', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: false,
        accelerator: 'integrated' as const,
        gpu_name: 'AMD Radeon(TM) Graphics',
        gpu_vram_gb: null,
        recommended_preset: 'low_end'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)
    await waitFor(() =>
      expect(screen.getByTestId('hardware-info')).toHaveTextContent(
        /Integrated graphics: AMD Radeon\(TM\) Graphics \(APU\)/
      )
    )
  })
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: FAIL — label text does not match (still renders old "No GPU detected" only) and/or a TS error on `accelerator`.

- [ ] **Step 3: Add the type**

In `desktop/src/renderer/src/lib/api.ts`, add `accelerator` to `SystemInfoResponse`:

```typescript
export interface SystemInfoResponse {
  cpu_count: number
  ram_gb: number
  cuda_available: boolean
  accelerator: 'cuda' | 'integrated' | 'cpu'
  gpu_name: string | null
  gpu_vram_gb: number | null
  recommended_preset: string
}
```

- [ ] **Step 4: Render the class-aware label**

In `desktop/src/renderer/src/views/AdminPanel.tsx`, replace the GPU `<li>` (currently lines 110–115) with:

```tsx
            <li>
              GPU:{' '}
              {systemInfo.accelerator === 'cuda'
                ? `${systemInfo.gpu_name ?? 'CUDA GPU'} (${systemInfo.gpu_vram_gb?.toFixed(1) ?? '?'} GB VRAM) — GPU acceleration available`
                : systemInfo.accelerator === 'integrated'
                  ? `Integrated graphics: ${systemInfo.gpu_name ?? 'unknown'} (APU) — no CUDA acceleration, runs on CPU`
                  : 'No GPU detected — CPU only'}
            </li>
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: PASS (existing "No GPU detected" assertion still matches the new `cpu` label, which contains that substring).

- [ ] **Step 6: Commit**

```bash
git add desktop/src/renderer/src/lib/api.ts desktop/src/renderer/src/views/AdminPanel.tsx desktop/src/renderer/src/views/AdminPanel.test.tsx
git commit -m "feat(desktop): class-aware hardware label from accelerator field"
```

---

### Task 4: Desktop — GPU/CPU device toggle

**Files:**
- Modify: `desktop/src/renderer/src/views/AdminPanel.tsx:152-192` (special-case the `device` field in the settings loop)
- Modify: `desktop/src/renderer/src/lib/settingsFields.ts:36-42` (device field hint)
- Modify: `desktop/src/renderer/src/views/AdminPanel.test.tsx` (toggle tests)

**Interfaces:**
- Consumes: `SystemInfoResponse.accelerator` (Task 3); existing `valueOf`/`setField`/`draft` machinery.
- Produces: a `data-testid="device-toggle"` control mapping GPU→`"auto"`, CPU→`"cpu"`, GPU disabled when `accelerator !== "cuda"`.

- [ ] **Step 1: Write the failing toggle tests**

Append to `desktop/src/renderer/src/views/AdminPanel.test.tsx` inside the describe block:

```typescript
  it('device toggle: GPU is selectable and default on a CUDA machine, and stores auto', async () => {
    const { deps, api } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: true,
        accelerator: 'cuda' as const,
        gpu_name: 'NVIDIA GeForce RTX 4060',
        gpu_vram_gb: 8,
        recommended_preset: 'high_end'
      }))
    })
    const user = userEvent.setup()
    render(<AdminPanel port={8765} deps={deps} />)

    const gpu = await screen.findByLabelText(/GPU \(recommended\)/i)
    const cpu = screen.getByLabelText(/CPU only/i)
    expect(gpu).toBeEnabled()
    expect(gpu).toBeChecked() // stored device 'auto' shows as GPU

    // Switch to CPU, then save persists 'cpu'.
    await user.click(cpu)
    const saveButton = screen.getByTestId('save-settings')
    await waitFor(() => expect(saveButton).toBeEnabled())
    await user.click(saveButton)
    await waitFor(() => expect(api.updateSettings).toHaveBeenCalledWith({ device: 'cpu' }))
  })

  it('device toggle: GPU is disabled and CPU forced when no CUDA GPU', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: false,
        accelerator: 'integrated' as const,
        gpu_name: 'AMD Radeon(TM) Graphics',
        gpu_vram_gb: null,
        recommended_preset: 'low_end'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const gpu = await screen.findByLabelText(/GPU \(recommended\)/i)
    expect(gpu).toBeDisabled()
    expect(screen.getByLabelText(/CPU only/i)).toBeChecked()
    expect(screen.getByTestId('device-gpu-note')).toBeInTheDocument()
  })
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: FAIL — no element labelled "GPU (recommended)"/"CPU only" yet (device still renders as the raw dropdown).

- [ ] **Step 3: Special-case the `device` field in the settings loop**

In `desktop/src/renderer/src/views/AdminPanel.tsx`, inside `SETTINGS_FIELDS.map((field) => { ... })`, after computing `const value = valueOf(field.key)` and before the existing `return (`, insert a device branch:

```tsx
          if (field.key === 'device') {
            const gpuAvailable = systemInfo?.accelerator === 'cuda'
            const isCpu = value === 'cpu' || !gpuAvailable
            return (
              <div className="admin-field" key={field.key}>
                <div className="admin-field-label">
                  <label htmlFor={field.key}>{field.label}</label>
                  <span className={`badge ${isRestartField ? 'restart' : 'live'}`}>
                    {isRestartField ? 'restart required' : 'live'}
                  </span>
                </div>
                <div className="device-toggle" data-testid="device-toggle">
                  <label>
                    <input
                      type="radio"
                      name="device"
                      value="gpu"
                      checked={!isCpu}
                      disabled={!gpuAvailable}
                      onChange={() => setField('device', 'auto')}
                    />
                    GPU (recommended)
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="device"
                      value="cpu"
                      checked={isCpu}
                      onChange={() => setField('device', 'cpu')}
                    />
                    CPU only
                  </label>
                </div>
                {!gpuAvailable && (
                  <p className="field-hint" data-testid="device-gpu-note">
                    No CUDA GPU on this machine — integrated/APU can&apos;t accelerate; running on
                    CPU.
                  </p>
                )}
                <p className="field-hint">{field.hint}</p>
              </div>
            )
          }
```

- [ ] **Step 4: Update the device field hint**

In `desktop/src/renderer/src/lib/settingsFields.ts`, update the `device` field's `hint` (leave `key`/`type`/`options` as-is; the loop now intercepts `device` so `type`/`options` are unused for it):

```typescript
  {
    key: 'device',
    label: 'Device',
    hint: 'GPU uses your NVIDIA (CUDA) GPU for faster inference; CPU runs on the processor. GPU is the default when a CUDA GPU is present, and disabled otherwise.',
    type: 'select',
    options: ALLOWED_DEVICES
  },
```

- [ ] **Step 5: Run the AdminPanel suite to verify it passes**

Run: `cd desktop && npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: PASS (all, including the pre-existing tests — none referenced a device `<select>`).

- [ ] **Step 6: Full desktop suite + typecheck**

Run: `cd desktop && npx vitest run && npm run typecheck`
Expected: PASS / no type errors.

- [ ] **Step 7: Commit**

```bash
git add desktop/src/renderer/src/views/AdminPanel.tsx desktop/src/renderer/src/lib/settingsFields.ts desktop/src/renderer/src/views/AdminPanel.test.tsx
git commit -m "feat(desktop): GPU/CPU device toggle gated on CUDA availability"
```

---

## Verification (end of plan)

- [ ] Sidecar: `cd sidecar && python -m pytest -q` → all pass.
- [ ] Desktop: `cd desktop && npx vitest run` → all pass; `npm run typecheck` → clean.
- [ ] Manual smoke (needs the app running): open Admin → "This machine" shows the RTX 4060 with "GPU acceleration available"; Device shows GPU (recommended) selected and enabled. `curl http://127.0.0.1:<port>/api/system-info` includes `"accelerator": "cuda"`.

---

## Notes for the implementer

- **Why the lister is injected:** `probe_hardware` shells out to PowerShell for `Win32_VideoController`, which can't run in the hardware-free test suite — tests pass a fake `adapter_lister`, exactly like `source_factory`/`detector_factory` elsewhere.
- **Why GPU maps to `"auto"` not `"cuda"`:** `"auto"` resolves to the GPU when present but degrades to CPU safely if the machine ever lacks a CUDA GPU, so a persisted setting never causes a capture-start failure. `"cuda"` remains a valid stored value for back-compat.
- **Two `HardwareInfo` types:** the dataclass in `hardware.py` (what `probe_hardware` returns) and the pydantic model in `schemas.py` (the API contract) are separate by design — keep the `accelerator` field in sync across both.
- **Existing-test breakage is intentional:** Task 1 Step 1 and Task 3 Step 1 update pre-existing tests/mocks that would otherwise fail once `accelerator` exists; do them as written.
