# Development setup — run SCANnCART on your machine

A step-by-step guide to go from a fresh clone to a running app. It covers the
**default path** (a webcam or Logitech StreamCam and the built-in `native`
detector, everything offline) and the optional extras (GPU acceleration,
Roboflow API backends). Nothing here needs a server, the cloud, or a network
connection once the pieces are installed.

If you just want the short version, it is:

```bash
git clone <repo-url> scanncart && cd scanncart
make install    # desktop npm install + sidecar venv
make test       # confirm both toolchains work (no camera needed)
make dev        # launch the app — it spawns the sidecar for you
```

Everything below explains those steps, gives the manual equivalent of each
`make` target, and covers the env quirks that are easy to lose an afternoon to.

---

## 0. Prerequisites

| Tool | Version | Why |
| --- | --- | --- |
| **Git** | any recent | clone the repo |
| **Node.js + npm** | **20+** (CI uses 20) | the Electron/React desktop app |
| **Python** | **exactly 3.12** | the sidecar — see the warning below |
| **GNU Make** | any | optional; wraps both toolchains |
| **uv** | any recent | optional but recommended; manages Python 3.12 without sudo |
| **A camera** | USB webcam or Logitech StreamCam | optional — only for live capture; the test suites need none |

**Python 3.12 is not a suggestion.** The pinned `numpy` / `torch` / `opencv`
wheels are not all available (or crash) on newer interpreters. A 3.13/3.14
venv has produced a broken numpy (native `_multiarray_umath` abort) and no
installable `torch`. If you use `uv`, `uv venv --python 3.12` fetches 3.12 for
you automatically.

**Make on Windows.** The `Makefile` assumes a POSIX shell. Use Git Bash or WSL,
or install GNU Make with `winget install GnuWin32.Make`. On Windows every
command in this guide also works from PowerShell — use the **manual** variants
in each step and `sidecar\.venv\Scripts\python.exe` instead of
`sidecar/.venv/bin/python`.

---

## 1. Clone the repo

```bash
git clone <repo-url> scanncart
cd scanncart
```

The project is two toolchains side by side — `sidecar/` (Python/FastAPI) and
`desktop/` (Electron/React). The desktop app's Electron main process spawns the
sidecar for you; you normally never start the sidecar by hand.

---

## 2. Install both toolchains

### Option A — one command

```bash
make install            # = desktop npm install  +  sidecar venv setup
```

### Option B — manual (same thing, no Make)

**Sidecar** (creates `sidecar/.venv` and installs `requirements.txt`):

```bash
cd sidecar
# POSIX:  python -m venv .venv && source .venv/bin/activate
# Windows: python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

Or with `uv` (no activation needed, and it fetches Python 3.12):

```bash
cd sidecar
uv venv --python 3.12 .venv
# Windows:
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
# Linux/macOS:
uv pip install --python .venv/bin/python -r requirements.txt
```

**Desktop** (installs Node deps):

```bash
cd desktop
npm install
```

> The first `npm install` runs Electron's binary postinstall and can take a
> while. If it fails with *"Electron uninstall"*, see
> [Troubleshooting](#8-troubleshooting).

---

## 3. Verify the install — no camera, GPU, or API key needed

Both test suites run entirely against fakes. This is the fastest way to know
your environment is sound before touching hardware.

```bash
make test               # desktop vitest + sidecar pytest
```

Manual equivalent:

```bash
(cd sidecar && .venv/Scripts/python.exe -m pytest -v)    # Windows
(cd sidecar && .venv/bin/python -m pytest -v)             # Linux/macOS
(cd desktop && npm test)
```

There is nothing to configure for this step — no `.env`, no camera, no
`settings.json`. If these pass, the toolchains are correctly installed.

---

## 4. Run the app

```bash
make dev                # = cd desktop && npm run dev
```

This starts electron-vite in dev mode. The Electron main process launches
`sidecar/run.py` as a child process, the sidecar prints `SIDECAR_PORT=<n>`, and
the app connects to it. You should see the **Live** view; click **Start** to
begin capture.

By default the app looks for the sidecar at `../sidecar/.venv` and
`../sidecar/run.py` (relative to the repo layout). Override either if your
setup differs:

```bash
cd desktop
SIDECAR_PYTHON=/path/to/python SIDECAR_SCRIPT=/path/to/run.py npm run dev
```

> **First capture start downloads `yolo11n.pt`** (the stock Ultralytics
> weights) if it isn't already in `sidecar/`. That is the only download the
> default path makes; afterwards everything is local. It also appears in the
> Model picker — leave those stock weights where they land, don't move them
> into `sidecar/models/`.

> **Opening the camera can take ~37 s** on a Logitech StreamCam (about 9.5 s
> to open plus ~18.7 s to set the 1080p mode). That is the device, not a
> hang — the sidecar's `/api/health` keeps answering throughout. A frame that
> never arrives is reported as an `error` status after a 3 s deadline.

Start/Stop, the live preview with overlay, the stats strip, and the session
item log all live in the **Live** view. The **Admin** panel has hardware info,
the settings form, the model picker, and a **Test connection** button for the
detector backend.

---

## 5. Run the sidecar on its own (backend development)

Useful when you're working on the Python service and don't want the Electron
GUI in the loop.

```bash
make sidecar-run        # = cd sidecar && .venv/…/python run.py
```

It prints the port it chose and serves REST + WebSocket on it. Then poke at it
by hand:

```bash
curl http://127.0.0.1:8765/api/health
curl http://127.0.0.1:8765/api/settings
curl http://127.0.0.1:8765/api/cameras          # names each index (?rescan=true to re-probe)
curl -X POST http://127.0.0.1:8765/api/capture/start
curl -X POST http://127.0.0.1:8765/api/capture/stop
```

If port 8765 is taken, `run.py` picks a free port and prints that instead —
read the `SIDECAR_PORT=<n>` line rather than assuming 8765.

---

## 6. Optional — GPU acceleration (NVIDIA/CUDA)

Skip this if you're CPU-only; everything works without it, just slower.

`requirements.txt` doesn't pin `torch`, so a plain install usually pulls a
**CPU-only** wheel even on a CUDA machine. `resolve_device("auto")` and
`GET /api/system-info` trust `torch.cuda.is_available()`, so a CPU wheel makes
a real GPU silently report as `cuda_available: false`.

Reinstall **`torch` and `torchvision` together** from PyTorch's CUDA index
(pick a `cuXXX` tag your driver supports — check `nvidia-smi`; newer drivers
are backwards compatible with older tags). Installing only `torch` leaves
`torchvision` mismatched and the import fails with
`operator torchvision::nms does not exist`.

```bash
cd sidecar
uv pip install --python .venv/Scripts/python.exe \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124 \
  --reinstall-package torch --reinstall-package torchvision
```

Verify with the venv's interpreter: `.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` (POSIX: `.venv/bin/python`).

The custom grocery model is ONNX, which runs under **onnxruntime**, a separate
runtime. To move it onto the GPU (measured ~20 ms/frame vs ~66 ms on CPU), swap
the CPU build for the version-paired GPU build in `requirements-cuda.txt`:

```bash
uv pip uninstall --python .venv/Scripts/python.exe onnxruntime
uv pip install   --python .venv/Scripts/python.exe -r requirements-cuda.txt
```

Three rules for that swap: the CUDA major version must match torch's
(`torch.version.cuda`); `onnxruntime-gpu` dlopen's its CUDA runtime from
`torch/lib`, which `inference.enable_onnx_cuda()` handles for you; and **never
have both `onnxruntime` and `onnxruntime-gpu` installed** — they share the
`onnxruntime` import name, so both break and uninstalling one deletes the
shared package directory. Full detail in
[`../sidecar/README.md`](../sidecar/README.md).

---

## 7. Optional — Roboflow API backends

The default `native` backend needs no key. The `local_api` and `cloud_api`
backends call a Roboflow Workflow over HTTP and do need one:

```bash
cd sidecar
cp .env.example .env        # then fill in ROBOFLOW_API_KEY
```

`.env` is gitignored. The key is read from the environment first, then that
file. It never reaches `Settings` — the API only ever reports
`roboflow_api_key_present: true|false`, never the value.

Then set `detector_backend` to `local_api` or `cloud_api` in the Admin panel
and press **Test connection** (`POST /api/detector/probe`) before starting
capture. `local_api` can also run without Docker via a **separate** venv — see
[`DETECTOR_BACKENDS.md`](DETECTOR_BACKENDS.md) §7a and
`sidecar/local_inference_server.py`.

---

## 8. Troubleshooting

The failures below are the ones seen in practice. They are environment
problems, not project ones.

### `python run.py` exits with code `255` / `4294967295` and no traceback

Windows Defender's real-time protection intermittently kills Python as it loads
native extension DLLs (numpy, OpenCV, torch) — a scan race, not a broken
install. The tell is that a heavy import succeeds only *some* of the time
(e.g. `import torch` passes 0–3 times out of 20). Exclude the repo and the
interpreter directory from Defender scanning (run in an **elevated**
PowerShell), then retry:

```powershell
Add-MpPreference -ExclusionPath "C:\path\to\scanncart"
Add-MpPreference -ExclusionPath "$env:APPDATA\uv"   # if using a uv-managed Python
```

### `Error: Electron uninstall` / "Electron failed to install correctly" on `npm run dev`

The Electron postinstall can leave `node_modules/electron/dist` partial (only
`locales/`, no `electron.exe`) with no `path.txt`. Re-running `npm install`
does not reliably fix it — `@electron/get` can report a cache hit and the
extract step silently no-ops. Add the Defender exclusion above first, then:

```powershell
cd desktop
Remove-Item -Recurse -Force node_modules/electron/dist
npm rebuild electron     # or: node node_modules/electron/install.js
```

If it still no-ops, extract the cached
`%LOCALAPPDATA%\electron\Cache\<hash>\electron-v<ver>-win32-x64.zip` into
`node_modules/electron/dist/` by hand and write a `path.txt` containing
`electron.exe`. Affects only launching the GUI; `npm test` and `npm run build`
are unaffected.

### A CUDA GPU reports `cuda_available: false`

You have a CPU-only torch wheel — see [step 6](#6-optional--gpu-acceleration-nvidiacuda).

### `RuntimeError: operator torchvision::nms does not exist`

`torch` and `torchvision` are on mismatched builds — reinstall **both** from
the same `cuXXX` index ([step 6](#6-optional--gpu-acceleration-nvidiacuda)).

### ONNX fails on the first frame: "no data transfer registered"

`onnxruntime-gpu`'s CUDA major version doesn't match torch's. Check
`.venv/Scripts/python.exe -c "import torch; print(torch.version.cuda)"` (POSIX: `.venv/bin/python`) and install the matching
`onnxruntime-gpu` (1.27+ = CUDA 13; 1.21–1.26 = CUDA 12.8), or reinstall the
CPU build.

### Imports suddenly fail after installing an onnxruntime build

You almost certainly have both `onnxruntime` and `onnxruntime-gpu` installed.
Delete `.venv/Lib/site-packages/onnxruntime*` and reinstall exactly one.

### The live feed is frozen with no explanation

A camera that opened but never delivers a frame is declared dead after a 3 s
deadline, and the sidecar sends an `error` status with the reason. On Windows
the sidecar pins the MSMF backend deliberately (the StreamCam is much faster
over MSMF); don't switch it to DirectShow. If a StreamCam is wedged, unplug and
replug it.

### `make: command not found`

Install GNU Make (Windows: `winget install GnuWin32.Make`) or use the manual
commands in each step. Every `make` target is a thin wrapper.

---

## 9. Where to go next

- [`../sidecar/README.md`](../sidecar/README.md) — sidecar setup detail, GPU,
  Roboflow, and manual verification.
- [`../desktop/README.md`](../desktop/README.md) — desktop layout and manual run.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the pieces fit together.
- [`PRD.md`](PRD.md) — the full product spec.
- [`DETECTOR_BACKENDS.md`](DETECTOR_BACKENDS.md) — `native` vs `local_api` vs
  `cloud_api`, and running the inference server without Docker.
- Dataset and training work (capture → clean → generate → train → `--val`) has
  its own operational chain in [`RUN_SHEET.md`](RUN_SHEET.md), with the reasoning
  in [`MODEL_TRAINING.md`](MODEL_TRAINING.md) and what to shoot in
  [`CAPTURE_CHECKLIST.md`](CAPTURE_CHECKLIST.md). That path needs data the repo
  does not carry, so it is separate from the steps above.
- Working with an agent? [`../CLAUDE.md`](../CLAUDE.md) is the module-by-module
  architecture map and full command reference.
