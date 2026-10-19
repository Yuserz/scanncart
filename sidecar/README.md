# SCANnCART Sidecar

Standalone Python service: camera capture → detection + tracking → WebSocket stream.
Detection runs through a swappable backend — YOLO11 weights in-process, or a
Roboflow Workflow over HTTP. See [../docs/DETECTOR_BACKENDS.md](../docs/DETECTOR_BACKENDS.md).

> Setting up the whole project from a fresh clone? Start with the step-by-step
> [development guide](../docs/DEVELOPMENT.md); this file covers sidecar-specific
> setup, GPU/CUDA details, and manual verification.

## Setup

```bash
cd sidecar
python -m venv .venv
# Windows: .venv\Scripts\activate   |   *nix: source .venv/bin/activate
pip install -r requirements.lock.txt
```

> No system `pip`/`venv`? [uv](https://docs.astral.sh/uv/) works without sudo:
> `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock.txt`

> **Use Python 3.12.** The pinned `numpy`/`torch`/`opencv` wheels are not all
> available (or crash) on very new interpreters — a 3.14 venv was seen with a
> broken numpy (native `_multiarray_umath` load aborting) and no installable
> `torch`. `uv venv --python 3.12` fetches 3.12 automatically if it's not on
> the machine.

### GPU (NVIDIA/CUDA) setup

`requirements.lock.txt` pins `torch==2.11.0` / `torchvision==0.26.0` as PyPI's **CPU** wheels, so
the base install runs on the CPU even on a CUDA machine. `setup.cmd` and `make sidecar-setup` then
run `scripts/cuda_torch.py`, which swaps in the *same versions'* CUDA builds when `nvidia-smi` is on
PATH. It picks the wheel tag from the card's compute capability: `cu128` for Turing (RTX 20xx /
GTX 16xx) and newer, Blackwell included, and `cu126` for older cards, whose kernels the cu128 builds
no longer carry. A build without kernels for the card still reports `is_available() == True` and
fails on the first kernel, so the script checks the GPU by running one, and `resolve_device`
(`app/settings.py`) makes the same check at runtime and falls back to the CPU.

By hand, keep `torch` **and** `torchvision` together and at the locked versions (a mismatched pair
fails at import with `RuntimeError: operator torchvision::nms does not exist`):

```bash
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision
```

Verify with the venv's interpreter: `.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` (POSIX: `.venv/bin/python`).

> Pin the version, not just the tag: unpinned, uv takes the newest build the tag's index carries,
> and that differs by tag (the `cu124` index stops at 2.6.0), so machines set up with different
> tags ran different torches. Verified on the dev machine (RTX 4060, compute capability 8.9):
> `torch==2.11.0+cu128`, `cuda_available: true`.

### Roboflow credentials (only for the API backends)

`native` needs nothing here. The two API backends need a Roboflow key:

```bash
cp .env.example .env      # then fill in ROBOFLOW_API_KEY
```

`.env` is gitignored. The key is read from the process environment first and
that file second, so a packaged deploy can inject it without shipping a file.
It deliberately never touches `Settings`: settings are serialized wholesale to
the renderer by `GET /api/settings`, which reports only
`roboflow_api_key_present: true|false` and never the value.

### Running `local_api` without Docker

Roboflow documents `inference server start`, which needs Docker. The same
server runs natively — `local_inference_server.py` builds the FastAPI app the
Docker image serves. Full detail in
[../docs/DETECTOR_BACKENDS.md](../docs/DETECTOR_BACKENDS.md) §7a.

```bash
uv venv --python 3.12 .venv-inference
uv pip install --python .venv-inference/Scripts/python.exe -r requirements-inference.txt
.venv-inference/Scripts/python.exe local_inference_server.py
# -> Roboflow inference server (no Docker) on http://127.0.0.1:9001
```

> **A separate venv is mandatory.** `inference` pins numpy/opencv versions that
> conflict with the ultralytics stack in `requirements.txt` — installing it
> into `.venv` breaks the `native` backend. The two processes share nothing but
> HTTP. Python must be **<3.13**; `inference` publishes no 3.13 wheels.

Then set `detector_backend: local_api` and `track_expiry_s: 2.0` or higher, and
use **Test connection** in the Admin Panel (`POST /api/detector/probe`) before
starting capture. Measured ~90 ms warm on the same PC; the first call is slower
because it downloads and loads the model.

### ONNX on the GPU (the custom grocery model)

The custom model is ONNX, so it runs under **onnxruntime**, not torch — a
separate runtime from the one the CUDA section above configures. Getting it onto
the GPU is worth doing: measured 19.7 ms vs 66.3 ms on CPU (50.7 vs 15.1 fps).

The GPU build is pinned in **`requirements-cuda.txt`** (version-paired to the
torch CUDA in the venv — never install both it and the CPU build, they share
the `onnxruntime` import name). Install it once instead of the CPU default:

```bash
uv pip uninstall --python .venv/Scripts/python.exe onnxruntime
uv pip install --python .venv/Scripts/python.exe -r requirements-cuda.txt
```

1. **The CUDA major version must match torch's.** `onnxruntime-gpu` 1.27 and
   later are built for CUDA 13 (`cublasLt64_13.dll`); 1.21–1.26 are built for
   CUDA 12.8. Check with
   `.venv/Scripts/python.exe -c "import torch; print(torch.version.cuda)"`
   (POSIX: `.venv/bin/python`) and pick the build to match — a mismatch reports `CUDAExecutionProvider` as available and then
   fails on the first frame with *"no data transfer registered"*.
2. **onnxruntime-gpu does not ship its CUDA runtime.** It dlopen's cublas,
   cublasLt, cudart and cudnn from the loader path. torch already ships matching
   builds in `torch/lib`, so `inference.enable_onnx_cuda()` adds that directory
   before the session is created. No CUDA toolkit install is needed.

> Never have both `onnxruntime` and `onnxruntime-gpu` installed: they share the
> `onnxruntime` import name, so both present breaks either, and uninstalling one
> deletes the shared package directory. If imports start failing, delete
> `.venv/Lib/site-packages/onnxruntime*` and reinstall exactly one.

Verify:

```bash
.venv/Scripts/python.exe -c "import onnxruntime as o; print(o.get_available_providers())"   # Windows
.venv/bin/python -c "import onnxruntime as o; print(o.get_available_providers())"           # Linux/macOS
```

`ultralytics` also pip-installs dependencies at import time and will swap the
CPU build back in on a CUDA machine, which breaks capture mid-session. `run.py`
sets `YOLO_AUTOINSTALL=false` to stop it; keep that if you write another
entrypoint.

### Windows Defender / antivirus (silent native-import crashes)

On some Windows machines, Defender's real-time protection **intermittently
kills Python as it loads native extension DLLs** (numpy, OpenCV, torch). The
crash is silent: the process exits with code `0xffffffff` / `4294967295` and
prints **no traceback**. Because `torch` loads many DLLs, the sidecar then dies
on almost every launch — the Electron app logs
`[sidecar] exited unexpectedly (code 4294967295)` and shows no live feed, while
`.venv/Scripts/python.exe run.py` on its own exits `255` with empty output.

The tell that it's a scan race (not a broken install) is that a heavy import
succeeds only *some* of the time — e.g. `import torch` passing 0–3 times out of
20 while `psutil` (few DLLs) passes 20/20.

Fix: exclude the repo and the interpreter directory from Defender scanning
(run in an **elevated / Administrator** PowerShell):

```powershell
Add-MpPreference -ExclusionPath "C:\path\to\scanncart"
Add-MpPreference -ExclusionPath "$env:APPDATA\uv"   # if using a uv-managed Python
```

Then confirm a heavy import is now reliable (expect 10/10 exit 0):

```powershell
$ok=0; 1..10 | %{ .venv\Scripts\python.exe -c "import torch,cv2,ultralytics" 2>$null; if ($?){$ok++} }; "OK=$ok/10"
```

## Run

```bash
.venv/Scripts/python.exe run.py   # Windows
.venv/bin/python run.py           # Linux/macOS
# prints: SIDECAR_PORT=8765
```

The default model is the trained grocery weight `models/scanncart-grocery-v2.1.pt` (with its
record `models/scanncart-grocery-v2.1.json`). The installed weights and their records are
committed, so a fresh clone or `git pull` has them; to train new ones see `docs/RUN_SHEET.md` and
[`models/README.md`](models/README.md). A stock
model such as `yolo11n.pt` is downloaded by Ultralytics into `sidecar/` on the first capture start
that selects it.

Settings persist to `data/settings.json`, loaded on startup and written back on
every `PATCH /api/settings` or preset apply. A missing or corrupt file falls
back to hardcoded defaults — never crashes startup. Hardware detection
(`GET /api/system-info`) requires `psutil`, in `requirements.txt`.

The fields are the model and device, capture resolution/fps, confidence
threshold, inference size (`imgsz`), frame skip, preview size, track expiry,
and the detector-backend block (`detector_backend`, the Roboflow
workspace/workflow ids, the two API URLs, and the remote transmit size,
timeout and retry count).

`settings_store.RESTART_REQUIRED_FIELDS` is the source of truth for which of
those need a stop/start; everything else the running pipeline re-reads live.
Changing a restart-required field mid-capture is refused with a `409`.

## Verify manually

1. Health: `curl http://127.0.0.1:8765/api/health`
2. Settings: `curl http://127.0.0.1:8765/api/settings`
3. Cameras: `curl http://127.0.0.1:8765/api/cameras` — names each index, so you
   can tell the built-in webcam from the StreamCam. Add `?rescan=true` to
   re-probe; the result is cached because opening every device is slow.
4. Backend reachable? `curl -X POST http://127.0.0.1:8765/api/detector/probe`
5. Start: `curl -X POST http://127.0.0.1:8765/api/capture/start`
6. Open `ws_test.html` (a local scratch file, not committed) in a browser to see live frames + boxes.
7. Stop: `curl -X POST http://127.0.0.1:8765/api/capture/stop`

> Opening a camera can take a while — a Logitech StreamCam measured ~37 s
> (~9.5 s to open plus ~18.7 s for the 1080p mode-set). That is the device, not
> the sidecar: `/api/health` keeps answering throughout.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -v   # Windows
.venv/bin/python -m pytest -v           # Linux/macOS
```

The whole suite runs against fakes — no camera, no GPU, no network, and no
Roboflow key. Don't add tests that assume any of them are present.
