# SCANnCART Sidecar (Phase 1)

Standalone Python service: camera capture → YOLO11 tracking → WebSocket stream.

## Setup

```bash
cd sidecar
python -m venv .venv
# Windows: .venv\Scripts\activate   |   *nix: source .venv/bin/activate
pip install -r requirements.txt
```

> No system `pip`/`venv`? [uv](https://docs.astral.sh/uv/) works without sudo:
> `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

### GPU (NVIDIA/CUDA) setup

`requirements.txt` doesn't pin `torch`, so a plain install pulls in
whatever `ultralytics` resolves — on Windows/Linux that's usually a
**CPU-only** wheel (`torch==X.Y.Z+cpu`), even on a machine with a CUDA
GPU. `resolve_device("auto")` and `GET /api/system-info` both trust
`torch.cuda.is_available()`, so a CPU-only wheel makes a real GPU silently
report as "not detected" — no error, just `cuda_available: false`.

If you have an NVIDIA GPU, reinstall `torch` **and** `torchvision` together
from PyTorch's CUDA index after the base install (adjust the `cuXXX` tag to
a version your driver supports — check `nvidia-smi`'s reported CUDA
version, newer drivers are backwards compatible with older `cuXXX`
wheels). Reinstalling only `torch` leaves `torchvision` on whatever
CPU/loose version `ultralytics` originally resolved, and a mismatched pair
fails at import time with `RuntimeError: operator torchvision::nms does
not exist` (torchvision registers its ops against the specific torch
build it was compiled for) — so always pin both to the same `cuXXX` tag:

```bash
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124 --reinstall-package torch --reinstall-package torchvision
```

Verify with `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

## Run

```bash
python run.py
# prints: SIDECAR_PORT=8765
```

On first capture start, Ultralytics downloads `yolo11n.pt` automatically.

Settings (model, device, capture resolution/fps, confidence threshold, frame
skip, preview size, track expiry) persist to `data/settings.json`, loaded on
startup and written back on every `PATCH /api/settings` or preset apply. A
missing or corrupt file just falls back to hardcoded defaults — never crashes
startup. Hardware detection (`GET /api/system-info`) requires `psutil`, now
in `requirements.txt`.

## Verify manually

1. Start: `curl -X POST http://127.0.0.1:8765/api/capture/start`
2. Health: `curl http://127.0.0.1:8765/api/health`
3. Settings: `curl http://127.0.0.1:8765/api/settings`
4. Open `ws_test.html` (a local scratch file, not committed) in a browser to see live frames + boxes.
5. Stop: `curl -X POST http://127.0.0.1:8765/api/capture/stop`

## Tests

```bash
python -m pytest -v
```

All 25 tests run with fakes — no camera and no network required.
