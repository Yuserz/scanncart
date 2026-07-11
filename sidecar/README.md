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

## Run

```bash
python run.py
# prints: SIDECAR_PORT=8765
```

On first capture start, Ultralytics downloads `yolo11n.pt` automatically.

## Verify manually

1. Start: `curl -X POST http://127.0.0.1:8765/api/capture/start`
2. Health: `curl http://127.0.0.1:8765/api/health`
3. Open `ws_test.html` (a local scratch file, not committed) in a browser to see live frames + boxes.
4. Stop: `curl -X POST http://127.0.0.1:8765/api/capture/stop`

## Tests

```bash
python -m pytest -v
```

All 25 tests run with fakes — no camera and no network required.
