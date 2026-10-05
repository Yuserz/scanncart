# SCANnCART

A capstone prototype for grocery stores: **a camera on a shopping cart** that
recognises products as the customer puts them in or takes them out, and keeps the
bill on the store's self-checkout tablet in step — no barcode scanning, no cashier.

- A **Logitech StreamCam** on the cart (1280×720 at 60 fps) feeds a **Python
  sidecar** that runs a YOLO11 model, fine-tuned on 7 grocery products, with
  object tracking.
- An **Electron + React desktop app** shows the live annotated feed, decides
  which movements are deposits (+1) and removals (−1), and keeps the basket's
  ledger.
- **pushcart-web** — a separate checkout you host yourself — owns the cart, the
  prices, the stock and the order, and runs the customer's tablet.

Detection runs locally on one PC with no internet dependency; the only network
hop is desktop → pushcart-web on the shop LAN. See [`docs/PRD.md`](docs/PRD.md)
for the product spec, [`docs/DEFENSE_READINESS.md`](docs/DEFENSE_READINESS.md)
for where every requirement stands, and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
for out-of-scope future work (edge hardware, cloud sync, etc).

## Architecture

Two independent toolchains, talking over localhost HTTP/WebSocket:

```
Logitech StreamCam ──USB──▶ sidecar/ (Python/FastAPI)  ──ws://127.0.0.1:<port>──▶ desktop/ (Electron/React)
                             OpenCV capture → YOLO11 track            live preview + overlay + item log
                             SQLite detection log                     REST for start/stop/health/logs
```

The desktop app's Electron main process spawns `sidecar/run.py` as a child
process on startup and shuts it down on quit; the renderer discovers the
sidecar's port over IPC and then talks to it directly (WebSocket for the
live frame stream, REST for start/stop/health/logs).

- **[`sidecar/`](sidecar/README.md)** — Python/FastAPI service: camera
  capture, YOLO11 inference + tracking, SQLite detection logging.
- **[`desktop/`](desktop/README.md)** — Electron + React + TypeScript UI:
  spawns/supervises the sidecar, renders the live view.

The **self-checkout integration** adds one hop: a cart tablet runs a separate
checkout (pushcart-web) that owns the cart, the order and the stock, and this
app feeds it the basket's contents. In **basket** mode a product carried into
the basket adds one, one carried out removes one, and anything ambiguous holds
the cart and waits for a staff check. It is off until an admin configures it. [`docs/POS_INTEGRATION.md`](docs/POS_INTEGRATION.md) is the
setup, the operator-less flow and the troubleshooting table; the ordered run
through both halves, to do before the shop goes live, is
[`docs/POS_SMOKE_TEST.md`](docs/POS_SMOKE_TEST.md).

## Setup — clone to running app

Five steps, and none of them need a camera, a GPU, or an API key. This is the
short form; [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) is the long one, with
the manual equivalent of every command below, the optional GPU/Roboflow
extras, and the environment quirks that are easy to lose an afternoon to.
[`QUICKSTART.md`](QUICKSTART.md) is a one-screen version of this section.

### 0. Prerequisites

| Tool | Version | Why |
| --- | --- | --- |
| **Git** | any recent | clone the repo |
| **Node.js + npm** | **20+** (CI uses 20) | the Electron/React desktop app |
| **Python** | **exactly 3.12** | the sidecar — see the warning below |
| **GNU Make** | any | optional; wraps both toolchains |
| **uv** | any recent | optional but recommended; manages Python 3.12 for you |
| **A camera** | USB webcam or Logitech StreamCam | optional — only for live capture; both test suites run on fakes |

**Python 3.12 is not a suggestion.** The pinned `numpy` / `torch` / `opencv`
wheels are not all available (or crash) on newer interpreters — a 3.13/3.14
venv has produced a broken numpy and no installable `torch`. With `uv`,
`uv venv --python 3.12` fetches 3.12 for you.

**Make is optional and needs a POSIX shell.** On Windows use Git Bash or WSL,
or install it with `winget install GnuWin32.Make`. Without Make, use the manual
commands in [step 2](#2-install-both-toolchains) and
[step 3](#3-verify-the-install--no-camera-gpu-or-api-key), and
`sidecar\.venv\Scripts\python.exe` instead of `sidecar/.venv/bin/python`.

### 1. Clone

```bash
git clone <repo-url> scanncart
cd scanncart
```

### 2. Install both toolchains

```bash
make install            # = desktop npm install  +  sidecar venv setup
```

Without Make — **sidecar**, using `uv` (fetches Python 3.12, no activation):

```bash
cd sidecar
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows
uv pip install --python .venv/bin/python -r requirements.txt           # Linux/macOS
```

or with plain `venv`:

```bash
cd sidecar
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt            # Windows
.venv/bin/python -m pip install -r requirements.txt                    # Linux/macOS
```

**Desktop**:

```bash
cd desktop
npm install
```

The first `npm install` runs Electron's binary postinstall and can take a
while. If it fails with *"Electron uninstall"*, see
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#8-troubleshooting).

### 3. Verify the install — no camera, GPU, or API key

Both suites run entirely against fakes, so this is the fastest way to know the
environment is sound before touching hardware.

```bash
make test               # desktop vitest + sidecar pytest
```

Without Make:

```bash
(cd sidecar && .venv/Scripts/python.exe -m pytest -v)   # Windows
(cd sidecar && .venv/bin/python -m pytest -v)            # Linux/macOS
(cd desktop && npm test)
```

There is nothing to configure for this step — no `.env`, no camera, no
`settings.json`. If these pass, the toolchains are correctly installed.

### 4. Run the app

```bash
make dev                # = cd desktop && npm run dev
```

This starts electron-vite in dev mode. The Electron main process launches
`sidecar/run.py` as a child process, the sidecar prints `SIDECAR_PORT=<n>`, and
the app connects to it. You should land on the **Live** view; click **Start**
to begin capture. The **Admin** panel has hardware info, the settings form, the
model picker, and a **Test connection** button for the detector backend.

Two things worth expecting on first launch:

- **The grocery model is not in git.** The default model is
  `sidecar/models/scanncart-grocery-v1.pt` with its record `scanncart-grocery-v1.json`
  beside it; weights are gitignored, so copy both into `sidecar/models/` from the
  team's share, or train them ([`docs/RUN_SHEET.md`](docs/RUN_SHEET.md)). Without them,
  pick a stock model (e.g. `yolo11n.pt`) in **Admin → Model**: it downloads into
  `sidecar/` on the first capture start and detects generic objects, not the store's
  products. Leave stock weights where they land rather than moving them into
  `sidecar/models/`.
- A Logitech StreamCam can take ~37 s to open and set its capture mode. That is
  the device, not a hang — `/api/health` keeps answering throughout, and a
  frame that never arrives is reported as an `error` status after a 3 s
  deadline.

If your layout differs from the repo, point the app at the sidecar explicitly:

```bash
cd desktop
SIDECAR_PYTHON=/path/to/sidecar/.venv/Scripts/python.exe SIDECAR_SCRIPT=/path/to/run.py npm run dev
```

### 5. Run the sidecar on its own (optional)

Useful when working on the Python service without the Electron GUI:

```bash
make sidecar-run        # = cd sidecar && .venv/…/python run.py
```

It prints the port it chose (`8765` unless that is taken) and serves REST +
WebSocket on it. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#5-run-the-sidecar-on-its-own-backend-development)
for `curl` checks and the GPU/Roboflow extras.

## Make targets

`make` or `make help` lists them all.

| Target | What it does |
| --- | --- |
| `make install` | desktop `npm install` + sidecar venv setup |
| `make test` | desktop vitest + sidecar pytest (fakes only — needs no camera) |
| `make dev` | run the desktop app in dev mode (spawns the sidecar) |
| `make build` | typecheck + build the desktop app |
| `make lint` / `make format` / `make typecheck` | the desktop toolchain |
| `make sidecar-run` / `make sidecar-test` | the sidecar alone |
| `make docs-check` / `make docs-sync` / `make docs-sync-check` | documentation links and code-owned numbers |
| `make doctor`, `make accept-v2`, `make verify-clamp`, `make verify-unsure`, `make annotate`, `make human-pass` | dataset/training gates that need local data the repo does not carry |
| `make verify-live-layout` | draws the built app in Electron and fails if the Live tab needs more than one screen (no local data, but a display) |
| `make replay-scenarios` | replays recorded deposit/removal clips through the scorer (needs recorded clips) |
| `make verify-pos-routes` / `make verify-pos-contract` | checks the self-checkout routes against a running pushcart-web / its source checkout |

## Using it

1. **Admin → Self-checkout:** enter the pushcart-web address, the shared secret and
   this cart's station; choose **Basket**; *Save*; *Test connection*.
2. **Basket test:** check the zones (outside / opening / inside) match the basket in
   the picture. Every detected box is labelled with its product, confidence and
   track id, and a **practice** session counts deposits and removals with no tablet.
3. **Live:** start capture; confirm *Capture fps* reads about 60. Close NVIDIA
   Broadcast, Discord or anything else that can hold the camera.
4. **Tablet:** the customer taps **Start shopping**, puts products in or takes them
   out, and taps **Finish**. A staff check (*Basket checked* on the desktop's Live
   view) clears an ambiguous movement; staff can lower a quantity on the tablet with
   the **staff code**, which an admin sets in pushcart-web's POS Mapping screen.

## Documentation map

| Doc | What it is for |
| --- | --- |
| [`docs/PRD.md`](docs/PRD.md) | The product requirements, numbered |
| [`docs/DEFENSE_READINESS.md`](docs/DEFENSE_READINESS.md) | Each requirement's status and evidence, open risks, demo-day checklist |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the pieces fit, with diagrams |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | The long-form setup and troubleshooting |
| [`docs/POS_INTEGRATION.md`](docs/POS_INTEGRATION.md) | Self-checkout setup, flow, basket mode, troubleshooting |
| [`docs/POS_INTEGRATION_SPEC.md`](docs/POS_INTEGRATION_SPEC.md) | The integration spec both repos implement |
| [`docs/POS_SMOKE_TEST.md`](docs/POS_SMOKE_TEST.md) | The joint go-live run through both halves |
| [`docs/CART_TRANSFER_SPEC.md`](docs/CART_TRANSFER_SPEC.md) | The deposit/removal rules and their acceptance gates (A–C) |
| [`docs/GATE_A_RUN_SHEET.md`](docs/GATE_A_RUN_SHEET.md) | The printable Gate A recording protocol |
| [`docs/MODEL_TRAINING.md`](docs/MODEL_TRAINING.md) | Why the model is trained the way it is |
| [`docs/RUN_SHEET.md`](docs/RUN_SHEET.md) | The shoot → label → train → validate → install chain, in order |
| [`docs/CAPTURE_CHECKLIST.md`](docs/CAPTURE_CHECKLIST.md) | What to photograph for the dataset |
| [`docs/DETECTOR_BACKENDS.md`](docs/DETECTOR_BACKENDS.md) | `native` vs `local_api` vs `cloud_api` |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Out-of-scope future work |

## For agents

[`CLAUDE.md`](CLAUDE.md) has a deeper architecture map (module-by-module) and
the full command reference, including how to run a single test in each
toolchain. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) covers how the pieces
fit together, and [`docs/DETECTOR_BACKENDS.md`](docs/DETECTOR_BACKENDS.md)
covers `native` vs `local_api` vs `cloud_api`.
