# SCANnCART

A capstone prototype for grocery stores: a Logitech StreamCam feeds a Python
sidecar running YOLO11 (Ultralytics) object detection + tracking, and an
Electron + React desktop app shows the live annotated feed, per-item stats,
and a session item log. Everything runs locally on one PC — no server, no
cloud, no network dependency. See [`docs/PRD.md`](docs/PRD.md) for the full
product spec and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for out-of-scope
future work (edge hardware, cloud sync, etc).

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
cd sidecar && .venv/Scripts/python.exe -m pytest -v     # Windows
cd sidecar && .venv/bin/python -m pytest -v              # Linux/macOS
cd desktop && npm test
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

- The first capture start downloads the stock `yolo11n.pt` weights into
  `sidecar/`. That is the only download the default path makes; leave those
  weights where they land rather than moving them into `sidecar/models/`.
- A Logitech StreamCam can take ~37 s to open and set its 1080p mode. That is
  the device, not a hang — `/api/health` keeps answering throughout, and a
  frame that never arrives is reported as an `error` status after a 3 s
  deadline.

If your layout differs from the repo, point the app at the sidecar explicitly:

```bash
cd desktop
SIDECAR_PYTHON=/path/to/python SIDECAR_SCRIPT=/path/to/run.py npm run dev
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

## For agents

[`CLAUDE.md`](CLAUDE.md) has a deeper architecture map (module-by-module) and
the full command reference, including how to run a single test in each
toolchain. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) covers how the pieces
fit together, and [`docs/DETECTOR_BACKENDS.md`](docs/DETECTOR_BACKENDS.md)
covers `native` vs `local_api` vs `cloud_api`.
