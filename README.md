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

## Quick start

**New to the project? Follow [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — a
step-by-step guide from clone to running app**, including prerequisites, the
manual equivalent of every `make` target, GPU/Roboflow extras, and the env
quirks that bite on a fresh machine.

A root `Makefile` wraps both toolchains (requires GNU Make — on Windows use
Git Bash/WSL, or `winget install GnuWin32.Make`). Run `make help` to list all
targets.

```bash
make install   # desktop npm install + sidecar venv setup
make test      # desktop vitest + sidecar pytest (fakes only — no camera needed)
make dev       # run the desktop app in dev mode (spawns the sidecar)
make build     # typecheck + build the desktop app
```

For package-specific setup, manual run instructions, and known env quirks,
see [`sidecar/README.md`](sidecar/README.md) and
[`desktop/README.md`](desktop/README.md).

## For agents

[`CLAUDE.md`](CLAUDE.md) has a deeper architecture map (module-by-module) and
the full command reference, including how to run a single test in each
toolchain.
