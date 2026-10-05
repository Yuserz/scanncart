# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SCANnCART is a capstone prototype for grocery stores: a Logitech StreamCam feeds a Python sidecar that runs YOLO11 (Ultralytics) object tracking, and an Electron + React desktop app displays the live annotated feed and a session item log. Detection runs locally on one PC — no server, no cloud, no network dependency; the one network hop is the optional self-checkout integration, where the desktop's main process tells pushcart-web (on the shop LAN) what is in the cart's basket. The camera rides on a pushcart, and **basket** mode — camera-only deposits and removals — is the product.

Detection sits behind a swappable backend (`detector_backend`): `native` runs the weights in-process and is the only one that satisfies the offline promise; `local_api` and `cloud_api` call a Roboflow Workflow over HTTP. See `docs/DETECTOR_BACKENDS.md` — including §7a for running `local_api` with no Docker. See `docs/PRD.md` for the full product spec and `docs/DEPLOYMENT.md` for out-of-scope future work (edge hardware, cloud sync, etc). `docs/RUN_SHEET.md` is the operational chain — shoot → clean/upload → generate → train → `--val` → install — as the order to run it in with the number each step has to report before the next is worth running; `docs/MODEL_TRAINING.md` is the reasoning behind it and `docs/CAPTURE_CHECKLIST.md` is what to shoot. `docs/POS_INTEGRATION.md` is the self-checkout integration with pushcart-web — the other half lives in a separate checkout — with its setup, the operator-less flow and its troubleshooting table; the spec it implements is `docs/POS_INTEGRATION_SPEC.md`, and `docs/POS_SMOKE_TEST.md` is the joint go-live run through both halves (the one check that needs the tablet, the counter and the camera at once).

Two independent toolchains live side by side and talk over localhost HTTP/WebSocket:

```
Logitech StreamCam ──USB──▶ sidecar/ (Python/FastAPI)  ──ws://127.0.0.1:<port>──▶ desktop/ (Electron/React)
                             OpenCV capture → YOLO11 track            live preview + overlay + item log
                             SQLite detection log                     REST for start/stop/health/logs
```

`desktop/src/main` spawns `sidecar/run.py` as a child process on app startup and shuts it down on quit; the renderer never launches the sidecar itself, it just discovers the port over IPC and talks to it directly.

## Where the detail lives (loaded on demand)

This file is the overview. The module-by-module notes live beside the code and load automatically
the first time work touches that folder; `/codebase-notes <area>` loads one by name.

| File | Covers |
| --- | --- |
| [`sidecar/app/CLAUDE.md`](sidecar/app/CLAUDE.md) | Every runtime module: capture lifecycle, camera + auto exposure, detection filters, pipeline, settings, models, roster, event loop |
| [`sidecar/tools/CLAUDE.md`](sidecar/tools/CLAUDE.md) | The dataset tools: clean/plan/build/doctor/train/accept, class-order rules, the workspace |
| [`desktop/CLAUDE.md`](desktop/CLAUDE.md) | Every desktop module: sidecar supervision, the self-checkout loop and basket rules, the views and hooks |
| [`sidecar/tests/CLAUDE.md`](sidecar/tests/CLAUDE.md) | Gates outside `make test`, the Makefile and doc-command checks, the hand-synced contracts, mirror guards and wording contracts |

Docs for people (setup, specs, readiness, training) are mapped in [`README.md`](README.md#documentation-map).

## Commands

A root `Makefile` wraps both toolchains (requires GNU Make — on Windows use Git Bash/WSL, or `winget install GnuWin32.Make`). Run `make help` to list targets. Key ones:

```bash
make install              # desktop npm install + sidecar venv setup
make dev                  # electron-vite dev (spawns the sidecar) — needs `make sidecar-setup` first
make test                 # desktop vitest + sidecar pytest (fakes only — this is what CI runs)
make docs-sync            # rewrite the numbers the docs state from the code that owns them
make docs-sync-check      # report doc numbers that no longer match the code (exits nonzero)
make verify-clamp         # re-check the frame-clamp claims against real weights + frames
make verify-unsure        # re-measure the unsure-phantom rule's cost and coverage (also needs a camera)
make verify-live-layout   # measure the Live tab's one-screen promise in real Electron (builds first)
make build                # typecheck + electron-vite build
make lint                 # eslint --cache --max-warnings 0 on desktop
```

`make verify-clamp`, `make verify-unsure` (need an installed weight and staged negatives; the second
also a camera) and `make verify-live-layout` (needs a display) are deliberately outside `make test`;
why, and the CI jobs that run the Makefile and the doc-command checks, are in
[`sidecar/tests/CLAUDE.md`](sidecar/tests/CLAUDE.md).

### Sidecar (Python, in `sidecar/`)

```bash
cd sidecar
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt  # or plain venv/pip, see README.md
.venv/Scripts/python.exe run.py        # prints SIDECAR_PORT=<n>; picks a free port if 8765 is taken
.venv/Scripts/python.exe -m pytest -v  # full suite — runs entirely against fakes (no camera, GPU, network, or API key)
.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v            # single file
.venv/Scripts/python.exe -m pytest tests/test_pipeline.py::test_name -v # single test
```

Every sidecar command names the venv's interpreter — `.venv/bin/python` on Linux/macOS — rather than
a bare `python`, because the uv path this file leads with activates nothing, and a bare one is the
system interpreter, which cannot import the app's dependencies at all.

The default model is the trained grocery weight `sidecar/models/scanncart-grocery-v1.pt` with its
record `scanncart-grocery-v1.json`; weights are gitignored, so a fresh clone copies them in or trains
them. A stock model such as `yolo11n.pt`, once selected, is auto-downloaded by Ultralytics into
`sidecar/` — the process cwd, because a bare `active_model` name (what `presets.py` writes) is
resolved **by name** — not into `sidecar/models/`, which the Model picker reads. Those stock weights
are runtime artifacts, gitignored by `*.pt` and kept on disk so the offline promise does not depend
on the download; do not "tidy" them into `models/`, where the picker would list each one twice and
the built-in entry would resolve to a download instead of the local file.

### Desktop (Electron + React + TypeScript, in `desktop/`)

```bash
cd desktop
npm run dev                # electron-vite dev, full app (needs sidecar venv set up first)
npm test                   # vitest run (headless; sidecar is always faked)
npm run test:watch
npx vitest run src/renderer/src/hooks/useSidecarStream.test.tsx   # single file
npm run typecheck          # tsc --noEmit for both node (main/preload) and web (renderer) tsconfigs
npm run lint                # eslint --cache --max-warnings 0
npm run build               # typecheck + electron-vite build
npm run build:win / :mac / :linux   # package with electron-builder
```

## Architecture at a glance

### Sidecar (`sidecar/app/`) — details in [`sidecar/app/CLAUDE.md`](sidecar/app/CLAUDE.md)

- `main.py` — FastAPI app over one `AppState`; routes, the WS handshake (status, class list, inference verdict), capture start/stop lifecycle, live settings patches.
- `camera.py` — capture thread + size-1 newest-frame buffer, MSMF + MJPG for 60 fps, control writes on the capture thread, app-side auto exposure.
- `camera_caps.py` / `camera_search.py` / `camera_derive.py` — measured calibration: what the camera honours, what to set, the settings patch.
- `inference.py` — `YoloDetector` (Ultralytics `track()`, BoT-SORT) and the Roboflow remote detector behind one protocol.
- `acceptance.py` — the one accept/reject decision: class allowlist, frame-clamp, frame-filling and unsure-phantom filters.
- `pipeline.py` — the per-frame loop: infer once per frame, filter, preview, log tracks, push frames.
- `settings.py` / `settings_store.py` — defaults, persistence, hot-reloadable vs restart-required, resize-mode guess.
- `models.py` / `roster.py` — installed weights and their records; class-list checks per dataset generation.
- `logging_store.py` (SQLite), `dataset_status.py`, `inference_health.py`, `hardware.py`, `presets.py`, `cameras.py`, `roboflow.py`, `credentials.py`, `tracking.py`, `schemas.py`.
- `run.py` / `loops.py` — port handshake, and the selector event loop + concurrency bound every uvicorn server here uses.

### Dataset tools (`sidecar/tools/`) — details in [`sidecar/tools/CLAUDE.md`](sidecar/tools/CLAUDE.md)

`clean_v2.py` (ingest/stage/upload), `plan_split.py`, `build_dataset.py` (local merge), `dataset_doctor.py`
(gate before the GPU), `train_model.py` (train / `--val` / `--install`), `accept_v2.py`,
`generate_version.py`, `label_progress.py`, plus the shared vocabulary (`label_classes.py`,
`generations.py`, `workspace.py`). Never imported by the runtime; their output lives in the gitignored
workspace `sidecar/data/datasets/`.

### Desktop (`desktop/src/`) — details in [`desktop/CLAUDE.md`](desktop/CLAUDE.md)

- `main/sidecar.ts`, `main/sidecarHealth.ts`, `main/index.ts` — spawn/supervise the sidecar, its port and health, IPC.
- Self-checkout (main process): `transferStream.ts` → `transferState.ts`/`transferGeometry.ts` (deposit/removal rules over three zones) → `basketLedger.ts` → `posSession.ts` (bind, sync, backoff) → `pos.ts`/`posClient.ts`/`posConfig.ts` (pushcart-web transport and config); `cartState.ts` for counter mode.
- `preload/index.ts` — the only main↔renderer bridge (port, health, self-checkout calls).
- Renderer: `views/LiveView.tsx`, `views/AdminPanel.tsx`, `views/BasketTestView.tsx`; hooks `useSidecarStream`, `useSidecarSettings`, `useSidecarHealth`, `usePosState`; `lib/ws.ts`, `lib/api.ts`, `lib/settingsFields.ts`.

### Testing conventions

- Sidecar tests always use fakes (`FakeFrameSource`, injected detectors/clocks) — never a real camera or GPU. Don't add tests that assume Ultralytics/OpenCV hardware is present.
- Desktop tests inject `spawnFn`/`wsFactory`/`apiFactory`/`streamFactory` rather than mocking modules — follow that pattern (see `sidecar.test.ts`, `useSidecarStream.test.tsx`) when adding new injectable dependencies.
- Almost everything is constructor-injected (frame source, detector, clock, logging store) specifically so pytest can run the whole pipeline against fakes with no camera/GPU/network.
- The WS and settings contracts are duplicated by hand between `sidecar/app/schemas.py`+`settings_store.py` and `desktop/src/renderer/src/lib/api.ts`+`settingsFields.ts`+`settingsDefaults.ts` — there is no shared schema generation, so change both sides together. The full list of mirrored contracts and the `mirror`-marked tests that check them is in [`sidecar/tests/CLAUDE.md`](sidecar/tests/CLAUDE.md).
