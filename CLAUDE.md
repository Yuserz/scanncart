# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SCANnCART is a capstone prototype for grocery stores: a Logitech StreamCam feeds a Python sidecar that runs YOLO11 (Ultralytics) object tracking, and an Electron + React desktop app displays the live annotated feed and a session item log. Everything runs locally on one PC — no server, no cloud, no network dependency. See `docs/PRD.md` for the full product spec and `docs/DEPLOYMENT.md` for out-of-scope future work (edge hardware, cloud sync, etc).

Two independent toolchains live side by side and talk over localhost HTTP/WebSocket:

```
Logitech StreamCam ──USB──▶ sidecar/ (Python/FastAPI)  ──ws://127.0.0.1:<port>──▶ desktop/ (Electron/React)
                             OpenCV capture → YOLO11 track            live preview + overlay + item log
                             SQLite detection log                     REST for start/stop/health/logs
```

`desktop/src/main` spawns `sidecar/run.py` as a child process on app startup and shuts it down on quit; the renderer never launches the sidecar itself, it just discovers the port over IPC and talks to it directly.

## Commands

A root `Makefile` wraps both toolchains (requires GNU Make — on Windows use Git Bash/WSL, or `winget install GnuWin32.Make`). Run `make help` to list targets. Key ones:

```bash
make install              # desktop npm install + sidecar venv setup
make dev                  # electron-vite dev (spawns the sidecar) — needs `make sidecar-setup` first
make test                 # desktop vitest + sidecar pytest
make build                # typecheck + electron-vite build
make lint                 # eslint --cache on desktop
```

### Sidecar (Python, in `sidecar/`)

```bash
cd sidecar
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt  # or plain venv/pip, see README.md
python run.py                          # prints SIDECAR_PORT=<n>; picks a free port if 8765 is taken
python -m pytest -v                    # full suite — 80+ tests, all run against fakes (no camera, no network)
python -m pytest tests/test_pipeline.py -v            # single file
python -m pytest tests/test_pipeline.py::test_name -v # single test
```

On first capture start, Ultralytics auto-downloads `yolo11n.pt` if not already present.

### Desktop (Electron + React + TypeScript, in `desktop/`)

```bash
cd desktop
npm run dev                # electron-vite dev, full app (needs sidecar venv set up first)
npm test                   # vitest run (headless; sidecar is always faked)
npm run test:watch
npx vitest run src/renderer/src/hooks/useSidecarStream.test.tsx   # single file
npm run typecheck          # tsc --noEmit for both node (main/preload) and web (renderer) tsconfigs
npm run lint                # eslint --cache
npm run build               # typecheck + electron-vite build
npm run build:win / :mac / :linux   # package with electron-builder
```

## Architecture

### Sidecar (`sidecar/app/`)

- **`main.py`** — `build_app(state_factory)` builds the FastAPI app. All sidecar state (settings, active pipeline, WS clients, logging store, current session) lives in a single `AppState` dataclass instance closed over by the route handlers — there's no global state. Routes: `GET /api/health`, `GET`/`PATCH /api/settings`, `GET /api/system-info`, `GET /api/presets`, `POST /api/settings/preset`, `POST /api/capture/start`, `POST /api/capture/stop`, `GET /api/logs`, `WS /ws/stream`. `source_factory`/`detector_factory` on `AppState` are the injection points tests use to swap in fakes. Settings patches are always applied via `_apply_settings_patch()`, which `setattr`s onto the existing `Settings` instance (never replaces it — `Pipeline` holds it by reference) and rejects (`409`) changes to `RESTART_REQUIRED_FIELDS` while capture is running.
- **`camera.py`** — `CameraCapture` owns an OpenCV `VideoCapture` and runs a background thread writing into a `LatestFrameBuffer` (a lock-guarded size-1 slot: newest frame always wins, no queue buildup/backpressure). `FakeFrameSource` is the test double.
- **`inference.py`** — `YoloDetector` wraps `ultralytics.YOLO(...).track(..., persist=True)` for detection + multi-object tracking (stable `track_id`s across frames) and normalizes results to 0–1-relative boxes via `normalize_detections`.
- **`pipeline.py`** — `Pipeline` runs its own thread (`process_once` in a loop) that: pulls the latest frame, optionally skips frames (`infer_frame_skip`), runs inference, JPEG-encodes a preview (`encode_preview_jpeg`, resized to `preview_height`), logs detections, and pushes a `FrameMessage` via the injected `on_message` callback. Track lifecycle (`entered_at`/`left_at`) is derived here: a track not seen for `track_expiry_s` is considered "left" and resolved in the log; `resolve_open_tracks()` force-closes everything still open on capture stop. `infer_frame_skip`, `preview_height`, and `track_expiry_s` are all read fresh from `self._settings` every call, so mutating them in place on a running pipeline takes effect immediately (no restart) — everything else (`active_model`, `device`, camera params, `conf_threshold`) is baked into `source`/`detector` at construction and needs a stop/start cycle.
- **`logging_store.py`** — `LoggingStore` is the sole SQLite writer (one connection, one `threading.Lock`, since the FastAPI request thread and the pipeline thread both touch it). Schema: `sessions` (one row per start/stop capture cycle) and `detection_events` (one row per track per session, `max_conf` tracked across the track's lifetime, `left_at` NULL while still in frame).
- **`schemas.py`** — Pydantic models shared by the WS frame protocol and REST responses (`Detection`, `FrameMessage`, `StatusMessage`, `HealthResponse`, `LogEvent`, `LogsResponse`, `SettingsResponse`, `SettingsUpdateRequest`, `SystemInfoResponse`, `PresetsResponse`).
- **`settings.py`** — plain dataclass `Settings` with hardcoded defaults (no env/file loading — stays pure so its own tests never depend on disk state); `resolve_device("auto")` picks `cuda` if torch reports it available, else `cpu`.
- **`settings_store.py`** — `load_settings()`/`save_settings()` persist `Settings` to `data/settings.json` (atomic write via temp file + `os.replace`; a missing/corrupt file or a bad individual field falls back to defaults rather than crashing startup). Also owns `HOT_RELOADABLE_FIELDS`/`RESTART_REQUIRED_FIELDS` (the single source of truth for which settings a running pipeline picks up live) and `compute_warnings()` (soft warnings surfaced in `SettingsResponse`, e.g. an uncommon capture resolution or a frame-skip/track-expiry mismatch).
- **`hardware.py`** — `probe_hardware()` reports CPU count/RAM via `psutil` and GPU name/VRAM via a lazily-imported `torch.cuda` (same `try/except ImportError` shape as `resolve_device()`).
- **`presets.py`** — `PRESETS` (`low_end`/`mid_range`/`high_end`) are partial settings patches an admin panel can apply wholesale; `recommend_preset()` picks one from `probe_hardware()`'s output using simple CPU/RAM/VRAM thresholds.
- **`run.py`** — entrypoint. `pick_port(8765)` falls back to an OS-assigned free port if 8765 is busy, then prints `SIDECAR_PORT=<n>` to stdout — this is the handshake the Electron main process parses to learn where to connect.

Almost everything is constructor-injected (frame source, detector, clock, logging store) specifically so pytest can run the whole pipeline against fakes with no camera/GPU/network.

### Desktop (`desktop/src/`)

Standard electron-vite three-target layout: `main` (Node, Electron APIs), `preload` (context-bridge), `renderer` (React, sandboxed, no Node access).

- **`main/sidecar.ts`** — `SidecarSupervisor` spawns the sidecar (`spawnFn` is injectable for tests), parses `SIDECAR_PORT=<n>` off stdout line-by-line, and forwards stderr so uvicorn's own log pipe never fills up and blocks the child. Reports the port and unexpected exits via callbacks.
- **`main/index.ts`** — resolves the sidecar's python/script paths relative to the packaged app (overridable via `SIDECAR_PYTHON`/`SIDECAR_SCRIPT` env vars), starts the supervisor on `app.whenReady()`, exposes the current port to the renderer over `ipcMain.handle('sidecar:port', ...)`, and kills the sidecar child on `before-quit`.
- **`preload/index.ts`** — exposes `window.api.getSidecarPort()` via `contextBridge`; that's the *only* main↔renderer bridge. Everything else (REST calls, WebSocket) goes straight from the renderer to `127.0.0.1:<port>` — Electron IPC is used only to learn the port.
- **`renderer/src/lib/ws.ts`** — `createStreamClient` wraps `ws://127.0.0.1:<port>/ws/stream` with auto-reconnect (so the UI tolerates the sidecar not being up yet) and a structural `WSLike` type so tests can inject a fake WebSocket.
- **`renderer/src/lib/api.ts`** — thin REST client (`health`/`start`/`stop`/`getLogs`/`getSettings`/`updateSettings`/`getSystemInfo`/`getPresets`/`applyPreset`) hitting `http://127.0.0.1:<port>/api/*`; `request<T>()` supports `GET`/`POST`/`PATCH` with an optional JSON body.
- **`renderer/src/lib/settingsFields.ts`** / **`lib/settingsDefaults.ts`** — client-side mirrors of the sidecar's `ALLOWED_MODELS`/`ALLOWED_DEVICES` and `Settings` defaults, kept in sync by hand (same tradeoff as the WS contract below) — drives `AdminPanel`'s form and its "Restore Defaults" action. The server's `hot_reloadable_fields`/`restart_required_fields` on `SettingsResponse` remain the source of truth for which fields require stopping capture.
- **`renderer/src/hooks/useSidecarStream.ts`** — the core state hook: wires REST + WS into React state, dedupes detections into a session item log by `track_id` (one row per item, keeping the max confidence seen). On WS open it seeds from `GET /api/logs`, but only when `statusState === 'running'` — this is reconnect recovery, not "load history on launch"; a fresh/idle app starts with an empty log, and `start()` resets the dedup set for a new session.
- **`renderer/src/hooks/useSidecarSettings.ts`** — fetches settings/system-info/presets on mount and polls `GET /api/health` (reusing the existing endpoint rather than a second WebSocket) to track `captureState` for save-gating. Exposes `update`/`applyPreset`/`restoreDefaults`.
- **`views/LiveView.tsx`** — renders the JPEG preview, detection-box overlay (`lib/overlay.ts` converts normalized boxes to CSS percentages), fps/latency stats strip, and the item log; Start/Stop drives `start()`/`stop()` from the hook.
- **`views/AdminPanel.tsx`** — hardware info, preset picker (recommended preset highlighted), and a form over the 9 settings fields with inline tradeoff guidance and hot-reload/restart-required badges. Edits are held in local `draft` state until Save; Save is disabled (with a warning) if the pending diff touches a restart-required field while capture is running.
- **`components/AppShell.tsx`** — owns Live/Admin view-switching state (`useState`, no router dependency) and renders a small nav.
- **`App.tsx`** — polls `window.api.getSidecarPort()` until non-null (the sidecar reports its port asynchronously after spawn), then mounts `AppShell`.

### Testing conventions

- Sidecar tests always use fakes (`FakeFrameSource`, injected detectors/clocks) — never a real camera or GPU. Don't add tests that assume Ultralytics/OpenCV hardware is present.
- Desktop tests inject `spawnFn`/`wsFactory`/`apiFactory`/`streamFactory` rather than mocking modules — follow that pattern (see `sidecar.test.ts`, `useSidecarStream.test.tsx`) when adding new injectable dependencies.
- The WS message contract (`FrameMessage`/`StatusMessage`/`Detection`) — and now the settings contract (`SettingsPayload`/allowed models/allowed devices/defaults) — are duplicated by hand between `sidecar/app/schemas.py`+`settings_store.py` (Pydantic/Python) and `desktop/src/renderer/src/lib/api.ts`+`settingsFields.ts`+`settingsDefaults.ts` (TS). There's no shared schema generation, so keep them in sync manually when either protocol changes.
