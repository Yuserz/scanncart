# SCANnCART Desktop (Phase 2 — Electron shell + Live View)

Electron + React + TypeScript UI that spawns and supervises the Python
[sidecar](../sidecar/README.md), then renders its live 720p WebSocket preview with
detection box overlays, a start/stop control, a stats strip, and an in-memory item
log. Scaffolded with electron-vite (React 19, Vite 7, Electron 39).

## Architecture

- **Main** (`src/main/`): `SidecarSupervisor` spawns `sidecar/run.py`, reads the
  `SIDECAR_PORT=<n>` line from its stdout, and exposes the port to the renderer via
  `ipcMain.handle('sidecar:port')`. The child is killed on quit.
- **Preload** (`src/preload/`): `contextBridge` exposes `window.api.getSidecarPort()`.
- **Renderer** (`src/renderer/src/`): `App` polls for the port, then mounts
  `LiveView`, which connects **directly** to `ws://127.0.0.1:<port>/ws/stream` and
  `http://127.0.0.1:<port>/api/...` (the frame stream is not proxied through IPC).

## Setup

```bash
cd desktop
npm install
```

## Test (headless — no display, camera, or model)

```bash
npm test          # vitest: 25 tests (renderer + main), all with fakes
npm run build     # typecheck (node + web) + bundle all three targets
```

## Run the full app (manual — needs the StreamCam)

The main process expects the sidecar's local venv. Ensure the sidecar is set up
(`../sidecar/README.md`), then:

```bash
npm run dev
```

Override the sidecar location if needed:

```bash
SIDECAR_PYTHON=/path/to/python SIDECAR_SCRIPT=/path/to/run.py npm run dev
```

Click **Start** to begin capture; live boxes render on the StreamCam feed.

> **Known env note:** on the current dev machine, the Electron binary postinstall
> did not complete (`node_modules/electron/dist` left partial → "Electron failed to
> install correctly"). Re-run `npm install` on a stable connection, or
> `rm -rf node_modules/electron ~/.cache/electron && npm install`, before `npm run dev`.
> This affects only launching the GUI; `npm test` and `npm run build` are unaffected.
