---
name: run-desktop
description: Use when asked to run, launch, start, screenshot, test, or drive the SCANnCART Electron desktop app (the Live View / Admin Panel UI with its Python sidecar), or to verify a change works in the real app rather than in the test suites.
---

# Run the SCANnCART desktop app

Electron app in `desktop/` that spawns the Python sidecar in `sidecar/` on
startup. For agent use, drive it with the Playwright script at
`.claude/skills/run-desktop/driver.mjs` — this is a Windows host with a real
desktop, so no xvfb; windows appear on the user's screen.

All commands run from the repo root.

## Prerequisites

- `sidecar/.venv` must exist (see `sidecar/README.md` / `make sidecar-setup`).
- `desktop/node_modules` must exist (`cd desktop && npm install`).
- `playwright-core` and `electron` are already devDependencies of `desktop/`.

## Build

```bash
cd desktop && npm run build   # typecheck + electron-vite build -> desktop/out/
```

Rebuild after any `desktop/src` change — the driver launches the built output,
not the dev server.

## Run (agent path)

```bash
node .claude/skills/run-desktop/driver.mjs smoke     # launch, screenshot Live view + Admin panel
node .claude/skills/run-desktop/driver.mjs capture   # Start -> real YOLO frames -> stats/item log -> Stop
```

- `smoke` verifies launch + sidecar REST (hardware info printed).
- `capture` runs real camera + YOLO inference (GPU). Needs a webcam attached;
  first use of a model downloads its weights. Takes ~1 min.
- Screenshots → `.claude/skills/run-desktop/shots/` (override `SCREENSHOT_DIR`).
  **Read the screenshots** — text output alone doesn't prove the UI rendered.

## Run (human path)

```bash
cd desktop && npm run dev   # electron-vite dev with HMR, opens a window
```

## Gotchas (all actually hit)

- **Launch Electron with the `desktop/` dir, not `out/main/index.js`.**
  `package.json` `main` points at the built output, and `app.getAppPath()`
  must be `desktop/` for the main process to resolve `../sidecar` (venv
  python + `run.py`). Launching the JS file directly breaks sidecar spawning.
- **`import 'playwright-core'` fails outside `desktop/`** — the driver uses
  `createRequire(desktop/package.json)` to resolve it. Keep that if you copy
  the pattern.
- **A dead sidecar hangs the app forever.** `main/index.ts` has no sidecar
  auto-restart; if the Python process dies before printing `SIDECAR_PORT=`,
  the renderer polls for a port indefinitely and `[data-testid="nav-live"]`
  never appears. The driver retries the whole launch up to 4×.
- **This machine intermittently kills processes during native DLL loads**
  (observed July 2026: even `import ctypes` died ~20% of tries in bad windows,
  correlating with `LiveKernelEvent` 141 GPU resets in the Application event
  log). If all 4 launch attempts fail, check
  `Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Windows Error Reporting'}`
  and recommend a reboot — it's the machine, not the code.
- **Wait on `[data-testid="nav-live"]`, not the window.** The window and
  renderer HTML load instantly; the AppShell only mounts after the sidecar
  port handshake completes.
- **MSMF "opened but zero frames" wedge (hit Sep 2026).** The StreamCam can
  report `isOpened()=True` while every `read()` fails (`cap_msmf.cpp: can't
  grab frame, error -1072875772`). It starts with 1080p60 mode switches on a
  USB 2.0 link (the StreamCam needs USB 3.0 for 1080p60) and spreads with
  repeated open/close cycles until even DSHOW can't open the device. A bare
  `VideoCapture(0).read()` loop is the ground truth — if it yields nothing,
  the machine needs a **physical unplug/replug** (PnP disable/enable needs an
  elevated shell), not a code fix. The sidecar now guards this: start waits
  up to `AppState.frame_wait_s` (6s) for a real frame, else answers 503 with
  recovery guidance, which LiveView shows as a red "Start failed" banner
  (`[data-testid="start-error"]`) while the state stays idle. With default
  1920x1080@60 settings on a USB 2.0 port, expect the banner on first Start —
  drop capture settings to 640x480@30 in the Admin panel before driving
  `capture` mode.

## UI handles

`data-testid` attributes: `nav-live`, `nav-admin`, `state`, `conn`,
`preview-placeholder`, `stats`, `item-log`, `det-box`, `hardware-info`,
`save-settings`, `restore-defaults`, `start-error` (red banner when the
sidecar rejects Start, e.g. dead camera). Start/Stop button:
`button[aria-label="Start"]` / `button[aria-label="Stop"]`. Frames streaming =
`img.preview-img` exists.
