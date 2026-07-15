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

## UI handles

`data-testid` attributes: `nav-live`, `nav-admin`, `state`, `conn`,
`preview-placeholder`, `stats`, `item-log`, `det-box`, `hardware-info`,
`save-settings`, `restore-defaults`. Start/Stop button:
`button[aria-label="Start"]` / `button[aria-label="Stop"]`. Frames streaming =
`img.preview-img` exists.
