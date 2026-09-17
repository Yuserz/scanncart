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
node .claude/skills/run-desktop/driver.mjs smoke      # launch, screenshot Live view + Admin panel
node .claude/skills/run-desktop/driver.mjs capture    # Start -> real YOLO frames -> stats/item log -> Stop
node .claude/skills/run-desktop/driver.mjs allowlist  # class allowlist field -> sidecar round-trip
```

- `smoke` verifies launch + sidecar REST (hardware info printed).
- `capture` runs real camera + YOLO inference (GPU). Needs a webcam attached;
  first use of a model downloads its weights. Takes ~1 min.
- `allowlist` drives the Admin class-allowlist field the way a user does —
  types `bottle, cup`, clicks Save — then asserts the sidecar received
  `["bottle","cup"]`, treats the field as hot-reloadable, and that the field is
  **not** drawn on the Live tuning card (a list field grouped on the live side
  renders as a range input bound to a string array). It PATCHes the original
  value back in a `finally`, so it is safe to rerun. This is the wiring check to
  run after touching `settingsFields.ts`, `AdminPanel.tsx`, or the settings
  schema — the unit tests cannot see a field that renders in the wrong view.
- Modes that assert print `PASS`/`FAIL` per check and **exit non-zero** when any
  fail, so they can gate a change.
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
  the pattern, and keep any copy of `driver.mjs` in this skill directory: it
  derives the repo root from its own path, so a copy sitting one level down
  (e.g. in `shots/`) resolves `desktop/` to the wrong directory and dies on the
  import with a bare MODULE_NOT_FOUND.
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
- **MSMF "device opened, zero frames" wedge (hit Sep 2026).** The StreamCam can
  report `isOpened()=True` while *every* `read()` fails — stderr fills with
  `cap_msmf.cpp: can't grab frame` / `-1072875772` (`0xC00D3704`). `open()`
  succeeding is not evidence of a working camera, and the device never recovers
  on its own. It follows a 1080p60 mode request (the StreamCam needs USB 3.0 for
  1080p60, and this host never delivered a frame in that mode — the port itself
  was never confirmed) and spreads across repeated open/close cycles until even
  DSHOW can't open the device. Ground truth is a bare
  `cv2.VideoCapture(0).read()` loop in isolation: if that yields no frame, the
  fix is a **physical unplug/replug** — ideally into a USB 3.0 port — because a
  software PnP disable/enable needs an elevated shell. Do not "fix" it by
  switching to DirectShow: `_default_capture` pins MSMF on purpose (60 fps at
  1080p versus ~15 fps on DSHOW) and only falls back when MSMF can't open at
  all. In the app the symptom is Start reporting success and the state sitting
  on `running` for ~3 s before an `error` status lands with a "stopped
  delivering frames" detail, which LiveView renders in the dismissible
  `[data-testid="live-error"]` banner. Shipped defaults are 640x480@30, which
  this camera streams fine, so the way to walk into it is applying the
  **`high_end` preset (1920x1080@60)** — check capture settings in Admin before
  driving `capture` mode, and after a replug re-check the Camera dropdown
  (`Detecting cameras…` / Rescan), since the index can shift.

## UI handles

`data-testid` attributes: `nav-live`, `nav-admin`, `state`, `conn`,
`preview-placeholder`, `stats`, `item-log`, `det-box`, `hardware-info`,
`save-settings`, `restore-defaults`, `live-error` (dismissible banner for a
capture that died, e.g. a stalled camera). Start/Stop button:
`button[aria-label="Start"]` / `button[aria-label="Stop"]`. Frames streaming =
`img.preview-img` exists.

Setting inputs use the bare setting key as their `id` (`#class_allowlist`,
`#capture_fps`), while the Live tuning card prefixes the same key with `tune-`
(`#tune-camera_exposure`) — that prefix is how the `allowlist` mode tells the
Admin field apart from a mis-placed tuning control.
