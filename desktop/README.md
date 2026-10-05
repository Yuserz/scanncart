# SCANnCART Desktop

Electron + React + TypeScript app that spawns and supervises the Python
[sidecar](../sidecar/README.md), renders its live WebSocket preview with detection
boxes, turns tracked boxes into the basket's deposits and removals, and syncs the
cart to pushcart-web. Built with electron-vite (React 19, Vite 7, Electron 39).

> Setting up the whole project from a fresh clone? Start with the step-by-step
> [development guide](../docs/DEVELOPMENT.md); this file covers desktop-specific
> setup and its known env notes.

## Architecture

- **Main** (`src/main/`):
  - `sidecar.ts` spawns `sidecar/run.py`, reads its `SIDECAR_PORT=<n>` line and kills
    the child on quit; `sidecarHealth.ts` notices a sidecar that is alive but no longer
    answering.
  - The self-checkout loop: `transferStream.ts` (its own WebSocket to the sidecar),
    `transferState.ts` / `transferGeometry.ts` (the deposit/removal rules over the
    outside/opening/inside zones), `basketLedger.ts` (the per-product ledger and review
    list), `posSession.ts` (binding to the tablet's session and syncing the cart) and
    `pos.ts` / `posClient.ts` / `posConfig.ts` (the pushcart-web transport and settings).
    It lives here, not in the renderer, because the renderer cannot reach pushcart-web
    and the shared secret must not travel into a page.
- **Preload** (`src/preload/`): `contextBridge` exposes the sidecar's port and health
  and the self-checkout calls (state, config, test connection, basket practice).
- **Renderer** (`src/renderer/src/`): `App` waits for the port, then mounts the
  **Live**, **Admin** and **Basket test** views, which talk **directly** to
  `ws://127.0.0.1:<port>/ws/stream` and `http://127.0.0.1:<port>/api/...`.

The module-by-module map is in [`../CLAUDE.md`](../CLAUDE.md).

## Setup

```bash
cd desktop
npm install
```

## Test (headless — no display, camera, or model)

```bash
npm test          # vitest: the whole renderer + main suite, all with fakes
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
SIDECAR_PYTHON=/path/to/sidecar/.venv/Scripts/python.exe SIDECAR_SCRIPT=/path/to/run.py npm run dev
```

Click **Start** to begin capture; live boxes render on the StreamCam feed. The
**Basket test** tab labels every box with its product, confidence and track id and
runs a practice basket with no tablet; self-checkout is configured in **Admin →
Self-checkout** ([`../docs/POS_INTEGRATION.md`](../docs/POS_INTEGRATION.md)).

> **Known env note — "Electron uninstall" / "Electron failed to install correctly" on `npm run dev`:**
> the Electron binary postinstall can leave `node_modules/electron/dist` partial
> (only `locales/`, no `electron.exe`) with no `path.txt`, so electron-vite throws
> `Error: Electron uninstall`. Re-running `npm install` does **not** reliably fix it:
> `@electron/get` reports a cache hit and the extract step can silently no-op
> (exit 0, nothing written) — likely the same Windows Defender interference
> documented in [`../sidecar/README.md`](../sidecar/README.md), so add the Defender
> exclusion for this repo first. Then rebuild:
>
> ```powershell
> Remove-Item -Recurse -Force node_modules/electron/dist
> npm rebuild electron   # or: node node_modules/electron/install.js
> ```
>
> If the extract still no-ops, do it by hand: the cached
> `%LOCALAPPDATA%\electron\Cache\<hash>\electron-v<ver>-win32-x64.zip` is valid —
> extract it into `node_modules/electron/dist/` and write `path.txt` containing
> `electron.exe`. Verify with `node -e "console.log(require('electron'))"` (prints
> the exe path) and `npx electron --version`. Affects only launching the GUI;
> `npm test` and `npm run build` are unaffected.
