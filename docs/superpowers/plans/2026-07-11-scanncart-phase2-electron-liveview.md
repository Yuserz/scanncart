# SCANnCART Phase 2 — Electron Shell + Live View Implementation Plan

> Executed in-session (not a cold handoff), so this is a task/interface breakdown
> rather than full-code-in-plan. Consumes the **Phase 1 sidecar contract unchanged**
> (spec §4): `ws://127.0.0.1:<port>/ws/stream` + REST `/api/health|/api/capture/start|stop`.

**Goal:** An Electron + React desktop app that spawns and supervises the Python
sidecar, then renders the live 720p WebSocket preview with detection box overlays,
a start/stop control, a stats strip, and an in-memory item log — verifiable via
headless unit tests with **no display, no camera, no model**.

**Scope (spec §10.2 only):** Electron shell + Live View. **Out of Phase 2:**
SQLite/`/api/logs` reconciliation (Phase 3), Admin panel (Phase 4), reconnect/respawn
hardening + OpenVINO + error banners (Phase 5). Basic spawn + WS reconnect *are* in
Phase 2 because the renderer mounts before the sidecar is ready.

## Stack (from official electron-vite scaffold; tool-picked versions)

Electron 39, React 19, Vite 7, electron-vite 5, TypeScript 5.9, Vitest 4 + React
Testing Library 16 + jsdom. Layout follows electron-vite convention
(`src/main`, `src/preload`, `src/renderer/src`), adapting the spec §2.2 illustrative
`desktop/electron` + `desktop/src` paths.

## Global constraints

- Renderer talks **directly** to the sidecar over browser-native `WebSocket`/`fetch`.
  Preload's only job is to hand the renderer the sidecar port + app readiness; the
  frame stream is **not** proxied through IPC.
- Boxes are normalized 0–1; overlay geometry must scale to the **actual rendered
  image dimensions** (JPEG letterboxes inside its container under `object-fit`),
  never the container size.
- The port handoff (main reads `SIDECAR_PORT=` from sidecar stdout → renderer gets it
  via IPC) plus WS reconnect must tolerate "sidecar not ready yet."
- Verification bar: headless unit tests green (jsdom/RTL, mock fetch/WS, pure geometry,
  injected spawn factory). Real Electron window rendering **live camera frames** is the
  user's manual step (same camera+model path deferred in Phase 1).

---

### Task 1 — REST client `src/renderer/src/lib/api.ts`

- **Produces:** `createApiClient(port: number)` → `{ health(), start(), stop() }`,
  each `fetch`-ing `http://127.0.0.1:<port>/api/...` and returning parsed JSON.
  `health()` → `{ state, active_model, device }`; `start()/stop()` POST → `{ state }`.
- **Test (mock `fetch`):** correct URL/method per call; JSON parsed; non-OK response
  rejects with a useful error.

### Task 2 — WebSocket client `src/renderer/src/lib/ws.ts`

- **Produces:** `createStreamClient({ port, onFrame, onStatus, onOpen, onClose, wsFactory? })`
  → `{ connect(), close() }`. Parses `type:"frame"` / `type:"status"` messages;
  auto-reconnects with capped backoff; `wsFactory` injectable (defaults to global
  `WebSocket`) so tests supply a fake.
- **Test (fake WebSocket):** frame message → `onFrame(payload)`; status message →
  `onStatus`; malformed JSON ignored (no throw); `close()` stops reconnect;
  a simulated drop schedules a reconnect.

### Task 3 — Overlay geometry `src/renderer/src/lib/overlay.ts`

- **Produces:** `boxToPixels(box, dispW, dispH)` and `layoutDetections(dets, dispW, dispH)`
  → pixel rects `{x,y,w,h,label}` using **explicit displayed image w/h**.
- **Test:** normalized `[0.5,0.5,1,1]` on 200×100 → `{x:100,y:50,w:100,h:50}`; label
  is `"cls conf%"`; empty list → empty.

### Task 4 — LiveView + hook `src/renderer/src/views/LiveView.tsx` + `hooks/useSidecarStream.ts`

- **Produces:** `useSidecarStream(port)` wiring Task 1+2 → `{ frame, status, items,
  start, stop, connected }` (items deduped in-memory by `track_id` for the session).
  `LiveView` renders the preview `<img src="data:image/jpeg;base64,...">`, a `<canvas>`
  box overlay (Task 3), Start/Stop button (Task 1), stats strip, and item-log list.
- **Test (RTL, inject fakes):** dispatching a frame renders detections/stats and a
  new item-log row; Start button calls `start()`; a duplicate `track_id` does not add
  a second row.

### Task 5 — Sidecar supervisor `src/main/sidecar.ts`

- **Produces:** `SidecarSupervisor({ spawnFn, pythonPath, scriptPath, onPort, onExit })`
  with `start()/stop()`. Spawns the sidecar, scans stdout line-by-line for
  `SIDECAR_PORT=<n>` → `onPort(n)`; on unexpected exit calls `onExit` (basic respawn
  hook). `spawnFn` injectable (defaults to `child_process.spawn`).
- **Test (fake spawn returning a scripted stdout stream):** emits `onPort(8765)` when
  the line appears; buffers partial lines; `stop()` kills the child; exit triggers `onExit`.

### Task 6 — Wire main + preload + renderer, build, smoke

- `src/main/index.ts`: construct supervisor on `app.whenReady`, hold the resolved port,
  expose it via `ipcMain.handle('sidecar:port', …)` (returns null until ready), kill the
  child on `window-all-closed`/`before-quit`.
- `src/preload/index.ts`: `contextBridge` expose `api.getSidecarPort()`; typed in
  `index.d.ts`.
- Renderer `App.tsx`: poll `getSidecarPort()` until non-null, then mount `LiveView`.
- **Verify:** `npm run build` (typecheck+bundle) green; full `vitest run` green.
  Optional boot smoke (window opens, renderer mounts) — live-frame render is the user's step.

---

## Definition of Done (Phase 2)

- `npm test` (vitest) green — no display/camera/model.
- `npm run build` green (typecheck + bundle main/preload/renderer).
- Manual step (user): `python sidecar/run.py` running, launch the app, click Start,
  see live boxes on the StreamCam feed.
- Contract with sidecar unchanged; Phase 3 (logging panel) layers on `/api/logs`.
