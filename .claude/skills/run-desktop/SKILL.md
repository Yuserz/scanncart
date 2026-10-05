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
node .claude/skills/run-desktop/driver.mjs capture    # Start -> real YOLO frames -> stats/item log -> reload mid-capture -> Stop
node .claude/skills/run-desktop/driver.mjs allowlist  # class allowlist field -> sidecar round-trip
node .claude/skills/run-desktop/driver.mjs dataset    # Admin dataset panel <-> sidecar snapshot
node .claude/skills/run-desktop/driver.mjs models     # installed weights + resize_mode check <-> sidecar
node .claude/skills/run-desktop/driver.mjs classlist  # a NON-ROSTER weight: listing flag -> Live banner + count chip
node .claude/skills/run-desktop/driver.mjs probe      # Test Connection geometry <-> a stub workflow
node .claude/skills/run-desktop/driver.mjs v1         # the v1 acceptance run: weights record -> Live strip -> item log
```

- `basket` drives the **Basket test** tab end to end: Start camera, both zones painted over the
  picture, a practice basket bound with no tablet (and the main process agreeing), a 20 s window
  in which a real deposit can be made and is reported (never asserted — nothing here moves a
  product), **capture still streaming with no error after that window**, Empty and restart, End,
  then the zone editor: B with four clicks per outline (a numbered marker from the first click, every point dragged with the real mouse and the drag undone and redone), Save read back through the POS config
  (points in true orientation), an empty outline refused with Save disabled, Revert, and back to
  A. It ends on the Live view checking that no Camera tuning row draws a control over its own
  label. The original zone fields are restored in a `finally`. The "still streaming" check exists
  because of a real failure (Oct 2026): with POS configured, its loop called `/api/capture/start`
  while the renderer's start was still opening the camera, the sidecar acquired a second capture
  beside the first, and one's error teardown closed the other's detector a few seconds in —
  `'NoneType' object has no attribute 'names'` — after "frames reached the preview" had already
  passed. Starts are serialized now (`start_lock` in `app/main.py`).
- `smoke` verifies launch + sidecar REST (hardware info printed).
- `probe` is the remote-backend wiring check, and the only one that needs a fake *outside* the
  app: `local_api` points at a self-hosted workflow endpoint, and this machine has neither an API
  key nor an inference server running, so the mode **starts a stub workflow server** on a free
  port, PATCHes `detector_backend: local_api` + `local_api_url` at it (and a 16:9 capture with a
  640-px transmit limit, so the sent geometry is a real shape), remounts Admin, and clicks **Test
  connection** three times with the stub answering differently each time: echoing the sent size
  (`data-state="same"`), re-framing to a square canvas (`resized`, and both sizes named), and
  omitting the size block entirely (`unreported` — which must *not* read as agreement). It asserts
  the app-side size is the one a real capture would send (640×360 from 1280×720), which is what
  proves the probe sends a capture-shaped frame through the real downscale rule. The stub answers
  the Phase 0 payload shape (docs/DETECTOR_BACKENDS.md §0); the server is closed and the settings
  restored in a `finally`. Use it after touching the probe route, `inference.py`'s
  `RemoteGeometry`, `lib/api.ts`'s `DetectorProbeResponse`, or the Admin Panel's
  `probe-geometry` line. It works without Docker and without an API key, which is also the
  cheapest way to confirm the `local_api` path end to end.
- `capture` runs real camera + YOLO inference (GPU). Needs a webcam attached;
  first use of a model downloads its weights. Takes ~1 min. Once frames are
  flowing it **reloads the page mid-capture** and requires the toolbar to come
  back saying `running` — the renderer has no memory of the Start it already
  did, so that only passes if the sidecar's handshake status is working. Use it
  after touching `main.py`'s `/ws/stream` route, `lib/ws.ts`, or
  `useSidecarStream.ts`.
- `allowlist` drives the Admin class-allowlist field the way a user does —
  types `bottle, cup`, clicks Save — then asserts the sidecar received
  `["bottle","cup"]`, treats the field as hot-reloadable, and that the field is
  **not** drawn on the Live tuning card (a list field grouped on the live side
  renders as a range input bound to a string array). It PATCHes the original
  value back in a `finally`, so it is safe to rerun. This is the wiring check to
  run after touching `settingsFields.ts`, `AdminPanel.tsx`, or the settings
  schema — the unit tests cannot see a field that renders in the wrong view.
- `dataset` checks the Admin Panel's **Dataset labeling** section against the
  sidecar's own `/api/dataset/status` JSON — on-screen totals, the **labeling
  worklist** (one row per cell still unlabeled, *cell for cell and in order*, with
  its own remaining count and the pair it came from, the null-marking rule on the
  hard negatives and nowhere else), a freshness line, and the **Tier A capture gap**
  (its totals and one row per cell still under target). That join is what unit tests
  cannot see: they can render the panel from a fake, or fetch the route, but not
  both. Use it after touching `AdminPanel.tsx`, `useDatasetStatus.ts`, or
  `sidecar/app/dataset_status.py`. It also fails if the section is missing, since a
  render error there takes the whole Admin view down with it.
  The worklist's order is checked by *name plus distance*, not by count: the right
  rows in the wrong order is the failure this guards (a teammate sent at the wrong
  cell first), and a length check would pass through a reversal. A finished backlog
  is the one state this machine cannot be put into, so it is asserted by what the
  panel renders — no list, the empty line — rather than set up.
- `models` checks the Model field's **installed weights** block against the
  sidecar's own `/api/models` JSON: that each listed weight is named with the
  `resize_mode` recorded beside it, and that `auto` answers with that same
  requirement — the feature's acceptance criterion, and the inverse of what this
  mode used to assert. It then overrides the record by hand (`letterbox`) and
  requires the warning to appear, and requires it to clear again on `stretch`, so
  a warning that never fired and one that never went away cannot both pass. It
  also checks the **measured recall**: the split and floor, the per-class rows
  with their instance counts, and an unmeasured class rendering as *not measured*
  rather than as `0.000` — the one place the JSON-vs-DOM join can be wrong
  without either side complaining. Then the **class × distance grid**: each column
  named with the frame count it ran on, a class missing the floor at `far`
  (0.550) sitting beside the split number that hides it (0.900), the miss named in
  the line under the table, and a dash — never a `0.000` — for a distance that held
  no instances of a class or never scored it at all. The scratch record carries two
  distances so both readings exist in one render.
  Nothing is installed on a fresh machine, so it writes a scratch weight and a
  record with `train_model.weight_record` + `train_model.validation_record` (the real
  writers — a hand-built JSON would agree with the reader by construction), then
  removes both in a `finally` and restores the settings. It finishes in the Live view,
  which it visits twice. The first visit checks the **stats-strip readout**: that the
  strip prints the geometry `auto` resolves to ("stretch", labelled as coming from
  `auto`) rather than the word `auto`, that the measured recall is beside it with the
  split it was measured on, and that a geometry the record agrees with is not flagged —
  the JSON-vs-DOM join this whole feature is about. The second PATCHes a **genuinely
  mismatched** saved setting (`letterbox`) and remounts the view, then requires the
  running-gated banner to be **absent while idle** while the strip's geometry tile is
  present and flagged — so "quiet" is the gate doing its work rather than there being
  nothing to warn about, and the tile cannot quietly drop the state it exists to report.
  That patch is what makes the idle check mean something: draft edits in the Admin form
  never leave the form, so a mismatch there is not a mismatch the Live view can see. The
  banner's *appearance* needs a live capture and is covered in `LiveView.test.tsx`.
  It then covers the case no mismatch check can reach — the **unrecorded** weight, where
  `auto` falls back to the format heuristic and the geometry is an assumption rather than
  a fact. It deletes the scratch record and requires the sidecar's entry to appear
  in the panel's server-warning list (`has no record of the geometry`, naming `--install` —
  the *remedy* half of the entry, so this also proves both sentences survive the wire),
  then writes the record back and requires it to clear — the same appear/clear control, on
  the half of the feature that has no record to contradict and so cannot be flagged as a
  mismatch. It patches `detector_backend: native` for that step, since the entry is
  native-only and inheriting the machine's saved backend would make the check about a
  setting it is not about.
  While the record is still missing it visits the Live view and requires
  `live-assumed-geometry` to be **absent while idle** with the geometry tile present and
  naming `letterbox` — the same gate-then-readout pair as the mismatch banner, and the
  only half of it a machine without a camera can check. That reading has to happen before
  the button below writes the record, since the state cannot be produced again afterwards
  without deleting the file a second time. The banner's *appearance* needs a live capture
  and is covered in `LiveView.test.tsx`. It also reads the strip's **`stat-requirement`**
  in both states — `assumed` (dim, while the record is missing) and `recorded` (naming the
  tool's `stretch` once it is back) — because that tile is *not* gated on capture running
  and so is the half of this feature a camera-less machine can check end to end. The
  Live view's own record-it-now button sits inside the banner, which this mode never
  reaches because it runs no capture — driving it needs a live capture *and* an
  unrecorded weight (the scratch fixture above, with `capture` mode's frame wait), so it
  is covered in `LiveView.test.tsx` instead. Its write path is the same route, and the
  Admin button below drives that for real.
  While that entry is on screen it **clicks the panel's `record-resize-mode` button**, which
  is the only place the write can be verified end to end: the mode travels from the sidecar's
  entry through the button to a real `models/<stem>.json` on disk. It waits on the entry
  changing rather than on the click, then asserts the file exists with
  `resize_mode: letterbox`, that the warning is replaced by the confirmation, that the weights
  list now says `auto` uses it (the acknowledgement is read off the refreshed listing, not
  remembered), and that the machine's `settings.json` was not touched. The tool's record is
  then written back over the app's for the clearing check. Use
  it after touching `AdminPanel.tsx`, `LiveView.tsx`, `useSidecarSettings.ts`,
  `useActiveWeights.ts`, `lib/resizeMode.ts`, `sidecar/app/models.py`,
  `sidecar/app/settings_store.py`, or the record `train_model.py --install` writes.
- `classlist` is the roster guard end to end, against a weight that really is non-roster — the
  one thing neither unit suite can do, since they can render the banner from a fake status or
  serve `/api/models` from a temp directory, but not put a real checkpoint through ultralytics in
  one process and read the real renderer's response to it. It **builds the scratch weight with
  ultralytics itself** (`DetectionModel('yolo11n.yaml', nc=21)` + the 21 distance-split names —
  one head output per product and distance, built from the generation's own roster rather than
  from a product list written out in this skill),
  because the failure is a *head* trained per product-and-distance: a stock `.pt` with a renamed
  class list would give the app class ids that do not resolve in the names dict it indexes by
  (`normalize_detections`), so the run would die on a `KeyError` instead of reporting a class
  list. The weights are untrained and that is fine — nothing here is about boxes. The record is
  written by the real `train_model.weight_record(class_names=…)`.
  It then checks both halves, in the order the app learns them: the **listing** (the sidecar's
  `/api/models` flags it, and the Admin Panel's *Weights on disk* row shows `21 classes` plus the
  sidecar's own distance sentence) and the **running capture** (Start → the Live view's
  `live-class-warnings` banner, and the `stat-classes` chip reading `21` / `1 finding` in amber).
  Both are also checked for being *painted* — non-zero box, inside the viewport, not hidden — since
  `textContent` cannot tell a rendered element from one with a zero-height box, and that is what
  the screenshots are there to corroborate by eye. Needs a **camera** that delivers frames (the
  banner only exists once the pipeline has inferred, and no mode can fake that); with none it says
  so and the banner checks fail rather than silently passing. It restores the settings and deletes
  the weight and record in a `finally`, and it stops capture before restoring, because
  `active_model` is restart-required and a 409 would otherwise leave the machine pointing at a
  deleted weight.
- `v1` is the acceptance run for the locally trained v1 weight, and it exists because of a real
  failure (2026-09-23): v1 measured 0.918 / 41 fps through `spec_check.py --defaults` while the app
  was running the same weight at `imgsz` 960 — 0.344 recall, 24 fps — and nothing recorded which
  profile was the right one. The tools cannot see that, because they read `settings.json`
  themselves; only the running renderer knows what it is running. So it asserts against the app:
  that `auto` resolves to the record's own mode and the tile says `geometry (auto)`, that the
  requirement chip reads `recorded` rather than `assumed`, and that the strip's score is **the
  record's** mean with the below-floor class named rather than averaged in (`94%`,
  `test recall · 1 below floor`). Every expectation is derived from `/api/models`, so nothing here
  is a second copy of v1's class list agreeing with itself by construction. Then the core check:
  Start, and **every class in the item log is one of the seven names in that weight's record** —
  plus the running model's verdict has to agree with the record's own finding count — for v1 that
  is **no** findings, since its seven names are complete for its own generation, so the chip must
  read `roster ok` and the banner must be absent — and never the `carry a distance` sentence,
  which would mean a head trained per product-and-distance. Last, the PRD's two live promises off
  the strip: >= 30 infer fps and
  < 150 ms, which are the tiles that fall when `imgsz` is wrong. It leaves `imgsz` alone on purpose
  (that is the knob that broke this weight, so the fps check is how a wrong one shows up), and
  stops capture before restoring `active_model` + `resize_mode` in a `finally`, since both are
  restart-required and a restore under a running capture is a 409. Needs a camera delivering frames
  *and* a product in front of it: with none it says so and fails the checks it cannot make rather
  than passing them quietly. Use it after touching `spec_check.py`, the weight record,
  `useActiveWeights.ts`, or `LiveView.tsx`'s stats strip.
- Modes that assert print `PASS`/`FAIL` per check and **exit non-zero** when any
  fail, so they can gate a change.
- Screenshots → `.claude/skills/run-desktop/shots/` (override `SCREENSHOT_DIR`).
  **Read the screenshots** — text output alone doesn't prove the UI rendered.
  A capture that stalls (the loading ring animates forever, and a screenshot waits
  for the page to settle) retries once with `animations: 'disabled'` and then warns;
  it never throws, so one stuck capture cannot hide the `PASS`/`FAIL` lines under it.

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
- **The API can outlive its own listening socket (fixed Sep 2026).** A sidecar
  process that is *alive* in `tasklist` — camera open, threads running, `curl`
  to `/api/health` simply hanging — with **nothing listening** on its port. The
  app looks frozen while everything on this side reports healthy. This was
  Windows-only and is now fixed: uvicorn's Windows default is
  `ProactorEventLoop`, and that loop's accept path closes the listening socket
  on **any** failed accept and never re-arms (`asyncio/proactor_events.py`,
  `BaseProactorEventLoop._start_serving`), so the first aborted connection — a
  renderer reloading mid-request, a probe cancelled when a window closes — logs
  `Accept failed on a socket` with `OSError: [WinError 64] The specified network
  name is no longer available` and ends the server for good. Every uvicorn server
  in the repo now hands uvicorn the selector loop instead (`app/loops.py` carries
  the mechanism and a three-line reproduction) — `run.py`, the annotator's server
  and the no-Docker inference server, which all had the same Windows default, and
  the annotator is the one where it hurts most: a labeling pass is a browser
  reloading pages at a server that has to outlive it. On the selector loop the same event is one logged line and the listener keeps accepting. Each of the
  three prints which loop it is on as it starts (`EVENT_LOOP=…`, beside the port
  it announces), so confirming the choice on a running build is a `grep` rather
  than an inference — and a startup log without that line is an older build. The
  sidecar's own stdout is forwarded to the main process's log with a `[sidecar]`
  prefix, so the line shows up as
  `[sidecar] EVENT_LOOP=SelectorLoop(selector=BatchedSelectSelector)` when the app
  is what launched it (uvicorn's own lines come through stderr, unprefixed, which
  is how the two are told apart in the same stream). The part in parentheses is
  the second half of the same fix and the one to look for if the app dies under
  load instead of hanging: `select()` takes a fixed number of descriptors — 512 in
  one call, 513 raising `ValueError: too many file descriptors in select()` out of
  the event loop — and a flood of connections that never sends anything is
  invisible to every request limit there is, so it is the *selector* that has to
  survive it. `BatchedSelectSelector` polls in batches and the same 520
  held-open connections that used to end a server now leave it listening; each
  server also caps concurrent connections (`SERVER_CONCURRENCY_LIMIT`, past which
  uvicorn answers **503** and closes), which is what a request flood sees. To
  recognise the shape on an older build, or to rule
  it out: the living process is the clue — `netstat -ano | findstr :8765` shows
  **no LISTENING row** while the pid is still in `tasklist`. The app no longer
  needs to be told either: the main process probes `/api/health` (three
  consecutive failures, each with its own deadline) and raises a
  `[data-testid="sidecar-unresponsive"]` notice over both views, which is the
  same shape stated on screen instead of in `netstat` — and a shed `503` counts
  as a failure there, which is the right call: a sidecar refusing to serve is not
  one this window can use. It names its own
  recovery — quit and start the app again, since nothing in the renderer can
  re-bind the socket — and a driver looking for a healthy launch should assert
  that notice is *absent*, because a wedged sidecar otherwise looks exactly like
  one that has not been used yet.
- **The backend's inference server can be silent too, and the app now says so.**
  `local_api` calls `sidecar/local_inference_server.py`, which nobody starts but
  you: it is not a child of the app, so a crashed or forgotten server leaves a
  run where the camera opens, frames stream, and not one item is ever logged —
  the detector's first frame is what fails, so the capture stops and
  `live-error` explains that without explaining *why*. The sidecar probes the
  configured URL in the background (`app/inference_health.py`; same
  transitions-only shape as the desktop's own health monitor, and **any HTTP
  response counts as an answer**, so a keyless workflow's 401 still reads as up)
  and pushes the verdict over the websocket. The Live view renders
  `[data-testid="inference-unresponsive"]` above the error banner, ungated by
  capture state and not dismissible, naming the two fixes (the local server's
  start command, which the sidecar writes into the verdict — its own
  `.venv-inference`, never a bare `python`, since that server's venv is not the
  one the sidecar runs in; or the `DETECTOR_BACKENDS.md` §7a setup step when
  that venv is not there yet — or `detector_backend` in Admin) plus the probe's
  own detail. The same verdict is on the **Admin Panel**, beside the backend picker
  (`[data-testid="inference-watch"]`), as the standing reading rather than a
  notice: the address actually being probed, the failure in the endpoint's own
  words, and how long ago it was checked. That is the line to read when the
  question is "wrong port, or server not started?" — the URL there is the *saved*
  setting, so a port typed into the field but not saved shows as two different
  addresses on one screen, and an age of minutes means the verdict predates the
  server you just started. Two consequences for a driver: an empty item log on a `local_api`
  run is a failure state rather than a quiet scene, so check that notice before
  believing an empty-counter measurement; and because it is also sent on connect
  (the handshake replays it, like the capture state and the class list), asserting
  it *absent* is a legitimate readiness check for `local_api` even though nothing
  in the window can probe the endpoint for itself.
- **A busier middle state: the same server, up and refusing.** A server that has
  answered *none* of the last few probes is the notice above; a server that is
  answering and turning requests away — its own connection cap (503), which is
  what `SERVER_CONCURRENCY_LIMIT` in `app/loops.py` exists to produce when
  something else is hammering it — is a different reading and must not be
  reported as the first one. The sidecar's client retries each refusal with
  backoff and drops only the frame, so the capture keeps running, and the dropped
  frame shows up as `[data-testid="stat-shed"]` in the Live stats strip (the
  suppressed tile's sibling, shown while it is happening). Treat it as a
  *partially* analysed run rather than a clean one: frames the server refused
  were never looked at, so an item log that is short or empty beside that tile is
  not evidence about the counter. Finding the flood is the fix — usually another
  tab or tool pointed at the same server — and it clears on its own.
- **This machine intermittently kills processes during native DLL loads**
  (observed July 2026: even `import ctypes` died ~20% of tries in bad windows,
  correlating with `LiveKernelEvent` 141 GPU resets in the Application event
  log). If all 4 launch attempts fail, check
  `Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Windows Error Reporting'}`
  and recommend a reboot — it's the machine, not the code.
- **Wait on `[data-testid="nav-live"]`, not the window.** The window and
  renderer HTML load instantly; the AppShell only mounts after the sidecar
  port handshake completes.
- **The Admin panel shows "Loading settings…" long after the app looks ready.**
  `useSidecarSettings.load()` awaits a `Promise.all` that includes
  `/api/system-info`, and that handler lazily imports **torch** on its first call.
  Every driver run spawns a *fresh* sidecar, so the import is cold each time and
  has been observed taking well over 30 s on this machine. The gate is the
  hardware probe, not the sidecar: `/api/dataset/status` and `/api/health` answer
  instantly in the same window. So wait for the panel to *settle* (the spinner to
  detach) rather than for any one section's selector — otherwise a slow import
  reads as "the panel is missing", which is what the `dataset` mode did on its
  first two runs.
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
  on `running` for ~3 s before an `error` status lands, which LiveView renders
  in the dismissible `[data-testid="live-error"]` banner and then tears capture
  back down to idle with Start re-enabled — no hang, no fake `running`.
  Observed verbatim on a wedged device: *"Capture stopped: Camera 0 stopped
  delivering frames for 3s (35 attempts) — it may have been unplugged,
  suspended, or taken by another program."* Shipped defaults are 640x480@30, which
  this camera streams fine, so the way to walk into it is applying the
  **`high_end` preset (1920x1080@60)** — check capture settings in Admin before
  driving `capture` mode, and after a replug re-check the Camera dropdown
  (`Detecting cameras…` / Rescan), since the index can shift.
- **A held device looks exactly like the MSMF wedge — check who holds the camera
  first (hit Oct 2026).** Another process owning the StreamCam produces the same
  symptoms as the staircase above: MSMF "opens" and every read fails with the
  same `-1072875772`, and after enough failed openers even DSHOW refuses. So
  before blaming MSMF or reaching for a replug, ask Windows who is using the
  webcam — it records this itself, per desktop app, under
  `HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged`:
  one key per program, and `LastUsedTimeStop = 0` means *in use right now* (a
  one-line `Get-ChildItem`/`Get-ItemProperty` read; no elevation, no device
  opens, answers in seconds). The calibration session that found this watched
  NVIDIA Broadcast take the camera ~20 s after a calibrate sweep released it —
  a replug changed nothing, killing Broadcast took the device from 0/15 frames
  to 20/20 immediately, and Broadcast then re-acquired it twice more (it lives
  in the tray and can relaunch), with Edge and Discord taking turns after it.
  Ordering matters: a held device can wedge MSMF for every later opener, so
  free the holder first, re-run the bare `read()` loop as ground truth, and
  only conclude "the staircase" when the registry shows no holder and frames
  still do not flow. Killing another program's process is the user's call,
  not the agent's — name the holder and ask.

## UI handles

`data-testid` attributes: `nav-live`, `nav-admin`, `state`, `conn`,
`preview-placeholder`, `stats`, `item-log`, `det-box`, `hardware-info`,
`save-settings`, `restore-defaults`, `live-error` (dismissible banner for a
capture that died, e.g. a stalled camera), `inference-unresponsive` (the notice
that the selected backend's server is not answering, above `live-error` in the
Live view, present without pressing anything), `sidecar-unresponsive` (the notice
that the sidecar is not answering, over both views), `inference-watch` (the standing
verdict beside the backend picker in the Admin Panel, with `inference-watch-url`
and `inference-watch-detail` inside it), and for the dataset panel
`dataset-progress` → `dataset-summary` / `dataset-unavailable` +
`dataset-backlog` (one `dataset-backlog-row` per cell, with `dataset-backlog-left`
and `dataset-backlog-null` inside them) / `dataset-backlog-summary` /
`dataset-backlog-done` / `dataset-distance` / `refresh-dataset`, with the capture gap in
`dataset-tier-a` → `dataset-tier-a-summary` + `dataset-tier-a-cells`, and for the
Model field `installed-models` (one `installed-model-<value>` per weight) +
`model-resize-mismatch` (only present while the setting disagrees) +
`unrecorded-resize-mode` (the assumption, with `record-resize-mode` — the button that
records it — inside it), `probe-result` (Test connection's line) and `probe-geometry`
(the remote geometry pair, with `data-state` of `same` / `resized` / `unreported`), and
below those
the measured score: `model-validation` → one `model-validation-test` per split, with
`model-validation-metrics` and one `model-validation-class-<name>` per class, or
`model-validation-missing` when the weights have a record but no measurement — plus, when
the record has a breakdown, `model-validation-by-distance` (the table) with one
`model-validation-<distance>-<class>` cell per class and distance, and
`model-validation-distance-misses` for the line naming what is under the floor at a distance. The Live
view's copies of the resize warning are `live-resize-mismatch` (an explicit setting
contradicting the record) and `live-assumed-geometry` (nothing recorded, so `auto` is
guessing) — both present only while capture runs, and mutually exclusive — the second
carrying its own `live-record-resize-mode` button and handing its slot to
`live-recorded` once a write lands. Also present only while
capture runs: `live-class-warnings` (the running model's class list judged against the roster of
that weight's own generation — v1's seven or v2's seven, the same names today, chosen by the
record and then by the names themselves — in the sidecar's own sentences; a 21-class head is not
an error state, so it
is not error-styled), and inside the listing above it
`installed-model-class-warning-<value>` for a weight whose *recorded* class list is wrong — the
same verdict one step earlier, before the weight is selected. The weights
readout in its stats strip is `stat-geometry` (the resolved mode,
`warn` when it contradicts the record) + `stat-recall` (the measured mean, `warn` when a
class is below the floor, `unmeasured` when nothing measured it) + `stat-requirement`
(the record's mode, `unmeasured` when nothing recorded it). All three are present while
idle too, and **absent** when the backend runs a remote model, since then these weights
are not what is running. `stat-classes` is the one tile on that strip describing the
*running* model, so it appears only once the sidecar has read a model's vocabulary and it
is present for a remote backend as well — `classes · roster ok`, or `classes · 1 finding`
(warn) with the sidecar's sentences in the title. Start/Stop button:
`button[aria-label="Start"]` / `button[aria-label="Stop"]`. Frames streaming =
`img.preview-img` exists.

Setting inputs use the bare setting key as their `id` (`#class_allowlist`,
`#capture_fps`), while the Live tuning card prefixes the same key with `tune-`
(`#tune-camera_exposure`) — that prefix is how the `allowlist` mode tells the
Admin field apart from a mis-placed tuning control.
