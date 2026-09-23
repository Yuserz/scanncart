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
  ultralytics itself** (`DetectionModel('yolo11n.yaml', nc=24)` + the 24 distance-split names),
  because the failure is a *head* trained per product-and-distance: a stock `.pt` with a renamed
  class list would give the app class ids that do not resolve in the names dict it indexes by
  (`normalize_detections`), so the run would die on a `KeyError` instead of reporting a class
  list. The weights are untrained and that is fine — nothing here is about boxes. The record is
  written by the real `train_model.weight_record(class_names=…)`.
  It then checks both halves, in the order the app learns them: the **listing** (the sidecar's
  `/api/models` flags it, and the Admin Panel's *Weights on disk* row shows `24 classes` plus the
  sidecar's own distance sentence) and the **running capture** (Start → the Live view's
  `live-class-warnings` banner, and the `stat-classes` chip reading `24` / `1 finding` in amber).
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
  plus the running model's verdict has to be *v1's*, i.e. the record's own finding count (v1 cannot
  predict Palmolive, so one finding is correct) and never the `carry a distance` sentence, which
  would mean a 24-output head. Last, the PRD's two live promises off the strip: >= 30 infer fps and
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

## UI handles

`data-testid` attributes: `nav-live`, `nav-admin`, `state`, `conn`,
`preview-placeholder`, `stats`, `item-log`, `det-box`, `hardware-info`,
`save-settings`, `restore-defaults`, `live-error` (dismissible banner for a
capture that died, e.g. a stalled camera), and for the dataset panel
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
capture runs: `live-class-warnings` (the running model's class list judged against the 8-class
roster, in the sidecar's own sentences — a 24-class head is not an error state, so it is not
error-styled), and inside the listing above it
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
