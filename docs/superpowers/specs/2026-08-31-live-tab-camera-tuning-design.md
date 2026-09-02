# Live Tab Camera Tuning — Design

Date: 2026-08-31
Status: Approved
Builds on `2026-08-30-camera-calibration-design.md`.

## Problem

Camera settings live in the Admin tab, one view away from the live feed they
change. Tuning brightness or exposure means editing a number in Admin,
switching to Live, and judging the result from memory of what it looked like
before.

Worse, the loop does not work at all today. Every camera-behaviour setting is
in `RESTART_REQUIRED_FIELDS` (`sidecar/app/settings_store.py:81`), so each
adjustment costs a full stop/start cycle — roughly 28–37 s on the Logitech
StreamCam, most of it spent reopening the device.

We want the controls that change what the camera sees to sit beside the image
they change, and to take effect while you watch.

## What the code already allows

The restart-required list is more conservative than the implementation
requires. Sorted by what each field would actually cost to make live:

| Tier | Fields | Why |
| --- | --- | --- |
| Already live | `infer_frame_skip`, `preview_height`, `preview_max_fps`, `track_expiry_s` | `Pipeline` re-reads them from `settings` every frame |
| Free | `conf_threshold` | `YoloDetector` reads `self._conf` on every `track()` call (`inference.py:105`); `RoboflowRemoteDetector` filters on it at `:237`. Restart-required only because no setter was ever wired. |
| Cheap | `camera_brightness`, `camera_exposure`, `camera_autofocus`, `camera_focus` | `camera.py:117-124` are plain `cap.set()` calls against an open handle. They sit inside `open()` only because that is where the settings were in scope. |
| Needs a reopen | `camera_index`, `capture_width`, `capture_height`, `capture_fps` | negotiated with the device at open time |
| Needs a detector rebuild | `active_model`, `device`, `imgsz`, `resize_mode`, all backend fields | baked into the detector at construction; `imgsz` is additionally fixed by the ONNX export |

The five fields in the "Free" and "Cheap" tiers are exactly the ones an
operator wants to tune against a live image. That is the feature.

## Decisions

1. **Scope.** Only live-adjustable settings move: the four camera controls,
   `conf_threshold`, and the four already-hot fields, plus the calibration
   card. Everything restart-required stays in Admin.
2. **Layout.** A collapsible "Camera tuning" card joins Performance and Item
   log in the existing side rail. The feed never moves or resizes when the
   card opens.
3. **Apply model.** Changes apply to the running camera immediately
   (debounced), but persist to `data/settings.json` only on an explicit Save.
   A Revert restores the last saved values.
4. **Calibration.** One button stops capture, runs the sweep behind a progress
   overlay, then restarts capture automatically and presents the
   recommendation for review.

## Approach

Widen the hot-reloadable set and give `PATCH /api/settings` a non-persisting
mode.

`_apply_settings_patch()` is already the single funnel every settings write
passes through, and it already `setattr`s onto the live `Settings` instance
that `Pipeline` holds by reference. It gains one responsibility: after the
setattr loop, push camera-control changes to the capture thread and confidence
to the detector.

### Alternatives rejected

**A separate `POST /api/camera/controls`** that writes to the device without
touching `Settings`. It leaves settings semantics alone, but creates two
different answers to "what is the current brightness" — the live device value
and the persisted one. Save then has to reconcile them, and every future
reader has to know which is authoritative. The cost is permanent.

**Relocating the form with no sidecar change.** Honest about effort, but it
ships a slower version of what already exists: a form beside a feed that goes
dark for 30 s on every nudge. It does not deliver the requirement.

## Design

### Sidecar

**`camera.py`.** Extract the control writes from `open()` into
`_write_controls(cap)`, called both there and by the capture thread. Add:

```python
def set_controls(self, **changes) -> None:
    """Queue control changes for the capture thread.

    cv2.VideoCapture is not thread-safe and _loop calls read() on a
    background thread, so set() from the FastAPI request thread would race
    it. The queue is drained between reads instead.
    """
    with self._controls_lock:
        self._pending_controls.update(changes)
```

`_loop` drains the dict after a successful read and merges the values onto
`self._brightness` and friends, so a later reopen keeps them. Costs at most
one frame of latency. Coalescing into a dict bounds the work to one apply per
loop pass even under a fast drag.

Ordering inside `_write_controls` is preserved from `open()`: autofocus off
before any focus value, or the device hunts away from what was just written.

**`inference.py`.** `set_conf(value)` on `YoloDetector` and
`RoboflowRemoteDetector` — one assignment each, since both read `self._conf`
per call. Invoked through `getattr(detector, "set_conf", None)`, matching the
established idiom in `_release()` for detectors that do and do not implement
`close()`.

**`settings_store.py`.** Move `conf_threshold`, `camera_brightness`,
`camera_exposure`, `camera_autofocus` and `camera_focus` from
`RESTART_REQUIRED_FIELDS` to `HOT_RELOADABLE_FIELDS`. `SettingsResponse`
already ships both sets to the renderer, so the "applies instantly" badges and
the save-gating follow automatically; no TypeScript mirror edit is needed for
the sets themselves.

**`main.py`.**

- `_apply_settings_patch(state, patch, persist=True)`.
- A `_push_live_settings(state, patch)` called after the setattr loop, routing
  camera keys to `state.source.set_controls(...)` and `conf_threshold` to
  `state.detector.set_conf(...)`, each guarded by `getattr` so fakes and
  detectors without the method are unaffected. `getattr(None, ..., None)` is
  also the idle case: with capture stopped there is no source or detector, and
  the setattr alone is the whole job.
- `PATCH /api/settings` accepts `?persist=false`.
- `POST /api/settings/save` persists whatever is currently in memory and
  returns a `SettingsResponse`.
- `GET /api/camera/profile` returns the stored profile for the current
  camera's device key. `camera_profiles.load_profiles()` exists but
  `main.py:59` imports only `save_profile` — profiles are written and never
  read back, so a calibration does not survive an app restart today. This
  closes that round trip, and is what tells the UI which controls a device
  honours. It returns 200 with a nullable `profile` field rather than the 404
  its `/apply` sibling uses: an uncalibrated camera is a normal state the card
  renders, not an error it handles.

### Renderer

**`components/CameraTuning.tsx`** — collapsible side-rail card, three
sections:

- *Image* — brightness, exposure, autofocus, focus. A slider is disabled with
  an explanatory hint when `profile.controls.<x>` is false. Focus is
  additionally disabled while autofocus is on, since writing a focus value
  under autofocus is meaningless.
- *Detection* — confidence threshold.
- *Stream* — frame skip, preview height, preview fps, track expiry.

The `/api/camera/quality` readout (brightness, sharpness, capture fps) moves
here from Admin and sits directly above the sliders. It is the objective
answer to "did that help", and its capture-fps number is the only warning
light for the exposure trap described under Risks.

The card header shows the active camera's name read-only, because the camera
is chosen in Admin and Calibrate needs a visible target.

**Capture state gets exactly one owner.** `LiveView` already drives capture
through `useSidecarStream`; `useSidecarSettings` polls `/api/health` for its
own `captureState`. Two hooks in one view means two answers to "are we
running". So `useSidecarSettings` accepts `{ pollHealth: false }` when hosted
inside `LiveView`, and `CameraTuning` receives `running`, `start` and `stop`
as props from the stream hook. Calibration's stop → calibrate → restart
sequence uses those same props, with the restart in a `finally` so a failed
sweep never strands the feed off.

**Dirty tracking needs its own baseline.** `PATCH ?persist=false` returns a
fresh `SettingsResponse`, and `AdminPanel`'s existing pattern resets its draft
whenever `settings` changes identity — which would clear the dirty state on
every slider tick. `CameraTuning` keeps a `savedRef` updated only on load and
on save. "Unsaved changes" is the diff against that; Revert is a
`persist: false` PATCH back to it.

**Slider writes are debounced 150 ms** on the trailing edge. A drag emits
dozens of events and each would otherwise be a full PATCH.

**Four fields are authored for the first time.** `camera_brightness`,
`camera_exposure`, `camera_autofocus` and `camera_focus` have no `FieldMeta`
entries today — they are settable by calibration and by hand-editing
`settings.json`, but invisible in the form. The exposure hint states the
log-base-2 seconds behaviour explicitly.

### Placement

`FieldGroup` gains `home: 'live' | 'admin'`. Both views filter the one list,
so a field cannot end up in both places or neither.

Admin keeps model, device, detector backend and Roboflow fields, presets,
hardware info, and the restart-required camera fields (`camera_index`,
capture resolution and fps, `imgsz`, `resize_mode`). It loses the nine
relocated fields, the calibration card, and the quality readout.

## Testing

Sidecar, against fakes as always — no camera, GPU or network:

- `set_controls` queues rather than writes: `cap.set` is untouched at call
  time, and the values have landed after one `_loop` pass.
- Controls set live survive a reopen.
- `set_conf` changes filtering with no detector reconstruction.
- `PATCH ?persist=false` mutates `state.settings` and leaves `settings.json`
  byte-identical.
- `POST /api/settings/save` persists what is in memory.
- The five moved fields no longer raise 409 while capture runs; every other
  restart-required field still does.
- `GET /api/camera/profile` returns a stored profile, and degrades cleanly
  when none exists for the current camera.

Desktop, via injected deps rather than module mocks:

- Sliders render disabled when `profile.controls.<x>` is false; focus is
  disabled while autofocus is on.
- A rapid drag produces one PATCH, not thirty.
- Dirty state survives the PATCH response — the `savedRef` regression, written
  down as a test.
- Revert restores the last saved values.
- Calibrate stops, calibrates and restarts, and still restarts when the sweep
  rejects.
- `AdminPanel` no longer renders the moved fields.
- `LiveView` reports a single capture state, with no duplicate health polling.

## Risks and accepted tradeoffs

**Save is global.** `save_settings()` writes the whole `Settings` object, so
pressing Save anywhere in Admin also persists unsaved Live tuning. Avoiding
that means per-field dirty tracking in the persistence layer — real complexity
for a single-operator prototype. Accepted: one settings object, one Save. The
consequence is that Revert may find nothing to revert after an unrelated Admin
save.

**Unsaved values are live truth.** Tuning exposure without saving, then
changing resolution and restarting capture, gives the reopened camera the
unsaved exposure. Only an app restart discards unsaved values. This is
deliberate — the in-memory object is what you are looking at.

**Exposure can destroy framerate.** Windows exposure is log-base-2 seconds, so
`-2` means a quarter-second per frame and caps the camera at 4 fps. Mitigated
by the adjacent capture-fps readout, which already carries this exact signal:
`/api/camera/quality` returns `verdicts.capture_fps`, computed against
`target_fps * 0.8`. The card surfaces that existing verdict beside the
exposure slider rather than introducing a second threshold to keep in sync.

**A camera with no profile shows every slider enabled.** Acceptable:
unsupported controls simply do nothing, and the quality readout shows that
nothing moved. Calibrating populates the profile.

**`cap.set()` may block on some backends**, stalling a read. Bounded by
coalescing changes into a single dict applied once per loop pass.

## Out of scope

- Making `camera_index`, capture resolution or `capture_fps` live. They
  require a device reopen by nature.
- Making `imgsz` live. The ONNX export fixes its input size.
- Retiring the Admin tab.
- Phase 3 of the calibration plan (the five-step wizard with near/far distance
  sampling), which remains planned and unstarted.
