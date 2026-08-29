# SCANnCART — Guided camera calibration — Design

**Date:** 2026-08-30
**Status:** Approved (pending spec review)
**Depends on:** existing `cameras.py`, `camera.py`, `pipeline.py`, `settings.py`,
`settings_store.py`, `schemas.py` (sidecar); `AdminPanel.tsx`, `settingsFields.ts`,
`api.ts` (desktop).

## Problem

The app configures a camera by asking for a resolution and framerate and hoping.
It never looks at the resulting image, and it never checks what the camera
actually delivered. A session on 2026-08-29 spent hours chasing "detection is
broken" and "tracking keeps disconnecting" through the model, the tracker, the
backend and the GPU. The cause was none of those:

| Symptom | Actual cause |
|---|---|
| 1% detection rate, tracks breaking | Frame brightness 23–39/255. Nothing was visible to detect. |
| "60 capture fps" in the UI | `capture_fps` reports the **requested** value. The camera delivered **12**. |
| Soft image, confidence pinned at 0.52–0.59 | Autofocus hunting; sharpness 4.8, drifting 6→31 on a static scene. |

Every one of those was diagnosable in seconds from the live feed. The app had
the frames and showed the user nothing.

## Key findings this design is built on

All measured on the target machine (Logitech StreamCam, Windows, OpenCV):

1. **The property getters lie.** Setting `CAP_PROP_EXPOSURE` on MSMF took
   delivered framerate from 12.3 → 30.3 fps while `cap.get()` still returned the
   old value and `AUTO_EXPOSURE` still read `0.0`. **Capability must be
   established by measuring the image and the frame rate, never by reading a
   property back.** This is also what makes the design brand-independent.
2. **Which controls work is backend-specific**, measured by image effect:

   | Control | MSMF | DSHOW |
   |---|---|---|
   | BRIGHTNESS | ✅ 87 → 182 | ✅ 6 → 93 |
   | EXPOSURE | ✅ 87 → 121 | ✅ 6 → 194 |
   | CONTRAST | ✅ | ✅ |
   | GAIN | ❌ none | ✅ 6 → 143 |
   | FOCUS | ❌ ignored | ✅ 0 → 30 |

3. **MSMF is the right backend** and already the default: 12–30 fps versus
   DSHOW's 4–10 fps at 720p, and the two controls that matter most (brightness,
   exposure) work on it. Focus is the one thing it will not set.
4. **The StreamCam's Smart Auto-Focus/Exposure is face-tracking based**
   (RightLight 3 targets faces in low light). A checkout counter contains no
   face, so the camera's auto systems have nothing to lock onto. **Auto is the
   wrong mode for this application**; locked manual values are correct.
5. **Longer exposure buys brightness and costs framerate.** Windows exposure is
   log₂ seconds: DSHOW's `-2` is 1/4 s, which caps the camera at 4 fps and
   matched the measured 4.0 exactly. Brightness and framerate are therefore a
   single constrained trade, not two independent knobs.

## Goals

- Tell the operator, from the live feed, whether the camera is good enough to
  detect — lighting, focus, framerate, and detection at the working distance.
- Find good capture settings **by measurement**, and present them for review
  before applying.
- Work for any camera, keyed on measured capability rather than brand.
- Stop `capture_fps` reporting a number nobody measured.

## Non-goals

- **No runtime adaptation.** Calibration measures and advises; it does not
  re-tune during a session. Lighting changed materially? Re-run it.
- **No blocking.** Every threshold is advisory. A demo must be able to start
  capture with bad lighting and a warning, never a refusal.
- **No autofocus control on MSMF.** It does not work; the wizard advises Logi
  G HUB / Logi Tune for that one setting rather than pretending.
- **No new capture backend**, no camera-brand SDKs, no UVC library dependency.

## Design

### 1. Measurement primitives — `sidecar/app/camera_quality.py` (new)

Pure functions over a frame, plus one timed loop. No OpenCV device handling, so
they are testable on synthetic arrays.

```python
@dataclass
class FrameQuality:
    brightness: float   # mean gray, 0-255
    contrast: float     # stdev of gray
    sharpness: float    # variance of Laplacian

def frame_quality(frame) -> FrameQuality: ...
def focus_stability(samples: list[float]) -> float:   # stdev/mean of sharpness
def measure_fps(read, seconds: float, clock=time.monotonic) -> float: ...
```

`measure_fps` takes a `read` callable and a clock so tests drive it with a fake
that yields frames on a synthetic schedule — no camera, no real waiting.

Thresholds live here as named constants with their justification, so the UI and
the tests share one source: `BRIGHTNESS_TARGET = 130`, `BRIGHTNESS_MIN = 110`,
`SHARPNESS_MIN`, `FOCUS_DRIFT_MAX`, `FPS_MIN = 25`.

### 2. Capability probe — `sidecar/app/camera_caps.py` (new)

Opens one device and measures, never trusting a getter.

```python
@dataclass
class ControlSupport:
    brightness: bool
    exposure: bool
    gain: bool
    focus: bool

@dataclass
class CameraProfile:
    device_key: str          # name + index + probed resolution
    backend: str             # "msmf" | "dshow"
    width: int
    height: int
    fps_auto_exposure: float      # measured
    fps_capped_exposure: float    # measured
    controls: ControlSupport
    recommended: dict             # the settings patch
    measured_at: float

def probe_controls(cap, read) -> ControlSupport: ...   # set, then measure the image
def calibrate(open_device, ...) -> CameraProfile: ...
```

A control counts as supported only if the measured image changes — that is the
whole lesson from finding #1.

### 3. Derivation — a pure function

```python
def derive_camera_settings(profile: CameraProfile) -> dict
```

The objective, stated once:

> Raise brightness toward `BRIGHTNESS_TARGET`, **subject to measured fps ≥
> `FPS_MIN`**, preferring brightness gained from exposure until fps would drop
> below the floor, then from the brightness control.

Pure and total: given a `CameraProfile` it returns a settings patch, with no I/O.
This is the piece that carries the real logic, and it is unit-tested against
hand-written profiles (a StreamCam-like profile, a webcam that supports nothing,
a camera that is already good) with no hardware.

### 4. Applying controls — `sidecar/app/camera.py` (change)

`CameraCapture.open()` applies the configured controls after opening and before
starting the read thread. New optional settings, all restart-required, all
"leave alone" when `None` so current behaviour is the default:
`camera_brightness`, `camera_exposure`, `camera_autofocus`, `camera_focus`.

### 5. Honest stats — `sidecar/app/pipeline.py` (change)

`Stats.capture_fps` currently reports `settings.capture_fps`, the requested
value. It becomes a **rolling measured rate** from the capture thread. This is
small but load-bearing: a wizard built on a fabricated number is worthless, and
this single field is what hid a 5× shortfall for the whole session.

### 6. API — `sidecar/app/main.py`

| Endpoint | Purpose |
|---|---|
| `GET /api/camera/quality` | One live `FrameQuality` + measured fps + threshold verdicts. Cheap; drives the wizard's live readout. Works while capture runs, reading the pipeline's latest frame. |
| `POST /api/camera/calibrate` | Runs the sweep (idle only — the device is exclusive), returns a `CameraProfile` **without applying it**. |
| `POST /api/camera/profile/apply` | Applies `profile.recommended` through the existing `_apply_settings_patch`. |

Profiles persist to `data/camera_profiles.json` keyed by `device_key`, reusing
`settings_store`'s atomic-write and corrupt-file-tolerance pattern.

### 7. Wizard — `desktop/src/renderer/src/views/CameraSetup.tsx` (new)

Five steps over the live preview, each with a live readout and an advisory
verdict. Reached from a button in the Admin panel's camera section.

| Step | Shows | Passes when |
|---|---|---|
| 1 Framing | live preview, "place an item where you will scan" | operator confirms |
| 2 Lighting | brightness | 110–160 |
| 3 Focus | sharpness + drift over ~10 s | above min, stable |
| 4 Framerate | measured fps, auto vs capped exposure | ≥ 25 |
| 5 Distance | detection confidence near and far | ≥ 0.6 at both |

It ends on the **review card** — the settings it recommends, the measured
evidence for each, and Apply. Consistent with the existing preset cards, and it
is why "review first" was chosen over auto-apply.

Step 5 needs detections, so it runs with capture live; steps 2–4 need exclusive
access and run with capture stopped. The wizard drives that transition itself
rather than making the operator think about it.

## Error handling

- **Camera busy or gone mid-calibration** — surfaced through the existing
  failure path (`CameraCapture.failure` → pipeline raises → error banner). No
  new mechanism.
- **A control the device ignores** — recorded as unsupported in the profile and
  shown as "not adjustable on this camera", with the G HUB hint for focus.
- **Calibration cannot run while capturing** — the endpoint refuses and says so,
  matching `GET /api/cameras`'s existing behaviour.
- **A corrupt or absent profile file** falls back to no profile, exactly as
  `settings_store` falls back to defaults.

## Testing

Following the project convention that no test touches a camera, GPU, or network:

- `camera_quality` — synthetic frames with known brightness/contrast; a
  gradient for sharpness; `measure_fps` against a fake clock and read callable.
- `camera_caps` — a fake `cap` whose image changes only for the controls it
  chooses to "support", asserting the probe believes the image and not the
  getter. This is the regression test for finding #1.
- `derive_camera_settings` — hand-written profiles: a StreamCam-like one (must
  cap exposure and land ≥25 fps), one supporting nothing (must not invent
  settings), one already good (must change nothing), and one where brightening
  would breach the fps floor (must stop short).
- API — `AppState` injection seams as with `camera_lister`/`hardware_prober`.
- Desktop — the wizard against a faked `ApiClient`, per existing `deps` pattern.

## Delivery order

Each phase is independently useful, so value lands before the wizard does.

1. **Measured `capture_fps` + a live quality readout in Admin.** Alone, this
   would have answered the whole 2026-08-29 session in five seconds.
2. **Capability probe, profile, derivation, apply.** The measured recommendation
   and its review card.
3. **The five-step wizard.** Framing and distance guidance on top.

## Open question for review

Step 5 asks the operator to place a real product to measure detection
confidence. That makes it the only step needing a physical prop, and it is the
step most likely to be skipped. Keep it, or drop step 5 and let the Live view's
existing confidence display serve that purpose?
