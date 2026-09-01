# Measured Camera Preset — Design

Date: 2026-09-02
Status: Approved
Builds on `2026-08-30-camera-calibration-design.md` and
`2026-08-31-live-tab-camera-tuning-design.md`.

## Problem

Calibration measures which controls a camera *honours*. It does not measure
what to *set them to*. `camera_derive.py` says so itself:

> This is a flat boost, not a value computed to land on BRIGHTNESS_TARGET: the
> control's units are device-specific and its transfer function is unknown
> without a dedicated brightness sweep, which is out of scope here.

So `derive_camera_settings` proposes two hardcoded constants —
`BRIGHTNESS_BOOST = 180.0` and `EXPOSURE_CAPPED = -6.0` — and nothing searches
for a value. `probe_controls` writes one extreme per control to see whether the
image moves at all, then throws that measurement away.

The consequence is visible on disk. Both stored profiles for the StreamCam
recommend almost nothing: `{"camera_autofocus": false}` at 720p and `{}` at
1080p. An operator who calibrates gets a support matrix, not a setup.

Two smaller gaps sit alongside it:

- **Autofocus is inferred, never measured.** `camera_derive.py:41` records
  this: `ControlSupport` has no `autofocus` field and `CAP_PROP_AUTOFOCUS` — a
  distinct UVC property — is never probed. The proposal to lock autofocus is a
  proxy from "some other control worked".
- **Focus support is detected by poking `30`** and checking sharpness moved.
  The sharpest setting is never looked for, even though sharpness is already
  computed.

## What a spec-based preset cannot do

The obvious alternative — ship a table of good values per camera model — is
disproved by the profiles already on disk. Same camera, same driver, two
resolutions:

| | brightness | exposure | focus |
|---|---|---|---|
| `Logitech StreamCam:1:1280x720` | ✗ | ✓ | ✗ |
| `Logitech StreamCam:1:1920x1080` | ✗ | ✗ | ✗ |

Control support is a property of the camera *in a mode*, which is why
`device_key_for()` includes resolution. Underneath that sits the founding
premise of the whole calibration feature: `cap.get()` lies, so the camera's own
self-report is not a spec anyone can build a table from.

The answer is not a preset from specs. It is a preset from measurement — which
calibration already is, three quarters built.

## Decisions

1. **Staged scene.** Before sweeping, the operator is asked to place a typical
   item where it will be scanned, under the lighting they will use. Focus is
   then swept for peak sharpness on a real subject at the real distance.
2. **Scope.** The sweep plus a scene gate inside the existing Live-tab tuning
   card. Phase 3's five-step wizard stays unbuilt; this borrows only its
   framing step and would hand the sweep to it for free later.
3. **Time budget.** About two minutes total, which buys accuracy rather than a
   race — see the cost model below.
4. **Search in the sidecar**, not the renderer.

## Approach

Add a sweep phase to `camera_caps.calibrate()`, backed by a device-free search
module. `probe_controls` still runs first and stays cheap: there is no point
sweeping a control the device demonstrably ignores.

### Alternatives rejected

**A client-driven live sweep** — the renderer PATCHes a control, polls
`/api/camera/quality`, repeats — using the live-control path and quality
readout built for the Live tab. It needs no exclusive access, no device
reopens, and the feed stays up. Rejected because with capture running the
detector runs too, and `measured_fps` is one of the numbers gating exposure:
the optimiser would be disturbing the metric it optimises against. It also puts
search logic in TypeScript, where it is hardest to test, and `set_controls`
coalesces onto the capture thread, so settle timing becomes a guess. The saved
opens go back into conservative waits.

**A brightness sweep only** — build exactly the sweep the docstring names and
leave focus alone. Rejected because focus is the control most likely to be
wrong on a fixed checkout camera and the one a store employee can least judge
by eye, and sharpness, the metric that answers it, is already computed.

## Design

### Cost model

Probes are cheap. Device opens are what cost minutes. One probe is a
`cap.set()`, ~3 discarded frames and ~3 averaged — about 0.4 s.

| Control | Range | Search | Probes | Cost |
|---|---|---|---|---|
| exposure | −13…0 (14 values) | binary onto target | ~4 | 1.6 s |
| brightness | 0–255 | binary onto target | ~8 | 3.2 s |
| focus | 0–1023, to ±16 | ternary for peak | ~18 | 7.2 s |

About 12 s of sweeping. Calibration goes from ~60 s to ~85 s with the **same
two device opens it performs today**.

### `camera_search.py` (new)

Pure, device-free, no I/O:

```python
@dataclass
class SearchResult:
    value: float
    metric: float
    probes: int
    span: float        # spread of metric values seen
    reached: bool      # target hit, for the to-target search

def search_to_target(probe, lo, hi, target, *, tolerance, max_probes) -> SearchResult
def search_for_peak(probe, lo, hi, *, min_span, max_probes) -> SearchResult
```

`probe: Callable[[float], float]` writes a value and returns the metric.
`camera_caps` supplies the closure; the algorithms never see a device, so they
are tested against synthetic response curves.

`span` is the flat-curve detector. A sharpness curve with no peak means nothing
was in frame — the failure the staged scene exists to prevent. Reporting it
beats returning a confident wrong answer.

The two metrics have different shapes, which is why there are two searches:
brightness against exposure or against brightness is **monotonic**, so binary
search converges onto a target band; sharpness against focus is **unimodal**,
peaking at the subject's distance, so ternary search finds the maximum.

### Sweep phase in `camera_caps`

Order is a dependency chain, not a preference:

1. **Autofocus off** — or the device hunts away from any focus value written.
2. **Exposure** onto the brightness target. Sharpness is unmeasurable on a
   badly exposed frame: a blown-out or black image has no gradient.
3. **Focus** for the sharpness peak, on a now-correctly-exposed image.
4. **Brightness** to trim whatever exposure could not reach.

**The exposure ceiling is arithmetic, not measured.** Windows exposure is log₂
seconds, so value `e` caps the camera at `1/2^e` fps. The floor it must clear
is the one `derive_camera_settings` already uses — `target_fps * 0.8`, falling
back to `FPS_MIN` (25.0) when no target is supplied. For the default
`capture_fps` of 30 that floor is 24, giving `e ≤ ⌊−log₂(24)⌋ = −5`
(1/32 s → 32 fps). The search's upper bound is clamped to that
rather than discovering the framerate cliff by falling off it. One real fps
sample at the chosen exposure verifies it afterwards — and that sample replaces
`fps_capped_exposure`'s hardcoded `-6` with the rate at the exposure actually
chosen, which is the honest number for the review card.

`ControlSupport` gains a measured `autofocus` field, probed through
`CAP_PROP_AUTOFOCUS`. Recommending a swept focus value while guessing about
autofocus would undo itself.

### Schema

`load_profiles` does `CameraProfile(**value)` and drops anything raising
`TypeError`, so every new field carries a default or the two profiles already
on disk vanish silently.

```python
@dataclass
class ControlSupport:
    ...
    autofocus: bool = False

@dataclass
class CameraProfile:
    ...
    measured: dict = field(default_factory=dict)
    sweep_version: int = 0      # 0 = predates the sweep
```

`measured` and `recommended` stay separate: **evidence versus policy**.
`measured` records what the device did —
`{"camera_focus": {"value": 412, "metric": 84.2, "baseline": 31.5, "reached": true}}`
— while `recommended` remains what `derive_camera_settings` proposes from it.
That split is what keeps `derive` pure, re-runnable against a stored profile,
and testable against hand-written ones, as its docstring already promises.

`sweep_version` earns its place beyond migration: these thresholds will change,
and a bump marks every older profile known-stale without re-deriving anything.

**It also fixes a live misreport.** The 1080p profile currently yields
`recommended: {}`, which the card renders as *"this camera did not respond to
any of the controls we can set"*. An un-swept profile would say the same while
being merely old. With `sweep_version == 0` the card says the calibration
predates level measurement and offers to re-run — a materially different
message from "your camera is deaf".

### `derive_camera_settings`

Policy changes; shape does not. It prefers `profile.measured[k]` when present
and falls back to today's constants otherwise, so it stays total and old
profiles still yield something. Two new invariants:

- Propose `camera_autofocus = False` only when `controls.autofocus` is true.
  On a device that ignores the property it is noise.
- **Suppress the focus recommendation when `controls.autofocus` is false.** If
  autofocus cannot be turned off, a swept focus value is worthless — the device
  hunts straight off it. Shipping it anyway would be the worst outcome:
  confident and wrong.

### Renderer

**The gate goes before the stop, and that is the point:** the operator needs
the live feed to position the item. Clicking Calibrate swaps the button for a
confirm panel while capture is still running:

> Place a typical item where it will be scanned, under the lighting you'll use.
> Use the feed to frame it. The feed stops for about 90 seconds.
> **[Ready] [Cancel]**

Ready runs the existing `stop → calibrate → start` sequence; Cancel returns
without touching capture. When capture is idle the panel adds a line suggesting
the feed be started first to frame the item — copy, not mechanism.

During the wait the card keeps its spinner and renders the phase list
**statically** — exposure → focus → brightness → confirming framerate — so the
operator knows what is happening with no new transport.

The review card gains the per-control evidence: the value chosen, the metric it
achieved, and the baseline it improved on. That evidence is what makes
review-first meaningful rather than ceremonial.

**One thing falls out for free.** Apply already routes through
`_apply_settings_patch`, which now pushes the four camera controls live. So
applying a profile takes effect on the running camera immediately — untrue when
calibration was first designed.

## Error handling

- **Flat curve** — `span` below threshold means no peak existed. Record no
  focus value; the card says *"Couldn't find a focus peak — was an item in
  view?"* and offers a re-run.
- **Target unreachable** — too dark, or the control saturates. Keep the best
  value found, flag `reached: false`, recommend it anyway (still the best
  available) and show that it missed the band.
- **Dropped frame mid-probe** — the existing `_sharpness → None` guard. Retry
  once, then skip that candidate. Never score it 0; that poisons a peak search.
- **Device will not reopen** — unchanged `RuntimeError` path.
- **Start refused after the sweep** — already covered by `finally { start() }`,
  the card's error banner, and the Start-gating added with the Live tab.

## Testing

Sidecar, against fakes as always — no camera, GPU or network:

- Search algorithms against synthetic curves: the peak of a known unimodal
  function found within tolerance; convergence on a monotone function; a flat
  curve yielding low `span`; `max_probes` respected.
- The sweep against a fake `cap` modelling brightness = f(exposure) and
  sharpness = g(focus) with a known optimum — asserting both the chosen values
  and the write ordering (autofocus before focus).
- `derive_camera_settings` against hand-written profiles: one with `measured`,
  one without (the old-profile path), one with autofocus unsupported (focus
  recommendation suppressed).
- `load_profiles` proving the two profiles already on disk survive the schema
  change with `sweep_version == 0`.
- The exposure ceiling computed, not measured: an fps floor of 24 (the default
  `capture_fps` of 30, times 0.8) yields −5, and no probe is spent finding it.

Desktop, via injected deps rather than module mocks:

- Calibrate shows the gate and does **not** call `stop()` until Ready.
- Cancel returns without stopping capture.
- A `sweep_version == 0` profile shows the stale message, not the
  "camera did not respond" message.
- The review card renders the measured evidence per control.

## Risks and accepted tradeoffs

**Values are valid only for the lighting at measure time.** Store lighting
changes between day and night and nothing re-measures. `measured_at` and the
existing quality readout are the mitigation, not a fix.

**Focus is subject-distance specific.** That is the intent, but moving the
counter layout invalidates it and nothing detects that.

**Unimodality is an assumption.** Two items at different distances can produce
two sharpness peaks and ternary search may land on either. The `span` check
catches flatness, not bimodality — which is why the copy says *a typical item*,
singular.

**Capture is down for ~90 s.** Already true at ~60 s today; this makes it
longer.

**The exposure ceiling assumes the driver honours log₂ seconds.** Some do not.
The final fps sample is what catches them.

## Out of scope

- Phase 3's five-step wizard and the `CameraSetup` view. This borrows its
  framing step only, and would hand the sweep to it unchanged later.
- Step 5's near/far confidence sampling and the `imgsz` recommendation
  `derive_camera_settings` already carries — it stays unexercised.
- Live per-phase progress over the existing `StatusMessage` WS. The transport
  exists; the plumbing is a prop chain from the stream hook into the card, and
  it is independent of whether the sweep works.
- Auto-recalibration on lighting drift.
- `gain` — `probe_controls` measures it, but no setting exposes it, so there is
  nothing to sweep.
