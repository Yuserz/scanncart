# Measured Camera Preset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make calibration recommend *values* for the camera controls, not just report which ones the device honours.

**Architecture:** A new device-free search module (`camera_search.py`) provides two algorithms — binary search onto a target for the monotonic controls, ternary search for a peak on focus. `camera_caps.calibrate()` gains a sweep phase that supplies probe closures over the open device and records what it found in a new `CameraProfile.measured` field. `camera_derive.derive_camera_settings()` then proposes those measured values instead of hardcoded constants, and stays pure. The Live tab's tuning card gains a staged-scene gate before the sweep and renders the measured evidence afterwards.

**Tech Stack:** Python 3.12, OpenCV (`cv2`), FastAPI/Pydantic v2, pytest — sidecar. React 19 + TypeScript, Vitest, Testing Library — desktop.

**Spec:** `docs/superpowers/specs/2026-09-02-measured-camera-preset-design.md`

## Global Constraints

- **Sidecar tests never touch hardware.** No camera, GPU, or network. Everything runs against fakes — see `sidecar/tests/test_camera_caps.py` for the established `_Cap` pattern.
- **Desktop tests inject `deps`, never mock modules.** Use `makeDeps()`/`baseSettings()` from `desktop/src/renderer/src/test/fakes.ts`.
- **Every new dataclass field carries a default.** `camera_profiles.load_profiles()` does `CameraProfile(**value)` inside `except (TypeError, ValueError): continue` — a field without a default silently deletes every profile already on disk.
- **The settings/profile contract is hand-mirrored.** `sidecar/app/schemas.py` ↔ `desktop/src/renderer/src/lib/api.ts`. There is no schema generation; changing one without the other is a silent data loss (Pydantic v2 drops unknown kwargs rather than raising).
- **Review-first.** Calibration measures and proposes. It applies nothing; the operator reviews and presses Apply.
- **Nothing trusts `cap.get()`.** Support and values are established by measuring the image, never by reading a property back. See the `camera_caps` module docstring.
- **Total calibration budget: about two minutes.** Today it is ~60 s; the sweep adds ~12 s of probing and no extra device open.
- Existing constants to reuse, never redefine: `BRIGHTNESS_MIN = 110.0`, `BRIGHTNESS_TARGET = 130.0`, `BRIGHTNESS_MAX = 160.0`, `SHARPNESS_MIN = 60.0`, `FOCUS_DRIFT_MAX = 0.25`, `FPS_MIN = 25.0` (all in `app/camera_quality.py`).
- Control bounds, mirroring `SettingsUpdateRequest`: brightness 0–255, exposure −13–0, focus 0–1023.

## File Structure

**Sidecar — create:**
- `sidecar/app/camera_search.py` — `SearchResult`, `search_to_target`, `search_for_peak`. Pure: no `cv2`, no I/O, no clock. One responsibility — find an optimum given a `probe` callable.
- `sidecar/tests/test_camera_search.py`

**Sidecar — modify:**
- `sidecar/app/camera_caps.py` — `ControlSupport.autofocus`; `probe_autofocus()`; `exposure_ceiling()`; `sweep_controls()`; `CameraProfile.measured` / `.sweep_version`; wire the sweep into `calibrate()`
- `sidecar/app/camera_derive.py` — prefer measured values; two autofocus invariants
- `sidecar/app/schemas.py` — `ControlSupportPayload.autofocus`, `CameraProfileResponse.measured` / `.sweep_version`
- `sidecar/tests/test_camera_caps.py`, `test_camera_derive.py`, `test_camera_profiles.py`

**Desktop — modify:**
- `desktop/src/renderer/src/lib/api.ts` — mirror the schema
- `desktop/src/renderer/src/components/CameraTuning.tsx` — scene gate, phase list, evidence, stale-profile message
- `desktop/src/renderer/src/components/CameraTuning.css`
- `desktop/src/renderer/src/components/CameraTuning.test.tsx`

**Docs — modify:** `CLAUDE.md`

---

## Task 1: The search algorithms

**Files:**
- Create: `sidecar/app/camera_search.py`
- Test: `sidecar/tests/test_camera_search.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SearchResult(value: float, metric: float, probes: int, span: float, reached: bool)`; `search_to_target(probe, lo, hi, target, *, tolerance, step=1.0, max_probes=10) -> SearchResult`; `search_for_peak(probe, lo, hi, *, step=1.0, min_span=0.0, max_probes=20) -> SearchResult`. `probe` is `Callable[[float], float]`.

- [ ] **Step 1: Write the failing tests**

Create `sidecar/tests/test_camera_search.py`:

```python
from app.camera_search import search_for_peak, search_to_target


def test_binary_search_lands_on_the_target_of_a_monotone_curve():
    """Brightness rises monotonically with exposure, so the value that hits
    the target band can be bracketed rather than swept exhaustively."""
    # metric(v) = 10*v + 200 -> metric == 130 at v == -7
    result = search_to_target(lambda v: 10 * v + 200, lo=-13, hi=0, target=130.0, tolerance=6.0)

    assert result.value == -7
    assert result.reached is True
    assert result.probes <= 5


def test_binary_search_reports_a_target_it_could_not_reach():
    """A room too dark for any exposure to fix. The best value found is still
    worth recommending; the caller needs to know it fell short."""
    result = search_to_target(lambda v: 20.0, lo=-13, hi=0, target=130.0, tolerance=6.0)

    assert result.reached is False
    assert result.metric == 20.0


def test_binary_search_respects_its_probe_budget():
    calls = []

    def probe(v):
        calls.append(v)
        return 0.0

    search_to_target(probe, lo=0, hi=1023, target=130.0, tolerance=0.1, max_probes=4)
    assert len(calls) <= 4


def test_ternary_search_finds_the_peak_of_a_unimodal_curve():
    """Sharpness against focus peaks at the subject's distance."""
    # An inverted parabola peaking at 400.
    result = search_for_peak(lambda v: 100.0 - ((v - 400) / 100.0) ** 2, lo=0, hi=1023, step=16)

    assert abs(result.value - 400) <= 32
    assert result.reached is True


def test_ternary_search_finds_a_peak_sitting_on_an_endpoint():
    """A camera focused at infinity peaks at one end of the range, where a
    plain ternary search never probes."""
    result = search_for_peak(lambda v: 100.0 - v / 100.0, lo=0, hi=1023, step=16)

    assert result.value == 0


def test_a_flat_curve_reports_no_peak_was_found():
    """Nothing in frame: sharpness does not move, so there is no peak. Saying
    so beats returning a confident wrong distance."""
    result = search_for_peak(lambda v: 42.0, lo=0, hi=1023, step=16, min_span=5.0)

    assert result.reached is False
    assert result.span == 0.0


def test_ternary_search_does_not_re-probe_a_value_it_already_measured():
    """Each probe costs a device write plus settle frames. The bracket shares
    an endpoint between iterations; measuring it twice doubles the cost."""
    calls = []

    def probe(v):
        calls.append(v)
        return 100.0 - ((v - 400) / 100.0) ** 2

    search_for_peak(probe, lo=0, hi=1023, step=16)
    assert len(calls) == len(set(calls))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_camera_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.camera_search'`

- [ ] **Step 3: Write the implementation**

Create `sidecar/app/camera_search.py`:

```python
"""Find the best value for a control, given something that measures the result.

Pure and device-free by design: `probe` is a callable the caller supplies, so
these algorithms are tested against synthetic response curves rather than a
camera. That matters because the two curves this codebase searches have
different shapes and each needs its own algorithm — brightness against
exposure (or against brightness) rises monotonically, so the value hitting a
target band can be bracketed; sharpness against focus peaks at the subject's
distance, so the maximum has to be hunted.

Values are snapped to `step` because every control here is an integer grid
(exposure is 14 whole stops, focus 0-1023), and a search that proposes 412.7
would have it silently truncated by the driver.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class SearchResult:
    value: float
    metric: float
    # Device writes actually spent. Each costs settle frames, so callers
    # budget in probes rather than seconds.
    probes: int
    # Spread of metric values seen across the search. Near-zero means the
    # control did nothing measurable — for a peak search that is "nothing was
    # in frame", which is the failure a staged scene exists to prevent.
    span: float
    # Whether the search achieved what it set out to: hit the target band
    # (search_to_target) or found a peak distinguishable from noise
    # (search_for_peak).
    reached: bool


def _snap(value: float, step: float) -> float:
    return round(value / step) * step


def search_to_target(
    probe: Callable[[float], float],
    lo: float,
    hi: float,
    target: float,
    *,
    tolerance: float,
    step: float = 1.0,
    max_probes: int = 10,
) -> SearchResult:
    """Bracket a monotonically increasing `probe` onto `target`.

    Returns the closest value found even when the target is out of reach —
    a room too dark for any exposure is still better served by the longest
    exposure than by nothing. `reached` says which happened.
    """
    metrics: list[float] = []
    best_value = _snap(lo, step)
    best_metric: float | None = None

    while len(metrics) < max_probes and hi - lo >= step:
        mid = _snap(lo + (hi - lo) / 2, step)
        if mid <= lo or mid >= hi:
            break
        metric = probe(mid)
        metrics.append(metric)
        if best_metric is None or abs(metric - target) < abs(best_metric - target):
            best_value, best_metric = mid, metric
        if abs(metric - target) <= tolerance:
            break
        # Monotonic: below target means we need more of the control.
        if metric < target:
            lo = mid
        else:
            hi = mid

    if best_metric is None:
        # The range was narrower than one step. Measure an endpoint rather
        # than returning a value nothing verified.
        best_metric = probe(best_value)
        metrics.append(best_metric)

    return SearchResult(
        value=best_value,
        metric=best_metric,
        probes=len(metrics),
        span=max(metrics) - min(metrics),
        reached=abs(best_metric - target) <= tolerance,
    )


def search_for_peak(
    probe: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    step: float = 1.0,
    min_span: float = 0.0,
    max_probes: int = 20,
) -> SearchResult:
    """Ternary search for the maximum of a unimodal `probe`.

    Both endpoints are measured first. A plain ternary search never evaluates
    them, and a camera focused at infinity peaks exactly there — so without
    this the most common fixed-camera answer is the one answer unreachable.
    """
    cache: dict[float, float] = {}

    def at(value: float) -> float:
        if value not in cache:
            cache[value] = probe(value)
        return cache[value]

    lo, hi = _snap(lo, step), _snap(hi, step)
    at(lo)
    at(hi)

    while hi - lo > step * 2 and len(cache) + 2 <= max_probes:
        m1 = _snap(lo + (hi - lo) / 3, step)
        m2 = _snap(hi - (hi - lo) / 3, step)
        if m1 >= m2:
            break
        if at(m1) < at(m2):
            lo = m1
        else:
            hi = m2

    best_value = max(cache, key=lambda v: cache[v])
    metrics = list(cache.values())
    span = max(metrics) - min(metrics)

    return SearchResult(
        value=best_value,
        metric=cache[best_value],
        probes=len(cache),
        span=span,
        reached=span >= min_span,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_camera_search.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_search.py sidecar/tests/test_camera_search.py
git commit -m "feat(sidecar): device-free search for a control's best value"
```

---

## Task 2: Measure autofocus instead of inferring it

**Files:**
- Modify: `sidecar/app/camera_caps.py` (`ControlSupport`, new `probe_autofocus`, `probe_controls`)
- Test: `sidecar/tests/test_camera_caps.py`

**Interfaces:**
- Consumes: `app.camera_quality.focus_drift`, `FOCUS_DRIFT_MAX` (both already exist).
- Produces: `ControlSupport.autofocus: bool = False`; `probe_autofocus(cap, read_frame, samples: int = 10) -> bool`. `probe_controls` now sets `support.autofocus`.

**Why:** `camera_derive.py:41` states the current recommendation is a proxy — `CAP_PROP_AUTOFOCUS` is a distinct UVC property that is never probed. Task 3 sweeps focus; recommending a focus value while guessing whether autofocus will hunt off it would undo the whole sweep.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_camera_caps.py`:

```python
class _AutofocusCap:
    """A device that either honours CAP_PROP_AUTOFOCUS=0 or ignores it.

    When it honours the lock, sharpness holds still. When it ignores it, the
    lens keeps hunting and sharpness wanders — which is exactly what
    focus_drift measures.
    """

    def __init__(self, honours_lock: bool):
        self.honours_lock = honours_lock
        self.locked = False
        self._tick = 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_AUTOFOCUS and self.honours_lock:
            self.locked = value == 0
        return True

    def get(self, prop):
        return 0.0

    def sharpness_now(self) -> float:
        self._tick += 1
        if self.locked:
            return 80.0
        # Hunting: stdev/mean well above FOCUS_DRIFT_MAX.
        return 80.0 if self._tick % 2 else 20.0


def _sharp_reader(cap):
    """A frame whose Laplacian variance tracks cap.sharpness_now()."""
    def read():
        level = cap.sharpness_now()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        # A checkerboard's variance scales with its amplitude, giving
        # frame_quality a real gradient to measure.
        frame[::2, ::2] = int(level)
        return frame
    return read


def test_autofocus_counts_as_supported_when_the_lock_actually_holds():
    cap = _AutofocusCap(honours_lock=True)
    assert camera_caps.probe_autofocus(cap, _sharp_reader(cap)) is True


def test_a_device_that_ignores_the_autofocus_lock_is_not_supported():
    """The StreamCam hunts for faces. If the lock does not take, any focus
    value we measure is one the lens will wander off within seconds."""
    cap = _AutofocusCap(honours_lock=False)
    assert camera_caps.probe_autofocus(cap, _sharp_reader(cap)) is False


def test_autofocus_is_unsupported_when_frames_cannot_be_read():
    """An unmeasurable control is an unsupported one, never a crash — the
    same rule the focus probe already follows."""
    cap = _AutofocusCap(honours_lock=True)
    assert camera_caps.probe_autofocus(cap, lambda: None) is False


def test_probe_controls_reports_autofocus_alongside_the_rest():
    cap = _Cap({cv2.CAP_PROP_BRIGHTNESS})
    support = probe_controls(cap, lambda: np.full((8, 8, 3), int(cap.level), dtype=np.uint8))

    assert hasattr(support, "autofocus")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_camera_caps.py -v -k autofocus`
Expected: FAIL — `AttributeError: module 'app.camera_caps' has no attribute 'probe_autofocus'`

- [ ] **Step 3: Write the implementation**

In `sidecar/app/camera_caps.py`, extend the import:

```python
from app.camera_quality import FOCUS_DRIFT_MAX, focus_drift, frame_quality
```

Add the field to `ControlSupport` — **last, with a default**, so existing profiles still load and the existing equality assertions still hold:

```python
@dataclass
class ControlSupport:
    brightness: bool = False
    exposure: bool = False
    gain: bool = False
    focus: bool = False
    # Measured, not inferred. camera_derive used to conclude "some control
    # worked, so autofocus is probably lockable" — a proxy it documented as
    # such. A swept focus value is worthless if the lock does not take.
    autofocus: bool = False
```

Add the probe above `probe_controls`:

```python
# Pin focus somewhere mid-range for the lock test. The exact distance does
# not matter — we are watching whether the lens stays put, not whether it is
# sharp.
AUTOFOCUS_PROBE_FOCUS = 300


def probe_autofocus(cap, read_frame: Callable[[], np.ndarray], samples: int = 10) -> bool:
    """Whether the device honours CAP_PROP_AUTOFOCUS=0.

    Measured, never asked: turn autofocus off, pin a focus value, then watch
    sharpness over a still scene. A lens still hunting moves; one that
    accepted the lock holds. `focus_drift`/`FOCUS_DRIFT_MAX` are the same
    stability test the calibration design uses for its focus step.

    A fixed-focus camera with no autofocus at all also holds still and so
    reports True. That is harmless — the recommendation it earns is a write
    to a property the device ignores — and the alternative would be
    distinguishing "no autofocus" from "locked autofocus" through a getter
    this module treats as a liar.
    """
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, AUTOFOCUS_PROBE_FOCUS)
    seen = [s for s in (_sharpness(read_frame) for _ in range(samples)) if s is not None]
    if len(seen) < 3:
        # Too few readings to call it stable. Unmeasurable is unsupported.
        return False
    return focus_drift(seen) <= FOCUS_DRIFT_MAX
```

At the end of `probe_controls`, before `return support`:

```python
    # Last: it writes CAP_PROP_FOCUS itself, and the focus check above needs
    # the device's own focus behaviour undisturbed by a lock.
    support.autofocus = probe_autofocus(cap, read_frame)
    return support
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_camera_caps.py tests/test_camera_profiles.py -v`
Expected: PASS — including the pre-existing `test_a_device_that_ignores_everything_supports_nothing`, whose `ControlSupport(brightness=False, exposure=False, gain=False, focus=False)` still compares equal because `autofocus` defaults to `False`.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_caps.py sidecar/tests/test_camera_caps.py
git commit -m "feat(sidecar): measure whether the autofocus lock actually takes"
```

---

## Task 3: The sweep

**Files:**
- Modify: `sidecar/app/camera_caps.py` (new `exposure_ceiling`, `ProbeUnavailable`, `sweep_controls`)
- Test: `sidecar/tests/test_camera_caps.py`

**Interfaces:**
- Consumes: `app.camera_search.search_to_target`, `search_for_peak`, `SearchResult` (Task 1); `ControlSupport.autofocus` (Task 2); `BRIGHTNESS_TARGET`, `FPS_MIN` from `app.camera_quality`.
- Produces:
  - `exposure_ceiling(fps_floor: float) -> int`
  - `class ProbeUnavailable(RuntimeError)`
  - `sweep_controls(cap, read_frame, support: ControlSupport, *, fps_floor: float) -> dict[str, dict]` returning e.g. `{"camera_focus": {"value": 400.0, "metric": 84.2, "baseline": 31.5, "reached": True, "probes": 9}}`. Keys are `Settings` field names.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_camera_caps.py`:

```python
from app.camera_caps import exposure_ceiling, sweep_controls


def test_the_exposure_ceiling_is_arithmetic_not_measured():
    """Exposure is log2 seconds, so value e caps delivery at 1/2^e fps. The
    default capture_fps of 30 gives a floor of 24, and -5 is 1/32 s -> 32 fps
    — the longest exposure that still clears it. No probe is spent finding
    this cliff by falling off it."""
    assert exposure_ceiling(24.0) == -5
    assert exposure_ceiling(30.0) == -5
    assert exposure_ceiling(12.0) == -4


def test_the_exposure_ceiling_stays_inside_the_settable_range():
    """SettingsUpdateRequest bounds camera_exposure to -13..0."""
    assert exposure_ceiling(0.0) == 0
    assert exposure_ceiling(100000.0) == -13


class _SweepCap:
    """A device with a known response curve for each control.

    brightness rises with exposure and with brightness; sharpness peaks at
    focus 400. Deliberately not the real physics — just monotone and unimodal
    respectively, which is all the searches assume.
    """

    def __init__(self, support_focus=True, support_exposure=True, support_brightness=True):
        self.exposure = -13.0
        self.brightness = 0.0
        self.focus = 0.0
        self.writes: list[int] = []
        self.support = dict(
            focus=support_focus, exposure=support_exposure, brightness=support_brightness
        )

    def set(self, prop, value):
        self.writes.append(prop)
        if prop == cv2.CAP_PROP_EXPOSURE and self.support["exposure"]:
            self.exposure = value
        elif prop == cv2.CAP_PROP_BRIGHTNESS and self.support["brightness"]:
            self.brightness = value
        elif prop == cv2.CAP_PROP_FOCUS and self.support["focus"]:
            self.focus = value
        return True

    def get(self, prop):
        return 0.0

    def level(self) -> float:
        # 0 at exposure -13, 130 at -7, clipped to a byte.
        return max(0.0, min(255.0, (self.exposure + 13) * 21.7 + self.brightness * 0.4))

    def sharp(self) -> float:
        return 100.0 - ((self.focus - 400) / 100.0) ** 2


def _sweep_reader(cap):
    def read():
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:] = int(cap.level())
        # Modulate half the pixels so Laplacian variance tracks cap.sharp().
        frame[::2, ::2] = int(max(0.0, min(255.0, cap.level() + cap.sharp())))
        return frame
    return read


def test_the_sweep_finds_an_exposure_that_hits_the_brightness_target():
    cap = _SweepCap(support_focus=False)
    support = ControlSupport(exposure=True, autofocus=True)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert "camera_exposure" in measured
    assert measured["camera_exposure"]["reached"] is True
    # -7 lands on the target; the ceiling for a 24 fps floor is -5, so the
    # search had room without needing to be clamped.
    assert measured["camera_exposure"]["value"] == -7


def test_the_sweep_never_proposes_an_exposure_that_would_break_the_fps_floor():
    """A dark room the shutter cannot fix. Without the ceiling the search
    walks to -1 and caps the camera at 2 fps."""
    cap = _SweepCap(support_focus=False)
    cap.level = lambda: 20.0
    support = ControlSupport(exposure=True, autofocus=True)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert measured["camera_exposure"]["value"] <= -5
    assert measured["camera_exposure"]["reached"] is False


def test_the_sweep_finds_the_focus_with_the_highest_sharpness():
    cap = _SweepCap(support_exposure=False, support_brightness=False)
    support = ControlSupport(focus=True, autofocus=True)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert abs(measured["camera_focus"]["value"] - 400) <= 32
    assert measured["camera_focus"]["baseline"] < measured["camera_focus"]["metric"]


def test_a_control_the_device_ignores_is_never_swept():
    """probe_controls already established support. Sweeping a deaf control
    spends probes to discover a flat line we already knew about."""
    cap = _SweepCap()
    support = ControlSupport(brightness=False, exposure=False, focus=False, autofocus=False)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert measured == {}
    assert cap.writes == []


def test_focus_is_not_swept_when_the_autofocus_lock_will_not_take():
    """The lens would hunt straight off whatever we measured, so the probes
    are wasted and the answer would be confidently wrong."""
    cap = _SweepCap()
    support = ControlSupport(focus=True, exposure=True, autofocus=False)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert "camera_focus" not in measured
    assert "camera_exposure" in measured


def test_the_sweep_writes_the_autofocus_lock_before_any_focus_value():
    """Order is a dependency, not a preference: a value written under a live
    autofocus is hunted away from immediately."""
    cap = _SweepCap()
    support = ControlSupport(focus=True, autofocus=True)
    sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert cap.writes.index(cv2.CAP_PROP_AUTOFOCUS) < cap.writes.index(cv2.CAP_PROP_FOCUS)


def test_a_control_whose_frames_never_arrive_is_omitted_rather_than_scored():
    """Scoring an unreadable candidate 0 would poison a peak search into
    choosing whichever value happened to read successfully."""
    cap = _SweepCap()
    support = ControlSupport(focus=True, autofocus=True)
    measured = sweep_controls(cap, lambda: None, support, fps_floor=24.0)

    assert measured == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_camera_caps.py -v -k "sweep or ceiling"`
Expected: FAIL — `ImportError: cannot import name 'exposure_ceiling'`

- [ ] **Step 3: Write the implementation**

In `sidecar/app/camera_caps.py`, add to the imports:

```python
import math

from app.camera_quality import (
    BRIGHTNESS_TARGET,
    FOCUS_DRIFT_MAX,
    focus_drift,
    frame_quality,
)
from app.camera_search import SearchResult, search_for_peak, search_to_target
```

Add the constants and the sweep:

```python
# Bounds mirror SettingsUpdateRequest's ge/le — a value outside them is
# rejected by the API that would have to apply it.
EXPOSURE_RANGE = (-13.0, 0.0)
BRIGHTNESS_RANGE = (0.0, 255.0)
FOCUS_RANGE = (0.0, 1023.0)
# Focus to +/- one step of this. Finer buys nothing: depth of field on a
# fixed counter camera is far wider than 16 units.
FOCUS_STEP = 16.0
# Half the distance between BRIGHTNESS_MIN and BRIGHTNESS_MAX: anywhere in
# the band the quality readout calls "ok" is a hit.
BRIGHTNESS_TOLERANCE = 25.0
# Sharpness must move at least this much across a focus sweep for the peak to
# be real rather than sensor noise. A soft StreamCam frame measures ~5.
MIN_SHARPNESS_SPAN = 10.0
# Frames discarded after a write, then averaged, per probe. A UVC device
# applies a control within a frame or two; averaging three rides out noise.
SETTLE_FRAMES = 3
AVERAGE_FRAMES = 3


class ProbeUnavailable(RuntimeError):
    """The device stopped delivering frames mid-probe, so this control cannot
    be measured. Raised rather than scored, because any score would be a
    fabricated data point the search would then optimise against."""


def exposure_ceiling(fps_floor: float) -> int:
    """The longest exposure, in log2 seconds, that still permits `fps_floor`.

    Windows exposure is log2 seconds: a value e holds the shutter open 2^e
    seconds and so caps delivery at 1/2^e fps. That makes the ceiling
    arithmetic. Measuring it would mean walking the search off the framerate
    cliff to discover where the edge was.
    """
    lo, hi = EXPOSURE_RANGE
    if fps_floor <= 0:
        return int(hi)
    return int(max(lo, min(hi, math.floor(-math.log2(fps_floor)))))


def _metric_probe(cap, read_frame, prop, metric_of):
    """A probe closure: write the control, let it settle, average the metric."""

    def probe(value: float) -> float:
        cap.set(prop, value)
        for _ in range(SETTLE_FRAMES):
            read_frame()
        seen = [m for m in (metric_of(read_frame) for _ in range(AVERAGE_FRAMES)) if m is not None]
        if not seen:
            # One retry: a single dropped read is normal right after a write.
            seen = [m for m in (metric_of(read_frame) for _ in range(AVERAGE_FRAMES)) if m is not None]
        if not seen:
            raise ProbeUnavailable(f"no frames while probing property {prop}")
        return sum(seen) / len(seen)

    return probe


def _mean_brightness(read_frame) -> float | None:
    frame = read_frame()
    return frame_quality(frame).brightness if frame is not None else None


def _record(result: SearchResult, baseline: float) -> dict:
    return {
        "value": result.value,
        "metric": round(result.metric, 1),
        "baseline": round(baseline, 1),
        "reached": result.reached,
        "probes": result.probes,
    }


def sweep_controls(
    cap,
    read_frame: Callable[[], np.ndarray],
    support: ControlSupport,
    *,
    fps_floor: float,
) -> dict[str, dict]:
    """Search each supported control for its best value.

    Order is a dependency chain, not a preference:

    1. Autofocus off, or the device hunts away from any focus value written.
    2. Exposure onto the brightness target — sharpness is unmeasurable on a
       badly exposed frame, because a blown-out or black image has no
       gradient for the Laplacian to find.
    3. Focus for the sharpness peak, now that the image is exposed properly.
    4. Brightness, only to trim what exposure could not reach. It amplifies
       noise along with the picture, so it goes last and does the least.

    Returns evidence keyed by `Settings` field name. A control that cannot be
    measured is omitted rather than guessed at; the caller distinguishes
    "no recommendation" from "recommendation of zero".
    """
    measured: dict[str, dict] = {}

    if support.autofocus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if support.exposure:
        baseline = _mean_brightness(read_frame) or 0.0
        try:
            result = search_to_target(
                _metric_probe(cap, read_frame, cv2.CAP_PROP_EXPOSURE, _mean_brightness),
                lo=EXPOSURE_RANGE[0],
                hi=float(exposure_ceiling(fps_floor)),
                target=BRIGHTNESS_TARGET,
                tolerance=BRIGHTNESS_TOLERANCE,
            )
        except ProbeUnavailable:
            pass
        else:
            measured["camera_exposure"] = _record(result, baseline)

    # Focus only when the lock will actually take. Otherwise the lens wanders
    # off whatever we find, and shipping it anyway would be the worst
    # outcome: confident and wrong.
    if support.focus and support.autofocus:
        baseline = _sharpness(read_frame) or 0.0
        try:
            result = search_for_peak(
                _metric_probe(cap, read_frame, cv2.CAP_PROP_FOCUS, _sharpness),
                lo=FOCUS_RANGE[0],
                hi=FOCUS_RANGE[1],
                step=FOCUS_STEP,
                min_span=MIN_SHARPNESS_SPAN,
            )
        except ProbeUnavailable:
            pass
        else:
            if result.reached:
                measured["camera_focus"] = _record(result, baseline)

    if support.brightness:
        baseline = _mean_brightness(read_frame) or 0.0
        try:
            result = search_to_target(
                _metric_probe(cap, read_frame, cv2.CAP_PROP_BRIGHTNESS, _mean_brightness),
                lo=BRIGHTNESS_RANGE[0],
                hi=BRIGHTNESS_RANGE[1],
                target=BRIGHTNESS_TARGET,
                tolerance=BRIGHTNESS_TOLERANCE,
            )
        except ProbeUnavailable:
            pass
        else:
            measured["camera_brightness"] = _record(result, baseline)

    return measured
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_camera_caps.py -v`
Expected: PASS, including all pre-existing tests in the file.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_caps.py sidecar/tests/test_camera_caps.py
git commit -m "feat(sidecar): sweep each supported control for its best value"
```

---

## Task 4: Wire the sweep into calibrate()

**Files:**
- Modify: `sidecar/app/camera_caps.py` (`CameraProfile`, `calibrate`)
- Test: `sidecar/tests/test_camera_caps.py`, `sidecar/tests/test_camera_profiles.py`

**Interfaces:**
- Consumes: `sweep_controls`, `exposure_ceiling` (Task 3).
- Produces: `CameraProfile.measured: dict = field(default_factory=dict)`; `CameraProfile.sweep_version: int = 0`. `calibrate()` now populates both and measures `fps_capped_exposure` at the swept exposure.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_camera_caps.py`:

```python
def test_calibrate_records_the_measured_values_and_stamps_the_sweep_version():
    """A profile that carries no `measured` is one taken before levels were
    searched for — the card needs to tell that apart from a deaf camera."""
    cap = _SweepCap()

    def open_device(index, backend):
        return _OpenableSweepCap(cap)

    profile = calibrate(0, 1280, 720, open_device=open_device, sample_seconds=0.01)

    assert profile.sweep_version == 1
    assert "camera_exposure" in profile.measured


def test_the_capped_fps_is_measured_at_the_exposure_actually_chosen():
    """It used to be measured at a hardcoded -6 regardless of what would be
    recommended, so the number on the review card described a setting nobody
    was going to apply."""
    cap = _SweepCap()

    def open_device(index, backend):
        return _OpenableSweepCap(cap)

    profile = calibrate(0, 1280, 720, open_device=open_device, sample_seconds=0.01)

    assert cap.exposure == profile.measured["camera_exposure"]["value"]
```

Add the fake wrapper alongside `_SweepCap`:

```python
class _OpenableSweepCap:
    """_SweepCap with the open/read/release surface calibrate() needs.

    calibrate reopens between probing and sweeping, so every open shares one
    underlying _SweepCap — the state under test has to survive the reopen.
    """

    def __init__(self, inner):
        self.inner = inner

    def isOpened(self):
        return True

    def set(self, prop, value):
        return self.inner.set(prop, value)

    def get(self, prop):
        return 0.0

    def read(self):
        return True, _sweep_reader(self.inner)()

    def release(self):
        pass
```

Append to `sidecar/tests/test_camera_profiles.py`:

```python
def test_a_profile_written_before_the_sweep_still_loads(tmp_path):
    """load_profiles builds CameraProfile(**value) inside a bare except that
    drops anything raising TypeError. A new field without a default would
    silently delete every profile already on an operator's disk."""
    path = tmp_path / "camera_profiles.json"
    path.write_text(
        json.dumps(
            {
                "StreamCam:1:1920x1080": {
                    "device_key": "StreamCam:1:1920x1080",
                    "backend": "msmf",
                    "width": 1920,
                    "height": 1080,
                    "fps_auto_exposure": 29.8,
                    "fps_capped_exposure": 30.8,
                    "controls": {"brightness": False, "exposure": False,
                                 "gain": True, "focus": False},
                    "recommended": {},
                    "measured_at": 1788276483.56,
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_profiles(str(path))

    assert "StreamCam:1:1920x1080" in loaded
    assert loaded["StreamCam:1:1920x1080"].sweep_version == 0
    assert loaded["StreamCam:1:1920x1080"].measured == {}
    assert loaded["StreamCam:1:1920x1080"].controls.autofocus is False
```

`test_camera_profiles.py` already imports `json` and `load_profiles` — no import changes needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_camera_caps.py tests/test_camera_profiles.py -v -k "sweep_version or capped_fps or before_the_sweep"`
Expected: FAIL — `AttributeError: 'CameraProfile' object has no attribute 'sweep_version'`

- [ ] **Step 3: Write the implementation**

In `sidecar/app/camera_caps.py`, extend `CameraProfile` — **new fields last, with defaults**:

```python
# Bumped when the sweep's algorithm or thresholds change, so older profiles
# are known-stale without re-deriving anything. 0 means the profile predates
# level measurement entirely.
SWEEP_VERSION = 1


@dataclass
class CameraProfile:
    device_key: str
    backend: str
    width: int
    height: int
    fps_auto_exposure: float
    fps_capped_exposure: float
    controls: ControlSupport
    recommended: dict = field(default_factory=dict)
    measured_at: float = 0.0
    # Evidence: what the device actually did, per control. Kept separate from
    # `recommended`, which is policy — that split is what lets
    # derive_camera_settings stay pure and be re-run against a stored profile.
    measured: dict = field(default_factory=dict)
    sweep_version: int = 0
```

In `calibrate()`, replace the block from `cap.set(cv2.CAP_PROP_EXPOSURE, -6)` through the `profile = CameraProfile(...)` construction with:

```python
        # The sweep runs on the freshly reopened handle: probe_controls left
        # every control pinned to its probe value, and searching from there
        # would confound the search with leftover probe state.
        fps_floor = target_fps * 0.8 if target_fps is not None else FPS_MIN
        measured = sweep_controls(cap, read_frame, controls, fps_floor=fps_floor)

        # Measure the capped rate at the exposure we will actually recommend,
        # not at a hardcoded -6. The old number described a setting nobody was
        # going to apply.
        chosen_exposure = measured.get("camera_exposure", {}).get("value", -6)
        cap.set(cv2.CAP_PROP_EXPOSURE, chosen_exposure)
        fps_capped = measure_fps(read_ok, seconds=sample_seconds)

        profile = CameraProfile(
            device_key=device_key_for(device_name, index, width, height),
            backend="msmf", width=width, height=height,
            fps_auto_exposure=round(fps_auto, 1),
            fps_capped_exposure=round(fps_capped, 1),
            controls=controls, measured_at=time.time(),
            measured=measured, sweep_version=SWEEP_VERSION,
        )
```

Add `FPS_MIN` to the `camera_quality` import if not already present.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest -q`
Expected: PASS — the whole suite, including `test_camera_calibrate_api.py`.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_caps.py sidecar/tests/test_camera_caps.py sidecar/tests/test_camera_profiles.py
git commit -m "feat(sidecar): calibrate records measured levels, not just support"
```

---

## Task 5: Recommend the measured values

**Files:**
- Modify: `sidecar/app/camera_derive.py`
- Test: `sidecar/tests/test_camera_derive.py`

**Interfaces:**
- Consumes: `CameraProfile.measured`, `ControlSupport.autofocus` (Tasks 2, 4).
- Produces: no new names. `derive_camera_settings` keeps its exact signature and stays pure and total.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_camera_derive.py`:

```python
def test_a_measured_exposure_is_preferred_over_the_hardcoded_constant():
    """EXPOSURE_CAPPED was a guess standing in for a sweep that did not
    exist. Where one has run, its answer wins."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(exposure=True, autofocus=True),
            measured={"camera_exposure": {"value": -8.0, "metric": 128.0,
                                          "baseline": 23.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=23.0,
    )

    assert patch["camera_exposure"] == -8.0


def test_a_measured_focus_is_recommended():
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(focus=True, autofocus=True),
            measured={"camera_focus": {"value": 400.0, "metric": 92.0,
                                       "baseline": 31.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=130.0,
    )

    assert patch["camera_focus"] == 400.0


def test_focus_is_withheld_when_the_autofocus_lock_will_not_take():
    """The lens hunts straight off it. Recommending it anyway would be
    confidently wrong, which is worse than recommending nothing."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(focus=True, autofocus=False),
            measured={"camera_focus": {"value": 400.0, "metric": 92.0,
                                       "baseline": 31.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=130.0,
    )

    assert "camera_focus" not in patch
    assert "camera_autofocus" not in patch


def test_autofocus_is_only_locked_on_a_device_that_honours_the_lock():
    """It used to be proposed off a proxy — 'some other control worked'."""
    patch = derive_camera_settings(
        _profile(controls=ControlSupport(brightness=True, autofocus=True)),
        measured_brightness=130.0,
    )
    assert patch["camera_autofocus"] is False

    patch = derive_camera_settings(
        _profile(controls=ControlSupport(brightness=True, autofocus=False)),
        measured_brightness=130.0,
    )
    assert "camera_autofocus" not in patch


def test_an_old_profile_still_yields_the_constant_based_recommendation():
    """sweep_version 0 profiles predate the sweep. derive stays total: they
    get the old behaviour rather than nothing."""
    patch = derive_camera_settings(
        _profile(controls=ControlSupport(brightness=True, exposure=True, autofocus=True)),
        measured_brightness=23.0,
    )

    assert patch["camera_exposure"] == -6.0
    assert patch["camera_brightness"] == 180.0


def test_a_measured_value_is_recommended_even_on_a_frame_that_is_not_dark():
    """The constants only fired below BRIGHTNESS_MIN because a guess is only
    worth risking on a clearly broken image. A measured optimum is not a
    guess, so it applies whenever it exists."""
    patch = derive_camera_settings(
        _profile(
            controls=ControlSupport(exposure=True, autofocus=True),
            measured={"camera_exposure": {"value": -8.0, "metric": 128.0,
                                          "baseline": 118.0, "reached": True}},
            sweep_version=1,
        ),
        measured_brightness=118.0,
    )

    assert patch["camera_exposure"] == -8.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_camera_derive.py -v`
Expected: FAIL — `assert patch["camera_exposure"] == -8.0` gets `-6.0`, and `test_focus_is_withheld...` gets a `camera_autofocus` key.

- [ ] **Step 3: Write the implementation**

Replace the body of `derive_camera_settings` in `sidecar/app/camera_derive.py` (keep the signature) with:

```python
def derive_camera_settings(
    profile: CameraProfile,
    measured_brightness: float,
    near_conf: float | None = None,
    far_conf: float | None = None,
    imgsz: int = 640,
    target_fps: float | None = None,
) -> dict:
    patch: dict = {}
    measured = profile.measured or {}

    # Measured, not inferred. This used to fire whenever any control worked,
    # as a proxy for "the camera is settled and controllable" — the code said
    # so. probe_autofocus now answers it directly, and proposing the lock on a
    # device that ignores the property is noise.
    if profile.controls.autofocus:
        patch["camera_autofocus"] = False

    fps_floor = target_fps * 0.8 if target_fps is not None else FPS_MIN
    too_dark = measured_brightness < BRIGHTNESS_MIN

    # A swept value is an observation, so it applies whenever it exists. The
    # constants below are guesses, and a guess is only worth risking on an
    # image that is already clearly broken — hence the `too_dark` gate on
    # those and not on these.
    if "camera_exposure" in measured and profile.fps_capped_exposure >= fps_floor:
        patch["camera_exposure"] = measured["camera_exposure"]["value"]
    elif too_dark and profile.controls.exposure and profile.fps_capped_exposure >= fps_floor:
        patch["camera_exposure"] = EXPOSURE_CAPPED

    if "camera_brightness" in measured:
        patch["camera_brightness"] = measured["camera_brightness"]["value"]
    elif too_dark and profile.controls.brightness:
        patch["camera_brightness"] = BRIGHTNESS_BOOST

    # Focus needs the lock to hold, or the lens wanders off the value within
    # seconds of applying it. Withhold the lock proposal too: on a device
    # that ignores CAP_PROP_AUTOFOCUS it buys nothing.
    if "camera_focus" in measured and profile.controls.autofocus:
        patch["camera_focus"] = measured["camera_focus"]["value"]

    # A distant item is a small item. Raising imgsz keeps more of it, and CUDA
    # made that affordable (~20 ms/frame on the custom model).
    if near_conf is not None and far_conf is not None:
        if near_conf - far_conf >= FAR_CONF_GAP and imgsz < IMGSZ_MAX:
            patch["imgsz"] = min(imgsz + IMGSZ_STEP, IMGSZ_MAX)

    return patch
```

Update the module docstring's second paragraph to:

```
The objective, once: recommend the value the sweep measured. Where no sweep
has run — a profile written before camera_search existed, or a control the
device ignores — fall back to a flat boost below BRIGHTNESS_MIN, which is a
guess and so is only risked on an image that is already clearly broken.
Exposure is preferred over brightness because it is real light rather than
amplification, but it is also what costs frames, so the fps floor gates it.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest -q`
Expected: PASS — the whole suite.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_derive.py sidecar/tests/test_camera_derive.py
git commit -m "feat(sidecar): recommend measured camera values over constants"
```

---

## Task 6: Carry the new fields to the renderer

**Files:**
- Modify: `sidecar/app/schemas.py:238-255`, `desktop/src/renderer/src/lib/api.ts`
- Test: `sidecar/tests/test_camera_calibrate_api.py`

**Interfaces:**
- Consumes: `CameraProfile.measured`, `.sweep_version`, `ControlSupport.autofocus` (Tasks 2, 4).
- Produces: `CameraProfileResponse.measured: dict`, `.sweep_version: int`; `ControlSupportPayload.autofocus: bool`. TS: `CameraControlSupport.autofocus: boolean`, `CameraProfileResponse.measured: Record<string, MeasuredControl>`, `.sweep_version: number`, and `interface MeasuredControl { value: number; metric: number; baseline: number; reached: boolean; probes: number }`.

**Why its own task:** `main.py:646` builds the response with `CameraProfileResponse(**asdict(profile))`, and Pydantic v2 **drops unknown keyword arguments silently**. A dataclass field with no matching model field never reaches the UI and raises nothing. `CLAUDE.md` names this contract as hand-mirrored for exactly this reason.

- [ ] **Step 1: Write the failing test**

Append to `sidecar/tests/test_camera_calibrate_api.py`:

```python
def test_the_calibrate_response_carries_the_measured_evidence():
    """CameraProfileResponse(**asdict(profile)) silently drops fields the
    model does not declare, so a schema that lags the dataclass loses data
    with no error anywhere."""
    profile = CameraProfile(
        device_key="StreamCam:0:1280x720", backend="msmf", width=1280, height=720,
        fps_auto_exposure=30.0, fps_capped_exposure=29.0,
        controls=ControlSupport(exposure=True, autofocus=True),
        measured={"camera_exposure": {"value": -7.0, "metric": 129.0,
                                      "baseline": 23.0, "reached": True, "probes": 4}},
        sweep_version=1,
    )
    state = AppState(calibrator=lambda: profile, db_path=":memory:")

    with TestClient(build_app(lambda: state)) as client:
        body = client.post("/api/camera/calibrate").json()

    assert body["sweep_version"] == 1
    assert body["measured"]["camera_exposure"]["value"] == -7.0
    assert body["controls"]["autofocus"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_camera_calibrate_api.py -v -k measured_evidence`
Expected: FAIL — `KeyError: 'sweep_version'`

- [ ] **Step 3: Write the implementation**

In `sidecar/app/schemas.py`:

```python
class ControlSupportPayload(BaseModel):
    brightness: bool = False
    exposure: bool = False
    gain: bool = False
    focus: bool = False
    autofocus: bool = False


class CameraProfileResponse(BaseModel):
    device_key: str
    backend: str
    width: int
    height: int
    fps_auto_exposure: float
    fps_capped_exposure: float
    controls: ControlSupportPayload
    recommended: dict = {}
    measured_at: float = 0.0
    # Evidence per control: value, metric, baseline, reached, probes. Mirrors
    # CameraProfile.measured — this model is built with **asdict(profile) and
    # Pydantic drops unknown keys silently, so a field missing here vanishes
    # between the sidecar and the UI with no error raised.
    measured: dict = {}
    # 0 means the profile predates the sweep, which the card must report
    # differently from "this camera responded to nothing".
    sweep_version: int = 0
```

In `desktop/src/renderer/src/lib/api.ts`, alongside `CameraControlSupport`:

```ts
// Mirrors sidecar/app/camera_caps.py::ControlSupport. `autofocus` is measured
// by probe_autofocus, not asked of the driver.
export interface CameraControlSupport {
  brightness: boolean
  exposure: boolean
  gain: boolean
  focus: boolean
  autofocus: boolean
}

// One entry of CameraProfile.measured: what the sweep found for one control.
// `baseline` is the metric before the sweep, so the card can show the
// improvement rather than a bare number.
export interface MeasuredControl {
  value: number
  metric: number
  baseline: number
  reached: boolean
  probes: number
}
```

and extend `CameraProfileResponse`:

```ts
export interface CameraProfileResponse {
  device_key: string
  backend: string
  width: number
  height: number
  fps_auto_exposure: number
  fps_capped_exposure: number
  controls: CameraControlSupport
  recommended: Record<string, unknown>
  measured_at: number
  measured: Record<string, MeasuredControl>
  // 0 = calibrated before levels were measured, not "nothing is supported".
  sweep_version: number
}
```

Then fix every `CameraProfileResponse` fixture the compiler flags — at minimum `desktop/src/renderer/src/components/CameraTuning.test.tsx`'s `PROFILE` and `desktop/src/renderer/src/views/LiveView.test.tsx`'s `PROFILE` — adding `autofocus: true`, `measured: {}`, `sweep_version: 1`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest -q` then `cd ../desktop && npm test && npm run typecheck`
Expected: PASS both suites, typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/schemas.py sidecar/tests/test_camera_calibrate_api.py desktop/src/renderer/src/lib/api.ts desktop/src/renderer/src/components/CameraTuning.test.tsx desktop/src/renderer/src/views/LiveView.test.tsx
git commit -m "feat: carry measured calibration evidence through to the renderer"
```

---

## Task 7: The staged-scene gate

**Files:**
- Modify: `desktop/src/renderer/src/components/CameraTuning.tsx`, `CameraTuning.css`
- Test: `desktop/src/renderer/src/components/CameraTuning.test.tsx`

**Interfaces:**
- Consumes: the existing `handleCalibrate`, `busy`, `onCameraBusy` in `CameraTuning.tsx`.
- Produces: `data-testid="tuning-scene-gate"`, `"tuning-scene-ready"`, `"tuning-scene-cancel"`, `"tuning-phases"`.

**Why the gate precedes the stop:** the operator needs the live feed to position the item. Stopping capture first would ask them to frame a shot they cannot see.

- [ ] **Step 1: Write the failing tests**

Append to `desktop/src/renderer/src/components/CameraTuning.test.tsx`:

```tsx
describe('the staged scene gate', () => {
  it('asks for a scene before it touches capture', async () => {
    // The operator frames the item using the live feed, so the gate has to
    // come before the stop, not after it.
    const order: string[] = []
    const { deps } = makeDeps({ getCameraProfile: async () => ({ profile: PROFILE }) })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => { order.push('start') }}
        stop={async () => { order.push('stop') }}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    expect(await screen.findByTestId('tuning-scene-gate')).toBeInTheDocument()
    expect(order).toEqual([])
  })

  it('runs the sweep once the operator confirms the scene', async () => {
    const order: string[] = []
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => {
        order.push('calibrate')
        return PROFILE
      }
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => { order.push('start') }}
        stop={async () => { order.push('stop') }}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))
    await userEvent.click(await screen.findByTestId('tuning-scene-ready'))

    await waitFor(() => expect(order).toEqual(['stop', 'calibrate', 'start']))
  })

  it('leaves capture alone when the operator backs out', async () => {
    const order: string[] = []
    const { deps } = makeDeps({ getCameraProfile: async () => ({ profile: PROFILE }) })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => { order.push('start') }}
        stop={async () => { order.push('stop') }}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))
    await userEvent.click(await screen.findByTestId('tuning-scene-cancel'))

    expect(screen.queryByTestId('tuning-scene-gate')).toBeNull()
    expect(order).toEqual([])
    expect(screen.getByTestId('tuning-calibrate')).toBeEnabled()
  })

  it('names the phases so a 90-second wait is not a blank spinner', async () => {
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: () => new Promise(() => {})
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))
    await userEvent.click(await screen.findByTestId('tuning-scene-ready'))

    expect(await screen.findByTestId('tuning-phases')).toHaveTextContent('focus')
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx`
Expected: FAIL — `Unable to find an element by: [data-testid="tuning-scene-gate"]`

- [ ] **Step 3: Write the implementation**

In `CameraTuning.tsx`, add state beside `busy`:

```tsx
  // The gate is shown while capture is still running on purpose: the
  // operator frames the item using the live feed. Stopping first would ask
  // them to aim at a picture they cannot see.
  const [gateOpen, setGateOpen] = useState(false)
```

Replace the calibrate button's `onClick` with `onClick={() => setGateOpen(true)}`, and add the gate immediately below it inside the Calibration section:

```tsx
          {gateOpen && !busy && (
            <div className="tuning-gate" data-testid="tuning-scene-gate">
              <p className="field-hint">
                Place a typical item where it will be scanned, under the lighting you&apos;ll use.
                Use the feed to frame it. The feed stops for about 90 seconds.
              </p>
              {!running && (
                <p className="field-hint">
                  Start the feed first if you want to see what you are framing.
                </p>
              )}
              <div className="tuning-gate-actions">
                <button
                  type="button"
                  className="btn-primary btn-small"
                  data-testid="tuning-scene-ready"
                  onClick={() => {
                    setGateOpen(false)
                    void handleCalibrate()
                  }}
                >
                  Ready
                </button>
                <button
                  type="button"
                  className="btn-outline btn-small"
                  data-testid="tuning-scene-cancel"
                  onClick={() => setGateOpen(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
```

Replace the existing `{calibrating && (...)}` progress hint with:

```tsx
          {busy && (
            <div className="field-hint" data-testid="tuning-phases">
              <p>Measuring camera — about 90 seconds. The feed resumes when it finishes.</p>
              {/* Static, not live: the phases run in one blocking sidecar
                  call, and streaming progress would need a transport this
                  card does not have. Naming them still beats a bare
                  spinner. */}
              <ol className="tuning-phase-list">
                <li>exposure</li>
                <li>focus</li>
                <li>brightness</li>
                <li>confirming framerate</li>
              </ol>
            </div>
          )}
```

Add to `CameraTuning.css`:

```css
.tuning-gate {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.tuning-gate-actions {
  display: flex;
  gap: 8px;
}

.tuning-phase-list {
  margin: 4px 0 0;
  padding-left: 18px;
  color: var(--dim);
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npm test && npm run typecheck && npm run lint`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/components/CameraTuning.tsx desktop/src/renderer/src/components/CameraTuning.css desktop/src/renderer/src/components/CameraTuning.test.tsx
git commit -m "feat(desktop): stage the scene before the calibration sweep"
```

---

## Task 8: Show the evidence, and tell a stale profile apart from a deaf camera

**Files:**
- Modify: `desktop/src/renderer/src/components/CameraTuning.tsx`
- Test: `desktop/src/renderer/src/components/CameraTuning.test.tsx`

**Interfaces:**
- Consumes: `CameraProfileResponse.measured`, `.sweep_version`, `MeasuredControl` (Task 6).
- Produces: `data-testid="tuning-evidence"`, `"tuning-stale-profile"`.

- [ ] **Step 1: Write the failing tests**

Append to `desktop/src/renderer/src/components/CameraTuning.test.tsx`:

```tsx
describe('the review card', () => {
  const SWEPT = {
    ...PROFILE,
    sweep_version: 1,
    recommended: { camera_focus: 400 },
    measured: {
      camera_focus: { value: 400, metric: 84.2, baseline: 31.5, reached: true, probes: 9 }
    }
  }

  it('shows what each value achieved, not just the value', async () => {
    // Review-first is only meaningful if the operator can see the evidence.
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => SWEPT
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))
    await userEvent.click(await screen.findByTestId('tuning-scene-ready'))

    const evidence = await screen.findByTestId('tuning-evidence')
    expect(evidence).toHaveTextContent('84.2')
    expect(evidence).toHaveTextContent('31.5')
  })

  it('says a peak could not be found rather than inventing one', async () => {
    // A flat sharpness curve means nothing was in frame — the failure the
    // scene gate exists to prevent.
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => ({ ...SWEPT, recommended: {}, measured: {} })
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))
    await userEvent.click(await screen.findByTestId('tuning-scene-ready'))

    expect(await screen.findByTestId('tuning-no-recommendation')).toHaveTextContent(
      /item in view/i
    )
  })

  it('tells the operator an old profile predates level measurement', async () => {
    // Otherwise sweep_version 0 reads as "this camera responded to nothing",
    // which is a completely different thing to do about it.
    const { deps } = makeDeps({
      getCameraProfile: async () => ({
        profile: { ...PROFILE, sweep_version: 0, recommended: {}, measured: {} }
      })
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByRole('button', { name: /Camera tuning/ }))

    expect(await screen.findByTestId('tuning-stale-profile')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx`
Expected: FAIL — `Unable to find an element by: [data-testid="tuning-evidence"]`

- [ ] **Step 3: Write the implementation**

In `CameraTuning.tsx`, add above the `return`:

```tsx
  // A stored profile from before the sweep existed. Its empty `recommended`
  // means "we never looked", not "this camera responded to nothing" — and
  // the fix for one is re-calibrating while the fix for the other is a
  // different camera.
  const profileIsStale = storedProfile !== null && storedProfile.sweep_version === 0
```

Inside the Calibration section, before the calibrate button:

```tsx
          {profileIsStale && !busy && (
            <p className="field-hint" data-testid="tuning-stale-profile">
              This camera was calibrated before we measured control levels. Re-calibrate to get
              recommended values.
            </p>
          )}
```

Replace the recommendation list inside `{profile && !calibrating && (...)}` with:

```tsx
              {Object.keys(profile.recommended).length === 0 ? (
                <p className="field-hint" data-testid="tuning-no-recommendation">
                  Nothing to change: this camera ignored every control we can set, or there was no
                  item in view for the focus sweep to find.
                </p>
              ) : (
                <>
                  <ul data-testid="tuning-evidence">
                    {Object.entries(profile.recommended).map(([k, v]) => {
                      const m = profile.measured?.[k]
                      return (
                        <li key={k}>
                          {k}: {String(v)}
                          {m && (
                            <span className="field-hint">
                              {' '}
                              — {m.metric} (was {m.baseline})
                              {m.reached ? '' : ', best available'}
                            </span>
                          )}
                        </li>
                      )
                    })}
                  </ul>
                  <button
                    className="btn-primary btn-small"
                    disabled={saving}
                    data-testid="tuning-apply-profile"
                    onClick={() => reported(applyProfile())}
                  >
                    {saving ? <Spinner /> : null} Apply these settings
                  </button>
                </>
              )}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npm test && npm run typecheck && npm run lint`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/components/CameraTuning.tsx desktop/src/renderer/src/components/CameraTuning.test.tsx
git commit -m "feat(desktop): show the measured evidence behind each recommendation"
```

---

## Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Update the architecture notes**

In `CLAUDE.md`, add a bullet after the `camera_quality.py` / `camera_caps.py` entries:

```
- **`camera_search.py`** — `search_to_target` (binary, for a monotone curve)
  and `search_for_peak` (ternary, for a unimodal one). Pure and device-free:
  `probe` is a callable the caller supplies, so the algorithms are tested
  against synthetic response curves rather than a camera. Two algorithms
  because the two curves differ — brightness rises monotonically with
  exposure, while sharpness peaks at the subject's distance. `SearchResult.span`
  is the flat-curve detector: a sharpness sweep that moves nothing means
  nothing was in frame, which is reported rather than resolved into a
  confident wrong answer.
```

Extend the `camera_caps.py` description with:

```
`calibrate()` runs `probe_controls` (which controls the device honours) and
then `sweep_controls` (what to set them to), in a fixed order: autofocus off,
exposure onto BRIGHTNESS_TARGET, focus for the sharpness peak, brightness to
trim the remainder. Order is a dependency — a focus value written under a live
autofocus is hunted away from, and sharpness is unmeasurable on a frame that is
blown out or black. `exposure_ceiling()` computes the longest exposure that
still clears the fps floor rather than discovering that cliff by falling off
it. `CameraProfile.measured` records the evidence (value, metric, baseline,
reached) separately from `recommended`, which is policy — that split is what
keeps `camera_derive` pure and re-runnable against a stored profile.
`sweep_version` is 0 for profiles written before the sweep existed, which the
UI must report differently from a camera that responded to nothing.
```

Update the `camera_derive.py` mention (or add one) to say it now prefers `profile.measured` and falls back to the old constants only for un-swept profiles, and that `camera_autofocus` is measured by `probe_autofocus` rather than inferred.

Add `CameraProfile.measured` / `sweep_version` and `ControlSupport.autofocus` to the testing-conventions paragraph's list of hand-mirrored contracts.

- [ ] **Step 2: Verify the whole suite**

Run: `cd sidecar && python -m pytest -q` then `cd ../desktop && npm test && npm run typecheck && npm run lint`
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: describe the measured camera sweep"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| `camera_search.py`, both algorithms, `span` flat-curve detector | 1 |
| Autofocus measured via `CAP_PROP_AUTOFOCUS`, `ControlSupport.autofocus` | 2 |
| Sweep order (autofocus → exposure → focus → brightness) | 3 |
| Exposure ceiling is arithmetic | 3 |
| Dropped frame: retry once, then omit — never score 0 | 3 |
| Flat curve → no focus value recorded | 3 (`min_span` gate) + 8 (copy) |
| Target unreachable → keep best, flag `reached: false` | 1, 3, 8 |
| `measured` / `sweep_version` schema with defaults | 4 |
| fps measured at the chosen exposure | 4 |
| Old profiles survive with `sweep_version == 0` | 4 |
| `derive` prefers measured; autofocus invariants | 5 |
| Focus suppressed when autofocus unsupported | 3 (not swept) + 5 (not recommended) |
| Schema ↔ `api.ts` mirror | 6 |
| Scene gate before the stop; idle hint; Cancel | 7 |
| Static phase list | 7 |
| Per-control evidence on the review card | 8 |
| Stale profile distinguished from a deaf camera | 8 |
| Docs | 9 |

Two spec items are deliberately not tasks, both already listed as out of scope in the spec: live WS progress, and `gain`. The "device will not reopen" error path is unchanged code, so it has no task.

**Placeholder scan:** none — every code step carries real code, and every test step carries real assertions.

**Type consistency:** `SearchResult(value, metric, probes, span, reached)` defined in Task 1 is consumed by name in Task 3's `_record`. The `measured` entry shape `{value, metric, baseline, reached, probes}` produced by Task 3's `_record` matches Task 5's test fixtures, Task 6's `MeasuredControl`, and Task 8's rendering. `exposure_ceiling` and `sweep_controls` signatures in Task 3's Interfaces match their call sites in Task 4. `ControlSupport.autofocus` added in Task 2 is used in Tasks 3, 5, and 6.

**One risk worth flagging to the executor:** Task 3's `_SweepCap` and `_sweep_reader` fakes must produce a frame whose `frame_quality().sharpness` genuinely tracks `cap.sharp()`. If the checkerboard construction does not yield a monotone relationship, the focus test will fail for a reason unrelated to the search. Verify by asserting the reader directly before debugging the sweep.
