# Guided Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tell the operator from the live feed whether the camera can actually detect, and find good capture settings by measurement rather than guesswork.

**Architecture:** Three layers, each independently testable. Pure frame-metric functions (`camera_quality.py`) know nothing about devices. A capability probe (`camera_caps.py`) opens one device and decides what it supports by *measuring the image*, never by reading a property back. A pure `derive_camera_settings()` turns a measured profile into a settings patch. The API and wizard are thin over those.

**Tech Stack:** Python 3.12, FastAPI, OpenCV (MSMF backend), pytest; Electron + React + TypeScript, vitest.

**Spec:** `docs/superpowers/specs/2026-08-30-camera-calibration-design.md`

## Global Constraints

- **No test may touch a camera, GPU, network, or Roboflow key.** Use fakes and constructor injection — the established convention in `sidecar/tests/`.
- **Never trust an OpenCV property getter.** Setting `CAP_PROP_EXPOSURE` on MSMF changed delivered fps 12.3 → 30.3 while `cap.get()` returned the old value. Support is proven by a measured change in the image or the frame rate.
- **All thresholds are advisory.** Nothing in this plan may block `POST /api/capture/start`.
- Named thresholds, defined once in `camera_quality.py`: `BRIGHTNESS_TARGET = 130.0`, `BRIGHTNESS_MIN = 110.0`, `BRIGHTNESS_MAX = 160.0`, `SHARPNESS_MIN = 60.0`, `FOCUS_DRIFT_MAX = 0.25`, `FPS_MIN = 25.0`.
- New settings fields are **restart-required** (add to `RESTART_REQUIRED_FIELDS`) and default to `None` meaning "leave the device alone", so current behaviour is unchanged until calibration applies something.
- Desktop mirrors of Python contracts are hand-synced: `api.ts`, `settingsFields.ts`, `settingsDefaults.ts`. Update all three when a settings field is added.
- Sidecar tests run from `sidecar/`: `.venv/Scripts/python.exe -m pytest -q`. Desktop from `desktop/`: `npm test`.

## File Structure

| File | Responsibility |
|---|---|
| `sidecar/app/camera_quality.py` (new) | Pure frame metrics + thresholds + `measure_fps`. No device handling. |
| `sidecar/app/camera_caps.py` (new) | `ControlSupport`, `CameraProfile`, `probe_controls`, `calibrate`. Opens devices. |
| `sidecar/app/camera_derive.py` (new) | `derive_camera_settings()` — pure, carries the real logic. |
| `sidecar/app/camera_profiles.py` (new) | Load/save profiles to `data/camera_profiles.json`. |
| `sidecar/app/camera.py` (modify) | Apply controls on open; expose measured capture fps. |
| `sidecar/app/pipeline.py` (modify) | Report measured `capture_fps`. |
| `sidecar/app/settings.py`, `settings_store.py`, `schemas.py`, `main.py` (modify) | New fields, validation, endpoints. |
| `desktop/.../lib/api.ts`, `settingsFields.ts`, `settingsDefaults.ts` (modify) | Contract mirrors. |
| `desktop/.../views/CameraSetup.tsx` (new) | Five-step wizard. |
| `desktop/.../views/AdminPanel.tsx` (modify) | Live quality readout + entry point. |

---

# Phase 1 — Honest numbers

Delivers the readout that would have answered the 2026-08-29 session in seconds.

### Task 1: Frame quality metrics

**Files:**
- Create: `sidecar/app/camera_quality.py`
- Test: `sidecar/tests/test_camera_quality.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FrameQuality(brightness: float, contrast: float, sharpness: float)`; `frame_quality(frame: np.ndarray) -> FrameQuality`; constants `BRIGHTNESS_TARGET`, `BRIGHTNESS_MIN`, `BRIGHTNESS_MAX`, `SHARPNESS_MIN`, `FOCUS_DRIFT_MAX`, `FPS_MIN`.

- [ ] **Step 1: Write the failing test**

```python
# sidecar/tests/test_camera_quality.py
import numpy as np

from app.camera_quality import BRIGHTNESS_MIN, FrameQuality, frame_quality


def test_a_uniform_dark_frame_reports_its_brightness():
    # The 2026-08-29 counter frame measured 23/255 and detected nothing.
    frame = np.full((64, 64, 3), 23, dtype=np.uint8)
    q = frame_quality(frame)
    assert isinstance(q, FrameQuality)
    assert 22 <= q.brightness <= 24
    assert q.brightness < BRIGHTNESS_MIN


def test_a_flat_frame_has_no_contrast_and_no_sharpness():
    q = frame_quality(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert q.contrast < 1.0
    assert q.sharpness < 1.0


def test_an_edged_frame_is_sharper_than_a_blurred_one():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, 32:] = 255
    import cv2

    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    assert frame_quality(frame).sharpness > frame_quality(blurred).sharpness
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_quality.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.camera_quality'`

- [ ] **Step 3: Write minimal implementation**

```python
# sidecar/app/camera_quality.py
"""Frame metrics and the thresholds that judge them.

Pure functions over arrays: no device handling, so tests run on synthetic
frames. The thresholds live here so the API, the wizard and the tests share one
source rather than three drifting copies.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# A frame this dark detected nothing on 2026-08-29 (measured 23/255).
BRIGHTNESS_MIN = 110.0
BRIGHTNESS_TARGET = 130.0
BRIGHTNESS_MAX = 160.0
# Laplacian variance. A soft StreamCam frame measured ~5; a focused one is
# in the hundreds.
SHARPNESS_MIN = 60.0
# stdev/mean of sharpness over a still scene. Autofocus hunting showed 6->31.
FOCUS_DRIFT_MAX = 0.25
FPS_MIN = 25.0


@dataclass
class FrameQuality:
    brightness: float
    contrast: float
    sharpness: float


def frame_quality(frame: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return FrameQuality(
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_quality.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_quality.py sidecar/tests/test_camera_quality.py
git commit -m "feat: frame quality metrics for camera calibration"
```

---

### Task 2: Focus drift and measured fps

**Files:**
- Modify: `sidecar/app/camera_quality.py`
- Test: `sidecar/tests/test_camera_quality.py`

**Interfaces:**
- Consumes: Task 1's module.
- Produces: `focus_drift(samples: list[float]) -> float`; `measure_fps(read: Callable[[], bool], seconds: float, clock=time.monotonic, sleep=time.sleep) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# append to sidecar/tests/test_camera_quality.py
import pytest

from app.camera_quality import focus_drift, measure_fps


def test_focus_drift_is_zero_for_a_steady_scene():
    assert focus_drift([100.0, 100.0, 100.0]) == 0.0


def test_focus_drift_flags_a_hunting_autofocus():
    # The StreamCam swung 6 -> 31 on a static scene.
    assert focus_drift([6.0, 31.0, 8.0, 29.0]) > 0.25


def test_focus_drift_of_too_few_samples_is_zero():
    assert focus_drift([]) == 0.0
    assert focus_drift([12.0]) == 0.0


def test_measure_fps_counts_successful_reads_per_second():
    now = {"t": 0.0}
    reads = {"n": 0}

    def clock():
        return now["t"]

    def read():
        reads["n"] += 1
        now["t"] += 0.05  # 20 fps
        return True

    fps = measure_fps(read, seconds=1.0, clock=clock, sleep=lambda s: None)
    assert 19.0 <= fps <= 21.0


def test_measure_fps_ignores_failed_reads():
    now = {"t": 0.0}
    n = {"i": 0}

    def clock():
        return now["t"]

    def read():
        n["i"] += 1
        now["t"] += 0.05
        return n["i"] % 2 == 0  # half the reads fail

    fps = measure_fps(read, seconds=1.0, clock=clock, sleep=lambda s: None)
    assert 9.0 <= fps <= 11.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_quality.py -v`
Expected: FAIL — `ImportError: cannot import name 'focus_drift'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to sidecar/app/camera_quality.py
import statistics
import time
from typing import Callable


def focus_drift(samples: list[float]) -> float:
    """Relative spread of sharpness over a still scene.

    Autofocus hunting shows up as a large spread with nothing moving. Fewer
    than two samples cannot show drift, so they report none.
    """
    if len(samples) < 2:
        return 0.0
    mean = statistics.fmean(samples)
    if mean <= 0.0:
        return 0.0
    return statistics.pstdev(samples) / mean


def measure_fps(
    read: Callable[[], bool],
    seconds: float = 3.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Frames actually delivered per second.

    The only trustworthy capability signal: a camera reports 60 while
    delivering 12, and setting exposure changes the rate without changing what
    any getter returns. `clock`/`sleep` are injected so tests run instantly.
    """
    start = clock()
    frames = 0
    while clock() - start < seconds:
        if read():
            frames += 1
        else:
            sleep(0.005)
    elapsed = clock() - start
    return frames / elapsed if elapsed > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_quality.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_quality.py sidecar/tests/test_camera_quality.py
git commit -m "feat: focus drift and measured fps helpers"
```

---

### Task 3: CameraCapture reports its measured rate

**Files:**
- Modify: `sidecar/app/camera.py` (the `CameraCapture` class)
- Test: `sidecar/tests/test_camera.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CameraCapture.measured_fps` — a `float` property, a rolling rate over the last second of successful reads, `0.0` before enough samples.

- [ ] **Step 1: Write the failing test**

```python
# append to sidecar/tests/test_camera.py
def test_capture_reports_the_rate_it_actually_delivers():
    """capture_fps used to report the *requested* value: the UI showed 60
    while the camera delivered 12, hiding a 5x shortfall all session."""
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self):
            time.sleep(0.01)  # ~100 fps ceiling
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open()
    time.sleep(1.2)
    rate = c.measured_fps
    c.release()

    assert rate > 10.0          # it is measuring something real
    assert rate < 300.0         # and not nonsense


def test_measured_fps_is_zero_before_any_frame():
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self): return False, None
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    assert c.measured_fps == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera.py -v`
Expected: FAIL — `AttributeError: 'CameraCapture' object has no attribute 'measured_fps'`

- [ ] **Step 3: Write minimal implementation**

In `CameraCapture.__init__`, after `self._consecutive_failures = 0`:

```python
        # Timestamps of recent successful reads, for the measured rate. The
        # requested fps is a request; this is what arrived.
        self._read_times: deque[float] = deque(maxlen=120)
```

Add `from collections import deque` to the imports.

In `_loop`, immediately after `self._consecutive_failures = 0`:

```python
            self._read_times.append(time.monotonic())
```

Add the property:

```python
    @property
    def measured_fps(self) -> float:
        """Frames delivered per second over the last second, 0.0 until known."""
        now = time.monotonic()
        recent = [t for t in self._read_times if now - t <= 1.0]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera.py sidecar/tests/test_camera.py
git commit -m "feat: CameraCapture measures its delivered frame rate"
```

---

### Task 4: Pipeline reports the measured rate

**Files:**
- Modify: `sidecar/app/pipeline.py` (`process_once`, `emit_preview`)
- Test: `sidecar/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `CameraCapture.measured_fps` (Task 3).
- Produces: `Stats.capture_fps` is the measured rate when the source exposes `measured_fps`, else falls back to `source.fps`.

- [ ] **Step 1: Write the failing test**

```python
# append to sidecar/tests/test_pipeline.py
def test_stats_report_the_measured_capture_rate():
    """Not the requested one: the UI showed 60 fps while 12 arrived."""
    class _Source(_StubSource):
        measured_fps = 12.5

    sent = []
    pipe = Pipeline(_Source(), _StubDetector(), Settings(capture_fps=60),
                    on_message=sent.append)
    msg = pipe.process_once()

    assert msg["stats"]["capture_fps"] == 12.5


def test_stats_fall_back_to_the_configured_rate_for_a_source_that_cannot_measure():
    # FakeFrameSource and other test doubles have no measured_fps.
    sent = []
    pipe = Pipeline(_StubSource(), _StubDetector(), Settings(), on_message=sent.append)
    msg = pipe.process_once()

    assert msg["stats"]["capture_fps"] == _StubSource.fps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `assert 30.0 == 12.5`

- [ ] **Step 3: Write minimal implementation**

Add near the top of `Pipeline`:

```python
    def _capture_fps(self) -> float:
        """What the camera actually delivers, falling back to its nominal rate
        for sources that cannot measure (test doubles)."""
        measured = getattr(self._source, "measured_fps", None)
        if measured:
            return float(measured)
        return float(getattr(self._source, "fps", 0.0))
```

In `process_once`, replace
`capture_fps=float(getattr(self._source, "fps", 0.0)),`
with
`capture_fps=self._capture_fps(),`

Do the same in `emit_preview`'s fallback `Stats(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/pipeline.py sidecar/tests/test_pipeline.py
git commit -m "fix: report the capture rate we measure, not the one we asked for"
```

---

### Task 5: `GET /api/camera/quality`

**Files:**
- Modify: `sidecar/app/schemas.py`, `sidecar/app/main.py`
- Test: `sidecar/tests/test_camera_quality_api.py` (new)

**Interfaces:**
- Consumes: `frame_quality`, thresholds (Task 1); `Pipeline` latest frame.
- Produces: `GET /api/camera/quality` → `CameraQualityResponse(available: bool, brightness: float, contrast: float, sharpness: float, capture_fps: float, verdicts: dict[str, str], detail: str)`. Each verdict is `"ok" | "low" | "high"`.

- [ ] **Step 1: Write the failing test**

```python
# sidecar/tests/test_camera_quality_api.py
import numpy as np
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.schemas import Detection


class _Src:
    width, height, fps = 128, 96, 30.0
    measured_fps = 28.0

    def open(self): return True
    def latest(self): return (1, np.full((96, 128, 3), 130, dtype=np.uint8))
    def release(self): pass


class _Det:
    names = {0: "milo"}
    def infer(self, frame):
        return [Detection(track_id=1, cls="milo", conf=0.9, box=(0.1, 0.1, 0.2, 0.2))]


def _client():
    state = AppState(source_factory=lambda s: _Src(),
                     detector_factory=lambda s, d: _Det(), db_path=":memory:")
    return TestClient(build_app(lambda: state)), state


def test_quality_is_unavailable_before_capture_starts():
    client, _ = _client()
    body = client.get("/api/camera/quality").json()
    assert body["available"] is False


def test_quality_reports_metrics_and_verdicts_while_running():
    client, _ = _client()
    with client:
        client.post("/api/capture/start")
        body = client.get("/api/camera/quality").json()
        client.post("/api/capture/stop")

    assert body["available"] is True
    assert 129 <= body["brightness"] <= 131
    assert body["verdicts"]["brightness"] == "ok"      # 130 is the target
    assert body["capture_fps"] == 28.0


def test_a_dark_frame_is_reported_as_low():
    class _Dark(_Src):
        def latest(self): return (1, np.full((96, 128, 3), 23, dtype=np.uint8))

    state = AppState(source_factory=lambda s: _Dark(),
                     detector_factory=lambda s, d: _Det(), db_path=":memory:")
    client = TestClient(build_app(lambda: state))
    with client:
        client.post("/api/capture/start")
        body = client.get("/api/camera/quality").json()
        client.post("/api/capture/stop")

    assert body["verdicts"]["brightness"] == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_quality_api.py -v`
Expected: FAIL — 404 on `/api/camera/quality`

- [ ] **Step 3: Write minimal implementation**

In `schemas.py`, beside `CamerasResponse`:

```python
class CameraQualityResponse(BaseModel):
    """Live image metrics, for the setup wizard's readout."""
    available: bool = False
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    capture_fps: float = 0.0
    verdicts: dict[str, str] = {}
    detail: str = ""
```

In `main.py`, import `CameraQualityResponse` and:

```python
from app.camera_quality import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    FPS_MIN,
    SHARPNESS_MIN,
    frame_quality,
)
```

Add the route beside `get_cameras`:

```python
    @app.get("/api/camera/quality", response_model=CameraQualityResponse)
    async def camera_quality():
        """Live image metrics. Reads the pipeline's newest frame rather than
        opening the device, so it works while capture holds it."""
        source = state.source
        if state.state != "running" or source is None:
            return CameraQualityResponse(
                available=False, detail="Start capture to measure the image."
            )
        got = source.latest()
        if got is None:
            return CameraQualityResponse(available=False, detail="No frame yet.")

        q = await run_in_threadpool(frame_quality, got[1])
        fps = float(getattr(source, "measured_fps", 0.0))
        return CameraQualityResponse(
            available=True,
            brightness=round(q.brightness, 1),
            contrast=round(q.contrast, 1),
            sharpness=round(q.sharpness, 1),
            capture_fps=round(fps, 1),
            verdicts={
                "brightness": "low" if q.brightness < BRIGHTNESS_MIN
                else "high" if q.brightness > BRIGHTNESS_MAX else "ok",
                "sharpness": "low" if q.sharpness < SHARPNESS_MIN else "ok",
                "capture_fps": "low" if 0 < fps < FPS_MIN else "ok",
            },
            detail="",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_quality_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/schemas.py sidecar/app/main.py sidecar/tests/test_camera_quality_api.py
git commit -m "feat: GET /api/camera/quality live image metrics"
```

---

### Task 6: Admin shows the live readout

**Files:**
- Modify: `desktop/src/renderer/src/lib/api.ts`, `desktop/src/renderer/src/hooks/useSidecarSettings.ts`, `desktop/src/renderer/src/views/AdminPanel.tsx`, `desktop/src/renderer/src/views/AdminPanel.css`
- Test: `desktop/src/renderer/src/views/AdminPanel.test.tsx`

**Interfaces:**
- Consumes: `GET /api/camera/quality` (Task 5).
- Produces: `ApiClient.getCameraQuality(): Promise<CameraQualityResponse>`; `useSidecarSettings` exposes `cameraQuality: CameraQualityResponse | null`, polled every 2 s while capture runs.

- [ ] **Step 1: Write the failing test**

```tsx
// append inside describe('AdminPanel', ...) in AdminPanel.test.tsx
it('shows live image quality with a warning when the frame is too dark', async () => {
  // Brightness 23/255 was the real cause of "detection is broken".
  const { deps } = makeDeps('running', {
    getCameraQuality: vi.fn(async () => ({
      available: true, brightness: 23, contrast: 27, sharpness: 4.8,
      capture_fps: 12, verdicts: { brightness: 'low', sharpness: 'low', capture_fps: 'low' },
      detail: ''
    }))
  })
  render(<AdminPanel port={8765} deps={deps} />)

  const panel = await screen.findByTestId('camera-quality')
  expect(panel).toHaveTextContent('23')
  expect(within(panel).getAllByTestId('quality-low').length).toBeGreaterThan(0)
})

it('hides the quality readout when capture is not running', async () => {
  const { deps } = makeDeps('idle', {
    getCameraQuality: vi.fn(async () => ({
      available: false, brightness: 0, contrast: 0, sharpness: 0,
      capture_fps: 0, verdicts: {}, detail: 'Start capture to measure the image.'
    }))
  })
  render(<AdminPanel port={8765} deps={deps} />)

  await screen.findByTestId('hardware-info')
  expect(screen.queryByTestId('camera-quality')).not.toBeInTheDocument()
})
```

Add `getCameraQuality` to the `makeDeps` fake's `api` object (returning `available: false` by default), and to the fakes in `useSidecarSettings.test.tsx`, `LiveView.test.tsx`, `useSidecarStream.test.tsx`; add an `/api/camera/quality` branch to `AppShell.test.tsx`'s `bodyForUrl`.

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: FAIL — `Unable to find an element by: [data-testid="camera-quality"]`

- [ ] **Step 3: Write minimal implementation**

`api.ts`:

```ts
export interface CameraQualityResponse {
  available: boolean
  brightness: number
  contrast: number
  sharpness: number
  capture_fps: number
  verdicts: Record<string, string>
  detail: string
}
```

Add `getCameraQuality(): Promise<CameraQualityResponse>` to `ApiClient`, and
`getCameraQuality: () => request<CameraQualityResponse>('/camera/quality', 'GET')`
to `createApiClient`.

`useSidecarSettings.ts` — add `cameraQuality` state, poll it on the existing health interval (it is cheap and only meaningful while running), and return it.

`AdminPanel.tsx` — render above the settings groups:

```tsx
{cameraQuality?.available && (
  <section className="camera-quality" data-testid="camera-quality">
    <h4>Live image</h4>
    <div className="quality-row">
      {[
        ['Brightness', cameraQuality.brightness, cameraQuality.verdicts.brightness, '110–160'],
        ['Sharpness', cameraQuality.sharpness, cameraQuality.verdicts.sharpness, 'higher is sharper'],
        ['Capture fps', cameraQuality.capture_fps, cameraQuality.verdicts.capture_fps, '≥ 25']
      ].map(([label, value, verdict, hint]) => (
        <div key={String(label)} className="quality-metric">
          <span className="quality-label">{label}</span>
          <span
            className={verdict === 'ok' ? 'quality-value' : 'quality-value bad'}
            data-testid={verdict === 'ok' ? 'quality-ok' : 'quality-low'}
          >
            {value}
          </span>
          <span className="field-hint">{hint}</span>
        </div>
      ))}
    </div>
  </section>
)}
```

Add `.camera-quality`, `.quality-row` (flex, gap 24px), `.quality-metric` (column), `.quality-value` and `.quality-value.bad` (use the existing warning colour token) to `AdminPanel.css`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test` then `npm run typecheck` and `npm run lint`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src
git commit -m "feat: live image quality readout in the Admin panel"
```

**Phase 1 is now independently useful — stop here and verify in the app before continuing.**

---

# Phase 2 — Measured capability and a reviewed recommendation

### Task 7: Camera control settings

**Files:**
- Modify: `sidecar/app/settings.py`, `settings_store.py`, `schemas.py`, `main.py`, `camera.py`
- Modify: `desktop/src/renderer/src/lib/api.ts`, `settingsDefaults.ts`, `settingsFields.ts`
- Test: `sidecar/tests/test_settings.py`, `sidecar/tests/test_camera.py`

**Interfaces:**
- Produces: `Settings.camera_brightness: float | None`, `camera_exposure: float | None`, `camera_autofocus: bool | None`, `camera_focus: float | None`; all in `RESTART_REQUIRED_FIELDS`; `CameraCapture` applies them on `open()`.

- [ ] **Step 1: Write the failing test**

```python
# append to sidecar/tests/test_camera.py
def test_open_applies_configured_controls():
    """Auto exposure and face-tracking autofocus are wrong for a counter:
    the StreamCam's smart AF/AE follows faces, and there is no face here."""
    sets = []

    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value):
            sets.append((prop, value)); return True
        def get(self, prop): return 0
        def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap(),
                      brightness=180.0, exposure=-6.0, autofocus=False, focus=30.0)
    c.open(); c.release()

    assert (cv2.CAP_PROP_BRIGHTNESS, 180.0) in sets
    assert (cv2.CAP_PROP_EXPOSURE, -6.0) in sets
    assert (cv2.CAP_PROP_AUTOFOCUS, 0) in sets
    assert (cv2.CAP_PROP_FOCUS, 30.0) in sets


def test_unset_controls_are_left_alone():
    """None means 'do not touch', so existing behaviour is unchanged."""
    sets = []

    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value):
            sets.append(prop); return True
        def get(self, prop): return 0
        def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open(); c.release()

    assert cv2.CAP_PROP_BRIGHTNESS not in sets
    assert cv2.CAP_PROP_EXPOSURE not in sets
```

Add to `sidecar/tests/test_settings.py::test_settings_defaults`:

```python
    assert s.camera_brightness is None
    assert s.camera_exposure is None
    assert s.camera_autofocus is None
    assert s.camera_focus is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera.py tests/test_settings.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'brightness'`

- [ ] **Step 3: Write minimal implementation**

`settings.py`:

```python
    # Device controls. None means "leave the camera alone", so behaviour is
    # unchanged until calibration proposes values. The StreamCam's automatic
    # focus and exposure track faces; a counter has none, which is why locked
    # manual values suit this app.
    camera_brightness: float | None = None
    camera_exposure: float | None = None
    camera_autofocus: bool | None = None
    camera_focus: float | None = None
```

`settings_store.py` — add all four to `RESTART_REQUIRED_FIELDS` and to `_valid_field`:

```python
    if name in ("camera_brightness", "camera_exposure", "camera_focus"):
        return value is None or isinstance(value, (int, float))
    if name == "camera_autofocus":
        return value is None or isinstance(value, bool)
```

`schemas.py` — add the four to `SettingsPayload` (as `float | None` / `bool | None`) and `SettingsUpdateRequest`.

`main.py` — add them to `_settings_response(...)` and pass them in `_default_source_factory`:

```python
def _default_source_factory(settings: Settings):
    return CameraCapture(
        settings.camera_index, settings.capture_width, settings.capture_height,
        settings.capture_fps,
        brightness=settings.camera_brightness,
        exposure=settings.camera_exposure,
        autofocus=settings.camera_autofocus,
        focus=settings.camera_focus,
    )
```

`camera.py` — accept the four kwargs, store them, and in `open()` after the existing `set` calls:

```python
        # Order matters: turn autofocus off before writing a focus value, or
        # the device may immediately hunt away from it.
        if self._autofocus is not None:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self._autofocus else 0)
        if self._focus is not None:
            self._cap.set(cv2.CAP_PROP_FOCUS, self._focus)
        if self._brightness is not None:
            self._cap.set(cv2.CAP_PROP_BRIGHTNESS, self._brightness)
        if self._exposure is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
```

Mirror the four fields in `api.ts` (`SettingsPayload`), `settingsDefaults.ts` (all `null`), and add them to `settingsFields.ts` only if you want them hand-editable — otherwise leave them out of `SETTINGS_FIELDS` so calibration owns them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest -q` and, in `desktop/`, `npm run typecheck && npm test`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add sidecar desktop/src/renderer/src/lib
git commit -m "feat: camera control settings applied on open"
```

---

### Task 8: Capability probe

**Files:**
- Create: `sidecar/app/camera_caps.py`
- Test: `sidecar/tests/test_camera_caps.py`

**Interfaces:**
- Consumes: `frame_quality`, `measure_fps` (Tasks 1–2).
- Produces: `ControlSupport(brightness: bool, exposure: bool, gain: bool, focus: bool)`; `CameraProfile(device_key, backend, width, height, fps_auto_exposure, fps_capped_exposure, controls, recommended, measured_at)`; `probe_controls(cap, read_frame) -> ControlSupport`.

- [ ] **Step 1: Write the failing test**

```python
# sidecar/tests/test_camera_caps.py
import numpy as np

from app.camera_caps import ControlSupport, probe_controls


class _Cap:
    """A device where only the named controls actually change the image —
    the others accept the write and do nothing, exactly like MSMF's GAIN."""

    def __init__(self, effective):
        self.effective = effective
        self.level = 40.0

    def set(self, prop, value):
        import cv2
        if prop in self.effective:
            self.level = 180.0
        return True   # always True, as OpenCV does

    def get(self, prop):
        return 0.0    # the getter lies; the probe must ignore it


def test_a_control_counts_only_when_the_image_changes():
    """Setting exposure on MSMF changed delivered fps while get() returned the
    old value. Support is proven by the image, never by a getter."""
    import cv2

    cap = _Cap({cv2.CAP_PROP_BRIGHTNESS})
    support = probe_controls(cap, lambda: np.full((8, 8, 3), int(cap.level), dtype=np.uint8))

    assert support.brightness is True
    assert support.gain is False
    assert support.exposure is False


def test_a_device_that_ignores_everything_supports_nothing():
    cap = _Cap(set())
    support = probe_controls(cap, lambda: np.full((8, 8, 3), int(cap.level), dtype=np.uint8))

    assert support == ControlSupport(brightness=False, exposure=False, gain=False, focus=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_caps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.camera_caps'`

- [ ] **Step 3: Write minimal implementation**

```python
# sidecar/app/camera_caps.py
"""Measured camera capability.

Nothing here trusts `cap.get()`. Setting CAP_PROP_EXPOSURE on MSMF took
delivered framerate from 12.3 to 30.3 fps while the getter kept returning the
old value, so a control counts as supported only when the image or the frame
rate visibly changes. That is also what keeps this brand-independent.
"""

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from app.camera_quality import frame_quality

# A control must move mean brightness by at least this much to count. Below it
# we cannot distinguish a real effect from sensor noise.
EFFECT_THRESHOLD = 6.0


@dataclass
class ControlSupport:
    brightness: bool = False
    exposure: bool = False
    gain: bool = False
    focus: bool = False


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


def _brightness(read_frame: Callable[[], np.ndarray], samples: int = 5) -> float:
    vals = []
    for _ in range(samples):
        frame = read_frame()
        if frame is not None:
            vals.append(frame_quality(frame).brightness)
    return sum(vals) / len(vals) if vals else 0.0


def probe_controls(cap, read_frame: Callable[[], np.ndarray]) -> ControlSupport:
    """Which controls actually do something on this device."""
    support = ControlSupport()
    baseline = _brightness(read_frame)
    for prop, name, value in (
        (cv2.CAP_PROP_BRIGHTNESS, "brightness", 255),
        (cv2.CAP_PROP_EXPOSURE, "exposure", -3),
        (cv2.CAP_PROP_GAIN, "gain", 255),
    ):
        before = _brightness(read_frame)
        cap.set(prop, value)
        after = _brightness(read_frame)
        setattr(support, name, abs(after - before) >= EFFECT_THRESHOLD)
        cap.set(prop, cap.get(prop))
    # Focus does not change mean brightness, so judge it on sharpness instead.
    before_sharp = frame_quality(read_frame()).sharpness
    cap.set(cv2.CAP_PROP_FOCUS, 30)
    after_sharp = frame_quality(read_frame()).sharpness
    support.focus = abs(after_sharp - before_sharp) >= 1.0
    _ = baseline
    return support
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_caps.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_caps.py sidecar/tests/test_camera_caps.py
git commit -m "feat: measure which camera controls actually work"
```

---

### Task 9: The derivation function

**Files:**
- Create: `sidecar/app/camera_derive.py`
- Test: `sidecar/tests/test_camera_derive.py`

**Interfaces:**
- Consumes: `CameraProfile`, `ControlSupport` (Task 8); thresholds (Task 1).
- Produces: `derive_camera_settings(profile: CameraProfile, measured_brightness: float, near_conf: float | None = None, far_conf: float | None = None, imgsz: int = 640) -> dict` — a settings patch, `{}` when nothing should change.

- [ ] **Step 1: Write the failing test**

```python
# sidecar/tests/test_camera_derive.py
from app.camera_caps import CameraProfile, ControlSupport
from app.camera_derive import derive_camera_settings


def _profile(**over):
    base = dict(
        device_key="Logitech StreamCam:1:1920x1080", backend="msmf",
        width=1280, height=720,
        fps_auto_exposure=12.3, fps_capped_exposure=30.3,
        controls=ControlSupport(brightness=True, exposure=True, gain=False, focus=False),
        recommended={}, measured_at=0.0,
    )
    base.update(over)
    return CameraProfile(**base)


def test_a_dark_streamcam_gets_exposure_capped_and_brightness_raised():
    """The measured case: 12 fps on auto exposure, 30 with it capped."""
    patch = derive_camera_settings(_profile(), measured_brightness=23.0)

    assert patch["camera_exposure"] is not None
    assert patch["camera_brightness"] is not None
    assert patch["camera_autofocus"] is False   # face-tracking AF is wrong here


def test_nothing_is_proposed_for_a_camera_that_is_already_good():
    patch = derive_camera_settings(
        _profile(fps_auto_exposure=30.0), measured_brightness=130.0
    )
    assert "camera_brightness" not in patch
    assert "camera_exposure" not in patch


def test_no_settings_are_invented_for_a_device_that_supports_nothing():
    patch = derive_camera_settings(
        _profile(controls=ControlSupport()), measured_brightness=23.0
    )
    assert "camera_brightness" not in patch
    assert "camera_exposure" not in patch


def test_exposure_is_not_lengthened_below_the_fps_floor():
    """Longer exposure buys brightness and costs frames; 1/4 s capped the
    camera at 4 fps. Brightness must not be bought below FPS_MIN."""
    patch = derive_camera_settings(
        _profile(fps_capped_exposure=10.0, fps_auto_exposure=10.0),
        measured_brightness=23.0,
    )
    assert "camera_exposure" not in patch


def test_a_weak_far_confidence_proposes_a_larger_inference_size():
    """A distant SKU is a small SKU; imgsz is the cheapest lever, affordable
    now that CUDA runs the model at ~20 ms."""
    patch = derive_camera_settings(
        _profile(), measured_brightness=130.0, near_conf=0.85, far_conf=0.35, imgsz=640
    )
    assert patch["imgsz"] > 640
    assert patch["imgsz"] % 32 == 0


def test_a_consistent_near_and_far_confidence_leaves_imgsz_alone():
    patch = derive_camera_settings(
        _profile(), measured_brightness=130.0, near_conf=0.85, far_conf=0.80, imgsz=640
    )
    assert "imgsz" not in patch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_derive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.camera_derive'`

- [ ] **Step 3: Write minimal implementation**

```python
# sidecar/app/camera_derive.py
"""Turn a measured CameraProfile into a settings patch.

Pure and total: no I/O, no device. All the judgement lives here so it can be
tested against hand-written profiles.

The objective, once: raise brightness toward BRIGHTNESS_TARGET subject to
measured fps >= FPS_MIN. Exposure is preferred because it is real light rather
than amplification, but it is also what costs frames, so the fps floor gates it.
"""

from app.camera_caps import CameraProfile
from app.camera_quality import BRIGHTNESS_MIN, BRIGHTNESS_TARGET, FPS_MIN

# Shorter than this and the image goes dark faster than brightness can rescue.
EXPOSURE_CAPPED = -6.0
BRIGHTNESS_BOOST = 180.0
# A far confidence this far below near means the model is losing small objects.
FAR_CONF_GAP = 0.25
IMGSZ_STEP = 320
IMGSZ_MAX = 1280


def derive_camera_settings(
    profile: CameraProfile,
    measured_brightness: float,
    near_conf: float | None = None,
    far_conf: float | None = None,
    imgsz: int = 640,
) -> dict:
    patch: dict = {}

    # The StreamCam's smart AF/AE follows faces. A counter has none, so auto
    # has nothing to lock onto and hunts. Lock it whenever we can.
    if profile.controls.focus:
        patch["camera_autofocus"] = False
    elif profile.controls.exposure or profile.controls.brightness:
        # Even without focus control, disabling AF stops the hunting.
        patch["camera_autofocus"] = False

    too_dark = measured_brightness < BRIGHTNESS_MIN
    if too_dark and profile.controls.exposure and profile.fps_capped_exposure >= FPS_MIN:
        patch["camera_exposure"] = EXPOSURE_CAPPED
    if too_dark and profile.controls.brightness:
        patch["camera_brightness"] = BRIGHTNESS_BOOST

    # A distant item is a small item. Raising imgsz keeps more of it, and CUDA
    # made that affordable (~20 ms/frame on the custom model).
    if near_conf is not None and far_conf is not None:
        if near_conf - far_conf >= FAR_CONF_GAP and imgsz < IMGSZ_MAX:
            patch["imgsz"] = min(imgsz + IMGSZ_STEP, IMGSZ_MAX)

    _ = BRIGHTNESS_TARGET
    return patch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_derive.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_derive.py sidecar/tests/test_camera_derive.py
git commit -m "feat: derive camera settings from a measured profile"
```

---

### Task 10: Calibrate and apply endpoints

**Files:**
- Create: `sidecar/app/camera_profiles.py`
- Modify: `sidecar/app/main.py`, `sidecar/app/schemas.py`
- Test: `sidecar/tests/test_camera_calibrate_api.py` (new)

**Interfaces:**
- Consumes: Tasks 8–9.
- Produces: `POST /api/camera/calibrate` → `CameraProfileResponse`; `POST /api/camera/profile/apply` → `SettingsResponse`; `AppState.calibrator: Callable[[], CameraProfile]` injection seam; `load_profiles(path)` / `save_profile(profile, path)` in `camera_profiles.py`.

- [ ] **Step 1: Write the failing test**

```python
# sidecar/tests/test_camera_calibrate_api.py
from fastapi.testclient import TestClient

from app.camera_caps import CameraProfile, ControlSupport
from app.main import AppState, build_app


def _profile():
    return CameraProfile(
        device_key="cam:1", backend="msmf", width=1280, height=720,
        fps_auto_exposure=12.3, fps_capped_exposure=30.3,
        controls=ControlSupport(brightness=True, exposure=True),
        recommended={"camera_exposure": -6.0, "camera_brightness": 180.0},
        measured_at=1.0,
    )


def _client(tmp_path):
    state = AppState(settings_path=str(tmp_path / "s.json"), db_path=":memory:",
                     calibrator=lambda: _profile())
    return TestClient(build_app(lambda: state)), state


def test_calibrate_returns_a_profile_without_applying_it(tmp_path):
    """Review-first: the operator sees the evidence before anything changes."""
    client, state = _client(tmp_path)
    body = client.post("/api/camera/calibrate").json()

    assert body["recommended"]["camera_exposure"] == -6.0
    assert body["fps_capped_exposure"] == 30.3
    assert state.settings.camera_exposure is None   # untouched


def test_apply_writes_the_recommendation(tmp_path):
    client, state = _client(tmp_path)
    client.post("/api/camera/calibrate")
    r = client.post("/api/camera/profile/apply")

    assert r.status_code == 200
    assert state.settings.camera_exposure == -6.0
    assert state.settings.camera_brightness == 180.0


def test_calibrate_is_refused_while_capture_is_running(tmp_path):
    """The device is exclusive; calibrating would fight the pipeline."""
    client, state = _client(tmp_path)
    state.state = "running"
    r = client.post("/api/camera/calibrate")

    assert r.status_code == 409


def test_apply_without_a_profile_is_a_404(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/api/camera/profile/apply").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_calibrate_api.py -v`
Expected: FAIL — `TypeError: AppState() got an unexpected keyword argument 'calibrator'`

- [ ] **Step 3: Write minimal implementation**

`camera_profiles.py` — mirror `settings_store`'s atomic write and tolerant read:

```python
"""Persist CameraProfiles to data/camera_profiles.json.

Same shape as settings_store: atomic write via a temp file plus os.replace, and
a missing or corrupt file yields no profiles rather than crashing startup.
"""

import json
import os
from dataclasses import asdict

from app.camera_caps import CameraProfile, ControlSupport

DEFAULT_PROFILES_PATH = "data/camera_profiles.json"


def load_profiles(path: str = DEFAULT_PROFILES_PATH) -> dict[str, CameraProfile]:
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for key, value in (raw or {}).items():
        try:
            value["controls"] = ControlSupport(**value.get("controls", {}))
            out[key] = CameraProfile(**value)
        except (TypeError, ValueError):
            continue
    return out


def save_profile(profile: CameraProfile, path: str = DEFAULT_PROFILES_PATH) -> None:
    profiles = load_profiles(path)
    profiles[profile.device_key] = profile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({k: asdict(v) for k, v in profiles.items()}, fh, indent=2)
    os.replace(tmp, path)
```

`schemas.py`:

```python
class ControlSupportPayload(BaseModel):
    brightness: bool = False
    exposure: bool = False
    gain: bool = False
    focus: bool = False


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
```

`main.py` — add to `AppState`:

```python
    # Injection seam, like camera_lister: tests supply a profile instead of
    # opening a device.
    calibrator: Callable[[], CameraProfile] | None = None
    last_profile: CameraProfile | None = None
```

and the routes:

```python
    @app.post("/api/camera/calibrate", response_model=CameraProfileResponse)
    async def calibrate_camera():
        """Measure the camera and return a recommendation. Applies nothing —
        the operator reviews it first."""
        if state.state == "running":
            raise HTTPException(
                status_code=409,
                detail="Stop capture before calibrating; the camera is exclusive.",
            )
        if state.calibrator is None:
            raise HTTPException(status_code=503, detail="No calibrator configured.")
        profile = await run_in_threadpool(state.calibrator)
        state.last_profile = profile
        save_profile(profile)
        return CameraProfileResponse(**asdict(profile))

    @app.post("/api/camera/profile/apply", response_model=SettingsResponse)
    async def apply_camera_profile():
        if state.last_profile is None:
            raise HTTPException(status_code=404, detail="Calibrate the camera first.")
        return _apply_settings_patch(state, state.last_profile.recommended)
```

Import `asdict` from `dataclasses`, `CameraProfile` from `camera_caps`, and `save_profile` from `camera_profiles`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_calibrate_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_profiles.py sidecar/app/main.py sidecar/app/schemas.py sidecar/tests/test_camera_calibrate_api.py
git commit -m "feat: calibrate and apply camera profile endpoints"
```

---

### Task 11: The real calibrator

**Files:**
- Modify: `sidecar/app/camera_caps.py`, `sidecar/app/main.py`
- Test: `sidecar/tests/test_camera_caps.py`

**Interfaces:**
- Consumes: `probe_controls`, `measure_fps`, `derive_camera_settings`.
- Produces: `calibrate(index, width, height, open_device=..., sample_seconds=3.0) -> CameraProfile`, wired as `AppState.calibrator`'s default.

- [ ] **Step 1: Write the failing test**

```python
# append to sidecar/tests/test_camera_caps.py
from app.camera_caps import calibrate


def test_calibrate_measures_both_exposure_modes():
    """12 fps on auto, 30 with exposure capped — the difference is the whole
    reason to calibrate."""
    class _Cap:
        def __init__(self): self.capped = False
        def isOpened(self): return True
        def set(self, prop, value):
            import cv2
            if prop == cv2.CAP_PROP_EXPOSURE: self.capped = True
            return True
        def get(self, prop): return 0.0
        def read(self):
            import numpy as np
            return True, np.full((8, 8, 3), 40, dtype=np.uint8)
        def release(self): pass

    cap = _Cap()
    profile = calibrate(1, 1280, 720, open_device=lambda i, b: cap,
                        device_name="Fake Cam", sample_seconds=0.05)

    assert profile.device_key.startswith("Fake Cam")
    assert profile.fps_auto_exposure > 0
    assert profile.fps_capped_exposure > 0
    assert isinstance(profile.recommended, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_caps.py -v`
Expected: FAIL — `ImportError: cannot import name 'calibrate'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to sidecar/app/camera_caps.py
def _default_open(index: int, backend: int):
    return cv2.VideoCapture(index, backend)


def calibrate(
    index: int,
    width: int,
    height: int,
    open_device: Callable[[int, int], object] = _default_open,
    device_name: str = "",
    sample_seconds: float = 3.0,
) -> CameraProfile:
    """Measure one camera and recommend settings.

    Opens the device exclusively, so the caller must have stopped capture.
    """
    from app.camera_derive import derive_camera_settings
    from app.camera_quality import measure_fps

    cap = open_device(index, cv2.CAP_MSMF)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(10):
            cap.read()

        def read_ok() -> bool:
            ok, _ = cap.read()
            return bool(ok)

        def read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        fps_auto = measure_fps(read_ok, seconds=sample_seconds)
        brightness = _brightness(read_frame)
        controls = probe_controls(cap, read_frame)

        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        fps_capped = measure_fps(read_ok, seconds=sample_seconds)

        profile = CameraProfile(
            device_key=f"{device_name}:{index}:{width}x{height}",
            backend="msmf", width=width, height=height,
            fps_auto_exposure=round(fps_auto, 1),
            fps_capped_exposure=round(fps_capped, 1),
            controls=controls, measured_at=time.time(),
        )
        profile.recommended = derive_camera_settings(profile, brightness)
        return profile
    finally:
        release = getattr(cap, "release", None)
        if callable(release):
            release()
```

In `main.py`'s `AppState.__post_init__`, default the calibrator when unset:

```python
        if self.calibrator is None:
            self.calibrator = lambda: calibrate(
                self.settings.camera_index,
                self.settings.capture_width,
                self.settings.capture_height,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_caps.py sidecar/app/main.py sidecar/tests/test_camera_caps.py
git commit -m "feat: real camera calibration sweep"
```

---

### Task 12: Review card in Admin

**Files:**
- Modify: `desktop/src/renderer/src/lib/api.ts`, `hooks/useSidecarSettings.ts`, `views/AdminPanel.tsx`, `views/AdminPanel.css`
- Test: `desktop/src/renderer/src/views/AdminPanel.test.tsx`

**Interfaces:**
- Consumes: Task 10's endpoints.
- Produces: `ApiClient.calibrateCamera()`, `ApiClient.applyCameraProfile()`; `useSidecarSettings` exposes `calibrate()`, `applyProfile()`, `profile`, `calibrating`.

- [ ] **Step 1: Write the failing test**

```tsx
it('shows the calibration result with its measured evidence, and applies on request', async () => {
  const { deps, api } = makeDeps('idle', {
    calibrateCamera: vi.fn(async () => ({
      device_key: 'Logitech StreamCam:1:1280x720', backend: 'msmf',
      width: 1280, height: 720,
      fps_auto_exposure: 12.3, fps_capped_exposure: 30.3,
      controls: { brightness: true, exposure: true, gain: false, focus: false },
      recommended: { camera_exposure: -6, camera_brightness: 180 },
      measured_at: 1
    }))
  })
  render(<AdminPanel port={8765} deps={deps} />)

  await userEvent.click(await screen.findByTestId('calibrate-camera'))

  const card = await screen.findByTestId('calibration-result')
  expect(card).toHaveTextContent('30.3')   // the measured evidence
  expect(card).toHaveTextContent('12.3')

  await userEvent.click(within(card).getByTestId('apply-profile'))
  expect(api.applyCameraProfile).toHaveBeenCalled()
})

it('disables calibration while capture is running', async () => {
  const { deps } = makeDeps('running')
  render(<AdminPanel port={8765} deps={deps} />)
  expect(await screen.findByTestId('calibrate-camera')).toBeDisabled()
})
```

Add `calibrateCamera` and `applyCameraProfile` to every `ApiClient` fake.

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: FAIL — no `calibrate-camera` element

- [ ] **Step 3: Write minimal implementation**

`api.ts` — add `CameraProfileResponse` mirroring the Python model, and the two methods (`POST /camera/calibrate`, `POST /camera/profile/apply`).

`useSidecarSettings.ts` — `calibrate()` sets `calibrating`, calls the API, stores `profile`; `applyProfile()` calls apply then `load()`.

`AdminPanel.tsx` — beside the camera field:

```tsx
<button
  className="btn-outline btn-small"
  disabled={running || calibrating}
  onClick={() => void calibrate()}
  data-testid="calibrate-camera"
  title={running ? 'Stop capture to calibrate' : 'Measure this camera'}
>
  {calibrating ? <Spinner /> : null} Calibrate camera
</button>

{profile && (
  <div className="calibration-result" data-testid="calibration-result">
    <p className="field-hint">
      Measured {profile.fps_auto_exposure} fps on automatic exposure,{' '}
      {profile.fps_capped_exposure} fps with it capped.
    </p>
    <ul>
      {Object.entries(profile.recommended).map(([k, v]) => (
        <li key={k}>{k}: {String(v)}</li>
      ))}
    </ul>
    <button className="btn-primary btn-small" data-testid="apply-profile"
            onClick={() => void applyProfile()}>
      Apply these settings
    </button>
  </div>
)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test && npm run typecheck && npm run lint`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src
git commit -m "feat: camera calibration review card in Admin"
```

**Phase 2 complete — verify in the real app before Phase 3.**

---

# Phase 3 — The wizard

### Task 13: Wizard shell and steps 1–4

**Files:**
- Create: `desktop/src/renderer/src/views/CameraSetup.tsx`, `CameraSetup.css`
- Modify: `desktop/src/renderer/src/components/AppShell.tsx`
- Test: `desktop/src/renderer/src/views/CameraSetup.test.tsx` (new)

**Interfaces:**
- Consumes: `getCameraQuality`, `calibrateCamera`, `applyCameraProfile`.
- Produces: `<CameraSetup port={number} deps={SettingsDeps} onDone={() => void} />`.

- [ ] **Step 1: Write the failing test**

```tsx
// desktop/src/renderer/src/views/CameraSetup.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CameraSetup } from './CameraSetup'

function deps(quality: Partial<Record<string, unknown>> = {}) {
  const api = {
    getCameraQuality: vi.fn(async () => ({
      available: true, brightness: 23, contrast: 27, sharpness: 4.8,
      capture_fps: 12,
      verdicts: { brightness: 'low', sharpness: 'low', capture_fps: 'low' },
      detail: '', ...quality
    }))
  }
  return { deps: { apiFactory: () => api as never }, api }
}

describe('CameraSetup', () => {
  it('tells the operator the frame is too dark, with the number', async () => {
    const { deps: d } = deps()
    render(<CameraSetup port={8765} deps={d as never} onDone={() => {}} />)

    await userEvent.click(await screen.findByTestId('step-next'))   // past framing
    const lighting = await screen.findByTestId('step-lighting')
    expect(lighting).toHaveTextContent('23')
    expect(within(lighting).getByTestId('step-verdict')).toHaveTextContent(/too dark/i)
  })

  it('never blocks the operator from continuing on a failed check', async () => {
    // Advisory, not enforced: a demo must be able to proceed.
    const { deps: d } = deps()
    render(<CameraSetup port={8765} deps={d as never} onDone={() => {}} />)

    await userEvent.click(await screen.findByTestId('step-next'))
    expect(await screen.findByTestId('step-next')).toBeEnabled()
  })

  it('passes the lighting step when the frame is bright enough', async () => {
    const { deps: d } = deps({ brightness: 130, verdicts: { brightness: 'ok' } })
    render(<CameraSetup port={8765} deps={d as never} onDone={() => {}} />)

    await userEvent.click(await screen.findByTestId('step-next'))
    expect(await screen.findByTestId('step-verdict')).toHaveTextContent(/looks good/i)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/renderer/src/views/CameraSetup.test.tsx`
Expected: FAIL — cannot resolve `./CameraSetup`

- [ ] **Step 3: Write minimal implementation**

```tsx
// desktop/src/renderer/src/views/CameraSetup.tsx
import { useEffect, useMemo, useRef, useState } from 'react'
import { createApiClient, type ApiClient, type CameraQualityResponse } from '../lib/api'
import type { SettingsDeps } from '../hooks/useSidecarSettings'

interface Step {
  id: string
  title: string
  instruction: string
  // null => nothing to judge yet (the framing step)
  verdict: (q: CameraQualityResponse | null) => { ok: boolean; text: string } | null
}

const STEPS: Step[] = [
  {
    id: 'framing',
    title: 'Framing',
    instruction: 'Point the camera at the counter and place one item where you will scan it.',
    verdict: () => null
  },
  {
    id: 'lighting',
    title: 'Lighting',
    instruction: 'Light the counter. Aim for a bright, evenly lit surface.',
    verdict: (q) =>
      q === null
        ? null
        : q.verdicts.brightness === 'ok'
          ? { ok: true, text: `Brightness ${q.brightness} — looks good.` }
          : {
              ok: false,
              text:
                q.brightness < 110
                  ? `Brightness ${q.brightness} — too dark. Detection will be unreliable below 110.`
                  : `Brightness ${q.brightness} — too bright; detail is washing out.`
            }
  },
  {
    id: 'focus',
    title: 'Focus',
    instruction: 'Keep the scene still. If the image drifts in and out, autofocus is hunting.',
    verdict: (q) =>
      q === null
        ? null
        : q.verdicts.sharpness === 'ok'
          ? { ok: true, text: `Sharpness ${q.sharpness} — looks good.` }
          : {
              ok: false,
              text: `Sharpness ${q.sharpness} — soft. This camera's autofocus follows faces, and a counter has none. Lock focus in Logi G HUB.`
            }
  },
  {
    id: 'framerate',
    title: 'Frame rate',
    instruction: 'Measured frames per second, not the rate we asked for.',
    verdict: (q) =>
      q === null
        ? null
        : q.verdicts.capture_fps === 'ok'
          ? { ok: true, text: `${q.capture_fps} fps — looks good.` }
          : {
              ok: false,
              text: `${q.capture_fps} fps — below 25. Long exposure in low light caps the frame rate; calibrating can cap exposure instead.`
            }
  }
]

export function CameraSetup({
  port,
  deps = {},
  onDone
}: {
  port: number
  deps?: SettingsDeps
  onDone: () => void
}): JSX.Element {
  const apiFactory = deps.apiFactory ?? createApiClient
  const apiRef = useRef<ApiClient | null>(null)
  const [index, setIndex] = useState(0)
  const [quality, setQuality] = useState<CameraQualityResponse | null>(null)

  useEffect(() => {
    const api = apiFactory(port)
    apiRef.current = api
    let cancelled = false
    const poll = async (): Promise<void> => {
      try {
        const q = await api.getCameraQuality()
        if (!cancelled) setQuality(q)
      } catch {
        // A failed poll leaves the last reading; the wizard is advisory.
      }
    }
    void poll()
    const t = setInterval(() => void poll(), 1000)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [port, apiFactory])

  const step = STEPS[index]
  const verdict = useMemo(() => step.verdict(quality), [step, quality])
  const last = index === STEPS.length - 1

  return (
    <div className="camera-setup" data-testid={`step-${step.id}`}>
      <h3>
        Step {index + 1} of {STEPS.length}: {step.title}
      </h3>
      <p className="field-hint">{step.instruction}</p>

      {verdict && (
        <p
          className={verdict.ok ? 'setup-ok' : 'admin-warning'}
          data-testid="step-verdict"
        >
          {verdict.text}
        </p>
      )}

      <div className="setup-actions">
        <button
          className="btn-outline btn-small"
          disabled={index === 0}
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          data-testid="step-back"
        >
          Back
        </button>
        {/* Always enabled: every check is advisory, never a gate. */}
        <button
          className="btn-primary btn-small"
          onClick={() => (last ? onDone() : setIndex((i) => i + 1))}
          data-testid="step-next"
        >
          {last ? 'Done' : 'Next'}
        </button>
      </div>
    </div>
  )
}
```

Add `.camera-setup`, `.setup-actions` (flex, gap 8px) and `.setup-ok` (accent colour) to `CameraSetup.css`, and render `<CameraSetup>` from `AppShell` when the Admin panel's "Set up camera" button sets that view.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test && npm run typecheck && npm run lint`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src
git commit -m "feat: camera setup wizard steps 1-4"
```

---

### Task 14: Step 5 — detection at near and far

**Files:**
- Modify: `sidecar/app/main.py`, `sidecar/app/schemas.py`, `desktop/.../CameraSetup.tsx`
- Test: `sidecar/tests/test_camera_distance_api.py` (new), `CameraSetup.test.tsx`

**Interfaces:**
- Produces: `POST /api/camera/distance-sample` → `DistanceSampleResponse(best_conf: float, detections: int, cls: str)`; the wizard sends `{"position": "near"}` then `{"position": "far"}` and passes both to apply, which re-derives with `near_conf`/`far_conf`.

- [ ] **Step 1: Write the failing test**

```python
# sidecar/tests/test_camera_distance_api.py
import numpy as np
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.schemas import Detection


class _Src:
    width, height, fps = 128, 96, 30.0
    measured_fps = 28.0

    def open(self): return True
    def latest(self): return (1, np.full((96, 128, 3), 130, dtype=np.uint8))
    def release(self): pass


class _Det:
    names = {0: "milo"}

    def infer(self, frame):
        return [Detection(track_id=1, cls="milo", conf=0.85, box=(0.1, 0.1, 0.2, 0.2))]


def _client(tmp_path):
    state = AppState(
        settings_path=str(tmp_path / "s.json"),
        source_factory=lambda s: _Src(),
        detector_factory=lambda s, d: _Det(),
        db_path=":memory:",
    )
    return TestClient(build_app(lambda: state)), state


def test_distance_sample_reports_the_best_confidence_seen(tmp_path):
    """Step 5's number: a near/far gap means the model is losing small
    objects, which is what drives the imgsz proposal."""
    client, _ = _client(tmp_path)
    with client:
        client.post("/api/capture/start")
        body = client.post(
            "/api/camera/distance-sample", json={"position": "near", "seconds": 0.2}
        ).json()
        client.post("/api/capture/stop")

    assert body["best_conf"] == 0.85
    assert body["detections"] >= 1
    assert body["cls"] == "milo"


def test_distance_sample_requires_capture_to_be_running(tmp_path):
    """There are no detections to sample when the pipeline is not running."""
    client, _ = _client(tmp_path)
    r = client.post("/api/camera/distance-sample", json={"position": "near"})

    assert r.status_code == 409


def test_both_positions_are_remembered_for_the_recommendation(tmp_path):
    client, state = _client(tmp_path)
    with client:
        client.post("/api/capture/start")
        client.post("/api/camera/distance-sample", json={"position": "near", "seconds": 0.2})
        client.post("/api/camera/distance-sample", json={"position": "far", "seconds": 0.2})
        client.post("/api/capture/stop")

    assert state.near_conf == 0.85
    assert state.far_conf == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera_distance_api.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Write minimal implementation**

`schemas.py`:

```python
class DistanceSampleRequest(BaseModel):
    position: Literal["near", "far"]
    seconds: float = Field(default=2.0, ge=0.1, le=10.0)


class DistanceSampleResponse(BaseModel):
    position: str
    best_conf: float = 0.0
    detections: int = 0
    cls: str = ""
```

`pipeline.py` — expose what the preview thread already keeps:

```python
    def current_detections(self) -> list[Detection]:
        """The most recent detections, for sampling without running inference."""
        with self._state_lock:
            return list(self._latest_detections)
```

`main.py` — add `near_conf: float | None = None` and `far_conf: float | None = None`
to `AppState`, then:

```python
    @app.post("/api/camera/distance-sample", response_model=DistanceSampleResponse)
    async def distance_sample(body: DistanceSampleRequest):
        """Best detection confidence over a short window, for one position.

        A distant item is a small item, so a near/far gap is the evidence that
        the model is losing small objects — which is what raises imgsz.
        """
        if state.state != "running" or state.pipeline is None:
            raise HTTPException(
                status_code=409, detail="Start capture before sampling detections."
            )

        def _sample():
            import time as _time

            best, count, label = 0.0, 0, ""
            deadline = _time.monotonic() + body.seconds
            while _time.monotonic() < deadline:
                for d in state.pipeline.current_detections():
                    count += 1
                    if d.conf > best:
                        best, label = d.conf, d.cls
                _time.sleep(0.05)
            return best, count, label

        best, count, label = await run_in_threadpool(_sample)
        if body.position == "near":
            state.near_conf = best
        else:
            state.far_conf = best
        return DistanceSampleResponse(
            position=body.position, best_conf=round(best, 2), detections=count, cls=label
        )
```

`main.py` — re-derive on apply so the distance evidence is actually used:

```python
    @app.post("/api/camera/profile/apply", response_model=SettingsResponse)
    async def apply_camera_profile():
        if state.last_profile is None:
            raise HTTPException(status_code=404, detail="Calibrate the camera first.")
        patch = dict(state.last_profile.recommended)
        if state.near_conf is not None and state.far_conf is not None:
            patch.update(
                derive_camera_settings(
                    state.last_profile,
                    measured_brightness=BRIGHTNESS_TARGET,
                    near_conf=state.near_conf,
                    far_conf=state.far_conf,
                    imgsz=state.settings.imgsz,
                )
            )
        return _apply_settings_patch(state, patch)
```

`api.ts` — add the client method:

```ts
sampleDistance: (position: 'near' | 'far') =>
  request<DistanceSampleResponse>('/camera/distance-sample', 'POST', { position })
```

`CameraSetup.tsx` — append a fifth `STEPS` entry whose body renders two buttons
calling `sampleDistance('near')` and `sampleDistance('far')`, stores both results,
and warns on a gap:

```tsx
{nearConf !== null && farConf !== null && nearConf - farConf >= 0.25 && (
  <p className="admin-warning" data-testid="distance-gap">
    Near {nearConf.toFixed(2)} vs far {farConf.toFixed(2)}. Distant items are being
    missed — a larger inference size will be proposed. If the gap persists after
    applying it, the training set needs examples at this distance.
  </p>
)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest -q` and `npm test`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add sidecar desktop/src/renderer/src
git commit -m "feat: measure detection at near and far distance"
```

---

### Task 15: Documentation

**Files:**
- Modify: `sidecar/README.md`, `docs/MODEL_TRAINING.md`, `CLAUDE.md`

- [ ] **Step 1: Write the docs**

`sidecar/README.md` — a "Camera setup" section: what the wizard checks, the thresholds and why (brightness 110–160, fps ≥ 25), that the StreamCam's smart AF/AE follows faces so manual is correct here, and that MSMF cannot set focus — use Logi G HUB for that one.

`docs/MODEL_TRAINING.md` — a short "Detection at distance" section: a distant SKU is a small SKU; `imgsz` and capture resolution widen the range a little, but a model strong near and weak far has a **dataset** problem. Record that the training set needs examples at the distances the counter uses, with scale augmentation, and that step 5's near/far numbers are the evidence.

`CLAUDE.md` — add `camera_quality.py`, `camera_caps.py`, `camera_derive.py`, `camera_profiles.py` to the sidecar module list; add the new endpoints to the routes line; record the invariant that **OpenCV property getters are not trustworthy — capability is established by measurement**.

- [ ] **Step 2: Verify the claims**

Check every referenced file, endpoint and constant exists. Run `.venv/Scripts/python.exe -m pytest -q` and `npm test`.

- [ ] **Step 3: Commit**

```bash
git add sidecar/README.md docs/MODEL_TRAINING.md CLAUDE.md
git commit -m "docs: camera setup and detection-at-distance guidance"
```

---

## Verification

After each phase, in the real app (not only tests):

1. `cd desktop && npm run build && npm run dev`
2. Phase 1 — Admin shows brightness/sharpness/fps while capturing, and `capture_fps` now matches a stopwatch rather than the configured 60.
3. Phase 2 — Calibrate with capture stopped; the card reports both measured framerates; Apply changes the settings; capture restarts brighter and faster.
4. Phase 3 — Walk the wizard end to end with a real product on the counter.
