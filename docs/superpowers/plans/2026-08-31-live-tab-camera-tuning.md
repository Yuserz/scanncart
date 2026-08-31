# Live Tab Camera Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the camera-behaviour settings and the calibration card from the Admin tab to the Live tab, and make the five fields that can respond to a running camera actually do so.

**Architecture:** Five fields move from `RESTART_REQUIRED_FIELDS` to `HOT_RELOADABLE_FIELDS`. `_apply_settings_patch()` — already the single funnel every settings write passes through — gains a `persist` flag and a step that pushes camera controls to the capture thread and confidence to the detector. The renderer grows a collapsible `CameraTuning` card in the Live tab's existing side rail, which applies changes live (debounced) and persists only on an explicit Save.

**Tech Stack:** Python 3.12 / FastAPI / OpenCV / Ultralytics (sidecar); Electron + React 18 + TypeScript / Vitest / Testing Library (desktop).

**Spec:** `docs/superpowers/specs/2026-08-31-live-tab-camera-tuning-design.md`

## Global Constraints

- Sidecar tests always run against fakes — never a real camera, GPU, or network. Do not add tests that assume Ultralytics/OpenCV hardware is present.
- Desktop tests inject `spawnFn`/`wsFactory`/`apiFactory`/`streamFactory` rather than mocking modules.
- The WS and settings contracts are duplicated by hand between `sidecar/app/schemas.py` + `settings_store.py` and `desktop/src/renderer/src/lib/api.ts` + `settingsFields.ts` + `settingsDefaults.ts`. Any change to one side must be mirrored in the other in the same task.
- `data/settings.json` is written atomically (temp file + `os.replace`) via `save_settings`. Never write it directly.
- The Roboflow API key is never placed on `Settings` and never serialized. Do not add it to any response.
- Field placement lives in exactly one list. A field must not be renderable from both `LiveView` and `AdminPanel`.
- Existing suite must stay green: 476 sidecar tests, 96 desktop tests, `npm run typecheck` and `npm run lint` clean.

---

## File Structure

**Created:**
- `desktop/src/renderer/src/components/CameraTuning.tsx` — the side-rail card: quality readout, three control sections, calibrate button, Save/Revert.
- `desktop/src/renderer/src/components/CameraTuning.css` — its styles.
- `desktop/src/renderer/src/components/CameraTuning.test.tsx` — its tests.
- `sidecar/tests/test_live_settings.py` — hot-apply and persistence-mode tests.

**Modified:**
- `sidecar/app/camera.py` — extract `_write_controls`, add `set_controls`/`_drain_controls`.
- `sidecar/app/inference.py` — `set_conf` on both detectors.
- `sidecar/app/settings_store.py` — move five fields between the two sets.
- `sidecar/app/schemas.py` — bounds on the four camera control fields; `StoredProfileResponse`.
- `sidecar/app/camera_caps.py` — extract `device_key_for`.
- `sidecar/app/main.py` — `persist` flag, `_push_live_settings`, two new routes.
- `desktop/src/renderer/src/lib/api.ts` — three client methods.
- `desktop/src/renderer/src/lib/settingsFields.ts` — four new `FieldMeta`, `FieldGroup.home`.
- `desktop/src/renderer/src/hooks/useSidecarSettings.ts` — `pollHealth` option, stored profile, live-apply/save/revert.
- `desktop/src/renderer/src/views/LiveView.tsx` — mount the card.
- `desktop/src/renderer/src/views/AdminPanel.tsx` — remove the relocated fields, calibration card, quality section.
- `CLAUDE.md` — contract note.

---

# Phase 1 — Sidecar: make the five fields live

## Task 1: Camera controls applied on the capture thread

**Files:**
- Modify: `sidecar/app/camera.py:73-135`
- Test: `sidecar/tests/test_camera.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `CameraCapture.set_controls(**changes) -> None`, accepting any of the keyword names `brightness`, `exposure`, `autofocus`, `focus`. Task 3 calls it.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_camera.py`:

```python
# --- live control changes -------------------------------------------------


class _RecordingCap:
    """Opens fine, yields frames forever, and records every set() with the
    name of the thread that made it."""

    def __init__(self):
        self.sets: list[tuple[int, object, str]] = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value, threading.current_thread().name))
        return True

    def read(self):
        time.sleep(0.001)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _wrote(cap, prop, value) -> bool:
    return any(p == prop and v == value for p, v, _ in cap.sets)


def _wait_for(predicate, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_control_writes_happen_on_the_capture_thread():
    """cv2.VideoCapture is not thread-safe and _loop is calling read() on a
    background thread, so a set() issued from the caller's thread would race
    it. The write must be deferred to the thread that owns the handle."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=140.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 140.0))
        writers = {t for p, _, t in cap.sets if p == cv2.CAP_PROP_BRIGHTNESS}
        assert threading.current_thread().name not in writers
    finally:
        src.release()


def test_set_controls_coalesces_a_fast_drag():
    """A slider drag emits dozens of values. Only the newest matters, and
    applying every one would stall reads behind a queue of set() calls."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=100.0)
        src.set_controls(brightness=110.0)
        src.set_controls(brightness=120.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 120.0))
        written = [v for p, v, _ in cap.sets if p == cv2.CAP_PROP_BRIGHTNESS]
        assert 110.0 not in written
    finally:
        src.release()


def test_controls_set_live_survive_a_reopen():
    """A restart (resolution change, say) rebuilds the handle. Values tuned
    live must come back with it, or a restart silently reverts them."""
    caps = []

    def factory(index):
        cap = _RecordingCap()
        caps.append(cap)
        return cap

    src = CameraCapture(0, 4, 4, 30, cap_factory=factory)
    src.open()
    src.set_controls(brightness=140.0)
    assert _wait_for(lambda: _wrote(caps[0], cv2.CAP_PROP_BRIGHTNESS, 140.0))
    src.release()

    src.open()
    src.release()
    assert _wrote(caps[1], cv2.CAP_PROP_BRIGHTNESS, 140.0)


def test_autofocus_is_written_before_focus_when_set_live():
    """Same ordering open() has always used: a focus value written while
    autofocus is on is immediately hunted away from."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(focus=30.0, autofocus=False)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_FOCUS, 30.0))
        props = [p for p, _, _ in cap.sets]
        assert props.index(cv2.CAP_PROP_AUTOFOCUS) < props.index(cv2.CAP_PROP_FOCUS)
    finally:
        src.release()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_camera.py -k "control or autofocus_is_written" -v`
Expected: FAIL with `AttributeError: 'CameraCapture' object has no attribute 'set_controls'`

- [ ] **Step 3: Write the implementation**

In `sidecar/app/camera.py`, add to `CameraCapture.__init__` after `self._focus = focus`:

```python
        # Control changes queued by another thread, drained by _loop between
        # reads. cv2.VideoCapture is not thread-safe, so set() must never be
        # called from the FastAPI request thread while read() is in flight.
        self._controls_lock = threading.Lock()
        self._pending_controls: dict[str, float | bool | None] = {}
```

Add these three methods to `CameraCapture`:

```python
    def _current_controls(self) -> dict:
        return {
            "autofocus": self._autofocus,
            "focus": self._focus,
            "brightness": self._brightness,
            "exposure": self._exposure,
        }

    @staticmethod
    def _write_controls(cap, controls: dict) -> None:
        """Write device controls in dependency order.

        Autofocus goes first: a focus value written while autofocus is on is
        immediately hunted away from. Keys absent or None mean "leave the
        camera alone" — see Settings.camera_brightness et al.
        """
        if controls.get("autofocus") is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if controls["autofocus"] else 0)
        if controls.get("focus") is not None:
            cap.set(cv2.CAP_PROP_FOCUS, controls["focus"])
        if controls.get("brightness") is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, controls["brightness"])
        if controls.get("exposure") is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, controls["exposure"])

    def set_controls(self, **changes) -> None:
        """Queue control changes for the capture thread.

        Accepts brightness/exposure/autofocus/focus. Updating a dict rather
        than appending to a queue coalesces a fast slider drag to its newest
        value, so the capture thread never works through a backlog.
        """
        with self._controls_lock:
            self._pending_controls.update(changes)

    def _drain_controls(self) -> None:
        with self._controls_lock:
            if not self._pending_controls:
                return
            changes = self._pending_controls
            self._pending_controls = {}
        # Merge onto the instance fields so a later reopen replays them.
        for name, value in changes.items():
            setattr(self, f"_{name}", value)
        # Deliberately outside the lock: cap.set() can block on some
        # backends, and holding the lock across it would stall the caller.
        self._write_controls(self._cap, changes)
```

Replace lines 115–124 of `open()` (the four control blocks) with:

```python
        self._write_controls(self._cap, self._current_controls())
```

In `_loop`, after `self._failing_since = None` and before `self._read_times.append(...)`:

```python
            self._drain_controls()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_camera.py -v`
Expected: PASS, including the pre-existing camera tests.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera.py sidecar/tests/test_camera.py
git commit -m "feat(sidecar): apply camera controls without reopening the device"
```

---

## Task 2: Confidence threshold settable on a live detector

**Files:**
- Modify: `sidecar/app/inference.py:84-100`, `sidecar/app/inference.py:158-172`
- Test: `sidecar/tests/test_inference.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `YoloDetector.set_conf(value: float) -> None` and `RoboflowRemoteDetector.set_conf(value: float) -> None`. Task 3 calls whichever exists via `getattr`.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_inference.py`:

```python
# --- live confidence changes ---------------------------------------------


def test_yolo_detector_conf_is_settable_without_rebuilding():
    """conf reaches ultralytics as a per-call kwarg, so changing it needs no
    new model load — that is what lets the Live tab tune it against a
    running camera."""
    calls = []

    class _Results:
        boxes = None

    class _Model:
        names = {0: "milo"}

        def track(self, frame, **kwargs):
            calls.append(kwargs["conf"])
            return [_Results()]

    det = YoloDetector("yolo11n.pt", "cpu", conf=0.5, model_factory=lambda p: _Model())
    det.infer(np.zeros((8, 8, 3), dtype=np.uint8))
    det.set_conf(0.8)
    det.infer(np.zeros((8, 8, 3), dtype=np.uint8))

    assert calls == [0.5, 0.8]


def test_remote_detector_conf_is_settable_without_rebuilding():
    """The workflow declares no parameters, so the remote path filters
    client-side — changing the threshold is a plain reassignment."""
    det = RoboflowRemoteDetector(client=None, conf=0.5)
    det.set_conf(0.8)
    assert det._conf == 0.8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_inference.py -k "settable_without_rebuilding" -v`
Expected: FAIL with `AttributeError: 'YoloDetector' object has no attribute 'set_conf'`

- [ ] **Step 3: Write the implementation**

Add to `YoloDetector`, immediately after `__init__`:

```python
    def set_conf(self, value: float) -> None:
        """Change the confidence threshold on a running detector.

        `conf` is passed to track() on every call (see infer), so there is
        nothing to rebuild — this is what makes conf_threshold hot-reloadable
        while the rest of the detector's construction arguments are not.
        """
        self._conf = float(value)
```

Add the same method to `RoboflowRemoteDetector`, immediately after `__init__`:

```python
    def set_conf(self, value: float) -> None:
        """Change the confidence threshold on a running detector.

        The workflow declares no parameters, so filtering happens client-side
        in infer() against this value on every response.
        """
        self._conf = float(value)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_inference.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/inference.py sidecar/tests/test_inference.py
git commit -m "feat(sidecar): allow the confidence threshold to change mid-capture"
```

---

## Task 3: Widen the hot-reloadable set and push changes to the running pipeline

**Files:**
- Modify: `sidecar/app/settings_store.py:75-106`, `sidecar/app/main.py:352-365`
- Create: `sidecar/tests/test_live_settings.py`

**Interfaces:**
- Consumes: `CameraCapture.set_controls` (Task 1), `set_conf` (Task 2).
- Produces: `_apply_settings_patch(state, patch, persist=True) -> SettingsResponse`. Task 4 calls it with `persist=False`.

- [ ] **Step 1: Write the failing tests**

Create `sidecar/tests/test_live_settings.py`:

```python
"""Settings that take effect on a running pipeline. Every test uses fakes —
no camera, GPU or network, per the suite's convention."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.schemas import Detection
from app.settings_store import HOT_RELOADABLE_FIELDS, RESTART_REQUIRED_FIELDS


class _Src:
    """Frame source that records the controls pushed to it."""

    width, height, fps = 64, 48, 30.0
    measured_fps = 29.0

    def __init__(self):
        self.controls: dict = {}

    def open(self):
        return True

    def latest(self):
        return (1, np.full((48, 64, 3), 130, dtype=np.uint8))

    def read(self):
        return np.full((48, 64, 3), 130, dtype=np.uint8)

    def set_controls(self, **changes):
        self.controls.update(changes)

    def release(self):
        pass


class _Det:
    names = {0: "milo"}

    def __init__(self):
        self.conf = 0.5

    def infer(self, frame):
        return [Detection(track_id=1, cls="milo", conf=0.9, box=(0.1, 0.1, 0.2, 0.2))]

    def set_conf(self, value):
        self.conf = float(value)


@pytest.fixture
def running(tmp_path):
    """A client with capture started against fakes, exposing the live
    source and detector so a test can assert what reached them."""
    src, det = _Src(), _Det()
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        source_factory=lambda s: src,
        detector_factory=lambda s, d: det,
    )
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        yield client, state, src, det
        client.post("/api/capture/stop")


# --- the field sets ------------------------------------------------------


LIVE_FIELDS = {
    "conf_threshold",
    "camera_brightness",
    "camera_exposure",
    "camera_autofocus",
    "camera_focus",
}


def test_the_five_tunable_fields_are_hot_reloadable():
    assert LIVE_FIELDS <= HOT_RELOADABLE_FIELDS


def test_they_are_no_longer_restart_required():
    assert LIVE_FIELDS.isdisjoint(RESTART_REQUIRED_FIELDS)


def test_fields_that_need_a_reopen_are_still_restart_required():
    """The reopen path is ~30 s on a StreamCam and cannot be avoided."""
    assert {"camera_index", "capture_width", "capture_height", "capture_fps"} <= (
        RESTART_REQUIRED_FIELDS
    )


# --- pushing to the running pipeline -------------------------------------


def test_a_camera_control_patch_reaches_the_open_device(running):
    client, _, src, _ = running
    r = client.patch("/api/settings", json={"camera_brightness": 180.0})

    assert r.status_code == 200
    assert src.controls["brightness"] == 180.0


def test_autofocus_reaches_the_device_as_a_bool(running):
    client, _, src, _ = running
    client.patch("/api/settings", json={"camera_autofocus": False})

    assert src.controls["autofocus"] is False


def test_a_conf_patch_reaches_the_running_detector(running):
    client, _, _, det = running
    client.patch("/api/settings", json={"conf_threshold": 0.8})

    assert det.conf == 0.8


def test_camera_controls_no_longer_409_while_running(running):
    client, _, _, _ = running
    r = client.patch("/api/settings", json={"camera_exposure": -6.0})

    assert r.status_code == 200


def test_a_restart_required_field_still_409s_while_running(running):
    client, _, _, _ = running
    r = client.patch("/api/settings", json={"capture_width": 640})

    assert r.status_code == 409
    assert "stop capture" in r.json()["detail"].lower()


def test_patching_while_idle_touches_no_device(tmp_path):
    """With capture stopped there is no source or detector; the setattr is
    the whole job and must not raise."""
    state = AppState(settings_path=str(tmp_path / "settings.json"), db_path=":memory:")
    with TestClient(build_app(lambda: state)) as client:
        r = client.patch("/api/settings", json={"camera_brightness": 180.0})

    assert r.status_code == 200
    assert state.settings.camera_brightness == 180.0


def test_a_source_without_set_controls_is_tolerated(tmp_path):
    """FakeFrameSource and any future source need not implement it."""

    class _Bare:
        width, height, fps = 64, 48, 30.0
        measured_fps = 29.0

        def open(self):
            return True

        def latest(self):
            return (1, np.zeros((48, 64, 3), dtype=np.uint8))

        def release(self):
            pass

    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        source_factory=lambda s: _Bare(),
        detector_factory=lambda s, d: _Det(),
    )
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        r = client.patch("/api/settings", json={"camera_brightness": 180.0})
        client.post("/api/capture/stop")

    assert r.status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_live_settings.py -v`
Expected: FAIL — the field-set tests fail on membership, the push tests fail because `src.controls` stays empty.

- [ ] **Step 3: Write the implementation**

In `sidecar/app/settings_store.py`, move the five names. `HOT_RELOADABLE_FIELDS` becomes:

```python
# Fields the running Pipeline re-reads from `settings` every frame/track update,
# or that _push_live_settings hands to the open camera and detector, so
# mutating them in place takes effect without stopping capture. Everything
# else is baked into source/detector objects at /api/capture/start time.
HOT_RELOADABLE_FIELDS = {
    "infer_frame_skip",
    "preview_height",
    "preview_max_fps",
    "track_expiry_s",
    # Read per inference call (YoloDetector passes it to track(); the remote
    # detector filters responses against it), so a setter is all it needs.
    "conf_threshold",
    # Plain cap.set() writes against the open handle — see
    # CameraCapture.set_controls, which defers them to the capture thread.
    "camera_brightness",
    "camera_exposure",
    "camera_autofocus",
    "camera_focus",
}
```

Delete those same five names from `RESTART_REQUIRED_FIELDS` (`conf_threshold` at line 87, and the four `camera_*` entries at lines 101–105 along with their comment).

In `sidecar/app/main.py`, add above `_apply_settings_patch`:

```python
# settings key -> CameraCapture.set_controls keyword.
_CAMERA_CONTROL_KEYS = {
    "camera_brightness": "brightness",
    "camera_exposure": "exposure",
    "camera_autofocus": "autofocus",
    "camera_focus": "focus",
}


def _push_live_settings(state: "AppState", patch: dict) -> None:
    """Hand hot-reloadable changes to the objects that already exist.

    Pipeline re-reads infer_frame_skip/preview_*/track_expiry_s from settings
    itself, but the camera and detector hold their own copies, so those two
    need telling. Both lookups go through getattr: with capture stopped there
    is no source or detector at all, and a source need not implement
    set_controls (FakeFrameSource does not).
    """
    controls = {
        kw: patch[key] for key, kw in _CAMERA_CONTROL_KEYS.items() if key in patch
    }
    if controls:
        set_controls = getattr(state.source, "set_controls", None)
        if callable(set_controls):
            set_controls(**controls)
    if "conf_threshold" in patch:
        set_conf = getattr(state.detector, "set_conf", None)
        if callable(set_conf):
            set_conf(patch["conf_threshold"])
```

Change `_apply_settings_patch` to:

```python
def _apply_settings_patch(
    state: "AppState", patch: dict, persist: bool = True
) -> SettingsResponse:
    if state.state == "running":
        locked = set(patch) & RESTART_REQUIRED_FIELDS
        if locked:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot change {sorted(locked)} while capture is running; stop capture first.",
            )
    for key, value in patch.items():
        setattr(state.settings, key, value)
    if "device" in patch:
        state.device = resolve_device(state.settings.device)
    _push_live_settings(state, patch)
    if persist:
        save_settings(state.settings, state.settings_path)
    return _settings_response(state)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_live_settings.py -v && python -m pytest -q`
Expected: PASS. The full suite must stay green — `compute_warnings` emits a warning naming `RESTART_REQUIRED_FIELDS`, so a test asserting on that string may need its expectation updated to the shortened list.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/settings_store.py sidecar/app/main.py sidecar/tests/test_live_settings.py
git commit -m "feat(sidecar): make the five camera-tuning fields hot-reloadable"
```

---

## Task 4: Non-persisting patches and an explicit save

**Files:**
- Modify: `sidecar/app/main.py:406-409`, `sidecar/app/schemas.py:121-124`
- Test: `sidecar/tests/test_live_settings.py`

**Interfaces:**
- Consumes: `_apply_settings_patch(state, patch, persist)` (Task 3).
- Produces: `PATCH /api/settings?persist=false` and `POST /api/settings/save`, both returning `SettingsResponse`. Task 6 wraps them.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_live_settings.py`:

```python
# --- persistence modes ---------------------------------------------------


def _saved(path) -> dict:
    import json

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_a_non_persisting_patch_changes_memory_only(tmp_path):
    """Tuning against a live feed must not write the file on every slider
    tick, and must not make an experiment the startup config."""
    path = tmp_path / "settings.json"
    state = AppState(settings_path=str(path), db_path=":memory:")
    with TestClient(build_app(lambda: state)) as client:
        client.patch("/api/settings", json={"conf_threshold": 0.7})  # persist
        before = _saved(path)
        r = client.patch("/api/settings?persist=false", json={"conf_threshold": 0.9})

    assert r.status_code == 200
    assert state.settings.conf_threshold == 0.9
    assert _saved(path) == before
    assert _saved(path)["conf_threshold"] == 0.7


def test_save_persists_what_is_in_memory(tmp_path):
    path = tmp_path / "settings.json"
    state = AppState(settings_path=str(path), db_path=":memory:")
    with TestClient(build_app(lambda: state)) as client:
        client.patch("/api/settings", json={"conf_threshold": 0.7})
        client.patch("/api/settings?persist=false", json={"conf_threshold": 0.9})
        r = client.post("/api/settings/save")

    assert r.status_code == 200
    assert r.json()["conf_threshold"] == 0.9
    assert _saved(path)["conf_threshold"] == 0.9


def test_a_non_persisting_patch_still_reaches_the_device(running):
    client, _, src, _ = running
    client.patch("/api/settings?persist=false", json={"camera_brightness": 200.0})

    assert src.controls["brightness"] == 200.0


def test_a_non_persisting_patch_still_respects_the_restart_lock(running):
    """persist=false is about the file, not about the lock."""
    client, _, _, _ = running
    r = client.patch("/api/settings?persist=false", json={"capture_width": 640})

    assert r.status_code == 409


def test_reset_fields_sets_a_control_back_to_none(tmp_path):
    """Revert's case: all four controls default to None, so on a fresh
    install the saved baseline IS None and exclude_none would make Revert a
    no-op on the primary path."""
    state = AppState(settings_path=str(tmp_path / "settings.json"), db_path=":memory:")
    with TestClient(build_app(lambda: state)) as client:
        client.patch("/api/settings", json={"camera_brightness": 180.0})
        r = client.patch("/api/settings", json={"reset_fields": ["camera_brightness"]})

    assert r.status_code == 200
    assert state.settings.camera_brightness is None
    assert r.json()["camera_brightness"] is None


def test_reset_fields_rejects_a_field_that_cannot_be_null(tmp_path):
    """Nulling imgsz would break capture; only the device controls are
    optional."""
    state = AppState(settings_path=str(tmp_path / "settings.json"), db_path=":memory:")
    with TestClient(build_app(lambda: state)) as client:
        r = client.patch("/api/settings", json={"reset_fields": ["imgsz"]})

    assert r.status_code == 422


def test_resetting_a_control_stops_writing_it_to_the_device(running):
    """None means 'leave the camera alone' — the device keeps whatever value
    it currently holds until the next reopen."""
    client, _, src, _ = running
    client.patch("/api/settings", json={"reset_fields": ["camera_brightness"]})

    assert src.controls["brightness"] is None


def test_camera_control_bounds_are_enforced(tmp_path):
    state = AppState(settings_path=str(tmp_path / "settings.json"), db_path=":memory:")
    with TestClient(build_app(lambda: state)) as client:
        assert client.patch("/api/settings", json={"camera_brightness": 999}).status_code == 422
        assert client.patch("/api/settings", json={"camera_exposure": 5}).status_code == 422
        assert client.patch("/api/settings", json={"camera_focus": -1}).status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_live_settings.py -k "persist or save or bounds" -v`
Expected: FAIL — `persist=false` is ignored so the file changes, and `POST /api/settings/save` returns 405.

- [ ] **Step 3: Write the implementation**

In `sidecar/app/schemas.py`, replace the four unbounded camera fields in `SettingsUpdateRequest` (lines 121–124) with:

```python
    # Bounds mirror settingsFields.ts's min/max for these controls. They are
    # generous because the meaningful range is device-specific; they exist to
    # reject nonsense, not to encode one camera's scale. Note that calibration
    # applies its recommendation through _apply_settings_patch directly and so
    # is not validated here.
    camera_brightness: float | None = Field(default=None, ge=0.0, le=255.0)
    # Windows exposure is log2 seconds: -6 is 1/64 s, 0 is one full second.
    camera_exposure: float | None = Field(default=None, ge=-13.0, le=0.0)
    camera_autofocus: bool | None = None
    camera_focus: float | None = Field(default=None, ge=0.0, le=1023.0)
```

Add to `SettingsUpdateRequest`, after the camera controls:

```python
    # exclude_none=True means a patch can never send a field back to null, so
    # without this Revert cannot restore "leave the camera alone" — which is
    # the default state of all four controls, and therefore the saved baseline
    # on a fresh install. Restricted to those four because they are the only
    # settings whose type admits None; nulling imgsz would break capture.
    reset_fields: list[str] | None = None

    @field_validator("reset_fields")
    @classmethod
    def _validate_reset_fields(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            unknown = set(v) - RESETTABLE_FIELDS
            if unknown:
                raise ValueError(f"reset_fields must be a subset of {sorted(RESETTABLE_FIELDS)}")
        return v
```

In `sidecar/app/settings_store.py`, beside the other field sets:

```python
# Settings that can be set back to None ("leave the camera alone"). Only the
# device controls: every other field has a non-optional type.
RESETTABLE_FIELDS = {
    "camera_brightness",
    "camera_exposure",
    "camera_autofocus",
    "camera_focus",
}
```

Import it in `schemas.py` alongside `ALLOWED_BACKENDS`.

In `sidecar/app/main.py`, replace the `update_settings` route with:

```python
    @app.patch("/api/settings", response_model=SettingsResponse)
    async def update_settings(body: SettingsUpdateRequest, persist: bool = True):
        """persist=false applies the change without writing settings.json.

        The Live tab's tuning card uses it so a slider drag reaches the camera
        immediately without every intermediate value becoming the config the
        app boots with. POST /api/settings/save commits what is in memory.
        """
        patch = body.model_dump(exclude_none=True)
        # exclude_none drops nulls, so "set this back to null" has to travel
        # as an explicit list of names — see SettingsUpdateRequest.reset_fields.
        for name in patch.pop("reset_fields", []):
            patch[name] = None
        return _apply_settings_patch(state, patch, persist=persist)

    @app.post("/api/settings/save", response_model=SettingsResponse)
    async def save_current_settings():
        """Persist the in-memory settings, including anything applied with
        persist=false. Writes the whole Settings object — see the design
        doc's 'Save is global' tradeoff."""
        save_settings(state.settings, state.settings_path)
        return _settings_response(state)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_live_settings.py -v && python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/main.py sidecar/app/schemas.py sidecar/tests/test_live_settings.py
git commit -m "feat(sidecar): add a non-persisting settings patch and an explicit save"
```

---

## Task 5: Read a stored camera profile back

**Files:**
- Modify: `sidecar/app/camera_caps.py:190-200`, `sidecar/app/main.py`, `sidecar/app/schemas.py`
- Test: `sidecar/tests/test_camera_profiles.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `camera_caps.device_key_for(device_name: str, index: int, width: int, height: int) -> str`, and `GET /api/camera/profile` returning `StoredProfileResponse` with a nullable `profile` field. Task 6 wraps the route; Task 10 reads `profile.controls`.

- [ ] **Step 1: Write the failing tests**

Append to `sidecar/tests/test_camera_profiles.py`:

```python
# --- reading a profile back ----------------------------------------------

from app.camera_caps import CameraProfile, ControlSupport, device_key_for
from app.camera_profiles import save_profile
from app.main import AppState, build_app
from fastapi.testclient import TestClient


def test_device_key_for_matches_what_calibration_writes():
    """One format string, used by both the writer and the reader — a second
    copy would drift and silently orphan every stored profile."""
    assert device_key_for("Logitech StreamCam", 1, 1280, 720) == (
        "Logitech StreamCam:1:1280x720"
    )


def _profile(device_key: str) -> CameraProfile:
    return CameraProfile(
        device_key=device_key,
        backend="MSMF",
        width=1280,
        height=720,
        fps_auto_exposure=29.9,
        fps_capped_exposure=30.8,
        controls=ControlSupport(brightness=True, exposure=True, gain=False, focus=False),
        recommended={"camera_autofocus": False},
        measured_at=1.0,
    )


def test_a_stored_profile_is_returned_for_the_current_camera(tmp_path):
    """Profiles were written and never read back, so a calibration did not
    survive an app restart. This is that round trip."""
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        camera_namer=lambda: ["Logitech StreamCam"],
    )
    state.settings.camera_index = 0
    state.settings.capture_width = 1280
    state.settings.capture_height = 720
    save_profile(_profile("Logitech StreamCam:0:1280x720"), str(tmp_path / "camera_profiles.json"))

    with TestClient(build_app(lambda: state)) as client:
        body = client.get("/api/camera/profile").json()

    assert body["profile"]["controls"]["brightness"] is True
    assert body["profile"]["controls"]["focus"] is False


def test_an_uncalibrated_camera_returns_a_null_profile(tmp_path):
    """A normal state the card renders, not an error it handles — hence 200
    with a null field rather than the 404 its /apply sibling uses."""
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        camera_namer=lambda: ["Some Other Camera"],
    )
    with TestClient(build_app(lambda: state)) as client:
        r = client.get("/api/camera/profile")

    assert r.status_code == 200
    assert r.json()["profile"] is None


def test_a_profile_for_a_different_resolution_does_not_match(tmp_path):
    """Control support is measured at a resolution; the key includes it."""
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        camera_namer=lambda: ["Logitech StreamCam"],
    )
    state.settings.camera_index = 0
    state.settings.capture_width = 640
    state.settings.capture_height = 480
    save_profile(_profile("Logitech StreamCam:0:1280x720"), str(tmp_path / "camera_profiles.json"))

    with TestClient(build_app(lambda: state)) as client:
        assert client.get("/api/camera/profile").json()["profile"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_camera_profiles.py -v`
Expected: FAIL with `ImportError: cannot import name 'device_key_for'`

- [ ] **Step 3: Write the implementation**

In `sidecar/app/camera_caps.py`, add above `calibrate`:

```python
def device_key_for(device_name: str, index: int, width: int, height: int) -> str:
    """The identity a CameraProfile is stored under.

    Includes the resolution because control support is measured at one: a
    device can accept exposure at 720p and ignore it at 1080p. Used by both
    calibrate() and GET /api/camera/profile, so the format lives here once.
    """
    return f"{device_name}:{index}:{width}x{height}"
```

Replace line 195's inline f-string with `device_key=device_key_for(device_name, index, width, height),`.

In `sidecar/app/schemas.py`, add after `CameraProfileResponse`:

```python
class StoredProfileResponse(BaseModel):
    """The saved profile for the camera currently configured, if any.

    `profile` is null for a camera that has never been calibrated — a normal
    state the UI renders, not an error, which is why this is a 200 rather
    than the 404 POST /api/camera/profile/apply returns.
    """
    profile: CameraProfileResponse | None = None
```

In `sidecar/app/main.py`, import `device_key_for` from `app.camera_caps` and `load_profiles` from `app.camera_profiles`, then add a helper beside the calibrate route:

```python
    def _profiles_path() -> str:
        # Sibling of settings_path rather than camera_profiles.py's hardcoded
        # default, so tests pointing settings_path at tmp_path never touch the
        # real data/ directory.
        return os.path.join(
            os.path.dirname(state.settings_path) or ".", "camera_profiles.json"
        )

    @app.get("/api/camera/profile", response_model=StoredProfileResponse)
    async def get_camera_profile():
        """The stored calibration for the camera currently configured.

        This is what tells the tuning card which controls the device honours,
        and it is why a calibration survives an app restart.
        """
        key = device_key_for(
            _resolve_camera_name(state),
            state.settings.camera_index,
            state.settings.capture_width,
            state.settings.capture_height,
        )
        profile = load_profiles(_profiles_path()).get(key)
        if profile is None:
            return StoredProfileResponse(profile=None)
        return StoredProfileResponse(profile=CameraProfileResponse(**asdict(profile)))
```

Replace the inline `profiles_path = os.path.join(...)` line in `calibrate_camera` with `save_profile(profile, _profiles_path())`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_camera_profiles.py -v && python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera_caps.py sidecar/app/schemas.py sidecar/app/main.py sidecar/tests/test_camera_profiles.py
git commit -m "feat(sidecar): read stored camera profiles back so calibration survives restart"
```

---

# Phase 2 — Renderer: plumbing

## Task 6: REST client methods

**Files:**
- Modify: `desktop/src/renderer/src/lib/api.ts`
- Test: `desktop/src/renderer/src/lib/api.test.ts`

**Interfaces:**
- Consumes: the routes from Tasks 4 and 5.
- Produces: on `ApiClient` — `updateSettings(patch: SettingsUpdate, persist?: boolean)`, `saveSettings(): Promise<SettingsResponse>`, `getCameraProfile(): Promise<StoredProfileResponse>`; and the exported type `StoredProfileResponse`. Tasks 8, 9 and 10 use all three.

- [ ] **Step 1: Write the failing tests**

Append to `desktop/src/renderer/src/lib/api.test.ts`:

```ts
describe('live tuning endpoints', () => {
  it('omits the persist flag by default so ordinary saves still write the file', async () => {
    const calls: string[] = []
    vi.stubGlobal('fetch', async (url: string) => {
      calls.push(url)
      return { ok: true, json: async () => ({}) } as Response
    })

    await createApiClient(9000).updateSettings({ conf_threshold: 0.7 })

    expect(calls[0]).toBe('http://127.0.0.1:9000/api/settings')
  })

  it('asks the sidecar not to persist when tuning live', async () => {
    const calls: string[] = []
    vi.stubGlobal('fetch', async (url: string) => {
      calls.push(url)
      return { ok: true, json: async () => ({}) } as Response
    })

    await createApiClient(9000).updateSettings({ conf_threshold: 0.9 }, false)

    expect(calls[0]).toBe('http://127.0.0.1:9000/api/settings?persist=false')
  })

  it('saves what is in memory', async () => {
    const calls: [string, string][] = []
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      calls.push([url, init?.method ?? 'GET'])
      return { ok: true, json: async () => ({}) } as Response
    })

    await createApiClient(9000).saveSettings()

    expect(calls[0]).toEqual(['http://127.0.0.1:9000/api/settings/save', 'POST'])
  })

  it('reads the stored camera profile', async () => {
    vi.stubGlobal('fetch', async () => ({
      ok: true,
      json: async () => ({ profile: null })
    }) as Response)

    await expect(createApiClient(9000).getCameraProfile()).resolves.toEqual({ profile: null })
  })
})
```

If `api.test.ts` does not exist, create it with `import { describe, expect, it, vi } from 'vitest'` and `import { createApiClient } from './api'` at the top, and add an `afterEach(() => vi.unstubAllGlobals())`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/lib/api.test.ts`
Expected: FAIL — `saveSettings is not a function`.

- [ ] **Step 3: Write the implementation**

In `desktop/src/renderer/src/lib/api.ts`, widen `SettingsUpdate`:

```ts
// Mirrors sidecar/app/schemas.py::SettingsUpdateRequest. `reset_fields` names
// settings to set back to null; it exists because the sidecar drops nulls from
// a patch (exclude_none), so "leave the camera alone" cannot travel as a value.
// Only the four camera controls are resettable — see RESETTABLE_FIELDS.
export type SettingsUpdate = Partial<SettingsPayload> & {
  reset_fields?: (keyof SettingsPayload)[]
}
```

Add after `CameraProfileResponse`:

```ts
// Mirrors sidecar/app/schemas.py::StoredProfileResponse. `profile` is null
// for a camera that has never been calibrated — a normal state, not an error.
export interface StoredProfileResponse {
  profile: CameraProfileResponse | null
}
```

In the `ApiClient` interface, replace the `updateSettings` line and add two methods:

```ts
  // persist=false applies the change to the running camera/detector without
  // writing settings.json — the Live tab's tuning card drags sliders through
  // this, then commits once via saveSettings().
  updateSettings(patch: SettingsUpdate, persist?: boolean): Promise<SettingsResponse>
  saveSettings(): Promise<SettingsResponse>
  // The stored calibration for the currently configured camera, or
  // { profile: null } if it has never been calibrated.
  getCameraProfile(): Promise<StoredProfileResponse>
```

In `createApiClient`'s returned object, replace `updateSettings` and add the two:

```ts
    updateSettings: (patch, persist = true) =>
      request<SettingsResponse>(
        `/settings${persist ? '' : '?persist=false'}`,
        'PATCH',
        patch
      ),
    saveSettings: () => request<SettingsResponse>('/settings/save', 'POST'),
    getCameraProfile: () => request<StoredProfileResponse>('/camera/profile', 'GET'),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/lib/api.test.ts && npm run typecheck`
Expected: PASS. Typecheck will flag every test fake implementing `ApiClient` that now lacks the two new methods. There are exactly two, and you must update both: `useSidecarSettings.test.tsx:42` and `AdminPanel.test.tsx:55`. Give each a `saveSettings` returning `baseSettings()` and a `getCameraProfile` returning `{ profile: null }` — a camera with no stored calibration is the right default for existing tests, none of which exercise one.

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/lib/api.ts desktop/src/renderer/src/lib/api.test.ts
git commit -m "feat(desktop): add live-tuning REST client methods"
```

---

## Task 7: Field metadata for the camera controls, and where each group lives

**Files:**
- Modify: `desktop/src/renderer/src/lib/settingsFields.ts`
- Test: `desktop/src/renderer/src/lib/settingsFields.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `FieldMeta` gains `type: 'boolean'` as a valid value; `FieldGroup` gains `home: 'live' | 'admin'`; four new `SETTINGS_FIELDS` entries keyed `camera_brightness`, `camera_exposure`, `camera_autofocus`, `camera_focus`. Tasks 10 and 13 filter on `home`.

- [ ] **Step 1: Write the failing tests**

Append to `desktop/src/renderer/src/lib/settingsFields.test.ts` (create it if absent, importing from `./settingsFields`):

```ts
describe('field placement', () => {
  it('gives every group a home', () => {
    for (const g of SETTINGS_GROUPS) {
      expect(['live', 'admin']).toContain(g.home)
    }
  })

  it('places every field in exactly one group', () => {
    const seen = SETTINGS_GROUPS.flatMap((g) => g.keys)
    expect(new Set(seen).size).toBe(seen.length)
  })

  it('describes every grouped key', () => {
    // A key in a group with no FieldMeta renders as nothing at all.
    for (const key of SETTINGS_GROUPS.flatMap((g) => g.keys)) {
      expect(SETTINGS_FIELDS.find((f) => f.key === key)).toBeDefined()
    }
  })

  it('keeps the fields that need a device reopen in admin', () => {
    const liveKeys = SETTINGS_GROUPS.filter((g) => g.home === 'live').flatMap((g) => g.keys)
    for (const key of ['camera_index', 'capture_width', 'capture_height', 'capture_fps', 'imgsz']) {
      expect(liveKeys).not.toContain(key)
    }
  })

  it('puts the five tunable fields on live', () => {
    const liveKeys = SETTINGS_GROUPS.filter((g) => g.home === 'live').flatMap((g) => g.keys)
    for (const key of [
      'conf_threshold',
      'camera_brightness',
      'camera_exposure',
      'camera_autofocus',
      'camera_focus'
    ]) {
      expect(liveKeys).toContain(key)
    }
  })

  it('warns about the exposure framerate trap in the hint', () => {
    const exposure = SETTINGS_FIELDS.find((f) => f.key === 'camera_exposure')
    expect(exposure?.hint).toMatch(/fps|framerate/i)
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/lib/settingsFields.test.ts`
Expected: FAIL — `home` is undefined on every group, and the four camera fields have no `FieldMeta`.

- [ ] **Step 3: Write the implementation**

In `desktop/src/renderer/src/lib/settingsFields.ts`, widen the type union:

```ts
export interface FieldMeta {
  key: keyof SettingsPayload
  label: string
  hint: string
  type: 'select' | 'number' | 'text' | 'boolean'
  options?: readonly string[]
  min?: number
  max?: number
  step?: number
}
```

Append these four entries to `SETTINGS_FIELDS`:

```ts
  {
    key: 'camera_brightness',
    label: 'Brightness',
    hint: 'Boosts the image after the sensor, so it costs no framerate — but it amplifies noise along with the picture. Reach for exposure first and use this to finish. Bounds mirror the sidecar; the meaningful range is device-specific.',
    type: 'number',
    min: 0,
    max: 255,
    step: 5
  },
  {
    key: 'camera_exposure',
    label: 'Exposure',
    hint: 'How long the shutter stays open, as log2 seconds: -6 is 1/64 s, -4 is 1/16 s, -2 is a quarter of a second. Each step up doubles the light and halves the framerate — -2 caps the camera at 4 fps. Watch the capture fps reading above after every change.',
    type: 'number',
    min: -13,
    max: 0,
    step: 1
  },
  {
    key: 'camera_autofocus',
    label: 'Autofocus',
    hint: "The StreamCam's autofocus hunts for faces, which a checkout counter does not have — it drifts off the item and back. Off, with a fixed focus, is steadier for a camera that never moves.",
    type: 'boolean'
  },
  {
    key: 'camera_focus',
    label: 'Focus',
    hint: 'Fixed focus distance, only meaningful with autofocus off. Lower is farther away. Adjust until the sharpness reading above stops rising.',
    type: 'number',
    min: 0,
    max: 1023,
    step: 5
  }
```

Replace `FieldGroup` and `SETTINGS_GROUPS` with:

```ts
export interface FieldGroup {
  label: string
  // Which view renders this group. One list, filtered by both views, so a
  // field cannot end up in both places or neither.
  home: 'live' | 'admin'
  keys: (keyof SettingsPayload)[]
}

export const SETTINGS_GROUPS: FieldGroup[] = [
  { label: 'Model & Device', home: 'admin', keys: ['active_model', 'device'] },
  {
    label: 'Roboflow API backends',
    home: 'admin',
    keys: [
      'local_api_url',
      'cloud_api_url',
      'roboflow_workspace',
      'roboflow_workflow_id',
      'remote_infer_size',
      'remote_timeout_s',
      'remote_max_retries'
    ]
  },
  {
    // Everything here needs the device reopened, which costs ~30 s on a
    // StreamCam — so it stays in Admin rather than on the tuning card.
    label: 'Camera & Capture',
    home: 'admin',
    keys: ['camera_index', 'capture_width', 'capture_height', 'capture_fps']
  },
  {
    label: 'Detection & Tracking',
    home: 'admin',
    keys: ['imgsz', 'resize_mode']
  },
  {
    label: 'Image',
    home: 'live',
    keys: ['camera_brightness', 'camera_exposure', 'camera_autofocus', 'camera_focus']
  },
  { label: 'Detection', home: 'live', keys: ['conf_threshold'] },
  {
    label: 'Stream',
    home: 'live',
    keys: ['infer_frame_skip', 'preview_height', 'preview_max_fps', 'track_expiry_s']
  }
]
```

**Keep the suite green in the same commit.** `AdminPanel` renders `SETTINGS_GROUPS` filtered only by the Roboflow condition, so the moment you add the three `home: 'live'` groups it would start rendering them — including `camera_autofocus`, whose new `'boolean'` type has no branch in `AdminPanel`'s `renderField` and would fall through to the number input. Task 13 adds the real filter, but one line of it has to land here. In `desktop/src/renderer/src/views/AdminPanel.tsx`, change:

```tsx
  const visibleGroups = SETTINGS_GROUPS.filter(
    (g) => g.label !== 'Roboflow API backends' || backendIsRemote
  )
```

to:

```tsx
  const visibleGroups = SETTINGS_GROUPS.filter(
    (g) => g.home === 'admin' && (g.label !== 'Roboflow API backends' || backendIsRemote)
  )
```

That is the whole Admin change for this task — leave the calibration card and quality readout alone; Task 13 removes those.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/lib/settingsFields.test.ts && npm test && npm run typecheck`
Expected: PASS. The full suite matters here: this task moves five fields out of Admin's groups, so any Admin test asserting on one of them by id will now fail. Those five (`conf_threshold`, `infer_frame_skip`, `preview_height`, `preview_max_fps`, `track_expiry_s`) move to the Live tab in Task 13 and their coverage moves with them — delete or retarget the failing assertions, and name every one you touched in your report.

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/lib/settingsFields.ts desktop/src/renderer/src/lib/settingsFields.test.ts
git commit -m "feat(desktop): describe the camera controls and give each group a home"
```

---

## Task 8: Hook gains an optional health poll and the stored profile

**Files:**
- Modify: `desktop/src/renderer/src/hooks/useSidecarSettings.ts`
- Test: `desktop/src/renderer/src/hooks/useSidecarSettings.test.tsx`

**Interfaces:**
- Consumes: `getCameraProfile()` (Task 6).
- Produces: `SettingsDeps` gains `pollHealth?: boolean` (default `true`); `SidecarSettings` gains `storedProfile: CameraProfileResponse | null`. Tasks 10 and 13 use both.

- [ ] **Step 1: Write the failing tests**

Append to `desktop/src/renderer/src/hooks/useSidecarSettings.test.tsx`:

```tsx
describe('hosting inside LiveView', () => {
  it('does not poll health when the host already owns capture state', async () => {
    // LiveView drives capture through useSidecarStream. A second poller here
    // would give the same view two answers to "are we running".
    let healthCalls = 0
    const api = fakeApi({ health: async () => { healthCalls++; return { state: 'idle', active_model: 'm', device: 'cpu' } } })

    renderHook(() => useSidecarSettings(9000, { apiFactory: () => api, pollHealth: false }))
    await act(async () => { await new Promise((r) => setTimeout(r, 50)) })

    expect(healthCalls).toBe(0)
  })

  it('keeps reading camera quality even with health polling off', async () => {
    // The two share one timer. The tuning card turns health off because
    // LiveView owns capture state, but its whole readout is quality.
    let qualityCalls = 0
    const api = fakeApi({
      getCameraQuality: async () => {
        qualityCalls++
        return { available: true, brightness: 128, contrast: 40, sharpness: 90, capture_fps: 29, target_fps: 30, verdicts: {}, detail: '' }
      }
    })

    const { result } = renderHook(() =>
      useSidecarSettings(9000, { apiFactory: () => api, pollHealth: false })
    )
    await waitFor(() => expect(result.current.cameraQuality).not.toBeNull())

    expect(qualityCalls).toBeGreaterThan(0)
  })

  it('does not enumerate cameras for a consumer that never reads them', async () => {
    // Enumerating opens every device (~30 s). The card has no camera list.
    let cameraCalls = 0
    const api = fakeApi({
      getCameras: async () => {
        cameraCalls++
        return { cameras: [], probed: true, detail: '' }
      }
    })

    const { result } = renderHook(() =>
      useSidecarSettings(9000, { apiFactory: () => api, pollCameras: false })
    )
    await waitFor(() => expect(result.current.settings).not.toBeNull())

    expect(cameraCalls).toBe(0)
    // Nothing else clears the initial true, and a stuck spinner reads as a
    // scan that never finishes.
    expect(result.current.camerasLoading).toBe(false)
  })

  it('still polls health by default, for the Admin panel', async () => {
    let healthCalls = 0
    const api = fakeApi({ health: async () => { healthCalls++; return { state: 'idle', active_model: 'm', device: 'cpu' } } })

    renderHook(() => useSidecarSettings(9000, { apiFactory: () => api }))
    await act(async () => { await new Promise((r) => setTimeout(r, 50)) })

    expect(healthCalls).toBeGreaterThan(0)
  })
})

describe('stored profile', () => {
  it('loads the saved calibration on mount', async () => {
    const profile = {
      device_key: 'StreamCam:0:1280x720',
      backend: 'MSMF',
      width: 1280,
      height: 720,
      fps_auto_exposure: 29.9,
      fps_capped_exposure: 30.8,
      controls: { brightness: true, exposure: true, gain: false, focus: false },
      recommended: {},
      measured_at: 1
    }
    const api = fakeApi({ getCameraProfile: async () => ({ profile }) })

    const { result } = renderHook(() => useSidecarSettings(9000, { apiFactory: () => api }))
    await waitFor(() => expect(result.current.storedProfile).not.toBeNull())

    expect(result.current.storedProfile?.controls.focus).toBe(false)
  })

  it('leaves it null for an uncalibrated camera', async () => {
    const api = fakeApi({ getCameraProfile: async () => ({ profile: null }) })

    const { result } = renderHook(() => useSidecarSettings(9000, { apiFactory: () => api }))
    await waitFor(() => expect(result.current.settings).not.toBeNull())

    expect(result.current.storedProfile).toBeNull()
  })
})
```

**Helper names.** The snippets above are written against a `fakeApi(...)` helper that does not exist. The real ones already live at the top of `useSidecarSettings.test.tsx`: `baseSettings(overrides: Partial<SettingsResponse>): SettingsResponse` and `makeDeps(overrides: Partial<ApiClient>): { deps: SettingsDeps; api: ApiClient }`, where `deps` already carries `apiFactory`. Adapt every snippet to those — `useSidecarSettings(9000, { apiFactory: () => api, pollHealth: false })` becomes:

```ts
    const { deps, api } = makeDeps({ health: vi.fn(async () => { healthCalls++; return { state: 'idle', active_model: 'm', device: 'cpu' } }) })
    renderHook(() => useSidecarSettings(9000, { ...deps, pollHealth: false }))
```

**Extract them first.** Task 10 creates a new test file that needs both helpers, so move `baseSettings` and `makeDeps` verbatim into a new `desktop/src/renderer/src/test/fakes.ts`, export both, and import them back into `useSidecarSettings.test.tsx`. Do this as the first commit of this task, with the existing hook tests still passing, before writing any new test. Do not rename them.

**There is a second copy.** `AdminPanel.test.tsx:55` builds its own full `ApiClient` fake with the same shape, differing only in that its returned deps carry `healthPollMs: 10_000`. Switch that file to import the shared helpers too and delete its local duplicate, passing the `healthPollMs` override at the call sites instead. Two full fakes means every future contract change edits both — this branch alone already forced that once, when Task 6 added two methods to the interface. Keep this in the same first commit, and confirm `AdminPanel.test.tsx` still passes before moving on.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/hooks/useSidecarSettings.test.tsx`
Expected: FAIL — health is polled regardless, and `storedProfile` is undefined.

- [ ] **Step 3: Write the implementation**

In `useSidecarSettings.ts`, add to `SettingsDeps`:

```ts
  // False when a host view already owns capture state (LiveView drives it
  // through useSidecarStream). Two pollers in one view means two answers to
  // "are we running", and they disagree during a start or stop. It suppresses
  // ONLY the health call — the camera-quality read shares that timer and the
  // tuning card depends on it.
  pollHealth?: boolean
  // False for a consumer that never reads `cameras`. Enumerating opens every
  // device (~30 s), so the tuning card must not trigger it on mount.
  pollCameras?: boolean
```

Add to `SidecarSettings`:

```ts
  // The saved calibration for the configured camera, or null if it has never
  // been calibrated. Says which controls the device actually honours.
  storedProfile: CameraProfileResponse | null
```

Inside the hook, after the other `deps` reads:

```ts
  const shouldPollHealth = deps.pollHealth ?? true
  const shouldPollCameras = deps.pollCameras ?? true
```

Add the state:

```ts
  const [storedProfile, setStoredProfile] = useState<CameraProfileResponse | null>(null)
```

Inside `load()`, extend the `Promise.all` and its destructuring:

```ts
      const [s, sys, p, prof] = await Promise.all([
        api.getSettings(),
        api.getSystemInfo(),
        api.getPresets(),
        // Cheap: reads one small JSON file, opens no device.
        api.getCameraProfile()
      ])
      setSettings(s)
      setSystemInfo(sys)
      setPresets(p.presets)
      setRecommended(p.recommended)
      setStoredProfile(prof.profile)
```

**The guard goes inside `pollHealth()`, not around it.** That function does two things on one timer: `api.health()` and `api.getCameraQuality()`. The tuning card passes `pollHealth: false` because `LiveView` owns capture state — but it still needs the quality readout, so suppressing the whole function would blank it. Change only the health half:

```ts
    const pollHealth = async (): Promise<void> => {
      // Skipped when a host view already owns capture state. The quality read
      // below is NOT skipped — it shares this timer and the tuning card needs
      // it regardless of who owns capture state.
      if (shouldPollHealth) {
        try {
          const h = await api.health()
          if (!cancelled) setCaptureState(h.state)
        } catch {
          // Sidecar not reachable yet; keep the last known capture state.
        }
      }
      try {
        const q = await api.getCameraQuality()
        if (!cancelled) setCameraQuality(q)
      } catch {
        // Sidecar not reachable yet; keep the last known reading.
      }
    }
```

The timer itself stays unconditional.

Guard the camera scan, which a consumer that never reads `cameras` should not trigger — it opens every device. Replace the `cameraTimer` line and the `cameraTicker` interval with:

```ts
    const cameraTimer = shouldPollCameras ? setTimeout(() => void refreshCameras(), 0) : null
```

```ts
    const cameraTicker = shouldPollCameras
      ? setInterval(() => {
          if (!cancelled) void refreshCameras()
        }, cameraPollMs)
      : null
```

and in the cleanup:

```ts
      if (cameraTimer !== null) clearTimeout(cameraTimer)
      if (cameraTicker !== null) clearInterval(cameraTicker)
```

When `shouldPollCameras` is false, set `camerasLoading` to false at the same point rather than leaving it stuck true — `useState(true)` is its initial value and nothing else would clear it:

```ts
    if (!shouldPollCameras) setCamerasLoading(false)
```

Place that inside the effect body, before the timers.

Add `shouldPollHealth` and `shouldPollCameras` to the effect's dependency array. Return `storedProfile` from the hook.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/hooks/useSidecarSettings.test.tsx && npm run typecheck`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/hooks/useSidecarSettings.ts desktop/src/renderer/src/hooks/useSidecarSettings.test.tsx
git commit -m "feat(desktop): make health polling optional and load the stored profile"
```

---

## Task 9: Live apply, save, and revert against a stable baseline

**Files:**
- Modify: `desktop/src/renderer/src/hooks/useSidecarSettings.ts`
- Test: `desktop/src/renderer/src/hooks/useSidecarSettings.test.tsx`

**Interfaces:**
- Consumes: `updateSettings(patch, persist)` and `saveSettings()` (Task 6).
- Produces: on `SidecarSettings` — `liveUpdate(patch: SettingsUpdate): Promise<void>`, `save(): Promise<SettingsResponse>`, `savedSettings: SettingsResponse | null`. Tasks 10 and 11 use all three.

- [ ] **Step 1: Write the failing tests**

Append to `desktop/src/renderer/src/hooks/useSidecarSettings.test.tsx`:

```tsx
describe('live apply and the saved baseline', () => {
  it('applies without persisting', async () => {
    const calls: [unknown, boolean | undefined][] = []
    const api = fakeApi({
      updateSettings: async (patch, persist) => {
        calls.push([patch, persist])
        return { ...BASE_SETTINGS, ...patch }
      }
    })

    const { result } = renderHook(() => useSidecarSettings(9000, { apiFactory: () => api }))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => { await result.current.liveUpdate({ conf_threshold: 0.9 }) })

    expect(calls).toEqual([[{ conf_threshold: 0.9 }, false]])
  })

  it('leaves the saved baseline untouched while tuning', async () => {
    // The regression this exists to catch: a live PATCH returns a fresh
    // settings object, and treating that as "saved" makes every slider tick
    // look committed, so Revert has nothing to go back to.
    const api = fakeApi({
      updateSettings: async (patch) => ({ ...BASE_SETTINGS, ...patch })
    })

    const { result } = renderHook(() => useSidecarSettings(9000, { apiFactory: () => api }))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => { await result.current.liveUpdate({ conf_threshold: 0.9 }) })

    expect(result.current.settings?.conf_threshold).toBe(0.9)
    expect(result.current.savedSettings?.conf_threshold).toBe(BASE_SETTINGS.conf_threshold)
  })

  it('moves the baseline on save', async () => {
    const api = fakeApi({
      updateSettings: async (patch) => ({ ...BASE_SETTINGS, ...patch }),
      saveSettings: async () => ({ ...BASE_SETTINGS, conf_threshold: 0.9 })
    })

    const { result } = renderHook(() => useSidecarSettings(9000, { apiFactory: () => api }))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => { await result.current.liveUpdate({ conf_threshold: 0.9 }) })
    await act(async () => { await result.current.save() })

    expect(result.current.savedSettings?.conf_threshold).toBe(0.9)
  })

  it('moves the baseline on an ordinary persisting update too', async () => {
    const api = fakeApi({
      updateSettings: async (patch) => ({ ...BASE_SETTINGS, ...patch })
    })

    const { result } = renderHook(() => useSidecarSettings(9000, { apiFactory: () => api }))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => { await result.current.update({ imgsz: 960 }) })

    expect(result.current.savedSettings?.imgsz).toBe(960)
  })
})
```

Define `BASE_SETTINGS` near the top of the file as a complete `SettingsResponse` literal if the file does not already have one, with `conf_threshold: 0.5` and `imgsz: 640`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/hooks/useSidecarSettings.test.tsx -t "live apply"`
Expected: FAIL — `liveUpdate is not a function`.

- [ ] **Step 3: Write the implementation**

In `useSidecarSettings.ts`, add to `SidecarSettings`:

```ts
  // Applies to the running camera/detector without writing settings.json.
  liveUpdate: (patch: SettingsUpdate) => Promise<void>
  // Commits whatever is in memory, including live-applied values.
  save: () => Promise<SettingsResponse>
  // The last persisted settings. Distinct from `settings`, which tracks the
  // live values — the difference between the two is "unsaved changes".
  savedSettings: SettingsResponse | null
```

Add the state:

```ts
  const [savedSettings, setSavedSettings] = useState<SettingsResponse | null>(null)
```

In `load()`, after `setSettings(s)`, add `setSavedSettings(s)`.

In `update()`, after `setSettings(r)`, add `setSavedSettings(r)` — an ordinary update persists, so it moves the baseline. Do the same in `applyPreset()`.

Add the two new callbacks:

```ts
  // Deliberately does not touch savedSettings: this is an uncommitted
  // experiment, and the gap between `settings` and `savedSettings` is exactly
  // what the tuning card reports as unsaved changes.
  const liveUpdate = useCallback(async (patch: SettingsUpdate): Promise<void> => {
    setError(null)
    try {
      setSettings(await apiRef.current!.updateSettings(patch, false))
    } catch (e) {
      setError(errorMessage(e))
    }
  }, [])

  const save = useCallback(async (): Promise<SettingsResponse> => {
    setSaving(true)
    setError(null)
    try {
      const r = await apiRef.current!.saveSettings()
      setSettings(r)
      setSavedSettings(r)
      return r
    } catch (e) {
      setError(errorMessage(e))
      throw e
    } finally {
      setSaving(false)
    }
  }, [])
```

Return `liveUpdate`, `save` and `savedSettings`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/hooks/useSidecarSettings.test.tsx && npm run typecheck`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/hooks/useSidecarSettings.ts desktop/src/renderer/src/hooks/useSidecarSettings.test.tsx
git commit -m "feat(desktop): add live apply, save, and a stable saved baseline"
```

---

# Phase 3 — Renderer: the tuning card

## Task 10: The card — quality readout and support-gated controls

**Files:**
- Create: `desktop/src/renderer/src/components/CameraTuning.tsx`, `desktop/src/renderer/src/components/CameraTuning.css`, `desktop/src/renderer/src/components/CameraTuning.test.tsx`

**Interfaces:**
- Consumes: `SETTINGS_GROUPS` with `home` (Task 7); `storedProfile` (Task 8); `liveUpdate`/`save`/`savedSettings` (Task 9).
- Produces: the component and its props:

```tsx
export interface CameraTuningProps {
  port: number
  running: boolean
  start: () => Promise<void>
  stop: () => Promise<void>
  cameraName: string
  deps?: SettingsDeps
  // Trailing-edge debounce for slider writes; overridden to 0 in tests.
  debounceMs?: number
}
```

Task 13 mounts it with exactly these props.

- [ ] **Step 1: Write the failing tests**

Create `desktop/src/renderer/src/components/CameraTuning.test.tsx`:

```tsx
import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CameraTuning } from './CameraTuning'

// Reuse the shape the hook tests already establish.
const PROFILE = {
  device_key: 'StreamCam:0:1280x720',
  backend: 'MSMF',
  width: 1280,
  height: 720,
  fps_auto_exposure: 29.9,
  fps_capped_exposure: 30.8,
  controls: { brightness: true, exposure: true, gain: false, focus: false },
  recommended: {},
  measured_at: 1
}

function renderCard(overrides = {}) {
  const api = fakeApi({
    getCameraProfile: async () => ({ profile: PROFILE }),
    getCameraQuality: async () => ({
      available: true,
      brightness: 128,
      contrast: 40,
      sharpness: 90,
      capture_fps: 29.4,
      target_fps: 30,
      verdicts: { brightness: 'ok', sharpness: 'ok', capture_fps: 'ok' },
      detail: ''
    }),
    ...overrides
  })
  return render(
    <CameraTuning
      port={9000}
      running={true}
      start={async () => {}}
      stop={async () => {}}
      cameraName="Logitech StreamCam"
      debounceMs={0}
      deps={{ apiFactory: () => api, pollHealth: false }}
    />
  )
}

describe('CameraTuning', () => {
  it('names the camera it is tuning', async () => {
    renderCard()
    // The camera is chosen in Admin, so the card has to say which one this is.
    expect(await screen.findByText(/Logitech StreamCam/)).toBeInTheDocument()
  })

  it('disables a control the device does not honour', async () => {
    renderCard()
    // Calibration measured focus support as false; a slider that does
    // nothing is worse than one that is visibly unavailable.
    await waitFor(() => expect(screen.getByLabelText('Focus')).toBeDisabled())
  })

  it('enables a control the device does honour', async () => {
    renderCard()
    await waitFor(() => expect(screen.getByLabelText('Brightness')).toBeEnabled())
  })

  it('disables focus while autofocus is on regardless of support', async () => {
    renderCard({
      getCameraProfile: async () => ({
        profile: { ...PROFILE, controls: { ...PROFILE.controls, focus: true } }
      }),
      getSettings: async () => ({ ...BASE_SETTINGS, camera_autofocus: true })
    })
    await waitFor(() => expect(screen.getByLabelText('Focus')).toBeDisabled())
  })

  it('shows the live quality readout', async () => {
    renderCard()
    expect(await screen.findByTestId('tuning-quality')).toHaveTextContent('29.4')
  })

  it('flags a failing metric by more than colour', async () => {
    renderCard({
      getCameraQuality: async () => ({
        available: true,
        brightness: 23,
        contrast: 10,
        sharpness: 12,
        capture_fps: 4.1,
        target_fps: 30,
        verdicts: { brightness: 'low', sharpness: 'low', capture_fps: 'low' },
        detail: ''
      })
    })
    await waitFor(() => expect(screen.getAllByTestId('quality-low').length).toBeGreaterThan(0))
  })

  it('leaves every control enabled for an uncalibrated camera', async () => {
    // No profile means no evidence either way; unsupported controls simply
    // do nothing, which the quality readout makes visible.
    renderCard({ getCameraProfile: async () => ({ profile: null }) })
    await waitFor(() => expect(screen.getByLabelText('Focus')).toBeEnabled())
  })
})
```

Import or re-declare `fakeApi` and `BASE_SETTINGS`; if they live in the hook test file, extract them to `desktop/src/renderer/src/test/fakes.ts` first and import from both places.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx`
Expected: FAIL — the module does not exist.

- [ ] **Step 3: Write the implementation**

Create `desktop/src/renderer/src/components/CameraTuning.tsx`:

```tsx
import { useState, type JSX } from 'react'
import { useSidecarSettings, type SettingsDeps } from '../hooks/useSidecarSettings'
import type { SettingsPayload } from '../lib/api'
import { SETTINGS_FIELDS, SETTINGS_GROUPS, type FieldMeta } from '../lib/settingsFields'
import { Spinner } from './Spinner'
import './CameraTuning.css'

export interface CameraTuningProps {
  port: number
  running: boolean
  start: () => Promise<void>
  stop: () => Promise<void>
  cameraName: string
  deps?: SettingsDeps
  debounceMs?: number
}

// Which measured control support gates which settings field. `gain` is
// measured by calibration but no setting exposes it, so it is absent here.
const SUPPORT_KEY: Partial<Record<keyof SettingsPayload, 'brightness' | 'exposure' | 'focus'>> = {
  camera_brightness: 'brightness',
  camera_exposure: 'exposure',
  camera_focus: 'focus'
}

const LIVE_GROUPS = SETTINGS_GROUPS.filter((g) => g.home === 'live')

export function CameraTuning({
  port,
  running,
  cameraName,
  deps
}: CameraTuningProps): JSX.Element {
  // LiveView owns capture state and passes it as `running`, so this instance
  // must not poll health too. It has no use for the camera list either, and
  // enumerating opens every device. Quality still reads — see useSidecarSettings.
  const { settings, storedProfile, cameraQuality, loading } = useSidecarSettings(port, {
    ...deps,
    pollHealth: deps?.pollHealth ?? false,
    pollCameras: deps?.pollCameras ?? false
  })
  const [open, setOpen] = useState(false)

  const valueOf = (key: keyof SettingsPayload): string | number =>
    (settings?.[key] ?? '') as string | number

  // A control is off-limits when calibration measured the device ignoring it.
  // With no profile there is no evidence either way, so everything stays
  // enabled — an unsupported control simply does nothing.
  const unsupported = (key: keyof SettingsPayload): boolean => {
    const support = SUPPORT_KEY[key]
    if (!support || !storedProfile) return false
    return storedProfile.controls[support] === false
  }

  const renderField = (field: FieldMeta): JSX.Element => {
    // Writing a focus value while autofocus is on is meaningless: the device
    // immediately hunts away from it.
    const autofocusOn = settings?.camera_autofocus === true
    const disabled =
      unsupported(field.key) || (field.key === 'camera_focus' && autofocusOn)

    return (
      <div className="tuning-field" key={field.key}>
        <label htmlFor={`tune-${field.key}`}>{field.label}</label>
        {field.type === 'boolean' ? (
          <input
            id={`tune-${field.key}`}
            type="checkbox"
            checked={valueOf(field.key) === true}
            disabled={disabled}
            readOnly
          />
        ) : (
          <input
            id={`tune-${field.key}`}
            type="range"
            value={Number(valueOf(field.key)) || 0}
            min={field.min}
            max={field.max}
            step={field.step}
            disabled={disabled}
            readOnly
          />
        )}
        {unsupported(field.key) && (
          <p className="field-hint" data-testid="unsupported-control">
            This camera ignored it during calibration.
          </p>
        )}
        <p className="field-hint">{field.hint}</p>
      </div>
    )
  }

  return (
    <div className="card tuning-card" data-testid="camera-tuning">
      <h4>
        <button
          type="button"
          className="tuning-toggle"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? '▾' : '▸'} Camera tuning
        </button>
        <span className="tuning-camera">{cameraName}</span>
      </h4>

      {loading && !settings && (
        <p className="field-hint">
          <Spinner /> Loading…
        </p>
      )}

      {cameraQuality?.available && (
        <div className="quality-row" data-testid="tuning-quality">
          {[
            ['Brightness', cameraQuality.brightness, cameraQuality.verdicts.brightness],
            ['Sharpness', cameraQuality.sharpness, cameraQuality.verdicts.sharpness],
            ['Capture fps', cameraQuality.capture_fps, cameraQuality.verdicts.capture_fps]
          ].map(([label, value, verdict]) => {
            const failing = verdict !== 'ok'
            return (
              <div key={String(label)} className="quality-metric">
                <span className="quality-label">{label}</span>
                <span
                  className={failing ? 'quality-value bad' : 'quality-value'}
                  data-testid={failing ? 'quality-low' : 'quality-ok'}
                >
                  {/* Colour alone fails a colourblind operator reading a
                      readout whose whole job is flagging a bad number. */}
                  {failing && (
                    <span className="quality-flag" aria-hidden="true">
                      ▲{' '}
                    </span>
                  )}
                  {value}
                  {failing && <span className="sr-only"> — outside the expected range</span>}
                </span>
              </div>
            )
          })}
        </div>
      )}

      {settings &&
        LIVE_GROUPS.map((group) => (
          <section className="tuning-group" key={group.label} hidden={!open}>
            <h5>{group.label}</h5>
            {group.keys.map((key) => {
              const field = SETTINGS_FIELDS.find((f) => f.key === key)
              return field ? renderField(field) : null
            })}
          </section>
        ))}

      {!running && (
        <p className="field-hint" data-testid="tuning-idle">
          Start capture to see changes take effect.
        </p>
      )}
    </div>
  )
}
```

Create `desktop/src/renderer/src/components/CameraTuning.css` with the card's own rules: `.tuning-card`, `.tuning-toggle` (a borderless button inheriting the heading font), `.tuning-camera` (muted, smaller), `.tuning-group`, `.tuning-field`, `.tuning-actions`, `.tuning-profile`.

**Move the shared rules rather than copying them.** The quality readout's styles currently live in `AdminPanel.css` (lines 394–435): `.quality-row`, `.quality-metric`, `.quality-label`, `.quality-value`, `.quality-value.bad`, `.quality-flag`, and `.sr-only`. Task 13 deletes Admin's only use of the quality ones, so copying would leave the originals orphaned in the wrong file and two copies to drift. Cut all seven out of `AdminPanel.css` and paste them into `desktop/src/renderer/src/assets/theme.css`, which already exists for exactly this purpose (it owns `.card`) and is imported once in `main.tsx`. Keep their comments — the `.quality-flag` and `.sr-only` ones explain accessibility decisions that are not obvious from the rules.

Leave `.field-hint`, `.btn-primary`, `.btn-outline` and `.btn-small` where they are. They are also defined only in `AdminPanel.css` and `LiveView.tsx` already depends on that file being loaded — a pre-existing coupling this branch does not worsen and should not expand scope to fix.

**Note:** controls are `readOnly` in this task. Task 11 makes them write.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx && npm run typecheck`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/components/CameraTuning.tsx desktop/src/renderer/src/components/CameraTuning.css desktop/src/renderer/src/components/CameraTuning.test.tsx
git commit -m "feat(desktop): add the camera tuning card with support-gated controls"
```

---

## Task 11: Debounced live writes, Save and Revert

**Files:**
- Modify: `desktop/src/renderer/src/components/CameraTuning.tsx`
- Test: `desktop/src/renderer/src/components/CameraTuning.test.tsx`

**Interfaces:**
- Consumes: `liveUpdate`, `save`, `savedSettings` (Task 9).
- Produces: no new exports.

- [ ] **Step 1: Write the failing tests**

Append to `CameraTuning.test.tsx`:

```tsx
describe('applying and committing', () => {
  it('coalesces a drag into one request', async () => {
    const calls: unknown[] = []
    const { container } = renderCard({
      updateSettings: async (patch: unknown) => {
        calls.push(patch)
        return { ...BASE_SETTINGS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))

    const slider = screen.getByLabelText('Brightness')
    fireEvent.change(slider, { target: { value: '100' } })
    fireEvent.change(slider, { target: { value: '150' } })
    fireEvent.change(slider, { target: { value: '180' } })

    // debounceMs is 0 in tests, but the trailing edge still collapses the
    // burst to one write — a real drag emits dozens.
    await waitFor(() => expect(calls.length).toBe(1))
    expect(calls[0]).toEqual({ camera_brightness: 180 })
  })

  it('does not persist while tuning', async () => {
    const persists: (boolean | undefined)[] = []
    renderCard({
      updateSettings: async (patch: unknown, persist?: boolean) => {
        persists.push(persist)
        return { ...BASE_SETTINGS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })

    await waitFor(() => expect(persists).toEqual([false]))
  })

  it('reports unsaved changes and clears them on save', async () => {
    let saved = false
    renderCard({
      updateSettings: async (patch: unknown) => ({ ...BASE_SETTINGS, ...(patch as object) }),
      saveSettings: async () => {
        saved = true
        return { ...BASE_SETTINGS, camera_brightness: 180 }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })

    await screen.findByTestId('tuning-dirty')
    await userEvent.click(screen.getByTestId('tuning-save'))

    expect(saved).toBe(true)
    await waitFor(() => expect(screen.queryByTestId('tuning-dirty')).toBeNull())
  })

  it('reverts to the last saved values', async () => {
    const calls: unknown[] = []
    renderCard({
      updateSettings: async (patch: unknown) => {
        calls.push(patch)
        return { ...BASE_SETTINGS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })
    await screen.findByTestId('tuning-dirty')

    await userEvent.click(screen.getByTestId('tuning-revert'))

    await waitFor(() =>
      expect(calls.at(-1)).toEqual({ camera_brightness: BASE_SETTINGS.camera_brightness })
    )
  })

  it('reverts a control back to "leave the camera alone"', async () => {
    // All four controls default to null, so this is the fresh-install path:
    // tune brightness for the first time, then Revert.
    const calls: unknown[] = []
    renderCard({
      getSettings: async () => ({ ...BASE_SETTINGS, camera_brightness: null }),
      updateSettings: async (patch: unknown) => {
        calls.push(patch)
        return { ...BASE_SETTINGS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })
    await screen.findByTestId('tuning-dirty')

    await userEvent.click(screen.getByTestId('tuning-revert'))

    await waitFor(() =>
      expect(calls.at(-1)).toEqual({ reset_fields: ['camera_brightness'] })
    )
  })

  it('offers nothing to save when nothing changed', async () => {
    renderCard()
    await screen.findByLabelText('Brightness')
    expect(screen.queryByTestId('tuning-dirty')).toBeNull()
  })
})
```

Add `fireEvent` and `userEvent` imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx -t "applying and committing"`
Expected: FAIL — the inputs are `readOnly`, so no request is made.

- [ ] **Step 3: Write the implementation**

In `CameraTuning.tsx`, pull `liveUpdate`, `save`, `savedSettings` and `saving` from the hook. Add a debounce ref and the dirty computation:

```tsx
  const debounceMs = props.debounceMs ?? 150
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  // A drag emits dozens of change events; each would otherwise be a full
  // PATCH. Trailing edge, keyed per field so two sliders do not cancel each
  // other.
  const applyDebounced = (key: keyof SettingsPayload, value: number | boolean): void => {
    const existing = timers.current.get(key)
    if (existing) clearTimeout(existing)
    timers.current.set(
      key,
      setTimeout(() => {
        timers.current.delete(key)
        void liveUpdate({ [key]: value } as SettingsUpdate)
      }, debounceMs)
    )
  }

  useEffect(
    () => () => {
      for (const t of timers.current.values()) clearTimeout(t)
    },
    []
  )

  // Unsaved changes are the gap between the live settings and the last
  // persisted ones — never a local draft, because a live PATCH returns a
  // fresh settings object that would otherwise look committed.
  const dirtyKeys =
    settings && savedSettings
      ? LIVE_GROUPS.flatMap((g) => g.keys).filter((k) => settings[k] !== savedSettings[k])
      : []
```

Replace `readOnly` on both inputs with real handlers:

```tsx
            onChange={(e) => applyDebounced(field.key, e.target.checked)}
```

for the checkbox, and for the range:

```tsx
            onChange={(e) => {
              const n = e.target.valueAsNumber
              if (!Number.isNaN(n)) applyDebounced(field.key, n)
            }}
```

Add the actions block before the `!running` hint:

```tsx
      {dirtyKeys.length > 0 && (
        <div className="tuning-actions">
          <span data-testid="tuning-dirty">
            {dirtyKeys.length} unsaved change{dirtyKeys.length === 1 ? '' : 's'}
          </span>
          <button
            className="btn-primary btn-small"
            disabled={saving}
            data-testid="tuning-save"
            onClick={() => void save()}
          >
            {saving ? <Spinner /> : null} Save
          </button>
          <button
            className="btn-outline btn-small"
            data-testid="tuning-revert"
            onClick={() => {
              if (!savedSettings) return
              // A saved value of null means "leave the camera alone", which is
              // the default for all four controls — so on a fresh install this
              // is the common case, not an edge one. exclude_none on the
              // sidecar drops nulls from a patch, so they travel by name in
              // reset_fields instead.
              const restore = dirtyKeys.filter((k) => savedSettings[k] !== null)
              const reset = dirtyKeys.filter((k) => savedSettings[k] === null)
              const patch: SettingsUpdate = Object.fromEntries(
                restore.map((k) => [k, savedSettings[k]])
              ) as SettingsUpdate
              if (reset.length > 0) patch.reset_fields = reset
              void liveUpdate(patch)
            }}
          >
            Revert
          </button>
        </div>
      )}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx && npm run typecheck && npm run lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/components/CameraTuning.tsx desktop/src/renderer/src/components/CameraTuning.test.tsx
git commit -m "feat(desktop): apply tuning changes live with explicit save and revert"
```

---

## Task 12: Calibrate without touching Start/Stop

**Files:**
- Modify: `desktop/src/renderer/src/components/CameraTuning.tsx`
- Test: `desktop/src/renderer/src/components/CameraTuning.test.tsx`

**Interfaces:**
- Consumes: `calibrate`, `calibrating`, `profile`, `applyProfile` (existing hook members); `start`/`stop` props (Task 10).
- Produces: no new exports.

- [ ] **Step 1: Write the failing tests**

Append to `CameraTuning.test.tsx`:

```tsx
describe('calibration from the live tab', () => {
  function renderWithLifecycle(overrides = {}, order: string[] = []) {
    const api = fakeApi({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => {
        order.push('calibrate')
        return PROFILE
      },
      ...overrides
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => { order.push('start') }}
        stop={async () => { order.push('stop') }}
        cameraName="Logitech StreamCam"
        debounceMs={0}
        deps={{ apiFactory: () => api, pollHealth: false }}
      />
    )
    return order
  }

  it('stops, calibrates, then restarts', async () => {
    // The camera is exclusive during a sweep, so the sidecar 409s while
    // capture runs. The operator should not have to know that.
    const order = renderWithLifecycle()
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    await waitFor(() => expect(order).toEqual(['stop', 'calibrate', 'start']))
  })

  it('restarts capture even when the sweep fails', async () => {
    // Otherwise a failed calibration leaves the operator staring at a dark
    // feed with no idea why.
    const order = renderWithLifecycle({
      calibrateCamera: async () => {
        order.push('calibrate')
        throw new Error('camera busy')
      }
    })
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    await waitFor(() => expect(order).toEqual(['stop', 'calibrate', 'start']))
  })

  it('shows the measured evidence before applying anything', async () => {
    renderWithLifecycle()
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    // Review-first: the operator sees the two framerates and the proposed
    // patch, and chooses.
    expect(await screen.findByTestId('tuning-profile')).toHaveTextContent('29.9')
    expect(screen.getByTestId('tuning-apply-profile')).toBeInTheDocument()
  })
})
```

Note the `order` array is captured by the closure in `renderWithLifecycle`; declare it before `render` as shown.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx -t calibration`
Expected: FAIL — no `tuning-calibrate` element.

- [ ] **Step 3: Write the implementation**

Add to `CameraTuning.tsx`:

```tsx
  // The sidecar refuses to calibrate while capture holds the camera, and the
  // sweep releases and reopens the device twice (~60-90 s on a StreamCam).
  // Driving the lifecycle here means the operator presses one button instead
  // of learning that constraint.
  const handleCalibrate = async (): Promise<void> => {
    await stop()
    try {
      await calibrate()
    } catch {
      // Surfaced through the hook's `error`; the restart below still runs.
    } finally {
      await start()
    }
  }
```

and the UI, inside the expanded region:

```tsx
      <button
        type="button"
        className="btn-outline btn-small"
        disabled={calibrating}
        data-testid="tuning-calibrate"
        onClick={() => void handleCalibrate()}
        title="Stops capture, measures the camera, then starts again"
      >
        {calibrating ? <Spinner /> : null} Calibrate camera
      </button>

      {calibrating && (
        <p className="field-hint" data-testid="tuning-calibrating">
          Measuring camera — about a minute. The feed resumes when it finishes.
        </p>
      )}

      {profile && !calibrating && (
        <div className="tuning-profile" data-testid="tuning-profile">
          <p className="field-hint">
            Measured {profile.fps_auto_exposure} fps on automatic exposure,{' '}
            {profile.fps_capped_exposure} fps with it capped.
          </p>
          {Object.keys(profile.recommended).length === 0 ? (
            <p className="field-hint" data-testid="tuning-no-recommendation">
              No settings to change: this camera did not respond to any of the controls
              we can set.
            </p>
          ) : (
            <>
              <ul>
                {Object.entries(profile.recommended).map(([k, v]) => (
                  <li key={k}>
                    {k}: {String(v)}
                  </li>
                ))}
              </ul>
              <button
                className="btn-primary btn-small"
                disabled={saving}
                data-testid="tuning-apply-profile"
                onClick={() => void applyProfile()}
              >
                {saving ? <Spinner /> : null} Apply these settings
              </button>
            </>
          )}
        </div>
      )}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npx vitest run src/renderer/src/components/CameraTuning.test.tsx && npm run typecheck`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/components/CameraTuning.tsx desktop/src/renderer/src/components/CameraTuning.test.tsx
git commit -m "feat(desktop): calibrate from the live tab without a manual stop/start"
```

---

## Task 13: Mount on Live, strip Admin

**Files:**
- Modify: `desktop/src/renderer/src/views/LiveView.tsx`, `desktop/src/renderer/src/views/AdminPanel.tsx`, `desktop/src/renderer/src/hooks/useSidecarSettings.ts`
- Test: `desktop/src/renderer/src/views/LiveView.test.tsx`, `desktop/src/renderer/src/views/AdminPanel.test.tsx`

**Interfaces:**
- Consumes: `CameraTuning` (Tasks 10–12); `SETTINGS_GROUPS[].home` (Task 7).
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

Append to `LiveView.test.tsx`:

```tsx
it('shows the camera tuning card in the side rail', async () => {
  renderLiveView()
  expect(await screen.findByTestId('camera-tuning')).toBeInTheDocument()
})

it('gives the tuning card the same capture state the toolbar uses', async () => {
  // One owner: LiveView drives capture through useSidecarStream, and the
  // card is told. A second health poller would disagree mid start/stop.
  renderLiveView()
  await screen.findByTestId('camera-tuning')
  expect(screen.getByTestId('tuning-idle')).toBeInTheDocument()
})
```

Append to `AdminPanel.test.tsx`:

```tsx
describe('after the tuning fields moved to Live', () => {
  it('no longer renders the relocated fields', async () => {
    renderAdmin()
    await screen.findByTestId('hardware-info')
    for (const key of [
      'conf_threshold',
      'infer_frame_skip',
      'preview_height',
      'preview_max_fps',
      'track_expiry_s'
    ]) {
      expect(document.getElementById(key)).toBeNull()
    }
  })

  it('no longer renders the calibration card or the quality readout', async () => {
    renderAdmin()
    await screen.findByTestId('hardware-info')
    expect(screen.queryByTestId('calibrate-camera')).toBeNull()
    expect(screen.queryByTestId('camera-quality')).toBeNull()
  })

  it('still renders the fields that need a restart', async () => {
    renderAdmin()
    await screen.findByTestId('hardware-info')
    expect(document.getElementById('capture_width')).not.toBeNull()
    expect(document.getElementById('imgsz')).not.toBeNull()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd desktop && npx vitest run src/renderer/src/views`
Expected: FAIL — no tuning card on Live; Admin still renders everything.

- [ ] **Step 3: Write the implementation**

`LiveView` has no camera name to pass: `FrameMessage` carries none, and the enumerated list lives behind the settings hook the card already owns. So the card resolves its own name and the prop becomes optional.

In `CameraTuning.tsx`, change the prop and the header:

```tsx
  // Optional: the card falls back to the configured index when the host
  // does not know the device's name.
  cameraName?: string
```

```tsx
        <span className="tuning-camera">
          {cameraName ?? (settings ? `Camera ${settings.camera_index}` : '')}
        </span>
```

The Task 10 tests pass `cameraName` explicitly, so they keep asserting on the literal name.

In `LiveView.tsx`, import `CameraTuning` and render it in the side rail after the stats card and before the log card:

```tsx
          <CameraTuning
            port={port}
            running={running}
            start={start}
            stop={stop}
            deps={deps?.settingsDeps}
          />
```

Add `settingsDeps?: SettingsDeps` to `StreamDeps` in `useSidecarStream.ts`, with a comment saying it is passed through to `CameraTuning` and never used by the stream hook itself — it lives there so `LiveView` keeps taking exactly one `deps` prop.

In `AdminPanel.tsx`:
- Change `visibleGroups` to filter on `home === 'admin'` as well as the existing Roboflow condition:

```tsx
  const visibleGroups = SETTINGS_GROUPS.filter(
    (g) => g.home === 'admin' && (g.label !== 'Roboflow API backends' || backendIsRemote)
  )
```

- Delete the `camera_index` calibration block (the `<>…</>` containing `calibrate-camera` and `calibration-result`) from `renderField`.
- Delete the entire `{cameraQuality?.available && (<section className="camera-quality" …>)}` block.
- Remove `cameraQuality`, `calibrate`, `calibrating`, `profile`, `applyProfile` from the destructured hook result.

**Delete the Admin tests for what moved.** `AdminPanel.test.tsx` has passing tests covering the calibration card, the quality readout, and the relocated fields; removing the source without removing them turns the suite red. Before deleting each one, check whether `CameraTuning.test.tsx` covers the same behaviour — the behaviour moved, it did not vanish. If a deleted Admin test asserted something the card's tests do not, port it across rather than losing the coverage. Name every test you delete in your report.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd desktop && npm test && npm run typecheck && npm run lint`
Expected: PASS across all 96+ tests.

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/views desktop/src/renderer/src/hooks/useSidecarStream.ts desktop/src/renderer/src/components/CameraTuning.tsx
git commit -m "feat(desktop): move camera tuning and calibration to the live tab"
```

---

## Task 14: Documentation

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Update the architecture notes**

In `CLAUDE.md`, under the `settings_store.py` bullet, replace the sentence describing `HOT_RELOADABLE_FIELDS`/`RESTART_REQUIRED_FIELDS` with:

```
Also owns `HOT_RELOADABLE_FIELDS`/`RESTART_REQUIRED_FIELDS` (the single source
of truth for which settings a running pipeline picks up live) and
`compute_warnings()`. The hot set covers more than the fields `Pipeline`
re-reads: `conf_threshold` reaches the detector through `set_conf()` and the
four `camera_*` controls reach the open device through
`CameraCapture.set_controls()`, both routed by `_push_live_settings()` in
`main.py`. `PATCH /api/settings?persist=false` applies without writing the
file, and `POST /api/settings/save` commits what is in memory — that pair is
what lets the Live tab's tuning card drag a slider without every intermediate
value becoming the startup config.
```

Under the `camera.py` bullet, append:

```
`set_controls()` queues control changes for the capture thread rather than
writing them inline: `cv2.VideoCapture` is not thread-safe and `_loop` is
calling `read()`, so a `set()` from the FastAPI request thread would race it.
Changes are coalesced into a dict, so a fast slider drag costs one write.
```

Add a `views/LiveView.tsx` note:

```
The side rail also hosts `components/CameraTuning.tsx` — the camera controls
that can change while capture runs, plus calibration. `SETTINGS_GROUPS[].home`
decides whether a group renders here or in `AdminPanel`; it is one list
filtered by both views, so a field cannot appear in both or neither.
```

Update the testing-conventions paragraph's list of hand-mirrored contracts to include `FieldGroup.home` and the four `camera_*` `FieldMeta` bounds (which mirror `SettingsUpdateRequest`'s `ge`/`le`).

- [ ] **Step 2: Verify the whole suite**

Run: `cd sidecar && python -m pytest -q` then `cd ../desktop && npm test && npm run typecheck && npm run lint`
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: describe live settings and the live-tab tuning card"
```

---

## Self-Review

**Spec coverage.** Every design section maps to a task: `camera.py` → 1, `inference.py` → 2, `settings_store.py` + `_push_live_settings` → 3, `persist=false` + save → 4, `device_key_for` + `GET /api/camera/profile` → 5, `api.ts` → 6, `FieldMeta`/`home` → 7, `pollHealth` + stored profile → 8, `savedRef` baseline → 9, the card and its support gating → 10, debounce + Save/Revert → 11, calibration orchestration → 12, placement and Admin stripping → 13, docs → 14. Both named risks are covered by tests: the exposure trap by the quality readout in Task 10, and the `savedRef` regression explicitly in Task 9.

**Two things the plan resolves that the spec left implicit.** The spec said "one apply per loop pass"; the implementation achieves it by updating a dict rather than draining a queue, and Task 1 tests that intermediate drag values are never written. And `pollHealth: false` also disables the `cameraQuality` refresh that shares that timer — Task 8 flags it, and the card owns its own hook instance (with `pollHealth` defaulting to false) so the readout keeps updating without a second poller in `LiveView`.

**One deliberate carry-over.** `SettingsUpdateRequest` uses `exclude_none=True`, so no PATCH can set a camera control back to `null` ("leave the camera alone"). That predates this work and the tuning card never needs it — sliders always send a number. Restoring `null` remains a job for Restore Defaults in Admin.
