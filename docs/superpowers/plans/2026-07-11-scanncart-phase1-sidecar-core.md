# SCANnCART Phase 1 — Sidecar Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone Python sidecar that owns the camera, runs YOLO11 with tracking, and streams a 720p preview + detection JSON over a localhost WebSocket, with a REST health/start/stop API — verifiable in a browser with no Electron.

**Architecture:** A FastAPI + Uvicorn app hosts a WebSocket (`/ws/stream`) and REST endpoints. A background capture thread writes frames into a size-1 "latest frame wins" buffer; a pipeline thread pulls the newest frame, runs `YOLO.track()`, encodes a downscaled JPEG, and broadcasts a frame message to connected WebSocket clients. All vision components sit behind small interfaces (`FrameSource`, `Detector`) so the pipeline is fully testable with fakes — no camera or model download required to run the test suite.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Ultralytics (YOLO11), OpenCV (`opencv-python`), PyTorch (CPU), Pydantic v2, pytest, numpy.

## Global Constraints

- Python version floor: **3.11**.
- Target device: **CPU-only** (AMD Ryzen 7 5700U, no CUDA). Device resolution defaults to `auto` → falls back to `cpu`.
- The sidecar is the **sole owner** of the camera; nothing else opens it.
- WebSocket frame message shape is fixed by the spec (`type`, `ts`, `seq`, `jpeg`, `detections[]`, `stats`) — see spec §4.1.
- Detection box coordinates are **normalized 0–1** as `[x1, y1, x2, y2]`.
- Preview stream is downscaled to **720p height** regardless of capture resolution.
- Default model is **`yolo11n.pt`** (COCO). No custom-model upload in Phase 1.
- All tests must run **without hardware or network** (fake frame source + fake detector; no real model download in the automated suite).
- Bind to **`127.0.0.1`** only. Default port **8765**; if taken, pick a free port and print `SIDECAR_PORT=<n>` to stdout.

---

### Task 1: Project scaffold, requirements, and settings

**Files:**
- Create: `sidecar/requirements.txt`
- Create: `sidecar/app/__init__.py` (empty)
- Create: `sidecar/app/settings.py`
- Create: `sidecar/tests/__init__.py` (empty)
- Create: `sidecar/tests/test_settings.py`
- Create: `sidecar/pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Settings` dataclass with fields: `active_model: str`, `camera_index: int`, `capture_width: int`, `capture_height: int`, `capture_fps: int`, `conf_threshold: float`, `infer_frame_skip: int`, `device: str`, `preview_height: int`.
  - `resolve_device(pref: str) -> str` — maps `"auto"` to `"cuda"`/`"cpu"`, passes through explicit values.

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
ultralytics==8.3.*
opencv-python==4.10.*
numpy==2.1.*
pydantic==2.9.*
pytest==8.3.*
httpx==0.27.*
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 3: Create empty package files**

Create `sidecar/app/__init__.py` and `sidecar/tests/__init__.py` as empty files.

- [ ] **Step 4: Write the failing test**

Create `sidecar/tests/test_settings.py`:

```python
from app.settings import Settings, resolve_device


def test_settings_defaults():
    s = Settings()
    assert s.active_model == "yolo11n.pt"
    assert s.capture_width == 1280
    assert s.capture_height == 720
    assert s.capture_fps == 60
    assert s.conf_threshold == 0.5
    assert s.infer_frame_skip == 0
    assert s.device == "auto"
    assert s.preview_height == 720


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    # Simulate torch missing -> must fall back to cpu, never crash.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_device("auto") == "cpu"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.settings'`

- [ ] **Step 6: Write minimal implementation**

Create `sidecar/app/settings.py`:

```python
from dataclasses import dataclass


@dataclass
class Settings:
    active_model: str = "yolo11n.pt"
    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 60
    conf_threshold: float = 0.5
    infer_frame_skip: int = 0
    device: str = "auto"
    preview_height: int = 720


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_settings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add sidecar/requirements.txt sidecar/pytest.ini sidecar/app/__init__.py sidecar/app/settings.py sidecar/tests/__init__.py sidecar/tests/test_settings.py
git commit -m "feat(sidecar): scaffold project with settings and device resolution"
```

---

### Task 2: Message schemas

**Files:**
- Create: `sidecar/app/schemas.py`
- Test: `sidecar/tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces (Pydantic v2 models):
  - `Detection(track_id: int | None, cls: str, conf: float, box: tuple[float, float, float, float])`
  - `Stats(infer_fps: float, capture_fps: float, latency_ms: float)`
  - `FrameMessage(type: Literal["frame"], ts: float, seq: int, jpeg: str, detections: list[Detection], stats: Stats)`
  - `StatusMessage(type: Literal["status"], state: str, detail: str = "")`
  - `HealthResponse(state: str, active_model: str, device: str)`

- [ ] **Step 1: Write the failing test**

Create `sidecar/tests/test_schemas.py`:

```python
from app.schemas import Detection, Stats, FrameMessage, StatusMessage, HealthResponse


def test_frame_message_serializes():
    msg = FrameMessage(
        type="frame",
        ts=1720598400.123,
        seq=42,
        jpeg="Zm9v",
        detections=[Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))],
        stats=Stats(infer_fps=22.4, capture_fps=60.0, latency_ms=88.0),
    )
    dumped = msg.model_dump()
    assert dumped["type"] == "frame"
    assert dumped["detections"][0]["cls"] == "banana"
    assert dumped["detections"][0]["box"] == (0.1, 0.2, 0.3, 0.4)
    assert dumped["stats"]["infer_fps"] == 22.4


def test_detection_allows_null_track_id():
    d = Detection(track_id=None, cls="apple", conf=0.5, box=(0.0, 0.0, 1.0, 1.0))
    assert d.track_id is None


def test_status_message_default_detail():
    s = StatusMessage(type="status", state="running")
    assert s.detail == ""


def test_health_response_fields():
    h = HealthResponse(state="idle", active_model="yolo11n.pt", device="cpu")
    assert h.active_model == "yolo11n.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas'`

- [ ] **Step 3: Write minimal implementation**

Create `sidecar/app/schemas.py`:

```python
from typing import Literal
from pydantic import BaseModel


class Detection(BaseModel):
    track_id: int | None
    cls: str
    conf: float
    box: tuple[float, float, float, float]


class Stats(BaseModel):
    infer_fps: float
    capture_fps: float
    latency_ms: float


class FrameMessage(BaseModel):
    type: Literal["frame"]
    ts: float
    seq: int
    jpeg: str
    detections: list[Detection]
    stats: Stats


class StatusMessage(BaseModel):
    type: Literal["status"]
    state: str
    detail: str = ""


class HealthResponse(BaseModel):
    state: str
    active_model: str
    device: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_schemas.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/schemas.py sidecar/tests/test_schemas.py
git commit -m "feat(sidecar): add API and WebSocket message schemas"
```

---

### Task 3: Latest-frame buffer and frame source interface

**Files:**
- Create: `sidecar/app/camera.py`
- Test: `sidecar/tests/test_camera.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LatestFrameBuffer` — thread-safe size-1 buffer. `put(seq: int, frame: np.ndarray) -> None`; `get() -> tuple[int, np.ndarray] | None` (returns the most recent put, `None` if nothing yet). Newest always wins.
  - `FrameSource` (typing.Protocol): `open() -> None`, `read() -> np.ndarray | None`, `release() -> None`, properties `width: int`, `height: int`, `fps: float`.
  - `FakeFrameSource(frames: list[np.ndarray], fps: float = 30.0)` — test double cycling through provided frames; `read()` returns frames in order then `None` when exhausted.

- [ ] **Step 1: Write the failing test**

Create `sidecar/tests/test_camera.py`:

```python
import numpy as np
from app.camera import LatestFrameBuffer, FakeFrameSource


def _frame(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val, dtype=np.uint8)


def test_buffer_returns_none_when_empty():
    buf = LatestFrameBuffer()
    assert buf.get() is None


def test_buffer_newest_wins():
    buf = LatestFrameBuffer()
    buf.put(1, _frame(10))
    buf.put(2, _frame(20))
    seq, frame = buf.get()
    assert seq == 2
    assert frame[0, 0, 0] == 20


def test_fake_frame_source_yields_then_none():
    src = FakeFrameSource([_frame(1), _frame(2)], fps=30.0)
    src.open()
    assert src.read()[0, 0, 0] == 1
    assert src.read()[0, 0, 0] == 2
    assert src.read() is None
    assert src.fps == 30.0
    src.release()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_camera.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.camera'`

- [ ] **Step 3: Write minimal implementation**

Create `sidecar/app/camera.py`:

```python
import threading
from typing import Protocol
import numpy as np


class LatestFrameBuffer:
    """Thread-safe size-1 buffer where the newest frame always wins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: tuple[int, np.ndarray] | None = None

    def put(self, seq: int, frame: np.ndarray) -> None:
        with self._lock:
            self._item = (seq, frame)

    def get(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            return self._item


class FrameSource(Protocol):
    width: int
    height: int
    fps: float

    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def release(self) -> None: ...


class FakeFrameSource:
    """Test double that yields the provided frames in order, then None."""

    def __init__(self, frames: list[np.ndarray], fps: float = 30.0) -> None:
        self._frames = frames
        self._i = 0
        h, w = (frames[0].shape[0], frames[0].shape[1]) if frames else (0, 0)
        self.width = w
        self.height = h
        self.fps = fps

    def open(self) -> None:
        self._i = 0

    def read(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def release(self) -> None:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_camera.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera.py sidecar/tests/test_camera.py
git commit -m "feat(sidecar): add latest-frame buffer and frame source interface"
```

---

### Task 4: OpenCV camera capture thread

**Files:**
- Modify: `sidecar/app/camera.py`
- Test: `sidecar/tests/test_camera_capture.py`

**Interfaces:**
- Consumes: `LatestFrameBuffer` (Task 3).
- Produces:
  - `CameraCapture(index, width, height, fps, cap_factory=cv2.VideoCapture)` — a `FrameSource` that opens an OpenCV device and runs a background thread pushing frames into an internal `LatestFrameBuffer`. Methods: `open()`, `latest() -> tuple[int, np.ndarray] | None`, `release()`. `cap_factory` is injectable so tests pass a fake capture without hardware. Exposes `is_open: bool`.

- [ ] **Step 1: Write the failing test**

Create `sidecar/tests/test_camera_capture.py`:

```python
import time
import numpy as np
from app.camera import CameraCapture


class FakeCap:
    """Mimics cv2.VideoCapture: isOpened/read/set/release."""

    def __init__(self, index):
        self.index = index
        self._opened = True
        self._n = 0

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        return True

    def read(self):
        self._n += 1
        return True, np.full((4, 4, 3), self._n % 256, dtype=np.uint8)

    def release(self):
        self._opened = False


def test_capture_thread_populates_latest_frame():
    cam = CameraCapture(0, 640, 480, 30, cap_factory=FakeCap)
    cam.open()
    assert cam.is_open
    # Give the thread a moment to produce at least one frame.
    deadline = time.time() + 2.0
    got = None
    while time.time() < deadline:
        got = cam.latest()
        if got is not None:
            break
        time.sleep(0.01)
    cam.release()
    assert got is not None
    seq, frame = got
    assert seq >= 1
    assert frame.shape == (4, 4, 3)
    assert not cam.is_open


def test_capture_open_failure_sets_not_open():
    class DeadCap(FakeCap):
        def isOpened(self):
            return False

    cam = CameraCapture(0, 640, 480, 30, cap_factory=DeadCap)
    opened = cam.open()
    assert opened is False
    assert cam.is_open is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_camera_capture.py -v`
Expected: FAIL — `ImportError: cannot import name 'CameraCapture'`

- [ ] **Step 3: Write minimal implementation**

Append to `sidecar/app/camera.py` (add `import cv2`, `import time` at top with the other imports):

```python
import cv2
import time


class CameraCapture:
    """Owns an OpenCV device and runs a background capture thread."""

    def __init__(self, index, width, height, fps, cap_factory=cv2.VideoCapture):
        self.index = index
        self.width = width
        self.height = height
        self.fps = float(fps)
        self._cap_factory = cap_factory
        self._cap = None
        self._buffer = LatestFrameBuffer()
        self._thread = None
        self._running = False
        self._seq = 0
        self.is_open = False

    def open(self) -> bool:
        self._cap = self._cap_factory(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            self.is_open = False
            return False
        self.is_open = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            self._seq += 1
            self._buffer.put(self._seq, frame)

    def latest(self):
        return self._buffer.get()

    def read(self):
        got = self._buffer.get()
        return None if got is None else got[1]

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self.is_open = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_camera_capture.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/camera.py sidecar/tests/test_camera_capture.py
git commit -m "feat(sidecar): add OpenCV camera capture thread"
```

---

### Task 5: Detection normalization and YOLO detector

**Files:**
- Create: `sidecar/app/inference.py`
- Test: `sidecar/tests/test_inference.py`

**Interfaces:**
- Consumes: `Detection` (Task 2).
- Produces:
  - `normalize_detections(xyxy, confs, clss, ids, names, width, height) -> list[Detection]` — pure function. `xyxy`: list of `[x1,y1,x2,y2]` in pixels; `confs`: list[float]; `clss`: list[int]; `ids`: list[int|None] or `None`; `names`: dict[int, str]; `width`/`height`: ints. Returns `Detection`s with boxes normalized 0–1, clamped to `[0,1]`.
  - `Detector` (Protocol): `infer(frame: np.ndarray) -> list[Detection]`, property `names: dict[int, str]`.
  - `YoloDetector(model_path, device, conf, model_factory=YOLO)` — wraps Ultralytics `YOLO`, calls `.track(persist=True, conf=..., device=..., verbose=False)`, converts the first result via `normalize_detections`. `model_factory` injectable for tests.

- [ ] **Step 1: Write the failing test**

Create `sidecar/tests/test_inference.py`:

```python
import numpy as np
from app.inference import normalize_detections, YoloDetector


def test_normalize_scales_and_labels():
    dets = normalize_detections(
        xyxy=[[64, 48, 128, 96]],
        confs=[0.9],
        clss=[0],
        ids=[7],
        names={0: "banana"},
        width=128,
        height=96,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "banana"
    assert d.track_id == 7
    assert d.conf == 0.9
    assert d.box == (0.5, 0.5, 1.0, 1.0)


def test_normalize_handles_missing_ids():
    dets = normalize_detections(
        xyxy=[[0, 0, 64, 48]], confs=[0.3], clss=[1], ids=None,
        names={1: "apple"}, width=64, height=48,
    )
    assert dets[0].track_id is None


def test_normalize_clamps_out_of_bounds():
    dets = normalize_detections(
        xyxy=[[-10, -10, 200, 200]], confs=[0.5], clss=[0], ids=[1],
        names={0: "x"}, width=100, height=100,
    )
    assert dets[0].box == (0.0, 0.0, 1.0, 1.0)


class _FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]])
        self.conf = np.array([0.8])
        self.cls = np.array([0.0])
        self.id = np.array([3.0])


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()
        self.names = {0: "bottle"}


class _FakeModel:
    def __init__(self, path):
        self.path = path
        self.names = {0: "bottle"}

    def track(self, frame, **kwargs):
        return [_FakeResult()]


def test_yolo_detector_infer_converts_results():
    det = YoloDetector("yolo11n.pt", device="cpu", conf=0.5, model_factory=_FakeModel)
    frame = np.zeros((40, 30, 3), dtype=np.uint8)
    out = det.infer(frame)
    assert len(out) == 1
    assert out[0].cls == "bottle"
    assert out[0].track_id == 3
    assert out[0].box[0] == 10.0 / 30.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.inference'`

- [ ] **Step 3: Write minimal implementation**

Create `sidecar/app/inference.py`:

```python
from typing import Protocol
import numpy as np
from app.schemas import Detection


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalize_detections(xyxy, confs, clss, ids, names, width, height) -> list[Detection]:
    out: list[Detection] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        box = (
            _clamp01(x1 / width),
            _clamp01(y1 / height),
            _clamp01(x2 / width),
            _clamp01(y2 / height),
        )
        track_id = None if ids is None or ids[i] is None else int(ids[i])
        out.append(
            Detection(
                track_id=track_id,
                cls=names[int(clss[i])],
                conf=float(confs[i]),
                box=box,
            )
        )
    return out


class Detector(Protocol):
    names: dict

    def infer(self, frame: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    def __init__(self, model_path, device, conf, model_factory=None):
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        self._model = model_factory(model_path)
        self._device = device
        self._conf = conf
        self.names = self._model.names

    def infer(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame, persist=True, conf=self._conf, device=self._device, verbose=False
        )
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else boxes.xyxy
        confs = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else boxes.conf
        clss = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else boxes.cls
        ids = None
        if boxes.id is not None:
            ids = boxes.id.tolist() if hasattr(boxes.id, "tolist") else boxes.id
        h, w = frame.shape[0], frame.shape[1]
        return normalize_detections(xyxy, confs, clss, ids, r.names, w, h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_inference.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/inference.py sidecar/tests/test_inference.py
git commit -m "feat(sidecar): add detection normalization and YOLO detector"
```

---

### Task 6: Pipeline orchestration and JPEG preview encoding

**Files:**
- Create: `sidecar/app/pipeline.py`
- Test: `sidecar/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `FrameSource`/`CameraCapture` (Tasks 3–4), `Detector` (Task 5), `Settings` (Task 1), `FrameMessage`/`Stats` (Task 2).
- Produces:
  - `encode_preview_jpeg(frame, target_height) -> str` — downscale to `target_height` (preserving aspect), JPEG-encode, return base64 string.
  - `Pipeline(source, detector, settings, on_message)` — orchestrates one processing step and a background loop. `on_message: Callable[[dict], None]` is invoked with each `FrameMessage.model_dump()`. Methods: `process_once() -> dict | None` (pull latest frame, infer, encode, build+emit message, return it), `start()`, `stop()`, property `is_running: bool`. Honors `settings.infer_frame_skip`.

- [ ] **Step 1: Write the failing test**

Create `sidecar/tests/test_pipeline.py`:

```python
import numpy as np
from app.pipeline import Pipeline, encode_preview_jpeg
from app.camera import CameraCapture
from app.settings import Settings
from app.schemas import Detection


def _frame(h=96, w=128, val=100):
    return np.full((h, w, 3), val, dtype=np.uint8)


class _StubDetector:
    names = {0: "banana"}

    def __init__(self):
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return [Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]


class _StubSource:
    """Always returns the same latest frame."""
    width = 128
    height = 96
    fps = 30.0

    def latest(self):
        return (5, _frame())

    def read(self):
        return _frame()


def test_encode_preview_jpeg_returns_base64():
    s = encode_preview_jpeg(_frame(720, 1280), target_height=360)
    assert isinstance(s, str)
    assert len(s) > 0


def test_process_once_builds_frame_message():
    msgs = []
    pipe = Pipeline(_StubSource(), _StubDetector(), Settings(), on_message=msgs.append)
    out = pipe.process_once()
    assert out is not None
    assert out["type"] == "frame"
    assert out["seq"] == 5
    assert out["detections"][0]["cls"] == "banana"
    assert out["jpeg"]
    assert "infer_fps" in out["stats"]
    assert msgs and msgs[0] is out


def test_process_once_returns_none_without_frame():
    class Empty:
        width = 1
        height = 1
        fps = 1.0

        def latest(self):
            return None

    pipe = Pipeline(Empty(), _StubDetector(), Settings(), on_message=lambda m: None)
    assert pipe.process_once() is None


def test_frame_skip_skips_inference():
    det = _StubDetector()
    settings = Settings(infer_frame_skip=1)  # process 1, skip 1, ...
    pipe = Pipeline(_StubSource(), det, settings, on_message=lambda m: None)
    pipe.process_once()  # processed (infer called)
    pipe.process_once()  # skipped (no infer)
    assert det.calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `sidecar/app/pipeline.py`:

```python
import base64
import threading
import time
from typing import Callable
import cv2
import numpy as np
from app.schemas import Detection, Stats, FrameMessage


def encode_preview_jpeg(frame: np.ndarray, target_height: int) -> str:
    h, w = frame.shape[0], frame.shape[1]
    if h > target_height:
        scale = target_height / h
        frame = cv2.resize(frame, (int(w * scale), target_height))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Pipeline:
    def __init__(self, source, detector, settings, on_message: Callable[[dict], None]):
        self._source = source
        self._detector = detector
        self._settings = settings
        self._on_message = on_message
        self._thread = None
        self.is_running = False
        self._frame_counter = 0
        self._last_infer_ts = None
        self._infer_fps = 0.0

    def process_once(self) -> dict | None:
        got = self._source.latest()
        if got is None:
            return None
        seq, frame = got

        skip = self._settings.infer_frame_skip
        self._frame_counter += 1
        if skip > 0 and (self._frame_counter - 1) % (skip + 1) != 0:
            return None

        t0 = time.time()
        detections = self._detector.infer(frame)
        t1 = time.time()

        if self._last_infer_ts is not None:
            dt = t1 - self._last_infer_ts
            if dt > 0:
                self._infer_fps = 1.0 / dt
        self._last_infer_ts = t1

        jpeg = encode_preview_jpeg(frame, self._settings.preview_height)
        stats = Stats(
            infer_fps=round(self._infer_fps, 1),
            capture_fps=float(getattr(self._source, "fps", 0.0)),
            latency_ms=round((t1 - t0) * 1000.0, 1),
        )
        msg = FrameMessage(
            type="frame", ts=t1, seq=seq, jpeg=jpeg,
            detections=detections, stats=stats,
        ).model_dump()
        self._on_message(msg)
        return msg

    def _loop(self) -> None:
        while self.is_running:
            produced = self.process_once()
            if produced is None:
                time.sleep(0.005)

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_pipeline.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/pipeline.py sidecar/tests/test_pipeline.py
git commit -m "feat(sidecar): add pipeline orchestration and JPEG preview encoding"
```

---

### Task 7: FastAPI app — health, start/stop, WebSocket broadcast

**Files:**
- Create: `sidecar/app/main.py`
- Test: `sidecar/tests/test_main.py`

**Interfaces:**
- Consumes: `Settings`/`resolve_device` (Task 1), `Pipeline` (Task 6), `HealthResponse` (Task 2), `CameraCapture` (Task 4), `YoloDetector` (Task 5).
- Produces:
  - `WSManager` — tracks connected WebSocket clients; `broadcast(message: dict)` sends JSON to all (drops dead ones). Thread-safe queue bridge so the pipeline's non-async `on_message` can hand messages to the async event loop.
  - `AppState` — holds `settings`, `pipeline` (or None), `ws_manager`, `state: str`, `device: str`. `build_app(state_factory=...) -> FastAPI`.
  - REST: `GET /api/health` → `HealthResponse`; `POST /api/capture/start` → `{"state": "running"}`; `POST /api/capture/stop` → `{"state": "idle"}`. WebSocket: `/ws/stream`.
  - Camera/detector construction is injectable via `AppState` fields (`source_factory`, `detector_factory`) so tests avoid hardware and model download.

- [ ] **Step 1: Write the failing test**

Create `sidecar/tests/test_main.py`:

```python
import numpy as np
from fastapi.testclient import TestClient
from app.main import build_app, AppState
from app.schemas import Detection


class _StubSource:
    width = 128
    height = 96
    fps = 30.0

    def open(self):
        return True

    def latest(self):
        return (1, np.full((96, 128, 3), 50, dtype=np.uint8))

    def read(self):
        return np.full((96, 128, 3), 50, dtype=np.uint8)

    def release(self):
        pass


class _StubDetector:
    names = {0: "banana"}

    def infer(self, frame):
        return [Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]


def _make_client():
    state = AppState(
        source_factory=lambda settings: _StubSource(),
        detector_factory=lambda settings, device: _StubDetector(),
    )
    return TestClient(build_app(lambda: state)), state


def test_health_reports_idle_and_model():
    client, _ = _make_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["active_model"] == "yolo11n.pt"
    assert body["device"] in ("cpu", "cuda")


def test_start_then_stop_transitions_state():
    client, _ = _make_client()
    assert client.post("/api/capture/start").json()["state"] == "running"
    assert client.get("/api/health").json()["state"] == "running"
    assert client.post("/api/capture/stop").json()["state"] == "idle"


def test_websocket_receives_a_frame_after_start():
    client, _ = _make_client()
    client.post("/api/capture/start")
    with client.websocket_connect("/ws/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        assert msg["detections"][0]["cls"] == "banana"
    client.post("/api/capture/stop")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sidecar && python -m pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Write minimal implementation**

Create `sidecar/app/main.py`:

```python
import asyncio
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.settings import Settings, resolve_device
from app.pipeline import Pipeline
from app.schemas import HealthResponse
from app.camera import CameraCapture
from app.inference import YoloDetector


def _default_source_factory(settings: Settings):
    return CameraCapture(
        settings.camera_index, settings.capture_width,
        settings.capture_height, settings.capture_fps,
    )


def _default_detector_factory(settings: Settings, device: str):
    return YoloDetector(settings.active_model, device=device, conf=settings.conf_threshold)


class WSManager:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=4)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    def submit(self, message: dict) -> None:
        # Called from the pipeline thread; hand off to the event loop.
        if self._loop is None:
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            return
        asyncio.run_coroutine_threadsafe(self._drain(), self._loop)

    async def _drain(self) -> None:
        while not self._queue.empty():
            msg = self._queue.get_nowait()
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws)


@dataclass
class AppState:
    settings: Settings = field(default_factory=Settings)
    source_factory: Callable = _default_source_factory
    detector_factory: Callable = _default_detector_factory
    ws_manager: WSManager = field(default_factory=WSManager)
    pipeline: Pipeline | None = None
    state: str = "idle"
    device: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = resolve_device(self.settings.device)


def build_app(state_factory: Callable[[], AppState] = AppState) -> FastAPI:
    app = FastAPI(title="SCANnCART Sidecar")
    state = state_factory()

    @app.on_event("startup")
    async def _startup():
        state.ws_manager.bind_loop(asyncio.get_running_loop())

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            state=state.state,
            active_model=state.settings.active_model,
            device=state.device,
        )

    @app.post("/api/capture/start")
    async def start():
        if state.state != "running":
            source = state.source_factory(state.settings)
            if hasattr(source, "open"):
                source.open()
            detector = state.detector_factory(state.settings, state.device)
            state.pipeline = Pipeline(
                source, detector, state.settings,
                on_message=state.ws_manager.submit,
            )
            state.pipeline.start()
            state.state = "running"
        return {"state": state.state}

    @app.post("/api/capture/stop")
    async def stop():
        if state.pipeline is not None:
            state.pipeline.stop()
            state.pipeline = None
        state.state = "idle"
        return {"state": state.state}

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        await state.ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            state.ws_manager.disconnect(ws)

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_main.py -v`
Expected: PASS (3 tests)

> Note: `test_websocket_receives_a_frame_after_start` relies on the pipeline thread emitting a frame; the stub source returns a frame immediately, so `submit` fires within the TestClient's WebSocket context. If the WS test is flaky under TestClient's threading, the reviewer may adjust it to poll `receive_json` with the stub, but keep the health/start-stop assertions strict.

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/main.py sidecar/tests/test_main.py
git commit -m "feat(sidecar): add FastAPI app with health, start/stop, and WS broadcast"
```

---

### Task 8: Runnable entrypoint, model bootstrap, and manual verification

**Files:**
- Create: `sidecar/run.py`
- Create: `sidecar/README.md`
- Test: `sidecar/tests/test_port.py`

**Interfaces:**
- Consumes: `build_app` (Task 7).
- Produces:
  - `pick_port(preferred: int) -> int` — return `preferred` if bindable on `127.0.0.1`, else an OS-assigned free port.
  - `run.py` main: resolve port, print `SIDECAR_PORT=<n>` to stdout, `uvicorn.run(build_app(), host="127.0.0.1", port=port)`. On first run, Ultralytics auto-downloads `yolo11n.pt` when the detector is constructed at capture start.

- [ ] **Step 1: Write the failing test**

Create `sidecar/tests/test_port.py`:

```python
import socket
from app_run import pick_port  # thin import shim, see step 3


def test_pick_port_returns_preferred_when_free():
    # Find a definitely-free port, release it, then ask pick_port for it.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    assert pick_port(free) == free


def test_pick_port_falls_back_when_taken():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    taken = holder.getsockname()[1]
    try:
        chosen = pick_port(taken)
        assert chosen != taken
        assert isinstance(chosen, int)
    finally:
        holder.close()
```

- [ ] **Step 2: Create the import shim and run test to verify it fails**

Create `sidecar/app_run.py` (keeps `pick_port` importable without launching uvicorn):

```python
from run import pick_port  # re-export

__all__ = ["pick_port"]
```

Run: `cd sidecar && python -m pytest tests/test_port.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run'`

- [ ] **Step 3: Write minimal implementation**

Create `sidecar/run.py`:

```python
import socket
from app.main import build_app


def pick_port(preferred: int) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", preferred))
        s.close()
        return preferred
    except OSError:
        s.close()
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.bind(("127.0.0.1", 0))
        port = s2.getsockname()[1]
        s2.close()
        return port


def main() -> None:
    import uvicorn
    port = pick_port(8765)
    print(f"SIDECAR_PORT={port}", flush=True)
    uvicorn.run(build_app(), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_port.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite**

Run: `cd sidecar && python -m pytest -v`
Expected: PASS — all tests from Tasks 1–8 green.

- [ ] **Step 6: Write the sidecar README**

Create `sidecar/README.md`:

````markdown
# SCANnCART Sidecar (Phase 1)

Standalone Python service: camera capture → YOLO11 tracking → WebSocket stream.

## Setup

```bash
cd sidecar
python -m venv .venv
# Windows: .venv\Scripts\activate   |   *nix: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
# prints: SIDECAR_PORT=8765
```

On first capture start, Ultralytics downloads `yolo11n.pt` automatically.

## Verify manually

1. Start: `curl -X POST http://127.0.0.1:8765/api/capture/start`
2. Health: `curl http://127.0.0.1:8765/api/health`
3. Open `ws_test.html` (below) in a browser to see live frames + boxes.
4. Stop: `curl -X POST http://127.0.0.1:8765/api/capture/stop`

## Tests

```bash
python -m pytest -v
```
````

- [ ] **Step 7: Manual end-to-end verification with real hardware**

This step is performed by a human with the StreamCam connected (tests above do NOT cover real hardware). Create a scratch `sidecar/ws_test.html` locally (not committed) that connects to `ws://127.0.0.1:8765/ws/stream`, draws the `jpeg` onto an `<img>`, and overlays boxes from `detections`. Confirm:
- `SIDECAR_PORT=` prints on launch.
- After `POST /api/capture/start`, live 720p frames render with bounding boxes on grocery-relevant COCO items (banana, apple, bottle, cup).
- `stats.infer_fps` is a plausible CPU number (~15–30).
- `POST /api/capture/stop` halts the stream; the camera LED turns off.

Document the observed `infer_fps` in the commit message.

- [ ] **Step 8: Commit**

```bash
git add sidecar/run.py sidecar/app_run.py sidecar/README.md sidecar/tests/test_port.py
git commit -m "feat(sidecar): add runnable entrypoint, port selection, and docs"
```

---

## Self-Review

**Spec coverage (Phase 1 slice of spec §10.1):**
- Camera capture (sole owner, threaded, latest-frame-wins) → Tasks 3, 4. ✓
- YOLO11 with tracking (`.track(persist=True)`, track IDs in output) → Task 5. ✓
- 720p JPEG preview + detection JSON (Option B transport) → Task 6, message shape from spec §4.1. ✓
- WebSocket stream (`/ws/stream`) → Tasks 2, 7. ✓
- REST health + start/stop → Task 7. ✓
- Device auto-detect (CPU fallback) → Task 1. ✓
- Port selection + stdout report → Task 8. ✓
- Runs without Electron; testable with fakes; manual hardware check → Tasks 7, 8. ✓
- Deferred to later phases (correctly out of Phase 1 scope): SQLite logging/sessions/dedup persistence (Phase 3), model upload/registry + settings persistence + camera enumeration (Phase 4), reconnect/respawn + OpenVINO (Phase 5). Tracking runs now; dedup *persistence* comes with logging.

**Placeholder scan:** No TBD/TODO/"handle edge cases" placeholders; every code step contains complete code. ✓

**Type consistency:** `Detection`, `Stats`, `FrameMessage`, `HealthResponse` names/fields consistent across Tasks 2, 5, 6, 7. `latest() -> tuple[int, np.ndarray] | None` used consistently by `CameraCapture` (Task 4) and `Pipeline` (Task 6). `on_message`/`submit` signature (`dict -> None`) consistent between Pipeline (Task 6) and WSManager (Task 7). `source_factory`/`detector_factory` signatures consistent between Task 7 impl and its tests. ✓

---

## Definition of Done (Phase 1)

- `python -m pytest -v` green (Tasks 1–8), zero hardware/network needed.
- `python run.py` prints `SIDECAR_PORT=`, serves health/start/stop, and streams frames over WebSocket.
- Manual hardware check (Task 8 Step 7) confirms live detection + boxes on the StreamCam.
- Next: Phase 2 plan (Electron shell + Live View) consumes this WebSocket + REST contract unchanged.
```
