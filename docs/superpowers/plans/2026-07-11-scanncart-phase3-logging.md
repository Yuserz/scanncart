# SCANnCART Phase 3 — Logging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist capture sessions and deduplicated detection events to SQLite in the sidecar, expose a minimal read endpoint, and have the desktop item log reconcile against it so it survives WebSocket reconnects.

**Architecture:** A new `logging_store.py` in the sidecar owns `data/scanncart.db` (single connection, `check_same_thread=False`, serialized by a lock) and is the sole DB writer. The `Pipeline` gains an optional logging hook that dedups by Ultralytics `track_id` (insert-once, update `max_conf`, time-based `left_at` resolution). `main.py` opens a session on `capture/start`, resolves + ends it on `capture/stop`, and serves `GET /api/logs` for the current session. The renderer seeds its item log from `/api/logs` on WS open, then merges live frames.

**Tech Stack:** Python 3.14, FastAPI, sqlite3 (stdlib), pytest. TypeScript, React, Vitest.

## Global Constraints

- Data model exactly per master spec §5 (`sessions`, `detection_events` tables and columns/indexes). `app_settings` is **out of scope** (Phase 4).
- SQLite is the **sole writer** from the sidecar; the renderer reads only via REST.
- Only detections with a non-null `track_id` are logged (no stable identity otherwise).
- Track expiry is **time-based**, default `track_expiry_s = 1.5`, configurable in `settings.py`.
- `confidence` and `entered_at` are frozen at first sighting; `max_conf` tracks the best confidence seen.
- Follow existing patterns: sidecar tests use FastAPI `TestClient` and stub source/detector; desktop tests use Vitest with injected `apiFactory`/`streamFactory` deps.
- Reference: [Phase 3 design](../specs/2026-07-11-scanncart-phase3-logging-design.md), [master spec](../specs/2026-07-11-scanncart-prototype-design.md).

---

### Task 1: `logging_store.py` — SQLite sessions + dedup events

**Files:**
- Create: `sidecar/app/logging_store.py`
- Test: `sidecar/tests/test_logging_store.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `EventRow` dataclass: `track_id:int, class_name:str, confidence:float, max_conf:float, entered_at:float, left_at:float|None`
  - `class LoggingStore(db_path: str, clock: Callable[[], float] = time.time)` with methods:
    - `start_session(model_name: str, device: str) -> int`
    - `end_session(session_id: int) -> None`
    - `record_detection(session_id: int, track_id: int, cls: str, conf: float, ts: float) -> None`
    - `resolve_left(session_id: int, track_id: int, ts: float) -> None`
    - `current_session_id() -> int | None`
    - `query_events(session_id: int) -> list[EventRow]`
    - `close() -> None`

- [ ] **Step 1: Write the failing tests**

Create `sidecar/tests/test_logging_store.py`:

```python
from app.logging_store import LoggingStore, EventRow


def _store():
    # ":memory:" keeps one connection alive for the store's lifetime.
    return LoggingStore(":memory:")


def test_start_session_returns_incrementing_ids():
    s = _store()
    a = s.start_session("yolo11n.pt", "cpu")
    b = s.start_session("yolo11n.pt", "cpu")
    assert a == 1 and b == 2
    assert s.current_session_id() == 2


def test_record_detection_inserts_one_row_per_track():
    s = _store()
    sid = s.start_session("yolo11n.pt", "cpu")
    s.record_detection(sid, 7, "banana", 0.80, ts=100.0)
    s.record_detection(sid, 7, "banana", 0.95, ts=100.2)  # update
    s.record_detection(sid, 7, "banana", 0.60, ts=100.4)  # lower, ignored for max
    rows = s.query_events(sid)
    assert len(rows) == 1
    r = rows[0]
    assert r.confidence == 0.80          # frozen at first sighting
    assert r.entered_at == 100.0         # frozen at first sighting
    assert r.max_conf == 0.95            # best seen
    assert r.left_at is None


def test_resolve_left_sets_left_at_once():
    s = _store()
    sid = s.start_session("yolo11n.pt", "cpu")
    s.record_detection(sid, 3, "apple", 0.9, ts=10.0)
    s.resolve_left(sid, 3, ts=12.5)
    s.resolve_left(sid, 3, ts=99.0)      # must not overwrite an already-set left_at
    rows = s.query_events(sid)
    assert rows[0].left_at == 12.5


def test_query_events_scoped_to_session_and_ordered():
    s = _store()
    s1 = s.start_session("m", "cpu")
    s.record_detection(s1, 1, "banana", 0.9, ts=5.0)
    s2 = s.start_session("m", "cpu")
    s.record_detection(s2, 2, "apple", 0.9, ts=2.0)
    s.record_detection(s2, 3, "orange", 0.9, ts=1.0)
    assert [r.track_id for r in s.query_events(s1)] == [1]
    assert [r.track_id for r in s.query_events(s2)] == [3, 2]  # ordered by entered_at


def test_current_session_id_none_when_empty():
    assert _store().current_session_id() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_logging_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.logging_store'`

- [ ] **Step 3: Write the implementation**

Create `sidecar/app/logging_store.py`:

```python
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at   REAL NOT NULL,
  ended_at     REAL,
  model_name   TEXT NOT NULL,
  device       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detection_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   INTEGER NOT NULL REFERENCES sessions(id),
  track_id     INTEGER NOT NULL,
  class_name   TEXT NOT NULL,
  confidence   REAL NOT NULL,
  entered_at   REAL NOT NULL,
  left_at      REAL,
  max_conf     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON detection_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_entered ON detection_events(entered_at);
"""


@dataclass
class EventRow:
    track_id: int
    class_name: str
    confidence: float
    max_conf: float
    entered_at: float
    left_at: float | None


class LoggingStore:
    """Sole SQLite writer for the sidecar. One connection guarded by a lock so
    the capture-request thread and the pipeline thread can share it safely."""

    def __init__(self, db_path: str, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def start_session(self, model_name: str, device: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at, model_name, device) VALUES (?, ?, ?)",
                (self._clock(), model_name, device),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (self._clock(), session_id),
            )
            self._conn.commit()

    def record_detection(
        self, session_id: int, track_id: int, cls: str, conf: float, ts: float
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, max_conf FROM detection_events "
                "WHERE session_id = ? AND track_id = ?",
                (session_id, track_id),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO detection_events "
                    "(session_id, track_id, class_name, confidence, entered_at, left_at, max_conf) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, track_id, cls, conf, ts, None, conf),
                )
            elif conf > row["max_conf"]:
                self._conn.execute(
                    "UPDATE detection_events SET max_conf = ? WHERE id = ?",
                    (conf, row["id"]),
                )
            self._conn.commit()

    def resolve_left(self, session_id: int, track_id: int, ts: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE detection_events SET left_at = ? "
                "WHERE session_id = ? AND track_id = ? AND left_at IS NULL",
                (ts, session_id, track_id),
            )
            self._conn.commit()

    def current_session_id(self) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM sessions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return int(row["id"]) if row is not None else None

    def query_events(self, session_id: int) -> list[EventRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT track_id, class_name, confidence, max_conf, entered_at, left_at "
                "FROM detection_events WHERE session_id = ? ORDER BY entered_at",
                (session_id,),
            ).fetchall()
        return [
            EventRow(
                track_id=r["track_id"],
                class_name=r["class_name"],
                confidence=r["confidence"],
                max_conf=r["max_conf"],
                entered_at=r["entered_at"],
                left_at=r["left_at"],
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_logging_store.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add sidecar/app/logging_store.py sidecar/tests/test_logging_store.py
git commit -m "feat(sidecar): add SQLite logging store for sessions and dedup events"
```

---

### Task 2: Pipeline dedup hook

**Files:**
- Modify: `sidecar/app/pipeline.py`
- Modify: `sidecar/app/settings.py`
- Test: `sidecar/tests/test_pipeline_logging.py`

**Interfaces:**
- Consumes: `LoggingStore.record_detection`, `LoggingStore.resolve_left` (Task 1).
- Produces:
  - `Settings.track_expiry_s: float = 1.5`
  - `Pipeline.__init__(..., logging_store=None, session_id=None, track_expiry_s=1.5, clock=time.time)` (all new params keyword-only-usable, defaulted so existing call sites are unaffected)
  - `Pipeline.resolve_open_tracks() -> None`

- [ ] **Step 1: Add the `track_expiry_s` setting**

In `sidecar/app/settings.py`, add the field to the `Settings` dataclass after `preview_height`:

```python
    preview_height: int = 720
    track_expiry_s: float = 1.5
```

- [ ] **Step 2: Write the failing tests**

Create `sidecar/tests/test_pipeline_logging.py`:

```python
from app.pipeline import Pipeline
from app.settings import Settings
from app.schemas import Detection


class FakeStore:
    def __init__(self):
        self.records = []   # (session_id, track_id, cls, conf, ts)
        self.resolved = []  # (session_id, track_id, ts)

    def record_detection(self, session_id, track_id, cls, conf, ts):
        self.records.append((session_id, track_id, cls, conf, ts))

    def resolve_left(self, session_id, track_id, ts):
        self.resolved.append((session_id, track_id, ts))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ScriptedSource:
    width = 128
    height = 96
    fps = 30.0

    def latest(self):
        import numpy as np
        return (1, np.full((96, 128, 3), 50, dtype=np.uint8))


class ScriptedDetector:
    """Returns whatever detection list is queued for the next infer() call."""
    names = {0: "banana"}

    def __init__(self, script):
        self._script = list(script)

    def infer(self, frame):
        return self._script.pop(0) if self._script else []


def _pipe(script, store, clock, expiry=1.5):
    return Pipeline(
        ScriptedSource(),
        ScriptedDetector(script),
        Settings(track_expiry_s=expiry),
        on_message=lambda m: None,
        logging_store=store,
        session_id=42,
        track_expiry_s=expiry,
        clock=clock,
    )


def test_new_track_is_recorded_once_per_frame():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det, det], store, clock)
    pipe.process_once()
    pipe.process_once()
    assert [r[1] for r in store.records] == [5, 5]   # recorded each frame; store dedups
    assert store.resolved == []                       # still present, not resolved


def test_untracked_detection_is_not_logged():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=None, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det], store, clock)
    pipe.process_once()
    assert store.records == []


def test_track_is_resolved_after_expiry():
    store, clock = FakeStore(), FakeClock()
    seen = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([seen, [], []], store, clock, expiry=1.0)
    clock.t = 0.0
    pipe.process_once()          # track 5 seen at t=0
    clock.t = 0.5
    pipe.process_once()          # empty; 0.5s gap < expiry → not resolved
    assert store.resolved == []
    clock.t = 2.0
    pipe.process_once()          # empty; 2.0s since last-seen > expiry → resolved
    assert store.resolved == [(42, 5, 0.0)]   # left_at = last-seen time


def test_resolve_open_tracks_flushes_remaining():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=8, cls="apple", conf=0.9, box=(0, 0, 0.5, 0.5))]
    pipe = _pipe([det], store, clock)
    clock.t = 3.0
    pipe.process_once()
    pipe.resolve_open_tracks()
    assert store.resolved == [(42, 8, 3.0)]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_pipeline_logging.py -v`
Expected: FAIL — `Pipeline.__init__() got an unexpected keyword argument 'logging_store'`

- [ ] **Step 4: Implement the hook in `pipeline.py`**

Update the `Pipeline` class. Change the imports at the top to include `time`:

```python
import base64
import threading
import time
from typing import Callable
import cv2
import numpy as np
from app.schemas import Detection, Stats, FrameMessage
```

Replace the `__init__` signature and body:

```python
    def __init__(
        self,
        source,
        detector,
        settings,
        on_message: Callable[[dict], None],
        logging_store=None,
        session_id=None,
        track_expiry_s: float = 1.5,
        clock: Callable[[], float] = time.time,
    ):
        self._source = source
        self._detector = detector
        self._settings = settings
        self._on_message = on_message
        self._logging_store = logging_store
        self._session_id = session_id
        self._track_expiry_s = track_expiry_s
        self._clock = clock
        self._open: dict[int, float] = {}   # track_id -> last-seen timestamp
        self._thread = None
        self.is_running = False
        self._frame_counter = 0
        self._last_infer_ts = None
        self._infer_fps = 0.0
```

In `process_once`, after `detections = self._detector.infer(frame)` and before building `msg`, log the detections. Add the call right after the infer-fps block, before `jpeg = ...`:

```python
        self._log_detections(detections)

        jpeg = encode_preview_jpeg(frame, self._settings.preview_height)
```

Add two methods to the class (e.g. after `process_once`):

```python
    def _log_detections(self, detections: list[Detection]) -> None:
        if self._logging_store is None or self._session_id is None:
            return
        now = self._clock()
        for d in detections:
            if d.track_id is None:
                continue
            self._logging_store.record_detection(
                self._session_id, d.track_id, d.cls, d.conf, now
            )
            self._open[d.track_id] = now
        for track_id, last_seen in list(self._open.items()):
            if now - last_seen > self._track_expiry_s:
                self._logging_store.resolve_left(self._session_id, track_id, last_seen)
                del self._open[track_id]

    def resolve_open_tracks(self) -> None:
        if self._logging_store is None or self._session_id is None:
            return
        for track_id, last_seen in list(self._open.items()):
            self._logging_store.resolve_left(self._session_id, track_id, last_seen)
        self._open.clear()
```

- [ ] **Step 5: Run the new + existing pipeline tests to verify they pass**

Run: `cd sidecar && python -m pytest tests/test_pipeline_logging.py tests/test_pipeline.py tests/test_settings.py -v`
Expected: PASS (all). Existing pipeline tests still pass because the new params are optional.

- [ ] **Step 6: Commit**

```bash
git add sidecar/app/pipeline.py sidecar/app/settings.py sidecar/tests/test_pipeline_logging.py
git commit -m "feat(sidecar): dedup + time-based track expiry logging in pipeline"
```

---

### Task 3: Wire sessions into `main.py` + `GET /api/logs`

**Files:**
- Modify: `sidecar/app/schemas.py`
- Modify: `sidecar/app/main.py`
- Modify: `sidecar/tests/test_main.py` (inject in-memory DB in the test client)
- Test: `sidecar/tests/test_logs_api.py`

**Interfaces:**
- Consumes: `LoggingStore` (Task 1), `Pipeline.resolve_open_tracks` + new `Pipeline` params (Task 2).
- Produces:
  - `schemas.LogEvent`, `schemas.LogsResponse`
  - `AppState.db_path: str`, `AppState.logging_store: LoggingStore | None`, `AppState.session_id: int | None`
  - `GET /api/logs` → `LogsResponse`

- [ ] **Step 1: Add response schemas**

In `sidecar/app/schemas.py`, append:

```python
class LogEvent(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    max_conf: float
    entered_at: float
    left_at: float | None = None


class LogsResponse(BaseModel):
    session_id: int | None = None
    events: list[LogEvent] = []
```

- [ ] **Step 2: Write the failing tests**

Create `sidecar/tests/test_logs_api.py`:

```python
import numpy as np
from fastapi.testclient import TestClient
from app.main import build_app, AppState
from app.schemas import Detection


class _StubSource:
    width, height, fps = 128, 96, 30.0

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


def _client():
    state = AppState(
        source_factory=lambda s: _StubSource(),
        detector_factory=lambda s, d: _StubDetector(),
        db_path=":memory:",
    )
    return TestClient(build_app(lambda: state))


def test_logs_empty_before_any_session():
    r = _client().get("/api/logs")
    assert r.status_code == 200
    assert r.json() == {"session_id": None, "events": []}


def test_logs_report_current_session_events_after_a_run():
    client = _client()
    client.post("/api/capture/start")
    # Pull a frame so the pipeline records at least one detection.
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
    client.post("/api/capture/stop")

    body = client.get("/api/logs").json()
    assert body["session_id"] == 1
    assert len(body["events"]) == 1
    ev = body["events"][0]
    assert ev["track_id"] == 1
    assert ev["class_name"] == "banana"
    assert ev["max_conf"] == 0.9
    assert ev["left_at"] is not None   # capture/stop resolved the open track
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd sidecar && python -m pytest tests/test_logs_api.py -v`
Expected: FAIL — `TypeError: AppState.__init__() got an unexpected keyword argument 'db_path'`

- [ ] **Step 4: Update `AppState` and wire the session lifecycle**

In `sidecar/app/main.py`, update imports:

```python
from app.schemas import HealthResponse, LogEvent, LogsResponse
from app.camera import CameraCapture
from app.inference import YoloDetector
from app.logging_store import LoggingStore
```

Extend `AppState` (add three fields; keep existing ones):

```python
@dataclass
class AppState:
    settings: Settings = field(default_factory=Settings)
    source_factory: Callable = _default_source_factory
    detector_factory: Callable = _default_detector_factory
    ws_manager: WSManager = field(default_factory=WSManager)
    pipeline: Pipeline | None = None
    state: str = "idle"
    device: str = ""
    db_path: str = "data/scanncart.db"
    logging_store: LoggingStore | None = None
    session_id: int | None = None

    def __post_init__(self):
        if not self.device:
            self.device = resolve_device(self.settings.device)
        if self.logging_store is None:
            self.logging_store = LoggingStore(self.db_path)
```

In `build_app`, update the `start` handler to open a session and pass the logging hook to the pipeline:

```python
    @app.post("/api/capture/start")
    async def start():
        if state.state != "running":
            source = state.source_factory(state.settings)
            if hasattr(source, "open"):
                source.open()
            detector = state.detector_factory(state.settings, state.device)
            state.session_id = state.logging_store.start_session(
                state.settings.active_model, state.device
            )
            state.pipeline = Pipeline(
                source, detector, state.settings,
                on_message=state.ws_manager.submit,
                logging_store=state.logging_store,
                session_id=state.session_id,
                track_expiry_s=state.settings.track_expiry_s,
            )
            state.pipeline.start()
            state.state = "running"
        return {"state": state.state}
```

Update the `stop` handler to resolve open tracks and end the session:

```python
    @app.post("/api/capture/stop")
    async def stop():
        if state.pipeline is not None:
            state.pipeline.stop()
            state.pipeline.resolve_open_tracks()
            state.pipeline = None
        if state.session_id is not None:
            state.logging_store.end_session(state.session_id)
            state.session_id = None
        state.state = "idle"
        return {"state": state.state}
```

Add the `/api/logs` endpoint (e.g. after the `stop` handler, before the websocket route):

```python
    @app.get("/api/logs", response_model=LogsResponse)
    async def logs():
        sid = state.logging_store.current_session_id()
        if sid is None:
            return LogsResponse(session_id=None, events=[])
        events = [
            LogEvent(
                track_id=r.track_id,
                class_name=r.class_name,
                confidence=r.confidence,
                max_conf=r.max_conf,
                entered_at=r.entered_at,
                left_at=r.left_at,
            )
            for r in state.logging_store.query_events(sid)
        ]
        return LogsResponse(session_id=sid, events=events)
```

- [ ] **Step 5: Keep the existing `test_main.py` off the real DB file**

In `sidecar/tests/test_main.py`, update `_make_client` so the shared state uses an in-memory DB (prevents tests writing `data/scanncart.db`):

```python
def _make_client():
    state = AppState(
        source_factory=lambda settings: _StubSource(),
        detector_factory=lambda settings, device: _StubDetector(),
        db_path=":memory:",
    )
    return TestClient(build_app(lambda: state)), state
```

- [ ] **Step 6: Run the sidecar suite to verify it passes**

Run: `cd sidecar && python -m pytest tests/test_logs_api.py tests/test_main.py tests/test_schemas.py -v`
Expected: PASS (all)

- [ ] **Step 7: Full sidecar regression + commit**

Run: `cd sidecar && python -m pytest -q`
Expected: PASS (all tests). Then:

```bash
git add sidecar/app/schemas.py sidecar/app/main.py sidecar/tests/test_main.py sidecar/tests/test_logs_api.py
git commit -m "feat(sidecar): sessions lifecycle + GET /api/logs for current session"
```

---

### Task 4: Desktop REST client — `getLogs()`

**Files:**
- Modify: `desktop/src/renderer/src/lib/api.ts`
- Modify: `desktop/src/renderer/src/lib/api.test.ts`

**Interfaces:**
- Consumes: `GET /api/logs` (Task 3).
- Produces:
  - `interface LogEvent { track_id, class_name, confidence, max_conf, entered_at, left_at: number | null }`
  - `interface LogsResponse { session_id: number | null; events: LogEvent[] }`
  - `ApiClient.getLogs(): Promise<LogsResponse>`

- [ ] **Step 1: Write the failing test**

In `desktop/src/renderer/src/lib/api.test.ts`, add inside the `describe('createApiClient', ...)` block:

```typescript
  it('getLogs() GETs /api/logs and returns parsed JSON', async () => {
    mockFetchOnce({
      session_id: 3,
      events: [
        {
          track_id: 1,
          class_name: 'banana',
          confidence: 0.8,
          max_conf: 0.91,
          entered_at: 100.0,
          left_at: null
        }
      ]
    })
    const api = createApiClient(8765)
    const r = await api.getLogs()
    expect(r.session_id).toBe(3)
    expect(r.events[0].class_name).toBe('banana')
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/logs')
    expect(init?.method ?? 'GET').toBe('GET')
  })
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd desktop && npx vitest run src/renderer/src/lib/api.test.ts`
Expected: FAIL — `api.getLogs is not a function`

- [ ] **Step 3: Implement `getLogs`**

In `desktop/src/renderer/src/lib/api.ts`, add the types after `StateResponse` and extend `ApiClient`:

```typescript
export interface LogEvent {
  track_id: number
  class_name: string
  confidence: number
  max_conf: number
  entered_at: number
  left_at: number | null
}

export interface LogsResponse {
  session_id: number | null
  events: LogEvent[]
}

export interface ApiClient {
  health(): Promise<HealthResponse>
  start(): Promise<StateResponse>
  stop(): Promise<StateResponse>
  getLogs(): Promise<LogsResponse>
}
```

Add the method to the returned object:

```typescript
  return {
    health: () => request<HealthResponse>('/health', 'GET'),
    start: () => request<StateResponse>('/capture/start', 'POST'),
    stop: () => request<StateResponse>('/capture/stop', 'POST'),
    getLogs: () => request<LogsResponse>('/logs', 'GET')
  }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd desktop && npx vitest run src/renderer/src/lib/api.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add desktop/src/renderer/src/lib/api.ts desktop/src/renderer/src/lib/api.test.ts
git commit -m "feat(desktop): add getLogs() REST client method and types"
```

---

### Task 5: Desktop — reconcile item log from `/api/logs` on connect

**Files:**
- Modify: `desktop/src/renderer/src/hooks/useSidecarStream.ts`
- Modify: `desktop/src/renderer/src/views/LiveView.test.tsx` (fake api needs `getLogs`)
- Test: `desktop/src/renderer/src/hooks/useSidecarStream.test.tsx`

**Interfaces:**
- Consumes: `ApiClient.getLogs` (Task 4), `LoggedItem` (existing).
- Produces: no new exports; behavior change — on WS open, `items` and the dedup set are seeded from `/api/logs` before live frames merge.

- [ ] **Step 1: Update the LiveView test harness fake api**

In `desktop/src/renderer/src/views/LiveView.test.tsx`, extend the fake `apiFactory` so `getLogs` exists (the hook now calls it on open):

```typescript
  const deps = {
    apiFactory: () => ({
      health: vi.fn(),
      start,
      stop,
      getLogs: vi.fn(async () => ({ session_id: null, events: [] }))
    }),
    streamFactory: (opts: StreamClientOptions) => {
      captured = opts
      return { connect: vi.fn(), close: vi.fn() }
    }
  }
```

- [ ] **Step 2: Write the failing hook test**

Create `desktop/src/renderer/src/hooks/useSidecarStream.test.tsx`:

```typescript
import { describe, it, expect, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useSidecarStream } from './useSidecarStream'
import type { StreamClientOptions, FrameMessage } from '../lib/ws'
import type { LogsResponse } from '../lib/api'

function frameWith(dets: FrameMessage['detections']): FrameMessage {
  return {
    type: 'frame',
    ts: 123,
    seq: 1,
    jpeg: 'AAAA',
    detections: dets,
    stats: { infer_fps: 1, capture_fps: 1, latency_ms: 1 }
  }
}

function makeDeps(logs: LogsResponse) {
  let opts: StreamClientOptions | null = null
  const deps = {
    apiFactory: () => ({
      health: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      getLogs: vi.fn(async () => logs)
    }),
    streamFactory: (o: StreamClientOptions) => {
      opts = o
      return { connect: vi.fn(), close: vi.fn() }
    }
  }
  return { deps, opts: () => opts! }
}

describe('useSidecarStream reconciliation', () => {
  it('seeds items from /api/logs on open, then merges live frames without duplicates', async () => {
    const { deps, opts } = makeDeps({
      session_id: 1,
      events: [
        {
          track_id: 7,
          class_name: 'banana',
          confidence: 0.8,
          max_conf: 0.9,
          entered_at: 100,
          left_at: null
        }
      ]
    })
    const { result } = renderHook(() => useSidecarStream(8765, deps))

    act(() => opts().onOpen?.())
    // Seeded from the persisted log.
    await waitFor(() => expect(result.current.items).toHaveLength(1))
    expect(result.current.items[0]).toMatchObject({ track_id: 7, cls: 'banana' })

    // A live frame for the already-seeded track must not duplicate it.
    act(() =>
      opts().onFrame?.(
        frameWith([{ track_id: 7, cls: 'banana', conf: 0.95, box: [0, 0, 0.5, 0.5] }])
      )
    )
    expect(result.current.items).toHaveLength(1)

    // A live frame for a new track appends.
    act(() =>
      opts().onFrame?.(
        frameWith([{ track_id: 8, cls: 'apple', conf: 0.7, box: [0, 0, 0.5, 0.5] }])
      )
    )
    expect(result.current.items).toHaveLength(2)
  })
})
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd desktop && npx vitest run src/renderer/src/hooks/useSidecarStream.test.tsx`
Expected: FAIL — items stays empty after `onOpen` (no seeding yet).

- [ ] **Step 4: Implement seeding in the hook**

In `desktop/src/renderer/src/hooks/useSidecarStream.ts`, replace the body of the `useEffect` so `onOpen` seeds from the persisted log. The key changes: add a `cancelled` guard and a `seedFromLogs` function, and call it from `onOpen`.

```typescript
  useEffect(() => {
    const api = apiFactory(port)
    apiRef.current = api
    seenRef.current = new Set()
    let cancelled = false

    const seedFromLogs = async (): Promise<void> => {
      try {
        const res = await api.getLogs()
        if (cancelled) return
        const seeded: LoggedItem[] = []
        for (const e of res.events) {
          if (seenRef.current.has(e.track_id)) continue
          seenRef.current.add(e.track_id)
          seeded.push({ track_id: e.track_id, cls: e.class_name, conf: e.max_conf, ts: e.entered_at })
        }
        if (seeded.length > 0) setItems((prev) => [...prev, ...seeded])
      } catch {
        // /api/logs unavailable (sidecar not ready): keep the live-only log.
      }
    }

    const onFrame = (msg: FrameMessage): void => {
      setFrame(msg)
      const fresh: LoggedItem[] = []
      for (const d of msg.detections) {
        if (d.track_id == null || seenRef.current.has(d.track_id)) continue
        seenRef.current.add(d.track_id)
        fresh.push({ track_id: d.track_id, cls: d.cls, conf: d.conf, ts: msg.ts })
      }
      if (fresh.length > 0) setItems((prev) => [...prev, ...fresh])
    }

    const onStatus = (msg: StatusMessage): void => setStatusState(msg.state)

    const client = streamFactory({
      port,
      onFrame,
      onStatus,
      onOpen: () => {
        setConnected(true)
        void seedFromLogs()
      },
      onClose: () => setConnected(false)
    })
    client.connect()

    return () => {
      cancelled = true
      client.close()
    }
  }, [port, apiFactory, streamFactory])
```

- [ ] **Step 5: Run the hook test + full desktop suite to verify they pass**

Run: `cd desktop && npx vitest run`
Expected: PASS (all files, including the updated `LiveView.test.tsx` and `api.test.ts`).

- [ ] **Step 6: Typecheck + commit**

Run: `cd desktop && npm run typecheck`
Expected: no errors. Then:

```bash
git add desktop/src/renderer/src/hooks/useSidecarStream.ts desktop/src/renderer/src/hooks/useSidecarStream.test.tsx desktop/src/renderer/src/views/LiveView.test.tsx
git commit -m "feat(desktop): reconcile item log from /api/logs on WS open"
```

---

## Verification (end of plan)

- [ ] Sidecar: `cd sidecar && python -m pytest -q` → all pass.
- [ ] Desktop: `cd desktop && npx vitest run` → all pass; `npm run typecheck` → clean.
- [ ] Manual smoke (optional, needs camera): start the app, click Start, confirm item-log rows persist across a manual sidecar reconnect and that `sidecar/data/scanncart.db` contains `sessions` and `detection_events` rows.

---

## Notes for the implementer

- **Thread-safety:** `LoggingStore` uses one connection with `check_same_thread=False` and a lock because the FastAPI request thread opens/closes sessions while the pipeline thread writes events. Do not open per-call connections.
- **`:memory:` in tests** keeps a single connection alive for the store's lifetime, so data persists within one `LoggingStore` instance — exactly what the tests rely on. Never pass `:memory:` in production (`AppState.db_path` defaults to the real file).
- **Why `record_detection` is called every frame:** the in-pipeline `_open` map tracks liveness for expiry; the store itself is idempotent on `(session_id, track_id)`, so calling it each frame only updates `max_conf`. This keeps the pipeline logic trivial.
- **`left_at` = last-seen time**, not the sweep time — the item left when it was last actually seen, per master spec §5.
