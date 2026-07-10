# SCANnCART Vision-Only Prototype — Design Spec

**Date:** 2026-07-11
**Status:** Approved for planning
**Source PRD:** [`docs/PRD.md`](../../PRD.md) · **Future scope:** [`docs/DEPLOYMENT.md`](../../DEPLOYMENT.md)

---

## 1. Summary

A local, single-PC prototype that turns a smart cart into a real-time grocery
scanner. A **Python sidecar** owns the USB camera, runs **YOLO11** object
detection with tracking, logs detected items to **SQLite**, and streams a
preview plus detection metadata to an **Electron + React** UI over localhost.
The UI provides a live view and a no-auth **admin panel** for managing models,
camera/inference settings, and logs.

Everything runs on one machine with no server and no network dependency.

### Key decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Model source | **Pretrained COCO YOLO11** (`yolo11n`) as default; custom models uploaded via admin | No custom grocery model exists yet; COCO includes grocery-relevant classes (banana, apple, orange, bottle, etc.) to prove the pipeline |
| Multi-version support | Accept **any Ultralytics-compatible `.pt`** (YOLOv8/9/10/11) | Ultralytics `YOLO(path)` loads all versions through one API — near-free to support; validated on upload |
| Frame transport | **Option B (hybrid):** sidecar sends downscaled **720p JPEG preview + detection JSON**; UI overlays boxes | Lighter payloads, smoother at target FPS, UI flexibility; inference still runs at full capture resolution |
| Integration topology | **Approach A:** FastAPI in sidecar exposing **REST (admin) + WebSocket (stream)**; Python owns camera, model, and SQLite | Clean boundaries, single DB writer, trivial file upload, UI is a pure client |
| Logging | **Option 1 (event-based dedup):** one row per item per session via Ultralytics **track IDs** | Matches "cart scanner" model; keeps DB small and log readable |
| Target hardware | **AMD Ryzen 7 5700U — CPU-only** (no CUDA); device auto-detect | Realistic ~15–30 fps inference with `yolo11n`; frame-dropping keeps display smooth; optional OpenVINO acceleration |
| Admin panel | 4 sections: **Models, Camera, Inference, Logs**; **no auth** | Local single-user prototype; tunable/demoable without code edits |

---

## 2. Architecture

### 2.1 Process topology

```
┌─────────────────────────────────────────────────────────┐
│  Electron App (Node main + React renderer)               │
│   • Spawns & supervises the Python sidecar on startup     │
│   • React UI = pure client (Live View + Admin Panel)      │
└───────────────┬───────────────────────┬──────────────────┘
                │ REST (admin/commands)  │ WebSocket (stream)
                ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│  Python Sidecar (FastAPI + Uvicorn)                       │
│   • Camera capture (OpenCV) — sole owner                   │
│   • YOLO11 tracking (Ultralytics .track())                │
│   • Model registry (default + uploaded .pt files)         │
│   • SQLite (sole writer)                                   │
└───────────────┬─────────────────────────────────────────┘
                ▼
   Logitech StreamCam (USB)
```

The two processes are independently runnable: the sidecar can be started alone
and exercised with `curl` / a browser before Electron is involved.

### 2.2 Project layout

```
scanncart/
├── docs/                          # PRD, DEPLOYMENT, this spec
├── sidecar/                       # Python service
│   ├── app/
│   │   ├── main.py                # FastAPI app: REST + WS wiring
│   │   ├── camera.py              # OpenCV capture thread (sole camera owner)
│   │   ├── inference.py           # YOLO11 load + track loop
│   │   ├── pipeline.py            # capture→infer→dedup→broadcast orchestration
│   │   ├── models_registry.py     # list/upload/validate/switch .pt models
│   │   ├── logging_store.py       # SQLite read/write, event dedup, CSV export
│   │   ├── settings.py            # runtime config (camera/inference/device)
│   │   └── schemas.py             # Pydantic models for API + WS messages
│   ├── models/                    # stored .pt files (default + uploaded)
│   ├── data/                      # scanncart.db (SQLite)
│   ├── requirements.txt
│   └── tests/
├── desktop/                       # Electron + React
│   ├── electron/
│   │   ├── main.ts                # window + spawn/supervise sidecar
│   │   └── preload.ts             # safe IPC bridge
│   ├── src/
│   │   ├── views/LiveView.tsx     # video + box overlay + controls + item log
│   │   ├── views/AdminPanel.tsx   # 4 admin sections (tabs)
│   │   ├── lib/ws.ts              # WebSocket client
│   │   └── lib/api.ts             # REST client
│   ├── package.json
│   └── tests/
└── README.md
```

---

## 3. Sidecar internals

### 3.1 Core pipeline

Three decoupled stages so a slow CPU never stalls the UI:

```
[Camera thread]        [Latest-frame buffer]        [Inference + broadcast]
capture @60fps  ──►  size-1, newest wins,   ──►  YOLO11.track() (~15-30 fps)
(OpenCV, own          stale frames dropped         → detections + 720p JPEG
 thread)                                            → WS broadcast to clients
```

- **Latest-frame buffer (size 1):** the camera thread always overwrites with the
  newest frame; inference consumes whatever is current. This "newest frame wins /
  drop stale" pattern keeps latency low and prevents backlog on CPU.
- **Tracking + dedup:** `model.track(persist=True)` returns a stable **track ID**
  per item. `pipeline.py` holds an in-memory set of active track IDs:
  - New ID → write one "item entered" event row (via `logging_store`).
  - ID absent for N consecutive frames → resolve `left_at` for that event.
  - Track's `max_conf` updated in place while alive.
- **Broadcast payload:** per processed frame, a downscaled **720p JPEG** preview
  **+** compact **detections JSON** (track_id, class, confidence, normalized box).
- **Device & acceleration:** `settings.py` auto-detects CUDA (falls back to CPU
  on the Ryzen). Optional **OpenVINO** export path for faster x86 CPU inference,
  toggled in admin.
- **Model hot-swap:** switching the active model pauses the pipeline, reloads the
  new `.pt`, resets the tracker, and resumes — no process restart.

### 3.2 Module responsibilities

Each module has one purpose and is independently testable:

| Module | Responsibility | Depends on |
|---|---|---|
| `camera.py` | Own the USB camera, produce frames | OpenCV |
| `inference.py` | Load model, run `.track()`, return detections | Ultralytics |
| `pipeline.py` | Orchestrate stages, dedup, throttle, broadcast | camera, inference, logging_store |
| `models_registry.py` | Validate/store/list/switch `.pt` files | Ultralytics (load-check) |
| `logging_store.py` | SQLite writes/reads, dedup rows, CSV export | sqlite3 |
| `settings.py` | Hold + update runtime config, persist to DB | logging_store |

---

## 4. API contract

### 4.1 WebSocket — `ws://127.0.0.1:<port>/ws/stream`

Server → client push, at display rate.

```jsonc
// One message per processed frame
{
  "type": "frame",
  "ts": 1720598400.123,           // epoch seconds
  "seq": 48213,                    // frame sequence number
  "jpeg": "<base64 720p JPEG>",    // downscaled preview
  "detections": [
    { "track_id": 12, "cls": "banana", "conf": 0.91,
      "box": [0.31, 0.22, 0.48, 0.55] }   // x1,y1,x2,y2 normalized 0-1
  ],
  "stats": { "infer_fps": 22.4, "capture_fps": 60, "latency_ms": 88 }
}
```
```jsonc
// Status / heartbeat
{ "type": "status",
  "state": "running|paused|camera_lost|model_switching|error",
  "detail": "..." }
```

### 4.2 REST API

JSON unless noted. Base path `/api`.

| Method / Path | Purpose |
|---|---|
| `GET /api/health` | Liveness + current state, active model, device |
| `POST /api/capture/start` · `/stop` | Start/stop the pipeline (UI controls) |
| `GET /api/models` | List models (name, version, classes, size, active?) |
| `POST /api/models` (multipart) | Upload `.pt`; validates by loading + reading classes |
| `POST /api/models/{id}/activate` | Hot-swap active model |
| `DELETE /api/models/{id}` | Remove an uploaded model (default cannot be deleted) |
| `GET /api/settings` · `PATCH /api/settings` | Read/update camera, inference, device config |
| `GET /api/cameras` | Enumerate available capture devices |
| `GET /api/logs?from=&to=&cls=&limit=&offset=` | Query detection events (paged) |
| `DELETE /api/logs` | Clear the log |
| `GET /api/logs/export.csv` | Download CSV |

### 4.3 Model-upload validation

On `POST /api/models`, the sidecar loads the file with `YOLO(path)` in a guarded
call, confirms it is a **detection** model, extracts `names` (classes) and version
metadata, and **rejects with a clear 4xx error** if it fails to load or is the
wrong task type. Only validated files enter the registry.

### 4.4 Port handling

The sidecar binds a fixed localhost port (default `8765`); if taken, it selects a
free port and reports the actual port to Electron via **stdout on spawn**, so the
UI always connects to the correct place.

---

## 5. Data model (SQLite)

DB at `sidecar/data/scanncart.db`, written only by `logging_store.py`.

```sql
-- One row per capture session (start/stop cycle)
CREATE TABLE sessions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at   REAL NOT NULL,          -- epoch seconds
  ended_at     REAL,                   -- null while running
  model_name   TEXT NOT NULL,          -- active model at session start
  device       TEXT NOT NULL           -- 'cpu' | 'cuda'
);

-- One row per detected item (event-based dedup: first appearance)
CREATE TABLE detection_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   INTEGER NOT NULL REFERENCES sessions(id),
  track_id     INTEGER NOT NULL,       -- Ultralytics track id (unique per session)
  class_name   TEXT NOT NULL,
  confidence   REAL NOT NULL,          -- confidence at first solid detection
  entered_at   REAL NOT NULL,          -- first-seen timestamp
  left_at      REAL,                   -- last-seen timestamp (null if unresolved)
  max_conf     REAL NOT NULL           -- best confidence observed for this track
);
CREATE INDEX idx_events_session ON detection_events(session_id);
CREATE INDEX idx_events_entered ON detection_events(entered_at);

-- Persisted runtime settings (single row, survives restarts)
CREATE TABLE app_settings (
  id                INTEGER PRIMARY KEY CHECK (id = 1),
  active_model      TEXT NOT NULL,
  camera_index      INTEGER NOT NULL DEFAULT 0,
  capture_width     INTEGER NOT NULL DEFAULT 1280,
  capture_height    INTEGER NOT NULL DEFAULT 720,
  capture_fps       INTEGER NOT NULL DEFAULT 60,
  conf_threshold    REAL NOT NULL DEFAULT 0.5,
  infer_frame_skip  INTEGER NOT NULL DEFAULT 0,   -- 0 = every frame, N = skip N
  device            TEXT NOT NULL DEFAULT 'auto',
  use_openvino      INTEGER NOT NULL DEFAULT 0
);
```

**Notes:**
- `sessions` scopes the item log to a run and prevents track-ID collisions across
  runs.
- `detection_events` is the Option 1 dedup surface: one row per item per session,
  updated in place for `left_at` / `max_conf`. An item in view for minutes = one
  row, not thousands.
- `app_settings` persists admin choices so the prototype restarts as left. Model
  files live on disk in `models/`; only the **active model name** is stored here.
- `GET /api/logs` joins `detection_events` to `sessions`; CSV export is a flat
  dump of the same query.

---

## 6. UI structure

React app, two top-level views.

### 6.1 Live View (main screen)

- Canvas showing the 720p preview from the WebSocket, with a **canvas overlay**
  drawing boxes + `class conf%` labels from the detection JSON (normalized coords
  scaled to display size).
- **Start/Stop** button → `/api/capture/start|stop`.
- **Live item log** side panel — running list of items detected this session
  (from WS, reconciled against `/api/logs`).
- **Stats strip** — infer fps / capture fps / latency from the `stats` field.

### 6.2 Admin Panel (4 tabs)

1. **Models** — list, upload `.pt` (drag-drop), see classes/version, activate, delete.
2. **Camera** — device dropdown, resolution (1080p/720p), fps.
3. **Inference** — confidence slider, device (auto/cpu/cuda), frame-skip, OpenVINO toggle.
4. **Logs** — paged table, filter by class/time, clear, export CSV.

---

## 7. Error handling

Satisfies the PRD's graceful-recovery requirement:

- **Camera disconnect** → sidecar emits `status: camera_lost`, retries reopen
  every 2s; UI shows a non-blocking banner; auto-recovers on replug.
- **Sidecar crash** → Electron main supervises the child; on unexpected exit it
  **respawns** with backoff (max retries); UI shows "reconnecting"; WS client
  auto-reconnects.
- **Model load failure** (bad upload) → REST returns a clear 4xx; UI shows the
  reason; active model unchanged.
- **Port conflict** → sidecar picks a free port and reports it on stdout.

---

## 8. Testing strategy

- **Sidecar (pytest):**
  - Model-registry validation (good/bad `.pt`).
  - Dedup logic — synthetic track-ID streams → correct row counts.
  - `logging_store` CRUD + CSV export.
  - Settings persistence round-trip.
  - API endpoints via FastAPI `TestClient`.
  - Camera/inference use a **fake frame source** (pre-recorded video / synthetic
    frames) so tests run without hardware.
- **Desktop (vitest + React Testing Library):**
  - WS message → overlay rendering.
  - Admin forms and REST client.
  - Electron spawn/supervise logic against a mock sidecar.
- **End-to-end smoke test:** sidecar + a canned video → assert frames + detections
  flow over WS and events land in SQLite.

---

## 9. Non-functional targets (from PRD)

| Target | Value | Notes for CPU-only hardware |
|---|---|---|
| Processing rate | ≥ 30 fps *display* | Display smooth via frame-dropping; inference ~15–30 fps on Ryzen CPU |
| End-to-end latency | < 150 ms | Achievable per processed frame; newest-frame-wins keeps it low |
| Reliability | ≥ 2 h continuous | Bounded buffers, no leaks, supervised respawn |
| Accuracy | ≥ 90% on common items | Depends on future custom model; COCO default proves pipeline only |

> Honest caveat: a *sustained true 60 fps inference* is not realistic on the
> Ryzen 7 5700U (CPU-only). Capture and display run at 60 fps; inference runs as
> fast as the CPU allows, and the newest frame always wins.

---

## 10. Build phasing

Each phase is independently demoable:

1. **Sidecar core** — camera → YOLO11 track → WS stream + REST health. Verify with
   browser / `curl`, no Electron.
2. **Electron shell + Live View** — spawn sidecar, show preview + overlays +
   start/stop.
3. **Logging** — sessions + dedup events + item-log panel.
4. **Admin panel** — Models, Camera, Inference, Logs tabs.
5. **Hardening** — reconnect/respawn, OpenVINO path, error banners, tests.

---

## 11. Out of scope (see DEPLOYMENT.md)

Centralized server inference, edge hardware, weight sensors/ESP32, cloud
sync/analytics dashboard, mobile control app, Docker containerization.
