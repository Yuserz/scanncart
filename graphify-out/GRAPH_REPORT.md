# Graph Report - scanncart  (2026-07-20)

## Corpus Check
- 90 files · ~48,091 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 771 nodes · 1211 edges · 50 communities (47 shown, 3 thin omitted)
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 213 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5fb5774c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- api.ts
- devDependencies
- Detection
- useSidecarStream.ts
- AppState
- CameraCapture
- Settings
- test_presets.py
- scripts
- test_settings_api.py
- YoloDetector
- SidecarSupervisor
- SCANnCART Vision-Only Prototype — Design Spec
- LoggingStore
- AppShell.tsx
- SCANnCART — Hardware-class detection + CPU/GPU choice — Design
- Global Constraints
- test_main.py
- compilerOptions
- Design
- tsconfig.node.json
- 📘 SCANnCART YOLO11 – Vision‑Only Prototype PRD
- Global constraints
- Architecture
- Design
- Global Constraints
- Global Constraints
- Global Constraints
- SCANnCART Phase 3 — Logging — Design
- SCANnCART — YOLO26 as experimental model options — Design
- SCANnCART — Live View fills the window height — Design
- run.py
- SCANnCART Sidecar (Phase 1)
- 🚀 SCANnCART – Deployment & Future Phases
- SCANnCART Desktop (Phase 2 — Electron shell + Live View)
- SCANnCART
- index.d.ts
- tsconfig.json
- index.ts
- Run the SCANnCART desktop app

## God Nodes (most connected - your core abstractions)
1. `Settings` - 45 edges
2. `AppState` - 28 edges
3. `Detection` - 28 edges
4. `Pipeline` - 24 edges
5. `_make_client()` - 23 edges
6. `WSManager` - 21 edges
7. `scripts` - 16 edges
8. `ApiClient` - 16 edges
9. `CameraCapture` - 16 edges
10. `LoggingStore` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_detection_allows_null_track_id()` --calls--> `Detection`  [INFERRED]
  sidecar/tests/test_schemas.py → sidecar/app/schemas.py
- `test_buffer_returns_none_when_empty()` --calls--> `LatestFrameBuffer`  [INFERRED]
  sidecar/tests/test_camera.py → sidecar/app/camera.py
- `AppState` --uses--> `CameraCapture`  [INFERRED]
  sidecar/app/main.py → sidecar/app/camera.py
- `WSManager` --uses--> `CameraCapture`  [INFERRED]
  sidecar/app/main.py → sidecar/app/camera.py
- `test_capture_open_failure_sets_not_open()` --calls--> `CameraCapture`  [INFERRED]
  sidecar/tests/test_camera_capture.py → sidecar/app/camera.py

## Import Cycles
- None detected.

## Communities (50 total, 3 thin omitted)

### Community 0 - "api.ts"
Cohesion: 0.07
Nodes (35): AppShell(), AppShellProps, NoopWS, View, errorMessage(), SettingsDeps, SidecarSettings, baseSettings() (+27 more)

### Community 1 - "devDependencies"
Cohesion: 0.04
Nodes (49): devDependencies, electron, electron-builder, @electron-toolkit/eslint-config-prettier, @electron-toolkit/eslint-config-ts, @electron-toolkit/tsconfig, electron-vite, eslint (+41 more)

### Community 2 - "Detection"
Cohesion: 0.09
Nodes (24): encode_preview_jpeg(), Pipeline, Detection, ndarray, Detection, _frame(), FakeClock, FakeStore (+16 more)

### Community 3 - "useSidecarStream.ts"
Cohesion: 0.06
Nodes (28): App(), AppProps, NoopWS, Spinner(), LoggedItem, SidecarStream, StreamDeps, useSidecarStream() (+20 more)

### Community 4 - "AppState"
Cohesion: 0.09
Nodes (28): BaseModel, AppState, build_app(), WSManager, ApplyPresetRequest, FrameMessage, HardwareInfo, HealthResponse (+20 more)

### Community 5 - "CameraCapture"
Cohesion: 0.07
Nodes (20): CameraCapture, _default_capture(), FakeFrameSource, FrameSource, LatestFrameBuffer, ndarray, Protocol, Test double that yields the provided frames in order, then None. (+12 more)

### Community 6 - "Settings"
Cohesion: 0.10
Nodes (41): Any, FastAPI, SettingsResponse, _apply_settings_patch(), _default_detector_factory(), _settings_response(), resolve_device(), Settings (+33 more)

### Community 7 - "test_presets.py"
Cohesion: 0.12
Nodes (25): HardwareInfo, HardwareInfo, _list_display_adapters(), probe_hardware(), Best-effort enumeration of display adapter names via Windows     Win32_VideoCont, Preset, recommend_preset(), _fake_torch_no_cuda() (+17 more)

### Community 8 - "scripts"
Cohesion: 0.07
Nodes (27): author, dependencies, @electron-toolkit/preload, @electron-toolkit/utils, description, homepage, main, name (+19 more)

### Community 9 - "test_settings_api.py"
Cohesion: 0.14
Nodes (20): _fake_hardware(), _make_client(), _StubDetector, _StubSource, test_apply_preset_while_idle_applies_settings(), test_apply_preset_while_running_is_rejected(), test_apply_unknown_preset_returns_404(), test_get_settings_returns_current_values_and_field_classification() (+12 more)

### Community 10 - "YoloDetector"
Cohesion: 0.14
Nodes (15): _clamp01(), Detector, normalize_detections(), Detection, ndarray, Protocol, YoloDetector, _FakeBoxes (+7 more)

### Community 11 - "SidecarSupervisor"
Cohesion: 0.11
Nodes (6): resolveSidecarPaths(), startSidecar(), SidecarSupervisor, SpawnedLike, SupervisorOptions, FakeChild

### Community 12 - "SCANnCART Vision-Only Prototype — Design Spec"
Cohesion: 0.09
Nodes (23): 10. Build phasing, 11. Out of scope (see DEPLOYMENT.md), 1. Summary, 2.1 Process topology, 2.2 Project layout, 2. Architecture, 3.1 Core pipeline, 3.2 Module responsibilities (+15 more)

### Community 13 - "LoggingStore"
Cohesion: 0.14
Nodes (9): EventRow, LoggingStore, Sole SQLite writer for the sidecar. One connection guarded by a lock so     the, _store(), test_current_session_id_none_when_empty(), test_query_events_scoped_to_session_and_ordered(), test_record_detection_inserts_one_row_per_track(), test_resolve_left_sets_left_at_once() (+1 more)

### Community 14 - "AppShell.tsx"
Cohesion: 0.20
Nodes (6): APP_DIR, { _electron: electron }, electronBin, REPO_ROOT, require, SKILL_DIR

### Community 15 - "SCANnCART — Hardware-class detection + CPU/GPU choice — Design"
Cohesion: 0.12
Nodes (15): 1. Backend detection — `sidecar/app/hardware.py`, 2. Device default semantics — `sidecar/app/settings.py`, 3. Schemas — `sidecar/app/schemas.py`, 4. UI — desktop, Data flow, Design, Error handling, Goals (+7 more)

### Community 16 - "Global Constraints"
Cohesion: 0.13
Nodes (14): Definition of Done (Phase 1), Global Constraints, prints: SIDECAR_PORT=8765, SCANnCART Phase 1 — Sidecar Core Implementation Plan, Self-Review, Task 1: Project scaffold, requirements, and settings, Task 2: Message schemas, Task 3: Latest-frame buffer and frame source interface (+6 more)

### Community 17 - "test_main.py"
Cohesion: 0.21
Nodes (9): _fake_hardware(), _make_client(), _StubDetector, _StubSource, test_cross_origin_requests_get_cors_headers(), test_health_reports_idle_and_model(), test_start_then_stop_transitions_state(), test_system_info_reports_accelerator() (+1 more)

### Community 18 - "compilerOptions"
Cohesion: 0.15
Nodes (13): compilerOptions, baseUrl, composite, jsx, paths, extends, include, @renderer/* (+5 more)

### Community 19 - "Design"
Cohesion: 0.17
Nodes (11): 1. Shared theme, 2. `AppShell` nav, 3. `LiveView` — sidebar dashboard layout, 4. `AdminPanel` — grouped sections, 5. Testing, Design, Goals, Non-goals (+3 more)

### Community 20 - "tsconfig.node.json"
Cohesion: 0.18
Nodes (10): compilerOptions, composite, types, extends, include, @electron-toolkit/tsconfig/tsconfig.node.json, electron.vite.config.*, electron-vite/node (+2 more)

### Community 21 - "📘 SCANnCART YOLO11 – Vision‑Only Prototype PRD"
Cohesion: 0.18
Nodes (11): 1. Overview, 2. Objectives, 3. Scope, 4.1 Architecture, 4. Tech Stack (Recommended), 5. Functional Requirements, 6. Non-Functional Requirements, 7. Success Metrics (+3 more)

### Community 22 - "Global constraints"
Cohesion: 0.18
Nodes (10): Definition of Done (Phase 2), Global constraints, SCANnCART Phase 2 — Electron Shell + Live View Implementation Plan, Stack (from official electron-vite scaffold; tool-picked versions), Task 1 — REST client `src/renderer/src/lib/api.ts`, Task 2 — WebSocket client `src/renderer/src/lib/ws.ts`, Task 3 — Overlay geometry `src/renderer/src/lib/overlay.ts`, Task 4 — LiveView + hook `src/renderer/src/views/LiveView.tsx` + `hooks/useSidecarStream.ts` (+2 more)

### Community 23 - "Architecture"
Cohesion: 0.18
Nodes (9): Architecture, Commands, Desktop (`desktop/src/`), Desktop (Electron + React + TypeScript, in `desktop/`), graphify, Sidecar (Python, in `sidecar/`), Sidecar (`sidecar/app/`), Testing conventions (+1 more)

### Community 24 - "Design"
Cohesion: 0.20
Nodes (9): Admin panel, App boot screen (`App.tsx` + new `App.css`), `components/Spinner.tsx` + `Spinner.css`, Design, LiveView start/stop feedback, Non-goals, Problem, SCANnCART — Loading animations for module/model waits — Design (+1 more)

### Community 25 - "Global Constraints"
Cohesion: 0.22
Nodes (9): Global Constraints, Notes for the implementer, SCANnCART Phase 3 — Logging — Implementation Plan, Task 1: `logging_store.py` — SQLite sessions + dedup events, Task 2: Pipeline dedup hook, Task 3: Wire sessions into `main.py` + `GET /api/logs`, Task 4: Desktop REST client — `getLogs()`, Task 5: Desktop — reconcile item log from `/api/logs` on connect (+1 more)

### Community 26 - "Global Constraints"
Cohesion: 0.22
Nodes (8): Global Constraints, Notes for the implementer, SCANnCART Device Detection Implementation Plan, Task 1: Sidecar — adapter enumeration + `accelerator` classification, Task 2: Sidecar — expose `accelerator` on `SystemInfoResponse`, Task 3: Desktop — `accelerator` type + class-aware hardware label, Task 4: Desktop — GPU/CPU device toggle, Verification (end of plan)

### Community 27 - "Global Constraints"
Cohesion: 0.22
Nodes (8): Global Constraints, SCANnCART Desktop UI/UX Redesign Implementation Plan, Task 1: Shared theme CSS variables, Task 2: `LiveView` sidebar-dashboard layout, Task 3: `AppShell` nav restyle, Task 4: `SETTINGS_GROUPS` data structure, Task 5: `AdminPanel` grouped-sections layout, Task 6: Manual verification pass

### Community 28 - "SCANnCART Phase 3 — Logging — Design"
Cohesion: 0.22
Nodes (9): 1. Scope, 2.1 `logging_store.py` (new — sole DB writer), 2.2 Dedup wired into the pipeline, 2.3 `GET /api/logs` (minimal), 2. Sidecar, 3. Desktop, 4. Testing, 5. Non-goals / caveats (+1 more)

### Community 30 - "SCANnCART — YOLO26 as experimental model options — Design"
Cohesion: 0.25
Nodes (7): Design, Desktop (hand-kept mirror, per project convention), Non-goals, Problem, SCANnCART — YOLO26 as experimental model options — Design, Sidecar, Testing

### Community 31 - "SCANnCART — Live View fills the window height — Design"
Cohesion: 0.29
Nodes (6): Design, Goals, Non-goals, Problem, SCANnCART — Live View fills the window height — Design, Testing

### Community 32 - "run.py"
Cohesion: 0.43
Nodes (4): main(), pick_port(), test_pick_port_falls_back_when_taken(), test_pick_port_returns_preferred_when_free()

### Community 33 - "SCANnCART Sidecar (Phase 1)"
Cohesion: 0.29
Nodes (7): GPU (NVIDIA/CUDA) setup, Run, SCANnCART Sidecar (Phase 1), Setup, Tests, Verify manually, Windows Defender / antivirus (silent native-import crashes)

### Community 34 - "🚀 SCANnCART – Deployment & Future Phases"
Cohesion: 0.33
Nodes (6): 1. Deployment Goals, 2. Centralized Architecture (Cart → Server), 3. Future Phases (Out-of-Scope for Prototype), 4. Deployment Non-Functional Targets, 5. Deployment Success Metrics, 🚀 SCANnCART – Deployment & Future Phases

### Community 35 - "SCANnCART Desktop (Phase 2 — Electron shell + Live View)"
Cohesion: 0.40
Nodes (5): Architecture, Run the full app (manual — needs the StreamCam), SCANnCART Desktop (Phase 2 — Electron shell + Live View), Setup, Test (headless — no display, camera, or model)

### Community 36 - "SCANnCART"
Cohesion: 0.50
Nodes (4): Architecture, For agents, Quick start, SCANnCART

### Community 49 - "Run the SCANnCART desktop app"
Cohesion: 0.25
Nodes (7): Build, Gotchas (all actually hit), Prerequisites, Run (agent path), Run (human path), Run the SCANnCART desktop app, UI handles

## Knowledge Gaps
- **226 isolated node(s):** `SKILL_DIR`, `REPO_ROOT`, `APP_DIR`, `require`, `{ _electron: electron }` (+221 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Settings` to `Detection`, `AppState`, `CameraCapture`, `test_settings_api.py`, `test_main.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `AppState` connect `AppState` to `Detection`, `CameraCapture`, `Settings`, `test_presets.py`, `test_settings_api.py`, `YoloDetector`, `LoggingStore`, `test_main.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `CameraCapture` connect `CameraCapture` to `Detection`, `AppState`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Are the 39 inferred relationships involving `Settings` (e.g. with `AppState` and `WSManager`) actually correct?**
  _`Settings` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `AppState` (e.g. with `CameraCapture` and `HardwareInfo`) actually correct?**
  _`AppState` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `Detection` (e.g. with `Detector` and `YoloDetector`) actually correct?**
  _`Detection` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `Pipeline` (e.g. with `AppState` and `WSManager`) actually correct?**
  _`Pipeline` has 16 INFERRED edges - model-reasoned connections that need verification._