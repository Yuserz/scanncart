# Graph Report - scanncart  (2026-08-29)

## Corpus Check
- 103 files · ~61,878 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1140 nodes · 2019 edges · 82 communities (56 shown, 26 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 55 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `24138285`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- api.ts
- devDependencies
- main.py
- useSidecarStream.ts
- test_detector_api.py
- CameraCapture
- test_detector_backends.py
- HardwareInfo
- scripts
- test_settings_api.py
- RoboflowRemoteDetector
- SidecarSupervisor
- SCANnCART Vision-Only Prototype — Design Spec
- LoggingStore
- AppShell.tsx
- SCANnCART — Hardware-class detection + CPU/GPU choice — Design
- Global Constraints
- test_roboflow_client.py
- compilerOptions
- Design
- tsconfig.node.json
- 📘 SCANnCART YOLO11 – Vision‑Only Prototype PRD
- Global constraints
- CLAUDE.md
- Design
- Global Constraints
- Global Constraints
- Global Constraints
- SCANnCART Phase 3 — Logging — Design
- SCANnCART — YOLO26 as experimental model options — Design
- SCANnCART — Live View fills the window height — Design
- test_roboflow_detector.py
- SCANnCART Sidecar (Phase 1)
- 🚀 SCANnCART – Deployment & Future Phases
- README.md
- IouTracker
- index.d.ts
- tsconfig.json
- preload/index.ts
- test_credentials.py
- 🔌 SCANnCART – Pluggable Detector Backends
- ApiClient
- AdminPanel.tsx
- 🧠 SCANnCART – Custom Model Training Guide
- SCANnCART — System Architecture (Low Fidelity)
- driver.mjs
- Run the SCANnCART desktop app
- WSManager
- useSidecarSettings.test.tsx
- arch_doc_check.py
- @electron-toolkit/eslint-config-prettier
- @electron-toolkit/eslint-config-ts
- @electron-toolkit/tsconfig
- electron-vite
- eslint
- eslint-plugin-react
- eslint-plugin-react-hooks
- eslint-plugin-react-refresh
- jsdom
- playwright-core
- prettier
- react
- react-dom
- @testing-library/react
- @testing-library/user-event
- @types/node
- @types/react
- @types/react-dom
- typescript
- vite
- @vitejs/plugin-react
- vitest

## God Nodes (most connected - your core abstractions)
1. `Settings` - 50 edges
2. `Detection` - 34 edges
3. `build_app()` - 33 edges
4. `IouTracker` - 31 edges
5. `make()` - 31 edges
6. `make_client()` - 24 edges
7. `frame()` - 24 edges
8. `_make_client()` - 23 edges
9. `HardwareInfo` - 21 edges
10. `Pipeline` - 21 edges

## Surprising Connections (you probably didn't know these)
- `test_local_without_a_key_is_allowed()` --uses--> `RoboflowRemoteDetector`  [INFERRED]
  tests/test_detector_api.py → app/inference.py
- `test_remote_backends_build_a_roboflow_detector()` --uses--> `RoboflowRemoteDetector`  [INFERRED]
  tests/test_detector_api.py → app/inference.py
- `test_auth_errors_raise_and_do_not_retry()` --uses--> `RoboflowAuthError`  [INFERRED]
  tests/test_roboflow_client.py → app/roboflow.py
- `_StubDetector` --uses--> `Detection`  [INFERRED]
  tests/test_logs_api.py → app/schemas.py
- `_StubDetector` --uses--> `Detection`  [INFERRED]
  tests/test_settings_api.py → app/schemas.py

## Import Cycles
- None detected.

## Communities (82 total, 26 thin omitted)

### Community 0 - "api.ts"
Cohesion: 0.19
Nodes (13): SidecarSettings, createApiClient(), DetectorProbeResponse, HealthResponse, LogEvent, LogsResponse, PresetInfo, PresetsResponse (+5 more)

### Community 1 - "devDependencies"
Cohesion: 0.29
Nodes (7): devDependencies, electron, electron-builder, @testing-library/jest-dom, electron, electron-builder, @testing-library/jest-dom

### Community 2 - "main.py"
Cohesion: 0.05
Nodes (60): BaseModel, FastAPI, SettingsResponse, _apply_settings_patch(), build_app(), _settings_response(), encode_preview_jpeg(), Pipeline (+52 more)

### Community 3 - "useSidecarStream.ts"
Cohesion: 0.08
Nodes (26): LoggedItem, SidecarStream, StreamDeps, useSidecarStream(), boxToPercent(), boxToPixels(), LabeledRect, layoutDetections() (+18 more)

### Community 4 - "test_detector_api.py"
Cohesion: 0.06
Nodes (44): AppState, backend_url(), _default_detector_factory(), 401/403 — missing, invalid, or unauthorized API key., RoboflowAuthError, main(), pick_port(), client_for() (+36 more)

### Community 5 - "CameraCapture"
Cohesion: 0.06
Nodes (19): CameraCapture, FakeFrameSource, FrameSource, LatestFrameBuffer, ndarray, Protocol, Test double that yields the provided frames in order, then None., Owns an OpenCV device and runs a background capture thread. (+11 more)

### Community 6 - "test_detector_backends.py"
Cohesion: 0.05
Nodes (77): field_validator, SettingsUpdateRequest, resolve_device(), Settings, compute_warnings(), load_settings(), Any, Load settings from disk, overlaying only known/valid fields onto the hardcoded… (+69 more)

### Community 7 - "HardwareInfo"
Cohesion: 0.13
Nodes (29): HardwareInfo, HardwareInfo, _list_display_adapters(), probe_hardware(), Best-effort enumeration of display adapter names via Windows…, Preset, recommend_preset(), _fake_torch_no_cuda() (+21 more)

### Community 8 - "scripts"
Cohesion: 0.07
Nodes (29): author, dependencies, @electron-toolkit/preload, @electron-toolkit/utils, @rollup/rollup-linux-x64-gnu, description, homepage, main (+21 more)

### Community 9 - "test_settings_api.py"
Cohesion: 0.14
Nodes (20): _fake_hardware(), _make_client(), _StubDetector, _StubSource, test_apply_preset_while_idle_applies_settings(), test_apply_preset_while_running_is_rejected(), test_apply_unknown_preset_returns_404(), test_get_settings_returns_current_values_and_field_classification() (+12 more)

### Community 10 - "RoboflowRemoteDetector"
Cohesion: 0.11
Nodes (16): _clamp01(), Detector, normalize_detections(), Detection, ndarray, Protocol, Runs detection through a Roboflow Workflow — cloud or self-hosted. Satisfies…, RoboflowRemoteDetector (+8 more)

### Community 11 - "SidecarSupervisor"
Cohesion: 0.12
Nodes (7): resolveSidecarPaths(), startSidecar(), SidecarSupervisor, SpawnedLike, SupervisorOptions, FakeChild, makeSupervisor()

### Community 12 - "SCANnCART Vision-Only Prototype — Design Spec"
Cohesion: 0.09
Nodes (23): 10. Build phasing, 11. Out of scope (see DEPLOYMENT.md), 1. Summary, 2.1 Process topology, 2.2 Project layout, 2. Architecture, 3.1 Core pipeline, 3.2 Module responsibilities (+15 more)

### Community 13 - "LoggingStore"
Cohesion: 0.15
Nodes (9): EventRow, LoggingStore, Sole SQLite writer for the sidecar. One connection guarded by a lock so the…, _store(), test_current_session_id_none_when_empty(), test_query_events_scoped_to_session_and_ordered(), test_record_detection_inserts_one_row_per_track(), test_resolve_left_sets_left_at_once() (+1 more)

### Community 14 - "AppShell.tsx"
Cohesion: 0.13
Nodes (8): App(), AppProps, NoopWS, AppShell(), AppShellProps, NoopWS, View, Spinner()

### Community 15 - "SCANnCART — Hardware-class detection + CPU/GPU choice — Design"
Cohesion: 0.12
Nodes (15): 1. Backend detection — `sidecar/app/hardware.py`, 2. Device default semantics — `sidecar/app/settings.py`, 3. Schemas — `sidecar/app/schemas.py`, 4. UI — desktop, Data flow, Design, Error handling, Goals (+7 more)

### Community 16 - "Global Constraints"
Cohesion: 0.13
Nodes (14): Definition of Done (Phase 1), Global Constraints, prints: SIDECAR_PORT=8765, SCANnCART Phase 1 — Sidecar Core Implementation Plan, Self-Review, Task 1: Project scaffold, requirements, and settings, Task 2: Message schemas, Task 3: Latest-frame buffer and frame source interface (+6 more)

### Community 17 - "test_roboflow_client.py"
Cohesion: 0.05
Nodes (68): Exception, _default_client_factory(), find_predictions(), first_result(), _httpx(), _looks_like_predictions(), Any, HTTP client for Roboflow Workflows. Talks to either Roboflow's serverless… (+60 more)

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

### Community 23 - "CLAUDE.md"
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

### Community 32 - "test_roboflow_detector.py"
Cohesion: 0.10
Nodes (41): find_image_size(), The `image: {width, height}` block a workflow reports alongside its…, FakeClient, frame(), make(), parametrize, Phase 4: RoboflowRemoteDetector, wiring, and /api/detector/probe. The…, Server echoes the dims it saw; trust those over our own. (+33 more)

### Community 33 - "SCANnCART Sidecar (Phase 1)"
Cohesion: 0.29
Nodes (7): GPU (NVIDIA/CUDA) setup, Run, SCANnCART Sidecar (Phase 1), Setup, Tests, Verify manually, Windows Defender / antivirus (silent native-import crashes)

### Community 34 - "🚀 SCANnCART – Deployment & Future Phases"
Cohesion: 0.33
Nodes (6): 1. Deployment Goals, 2. Centralized Architecture (Cart → Server), 3. Future Phases (Out-of-Scope for Prototype), 4. Deployment Non-Functional Targets, 5. Deployment Success Metrics, 🚀 SCANnCART – Deployment & Future Phases

### Community 35 - "README.md"
Cohesion: 0.18
Nodes (9): Architecture, Run the full app (manual — needs the StreamCam), SCANnCART Desktop (Phase 2 — Electron shell + Live View), Setup, Test (headless — no display, camera, or model), Architecture, For agents, Quick start (+1 more)

### Community 36 - "IouTracker"
Cohesion: 0.10
Nodes (34): Box, iou(), IouTracker, Detection, Forget every track and restart ids. Call between capture sessions., Intersection-over-union of two xyxy boxes. Degenerate boxes score 0.0., Assigns stable `track_id`s to detections across calls. `expiry_s` should match…, Return copies of `detections` with `track_id` populated. (+26 more)

### Community 49 - "test_credentials.py"
Cohesion: 0.11
Nodes (30): has_api_key(), load_api_key(), load_env_file(), parse_env_file(), Credential loading for the sidecar. `Settings` is persisted to…, Parse KEY=value lines. Blank lines, `#` comments, a leading `export `, and…, Read and parse an env file. A missing or unreadable file yields {}., The Roboflow API key, or None if unset. The real environment wins over the file… (+22 more)

### Community 50 - "🔌 SCANnCART – Pluggable Detector Backends"
Cohesion: 0.07
Nodes (28): 0. Phase 0 findings — the workflow, as it actually is, 10. Testing, 11. Development plan, 12. Open questions, 13. Checklist, 1. Why, 2. Non-goals, 3. The seam (+20 more)

### Community 51 - "ApiClient"
Cohesion: 0.13
Nodes (7): SettingsDeps, ApiClient, AdminPanel(), AdminPanelProps, describeGpu(), baseSettings(), makeDeps()

### Community 52 - "AdminPanel.tsx"
Cohesion: 0.23
Nodes (14): SettingsPayload, ALLOWED_BACKENDS, ALLOWED_DEVICES, ALLOWED_MODELS, BACKEND_HINTS, BACKEND_LABELS, EXPERIMENTAL_MODELS, FieldGroup (+6 more)

### Community 53 - "🧠 SCANnCART – Custom Model Training Guide"
Cohesion: 0.13
Nodes (15): 1. Project Type — pick **Object Detection**, 2. Dataset Size — how many images for 10 classes, 3. Variation Beats Volume, 4. Two Traps to Avoid, 5. Capture Workflow, 6. Training, 7. Integrating the Trained Weights, 8. Quick Checklist (+7 more)

### Community 54 - "SCANnCART — System Architecture (Low Fidelity)"
Cohesion: 0.18
Nodes (11): 1. Context — what the system is, 2. Containers — the two processes, 3.1 The AI model, low fidelity, 3. Sidecar internals — the processing chain, 4. Runtime flow — one capture session, 5. Interfaces — the contract surface, 6. Data model — low fidelity, 7. Configuration & lifecycle (+3 more)

### Community 55 - "driver.mjs"
Cohesion: 0.20
Nodes (6): APP_DIR, { _electron: electron }, electronBin, REPO_ROOT, require, SKILL_DIR

### Community 56 - "Run the SCANnCART desktop app"
Cohesion: 0.25
Nodes (7): Build, Gotchas (all actually hit), Prerequisites, Run (agent path), Run (human path), Run the SCANnCART desktop app, UI handles

### Community 58 - "useSidecarSettings.test.tsx"
Cohesion: 0.50
Nodes (4): errorMessage(), baseSettings(), makeDeps(), useSidecarSettings()

### Community 59 - "arch_doc_check.py"
Cohesion: 0.67
Nodes (3): main(), PostToolUse hook: flag when a change likely invalidates the architecture docs.…, relevant()

## Knowledge Gaps
- **269 isolated node(s):** `SKILL_DIR`, `REPO_ROOT`, `APP_DIR`, `require`, `{ _electron: electron }` (+264 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `IouTracker` connect `IouTracker` to `test_roboflow_detector.py`, `main.py`, `test_detector_api.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `Detection` connect `main.py` to `test_settings_api.py`, `RoboflowRemoteDetector`, `IouTracker`, `test_detector_api.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `Settings` connect `test_detector_backends.py` to `test_settings_api.py`, `main.py`, `test_detector_api.py`, `CameraCapture`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Settings` (e.g. with `AppState` and `backend_url()`) actually correct?**
  _`Settings` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Detection` (e.g. with `Detector` and `RoboflowRemoteDetector`) actually correct?**
  _`Detection` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `build_app()` (e.g. with `HardwareInfo` and `RoboflowAuthError`) actually correct?**
  _`build_app()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `SKILL_DIR`, `REPO_ROOT`, `APP_DIR` to the rest of the system?**
  _269 weakly-connected nodes found - possible documentation gaps or missing edges._