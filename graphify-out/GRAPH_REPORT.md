# Graph Report - scanncart  (2026-08-29)

## Corpus Check
- 111 files · ~72,240 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1336 nodes · 2370 edges · 102 communities (76 shown, 26 thin omitted)
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 508 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `719fc113`
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
- test_cameras.py
- _make_client
- Detection
- test_detector_backends.py
- TrackingSource
- main.py
- test_capture_teardown.py
- _valid_field
- test_main.py
- case-tests.mjs
- settings.py
- probe_hardware
- SettingsUpdateRequest
- backend-check.mjs
- capture-debug.mjs
- FakeSource
- run.py
- test_schemas.py
- settings_store.py
- SCANnCART

## God Nodes (most connected - your core abstractions)
1. `Settings` - 68 edges
2. `AppState` - 61 edges
3. `Detection` - 43 edges
4. `IouTracker` - 39 edges
5. `WSManager` - 33 edges
6. `make()` - 33 edges
7. `Pipeline` - 32 edges
8. `frame()` - 26 edges
9. `TrackingSource` - 24 edges
10. `make_client()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Preset` --uses--> `HardwareInfo`  [INFERRED]
  sidecar/app/presets.py → sidecar/app/hardware.py
- `test_detection_allows_null_track_id()` --calls--> `Detection`  [INFERRED]
  sidecar/tests/test_schemas.py → sidecar/app/schemas.py
- `test_status_message_default_detail()` --calls--> `StatusMessage`  [INFERRED]
  sidecar/tests/test_schemas.py → sidecar/app/schemas.py
- `test_health_response_fields()` --calls--> `HealthResponse`  [INFERRED]
  sidecar/tests/test_schemas.py → sidecar/app/schemas.py
- `test_settings_defaults()` --calls--> `Settings`  [INFERRED]
  sidecar/tests/test_settings.py → sidecar/app/settings.py

## Import Cycles
- None detected.

## Communities (102 total, 26 thin omitted)

### Community 0 - "api.ts"
Cohesion: 0.18
Nodes (15): errorMessage(), SidecarSettings, CameraInfo, CamerasResponse, createApiClient(), DetectorProbeResponse, HealthResponse, LogEvent (+7 more)

### Community 1 - "devDependencies"
Cohesion: 0.29
Nodes (7): devDependencies, electron, electron-builder, @testing-library/jest-dom, electron, electron-builder, @testing-library/jest-dom

### Community 2 - "main.py"
Cohesion: 0.12
Nodes (14): encode_preview_jpeg(), Pipeline, Detection, Exception, ndarray, Ask the loop to finish without waiting for it.          Separate from `stop()`, _frame(), Always returns the same latest frame. (+6 more)

### Community 3 - "useSidecarStream.ts"
Cohesion: 0.08
Nodes (24): LoggedItem, SidecarStream, StreamDeps, useSidecarStream(), LogsResponse, boxToPercent(), boxToPixels(), LabeledRect (+16 more)

### Community 4 - "test_detector_api.py"
Cohesion: 0.13
Nodes (25): backend_url(), _default_detector_factory(), main(), client_for(), FakeDetector, _no_real_key(), Phase 4 wiring: detector_factory selection, /api/detector/probe, and the error m, Never read the developer's real sidecar/.env during tests. (+17 more)

### Community 5 - "CameraCapture"
Cohesion: 0.07
Nodes (20): CameraCapture, _default_capture(), FakeFrameSource, FrameSource, LatestFrameBuffer, ndarray, Protocol, Test double that yields the provided frames in order, then None. (+12 more)

### Community 6 - "test_detector_backends.py"
Cohesion: 0.13
Nodes (32): Settings, compute_warnings(), load_settings(), Load settings from disk, overlaying only known/valid fields onto the     hardco, Atomic write: write to a temp file then os.replace() into place, so a     crash, Soft warnings surfaced in SettingsResponse. `api_key_present` is passed     in, save_settings(), test_backend_fields_round_trip() (+24 more)

### Community 7 - "HardwareInfo"
Cohesion: 0.21
Nodes (11): Preset, HardwareInfo, recommend_preset(), test_recommend_preset_high_end_for_realistic_4gb_gpu(), test_recommend_preset_high_end_for_strong_gpu(), test_recommend_preset_ignores_genuinely_weak_gpu(), test_recommend_preset_ignores_weak_gpu(), test_recommend_preset_low_end_fallback() (+3 more)

### Community 8 - "scripts"
Cohesion: 0.06
Nodes (30): author, dependencies, @electron-toolkit/preload, @electron-toolkit/utils, description, homepage, main, name (+22 more)

### Community 9 - "test_settings_api.py"
Cohesion: 0.14
Nodes (20): _fake_hardware(), _make_client(), _StubDetector, _StubSource, test_apply_preset_while_idle_applies_settings(), test_apply_preset_while_running_is_rejected(), test_apply_unknown_preset_returns_404(), test_get_settings_returns_current_values_and_field_classification() (+12 more)

### Community 10 - "RoboflowRemoteDetector"
Cohesion: 0.11
Nodes (17): _clamp01(), Detector, normalize_detections(), Detection, ndarray, Protocol, Runs detection through a Roboflow Workflow — cloud or self-hosted.      Satisf, RoboflowRemoteDetector (+9 more)

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
Nodes (67): _default_client_factory(), find_predictions(), first_result(), _httpx(), _looks_like_predictions(), Any, Exception, HTTP client for Roboflow Workflows.  Talks to either Roboflow's serverless endpo (+59 more)

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
Cohesion: 0.18
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
Nodes (43): find_image_size(), The `image: {width, height}` block a workflow reports alongside its     predicti, FakeClient, frame(), make(), Phase 4: RoboflowRemoteDetector, wiring, and /api/detector/probe.  The REAL_RESP, Server echoes the dims it saw; trust those over our own., Adding a tracking block upstream later needs no code change here. (+35 more)

### Community 33 - "SCANnCART Sidecar (Phase 1)"
Cohesion: 0.22
Nodes (9): GPU (NVIDIA/CUDA) setup, Roboflow credentials (only for the API backends), Run, Running `local_api` without Docker, SCANnCART Sidecar, Setup, Tests, Verify manually (+1 more)

### Community 34 - "🚀 SCANnCART – Deployment & Future Phases"
Cohesion: 0.33
Nodes (6): 1. Deployment Goals, 2. Centralized Architecture (Cart → Server), 3. Future Phases (Out-of-Scope for Prototype), 4. Deployment Non-Functional Targets, 5. Deployment Success Metrics, 🚀 SCANnCART – Deployment & Future Phases

### Community 35 - "README.md"
Cohesion: 0.40
Nodes (5): Architecture, Run the full app (manual — needs the StreamCam), SCANnCART Desktop (Phase 2 — Electron shell + Live View), Setup, Test (headless — no display, camera, or model)

### Community 36 - "IouTracker"
Cohesion: 0.07
Nodes (43): Box, iou(), IouTracker, Detection, Local object tracking for detector backends that don't supply track ids.  `Yol, Ensure this id is never minted locally.          Only matters when a response, Forget every track and restart ids. Call between capture sessions., Intersection-over-union of two xyxy boxes. Degenerate boxes score 0.0. (+35 more)

### Community 49 - "test_credentials.py"
Cohesion: 0.09
Nodes (34): has_api_key(), load_api_key(), load_env_file(), parse_env_file(), Credential loading for the sidecar.  `Settings` is persisted to data/settings.js, Parse KEY=value lines. Blank lines, `#` comments, a leading `export `,     and m, Read and parse an env file. A missing or unreadable file yields {}., The Roboflow API key, or None if unset.      The real environment wins over the (+26 more)

### Community 50 - "🔌 SCANnCART – Pluggable Detector Backends"
Cohesion: 0.06
Nodes (34): 0. Phase 0 findings — the workflow, as it actually is, 10. Testing, 11. Development plan, 12. Open questions, 13. Checklist, 1. Why, 2. Non-goals, 3. The seam (+26 more)

### Community 52 - "AdminPanel.tsx"
Cohesion: 0.20
Nodes (18): useSidecarSettings(), SettingsPayload, ALLOWED_BACKENDS, ALLOWED_DEVICES, ALLOWED_MODELS, BACKEND_HINTS, BACKEND_LABELS, EXPERIMENTAL_MODELS (+10 more)

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

### Community 57 - "WSManager"
Cohesion: 0.18
Nodes (23): BaseModel, AppState, build_app(), WSManager, ApplyPresetRequest, CameraInfo, CamerasResponse, DetectorProbeResponse (+15 more)

### Community 58 - "useSidecarSettings.test.tsx"
Cohesion: 0.32
Nodes (6): SettingsDeps, baseSettings(), makeDeps(), AdminPanelProps, baseSettings(), makeDeps()

### Community 59 - "arch_doc_check.py"
Cohesion: 0.67
Nodes (3): main(), PostToolUse hook: flag when a change likely invalidates the architecture docs., relevant()

### Community 82 - "test_cameras.py"
Cohesion: 0.09
Nodes (33): CameraDevice, list_cameras(), list_device_names(), probe_index(), Camera enumeration — turns OpenCV's bare device indices into named devices.  `ca, Every camera index that opens, paired with a name where one is known.      Stops, Best-effort camera names in Windows enumeration order. Returns [] on any     fai, The (width, height) a device opens at, or None if the index is unusable.      As (+25 more)

### Community 83 - "_make_client"
Cohesion: 0.09
Nodes (21): _fake_hardware(), _FakeDetector, _make_client(), Detection, HardwareInfo, ndarray, Full capture lifecycle through the REST + WebSocket interface., Start capture → receive a WebSocket frame → verify it has a         non-empty J (+13 more)

### Community 84 - "Detection"
Cohesion: 0.12
Nodes (17): Detection, _client(), _StubDetector, _StubSource, test_logs_empty_before_any_session(), test_logs_report_current_session_events_after_a_run(), FakeClock, FakeStore (+9 more)

### Community 85 - "test_detector_backends.py"
Cohesion: 0.13
Nodes (20): Settings-layer coverage for the pluggable detector backends.  See docs/DETECTOR_, Only presence is ever exposed; the value must not have a field at all., Reproduced live in Phase 0: a 3250 ms round trip expired a stationary     item u, All eight are baked into the detector at capture-start, so none can be     hot-s, test_cloud_backend_warns_about_cost_and_offline_guarantee(), test_cloud_backend_warns_when_url_is_not_https(), test_cloud_backend_with_https_has_no_scheme_warning(), test_every_backend_field_requires_restart() (+12 more)

### Community 86 - "TrackingSource"
Cohesion: 0.14
Nodes (13): Event, FakeFrameSource, Could not connect. Usually the local inference server isn't running., RoboflowUnavailable, ExplodingDetector, Frame source that records whether it was released., Raises on the Nth infer, like a remote backend losing its server., state() (+5 more)

### Community 87 - "main.py"
Cohesion: 0.12
Nodes (11): FastAPI, HardwareInfo, _list_display_adapters(), Best-effort enumeration of display adapter names via Windows     Win32_VideoCont, Call the first of `names` that exists. Frame sources expose `release()`     and, Return to idle and free every resource capture acquired.      Two callers race, _release(), _teardown_capture() (+3 more)

### Community 88 - "test_capture_teardown.py"
Cohesion: 0.22
Nodes (15): OkDetector, Capture failure and teardown paths.  These cover four bugs that were unreachab, Both callers tear down; the loser must no-op, not crash.      The pipeline threa, A slow camera open must not freeze the rest of the sidecar.      A Logitech Stre, test_a_detector_that_fails_at_start_releases_the_camera(), test_a_mid_capture_failure_frees_the_camera_and_the_detector(), test_a_mid_capture_failure_returns_the_sidecar_to_idle(), test_start_does_not_block_the_event_loop_on_a_slow_camera() (+7 more)

### Community 89 - "_valid_field"
Cohesion: 0.12
Nodes (16): Any, _valid_field(), test_blank_identifiers_rejected(), test_infer_size_in_range_accepted(), test_infer_size_out_of_range_rejected(), test_invalid_backends_rejected(), test_retries_in_range_accepted(), test_retries_out_of_range_rejected() (+8 more)

### Community 90 - "test_main.py"
Cohesion: 0.21
Nodes (9): _fake_hardware(), _make_client(), _StubDetector, _StubSource, test_cross_origin_requests_get_cors_headers(), test_health_reports_idle_and_model(), test_start_then_stop_transitions_state(), test_system_info_reports_accelerator() (+1 more)

### Community 91 - "case-tests.mjs"
Cohesion: 0.18
Nodes (12): api(), APP_DIR, CASE, { _electron: electron }, electronBin, health(), killInferenceServer(), log() (+4 more)

### Community 92 - "settings.py"
Cohesion: 0.29
Nodes (10): SettingsResponse, _apply_settings_patch(), _settings_response(), resolve_device(), _fake_torch(), test_resolve_device_auto_falls_back_to_cpu(), test_resolve_device_cpu_always_passthrough(), test_resolve_device_cuda_and_auto_use_gpu_when_available() (+2 more)

### Community 93 - "probe_hardware"
Cohesion: 0.35
Nodes (11): probe_hardware(), _fake_torch_no_cuda(), test_accelerator_cpu_when_lister_raises(), test_accelerator_cpu_when_no_adapters(), test_accelerator_cpu_when_only_microsoft_basic_adapter(), test_accelerator_cuda_when_gpu_available(), test_accelerator_integrated_for_non_cuda_adapter(), test_probe_hardware_falls_back_when_torch_missing() (+3 more)

### Community 94 - "SettingsUpdateRequest"
Cohesion: 0.18
Nodes (6): SettingsUpdateRequest, test_update_request_accepts_a_valid_backend(), test_update_request_omitting_backend_fields_is_fine(), test_update_request_rejects_a_schemeless_url(), test_update_request_rejects_an_unknown_backend(), test_update_request_rejects_blank_workspace()

### Community 95 - "backend-check.mjs"
Cohesion: 0.25
Nodes (6): APP_DIR, { _electron: electron }, electronBin, REPO_ROOT, require, SKILL_DIR

### Community 96 - "capture-debug.mjs"
Cohesion: 0.25
Nodes (6): APP_DIR, { _electron: electron }, electronBin, REPO_ROOT, require, SKILL_DIR

### Community 97 - "FakeSource"
Cohesion: 0.32
Nodes (6): FakeSource, _raising_factory(), Otherwise a retry hits a camera already held open by the failed attempt., test_capture_start_maps_roboflow_errors(), test_failed_start_closes_the_camera(), test_failed_start_leaves_state_idle()

### Community 98 - "run.py"
Cohesion: 0.47
Nodes (3): pick_port(), test_pick_port_falls_back_when_taken(), test_pick_port_returns_preferred_when_free()

### Community 99 - "test_schemas.py"
Cohesion: 0.33
Nodes (4): test_detection_allows_null_track_id(), test_frame_message_serializes(), test_health_response_fields(), test_status_message_default_detail()

### Community 100 - "settings_store.py"
Cohesion: 0.40
Nodes (4): min_track_expiry_s(), The lowest track_expiry_s that is safe for `backend`., test_local_api_floor_is_lower_than_the_cloud_floor(), test_unknown_backend_falls_back_to_the_conservative_floor()

### Community 101 - "SCANnCART"
Cohesion: 0.50
Nodes (4): Architecture, For agents, Quick start, SCANnCART

## Knowledge Gaps
- **295 isolated node(s):** `SKILL_DIR`, `REPO_ROOT`, `APP_DIR`, `require`, `{ _electron: electron }` (+290 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AppState` connect `WSManager` to `FakeSource`, `main.py`, `IouTracker`, `CameraCapture`, `test_detector_backends.py`, `test_detector_api.py`, `test_settings_api.py`, `RoboflowRemoteDetector`, `LoggingStore`, `test_roboflow_client.py`, `test_cameras.py`, `_make_client`, `Detection`, `TrackingSource`, `main.py`, `test_capture_teardown.py`, `test_main.py`, `SettingsUpdateRequest`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `Settings` connect `test_detector_backends.py` to `FakeSource`, `main.py`, `test_detector_api.py`, `CameraCapture`, `test_settings_api.py`, `_make_client`, `Detection`, `test_detector_backends.py`, `TrackingSource`, `main.py`, `test_capture_teardown.py`, `WSManager`, `test_main.py`, `settings.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `WSManager` connect `WSManager` to `main.py`, `IouTracker`, `CameraCapture`, `test_detector_backends.py`, `RoboflowRemoteDetector`, `LoggingStore`, `test_roboflow_client.py`, `test_cameras.py`, `TrackingSource`, `main.py`, `SettingsUpdateRequest`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 59 inferred relationships involving `Settings` (e.g. with `AppState` and `WSManager`) actually correct?**
  _`Settings` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `AppState` (e.g. with `CameraCapture` and `CameraDevice`) actually correct?**
  _`AppState` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `Detection` (e.g. with `Detector` and `RoboflowRemoteDetector`) actually correct?**
  _`Detection` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `IouTracker` (e.g. with `AppState` and `_default_detector_factory()`) actually correct?**
  _`IouTracker` has 30 INFERRED edges - model-reasoned connections that need verification._