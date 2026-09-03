# Native Offline Detection Optimizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `native` backend — the only one satisfying the PRD's offline promise — observable, reproducibly GPU-capable, and safe for a locally trained `.pt` fast path.

**Architecture:** The ONNX session is built exactly once per `YoloDetector`, with the requested device, and is released deterministically on capture stop (`close()`). `POST /api/detector/probe` for `native` stops being a filesystem check and becomes a real build + warmup + measured inference that reports latency, class names and the actual execution provider. The GPU onnxruntime build gets its own pinned requirements file so fresh installs cannot silently fall back to CPU, and `compute_warnings()` surfaces that fallback in the UI. `resolve_resize_mode("auto")` becomes format-aware (custom `.onnx` → stretch, custom `.pt` → letterbox) so a trained `.pt` gets the preprocessing it was trained with.

**Tech Stack:** Python 3.12, ultralytics 8.4.x, onnxruntime/onnxruntime-gpu 1.22.x, FastAPI/Pydantic v2, pytest — sidecar. React 19 + TypeScript, Vitest — desktop.

**Spec:** `docs/superpowers/specs/2026-09-04-native-offline-detection-optimizations-design.md`

## Global Constraints

- **Sidecar tests never touch hardware.** No camera, GPU, ultralytics, or network. `YoloDetector` behaviour is tested through the existing `model_factory` injection seam with fake model/predictor/backend objects.
- **The probe must never trigger a download.** A missing weights file short-circuits with the old "will download on first start" answer.
- **`close()` must never raise** and must be idempotent — `_teardown_capture` calls it on every stop, including error paths.
- **The settings/probe contract is hand-mirrored.** `sidecar/app/schemas.py` ↔ `desktop/src/renderer/src/lib/api.ts`. A field in one without the other is a silent type mismatch, not an error.
- **No new dependencies** in the sidecar's default install — onnxruntime stays in `requirements.txt` (CPU); the GPU build is opt-in via `requirements-cuda.txt` and must not be listed in both.
- **`compute_warnings()` stays pure** — the onnxruntime import is lazy and failure means "unavailable", never a raise.

## File Structure

**Sidecar — create:**
- `sidecar/requirements-cuda.txt` — pinned `onnxruntime-gpu>=1.22,<1.23` with the CUDA-version pairing rule

**Sidecar — modify:**
- `sidecar/app/inference.py` — lazy `names`, `provider` property, `close()` on `YoloDetector`
- `sidecar/app/schemas.py` — `DetectorProbeResponse.provider`
- `sidecar/app/main.py` — native branch of `POST /api/detector/probe` builds + warms + measures
- `sidecar/app/settings_store.py` — `_cuda_provider_available()`, CPU-fallback warning, format-aware `resolve_resize_mode`, custom-`.pt`-with-stretch warning
- `sidecar/tests/test_inference.py`, `test_detector_api.py`, `test_settings_store.py`

**Desktop — modify:**
- `desktop/src/renderer/src/lib/api.ts` — `DetectorProbeResponse.provider`
- `desktop/src/renderer/src/lib/settingsFields.ts` — native backend hint mentions onnxruntime-gpu
- `desktop/src/renderer/src/test/fakes.ts` — fake probe returns `provider`

**Docs — modify:**
- `docs/MODEL_TRAINING.md` §7 — custom weights are first-class; format-aware `resize_mode`; `.pt` GPU path

---

## Task 1: One session, released on stop (`YoloDetector`)

**Files:** `sidecar/app/inference.py`, `sidecar/tests/test_inference.py`

**Why:** `self.names = self._model.names` in the constructor makes ultralytics build a full predictor + ONNX session with the *default* device (verified: `YOLO.names` sets up `AutoBackend` for export formats), then the first `infer()` builds a second session with the real device. Measured 16.3 s cold warmup with two session builds. And `_teardown_capture`'s `_release(detector, "close")` no-ops for native, so VRAM lives until GC.

- [x] **Step 1: Write the failing tests** — `test_names_are_lazy_until_the_first_infer`, `test_provider_is_none_until_the_session_exists`, `test_provider_falls_back_to_the_torch_device`, `test_close_nulls_the_wrapper_references_and_is_idempotent`, `test_close_nulls_the_backend_session`
- [x] **Step 2: Implement** — `self.names = {}` in `__init__` (never touching `self._model.names`); populate from `r.names` on first `infer`; `provider` property reads `predictor.model.session.get_providers()[0]` (ORT) or `predictor.model.device` (torch), None until the session exists; `close()` nulls `backend.session/model/net`, `predictor.model`, `model.predictor/model`, drops `self._model`, and calls `torch.cuda.empty_cache()` on CUDA — all in `try/except`
- [x] **Step 3: Verify** — `cd sidecar && python -m pytest tests/test_inference.py -v`

## Task 2: The native probe becomes real

**Files:** `sidecar/app/schemas.py`, `sidecar/app/main.py`, `sidecar/tests/test_detector_api.py`

**Why:** The remote branch already builds the detector, warms it and reports latency + classes. Native only did `os.path.exists`. A corrupt model or silent CPU fallback therefore surfaced only as a failed or slow capture start.

- [x] **Step 1: Schema** — `DetectorProbeResponse.provider: str | None = None` (native-only, documented)
- [x] **Step 2: Route** — keep the missing-file short-circuit ("will download on first start", never probes); when present, run `state.detector_factory` in the threadpool, one warmup `infer` then one measured `infer` on a 64×64 synthetic frame; report `latency_ms`, sorted `class_names`, `provider`; `finally: close()`; any exception → `reachable: false` with the message
- [x] **Step 3: Tests** — `test_probe_native_builds_and_measures_the_detector` (latency, classes, provider, exactly 2 infer calls, closed), `test_probe_native_missing_weights_short_circuits_without_building` (factory never called), `test_probe_native_load_failure_is_unreachable`, `test_probe_native_load_failure_still_closes_the_detector`; extend `FakeDetector` with `provider` + `infer_calls`; patch `os.path.exists` to keep the tests hermetic (the real `models/` file is gitignored)
- [x] **Step 4: Verify** — `cd sidecar && python -m pytest tests/test_detector_api.py -v`

## Task 3: Desktop mirror

**Files:** `desktop/src/renderer/src/lib/api.ts`, `desktop/src/renderer/src/test/fakes.ts`

- [x] `DetectorProbeResponse.provider: string | null` with a comment naming what it answers ("am I really on the GPU?")
- [x] `fakes.ts` probe fake returns `provider: 'CPUExecutionProvider'`
- [x] **Verify** — `npm run typecheck` and `npm test`

## Task 4: Reproducible GPU runtime (Phase B)

**Files:** `sidecar/requirements-cuda.txt` (create), `sidecar/app/settings_store.py`, `sidecar/tests/test_settings_store.py`, `desktop/src/renderer/src/lib/settingsFields.ts`

**Why:** `requirements.txt` declares `onnxruntime>=1.20`, which pip resolves to the CPU wheel. The working venv's `onnxruntime-gpu 1.22.0` (CUDA 12, matching torch 2.6.0+cu124 — verified running `CUDAExecutionProvider` on the grocery ONNX, ~62 ms vs ~300–500 ms CPU) is unreproducible; a fresh `make install` silently regresses.

- [x] **Step 1: Requirements file** — `onnxruntime-gpu>=1.22,<1.23`; document the pairing rule (1.29+ wants CUDA 13, which torch does not ship → "no data transfer registered") and that both wheels share the import name
- [x] **Step 2: Warning** — `_cuda_provider_available()` (lazy import, failure = unavailable); in `compute_warnings()`: native + `.onnx` + `resolve_device(settings.device) == "cuda"` + no CUDA EP → warning naming `requirements-cuda.txt`
- [x] **Step 3: Tests** — warn when EP missing, not when present, never for `device="cpu"`, never for a `.pt`; monkeypatch `_cuda_provider_available`
- [x] **Step 4: Desktop hint** — append the onnxruntime-gpu sentence to `BACKEND_HINTS["native"]`

## Task 5: Format-aware `resize_mode` + refreshed training docs (Phase C)

**Files:** `sidecar/app/settings_store.py`, `sidecar/tests/test_settings_store.py`, `docs/MODEL_TRAINING.md`

**Why:** `resolve_resize_mode("auto")` resolved to stretch for *any* custom model, but a locally trained `.pt` (the only way to obtain one — ultralytics cannot export ONNX→`.pt`, verified) is letterbox-trained. Silently wrong preprocessing.

- [x] **Step 1: Resolution** — `auto` → `stretch` only for custom `.onnx`; everything else → `letterbox`
- [x] **Step 2: Warning** — custom `.pt` + explicit `stretch` → warning; never for `.onnx` + stretch or stock weights
- [x] **Step 3: Tests** — `test_auto_resolves_custom_onnx_to_stretch`, `test_auto_resolves_custom_pt_to_letterbox`, `test_auto_resolves_stock_weights_to_letterbox`, `test_explicit_modes_win_over_auto`, plus the three warning cases
- [x] **Step 4: Docs** — `MODEL_TRAINING.md §7`: drop the stale "custom weights will not load as-is / whitelist" warning; state that any `models/*.pt` is selectable with no edits in either codebase; document the format-aware `resize_mode` and the `.pt`-on-torch GPU path

## Task 6: Full-suite verification

- [x] `cd sidecar && python -m pytest -q` — whole sidecar suite hermetic (no camera/GPU/network)
- [x] `cd desktop && npm run typecheck`
- [x] `cd desktop && npm test`
- [x] **Commit** the spec, plan, sidecar, desktop and docs changes together
- [x] **Manual, on hardware:** Test Connection in the Admin Panel reported "models/scanncart-grocery.onnx loaded. Inference on CUDAExecutionProvider. — 60.8 ms — 7 classes" (53–61 ms across runs); start/stop ran 3 clean cycles against the real StreamCam + ONNX on CUDA, then the machine's known StreamCam freeze surfaced as a clean WS error and teardown recovered — the probe's warm session also kills the ~16 s cold first-frame hit (capture starts with the session already built)

## Test Checklist

- [x] `YoloDetector.names` is `{}` until first `infer`, then populated from the result
- [x] `provider` is `None` pre-session, ORT provider / torch device afterwards
- [x] `close()` nulls the wrapper + backend references, drops `_model`, idempotent, never raises
- [x] Native probe: builds via the factory, warmup + measured infer (2 calls), latency/classes/provider reported, detector closed on both success and failure
- [x] Native probe: missing file short-circuits, factory never called, no download
- [x] `DetectorProbeResponse.provider` mirrors into `api.ts` and the desktop fakes typecheck
- [x] `_cuda_provider_available` warning fires only for native + `.onnx` + resolved-cuda + missing CUDA EP
- [x] `resolve_resize_mode`: custom `.onnx` → stretch, custom `.pt` → letterbox, stock → letterbox, explicit wins
- [x] Custom `.pt` + explicit stretch warns; `.onnx` + stretch and stock + stretch do not