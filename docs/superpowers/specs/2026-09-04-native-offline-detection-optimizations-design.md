# SCANnCART — Native Offline Detection Optimizations — Design

**Date:** 2026-09-04
**Status:** Implemented — Phases A–C landed the same day; see "Verification (real app)" below
**Depends on:** `sidecar/app/inference.py`, `sidecar/app/main.py`,
`sidecar/app/settings_store.py`, `sidecar/app/settings.py`, `sidecar/requirements.txt`,
`desktop/src/renderer/src/lib/settingsFields.ts`, `docs/MODEL_TRAINING.md`.

## Problem

The `native` backend — the only one that satisfies the PRD's offline promise — has four
concrete defects and one missing fast path, all found by reading the code and verifying
against the live sidecar venv (ultralytics 8.4.92, torch 2.6.0+cu124, onnxruntime-gpu
1.22.0, CUDA available):

1. **A fresh install silently loses the GPU.** `requirements.txt` declares
   `onnxruntime>=1.20`, which pip resolves to the **CPU** wheel. The working venv's GPU
   build (`onnxruntime-gpu 1.22.0`) is not reproducible; anyone re-running `make install`
   gets the CPU runtime while `device: "auto"` still resolves to `cuda`. Ultralytics then
   logs "CUDA requested but CUDAExecutionProvider not available. Using CPU..." — a line
   nobody in the app ever sees. Measured on this machine, that is **~62 ms/frame on CUDA
   vs ~300–500 ms on CPU** for the grocery ONNX (2026-09-04, synthetic frames).

2. **The native detector probe checks nothing.** `POST /api/detector/probe` for `native`
   only does `os.path.exists(model)`. The remote branch builds the detector, runs a
   warmup inference and reports latency + class names; native gets none of that. So a
   corrupt `.onnx`, a model that fails to load, or a silent CPU fallback all surface
   only as a failed or slow capture start.

3. **The ONNX session is built twice at startup.** `YoloDetector.__init__` sets
   `self.names = self._model.names`; for export formats ultralytics' `names` property
   sets up a full predictor (verified: `ultralytics/engine/model.py` builds
   `AutoBackend` when `self.model` is not a torch module), so the constructor builds one
   session with the **default** device and the first `infer` builds a second with the
   requested device. Each build costs seconds — measured 16.3 s warmup on a cold process
   — and when the devices differ (auto→cuda vs cpu) the first session is pure waste.

4. **`YoloDetector` has no `close()`.** `_teardown_capture` calls
   `_release(detector, "close")`, which is a no-op for native. On CUDA the loaded model
   keeps VRAM until Python GC after every stop/start cycle.

5. **The documented fast path — a `.pt` on CUDA — is blocked and mis-guided.**
   `docs/DETECTOR_BACKENDS.md` says "the way to actually go faster is a `.pt` on CUDA",
   and the numbers agree (docs: 38 ms isolated / 25 ms in-app for yolo11n `.pt` + CUDA vs
   51/91 ms for the custom ONNX on CPU). But ultralytics **cannot export ONNX → `.pt`**
   (verified in `engine/exporter.py`: PyTorch is the source format, "Model is already in
   PyTorch format"), so the only route is retraining from the free dataset export — and
   `docs/MODEL_TRAINING.md §7` is stale there. It still claims `ALLOWED_MODELS` is a
   hardcoded whitelist that rejects custom paths; `is_custom_model()` has accepted any
   `models/*.pt`/`models/*.onnx` for a while. Worse, `resolve_resize_mode("auto", …)`
   resolves to **stretch** for *any* custom model, but a locally-trained `.pt` is
   letterbox-trained — dropping one in with `resize_mode: "auto"` silently mismatches the
   preprocessing.

## Design

### Phase A — Make the native path observable: real probe, one session, released on stop

#### A1. Upgrade `POST /api/detector/probe` for `native`

Keep the cheap `os.path.exists` short-circuit (report `reachable: true` with
"not on disk; ultralytics will download it on first start" as today — the probe must
never trigger a download or a network hit). When the file is present, do what the remote
branch already does: build the detector through `state.detector_factory` (the same code
path capture start uses), run one warmup inference on a synthetic frame, then one
measured call, and return:

- `latency_ms` — warm inference latency.
- `class_names` — `sorted(str(v) for v in detector.names.values())`, i.e. the 7 SKUs.
- `detail` — including the execution provider/device actually in use, so an operator can
  see "onnxruntime CUDA" vs "onnxruntime CPU" (or "torch cuda:0") before starting.

Effects, mirroring the existing `local_api` warm-up recommendation in
`docs/DETECTOR_BACKENDS.md §4`: the session is warm before capture starts (killing the
~16 s cold first-frame hit), a model that fails to load surfaces in the Admin Panel
instead of as a failed capture, and the silent CPU fallback becomes visible.

Add `provider` to `YoloDetector` — a stable string read from the autobackend
(`self._model.predictor.model.session.get_providers()[0]` for ORT, the resolved
device string otherwise), guarded for a not-yet-built session. The probe reports it;
`DetectorProbeResponse` gains an optional `provider: str | None` field (hand-kept mirror
in `desktop/src/renderer/src/lib/api.ts`).

#### A2. One session: stop reading `self._model.names` in the constructor

`YoloDetector.__init__` must not touch `self._model.names`. Instead:

- Initialize `self.names: dict = {}`.
- Populate it on the first `infer` from the result (`r.names`), exactly as
  `RoboflowRemoteDetector` already does — `normalize_detections` takes `names` per call
  anyway.

That deletes the redundant constructor-time session build; exactly one ONNX session is
created, with the requested device. (The `Detector` protocol only requires `names` to be
readable; nothing reads it before capture starts except the probe, which runs an
inference first anyway.)

#### A3. `YoloDetector.close()`

Add `close()` that drops the model and predictor references and, when the device is CUDA,
calls `torch.cuda.empty_cache()` (soft; no-op on CPU). `_teardown_capture` already calls
`_release(detector, "close")` — today it no-ops for native, so this is all the plumbing
needs. Detectors keep `close()`; frame sources keep `release()`.

### Phase B — Make the GPU runtime reproducible

- New `sidecar/requirements-cuda.txt` pinning `onnxruntime-gpu==1.22.*` — CUDA 12.4,
  the same major CUDA torch 2.6.0+cu124 ships (verified working on this machine through
  `enable_onnx_cuda()`). Document the pairing rule in the file: the onnxruntime-gpu
  minor must match torch's bundled CUDA (1.29+ wants CUDA 13 and fails with "no data
  transfer registered"); never install both `onnxruntime` and `onnxruntime-gpu` (shared
  import name).
- `requirements.txt` keeps the CPU `onnxruntime>=1.20` as the portable default; the
  README's "ONNX on the GPU" note points at `requirements-cuda.txt`.
- `compute_warnings()` gains a check: when `detector_backend == "native"`, the model is
  `.onnx`, the resolved device is `cuda`, and `CUDAExecutionProvider` is not in
  `onnxruntime.get_available_providers()` → warn that inference will silently run on CPU
  and name the fix. Import onnxruntime lazily inside the check and guard `ImportError`
  (CPU-only environments). Mirror a matching hint line in
  `desktop/src/renderer/src/lib/settingsFields.ts` `BACKEND_HINTS["native"]`.
- Keep `enable_onnx_cuda()` (it worked in verification) but make it idempotent and have
  the comment name the tested pairing.

### Phase C — The fast path: a locally-trained `.pt` on CUDA

- **Docs only, plus one real fix.** Retraining from the dataset export link (in the
  cached `environment.json`, free on any Roboflow plan) is the only route to a `.pt` —
  verified ONNX→`.pt` is unsupported. Update `docs/MODEL_TRAINING.md §7`:
  - Fix the stale whitelist claim: any `models/*.pt` is already selectable via
    `is_custom_model()`; no `ALLOWED_MODELS` edit needed in either codebase.
  - Add a "train & drop in" runbook: notebook (`notebooks/train-yolo11-object-detection-on-custom-dataset.ipynb`)
    → `best.pt` → copy to `sidecar/models/scanncart-grocery.pt` → select in Admin Panel
    → `device: "auto"` (resolves to cuda). Expected ~38 ms isolated / ~25 ms in-app per
    the docs' yolo11n measurements, with the stretch/letterbox question settled below.
- **The preprocessing fix.** `resolve_resize_mode("auto", model)` currently returns
  `"stretch"` for any custom model under `models/`. A Roboflow-exported `.onnx` is
  stretch-trained — correct. A locally-trained `.pt` is letterbox-trained — currently
  mismatched. Change the resolution to be format-aware: `auto` → `"stretch"` for a custom
  `.onnx` (Roboflow export), `"letterbox"` for a custom `.pt` (ultralytics-native
  training, the only realistic way to obtain one — Roboflow `.pt` exports are Core-gated).
  Add a `compute_warnings()` note for a custom `.pt` with an explicit `resize_mode:
  "stretch"` (reminder it is only right for a Roboflow-exported `.pt`).
- **Non-goal here:** `half=True` and int8 quantization for the `.pt`/`.onnx` — noted as
  follow-ups in `docs/DEPLOYMENT.md` out-of-scope territory; do not add settings fields
  for them yet.

## Testing

Per repo convention — injected fakes, no camera, no GPU, no network. The new
`YoloDetector` behaviour is tested with the existing `model_factory` injection seam and a
fake model object (a real ultralytics load is never required).

| Test file | Covers |
|-----------|--------|
| `tests/test_inference.py` *(extend)* | `names` is empty until first `infer`, populated from the result; exactly one `infer` builds the session (fake model counts `track`/session calls); `close()` drops the reference and is idempotent; `provider` reflects the device for torch and ORT-style fakes |
| `tests/test_settings_store.py` *(extend)* | `resolve_resize_mode` auto: custom `.onnx` → stretch, custom `.pt` → letterbox, stock weights → letterbox; warning emitted for native + `.onnx` + `device=cuda` when `onnxruntime.get_available_providers()` lacks CUDA (monkeypatched), and not when it has it or device is cpu; custom `.pt` + explicit stretch warning |
| `tests/test_main.py` *(extend)* | `POST /api/detector/probe` native branch with an injected `detector_factory` fake: reports `latency_ms`, 7 class names and `provider`; missing file short-circuits with the "will download" detail and builds nothing; a factory that raises returns `reachable: false` with the message |
| `tests/test_settings_api.py` *(extend)* | `DetectorProbeResponse` accepts `provider`; nothing new leaks a secret |
| desktop `settingsFields`/`api` tests | `provider` field mirrors; backend hint copy renders |

The CPU-fallback warning path imports onnxruntime lazily inside the check and is guarded
for `ImportError`, so tests never need the package installed.

## Verification (real app, 2026-09-04)

- **Test Connection (Admin Panel):** "✓ models/scanncart-grocery.onnx loaded. Inference on
  **CUDAExecutionProvider**. — 60.8 ms — 7 classes" (reproduced at 53–61 ms across runs;
  also verified standalone over REST). The provider/classes/latency all come from the
  new probe path, so an operator sees the GPU is real *before* capture starts.
- **Start/stop capture:** 3 clean cycles against the real StreamCam + grocery ONNX on
  CUDA (start → frames → stop, no leaks across cycles). The 4th cycle hit this machine's
  known StreamCam freeze (`CvCapture_MSMF::grabFrame: can't grab frame`); the new
  pipeline error path surfaced it cleanly over WS — "Camera 1 stopped delivering frames
  for 3s (35 attempts)" — and teardown recovered without wedging the sidecar.
- **One session:** exactly one ONNX session build per `YoloDetector` construction (was
  two); the constructor no longer touches `model.names`.

## Measured (2026-09-04, this machine)

Ground truth for the claims above, taken with the repo's own `YoloDetector` against
`models/scanncart-grocery.onnx` on synthetic 1280×720 frames:

| Path | Result |
|------|--------|
| `device="cuda"`, warmup | 16.3 s first call (two session builds — see A2; cuDNN autotune) |
| `device="cuda"`, steady | 44–74 ms, avg **62 ms** — `Using ONNX Runtime 1.22.0 with CUDAExecutionProvider` |
| `device="cpu"`, steady | 281–574 ms across two clean processes (~400–500 ms typical) |
| Session builds per `YoloDetector` construction | **2** (constructor via `.names`, then first `infer`) — both CUDA here; first is wasted when the requested device differs |
| ONNX → `.pt` export | Unsupported by ultralytics (PyTorch is the source format) |

## Non-goals

- No new settings fields except the optional `provider` on the probe response — the
  existing `device`/`imgsz`/`resize_mode` knobs already cover tuning.
- No DirectML / TensorRT provider plumbing. Ultralytics' ONNX backend only auto-selects
  CUDA/CoreML; adding DirectML for non-NVIDIA GPUs means bypassing the backend, which is a
  bigger seam change than this spec wants. Noted as an open question.
- No int8 quantization or FP16 inference settings (see Phase C non-goal).
- No change to the remote backends, tracking, or the renderer's detection-dedup logic.
- The class-name display mapping (mixed `snake_case`/Title Case SKUs in the item log)
  remains open from `docs/DETECTOR_BACKENDS.md §12`; noted, not designed here.