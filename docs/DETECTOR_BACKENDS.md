# 🔌 SCANnCART – Pluggable Detector Backends

> Spec and development plan for running detection through **three interchangeable backends** —
> native weights, a self-hosted Roboflow server, or Roboflow's cloud API — selectable at runtime
> from the Admin Panel.
>
> Related: [PRD.md](./PRD.md) · [ARCHITECTURE.md](./ARCHITECTURE.md) · [MODEL_TRAINING.md](./MODEL_TRAINING.md) · [DEPLOYMENT.md](./DEPLOYMENT.md)

---

## 0. Phase 0 findings — the workflow, as it actually is

Fetched from `GET https://api.roboflow.com/{workspace}/workflows/{workflow_id}` and verified by
running a real dataset image through the live endpoint.

### Definition

```
inputs :  image  (InferenceImage)          — one input, no parameters
steps  :  model  (roboflow_core/inner_workflow@v1)
              workflow_id       : scanncart-grocery
              parameter_bindings: image    -> $inputs.image
                                  model_id -> yusri-caloyloy/scanncart-grocery-1-yolo11n-t1
outputs:  predictions  (JsonField, coordinates_system "own")
              selector: $steps.model.predictions
```

**The real model id is `yusri-caloyloy/scanncart-grocery-1-yolo11n-t1`** — not `scanncart-grocery/1`.
Project: object-detection, 1815 images, **7 classes**:

```
century_tuna_flakes_in_oil_155_grams   555 sardines 155grams
lucky_me_pancit_canton_calamansi_flavor   safeguard_pure_white_60g
silver_swan_sukang_puti_200ML   Bear Brand Fortified Powdered Milk 33g
Milo Chocolate Drink 22g Sachet
```

⚠️ Class names mix `snake_case` and `Title Case With Spaces`. They surface verbatim in the item log,
so a display-name mapping belongs somewhere in the renderer.

### Real response

```jsonc
{ "predictions": {                        // <- output name
    "image": { "width": 480, "height": 640 },   // dims of the image WE sent
    "predictions": [                            // <- the detections
      { "x": 229.0, "y": 416.0, "width": 456.0, "height": 444.0,
        "confidence": 0.9426, "class_id": 3,
        "class": "century_tuna_flakes_in_oil_155_grams",
        "detection_id": "7d58050c-…", "parent_id": "image" }
    ] } }
```

### What this settles

| Question | Answer | Consequence |
|----------|--------|-------------|
| Tracking block? | **No** | `IouTracker` is the **primary** path, not a fallback. `tracker_id` is absent from the response. |
| Any "Logic" beyond the model? | **No** — a single `inner_workflow` wrapper | Removes `local_api`'s main advantage over native. See revised recommendation below. |
| Coordinate convention | Centre `x`/`y` + `width`/`height`, in **transmitted-image** pixels | Normalize against `predictions.image`, not the local frame. |
| Class field name | **`class`** — not `class_name` | Parse `class` first, `class_name` as fallback. |
| Response envelope | Bare list → `{"predictions": {"image", "predictions"}}` | `find_predictions()` handles it; verified against this exact payload. |

### Revised recommendation

Because the workflow contains **no logic to preserve**, `local_api`'s edge over running the weights
directly is gone — it is now just an extra process and an HTTP hop in front of the same model.
**`native` is the target more strongly than before**, with `cloud_api` retained for demos and
offline evaluation. `local_api` stays supported (it is nearly free once the client exists) but is no
longer the recommended interim.

### Two latent problems

1. **The saved workflow is technically invalid.** `POST …/describe_interface` returns 400:
   `parameter_bindings.model_id` is the literal string `yusri-caloyloy/scanncart-grocery-1-yolo11n-t1`
   where the schema wants a `$inputs.*` / `$steps.*` selector. It **runs today**, but any tightening
   of Roboflow's validation breaks it. Worth re-saving the workflow from the UI.
2. **JPEG quality drives payload size.** A 640-px frame at OpenCV's default quality 95 came to
   **128 KB**, not the 40–60 KB estimated in §4. Set quality ~80 on encode, or `cloud_api` costs
   ~2.5× the projected bandwidth.

---

## 1. Why

The custom grocery model lives in Roboflow. Its `.pt` weights are gated behind a paid Core plan, so
until we either subscribe or retrain from a free dataset export, the model is only reachable over an
API. Meanwhile the PRD's core promise is **one PC, no cloud, no network dependency** — which an API
in the hot path would break.

Rather than pick one and rewrite later, make the detector a swappable backend. The seam already
exists: `Detector` in `sidecar/app/inference.py` is a `Protocol`, and `AppState.detector_factory` is
already an injection point.

| Backend | Model runs | Network at runtime | Per-call cost | Extra process | Needs weights |
|---------|-----------|--------------------|---------------|---------------|---------------|
| `native` | In the sidecar process | ❌ None | ❌ Free | ❌ None | ✅ `.pt` on disk |
| `local_api` | Local Roboflow server | ❌ None | ❌ Free | ✅ One process (Docker **not** required — see §7a) | ❌ |
| `cloud_api` | Roboflow serverless | ✅ Required | ✅ Per inference | ❌ None | ❌ |

### Recommendation

**Ship on `native`. Develop on `local_api`. Demo on `cloud_api`.**

- `native` is the only backend that satisfies the PRD as written, and the only one with no latency
  floor. It is the target.
- `local_api` is the best interim: it runs the **full Workflow** (not just the model, so any
  filtering/remapping logic in `...-t1-logic` is preserved), stays offline, and costs nothing.
- `cloud_api` needs no GPU and no setup — useful for a laptop demo or for checking the local model
  against the hosted one.

---

## 2. Non-goals

- **Not** streaming video to the cloud. Roboflow's WebRTC Video Streaming API is a real product and
  it solves the bandwidth problem, but it puts a 100–300 ms network round trip and a live uplink
  dependency in front of every detection. Out of scope, per PRD §"no network dependency".
- **Not** replacing `camera.py` / `pipeline.py` with `InferencePipeline`. That would hand frame
  capture, tracking, and lifecycle to a third-party runtime and delete most of the sidecar.
- **Not** running inference at capture rate. See §4.

---

## 3. The seam

Every backend is one class satisfying the existing protocol. Nothing in `pipeline.py` changes.

```python
class Detector(Protocol):
    names: dict
    def infer(self, frame: np.ndarray) -> list[Detection]: ...
```

| Class | Backend | Notes |
|-------|---------|-------|
| `YoloDetector` | `native` | Exists today. Ultralytics `.track(persist=True)`. |
| `RoboflowRemoteDetector` | `local_api`, `cloud_api` | One class; the two backends differ only by base URL and whether auth is sent. |

Selection happens in `_default_detector_factory` (`sidecar/app/main.py:42`), which already receives
`(settings, device)`.

```python
def _default_detector_factory(settings: Settings, device: str):
    if settings.detector_backend == "native":
        return YoloDetector(settings.active_model, device=device, conf=settings.conf_threshold)
    return RoboflowRemoteDetector(
        api_url=_backend_url(settings),
        workspace=settings.roboflow_workspace,
        workflow_id=settings.roboflow_workflow_id,
        api_key=load_api_key(),                      # env only — never from settings
        infer_size=settings.remote_infer_size,
        conf=settings.conf_threshold,
        timeout_s=settings.remote_timeout_s,
        max_retries=settings.remote_max_retries,
        tracker=IouTracker(expiry_s=settings.track_expiry_s),
    )
```

---

## 4. Decoupling inference rate from capture rate

**This is the design decision that makes remote backends viable at all.**

60 fps implies a 16.7 ms budget. No network backend can meet it. But detection does not need to run
at capture rate — a grocery item placed on a counter is stationary. Detecting it within a few
hundred milliseconds is indistinguishable, at the counter, from detecting it in 16 ms.

The mechanism already exists and needs no new code:

- `LatestFrameBuffer` (`camera.py`) is a lock-guarded, size-1, newest-wins slot. A slow consumer
  never builds a queue; a 200 ms call simply reads whatever frame is current when it returns.
- `infer_frame_skip` is re-read from `self._settings` every `process_once`, so the detection rate is
  hot-tunable while capture runs.
- `track_expiry_s` (default 1.5 s) already tolerates multi-frame gaps.

Preview stays at 60 fps regardless — it is a capture + JPEG encode path with no model in it.

### Bandwidth, once decoupled

Send **640 px, not 1080p**. YOLO11 resizes to `imgsz=640` internally, so transmitting full frames is
pure waste.

| | Per frame | At 5 fps |
|---|---|---|
| 1080p JPEG | ~300 KB | ~12 Mbps |
| 640² JPEG | ~40–60 KB | **~2 Mbps** |

Because boxes come back **normalized to 0–1**, the downscale is transparent to everything
downstream — `lib/overlay.ts` converts to CSS percentages either way. Resolution of the transmitted
frame never has to match the preview.

**Suggested defaults per backend** (`infer_frame_skip` at `capture_fps=60`):

| Backend | `infer_frame_skip` | Effective detect rate |
|---------|-------------------|----------------------|
| `native` (GPU) | `0`–`1` | 30–60 fps |
| `native` (CPU) | `3`–`7` | 8–15 fps |
| `local_api` | `1`–`3` | 15–30 fps |
| `cloud_api` | `59`–`119` | 0.5–1 fps (see measured latency below) |

### ⚠️ Measured latency, and why `track_expiry_s` must rise

Three live calls to the serverless endpoint with a 640-px frame took **1626 ms, 600 ms, and
3250 ms** — far above the 150–300 ms this spec originally estimated. The third call reproduced
the failure directly: the same stationary item came back as **track id 1, 1, then 2**, because a
3250 ms round trip outlasted the default `track_expiry_s=1.5`, so the tracker forgot the track
and re-issued its id. One physical item, two log rows.

A **self-hosted** server is a different story. Measured against the Docker-free local server of
§7a on the same PC: **4384 ms cold** (first call — model download plus load), then **91, 88, 85,
94, 90 ms warm**. That is 7–36× faster than the cloud, so a single shared floor would have made
`local_api` wait 5 s to notice an item left for no reason.

The floor is therefore **per-backend**, via `settings_store.MIN_TRACK_EXPIRY_S_BY_BACKEND`:

| Backend | Measured round trip | Floor | Warn below it |
|---------|--------------------|-------|----------------|
| `local_api` | ~90 ms warm (4.4 s cold) | **2.0 s** | ✅ |
| `cloud_api` | 600–3250 ms | **5.0 s** | ✅ |
| anything else | — | 5.0 s (`MIN_REMOTE_TRACK_EXPIRY_S`, conservative fallback) | ✅ |

`min_track_expiry_s(backend)` is the accessor; `minTrackExpiryS()` in `settingsFields.ts` mirrors
it. **Set `track_expiry_s` to 5.0+ on `cloud_api`, 2.0+ on `local_api`.** The cold-start call is
the one hazard on local — warm it with `POST /api/detector/probe` before starting capture and the
4.4 s never lands mid-session.

---

## 5. The `track_id` problem

`pipeline.py` derives its entire item log from stable `track_id`s — `entered_at`/`left_at`,
expiry, and the renderer's dedup in `useSidecarStream.ts`. Without them the log collapses into
duplicate rows.

| Backend | Source of `track_id` |
|---------|---------------------|
| `native` | Ultralytics `.track(persist=True)` — native, stable |
| `local_api` / `cloud_api` | `tracker_id` on the response **if** the Workflow has a tracking block; otherwise assigned locally |

**Settled in Phase 0: there is no tracking block.** The response carries no `tracker_id`, so
`IouTracker` is the primary source of ids for both remote backends — not a fallback.

### `app/tracking.py` — the fallback

A small greedy IoU tracker, ~60 lines, **no new dependency**:

- Match each incoming box to the highest-IoU active track above a threshold (~0.3).
- Unmatched detection → new incrementing id.
- Track unseen for `expiry_s` → dropped.

Rejected alternative: `supervision`'s ByteTrack. It is better at fast, occluding, crossing motion —
none of which describes items sitting on a checkout counter — and it pulls a heavy dependency tree
for that. Revisit only if IoU matching proves insufficient in practice.

Pass-through rule: if the response carries a usable `tracker_id`, use it and skip the local tracker
entirely. Never mix the two id spaces in one session.

---

## 6. Settings

### New fields on `Settings` (`sidecar/app/settings.py`)

```python
detector_backend: str = "native"          # native | local_api | cloud_api
roboflow_workspace: str = "yusri-caloyloy"
roboflow_workflow_id: str = "scanncart-grocery-vscanncart-grocery-1-yolo11n-t1-logic"
local_api_url: str = "http://127.0.0.1:9001"
cloud_api_url: str = "https://serverless.roboflow.com"
remote_infer_size: int = 640
remote_timeout_s: float = 5.0
remote_max_retries: int = 2
```

All eight are **restart-required** — each is baked into the detector at construction. Add them to
`RESTART_REQUIRED_FIELDS` in `settings_store.py`.

Add `ALLOWED_BACKENDS = {"native", "local_api", "cloud_api"}` beside `ALLOWED_MODELS`, with a
matching branch in `_valid_field()` and a `field_validator` on `SettingsUpdateRequest`.

### 🔑 The API key is not a setting

`Settings` is persisted to `data/settings.json` and returned wholesale by `GET /api/settings` to the
renderer. **A secret must never enter that object.**

- Key lives in `sidecar/.env` as `ROBOFLOW_API_KEY` (already gitignored; `.env.example` committed).
- Loaded by a new `app/credentials.py` — a small `KEY=value` parser, with the real environment
  taking precedence over the file. No `python-dotenv` dependency; matches the house style of
  `settings.py` staying loader-free. (Named `credentials`, not `secrets`, to avoid shadowing the
  stdlib module name.)
- `SettingsResponse` exposes **`roboflow_api_key_present: bool`** only. Never the value.

Rationale for `.env` over Electron passing it through `spawn`: keeps the desktop app out of secret
handling entirely, and the sidecar stays independently runnable via `python run.py`.

### New warnings in `compute_warnings()`

| Condition | Warning |
|-----------|---------|
| `cloud_api` selected | Network dependency + per-inference cost; contradicts the PRD's offline guarantee. |
| Remote backend, no API key | Will fail at capture start — set `ROBOFLOW_API_KEY` in `sidecar/.env`. |
| `local_api` selected | Requires an inference server running on `local_api_url` (§7a — no Docker needed). |

`compute_warnings()` takes `api_key_present: bool | None` rather than reading the key itself, so
settings tests never touch the filesystem; `AppState.api_key_probe` is the matching injection
seam, mirroring `hardware_prober`. Checking that `active_model` actually resolves on disk lives
in `/api/detector/probe` (§8) instead of here, keeping `compute_warnings()` pure.

---

## 7. HTTP client — no new dependencies

`httpx>=0.28` is **already** in `sidecar/requirements.txt`. The Workflow endpoint is a plain JSON
POST, identical for cloud and self-hosted apart from the base URL:

```
POST {api_url}/{workspace}/workflows/{workflow_id}
Authorization: Bearer {api_key}        # header auth, never query string or body
Content-Type: application/json

{ "inputs": { "image": { "type": "base64", "value": "<jpeg>" } } }
```

Using `httpx` directly rather than `inference-sdk` buys: zero new dependencies, direct control of
timeout and retry, and no Python-version ceiling — `inference-sdk`'s runtime sibling `inference`
caps at `>=3.10,<3.13`, and the repo targets 3.12. It costs hand-maintaining the request shape,
which is consistent with the WS and settings contracts already hand-synced per CLAUDE.md.

`app/roboflow.py` owns one well-named function plus typed errors:

| Exception | Raised on | Surfaced as |
|-----------|-----------|-------------|
| `RoboflowAuthError` | 401 / 403 | `401` from capture start |
| `RoboflowUnavailable` | Connection refused (local server down) | `503` |
| `RoboflowTimeout` | Exceeded `remote_timeout_s` after retries | `504` |
| `RoboflowError` | Anything else, incl. malformed body | `502` |

Retries: `max_retries` attempts with exponential backoff, on timeout and 5xx only — **never** on
4xx, and never so long that it stalls the pipeline thread past `track_expiry_s`.

### Parsing defensively

The response is a **list** (one entry per input image); each entry is a dict keyed by the workflow's
own output names. Do not hard-code those names:

1. Take entry `[0]`.
2. Scan its values for the first `dict` containing a `predictions` list.
3. Per prediction read only `x`, `y`, `width`, `height`, `confidence`, `class` (fallback
   `class_name`), and `tracker_id` if present. **Verified: this workflow emits `class`.**
4. Convert centre-xywh in transmitted-image pixels → normalized xyxy, clamped 0–1, reusing
   `_clamp01` from `inference.py`.

**Never** read or retain segmentation `points`, and never log a base64 image field — workflow image
outputs are hundreds of KB.

`names` for the `Detector` protocol is built lazily from class names observed, since a remote
backend has no model manifest to read.

---

## 7a. Running `local_api` without Docker

Roboflow documents `inference server start`, which pulls and runs a Docker image. On a machine with
no Docker — the capstone PC included — the same server runs natively, because the image is only a
thin wrapper around a FastAPI app the `inference` wheel already contains.

`sidecar/local_inference_server.py` builds that app with the same wiring as the image's
`docker/config/cpu_http.py`, and `sidecar/requirements-inference.txt` pins what it needs.

### Setup (once)

```bash
cd sidecar
uv venv --python 3.12 .venv-inference
uv pip install --python .venv-inference/Scripts/python.exe -r requirements-inference.txt
```

**A separate venv is mandatory.** `inference` pins numpy/opencv versions that conflict with the
sidecar's ultralytics stack — installing it into `.venv` breaks `native`. The two processes share
nothing but HTTP. Python must be **<3.13**; `inference` publishes no 3.13 wheels.

### Run

```bash
.venv-inference/Scripts/python.exe local_inference_server.py
# → Roboflow inference server (no Docker) on http://127.0.0.1:9001
```

Then point the sidecar at it — `detector_backend: local_api`, `track_expiry_s: 2.0+` — and press
**Test connection** in the Admin Panel.

### Two things the wheel does not ship

Both are Docker-image assumptions that the launcher works around; neither needs a patched
`inference` install.

1. **The HTTP-server dependencies are undeclared.** `import
   inference.core.interfaces.http.http_api` fails on `asgi_correlation_id`, then
   `fastapi_cprofile`, then more — the image bakes them in rather than declaring them on the
   wheel. `requirements-inference.txt` lists them explicitly.
2. **The landing page is mounted unconditionally from a relative path.** `HttpInterface.__init__`
   calls `StaticFiles(directory="./inference/landing/out/static")`, which exists in the image's
   working directory but not in the wheel, and there is no flag to disable it. The launcher
   creates those directories under `sidecar/.inference-runtime/` and `chdir`s there **after**
   importing `inference`, so the real package can never be shadowed by the stub tree. The landing
   UI 404s; the sidecar only ever calls the workflow endpoint, so nothing notices.

### Credentials

The server reads `ROBOFLOW_API_KEY` through the sidecar's own `app.credentials` loader, so both
processes read one `sidecar/.env`. The key is needed to pull the private grocery model's weights
the first time a workflow runs; afterwards the weights are cached on disk and the server keeps
working with no internet.

### Verified

| Check | Result |
|-------|--------|
| `GET /info` | `{"name":"Roboflow Inference Server","version":"1.5.1"}` |
| Workflow POST via `WorkflowClient` | 200, parsed by `find_predictions()` |
| Warm round trip | 85–94 ms |
| `POST /api/detector/probe` (sidecar → local server) | `reachable: true`, `latency_ms: 77.6` |

---

## 8. New endpoint: `POST /api/detector/probe`

Validates the selected backend **before** the user hits Start, so a misconfigured URL or missing key
surfaces in the Admin Panel instead of as a failed capture.

Returns `{ reachable, backend, latency_ms, class_names, detail }`. Powers a **"Test Connection"**
button beside the backend picker. For `native` it checks the weights file resolves; for remote
backends it posts a tiny synthetic frame.

---

## 9. Desktop mirrors

Per the hand-sync convention in CLAUDE.md:

| File | Change |
|------|--------|
| `lib/settingsFields.ts` | `ALLOWED_BACKENDS`, field defs + tradeoff copy for the 8 new fields |
| `lib/settingsDefaults.ts` | Mirror the new defaults |
| `lib/api.ts` | `SettingsPayload` gains the new fields + `roboflow_api_key_present`; add `probeDetector()` |
| `views/AdminPanel.tsx` | Backend picker (segmented control), "Test Connection" button, key-missing banner |
| `views/LiveView.tsx` | Surface backend errors from capture start; show active backend in the stats strip |

The backend picker is restart-required, so it reuses the existing save-gating that already disables
Save while capture runs.

---

## 10. Testing

Per repo convention — **injected fakes, no network, no camera, no GPU**.

| Test file | Covers |
|-----------|--------|
| `tests/test_tracking.py` | IoU matching, new-id assignment, expiry, id stability across gaps |
| `tests/test_roboflow_client.py` | Request shape, header auth, retry/backoff, each typed error |
| `tests/test_roboflow_detector.py` | Defensive parsing, coordinate conversion, `tracker_id` pass-through vs. local fallback, empty/malformed responses |
| `tests/test_settings_store.py` | Validators + warnings for the new fields *(extend)* |
| `tests/test_settings_api.py` | `PATCH` accepts/rejects backends; key never serialized *(extend)* |

`RoboflowRemoteDetector` takes a `client_factory` injection seam, mirroring `model_factory` on
`YoloDetector`. One test must assert the API key appears in **no** settings response.

The single smoke test that does hit the network is marked `@pytest.mark.network` and skipped unless
`ROBOFLOW_API_KEY` is set, so `make test` stays hermetic.

---

## 11. Development plan

Phases are ordered so each lands independently testable. **Phase 0 gates Phase 4.**

| # | Phase | Deliverable | Depends on |
|---|-------|-------------|-----------|
| **0** | ✅ **Ground the workflow** | Definition read, live response captured against a real dataset image. See §0. | — |
| 1 | ✅ Local tracker | `app/tracking.py` + `tests/test_tracking.py` (23 tests). No new deps. | — |
| 2 | ✅ Settings plumbing | 8 fields, validators, warnings, `credentials.py`, `SettingsResponse` + 105 tests | — |
| 3 | ✅ HTTP client | `app/roboflow.py`, 4 typed errors, retries, response discovery + 42 tests | 2 |
| 4 | ✅ Detector | `RoboflowRemoteDetector`, factory branch, `/api/detector/probe`, error mapping + 54 tests | — |
| 5 | Desktop | Mirrors, backend picker, Test Connection, error surfacing + vitest | 2, 4 |
| 6 | Docs | CLAUDE.md architecture notes, README setup, ARCHITECTURE.md §3/§7, MODEL_TRAINING.md §7 cross-ref | 5 |

**Phases 0–4 are complete** — full sidecar suite green at 330 passed, still zero new
dependencies, and verified end-to-end against the live workflow. Phase 5 (desktop) is next.

---

## 12. Open questions

- [x] ~~Does the Workflow contain a tracking block?~~ **No** — `IouTracker` is the primary path.
- [x] ~~Real output names and parameters?~~ One output `predictions`, one input `image`, no parameters.
- [ ] **Class-name display mapping** — the 7 classes mix `snake_case` and `Title Case`; needs a
      presentation layer in the renderer.
- [ ] **Re-save the workflow** to fix the invalid `model_id` binding (§0).
- [ ] **Was the model trained on Roboflow or in Colab?** If Colab, `best.pt` already exists on our
      side and `native` needs no subscription and no retrain.
- [ ] **Cloud cost per inference** at 3–5 fps against the plan quota — may make `cloud_api`
      demo-only in practice.

---

## 13. Checklist

- [x] Workflow definition read; real response captured *(Phase 0)*
- [x] `app/tracking.py` + tests green
- [x] 8 settings fields, validators, warnings; all in `RESTART_REQUIRED_FIELDS`
- [x] `ROBOFLOW_API_KEY` read from `sidecar/.env`; **never** in `settings.json` or any API response
- [x] `app/roboflow.py` with timeout, bounded retries, four typed errors
- [x] Header auth only — never query string, never request body
- [x] Frames downscaled to `remote_infer_size` before transmit
- [x] Response parsed by discovery, not hard-coded output names
- [x] Image/base64 outputs never logged or retained
- [x] `track_id` stable across a session on every backend
- [x] `/api/detector/probe` (endpoint done; Admin Panel button is Phase 5)
- [ ] Desktop mirrors updated by hand (both directions verified)
- [ ] `make test` hermetic — no network, no camera, no GPU
- [ ] Live View verified end-to-end on all three backends
- [ ] Pasted API key rotated
