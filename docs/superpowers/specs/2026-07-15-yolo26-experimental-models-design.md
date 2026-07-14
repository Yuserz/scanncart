# SCANnCART — YOLO26 as experimental model options — Design

**Date:** 2026-07-15
**Status:** Approved (discussed in-session)
**Depends on:** `sidecar/app/settings_store.py`, `sidecar/requirements.txt`,
`desktop/src/renderer/src/lib/settingsFields.ts`, `desktop/src/renderer/src/views/AdminPanel.tsx`.

## Problem

The admin panel's model picker is a hardcoded YOLO11-only whitelist
(`yolo11n/s/m/l/x`). The detector itself (`YoloDetector`) is model-agnostic —
any family the installed `ultralytics` package ships loads fine. We want an
experimental lane for YOLO26 (NMS-free, notably faster CPU inference than
YOLO11 per Ultralytics' published benchmarks) without destabilizing the
supported YOLO11 path, and with the hardware each option needs clearly
spelled out in the UI.

Decisions already made in discussion:

- **Stay on YOLO11 as the supported set.** Presets remain YOLO11-only.
- **Do not add older families** (v8/v5/v3…): strictly worse
  accuracy-per-compute than YOLO11; risk with no reward.
- **Add only `yolo26n.pt` / `yolo26s.pt` / `yolo26m.pt`**, clearly badged as
  experimental (l/x sizes serve no experiment the m size can't).

## Design

### Sidecar

- `settings_store.py`: new `EXPERIMENTAL_MODELS = {"yolo26n.pt",
  "yolo26s.pt", "yolo26m.pt"}`; `ALLOWED_MODELS` becomes the YOLO11 set
  union `EXPERIMENTAL_MODELS`. Validation (`_valid_field`, Pydantic
  validator in `schemas.py`) picks the new names up automatically.
- `compute_warnings()`: when `settings.active_model` is experimental, append
  a soft warning: the model is experimental, presets/tuning guidance are
  calibrated for YOLO11, it requires `ultralytics >= 8.4` in the sidecar
  venv, and weights auto-download on the first capture start (needs internet
  once). This surfaces in `SettingsResponse.warnings`, which AdminPanel
  already renders after save.
- `requirements.txt`: bump `ultralytics>=8.3` → `ultralytics>=8.4`
  (installed venv has 8.4.92; YOLO26 weights load from the late-8.3.x line,
  so 8.4 is a safe floor).
- Presets (`presets.py`) unchanged.

### Desktop (hand-kept mirror, per project convention)

- `settingsFields.ts`: add the three names to `ALLOWED_MODELS`; new
  `EXPERIMENTAL_MODELS` list and `MODEL_SPEC_HINTS: Record<string, string>`
  with per-model hardware guidance:
  - `yolo26n.pt` — lightest YOLO26; roughly yolo11n-class hardware (modern
    4-core CPU, 8 GB RAM); NMS-free design typically runs faster than
    yolo11n on CPU.
  - `yolo26s.pt` — strong CPU (8+ cores) or entry CUDA GPU (≥2 GB VRAM) to
    hold ~30 fps.
  - `yolo26m.pt` — discrete CUDA GPU (≥4 GB VRAM); CPU-only machines will
    fall behind in real time.
- `AdminPanel.tsx`: experimental options render as `"<name> (experimental)"`
  in the dropdown; when the currently selected model has a spec hint, an
  extra amber hint line (`data-testid="model-spec-hint"`) renders under the
  Model field showing that model's hardware requirements. Everything else
  (badges, save gating, restart-required flow) is unchanged —
  `active_model` is already a restart-required field.
- `AdminPanel.css`: one `.field-hint.experimental` amber variant.

## Testing

- Sidecar: PATCH `/api/settings` accepts `yolo26n.pt`; `compute_warnings`
  includes an "experimental" warning for a yolo26 model and not for yolo11n;
  settings round-trip persists a yolo26 selection.
- Desktop: with `yolo26n.pt` selected, the spec hint renders and the option
  label carries "(experimental)"; with a yolo11 model selected, no spec hint.
- End-to-end (manual/driver): select yolo26n in the admin panel, save, start
  capture, confirm frames stream. Caveat: currently blocked on this machine
  by the intermittent Defender native-DLL kill; verify when the sidecar
  starts cleanly.

## Non-goals

- No preset changes, no older YOLO families, no free-form custom `.pt` path
  (a natural follow-up for grocery fine-tunes, but out of scope here).
