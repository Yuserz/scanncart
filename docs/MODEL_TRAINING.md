# 🧠 SCANnCART – Custom Model Training Guide

> How to build, label, and train a **custom 10-class grocery item detector** for SCANnCART,
> and how to drop the resulting weights into the sidecar.
>
> Related: [PRD.md](./PRD.md) · [ARCHITECTURE.md](./ARCHITECTURE.md) · [DEPLOYMENT.md](./DEPLOYMENT.md)

---

## 1. Project Type — pick **Object Detection**

When creating the Roboflow project, choose **Object Detection**. Not classification, not segmentation.

| Type | Verdict | Why |
|------|---------|-----|
| **Object Detection** | ✅ **Use this** | Outputs bounding boxes + `track_id`, which is exactly what the sidecar consumes. |
| Instance Segmentation | ❌ | Trains fine, but costs ~2–3× the labeling time (polygons vs 2-click boxes) for masks the app discards. |
| Classification | ❌ | One label per image. Cannot handle multiple items on the counter at once — kills the core use case. |
| Keypoint / Multimodal / Semantic Seg. | ❌ | Not applicable to item counting. |

**Why detection specifically:** `sidecar/app/inference.py` calls `YOLO(...).track(persist=True)` and
`normalize_detections()` converts `xyxy` boxes into 0–1 relative coordinates. The renderer
(`lib/overlay.ts` → `views/LiveView.tsx`) turns those into CSS percentage rectangles. Anything that
isn't a box has nowhere to go in this pipeline.

### Other fields on the create-project screen

| Field | Recommendation |
|-------|----------------|
| **Project Name** | e.g. `scanncart-grocery-10` |
| **Visibility** | **Public** if you're on the Roboflow free tier and the data isn't sensitive. **Private** if store/brand imagery can't be redistributed. |
| **Annotation Group** | `grocery-item` |
| "Ask the Roboflow Agent" box | Ignore it — the "Reading nutrition labels" text is just placeholder. |

---

## 2. Dataset Size — how many images for 10 classes

Three honest tiers. **The middle one is the target for this capstone.**

| Tier | Images **per class** | Instances per class | Result |
|------|---------------------|---------------------|--------|
| Bare minimum / demo | 100–150 | ~250 | Works only in the exact lighting you filmed. Falls apart elsewhere. |
| 🎯 **Recommended** | **250–350** | **600–1000** | Reliable at a fixed checkout station. mAP50 ≈ 0.90+. |
| Production-grade | 800–1500 | 3000+ | Ultralytics' official guidance. Overkill for a prototype. |

> **Total target: ~2,500–3,500 raw images** across all 10 classes.

### Instances matter more than images

One photo with 4 items on the counter = **4 labeled instances**. Budget the set like this:

| Bucket | Count | Purpose |
|--------|-------|---------|
| **Solo shots** | ~150 per item (~1,500 total) | Teaches what each class looks like, all angles. |
| **Multi-item scenes** | ~800–1,200 total | 3–6 items together, overlapping, partially occluded. **This is what the app actually sees.** One image feeds 4+ classes at once — highest value per label. |
| **Hard negatives** | ~150–250 total (5–10%) | Empty counter, hands, bags, wallet, phone, conveyor. **Images with zero annotations.** Cheapest possible way to kill false positives — and the step most people skip. Roboflow supports null-annotation images directly. |

---

## 3. Variation Beats Volume

300 images shot in one 5-minute session from one angle are worth less than 120 genuinely varied ones.
Deliberately cover every axis below:

| Axis | What to vary |
|------|--------------|
| **Orientation** | Front label, back, side, upside down, lying flat vs standing. A cereal box face-down is a visually different object. |
| **Scale / distance** | Close to lens vs at the edge of frame. |
| **Occlusion** | 30–50% covered by another item or a hand. |
| **Lighting** | Store fluorescents, daylight, dimmed, glare on plastic wrap and glossy labels. |
| **Deformation** | Crumpled chip bags, dented cans, squeezed bottles, creased pouches. |
| **Background** | Different counter surfaces, cluttered vs clean, conveyor vs static table. |
| **Motion** | Some mild motion blur — items get placed, not posed. |

---

## 4. Two Traps to Avoid

### ⚠️ Visually similar SKUs need extra data
If two of the 10 items are the same brand in different flavors (e.g. two noodle variants differing
only by a color band), budget **1.5–2× the images** for that pair. Otherwise the model flips between
them frame to frame — and because `pipeline.py` dedupes by `track_id`, **one physical item gets
logged as two different products.** That's a visible, demo-breaking bug, not just a metric dip.

### ⚠️ Don't fake volume with augmentation
Roboflow will happily 3× your set to ~9,000 "training images." Augmented copies are not new
information. **Never report the augmented count as your dataset size** — report raw images and
instance counts.

Keep augmentation modest and physically plausible:

| Augmentation | Setting |
|--------------|---------|
| Horizontal flip | ✅ On |
| Rotation | ✅ ±15° |
| Brightness / exposure | ✅ ±20% |
| Blur | ✅ Slight (≤1 px) |
| Vertical flip | ❌ Off — unless items genuinely appear upside down |
| Heavy mosaic / cutout | ❌ Off — the camera is fixed; it just adds noise |

---

## 5. Capture Workflow

1. **Use the real rig.** Record on the actual Logitech StreamCam, at the deployed mount height,
   at the sidecar's configured resolution. Defaults live in `sidecar/app/settings.py`:
   `capture_width=1280`, `capture_height=720`, `capture_fps=60`.
   *Domain match beats volume — 200 images from the real rig outperform 1,000 web-scraped product photos.*
2. **Record video, then extract frames at ~1 fps** — not 30. Consecutive frames are near-duplicates
   that inflate your count and leak between train/val splits.
3. **Dedup.** Run Roboflow's similarity/duplicate detection to catch stragglers.
4. **Label boxes tight** to the object. Include partially visible items at frame edges — the app will
   see those constantly.
5. **Split 70 / 20 / 10** train / valid / test — **by capture session, not randomly.** Random
   splitting puts near-identical frames on both sides and yields a flattering mAP that won't hold up
   in the store.

---

## 6. Training

Start from **`yolo11s.pt`**, not `yolo11n.pt`. With only 10 classes and a few thousand images, `s`
trains in roughly the same wall-clock time on a decent GPU and is noticeably better on small and
occluded items. The sidecar's `mid_range` preset (`sidecar/app/presets.py`) already defaults to it.

```bash
yolo detect train \
  model=yolo11s.pt \
  data=path/to/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  patience=25 \
  project=runs/scanncart \
  name=grocery10
```

| Parameter | Guidance |
|-----------|----------|
| `epochs` | 100 is a sane start. Watch for val loss plateau; `patience=25` early-stops for you. |
| `imgsz` | 640 matches the 1280×720 capture well. Only raise to 960 if small items (sachets, sauce packets) are being missed. |
| `batch` | 16 on ~8 GB VRAM; drop to 8 if you hit OOM. |
| Transfer learning | Keep the COCO-pretrained backbone (the default). Do **not** train from scratch at this dataset size. |

### What "good" looks like

| Metric | Target |
|--------|--------|
| **mAP50** | ≥ 0.90 |
| **mAP50-95** | ≥ 0.65 |
| **Per-class recall** | ≥ 0.85 for *every* class — check the per-class table, not just the average |
| Confusion matrix | Low off-diagonal mass between similar SKUs; low "background" column (false positives) |

If one class lags badly, that's a **data problem, not a training problem** — go add 100 more varied
images of that item rather than tweaking hyperparameters.

---

## 7. Integrating the Trained Weights

Custom weights are **first-class** — no whitelist edits needed. Any `.pt` (or `.onnx`) dropped
into `sidecar/models/` is valid: `is_custom_model()` (`sidecar/app/settings_store.py`) accepts it,
the `active_model` validator passes it, and the Admin Panel's Model picker offers it (the
renderer's `ALLOWED_MODELS` mirror in `settingsFields.ts` covers `models/...` paths via the
`CUSTOM_MODEL` convention). The `yolo11n/s/m/l/x.pt` entries in that list exist only to keep the
stock COCO weights selectable.

Checklist to wire in `best.pt`:

1. Copy the trained weights into `sidecar/models/` with a descriptive name, e.g.
   `scanncart-grocery10.pt`.
2. (Optional) Add a `MODEL_SPEC_HINTS` entry in
   `desktop/src/renderer/src/lib/settingsFields.ts` so the Admin Panel can show hardware guidance
   for it.
3. Select it in the Admin Panel. `active_model` is a **restart-required** field
   (`RESTART_REQUIRED_FIELDS`), so capture must be stopped before saving.

Two things the old whitelist path silently decided for you:

- **`resize_mode` is format-aware now.** `auto` resolves to `stretch` for a custom `.onnx` (a
  Roboflow export, trained stretched) and to `letterbox` for a custom `.pt` (a locally trained
  checkpoint — the output of this guide — trained letterboxed). Only force `stretch` for a `.pt`
  if you know the export trained stretched; see `resolve_resize_mode()` in
  `sidecar/app/settings_store.py`.
- **GPU.** A `.pt` runs on torch directly, so `device: "auto"` resolving to `cuda` is the fast
  path (docs measured ~25 ms/frame in-app on a GTX 1050 Ti, vs ~91 ms for the custom ONNX on
  CPU). The CUDA requirement is on the torch install, not on onnxruntime — that only matters
  for `.onnx` models (see `docs/DETECTOR_BACKENDS.md §1a` and `sidecar/requirements-cuda.txt`).

### Post-integration tuning

| Setting | Note |
|---------|------|
| `conf_threshold` (default `0.5`) | A custom 10-class model is usually more confident than the COCO baseline. Try `0.6` to cut flicker. Restart-required. |
| `track_expiry_s` (default `1.5`) | Raise if items briefly drop out and re-enter as a new `track_id` (causing duplicate log rows). Hot-reloadable. |
| `infer_frame_skip` (default `0`) | Only touch if fps is short of target. Hot-reloadable. |

---

## 8. Quick Checklist

- [ ] Roboflow project created as **Object Detection**
- [ ] 10 classes defined, names final (renaming later invalidates existing labels)
- [ ] ~250–350 raw images per class captured on the real StreamCam rig
- [ ] ≥800 multi-item scenes with overlap and occlusion
- [ ] 5–10% hard negatives (background-only, zero annotations)
- [ ] All variation axes in §3 covered
- [ ] Extra data collected for visually similar SKU pairs
- [ ] Frames extracted at ~1 fps, duplicates removed
- [ ] Split 70/20/10 **by capture session**
- [ ] Modest augmentation only; raw counts reported, not augmented
- [ ] Trained from `yolo11s.pt`, mAP50 ≥ 0.90, every class recall ≥ 0.85
- [ ] Weights added to `ALLOWED_MODELS` in **both** the sidecar and the desktop mirror
- [ ] Verified end-to-end in Live View with `conf_threshold` / `track_expiry_s` retuned
