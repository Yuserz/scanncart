# 🧠 SCANnCART – Custom Model Training Guide

> How to build, label, and train a **custom grocery SKU detector** for SCANnCART,
> and how to drop the resulting weights into the sidecar. §1–§7 are the method;
> **§8 is this project's concrete roster, naming, and split convention.**
>
> Related: [PRD.md](./PRD.md) · [ARCHITECTURE.md](./ARCHITECTURE.md) · [DEPLOYMENT.md](./DEPLOYMENT.md) ·
> [CAPTURE_CHECKLIST.md](./CAPTURE_CHECKLIST.md) (what this project still has to shoot)

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
| **Project Name** | v1: `scanncart-grocery` · v2: `snc-grocery` (see §8.2) |
| **Visibility** | **Public** if you're on the Roboflow free tier and the data isn't sensitive. **Private** if store/brand imagery can't be redistributed. |
| **Annotation Group** | `grocery-item` |
| "Ask the Roboflow Agent" box | Ignore it — the "Reading nutrition labels" text is just placeholder. |

---

## 2. Dataset Size — how many images per class

Three honest tiers. **The middle one is the target for this capstone.**

| Tier | Images **per class** | Instances per class | Result |
|------|---------------------|---------------------|--------|
| Bare minimum / demo | 100–150 | ~250 | Works only in the exact lighting you filmed. Falls apart elsewhere. |
| 🎯 **Recommended** | **250–350** | **600–1000** | Reliable at a fixed checkout station. mAP50 ≈ 0.90+. |
| Production-grade | 800–1500 | 3000+ | Ultralytics' official guidance. Overkill for a prototype. |

> **Total target: ~2,000–2,800 raw images** across all 8 classes (§8.1).

### Instances matter more than images

One photo with 4 items on the counter = **4 labeled instances**. Budget the set like this:

| Bucket | Count | Purpose |
|--------|-------|---------|
| **Solo shots** | ~150 per item (~1,500 total) | Teaches what each class looks like, all angles. |
| **Multi-item scenes** | ~800–1,200 total | 3–6 items together, overlapping, partially occluded. **This is what the app actually sees.** One image feeds 4+ classes at once — highest value per label. |
| **Hard negatives** | ~150–250 total (5–10%) | Empty counter, hands, bags, wallet, phone, conveyor. Cheapest possible way to kill false positives — and the step most people skip. See the caveat below on how they enter a version. |

> **How these enter a version.** Roboflow models a background image as a **null annotation**: the
> image is marked as deliberately containing no object of interest, with the annotator's
> **Mark Null** tool (the empty-set button, keyboard shortcut **N**). A null-marked image counts as
> annotated and is included when you generate a version; a merely unannotated one is excluded and
> shows up under the project's `unannotated` count instead. There is no API for this — the upload
> endpoint has no null flag and the annotate endpoint rejects an empty annotation file — so it is
> a per-image click in the UI. Mind the "include images without annotations" toggle at version
> generation: it takes *every* unreviewed image, not just the ones you meant.
>
> **Verifying it worked.** A null annotation is distinguishable from an unlabeled image, but only
> by the *type* of the search API's `annotations` field — not by its emptiness:
>
> | `annotations` | Meaning | Counted as |
> |---|---|---|
> | `[]` (empty **list**) | not labeled yet | unannotated |
> | `{"count": 0, "classes": {}}` (**dict**) | null annotation | annotated |
> | `{"count": n, "classes": {...}}` (**dict**) | labeled | annotated |
>
> `count: 0` **inside an object** is a decision someone made; no object at all is outstanding
> work. Confirmed on the v1 project: all 1,516 images return an object, **16 of them with
> `count: 0`**, and v1 reports `unannotated: 0` — so v1 does carry a small hard-negative set.
> `label_progress.py` lists nulls explicitly, so C2b is checkable rather than guesswork.
> (Do **not** use the project's `unannotated` counter for this: it does not decrement on delete
> and is not a reliable total either.)
>
> The variant that needs **no** extra step is a frame containing **one roster product plus the
> clutter you want ignored**: it earns its place through the product's box, and everything
> unlabeled in it teaches the model background. Shoot that for the bulk of the bucket and keep the
> pure-background set small (~30 frames), since marking it is manual.
> See `docs/CAPTURE_CHECKLIST.md` §C2 and `clean_v2.py sanity`.

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
   in the store. In v2 the capture session is the upload batch — see §8.3 for the rule and the two
   coverage checks it needs.

---

## 6. Training

### Generate the version from reviewed settings, not from the UI

The version's preprocessing is a one-shot decision — it bakes the geometry every training image is
stored at, and it cannot be edited afterwards (different settings mean a different version number).
`sanity` used to report that there was no API for it; there is, so it lives in code:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --dry-run
sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --yes
sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --verify 2
```

That sends **auto-orient on + `Stretch to` 640×640**, augmentation empty, which is exactly what v1's
version records (`scanncart-grocery/1`, mAP 98.21). Two things about that geometry are easy to get
wrong and expensive to notice:

- **Stretch, not `Fit within`.** At 1280×720 the x axis is the binding one, so both options scale it
  by 0.5 — but stretch keeps the y scale at 0.889 where letterboxing drops it to 0.5. Objects keep
  ~1.8× more vertical pixels, and the `far` cells are precisely the ones short of pixels.
- **The geometry has to travel with the weights.** `resolve_resize_mode()`'s format heuristic
  answers *letterbox* for a custom `.pt` (Roboflow's own `.onnx` exports get stretch), which is the
  wrong answer for these weights — every object would be presented at 0.56× the canvas they were
  trained at. So the requirement is *recorded* beside them by `--install` and `auto` honours it; a
  weight with no record has to be set to `resize_mode: stretch` by hand. `generate_version.py
  --verify` prints this with the version it checked, and the Admin Panel's model entry carries it.

`--verify` also refuses a version that filters nulls: `filter-null` drops null-annotated images,
which *are* the hard-negative set, and the resulting count still looks plausible.

### Train

Start from **`yolo11s.pt`**, not `yolo11n.pt`. With only 8 classes and a few thousand images, `s`
trains in roughly the same wall-clock time on a decent GPU and is noticeably better on small and
occluded items. The sidecar's `mid_range` preset (`sidecar/app/presets.py`) already defaults to it.

The run is scripted, so the hyperparameters live in one place and the drop-in step cannot be
forgotten (`train_model.py` carries `EPOCHS`/`IMGSZ`/`BATCH`/`PATIENCE` and the test suite pins them
to the table below):

```bash
# fetch the version's YOLOv11 PyTorch export (waits for it to build, then unpacks it)
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --download --version 2

# check the export against the roster, then print the exact run - changes nothing
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py

# train, then report the final-epoch metrics against 6's targets
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --yes

# the acceptance number: per-class recall on the test split against the 0.85 floor.
# Also writes those numbers down, so the next command can carry them into the weights
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --val

# install the best checkpoint as sidecar/models/scanncart-grocery-v2.pt, recording the
# resize_mode it needs *and* the score above (--version names the dataset version it came from)
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --install --version 2
```

(The last four read the export downloaded by the first; `--dataset-dir <folder>` — the same flag
as the older `--export-dir` — points them at a copy you unzipped or ingested by hand instead.
`--val` and a bare `--install` find the run themselves — an explicit `--run-dir <folder>` still
wins, and is what you want when the newest run is not the one you mean.)

#### v1's local build — the same script, a different generation

v1's weights so far are the Roboflow-hosted `.onnx`. This builds a `.pt` on this machine from the
same 1,815 annotated images, so the app can run a natively-loaded model now instead of waiting for
v2's set to be finished. It is the same four commands with `--generation v1`:

```bash
# check the ingested export's seven classes and print the run - changes nothing
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1

# train (writes to the dataset workspace, never into the repo)
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1 --yes

# per-class recall on the test split, against the same 0.85 floor v2 is judged by
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1 --val

# the drop-in: sidecar/models/scanncart-grocery-v1.pt plus its record
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1 --install
```

Three things differ from v2, and none of them is a flag you have to remember — `generations.py`
holds them per generation (§8.5):

- **Seven classes, not eight.** v1's export is judged against *v1's* class list, so expect the check
to print `note: 1 class(es) another generation declares are not in this one … Palmolive Naturals
Bar Soap 85g`. That is the dataset described correctly, not a problem to fix here — v1 has no
Palmolive (§8.1's table), and the app will say the same thing about the installed weight.
- **No per-distance grid.** Distance is a Roboflow tag, and v1's images predate the tagging, so
`--val` reports the per-class table and says the breakdown does not apply. It is *not* a manifest
that went missing, and the tool says which of the two it means.
- **Its resize requirement is frozen** at `stretch`, measured from the export's own frames (all
1,815 are 640×640 with no constant border). v2's is derived from the preprocessing that generated
its version; v1's version already exists and will never be regenerated, so re-deriving it from
v2's constant would let a later change to v2 silently rewrite it.

Any generation can be pointed at another folder or another class list with `--dataset-dir` and
`--classes` — the generation is only the set of defaults those override.

**What the run is allowed to take.** This box is shared with the app, a browser and the sidecar, so
the run prints its budget before it starts and works inside it: CPU and RAM under
`--max-use-percent` (20% by default), the dataloader process count derived from that instead of
ultralytics' default of eight, `torch`'s thread count clamped before anything allocates, and the
batch size derived from the VRAM share — `--batch` is a *ceiling*, `--max-vram-percent` is the
lever. Measured on this machine with the defaults: `batch 8`, `workers 2`, `yolo11s` at 640 over
1,265 training images, ≈42 s/epoch, GPU at ~85%.

> The VRAM number needs one honest caveat: the share is applied through the **batch size**, not the
> allocator, so 35% is not a hard ceiling. Measured mid-run, training held ≈5.9 GB of the 8.6 GB
> card — the optimizer state and activations are real, and `resources.py`'s ~0.6 GB figure is a
> forward pass. `--batch 4` is how to take it lower; `--max-use-percent 60` when the box is
> otherwise idle; `--workers 1` to give the CPU back.

`--val` and `--install` are separate commands, so the measurement reaches the weights through a
file in the run directory (`val_metrics.json`). Each measurement records a **hash of the
checkpoint it measured**, and `--install` attaches only the ones that match the file it is
copying: re-training into the same `--run-dir` replaces `best.pt` in place, and a score belonging
to the previous checkpoint must not end up presented as this one's. Installing with no
measurement is allowed and says so on the spot — the panel then shows no score for that weight,
which is the truth rather than a zero.

The export check runs **before** the GPU does, deliberately. A version generated with a short
class list trains a model whose boxes come back under the wrong label — no error, plausible
confidences, and only visible as bad detections in the app. `--export-dir` also has to point at
the unzipped folder, since the export's own `data.yaml` uses paths relative to wherever Roboflow
expected the archive to be extracted; the tool writes a `data.scanncart.yaml` beside it from
absolute paths rather than assuming a layout.

It also **refuses** to train when the export's classes and §8.1's roster disagree, and reports
only the two aggregate numbers a training log can answer — per-class recall needs the separate
validation pass that `--val` runs, because a passing average can hide a failing class.

The command it performs, for reference:

```bash
yolo detect train \
  model=yolo11s.pt \
  data=path/to/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  patience=25 \
  project=runs/scanncart \
  name=scanncart-grocery-v2
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

`train_model.py --yes` reports the first two, and only those, because they are the only numbers a
training log contains. **`train_model.py --val` reports the third**: it runs a validation pass and
prints per-class recall against the 0.85 floor, with the instance count beside each number
(`milo 0.950 >= 0.85 (n=12)`), so the verdict names the class rather than the average — a model
can clear both aggregates while failing one distance's worth of cells, which is the thing this
dataset exists to fix.

Three outcomes, not two. A class **below** the floor is a miss; a class the split holds no
instances of is reported as `[SKIP]`, not as 0.000 — one means add images of that item, the other
means add captures to that split, and the two are opposite instructions. `--val` reads the `test`
split by default, and `plan_split.py --holdout-session s3` is what makes that an unseen-session
number rather than held-out frames of a session the model trained on; `--split valid` measures the
split the run *selected* on, which is a selection number and not an acceptance one.

**The same recall is reported per distance**, because that floor is computed over the whole split
and therefore averages `close`, `mid` and `far` together — so a class whose test frames happen to
be mostly `close` can clear 0.85 while barely finding the item at `far`. `--val` runs three more
passes, one per distance, and prints a class × distance grid beside the per-class table:

```
class                                     close          mid          far          all
---------------------------------------------------------------------------------------
555 sardines 155grams                  0.94 (12)            -     0.61 (28)!    0.88 (40)
```

`!` marks a cell under the floor, `-` marks a distance that held no instances of that class (the
same third state as above), and `all` is the split's own number — so the miss is read next to the
average that hid it, and the misses are also named in one line under the table.

Two things about how it is measured, and one about what it means:

- **It costs three extra validation passes**, not a copy of the dataset. Ultralytics accepts a
  *directory* or a **file of image paths**, so each distance is one pass over a list of the frames
  carrying that tag; the lists and their `data.yaml`s are written into `val-by-distance/` in the
  run directory, so a surprising row can be traced to the exact image set behind it.
  `--no-per-distance` skips them, and a run whose manifest is missing says the breakdown was
  *skipped* rather than quietly reporting nothing.
- **Each distance gets the same class list and the same dataset root**, written by the same
  function the run's own `data.yaml` comes from — a distance pass reading different `names` would
  score the model against labels it never trained on, and nothing in the output would look wrong.
- **Distance is a tag, not a class.** The class list is the same eight names at every distance: a
  `Palmolive Naturals Bar Soap 85g` box is that class whether the bar is 20 cm or 2 m from the
  camera. Nothing here produces a `palmolive close` class, and the trained model has **eight**
  outputs, not twenty-four. `Palmolive @ close` is a *cell* — a unit of coverage and of work — and
  it is what the reports and the capture folders are organised by.

The numbers are **written down, not just printed** — into `val_metrics.json` in the run, then
into the weights' record by the following `--install` — and the Admin Panel's Model field shows
them under whichever weight is selected: the split, the floor, the aggregates, and every class
with its instance count. Measuring the other split *adds* a block rather than replacing the `test`
one, with `test` first, so a `--val --split valid` cannot quietly swap an acceptance number for a
selection one. The `[SKIP]` case survives the trip as `recall: null`, which is what keeps a class
the split never asked about from arriving in the panel looking like a total miss.

One thing that reading has to get right, and does not announce itself: ultralytics indexes recall
positionally against the classes the split actually contains, so a class missing from the split
shifts every row after it. Reading `names` and `recall` as parallel lists would print one class's
recall under another class's name — no error, plausible number, and with a per-distance holdout it
is the normal case rather than the edge one.
| Confusion matrix | Low off-diagonal mass between similar SKUs; low "background" column (false positives) |

#### When the miss is the second object

`--val` answers *which class is short*. It cannot answer *which object was missed*, and on a
crowded frame those are different defects pointing at opposite fixes: a class at 0.75 reads as
"shoot more of this item", while the same number read per instance can say "this item is never
missed when it is alone, and only the second one in a frame is lost" — which is a capture-plan
finding, not a coverage one.

`audit_recall.py` measures that. It matches per instance the way the validator does — class-aware,
best IoU first, one prediction per label — but at the app's operating `conf_threshold` instead of
the best-F1 threshold, then splits the result by how many objects each frame's labels carry:

```
sidecar/.venv/Scripts/python.exe sidecar/tools/audit_recall.py --generation v1

class                                             inst  found  recall   med hit  med miss
555 sardines 155grams                               97     73   0.753    0.4151    0.0754  <-- below 0.85
century_tuna_flakes_in_oil_155_grams                50     50   1.000    0.6482    0.0000
TOTAL                                              401    368   0.918

crowding: instances found / labelled, and frames that came back complete
single  1 object      265/265   100.0%   frames 265/265  100.0%
multi   2+ objects    103/136    75.7%   frames 36/62    58.1%
```

Five readings, and the last two are the ones that decide what to do:

- **The crowding pair is the headline.** A gap between the buckets means instances are found one
  at a time and lost when several share a frame; no gap means the misses are spread across frames
  and the fix is more shots of the classes below. The tool prints that as a sentence rather than
  leaving the reader to infer it, and it **refuses to compare** when either bucket holds fewer than
  20 instances — `--limit 20` would otherwise print a confident verdict off two objects.
- **The area columns** separate "cannot see the object" from "cannot see the second one". A class
  whose missed instances are a fraction of the area of the ones it found has a small-object
  problem; equal areas on both sides means size is not what separates them.
- **Distance is not the axis here; crowding is.** Distance is a Roboflow *tag* and needs a manifest
  (`--val`'s per-distance grid). Crowding comes free from any export's labels, so this works on v1,
  which predates the tagging entirely.
- **`--conf-sweep` turns a miss into a slider.** A miss ranked below `conf` is not a model gap, and
  `conf_threshold` is hot-reloadable — on v1 the crowded bucket reads 75.7% at the shipped 0.5 and
  89.0% at 0.1, while the single bucket sits at 100.0% throughout. That is most of the gap turning
  out to be a threshold rather than a capability, which `--val` cannot show at any setting.
- **`--iou-sweep` tells a suppressed box from an unseen one.** Two tins side by side overlap, and
  ultralytics' NMS default (0.7) merges boxes overlapping more than that — so a *detected* second
  tin can be discarded inside the model. A large recovery at 0.9 means that share of the crowded
  recall is an inference setting; on v1 there is none (365/368/368), so the misses are real.
- **On v1 the whole miss is crowding, and nothing else.** Every one of the 265 single-object frames
  in the test split had its instance found; the recall the floor is computed over (0.918) is the
  crowded bucket pulling it down. One class is still under the floor — `555 sardines 155grams` at
  0.753, the tin most easily confused with the one beside it.

It reads the generation's dataset and class list from `generations.py`, so it cannot be pointed at
one generation's images while judging the other's names, and it reports any predicted class that is
not on that list — the same 24-output mistake the runtime catches, caught at the measuring step.

One thing it has to get right that is easy to get wrong, and did: **a label line is one of two
forms, told apart by field count.** `cls cx cy w h` is a box (centre plus size, five fields);
`cls x1 y1 x2 y2 ...` is a polygon (seven or more, reduced to its bounding box). They need
different maths, and applying the polygon rule to a box is not a near miss — it places a real
instance in the wrong part of the frame, and collapses to zero area wherever the width is smaller
than the centre. Nothing in the output announces it: it simply reads as a class the model cannot
find. On v1 that misreading understated three classes (Bear Brand 0.600 for a true 1.000, silver
swan 0.851 for 0.959) and made the overall number 0.840 where it is 0.918. Ultralytics converts
both forms itself, so **`--val` is the reference to check this against** — if the two disagree on
a class by more than a threshold's worth, the parser is the suspect, not the model.

#### Do the PRD's targets hold

Recall is one of three promises. The PRD also asks for **≥ 30 fps processing** (§5, restated in §7)
and **end-to-end latency < 150 ms** (§6), and nothing above measures either — a weight can clear
every floor here and still be too slow to run, which is the one failure that makes the rest of the
numbers unusable in a live capture.

`spec_check.py` measures all three through the app's own code, on real test-split frames resized
to the configured capture geometry:

```
sidecar/.venv/Scripts/python.exe sidecar/tools/spec_check.py --generation v1
sidecar/.venv/Scripts/python.exe sidecar/tools/spec_check.py --generation v1 --defaults
sidecar/.venv/Scripts/python.exe sidecar/tools/spec_check.py --generation v1 --strict

config     shipped defaults  (Settings(), as built)
device     cuda:0   capture 640x480   imgsz 640   conf 0.5   preview_height 720   frame_skip 0

isolated   YoloDetector.infer over 80 frames, wall clock
           mean    24.7  p50    23.6  p95    34.0  max    52.7  ms
in-app     Pipeline.process_once over 80 frames, wall clock - what a frame costs the sidecar
           mean    24.5  p50    23.3  p95    35.0  max    49.9  ms
           the pipeline's own latency_ms stat: mean    23.3 ...   <- detection only

  [PASS] isolated infer fps                    40.5 fps   >= 30   (PRD 5, 7)
  [PASS] in-app pipeline fps                   40.7 fps   >= 30   (PRD 5, 7)
  [PASS] in-app mean latency                   24.5 ms    < 150   (PRD 6, 7)
  [PASS] in-app p95 latency                    35.0 ms    < 150   (PRD 6, 7)
  [PASS] instance recall at app conf          0.918 ratio >= 0.9   (PRD 7)
```

Three things about that reading:

- **Two speeds and two latencies, because they are different instruments.** `isolated` is the
  detector's own cost; `in-app` is what a frame costs the sidecar once tracking, the allowlist
  filter, logging, the 720p JPEG encode and the message build are inside it. An isolated figure
  alone would pass a detector sitting inside a pipeline that cannot hold 30 fps, and a mean alone
  would hide the frame that stutters — which is what the p95 column is for.
- **The pipeline's own `latency_ms` is not the end-to-end figure.** It is `t1 - t0` around
  `detector.infer`, so it stops before the encode: 23.3 ms where the frame really costs 24.5 ms.
  The headroom is wide enough that it changes no verdict, but reading that stat as the frame cost
  understates a Live frame by about a tenth.
- **It reports one accuracy number on purpose.** PRD §7 asks for ≥ 90% on common grocery items, so
  that is what is checked; the per-class and per-crowding breakdown is `audit_recall.py`'s, and
  this tool calls that module's matcher rather than growing a second implementation of it.

Accuracy here is per-instance at the app's `conf_threshold`, so it is the recall a Live view
actually experiences rather than an available maximum, and the record's own `--val` aggregates are
printed beside it with the note that they come from a different operating point — the two are
allowed to differ, and the conf sweep in `audit_recall.py` is what shows the slope between them.
`--strict` exits non-zero when any target fails, which is what makes this a gate rather than a
report.

#### The configuration line, and why v1 was failing without it

By default the settings come from `data/settings.json` — the profile the app on this machine
actually runs — and the report names the file it read. That default exists because of what it
found. Measured against v1's own geometry the weight clears all five targets; measured against
this machine's saved profile, which sets `imgsz` 960, it fails three:

```
config     saved settings    (sidecar/data/settings.json)
device     cuda:0   capture 640x480   imgsz 960   conf 0.5   preview_height 720   frame_skip 0

  [FAIL] isolated infer fps                    27.6 fps   >= 30   (PRD 5, 7)
  [FAIL] in-app pipeline fps                   24.1 fps   >= 30   (PRD 5, 7)
  [PASS] in-app mean latency                   41.5 ms    < 150   (PRD 6, 7)
  [PASS] in-app p95 latency                    53.9 ms    < 150   (PRD 6, 7)
  [FAIL] instance recall at app conf          0.344 ratio >= 0.9   (PRD 7)
```

`audit_recall.py` confirms the accuracy half in one run each: **0.918 at `--imgsz 640` against
0.344 at `--imgsz 960`**, with every class below the floor at 960 instead of one, and
single-object frames collapsing from 265/265 to 55/265. The pattern names the cause — the
close-up objects, already at their largest scale at 640, get pushed past what the head can
localise, while the smaller crowded-frame objects survive better (61% against 20.8%).

The rule this leaves behind: **`imgsz` is a per-weight inference requirement, not a free knob.**
Unlike `resize_mode` it is not recorded in `models/<stem>.json`, so a profile tuned to 960 — which
v2's 8-class set may well want, for objects twice as far away — silently destroys a weight trained
at 640, and the Admin panel has nothing to contradict. Until the record carries it, this tool is
what catches the pair, which is why the configuration line is printed rather than assumed.

---

#### What to do about a class below the floor

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

1. Install the trained weights into `sidecar/models/` with the generation's name, e.g.
   `scanncart-grocery-v2.pt` — see §8.2 for the naming rule, and `sidecar/models/README.md` for
   the short version of it. `train_model.py --install` does the copy and **refuses to overwrite an
   existing weight** (the picker is keyed by filename, so an overwrite silently replaces the
   model a running app is configured with). Nothing else is needed on the sidecar side:
   `is_custom_model()` accepts any `.pt`/`.onnx` directly under `models/`.

   It also writes **`models/<name>.json`** beside the weights, recording the `resize_mode` this
   generation requires (from `generate_version.REQUIRED_RESIZE_MODE`, so the requirement and the
   preprocessing that produced it cannot drift apart). That record is the only place the
   requirement can come from: a `.pt` records the training run, not the dataset geometry, and the
   filename is a convention. The same file records the export's **`class_names`** — what these
   weights predict. A `.pt` keeps no label set, so a version generated from a distance-split class
   list trains 24 outputs and nothing in the file says so; written down, the Model field's *Weights
   on disk* list flags it against the roster (§8.1) **before the weight is ever selected**. Pass
   `--version <n>` to the install so the record can also name the dataset version the weights came
   from.
2. Select it in the Admin Panel's Model picker, which lists **whatever the sidecar reports under
   `sidecar/models/`** — no renderer edit, and no shipping-notes comment to keep in sync. A model
   that is selected but whose file is missing stays selectable (you should be able to see what a
   broken config says rather than have it silently vanish), and `POST /api/detector/probe` reports
   the missing file. The same field lists every installed weight with the mode it needs and notes
   that `auto` uses it, so what is on disk and what each one expects are both visible before
   anything is selected — or, for a weight with no record, which way `auto` will guess. It shows
   how many classes each weight predicts, and flags any whose recorded class list is not the
   8-name roster (§8.1) — the 24-class failure, from its listing rather than from a run. Under the
   selected weight it also shows **what that model scored** (§6's per-class recall), read from the
   record rather than from a terminal you may not still have open.
3. **Leave `resize_mode` on `auto`.** The record written in step 1 is what `auto` consults
   (`app/models.py`'s `requirement_for()` → `resolve_resize_mode()`), so the weights are run at
   their Stretch-trained geometry without anyone setting the field — before that record existed,
   `auto` answered *letterbox* for a local `.pt`, the wrong geometry for a Stretch-trained
   version, silently. Setting `stretch` by hand is equivalent; setting anything *else* overrides
   the record, and the Model field flags it. Weights with **no** record are listed with what
   `auto` gives them (`auto gives letterbox`) rather than a `stretch` it cannot know — the guess
   and a fact must not render alike — and, since nothing about that case *contradicts* anything
   for the Model field to flag, the sidecar reports the assumption itself (`resize_guess`), as a
   structured entry the Admin Panel renders above the form **with a `Record it now` button** — and
   which the Live view renders too while a capture runs, button included, since that is the screen
   where the weaker `far` detections show up and the fix needs no restart. That
   button is the one thing this field can do about its own warning: it writes the mode `auto`
   resolved to — letterbox here — into `models/<stem>.json` via `POST /api/models/record`, which is
   also the shorter path for weights `--install` never installed. Because the mode is the one
   already running, clicking it cannot change what the detector does; it turns a fallback into a
   stated fact, and it is an *attestation* rather than a measurement, which is why `--install`
   (requirement + version + `--val` numbers) stays the writer to prefer when the run still exists.
4. `active_model` is a **restart-required** field (`RESTART_REQUIRED_FIELDS`), so capture must be
   stopped before saving.

Two things the old whitelist path silently decided for you:

- **`resize_mode` is requirement-first, format-aware second, and never silent third.** `auto`
  uses the mode recorded beside the selected weights, and only falls back to the format heuristic
  — `stretch` for a custom `.onnx` (a Roboflow export, trained stretched), `letterbox` for a
  custom `.pt` (a locally trained checkpoint, trained letterboxed) — when there is no record.
  This guide trains from a `Stretch to` version, so the heuristic's answer for *its* weights is
  wrong; the record is what makes `auto` right, which is why `--install` writes one and why the
  capture path reads it (`requirement_for()` in `sidecar/app/models.py`). The fallback is a guess
  rather than a fact, so it is *reported* rather than left to be noticed: a native capture on
  `auto` with an unrecorded custom `.pt` gets `resize_guess`'s entry, which names the assumption,
  the `--install` command that removes it, and the panel button that records the same fact from
  what the operator knows. See `resolve_resize_mode()` / `resize_guess()` in
  `sidecar/app/settings_store.py`, `record_requirement()` in `sidecar/app/models.py`, and
  `sidecar/models/README.md`.
- **GPU.** A `.pt` runs on torch directly, so `device: "auto"` resolving to `cuda` is the fast
  path (docs measured ~25 ms/frame in-app on a GTX 1050 Ti, vs ~91 ms for the custom ONNX on
  CPU). The CUDA requirement is on the torch install, not on onnxruntime — that only matters
  for `.onnx` models (see `docs/DETECTOR_BACKENDS.md §1a` and `sidecar/requirements-cuda.txt`).

### Post-integration tuning

| Setting | Note |
|---------|------|
| `conf_threshold` (default `0.5`) | A custom 8-class model is usually more confident than the COCO baseline. Try `0.6` to cut flicker. Restart-required. |
| `track_expiry_s` (default `1.5`) | Raise if items briefly drop out and re-enter as a new `track_id` (causing duplicate log rows). Hot-reloadable. |
| `infer_frame_skip` (default `0`) | Only touch if fps is short of target. Hot-reloadable. |

---

## 8. This project's convention — roster, versions, splitting

Sections 1–7 are the general method. This section is the concrete convention the SCANnCART
datasets follow, so the guidance and the project cannot drift apart: the roster is fixed before
anyone labels, the weight filename carries the model generation, and the capture batch is the
split key.

#### The toolset, and where its data lives

The tools referenced below are tracked code in `sidecar/tools/`, and their tests run in the normal
sidecar suite (`make test`), so renamed classes or a stray folder mapping fail there rather than
silently producing mislabeled data.

Their **data is not tracked and does not live beside them.** Staged images, the manifest, the split
plans, the upload resume state and the `.env` holding `ROBOFLOW_API_KEY` all live in a dataset
*workspace*, **`sidecar/data/datasets/`** — gitignored, and the same directory the hard-negative
capture lives in. That split is what keeps the code reviewable: gigabytes of JPEGs and an API key
stay out of the tree. Point `SCANNCART_DATASET_ROOT` elsewhere to relocate it
(`sidecar/tools/workspace.py` is the single source of that path). Every path inside the workspace is
relative to its output directory, so moving the whole thing together is safe, and moving pieces of
it is not.

One thing that used to live there is gone on purpose: v1's local image set (`cleaned/`, 5 GB of
images with no labels) was deleted to reclaim disk on a drive that was 87% full.
`sidecar/tools/workspace.py` records both recovery routes — an export from the
`scanncart-grocery` project (v1's 1,516 annotated images, one generated version), or v1's six raw
per-class capture archives (1,501 images, 3.9 GB), now plain files in
`sidecar/data/datasets/v1-source-zips/`.

Those zips used to be held by the git tag `archive/scanncart-v1-source-zips`, pinned to an
abandoned commit so `git gc` could not expire them. That was a bad trade and it has been undone:
verified out of the object store first, then pruned, which took `.git` from **3.9 GB to 60 MB**
(it is 1.000x-compressible JPEG data, so nothing was ever going to shrink it in place). At 3.9 GB
they are the largest single reclaim left on the disk, and their `README.md` lists what is in them
and how to delete them if an export covers you.

#### How the data is uploaded: one project, three tags, and a session

Asked whether the uploads should be nested into folders instead, the honest answer is that the
platform does not offer it: Roboflow's folders group **projects**, not images inside one, and the
forum answer to exactly this question is "Roboflow doesn't support this kind of folder structure".
Inside a project the per-image primitives are **tags**, **batches** (upload grouping) and
**annotation attributes** — and attributes attach to a *box*, so they cannot carry a distance that
belongs to the whole frame. Flat project plus tags is not a compromise; it is the intended shape.

What this project puts in each:

| Primitive | Value | Why there and not elsewhere |
|---|---|---|
| Project | `snc-grocery`, object detection | One detector, one class space. Per-distance or per-class projects would fragment the confusable pairs (`555 sardines` vs `century tuna`) into sets no single model has to tell apart, and would have to be merged back before training. |
| Batch | `<class>_<distance>` (`bear-brand-milk_mid`), suffixed `_<session>` for later sessions | The unit a split is assigned to. A batch is one (capture session, class, distance) bucket. |
| Tag ×2 | the class slug and the distance | The *filter*. `tag:milo tag:mid` selects a cell; there is no `class:` filter until the images are annotated, so during labeling the tag is the only handle. Also the provenance `sanity` checks the class roster against. |
| Tag ×3 | the capture session (`s1`, `s2`, `neg1`) | See below. |
| Split | train / valid / test, assigned per image | Never set at upload; §8.3 owns it. |

**Why the session is a tag and not just metadata.** Roboflow's documented dataset-search filters are
`tag`, `filename`, `split`, `class`, `date`, `like-image`, `job` and min/max width/height/annotation
counts — **there is no `batch:` filter**. A batch can group images in the UI but cannot *select*
them, so if the session lived only in the batch name there would be no query for "everything from
session 2", and no way to check whether train and test share a session. The session is therefore
stamped as a third tag by `retag`, and it is what makes the leakage check in §8.3 expressible:

```
tag:s2                  every frame from the second capture session
tag:s1 -split:train     frame from s1 that train does NOT have - the leak, if any
tag:milo tag:mid tag:s2 exactly the batch, when a cell has been shot twice
```

Two mechanical notes, both from doing it the hard way. Only **one** tag survives the upload call (a
repeated form field keeps its last value), so `tags[0]` is always the distance and the class and
session are applied afterwards by `retag`. And `batch` is silently ignored as a multipart form
field — it must go in the querystring, or every image lands in "Uploaded via API" with the right
name and the wrong batch.

The session defaults to `s1`, and a manifest written before sessions were recorded carries no
`session` field at all: `plan_split` reads those as `s1`, which is true for the v2 set. That default
is an assumption rather than a fact, so `sanity` verifies it against the tags on the images —
`[.ok.] capture session: s1`, or a warning naming the session tag that is missing.

### 8.1 Class roster — 8 classes, defined once

The class list is **project-wide**, not per version, and lives in the project's
**Settings → Classes** tab. Set it before labeling starts, then turn on **Lock Classes** so a
labeler cannot add a ninth by typing one. Renaming a class later rewrites every annotation that
used it, and deleting one deletes those annotations — neither is reversible.

| # | Class name (use exactly this) | v1 count | v2 upload tag | v2 images close / mid / far |
|---|-------------------------------|---------:|---------------|------------------------------|
| 1 | `Bear Brand Fortified Powdered Milk 33g` | 310 | `bear-brand-milk` | 106 / 23 / 22 |
| 2 | `lucky_me_pancit_canton_calamansi_flavor` | 352 | `lucky-me-pancit` | 234 / 19 / 13 |
| 3 | `555 sardines 155grams` | 479 | `555-sardines` | 34 / 17 / 28 |
| 4 | `century_tuna_flakes_in_oil_155_grams` | 302 | `century-tuna` | 150 / 0 / 0 |
| 5 | `silver_swan_sukang_puti_200ML` | 302 | `silver-swan-vinegar` | 242 / 5 / 35 |
| 6 | `Milo Chocolate Drink 22g Sachet` | 68 | `milo` | 65 / 34 / 43 |
| 7 | `safeguard_pure_white_60g` | 0 | `safeguard` | 184 / 41 / 54 |
| 8 | `Palmolive Naturals Bar Soap 85g` *(new to v2)* | — | `palmolive` | 0 / 13 / 21 |

Rows 1–7 are the **existing v1 class names, copied verbatim**. That is the whole point of the
table: v1's 1,815 images and v2's are then the same classes, so the two sets can be merged or
compared instead of forming two disjoint models. "v1 count" is Roboflow's per-class count in the
`scanncart-grocery` project.

Row 8 is genuinely new — v1 has no Palmolive — so there was no inherited name to reuse. It follows
the same `<brand>_<product>_<size>` shape as row 7, so the roster reads as one list rather than
seven inherited names plus an odd one out. It is also the class with no close-range images at all.

The **v2 upload tag** column is what each image actually carries in `snc-grocery`: its class tag
plus one `close`/`mid`/`far` tag. Tags are metadata — they let you pull "every `bear-brand-milk`
image at mid distance" before labeling, and they drive the coverage checks in §8.3. They are
**not** the class list; the class list is set by hand on the Classes tab.

> Roboflow recommends alphanumeric class names with `-` as the only permitted special character.
> v1's names already contain spaces and are baked into 1,815 labeled images, so v2 keeps them
> verbatim — continuity beats tidiness here.

#### Auto Label descriptions — the prompt for each of the 8 classes

Auto Label (Roboflow's AI labeling) searches with a **description** and writes what it finds under
the **class name**, so the two are separate fields and this table is the first of them. The
description is a *prompt*: it describes how the object looks, not what the class is called.

Three rules shape the wording, and all three come from how these models behave:

- **Package type first.** `tin`, `foil sachet`, `bottle`, `bar in a wrapper` are the strongest
discriminators in this roster and they survive blur at `far`, where a colour word does not.
- **Visible brand text is a feature, not decoration.** The wordmark on the label is something a
  foundation model can read, and it is what separates the two pairs this dataset is most likely
  to get wrong (see §4): `555 sardines`/`century_tuna` (both tins) and `lucky_me`/`milo` (both
  small green packs).
- **No negations.** "not a sardine tin" is not a prompt; the positive form ("tuna flakes in oil")
  is. Every class needs its own description — Auto Label **rejects** a preview or a job when two
  classes share one.

Colours below were measured from the staged images rather than from memory: for each class, the
hue of the *saturated* pixels (branding and packaging, ignoring the counter) across 15 frames.
Where a class's frames are one colour edge to edge — `silver_swan` and `lucky_me`, whose backdrop
is green in 15/15 — the measurement is the backdrop and no colour claim is made.

| Class name (select this) | Auto Label description (paste this) | Why it is worded that way |
|---|---|---|
| `Bear Brand Fortified Powdered Milk 33g` | `a small foil sachet of powdered milk, white with blue branding and a bear logo` | Sachet, not a tin. Blue in 15/15 frames. The bear is a shape anchor that survives `far` better than the lettering |
| `lucky_me_pancit_canton_calamansi_flavor` | `a pack of instant pancit canton noodles with LUCKY ME branding` | Separates it from `milo` by *contents* (noodles) rather than by colour. No colour word on purpose: this class's frames measure green backdrop in 15/15, so a packaging colour here would be a guess — and it is the half of the pair that needs none, since "instant pancit canton noodles" is not a phrase a chocolate drink matches |
| `555 sardines 155grams` | `a flat oval tin of sardines with a red label and the number 555` | Red is measured on the tin itself (centre tiles, yellow at the rim). Oval separates it from `century_tuna`'s round tin |
| `century_tuna_flakes_in_oil_155_grams` | `a flat round tin can of tuna flakes in oil with a yellow and blue label, CENTURY TUNA` | "Round" + "flakes in oil" against `555`'s oval sardine tin. Yellow/orange dominated these frames (12/15); confirm the label colour by eye |
| `silver_swan_sukang_puti_200ML` | `a small plastic bottle of white vinegar with a SILVER SWAN label` | The only **bottle** in the roster, which is the whole discriminator. No colour word: this class's frames are green backdrop in 15/15, so packaging colour is unmeasurable from them |
| `Milo Chocolate Drink 22g Sachet` | `a small foil sachet of chocolate malt drink mix, green MILO packaging` | Green in 14/15. "Chocolate malt drink mix" is the pair-breaker against `lucky_me` |
| `safeguard_pure_white_60g` | `a white rectangular soap bar in a blue and white wrapper, SAFEGUARD` | Blue in 15/15, and "white bar" comes from the class name. Contrasts with `palmolive`, the other soap |
| `Palmolive Naturals Bar Soap 85g` | `a rectangular bar of soap in a pink wrapper, PALMOLIVE Naturals` | The one line worth eyeballing: pink/purple led in 9/15 frames with orange in 6/15, so if your bar's wrapper is the orange variant, swap `pink` for `orange` |

**Do not give the `negative` batch a description.** It is a batch key rather than a labelable class
(§8.1, `label_classes.PSEUDO_CLASS_SLUGS`): those frames are supposed to contain *no* roster object,
so a labeler's job is to mark each one **null** (`N`) — and auto-labeling them would do the
*opposite*, filling the hard-negative set with boxes. Auto Label runs on a batch, so the way to
keep them out is to run the job on the product batches and leave `negative_neg1` alone.

Two operational notes that decide whether this is worth doing at all:

- **Free to try, then decide.** "Generate Test Results" runs the model on a handful of images and
  does **not** consume credits; the confidence threshold you set there is the one the batch job
  uses. Measure on the `close` cells first: those are large, unambiguous objects, and they are
  where a foundation model's boxes are most reliable. On the `far` cells (objects 20–40 px across)
  it is weakest — exactly where a wrong box costs the most, since `far` is the bucket this dataset
  exists to fix.
- **Review is not optional, and the progress tool only half-covers it.** `label_progress.py`
  compares the drawn class against the class tag, so a *wrong class* is caught automatically; a
  **wrong box** is not visible to it. Pre-labels are a head start on drawing, not a substitute for
  looking.

#### Getting the exact names onto the images: `sidecar/tools/label_classes.py`

The class names are inconsistently formatted (`555 sardines 155grams` vs
`century_tuna_flakes_in_oil_155_grams` vs `silver_swan_sukang_puti_200ML` are three conventions for
the same idea), so applying them by hand to 1,383 images drifts. The tool removes that drift, and
two hard limits shape what it can do:

- **A tag cannot carry the class name.** The tag API rejects spaces:
  `Valid characters: a-z A-Z 0-9 -_:/.[]<>{}@, length 1-64`. Three of the seven v1 names contain
  spaces, so the tags keep the short slug — which is also what keeps `tag:` filters and the batch
  keys working.
- **A class's order, and Lock Classes, cannot be set by API.** Those live on the project's
  Settings → Classes tab only, so the tool *emits* the list in the order to create it.

Class *creation* is separate, and it does work by API — undocumented, but verified: annotating a
throwaway image with an unknown class name registers that class, and the class survives deleting
the image. `--create-classes` does that for any name that is missing.

What it does besides: writes `class_name` **image metadata** — the exact v1 name, which has no
character restriction and round-trips intact — so the authoritative name travels with the data and
any later export or audit needs no slug lookup table. It also verifies the mapping against the
**live** v1 project, so a typo here cannot silently break continuity.

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/label_classes.py           # report + class list
sidecar/.venv/Scripts/python.exe sidecar/tools/label_classes.py --apply   # write the metadata
```

Palmolive is the one class v2 adds, so it had no v1 name to inherit. Until the variant was
confirmed its 34 images were deliberately left carrying only their `palmolive` slug rather than a
placeholder — baking in a name that is about to change is worse than reporting a gap. It is now
named (`Palmolive Naturals Bar Soap 85g`) and behaves like every other class: `--apply` writes it to
those images and `--create-classes` creates it in the project.

#### Distance words stay out of the class list — and the tools check

The coverage tables write a cell as `Palmolive Naturals Bar Soap 85g @ close`, and the capture
folders on disk are `PALMOLIVE/CLOSE/`. Neither is a class. Distance is a **tag** on the image
(§8.3) and a **cell** in these reports; the class list is the 8 product names above and nothing
else.

Getting that wrong is quiet rather than loud. A project whose classes are `… close`, `… mid`,
`… far` trains a head with 24 outputs instead of 8, and every box then comes back labelled with a
name the app's own roster does not contain — with no error anywhere, because nothing about a class
*name* says what it is. So it is checked everywhere a class list is visible:

| Where | What it sees | Why there |
|---|---|---|
| `clean_v2.py sanity` | the **live** project's class list | the check §9 already gates a shoot on, so it is the earliest point — before any frames are shot, and blocking (`sanity` exits 1) rather than a warning |
| `label_classes.py` (any run, `--apply` or not) | the **live** project's Classes tab | the same project, in the run that also writes the names and the tags |
| `generate_version.py --yes` | the **live** project's class list, at the moment of generation | the last point where the fix is free: it refuses before the `POST`, so a bad list costs no version number. Fail-closed (exit 2), nothing generated |
| `train_model.check_export` | the class list a **generated version** was built with | before the GPU runs, and before the download is trusted — otherwise the mistake costs a training run on top of the version number |
| `train_model.py --install` → `models/<stem>.json` → the Admin Panel's *Weights on disk* listing | the class list **recorded beside the weights** | before the weight is even selected. Same judgement (`app/roster.py`), the earliest moment it can be made: the export is the only thing that ever knew the list, and it is gone by the next session |
| `app/roster.py` (Test connection in the Admin Panel, **and the Live view while capture runs**) | the class list a **weight already predicts** | the last line, and the only one that can catch a model that arrived from somewhere else entirely. It warns rather than blocks — a loaded model cannot be argued with — and it is the only check that can run at all here, since a `.pt` records no roster and the export is gone |

The last two rows are also demonstrable on demand rather than on trust: `node
.claude/skills/run-desktop/driver.mjs classlist` builds a genuine 24-output scratch weight, reads the
flag off the Admin listing and the sentence off the running app's banner and chip, then deletes the
weight and puts the settings back (`RUN_SHEET.md` §8).

The last row is not a check you have to remember to press. The **running app reports it by
itself**, because a probe is only run by someone who already suspects the model — and the case this
exists for is a weight that arrived from somewhere else and looks fine everywhere else. `Pipeline`
calls `on_class_list` **once**, the first time the detector's names are knowable, and `main.py`
turns that into `class_warnings` on a `StatusMessage`: broadcast to whoever is watching, and carried
in the WebSocket **handshake** as well, so a renderer that reloads mid-capture learns it rather than
having to infer a broken model from the labels going past. The Live view renders the sidecar's own
sentences (`data-testid="live-class-warnings"`) — replaced on each status, never merged, and not
styled as an error, because a capture running a 24-class weight is not failing, it is producing the
wrong three rows for every item. Beside that banner the same strip carries the running model's class
**count** in a chip of its own — `classes · roster ok`, or `classes · 1 finding` in amber — because the
count is the one thing the banner cannot say, and it is what makes `24` where `8` was expected legible
as a number. The chip renders nothing until the sidecar has read a model's vocabulary: `0 classes`
would be a claim about a head that has not spoken yet.

**And one step earlier still: the record.** A `.pt` records no roster, so a weight that is never
run and never probed has nothing to judge — which is why `train_model.py --install` writes the
export's `class_names` into `models/<stem>.json` beside the requirement. `app/models.read_record`
reads it and `installed_models` judges it with the *same* `roster.class_list_problems`, so the
Admin Panel's *Weights on disk* list carries the same sentences the probe and the Live view do,
before the model is selected — let alone started. It is the same fact as the probe's, from the
only other place it can come from: the install is the moment the export is still open, and the
export is the only thing that ever knew the list. Recorded as an **empty list when nothing is
known** (a hand-copied weight, or a record written before this field existed) rather than written
as "no classes" — `class_list_problems([])` would otherwise read as a weight that predicts none
of the 8, which is a verdict about a list nobody has seen.

**Why it cannot be checked at start.** `YoloDetector.names` is deliberately empty until the first
inference — reading it at construction would build a second ONNX session — so a start-time check
would announce "this model predicts none of the 8" for a perfectly good weight. The names are only
knowable *after* an inference, so that is where the check lives; a remote detector fills `names`
from its first response, so both backends report through the same path.

All four build-time checks call the same predicate, `label_classes.distance_tokens_in()`, so they
cannot disagree about the same name; the runtime copy is held to it by a drift guard
(`sidecar/tests/test_roster.py`). The runtime check is also the one that reports the *opposite*
failure — a model that can only predict some of the 8 — because nothing it does predict is wrong,
so nothing else would ever notice. It matches **whole tokens** against a fixed word list (`close`,
`closeup`, `mid`, `middle`, `far`, `near`, `distance`), which is what keeps a legitimate name from
tripping it: "Farmer's Choice" tokenises to `farmer`, not `far`. Each finding is phrased as the
*fix* rather than the symptom — the offending classes have to be removed and their annotations
moved onto the product class (a class-list edit alone orphans the boxes), then the version
regenerated, since a version number cannot be reused (§8.2). A drift guard in
`sidecar/tests/test_dataset_tools.py` keeps the word list covering every spelling
`clean_v2.DISTANCE_MAP` accepts, so a new folder spelling without a matching guard entry fails the
suite instead of opening a hole.

The cheapest place to catch it is the `sanity` run §9 already gates a shoot on, which is why the
check lives there as a blocking problem rather than a warning — every other class-list finding is
work *not done yet*, while a distance in a class name is work done in the wrong shape. Keep **Lock
Classes** on as well, so a ninth class cannot be typed into existence between runs.

### 8.2 Version numbers and weight naming

Two different numbers are in play, and they deliberately disagree:

| | Numbering | v1 | v2 |
|---|---|---|---|
| **Roboflow project version** | chronological, per project, created when you generate a dataset | `scanncart-grocery` v1 — 1,815 images (train 1,265 / valid 223 / test 327) | `snc-grocery` **v2** — v1 was burned emptily by accident, see the note below |
| **Model generation** | the `-vN` suffix on the weights file, yours to choose | `sidecar/models/scanncart-grocery.onnx` (unsuffixed baseline — the Roboflow export), or `scanncart-grocery-v1.pt` once §6's local v1 build has run | `sidecar/models/scanncart-grocery-v2.onnx` (or `scanncart-grocery-v2.pt` from a local run) |

- Name weights `scanncart-grocery-v<N>` — `.pt` when trained locally, `.onnx` when exported from
  Roboflow. The suffix is the **model generation, not the Roboflow version**: v2's weights come
  from `snc-grocery` *version 1*. Say which number you mean.
> ⚠️ **`POST /:workspace/:project/generate` generates a version.** It is not a read-only probe:
> it returns `{"version": "1", "message": "Dataset version generation initialized"}` and consumes
> a version number, even on a project with zero annotated images (which produces an empty, useless
> version). **This actually happened** to `snc-grocery` during setup — version 1 was created empty
> and had to be moved to Trash. **Never call that endpoint to inspect anything.**
>
> **The consequence, stated plainly:** that empty version 1 is in the workspace Trash, and the next
> version this project generates will be **version 2**. So the sentence above — "v2's dataset is
> `snc-grocery` v1" — no longer describes reality; read it as "the first *useful* generation".
>
> What to do about it: nothing, or clean it up deliberately. `GET /:workspace/trash` lists it
> (`{"type": "version", "id": "1", ...}`), the 30-day retention keeps it recoverable, and
> removing it permanently is a Trash action in the UI. Re-deleting by API returns 404 — it is
> already trashed, and the number is not worth chasing. **The convention that carries meaning is
> the weight filename**, `scanncart-grocery-v2.*`; the Roboflow version number is whatever the
> project assigns.

- **Never export over `scanncart-grocery.onnx`.** The Admin Panel's model picker is keyed by
  filename, so keeping both files lets you A/B v2 against the baseline in Live View and roll back
  from a dropdown. Update `CUSTOM_MODEL` / `MODEL_LABELS` in
  `desktop/src/renderer/src/lib/settingsFields.ts` and label it "8 SKUs", not 7. (A `.pt` needs no
  whitelist edit at all — see §7.)
- Match v1's version settings when generating v2: **auto-orient** on, resize **640×640 Stretch to**,
  and no augmentation (v1 used none). This is what §7's recorded requirement is derived from
  (`REQUIRED_RESIZE_MODE` ← `PREPROCESSING`), so `auto` runs the resulting `.pt` stretched. An export
  trained letterboxed instead needs an explicit `resize_mode: letterbox` — and, since a `.onnx`
  carries no record, `auto` would otherwise put it right by accident rather than by design.

### 8.3 Splitting — by capture batch

**The rule protects against a session, not a cell.** A batch named `<class>_<distance>` is what a
split is *assigned* to, but what actually needs holding out is the **capture session**: frames from
one sitting share a rig state, a day and a lighting setup, so train and test sharing a session
measures "recognised the same setup", not "recognised the product". `plan_split` therefore prices a
session spread across splits, prints a session × split table, and states the verdict in words. With
the single session this set currently has, every assignment that uses all three splits pays the same
penalty and the plan is unchanged - the report says so, in those words, rather than implying an
unseen-session number it does not have. A later session that re-shoots cells train already covers is
what converts those readings into a genuine held-out estimate.

Every image is uploaded into a batch named `<class>_<distance>` (`bear-brand-milk_mid`; 21 batches
for v2), suffixed `_<session>` for any later capture so the two never merge. That batch is the unit
the split is assigned to: assign whole batches at dataset generation, targeting 70/20/10, and never
cut one batch across two splits.

Two things to check the moment the version finishes generating:

- **Per-class coverage.** 21 batches will not distribute themselves evenly at 70/20/10, and a whole
  class can land in train only. `century-tuna_close` is a single 150-image batch, and
  `silver-swan-vinegar_mid` is one batch of 5.
- **Distance coverage per split.** Splitting by bucket makes each batch's distance uniform, so
  assigning whole buckets can leave valid all-`mid` and test all-`far`. That confounds distance with
  split and makes the per-distance numbers §3 exists to produce unreadable. Aim for a mix of
  `close`/`mid`/`far` in every split, and filter by the distance tag to verify it.

The near-duplicate collapse in the capture pipeline (extract at ~1 fps, then dedupe) is what makes
this safe: the frames that leak across splits in a naive random split have already been reduced to
one per cluster. If those coverage trade-offs hurt more than they help, a stratified split is
defensible *because* the dedup ran — but do not skip the dedup and then split randomly.

#### Planning the split: `sidecar/tools/plan_split.py`

The paragraph above describes a conflict, so the tool computes **both** sides of it instead of
picking one. It reads the manifest, searches whole-batch assignments (deterministic anneal), and
writes `SPLIT_PLAN.md` plus `split_plan_a.json` / `split_plan_b.json`:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py --commands
```

| | Plan A — strict by-batch | Plan B — stratified per cell |
|---|---|---|
| Rule | whole batches only, §8.3 honoured exactly | each `(class, distance)` cell split 70/20/10 by image |
| 70/20/10 | 969 / 275 / 139 | 969 / 278 / 136 |
| Unmeasurable class-slots | **6** | **0** |
| Cells with **no train images** | **10 of 21** | **0 of 21** |
| Session isolation | kept | given up *within* a cell |

That last row is the one that decides which plan can be **trained**, as opposed to merely scored. A
class × distance cell with no train images is one the model is never taught, so a valid/test
reading for it measures the absence of training rather than the model. Plan A cannot avoid this and
should not be made to: with 21 single-session cells, pinning every cell into train leaves nothing
held out at all, which is why the number is **reported** (in `SPLIT_PLAN.md` and on the console)
rather than enforced inside the search. Plan B's 0 is structural - it splits within a cell, so every
cell lands on both sides - which is why Plan B is the training split.

#### The acceptance number: `--holdout-session`

Both plans above share one limitation, and no amount of re-weighting removes it: `test` comes from
the same capture session their `train` came from. Under either, the honest reading of a test number
is *held-out frames*, not *an unseen session*. Plan C is the third plan, and it exists only once a
later session does:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py --holdout-session s3
```

It puts every frame of that session in `test` - whole and unsampled, because one held-out frame in
train re-introduces exactly the shared rig state the plan is removing, invisibly - and splits the
remainder train/valid. It writes `split_plan_c.json` and a per-cell table of what the acceptance
set covers.

**It refuses to produce a plan that would make a cell unlearnable.** If holding the session out
leaves any `(class, distance)` cell with no train images, it prints which cells and exits non-zero
without writing anything: such a plan cannot be trained into a model the number would describe. The
consequence for capture is the whole design of the held-out shoot - **the held-out session must
re-shoot cells an earlier session also covers** (CAPTURE_CHECKLIST.md Tier D), so that `train`
keeps them and `test` is a second look rather than a first one.

The protocol, once a plan is written: generate the version from it, train on `train`, select on
`valid`, and quote `test` as performance on an **unseen capture session** - naming the session,
because "unseen" is a claim about provenance rather than about a percentage. Then leave that
session alone: it is spent the moment a result is tuned against it. Note that train and valid do
share their session(s) under Plan C, which costs model *selection* some of its independence - that
is the deliberate price of a test set that has none, and it is why `valid` is not the number to
quote.

**Use Plan B to evaluate distance robustness** — that is what v2 is for, and under Plan A a weak
mid-distance number cannot be told apart from an unlucky batch assignment. **Keep Plan A for a final
acceptance run**, where the honest estimate on a genuinely unseen session is the number to quote.

Two mechanical facts the plan depends on, both verified against the API rather than assumed:

- **A batch is exactly its two tags.** A batch is named `<class>_<distance>` and both of those are
already tags on every image in it, so the query `tag:bear-brand-milk tag:close` selects exactly the
`bear-brand-milk_close` batch. That is what lets Plan A be applied in the UI with 21 bulk
*Select all matching* → *Change Dataset Split* actions, non-destructively. `SPLIT_PLAN.md` prints
the query for every batch.
- **An existing split cannot be changed by re-uploading.** Re-sending identical bytes returns
  `{"duplicate": true}` and leaves the image on its current split — there is no split endpoint.
  So Plan B (which varies the split *within* a batch, something no tag query can express) requires
  `wipe` then `upload --split-plan`. A wipe deletes every image, so `label_classes.py --apply` must
  be re-run afterwards.

### 8.4 Dataset visibility — `snc-grocery` is public, and that is the plan's default

`sanity` and `upload` both warn that the project is public, and neither warning says "make it
private" any more, because **there is no toggle to flip.** Roboflow's free plan publishes datasets
by default — their docs put it plainly: *"the free Public plan has public data by default… on paid
plans, data is private by default"* — so private is a **plan** property, not a per-project setting.

This matters here because these are captures of a real shop floor with real inventory, not a
sample dataset. The options, in order of preference:

1. **Accept it and label anyway.** Fine if the imagery is not sensitive; the assets are product
   packs and a counter, not people. Note the C2b hard negatives do include hands in frame.
2. **Move to a paid plan** before uploading, if the imagery cannot be redistributed. Do this
   *before* the first `upload`, because a wipe-and-re-upload is the only way back and the plan's
   defaults decide what a new upload is.
3. **Keep captures local** and export a dataset instead — the v1 annotation set is already
   recoverable without Roboflow (§8's workspace notes), so labelling is the only thing that
   genuinely needs the hosted project.

What is *not* an option is treating the warning as noise: public means the images (and the class
names, and the split) are readable by anyone with the URL, and there is no "unpublish" button to
press once they are up.

### 8.5 One trainer, two generations — `sidecar/tools/generations.py`

The trainer was v2's: the export directory, the 8-class roster and the `scanncart-grocery-v2`
names were constants in it, so training v1 meant editing that file or copying it — and a copy is
how two generations' "same" checks drift apart, which is the failure §8.1's guards exist to catch.
`generations.py` now holds one **spec** per generation (dataset directory, class list, manifest,
resize requirement, Roboflow project), and `train_model.py` reads it:

| Field | v1 | v2 | Why it is per generation |
|---|---|---|---|
| `classes` | the 7 names `scanncart-grocery` declares | the 8 names of §8.1's roster | An export is judged against *its own* list. Judging v1 against v2's eight would refuse a set that is correct — and a guard that cries wolf is how the real mismatch gets waved through |
| `manifest` | `None` | `cleaned-v2/manifest.json` | Distance is a Roboflow **tag**, and a YOLO export drops tags; v1 predates the tagging, so `None` means "this set has no distance axis", which is a different sentence from "the manifest went missing" |
| `resize_mode` | `stretch`, frozen | `generate_version.REQUIRED_RESIZE_MODE` | v2's is derived from the preprocessing that generated its version; v1's version already exists, and binding it to v2's constant would let a change to v2 rewrite a requirement about an export that cannot change |
| `export_dir` | the ingested export | `export-v2` (`--download` fills it) | The one path that genuinely differs |
| `roboflow_project` | `scanncart-grocery` | `snc-grocery` | The download URL and the record's `source` both name it |

The claim `--generation` makes is therefore narrow and checkable: it selects a *spec*, not a code
path. `--dataset-dir` and `--classes` override the two inputs it describes, so a dataset that is
neither of these two is still trainable without editing anything — and `check_export` prints which
classes another generation declares that this one cannot predict, because a v1 weight is correct
and still cannot predict Palmolive.

---

## 9. Quick Checklist

> `docs/RUN_SHEET.md` is this list as an order of operations: the exact command for each step and
the number that must be true before the next one is worth running. Use it while doing the work;
this section is the summary of what has to be true when you are done.

- [ ] Decide on dataset visibility **before uploading** — the free plan publishes by default and
      there is no per-project toggle (§8.4)
- [ ] Roboflow project created as **Object Detection**
- [ ] All 8 classes defined from §8.1, names final, **Lock Classes** on (renaming later
      invalidates existing labels) — `label_classes.py` prints the list in order and checks it,
      including that no class name carries a distance word (`… close`/`… mid`/`… far` is a 24-class
      head, not a per-distance one; §8.1)
- [ ] ~250–350 raw images per class captured on the real StreamCam rig, spread across the
      three distances (`CAPTURE_CHECKLIST.md` has the per-cell gaps)
- [ ] ≥800 multi-item scenes with overlap and occlusion
- [ ] 5–10% hard negatives — mostly *one roster product + clutter*; any pure-background frames
      marked with the null tool (**N**) before version generation (§2, `CAPTURE_CHECKLIST.md` §C2)
- [ ] Project checks out before shooting: `clean_v2.py sanity` reports no blocking problems
      (class list defined and Lock Classes on, **no class name carrying a distance**, version
      preprocessing at 640×640 Stretch to)
- [ ] All variation axes in §3 covered
- [ ] Extra data collected for visually similar SKU pairs
- [ ] Frames extracted at ~1 fps, duplicates removed
- [ ] Split planned with `plan_split.py` and applied (Plan B by wipe + `upload --split-plan`,
      or Plan A by 21 tag-query bulk actions), then per-class and per-distance coverage checked (§8.3)
- [ ] Modest augmentation only; raw counts reported, not augmented
- [ ] Trained from `yolo11s.pt`, mAP50 ≥ 0.90, and `train_model.py --val` reports recall ≥ 0.85
      for *every* class **and at every distance** (no `[WARN]`, no `[SKIP]`, and nothing under
      "below the floor at a distance")
- [ ] Version generated with `generate_version.py`, and `--verify` reports it *matches*
      (auto-orient on, 640×640 `Stretch to`, augmentation empty, no `filter-null`). `--yes`
      refuses outright if any class name carries a distance, before the POST — so a refusal
      there means fixing the Classes tab and moving the boxes, not regenerating
- [ ] Weights named `scanncart-grocery-v2.*` (§8.2) and dropped in `sidecar/models/` — the picker
      discovers them, so no whitelist edit in either the sidecar or the desktop mirror
- [ ] A record beside them (`models/scanncart-grocery-v2.json`, written by `--install`) naming the
      `resize_mode` they need, so the Model field lists it and warns on a mismatch — and carrying
      the `--val` per-class recall and its per-distance breakdown (§6), so the panel shows what
      the model scored *and* where it is weakest
- [ ] `resize_mode` left on `auto` — it uses the recorded requirement (§6). An explicit
      `letterbox` overrides it, and is the only value the Model field will warn about. No
      "no record of the geometry" entry above the form (that one means no record was left beside
      these weights, so `auto` is guessing — record it, or `--install` again, rather than shipping
      weights nobody can say the geometry of)
- [ ] The class-list guard demonstrated rather than assumed: `node
      .claude/skills/run-desktop/driver.mjs classlist` (app closed, camera free, `npm run build`
      first) prints **13 `PASS` lines, exit 0** against a scratch 24-output weight — the Admin
      listing flags it before it is selected, and the Live view names the distance while it runs —
      then removes the weight and restores the settings. That is what makes step 7's reading
      (*8 classes, no class warning*) a reading instead of a hope (`RUN_SHEET.md` §8)
- [ ] Verified end-to-end in Live View with `conf_threshold` / `track_expiry_s` retuned
