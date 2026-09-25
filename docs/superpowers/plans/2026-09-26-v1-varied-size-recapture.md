# Plan — varied-size capture and retrain for the v1 model

> What the v1 weights (`scanncart-grocery`) actually need in order to hold up on items at
> different distances and sizes on the counter, in the order the work has to happen, with the
> number each step must report before the next one is worth running.
>
> Related: [CAPTURE_CHECKLIST.md](../../CAPTURE_CHECKLIST.md) (§v2's tier table — reused for
> method, **not** for its numbers) · [RUN_SHEET.md](../../RUN_SHEET.md) (the v2 path this
> mirrors) · [MODEL_TRAINING.md](../../MODEL_TRAINING.md) §8 · `sidecar/tools/generations.py`.

## 0. The two things this is not

**It is not a framing or preprocessing change.** Letterbox vs stretch changes how many pixels an
object is given at inference; it does not change what the weights learned. The A/B that was run on
the live camera returned **zero detections in both modes**, which is a statement about that scene,
not about the geometries. `resize_mode` should stay on `auto` for v1 regardless: v1's export was
generated with `Stretch to 640×640` and the record beside the weights says `stretch`, so `auto`
already runs the geometry they were trained at.

**It cannot be done inside v1's existing version.** A Roboflow version is frozen the moment it is
generated: the number is consumed, the preprocessing is baked, and the class list is fixed. Adding
images to the project does not change version 1. So "retrain v1" has to mean *generate a new
version of the same project and retrain from that*, and §3 is about what has to stay identical for
the result to still be a v1 drop-in.

## 1. The diagnosis, measured rather than felt

| Reading | Value | Command |
|---|---|---|
| Recall on frames holding **one** labelled object | **265/265 = 100.0%** | `audit_recall.py --generation v1` |
| Recall on frames holding **2+** | **103/136 = 75.7%** | same run |
| Empty-counter frames that produced a detection | **25/50** | `clamp_probe.py --generation v1 --conf 0.5` |
| …of those, a clamped full-frame box | **19/25** (real product frames: **0/60**) | same run |

The crowding split is the whole story. **Not one solo instance in v1's 327-frame test split was
missed**, so every point of recall that shooting more solo shots could buy is already bought, and
the aggregate the 0.85 floor is computed over (0.918) is entirely the crowded bucket pulling it
down. "It misses things at varying sizes" on a counter is therefore two defects, and neither is
scale:

1. **Crowding and occlusion** — the misses are consistently *the second instance*: the one behind,
   smaller, or partly covered. v1 was shot as per-class solo captures, so it was never taught that
   case, and it is the case the counter always produces.
2. **Distance under-representation** — the same capture regime means small objects are rare. The
   v2 set only moved from 1,015 close / 152 mid / 216 far while being *built around the distance
   axis*; v1 predates the axis entirely and has no per-distance reading at all.
3. **A separate false-positive defect** — an empty counter resolves into a product because 145–187
   labels per class in v1's training set are pinned against an image border. This is not a miss,
   but it is in the same session's work, and it is the cheapest item in the plan to fix.

## 2. Rule out the free levers first — day 0, no camera

Three of the four candidate causes are settings, and all three are measurable before anything is
shot. **Gate: if these explain the misses, stop — no capture and no retrain is needed.**

| Lever | Command | What it settles | v1 status today |
|---|---|---|---|
| Decision threshold | `audit_recall.py --generation v1 --conf-sweep` | a miss ranked *below* `conf_threshold` is a slider, not a retrain. Prints the false positives each threshold costs, split by crowding | `conf_threshold` is hot-reloadable, so this is free |
| NMS IoU | `audit_recall.py --generation v1 --iou-sweep` | two tins side by side overlap; at ultralytics' default NMS a genuinely-detected second tin is discarded **inside the model** | **not exposed**: `YoloDetector.__init__` takes `conf`/`imgsz`/`resize_mode` and no `iou`, so today the run is at the library default. The sweep is what decides whether exposing it is worth doing (§8) |
| `imgsz` | Admin panel vs `train_model.py --val` | a saved `imgsz` that is not the trained size is a config trap, not a model gap — v1 reads **0.344 recall / 24 fps** at `imgsz 960` against **0.918 / 41** at 640 | check it matches 640 |

Record the three readings before shooting. They are also the "before" column of §7, and the conf
sweep is the only one that costs nothing to re-run.

**Prerequisite for all of §2 and §3:** the tools read the workspace at
`sidecar/data/datasets/` (`workspace.py`; override with `SCANNCART_DATASET_ROOT`). On this machine
that directory **does not exist** — `sidecar/data/` holds only `settings.json`,
`camera_profiles.json` and `scanncart.db` — so the v1 export has to be re-ingested there before any
of these commands run. Fetch it as the project's YOLOv11 PyTorch export, version 1 of
`scanncart-grocery`, and unpack it to `sidecar/data/datasets/scanncart-grocery-v1/` (that path is
what `generations.V1.export_dir` points at, and what `check_export` re-measures the
`640×640, no constant border` claim from).

## 3. What "v1" has to mean for the result to be a drop-in

The refreshed weights must be indistinguishable from v1's to the running app. Three things are
pinned, and all three are checked by the tools rather than remembered:

| Pinned | Value | Enforced by |
|---|---|---|
| Class list | v1's **seven** names, in v1's order (`generations.V1_CLASSES`) | `train_model.check_export` refuses a mismatch and names Palmolive as the class this generation can never predict |
| Version preprocessing | `auto-orient` + **`Stretch to 640×640`** | `generate_version.py --verify <n>` |
| Runtime geometry | `resize_mode: stretch` in the record `--install` writes | `resolve_resize_mode()` — `auto` honours the record; the heuristic answers *letterbox* for a locally trained `.pt` |

Consequences worth stating plainly:

- **Do not add Palmolive to `scanncart-grocery`.** If it is created there, the new version's class
  list is 8 and `check_export` will refuse it — loudly, which is the point, but after a session on
  the camera. So do **not** run `label_classes.py --create-classes` against the v1 project; the
  seven names already exist there and `--apply` (metadata) is the useful half. Note that
  `label_classes.py`'s own `--project` defaults to `snc-grocery`, so this is a flag someone has to
  get right twice, not a default that protects them.
- **The v1 refresh is a new generation of artifacts with an old name.** `--install` writes
  `scanncart-grocery-v1.pt`, `run_name` is `scanncart-grocery-v1`, and the record's `source` carries
  the version number — so the *provenance* separates two refreshes while the *filename* does not.
  Keep the current weight aside before installing (§7) or the before/after comparison is gone.
- **`generations.py`'s v1 comment becomes false.** It says v1 "already exists and will never be
  regenerated", which is exactly the premise this work invalidates. Either point `V1.export_dir`
  and `V1.manifest` at the refreshed set, or add a `v1b` spec and take the naming hit. One of the
  two, decided before the code is touched — see §8.

**No trainer change is needed.** `train_model.py` already takes `--dataset-dir`/`--export-dir`,
`--manifest`, `--project`, `--version` and `--classes`; `clean_v2.py`'s `upload`/`retag`/`wipe`/
`sanity` and `generate_version.py`/`label_classes.py` all take `--project`. So the v1 path is flags,
not forks — with one exception in §8 (the auditor). **Name the project on every call**: it defaults
to `snc-grocery` in `generate_version.py` and `label_classes.py`, and `clean_v2.py` falls back to
`ROBOFLOW_PROJECT_ID` in the workspace `.env` (which is v2's), so a bare command silently targets
the wrong project rather than failing.

## 4. The capture spec — "varied size" as a number, not an adjective

### The size axis

Objects have to be shot at sizes the app can be *shown* to fail on. Define the three bands by box
width **in preview pixels**, because that is what the operator can read live: the Live view's
`det-size` readout (added in `ac583d3`) prints exactly `(x2-x1)·naturalWidth × (y2-y1)·naturalHeight`
for every tracked box. At the capture resolution the rig already uses (1280×720), stretched to
`imgsz 640`, x scales by 0.5 and y by 0.889:

| Band | Box width, live preview (1280 wide) | Same box at `imgsz 640` | Why |
|---|---|---|---|
| close | **≥ 300 px** | ≥ 150 px | v1's regime — already covered, shoot little of it |
| mid | **120–300 px** | 60–150 px | the axis v2 exists to fix and is still thinnest |
| far | **40–120 px** | 20–60 px | where a wrong box costs most, and where v1 has almost nothing |

Put tape on the counter at each band and **check the first frame of each cell against the `det-size`
readout** before shooting the rest. A cell shot at the wrong distance lands somewhere already thin,
and on v1 there are no tags to catch it after the fact.

### Tier 1 — measure the gap before filling it

v1 has no distance tags, so the v2 checklist's tier table **cannot be copied**: it describes v2's
cells. The gap has to be measured from v1's own labels, which are already in the export. For each of
the seven classes, histogram the label box widths and report the smallest decile — that names the
classes with **no instances in the far band at all**, which is the only defensible shoot list. This
is a small addition to the audit toolchain (§8), and it is step one of the work rather than a
prelude, because without it the shoot list is a guess.

### Tier 2 — crowded frames (the measured lever)

The only bucket with a measured recall gap behind it. Target **~60–80 scenes per class**, and the
counts matter less than the composition:

- **3–6 roster products per frame**, touching and overlapping, **30–50% of at least one item
  occluded**.
- **Rotate which item is the hero.** A set where `safeguard` is always nearest and fully visible
  teaches the model `safeguard` and nothing about finding a safeguard behind two tins. Each class
  must appear in the occluded/background position in at least a fifth of its scenes.
- **Go past pairs.** v1's crowded bucket is mostly two-item frames and that is the crowded case it
  already handles (75.7%). The bulk should be 4–6 items.
- **Put the similar-SKU pairs inside an already-crowded frame**: `555 sardines` / `century tuna`
  (both tins) and `lucky me pancit` / `milo` (both sachets). One flip there logs one physical item
  as two products.
- **Mix bands within a frame** — some items near, some far. That is what the counter does.
- Vary arrangement between shots: grid, pile, cluster, one rotated.

### Tier 3 — hard negatives

30–50 frames, and they must be **marked null with the annotator's `N` tool, never drawn on and
never auto-labelled**: an uploaded frame that is left unannotated is *excluded* from the version,
which is how a negative set disappears silently. Two shapes:

- **mostly one roster product + clutter** (a hand reaching in, a phone, a bag, a wallet) — no extra
  step beyond normal labelling;
- **~30 pure empty counter** frames in their own folder, each null-marked. An earlier session left
  **60 StreamCam frames with empty labels** in `sidecar/data/datasets/hard_negatives/` (50 survive
  the cleaner) — eyeball those before shooting new ones; the marking is the outstanding work, not
  the capture.

### The holding rules

Same rig, same mount height, same resolution, **~1 fps frame extraction**, and a **separate sitting**
from any earlier session with its own session tag. Reuse the taped distance marks. A cell of 40
near-identical frames is worth less than 20 varied ones.

## 5. The pipeline, in order, with the checkpoint each step must report

Replace `<v1-folder>` with the capture root and `<n>` with the version number
`generate_version.py` reports. Every command runs from the repo root. The v1 project is named
explicitly on every call because the default is `snc-grocery`.

| # | Command | Must report |
|---|---|---|
| 0 | `clean_v2.py sanity --project scanncart-grocery` | exit 0, no `[FAIL]`; **7** classes, and no class name carrying a distance. `[WARN] project is PUBLIC` is expected |
| 1 | `clean_v2.py scaffold --root <v1-folder> --dry-run` then without it | folders created by the same `CLASS_MAP` the ingest uses; do not rename them afterwards |
| 2 | shoot Tiers 1–3 | first frame of each cell checked against the Live `det-size` readout |
| 3 | `clean_v2.py clean --src <v1-folder> --session v1s2 --out sidecar/data/datasets/cleaned-v1-s2` | `ingested N candidate images`, then the staged count is N minus dedup, with the REPORT's dedup table showing *where* frames were dropped; **0 images under a `[warn]` for an unmapped folder** (a typo lands in that report and nowhere else). Use a **new `--out`** — `clean` clears the class folders in its output, so pointing it at `cleaned-v2` wipes v2's staged set |
| 4 | `clean_v2.py upload --out …/cleaned-v1-s2 --session v1s2 --project scanncart-grocery` | resumable via `upload_state.json`; batches are session-suffixed (`tuna_mid_v1s2`) |
| 5 | `clean_v2.py retag --out …/cleaned-v1-s2 --session v1s2 --project scanncart-grocery` | reads the tags back and verifies the set it *wanted*, not the one it wrote |
| 6 | `label_classes.py --apply --project scanncart-grocery` | mapping matches the live project. **Not `--create-classes`** (§3) |
| 7 | label every cell, biggest first (`label_progress.py` prints that order) | `X/Y decided` climbing; **0 wrong-class lines** |
| 8 | mark the Tier 3 frames **null** (`N`) | `label_progress.py`'s `(negative: …)` line reads `N/N`. Until it does, those frames are not in the version at all |
| 9 | `plan_split.py` → apply | `0` unmeasurable class-slots. Plan B while iterating (stratified per cell), Plan A for a final acceptance run (strict by session) |
| 10 | `generate_version.py --project scanncart-grocery --dry-run` → `--yes` → `--verify <n>` | `--dry-run`: `auto-orient: True`, **`Stretch to 640×640`**, augmentation `{}`. `--verify`: **`matches generate_version.py.`**, three non-zero splits, and it refuses a version that filtered the nulls |
| 11 | `train_model.py --generation v1 --dataset-dir sidecar/data/datasets/export-v1-s2 --download --version <n> --project scanncart-grocery --manifest sidecar/data/datasets/cleaned-v1-s2/manifest.json` | `data.yaml` lists **seven** names, plus the `note` naming Palmolive as the class this generation can never predict. `--dataset-dir` keeps the frozen v1 export (`scanncart-grocery-v1/`) intact as the "before" baseline — `--download` would otherwise refuse to overwrite it |
| 12 | the same command `--yes` | a run directory; **`mAP50 ≥ 0.90`**. Pass `--run-project` if you want the two refreshes' runs side by side instead of the second overwriting the first |
| 13 | `train_model.py --val --split test` with the same `--dataset-dir --manifest` | **every class ≥ 0.85**, no `[WARN]`/`[SKIP]` line, **and** the class × distance grid: nothing under *below the floor at a distance*. `-` means that band held no instances — add images of the item, not to the split. Without `--manifest` this grid is skipped, which is the failure mode this whole plan is designed to avoid |
| 14 | `train_model.py --generation v1 --install` (add `--force` only when replacing an existing weight) | `models/scanncart-grocery-v1.pt` + `.json`. The install prints the score it picked up from `--val`; a panel showing nothing is then never a surprise |
| 15 | `resize_mode` | leave it on **`auto`**. The Admin panel lists `requirement (recorded): stretch` and flags an explicit value that contradicts it |

## 6. Verification — what "done" is, and the number that decides it

Accuracy:

```bash
# the before/after that this plan exists for — same tool, same split, same operating point
./sidecar/.venv/Scripts/python.exe sidecar/tools/audit_recall.py \
  --generation v1 --weights models/scanncart-grocery-v1.pt
./sidecar/.venv/Scripts/python.exe sidecar/tools/audit_recall.py \
  --generation v1 --weights models/scanncart-grocery-v1.pt --conf-sweep --iou-sweep
```

**The decision number is the crowded bucket, not the mean**: `multi` was **103/136 (75.7%)** and has
to reach the **0.85** floor with the `single` bucket staying at 265/265. Re-run the sweeps
afterwards — if the residual misses sit below a threshold or inside NMS, that is a *setting* to
expose, not another capture round.

The app's own promises, which no training metric covers:

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/spec_check.py --generation v1
node .claude/skills/run-desktop/driver.mjs v1
```

| Check | Expect |
|---|---|
| `spec_check.py` | `all 5 PRD targets pass`; `config` says which settings were measured (`saved settings` by default, so the number describes *this machine*) |
| `driver.mjs v1` | **14 `PASS` lines, exit 0** with a product in front of the camera; the strip reads `stretch` `requirement (recorded)`; the item log shows **only v1's seven names, ever** — a `Palmolive` row means a v2 weight is loaded, a COCO name means the stock weight is |
| the roster chip | `7` · `classes · roster ok` |
| `clamp_probe.py --generation v1 --conf 0.5` | the negative population's clamped-box count drops from **19/25** and the frames with any detection at all from **25/50**; the real-product population stays at **0/60** |

## 7. Order, effort, and what to do first

1. **Day 0** — re-ingest v1's export into `sidecar/data/datasets/scanncart-grocery-v1/`; run §2's
   three readings; copy the current weight aside as `scanncart-grocery.before.pt` (gitignored,
   nothing tracks it).
2. **The size histogram** from v1's own labels → the shoot list. This is the step that turns
   "varied sizes" into a number of frames per class per band:

   ```bash
   ./sidecar/.venv/Scripts/python.exe sidecar/tools/audit_recall.py --generation v1 --size-histogram
   ```

   It reads labels only — no weight, no GPU, no export download — and its shoot list is the tier
   plan's input. Re-run it with `--dataset-dir`/`--manifest` after a session lands to see the bands
   fill in.
3. **Tier 2 before Tier 1 if time is short.** Crowding is the bucket with a measured 24-point gap
   behind it; the distance tail is real but unmeasured on v1 until item 2 says where it is.
4. **Tier 3 is ~30 frames and one keystroke each** — it fixes a phantom item-log row per session and
   is the highest value per minute in the plan.
5. Shoot, then §5 in order. The long pole is labelling, and `label_progress.py`'s
   *largest-cell-first* order is the throughput plan.
6. **Acceptance is §6's crowded bucket**, and it is only meaningful if the shoot happened on a
   different sitting than the frames the model was trained on.

## 8. Code work this plan needs (all small, all nameable)

Items **2** and **3** shipped with this plan, so the acceptance number is measurable against a
refreshed dataset and the Tier 1 shoot list is derivable; the rest is outstanding.

| # | Change | Why it is necessary rather than nice |
|---|---|---|
| 1 | `generations.py`: point `V1.export_dir`/`V1.manifest` at the refreshed set, **or** add a `v1b` spec | Its current comment asserts v1 "will never be regenerated", which this work makes false. Leaving it makes the tool's own documentation a lie about its dataset |
| 2 | `audit_recall.py`: `--dataset-dir`/`--export-dir` and `--manifest` overrides — **done**, spelled as the trainer spells them | The trainer has them; the auditor did not, so it could only measure against `generation.export_dir`. That means the *before* number is readable but the *after* one is not — the tool that decides acceptance (§6) cannot see the new dataset |
| 3 | per-class label-box-width histogram — **done**, as `--size-histogram` on the auditor rather than a second tool | v1 has no distance tags, so the shoot list has to come from the labels. Without it Tier 1 is a guess and "varied size" stays an adjective. It lives in the auditor because that is where the polygon-aware label reader already is, and a second reader is the 484-zero-area-labels bug waiting to happen |
| 4 | expose NMS `iou` on `YoloDetector` — **only if** `--iou-sweep` says the second tin is suppressed | Today `conf`/`imgsz`/`resize_mode` are the only knobs, so a real recall loss inside the model cannot be distinguished from a model gap or fixed without a hard-coded edit |
| 5 | `docs/CAPTURE_CHECKLIST.md` is v2's; the v1 reshoot needs its own section (or a checklist of its own) | Its tier table is a list of v2's cells. Reusing it would shoot the wrong cells while looking rigorous |

## 9. What this plan cannot give you

- **The per-distance grid needs the manifest**, and v1's generation declares none. It appears only
  if the new captures are staged and tagged through `clean_v2.py` and `--manifest` is passed at
  train and validate time. Skip that discipline and the grid is silently skipped — the one place
  this toolchain reports a missing axis instead of a passing one, which is why §5 step 13 spells it
  out.
- **No distance-grid reading can be quoted between *sittings*.** Every batch here is one session, so
  validation is on held-out *frames* of a session the training data came from. An unseen-session
  claim needs a second sitting held out (`plan_split.py --holdout-session`), and it belongs to the
  acceptance run, not to every iteration.
- **A same-scene A/B on the live camera proves nothing** about detection quality, in either
  direction, and the one already run (zero detections in both modes) is the demonstration.
- **Nothing here is verified on this machine.** `sidecar/data/datasets/` does not exist locally, so
  no command in §2 or §5 has been executed; the numbers quoted are the ones the tools' own
  documentation records, and every one of them is re-derivable with the command in its row.
