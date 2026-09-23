# Run sheet — from a labeled Roboflow project to `scanncart-grocery-v2` weights

The operational half of `MODEL_TRAINING.md`. That document explains *why* each decision is what it
is; this one is the order to do them in, with the number each step has to report before the next
one is worth running. Nothing here explains a choice — follow the links if a step looks arbitrary,
because several of them are.

Every command runs from the repo root. Replace nothing except the bracketed paths.

---

## 0. Before anything: what is actually missing

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py
```

**This is a snapshot, not a live feed** — it reads the file the tool writes, and the Admin Panel's
*Dataset labeling* section reads the same one. Re-run it (and press *Refresh* there) after any
labeling session; it will look frozen otherwise.

| The number | Today | Move on when |
|---|---|---|
| `X/Y decided` | **858/1433 (59.9%)** | `1433/1433` |
| wrong-class lines | none currently flagged | none, and no `class mismatch` in the run |
| `(negative: …)` nulls | **0/50** | `50/50` |
| Tier A capture gap | **186 of 253 across 8 cells** | `0 to shoot` |
| `mid`/`far` cells labeled | **0 of the 368** staged mid+far images (`close 1015 · mid 152 · far 216`) | non-zero throughout both columns |

The `mid`/`far` columns are the point of this dataset, so treat a `far` cell as higher priority
than a `close` cell of the same size. The 50 negatives are the cheapest item on the list and the
one whose loss is silent: **mark each null (`N`)**, because an unannotated image is *excluded* from
a version while a null-marked one is included as the hard-negative set.

## 1. Stop shooting: the project still checks out

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py sanity
```

**Pass condition: exit code 0, and no `[FAIL]` line.** Warnings are allowed; failures are not.

| Expect | Why it matters |
|---|---|
| `[.ok.] project type is object-detection` | only object detection yields boxes + `track_id` |
| `[.ok.] class list … 8 class(es) defined` | and no `distance in a class name` — that would be a `[FAIL]` |
| `[WARN] project is PUBLIC` | expected on the free plan; there is no toggle |
| `[WARN] 1 version(s) in Trash: 1` … `next generation is version 2` | v1 was burned empty, so your weights are **v2** |
| `[WARN] preprocessing not set yet` | normal before step 4; `--verify` is what confirms it afterwards |

Two things this print has said that people misread: `images: 0 annotated` **is not** your labeling
count (that API field does not move; `label_progress.py` is the real reading), and
`unannotated` does not decrement on delete.

## 2. Shoot the gaps

Tier A first — it is the part that fills **empty** cells, and one of them (`century-tuna` at mid
and far) is a cell the model would otherwise never be taught at all:

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py scaffold --root <tierA-folder> --dry-run
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py scaffold --root <tierA-folder>
```

**The folders it creates are the ingest contract — do not rename them.** Then shoot, and check
`label_progress.py` again: the gap block must shrink by what you shot.

Tier D (`s3`, the held-out acceptance session) is a **separate, later** shoot — see step 9. If you
scaffold it now, its coverage preflight will name the cells that a hold-out would break on:

```
held-out coverage check: 3 of these 24 cell(s) have NO images outside s3 …
    palmolive/close
    century-tuna/far
    century-tuna/mid
```

Those three are Tier A's own empty cells. **Tier A has to land before `s3` can be held out** — that
is the whole meaning of that check, and it is cheaper to read it now than to discover it as a
refused plan after a session on the camera.

## 3. Stage, upload, retag — per session

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean \
  --src <capture-folder> --session s2 --out sidecar/data/datasets/cleaned-v2-s2
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload \
  --out sidecar/data/datasets/cleaned-v2-s2 --session s2
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py retag \
  --out sidecar/data/datasets/cleaned-v2-s2 --session s2
```

| Checkpoint | Expect |
|---|---|
| `clean` | `ingested N candidate images`, then staged count = **1,383 + what you shot** (Tier A's full 253 gives 1,636); the REPORT's dedup table shows *where* frames were dropped, not just how many |
| `clean` | 0 images under a `[warn]` for an unmapped folder — a typo'd folder lands in this report and nowhere else |
| `upload` | resumable via `upload_state.json`; a re-upload returns `{"duplicate": true}` and **leaves the split alone** |
| `retag` | it reads the tags back and verifies the set it *wanted*, not the one it wrote |

`--session` is a tag as well as a record (Roboflow's search has a `tag:` filter and no `batch:`
one), and the split is fixed at upload — which is why Tier D plans it in between.

## 4. Generate the version — once

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --dry-run
./sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --yes
./sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --verify 2
```

| Checkpoint | Expect |
|---|---|
| `--dry-run` | `auto-orient: True`, `Stretch to 640×640`, augmentation `{}` — nothing generated |
| `--yes` | `project … is at version 1; this generation will be 2` → **`generating version 2`** |
| `--verify 2` | **`matches generate_version.py.`** and `splits:` with three non-zero entries |
| refuse (exit 2) | `refusing to generate: N class name(s) carry a distance` — fix the Classes tab, move the boxes, **then** re-run. Nothing was spent |

A version number is consumed and cannot be reused. Do not generate before labeling is finished and
the negatives are null-marked: only annotated images enter, so an early version is a smaller one at
the same price.

## 5. Train

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --download --version 2
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --yes
```

> **v1's local build is the same steps with `--generation v1`, and no `--download`** — its export
> is already ingested in the dataset workspace. Three readings differ, all of them from the
> generation rather than from a flag: the check prints **seven** classes plus a `note` naming
> Palmolive as the class this generation can never predict (v1 has no Palmolive — §8.1's table),
> `--val` reports the per-class table and **no distance grid** (v1's images predate the tags), and
> `--install` writes `scanncart-grocery-v1.pt`. It exists so the app has a natively-loaded model
> before v2's set is finished; `MODEL_TRAINING.md` §6 has the whole chain.

| Checkpoint | Expect |
|---|---|
| `--download` | the export's `data.yaml` lists **8** class names; anything else stops here, and a distance-bearing list is named as such rather than as "extra classes" |
| `--yes` | a run directory under the project's runs; `mAP50` ≥ **0.90** |

The two metrics a training log contains are means. `mAP50` clearing 0.90 is necessary and not
sufficient — the number that matters is the next step's.

## 6. Validate — per class **and** per distance

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --val
```

Defaults to `--split test` (the acceptance split), and runs three extra passes for the distance
breakdown. **Pass condition: every class at or above the 0.85 floor, and nothing under "below the
floor at a distance".**

```
class                                     close          mid          far          all
Milo Chocolate Drink 22g Sachet        0.950 (65)  0.830 (34)!  0.610 (43)!  0.890 (140)
```

(Illustrative numbers, real format: this is the shape the grid prints, not a measurement of any
weight you have trained yet.)

Three states, and the difference is an instruction rather than a style: a number clears the floor,
`!` is below it, and `-` means that distance holds **no instances** of that class — add images of
the item, not to the split. `--no-per-distance` skips the extra passes; a run with no usable
manifest says the breakdown was *skipped* rather than reporting nothing.

## 7. Install

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --install
```

Writes `sidecar/models/scanncart-grocery-v2.pt` **plus** `scanncart-grocery-v2.json` beside it,
carrying the `resize_mode` the version requires (`stretch`) and the recall `--val` measured. Refuses
to overwrite an existing weight without `--force`, because the picker is keyed by filename.

| Checkpoint | Expect |
|---|---|
| the Admin Panel's Model field | the weight listed, with `requirement (recorded): stretch` and the measured score |
| `resize_mode` | leave it on **`auto`** — it honours the record. An explicit `letterbox` overrides it and is the one value worth warning about |
| Test connection | `8 classes`, and no class warning. A 24-class weight would read `24 classes` with the distance problem named |

The install measures accuracy. It does not measure speed, and it does not measure the weight
against the settings actually in force — so the PRD's other two promises are the check that
follows, and it is the one that catches a weight installed into a profile it was not trained for.

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/spec_check.py --generation v2
```

| Checkpoint | Expect |
|---|---|
| the run | `all 5 PRD targets pass`, or a list naming the ones that did not (`--strict` makes that exit non-zero) |
| `config` | which settings were measured. `saved settings` by default, so the number describes **the app on this machine**; `--defaults` measures the shipped set instead |
| the two speeds | `in-app pipeline fps` is the one that decides — `isolated` is the detector alone and will be flattering with nothing around it |
| recall vs `--val` | allowed to differ. This figure is at the app's `conf_threshold`; `--val` picks each class's best-F1 threshold |
| a profile-shaped failure | `imgsz` is **not** recorded beside the weights (only `resize_mode` is). v1 at a saved `imgsz` 960 reads 0.344 recall and 24 fps against 0.918 and 41 — check Admin's `imgsz` matches the run before blaming the model |

That check measures the weight, not the app. The running app is a second, separate reading, and it
is the one a user experiences — so with a camera attached, the same acceptance in the UI:

```bash
node .claude/skills/run-desktop/driver.mjs v1
```

| Checkpoint | Expect |
|---|---|
| the run | **14 `PASS` lines, exit 0**, once a product is in front of the camera — 6 of them before Start, so a camera-less machine still gets the strip half. Without a product the class check fails with `0 row(s)`, which is the correct reading rather than a bug |
| the strip | `stretch` `geometry (auto)` · `stretch` `requirement (recorded)` · `94%` `test recall · 1 below floor` — all four derived from the weight's own record |
| the item log | only v1's seven names, ever. A COCO name means the stock weight is still loaded; a `Palmolive` row means a v2 weight is |
| the running verdict | `7` + `classes · 1 finding`, with Palmolive named — v1 genuinely cannot predict it. `roster ok` here would mean the gap went unnoticed |

## 8. Prove the class-list guard fires, before you trust its silence

Step 7's reassuring reading — *Test connection: 8 classes, no class warning* — is only a reading if
the check can produce the other one. This step makes the running app meet a weight that really has
24 outputs, so both halves of the guard in `MODEL_TRAINING.md` §8.1 are seen refusing to stay
quiet.

**Close the running app first** — the camera is exclusive, and the banner only exists once the
pipeline has inferred, so a dev app holding the device leaves this mode unable to check the half of
the guard that matters.

```bash
cd desktop && npm run build && cd ..   # the driver launches desktop/out, not the dev server
node .claude/skills/run-desktop/driver.mjs classlist
```

| Checkpoint | Expect |
|---|---|
| the run | **13 `PASS` lines, exit 0** — one per check, so a new check raises the number; any `FAIL` exits non-zero, which is what lets it gate a change |
| the fixture | `scratch weight: 10669041 bytes` — a real `DetectionModel('yolo11n.yaml', nc=24)`, built and deleted by the mode itself |
| Admin · *Weights on disk* | `… — 24 classes` with ⚠ `24 of 24 class name(s) carry a distance`, **before** the weight is selected |
| Live · banner and chip | the same sentence in `live-class-warnings`, and `24` · `classes · 1 finding` in amber — from a running capture, not a fake status |
| the banner is *painted* | `{"w":952,"h":200,"onScreen":true,"shown":true}`: a non-zero box inside the viewport, because `textContent` cannot tell a rendered element from a collapsed one |
| cleanup | `scratch weight removed` and `settings put back` — both in a `finally`, so they run even when a check throws |
| screenshots | `.claude/skills/run-desktop/shots/classlist-0*.png`. Read them: the text alone does not prove the UI rendered |

**It needs no v2 weights**, so it runs today, before anything is trained — and that is the point of
doing it *before* step 7's reading is trusted: it is a check on the checker. The scratch `.pt` is
deleted at the end, so `scanncart-grocery-v2.json` stays the only class list on disk. The mode's own
notes (what it covers, what it needs, the launch gotchas) are in `.claude/skills/run-desktop/SKILL.md`.

## 9. The acceptance number (Tier D, `s3`)

Only after Tier A has landed. This is what turns the quoted accuracy into an *unseen-session*
estimate instead of a held-out-frame one — until then, `test` is drawn from the same session `train`
saw, and the tool says so outright.

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean \
  --src <s3-capture-folder> --session s3 --out sidecar/data/datasets/cleaned-v2-s3
./sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py \
  --out sidecar/data/datasets/cleaned-v2 --include sidecar/data/datasets/cleaned-v2-s3 --holdout-session s3
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload \
  --out sidecar/data/datasets/cleaned-v2-s3 --session s3 \
  --split-plan sidecar/data/datasets/cleaned-v2/split_plan_c.json
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py retag \
  --out sidecar/data/datasets/cleaned-v2-s3 --session s3
```

| Checkpoint | Expect |
|---|---|
| `plan_split` | **must not** print `CANNOT hold out session s3`. If it does, a cell is covered only by this session — shoot it into an earlier session first |
| the report | all of `s3` in `test`, none in `train`; the Admin Panel's session spread shows the same |
| the protocol | quote **`test`**, and say the session. Do not re-shoot or top up `s3` after seeing the result — a test set is spent the moment it is tuned against |

---

## What "done" looks like

| | |
|---|---|
| labels | `1433/1433 decided`, 50 nulls marked, no class mismatches |
| version | 2, `--verify` matches, 8 classes, `Stretch to 640` |
| weights | `scanncart-grocery-v2.pt` + its record, `auto` honouring `stretch` |
| measured | every class ≥ 0.85 **and** every distance cell reported, unbracketed by `!` |
| quoted | the `test` score, labelled with the capture session it came from |
