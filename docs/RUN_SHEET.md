# Run sheet — from a staged capture session to `scanncart-grocery-v2` weights

The operational half of `MODEL_TRAINING.md`. That document explains *why* each decision is what it
is; this one is the order to do them in, with the number each step has to report before the next
one is worth running. Nothing here explains a choice — follow the links if a step looks arbitrary,
because several of them are.

Every command runs from the repo root. Replace nothing except the bracketed paths.

**v2's set is assembled locally**: a session is staged, labeled in the local annotator (§4), split
by the local planner (§5) and merged with v1's already-ingested export (§6) into one seven-class
set. That is what changed from the earlier flow — there is no annotated Roboflow version to train
from, the labeling happens on this machine, and **no step below needs the API**: the split is frozen
from the planner's own plan file rather than read back from the project. The Roboflow version path
survives in the appendix, because it is how v1's export was made and the only route back to the
server-side labels.

---

## 0. Before anything: what is actually missing

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py
```

**This is a snapshot, not a live feed** — it reads the labels on disk (or the project, when no
local labeling session exists yet: the first line says which) and writes the file the Admin Panel's
*Dataset labeling* section reads. Re-run it (and press *Refresh* there) after any labeling session;
it will look frozen otherwise.

| The number | Today | Move on when |
|---|---|---|
| `X/Y decided` | **858/1433 (59.9%)** | `1433/1433` |
| wrong-class lines | **0** — the three flagged frames (`century-tuna_0078`, `lucky-me-pancit_0159`, `silver-swan-vinegar_0077`) were corrected in the project, not here: the drawn class was the error, and the boxes already sat on the object the frame's tag names | none, and no `class mismatch` in the run |
| `(negative: …)` nulls | **0/50** | `50/50` |
| Tier A capture gap | **132 of 186 across 6 cells** | `0 to shoot` |
| `machine-only` decisions | whatever §4 left unconfirmed | `0` in `valid` and `test`, or §11 refuses |
| `mid`/`far` cells labeled | **0 of the 368** staged mid+far images (`close 1015 · mid 152 · far 216`) | non-zero throughout both columns |

The `858` in that table were decisions made in **Roboflow's annotator**, not on this machine, and
§4's `import_labels.py` has since brought them across (858 pulled, 0 refusals, into
`annotations-v2/`) - the split is frozen from **plan B** (`splits.json`: 1004/288/141 over the staged
1433) and §6-§7 have run on the result (`merged-v2`: 1,863/400/243, doctor `[ok]`). The three
wrong-class frames were fixed in the project rather than in the local tree - the drawn class was the
mistake (each drawing already boxes the object the v1 model's own box covers on that photo), so the
same geometry was rewritten under the tag's class and the pull re-run (`--force`: 858 again, 0
refusals) - and the snapshot now reads `no class mismatches`. The rule still
holds for any later session labeled on the server: the merge reads the local tree, so until
`import_labels.py` runs §6 refuses the rebuild as v1 alone.

The `mid`/`far` columns are the point of this dataset, so treat a `far` cell as higher priority
than a `close` cell of the same size. The 50 negatives are the cheapest item on the list and the
one whose loss is silent: **mark each null (`N`)**, because an unannotated image is *excluded* from
a version while a null-marked one is included as the hard-negative set.

## 1. Sanity: the project still checks out

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py sanity
```

**Pass condition: exit code 0, and no `[FAIL]` line.** Warnings are allowed; failures are not.

| Expect | Why it matters |
|---|---|
| `[.ok.] project type is object-detection` | only object detection yields boxes + `track_id` |
| `[.ok.] class list … 7 class(es) defined` | and no `distance in a class name` — that would be a `[FAIL]`. Distance is a tag, never a class (§8.1) |
| `[WARN] project is PUBLIC` | expected on the free plan; there is no toggle |
| `[WARN] 1 version(s) in Trash: 1` … `next generation is version 2` | v1 was burned empty, so a generated version would be **2** |
| `[WARN] preprocessing not set yet` | normal before §13; `--verify` is what confirms it afterwards |

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

Tier D (`s3`, the held-out acceptance session) is a **separate, later** shoot — see §13. If you
scaffold it now, its coverage preflight will name the cells that a hold-out would break on:

```
held-out coverage check: N of these 21 cell(s) have NO images outside s3 …
    <class>/far
    <class>/mid
```

(The cells it names are Tier A's own empty ones — 21 is seven classes across three distances.)
**Tier A has to land before `s3` can be held out** — that
is the whole meaning of that check, and it is cheaper to read it now than to discover it as a
refused plan after a session on the camera.

## 3. Stage the session

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean \
  --src <capture-folder> --session s2 --out sidecar/data/datasets/cleaned-v2-s2
```

| Checkpoint | Expect |
|---|---|
| `clean` | `ingested N candidate images`, then a staged count you can add up; the REPORT's dedup table shows *where* frames were dropped, not just how many |
| `clean` | 0 images under a `[warn]` for an unmapped folder — a typo'd folder lands in this report and nowhere else |
| `clean` | the session tag is on every image. It is recorded in the manifest **and** written as a tag, because a batch can group images but only a tag can select them |

`--session` is what makes the session × split spread in the Admin Panel possible, and a frame with
no session tag cannot be pinned into a held-out one later.

## 4. Label it locally

```bash
make annotate                      # sidecar/.venv/Scripts/python.exe -m annotate.run
```

It prints `images`, `labels`, the worklist size, which weight is suggesting, and
`ANNOTATE_PORT=<n>` — open `http://127.0.0.1:<n>`. Labels land in
`sidecar/data/datasets/annotations-v2/` and **nothing leaves the machine** unless you pass
`--hosted`.

| The panel | What it means |
|---|---|
| the worklist | most-outstanding first: the thing that stalls a dataset is a *(class, distance)* cell nobody opened, not a class nobody started |
| suggestion confidence | what the local weight proposed; a frame the weight missed is where the `--hosted` second opinion is worth its per-call cost |
| `N` | mark a null — **the hard negatives are annotated, not skipped.** An unannotated frame is excluded from a build; a null-marked one is included as a *background* frame |
| `u` / *Review* | the machine-only worklist: frames whose boxes the weight drew and a person has not confirmed |
| `f` / *Confirm* | confirms the current frame's boxes, which is what clears it from that worklist |
| `d` / *Draw* | the `far` frames in `test`/`valid` a weight was asked about and proposed nothing for — the checklist's draw section, served from the same rule the document is rendered from |
| `m` / *Nulls* | the hard negatives in `test`/`valid` with no null yet — the checklist's null section. One of these at a time: `Review`, `Draw` and `Nulls` are the document's three lists, and turning one on turns the others off |
| the three filters | `Split`, `Distance`, `State` narrow whichever list is on screen; `unassigned` and `unrecorded` are the frames with nothing recorded — the hard negatives, and any frame no plan has reached |
| the gate chip | `gate: 80 machine-only in test/valid (test 25 / valid 55) - stop at 153`, amber, and `gate: clear` in green once they are clean (then it is inert — there is nowhere to go). **Click it** to open exactly those frames: the review pass, narrowed to the pair the run refuses on. It is the number §4's checklist header prints and the one §13's gate reads, recomputed on every save — **work the review pass down to the `stop at` figure and the gate clears**; the rest of the machine-only frames are in `train`, where unread boxes cost nothing |

**Confirm or correct every machine-only frame.** §11 refuses to report an acceptance number over a
set with unconfirmed ones, because a box a weight drew and nobody looked at is not evidence — and
the item log has no way to tell the two apart on its own. The annotator keeps that state
monotonically: re-saving an unchanged frame cannot silently re-mark it as confirmed.

### Work that pass from the checklist

```bash
make human-pass                          # render it from the store
make human-pass HUMAN_PASS_ARGS=--check  # exit 1 when it has stopped matching the set
make human-pass HUMAN_PASS_ARGS=--status # nonzero while the gate splits still hold machine-only work
```

`_human_pass.md` (in the workspace, beside `annotations-v2/`) is **rendered** from the store rather
than kept by hand, because every number in it moves as the pass is worked: the header's per-split
machine-only counts, the count the header falls to once `valid` and `test` are clean, and three
frame lists — the review worklist in the app's own order, the `far` frames in `valid`/`test` the
weight was asked about and proposed nothing for, and the hard negatives still without a null. Run it
before sitting down and again after a session: a checklist that no longer matches the set is the one
way this document can be wrong, and each name it lists is read back off `provenance.json` and the
labels on disk, so `--check` turns that into a nonzero exit rather than something to notice by eye.

`--status` answers the other question, and the one a script asks: it prints the three sections'
counts and exits nonzero (the tool exits 1; `make` reports a failed recipe as 2, so test for
nonzero rather than for 1) while any machine-only decision remains in `test` or `valid` — the same
pair §11's gate refuses to measure over. The draw and null counts are reported beside it but do not
hold it open: a missing `far` frame and an unmarked negative cost evidence (the per-distance recall
and the false-positive half), while a weight's unread boxes in a measured split make the number
itself unsound. Neither mode writes the document, so both are safe to run before you know the
answer.

All three lists are workable in the page as well as readable here: `Review (u)`, `Draw (d)` and
`Nulls (m)` are the sections in order, and the three filters beside the worklist narrow whichever
one is on — the same calls this document is rendered from, so working a section on screen and
reading it here cannot diverge. The header's **gate chip** is this header: the same count, recomputed
on every save, so you can watch `stop at` arrive instead of re-rendering this file. Draw the `far`
frames before confirming the `valid`/`test` boxes; the nulls are a keypress each and can go last.

### If part of the labeling happened in Roboflow instead

The merge reads the local tree, never the project, so a decision that exists only on Roboflow is
invisible to `build_dataset.py` — the frame reads as outstanding work and the merged set is built
without it. `import_labels.py` pulls them across: one GET per annotated image, boxes in the
uploaded frame's own pixel space, class *names* mapped onto the local roster (the project's class
order is not this dataset's).

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/import_labels.py --dry-run
./sidecar/.venv/Scripts/python.exe sidecar/tools/import_labels.py
```

| Expect | Why it matters |
|---|---|
| `N labeled, M null` written into `annotations-v2/` | the same tree `make annotate` writes, so the two routes mix; a null becomes the *empty* label file a hard negative needs, not "no file" |
| a frame is refused **whole**, never half-written | a dropped box or a clamped one changes what the frame says, and nothing downstream could tell |
| `exit 2` with the refusals listed | a class this dataset does not have (the project's eighth), a frame whose aspect is not the staged file's, a box that is not a valid row — each is fixed in the project, not here |
| a second run pulls nothing | frames already decided here are left alone, so an interrupted pull resumes and a person's later edit in the annotator survives; `--force` is the deliberate override |

Roboflow records *that* an image is annotated, never *who* drew it: a hand-drawn box and one
accepted from Label Assist come back identical. So the pulled rows are recorded as the project's
decisions (provider `roboflow:snc-grocery`) — the footing v1's hand-labelled frames are already on —
unless `--awaiting-review` says otherwise, which records them as a provider's unread boxes and puts
them in the *Review* worklist (test split first) and in §11's machine-only count.

## 5. Plan the split, then freeze it

`plan_split` decides 70/20/10 over the batches and writes the plans; `label_progress
--capture-splits` turns the one you chose into the per-frame `splits.json` the annotator displays
and §6 refuses to build without. **Neither needs Roboflow**, so this step is not the deadline it
used to be.

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py --out sidecar/data/datasets/cleaned-v2
./sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py --capture-splits \
  --split-plan sidecar/data/datasets/cleaned-v2/split_plan_b.json
```

`plan_split` reads `--out` **and the staged sets beside it** (the hard negatives, whose own manifest
is what makes them a set) — the same `default_extras` rule the annotator, the snapshot and the merge
use, so the plan covers every frame a build will read. A later session staged in its own directory is
added with `--include`; the plan files themselves land in `--out`. Its last line prints the `freeze`
command above with the right filename in it.

| Checkpoint | Expect |
|---|---|
| `plan_split` | `staged sets: cleaned-v2, cleaned-negatives`, then a plan, then the count of `(class, distance)` cells it leaves with **no train images**. With one session that number cannot be zero — it is reported rather than enforced, because a whole-batch plan cannot satisfy it without giving up the hold-out |
| `label_progress --capture-splits` | `assignment: …/split_plan_b.json (a split plan - the project is not read)`, then `splits -> …/splits.json (N frame(s): train …, valid …, test …)`, exit 0. It **fails closed**: any staged frame the plan never placed, or whose split is unusable, exits 2 *after* writing the partial map, so you can read which frames are the problem |

Freeze **B** for the distance numbers (A splits strictly by batch, and is worth keeping for a final
acceptance run; after Tier D there is a **C**, which is the one to freeze then). Without
`--split-plan`, the same flag reads the assignment from the project instead — that is the route the
Roboflow appendix uses, and the only one with a deadline, because an assignment that exists only on
the server cannot be read once the machine is offline.

Until `splits.json` exists every frame reads as `?` in the report and an unplaced frame is what §6
refuses — so freeze before the first build, not after the first failure.

## 6. Merge the two generations

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/build_dataset.py --dry-run
./sidecar/.venv/Scripts/python.exe sidecar/tools/build_dataset.py --force
```

v2's set is a continuation of v1's, not a replacement, so the merged set is v1's 1,815 frames (less
the ones the dedup pass drops) plus the locally labeled ones, at the splits §5 assigned:

| Checkpoint | Expect |
|---|---|
| `--dry-run` | `v1` counts per split, `v2 N decided frame(s)` per split, and `v2 M frame(s) with no decision yet`. **Nothing is written**, and it exits 2 on the same refusals the build has — including a `--v2` nothing has been decided in, which is the one a build would otherwise turn into v1 alone |
| the pass gate | `the human pass's gate, on the copy that has not landed yet:` and `no machine-only decisions in valid or test - clean`. A dirty one **exits 2 without touching the previous set** — §8 and §11 both refuse a set built this way, so building it could only trade a usable set for an unusable one. Work §4's pass, then build again; `--allow-machine-only` lands it anyway for *reading* (and says the run will refuse it) |
| the build | every frame is written at `640×640` and **stretched** — the geometry both generations' record requires, made a fact about the files instead of a setting someone has to get right |
| the build | it **exits 2** rather than guessing: a `--v2` that is not a set `clean_v2.py clean` staged (no manifest, or one whose frames are not on disk — v2 is read through its manifest, so a wrong directory merges v1 alone and every check downstream still passes; the hard-negative set named here by a slip for `--extras` is refused the same way), a v2 side that would contribute **no frame at all** — not one has a decision yet, or `--allow-unassigned` drops every decided one — which is the merge that would be v1's frames under the merged name (individual undecided frames are skipped and counted; this is the whole side), a v1 class name it does not know, or a decided frame with no split in `splits.json`. and it runs **the doctor on the staging copy** before the swap, so a merge §7 would refuse never replaces the set it was rebuilding (`--allow-unassigned` skips that gate and says so — it is a reading, not a trainable set). `--force` assembles the new set **beside** the old one and swaps it in only if the build finishes, so a refusal — or a `Ctrl-C` — leaves the previous set byte-for-byte intact |
| `contact_sheet.jpg` | 30 frames, spread across the splits. **Open it.** The merge remaps v1's class *indices* by name (its `data.yaml` declares the seven in a different order), which no check in the toolchain can see — this sheet is the only thing that can |

## 7. Doctor the set before you train it

```bash
make doctor
```

A merged set has been through this check already: `build_dataset.py` runs the same gate on its
staging copy, so what §6 swaps in has been read once — this command is how the verdict is
*reproduced* (and how a set somebody else built is checked), and it is also the gate §8 and §11
refuse on.

Six failures produce a dataset that trains happily and measures the wrong thing, and none of them
raises anything on its own:

| It looks for | Why it is silent otherwise |
|---|---|
| the class names in the wrong **order** | `check_export` compares names by *membership*; every label row indexes the list by position, so the same seven names in another order relabel every box. The block below prints the list **in order** with the generation it judged it against, and when the rows index the *other* generation's order it names that generation and the flag (`--generation v1`) instead of a permutation — relabelling a correct set to satisfy the wrong expectation is the one fix worse than the finding |
| a label row nothing can read | a wrong class index, four fields, a coordinate outside 0..1: the trainer drops the row or clips the box and the frame reads as labeled |
| a frame drawn under a class other than the one it was **staged as** | each v2 frame is filed under `<PRODUCT>/<DISTANCE>/` by `clean_v2.py`, and `build_dataset.py` copies that staged class into `merge_report.json` (`tags`, name → slug) because the built set holds no manifest. A label drawn under a neighbouring product is readable, in range and in a valid box — no other check can see it, and it trains a model that calls one product by another's name. The same rule as §4's mismatch count, applied to the merged set; a report **without** the block (an older build, or a set another tool wrote) is a warning that the check did not run, never a pass |
| a test frame that duplicates a train one | the test number then measures recall of a photograph the model has seen |
| a merge report describing a *different* build | it is what §11 reads its machine-only gate from, and a leftover file looks fine in isolation |
| a merge report whose **v2 side contributed nothing** | three splits, the right names in order, every label readable — because every frame is v1's. Nothing on disk distinguishes the two sources; the report is the only file that says what the set's name promises, and it is what a weight named `scanncart-grocery-v2.pt` would otherwise be trained from (§6 refuses to build one now, so this is the sets built before it could) |

**Pass condition: exit 0 and `[ok]`.** Missing label files are the one to watch for — an *absent*
label is not an *empty* one (the empty ones are the deliberate backgrounds), and a doctor that
conflated them would call a half-labeled set clean. `--json` is the same readout for a script;
`--no-duplicates` skips the only slow part (~6 s over the merged set).

The distance mix is a **reading**, printed with the counts and not a verdict: the `v2 distances:`
line says what §9's per-distance grid will be able to measure (`train close 598; valid close 177;
test close 36` is what a `close`-only set looks like). Two gaps are warnings, because a set holding
only `close` frames is still trainable — it just cannot be evidence about `far`: a distance the
model **trains on** with no `test` frame can never be measured (the split plan is what has to change,
`plan_split.py`), and a distance the plan follows that is in **no** split is §0's capture gap. Both
readings come out of `merge_report.json`, recorded by §6 because the built set does not name the
staged directory whose manifest still holds the axis: the mix per split (`sources.v2.distances`) for
this line, and the same distances per frame (`distances`) for §9's three passes, so the grid does
not depend on that directory surviving. Those blocks count the frames that are **in** the set — a
test frame the build dropped as a duplicate is not one of them — so the two sides add up to the
split counts printed above them; a report whose own numbers do not add up (a side counted where its
frames were staged rather than where they survived) is refused as a report describing a different
set, with the same rebuild that fixes a stale frame count.

`--generation` defaults to **`auto`**: the doctor asks the set which generation it is — the
`generation` its own `data.yaml` records, else the one in `merge_report.json`, else the order its
names are in — and prints how it decided, so the claim is visible rather than assumed:
`dataset: … (generation v2 - from the order its class list declares)`, or `generation v1 - from its
data.yaml`, or `- from the default - nothing in the set says` when nothing answers. v1's export and
the merged set declare the same seven names in different orders, so without this, forgetting the flag
on a v1 set would fail a set that is correct and send you to fix the wrong thing. Name one
explicitly (`make doctor DOCTOR_GEN=v2`) only to ask a deliberate what-if.

This is also a **gate, and not advice**: §8 and §11 run the same check inline and refuse — `train`
exits 2 before the model is even pulled, `accept-v2` exits non-zero before a frame is measured, both
printing this same block (with `doctor` in their `--json` output). So a number can never be
produced over a set with a class list in the wrong order, a label row the loader drops, or a test
frame duplicating a train one. Running it here is how you see the problem early and read it in
full; skipping it does not get you past the gate.

Two shapes count as a label row, because the loader accepts both: `cls cx cy w h`, and the polygon
Roboflow writes for a mask (`cls x1 y1 x2 y2 ...`), which the loader reduces to the box around its
own points — 1,921 of v1's 2,111 rows are that shape, so the count is printed beside each split's
boxes when it is not zero. A file that *mixes* the two is the quiet case: one polygon row makes the
loader read every row in it as a polygon, so the box rows train as boxes nobody drew.

## 8. Train

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py \
  --dataset-dir sidecar/data/datasets/merged-v2 --yes
```

> **v1's local build is the same steps with `--generation v1` and no `--dataset-dir`** — its export
> is already ingested in the dataset workspace. It exists so the app has a natively-loaded model
> before v2's set is finished; `MODEL_TRAINING.md` §6 has the whole chain.
>
> The gate applies to it too, and it is worth knowing before re-running it: **v1's raw export fails
> the doctor** — 24 of its 327 test frames are near-duplicates of train frames, and 3 frames sit in
> files that mix box and polygon rows. Both are fixed by `build_dataset.py`, which drops the
> duplicates and reduces every row to a box as it merges, so the merged set is what every step from
> here on trains and measures on (v1's own numbers included, via §11).

| Checkpoint | Expect |
|---|---|
| the check | the set declares **7** class names; anything else stops here. A distance-bearing list is named as such rather than as "extra classes" |
| the pass gate | `pass gate: acceptance gate: no machine-only decisions in valid or test - clean`, or `Not gated` when the directory is a Roboflow export rather than a set `build_dataset.py` assembled. A dirty one exits 2 with `Nothing was trained:` and no GPU time spent — a weight's unread boxes in `valid`/`test` make the run's own numbers a measurement of the annotator, so the pass is a precondition of the run, not only of §11. **A finished pass does not clear it on its own**: the counts are the build's, so rebuild the set (§6) after working it |
| the doctor | a `classes: 7 declared, in v2's order:` block (the names numbered, then the per-split counts), the `v2 distances:` line (what §9's grid can measure), and `[ok] the set is internally consistent: names in v2's order, every label readable, every staged frame's labels naming its own class …` — the same check as `make doctor`, run inline because this is the command that acts on the set, and judged against the generation this run named. Any `[FAIL]` line exits 2 with `Nothing was trained:` and no GPU time spent; a `[warn]` line about a distance does not stop the run, it says what the grid will not be able to say |
| the augmentation line | `augmentation: fliplr=0.5, flipud=0, degrees=15, scale=0.5, translate=0.1, hsv_v=0.2, mosaic=0, erasing=0` — §4's table, printed because ultralytics' defaults are *not* it (`degrees=0.0`, `mosaic=1.0`). `--degrees`/`--scale` change the two that matter; the rest are the doc's |
| `--yes` | a run directory under the project's runs; `mAP50` ≥ **0.90** |

The two metrics a training log contains are means. `mAP50` clearing 0.90 is necessary and not
sufficient — the number that matters is the next step's.

## 9. Validate — per class **and** per distance

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --val
```

Defaults to `--split test` (the acceptance split), runs three extra passes for the distance
breakdown, and measures at the **`imgsz` the run recorded** rather than at its own default.
**Pass condition: every class at or above the 0.85 floor, and nothing under "below the floor at a
distance".** §8's pass gate applies here too, and for the same reason: this command is where the
quoted number comes from.

The distances come from the **set itself** — `merge_report.json`'s per-frame `distances`, written
by §6 — so the grid runs wherever the merged set is, without the staged directory it was built from.
The run says which source answered (`distances: 811 frame(s) from the set's own merge_report.json`),
an explicit `--manifest` answers first, and the staged manifest is the fallback for a set built
before the map existed.

```
class                                     close          mid          far          all
Milo Chocolate Drink 22g Sachet        0.950 (65)  0.830 (34)!  0.610 (43)!  0.890 (140)
```

(Illustrative numbers, real format: this is the shape the grid prints, not a measurement of any
weight you have trained yet.)

Three states, and the difference is an instruction rather than a style: a number clears the floor,
`!` is below it, and `-` means that distance holds **no instances** of that class — add images of
the item, not to the split. `--no-per-distance` skips the extra passes; a run where nothing
recorded distances says the breakdown was *skipped* rather than reporting nothing.

## 10. Install

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --install
```

Writes `sidecar/models/scanncart-grocery-v2.pt` **plus** `scanncart-grocery-v2.json` beside it,
carrying everything `--val` and the run's own `args.yaml` know and nothing else does:
`resize_mode` (`stretch`), `class_names` (the label set), `imgsz` (the size the app must feed it)
and the `augmentation` table. Refuses to overwrite an existing weight without `--force`, because
the picker is keyed by filename.

| Checkpoint | Expect |
|---|---|
| the Admin Panel's Model field | the weight listed, with `requirement (recorded): stretch` and the measured score |
| `resize_mode` | leave it on **`auto`** — it honours the record. An explicit `letterbox` overrides it and is the one value worth warning about |
| `imgsz` | set `Settings.imgsz` to the recorded value if the run used a non-default one — the record is a readout, not an input, so nothing changes it for you |
| Test connection | `7 classes`, and no class warning. A distance-split weight would read `21 classes` with the distance problem named |

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/spec_check.py --generation v2
```

| Checkpoint | Expect |
|---|---|
| the run | `all 5 PRD targets pass`, or a list naming the ones that did not (`--strict` makes that exit non-zero) |
| `config` | which settings were measured. `saved settings` by default, so the number describes **the app on this machine** |
| the two speeds | `in-app pipeline fps` is the one that decides — `isolated` is the detector alone and will be flattering with nothing around it |
| recall vs `--val` | allowed to differ. This figure is at the app's `conf_threshold`; `--val` picks each class's best-F1 threshold |
| a profile-shaped failure | check `imgsz` against the run before blaming the model: the setting is what the app resizes to, and it is the one field failure that looks exactly like a weak model |

With a camera attached, the same acceptance in the running app:

```bash
node .claude/skills/run-desktop/driver.mjs v1
```

| Checkpoint | Expect |
|---|---|
| the run | **14 `PASS` lines, exit 0**, once a product is in front of the camera — 6 of them before Start, so a camera-less machine still gets the strip half |
| the strip | `stretch` `geometry (auto)` · `stretch` `requirement (recorded)` · the measured `test recall · N below floor` — all derived from the weight's own record |
| the item log | only the seven allowed names, ever. A COCO name means the stock weight is still loaded |
| the running verdict | `7` + `classes · roster ok`. A finding here means the roster was applied to the wrong generation — and `carry a distance` means a distance-split head |

## 11. Accept: does v2 actually beat v1?

```bash
make accept-v2
```

Both weights are measured here, on the merged set's own `test` split, at each weight's recorded
`imgsz` — the baseline is not a number quoted from v1's export. It answers five questions and
**exits 1 with the reasons** rather than printing a verdict:

Each weight is also measured **in its own class order**. A label row's class index is a position in
the *set's* declared list, and a model's head emits indices into the order **it** was trained on, so
the two generations do not share an index-to-name mapping: the run prints which order each weight
used, measures a weight whose order differs from the set's on a remapped view of the same frames
(the class column moves, nothing else does), and **refuses** a weight with no class list recorded
beside it. A baseline scoring ~0 for every class but one is that mismatch, not a weak model — read
the `baseline class order` line before the per-class table.

| Question | Fails when |
|---|---|
| is the set internally consistent? | the doctor refuses it — a wrong class order, a label row the loader drops, a frame drawn under a class other than the one it was staged as, a test frame duplicating a train one. This one exits *before* any measurement, so those numbers are never produced at all |
| is the set fit to be measured? | the merge report says any `machine_only` decision is left in `valid` or `test` (§4's confirm step), or the report does not carry the field at all, or it carries no stamp naming the `provenance.json` those counts came from — or the stamp no longer matches it, which is what a decision saved *after* the build looks like. A gate that cannot be evaluated is a *failure*, not a pass |
| is every class above the floor? | either weight below 0.85 on any class |
| did anything regress? | a class more than the tolerance below v1 |
| **is crowding better, on the cells it was re-shot for?** | of the frames whose **labels** hold two or more items, the candidate finds (reports two or more items on) no fewer than the baseline, on the whole split **and** on every distance in `--claim-distances` (default `mid,far`). Frames a weight reports two items on whose labels hold one are doubled detections: printed per weight, never credited — the raw count used to credit them, and v1 "won" `mid` with 3 doubled frames where the labels hold no crowd at all. A claimed distance with frames but **no truly crowded frame** is *not measured*, like one with no frames — the count is cut by the distance each frame was filed under, because a total can rise while the cell the re-shoot was paid for gets worse (`close` frames are cheap to gain). A claimed distance the split files **no frame at** is a failure, not a pass: `-` in the table means nothing was asked, so `mid` unmeasured is not `mid` held. `CROWD_MIN=2` over the same file list is the claim this dataset exists to make — v1 misses items when several are side by side, and `far` frames are where it is worst |

`--iou-sweep` reports the same count at several NMS thresholds **in each claimed distance's own
row**, which is what separates "these weights find the second item" from "the threshold let a
duplicate through" — a count that only appears at one threshold is the threshold's, not the
model's, and the row it is quoted for is the one that has to say so. `--claim-distances none` is
the run that is not making the per-distance claim at all (the whole-split count only).

## 12. Prove the class-list guard fires, before you trust its silence

§10's reassuring reading — *Test connection: 7 classes, no class warning* — is only a reading if
the check can produce the other one. This step makes the running app meet a weight that really has
a distance in its class names, so both halves of the guard in `MODEL_TRAINING.md` §8.1 are seen
refusing to stay quiet.

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
| the fixture | `scratch weight: <n> bytes` — a real `DetectionModel('yolo11n.yaml', nc=21)` (one head output per product and distance, read from the roster), built and deleted by the mode itself |
| Admin · *Weights on disk* | `… — 21 classes` with ⚠ `21 of 21 class name(s) carry a distance`, **before** the weight is selected |
| Live · banner and chip | the same sentence in `live-class-warnings`, and `21` · `classes · 1 finding` in amber — from a running capture, not a fake status |
| the banner is *painted* | `{"w":952,"h":200,"onScreen":true,"shown":true}`: a non-zero box inside the viewport, because `textContent` cannot tell a rendered element from a collapsed one |
| cleanup | `scratch weight removed` and `settings put back` — both in a `finally`, so they run even when a check throws |

**It needs no v2 weights**, so it runs today, before anything is trained — and that is the point of
doing it *before* §10's reading is trusted: it is a check on the checker. The mode's own notes are
in `.claude/skills/run-desktop/SKILL.md`.

## 13. The acceptance number (Tier D, `s3`)

Only after Tier A has landed. This is what turns the quoted accuracy into an *unseen-session*
estimate instead of a held-out-frame one — until then, `test` is drawn from the same session `train`
saw, and the tool says so outright.

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean \
  --src <s3-capture-folder> --session s3 --out sidecar/data/datasets/cleaned-v2-s3
./sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py \
  --out sidecar/data/datasets/cleaned-v2 --include sidecar/data/datasets/cleaned-v2-s3 --holdout-session s3
./sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py --capture-splits \
  --split-plan sidecar/data/datasets/cleaned-v2/split_plan_c.json
# label s3 in the annotator (--out sidecar/data/datasets/cleaned-v2-s3), then merge:
./sidecar/.venv/Scripts/python.exe sidecar/tools/build_dataset.py --force \
  --extras sidecar/data/datasets/cleaned-v2-s3
```

(Uploading `s3` with `--split-plan …_c.json` is only needed if the project is still in play; the
freeze above is the whole step for the local chain.)

`--extras` **adds** to the default rather than replacing it, so the hard negatives beside `--v2` stay
in the merge once anything else is named — a merge that dropped them would be missing the frames that
teach the model what the products are not, and nothing downstream would say so. A run that means
*only* the sets it names says so with `--no-extras` (`label_progress.py` and the annotator take the
same pair, so the same rule holds for the snapshot and the worklist). And a frame only enters the
merge once it is *decided*, so label `s3` before this.

Then §7–§12 again on the rebuilt merged set — the doctor, the run, the install and the acceptance
take the same commands.

| Checkpoint | Expect |
|---|---|
| `plan_split` | `staged sets: cleaned-v2, cleaned-negatives, cleaned-v2-s3`; **must not** print `CANNOT hold out session s3`. If it does, a cell is covered only by this session — shoot it into an earlier session first |
| the report and the freeze | all of `s3` in `test`, none in `train` — Plan C's own table, and the same split in `splits.json` after the freeze; the Admin Panel's session spread agrees after the next `label_progress` |
| the protocol | quote **`test`**, and say the session. Do not re-shoot or top up `s3` after seeing the result — a test set is spent the moment it is tuned against |

## Appendix — the Roboflow version path

Kept because it is how v1's export was made, and because a version is the only route back to a set
if the labels ever have to be redone on the server: upload the staged session, generate the version
with the geometry in code, and download the export rather than using the **Download** button.

```bash
./sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --dry-run
./sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --check   # reads the project
./sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --yes
./sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --verify 2
./sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --download --version 2
```

`--check` is the one to read before committing: it reads the project and prints everything `--yes`
would bake in — the class list **in order** with any extra/missing class named against the roster,
the geometry being sent, and how many images carry a decision (`858 labeled, 0 marked null, 575 not
looked at yet` is what the project looked like the last time this was run). Only annotated frames
enter a version, so that last number *is* the size of the set it would freeze — a version is a
snapshot, not a work in progress, and it cannot be revisited. Spending nothing is the point: it is
the same readout `--yes` prints one line before the POST.

In Code, **Download → YOLOv11 PyTorch** if you must do it by hand: a version page's *Download*
gives a zip, and the export is only a build once it appears on the page (the polling is what
`--download` automates).

A version number is consumed and cannot be reused, and `--yes` (or `--check`) refuses **before** the
POST — fail closed, exit 2, nothing spent — on any of three states: a class list carrying a
*distance* (it would train one output per product-and-distance), one whose names are the roster's in
**another order** (the export declares the project's order, so the version would bake a head whose
indices mean a different product, and the first check downstream that can see it is the doctor, on an
export that already exists), and a project where **nothing is annotated** (the version would freeze
an empty set). The first two print their two lists side by side: the fix is Settings → Classes, and
the order is not settable by API.

A class that is present but not in the roster — an extra, a missing name — is **reported as a
warning, not refused**: that verdict belongs to `label_classes.py`, which is the tool that reads the
live class list and can create a missing class, and its remedy is the Classes tab. It is printed
loudly because a version declares every class it is given, so the head ends up with outputs the app's
roster cannot name. (When this was last checked, the project still held `Palmolive Naturals Bar Soap
85g` — dropped from the roster in §8.1 — as class 2 of 8.)

---

## What "done" looks like

| | |
|---|---|
| labels | `1433/1433 decided`, 50 nulls marked, 0 machine-only, no class mismatches |
| split | `splits.json` frozen from the chosen plan (no API needed), and no decided frame missing from it |
| set | `build_dataset.py` exits 0, `make doctor` says `[ok]` (and §8/§11 refuse without it), the contact sheet reads right |
| weights | `scanncart-grocery-v2.pt` + its record, `auto` honouring `stretch`, `imgsz` matching the run |
| measured | every class ≥ 0.85 **and** every distance cell reported, unbracketed by `!` |
| accepted | `make accept-v2` exits 0: the gate clean, no class below the floor, and **no fewer truly crowded frames found than v1** — whole split **and** on every claimed distance, with the `mid`/`far` rows holding at least one frame whose labels have two or more items |
| quoted | the `test` score, labelled with the capture session it came from |
