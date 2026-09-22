# SCANnCART v2 — capture checklist for the missing buckets

> What to shoot next, in priority order, and exactly how to file it so the cleaning and
> upload tooling picks it up without rework.
>
> Related: [MODEL_TRAINING.md](./MODEL_TRAINING.md) §2 (the buckets) · §8 (this project's
> roster, weight naming, and split rule).

## Why this exists

The v2 set (1,383 images in `snc-grocery`) is *distance-aware*, which is real progress over v1,
but an audit against §2 found it is missing the two buckets that carry the most value per label,
and that 8 of its 24 product×distance cells are too thin to measure at all.

Reproduce the numbers any time with:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/audit_v2.py     # -> sidecar/data/datasets/cleaned-v2/AUDIT.md
```

**Current shape:** 1,383 images → 1,015 close / 152 mid / 216 far, across 21 of 24 cells.
**§2 target:** ~1,200 solo + ~640–960 multi-item + 150–250 hard negatives.

## Priority order

| Tier | What | New images | Why it comes first |
|------|------|-----------:|--------------------|
| **A** | Rescue the 8 cells below 20 images | ~253 | A cell that small cannot be validated — it contributes almost nothing to a 20% valid split, so a per-class × per-distance reading is impossible |
| **B** | Bring `mid` up to 100 per class | ~202 | Mid is the axis v2 exists to fix, and it is still the smallest by a wide margin |
| **C1** | Multi-item scenes | ~640 (80/class) | §2's highest value per label: one image feeds 4+ classes and it is what the app actually sees |
| **C2** | Hard negatives | ~150–250 | Cheapest possible way to kill false positives — but read the caveat in §C2 first |
| **D** | A held-out session for the acceptance number | ~360 | Nothing in A–C can fix this: while every batch is one session, `test` is held-out *frames* of a session `train` saw, so no split of this set can support an unseen-session claim. It comes last because it is a **re-shoot** — it can only be planned once A–C have put every cell in `train` |

Doing A + B takes the set to **1,838** images and lifts the distance spread from
1,015/152/216 to **1,055/500/283** — mid goes from 11% of the set to 27%. That is the single
highest-leverage change available.

---

## Tier A — the 8 cells that cannot be measured

Target 40 per cell. At 70/20/10 that puts ~8 in valid and ~4 in test: enough to see whether a
class is failing at a distance, not enough for a tight estimate — which is the point. These are
8 of the 21 populated cells; 12 are under 40, and none of the 24 is empty by accident — the three
missing cells are exactly the three below.

| Class | Distance | Have | Capture | Why |
|-------|----------|-----:|--------:|-----|
| `century-tuna` | mid | 0 | **40** | Empty cell — the class is invisible at mid distance |
| `century-tuna` | far | 0 | **40** | Empty cell — likewise at far |
| `palmolive` | close | 0 | **40** | Empty cell, and Palmolive is the one *new* class v2 adds |
| `silver-swan-vinegar` | mid | 5 | **35** | Near-empty; the 5 that exist are one batch |
| `lucky-me-pancit` | far | 13 | **27** | Second-largest class by close images (234 of 266) but almost nothing far |
| `palmolive` | mid | 13 | **27** | New class, thin everywhere |
| `555-sardines` | mid | 17 | **23** | One batch, and the class is small overall (79) |
| `lucky-me-pancit` | mid | 19 | **21** | Same gap as its far cell |

**Tier A total: 253 images.**

### Lay out the folders before you shoot, not after

The product folders in the table above are the ingest contract: `clean_v2.py` maps a folder to a
class by **exact name**, and a name it does not recognise is skipped with one `[warn]` line in a
report nobody reads until after the session. The 8 cells live in 5 folders — `TUNA`,
`PALMOLIVE`, `SARDINES`, `LUCKY ME` and `Silver Swan` — two of them two words, and one of those in
mixed case while its siblings are all-caps, which is exactly where a hand-made tree goes wrong.
`scaffold` emits the tree from the same `CLASS_MAP`/`DISTANCE_MAP` the ingest uses, so the folders
it creates are by construction ones the ingest accepts:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py scaffold --root <capture-folder> --dry-run
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py scaffold --root <capture-folder>
```

It writes a `README.md` in that folder listing each cell's target image count and the exact
`clean` → `upload --session s2` → `retag --session s2` commands to run when the session is done
(the projection "1383 → 1636" only prints when the staged set can actually be counted). Do not
rename the folders afterwards.

**The same gap is visible in the app**, so you do not need this file open while shooting. The
Admin Panel's *Dataset labeling* section carries a **capture gap (Tier A)** block — `186 of 253
images still to shoot across 8 cell(s)`, then one line per cell with what is left and its
`have/target` (`0/40` for the empty ones). It comes from `label_progress.py`, which reads these
targets from the same table `scaffold` builds folders from, and it shrinks as a capture session
lands. Its numbers are a snapshot of the last tool run, like the rest of that panel — re-run
`label_progress.py` (and press *Refresh*) after uploading a session to see it move.

Two of these deserve special care. `century-tuna` and `palmolive` have **no** mid *or* close
reference at all in one direction, so shoot their missing cells first and in one sitting, at the
same distances and lighting as the rest — a mid shot that doesn't match the existing mid
distance is worse than no mid shot.

---

## Tier B — the mid axis

Target 100 per class at mid. These are the three mid cells Tier A does not already cover — every
other mid cell is empty or under 20 — and all three are still well short.

| Class | Have at mid | Capture | Note |
|-------|------------:|--------:|------|
| `bear-brand-milk` | 23 | **77** | |
| `milo` | 34 | **66** | |
| `safeguard` | 41 | **59** | Already the best-covered class at mid |

**Tier B total: 202 images.**

Why 100 and not §2's 250–350 per class: §2's number is *per class*, and v2 spreads each class
across three distances. 100 per cell is ~300 per class where all three cells fill, which lands on
§2's recommended tier — while keeping the distance dimension wide enough to be worth having.

---

## Tier C1 — multi-item scenes

Target **80 scenes per class** (~640 total), 3–6 roster products per frame, overlapping and
partially occluded. This is the bucket a solo-shot set cannot substitute for: one frame with four
products teaches all four at once, and it is the composition the counter actually produces.

Shoot it as its own session, not as an afterthought to solo work:

- 3–6 products together, **touching and overlapping** — occluding 30–50% of at least one item.
- Include the *visually similar* pairs from §4: `555-sardines`/`century-tuna` (both tins),
  `lucky-me-pancit`/`milo` (both sachets). These are where the model flips, and one flip logs a
  single physical item as two products.
- Mix distances inside the frame — some items near, some far. That is the real counter.
- Vary arrangement between shots: grid, pile, cluster, one item rotated.

Every object that belongs to the 8-class roster gets a box. That is what makes a scene count.

### Why this tier is load-bearing, measured rather than assumed

`audit_recall.py` splits a trained weight's per-instance recall by how many objects each frame's
labels carry, and on the v1 model the split is the whole story:

```
single  1 object      265/265   100.0%   (every single-object frame found)
multi   2+ objects    103/136    75.7%
```

**Not one single-object instance in v1's 327-frame test split was missed** — the recall the 0.85
floor is computed over (0.918) is entirely the crowded bucket pulling it down, and v1 was shot as
per-class solo captures. So every point of recall that solo shots could buy is already bought, and
C1 is the only bucket left that moves the number. Two consequences for how to shoot it:

- **Each class has to appear in the occluded position, not always as the hero object.** The misses
  are consistently the second instance — the one behind, or smaller, or partly covered. A set where
  `safeguard` is always the front item teaches the model `safeguard` and teaches it nothing about
  finding a safeguard behind two tins. Rotate which item is nearest, largest and fully visible.
- **Two-item frames are the best-covered crowded case, so go past them.** v1's crowded bucket is
  mostly pairs. Aim the bulk of C1 at 4–6 items, and put the similar-SKU pairs in a frame that
  already holds other products, which is the version a counter produces and the one that hides the
  second tin.

## Tier C2 — hard negatives

Content to capture, straight from §2: an empty counter, hands, bags, a wallet, a phone, the
conveyor.

**The mechanism is confirmed to exist, and it is a UI action.** Roboflow represents a background
image as a **null annotation**: an image deliberately marked as containing no object of interest.
You mark it in the annotator with the **Mark Null** tool — the empty-set button in the right-hand
toolbar, keyboard shortcut **N** — and it then counts as annotated, so it is eligible to enter a
generated version. A plain unannotated image is not: Roboflow excludes it from the version and
counts it under the project's `unannotated` total.

The trap is the **"include images without annotations"** toggle at version generation: it sweeps
in *every* unreviewed image, not just the ones you meant. Mark the ~30 frames individually with
the null tool instead of using it.

**This cannot be done from the upload API.** The API exposes no null flag, and the annotate
endpoint rejects an empty annotation file (`Unrecognized annotation format`) because it sniffs the
body to infer the format. For 30 frames, marking them by hand is the sane route — so C2b is cheap
but not scriptable, which is the main reason to keep it small.

**Verifying it worked is scriptable, though.** A null annotation is distinguishable from an
unlabeled image by the *type* of the search API's `annotations` field:

| `annotations` | Meaning |
|---|---|
| `[]` (empty list) | not labeled yet |
| `{"count": 0, "classes": {}}` (object) | **null annotation** — deliberate background |
| `{"count": n>0, ...}` (object) | labeled |

`count: 0` inside an object is a decision someone made; no object at all is outstanding work.
This is confirmed against the v1 project, which returns an object for all 1,516 images with **16
of them at `count: 0`** while reporting `unannotated: 0` — so v1 carries a small hard-negative
set, and the mechanism has been used before. `label_progress.py` lists nulls explicitly:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py
```

Run `sanity` (below) before shooting to see this verdict, the class list, and the version
preprocessing in one place.

Both variants, and what is safe either way:

| Variant | Looks like | Files under | Enters a version? | Work to file it |
|---------|-----------|-------------|-------------------|-----------------|
| **C2a (do this)** | One roster product on the counter **plus** the clutter you want ignored — a hand reaching in, a phone, a bag, a wallet | that product's folder | ✅ yes — the product's box earns it a place, and every unlabeled object teaches background | none beyond normal labeling |
| **C2b (also do)** | Purely empty counter, no roster product at all | `NEGATIVES/` | ✅ yes, **once each frame is marked with the null tool (N)** — an unmarked one is silently excluded | one keystroke per frame in the annotator |

Shoot **C2a for the bulk** of Tier C2: it needs no extra step and it targets exactly the false
positives that matter (hands placing items, a phone on the counter). Add the small C2b set (~30
frames) for pure background — now that the route is confirmed, the only cost is pressing **N** on
each one before generating the version.

**Some of C2b already exists.** An earlier session captured **60 frames off the deployment camera**
(`sidecar/data/datasets/hard_negatives/images/cam0_*.jpg` — empty labels, 4.1 MB). The cleaner kept
**50** of them (10 collapsed as near-duplicates, which is what a static mount produces). Eyeball
them before uploading: they are only worth including if they really are empty counter.

```bash
# stages 50, batch `negative`, tag `negative` — into its own out dir, no distance needed
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean \
  --src sidecar/data/datasets/hard_negatives \
  --negatives sidecar/data/datasets/hard_negatives/images \
  --out sidecar/data/datasets/cleaned-negatives

# then upload them as their own session batch, and mark each null (N) in the annotator
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload \
  --out sidecar/data/datasets/cleaned-negatives --session neg1
```

They are a **new session**, so upload them with a `--session` suffix: `negative_neg1` is then its
own batch, which is what keeps §8.3's split-by-batch a split by session. And like every other
session added after the fact, they are not in `split_plan_b.json` — re-run `plan_split.py` if you
want them spread across train/valid/test rather than defaulting to train.

> **Do not skip the null-marking step.** An uploaded `NEGATIVES/` frame that is never marked is
> indistinguishable, to Roboflow, from an image nobody got around to labeling: it is excluded
> from the version and shows up in the `unannotated` count instead. `sanity` reports that count,
> but it cannot tell the two cases apart — so the check that a C2b set made it in is the
> *version's* image count after generation, not the project's.

---

## Tier D — the held-out session (`s3`), for an acceptance number worth quoting

**Why this is a separate shoot and not "30 more frames of everything".** A–C all add to `s1` (and
`neg1`), the single capture every current batch belongs to. Under Plan A and Plan B alike, `test`
is drawn from that same session, so an accuracy number computed on it is measuring performance on
a **held-out frame** — same rig height, same day, same lighting, sometimes the same half-second as
a training frame. The question a capstone actually asks ("does this work on a new capture?") needs
a session `train` never saw. That is what `s3` is, and it is the only thing that makes the quoted
number an *unseen-session* estimate.

**The rule, and it is the one thing to get right: `s3` must re-shoot cells `s1`/`s2` already
cover.** Holding out a whole session removes its frames from `train`, so a cell whose *only*
coverage is `s3` ends up with no training images at all — the model is never taught it, and its
test reading then measures the absence of training rather than the model. This is checked in code,
not left to discipline:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py --holdout-session s3
```

It pins every frame of `s3` to `test`, splits the remainder train/valid, writes the plan to
`split_plan_c.json`, and **refuses** (exit 2, nothing written) if holding the session out would
leave any cell unlearnable. Pass `--include <s3-staged-dir>` when the held-out session is staged
separately (which it is, per the tier's own scaffold) — a planner reading only s3 would call every
cell s3-only and refuse a plan that is in fact fine:

```
CANNOT hold out session s3: it is the only coverage of 2 cell(s), so holding it out leaves them
with no train images and nothing for the model to learn:
    century-tuna/mid
    palmolive/close
```

That failure has exactly two fixes, and both are capture decisions rather than planning ones:
shoot the named cells into an earlier session, or drop them from the held-out shoot. Until `s3`
exists the same command says so and names the sessions it does know.

**What to shoot: 15 per cell, across all 24 cells** — every product at every distance, so after
Tier A the whole grid is covered. `scaffold --tier d` lays the tree out (and prints these exact
commands in its README), so the folders cannot be mistyped:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py scaffold --tier d --root <s3-capture-folder> --dry-run
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py scaffold --tier d --root <s3-capture-folder>
```

It also **answers the rule above before you shoot**, by reading the staged sets and naming the
cells whose only coverage would be this session. Today, with 21 of the 24 cells covered:

```
held-out coverage check: 3 of these 24 cell(s) have NO images outside s3, so shooting them
only into this session leaves them with no train images - `plan_split --holdout-session`
will refuse the plan (exit 2):
    palmolive/close
    century-tuna/far
    century-tuna/mid
  Shoot these into an earlier session (they are Tier A's own cells) or drop them from
this shoot. Everything else here is safe to re-shoot into s3.
```

Those three are exactly the empty cells Tier A exists to fill, which is the answer stated as a
capture decision rather than discovered as a planning error. The check runs on `--dry-run` too,
and every remaining cell is safe. Nothing staged yet reads as *cannot tell* rather than as 24
alarming lines — with no coverage there is no answer, and the fix is to stage the earlier sessions
first. It reads `<workspace>/cleaned-v2` by default; pass `--include <dir>` (repeatable) for any
further staged set, e.g. once Tier A's session is staged into `cleaned-v2-s2`.

| | Cells | Per cell | Images |
|---|---:|---:|---:|
| close / mid / far × 8 classes | 24 | 15 | **360** |

The 15 is chosen so the *aggregate* is quotable rather than the cell: it puts ~120 images on each
distance, which is a per-distance reading you can defend, and ~10–15 per (class, distance) cell,
which is enough to notice that a class collapsed without pretending to a tight per-cell estimate.
Covering every cell is what keeps the estimate unbiased — choosing the cells you expect to do well
on is how a held-out set stops being one. **Commit to the sample before you see any result**: same
15-per-cell grid, decided here, not after a model has been trained once. (Note this is why `test`
lands well above its 10% target: test *is* `s3`. A session big enough to quote is worth more than
hitting the ratio.)

Run **Tier A first** if you have not: a cell that only `s3` covers would have no train images.
Shoot it as one sitting on a **different day** from the `s1` frames — that is the entire variable
being isolated, so a same-day re-shoot buys much less than it looks like it does. Reuse the taped
close/mid/far marks, and file it under a new session so it gets its own batches. **The order below
is not interchangeable**: the split is set at upload time, and re-uploading an image later returns
`{"duplicate": true}` and leaves its split as it was — so the acceptance split has to be planned
between `clean` and `upload`.

```bash
# 1. stage (writes s3's manifest; the planner reads the staged set, not the project)
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean \
  --src <s3-capture-folder> --session s3 --out sidecar/data/datasets/cleaned-v2-s3
# 2. plan the acceptance split ACROSS sessions - without --include the planner sees only s3
#    and refuses, correctly for what it can see: every cell would be s3-only
sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py \
  --out sidecar/data/datasets/cleaned-v2 --include sidecar/data/datasets/cleaned-v2-s3 --holdout-session s3
# 3. upload with that split, then retag
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload \
  --out sidecar/data/datasets/cleaned-v2-s3 --session s3 \
  --split-plan sidecar/data/datasets/cleaned-v2/split_plan_c.json
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py retag \
  --out sidecar/data/datasets/cleaned-v2-s3 --session s3
```

**Then the protocol, which is short but has one rule:** generate the version from that plan, train
on `train`, select on `valid`, and quote **`test`** as *performance on an unseen capture session* —
and say the session, because "unseen" is a claim about provenance, not about a percentage. Do not
top up, re-shoot or re-upload `s3` after seeing the result: a test set is spent the moment it is
tuned against, and this one is ~315 images that cost a session to build.

The Admin Panel's *Dataset labeling* section shows each session with its train/valid/test spread,
so you can see the moment this lands — `s3` appearing with all its frames in `test` and none in
`train` is the signal that the acceptance number is now an unseen-session one.

---

## How to shoot it

- **Same rig, same mount height, same resolution** as the existing captures. Domain match beats
  volume: 200 frames from the real rig outperform 1,000 off-rig ones.
- **Re-mark the three distances.** Put tape on the counter at the close/mid/far positions and use
  the same marks for every class and every session. If the original marks are gone, match by
  on-screen size against an existing image in each bucket rather than guessing — a "mid" shot at
  the wrong distance lands in a cell that is already thin.
- **Extract frames at ~1 fps**, not 30. The cleaner collapses near-duplicates, but the cheaper
  move is to not shoot them.
- Work §3's variation axes deliberately within each cell: orientation, occlusion, lighting,
  background, deformation. A cell of 40 near-identical frames is worth less than 20 varied ones.

## How to file it

The cleaning tool maps **product folder name → class** and **distance folder name → tag**. Use
these exact names, or add to `CLASS_MAP`/`DISTANCE_MAP` in `sidecar/tools/clean_v2.py` first:

```
<your new capture root>/
    BEARBRAND/       CLOSE/  MID/  FAR/
    LUCKY ME/        CLOSE/  MID/  FAR/
    MILO/            CLOSE/  MID/  FAR/
    PALMOLIVE/       CLOSE/  MID/  FAR/
    SAFEGUARD/       CLOSE/  MID/  FAR/     # close-up / Mid-shot / Far-shot also accepted
    SARDINES/        CLOSE/  MID/  FAR/
    Silver Swan/     CLOSE/  MID/  FAR/
    TUNA/            CLOSE/  MID/  FAR/
    NEGATIVES/       CLOSE/  MID/  FAR/     # Tier C2b only, if hand-filed
```

Only create the cells you are filling — an empty folder is harmless, it just reports a gap.
HEIC, JPG and PNG all work; the tool converts, fixes orientation, strips GPS EXIF and dedupes.

**Frames off the app's own camera need no distance**, and no tree: pass `--negatives <folder>` to
`clean` and a flat directory is staged under the `negative` pseudo-class with an empty distance.
That is deliberate rather than a shortcut — a null-annotated frame carries no box, so close/mid/far
would be a value nobody observed, and it would land in the coverage report as a real distance cell.

## Ingest it

**Use a new output directory per session, and name the session.** `clean` clears the `*.jpg` files
in its output's class folders before staging, so pointing `--out` at `cleaned-v2` would wipe
session 1's staged set. `--session` goes on **all three** commands: at `clean` it is written into
the manifest (which is what `plan_split` reads) and into the tags the images should end up with, at
`upload` it keeps the new batches separate (`milo_mid_s2`, not `milo_mid`), and at `retag` it is what
actually stamps the session tag on the images. Forget it at `clean` and the new capture is recorded
as session 1 - the split planner then treats two sittings as one, silently.

```bash
# 0. before shooting: does the project match what this checklist assumes?
#    (class list, version preprocessing, tag/manifest agreement, hard-negative verdict)
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py sanity

# 1. clean the new capture (own src, own out)
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean \
  --src "C:\Users\yusri\Downloads\Downloads\PRODUCTS\CLOSE MID FAR s2" \
  --session s2 \
  --out sidecar/data/datasets/cleaned-v2-s2

# 2. upload into session-tagged batches (milo_mid_s2, not milo_mid)
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload \
  --out sidecar/data/datasets/cleaned-v2-s2 --session s2

# 3. add the class tag, the session tag, and the distance/session metadata
#    (only ONE tag survives the upload call, so the class and session land here)
sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py retag \
  --out sidecar/data/datasets/cleaned-v2-s2 --session s2
```

Both `upload` and `retag` are resumable, so an interrupted run just continues. Check the plan
first with `upload --dry-run`, and re-run `sanity --session s2` after `retag` to confirm the session
tag landed on every image — `plan_split` reads the session from the manifest and assumes the tags
agree.

## Name the classes

Before labeling, the images must carry the exact v1 class names so labels stay continuous with the
1,815 already-annotated v1 images. Two hard limits, both verified against the API:

- **A tag cannot hold a class name** — the tag API rejects spaces, and three of the seven v1 names
  have them. Tags keep the short slug.
- **A class's *order* and Lock Classes cannot be set by API** — those live on the project's
  *Settings → Classes* tab only. So the tool prints the list in the order to create it, and you set
  **Lock Classes** by hand before labeling.

Class *creation* is a third thing: undocumented, but reachable — annotating a throwaway image with
an unknown class name registers that class, and it survives deleting the image. `--create-classes`
does exactly that. It is slow to read back, so re-run without the flag to confirm.

So `label_classes.py` writes the exact name into `class_name` **image metadata** (which has no such
restriction), validates the mapping against the live v1 project, and creates any class that is
missing:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/label_classes.py --apply            # write metadata
sidecar/.venv/Scripts/python.exe sidecar/tools/label_classes.py --create-classes   # seed classes
```

## Split it

Then plan the split, before generating a version. `plan_split.py` computes two plans against the
real batch structure and reports the coverage of each:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py --commands
```

| | Plan A — strict by-batch | Plan B — stratified per cell |
|---|---|---|
| Honours §8.3's session rule | yes | gives it up *within* a cell |
| Unmeasurable class-slots | **6** | **0** |

Use **Plan B** while iterating on distance robustness — under Plan A a weak mid-distance result
cannot be told apart from an unlucky batch assignment. Keep **Plan A** for a final acceptance run,
where the honest number on an unseen session is what matters.

Both plans draw `test` from the same session their `train` came from, which is what Tier D's
held-out session fixes. Once `s3` exists, plan the acceptance run against it directly:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py --holdout-session s3
```

That writes **Plan C** — all of `s3` in `test`, the rest split train/valid, and a per-cell table of
what the acceptance set does and does not cover. It refuses rather than warn if holding the session
out would leave a cell unlearnable (Tier D explains the fix). One thing to expect: Plan C
deliberately misses the 70/20/10 target on `test`, because `test` *is* the session — a session big
enough to quote is worth more than hitting 10%.

Applying a plan is not symmetric, and the difference is worth knowing before you start:

- **Plan A applies non-destructively.** A batch *is* its two tags, so `tag:bear-brand-milk tag:close`
  selects exactly that batch: paste the query (printed per batch in `SPLIT_PLAN.md`), *Select all
  matching*, then *Change Dataset Split* from the bulk actions menu. 21 actions, nothing deleted.
- **Plan B needs a wipe and re-upload.** It varies the split *inside* a batch, which no tag query can
  express, and an image's split cannot be changed by re-uploading it (identical bytes return
  `{"duplicate": true}` and leave the split as it was). A wipe deletes every image, so re-run
  `label_classes.py --apply` afterwards.

## Generate the version

Once the labeling is done, the version is generated **with the reviewed settings**, not by
clicking through the UI:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --dry-run   # show the body
sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --yes       # generate
sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --verify 2  # read it back
```

Generating from that file is not automation for its own sake: the preprocessing is a one-shot
decision that bakes the geometry every training image is stored at, it cannot be edited afterwards
(a new set of settings means a new version number), and it is the largest silent accuracy lever in
the pipeline. `sanity` cannot pre-set it, and `--verify` is what makes the *resulting* version
checkable instead of assumed.

The setting that matters is `auto-orient` + **`Stretch to` 640×640**, and it is the same one v1 used
(`scanncart-grocery/1`, mAP 98.21). At 1280×720 the horizontal axis is the binding one, so stretch
and letterbox both downscale x by 0.5 — but stretch keeps the vertical scale at 0.889 where
letterbox drops it to 0.5, so every object keeps ~1.8× more vertical pixels. That is exactly what
the `far` cells are short of.

**The condition that comes with it:** these weights have to be *run* stretched, while
`resolve_resize_mode()`'s format heuristic answers *letterbox* for a locally trained `.pt`. What
closes that gap is the **record** `train_model.py --install` writes beside them: `auto` honours it, so
the default is the geometry they trained at. Without a record (a hand-copied weight) `auto` falls
back to the heuristic and letterboxes every object to 0.56× the canvas it was trained at — no
error, no warning, just weaker detections where they were already weakest. The Admin Panel's entry
for the weights says which of the two it is.

## Checklist

- [ ] Create the Tier A folder tree: `clean_v2.py scaffold --root <capture-folder>` (do not rename
      the folders — a typo makes `clean` skip that product silently)
- [ ] Tier A: `century-tuna` mid + far (40 each)
- [ ] Tier A: `palmolive` close (40) — the new class, currently zero at close
- [ ] Tier A: `silver-swan-vinegar` mid (35)
- [ ] Tier A: `lucky-me-pancit` far (27) and mid (21)
- [ ] Tier A: `palmolive` mid (27)
- [ ] Tier A: `555-sardines` mid (23)
- [ ] Tier B: `bear-brand-milk` mid (77), `milo` mid (66), `safeguard` mid (59)
- [ ] Tier C1: multi-item scenes, 80 per class, with the similar-SKU pairs inside at least 20 of them
- [ ] Tier C1: each class shot in the occluded/background position in at least 20 scenes — not
      always the nearest item, since the misses are the second instance and not the first
- [ ] Tier C1: the bulk of the scenes at 4–6 items, not only pairs (v1's crowded bucket was mostly
      two-item frames, and that is the crowded case the model already handles)
- [ ] Tier C2a: hard negatives shot as *one roster product + clutter* (hands, phone, bag, wallet)
- [ ] Tier C2b: ~30 pure background frames, then mark each one with the null tool (**N**)
      (an earlier session left 60 StreamCam frames with empty labels in
      `sidecar/data/datasets/hard_negatives/` — eyeball those before shooting new ones)
- [ ] Mark the C2b frames null **before** generating a version, or they are silently excluded
- [ ] Confirm it registered: `label_progress.py` reports the expected number of null annotations
- [ ] Label every cell in the order under [The labeling order](#the-labeling-order), then re-run `label_progress.py`
- [ ] Mark the 50 negative frames **null** — not skipped, not auto-labeled ([The null rule](#the-null-rule-50-frames-and-they-are-the-whole-point))
- [ ] Generate the version from the reviewed settings: `generate_version.py --dry-run`, then `--yes`
- [ ] `generate_version.py --verify <n>` — it must say it *matches*, and report the image count
- [ ] Download the version's export: `train_model.py --download --version <n>`, then check it with
      `train_model.py` — it must not report a class-list mismatch
- [ ] Train: `train_model.py --yes` (`yolo11s.pt`, `imgsz=640`), target mAP50 ≥ 0.90
- [ ] Validate: `train_model.py --val` — recall ≥ 0.85 for **every** class, and no `[WARN]` or
      `[SKIP]` line (a skipped class was never measured: the split holds no instances of it)
- [ ] Read the class × distance grid `--val` prints as well: nothing under *below the floor at a
      distance*. The per-class floor averages the three distances, so a class can pass it on a
      test split that happens to be mostly `close` while missing the item at `far` — which is the
      bucket this whole checklist exists to fill (MODEL_TRAINING.md §6). `-` in a cell means that
      distance held no instances of the class, `!` means it missed the floor there
- [ ] Install: `train_model.py --install --version <n>` → `sidecar/models/scanncart-grocery-v2.pt`,
      plus the `.json` record beside it carrying the `resize_mode` these weights need **and**
      `--val`'s per-class recall — the install prints the score it picked up, or says none was
      found, so a panel that shows nothing is never a surprise
- [ ] Leave **`resize_mode` on `auto`** — it uses the recorded requirement. The Admin Panel's
      Model field lists that requirement and flags an explicit value that contradicts it, so this
      is a check rather than something to remember
- [ ] Re-mark the close/mid/far distances before shooting, and reuse them across every class
- [ ] Confirm you are happy with the project being **public** before `upload` — the free plan's
      default, with no per-project toggle (see MODEL_TRAINING.md §8.4)
- [ ] Run `clean` → `upload --session s2` → `retag --session s2`
- [ ] Re-run `clean_v2.py sanity` and confirm the class list and preprocessing are right
- [ ] `label_classes.py --create-classes`, then **Lock Classes** on *Settings → Classes*
      (creation is scriptable; the order it lists them in is what makes keyboard labeling predictable)
- [ ] `label_classes.py --apply`, so every image carries its exact v1 class name as metadata
- [ ] `plan_split.py`, then apply Plan B (wipe → `upload --split-plan` → `retag` → `label_classes`)
- [ ] Tier D: shoot `s3` — 15 per cell over every populated cell (~315), on a **different day**,
      re-shooting only cells `s1`/`s2` already cover (a cell only `s3` holds becomes unlearnable)
- [ ] `plan_split.py --holdout-session s3` — it must **not** print `CANNOT hold out session`
- [ ] Upload `s3` with `--session s3 --split-plan split_plan_c.json`, then confirm in the Admin
      Panel that `s3` sits entirely in `test`
- [ ] Quote the version's `test` number as *unseen capture session `s3`*, and do not retune against it
- [ ] Re-run `audit_v2.py` and confirm the tiers closed

## The labeling order

Label **whole cells, largest first** — a cell is one class × one distance (`safeguard @ close`).
`label_progress.py` already prints them in that order under *"N cell(s) still have unlabeled
images; largest first"*, so the order is not something to decide by feel:

| Order | Cell | Frames | Why here |
|------:|------|-------:|----------|
| 1 | `silver_swan_sukang_puti_200ML @ close` | 242 | Biggest single cell; close range, so boxes are unambiguous and drawing is fast |
| 2 | `lucky_me_pancit_canton_calamansi_flavor @ close` | 234 | Second biggest, and one half of the similar-SKU pair (§4) that needs volume to stay separable from `milo` |
| 3 | `safeguard_pure_white_60g @ close` | 184 | Third biggest, and `safeguard` is a v2-only class (v1 holds 0 images of it), so its own volume is all it has |
| 4 | `century_tuna_flakes_in_oil_155_grams @ close` | 150 | Its *only* populated cell — labeling it is what makes the class trainable at all |
| 5 | `Bear Brand Fortified Powdered Milk 33g @ close` | 106 | Last of the 100+ cells; after this every session is a smaller cell |
| 6 | `Milo Chocolate Drink 22g Sachet @ close` | 65 | The other half of the `lucky_me`/`milo` pair — label it while both are fresh |
| 7 | `safeguard_pure_white_60g @ far` | 54 | First `far` cell: the hard end, and the bucket this dataset exists to fix |
| … | the remaining 14 cells | 5–43 each | Straight from the tool's list |
| last | `(negative: background frames, nothing to draw)` | 50 | **Not drawn at all** — see below |

Three rules behind that order:

- **Whole cells, not a scatter.** The split is by batch (§8.3 of MODEL_TRAINING.md), so a batch is
  labeled or it is not — a half-labeled cell is a half-labeled cell in *every* split, and the
  per-class `--val` recall floor cannot be read off it.
- **Biggest first.** Throughput per labeling session, and it front-loads the failure that actually
  stalls a project: one distance bucket nobody opened. A 60% total with `century-tuna` at zero is
  worse than a 40% total spread across every class.
- **`close` gets you to a trainable version soonest; `far` is where review costs the most.** A
  foundation model's boxes are most reliable on large `close` objects and weakest at `far` (20–40 px
  across), which is also where a wrong box hurts most. Pre-label (§8.1) the `close` cells; draw the
  `far` ones yourself.

### The null rule (50 frames, and they are the whole point)

The 50 frames in the `negative` batch are the Tier C2b hard negatives — background and clutter with
**no roster product in them**. They are not "images nobody got to yet":

| Action | What happens to them |
|---|---|
| **Mark each null (`N`)** | ✅ Included in the version as negatives. This is the only correct action for all 50 |
| Leave unannotated | ❌ **Excluded** from the version. The hard-negative set silently disappears — no error, no warning |
| Draw a box on one | ❌ Teaches the model that a shelf, a hand, or a phone is a product |
| Auto-label them | ❌ Same as above, at scale: a labeler searches for objects and these frames are defined by having none |

`generate_version.py --verify <n>` refuses a version that filters nulls, so a version that quietly
dropped all 50 is caught at the generate step rather than at training time. `label_progress.py`
reports them as a separate line (*"mark each null (N), do not draw"*) and counts them in the total,
which is why the snapshot reads `0/1433` rather than `0/1383`: the 50 nulls are part of the target,
not an allowance.

The nuance that makes them work: **a null means "no roster object here"** — the C2a frames (one
product *plus* clutter) are not nulls. They are fully labeled, with the one product boxed and the
hand/phone/bag/wallet deliberately **not** boxed. Only C2b is null-marked.

## Tracking the labeling itself

Once labeling starts, `label_progress.py` answers the two questions that matter — how much is
left per class and distance, and whether anything is labeled with the wrong class:

```bash
sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py
```

It prints a class × distance grid of `done/total`, the per-split breakdown, the largest
still-empty cells first (so a labeling session starts where it counts), and any image whose drawn
class does not match its class tag. It also lists the null annotations, which is the C2b check
above. A total percentage on its own hides the thing that actually stalls a project: one distance
bucket nobody has opened.

It also writes a machine-readable snapshot beside the report (`label_progress.json`). That file is
what the desktop app's Admin Panel reads, under **Dataset labeling** — the same list as an ordered
worklist (biggest cell first, finished cells dropped as they are worked through, the hard negatives
flagged *mark each one **null** (N) — do not draw*), a Refresh button, and a line saying how many
minutes old the reading is. The order is the tool's, not the panel's: there is one place deciding
what to label next, and it is the same program that counts. The app never calls Roboflow
for this: the tool does the talking and the app reads the file, so no dataset API key or network
dependency enters the runtime, and the panel works offline.

## What "done" looks like

| Check | Command | Expect |
|-------|---------|--------|
| Project is ready to receive a session | `clean_v2.py sanity` | no blocking problems |
| Class names pinned to v1 | `label_classes.py` | mapping matches the live v1 project |
| Split is measurable | `plan_split.py` → SPLIT_PLAN.md | 0 unmeasurable class-slots |
| No unmeasurable cell | `audit_v2.py` → AUDIT.md | every cell ≥ 40 |
| Mid axis rescued | same | mid ≈ 500, ~27% of the set |
| Multi-item exists | same | non-zero, with same-class repeats on **product** classes rather than clutter |
| Negatives exist | same | `person` / phone / bag present in labeled frames |
| Splits stay honest | Roboflow version page | split by batch, every class and distance in each split |
| Acceptance number is an unseen session | `plan_split.py --holdout-session s3` | Plan C written, no `CANNOT hold out` — `s3` in `test`, every cell still in `train` |
| The version's geometry is the reviewed one | `generate_version.py --verify <n>` | `matches generate_version.py`, and 640×640 `Stretch to` |
