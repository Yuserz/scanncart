# Detection scope: the zone gate, and what to do with v2

Status: plan, not implemented. Written against the working tree on `v1-local-build`, 2026-09-26.
Read alongside [MODEL_TRAINING.md](./MODEL_TRAINING.md) §8, [RUN_SHEET.md](./RUN_SHEET.md) and
[CAPTURE_CHECKLIST.md](./CAPTURE_CHECKLIST.md).

## 1. What was asked

> The model should detect an item accurately at any size and angle, but still have **proximity
> priority** — only detect the item when it is within the stated scope. Can we set a detection
> radius in the app, like Tapo/TP-Link, where we shade an area in the camera's line of sight and it
> only detects inside that area?

Yes. But the request contains two different things wearing one word, and separating them is the
whole answer:

- **Spatial scope** — *where in the frame* an item has to be. This is the Tapo shaded area. It is a
  property of the counter, not of the model, and it belongs in the app.
- **Depth / proximity** — *how close* an item is. This is a property of the weights, and it is what
  "any size" is asking for. It belongs in the dataset.

Building the first one at the app level and the second one at the data level is the plan. Building
either of them as *classes* is the trap this repo already guards against, and §3 says why.

## 2. Audit — where the project actually is

### 2.1 The shipping weight (v1)

`sidecar/models/scanncart-grocery-v1.pt` + its record. Seven classes, `resize_mode: stretch`,
generation `v1`, measured on the `test` split:

| metric | value |
|---|---|
| precision | 0.982 |
| recall | 0.943 |
| mAP50 | 0.965 |
| mAP50-95 | 0.935 |

Per class, against the 0.85 floor:

| class | recall | instances |
|---|---:|---:|
| 555 sardines 155grams | **0.732** | 97 |
| Bear Brand Fortified Powdered Milk 33g | 1.000 | 50 |
| Milo Chocolate Drink 22g Sachet | 1.000 | **6** |
| century_tuna_flakes_in_oil_155_grams | 1.000 | 50 |
| lucky_me_pancit_canton_calamansi_flavor | 0.912 | 74 |
| safeguard_pure_white_60g | 1.000 | 50 |
| silver_swan_sukang_puti_200ML | 0.959 | 74 |

Three things to read out of that:

1. **One class is below the floor** — 555 sardines at 0.732. That is a v1 defect with a name.
2. **`per_distance` is empty**, and always will be: v1 has no distance axis
   (`generations.V1.manifest = None`). So nothing in v1's record can say whether the failures are
   at `close`, `mid` or `far` — which is precisely the question the user is asking. The number that
   would answer "does it work in my checkout lane" does not exist for v1 and cannot be recovered
   from it.
3. **Milo's 1.000 is over 6 instances.** A perfect score on six frames is not evidence.

Training config from `runs/scanncart-grocery-v1/args.yaml`, the part that bears on "any size and
angle":

| param | value | reading |
|---|---|---|
| `imgsz` | 640 | fine |
| `scale` | 0.5 | ultralytics default; present |
| `degrees` | **0.0** | **no rotation augmentation at all** |
| `perspective` | 0.0 | no |
| `mosaic` | 1.0 / `close_mosaic` 10 | standard |
| `fliplr` | 0.5 | horizontal only; note the app mirrors the *preview* only, so this is consistent |

So "any angle" was never trained for. The model saw upright products in v1's 1,815 frames and was
never asked to hold a prediction through a rotation. A product held at 45° in the lane is outside
what v1 was taught, and `degrees: 0.0` is the reason.

### 2.2 The v2 model

**There is no v2 model.** `sidecar/models/` holds one weight and one record, both v1. `runs/` holds
one training run, v1. The Model picker's `models/scanncart-grocery-v2.pt` entry and
`CUSTOM_MODEL_V2` in `settingsFields.ts` are labels waiting for a file.

What v2 actually is right now is a **half-labeled dataset**, and the shape of the half is the
important part. From `cleaned-v2/label_progress.json` (generated 2026-09-23):

| | images | decided |
|---|---:|---:|
| total | 1,433 | 858 (59.9%) |

Split by distance, the labeling is **entirely `close`**:

| distance | staged | decided |
|---|---:|---:|
| close | 1,015 | 858 |
| mid | 152 | **0** |
| far | 216 | **0** |

Every `mid` and every `far` frame is staged and none is labeled. Per cell, the labeled ones are
exactly the close cells: bear-brand-milk|close 106/106, century-tuna|close 128/150,
lucky-me-pancit|close 214/234, safeguard|close 168/184, silver-swan-vinegar|close 242/242. The
unlabeled remainder is: all of milo (142), all of 555-sardines (79), all 50 hard negatives, and every
mid/far frame of every class.

`tier_a` reports **186 of a 253-image target still missing across 8 cells**, including two cells
that do not exist at all:

- `century-tuna|far` 0/40 and `century-tuna|mid` 0/40 — century-tuna has *only* close frames
- `palmolive|close` 0/40 — palmolive has only mid and far
- `silver-swan-vinegar|mid` 5/35, `lucky-me-pancit|far` 13/27, and 4 more

The honest reading: **v2's whole reason to exist is the distance axis, and the distance axis is
0% labeled.** Training today would produce a model that is v1 with one extra class and a different
split — no better at size, and no better at angle. That is the finding that decides the plan.

### 2.3 The split, and what the test number means

`split_plan_b.json` is applied (1,433 entries, name → split; train 1,019 / valid 278 / test 136).
That is **Plan B, stratified per `(class, distance)` cell**, and `label_progress.sessions` shows the
consequence directly:

```
s1   train 969   valid 278   test 136   splits: 3
```

One capture session, present in all three splits. `SPLIT_PLAN.md` says it in its own words:

> Every batch is capture session s1, so train, valid and test share one. Unavoidable with a single
> session, and the honest reading of these numbers is "held-out frames", not "an unseen session".

Plan B was the right call *for a dataset with one session* — it is the only plan that gets every
cell into every split (Plan A leaves 10 of 21 cells with no train images). But it means the test
number is measured on frames from the same shoot, same counter, same lighting, same day as
training. It will read optimistic, and it cannot answer "does this work for a customer who walks up
tomorrow". v1's 0.943 has exactly the same character.

**Tier D (`capture-s3`, the held-out acceptance re-shoot) is scaffolded and empty — 0 images.**
So there is no held-out session set, and the acceptance number the project's own RUN_SHEET wants
does not yet exist either.

Secondary blocker: `manifest.json`'s entries carry `tags: ["close", "bear-brand-milk"]` and no
`session` key, even though `clean_v2 --session` is supposed to stamp one. `plan_split
--holdout-session` reads sessions off the tags, so it **cannot be run against the staged set as it
stands** — it would see one unnamed session. Fixing the manifest (or re-running the retag with a
session) is a prerequisite for the plan in §4.3.

### 2.4 There is no zone feature

`rg` over both toolchains for `zone|roi|polygon|radius|region` returns only `audit_recall.py`'s
polygon-label handling and its tests. Nothing in `settings.py`, nothing in `Pipeline`, nothing in
the renderer's overlay — the words "polygon" here mean Roboflow's label format, not a detection
region. This feature does not exist in any form; it is net-new, not a repair.

The good news is that the seam is already there and is the right one. `Pipeline.process_once`
already filters detections in the app layer, in a documented order, right after inference:

```
infer → class_allowlist → drop_clamped_detections → _log_detections → render_preview
```

A zone gate is one more pure function in that chain, and everything that makes the clamp filter
work — a pure predicate, a count carried on `Stats`, a hot-reloadable setting, tests against
synthetic boxes — transfers directly.

## 3. The two ways to build "proximity priority"

### Way A — distance as classes (rejected)

Train v2 with 24 classes: `sardines close`, `sardines mid`, `sardines far`, … Then a `far` detection
*is* the proximity signal, and "only detect in scope" becomes "drop the `close` and `far` classes"
via `class_allowlist`.

This is the trap, and the repo already documents it in three places
(`label_classes.distance_tokens_in`, `roster.class_list_problems`, MODEL_TRAINING §8.1). What it
costs:

- **The head stops being per-product.** 24 outputs, and every box comes back under a name
  (`palmolive close`) that is not on the app's roster. `class_list_problems` fires on every capture,
  and the item log reports one product under three labels. Nothing errors anywhere — a class *name*
  records none of this.
- **The classes are mutually exclusive, so the model must decide distance before identity.** Those
  are the same pixels. It spends capacity on a task the camera geometry already answers, and every
  distance confusion becomes a product confusion — they are the same output layer.
- **It is irreversible-ish.** A version number cannot be reused; the class list has to be fixed,
  annotations moved, and the version regenerated. `generate_version --yes` now fails closed on
  exactly this list, so the pipeline would refuse to produce it.
- **It does not solve the request.** The user wants to *set* the radial scope by hand, per store,
  from the app. A distance-class head hard-codes one scope at training time and cannot be re-aimed
  without a retrain.

### Way B — distance as a data axis, scope as an app gate (recommended)

Keep **8 classes**. Distance stays what the toolchain already made it: a Roboflow *tag*, a
`(class, distance)` *cell* in the reports, and a guarantee that the training set contains the item
at three apparent sizes. The model learns one product identity that survives scale, which is what
"detects at any size" means. Proximity and scope become a geometric gate in the app, tuned live by
the operator.

Why this wins:

- **It works with v1 today.** The zone is model-independent. Every weight that will ever be
  installed gets it, including a future v3.
- **It is the thing the user actually asked for** — a shaded region, set by hand, like Tapo.
- **It is measurable.** Gate counts ride on `Stats` like `suppressed` does, so "the zone is doing
  something" is a number on screen rather than a hope.
- **It keeps the roster clean**, so `stat-classes` keeps reading `7` or `8` where an operator
  expects it, and no warning nobody can act on appears.
- **It is live-tunable.** A slider drag re-scopes the lane without stopping capture, which is the
  whole point of a scope you set by eye on a real counter.

## 4. The plan

### Phase 0 — the detection zone in the app (no new data needed)

The Tapo-style shaded area. Everything here is app-side, ships against v1, and is testable against
fakes.

**Settings** (`sidecar/app/settings.py`), all hot-reloadable:

```
zone_enabled: bool = False
zone_points: list[list[float]] = []     # normalized [x, y] in the TRUE (unmirrored) frame
zone_mode: str = "center"               # "center" | "bottom-center" | "overlap"
zone_min_overlap: float = 0.5           # only read when zone_mode == "overlap"
```

`zone_points` is one polygon in one coordinate space, whatever shape the UI drew — a rectangle and
an ellipse differ only in the handles the UI offers, so the sidecar has one rule to enforce. Empty
`zone_points` or `zone_enabled: False` means the whole frame, which is today's behaviour and keeps
the default non-destructive.

**The predicate** (`sidecar/app/pipeline.py`, next to the clamp filter):

```
def in_detection_zone(box, points, mode="center", min_overlap=0.5) -> bool
```

Pure and total, so it is tested against synthetic boxes and polygons rather than a camera — the same
shape as `is_clamped_to_frame`. Recommended default `center`, with `bottom-center` offered because
on a counter the *contact point* is the honest "is it in the lane" test: a box can be half outside
the zone while the item sits well inside it. `overlap` is there for a caller that wants to be
strict; it needs area intersection, which is why it is the third option and not the first.

**Wiring**, in `process_once`, immediately after `drop_clamped_detections` so the order is
"what classes, what shapes, what area" and the log/overlay/DB all agree:

```
counts.gated = len(detections) - len(kept)
```

**Two load-bearing details:**

1. **Gate on the true frame, draw on the mirrored one.** `preview_mirror` is on by default, and the
   boxes are reflected by `mirrored_detections`. The zone polygon must be reflected by the *same*
   rule (`x' = 1 - x`, per point) or the shaded area on screen will be on the opposite side of the
   lane from the area the model is gated by — the operator aims at the wrong half of the frame and
   the overlay looks correct in isolation. Put the reflection in `lib/overlay.ts` beside
   `boxToPercent`, one helper, and pin with a test that a rectangle mirrors to the same corners the
   sidecar's rule produces. Deliberately *not* the renderer eyeballing `preview_mirror` separately.
2. **The count is visible.** Add `gated` to `Stats` and a chip to the Live stats strip next to
   `suppressed`. A gate nobody can see is indistinguishable from a model that stopped detecting —
   and the operator setting the zone by eye needs to watch that number move while dragging.

**UI** (`LiveView` + `AdminPanel`, following the `SETTINGS_GROUPS[].home` split):

- A draggable polygon/rect handle on the live preview, with `Full frame` / `Centered rect` /
  `Ellipse` presets. These three cover what Tapo offers, and ellipse is the literal "detection
  radius" reading of the request.
- Pushed on drag end via `PATCH /api/settings?persist=false`; committed with the existing
  `POST /api/settings/save`, matching how the tuning card already treats sliders.
- Safe mid-capture, because every field here is hot-reloadable — the same property that lets the
  Live view's tuning card work without a stop/start cycle.

### Phase 1 — finish v2 labeling, `far` first

The labeling backlog already orders this: `label_progress` derives `labeling_backlog` with
**most-remaining-first**, which right now puts the untouched distance cells at the top. The order
that buys the most:

1. `century-tuna|mid` and `century-tuna|far` — 0/40 each. A class with *only* close data cannot
   even be measured at distance, so these two are the difference between "has a mid/far recall" and
   "cannot be tested there".
2. `palmolive|close` — 0/40. It is the v2-only class and it has no close frames, so its close cell
   is dark.
3. The rest of the 8-cell `tier_a` gap, then milo (142, 0 decided) and 555-sardines (79, 0 decided)
   — note 555-sardines is the class v1 fails at 0.732, so its v2 close frames are worth more than
   their count suggests.
4. The 50 hard negatives (0 decoded) — they are what `suppress_clamped_detections` expects to
   become the wrong default *because of*, per `settings.py`'s own comment.

Before any labeling: re-run `label_classes.py` and confirm the project class list has **8 names and
no distance tokens**. That is the fail-closed check that keeps v2 from becoming Way A by accident.

### Phase 2 — make the acceptance number honest

1. **Stamp the session.** Re-run `clean_v2 --session s1` (or retag) so `manifest.json` carries it —
   `plan_split --holdout-session` cannot run without it, and neither can the session × split table.
2. **Shoot Tier D** — `capture-s3`, already scaffolded by `clean_v2.scaffold` with its folders,
   README and filing commands, plus the coverage preflight that names the cells it must cover.
   This is the held-out acceptance re-shoot and it is the only way the test split becomes an unseen
   session. It is also where "any angle" gets real evidence: shoot the products **tilted, rotated and
   in-hand**, not just nearer and further, because `degrees: 0.0` means nothing else will teach it.
3. Plan with `plan_split --holdout-session s3`. It fails closed (exit 2, no plan written) if
   holding the session out would leave a cell unlearnable — the right behaviour, and the reason
   Phase 1 has to come first.

### Phase 3 — train, then measure what was asked

Rotation augmentation, modestly — this is the v1 gap that has never been addressed:

```
--degrees 12 --perspective 0.0005 --scale 0.6
```

`degrees` is the direct fix for "any angle"; `perspective` helps the counter's camera angle;
raising `scale` past the 0.5 default widens the size range on top of the real mid/far frames. Keep
`mosaic 1.0` / `close_mosaic 10` — multi-scale composition is what `any size` leans on.

Then the measurement, which is the answer to "does it meet our requirements":

```
sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v2 --val --split test  # per-class table + per-distance grid
```

The per-distance breakdown is on by default (three extra passes, one per distance, over the
manifest's tag map) and `--no-per-distance` is there to skip it. Skipping prints that the breakdown
was *skipped* rather than nothing, because a section that silently vanishes reads as "every
distance passed".

The per-distance grid is the whole point: it prints `!` for a cell under the floor and `-` for a
class the distance holds no instances of, so **`far` clearing 0.85 on the held-out session** is a
number in a file, not a judgement call. Let `--install` write the record, then set
`resize_mode: auto` in the app — `models.requirement_for` honours the recorded requirement, so
`auto` resolves to `stretch` (v2's `REQUIRED_RESIZE_MODE`) instead of guessing letterbox.

## 5. Acceptance criteria

The request, restated as things that are true or false:

| requirement | how it is proven |
|---|---|
| detects accurately | per-class recall ≥ 0.85 on `test`, with instance counts beside each |
| at any size | per-distance grid clears the floor at `close`, `mid` **and** `far` |
| at any angle | Tier D frames shot tilted/rotated; the same grid clears with them in `test` |
| only in scope | zone gate on, `Stats.gated` > 0 while an item is held outside the shaded area, and 0 detections logged from outside it |
| held-out honesty | the `test` split is capture session `s3`, which `train` never saw |

Until Phase 2 lands, the first line will be true and the fourth is achievable; lines 2, 3 and 5 are
the ones that are currently unproven and unprovable.

## 6. Cost, and what could go wrong

- **Phase 0 is small** — one pure predicate, four settings fields, one `Stats` field, one overlay
  helper, one handle UI. It is the highest value per hour in this plan and it does not depend on any
  data that does not exist yet.
- **Phase 1 is the expensive one**, and it is operator hours, not code. 186 images for Tier A plus
  ~271 undecided elsewhere, then Tier D on top.
- **The zone can be set wrong**, which is why it defaults off and the count is visible. A zone
  smaller than the lane silently drops real items, and that failure looks exactly like a model that
  has stopped working — the `gated` chip is the only thing that distinguishes them.
- **Phase 3 raises risk before it lowers it.** `degrees 12` on a dataset of 1,000-ish training
  images costs some mAP; if the held-out session says the tilt evidence is thin, tilt the *camera
  captures* rather than the augmentation.
- **Do not let Phase 1's urgency reopen Way A.** The class list is the one irreversible decision in
  the chain, and `label_classes.py` / `generate_version.py` are the two gates that already fail
  closed on it.
