# sidecar/models/

Custom grocery weights live here. The installed weights (`*.pt`, `*.onnx`) and the `*.json`
records beside them are **committed** (see `../.gitignore`), so a `git pull` brings a new model
to every checkout. After `train_model.py --install` puts a new weight here, commit the `.pt` and
its `.json` together. Keep each file under GitHub's 100 MB limit (the YOLO11s weights are about
19 MB).

## What goes here

A `.pt` (or `.onnx`) file **directly** in this directory is a selectable model:

- `settings_store.is_custom_model()` accepts any `models/<name>.pt|.onnx`, so no whitelist
  edit is needed in the sidecar or in the desktop mirror.
- `app/models.py` lists what is on disk, and the Admin Panel's Model picker offers it. The
  picker is keyed by **filename**, which is why the naming rule below is the thing that
  carries meaning.
- `active_model` stores the value as `models/<name>` — exactly what the picker shows.

## The record beside the weights

`train_model.py --install` writes `<name>.json` next to the `.pt`. It carries the one thing
nothing else knows: the **`resize_mode` these weights have to be run with**.

```json
{
  "generation": "v2",
  "resize_mode": "stretch",
  "source": "snc-grocery version 2",
  "installed_at": "2026-09-22T09:13:02",
  "class_names": ["Bear Brand …", "lucky_me_…", "555 sardines 155grams", "…"],
  "imgsz": 640,
  "augmentation": {
    "fliplr": 0.5,
    "flipud": 0.0,
    "degrees": 15.0,
    "scale": 0.5,
    "translate": 0.1,
    "hsv_v": 0.2,
    "mosaic": 0.0,
    "erasing": 0.0
  },
  "validation": [
    {
      "split": "test",
      "floor": 0.85,
      "measured_at": "2026-09-22T10:40:00",
      "aggregates": { "precision": 0.9, "recall": 0.82, "mAP50": 0.88, "mAP50-95": 0.61 },
      "per_class": [
        { "name": "bear-brand", "recall": 0.9, "instances": 10 },
        { "name": "lucky-me", "recall": null, "instances": 0 }
      ]
    }
  ]
}
```

One field says where the weights came from rather than what they need:

- **`source`** — which dataset produced them, and the one thing the Admin Panel prints in parentheses
  after a weight's name. It is the Roboflow version when `--version` named one, and otherwise the
  *run's* own answer: `--install` reads the run it is installing (`args.yaml`'s `data=`, or the
  directory it trained on itself when `--yes` created that run) and, when that
  names a directory carrying `merge_report.json`, records **`local merge <directory>`** — a weight
  trained from a **local merge** has no version number to name, and `build_dataset.py`'s output is
  named by where it sits. An install that can answer neither way writes the field empty and says what
  it is empty of. The three v2 weights were installed before the derived half existed and carry those
  same strings today, filled in by `sidecar/tools/backfill_provenance.py`, which does the identical
  reading from the other end — the weight's bytes find the run, and the run names the set — giving
  `local merge merged-v2`, `local merge merged-v2-letterbox` and `local merge merged-v2-letterbox-synth`:

  ```bash
  sidecar/.venv/Scripts/python.exe sidecar/tools/backfill_provenance.py --dry-run
  ```

  It writes `source` and nothing else, merges rather than replacing (the class list beside it is the
  one field nothing can recover), and reports rather than guesses in the four states that have no
  answer — no run holds the weight's bytes, two do, the dataset directory is gone, or the dataset is
  a version's export whose number is not in the run. A weight with **no** record is deliberately left
  alone: writing one would flip the `recorded` flag, and that flag is what the app's `auto` gives
  `letterbox` entry is gated on.

Four more fields travel with it, and each one is a fact nothing else keeps:

- **`class_names`** — the label set the model predicts, in the order it indexes them. A `.pt`
  keeps no class list and the export that carried it is gone by the next session, so a weight
  trained from a project whose classes were split by distance (`safeguard close` / `mid` / `far`
  instead of one `safeguard`) would otherwise load without complaint and log one product under
  three labels. Written down, the Admin Panel's *Weights on disk* list flags it **before the
  weight is selected**, against the roster of its own generation.
- **`imgsz`** — the size the run trained at. `Settings.imgsz` decides what the app resizes every
  frame to before detection, and nothing else relates that setting to the weights: a model
  trained at 960 runs at 640 by default, and the `far` detections come back weaker with no
  explanation anywhere. `resize_mode` was the first half of "the app feeds the model the wrong
  geometry"; this is the second.
- **`augmentation`** — the table that trained, read back out of the run's own `args.yaml`
  rather than echoed from the flags, because `--install` is a separate command from `--yes`.
  It is evidence rather than an input (nothing in the app branches on it) and it exists because
  augmentation leaves no trace in `results.csv`.
- **`generation`** — which dataset trained these weights, and therefore which roster's class
  names they are judged against. A record naming a generation this app has no roster for is read
  as *not recorded*, so a record written by a later tool is judged by the names it carries.

All four are **omitted rather than written empty** when the run did not know them, so a re-install
of the same weights writes a byte-identical file. `POST /api/models/record` (the *Record it now*
button) merges into an existing record rather than replacing it, so adding one field cannot delete
what a training run measured.

The *Weights on disk* listing reads those fields as **one completeness view** rather than as a note
per warning: every weight on disk gets the record's four facts — geometry, generation, validation and
provenance — each either stated or shown as a gap **with the command that fills it**, under a count
(`2 of 4 recorded`) so a weight nothing was written about is legible at a glance. The facts and the
writers are the sidecar's (`app/models.record_facts`), because the writers are the tools in this
directory plus the panel's own button — and *which* of those applies is a fact about the file rather
than a rendering choice: geometry's writer is `train_model.py --install` for a weight the tool
installed and the button for one copied in by hand, and the provenance's is
`backfill_provenance.py` when this machine's source index can already name the dataset and
`--install` when nothing can. A gap and a guess are also kept apart, because only one of them is a
fact: the geometry row says what `auto` will use instead.

`validation` is what `--val` measured, carried in by the following `--install`, and the Admin
Panel shows it beside the requirement. Three things about its shape are deliberate:

- **`split` is in the block.** `test` is the acceptance split and `valid` is the one training
  selected on, so the same 0.95 is a different claim depending on which produced it. Running
  `--val --split valid` *adds* a block rather than replacing the `test` one, and `test` sorts
  first so the acceptance number is the one read first.
- **`recall: null` means the split held no instances of that class**, which is not a score of
  zero. Ultralytics answers 0.0 there, and a zero would send you after images of an item when
  what is missing is captures in that split.
- **No `below_floor` list.** It is `recall < floor`, derived where it is rendered; a stored
  copy could disagree with the numbers beside it and there would be no way to tell which was
  right.

`--val` records this in the *run* directory (`val_metrics.json`), not here, and `--install`
copies it in — the two are separate commands, so the numbers travel through a file. Each
measurement carries a hash of the checkpoint it describes, and `--install` only attaches the
ones matching the file it is installing: re-training into the same run directory replaces
`best.pt` in place, and a score belonging to the previous checkpoint must not be presented as
this one's. Installing without a measurement is fine and says so — the panel has nothing to
show for that weight until one is taken.

The requirement cannot be recovered from the weight or from its name: a `.pt` records the
training run, not the dataset geometry, and the filename is a convention. Nor can it be inferred
from the suffix — a `.pt` trained on a `Stretch to` version needs *stretch*, while
`resolve_resize_mode()`'s format heuristic answers *letterbox* for any local `.pt`.

So it travels with the file, and **`resize_mode: auto` honours it.** `app/models.py`'s
`requirement_for()` reads the record and `resolve_resize_mode()` prefers it to the heuristic, so
the weights are run at the geometry they were trained at whether or not anyone set the field by
hand. The Admin Panel's Model field reads the same record: the installed weights are listed with
the mode each needs, and the field warns only when an *explicit* value contradicts it — `auto`
is now the correct setting rather than a quiet trap.

**No record means no requirement.** `auto` then falls back to the format heuristic, which is a
guess, so the panel names the guess (`auto gives letterbox`) instead of assuming `stretch` — and
the sidecar reports that case as a **structured entry** (`SettingsResponse.unrecorded_resize_mode`,
not a line in `warnings`) because the panel's *mismatch* check cannot: it reports a setting
contradicting the record, and here there is no record to contradict. The guess is the one resize
case nothing could flag as wrong, so it is reported as what it is.

The entry carries the mode `auto` resolved to, the sentence explaining the risk, and a second
sentence saying how to stop assuming — and the Admin Panel renders a **`Record it now` button**
beside them. The prose is split in two because **two views** render this entry and only one of them
can perform the fix: the Admin Panel has the button, and the **Live view surfaces the same
assumption while a capture runs**, which is where the weak `far` detections it costs are actually
being watched. A view with no button cannot be told to press one, so the remedy names `--install`
and the Admin Panel instead of a control that may not be on screen — and **both views now carry the
button**, since writing the mode `auto` already resolved to cannot disturb a running detector. The
Live view's copy sits inside its running-gated banner (the same write, one implementation in
`useActiveWeights`), and the stats strip carries the requirement ungated in either state:
`requirement (recorded)` when a record names it, `requirement (assumed)` when nothing does. The button writes the requirement into
`models/<stem>.json` through `POST /api/models/record` (`app/models.record_requirement`). That is
for the weights `--install` never installed: a copy from another machine, a stock checkpoint, a
checkpoint whose training run was never recorded — the operator is the only source of the fact,
and the app is a shorter path to it than hand-writing JSON. The write **merges** rather than
replaces, so a record's other fields (which dataset version, what `--val` measured) survive being
corrected.

What the button records is the mode `auto` had already resolved to, so clicking it cannot change
what the detector does — it converts a fallback into a stated fact. And it is an *attestation*:
unlike `--install`, nothing measured anything. Recording `stretch` for a stretched `.pt` is the
same write with a different mode, which is also how a `.pt` trained on a `Stretch to` version
gets a record when the run that produced it is long gone:

```bash
curl -X POST http://127.0.0.1:<port>/api/models/record \
  -H 'Content-Type: application/json' \
  -d '{"model": "models/scanncart-grocery-v2.pt", "resize_mode": "stretch"}'
```

`--install` remains the writer to prefer when the training run is still around, because it
records the requirement *and* the version and measurement that came with it — and the button
agrees rather than implying otherwise: what it writes carries no `source`, so its confirmation
says that field is still empty and names the same two writers that fill it (`--install --version N`
for a generated version, `backfill_provenance.py` for a local merge). That second writer keeps working
after the run itself is gone, because a run's answer is written down while the run still exists: the
machine-local **source index** (`app/source_index.py`, `sidecar/data/source_index.json`) maps a
weight's bytes to the dataset it came from, so `backfill_provenance.py` reads it instead of matching
a run, and `--install` falls back to it when neither `--version` nor the run can answer. The panel
shows that entry beside a weight whose record is still silent - as the index's answer, not as a
`source`, since the record is what travels with the weights. JSON, not a
`models/`-wide manifest: the weight is the thing that gets copied to another machine, and a
manifest left behind would lose the requirement at exactly the moment it is needed.

## Naming rule (MODEL_TRAINING.md §8.2)

```
scanncart-grocery.onnx       the unsuffixed v1 baseline — never export over it
scanncart-grocery-v1.pt      v1's seven classes, trained locally (`--generation v1`)
scanncart-grocery-v2.pt      v2, trained locally    <- what train_model.py installs
scanncart-grocery-v2.1.pt    v2's seven retrained (`--variant v2.1`) — the same classes and
                             geometry, a later version of the same project
scanncart-grocery-v3.pt      the next generation
```

`-vN` is the **generation** — the class list, its order and the geometry — and `.<M>` is a second
run of one (`MODEL_TRAINING.md` §8.2). A variant is named by `train_model.py --variant v2.1`, which
moves the export directory, the run, its `--val` run and this filename together so a retrain cannot
overwrite the run the weight it retrains came from. Its **record keeps `generation: v2`** — the
seven names are v2's, and a `v2.1` in `class_names`' vocabulary is not a roster this app has — so
the filename and the recorded `source` (`snc-grocery version 3`) are what tell the two weights
apart, and `--version` is what makes a retrain distinguishable at all. What the record then shows
is the whole answer to "which classes is this weight on, and how good is it": `class_names` plus
`--val`'s per-class recall, its per-distance grid, `imgsz` and the augmentation table.

A `-v1.pt` used to be *expected* to be flagged in the Admin Panel's *Weights on disk* list, back
when v2 declared eight classes and v1's seven could never predict Palmolive. With Palmolive
dropped the two generations declare the same seven names, so that finding is now silent and the
silence is the correct reading — a clean listing is what "these weights can predict everything the
roster names" looks like. Only a weight whose recorded class list really is missing a name, or
carries a distance in one, is flagged.

The `-vN` suffix is the **model generation**, not the Roboflow version number, and they are
different facts about different artifacts: `v2` names the dataset *convention* (the class list, its
order, the geometry), while the Roboflow version is one frozen snapshot of the project. On this
machine they have never coincided — v2's weights were trained from the local merge
(`build_dataset.py`'s `merged-v2`), not from a generated version, which is why `--version` had
nothing to write for them and their records carried an empty `source` until
`backfill_provenance.py` named the merge each one came from. And `snc-grocery` has never generated
one at all (its only version is the empty `1` in the Trash, which is why the next generation takes
number 2). That is what `--version` is for: a retrain
installed with it records which snapshot it came from, and the filename plus that field are what
tell two weights of one generation apart.
Keeping the baseline alongside the new generation is the point — it is what lets you A/B
them in Live View and roll back from the dropdown.

`train_model.py --install` refuses to overwrite an existing weight. `--force` replaces one
deliberately; a different generation should get a different filename instead.

## The setting that goes with a locally trained model

**`resize_mode: stretch`** for a `.pt` trained from a `Stretch to`-generated version — and a
weight installed with its record does not need to be told that: `auto` uses the recorded
requirement, so the default is the geometry it was trained at.

Setting `stretch` by hand is equivalent and still works; setting *anything else* is an override,
and the Model field says so. What `auto` cannot do is guess for a weight **with no record**:there it falls back to the heuristic, which answers *letterbox* for a local `.pt` and so presents every
object at 0.56× the canvas it was trained at — no error, just weaker detections, worst on the
`far` cells. Nothing is silent about it any more, though nothing enforces it either: the sidecar
reports the assumption as an entry with a remedy (native + `auto` + a custom `.pt` with no
record), while an explicit `letterbox` is a decision and is left alone. Either record one — the
`Record it now` button (in either view — the Live view's copy sits in its banner while a capture
runs), or `--install` — or set `stretch` deliberately; setting
`letterbox` by hand answers the *warning* without answering the record, which is the difference
between silencing a report and supplying the fact.

`active_model` and `resize_mode` are both **restart-required** fields, so capture must be
stopped before saving them.

The installed weights are tracked in this repository (see the top of this file). v2's are produced by `../tools/train_model.py` from the
merged set `../tools/build_dataset.py` writes (v1's export plus the locally labeled v2 frames), and
`../tools/dataset_doctor.py` checks that set before the GPU runs. See `docs/MODEL_TRAINING.md`
§6–§8 and `docs/RUN_SHEET.md` for the run and the integration steps.

## Publishing weights for other machines

The weights are **committed** in this folder, so `git pull` is the main way they reach another
machine: after `train_model.py --install`, commit the `.pt` and its `.json` together. They are also
published as the assets of **one GitHub Release, tagged `models`**, which fills in anything a
checkout lacks: `setup.cmd` / `scripts/setup.ps1`, `scripts/setup.sh`,
`make install` and `make models` all run `scripts/fetch_models.py`, which downloads each `.pt`,
`.onnx` and `.json` attached to that release into this folder — skipping any file already here, so
a machine's own training run is never overwritten. The app's default is
`scanncart-grocery-v2.1.pt`, so that one and its record are the two files a release must
carry; attach the other weights too if other machines should be able to A/B them.

To publish, from the machine that trained them (once, then again after each retrain):

1. On GitHub: **Releases → Draft a new release**, tag `models` (create it on `main`), title
   e.g. *Model weights*.
2. Attach the weight **and its `.json` record** from this folder — the record is what tells
   `resize_mode: auto` the geometry the weight was trained at, so a weight without it runs at
   the wrong one: `scanncart-grocery-v2.1.pt` + `.json` (the default), and optionally
   `scanncart-grocery-v1`, `-v2-stretch`, `-v2-letterbox`, `-v2-letterbox-synth` with theirs.
3. Publish. A machine set up before the release existed gets the files with `make models` (or
   by re-running the setup).

With the GitHub CLI the same is one command from the repo root:

```bash
gh release create models --title "Model weights" --notes "Default: scanncart-grocery-v2.1" sidecar/models/scanncart-grocery-v2.1.pt sidecar/models/scanncart-grocery-v2.1.json
```

and a retrain replaces an asset in place with
`gh release upload models <file> --clobber`. Because the fetch never replaces a file already on a
machine, a retrained weight reaches existing checkouts only under a **new name** (a `--variant`, as
the naming rule above says) — a same-named file has to be deleted there first, which is a decision
the person at that machine makes rather than a side effect of setup.

## What does **not** belong here

Ultralytics' stock weights (`yolo11n/s/m/x.pt`, `yolo26n/m.pt`). A bare `active_model` name —
which is what `app/presets.py` writes — is resolved by ultralytics *by name*, downloading the
file into the process cwd when it is not there. The sidecar's cwd is `sidecar/`, so that is where
they land, and `../.gitignore` ignores them there.

Moving them in would be worse than untidy. The picker reads this directory, so every stock weight
would be listed **twice** — once as the built-in `yolo11n.pt` and once as the custom
`models/yolo11n.pt` — and the built-in entry, which is what `low_end`/`mid_range` presets select,
would resolve to a *download* instead of the local file. That is the one outcome the offline
promise cannot afford, and `app/models.py`'s "list what is on disk" would be the thing causing it.
These are runtime artifacts either way, not models this project trained.
