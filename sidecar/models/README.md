# sidecar/models/

Custom grocery weights live here. The directory is **gitignored** (see `../.gitignore`) —
weights are build outputs, megabytes each, and a fresh clone should not pay for them. Only
this README is tracked, so the convention travels with the code.

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
records the requirement *and* the version and measurement that came with it. JSON, not a
`models/`-wide manifest: the weight is the thing that gets copied to another machine, and a
manifest left behind would lose the requirement at exactly the moment it is needed.

## Naming rule (MODEL_TRAINING.md §8.2)

```
scanncart-grocery.onnx       the unsuffixed v1 baseline — never export over it
scanncart-grocery-v1.pt      v1's seven classes, trained locally (`--generation v1`)
scanncart-grocery-v2.pt      v2, trained locally    <- what train_model.py installs
scanncart-grocery-v3.pt      the next generation
```

A `-v1.pt` used to be *expected* to be flagged in the Admin Panel's *Weights on disk* list, back
when v2 declared eight classes and v1's seven could never predict Palmolive. With Palmolive
dropped the two generations declare the same seven names, so that finding is now silent and the
silence is the correct reading — a clean listing is what "these weights can predict everything the
roster names" looks like. Only a weight whose recorded class list really is missing a name, or
carries a distance in one, is flagged.

The `-vN` suffix is the **model generation**, not the Roboflow version number. They
deliberately disagree: v2's weights come from `snc-grocery` **version 2**, while the model
filename says `v2` because that is what the picker, the docs and the training runs call it.
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

No weights are tracked in this repository. v2's are produced by `../tools/train_model.py` from the
merged set `../tools/build_dataset.py` writes (v1's export plus the locally labeled v2 frames), and
`../tools/dataset_doctor.py` checks that set before the GPU runs. See `docs/MODEL_TRAINING.md`
§6–§8 and `docs/RUN_SHEET.md` for the run and the integration steps.

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
