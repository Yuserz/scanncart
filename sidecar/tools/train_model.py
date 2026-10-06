#!/usr/bin/env python
"""Train one generation's weights locally, then install them where the picker finds them.

Four one-shot decisions meet here, and each one has already bitten this project:

1. **Which generation this run is for.** `--generation` picks the dataset, the class list the
   export has to declare, the resize requirement, and the names of everything written - all
   from `generations.py`, because v1 and v2 differ in exactly the ways that matter and none of
   them is a detail: the same seven names declared by two different datasets, no distance axis
   against three, a frozen geometry requirement against one `generate_version.py` owns. Editing
   this file to train the other one is what that spec exists to stop.
2. **The export's class list.** A trained model's output indices are whatever order the dataset
   declared. If the export's `names` disagree with what its generation expects, every box comes
   back under the wrong label - with no error, and with plausible-looking confidences. So the
   export is checked *before* anything is trained, and the check re-measures how the frames were
   resized too, since that is what the record will claim about them.
3. **The training run.** Hyperparameters are `MODEL_TRAINING.md` 6's, in code, so the run that
   produced a shipped model is reproducible from this file rather than from memory - and the
   augmentation is 4's table, written out for the same reason: ultralytics' defaults are *not*
   that table (`degrees` 0.0, `mosaic` 1.0), so inheriting them would train a model the doc does
   not describe. `--degrees`/`--scale` override the two that matter; the whole table is printed
   before the run and recorded beside the weights.
4. **The drop-in.** `models/scanncart-grocery-<generation>.pt` is the name the picker lists and
   the validator accepts. The previous generation's weights were never installed at all - there
   is no tracked `sidecar/models/` in this checkout - so the step that turns a good `best.pt`
   into a selectable model is the one worth scripting. `install()` also writes a small record
   beside it (`models/scanncart-grocery-<generation>.json`) carrying the `resize_mode` these
   weights require, because nothing else knows it: the checkpoint stores the training run, not
   the dataset geometry, and `auto` resolves to the *wrong* geometry for a locally trained
   `.pt`. The Admin Panel's Model field reads that record and flags a mismatch.

`--val` is the fourth step, and it is here because the two numbers a training log carries are
**means**. A run can clear both while failing one whole class - which is the failure this dataset
exists to fix, since the mean over 8 classes is what hides `century-tuna` at `far` between a
comfortable `milo` at `close` and a comfortable `sardines` at `mid`. The pass reports recall per
class against 6's 0.85 floor, so the verdict names the class instead of the average, and it reads
the `test` split by default - the one `plan_split.py --holdout-session` makes a capture session
the run never saw. It also **writes those numbers down** (`val_metrics.json`, in the run) so
`--install` can carry them into the weights' record, and the Admin Panel shows what the model
scored beside what it needs - read from the weights rather than remembered from a terminal.

**The machine stays usable.** This box is shared with the Electron app, a browser and the
sidecar, and an earlier dataset pass was the reason `resources.py` exists. So CPU and RAM stay
inside `--max-use-percent` (20% by default), the dataloader process count is derived from that
budget instead of ultralytics' default of 8, `torch`'s thread count is clamped before the run, and
the batch size comes from the VRAM share rather than from the constant below - a batch that OOMs on
a shared card takes the other applications on it down with the run. All of it is printed as a
`budget` block before training starts, so what the run may take is readable rather than assumed.
`--max-use-percent 60` when the box is otherwise idle.

Deliberately **not** part of this: generating the version (that is `generate_version.py`, and it
must happen first) and choosing `resize_mode`. The latter is a settings field, and for these
weights it must be `stretch` - see `print_reminders()`.

The one string here that could not be checked offline is `EXPORT_FORMAT` (the API's identifier
for the version page's "YOLOv11 PyTorch"): there is no generated version to ask. A wrong value
comes back as an error body rather than a bad download, and `--format` retries it without an edit.

    # v2, the generation the app is being built towards: fetch the version's export (needs
    # --version, the number generate_version.py reported)
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --download --version 2

    # check the export, print the exact training command, change nothing
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py

    # train (writes to the dataset workspace, not into the repo)
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --yes

    # install the best checkpoint from that run as models/scanncart-grocery-v2.pt
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --install

    # the acceptance number: per-class recall on the test split against 6's 0.85 floor,
    # then the same recall split by distance (three extra passes - `--no-per-distance` skips it)
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --val

v1 is the same four commands with `--generation v1`, and two differences follow from the spec
rather than from a flag: its seven-class export is judged against *its* class list (the two lists
are the same seven names today, but the per-generation check is what keeps them trainable when a
later generation adds one), and `--val` has no per-distance breakdown to run, because v1 predates
the distance tags and its export carries none.
Its dataset directory is the hand-downloaded export ingested into the workspace:

    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1 --yes
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1 --val
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_model.py --generation v1 --install

Any generation can be pointed at another dataset and another class list - which is `--dataset-dir`
and `--classes` - because the spec describes the two sets this project has rather than the two it
will ever have (`--classes` replaces the expected list, so an export that gained a class after its
version was generated is trainable without editing the spec).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

import generations
import resources  # imported first: it sets OMP/MKL thread limits at import time

# Two different things are called `WORKSPACE` in these tools, and importing both unaliased
# binds the wrong one: `label_classes.WORKSPACE` is the Roboflow account namespace
# (`yusri-caloyloy`), `workspace.WORKSPACE` is the directory the dataset tools read and write.
# Aliased rather than relied on by import order - the collision put a Windows path into the
# export URL the first time this ran live.
from generations import DEFAULT as DEFAULT_GENERATION
from generations import Generation
from label_classes import DISTANCE_ORDER, SPLIT_NAMES, distance_tokens_in, load_key
from label_classes import WORKSPACE as ROBOFLOW_WORKSPACE
from workspace import DATA_YAML_NAME, MERGE_REPORT_NAME, SCANNCART_DATA_YAML_NAME
from workspace import SIDECAR_ROOT
from workspace import WORKSPACE as DATASET_ROOT

# ---------------------------------------------------------------------------
# The run's hyperparameters (MODEL_TRAINING.md 6). Which *generation* they are for
# is not here: that is `generations.py`, because those names are read by the Admin
# Panel's picker, `settings_store.is_custom_model` and 8.2, and all three have to
# agree. Nothing in this file may assume it knows which generation it is running.
# ---------------------------------------------------------------------------
BASE_MODEL = "yolo11s.pt"

# MODEL_TRAINING.md 6. `s` rather than `n` because with 8 classes and a few thousand
# images the larger backbone costs roughly the same wall clock on a 4060 and is
# distinctly better on small and occluded items - which is what the `far` cells are.
EPOCHS = 100
IMGSZ = 640
# The *ceiling*, not the value: `Budget.batch_size()` decides what actually fits the VRAM share
# (`--max-vram-percent`), because a batch that OOMs on a shared card takes every other
# application on it down with the run.
BATCH = 16
PATIENCE = 25

# ---------------------------------------------------------------------------
# Augmentation (MODEL_TRAINING.md 4's table). Written out rather than left to
# ultralytics' defaults, because the defaults are not that table: `degrees` is
# 0.0 (so no rotation at all) and `mosaic` is 1.0 (the "heavy mosaic" 4 says to
# leave off). A run that silently disagrees with the doc is the kind of thing
# this repo writes the value down for.
# ---------------------------------------------------------------------------
# Rotation, +-degrees. The one augmentation whose *absence* was measurable here: with no
# rotation the model only ever sees a sachet at the angle it was shot at, and a hand-held
# item at the counter arrives at any of them.
DEGREES = 15.0
# Scale jitter, +-fraction. Ultralytics' own default, named rather than inherited: it is the
# augmentation that stands in for the distance axis on a set whose `far` cells are thin, and a
# recorded value of 0.5 is a fact about the run rather than a coincidence.
SCALE = 0.5
# The rest of 4's table, in ultralytics' names. `flipud` and `mosaic`/`erasing` are the three
# exclusions, kept as explicit zeros so the record says they were decided rather than forgotten:
# the camera is fixed and items do not appear upside down, so a vertical flip or a mosaic of four
# frames is augmentation the model can never be asked to undo at inference time.
AUGMENTATION: dict[str, float] = {
    "fliplr": 0.5,      # horizontal flip: on
    "flipud": 0.0,      # vertical flip: off - items do not appear upside down
    "translate": 0.1,   # +-10% translation, ultralytics' default
    "hsv_v": 0.2,       # brightness/exposure: +-20%
    "mosaic": 0.0,      # heavy mosaic: off - the camera is fixed, it only adds noise
    "erasing": 0.0,     # cutout: off, same argument
}
# The keys of `AUGMENTATION` plus the two that have their own flags, i.e. everything the record
# carries. One list, so the record and the values it is compared against cannot disagree.
AUGMENTATION_KEYS: tuple[str, ...] = ("fliplr", "flipud", "degrees", "scale", "translate", "hsv_v", "mosaic", "erasing")

# MODEL_TRAINING.md 6's acceptance table, minus the per-class rule. Only the two
# aggregate numbers can be read out of a training run's own log; per-class recall
# needs a validation pass, which is deliberately left to `--val` so this tool cannot
# report a pass on an average that hides a failing class.
TARGETS: dict[str, float] = {
    "metrics/mAP50(B)": 0.90,
    "metrics/mAP50-95(B)": 0.65,
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# MODEL_TRAINING.md 6's per-class floor. Every class, not the average: the thing this
# dataset exists to move is the worst cell (`century-tuna` at `far`), and a mean over 8
# classes is exactly what hides it.
RECALL_FLOOR = 0.85

# Which split `--val` reads by default. `test` is the acceptance split - and with Plan C
# (plan_split.py --holdout-session) it is a capture session the run never saw, which is what
# makes the number worth quoting. `val` is what training selected on, so reading it back
# would quote the selection number as if it were an acceptance one.
DEFAULT_SPLIT = "test"
# The yaml *key* for the validation split is `val`, not `valid` (see write_data_yaml);
# ultralytics resolves `split=` against these keys, so these are the accepted values.
VALIDATION_SPLITS = ("train", "val", "test")

API = "https://api.roboflow.com"
# The version page's "YOLOv11 PyTorch" export, as the API's format identifier. It is the one
# string in this file that could not be verified offline - there is no generated version to
# ask yet - so a wrong value is treated as recoverable: the API answers with an error body
# rather than a bad download, and `--format` retries it without editing this file.
EXPORT_FORMAT = "yolov11"

# Where the runs land: the dataset workspace (gitignored), because they are checkpoints and plot
# summaries rather than source. The *export* directory is per generation (`generations.py`) - it
# is the one path here that differs between them, since v2's is fetched and v1's was ingested.
DEFAULT_RUN_PROJECT = DATASET_ROOT / "runs"
DEFAULT_MODELS_DIR = SIDECAR_ROOT / "models"
# What --val's run is called: the generation's run name plus this, so a val run sits in the same
# `--run-project` as the training run it measured without ever being mistaken for one.
VAL_NAME_SUFFIX = "-val"
# Where --val writes the numbers themselves, in the run directory it measured - see
# `val_metrics_path()` for why they live there rather than in the repo or in `models/`.
VAL_METRICS_NAME = "val_metrics.json"

# The three distances the set is staged in, in the order they are reported
# (`label_classes.DISTANCE_ORDER`, MODEL_TRAINING.md 8.3). **Not a class axis**: the class list is
# the same names at every distance, and a box drawn on a distant sachet is the same class as one
# drawn on a near one. This axis exists because the per-class floor is computed over every distance
# mixed together, so a class whose test frames happen to be mostly `close` can clear 0.85 while
# being unable to find the item at `far` - the bucket this dataset exists to fix. Splitting the
# number by distance is what turns that from an argument into a measurement.
# Where the per-distance file lists and their yamls go: inside the run directory, beside the
# numbers they produced, so a surprising row can be traced back to the exact image set it came
# from instead of being a line in a log.
DISTANCE_DIR_NAME = "val-by-distance"


def val_name(generation: Generation) -> str:
    """What `--val`'s run directory is called for this generation - see VAL_NAME_SUFFIX.

    A function rather than a constant because it is per generation, and because every call site
    that spelled it out is one more place the run name could be typed differently from the
    generation's own.
    """
    return f"{generation.run_name}{VAL_NAME_SUFFIX}"


def ultralytics_yolo():
    """`ultralytics.YOLO`, imported on use so importing this file costs no torch.

    Same shape as `app/hardware.py`'s lazy `torch.cuda` import and `app/models.py`'s "one
    directory read, no torch": a tool that answers `--export-dir` questions should not drag
    a GPU stack in behind it.
    """
    from ultralytics import YOLO

    return YOLO


# ---------------------------------------------------------------------------
# Fetch the export, so the step after `generate_version.py` needs no clicks either
# ---------------------------------------------------------------------------


def export_link(body: object) -> str | None:
    """The download URL from an export response, or None while it is still building.

    The API answers 202 with `{"ready": false, "progress": ...}` and 200 with
    `{"export": {"link": ...}}`. The body is what says which - the status code alone does
    not distinguish "ready" from "accepted", so nothing here reads it.
    """
    if not isinstance(body, dict):
        return None
    export = body.get("export")
    if isinstance(export, dict) and isinstance(export.get("link"), str):
        return export["link"]
    return None


def export_progress(body: object) -> float | None:
    if isinstance(body, dict) and isinstance(body.get("progress"), (int, float)):
        return float(body["progress"])
    return None


def extract_zip(archive: Path, dest: Path) -> list[Path]:
    """Extract a downloaded export, refusing any member that escapes `dest`.

    A zip is an untrusted input: a member named `../../.env` writes outside the directory it
    was extracted into. Nothing here expects Roboflow to ship one, which is exactly why the
    check is cheap to keep - it costs one `resolve()` per member.
    """
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise SystemExit(f"refusing to extract {member!r}: it escapes {dest}")
            written.append(target)
        zf.extractall(dest)
    return written


def download_export(
    project: str,
    version: int,
    dest: Path,
    key: str,
    fmt: str = EXPORT_FORMAT,
    get=None,
    sleep=None,
    timeout_s: int = 900,
    generation_name: str = DEFAULT_GENERATION.name,
) -> Path:
    """Ask for the version's export, wait for it, download and extract it.

    `get` and `sleep` are injectable for the same reason `camera_search.probe` takes a
    callable and the desktop injects `spawnFn`: the polling and the failure paths are the
    interesting parts and neither needs a network to be tested.
    """
    get = get or httpx.get
    sleep = sleep or time.sleep
    url = f"{API}/{ROBOFLOW_WORKSPACE}/{project}/{version}/{fmt}"
    deadline = time.monotonic() + timeout_s

    while True:
        r = get(url, params={"api_key": key, "nocache": "true"}, timeout=120)
        if r.status_code not in (200, 202):
            raise SystemExit(f"export request failed (HTTP {r.status_code}): {r.text[:200]}")
        body = r.json()
        link = export_link(body)
        if link:
            break
        if time.monotonic() > deadline:
            raise SystemExit(f"the export was still building after {timeout_s}s - re-run to resume")
        progress = export_progress(body)
        print("  building the export" + (f" {progress * 100:.0f}%" if progress else ""))
        sleep(10)

    # The link is a redirect to signed storage, so it has to be followed.
    archive = dest / f"{generation_name}-export.zip"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading the {fmt} export -> {archive}")
    r = get(link, timeout=900, follow_redirects=True)
    if r.status_code != 200:
        raise SystemExit(f"download failed (HTTP {r.status_code})")
    archive.write_bytes(r.content)
    extract_zip(archive, dest)
    return dest


# ---------------------------------------------------------------------------
# Export: find it, check it, normalize the yaml for ultralytics
# ---------------------------------------------------------------------------


def find_split_dirs(export_dir: Path) -> dict[str, Path]:
    """Locate the train/valid/test image directories inside a downloaded export.

    Roboflow's YOLO exports put `<split>/images/` at the archive root, but the zip
    sometimes extracts one level deeper, so both shapes are accepted. Only the image
    directory is returned: this is what the training run reads.
    """
    roots = [export_dir, *(p for p in sorted(export_dir.glob("*")) if p.is_dir())]
    found: dict[str, Path] = {}
    for root in roots:
        for split in SPLIT_NAMES:
            images = root / split / "images"
            if images.is_dir() and split not in found:
                found[split] = images
        if found:
            break
    return found


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def yaml_bodies(export_dir: Path):
    """Every readable `data.yaml` under the export, in path order, as mappings.

    One generator rather than one parse loop per reader: a built set carries two of these (the
    dataset's own `data.yaml` and the normalized `data.scanncart.yaml` `write_data_yaml` writes),
    and both `read_export_names` and `read_export_generation` walk them the same way - the first
    candidate that carries the field wins, so neither reader has to know which file it landed in.
    An unreadable or non-mapping candidate is skipped rather than raised: these are reads of
    somebody else's export, and the callers report absence.
    """
    import yaml  # PyYAML ships with ultralytics; see requirements.txt

    for candidate in sorted(export_dir.rglob(DATA_YAML_NAME)):
        try:
            body = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(body, dict):
            yield body


def read_export_resize_mode(export_dir: Path) -> str | None:
    """The `resize_mode` a built set declares for the weights trained on it, or None.

    `build_dataset.py` writes it (`--geometry fit` -> letterbox, the default -> stretch), because
    the geometry is a fact about the frames it wrote. A Roboflow export declares none, and then the
    generation's own requirement stands.
    """
    for body in yaml_bodies(export_dir):
        mode = body.get("resize_mode")
        if mode in ("stretch", "letterbox"):
            return str(mode)
    return None


def read_export_names(export_dir: Path) -> list[str] | None:
    """The class names the export declares, in the order it indexes them.

    Read from the export's own `data.yaml` rather than assumed, because this is the
    order the trained model's outputs will be in. A missing or unreadable yaml returns
    `None` - a defect the caller reports, not a crash.
    """
    for body in yaml_bodies(export_dir):
        names = body.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def read_export_generation(export_dir: Path) -> str | None:
    """The generation an export says it indexes, when its `data.yaml` names a known one.

    Written by `build_dataset.py` beside `names`, and read for the same reason it is written: the
    names list is ordered, so it is only readable against a generation - and a set handed to
    somebody else, or opened after the merge report has gone missing, should not have to be
    identified by matching its order against every roster by hand. `None` means *not recorded*,
    never a guess, and a value naming no generation this app has is treated as not recorded rather
    than quoted as a fact about the labels.
    """
    for body in yaml_bodies(export_dir):
        recorded = body.get("generation")
        if isinstance(recorded, str) and recorded in generations.GENERATIONS:
            return recorded
    return None


def frame_geometry(splits: dict[str, Path], sample: int = 12) -> list[str]:
    """What a sample of the export's frames says about how they were resized.

    Reported rather than used, and the distinction is the point: the `resize_mode` recorded beside
    the weights is the fact this is checked *against*, so deriving it here would turn one
    measurement into two sources of truth. What it can do is say whether the export about to be
    trained from looks like the geometry the record will claim - and a `stretch` requirement over
    letterboxed frames is exactly the mismatch that has silently cost this project a model before
    (`auto` resolving to letterbox for a `.pt` trained on a `Stretch to 640` version).

    The signature is in the border: a stretched frame fills its edge with image content, while a
    fitted one pads with a constant colour, so the padded axis is flat - a near-zero standard
    deviation along that whole edge - and near-identical across images. Square frames whose edges
    vary is what "Stretch to WxH" produces.

    Never raises and never decides anything on its own: unreadable frames, no frames at all, and a
    PIL that will not import all answer notes, because a geometry reading is worth having and is
    not worth failing a training run over.
    """
    images = [
        path
        for split in SPLIT_NAMES
        if (directory := splits.get(split)) is not None
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        return ["frames  no images to sample, so the geometry was not measured"]
    picked = images[:: max(1, len(images) // sample)][:sample]
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return ["frames  PIL/numpy unavailable, so the geometry was not measured"]

    sizes: dict[tuple[int, int], int] = {}
    padded: list[str] = []
    for path in picked:
        try:
            with Image.open(path) as image:
                size = image.size
                pixels = np.asarray(image.convert("RGB"), dtype="float32")
        except Exception as exc:  # noqa: BLE001 - a note, never a failure
            return [f"frames  could not read {path.name} ({type(exc).__name__})"]
        sizes[size] = sizes.get(size, 0) + 1
        edges = {
            "top": pixels[0],
            "bottom": pixels[-1],
            "left": pixels[:, 0],
            "right": pixels[:, -1],
        }
        flat = [name for name, edge in edges.items() if float(edge.std()) < 2.0]
        if flat:
            padded.append(f"{path.name} ({'/'.join(flat)})")

    shape = ", ".join(f"{w}x{h} x{n}" for (w, h), n in sorted(sizes.items(), key=lambda kv: -kv[1]))
    note = f"frames  {shape} of {len(picked)} sampled: "
    if padded:
        note += (
            f"{len(padded)} with a constant border ({'; '.join(padded[:2])}) - that is *fitting* "
            "with padding, which is not what a `stretch` requirement assumes. Check this "
            "generation's resize_mode against how the version was actually generated"
        )
    elif any(w != h for (w, h) in sizes):
        note += (
            "frames keep their own shape (not square), i.e. a `fit` build - the trainer "
            "letterboxes each one, so these weights need resize_mode letterbox"
        )
    else:
        note += "no constant border, i.e. stretched - consistent with the recorded requirement"
    return [note]


def pass_gate(dataset: Path) -> tuple[bool, list[str]]:
    """Whether the human pass is finished for the set about to be trained or measured.

    A weight's unread boxes in `valid`/`test` do not make the *training* wrong - they are the rows
    the set is built from, read back off disk like any other. What they make wrong is every number
    the run produces: `test` becomes a measurement of the annotator rather than of the model, and
    `valid` is the split the run *selects* on, so the checkpoint that comes out was chosen against
    an untrusted signal. That is already the acceptance gate's verdict; asking it here is asking the
    same question before ~2 h of GPU instead of after it, because the run whose headline figure has
    to be discarded is a run that has to be repeated.

    The rule is `accept_v2`'s own - `machine_only_verdict`, both halves, so "is this set's number
    quotable" has one owner and one wording - and it is imported here rather than at module scope
    for the same reason the doctor below is: `accept_v2` imports *this* module (for `IMGSZ`), so a
    module-level import is a cycle.

    A set with no merge report is **not** gated, and says so: `merge_report.json` is what
    `build_dataset.py` leaves beside a set it assembled, only a locally built set has machine-only
    decisions to count, and the other documented route (a Roboflow version export, whose labels came
    from that project's own annotator) has no provenance for this to read. Refusing there would
    block a path that is not dirty, merely unanswerable - the difference `accept_v2` draws as
    "cannot verify" and this one draws as "not this gate's set".
    """
    import accept_v2

    report = accept_v2.read_report(dataset)
    if not report:
        return True, [
            f"pass gate: {dataset} carries no {MERGE_REPORT_NAME}, so it is not a set "
            "build_dataset.py assembled - there is no machine-only state to check (a Roboflow "
            "export has none). Not gated."
        ]
    return accept_v2.machine_only_verdict(report)


def labels_class_order(generation: Generation, dataset: Path | None = None) -> list[str] | None:
    """The order the labels at `dataset` index their classes in, or `None` when it declares none.

    The set's own `data.yaml` is the only artifact that says which position means which product -
    it is the same read `check_export` judges and the same list a training run indexes a head by -
    so this is `read_export_names` with the generation's directory as the default.
    """
    return read_export_names(Path(dataset) if dataset is not None else generation.export_dir)


def require_labels_order(
    generation: Generation, dataset: Path | None = None, *, tool: str = ""
) -> tuple[str, ...]:
    """Refuse a set whose labels are not in `generation.classes` order, or return that order.

    This is the hole `check_export` names in its own docstring and then leaves open: it compares the
    export's names to the generation's by *membership*, so two lists holding the same products in
    different positions pass it - and every reader downstream takes a label row's first column as a
    **position in `generation.classes`**. The rows are then attributed to the wrong products, and
    nothing anywhere errors: `audit_recall.py --generation v2 --dataset-dir <v1's export>` printed
    `0.000` recall for every class, counted 97 instances of a product that set holds none of, and
    read out like a model that had learned nothing rather than a set read through another list. The
    tools that train do not need this (an export's order *is* the order its head was trained in), but
    the tools that *measure* one against labels do, which is why every one of them asks here.

    What this deliberately does **not** claim: that a weight from another generation cannot be
    measured. `audit_recall.collect` re-keys every detection through the *model's own* names before
    anything is scored, and reports what it cannot - so a v1 head audited against a v2-ordered set
    scores correctly, and refusing it would be the cry-wolf guard `check_export` warns about. What no
    re-keying can repair is the labels, which carry a bare index and no name at all: that is the
    one side of the comparison whose order has to be known rather than noticed.
    """
    where = Path(dataset) if dataset is not None else generation.export_dir
    names = labels_class_order(generation, dataset)
    prefix = f"{tool}: " if tool else ""
    if names is None:
        raise SystemExit(
            f"{prefix}no class list could be read from {where}"
            " - either there is no export there (generate or download it first) or its `data.yaml`"
            " names no classes, and either way nothing says which product a label row's index means."
        )
    if list(names) != list(generation.classes):
        raise SystemExit(
            f"{prefix}the labels at {where} are in another class order than "
            f"`{generation.name}`'s, so every label row would be read as a position in the wrong "
            "list and the products below it would be somebody else's. Both lists hold the same "
            "names; that is exactly why this is silent.\n"
            f"  the set declares {len(names)}: " + ", ".join(names) + "\n"
            f"  {generation.name} lists {len(generation.classes)}: " + ", ".join(generation.classes)
            + "\n"
            "Measure the set with its own generation (`--generation <that set's>`, or the "
            "`--dataset-dir` that pairs with this one), or regenerate the export with this "
            "generation's class order."
        )
    return tuple(names)


def check_export(
    export_dir: Path, generation: Generation = DEFAULT_GENERATION
) -> tuple[dict[str, Path], list[str]]:
    """Everything wrong with this export for `generation`, as a list. Also the split dirs.

    Runs before training on purpose: a class-list mismatch found after an hour of GPU time is an
    hour of GPU time, and found after deployment it is every box in the app.

    Judged against the *generation's* class list rather than one shared roster: the two entries are
    the same seven names today, but an export is judged against the list its own dataset declared,
    and a per-generation list is what stops a future addition from refusing a correct v1 set (or
    waving through a v2 head that lost a class). A check that cries wolf is how the real mismatch
    gets waved through.

    The membership half is `generations.class_gaps`, the same owner the merge judges each of its
    source lists with (`build_dataset.translation_problem`) - one implementation of "what does each
    list declare that the other does not", so an export cannot be judged one way here and the other
    way there. Only the sentences are this caller's, because what follows from a mismatch in a
    training run is about the head's outputs (a class the app cannot name, a product it can never
    predict) rather than about label rows.

    Note what this check does *not* do: it compares names by membership, so it cannot see a
    mis-mapped class *index* in the labels. For a Roboflow export that never mattered (the export's
    own order is the training order); for a locally built dataset it is the failure mode worth
    guarding, which is why `build_dataset` remaps indices by name and prints a contact sheet.
    """
    problems: list[str] = []
    if not export_dir.is_dir():
        return {}, [f"no export at {export_dir}"]

    splits = find_split_dirs(export_dir)
    for split in SPLIT_NAMES:
        if split not in splits:
            problems.append(f"no {split}/images directory in the export")
    for split, directory in splits.items():
        n = count_images(directory)
        if n == 0:
            problems.append(f"{split}/images is empty")
        else:
            print(f"  {split:6} {n:6} images")
    for note in frame_geometry(splits):
        print(f"  {note}")

    names = read_export_names(export_dir)
    expected = list(generation.classes)
    if names is None:
        problems.append("could not read `names` from the export's data.yaml")
    else:
        print(f"  classes {len(names)}: " + ", ".join(names))
        # The relational half is `generations.class_gaps` - the same owner the merge judges each of
        # its source lists with (`build_dataset.translation_problem`) - and the sentences below are
        # this caller's, because what follows from a mismatch here is about a *head* rather than
        # about label rows: an output the roster cannot name, or a product it can never predict.
        gaps = generations.class_gaps(names, expected)
        if gaps.target_only:
            problems.append(f"the export has no class for: {', '.join(gaps.target_only)}")
        if gaps.source_only:
            problems.append(
                f"the export declares classes that are not {generation.name} classes: "
                + ", ".join(gaps.source_only)
            )
            # The most likely cause, named, because the symptom does not point at it: a version
            # generated from a project whose class list has distances in it (one product split
            # into `close`/`mid`/`far`) trains a head with an output per product-and-distance.
            # Left as "extra classes", this reads as a project-id mix-up and sends the operator
            # to check the wrong thing - and the fix is a regenerate, not an edit here.
            tainted = {name: distance_tokens_in(name) for name in gaps.source_only}
            tainted = {name: words for name, words in tainted.items() if words}
            if tainted:
                problems.append(
                    "and "
                    + ", ".join(repr(name) for name in tainted)
                    + " carry a *distance* rather than a product: distance is a tag on the image "
                    "(MODEL_TRAINING.md 8.1), so those classes split one product into three and "
                    "this would train one output per product-and-distance. Fix the project's "
                    "class list, move the annotations onto the product class, and regenerate the "
                    "version - the app's own roster is the product names"
                )

        # Not a problem - it is a property of the dataset - but it is the exact thing the app will
        # say about these weights (`roster.class_list_problems`, the third finding: nothing they
        # *do* predict is wrong, which is why it is easy to miss), so it is better known before an
        # hour of GPU time than discovered afterwards. Read off the spec table rather than named
        # here, so it stays true if a third generation ever declares something neither of these
        # has.
        declared_anywhere = {n for g in generations.GENERATIONS.values() for n in g.classes}
        elsewhere = sorted(generations.class_gaps(declared_anywhere, expected).source_only)
        if elsewhere:
            print(
                f"  note  {len(elsewhere)} class(es) another generation declares are not in this"
                f" one, so these weights can never predict them: {', '.join(elsewhere)}"
            )
    return splits, problems


def write_data_yaml(export_dir: Path, splits: dict[str, Path], path: Path | None = None) -> Path:
    """Write a normalized `data.yaml` with absolute split paths.

    The export's own yaml uses paths relative to wherever Roboflow expected the archive
    to be extracted (`../train/images`, and the SDK rewrites them per format for exactly
    this reason). Writing one from absolute paths removes that assumption, so the run
    works from any extraction directory and from any cwd.
    """
    import yaml

    names = read_export_names(export_dir)
    body: dict = {
        "path": str(export_dir.resolve()),
        "nc": len(names) if names else 0,
        "names": names or [],
    }
    for split in SPLIT_NAMES:
        if split in splits:
            # Ultralytics resolves a relative entry against `path`; a bare directory
            # (not a glob) is what it expects for an image folder.
            body[split if split != "valid" else "val"] = str(splits[split].resolve())

    target = path or (export_dir / SCANNCART_DATA_YAML_NAME)
    target.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# The run's own log: the numbers the acceptance table is about
# ---------------------------------------------------------------------------


def read_results(path: Path) -> list[dict[str, float]]:
    """Parse ultralytics' `results.csv`, one dict per epoch.

    Column names carry a leading space in some ultralytics versions and not others, so
    they are stripped; a row that parses to nothing is skipped rather than kept as an
    epoch with no metrics in it.
    """
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                try:
                    row[key.strip()] = float(value)
                except (TypeError, ValueError):
                    continue
            if row:
                rows.append(row)
    return rows


def metric(row: dict[str, float], name: str) -> float | None:
    """Look a metric up by exact name or by its `(B)`-stripped form.

    Ultralytics labels detection metrics `metrics/mAP50(B)` and drops the suffix in
    some versions; both are answered here so a metric reported by the run is never
    silently missing from the verdict.

    The fallback tolerates only a *task suffix* - `base` followed by `(`. A plain
    `startswith(base)` would answer `mAP50` with `mAP50-95`'s value, since one is a prefix
    of the other, and would then report a passing mAP50 that was never measured.
    """
    base = name.replace("(B)", "")
    for key in (name, base, f"{base}(B)"):
        if key in row:
            return row[key]
    for key, value in row.items():
        if key.startswith(f"{base}("):
            return value
    return None


def final_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return rows[-1] if rows else {}


def best_epoch(rows: list[dict[str, float]], key: str = "metrics/mAP50-95(B)") -> tuple[int, float] | None:
    """(epoch, value) for the best row. The epoch is the log's own number, which is
    **1-based** - `Trainer.save_metrics` writes `self.epoch + 1` - so it is printed as-is
    rather than converted."""
    scored = [(int(r["epoch"]), v) for r in rows if (v := metric(r, key)) is not None]
    return max(scored, key=lambda kv: kv[1]) if scored else None


def verdict(rows: list[dict[str, float]]) -> tuple[list[str], list[str]]:
    """(passed, failed) against TARGETS, from the final epoch.

    Deliberately reports only what a training log can answer. `patience` means the final
    epoch is not necessarily the best one, and neither number says anything about a
    single class - a model passing both here can still be failing `far` items, which is
    what `--val` exists to check.
    """
    last = final_metrics(rows)
    passed: list[str] = []
    failed: list[str] = []
    for name, target in TARGETS.items():
        value = metric(last, name)
        label = name.replace("metrics/", "")
        if value is None:
            failed.append(f"{label}: not in the run log (expected >= {target:.2f})")
        elif value >= target:
            passed.append(f"{label} {value:.3f} >= {target:.2f}")
        else:
            failed.append(f"{label} {value:.3f} < {target:.2f}")
    return passed, failed


# ---------------------------------------------------------------------------
# The validation pass: the per-class number a training log does not contain
# ---------------------------------------------------------------------------


def validation_kwargs(
    data_yaml: Path, split: str, project: Path, name: str, imgsz: int = IMGSZ
) -> dict:
    """The validation pass, as kwargs - one dict so the printed pass and the call agree.

    `imgsz` is pinned to the size the run trained at rather than inherited from the
    checkpoint, so the two numbers being compared describe the same model. It is a parameter
    rather than the constant because the constant is only right while nobody passes `--imgsz`:
    a run trained at 960 and measured at 640 is a *different* configuration, and the number it
    reports is not the trained model's (`main` feeds this the run's own `args.yaml`).

    `name` is a parameter because the per-distance passes are separate runs of it: they share
    one `project` directory, and without distinct names the last pass would overwrite the
    confusion matrix the overall pass drew - leaving the artifact on disk describing `far`
    while the terminal had just printed the split's numbers.
    """
    return {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "project": str(project),
        "name": name,
        # Re-runs reuse the directory. The plots are a reading aid; piling up `-val2`,
        # `-val3` ... would make the current one the newest rather than the one you opened.
        "exist_ok": True,
    }


def as_list(value: object) -> list:
    """`list(value)`; None and a non-iterable answer empty.

    Not `value or []`. Ultralytics hands back **numpy arrays**, and a one-element array is
    truthy or falsy by its *contents*: `box.r` for a split whose only scored class recalls
    0.0 is `array([0.0])`, which is falsy. `or []` therefore turned a measured zero into
    "nothing was measured" - and a measured zero is the loudest result this step can
    produce, so swallowing it is the worst possible failure here. (The live probe caught
    this; a fake built from Python lists never would, since `[0.0]` is truthy.)
    """
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def names_by_index(names: object) -> dict[int, str]:
    """`{class id: name}` from ultralytics' `names`, which is a dict in 8.x."""
    if isinstance(names, dict):
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            return {}
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def per_class_recall(metrics: object) -> list[tuple[str, float | None, int]]:
    """(name, recall, instances) for every class the model knows.

    `box.r` is **positional**, not indexed by class id: `r[i]` belongs to
    `ap_class_index[i]`. Ultralytics builds `ap_class_index` from the classes that have
    ground-truth labels in the split, so a class missing from the split shifts every row
    after it - and with a per-distance holdout that is the normal case, not the edge one.
    Zipping `names` against `box.r` would therefore report one class's recall under another
    class's name: no error, and a plausible number.

    A class with no ground-truth instances gets `None` rather than the 0.0 ultralytics would
    answer. Zero reads as a total miss, when the truth is that the split never asked - and
    the two call for opposite actions (more data for the class vs. more captures for the
    split), so they are kept apart here and reported apart below.
    """
    names = names_by_index(getattr(metrics, "names", None))
    box = getattr(metrics, "box", None)
    recalls = as_list(getattr(box, "r", None))
    index = as_list(getattr(box, "ap_class_index", None))
    counts = as_list(getattr(metrics, "nt_per_class", None))

    scored: dict[int, float] = {}
    for position, class_id in enumerate(index):
        if position < len(recalls):
            scored[int(class_id)] = float(recalls[position])

    rows: list[tuple[str, float | None, int]] = []
    for class_id, name in sorted(names.items()):
        n = int(counts[class_id]) if class_id < len(counts) else 0
        rows.append((name, scored.get(class_id), n))
    return rows


def recall_report(
    rows: list[tuple[str, float | None, int]], floor: float = RECALL_FLOOR
) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    """(passed, failed, unmeasured, {class: recall}) against the floor.

    Three outcomes rather than two. A class the split never asked about is neither a pass
    nor a miss, and folding it into either one hides the cell that needs more captures -
    which is what 8.3's coverage check exists to catch.
    """
    passed: list[str] = []
    failed: list[str] = []
    unmeasured: list[str] = []
    values: dict[str, float] = {}
    for name, recall, n in rows:
        if recall is None:
            why = "no ground-truth instances in the split" if n == 0 else "not scored"
            unmeasured.append(f"{name}: {why} (nothing to measure)")
            continue
        values[name] = recall
        if recall >= floor:
            passed.append(f"{name} {recall:.3f} >= {floor:.2f} (n={n})")
        else:
            failed.append(f"{name} {recall:.3f} < {floor:.2f} (n={n})")
    return passed, failed, unmeasured, values


def aggregate_metrics(metrics: object) -> dict[str, float]:
    """Mean precision/recall and mAP from the pass itself.

    Separate from `verdict()`, which reads the **training log** - the valid split, final
    epoch. These describe whichever split was asked for, so `--split test` makes them the
    acceptance numbers rather than the selection ones. `mAP50`/`mAP50-95` carry the same
    targets as TARGETS; precision and recall are reported without one because 6 sets none.
    """
    box = getattr(metrics, "box", None)
    out: dict[str, float] = {}
    for label, attr in (
        ("precision", "mp"),
        ("recall", "mr"),
        ("mAP50", "map50"),
        ("mAP50-95", "map"),
    ):
        try:
            out[label] = float(getattr(box, attr))
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def validate(
    weights: Path,
    data_yaml: Path,
    split: str,
    project: Path,
    name: str,
    yolo=None,
    imgsz: int = IMGSZ,
):
    """Load `weights` and validate the split; returns ultralytics' metrics object.

    `yolo` is injectable for the same reason `download_export`'s `get`/`sleep` are: the pass
    needs a GPU and a dataset, and what is worth testing is what is *done* with its answer.
    """
    if yolo is None:
        yolo = ultralytics_yolo()
    return yolo(str(weights)).val(
        **validation_kwargs(data_yaml, split, project, name, imgsz=imgsz)
    )


def weights_sha256(weights: Path) -> str:
    """The identity of the file a measurement describes.

    A hash rather than a timestamp, because `--val` and `--install` are separate commands:
    re-training into the same `--run-dir` replaces `best.pt` in place, and a stale
    measurement reads exactly like a fresh one. Attaching one would put a number in front of
    an operator that these weights never scored - the failure this record exists to prevent,
    so the record checks rather than assumes.
    """
    digest = hashlib.sha256()
    with weights.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def val_metrics_path(best: Path) -> Path:
    """Where a validation pass writes its numbers: in the run, beside `results.csv`.

    In the run directory rather than in the repo or in `models/`, because that is what the
    numbers are about: the documented sequence runs `--val` and then `--install` as separate
    commands, so the fact has to survive on disk between them, and a file sitting next to the
    checkpoint it measured cannot be attached to a different one by accident.
    """
    return best.parent.parent / VAL_METRICS_NAME


def class_rows(rows: list[tuple[str, float | None, int]]) -> list[dict]:
    """`per_class_recall()`'s rows as the dicts that get written down.

    One conversion, used by the split's record and by every distance breakdown under it, so the
    table printed and the numbers recorded cannot be two parses of the same pass that disagree.
    """
    return [{"name": name, "recall": recall, "instances": n} for name, recall, n in rows]


def validation_record(
    rows: list[tuple[str, float | None, int]],
    split: str,
    aggregates: dict[str, float],
    floor: float = RECALL_FLOOR,
    weights_hash: str = "",
) -> dict:
    """One split's measurement, as the block that gets written down.

    Built from `per_class_recall()`'s rows rather than from the printed report, so the table
    the operator reads and the numbers the panel shows later are the same parse of the same
    pass - and a class the split holds no instances of stays `recall: None` here too. That is
    what keeps "never measured" out of the 0.000 slot, in the record as well as on screen:
    the two call for opposite actions, and a number in a file outlives the terminal it was
    printed to.
    """
    return {
        "split": split,
        "floor": floor,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights_sha256": weights_hash,
        "aggregates": aggregates,
        "per_class": class_rows(rows),
    }


def load_val_metrics(path: Path, weights_hash: str) -> list[dict]:
    """The measurements in `path` made on `weights_hash`. Never raises.

    A missing file, a corrupt one and a measurement of *different* weights all answer `[]`,
    because they mean the same thing to the caller: nothing is known about these weights. The
    hash is what separates the third case from the others - it is how re-training into the
    same run directory drops the previous checkpoint's numbers instead of carrying them over.
    """
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(body, list):
        return []
    return [
        block
        for block in body
        if isinstance(block, dict)
        and block.get("weights_sha256") == weights_hash
        and isinstance(block.get("split"), str)
    ]


def write_val_metrics(path: Path, block: dict) -> None:
    """Record one split's measurement, replacing any earlier one for that split.

    Merged rather than overwritten: `--val --split valid` after `--val` (which reads `test`)
    would otherwise delete the acceptance number in favour of the selection one - and the
    selection number is the one that flatters the run. Keyed on the split so both can be
    held at once, with `test` first so the file reads in the order the panel shows it.
    """
    kept = [
        b
        for b in load_val_metrics(path, block["weights_sha256"])
        if b.get("split") != block["split"]
    ]
    merged = kept + [block]
    merged.sort(key=lambda b: (b.get("split") != DEFAULT_SPLIT, str(b.get("split"))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The per-distance breakdown: what the per-class number averages away
# ---------------------------------------------------------------------------


def recorded_distances(pairs) -> dict[str, str]:
    """`{filename: distance}` from `(name, distance)` pairs, keeping only what the plan follows.

    One rule for both places a run's distances can come from - a built set's own report and the
    staged manifest - because two readers that disagree about what counts as a distance is how the
    same frame ends up in two columns of one grid. A name nothing can use, or a distance outside
    `DISTANCE_ORDER`, is dropped rather than carried: an unknown one would make `images_by_distance`
    key on a column no pass is ever run for.
    """
    out: dict[str, str] = {}
    for name, distance in pairs:
        if isinstance(name, str) and name and distance in DISTANCE_ORDER:
            out[name] = distance
    return out


def distances_in(body: object) -> dict[str, str]:
    """`{filename: distance}` from either shape a record of them can take, else `{}`.

    The *shape* is the only thing that ever differed between the two files that carry this fact:
    a staged manifest is a list of entries whose `new_name` and `distance` are the pair, while a
    built set's own `merge_report.json` already holds the mapping under `distances`. Both routes
    end in `recorded_distances`, the one rule for what counts as a distance - a second reader that
    validated differently is how the same frame ends up in two columns of one grid.

    `{}` for anything else, including a body that is neither shape: this reads files written by
    another program, and the caller's contract is a lost breakdown rather than an exception.
    """
    if isinstance(body, list):
        return recorded_distances(
            (entry.get("new_name"), entry.get("distance"))
            for entry in body
            if isinstance(entry, dict)
        )
    pairs = body.get("distances") if isinstance(body, dict) else None
    if isinstance(pairs, dict):
        return recorded_distances(pairs.items())
    return {}


def distance_map(path: Path | None) -> dict[str, str]:
    """`{export filename: distance}` from any file that records them, else `{}`.

    The join happens on filenames because a **YOLO export carries no tags**: the distance is a
    Roboflow tag, and a version export is images, labels and a yaml. Two files hold the joined
    pair, and this reads both - the manifest the dataset tooling wrote, and the built set's own
    `merge_report.json`, which `dataset_distances` prefers because the set is the artifact under
    test: the staged directory it was built from can be moved, cleaned or on another machine
    entirely by the time the grid runs.

    `None` means *this generation declares no distance axis* (`generations.Generation.manifest`),
    which is why it is not defaulted to a path here: v1 predates the tagging, so "no manifest"
    would be a lie about a file, and the caller gives that case its own sentence. Never raises,
    and answers `{}` for a missing, unreadable or unrecognised file too: a run that cannot find
    distances should lose the breakdown, not the validation.
    """
    if path is None:
        return {}
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return distances_in(body)


def dataset_distances(
    dataset: Path | None, manifest: Path | None, explicit: bool = False
) -> tuple[dict[str, str], str]:
    """`({filename: distance}, where they came from)` for a run's per-distance passes.

    The order is the point. A built set carries its own distances, and it is what `--val` measures:
    preferring them means the grid runs wherever the set is, which is the difference between a
    breakdown that exists and one that dies with a `cleaned-v2/` directory nobody kept. The staged
    manifest is the fallback - an older build, or a run over a directory that is not a built set -
    and `--manifest` named by hand answers first, because naming a file is a decision rather than a
    default. An empty answer with an empty sentence is the third state: nothing recorded distances,
    which the caller reports as a skipped breakdown instead of a clean one.
    """
    from_manifest = distance_map(manifest)
    if explicit and from_manifest:
        return from_manifest, f"from the manifest {manifest}"
    report = dataset / MERGE_REPORT_NAME if dataset is not None else None
    if found := distance_map(report):
        return found, "from the set's own merge_report.json"
    if from_manifest:
        return from_manifest, f"from the manifest {manifest}"
    return {}, ""


def images_by_distance(
    directory: Path, distances: dict[str, str]
) -> tuple[dict[str, list[Path]], list[str]]:
    """(distance -> images, names with no recorded distance) for one split's directory.

    The unmatched names are returned rather than dropped: images in a split that no distance
    accounts for are in the overall number and in no distance row, and a report that silently
    omits them would make the two look reconcilable when they are not.
    """
    found: dict[str, list[Path]] = {d: [] for d in DISTANCE_ORDER}
    unknown: list[str] = []
    if not directory.is_dir():
        return found, unknown
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        distance = distances.get(path.name)
        if distance is None:
            unknown.append(path.name)
        else:
            found[distance].append(path)
    return found, unknown


def write_distance_list(path: Path, images: list[Path]) -> Path:
    """One absolute image path per line - how ultralytics is handed a *subset* of a split.

    A dataset entry may be a directory, which it globs, or a **file**, which it reads as a list
    of images (`BaseDataset.get_img_files`). The file is what makes this affordable: measuring
    `far` alone needs a subset, and materialising three of them by copying would cost a
    gigabyte of JPEGs for a number that three text files answer for free.

    The paths are absolute because a relative line is resolved against the list's own directory,
    and the images are in the export, not in the run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p.resolve()}\n" for p in images), encoding="utf-8")
    return path


def distance_data_yaml(
    export_dir: Path, splits: dict[str, Path], split: str, list_path: Path, path: Path
) -> Path:
    """The run's `data.yaml` with one split pointed at a file list instead of a directory.

    Written by the same function the run itself uses, so a distance pass and the overall pass
    cannot be reading different class lists, a different `nc`, or a different dataset root.
    """
    return write_data_yaml(export_dir, {**splits, split: list_path}, path)


def distance_breakdown(
    weights: Path,
    export_dir: Path,
    splits: dict[str, Path],
    split: str,
    project: Path,
    distances: dict[str, str],
    yolo=None,
    out_dir: Path | None = None,
    val: str = "",
) -> tuple[list[dict], list[str]]:
    """One validation pass per distance, as blocks, plus the notes on what could not be measured.

    A pass per distance rather than a slice of one, because ultralytics answers recall for a
    whole split and `DetMetrics` holds no per-image breakdown to cut up afterwards. Each pass is
    the library's own matching over a different image set, so a `far` number and the overall
    number are the *same measurement* over different sets - which is what makes comparing them
    worth anything. Getting both from a single pass would mean reimplementing IoU matching here,
    and the two numbers would then disagree in ways neither of them could explain.

    `--no-per-distance` exists because this costs three extra passes: cheap on a test split of a
    few hundred images, less so on a full one, and never worth it when the answer is not wanted.
    """
    images_dir = splits.get(split)
    if images_dir is None:
        return [], [f"no {split}/images directory in the export to break down"]

    by_distance, unknown = images_by_distance(images_dir, distances)
    if not any(by_distance.values()):
        return [], [
            f"no distances for the {split} split: {len(unknown)} image(s) and none carried a"
            " recorded distance - the breakdown is skipped, not reported as clean"
        ]

    base = out_dir or (project / DISTANCE_DIR_NAME)
    blocks: list[dict] = []
    notes: list[str] = []
    for distance in DISTANCE_ORDER:
        images = by_distance[distance]
        if not images:
            notes.append(f"{distance}: no images in the {split} split, so it is unmeasured")
            continue
        list_path = write_distance_list(base / f"{split}-{distance}.txt", images)
        yaml_path = distance_data_yaml(
            export_dir, splits, split, list_path, base / f"data-{split}-{distance}.yaml"
        )
        metrics = validate(
            weights,
            yaml_path,
            split,
            project,
            name=f"{val or val_name(DEFAULT_GENERATION)}-{distance}",
            yolo=yolo,
        )
        # No `floor` here: the block these hang under carries the one they were judged
        # against, and a copy per distance could only ever disagree with it.
        blocks.append(
            {
                "distance": distance,
                "images": len(images),
                "aggregates": aggregate_metrics(metrics),
                "per_class": class_rows(per_class_recall(metrics)),
            }
        )
    if unknown:
        notes.append(
            f"{len(unknown)} image(s) in the {split} split carry no distance and are counted in"
            " the overall number only"
        )
    return blocks, notes


def _grid_cell(recall: float | None, instances: int, floor: float) -> str:
    """One cell: the number and its instance count, `!` when below the floor, `-` when unasked.

    `-` rather than `0.000` for a class the subset held no instances of, for the reason the
    per-class report has three states: "this distance has none of that item" and "it missed
    every one it had" lead to opposite work.
    """
    if recall is None:
        return "-"
    return f"{recall:.3f} ({instances}){'!' if recall < floor else ''}"


def distance_grid(
    rows: list[tuple[str, float | None, int]], blocks: list[dict], floor: float = RECALL_FLOOR
) -> list[str]:
    """The class x distance table, as lines.

    `rows` is the overall pass, and it supplies both the class order and the `all` column - so
    the far/close comparison is read next to the number it is being averaged into. Rows are
    every class the model knows, not only the ones a subset scored: a class missing from a
    distance *is* the finding there.
    """
    names = [name for name, _recall, _n in rows]
    overall = {name: (recall, n) for name, recall, n in rows}
    per_distance = {
        block["distance"]: {c["name"]: c for c in block["per_class"]} for block in blocks
    }

    head = "class".ljust(max([len(n) for n in names] + [len("class")])) + "".join(
        d.rjust(13) for d in DISTANCE_ORDER
    )
    head += "all".rjust(13)
    lines = [head, "-" * len(head)]
    for name in names:
        cells = []
        for distance in DISTANCE_ORDER:
            entry = per_distance.get(distance, {}).get(name)
            cells.append(
                _grid_cell(entry["recall"], entry["instances"], floor) if entry else "-"
            )
        recall, instances = overall.get(name, (None, 0))
        cells.append(_grid_cell(recall, instances, floor))
        lines.append(name.ljust(len(head) - 13 * len(cells)) + "".join(c.rjust(13) for c in cells))
    return lines


def distance_misses(blocks: list[dict], floor: float = RECALL_FLOOR) -> list[str]:
    """Every (class, distance) under the floor, worst first - the actionable half of the grid."""
    found: list[tuple[float, str]] = []
    for block in blocks:
        for cell in block["per_class"]:
            recall = cell["recall"]
            if recall is not None and recall < floor:
                found.append(
                    (
                        recall,
                        f"{block['distance']} {cell['name']} {recall:.3f}"
                        f" (n={cell['instances']})",
                    )
                )
    return [line for _recall, line in sorted(found)]


# ---------------------------------------------------------------------------
# Training, then the drop-in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hyper:
    """MODEL_TRAINING.md 6's run, plus the two knobs this machine has a say in.

    Grouped rather than passed loose because two callers need the same answer and the printed
    command has to be the run that happens: `train_kwargs()` builds the dict ultralytics is
    called with and `command_line()` prints it, both from this one value.
    """

    epochs: int = EPOCHS
    imgsz: int = IMGSZ
    # The ceiling. `budget.batch_size()` may lower it, never raise it.
    batch: int = BATCH
    patience: int = PATIENCE
    base_model: str = BASE_MODEL
    # "auto" resolves through `resources.resolve_device` - CUDA when it is real, else CPU.
    device: str = "auto"
    # Augmentation, as the two flags that have one (`AUGMENTATION` carries the rest).
    degrees: float = DEGREES
    scale: float = SCALE
    # Dataloader processes. Left at 0, `derive_workers` answers from the CPU budget.
    workers: int = 0


def derive_workers(cpu_threads: int) -> int:
    """How many dataloader processes the CPU budget allows.

    Ultralytics defaults to 8, which on a 6-core box is most of the machine: each worker is a
    *process* decoding and augmenting in Python, so the OMP/MKL thread caps `resources` applies
    inside the training process do not bound them - they are what the budget would otherwise miss.
    Two keeps a 4060 fed at 640 while leaving the box usable, and the trade is deliberate: some GPU
    idle time in exchange for a machine that can still run the app beside it. `--workers` raises it
    when the box is otherwise idle.
    """
    return max(1, min(2, cpu_threads))


def augmentation_kwargs(hyper: Hyper | None = None) -> dict[str, float]:
    """4's table as ultralytics' hyperparameters, with this run's rotation and scale.

    One function so the printed command, the call ultralytics gets and the record written beside
    the weights are three readings of one dict - a run whose record disagreed with what trained
    would be worse than no record, because it would be believed.
    """
    values = dict(AUGMENTATION)
    values["degrees"] = DEGREES if hyper is None else hyper.degrees
    values["scale"] = SCALE if hyper is None else hyper.scale
    return {key: values[key] for key in AUGMENTATION_KEYS}


def train_kwargs(
    data_yaml: Path, project: Path, run_name: str, hyper: Hyper, device: str
) -> dict:
    """The training run, as kwargs. One dict so the printed command and the call that
    runs cannot drift apart."""
    return {
        "data": str(data_yaml),
        "epochs": hyper.epochs,
        "imgsz": hyper.imgsz,
        "batch": hyper.batch,
        "patience": hyper.patience,
        "project": str(project),
        "name": run_name,
        "device": device,
        "workers": hyper.workers,
        **augmentation_kwargs(hyper),
    }


def command_line(data_yaml: Path, project: Path, run_name: str, hyper: Hyper, device: str) -> str:
    kw = train_kwargs(data_yaml, project, run_name, hyper, device)
    return (
        f"yolo detect train model={hyper.base_model} data={data_yaml} epochs={kw['epochs']} "
        f"imgsz={kw['imgsz']} batch={kw['batch']} patience={kw['patience']} "
        f"degrees={kw['degrees']:g} scale={kw['scale']:g} "
        f"project={project} name={run_name} device={device} workers={kw['workers']}"
    )


def run_dir(project: Path, run_name: str) -> Path:
    """Where ultralytics puts this run, including its `-2`, `-3` suffix on a re-run."""
    base = project / run_name
    if not base.exists():
        return base
    n = 2
    while (base.parent / f"{run_name}-{n}").exists():
        n += 1
    return base.parent / f"{run_name}-{n}"


def find_best(run: Path) -> Path | None:
    for candidate in (run / "weights" / "best.pt", run / "weights" / "last.pt"):
        if candidate.is_file():
            return candidate
    return None


def run_args(run: Path) -> dict:
    """The training arguments ultralytics saved into the run (`args.yaml`), or `{}`.

    Read for the facts a *flag* cannot answer: `--val` and `--install` are separate commands from
    `--yes` in the documented sequence, so their `--imgsz`/`--degrees` are whatever the caller
    typed this time (or the module defaults) rather than what the run that produced these weights
    used. Pinning a validation pass, or a record, to the wrong size is not a wrong number on
    screen - it is a number about a different configuration, believed because it is written down.

    Ultralytics writes the run's own arguments beside its weights, which makes them a measurement
    of the run instead of a claim about it. Missing, unreadable and half-written all give `{}`,
    and the caller keeps its own values: a run predating this read is not an error.

    `yaml.YAMLError` rather than `ValueError`, which is not what a malformed document raises here:
    PyYAML's parser and scanner errors derive from `Exception` directly, so the JSON-shaped
    `except ValueError` this repo uses elsewhere would let a half-written file take the command
    down - at the one moment it is reading a file to avoid a wrong *number*.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml ships with ultralytics
        return {}
    try:
        body = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return body if isinstance(body, dict) else {}


def trained_value(args: dict, key: str, fallback: float) -> float:
    """`args[key]` when it is a real number, else `fallback`.

    `bool` is excluded even though it is an `int`: `True` would arrive as `1.0` and a record
    saying `degrees: 1.0` is a wrong number rather than a missing one. A string ("640", which is
    how a hand-edited yaml arrives) is a fallback too, deliberately: coercing it would mean
    guessing a unit, and the fallback is at least the value this command was built with.
    """
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(fallback)
    return float(value)


def latest_run(project: Path, run_name: str, val: str = "") -> Path | None:
    """The most recently written run of `run_name` under `project`, or None if there is none.

    `run_dir()` answers "where would the *next* run go", which is what training needs and
    the opposite of what `--val` and a bare `--install` need: once a run has finished, that
    directory does not exist yet - the finished one is `<name>`, so `run_dir()` looked for
    `<name>-2` and failed. Ordered by modification time because the `-10` suffix does not sort
    after `-9`.

    `val` is this generation's val-run name, excluded so a measurement pass is never offered as
    the run to install from.
    """
    runs = [
        p
        for p in project.glob(f"{run_name}*")
        if p.is_dir() and p.name != val and find_best(p)
    ]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def resolve_run(
    project: Path, generation: Generation, explicit: str = "", training: bool = False
) -> Path:
    """Which run `--val`/`--install` should read, and which `--yes` will write.

    A fresh training run is named before it exists; a finished one has to be *found* - an
    explicit `--run-dir`, else the newest that has weights. The parameters are named to
    keep this file's own history out of it: `run_dir` is both a function here and the
    `--run-dir` argument, and a bare `run_dir` in this body would bind whichever of the two
    a reader happens to assume.
    """
    if explicit:
        return Path(explicit).expanduser()
    if training:
        return run_dir(project, generation.run_name)
    return latest_run(project, generation.run_name, val_name(generation)) or run_dir(
        project, generation.run_name
    )


def weight_record(
    generation: Generation = DEFAULT_GENERATION,
    source_version: int = 0,
    project: str = "",
    validation: list[dict] | None = None,
    class_names: list[str] | None = None,
    imgsz: int = 0,
    resize_mode: str | None = None,
    augmentation: dict[str, float] | None = None,
) -> dict:
    """What travels with the weights, as a dict. Written beside them by `install()`.

    `resize_mode` is the field this file exists for. A locally trained `.pt` resolves
    `resize_mode: auto` to **letterbox**, while the version it was trained from was generated
    with `Stretch to 640` - so the model expects the stretched geometry and the setting
    silently gives it the letterboxed one, worst on the `far` cells where the pixels were
    already scarce. Nothing errors. There is no other place the requirement could be known
    from: the checkpoint records the training run, not the dataset geometry, and the filename
    is a convention rather than a fact.

    Taken from the generation rather than written out here, and the two sources it allows are
    different on purpose: v2's is `generate_version.REQUIRED_RESIZE_MODE`, because that block *is*
    the preprocessing that produced its export, while v1's is a value frozen at the geometry its
already-generated version was made with (the measurement is in `generations.py`). Deriving v1's
    from v2's constant would let a later change to v2's preprocessing silently rewrite a
    requirement about an export that already exists and will never be regenerated.

    `class_names` is the export's own class list, in the order the model indexes them, and it
    is recorded for the same reason: a class list is a property of the *weights* and nothing
    else keeps it. The export that produced it is gone by the time anyone runs it, so a model
    trained from a project whose class list was split by distance - `safeguard close` /
    `safeguard mid` / `safeguard far` instead of one `safeguard` - has 21 outputs, loads without
    complaint, and logs one product under three labels with nothing erroring anywhere. Written
    down, the *listing* can catch that before those weights ever run, instead of leaving it to
    someone pressing Test connection (`app/roster.class_list_problems` judges the names; the
    Admin Panel shows the findings beside the weight). A `--install` that knows the names and
    does not write them down is the one place the fact existed and was thrown away.

    Omitted rather than written empty when the export declared none - an empty list would read
    as "this model predicts nothing" instead of "not recorded", and the reader treats that
    absence the same way. In practice `check_export` has already refused a version whose
    `names` could not be read, so this is the shape of the field rather than a live path.

    `validation` is `--val`'s per-class recall, attached only when a measurement of *these*
    weights was found (`load_val_metrics`) - so the panel can show what the model scored
    rather than only what it needs. Left out entirely, rather than written as null, when there
    is none: a record without numbers and a record explicitly saying "no numbers" are the same
    state, and the field's absence keeps the file byte-identical for a re-install.

    `weights_sha256` is stripped from those blocks on the way in. The record sits beside the
    weight it describes, so a hash inside it would be a copy of something already implied by
    its location - and a copy that a `--force` replacement of the `.pt` would leave wrong.

    No `model` field: the record's own filename is the model, and a copy of it named to
    something else would then disagree with the file it sits beside - which is how a
    field-by-field record turns into a second, wrong answer to "which weights are these?".

    `imgsz` is the size the run trained at, recorded for the mirror image of the `resize_mode`
    problem: `Settings.imgsz` decides what the app resizes every frame to before detection, and
    nothing relates that field to the weights - so a model trained at 960 runs at 640 by default
    and the `far` detections come back weaker with no explanation anywhere. `resize_mode` was the
    first half of "the app feeds the model the wrong geometry"; this is the second.

    `augmentation` is what trained, not what this command's flags said - `main` reads it out of the
    run's own `args.yaml`. It is evidence rather than an input: nothing in the app branches on it,
    and it exists so "which run produced these weights" has an answer that survives the terminal,
    for the one augmentation whose absence was measurable (rotation) and for the three exclusions
    the doc names.
    """
    record = {
        "generation": generation.name,
        # The set's own declaration when it has one (a local `fit` build is letterbox whatever the
        # generation's default is), else the generation's requirement.
        "resize_mode": resize_mode or generation.resize_mode,
        # Where it came from, so the panel can say which dataset produced these weights
        # rather than only what they need.
        "source": f"{project} version {source_version}" if source_version else "",
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if class_names:
        record["class_names"] = [str(name) for name in class_names]
    if imgsz:
        record["imgsz"] = int(imgsz)
    if augmentation:
        record["augmentation"] = {key: float(augmentation[key]) for key in AUGMENTATION_KEYS if key in augmentation}
    if validation:
        record["validation"] = [
            {k: v for k, v in block.items() if k != "weights_sha256"} for block in validation
        ]
    return record


def install(
    source: Path,
    models_dir: Path,
    name: str = DEFAULT_GENERATION.weight_name,
    force: bool = False,
    record: dict | None = None,
) -> Path:
    """Copy the checkpoint to `models/<name>`, refusing to clobber one.

    Refusing rather than overwriting: the picker is keyed by filename, so an overwrite
    silently replaces the model a running app is configured with, and there is no
    version history to fall back to. `--force` is the deliberate way to do that.

    `record` is written to `models/<stem>.json` next to the weight. It is a separate file
    rather than a `models/`-global list because the weight is the thing that moves: copying
    the `.pt` to another machine and leaving the manifest behind would lose the requirement
    at exactly the moment it is needed.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / name
    if target.exists() and not force:
        raise SystemExit(
            f"{target} already exists - refusing to overwrite it (--force to replace).\n"
            "Name the new generation instead: MODEL_TRAINING.md 8.2."
        )
    shutil.copy2(source, target)
    if record is not None:
        (models_dir / f"{target.stem}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    return target


def print_reminders(
    resize_mode: str | None, generation: Generation = DEFAULT_GENERATION
) -> None:
    print()
    print("Next, and this is the field that decides whether any of the above is visible:")
    print()
    if resize_mode is None:
        # Only reachable if the version's resize format stops being one of the two the
        # sidecar can reproduce. Saying so is the point: guessing would put a wrong
        # requirement in the record the panel checks against.
        print("  This generation's preprocessing has no sidecar `resize_mode` equivalent, so")
        print("  nothing was recorded. Set resize_mode by hand to match how it was trained.")
        return
    print(f"  Nothing to set: these weights need resize_mode: {resize_mode}, and it has")
    print(
        f"  been recorded as models/{Path(generation.weight_name).stem}.json. `auto` honours the record,"
    )
    print("  so leaving the field on its default is the correct geometry.")
    print()
    print(f"  Setting `{resize_mode}` by hand is equivalent. Setting anything else overrides the")
    print("  record, which the Admin Panel's Model field flags - it is the only way to get the")
    print("  geometry wrong from here now.")
    print()
    print("  active_model is restart-required: stop capture before saving it.")


def print_validation_note(validation: list[dict]) -> None:
    """Say whether `--val`'s numbers travelled with the weights, and what they were.

    Printed at install time because that is the last moment the absence is still fixable: the
    record is what the panel reads, so installing without a measurement means the panel will
    have nothing to show, and saying so here is cheaper than having it noticed later.
    """
    print()
    if not validation:
        print("  No measurement of *these* weights was found, so none was recorded: `--val`")
        print("  writes the per-class recall the Admin Panel's Model field displays. A `--val`")
        print("  that ran before a re-training measures the earlier checkpoint, and is")
        print("  deliberately not attached to this one.")
        return
    for block in validation:
        classes = block.get("per_class") or []
        floor = block.get("floor", RECALL_FLOOR)
        below = [
            c["name"]
            for c in classes
            if isinstance(c.get("recall"), (int, float)) and c["recall"] < floor
        ]
        measured = sum(1 for c in classes if isinstance(c.get("recall"), (int, float)))
        line = f"  {block['split']}: {measured}/{len(classes)} classes measured"
        if below:
            line += f", {len(below)} below the {floor:.2f} floor ({', '.join(below)})"
        else:
            line += f", every one at or above the {floor:.2f} floor"
        print(line)
    print("  Recorded beside the weights, so the Admin Panel shows what this model scored as")
    print("  well as what it needs.")


def parse_classes(spec: str) -> tuple[str, ...]:
    """`--classes` as a tuple, refusing an empty entry rather than accepting "" as a class name.

    Dropping blanks quietly is the tolerance that turns `--classes "a,,b"` into a class list
    nobody wrote; an empty name is a mistake in the command, and the export check - which would
    report the empty string as a class the export does not declare - is the wrong place to find
    that out.
    """
    parts = [part.strip() for part in spec.split(",")]
    if not parts or any(not part for part in parts):
        raise SystemExit(
            f"--classes needs a comma-separated list of names with no empty entries, got {spec!r}"
        )
    return tuple(parts)


def main(argv: list[str] | None = None, yolo=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--generation",
        default=DEFAULT_GENERATION.name,
        choices=sorted(generations.GENERATIONS),
        help=(
            "which generation's dataset, class list and artifact names this run is for "
            "(default %(default)s)"
        ),
    )
    # Two option strings, one destination. This flag was `--export-dir` while the tool was v2's,
    # and the documents that name a command still spell it that way; `--dataset-dir` is the honest
    # name now that the folder can be a hand-ingested dataset rather than a downloaded export.
    ap.add_argument(
        "--dataset-dir",
        "--export-dir",
        dest="dataset_dir",
        default="",
        help="the dataset to train from (default: the generation's own export directory)",
    )
    ap.add_argument(
        "--classes",
        default="",
        help=(
            "comma-separated class list the dataset must declare, replacing the generation's "
            "(for an export that gained a class after its version was generated)"
        ),
    )
    ap.add_argument(
        "--project",
        default="",
        help="the Roboflow project (default: the generation's own)",
    )
    ap.add_argument("--download", action="store_true", help="fetch the version's export first")
    ap.add_argument(
        "--version",
        type=int,
        default=0,
        metavar="N",
        help="the Roboflow version to export, and to name as the provenance of these weights",
    )
    ap.add_argument("--format", default=EXPORT_FORMAT)
    ap.add_argument("--run-project", default=str(DEFAULT_RUN_PROJECT))
    ap.add_argument(
        "--name",
        default="",
        help="--install under models/<NAME>.pt instead of the generation's own name - for keeping "
        "two weights of one generation side by side (e.g. scanncart-grocery-v2-letterbox)",
    )
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    ap.add_argument("--run-dir", default="", help="install from this run instead of training")
    ap.add_argument("--yes", action="store_true", help="actually train")
    ap.add_argument(
        "--val",
        action="store_true",
        help=(
            f"validate on --split, check per-class recall against {RECALL_FLOOR}, "
            "and record it for --install"
        ),
    )
    ap.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=VALIDATION_SPLITS,
        help=f"the split --val measures (default {DEFAULT_SPLIT}, the acceptance split)",
    )
    ap.add_argument(
        "--no-per-distance",
        action="store_true",
        help=(
            "skip the per-distance breakdown (three extra validation passes: one per "
            "distance, so a class that only scores well up close cannot pass as good at far)"
        ),
    )
    ap.add_argument(
        "--manifest",
        default="",
        help=(
            "the dataset manifest to read each image's distance from (a tag a YOLO export loses); "
            "naming one answers first, otherwise the set's own merge_report.json is read and this "
            "defaults to the generation's - a generation with no distance axis has neither"
        ),
    )
    ap.add_argument("--install", action="store_true", help="install best.pt under the generation's name, with the record --val wrote")
    ap.add_argument("--force", action="store_true", help="allow --install to replace an existing weight")
    # The machine's share, in the same flags the other dataset tools use - see resources.py.
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument(
        "--batch",
        type=int,
        default=BATCH,
        help="the ceiling: the VRAM share may lower it, never raise it",
    )
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument(
        "--degrees",
        type=float,
        default=DEGREES,
        help="rotation augmentation, +-degrees (MODEL_TRAINING.md 4: 15)",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=SCALE,
        help="scale-jitter augmentation, +-fraction of the image (4's table uses ultralytics' 0.5)",
    )
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1 ...")
    ap.add_argument(
        "--workers", type=int, default=0, help="dataloader processes (0 = derived from the CPU budget)"
    )
    ap.add_argument(
        "--max-use-percent", type=int, default=resources.USE_PERCENT, help="CPU + RAM ceiling"
    )
    ap.add_argument(
        "--max-vram-percent", type=int, default=resources.VRAM_PERCENT, help="VRAM ceiling"
    )
    ap.add_argument("--disk-reserve-gb", type=float, default=resources.DISK_RESERVE_GB)
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON and exit")
    args = ap.parse_args(argv)

    # Named once: it decides whether this command *acts* on the dataset (train, validate) or only
    # describes what it would do, and the doctor gate below belongs on the first of those.
    will_run = args.yes or args.val

    generation = generations.get(args.generation)
    if args.classes:
        # `replace()` on the frozen spec rather than a second code path: everything downstream
        # reads it - the export check, the record, the weight's name - so an override that reached
        # only the check would leave the record claiming a class list nothing was trained for.
        generation = replace(generation, classes=parse_classes(args.classes))

    export_dir = (
        Path(args.dataset_dir).expanduser() if args.dataset_dir else generation.export_dir
    )
    run_project = Path(args.run_project).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    # The Roboflow project this generation's versions come from - the download URL and the record's
    # `source` both need it, and the spec is where that fact is already written down.
    project = args.project or generation.roboflow_project
    vname = val_name(generation)

    if args.download:
        if not args.version:
            raise SystemExit("--download needs --version <n> (the number generate_version.py reported)")
        if read_export_names(export_dir) is not None:
            print(f"export: {export_dir} (already downloaded - delete it to fetch it again)")
        else:
            print(f"downloading {project} version {args.version} as {args.format}")
            download_export(
                project,
                args.version,
                export_dir,
                load_key(project),
                fmt=args.format,
                generation_name=generation.name,
            )

    print(
        f"generation: {generation.name} - {len(generation.classes)} classes, "
        f"resize_mode {generation.resize_mode}"
    )
    print(f"export: {export_dir}")
    missing_export = not export_dir.is_dir()
    splits, problems = check_export(export_dir, generation)
    summary: dict = {
        "generation": generation.name,
        "export": str(export_dir),
        "problems": problems,
    }

    if problems:
        print()
        for p in problems:
            print(f"PROBLEM: {p}")
        if args.json:
            print(json.dumps(summary, indent=1))
        print()
        if missing_export:
            print("Download the version's export (Export -> YOLOv11 PyTorch), unzip it, and point")
            print("--export-dir at the *unzipped folder* - not at the .zip.")
        else:
            print("Nothing was trained. Fix the problems above, then re-run.")
            print("The class-list check is the one to take seriously: it trains a model whose")
            print("boxes come back under the wrong label, with no error at all. Fix the project's")
            print("classes and regenerate the version (MODEL_TRAINING.md 8.1).")
        return 2

    if will_run:
        # The human pass first: it is the cheapest question here and the one whose failure is most
        # expensive to discover late - a box a weight drew and nobody read, in `valid` or `test`,
        # makes the run's own numbers a measurement of the annotator, and finding that out after the
        # GPU run costs the run. Printed rather than silent on the way *past* it too, because "the
        # gate was asked and answered clean" is a fact about the set worth seeing once per run.
        ok, gate_lines = pass_gate(export_dir)
        summary["pass_gate"] = {"ok": ok, "lines": gate_lines}
        print()
        for line in gate_lines:
            print(f"  {line}")
        if not ok:
            print()
            print("Nothing was trained: the human pass is not finished for this set.")
            print(
                "  `make human-pass HUMAN_PASS_ARGS=--status` prints what is left in one line; work"
                " it in the annotator, then rebuild the set with build_dataset.py - the counts in"
                " the report are the ones from build time, so a finished pass still needs the"
                " rebuild before this gate clears."
            )
            if args.json:
                print(json.dumps(summary, indent=1))
            return 2

        # The doctor, before anything is trained or measured, because the failures it looks for are
        # the ones that train happily and measure the wrong thing: a class list in the wrong order,
        # a label row nothing can read, a frame drawn under a class other than the one it was
        # staged as, a test frame that duplicates a train one. Gate rather than
        # advice, so a run cannot be measured on a set nobody checked - `make doctor` is the same
        # check on its own, and `accept_v2.py` refuses on the same verdict before it measures.
        #
        # Imported here, not at module scope: `dataset_doctor` imports *this* module (and
        # `build_dataset`), so a module-level import would be a cycle - and this file is imported by
        # tests and by `accept_v2` that must not pay for the doctor's dataset reads.
        import dataset_doctor

        print()
        report = dataset_doctor.check_and_report(export_dir, generation, exports=(splits, problems))
        summary["doctor"] = report
        if report["errors"]:
            print()
            print("Nothing was trained: the set has to pass the doctor first.")
            print(
                "  `make doctor` runs that check on its own; a merged set is rebuilt with"
                " build_dataset.py, which is also what drops duplicate test frames."
            )
            if args.json:
                print(json.dumps(summary, indent=1))
            return 2

    data_yaml = write_data_yaml(export_dir, splits)
    run = resolve_run(run_project, generation, args.run_dir, training=args.yes)

    # Ultralytics is imported here, not at module level, so this file can be imported and tested
    # without torch - the same shape as app/hardware.py's lazy torch import and app/models.py's
    # "one directory read, no torch".
    if will_run:
        yolo = yolo or ultralytics_yolo()

    # `--device auto` is resolved before the command is printed, because a printed command that
    # does not name the device the run will use is one nobody can rerun. That spends the torch
    # import on a dry run with the default device; an explicit `--device cpu|0` still costs
    # nothing, and it is the import a run needs anyway.
    device = resources.resolve_device(args.device)

    # The machine's share. `apply()` comes first and unconditionally, because `measure()` reads
    # RAM but *not* the card - the VRAM totals are filled in by `apply()` - so asking for a batch
    # size before it would take `batch_size()`'s no-VRAM fallback and quietly ignore the ceiling.
    # It costs the torch import on a CPU-only dry run too, which is why the note below says so.
    #
    # The VRAM ceiling is deliberately not enforced by the allocator here (`--hard-vram-cap`
    # exists in audit_v2 with the measurement behind it): it fights ultralytics and turns a
    # working batch into an OOM, and the batch derived below is what keeps the card's other users
    # safe instead - which bounds the *batch* rather than the card. That distinction is worth
    # stating with the measurement: training yolo11s at 640, batch 8, holds about 5.9 GB of this
    # machine's 8.6 GB card, because optimizer state and activations are not the ~0.6 GB a forward
    # pass costs. The share is a policy for sizing the batch, not a wall; `--batch 4` is the lever
    # when something else needs the card.
    budget = resources.measure(args.max_use_percent, args.disk_reserve_gb, args.max_vram_percent)
    budget = resources.apply(budget, device)
    batch = min(args.batch, budget.batch_size(args.imgsz))
    if batch < args.batch:
        budget.notes.append(
            f"batch {args.batch} -> {batch}: the VRAM share is {budget.vram_cap_gb:.1f} GB "
            "(raise --max-vram-percent for more)"
        )
    hyper = Hyper(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        patience=args.patience,
        base_model=args.base_model,
        device=device,
        workers=args.workers or derive_workers(budget.cpu_threads),
        degrees=args.degrees,
        scale=args.scale,
    )

    print()
    print(f"data.yaml: {data_yaml}")
    print()
    print("this run:")
    print(f"  {command_line(data_yaml, run_project, generation.run_name, hyper, device)}")
    print(
        "  augmentation: "
        + ", ".join(f"{key}={value:g}" for key, value in augmentation_kwargs(hyper).items())
        + "   (MODEL_TRAINING.md 4)"
    )
    if will_run:
        print()
        print(budget.describe())
    summary["data_yaml"] = str(data_yaml)
    summary["run_dir"] = str(run)
    summary["hyper"] = {
        "epochs": hyper.epochs,
        "imgsz": hyper.imgsz,
        "batch": hyper.batch,
        "patience": hyper.patience,
        "base_model": hyper.base_model,
        "device": device,
        "workers": hyper.workers,
        **augmentation_kwargs(hyper),
    }

    # What the *run* trained with, where there is a run to ask. `--val` and `--install` are
    # separate commands in the documented sequence, so this command's own `--imgsz`/`--degrees`
    # are a fallback rather than a fact - and the difference only shows up as a number nobody
    # can account for (measured at a size the run never used, or a record claiming an
    # augmentation that was not applied). Ultralytics' `args.yaml` is the fact.
    trained = run_args(run)
    trained_imgsz = int(trained_value(trained, "imgsz", hyper.imgsz))
    augmentation = {
        key: trained_value(trained, key, augmentation_kwargs(hyper)[key])
        for key in AUGMENTATION_KEYS
    }

    if not (args.yes or args.install or args.val):
        if args.json:
            print(json.dumps(summary, indent=1))
        print()
        print("Nothing ran. Re-run with --yes to train, --val to validate, or --install to install.")
        return 0

    if args.yes:
        # The free-space check needs a path that exists, and on a first run the run project does
        # not - `shutil.disk_usage` raises FileNotFoundError on a missing directory, which is how
        # the very first training run on this machine died a second before it started. Creating it
        # here is the same write ultralytics would make a moment later, and it means the reserve is
        # checked against the filesystem that will actually receive the checkpoints.
        run_project.mkdir(parents=True, exist_ok=True)
        free_gb = budget.check_disk(str(run_project))
        print()
        print(
            f"training {hyper.base_model} for up to {hyper.epochs} epochs "
            f"(patience {hyper.patience}, batch {hyper.batch}) -> {run}"
        )
        print(f"  {free_gb:.1f} GB free at {run_project} (reserve {budget.disk_reserve_gb:.0f} GB)")
        model = yolo(hyper.base_model)
        model.train(**train_kwargs(data_yaml, run_project, generation.run_name, hyper, device))

    best = find_best(run)
    if best is None:
        raise SystemExit(f"no weights/best.pt under {run} - the run did not complete")

    results_csv = run / "results.csv"
    rows = read_results(results_csv) if results_csv.is_file() else []
    if rows:
        passed, failed = verdict(rows)
        top = best_epoch(rows)
        print()
        # The epoch column is 1-based (see best_epoch) - printed as recorded.
        print(f"final epoch {int(final_metrics(rows).get('epoch', 0))}: " + ", ".join(passed + failed))
        if top:
            print(f"best {top[1]:.3f} at epoch {top[0]}")
        print()
        for p in passed:
            print(f"  [.ok.] {p}")
        for f in failed:
            print(f"  [WARN] {f}")
        if failed:
            print()
            print("  A miss here is a data problem before it is a hyperparameter problem:")
            print("  add images for the failing cells rather than retuning (MODEL_TRAINING.md 6).")
        summary["metrics"] = final_metrics(rows)
    print()
    print(f"best checkpoint: {best}")
    summary["best"] = str(best)

    if args.val:
        print()
        print(f"validating {best} on the {args.split} split (floor {RECALL_FLOOR:.2f})")
        if trained_imgsz != hyper.imgsz:
            # Named rather than silent: the run trained at one size and this command was built
            # for another (`--imgsz`), and the pass follows the run.
            print(f"  note: measured at {trained_imgsz}, the size this run trained at")
        metrics = validate(
            best, data_yaml, args.split, run_project, vname, yolo=yolo, imgsz=trained_imgsz
        )
        rows = per_class_recall(metrics)
        passed, failed, unmeasured, values = recall_report(rows)
        aggregates = aggregate_metrics(metrics)
        summary["split"] = args.split
        summary["split_metrics"] = aggregates
        summary["per_class_recall"] = values
        summary["classes_below_floor"] = [line.split()[0] for line in failed]
        # The per-distance passes run before anything is written, because the breakdown belongs
        # *inside* the block it qualifies: written the other way round, a failure or an
        # interrupt halfway through would leave a stored overall number whose `far` rows
        # describe an older image set - the kind of mixture no reader could detect.
        per_distance: list[dict] = []
        distance_notes: list[str] = []
        # `None` means this generation has no distance axis - v1 predates the tags - which is not
        # the same statement as a manifest that went missing. The printout below says which.
        manifest = Path(args.manifest).expanduser() if args.manifest else generation.manifest
        if manifest is None:
            print()
            print(f"  note: {generation.name} has no distance axis - its export carries no distance")
            print("        tags - so there is no per-distance breakdown to run. The per-class table")
            print("        above is the whole readout for this generation.")
        elif not args.no_per_distance:
            distances, distance_source = dataset_distances(
                export_dir, manifest, explicit=bool(args.manifest)
            )
            print()
            print(f"splitting the {args.split} split by distance for three more passes")
            if distance_source:
                print(f"  distances: {len(distances)} frame(s) {distance_source}")
            per_distance, distance_notes = distance_breakdown(
                best,
                export_dir,
                splits,
                args.split,
                run_project,
                distances,
                yolo=yolo,
                out_dir=best.parent.parent / DISTANCE_DIR_NAME,
                val=vname,
            )
            for note in distance_notes:
                print(f"  note: {note}")
            summary["per_distance"] = {
                block["distance"]: {
                    cell["name"]: cell["recall"] for cell in block["per_class"]
                }
                for block in per_distance
            }

        # Written down as well as printed: `--install` is usually a separate command, so the
        # numbers have to outlive this process to reach the record the panel reads.
        block = validation_record(
            rows, args.split, aggregates, RECALL_FLOOR, weights_sha256(best)
        )
        block["per_distance"] = per_distance
        measured_file = val_metrics_path(best)
        write_val_metrics(measured_file, block)
        summary["validation"] = [block]

        print()
        print(f"the {args.split} split:")
        for label, value in aggregates.items():
            target = TARGETS.get(f"metrics/{label}(B)")
            mark = f"  (target >= {target:.2f})" if target is not None else ""
            print(f"  {label:10} {value:.3f}{mark}")
        print()
        for line in passed:
            print(f"  [.ok.] {line}")
        for line in failed:
            print(f"  [WARN] {line}")
        for line in unmeasured:
            print(f"  [SKIP] {line}")
        if failed:
            print()
            print(f"  {len(failed)} class(es) below the {RECALL_FLOOR:.2f} floor. That is a data problem")
            print("  before it is a hyperparameter problem: add images for those cells rather than")
            print("  retuning (MODEL_TRAINING.md 6).")
        if unmeasured:
            print()
            print("  A skipped class was never measured: the split holds no instances of it, so the")
            print("  floor cannot be claimed either way for it (MODEL_TRAINING.md 8.3).")
        # The comparison the per-class list cannot make. `test` mixes every distance, so a
        # class whose frames happen to be mostly `close` reads as healthy while being unable to
        # find the item at `far` - which is the cell this whole dataset exists to fix.
        if per_distance:
            print()
            print(f"the {args.split} split by distance (floor {RECALL_FLOOR:.2f}):")
            for line in distance_grid(rows, per_distance):
                print(f"  {line}")
            print()
            print("  ! = below the floor   ·   - = no instances of that class at that distance")
            misses = distance_misses(per_distance)
            if misses:
                print()
                print(f"  below the floor at a distance: " + " · ".join(misses))
            else:
                print()
                print(
                    "  no class is below the floor at any distance - including the ones the"
                    " mean over all distances would have hidden."
                )
        print()
        print(f"  recorded in {measured_file} - --install carries it into the weights' record.")

    if args.install:
        # Read back rather than threaded through from above: `--val` and `--install` are
        # separate commands in the documented sequence, so the file is the only path that
        # works for both, and having one path means the same one is exercised either way.
        validation = load_val_metrics(val_metrics_path(best), weights_sha256(best))
        # The export's names, not the generation's list: this records what the weights *are*, and
        # `check_export` has already refused an export whose names disagreed with the list - so the
        # two agreeing is a precondition of reaching here, and a recorded set that differs means
        # something changed after this run (MODEL_TRAINING.md 8.1).
        record = weight_record(
            generation,
            args.version,
            project,
            validation=validation,
            class_names=read_export_names(export_dir),
            imgsz=trained_imgsz,
            augmentation=augmentation,
            resize_mode=read_export_resize_mode(export_dir),
        )
        target = install(
            best,
            models_dir,
            name=f"{args.name}.pt" if args.name else generation.weight_name,
            force=args.force,
            record=record,
        )
        print(f"installed: {target}")
        summary["installed"] = str(target)
        summary["record"] = record
        print(f"recorded:  {target.with_suffix('.json')}")
        print_reminders(record["resize_mode"], generation)
        print_validation_note(validation)
    else:
        print()
        # `--generation` is in both hints because it is not implied by `--run-dir`: the val run's
        # name and the weight's name both come from it, so a command that left it out would
        # measure or install under the wrong generation's names.
        if not args.val:
            print(f"Check the acceptance split with: --val --generation {generation.name} --run-dir {run}")
        print(f"Install it with: --install --generation {generation.name} --run-dir {run}")

    if args.json:
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
