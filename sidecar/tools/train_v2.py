#!/usr/bin/env python
"""Train the v2 weights locally, then install them where the picker finds them.

Three one-shot decisions meet here, and each one has already bitten this project:

1. **The export's class list.** A trained model's output indices are whatever order the
   dataset declared. If the export's `names` and the roster in `label_classes.py` disagree,
   every box comes back under the wrong label - with no error, and with plausible-looking
   confidences. So the export is checked against the roster *before* anything is trained.
2. **The training run.** Hyperparameters are `MODEL_TRAINING.md` 6's, in code, so the run
   that produced a shipped model is reproducible from this file rather than from memory.
3. **The drop-in.** `models/scanncart-grocery-v2.pt` is the name the picker lists and the
   validator accepts. The previous generation's weights were never installed at all - there
   is no `sidecar/models/` in this checkout - so the step that turns a good `best.pt` into a
   selectable model is the one worth scripting. `install()` also writes a small record beside
   it (`models/scanncart-grocery-v2.json`) carrying the `resize_mode` these weights require,
   because nothing else knows it: the checkpoint stores the training run, not the dataset
   geometry, and `auto` resolves to the *wrong* geometry for a locally trained `.pt`. The
   Admin Panel's Model field reads that record and flags a mismatch.

`--val` is the fourth step, and it is here because the two numbers a training log carries are
**means**. A run can clear both while failing one whole class - which is the failure this dataset
exists to fix, since the mean over 8 classes is what hides `century-tuna` at `far` between a
comfortable `milo` at `close` and a comfortable `sardines` at `mid`. The pass reports recall per
class against 6's 0.85 floor, so the verdict names the class instead of the average, and it reads
the `test` split by default - the one `plan_split.py --holdout-session` makes a capture session
the run never saw. It also **writes those numbers down** (`val_metrics.json`, in the run) so
`--install` can carry them into the weights' record, and the Admin Panel shows what the model
scored beside what it needs - read from the weights rather than remembered from a terminal.

Deliberately **not** part of this: generating the version (that is `generate_version.py`, and it
must happen first) and choosing `resize_mode`. The latter is a settings field, and for these
weights it must be `stretch` - see `print_reminders()`.

The one string here that could not be checked offline is `EXPORT_FORMAT` (the API's identifier
for the version page's "YOLOv11 PyTorch"): there is no generated version to ask. A wrong value
comes back as an error body rather than a bad download, and `--format` retries it without an edit.

    # fetch the version's export (needs --version, the number generate_version.py reported)
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_v2.py --download --version 2

    # check the export, print the exact training command, change nothing
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_v2.py

    # train (writes to the dataset workspace, not into the repo)
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_v2.py --yes

    # install the best checkpoint from that run as models/scanncart-grocery-v2.pt
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_v2.py --install

    # the acceptance number: per-class recall on the test split against 6's 0.85 floor,
    # then the same recall split by distance (three extra passes - `--no-per-distance` skips it)
    sidecar/.venv/Scripts/python.exe sidecar/tools/train_v2.py --val
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
from pathlib import Path

import httpx

# Two different things are called `WORKSPACE` in these tools, and importing both unaliased
# binds the wrong one: `label_classes.WORKSPACE` is the Roboflow account namespace
# (`yusri-caloyloy`), `workspace.WORKSPACE` is the directory the dataset tools read and write.
# Aliased rather than relied on by import order - the collision put a Windows path into the
# export URL the first time this ran live.
from generate_version import REQUIRED_RESIZE_MODE
from label_classes import SLUG_TO_CLASS, distance_tokens_in, load_key
from label_classes import WORKSPACE as ROBOFLOW_WORKSPACE
from workspace import DEFAULT_OUT, SIDECAR_ROOT
from workspace import WORKSPACE as DATASET_ROOT

# ---------------------------------------------------------------------------
# The generation. Changing any of these names *is* changing the generation -
# they are read by the Admin Panel's picker, `settings_store.is_custom_model`
# and MODEL_TRAINING.md 8.2, which all have to agree.
# ---------------------------------------------------------------------------
GENERATION = "v2"
WEIGHT_NAME = f"scanncart-grocery-{GENERATION}.pt"
BASE_MODEL = "yolo11s.pt"
RUN_NAME = f"scanncart-grocery-{GENERATION}"

# MODEL_TRAINING.md 6. `s` rather than `n` because with 8 classes and a few thousand
# images the larger backbone costs roughly the same wall clock on a 4060 and is
# distinctly better on small and occluded items - which is what the `far` cells are.
EPOCHS = 100
IMGSZ = 640
BATCH = 16
PATIENCE = 25

# MODEL_TRAINING.md 6's acceptance table, minus the per-class rule. Only the two
# aggregate numbers can be read out of a training run's own log; per-class recall
# needs a validation pass, which is deliberately left to `--val` so this tool cannot
# report a pass on an average that hides a failing class.
TARGETS: dict[str, float] = {
    "metrics/mAP50(B)": 0.90,
    "metrics/mAP50-95(B)": 0.65,
}

SPLITS = ("train", "valid", "test")
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

# Where the export and the run land. Both are gigabytes of JPEGs and checkpoints, so
# they go in the dataset workspace (gitignored) rather than beside the code.
DEFAULT_EXPORT_DIR = DATASET_ROOT / f"export-{GENERATION}"
DEFAULT_RUN_PROJECT = DATASET_ROOT / "runs"
DEFAULT_MODELS_DIR = SIDECAR_ROOT / "models"
# Where --val writes its confusion matrix and per-class table: beside the run, not in the repo.
VAL_NAME = f"{RUN_NAME}-val"
# Where --val writes the numbers themselves, in the run directory it measured - see
# `val_metrics_path()` for why they live there rather than in the repo or in `models/`.
VAL_METRICS_NAME = "val_metrics.json"

# The three distances the set is staged in, in the order they are reported
# (`clean_v2.DISTANCE_ORDER`, MODEL_TRAINING.md 8.3). **Not a class axis**: the class list is the
# same 8 names at every distance, and a box drawn on a distant sachet is the same class as one
# drawn on a near one. This exists because the per-class floor is computed over every distance
# mixed together, so a class whose test frames happen to be mostly `close` can clear 0.85 while
# being unable to find the item at `far` - the bucket this dataset exists to fix. Splitting the
# number by distance is what turns that from an argument into a measurement.
DISTANCE_ORDER = ("close", "mid", "far")
# Where the per-distance file lists and their yamls go: inside the run directory, beside the
# numbers they produced, so a surprising row can be traced back to the exact image set it came
# from instead of being a line in a log.
DISTANCE_DIR_NAME = "val-by-distance"
# The manifest `clean_v2.py` writes, which is the only place class and distance are joined to a
# filename on this machine - see `distance_map()`.
DEFAULT_MANIFEST = DEFAULT_OUT / "manifest.json"


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
    archive = dest / f"{GENERATION}-export.zip"
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
        for split in SPLITS:
            images = root / split / "images"
            if images.is_dir() and split not in found:
                found[split] = images
        if found:
            break
    return found


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_export_names(export_dir: Path) -> list[str] | None:
    """The class names the export declares, in the order it indexes them.

    Read from the export's own `data.yaml` rather than assumed, because this is the
    order the trained model's outputs will be in. A missing or unreadable yaml returns
    `None` - a defect the caller reports, not a crash.
    """
    import yaml  # PyYAML ships with ultralytics; see requirements.txt

    for candidate in sorted(export_dir.rglob("data.yaml")):
        try:
            body = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = (body or {}).get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def check_export(export_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Everything wrong with this export, as a list. Also returns the split dirs.

    Runs before training on purpose: a class-list mismatch found after an hour of GPU
    time is an hour of GPU time, and found after deployment it is every box in the app.
    """
    problems: list[str] = []
    if not export_dir.is_dir():
        return {}, [f"no export at {export_dir}"]

    splits = find_split_dirs(export_dir)
    for split in SPLITS:
        if split not in splits:
            problems.append(f"no {split}/images directory in the export")
    for split, directory in splits.items():
        n = count_images(directory)
        if n == 0:
            problems.append(f"{split}/images is empty")
        else:
            print(f"  {split:6} {n:6} images")

    names = read_export_names(export_dir)
    roster = sorted(SLUG_TO_CLASS.values())
    if names is None:
        problems.append("could not read `names` from the export's data.yaml")
    else:
        print(f"  classes {len(names)}: " + ", ".join(names))
        missing = [n for n in roster if n not in names]
        extra = [n for n in names if n not in roster]
        if missing:
            problems.append("the export has no class for: " + ", ".join(missing))
        if extra:
            problems.append(
                "the export declares classes that are not v2 classes: " + ", ".join(extra)
            )
            # The most likely cause, named, because the symptom does not point at it: a version
            # generated from a project whose class list has distances in it (one product split
            # into `close`/`mid`/`far`) trains a head with an output per product-and-distance.
            # Left as "extra classes", this reads as a project-id mix-up and sends the operator
            # to check the wrong thing - and the fix is a regenerate, not an edit here.
            tainted = {name: distance_tokens_in(name) for name in extra}
            tainted = {name: words for name, words in tainted.items() if words}
            if tainted:
                problems.append(
                    "and "
                    + ", ".join(repr(name) for name in tainted)
                    + " carry a *distance* rather than a product: distance is a tag on the image "
                    "(MODEL_TRAINING.md 8.1), so those classes split one product into three and "
                    "this would train one output per product-and-distance. Fix the project's "
                    "class list, move the annotations onto the product class, and regenerate the "
                    "version - the app's own roster is the 8 product names"
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
    for split in SPLITS:
        if split in splits:
            # Ultralytics resolves a relative entry against `path`; a bare directory
            # (not a glob) is what it expects for an image folder.
            body[split if split != "valid" else "val"] = str(splits[split].resolve())

    target = path or (export_dir / "data.scanncart.yaml")
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
    data_yaml: Path, split: str, project: Path, name: str = VAL_NAME
) -> dict:
    """The validation pass, as kwargs - one dict so the printed pass and the call agree.

    `imgsz` is pinned to the size the run trained at rather than inherited from the
    checkpoint, so the two numbers being compared describe the same model.

    `name` is a parameter because the per-distance passes are separate runs of it: they share
    one `project` directory, and without distinct names the last pass would overwrite the
    confusion matrix the overall pass drew - leaving the artifact on disk describing `far`
    while the terminal had just printed the split's numbers.
    """
    return {
        "data": str(data_yaml),
        "split": split,
        "imgsz": IMGSZ,
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
    weights: Path, data_yaml: Path, split: str, project: Path, yolo=None, name: str = VAL_NAME
):
    """Load `weights` and validate the split; returns ultralytics' metrics object.

    `yolo` is injectable for the same reason `download_export`'s `get`/`sleep` are: the pass
    needs a GPU and a dataset, and what is worth testing is what is *done* with its answer.
    """
    if yolo is None:
        yolo = ultralytics_yolo()
    return yolo(str(weights)).val(**validation_kwargs(data_yaml, split, project, name))


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


def distance_map(manifest: Path | None = None) -> dict[str, str]:
    """`{export filename: distance}` from the dataset workspace's manifest.

    The join happens on filenames because a **YOLO export carries no tags**: the distance is a
    Roboflow tag, and a version export is images, labels and a yaml. The manifest the dataset
    tooling writes is the only place on this machine where a filename is still joined to both
    its class and its distance, so it is what this reads.

    Never raises, and answers `{}` for a missing, unreadable or surprising file: a run that
    cannot find distances should lose the breakdown, not the validation. The caller says so out
    loud (`distance_breakdown` returns a note), because an absent section with no explanation
    reads as "nothing to report" when it means "not measured" - the opposite claim.
    """
    path = Path(manifest) if manifest is not None else DEFAULT_MANIFEST
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(body, list):
        return {}
    out: dict[str, str] = {}
    for entry in body:
        if not isinstance(entry, dict):
            continue
        name = entry.get("new_name")
        distance = entry.get("distance")
        if isinstance(name, str) and name and distance in DISTANCE_ORDER:
            out[name] = str(distance)
    return out


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
            f"no distances for the {split} split: {len(unknown)} image(s) and none matched"
            " the manifest - the breakdown is skipped, not reported as clean"
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
            weights, yaml_path, split, project, yolo=yolo, name=f"{VAL_NAME}-{distance}"
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


def train_kwargs(data_yaml: Path, project: Path) -> dict:
    """The training run, as kwargs. One dict so the printed command and the call that
    runs cannot drift apart."""
    return {
        "data": str(data_yaml),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "patience": PATIENCE,
        "project": str(project),
        "name": RUN_NAME,
    }


def command_line(data_yaml: Path, project: Path) -> str:
    kw = train_kwargs(data_yaml, project)
    return (
        f"yolo detect train model={BASE_MODEL} data={data_yaml} epochs={kw['epochs']} "
        f"imgsz={kw['imgsz']} batch={kw['batch']} patience={kw['patience']} "
        f"project={project} name={RUN_NAME}"
    )


def run_dir(project: Path) -> Path:
    """Where ultralytics puts this run, including its `-2`, `-3` suffix on a re-run."""
    base = project / RUN_NAME
    if not base.exists():
        return base
    n = 2
    while (base.parent / f"{RUN_NAME}-{n}").exists():
        n += 1
    return base.parent / f"{RUN_NAME}-{n}"


def find_best(run: Path) -> Path | None:
    for candidate in (run / "weights" / "best.pt", run / "weights" / "last.pt"):
        if candidate.is_file():
            return candidate
    return None


def latest_run(project: Path) -> Path | None:
    """The most recently written run under `project`, or None if there is none.

    `run_dir()` answers "where would the *next* run go", which is what training needs and
    the opposite of what `--val` and a bare `--install` need: once a run has finished, that
    directory does not exist yet, so a bare `--install` looked for `<name>-2` and failed.
    Ordered by modification time because the `-10` suffix does not sort after `-9`.
    """
    runs = [
        p
        for p in project.glob(f"{RUN_NAME}*")
        if p.is_dir() and p.name != VAL_NAME and find_best(p)
    ]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def resolve_run(project: Path, explicit: str = "", training: bool = False) -> Path:
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
        return run_dir(project)
    return latest_run(project) or run_dir(project)


def weight_record(
    source_version: int = 0,
    project: str = "",
    validation: list[dict] | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """What travels with the weights, as a dict. Written beside them by `install()`.

    `resize_mode` is the field this file exists for. A locally trained `.pt` resolves
    `resize_mode: auto` to **letterbox**, while the version it was trained from was generated
    with `Stretch to 640` - so the model expects the stretched geometry and the setting
    silently gives it the letterboxed one, worst on the `far` cells where the pixels were
    already scarce. Nothing errors. There is no other place the requirement could be known
    from: the checkpoint records the training run, not the dataset geometry, and the filename
    is a convention rather than a fact.

    Derived from `generate_version.REQUIRED_RESIZE_MODE` rather than written out here, so the
    requirement and the preprocessing that produced it cannot drift apart.

    `class_names` is the export's own class list, in the order the model indexes them, and it
    is recorded for the same reason: a class list is a property of the *weights* and nothing
    else keeps it. The export that produced it is gone by the time anyone runs it, so a model
    trained from a project whose class list was split by distance - `palmolive close` /
    `palmolive mid` / `palmolive far` instead of one `palmolive` - has 24 outputs, loads without
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
    """
    record = {
        "generation": GENERATION,
        "resize_mode": REQUIRED_RESIZE_MODE,
        # Where it came from, so the panel can say which dataset produced these weights
        # rather than only what they need.
        "source": f"{project} version {source_version}" if source_version else "",
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if class_names:
        record["class_names"] = [str(name) for name in class_names]
    if validation:
        record["validation"] = [
            {k: v for k, v in block.items() if k != "weights_sha256"} for block in validation
        ]
    return record


def install(
    source: Path,
    models_dir: Path,
    name: str = WEIGHT_NAME,
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


def print_reminders(resize_mode: str | None = REQUIRED_RESIZE_MODE) -> None:
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
    print(f"  been recorded as models/{Path(WEIGHT_NAME).stem}.json. `auto` honours the record,")
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


def main(argv: list[str] | None = None, yolo=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    ap.add_argument("--project", default="snc-grocery")
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
        default=str(DEFAULT_MANIFEST),
        help="the dataset manifest to read each image's distance from (a tag a YOLO export loses)",
    )
    ap.add_argument(
        "--install",
        action="store_true",
        help=f"install best.pt as {WEIGHT_NAME}, with the record --val wrote",
    )
    ap.add_argument("--force", action="store_true", help="allow --install to replace an existing weight")
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON and exit")
    args = ap.parse_args(argv)


    export_dir = Path(args.export_dir).expanduser()
    project = Path(args.run_project).expanduser()
    models_dir = Path(args.models_dir).expanduser()

    if args.download:
        if not args.version:
            raise SystemExit("--download needs --version <n> (the number generate_version.py reported)")
        if read_export_names(export_dir) is not None:
            print(f"export: {export_dir} (already downloaded - delete it to fetch it again)")
        else:
            print(f"downloading {args.project} version {args.version} as {args.format}")
            download_export(
                args.project, args.version, export_dir, load_key(args.project), fmt=args.format
            )

    print(f"export: {export_dir}")
    missing_export = not export_dir.is_dir()
    splits, problems = check_export(export_dir)
    summary: dict = {"export": str(export_dir), "problems": problems}

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

    data_yaml = write_data_yaml(export_dir, splits)
    run = resolve_run(project, args.run_dir, training=args.yes)
    print()
    print(f"data.yaml: {data_yaml}")
    print()
    print("this run:")
    print(f"  {command_line(data_yaml, project)}")
    summary["data_yaml"] = str(data_yaml)
    summary["run_dir"] = str(run)

    if not (args.yes or args.install or args.val):
        if args.json:
            print(json.dumps(summary, indent=1))
        print()
        print("Nothing ran. Re-run with --yes to train, --val to validate, or --install to install.")
        return 0

    # Ultralytics is imported here, not at module level, so this file can be imported and
    # tested without torch - the same shape as app/hardware.py's lazy torch import and
    # app/models.py's "one directory read, no torch".
    if args.yes or args.val:
        yolo = yolo or ultralytics_yolo()

    if args.yes:
        print()
        print(f"training {BASE_MODEL} for up to {EPOCHS} epochs (patience {PATIENCE}) -> {run}")
        model = yolo(BASE_MODEL)
        model.train(**train_kwargs(data_yaml, project))

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
        metrics = validate(best, data_yaml, args.split, project, yolo=yolo)
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
        if not args.no_per_distance:
            print()
            print(f"splitting the {args.split} split by distance for three more passes")
            per_distance, distance_notes = distance_breakdown(
                best,
                export_dir,
                splits,
                args.split,
                project,
                distance_map(Path(args.manifest).expanduser()),
                yolo=yolo,
                out_dir=best.parent.parent / DISTANCE_DIR_NAME,
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
        # The export's names, not the roster's: this records what the weights *are*, and
        # `check_export` has already refused a version that disagreed with the roster - so the
        # two agreeing is a precondition of reaching here, and a recorded list that differs
        # from the roster means something changed after this run (MODEL_TRAINING.md 8.1).
        record = weight_record(
            args.version,
            args.project,
            validation=validation,
            class_names=read_export_names(export_dir),
        )
        target = install(best, models_dir, force=args.force, record=record)
        print(f"installed: {target}")
        summary["installed"] = str(target)
        summary["record"] = record
        print(f"recorded:  {target.with_suffix('.json')}")
        print_reminders(record["resize_mode"])
        print_validation_note(validation)
    else:
        print()
        if not args.val:
            print(f"Check the acceptance split with: --val --run-dir {run}")
        print(f"Install it with: --install --run-dir {run}")

    if args.json:
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
