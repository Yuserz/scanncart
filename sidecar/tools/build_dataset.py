#!/usr/bin/env python
"""Merge v1's export and v2's locally labeled frames into one 7-class training set.

WHY A MERGE
-----------
v2 is a *continuation* of v1, not a replacement: the eight classes were reduced to the seven v1
already had (Palmolive dropped), and the point of v2 is a fresh capture session with a distance
axis, better balance and hard negatives - not a new roster. So the two sets are the same seven
products, which is what makes merging them a copy rather than a relabeling - v1's project declares
them in a *different order*, and translating that is the one transformation below:

    train   v1's 1,265 frames + v2's decided frames whose split is train
    valid   v1's   223                    + v2's, split valid
    test    v1's   327                    + v2's, split test

The output is a directory in the layout `train_model.py` already reads (`<split>/images`,
`<split>/labels`, `data.yaml` with `names`), so training the merged set is:

    python sidecar/tools/build_dataset.py
    python sidecar/tools/train_model.py --dataset-dir <workspace>/merged-v2

WHAT IS ACTUALLY TRANSFORMED, AND WHY THAT IS SO LITTLE
------------------------------------------------------
**v1's class indices.** v1's `data.yaml` declares the seven names in a *different order* from
canonical (`555 sardines` is 0 there and 2 here), so its label rows are remapped by **name**:
index -> name from v1's own yaml -> canonical index. `train_model.check_export` compares class
lists by *membership*, never by order, so a mis-mapped index passes every existing guard and trains
happily on boxes labelled as the wrong product. That is why this file carries the name table
explicitly, refuses a name it does not know, and prints a 30-frame contact sheet - the sheet is the
one check that can see the result, and it is the only reason to trust this step.

**The geometry.** Every frame is written at `<size>x<size>` (640 by default) with the image
*stretched*, not fitted: both generations' recorded requirement is `stretch`, and a frame that is
already square needs no resampling at train time, so the requirement becomes a fact about the files
rather than a setting someone has to get right. YOLO labels are normalized, so stretching needs no
label change - which is exactly why this is safe to do once, here, instead of on every batch.

**v2's labels are translated by name too.** The annotator writes each row's `cls` column as a
position in *its own* list (`annotate.store.CLASS_NAMES`, derived from the same `SLUG_TO_CLASS` this
file reads), and the merged set declares a generation's order - so those are two facts, and the row
has to be translated between them exactly as v1's are. Today the two lists hold the same names in the
same order, which makes the translation the identity and the written rows byte-identical to the copy
this used to do; that is now a *result* rather than the reason, and the reason was the problem: a
build declaring any other order used to mislabel every v2 row with nothing downstream able to see it,
because the declaration still agreed with the generation it recorded (see `declared_order_problem`).

**The frames' tags are copied too.** Each v2 frame was filed under a product by `clean_v2.py`,
which recorded that folder as the manifest's `class` - the tag every other tool treats as what the
frame *is*. That fact is nowhere in the built set (the manifest lives in the staged directory, which
the set does not name), so each written frame's tag is recorded in `merge_report.json` and
`dataset_doctor.py` reads it back against the labels it will train on. A label drawn under a
neighbour's class is the one silent failure that reaches the GPU as a model which calls a product by
the wrong name: the rows are readable, the indices are in range, and every other check passes. v1's
frames carry no tag (their export records none, and a filename is a convention rather than a fact),
which the report says by not listing them rather than by writing a zero.

**Distances are copied the same way, and twice.** Each written v2 frame records the distance it was
filed under, per frame at the top level (`distances`, name -> `close`/`mid`/`far`) and summed per
split under `sources.v2.distances`. The per-frame map is what `train_model.py --val` reads its
per-distance passes from, so the grid survives the set outliving the staged directory it was built
from - the set is the artifact under test, and the manifest is elsewhere the moment it is. The
per-split mix is what `dataset_doctor.py` reads, and it is derived from the same map rather than
counted a second time, so the doctor's line and the report cannot disagree. The doctor warns when a
distance the model would train on has no frame in `test` (nothing could measure it), or when a
distance the plan follows is in no split at all (the capture gap). Both are warnings rather than
failures: a set holding only `close` frames is still trainable, it just cannot be evidence about
`far`.

**The per-side counts describe the set, not the intent.** `sources.<side>.images` counts the frames
of that side that are *in* it, per split - a test frame the dedup pass dropped is not one of them -
so the report's blocks agree with each other: the sides add up to `splits`, `tags` and `distances`
are those same frames again, and `dataset_doctor.py` refuses a report whose sides account for a
different size than the set on disk. Counting what each side *handed over* instead produced a
report that read `v2: test 83` on one line and `test close 36` on the next.

**The order itself is read, not restated.** `CANONICAL_NAMES` is `DECLARED_GENERATION.classes` -
`generations.V2`, the spec - because the set's `names` list is the one place the label rows' `cls`
column is interpreted, and a second derivation of it (this file used to do
`tuple(SLUG_TO_CLASS.values())`) is a second answer that only agrees while both expressions do. The
build writes the generation down beside it (`data.yaml`, `merge_report.json`), so a reader - a
person or `dataset_doctor.py` - can see which order the rows index rather than infer it. Both sides
of the merge are translated into that order by name, which is what makes it a choice: what
`declared_order_problem` refuses before the swap is a declaration the rows cannot be translated into,
not a declaration that is not the annotator's. Whether a source list can be translated at all is one
rule (`translation_problem`, with `annotator_class_problem` for the side the annotator owns), asked
by each side before it writes a frame and by the dry run in the same words - so v1's export and the
annotator's tree cannot be judged differently, or a run's report disagree with the build that follows
it.

HOW IT WRITES, AND WHAT THAT BUYS
---------------------------------
The build is assembled in a `.building` directory beside `--out` and swapped in at the end. The
split directories are replaced wholesale (a frame the new build does not write must not survive
inside them), anything the build does not write is left alone. Every refusal in the list below
costs nothing but the run: `--force` used to delete the three split directories *before*
merging, which made the command an operator re-runs the one that could destroy the only copy of a
two-minute build. The dry run is still the cheapest answer; this is what makes the expensive wrong
answer survivable.

**The doctor runs on that copy, before the swap** (`dataset_doctor.check_and_report`), which turns
§7's check from a command a minute later into part of the build: a merge that would fail it never
becomes the merged set, and what lands has already been read once - the same verdict `train_model`
and `accept_v2` refuse on. One build skips it, `--allow-unassigned`, whose own contract is that it
is "a dry run, never a trainable set"; the skip is printed. The check includes the duplicate scan
even though this build just dropped duplicates, because a build printing a different verdict from
`make doctor` a minute later is worse than six seconds of fingerprinting.

The two yamls are the exception to "everything is swapped in": `data.yaml` is written into the
staging copy (the doctor reads the class list off it) with the **final** paths in it, while
`data.scanncart.yaml` - what ultralytics resolves against - is written after the swap, because it
names the directory it describes and that name only exists once the staging copy is `--out`.

WHAT IT REFUSES
---------------
* A v1 class name that is not in the canonical roster (exit 2): the alternative is writing an
  index that means another product.
* A v2 frame with no decision - that is outstanding work, not an empty label. Copying it in would
  turn "nobody has looked" into "background", which is the one state confusion this project has
  already been bitten by.
* A v2 frame with no split in `splits.json` (exit 2, unless `--allow-unassigned`): a frame the
  planner never placed cannot be put in a split without inventing one, and the acceptance number
  depends on which side of the train/test line it lands.
* A merge whose own report still holds machine-only decisions in `valid`/`test` (exit 2,
  `pass_gate` over the report just written, unless `--allow-machine-only`): a weight's unread boxes
  make those two splits a measurement of the annotator, so `train_model.py --yes/--val` and
  `accept_v2.py` both refuse this set - the build is refused at the same door, and one door earlier,
  because the alternative is an unusable set replacing a usable one. Work the pass, then build again.
* A source class list that cannot be translated into this dataset's order (exit 2, in the v1/v2
  side's own problems *before* a frame is written, and reported by `--dry-run` the same way):
  `translation_problem` is one rule and one sentence for a question both sides ask, so a v1 export
  declaring a name with no position here and an annotator that can draw one are the same refusal.
  It is asked once up front rather than per row, so the verdict does not depend on which frames
  happen to be staged.
* A set whose *own* class declaration does not hold together (exit 2, `declared_order_problem` on
  the staging copy, and deliberately outside the two flags below): the `names` and the `generation`
  written into `data.yaml` must agree, and the annotator's list must be translatable into those names
  by the same rule the merge already asked (see the bullet above) - the order a measuring tool reads
  off the declaration is the order it scores against, so a set that is wrong here is one
  `audit_recall`/`spec_check`/`accept_v2` refuse after the previous set is gone.
* A merge that fails the doctor (exit 2, `dataset_doctor.check_and_report` on the staging copy):
  the set §7 would refuse a minute later is refused here instead, before it has replaced anything.
* A `--v2` that is not a staged set at all (exit 2, `v2_set_problem`): this is the failure with no
  symptom. The v2 side is read through its *manifest*, so a directory `clean_v2.py clean` never
  wrote - a typo, a set that has since been moved, a half-copied workspace - contributes zero frames
  and no error: the merged set keeps the seven right names, the doctor still says `[ok]`, and v1's
  frames are quietly the whole of it. Naming the hard-negative set here by a slip for `--extras` is
  the other shape, and is refused the same way: it is not what `--v2` means.
* A v2 side that would contribute no frame at all (exit 2, `v2_contribution_problem`): the staged
  set is real and every frame in it is skipped - not one has a decision yet, or `--allow-unassigned`
  drops the only decided ones. The build is then v1's frames alone (nothing at all, under `--no-v1`)
  under the merged set's name, which is the same artifact the bullet above refuses, reached through
  a directory that passes every other check in this file.

    sidecar/.venv/Scripts/python.exe sidecar/tools/build_dataset.py --dry-run
    sidecar/.venv/Scripts/python.exe sidecar/tools/build_dataset.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import resources  # noqa: F401  (must precede numpy/PIL work: sets thread limits at import)

from PIL import Image, ImageDraw
import numpy as np
from clean_v2 import NEAR_HAMMING, NEAR_MSE, NEGATIVE_CLS, _hamming, _mse, dhash64, grey_thumb
import generations
from label_classes import FIT_SPLITS, SPLIT_NAMES
from train_model import read_export_generation, read_export_names, write_data_yaml
from workspace import (
    ANNOTATIONS_DIRNAME,
    DATA_YAML_NAME,
    MANIFEST_NAME,
    MERGE_REPORT_NAME,
    PROVENANCE_NAME,
    SPLITS_NAME,
    WORKSPACE,
    resolve_extras,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# The generation this merged set is built for, and therefore the order every label row in it
# indexes. Taken from the spec rather than re-derived from `label_classes.SLUG_TO_CLASS`: the second
# derivation is a second answer to "what order do these labels index", which is the one question a
# dataset's `names` list has to settle - and it agreed with `generations.V2` only while both
# expressions happened to name the same seven in the same order. The set also *records* this name
# (`data.yaml`, `merge_report.json`), so `dataset_doctor` can tell what it is looking at instead of
# being told.
DECLARED_GENERATION = generations.V2
CANONICAL_NAMES: tuple[str, ...] = DECLARED_GENERATION.classes
# Both are read back off the built set before it is swapped in (`declared_order_problem`), and the
# label rows of both sides are translated into `CANONICAL_NAMES` by name - which is what lets this
# order be a choice rather than a constraint on the annotator, and what the read-back then checks.
INDEX_BY_NAME: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_NAMES)}

DEFAULT_OUT = WORKSPACE / "merged-v2"
# Where the locally labeled frames are staged, from the same rule `annotate/run.py` uses.
DEFAULT_V2 = WORKSPACE / "cleaned-v2"
SIZE = 640
# How many frames the contact sheet shows. Thirty because it is the smallest number that can
# show every class at least twice at seven classes, and because a person will actually look at
# thirty. It is a gate, not a gallery.
CONTACT_FRAMES = 30


@dataclass
class Side:
    """What one source contributed: counts, and everything that went wrong."""

    name: str
    # frame name -> the split it went into. `summarise` keeps only the names that survived into the
    # set it wrote, so a dropped duplicate is not counted as a frame this side contributed.
    placed: dict[str, str] = field(default_factory=dict)
    classes: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    background: Counter = field(default_factory=lambda: Counter())
    machine_only: Counter = field(default_factory=lambda: Counter())
    # name -> the class slug the frame was staged as (`clean_v2.py`'s manifest `class`). Only v2's
    # side has these: v1's export records no tag, so `summarise` writes the map for the frames that
    # are actually in the set and `dataset_doctor` checks each one's labels against it.
    tags: dict[str, str] = field(default_factory=dict)
    # name -> the distance the frame was shot at (the `<PRODUCT>/<DISTANCE>/` folder `clean_v2.py`
    # filed it under, the manifest's `distance`). Only v2's side has these, for the same reason as
    # `tags` - v1 predates the distance axis - and `summarise` turns them into the per-split mix
    # the doctor reads: the distance is what the whole v2 set exists for, and it is otherwise gone
    # the moment the merge is built, because the built set does not name the staged directory that
    # holds the manifest the distance came from.
    distances: dict[str, str] = field(default_factory=dict)
    # Polygon rows reduced to their bounding boxes, counted because it is the one transformation
    # here that changes the numbers in a label file rather than just their class index.
    polygons: int = 0
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def split_dirs(out: Path, split: str) -> tuple[Path, Path]:
    return out / split / "images", out / split / "labels"


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def translation_problem(
    names: list[str],
    target: list[str] | tuple[str, ...] = CANONICAL_NAMES,
    *,
    source: str,
    coverage: bool = False,
) -> str | None:
    """Why rows written against `names` cannot all be translated into `target`, or None if they can.

    One rule for a question every part of this merge asks - `build_v1` of v1's export, `build_v2`
    and `preview_v2` of the annotator's tree, `dry_run` of the same two lists, and
    `declared_order_problem` of the set's own declaration once it has been written - because a label
    row is a *position* in the list that wrote it, so "has this list got a position for every name
    that one declares" has one answer and one wording. Asked before any frame is written, which is
    why the same situation does not also have to be discovered row by row while the merge runs.

    A name the source declares and `target` has no position for is the fatal direction: that box can
    be filed under no product, so the frame it sits on cannot enter the set at all.

    `coverage` adds the other direction - a `target` class `names` can never produce. Asked of the
    annotator's tree, because that side draws the frames: a class it cannot draw is one no staged
    frame can ever be labelled as, and the set would learn it from v1 alone. Deliberately *not*
    asked of v1's export, where a class with no frames is a capture gap rather than a translation
    failure - the shape `check_export` reports as a note about the other generation, and a refusal
    here would block a merge that is not wrong.

    `source` is what the sentence calls the list that wrote the rows (`v1`, `the annotator`), so a
    reader can see which of the two moved.

    The relation itself is `generations.class_gaps`, the same owner `train_model.check_export`
    judges an export with - one implementation of "what does each list declare that the other does
    not", so the two tools cannot read the same pair of lists two ways. Only the sentences are this
    caller's, and that is deliberate: what follows from a mismatch here is about *label rows*, where
    a training run's is about the head's outputs.
    """
    gaps = generations.class_gaps(names, target)
    lines: list[str] = []
    if gaps.source_only:
        lines.append(
            f"{source} declares class(es) this dataset does not have, so its label rows cannot be "
            "translated: " + ", ".join(repr(name) for name in gaps.source_only)
        )
    if coverage and gaps.target_only:
        lines.append(
            f"this dataset declares class(es) {source} can never draw: "
            + ", ".join(repr(name) for name in gaps.target_only)
        )
    return "\n".join(lines) or None


def annotator_class_problem(target: list[str] | tuple[str, ...] = CANONICAL_NAMES) -> str | None:
    """Whether the annotator's own class list can be translated into `target`, or None if it can.

    Its rows' `cls` column is a position in `annotate.store.CLASS_NAMES`, which is a runtime fact
    rather than a file this tool owns, so the list is read here and judged by the one rule above -
    with `coverage`, because this is the side that draws the frames. `target` is `CANONICAL_NAMES`
    for the merge, asked by `build_v2` before it writes a frame and by `preview_v2` so the dry run
    refuses what a build would refuse, and the *set's own declared names* for
    `declared_order_problem`'s read-back, because that is the list the artifact carries.

    The import is local for the direction `read_v2` explains: `annotate/` is the authoring tool, and
    this script imports it, never the other way round.
    """
    from annotate.store import CLASS_NAMES

    return translation_problem(list(CLASS_NAMES), target, source="the annotator", coverage=True)


def remap_row(parts: list[str], names: list[str] | None) -> tuple[str | None, str, bool]:
    """One label row as `cls cx cy w h`, with the class translated by name.

    Returns `(row, problem, was_polygon)`: exactly one of `row`/`problem` is set.

    Two row shapes are accepted, because v1's export contains both and ultralytics accepts both:
    a box (`cls cx cy w h`) and a **polygon** (`cls x1 y1 x2 y2 ...`, which is what Roboflow
    writes when the annotation was drawn as a mask - 1,921 of v1's 2,111 rows). A polygon is
    reduced to the bounding box of its own points, which is exactly what ultralytics does at load
    time (`segments2boxes`, gated on any row having more than six fields), so this changes nothing
    about what the model sees - and it means the frame can be *looked at*: the contact sheet draws
    these rows, and a polygon written through verbatim would draw nonsense.

    A row that maps to nothing is a *problem* rather than a dropped row: silently losing a box
    changes what the model is trained on, which is the class of failure this file exists to make
    impossible.    `names` is the source's own class list - v1's export's, or the annotator's
    `CLASS_NAMES` for the v2 side - because it is the only thing that can say what those positions
    mean, and `None` (an unreadable `data.yaml`) is reported once by the caller.
    """
    try:
        index = int(float(parts[0]))
    except (IndexError, ValueError):
        return None, f"non-numeric class: {' '.join(parts)!r}", False
    name = names[index] if names and 0 <= index < len(names) else None
    if name is None:
        return None, f"class index {index} is not in the source class list", False
    target = INDEX_BY_NAME.get(name)
    if target is None:
        return (
            None,
            f"{name!r} is not in this dataset's classes ({', '.join(CANONICAL_NAMES)}) - it cannot "
            "be merged without relabelling",
            False,
        )

    coords = parts[1:]
    if len(coords) == 4:
        return f"{target} {' '.join(coords)}", "", False
    # A polygon needs whole points: a class field, then x/y pairs. Less than three of them is not
    # a polygon, and an odd coordinate count is not a point list at all - both are refused rather
    # than reduced to the one number a min/max over the wrong pairing would produce.
    if len(coords) >= 6 and len(coords) % 2 == 0:
        try:
            values = [float(v) for v in coords]
        except ValueError:
            return None, f"polygon with a non-numeric coordinate: {' '.join(parts)!r}", True
        xs, ys = values[0::2], values[1::2]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        width, height = right - left, bottom - top
        return (
            f"{target} {(left + right) / 2:.6f} {(top + bottom) / 2:.6f} {width:.6f} {height:.6f}",
            "",
            True,
        )
    return None, f"row with {len(parts)} fields is neither a box nor a polygon: {' '.join(parts)[:60]!r}", False


def remap_rows(text: str, names: list[str] | None) -> tuple[list[str], list[str], int]:
    """One label file's rows, translated. Returns `(rows, problems, polygons_reduced)`."""
    rows: list[str] = []
    problems: list[str] = []
    polygons = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row, problem, was_polygon = remap_row(stripped.split(), names)
        if problem:
            problems.append(problem)
            continue
        if was_polygon:
            polygons += 1
        if row is not None:
            rows.append(row)
    return rows, problems, polygons


def stretch_image(path: Path, size: int = SIZE) -> Image.Image:
    """The frame at `size`x`size`, stretched - which is the requirement, not a convenience.

    `Image.resize` with a square target and no `ImageOps.pad` is exactly that. Resampling happens
    before the JPEG write rather than being left to the trainer, so every frame in the set has the
    same geometry and the recorded `resize_mode` describes the files instead of a runtime choice.
    """
    with Image.open(path) as image:
        frame = image.convert("RGB") if image.mode not in ("RGB", "L") else image
        return frame.resize((size, size), Image.LANCZOS)


def write_frame(
    source: Path, rows: list[str] | None, out: Path, split: str, name: str, size: int = SIZE
) -> None:
    """One image plus its label file, in the layout `train_model.py` reads.

    `rows is None` means a source with no label file at all, which is written as empty only when
    the caller has already decided that is a deliberate background frame.
    """
    images, labels = split_dirs(out, split)
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    stretch_image(source, size).save(images / name, quality=92)
    body = "\n".join(rows or [])
    (labels / f"{Path(name).stem}.txt").write_text(body + ("\n" if body else ""), encoding="utf-8")


def build_v1(v1_dir: Path, out: Path, size: int = SIZE) -> Side:
    """Copy v1's export in, remapping its class indices by name.

    The export's own class list is checked against this dataset's order before a frame is written
    (`translation_problem`), because a row says which product it is by *position*: a name with no
    position here cannot be translated at all, and finding that one label file at a time would leave
    a half-written side behind.

    The split directories are v1's own: the merged set inherits the split each frame already had,
    which is what keeps the comparison honest - v1's numbers were measured on those frames in
    those splits, and moving them here would make the two weights' scores incomparable.
    """
    side = Side(name="v1")
    names: list[str] | None = None
    for split in SPLIT_NAMES:
        images, labels = split_dirs(v1_dir, split)
        if not images.is_dir():
            side.problems.append(f"no {split}/images in {v1_dir}")
            continue
        if names is None:
            names = read_export_names(v1_dir)
            if names is None:
                side.problems.append(
                    f"could not read `names` from {v1_dir}/data.yaml - v1's class indices cannot "
                    "be translated, so nothing was merged"
                )
                return side
            # The source list against this dataset's order, asked once here rather than per row:
            # v1's export declares its seven in another order, and a name this dataset has no
            # position for cannot be translated at all (`translation_problem`).
            problem = translation_problem(names, source="v1")
            if problem:
                side.problems.append(problem)
                return side
        for image in list_images(images):
            label_path = labels / f"{image.stem}.txt"
            if not label_path.is_file():
                # An absent label file is *not* an empty one. Roboflow writes a file for every image
                # it exports, so an absent one means this frame arrived without its annotation -
                # and writing it out empty would turn "nobody labelled this" into "deliberately
                # background", which is the one state confusion this project has already been
                # bitten by (and v1's 17 genuinely-empty files are the other state, kept as
                # background frames because that is what they are).
                side.problems.append(f"{split}/{image.name}: no label file beside it")
                continue
            rows, problems, polygons = remap_rows(label_path.read_text(encoding="utf-8"), names)
            side.polygons += polygons
            for problem in problems[:3]:
                side.problems.append(f"{split}/{image.name}: {problem}")
            if problems:
                # Not written: a frame whose boxes could not be translated would train as a
                # background frame, which is a worse outcome than a loud refusal.
                continue
            write_frame(image, rows, out, split, image.name, size)
            side.placed[image.name] = split
            if rows:
                for row in rows:
                    side.classes[split][CANONICAL_NAMES[int(row.split()[0])]] += 1
            else:
                side.background[split] += 1
    if side.polygons:
        side.notes.append(
            f"[polygons] {side.polygons} polygon row(s) reduced to their bounding boxes - the "
            "same conversion ultralytics applies at load time (segments2boxes)"
        )
    return side


def v2_set_problem(v2_dir: Path, frames: list) -> str | None:
    """Why `v2_dir` cannot be the staged set `--v2` names, given the frames the store read from it.

    The v1 side already fails closed on a wrong directory: its export layout is checked split by
    split, so a `--v1` that is not an export is three problems rather than a merge of nothing. This
    side had no equivalent, and its failure is the quiet one - v2 is read through its *manifest*, so
    a directory that is not a staged set contributes zero frames and no error at all. `build_v1`'s
    1,815 frames then merge alone, the seven names in `data.yaml` are still right, the doctor still
    prints `[ok]`, and an entire capture is missing from a set everyone believes is v2's.

    Two shapes, because they are the two ways it happens:
    * staged frames, and every one of them is the hard-negative pseudo-class - that is the set
      `--extras` exists for, most often named here by a slip for `--extras`;
    * no staged frames - never staged (`clean_v2.py clean` writes the manifest this reads, so a
      directory without one is not a set), staged and then moved or half-copied (frames are located
      through the manifest, and `LabelStore.frames` skips a name it cannot find), or a downloaded
      export, which is what `--v1` is for. The manifest is read here rather than inferred from the
      count, because "there is no manifest" and "its manifest names 1,383 frames, none of which are
      on disk" are different accidents with different fixes.

    Deliberately says nothing about *decisions*: a staged set nobody has labeled yet is still the
    staged set, and this function's question is whether the directory is one at all. What *building*
    from such a set means is `v2_contribution_problem`'s, where that refusal lives - a division that
    keeps this answer a fact about the directory rather than about the work left in it.
    """
    if frames:
        if all(frame.slug == NEGATIVE_CLS for frame in frames):
            return (
                f"{v2_dir} holds {len(frames)} hard-negative frame(s) and nothing else - "
                f"`{NEGATIVE_CLS}` is the pseudo-class `clean_v2.py clean --negatives` stages, so "
                "this is the set `--extras` reads, not the staged product set. Merging it as v2 "
                "would leave the actual capture out of the merged set with nothing saying so: name "
                "it with `--extras` instead"
            )
        return None

    manifest = v2_dir / MANIFEST_NAME
    if not manifest.is_file():
        # The one that most often *looks* fine: a downloaded export has `train/images`, `valid/`,
        # `test/` and a `data.yaml` full of the right names, so the mistake reads as a staged set
        # until it is asked for a manifest.
        export = (v2_dir / DATA_YAML_NAME).is_file() or (v2_dir / "train" / "images").is_dir()
        return (
            f"{v2_dir} has no manifest.json, so it is not a set `clean_v2.py clean` staged"
            + (" (it looks like a downloaded export - that is what `--v1` is for)" if export else "")
            + ". Nothing from it was merged, and the rest of this build would be v1's frames "
            "alone under the merged set's name - stage it with `clean_v2.py clean`, or point "
            "`--v2` at the staged set"
        )
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        entries = None
    if not isinstance(entries, list):
        return (
            f"the manifest at {manifest} could not be read, so nothing from {v2_dir} was merged "
            "and the rest of this build would be v1's frames alone under the merged set's name - "
            "re-stage the set (`clean_v2.py clean`) rather than building from it"
        )
    named = len([e for e in entries if isinstance(e, dict)])
    if not named:
        return (
            f"{manifest} names no frame, so nothing from {v2_dir} was merged and the rest of this "
            "build would be v1's frames alone under the merged set's name - stage this capture "
            "first (an empty set means the source folder had no images in it)"
        )
    return (
        f"{manifest} names {named} frame(s) and not one of them could be read from {v2_dir}: the "
        "staged images are missing (the frames are located through the manifest, so a workspace "
        "moved in pieces, or a set copied without its class folders, reads as this). Nothing from "
        "it was merged, and v1's frames alone under the merged set's name would not say so"
    )


def v2_contribution_problem(
    v2_dir: Path, frames: list, splits: dict[str, str], allow_unassigned: bool = False
) -> str | None:
    """Why a staged set that would contribute no frame cannot be the v2 side of a build.

    `v2_set_problem` answers "is this the staged set at all"; this answers the step after it, where
    the set is real and nothing in it reaches the output: not one frame has a decision yet, or -
    under `--allow-unassigned` - every decided frame has no split and the flag drops it. Either way
    the build is v1's frames alone (nothing at all, under `--no-v1`) under the merged set's name,
    which is the artifact a wrong `--v2` produces, with the same absence of a symptom: the seven
    names stay right, the doctor prints `[ok]`, and the merge report shows a v2 side that reads
    exactly like a capture nobody got round to.

    `allow_unassigned` is what keeps this from standing in front of the guard that already owns the
    state. Without the flag, decided frames with no split are the unassigned problem's - its
    sentence names the frames and the exact command that freezes them - so this stays silent rather
    than adding a second reading of the same refusal. The flag is precisely what turns that refusal
    into the silent "every decided frame was left out" this exists to catch, which is why the same
    state is a contribution problem once it is passed.

    Not folded into `v2_set_problem`, whose silence about decisions is deliberate and still right
    for its own question: a staged set with nothing labeled yet is a perfectly good staged set, and
    only *building* from it is not.
    """
    if not frames:
        return None  # `v2_set_problem` already owns a set with no frames in it
    placed = [
        frame for frame in frames if frame.state != "unlabeled" and splits.get(frame.name) in SPLIT_NAMES
    ]
    if placed:
        return None
    undecided = sum(1 for frame in frames if frame.state == "unlabeled")
    if undecided == len(frames):
        # "the v2 side" rather than the directory: `frames` is `--v2`'s set plus every `--extras`
        # set, and the hard negatives live in a directory of their own - attributing all of them to
        # `--v2` would send an operator looking in a folder that holds 1,383 of the 1,433.
        state = (
            f"the v2 side holds {len(frames)} staged frame(s) and not one of them has a decision "
            "yet, so every frame would be skipped"
        )
    elif not allow_unassigned:
        # The state the unassigned guard refuses on, and its sentence is the one to read: it names
        # the frames and the freeze command. Saying it twice would only add a second refusal to a
        # build that already stops.
        return None
    else:
        state = (
            f"the v2 side holds {len(frames)} staged frame(s), {undecided} of them undecided and "
            f"the other {len(frames) - undecided} with no split in {v2_dir / SPLITS_NAME}, so "
            "`--allow-unassigned` would leave every one of them out"
        )
    return (
        state + ", and the build would take nothing from v2: the set it writes would carry the "
        "merged name with v1's frames alone in it (and nothing at all, under --no-v1). That is the "
        "merge a wrong `--v2` produces, with the same silence - so label some frames first "
        "(`label_progress.py` reports what is still outstanding) and build again, or point --v2 at "
        "a set that has been labeled"
    )


def read_v2(v2_dir: Path, annotations: Path, extras: list[Path]) -> tuple[LabelStore, list, str | None]:
    """The v2 side as its store reads it: the store, its staged frames, and the reason this is not a
    staged set if it is not one (`v2_set_problem`).

    One reading shared by all three callers - `build_v2`, `preview_v2`, and `main`'s pre-flight -
    so what a build refuses and what the dry run reports cannot become two judgements about the same
    directory. The pre-flight reads it a second time rather than threading the frames through the
    build, which is a manifest read against the 1,815 v1 frames a build would otherwise rewrite
    before reaching the same verdict.

    The `annotate` import is here rather than at module scope because `annotate/` is the authoring
    tool: this script imports it, never the other way round (`sidecar/annotate/store.py` says why).
    """
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from annotate.store import LabelStore

    store = LabelStore(out=v2_dir, annotations=annotations, extras=extras)
    frames = store.frames()
    return store, frames, v2_set_problem(v2_dir, frames)


def build_v2(
    v2_dir: Path,
    annotations: Path,
    out: Path,
    extras: list[Path] | None = None,
    size: int = SIZE,
    allow_unassigned: bool = False,
) -> Side:
    """Copy v2's decided frames in, with the labels the annotator wrote translated by name against
    the set's declared order, and the split the plan set.

    Only *decided* frames enter. An outstanding frame has no label file, and copying it as an empty
    one is how "nobody has looked yet" turns into "there is nothing here" - the exact state
    confusion the annotator's three-state contract exists to prevent.

    Refuses outright when `--v2` is not the staged set it is named as (`v2_set_problem`), before a
    single frame is written: no other check in this file can see that mistake, and the merge it
    produces looks correct in every way except that it is v1's frames alone. The second way that
    merge happens is a set that *is* the staged set and contributes nothing - every frame skipped -
    which is what `v2_contribution_problem` names (see its docstring for why the unassigned guard
    is left to speak for the state it already owns).
    """
    side = Side(name="v2")
    store, frames, problem = read_v2(v2_dir, annotations, extras or [])
    if problem:
        # Nothing is merged and no "0 frames with no decision" note is left beside it: a directory
        # that is not the staged set is not a side of this merge, and the frames it does hold must
        # not be written before `main` refuses them. `build_v1` returns early for the same reason.
        side.problems.append(problem)
        return side
    # The annotator's own class list, and whether the merge can translate it at all - the same
    # question `build_v1` asks of v1's export, asked through the same rule and answered in the same
    # sentence (`translation_problem`), so one side's refusal cannot read like the other's absence.
    # Asked up front rather than per row: a class this dataset has no position for is one refusal
    # here, not one per label file while the merge runs. Rows are translated by name below, and the
    # annotator's list is read there because it is what their `cls` column indexes.
    from annotate.store import CLASS_NAMES as ANNOTATOR_ORDER

    problem = annotator_class_problem()
    if problem:
        side.problems.append(problem)
        return side

    splits = store.splits()
    unassigned: list[str] = []

    for frame in frames:
        if frame.state == "unlabeled":
            continue
        split = splits.get(frame.name)
        if split not in SPLIT_NAMES:
            unassigned.append(frame.name)
            continue
        # Translated by name, row by row, exactly as v1's are: a `cls` column is a *position* in the
        # list that wrote it, and this set declares another generation's list. The rows come from the
        # store already reduced to boxes (`Box.as_row`), so there is nothing here for the polygon
        # branch of `remap_row` to reduce - what it does is the translation, and a row that cannot be
        # translated is a problem rather than a dropped box: writing the frame without it would turn
        # a product into background, which is the failure this file exists to refuse.
        rows, row_problems, _polygons = remap_rows(
            "\n".join(box.as_row() for box in store.read_boxes(frame.name)), ANNOTATOR_ORDER
        )
        for detail in row_problems[:3]:
            side.problems.append(f"{split}/{frame.name}: {detail}")
        if row_problems:
            continue
        write_frame(frame.image, rows, out, split, frame.name, size)
        if frame.slug:
            # `frame.slug` is the manifest's `class` - the class the frame was staged as, which is
            # what its labels have to agree with. Recorded for every frame written, background
            # frames included: the tag is a fact about the frame, not about the drawing.
            side.tags[frame.name] = frame.slug
        # The other fact the staged manifest holds and the built set would otherwise lose, recorded
        # for every frame written. A hard negative staged outside the distance cells has none, and
        # `unknown` says that rather than dropping it from a map whose shape says "one per frame".
        side.distances[frame.name] = frame.distance or "unknown"
        side.placed[frame.name] = split
        if rows:
            for row in rows:
                side.classes[split][CANONICAL_NAMES[int(row.split()[0])]] += 1
        else:
            side.background[split] += 1
        if frame.machine_only:
            side.machine_only[split] += 1

    if unassigned:
        message = (
            f"{len(unassigned)} decided frame(s) have no split in {v2_dir / SPLITS_NAME}: "
            f"{', '.join(sorted(unassigned)[:5])}"
            + (" ..." if len(unassigned) > 5 else "")
        )
        if allow_unassigned:
            # A note, not a problem: this is the "how much is still unplaced" reading, and the
            # frames are left out rather than put in a split nobody planned for them.
            side.notes.append(message + " (left out of this build)")
        else:
            side.problems.append(
                message + " - freeze the assignment first: `label_progress.py --capture-splits "
                "--split-plan <plan_split.py's file>` (no API needed), or `--capture-splits` on its "
                "own while the project is still reachable; or pass --allow-unassigned to build "
                "without them"
            )
    contribution = v2_contribution_problem(v2_dir, frames, splits, allow_unassigned)
    if contribution:
        # Appended after the block above, so that when both fire the sentence that names the frames
        # and the command that fixes them comes first. Nothing was written in this state (every
        # frame was skipped), so this is a verdict on work never done rather than a half-built side.
        side.problems.append(contribution)
    return side


def fingerprint(path: Path) -> tuple[int, np.ndarray]:
    with Image.open(path) as image:
        frame = image.convert("L")
        return dhash64(frame), grey_thumb(frame)


def find_test_duplicates(out: Path) -> list[dict]:
    """Test frames that are near-duplicates of a train or valid frame, without touching them.

    The direction is deliberate. A test frame that is a near-copy of a training frame does not
    measure the model, it measures recall of a photograph it has seen - and it is the *test* side
    that has to go, because the train side is what the model learns from and because the same
    frame appearing twice in train is merely redundant. `clean_v2`'s thresholds are reused rather
    than re-chosen: they were tuned against this project's own frames, and a second pair of cut
    points would be a second opinion about what "near-duplicate" means.

    v1's splits were assigned before any of this, so a v1 test frame that duplicates a v1 train
    frame is exactly the case this catches - and it is the one that quietly inflated v1's own
    numbers.

    **Reported here and deleted by `drop_test_duplicates` below**, rather than one function that
    does both: `dataset_doctor.py` has to be able to *see* a leak in a set somebody else built
    (or in one this tool did not drop from), and a check that could only run by deleting is a check
    nobody would run on the set they are about to train on.
    """
    train_val: list[tuple[str, int, np.ndarray]] = []
    for split in FIT_SPLITS:
        for image in list_images(split_dirs(out, split)[0]):
            digest, thumb = fingerprint(image)
            train_val.append((f"{split}/{image.name}", digest, thumb))

    found: list[dict] = []
    for image in list_images(split_dirs(out, "test")[0]):
        digest, thumb = fingerprint(image)
        for other, their_digest, their_thumb in train_val:
            if _hamming(digest, their_digest) <= NEAR_HAMMING and _mse(thumb, their_thumb) < NEAR_MSE:
                found.append({"name": image.name, "duplicate_of": other})
                break
    return found


def drop_test_duplicates(out: Path, notes: list[str]) -> list[dict]:
    """`find_test_duplicates`, applied: the leaky test frames and their label files are removed.

    Re-reading the list rather than deleting inside the scan keeps one definition of "duplicate"
    for both the build and the doctor - and unlinking during the scan would have made the doctor's
    version of it impossible to write without a second copy of the rule.
    """
    dropped = find_test_duplicates(out)
    for entry in dropped:
        image = split_dirs(out, "test")[0] / entry["name"]
        if image.is_file():
            image.unlink()
        label = split_dirs(out, "test")[1] / f"{Path(entry['name']).stem}.txt"
        if label.is_file():
            label.unlink()
    if dropped:
        notes.append(
            f"[dedup] dropped {len(dropped)} test frame(s) that duplicate a train/valid frame "
            f"({', '.join(d['name'] for d in dropped[:3])}{' ...' if len(dropped) > 3 else ''})"
        )
    return dropped


def write_names_yaml(out: Path, splits: dict[str, Path], path: Path | None = None) -> Path:
    """The dataset's own `data.yaml`: the target generation's names, in its order.

    Written here rather than in `train_model.write_data_yaml` for the same reason that function
    reads `names` back off an export: the class list is a fact about the dataset, and the trainer's
    job is to check it, not to invent it. Splits are listed too, so the file is readable on its own
    even though ultralytics only ever sees the normalized `data.scanncart.yaml`. The generation
    name goes in beside the names, so the file says which order those names are in rather than
    leaving it to a reader who knows the roster by heart.

    `path` is where to write it when that is not `out` - the build writes this file into its
    staging copy (the doctor reads `data.yaml` for the class list, and the staging copy is what the
    doctor is handed) while `out` stays the directory the file's *contents* have to name, because
    the paths it lists are the ones that exist after the swap.
    """
    import yaml

    body: dict = {
        "path": str(out.resolve()),
        "nc": len(CANONICAL_NAMES),
        "names": list(CANONICAL_NAMES),
        # Which generation those names are, in order: `dataset_doctor` reads it (and the merged
        # set's merge report) to judge the labels against the generation they actually index.
        "generation": DECLARED_GENERATION.name,
        "source": "merged v1 export + v2 local labels (build_dataset.py)",
    }
    for split in SPLIT_NAMES:
        if split in splits:
            # Ultralytics' key for the validation split is `val`; the directory stays `valid`.
            body[split if split != "valid" else "val"] = str(splits[split].resolve())
    target = path or (out / DATA_YAML_NAME)
    target.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return target


def declared_order_problem(staging: Path, out: Path) -> str | None:
    """Why a tool that measures a weight against the set in `staging` would refuse it, or None.

    Every tool that measures a weight against labelled frames asks
    `train_model.require_labels_order` before it measures anything (`audit_recall`, `spec_check`,
    `clamp_probe`, `unsure_probe` and `accept_v2` in one form or another), and what that rule
    refuses is a set whose declared class list is not the generation's: a label row's class column
    is a bare *position* in that list, so a set measured against another order has every box
    attributed to whichever product sits at that position, and `check_export` cannot see it because
    it compares names by membership. This build writes that list and the rows in the same pass, from
    the same constants - so the check is on the artifact rather than on the intent: read the class
    list and the generation back off the `data.yaml` about to be swapped in, and refuse when the
    file does not hold together. A build that emitted one would have spent 2,300 frames rewriting a
    usable set into one that every tool able to say what is in it refuses - the same argument the
    doctor is run on the staging copy for, taken one door further back. It is deliberately not
    inside the `--allow-machine-only`/`--allow-unassigned` branch: those flags land a set for
    *reading*, and the order is a fact about the artifact rather than about how much of the
    annotation pass has been worked, so a set that is not worth training still has to be a set whose
    own two fields agree.

    Two things can disagree, and each needs its own sentence because the remedies are opposite:

    **The declaration and the set's own record.** `write_names_yaml` writes `generation` beside
    `names`, and a reader resolves the first to read the second - `dataset_doctor --generation auto`
    (its default) does exactly that. A set whose `names` are one generation's list while its
    `generation` names another is therefore one `make doctor` rejects a minute later, and the
    build's own doctor run cannot see it: that run is handed the generation the build *meant*,
    which is the one thing the set's record and the set's list can disagree about.

    **The declaration and the translation table.** Both sides go through `remap_row`: a source row's
    `cls` is a *position* in the list that wrote it (`v1`'s export for one side,
    `annotate.store.CLASS_NAMES` for the other), `remap_row` turns that into a product name, and the
    name into a position in the declared list. That translation is total - and so cannot refuse a row
    it should have kept, nor silently file one under a neighbour - exactly when the two lists name
    the same products, which is what `translation_problem` answers and what this clause asks here
    (via `annotator_class_problem`, against the names the artifact declares rather than against
    `CANONICAL_NAMES`). It is checked rather than assumed from "the annotator writes our order"
    because that assumption was the thing worth removing: while the two orders agree the remap is
    the identity, and the day a roster moves (a class added to the annotator's list, or a literal
    generation like `V1_CLASSES` left behind by one) a declaration that is right about its own names
    can still be a set whose rows cannot all be translated. A build declaring an order the annotator
    does *not* write in is fine now - that is what the remap is for.

    Neither remedy is in the data: both are constants disagreeing with each other, so the message
    names the two lists and the build stops before `out` is touched.
    """
    names = read_export_names(staging)
    if not names:
        return (
            f"the set built in {staging} declares no class list, so nothing says which product a "
            "label row's class position means - `write_names_yaml` is what writes it, so this is a "
            "defect in this file rather than in the data (or this is not a set this build made), "
            f"and {out} was not touched"
        )

    declared = read_export_generation(staging)
    if declared is None:
        return (
            f"the set built in {staging} names no generation its rows index (`generation:` in "
            f"{DATA_YAML_NAME}), so its class list could only be read by guessing which generation "
            "it is - `write_names_yaml` records that name, so this is a defect in this file (or "
            f"this is not a set this build made), and {out} was not touched"
        )
    found = generations.order_of(names)
    if found is None or found.name != declared:
        return (
            f"the set built in {staging} records `{declared}` but declares names that are "
            + (f"{found.name}'s order" if found is not None else "no generation's list")
            + ", so its own two fields disagree - a `--generation auto` reader (which is what "
            "`make doctor` runs) would judge these labels against the order they are not in and "
            "refuse the set. Both fields are written by `write_names_yaml` from "
            "`DECLARED_GENERATION` and `CANONICAL_NAMES`, so this is a defect in this file; "
            f"{out} was not touched.\n"
            "  declared: " + ", ".join(names) + "\n"
            f"  {declared} lists: " + ", ".join(generations.GENERATIONS[declared].classes)
        )

    # The table both sides are translated through, asked by the rule the merge itself asks of each
    # source list (`translation_problem`, via `annotator_class_problem`) - here against the names the
    # artifact actually declares rather than against `CANONICAL_NAMES`, which is the only difference
    # between this read-back and the up-front check in `build_v2`. A name that exists on only one of
    # the two lists is a class this set cannot carry, and the remedy is a roster change: the two
    # lists move together or neither does.
    table = annotator_class_problem(list(names))
    if table:
        return (
            f"the set built in {staging} declares {declared}'s class order, but the annotator's tree "
            "cannot be translated into it - a roster change rather than a labelling one, and one the "
            f"merge cannot paper over. {out} was not touched.\n" + table
        )
    return None


def contact_sheet(out: Path, path: Path, limit: int = CONTACT_FRAMES) -> dict:
    """A grid of frames with the boxes the trainer will see, for a human to look at.

    This is the gate on the one transformation nothing else can check. `check_export` compares
    class *names*, so a label file whose index was translated wrongly is a valid file describing
    the wrong product - it trains, and the only way to notice is to look. So the sheet draws each
    frame's boxes with the class name the merged dataset assigns them, and it spreads across
    splits, so a mistake in one side's mapping shows up rather than being averaged away.

    Written with PIL rather than OpenCV: this is a report, it needs no camera and no detector, and
    the tools' tests already run without a display.
    """
    picked: list[tuple[Path, Path]] = []
    per_split: dict[str, int] = {}
    take = max(1, limit // len(SPLIT_NAMES))
    for split in SPLIT_NAMES:
        images = list_images(split_dirs(out, split)[0])
        if not images:
            continue
        step = max(1, len(images) // take)
        # Capped per split, not only at the end: a step of `len // take` yields one more than `take`
        # whenever it does not divide evenly, and a `picked[:limit]` over the concatenation would
        # then spend that slack on `train` - the split whose frames are already the most familiar -
        # and could leave `test` off the sheet entirely, which is the one thing it is a gate on.
        for image in images[::step][:take]:
            picked.append((image, split_dirs(out, split)[1] / f"{image.stem}.txt"))
            per_split[split] = per_split.get(split, 0) + 1
    picked = picked[:limit]

    cell = 224
    columns = min(6, max(1, len(picked)))
    rows = max(1, (len(picked) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell, rows * (cell + 16)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(picked):
        frame = stretch_image(image, cell)
        boxes = []
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    boxes.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
        draw.rectangle(
            (
                (index % columns) * cell,
                (index // columns) * (cell + 16),
                (index % columns) * cell + cell - 1,
                (index // columns) * (cell + 16) + cell - 1,
            ),
            outline=(60, 60, 60),
        )
        sheet.paste(frame, ((index % columns) * cell, (index // columns) * (cell + 16)))
        for cls, cx, cy, w, h in boxes:
            left = (index % columns) * cell + (cx - w / 2) * cell
            top = (index // columns) * (cell + 16) + (cy - h / 2) * cell
            right = left + w * cell
            bottom = top + h * cell
            colour = (0, 220, 120) if 0 <= cls < len(CANONICAL_NAMES) else (255, 60, 60)
            draw.rectangle((left, top, right, bottom), outline=colour, width=2)
            name = CANONICAL_NAMES[cls] if 0 <= cls < len(CANONICAL_NAMES) else f"INDEX {cls}!"
            draw.text((left + 2, max(0, top - 11)), f"{cls} {name[:18]}", fill=colour)
        caption = f"{image.stem[:28]}"
        draw.text(
            ((index % columns) * cell + 2, (index // columns) * (cell + 16) + cell + 2),
            caption,
            fill=(180, 180, 180),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    # The per-split counts travel with the sheet because the spread is what makes it a gate: a
    # sheet that is entirely `train` cannot show that a class index was translated wrongly on the
    # side whose frames a person has never seen. Reported, so a build says which it actually got.
    return {"path": str(path), "frames": len(picked), "columns": columns, "per_split": per_split}


def annotation_state(annotations: Path) -> dict:
    """The state the machine-only counts were read from, so a gate can tell they are still current.

    Those counts are a measurement of one moment: the build's. A person opening the annotator
    afterwards - or `_suggest_all.py` saving a suggestion in bulk - changes the answer without
    touching the merged set, and nothing in the report would say so, which is how a set with
    unreviewed boxes in `valid`/`test` passed a gate that had already read zero. So the report
    carries the identity of the file every decision is written into (`provenance.json`, via
    `annotate/store.py`), and `accept_v2` re-reads it before believing the counts - the rule
    `--val`'s measurements follow (`weights_sha256`): a number travels with the identity of the
    thing it measured.

    Empty when there is no such file. A label tree written some other way has no machine-only
    provenance either way, and a report that cannot name what it read is one the gate refuses -
    not one it reads as "no machine-only decisions".
    """
    path = Path(annotations).expanduser() / PROVENANCE_NAME
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    return {"provenance": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def pass_gate(report: dict) -> tuple[bool, list[str]]:
    """Whether the set about to land is one a run could use, and the sentences either way.

    The rule is `accept_v2`'s own (`machine_only_gate`, over the report this build has just written
    rather than the copy already on disk - they are the same document, and the one that has not
    landed yet is the one that can still be abandoned). `train_model --yes/--val` refuses the same
    thing before it runs, which is what makes this the first of three doors rather than a fourth
    rule: a set whose `valid`/`test` hold a weight's unread boxes is one no run will train or
    measure, so building it can only replace a set that could have been used with one that cannot.

    Deliberately the **counts half only**, where the two readers ask both halves. The stamp is
    there to tell a later reader whether the counts still describe the state on disk, and a build's
    counts are its own measurement of the store it just read - asking the stamp here would refuse a
    set with *no* `provenance.json` at all, which is not a dirty set: it is the Roboflow-labeling
    route, where the labels were drawn in that project and there are no machine-made decisions to
    read. That report carries `annotations: {}`, and the run that later refuses it says "cannot
    verify" rather than "dirty" - which is the honest difference, and it belongs to that door.

    Imported inside the function for the same reason `dataset_doctor` is in `main`: `accept_v2`
    imports `train_model`, and this module is on that chain.
    """
    import accept_v2

    return accept_v2.machine_only_gate(report)


def summarise(
    out: Path,
    sides: list[Side],
    dropped: list[dict],
    machine_only: Counter,
    size: int = SIZE,
    notes: list[str] | None = None,
    annotations_state: dict | None = None,
) -> dict:
    """The report: what is in each split, and what a reader has to know before trusting it.

    Counted by re-reading the files it just wrote rather than from the counters the two sides
    kept, so the report describes the dataset on disk - the thing training will actually read -
    instead of the intent behind it.
    """
    notes = notes if notes is not None else []
    per_split: dict[str, dict] = {}
    written: set[str] = set()
    split_of: dict[str, str] = {}
    for split in SPLIT_NAMES:
        images = list_images(split_dirs(out, split)[0])
        written.update(image.name for image in images)
        split_of.update({image.name: split for image in images})
        labels = split_dirs(out, split)[1]
        classes: Counter = Counter()
        background = 0
        boxes = 0
        for image in images:
            label = labels / f"{image.stem}.txt"
            rows = [r.split() for r in label.read_text(encoding="utf-8").splitlines()] if label.is_file() else []
            rows = [r for r in rows if len(r) == 5]
            boxes += len(rows)
            for row in rows:
                index = int(float(row[0]))
                # An index outside the roster cannot come from this build (both sides are
                # validated), and it is counted under a name rather than dropped, so a hand-edited
                # label file shows up in the report instead of vanishing from it.
                name = (
                    CANONICAL_NAMES[index]
                    if 0 <= index < len(CANONICAL_NAMES)
                    else f"!out-of-range index {index}"
                )
                classes[name] += 1
            if not rows:
                background += 1
        undeclared = generations.class_gaps(classes, CANONICAL_NAMES).source_only
        unexpected = {name: classes[name] for name in undeclared}
        per_split[split] = {
            "images": len(images),
            "boxes": boxes,
            "background": background,
            "per_class": {name: classes.get(name, 0) for name in CANONICAL_NAMES},
            "unexpected_classes": unexpected,
        }
        if unexpected:
            notes.append(
                f"[!] {split} holds label rows for class(es) this dataset does not declare: "
                f"{unexpected}"
            )
    images_in_set = {
        side.name: dict(Counter(split_of[name] for name in side.placed if name in written))
        for side in sides
    }
    # The tags of the frames that are *here*: a test frame dropped as a duplicate is not in the set,
    # so its tag is not in the report either - a report describing frames the set does not hold
    # would be describing a different build, which is the one thing this file exists to rule out
    # (`dataset_doctor` reads it exactly that way). Omitted when empty, so "this build recorded no
    # tags" and "it recorded tags and none of them landed" stay different states rather than both
    # reading as an empty map.
    tags = {name: slug for side in sides for name, slug in side.tags.items() if name in written}
    # The distance of each frame that is *here*, built and filtered the same way as `tags`: a test
    # frame dropped as a duplicate is not in the set, so its distance is not in the map either.
    # This is the per-frame record `train_model.py --val` reads its three per-distance passes from.
    distances = {
        name: distance for side in sides for name, distance in side.distances.items()
        if name in written
    }
    # ...and the same fact summed per split, which is the shape the doctor's coverage question can
    # be asked of - derived from the map rather than counted again, so the two cannot disagree.
    mix: dict[str, Counter] = {split: Counter() for split in SPLIT_NAMES}
    for name, distance in distances.items():
        mix[split_of[name]][distance] += 1
    distances_by_split = {split: dict(counter) for split, counter in mix.items() if counter}
    body: dict = {
        "generation": DECLARED_GENERATION.name,
        "classes": list(CANONICAL_NAMES),
        "size": size,
        "splits": per_split,
        "sources": {
            side.name: {
                "images": images_in_set[side.name],
                "background": dict(side.background),
                "machine_only": dict(side.machine_only),
                "polygons_reduced": side.polygons,
                "notes": side.notes,
                "problems": side.problems,
                # v2's own frames' distance mix per split (`tags` carries the class, this carries
                # how far away the frame was). Empty for a side that has no distances, which is
                # what v1's frames are - they predate the axis rather than having lost it.
                "distances": distances_by_split if side.name == DECLARED_GENERATION.name else {},
            }
            for side in sides
        },
        # The acceptance gate, per split. Zero is required in valid and test: a decision a weight
        # made and nobody confirmed is not evidence about the model.
        "machine_only_by_split": dict(machine_only),
        # Cross-cutting notes from the build itself (what the dedup pass dropped), as opposed to
        # `sources`, which is per side. Kept apart so neither has to be read as the other.
        "notes": list(notes),
        "dropped_test_duplicates": dropped,
        # v1's 1,815 frames were drawn by hand in Roboflow before the annotator existed, so this
        # tool cannot verify them; only v2's can carry provenance. Said out loud, because a report
        # that prints `machine_only: {}` for v1 reads as "checked and clean" when it means "not
        # measurable from these files".
        "provenance_note": (
            "machine_only is recorded per frame by the local annotator, so it covers v2's frames "
            "only; v1's were labelled by hand in Roboflow (v1 era) and carry no provenance either way"
        ),
    }
    if tags:
        body["tags"] = tags
    if distances:
        body["distances"] = distances
    if annotations_state:
        body["annotations"] = annotations_state
    return body


def render_report(report: dict, side_lines: list[str], sheet: dict) -> str:
    lines = [
        "# Merged v2 dataset",
        "",
        f"{len(report['classes'])} classes at {report['size']}x{report['size']} (stretched), in "
        f"{report.get('generation', DECLARED_GENERATION.name)}'s order. Built by "
        "`sidecar/tools/build_dataset.py`.",
        "",
        "| Split | images | boxes | background | "
        + " | ".join(report["classes"])
        + " |",
        "|-------|-------:|------:|-----------:|"
        + "|".join(["---:"] * len(report["classes"]))
        + "|",
    ]
    for split in SPLIT_NAMES:
        row = report["splits"][split]
        lines.append(
            f"| {split} | {row['images']} | {row['boxes']} | {row['background']} | "
            + " | ".join(str(row["per_class"][name]) for name in report["classes"])
            + " |"
        )
    lines += ["", "## Where it came from", ""] + side_lines
    if report.get("notes"):
        lines += [""] + [f"- {note}" for note in report["notes"]]
    lines += ["", "## Provenance", "", f"- {report['provenance_note']}"]
    if report.get("tags"):
        lines.append(
            f"- tags: {len(report['tags'])} frame(s) carry the class they were staged as; "
            "`dataset_doctor.py` checks every one of their labels against it"
        )
    if report.get("distances"):
        lines.append(
            f"- distances: {len(report['distances'])} frame(s) carry the distance they were shot"
            " at; `train_model.py --val` reads that map for its per-distance passes, so the grid"
            " does not depend on the staged directory this set was built from"
        )
    v2_mix = ((report.get("sources") or {}).get(DECLARED_GENERATION.name) or {}).get("distances")
    if v2_mix:
        lines.append(
            "- distances of v2's frames, per split: "
            + "; ".join(
                f"{split} "
                + ", ".join(f"{name} {count}" for name, count in sorted(v2_mix[split].items()))
                for split in SPLIT_NAMES
                if v2_mix.get(split)
            )
        )
    for split, count in sorted(report["machine_only_by_split"].items()):
        lines.append(f"- machine-only decisions in `{split}`: {count}")
    if report["dropped_test_duplicates"]:
        lines += ["", "## Test frames dropped as duplicates", ""]
        lines += [
            f"- `{d['name']}` duplicates `{d['duplicate_of']}`"
            for d in report["dropped_test_duplicates"]
        ]
    lines += [
        "",
        "## Before training",
        "",
        f"- Look at `{sheet['path']}` ({sheet['frames']} frames). This is the only check that can "
        "see a mis-translated class index: `check_export` compares class names by membership, so a "
        "wrong mapping produces a valid file about the wrong product.",
        "",
    ]
    return "\n".join(lines)


def staging_dir(out: Path) -> Path:
    """Where a build is assembled: beside `out`, so the swap is a rename rather than a copy.

    A dot-prefixed sibling rather than a system temp directory (which may be another filesystem,
    and would make the swap a 2.5 GB copy), and named after the set it will become, so an operator
    who finds one after a kill knows exactly which build left it.
    """
    return out.parent / f".{out.name}.building"


def prepare_staging(staging: Path) -> None:
    """An empty directory to build into, clearing the leftovers of a build that was killed.

    Said out loud when one is found: a `.building` directory nobody can account for is gigabytes of
    JPEGs on a disk this project already watches, and "the tool prints something about it" is what
    makes it deletable rather than mysterious. Refused outright if the path is occupied by something
    that is not a directory of ours - a symlink or a file sitting there is not a week-old build.
    """
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        raise SystemExit(f"{staging} is in the way and is not a build directory - move it first")
    if staging.is_dir() and any(staging.iterdir()):
        print(f"clearing a leftover build directory: {staging} (a previous run did not finish)")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)


def swap_into_place(staging: Path, out: Path) -> None:
    """Move a finished build into `out`, nothing deleted until every frame has landed.

    Two passes through the staging directory. First each entry the build produced is *moved aside*
    into a sibling `.previous` - the old `train/`, the old `MERGE_REPORT.md` - and only then is the
    new one renamed in, and only then is the `.previous` directory dropped. The older code deleted
    the three split directories before it merged anything, so a refusal half a minute in left the
    last good build destroyed and nothing in its place; here `out` is untouched until the new build
    is actually on disk, and the one window that can leave it mixed - a rename refused by something
    holding a file open - is caught and named rather than left to be discovered.

    Entry by entry rather than renaming the whole directory, because everything the build does not
    write is promised to be left alone: `out` may hold an operator's own file, or a leftover this
    tool knows nothing about, and a wholesale rename would carry it into `.previous` and delete it.

    Replacing a split directory is implied rather than merged into (`rmtree` of the old, rename of
    the new): a frame dropped by the dedup pass, or one that stopped being *decided*, must not
    survive inside the new `train/` from the previous build. `.previous` exists only for the length
    of the swap, so a `.previous` found on disk is that crash, holding the entries that had already
    been moved aside.
    """
    previous = out.parent / f".{out.name}.previous"
    shutil.rmtree(previous, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    try:
        for entry in sorted(staging.iterdir()):
            target = out / entry.name
            if target.exists() or target.is_symlink():
                previous.mkdir(parents=True, exist_ok=True)
                os.replace(target, previous / entry.name)
        for entry in sorted(staging.iterdir()):
            os.replace(entry, out / entry.name)
    except OSError as error:
        # The one failure this swap cannot rule out, and the worst one to leave unexplained: a
        # rename refused (a handle on a JPEG, in Explorer or a viewer) leaves `out` half new and
        # half old. Named, with the old entries' location, because a mixed build that takes a
        # minute to find is worse than a sentence - re-running rebuilds and swaps over it.
        raise SystemExit(
            f"the swap failed ({error}): {out} may now hold a mix of the two builds. The entries "
            f"it had before are in {previous}; re-running the build swaps over whatever is there"
        )
    shutil.rmtree(previous, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--v1", default=str(generations.V1.export_dir), help="v1's export (the hand-downloaded one)"
    )
    ap.add_argument(
        "--v2",
        default=str(DEFAULT_V2),
        help="the staged v2 set (`clean_v2.py clean`'s output - refused if it is not one)",
    )
    ap.add_argument(
        "--annotations",
        default="",
        help=f"where the v2 labels live (default: <v2>/../{ANNOTATIONS_DIRNAME}, as `make annotate` uses)",
    )
    ap.add_argument(
        "--extras",
        action="append",
        default=[],
        metavar="DIR",
        help="another staged set to merge (repeatable). Adds to the staged hard negatives beside "
        "--v2, which are their own set with their own manifest and are the frames that teach the "
        "model what the products are not - pass --no-extras to mean only what you name",
    )
    ap.add_argument(
        "--no-extras",
        action="store_true",
        help="merge only the sets named with --extras, not the staged hard negatives beside --v2",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="where the merged set is written")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--contact-sheet", default="", help="where to write the sheet (default: <out>/contact_sheet.jpg)")
    ap.add_argument("--contact-frames", type=int, default=CONTACT_FRAMES)
    ap.add_argument("--no-v1", action="store_true", help="build v2's frames alone, for a dry run")
    ap.add_argument(
        "--allow-unassigned",
        action="store_true",
        help="count frames with no split instead of failing - a dry run, never a trainable set",
    )
    ap.add_argument(
        "--allow-machine-only",
        action="store_true",
        help="land a set whose own report still holds machine-only decisions in valid/test - a set "
        "no run will train or measure, for reading the merge rather than using it",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild over an existing merged set (the new build is assembled beside it and swapped "
        "in only if it finishes)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report what would be merged and stop")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    v1_dir = Path(args.v1).expanduser()
    v2_dir = Path(args.v2).expanduser()
    annotations = (
        Path(args.annotations).expanduser()
        if args.annotations
        else v2_dir.parent / ANNOTATIONS_DIRNAME
    )
    # Resolved after `--v2`, because the default is the set staged *beside* it - the same rule
    # `--annotations` follows, and what keeps a build of a set staged somewhere else from merging
    # this workspace's hard negatives into it (`workspace.resolve_extras`).
    #
    # A union rather than either/or: the hard negatives are a second *set*, staged on their own with
    # their own manifest, so naming `cleaned-v2-s3` here must not quietly drop the 50 frames that
    # teach the model what these products are not. `--no-extras` is how a build that means only what
    # it names says so.
    extras = resolve_extras(v2_dir.parent, args.extras, include_defaults=not args.no_extras)

    if args.dry_run:
        # The dry run reads both sources and reports, without writing anything. It exists because
        # the interesting failures - an unmappable name, a frame nobody placed in a split - are
        # discoverable from the manifests alone, and finding them before 2,300 JPEGs are rewritten
        # is the difference between a two-second answer and a two-minute one.
        return dry_run(args, v1_dir, v2_dir, annotations, extras)

    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(
            f"{out} is not empty - pass --force to rebuild it (the new build is assembled beside "
            "it and swapped in only if it finishes, so a refusal leaves this set untouched)"
        )

    # Taken before the first annotation read rather than after the build: a decision saved *while*
    # this runs then shows up as a digest mismatch (the gate refuses) instead of being folded into
    # a report that claims to describe a state it raced.
    annotations_state = annotation_state(annotations)

    # Cheap refusal first: a `--v2` that is not a staged set should not cost 1,815 v1 frames' worth
    # of JPEG rewriting before it is noticed. The build reads the directory again (it needs the
    # store anyway) - this is about the two seconds, not about safety, since nothing is written to
    # `out` until the swap at the end.
    store, frames, problem = read_v2(v2_dir, annotations, extras)
    if not problem:
        # And the same cost argument for the second silent one, which is the state a rebuild is in
        # whenever the decisions live somewhere this machine cannot see: the manifest is full, no
        # frame has a label beside it, and the merge would be v1's frames alone. `build_v2` would
        # reach the same verdict after rewriting v1's export into the staging copy.
        problem = v2_contribution_problem(v2_dir, frames, store.splits(), args.allow_unassigned)
    if problem:
        raise SystemExit(f"not rebuilding from this --v2: {problem}")

    # Assembled beside `out` and swapped in below, so a refusal - a v1 name that cannot be
    # translated, a decided frame with no split, a `--v2` that is not a staged set - leaves the
    # previous merged set exactly as it was. This used to delete the three split directories first,
    # which made `--force` (the command an operator re-runs) the one that could destroy the only
    # copy of a two-minute build.
    staging = staging_dir(out)
    prepare_staging(staging)
    print(f"building in {staging} (moved into place only if this finishes)")
    try:
        sides: list[Side] = []
        if not args.no_v1:
            sides.append(build_v1(v1_dir, staging, args.size))
        sides.append(
            build_v2(
                v2_dir,
                annotations,
                staging,
                extras,
                args.size,
                args.allow_unassigned,
            )
        )

        problems = [(side.name, p) for side in sides for p in side.problems]
        if problems:
            # Every problem here is fatal, and each one is a way to train on wrong labels: v1's
            # indices cannot be translated at all, a `--v2` that is not a staged set contributes no
            # frames at all, or a v2 frame cannot be placed on a side of the train/test line. The
            # build stops with the frames that would have been merged rather than writing a set
            # that is quietly smaller or quietly mislabelled.
            for name, problem in problems:
                print(f"  ! {name}: {problem}")
            raise SystemExit("refusing to write a dataset from the problems above")

        # Test-side near-duplicates are dropped after both sides are in, because a v2 test frame
        # can duplicate a v1 train frame - which is exactly the leak that makes a merged set's test
        # number optimistic rather than the sum of its parts.
        notes: list[str] = []
        dropped = drop_test_duplicates(staging, notes)

        machine_only: Counter = Counter()
        for side in sides:
            machine_only.update(side.machine_only)
        report = summarise(
            staging, sides, dropped, machine_only, args.size, notes, annotations_state=annotations_state
        )

        # Drawn from the staging copy, recorded at the path it will have: this report outlives the
        # staging directory, and a sheet path pointing into a directory that no longer exists is
        # worse than no path at all.
        sheet_path = Path(args.contact_sheet).expanduser() if args.contact_sheet else out / "contact_sheet.jpg"
        sheet = contact_sheet(staging, sheet_path if args.contact_sheet else staging / "contact_sheet.jpg", args.contact_frames)
        sheet["path"] = str(sheet_path)
        report["contact_sheet"] = sheet

        side_lines = []
        for side in sides:
            # The report's count, so this line and the `sources` block cannot be two spellings.
            images = report["sources"][side.name]["images"]
            counts = ", ".join(f"{split} {images.get(split, 0)}" for split in SPLIT_NAMES)
            side_lines.append(
                f"- **{side.name}**: {counts if any(images.values()) else 'nothing merged'}"
            )
            if side.polygons:
                side_lines.append(
                    f"  - {side.polygons} polygon row(s) reduced to their bounding boxes, the same "
                    "conversion ultralytics applies at load time"
                )
            if side.machine_only:
                side_lines.append(
                    f"  - machine-only decisions: "
                    + ", ".join(f"{split} {n}" for split, n in sorted(side.machine_only.items()))
                )

        (staging / MERGE_REPORT_NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")
        (staging / "MERGE_REPORT.md").write_text(render_report(report, side_lines, sheet), encoding="utf-8")

        # The declaration goes into the staging copy before the doctor reads it - `data.yaml` is
        # where the doctor finds the class list, and it is the one the swap will carry into place.
        # Its split paths are the *final* ones (`splits` is built from `out`, not from `staging`),
        # because the file outlives the directory it is written into.
        splits = {
            split: split_dirs(out, split)[0]
            for split in SPLIT_NAMES
            if list_images(split_dirs(staging, split)[0])
        }
        write_names_yaml(out, splits, path=staging / DATA_YAML_NAME)

        # Before both gates, and outside the flag that skips them, because this one is not a
        # judgement about the annotation state: it reads back the declaration just written and
        # refuses a set whose own fields disagree about which product every label row means. A set
        # like that is refused by every tool that measures a weight against labelled frames
        # (`train_model.require_labels_order`), so landing it would replace a usable set with one
        # nothing can report on - and the reading flags' sets are read by those same tools.
        order = declared_order_problem(staging, out)
        if order:
            raise SystemExit(order)

        # The doctor, *here*, rather than only as §7's command a minute later: §7 runs it on the
        # merged set, and by the time it does, the only copy of the previous build is gone. This is
        # the same gate `train_model.py` and `accept_v2.py` refuse on, taken on the copy that has
        # not landed yet - so a merge that would fail it never becomes the merged set, and what
        # comes out is a set that has already been checked.
        #
        # The duplicate scan is included, even though the build just dropped duplicates itself: it
        # is the same check `make doctor` runs, and a build that prints a different verdict from the
        # command a minute later is worse than six seconds of fingerprinting.
        #
        # `--allow-unassigned` is the one build that skips it, and the flag is why: it says the set
        # is for reading how much is still unplaced, "never a trainable set", while the doctor judges
        # a set that could be trained (all three splits present, no leaked test frames). Refusing the
        # artifact that flag exists to produce would only add a third reason to ignore the output.
        if args.allow_unassigned:
            print("not doctored: --allow-unassigned means this set is not meant to be trained")
        else:
            # The human pass's gate first, then the doctor - the order `train_model` asks them in,
            # and the gate that is cheapest to answer and most expensive to discover late: a weight's
            # unread boxes in `valid`/`test` make the set's own numbers a measurement of the
            # annotator, so `train_model --yes/--val` and `accept_v2` both refuse this set. Building
            # it would replace a set that could be used with one that cannot, and the frames to work
            # are named in the annotator, not here.
            #
            # `--allow-machine-only` is the deliberate override, its sibling in spirit to the
            # `--allow-unassigned` above and for the same reason: reading a merge (the mix, the
            # contact sheet, how much is left in `far`) is a real thing to want from a set that is
            # not meant to be trained. It says out loud what it produced, so the refusal a run gives
            # later is not a surprise.
            ok, gate_lines = pass_gate(report)
            print()
            print("the human pass's gate, on the copy that has not landed yet:")
            for line in gate_lines:
                print(f"  {line}")
            if not ok:
                if not args.allow_machine_only:
                    raise SystemExit(
                        f"the merge did not pass the human pass's gate, so {out} was not touched - "
                        "work the frames above in the annotator, then build again"
                    )
                print(
                    "  landing it anyway: --allow-machine-only means this set is for reading, not "
                    "for training - a run will refuse it"
                )

            # Imported here, not at module scope: `dataset_doctor` imports *this* module (for
            # `find_test_duplicates`), so a module-level import would be a cycle.
            import dataset_doctor

            print()
            print(
                "the doctor, on the copy that has not landed yet "
                f"(judged against {DECLARED_GENERATION.name}'s order):"
            )
            doctor_report = dataset_doctor.check_and_report(staging, DECLARED_GENERATION)
            if doctor_report["errors"]:
                raise SystemExit(
                    f"the merge did not pass the doctor, so {out} was not touched - resolve the "
                    "[FAIL] lines above and build again"
                )

        swap_into_place(staging, out)
    finally:
        # Nothing is left behind on a refusal, a crash in the middle of the merge, or a swap that
        # failed halfway - the frames are reproducible, the disk is the scarce thing here.
        shutil.rmtree(staging, ignore_errors=True)

    # `data.scanncart.yaml` is written *after* the swap, unlike `data.yaml`, and the difference is
    # what each one is read by: `write_data_yaml` resolves the split directories it is given and
    # names the directory it describes (`path:`, then `train: <abs>/train/images`), so it can only be
    # written once the staging copy *is* `out` - written early it would name a directory that stops
    # existing the moment the build succeeds. Nothing before the swap reads it; the doctor reads the
    # class declaration, which is `data.yaml`. It reads `names` from that file too, hence the order.
    write_data_yaml(out, splits)

    print(f"merged dataset -> {out}")
    for split in SPLIT_NAMES:
        row = report["splits"][split]
        print(
            f"  {split:<6} {row['images']:>5} image(s), {row['boxes']:>5} box(es), "
            f"{row['background']:>3} background"
        )
    for split, count in sorted(machine_only.items()):
        print(f"  machine-only decisions in {split}: {count}")
    if dropped:
        print(f"  dropped {len(dropped)} test frame(s) that duplicate a train/valid frame")
    print(f"{DATA_YAML_NAME}  -> {out / DATA_YAML_NAME}")  # written early, carried by the swap
    print(f"contact sheet -> {sheet['path']}  ({sheet['frames']} frames)")
    print()
    print("Look at the contact sheet before training. It is the only check that can see a")
    print("mis-translated class index - `check_export` compares class names by membership.")
    print()
    print("  train_model.py --dataset-dir", out)
    return 0


def dry_run(args, v1_dir: Path, v2_dir: Path, annotations: Path, extras: list[Path]) -> int:
    """Report what a build would merge, from the manifests and label files alone.

    Deliberately does not import the image stack: this path answers "can these two sets be merged
    at all", which is a question about names and splits, and it should be answerable on a machine
    that has neither the frames nor a GPU.
    """
    print(f"v1   {v1_dir}")
    print(f"v2   {v2_dir}")
    print(f"out  {Path(args.out).expanduser()}")
    problems = 0
    if not args.no_v1:
        names = read_export_names(v1_dir)
        if names is None:
            print("  ! v1: no usable data.yaml, so its class indices cannot be translated")
            problems += 1
        else:
            print(f"  v1 declares {len(names)} class(es): {', '.join(names)}")
            # The same rule and sentence a build refuses on, so the dry run cannot report a merge
            # as fine that the build would stop on a second later.
            verdict = translation_problem(names, source="v1")
            if verdict:
                print(f"  ! {verdict}")
                problems += 1
            for split in SPLIT_NAMES:
                images = list_images(split_dirs(v1_dir, split)[0])
                print(f"  v1 {split:<6} {len(images):>5} image(s)")
    side = preview_v2(v2_dir, annotations, extras)
    decided = Counter(side.placed.values())
    for split in SPLIT_NAMES:
        print(f"  v2 {split:<6} {decided.get(split, 0):>5} decided frame(s)")
    for note in side.notes:
        print(f"  v2 {note}")
    for problem in side.problems:
        print(f"  ! v2: {problem}")
        problems += 1
    print()
    print("no files written (--dry-run)" + (" - resolve the problems above first" if problems else ""))
    return 2 if problems else 0


def preview_v2(v2_dir: Path, annotations: Path, extras: list[Path]) -> Side:
    """`build_v2` without writing anything: the dry run's half of the same reading.

    Same store, same split lookup, same `--v2` guard, so the counts a `--dry-run` prints are the
    counts a build would take in and the directory it refuses is the one a build would refuse -
    which is what makes it worth running first. They are the *decided* frames, not the frames that
    land: the dedup pass runs after the merge and can still drop test frames, so the report's
    per-side counts (`sources.<side>.images`) are the ones that describe the set.
    """
    side = Side(name="v2")
    store, frames, problem = read_v2(v2_dir, annotations, extras)
    if problem:
        # No note either: "0 frame(s) with no decision yet" over a directory that is not a set reads
        # as good news, which is exactly the reading this guard exists to stop.
        side.problems.append(problem)
        return side
    # The same verdict `build_v2` takes, because this is that function without the writes: a dry run
    # that reported a set fine and then met a refusal a second later would be worse than no dry run.
    problem = annotator_class_problem()
    if problem:
        side.problems.append(problem)
        return side
    splits = store.splits()
    undecided = 0
    unassigned: list[str] = []
    for frame in frames:
        if frame.state == "unlabeled":
            undecided += 1
            continue
        split = splits.get(frame.name)
        if split in SPLIT_NAMES:
            side.placed[frame.name] = split
            if frame.machine_only:
                side.machine_only[split] += 1
        else:
            unassigned.append(frame.name)
    side.notes.append(f"{undecided} frame(s) with no decision yet")
    if unassigned:
        side.problems.append(
            f"{len(unassigned)} decided frame(s) have no split in {v2_dir / SPLITS_NAME}"
        )
    contribution = v2_contribution_problem(v2_dir, frames, splits)
    if contribution:
        # The dry run's half of the same guard, and the reason it is worth running first: a set
        # nobody has decided anything in reads as `0 decided` in the counts above, which is good
        # news unless the run says the build would refuse it.
        side.problems.append(contribution)
    return side


if __name__ == "__main__":
    raise SystemExit(main())
