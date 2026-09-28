#!/usr/bin/env python
"""Check a built dataset before a training run, because the failures here are all silent.

`build_dataset.py` writes the merged set, `train_model.py` trains on it, and `accept_v2.py` judges
the result. Between those three, six classes of mistake produce a *valid* dataset that trains a
*wrong* model - none of them raises anything, and two are invisible even to a human looking at the
frames:

1. **A class index in the wrong order.** `check_export` compares class *names*, by membership: an
   export whose `names` list is the right seven in the wrong order passes every check the toolchain
   has, and every box then comes back under the wrong product. This is the failure `build_dataset`
   remaps by name to avoid, so the doctor checks the merged set declares the generation's names **in
   order** rather than as a set - and it says *which* order the labels index, in print and in
   `--json`, because "the right names in the wrong order" cannot be acted on without knowing which
   order they are. When they are another generation's list in that generation's order - the one
   case where the fix is a flag rather than a rebuild, and the case v1 and v2 make permanent, since
   they declare the same seven names in different orders - it names that generation and the flag.
   `--generation` therefore defaults to `auto`: the set is asked which generation it is (its own
   `merge_report.json`, then the order its `data.yaml` declares) instead of being assumed to be the
   newest one, so forgetting the flag cannot produce a verdict about the wrong expectation.
2. **A label row nothing can read.** A missing label file, a row with four fields, a class index
   past the end of the roster, a coordinate outside 0..1: ultralytics skips the row or clips the box
   and training proceeds, so the frame reads as labeled while contributing nothing or the wrong
   thing. Absent and empty are different states here, and only one of them is a decision. The two
   row *shapes* the loader accepts - a box and a polygon - are pinned against the loader itself
   rather than assumed: `row_shape` carries the measurement, and the tests re-take it.
3. **A frame drawn under a product other than the one it was staged as.** `clean_v2.py` filed every
   v2 frame under `<PRODUCT>/<DISTANCE>/` and recorded that folder as the frame's `class` - the tag
   the rest of the toolchain treats as what the frame *is*. `build_dataset.py` copies each written
   frame's tag into `merge_report.json` (the built set holds no manifest, so that is the last place
   the fact exists), and the doctor compares every tagged frame's rows against its tag. A label
   drawn under a neighbouring product is readable, in range and in a valid box: nothing else in
   this file can see it, and what it trains is a model that calls one product by another's name.
   The rule is `label_classes.tag_mismatch`'s, the same function `label_progress.py`'s two sources
   call (only a wrong *class* is a finding; a background frame keeps its tag, and a box on a
   hard-negative frame is different material rather than a mistake) - this file only adapts a built
   set's shape to it. v1's frames carry no tag, and a report from a build older than the tag copy
   has none either - that is a warning naming the gap, not a silent pass.
4. **A test frame that is a near-duplicate of a training frame.** The test number then measures
   recall of a photograph the model has seen. `build_dataset` drops these as it merges; the doctor
   uses the same thresholds (`build_dataset.find_test_duplicates`) because a second opinion about
   what "near-duplicate" means is how one of them ends up wrong.
5. **A merge report that no longer describes the set.** It is what `accept_v2` reads the
   `machine_only` gate from, so a report left over from an earlier build is a gate reading a
   different dataset - and it says nothing, because it is a file that looks fine in isolation.
6. **A merge whose v2 side contributed nothing.** `build_dataset.py` refuses to make one now, but a
   set built before that guard carries no mark of it on disk: three splits, the right names in
   order, every label readable - because all of it is v1's. The one file that says otherwise is the
   set's own `merge_report.json` (`sources.v2` empty), so the doctor reads it and fails the set
   rather than letting a weight named `scanncart-grocery-v2.pt` be a v1 model with a new filename.

One thing here is a *reading* rather than a check, and it belongs to the claim the set exists to
make: the distance mix of v2's frames, which the merge report records (`sources.v2.distances`)
because the built set does not name the staged directory whose manifest still holds it. A distance
the model would train on and `test` holds no frame of can never be measured, and a distance the
plan follows that is in no split at all is the capture gap - both are warnings rather than
failures, because a set holding only `close` frames is still perfectly trainable; what it cannot be
is evidence about `far`.

Nothing here writes, fixes or trains: it reads, reports, and exits non-zero on anything that would
make a measurement untrustworthy. Warnings (no merge report, a frame count that is merely thin) do
not fail it - only things that change what a number means.

It is also the *gate* on the two tools that act on a set rather than read it: `train_model.py`
refuses to train and `accept_v2.py` refuses to measure until this check passes on the set in front
of them. Running it by hand is how a problem is diagnosed; the gate is what makes it impossible to
skip. `check_and_report` below is the half those two share.

    sidecar/.venv/Scripts/python.exe sidecar/tools/dataset_doctor.py --dataset data/datasets/merged-v2
    sidecar/.venv/Scripts/python.exe sidecar/tools/dataset_doctor.py --json

    # a slower, exact duplicate scan (every test frame against every train frame)
    sidecar/.venv/Scripts/python.exe sidecar/tools/dataset_doctor.py --duplicates
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIDECAR_ROOT = HERE.parent
if str(SIDECAR_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(SIDECAR_ROOT))

import generations  # noqa: E402
import train_model  # noqa: E402
from build_dataset import DEFAULT_OUT as DEFAULT_DATASET  # noqa: E402
from build_dataset import find_test_duplicates  # noqa: E402
from label_classes import (  # noqa: E402
    DISTANCE_ORDER,
    FIT_SPLITS,
    PSEUDO_CLASS_SLUGS,
    SPLIT_NAMES,
    SLUG_TO_CLASS,
    tag_mismatch,
)
from workspace import MERGE_REPORT_NAME  # noqa: E402

# How many offending rows to name. A file with 300 bad rows is one finding, not 300 - and the
# operator needs the first few to see *which* mistake it is, not a full census.
SHOW = 5


def read_label(label: Path) -> tuple[list[list[str]], list[str]]:
    """`(rows, problems)` for one label file. A missing file is one problem, not zero rows.

    The distinction the whole dataset pipeline is built around: an **absent** label file means
    nobody has decided, while an **empty** one means somebody decided there is no item here. A
    doctor that read both as "no boxes" would report a half-labeled set as clean.
    """
    try:
        text = label.read_text(encoding="utf-8")
    except OSError:
        return [], [
            f"no label file at {label.name} - absent is not the same as empty (the loader would "
            "train this frame as a background frame, which is a decision nobody recorded here)"
        ]
    rows = [line.split() for line in text.splitlines() if line.strip()]
    return rows, []


def row_shape(row: list[str]) -> str:
    """Which of the loader's two shapes this row is read as: `box`, `polygon`, or `neither`.

    Measured against the loader rather than assumed - `ultralytics.data.utils.verify_image_label`,
    which is what both training and validation go through - and the measurement is short enough to
    keep here: five fields are a box; a file in which **any** row has more than six fields is read
    as a polygon file, where every row is `cls x1 y1 x2 y2 ...` and becomes the box around its own
    points (`segments2boxes`); anything else (six fields, an odd coordinate count) fails the
    loader's column assert and the whole *image* is dropped as corrupt.

    That `any` is per **file** - which is why this is a fact about a row only against the file it
    is in, and why `label_problems` re-reads every row in that context. v1's export is why it
    matters at all: 1,921 of its 2,111 rows are polygons, so a doctor that took "five fields" as the
    definition of a label would refuse the project's own set.
    """
    coords = len(row) - 1
    if coords == 4:
        return "box"
    if coords >= 6 and coords % 2 == 0:
        return "polygon"
    return "neither"


def label_problems(rows: list[list[str]], names: list[str]) -> list[str]:
    """Everything wrong with one frame's rows, as sentences. Pure, so the rules are testable.

    A *file* is one unit to the loader: it decides from the whole row list whether this is a polygon
    file, converts every row accordingly, and drops the entire image on any exception. So a row is
    judged in the context of its file here too - and the mixed case is the one that has no error of
    its own: one polygon row turns *every* row into a point list, including the box rows, whose four
    numbers then become two points and train as a box nobody drew.
    """
    segment_file = any(len(row) > 6 for row in rows)  # the loader's own predicate
    problems: list[str] = []
    for row in rows:
        shape = row_shape(row)
        if shape == "neither":
            problems.append(
                f"row with {len(row)} field(s) is neither `cls cx cy w h` nor a polygon - the "
                f"loader drops the whole frame for it: {' '.join(row)[:60]!r}"
            )
            continue
        if segment_file and shape == "box":
            # Reported and then read on as the box it was written as, not `continue`d: the row can
            # be wrong twice, and sending the operator back a second time for a fault this file was
            # already open for is the thing this function's other branches avoid.
            problems.append(
                "a box row in a file that contains a polygon row: the loader reads every row in "
                "this file as a polygon, so these four numbers become two points and the box it "
                "trains is not the one written here"
            )
        try:
            index = int(float(row[0]))
            values = [float(value) for value in row[1:]]
        except ValueError:
            problems.append(f"non-numeric row: {' '.join(row)[:60]!r}")
            continue
        if not 0 <= index < len(names):
            # The one that trains a model you cannot explain: ultralytics drops the row, so the
            # frame loses a box and nothing says so.
            problems.append(f"class index {index} is outside the {len(names)} declared classes")
        outside = [value for value in values if not 0.0 <= value <= 1.0]
        if outside:
            problems.append(
                f"coordinate outside 0..1 ({outside[0]:.4f}) - "
                + (
                    "the trainer clips the box silently"
                    if shape == "box"
                    else "the loader drops the whole frame as corrupt"
                )
            )
        if shape == "box":
            if values[2] <= 0 or values[3] <= 0:
                problems.append("box with no extent (w/h <= 0)")
        else:
            xs, ys = values[0::2], values[1::2]
            if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
                problems.append("polygon with no extent (its points are in a line)")
    return problems


def tag_problems(
    tags: dict[str, str],
    drawn: dict[str, set[int]],
    names: list[str],
) -> tuple[list[tuple[str, str, list[str]]], list[str]]:
    """`(mismatches, unknown_slugs)` for frames whose rows draw a class other than their tag's.

    The tag is the class a frame was *staged as* - the folder `clean_v2.py` filed it under, copied
    through by `build_dataset.py` because the built set holds no manifest. A label drawn under a
    neighbouring product is readable, in range and in a valid box, so no other check in this file
    can see it, and it reaches the GPU as a model that calls one product by another's name. This is
    the one rule that compares a frame's labels against a fact *outside* them.

    The rule itself is `label_classes.tag_mismatch`'s, the same function `label_progress.py`'s two
    sources call - this is only the adapter from a built set's shape (one class per frame, as an
    index into `names`) to it, plus the two doctor-specific readings: the class list indexes the
    drawings, so a row whose index is outside it belongs to `label_problems` and is skipped here
    rather than restated, and a slug nothing resolves is *reported* as an unknown so an unreadable
    tag becomes a sentence about the gap instead of a silent pass over frames nothing could judge.
    """
    mismatches: list[tuple[str, str, list[str]]] = []
    unknown: list[str] = []
    for name, slug in sorted(tags.items()):
        if slug in PSEUDO_CLASS_SLUGS:
            continue
        want = SLUG_TO_CLASS.get(slug)
        if want is None:
            unknown.append(slug)
            continue
        indices = drawn.get(name) or set()
        if not indices:
            continue
        wrong = tag_mismatch(slug, (names[i] for i in indices if 0 <= i < len(names)))
        if wrong:
            mismatches.append((name, want, list(wrong)))
    return mismatches, unknown


def plan_order(names, order: tuple[str, ...] = DISTANCE_ORDER) -> list[str]:
    """`names` in the plan's own order, with anything the plan does not follow sorted last.

    The one ranking of the distance axis. Two sentences read it - the unmeasured-distance
    findings and the mix line - and a second `{name: index}` construction is how the two
    come to disagree about which column of the grid a name belongs to. A name outside the
    plan is not an error to either of them (the hard negatives are recorded as `unknown`),
    so it sorts after the planned ones by name rather than raising.
    """
    rank = {name: index for index, name in enumerate(order)}
    return sorted(names, key=lambda name: (rank.get(name, len(rank)), name))


def distance_problems(
    distances: dict[str, dict[str, int]] | None,
    expected: tuple[str, ...] = DISTANCE_ORDER,
) -> list[str]:
    """What the v2 distance mix can and cannot measure, as sentences. Pure, so the rules are testable.

    Two states, and they need different actions:

    * a distance the model would **train on** (`train`/`valid`) that holds no `test` frame - the
      grid `--val` prints can never fill that cell, so the model is asked to learn a distance and
      never measured on it. The fix is in the split plan (`plan_split.py`), not the camera.
    * a distance the plan follows (`train_model.DISTANCE_ORDER`) that is in **no** split - no
      decided frame of it exists in the set at all, which is capture or labelling work rather
      than a planning one.

    `None` - no mix recorded, which is what a report written before the merge recorded it looks
    like - answers nothing rather than guessing; the caller says the mix was not read. `unknown`
    (the hard negatives staged outside the distance cells) is not a distance and is left out of
    both answers, because nothing was planned at it either way.
    """
    if distances is None:
        return []

    def counts(split: str) -> dict[str, int]:
        block = distances.get(split)
        if not isinstance(block, dict):
            return {}
        return {
            name: value
            for name, value in block.items()
            if isinstance(name, str) and isinstance(value, int) and name != "unknown"
        }

    test_counts = counts("test")
    trained: dict[str, int] = {}
    for split in FIT_SPLITS:
        for name, value in counts(split).items():
            trained[name] = trained.get(name, 0) + value
    order = list(expected)
    unmeasured = plan_order((name for name in trained if name not in test_counts), expected)
    present = set(test_counts) | set(trained)
    absent = [name for name in order if name not in present]

    out: list[str] = []
    for name in unmeasured:
        out.append(
            f"`{name}` is trained on ({trained[name]} frame(s) in train/valid) and holds no test "
            "frame, so the model is asked to learn that distance and never measured on it - the "
            "split plan has to put frames of it in `test`"
        )
    if absent:
        out.append(
            "no v2 frame at "
            + " or ".join(f"`{name}`" for name in absent)
            + " is in this set, so a per-distance number can only ever be quoted for "
            + (", ".join(f"`{name}`" for name in order if name in present) or "none of it")
            + " - the missing cells are capture work before they are anything else"
        )
    return out


def side_count_problems(sources: dict, counts: dict[str, dict]) -> list[str]:
    """Sentences for a report whose sides do not add up to the set they describe. Pure.

    Nothing on disk says which side a frame came from, so summing `sources.<side>.images` is the only
    cross-check available - and it is the one that catches a side counted *before* the dedup pass
    dropped its frames (the report this rule was written for read `v2: test 83` two lines above a mix
    saying `test close 36`). Silence when no side names a frame count: `sources` may be absent (not
    the builder's report) and that is not this rule's claim to make. A split no side mentions counts
    as zero - a side that placed nothing there omits it - and is exactly what the comparison notices.
    """
    named: dict[str, int] = {}
    for block in sources.values():
        images = block.get("images") if isinstance(block, dict) else None
        if not isinstance(images, dict):
            continue
        for split, value in images.items():
            if isinstance(value, int):
                named[split] = named.get(split, 0) + value
    if not named:
        return []
    return [
        f"{split}: its sides account for {named.get(split, 0)} frame(s), disk has "
        f"{counts[split]['images']}"
        for split in counts
        if named.get(split, 0) != counts[split]["images"]
    ]


def recorded_generation(dataset: Path) -> str | None:
    """The generation `merge_report.json` says this set was built for, when it names a known one.

    A *statement* rather than an inference, and the strongest of the two pieces of evidence
    `resolve_generation` reads: the report is written by `build_dataset.py`, which is the tool that
    chose the class order the labels were written in. Missing, corrupt or unexpected values are
    `None` - never a guess, and never a crash, because this is read on the same path that reports
    an unreadable report as a warning.
    """
    path = dataset / MERGE_REPORT_NAME
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = body.get("generation") if isinstance(body, dict) else None
    return name if isinstance(name, str) and name in generations.GENERATIONS else None


def resolve_generation(dataset: Path) -> tuple[generations.Generation, str]:
    """Which generation this dataset is, and how that was decided - for `--generation auto`.

    The flag exists because the failure it prevents is an operator's memory: a v1 set judged
    against v2's order fails, and the finding is about the expectation rather than the data - a
    verdict that would send someone to fix labels that are correct. So the set is asked, in order
    of directness:

    1. the `generation` its own `data.yaml` records (`build_dataset` writes it beside `names`),
       which is the statement made by the file whose list the rows actually index.
    2. `merge_report.json`'s `generation` - the build's provenance note. Read when the yaml has
       none, and *named in the answer* when the two disagree: two files in one directory claiming
       two generations is not something a header should quietly pick a winner from.
    3. the `names` the yaml declares, matched **in order** against each generation's class list
       (`generations.order_of`): the one thing in a names list that tells these two generations
       apart, so a set built by something else is still answerable.
    4. otherwise the tool's default, *said out loud* (the second half of the return value is what
       the header prints) - because "judged as v2" is a claim, and a claim nobody made on purpose
       is the kind a reader has to be told about.
    """
    declared = train_model.read_export_generation(dataset)
    recorded = recorded_generation(dataset)
    if declared:
        disagreement = (
            f" (merge_report.json says {recorded})" if recorded and recorded != declared else ""
        )
        return generations.get(declared), f"from its data.yaml{disagreement}"
    if recorded:
        return generations.get(recorded), "from merge_report.json"
    found = generations.order_of(train_model.read_export_names(dataset) or ())
    if found is not None:
        return found, "from the order its class list declares"
    return generations.DEFAULT, "from the default - nothing in the set says"


def order_problem(
    names: list[str], generation, claims: list[tuple[str, str]] | None = None
) -> str | None:
    """The order finding for a declared class list, or None when there is nothing to say.

    Pure, and deliberately narrow: a *membership* mismatch is `check_export`'s error and not
    repeated here (the same situation printed twice is how a reader learns to skim), so this speaks
    only when the names are the expected ones in another order.

    Two sentences, because the two cases need different actions. When the order is another
    generation's, this is not a corrupt set at all - it is a set being judged against the wrong
    generation, and the fix is `--generation <theirs>`; relabelling it to satisfy this check would
    corrupt every box in it. Anything else is a genuine permutation, where the list *is* the
    problem.

    `claims` is what the set says about *itself* - `(generation, where it is written)` pairs, from
    its `data.yaml` and its merge report - and any of them naming a generation other than the one
    the rows index is quoted, because that contradiction is the one thing a reader cannot see from
    the list alone: two files in one directory disagreeing about what the labels mean is a
    different problem from a set that simply records nothing.
    """
    expected = list(generation.classes)
    if not names or names == expected:
        return None
    if sorted(names) != sorted(expected):
        return None
    other = generations.order_of(names)
    if other is not None:
        disagreeing = [f"{name} in its {where}" for name, where in claims or () if name != other.name]
        disagreement = (
            f" (the set records {', '.join(disagreeing)}, so its own record disagrees)"
            if disagreeing
            else ""
        )
        return (
            f"the labels index {other.name}'s order, not {generation.name}'s{disagreement} - a "
            "label row is a *position* in this list, so every box would be trained under a "
            f"neighbouring product. Judge the set as the generation it is (`--generation "
            f"{other.name}`) rather than relabelling it. Declared: {', '.join(names)}. Expected "
            f"for {generation.name}: {', '.join(expected)}"
        )
    return (
        "the class names are the right ones in the wrong ORDER - label rows index this list, so "
        f"every box would be trained under a neighbouring class. Declared: {', '.join(names)}. "
        f"Expected: {', '.join(expected)}"
    )


def check_dataset(
    dataset: Path,
    generation=generations.DEFAULT,
    duplicates: bool = True,
    exports: tuple[dict[str, Path], list[str]] | None = None,
    decided_by: str = "as asked",
) -> dict:
    """The whole check, as data. `errors` decide the exit code; `warnings` are for reading.

    One function for the report and the exit code, so the thing printed and the thing that fails
    cannot be two different checks - and `--json` is the same readout rather than a second one.

    `exports` is `check_export`'s own `(splits, problems)`, for the two callers that have to run it
    anyway (`train_model` needs the split paths to write its data.yaml, `accept_v2` the same): it is
    the same read of the same files a moment earlier, and running it twice would print its per-split
    lines twice in one run.

    `decided_by` is how `generation` was chosen (`resolve_generation`'s second return value for a
    run that did not name one). It rides in the report because the class order was judged against
    that generation: a `--json` reader has to be able to see *why* the expectation is what it is.
    """
    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {
        "dataset": str(dataset),
        "generation": generation.name,
        "generation_decided_by": decided_by,
    }

    splits, problems = exports if exports is not None else train_model.check_export(dataset, generation)
    errors.extend(problems)
    if not splits:
        return {"dataset": str(dataset), "generation": generation.name,
                "generation_decided_by": decided_by, "errors": errors,
                "warnings": warnings, "splits": {}}

    names = train_model.read_export_names(dataset) or []
    # Which order these labels index, asked of the names themselves rather than assumed: the same
    # seven names in another order relabel every box, and the two generations this app has declare
    # exactly that - so the report carries the declaration, the expectation, and which generation's
    # order the declaration *is*, and `order_problem` turns the three into the one sentence that
    # says what to do about it.
    expected = list(generation.classes)
    matched = generations.order_of(names)
    report["names"] = names
    report["expected_names"] = expected
    report["order_generation"] = matched.name if matched is not None else None
    claims = [
        (name, where)
        for name, where in (
            (train_model.read_export_generation(dataset), "data.yaml"),
            (recorded_generation(dataset), "merge_report.json"),
        )
        if name
    ]
    problem = order_problem(names, generation, claims)
    if problem:
        errors.append(problem)

    counts: dict[str, dict] = {}
    # name -> the class indices its rows draw, for the tag check below. Read while the label files
    # are already open rather than in a second pass, and kept as *indices* so a set with a broken
    # names list stays `label_problems`' business rather than crashing this one.
    drawn: dict[str, set[int]] = {}
    where: dict[str, str] = {}
    for split in SPLIT_NAMES:
        if split not in splits:
            continue
        images = sorted(
            p for p in splits[split].iterdir() if p.suffix.lower() in train_model.IMAGE_SUFFIXES
        )
        labels_dir = splits[split].parent / "labels"
        rows_total = 0
        background = 0
        polygons = 0
        bad: list[str] = []
        bad_frames: set[str] = set()
        orphan_labels: list[str] = []
        stems = set()
        for image in images:
            stems.add(image.stem)
            rows, unreadable = read_label(labels_dir / f"{image.stem}.txt")
            indices: set[int] = set()
            for row in rows:
                try:
                    indices.add(int(float(row[0])))
                except ValueError:
                    continue
            drawn.setdefault(image.name, set()).update(indices)
            where.setdefault(image.name, split)
            # Both halves matter, and both have to survive: the file's own problem (absent, not
            # empty) and the rows' problems. Dropping either one is how a set with *no labels at
            # all* reads as "every frame is a background" and passes.
            found = unreadable + label_problems(rows, names)
            if found:
                # Frames, not rows: one frame with three bad rows is one frame that trains as the
                # wrong thing, and reporting it as three would overstate how much is broken.
                bad_frames.add(f"{split}/{image.name}")
                bad += [f"{split}/{image.name}: {problem}" for problem in found]
            rows_total += len(rows)
            # Counted because a polygon is a box the loader derives rather than reads: the frame is
            # fine and the number it contributes is not the rectangle anybody wrote down. Same unit
            # as the merge report's `polygons_reduced`, so the two cannot be read as different
            # quantities.
            polygons += sum(1 for row in rows if row_shape(row) == "polygon")
            if not found and not rows:
                background += 1
        for label in sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []:
            if label.stem not in stems:
                orphan_labels.append(label.name)
        counts[split] = {
            "images": len(images),
            "boxes": rows_total,
            "background": background,
            "polygons": polygons,
            "label_problems": len(bad),
            "problem_frames": len(bad_frames),
            "orphan_labels": len(orphan_labels),
        }
        if bad:
            errors.append(
                f"{split}: {len(bad_frames)} frame(s) would train as something other than what "
                f"they say ({len(bad)} label problem(s))"
            )
            for line in bad[:SHOW]:
                errors.append(f"  {line}")
            if len(bad) > SHOW:
                errors.append(f"  ... and {len(bad) - SHOW} more")
        if orphan_labels:
            warnings.append(
                f"{split}: {len(orphan_labels)} label file(s) with no image "
                f"({', '.join(orphan_labels[:3])}{' ...' if len(orphan_labels) > 3 else ''}) - "
                "harmless for training, but they are left over from something"
            )
        # No "this split is empty" error here: `check_export` already refuses an empty images
        # directory, and the same situation reported twice is how a reader learns to skim the list.

    report["splits"] = counts

    # The merge report is what `accept_v2` reads the machine-only gate from, so a stale one is a
    # gate reading a different dataset. Checked against the frames on disk rather than trusted.
    merge_path = dataset / MERGE_REPORT_NAME
    if merge_path.is_file():
        try:
            merge = json.loads(merge_path.read_text(encoding="utf-8"))
        except ValueError:
            merge = None
        if not isinstance(merge, dict):
            warnings.append("merge_report.json is unreadable, so the acceptance gate cannot be checked")
        else:
            tags = merge.get("tags") if isinstance(merge.get("tags"), dict) else None
            stale: list[str] = []
            recorded = merge.get("splits") if isinstance(merge.get("splits"), dict) else {}
            # Above the staleness check, which needs them: the sides' own counts are part of it.
            sources = merge.get("sources") if isinstance(merge.get("sources"), dict) else {}
            v2_block = sources.get("v2") if isinstance(sources.get("v2"), dict) else {}
            v2_images = v2_block.get("images")
            if not isinstance(v2_images, dict):
                v2_images = None
            # Kept apart from `images` rather than folded into it: one is how many frames came in,
            # the other is what the per-distance grid will be able to say about them.
            v2_distances = (
                v2_block.get("distances") if isinstance(v2_block.get("distances"), dict) else None
            )
            for split, block in counts.items():
                was = (recorded.get(split) or {}).get("images")
                if isinstance(was, int) and was != block["images"]:
                    stale.append(f"{split}: report says {was}, disk has {block['images']}")
            # The builder writes tags only for the frames it writes, so a tag naming a frame this
            # set does not hold is evidence the report describes a *different* build - the same
            # finding as a stale frame count, and it belongs in the same sentence.
            for name in tags or {}:
                if name not in drawn:
                    stale.append(f"its tags name {name}, which this set does not hold")
            stale += side_count_problems(sources, counts)
            if stale:
                errors.append(
                    "the merge report does not describe this dataset ("
                    + "; ".join(stale)
                    + ") - rebuild it, or the acceptance gate is reading the old build"
                )
            if tags is None:
                # Older builds, and any tool but `build_dataset.py`: nothing on the set says what
                # each v2 frame *is*, so there is nothing to check the drawings against. A warning
                # rather than an error - the set may be perfectly good - but the one reading this
                # must not take silence for a pass.
                warnings.append(
                    "the merge report carries no `tags` block, so no frame's labels were checked "
                    "against the class it was staged as - rebuild with `build_dataset.py` to record "
                    "them, or read this set as unverified rather than clean"
                )
            else:
                mismatched, unknown = tag_problems(tags, drawn, names)
                if unknown:
                    shown = ", ".join(sorted(set(unknown))[:3])
                    warnings.append(
                        f"{len(set(unknown))} class slug(s) in the tags are not classes this app "
                        f"knows ({shown}{' ...' if len(set(unknown)) > 3 else ''}), so the frames "
                        "staged under them were not judged against anything"
                    )
                if mismatched:
                    errors.append(
                        f"{len(mismatched)} frame(s) are drawn under a class other than the one "
                        "they were staged as - the model would learn that product under the name "
                        "drawn here:"
                    )
                    for name, want, wrong in mismatched[:SHOW]:
                        errors.append(
                            f"  {where.get(name, '?')}/{name}: staged as {want}, drawn as "
                            f"{', '.join(wrong)}"
                        )
                    if len(mismatched) > SHOW:
                        errors.append(f"  ... and {len(mismatched) - SHOW} more")
            if "machine_only_by_split" not in merge:
                warnings.append(
                    "the merge report carries no `machine_only_by_split`, so `make accept-v2` "
                    "cannot evaluate its gate on this set"
                )
            # What the set says about *where it came from*, as opposed to what is on its disk. A
            # merge is v1's export plus v2's local frames, so a report whose `sources.v2` counts are
            # empty describes a set that is v1's frames under the merged name - the one failure in
            # this file with no other symptom, because every frame, name and label on disk is
            # consistent: they are simply all v1's. `build_dataset.py` refuses to *build* that now
            # (`v2_contribution_problem`), and this is the same verdict for the sets built before it
            # could, read off the report rather than re-derived, since nothing on disk distinguishes
            # the two sources. Absent `sources` (a report from another tool) is silence, not a
            # claim - only an explicitly empty v2 side is one.
            if v2_images is not None and not any(v2_images.values()):
                errors.append(
                    "the merge report says the v2 side contributed no frames at all (`sources.v2` "
                    "is empty), so this set is v1's frames under the merged name - a weight trained "
                    "on it would be a v1 model wearing v2's filename. Rebuild once the staged "
                    "frames have decisions: `build_dataset.py --force` refuses until v2 contributes "
                    "something"
                )
            if v2_images is not None and v2_distances is None:
                # A builder's report from before the merge recorded the axis, or a hand-written
                # one. The distance check cannot run over it, and silence would read as coverage.
                warnings.append(
                    "the merge report records a v2 side and no distance mix "
                    "(`sources.v2.distances`), so what the per-distance grid can measure was not "
                    "read - rebuild with `build_dataset.py` to record it"
                )
            for sentence in distance_problems(v2_distances):
                warnings.append(sentence)
            report["merge_report"] = {
                "images": {split: (recorded.get(split) or {}).get("images") for split in counts},
                "v2_images": v2_images,
                "v2_distances": v2_distances,
                "machine_only_by_split": merge.get("machine_only_by_split"),
            }
            if tags is not None:
                # The tag check's evidence for a `--json` reader: how many frames carried one, and
                # which ones are the finding - so a witness of a failed run can show the frames
                # without re-deriving them.
                report["tags"] = {
                    "tagged": len(tags),
                    "mismatched": [
                        f"{where.get(name, '?')}/{name}" for name, _, _ in mismatched
                    ],
                    "unknown_slugs": sorted(set(unknown)),
                }
    else:
        warnings.append(
            "no merge_report.json - this set did not come from build_dataset.py, so provenance "
            "(and therefore the machine-only gate) is not checkable"
        )

    if duplicates:
        # A frame the fingerprinter cannot decode (a truncated JPEG, a stray text file with an image
        # extension) must not crash the check and must not read as "no duplicates found" either:
        # the loader drops such frames silently, so a scan that could not run is a finding of its
        # own - `accept_v2`'s machine-only gate is refused for the same reason.
        try:
            found = find_test_duplicates(dataset)
        except OSError as exc:
            found = []
            errors.append(
                f"the duplicate scan could not read a frame ({exc}) - so an unmeasured split count "
                "cannot be read as \"no duplicates\""
            )
        report["test_duplicates"] = found
        if found:
            errors.append(
                f"{len(found)} test frame(s) are near-duplicates of a train/valid frame - the test "
                "number would measure recall of a photograph the model has seen: "
                + ", ".join(f"{d['name']} == {d['duplicate_of']}" for d in found[:3])
                + (" ..." if len(found) > 3 else "")
            )

    report["errors"] = errors
    report["warnings"] = warnings
    return report


def class_lines(report: dict) -> list[str]:
    """The declared class list, numbered, and which generation's order it is.

    Enumerated rather than counted because the order *is* the fact being reported: `7 declared`
    cannot be compared with anything, while a numbered list can be read against the generation's
    own at a glance. The generation is named for the same reason - "the canonical order" is not one
    list any more, and a reader who is told only that something is wrong with *an* order has no way
    to tell which of the two this set indexes without going to look.
    """
    names = list(report.get("names") or [])
    judged = report.get("generation") or generations.DEFAULT.name
    order = report.get("order_generation")
    if not names:
        # No names to index: `check_export` has already said why (a missing data.yaml), and a list
        # of zero is not a class order anybody can read.
        return [f"classes: {len(report.get('expected_names') or [])} expected, none declared"]
    if order == judged:
        headline = f"classes: {len(names)} declared, in {judged}'s order:"
    elif order:
        headline = (
            f"classes: {len(names)} declared, in {order}'s order - this check judged them against "
            f"{judged}'s:"
        )
    else:
        headline = (
            f"classes: {len(names)} declared, in an order no generation declares (judged "
            f"against {judged}'s):"
        )
    return [headline] + [f"  {index:>2}  {name}" for index, name in enumerate(names, start=1)]


def report_lines(report: dict) -> list[str]:
    """The human reading of a report: what it measured, then warnings, then the verdict.

    One implementation for all three callers - `main` below, and the two tools that *act* on a set
    (`train_model.py` before it trains, `accept_v2.py` before it measures) - so what an operator
    read when they ran the doctor by hand cannot differ from the block that refused their run.
    """
    lines = class_lines(report)
    for split, block in report.get("splits", {}).items():
        lines.append(
            f"  {split:<6} {block['images']:>5} image(s), {block['boxes']:>5} box(es), "
            f"{block['background']:>3} background, {block['problem_frames']:>3} bad label(s)"
        )
        if block.get("polygons"):
            lines.append(
                f"         {block['polygons']:>5} polygon row(s) - the loader reduces each to the "
                "box around its own points, so those numbers are the loader's, not the annotator's"
            )
    # What the per-distance grid will be able to say, printed with the counts it belongs to: the
    # warning below names a gap, and this is the line that lets a reader see the whole mix at once.
    mix = (report.get("merge_report") or {}).get("v2_distances")
    if mix:
        # `plan_order` owns the column order (`close mid far`, not alphabetical): the line reads
        # against the grid `--val` prints, so the two should not have to be re-sorted to match.
        rows = []
        for split in SPLIT_NAMES:
            counts = mix.get(split)
            if not counts:
                continue
            rows.append(
                f"{split} "
                + ", ".join(f"{name} {counts[name]}" for name in plan_order(counts))
            )
        if rows:
            lines.append("  v2 distances: " + "; ".join(rows))
    lines.append("")
    for warning in report["warnings"]:
        lines.append(f"  [warn] {warning}")
    for error in report["errors"]:
        lines.append(f"  [FAIL] {error}" if not error.startswith("  ") else f"  {error}")
    if not report["errors"]:
        # Which order counted as "in order" is named here too: with two generations declaring the
        # same seven names, "in order" on its own is a sentence about a list nobody named. The tag
        # clause is conditional because a report with no tags means the check did not run, and an
        # `[ok]` that claimed it had would be the exact silence this file exists to break.
        judged = report.get("generation") or generations.DEFAULT.name
        staged = (
            ", every staged frame's labels naming its own class" if report.get("tags") else ""
        )
        if "test_duplicates" in report:
            lines.append(
                f"  [ok] the set is internally consistent: names in {judged}'s order, every label "
                f"readable{staged}, no test frame duplicating a train one"
            )
        else:
            lines.append(
                f"  [ok] names in {judged}'s order and every label readable{staged} "
                "(duplicate scan skipped)"
            )
    return lines


def check_and_report(
    dataset: Path,
    generation=generations.DEFAULT,
    duplicates: bool = True,
    exports: tuple[dict[str, Path], list[str]] | None = None,
) -> dict:
    """`check_dataset`, printed in the doctor's own words, for a caller that is about to act.

    The verdict is `report["errors"]`, exactly as this tool's exit code is: an empty list means the
    set passed, and the caller may go on to train or to measure. Thin on purpose - it exists so
    `train_model.py` and `accept_v2.py` gate on the doctor *as the doctor*, in print and in verdict,
    rather than each re-deriving what a pass looks like.
    """
    report = check_dataset(dataset, generation, duplicates=duplicates, exports=exports)
    for line in report_lines(report):
        print(line)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="the dataset to check")
    ap.add_argument(
        "--generation",
        default="auto",
        choices=[*sorted(generations.GENERATIONS), "auto"],
        help=(
            "which class list and requirement the set is judged against; `auto` (the default) reads "
            "it off the set itself - merge_report.json, then the order its data.yaml declares - so a "
            "v1 set cannot be judged against v2's order by omission"
        ),
    )
    ap.add_argument(
        "--no-duplicates",
        action="store_true",
        help="skip the near-duplicate scan (it fingerprints every train/valid frame)",
    )
    ap.add_argument("--json", action="store_true", help="emit the check as JSON")
    args = ap.parse_args(argv)

    dataset = Path(args.dataset).expanduser()
    if not dataset.is_dir():
        raise SystemExit(f"no dataset at {dataset}")

    if args.generation == "auto":
        generation, decided_by = resolve_generation(dataset)
    else:
        generation, decided_by = generations.get(args.generation), "as asked"

    # `--json` must leave stdout carrying nothing but JSON, or a caller cannot parse it at all - so
    # in that mode every human-facing line, including this header, is a note on stderr. The header
    # says how the generation was decided, because that is what every line under it is judged
    # against - and `auto` is a resolution, not an assumption to hide.
    header = f"dataset: {dataset}  (generation {generation.name} - {decided_by})"
    print(header, file=sys.stderr if args.json else sys.stdout)
    # `check_export` measures as it goes and prints what it measured, which human mode wants inline.
    with contextlib.redirect_stdout(sys.stderr if args.json else sys.stdout):
        report = check_dataset(
            dataset,
            generation,
            duplicates=not args.no_duplicates,
            decided_by=decided_by,
        )

    if not args.json:
        for line in report_lines(report):
            print(line)
    else:
        print(json.dumps(report, indent=1))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
