#!/usr/bin/env python
"""Decide whether v2 may replace v1, by measuring both on the same frames.

`make accept-v2`. One command, one verdict, and every number in it comes from the merged
dataset's own test split - the split `plan_split.py --holdout-session` draws from a capture
session training never saw.

Five questions, and the command fails closed on all five - the first of which is asked before any
number exists:

1. **Is the set internally consistent?** The doctor (`dataset_doctor.py`) has to pass on it first:
   a class list in an order that relabels every box, a label row the loader drops, a frame drawn
   under a class other than the one it was staged as, a test frame that duplicates a training
   frame. Each of those trains and validates happily, so the acceptance
   number would be a measurement of the *set* rather than of the model - and the same gate is on
   `train_model.py`, so a set that fails it cannot be trained into weights to accept either.
2. **Is the set fit to be measured?** A decision a weight made and nobody reviewed is not evidence
   about the model, so `machine_only_by_split` in the merge report has to be zero for `valid` and
   `test` - and the report has to name the annotation state it counted, because those counts are
   only evidence while the decisions behind them are unchanged (`annotations`, checked against the
   `provenance.json` on disk). A set whose report is missing, stale or unverifiable cannot be
   asked, and "cannot be asked" is reported as a failure rather than a pass: the one thing this
   command must never do is print a number over a split it did not check.
3. **Does the candidate clear 6's per-class floor?** Per class, not the mean - the mean over seven
   classes is exactly what hid `century-tuna` at `far` while `milo` at `close` carried it.
4. **Did anything regress against v1?** The baseline is measured here and now, on the same frames
   and at the same `imgsz`. v1's published number came from its own export over its own split, so
   quoting it as a comparison would be comparing two different measurements. A class may lose up
   to `--regression-tolerance` (0.02) before it counts - the floor is the bar, and a wobble above
   it is not a regression.
5. **Does it beat v1 where this dataset exists to win?** The crowded counter: how many test frames
   each weight finds **two or more** items on, which is a count and can be checked rather than
   argued. Asked *per distance*, because the cells this dataset exists to win are the `mid`/`far`
   ones and a total over the split cannot answer for them: on a split whose crowded frames are all
   `close`, the total can rise while the captures the project was re-shot for get worse. So the
   table below is cut by the distance each frame was filed under and the claim is made on
   `--claim-distances` (default `mid,far`); a claimed distance the split files no frame at is a
   **failure**, not a pass with a caveat - "cannot be asked" is not "held", which is the same rule
   question 2 follows. `--claim-distances none` is for a run that is not making the claim (and
   `--no-crowding` skips the whole question, both passes included). `--iou-sweep` re-counts the
   candidate at several NMS iou values, in the same per-distance rows, and says whether that count
   moved: if it does, the bucket is a property of the threshold and has to be quoted with it; if it
   does not, the count is the model's. Nothing is auto-tuned - the sweep reports.

Both weights are required and read from disk, not guessed: a missing file is a hard error before
any measurement, the same way `make verify-clamp` refuses to skip. Each pass is `train_model`'s
own `validate`, so this command cannot measure something the documented sequence would not.

Each weight is measured **in its own class order**, which is not a formality. A label row's class
index is a position in the *set's* declared list, and ultralytics matches a prediction to a label by
that index - so a weight whose head indexes another order is scored under the wrong product
everywhere the two permutations differ. `--install` records each weight's own export class list
beside it, so the set's order and each weight's are both known: a weight that indexes the set's
order is measured as the set stands, and one that indexes a reordering of it is measured on a
remapped view of the same frames (only the class column moves). Measured before that existed, v1
scored `mAP50` 0.142 on this set with `century_tuna` - the permutation's one fixed point - as its
only credited class, against 0.966 on the same frames in its own order. A weight with no recorded
class list is refused rather than assumed to match, because assuming is what produced the 0.142.

    sidecar/.venv/Scripts/python.exe sidecar/tools/accept_v2.py \\
        --baseline sidecar/models/scanncart-grocery-v1.pt \\
        --candidate sidecar/models/scanncart-grocery-v2.pt

    # without the counting passes (two predicts over the split, plus one per sweep value)
    sidecar/.venv/Scripts/python.exe sidecar/tools/accept_v2.py --no-crowding --baseline … --candidate …

    # the whole-split count only, making no per-distance claim (a run that is not re-shooting
    # anything, or an exploratory read on a set whose distances are not what it is about)
    sidecar/.venv/Scripts/python.exe sidecar/tools/accept_v2.py --claim-distances none --baseline … --candidate …

    # the same verdict as JSON, for a script
    sidecar/.venv/Scripts/python.exe sidecar/tools/accept_v2.py --json --baseline … --candidate …
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# The sidecar root, so `train_model` (a sibling here) and `app.models` (the record reader) are
# importable when this file is run as a script - `sys.path[0]` is `tools/` in that case. Same
# guard as `build_dataset.py` and `annotate/run.py`.
HERE = Path(__file__).resolve().parent
SIDECAR_ROOT = HERE.parent
if str(SIDECAR_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(SIDECAR_ROOT))

import dataset_doctor  # noqa: E402
import generations  # noqa: E402
import train_model  # noqa: E402
from build_dataset import DEFAULT_OUT as DEFAULT_DATASET  # noqa: E402
from workspace import MERGE_REPORT_NAME  # noqa: E402

# The confidence floor every measurement here runs at, pinned rather than read from
# data/settings.json: the acceptance number has to mean the same thing on every machine, and 0.5 is
# the operating point the other published tables were measured at (`clamp_probe` pins the same).
CONF = 0.5
# NMS iou for the counting passes. Ultralytics' default, named because `--iou-sweep` exists to say
# whether this value is load-bearing for the bucket being quoted.
IOU = 0.7
IOU_SWEEP = (0.5, 0.7, 0.9)
# How much a class may lose against v1 before it counts as a regression. Not zero: the floor is the
# bar, and two measurements of the same weights on the same split differ by more than nothing.
REGRESSION_TOLERANCE = 0.02
# The crowding claim, in the one form that can be checked from a file list: a frame the model finds
# two or more items on. Not "more detections in total" - a model that finds one item on four frames
# and two on none is not solving the crowded counter.
CROWD_MIN = 2
GATE_SPLITS = ("valid", "test")
# The distances the crowding claim is about: the `mid`/`far` cells this dataset was re-shot to win.
# A default rather than a law, and a *named* one rather than "whatever the split holds" - a claim
# that shrinks to the distances a split happens to file is not a claim. Overridable with
# `--claim-distances` (and `none` for a run that is making no per-distance claim at all).
CLAIM_DISTANCES: tuple[str, ...] = ("mid", "far")
# The row a frame that records no distance lands in. Named rather than dropped: v1's export frames
# carry no distance tag (the axis was added for v2), so on the merged set this is most of the
# split, and rows that did not add up to the total would look like a fault in the counting. It is
# never a cell, so it is never the claim's - `--claim-distances unattributed` is allowed only
# because refusing one row the table prints would be the more surprising rule.
UNATTRIBUTED = "unattributed"
# The fallback sweep row, for a run that claims no distance: the whole split, which is what the
# sweep covered before the per-distance table existed.
WHOLE_SPLIT = "the whole split"


# ---------------------------------------------------------------------------
# The gate: what the dataset itself says about how it was annotated
# ---------------------------------------------------------------------------


def read_report(dataset: Path) -> dict:
    """The merge report `build_dataset.py` wrote, or `{}`.

    It is the dataset's own account of what went into each split and the only artifact carrying
    provenance: the frames are JPEGs and label files, and nothing in a label file says whether a
    person or a weight drew it. A missing or corrupt report answers `{}`, which the gate below
    treats as *cannot verify* rather than as clean.
    """
    try:
        body = json.loads((dataset / MERGE_REPORT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def machine_only_gate(
    report: dict, splits: tuple[str, ...] = GATE_SPLITS
) -> tuple[bool, list[str]]:
    """Whether the measured splits hold any unread box, and the sentences to print either way.

    The rule is *zero in `valid` and `test`* (a machine's unread boxes in `train` are cheap; the
    same boxes in `test` make the acceptance number a measurement of the annotator). Three
    outcomes, and the third is the one worth naming: a report that does not carry the field at all
    - an older build, or a hand-made set - has not passed the gate, it has not been asked.
    """
    counts = report.get("machine_only_by_split")
    if not isinstance(counts, dict):
        return False, [
            "acceptance gate: cannot verify - the merge report carries no "
            "`machine_only_by_split`, so nothing says whether a weight drew the boxes this "
            "measurement is about. Rebuild the set with build_dataset.py."
        ]
    lines: list[str] = []
    ok = True
    for split in splits:
        count = counts.get(split) or 0
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            ok = False
            lines.append(f"acceptance gate: {split} reports an unusable count ({count!r})")
            continue
        if count:
            ok = False
            lines.append(
                f"acceptance gate: {count} machine-only decision(s) in `{split}` - a weight's "
                "unread boxes make this split a measurement of the annotator, not of the model. "
                "Review them in the annotator and rebuild."
            )
    if ok:
        lines.append("acceptance gate: no machine-only decisions in " + " or ".join(splits) + " - clean")
    return ok, lines


def annotation_state_gate(report: dict) -> tuple[bool, list[str]]:
    """Whether the annotation decisions the report was built from are still the ones on disk.

    `machine_only_by_split` is a measurement of one moment, and nothing after that moment updates
    it: a person opening the annotator and saving a suggestion changes the answer without touching
    the merged set. So the report carries the identity of the `provenance.json` it counted
    (`build_dataset.annotation_state`), and this re-reads it. Three answers, all of them the same
    failure as a missing field: a report that names no file, one whose file is gone, and one whose
    file has changed - because "the counts are current" is the claim being checked, and this is the
    only half of the gate that can see a *stale* zero.

    Reads the file, so it is separate from `machine_only_gate`: that one stays a pure verdict over
    the report's own counts, and `machine_only_verdict` is what this command asks.
    """
    stamp = report.get("annotations")
    named = stamp.get("provenance") if isinstance(stamp, dict) else None
    if not isinstance(named, str) or not named:
        return False, [
            "acceptance gate: cannot verify which annotation decisions this set was built from - "
            "its merge report carries no `annotations` stamp, which is what a build made before "
            "the annotator's provenance existed (or one over a label tree with no "
            "`provenance.json`) looks like. Review the frames and rebuild with build_dataset.py."
        ]
    path = Path(named)
    try:
        raw = path.read_bytes()
    except OSError:
        return False, [
            f"acceptance gate: the annotations this set was built from are not readable at {path} "
            "- so whether the machine-only counts beside them are still true cannot be checked. "
            "Restore the annotation tree and rebuild with build_dataset.py."
        ]
    digest = stamp.get("sha256")
    if not isinstance(digest, str) or hashlib.sha256(raw).hexdigest() != digest:
        return False, [
            "acceptance gate: the annotation decisions changed after this set was built "
            f"({path} differs), so `machine_only_by_split` describes the state at build time and "
            "not the state being measured. Review the frames, then rebuild with build_dataset.py "
            "before trusting this number."
        ]
    return True, []


def machine_only_verdict(report: dict) -> tuple[bool, list[str]]:
    """Both halves of the machine-only gate: the counts, and the state they were read from.

    Ordered so a stale or unverifiable report is named before the counts of a state nobody can
    confirm - and composed here rather than in `main` so the command's answer and the tests' answer
    cannot become two judgements about the same report.
    """
    state_ok, state_lines = annotation_state_gate(report)
    counts_ok, counts_lines = machine_only_gate(report)
    return state_ok and counts_ok, state_lines + counts_lines


# ---------------------------------------------------------------------------
# The number: per-class recall, both weights, one split
# ---------------------------------------------------------------------------


def measure(
    weights: Path,
    data_yaml: Path,
    split: str,
    project: Path,
    name: str,
    yolo,
    imgsz: int,
) -> tuple[dict[str, float | None], dict[str, float]]:
    """One weight's per-class recall and aggregates on `split`.

    Both weights go through `train_model.validate` - the same pass `--val` runs - at the same
    `imgsz` on the same `data.yaml`: the comparison this command exists to make is only a
    comparison if nothing but the weights differs between the two runs.
    """
    metrics = train_model.validate(weights, data_yaml, split, project, name, yolo=yolo, imgsz=imgsz)
    rows = train_model.per_class_recall(metrics)
    return {name: recall for name, recall, _ in rows}, train_model.aggregate_metrics(metrics)


def compare(
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
    floor: float,
    tolerance: float,
) -> tuple[list[str], list[str], list[str]]:
    """(passed, failed, unmeasured) for the candidate, against the floor and against v1.

    Pure, and deliberately three outcomes rather than two: a class neither side was measured on is
    not a pass and not a miss (6's own three states), and it joins neither list - "add images of
    that item" and "add captures to that split" are opposite instructions.

    A class the candidate was measured on while the baseline was not is a pass with the reason
    attached rather than a comparison: v1 has no measurement there to lose against, and crediting
    the candidate with a win over a missing number is how a gate stops meaning anything.
    """
    passed: list[str] = []
    failed: list[str] = []
    unmeasured: list[str] = []
    for name in sorted(set(baseline) | set(candidate)):
        candidate_recall = candidate.get(name)
        if candidate_recall is None:
            unmeasured.append(f"{name}: not measured on this split (no ground-truth instances)")
            continue
        baseline_recall = baseline.get(name)
        against = "" if baseline_recall is None else f", v1 {baseline_recall:.3f}"
        if candidate_recall < floor:
            failed.append(f"{name}: {candidate_recall:.3f} < {floor:.2f} floor{against}")
            continue
        if baseline_recall is not None and candidate_recall < baseline_recall - tolerance:
            failed.append(
                f"{name}: {candidate_recall:.3f} regressed from v1's {baseline_recall:.3f} "
                f"(more than the {tolerance:.2f} tolerance)"
            )
            continue
        passed.append(
            f"{name}: {candidate_recall:.3f} >= {floor:.2f}{against or ', v1 not measured'}"
        )
    return passed, failed, unmeasured


# ---------------------------------------------------------------------------
# The claim: two or more items on a frame
# ---------------------------------------------------------------------------


def frame_instances(predict, images: list[Path], conf: float, iou: float) -> dict[str, list[str]]:
    """`{image name: [class name, ...]}`, one entry per detection above `conf`.

    The seam is `predict(image, conf, iou) -> [class name]`, so the counting is testable with no
    GPU, no weight and no frame on disk - the same reason `validate` takes an injectable `yolo`.
    No geometry passes through here: the claim is about how many *items* were found on a frame, and
    the coordinates only matter through NMS, for which `iou` is the lever.
    """
    return {image.name: list(predict(image, conf, iou)) for image in images}


def crowding(counts: dict[str, list[str]]) -> dict:
    """How many frames hold two or more items, in total and for one product.

    Per product as well as in total because the two answer different questions: "two or more items"
    is the crowded counter this dataset exists for, while "two or more of *one* product" is the
    near-duplicate case 4 warns about - the one that logs a single physical item twice, because the
    tracker sees a second instance and mints a second `track_id`. Both are reported; the verdict
    uses the first.
    """
    crowded = {name: len(items) for name, items in counts.items() if len(items) >= CROWD_MIN}
    same_product: Counter = Counter()
    for items in counts.values():
        if len(items) < CROWD_MIN:
            continue
        for name, count in Counter(items).items():
            if count >= CROWD_MIN:
                same_product[name] += 1
    return {
        "frames": len(counts),
        "crowded_frames": len(crowded),
        "crowded_names": sorted(crowded),
        "same_product_frames": dict(sorted(same_product.items())),
        "detections": sum(len(items) for items in counts.values()),
    }


def group_images(images: list[Path], distances: dict[str, str]) -> dict[str, list[Path]]:
    """The split's images, cut by the distance each was filed under. Empty buckets are dropped.

    Grouped from the *file list* rather than from a counting pass, so a caller that needs the
    frames of one distance (the sweep, which re-predicts over one row) does not have to reconstruct
    them from names. The buckets partition the list exactly - every image lands in one of them,
    `UNATTRIBUTED` included - which is what makes the per-distance rows add up to the total.
    """
    buckets: dict[str, list[Path]] = {name: [] for name in (*train_model.DISTANCE_ORDER, UNATTRIBUTED)}
    for image in images:
        buckets[distances.get(image.name) or UNATTRIBUTED].append(image)
    return {name: found for name, found in buckets.items() if found}


def crowd_by_distance(counts: dict[str, list[str]], distances: dict[str, str]) -> dict[str, dict]:
    """The same crowded-frame count, cut by the distance each frame was captured at.

    Pure regrouping of one counting pass, so a per-distance row and the total above it are the
    *same measurement over different frames* - the property `distance_breakdown` states for the
    per-distance validation passes, and the one that makes comparing the two rows worth anything.
    A frame whose distance is not recorded is counted, in `UNATTRIBUTED`: it is in the total, and a
    table that silently dropped it would not add up.
    """
    buckets: dict[str, dict[str, list[str]]] = {
        name: {} for name in (*train_model.DISTANCE_ORDER, UNATTRIBUTED)
    }
    for name, items in counts.items():
        buckets[distances.get(name) or UNATTRIBUTED][name] = items
    return {name: crowding(items) for name, items in buckets.items() if items}


def distance_claim(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    claimed: tuple[str, ...],
    split: str,
) -> tuple[list[str], list[str]]:
    """(passed, failed) for the crowding claim, one pair of sentences per claimed distance.

    Two outcomes rather than three, and the missing one is the point: `compare` can afford an
    `unmeasured` list because a class the split never asked about is nobody's fault, but here a
    claimed distance with no frames is the claim itself failing - the same reading
    `machine_only_gate` takes of a report it cannot verify. A run where `far` was not measured has
    not shown that v2 holds up at `far`; it has shown nothing about it, which is exactly the state
    this command exists to refuse to call a pass.
    """
    passed: list[str] = []
    failed: list[str] = []
    for distance in claimed:
        left, right = baseline.get(distance), candidate.get(distance)
        frames = (right or left or {}).get("frames", 0)
        if not frames:
            failed.append(
                f"crowding at `{distance}`: not measured - the {split} split files no frame at "
                "that distance, so the crowded-frame count above says nothing about it. Re-shoot "
                "the cell (docs/CAPTURE_CHECKLIST.md) and rebuild the set."
            )
            continue
        base = (left or {}).get("crowded_frames", 0)
        cand = (right or {}).get("crowded_frames", 0)
        if cand < base:
            failed.append(
                f"crowding at `{distance}`: v2 finds two or more items on {cand} of {frames} "
                f"frame(s), v1 on {base}"
            )
            continue
        passed.append(f"crowding at `{distance}`: {cand} of {frames} frame(s), v1 {base} - held")
    return passed, failed


def sweep_note(sweep: dict[float, dict]) -> str:
    """Whether the crowded-frame count is a property of the weights or of the NMS threshold.

    The one sentence `--iou-sweep` exists to produce. If the count moves with `iou`, the number is
    only meaningful with the threshold printed beside it; if it does not, the count is the model's,
    and a later change to the detector's NMS cannot rewrite this acceptance.
    """
    values = {iou: block["crowded_frames"] for iou, block in sorted(sweep.items())}
    spread = ", ".join(f"{iou:g}:{count}" for iou, count in values.items())
    if len(set(values.values())) == 1:
        return (
            f"the crowded-frame count does not move across NMS iou ({spread}) - it is a property "
            "of these weights, not of the threshold"
        )
    return (
        f"the crowded-frame count moves across NMS iou ({spread}) - so the number has to be quoted "
        "with the threshold it was measured at, and this is what says which"
    )


# ---------------------------------------------------------------------------
# The order: which product each class index means, per weight
# ---------------------------------------------------------------------------


def labels_dir(images: Path) -> Path:
    """Where one split's label files live, given its images directory.

    The rule ultralytics itself uses (`img2label_paths`: rewrite the `images` path segment to
    `labels`, replace the suffix with `.txt`), and the layout every set this tool measures was built
    in - Roboflow's export and `build_dataset`'s merged set are both `<split>/images` +
    `<split>/labels`. Stated once because the view below has to *write* its labels where the same
    rule will come looking for them.
    """
    return images.parent / "labels"


def copy_or_link(source: Path, target: Path) -> None:
    """Hardlink `source` at `target`, falling back to a copy.

    A view is a throwaway second look at frames that already exist, so it should not spend the disk
    on duplicates: a hardlink costs nothing on the same volume and is invisible to the reader. The
    fallback is for where it is not available at all (a dataset root on another drive), where a copy
    is the only thing that works and the size is the dataset's rather than the tool's.
    """
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def remap_labels(images: Path, target: Path, mapping: dict[int, int]) -> tuple[int, int]:
    """Write `images`' labels into `target` with each row's class index remapped.

    Returns `(files, rows)`. Everything but the class column is copied verbatim - the box and the
    whitespace are the drawing, and the class index is the only part of a label row that means
    something in an order. A row whose index is not in the mapping is a hard error rather than a
    dropped box: a set holding a class its own declared list does not name is broken in a way no
    remap can fix, and passing it through would produce a measurement of the wrong thing.
    """
    source = labels_dir(images)
    target.mkdir(parents=True, exist_ok=True)
    files = rows = 0
    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in train_model.IMAGE_SUFFIXES:
            continue
        label = source / f"{image.stem}.txt"
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        out: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            try:
                index = int(parts[0])
            except (IndexError, ValueError):
                raise SystemExit(f"{label} carries a label row with no class index: {line!r}")
            if index not in mapping:
                raise SystemExit(
                    f"{label} draws class {index}, which no entry of the set's declared class list "
                    "names - the labels and the class list disagree about this frame."
                )
            parts[0] = str(mapping[index])
            out.append(" ".join(parts))
            rows += 1
        (target / f"{image.stem}.txt").write_text(
            "\n".join(out) + ("\n" if out else ""), encoding="utf-8"
        )
        files += 1
    return files, rows


def order_view(
    splits: dict[str, Path],
    order: tuple[str, ...],
    set_order: tuple[str, ...],
    project: Path,
    name: str,
) -> Path:
    """A whole-set `data.yaml` whose labels index `order` - this set, in another weight's order.

    A YOLO head emits a class *index*, and ultralytics matches a prediction to a label by that
    index; a label row's index is a position in the set's declared class list. So a weight whose
    head indexes a different order shares no index with the labels except where the two permutations
    happen to agree, and it is scored under the wrong product everywhere else. That is not a
    hypothetical - it is what this command did to its own baseline before this existed, and the
    measurement is in this module's docstring. The comparison is only a comparison once each weight
    is measured in its own order, because the *names* are what the two generations share.

    Every split is written, not only the one this run measures: a view is a *set*, and one that
    answered for `test` and lied about `train` would be a trap for the next `--split`. Images are
    hardlinked and only the label column is rewritten, so the frames, the boxes and the filenames
    are the ones being compared - the mapping is the only thing that differs from the set.

    The two orders must name the *same* classes. When they do not there is no remap to make - the
    weight cannot predict one of the set's products and the set has no ground truth for one of the
    weight's - and the two lists are printed instead, because that is the thing to fix and no
    number here would mean anything until it is.
    """
    gaps = generations.class_gaps(set_order, order)
    missing = sorted(gaps.source_only)
    extra = sorted(gaps.target_only)
    if missing or extra:
        raise SystemExit(
            "this weight's class list is not a reordering of the set's, so it cannot be measured "
            "against it: it predicts \n  " + "\n  ".join(extra or ["(nothing extra)"]) +
            "\nthat the set has no ground truth for, and the set has \n  " +
            "\n  ".join(missing or ["(nothing missing)"]) +
            "\nthat it cannot predict. A comparison needs both sides to name the same products."
        )
    mapping = {index: order.index(name) for index, name in enumerate(set_order)}
    view = Path(project).expanduser() / f"{name}-order"
    body: dict = {"path": str(view.resolve()), "nc": len(order), "names": list(order)}
    for split, images in sorted(splits.items()):
        if not images.is_dir():
            continue
        target_images = view / split / "images"
        target_images.mkdir(parents=True, exist_ok=True)
        for image in sorted(images.iterdir()):
            if image.suffix.lower() in train_model.IMAGE_SUFFIXES:
                copy_or_link(image, target_images / image.name)
        remap_labels(images, view / split / "labels", mapping)
        body[split if split != "valid" else "val"] = str((view / split / "images").resolve())
    yaml = view / "data.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    import yaml as yaml_module

    yaml.write_text(yaml_module.safe_dump(body, sort_keys=False), encoding="utf-8")
    return yaml


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def weight_order(weights: Path) -> tuple[str, ...]:
    """The class order `weights`' head indexes, from the record written beside it.

    A `.pt` stores no label set, so that record is the only place this is knowable - the same fact
    `app/models.py` reads for the app's own side of the question, and the same "not recorded is not
    a guess" rule `requirement_for` follows for geometry. Read through the app's reader rather than
    by parsing the JSON here, so which values count as a usable class list stays one rule.

    A weight without one is refused rather than assumed to match the set. Assuming is the mistake
    that produced the 0.142 above, and it is *silent*: the run reports a number, no error appears
    anywhere, and the number is about the mapping rather than the model.
    """
    from app.models import read_record

    names = read_record(weights).get("class_names") or []
    if not names:
        raise SystemExit(
            f"no class order is recorded beside {weights.name}, so there is no way to know which "
            "product each of its head's outputs means - and comparing it on an assumption is what "
            "makes a model score ~0 for a reason that is not the model. Re-install it with "
            "`train_model.py --install` (or run `--val`), which writes the export's own class list "
            "beside the weight."
        )
    return tuple(names)


def recorded_imgsz(weights: Path) -> int | None:
    """The size recorded beside `weights`, or None.

    Read through `app.models.read_record` rather than by parsing the JSON here: which values count
    as a usable size is the app's rule (it is also what the settings warning compares against), and
    a second reader would be a second opinion about it.
    """
    from app.models import read_record

    return read_record(weights).get("imgsz")


def imgsz_of(weights: Path, fallback: int = train_model.IMGSZ) -> int:
    """The size to measure `weights` at: its own recorded size, else the run default."""
    return recorded_imgsz(weights) or fallback


def class_names_for(dataset: Path, generation) -> tuple[str, ...]:
    """The names the model's class indices mean, for the counting passes.

    The *dataset's* own list rather than the generation's, because that is what the export declared
    and therefore what the head indexes - `check_export` compares the two by membership, so a
    generation list in another order would relabel every counted detection without anything
    erroring. The generation's list is the fallback for a set whose names cannot be read, which
    `check_export` has already refused by the time this is called.
    """
    return tuple(train_model.read_export_names(dataset) or generation.classes)


def default_predict(yolo, weights: Path, classes: tuple[str, ...], imgsz: int):
    """`predict(image, conf, iou) -> [class name]`, backed by ultralytics' own `predict`.

    Built per weight rather than per frame: constructing `YOLO(...)` loads the checkpoint, so a
    closure that did it inside would reload it 300 times per pass. `imgsz` is the size the
    validation pass used, so a detection counted here and a recall measured there describe the same
    configuration.
    """
    model = yolo(str(weights))

    def predict(image: Path, conf: float, iou: float) -> list[str]:
        results = model.predict(str(image), conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        names: list[str] = []
        for box in getattr(results[0], "boxes", None) or []:
            index = getattr(box, "cls", None)
            try:
                value = int(index[0]) if hasattr(index, "__getitem__") else int(index)
            except (TypeError, IndexError, ValueError):
                continue
            if 0 <= value < len(classes):
                names.append(classes[value])
        return names

    return predict


def split_images(directory: Path) -> list[Path]:
    """The images in one split, sorted - the same list `check_export` counted."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in train_model.IMAGE_SUFFIXES)


def main(argv: list[str] | None = None, yolo=None, predict_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="the merged set to measure on")
    ap.add_argument("--baseline", default="", help="the weights being replaced (v1)")
    ap.add_argument("--candidate", default="", help="the weights being accepted (v2)")
    ap.add_argument("--split", default=train_model.DEFAULT_SPLIT, choices=train_model.VALIDATION_SPLITS)
    ap.add_argument("--floor", type=float, default=train_model.RECALL_FLOOR)
    ap.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--iou", type=float, default=IOU, help="NMS iou for the counting passes")
    ap.add_argument(
        "--iou-sweep",
        action="store_true",
        help="re-count the candidate at several NMS iou values, in the same per-distance rows, to "
        "say whether the crowded-frame count is a property of the weights or of the threshold",
    )
    ap.add_argument(
        "--claim-distances",
        default=",".join(CLAIM_DISTANCES),
        metavar="A,B",
        help="the distances the crowding claim is about, comma-separated, or `none` to make no "
        "per-distance claim (default: " + ",".join(CLAIM_DISTANCES) + ")",
    )
    ap.add_argument(
        "--no-crowding",
        action="store_true",
        help="skip the counting passes (two predicts over the split, plus one per sweep value)",
    )
    ap.add_argument("--imgsz", type=int, default=0, help="0 = the candidate's recorded size")
    ap.add_argument("--run-project", default=str(train_model.DEFAULT_RUN_PROJECT))
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    args = ap.parse_args(argv)

    if not args.baseline or not args.candidate:
        raise SystemExit("--baseline and --candidate are both required: this command compares two")
    # Validated before the weights, the doctor and every inference pass: a mistyped distance is an
    # argument error, and discovering it after ten minutes of GPU time would be a puzzle about the
    # dataset instead of a typo in the command line.
    claim_distances = claim_distance_list(args.claim_distances)
    baseline = Path(args.baseline).expanduser()
    candidate = Path(args.candidate).expanduser()
    for label, weights in (("baseline", baseline), ("candidate", candidate)):
        if not weights.is_file():
            # Loud rather than skipped, for `make verify-clamp`'s reason: a skip here is a silent
            # pass on the only command that says whether v2 may replace v1.
            raise SystemExit(
                f"no {label} weights at {weights} - install them first (train_model.py --install)"
            )
    dataset = Path(args.dataset).expanduser()

    generation = generations.V2
    print(f"dataset:   {dataset}")
    splits, problems = train_model.check_export(dataset, generation)
    # The doctor's read of the same set, before a single frame is measured. It is what makes the
    # number below a fact about the model: a wrong class *order*, a label row the loader drops, a
    # label drawn under a product other than the frame's own staged class and a duplicated test
    # frame all pass every other check in this file. `exports` reuses the check above
    # so the per-split lines are not printed twice - the verdict is the same read either way.
    report = dataset_doctor.check_and_report(dataset, generation, exports=(splits, problems))
    if report["errors"]:
        raise SystemExit(
            "the dataset cannot be measured as it stands - the doctor refused it above, and `make "
            "doctor` runs that check on its own"
        )

    data_yaml = train_model.write_data_yaml(dataset, splits)
    imgsz = args.imgsz or imgsz_of(candidate)
    # What the *set's* labels index. The candidate is normally trained on it, so this is also the
    # candidate's order - but the baseline's is its own generation's, and nothing about a `.pt`
    # suggests otherwise, which is why each weight is resolved separately below.
    set_order = class_names_for(dataset, generation)
    run_project = Path(args.run_project).expanduser()
    print(f"split:     {args.split}   imgsz {imgsz}   floor {args.floor:.2f}   conf {args.conf:g}")
    print(f"baseline:  {baseline}")
    print(f"candidate: {candidate}")
    orders: dict[str, dict] = {}
    measured: dict[str, tuple[tuple[str, ...], Path]] = {}
    for label, weights in (("baseline", baseline), ("candidate", candidate)):
        order = weight_order(weights)
        if list(order) == list(set_order):
            measured[label] = (order, data_yaml)
            note = "its recorded class order is the set's, so it is measured as the set stands"
        else:
            view = order_view(
                splits,
                order,
                set_order,
                run_project,
                f"{generation.run_name}-accept-{label}",
            )
            measured[label] = (order, view)
            note = (
                "its recorded class order is a reordering of the set's, so its labels were "
                f"remapped into it ({view})"
            )
        orders[label] = {"order": list(order), "data_yaml": str(measured[label][1]),
                         "remapped": measured[label][1] != data_yaml}
        print(f"  {label} class order: {len(order)} name(s) - {note}")
    print()

    summary: dict = {
        "dataset": str(dataset),
        "doctor": report,
        "split": args.split,
        "imgsz": imgsz,
        "floor": args.floor,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "conf": args.conf,
        # Which class order each weight was measured in, and whether that took a remapped view of
        # the set: the evidence behind every per-class number below, and the fact that says a
        # baseline scoring ~0 was a mapping rather than a model.
        "orders": orders,
    }

    gate_ok, gate_lines = machine_only_verdict(read_report(dataset))
    for line in gate_lines:
        print(f"  {line}")
    summary["gate"] = {"ok": gate_ok, "lines": gate_lines}

    if yolo is None:
        yolo = train_model.ultralytics_yolo()
    baseline_recall, baseline_aggregates = measure(
        baseline,
        measured["baseline"][1],
        args.split,
        run_project,
        f"{generation.run_name}-accept-baseline",
        yolo,
        imgsz,
    )
    candidate_recall, candidate_aggregates = measure(
        candidate,
        measured["candidate"][1],
        args.split,
        run_project,
        f"{generation.run_name}-accept-candidate",
        yolo,
        imgsz,
    )
    passed, failed, unmeasured = compare(
        baseline_recall, candidate_recall, args.floor, args.regression_tolerance
    )

    print()
    print(f"per-class recall on {args.split} (floor {args.floor:.2f}):")
    every = sorted(set(baseline_recall) | set(candidate_recall))
    width = max([len(name) for name in every] + [5])
    for name in every:
        left, right = baseline_recall.get(name), candidate_recall.get(name)
        mark = "!" if (right is not None and right < args.floor) else " "
        print(f"  {name:<{width}}  v1 {_cell(left)}   v2 {_cell(right)} {mark}")
    print()
    for key in ("precision", "recall", "mAP50", "mAP50-95"):
        left, right = baseline_aggregates.get(key), candidate_aggregates.get(key)
        if left is None and right is None:
            continue
        print(f"  {key:<12} v1 {_cell(left)}   v2 {_cell(right)}")
    summary["baseline"] = {"per_class": baseline_recall, "aggregates": baseline_aggregates}
    summary["candidate"] = {"per_class": candidate_recall, "aggregates": candidate_aggregates}
    summary["passed"] = passed
    summary["failed"] = failed
    summary["unmeasured"] = unmeasured

    # The gate is a failure whatever else passed: a number measured over unreviewed machine boxes
    # says something about the annotator, and accepting it would put that in the app's model.
    failures = list(failed)
    if not gate_ok:
        failures.append(
            "the machine-only gate: the numbers above are not evidence about v2 (see the gate "
            "lines at the top)"
        )

    if not args.no_crowding:
        images = split_images(splits[args.split])
        factory = predict_factory or default_predict
        # Each weight's own order names the boxes it predicts, for the reason the per-class table
        # needs `order_view`: the name of a detection is `classes[index]`, so reading v1's indices
        # through the set's list credits the wrong product in `same_product_frames` below - the
        # count of frames is the same either way, but which product it names is not.
        print()
        print(f"crowding on {args.split} (>= {CROWD_MIN} items on a frame, conf {args.conf:g}):")
        baseline_counts = frame_instances(
            factory(yolo, baseline, measured["baseline"][0], imgsz), images, args.conf, args.iou
        )
        candidate_counts = frame_instances(
            factory(yolo, candidate, measured["candidate"][0], imgsz), images, args.conf, args.iou
        )
        baseline_crowd = crowding(baseline_counts)
        candidate_crowd = crowding(candidate_counts)
        for label, block in (("v1", baseline_crowd), ("v2", candidate_crowd)):
            print(
                f"  {label} {block['crowded_frames']:>4} of {block['frames']} frame(s) with two or "
                f"more items, {block['detections']} detection(s) in total"
            )
        if candidate_crowd["same_product_frames"]:
            print(
                "  frames with two of the *same* product (the near-duplicate case 4 warns about): "
                + ", ".join(f"{name} {n}" for name, n in candidate_crowd["same_product_frames"].items())
            )
        if candidate_crowd["crowded_frames"] < baseline_crowd["crowded_frames"]:
            failures.append(
                f"crowding: v2 finds two or more items on {candidate_crowd['crowded_frames']} "
                f"frame(s), v1 on {baseline_crowd['crowded_frames']}"
            )

        # The same two passes, cut by distance. `dataset_distances` reads the *set's* own
        # `merge_report.json` first (then the staged manifest), which is what `--val`'s per-distance
        # grid does and for the same reason: the set is the artifact being measured, and the
        # directory it was built from can be gone by the time this runs.
        distances, distance_source = train_model.dataset_distances(dataset, generation.manifest)
        baseline_by = crowd_by_distance(baseline_counts, distances)
        candidate_by = crowd_by_distance(candidate_counts, distances)
        print()
        print(
            "  by distance ("
            + (distance_source or "nothing in this set records a distance")
            + "):"
        )
        rows = (*train_model.DISTANCE_ORDER, UNATTRIBUTED)
        width = max(len(name) for name in rows)
        print(f"    {'distance':<{width}}  {'frames':>6}  {'v1':>5}  {'v2':>5}")
        for name in rows:
            left, right = baseline_by.get(name), candidate_by.get(name)
            frames = (right or left or {}).get("frames", 0)
            mark = "*" if name in claim_distances else " "
            print(
                f"    {name:<{width}}  {frames:>6}  {_crowd_cell(left):>5}  "
                f"{_crowd_cell(right):>5} {mark}"
            )
        if claim_distances:
            print(f"    (* = a cell the claim is about: {', '.join(claim_distances)})")
        if not claim_distances:
            # `--claim-distances none`: the table below is a readout and nothing in it fails the run
            # - this is the run that is not re-shooting anything. Deliberately separate from the
            # case below, where a claim *was* made and the set cannot answer it.
            passed_claim, failed_claim = [], []
        elif not distances:
            # One sentence for one fact, rather than the two `mid`/`far` "not measured" lines the
            # claim would produce: nothing here records a distance *at all*, and that is the thing
            # to fix.
            passed_claim, failed_claim = [], [
                "crowding by distance: not measured - no frame in this set records a distance, so "
                "there is no `mid`/`far` row to make the claim on. The set's `merge_report.json` is "
                "what carries them; rebuild it with build_dataset.py."
            ]
        else:
            passed_claim, failed_claim = distance_claim(
                baseline_by, candidate_by, claim_distances, args.split
            )
        for line in passed_claim:
            print(f"    {line}")
        failures.extend(failed_claim)
        summary["crowding"] = {
            "iou": args.iou,
            "baseline": baseline_crowd,
            "candidate": candidate_crowd,
            "by_distance": {"v1": baseline_by, "v2": candidate_by},
            "distances_from": distance_source,
            "claim": {
                "distances": list(claim_distances),
                "passed": passed_claim,
                "failed": failed_claim,
            },
        }

        if args.iou_sweep:
            # Per distance, because the count being quoted is now a per-distance one and the sweep
            # exists to say whether *that* number is the model's. It is also no more expensive than
            # the whole-split sweep was: each pass covers the frames of one row, and the rows
            # partition the split - so three values over `mid` plus three over `far` is cheaper than
            # three over everything, and it is about the cells the claim is made on.
            by_image = group_images(images, distances)
            sweeps: dict[str, dict[float, dict]] = {}
            if claim_distances:
                for distance in claim_distances:
                    group = by_image.get(distance)
                    if not group:
                        # No sweep line for a row with no frames: `sweep_note` would read an empty
                        # count as "does not move", which is the one thing it must not say.
                        continue
                    sweeps[distance] = {
                        value: crowding(
                            frame_instances(
                                factory(yolo, candidate, measured["candidate"][0], imgsz),
                                group,
                                args.conf,
                                value,
                            )
                        )
                        for value in IOU_SWEEP
                    }
            else:
                sweeps[WHOLE_SPLIT] = {
                    value: crowding(
                        frame_instances(
                            factory(yolo, candidate, measured["candidate"][0], imgsz),
                            images,
                            args.conf,
                            value,
                        )
                    )
                    for value in IOU_SWEEP
                }
            for distance, sweep in sweeps.items():
                note = sweep_note(sweep)
                print(f"    {distance}: {note}")
            if sweeps:
                summary["iou_sweep"] = {
                    name: {f"{iou:g}": block for iou, block in sweep.items()}
                    for name, sweep in sweeps.items()
                }
                summary["iou_sweep_note"] = {
                    name: sweep_note(sweep) for name, sweep in sweeps.items()
                }

    print()
    # The verdict is `failures`, not the per-class list: the gate and the crowding claim matter as
    # much as a class under its floor, and branching on `failed` alone printed "accepted:" beside an
    # exit code of 1 whenever one of those - or, now, an unmeasured `mid`/`far` row - was the reason.
    if failures:
        print("not accepted:")
        for line in failures:
            print(f"  ! {line}")
    else:
        # Crowding is named only when it was measured: a `--no-crowding` run that printed "the
        # crowded-frame count did not fall" would be claiming the one thing it skipped.
        clauses = [
            f"every class on {args.split} clears {args.floor:.2f}",
            "nothing regressed against v1",
        ]
        if not args.no_crowding:
            where = "the whole split" + (
                " and " + ", ".join(f"`{name}`" for name in claim_distances)
                if claim_distances
                else ""
            )
            clauses.append(f"the crowded-frame count did not fall over {where}")
        print("accepted: " + ", ".join(clauses))
    for line in unmeasured:
        print(f"  - {line}")
    summary["accepted"] = not failures

    if args.json:
        print(json.dumps(summary, indent=1))
    return 0 if not failures else 1


def _cell(value: float | None) -> str:
    """One recall in the comparison table, `    -` when the split never asked about that class."""
    return "    -" if value is None else f"{value:.3f}"


def _crowd_cell(block: dict | None) -> str:
    """One crowding count in the per-distance table, `-` when that distance has no frame at all.

    A missing row and a zero are deliberately different glyphs: `0` is a measured and clean row
    (the weight found nothing to be crowded), `-` is a row nothing was asked in.
    """
    return "-" if block is None else str(block["crowded_frames"])


def claim_distance_list(raw: str) -> tuple[str, ...]:
    """`--claim-distances`'s string as a tuple of distances, refusing anything unrecognised.

    Validated rather than trusted, because a typo (`middle`) would otherwise produce a claim that
    is quietly about nothing: the unknown name has no frames, so the run would fail with a
    sentence about re-shooting a cell nobody has a folder for. The axis is named in the refusal.
    """
    names = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if names in ((), ("none",)):
        return ()
    known = (*train_model.DISTANCE_ORDER, UNATTRIBUTED)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(
            f"unknown --claim-distances value(s): {', '.join(unknown)} - the distances this set "
            f"files frames at are {', '.join(known)}, or `none` for no claim"
        )
    return names


if __name__ == "__main__":
    raise SystemExit(main())
