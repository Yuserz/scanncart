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
   argued. `--iou-sweep` re-counts the candidate at several NMS iou values and says whether that
   count moved: if it does, the bucket is a property of the threshold and has to be quoted with
   it; if it does not, the count is the model's. Nothing is auto-tuned - the sweep reports.

Both weights are required and read from disk, not guessed: a missing file is a hard error before
any measurement, the same way `make verify-clamp` refuses to skip. Each pass is `train_model`'s
own `validate`, so this command cannot measure something the documented sequence would not.

    sidecar/.venv/Scripts/python.exe sidecar/tools/accept_v2.py \\
        --baseline sidecar/models/scanncart-grocery-v1.pt \\
        --candidate sidecar/models/scanncart-grocery-v2.pt

    # without the counting passes (two predicts over the split, plus one per sweep value)
    sidecar/.venv/Scripts/python.exe sidecar/tools/accept_v2.py --no-crowding --baseline … --candidate …

    # the same verdict as JSON, for a script
    sidecar/.venv/Scripts/python.exe sidecar/tools/accept_v2.py --json --baseline … --candidate …
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
# Wiring
# ---------------------------------------------------------------------------


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
        help="re-count the candidate at several NMS iou values, to say whether the crowded-frame "
        "count is a property of the weights or of the threshold",
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
    print(f"split:     {args.split}   imgsz {imgsz}   floor {args.floor:.2f}   conf {args.conf:g}")
    print(f"baseline:  {baseline}")
    print(f"candidate: {candidate}")
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
    }

    gate_ok, gate_lines = machine_only_verdict(read_report(dataset))
    for line in gate_lines:
        print(f"  {line}")
    summary["gate"] = {"ok": gate_ok, "lines": gate_lines}

    if yolo is None:
        yolo = train_model.ultralytics_yolo()
    run_project = Path(args.run_project).expanduser()
    baseline_recall, baseline_aggregates = measure(
        baseline,
        data_yaml,
        args.split,
        run_project,
        f"{generation.run_name}-accept-baseline",
        yolo,
        imgsz,
    )
    candidate_recall, candidate_aggregates = measure(
        candidate,
        data_yaml,
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
        failures.append("the machine-only gate (see above)")

    if not args.no_crowding:
        images = split_images(splits[args.split])
        factory = predict_factory or default_predict
        classes = class_names_for(dataset, generation)
        print()
        print(f"crowding on {args.split} (>= {CROWD_MIN} items on a frame, conf {args.conf:g}):")
        baseline_crowd = crowding(
            frame_instances(factory(yolo, baseline, classes, imgsz), images, args.conf, args.iou)
        )
        candidate_crowd = crowding(
            frame_instances(factory(yolo, candidate, classes, imgsz), images, args.conf, args.iou)
        )
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
        summary["crowding"] = {"iou": args.iou, "baseline": baseline_crowd, "candidate": candidate_crowd}
        if args.iou_sweep:
            sweep = {
                value: crowding(
                    frame_instances(factory(yolo, candidate, classes, imgsz), images, args.conf, value)
                )
                for value in IOU_SWEEP
            }
            note = sweep_note(sweep)
            print(f"  {note}")
            summary["iou_sweep"] = {f"{iou:g}": block for iou, block in sweep.items()}
            summary["iou_sweep_note"] = note

    print()
    if failed:
        print("not accepted:")
        for line in failed:
            print(f"  ! {line}")
    else:
        print(
            f"accepted: every class on {args.split} clears {args.floor:.2f}, nothing regressed "
            "against v1, and the crowded-frame count did not fall"
        )
    for line in unmeasured:
        print(f"  - {line}")
    if not gate_ok:
        print("  ! the machine-only gate failed - the numbers above are not evidence about v2")
    summary["accepted"] = not failures

    if args.json:
        print(json.dumps(summary, indent=1))
    return 0 if not failures else 1


def _cell(value: float | None) -> str:
    """One recall in the comparison table, `    -` when the split never asked about that class."""
    return "    -" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
