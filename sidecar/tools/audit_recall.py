#!/usr/bin/env python
"""Per-instance recall for a trained weight, split by how crowded the frame was.

`train_model.py --val` reports per-class and per-distance recall, and that is the number
MODEL_TRAINING.md 8.3 asks for. This tool answers a question that number cannot: **when the
model misses an object, is it missing the object or the second one?** Those are different
defects with different fixes, and a recall figure averages them into one number.

Take v1's `555 sardines 155grams` at 0.742 against the 0.85 floor. Read as "the model often
cannot see sardines" it argues for more sardine shots. Read per-instance it says something
else entirely: no frame containing a sardine came back empty, and every miss was an
additional tin in a frame that already had one. Same number, opposite conclusion - which is
why the split below exists.

Four things make this not a re-run of `--val`:

**Per-instance matching, not per-frame.** Ultralytics' validator matches class-aware by IoU
and reports mAP and mean recall; its threshold is chosen per class at best F1. This matches
the same way - class-aware, best pair first, one prediction per label - but at *the app's*
operating point (`Settings.conf_threshold`, 0.5), so the recall here is the recall a Live
view actually experiences rather than the best one available.

**Crowding as an axis.** Every frame is bucketed by how many instances its labels carry, and
the instance-level recall of the two buckets is printed separately. Distance is the axis the
dataset was *built* around (v2's close/mid/far); crowding is the axis it turns out to matter
on, and unlike distance it comes free from any export's labels - no tags, no manifest.

**Predicted rather than tracked.** The app's detector runs `model.track()`, because a live
capture needs stable ids. An audit is not a live capture: tracking carries state from one
image to the next, so a per-image number would depend on the order the images were visited
and would quietly punish whichever frame happened to be read first. This runs `predict()`,
one frame at a time, and reads boxes in the same stretch geometry the app feeds the model
(`YoloDetector.infer`'s `_stretch` branch) - so the *input* is the app's and only the
bookkeeping is per-frame.

**Polygon-aware labels.** Roboflow exports for this project are mostly polygons, not boxes:
`cls x1 y1 x2 y2 x3 y3 ...` rather than `cls cx cy w h`. Ultralytics detects the longer form
and converts each polygon with `segments2boxes` before training, so the training was always
correct. Anything reading a label file directly has to do the same conversion - a reader that
assumes `cls cx cy w h` finds `fields[3] * fields[4]` and reports an "area" computed from two
polygon coordinates, which is how a working dataset gets misdiagnosed as 484 zero-area
labels. `parse_labels` below handles both forms, and the tests pin both.

Two sweeps, for the two ways a miss can be an inference setting rather than a model gap:

    --conf-sweep    a miss ranked *below* conf is a threshold, and `conf_threshold` is
                    hot-reloadable - so the finding is a slider, not a retrain.
    --iou-sweep     two tins side by side overlap; NMS at its default 0.7 merges boxes that
                    overlap more than that, so a genuinely detected second tin is discarded
                    inside the model before anything sees it.

The conf sweep costs nothing extra: NMS sorts by confidence descending and only ever lets a
higher-confidence box suppress a lower one, so running once at the lowest threshold and
filtering afterwards yields exactly the boxes a run at any higher threshold would. The NMS
sweep does re-run, because that merge happens inside the model.

Resource envelope: see resources.py. The caps the other dataset tools take are honoured here
too, and this pass is cheap - it is inference over a few hundred images, not a decode of
thousands of 12 MP frames.

    python sidecar/tools/audit_recall.py --generation v1
    python sidecar/tools/audit_recall.py --generation v1 --conf-sweep --iou-sweep
    python sidecar/tools/audit_recall.py --weights runs/scanncart-grocery-v2/weights/best.pt

Reads the ignored dataset workspace; needs no camera, no network and no API key.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import resources  # must precede numpy/torch: sets OMP/MKL thread limits
from generations import DEFAULT, GENERATIONS, Generation
from generations import get as generation_for
from workspace import SIDECAR_ROOT

DEFAULT_SPLIT = "test"
# Settings.conf_threshold's default: the threshold the Live view actually runs at.
DEFAULT_CONF = 0.5
DEFAULT_IMGSZ = 640
# What counts as having found a labelled instance. 0.5 is ultralytics' own mAP threshold, so a
# number here is comparable with a number from --val rather than being a third convention.
DEFAULT_IOU_MATCH = 0.5
# The floors MODEL_TRAINING.md 8.3 sets, applied per class here: a class under it is flagged
# rather than silently averaged into a passing total.
RECALL_FLOOR = 0.85
# How many labelled instances a crowding bucket needs before the two are worth comparing. A gap
# computed off a handful of instances is noise that reads as a finding, and `--limit` is the
# usual way to land there - so the verdict refuses rather than asserting something it cannot
# support. 20 is not derived from anything: it is large enough that one or two instances moving
# cannot flip the comparison.
MIN_BUCKET_INSTANCES = 20

# The buckets every frame is sorted into. Named rather than a boolean because the report and
# the tests both print them, and "single" / "multi" is what the finding is stated in.
SINGLE = "single"
MULTI = "multi"
BUCKETS = (SINGLE, MULTI)

# What a sweep measures when asked for one. The conf list brackets the app's 0.5 on both sides
# so the slope is visible; the iou list runs from a stricter merge than the default to nearly
# no merge at all, because the question is only whether boxes come *back*.
CONF_SWEEP = (0.1, 0.2, 0.3, 0.5)
IOU_SWEEP = (0.5, 0.7, 0.9)

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Instance:
    """One labelled object: its class index and its box in 0-1-relative xyxy."""

    cls: int
    box: Box


@dataclass(frozen=True)
class Prediction:
    """One predicted object, carrying its confidence so a threshold can be applied later."""

    cls: int
    box: Box
    conf: float


@dataclass(frozen=True)
class FrameRecord:
    """One image's labels and everything the model predicted on it.

    Kept together rather than accumulated straight into a report because the report is a
    function of the *threshold* as well: one inference pass at the lowest conf answers every
    conf in the sweep, so long as the raw predictions survive to be re-filtered.
    """

    truth: tuple[Instance, ...]
    preds: tuple[Prediction, ...]
    # Predicted class names this generation does not declare. A weight whose head has more
    # outputs than the dataset it is audited against shows up here.
    off_roster: tuple[str, ...] = ()


@dataclass(frozen=True)
class Match:
    """Which labelled instances were found, which were not, and which predictions were used."""

    matched_truth: tuple[int, ...]
    missed_truth: tuple[int, ...]
    matched_pred: tuple[int, ...]

    @property
    def spurious(self) -> int:
        """Predictions left over: matched nothing, so they are false positives."""
        return len(self.matched_pred)


# ---------------------------------------------------------------------------
# The pure layer: labels, geometry, matching - no torch, no cv2, no filesystem
# ---------------------------------------------------------------------------


def parse_labels(text: str) -> list[Instance]:
    """Every instance in one YOLO label file, as a normalized xyxy box.

    **Two forms, and the field count is what tells them apart**:

        cls cx cy w h                  exactly 5 - a box, centre plus size (YOLO's own format)
        cls x1 y1 x2 y2 x3 y3 ...      7 or more, even - a polygon, reduced to its bounding box

    They need different maths and neither is optional. A box needs `cx ± w/2`, `cy ± h/2`; a
    polygon needs `min`/`max` over the two strided halves of its vertices. Applying the polygon
    rule to a box is *not* a near miss: `min(cx, w)` on a centred object yields a small box in
    the wrong place - a real instance at `1 0.508 0.578 0.984 0.844` comes out as x 0.508-0.984
    instead of 0.016-1.0, and when `w < cx` it degenerates to zero width entirely. That misreading
    is invisible in code review (both branches "look like" box maths), makes a class with
    box-format labels look far worse than it is, and produced a spurious report of 484 zero-area
    labels on a dataset that was fine. Exactly one of the two forms is right, and the count says
    which - 5 is a box, 6 cannot be either (a polygon needs at least three points), and no export
    here contains a 6-field line.

    Lines with fewer than five fields, or unparseable numbers, are skipped rather than defaulted:
    a malformed line is not an instance, and guessing one would inflate the denominator.
    """
    out: list[Instance] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(coords) == 4:
            cx, cy, w, h = coords
            out.append(Instance(cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
            continue
        xs, ys = coords[0::2], coords[1::2]
        if not xs or not ys:
            continue
        out.append(Instance(cls, (min(xs), min(ys), max(xs), max(ys))))
    return out


def read_labels(path: Path) -> list[Instance]:
    """`parse_labels` for a file that may not exist - an unlabelled image is not an error."""
    if not path.exists():
        return []
    return parse_labels(path.read_text(encoding="utf-8"))


def area(box: Box) -> float:
    """The box's area, in the same relative units the boxes are in."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def match_instances(
    truth: tuple[Instance, ...] | list[Instance],
    preds: tuple[Prediction, ...] | list[Prediction],
    iou_match: float = DEFAULT_IOU_MATCH,
) -> Match:
    """Greedy class-aware one-to-one matching, best pair first.

    Three properties, each of which a cheaper version gets wrong in a way that reads as a
    model failure:

    **Class-aware.** A confident box of the wrong class is not a hit. Only equal class indices
    are paired, so a tin mistaken for another tin counts as one miss and one false positive
    rather than as a success.

    **One-to-one.** Two predictions on the same labelled object are one hit and one false
    positive, not two hits. Without the "already used" sets, a model that emits duplicate
    boxes would score *better* than one that emits a single correct box.

    **Best pair first.** Pairs are consumed in descending IoU, so a prediction that overlaps
    two labels goes to the one it matches better instead of to whichever label was listed
    first - which is what makes the result independent of label ordering.
    """
    scored: list[tuple[float, int, int]] = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(preds):
            if t.cls != p.cls:
                continue
            score = iou(t.box, p.box)
            if score >= iou_match:
                scored.append((score, ti, pi))
    scored.sort(reverse=True)

    used_t: set[int] = set()
    used_p: set[int] = set()
    for _, ti, pi in scored:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
    return Match(
        matched_truth=tuple(sorted(used_t)),
        missed_truth=tuple(i for i in range(len(truth)) if i not in used_t),
        matched_pred=tuple(sorted(used_p)),
    )


@dataclass
class ClassTally:
    """One class's instance count, hits, and the box areas on each side of the split.

    The areas are the tell between "cannot see the object" and "cannot see the second one":
    a missed instance half the area of the ones it is scored against is a small-object
    problem, and the same areas on both sides means size is not what separates them.
    """

    instances: int = 0
    found: int = 0
    hit_areas: list[float] = field(default_factory=list)
    miss_areas: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.found / self.instances if self.instances else 0.0

    @property
    def median_hit_area(self) -> float:
        return statistics.median(self.hit_areas) if self.hit_areas else 0.0

    @property
    def median_miss_area(self) -> float:
        return statistics.median(self.miss_areas) if self.miss_areas else 0.0


@dataclass
class BucketTally:
    """One crowding bucket's instance count and how often a frame in it came back clean."""

    labelled: int = 0
    found: int = 0
    frames: int = 0
    frames_clean: int = 0

    @property
    def recall(self) -> float:
        return self.found / self.labelled if self.labelled else 0.0

    @property
    def frame_rate(self) -> float:
        return self.frames_clean / self.frames if self.frames else 0.0


@dataclass
class RecallReport:
    """Per-class and per-crowding recall, accumulated frame by frame at one threshold.

    Deliberately a plain accumulator with no model in it: every number the tool prints is
    produced here, so the whole accounting layer - matching, bucketing, areas - is tested
    against hand-built frames with no weights, no GPU and no dataset on disk. The tool's only
    untested part is the inference loop that fills the `FrameRecord`s.
    """

    classes: tuple[str, ...]
    conf: float = DEFAULT_CONF
    iou_match: float = DEFAULT_IOU_MATCH
    per_class: dict[int, ClassTally] = field(default_factory=dict)
    buckets: dict[str, BucketTally] = field(default_factory=dict)
    off_roster: Counter[str] = field(default_factory=Counter)
    frames: int = 0

    def __post_init__(self) -> None:
        self.per_class = {i: ClassTally() for i in range(len(self.classes))}
        self.buckets = {name: BucketTally() for name in BUCKETS}

    def add(self, record: FrameRecord) -> Match:
        """Fold one frame in, and return what matched - so a caller can inspect a frame.

        Predictions below `conf` are dropped first, which is what makes the conf sweep free:
        the same records re-reported at a higher threshold are the same boxes a run at that
        threshold would have kept (NMS sorts by confidence descending and only ever lets a
        higher-confidence box suppress a lower one).
        """
        preds = [p for p in record.preds if p.conf >= self.conf]
        match = match_instances(record.truth, preds, self.iou_match)

        self.frames += 1
        bucket = self.buckets[SINGLE if len(record.truth) == 1 else MULTI]
        bucket.frames += 1
        bucket.labelled += len(record.truth)
        bucket.found += len(match.matched_truth)
        if not match.missed_truth:
            bucket.frames_clean += 1

        hit = set(match.matched_truth)
        for i, instance in enumerate(record.truth):
            tally = self.per_class[instance.cls]
            tally.instances += 1
            if i in hit:
                tally.found += 1
                tally.hit_areas.append(area(instance.box))
            else:
                tally.miss_areas.append(area(instance.box))
        for name in record.off_roster:
            self.off_roster[name] += 1
        return match

    @property
    def instances(self) -> int:
        return sum(t.instances for t in self.per_class.values())

    @property
    def found(self) -> int:
        return sum(t.found for t in self.per_class.values())

    @property
    def recall(self) -> float:
        return self.found / self.instances if self.instances else 0.0

    @property
    def below_floor(self) -> list[str]:
        """Class names under the floor, worst first - the shortlist, not the whole table."""
        under = [
            (self.classes[i], t)
            for i, t in self.per_class.items()
            if t.instances and t.recall < RECALL_FLOOR
        ]
        return [name for name, _ in sorted(under, key=lambda pair: pair[1].recall)]


def measure(
    records: list[FrameRecord] | tuple[FrameRecord, ...],
    classes: tuple[str, ...],
    conf: float = DEFAULT_CONF,
    iou_match: float = DEFAULT_IOU_MATCH,
) -> RecallReport:
    """Build a report from already-collected predictions, at one threshold.

    Pure and total: the same records can be measured at every conf in a sweep, which is the
    whole reason the runner collects before it reports.
    """
    report = RecallReport(classes=classes, conf=conf, iou_match=iou_match)
    for record in records:
        report.add(record)
    return report


# ---------------------------------------------------------------------------
# The impure layer: files, frames, and the model
# ---------------------------------------------------------------------------


def split_dirs(generation: Generation, split: str) -> tuple[Path, Path]:
    """Where one split's images and labels live in a YOLO export."""
    root = generation.export_dir / split
    return root / "images", root / "labels"


def frame_paths(generation: Generation, split: str, limit: int | None = None) -> list[Path]:
    """Every labelled image in the split, in a fixed order.

    Sorted so two runs visit the frames identically - a report that depends on directory
    order is not reproducible, and "same tool, same numbers" is the point of tracking this.
    """
    images, labels = split_dirs(generation, split)
    if not images.is_dir():
        raise SystemExit(f"no such split: {images} (generate or download the export first)")
    paths = [p for p in sorted(images.glob("*.jpg")) if (labels / f"{p.stem}.txt").exists()]
    return paths[:limit] if limit else paths


def load_records_only(
    generation: Generation, split: str, limit: int | None = None
) -> list[tuple[Path, tuple[Instance, ...]]]:
    """Each frame's path and labels, without inferences - useful for a dry count."""
    images, labels = split_dirs(generation, split)
    out = []
    for path in frame_paths(generation, split, limit):
        out.append((path, tuple(read_labels(labels / f"{path.stem}.txt"))))
    return out


def prepare(frame, imgsz: int, resize_mode: str):
    """The frame as the model should see it - the app's own convention, not a second one.

    For `stretch` this is `YoloDetector.infer`'s branch: square the frame before predict, so
    ultralytics' letterbox adds no padding and the object arrives at its training scale.
    Boxes need no un-warping because they are read in relative coordinates, where a per-axis
    scale cancels out.
    """
    import cv2

    if resize_mode == "stretch":
        return cv2.resize(frame, (imgsz, imgsz))
    return frame


def collect(
    generation: Generation,
    weights: str,
    split: str,
    conf: float,
    imgsz: int,
    resize_mode: str,
    device: str,
    limit: int | None = None,
) -> list[FrameRecord]:
    """One inference pass: every frame's labels, predictions, and off-roster names.

    Predict, not track - see the module docstring. `conf` should be the *lowest* threshold
    wanted, since higher ones are applied later by `measure`.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    wanted = {name: i for i, name in enumerate(generation.classes)}
    images, labels = split_dirs(generation, split)
    records: list[FrameRecord] = []
    for path in frame_paths(generation, split, limit):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = model.predict(
            prepare(frame, imgsz, resize_mode),
            conf=conf,
            imgsz=imgsz,
            device=_device_arg(device),
            verbose=False,
        )[0]
        names = result.names
        boxes = result.boxes
        preds: list[Prediction] = []
        off: list[str] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            clss = boxes.cls.tolist()
            h, w = result.orig_shape if hasattr(result, "orig_shape") else frame.shape[:2]
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                name = names[int(k)]
                box = (x1 / w, y1 / h, x2 / w, y2 / h)
                if name in wanted:
                    preds.append(Prediction(wanted[name], box, float(c)))
                else:
                    off.append(name)
        records.append(
            FrameRecord(truth=tuple(read_labels(labels / f"{path.stem}.txt")),
                        preds=tuple(preds), off_roster=tuple(off))
        )
    return records


def _device_arg(device: str):
    """`resources.resolve_device`'s form for ultralytics, which wants an index not a name."""
    if device.isdigit():
        return int(device)
    return 0 if device.startswith("cuda") else device


def device_label(device: str) -> str:
    """How the device reads in the report.

    `resolve_device("auto")` returns `"0"` - ultralytics' index - which is not a name anything
    else in this project uses, and `device 0` next to `conf>=0.5` reads as a second number.
    `cuda:0` is what the app reports and what `torch.cuda` answers, so the report says that.
    """
    return f"cuda:{device}" if device.isdigit() else device


def sweep_iou(
    generation: Generation,
    weights: str,
    split: str,
    conf: float,
    iou_match: float,
    imgsz: int,
    resize_mode: str,
    device: str,
    ious: tuple[float, ...],
    limit: int | None = None,
) -> list[tuple[float, RecallReport]]:
    """Re-run the pass per NMS IoU and report each - the merge happens inside the model.

    `predict(iou=...)` is ultralytics' NMS threshold, a different number from `iou_match`
    (which decides whether a surviving box is a hit). Leaving it alone would be simpler, but
    the question "were these instances suppressed or unseen" has no other answer: the model
    returns the post-NMS list, so anything merged away is gone before this file sees it.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    wanted = {name: i for i, name in enumerate(generation.classes)}
    images, labels = split_dirs(generation, split)
    paths = frame_paths(generation, split, limit)

    frames = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frames.append((path, frame))

    out: list[tuple[float, RecallReport]] = []
    for iou in ious:
        records: list[FrameRecord] = []
        for path, frame in frames:
            result = model.predict(
                prepare(frame, imgsz, resize_mode),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=_device_arg(device),
                verbose=False,
            )[0]
            names = result.names
            boxes = result.boxes
            preds: list[Prediction] = []
            if boxes is not None and len(boxes):
                h, w = result.orig_shape if hasattr(result, "orig_shape") else frame.shape[:2]
                for (x1, y1, x2, y2), c, k in zip(
                    boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
                ):
                    name = names[int(k)]
                    if name in wanted:
                        preds.append(
                            Prediction(wanted[name], (x1 / w, y1 / h, x2 / w, y2 / h), float(c))
                        )
            records.append(
                FrameRecord(truth=tuple(read_labels(labels / f"{path.stem}.txt")),
                            preds=tuple(preds))
            )
        out.append((iou, measure(records, generation.classes, conf, iou_match)))
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def report_lines(
    report: RecallReport,
    *,
    weights: str,
    generation: Generation,
    split: str,
    resize_mode: str,
    device: str,
) -> list[str]:
    """The whole reading as lines, so the CLI is a print and the tests can read it."""
    out = [
        f"weights    {weights}",
        f"dataset    {generation.export_dir}",
        f"split      {split}   generation {generation.name}   {len(report.classes)} classes",
        f"preproc    {resize_mode}   device {device_label(device)}",
        f"conf>={report.conf}   iou_match>={report.iou_match}   frames {report.frames}",
        "",
        f"{'class':<48}{'inst':>6}{'found':>7}{'recall':>8}"
        f"{'med hit':>10}{'med miss':>10}",
    ]
    for i, name in enumerate(report.classes):
        tally = report.per_class[i]
        if not tally.instances:
            continue
        flag = "   <-- below 0.85" if tally.recall < RECALL_FLOOR else ""
        out.append(
            f"{name:<48}{tally.instances:6}{tally.found:7}{tally.recall:8.3f}"
            f"{tally.median_hit_area:10.4f}{tally.median_miss_area:10.4f}{flag}"
        )
    out.append(
        f"{'TOTAL':<48}{report.instances:6}{report.found:7}{report.recall:8.3f}"
    )
    out.append("")
    out.append("crowding: instances found / labelled, and frames that came back complete")
    for name in BUCKETS:
        bucket = report.buckets[name]
        label = "1 object" if name == SINGLE else "2+ objects"
        out.append(
            f"{name:<8}{label:<12}{bucket.found:5}/{bucket.labelled:<5}"
            f"{bucket.recall:7.1%}   frames {bucket.frames_clean}/{bucket.frames}"
            f"  {bucket.frame_rate:6.1%}"
        )
    if report.off_roster:
        out.append("")
        out.append("off-roster predictions (not in this generation's class list):")
        for name, n in report.off_roster.most_common():
            out.append(f"  {name:<48}{n:6}")
    else:
        out.append("")
        out.append("off-roster predictions: none")
    if report.below_floor:
        out.append("")
        out.append("below the floor: " + ", ".join(report.below_floor))
    return out


def crowding_verdict(report: RecallReport, minimum: int = MIN_BUCKET_INSTANCES) -> str:
    """The one-sentence read, which is the thing a number alone cannot give.

    Stated as a comparison rather than a threshold: an absolute figure for the crowded bucket
    would be a second floor nobody agreed on, while the *gap* between the two buckets is a
    property of the model that either is there or is not.

    Three readings, one refusal. The refusal matters as much as the others: a bucket holding a
    couple of instances produces a ratio that is all noise, and printing it as a verdict is how
    a `--limit 20` smoke test gets mistaken for a result. The reverse case is not folded into
    "no gap" either - a crowded bucket that scores *better* is a finding about solo shots, and
    calling it "no gap" would hide the direction.
    """
    single, multi = report.buckets[SINGLE], report.buckets[MULTI]
    thin = [name for name, b in ((SINGLE, single), (MULTI, multi)) if b.labelled < minimum]
    if thin:
        return (
            f"Not enough to compare yet: {single.labelled} single / {multi.labelled} crowded "
            f"instances, and the {' and '.join(thin)} bucket is under {minimum}. Re-run without "
            "--limit before reading the crowding split."
        )
    gap = single.recall - multi.recall
    if gap > 0.05:
        return (
            f"Crowding gap of {gap:.1%} ({single.recall:.1%} single vs {multi.recall:.1%} "
            "crowded): instances are found one at a time but lost when several share a frame, "
            "so the fix is more multi-item scenes rather than more solo shots."
        )
    if gap < -0.05:
        return (
            f"The crowded bucket scores *higher* ({multi.recall:.1%} vs {single.recall:.1%} "
            "single): solo objects are the ones being missed, which points at small or isolated "
            "objects rather than at crowded scenes."
        )
    return (
        f"No crowding gap to speak of ({single.recall:.1%} single vs {multi.recall:.1%} "
        "crowded): the misses are spread across frames, so the fix is more shots of the "
        "classes below rather than more multi-item scenes."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-instance recall for a trained weight, split by frame crowding."
    )
    ap.add_argument(
        "--generation", choices=sorted(GENERATIONS), default=DEFAULT.name,
        help="whose dataset and class list to measure against",
    )
    ap.add_argument(
        "--weights", default=None,
        help="default: models/<generation>.pt as installed; pass runs/<run>/weights/best.pt "
             "to measure a run that has not been installed",
    )
    ap.add_argument("--split", default=DEFAULT_SPLIT, help="train / valid / test")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="operating confidence")
    ap.add_argument("--iou-match", type=float, default=DEFAULT_IOU_MATCH,
                    help="IoU at which a prediction counts as finding a label")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument(
        "--resize-mode", default=None, choices=("stretch", "letterbox"),
        help="default: the generation's own (models.py records it on install)",
    )
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--conf-sweep", action="store_true",
                    help="report recall at several confidence thresholds")
    ap.add_argument("--iou-sweep", action="store_true",
                    help="re-run at several NMS IoUs, to tell a suppressed box from an unseen one")
    # The machine's share, in the same flags the other dataset tools use - see resources.py.
    ap.add_argument("--max-use-percent", type=int, default=resources.USE_PERCENT)
    ap.add_argument("--max-vram-percent", type=int, default=resources.VRAM_PERCENT)
    ap.add_argument("--disk-reserve-gb", type=float, default=resources.DISK_RESERVE_GB)
    ap.add_argument("--hard-vram-cap", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generation = generation_for(args.generation)
    weights = args.weights or str(SIDECAR_ROOT / "models" / generation.weight_name)
    resize_mode = args.resize_mode or generation.resize_mode

    if not Path(weights).exists():
        raise SystemExit(f"no such weight: {weights}")

    budget = resources.measure(args.max_use_percent, args.disk_reserve_gb, args.max_vram_percent)
    device = resources.resolve_device(args.device)
    budget = resources.apply(budget, device, args.hard_vram_cap)

    # One pass at the lowest threshold anyone asked about: every other conf is a filter over
    # these same records (see RecallReport.add). Only the NMS sweep needs its own pass.
    lowest = min([args.conf, *(CONF_SWEEP if args.conf_sweep else ())])
    records = collect(
        generation, weights, args.split, lowest, args.imgsz, resize_mode, device, args.limit
    )
    report = measure(records, generation.classes, args.conf, args.iou_match)

    for line in report_lines(
        report, weights=weights, generation=generation, split=args.split,
        resize_mode=resize_mode, device=device,
    ):
        print(line)
    print()
    print(crowding_verdict(report))

    if args.conf_sweep:
        print()
        print("confidence sweep (instances found / labelled, then the crowded bucket):")
        print(f"{'conf':>6}{'instances':>12}{'recall':>9}{'single':>9}{'multi':>9}")
        for conf in CONF_SWEEP:
            r = measure(records, generation.classes, conf, args.iou_match)
            print(
                f"{conf:6.2f}{r.found:6}/{r.instances:<5}{r.recall:9.3f}"
                f"{r.buckets[SINGLE].recall:9.1%}{r.buckets[MULTI].recall:9.1%}"
            )
        print()
        print("A rise here means the misses are ranked below the threshold, and conf_threshold")
        print("is hot-reloadable - so that part of the recall is a setting, not the model.")

    if args.iou_sweep:
        print()
        print("NMS IoU sweep (the merge happens inside the model, hence a second pass):")
        print(f"{'iou':>6}{'instances':>12}{'recall':>9}{'single':>9}{'multi':>9}")
        for iou, r in sweep_iou(
            generation, weights, args.split, args.conf, args.iou_match, args.imgsz,
            resize_mode, device, IOU_SWEEP, args.limit,
        ):
            print(
                f"{iou:6.2f}{r.found:6}/{r.instances:<5}{r.recall:9.3f}"
                f"{r.buckets[SINGLE].recall:9.1%}{r.buckets[MULTI].recall:9.1%}"
            )
        print()
        print("A large recovery means boxes were detected and then merged away, so that share")
        print("of a crowded-frame recall is an inference setting rather than a model gap.")

    print()
    print(f"resource envelope: {budget.cpu_threads} threads, RAM cap {budget.ram_cap_gb:.1f} GB, "
          f"VRAM cap {budget.vram_cap_gb:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
