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
                    hot-reloadable - so the finding is a slider, not a retrain. It reports
                    what each threshold costs on *both* sides, because recall alone can only
                    ever improve as the threshold falls: alongside the instance counts it
                    prints `extra`, the predictions that matched no label. Both are split by
                    crowding and, when the generation has a distance axis, by distance - a
                    threshold that buys `far` while costing `close` is a different decision
                    from one that buys both.
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
    # a refreshed dataset, measured beside the frozen export it was built on
    python sidecar/tools/audit_recall.py --generation v1 \\
        --dataset-dir sidecar/data/datasets/export-v1-s2 \\
        --manifest sidecar/data/datasets/cleaned-v1-s2/manifest.json
    # the labels' own size distribution - no weights, no GPU, no model download
    python sidecar/tools/audit_recall.py --generation v1 --size-histogram

Reads the ignored dataset workspace; needs no camera, no network and no API key.

**Two of this tool's inputs are per-dataset rather than per-weight** - the export the labels are
read from, and the manifest each frame's distance is joined on - and both are named by the
*generation spec*. That is right for the generation the app ships and wrong for the one being
built: a refresh of v1's set is staged and exported *beside* the frozen `scanncart-grocery-v1/`
export, which has to stay untouched as the "before" baseline. Without an override the tool that
decides acceptance could only read the dataset the spec names, so the after number would be
unmeasurable - and, with v1's manifest pinned to `None`, the per-distance columns would vanish
from the comparison that the recapture work exists to produce. `--dataset-dir` and `--manifest`
are the pair `train_model.py` already takes, spelled the same way on purpose, so one command line
transfers between the two tools.

**`--size-histogram` is the one mode here that reads the labels rather than the weights.** It
histograms every labelled box's width per class and turns the result into a shoot list, and it
exists because v1's set predates the distance tags (`generations.V1.manifest is None`): "which
classes have no small instances" cannot be read off the tags, and a capture plan built from folder
names is a guess. It resolves no weight at all - the question is what to shoot next, which is
asked before there is a refreshed weight to measure.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import resources  # must precede numpy/torch: sets OMP/MKL thread limits
from generations import DEFAULT, GENERATIONS, Generation
from generations import get as generation_for
# The distance join and its ordering are imported rather than re-derived: `train_model` already
# reads the manifest on filenames (`distance_map`) and owns `DISTANCE_ORDER`, and a second copy is
# how the two tools' `far` comes to mean different files. It is a light module - `httpx` and these
# same helpers, nothing that loads a model - so importing it costs this tool nothing at start-up.
from train_model import DISTANCE_ORDER, distance_map
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

# The size bands, in pixels of the capture preview - the unit the Live view's `det-size` readout
# prints, so an operator can check the first frame of a cell against this table while shooting.
# The thresholds are the recapture plan's shoot bands
# (`docs/superpowers/plans/2026-09-26-v1-varied-size-recapture.md` section 4): at the rig's
# 1280-wide capture they are the close / mid / far positions the tape marks stand for. A box is
# measured as a *share* of frame width and multiplied back by the capture width, which is what
# makes the reading resolution-independent - a wrong `--capture-width` scales every row together
# and can reorder nothing.
CLOSE_MIN_PX = 300
MID_MIN_PX = 120
FAR_MIN_PX = 40
DEFAULT_CAPTURE_WIDTH = 1280
# Below FAR_MIN_PX is counted, and as its own band rather than folded into `small`. A 30 px sachet
# and a 100 px tin are the same item to a threshold and different problems to a shoot list: one is
# further away than the far mark, the other is a small package at it. Folding them together hides
# the tail, which is the half a shoot list is for.
SIZE_BANDS = ("large", "medium", "small", "tiny")
# How many instances a class needs in a band before that band is covered. The recapture plan's
# per-cell target, applied per class here: a floor for ordering a shoot list, not a claim that 40
# images of an item is enough to learn it.
DEFAULT_SIZE_TARGET = 40
# Labels carrying a class index this generation does not declare - an export's head disagreeing
# with the spec. Counted under this name rather than dropped, for the same reason
# `FrameRecord.off_roster` exists: a mismatch that is invisible reads as a clean dataset.
OFF_ROSTER_LABELS = "(labels beyond this generation's class list)"

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
    # Which capture distance this frame came from, or "" when it carries none. Read from the
    # manifest, because a YOLO export keeps no tags and the distance is a Roboflow tag - the same
    # join `train_model.distance_map` does for `--val`. Empty is a real answer here: it is what a
    # generation with no distance axis gives every frame.
    distance: str = ""


@dataclass(frozen=True)
class Match:
    """Which labelled instances were found, which were not, and which predictions were used."""

    matched_truth: tuple[int, ...]
    missed_truth: tuple[int, ...]
    matched_pred: tuple[int, ...]
    # How many predictions were offered to the match at all. Carried because `spurious` below is a
    # subtraction, and the leftovers are not derivable from the ones that were paired.
    predictions: int = 0

    @property
    def spurious(self) -> int:
        """Predictions left over: matched nothing, so they are false positives.

        Counted rather than assumed, and the count is what a threshold decision needs: a miss and
        a false positive are the two costs of `conf_threshold` and only one of them was visible
        before. A box of the *wrong* class counts here as well as a box on nothing, since
        `match_instances` only ever pairs equal classes - so one detection on the wrong item is
        one miss and one false positive, which is the honest reading of it.
        """
        return self.predictions - len(self.matched_pred)


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


def box_width_px(box: Box, capture_width: int) -> float:
    """One box's width in capture pixels, from its 0-1-relative x extent.

    Relative is the right unit to start from: `stretch` scales x and y by different factors, and
    both an export's labels and the app's own detections are stored as a share of the frame, where
    a per-axis scale cancels out. Multiplying back by the capture width is what makes the number
    in the table the number the operator can read off the Live overlay.
    """
    return (box[2] - box[0]) * capture_width


def band_for(width_px: float) -> str:
    """Which of `SIZE_BANDS` one labelled box falls in, by its width in capture pixels."""
    if width_px >= CLOSE_MIN_PX:
        return "large"
    if width_px >= MID_MIN_PX:
        return "medium"
    if width_px >= FAR_MIN_PX:
        return "small"
    return "tiny"


@dataclass
class SizeTally:
    """One class's labelled instances, counted into the size bands.

    Labels only, deliberately: nothing here needs a weight, so a shoot list is available before
    there is a model to audit - which is the order the recapture work happens in.
    """

    counts: dict[str, int] = field(default_factory=lambda: {band: 0 for band in SIZE_BANDS})
    # The narrowest box seen, in capture pixels, or 0.0 when the class has none in this split.
    smallest: float = 0.0

    @property
    def instances(self) -> int:
        return sum(self.counts.values())

    def add(self, width_px: float) -> None:
        self.counts[band_for(width_px)] += 1
        if not self.smallest or width_px < self.smallest:
            self.smallest = width_px

    def thin(self, target: int) -> tuple[str, ...]:
        """The bands this class is short in, in `SIZE_BANDS` order.

        A class with *no* instances at all in the split is short in every band, which is the
        reading that matters most: its cells have never been measured, and a table that dropped
        it would look like a covered dataset.
        """
        return tuple(band for band in SIZE_BANDS if self.counts[band] < target)


def size_lines(
    tallies: dict[str, SizeTally],
    target: int = DEFAULT_SIZE_TARGET,
    capture_width: int = DEFAULT_CAPTURE_WIDTH,
) -> list[str]:
    """The per-class width histogram and the shoot list it implies.

    Pure and rendered as text in its own function, for the same reason `conf_sweep_lines` is: the
    CLI becomes a print, so the table can be read by a test with no dataset on disk.

    Ordered by *gap* rather than alphabetically or by size. The classes short in the most bands
    come first because that is the shoot list's own order, and an alphabetical table would lead
    with whichever product name starts with a digit (v1 has one) - the one ordering that answers
    no question at all.
    """
    out = [
        "label widths, per class, as the share of frame width the boxes occupy:",
        f"  bands, in capture pixels at {capture_width} wide:  large >= {CLOSE_MIN_PX}"
        f"   medium {MID_MIN_PX}-{CLOSE_MIN_PX}   small {FAR_MIN_PX}-{MID_MIN_PX}"
        f"   tiny < {FAR_MIN_PX}",
        "  `!` marks a band holding fewer instances than the target. `smallest` is the narrowest",
        "  box the class has in this split, in the same pixels - the number the Live `det-size`",
        "  readout prints while shooting.",
        "",
    ]
    if not tallies:
        out.append("  no labelled instances in this split (nothing to histogram)")
        return out

    ordered = sorted(
        tallies,
        key=lambda name: (
            -len(tallies[name].thin(target)), -tallies[name].instances, name
        ),
    )
    head = f"{'class':<38}{'instances':>10}" + "".join(f"{band:>9}" for band in SIZE_BANDS)
    head += f"{'smallest':>10}"
    out.append(head)
    for name in ordered:
        tally = tallies[name]
        row = f"{name[:37]:<38}{tally.instances:>10}"
        for band in SIZE_BANDS:
            count = tally.counts[band]
            row += f"{count:>8}{'!' if count < target else ' '}"
        smallest = f"{tally.smallest:.0f}" if tally.smallest else "-"
        out.append(row + f"{smallest:>10}")

    # Already in `ordered`'s gap order, so the list and the table cannot disagree about which
    # class is worst.
    gaps: list[tuple[str, tuple[str, ...]]] = []
    for name in ordered:
        thin = tallies[name].thin(target)
        if thin:
            gaps.append((name, thin))
    out += ["", f"shoot list - bands under the {target}-instance target, worst first:"]
    if not gaps:
        out.append(
            f"  every class clears {target} instance(s) in every band "
            f"({sum(t.instances for t in tallies.values())} labelled instance(s) in this split)"
        )
    for name, thin in gaps:
        cells = "   ".join(
            f"{band} {tallies[name].counts[band]}/{target}" for band in thin
        )
        out.append(f"  {name[:37]:<38}{cells}")

    out += [
        "",
        "`small` is the band the recapture plan calls far, and `tiny` is beyond it: instances",
        "below the far mark's size, which is either an item further away than the mark or a",
        "smaller package at it. Shoot a class into the band it is short in, checked against the",
        "Live readout rather than the folder name, and put the whole set in a new session.",
    ]
    return out


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
        predictions=len(preds),
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
    # The same accounting, keyed by capture distance. A third axis rather than a relabelling of the
    # crowding one: `--val` reports the per-class split by distance, and crowding is the axis the
    # misses turned out to live on, so a threshold that moves one and not the other is the finding.
    by_distance: dict[str, BucketTally] = field(default_factory=dict)
    off_roster: Counter[str] = field(default_factory=Counter)
    frames: int = 0
    # Predictions that matched no label at this threshold - the other cost of `conf_threshold`,
    # and the one a recall figure cannot show, since raising the threshold only ever removes boxes.
    spurious: int = 0

    def __post_init__(self) -> None:
        self.per_class = {i: ClassTally() for i in range(len(self.classes))}
        self.buckets = {name: BucketTally() for name in BUCKETS}
        self.by_distance = {name: BucketTally() for name in DISTANCE_ORDER}

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
        self.spurious += match.spurious
        # One fold over the one match, applied to whichever tallies this frame belongs to. Written
        # as a shared loop rather than two copies so the crowding and distance axes cannot come to
        # count `frames_clean` differently - a frame neither of them disagrees about would then
        # read as complete on one axis and incomplete on the other, with nothing to say which.
        tallies = [self.buckets[SINGLE if len(record.truth) == 1 else MULTI]]
        if record.distance:
            tallies.append(self.by_distance[record.distance])
        for tally in tallies:
            tally.frames += 1
            tally.labelled += len(record.truth)
            tally.found += len(match.matched_truth)
            if not match.missed_truth:
                tally.frames_clean += 1

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
    def distances(self) -> list[str]:
        """The distances this split actually held frames for, in the canonical order."""
        return [name for name in DISTANCE_ORDER if self.by_distance[name].labelled]

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


def resolve_dataset(
    generation: Generation, dataset_dir: str = "", manifest: str = ""
) -> Generation:
    """The generation with its dataset, and its distance join, pointed somewhere else.

    `replace()` on the frozen spec rather than a second code path, exactly as `train_model` does
    it: everything downstream reads `export_dir` and `manifest` off the generation, so an override
    that reached only the frame reader would leave the distance join, the report's `dataset` line
    and the "no such axis" sentence all describing the dataset that was *not* measured.

    An absent flag leaves its field alone, which is why a `--manifest` is the whole of the fix for
    a generation whose spec declares no distance axis: `distance_map(None)` answers `{}` and the
    breakdown is skipped, while a path makes the join happen.
    """
    if not dataset_dir and not manifest:
        return generation
    return replace(
        generation,
        export_dir=Path(dataset_dir).expanduser() if dataset_dir else generation.export_dir,
        manifest=Path(manifest).expanduser() if manifest else generation.manifest,
    )


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


def label_sizes(
    generation: Generation,
    split: str,
    capture_width: int = DEFAULT_CAPTURE_WIDTH,
    limit: int | None = None,
) -> dict[str, SizeTally]:
    """Every labelled instance in the split, counted into its own class's size bands.

    Built on `load_records_only` so the polygon-vs-box parsing is the *same* one the recall audit
    uses. A second reader here is exactly how a dataset gets misdiagnosed: a polygon read as
    `cx cy w h` yields boxes in the wrong place and, on centred objects, zero-width ones - which
    in this table would read as a class that is only ever shot absurdly far away.
    """
    out: dict[str, SizeTally] = {}
    for _path, instances in load_records_only(generation, split, limit):
        for instance in instances:
            name = (
                generation.classes[instance.cls]
                if 0 <= instance.cls < len(generation.classes)
                else OFF_ROSTER_LABELS
            )
            out.setdefault(name, SizeTally()).add(
                box_width_px(instance.box, capture_width)
            )
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
    # Joined on the export filename, the same key `train_model.images_by_distance` uses - the two
    # would otherwise disagree about which image is `far` while both looking right.
    distances = distance_map(generation.manifest)
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
                        preds=tuple(preds), off_roster=tuple(off),
                        distance=distances.get(path.name, ""))
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


def sweep_confs(
    operating: float, ladder: tuple[float, ...] = CONF_SWEEP
) -> tuple[float, ...]:
    """The ladder plus the threshold actually operating, sorted.

    A table that omitted the row for the threshold in force could not justify it, only describe
    its neighbours - and the threshold in force is the one a decision is about. Same reason
    `clamp_probe.sweep_tolerances` folds its shipped tolerance in: the constant and this ladder are
    edited by different people at different times, so "is my value in the table" cannot be left to
    luck.
    """
    return tuple(sorted({*ladder, round(float(operating), 6)}))


def distance_cell(tally: BucketTally) -> str:
    """One distance's recall at one threshold, `-` when the split holds none of its frames,
    `!` when it is under the floor.

    The same three states `train_model`'s class x distance grid uses, for the same reason: a
    distance the split never asked about and a distance that scored zero lead to opposite work,
    and a `0.0%` would state the second while meaning the first.
    """
    if not tally.labelled:
        return f"{'-':>9}"
    return f"{tally.recall:8.1%}{'!' if tally.recall < RECALL_FLOOR else ' '}"


def distance_note(generation: Generation, split: str, has_distance: bool) -> str:
    """Why the per-distance columns are missing - two causes with opposite fixes.

    `manifest is None` means *this generation was never tagged* and no amount of re-running finds
    a distance for it; a manifest that matched nothing means the join failed and someone should
    look. A single "no distances" sentence would hide the second behind the first, which is the
    failure `train_model.distance_notes` already refuses on its side.
    """
    if has_distance:
        return ""
    if generation.manifest is None:
        return (
            f"  no per-distance columns: {generation.name} declares no distance axis, so its"
            " frames were never tagged and there is nothing to break down"
        )
    return (
        f"  no per-distance columns: the manifest matched none of the {split} frames - the"
        " breakdown is skipped, not reported as clean"
    )


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


def conf_sweep_lines(
    records: list[FrameRecord] | tuple[FrameRecord, ...],
    generation: Generation,
    split: str,
    operating: float,
    iou_match: float = DEFAULT_IOU_MATCH,
) -> list[str]:
    """The confidence sweep as lines: what each threshold costs, on both sides at once.

    Its own function for exactly the reason `report_lines` is one - the CLI becomes a print and a
    test can read the table with no weights, no dataset and no GPU. It matters more here than
    there: the per-distance columns only exist for a generation that has a distance axis, and the
    only generation that does has no export on this machine yet, so a table built inline in
    `main()` would have its whole distance half unobservable until v2 is downloaded.

    The two sides are the point. Recall alone can only improve as the threshold falls, so a sweep
    of it recommends 0.0; `extra` is what the same move costs, and neither number is decisive
    without the other.

    The ladder is folded here rather than passed in, so a caller cannot hand this the plain
    `CONF_SWEEP` and quietly render a table with no row for the threshold actually operating - the
    bug the `sweep_confs` fold exists to prevent. Taking `operating` and deriving the rest makes
    "the running threshold is in the table" a property of this function.
    """
    confs = sweep_confs(operating)
    has_distance = any(record.distance for record in records)
    out = [
        "confidence sweep - what each threshold costs on the real frames:",
        f"  `extra` is predictions that matched no label (false positives), `!` is below "
        f"{RECALL_FLOOR}",
    ]
    head = f"{'conf':>6}{'instances':>12}{'recall':>9}{'single':>9}{'multi':>9}{'extra':>8}"
    if has_distance:
        head += "".join(f"{distance:>9}" for distance in DISTANCE_ORDER)
    out.append(head)

    for conf in confs:
        report = measure(records, generation.classes, conf, iou_match)
        row = (
            f"{conf:6.2f}{report.found:6}/{report.instances:<5}{report.recall:9.3f}"
            f"{report.buckets[SINGLE].recall:9.1%}{report.buckets[MULTI].recall:9.1%}"
            f"{report.spurious:8}"
        )
        if has_distance:
            row += "".join(
                distance_cell(report.by_distance[distance]) for distance in DISTANCE_ORDER
            )
        if abs(conf - operating) < 1e-9:
            row += "   <-- operating"
        out.append(row)

    out += [
        "",
        "A rise here means the misses are ranked below the threshold, and conf_threshold is",
        "hot-reloadable - so that part of the recall is a setting, not the model. Read it against",
        "`extra`, which is what the same move costs the other way: a threshold that buys recall by",
        "admitting boxes on nothing is not a better operating point.",
    ]
    note = distance_note(generation, split, has_distance)
    if note:
        out.append(note)
    return out


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
    # The trainer's own spelling, so a command that reads a refreshed export there reads it here
    # too. `--export-dir` is the name the documents still use; both reach one destination.
    ap.add_argument(
        "--dataset-dir",
        "--export-dir",
        dest="dataset_dir",
        default="",
        help="the export to read labels from (default: the generation's own)",
    )
    ap.add_argument(
        "--manifest",
        default="",
        help=(
            "the manifest to join each image's distance from (default: the generation's own, "
            "which is None for a generation that was never tagged)"
        ),
    )
    ap.add_argument(
        "--size-histogram",
        action="store_true",
        help=(
            "histogram every labelled box's width per class and print the shoot list; reads "
            "labels only, so it runs with no weights"
        ),
    )
    ap.add_argument(
        "--capture-width",
        type=int,
        default=DEFAULT_CAPTURE_WIDTH,
        help="the capture width the size bands are stated in (default %(default)s)",
    )
    ap.add_argument(
        "--size-target",
        type=int,
        default=DEFAULT_SIZE_TARGET,
        help="instances a class needs in a band before the band counts as covered (%(default)s)",
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
                    help="report recall and false positives at several confidence thresholds, "
                         "split by crowding and (when the dataset has one) by distance")
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
    generation = resolve_dataset(
        generation_for(args.generation), args.dataset_dir, args.manifest
    )
    weights = args.weights or str(SIDECAR_ROOT / "models" / generation.weight_name)
    resize_mode = args.resize_mode or generation.resize_mode

    # Before the weight is resolved, because this mode asks what to shoot next and that question
    # comes before there is a refreshed weight to measure. Everything above it - the resource
    # envelope included - belongs to inference, which this mode never does.
    if args.size_histogram:
        print(
            f"generation {generation.name}   split {args.split}"
            f"   dataset {generation.export_dir}"
        )
        tallies = label_sizes(generation, args.split, args.capture_width, args.limit)
        for line in size_lines(tallies, args.size_target, args.capture_width):
            print(line)
        return 0

    if not Path(weights).exists():
        raise SystemExit(f"no such weight: {weights}")

    budget = resources.measure(args.max_use_percent, args.disk_reserve_gb, args.max_vram_percent)
    device = resources.resolve_device(args.device)
    budget = resources.apply(budget, device, args.hard_vram_cap)

    # One pass at the lowest threshold anyone asked about: every other conf is a filter over
    # these same records (see RecallReport.add). Only the NMS sweep needs its own pass.
    lowest = min(sweep_confs(args.conf)) if args.conf_sweep else args.conf
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
        for line in conf_sweep_lines(
            records, generation, args.split, args.conf, args.iou_match
        ):
            print(line)

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
