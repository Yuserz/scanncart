#!/usr/bin/env python
"""Measure the unsure-phantom rule the way `clamp_probe.py` measures the clamp rule.

`app/acceptance.py` shipped a third rule - a low-confidence frame-spanning box or edge band - and
the numbers justifying it came from scratch harnesses that were then deleted. So the owner's docstring cited a
measurement with no way to re-run it, which is the failure `clamp_probe.py`'s own docstring opens
with. This is the tool that re-derives those numbers, and it exists so that moving either threshold
(`PHANTOM_CONF_CEILING`, `FRAME_SPANNING_AREA`) or the band bounds has to move them consciously
instead of silently.

**Four populations, each from its real source.**

    export      the whole labelled set, split by split, through `audit_recall.collect` and its own
                one-to-one class-aware matcher - the denominator any cost is a share of
    negatives   the 50 stored empty-counter frames (`clamp_probe`'s own population): the family the
                defect was found on, whole-frame boxes at high confidence
    control     60 real product frames (`clamp_probe`'s own sample), the population that says what
                catching a phantom would cost in items the app must keep registering
    live        an optional empty-counter window through `app.camera.CameraCapture`, which is the
                only real source for the edge band: it does not occur in the stored negatives, and it
                is the family that wrote item-log rows on an empty counter

**It reads the rule, it does not restate it.** The ceiling, the spanning area and the band bounds
come from `app.acceptance`, and the decision is that module's own `accept_detections` - this file's
`unsure()` is the same rule with the ceiling as a parameter, so the table can price candidates, and
the two agree by construction (pinned in `tests/test_unsure_probe.py`). What is this tool's own is
the *naming*: which shape a box takes, and the four buckets a population falls into. That is
reporting, and it is the one part that cannot be imported.

**The two ways a detection survives, and only one of them is a finding.** A survivor that is shaped
and confident enough to keep is the ceiling's own limit - the confident members of both shapes are
real products in the export (0.92-0.97), so no ceiling drops them without dropping items, and the
report prices the ceiling that would. A survivor that is *below* the ceiling and shaped like neither
is the residue: a detection the rule was meant to catch and did not. It is printed by name with its
shape and confidence, and `--strict` fails on it.

**The table is one axis, deliberately.** Shape candidates were priced before this shipped and the
acceptance module's docstring keeps that ledger, so what is worth sweeping is the confidence
ceiling: the shipped row is printed beside the confidence of the last real detection it drops and the
first it keeps, and a gap between those two means every ceiling inside it is the same rule over this
data - which is what makes the shipped value a free choice rather than a fitted one.

**What the live window's claim is, and what it is not - two contracts, because one cannot do both
jobs.** The band family exists only in front of a camera, so this is the one population a run can
fail to have any of. So the window is run through the profile's own settings - not through a
rules-off pass - and the model's output is recorded at the detector boundary as well, so the run
**fails** the moment a shaped detection under the ceiling escapes into what the Pipeline kept: a
rule that is off, a predicate that moved, or a decision that never saw the detection. That is a
failure whatever the scene holds, and it is the one live check no stored population can make. It does **not** fail merely
because the counter offered nothing, which is the rule working rather than the rule untested. The
window's own status rides in both readings - frames, detections, band-shaped count, and the frames'
mean luminance, read off the same stream the model was shown - so a quiet counter and a covered lens
are told apart beside the verdict instead of by rerunning it. A run that has to *prove* the family
was exercised passes `--require-live-band` (`make verify-unsure UNSURE_REQUIRE_BAND=1`), which turns
those same numbers into a failure.

**What this machine's live source actually yielded**, over the windows run while this was written -
recorded because it is the evidence the live claims were decided on, and because it moved between
runs on the same dark counter (mean luminance 14-15/255, against 32/255 for the stored negatives):

    window                      frames   detections   band-shaped   escaped   luminance
    30 s                          1645            0             0         0   14/255
    30 s                          1710          866             0         0   15/255
    30 s                          1542         1282             2         0   16/255
    12 s, rule off in the profile  642          244             0        19   15/255

So the whole-frame family is one this counter reproduces constantly and refuses as designed; the
band family appeared in one window only (2 detections, both refused), which is exactly the shape of
"a family a scene either offers or does not"; and the escape claim is not an assertion, because with
the rule switched off in the profile the next window failed it with 19 escapes of 244. A window that
offers no band says so in the reading, and `--require-live-band` is how a lit counter goes on record
as one that did.

**No network.** Needs the export (for the cost), the stored negatives, and - unless
`--live-seconds 0` - the camera. It fails loudly rather than skipping when they are absent, for the
reason the clamp probe gives: a skip here is a silent pass on the only check that says the empty
counter stays empty.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import resources  # must precede numpy/torch: sets OMP/MKL thread limits

import audit_recall
from audit_recall import FrameRecord, match_instances
from clamp_probe import (
    NEGATIVES_DIR,
    REAL_SAMPLE,
    REAL_SPLIT,
    Fixed,
    labelled_frames,
    negative_frames,
    read_frames,
)
from generations import DEFAULT, GENERATIONS, Generation
from generations import get as generation_for
from label_classes import SPLIT_NAMES
from workspace import SIDECAR_ROOT

# This tool drives `app` code, and run as a script Python puts `tools/` on the path - not the sidecar
# root - so `from app.acceptance import ...` would raise ModuleNotFoundError for whoever followed the
# doc. The test suite never sees it, because pytest.ini adds the sidecar root: the tool works under
# test and dies by hand. Same shim as spec_check.py.
sys.path.append(str(SIDECAR_ROOT))

from app.acceptance import (  # noqa: E402  (needs the path above)
    BAND_MAX_HEIGHT,
    BAND_MIN_WIDTH,
    CLAMPED_REASON,
    FRAME_FILLING_REASON,
    FRAME_SPANNING_AREA,
    PHANTOM_CONF_CEILING,
    UNSURE_REASON,
    accept_detections,
    is_edge_band,
    pinned_edges,
)
from app.schemas import Detection  # noqa: E402
from app.settings_store import DEFAULT_SETTINGS_PATH  # noqa: E402

#: The ceilings the shipped one is judged against. It brackets the constant on **both** sides: the
#: higher rows are the argument, because what 0.90 would buy and what it would cost is the reason
#: 0.85 is where it is. A ladder that stopped at the shipped value could only show the cheaper
#: direction, which is the direction nobody needs convincing about.
CEILING_LADDER = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

#: The whole labelled set, split by split, so the cost is a share of every frame the weights were
#: measured on rather than of one split's. Read from the tools' shared vocabulary rather than spelled
#: out a second time, which is the drift `test_dataset_drift_guards.py` exists to catch - and which
#: it caught in the first draft of this constant.
EXPORT_SPLITS = SPLIT_NAMES

#: The control population, read from the clamp probe's own constants so the two tools describe the
#: same 60 frames rather than two samples that happen to have the same size - a cost quoted against
#: "real products" has to be the same products.
CONTROL_SPLIT = REAL_SPLIT
CONTROL_SAMPLE = REAL_SAMPLE

#: How long the optional live window runs. Long enough to hold the band family, which arrived on
#: ~13% of frames in the session that found it, and short enough to sit through.
LIVE_SECONDS = 30

#: How many survivors to name before summarising. A handful of names is a readout; a thousand is a log.
MAX_NAMED = 6

#: What the shipped rule may cost on the 60-frame real product control. **Measured, not chosen**: at
#: the shipped ceiling the rule refuses 1 of the control's 60 detections, so this is that price - and
#: a change that takes it to 2 is a different rule, which is the whole reason the number is pinned
#: here instead of being re-argued in a docstring.
CONTROL_COST_CEILING = 1

#: The app's own settings file - the same file and the same absolute resolution as `clamp_probe`'s,
#: because `conf_threshold` and `imgsz` are the operating point every detection here is produced at.
SETTINGS_PATH = SIDECAR_ROOT / DEFAULT_SETTINGS_PATH

SHAPE_SPANNING = "spanning"
SHAPE_BAND = "band"
SHAPE_NONE = "neither"

Box = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# The pure layer: the rule with its ceiling as an argument, the table, the buckets and the claims
# ---------------------------------------------------------------------------


def shaped(box: Box) -> bool:
    """Whether a box takes either of the rule's two shapes - the half that does not read confidence.

    The same two alternatives `is_unsure_phantom` tests, named once so a candidate ceiling can be
    priced without restating the shape half in a second place.
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1) >= FRAME_SPANNING_AREA or is_edge_band(box)


def shape_name(box: Box) -> str:
    """Which of the two shapes a box takes; `SHAPE_NONE` is a detection the rule cannot see."""
    if (box[2] - box[0]) * (box[3] - box[1]) >= FRAME_SPANNING_AREA:
        return SHAPE_SPANNING
    return SHAPE_BAND if is_edge_band(box) else SHAPE_NONE


def unsure(box: Box, conf: float, ceiling: float = PHANTOM_CONF_CEILING) -> bool:
    """The rule at an arbitrary ceiling: what the table sweeps, shipping at the module's constant.

    Identical to `app.acceptance.is_unsure_phantom` at the shipped ceiling - the tests pin that over
    a grid - because the shipped rule is the one row this table exists to justify, and a second
    definition is the drift the tool is supposed to catch.
    """
    return conf < ceiling and shaped(box)


@dataclass(frozen=True)
class Shot:
    """One detection as the model produced it: the pair the rule reads, plus the label for naming.

    `box` and `conf` are the rule's whole input; `cls` rides along so a survivor can be reported as
    the product the model claimed rather than as four coordinates.
    """

    box: Box
    conf: float
    cls: str = ""

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    @property
    def pinned(self) -> int:
        return pinned_edges(self.box)

    def describe(self) -> str:
        return (
            f"{self.cls or '?':<28} conf {self.conf:.3f}  area {self.area:.3f}  "
            f"pinned {self.pinned}  {shape_name(self.box):<8} box=("
            + ", ".join(f"{value:.4f}" for value in self.box)
            + ")"
        )


@dataclass(frozen=True)
class Row:
    """One candidate ceiling and what it costs on every population."""

    ceiling: float
    caught_negatives: int
    negatives: int
    caught_live: int
    live: int
    lost_control: int
    control: int
    dropped_real: int
    matched_real: int
    #: The two shape rules' cost on the same matched population, so "the cheapest of the three" is a
    #: reading this report carries rather than a sentence in a docstring.
    clamped_real: int
    filling_real: int
    #: The confidence of the last real detection this ceiling drops and the first one it keeps. Read
    #: as the gap the ceiling sits in rather than as a property of the ceiling.
    dropped_max_conf: float | None
    kept_min_conf: float | None

    @property
    def caught(self) -> int:
        return self.caught_negatives + self.caught_live

    @property
    def phantoms(self) -> int:
        return self.negatives + self.live

    @property
    def dropped_share(self) -> float:
        return self.dropped_real / self.matched_real if self.matched_real else 0.0

    @property
    def in_a_gap(self) -> bool:
        """Whether no real detection sits between this ceiling and the nearest one it keeps.

        True means every ceiling in that interval is the same rule over this data: moving the
        constant inside the gap buys nothing and costs nothing, which is what makes the shipped
        value a free choice rather than a tuned one. A ceiling that drops no real detection at all is
        trivially in that state - there is nothing between it and the first one it keeps.
        """
        if self.dropped_max_conf is None:
            return True
        if self.kept_min_conf is None:
            return False
        return self.dropped_max_conf < self.kept_min_conf


def ceiling_table(
    matched_real: tuple[Shot, ...],
    negatives: tuple[Shot, ...],
    live: tuple[Shot, ...],
    control: tuple[Shot, ...],
    ceilings: tuple[float, ...] = CEILING_LADDER,
) -> list[Row]:
    """Every candidate ceiling scored against all four populations. Pure, so it is testable."""

    def caught(shots: tuple[Shot, ...], ceiling: float) -> int:
        return sum(1 for shot in shots if unsure(shot.box, shot.conf, ceiling))

    shaped_real = [shot for shot in matched_real if shaped(shot.box)]
    # The two shape rules' cost on the same population: constant across the ladder, which is why it
    # rides on every row rather than sitting in a footnote of its own.
    clamped_real = sum(1 for shot in matched_real if shot.pinned == 4)
    filling_real = sum(1 for shot in matched_real if shot.pinned == 3)
    rows: list[Row] = []
    for ceiling in ceilings:
        dropped = [shot for shot in shaped_real if unsure(shot.box, shot.conf, ceiling)]
        kept = [shot for shot in shaped_real if not unsure(shot.box, shot.conf, ceiling)]
        rows.append(
            Row(
                ceiling=ceiling,
                caught_negatives=caught(negatives, ceiling),
                negatives=len(negatives),
                caught_live=caught(live, ceiling),
                live=len(live),
                lost_control=caught(control, ceiling),
                control=len(control),
                dropped_real=len(dropped),
                matched_real=len(matched_real),
                clamped_real=clamped_real,
                filling_real=filling_real,
                dropped_max_conf=max((shot.conf for shot in dropped), default=None),
                kept_min_conf=min((shot.conf for shot in kept), default=None),
            )
        )
    return rows


def row_for(rows: tuple[Row, ...] | list[Row], ceiling: float) -> Row | None:
    """The row standing for `ceiling`, or None if the ladder never reached it."""
    for row in rows:
        if abs(row.ceiling - ceiling) < 1e-9:
            return row
    return None


def catch_cost(rows: tuple[Row, ...] | list[Row], shot: Shot) -> str:
    """What it would take to catch one survivor, in the terms the table is priced in.

    The answer is the point of naming survivors at all: for a confident box it is a ceiling, and for
    a box shaped like neither it is a rule that does not exist - which is the honest form of "this
    one is not closable by the constant".
    """
    if shape_name(shot.box) == SHAPE_NONE:
        return (
            "no ceiling catches this: it is shaped like neither, so the shape is what would have to "
            "move - and that is the axis the export priced as costing real items"
        )
    candidates = [row for row in rows if row.ceiling > shot.conf + 1e-9]
    if not candidates:
        return "no ceiling on the ladder reaches it"
    row = min(candidates, key=lambda item: item.ceiling)
    return (
        f"catching it needs ceiling {row.ceiling:.2f}, which costs {row.lost_control} of "
        f"{row.control} control detections and drops {row.dropped_real} of {row.matched_real} matched"
    )


@dataclass(frozen=True)
class Ledger:
    """What the shipped decision did to one population, in the four buckets that matter.

    The buckets partition the population: refused under each rule, and - for what got through -
    `blind` (shaped but confident enough to keep: the ceiling's own limit), `residue` (below the
    ceiling and shaped like neither: the shapes' blind spot) and `plain`.
    """

    label: str
    frames: int
    raw: tuple[Shot, ...]
    #: What the app's own decision function keeps, over `raw`.
    decided: tuple[Shot, ...]
    #: The raw detections it refuses. Recovered by identity, which holds because `decide` returns the
    #: very objects it was handed - so this is the same list the counts were made from, not a second
    #: reading of the same rule.
    refused_shots: tuple[Shot, ...]
    refused: dict[str, int]
    residue: tuple[Shot, ...]
    blind: tuple[Shot, ...]
    plain: tuple[Shot, ...]
    #: What the real `Pipeline` itself kept, when the population was walked through one. `None` for
    #: the export, which is scored by `audit_recall.collect` with no pipeline in the loop.
    pipeline_kept: tuple[Shot, ...] | None = None


    @property
    def kept(self) -> tuple[Shot, ...]:
        """What reached the frame: the Pipeline's own output where one ran, the decision otherwise.

        The Pipeline's own list is preferred because it is the app's real behaviour, tracker and all -
        and the two can differ by a detection, since `track(persist=True)` carries state between
        passes and ByteTrack promotes low-confidence boxes it kept. That difference is why the
        end-to-end claim is measured on the Pipeline's list rather than on this file's arithmetic.
        """
        return self.pipeline_kept if self.pipeline_kept is not None else self.decided




def decide(
    shots: tuple[Shot, ...], *, clamped: bool, filling: bool, unsure_: bool
) -> tuple[list[Shot], dict[str, int]]:
    """The app's own accept/reject decision over raw detections, with the survivors mapped back.

    `accept_detections` is asked rather than re-implemented - the order the rules are applied in is
    the whole reason that function exists - and the survivors are read back positionally, which is
    sound because the decision preserves the input's order and only ever drops.
    """
    rows = [Detection(track_id=None, cls=shot.cls, conf=shot.conf, box=shot.box) for shot in shots]
    result = accept_detections(
        rows, suppress_clamped=clamped, suppress_frame_filling=filling, suppress_unsure=unsure_
    )
    survivors: list[Shot] = []
    index = 0
    for detection in result.accepted:
        while index < len(shots) and (
            shots[index].box,
            shots[index].conf,
        ) != (detection.box, detection.conf):
            index += 1
        if index < len(shots):
            survivors.append(shots[index])
            index += 1
    return survivors, result.rejected


def ledger(
    label: str,
    frames: int,
    shots: tuple[Shot, ...],
    *,
    clamped: bool,
    filling: bool,
    unsure_: bool,
    ceiling: float = PHANTOM_CONF_CEILING,
    pipeline_shots: tuple[Shot, ...] | None = None,
) -> Ledger:
    """One population under the shipped decision, in the four buckets the report prints."""
    decided, refused = decide(shots, clamped=clamped, filling=filling, unsure_=unsure_)
    kept = pipeline_shots if pipeline_shots is not None else tuple(decided)
    blind: list[Shot] = []
    residue: list[Shot] = []
    plain: list[Shot] = []
    for shot in kept:
        if shot.conf < ceiling and shape_name(shot.box) == SHAPE_NONE:
            residue.append(shot)
        elif shaped(shot.box):
            blind.append(shot)
        else:
            plain.append(shot)
    kept_ids = {id(shot) for shot in decided}
    return Ledger(
        label=label,
        frames=frames,
        raw=shots,
        decided=tuple(decided),
        refused_shots=tuple(shot for shot in shots if id(shot) not in kept_ids),
        refused=refused,
        residue=tuple(residue),
        blind=tuple(blind),
        plain=tuple(plain),
        pipeline_kept=pipeline_shots,
    )


@dataclass(frozen=True)
class Census:
    """Every population and every number the claims and the report read, in one object.

    One object rather than a dozen arguments so a claim cannot read a number the report does not
    print, and so the whole verdict is a single-argument pure function over hand-built data - which
    is what makes the claims testable without a camera, a weight or 6.6 GB of dataset.
    """

    ceiling: float
    rows: tuple[Row, ...]
    negatives: Ledger
    control: Ledger
    live: Ledger
    export: Ledger
    #: split -> (frames, predictions, matched), the per-split readout behind the export cost.
    per_split: tuple[tuple[str, int, int, int], ...]
    predictions: int
    shipped_flags: tuple[bool, bool, bool]
    #: What the live window's frames looked like - how a silent window is read as a quiet counter or
    #: as a lens pointed at nothing.
    live_scene: Scene
    #: Whether this run has to *prove* the band family was exercised (`--require-live-band`).
    require_live_band: bool

    @property
    def live_measured(self) -> bool:
        return self.live_scene.frames > 0

    @property
    def shipped(self) -> Row | None:
        return row_for(self.rows, self.ceiling)

    @property
    def populations(self) -> tuple[Ledger, ...]:
        return (self.negatives, self.control, self.live, self.export)

    @property
    def residue(self) -> tuple[Shot, ...]:
        """What the rule was meant to catch and did not: empty-counter survivors only.

        Restricted to the empty counter and to the live window on purpose. A small low-confidence
        prediction on a real product frame is ordinary model behaviour, not a residue, and counting
        it would turn this claim into one that can never hold.
        """
        return (*self.negatives.residue, *self.live.residue)


@dataclass(frozen=True)
class Scene:
    """What a live window's frames actually looked like, one sample per frame.

    The evidence a silent window needs: 0 detections on a lit counter and 0 detections on a black one
    are the same two numbers and opposite findings, and only the pixels tell them apart. Read from
    the same source the pipeline reads (`source.latest()`), one sample per processed frame, so the
    brightness belongs to the stream the model was shown rather than to a second look at the device.
    """

    luminance: tuple[float, ...]

    @property
    def frames(self) -> int:
        return len(self.luminance)

    @property
    def mean(self) -> float:
        return sum(self.luminance) / len(self.luminance) if self.luminance else 0.0

    def describe(self) -> str:
        if not self.luminance:
            return "no frames"
        return (
            f"mean luminance {self.mean:.0f}/255 (min {min(self.luminance):.0f}, "
            f"max {max(self.luminance):.0f})"
        )


@dataclass(frozen=True)
class Check:
    """One checkable statement about the shipped rule, with the reading behind it."""

    ok: bool
    claim: str
    reading: str

    def line(self) -> str:
        return f"  [{'PASS' if self.ok else 'FAIL'}] {self.claim:<56} {self.reading}"


def checks(census: Census) -> tuple[Check, ...]:
    """The statements the report stands behind, each with the number that decides it.

    A table alone would let a ceiling that had begun eating real items print a column nobody reads,
    and the empty-counter check alone would pass on a window where the model happened to produce
    nothing. So: the cost, the catch from both sources, the end-to-end run, the control, and the one
    claim that separates "the rule is narrow" from "the rule is a rule for this defect".
    """
    shipped = census.shipped
    if shipped is None:
        return (
            Check(
                False,
                "the shipped ceiling is on the ladder",
                "the table cannot justify a rule it does not contain",
            ),
        )

    return (
        Check(
            shipped.in_a_gap,
            "the ceiling sits in a gap no real detection occupies",
            f"drops <= {_conf(shipped.dropped_max_conf)}, keeps >= {_conf(shipped.kept_min_conf)}",
        ),
        Check(
            shipped.dropped_real < shipped.clamped_real,
            "it costs fewer real detections than the clamp rule does",
            f"{shipped.dropped_real} of {shipped.matched_real} matched ({shipped.dropped_share:.2%})"
            f", vs the clamp rule's {shipped.clamped_real}",
        ),
        Check(
            shipped.caught_negatives > 0 and not census.negatives.kept,
            "it catches the stored empty counter's own family",
            f"{shipped.caught_negatives} of {shipped.negatives} refused, and the Pipeline kept "
            f"{len(census.negatives.kept)} of {len(census.negatives.raw)}",
        ),
        Check(
            not live_escapes(census),
            "no shaped detection under the ceiling escaped in the live window",
            live_reading(census),
        ),
        Check(
            live_bands(census) > 0 or not census.require_live_band,
            "the live window exercised the band family",
            live_band_reading(census),
        ),
        Check(
            shipped.lost_control <= CONTROL_COST_CEILING,
            "its price on the real product control has not moved",
            f"{shipped.lost_control} of {shipped.control} detections refused over "
            f"{census.control.frames} frames, against a measured {CONTROL_COST_CEILING}",
        ),
        Check(
            not census.residue,
            "no detection under the ceiling survives outside both shapes",
            f"{len(census.residue)} residue over "
            f"{census.negatives.frames + census.live.frames} empty-counter frames",
        ),
        Check(
            census.shipped_flags[2],
            "the rule is on in the profile this measured",
            f"unsure={census.shipped_flags[2]}, clamp={census.shipped_flags[0]}, "
            f"filling={census.shipped_flags[1]}",
        ),
    )


def verdict(checks_: tuple[Check, ...]) -> str:
    """One sentence for the whole block, naming the failures rather than only counting them."""
    bad = [check for check in checks_ if not check.ok]
    if not bad:
        return f"all {len(checks_)} claims hold"
    return f"{len(bad)} of {len(checks_)} claims FAIL: " + "; ".join(check.claim for check in bad)


def strict_failed(checks_: tuple[Check, ...]) -> bool:
    """Whether a `--strict` run should exit non-zero."""
    return any(not check.ok for check in checks_)


def _conf(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------
# The impure layer: the populations, one frame at a time
# ---------------------------------------------------------------------------


def shot_of(row) -> Shot:
    """One detection as the pair the rule reads - a frame-message row or a `Detection`."""
    if isinstance(row, dict):
        box, conf, cls = row["box"], row["conf"], row["cls"]
    else:
        box, conf, cls = row.box, row.conf, row.cls
    return Shot(box=tuple(box), conf=float(conf), cls=str(cls))


class Recorded:
    """A detector that remembers what the model produced, before any rule saw it.

    The live window has to run through the app's own settings - that is the only way a detection
    escaping the shipped decision can be observed at all - and the Pipeline's frame message carries
    only what survived it. So both sides are taken from the same pass, at the seam
    `Pipeline.process_once` reads: `seen` is the model's output and the message is the decision's.

    A proxy rather than a second inference pass, because a live scene cannot be re-run: two passes
    would compare two moments of a counter and call the difference a rule.
    """

    def __init__(self, detector) -> None:
        self._detector = detector
        self.seen: list[Shot] = []

    def infer(self, frame):
        detections = self._detector.infer(frame)
        self.seen.extend(shot_of(detection) for detection in detections)
        return detections

    def __getattr__(self, name: str):
        """Everything else - `names`, `set_conf`, `close` - is the wrapped detector's."""
        return getattr(self._detector, name)


def walk(frames: list, source: Fixed, pipeline) -> tuple[tuple[Shot, ...], int]:
    """A population through the real `Pipeline.process_once`, at whatever the settings now say.

    One frame at a time rather than a buffered list, so a live source's memory does not scale with
    the window, and so the same function serves the stored populations and the camera.
    """
    shots: list[Shot] = []
    for frame in frames:
        source.set(frame)
        message = pipeline.process_once()
        if message is None:
            continue
        shots.extend(shot_of(row) for row in message["detections"])
    return tuple(shots), len(frames)


def live_escapes(census: "Census") -> tuple[Shot, ...]:
    """Detections the rule was supposed to refuse and did not, in the live window.

    The escape contract. A shaped detection under the ceiling in what the Pipeline kept means the
    rule did not act on a detection it was built for - the profile has it off, the predicate moved,
    or the decision never saw it - and that is a failure whatever the scene contains. A window with
    nothing in it escapes nothing, which is why this is a claim about what was *seen* rather than
    about the shape of the session.
    """
    return tuple(
        shot
        for shot in census.live.kept
        if shot.conf < census.ceiling and shaped(shot.box)
    )


def live_bands(census: "Census") -> int:
    """Band-shaped detections the live window produced, refused or not: was the family exercised?"""
    return sum(1 for shot in census.live.raw if shape_name(shot.box) == SHAPE_BAND)


def live_reading(census: "Census") -> str:
    """What the live window yielded, in the words the escape claim is decided in."""
    if not census.live_measured:
        return (
            "the live window sampled no frame at all (--live-seconds 0, or a window too short to "
            "deliver one), so nothing was seen to escape"
        )
    escaped = live_escapes(census)
    bands = live_bands(census)
    spanning = sum(1 for shot in census.live.raw if shape_name(shot.box) == SHAPE_SPANNING)
    if not census.live.raw:
        return (
            f"the window produced no detection at all over {census.live_scene.frames} frames "
            f"({census.live_scene.describe()}) - nothing to escape, and nothing was claimed"
        )
    return (
        f"{len(escaped)} escaped, of {len(census.live.raw)} raw over {census.live_scene.frames} "
        f"frames ({bands} band-shaped, {spanning} spanning)"
    )


def live_band_reading(census: "Census") -> str:
    """Whether the band family was exercised, and what the run's own contract says about that."""
    demand = (
        "--require-live-band is set, so this run has to prove it"
        if census.require_live_band
        else "--require-live-band is not set, so a quiet counter passes this claim"
    )
    if not census.live_measured:
        return f"the window sampled no frame at all (--live-seconds 0, or too short); {demand}"
    bands = live_bands(census)
    return (
        f"{bands} band-shaped detection(s) in {census.live_scene.frames} frames, "
        f"{census.live_scene.describe()}; {demand}"
    )


def live_window(
    pipeline, source, recorder, seconds: float, clock=time.monotonic, pause=time.sleep
) -> tuple[tuple[Shot, ...], tuple[Shot, ...], Scene]:
    """A window off the camera through the app's own `CameraCapture`, at the settings it was given.

    Returns `(model, kept, scene)`: what the model produced, what the Pipeline's decision let
    through, and what the frames looked like - all from one pass, because a live scene cannot be
    re-run. The rule is *on* here, which is the point: a shaped detection under the ceiling in `kept`
    is the escape this tool exists to catch, and it is only observable when the app's own decision
    is the thing being run.

    The frames are sampled as well as counted, because the population's status has to be readable
    beside the verdict: a window that produced nothing on a lit counter is the rule doing its job,
    and the same window on a covered lens is a measurement of nothing.
    """
    deadline = clock() + seconds
    kept: list[Shot] = []
    luminance: list[float] = []
    while clock() < deadline:
        message = pipeline.process_once()
        if message is None:
            pause(0.005)
            continue
        kept.extend(shot_of(row) for row in message["detections"])
        frame = source.latest()[1]
        if frame is not None:
            luminance.append(float(frame.mean()))
    return tuple(recorder.seen), tuple(kept), Scene(luminance=tuple(luminance))


def export_population(generation: Generation, records: list[FrameRecord]) -> tuple[Shot, ...]:
    """Every prediction the export produced, matched or not - what the decision table is a census of."""
    names = generation.classes
    return tuple(
        Shot(
            box=prediction.box,
            conf=prediction.conf,
            cls=names[prediction.cls] if prediction.cls < len(names) else "",
        )
        for record in records
        for prediction in record.preds
    )


def export_shots(
    generation: Generation, records: list[FrameRecord]
) -> tuple[tuple[Shot, ...], int]:
    """The export's ground-truth-matched detections, and how many were offered.

    Matching is `audit_recall`'s own one-to-one, class-aware matcher - the same one `--val` and
    `accept_v2` score against - so the cost this reports means here what it means in the run that
    produced the weights. Only *matched* predictions are priced: an unmatched one has no labelled
    object to be a false negative against, and counting it would let a rule look cheap by dropping
    noise.
    """
    names = generation.classes
    matched: list[Shot] = []
    for record in records:
        for index in match_instances(record.truth, record.preds).matched_pred:
            prediction = record.preds[index]
            matched.append(
                Shot(
                    box=prediction.box,
                    conf=prediction.conf,
                    cls=names[prediction.cls] if prediction.cls < len(names) else "",
                )
            )
    return tuple(matched), sum(len(record.preds) for record in records)


def report_lines(
    census: Census,
    *,
    generation: Generation,
    weights: str,
    device: str,
    conf: float,
    shipped_conf: float,
    imgsz: int,
    live_seconds: float,
    checks_: tuple[Check, ...],
) -> list[str]:
    """The whole report, pure and printable - one function so a caller cannot print half of it."""
    shipped = census.shipped
    out = [
        f"weights    {weights}",
        f"dataset    {generation.export_dir}",
        f"splits     {', '.join(EXPORT_SPLITS)}   generation {generation.name}   "
        f"{len(generation.classes)} classes",
        f"device     {audit_recall.device_label(device)}",
        f"the rule   conf < {census.ceiling}  and  (area >= {FRAME_SPANNING_AREA}"
        f"  or an edge band: width >= {BAND_MIN_WIDTH}, height <= {BAND_MAX_HEIGHT})",
        "           (every constant read from app/acceptance.py, which owns the rule)",
        f"operating  conf>={conf}   imgsz {imgsz}   every number below is Pipeline.process_once",
    ]
    if abs(conf - shipped_conf) > 1e-9:
        out.append(
            f"           this profile runs conf>={conf}, while Settings() ships {shipped_conf} - a"
            " phantom is a low-confidence detection, so the counts scale with it"
        )
    out += [
        "",
        "what the shipped decision does to each population (raw detections, then where they went)",
        "  refused counts are each rule's own, under the first rule that refused the detection -",
        "  the order the app applies them in, so they sum to the refused total and not to a box's",
        "  coincidences.",
        f"  {'population':<30}{'frames':>8}{'raw':>7}{'clamp':>7}{'fill':>6}{'unsure':>8}"
        f"{'blind':>7}{'residue':>9}{'plain':>7}",
    ]
    for item in census.populations:
        out.append(
            f"  {item.label:<30}{item.frames:>8}{len(item.raw):>7}"
            f"{item.refused.get(CLAMPED_REASON, 0):>7}"
            f"{item.refused.get(FRAME_FILLING_REASON, 0):>6}"
            f"{item.refused.get(UNSURE_REASON, 0):>8}"
            f"{len(item.blind):>7}{len(item.residue):>9}{len(item.plain):>7}"
        )
    matched = sum(entry[3] for entry in census.per_split)
    out += [
        "  the export, split by split: frames, predictions, ground-truth-matched",
    ]
    for split, frames, predictions, hits in census.per_split:
        out.append(f"  {split:<10}{frames:>7}{predictions:>16}{hits:>12}")
    out += [
        f"  matched one-to-one: {matched} of {census.predictions} predictions",
        "  'blind' is shaped but confident enough to keep - the ceiling's own limit; 'residue' is",
        "  below the ceiling and shaped like neither: the rule's blind spot, and a failure.",
        "",
        "the ceiling, priced on every population",
        f"  {'ceiling':>8}{'negatives':>12}{'live':>12}{'control lost':>14}{'real dropped':>14}"
        f"{'gap':>22}",
    ]
    for row in census.rows:
        mark = "*" if shipped is not None and abs(row.ceiling - shipped.ceiling) < 1e-9 else " "
        gap = (
            f"({_conf(row.dropped_max_conf)}, {_conf(row.kept_min_conf)})"
            if row.in_a_gap
            else "on a knife edge"
        )
        out.append(
            f" {mark}{row.ceiling:>7.2f}{row.caught_negatives:>7}/{row.negatives:<5}"
            f"{row.caught_live:>7}/{row.live:<5}{row.lost_control:>7}/{row.control:<6}"
            f"{row.dropped_real:>7}/{row.matched_real:<6}{gap:>22}"
        )
    out += [
        "  * the shipped ceiling. 'gap' is the confidence between the last real detection it drops",
        "    and the first one it keeps: a gap means the same rule either side of it, so the",
        "    constant is free inside that interval and this table cannot tune it.",
    ]
    if shipped is not None:
        out.append(
            f"  the three rules on the same matched population: clamp {shipped.clamped_real}, "
            f"frame-filling {shipped.filling_real}, unsure {shipped.dropped_real} of "
            f"{shipped.matched_real} ({shipped.dropped_share:.2%})"
        )
    out += [
        "",
        "survivors, named, and what catching each would cost",
    ]
    named = 0
    for item in census.populations:
        for shot in (*item.blind, *item.residue):
            if named >= MAX_NAMED:
                break
            named += 1
            out.append(f"  {item.label:<10} {shot.describe()}")
            out.append(f"  {'':<10} -> {catch_cost(census.rows, shot)}")
    if not named:
        out.append("  none: no detection survived the shipped decision in any population")
    costs = census.control.refused_shots
    out.append("")
    out.append(
        f"what it costs: {len(costs)} of {census.control.frames} real product control detections "
        "refused"
    )
    for shot in costs[:MAX_NAMED]:
        out.append(f"  {census.control.label:<10} {shot.describe()}")
    if not costs:
        out.append("  none: the control's products all reach the frame")
    out += ["", "the live window, and what it actually yielded"]
    if census.live_measured:
        out.append(
            f"  {census.live_scene.frames} frames over {live_seconds:.0f}s, "
            f"{len(census.live.raw)} detections, {live_bands(census)} band-shaped, "
            f"{census.live_scene.describe()}"
        )
        out.append(f"  {live_reading(census)}")
        out.append(f"  {live_band_reading(census)}")
    else:
        out.append(
            "  no frames: skipped (--live-seconds 0) or too short to deliver one. The edge-band"
        )
        out.append(
            f"  family exists only in front of a camera, so it was not measured - a {LIVE_SECONDS:.0f}s"
            " window is the default"
        )

    out += ["", "claims", *[check.line() for check in checks_], "", f"  {verdict(checks_)}"]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Re-measure the unsure-phantom rule's cost and coverage against real data."
    )
    ap.add_argument(
        "--generation", choices=sorted(GENERATIONS), default=DEFAULT.name,
        help="whose class list, resize_mode and export to measure against",
    )
    ap.add_argument("--weights", default=None, help="default: models/<generation>.pt as installed")
    ap.add_argument("--negatives", default=str(NEGATIVES_DIR),
                    help="the stored empty-counter frames (default: the clamp probe's own set)")
    ap.add_argument("--export-limit", type=int, default=None,
                    help="frames per split, for a smoke run only: the cost claims are about the "
                         "whole export, and a sample this small cannot support the comparison with "
                         "the clamp rule")
    ap.add_argument("--split", default=CONTROL_SPLIT, help="where the 60-frame control comes from")
    ap.add_argument("--real-sample", type=int, default=CONTROL_SAMPLE,
                    help=f"control frames (default {CONTROL_SAMPLE})")
    ap.add_argument("--live-seconds", type=float, default=LIVE_SECONDS,
                    help="length of the live empty-counter window; 0 skips it")
    ap.add_argument("--require-live-band", action="store_true",
                    help="fail unless the live window actually produced a band-shaped detection - "
                         "for a run that has to prove the band family, which a quiet counter cannot")
    ap.add_argument("--conf", type=float, default=None,
                    help="detection confidence to measure at; default is the app's own profile")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any claim fails, for use as a gate")
    ap.add_argument("--max-use-percent", type=int, default=resources.USE_PERCENT)
    ap.add_argument("--max-vram-percent", type=int, default=resources.VRAM_PERCENT)
    ap.add_argument("--disk-reserve-gb", type=float, default=resources.DISK_RESERVE_GB)
    ap.add_argument("--hard-vram-cap", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generation = generation_for(args.generation)
    weights = args.weights or str(SIDECAR_ROOT / "models" / generation.weight_name)
    if not Path(weights).exists():
        raise SystemExit(f"no such weight: {weights}")

    budget = resources.measure(args.max_use_percent, args.disk_reserve_gb, args.max_vram_percent)
    device = resources.resolve_device(args.device)
    budget = resources.apply(budget, device, args.hard_vram_cap)

    import cv2

    from app.camera import CameraCapture
    from app.inference import YoloDetector
    from app.models import requirement_for
    from app.pipeline import Pipeline
    from app.settings import Settings
    from app.settings_store import load_settings, resolve_resize_mode

    # The app's own profile when there is one, its defaults when there is not - `conf_threshold` and
    # `imgsz` are the operating point every detection here is produced at, so measuring the shipped
    # defaults on a machine that runs something else would describe a pipeline nobody uses.
    settings = load_settings(str(SETTINGS_PATH)) if SETTINGS_PATH.exists() else Settings()
    if args.conf is not None:
        settings.conf_threshold = args.conf
    # What the profile ships with, captured before the collection forces the rules off - so the
    # kept-pass and the claims describe the settings the operator is actually running.
    shipped_flags = (
        settings.suppress_clamped_detections,
        settings.suppress_frame_filling_detections,
        settings.suppress_unsure_phantoms,
    )
    # Forced, first for `clamp_probe`'s reasons and then this tool's own: a frame skip would report
    # the fraction of frames this happened to look at, an allowlist would shrink the populations by
    # class, and the sibling suppressions would refuse the very detections this tool exists to count.
    settings.infer_frame_skip = 0
    settings.class_allowlist = []
    settings.suppress_clamped_detections = False
    settings.suppress_frame_filling_detections = False
    settings.suppress_unsure_phantoms = False

    negatives_frames = read_frames(negative_frames(Path(args.negatives)), cv2)
    control_frames = read_frames(labelled_frames(generation, args.split, args.real_sample), cv2)

    source = Fixed()
    detector = YoloDetector(
        weights, device=device, conf=settings.conf_threshold, imgsz=settings.imgsz,
        resize_mode=generation.resize_mode,
    )
    pipeline = Pipeline(source, detector, settings, on_message=lambda _message: None)
    # Each population's two passes are **adjacent**, and that is not tidiness: `track(persist=True)`
    # carries state between passes, so a shipped pass that follows another population's frames can
    # see a detection the raw pass over its own frames did not - the difference was measured at one
    # detection on the 50 stored negatives before this ordering, and a claim that flakes by one is a
    # claim nobody trusts. The ships-then-flips order is the live-reload path
    # `PATCH /api/settings` takes, not a start-up configuration nobody runs.
    try:
        negatives_raw, negatives_frames_seen = walk(negatives_frames, source, pipeline)
        (
            settings.suppress_clamped_detections,
            settings.suppress_frame_filling_detections,
            settings.suppress_unsure_phantoms,
        ) = shipped_flags
        negatives_kept, _ = walk(negatives_frames, source, pipeline)
        settings.suppress_clamped_detections = False
        settings.suppress_frame_filling_detections = False
        settings.suppress_unsure_phantoms = False
        control_raw, control_frames_seen = walk(control_frames, source, pipeline)
        (
            settings.suppress_clamped_detections,
            settings.suppress_frame_filling_detections,
            settings.suppress_unsure_phantoms,
        ) = shipped_flags
        control_shipped, _ = walk(control_frames, source, pipeline)
    finally:
        detector.close()

    records: list[FrameRecord] = []
    per_split: list[tuple[str, int, int, int]] = []
    for split in EXPORT_SPLITS:
        got = audit_recall.collect(
            generation, weights, split, settings.conf_threshold, settings.imgsz,
            generation.resize_mode, device, args.export_limit,
        )
        records.extend(got)
        per_split.append(
            (
                split,
                len(got),
                sum(len(record.preds) for record in got),
                sum(len(match_instances(record.truth, record.preds).matched_pred) for record in got),
            )
        )
    matched_real, predictions = export_shots(generation, records)

    negative_ledger = ledger(
        "negatives (empty counter)", negatives_frames_seen, negatives_raw,
        clamped=shipped_flags[0], filling=shipped_flags[1], unsure_=shipped_flags[2],
        pipeline_shots=negatives_kept,
    )
    control_ledger = ledger(
        "control (real products)", control_frames_seen, control_raw,
        clamped=shipped_flags[0], filling=shipped_flags[1], unsure_=shipped_flags[2],
        pipeline_shots=control_shipped,
    )
    export_ledger = ledger(
        "export (every prediction)", len(records), export_population(generation, records),
        clamped=shipped_flags[0], filling=shipped_flags[1], unsure_=shipped_flags[2],
    )

    live_seconds = args.live_seconds
    live_scene = Scene(luminance=())
    live_raw: tuple[Shot, ...] = ()
    live_kept: tuple[Shot, ...] = ()
    if live_seconds > 0:
        # The live window runs the profile's own rules, so what escaped is observed rather than
        # re-derived - which is why the three flags are set here instead of being left wherever the
        # walks above finished with them.
        (
            settings.suppress_clamped_detections,
            settings.suppress_frame_filling_detections,
            settings.suppress_unsure_phantoms,
        ) = shipped_flags
        camera = CameraCapture(
            settings.camera_index, settings.capture_width, settings.capture_height,
            settings.capture_fps, brightness=settings.camera_brightness,
            exposure=settings.camera_exposure, autofocus=settings.camera_autofocus,
            focus=settings.camera_focus,
        )
        camera.open()
        live_detector = YoloDetector(
            settings.active_model, device=device, conf=settings.conf_threshold,
            imgsz=settings.imgsz,
            resize_mode=resolve_resize_mode(
                settings.resize_mode, settings.active_model,
                requirement_for(settings.active_model),
            ),
        )
        recorder = Recorded(live_detector)
        live_pipeline = Pipeline(camera, recorder, settings, on_message=lambda _m: None)
        try:
            live_raw, live_kept, live_scene = live_window(live_pipeline, camera, recorder, live_seconds)
        finally:
            live_detector.close()
            camera.release()
    live_ledger = ledger(
        "live (empty counter)", live_scene.frames, live_raw,
        clamped=shipped_flags[0], filling=shipped_flags[1], unsure_=shipped_flags[2],
        pipeline_shots=live_kept,
    )

    census = Census(
        ceiling=PHANTOM_CONF_CEILING,
        rows=tuple(
            ceiling_table(matched_real, negatives_raw, live_raw, control_raw)
        ),
        negatives=negative_ledger,
        control=control_ledger,
        live=live_ledger,
        export=export_ledger,
        per_split=tuple(per_split),
        predictions=predictions,
        shipped_flags=shipped_flags,
        live_scene=live_scene,
        require_live_band=args.require_live_band,
    )
    checks_ = checks(census)

    for line in report_lines(
        census, generation=generation, weights=weights, device=device,
        conf=settings.conf_threshold, shipped_conf=Settings().conf_threshold, imgsz=settings.imgsz,
        live_seconds=live_seconds, checks_=checks_,
    ):
        print(line)

    print()
    print(
        f"resource envelope: {budget.cpu_threads} threads, RAM cap {budget.ram_cap_gb:.1f} GB, "
        f"VRAM cap {budget.vram_cap_gb:.1f} GB"
    )
    print("The rule itself, and where it is applied: app/acceptance.py (is_unsure_phantom)")
    print("The clamp rule's own cost, and the tolerance behind it: tools/clamp_probe.py")
    return 1 if args.strict and strict_failed(checks_) else 0


if __name__ == "__main__":
    raise SystemExit(main())
