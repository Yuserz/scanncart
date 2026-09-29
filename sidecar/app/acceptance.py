"""The one owner of the accept/reject decision for a frame's detections.

Every question of the form "does this detection reach the frame, the item log and the overlay?" is
answered here, for all the rules at once, and `Pipeline.process_once` calls exactly one function to
ask it. Before this module the decision was two rules written inline in that loop - the class
allowlist and the frame-clamp filter - which was fine while there were two and stops being fine the
moment there are three, because the rules interact: a box refused by the allowlist must not also be
counted as a shape, or the counts an operator reads beside the item log add up to more detections
than the frame held.

**Why a reject path is worth having at all.** The grocery head predicts seven products and has no
background, unknown or "none of the above" output, so a frame containing no product at all is still
forced onto one of the seven, confidently: on twenty live camera frames with nothing placed, the
model returned one box covering ~96% of the image as *Bear Brand* on every frame, at 0.90-0.94.
Every runtime gate that existed sat downstream of that label - a class list cannot refuse a label
that is on it, and the confidence threshold needs 0.95 to bite, which would take real items with
it. So the reject path is a decision about *shape*, made before anything else sees the detection.

**What that costs, measured, because it is not free.** Run through this module's own rule over the
whole v1 export - 1815 labelled frames, 2051 detections, 2018 of them matching their ground truth
one-to-one and class-for-class at the generation's own geometry:

    rule                          matched detections dropped   live phantoms   empty-counter phantoms
    clamped (all four edges)                36 / 2018             0 / 20             19 / 25
    frame-filling (three edges)            252 / 2018            20 / 20             25 / 25
    unsure (shape + confidence)              6 / 2018             -                 19 / 25

`tools/unsure_probe.py` re-derives all three of those rows from the repo (`make verify-unsure`), and
that is the only reason the third can be trusted: its figures were first produced by scratch
harnesses that were deleted. The figures themselves are written down **once**, in `MEASURED_COST`
below - these rows, `Settings`' comments beside the three flags and the desktop's own field hints
are all checked against it (`tests/test_cost_figures.py`), and that tool fails its own run when a
fresh measurement stops agreeing with it, so the table above cannot quietly outlive its measurement.
The clamp rule keeps its own tool and its own five claims.

The two shape populations **overlap**, and that is the honest finding rather than a tuning problem:
of the 252 ground-truth-matched detections the frame-filling rule would drop, the closest to the
phantoms is a `lucky_me_pancit_canton_calamansi_flavor` pouch at area 0.972 and confidence 0.966
with three edges inside 1% - named by the tool's own cost block. It is a real product, held right up
to the lens. No single-frame box statistic separates it from the phantom, and the per-class
alternative was measured and fails too: the weights' own training labels contain near-full-frame
boxes for *every* class (Bear Brand up to 0.995 of the frame), so "bigger than this class was ever
taught" rejects 0 of the 20 live phantoms.

**What the rules leave behind, and it is not nothing - the family they cover is one of two.** Every
frame through two real `Pipeline`s (the default, and both rules off) at the app's own capture mode,
640x480@60 and `letterbox`, with no toggle touched:

    population                                raw    survives the default
    live camera, empty counter, 2700 frames  2700     0   (314 clamped, 2386 frame-filling)
    the 50 stored empty-counter negatives      12     0   ( 11 clamped,    1 frame-filling)
    60 real product frames, train split        60    59   (the close-up named above)

Those 2700 frames are one scene, and in it the phantom is Bear Brand on every frame, 0.96-0.99 of the
frame at 0.868-0.954 confidence, pinned to three or four edges - which is why the two rules above
cover it whole. **A second scene showed the model has more than one way to say this.** A *band* lying
on the bottom edge - ~0.46 of the frame's height, ~0.92 of its width, 0.50-0.79 confidence - reached
the frame on 158 of 1316 live frames and wrote **10 item-log rows in 25 seconds** on an empty
counter, and a live window at the geometry v1's own record requires (`stretch`) produced 1024
detections that were frame-spanning boxes at 0.500-0.831. Neither family pins three or four edges,
so neither shape rule could see them, and shape alone cannot separate them from real products: the
nearest real box to the band is a `safeguard_pure_white_60g` at 0.956 confidence and 0.635 of the
frame's height, *the same shape*, and a real close-up is the same frame-spanning shape the stretch
phantom is. Confirmed by pricing the shape-only attempts against the 59 real detections the default
keeps: the band alone costs 1, a confidence floor alone 4, `pinned >= 2` 6, `pinned >= 1` 25.

**The third rule, and why it is the pair rather than either half.** `is_unsure_phantom` reads the
shape *and* the confidence together: below `PHANTOM_CONF_CEILING` (0.85) and either frame-spanning or
an edge band. That is the one partition every measurement agrees on - the confident members of both
shapes are real products while the uncertain ones are phantoms in every population - and priced over
the whole export it drops **6 of the 2018 ground-truth-matched detections, 0.30%**, every one of them
a large box the model was unsure of (0.719-0.839), against the clamp rule's 36 and the frame-filling
rule's 252 on the same population. It ships on (`suppress_unsure_phantoms`) and it is the cheapest of
the three rules by a factor of six. Coverage, each population through the real `Pipeline` with the
rules as they ship: the 50 stored negatives 0 of 25 raw; 2700 live frames at the app's geometry 0 of
2700; a live window at the required geometry 0 of 1024; the band family 0 of 862 in a 1500-frame
window. Through the app's own routes on an empty counter: a 20 s capture logged 0 item-log rows with
the rule on and **17** with it off over the same scene, and the rule turned back on logged 0 again in
1549 frames.

**The band family is the one population a tool has to ask for, and a silent scene measures it not at
all.** It exists only in front of a camera - the stored negatives are whole-frame boxes - so
`tools/unsure_probe.py` opens a live empty-counter window and prices the band shape against it. A
window that produced *no* detections is reported as unexercised rather than as coverage, which is
what the day this was written found: 1768 frames, 0 detections, a scene whose mean luminance was
14/255 against the stored negatives' 32. A lit counter is the only thing that measures that family,
and the tool says so instead of passing on nothing.

Which is why the frame-filling rule ships **on** (`suppress_frame_filling_detections`), and why that
cost is written here rather than discovered later. On the day this default flipped, the alternative
was a lever nobody flips: a fresh install showed the operator `Bear Brand` on an empty counter with
an item-log row to match, and a setting buried in the Admin panel does not change what the app does
out of the box. So the ~12% is paid deliberately, on the operator's behalf, for a defect the app can
otherwise be dismissed for - and the single control frame it costs (one `lucky_me_pancit_canton`
close-up, IoU 0.879, of the 60 in `tools/clamp_probe.py`'s real population) is the named price of
that default rather than something nobody looked at.

The escape hatch is the setting itself: it is hot-reloadable, so an operator watching an item held
close stop registering turns it off mid-capture and the next frame goes through. The rule that
removes the phantom *without* that cost is the model's - train the staged hard negatives in, which
is what v2's dataset exists for. `tools/clamp_probe.py` forces this rule off when it reports, so its
five claims keep describing the clamp rule's own cost rather than this one's, and
`tools/unsure_probe.py` (`make verify-unsure`) re-derives this rule's own - both its price on the
export and its catch on the two phantom families from their real sources. `Settings` is where the
shipping value lives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.schemas import Detection

#: Why a detection was declined. Strings rather than an enum because they end up beside the item
#: log as counts, where the readable word is the point.
CLASS_REASON = "class"
CLAMPED_REASON = "clamped"
FRAME_FILLING_REASON = "frame_filling"
UNSURE_REASON = "unsure"


@dataclass(frozen=True)
class RuleCost:
    """The measured price of the rules, and the one place those figures are written down.

    A record rather than a paragraph because the same four numbers are quoted in six files that
    cannot import each other: this module's own docstring, `Settings`' comments beside the three
    flags, `CLAUDE.md`, this module's own tests, and the desktop's settings defaults and field
    hints. Each was hand-typed,
    and the frame-filling rule's price is what that cost: the descriptions in `Settings` and both
    desktop hints had kept a denominator and a count from a deleted scratch harness, while the
    table above them had already been re-derived from the export - in the same commit that added
    the tool which contradicted them. The retired figures are tombstones in
    `tests/test_cost_figures.py`, which is also why this paragraph describes them instead of
    quoting them: the guard cannot tell a live number from one being remembered.

    So the counts live here and nowhere else. The prose copies are **checked against** them
    (`tests/test_cost_figures.py`, which also refuses the retired spellings), and
    `tools/unsure_probe.py` fails its own run when a fresh measurement stops agreeing with this
    record under `--strict` - so re-measuring forces the figure to move here rather than leaving
    every copy behind it, none of which a re-run ever touches. Nothing here is derived from the
    weights: this is a measurement of one operating point, quoted at it, and the tool prints that
    operating point beside its answer.
    """

    #: Labelled frames in the whole v1 export, and the detections the model produced over them.
    frames: int
    predictions: int
    #: Detections matching their ground truth one-to-one and class-for-class: the denominator every
    #: share below is a fraction of, and the population the three costs are comparable on.
    matched: int
    #: Each rule's own count on that population, under the first rule that refused the detection -
    #: the order `accept_detections` applies them in.
    clamp: int
    frame_filling: int
    unsure: int
    #: The `conf_threshold` this was measured at. Carried rather than assumed because a cost quoted
    #: without it is not reproducible: a phantom is a low-confidence detection, so the counts scale
    #: with the threshold and the tool refuses to compare its run against this record without it.
    conf: float = 0.5

    @property
    def rules(self) -> dict[str, int]:
        """The three shape rules' counts, keyed by the reason a detection is refused under.

        Keyed by the reason vocabulary rather than by a spelling of each rule's name, so a reader
        that already holds `CLAMPED_REASON` does not need a second mapping to find its price.
        """
        return {
            CLAMPED_REASON: self.clamp,
            FRAME_FILLING_REASON: self.frame_filling,
            UNSURE_REASON: self.unsure,
        }

    def share(self, count: int) -> str:
        """`count` as a share of the matched population, at two significant figures.

        Two, because that is what every sentence already quoted and the conventions differ by
        magnitude: 36/2018 is 1.8%, 252/2018 is 12%, 6/2018 is 0.30%. Stated once here so a
        re-measurement cannot leave a percentage that means something other than the count printed
        next to it.
        """
        percent = 100.0 * count / self.matched
        if percent >= 10.0:
            return f"{percent:.0f}%"
        if percent >= 1.0:
            return f"{percent:.1f}%"
        return f"{percent:.2f}%"


#: The figures every sentence about these rules quotes. See `RuleCost` for why they live here;
#: `tools/unsure_probe.py` re-derives them from the export (`make verify-unsure`) and fails when its
#: own measurement disagrees, which is what keeps this record a measurement rather than a belief.
MEASURED_COST = RuleCost(frames=1815, predictions=2051, matched=2018, clamp=36, frame_filling=252, unsure=6)

#: How close to a frame edge, as a fraction of width/height, a box must sit on **all four** sides
#: before it counts as a prediction the model wanted larger than the image.
#:
#: Chosen from measurement, not taste. Across the 25 detections the 50 empty-counter negatives
#: produced and the 60 detections from labelled product frames - both at `conf_threshold` 0.5, and
#: both re-runnable with `tools/clamp_probe.py --generation v1 --conf 0.5`, which re-checks the
#: end-to-end behaviour too - the distance from the nearest edge on a box's *worst* side separated
#: the two populations at this value:
#:
#:     worst edge      phantoms caught     real lost     training labels rejected
#:     <= 0.002            19/25             0/60             0.82%
#:     <= 0.010            19/25             0/60             1.43%
#:     <= 0.020            23/25             1/60             1.70%
#:
#: 0.01 is the loosest value that loses **no** real detection of that 60-frame control: the
#: tightest real one sits at 0.0109, so 0.02 would buy four more phantoms by dropping a real item.
#: The six phantoms this still misses lie between 0.0124 and 0.0286 - *inside* the band real
#: detections occupy - which is why the rule stops here rather than chasing them: past this point
#: it stops being a phantom filter and starts being a frame-filling-object filter. (It is also not
#: free over the whole export rather than the control: 36 of 2018 ground-truth-matched detections,
#: 1.8%, which is that same 1.43% seen through a bigger sample.)
#:
#: Those counts move with `conf_threshold` and are quoted at its 0.5 default: a phantom is a
#: *low-confidence* detection, so the same weights over the same 50 frames produce 25 of them at 0.5
#: and 15 at 0.7. The rule holds at both - no real detection is inside it at either - but a number
#: quoted without the threshold it was measured at is not reproducible, which is why the tool above
#: prints the operating point and flags it when the running profile differs from the default.
CLAMPED_EDGE_TOLERANCE = 0.01


def pinned_edges(box, tolerance: float = CLAMPED_EDGE_TOLERANCE) -> int:
    """How many of the four frame edges this box sits within `tolerance` of.

    The one measurement both shape rules are expressed in, so they cannot drift apart: the clamp
    rule is four pinned edges and the frame-filling rule is three. Pure and total, so a rule can be
    tested against synthetic boxes rather than a camera.

    **Each side keeps its own comparison direction, and that is load-bearing rather than tidy.**
    A box at `(0.01, 0.01, 0.99, 0.99)` is pinned on all four sides at a tolerance of 0.01 and must
    stay that way: written as a distance, `1.0 - 0.99` is `0.010000000000000009`, so a
    `distance <= tolerance` test rejects it while the shipped `x2 >= 1.0 - tolerance` accepts it.
    The boundary is inclusive and real detections sit one ten-thousandth away from it, so which
    side of the float a side is measured from is the difference between a rule that behaves as
    documented and one that silently misses its own edge case.
    """
    x1, y1, x2, y2 = box
    return sum(
        (
            x1 <= tolerance,
            y1 <= tolerance,
            x2 >= 1.0 - tolerance,
            y2 >= 1.0 - tolerance,
        )
    )


def is_clamped_to_frame(box, tolerance: float = CLAMPED_EDGE_TOLERANCE) -> bool:
    """Whether a box is pinned to *all four* frame edges: a prediction larger than the image.

    All four, not one or two. A real close-up overflows the frame on the sides the object leaves
    through, so touching a single edge means nothing — v1's own labels show that, with 63% of Bear
    Brand's training instances touching one. Touching *every* side is the shape a box takes when
    the model wanted something bigger than the canvas and `normalize_detections` trimmed it, and
    that is the signature the empty-counter false positive has: `x1 0.0000`, `y1 0.0001`,
    `x2 0.9995`, `y2` exactly `1.0000`, held across 16 consecutive frames.
    """
    return pinned_edges(box, tolerance) == 4


def is_frame_filling(box, tolerance: float = CLAMPED_EDGE_TOLERANCE) -> bool:
    """Whether a box is pinned to *exactly three* frame edges: the clamped shape, one edge short.

    This is the live false positive's shape, and it is why the clamp rule alone did not catch it.
    Measured with nothing placed in front of the camera, twenty frames of twenty returned one
    `Bear Brand` box at 0.90-0.94 sitting at `(0.05, 0.0, 1.0, 1.0)` - pinned to the top, right and
    bottom edges and stopping 5% inside on the left, which is five times the tolerance the all-four
    rule tests at. Same defect, one side short of the rule that was looking for it.

    Exactly three rather than "at least three", because the two predicates are applied in order by
    `accept_detections`: a box pinned on all four is the clamp rule's, and is counted as such.

    Off by default: see this module's docstring for the measured price and why it is the operator's
    call rather than ours.
    """
    return pinned_edges(box, tolerance) == 3


#: The confidence below which a frame-spanning or edge-band box is this model hallucinating.
#:
#: The one number the shape rules alone could not supply, and it is measured rather than chosen for
#: the look of it. Over the whole v1 export - 1815 labelled frames, 2051 detections, 2018 of them
#: matching their ground truth class-for-class at the generation's own geometry - every large box and
#: every edge band the model is *sure* about is a real product: the 6 ground-truth-matched
#: detections this rule does drop sit at 0.719-0.839, while the ones it must not sit at 0.92-0.97.
#: On the phantom side every population is below it: the 50 stored negatives at 0.510-0.928 (19 of
#: 25 under the ceiling), a live empty-counter window at the app's geometry 0.500-0.858 across 862
#: detections, and the same window at the geometry v1's record requires 0.500-0.831 across 1024. On
#: the export the 6 matched detections it drops sit at 0.719-0.839 and the lowest-confidence shaped
#: one it keeps sits at 0.856 - a gap no matched detection occupies, which `tools/unsure_probe.py`
#: prints and fails a claim on if it ever closes.
#:
#: So it is a floor for a *shape*, not a second `conf_threshold`: a small box the model is unsure
#: about is kept, because no measurement here says small low-confidence boxes are phantoms and the
#: export holds real ones at 0.52-0.63.
PHANTOM_CONF_CEILING = 0.85

#: How much of the frame a box must cover, in area, before "the model was not sure about it" is
#: evidence of a hallucination rather than of a hard item. Measured: the phantom family the app sees
#: under the geometry v1's record requires spans 0.88-0.98 of the frame (1024 of 1024 detections in
#: a live empty-counter window), while the matched detections this half costs sit at 0.719-0.839.
FRAME_SPANNING_AREA = 0.85

#: The edge-band shape: a wide, short rectangle lying along the top or bottom edge. This is the
#: second family the model produces on an empty counter - `(0.0756, 0.5407, 0.9991, 1.0)` at 0.677,
#: read off a live window - and it is why the two shape rules above are not enough: it pins one or
#: two edges, never three or four. Bounds measured on the same window (heights 0.46 against the
#: 0.65 here; widths 0.92 against the 0.5) with the nearest real band-shaped detection in the export
#: at 0.635 of the frame's height and 0.956 confidence - which the ceiling above excludes, and which
#: is why the shape needs the floor to be a rule rather than a cut.
BAND_MAX_HEIGHT = 0.65
BAND_MIN_WIDTH = 0.5


def is_edge_band(box, tolerance: float = CLAMPED_EDGE_TOLERANCE) -> bool:
    """Whether a box is a wide, short band lying on the top or bottom edge of the frame."""
    x1, y1, x2, y2 = box
    on_horizontal_edge = y1 <= tolerance or y2 >= 1.0 - tolerance
    return on_horizontal_edge and (x2 - x1) >= BAND_MIN_WIDTH and (y2 - y1) <= BAND_MAX_HEIGHT


def is_unsure_phantom(box, conf: float) -> bool:
    """Whether a detection is this model's uncertain guess at a counter with nothing on it.

    Two shapes and one confidence: a box covering most of the frame, or a wide band along a
    horizontal edge, that the model is not sure about. Shape alone cannot decide it - the same band
    shape is a real `safeguard_pure_white_60g` held at the top of the frame and the same
    frame-spanning shape is a real close-up - and confidence alone cannot either, because the
    earlier whole-frame family reaches 0.954 (those are the two shape rules' job). What separates
    the two populations in every measurement is the pair: the confident members of both shapes are
    real products, and the unconfident members are phantoms.
    """
    return conf < PHANTOM_CONF_CEILING and (
        (box[2] - box[0]) * (box[3] - box[1]) >= FRAME_SPANNING_AREA or is_edge_band(box)
    )


def drop_clamped_detections(
    detections: list[Detection], tolerance: float = CLAMPED_EDGE_TOLERANCE
) -> tuple[list[Detection], int]:
    """The detections that are not frame-clamped, and how many were removed.

    Returns the count as well as the list because a suppression nobody can see is the failure this
    exists to avoid: the phantom is dropped from the overlay, the item log and the database, and
    without a number travelling with it there is no evidence the rule did anything at all — the
    operator simply sees a model that appears not to have this defect.

    Kept as a named helper for the tools that measure the clamp rule on its own
    (`tools/clamp_probe.py`); `Pipeline` goes through `accept_detections` so the rules share one
    decision.
    """
    kept = [d for d in detections if not is_clamped_to_frame(d.box, tolerance)]
    return kept, len(detections) - len(kept)


@dataclass(frozen=True)
class Acceptance:
    """What survived the decision, and how many were declined under each reason.

    `rejected` is keyed by the first reason that applied to a detection, so the counts sum to the
    number that was refused rather than to a detection's coincidences.
    """

    accepted: list[Detection]
    rejected: dict[str, int]

    @property
    def suppressed(self) -> int:
        """How many detections the *shape* rules dropped - and deliberately not the class rule.

        This is the number `Stats.suppressed` carries to the Live view, where it is a readout of a
        model defect the operator did not ask for. A class-allowlist drop is the operator's own
        narrowing - they chose the list - and counting it here would put their decision on a tile
        that means "the weights predicted something wrong", as well as change a value the empty-feed
        end-to-end test already pins at zero.
        """
        return (
            self.rejected.get(CLAMPED_REASON, 0)
            + self.rejected.get(FRAME_FILLING_REASON, 0)
            + self.rejected.get(UNSURE_REASON, 0)
        )


def accept_detections(
    detections: Sequence[Detection],
    *,
    class_allowlist: Iterable[str] = (),
    suppress_clamped: bool = False,
    suppress_frame_filling: bool = False,
    suppress_unsure: bool = False,
    tolerance: float = CLAMPED_EDGE_TOLERANCE,
) -> Acceptance:
    """The accept/reject decision, in one place, over every rule the app has.

    Rules are applied in the order they are stated, and a detection is counted under the **first**
    rule that refuses it:

    1. **The class allowlist**, when one is set. An empty list means keep everything, which is the
       default: a class list is the operator narrowing the view, not a verdict about the model.
    2. **The clamp filter** (`suppress_clamped`, on by default): four edges.
    3. **The frame-filling filter** (`suppress_frame_filling`, on by default): three edges.
    4. **The unsure-phantom filter** (`suppress_unsure`, on by default): a frame-spanning box or an
       edge band that the model is not confident about - the two families the shape rules miss.

    Everything this function does not accept is absent from the returned list, which is the only
    list `Pipeline` logs or streams - so declining a detection here is what keeps it out of the
    item log, the overlay and the frame message at once, rather than being counted beside them.

    Pure: no settings object, no clock, no I/O - so the policy can be tested against synthetic
    boxes and the measured populations, and the caller decides what the settings mean.
    """
    allow = {str(name) for name in class_allowlist}
    accepted: list[Detection] = []
    rejected: dict[str, int] = {}

    def refuse(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for detection in detections:
        if allow and detection.cls not in allow:
            refuse(CLASS_REASON)
            continue
        if suppress_clamped and is_clamped_to_frame(detection.box, tolerance):
            refuse(CLAMPED_REASON)
            continue
        if suppress_frame_filling and is_frame_filling(detection.box, tolerance):
            refuse(FRAME_FILLING_REASON)
            continue
        if suppress_unsure and is_unsure_phantom(detection.box, detection.conf):
            refuse(UNSURE_REASON)
            continue
        accepted.append(detection)

    return Acceptance(accepted=accepted, rejected=rejected)
