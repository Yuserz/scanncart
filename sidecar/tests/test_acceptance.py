"""The accept/reject decision: one owner, and what it declines never ships.

`app/acceptance.py` is the single place that answers "does this detection reach the frame, the item
log and the overlay?". It was two rules written inline in `Pipeline.process_once` (`class_allowlist`
and the frame-clamp filter); the same policy, plus a third shape the clamp rule leaves behind, now
has one owner, so a rule can be read, tested and measured without hunting through the loop.

The tests below are in two halves, and they are testing different things:

* the **pure layer** (`accept_detections` and the shape predicates) - which detection goes, under
  which reason, and what the defaults are;
* the **wiring** (`Pipeline.process_once`) - that a declined detection is actually gone from the
  frame message, the logging store and therefore the overlay, rather than merely counted beside
  them. A suppression whose only trace is a number is how an operator ends up looking at a model
  that appears healthy while an item it can see is not being logged.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.acceptance import (
    CLAMPED_REASON,
    CLASS_REASON,
    FRAME_FILLING_REASON,
    UNSURE_REASON,
    accept_detections,
    is_clamped_to_frame,
    is_frame_filling,
    pinned_edges,
)
from app.pipeline import Pipeline
from app.schemas import Detection
from app.settings import Settings

# The measurement this whole module exists for: a live camera frame with nothing placed in it
# produced this box, as Bear Brand, on 20 of 20 frames (see the mission's investigation). It is
# pinned to three edges and stops ~5% short of the left one, which is why the all-four-edges rule
# - whose tolerance is 0.01 - never saw it.
PHANTOM = (0.05, 0.0, 1.0, 1.0)
# The shape the shipped clamp rule already catches: the model wanted something larger than the
# image, and normalization trimmed every side.
CLAMPED = (0.0, 0.0, 1.0, 1.0)
# A real product held off the edges entirely.
ORDINARY = (0.2, 0.2, 0.8, 0.8)
# Two edges: a close-up that overflows the frame on the sides the object leaves through, which the
# rule must not touch.
TWO_SIDED = (0.0, 0.0, 0.8, 0.8)


def _det(cls="banana", box=ORDINARY, conf=0.9, track_id=1):
    return Detection(track_id=track_id, cls=cls, conf=conf, box=box)


# --------------------------------------------------------------------------
# the pure layer: the shapes
# --------------------------------------------------------------------------


def test_pinned_edges_counts_how_many_sides_sit_on_the_frame():
    assert pinned_edges(CLAMPED) == 4
    assert pinned_edges(PHANTOM) == 3
    assert pinned_edges(TWO_SIDED) == 2
    assert pinned_edges(ORDINARY) == 0


def test_the_clamp_predicate_is_exactly_four_pinned_edges():
    """The shipped rule keeps its meaning: all four, never three.

    Pinned here rather than assumed, because the new rule is one edge away from it and a future
    loose reading ("at least three") would silently widen the shipped suppression instead.
    """
    assert is_clamped_to_frame(CLAMPED) is True
    assert is_clamped_to_frame(PHANTOM) is False
    assert is_clamped_to_frame(TWO_SIDED) is False


def test_the_frame_filling_predicate_is_exactly_three_pinned_edges():
    assert is_frame_filling(PHANTOM) is True
    assert is_frame_filling(CLAMPED) is False  # the clamp rule's shape, one edge further
    assert is_frame_filling(TWO_SIDED) is False
    assert is_frame_filling(ORDINARY) is False


# --------------------------------------------------------------------------
# the pure layer: the decision
# --------------------------------------------------------------------------


def test_nothing_is_dropped_when_every_rule_is_off():
    detections = [_det(box=CLAMPED), _det(box=PHANTOM), _det(box=ORDINARY)]
    result = accept_detections(detections)
    assert result.accepted == detections
    assert result.rejected == {}


def test_the_allowlist_decides_by_class_and_an_empty_one_keeps_everything():
    detections = [_det(cls="banana"), _det(cls="apple")]
    assert accept_detections(detections).accepted == detections

    result = accept_detections(detections, class_allowlist=("banana",))
    assert [d.cls for d in result.accepted] == ["banana"]
    assert result.rejected == {CLASS_REASON: 1}


def test_each_rule_reports_its_own_reason_and_nothing_is_counted_twice():
    """A box the allowlist already refused is not also counted as a shape.

    The counts are what an operator reads beside the item log, so a detection that is gone for two
    reasons has to be reported once - the first reason that applied - or the numbers add up to more
    detections than the frame ever held.
    """
    detections = [
        _det(cls="apple", box=CLAMPED),      # refused by class before its shape is considered
        _det(cls="banana", box=CLAMPED),
        _det(cls="banana", box=PHANTOM),
        _det(cls="banana", box=ORDINARY),
    ]
    result = accept_detections(
        detections,
        class_allowlist=("banana",),
        suppress_clamped=True,
        suppress_frame_filling=True,
    )
    assert len(result.accepted) == 1
    assert result.accepted[0].box == ORDINARY
    assert result.rejected == {CLASS_REASON: 1, CLAMPED_REASON: 1, FRAME_FILLING_REASON: 1}


def test_every_detection_is_either_accepted_or_counted():
    detections = [
        _det(cls="apple", box=CLAMPED),
        _det(cls="banana", box=CLAMPED),
        _det(cls="banana", box=PHANTOM),
        _det(cls="banana", box=TWO_SIDED),
        _det(cls="banana", box=ORDINARY),
    ]
    result = accept_detections(
        detections, class_allowlist=("banana",), suppress_clamped=True,
        suppress_frame_filling=True,
    )
    assert len(result.accepted) + sum(result.rejected.values()) == len(detections)


def test_an_allowlist_drop_is_not_counted_as_suppressed():
    """The Live view's `suppressed` tile is about the model, not about the operator's class list.

    `Stats.suppressed` reads as "the weights predicted something wrong and the app hid it". A class
    the operator excluded is their own narrowing, so it must not appear there - the empty-feed
    end-to-end test pins the same value at zero over a stream where every box is off the list.
    """
    narrowed = accept_detections([_det(cls="person", box=PHANTOM)], class_allowlist=("banana",))
    assert narrowed.accepted == []
    assert narrowed.rejected == {CLASS_REASON: 1}
    assert narrowed.suppressed == 0

    shape = accept_detections([_det(cls="banana", box=PHANTOM)], suppress_frame_filling=True)
    assert shape.rejected == {FRAME_FILLING_REASON: 1}
    assert shape.suppressed == 1


def test_the_frame_filling_rule_is_on_by_default():
    """On, because the phantom it removes is the defect the app is judged on.

    A fresh install logged `Bear Brand` at 0.90-0.94 on an empty counter, with an item-log row, and
    a lever nobody flips is not a fix. The price is measured and named rather than hidden: it also
    drops one of the 60 detections in the clamp tool's control population, a real frame-filling
    close-up at area 0.972 with three edges inside the 1% tolerance, and 252 of the 2018
    ground-truth-matched detections (12%). That is
    the trade this default makes on the operator's behalf, and the setting is the way back out.
    """
    assert Settings().suppress_frame_filling_detections is True
    result = accept_detections([_det(box=PHANTOM)], suppress_clamped=True, suppress_frame_filling=True)
    assert result.accepted == []


def test_the_shipped_default_drops_the_phantom_through_the_pipeline():
    """The default, not the flag: a bare `Settings()` has to be what removes it."""
    message = _pipeline([PHANTOM, ORDINARY]).process_once()
    assert [d["box"] for d in message["detections"]] == [ORDINARY]
    assert message["stats"]["suppressed"] == 1


# --------------------------------------------------------------------------
# the wiring: what the pipeline does about it
# --------------------------------------------------------------------------


class _StubSource:
    width = 128
    height = 96
    fps = 30.0

    def latest(self):
        return (1, np.full((96, 128, 3), 100, dtype=np.uint8))


class _BoxDetector:
    """Emits whatever boxes it was handed, so the decision is the only variable."""

    names = {0: "banana"}

    def __init__(self, *boxes, conf=0.9):
        self._boxes = boxes
        # The confidence is the caller's, because one rule now reads it: a phantom is defined by a
        # shape *and* an unsure model, so a double that always answered 0.9 could only test half.
        self._conf = conf

    def infer(self, frame):
        return [
            Detection(track_id=i + 1, cls="banana", conf=self._conf, box=box)
            for i, box in enumerate(self._boxes)
        ]


class _RecordingStore:
    def __init__(self):
        self.rows = []

    def record_detection(self, session_id, track_id, cls, conf, ts):
        self.rows.append((track_id, cls, conf))

    def resolve_left(self, session_id, track_id, ts):
        pass


def _pipeline(boxes, settings=None, store=None, conf=0.9):
    # The mirror is off so the box this test reasons about is the box that comes back: it is a
    # preview-path concern with its own tests in test_pipeline.py.
    settings = settings or Settings(preview_mirror=False)
    return Pipeline(
        _StubSource(), _BoxDetector(*boxes, conf=conf), settings,
        on_message=lambda _m: None,
        logging_store=store if store is not None else _RecordingStore(),
        session_id=1,
    )


def test_a_frame_filling_detection_reaches_nothing_when_the_rule_is_on():
    """Gone from the message (so the overlay), the item log and the store - not counted beside them.

    The overlay draws the message's `detections`, so the message is the assertion for both; what
    this pins is that the suppression happens *before* `_log_detections`, because a rule applied
    after logging would still fill the item log with an item nobody can see on screen.
    """
    store = _RecordingStore()
    settings = Settings(preview_mirror=False, suppress_frame_filling_detections=True)
    pipe = _pipeline([PHANTOM, ORDINARY], settings=settings, store=store)

    message = pipe.process_once()

    assert [d["box"] for d in message["detections"]] == [ORDINARY]
    # The phantom is track 1 and the ordinary box track 2, so the store holding only track 2 is the
    # item log not having seen it - which a count beside the message would not have caught.
    assert [row[0] for row in store.rows] == [2]
    assert message["stats"]["suppressed"] == 1


def test_the_phantom_is_kept_when_the_rule_is_turned_off():
    store = _RecordingStore()
    settings = Settings(preview_mirror=False, suppress_frame_filling_detections=False)
    pipe = _pipeline([PHANTOM], settings=settings, store=store)

    message = pipe.process_once()

    assert len(message["detections"]) == 1
    assert message["stats"]["suppressed"] == 0
    assert len(store.rows) == 1


def test_the_suppressed_count_adds_both_shapes_together():
    settings = Settings(
        preview_mirror=False,
        suppress_clamped_detections=True,
        suppress_frame_filling_detections=True,
    )
    message = _pipeline([CLAMPED, PHANTOM, ORDINARY], settings=settings).process_once()

    assert [d["box"] for d in message["detections"]] == [ORDINARY]
    assert message["stats"]["suppressed"] == 2


def test_turning_the_rule_off_mid_capture_takes_effect_without_a_restart():
    settings = Settings(preview_mirror=False)
    pipe = _pipeline([PHANTOM], settings=settings)
    assert pipe.process_once()["detections"] == []

    settings.suppress_frame_filling_detections = False

    assert len(pipe.process_once()["detections"]) == 1


# The second family, as it appears live: a band on the bottom edge, one or two edges pinned, low
# confidence. Both boxes below were read off a real capture (`data/_frozen.json`, this session's
# 1500-frame window at the app's own geometry) rather than invented.
BAND_LIVE = (0.0756, 0.5407, 0.9991, 1.0)          # conf 0.677, area 0.424, two edges
BAND_LIVE_LOW = (0.0789, 0.5405, 1.0, 1.0)         # conf 0.503, same family
BAND_LIVE_PINNED_ONE = (0.1644, 0.5216, 0.7671, 1.0)  # conf 0.589, one edge, area 0.288
# The same phantom under the geometry v1's own record requires (`stretch`, where the model reads the
# whole counter): a frame-spanning box pinned to two edges, which neither shape rule touches.
STRETCH_PHANTOM = (0.0647, 0.017, 0.9903, 1.0)     # conf 0.531, area 0.910, two edges
# Real detections from the same export that the rule must NOT take: a confident close-up, and a
# small box the model was unsure about (a global confidence floor would drop the second one).
REAL_CONFIDENT_CLOSE_UP = (0.0393, 0.0292, 0.9553, 0.9686)  # conf 0.973, area 0.860
REAL_SMALL_UNSURE = (0.0589, 0.3452, 0.4302, 0.7247)        # conf 0.626, area 0.141


def test_the_live_band_phantom_is_declined():
    """The family that wrote 10 item-log rows on an empty counter in 25 s.

    It pins one or two edges, so neither shape rule reaches it - and it is the box that made the
    reported symptom true on the shipping path after the frame-filling rule shipped.
    """
    for box, conf in ((BAND_LIVE, 0.677), (BAND_LIVE_LOW, 0.503), (BAND_LIVE_PINNED_ONE, 0.589)):
        assert is_clamped_to_frame(box) is False
        assert is_frame_filling(box) is False
        result = accept_detections([_det(cls="Bear Brand", box=box, conf=conf)],
                                   suppress_clamped=True, suppress_frame_filling=True,
                                   suppress_unsure=True)
        assert result.accepted == [], box
        assert result.rejected == {UNSURE_REASON: 1}
        assert result.suppressed == 1


def test_the_phantom_under_the_weights_own_geometry_is_declined_too():
    """`stretch` is what v1's record requires; the phantom is a frame-spanning box there.

    Measured over a live empty-counter window at that geometry: 1024 detections, every one of them
    spanning 0.88-0.98 of the frame at 0.500-0.831 confidence, and the two shape rules declining
    292 of them. Shipping a rule that only knew the band shape would leave the app exposed on the
    configuration the record says is correct.
    """
    result = accept_detections([_det(cls="Bear Brand", box=STRETCH_PHANTOM, conf=0.531)],
                               suppress_unsure=True)
    assert result.accepted == []
    assert result.rejected == {UNSURE_REASON: 1}


def test_a_confident_close_up_is_kept_and_a_small_unsure_box_is_kept():
    """The rule is a shape *and* a confidence floor, not a second `conf_threshold`.

    Both boxes are real, from the export the rule's cost was measured on: a close-up the model is
    sure about (0.973) must survive whatever its shape, and a small box it is unsure about (0.626)
    must survive because no shape rule has ever claimed small boxes are phantoms.
    """
    kept = accept_detections(
        [_det(cls="Bear Brand", box=REAL_CONFIDENT_CLOSE_UP, conf=0.973),
         _det(cls="555 sardines", box=REAL_SMALL_UNSURE, conf=0.626, track_id=2)],
        suppress_clamped=True, suppress_frame_filling=True, suppress_unsure=True,
    )
    assert [d.box for d in kept.accepted] == [REAL_CONFIDENT_CLOSE_UP, REAL_SMALL_UNSURE]
    assert kept.suppressed == 0


def test_the_uncertainty_ceiling_is_exclusive_and_the_shape_is_not():
    """The boundary, both sides, because a rule that reads two numbers has two of them.

    `conf` at exactly the ceiling is kept - the rule says *below* it, and a detection sitting on the
    boundary is as likely to be a hard item as a phantom - while `area` at exactly the spanning
    threshold is dropped. Pinned because the export's own near-misses live within a thousandth of
    both: the confident band is 0.956 and the largest real box the rule takes is 0.853 of the frame.
    """
    square = (0.0, 0.0, 0.9220, 0.9220)  # area exactly 0.850
    assert round((square[2] - square[0]) * (square[3] - square[1]), 3) == 0.850

    at_ceiling = accept_detections([_det(box=square, conf=0.85)], suppress_unsure=True)
    assert [d.box for d in at_ceiling.accepted] == [square]

    below = accept_detections([_det(box=square, conf=0.8499)], suppress_unsure=True)
    assert below.accepted == []
    assert below.rejected == {UNSURE_REASON: 1}

    # And the empty input stays empty rather than raising on the new predicate.
    empty = accept_detections([], suppress_unsure=True)
    assert empty.accepted == [] and empty.rejected == {} and empty.suppressed == 0


def test_a_box_matching_two_rules_is_counted_once_under_the_earlier_one():
    """A clamped phantom is also frame-spanning and unsure - and must still count as clamped.

    The counts are read beside the item log, so a detection refused twice has to be reported once;
    and the operator's diagnosis differs: `clamped` says the model wanted more than the image,
    `unsure` says it was guessing. The first reason that applies is the one that travels.
    """
    result = accept_detections(
        [_det(box=(0.0, 0.0, 1.0, 1.0), conf=0.55)],
        suppress_clamped=True, suppress_frame_filling=True, suppress_unsure=True,
    )
    assert result.accepted == []
    assert result.rejected == {CLAMPED_REASON: 1}
    assert result.suppressed == 1


def test_the_unsure_rule_is_on_by_default_and_drops_the_band_through_the_pipeline():
    """At the confidence the live capture actually produced, which is the whole point of the rule."""
    assert Settings().suppress_unsure_phantoms is True
    store = _RecordingStore()
    message = _pipeline([BAND_LIVE, ORDINARY], store=store, conf=0.677).process_once()
    assert [d["box"] for d in message["detections"]] == [ORDINARY]
    assert message["stats"]["suppressed"] == 1
    # The band is track 1 and the ordinary box track 2: the store holding only the second is the
    # item log never having seen the phantom.
    assert [row[0] for row in store.rows] == [2]


def test_a_confident_band_still_reaches_the_frame_and_the_log():
    """The same shape above the ceiling is untouched - the rule reads the pair, not the shape."""
    message = _pipeline([BAND_LIVE], conf=0.96).process_once()
    assert len(message["detections"]) == 1
    assert message["stats"]["suppressed"] == 0


def test_turning_the_unsure_rule_off_mid_capture_takes_effect_without_a_restart():
    settings = Settings(preview_mirror=False)
    pipe = _pipeline([BAND_LIVE], settings=settings, conf=0.677)
    assert pipe.process_once()["detections"] == []

    settings.suppress_unsure_phantoms = False

    assert len(pipe.process_once()["detections"]) == 1


def test_the_pipeline_default_is_the_setting_default():
    """`_pipeline` leaves the rule alone, so this is `Settings()` speaking.

    Guards the shape of the flip rather than the value: the pipeline passes the setting straight
    through, so a rule that is on by default cannot be neutered by a pipeline-side default.
    """
    assert Settings().suppress_frame_filling_detections is True
    assert _pipeline([PHANTOM]).process_once()["detections"] == []
