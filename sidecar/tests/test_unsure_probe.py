"""Tests for `unsure_probe.py`: the unsure-phantom rule's price, and whether its numbers hold.

The rule shipped with figures from scratch harnesses that no longer exist, so this file's real
subject is the *tool*: that it applies the app's own rule rather than a copy of it, that its buckets
and claims say what the report says they say, and that the gate cannot be quietly defanged. Nothing
here touches the dataset workspace or a camera - the tool reads both lazily, from inside `main()`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import unsure_probe

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_ROOT = REPO_ROOT / "sidecar"


# ---------------------------------------------------------------------------
# the pure layer: the rule, the buckets, the table
# ---------------------------------------------------------------------------


def _shot(box, conf, cls="thing"):
    return unsure_probe.Shot(box=box, conf=conf, cls=cls)


def _square(x1, y1, x2, y2):
    return (x1, y1, x2, y2)


def test_the_tool_does_not_own_a_second_copy_of_the_rule():
    """`unsure()` is the app's predicate with the ceiling as an argument, not a re-implementation.

    The shipped rule is the one row this table exists to justify. A second definition - a different
    `0.85`, a shape test written from memory - would let the tool print a table, a verdict and "all
    claims hold" about a rule no build applies, and nothing else in the suite would notice: the
    pipeline's own tests only exercise the app's version.
    """
    assert not hasattr(unsure_probe, "is_unsure_phantom")
    source = Path(unsure_probe.__file__).read_text(encoding="utf-8")
    assert "from app.acceptance import" in source
    assert "PHANTOM_CONF_CEILING" in source
    assert unsure_probe.unsure.__defaults__ == (unsure_probe.PHANTOM_CONF_CEILING,)


def test_the_tool_s_rule_agrees_with_the_rule_the_pipeline_actually_applies():
    """The drift guard for the whole file, over both shapes and the ceiling's own boundary."""
    acceptance = pytest.importorskip("app.acceptance")
    boxes = [
        (0.0, 0.0, 1.0, 1.0),                    # spans everything
        (0.0616, 0.0074, 1.0, 0.9583),           # the export's own confident spanning box
        (0.0, 0.5387, 0.9146, 0.9999),           # the live bottom band
        (0.0756, 0.5407, 0.9991, 1.0),           # a wider one
        (0.2, 0.2, 0.4, 0.4),                    # small, ordinary
        (0.0, 0.2, 1.0, 0.4),                    # wide but off both horizontal edges
        (0.0, 0.54, 0.41, 1.0),                  # on the bottom edge, too narrow for the band
        (0.0, 0.54, 0.9, 0.999),                 # a band-shaped box just off the edge tolerance
    ]
    for conf in (0.0, 0.5, 0.8499, 0.85, 0.8501, 1.0):
        for box in boxes:
            assert unsure_probe.unsure(box, conf) == acceptance.is_unsure_phantom(box, conf), (
                box,
                conf,
            )
            # The ceiling is an argument, so the shipped value has to reproduce the app's rule
            # exactly - and the shape half is the rule at a ceiling nothing can be under.
            assert unsure_probe.unsure(box, conf, unsure_probe.PHANTOM_CONF_CEILING) == (
                acceptance.is_unsure_phantom(box, conf)
            )


def test_the_shape_half_is_exactly_what_the_rule_tests_before_confidence():
    """`shaped()` names the two alternatives the app's predicate tests, and nothing else.

    It is the tool's own (the report needs to say *which* shape a box took) and it is the only part
    that is: if it drifts from what `is_unsure_phantom` looks at, the naming in the report would be
    about boxes the rule never saw.
    """
    acceptance = pytest.importorskip("app.acceptance")
    boxes = [
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.5387, 0.9146, 0.9999),
        (0.2, 0.2, 0.4, 0.4),
        (0.0, 0.2, 1.0, 0.4),
    ]
    for box in boxes:
        # Confidence 0 clears every ceiling there is, so the rule is reduced to its shape half -
        # which is what `shaped` and `shape_name` claim to be.
        assert unsure_probe.shaped(box) == acceptance.is_unsure_phantom(box, 0.0)
        assert (unsure_probe.shape_name(box) != unsure_probe.SHAPE_NONE) == unsure_probe.shaped(box)


def test_the_ceiling_table_prices_every_population_on_the_same_axis():
    """One row per candidate ceiling, each scored against all four populations."""
    matched = (_shot((0.0, 0.0, 1.0, 1.0), 0.799), _shot((0.1, 0.1, 0.4, 0.4), 0.55))
    negatives = (_shot((0.0, 0.0, 1.0, 1.0), 0.51), _shot((0.0, 0.54, 0.92, 1.0), 0.79))
    live = (_shot((0.0, 0.54, 0.92, 1.0), 0.6),)
    control = (_shot((0.0, 0.0, 1.0, 1.0), 0.83), _shot((0.0, 0.0, 1.0, 1.0), 0.96))

    rows = unsure_probe.ceiling_table(matched, negatives, live, control, (0.75, 0.85, 0.90))
    by_ceiling = {row.ceiling: row for row in rows}

    assert by_ceiling[0.75].caught_negatives == 1        # only the 0.51 spanning box
    assert by_ceiling[0.85].caught_negatives == 2        # and the 0.79 band
    assert by_ceiling[0.85].caught_live == 1
    assert by_ceiling[0.75].lost_control == 0            # the 0.83 one is a product, kept
    assert by_ceiling[0.85].lost_control == 1
    assert by_ceiling[0.90].lost_control == 1            # the 0.83 one, and no more
    assert not unsure_probe.unsure(control[1].box, control[1].conf, 0.95)  # the 0.96 product
    assert by_ceiling[0.85].dropped_real == 1            # the shaped 0.799; the small box is kept
    assert by_ceiling[0.85].matched_real == 2
    assert by_ceiling[0.85].dropped_max_conf == 0.799
    assert by_ceiling[0.85].kept_min_conf is None        # nothing shaped survives at 0.85
    assert by_ceiling[0.85].clamped_real == 1            # the 1x1 matched detection, on every row
    assert by_ceiling[0.75].dropped_max_conf is None


def test_a_ceiling_that_drops_nothing_is_in_a_gap_and_a_knife_edge_is_not():
    """The claim that the constant is a free choice, and the state that says it is not.

    Read as the data, not as advice: a gap means every ceiling inside it is the same rule over this
    population, so the table cannot tune the constant. A ceiling with a real detection on either side
    of it is exactly the case where moving it buys something and costs something.
    """
    in_gap = unsure_probe.Row(
        ceiling=0.85, caught_negatives=1, negatives=2, caught_live=0, live=0, lost_control=0,
        control=10, dropped_real=0, matched_real=5, clamped_real=1, filling_real=1,
        dropped_max_conf=None, kept_min_conf=0.92,
    )
    knife = unsure_probe.Row(
        ceiling=0.85, caught_negatives=1, negatives=2, caught_live=0, live=0, lost_control=0,
        control=10, dropped_real=1, matched_real=5, clamped_real=1, filling_real=1,
        dropped_max_conf=0.846, kept_min_conf=0.844,
    )
    assert in_gap.in_a_gap and not knife.in_a_gap


def test_the_buckets_partition_what_the_pipeline_kept():
    """Every kept detection is in exactly one bucket, and the refusals are attributed in order.

    The four buckets are the report's table, so a detection landing in two of them (or none) would
    make the columns add up to something that is not the population - which is the failure mode the
    ordered `accept_detections` call exists to prevent in the app itself.
    """
    shots = (
        _shot((0.0, 0.0, 1.0, 1.0), 0.6),           # four edges -> clamp
        _shot((0.0, 0.0, 0.95, 1.0), 0.6),          # three edges -> frame-filling
        _shot((0.03, 0.03, 0.97, 0.97), 0.6),       # spanning, under the ceiling -> unsure
        _shot((0.0, 0.5387, 0.9146, 0.9999), 0.6),  # the band -> unsure
        _shot((0.03, 0.03, 0.97, 0.97), 0.95),      # shaped, confident -> the ceiling's limit
        _shot((0.2, 0.2, 0.4, 0.4), 0.6),           # small and unsure -> the shapes' blind spot
        _shot((0.2, 0.2, 0.4, 0.4), 0.95),          # small and confident -> ordinary
    )
    item = unsure_probe.ledger(
        "hand-built", 7, shots, clamped=True, filling=True, unsure_=True
    )
    assert item.refused == {"clamped": 1, "frame_filling": 1, "unsure": 2}
    assert len(item.blind) == 1 and len(item.residue) == 1 and len(item.plain) == 1
    assert len(item.kept) == 3
    assert len(item.residue) + len(item.blind) + len(item.plain) == len(item.kept)
    assert set(item.refused_shots) == set(shots[:4])


def test_the_pipeline_s_own_survivors_are_what_the_buckets_describe():
    """Where a Pipeline ran, its list wins - because the app's tracker is part of the behaviour.

    `track(persist=True)` carries state between passes, so a rules-off pass and a shipped pass over
    identical frames can differ by a detection. Reporting the decision function's arithmetic when the
    Pipeline kept something else would describe a run nobody had - so the end-to-end numbers are the
    Pipeline's, and the cost table (which is arithmetic over one pass) says so separately.
    """
    shots = (_shot((0.0, 0.0, 1.0, 1.0), 0.9),)
    from_pipeline = (_shot((0.0, 0.0, 1.0, 1.0), 0.9), _shot((0.1, 0.1, 0.2, 0.2), 0.9))
    item = unsure_probe.ledger(
        "walked", 1, shots, clamped=False, filling=False, unsure_=True, pipeline_shots=from_pipeline
    )
    assert len(item.decided) == 1
    assert len(item.kept) == 2
    assert item.pipeline_kept == from_pipeline


def _row(**overrides):
    """The shipped row, built **from the record** rather than typed here.

    The export cost in this fixture is `app/acceptance.py::MEASURED_COST`'s, so a moved measurement
    cannot leave the fixture asserting the old one - which is the failure this whole arrangement
    exists to prevent, and a test that re-typed `6/2018, 36, 252` would be the sixth copy of it.
    """
    base = dict(
        ceiling=unsure_probe.PHANTOM_CONF_CEILING,
        caught_negatives=19, negatives=25, caught_live=3, live=3, lost_control=1, control=60,
        dropped_real=unsure_probe.MEASURED_COST.unsure,
        matched_real=unsure_probe.MEASURED_COST.matched,
        clamped_real=unsure_probe.MEASURED_COST.clamp,
        filling_real=unsure_probe.MEASURED_COST.frame_filling,
        dropped_max_conf=0.839, kept_min_conf=0.856,
    )
    base.update(overrides)
    return unsure_probe.Row(**base)


def _census(**overrides):
    """A Census with only the fields a claim cares about named, and everything else minimal."""
    base = {
        "ceiling": unsure_probe.PHANTOM_CONF_CEILING,
        "rows": (_row(),),
        "negatives": unsure_probe.ledger(
            "negatives", 50, (_shot((0.0, 0.0, 1.0, 1.0), 0.6),),
            clamped=True, filling=True, unsure_=True,
        ),
        "control": unsure_probe.ledger(
            "control", 60, (_shot((0.2, 0.2, 0.4, 0.4), 0.95),),
            clamped=True, filling=True, unsure_=True,
        ),
        "live": unsure_probe.ledger(
            "live", 100, (_shot((0.0, 0.54, 0.92, 1.0), 0.6),),
            clamped=True, filling=True, unsure_=True,
        ),
        "export": unsure_probe.ledger(
            "export", 1, (_shot((0.2, 0.2, 0.4, 0.4), 0.95),),
            clamped=True, filling=True, unsure_=True,
        ),
        "per_split": (("train", 10, 10, 10),),
        "predictions": 10,
        "shipped_flags": (True, True, True),
        "conf": unsure_probe.MEASURED_COST.conf,
        "live_scene": unsure_probe.Scene(luminance=(120.0, 121.0, 119.0)),
        "require_live_band": False,
    }
    base.update(overrides)
    return unsure_probe.Census(**base)


def test_the_claims_separate_a_working_rule_from_one_that_did_nothing():
    """`--strict` gates on sentences that must be true of the shipped rule, not on a table.

    A table alone would let a ceiling that had begun eating real detections print a middle column
    nobody reads, and the empty-counter check alone would pass on a window where the model happened
    to produce nothing. So the failure cases are exercised here: a ceiling no longer in a gap, a rule
    that catches nothing, and a rule that is off in the profile being measured.
    """
    good = unsure_probe.checks(_census())
    assert all(check.ok for check in good), unsure_probe.verdict(good)

    # A ceiling that has moved onto a real detection's confidence is a knife edge, not a free choice.
    knife = list(_census().rows)
    knife[0] = unsure_probe.Row(
        ceiling=0.85, caught_negatives=19, negatives=25, caught_live=3, live=3, lost_control=1,
        control=60, dropped_real=6, matched_real=2018, clamped_real=36, filling_real=252,
        dropped_max_conf=0.86, kept_min_conf=0.85,
    )
    assert not unsure_probe.checks(_census(rows=tuple(knife)))[0].ok

    # A rule that catches nothing on either source is the "did nothing" case.
    silent = _census(
        rows=(
            unsure_probe.Row(
                ceiling=0.85, caught_negatives=0, negatives=25, caught_live=0, live=0,
                lost_control=0, control=60, dropped_real=0, matched_real=2018, clamped_real=36,
                filling_real=252, dropped_max_conf=None, kept_min_conf=None,
            ),
        )
    )
    failed = [check.claim for check in unsure_probe.checks(silent) if not check.ok]
    assert "it catches the stored empty counter's own family" in failed

    # A settled install that has the rule switched off is not a measurement of the shipped default.
    off = _census(shipped_flags=(True, True, False))
    assert "the rule is on in the profile this measured" in [
        check.claim for check in unsure_probe.checks(off) if not check.ok
    ]


def _empty_live_census(**overrides):
    """The run this machine actually produced: a live window with nothing in it at all."""
    base = {
        "live": unsure_probe.ledger("live", 3, (), clamped=True, filling=True, unsure_=True),
        "live_scene": unsure_probe.Scene(luminance=(14.0, 13.0, 15.0)),
    }
    base.update(overrides)
    return _census(**base)


def _claim(census, needle):
    return next(check for check in unsure_probe.checks(census) if needle in check.claim)


def test_a_quiet_counter_passes_the_escape_claim_and_says_what_the_scene_looked_like():
    """The contract this gate needed: silence is not failure, and it is not evidence either.

    A window that produced no detection has nothing that could have escaped, so the escape claim
    holds - and the reading carries the frames, the detections and the brightness, which is what
    separates a quiet counter from a covered lens.
    """
    census = _empty_live_census()
    escape = _claim(census, "escaped")
    band = _claim(census, "exercised the band family")
    assert escape.ok and band.ok
    assert "no detection at all" in escape.reading and "luminance 14/255" in escape.reading
    assert "0 band-shaped" in band.reading and "not set" in band.reading


def test_the_band_claim_fails_when_the_run_has_to_prove_it():
    """`--require-live-band` is the flag for a run that cannot accept "nothing was there to test".

    Without it, a dark or empty scene passes the family claim and the report says the family was not
    exercised. With it, the same window is a failure naming the same numbers - so a scene that
    offers no band is a decision someone made rather than a silent pass.
    """
    demanded = _empty_live_census(require_live_band=True)
    claim = _claim(demanded, "exercised the band family")
    assert not claim.ok
    assert "is set, so this run has to prove it" in claim.reading
    assert unsure_probe.strict_failed(unsure_probe.checks(demanded))
    # And the escape claim is unaffected by the flag: it is about what was seen, not what was hoped.
    assert _claim(demanded, "escaped").ok


def test_a_band_that_escapes_the_rule_fails_even_on_a_quiet_default_run():
    """The failure the gate exists for: a shaped detection under the ceiling that reached the frame.

    Whatever the scene, whatever the flags, a detection the rule was built to refuse showing up in
    what the Pipeline kept means the rule did not act on it - the profile has it off where the
    settings say otherwise, the predicate moved, or the decision never saw it. It is priced as an
    escape rather than folded into the residue count because it is the one live failure this tool
    can catch on a scene nobody is standing in front of.
    """
    escaping = _shot((0.0, 0.54, 0.92, 1.0), 0.6)  # a band, under the ceiling
    walk = unsure_probe.ledger(
        "live", 3, (escaping,), clamped=True, filling=True, unsure_=False,
        pipeline_shots=(escaping,),
    )
    census = _empty_live_census(live=walk)
    escape = _claim(census, "escaped")
    assert not escape.ok
    assert "1 escaped" in escape.reading and "1 band-shaped" in escape.reading
    assert unsure_probe.strict_failed(unsure_probe.checks(census))


def test_a_ceiling_off_the_ladder_fails_loudly_rather_than_printing_a_verdict():
    """Without its own row there is nothing for the report to stand behind."""
    assert any(
        not check.ok and "on the ladder" in check.claim
        for check in unsure_probe.checks(_census(ceiling=0.87))
    )


def test_the_verdict_names_the_failures_instead_of_only_counting_them():
    checks_ = unsure_probe.checks(_census(shipped_flags=(True, True, False)))
    assert unsure_probe.strict_failed(checks_)
    text = unsure_probe.verdict(checks_)
    assert "1 of 9 claims FAIL" in text and "the rule is on" in text
    assert unsure_probe.verdict(unsure_probe.checks(_census())) == "all 9 claims hold"


def _record_claim(census) -> unsure_probe.Check:
    """The claim about the record, found by what it says rather than by where it sits in the tuple."""
    return next(
        check for check in unsure_probe.checks(census)
        if "app/acceptance.py records" in check.claim
    )


def test_the_record_claim_passes_when_the_cost_is_the_one_the_app_records():
    """A run at the app's own operating point measures the record, or this claim fires."""
    check = _record_claim(_census())
    assert check.ok
    assert "matches app/acceptance.py::MEASURED_COST" in check.reading


def test_a_moved_cost_fails_the_record_claim_and_names_the_edit():
    """The half of the one-owner arrangement a bare checkout cannot run.

    `tests/test_cost_figures.py` checks five prose copies against `MEASURED_COST` on every commit,
    and cannot re-measure: it has no weights, no export and no GPU. So if a fresh measurement moved
    the numbers and nothing said so, the copies would go on describing an older model - which is
    exactly how a desktop hint came to advertise a denominator its own tool had already replaced.
    The failure has to name the record, because the remedy is an edit to a file this run never
    touched.
    """
    census = _census(rows=(_row(dropped_real=unsure_probe.MEASURED_COST.unsure + 1),))
    check = _record_claim(census)
    assert not check.ok
    assert "MOVED" in check.reading
    assert "MEASURED_COST" in check.reading and "tests/test_cost_figures.py" in check.reading
    assert unsure_probe.strict_failed(unsure_probe.checks(census))


def test_another_operating_point_says_it_cannot_compare_rather_than_that_the_cost_moved():
    """`--conf` is a knob, so a run at another threshold is a different question, not a mover.

    A phantom is a low-confidence detection, so the counts scale with the threshold: comparing a
    0.7 run against a record quoted at 0.5 would report a stale record every time someone measured
    a variant. The claim says which it is instead, and the reading is the only place that says
    *why* - the alternative is a gate that cries wolf on its own knob.
    """
    census = _census(conf=0.7, rows=(_row(dropped_real=unsure_probe.MEASURED_COST.unsure + 4),))
    check = _record_claim(census)
    assert check.ok
    assert "not comparable" in check.reading and "conf>=0.7" in check.reading


def test_the_catch_cost_says_the_ceiling_and_the_alternative_for_each_shape():
    """What it would take to catch a survivor, in the terms the table is priced in."""
    rows = unsure_probe.ceiling_table((), (), (), (), (0.80, 0.85, 0.90))
    band = _shot((0.0, 0.54, 0.92, 1.0), 0.85)
    residue = _shot((0.2, 0.2, 0.4, 0.4), 0.6)
    assert "ceiling 0.90" in unsure_probe.catch_cost(rows, band)
    assert "no ceiling catches this" in unsure_probe.catch_cost(rows, residue)
    # 0.90 is the top of the ladder, and a ceiling only catches what is strictly under it.
    assert "no ceiling on the ladder reaches it" in unsure_probe.catch_cost(rows, _shot(band.box, 0.90))


def _array(value: float):
    """A one-pixel stand-in for a frame, which is all `Scene` reads: the mean pixel value."""

    class Fake:
        def mean(self):
            return value

    return Fake()


def test_the_live_window_stops_at_its_deadline_and_counts_only_detections():
    """The window is a duration, not a frame count, and a frame with nothing on it adds nothing."""

    class StubPipeline:
        def __init__(self):
            self.calls = 0

        def process_once(self):
            self.calls += 1
            if self.calls == 2:
                return None
            return {
                "detections": [
                    {"cls": "thing", "conf": 0.7, "box": (0.0, 0.0, 1.0, 1.0), "track_id": None}
                ],
                "stats": {},
            }

    pipeline = StubPipeline()
    class StubSource:
        def __init__(self):
            self.frames = []

        def latest(self):
            return (len(self.frames), self.frames[-1] if self.frames else None)

    class StubRecorder:
        seen = ()

    source = StubSource()
    source.frames.append(_array(14.0))
    ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 3.0])
    model, kept, scene = unsure_probe.live_window(
        pipeline, source, StubRecorder(), 2.0, clock=lambda: next(ticks), pause=lambda _s: None
    )
    # Two calls in two seconds, one of which returned no frame at all - so one frame is counted and
    # the loop stops when the clock reaches the deadline rather than when frames run out.
    assert pipeline.calls == 2
    assert len(kept) == 1 and kept[0].box == (0.0, 0.0, 1.0, 1.0)
    assert model == ()  # what the model produced comes from the recorder, not from the message
    # And the frame that was shown is what the scene records, so a silent window can be read as dark.
    assert scene.frames == 1 and scene.mean == pytest.approx(14.0)
    assert "luminance 14/255" in scene.describe()


def test_the_escape_and_the_band_claim_both_hold_through_the_real_pipeline():
    """The two live contracts, exercised through the real `Pipeline` on identical frames.

    One frame holding a band-shaped detection that the rule was built to refuse. With the rule off it
    reaches the frame and the escape claim fails; with the rule on it is refused, the claim holds,
    and the band family is still recorded as *seen* - which is what lets a run insist on it.
    """
    pytest.importorskip("numpy")
    from app.pipeline import Pipeline
    from app.settings import Settings
    from tests.test_pipeline import _StubSource

    schemas = pytest.importorskip("app.schemas")

    class BandDetector:
        names = {0: "thing"}

        def infer(self, frame):
            return [
                schemas.Detection(
                    track_id=None, cls="thing", conf=0.6, box=(0.0, 0.54, 0.92, 1.0)
                )
            ]

    def run(rule_on: bool):
        settings = Settings()
        settings.suppress_clamped_detections = False
        settings.suppress_frame_filling_detections = False
        settings.suppress_unsure_phantoms = rule_on
        recorder = unsure_probe.Recorded(BandDetector())
        source = _StubSource()
        pipeline = Pipeline(source, recorder, settings, on_message=lambda _m: None)
        ticks = iter([0.0, 0.0, 0.5, 1.0])
        model, kept, scene = unsure_probe.live_window(
            pipeline, source, recorder, 1.0, clock=lambda: next(ticks), pause=lambda _s: None
        )
        return unsure_probe.ledger(
            "live", scene.frames, model,
            clamped=False, filling=False, unsure_=rule_on, pipeline_shots=kept,
        )

    escaped = run(rule_on=False)
    # Every frame's detection reached the frame, because nothing refused it - and the boxes come
    # back mirrored, which is the frame message's convention (`preview_mirror`) and does not touch
    # any input the rule reads: area, band bounds, pinned sides and confidence are all mirror-safe.
    assert len(escaped.kept) == escaped.frames == len(escaped.raw) > 0
    census = _empty_live_census(live=escaped)
    assert not _claim(census, "escaped").ok
    assert unsure_probe.strict_failed(unsure_probe.checks(census))

    caught = run(rule_on=True)
    assert caught.kept == ()
    assert unsure_probe.live_bands(_empty_live_census(live=caught)) == len(caught.raw) > 0  # seen
    demanded = _empty_live_census(live=caught, require_live_band=True)
    assert all(check.ok for check in unsure_probe.checks(demanded))


def test_the_recorder_takes_the_model_s_output_at_the_seam_the_pipeline_reads():
    """Both sides of the live decision from one pass, because a live scene cannot be re-run.

    The proxy has to be transparent - the Pipeline reads `.infer()` and `getattr(detector, "names")` -
    and it has to record *before* any rule sees the detections, which is what makes the escape claim
    about the model rather than about the decision it is already scored by.
    """

    class Model:
        names = {0: "thing"}
        conf = 0.5

        def infer(self, frame):
            return [pytest.importorskip("app.schemas").Detection(
                track_id=None, cls="thing", conf=0.6, box=(0.0, 0.0, 1.0, 1.0)
            )]

    recorder = unsure_probe.Recorded(Model())
    assert recorder.names == {0: "thing"} and recorder.conf == 0.5  # forwarded, not shadowed
    out = recorder.infer(None)
    assert len(recorder.seen) == 1
    assert recorder.seen[0] == unsure_probe.Shot(box=(0.0, 0.0, 1.0, 1.0), conf=0.6, cls="thing")
    assert out[0].cls == "thing"  # and the Pipeline gets the real detections back


def test_the_report_prints_the_numbers_the_claims_read():
    """Every claim's reading is a number the report printed, from one Census."""
    lines = unsure_probe.report_lines(
        _census(),
        generation=pytest.importorskip("generations").V1,
        weights="models/scanncart-grocery-v1.pt",
        device="0",
        conf=0.5,
        shipped_conf=0.5,
        imgsz=640,
        live_seconds=30.0,
        checks_=unsure_probe.checks(_census()),
    )
    text = "\n".join(lines)
    assert "conf < 0.85" in text and "area >= 0.85" in text
    assert "the three rules on the same matched population: clamp 36, frame-filling 252, unsure 6" in text
    assert "(0.839, 0.856)" in text
    assert "matched one-to-one: 10 of 10 predictions" in text
    assert "all 9 claims hold" in text
    # The live window's own block: what it yielded, then the two readings the claims are made from.
    assert "the live window, and what it actually yielded" in text
    assert "mean luminance 120/255" in text


def test_the_tool_does_not_import_a_gpu_library_at_module_level():
    """Importing the tool must stay cheap: the tests import it, and `main()` owns the heavy imports."""
    source = Path(unsure_probe.__file__).read_text(encoding="utf-8")
    for module in ("cv2", "torch", "ultralytics"):
        assert not re.search(rf"^(from|import)\s+{module}", source, re.MULTILINE), module
    assert re.search(r"^\s+import cv2", source, re.MULTILINE)
    assert re.search(r"^\s+from app\.acceptance import", source, re.MULTILINE)
    # And it puts the sidecar root on the path before importing the app, which is what makes
    # `python tools/unsure_probe.py` work outside pytest.
    assert re.search(r"^sys\.path\.[a-z]+\(str\(SIDECAR_ROOT\)\)", source, re.MULTILINE)
    assert unsure_probe.SETTINGS_PATH.is_absolute()


def test_the_two_settings_that_would_change_the_answer_are_forced_in_main():
    """A frame skip, an allowlist or a sibling suppression would each hide what this tool counts."""
    source = Path(unsure_probe.__file__).read_text(encoding="utf-8")
    main_body = source[source.index("def main(") :]
    assert "settings.infer_frame_skip = 0" in main_body
    assert "settings.class_allowlist = []" in main_body
    assert "settings.suppress_clamped_detections = False" in main_body
    assert "settings.suppress_frame_filling_detections = False" in main_body
    # And the populations are walked through the app's own pipeline, with its own settings object.
    assert "Pipeline(source, detector, settings" in main_body


def test_the_population_s_sampling_rules_are_pinned():
    """The control is the clamp probe's own sample; a cost quoted against another is not comparable."""
    import clamp_probe

    assert unsure_probe.CONTROL_SPLIT == clamp_probe.REAL_SPLIT
    assert unsure_probe.CONTROL_SAMPLE == clamp_probe.REAL_SAMPLE
    assert unsure_probe.NEGATIVES_DIR == clamp_probe.NEGATIVES_DIR
    assert unsure_probe.NEGATIVES_DIR.is_absolute()
    assert unsure_probe.LIVE_SECONDS > 0


def test_the_measured_price_on_the_control_is_pinned_and_documented():
    """`CONTROL_COST_CEILING` is a measurement, not a target: the claim fails when it moves."""
    assert unsure_probe.CONTROL_COST_CEILING == 1
    source = Path(unsure_probe.__file__).read_text(encoding="utf-8")
    assert "Measured, not chosen" in source


def _recipe(text: str, target: str) -> str:
    """One target's recipe lines. Make indents them with a tab, and the next target is untabbed."""
    start = text.index(f"\n{target}:") + 1
    lines = []
    for line in text[start:].splitlines()[1:]:
        if not line.startswith("\t"):
            break
        lines.append(line)
    return "\n".join(lines)


def test_the_make_gate_keeps_its_strict_flag_and_stays_out_of_the_fast_suite():
    """`verify-unsure` is a gate, and a gate whose flag goes missing is a report nobody reads.

    Two ways it could stop gating without anyone noticing. **`--strict` dropped** leaves a target that
    prints a table and PASS/FAIL lines and exits 0 either way. **Added to `test`'s dependencies** puts
    a 6.6 GB workspace, an installed weight and a camera on CI's bare checkout.
    """
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(text, "verify-unsure")
    assert "unsure_probe.py" in recipe and "--strict" in recipe
    assert "--live-seconds" in recipe and "--generation" in recipe
    # The band contract is reachable from the target, and off unless someone asks for it: the
    # default run certifies that nothing escaped, which a quiet counter can honestly pass.
    assert "$(UNSURE_BAND_FLAG)" in recipe
    assert "UNSURE_REQUIRE_BAND ?= 0" in text and "--require-live-band" in text
    assert "verify-unsure" in text[text.index(".PHONY") : text.index("help:")]
    test_line = next(line for line in text.splitlines() if line.startswith("test:"))
    assert "verify-unsure" not in test_line.split(":", 1)[1].split()


def test_the_documented_gate_command_is_the_one_that_exists():
    """CLAUDE.md and README name the target, so the name has to be real in the Makefile."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "make verify-unsure" in claude and "make verify-unsure" in readme
    assert "\nverify-unsure:" in makefile
    assert '"  verify-unsure' in makefile


def test_the_owner_and_the_tool_point_at_each_other():
    """The rule's justification and its reproducer have to be one hop apart, in both directions.

    The docstring is where a reader looks for the cost, and the tool is where they re-derive it -
    the same pairing `clamp_probe.py` has with the clamp rule's claims.
    """
    acceptance = (SIDECAR_ROOT / "app" / "acceptance.py").read_text(encoding="utf-8")
    tool = Path(unsure_probe.__file__).read_text(encoding="utf-8")
    assert "tools/unsure_probe.py" in acceptance
    assert "app/acceptance.py" in tool
    assert re.search(r"6 of the 2018|6 of 2018", acceptance)
