"""Tests for `clamp_probe.py`: the frame-clamp rule, and whether the suppression earns its place.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pytest
import audit_recall
import clamp_probe
import generations
import spec_check
import workspace

# Rebuilt from the modules that own each fact, not copied by value.
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = REPO_ROOT / "docs" / "CAPTURE_CHECKLIST.md"

from tests.dataset_tool_helpers import _fake_generation


# ---------------------------------------------------------------------------
# clamp_probe: the frame-clamp rule, and whether the suppression earns its place
# ---------------------------------------------------------------------------


def _prov(frames=50, fired=0, detections=0, suppressed=0, boxes=(), classes=()) -> clamp_probe.Pass:
    """A `Pass` with only the fields a test cares about named."""
    return clamp_probe.Pass(
        frames=frames, fired=fired, detections=detections, suppressed=suppressed,
        boxes=boxes, classes=classes,
    )


def _rules(phantoms, real, labels, tolerances=(0.002, 0.01, 0.02)):
    return clamp_probe.rule_table(tuple(phantoms), tuple(real), tuple(labels), tolerances)


def test_the_worst_edge_measure_reads_a_clamped_box_as_zero_and_a_centred_one_as_a_half():
    """The one number the whole table is a sweep of, checked at its two extremes.

    0 means every side is pinned - the shape a prediction takes when the model wanted something
    larger than the canvas. A centred object is a half on all four sides. Anything in between is
    a box partly inside the frame, which is what a genuine close-up looks like.
    """
    assert clamp_probe.worst_edge((0.0, 0.0, 1.0, 1.0)) == 0.0
    assert clamp_probe.worst_edge((0.25, 0.25, 0.75, 0.75)) == 0.25
    # The exact shape a live phantom had: three sides pinned, one a ten-thousandth in. Still
    # reads as ~0 rather than as a partial box, which is why the tolerance around it is 0.01.
    assert clamp_probe.worst_edge((0.0, 0.0001, 0.9995, 1.0)) < 0.001


def test_the_worst_edge_is_the_worst_side_so_a_partial_box_does_not_read_as_clamped():
    """Why it is a max and not a min or a count of pinned sides.

    A real close-up overflows the frame on the sides the object leaves through, so two or three of
    its sides are pinned and a *count* would call it a phantom. Taking the worst side is what
    separates them: the box below is flush against the left and top and runs off the right, and it
    still reads as 0.4 because the bottom edge is well inside the frame.
    """
    partial = (0.0, 0.0, 1.4, 0.6)
    assert clamp_probe.worst_edge(partial) == pytest.approx(0.4)
    # Two of its four sides are flush against the frame; the measure still refuses it, because the
    # bottom edge is 0.4 away - which is the distinction a count of pinned sides would lose.
    assert clamp_probe.worst_edge(partial) > 0.01


def test_the_tool_does_not_own_a_second_copy_of_the_tolerance():
    """It reads `CLAMPED_EDGE_TOLERANCE` from the pipeline inside `main()` rather than defining it.

    A second `0.01` here would let this tool go on producing a table and a verdict justifying a
    number the app no longer applies - the constant and the tool are edited at different times by
    different reasons, and a duplicated literal is the one way they can disagree silently.
    """
    assert not hasattr(clamp_probe, "CLAMPED_EDGE_TOLERANCE")
    source = Path(clamp_probe.__file__).read_text(encoding="utf-8")
    assert re.search(
        r"^\s+from app\.pipeline import CLAMPED_EDGE_TOLERANCE", source, re.MULTILINE
    )
    # And it ends up in the report rather than being read and dropped.
    assert "tolerance {tolerance}" in source


def test_the_tool_s_measure_agrees_with_the_rule_the_pipeline_actually_applies():
    """The drift guard for the whole file, and the reason this tool is allowed to exist.

    `worst_edge(box) <= t` and `is_clamped_to_frame(box, t)` must be the same predicate - the first
    is the sweep's axis, the second is what ships. If they ever diverge, this tool would go on
    producing a table, a verdict and a "all claims hold" line justifying a rule no build applies,
    and nothing else in the suite would notice, because the pipeline's own tests only exercise the
    pipeline's version.
    """
    pytest.importorskip("numpy")
    from app.pipeline import is_clamped_to_frame

    boxes = [
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0001, 0.9995, 1.0),
        (0.0, 0.0, 1.4, 0.6),          # a partial box: pinned on two sides, not four
        (0.25, 0.25, 0.75, 0.75),
        (0.0099, 0.0099, 0.9901, 0.9901),
        (0.0101, 0.0, 0.999, 1.0),
        (-0.3, -0.3, 1.3, 1.3),
    ]
    for tolerance in (0.002, 0.01, 0.02, 0.5):
        for box in boxes:
            # Parenthesised: `a <= b == c` chains as `(a <= b) and (b == c)`, which compares a
            # tolerance against a bool and passes for the wrong reason.
            assert (clamp_probe.worst_edge(box) <= tolerance) == is_clamped_to_frame(
                box, tolerance
            ), box


def test_the_ladder_always_contains_the_tolerance_that_is_actually_running():
    """A table that omitted the shipped row would justify a rule nothing runs.

    The constant and this file are edited by different people at different times, so the shipped
    value is folded in rather than assumed to be one of the candidates. The report marks it with
    `<-- shipped`, and if it were missing there would be nothing to mark.
    """
    ladder = clamp_probe.sweep_tolerances(0.0075)
    assert 0.0075 in ladder
    assert clamp_probe.sweep_tolerances(0.01) == clamp_probe.sweep_tolerances(0.01)
    # Already a candidate: folded in, not duplicated, and still sorted.
    assert list(clamp_probe.sweep_tolerances(0.01)) == sorted(clamp_probe.sweep_tolerances(0.01))
    assert clamp_probe.sweep_tolerances(0.01).count(0.01) == 1


def test_a_rule_row_prices_a_candidate_on_all_three_populations():
    """Caught phantoms, real detections lost, and training labels rejected - in one row.

    The three are different costs and the middle one is the one that decides: a tolerance that
    caught every phantom by discarding real items would look like a total success in the first
    column alone.
    """
    rows = _rules(
        phantoms=[(0.0, 0.0, 1.0, 1.0), (0.03, 0.03, 0.97, 0.97)],
        real=[(0.05, 0.05, 0.95, 0.95)],
        labels=[(0.0, 0.0, 1.0, 1.0), (0.2, 0.2, 0.8, 0.8)],
        tolerances=(0.01, 0.2),
    )
    tight = clamp_probe.row_for(rows, 0.01)
    assert (tight.caught, tight.phantoms) == (1, 2)
    assert (tight.real_lost, tight.real) == (0, 1)
    assert (tight.labels_rejected, tight.labels) == (1, 2)
    assert tight.rejected_share == pytest.approx(0.5)

    loose = clamp_probe.row_for(rows, 0.2)
    assert (loose.caught, loose.real_lost) == (2, 1)  # a wider rule prices a real item in
    assert loose.rejected_share == pytest.approx(1.0)


def test_free_means_it_catches_something_and_costs_no_measured_real_detection():
    """"0 real lost" is not a licence on its own - a rule that catches nothing is also free.

    Without the `caught > 0` half, the widest tolerance in any table that happened to lose no real
    detection would be reported as the recommendation, and "loosen the rule until it does nothing"
    would read as the optimum.
    """
    assert clamp_probe.RuleRow(0.01, caught=5, phantoms=5, real_lost=0, real=9, labels_rejected=0, labels=9).is_free
    assert not clamp_probe.RuleRow(0.01, 0, 5, 0, 9, 0, 9).is_free
    assert not clamp_probe.RuleRow(0.01, 5, 5, 1, 9, 0, 9).is_free


def test_the_loosest_free_row_is_the_bound_on_the_evidence_not_a_recommendation():
    """How far the rule could go and still be free - and it is not the shipped tolerance.

    The point of reporting it is that the shipped constant has margin at the operating point
    measured, so the table cannot be read as "0.01 was nearly too tight". What it is not is advice:
    moving the constant to the loosest free row would spend the entire margin on the handful of
    phantoms the next row buys.
    """
    rows = _rules(
        phantoms=[(0.0, 0.0, 1.0, 1.0)],
        real=[(0.04, 0.04, 0.96, 0.96)],
        labels=[],
    )
    assert clamp_probe.loosest_free_row(rows).tolerance == 0.02
    assert clamp_probe.next_looser_row(rows, 0.01).tolerance == 0.02
    assert clamp_probe.next_looser_row(rows, 0.02) is None
    # Nothing is free when every candidate is priced by a real detection, so there is no bound.
    priced = _rules(phantoms=[(0.0, 0.0, 1.0, 1.0)], real=[(0.0, 0.0, 1.0, 1.0)], labels=[])
    assert clamp_probe.loosest_free_row(priced) is None


def test_the_distribution_line_handles_a_population_that_produced_nothing():
    """An empty population is stated, not rendered as `min 0.0000` off an empty list.

    That would print a distribution for a set with no members and read as "every detection was
    perfectly clamped" - the opposite of "this pass found nothing to measure".
    """
    assert clamp_probe.distribution([]) == "no detections"
    line = clamp_probe.distribution([0.3, 0.1, 0.2])
    assert "min 0.1000" in line and "median 0.2000" in line and "max 0.3000" in line


def test_the_claims_separate_a_working_filter_from_one_that_did_nothing():
    """The two readings that make the end-to-end block worth taking.

    A suppression that never fires and a suppression that works both leave the item log clean, and
    only the off/on comparison tells them apart. `on.suppressed == 0` and
    `on.fired == off.fired` are the same numbers a broken flip produces, so both are claims rather
    than rows to eyeball.
    """
    rows = _rules(phantoms=[(0.0, 0.0, 1.0, 1.0)], real=[(0.4, 0.4, 0.6, 0.6)], labels=[])
    shipped = clamp_probe.row_for(rows, 0.01)

    off = _prov(fired=25, detections=25)
    on = _prov(fired=6, detections=6, suppressed=19)
    real_off = _prov(frames=60, fired=60, detections=60)
    real_on = _prov(frames=60, fired=60, detections=60)
    assert not clamp_probe.strict_failed(clamp_probe.checks(off, on, real_off, real_on, shipped))

    # The setting never took effect: the on pass reproduces the off pass exactly.
    inert = clamp_probe.checks(off, _prov(fired=25, detections=25), real_off, real_on, shipped)
    assert clamp_probe.strict_failed(inert)
    assert [c.claim for c in inert if not c.ok] == [
        "the Pipeline drops detections for it",
        "fewer empty-counter frames show a phantom with it on",
    ]

    # The filter works and eats a real item, which is the failure the middle column exists for.
    lossy = clamp_probe.checks(off, on, real_off, _prov(frames=60, fired=59, detections=59), shipped)
    assert [c.claim for c in lossy if not c.ok] == ["the Pipeline loses no real product detection"]


def test_a_tolerance_off_the_ladder_fails_loudly_rather_than_printing_a_verdict():
    """The one case where the report has no standing to make any claim at all.

    Every other claim is about numbers; this one is about whether the tool can see the rule it is
    supposed to be checking. If the ladder missed the shipped value there is nothing to justify, so
    it fails rather than passing the claims it can still compute.
    """
    result = clamp_probe.checks(_prov(), _prov(), _prov(), _prov(), None)
    assert len(result) == 1 and not result[0].ok
    assert "ladder" in result[0].claim
    assert clamp_probe.strict_failed(result)


def test_the_verdict_names_the_failures_instead_of_only_counting_them():
    """"2 of 5 failed" alone sends the reader back through the block to find out which."""
    good = clamp_probe.Check(True, "a claim that holds", "1 of 1")
    bad = clamp_probe.Check(False, "a claim that does not", "0 of 1")
    assert clamp_probe.verdict((good, good)) == "all 2 claims hold"
    assert clamp_probe.verdict((good, bad)) == "1 of 2 claims FAIL: a claim that does not"
    assert not clamp_probe.strict_failed((good,))
    assert clamp_probe.strict_failed((good, bad))


def test_the_phantom_breakdown_names_only_the_classes_that_recur():
    """One stray detection is not a pattern, and the actionable half of the finding is *which*
    products - so a class seen once is dropped rather than listed."""
    off = _prov(classes=(("Milo", 16), ("Bear Brand", 6), ("tuna", 3), ("sardines", 1)))
    assert clamp_probe.phantom_classes(off) == [("Milo", 16), ("Bear Brand", 6), ("tuna", 3)]
    assert clamp_probe.phantom_classes(_prov(classes=(("sardines", 2),))) == []


def test_the_fixed_source_hands_back_what_was_put_in_and_never_reports_failure():
    """The source contract `Pipeline` relies on: `(seq, frame)` or None, and a `failure` it reads
    every frame. A source that reported failure would end the sweep early with a traceback instead
    of a reading, and one that kept returning the last frame would silently multiply it."""
    source = clamp_probe.Fixed()
    assert source.failure is None
    first, second = object(), object()
    source.set(first)
    seq_1, got_1 = source.latest()
    source.set(second)
    seq_2, got_2 = source.latest()
    assert (got_1, got_2) == (first, second)
    assert seq_2 > seq_1  # the seq a frame message carries, so two frames cannot read as one
    # Re-read without a new frame: the same frame, at the same sequence number.
    assert source.latest() == (seq_2, second)


def test_a_clean_frame_count_is_what_the_operator_reads_as_an_empty_counter():
    """`frames - fired` rather than a second counter, so the two cannot disagree."""
    pass_ = _prov(frames=50, fired=25, detections=25)
    assert pass_.clean == 25
    assert _prov(frames=50).clean == 50


def test_the_report_says_so_when_the_profile_runs_a_different_threshold():
    """The finding that made the operating point a line in the report.

    A phantom is a *low-confidence* detection, so `conf_threshold` is the one setting the counted
    populations are not stable across: this machine's profile runs 0.7 while the published table
    was measured at the shipped 0.5, and the same weights produce 25 phantoms at one and 15 at the
    other. The rule holds at both, but a number quoted without its threshold is not reproducible,
    so a divergence is stated rather than left to be inferred.
    """
    rows = _rules(phantoms=[(0.0, 0.0, 1.0, 1.0)], real=[(0.4, 0.4, 0.6, 0.6)], labels=[])
    kwargs = dict(
        generation=generations.V1, weights="models/x.pt", split="train", device="cpu",
        tolerance=0.01, imgsz=640, rows=rows, off=_prov(fired=25, detections=25),
        on=_prov(fired=6, detections=6, suppressed=19), real_off=_prov(frames=60, detections=60),
        real_on=_prov(frames=60, detections=60), labels=0,
    )
    same = "\n".join(clamp_probe.report_lines(**kwargs, conf=0.5, shipped_conf=0.5, checks_=()))
    assert "conf>=0.5" in same and "this profile runs" not in same

    differ = "\n".join(clamp_probe.report_lines(**kwargs, conf=0.7, shipped_conf=0.5, checks_=()))
    assert "conf>=0.7" in differ
    assert "Settings() ships 0.5" in differ


def test_every_number_the_report_prints_is_a_pipeline_reading():
    """The structural decision the module docstring makes, pinned as a fact about the source.

    A separate `detector.infer` pass for the distributions would cost the same inferences while
    letting the table and the end-to-end block describe different frames - the table could be a
    25-phantom population and the block a 15-phantom one, with nothing on screen to show it. The
    docstring claims the off pass is both the control and the source; this checks that nothing
    reaches for the detector directly.
    """
    source = Path(clamp_probe.__file__).read_text(encoding="utf-8")
    assert "detector.infer" not in source
    assert "Pipeline.process_once" in clamp_probe.__doc__
    # The one reading that is not a detection at all, and is named as such.
    assert re.search(r"^\s+return tuple\(out\)", source, re.MULTILINE)


def test_clamp_probe_does_not_import_a_gpu_library_at_module_level():
    """Same shape as the other tools: the rule and the table are testable with no torch, no cv2.

    The sweep needs both, so they are imported inside `main()` - which is also what lets this file
    import the tool at all.
    """
    source = Path(clamp_probe.__file__).read_text(encoding="utf-8")
    for module in ("cv2", "torch", "ultralytics"):
        assert not re.search(rf"^(from|import)\s+{module}", source, re.MULTILINE), module
    assert re.search(r"^\s+import cv2", source, re.MULTILINE)
    assert re.search(r"^\s+from app\.pipeline import", source, re.MULTILINE)


def test_clamp_probe_put_the_sidecar_root_on_the_path_and_reads_the_app_s_settings():
    """Both failures it shares with spec_check: a bare ModuleNotFoundError by hand, and a
    relative settings path resolving to two different profiles depending on who started it."""
    source = Path(clamp_probe.__file__).read_text(encoding="utf-8")
    assert re.search(r"^sys\.path\.[a-z]+\(str\(SIDECAR_ROOT\)\)", source, re.MULTILINE)
    assert clamp_probe.SETTINGS_PATH.is_absolute()
    assert clamp_probe.SETTINGS_PATH.name == "settings.json"
    assert clamp_probe.SETTINGS_PATH.parent.name == "data"


def test_the_two_settings_that_would_change_the_answer_are_forced_in_main():
    """A profile with a frame skip would report the fraction of frames this looked at; an
    allowlist would shrink the real population by class and turn a phantom of an excluded one into
    a non-observation. Both are the difference between measuring the app and measuring a config."""
    source = Path(clamp_probe.__file__).read_text(encoding="utf-8")
    assert "settings.infer_frame_skip = 0" in source
    assert "settings.class_allowlist = []" in source
    main_body = source[source.index("def main(") :]
    assert "infer_frame_skip = 0" in main_body and "class_allowlist = []" in main_body


def test_labelled_frames_keeps_a_hard_negative_out_of_the_control_group(tmp_path):
    """The exclusion the check cannot get wrong, because getting it wrong is invisible.

    The hard negatives live in the same Roboflow project as the products, so a frame with no boxes
    is the *other* population. `audit_recall.frame_paths` accepts any frame with a label file - an
    empty one included - so using it here would put negatives on both sides of the comparison and
    the phantom rate would come out lower than it is, with nothing on screen to say why.
    """
    gen = _fake_generation(tmp_path)
    images, labels = audit_recall.split_dirs(gen, "train")
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem, text in (("a", "0 0.5 0.5 0.2 0.2\n"), ("b", ""), ("c", "1 0.4 0.4 0.2 0.2\n")):
        (images / f"{stem}.jpg").write_bytes(b"")
        (labels / f"{stem}.txt").write_text(text, encoding="utf-8")

    got = {p.name for p in clamp_probe.labelled_frames(gen, "train", 60)}
    assert got == {"a.jpg", "c.jpg"}


def test_a_split_with_no_labelled_frame_refuses_rather_than_measuring_an_empty_control(tmp_path):
    """"0 real detections lost" is trivially true of no real detections, so the empty case stops."""
    gen = _fake_generation(tmp_path)
    images, labels = audit_recall.split_dirs(gen, "train")
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        clamp_probe.labelled_frames(gen, "train", 60)
    assert "control population" in str(exc.value)

    with pytest.raises(SystemExit) as missing:
        clamp_probe.labelled_frames(gen, "test", 60)
    assert "no such split" in str(missing.value)


def test_training_labels_are_read_from_both_label_forms(tmp_path):
    """Boxes and polygons, through `audit_recall.parse_labels` rather than a second reader.

    A polygon is the common form in this project's exports, and a reader assuming `cls cx cy w h`
    misreads it in a way that is invisible in review - see `parse_labels`' own docstring for the
    484-zero-area-labels report that came from exactly that.
    """
    gen = _fake_generation(tmp_path)
    _, labels = audit_recall.split_dirs(gen, "train")
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")  # box -> 0.4x0.5
    (labels / "b.txt").write_text(
        "1 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0\n", encoding="utf-8"
    )  # polygon -> the full frame
    assert clamp_probe.training_labels(gen, "train") == ((0.4, 0.3, 0.6, 0.7), (0.0, 0.0, 1.0, 1.0))


def test_the_sampling_rules_behind_the_published_table_are_pinned():
    """Sanity on the two population sizes the table rests on, from the tool's own constants.

    The table is `caught / phantoms` over one sample and `lost / real` over another, so a change to
    either sampling rule silently rescales a column of published numbers. Both are stated once,
    here, and the doc's figures are quoted for the same pair.
    """
    assert clamp_probe.REAL_SPLIT == "train"
    assert clamp_probe.REAL_SAMPLE == 60
    assert clamp_probe.NEGATIVES_DIR.name == "negative"
    assert clamp_probe.NEGATIVES_DIR.is_absolute()


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
    """`verify-clamp` is a gate, and a gate whose flag goes missing is a report nobody reads.

    Two ways it could stop gating without anyone noticing. **`--strict` dropped** leaves a target
    that prints five PASS/FAIL lines and exits 0 either way - a check with no failure mode, which
    is the shape `spec_check.py`'s `--strict` note already warns about. **Added to `test`'s
    dependencies** puts a 6.6 GB workspace and an installed weight on CI's bare checkout, where it
    would exit non-zero on "no such weight" for every push and every fresh clone.

    Nothing else in the repo reads the Makefile, so without this the two are one careless edit
    apart from the behaviour they were built to avoid.
    """
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = _recipe(text, "verify-clamp")
    assert "clamp_probe.py" in recipe
    assert "--strict" in recipe
    # Phony, or a directory of that name on PATH would shadow it and the target would no-op.
    assert "verify-clamp" in text[text.index(".PHONY") : text.index("help:")]
    # And it is reachable on its own but not dragged into the suite CI runs. Read as the target's
    # prerequisites rather than as a substring of a fixed-width window: a window is a magic number
    # that silently stops covering the name it is looking for.
    test_line = next(line for line in text.splitlines() if line.startswith("test:"))
    assert "verify-clamp" not in test_line.split(":", 1)[1].split()


def test_the_documented_gate_command_is_the_one_that_exists():
    """CLAUDE.md lists `verify-clamp` among the key targets, so the name has to be real."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "make verify-clamp" in claude
    assert "\nverify-clamp:" in makefile
    # Named in the help output too, since that is the only index of the targets.
    assert '"  verify-clamp' in makefile


def test_the_checklist_points_at_the_tool_that_produced_its_numbers():
    """The doc quotes this tool's numbers, so the doc has to name it.

    Before this tool existed those numbers came from scratch scripts that were never tracked, and
    the checklist cited a measurement with no way to re-run it. That is the drift this pins: a
    number in the doc whose provenance is a file that no longer exists.
    """
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "clamp_probe.py" in text
    # The whole command, not just the two names - the prose below the block also mentions
    # `--conf 0.5`, so asserting on the flag alone would still pass with it removed from the
    # command it is supposed to be in.
    assert "clamp_probe.py --generation v1 --conf 0.5" in text
