"""Tests for `audit_recall.py`: the crowding numbers, and a refreshed dataset.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
import pytest
import annotate.store
import audit_recall
import clamp_probe
import clean_v2
import dataset_doctor
import generations
import label_classes
import label_progress
import spec_check
import train_model
import workspace

# Rebuilt from the modules that own each fact, not copied by value.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "MODEL_TRAINING.md"

from tests.dataset_tool_helpers import _fake_generation, _local_args, _staged_local


# --- audit_recall: the numbers the crowding finding is stated in ----------------------------
#
# Everything the tool prints comes out of `RecallReport.add` / `measure`, which take records
# rather than a model, so the whole accounting layer is tested here against hand-built frames.
# The tool's only untested part is the inference loop that fills the records.


def _instance(cls: int, box: tuple[float, float, float, float]) -> audit_recall.Instance:
    return audit_recall.Instance(cls, box)


def _pred(cls: int, box: tuple[float, float, float, float], conf: float = 0.9):
    return audit_recall.Prediction(cls, box, conf)


BOXY = (0.1, 0.1, 0.3, 0.3)  # arbitrary but non-degenerate: areas and IoUs elsewhere use it


def test_parse_labels_reads_a_polygon_as_its_bounding_box():
    """The bug this tool exists downstream of.

    v1's export is mostly polygons - `cls x1 y1 x2 y2 ...` - so a reader that assumes
    `cls cx cy w h` multiplies two polygon *coordinates* together and calls the result an
    area. That misreading is what produced a report of 484 zero-area labels in a dataset that
    is fine, so the conversion gets pinned here rather than trusted to whoever reads the file
    next.
    """
    # A triangle-and-a-half wide: x from 0.1 to 0.5, y from 0.2 to 0.6.
    polygon = "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6"
    (found,) = audit_recall.parse_labels(polygon)
    assert found == _instance(0, (0.1, 0.2, 0.5, 0.6))
    # The claim that matters: the box is derived, not read, so a `w`/`h` reading is impossible.
    assert audit_recall.area(found.box) == pytest.approx(0.4 * 0.4)


def test_parse_labels_reads_a_plain_box_centre_and_size():
    """A 5-field line is `cx cy w h`, not four corner coordinates.

    This test previously asserted the *wrong* answer - it read the line as corners, which is how
    the tool shipped understating every class whose labels were in box format, and then agreed
    with itself afterwards. The values below are a real line from the v1 export, and the expected
    box is what ultralytics itself derives from it (its own labels cache reports `bbox_format:
    xywh` with exactly these numbers).
    """
    (found,) = audit_recall.parse_labels("1 0.5078125 0.578125 0.984375 0.84375")
    assert found.cls == 1
    assert found.box == pytest.approx((0.015625, 0.15625, 1.0, 1.0))


def test_a_box_is_centred_on_its_cx_cy_whatever_its_size():
    """The property that separates the two readings, over cases that break a corner-based one.

    Reading `cx cy w h` as corners keeps the box inside the frame and roughly object-sized for
    large `cx`, which is why it survives a glance; it fails hardest where `w < cx`, collapsing to
    zero width. Centring is the check that holds for every value.
    """
    for cx, cy, w, h in ((0.3, 0.4, 0.4, 0.4), (0.9, 0.2, 0.05, 0.7), (0.5, 0.5, 1.0, 1.0)):
        (found,) = audit_recall.parse_labels(f"0 {cx} {cy} {w} {h}")
        x1, y1, x2, y2 = found.box
        assert (x1 + x2) / 2 == pytest.approx(cx)
        assert (y1 + y2) / 2 == pytest.approx(cy)
        assert x2 - x1 == pytest.approx(w)
        assert y2 - y1 == pytest.approx(h)
        # And the size is the size: a zero area would be the corner reading on a small object.
        assert audit_recall.area(found.box) == pytest.approx(w * h)


def test_the_two_label_forms_are_told_apart_by_field_count():
    """A mixed export is normal, so the decision has to be per line and not per file.

    Both lines below describe the same square; the polygon spells out its four corners, the box
    gives its centre and size. One file may hold both - 32 of v1's test-split lines are boxes and
    the rest are polygons - so a per-file guess would get one of them wrong.
    """
    parsed = audit_recall.parse_labels(
        "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n"  # four corners
        "0 0.3 0.4 0.4 0.4"  # centre plus size
    )
    assert len(parsed) == 2
    assert all(i.cls == 0 for i in parsed)
    # Same square, reached two ways - and only the counts say which maths to use.
    # `approx` because the box branch is `cx - w/2`, which lands on 0.09999999999999998.
    for inst in parsed:
        assert inst.box == pytest.approx((0.1, 0.2, 0.5, 0.6))


def test_parse_labels_skips_a_malformed_line_rather_than_inventing_an_instance():
    """A line that is not five fields is not an object. Counting it would inflate the
    denominator, and a skipped line at least shows up as a frame whose count is one short."""
    text = "0 0.1 0.2 0.3 0.4\n\n0 1 2\nnot-a-number 0.1 0.2 0.3 0.4\n1 0.5 0.5 0.6 0.6\n"
    found = audit_recall.parse_labels(text)
    assert [i.cls for i in found] == [0, 1]


def test_match_instances_is_class_aware():
    """A confident box of the wrong class is not a hit.

    Two tins that look alike are the realistic version of this: scoring a cross-class box as
    a hit would report the confusion as a success and hide exactly the failure that matters
    for a grocery basket.
    """
    truth = [_instance(0, BOXY)]
    match = audit_recall.match_instances(truth, [_pred(1, BOXY)], 0.5)
    assert match.matched_truth == ()
    assert match.missed_truth == (0,)


def test_match_instances_is_one_to_one():
    """Duplicate boxes on one object are one hit, not two.

    Without the used-sets, a model that emits the same box twice would score higher than one
    that emits it once - recall going *up* as the predictions get worse.
    """
    truth = [_instance(0, BOXY)]
    preds = [_pred(0, BOXY, 0.9), _pred(0, (0.11, 0.11, 0.31, 0.31), 0.8)]
    match = audit_recall.match_instances(truth, preds, 0.5)
    assert match.matched_truth == (0,)
    assert len(match.matched_pred) == 1


def test_a_duplicate_box_is_a_false_positive_as_well_as_a_non_hit():
    """The other half of the one-to-one rule, and the number a threshold decision needs.

    The test above pins that a second box on one object is not a second *hit*. What it does not
    say is that the second box still exists: it is a prediction that matched no label, so it is a
    false positive, and `spurious` counts it. Dropping it silently would make a model that emits
    two boxes per object look exactly as clean as one that emits one.

    `predictions` is carried on the match because `spurious` is a subtraction - the leftovers are
    not derivable from the pairs that were made.
    """
    truth = [_instance(0, BOXY)]
    match = audit_recall.match_instances(
        truth, [_pred(0, BOXY, 0.9), _pred(0, (0.11, 0.11, 0.31, 0.31), 0.8)], 0.5
    )
    assert match.predictions == 2
    assert match.spurious == 1

    # A frame where every prediction found its label has none, so this is not counting hits.
    exact = audit_recall.match_instances(truth, [_pred(0, BOXY, 0.9)], 0.5)
    assert exact.predictions == 1 and exact.spurious == 0
    # The old, wrong reading of this property was `len(matched_pred)` - which on the frame above
    # is 1, so it would report a false positive on a perfect prediction and count hits as errors.
    assert exact.spurious != len(exact.matched_pred)


def test_a_wrong_class_box_is_one_miss_and_one_false_positive():
    """Not a hit and not nothing - the pair is what a confused model actually costs.

    For a grocery basket this is the expensive shape: the tin is not logged under its own name and
    a second row appears under a name that was never there.
    """
    match = audit_recall.match_instances([_instance(0, BOXY)], [_pred(1, BOXY)], 0.5)
    assert match.missed_truth == (0,)
    assert match.matched_pred == ()
    assert match.spurious == 1


def test_false_positives_are_counted_at_the_threshold_being_measured():
    """What makes `--conf-sweep` able to price a threshold rather than only sell one.

    Recall alone can only improve as the threshold falls, so a sweep of it recommends 0.0. The
    duplicate here sits at 0.55: below the operating point it is a false positive, above it is
    gone. Same records, two answers - which is the whole reason the records are kept raw and
    re-filtered per conf.
    """
    records = [
        audit_recall.FrameRecord(
            truth=(_instance(0, BOXY),),
            preds=(_pred(0, BOXY, 0.9), _pred(0, (0.11, 0.11, 0.31, 0.31), 0.55)),
        )
    ]
    assert audit_recall.measure(records, ("tin",), conf=0.5).spurious == 1
    assert audit_recall.measure(records, ("tin",), conf=0.7).spurious == 0
    # And the hit is a hit either way, so the two readings differ only in the cost column.
    assert audit_recall.measure(records, ("tin",), conf=0.5).found == 1
    assert audit_recall.measure(records, ("tin",), conf=0.7).found == 1


def test_the_report_splits_recall_by_distance_as_well_as_by_crowding():
    """The axis the threshold question is asked on: does a move help `far` and cost `close`?

    A third fold over the same match rather than a second pass, so the distance reading and the
    crowding reading cannot disagree about a frame.
    """
    report = audit_recall.measure(
        [_record(1, 1, distance="far"), _record(1, 0, distance="far"),
         _record(1, 1, distance="close")],
        ("tin",),
    )
    assert report.by_distance["far"].labelled == 2
    assert report.by_distance["far"].found == 1
    assert report.by_distance["far"].recall == pytest.approx(0.5)
    assert report.by_distance["close"].recall == pytest.approx(1.0)
    assert report.by_distance["mid"].labelled == 0
    # The frame counts too: they are the "came back complete" half of the tally, and a distance
    # bucket that counted instances but not frames would print a rate with no denominator behind
    # it. One of the two `far` frames had its label found, the other did not.
    assert report.by_distance["far"].frames == 2
    assert report.by_distance["far"].frames_clean == 1
    assert report.by_distance["close"].frames_clean == 1
    # Canonical order, and only the distances that held frames.
    assert report.distances == ["close", "far"]


def test_a_frame_with_no_distance_folds_into_no_distance_bucket():
    """v1's case, and the one that must not read as a clean bill of health.

    An untagged frame contributes to the overall number and to no distance row, so every bucket
    stays empty and `distances` is empty - which is what the tool turns into its "no such axis"
    sentence rather than a zero.
    """
    report = audit_recall.measure([_record(1, 1)], ("tin",))
    assert report.distances == []
    assert all(tally.labelled == 0 for tally in report.by_distance.values())
    assert report.found == 1  # still counted, just not attributed to a distance


def test_a_distance_cell_reads_a_dash_a_percentage_or_a_flagged_percentage():
    """Three states, because "this distance has none of that item" and "it missed all of them"
    lead to opposite work - and a `0.0%` would state the second while meaning the first."""
    unasked = audit_recall.BucketTally()
    assert audit_recall.distance_cell(unasked) == "        -"

    passing = audit_recall.BucketTally(labelled=10, found=9)
    assert audit_recall.distance_cell(passing).strip() == "90.0%"

    failing = audit_recall.BucketTally(labelled=10, found=4)
    cell = audit_recall.distance_cell(failing)
    assert cell.strip() == "40.0%!"
    assert len(cell) == len(audit_recall.distance_cell(unasked))  # columns line up


def test_the_distance_note_tells_apart_never_tagged_from_the_join_failing():
    """Two causes with opposite fixes: v1 has no distance axis at all (`manifest is None`, so
    re-running finds nothing) while a manifest that matched nothing is a broken join someone
    should look at. One sentence for both would hide the second behind the first."""
    v1 = generations.get("v1")
    never = audit_recall.distance_note(v1, "test", has_distance=False)
    assert "declares no distance axis" in never and "never tagged" in never

    v2 = generations.get("v2")
    failed = audit_recall.distance_note(v2, "test", has_distance=False)
    assert "matched none of the test frames" in failed
    assert "not reported as clean" in failed
    assert never != failed

    # And nothing at all when the columns are there, so a caller can print it unconditionally.
    assert audit_recall.distance_note(v1, "test", has_distance=True) == ""


def test_the_sweep_folds_in_the_threshold_that_is_actually_operating():
    """A table that omits the row for the threshold in force describes its neighbours instead.

    Same rule as `clamp_probe.sweep_tolerances`: the ladder and the running value are edited by
    different people at different times, so the row a decision is about cannot be left to luck.
    `--conf 0.7` is exactly the case - 0.7 is in no ladder, and without this the sweep would show
    0.1/0.2/0.3/0.5 and no row for the threshold being reasoned about.
    """
    folded = audit_recall.sweep_confs(0.7)
    assert 0.7 in folded
    assert set(audit_recall.CONF_SWEEP) <= set(folded)
    assert list(folded) == sorted(folded)
    # The shipped operating point is already a candidate: folded in, not duplicated.
    assert audit_recall.sweep_confs(0.5).count(0.5) == 1
    assert list(audit_recall.sweep_confs(0.5)) == list(audit_recall.CONF_SWEEP)

    # And the table is rendered from it, asserted as a row rather than as a substring: the fold is
    # derived inside `conf_sweep_lines` and cannot be passed in, so a `--conf 0.7 --conf-sweep`
    # run cannot print neighbours and no row for the threshold being reasoned about.
    lines = audit_recall.conf_sweep_lines(
        [_record(1, 1)], generations.get("v2"), "test", operating=0.7
    )
    rows = [line.lstrip() for line in lines if line.lstrip().startswith("0.")]
    assert len(rows) == len(audit_recall.CONF_SWEEP) + 1
    assert any(row.startswith("0.70") and "<-- operating" in row for row in rows)


def test_the_sweep_table_gains_a_column_per_distance_and_marks_the_operating_row():
    """The rendering, read as text - which is the only way to see the distance half today.

    v2 is the one generation with a distance axis and its export has not been downloaded yet, so a
    table built inline in `main()` would have this half unobservable until then. Every assertion
    below is about the printed table rather than about the accumulator behind it.
    """
    records = [_record(1, 1, distance="far"), _record(1, 0, distance="far"),
               _record(1, 1, distance="close")]
    lines = audit_recall.conf_sweep_lines(records, generations.get("v2"), "test", operating=0.5)
    head = next(line for line in lines if line.startswith("  conf"))
    for distance in ("close", "mid", "far"):
        assert distance in head
    # `extra` is the other half of the decision, so it is in the header too.
    assert "extra" in head

    # The conf is right-aligned in a 6-wide column, so a data row is "  0.50" - matched on the
    # stripped value rather than on a guess at the padding.
    rows = [line for line in lines if line.lstrip().startswith("0.")]
    assert len(rows) == len(audit_recall.CONF_SWEEP)  # 0.5 is already on the ladder
    # Exactly one row is the operating one, and it is the 0.5.
    marked = [row for row in rows if "<-- operating" in row]
    assert len(marked) == 1 and marked[0].lstrip().startswith("0.50")
    # A distance that held no frames reads as `-`, not as 0.0% - `mid` here.
    assert "-" in marked[0]
    # And with columns present the "no distance axis" note is not printed.
    assert not any("no per-distance columns" in line for line in lines)
    # Every data row is exactly as wide as the header, marker aside. Exact rather than "no wider
    # than": the three-state cell puts a `!` *inside* its nine characters, so a flag that overflowed
    # would shift every column after it - and a `!` in the middle (mid-distance under the floor) is
    # the case that would collide with the next column's leading space.
    assert {len(row.replace("   <-- operating", "")) for row in rows} == {len(head)}


def test_the_sweep_table_omits_the_distance_columns_and_says_why_when_there_are_none():
    """v1's case: a table with no distance columns has to say which of the two causes it is,
    or a reader cannot tell "never tagged" from "the join broke"."""
    records = [_record(1, 1), _record(1, 0)]
    lines = audit_recall.conf_sweep_lines(records, generations.get("v1"), "test", operating=0.5)
    head = next(line for line in lines if line.startswith("  conf"))
    assert "far" not in head and "close" not in head
    assert "extra" in head
    # The row still marks the operating threshold, so the columns are absent, not the information.
    assert any("<-- operating" in line for line in lines)
    # Uniform rows, counted the same way as the distance case - and the header is narrower here,
    # because three distance columns are missing rather than empty.
    table = [line for line in lines if line.lstrip().startswith("0.")]
    assert table
    assert {len(row.replace("   <-- operating", "")) for row in table} == {len(head)}
    assert len(head) == 80 - 27
    note = next(line for line in lines if "no per-distance columns" in line)
    assert "declares no distance axis" in note


def test_the_training_doc_describes_the_columns_the_sweep_prints():
    """The doc's `--conf-sweep` bullet is the only place the table is explained to a user.

    It described a recall-only table, which is precisely the reading that recommends 0.0 - so the
    two things a threshold decision needs are pinned: that `extra` exists and what it means, and
    that the row for the running threshold is marked. Same shape as the guard that keeps
    `spec_check.py` named in the acceptance section.
    """
    text = DOC.read_text(encoding="utf-8")
    start = text.index("**`--conf-sweep`")
    section = text[start : text.index("**`--iou-sweep`", start)]
    assert "`extra`" in section
    assert "<-- operating" in section
    # And the distance columns, since they are the axis a v2 threshold decision is asked on.
    assert "Distance columns" in section
    # What the column *means*, not only that it exists: the table's own header also names `extra`,
    # so a name-only assertion stays green while the definition is deleted - and the definition is
    # the half a reader needs to read the number against recall.
    assert "predictions that matched no label" in section


def test_collect_joins_distance_on_the_export_filename_and_not_a_second_reader():
    """The join is the trainer's, imported rather than re-derived.

    A YOLO export carries no tags, so the distance comes from the workspace manifest keyed on the
    export filename - and `train_model.distance_map` is where that rule already lives, for `--val`.
    A second reader here could disagree with the trainer about which image is `far` while both
    looked correct, which is the drift this repo keeps hand-mirrored contracts in tests for.
    """
    source = Path(audit_recall.__file__).read_text(encoding="utf-8")
    assert re.search(
        r"^from train_model import DISTANCE_ORDER, distance_map", source, re.MULTILINE
    )
    assert "distance_map(generation.manifest)" in source
    assert 'distances.get(path.name, "")' in source
    # No second copy of the ordering, and no second manifest parser.
    assert "DISTANCE_ORDER = (" not in source


def test_the_distance_axis_is_defined_once_and_every_tool_reads_that_one():
    """Three tools, one axis: `clean_v2` files frames by it, `label_progress` prints its columns
    and `train_model` validates one pass per distance. A second literal is a tool that silently
    stops knowing about a fourth distance - the frames get staged under a folder nothing else
    understands, `recorded_distances` drops them from the grid, and the doctor goes on printing
    the row. The identity assertions are the point: one object, four readers.
    """
    assert clean_v2.DISTANCE_ORDER is label_classes.DISTANCE_ORDER
    assert label_progress.DISTANCE_ORDER is label_classes.DISTANCE_ORDER
    assert train_model.DISTANCE_ORDER is label_classes.DISTANCE_ORDER
    assert dataset_doctor.DISTANCE_ORDER is label_classes.DISTANCE_ORDER
    # Every module in the tools tree, not only those four: a *new* tool that decides to spell the
    # axis out is the drift this guard exists for, and the narrow version of this scan was
    # mutation-checked vacuous - a definition added to `audit_v2.py` passed it.
    offenders = [
        path.name
        for path in sorted((Path(label_classes.__file__).parent).glob("*.py"))
        if path.name != "label_classes.py"
        and "DISTANCE_ORDER = " in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    # And no folder spelling can name a distance the axis does not have.
    assert set(clean_v2.DISTANCE_MAP.values()) <= set(label_classes.DISTANCE_ORDER)


def test_every_reader_of_the_tag_rule_goes_through_the_one_owner(tmp_path, monkeypatch):
    """Neither reader restates the rule: each hands its frame's drawn classes to the one
    `label_classes.tag_mismatch`. The spy is the assertion because each reader's *verdict* is
    already covered by its own tests - this is the wiring between them, which no verdict test can
    see. `label_progress` is exercised through its local store, whose shape differs from the
    Roboflow source's summary and which needs no fake project.
    """
    seen: list[str] = []

    def spy(slug: str, drawn: object) -> tuple[str, ...]:
        seen.append(slug)
        return ()

    out = _staged_local(tmp_path, frames=1)
    store = annotate.store.LabelStore(out=out, annotations=tmp_path / "annotations-v2")
    store.write("milo_0001.jpg", [annotate.store.Box(0, 0.5, 0.5, 0.2, 0.2)])
    monkeypatch.setattr(label_progress, "tag_mismatch", spy)
    label_progress.local_progress(_local_args(out), out)
    assert seen == ["milo"]

    monkeypatch.setattr(dataset_doctor, "tag_mismatch", spy)
    assert dataset_doctor.tag_problems({"x.jpg": "milo"}, {"x.jpg": {1}}, ["a", "b"]) == ([], [])
    assert seen == ["milo", "milo"]


def test_match_instances_gives_a_contested_prediction_to_the_better_label():
    """Best pair first, which is what makes the result independent of label order."""
    left = (0.1, 0.1, 0.3, 0.3)
    right = (0.12, 0.1, 0.32, 0.3)  # overlaps `left` heavily
    pred = (0.119, 0.1, 0.319, 0.3)  # nearer `right` than `left`
    for truth in ([_instance(0, left), _instance(0, right)],
                  [_instance(0, right), _instance(0, left)]):
        match = audit_recall.match_instances(truth, [_pred(0, pred)], 0.5)
        better = 0 if truth[0].box == right else 1
        assert match.matched_truth == (better,), "the closer label should win either order"


def _record(
    n_labels: int, n_hits: int, conf: float = 0.9, distance: str = ""
) -> audit_recall.FrameRecord:
    """A frame with `n_labels` boxes in a row, of which the first `n_hits` are predicted."""
    boxes = [(0.1 + 0.2 * i, 0.1, 0.25 + 0.2 * i, 0.25) for i in range(n_labels)]
    truth = tuple(_instance(0, b) for b in boxes)
    preds = tuple(_pred(0, b, conf) for b in boxes[:n_hits])
    return audit_recall.FrameRecord(truth=truth, preds=preds, distance=distance)


def test_the_report_splits_recall_by_frame_crowding():
    """The finding, as an assertion: same hit *rate* per frame, different per-instance rate."""
    records = [_record(1, 1), _record(1, 0), _record(2, 1), _record(3, 1)]
    report = audit_recall.measure(records, ("tin",))

    single, multi = report.buckets[audit_recall.SINGLE], report.buckets[audit_recall.MULTI]
    assert (single.found, single.labelled) == (1, 2)
    assert (multi.found, multi.labelled) == (2, 5)
    # Three of the four frames found *something*, but only two found everything - which is
    # the gap a frame-level count cannot see and this whole tool is about.
    assert report.buckets[audit_recall.MULTI].frames_clean == 0
    assert report.recall == pytest.approx(3 / 7)


def test_a_frame_with_only_one_of_two_instances_found_is_not_a_clean_frame():
    """The single reading that separates "missed the object" from "missed the second one"."""
    report = audit_recall.measure([_record(2, 1)], ("tin",))
    multi = report.buckets[audit_recall.MULTI]
    assert multi.frames == 1 and multi.frames_clean == 0
    assert multi.recall == 0.5


def test_a_missed_instance_half_the_area_of_a_hit_shows_up_in_the_areas():
    """Why the areas are reported: size is how "small object" is told from "second object"."""
    big = (0.1, 0.1, 0.4, 0.4)  # area 0.09
    small = (0.5, 0.5, 0.6, 0.6)  # area 0.01
    record = audit_recall.FrameRecord(
        truth=(_instance(0, big), _instance(0, small)),
        preds=(_pred(0, big),),
    )
    tally = audit_recall.measure([record], ("tin",)).per_class[0]
    assert tally.median_hit_area == pytest.approx(0.09)
    assert tally.median_miss_area == pytest.approx(0.01)


def test_a_higher_threshold_never_finds_more_than_a_lower_one():
    """This is what makes the conf sweep cost no extra pass.

    `collect` runs once at the lowest threshold asked about and `measure` re-filters, which is
    sound only if the found set shrinks as the threshold rises. If it can grow, then a run at
    a higher conf would not have produced the boxes the report is counting.
    """
    records = [_record(2, 2, conf=0.9), _record(2, 1, conf=0.9), _record(1, 1, conf=0.2)]
    thresholds = [0.1, 0.3, 0.5, 0.9]
    found = [audit_recall.measure(records, ("tin",), c).found for c in thresholds]
    assert found == sorted(found, reverse=True)
    # Five instances labelled and exactly one prediction sits near the slope: the conf-0.2 hit
    # is counted at 0.1 and gone from 0.3 up, while the conf-0.9 hits survive at 0.9 (the
    # filter is `>=`). So the sequence is 4 then 3, flat after - one step, not a slide.
    assert found == [4, 3, 3, 3]


def test_off_roster_predictions_are_counted_and_never_matched():
    """A weight with more head outputs than the dataset declares is caught, not ignored.

    Left in the matched set, an off-roster name would be filtered against an index that does
    not exist; dropped silently, a 24-class weight would audit as a clean 7-class one.
    """
    record = audit_recall.FrameRecord(
        truth=(_instance(0, BOXY),), preds=(_pred(0, BOXY),), off_roster=("tin close",)
    )
    report = audit_recall.measure([record], ("tin",))
    assert report.off_roster["tin close"] == 1
    assert report.found == 1


def test_the_floor_shortlist_is_worst_first_and_only_names_real_shortfalls():
    classes = ("good", "bad", "worse", "unlabelled")
    records = []
    for cls, hits, n in ((0, 9, 10), (1, 8, 10), (2, 5, 10)):
        boxes = [(0.1 + 0.2 * i, 0.1, 0.25 + 0.2 * i, 0.25) for i in range(n)]
        records.append(
            audit_recall.FrameRecord(
                truth=tuple(_instance(cls, b) for b in boxes),
                preds=tuple(_pred(cls, b) for b in boxes[:hits]),
            )
        )
    report = audit_recall.measure(records, classes)
    assert report.below_floor == ["worse", "bad"]


def test_the_verdict_names_the_fix_the_gap_points_at():
    """The sentence is the deliverable - a recall pair alone does not say what to do.

    The buckets are built past `MIN_BUCKET_INSTANCES` on purpose: a verdict off a handful of
    instances is the thing the next test forbids, so a test of the three readings has to be
    big enough to earn one.
    """
    # single 20/20, crowded 15/35 - a wide gap.
    crowded = audit_recall.measure(
        [_record(1, 1)] * 20 + [_record(2, 1)] * 10 + [_record(3, 1)] * 5, ("tin",)
    )
    assert "multi-item scenes" in audit_recall.crowding_verdict(crowded)

    # single 16/20, crowded 16/20 - the same rate either way.
    spread = audit_recall.measure(
        [_record(1, 1)] * 16 + [_record(1, 0)] * 4
        + [_record(2, 2)] * 6 + [_record(2, 1)] * 4,
        ("tin",),
    )
    assert "more shots" in audit_recall.crowding_verdict(spread)

    # single 10/20, crowded 20/20 - the reverse direction, which is not "no gap".
    backwards = audit_recall.measure(
        [_record(1, 1)] * 10 + [_record(1, 0)] * 10 + [_record(2, 2)] * 10, ("tin",)
    )
    verdict = audit_recall.crowding_verdict(backwards)
    assert "scores *higher*" in verdict and "solo objects" in verdict


def test_the_verdict_refuses_to_compare_a_bucket_that_is_too_thin():
    """The refusal is the load-bearing half.

    `--limit 20` is how a smoke test is run, and it leaves the single-object bucket holding a
    couple of instances - 0% or 100%, either way noise. Printed as a verdict that reads as a
    finding about the model, so the tool says it cannot tell yet instead.
    """
    thin = audit_recall.measure([_record(1, 1), _record(1, 0), _record(2, 1)], ("tin",))
    verdict = audit_recall.crowding_verdict(thin)
    assert "Not enough to compare yet" in verdict
    assert "without --limit" in verdict
    # And it passes no judgement in either direction while it cannot tell.
    assert "multi-item scenes" not in verdict and "solo objects" not in verdict


def test_the_verdict_compares_once_the_buckets_are_big_enough():
    """The other side of the guard, and where the boundary sits: exactly the minimum is enough.

    Both buckets land on exactly 20 and 24 instances, so this pins the comparison as inclusive
    (`< minimum` refuses, `== minimum` reads) rather than leaving the boundary to chance.
    """
    big = audit_recall.measure([_record(1, 1)] * 20 + [_record(2, 1)] * 12, ("tin",))
    assert audit_recall.MIN_BUCKET_INSTANCES == 20
    assert big.buckets[audit_recall.SINGLE].labelled == 20
    verdict = audit_recall.crowding_verdict(big)
    assert "Not enough to compare yet" not in verdict
    assert "multi-item scenes" in verdict


def test_the_report_names_the_device_the_way_the_app_does():
    """`resolve_device` answers ultralytics' index (`0`), which beside `conf>=0.5` in the header
    reads as a second threshold. The report says `cuda:0`, which is what the app logs."""
    assert audit_recall.device_label("0") == "cuda:0"
    assert audit_recall.device_label("cpu") == "cpu"
    assert audit_recall.device_label("cuda") == "cuda"


def test_audit_recall_does_not_import_ultralytics_at_module_level():
    """Same shape as the trainer: importable, and its accounting testable, with no torch."""
    source = Path(audit_recall.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^(from|import)\s+ultralytics", source, re.MULTILINE)
    assert re.search(r"^\s+from ultralytics import", source, re.MULTILINE)


def test_audit_recall_measures_at_the_app_s_operating_point_by_default():
    """The one number that has to agree with the running app, since the tool covers all the
    others with flags."""
    from app.settings import Settings

    defaults = Settings()
    assert audit_recall.DEFAULT_CONF == defaults.conf_threshold
    assert audit_recall.DEFAULT_IMGSZ == defaults.imgsz
    # And the floor both the tool and the runbook quote, rather than a second one.
    assert audit_recall.RECALL_FLOOR == 0.85


def test_the_training_doc_points_at_the_recall_audit():
    """The tool is only discoverable if the doc that owns the acceptance number names it.

    `--val` is what the runbook quotes, so `audit_recall.py` is the second command a reader of
    that section has to be told about - otherwise a class below the floor gets acted on with
    half the diagnosis and the crowding gap is never looked for.
    """
    text = DOC.read_text(encoding="utf-8")
    assert "audit_recall.py" in text
    # Named where the reading lives, in the same section as the floor it is read against.
    floor_at = text.index("### What \"good\" looks like")
    section = text[floor_at : text.index("## 7. Integrating", floor_at)]
    assert "audit_recall.py" in section
    # And both sweeps, since each is a separate thing to run rather than a default.
    assert "--conf-sweep" in section and "--iou-sweep" in section


def test_frame_paths_ignores_an_image_with_no_label(tmp_path):
    """An unlabelled image has no ground truth, so it cannot contribute a recall number."""
    gen = dataclasses.replace(generations.V1, export_dir=tmp_path)
    images, labels = tmp_path / "test" / "images", tmp_path / "test" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem in ("a", "b"):
        (images / f"{stem}.jpg").write_bytes(b"")
    (labels / "a.txt").write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")

    assert [p.name for p in audit_recall.frame_paths(gen, "test")] == ["a.jpg"]


def test_frame_paths_says_so_when_the_split_is_not_there(tmp_path):
    """A missing export is the common case (v2 is not downloaded yet) and deserves a sentence
    rather than an empty report that reads as a model that detected nothing."""
    gen = dataclasses.replace(generations.V1, export_dir=tmp_path)
    with pytest.raises(SystemExit) as exc:
        audit_recall.frame_paths(gen, "test")
    assert "no such split" in str(exc.value)


# --- audit_recall: a refreshed dataset, and the labels' own size gap --------------------------


def test_the_dataset_override_replaces_what_the_spec_names_and_keeps_its_identity():
    """`replace()` on the frozen spec, not a second generation.

    The name, the class list and the weight's filename have to survive an override: they are what
    the record `--install` writes and what the app's roster check reads, so a "refreshed v1" that
    changed any of them would not be a drop-in for the model it is meant to replace.
    """
    spec = generations.get("v1")
    assert audit_recall.resolve_dataset(spec) is spec  # no flags, no copy
    moved = audit_recall.resolve_dataset(spec, "D:/scanncart/export-v1-s2")
    assert moved.export_dir == Path("D:/scanncart/export-v1-s2")
    # The manifest was not overridden, so it stays what the spec declared - `None` for v1.
    assert moved.manifest is None
    assert (moved.name, moved.classes, moved.weight_name) == (
        spec.name,
        spec.classes,
        spec.weight_name,
    )


def test_the_manifest_override_moves_v1_off_the_never_tagged_sentence():
    """The flag's whole effect on what a reader is told, which is why it is not the same flag.

    `distance_note` separates two causes because their fixes are opposite: a generation with no
    distance axis is one nobody tagged and re-running finds nothing, while a manifest that matched
    nothing is a broken join someone should look at. v1's spec declares no axis, so this is the
    pair of readings a refreshed set flips between.
    """
    v1 = generations.get("v1")
    assert "declares no distance axis" in audit_recall.distance_note(v1, "test", False)

    tagged = audit_recall.resolve_dataset(v1, "", "C:/w/cleaned-v1-s2/manifest.json")
    note = audit_recall.distance_note(tagged, "test", False)
    assert "declares no distance axis" not in note
    assert "matched none of the test frames" in note


def test_both_spellings_of_the_dataset_flag_reach_one_destination():
    """The trainer's pair, spelled the same way on purpose.

    The documents still call it `--export-dir`, and a command that reads a refreshed export in the
    trainer has to read the same one here - a second spelling would be a second flag to keep in
    step, which is how one tool ends up measuring the dataset the other did not train on.
    """
    for flag in ("--dataset-dir", "--export-dir"):
        assert audit_recall.parse_args(["--generation", "v1", flag, "X"]).dataset_dir == "X"
    default = audit_recall.parse_args(["--generation", "v1"])
    assert (default.dataset_dir, default.manifest) == ("", "")


def test_a_refreshed_export_is_measured_without_editing_the_spec(tmp_path):
    """End to end through the filesystem, and both halves of the override: the header a reader
    checks first, and the labels the number comes from."""
    root = tmp_path / "export-v1-s2"
    gen = audit_recall.resolve_dataset(generations.get("v1"), str(root))
    images, labels = audit_recall.split_dirs(gen, "test")
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"")
    (labels / "a.txt").write_text("0 0.5 0.5 0.02 0.02\n", encoding="utf-8")

    name = generations.V1.classes[0]
    tallies = audit_recall.label_sizes(gen, "test", 1280)
    assert set(tallies) == {name}
    # 0.02 of a 1280-wide capture is 25.6 px - below the far band's own floor, hence `tiny`.
    assert tallies[name].counts["tiny"] == 1
    assert tallies[name].smallest == pytest.approx(25.6)

    lines = audit_recall.report_lines(
        audit_recall.measure([_record(1, 1)], gen.classes),
        weights="w.pt",
        generation=gen,
        split="test",
        resize_mode="stretch",
        device="cpu",
    )
    assert f"dataset    {root}" in lines


def test_the_size_bands_are_the_recapture_plans_bands_at_their_boundaries():
    """The plan's three shoot bands, plus the tail below the far one.

    The lower edge is inclusive, and that is the only thing here that can be got wrong silently: a
    table that disagreed with the tape marks by a pixel would be a shooting instruction nobody
    could follow.
    """
    assert audit_recall.SIZE_BANDS == ("large", "medium", "small", "tiny")
    assert [audit_recall.band_for(w) for w in (300, 299.9, 120, 119.9, 40, 39.9)] == [
        "large",
        "medium",
        "medium",
        "small",
        "small",
        "tiny",
    ]


def test_a_box_is_measured_as_a_share_of_frame_width():
    """Why the reading is relative at all, and what `--capture-width` is for.

    `stretch` scales x and y by different factors, and both an export's labels and the app's own
    detections are stored as a share of the frame - where a per-axis scale cancels out. The flag
    converts that share back to the pixels an operator reads off the overlay, so a wrong value
    moves every row together and can reorder nothing.
    """
    assert audit_recall.box_width_px((0.1, 0.0, 0.6, 0.5), 1280) == pytest.approx(640)
    assert audit_recall.box_width_px((0.1, 0.0, 0.6, 0.5), 640) == pytest.approx(320)
    half = audit_recall.box_width_px((0.1, 0.0, 0.6, 0.5), 1280)
    assert audit_recall.band_for(half) == "large"


def test_an_instance_below_the_far_band_is_counted_rather_than_folded_into_it():
    """`tiny` is its own band on purpose: a 25 px sachet and a 100 px tin are the same problem to
    a threshold and different ones to a shoot list, and folding them together hides the tail that
    a shoot list exists to name."""
    tally = audit_recall.SizeTally()
    for width in (25.6, 60.0, 150.0, 400.0):
        tally.add(width)
    assert tally.counts == {"large": 1, "medium": 1, "small": 1, "tiny": 1}
    assert tally.instances == 4 and tally.smallest == 25.6


def test_a_class_the_split_never_labelled_is_short_in_every_band():
    """A class with no instances has no measured cell, and a row of zeros is exactly the reading
    that looks like a covered dataset."""
    tally = audit_recall.SizeTally()
    assert (tally.instances, tally.smallest) == (0, 0.0)
    assert tally.thin(1) == audit_recall.SIZE_BANDS

    lines = audit_recall.size_lines({"tuna": tally}, target=1)
    row = next(line for line in lines if line.startswith("tuna"))
    assert row.count("!") == len(audit_recall.SIZE_BANDS)
    assert row.rstrip().endswith("-")  # nothing to print as a smallest


def test_the_shoot_list_leads_with_the_class_short_in_the_most_bands():
    """Gap order, decided by the tool rather than by the reader.

    v1 has a class whose name starts with a digit, so an alphabetised table would lead with the
    one ordering that answers no question at all - and the list is a worklist, so the class
    missing the most bands is the one to shoot first.
    """
    thin = audit_recall.SizeTally()
    thin.add(500)
    thin.add(60)  # large and small only: four thin bands against the other class's none
    roomy = audit_recall.SizeTally()
    for _ in range(3):
        for width in (500, 150, 60, 25):
            roomy.add(width)

    lines = audit_recall.size_lines({"555 sardines": thin, "milo": roomy}, target=2)
    start = lines.index("shoot list - bands under the 2-instance target, worst first:")
    shoot = [line for line in lines[start + 1 :] if line.startswith("  ")]
    assert shoot[0].strip().startswith("555 sardines")
    assert "medium 0/2" in shoot[0] and "large 1/2" in shoot[0]
    # A class that clears the target in every band is not on the list at all.
    assert not any("milo" in line for line in shoot)
    # The table above it is in the same order, so the list and the histogram cannot disagree.
    table = [line for line in lines if line.startswith(("555", "milo"))]
    assert table[0].startswith("555")


def test_the_shoot_list_says_so_when_every_band_is_covered():
    """An empty shoot list has to read as an answer, not as a list that failed to print."""
    tally = audit_recall.SizeTally()
    for width in (500, 150, 60, 25):
        tally.add(width)
    lines = audit_recall.size_lines({"milo": tally}, target=1)
    assert any("every class clears 1 instance(s) in every band" in line for line in lines)


def test_the_histogram_says_so_when_the_split_holds_no_labels():
    """An empty split is the way this run goes wrong - a `--split train` against an export that
    only shipped `test` - so it is stated rather than rendered as a blank table."""
    lines = audit_recall.size_lines({})
    assert any("no labelled instances" in line for line in lines)


def test_the_size_histogram_runs_before_the_weight_is_resolved(tmp_path, capsys):
    """The mode answers "what do I shoot next", which is asked *before* a refreshed weight exists,
    so it must not go looking for one. `models/scanncart-grocery-v1.pt` is exactly the file that
    is not on this machine, which is what makes this an assertion about ordering rather than
    about a fixture."""
    gen = _fake_generation(tmp_path)
    images, labels = audit_recall.split_dirs(gen, "test")
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"")
    (labels / "a.txt").write_text("0 0.5 0.5 0.02 0.02\n", encoding="utf-8")

    code = audit_recall.main(
        ["--generation", "v1", "--dataset-dir", str(tmp_path), "--size-histogram"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert f"dataset {tmp_path}" in printed
    assert "label widths, per class" in printed
    assert "shoot list" in printed


def test_the_recapture_plan_and_the_tool_agree_on_the_bands():
    """The bands are the plan's, quoted rather than re-derived by whoever reads the tool next.

    The plan states them in the unit an operator can read off the Live overlay, and the tool
    converts the same pixels at the capture width - so the two have to be edited together, or the
    shoot list starts recommending sizes the tape marks do not produce.
    """
    text = (
        REPO_ROOT / "docs/superpowers/plans/2026-09-26-v1-varied-size-recapture.md"
    ).read_text(encoding="utf-8")
    for band in ("≥ 300 px", "120–300 px", "40–120 px"):
        assert band in text
    # And the two overrides section 8 asks for, named the way the tool spells them.
    assert "--dataset-dir" in text and "--manifest" in text
