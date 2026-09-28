"""Tests for `sidecar/tools/dataset_doctor.py` - the check between `build_dataset` and training.

No GPU, no workspace, no weights: every case here is a small dataset under `tmp_path` plus the
pure rules. What is worth asserting is not that the doctor runs, but that it fails on exactly the
six silent failures its docstring names - an out-of-order class list, a label row nothing can
read, a frame drawn under a class other than the one it was staged as, a test frame that duplicates
a train one, a merge report describing a different build, and a merge report whose v2 side
contributed nothing - and that it stays quiet on the states that are decisions rather than mistakes
(an empty label, a box on a hard negative, a frame count that is merely thin, a set that did not
come from `build_dataset`, a report that says nothing either way about its sources).

    sidecar/.venv/Scripts/python.exe -m pytest tests/test_dataset_doctor.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dataset_doctor
import generations
import label_classes
from label_classes import SLUG_TO_CLASS

V2 = generations.V2


# --------------------------------------------------------------------------
# Fixtures: a small merged set, built the way `build_dataset.py` writes one
# --------------------------------------------------------------------------


def _jpeg(path: Path, data: bytes) -> None:
    """A real JPEG from `data`, so two frames written from the same bytes are a near-duplicate."""
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 3)
    Image.fromarray(pixels).save(path, quality=60)


def _noise() -> bytes:
    """Random pixels, not a ramp: two frames of a smooth gradient fingerprint as near-identical
    whatever their offset, which would make every frame in the fixture a duplicate of every other."""
    import os

    return os.urandom(64 * 64 * 3)


def _dataset(
    tmp_path: Path,
    *,
    names: list[str] | None = None,
    rows: str = "0 0.5 0.5 0.2 0.2\n",
    counts: dict[str, int] | None = None,
    labels: bool = True,
    merge: object = "auto",
    duplicate_test: bool = False,
    yaml_generation: str | None = None,
    tags: dict[str, str] | None = None,
) -> Path:
    """The shapes `build_dataset.py` writes: three splits, labels beside them, `data.yaml`, report."""
    import yaml

    dataset = tmp_path / "merged-v2"
    counts = counts or {"train": 2, "valid": 2, "test": 2}
    shared: bytes | None = _noise() if duplicate_test else None
    for split, count in counts.items():
        for index in range(count):
            name = f"{split}_{index:04d}.jpg"
            data = shared if (duplicate_test and index == 0 and split in ("train", "test")) else _noise()
            _jpeg(dataset / split / "images" / name, data)
            if labels:
                (dataset / split / "labels").mkdir(parents=True, exist_ok=True)
                (dataset / split / "labels" / f"{split}_{index:04d}.txt").write_text(
                    rows, encoding="utf-8"
                )
    yaml_body = {
        "path": str(dataset),
        "nc": len(names or V2.classes),
        "names": names or list(V2.classes),
    }
    if yaml_generation:
        yaml_body["generation"] = yaml_generation
    (dataset / "data.yaml").write_text(yaml.safe_dump(yaml_body), encoding="utf-8")
    if merge == "auto":
        merge = {
            # `build_dataset.py` records which generation the set was built for, and the class
            # order it wrote is that generation's - the doctor reads both rather than assuming.
            "generation": V2.name,
            "classes": list(names or V2.classes),
            "splits": {split: {"images": count} for split, count in counts.items()},
            "machine_only_by_split": {"valid": 0, "test": 0},
        }
    if isinstance(merge, dict):
        # `build_dataset.py` records each written v2 frame's staged class in `tags`, and the doctor
        # checks every drawing against it - a report without the block is the state the tests below
        # call "unverified", so it is written only when a test asks for it.
        if tags is not None:
            merge = {**merge, "tags": tags}
        (dataset / "merge_report.json").write_text(json.dumps(merge), encoding="utf-8")
    return dataset


# --------------------------------------------------------------------------
# The pure rules: what a label row has to be, and absent-vs-empty
# --------------------------------------------------------------------------


def test_a_well_formed_row_has_nothing_to_report():
    assert dataset_doctor.label_problems([["0", "0.5", "0.5", "0.2", "0.2"]], ["a"]) == []


def test_every_way_a_row_can_be_wrong_is_named():
    """Each sentence has to say *which* mistake, because the fixes differ: a wrong index is a
    remap, a bad coordinate is a re-annotation, and a short row is a truncated write."""
    names = ["a", "b"]
    short = dataset_doctor.label_problems([["0", "0.5", "0.5"]], names)
    assert any("3 field(s)" in problem for problem in short)

    text = dataset_doctor.label_problems([["a", "0.5", "0.5", "0.2", "0.2"]], names)
    assert any("non-numeric" in problem for problem in text)

    index = dataset_doctor.label_problems([["7", "0.5", "0.5", "0.2", "0.2"]], names)
    assert any("class index 7 is outside the 2 declared classes" in problem for problem in index)

    outside = dataset_doctor.label_problems([["0", "1.5", "0.5", "0.2", "0.2"]], names)
    assert any("outside 0..1" in problem for problem in outside)

    flat = dataset_doctor.label_problems([["0", "0.5", "0.5", "0.0", "0.2"]], names)
    assert any("no extent" in problem for problem in flat)


def test_a_row_with_several_faults_is_reported_once_for_each(tmp_path):
    """A single row can be wrong twice - and reporting only the first fault sends the operator back
    to fix a file they will have to look at again."""
    problems = dataset_doctor.label_problems([["9", "2.0", "0.5", "0.2", "0.2"]], ["a"])
    assert len(problems) == 2


def test_a_polygon_row_is_a_row_the_loader_reads_rather_than_a_broken_one():
    """1,921 of v1's 2,111 label rows are polygons - Roboflow writes a mask that way.

    A doctor that called every non-five-field row unreadable would refuse the project's own export,
    and a guard that refuses a good set is how the real finding gets waved through. What the loader
    does with the row is the definition here, and the drift guard at the end of this file re-takes
    that measurement against the loader itself.
    """
    names = ["a"]
    triangle = ["0", "0.1", "0.1", "0.5", "0.1", "0.3", "0.4"]  # 3 points
    assert dataset_doctor.row_shape(triangle) == "polygon"
    assert dataset_doctor.label_problems([triangle], names) == []
    assert dataset_doctor.label_problems([["0", "0.5", "0.5", "0.2", "0.2"]], names) == []

    # A polygon is still checked: the class indexes the declared list, and its points have to
    # describe a box with some area - the loader builds the rectangle around them.
    out_of_range = dataset_doctor.label_problems([["0", "0.1", "0.1", "1.4", "0.1", "0.3", "0.4"]], names)
    assert any("outside 0..1" in problem for problem in out_of_range)
    assert any("drops the whole frame" in problem for problem in out_of_range)
    flat = dataset_doctor.label_problems([["0", "0.1", "0.5", "0.2", "0.5", "0.3", "0.5"]], names)
    assert any("no extent" in problem for problem in flat)
    wrong_class = dataset_doctor.label_problems([["9", "0.1", "0.1", "0.5", "0.1", "0.3", "0.4"]], names)
    assert any("class index 9" in problem for problem in wrong_class)


def test_the_row_shapes_that_are_not_box_or_polygon_are_named_by_the_column_count():
    """Six fields and an odd coordinate count are the cases with no reading at all: >6 makes a file
    a polygon file and 5 is a box, so 6 hits the loader's column assert and the whole frame is
    dropped - which is the failure the row count is in the message for.

    (Seven fields is *not* one of them: 6 coordinates is 3 points, which is the smallest polygon
    the loader accepts - the case above.)
    """
    assert dataset_doctor.row_shape(["0", "0.1", "0.1", "0.5", "0.1", "0.5"]) == "neither"
    assert dataset_doctor.row_shape(["0", "0.1", "0.1", "0.5", "0.1", "0.5", "0.3", "0.4"]) == "neither"

    problems = dataset_doctor.label_problems([["0", "0.1", "0.1", "0.5", "0.1", "0.5"]], ["a"])
    assert any("6 field(s)" in problem and "drops the whole frame" in problem for problem in problems)


def test_a_file_that_mixes_a_box_and_a_polygon_is_the_silent_case():
    """The loader decides per *file*: one polygon row makes every row a point list, so the box rows
    are reshaped into two-point polygons and train as a box nobody drew - no exception, no warning,
    and a label file that reads correctly to a human. Named rather than counted, because the fix is
    "re-export this frame in one shape"."""
    mixed = [["0", "0.5", "0.5", "0.2", "0.2"], ["0", "0.1", "0.1", "0.5", "0.1", "0.3", "0.4"]]
    problems = dataset_doctor.label_problems(mixed, ["a"])
    assert len(problems) == 1
    assert "box row in a file that contains a polygon row" in problems[0]

    # ...and the polygon rows themselves are still read as polygons, not flagged as the mistake.
    assert not any("neither" in problem for problem in problems)


def test_the_doctor_and_the_loader_agree_on_what_a_row_is(tmp_path):
    """The drift guard for `row_shape`: the definition is the *loader's*, so it is re-measured here
    against `ultralytics.data.utils.verify_image_label` - the function both training and validation
    go through - rather than restated from the comment above it.

    `label_problems` reports a row exactly when the loader refuses the frame its in, and the one
    place they deliberately differ is asserted too: a mixed file is accepted by the loader (as
    garbage) and refused here.
    """
    from ultralytics.data.utils import verify_image_label

    import numpy as np
    from PIL import Image

    image = tmp_path / "frame.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype="uint8")).save(image)
    cases = {
        "box": "0 0.5 0.5 0.2 0.2\n",
        "polygon": "0 0.1 0.1 0.5 0.1 0.3 0.4\n",
        "six_fields": "0 0.1 0.1 0.5 0.1 0.5\n",
        "odd_polygon": "0 0.1 0.1 0.5 0.1 0.5 0.3\n",
        "far_coordinate": "0 0.1 0.1 1.4 0.1 0.3 0.4\n",
        "empty": "",
    }
    for name, body in cases.items():
        label = tmp_path / f"{name}.txt"
        label.write_text(body, encoding="utf-8")
        _file, _lb, _shape, _segments, _keypoints, _nm, _nf, _ne, nc, _msg = verify_image_label(
            (str(image), str(label), "", False, 7, 0, 2, False)
        )
        corrupt = nc != 0  # `nc`: the loader dropped the frame as corrupt
        rows = [line.split() for line in body.splitlines() if line.strip()]
        # An empty file is the loader's background case: no rows, nothing for the doctor to say.
        assert (dataset_doctor.label_problems(rows, ["a"]) != []) == corrupt, name
        assert (not corrupt) or name != "empty"

    # The one deliberate difference, asserted rather than described: a file that mixes the two
    # shapes loads *without* complaint - the box row is reshaped into a two-point polygon and the
    # frame trains on a box nobody drew - and it is exactly what the doctor reports.
    mixed = "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.5 0.1 0.3 0.4\n"
    (tmp_path / "mixed.txt").write_text(mixed, encoding="utf-8")
    _file, _lb, _shape, _segments, _keypoints, _nm, _nf, _ne, nc, _msg = verify_image_label(
        (str(image), str(tmp_path / "mixed.txt"), "", False, 7, 0, 2, False)
    )
    assert nc == 0
    rows = [line.split() for line in mixed.splitlines()]
    assert any("box row in a file that contains a polygon row" in p for p in dataset_doctor.label_problems(rows, ["a"]))


def test_an_absent_label_file_is_not_an_empty_one(tmp_path):
    """The distinction the whole dataset pipeline is built on: absent means nobody decided, empty
    means somebody decided there is no item here. A doctor that conflated them would pass a
    half-labelled set as clean."""
    missing = tmp_path / "gone.txt"
    rows, problems = dataset_doctor.read_label(missing)
    assert rows == [] and len(problems) == 1 and "absent is not the same as empty" in problems[0]

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    rows, problems = dataset_doctor.read_label(empty)
    assert rows == [] and problems == []

    blank = tmp_path / "blank.txt"
    blank.write_text("\n  \n", encoding="utf-8")
    assert dataset_doctor.read_label(blank) == ([], [])


# --------------------------------------------------------------------------
# The checks that decide the exit code
# --------------------------------------------------------------------------


def test_a_clean_set_passes_with_nothing_to_say(tmp_path):
    """"Clean" is the whole current shape, tags included: every frame's drawing agrees with the
    class it was staged as, which is the one check a report from an older build cannot run.
    """
    bear = next(slug for slug, name in SLUG_TO_CLASS.items() if name == V2.classes[0])
    tags = {
        f"{split}_{index:04d}.jpg": bear
        for split in ("train", "valid", "test")
        for index in range(2)
    }
    report = dataset_doctor.check_dataset(_dataset(tmp_path, tags=tags), V2)
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["splits"]["train"] == {
        "images": 2,
        "boxes": 2,
        "background": 0,
        "polygons": 0,
        "label_problems": 0,
        "problem_frames": 0,
        "orphan_labels": 0,
    }


def test_the_right_names_in_the_wrong_order_is_an_error(tmp_path):
    """The failure `check_export` cannot see: it compares class *names* by membership, and every
    label row indexes the list by position - so the same seven names in another order relabels
    every box in the set while every existing guard passes."""
    names = list(reversed(list(V2.classes)))
    report = dataset_doctor.check_dataset(_dataset(tmp_path, names=names), V2)
    assert any("wrong ORDER" in error for error in report["errors"])
    assert not any("missing" in error for error in report["errors"])


def test_a_missing_class_is_left_to_the_export_check(tmp_path):
    """Membership is `check_export`'s job, and saying it twice would print the same situation as two
    findings - the doctor adds order, not a second membership verdict."""
    report = dataset_doctor.check_dataset(_dataset(tmp_path, names=list(V2.classes)[:-1]), V2)
    assert report["errors"]
    assert not any("wrong ORDER" in error for error in report["errors"])


def test_a_label_row_that_cannot_be_read_fails_the_set(tmp_path):
    report = dataset_doctor.check_dataset(_dataset(tmp_path, rows="9 0.5 0.5 0.2 0.2\n"), V2)
    assert any("outside the" in error for error in report["errors"])
    assert report["splits"]["train"]["problem_frames"] == 2


def test_a_frame_with_no_label_file_at_all_fails_the_set(tmp_path):
    dataset = _dataset(tmp_path, labels=False)
    report = dataset_doctor.check_dataset(dataset, V2)
    assert any("absent is not the same as empty" in error for error in report["errors"])


def test_an_empty_label_is_background_and_not_a_failure(tmp_path):
    """The hard negatives are real data, and a doctor that failed on them would make the one set
    this project most needs impossible to build."""
    report = dataset_doctor.check_dataset(_dataset(tmp_path, rows=""), V2)
    assert report["errors"] == []
    assert report["splits"]["train"]["background"] == 2
    assert report["splits"]["train"]["boxes"] == 0


def test_an_orphan_label_is_a_warning_not_a_failure(tmp_path):
    dataset = _dataset(tmp_path)
    (dataset / "train" / "labels" / "gone.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    report = dataset_doctor.check_dataset(dataset, V2)
    assert report["errors"] == []
    assert any("with no image" in warning for warning in report["warnings"])
    assert report["splits"]["train"]["orphan_labels"] == 1


def test_a_split_with_no_images_is_reported_once_not_twice(tmp_path):
    """`check_export` owns this one. The doctor measuring the same emptiness a second time would
    print one situation as two findings, which is how a reader learns to skim the list."""
    report = dataset_doctor.check_dataset(
        _dataset(tmp_path, counts={"train": 2, "valid": 2, "test": 0}), V2
    )
    assert report["errors"] == ["no test/images directory in the export"]


def test_a_test_frame_duplicating_a_train_frame_is_an_error(tmp_path):
    """The number this command exists to protect: a test split containing a copy of a training
    photograph measures recall of an image the model has seen."""
    report = dataset_doctor.check_dataset(_dataset(tmp_path, duplicate_test=True), V2)
    assert report["test_duplicates"]
    assert any("near-duplicates" in error for error in report["errors"])


def test_the_duplicate_scan_can_be_skipped(tmp_path):
    """It fingerprints every train and valid frame, which is the slow part of the check - and it is
    the only part that is not a read of the set's own files."""
    report = dataset_doctor.check_dataset(_dataset(tmp_path, duplicate_test=True), V2, duplicates=False)
    assert report["errors"] == []
    assert "test_duplicates" not in report


def test_a_stale_merge_report_fails_the_set(tmp_path):
    """`accept_v2` reads the machine-only gate out of this file, so a report left over from an
    earlier build is a gate reading a different dataset - and nothing else would say so."""
    dataset = _dataset(tmp_path)
    (dataset / "merge_report.json").write_text(
        json.dumps({"splits": {"train": {"images": 9}, "valid": {"images": 2}, "test": {"images": 2}}}),
        encoding="utf-8",
    )
    report = dataset_doctor.check_dataset(dataset, V2)
    assert any("does not describe this dataset" in error for error in report["errors"])
    assert any("report says 9, disk has 2" in error for error in report["errors"])


def test_a_report_without_the_gate_field_is_a_warning(tmp_path):
    """Not a failure: the set itself is fine. It is the *acceptance* command that cannot be run
    against it, and that is worth a sentence rather than an exit code."""
    report = dataset_doctor.check_dataset(_dataset(tmp_path, merge={"splits": {"train": {"images": 2}}}), V2)
    assert report["errors"] == []
    assert any("machine_only_by_split" in warning for warning in report["warnings"])


def test_a_merge_whose_v2_side_contributed_nothing_fails(tmp_path):
    """The set's own account of itself: three splits, the right names in order, every label readable
    - because every frame is v1's. `build_dataset.py` refuses to *build* this now, so what this
    catches is a set built before it could: nothing on disk distinguishes the two sources, and the
    report is the only file that says what the set's name promises.
    """
    dataset = _dataset(tmp_path)
    body = json.loads((dataset / "merge_report.json").read_text(encoding="utf-8"))
    body["sources"] = {
        "v1": {"images": {"train": 2, "valid": 2, "test": 2}},
        "v2": {"images": {}},
    }
    (dataset / "merge_report.json").write_text(json.dumps(body), encoding="utf-8")

    report = dataset_doctor.check_dataset(dataset, V2)
    assert any("v2 side contributed no frames at all" in error for error in report["errors"])
    assert any("v1's frames under the merged name" in error for error in report["errors"])
    assert report["merge_report"]["v2_images"] == {}  # the evidence rides along in `--json`


def test_any_v2_contribution_or_any_silence_about_sources_is_not_a_finding(tmp_path):
    """Both halves of the deliberate narrowness: this is a *claim* about an empty v2 side, so a
    report counting one v2 frame passes, and a report with no `sources` block at all - another
    tool's, or one written before the builder recorded sides - is silence rather than a claim. The
    two sides add up to what `test` holds, because a report that names sides has to add up
    (`side_count_problems`).
    """
    dataset = _dataset(tmp_path)
    body = json.loads((dataset / "merge_report.json").read_text(encoding="utf-8"))
    body["sources"] = {
        "v1": {"images": {"train": 2, "valid": 2, "test": 1}},
        "v2": {"images": {"test": 1}},
    }
    (dataset / "merge_report.json").write_text(json.dumps(body), encoding="utf-8")
    report = dataset_doctor.check_dataset(dataset, V2)
    assert report["errors"] == []
    assert report["merge_report"]["v2_images"] == {"test": 1}

    silent = dataset_doctor.check_dataset(_dataset(tmp_path / "other"), V2)
    assert silent["errors"] == []
    assert silent["merge_report"]["v2_images"] is None


def test_a_report_whose_sides_do_not_add_up_is_the_stale_report_error(tmp_path):
    """The shape of the real disagreement: a v2 side counted where the frames were *staged* rather
    than where they survived, so the tag map and the mix describe fewer frames than the side claims.
    The verdict is the existing one - a report that does not describe this set - because the fix is
    the same: rebuild it. The sentence's numbers are pinned here rather than by a unit test on
    `side_count_problems`, which would restate this. Silence when no side names a count is already
    pinned by every fixture whose report carries no `sources` block.
    """
    dataset = _dataset(tmp_path)
    body = json.loads((dataset / "merge_report.json").read_text(encoding="utf-8"))
    body["sources"] = {"v1": {"images": {"train": 2, "valid": 2}}, "v2": {"images": {"test": 83}}}
    (dataset / "merge_report.json").write_text(json.dumps(body), encoding="utf-8")

    report = dataset_doctor.check_dataset(dataset, V2)
    error = next(error for error in report["errors"] if "does not describe this dataset" in error)
    assert "test: its sides account for 83 frame(s), disk has 2" in error
    assert report["merge_report"]["v2_images"] == {"test": 83}  # the evidence rides along


def test_a_corrupt_merge_report_is_a_warning_rather_than_a_crash(tmp_path):
    dataset = _dataset(tmp_path, merge=None)
    (dataset / "merge_report.json").write_text("{not json", encoding="utf-8")
    report = dataset_doctor.check_dataset(dataset, V2)
    assert report["errors"] == []
    assert any("unreadable" in warning for warning in report["warnings"])


def test_a_set_that_did_not_come_from_the_builder_says_so(tmp_path):
    report = dataset_doctor.check_dataset(_dataset(tmp_path, merge=None), V2)
    assert report["errors"] == []
    assert any("did not come from build_dataset.py" in warning for warning in report["warnings"])


def test_a_dataset_that_is_not_there_is_one_error_not_a_traceback(tmp_path):
    report = dataset_doctor.check_dataset(tmp_path / "nope", V2)
    assert any("no export at" in error for error in report["errors"])


# --------------------------------------------------------------------------
# The tags: what each frame was staged as, against what its labels draw
# --------------------------------------------------------------------------


def test_the_tag_rule_leaves_the_other_checks_their_findings():
    """Three deliberate silences, pinned as rules rather than through a dataset: a hard negative
    (`label_classes.tag_mismatch`'s exemption, so no two checks can disagree), a class index outside the
    list (already `label_problems`' sentence - restating it as a wrong product is how a reader
    learns to skim), and a frame with nothing drawn (a background decision, not a mistake).
    """
    names = ["a", "b"]
    assert dataset_doctor.tag_problems({"x.jpg": "negative"}, {"x.jpg": {0}}, names) == ([], [])
    assert dataset_doctor.tag_problems({"x.jpg": "milo"}, {"x.jpg": {9}}, names) == ([], [])
    assert dataset_doctor.tag_problems({"x.jpg": "milo"}, {"x.jpg": set()}, names) == ([], [])
    assert dataset_doctor.tag_problems({"x.jpg": "milo"}, {"x.jpg": {1}}, names) == (
        [("x.jpg", SLUG_TO_CLASS["milo"], ["b"])],
        [],
    )


def test_a_frame_drawn_under_another_class_fails_the_set(tmp_path):
    """The failure with no other symptom: readable rows, in-range indices, valid boxes - and a
    model that calls one product by another's name. The tag is the only fact outside a label file
    that says which product the frame *is*, and it travels in the merge report because the built
    set holds no manifest.
    """
    report = dataset_doctor.check_dataset(_dataset(tmp_path, tags={"train_0000.jpg": "milo"}), V2)

    assert any(
        f"train/train_0000.jpg: staged as {SLUG_TO_CLASS['milo']}, drawn as {V2.classes[0]}"
        in error
        for error in report["errors"]
    )
    assert report["tags"]["mismatched"] == ["train/train_0000.jpg"]
    assert report["splits"]["train"]["images"] == 2  # a finding about the drawing, not the set


def test_frames_drawn_under_their_own_class_pass_and_the_ok_line_says_so(tmp_path):
    """Agreement is checked, not assumed: the report carries the count and the `[ok]` line names
    the fact - a pass that does not say what it looked at is one an operator cannot audit.
    """
    bear = next(slug for slug, name in SLUG_TO_CLASS.items() if name == V2.classes[0])
    tags = {
        f"{split}_{index:04d}.jpg": bear
        for split in ("train", "valid", "test")
        for index in range(2)
    }
    report = dataset_doctor.check_dataset(_dataset(tmp_path, tags=tags), V2)

    assert report["errors"] == []
    assert report["tags"] == {"tagged": 6, "mismatched": [], "unknown_slugs": []}
    assert any(
        "staged frame's labels naming its own class" in line
        for line in dataset_doctor.report_lines(report)
    )


def test_a_background_frame_and_a_hard_negative_keep_their_tags(tmp_path):
    """The two states that look like mismatches and are not: a background frame keeps the tag it
    was staged under (nobody drew anything - that is a decision), and a box on a hard negative is
    "product plus clutter" rather than a mistake. Both rules are `label_classes.tag_mismatch`'s,
    because a second opinion about what "drawn wrong" means is how one of the two ends up wrong.
    """
    bear = next(slug for slug, name in SLUG_TO_CLASS.items() if name == V2.classes[0])
    dataset = _dataset(tmp_path, tags={"train_0000.jpg": bear, "train_0001.jpg": "negative"})
    # An empty label file is the null annotation: decided, and nothing drawn.
    (dataset / "train" / "labels" / "train_0000.txt").write_text("", encoding="utf-8")
    report = dataset_doctor.check_dataset(dataset, V2)

    assert report["errors"] == []
    assert report["tags"]["tagged"] == 2


def test_the_tag_rule_is_decided_once_and_the_awkward_tags_are_its_own():
    """`label_classes.tag_mismatch` is where "drawn wrong" is decided. The two cases here are the
    ones no caller's own tests reach: the shape a *remote* source hands it (a class summary with
    `None`/empty entries, which name no product), and several wrong classes coming back sorted
    rather than in the order the rows happened to be in. The exemption cases - hard negative,
    nothing drawn - are pinned through `tag_problems` below, where they are read.
    """
    assert label_classes.tag_mismatch("milo", [None, ""]) == ()
    assert label_classes.tag_mismatch(
        "milo", ["555 sardines 155grams", "safeguard_pure_white_60g"]
    ) == ("555 sardines 155grams", "safeguard_pure_white_60g")


def test_a_tag_for_a_frame_the_set_does_not_hold_is_a_stale_report(tmp_path):
    """The builder records tags only for the frames it writes, so a tag naming a frame that is
    not here means the report describes a different build - the same finding as a stale split
    count, said in the same sentence.
    """
    report = dataset_doctor.check_dataset(
        _dataset(tmp_path, tags={"ghost_0001.jpg": "bear-brand-milk"}), V2
    )

    error = next(error for error in report["errors"] if "does not describe this dataset" in error)
    assert "its tags name ghost_0001.jpg, which this set does not hold" in error


def test_an_unknown_tag_slug_is_a_warning_naming_the_gap(tmp_path):
    """A hand-edited manifest can stage a frame under a folder no class maps to. The frame is not
    wrong - what is missing is an expectation to judge it against, and that has to be a sentence
    rather than a silent pass over exactly the frames the check could not read.
    """
    report = dataset_doctor.check_dataset(
        _dataset(tmp_path, tags={"train_0000.jpg": "grammar-crackers"}), V2
    )

    assert report["errors"] == []
    assert any("not classes this app knows" in warning for warning in report["warnings"])
    assert report["tags"]["unknown_slugs"] == ["grammar-crackers"]


def test_a_report_without_tags_says_the_check_did_not_run(tmp_path):
    """v1 carries no tags and a report from a build older than the tag copy carries none either,
    so the check cannot run over the whole frame set - and that must read as "unverified", never
    as a pass. The `[ok]` line drops the claim it cannot make.
    """
    report = dataset_doctor.check_dataset(_dataset(tmp_path), V2)

    assert report["errors"] == []
    assert any("no `tags` block" in warning for warning in report["warnings"])
    assert "naming its own class" not in "\n".join(dataset_doctor.report_lines(report))


# --------------------------------------------------------------------------
# The distance mix: what the per-distance grid will be able to say
# --------------------------------------------------------------------------


def test_the_distance_rule_says_which_gaps_need_which_fix():
    """Two states with two different actions: a distance trained on and never measured is the
    split plan's problem, and a distance in no split at all is the camera's. `unknown` (the hard
    negatives, staged outside the distance cells) is neither - nothing was planned at it.
    """
    gap = dataset_doctor.distance_problems({"train": {"close": 10, "mid": 2}, "test": {"close": 3}})

    assert any("`mid` is trained on (2 frame(s)" in line for line in gap)
    assert any("no v2 frame at" in line and "`far`" in line for line in gap)
    assert "no v2 frame at" not in next(line for line in gap if "trained on" in line)
    assert dataset_doctor.distance_problems(
        {
            "train": {"close": 1, "unknown": 9, "mid": 1, "far": 1},
            "test": {"close": 1, "unknown": 4, "mid": 1, "far": 1},
        }
    ) == []
    # The hard negatives are `unknown`, and a set whose only `unknown` frames are untrained-on
    # still gets the plan's gaps named - the exemption is for the pseudo-distance, not the check.
    assert any(
        "no v2 frame at" in line
        for line in dataset_doctor.distance_problems(
            {"train": {"close": 1, "unknown": 9}, "test": {"close": 1}}
        )
    )
    assert dataset_doctor.distance_problems(None) == []


def test_the_mix_is_printed_and_a_complete_one_earns_no_warning(tmp_path):
    """The readout is how a set that covers the plan says so without a sentence, and a set whose
    three distances are all measured gets no distance warning at all - the warnings are the gaps.
    """
    merge = {
        "generation": V2.name,
        "splits": {split: {"images": 2} for split in ("train", "valid", "test")},
        "machine_only_by_split": {},
        "sources": {
            "v2": {
                # Every frame is v2's here; one per split carries no distance, so `images` (the
                # set) is the fixture's 2 while the mix holds one fewer.
                "images": {"train": 2, "valid": 2, "test": 2},
                "distances": {
                    "train": {"close": 1, "mid": 1},
                    "valid": {"far": 1},
                    "test": {"close": 1, "mid": 1, "far": 1},
                },
            }
        },
    }
    report = dataset_doctor.check_dataset(_dataset(tmp_path, merge=merge), V2)

    assert report["errors"] == []
    assert not any("is trained on" in w or "no v2 frame at" in w for w in report["warnings"])
    printed = "\n".join(dataset_doctor.report_lines(report))
    assert "v2 distances: train close 1, mid 1; valid far 1; test close 1, mid 1, far 1" in printed


def test_the_mix_line_sorts_a_distance_the_plan_does_not_follow_last(tmp_path):
    """`plan_order` is the one ranking both the mix line and the distance findings read, and a
    name outside the plan is not an error to either: the hard negatives are recorded as `unknown`,
    so the line stays printable and in grid order rather than raising or scattering it among the
    planned columns by name."""
    merge = {
        "generation": V2.name,
        "splits": {split: {"images": 2} for split in ("train", "valid", "test")},
        "machine_only_by_split": {},
        "sources": {
            "v2": {
                "images": {"train": 2, "valid": 2, "test": 2},
                "distances": {"train": {"unknown": 1, "close": 1}},
            }
        },
    }
    report = dataset_doctor.check_dataset(_dataset(tmp_path, merge=merge), V2)

    printed = "\n".join(dataset_doctor.report_lines(report))
    assert "v2 distances: train close 1, unknown 1" in printed


def test_a_distance_the_test_split_cannot_measure_is_named(tmp_path):
    """The grid cannot fill a cell no test frame reaches: the model is asked to learn `mid` and
    never measured on it, so the split plan - not the camera - is what has to change.
    """
    merge = {
        "generation": V2.name,
        "splits": {split: {"images": 2} for split in ("train", "valid", "test")},
        "machine_only_by_split": {},
        "sources": {
            "v2": {
                # The mix is what this rule reads, not the set's arithmetic.
                "images": {"train": 2, "valid": 2, "test": 2},
                "distances": {
                    "train": {"close": 2, "mid": 4},
                    "valid": {"close": 1},
                    "test": {"close": 2},
                },
            }
        },
    }
    report = dataset_doctor.check_dataset(_dataset(tmp_path, merge=merge), V2)

    assert report["errors"] == []
    assert any("`mid` is trained on (4 frame(s)" in warning for warning in report["warnings"])
    assert any("no v2 frame at `far`" in warning for warning in report["warnings"])


def test_a_v2_side_without_a_recorded_mix_says_it_was_not_read(tmp_path):
    """A builder's report from before the merge recorded the axis, or a hand-written one: the
    distance check cannot run over it, and silence would read as coverage. A report with no
    `sources` block at all is a different state - not the builder's account, so not this file's
    claim to check (the same narrowness the v2-contribution rule keeps).
    """
    merge = {
        "generation": V2.name,
        "splits": {split: {"images": 2} for split in ("train", "valid", "test")},
        "machine_only_by_split": {},
        "sources": {"v2": {"images": {"train": 2, "valid": 2, "test": 2}}},
    }
    report = dataset_doctor.check_dataset(_dataset(tmp_path, merge=merge), V2)
    assert any("no distance mix" in warning for warning in report["warnings"])

    silent = dataset_doctor.check_dataset(_dataset(tmp_path / "other"), V2)
    assert not any("distance" in warning for warning in silent["warnings"])


# --------------------------------------------------------------------------
# The command: exit code, and a JSON readout that is actually JSON
# --------------------------------------------------------------------------


def test_the_exit_code_is_the_report(tmp_path, capsys):
    assert dataset_doctor.main(["--dataset", str(_dataset(tmp_path / "ok")), "--no-duplicates"]) == 0
    assert "[ok]" in capsys.readouterr().out

    broken = _dataset(tmp_path / "bad", rows="9 0.5 0.5 0.2 0.2\n")
    assert dataset_doctor.main(["--dataset", str(broken), "--no-duplicates"]) == 2
    assert "[FAIL]" in capsys.readouterr().out


def test_json_mode_leaves_stdout_carrying_nothing_but_json(tmp_path, capsys):
    """A caller piping `--json` into a parser gets a parser error if any human line shares stdout -
    and the check's own measurements print as it goes."""
    dataset = _dataset(tmp_path)
    code = dataset_doctor.main(["--dataset", str(dataset), "--json", "--no-duplicates"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert code == 0
    assert report["dataset"] == str(dataset)
    assert report["splits"]["train"]["images"] == 2
    assert "dataset:" in captured.err  # the human lines, on the other stream


def test_a_missing_dataset_directory_exits_with_a_sentence(tmp_path):
    with pytest.raises(SystemExit) as caught:
        dataset_doctor.main(["--dataset", str(tmp_path / "nope")])
    assert "no dataset at" in str(caught.value)


def test_an_unknown_generation_is_refused_by_name(tmp_path):
    with pytest.raises(SystemExit):
        dataset_doctor.main(["--dataset", str(_dataset(tmp_path)), "--generation", "v9"])


# --------------------------------------------------------------------------
# Which order the labels index, and which generation this set is
# --------------------------------------------------------------------------


def test_a_v1_order_is_named_as_v1s_and_not_as_a_permutation(tmp_path):
    """The finding has to say *which* order the rows are in, because the two answers need different
    actions: a set that is genuinely v1's is not broken - it is being judged against v2's
    expectation - and relabelling it to satisfy this check would corrupt every box in it.

    v1 and v2 declare the same seven names and differ only in order, so this is the case both
    generations make permanent, not a hypothetical one.
    """
    report = dataset_doctor.check_dataset(_dataset(tmp_path, names=list(generations.V1.classes)), V2)

    assert report["order_generation"] == "v1"
    assert report["names"] == list(generations.V1.classes)
    assert report["expected_names"] == list(V2.classes)
    error = next(error for error in report["errors"] if "v1's order" in error)
    assert "--generation v1" in error  # the fix, not a rebuild
    assert "relabelling it" in error


def test_a_record_that_disagrees_with_the_labels_is_said_out_loud(tmp_path):
    """The one disagreement a reader cannot see from the check itself: the set's *own* files say v2
    while its rows index v1's order. Both are read here, so naming the file is what turns "wrong
    order" into "this set contradicts itself" - the merge report in one case, the yaml beside the
    names in the other (which is the shape a hand-sorted `names` list takes).
    """
    merge = {
        "generation": "v2",
        "splits": {"train": {"images": 2}, "valid": {"images": 2}, "test": {"images": 2}},
        "machine_only_by_split": {},
    }
    report = dataset_doctor.check_dataset(
        _dataset(tmp_path / "report", names=list(generations.V1.classes), merge=merge), V2
    )
    error = next(error for error in report["errors"] if "v1's order" in error)
    assert "v2 in its merge_report.json" in error

    report = dataset_doctor.check_dataset(
        _dataset(tmp_path / "yaml", names=list(generations.V1.classes), yaml_generation="v2", merge=None),
        V2,
    )
    error = next(error for error in report["errors"] if "v1's order" in error)
    assert "v2 in its data.yaml" in error

    # Nothing recorded and the rows are another generation's: no clause, because there is nothing to
    # quote - a missing record is a warning elsewhere, not a contradiction here.
    quiet = dataset_doctor.order_problem(list(generations.V1.classes), V2)
    assert "the set records" not in quiet and "v1's order" in quiet


def test_a_set_whose_names_are_not_any_generations_order_is_not_called_one(tmp_path):
    """The negative the lookup has to get right: a permutation that is nobody's list is a data
    problem, not a generation mix-up, and offering `--generation` for it would send the operator to
    a flag that cannot help. `order_generation` is how a caller tells the two apart."""
    names = [*list(V2.classes)[:-1], "Palmolive Naturals Bar Soap 85g"]  # one name swapped out
    report = dataset_doctor.check_dataset(_dataset(tmp_path, names=names), V2)

    assert report["order_generation"] is None
    # Membership is `check_export`'s finding, and the order check stays quiet about it.
    assert not any("wrong ORDER" in error for error in report["errors"])


def test_a_permutation_of_the_right_names_is_still_a_wrong_order(tmp_path):
    """The fallback sentence, kept for the case that is actually a corrupted list rather than a
    generation mix-up - and asserted together with the named one, so the two cannot collapse into
    the same message."""
    names = list(reversed(list(V2.classes)))
    report = dataset_doctor.check_dataset(_dataset(tmp_path, names=names), V2)

    assert report["order_generation"] is None
    assert any("wrong ORDER" in error for error in report["errors"])


def test_the_report_prints_the_ordered_list_and_names_the_generation(tmp_path):
    """What "have the doctor say which order its labels index" means in print: a numbered list to
    compare against, the generation it was judged against, and the same naming in the `[ok]` line.
    `7 declared` cannot be read against anything."""
    report = dataset_doctor.check_dataset(_dataset(tmp_path), V2)
    lines = dataset_doctor.report_lines(report)

    assert lines[0] == "classes: 7 declared, in v2's order:"
    assert [line.strip().split("  ", 1)[-1] for line in lines[1:8]] == list(V2.classes)
    assert any("[ok] the set is internally consistent: names in v2's order" in line for line in lines)

    wrong = dict(report, generation="v2", names=list(generations.V1.classes), order_generation="v1")
    assert dataset_doctor.class_lines(wrong)[0] == (
        "classes: 7 declared, in v1's order - this check judged them against v2's:"
    )
    permutation = dataset_doctor.class_lines(
        dict(report, names=list(reversed(list(V2.classes))), order_generation=None)
    )
    assert permutation[0] == (
        "classes: 7 declared, in an order no generation declares (judged against v2's):"
    )
    assert dataset_doctor.class_lines(dict(report, names=[], order_generation=None)) == [
        "classes: 7 expected, none declared"
    ]


def test_the_generation_is_resolved_from_the_set_rather_than_the_caller(tmp_path):
    """`--generation auto`, in its cases, in order of directness: the statement beside the names,
    the build's own record, what the class list itself says, and the default *said out loud* rather
    than assumed.

    Two files in one dataset naming two generations is the case a header must not quietly pick a
    winner from, so it is asserted too: the yaml wins (it is the file the label rows index), and the
    other claim is quoted in the answer.
    """
    stated, how = dataset_doctor.resolve_generation(
        _dataset(tmp_path / "stated", yaml_generation="v1", merge=None)
    )
    assert (stated.name, how) == ("v1", "from its data.yaml")

    disputed, how = dataset_doctor.resolve_generation(
        _dataset(tmp_path / "disputed", yaml_generation="v1")  # merge_report.json says v2
    )
    assert disputed.name == "v1" and "merge_report.json says v2" in how

    built, how = dataset_doctor.resolve_generation(_dataset(tmp_path / "built"))
    assert (built.name, how) == ("v2", "from merge_report.json")

    # A set that did not come from the builder: its own declared order is still a statement.
    declared, how = dataset_doctor.resolve_generation(
        _dataset(tmp_path / "ingested", names=list(generations.V1.classes), merge=None)
    )
    assert (declared.name, how) == ("v1", "from the order its class list declares")

    # A report naming a generation overrides the class list - a build states the order it wrote.
    recorded, how = dataset_doctor.resolve_generation(
        _dataset(
            tmp_path / "wrong",
            names=list(generations.V1.classes),
            merge={"generation": "v2", "splits": {"train": {"images": 2}}},
        )
    )
    assert (recorded.name, how) == ("v2", "from merge_report.json")

    # A generation name nothing recognises is not a fact about the labels: it falls through to the
    # order, and a value this app has no roster for never gets quoted as one.
    unknown, how = dataset_doctor.resolve_generation(
        _dataset(tmp_path / "later", yaml_generation="v9", merge=None)
    )
    assert (unknown.name, how) == ("v2", "from the order its class list declares")
    assert dataset_doctor.recorded_generation(
        _dataset(tmp_path / "junk", merge={"generation": 3, "splits": {}})
    ) is None

    # Nothing recognisable: the default, and named as a default.
    fallback, how = dataset_doctor.resolve_generation(
        _dataset(tmp_path / "odd", names=list(reversed(list(V2.classes))), merge=None)
    )
    assert fallback.name == generations.DEFAULT.name
    assert "default" in how


def test_a_set_that_records_its_generation_is_never_judged_as_the_other_one(tmp_path):
    """The two halves of "which order do these labels index": a set that says it is v1 *and* indexes
    v1's list gets no order finding at all, and one whose yaml contradicts its own rows is caught by
    the order check rather than believed - the recorded generation is a statement about the labels,
    not a substitute for reading them."""
    clean = dataset_doctor.check_dataset(
        _dataset(tmp_path / "says-v1", names=list(generations.V1.classes), yaml_generation="v1", merge=None),
        generations.V1,
    )
    assert clean["errors"] == []
    assert clean["order_generation"] == "v1"

    contradicted = dataset_doctor.check_dataset(
        _dataset(tmp_path / "says-v2", names=list(generations.V1.classes), yaml_generation="v2", merge=None),
        V2,
    )
    assert contradicted["order_generation"] == "v1"
    assert any("v1's order" in error for error in contradicted["errors"])


def test_the_command_judges_a_v1_set_as_v1_without_being_told(tmp_path, capsys):
    """The omission this default exists to make impossible: judging v1's set against v2's order
    would fail a set that is correct, and the exit code would send the operator to fix labels that
    have nothing wrong with them. The header says how the generation was decided, and `--json`
    carries the same fact for a script."""
    dataset = _dataset(tmp_path, names=list(generations.V1.classes), merge=None)
    code = dataset_doctor.main(["--dataset", str(dataset), "--json", "--no-duplicates"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert code == 0
    assert report["generation"] == "v1"
    assert report["generation_decided_by"] == "from the order its class list declares"
    # What the block itself says, in human mode: the numbered list and the order it is in.
    dataset_doctor.main(["--dataset", str(dataset), "--no-duplicates"])
    out = capsys.readouterr().out
    assert "classes: 7 declared, in v1's order:" in out
    assert "   1  " + generations.V1.classes[0] in out
    assert report["order_generation"] == "v1"
    assert "generation v1" in captured.err


def test_an_explicit_generation_is_still_honoured_and_marked_as_asked(tmp_path, capsys):
    """`auto` is a default, not a veto: naming a generation is how a deliberate what-if is asked
    ("would v2's order accept this set?"), and the report says the answer was asked for."""
    dataset = _dataset(tmp_path, names=list(generations.V1.classes), merge=None)
    code = dataset_doctor.main(
        ["--dataset", str(dataset), "--generation", "v2", "--json", "--no-duplicates"]
    )
    report = json.loads(capsys.readouterr().out)

    assert code == 2
    assert report["generation"] == "v2"
    assert report["generation_decided_by"] == "as asked"
    assert report["order_generation"] == "v1"
