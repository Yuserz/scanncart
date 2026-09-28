"""Tests for the distance breakdown: the words, the per-distance passes, and the grid.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import json
from pathlib import Path
import clean_v2
import generations
import label_classes
import train_model
import workspace

# Rebuilt from the modules that own each fact, not copied by value.
V2 = generations.V2
VAL_NAME = train_model.val_name(V2)

from tests.dataset_tool_helpers import (
    _FakeBox,
    _FakeMetrics,
    _NAMES,
    _export_with_names,
    _manifest_with_distances,
    _report_with_distances,
)


# --- distance words in a class name: the 24-output hazard ---------------------------------


def test_a_distance_in_a_class_name_is_recognised():
    """Distance is a tag, not a category, so a class named after one splits a product three ways
    and the trained head gets an output per product-and-distance. Nothing in a class *name*
    records that, which is why it is checked rather than agreed by convention.
    """
    for name in (
        "palmolive close",
        "Palmolive Naturals Bar Soap 85g far",
        "safeguard-mid",
        "milo closeup",
        "bear_brand_MIDDLE",
        "555 sardines near",
        "palmolive distance",
    ):
        assert label_classes.distance_tokens_in(name), name


def test_a_legitimate_class_name_is_never_flagged():
    """Whole tokens only, because a false alarm here is a hard failure on a correct project -
    and the roster is full of words that contain a distance word without being one."""
    for name in (
        "Bear Brand Fortified Powdered Milk 33g",
        "Milo Chocolate Drink 22g Sachet",
        "Palmolive Naturals Bar Soap 85g",
        "century_tuna_flakes_in_oil_155_grams",
        "lucky_me_pancit_canton_calamansi_flavor",
        "silver_swan_sukang_puti_200ML",
        "safeguard_pure_white_60g",
        "Farmer's Choice Fresh Milk",  # tokenises to `farmer`, which is not `far`
        "Midfield Brand Coffee",  # `midfield`, not `mid`
        "",
    ):
        assert label_classes.distance_tokens_in(name) == [], name
    assert all(label_classes.distance_tokens_in(n) == [] for n in label_classes.SLUG_TO_CLASS.values())


def test_the_distance_words_cover_every_spelling_the_folders_accept():
    """The checklist can spell a distance folder five ways, and each one is a way for the check
    to be blind to a class named after it. Same shape as the class-name drift guard: two places
    that must agree, kept honest by a test rather than by memory.
    """
    for spelling in clean_v2.DISTANCE_MAP:
        # Asserted through the predicate rather than on the token set: a spelling can be two
        # tokens (`CLOSE-UP`), and what has to hold is that a class named that way gets flagged -
        # not that every token of it is a keyword.
        assert label_classes.distance_tokens_in(spelling), (
            f"a class named after the folder spelling {spelling!r} would not be flagged"
        )
    # And the plain words themselves, which is how the tools spell them everywhere but in a
    # folder name.
    for word in ("close", "mid", "far"):
        assert label_classes.distance_tokens_in(f"palmolive {word}") == [word]


def test_an_export_whose_classes_carry_distances_says_which_and_why(tmp_path):
    """The symptom does not point at the cause: 24 classes read as a project-id mix-up, and the
    operator goes to check the wrong thing before the wrong fix (a regenerate, not an edit).
    """
    names = sorted(label_classes.SLUG_TO_CLASS.values()) + [
        "Palmolive Naturals Bar Soap 85g close",
        "Palmolive Naturals Bar Soap 85g far",
    ]
    root = _export_with_names(tmp_path / "export", {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["e.jpg"]})
    import yaml

    body = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    body["names"] = names
    body["nc"] = len(names)
    (root / "data.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")

    _splits, problems = train_model.check_export(root)

    assert any("not v2 classes" in p for p in problems)
    told = next(p for p in problems if "carry a *distance*" in p)
    assert "'Palmolive Naturals Bar Soap 85g close'" in told
    assert "'Palmolive Naturals Bar Soap 85g far'" in told
    assert "one output per product-and-distance" in told


def test_an_export_with_extra_classes_that_are_not_distances_says_only_that(tmp_path):
    """The distance sentence must not appear for the other causes - a project id mix-up, or a
    roster that grew - or it becomes advice that is wrong half the time.
    """
    names = sorted(label_classes.SLUG_TO_CLASS.values()) + ["some-other-project-thing"]
    root = _export_with_names(tmp_path / "export", {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["e.jpg"]})
    import yaml

    body = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    body["names"] = names
    body["nc"] = len(names)
    (root / "data.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")

    _splits, problems = train_model.check_export(root)

    assert any("not v2 classes" in p for p in problems)
    assert not any("carry a *distance*" in p for p in problems)


def test_sanity_blocks_a_class_list_that_has_a_distance_in_it():
    """The one class-list problem further labeling makes worse. `sanity` is what the checklist
    gates a shoot on, so this is where it can be caught before a session is shot and before a
    version number is spent - the two points the other call sites are too late for.

    Blocking rather than a warning is the whole point: every other class-list finding is work
    not done yet, while this one is work done in the wrong shape.
    """
    rows = clean_v2.class_list_rows(
        {
            "555 sardines 155grams close": 1,
            "555 sardines 155grams mid": 2,
            "555 sardines 155grams far": 3,
            "Milo Chocolate Drink 22g Sachet": 4,
        }
    )

    tainted = [(level, title, detail) for level, title, detail in rows if "carry a distance" in title]
    assert len(tainted) == 1
    level, title, detail = tainted[0]
    assert level == "fail"
    # Names *which*, because a long list is not readable by eye at this point.
    assert "'555 sardines 155grams mid'" in title
    # And names the two halves of the fix, which are different actions: the class list is not
    # enough on its own (the boxes have to be moved), and the version cannot be edited.
    assert "MOVE" in detail and "regenerate" in detail
    assert "21 instead of 7" in detail


def test_sanity_does_not_flag_a_correct_class_list():
    """The false alarm this must never raise: the real roster plus Palmolive is exactly the
    list a correct project has - and every one of these seven is a name v1's own project already
    declares, which is the property the drop had to preserve.
    """
    roster = list(clean_v2.V1_CLASSES)
    rows = clean_v2.class_list_rows({name: i for i, name in enumerate(roster)})

    assert [level for level, _title, _detail in rows] == ["ok"]
    assert "7 class(es) defined" in rows[0][2]
    assert not any("carry a distance" in title for _l, title, _d in rows)


def test_sanity_still_fails_a_project_with_no_class_list():
    """Preserved behaviour, and it stays a single row: with no classes there is nothing to
    inspect for distances, so a second row would be noise on an empty project.
    """
    rows = clean_v2.class_list_rows({})

    assert [level for level, _title, _detail in rows] == ["fail"]
    assert rows[0][1] == "no classes defined yet"
    assert "Lock Classes BEFORE labeling" in rows[0][2]


# --- recall by distance: the axis the per-class number averages away ----------------------


def test_the_distance_map_reads_the_manifest_and_ignores_what_it_cannot_use(tmp_path):
    """Distance lives on the image as a *tag*, and a YOLO export drops tags - so the manifest
    is the only place the two are still joined, and this is the read that joins them."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            [
                {"new_name": "a.jpg", "distance": "close"},
                {"new_name": "b.jpg", "distance": "far"},
                {"new_name": "c.jpg"},  # no distance at all
                {"new_name": "d.jpg", "distance": "sideways"},  # not one of ours
                {"new_name": "", "distance": "mid"},  # nothing to key on
                "not a dict",
                {"new_name": "e.jpg", "distance": "mid"},
            ]
        ),
        encoding="utf-8",
    )
    assert train_model.distance_map(path) == {"a.jpg": "close", "b.jpg": "far", "e.jpg": "mid"}


def test_the_sets_own_report_answers_for_distances_before_the_staged_manifest(tmp_path):
    """`--val` splits the split it is measuring, so the distances have to travel with the set: the
    `merge_report.json` was written by the build that wrote the frames, while the staged manifest
    describes the directory the set was built *from* - elsewhere the moment the set is the artifact
    under test. Both sources go through the same validation, so neither can admit a distance the
    plan does not follow; and `--manifest` named by hand still answers first, because naming a file
    is a decision rather than a default.
    """
    dataset = _export_with_names(
        tmp_path / "set",
        {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg", "edge.jpg"]},
    )
    _report_with_distances(
        dataset, [("near.jpg", "close"), ("edge.jpg", "far"), ("junk.jpg", "sideways")]
    )
    manifest = _manifest_with_distances(tmp_path, [("near.jpg", "mid")])

    # The set's own account wins, and a distance the plan does not follow is not a distance.
    assert train_model.dataset_distances(dataset, manifest) == (
        {"near.jpg": "close", "edge.jpg": "far"},
        "from the set's own merge_report.json",
    )

    # An explicit --manifest is a decision: it answers instead of the report.
    assert train_model.dataset_distances(dataset, manifest, explicit=True) == (
        {"near.jpg": "mid"},
        f"from the manifest {manifest}",
    )
    # ...and an explicit one that has nothing falls through to the report rather than killing the
    # grid, because the point of the reading is that it happens at all.
    assert train_model.dataset_distances(dataset, tmp_path / "elsewhere.json", explicit=True)[1] == (
        "from the set's own merge_report.json"
    )

    # No report (an older build, or a directory that is not a built set): the manifest answers.
    bare = _export_with_names(
        tmp_path / "bare", {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg"]}
    )
    assert train_model.dataset_distances(bare, manifest) == (
        {"near.jpg": "mid"},
        f"from the manifest {manifest}",
    )

    # Neither: an empty answer and no source sentence, which is the caller's cue to report the
    # breakdown as skipped rather than clean.
    assert train_model.dataset_distances(bare, tmp_path / "nowhere.json") == ({}, "")
    assert train_model.dataset_distances(bare, None) == ({}, "")


def test_a_missing_or_corrupt_distance_record_is_no_distances_rather_than_a_failure(tmp_path):
    """A run that cannot find distances must lose the breakdown, not the validation: these files
    are written by separate programs, and the acceptance number does not depend on them. One
    reader answers for both shapes a record takes, so missing, corrupt and unrecognised are one
    answer rather than three."""
    assert train_model.distance_map(tmp_path / "nope.json") == {}
    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")
    assert train_model.distance_map(tmp_path / "corrupt.json") == {}
    (tmp_path / "shape.json").write_text('{"a": 1}', encoding="utf-8")
    assert train_model.distance_map(tmp_path / "shape.json") == {}
    # The set's own report is the merged reader's *other shape*, not a second reader: its
    # `distances` object goes through this same call. Which file `dataset_distances` asks first
    # is the precedence test above.
    report = tmp_path / "merge_report.json"
    report.write_text(json.dumps({"distances": {"a.jpg": "far"}}), encoding="utf-8")
    assert train_model.distance_map(report) == {"a.jpg": "far"}


def test_an_image_the_manifest_does_not_know_is_reported_rather_than_dropped(tmp_path):
    """A subset does not have to be complete, but it has to be honest about it: an image in
    the split and not in any distance is in the *overall* number, so a breakdown that quietly
    omitted it would not add up to the number printed above it."""
    images = tmp_path / "test" / "images"
    images.mkdir(parents=True)
    for name in ("known.jpg", "stranger.jpg"):
        (images / name).write_bytes(b"x")

    found, unknown = train_model.images_by_distance(images, {"known.jpg": "close"})

    assert unknown == ["stranger.jpg"]
    assert [p.name for p in found["close"]] == ["known.jpg"]


def test_images_are_grouped_by_distance_and_the_unplaced_ones_are_counted(tmp_path):
    """The unplaced ones matter: they are in the overall number and in no distance row, so
    dropping them silently would make a breakdown that does not add up to the total."""
    images = tmp_path / "test" / "images"
    images.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg", "notes.txt"):
        (images / name).write_bytes(b"x")

    found, unknown = train_model.images_by_distance(
        images, {"a.jpg": "close", "b.jpg": "far", "c.jpg": "far"}
    )

    assert {d: [p.name for p in v] for d, v in found.items()} == {
        "close": ["a.jpg"],
        "mid": [],
        "far": ["b.jpg", "c.jpg"],
    }
    # Every image was placed, and the non-image was not counted as an unplaced one.
    assert unknown == []


# --- the passes themselves --------------------------------------------------------------


def _named_export(tmp_path: Path) -> Path:
    """An export whose test split holds one frame at each distance, plus its manifest."""
    return _export_with_names(
        tmp_path / "export",
        {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg", "middle.jpg", "edge.jpg"]},
    )


def _breakdown(tmp_path, distances):
    """The breakdown over the named export, with a fake `yolo` that records which pass ran."""
    root = _named_export(tmp_path)
    splits, problems = train_model.check_export(root)
    assert problems == [] and splits
    calls: list[str] = []

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            return _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.9]), counts=[10, 0, 0, 0])

    blocks, notes = train_model.distance_breakdown(
        tmp_path / "weights" / "best.pt",
        root,
        splits,
        "test",
        tmp_path / "runs",
        distances,
        yolo=_Model,
        out_dir=tmp_path / "val-by-distance",
    )
    return blocks, notes, calls


def test_the_breakdown_runs_one_pass_per_distance_with_its_own_plot_directory(tmp_path):
    """Three passes rather than a slice of one, because `DetMetrics` holds no per-image
    breakdown to cut up. Each gets its own plot directory: they share the run's project, so a
    shared name would leave the confusion matrix on disk describing `far` while the terminal
    had just printed the split's numbers."""
    blocks, notes, calls = _breakdown(
        tmp_path, {"near.jpg": "close", "middle.jpg": "mid", "edge.jpg": "far"}
    )

    assert [b["distance"] for b in blocks] == ["close", "mid", "far"]
    assert [b["images"] for b in blocks] == [1, 1, 1]
    assert calls == [
        f"{VAL_NAME}-close",
        f"{VAL_NAME}-mid",
        f"{VAL_NAME}-far",
    ]
    assert notes == []
    # The exact image set behind each number is on disk, so a surprising row can be traced
    # back to what produced it instead of being a line in a log.
    lists = sorted((tmp_path / "val-by-distance").glob("test-*.txt"))
    assert [p.name for p in lists] == ["test-close.txt", "test-far.txt", "test-mid.txt"]
    assert [p.name for p in sorted((tmp_path / "val-by-distance").glob("data-*.yaml"))] == [
        "data-test-close.yaml",
        "data-test-far.yaml",
        "data-test-mid.yaml",
    ]


def test_a_distance_with_no_images_is_unmeasured_and_says_which(tmp_path):
    """Silence here would read as "every distance was measured and none of them missed"."""
    blocks, notes, _calls = _breakdown(tmp_path, {"near.jpg": "close"})

    assert [b["distance"] for b in blocks] == ["close"]
    # The unmeasured distances first, then whatever could not be placed - the order the loop
    # walks, so a reader knows which lines are about the axis and which about the images.
    assert notes[:2] == [
        "mid: no images in the test split, so it is unmeasured",
        "far: no images in the test split, so it is unmeasured",
    ]
    assert notes[2].startswith("2 image(s) in the test split carry no distance")


def test_images_in_the_split_that_no_distance_accounts_for_are_named(tmp_path):
    """`middle.jpg` and `edge.jpg` are unplaced: they are in the overall number and in no
    distance row, which is the only way the two can look unreconcilable without a bug."""
    _blocks, notes, _calls = _breakdown(tmp_path, {"near.jpg": "close"})

    assert any(
        note.startswith("2 image(s) in the test split carry no distance") for note in notes
    ), notes


def test_no_distance_matches_at_all_reads_as_skipped_not_as_clean(tmp_path):
    """The failure mode worth naming: a breakdown that silently found nothing looks exactly
    like a breakdown where every distance passed."""
    blocks, notes, calls = _breakdown(tmp_path, {})

    assert blocks == [] and calls == []
    assert len(notes) == 1
    assert "skipped, not reported as clean" in notes[0]


def test_the_breakdown_survives_a_manifest_that_is_not_there(tmp_path):
    """The acceptance number must not depend on the dataset workspace being present."""
    blocks, notes, calls = _breakdown(tmp_path, train_model.distance_map(tmp_path / "nope.json"))

    assert blocks == [] and calls == []
    assert "skipped, not reported as clean" in notes[0]


# --- the grid: what the per-class number averages away ------------------------------------


def _distance_block(distance, rows):
    return {
        "distance": distance,
        "images": sum(n for _name, _recall, n in rows),
        "aggregates": {},
        "per_class": [{"name": n, "recall": r, "instances": i} for n, r, i in rows],
    }


def test_the_grid_puts_a_far_miss_beside_the_number_that_hides_it():
    """The feature in one assertion: `milo` clears the floor on the split as a whole while
    missing 2 of every 5 items at `far`, and the per-class table cannot say so because it
    averages the distances together. This is the dataset's own purpose - the `far` cells."""
    rows = [("bear-brand", 0.94, 40), ("milo", 0.88, 40)]
    blocks = [
        _distance_block("close", [("bear-brand", 0.97, 20), ("milo", 0.95, 20)]),
        _distance_block("far", [("bear-brand", 0.90, 20), ("milo", 0.61, 20)]),
    ]

    lines = train_model.distance_grid(rows, blocks)
    text = "\n".join(lines)

    assert lines[0].split() == ["class", "close", "mid", "far", "all"]
    # The miss is marked, and the pass it sits next to is not.
    assert "0.610 (20)!" in text and "0.950 (20)" in text and "0.970 (20)" in text
    # The overall number is on the same line, which is the comparison being made.
    assert "0.880 (40)" in text
    assert train_model.distance_misses(blocks) == ["far milo 0.610 (n=20)"]


def test_a_distance_that_never_asked_about_a_class_is_a_dash_not_a_zero():
    """Three states, as in the per-class report: a subset holding no instances of a class is
    not a class the model missed, and `0.000` would send someone to add images for the *item*
    when the split is what has none."""
    rows = [("bear-brand", 0.90, 40), ("milo", 0.90, 40)]
    blocks = [
        _distance_block("close", [("bear-brand", 0.97, 20)]),  # milo absent entirely
        # present but unmeasured in every distance of this block
        _distance_block("far", [("bear-brand", 0.90, 20), ("milo", None, 0)]),
    ]

    lines = train_model.distance_grid(rows, blocks)
    text = "\n".join(lines)

    assert text.count("0.000") == 0
    milo = next(line for line in lines if line.startswith("milo"))
    # close is absent, mid has no block at all, far is present-but-unasked, and `all` is 0.900.
    assert milo.split() == ["milo", "-", "-", "-", "0.900", "(40)"]
    assert train_model.distance_misses(blocks) == []


def test_the_grid_lists_every_class_the_model_knows_not_only_the_scored_ones():
    """A class missing from a distance *is* the finding there, so it has to appear as a row."""
    rows = [("bear-brand", 0.90, 20), ("milo", None, 0)]
    lines = train_model.distance_grid(rows, [])

    assert [line.split()[0] for line in lines[2:]] == ["bear-brand", "milo"]
    assert lines[-1].split()[1:] == ["-", "-", "-", "-"]


def test_distance_misses_is_worst_first():
    """The list is a to-do order, so the worst cell leads."""
    blocks = [
        _distance_block("close", [("milo", 0.80, 20)]),
        _distance_block("far", [("milo", 0.40, 20), ("bear-brand", 0.71, 20)]),
    ]
    assert train_model.distance_misses(blocks) == [
        "far milo 0.400 (n=20)",
        "far bear-brand 0.710 (n=20)",
        "close milo 0.800 (n=20)",
    ]
