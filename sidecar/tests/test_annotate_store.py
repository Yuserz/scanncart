"""Tests for `annotate/store.py` - the three states, the provenance, and the two refusals.

The states are the point of this file. A label file that is **absent** means outstanding work; an
**empty** one means somebody decided there is no item here. That distinction is what the cloud path
had to read off the *type* of an API field, and it is what a naive "the file is empty, so it is not
labeled" sweep destroys - re-opening the hard negatives, whose whole purpose is to be training
images with no boxes, and taking the false-positive half of the acceptance numbers with them.

Everything runs against `tmp_path`: the real workspace is 5 GB of gitignored captures that CI does
not have and must not need.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotate.store import (
    CLASS_NAMES,
    SLUG_BY_NAME,
    UNRECORDED,
    Box,
    LabelStore,
    box_from_row,
    counts,
    draw_worklist,
    filter_values,
    matches_filter,
    review_rank,
    review_worklist,
    suggested_nothing,
    summary,
)


def _box(cls: int = 0, cx: float = 0.5, cy: float = 0.5, w: float = 0.2, h: float = 0.2) -> Box:
    return Box(cls=cls, cx=cx, cy=cy, w=w, h=h)


def _negatives(tmp_path: Path, frames: int = 1) -> Path:
    """A staged hard-negative set, in the shape `clean_v2.ingest_negatives` writes."""
    negatives = tmp_path / "cleaned-negatives"
    (negatives / "negative").mkdir(parents=True)
    entries = []
    for i in range(1, frames + 1):
        name = f"negative_{i:04d}.jpg"
        (negatives / "negative" / name).write_bytes(f"jpeg-{i}".encode())
        entries.append({"new_name": name, "class": "negative", "distance": "", "session": "neg1"})
    (negatives / "manifest.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return negatives


def _staged(tmp_path: Path, frames: int = 3, slug: str = "milo") -> Path:
    """A staged set in the shape `clean_v2.stage` writes: `<out>/<slug>/<name>` + a manifest."""
    out = tmp_path / "cleaned-v2"
    (out / slug).mkdir(parents=True)
    entries = []
    for i in range(1, frames + 1):
        name = f"{slug}_{i:04d}.jpg"
        (out / slug / name).write_bytes(f"jpeg-{i}".encode())
        entries.append(
            {
                "new_name": name,
                "class": slug,
                "distance": "mid",
                "session": "s1",
                "batch": f"{slug}_mid",
                "tags": ["mid", slug],
            }
        )
    (out / "manifest.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return out


def _store(tmp_path: Path, frames: int = 3, slug: str = "milo") -> LabelStore:
    return LabelStore(out=_staged(tmp_path, frames, slug), annotations=tmp_path / "annotations-v2")


# --------------------------------------------------------------------------
# The three states
# --------------------------------------------------------------------------


def test_an_absent_label_file_and_an_empty_one_are_different_states(tmp_path):
    """The distinction the whole store is shaped around, asserted at the two places that read it.

    `read_boxes` answers `[]` for both - a box list is a box list - so the *state* has to come from
    `state_of`, and the frame's own record has to agree with it. Getting this wrong is not a display
    bug: a null re-read as "outstanding" sends an operator to re-open 50 hard negatives, and
    re-saving one of them with a box is how a background frame becomes a false-positive example.
    """
    store = _store(tmp_path)
    negatives = _negatives(tmp_path)
    store.extras = [negatives]

    # A product frame: absent, then boxes. An empty label here is refused (see the refusals
    # below), so a product frame can never reach the `null` state.
    assert store.state_of("milo_0001.jpg") == "unlabeled"
    assert store.read_boxes("milo_0001.jpg") == []
    store.write("milo_0001.jpg", [_box()])
    assert store.state_of("milo_0001.jpg") == "labeled"
    assert len(store.read_boxes("milo_0001.jpg")) == 1

    # A hard negative: absent, then a deliberate empty file - and the two read differently even
    # though `read_boxes` answers `[]` for both.
    assert store.state_of("negative_0001.jpg") == "unlabeled"
    store.write("negative_0001.jpg", [], null=True)
    assert store.state_of("negative_0001.jpg") == "null"
    assert store.read_boxes("negative_0001.jpg") == []

    frame = store.frame("milo_0001.jpg")
    assert frame is not None and frame.state == "labeled" and frame.boxes == 1
    negative = store.frame("negative_0001.jpg")
    assert negative is not None and negative.state == "null" and negative.boxes == 0


def test_a_saved_row_round_trips_through_the_trainers_own_format(tmp_path):
    """YOLO txt, because that is what ultralytics eats - so the dataset build is a copy, not a
    translation, and there is no format whose semantics have to be re-derived later."""
    store = _store(tmp_path)
    store.write("milo_0001.jpg", [_box(cls=3, cx=0.25, cy=0.75, w=0.5, h=0.1)])

    text = (tmp_path / "annotations-v2" / "milo_0001.txt").read_text(encoding="utf-8")
    assert text.strip() == "3 0.250000 0.750000 0.500000 0.100000"

    (box,) = store.read_boxes("milo_0001.jpg")
    assert (box.cls, box.cx, box.cy, box.w, box.h) == (3, 0.25, 0.75, 0.5, 0.1)


def test_the_class_order_that_a_row_indexes_is_written_down_beside_the_labels(tmp_path):
    """`cls` is a number, so the file that says which number is which product has to exist.

    A dataset whose `names` are in another order relabels every frame and trains happily, which is
    the failure `train_model.check_export` cannot see (it compares names by membership). Writing the
    order into the annotation tree is what lets the build refuse rather than guess.
    """
    store = _store(tmp_path)
    path = store.write_classes()

    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["names"] == list(CLASS_NAMES)
    assert len(body["slugs"]) == len(body["names"])
    assert all(SLUG_BY_NAME[name] == slug for name, slug in zip(body["names"], body["slugs"]))


# --------------------------------------------------------------------------
# The two refusals
# --------------------------------------------------------------------------


def test_a_null_is_refused_on_a_product_frame_and_a_box_on_a_hard_negative(tmp_path):
    """Two mistakes, in opposite directions, and both of them train the model wrong.

    A null on a product frame teaches the head that the product is not in the picture; a box on a
    hard negative destroys the only frames in the set that say "nothing of interest here". The store
    refuses both, which is why the endpoint surfaces this as a 409 rather than as a 500.
    """
    store = _store(tmp_path)
    store.extras = []
    with pytest.raises(ValueError, match="not a hard negative"):
        store.write("milo_0001.jpg", [], null=True)

    store.extras = [_negatives(tmp_path)]
    with pytest.raises(ValueError, match="carries no box by definition"):
        store.write("negative_0001.jpg", [_box()])
    # And the deliberate null is exactly what that frame does accept.
    store.write("negative_0001.jpg", [], null=True)
    assert store.state_of("negative_0001.jpg") == "null"


def test_an_out_of_frame_box_is_refused_before_anything_is_written(tmp_path):
    """A normalized coordinate outside 0..1 is not a label that "looks wrong" later - the trainer
    clips or drops it, and the frame reads as labeled while contributing nothing."""
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="out-of-frame"):
        store.write("milo_0001.jpg", [Box(cls=0, cx=1.4, cy=0.5, w=0.2, h=0.2)])
    with pytest.raises(ValueError, match="out-of-frame"):
        store.write("milo_0001.jpg", [Box(cls=0, cx=0.5, cy=0.5, w=0.0, h=0.2)])

    assert store.state_of("milo_0001.jpg") == "unlabeled"


def test_a_row_that_is_not_a_box_is_dropped_rather_than_crashing(tmp_path):
    """The reader runs over files a human may have edited: a comment, a short row and a non-numeric
    field are all "no box", because the alternative is an annotator that cannot open the folder."""
    assert box_from_row("0 0.5 0.5 0.2 0.2") == Box(0, 0.5, 0.5, 0.2, 0.2)
    assert box_from_row("") is None
    assert box_from_row("0 0.5 0.5") is None
    assert box_from_row("0 0.5 x 0.2 0.2") is None


# --------------------------------------------------------------------------
# Machine-only, written on save and never on suggestion
# --------------------------------------------------------------------------


def test_a_suggestion_alone_writes_no_label_and_marks_nothing_machine_only(tmp_path):
    """The rule that keeps the acceptance gate honest: provenance about boxes that are *not on disk
    yet* is not provenance about a frame somebody has finished."""
    store = _store(tmp_path)
    store.record_suggestion("milo_0001.jpg", [_box()], "local:v1")

    frame = store.frame("milo_0001.jpg")
    assert frame is not None
    assert frame.state == "unlabeled" and frame.machine_only is False
    assert store.read_boxes("milo_0001.jpg") == []
    # The suggestion is still remembered, which is what makes the next save comparable to it.
    assert (store.provenance()["milo_0001.jpg"]["suggestion"]["provider"]) == "local:v1"


def test_saving_a_suggestion_unchanged_is_machine_only_and_editing_it_is_not(tmp_path):
    """The comparison happens on the server, at save, against the rows it recorded itself.

    A client's own claim about what it drew is not evidence - and the field this decides
    (`machine_only`) is what `build_dataset` reports per split and what stops a machine's unread
    boxes from being counted as reviewed work.
    """
    store = _store(tmp_path)
    suggested = [_box(cls=1, cx=0.4, cy=0.4, w=0.3, h=0.3)]
    store.record_suggestion("milo_0001.jpg", suggested, "local:v1")

    record = store.write("milo_0001.jpg", suggested)
    assert record["machine_only"] is True
    assert record["provider"] == "local:v1"

    # One coordinate moved by a human: no longer machine-only, whatever the client says.
    edited = [Box(cls=1, cx=0.41, cy=0.4, w=0.3, h=0.3)]
    record = store.write("milo_0001.jpg", edited)
    assert record["machine_only"] is False
    assert store.frame("milo_0001.jpg").machine_only is False


def test_confirming_a_weights_boxes_clears_machine_only_without_touching_them(tmp_path):
    """The ordinary outcome of a review pass, and the one thing the comparison cannot express: a
    person looked at a weight's boxes, agreed, and saved them *unchanged*. Without this the frame
    is machine-only for ever - the rows still equal the suggestion, so nothing in the files could
    tell - and the acceptance gate (zero in valid/test) is unreachable.
    """
    store = _store(tmp_path)
    suggested = [_box(cls=1, cx=0.4, cy=0.4, w=0.3, h=0.3)]
    store.record_suggestion("milo_0001.jpg", suggested, "local:v1")
    assert store.write("milo_0001.jpg", suggested)["machine_only"] is True

    record = store.write("milo_0001.jpg", suggested, confirm=True)

    assert record["machine_only"] is False
    assert record["confirmed_at"]
    # The rows are untouched, which is the point: a confirmation is about who checked them.
    assert store.read_boxes("milo_0001.jpg") == suggested
    assert store.frame("milo_0001.jpg").machine_only is False

    # And it sticks: a later identical save cannot put the frame back in the gate's count.
    assert store.write("milo_0001.jpg", suggested)["machine_only"] is False


def test_a_confirmation_does_not_survive_a_change_of_the_boxes(tmp_path):
    """The other direction, and it needs no new rule: edited rows are not `unchanged`, so they are
    a person's work by the ordinary path."""
    store = _store(tmp_path)
    suggested = [_box(cls=1, cx=0.4, cy=0.4, w=0.3, h=0.3)]
    store.record_suggestion("milo_0001.jpg", suggested, "local:v1")
    store.write("milo_0001.jpg", suggested, confirm=True)

    record = store.write("milo_0001.jpg", [_box(cls=1, cx=0.41, cy=0.4, w=0.3, h=0.3)])

    assert record["machine_only"] is False  # still reviewed, now as an edit
    assert record["provider"] == "unknown"  # and no longer attributed to the weight


def test_a_label_written_with_no_suggestion_behind_it_is_never_called_machine_only(tmp_path):
    """Drawn by hand from the start: `provider` is `unknown` rather than `human`, because the store
    genuinely does not know - and the safe reading of "I do not know who drew this" for a gate that
    excludes machine work is *not* machine-only."""
    store = _store(tmp_path)
    record = store.write("milo_0001.jpg", [_box()])

    assert record["machine_only"] is False
    assert record["provider"] == "unknown"
    assert record["image_sha256"]


# --------------------------------------------------------------------------
# The review pass: the gate's own priority, and nothing already checked
# --------------------------------------------------------------------------


def test_the_review_pass_is_the_acceptance_splits_first_and_only_the_unreviewed(tmp_path):
    """A second pass over the same set, ordered by what the gate costs: unread boxes in `test`
    make the acceptance number a measurement of the annotator, `valid` is next, `train` is cheap
    until those are clean, and a frame no plan has placed is last - named, not mistaken for a
    fourth split.
    """
    store = _store(tmp_path, frames=4)
    splits = {
        "milo_0001.jpg": "train",
        "milo_0002.jpg": "test",
        "milo_0003.jpg": "valid",
        # milo_0004.jpg is deliberately absent from `splits.json`.
    }
    # 2 and 4 are a machine's unread boxes; 1 and 3 were drawn (or edited) by hand.
    store.record_suggestion("milo_0002.jpg", [_box()], "local:v1")
    store.write("milo_0002.jpg", [_box()])
    store.record_suggestion("milo_0004.jpg", [_box()], "local:v1")
    store.write("milo_0004.jpg", [_box()])
    store.write("milo_0001.jpg", [_box()])
    store.record_suggestion("milo_0003.jpg", [_box()], "local:v1")
    store.write("milo_0003.jpg", [_box(cx=0.41)])

    worklist = review_worklist(store.frames(), splits)

    assert [frame.name for frame in worklist] == ["milo_0002.jpg", "milo_0004.jpg"]
    assert all(frame.machine_only for frame in worklist)
    # The order is a *selection* first: the hand-drawn frames are not merely sorted last.
    assert len(worklist) == 2


def test_the_review_pass_starts_with_the_frames_the_v2_verdict_rests_on(tmp_path):
    """Within a split, the order is the verdict's own thin places rather than the cell order.

    Two signals, both of them where the acceptance evidence is weakest: a `far`/`mid` frame is an
    instance the per-distance grid can quote (`far` is where v1 finds nothing at all), and a frame
    holding two or more boxes is one the crowded counter is counted over. The split still outranks
    both - every `test` frame comes before `train`'s crowded `far` one - because the gate reads
    valid/test and this pass has to clear it soonest.
    """
    out = tmp_path / "cleaned-v2"
    (out / "milo").mkdir(parents=True)
    rows = [
        ("milo_0001.jpg", "close", 1, "test"),
        ("milo_0002.jpg", "close", 2, "test"),
        ("milo_0003.jpg", "mid", 1, "test"),
        ("milo_0004.jpg", "mid", 2, "test"),
        ("milo_0005.jpg", "far", 1, "test"),
        ("milo_0006.jpg", "far", 2, "test"),
        ("milo_0007.jpg", "far", 2, "train"),
    ]
    (out / "manifest.json").write_text(
        json.dumps(
            [
                {"new_name": name, "class": "milo", "distance": distance, "session": "s1"}
                for name, distance, _, _ in rows
            ]
        ),
        encoding="utf-8",
    )
    (out / "splits.json").write_text(
        json.dumps({name: split for name, _, _, split in rows}), encoding="utf-8"
    )
    store = LabelStore(out=out, annotations=tmp_path / "annotations-v2")
    for name, _, boxes, _ in rows:
        (out / "milo" / name).write_bytes(b"jpeg")
        suggested = [_box(cx=0.3 + 0.2 * i) for i in range(boxes)]
        store.record_suggestion(name, suggested, "local:v1")
        store.write(name, suggested)  # saved unchanged: a machine's, and unreviewed

    worklist = review_worklist(store.frames(), store.splits())

    assert [frame.name for frame in worklist] == [
        "milo_0006.jpg",  # test, far, two items - on both the grid's and the counter's evidence
        "milo_0005.jpg",  # test, far
        "milo_0004.jpg",  # test, mid, two items
        "milo_0003.jpg",  # test, mid
        "milo_0002.jpg",  # test, close, two items - still ahead of every single-item close frame
        "milo_0001.jpg",  # test, close
        "milo_0007.jpg",  # train: the split outranks the distance, so this one stays last
    ]


def test_the_draw_list_is_only_what_a_weight_was_asked_about_and_found_nothing_in(tmp_path):
    """The pass's other half, and one function for both readers: the checklist's draw section and
    `GET /api/frames?draw=1` are the same call, so the rule is pinned here rather than in either.

    The claim the list makes is about the *model* - v1 was put to this frame and saw nothing - which
    is why the recorded suggestion has to be checked strictly. A frame nobody asked about looks
    identical to a truthiness test (`no boxes` either way) and is outstanding work of a different
    kind, and one the weight *did* propose a box for belongs to the review pass, where a person
    confirms it. `far` and the gate splits scope the rest: drawing a `train` frame is real work with
    no effect on the number the pass is running to make checkable.
    """
    out = tmp_path / "cleaned-v2"
    (out / "milo").mkdir(parents=True)
    rows = [
        ("milo_0001.jpg", "far", "valid", 0),      # asked, found nothing
        ("milo_0002.jpg", "far", "test", 1),       # asked, proposed a box
        ("milo_0003.jpg", "far", "valid", None),   # never asked
        ("milo_0004.jpg", "far", "train", 0),      # nothing found, but `train` costs the gate nothing
        ("milo_0005.jpg", "mid", "valid", 0),      # nothing found, and `mid` is not the thin axis
    ]
    (out / "manifest.json").write_text(
        json.dumps(
            [
                {"new_name": name, "class": "milo", "distance": distance, "session": "s2"}
                for name, distance, _, _ in rows
            ]
        ),
        encoding="utf-8",
    )
    (out / "splits.json").write_text(
        json.dumps({name: split for name, _, split, _ in rows}), encoding="utf-8"
    )
    store = LabelStore(out=out, annotations=tmp_path / "annotations-v2")
    for name, _, _, boxes in rows:
        (out / "milo" / name).write_bytes(b"jpeg")
        if boxes is None:
            continue
        suggested = [_box(cx=0.3 + 0.2 * i) for i in range(boxes)]
        store.record_suggestion(name, suggested, "local:v1")
        if boxes:
            store.write(name, suggested)  # saved unchanged: a machine's, and unreviewed

    draws = draw_worklist(store.frames(), store.splits(), store.provenance())

    assert [frame.name for frame in draws] == ["milo_0001.jpg"]

    # What the one inclusion rests on, at its source: asked-and-empty is not the same record as
    # never-asked, and the store writes the first as an empty list (`record_suggestion(n, [], ...)`).
    assert suggested_nothing(store.provenance()["milo_0001.jpg"]) is True
    assert suggested_nothing({"suggestion": {"boxes": []}}) is True
    assert suggested_nothing({"suggestion": {"boxes": ["0 0.5 0.5 0.1 0.1"]}}) is False
    assert suggested_nothing({}) is False
    assert suggested_nothing({"suggestion": None}) is False


def test_a_filter_can_name_a_value_that_was_never_recorded_and_the_empty_one_means_any():
    """What `GET /api/frames` narrows with. The distinction that matters is the same one the store's
    three states turn on: an empty parameter has to keep meaning the whole set, so the frames whose
    distance or split is missing are asked for by the store's own token for one (`UNRECORDED`, which
    `Frame.cell` spells too) - otherwise the hard negatives, and every frame no plan has reached,
    could never be singled out.
    """
    assert filter_values("") is None
    assert filter_values("  ,  ") is None
    assert filter_values("test, valid") == frozenset({"test", "valid"})
    assert filter_values(" test , test ") == frozenset({"test"})

    assert matches_filter(None, "test") is True
    assert matches_filter(None, "") is True
    assert matches_filter(filter_values("test"), "test") is True
    assert matches_filter(filter_values("test,valid"), "valid") is True
    assert matches_filter(filter_values("far"), "close") is False
    assert matches_filter(filter_values(UNRECORDED), "") is True
    assert matches_filter(filter_values("far"), "") is False


def test_a_split_nobody_planned_for_sorts_after_an_unplaced_frame():
    """`review_rank` is total: a project that assigns a split this app does not know about must
    not crash the pass, and must not silently outrank the frames whose split is simply missing -
    both are "not where the gate looks", and they belong together at the end."""
    assert review_rank("test") < review_rank("valid") < review_rank("train") < review_rank("")
    assert review_rank("holdout") > review_rank("")


def test_the_review_pass_is_empty_rather_than_wrong_when_nothing_is_unreviewed(tmp_path):
    """The state the gate is driving at: a clean set has no review work, and the pass says so
    rather than falling back to the whole worklist (which would look like a review pass)."""
    store = _store(tmp_path, frames=2)
    store.write("milo_0001.jpg", [_box()])

    assert review_worklist(store.frames(), {}) == []


# --------------------------------------------------------------------------
# The worklist and the summary
# --------------------------------------------------------------------------


def test_the_worklist_puts_the_outstanding_far_frames_first(tmp_path):
    """The order is the worklist, and it is a judgement about where an hour is best spent: `far` is
    the axis this dataset exists for, and an untouched cell is what stalls a project."""
    out = tmp_path / "cleaned-v2"
    out.mkdir()
    entries = [
        {"new_name": "milo_0001.jpg", "class": "milo", "distance": "close", "session": "s1"},
        {"new_name": "milo_0002.jpg", "class": "milo", "distance": "far", "session": "s1"},
        {"new_name": "milo_0003.jpg", "class": "milo", "distance": "mid", "session": "s1"},
    ]
    for entry in entries:
        (out / entry["new_name"]).write_bytes(b"jpeg")
    (out / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    store = LabelStore(out=out, annotations=tmp_path / "annotations-v2")
    store.write("milo_0001.jpg", [_box()])  # the close frame is already done

    assert [f.name for f in store.frames()] == [
        "milo_0002.jpg",  # outstanding, far
        "milo_0003.jpg",  # outstanding, mid
        "milo_0001.jpg",  # decided, so last whatever its cell
    ]


def test_the_summary_is_the_shape_the_desktop_panel_already_reads(tmp_path):
    """Same keys the Roboflow writer produces, plus the two this path adds.

    `app/dataset_status.py` must not be able to tell the two writers apart - otherwise the panel
    would need a second reader for the local case, and the two would disagree about what "decided"
    means. The additions are the honest part: `pseudo` and `machine_only_by_split`.
    """
    store = _store(tmp_path, frames=4)
    store.record_suggestion("milo_0001.jpg", [_box()], "local:v1")
    store.write("milo_0001.jpg", [_box()])               # machine-only
    store.write("milo_0002.jpg", [_box(), _box(cls=1)])  # by hand

    frames = store.frames()
    body = summary(frames, {"milo_0001.jpg": "test", "milo_0002.jpg": "train"})

    assert body["total"] == 4
    assert body["decided"] == 2
    assert body["pseudo"] == 1
    assert body["reviewed"] == 1
    assert body["by_cell"] == {"milo|mid": [2, 4]}
    assert body["pseudo_by_cell"] == {"milo|mid": 1}
    assert body["machine_only_by_split"] == {"test": 1}
    assert body["by_class"] == {"milo": [2, 4]}
    # The reader's own shape, so `dataset_status._sessions` needs no branch for this writer.
    assert body["sessions"] == [
        {"name": "s1", "train": 1, "valid": 0, "test": 1, "decided": 2, "total": 4, "splits": 2}
    ]

    totals = counts(frames)
    assert (totals.total, totals.decided, totals.unlabeled) == (4, 2, 2)


def test_a_set_listed_twice_is_still_one_set(tmp_path):
    """Reachable now that the hard negatives are a *default* extra: a caller that names the same
    directory again (or names the staged set itself as an extra) would otherwise walk it twice,
    which doubles every frame in the worklist and every count in the per-split report - a doubling
    nothing flags, because the second write of a frame lands on the same filename."""
    out = _staged(tmp_path)
    negatives = _negatives(tmp_path)

    store = LabelStore(out=out, annotations=tmp_path / "annotations-v2", extras=[negatives, negatives, out])

    assert store.extras == [negatives]
    assert [f.name for f in store.frames()].count("negative_0001.jpg") == 1
    assert len(store.frames()) == 4  # three product frames and one negative


def test_a_frame_with_no_distance_reports_the_cell_the_panel_already_keys_on(tmp_path):
    """The hard negatives carry no distance, and the Roboflow writer spells that `negative|unknown`.
    A local snapshot that spelled the same cell differently would split one bucket into two in the
    panel's worklist - a bug that looks like missing data."""
    store = LabelStore(
        out=tmp_path / "none",
        annotations=tmp_path / "annotations-v2",
        extras=[_negatives(tmp_path)],
    )
    store.write("negative_0001.jpg", [], null=True)

    (frame,) = store.frames()
    assert frame.cell == "negative|unknown"
    assert summary([frame])["by_cell"] == {"negative|unknown": [1, 1]}


def test_splits_are_read_when_they_exist_and_absent_is_not_an_error(tmp_path):
    """The split file is captured before the machine goes offline; without it the annotator still
    works, and the summary simply has no per-split numbers rather than an invented ones."""
    store = _store(tmp_path)
    assert store.splits() == {}

    (store.out / "splits.json").write_text(
        json.dumps({"milo_0001.jpg": "test", "milo_0002.jpg": "train"}), encoding="utf-8"
    )
    assert store.splits() == {"milo_0001.jpg": "test", "milo_0002.jpg": "train"}
