"""Tests for `annotate/human_pass.py` - the checklist as a *render* of the store.

The document's whole value is that it is not a second opinion. Every count and every name in it is
read back off the store, so a checklist that agrees with the store can be walked as a list, and one
that does not is worse than no checklist at all - it sends a person to re-open frames that are
already done while the ones that are still outstanding go unmentioned, and nothing about the
document says which is which. So these tests assert the three lists against the store's own
selectors, the stopping point against the pair the acceptance gate reads, and `--check` against a
store that has moved on.

Everything runs against `tmp_path`: the real workspace is 5 GB of gitignored captures CI does not
have, and the counts in it move every time a person presses `f`.
"""

from __future__ import annotations

import json
from pathlib import Path

from annotate import human_pass
from annotate.human_pass import HUMAN_PASS_NAME, REVIEW_SPLITS, Checklist, checklist, render
from annotate.store import Box, LabelStore, review_worklist

PROVIDER = "local:v1"


def _box(cx: float = 0.5) -> Box:
    return Box(cls=0, cx=cx, cy=0.5, w=0.2, h=0.2)


def _boxes(count: int) -> list[Box]:
    return [_box(cx=0.3 + 0.2 * index) for index in range(count)]


def _staged(
    tmp_path: Path,
    rows: list[tuple[str, str, str, str]],
    splits: dict[str, str] | None = None,
) -> Path:
    """`(name, slug, distance, split)` rows as a staged set: the images, a manifest, the splits.

    `splits` overrides the rows' own fourth field for the cases where the two have to disagree -
    a frame no plan has placed yet, or a set whose split map holds names this one does not stage.
    """
    out = tmp_path / "cleaned-v2"
    entries: list[dict] = []
    for name, slug, distance, _ in rows:
        image = out / slug / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpeg")
        entries.append({"new_name": name, "class": slug, "distance": distance, "session": "s1"})
    (out / "manifest.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    placement = splits if splits is not None else {name: split for name, _, _, split in rows}
    (out / "splits.json").write_text(json.dumps(placement), encoding="utf-8")
    return out


def _negatives(tmp_path: Path, names: list[str]) -> Path:
    """A staged hard-negative set, in the shape `clean_v2.ingest_negatives` writes it.

    Its own manifest and its own directory, which is what makes it an *extra*: the null section is
    empty or not depending on whether this set was resolved, and the document has to say which.
    """
    negatives = tmp_path / "cleaned-negatives"
    entries: list[dict] = []
    for name in names:
        image = negatives / "negative" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpeg")
        entries.append({"new_name": name, "class": "negative", "distance": "", "session": "neg1"})
    (negatives / "manifest.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return negatives


def _store(
    tmp_path: Path,
    rows: list[tuple[str, str, str, str]],
    negatives: list[str] = (),
    splits: dict[str, str] | None = None,
) -> LabelStore:
    out = _staged(tmp_path, rows, splits)
    extras = [_negatives(tmp_path, negatives)] if negatives else []
    return LabelStore(out=out, annotations=tmp_path / "annotations-v2", extras=extras)


def _machine_only(store: LabelStore, name: str, boxes: int = 1) -> None:
    """A weight's unread boxes - the state the gate refuses to measure and this pass exists for."""
    suggested = _boxes(boxes)
    store.record_suggestion(name, suggested, PROVIDER)
    store.write(name, suggested)


def _suggested_nothing(store: LabelStore, name: str) -> None:
    """Asked and proposed nothing: a suggestion on the frame, and no label file written."""
    store.record_suggestion(name, [], PROVIDER)


def _items(store: LabelStore) -> Checklist:
    return checklist(
        store.frames(),
        store.splits(),
        store.provenance(),
        out=store.out,
        annotations=store.annotations,
        extras=tuple(store.extras),
    )


# --------------------------------------------------------------------------
# The counts, and the stopping point they imply
# --------------------------------------------------------------------------


def test_the_header_counts_machine_only_per_split_and_the_stop_is_what_the_gate_cannot_clear(tmp_path):
    """The two numbers the section is worked by: how many are outstanding, and what the header
    falls to once `test` and `valid` are - the gate's pair, not the whole list.

    A frame no plan has placed is counted too, and named `unplaced` rather than dropped: it is
    outstanding work like any other, and a header whose parts did not add up to the total would be
    the one thing a person checking the checklist by eye could not do.
    """
    store = _store(
        tmp_path,
        [
            ("milo_0001.jpg", "milo", "far", "test"),
            ("milo_0002.jpg", "milo", "mid", "test"),
            ("milo_0003.jpg", "milo", "far", "valid"),
            ("milo_0004.jpg", "milo", "mid", "train"),
            ("milo_0005.jpg", "milo", "mid", "train"),
            ("milo_0006.jpg", "milo", "close", "train"),
            ("milo_0007.jpg", "milo", "close", "test"),  # hand-drawn: not machine-only
            ("milo_0008.jpg", "milo", "close", ""),      # not in splits.json at all
        ],
        splits={
            "milo_0001.jpg": "test",
            "milo_0002.jpg": "test",
            "milo_0003.jpg": "valid",
            "milo_0004.jpg": "train",
            "milo_0005.jpg": "train",
            "milo_0006.jpg": "train",
            "milo_0007.jpg": "test",
        },
    )
    for name in (
        "milo_0001.jpg",
        "milo_0002.jpg",
        "milo_0003.jpg",
        "milo_0004.jpg",
        "milo_0005.jpg",
        "milo_0006.jpg",
        "milo_0008.jpg",
    ):
        _machine_only(store, name)
    store.write("milo_0007.jpg", [_box()])

    items = _items(store)
    text = render(items)

    assert items.counts == {"test": 2, "valid": 1, "train": 3, "": 1}
    assert len(items.review) == 7
    # `gate` is the pair the acceptance gate asks about; `remaining` is everything else, which is
    # what the stopping point has to be - `train` alone would miss the unplaced frame.
    assert items.gate == 3
    assert items.remaining == 4
    assert "**7 machine-only awaiting review** (test 2 / valid 1 / train 3 / unplaced 1)" in text
    assert "reaches **4**" in text
    assert "`test` and `valid` are done (3 frames)" in text


def test_the_breakdown_is_in_the_order_the_pass_works_the_splits(tmp_path):
    """`test` first, `valid` next, `train` after - the same order `review_rank` gives the list, so
    the header and the list below it read the same way round."""
    store = _store(
        tmp_path,
        [
            ("milo_0001.jpg", "milo", "mid", "train"),
            ("milo_0002.jpg", "milo", "mid", "valid"),
            ("milo_0003.jpg", "milo", "mid", "test"),
        ],
    )
    for name in ("milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"):
        _machine_only(store, name)

    assert "(test 1 / valid 1 / train 1)" in render(_items(store))


# --------------------------------------------------------------------------
# The three lists
# --------------------------------------------------------------------------


def test_the_review_section_is_rendered_from_the_apps_own_worklist(tmp_path):
    """The list is `review_worklist()` - the same call `/api/frames?review=1` serves - and not the
    store's plain frame order.

    The fixture is built so the two disagree: a `far` frame in `train` sorts first in `frames()`
    (outstanding-and-far leads the worklist) while the review list has to lead with `test`, because
    the gate is what the pass is clearing. Asserting the doc against `frames()` order would pass on
    a document that listed the right names in the wrong order, which is a checklist that quietly
    sends the first hour to `train`.
    """
    store = _store(
        tmp_path,
        [
            ("milo_0001.jpg", "milo", "far", "train"),
            ("milo_0002.jpg", "milo", "close", "test"),
        ],
    )
    _machine_only(store, "milo_0001.jpg")
    _machine_only(store, "milo_0002.jpg")

    store_order = [frame.name for frame in store.frames()]
    items = _items(store)
    text = render(items)

    assert store_order == ["milo_0001.jpg", "milo_0002.jpg"]  # far first, whatever the split
    assert [frame.name for frame in items.review] == ["milo_0002.jpg", "milo_0001.jpg"]
    assert [frame.name for frame in items.review] == [
        frame.name for frame in review_worklist(store.frames(), store.splits())
    ]

    # And the rendered bullets carry that order, with each frame's own split named on it.
    assert text.index("The 1 in `test` and `valid`:") < text.index("- `milo_0002.jpg`")
    assert text.index("- `milo_0002.jpg`") < text.index("The other 1 machine-only frame(s)")
    assert "- `milo_0002.jpg`  -  milo|close  -  split test" in text
    assert "- `milo_0001.jpg`  -  milo|far  -  split train" in text


def test_the_draw_list_needs_a_weight_that_was_asked_and_found_nothing(tmp_path):
    """The draw section's claim is about the *model* - v1 was put to this frame and saw nothing -
    so a frame with no recorded suggestion is not in it.

    That is the difference between evidence and outstanding work: a frame nobody asked about would
    look identical in a truthiness test (`no boxes proposed` and `no proposal at all` are both
    falsy), and it would land in a section that reads as a statement about what v1 cannot do. The
    weight finding something is excluded for the opposite reason: those boxes are in the review
    pass, where a person confirms them.
    """
    store = _store(
        tmp_path,
        [
            ("sardines_0001.jpg", "555-sardines", "far", "valid"),
            ("sardines_0002.jpg", "555-sardines", "far", "test"),
            ("sardines_0003.jpg", "555-sardines", "far", "valid"),
            ("sardines_0004.jpg", "555-sardines", "far", "valid"),
            ("sardines_0005.jpg", "555-sardines", "far", "train"),
            ("sardines_0006.jpg", "555-sardines", "mid", "valid"),
        ],
    )
    _suggested_nothing(store, "sardines_0001.jpg")
    _suggested_nothing(store, "sardines_0002.jpg")
    _machine_only(store, "sardines_0004.jpg")  # asked, and it proposed a box: review work
    _suggested_nothing(store, "sardines_0005.jpg")
    _suggested_nothing(store, "sardines_0006.jpg")

    items = _items(store)
    text = render(items)

    assert [frame.name for frame in items.draws] == ["sardines_0001.jpg", "sardines_0002.jpg"]
    assert "2 frame(s): every `far` frame in `test`/`valid`" in text
    assert "- `sardines_0001.jpg`  -  555-sardines|far  -  split valid" in text
    assert [frame.name for frame in items.review] == ["sardines_0004.jpg"]
    # A frame that is none of the three - never asked, not a negative, not machine-only - is
    # deliberately not listed anywhere. The document is the pass, not the whole worklist.
    assert "sardines_0003.jpg" not in text


def test_the_null_list_is_the_undecided_hard_negatives_in_the_gate_splits(tmp_path):
    """An empty label file is the *decision* that there is no item here, so a negative that
    already carries one is finished work and leaves the list - the same three states the store is
    shaped around, read from the document's side."""
    store = _store(
        tmp_path,
        [("milo_0001.jpg", "milo", "mid", "valid")],
        negatives=[
            "negative_0001.jpg",
            "negative_0002.jpg",
            "negative_0003.jpg",
            "negative_0004.jpg",
        ],
        splits={
            "milo_0001.jpg": "valid",
            "negative_0001.jpg": "valid",
            "negative_0002.jpg": "test",
            "negative_0003.jpg": "train",
            "negative_0004.jpg": "valid",
        },
    )
    store.write("negative_0004.jpg", [], null=True)

    items = _items(store)
    text = render(items)

    assert [frame.name for frame in items.nulls] == ["negative_0001.jpg", "negative_0002.jpg"]
    assert "2 frame(s), one keypress each" in text
    # A hard negative has no distance, so its line names no cell - and the negative that is
    # already marked null is gone from the section rather than listed as done.
    assert "- `negative_0001.jpg`  -  split valid" in text
    assert "negative_0004.jpg" not in text


def test_an_empty_section_says_which_kind_of_empty_it_is(tmp_path):
    """`None left` and `nothing to report` are opposite operator actions.

    The first says the pass is done. The second says the negatives were never in this worklist at
    all - `--out` points elsewhere, or `--no-extras` was passed - and it exists because the extras
    list is a *union* whose absence is otherwise invisible: an un-resolved negative set reports a
    perfectly consistent, smaller set with nothing to flag.
    """
    rows = [
        ("sardines_0001.jpg", "555-sardines", "far", "valid"),
        ("sardines_0002.jpg", "555-sardines", "mid", "valid"),
    ]
    done = _store(tmp_path / "with-extras", rows, negatives=["negative_0001.jpg"])
    done.write("sardines_0001.jpg", [_box()])                # a person drew it: no draw work left
    _machine_only(done, "sardines_0002.jpg")
    done.write("sardines_0002.jpg", [_box(cx=0.55)], confirm=True)
    done.write("negative_0001.jpg", [], null=True)

    text = render(_items(done))
    assert "None left: every `far` frame" in text
    assert "None left: every hard negative staged" in text

    no_extras = _store(tmp_path / "without-extras", rows)
    assert "No hard-negative set is in this worklist at all" in render(_items(no_extras))


def test_a_clean_set_says_so_instead_of_listing_the_worklist(tmp_path):
    """With nothing machine-only, the review section has to read as finished - an empty list under
    a heading that says "work from the top until…" is a section nobody can tell apart from one
    whose frames were dropped by a bug."""
    store = _store(
        tmp_path,
        [
            ("sardines_0001.jpg", "555-sardines", "far", "valid"),
            ("sardines_0002.jpg", "555-sardines", "mid", "valid"),
        ],
        negatives=["negative_0001.jpg"],
    )
    store.write("sardines_0001.jpg", [_box()])
    store.write("sardines_0002.jpg", [_box()], confirm=True)
    store.write("negative_0001.jpg", [], null=True)

    items = _items(store)
    text = render(items)

    assert items.review == [] and items.gate == 0
    assert "**nothing is machine-only**" in text
    assert "Nothing is awaiting review" in text
    assert "reaches **0**" not in text


# --------------------------------------------------------------------------
# The document around the lists
# --------------------------------------------------------------------------


def test_the_document_names_the_store_it_describes_and_says_not_to_edit_it(tmp_path):
    """The banner is the mechanism, not decoration: a checklist written by hand is the drift this
    module exists to prevent, and a checklist from another workspace is a different set entirely -
    both have to be legible from the file itself, since neither travels with a console."""
    store = _store(tmp_path, [("milo_0001.jpg", "milo", "mid", "valid")])

    text = render(_items(store))

    assert "do not edit by hand" in text
    assert "make human-pass" in text
    assert str(store.out) in text
    assert str(store.annotations) in text
    assert "no extra staged set (the hard negatives are not in this worklist)" in text


def test_the_gate_pair_is_the_pair_the_acceptance_gate_reads():
    """A drift guard, not a computation: the annotator cannot import `accept_v2` (it drags the
    trainer in for a two-name tuple), so the pass spells its own pair and this pins the two
    together. Judged against a wider or narrower set, the document would send a person to the wrong
    frames and call the pass finished at the wrong point.
    """
    import accept_v2
    from label_classes import SPLIT_NAMES

    assert set(REVIEW_SPLITS) == set(accept_v2.GATE_SPLITS)
    assert set(REVIEW_SPLITS) <= set(SPLIT_NAMES)


def test_the_passes_lists_are_the_stores_own_selectors():
    """A drift guard rather than a computation, and it is about *where* the rules live.

    The three selectors used to be defined in this module, and the page now serves the draw list
    from `?draw=1` - so a copy here is how the document's sections and the list on screen come to
    disagree about which frames the pass is, with both of them looking right. `REVIEW_SPLITS` is the
    same story one level up: the pair lives with the selector that reads it (`store.GATE_SPLITS`),
    and this module keeps only the name it prints them by.
    """
    from annotate.store import GATE_SPLITS, draw_worklist, gate_frames, null_worklist

    assert human_pass.REVIEW_SPLITS is GATE_SPLITS
    assert human_pass.draw_worklist is draw_worklist
    assert human_pass.null_worklist is null_worklist
    # The gate number is the page's readout too: `Checklist.gate` and `GET /api/frames`'s chip are
    # the same call, so the finish line on screen cannot drift from the one in this document.
    assert human_pass.gate_frames is gate_frames
    # Nothing is left here that the store owns: `draw_frames`/`null_frames`/`suggested_nothing`
    # were the second copy, and an importable name would be one again.
    assert not hasattr(human_pass, "draw_frames")
    assert not hasattr(human_pass, "suggested_nothing")


def test_the_class_keys_read_off_the_stores_own_class_list(tmp_path):
    """`1`-`N` in the instructions is `len(CLASS_SLUGS)`, so adding a class cannot leave the
    document telling an operator to press a key that is not bound to anything."""
    from annotate.store import CLASS_SLUGS

    store = _store(tmp_path, [("milo_0001.jpg", "milo", "mid", "valid")])
    _machine_only(store, "milo_0001.jpg")

    assert f"`1`-`{len(CLASS_SLUGS)}` pick the class" in render(_items(store))


# --------------------------------------------------------------------------
# --check: the point of the exercise
# --------------------------------------------------------------------------


def test_check_fails_when_the_store_moved_on_and_passes_when_it_did_not(tmp_path, capsys):
    """A stale checklist is a *wrong* one, so it has to be possible to notice without reading it.

    The write is compared byte for byte against a fresh render, which is also why the document
    carries no timestamp: a clock inside it would make this comparison fail every time and the flag
    would be useless. The diff is printed so the failure says which frames moved, not just that
    something did.
    """
    store = _store(
        tmp_path,
        [
            ("milo_0001.jpg", "milo", "far", "test"),
            ("milo_0002.jpg", "milo", "mid", "valid"),
        ],
    )
    _machine_only(store, "milo_0001.jpg")
    arguments = [
        "--out",
        str(store.out),
        "--annotations",
        str(store.annotations),
    ]

    assert human_pass.main(arguments) == 0
    path = tmp_path / HUMAN_PASS_NAME
    written = path.read_text(encoding="utf-8")
    assert "**1 machine-only awaiting review** (test 1)" in written

    capsys.readouterr()
    assert human_pass.main(["--check", *arguments]) == 0
    assert "is current" in capsys.readouterr().out

    # Somebody confirms the frame in the app - the store moves on, the document does not.
    store.write("milo_0001.jpg", _boxes(1), confirm=True)
    capsys.readouterr()
    assert human_pass.main(["--check", *arguments]) == 1
    complained = capsys.readouterr().out
    assert "no longer matches the store" in complained
    assert "milo_0001.jpg" in complained

    assert human_pass.main(arguments) == 0
    assert human_pass.main(["--check", *arguments]) == 0
    assert "**nothing is machine-only**" in path.read_text(encoding="utf-8")


def test_check_reports_a_missing_file_rather_than_writing_one(tmp_path, capsys):
    """`--check` must never write: it is the thing a script or a person runs to ask a question, and
    a check that quietly writes the answer cannot be run before you know the answer."""
    store = _store(tmp_path, [("milo_0001.jpg", "milo", "mid", "valid")])

    assert human_pass.main(["--out", str(store.out), "--check"]) == 1
    assert "is missing" in capsys.readouterr().out
    assert not (tmp_path / HUMAN_PASS_NAME).exists()


def test_the_set_has_to_be_staged_before_the_checklist_can_describe_it(tmp_path):
    """No manifest means there is no worklist to render, and the tool fails loudly instead of
    writing a document with zero everything - which would read as a finished pass."""
    import pytest

    with pytest.raises(SystemExit) as refused:
        human_pass.main(["--out", str(tmp_path / "not-staged")])

    assert "no manifest at" in str(refused.value)
    assert not (tmp_path / HUMAN_PASS_NAME).exists()


# --------------------------------------------------------------------------
# --status: the pass as one exit code
# --------------------------------------------------------------------------


def test_status_prints_the_three_sections_and_exits_on_the_gate_not_the_pass(tmp_path, capsys):
    """The question a script has is *is the gate clear*, and the answer has to be the same one the
    document gives: these lines come off the same `Checklist`, so they cannot be a second opinion.

    The exit code is the gate's own rule (no machine-only decision in `test`/`valid`) and not "every
    section is empty": the draw and null lists are outstanding *evidence* - they move the
    per-distance recall and the false-positive half - while a weight's unread boxes in the two
    measured splits are what make the acceptance number a measurement of the annotator. A run that
    waited on the other two would refuse to measure a set whose acceptance number is already sound.

    It writes nothing, which is what makes it safe to run in CI and before the file is rendered.
    """
    store = _store(
        tmp_path,
        [
            ("milo_0001.jpg", "milo", "far", "test"),
            ("milo_0002.jpg", "milo", "far", "valid"),
            ("milo_0003.jpg", "milo", "mid", "train"),
        ],
        negatives=["negative_0001.jpg"],
        splits={
            "milo_0001.jpg": "test",
            "milo_0002.jpg": "valid",
            "milo_0003.jpg": "train",
            "negative_0001.jpg": "valid",
        },
    )
    _machine_only(store, "milo_0001.jpg")  # a gate split: this is what blocks
    _machine_only(store, "milo_0003.jpg")  # `train`: tolerated, and only reported
    _suggested_nothing(store, "milo_0002.jpg")  # a draw to do, which does not block

    arguments = ["--out", str(store.out), "--annotations", str(store.annotations)]

    assert human_pass.main(["--status", *arguments]) == 1
    printed = capsys.readouterr().out
    assert f"set      {store.out}" in printed
    assert "cleaned-negatives" in printed  # the nulls below are empty *or not* because of this
    assert "review   2 machine-only (1 in test/valid, 1 outside them)" in printed
    assert "draw     1 far frame(s) in test/valid a weight was asked about and found nothing in" in printed
    assert "nulls    1 hard negative(s) in test/valid with no null yet" in printed
    assert "gate     blocked by 1 - clear it in test/valid before the acceptance number" in printed
    assert not (tmp_path / HUMAN_PASS_NAME).exists()

    # Clearing the one gate frame flips the code, and the draw still outstanding does not keep it
    # open: the number the pass exists to make quotable is already sound.
    store.write("milo_0001.jpg", _boxes(1), confirm=True)
    capsys.readouterr()

    assert human_pass.main(["--status", *arguments]) == 0
    after = capsys.readouterr().out
    assert "review   1 machine-only (0 in test/valid, 1 outside them)" in after
    assert "draw     1 far frame(s)" in after
    assert "gate     clear - nothing machine-only in test/valid" in after
    assert not (tmp_path / HUMAN_PASS_NAME).exists()


def test_status_and_check_are_one_mode_at_a_time(tmp_path):
    """Both write nothing, but a run has one exit code and they answer different questions - "is the
    file current" and "is the pass finished" - so asking both is refused rather than resolved by
    precedence."""
    import pytest

    with pytest.raises(SystemExit):
        human_pass.build_parser().parse_args(["--check", "--status", "--out", str(tmp_path)])
