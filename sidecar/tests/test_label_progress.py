"""Tests for `label_progress.py`: the three annotation states, and the local source.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import json
import re
import annotate.store
import build_dataset
import clean_v2
import label_classes
import label_progress

from tests.dataset_tool_helpers import _local_args, _staged_local


# --------------------------------------------------------------------------


def test_state_of_distinguishes_the_three_states():
    """Only the *type* of `annotations` separates "marked null" from "not looked at".
    Collapsing them would silently re-open every hard negative on the next labeling pass.
    """
    assert label_progress.state_of({"annotations": []}) == ("unlabeled", 0)
    assert label_progress.state_of({"annotations": {"count": 0, "classes": {}}}) == ("null", 0)
    assert label_progress.state_of({"annotations": {"count": 2, "classes": {"x": 2}}}) == ("labeled", 2)
    # A missing field must not crash or be read as a null annotation.
    assert label_progress.state_of({}) == ("unlabeled", 0)


def test_annotated_classes_is_empty_for_a_null_annotation():
    assert not label_progress.annotated_classes({"annotations": {"count": 0, "classes": {}}})
    assert not label_progress.annotated_classes({"annotations": []})


def test_slug_of_ignores_the_distance_tag():
    rec = {"tags": ["close", "milo"]}
    assert label_progress.slug_of(rec) == "milo"
    assert label_progress.distance_of(rec) == "close"


def test_a_hard_negative_frame_resolves_to_its_pseudo_class_not_unknown():
    """Otherwise the frame lands in an `unknown` bucket, which reads as a filing bug
    rather than as the deliberate background set it is - and `unknown` is not in
    SLUG_TO_CLASS, so no name would resolve for the row either.
    """
    rec = {"tags": [clean_v2.NEGATIVE_CLS]}
    assert label_progress.slug_of(rec) == clean_v2.NEGATIVE_CLS
    assert label_progress.distance_of(rec) == ""


def test_a_background_row_is_named_as_background_and_never_as_a_ninth_class():
    """The progress table lists every slug present, so the pseudo-class gets a row. It
    has to read as "nothing to draw here", not as a class someone forgot to label.
    """
    pseudo = clean_v2.NEGATIVE_CLS
    assert label_progress.display_name(pseudo) != pseudo
    assert pseudo in label_progress.display_name(pseudo)
    assert label_progress.display_name(pseudo).startswith("(")
    # Real classes still print their real name.
    assert label_progress.display_name("milo") == "Milo Chocolate Drink 22g Sachet"


# --------------------------------------------------------------------------
# 3b. the local source: the same snapshot, from the annotator's own store
# --------------------------------------------------------------------------
#
# This is the source that keeps the tooling usable after the machine goes offline, and the
# reason the acceptance gate can be read off the panel at all: provenance is a local fact, so
# `awaiting_review` and `machine_only_by_split` exist only here. The Roboflow source leaves
# them out rather than writing a zero - see `roboflow_progress`.


def test_the_local_source_reads_the_annotators_labels_and_needs_no_key(tmp_path, monkeypatch):
    """Offline by construction: no `httpx.Client`, no `load_key`, no Roboflow project. The
    frames and their labels are on disk, so the numbers the panel renders come from them.
    """
    out = _staged_local(tmp_path)

    def explode(*a, **k):  # pragma: no cover - only fires if the local path touches the network
        raise AssertionError("the local source must not open a client")

    monkeypatch.setattr(label_progress.httpx, "Client", explode)
    monkeypatch.setattr(label_progress, "load_key", explode)

    progress = label_progress.local_progress(_local_args(out), out)

    assert progress.summary["source"] == "local"
    assert progress.summary["total"] == 3
    assert progress.summary["decided"] == 0
    assert progress.summary["by_cell"] == {"milo|mid": [0, 3]}
    assert progress.summary["classes"]["milo"] == "Milo Chocolate Drink 22g Sachet"


def test_the_local_source_reports_the_work_a_machine_did_and_nobody_reviewed(tmp_path):
    """The one number the Roboflow source cannot answer, and the one the acceptance gate reads:
    a decision a weight made, saved unchanged, is *decided* but not reviewed."""
    out = _staged_local(tmp_path, frames=2)
    store = annotate.store.LabelStore(out=out, annotations=tmp_path / "annotations-v2")
    from annotate.store import Box

    suggested = [Box(0, 0.5, 0.5, 0.2, 0.2)]
    store.record_suggestion("milo_0001.jpg", suggested, "local:scanncart-grocery-v1")
    store.write("milo_0001.jpg", suggested)          # machine-only
    store.write("milo_0002.jpg", [Box(0, 0.4, 0.4, 0.2, 0.2)])  # by hand
    # `splits.json` sits beside the staged set, which is where `build_dataset.py` captures it
    # and what `LabelStore.splits()` reads. The annotations tree is for labels only.
    (out / "splits.json").write_text(
        json.dumps({"milo_0001.jpg": "test", "milo_0002.jpg": "train"}), encoding="utf-8"
    )

    summary = label_progress.local_progress(_local_args(out), out).summary

    assert (summary["decided"], summary["pseudo"], summary["reviewed"]) == (2, 1, 1)
    # Per split, because that is where the gate lives: a machine's unread boxes in `train` are
    # cheap, and the same boxes in `test` make the acceptance number a measure of the annotator.
    assert summary["machine_only_by_split"] == {"test": 1}
    assert summary["by_split"] == {"test": [1, 1], "train": [1, 1]}


def test_the_local_source_and_the_project_source_share_every_key_they_both_set(tmp_path):
    """A zero would be a claim the writer cannot support. `dataset_status` reads absence as
    unknown, and the panel says so instead of reporting a clean bill of health."""
    out = _staged_local(tmp_path)
    progress = label_progress.local_progress(_local_args(out), out)
    summary = progress.summary
    assert set(summary) - {"pseudo", "reviewed", "machine_only_by_split"}
    assert "source" in summary
    # And the shared keys are all present, which is what lets one reader serve both writers.
    for key in (
        "total", "decided", "null_annotations", "percent", "by_class", "by_cell",
        "by_split", "sessions", "mismatches", "tier_a", "classes", "generated_at", "project",
    ):
        assert key in summary, key


def test_a_frame_with_no_split_entry_is_not_a_fourth_split(tmp_path):
    """`splits.json` is captured before the machine goes offline; a frame it does not mention is
    unplaced work, not a split of its own - and the session rows only carry the three real ones."""
    out = _staged_local(tmp_path, frames=2)
    (out / "splits.json").write_text(json.dumps({"milo_0001.jpg": "train"}), encoding="utf-8")

    summary = label_progress.local_progress(_local_args(out), out).summary

    assert summary["by_split"]["train"] == [0, 1]
    assert summary["by_split"]["?"] == [0, 1]  # reported, and never mistaken for a split
    assert summary["sessions"] == [
        {"name": "s2", "train": 1, "valid": 0, "test": 0, "decided": 0, "total": 2, "splits": 1}
    ]


def test_a_label_drawn_with_the_wrong_class_is_a_mismatch_locally_too(tmp_path):
    """The check exists on both paths because it catches the same typo: locally the drawn class
    is a label row rather than an API's name summary, and a hard negative is exempt - a box on
    one is C2a's "product plus clutter" material, not a mistake."""
    out = _staged_local(tmp_path, frames=1)
    store = annotate.store.LabelStore(out=out, annotations=tmp_path / "annotations-v2")

    wrong = list(label_classes.SLUG_TO_CLASS).index("safeguard")
    store.write("milo_0001.jpg", [annotate.store.Box(wrong, 0.5, 0.5, 0.2, 0.2)])

    summary = label_progress.local_progress(_local_args(out), out).summary

    assert summary["mismatches"] == 1
    assert len(label_progress.local_progress(_local_args(out), out).mismatches) == 1


