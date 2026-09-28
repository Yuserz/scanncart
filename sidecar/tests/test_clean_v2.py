"""Tests for `clean_v2.py`: dropping a class, and the coverage preflight a held-out tier answers.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
import clean_v2
import plan_split

from tests.dataset_tool_helpers import _batch, _entries, _two_session_batches


# --------------------------------------------------------------------------
# Dropping a class: the commands that remove the rest of its footprint
# --------------------------------------------------------------------------


def _staged_with_two_classes(tmp_path: Path) -> Path:
    """A staged set shaped like the real one: two class folders, a manifest pair, one class that
    is being dropped and one that must come through untouched.
    """
    out = tmp_path / "cleaned-v2"
    for slug, names in {
        "palmolive": ["palmolive_0001.jpg", "palmolive_0002.jpg"],
        "milo": ["milo_0001.jpg"],
    }.items():
        (out / slug).mkdir(parents=True)
        for name in names:
            (out / slug / name).write_bytes(b"jpeg")
    entries = [
        {"new_name": "palmolive_0001.jpg", "class": "palmolive", "distance": "mid"},
        {"new_name": "palmolive_0002.jpg", "class": "palmolive", "distance": "far"},
        {"new_name": "milo_0001.jpg", "class": "milo", "distance": "close"},
    ]
    (out / "manifest.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["new_name", "class", "distance"])
        writer.writeheader()
        writer.writerows(entries)
    return out


def test_drop_parks_the_staged_files_and_rewrites_both_manifests(tmp_path, capsys):
    """The one command that removes a class's footprint, run end to end with no network.

    All four halves matter and each fails quietly on its own: the JPEGs have to leave the live
    set (or the next `clean` and the split planner still count them), *both* manifests have to
    shrink (the CSV is what a person reads, the JSON is what the tools read), the other class has
    to be untouched, and the JPEGs have to be **parked rather than deleted** - the capture is real
    work, so re-shooting that product later should be an upload.
    """
    out = _staged_with_two_classes(tmp_path)

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0
    print_out = capsys.readouterr().out

    assert "2 staged image(s)" in print_out
    assert "3 -> 1 entries (2 removed)" in print_out
    # Parked, not deleted, and out of the live folders.
    assert sorted(p.name for p in (out / "dropped" / "palmolive").iterdir()) == [
        "palmolive_0001.jpg",
        "palmolive_0002.jpg",
    ]
    assert not (out / "palmolive").exists()
    # The untouched class is exactly where it was.
    assert (out / "milo" / "milo_0001.jpg").exists()

    kept = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [e["new_name"] for e in kept] == ["milo_0001.jpg"]
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["new_name"] for r in rows] == ["milo_0001.jpg"]
    # And the CSV's own header survives, so a column added to `write_manifest` cannot be dropped
    # by this command.
    assert list(rows[0]) == ["new_name", "class", "distance"]


def test_drop_dry_run_changes_nothing(tmp_path, capsys):
    """A dry run that had already moved the files would be the worst kind of flag."""
    out = _staged_with_two_classes(tmp_path)

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out), "--dry-run"]) == 0

    assert "nothing changed" in capsys.readouterr().out
    assert (out / "palmolive" / "palmolive_0001.jpg").exists()
    assert len(json.loads((out / "manifest.json").read_text(encoding="utf-8"))) == 3


def test_drop_says_so_when_the_class_is_not_staged(tmp_path, capsys):
    """Re-running the command has to be safe: idempotence is what makes it usable in a runbook."""
    out = _staged_with_two_classes(tmp_path)
    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0
    capsys.readouterr()

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0
    assert "nothing to drop" in capsys.readouterr().out


def test_a_leftover_file_in_the_dropped_folder_is_not_swept_away(tmp_path):
    """Only the names the manifest lists are moved. A file an operator put there by hand is theirs,
    and the empty-bucket cleanup must not turn into a delete of something it was never told about.
    """
    out = _staged_with_two_classes(tmp_path)
    (out / "palmolive" / "hand-added.jpg").write_bytes(b"jpeg")

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0

    assert (out / "palmolive" / "hand-added.jpg").exists()
    assert not (out / "palmolive" / "palmolive_0001.jpg").exists()


def test_only_a_drawn_box_counts_as_labeled_when_dropping_from_the_project():
    """The guard that makes `--delete-on-server` safe to run, judged the way the rest of the tools
    judge it: by the *type* of `annotations`, never by emptiness. A null-annotated hard negative is
    `{"count": 0}` - a decision nobody needs protecting from - while a positive count is somebody's
    drawn boxes, and a manifest cannot put those back.
    """
    index = {
        "drawn.jpg": {"id": "a", "annotations": {"count": 3, "classes": {"x": 3}}},
        "null.jpg": {"id": "b", "annotations": {"count": 0}},
        "unlabeled.jpg": {"id": "c", "annotations": []},
    }

    assert clean_v2.labeled_on_server(index, ["drawn.jpg", "null.jpg", "unlabeled.jpg"]) == [
        "drawn.jpg"
    ]
    # An image the project does not have is not "labeled", and not an error either - it is the
    # ordinary case for a frame that was staged and never uploaded.
    assert clean_v2.labeled_on_server(index, ["never-uploaded.jpg"]) == []


def test_with_nothing_staged_every_held_out_cell_is_a_gap():
    """True, and the reason the printer has a separate "cannot tell" branch: one alarming
    line per cell on a project that has simply not been staged yet is how a real warning
    gets trained away."""
    gaps = clean_v2.holdout_gaps(clean_v2.TIER_D, {})

    assert len(gaps) == len(clean_v2.TIER_D.cells) == 21
    assert ("century-tuna", "mid") in gaps and ("milo", "close") in gaps


def test_the_two_cells_tier_a_has_not_shot_read_as_gaps_against_the_rest_of_the_grid():
    """Today's actual state, which is the print's whole purpose: the staged set covers 19 of
    the 21 cells, so a Tier D hold-out would make exactly these two unlearnable - and they
    are two of the cells Tier A is already shooting. The next session has to land before
    `s3` can be held out.

    Two, not three: Palmolive's `close` cell was the third and left with the class, which is
    also why the grid is 21 cells rather than 24.
    """
    coverage = {
        (clean_v2.CLASS_MAP[product], distance): 15
        for (product, distance) in clean_v2.TIER_D.cells
        if (product, distance) not in {("TUNA", "mid"), ("TUNA", "far")}
    }

    assert sorted(clean_v2.holdout_gaps(clean_v2.TIER_D, coverage)) == [
        ("century-tuna", "far"),
        ("century-tuna", "mid"),
    ]


def test_a_missing_or_unreadable_manifest_is_no_coverage_not_a_crash(tmp_path):
    """Before a shoot there is nothing staged, which is the ordinary case - so a directory
    that does not exist, one with no manifest, and one holding garbage all have to read as
    "no coverage" rather than taking the scaffold down."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    assert clean_v2.staged_coverage([tmp_path / "nope", empty, broken]) == {}


def test_the_scaffold_reads_the_staged_coverage_it_reports(tmp_path, capsys):
    """End to end through the command: the real staged set is 1,383 images and the three
    cells Tier A has not shot yet are exactly the ones a Tier D hold-out would break on, so
    the report has to name them rather than pass silently."""
    staged = tmp_path / "cleaned-v2"
    staged.mkdir()
    entries = [
        {"new_name": f"{cls}_{d}_{i}.jpg", "class": cls, "distance": d}
        for cls, cells in {
            "milo": [("close", 40), ("far", 40)],
            "century-tuna": [("close", 30)],
        }.items()
        for d, n in cells
        for i in range(n)
    ]
    (staged / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    root = tmp_path / "s3-capture"

    code = clean_v2.main(
        ["scaffold", "--tier", "d", "--root", str(root), "--out", str(staged)]
    )
    out = capsys.readouterr().out

    assert code == 0
    # The folders exist, so the shoot has somewhere to file into.
    assert (root / "MILO" / "FAR").is_dir()
    assert (root / "TUNA" / "CLOSE").is_dir()
    assert "held-out coverage check" in out
    # Named as the cells a hold-out would break on...
    assert "century-tuna/far" in out and "century-tuna/mid" in out
    # ...and *not* the cells this staged set already covers, since the gap list is what the
    # shoot has to act on (<class>/<distance> is the gap list's own format, not a folder).
    assert "milo/close" not in out and "milo/far" not in out


def test_the_scaffold_says_it_cannot_tell_when_nothing_is_staged(tmp_path, capsys):
    """The case a fresh project is in, and the one where a wall of gaps would be worst: with
    nothing staged there is no coverage to compare against, so the honest answer is that the
    question cannot be answered yet - not 24 lines of `no train images` for a shoot nobody
    has done.
    """
    root = tmp_path / "s3-capture"

    code = clean_v2.main(
        ["scaffold", "--tier", "d", "--root", str(root), "--out", str(tmp_path / "unused")]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "no staged set to compare against" in out
    assert "century-tuna/mid" not in out


def test_a_cell_only_the_held_out_session_covers_is_reported_as_unlearnable():
    """The mistake the rule prevents: hold out the only session that ever shot a cell and
    the model is never taught it, so its test reading measures the absence of training.
    `main` refuses to write a plan in this state rather than warning about it."""
    batches = _two_session_batches() + [_batch("safeguard_mid_s3", "safeguard", "mid", 30, "s3")]
    plan = plan_split.holdout_plan(batches, "s3")
    s = plan_split.summarize(plan, _entries(batches), batches)

    assert s["no_train_cells"] == [("safeguard", "mid")]


def test_a_whole_batch_plan_reports_the_cells_it_leaves_unlearnable():
    """Plan A's accepted compromise, made visible. Which cells went missing is only
    actionable if they are named, and the consequence is spelled out because "no train
    images" reads like a coverage nicety rather than an unlearnable class."""
    batches = [_batch("milo_close", "milo", "close", 10), _batch("milo_far", "milo", "far", 10)]
    entries = _entries(batches)
    plan = {e["new_name"]: ("train" if e["batch"] == "milo_close" else "test") for e in entries}
    text = plan_split.render("t", plan, batches, entries, "note")

    assert "1 of 2 cell(s) have no train images" in text
    assert "`milo/far`" in text


def test_the_acceptance_report_claims_only_the_cells_the_session_shot():
    """The scope of the claim, not just its size: a test session that skipped a cell says
    nothing about that cell, and saying so is what keeps the number honest."""
    batches = _two_session_batches() + [_batch("safeguard_mid", "safeguard", "mid", 30, "s1")]
    entries = _entries(batches)
    plan = plan_split.holdout_plan(batches, "s3")
    text = plan_split.acceptance_report(plan, batches, entries, "s3")

    assert "unseen capture session" in text
    assert "Cells in the acceptance set: 2 of 3." in text
    assert "absent from the acceptance set" in text
    assert "`safeguard/mid`" in text

    # And when the session shot every cell, the claim widens to the whole grid.
    full = _two_session_batches()
    full_plan = plan_split.holdout_plan(full, "s3")
    assert "Every cell is in the acceptance set" in plan_split.acceptance_report(
        full_plan, full, _entries(full), "s3"
    )


