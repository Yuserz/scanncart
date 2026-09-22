"""Tests for the dataset prep tools in `sidecar/tools/`. No camera, GPU, network or API key.

Two kinds of test here, and the second kind is the one that earns its keep:

1. Behavioural - the split planner's proportions, its minimums, its determinism.
2. Drift guards - the v1 class names exist in three places that must agree (the doc,
   the cleaner's folder mapping, and the labeling tool's mapping). CLAUDE.md already
   calls out hand-synced contracts as this repo's standing risk, and a mismatch here
   is silent: it would surface as labels that look right and are not.

These run as part of the normal sidecar suite (pytest.ini puts `tools` on pythonpath),
so `make test` catches dataset drift without a separate invocation:

    sidecar/.venv/Scripts/python.exe -m pytest tests/test_dataset_tools.py -v

Nothing here touches the dataset workspace. The tools read it lazily, from inside
main(), which is what lets the pure functions be tested with the workspace absent.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

import audit_recall
import clean_v2
import generate_version
import generations
import label_classes
import label_progress
import plan_split
import resources
import train_model

# The default generation's artifacts, so the assertions below name exactly what the tool writes
# rather than a second copy of the naming rule - `generations.py` owns that, and the v1 half of
# it is covered by the tests at the end of this file.
V2 = generations.V2
RUN_NAME = V2.run_name
WEIGHT_NAME = V2.weight_name
VAL_NAME = train_model.val_name(V2)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "MODEL_TRAINING.md"
CHECKLIST = REPO_ROOT / "docs" / "CAPTURE_CHECKLIST.md"


def _batch(name: str, cls: str, distance: str, n: int, session: str = "s1") -> plan_split.Batch:
    return plan_split.Batch(
        name, cls, distance, session, [f"{name}_{i}.jpg" for i in range(n)]
    )


def _entries(batches: list[plan_split.Batch]) -> list[dict]:
    return [
        {"new_name": img, "class": b.cls, "distance": b.distance, "batch": b.name, "tags": []}
        for b in batches
        for img in b.images
    ]


# --------------------------------------------------------------------------
# 1. stratify
# --------------------------------------------------------------------------


def test_stratify_hits_target_proportions():
    batches = [_batch(f"c{d}", "cls", d, 100) for d in ("close", "mid", "far")]
    plan = plan_split.stratify(batches)
    total = sum(len(b.images) for b in batches)
    counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLITS}

    assert sum(counts.values()) == total
    assert counts["train"] / total == pytest.approx(0.70, abs=0.02)
    assert counts["valid"] / total == pytest.approx(0.20, abs=0.02)
    assert counts["test"] / total == pytest.approx(0.10, abs=0.02)


def test_stratify_gives_each_split_a_slice_of_tiny_cells():
    """A 5-image cell that sends nothing to test makes that cell unmeasurable there,
    which is the failure the whole plan exists to avoid."""
    plan = plan_split.stratify([_batch("tiny", "cls", "mid", 5)])
    counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLITS}
    assert counts == {"train": 3, "valid": 1, "test": 1}


def test_stratify_handles_a_two_image_cell():
    plan = plan_split.stratify([_batch("pair", "cls", "far", 2)])
    counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLITS}
    assert counts["train"] >= 1
    assert counts["valid"] == 1


def test_stratify_never_starves_train():
    """valid and test minimums must not eat the whole cell."""
    for n in range(2, 30):
        plan = plan_split.stratify([_batch("c", "cls", "close", n)])
        counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLITS}
        assert counts["train"] >= 1, f"train empty for cell of {n}"
        assert sum(counts.values()) == n


def test_stratify_is_deterministic():
    """`stratify` must not depend on PYTHONHASHSEED, or the plan is not a plan."""
    batches = [_batch(f"c{i}", "cls", "mid", 37) for i in range(3)]
    assert plan_split.stratify(batches) == plan_split.stratify(batches)


# --------------------------------------------------------------------------
# 2. evaluate (the anneal's objective)
# --------------------------------------------------------------------------


def test_evaluate_penalises_a_split_with_no_mid_images():
    batches = [_batch("a", "cls1", "close", 70), _batch("b", "cls2", "mid", 20), _batch("c", "cls3", "far", 10)]
    total = 100
    balanced, _ = plan_split.evaluate({"a": "train", "b": "valid", "c": "test"}, batches, total)
    # Put everything on train: right size, no distance coverage anywhere.
    lopsided, _ = plan_split.evaluate({"a": "train", "b": "train", "c": "train"}, batches, total)
    assert lopsided > balanced


def test_evaluate_prefers_the_target_proportions():
    batches = [_batch(f"b{i}", "cls", "close", 10) for i in range(10)]
    total = 100
    near, _ = plan_split.evaluate({f"b{i}": ("train" if i < 7 else "valid" if i < 9 else "test") for i in range(10)}, batches, total)
    off, _ = plan_split.evaluate({f"b{i}": ("train" if i < 3 else "valid") for i in range(10)}, batches, total)
    assert near < off


def test_anneal_is_reproducible_and_covers_every_batch():
    batches = [_batch(f"b{i}", f"cls{i % 3}", ("close", "mid", "far")[i % 3], 10 + i) for i in range(9)]
    total = sum(b.n for b in batches)
    first = plan_split.anneal(batches, total, iters=2_000)
    second = plan_split.anneal(batches, total, iters=2_000)
    assert first == second
    assert set(first) == {b.name for b in batches}
    assert all(v in plan_split.SPLITS for v in first.values())


# --------------------------------------------------------------------------
# 2b. capture sessions: the unit the leakage argument is really about
# --------------------------------------------------------------------------


def test_a_capture_session_split_across_splits_costs_more():
    """§8.3's rule protects against frames that share a rig state, a day and a lighting
    setup - and that is a property of the *session*, not of the cell. Two assignments with
    identical size, distance and class coverage must therefore rank by session integrity,
    or the search has no reason to hold a session out.
    """
    s1 = [_batch("milo_close", "milo", "close", 10, "s1"), _batch("milo_far", "milo", "far", 10, "s1")]
    s2 = [_batch("milo_close_s2", "milo", "close", 10, "s2"), _batch("milo_far_s2", "milo", "far", 10, "s2")]
    batches = s1 + s2
    total = sum(b.n for b in batches)

    # Same sizes (20 train / 20 test) and same distance+class coverage either way; the
    # only difference is whether each session stays in one split.
    interleaved = {
        "milo_close": "train",
        "milo_far": "test",
        "milo_close_s2": "test",
        "milo_far_s2": "train",
    }
    aligned = {
        "milo_close": "train",
        "milo_far": "train",
        "milo_close_s2": "test",
        "milo_far_s2": "test",
    }
    cost_bad, info_bad = plan_split.evaluate(interleaved, batches, total)
    cost_good, info_good = plan_split.evaluate(aligned, batches, total)

    assert info_bad["size"] == info_good["size"]  # the comparison is fair
    assert info_bad["dist"] == info_good["dist"]
    assert cost_good < cost_bad
    assert info_bad["session_splits"]["s1"] == {"train", "test"}
    assert info_good["session_splits"]["s1"] == {"train"}


def _session_penalty(info: dict) -> float:
    return plan_split.W_SESSION_SPLIT * sum(len(v) - 1 for v in info["session_splits"].values())


def test_one_session_pays_the_same_penalty_in_every_three_split_assignment():
    """The reason adding the session term did not move the already-applied plan, stated as
    a property rather than a hope: any assignment that uses all three splits pays the same
    session penalty, and the 70/20/10 size rule (weight 40, against 1.2 here) is what
    forces an assignment to use all three. So the term is constant over the region the
    optimum lives in, and cannot reorder it.

    If the weights are ever retuned, this is the test that says whether retuning moved the
    v2 split - which matters, because an image's split cannot be changed by re-uploading
    it, so moving the plan means wiping and re-uploading 1,383 images.
    """
    batches = [_batch(f"b{i}", f"cls{i % 3}", ("close", "mid", "far")[i % 3], 10 + i) for i in range(6)]
    total = sum(b.n for b in batches)
    rotate_a = {b.name: plan_split.SPLITS[i % 3] for i, b in enumerate(batches)}
    rotate_b = {b.name: plan_split.SPLITS[(i + 1) % 3] for i, b in enumerate(batches)}
    _cost_a, info_a = plan_split.evaluate(rotate_a, batches, total)
    _cost_b, info_b = plan_split.evaluate(rotate_b, batches, total)

    assert _session_penalty(info_a) == _session_penalty(info_b)
    assert _session_penalty(info_a) == 2 * plan_split.W_SESSION_SPLIT  # one session, 3 splits

    # An assignment can dodge the penalty completely (everything in train) - and it does
    # so by leaving valid and test empty, which is exactly what the size rule forbids. So
    # the term cannot be exploited: paying it is cheaper than the only way to avoid it.
    all_train = {b.name: "train" for b in batches}
    _cost_all_train, info_all_train = plan_split.evaluate(all_train, batches, total)
    assert _session_penalty(info_all_train) == 0
    assert info_all_train["size"]["valid"] == 0 and info_all_train["size"]["test"] == 0


def test_two_sessions_of_one_cell_are_two_batches_not_one(tmp_path):
    """The merge this guards against: grouping by the manifest's `batch` alone collapses
    `milo_mid` and `milo_mid_s2` into a single unit, which would let one capture session
    land in two splits while the planner thought it had honoured §8.3.
    """
    rows = [
        {"new_name": "milo_0001.jpg", "class": "milo", "distance": "mid", "batch": "milo_mid", "session": "s1", "tags": ["mid", "milo", "s1"]},
        {"new_name": "milo_0002.jpg", "class": "milo", "distance": "mid", "batch": "milo_mid", "session": "s2", "tags": ["mid", "milo", "s2"]},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    batches, entries = plan_split.load_batches(tmp_path)

    assert [b.name for b in batches] == ["milo_mid", "milo_mid_s2"]
    assert [b.session for b in batches] == ["s1", "s2"]
    assert all(b.n == 1 for b in batches)
    assert len(entries) == 2


def test_a_manifest_without_sessions_reads_as_the_first_capture(tmp_path):
    """Manifests written before the session was recorded describe the first capture. That
    is an assumption, so it is spelled out here rather than inferred from a missing key.
    """
    rows = [
        {"new_name": "milo_0001.jpg", "class": "milo", "distance": "mid", "batch": "milo_mid", "tags": ["mid", "milo"]},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    batches, _ = plan_split.load_batches(tmp_path)
    assert batches[0].session == plan_split.DEFAULT_SESSION
    # And the name is unsuffixed, matching what the uploader creates for session 1.
    assert batches[0].name == "milo_mid"


def test_the_report_says_when_a_single_session_makes_leakage_unavoidable():
    """The verdict is the deliverable: the numbers above it are read differently when
    train and test share a capture session, so the report has to say which it is."""
    batches = [_batch("milo_close", "milo", "close", 10, "s1"), _batch("milo_far", "milo", "far", 10, "s1")]
    entries = _entries(batches)
    plan = {n: ("train" if i % 2 else "test") for i, n in enumerate(e["new_name"] for e in entries)}
    text = plan_split.render("t", plan, batches, entries, "note")
    assert "Every batch is capture session s1" in text
    assert "Unavoidable with a single session" in text

    # Same shape, two sessions, and the verdict flips to naming the leak.
    split = [
        _batch("milo_close", "milo", "close", 10, "s1"),
        _batch("milo_far", "milo", "far", 10, "s2"),
    ]
    split_entries = _entries(split)
    split_plan = {e["new_name"]: ("train" if e["batch"] == "milo_close" else "test") for e in split_entries}
    text2 = plan_split.render("t", split_plan, split, split_entries, "note")
    assert "No session crosses a split" in text2
    assert "Unavoidable" not in text2


# --------------------------------------------------------------------------
# 3. labeling state detection
# --------------------------------------------------------------------------
# 3. Plan C: a whole capture session held out (the acceptance run)
# --------------------------------------------------------------------------


def _two_session_batches() -> list[plan_split.Batch]:
    """Two sessions over the same two cells - the shape Tier C shoots for."""
    return [
        _batch("milo_close", "milo", "close", 40, "s1"),
        _batch("milo_far", "milo", "far", 40, "s1"),
        _batch("milo_close_s3", "milo", "close", 20, "s3"),
        _batch("milo_far_s3", "milo", "far", 20, "s3"),
    ]


def test_holdout_puts_the_whole_session_in_test_and_none_of_it_anywhere_else():
    """Whole, not sampled. One held-out frame in train re-introduces the shared rig
    state, day and lighting that makes a same-session test optimistic - and it does it
    invisibly, since one frame looks like nothing in a 1,400-image set."""
    batches = _two_session_batches()
    plan = plan_split.holdout_plan(batches, "s3")
    held = {n for b in batches if b.session == "s3" for n in b.images}
    rest = {n for b in batches if b.session != "s3" for n in b.images}

    assert {plan[n] for n in held} == {"test"}
    assert {plan[n] for n in rest} <= {"train", "valid"}
    assert len(plan) == sum(b.n for b in batches)


def test_holdout_spends_the_test_slot_on_the_session_and_nothing_else():
    """A stray test frame from the train session is exactly the leak the plan exists to
    remove, so the remainder divides train/valid only."""
    batches = _two_session_batches()
    plan = plan_split.holdout_plan(batches, "s3")
    s = plan_split.summarize(plan, _entries(batches), batches)

    assert s["session_splits"]["s3"] == {"test"}
    assert s["size"]["test"] == 40
    assert s["size"]["train"] + s["size"]["valid"] == 80


def test_holdout_is_deterministic():
    """A plan that comes out different on every run is not a plan."""
    assert plan_split.holdout_plan(_two_session_batches(), "s3") == plan_split.holdout_plan(
        _two_session_batches(), "s3"
    )


def test_a_cell_with_no_test_share_gets_no_test_image():
    """The tiny-cell grant must not fire when the caller asked for no test frames - a
    6-image cell inventing one would put the train session's own frame on test."""
    assert plan_split._split_counts(6, 0.2, 0.0)[2] == 0
    assert plan_split._split_counts(6, 0.2, 0.1)[2] == 1


def test_holding_out_a_session_that_re_shoots_covered_cells_keeps_them_learnable():
    """The rule Tier C is built around: the held-out session must re-shoot cells an
    earlier session also covers. When it does, the pin holds by construction."""
    batches = _two_session_batches()
    plan = plan_split.holdout_plan(batches, "s3")
    assert plan_split.summarize(plan, _entries(batches), batches)["no_train_cells"] == []


def _tier_d_like(cells: dict[tuple[str, str], int]) -> clean_v2.TierPlan:
    """A held-out tier over hand-picked (folder name, distance) cells.

    Built rather than borrowed from `TIER_D_CELLS` so the cells in a test can be small
    enough to reason about; the real plan's own coverage is checked against the real staged
    sets by `test_the_scaffold_reads_the_staged_coverage_it_reports`.
    """
    return clean_v2.TierPlan(
        key="d", title="t", cells=cells, session="s3", note="", out="out", holdout=True
    )


def test_the_holdout_preflight_names_the_cells_the_planner_refuses_on(tmp_path):
    """The two must agree, and this is the assertion that makes them: the preflight is asked
    at scaffold time, weeks before the plan is run, and a preflight that disagrees with the
    refusal is worse than none - it would send a session to the camera and then fail.

    The rule, from `plan_split.summarize`: a cell has no train images when no image of it is
    in `train`. Under a held-out session that is exactly a cell with zero coverage outside
    the session, which is the zero the preflight tests.
    """
    outside = [
        _batch("milo_close", "milo", "close", 40, "s1"),
        _batch("milo_far", "milo", "far", 40, "s1"),
    ]
    held = [
        _batch("milo_close_s3", "milo", "close", 15, "s3"),
        _batch("milo_far_s3", "milo", "far", 15, "s3"),
        # This cell exists *only* in the held-out session: the mistake Tier D's rule names.
        _batch("safeguard_mid_s3", "safeguard", "mid", 15, "s3"),
    ]
    manifest = tmp_path / "cleaned-v2"
    manifest.mkdir()
    (manifest / "manifest.json").write_text(json.dumps(_entries(outside)), encoding="utf-8")

    plan = _tier_d_like({("MILO", "close"): 15, ("MILO", "far"): 15, ("SAFEGUARD", "mid"): 15})
    preflight = clean_v2.holdout_gaps(plan, clean_v2.staged_coverage([manifest]))

    batches = outside + held
    planner = plan_split.summarize(
        plan_split.holdout_plan(batches, "s3"), _entries(batches), batches
    )["no_train_cells"]

    assert preflight == planner == [("safeguard", "mid")]


def test_the_preflight_is_silent_when_the_cells_are_already_covered(tmp_path):
    """The false alarm this must never raise: every cell re-shot by the held-out session is
    also covered earlier, which is the whole design of the tier."""
    manifest = tmp_path / "cleaned-v2"
    manifest.mkdir()
    (manifest / "manifest.json").write_text(
        json.dumps(
            _entries(
                [
                    _batch("milo_close", "milo", "close", 40, "s1"),
                    _batch("milo_far", "milo", "far", 40, "s1"),
                ]
            )
        ),
        encoding="utf-8",
    )
    plan = _tier_d_like({("MILO", "close"): 15, ("MILO", "far"): 15})

    assert clean_v2.holdout_gaps(plan, clean_v2.staged_coverage([manifest])) == []


def test_a_tier_that_is_not_held_out_reports_no_gaps():
    """Tier A *is* the fix for these cells, so naming its own cells as unlearnable would be
    the check arguing with its own purpose."""
    assert clean_v2.holdout_gaps(clean_v2.TIER_A, {}) == []


def test_with_nothing_staged_every_held_out_cell_is_a_gap():
    """True, and the reason the printer has a separate "cannot tell" branch: 24 alarming
    lines on a project that has simply not been staged yet is how a real warning gets
    trained away."""
    gaps = clean_v2.holdout_gaps(clean_v2.TIER_D, {})

    assert len(gaps) == len(clean_v2.TIER_D.cells) == 24
    assert ("century-tuna", "mid") in gaps and ("milo", "close") in gaps


def test_the_three_cells_tier_a_has_not_shot_read_as_gaps_against_the_rest_of_the_grid():
    """Today's actual state, which is the print's whole purpose: the staged set covers 21 of
    the 24 cells, so a Tier D hold-out would make exactly these three unlearnable - and they
    are the three Tier A is already shooting. The next session has to land before `s3` can
    be held out.
    """
    coverage = {
        (clean_v2.CLASS_MAP[product], distance): 15
        for (product, distance) in clean_v2.TIER_D.cells
        if (product, distance)
        not in {("TUNA", "mid"), ("TUNA", "far"), ("PALMOLIVE", "close")}
    }

    assert sorted(clean_v2.holdout_gaps(clean_v2.TIER_D, coverage)) == [
        ("century-tuna", "far"),
        ("century-tuna", "mid"),
        ("palmolive", "close"),
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
    assert (root / "PALMOLIVE" / "CLOSE").is_dir()
    assert "held-out coverage check" in out
    # Named as the cells a hold-out would break on...
    assert "palmolive/close" in out and "century-tuna/mid" in out
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


# --------------------------------------------------------------------------
# 4. Version generation: the one-shot geometry settings
# --------------------------------------------------------------------------


def test_the_version_settings_are_v1s_settings():
    """Pinned against the version that actually shipped: `scanncart-grocery/1` records
    exactly this block, and it scored mAP 98.21. Changing any of it is a decision, not a
    tidy-up, so it has to be a deliberate edit here rather than a silent retune."""
    settings = generate_version.version_settings()
    assert settings == {
        "preprocessing": {
            "auto-orient": True,
            "resize": {"width": 640, "height": 640, "format": "Stretch to"},
        },
        "augmentation": {},
    }


def test_sanity_and_the_generator_agree_on_the_geometry():
    """One source of truth. `sanity` judges an existing version against this and
    `generate_version` sends it; two copies could disagree, and the failure would be a
    check that approves a version the generator would never produce."""
    assert clean_v2.EXPECTED_RESIZE == generate_version.EXPECTED_RESIZE
    assert generate_version.EXPECTED_SIZE == clean_v2.EXPECTED_RESIZE[0]


def test_a_matching_version_raises_nothing():
    assert generate_version.compare_preprocessing(generate_version.version_settings()["preprocessing"]) == []


def test_a_letterboxed_version_is_flagged_with_the_arithmetic():
    """The difference this catches is not cosmetic: at 1280x720 the x axis is the binding
    one, so `Fit within` gives back the same 0.5 scale horizontally but drops the vertical
    scale from 0.889 to 0.5 - every object at 1.8x fewer pixels, worst on the far cells."""
    problems = generate_version.compare_preprocessing(
        {"auto-orient": True, "resize": {"width": 640, "height": 640, "format": "Fit within"}}
    )
    assert len(problems) == 1
    assert "Fit within" in problems[0] and "padding" in problems[0]


def test_a_wrong_size_or_missing_auto_orient_is_flagged():
    problems = generate_version.compare_preprocessing(
        {"resize": {"width": 416, "height": 416, "format": "Stretch to"}}
    )
    assert any("auto-orient" in p for p in problems)
    assert sum("width" in p or "height" in p for p in problems) == 2


def test_a_version_with_no_preprocessing_is_reported_not_crashed():
    """A version generated by hand, or by an older tool, is the ordinary case - so this is
    total over its input rather than trusting the response's shape."""
    assert generate_version.compare_preprocessing(None)
    assert generate_version.compare_preprocessing([])
    assert generate_version.compare_preprocessing("Stretch to")


def test_filter_null_is_flagged_as_dropping_the_hard_negatives():
    """The dangerous step, named as such: null annotations ARE the hard-negative set, so
    filtering them removes the background frames the model was meant to learn from - and
    the count afterwards would still look plausible."""
    problems = generate_version.compare_preprocessing(
        {
            "auto-orient": True,
            "resize": {"width": 640, "height": 640, "format": "Stretch to"},
            "filter-null": {"percent": 50},
        }
    )
    assert len(problems) == 1
    assert "hard negatives" in problems[0]


def test_the_next_version_number_counts_a_trashed_version():
    """A trashed version still holds its number, which is why this project's next
    generation is 2 and why the weights are named v2 rather than v1."""
    assert generate_version.next_version({"versions": [{"id": "ws/proj/1"}]}) == 2
    assert generate_version.next_version({"versions": [{"id": "ws/proj/1"}, {"id": "ws/proj/3"}]}) == 4
    assert generate_version.next_version({"versions": []}) is None
    assert generate_version.next_version({}) is None


def test_only_class_names_that_carry_a_distance_are_returned():
    """The predicate half, and the false alarm it must never raise: the real roster - and
    `Palmolive` above all - has to come back empty.
    """
    roster = {name: i for i, name in enumerate(label_classes.SLUG_TO_CLASS.values())}
    found = generate_version.classes_with_distance(
        {**roster, "Palmolive Naturals Bar Soap 85g far": 9, "safeguard-mid": 10}
    )

    assert found == {"Palmolive Naturals Bar Soap 85g far": ["far"], "safeguard-mid": ["mid"]}
    assert generate_version.classes_with_distance(roster) == {}
    assert generate_version.classes_with_distance({}) == {}
    assert generate_version.classes_with_distance(None) == {}


class _Resp:
    """Just enough of an httpx response for `generate_version.main`."""

    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = {} if body is None else body

    def json(self):
        return self._body


class _FakeRoboflow:
    """One project read and one generate POST, which is all `--yes` does."""

    def __init__(self, project_body):
        self.project_body = project_body
        self.posts: list = []

    def get(self, url, **_kwargs):
        return _Resp(200, self.project_body)

    def post(self, url, **kwargs):
        self.posts.append(kwargs.get("json"))
        return _Resp(200, {"version": 2, "message": "building"})


def _generator(monkeypatch, classes: dict) -> _FakeRoboflow:
    fake = _FakeRoboflow({"project": {"classes": classes}, "versions": [{"id": "ws/proj/1"}]})
    monkeypatch.setattr(generate_version, "load_key", lambda _project: "k")
    monkeypatch.setattr(generate_version, "httpx", fake)
    return fake


def test_the_generator_refuses_a_class_list_with_a_distance_in_it(monkeypatch, capsys):
    """The whole point of the preflight: a version generated from a distance-split class list
    trains one head output per product-and-distance, and a version number cannot be reused -
    so this refuses and spends nothing.
    """
    fake = _generator(
        monkeypatch,
        {
            "Palmolive Naturals Bar Soap 85g close": 1,
            "Palmolive Naturals Bar Soap 85g mid": 2,
            "Palmolive Naturals Bar Soap 85g far": 3,
            "Milo Chocolate Drink 22g Sachet": 4,
        },
    )

    assert generate_version.main(["--yes", "--project", "snc-grocery"]) == 2
    # Nothing was sent, so nothing was spent - that is the difference between a refusal and
    # a warning printed after the fact.
    assert fake.posts == []
    out = capsys.readouterr().out
    assert "refusing to generate" in out
    assert "'Palmolive Naturals Bar Soap 85g mid'" in out
    assert "MOVE" in out and "no version number was spent" in out


def test_the_generator_still_generates_a_clean_class_list(monkeypatch, capsys):
    """The other half: a guard that also blocks the correct class list would be worse than
    none, so the roster has to go through untouched - same body as ever.
    """
    roster = {name: i for i, name in enumerate(label_classes.SLUG_TO_CLASS.values())}
    fake = _generator(monkeypatch, roster)

    assert generate_version.main(["--yes", "--project", "snc-grocery"]) == 0
    assert fake.posts == [generate_version.version_settings()]
    assert "generating version 2" in capsys.readouterr().out


def test_the_version_settings_are_documented_where_a_reader_will_look():
    """Drift guard: the checklist and MODEL_TRAINING both send a reader to the generator
    rather than to the UI, so the file has to exist under the name they use."""
    for doc in (CHECKLIST, DOC):
        text = doc.read_text(encoding="utf-8")
        assert "generate_version.py" in text, f"{doc.name} does not name the generator"


def test_hard_negatives_are_judged_from_the_project_not_the_manifest():
    """The bug this guards, with the real numbers it fired on: the negatives are their own
    set with their own manifest, so this manifest holds none *by design* - and counting it
    reported "staged: 0" while all 50 were uploaded, sending a reader off to re-shoot work
    that was already done. `sanity` needs the network, so the verdict is a pure function."""
    status, title, detail = clean_v2.hard_negative_verdict(on_server=50, staged_here=0)
    assert status == "ok"
    assert title == "hard negatives: 50 in the project"
    assert "expected" in detail and "null annotation" in detail

    # Nothing anywhere: this is a real gap, and the only case that warns.
    status, title, detail = clean_v2.hard_negative_verdict(on_server=0, staged_here=0)
    assert status == "warn"
    assert "none in the project" in detail

    # Staged but not uploaded is the other real gap, and it is not the same action as
    # "shoot them": the frames exist, so the fix is an upload rather than a camera.
    status, _title, _detail = clean_v2.hard_negative_verdict(on_server=0, staged_here=30)
    assert status == "warn"

    # Negatives in this manifest is unusual enough to say so, not to fail on.
    status, title, detail = clean_v2.hard_negative_verdict(on_server=50, staged_here=50)
    assert status == "ok" and "50 staged here" in title and "unusual" in detail


def test_the_holdout_plan_is_documented_in_the_checklist():
    """Drift guard. The Tier C shoot is described in CAPTURE_CHECKLIST.md and executed by
    `--holdout-session`; a reader who follows the document must land on a flag that
    exists, and the session it names must be the one the planner is told to hold out."""
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "--holdout-session" in text
    assert "plan_split.py" in text
    assert re.search(r"--holdout-session\s+s3", text)


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
# 4. drift guards: the class names live in three places that must agree
# --------------------------------------------------------------------------


def _doc_class_names() -> list[str]:
    """The §8.1 roster, read out of the shipped doc."""
    text = DOC.read_text(encoding="utf-8")
    section = text.split("### 8.1", 1)[1].split("### 8.2", 1)[0]
    names = []
    for line in section.splitlines():
        # | 1 | `The Class Name` | 310 | `slug` | 106 / 23 / 22 |
        m = re.match(r"^\|\s*\d+\s*\|\s*`([^`]+)`", line)
        if m:
            names.append(m.group(1))
    return names


def test_doc_has_a_class_table_to_read():
    assert len(_doc_class_names()) == 8, "expected the 8-row roster in MODEL_TRAINING.md section 8.1"


def test_label_classes_matches_the_documented_roster():
    """The mapping must be exactly the doc's roster - no more, no fewer."""
    assert set(label_classes.SLUG_TO_CLASS.values()) == set(_doc_class_names()), (
        "label_classes.SLUG_TO_CLASS drifted from the doc's roster"
    )


def test_mapped_class_names_are_real_names_not_placeholders():
    """A label is matched by exact string, so a placeholder in the mapping would create
    a class no annotation can ever match. Guards promoting `palmolive_<product>_<size>`.
    """
    for slug, name in label_classes.SLUG_TO_CLASS.items():
        assert "<" not in name and ">" not in name, f"{slug} is still a placeholder"
        assert name.strip() == name and name, f"{slug} has surrounding whitespace"


def test_every_class_slug_is_named():
    """A slug that is neither mapped nor declared reaches the images as a raw slug.

    Palmolive was the last placeholder and naming it is what emptied this dict, so an
    entry appearing here again means a class was staged without deciding its name.
    """
    assert label_classes.NEW_CLASS_SLUGS == {}, (
        "a class slug is declared but unnamed - put its real name in SLUG_TO_CLASS, or "
        "the images keep a slug no label can use"
    )


def test_clean_v2_class_map_agrees_with_label_classes():
    """clean_v2 decides the tags; label_classes decides the names. They key off the
    same slugs, so a slug renamed in one and not the other breaks continuity silently.

    `negative` is excluded deliberately: it is the §2 hard-negative *pseudo*-class, a
    tag and batch key with nothing to annotate, so it must not be a labelable class.
    """
    real_slugs = set(clean_v2.CLASS_MAP.values()) - label_classes.PSEUDO_CLASS_SLUGS
    assert real_slugs == set(label_classes.SLUG_TO_CLASS) | set(label_classes.NEW_CLASS_SLUGS)
    for pseudo in label_classes.PSEUDO_CLASS_SLUGS:
        assert pseudo not in label_classes.SLUG_TO_CLASS
        assert pseudo not in label_classes.NEW_CLASS_SLUGS


def test_the_hard_negative_pseudo_class_is_not_treated_as_a_labelable_slug():
    """`negative` is a tag and batch key with nothing to annotate. Without the exclusion,
    staging any hard negative makes label_classes report a false "unmapped slug" problem -
    and Tier C2b stages exactly those frames.
    """
    pseudo = clean_v2.CLASS_MAP["NEGATIVES"]
    assert pseudo in label_classes.PSEUDO_CLASS_SLUGS
    assert pseudo not in label_classes.SLUG_TO_CLASS
    assert pseudo not in label_classes.NEW_CLASS_SLUGS


def test_v2_only_classes_are_mapped_but_exempt_from_v1_continuity():
    """Palmolive is new, so v1 has no name to compare it against. The exemption has to be
    a subset of the mapping, or a typo here quietly drops a real class out of the check.
    """
    assert label_classes.V2_ONLY_SLUGS <= set(label_classes.SLUG_TO_CLASS)
    assert label_classes.V2_ONLY_SLUGS == {"palmolive"}


def test_class_seed_bytes_come_from_the_name_not_the_position():
    """Index-derived seeds repeat across runs, so adding one class to an already-seeded
    project re-uploads bytes the workspace has seen, gets back the old (annotated) asset
    from its SHA-256 dedup, and 409s - which is the ordinary "add the eighth class later"
    case, and is exactly how `--create-classes` failed once the project had 7.
    """
    assert label_classes._seed_jpeg("Bear Brand Fortified Powdered Milk 33g") != label_classes._seed_jpeg(
        "Palmolive Naturals Bar Soap 85g"
    )
    # Stable within a run and across processes - no hash()/PYTHONHASHSEED dependence.
    assert label_classes._seed_jpeg("MILO") == label_classes._seed_jpeg("MILO")
    # The retry nonce must actually change the bytes, or the retry is a no-op.
    assert label_classes._seed_jpeg("MILO") != label_classes._seed_jpeg("MILO", 1)


def test_palmolive_is_named_in_the_style_of_the_class_beside_it():
    """The one class with no v1 name to inherit now has a real one, shaped like
    `safeguard_pure_white_60g`: brand, variant, size.
    """
    name = label_classes.SLUG_TO_CLASS["palmolive"]
    assert name == "Palmolive Naturals Bar Soap 85g"
    assert name.startswith("Palmolive ") and name.endswith("85g")


def test_clean_v2_class_map_covers_every_source_folder_with_spaces():
    """The source folders are the real input contract - a rename there must not
    silently drop a class (ingest skips unmapped folders with only a report line).
    """
    for folder in ("BEARBRAND", "LUCKY ME", "MILO", "PALMOLIVE", "SAFEGUARD", "SARDINES", "Silver Swan", "TUNA"):
        assert folder in clean_v2.CLASS_MAP, f"source folder {folder!r} is unmapped"


def test_hard_negative_frames_get_no_invented_distance():
    """A null-annotated frame carries no box, so close/mid/far would be a value nobody
    observed - and it would surface as a real cell in the per-distance coverage report.
    """
    assert clean_v2.batch_name("negative", "") == "negative"
    assert clean_v2.tags_for("negative", "") == ["negative"]
    # The roster keeps distance-first order, because only tags[0] is sent at upload.
    assert clean_v2.tags_for("milo", "mid") == ["mid", "milo"]
    assert clean_v2.batch_name("milo", "mid") == "milo_mid"
    assert clean_v2.NEGATIVE_CLS == clean_v2.CLASS_MAP["NEGATIVES"]


def test_the_capture_session_is_a_tag_and_never_tag_zero():
    """The session has to be a tag because Roboflow's dataset search has no `batch:`
    filter - a batch can group images but cannot select them - and it has to come last
    because `upload_one` sends only `tags[0]`, which must stay the distance.
    """
    assert clean_v2.tags_for("milo", "mid", "s2") == ["mid", "milo", "s2"]
    assert clean_v2.tags_for("milo", "mid", "s2")[0] == "mid"
    # A frame with no distance still carries its session.
    assert clean_v2.tags_for("negative", "", "neg1") == ["negative", "neg1"]
    # Absent session: unchanged, so every pre-session call site keeps working.
    assert clean_v2.tags_for("milo", "mid") == ["mid", "milo"]


def test_tier_a_is_the_documented_eight_cells():
    cells = clean_v2.TIER_A_CELLS
    assert len(cells) == 8
    assert sum(cells.values()) == 253
    for product, distance in cells:
        # Every name must be one the ingest maps, or scaffold builds a tree it then skips.
        assert product in clean_v2.CLASS_MAP, f"{product!r} is not a CLASS_MAP folder"
        assert clean_v2.DISTANCE_MAP[distance.upper()] in ("close", "mid", "far")


# The Tier A table as an operator reads it: `| \`century-tuna\` | mid | 0 | **40** | ... |`.
# Anchored on the backticked slug so it cannot match the surrounding prose.
_TIER_A_ROW = re.compile(
    r"^\|\s*`(?P<slug>[a-z0-9-]+)`\s*\|\s*(?P<distance>close|mid|far)\s*\|"
    r"\s*\d+\s*\|\s*\*\*(?P<capture>\d+)\*\*\s*\|"
)
_TIER_A_TOTAL = re.compile(r"\*\*Tier A total: (\d+) images\.\*\*")


def _checklist_tier_a_section() -> str:
    text = CHECKLIST.read_text(encoding="utf-8")
    start = text.index("## Tier A")
    return text[start : text.index("## Tier B", start)]


def test_tier_a_table_in_the_checklist_matches_the_scaffold_cells():
    """Tier A's numbers live in two places that must agree: the table an operator shoots from,
    and `TIER_A_CELLS`, which `scaffold` turns into folders. A cell added to one and not the
    other is a session that gets skipped or has nowhere to land, and both fail silently -
    the same class of bug as the folder-name mapping, one level up.

    The table's `Have` column is deliberately not compared: it is a snapshot of the staged set
    and is *supposed* to move as images land, while the capture targets are the contract.
    """
    documented: dict[tuple[str, str], int] = {}
    for line in _checklist_tier_a_section().splitlines():
        match = _TIER_A_ROW.match(line)
        if match:
            documented[(match["slug"], match["distance"])] = int(match["capture"])

    expected = {
        (clean_v2.CLASS_MAP[product], distance): want
        for (product, distance), want in clean_v2.TIER_A_CELLS.items()
    }
    assert documented == expected, (
        "CAPTURE_CHECKLIST.md's Tier A table and clean_v2.TIER_A_CELLS disagree"
    )

    total = _TIER_A_TOTAL.search(_checklist_tier_a_section())
    assert total, "the Tier A section lost its stated total"
    assert int(total.group(1)) == sum(expected.values())


def test_the_progress_snapshot_takes_its_capture_plan_from_the_same_table():
    """The Admin Panel's capture gap and the folder skeleton must be the same plan.

    `label_progress.py` writes the Tier A targets into the snapshot the sidecar reads, so
    if it ever grew its own copy of them the panel would confidently point at cells that
    `scaffold` does not create - and the two would only disagree after a capture session.
    """
    assert label_progress.TIER_A_CELLS is clean_v2.TIER_A_CELLS
    assert label_progress.CLASS_MAP is clean_v2.CLASS_MAP


def test_the_scaffold_subcommand_is_wired_up(tmp_path):
    """`scaffold_tree` is only useful if the CLI reaches it. The checklist tells the operator
    to run this, so a renamed or unregistered subcommand is an instruction that errors out.
    """
    assert clean_v2.main(["scaffold", "--root", str(tmp_path), "--dry-run"]) == 0
    assert clean_v2.main(["scaffold", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "README.md").is_file()
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 5  # 8 cells, 5 products


def test_scaffold_builds_a_tree_the_ingest_accepts(tmp_path):
    """The round trip that matters. `ingest` skips an unrecognised product folder with only
    a report line, so a folder typed from the checklist ("Lucky Me") is a whole capture
    session thrown away - and you find out after shooting it.
    """
    made = clean_v2.scaffold_tree(clean_v2.TIER_A, tmp_path, current_total=1383)
    assert len(made) == 8

    report: list[str] = []
    assert clean_v2.ingest(tmp_path, report) == []  # empty cells: nothing to ingest yet
    assert not [line for line in report if "unmapped" in line], report
    # Each empty cell is named, which is how a partly-finished session gets noticed.
    assert len([line for line in report if "[gap]" in line]) == 8

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "253 images to shoot" in readme
    assert "from 1383 to 1636" in readme
    # The README must name the folders as they exist on disk, spaces and all.
    assert "`Silver Swan/` | `MID`" in readme and "`LUCKY ME/` | `MID`" in readme


def test_scaffold_omits_the_projection_when_the_staged_set_is_unknown(tmp_path):
    """A projected total baked into the file would quietly become wrong; better to say
    nothing than to state a number nobody measured."""
    clean_v2.scaffold_tree(clean_v2.TIER_A, tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "would take the staged set" not in readme


def test_scaffold_refuses_a_name_the_ingest_would_skip(tmp_path):
    bad_product = dataclasses.replace(clean_v2.TIER_A, cells={("Lucky Me", "mid"): 5})
    bad_distance = dataclasses.replace(clean_v2.TIER_A, cells={("MILO", "middleish"): 5})
    with pytest.raises(SystemExit):
        clean_v2.scaffold_tree(bad_product, tmp_path)
    with pytest.raises(SystemExit):
        clean_v2.scaffold_tree(bad_distance, tmp_path)


def test_tier_d_is_the_whole_grid_and_its_total_is_the_documented_one():
    """Derived, not hand-listed: Tier D is a re-shoot of cells earlier sessions cover, so
    every product x distance is in it by construction and there is no per-cell judgement
    that could disagree with the checklist's arithmetic."""
    products = [p for p in clean_v2.CLASS_MAP if p != "NEGATIVES"]
    assert len(products) == 8
    assert set(clean_v2.TIER_D_CELLS) == {
        (p, d) for p in products for d in clean_v2.DISTANCE_ORDER
    }
    assert len(clean_v2.TIER_D_CELLS) == 24
    assert set(clean_v2.TIER_D_CELLS.values()) == {clean_v2.TIER_D_PER_CELL}
    assert sum(clean_v2.TIER_D_CELLS.values()) == 360

    # And the checklist quotes the same total, so a retuned per-cell count cannot leave the
    # document telling a reader to shoot a different number than the scaffold lays out.
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "360" in text
    assert re.search(r"15 per cell", text)


def test_tier_d_scaffold_names_the_precondition_and_the_planning_order(tmp_path):
    """The two things that make this session different from every other: it must re-shoot
    only cells train already covers, and its split has to be planned between `clean` and
    `upload` - because a split cannot be changed by re-uploading an image afterwards."""
    made = clean_v2.scaffold_tree(clean_v2.TIER_D, tmp_path)
    assert len(made) == 24
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "360 images to shoot" in readme
    assert "Re-shoot only cells an earlier session already covers" in readme
    # `#` lines are comments inside the same bash block, so read the runnable ones.
    commands = [c for c in clean_v2.tier_commands(clean_v2.TIER_D, tmp_path) if not c.startswith("#")]
    # clean -> plan_split -> upload -> retag, in that order.
    assert len(commands) == 4
    assert " clean " in commands[0]
    assert "--holdout-session s3" in commands[1] and "--include" in commands[1]
    assert "--split-plan" in commands[2]
    assert " retag " in commands[3]
    assert "CANNOT hold out session s3" in readme  # the failure to look for

    # Tier A's session has no holdout step, for the same reason it has no acceptance role.
    a_commands = clean_v2.tier_commands(clean_v2.TIER_A, tmp_path)
    assert len(a_commands) == 3
    assert not any("--holdout-session" in c for c in a_commands)
    assert all("s2" in c for c in a_commands)


def test_the_readme_commands_are_pasteable_into_a_shell(tmp_path):
    """They are pasted, not read: the README's block is the whole runbook for a session that
    cost a day to shoot. On Windows `Path.__str__` renders `sidecar\data\...`, and bash eats
    that backslash as an escape - so `clean --src` resolves to a directory that does not
    exist, ingests nothing, and reports success over an empty staged set.
    """
    for plan in (clean_v2.TIER_A, clean_v2.TIER_D):
        for command in clean_v2.tier_commands(plan, tmp_path):
            assert "\\" not in command, command


def test_ingest_would_skip_the_camera_frames_so_negatives_needs_its_own_path(tmp_path):
    """The camera writes cam0_*.jpg flat into images/. Guards the reason --negatives
    exists at all: the product-tree walk has nothing to map that folder to.
    """
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "cam0_000000.jpg").write_bytes(b"\xff\xd8\xff")
    report: list[str] = []
    assert clean_v2.ingest(tmp_path, report) == []
    assert any("unmapped product folder" in line for line in report)


def test_ingest_negatives_walks_a_flat_folder_with_no_distance(tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(3):
        (tmp_path / "images" / f"cam0_{i:06d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([i]))
    (tmp_path / "images" / "notes.txt").write_text("not an image")
    (tmp_path / "images" / "empty.jpg").write_bytes(b"")

    report: list[str] = []
    found = clean_v2.ingest_negatives(tmp_path / "images", report)

    assert [f.name for f in found] == [f"cam0_{i:06d}.jpg" for i in range(3)]
    assert {f.cls for f in found} == {clean_v2.NEGATIVE_CLS}
    assert {f.distance for f in found} == {""}
    # Unusable files are reported rather than staged or silently dropped.
    assert any("zero-byte" in line for line in report)
    assert any("negatives" in line for line in report)


def test_distance_map_covers_every_spelling_the_sources_use():
    for spelling in ("CLOSE", "CLOSE-UP", "MID", "MIDDLE", "MID-SHOT", "FAR", "FAR-SHOT"):
        assert clean_v2.DISTANCE_MAP.get(spelling) in ("close", "mid", "far")
    assert clean_v2.DISTANCE_MAP["CLOSE-UP"] == "close"
    assert clean_v2.DISTANCE_MAP["MIDDLE"] == "mid"
    assert clean_v2.DISTANCE_MAP["FAR-SHOT"] == "far"


def test_sanity_expected_resize_matches_the_sidecar_default():
    """v2 is trained at whatever preprocessing the version used, but inference is pinned
    to settings.imgsz - so these two numbers must not drift apart.
    """
    settings = (REPO_ROOT / "sidecar" / "app" / "settings.py").read_text(encoding="utf-8")
    imgsz = re.search(r"^\s*imgsz:\s*int\s*=\s*(\d+)", settings, re.MULTILINE)
    assert imgsz, "could not find imgsz in settings.py"
    want_w, want_h, want_fmt = clean_v2.EXPECTED_RESIZE
    assert (want_w, want_h) == (int(imgsz.group(1)), int(imgsz.group(1)))
    assert want_fmt == "Stretch to", "the custom .onnx path resolves resize_mode 'auto' to stretch"


def test_v1_class_names_in_the_doc_are_the_ones_used_for_continuity():
    """The whole point of the roster: these are v1's names, not tidy substitutes."""
    doc = set(_doc_class_names())
    for name in label_classes.SLUG_TO_CLASS.values():
        assert name in doc
    # v1's names deliberately break Roboflow's "alphanumeric plus -" advice, because
    # 1,815 images are already labeled with them. Guard that we did not "fix" them.
    assert any(" " in n for n in label_classes.SLUG_TO_CLASS.values())


# --------------------------------------------------------------------------
# 9. train_model - the export check, the run's verdict, and the drop-in
# --------------------------------------------------------------------------


def _fake_export(
    root: Path,
    *,
    names: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    per_split: int = 3,
    nested: bool = False,
) -> Path:
    """A minimal stand-in for an unzipped YOLOv11 PyTorch export.

    `nested` reproduces the shape Roboflow's zip sometimes extracts into, where the
    split folders sit one level down - the reason `find_split_dirs` accepts both.
    """
    import yaml

    base = root / "export" if nested else root
    for split in splits:
        images = base / split / "images"
        images.mkdir(parents=True)
        for i in range(per_split):
            (images / f"{split}_{i}.jpg").write_bytes(b"\xff\xd8\xff")
    (base / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names or []),
                "names": names if names is not None else sorted(label_classes.SLUG_TO_CLASS.values()),
            }
        ),
        encoding="utf-8",
    )
    return root


def test_export_check_passes_on_the_roster(tmp_path):
    root = _fake_export(tmp_path / "export-v2")
    splits, problems = train_model.check_export(root)
    assert problems == []
    assert set(splits) == {"train", "valid", "test"}
    assert train_model.count_images(splits["train"]) == 3


def test_export_missing_a_class_is_a_problem_not_a_warning(tmp_path):
    """The failure this check exists for: a model trained against a short class list emits
    boxes under the wrong label with no error at all, so it has to stop before the GPU run.
    """
    names = sorted(label_classes.SLUG_TO_CLASS.values())[:-1]  # drop one
    root = _fake_export(tmp_path / "export-v2", names=names)
    _splits, problems = train_model.check_export(root)
    assert any("no class for" in p for p in problems)
    assert any(sorted(label_classes.SLUG_TO_CLASS.values())[-1] in p for p in problems)


def test_export_with_a_phantom_class_is_a_problem(tmp_path):
    root = _fake_export(tmp_path / "export-v2", names=["croutons"])
    _splits, problems = train_model.check_export(root)
    assert any("not v2 classes" in p for p in problems)


def test_export_check_reports_a_missing_split_and_an_empty_one(tmp_path):
    root = _fake_export(tmp_path / "export-v2", splits=("train", "valid"), per_split=0)
    _splits, problems = train_model.check_export(root)
    assert any("no test/images directory" in p for p in problems)
    assert any("train/images is empty" in p for p in problems)


def test_find_split_dirs_accepts_the_nested_extraction(tmp_path):
    nested = _fake_export(tmp_path / "nested", nested=True)
    assert set(train_model.find_split_dirs(nested)) == {"train", "valid", "test"}


def test_missing_export_explains_what_to_download(tmp_path, capsys):
    """The operator-facing half of the check: this is the state the repo is in today (no
    version 2 generated yet), so the message is what a reader meets first."""
    splits, problems = train_model.check_export(tmp_path / "nope")
    assert splits == {}
    assert any("no export at" in p for p in problems)

    code = train_model.main(["--export-dir", str(tmp_path / "nope")])
    out = capsys.readouterr().out
    assert code == 2
    assert "YOLOv11 PyTorch" in out  # named the way the version page names it
    assert "not at the .zip" in out


def test_cli_dry_run_checks_the_export_and_runs_nothing(tmp_path, capsys):
    """No --yes means no training and no file written outside the export, which is the
    only safe way to show a reviewer what the run would be."""
    root = _fake_export(tmp_path / "export-v2")
    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(tmp_path / "models"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Nothing ran" in out
    assert "yolo detect train" in out and f"model={train_model.BASE_MODEL}" in out
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "models").exists()


def test_cli_install_from_a_finished_run_copies_the_checkpoint(tmp_path, capsys):
    """The drop-in, at the CLI level: a completed run installs, and the resize_mode reminder
    comes with it - that field is what makes stretched training pay off."""
    root = _fake_export(tmp_path / "export-v2")
    run = tmp_path / "runs" / RUN_NAME
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"weights")
    (run / "results.csv").write_text(_RESULTS_CSV, encoding="utf-8")

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(tmp_path / "models"),
            "--run-dir",
            str(run),
            "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "models" / WEIGHT_NAME).read_bytes() == b"weights"
    assert "resize_mode: stretch" in out


def test_written_data_yaml_uses_absolute_paths_and_the_export_s_own_name_order(tmp_path):
    """The export's yaml is relative to wherever Roboflow expected the archive to be
    extracted. Rewriting it from absolute paths is what makes the run work from any cwd -
    and the *order* has to be the export's, because that is the model's output indexing.
    """
    import yaml

    roster = sorted(label_classes.SLUG_TO_CLASS.values())
    root = _fake_export(tmp_path / "export-v2", names=list(reversed(roster)))
    splits = train_model.find_split_dirs(root)
    written = train_model.write_data_yaml(root, splits)
    body = yaml.safe_load(written.read_text(encoding="utf-8"))

    assert body["names"] == list(reversed(roster))
    assert body["nc"] == len(roster)
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        assert Path(body[key]).is_absolute()
        assert Path(body[key]) == splits[split].resolve()
    # Written into the export, not over it: the original is the provenance record.
    assert written.name == "data.scanncart.yaml"


_RESULTS_CSV = """epoch, time, train/box_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
1, 12.0, 1.20, 0.700, 0.650, 0.710, 0.480
2, 24.0, 0.90, 0.870, 0.860, 0.890, 0.630
3, 36.0, 0.80, 0.890, 0.880, 0.910, 0.620
"""


def _results(tmp_path, text: str = _RESULTS_CSV) -> Path:
    path = tmp_path / "results.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_results_csv_columns_are_stripped_and_rows_parsed(tmp_path):
    rows = train_model.read_results(_results(tmp_path))
    assert len(rows) == 3
    assert rows[-1]["metrics/mAP50-95(B)"] == 0.620
    assert rows[0]["epoch"] == 1.0


def test_verdict_fails_on_the_final_epoch_even_when_an_earlier_one_passed(tmp_path):
    """patience means the last epoch need not be the best one; the verdict deliberately
    quotes the end of the run, and `best_epoch` is reported beside it so the two are read
    together rather than the better number being cherry-picked.
    """
    rows = train_model.read_results(_results(tmp_path))
    passed, failed = train_model.verdict(rows)
    assert any("mAP50-95" in f for f in failed)
    # mAP50 clears its target in the same row, so the two are decided independently - and
    # the lookup must not have answered mAP50 with mAP50-95's value.
    assert any(p.startswith("mAP50(B) 0.910") for p in passed)
    assert not any("mAP50(B)" in f for f in failed)
    # The log's epoch column is 1-based (ultralytics writes `self.epoch + 1`), so the
    # 0.630 row is epoch 2 and must be reported as 2, not silently rewritten to 1.
    assert train_model.best_epoch(rows) == (2, 0.630)


def test_verdict_passes_when_both_targets_are_met(tmp_path):
    text = _RESULTS_CSV.replace("0.910, 0.620", "0.940, 0.700")
    rows = train_model.read_results(_results(tmp_path, text))
    passed, failed = train_model.verdict(rows)
    assert failed == []
    assert len(passed) == 2


def test_verdict_reports_a_missing_metric_rather_than_assuming_it():
    rows = [{"epoch": 1.0, "metrics/mAP50(B)": 0.95}]
    _passed, failed = train_model.verdict(rows)
    assert any("not in the run log" in f for f in failed)


def test_metric_matches_either_the_suffixed_or_bare_column_name():
    assert train_model.metric({"metrics/mAP50": 0.5}, "metrics/mAP50(B)") == 0.5
    assert train_model.metric({"metrics/mAP50(B)": 0.6}, "metrics/mAP50(B)") == 0.6
    assert train_model.metric({"metrics/mAP50-95(B)": 0.4}, "metrics/mAP50(B)") is None


def test_find_best_prefers_best_over_last(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"last")
    assert train_model.find_best(tmp_path).name == "last.pt"
    (weights / "best.pt").write_bytes(b"best")
    assert train_model.find_best(tmp_path).name == "best.pt"
    assert train_model.find_best(tmp_path / "empty") is None


def test_run_dir_suffixes_instead_of_overwriting_an_earlier_run(tmp_path):
    assert train_model.run_dir(tmp_path, RUN_NAME) == tmp_path / RUN_NAME
    (tmp_path / RUN_NAME).mkdir()
    assert train_model.run_dir(tmp_path, RUN_NAME) == tmp_path / f"{RUN_NAME}-2"
    (tmp_path / f"{RUN_NAME}-2").mkdir()
    assert train_model.run_dir(tmp_path, RUN_NAME) == tmp_path / f"{RUN_NAME}-3"


def test_install_copies_and_refuses_to_clobber(tmp_path):
    """The picker is keyed by filename, so an overwrite silently replaces the model a
    running app is configured with - and there is no history to roll back to.
    """
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    models = tmp_path / "models"

    target = train_model.install(src, models)
    assert target == models / WEIGHT_NAME
    assert target.read_bytes() == b"weights"

    src.write_bytes(b"newer")
    with pytest.raises(SystemExit):
        train_model.install(src, models)
    assert target.read_bytes() == b"weights"

    assert train_model.install(src, models, force=True).read_bytes() == b"newer"


def test_the_recorded_requirement_is_the_one_the_version_geometry_implies():
    """The requirement is derived from the version's preprocessing, not written out twice:
    `Stretch to 640` is what the export was generated with, and `stretch` is what the sidecar
    setting has to be. If a future generation changes the format, the mapping is the thing
    that has to change with it - and `REQUIRED_RESIZE_MODE` going `None` is how that surfaces
    instead of a confidently wrong requirement landing in the record.
    """
    fmt = generate_version.PREPROCESSING["resize"]["format"]
    assert fmt in generate_version.RESIZE_MODE_BY_FORMAT
    assert generate_version.REQUIRED_RESIZE_MODE == "stretch"
    # The mapping's *values* are the sidecar's `resize_mode` vocabulary, so the app-side test
    # (`test_models.py`) is what checks the one it produces is a value the settings PATCH
    # would accept - the contract belongs at that boundary, not in this file.
    # Roboflow's "Fill within" scales *and crops*, which the sidecar cannot reproduce - so it
    # is absent rather than mapped to the nearest-sounding value.
    assert "Fill within" not in generate_version.RESIZE_MODE_BY_FORMAT


def test_install_writes_the_record_beside_the_weights(tmp_path):
    """The requirement travels with the file, not in a `models/`-global manifest: the weight
    is the thing that gets copied to another machine, and a manifest left behind would lose
    the requirement at exactly the moment it is needed.
    """
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    models = tmp_path / "models"

    target = train_model.install(
        src, models, record=train_model.weight_record(V2, 2, "snc-grocery")
    )

    record = json.loads((models / f"{target.stem}.json").read_text(encoding="utf-8"))
    # No `model` field on purpose: the filename is the model, and a record that repeated it
    # would be a second answer that a copied file could make wrong.
    assert "model" not in record
    assert record["generation"] == V2.name
    assert record["resize_mode"] == "stretch"
    assert record["source"] == "snc-grocery version 2"
    assert record["installed_at"]
    # Beside the weight, named after it - that is the pairing the sidecar reads.
    assert target.with_suffix(".json").is_file()


def test_a_record_is_optional_so_install_still_works_alone(tmp_path):
    """`install()` is also the plain "copy this checkpoint in" path, and it has to keep
    working without a record - the reader treats a missing one as an unknown requirement."""
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    target = train_model.install(src, tmp_path / "models")
    assert target.read_bytes() == b"weights"
    assert not target.with_suffix(".json").exists()


def test_cli_install_records_and_announces_the_resize_mode(tmp_path, capsys):
    """The record is only half of it: the operator still has to set the field, so the run says
    which value and where the panel will check it."""
    root = _fake_export(tmp_path / "export-v2")
    models = tmp_path / "models"
    _finished_run(tmp_path / "runs", RUN_NAME)

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(models),
            "--version",
            "2",
            "--install",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "resize_mode: stretch" in out
    assert f"{WEIGHT_NAME[:-3]}.json" in out
    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    assert record["source"] == "snc-grocery version 2"


def test_install_creates_the_models_directory(tmp_path):
    """There was no `sidecar/models/` in this checkout at all, which is how a finished
    `best.pt` ends up never becoming a selectable model."""
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    models = tmp_path / "absent" / "models"
    assert train_model.install(src, models).is_file()
    assert models.is_dir()


def test_run_command_matches_the_hyperparameters_the_doc_quotes():
    """One source of truth for the run. Retuning either the tool or MODEL_TRAINING.md 6
    alone leaves the doc describing a run nobody performs."""
    doc = DOC.read_text(encoding="utf-8")
    for key, value in (
        ("epochs", train_model.EPOCHS),
        ("imgsz", train_model.IMGSZ),
        ("batch", train_model.BATCH),
        ("patience", train_model.PATIENCE),
    ):
        assert f"{key}={value}" in doc, f"MODEL_TRAINING.md does not quote {key}={value}"
    assert train_model.BASE_MODEL in doc

    hyper = train_model.Hyper()
    line = train_model.command_line(Path("d.yaml"), Path("runs"), RUN_NAME, hyper, "0")
    for token in (
        f"model={train_model.BASE_MODEL}",
        f"epochs={train_model.EPOCHS}",
        f"imgsz={train_model.IMGSZ}",
        f"name={RUN_NAME}",
        "device=0",
        f"workers={hyper.workers}",
    ):
        assert token in line
    # The printed command and the call that runs must not be able to disagree.
    kwargs = train_model.train_kwargs(Path("d.yaml"), Path("runs"), RUN_NAME, hyper, "0")
    assert kwargs["epochs"] == train_model.EPOCHS and kwargs["name"] == RUN_NAME
    assert kwargs["device"] == "0"


class _Resp:
    """Just enough of an httpx.Response for the export path: a status, a JSON body, bytes."""

    def __init__(self, status_code=200, body=None, content=b""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.content = content
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def test_export_link_reads_only_a_ready_body():
    """202-with-progress and 200-with-link are told apart by the body, because the status
    code does not distinguish "accepted" from "ready" on its own."""
    assert train_model.export_link({"ready": False, "progress": 0.4}) is None
    assert train_model.export_link({}) is None
    assert train_model.export_link(None) is None
    assert train_model.export_link({"export": {"link": "https://x/y.zip"}}) == "https://x/y.zip"
    assert train_model.export_progress({"ready": False, "progress": 0.4}) == 0.4
    assert train_model.export_progress({}) is None


def _export_zip(tmp_path) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train/images/a.jpg", b"\xff\xd8\xff")
        zf.writestr("data.yaml", "nc: 8\nnames: [a, b, c, d, e, f, g, h]\n")
    return buf.getvalue()


def test_download_export_polls_until_the_link_appears(tmp_path):
    """Three answers in order: building, building, ready. Then the archive, extracted into
    the destination - which is what makes the whole chain runnable without a click."""
    calls: list[str] = []
    payload = _export_zip(tmp_path)

    def fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith(".zip"):
            return _Resp(200, content=payload)
        if len([c for c in calls if c == url]) < 3:
            return _Resp(202, {"ready": False, "progress": 0.5})
        return _Resp(200, {"export": {"link": "https://files/x.zip"}})

    slept: list[float] = []
    dest = tmp_path / "export-v2"
    train_model.download_export(
        "snc-grocery", 2, dest, "key", get=fake_get, sleep=slept.append
    )

    assert slept == [10, 10]
    assert (dest / "train" / "images" / "a.jpg").is_file()
    assert (dest / "data.yaml").is_file()
    # The whole URL, workspace included. Asserting only the project and version is what let a
    # shadowed `WORKSPACE` (the dataset *directory*) reach the live API as a path segment: the
    # request still carried "/snc-grocery/2/" while addressing `C:\codes\...\datasets`.
    assert calls[0] == (
        f"https://api.roboflow.com/{label_classes.WORKSPACE}/snc-grocery/2/{train_model.EXPORT_FORMAT}"
    )
    assert str(train_model.DATASET_ROOT) != label_classes.WORKSPACE
    assert ":\\" not in calls[0]  # no filesystem path leaked into the URL


def test_download_export_refuses_a_bad_request_instead_of_looping(tmp_path):
    """A wrong --format or version is an error body, not a 202 that polls for 15 minutes."""
    def fake_get(url, **kwargs):
        return _Resp(404, {"error": "format not available"})

    with pytest.raises(SystemExit):
        train_model.download_export("p", 2, tmp_path / "d", "k", get=fake_get, sleep=lambda _s: None)


def test_download_export_gives_up_on_a_stuck_export(tmp_path):
    with pytest.raises(SystemExit):
        train_model.download_export(
            "p", 2, tmp_path / "d", "k",
            get=lambda url, **kw: _Resp(202, {"ready": False, "progress": 0.1}),
            sleep=lambda _s: None,
            timeout_s=0,
        )


def test_extract_zip_refuses_a_member_that_escapes_the_destination(tmp_path):
    """A zip is an untrusted input; `../../` in a member name writes outside the directory
    it was extracted into."""
    import zipfile

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(SystemExit):
        train_model.extract_zip(archive, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()


def test_cli_skips_a_download_that_is_already_there(tmp_path, capsys):
    root = _fake_export(tmp_path / "export-v2")
    code = train_model.main([
        "--export-dir", str(root), "--download", "--version", "2",
        "--run-project", str(tmp_path / "runs"), "--models-dir", str(tmp_path / "models"),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "already downloaded" in out


def test_cli_download_without_a_version_says_which_flag_is_missing(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        train_model.main(["--export-dir", str(tmp_path / "x"), "--download"])
    assert "--version" in str(excinfo.value)


def test_train_model_does_not_import_ultralytics_at_module_level():
    """Same shape as app/hardware.py's lazy torch import: the tool has to be importable,
    and its pure functions testable, without torch on the path."""
    source = Path(train_model.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^(from|import)\s+ultralytics", source, re.MULTILINE)
    assert re.search(r"^\s+from ultralytics import", source, re.MULTILINE)


# --- --val: the per-class number, and the two ways to get it wrong ---------------------


class _FakeBox:
    """The subset of ultralytics' `Metric` the recall report reads.

    `r` is positional against `ap_class_index` in the real object (metrics.py builds both
    from `unique_classes`, which is derived from the split's *labels*), and that is the
    detail worth reproducing here exactly.
    """

    def __init__(self, index, recall, map50=0.0, map_=0.0, mp=0.0, mr=0.0):
        self.ap_class_index = index
        self.r = recall
        self.map50 = map50
        self.map = map_
        self.mp = mp
        self.mr = mr


class _FakeMetrics:
    def __init__(self, names, box, counts=None):
        self.names = names
        self.box = box
        self.nt_per_class = counts if counts is not None else [10] * len(names)


_NAMES = {0: "bear-brand", 1: "century-tuna", 2: "lucky-me", 3: "milo"}


def test_per_class_recall_reads_r_positionally_against_ap_class_index():
    """The bug this test exists for, in one line: `r[i]` belongs to `ap_class_index[i]`.

    Classes 1 and 2 have no ground truth in the split, so ultralytics omits them - and a
    report that zipped `names` against `box.r` would put 0.95 (milo) on century-tuna and
    then call the whole thing a pass. Nothing about that output looks wrong.
    """
    box = _FakeBox(index=[0, 3], recall=[0.90, 0.95])
    metrics = _FakeMetrics(_NAMES, box, counts=[10, 0, 0, 12])

    rows = train_model.per_class_recall(metrics)
    assert [(name, recall) for name, recall, _n in rows] == [
        ("bear-brand", 0.90),
        ("century-tuna", None),
        ("lucky-me", None),
        ("milo", 0.95),
    ]
    assert [n for _name, _r, n in rows] == [10, 0, 0, 12]


def test_a_single_scored_class_is_not_dropped_by_numpy_truthiness():
    """The live probe's find, pinned. Ultralytics answers with numpy arrays, and `box.r` for
    a split whose only scored class recalls 0.0 is `array([0.0])` - which is *falsy* because
    a one-element array is truthy by its contents. `list(box.r or [])` thus reported
    "nothing was measured" for a class that had been measured and scored zero: the loudest
    result this step can produce, discarded quietly. A fake built from Python lists cannot
    catch this - `[0.0]` is truthy - which is exactly why the probe was worth running.
    """
    np = pytest.importorskip("numpy")
    box = _FakeBox(index=np.array([0]), recall=np.array([0.0]))
    metrics = _FakeMetrics({0: "bear-brand"}, box, counts=np.array([10]))

    rows = train_model.per_class_recall(metrics)
    assert rows == [("bear-brand", 0.0, 10)]

    passed, failed, unmeasured, values = train_model.recall_report(rows)
    assert failed == ["bear-brand 0.000 < 0.85 (n=10)"]
    assert passed == [] and unmeasured == []
    assert values == {"bear-brand": 0.0}


def test_a_class_the_split_never_asked_about_is_not_a_zero():
    """A missing class has no recall, and 0.0 is not the same answer: one means go capture
    images for the split, the other means go capture images for the class."""
    box = _FakeBox(index=[0, 3], recall=[0.90, 0.95])
    metrics = _FakeMetrics(_NAMES, box, counts=[10, 0, 0, 12])
    passed, failed, unmeasured, values = train_model.recall_report(train_model.per_class_recall(metrics))

    assert passed == ["bear-brand 0.900 >= 0.85 (n=10)", "milo 0.950 >= 0.85 (n=12)"]
    assert failed == []
    assert "century-tuna" not in values and "lucky-me" not in values
    assert len(unmeasured) == 2
    assert all("no ground-truth instances" in line for line in unmeasured)


def test_recall_report_puts_a_measured_miss_under_the_floor():
    """A class the model predicted badly *was* measured, so it fails rather than skipping -
    even though it is the near-empty cell you would most expect to read as unmeasured."""
    box = _FakeBox(index=[0, 1], recall=[0.62, 0.99], map50=0.8, map_=0.6, mp=0.8, mr=0.7)
    metrics = _FakeMetrics(_NAMES, box, counts=[4, 30, 0, 0])
    passed, failed, unmeasured, values = train_model.recall_report(train_model.per_class_recall(metrics))

    assert failed == ["bear-brand 0.620 < 0.85 (n=4)"]
    assert passed == ["century-tuna 0.990 >= 0.85 (n=30)"]
    assert len(unmeasured) == 2
    assert values["bear-brand"] == 0.62


def test_the_floor_is_the_one_the_doc_states():
    """6 sets 0.85 for *every* class. The number lives in code and in the doc, and only one
    of them can be edited silently."""
    doc = DOC.read_text(encoding="utf-8")
    assert f"≥ {train_model.RECALL_FLOOR:.2f} for *every* class" in doc


def _val_block(split="test", recall=0.9, weights_hash="hash-a", floor=0.85):
    return train_model.validation_record(
        [("bear-brand", recall, 10), ("lucky-me", None, 0)],
        split,
        {"mAP50": 0.88},
        floor,
        weights_hash,
    )


def test_the_measurement_carries_the_split_and_the_floor_it_was_judged_against(tmp_path):
    """A recall on its own is not a result: `test` is the acceptance split and `valid` is the
    one training selected on, and the floor is what turns a number into a verdict. Both travel
    with the numbers, because whatever reads them later was not there when they were made.
    """
    block = _val_block()
    assert block["split"] == "test"
    assert block["floor"] == 0.85
    assert block["weights_sha256"] == "hash-a"
    assert [c["name"] for c in block["per_class"]] == ["bear-brand", "lucky-me"]
    # `None`, not 0.0: the class the split never asked about is recorded as unmeasured, so
    # the distinction survives into a file that outlives the terminal it was printed to.
    assert block["per_class"][1]["recall"] is None


def test_a_measurement_of_other_weights_is_not_attached(tmp_path):
    """The reason the hash is here. `--val` and `--install` are separate commands, and
    re-training into the same run directory replaces `best.pt` in place - so a measurement of
    the previous checkpoint is the ordinary case, not the exotic one, and attaching it would
    show an operator a score these weights never got.
    """
    path = tmp_path / train_model.VAL_METRICS_NAME
    train_model.write_val_metrics(path, _val_block(weights_hash="old-weights"))

    assert train_model.load_val_metrics(path, "old-weights")[0]["split"] == "test"
    assert train_model.load_val_metrics(path, "new-weights") == []
    # A missing or unreadable file is the same answer, for the same reason.
    assert train_model.load_val_metrics(tmp_path / "nope.json", "old-weights") == []
    (tmp_path / "corrupt.json").write_text("not json", encoding="utf-8")
    assert train_model.load_val_metrics(tmp_path / "corrupt.json", "old-weights") == []


def test_measuring_the_other_split_keeps_the_one_already_there(tmp_path):
    """`--val --split valid` after `--val` would otherwise delete the acceptance number in
    favour of the selection one - and the selection number is the flattering one, so losing
    the other would be the quietest possible way to overstate a model. `test` stays first.
    """
    path = tmp_path / train_model.VAL_METRICS_NAME
    train_model.write_val_metrics(path, _val_block(split="valid", recall=0.99))
    train_model.write_val_metrics(path, _val_block(split="test", recall=0.62))

    kept = train_model.load_val_metrics(path, "hash-a")
    assert [b["split"] for b in kept] == ["test", "valid"]
    assert kept[0]["per_class"][0]["recall"] == 0.62
    # Re-measuring a split replaces it rather than stacking a second opinion.
    train_model.write_val_metrics(path, _val_block(weights_hash="hash-b"))
    assert [b["split"] for b in train_model.load_val_metrics(path, "hash-b")] == ["test"]


def test_the_recorded_measurement_has_no_hash_and_no_missing_field(tmp_path):
    """Two deliberate omissions. The hash is stripped: the record sits beside the weight it
    describes, so a copy inside it is implied by its location and a `--force` replacement of
    the `.pt` alone would leave it wrong. And `validation` is absent - not null - when there
    are no numbers, so a re-install writes a byte-identical file.
    """
    with_numbers = train_model.weight_record(V2, 2, "snc-grocery", validation=[_val_block()])
    (block,) = with_numbers["validation"]
    assert "weights_sha256" not in block
    assert block["floor"] == 0.85 and block["split"] == "test"

    without = train_model.weight_record(V2, 2, "snc-grocery")
    assert "validation" not in without
    assert train_model.weight_record(V2, 2, "snc-grocery", validation=[]) == without


def test_the_record_carries_the_class_list_the_export_declared():
    """The second fact only this run knows. A `.pt` stores the training run, not its label set,
    so a model trained from a distance-split project predicts three labels per product and
    nothing about the file says so - writing the list down is what lets the app's listing call
    that out before the weights are ever run.
    """
    names = [f"Palmolive Naturals Bar Soap 85g {d}" for d in ("close", "mid", "far")]
    record = train_model.weight_record(V2, 2, "snc-grocery", class_names=names)
    assert record["class_names"] == names

    # Absent, not empty, when the export declared none: an empty list would read as "this model
    # predicts nothing" instead of "not recorded", and those are opposite instructions.
    assert "class_names" not in train_model.weight_record(V2, 2, "snc-grocery")
    assert "class_names" not in train_model.weight_record(V2, 2, "snc-grocery", class_names=[])


def test_the_runbook_and_the_checklist_quote_the_val_step():
    """The floor and the command that measures it have to appear together. A per-class
    number that lives only in the tool is a number nobody is asked for."""
    for doc in (DOC, CHECKLIST):
        text = doc.read_text(encoding="utf-8")
        assert "train_model.py --val" in text, f"{doc.name} does not quote the --val step"


def test_the_default_split_is_the_acceptance_one():
    """`test`, not `val`: the validation split is what training selected on, so quoting it
    back would present the selection number as an acceptance one."""
    assert train_model.DEFAULT_SPLIT == "test"
    assert train_model.DEFAULT_SPLIT in train_model.VALIDATION_SPLITS
    # The yaml key is `val` (write_data_yaml maps valid -> val), so `valid` would be rejected
    # by ultralytics with a FileNotFoundError rather than silently measured.
    assert "valid" not in train_model.VALIDATION_SPLITS
    kw = train_model.validation_kwargs(Path("d.yaml"), "test", Path("runs"), VAL_NAME)
    assert kw["split"] == "test" and kw["imgsz"] == train_model.IMGSZ
    assert kw["name"] == VAL_NAME and kw["exist_ok"] is True


def test_aggregate_metrics_reports_only_what_the_pass_produced():
    box = _FakeBox(index=[0], recall=[0.9], map50=0.91, map_=0.62, mp=0.88, mr=0.86)
    assert train_model.aggregate_metrics(_FakeMetrics(_NAMES, box)) == {
        "precision": 0.88,
        "recall": 0.86,
        "mAP50": 0.91,
        "mAP50-95": 0.62,
    }
    # A pass that answered nothing is not a pass with zeros in it.
    assert train_model.aggregate_metrics(object()) == {}
    assert train_model.per_class_recall(object()) == []


def test_validate_passes_the_split_and_data_yaml_to_the_injected_loader(tmp_path):
    """`yolo` is injectable for the same reason `download_export`'s `get`/`sleep` are: the
    pass needs a GPU, and what is worth pinning is the arguments and the answer."""
    calls: list = []
    metrics = _FakeMetrics(_NAMES, _FakeBox(index=[0, 1], recall=[0.9, 0.88]))

    class _Model:
        def __init__(self, weights):
            calls.append(("load", weights))

        def val(self, **kwargs):
            calls.append(("val", kwargs))
            return metrics

    data_yaml = tmp_path / "data.scanncart.yaml"
    returned = train_model.validate(
        tmp_path / "weights" / "best.pt",
        data_yaml,
        "test",
        tmp_path / "runs",
        VAL_NAME,
        yolo=_Model,
    )

    assert returned is metrics
    assert calls[0] == ("load", str(tmp_path / "weights" / "best.pt"))
    kind, kwargs = calls[1]
    assert kind == "val"
    assert kwargs["split"] == "test"
    assert kwargs["data"] == str(data_yaml)
    assert kwargs["imgsz"] == train_model.IMGSZ


def _finished_run(root: Path, name: str, *, mtime: float | None = None) -> Path:
    run = root / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"weights")
    (run / "results.csv").write_text(_RESULTS_CSV, encoding="utf-8")
    if mtime is not None:
        import os

        os.utime(run, (mtime, mtime))
    return run


def test_latest_run_finds_a_finished_run_where_run_dir_would_name_the_next_one(tmp_path):
    """`run_dir` answers "where would the *next* run go" - the opposite of what a bare
    `--val`/`--install` needs, which used to look for `<name>-2` and fail."""
    runs = tmp_path / "runs"
    runs.mkdir()
    assert train_model.latest_run(runs, RUN_NAME) is None
    assert train_model.resolve_run(runs, V2) == runs / RUN_NAME

    first = _finished_run(runs, RUN_NAME, mtime=1_000)
    second = _finished_run(runs, f"{RUN_NAME}-2", mtime=2_000)
    # The val output is not a run: it has no weights, so it cannot be installed or measured.
    (runs / VAL_NAME / "weights").mkdir(parents=True)

    assert train_model.latest_run(runs, RUN_NAME, VAL_NAME) == second
    # Newest, not highest-numbered: lexically `-9` sorts above `-2` and `-10` below it, so the
    # suffix cannot be ranked as a number without parsing it back out of the name.
    _finished_run(runs, f"{RUN_NAME}-9", mtime=1_500)
    assert train_model.latest_run(runs, RUN_NAME, VAL_NAME) == second
    assert train_model.resolve_run(runs, V2, str(first)) == first
    # Training still asks run_dir for the next free name, gap-filling included - that contract
    # is unchanged.
    assert train_model.resolve_run(runs, V2, training=True) == runs / f"{RUN_NAME}-3"


def test_cli_val_reports_recall_by_distance_and_writes_it_into_the_record(tmp_path, capsys):
    """The whole step, end to end: the split's number, the same recall split by distance, and
    the breakdown landing in the file `--install` carries into the weights' record.

    The fake answers differently for the `far` pass, which is what a real model does and what
    the feature exists to expose - a class that passes the split while missing most items at
    distance. Written before, this would be a stored overall number and a `far` row describing
    two different runs.
    """
    root = _export_with_names(
        tmp_path / "export",
        {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg", "middle.jpg", "edge.jpg"]},
    )
    manifest = _manifest_with_distances(
        tmp_path, [("near.jpg", "close"), ("middle.jpg", "mid"), ("edge.jpg", "far")]
    )
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)
    calls: list[str] = []

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            if kwargs["name"].endswith("-far"):
                return _FakeMetrics(
                    _NAMES, _FakeBox(index=[0], recall=[0.62], map50=0.4), counts=[10, 0, 0, 0]
                )
            return _FakeMetrics(
                _NAMES, _FakeBox(index=[0], recall=[0.95], map50=0.9), counts=[10, 0, 0, 0]
            )

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--manifest",
            str(manifest),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert calls == [
        VAL_NAME,
        f"{VAL_NAME}-close",
        f"{VAL_NAME}-mid",
        f"{VAL_NAME}-far",
    ]
    assert "the test split by distance (floor 0.85):" in out
    assert "! = below the floor" in out
    assert "below the floor at a distance: far bear-brand 0.620 (n=10)" in out

    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert [entry["distance"] for entry in block["per_distance"]] == ["close", "mid", "far"]
    assert [entry["images"] for entry in block["per_distance"]] == [1, 1, 1]
    far = block["per_distance"][2]
    assert (far["per_class"][0]["name"], far["per_class"][0]["recall"]) == ("bear-brand", 0.62)
    # The floor is on the block, not repeated per distance: the breakdown slices *this*
    # measurement, so a copy per distance could only disagree with the one they were judged on.
    assert "floor" not in far and block["floor"] == train_model.RECALL_FLOOR

    # And the same file read by the sidecar's own reader, which is what the panel uses - the
    # writer and the reader asserted against each other rather than against a fixture.
    from app.models import read_record

    record = train_model.weight_record(V2, 2, "snc-grocery", validation=[block])
    (models_dir := tmp_path / "models-with-record").mkdir()
    weights = models_dir / WEIGHT_NAME
    weights.write_bytes(b"weights")
    weights.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")

    (parsed,) = read_record(weights)["validation"]
    assert [d.distance for d in parsed.per_distance] == ["close", "mid", "far"]
    assert parsed.per_distance[2].per_class[0].recall == 0.62
    assert parsed.per_distance[2].images == 1


def test_cli_no_per_distance_skips_the_extra_passes(tmp_path, capsys):
    """Three extra passes are cheap on a test split and not free on a whole one - and never
    worth running when the answer is not wanted. The record then simply has no breakdown, which
    is the same shape an older record has."""
    root = _export_with_names(
        tmp_path / "export", {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg"]}
    )
    manifest = _manifest_with_distances(tmp_path, [("near.jpg", "close")])
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)
    calls: list[str] = []

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            return _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.95]), counts=[10, 0, 0, 0])

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(tmp_path / "models"), "--manifest", str(manifest),
            "--no-per-distance", "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0 and calls == [VAL_NAME]
    assert "by distance" not in out
    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert block["per_distance"] == []


def test_cli_val_without_a_usable_manifest_says_the_breakdown_was_skipped(tmp_path, capsys):
    """A missing or unmatched manifest is one of the two ways `--val` on this machine cannot
    split by distance (an export whose images no distance accounts for is the other). Either
    way the run must say it skipped, because a section that simply does not appear reads as
    "every distance passed"."""
    root = _export_with_names(
        tmp_path / "export", {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg"]}
    )
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **_kwargs):
            return _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.95]), counts=[10, 0, 0, 0])

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(tmp_path / "models"),
            "--manifest", str(tmp_path / "nowhere" / "manifest.json"), "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "note: no distances for the test split" in out
    assert "skipped, not reported as clean" in out
    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert block["per_distance"] == []


def test_cli_val_reports_per_class_recall_without_being_told_the_run(tmp_path, capsys):
    """The whole step at the CLI level: no `--run-dir`, because that is the form the docs
    quote - the run it just trained, found by being the newest one with weights."""
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    _finished_run(runs, f"{RUN_NAME}-2")

    metrics = _FakeMetrics(
        _NAMES,
        _FakeBox(index=[0, 1, 3], recall=[0.90, 0.62, 0.95], map50=0.88, map_=0.61, mp=0.9, mr=0.82),
        counts=[10, 30, 0, 12],
    )
    loaded: list[str] = []

    class _Model:
        def __init__(self, weights):
            loaded.append(weights)

        def val(self, **kwargs):
            assert kwargs["split"] == "test"
            return metrics

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            # An explicit manifest, so this test's outcome cannot depend on the dataset workspace
            # on the machine running it: no distances means the breakdown is skipped, which is
            # the state this test is about.
            "--manifest",
            str(tmp_path / "no-manifest.json"),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert loaded == [str(runs / f"{RUN_NAME}-2" / "weights" / "best.pt")]
    assert "[.ok.] bear-brand 0.900 >= 0.85 (n=10)" in out
    assert "[.ok.] milo 0.950 >= 0.85 (n=12)" in out
    assert "[WARN] century-tuna 0.620 < 0.85 (n=30)" in out
    assert "[SKIP] lucky-me: no ground-truth instances in the split" in out
    assert "target >= 0.90" in out  # the mAP50 row keeps 6's aggregate target
    assert "below the 0.85 floor" in out
    assert f"--install --generation {V2.name} --run-dir" in out
    # --val alone installs nothing.
    assert not (tmp_path / "models").exists()


def test_cli_the_numbers_from_val_reach_the_record_via_a_separate_install(tmp_path, capsys):
    """The documented sequence, run the way the docs run it: `--val` in one command,
    `--install` in another. The numbers therefore travel through a file, and this is the test
    that would fail if either end of that file stopped agreeing with the other - which would
    leave the panel showing nothing while the terminal had printed a full table.
    """
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    run = _finished_run(runs, RUN_NAME)

    metrics = _FakeMetrics(
        _NAMES,
        _FakeBox(index=[0, 1], recall=[0.62, 0.99], map50=0.88, map_=0.61, mp=0.9, mr=0.82),
        counts=[30, 12, 0, 0],
    )

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **_kwargs):
            return metrics

    measured = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models),
            "--manifest", str(tmp_path / "no-manifest.json"), "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out
    assert measured == 0
    assert str(run / train_model.VAL_METRICS_NAME) in out
    # --val alone still installs nothing: the numbers are written, not the model.
    assert not models.exists()

    installed = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models), "--install", "--version", "2",
        ],
        yolo=lambda *_a, **_k: pytest.fail("an install must not need the model runtime"),
    )
    out = capsys.readouterr().out
    assert installed == 0

    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    # The export's own class list, read off its data.yaml at this boundary rather than passed
    # in by the test: the wiring is the half that could go missing while every unit below it
    # still passes, and its absence is invisible until a bad model runs unnoticed.
    assert record["class_names"] == sorted(label_classes.SLUG_TO_CLASS.values())
    (block,) = record["validation"]
    assert block["split"] == "test" and block["floor"] == train_model.RECALL_FLOOR
    assert [(c["name"], c["recall"]) for c in block["per_class"]] == [
        ("bear-brand", 0.62),
        ("century-tuna", 0.99),
        ("lucky-me", None),
        ("milo", None),
    ]
    # `--install` reports what it picked up, so an install without numbers is visible at the
    # moment it happens rather than only when someone opens the panel.
    assert "test: 2/4 classes measured, 1 below the 0.85 floor (bear-brand)" in out
    # And the same record read by the sidecar's own reader, which is what the panel uses -
    # the writer and the reader asserted against each other, not against a fixture.
    from app.models import read_record

    (parsed,) = read_record(models / WEIGHT_NAME)["validation"]
    assert parsed.split == "test"
    assert [c.name for c in parsed.per_class if c.recall is None] == ["lucky-me", "milo"]


def test_cli_install_says_so_when_nothing_was_measured(tmp_path, capsys):
    """Silence would read as "the numbers are there and the panel will show them". The one
    thing this step must not do is leave the operator believing a score was recorded.
    """
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    _finished_run(runs, RUN_NAME)

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models), "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No measurement of *these* weights was found" in out
    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    assert "validation" not in record


def test_cli_a_measurement_of_a_replaced_checkpoint_is_not_installed(tmp_path, capsys):
    """The hash, at the CLI level. A `--val` of the previous checkpoint sits in the same run
    directory; re-training replaces `best.pt` in place. If the numbers were attached anyway,
    the panel would show a score these weights never got - and nothing about the output would
    look wrong.
    """
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    run = _finished_run(runs, RUN_NAME)

    metrics = _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.99]), counts=[10, 0, 0, 0])

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **_kwargs):
            return metrics

    train_model.main(
        ["--export-dir", str(root), "--run-project", str(runs), "--val"], yolo=_Model
    )
    capsys.readouterr()
    assert train_model.load_val_metrics(
        run / train_model.VAL_METRICS_NAME, train_model.weights_sha256(run / "weights" / "best.pt")
    )

    # The checkpoint is replaced - same path, different bytes - as a re-train into the same
    # --run-dir does.
    (run / "weights" / "best.pt").write_bytes(b"a different checkpoint")

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models), "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No measurement of *these* weights was found" in out
    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    assert "validation" not in record


def test_cli_val_says_nothing_ran_when_asked_for_neither_action(tmp_path, capsys):
    root = _fake_export(tmp_path / "export-v2")
    code = train_model.main(
        ["--export-dir", str(root), "--run-project", str(tmp_path / "runs")],
        yolo=lambda *_a, **_k: pytest.fail("--val was not asked for; nothing should load"),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--val to validate" in out


def test_cli_val_on_a_split_with_no_run_is_an_error_not_a_silent_pass(tmp_path):
    """No weights means no number. Reporting an empty table as success is the one outcome
    that would make `--val` worse than not having it."""
    root = _fake_export(tmp_path / "export-v2")
    with pytest.raises(SystemExit) as excinfo:
        train_model.main(
            ["--export-dir", str(root), "--run-project", str(tmp_path / "runs"), "--val"],
            yolo=lambda *_a, **_k: pytest.fail("nothing to load"),
        )
    assert "the run did not complete" in str(excinfo.value)


def test_cli_install_without_a_run_dir_installs_the_finished_run(tmp_path, capsys):
    """The docs' own `train_model.py --install` line, which used to need a `--run-dir` the
    examples did not mention."""
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    _finished_run(runs, RUN_NAME)

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "models" / WEIGHT_NAME).read_bytes() == b"weights"
    assert "resize_mode: stretch" in out


def test_no_weight_files_are_tracked():
    """63 MB of COCO weights were committed before the `*.pt` rule existed and have been taken
    out of the index with `git rm --cached` (the files stay on disk - ultralytics needs them
    there). Checked through git rather than by reading `.gitignore`, because that is the only
    thing that knows: an ignore rule does **not** untrack something already in the index, which
    is how three weight files stayed committed while the file that bans them sat right beside
    them. Nothing in the working tree changes if this regresses; that is the whole problem.
    """
    ls = subprocess.run(
        ["git", "ls-files", "sidecar"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    if ls.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = [p for p in ls.stdout.splitlines() if p.endswith((".pt", ".onnx"))]
    assert tracked == [], f"weights are tracked again: {tracked}"


def test_models_readme_is_tracked_and_names_the_weights_the_tool_installs(tmp_path):
    """Two files have to agree about the filename the picker will show: the tool that
    installs the weights and the README in the directory they land in. And the weights
    themselves must stay out of git while the README stays in.
    """
    models = REPO_ROOT / "sidecar" / "models"
    readme = models / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert WEIGHT_NAME in text
    assert "resize_mode: stretch" in text

    ignore = (REPO_ROOT / "sidecar" / ".gitignore").read_text(encoding="utf-8")
    # `models/*`, not `models/`: excluding the directory excludes its contents too, and
    # git does not re-include a path whose parent directory is excluded - so the negation
    # would be dead and this README silently untracked. (`git check-ignore -v` on it used
    # to report the *directory* pattern, i.e. ignored.)
    assert "models/*" in ignore
    assert "!models/README.md" in ignore


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
        [
            "Palmolive Naturals Bar Soap 85g close",
            "Palmolive Naturals Bar Soap 85g mid",
            "Palmolive Naturals Bar Soap 85g far",
            "Milo Chocolate Drink 22g Sachet",
        ]
    )

    tainted = [(level, title, detail) for level, title, detail in rows if "carry a distance" in title]
    assert len(tainted) == 1
    level, title, detail = tainted[0]
    assert level == "fail"
    # Names *which*, because a list of 24 is not readable by eye at this point.
    assert "'Palmolive Naturals Bar Soap 85g mid'" in title
    # And names the two halves of the fix, which are different actions: the class list is not
    # enough on its own (the boxes have to be moved), and the version cannot be edited.
    assert "MOVE" in detail and "regenerate" in detail
    assert "24 instead of 8" in detail


def test_sanity_does_not_flag_a_correct_class_list():
    """The false alarm this must never raise: the real roster plus Palmolive is exactly the
    list a correct project has, and `Palmolive` above all must not read as a distance.
    """
    roster = list(clean_v2.V1_CLASSES) + ["Palmolive Naturals Bar Soap 85g"]
    rows = clean_v2.class_list_rows({name: i for i, name in enumerate(roster)})

    assert [level for level, _title, _detail in rows] == ["ok"]
    assert "8 class(es) defined" in rows[0][2]


def test_sanity_still_fails_a_project_with_no_class_list():
    """Preserved behaviour, and it stays a single row: with no classes there is nothing to
    inspect for distances, so a second row would be noise on an empty project.
    """
    rows = clean_v2.class_list_rows({})

    assert [level for level, _title, _detail in rows] == ["fail"]
    assert rows[0][1] == "no classes defined yet"
    assert "Lock Classes BEFORE labeling" in rows[0][2]


# --- recall by distance: the axis the per-class number averages away ----------------------


def _manifest_with_distances(tmp_path: Path, pairs: list[tuple[str, str]]) -> Path:
    """A manifest in the shape `clean_v2.py` writes, carrying only what a distance needs.

    Written field-for-field as the tool writes it (`new_name` and `distance`), because the
    reader is filename-keyed and a fixture with invented keys would agree with itself and
    nothing else.
    """
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            [{"class": "x", "distance": d, "new_name": n, "old_name": n} for n, d in pairs]
        ),
        encoding="utf-8",
    )
    return path


def _export_with_names(root: Path, per_split: dict[str, list[str]]) -> Path:
    """A minimal export whose images are *named*, so a manifest can be joined to them."""
    import yaml

    for split, names in per_split.items():
        images = root / split / "images"
        images.mkdir(parents=True)
        for name in names:
            (images / name).write_bytes(b"\xff\xd8\xff")
    # The real roster, not a short list: `check_export` refuses an export whose classes are
    # not the v2 eight, and that check is the whole reason a distance pass can trust the yaml
    # `distance_data_yaml` writes.
    names = sorted(label_classes.SLUG_TO_CLASS.values())
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names),
                "names": names,
            }
        ),
        encoding="utf-8",
    )
    return root


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


def test_a_missing_or_corrupt_manifest_is_no_distances_rather_than_a_failure(tmp_path):
    """A run that cannot find distances must lose the breakdown, not the validation: this file
    is written by a separate program, and the acceptance number does not depend on it."""
    assert train_model.distance_map(tmp_path / "nope.json") == {}
    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")
    assert train_model.distance_map(tmp_path / "corrupt.json") == {}
    (tmp_path / "shape.json").write_text('{"a": 1}', encoding="utf-8")
    assert train_model.distance_map(tmp_path / "shape.json") == {}


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


# --------------------------------------------------------------------------
# 10. generations - one trainer, two datasets, and what has to differ
# --------------------------------------------------------------------------


def test_the_generations_share_a_roster_and_differ_by_the_class_v2_adds():
    """v1 is not a different product list: it is the same seven names v1's project declares, and
    v2 adds Palmolive. Both halves matter - the shared part is what keeps a v1 weight's labels
    continuous with v2's, and the added class is why a v1 weight can never predict everything the
    app expects."""
    assert set(generations.V1.classes) < set(generations.V2.classes)
    assert len(generations.V1.classes) == 7 and len(generations.V2.classes) == 8
    assert generations.added_over(generations.V1, generations.V2) == (
        "Palmolive Naturals Bar Soap 85g",
    )
    # Nothing the other way round: v1 declares no class v2 lacks.
    assert generations.added_over(generations.V2, generations.V1) == ()
    # The v2 side is the tools' own copy of the roster, not a retyped list.
    assert generations.V2.classes == tuple(label_classes.SLUG_TO_CLASS.values())


def test_the_generation_names_are_the_ones_the_picker_looks_for():
    """A generation's name is not a label: it is the filename the Admin Panel lists and
    `settings_store.is_custom_model` validates (MODEL_TRAINING.md 8.2)."""
    assert generations.V1.weight_name == "scanncart-grocery-v1.pt"
    assert generations.V2.weight_name == "scanncart-grocery-v2.pt"
    assert generations.V1.run_name == "scanncart-grocery-v1"
    assert train_model.val_name(generations.V2) == "scanncart-grocery-v2-val"
    assert generations.DEFAULT.name == "v2"
    with pytest.raises(SystemExit, match="unknown generation"):
        generations.get("v3")


def test_a_v1_export_is_judged_against_v1s_class_list(tmp_path):
    """Why the check stopped being one shared roster: v1's project has seven classes, so judging
    its export against v2's eight would refuse a set that is correct - and a check that cries wolf
    is how the real mismatch gets waved through."""
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    _splits, problems = train_model.check_export(root, generations.V1)
    assert problems == []

    # The same export read as a v2 one: exactly one missing class, and it is the one v2 adds.
    _splits, v2_problems = train_model.check_export(root, generations.V2)
    assert v2_problems == ["the export has no class for: Palmolive Naturals Bar Soap 85g"]


def test_the_export_check_says_which_classes_this_generation_can_never_predict(tmp_path, capsys):
    """Not a problem - it is a property of the dataset - but it is the sentence the app will show
    about these weights, so it is better known before an hour of GPU time than after it."""
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    train_model.check_export(root, generations.V1)
    out = capsys.readouterr().out
    assert "can never predict them: Palmolive Naturals Bar Soap 85g" in out

    # And silence for the generation that declares everything, so the line means something when
    # it does appear.
    complete = _fake_export(tmp_path / "export-v2", names=list(generations.V2.classes))
    train_model.check_export(complete, generations.V2)
    assert "can never predict" not in capsys.readouterr().out


def test_v1_declares_no_distance_axis_where_v2_names_a_manifest():
    """`None` is the generation saying it has no such axis - v1's export carries no distance tags
    - which is why `--val` gives it its own sentence rather than reporting a manifest that went
    missing, and why an absent breakdown there cannot be read as "every distance passed"."""
    assert generations.V1.manifest is None
    assert generations.V2.manifest is not None
    assert generations.V2.manifest.name == "manifest.json"
    # A generation with no axis has no distances to look up, whatever it is asked for.
    assert train_model.distance_map(generations.V1.manifest) == {}


def test_the_geometry_reading_tells_stretched_frames_from_padded_ones(tmp_path):
    """Why this reading exists: `resize_mode: auto` resolving to letterbox for a stretch-trained
    `.pt` is a failure this project has already paid for, so the frames are measured rather than
    trusted from a constant. A stretched frame fills its border with content; a fitted one pads
    with a constant colour, and that is the whole signature."""
    numpy = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")

    content = numpy.random.default_rng(0).integers(40, 210, (64, 64, 3), dtype="uint8")

    def _images(name: str, pixels) -> Path:
        directory = tmp_path / name
        directory.mkdir(parents=True)
        image_module.fromarray(pixels).save(directory / "frame.jpg")
        return directory

    stretched = _images("stretched", content)
    letterboxed_pixels = numpy.zeros((64, 64, 3), dtype="uint8")
    letterboxed_pixels[16:-16] = content[16:-16]
    letterboxed = _images("letterboxed", letterboxed_pixels)

    (stretched_note,) = train_model.frame_geometry({"train": stretched}, sample=1)
    assert "no constant border, i.e. stretched" in stretched_note

    (padded_note,) = train_model.frame_geometry({"train": letterboxed}, sample=1)
    assert "constant border" in padded_note
    assert "fitting" in padded_note
    # A reading that cannot be taken is a note, never a failure: the frames are a claim about the
    # geometry, and a training run should not die because one file would not open.
    assert train_model.frame_geometry({}) == [
        "frames  no images to sample, so the geometry was not measured"
    ]


def test_the_run_is_bounded_by_the_machine_budget():
    """The trainer shares this box with the app, so its dataloader count comes from the CPU budget
    rather than ultralytics' default of eight processes, and its batch from the VRAM share rather
    than from the constant - which is what stops a run taking the machine over."""
    assert train_model.derive_workers(12) == 2
    assert train_model.derive_workers(1) == 1  # never zero workers
    assert resources.CPU_THREADS < (os.cpu_count() or 4)  # something is always left free
    budget = resources.Budget(vram_cap_gb=3.0)
    assert budget.batch_size(640) == 8
    # The same share at four times the pixels fits far less, which is the whole point of deriving
    # it: a batch that OOMs takes every other application on the card down with the run.
    assert budget.batch_size(1280) < budget.batch_size(640)
    # An explicit --batch is a ceiling, never a way past the card's share.
    assert min(16, budget.batch_size(640)) == 8
    assert min(4, budget.batch_size(640)) == 4


def test_cli_val_for_a_generation_with_no_distance_axis_says_there_is_none(tmp_path, capsys):
    """The per-class table is the whole readout for v1: no grid, and no complaint about a file
    that never existed. The distinction is the reason `manifest` is `None` rather than a path."""
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    runs = tmp_path / "runs"
    _finished_run(runs, generations.V1.run_name)

    metrics = _FakeMetrics(
        _NAMES,
        _FakeBox(index=[0], recall=[0.91], map50=0.9, map_=0.63, mp=0.9, mr=0.88),
        counts=[20, 20, 20, 20],
    )

    class _Model:
        def __init__(self, weights):
            self.weights = weights

        def val(self, **kwargs):
            return metrics

    code = train_model.main(
        [
            "--generation",
            "v1",
            "--dataset-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "has no distance axis" in out
    assert "no distances for the" not in out
    assert "split by distance" not in out
    # The per-class readout itself is the same one v2 gets, floor and instance counts included.
    assert "[.ok.] bear-brand 0.910 >= 0.85 (n=20)" in out


def test_cli_train_creates_the_run_project_and_passes_the_budget_to_the_run(tmp_path, capsys):
    """The `--yes` path, which is the one this project actually runs and the one no other test
    reaches. Two things in it are load-bearing: the run project may not exist yet - the free-space
    check raises FileNotFoundError on a missing directory, which is exactly how the first v1
    training run died a second before it started - and the batch/worker/device that reach
    ultralytics have to be the ones the budget derived, not the constants.
    """
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    runs = tmp_path / "runs"  # deliberately absent
    pulled: list[str] = []
    trained: list[dict] = []

    class _Model:
        def train(self, **kwargs):
            trained.append(kwargs)
            # A checkpoint, because the tool reads one back to report on the run; a fake that
            # left it out would test only the failure path.
            weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "best.pt").write_bytes(b"weights")

    def _yolo(name):
        pulled.append(name)
        return _Model()

    code = train_model.main(
        [
            "--generation",
            "v1",
            "--dataset-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--yes",
        ],
        yolo=_yolo,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert runs.is_dir()
    assert pulled == [train_model.BASE_MODEL]
    (kwargs,) = trained
    assert kwargs["name"] == generations.V1.run_name
    assert kwargs["epochs"] == train_model.EPOCHS
    assert kwargs["imgsz"] == train_model.IMGSZ
    # The machine's share, not the constants: a batch that OOMs on a shared card takes the other
    # applications on it down with the run.
    assert kwargs["batch"] <= train_model.BATCH
    assert kwargs["workers"] == train_model.derive_workers(resources.CPU_THREADS)
    assert kwargs["device"] in ("0", "cpu")
    assert "resource budget" in out and "GB free at" in out
    # --yes alone installs nothing: the drop-in is its own step.
    assert not (tmp_path / "models").exists()


def test_cli_v1_installs_under_v1s_own_name_with_its_own_class_list(tmp_path, capsys):
    """The drop-in is what makes a trained checkpoint selectable, and for v1 it has to be v1's
    name with v1's seven classes recorded: those weights are correct and still unable to predict
    Palmolive, and the record is where the app can see that before running them.
    """
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    models = tmp_path / "models"
    _finished_run(tmp_path / "runs", generations.V1.run_name)

    code = train_model.main(
        [
            "--generation",
            "v1",
            "--dataset-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(models),
            "--version",
            "1",
            "--install",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert (models / "scanncart-grocery-v1.pt").read_bytes() == b"weights"
    # Nothing lands under the other generation's name: the filename is the picker's key, so a v1
    # run installing as v2 would silently replace the model the app is built towards.
    assert not (models / generations.V2.weight_name).exists()

    record = json.loads((models / "scanncart-grocery-v1.json").read_text("utf-8"))
    assert record["generation"] == "v1"
    assert record["resize_mode"] == "stretch"
    assert record["source"] == "scanncart-grocery version 1"
    assert record["class_names"] == list(generations.V1.classes)
    assert "Palmolive Naturals Bar Soap 85g" not in record["class_names"]
    assert "scanncart-grocery-v1" in out


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


def test_parse_labels_reads_a_plain_box_too():
    """Both forms train correctly, so both must measure correctly - a mixed export is normal."""
    (found,) = audit_recall.parse_labels("0 0.30 0.40 0.40 0.40")
    assert found == _instance(0, (0.3, 0.4, 0.4, 0.4))


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


def _record(n_labels: int, n_hits: int, conf: float = 0.9) -> audit_recall.FrameRecord:
    """A frame with `n_labels` boxes in a row, of which the first `n_hits` are predicted."""
    boxes = [(0.1 + 0.2 * i, 0.1, 0.25 + 0.2 * i, 0.25) for i in range(n_labels)]
    truth = tuple(_instance(0, b) for b in boxes)
    preds = tuple(_pred(0, b, conf) for b in boxes[:n_hits])
    return audit_recall.FrameRecord(truth=truth, preds=preds)


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
