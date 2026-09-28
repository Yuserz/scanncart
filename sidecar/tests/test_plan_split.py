"""Tests for `plan_split.py`: proportions, the capture-session term, and the whole-session holdout.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import json
import re
import pytest
import clean_v2
import plan_split

from tests.dataset_tool_helpers import _batch, _entries, _two_session_batches


# --------------------------------------------------------------------------
# 1. stratify
# --------------------------------------------------------------------------


def test_stratify_hits_target_proportions():
    batches = [_batch(f"c{d}", "cls", d, 100) for d in ("close", "mid", "far")]
    plan = plan_split.stratify(batches)
    total = sum(len(b.images) for b in batches)
    counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLIT_NAMES}

    assert sum(counts.values()) == total
    assert counts["train"] / total == pytest.approx(0.70, abs=0.02)
    assert counts["valid"] / total == pytest.approx(0.20, abs=0.02)
    assert counts["test"] / total == pytest.approx(0.10, abs=0.02)


def test_stratify_gives_each_split_a_slice_of_tiny_cells():
    """A 5-image cell that sends nothing to test makes that cell unmeasurable there,
    which is the failure the whole plan exists to avoid."""
    plan = plan_split.stratify([_batch("tiny", "cls", "mid", 5)])
    counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLIT_NAMES}
    assert counts == {"train": 3, "valid": 1, "test": 1}


def test_stratify_handles_a_two_image_cell():
    plan = plan_split.stratify([_batch("pair", "cls", "far", 2)])
    counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLIT_NAMES}
    assert counts["train"] >= 1
    assert counts["valid"] == 1


def test_stratify_never_starves_train():
    """valid and test minimums must not eat the whole cell."""
    for n in range(2, 30):
        plan = plan_split.stratify([_batch("c", "cls", "close", n)])
        counts = {s: sum(1 for v in plan.values() if v == s) for s in plan_split.SPLIT_NAMES}
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
    assert all(v in plan_split.SPLIT_NAMES for v in first.values())


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
    rotate_a = {b.name: plan_split.SPLIT_NAMES[i % 3] for i, b in enumerate(batches)}
    rotate_b = {b.name: plan_split.SPLIT_NAMES[(i + 1) % 3] for i, b in enumerate(batches)}
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


