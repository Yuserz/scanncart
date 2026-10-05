"""Tests for tools/gate_a_reviewer.py.

The reviewer's own docstring says the suite runs against fakes and no real camera — the
recorder already established that convention. Playback is the OpenCV surface this tool
touches, so the tests here keep the window/keyboard surface behind injectable callables
and grade against dict rows rather than pixels.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path

import pytest

from tools.gate_a_reviewer import (
    GATE_A_THRESHOLDS,
    Grade,
    adjudication_header,
    append_grade,
    format_grade_line,
    grade_trial,
    group_history,
    load_grades,
    load_history,
    load_manifest,
    merge_reviewers,
    parse_args,
    parse_import_csv,
    parse_yes_no,
    print_summary,
    run,
    summarize,
    write_adjudication_worksheet,
)


def _trial(trial_id="DEP-01", kind="deposit", sku="Bear Brand Sachet", **extra):
    row = {
        "trial_id": trial_id,
        "kind": kind,
        "sku": sku,
        "actor": "A",
        "light": "L1",
        "scenario": f"Deposit: {sku}",
        "file_name": f"{trial_id}.mp4",
    }
    row.update(extra)
    return row


def _manifest(tmp_path: Path, trials) -> Path:
    session = tmp_path / "s1"
    session.mkdir(parents=True, exist_ok=True)
    with open(session / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"session": "s1", "trials": trials}, f)
    return session


def _grade(trial_id="DEP-01", kind="deposit", verdict="PASS", **kw):
    fields = dict(
        sku_legible=True, direction_correct=True, endpoint_reached=True,
        hold_valid=True, no_ambiguity=True, notes="",
    )
    fields.update(kw)
    if verdict == "FAIL":
        fields["sku_legible"] = False
    return Grade(
        trial_id=trial_id, kind=kind, sku=None, actor="A", light="L1",
        reviewer="R1", **fields,
    )


# --- Grade verdict and CSV row -------------------------------------------------


def test_grade_verdict_all_or_nothing():
    assert _grade().verdict == "PASS"
    assert _grade(verdict="FAIL").verdict == "FAIL"
    assert _grade(hold_valid=False).verdict == "FAIL"


def test_grade_to_row_shape():
    row = _grade().to_row()
    assert row["verdict"] == "PASS"
    assert row["sku_legible"] == "1"
    assert row["hold_valid"] == "1"
    assert row["notes"] == ""
    assert row["graded_at"] == ""  # test doubles stamp explicitly; see the CLI tests


# --- graded_at stamping ---------------------------------------------------------


def test_grade_trial_stamps_utc_iso_time():
    answers = iter(["1y", "2y", "3y", "4y", "5y", ""])
    g = grade_trial(
        _trial(trial_id="DEP-01", kind="deposit", sku="Bear Brand Sachet"),
        reviewer="R1",
        input_func=lambda _p: next(answers),
        print_func=lambda _s: None,
    )
    # ISO-8601 with an explicit UTC offset, sortable as text.
    assert g.graded_at.endswith("+00:00")
    assert "T" in g.graded_at
    datetime.fromisoformat(g.graded_at)  # parses


def test_regraded_row_carries_its_own_later_timestamp(tmp_path):
    # The scorecard is an append-only log: both grades stay on disk, and "latest"
    # is decided by graded_at, not file position.
    path = tmp_path / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01", verdict="FAIL", graded_at="2026-10-02T10:00:00+00:00"))
    append_grade(path, _grade("DEP-01", graded_at="2026-10-02T11:30:00+00:00"))
    history = load_history(path)
    assert len(history) == 2
    assert history[0]["graded_at"] == "2026-10-02T10:00:00+00:00"
    rows = load_grades(path)
    assert rows["DEP-01"]["graded_at"] == "2026-10-02T11:30:00+00:00"
    assert rows["DEP-01"]["verdict"] == "PASS"


def test_summarize_surfaces_latest_regrade_time():
    grades = {
        "DEP-01": _grade("DEP-01", "deposit", graded_at="2026-10-02T10:00:00+00:00").to_row(),
        "DEP-02": _grade("DEP-02", "deposit", graded_at="2026-10-02T10:05:00+00:00").to_row(),
    }
    s = summarize(grades)
    assert s["latest_regrade"] == "2026-10-02T10:05:00+00:00"


def test_summarize_latest_regrade_none_when_unstamped():
    # Rows written before the field existed carry an empty graded_at.
    grades = {"DEP-01": _grade("DEP-01", "deposit").to_row()}
    s = summarize(grades)
    assert s["latest_regrade"] is None


def test_print_summary_shows_latest_grade_line(capsys):
    grades = {
        "DEP-01": _grade("DEP-01", "deposit", graded_at="2026-10-02T10:00:00+00:00").to_row(),
    }
    print_summary(summarize(grades))
    out = capsys.readouterr().out
    assert "Most recent grade: 2026-10-02T10:00:00+00:00" in out


# --- Yes/no parsing ------------------------------------------------------------


def test_parse_yes_no_accepts_variants_and_reasks():
    answers = iter(["YES", "n", "bogus", "y"])
    assert parse_yes_no("> ", lambda _p: next(answers)) is True
    assert parse_yes_no("> ", lambda _p: next(answers)) is False
    assert parse_yes_no("> ", lambda _p: next(answers)) is True


def test_parse_yes_no_accepts_explicit_prefix_for_current_question():
    # "2n" answered while question 2 is on screen; spacing and long-form both work.
    answers = iter(["2n", "3 y", "4yes"])
    assert parse_yes_no("> ", lambda _p: next(answers), current=1) is False
    assert parse_yes_no("> ", lambda _p: next(answers), current=2) is True
    assert parse_yes_no("> ", lambda _p: next(answers), current=3) is True


def test_parse_yes_no_rejects_prefix_for_wrong_indicator():
    # Question 2 is on screen; "1y" targets indicator 1 -> re-ask, then bare answer.
    answers = iter(["1y", "y"])
    got = parse_yes_no("> ", lambda _p: next(answers), current=1)
    assert got is True


def test_parse_yes_no_rejects_invalid_prefixed_verdict():
    # "3x" carries a prefix but no y/n verdict; then "3n" lands correctly.
    answers = iter(["3x", "3n"])
    assert parse_yes_no("> ", lambda _p: next(answers), current=2) is False


def test_parse_yes_no_reasks_on_empty_answer():
    answers = iter(["", "1y"])
    assert parse_yes_no("> ", lambda _p: next(answers), current=0) is True


# --- Grading flow ---------------------------------------------------------------


def test_grade_trial_no_transfer_skips_sku_question():
    # A no-transfer asks 4 questions (2..5); answers are explicit so the mapping
    # cannot drift: 2n(dir), 3n(endpoint NOT withheld), 4y, 5y, then notes.
    answers = iter(["2n", "3n", "4y", "5y", ""])
    g = grade_trial(
        _trial(trial_id="NOT-01", kind="no_transfer", sku=None),
        reviewer="R1",
        input_func=lambda _p: next(answers),
        print_func=lambda _s: None,
    )
    assert g.sku_legible is True  # not asked; defaulted
    assert g.direction_correct is False
    assert g.endpoint_reached is False  # reviewer says the endpoint was NOT correctly withheld
    assert g.verdict == "FAIL"


def test_grade_trial_transfer_explicit_answers_in_wrong_order_still_lands():
    # The failure mode the explicit format exists for: answers given out of order
    # (3 before 2) are each pinned to their own indicator, never shifted by position.
    # Q2 on screen gets "2y"; Q3 gets "3n"; the rest are explicit too.
    answers = iter(["1y", "2y", "3n", "4y", "5y", ""])
    g = grade_trial(
        _trial(trial_id="DEP-01", kind="deposit", sku="Bear Brand Sachet"),
        reviewer="R1",
        input_func=lambda _p: next(answers),
        print_func=lambda _s: None,
    )
    assert (g.sku_legible, g.direction_correct, g.endpoint_reached,
            g.hold_valid, g.no_ambiguity) == (True, True, False, True, True)
    assert g.verdict == "FAIL"


def test_grade_trial_bare_answers_still_work_positionally():
    # Bare y/n remain valid for a human keeping rhythm; this pins the compatible path.
    answers = iter(["y", "y", "n", "y", "y", "seen it"])
    g = grade_trial(
        _trial(trial_id="DEP-02", kind="deposit", sku="Lucky Me Pancit"),
        reviewer="R1",
        input_func=lambda _p: next(answers),
        print_func=lambda _s: None,
    )
    assert (g.sku_legible, g.direction_correct, g.endpoint_reached,
            g.hold_valid, g.no_ambiguity) == (True, True, False, True, True)
    assert g.notes == "seen it"


def test_grade_trial_prompts_name_the_indicator_key(capsys):
    prompts: list[str] = []
    answers = iter(["1y", "2y", "3y", "4y", "5y", ""])
    grade_trial(
        _trial(trial_id="DEP-03", kind="deposit", sku="Milo Sachet"),
        reviewer="R1",
        input_func=lambda p: (prompts.append(p), next(answers))[1],
    )
    out = capsys.readouterr().out
    # The prompt string itself carries the indicator key (passed to input_func), and
    # the printed question carries its number.
    assert prompts[0].strip() == "[1y / 1n] >"
    assert prompts[-2].strip() == "[5y / 5n] >"
    assert "  3. " in out  # questions carry their indicator number


# --- Scorecard persistence -------------------------------------------------------


def test_append_then_load_roundtrip(tmp_path):
    path = tmp_path / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01"))
    append_grade(path, _grade("DEP-02", sku_legible=False, verdict="FAIL"))
    grades = load_grades(path)
    assert grades["DEP-01"]["verdict"] == "PASS"
    assert grades["DEP-02"]["sku_legible"] == "0"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_load_grades_picks_latest_even_when_file_order_differs(tmp_path):
    # A hand edit could reorder rows; graded_at, not file position, decides "latest".
    path = tmp_path / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01", graded_at="2026-10-02T11:30:00+00:00"))
    append_grade(path, _grade("DEP-01", verdict="FAIL", sku_legible=False, graded_at="2026-10-02T10:00:00+00:00"))
    grades = load_grades(path)
    assert grades["DEP-01"]["verdict"] == "PASS"  # 11:30 wins despite being first in file


def test_group_history_orders_each_timeline_by_graded_at():
    history = [
        {"trial_id": "DEP-01", "graded_at": "2026-10-02T11:30:00+00:00", "verdict": "PASS"},
        {"trial_id": "DEP-01", "graded_at": "2026-10-02T10:00:00+00:00", "verdict": "FAIL"},
        {"trial_id": "DEP-02", "graded_at": "", "verdict": "PASS"},
    ]
    grouped = group_history(history)
    assert [r["graded_at"] for r in grouped["DEP-01"]] == [
        "2026-10-02T10:00:00+00:00", "2026-10-02T11:30:00+00:00"
    ]
    assert grouped["DEP-02"][0]["verdict"] == "PASS"


def test_format_grade_line_renders_marks_notes_and_unstamped():
    line = format_grade_line({
        "graded_at": "2026-10-02T10:00:00+00:00", "verdict": "FAIL",
        "sku_legible": "0", "direction_correct": "1", "endpoint_reached": "1",
        "hold_valid": "1", "no_ambiguity": "1", "notes": "blurry label",
    })
    assert line == "2026-10-02T10:00:00+00:00  FAIL  0/1/1/1/1  blurry label"
    unstamped = format_grade_line({"verdict": "PASS", "sku_legible": "1",
                                   "direction_correct": "1", "endpoint_reached": "1",
                                   "hold_valid": "1", "no_ambiguity": "1", "notes": ""})
    assert unstamped.startswith("(unstamped)")
    assert unstamped.endswith("1/1/1/1/1")  # no trailing separator for empty notes


# --- Summary thresholds -----------------------------------------------------------


def test_summarize_all_pass_is_go():
    grades = {}
    for i in range(1, 31):
        grades[f"DEP-{i:02d}"] = _grade(f"DEP-{i:02d}", "deposit").to_row()
        grades[f"REM-{i:02d}"] = _grade(f"REM-{i:02d}", "removal").to_row()
        grades[f"NOT-{i:02d}"] = _grade(f"NOT-{i:02d}", "no_transfer").to_row()
    s = summarize(grades)
    assert s["overall"] is True
    assert s["complete"] is True
    assert s["counts"]["sku_legible"]["actual"] == 60
    assert s["counts"]["no_transfer_false"]["actual"] == 0


def test_summarize_one_failed_deposit_is_no_go():
    grades = {}
    for i in range(1, 31):
        verdict = "FAIL" if i == 1 else "PASS"
        g = _grade(f"DEP-{i:02d}", "deposit", verdict=verdict,
                   sku_legible=(verdict == "PASS"))
        grades[f"DEP-{i:02d}"] = g.to_row()
        grades[f"REM-{i:02d}"] = _grade(f"REM-{i:02d}", "removal").to_row()
        grades[f"NOT-{i:02d}"] = _grade(f"NOT-{i:02d}", "no_transfer").to_row()
    s = summarize(grades)
    assert s["counts"]["deposit_pass"]["actual"] == 29
    assert s["counts"]["deposit_pass"]["pass"] is True  # exactly at the floor
    assert s["counts"]["sku_legible"]["actual"] == 59
    assert s["counts"]["sku_legible"]["pass"] is True  # 59 >= 57
    assert s["overall"] is True


def test_summarize_no_transfer_false_commit_counts_endpoint():
    grades = {}
    for i in range(1, 31):
        bad = i == 5
        # In the CSV, endpoint_reached for a no-transfer means "endpoint correctly withheld";
        # the bad trial is the one where that confirmation is absent ("0").
        grades[f"NOT-{i:02d}"] = _grade(
            f"NOT-{i:02d}", "no_transfer", endpoint_reached=not bad, verdict="FAIL" if bad else "PASS"
        ).to_row()
        grades[f"DEP-{i:02d}"] = _grade(f"DEP-{i:02d}", "deposit").to_row()
        grades[f"REM-{i:02d}"] = _grade(f"REM-{i:02d}", "removal").to_row()
    s = summarize(grades)
    assert s["counts"]["no_transfer_false"]["actual"] == 1
    assert s["counts"]["no_transfer_false"]["pass"] is False
    assert s["overall"] is False


def test_summarize_incomplete_sample_flags_it():
    grades = {"DEP-01": _grade("DEP-01", "deposit").to_row()}
    s = summarize(grades)
    assert s["complete"] is False
    assert s["graded"] == 1


def test_print_summary_verdict_lines(capsys):
    s = summarize({"DEP-01": _grade("DEP-01", "deposit").to_row()})
    print_summary(s)
    out = capsys.readouterr().out
    assert "NO-GO" in out and "incomplete" in out


# --- CLI -----------------------------------------------------------------------


def test_parse_args_defaults():
    a = parse_args([])
    assert a.out == "data/gate_a_footage"
    assert a.session == "s1"
    assert a.reviewer == "R1"
    assert a.trial is None
    assert a.summary is False
    assert a.no_playback is False


def test_run_missing_manifest_errors(tmp_path, capsys):
    args = parse_args(["--out", str(tmp_path / "nowhere"), "--session", "s1"])
    assert run(args) == 2
    assert "No manifest" in capsys.readouterr().out


def test_run_summary_only_mode(tmp_path, capsys):
    session = _manifest(tmp_path, [_trial()])
    path = session / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01"))
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--summary"])
    assert run(args) == 0
    out = capsys.readouterr().out
    assert "Gate A summary" in out


def test_run_grades_one_trial_with_no_playback(tmp_path, capsys):
    _manifest(tmp_path, [_trial(), _trial("DEP-02")])
    # 5 y answers + notes for each of the two trials
    answers = iter(["y", "y", "y", "y", "y", "", "n", "n", "n", "n", "n", "occluded"])
    args = parse_args(
        ["--out", str(tmp_path), "--session", "s1", "--trial", "DEP-01",
         "--no-playback", "--reviewer", "R1"]
    )
    # --trial limits to DEP-01 only
    consumed = iter(["y", "y", "y", "y", "y", ""])
    assert run(args, input_func=lambda _p: next(consumed)) == 0
    session = tmp_path / "s1"
    rows = load_grades(session / "scorecard_R1.csv")
    assert rows["DEP-01"]["verdict"] == "PASS"
    out = capsys.readouterr().out
    assert "PASS recorded" in out or "-> PASS" in out


def test_run_regrade_appends_and_latest_wins(tmp_path, capsys):
    session = _manifest(tmp_path, [_trial()])
    path = session / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01", verdict="FAIL", sku_legible=False, graded_at="2026-10-02T10:00:00+00:00"))
    answers = iter(["y", "y", "y", "y", "y", "clean rewatch"])
    args = parse_args(
        ["--out", str(tmp_path), "--session", "s1", "--trial", "DEP-01",
         "--no-playback", "--reviewer", "R1"]
    )
    assert run(args, input_func=lambda _p: next(answers)) == 0
    rows = load_grades(path)
    assert rows["DEP-01"]["verdict"] == "PASS"
    assert rows["DEP-01"]["notes"] == "clean rewatch"
    with open(path, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2  # history retained, not replaced


def test_run_unknown_trial_id_errors(tmp_path, capsys):
    _manifest(tmp_path, [_trial()])
    args = parse_args(
        ["--out", str(tmp_path), "--session", "s1", "--trial", "ZZZ-99", "--no-playback"]
    )
    assert run(args) == 2
    assert "not in the manifest" in capsys.readouterr().out


# --- --history: the grade timeline ------------------------------------------------


def test_run_history_prints_timeline_oldest_first(tmp_path, capsys):
    session = _manifest(tmp_path, [_trial()])
    path = session / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01", verdict="FAIL", sku_legible=False, graded_at="2026-10-02T10:00:00+00:00"))
    append_grade(path, _grade("DEP-01", graded_at="2026-10-02T11:30:00+00:00"))
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--history"])
    assert run(args) == 0
    out = capsys.readouterr().out
    assert "Gate A grade history" in out
    assert "DEP-01" in out and "2 grade(s), latest PASS" in out
    # Oldest grade first within the trial's timeline.
    assert out.index("10:00:00") < out.index("11:30:00")
    assert "blurry-label-free" not in out  # sanity: notes render verbatim
    assert "0/1/1/1/1" in out and "1/1/1/1/1" in out


def test_run_history_filters_by_trial(tmp_path, capsys):
    session = _manifest(tmp_path, [_trial(), _trial("DEP-02")])
    path = session / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01", graded_at="2026-10-02T10:00:00+00:00"))
    append_grade(path, _grade("DEP-02", verdict="FAIL", sku_legible=False, graded_at="2026-10-02T10:01:00+00:00"))
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--history", "--trial", "DEP-02",
    ])
    assert run(args) == 0
    out = capsys.readouterr().out
    assert "DEP-02" in out
    assert "DEP-01" not in out


def test_run_history_empty_scorecard_prints_header_and_zero(tmp_path, capsys):
    _manifest(tmp_path, [_trial()])
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--history"])
    assert run(args) == 0
    out = capsys.readouterr().out
    assert "Gate A grade history" in out and "0 grade(s)" in out


def test_run_history_unknown_trial_reports_no_grades(tmp_path, capsys):
    session = _manifest(tmp_path, [_trial()])
    append_grade(session / "scorecard_R1.csv", _grade("DEP-01"))
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--history", "--trial", "REM-01",
    ])
    assert run(args) == 0
    assert "No grades recorded for REM-01" in capsys.readouterr().out


def test_run_history_conflicts_with_summary_and_from_csv(tmp_path, capsys):
    _manifest(tmp_path, [_trial()])
    base = ["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--history"]
    args = parse_args([*base, "--summary"])
    assert run(args) == 2
    assert "cannot be combined" in capsys.readouterr().out
    args = parse_args([*base, "--from-csv", "paper.csv"])
    assert run(args) == 2
    assert "cannot be combined" in capsys.readouterr().out


def test_run_history_unstamped_rows_sort_first_and_say_so(tmp_path, capsys):
    session = _manifest(tmp_path, [_trial()])
    path = session / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01", verdict="FAIL", sku_legible=False))  # legacy: unstamped
    append_grade(path, _grade("DEP-01", graded_at="2026-10-02T11:00:00+00:00"))
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--history"])
    assert run(args) == 0
    out = capsys.readouterr().out
    assert "(unstamped)" in out
    assert out.index("(unstamped)") < out.index("11:00:00")


def test_manifest_loader_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "s1")


# --- paper-scorecard import (--from-csv) ----------------------------------------

IMPORT_HEADER = "trial_id,sku_legible,direction_correct,endpoint_reached,hold_valid,no_ambiguity,notes"


def _import_manifest():
    return {
        "DEP-01": _trial("DEP-01", "deposit"),
        "DEP-02": _trial("DEP-02", "deposit"),
        "NOT-01": _trial("NOT-01", "no_transfer", sku=None),
    }


def test_parse_import_csv_happy_path_with_scrambled_headers():
    # Column order should not matter — the header is what binds a cell to a field.
    text = (
        "notes,no_ambiguity,hold_valid,endpoint_reached,direction_correct,sku_legible,trial_id\n"
        "clear label,1,1,1,1,1,DEP-01\n"
    )
    grades, rejections = parse_import_csv(text, _import_manifest(), reviewer="R2")
    assert rejections == []
    g = grades["DEP-01"]
    assert (g.sku_legible, g.direction_correct, g.endpoint_reached, g.hold_valid, g.no_ambiguity) == (
        True, True, True, True, True
    )
    assert g.verdict == "PASS"
    assert g.notes == "clear label"
    assert g.reviewer == "R2"


def test_parse_import_csv_accepts_paper_marks_and_derives_verdict():
    # x and 1 both mean checked; a single unchecked indicator flips the verdict.
    text = (
        f"{IMPORT_HEADER}\n"
        "DEP-01,x,1,yes,checked,true,\n"
        "DEP-02,1,1,1,1,0,\n"
    )
    grades, rejections = parse_import_csv(text, _import_manifest(), reviewer="R1")
    assert rejections == []
    assert grades["DEP-01"].verdict == "PASS"
    assert grades["DEP-02"].verdict == "FAIL"
    assert grades["DEP-02"].no_ambiguity is False


def test_parse_import_csv_no_transfer_may_leave_sku_cell_empty():
    # The paper sheet has no SKU question for no-transfers; empty defaults to True.
    text = (
        f"{IMPORT_HEADER}\n"
        "NOT-01,,y,y,y,y,no transfer observed\n"
    )
    grades, rejections = parse_import_csv(text, _import_manifest(), reviewer="R1")
    assert rejections == []
    g = grades["NOT-01"]
    assert g.sku_legible is True
    assert g.kind == "no_transfer"
    assert g.notes == "no transfer observed"


def test_parse_import_csv_rejects_empty_indicator_on_a_transfer():
    # On a transfer, an empty indicator cell is missing evidence, not a default.
    text = (
        f"{IMPORT_HEADER}\n"
        "DEP-01,y,y,,y,,\n"
    )
    grades, rejections = parse_import_csv(text, _import_manifest(), reviewer="R1")
    assert grades == {}
    assert len(rejections) == 1
    assert "endpoint_reached" in rejections[0]


def test_parse_import_csv_rejects_unknown_trial_and_duplicates():
    text = (
        f"{IMPORT_HEADER}\n"
        "ZZZ-99,1,1,1,1,1,\n"
        "DEP-01,1,1,1,1,1,\n"
        "dep-01,1,1,1,1,0,dup\n"
    )
    grades, rejections = parse_import_csv(text, _import_manifest(), reviewer="R1")
    # The valid row lands; the unknown trial and the (case-insensitive) duplicate do not.
    assert grades["DEP-01"].verdict == "PASS"
    assert len(rejections) == 2
    assert any("ZZZ-99" in r for r in rejections)
    assert any("duplicate" in r for r in rejections)


def test_parse_import_csv_garbage_mark_is_rejected_not_guessed():
    text = (
        f"{IMPORT_HEADER}\n"
        "DEP-01,maybe,1,1,1,,\n"
    )
    grades, rejections = parse_import_csv(text, _import_manifest(), reviewer="R1")
    assert grades == {}
    assert "maybe" in rejections[0]


def test_parse_import_csv_requires_header_and_indicator_columns():
    # No header: DictReader would treat data as fieldnames -> structural error.
    with pytest.raises(ValueError, match="header"):
        parse_import_csv("DEP-01,1,1,1,1,\n", _import_manifest(), reviewer="R1")
    # Header present but an indicator column missing.
    with pytest.raises(ValueError, match="missing indicator"):
        parse_import_csv("trial_id,sku_legible,notes\nDEP-01,1,\n", _import_manifest(), reviewer="R1")


def test_parse_import_csv_stamps_import_time():
    text = f"{IMPORT_HEADER}\nDEP-01,1,1,1,1,1,\n"
    grades, _ = parse_import_csv(text, _import_manifest(), reviewer="R1")
    datetime.fromisoformat(grades["DEP-01"].graded_at)  # parses
    assert grades["DEP-01"].graded_at.endswith("+00:00")


def test_parse_import_csv_rejects_a_cell_count_mismatch():
    # A short row would slide every cell after the gap onto the wrong column — the
    # shifted-column failure the header format exists to prevent.
    text = (
        f"{IMPORT_HEADER}\n"
        "DEP-01,1,1,1,1,from paper\n"
    )
    grades, rejections = parse_import_csv(text, _import_manifest(), reviewer="R1")
    assert grades == {}
    assert len(rejections) == 1
    assert "shifted columns" in rejections[0]


def test_run_from_csv_imports_into_scorecard(tmp_path, capsys):
    _manifest(tmp_path, [_trial(), _trial("DEP-02")])
    import_file = tmp_path / "paper.csv"
    import_file.write_text(
        f"{IMPORT_HEADER}\nDEP-01,1,1,1,1,1,from paper\n", encoding="utf-8"
    )
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--from-csv", str(import_file),
    ])
    assert run(args) == 0
    out = capsys.readouterr().out
    assert "DEP-01: PASS imported" in out
    assert "1 imported, 0 rejected" in out
    session = tmp_path / "s1"
    rows = load_grades(session / "scorecard_R1.csv")
    assert rows["DEP-01"]["verdict"] == "PASS"
    assert rows["DEP-01"]["notes"] == "from paper"
    assert rows["DEP-01"]["graded_at"]  # stamped at import
    assert "Gate A summary" in out  # verdict block follows the import


def test_run_from_csv_merges_with_existing_and_replaces(tmp_path, capsys):
    session = _manifest(tmp_path, [_trial()])
    path = session / "scorecard_R1.csv"
    append_grade(path, _grade("DEP-01", verdict="FAIL", sku_legible=False))
    import_file = tmp_path / "paper.csv"
    import_file.write_text(f"{IMPORT_HEADER}\nDEP-01,1,1,1,1,1,rewatch\n", encoding="utf-8")
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--from-csv", str(import_file),
    ])
    assert run(args) == 0
    rows = load_grades(path)
    assert rows["DEP-01"]["verdict"] == "PASS"
    assert rows["DEP-01"]["notes"] == "rewatch"
    with open(path, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2  # appended to the history; latest wins


def test_run_from_csv_rejections_exit_nonzero_but_keep_valid_rows(tmp_path, capsys):
    _manifest(tmp_path, [_trial(), _trial("DEP-02")])
    import_file = tmp_path / "paper.csv"
    import_file.write_text(
        f"{IMPORT_HEADER}\nDEP-01,1,1,1,1,1,\nZZZ-99,1,1,1,1,1,\n", encoding="utf-8"
    )
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--from-csv", str(import_file),
    ])
    assert run(args) == 2
    out = capsys.readouterr().out
    assert "REJECTED" in out and "ZZZ-99" in out
    rows = load_grades(tmp_path / "s1" / "scorecard_R1.csv")
    assert rows["DEP-01"]["verdict"] == "PASS"  # the valid row still landed


def test_run_from_csv_stdin_dash(tmp_path, capsys, monkeypatch):
    _manifest(tmp_path, [_trial()])
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(f"{IMPORT_HEADER}\nDEP-01,y,n,y,y,y,\n")
    )
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--from-csv", "-",
    ])
    assert run(args) == 0
    rows = load_grades(tmp_path / "s1" / "scorecard_R1.csv")
    assert rows["DEP-01"]["verdict"] == "FAIL"  # direction marked n


def test_run_from_csv_conflicts_with_trial(tmp_path, capsys):
    _manifest(tmp_path, [_trial()])
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--from-csv", "paper.csv", "--trial", "DEP-01",
    ])
    assert run(args) == 2
    assert "mutually exclusive" in capsys.readouterr().out


def test_run_from_csv_missing_file_errors(tmp_path, capsys):
    _manifest(tmp_path, [_trial()])
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--from-csv", str(tmp_path / "nope.csv"),
    ])
    assert run(args) == 2
    assert "Error" in capsys.readouterr().out


def test_run_from_csv_empty_file_errors(tmp_path, capsys):
    _manifest(tmp_path, [_trial()])
    import_file = tmp_path / "paper.csv"
    import_file.write_text("", encoding="utf-8")
    args = parse_args([
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R1",
        "--from-csv", str(import_file),
    ])
    assert run(args) == 2
    assert "no grade rows" in capsys.readouterr().out


# --- two-reviewer merge (--merge) -------------------------------------------------


def _two_reviewers(tmp_path: Path, r1_rows, r2_rows):
    """Write both scorecards from Grade doubles and return the session dir."""
    session = _manifest(tmp_path, [_trial(), _trial("DEP-02"), _trial("DEP-03")])
    for grade in r1_rows:
        append_grade(session / "scorecard_R1.csv", grade)
    for grade in r2_rows:
        append_grade(session / "scorecard_R2.csv", grade)
    return session


def test_merge_reviewers_classifies_agree_disagree_and_missing():
    manifest = {"DEP-01": _trial(), "DEP-02": _trial("DEP-02"), "DEP-03": _trial("DEP-03")}
    a = {
        "DEP-01": _grade("DEP-01", verdict="PASS").to_row(),
        "DEP-02": _grade("DEP-02", verdict="PASS").to_row(),
    }
    b = {
        "DEP-01": _grade("DEP-01", verdict="PASS").to_row(),
        "DEP-02": _grade("DEP-02", verdict="FAIL").to_row(),
        "DEP-03": _grade("DEP-03", verdict="PASS").to_row(),
    }
    classified = {item["trial_id"]: item for item in merge_reviewers(a, b, manifest)}
    assert classified["DEP-01"]["status"] == "agree"
    assert classified["DEP-02"]["status"] == "disagree"
    assert classified["DEP-03"]["status"] == "missing_a"
    assert classified["DEP-03"]["b"]["verdict"] == "PASS"


def test_merge_reviewers_reports_ungraded_trials():
    manifest = {"DEP-01": _trial()}
    classified = merge_reviewers({}, {}, manifest)
    assert classified[0]["status"] == "missing_both"


def test_merge_ignores_trial_ids_outside_the_manifest():
    # A stale scorecard row for a retired trial is not this session's business.
    manifest = {"DEP-01": _trial()}
    a = {"DEP-01": _grade("DEP-01").to_row(), "DEP-99": _grade("DEP-99").to_row()}
    b = {"DEP-01": _grade("DEP-01", verdict="FAIL").to_row()}
    classified = merge_reviewers(a, b, manifest)
    assert [item["trial_id"] for item in classified] == ["DEP-01"]
    assert classified[0]["status"] == "disagree"


def test_run_merge_writes_worksheet_and_exits_nonzero(tmp_path, capsys):
    session = _two_reviewers(
        tmp_path,
        [_grade("DEP-01", verdict="PASS", graded_at="2026-10-02T10:00:00+00:00"),
         _grade("DEP-02", verdict="FAIL", sku_legible=False)],
        [_grade("DEP-01", verdict="PASS", graded_at="2026-10-02T10:05:00+00:00"),
         _grade("DEP-02", verdict="PASS")],
    )
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"])
    assert run(args) == 1
    out = capsys.readouterr().out
    assert "DEP-02" in out and "DISAGREE" in out
    worksheet = session / "adjudication_R1_R2.csv"
    assert worksheet.exists()
    with open(worksheet, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == adjudication_header()
    assert len(rows) == 2  # header + the one disputed trial
    dispute = dict(zip(rows[0], rows[1]))
    assert dispute["trial_id"] == "DEP-02"
    assert dispute["verdict_r1"] == "FAIL" and dispute["verdict_r2"] == "PASS"
    assert dispute["sku_legible_r1"] == "0" and dispute["sku_legible_r2"] == "1"
    # The adjudication columns are blank: a mark nobody entered is not invented.
    for column in ("sku_legible", "direction_correct", "endpoint_reached", "hold_valid", "no_ambiguity", "notes"):
        assert dispute[column] == ""
    # Neither scorecard was touched — the merge reads them, it never writes them.
    assert load_grades(session / "scorecard_R1.csv")["DEP-02"]["verdict"] == "FAIL"
    assert len(load_history(session / "scorecard_R1.csv")) == 2


def test_run_merge_full_agreement_exits_zero_without_a_worksheet(tmp_path, capsys):
    session = _two_reviewers(
        tmp_path,
        [_grade("DEP-01"), _grade("DEP-02")],
        [_grade("DEP-01"), _grade("DEP-02")],
    )
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"])
    assert run(args) == 0
    out = capsys.readouterr().out
    assert "No disputed trials" in out
    assert not (session / "adjudication_R1_R2.csv").exists()


def test_run_merge_reads_latest_grade_per_trial_on_both_sides(tmp_path, capsys):
    # R1 regraded DEP-01 from FAIL to PASS; the merge must read the regrade, not the
    # superseded row, or a resolved trial would keep coming back as disputed.
    session = _two_reviewers(
        tmp_path,
        [_grade("DEP-01", verdict="FAIL", sku_legible=False, graded_at="2026-10-02T10:00:00+00:00"),
         _grade("DEP-01", graded_at="2026-10-02T12:00:00+00:00")],
        [_grade("DEP-01", graded_at="2026-10-02T10:05:00+00:00")],
    )
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"])
    assert run(args) == 0
    assert "No disputed trials" in capsys.readouterr().out
    assert not (session / "adjudication_R1_R2.csv").exists()


def test_run_merge_refuses_to_overwrite_an_existing_worksheet(tmp_path, capsys):
    session = _two_reviewers(
        tmp_path,
        [_grade("DEP-01"), _grade("DEP-02", verdict="FAIL", sku_legible=False)],
        [_grade("DEP-01"), _grade("DEP-02")],
    )
    worksheet = session / "adjudication_R1_R2.csv"
    worksheet.write_text("adjudicator marks in progress\n", encoding="utf-8")
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"])
    assert run(args) == 2
    assert "already exists" in capsys.readouterr().out
    assert worksheet.read_text(encoding="utf-8") == "adjudicator marks in progress\n"


def test_run_merge_missing_secondary_scorecard_errors(tmp_path, capsys):
    _two_reviewers(tmp_path, [_grade("DEP-01")], [])
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"])
    assert run(args) == 2
    assert "no scorecard for R2" in capsys.readouterr().out


def test_run_merge_missing_primary_scorecard_errors(tmp_path, capsys):
    _two_reviewers(tmp_path, [], [_grade("DEP-01")])
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"])
    assert run(args) == 2
    assert "no scorecard for R1" in capsys.readouterr().out


def test_run_merge_conflicts_with_other_modes(tmp_path, capsys):
    _two_reviewers(tmp_path, [_grade("DEP-01")], [_grade("DEP-01")])
    base = ["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"]
    for extra in (["--summary"], ["--history"], ["--from-csv", "paper.csv"], ["--trial", "DEP-01"]):
        args = parse_args([*base, *extra])
        assert run(args) == 2, extra
        assert "cannot be combined" in capsys.readouterr().out


def test_run_merge_rejects_the_same_reviewer_twice(tmp_path, capsys):
    _manifest(tmp_path, [_trial()])
    args = parse_args(["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R1"])
    assert run(args) == 2
    assert "other than --reviewer" in capsys.readouterr().out


def test_merge_worksheet_feeds_the_from_csv_adjudication_loop(tmp_path, capsys):
    """The §4 loop: merge writes the worksheet, the adjudicator fills and imports it,
    and the adjudicated grade lands on the adjudicator's scorecard as the latest."""
    session = _two_reviewers(
        tmp_path,
        [_grade("DEP-01"), _grade("DEP-02", verdict="FAIL", sku_legible=False)],
        [_grade("DEP-01"), _grade("DEP-02")],
    )
    worksheet = session / "adjudication_R1_R2.csv"
    merge_args = ["--out", str(tmp_path), "--session", "s1", "--reviewer", "R1", "--merge", "R2"]
    assert run(parse_args(merge_args)) == 1

    # The adjudicator (R2 per the run sheet) regrades the disputed clip and fills the
    # blank indicator columns in place — the header, not column position, decides.
    with open(worksheet, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    for name, mark in (
        ("sku_legible", "1"), ("direction_correct", "1"), ("endpoint_reached", "1"),
        ("hold_valid", "1"), ("no_ambiguity", "1"), ("notes", "regraded: label clear"),
    ):
        rows[1][header.index(name)] = mark
    with open(worksheet, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    import_args = [
        "--out", str(tmp_path), "--session", "s1", "--reviewer", "R2",
        "--from-csv", str(worksheet),
    ]
    assert run(parse_args(import_args)) == 0
    adjudicated = load_grades(session / "scorecard_R2.csv")
    assert adjudicated["DEP-02"]["verdict"] == "PASS"
    assert adjudicated["DEP-02"]["notes"] == "regraded: label clear"
    assert adjudicated["DEP-02"]["graded_at"]  # stamped at import; latest wins

    # A re-merge still reports the reviewers' (real) disagreement — the adjudicated
    # verdict lives on R2's scorecard, and the merge reads scorecards, so the disputed
    # set is a worklist, not a verdict — but it refuses to rewrite the now-filled
    # worksheet rather than discarding the adjudication marks.
    assert run(parse_args(merge_args)) == 2
    assert "already exists" in capsys.readouterr().out
    with open(worksheet, newline="", encoding="utf-8") as f:
        assert "regraded: label clear" in f.read()


def test_adjudication_header_carries_both_reviewers_and_the_import_columns():
    header = adjudication_header()
    assert header[:3] == ["trial_id", "kind", "sku"]
    for suffix in ("r1", "r2"):
        assert f"verdict_{suffix}" in header
        for column in ("sku_legible", "direction_correct", "endpoint_reached", "hold_valid", "no_ambiguity"):
            assert f"{column}_{suffix}" in header
    # The tail is the import header exactly: what --from-csv reads back.
    assert header[-6:] == ["sku_legible", "direction_correct", "endpoint_reached", "hold_valid", "no_ambiguity", "notes"]
