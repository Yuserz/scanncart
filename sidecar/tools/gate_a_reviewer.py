"""Interactive human reviewer for the recorded Gate A trial clips.

The recorder (`tools/gate_a_recorder.py`) produces one MP4 per trial plus a
`manifest.json` describing them. This tool walks a reviewer through those clips in
run-sheet order, plays each one back with simple controls, and grades it against the
five binary indicators `docs/GATE_A_RUN_SHEET.md` §4.1 defines:

    1. SKU legible
    2. Direction correct
    3. Endpoint reached (or correctly withheld for a no-transfer)
    4. Hold duration valid (>= 1.0 s)
    5. No visual ambiguity

Verdict is PASS only when all five indicators are 1. Grades are written incrementally
to a CSV scorecard, so an interrupted review session resumes where it left off, and a
`--summary` pass re-computes the Gate A thresholds (>= 57/60 readable SKUs, >= 29/30
deposits, >= 29/30 removals, 0/30 no-transfer false commits) from the CSV alone.

Usage (run from sidecar/):
    .venv/Scripts/python.exe tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R1
    .venv/Scripts/python.exe tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R1 --trial DEP-01
    .venv/Scripts/python.exe tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --summary
    .venv/Scripts/python.exe tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --from-csv paper.csv

Playback controls (in the OpenCV window): SPACE pause/resume, LEFT/RIGHT arrow seek
1 s, S restart clip, Q or ESC advance to grading. When no window can be opened
(headless/CI), the reviewer is still shown the clip's metadata and can grade from the
paper scorecard — grading is never gated on a display.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCORECARD_FIELDS = [
    "trial_id",
    "kind",
    "sku",
    "actor",
    "light",
    "reviewer",
    "sku_legible",
    "direction_correct",
    "endpoint_reached",
    "hold_valid",
    "no_ambiguity",
    "verdict",
    "graded_at",
    "notes",
]

#: The Gate A floors from docs/GATE_A_RUN_SHEET.md §4.2 / §6, named once so the
#: summary printer and any future tooling read the same numbers the spec states.
GATE_A_THRESHOLDS = {
    "sku_legible_min": 57,  # of 60 transfers
    "deposit_pass_min": 29,  # of 30
    "removal_pass_min": 29,  # of 30
    "no_transfer_false_max": 0,  # of 30
}

VALID_KINDS = ("deposit", "removal", "no_transfer")


@dataclass(frozen=True)
class Grade:
    """One reviewer's verdict for one trial — the row a scorecard carries.

    `graded_at` is the UTC moment the verdict was recorded, written on every save so
    a regrade's row carries its own time and the scorecard reads as an ordered history
    (latest grade per trial wins) rather than a set of indistinguishable overwrites.
    """

    trial_id: str
    kind: str
    sku: str | None
    actor: str
    light: str
    reviewer: str
    sku_legible: bool
    direction_correct: bool
    endpoint_reached: bool
    hold_valid: bool
    no_ambiguity: bool
    notes: str = ""
    graded_at: str = ""

    @property
    def verdict(self) -> str:
        return "PASS" if all(
            (self.sku_legible, self.direction_correct, self.endpoint_reached, self.hold_valid, self.no_ambiguity)
        ) else "FAIL"

    def to_row(self) -> dict[str, str]:
        return {
            "trial_id": self.trial_id,
            "kind": self.kind,
            "sku": self.sku or "",
            "actor": self.actor,
            "light": self.light,
            "reviewer": self.reviewer,
            "sku_legible": "1" if self.sku_legible else "0",
            "direction_correct": "1" if self.direction_correct else "0",
            "endpoint_reached": "1" if self.endpoint_reached else "0",
            "hold_valid": "1" if self.hold_valid else "0",
            "no_ambiguity": "1" if self.no_ambiguity else "0",
            "verdict": self.verdict,
            "graded_at": self.graded_at,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Manifest / scorecard I/O
# ---------------------------------------------------------------------------


def load_manifest(session_dir: Path) -> dict[str, Any]:
    """Read `manifest.json` from a recording session directory."""
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Record a session first with tools/gate_a_recorder.py."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_grades(scorecard_path: Path) -> dict[str, dict[str, str]]:
    """Existing grades keyed by trial_id (an empty file or absent file means none)."""
    if not scorecard_path.exists():
        return {}
    grades: dict[str, dict[str, str]] = {}
    with open(scorecard_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            trial_id = (row.get("trial_id") or "").strip()
            if trial_id:
                grades[trial_id] = row
    return grades


def append_grade(scorecard_path: Path, grade: Grade) -> None:
    """Append one grade, writing the header first when the file is new."""
    new_file = not scorecard_path.exists()
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scorecard_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(grade.to_row())


def overwrite_grades(scorecard_path: Path, grades: dict[str, dict[str, str]]) -> None:
    """Rewrite the whole scorecard in run-sheet order (used when regrading).

    Ties — two rows sharing a trial_id can no longer happen (the dict is keyed by
    trial_id), but a *replaced* row's `graded_at` is the regrade's own time, which is
    what makes the file an ordered history.
    """
    ordered = sorted(grades.values(), key=lambda row: row.get("trial_id", ""))
    with open(scorecard_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def play_clip(
    video_path: Path,
    window_name: str,
    input_func: Callable[[], int] = lambda: cv2.waitKey(0) & 0xFF,
    destroy_func: Callable[[], None] = cv2.destroyAllWindows,
    imshow_func: Callable[[str, Any], None] = cv2.imshow,
) -> None:
    """Play one clip in a loop until the reviewer advances (Q, ESC, or window closed).

    SPACE pauses/resumes; LEFT/RIGHT seek a second; S restarts. The window is checked
    for closure (`getWindowProperty`) so a reviewer closing it does not spin forever.
    The headless path is simply never entered: callers without a display do not call
    this — grading continues from the printed metadata instead.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: could not open {video_path.name} for playback; grading from metadata only.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay_ms = max(1, int(1000.0 / fps))
    playing = True

    try:
        while True:
            if playing:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Clip ended: loop it so the reviewer can rewatch until satisfied.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                imshow_func(window_name, frame)

            key = input_func() if playing else input_func()
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                return
            if key == ord(" "):  # SPACE toggles pause
                playing = not playing
            elif key == ord("s") or key == ord("S"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                playing = True
            elif key == 81 or key == 2:  # LEFT arrow
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - fps))
            elif key == 83 or key == 3:  # RIGHT arrow
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos + fps)
            elif key != -1 and key != 255:
                playing = True  # any other key resumes
    finally:
        cap.release()
        destroy_func()


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

#: The five indicators, in the order they are asked. Each question is answered by a
#: key (`1`..`5`) plus `y`/`n` — `"3n"` — so an answer is always attached to the
#: indicator it judges. A bare `y`/`n` answers the *current* question, which keeps a
#: reviewer's rhythm working and the run sheet's per-kind question counts out of the
#: answer's hands (a no-transfer asks four questions, a transfer five; nobody should
#: have to count either to answer correctly).
QUESTIONS = [
    "SKU legible from the label? (transfers only)",
    "Direction of motion clearly observed as expected?",
    "Endpoint behavior correct (deposit released / removal cleared / no-transfer withheld)?",
    "Hold >= 1.0 s in the completion zone?",
    "No blocking occlusion during the critical transition?",
]

class _Skip(Exception):
    """Raised by the explicit-prefix parse when the answer targets another indicator."""


def _parse_explicit(raw: str) -> tuple[int, str] | None:
    """`3n` / `2 y` / `5yes` -> (index, y-or-n); None when the answer carries no prefix."""
    token = raw.strip().lower()
    if len(token) >= 2 and token[0] in "12345" and token[1] in " ":
        stripped = token[2:].strip()
    elif len(token) >= 2 and token[0] in "12345" and token[1] in "yn":
        stripped = token[1:]
    else:
        return None
    if stripped in ("y", "yes"):
        return int(token[0]) - 1, "y"
    if stripped in ("n", "no"):
        return int(token[0]) - 1, "n"
    raise _Skip


def parse_yes_no(prompt: str, input_func: Callable[[str], str], current: int = 0) -> bool:
    """Ask one indicator and return its y/n verdict.

    Accepts `3n` (explicit: indicator 3 is "no"), `3 y`, or a bare `y`/`n` for the
    question just printed. Anything else — including a prefix naming a *different*
    indicator than the one on screen — re-asks with a hint naming the expected key.
    """
    while True:
        raw = input_func(prompt).strip().lower()
        if not raw:
            print(f"  Answer {current + 1}y or {current + 1}n (or bare y/n) — nothing entered.")
            continue
        try:
            parsed = _parse_explicit(raw)
        except _Skip:
            print(f"  '{raw}' is not a valid y/n answer. Use {current + 1}y / {current + 1}n or bare y/n.")
            continue
        if parsed is not None:
            index, verdict = parsed
            if index != current:
                print(f"  That answers indicator {index + 1}, but the question on screen is {current + 1}. "
                      f"Answer {current + 1}y / {current + 1}n.")
                continue
            return verdict == "y"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(f"  '{raw}' is not a valid y/n answer. Use {current + 1}y / {current + 1}n or bare y/n.")


def utc_now_iso() -> str:
    """UTC moment, second resolution, ISO-8601 with an explicit offset — sortable as text."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Paper-scorecard import (--from-csv)
# ---------------------------------------------------------------------------

#: The five indicator columns an import file must carry. `trial_id` selects the trial;
#: everything else about it (kind, sku, actor, light) comes from the manifest, which is
#: the authority — a typo'd kind in a hand-typed row must not rewrite the scorecard.
IMPORT_INDICATOR_COLUMNS = (
    "sku_legible",
    "direction_correct",
    "endpoint_reached",
    "hold_valid",
    "no_ambiguity",
)


def _parse_mark(cell: str, column: str, kind: str) -> bool | None:
    """A paper mark transcribed into text: y/n/1/0/x/checked... None means unparseable.

    `sku_legible` is the one column a no-transfer row may leave empty — the paper sheet
    has no SKU question for those trials — and defaults to the same True the interactive
    grader uses. Every other column must carry a mark.
    """
    raw = (cell or "").strip().lower()
    if raw in ("", "-", "n/a"):
        if column == "sku_legible" and kind == "no_transfer":
            return True
        return None
    if raw in ("y", "yes", "1", "x", "true", "checked"):
        return True
    if raw in ("n", "no", "0", "false", "unchecked"):
        return False
    return None


def parse_import_csv(
    text: str,
    manifest_trials: dict[str, dict[str, Any]],
    reviewer: str,
) -> tuple[dict[str, Grade], list[str]]:
    """Parse a transcribed paper scorecard into grades keyed by trial_id.

    The file must have a header row naming `trial_id`, the five indicator columns and —
    optionally — `notes`, in any order: a self-describing header is what keeps a row
    from sliding between columns, the same positional-ambiguity failure the interactive
    prompts were reworked to kill. Verdicts are derived from the indicators, never read
    from the file (a hand-typed verdict could disagree with the marks beside it).

    Row-level problems (unknown trial, duplicate trial, unparseable mark, a cell count
    that does not match the header) are returned as rejection messages rather than
    raising, so one bad line does not lose the other transcriptions. Structural
    problems (no header, missing columns) raise ValueError.
    """
    text = text.lstrip("\ufeff")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return {}, []

    header = [h.strip().lower() for h in rows[0]]
    if "trial_id" not in header:
        raise ValueError(
            "import CSV needs a header row: trial_id, sku_legible, direction_correct, "
            "endpoint_reached, hold_valid, no_ambiguity[, notes]"
        )
    missing = [c for c in IMPORT_INDICATOR_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"import CSV is missing indicator column(s): {', '.join(missing)}")
    column_index = {name: header.index(name) for name in ["trial_id", *IMPORT_INDICATOR_COLUMNS, "notes"] if name in header}

    grades: dict[str, Grade] = {}
    rejections: list[str] = []
    for lineno, cells in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in cells):
            continue  # blank line between blocks
        if len(cells) != len(header):
            # A short or long row would shift every cell after the gap onto the wrong
            # column — rejected rather than guessed at.
            rejections.append(
                f"line {lineno}: expected {len(header)} cells, got {len(cells)} — row may have shifted columns"
            )
            continue

        def cell(name: str, _cells=cells) -> str:
            return _cells[column_index[name]] if name in column_index else ""

        trial_id = cell("trial_id").strip().upper()
        if not trial_id:
            rejections.append(f"line {lineno}: no trial_id")
            continue
        trial = manifest_trials.get(trial_id)
        if trial is None:
            rejections.append(f"line {lineno}: trial '{trial_id}' is not in the manifest")
            continue
        if trial_id in grades:
            rejections.append(f"line {lineno}: duplicate trial '{trial_id}' in the import file")
            continue
        kind = trial.get("kind", "")

        values: dict[str, bool] = {}
        bad: str | None = None
        for column in IMPORT_INDICATOR_COLUMNS:
            mark = _parse_mark(cell(column), column, kind)
            if mark is None:
                raw = cell(column).strip()
                shown = raw if raw else "(empty)"
                bad = (
                    f"line {lineno}: {trial_id} column {column}: {shown!r} is not a recognizable "
                    "mark (y/n/1/0/x — sku_legible may be empty on a no-transfer)"
                )
                break
            values[column] = mark
        if bad:
            rejections.append(bad)
            continue

        grades[trial_id] = Grade(
            trial_id=trial_id,
            kind=kind,
            sku=trial.get("sku"),
            actor=trial.get("actor", ""),
            light=trial.get("light", ""),
            reviewer=reviewer,
            sku_legible=values["sku_legible"],
            direction_correct=values["direction_correct"],
            endpoint_reached=values["endpoint_reached"],
            hold_valid=values["hold_valid"],
            no_ambiguity=values["no_ambiguity"],
            notes=cell("notes").strip(),
            graded_at=utc_now_iso(),
        )
    return grades, rejections


def grade_trial(
    trial: dict[str, Any],
    reviewer: str,
    input_func: Callable[[str], str],
    print_func: Callable[[str], None] = print,
) -> Grade:
    """Collect the five binary indicators and optional notes for one trial.

    Every prompt names its indicator key and the expected answer shape (`2y / 2n`), so
    an answer is never positionally ambiguous — the playtest's failure mode, where a
    fixed rhythm of keystrokes silently graded the wrong indicator when the SKU
    question was skipped for a no-transfer.
    """
    kind = trial.get("kind", "")
    sku = trial.get("sku")
    print_func(f"  Trial: [{trial['trial_id']}] {trial.get('scenario', '')}")
    if sku:
        print_func(f"  Expected SKU: {sku}")

    # Indicator 1 is only asked for transfers; a no-transfer defaults it and the
    # summary treats a PASS row as the only row whose SKU counts as readable.
    asked = QUESTIONS if sku else QUESTIONS[1:]
    answers: dict[int, bool] = {}
    for offset, question in enumerate(asked):
        index = offset if sku else offset + 1
        print_func(f"  {index + 1}. {question}")
        answers[index] = parse_yes_no(
            f"     [{index + 1}y / {index + 1}n] > ", input_func, current=index,
        )

    notes = input_func("  Notes (blank for none) > ").strip()

    return Grade(
        trial_id=trial["trial_id"],
        kind=kind,
        sku=sku,
        actor=trial.get("actor", ""),
        light=trial.get("light", ""),
        reviewer=reviewer,
        sku_legible=answers.get(0, True),
        direction_correct=answers[1],
        endpoint_reached=answers[2],
        hold_valid=answers[3],
        no_ambiguity=answers[4],
        notes=notes,
        graded_at=utc_now_iso(),
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(grades: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Compute the four Gate A metrics from grade rows.

    Only PASS rows carry a readable SKU in the readability numerator (a FAIL on any
    other indicator can coexist with a legible SKU, but the spec's floor is stated over
    transfers whose review confirms the whole interaction, so PASS is the conservative
    reading). For a no-transfer, the CSV's `endpoint_reached` column means "the reviewer
    confirms no endpoint was reached" — a false commit is a row where that confirmation
    is *absent* (`0`), i.e. the item crossed into a terminal state a ledger would have
    accepted. The question asked during grading is phrased positively for the same
    reason: the reviewer answers what they observed, and the summary reads the verdict
    the same way for every kind.
    """
    deposits = [g for g in grades.values() if g.get("kind") == "deposit"]
    removals = [g for g in grades.values() if g.get("kind") == "removal"]
    no_transfers = [g for g in grades.values() if g.get("kind") == "no_transfer"]

    deposit_pass = sum(1 for g in deposits if g.get("verdict") == "PASS")
    removal_pass = sum(1 for g in removals if g.get("verdict") == "PASS")
    no_transfer_false = sum(1 for g in no_transfers if g.get("endpoint_reached") != "1")
    transfers = deposits + removals
    sku_legible = sum(
        1 for g in transfers if g.get("verdict") == "PASS" and g.get("sku_legible") == "1"
    )

    t = GATE_A_THRESHOLDS
    checks = {
        "sku_legible": (sku_legible, len(transfers), t["sku_legible_min"], sku_legible >= t["sku_legible_min"]),
        "deposit_pass": (deposit_pass, len(deposits), t["deposit_pass_min"], deposit_pass >= t["deposit_pass_min"]),
        "removal_pass": (removal_pass, len(removals), t["removal_pass_min"], removal_pass >= t["removal_pass_min"]),
        "no_transfer_false": (
            no_transfer_false,
            len(no_transfers),
            t["no_transfer_false_max"],
            no_transfer_false <= t["no_transfer_false_max"],
        ),
    }
    overall = all(c[3] for c in checks.values())
    # The most recent grade in the file — the regrade history's latest entry. Rows
    # written before this field existed carry an empty `graded_at` and sort first,
    # which is the honest reading: their moment is simply not recorded.
    stamped = [row.get("graded_at", "") for row in grades.values() if row.get("graded_at")]
    return {
        "counts": {name: {"actual": actual, "total": total, "threshold": threshold, "pass": ok} for name, (actual, total, threshold, ok) in checks.items()},
        "overall": overall,
        "graded": len(grades),
        "complete": len(grades) == 90,
        "latest_regrade": max(stamped) if stamped else None,
    }


def print_summary(summary: dict[str, Any], print_func: Callable[[str], None] = print) -> None:
    """Render the summary block with the pass/fail column the paper scorecard ends with."""
    label = {
        "sku_legible": "Transfers with legible SKU",
        "deposit_pass": "Deposit trials validated",
        "removal_pass": "Removal trials validated",
        "no_transfer_false": "No-transfer false commits",
    }
    print_func("")
    print_func("Gate A summary")
    print_func("-" * 60)
    for name, c in summary["counts"].items():
        verdict = "PASS" if c["pass"] else "FAIL"
        print_func(f"  {label[name]:<30} {c['actual']:>3} / {c['total']:<3} (min/max {c['threshold']:>3})  {verdict}")
    print_func("-" * 60)
    graded = summary["graded"]
    if not summary["complete"]:
        print_func(f"  {graded} of 90 trials graded — incomplete sample; thresholds need the full set.")
    if summary.get("latest_regrade"):
        print_func(f"  Most recent grade: {summary['latest_regrade']}")
    print_func(f"  Overall Gate A verdict: {'GO' if summary['overall'] else 'NO-GO'}"
               + ("" if summary["complete"] else " (incomplete)"))
    print_func("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Gate A clip reviewer: playback, grading, CSV scorecard, and summary.",
    )
    parser.add_argument(
        "--out",
        default="data/gate_a_footage",
        help="Recording output directory containing <session>/manifest.json (default: data/gate_a_footage).",
    )
    parser.add_argument(
        "--session",
        default="s1",
        help="Recording session ID to review (default: s1).",
    )
    parser.add_argument(
        "--reviewer",
        default="R1",
        help="Reviewer identifier written to the scorecard (default: R1).",
    )
    parser.add_argument(
        "--trial",
        help="Grade only this trial ID (e.g. DEP-01); a previous grade is replaced.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the Gate A summary from the existing scorecard and exit (no playback).",
    )
    parser.add_argument(
        "--from-csv",
        help=(
            "Import grades from a transcribed CSV instead of interactive grading "
            "(header: trial_id + the five indicator columns, notes optional; '-' reads stdin)."
        ),
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Skip video playback (grade from metadata / paper scorecard only).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, input_func: Callable[[str], str] = input) -> int:
    session_dir = Path(args.out) / args.session
    scorecard_path = session_dir / f"scorecard_{args.reviewer}.csv"

    if args.from_csv and args.trial:
        print("Error: --from-csv and --trial are mutually exclusive.")
        return 2

    try:
        manifest = load_manifest(session_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 2

    manifest_trials = {t.get("trial_id"): t for t in manifest.get("trials", [])}

    # --- paper-scorecard import mode ---
    if args.from_csv:
        try:
            if args.from_csv == "-":
                text = sys.stdin.read()
            else:
                text = Path(args.from_csv).read_text(encoding="utf-8-sig")
            imported, rejections = parse_import_csv(text, manifest_trials, reviewer=args.reviewer)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}")
            return 2
        if not imported and not rejections:
            print(f"Error: no grade rows found in {args.from_csv}.")
            return 2
        grades = load_grades(scorecard_path)
        for trial_id, grade in imported.items():
            grades[trial_id] = grade.to_row()
            print(f"  {trial_id}: {grade.verdict} imported")
        for message in rejections:
            print(f"  REJECTED {message}")
        overwrite_grades(scorecard_path, grades)
        print(f"  {len(imported)} imported, {len(rejections)} rejected -> {scorecard_path.name}")
        print_summary(summarize(grades))
        # A rejected row means the transcription was not fully taken: exit nonzero so a
        # scripted import notices, while the valid rows still land on the scorecard.
        return 0 if not rejections else 2

    # --- summary-only mode ---
    if args.summary:
        grades = load_grades(scorecard_path)
        print_summary(summarize(grades))
        return 0

    trials = list(manifest_trials.values())
    if not trials:
        print(f"Manifest at {session_dir / 'manifest.json'} carries no trials. Record clips first.")
        return 2

    if args.trial:
        trial = manifest_trials.get(args.trial.upper())
        if trial is None:
            print(f"Error: trial '{args.trial}' is not in the manifest.")
            return 2
        trials = [trial]

    grades = load_grades(scorecard_path)
    pending = [t for t in trials if t["trial_id"] not in grades] if not args.trial else trials

    mode = "regrading" if args.trial else "review"
    print(f"Gate A Reviewer: session '{args.session}', reviewer {args.reviewer}")
    print(f"  {len(pending)} trial(s) to grade ({len(grades)} already in {scorecard_path.name}) — {mode}")
    if not pending:
        print("Nothing to grade. Re-run with --summary for the verdict, or --trial <ID> to regrade.")
        return 0

    for idx, trial in enumerate(pending, start=1):
        print()
        print("=" * 60)
        print(f"Clip {idx}/{len(pending)}: [{trial['trial_id']}] {trial.get('scenario', '')}")
        print(f"  Actor {trial.get('actor', '?')} | Light {trial.get('light', '?')}"
              + (f" | SKU {trial['sku']}" if trial.get("sku") else ""))
        print(f"  File: {trial.get('file_name', trial['trial_id'] + '.mp4')}")
        print("=" * 60)

        clip_path = session_dir / trial.get("file_name", f"{trial['trial_id']}.mp4")
        if not args.no_playback and clip_path.exists():
            print("  Playback: SPACE pause | LEFT/RIGHT seek 1s | S restart | Q grade")
            play_clip(clip_path, window_name=f"Gate A {trial['trial_id']}")
        elif not clip_path.exists():
            print("  (clip file missing — grading from metadata only)")

        try:
            grade = grade_trial(trial, args.reviewer, input_func=input_func)
        except KeyboardInterrupt:
            # Ctrl-C is a normal stopping point mid-session: grades written so far are
            # already on disk, and the next run resumes. Re-raised after the note so the
            # shell still sees the interrupt.
            print("\n  Review interrupted — grades so far are saved to", scorecard_path.name)
            raise
        except EOFError:
            # Piped stdin running dry (headless/scripted runs) lands here; the session
            # pauses cleanly instead of crashing with a traceback past the partial run.
            print("\n  Input ended — grades so far are saved to", scorecard_path.name)
            return 0

        # A regrade (or a --trial run over an already-graded trial) must replace the
        # previous row, not append a second one — the scorecard is keyed by trial_id.
        grades[grade.trial_id] = grade.to_row()
        overwrite_grades(scorecard_path, grades)
        print(f"  -> {grade.verdict} recorded to {scorecard_path.name}")

    print()
    print_summary(summarize(grades))
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
