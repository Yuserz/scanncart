#!/usr/bin/env python
"""Every gate between the staged data and a training run, on one page, with a verdict.

`docs/RUN_SHEET.md` is a chain, and "are we ready to train?" means running five commands and knowing
which number belongs to which question: §0's `label_progress.py` for what is captured and labeled,
§4's `make human-pass --status` for the human pass, §6 for whether the merged set still *is* the set
the labels describe, §7's `make doctor` for the set's internal consistency, and §8's own budget for
whether the machine can take the run. This is the page that answers it in one command.

**It delegates, and re-derives nothing.** Every row calls the owner:
`label_progress.local_progress` for the label state, `train_model.pass_gate` for the human pass (which
*is* `accept_v2.machine_only_verdict`, both halves - the counts and the label state they were read
from), `accept_v2.annotation_state_gate` for the merge's currency, `dataset_doctor.check_dataset` for
consistency, `resources` for the machine. A second implementation of any of those rules is how this
page would come to disagree with the commands it summarises - and a readiness check that says green
where `--yes` exits 2 is worse than no readiness check at all, because it is the one an operator
believes. Each failing row prints the gate's own sentences rather than this file's paraphrase of them.

**Two kinds of "not ready", and the difference decides what you do next.** Some rows *refuse* the
run: `--yes` exits 2 with no GPU time spent, and they are marked `FAIL`. Others only say the dataset
is unfinished, and they are marked `WARN`: a hard negative nobody has marked null is *excluded* from
the set rather than fatal (it is the false-positive half, not the run), a short capture cell is a
weaker `far` number rather than no number, and frames nobody has looked at are the distance between a
model that trains and one worth quoting. The verdict names both counts, and `--strict` is the other
question - *is it finished* - for the run whose numbers are going into the acceptance gate:

    sidecar/.venv/Scripts/python.exe sidecar/tools/readiness.py
    sidecar/.venv/Scripts/python.exe sidecar/tools/readiness.py --strict

Offline and read-only: no API, no key, and no file written (the label snapshot is
`label_progress.py`'s artifact, not this one's). It does read every checkpoint's size and hash for
nothing - it does not touch the run project at all - and the budget row imports torch, which is the
one slow line here; the data rows are asked first so a missing set fails in a second rather than after
it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import accept_v2
import build_dataset
import dataset_doctor
import generations
import label_classes
import label_progress
import resources
import train_model
import workspace

#: The three states a row can be in. `warn` is deliberately not `fail`: an unfinished dataset still
#: trains, and a check that cannot tell the two apart is one an operator learns to ignore.
OK, WARN, FAIL = "ok", "warn", "fail"

#: The `imgsz` the batch line quotes second, because two of this project's weights are trained at 960
#: and `Settings.imgsz` follows the weight - so "batch 8" alone describes one of the two runs.
ALT_IMGSZ = 960


@dataclass(frozen=True)
class Row:
    """One gate: what it is, where it stands, the number behind it, and whether it refuses the run."""

    name: str
    state: str
    detail: str
    #: True when `train_model.py --yes` would exit 2 on this. See the module docstring: the two kinds
    #: of "not ready" are the whole point of the sheet, so the flag is part of the row rather than
    #: something the verdict re-derives from the state.
    blocks: bool = False
    #: The owner's own sentences, printed under the table when the row is not `ok`, so the reason is
    #: the gate's wording and not a summary of it.
    lines: tuple[str, ...] = ()


def joined(lines: list[str] | tuple[str, ...]) -> str:
    """The owner's sentences as one line, for the table cell.

    Collapsed rather than truncated: these gates write one or two sentences, and a detail column that
    cut the second one off would hide the sentence that says what to do about it.
    """
    return " ".join(part.strip() for part in lines if part and part.strip())


def staged_rows(staged: Path) -> list[Row]:
    """§0-§2 from the annotator's store: what is captured, labeled, placed and marked.

    One read of the store, four questions - `label_progress.local_progress` is the reader the panel
    and the run sheet already use, so the numbers here are the numbers those pages show.
    """
    args = label_progress.parse_args(["--out", str(staged)])
    # The staged hard negatives are a set of their own *beside* the staged session
    # (`workspace.DEFAULT_EXTRAS`), and `local_progress` reads that list off the namespace rather than
    # resolving it. Same union, same owner as `label_progress.main` applies, so a page built here and
    # a page built by that command cannot count different frames.
    args.extras = [str(path) for path in workspace.default_extras(staged.parent)]
    progress = label_progress.local_progress(args, staged)

    summary = progress.summary
    total = int(summary.get("total") or 0)
    decided = int(summary.get("decided") or 0)
    left = max(0, total - decided)
    # `|` and not a middle dot: this runs on a Windows console in a cp1252 code page, where the
    # typographic separators the tables in this repo use come out as a replacement character.
    per_split = " | ".join(
        f"{split} {progress.split_stat[split][0]}/{progress.split_stat[split][1]}"
        for split in label_classes.SPLIT_NAMES
        if split in progress.split_stat
    )
    rows = [
        Row(
            "labels",
            OK if left == 0 else WARN,
            f"{decided}/{total} decided" + (f", {left} left" if left else "")
            + (f"   [{per_split}]" if per_split else ""),
        )
    ]

    # The hard negatives are a *pseudo-class* (`label_classes.PSEUDO_CLASS_SLUGS`) because they carry
    # no distance by design, which is also why they are summed here rather than read off a cell.
    negative_done = sum(parts[0] for (slug, _), parts in progress.cells.items() if slug in label_classes.PSEUDO_CLASS_SLUGS)
    negative_total = sum(parts[1] for (slug, _), parts in progress.cells.items() if slug in label_classes.PSEUDO_CLASS_SLUGS)
    if not negative_total:
        rows.append(
            Row(
                "hard negatives",
                WARN,
                "none staged - the frames that teach the model what the products are *not* "
                "(checklist Tier C2b) are absent from this set entirely",
            )
        )
    elif negative_done < negative_total:
        rows.append(
            Row(
                "hard negatives",
                WARN,
                f"{negative_done}/{negative_total} marked null - an unmarked frame is *excluded* from "
                "the set rather than trained as background, and nothing else says so",
            )
        )
    else:
        rows.append(Row("hard negatives", OK, f"{negative_total}/{negative_total} marked null"))

    # A frame's split is a position in the *set's* declaration, and `build_dataset.py` refuses a frame
    # it cannot place - so a decided frame outside the three names is a blocker, not a warning. Taken
    # as "any key that is not one of the three" rather than by matching the bucket the reader writes
    # for them, because the thing that matters is that the build cannot use the label, whatever the
    # bucket is spelled.
    unplaced = {s: parts for s, parts in progress.split_stat.items() if s not in label_classes.SPLIT_NAMES}
    unplaced_decided = sum(parts[0] for parts in unplaced.values())
    unplaced_total = sum(parts[1] for parts in unplaced.values())
    if unplaced_decided:
        rows.append(
            Row(
                "placement",
                FAIL,
                f"{unplaced_decided} decided frame(s) have no split in {workspace.SPLITS_NAME} - "
                "build_dataset.py refuses a decided frame it cannot place",
                blocks=True,
            )
        )
    else:
        rows.append(
            Row(
                "placement",
                OK,
                "every staged frame with a decision is in a split"
                + (
                    f" ({unplaced_total} undecided frame(s) unplaced, which a build skips)"
                    if unplaced_total
                    else ""
                ),
            )
        )

    # §2: the capture gap, from the same block the panel renders - the empty cells are named because
    # a cell at zero is a product the model is *never taught* at that distance, which no later number
    # can recover (the grid prints `-` for it).
    tier_a = summary.get("tier_a") or {}
    remaining = int(tier_a.get("remaining") or 0)
    target = int(tier_a.get("target") or 0)
    cells = tier_a.get("cells") or []
    empty = [c for c in cells if c.get("remaining") and not c.get("have")]
    if not target:
        rows.append(Row("capture gap", OK, f"no Tier A targets for generation {summary.get('generation') or 'this set'}"))
    elif remaining:
        at_zero = "; ".join(
            f"{label_progress.display_name(c['slug'])} @ {c['distance']} (0/{c['target']})" for c in empty[:3]
        )
        rows.append(
            Row(
                "capture gap",
                WARN,
                f"{remaining} of {target} Tier A image(s) still to shoot across "
                f"{tier_a.get('cells_under_target')} cell(s)"
                + (f"; {len(empty)} at zero: {at_zero}" if empty else ""),
            )
        )
    else:
        rows.append(Row("capture gap", OK, f"all {len(cells)} Tier A cell(s) at target"))
    return rows


def generation_of(dataset: Path, name: str) -> tuple[generations.Generation, str]:
    """Which generation's class list and requirement the set is judged against, and how that was decided.

    `auto` is the doctor's own resolution - its `data.yaml`, then its merge report, then the order its
    names are in - rather than a second reading of the same artifacts here. A copy of that order would
    eventually disagree with the doctor's, and this row exists precisely to report what the doctor
    will say.
    """
    if name == "auto":
        return dataset_doctor.resolve_generation(dataset)
    return generations.get(name), "as asked"


def doctor_detail(report: dict) -> str:
    """A passing doctor, in the terms the run sheet's §7 block prints - from the report's own fields."""
    splits = report.get("splits") or {}
    images = sum(int(block.get("images") or 0) for block in splits.values())
    boxes = sum(int(block.get("boxes") or 0) for block in splits.values())
    bad = sum(int(block.get("problem_frames") or 0) for block in splits.values())
    warnings = len(report.get("warnings") or [])
    return (
        f"{len(report.get('names') or [])} class(es) in {report.get('generation')}'s order, "
        f"{images} image(s) / {boxes} box(es), {bad} bad label row(s)"
        + (f", {warnings} warning(s) (see `make doctor`)" if warnings else "")
    )


def dataset_rows(dataset: Path, generation_name: str) -> list[Row]:
    """§4, §6 and §7: the pass gate, whether the build still matches the labels, and the doctor."""
    report = accept_v2.read_report(dataset)
    rows: list[Row] = []

    gated, gate_lines = train_model.pass_gate(dataset)
    rows.append(Row("pass gate", OK if gated else FAIL, joined(gate_lines), blocks=not gated, lines=tuple(gate_lines)))

    if report:
        current, currency_lines = accept_v2.annotation_state_gate(report)
        # The gate answers with a sentence only when it fails (`(True, [])` is its clean answer), so
        # the passing row states the evidence it was checked against - the stamp's own file and
        # digest - rather than leaving the cell empty and asking a reader to trust it.
        stamp = report.get("annotations") if isinstance(report.get("annotations"), dict) else {}
        named = str(stamp.get("provenance") or "")
        digest = str(stamp.get("sha256") or "")
        detail = joined(currency_lines) or (
            f"built from the current label state ({Path(named).name} unchanged"
            + (f", sha {digest[:12]}" if digest else "")
            + ")"
        )
        rows.append(
            Row(
                "merge currency",
                OK if current else FAIL,
                detail,
                blocks=not current,
                lines=tuple(currency_lines),
            )
        )
    else:
        # Not a refusal, and deliberately so: `train_model.pass_gate` calls this same state "not this
        # gate's set" for a Roboflow export, which has no provenance to read. What it *is* is the one
        # thing that cannot be checked at all - a set built before the last labeling session looks
        # exactly like a current one - so it reads as unfinished rather than as clean.
        rows.append(
            Row(
                "merge currency",
                WARN,
                f"no {workspace.MERGE_REPORT_NAME} - nothing records which label state this set was "
                "built from, so a set built before the last labeling session cannot be told apart "
                "from a current one",
            )
        )

    generation, decided_by = generation_of(dataset, generation_name)
    # The check is "the set as data", but it is not a silent one: it prints the frame-geometry line on
    # the way past (that measurement reports as it goes). Captured rather than let through, so this
    # page stays one page - and passed on rather than dropped, so a failing row still shows everything
    # `make doctor` would have printed for the same set.
    chatter = io.StringIO()
    with contextlib.redirect_stdout(chatter):
        checked = dataset_doctor.check_dataset(dataset, generation, duplicates=True, decided_by=decided_by)
    said = [line for line in chatter.getvalue().splitlines() if line.strip()]
    errors = [str(error) for error in checked.get("errors") or []]
    rows.append(
        Row(
            "doctor",
            FAIL if errors else OK,
            doctor_detail(checked)
            if not errors
            else f"{len(errors)} finding(s) - `--yes` exits 2 before a model is loaded; `make doctor` prints the same block",
            blocks=bool(errors),
            # Blanks dropped: `report_lines` separates its blocks with them, and a list of lines this
            # printer indents again would put empty rows between the findings it was collected for.
            lines=tuple(
                line
                for line in (*said, *dataset_doctor.report_lines(checked))
                if line.strip()
            )
            if errors
            else (),
        )
    )
    return rows


def existing(path: Path) -> Path:
    """`path` or the nearest parent that is a directory, for measuring free space.

    The run project may not exist yet - on this machine it is created by the first `--yes` - and
    `shutil.disk_usage` raises on a missing directory. Measuring the nearest real parent answers the
    same question about the same filesystem, and the row says which path it measured.
    """
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    return Path.cwd()


def budget_rows(run_project: Path) -> list[Row]:
    """§8: whether this machine can take the run - the disk reserve, then the card.

    Last, because it is the only row that imports torch. A run whose set is missing should fail in a
    second, not after a ten-second CUDA probe.
    """
    device = resources.resolve_device("auto")
    # `measure` before `apply`, and `apply` unconditionally: the VRAM totals are filled in by the
    # second, so asking for a batch size first would silently take the no-card fallback and report a
    # batch nobody would get.
    budget = resources.apply(
        resources.measure(resources.USE_PERCENT, resources.DISK_RESERVE_GB, resources.VRAM_PERCENT),
        device,
    )

    where = existing(run_project)
    try:
        free = budget.check_disk(str(where))
        disk = Row("disk", OK, f"{free:.1f} GB free at {where} (reserve {budget.disk_reserve_gb:.0f} GB)")
    except SystemExit as refusal:
        # The owner's own refusal sentence, which is the one `--yes` would print a second later.
        message = str(refusal)
        disk = Row("disk", FAIL, message, blocks=True, lines=(message,))

    batch = budget.batch_size(train_model.IMGSZ)
    vram = Row(
        "vram",
        OK if device != "cpu" else WARN,
        (
            f"device {device}, batch {batch} at {train_model.IMGSZ} px "
            f"({budget.batch_size(ALT_IMGSZ)} at {ALT_IMGSZ}), card share {budget.vram_cap_gb:.1f} GB"
            if device != "cpu"
            else f"device {device} - no card is visible, so a run would take CPU time: correct but much slower"
        ),
    )
    return [disk, vram]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out",
        default=str(workspace.DEFAULT_OUT),
        help="the staged session the labels are read from (default %(default)s)",
    )
    ap.add_argument(
        "--dataset-dir",
        default=str(build_dataset.DEFAULT_OUT),
        help="the merged set the run would train on (default %(default)s)",
    )
    ap.add_argument(
        "--generation",
        default="auto",
        choices=[*sorted(generations.GENERATIONS), "auto"],
        help=(
            "which class list the doctor judges the set against; `auto` (the default) reads it off "
            "the set, as `make doctor` does"
        ),
    )
    ap.add_argument("--run-project", default=str(train_model.DEFAULT_RUN_PROJECT))
    ap.add_argument(
        "--strict",
        action="store_true",
        help=(
            "fail on an unfinished dataset too: warnings (unlabeled frames, unmarked hard negatives, "
            "a capture gap) exit 1 rather than a run only the refusing gates can stop"
        ),
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit the verdict as JSON and nothing else, for a script to read; the two input errors below still print prose",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    staged = Path(args.out).expanduser()
    dataset = Path(args.dataset_dir).expanduser()
    run_project = Path(args.run_project).expanduser()

    # The header is for the table a person reads. `--json` is the other reader, and a document with
    # two prose lines in front of it is not JSON - so this is the one branch `--json` skips.
    if not args.json:
        print(f"staged set: {staged}")
        print(f"dataset:    {dataset}")
        print()

    # The two inputs are asked about before anything is measured: a page full of `0/0 decided` over a
    # directory that does not exist reads as a finished dataset, which is the one wrong answer this
    # tool must not give.
    if not (staged / workspace.MANIFEST_NAME).is_file():
        print(f"  there is no {workspace.MANIFEST_NAME} at {staged} - that is not a staged set")
        print("  (`clean_v2.py clean` writes one; pass --out to point at another)")
        return 2
    if not dataset.is_dir():
        print(f"  there is no merged set at {dataset} - nothing to train on yet")
        print("  (`build_dataset.py --force` assembles it; pass --dataset-dir for another)")
        return 2

    rows = staged_rows(staged) + dataset_rows(dataset, args.generation) + budget_rows(run_project)

    blocking = [row for row in rows if row.blocks]
    warnings = [row for row in rows if row.state == WARN and not row.blocks]

    if args.json:
        print(
            json.dumps(
                {
                    "staged": str(staged),
                    "dataset": str(dataset),
                    "rows": [
                        {"name": r.name, "state": r.state, "detail": r.detail, "blocks": r.blocks}
                        for r in rows
                    ],
                    "blocking": [r.name for r in blocking],
                    "warnings": [r.name for r in warnings],
                    "ready": not blocking,
                    "strict": bool(args.strict),
                },
                indent=1,
            )
        )
        return 1 if blocking or (args.strict and warnings) else 0

    width = max(len(row.name) for row in rows)
    print("gate")
    for row in rows:
        print(f"  {row.name:<{width}}  {row.state:<5} {row.detail}")

    # The owners' sentences, only where they fired: the table is the index, and this is what a reader
    # needs in hand to act on a red row without running the gate again.
    for row in [*blocking, *warnings]:
        if row.lines:
            print()
            print(f"{row.name}:")
            for line in row.lines:
                print(f"  {line}")

    print()
    if blocking:
        print(f"NOT READY TO TRAIN: {len(blocking)} gate(s) refuse this run - {', '.join(r.name for r in blocking)}")
        print("  `train_model.py --yes` exits 2 on these without spending GPU time.")
    else:
        print("no gate refuses this run: `train_model.py --yes` would start.")
    if warnings:
        print(f"{len(warnings)} warning(s) - the run would proceed and the dataset is unfinished: {', '.join(r.name for r in warnings)}")
        if args.strict:
            print("  --strict: an unfinished dataset is a failure here, so this exits 1.")
    else:
        print("nothing unfinished: every row is at its target.")
    return 1 if blocking or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
