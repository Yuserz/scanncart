"""Render the human-pass checklist from the store, so its numbers cannot drift from the data.

`_human_pass.md` is the one document in this workflow a person works *from*: sit down, clear the
machine-only decisions the acceptance gate refuses to measure, draw what the local weight could not
see, mark the hard negatives null - and stop when the header says the two gate splits are clean.

It used to be kept by hand, which is the whole problem. Every number in it and every name in its
three lists is derived from the store - which frames carry `machine_only`, which `far` frames in
`valid`/`test` a weight was asked about and proposed nothing for, which staged negatives still have
no null - and all of it moves the moment a frame is confirmed, drawn or marked. A hand-kept copy is
wrong from the first keypress, and it fails *silently*: the operator works a list that no longer
describes the set, and nothing about it distinguishes a frame that is still outstanding from one
the document simply forgot. So the document is a **render** of the store, and this is the renderer.

Three things make the drift impossible rather than merely unlikely.

**The review list is the app's own.** It is `review_worklist()` - the same call
`GET /api/frames?review=1` serves - so "test first, then the crowded far/mid cells" is derived from
the ordering rule rather than retyped as prose about it, and this document's list cannot come out in
a different order from the page's.

**The counts are `summary()`'s**, the function the desktop panel's own snapshot is built from, so
the header an operator reads here and the chip the panel shows cannot be two arithmetics. The
stopping point is arithmetic on the same list: what is left of it once the two splits the gate asks
about are clear.

**The remaining two lists are the store's own selectors** - `draw_worklist()` and
`null_worklist()`, which `GET /api/frames?draw=1` serves the page from the same call, so "which
frames is this pass about" has one answer rather than one per reader. They are filters over
`frames()`, in that order, so the names appear in the order the app would show them and a frame can
only be in a list for a reason the store can state - `suggested_nothing()` is the reason for the
draw list, and it is deliberately narrower than "no decision yet": the section's claim is *the
weight was asked and found nothing*, which is evidence about the model, while a frame nobody
suggested is outstanding work of a different kind and would enter the review pass the moment it
were.

No `generated_at` in the file, deliberately. The document exists to be comparable with a fresh
render, and a clock inside it would make every comparison fail; the file's mtime is the timestamp.
`--check` is that comparison - it writes nothing and exits 1 when the file on disk is not what this
would write - which is what turns a stale checklist from a silent problem into one anything can
notice (`make human-pass` writes it, `make human-pass HUMAN_PASS_ARGS=--check` verifies it).

`--status` is the same read with the document left out of it: the three sections' counts and one
exit code, nonzero while `test` or `valid` still holds a machine-only decision. That is the
question a *script* has - is the gate clear yet - and answering it by rendering markdown and
parsing it back would be a second implementation of the rule, which is the thing this module is
built not to have.

    sidecar/.venv/Scripts/python.exe -m annotate.human_pass [--out DIR] [--check | --status]
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path

SIDECAR_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = SIDECAR_DIR / "tools"
# `label_classes` and `workspace` live under `tools/`, which is not on `sys.path` when Python
# starts. The same block `annotate/run.py` carries, for the same reason: the package has no
# `__init__.py` to hold it once, and this is its second entrypoint.
if str(SIDECAR_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(SIDECAR_DIR))
if str(TOOLS_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(TOOLS_DIR))

from workspace import DEFAULT_OUT, MANIFEST_NAME, resolve_extras  # noqa: E402

from .store import (  # noqa: E402
    ANNOTATIONS_DIRNAME,
    CLASS_SLUGS,
    GATE_SPLITS,
    Frame,
    LabelStore,
    draw_worklist,
    gate_frames,
    null_worklist,
    review_rank,
    review_worklist,
    summary,
)

# Where the document goes: beside the staged set, i.e. in the workspace next to the
# `annotations-v2/` tree it describes. Moved with `SCANNCART_DATASET_ROOT` like everything else in
# there, and gitignored - it is a readout of one machine's store, not a file the repo carries.
HUMAN_PASS_NAME = "_human_pass.md"

# The two splits the human pass is *about*: the ones the acceptance gate requires to be clear of
# machine-only decisions (`accept_v2.GATE_SPLITS` - a weight's unread boxes in either make the
# acceptance number a measurement of the annotator). Spelled in the pass's own order, farthest
# first, because this module only ever prints them; the rule lives in `store.GATE_SPLITS` now that
# the page's draw view reads it too, and `tests/test_annotate_human_pass.py` pins both that this is
# that same tuple and that the pair is the gate's.
REVIEW_SPLITS: tuple[str, ...] = GATE_SPLITS

# How much of a unified diff `--check` prints. The whole document is well under this; the cap is
# there so a wholesale rewrite (a different workspace, say) cannot print 500 lines at a terminal.
DIFF_LINES = 40


@dataclass(frozen=True)
class Checklist:
    """The document's contents, before it is a document.

    A record rather than the string, so the numbers are testable without parsing markdown: a test
    asserts the header's counts, the three lists and the stopping point directly, and the text is
    left as a formatting concern.
    """

    counts: dict[str, int]      # machine-only decisions per split, from `summary()`
    splits: dict[str, str]      # name -> split, so a listed frame can name its own
    review: list[Frame]         # the app's review list, in the app's order
    draws: list[Frame]
    nulls: list[Frame]
    out: Path
    annotations: Path
    extras: tuple[Path, ...]

    @property
    def gate(self) -> int:
        """How much of `review` sits in the two splits the acceptance gate asks about.

        `store.gate_frames`, which is also what the page's gate readout counts - so the number an
        operator drives to zero and the number the page shows clearing are one call.
        """
        return len(gate_frames(self.review, self.splits))

    @property
    def remaining(self) -> int:
        """What the header falls to once those two are clear: the pass's stopping point.

        Arithmetic on the list rather than the `train` bucket, because the pair that has to be clear
        is a *rule* (`REVIEW_SPLITS`) and what is left over is whatever the store holds outside it -
        including a frame no split was ever assigned to, which is still outstanding.
        """
        return len(self.review) - self.gate

    @property
    def clean(self) -> bool:
        """Whether the gate has nothing left to ask about, which is what `--status` exits on.

        The one bit that is a verdict rather than a count, and it is the acceptance gate's own
        (`accept_v2.GATE_SPLITS`): a weight's unread boxes in the two measured splits are what make
        the acceptance number a measurement of the annotator. The other two sections are outstanding
        *evidence* (frames v1 cannot see, negatives with no null) rather than blockers - the draws
        move the per-distance recall the pass is run to quote, and the nulls move the
        false-positive half, but neither invalidates what `test` measures.
        """
        return self.gate == 0


def checklist(
    frames: list[Frame],
    splits: dict[str, str],
    provenance: dict,
    *,
    out: Path,
    annotations: Path,
    extras: tuple[Path, ...] = (),
) -> Checklist:
    """Everything the document says, as data. Pure: no filesystem, no clock."""
    return Checklist(
        counts=summary(frames, splits)["machine_only_by_split"],
        splits=splits,
        review=review_worklist(frames, splits),
        draws=draw_worklist(frames, splits, provenance),
        nulls=null_worklist(frames, splits),
        out=Path(out),
        annotations=Path(annotations),
        extras=tuple(Path(p) for p in extras),
    )


def pass_path(out: Path) -> Path:
    """Where the document for a staged set goes: beside it, in the workspace."""
    return Path(out).expanduser().parent / HUMAN_PASS_NAME


def _split_label(split: str) -> str:
    """The split as the document spells it. A frame nobody planned for is *unplaced*, not blank."""
    return split or "unplaced"


def _frame_lines(frames: list[Frame], splits: dict[str, str], cell: bool = True) -> list[str]:
    """One bullet per frame, in the order given: ``- `name`  -  cell  -  split test``.

    The cell is the `slug|distance` key the coverage tables use, so a name in this document can be
    looked up in a report without a second translation; the negatives carry no distance and so no
    cell (`cell=False`), which is also why they are the one list that does not name a class.
    """
    lines = []
    for frame in frames:
        parts = [f"`{frame.name}`"]
        if cell:
            parts.append(frame.cell)
        parts.append(f"split {_split_label(splits.get(frame.name, ''))}")
        lines.append("- " + "  -  ".join(parts))
    return lines


def _extras_note(extras: tuple[Path, ...]) -> str:
    """What else the worklist covers, or a sentence saying that it covers nothing else.

    Named rather than omitted, because the hard negatives are a staged set of their own: if this
    line is empty the null list below is empty *by construction*, and an operator who staged them
    somewhere else needs to see that rather than read an empty section as a finished one.
    """
    if not extras:
        return "no extra staged set (the hard negatives are not in this worklist)"
    return ", ".join(str(path) for path in extras)


def render(items: Checklist) -> str:
    """The document, as one string. Pure: a `Checklist` in, markdown out."""
    ordered = sorted(
        items.counts.items(), key=lambda item: (review_rank(item[0]), item[0])
    )
    lines: list[str] = [
        "<!-- Rendered from the annotator's store by `annotate.human_pass` - do not edit by hand.",
        "     A hand edit is exactly the drift this file exists to prevent: the counts and the frame",
        "     lists below are read back off `provenance.json`, the staged manifests and the labels",
        "     on disk, and they move every time a frame is confirmed, drawn or marked null.",
        "",
        f"       set      {items.out}",
        f"       labels   {items.annotations}",
        f"       also     {_extras_note(items.extras)}",
        "",
        "     Rewrite it with `make human-pass` (or "
        "`sidecar/.venv/Scripts/python.exe -m annotate.human_pass`); `--check`",
        "     fails instead of writing when the file no longer matches what the store says. -->",
        "",
        "# The human pass (valid + test only)",
        "",
        "Machine side is done: every mid/far frame the v1 weight could pre-fill is saved as a",
        "**machine-only** decision, and the staleness gate refuses to measure the set until a",
        "person clears the ones in `valid` and `test`.",
        "",
    ]
    if ordered:
        breakdown = " / ".join(f"{_split_label(split)} {n}" for split, n in ordered)
        lines.append(
            f"Header right now: **{len(items.review)} machine-only awaiting review** ({breakdown})."
        )
    else:
        lines.append(
            "Header right now: **nothing is machine-only** - every decision on disk has been touched "
            "by a person."
        )
    lines += ["", "## 1. Review pass - `u`, then `f` or fix", ""]

    inside = gate_frames(items.review, items.splits)
    named = {frame.name for frame in inside}
    rest_frames = [f for f in items.review if f.name not in named]

    if not items.review:
        lines += [
            "Nothing is awaiting review: no decision on disk is a weight's unread work. The lists",
            "below are the whole of what is left.",
            "",
        ]
    else:
        lines += [
            "Press `u` (Review). The list is ordered test first, then valid, then train, and within a",
            "split the far/mid frames and the ones holding two or more items come first - those are the",
            "frames the verdict is quoted from, so the pass's first hour lands on the evidence that",
            "matters. The info line under the image shows the split.",
            f"`f` = confirm as-is (weight was right), `1`-`{len(CLASS_SLUGS)}` pick the class, drag on the image to",
            "draw, `Delete` removes the last box, `Enter` saves an edit, `->`/`<-` move.",
            "",
        ]
        if items.gate:
            lines += [
                f"Work from the top until the header count reaches **{items.remaining}** - that is when",
                f"`test` and `valid` are done ({items.gate} frames). The names below are in the app's own",
                "order, so the frame at the top of this list is the frame the page opens on.",
                "",
                f"The {items.gate} in `test` and `valid`:",
                "",
                *_frame_lines(inside, items.splits),
                "",
            ]
        else:
            lines += [
                "`test` and `valid` are already clear: the gate has nothing left to ask for, and the",
                "frames below are all outside it. Nothing to do in this section.",
                "",
            ]
        if rest_frames:
            lines += [
                f"The other {len(rest_frames)} machine-only frame(s) are outside `test`/`valid` (`train`, or",
                "unplaced). A weight's unread boxes are tolerated there, so they are not part of the",
                "stopping point above - they are listed so that the rest of the outstanding work is",
                "visible, and are worth taking only once the two splits above are clean.",
                "",
                *_frame_lines(rest_frames, items.splits),
                "",
            ]

    lines += ["## 2. Draw pass - the far frames v1 could not see at all", ""]
    if items.draws:
        lines += [
            f"{len(items.draws)} frame(s): every `far` frame in `test`/`valid` the weight was asked about",
            "and proposed nothing for. They are in the app's list order, inside the unlabeled `far`",
            "block that runs ahead of everything already decided (`train` frames are mixed into it -",
            "skip those). Open each, draw every item you can see, and `Enter` to save:",
            "",
            *_frame_lines(items.draws, items.splits),
            "",
        ]
    else:
        lines += [
            "None left: every `far` frame in `test`/`valid` the weight was asked about carries a",
            "decision now.",
            "",
        ]

    lines += ["## 3. Nulls - the hard negatives", ""]
    if items.nulls:
        lines += [
            f"{len(items.nulls)} frame(s), one keypress each: open it, look at the counter, press `n`",
            "(Mark null) - it saves immediately. A frame with an actual item in it is not one of",
            "these; the store refuses a null on a product frame anyway.",
            "",
            *_frame_lines(items.nulls, items.splits, cell=False),
            "",
        ]
    elif not items.extras:
        lines += [
            "No hard-negative set is in this worklist at all - `also` above names none, so this",
            "section has nothing to report rather than nothing to do. If the negatives are staged,",
            "`--out` is pointing at the wrong workspace (or `--no-extras` was passed): they are a set",
            "of their own, staged beside the set that is being labelled.",
            "",
        ]
    else:
        lines += [
            "None left: every hard negative staged into `test`/`valid` carries its null.",
            "",
        ]

    lines += [
        "When those three lists are empty, say so and the rest runs: rebuild `merged-v2`, doctor,",
        "train v2 on the merged set, `--val` with the per-distance grid, then `accept_v2` beside v1.",
        "`docs/RUN_SHEET.md` has the commands and what each number has to report before the next step",
        "is worth running.",
    ]
    return "\n".join(lines) + "\n"


def status(items: Checklist) -> list[str]:
    """The pass's progress as lines, for a caller that wants the numbers and not the document.

    The same `Checklist` `render()` takes, so a script reading these counts and a person reading
    the file cannot be told two different things - and the lines quote the sections in the order
    the document lists them, so the fourth line is the same stopping point the header prints.

    The last line is the verdict: `blocked` while the gate splits hold machine-only decisions,
    `clear` once they do not. `main` turns exactly that into the exit code.
    """
    pair = "/".join(REVIEW_SPLITS)
    other = len(items.review) - items.gate
    lines = [
        f"set      {items.out}",
        f"labels   {items.annotations}",
        f"also     {_extras_note(items.extras)}",
        f"review   {len(items.review)} machine-only ({items.gate} in {pair}, {other} outside them)",
        f"draw     {len(items.draws)} far frame(s) in {pair} a weight was asked about and found nothing in",
        f"nulls    {len(items.nulls)} hard negative(s) in {pair} with no null yet",
    ]
    if items.clean:
        lines.append(f"gate     clear - nothing machine-only in {pair}")
    else:
        lines.append(
            f"gate     blocked by {items.gate} - clear {'them' if items.gate != 1 else 'it'} "
            f"in {pair} before the acceptance number means anything"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    """The three flags that decide *which store* is read, plus the two read-only modes.

    The same `--out`/`--extras`/`--annotations` trio `annotate/run.py` takes, and for the same
    reason: a checklist about a set staged elsewhere has to be reachable, and the parts of that
    which could drift silently - the default set and the extras *union* - are imported from
    `workspace` rather than retyped here.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT), help="the staged set the pass is about (has the manifest)"
    )
    ap.add_argument(
        "--extras",
        action="append",
        default=[],
        metavar="DIR",
        help="another staged set in the worklist (repeatable). Adds to the hard negatives staged "
        "beside --out - pass --no-extras to mean only what you name",
    )
    ap.add_argument(
        "--no-extras",
        action="store_true",
        help="report only the sets named with --extras, not the staged hard negatives beside --out",
    )
    ap.add_argument(
        "--annotations",
        default="",
        # Spelled from the store's own constant rather than typed into the help text: a second
        # copy in prose is a second copy, and this one is read by whoever is deciding where to
        # point the tool.
        help=f"where the labels live (default: <workspace>/{ANNOTATIONS_DIRNAME})",
    )
    # One mode at a time: both write nothing, but they answer different questions ("is the file
    # current" versus "is the pass finished") and a run that asked both would have only one exit
    # code to answer with.
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 when the file on disk is not what would be written",
    )
    mode.add_argument(
        "--status",
        action="store_true",
        help="write nothing; print the three sections' counts and exit 1 while `test`/`valid` "
        "hold machine-only decisions",
    )
    return ap


def _report_difference(path: Path, expected: str, current: str) -> None:
    """Print how the file on disk differs, capped. `expected` is what the store says now."""
    diff = list(
        difflib.unified_diff(
            current.splitlines(),
            expected.splitlines(),
            fromfile=f"{path} (on disk)",
            tofile="a fresh render",
            lineterm="",
            n=1,
        )
    )
    print("\n".join(diff[:DIFF_LINES]))
    if len(diff) > DIFF_LINES:
        print(f"... and {len(diff) - DIFF_LINES} more line(s) of the diff")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised by hand
    args = build_parser().parse_args(argv)
    out = Path(args.out).expanduser()
    if not (out / MANIFEST_NAME).is_file():
        raise SystemExit(f"no manifest at {out} - point --out at a staged set (clean_v2.py clean)")
    annotations = (
        Path(args.annotations).expanduser()
        if args.annotations
        else out.parent / ANNOTATIONS_DIRNAME
    )
    extras = resolve_extras(out.parent, args.extras, include_defaults=not args.no_extras)

    store = LabelStore(out=out, annotations=annotations, extras=extras)
    items = checklist(
        store.frames(),
        store.splits(),
        store.provenance(),
        out=out,
        annotations=annotations,
        extras=tuple(extras),
    )
    if args.status:
        # Before `render`, so it is plain that this mode never builds - let alone writes - the
        # document. The exit code is the whole point: 1 while the gate splits are dirty.
        print("\n".join(status(items)))
        return 0 if items.clean else 1

    text = render(items)
    path = pass_path(out)

    if args.check:
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            print(f"{path} is missing - run without --check to write it")
            return 1
        if current == text:
            print(f"{path} is current")
            return 0
        print(f"{path} no longer matches the store:")
        _report_difference(path, text, current)
        return 1

    path.parent.mkdir(parents=True, exist_ok=True)
    # `newline="\n"` on purpose: the document is compared byte for byte by `--check`, and the
    # default translation would make that comparison depend on the platform that wrote last.
    path.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"wrote {path}\n"
        f"  {len(items.review)} machine-only awaiting review "
        f"({items.gate} of them in {'/'.join(REVIEW_SPLITS)})"
    )
    print(f"  {len(items.draws)} far frame(s) to draw, {len(items.nulls)} null(s) to mark")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
