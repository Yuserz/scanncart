"""One engine for every claim this repo makes, and the one way they are reported.

A *claim* here is a statement the code makes about itself - "the rule catches the phantom shape at
all", "no measured real detection is inside the rule", "the shipped tolerance is on the ladder" -
next to the reading that decides it. Five surfaces decide claims, and until this module existed four
of them carried their own copy of the plumbing: `clamp_probe.py` and `unsure_probe.py` were the same
`Check`/`verdict`/`strict_failed` block twice over (differing only in a hand-tuned pad width, 52
against 56), `spec_check.py` was a third spelling of the same idea for its PRD targets, and the two
text guards in `sidecar/tests/` reported their findings as bespoke assertion prose. So a claim was
declared four ways, counted four ways, and failed in four voices - and the sentence a reader trusts
("all 5 claims hold") was produced by each of them separately.

This module is that copy, taken once. It owns:

- **the record.** `Claim(statement, reading, holds)` - what is claimed, what decided it, and the
  answer. Both halves are required: a claim with nothing next to it is a verdict with no evidence,
  which is the shape a reader cannot check.
- **the table.** `Claims` is the claims one run or one scan decided, plus the sentence its failures
  read as. It validates itself at construction, because every way of writing a table that verifies
  nothing is a way of writing one that passes: no rows at all, one statement listed twice (the two
  read as one claim and the second's reading is never looked at), more failures than decided claims,
  and a blank noun or verb in the verdict it will print.
- **one verdict, one report, one exit.** `verdict()` is the single sentence - `all 5 claims hold`, or
  `1 of 5 claims FAIL: <the failing statements>` - and `exit_code(strict)` is the single rule behind
  the three tools' `--strict`. The wording stays the table's: `kind`, `holds_verb` and `fails_verb`
  are columns rather than constants precisely because two of them are *documented output* -
  `docs/RUN_SHEET.md` states `all 5 PRD targets pass`, and `docs/MODEL_TRAINING.md` pastes a
  `spec_check` row - so this engine changes the plumbing and not one documented sentence.

`rows` is what the caller decided to carry, and the two directions are both honest: a tool that
prints its claims carries every one it decided, which is what makes `all N claims hold` a reading
rather than a summary of the ones it happened to remember; a *scan* that only ever reports on failure
carries the findings and states how many claims it decided in `read`, so its failure reads as
`2 of 36 claims FAIL` in the same words a tool's does. `row` is the one escape hatch, for a table
whose reading has columns rather than a sentence (`spec_check` prints a measured value, its unit and
the target it was compared against); a table that does not supply one gets the shared row.

This module imports nothing but the standard library on purpose: the guards that read prose live in
the `docs`-marked subset, which runs on a bare checkout with pytest and pydantic and no numpy, cv2 or
ultralytics at all.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

__all__ = ["Claim", "Claims", "findings"]


@dataclasses.dataclass(frozen=True)
class Claim:
    """One statement, the reading that decided it, and whether it held.

    `statement` is the claim in the words a reader would repeat it in ("no measured real detection is
    inside the rule"); `reading` is the measurement or the scan result that settles it (`0 of 60
    product detections`). Keeping the two apart is what lets one report render both a tool's
    measured row and a guard's finding - a finding is exactly a claim whose reading came out wrong.
    """

    #: What is claimed, in the sentence the report will print.
    statement: str
    #: The reading that decides it - a number, a count of items, or why the scan flagged it.
    reading: str
    #: Whether the claim held. `False` is the only thing a failure is.
    holds: bool = True

    def __post_init__(self) -> None:
        # A claim that names nothing and a claim with no reading are the same defect twice: the
        # first prints a blank row, the second prints a verdict nobody can check. Both fail here
        # rather than in a report that looks like a passing one.
        if not self.statement.strip():
            raise ValueError("a claim with no statement names nothing to check")
        if not self.reading.strip():
            raise ValueError(
                f"the claim {self.statement!r} carries no reading, so nothing decides it - a claim "
                "is a statement alongside the number or the scan result that settles it"
            )

    def line(self, pad: int) -> str:
        """The row a reader sees: the mark, the statement, and the reading that decided it.

        `pad` is the caller's - `Claims` computes one width for the whole table so the readings line
        up, which is what the two probes used to hand-tune per file (52 in one, 56 in the other,
        neither of them a fact about anything).
        """
        return f"  [{'PASS' if self.holds else 'FAIL'}] {self.statement:<{pad}} {self.reading}"


@dataclasses.dataclass(frozen=True)
class Claims:
    """The claims one run or one scan decided, and the one report they are rendered through.

    Everything a failure needs is here, so no caller composes its own: `verdict()` is the sentence,
    `block()` is the section a tool prints (header, every row, the verdict), `failure()` is what a
    test raises (the verdict, the failing rows, the remedy), and `exit_code()` is what a `--strict`
    run returns. The three tools used to spell each of those out for themselves.
    """

    #: Names what was checked, in the words a failure message opens with.
    name: str
    #: The module that owns the facts these claims are about, for the failure messages.
    owner: str
    #: The claims carried into the report - every one decided, or (for a scan) only the findings.
    rows: tuple[Claim, ...]
    #: How many claims were decided, when that is not `len(rows)`. Never smaller than `len(rows)`:
    #: a table cannot fail more claims than it decided.
    read: int | None = None
    #: The noun the verdict counts ("claims", "PRD targets").
    kind: str = "claims"
    #: The verb for the all-hold direction ("hold", "pass").
    holds_verb: str = "hold"
    #: The verb for the failing direction ("FAIL", "fail").
    fails_verb: str = "FAIL"
    #: What a reader should do about it, appended to a failure - the same-edit rule, in the words
    #: of whoever wrote the claim.
    remedy: str = ""
    #: A table whose reading has columns renders its own row; everything else gets `Claim.line`.
    row: Callable[[Claim], str] | None = None

    def __post_init__(self) -> None:
        # Every way of writing a table that verifies nothing fails at construction rather than
        # passing green, the rule `exemptions.py` and `wording.py` hold their own tables to.
        if not self.name.strip():
            raise ValueError("a claim table with no name cannot report a failure in any words")
        if not self.owner.strip():
            raise ValueError(f"{self.name}: no owner named, so a failure cannot say where the fact lives")
        if not self.rows:
            raise ValueError(
                f"{self.name}: a claim table with no rows verifies nothing - with nothing in it "
                "every check that reads it passes, so a run or a scan that decided no claim at all "
                "has to fail here rather than report a clean one"
            )
        if self.decided < len(self.rows):
            raise ValueError(
                f"{self.name}: {len(self.rows)} row(s) out of {self.decided} decided "
                f"{self.kind} - a table cannot fail more than it decided"
            )
        seen: set[str] = set()
        for row in self.rows:
            if row.statement in seen:
                raise ValueError(
                    f"{self.name}: {row.statement!r} is listed twice, so one claim is counted as "
                    "two and the second reading is never looked at"
                )
            seen.add(row.statement)
        for word in (self.kind, self.holds_verb, self.fails_verb):
            if not word.strip():
                raise ValueError(f"{self.name}: the verdict it prints has a blank word")

    @property
    def decided(self) -> int:
        """How many claims were decided: the declared count, or one per row."""
        return len(self.rows) if self.read is None else self.read

    @property
    def failed(self) -> tuple[Claim, ...]:
        """The claims that did not hold, in table order - the ones a failure names."""
        return tuple(row for row in self.rows if not row.holds)

    @property
    def holds(self) -> bool:
        """Whether every claim held."""
        return not self.failed

    @property
    def pad(self) -> int:
        """The width the shared row pads statements to: the longest plus two."""
        return max(len(row.statement) for row in self.rows) + 2

    def line(self, row: Claim) -> str:
        """One row, in this table's own layout."""
        return self.row(row) if self.row is not None else row.line(self.pad)

    def verdict(self) -> str:
        """The single sentence: what held, or which claims did not.

        One spelling for the whole repo, with the words the table supplies - `all 5 claims hold`,
        `1 of 9 claims FAIL: the rule is on in the profile this measured`, `all 5 PRD targets pass`.
        Naming the failures rather than counting them is the point: a count alone sends the reader
        back to a report to find out which one moved.
        """
        bad = self.failed
        if not bad:
            return f"all {self.decided} {self.kind} {self.holds_verb}"
        return (
            f"{len(bad)} of {self.decided} {self.kind} {self.fails_verb}: "
            + "; ".join(row.statement for row in bad)
        )

    def block(self) -> list[str]:
        """The section a tool prints: a blank line, the header, every row, the verdict.

        The header is the table's `kind`, which is the word the rows are counted in, so the block
        reads as `claims` / `[PASS] …` / `all 5 claims hold` for one table and `PRD targets` / … /
        `all 5 PRD targets pass` for another.
        """
        return ["", self.kind, *(self.line(row) for row in self.rows), "", f"  {self.verdict()}"]

    def failure(self, note: str = "") -> str:
        """What a test raises: the verdict, the failing rows, and what to do about them.

        A failure has to name the claim, show the reading that came out wrong, and say what the
        reader does next - so the verdict opens it, the failing rows carry the detail, and the last
        line names the owner of the fact (with the remedy, when the table has one). Only the failing
        rows are printed: a table carrying every claim it decided is a report, and a failure is a
        list.

        `note` is what the caller's own scan adds - a guard's waived exemptions, for one - rendered
        last, so the one report shape still leaves room for the trace only that caller could have.
        """
        tail = f"the fact these claims are about is owned by `{self.owner}`"
        if self.remedy.strip():
            tail += f" - {self.remedy}"
        lines = [f"{self.name} - {self.verdict()}", *(self.line(row) for row in self.failed), tail]
        if note.strip():
            lines.append(note)
        return "\n".join(lines)

    def exit_code(self, strict: bool) -> int:
        """What a `--strict` run returns: non-zero exactly when a claim did not hold.

        The rule the three tools spelled out separately (`1 if args.strict and strict_failed(...)
        else 0`), in one place, because the flag is the caller's and the answer is the table's.
        """
        return 1 if strict and not self.holds else 0

    def assert_holds(self, note: str = "") -> None:
        """Raise this table's failure unless every claim held - what the text scans call."""
        if not self.holds:
            raise AssertionError(self.failure(note))


def findings(
    name: str,
    owner: str,
    rows: Iterable[tuple[str, str]],
    read: int,
    remedy: str = "",
) -> Claims:
    """A scan's table, from its `(where, why)` findings and the number of claims it decided.

    The text guards report the same shape a measured run does, and this is the one line between
    their scans and the engine: each finding is a claim that did not hold - where the claim is
    ("`docs/RUN_SHEET.md:12: '0.918 floor'`) and what the scan found instead ("the floor is 0.85")
    - and `read` is how many claims the scan decided, so a failure reads `1 of 41 claims FAIL`
    rather than counting only the ones that went wrong. A scan that decided nothing is refused by
    `Claims` itself, which is the failure a broken pattern produces.

    Called only when a scan found something - a clean scan has no table, and the refusal above is
    why that is not a hole: a guard that reaches this with no findings and no claim read is one whose
    pattern stopped matching, and it fails rather than reporting a green run.
    """
    return Claims(
        name=name,
        owner=owner,
        read=read,
        remedy=remedy,
        rows=tuple(Claim(where, why, holds=False) for where, why in rows),
    )
