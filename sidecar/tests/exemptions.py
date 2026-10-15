"""The one shape behind this suite's exemption tables: a phrase a scan is told to leave alone.

Three guards keep such a table. `test_docs_claims` excuses phrases in the docs it reads,
`test_class_order` excuses sites in the tools tree, and `test_docs_interpreters` excuses pasted
commands in the instruction docs. What each one *scans* differs; what it writes down does not - a
phrase, the places that carry it, the reason that occurrence is not a finding, and the one shape it is
excused from - and neither does the rule that keeps it honest: the phrase has to still be there, or
the exemption is covering text nobody read and the next thing with that shape is waved through.

A *shape* is the scan step a row is written for, as the guard spells it: the arm's own name where a
scan has one step, and arm plus pattern where it has several. That is the part that keeps an exemption
from spreading - a row is only ever answered by the shape it names, so a phrase written for a count
cannot silence the same words under another pattern of the same scan.

So the row, the lookup and that rule live here, and a guard supplies only its own shape names and its
own reader. `assert_still_present` takes the reader because what a place *is* differs per guard (a doc's
own text; the sites a scan rendered under one path) while the rule does not - and a row whose phrase is
gone fails in the same words wherever it was written. A guard that keeps no rows at all has to say so
(`designed_empty=True`), because an empty table is the one failure that rule cannot see from inside.

And a scan that skips something records a `Waiver` for it - what was let through, where, under which
shape and for which reason - so the guard's own test can state the rows it applied and a failure report
can carry them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable


@dataclasses.dataclass(frozen=True)
class Exemption:
    """One phrase a scan is told to leave alone.

    `phrase` is the text that does the silencing and the row's identity; `places` is every place it is
    allowed to appear - matched by the lookup, and all of them read by the liveness rule - so the same
    phrase excused for the same reason in two docs is one row rather than two. `shape` is a single
    scan step and never a set: that is what keeps an exemption written for one shape from silencing
    another.
    """

    phrase: str
    reason: str
    shape: str
    places: tuple[str, ...]

    def __post_init__(self) -> None:
        # A row that pins nothing fails at construction rather than passing green, the rule the wording
        # engine holds its tables to: a blank phrase silences everything (the empty string is in any
        # text), a blank reason leaves the next reader with no way to judge the row, a row with no place
        # is never asked about, and a row with no shape cannot be asked for.
        if not self.phrase:
            raise ValueError("an exemption with no phrase would match any text it was asked about")
        if not self.reason.strip():
            raise ValueError(f"the exemption for {self.phrase!r} states no reason")
        if not self.places:
            raise ValueError(f"the exemption for {self.phrase!r} names no place")
        if not self.shape:
            raise ValueError(f"the exemption for {self.phrase!r} names no shape")


@dataclasses.dataclass(frozen=True)
class Waiver:
    """One row applied: what a scan skipped, where, under which shape, and why.

    The trace an exemption leaves behind when it does its work. A scan collects these as it runs, so
    "this was accepted because a row excuses it" becomes a fact the guard can state - which is the one
    thing a table cannot say for itself, since a row is *live* whether or not any scan reaches it. The
    row's own reason travels with the record, because that is what a reader needs when asking why a
    sentence or a site was let through.
    """

    shape: str
    place: str
    phrase: str
    reason: str

    def __str__(self) -> str:
        """One line, for a failure message: what was let through, where, under which shape, and why."""
        return f"{self.phrase!r} in {self.place} [{self.shape}]: {self.reason}"


class Exemptions:
    """A guard's rows, asked as `reason(shape, place, text)`.

    Keyed by phrase rather than by (place, phrase): a phrase and the reason it is allowed therefore
    travel in one object, and a scan asks one question - is this text excused where I found it, under
    the shape I matched it with? - instead of searching places itself. Two rows for one phrase are
    refused, because the second one's reason and shape would be unreachable: whichever row answers
    first, the other reads as a rule while covering nothing. A caller that wants the `Waiver` for each
    application passes the list it collects; one that only wants the verdict passes nothing.
    """

    def __init__(self, *rows: Exemption) -> None:
        self._rows: dict[str, Exemption] = {}
        for row in rows:
            if row.phrase in self._rows:
                raise ValueError(
                    f"more than one exemption for {row.phrase!r} - the first answers and the rest "
                    "read as rules that cover nothing"
                )
            self._rows[row.phrase] = row

    @property
    def rows(self) -> tuple[Exemption, ...]:
        """Every row, in the order the table was written."""
        return tuple(self._rows.values())

    @property
    def places(self) -> tuple[str, ...]:
        """Every place any row names, in the table's order - what a caller matches its own rendered
        text against when the place has to be found inside a longer string."""
        return tuple(dict.fromkeys(place for row in self._rows.values() for place in row.places))

    @property
    def shapes(self) -> frozenset[str]:
        """The shapes these rows excuse. A guard checks its own scan steps against this: a row whose
        shape no scan asks with is a rule nothing can reach."""
        return frozenset(row.shape for row in self._rows.values())

    def __len__(self) -> int:
        return len(self._rows)

    def reason(
        self, shape: str, place: str, text: str, waived: list[Waiver] | None = None
    ) -> str | None:
        """The reason `text` is not a finding *for this shape* in this place, if a row covers it.

        The shape and the place are both required matches, so a row written for one scan step cannot
        silence another, and a phrase excused in one doc cannot silence the same words in a different
        one. Each application is recorded in `waived` when the caller collects them, so a scan can say
        which rows it applied rather than leaving that to be inferred from the rows still being there.
        """
        for row in self._rows.values():
            if row.shape == shape and place in row.places and row.phrase in text:
                if waived is not None:
                    waived.append(
                        Waiver(shape=shape, place=place, phrase=row.phrase, reason=row.reason)
                    )
                return row.reason
        return None


def waived_note(waived: Iterable[Waiver]) -> str:
    """What a scan let through, as a line for a failure message - and nothing at all if it let nothing
    through.

    The report half of the trace: a reader looking at a flagged sentence or an offender list should also
    see the ones a row excused, with the reasons those rows give, so "nothing else was touched" is
    stated rather than assumed.
    """
    collected = list(waived)
    if not collected:
        return ""
    return "\n(waived by an exemption: " + "; ".join(str(waiver) for waiver in collected) + ")"


def assert_still_present(
    table: Exemptions, read: Callable[[str], str], *, designed_empty: bool = False
) -> None:
    """Every row's phrase still occurs in every place the row names, or the exemption covers nothing.

    `read` is the caller's, because what a place is differs per guard - a doc's own text, or the sites
    one path's scan rendered - while the rule is the same either way. One caller per guard, from its
    own live-exemptions test, so the failure names the phrase, the place and the reason in the same
    words wherever the rot happens.

    An empty table has to be *declared* empty. With no rows there is nothing to check, so this rule
    cannot see - from the inside - the one failure it exists for: a table whose rows were deleted
    passes trivially and says nothing. `designed_empty=True` is that declaration, and it is read in both
    directions, so it cannot outlive the emptiness it describes: a table that grows its first row has to
    drop it in the same edit rather than keep a note that has stopped being true.
    """
    if designed_empty:
        assert len(table) == 0, (
            f"this table is declared empty on purpose and holds {len(table)} row(s) - drop the "
            "declaration, or say which row it is there for"
        )
    else:
        assert len(table), (
            "this table is empty and nothing says that is deliberate: either its rows are gone - the "
            "deletion nothing below this line can see, since with no rows every phrase trivially "
            "holds - or this guard keeps none for now, in which case it says so with "
            "`designed_empty=True` and the reason the table is empty is written beside the table"
        )
    for row in table.rows:
        for place in row.places:
            assert row.phrase in read(place), (
                f"{place} no longer holds {row.phrase!r}, so the exemption ({row.reason}) covers "
                "nothing - drop it, or point it at what replaced it"
            )
