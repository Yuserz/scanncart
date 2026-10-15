"""Drift guard: a stated count has to match the items enumerated under it.

The sibling of `test_docs_claims.py`, pointed at a different failure. That guard checks a sentence
against the *code* fact it copies - a class count, a default, a floor. This one checks a sentence
against the text underneath it: `Four states have no answer here:` followed by five of them, and
`Five things make this not a re-run:` followed by four. Nothing outside the file can answer that,
which is why it survived: both halves of the claim read as true one at a time, and the reader who
would have counted them is the reader the sentence was written for.

Not hypothetical - it is the shape that turned up three times in one session, twice in a *prose*
sentence and once in a list - so the guard reads the tools whose docstrings state counts and the
docs they were found in, and it is written so the prose case cannot hide: a claim whose items cannot
be counted is not skipped, it is **asked for a row** (below).

The claim shape. A capitalised number word, up to three words of qualification, and a count noun -
`Three rules`, `Five of those states`, `Two different numbers`, `Two things this print has said`.
Capitalisation is the anchor, and it is what keeps ordinary prose out: the same words mid-sentence
(`two runs hold them`, `a class the split holds no instances of`) are not claims, and a guard that
flagged them would be turned off within a week. The count nouns are the ones these corpora actually
use, listed rather than guessed at; a sentence counting with a noun that is not in the list is out of
scope rather than silently passed, because the list is this guard's boundary and it is visible.

The items. The run of item-led paragraphs immediately after the claim's own paragraph: one opening
with `**` (the tools' docstring style) or with a `-`/`*`/`1.` list marker. Wrapped lines belong to
the item above them, a nested item does not start a second one, and a paragraph that does not open
with a marker ends the run - so a table, a code fence or a paragraph of prose below a claim
contributes nothing and the claim reads as *uncountable* rather than as a count of zero.

The rows. A claim this scan cannot count is not excused by default: it needs a row in `EXEMPT` naming
its phrase, the place, and *why* its items are not a list. That is the one edit the next person makes
when they write `Three reasons:` in prose - one line at the foot of this file - and it is what keeps
the prose case from being the blind spot it was. A claim whose items *are* a list gets a row only
when the number is deliberately not that list's length, with the reason (`MODEL_TRAINING.md`'s "Two
things about how it is measured, and one about what it means:" is the one today). The two kinds are
different *shapes*, so a row written for prose text cannot silence a list whose length disagrees.

What this deliberately does not do. It does not read the *content* of the items - a list of five
wrong things under `Five things` passes, because knowing they are wrong is a reviewer's job and not a
pattern's. It does not read the `CLAUDE.md` files, the annotator or `app/`: each carries the same
shape and each is a corpus of its own (the CLAUDE.md files are read by `test_docs_claims.py` for
their code claims). It does not read a number that is a *data* count - the run sheet's
`858/1433 decided` moves with the dataset, and a guard on those fails on a labeling session, which is
how a guard stops being read. And it asserts nothing about bytes: the repo stores LF
(`.gitattributes`), a Windows checkout may hand it CRLF, and this guard is about counts.
"""

from __future__ import annotations

import dataclasses
import re
from functools import lru_cache
from pathlib import Path

import pytest

from tests.exemptions import Exemption, Exemptions, Waiver, assert_still_present, waived_note

pytestmark = pytest.mark.docs

REPO_ROOT = Path(__file__).resolve().parents[2]

# The corpus: the tools whose docstrings state counts, and the docs. `sidecar/tools/*.py` is where
# this was found by hand, so the docstrings are read as text - a docstring is prose a reader reads,
# and nothing compiles its arithmetic.
TOOLS_GLOB = "sidecar/tools/*.py"
DOCS_GLOB = "docs/*.md"

# The count words the claim shape is anchored on. `One` is out on purpose: "One thing this does" is a
# sentence about a single thing rather than a claim with a list under it, and it is the shape a list
# is least likely to follow.
NUMBER_WORDS = {
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
    "Six": 6,
    "Seven": 7,
    "Eight": 8,
    "Nine": 9,
    "Ten": 10,
    "Eleven": 11,
    "Twelve": 12,
}

# The nouns these corpora count with. Measured, not guessed: every claim in the two corpora uses one
# of them, and a claim using another is out of scope until it is added here - a visible edit rather
# than a silent pass.
COUNT_NOUNS = (
    "answers",
    "arms",
    "cases",
    "copies",
    "defects",
    "directions",
    "documents",
    "doors",
    "files",
    "findings",
    "flags",
    "halves",
    "jobs",
    "kinds",
    "layers",
    "modes",
    "numbers",
    "outcomes",
    "parts",
    "paths",
    "reasons",
    "rules",
    "sides",
    "sources",
    "spellings",
    "states",
    "steps",
    "sweeps",
    "switches",
    "things",
    "values",
    "ways",
)

CLAIM = re.compile(
    r"(?<![A-Za-z0-9-])(?P<word>" + "|".join(NUMBER_WORDS) + r")"
    r"(?: [A-Za-z][A-Za-z-]*){0,3} (?P<noun>" + "|".join(COUNT_NOUNS) + r")\b"
)

# A paragraph that opens with an item marker: a Markdown bullet or ordered-list item, or the
# tools' `**Heading.**` docstring style. The indent comes along so a nested item can be told from a
# wrapped line, and a dash with no space after it (`    --conf-sweep`) is not a marker.
#
# The bold form is anchored on where the heading's own stop sits - *inside* the span, as in
# `**Heading.** body` - and that is load-bearing rather than cosmetic: a *paragraph that continues
# the argument* can also open with a bold span, and `MODEL_TRAINING.md`'s "**The same recall is
# reported per distance**, because ..." follows the `Three outcomes` claim, so a walk that read any
# first bold span as an item would report that prose claim as a list of one. A heading is a span whose
# own last character is a full stop or a colon; a bolded phrase with the stop outside it is emphasis
# in a sentence.
ITEM = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<bullet>[*+-])[ \t]|(?P<ordered>\d+[.)])[ \t]|(?P<bold>\*\*[^*]*[.:]\*\*))"
)

ARM = "stated counts"

# One shape per (scan step, pattern, judgement), which is the rule `tests/exemptions.py` states: a
# row is answered only by the shape it names. `list` is a claim whose items are a list, and a row may
# name it only when the number is deliberately not that list's length; `prose` is a claim whose items
# are not a list at all, which needs a row saying so.
SHAPE_LIST = f"{ARM}: list"
SHAPE_PROSE = f"{ARM}: prose"
SHAPES = (SHAPE_LIST, SHAPE_PROSE)

# The claims the corpus carries today, measured rather than assumed: 16 whose items are a list this
# guard checks directly, and 20 more whose items are prose and which the table below names. A scan that
# stopped matching - a regex edit, a file renamed out of a glob - would otherwise pass by finding
# nothing, which is the one failure this guard cannot report from the inside.
CLAIMS_READ = 36


@dataclasses.dataclass(frozen=True)
class Claim:
    """One stated count, with what is under it.

    `headline` is the matched text (`Five states have no answer here`) and `snippet` is the claim's own
    paragraph, which is what an exemption row is matched against: a row excuses a sentence, not a
    file. `items` is `None` when the paragraphs below the claim open with no marker at all - the
    uncountable case, which is not the same as a count of zero.
    """

    path: str
    line: int
    stated: int
    headline: str
    snippet: str
    items: int | None


def _files() -> tuple[str, ...]:
    """The corpus, as repo-relative paths: every tool and every doc the two globs match."""
    return tuple(
        sorted(
            str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            for glob in (TOOLS_GLOB, DOCS_GLOB)
            for path in REPO_ROOT.glob(glob)
        )
    )


@lru_cache(maxsize=None)
def _text(path: str) -> str:
    """A corpus file's text, newline-normalized: the repo stores LF and a Windows checkout may hand
    this CRLF, while the claims this reads are about counts rather than about bytes."""
    return (REPO_ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")


def _flattened(path: str) -> str:
    """A file's text with every run of whitespace collapsed to one space.

    This is the reader the exemption rows are held live against, and it is the same space a claim's
    snippet is matched in, because both sides of that comparison are prose: a sentence here is wrapped
    at 100 columns, so a phrase taken from it may straddle a line break. Collapsing is what makes a
    row's phrase a phrase rather than a column position.
    """
    return " ".join(_text(path).split())


def _style(opener: re.Match[str]) -> str:
    """Which kind of marker opened an item: one list is written in one style, and a change of style is
    where one list ends and the next begins.

    This is load-bearing rather than tidy. `DETECTOR_BACKENDS.md`'s "Two rules, documented in the file
    and the sidecar README:" is followed by its two bullets and then, with no blank line between, by
    an *ordered* item that belongs to the section after it - so a run that only looked for markers
    where they occur would count three items under a claim of two.
    """
    for name in ("bullet", "ordered", "bold"):
        if opener.group(name):
            return name
    raise AssertionError(f"no marker in {opener.group(0)!r}")


def _item_run(lines: list[str], start: int, stop: int | None = None) -> int:
    """The items of a list beginning at or after line `start` (0-based), or 0 for none.

    The typography these files already use: skip blank lines, then count each paragraph that opens
    with a marker, consuming its wrapped lines. A non-blank line opening with a marker at the item's
    own indent or shallower is a sibling item; a paragraph that opens with no marker ends the run. A
    table, a fenced block, a paragraph of prose or a line of code therefore contributes nothing.

    `stop` bounds the run to a paragraph, which one caller needs: a claim's own bullets are often
    written under it with no blank line between, and an unbounded run from there would walk on into
    the next topic's list and count that as part of this claim.
    """
    index = start
    items = 0
    run_style: str | None = None
    while True:
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines) or (stop is not None and index >= stop):
            break
        opener = ITEM.match(lines[index])
        if not opener:
            break
        style = _style(opener)
        if run_style is not None and style != run_style:
            break
        run_style = style
        indent = len(opener.group("indent"))
        items += 1
        index += 1
        while index < len(lines) and lines[index].strip():
            sibling = ITEM.match(lines[index])
            if sibling and len(sibling.group("indent")) <= indent and _style(sibling) == run_style:
                break
            index += 1
    return items


def _items_under(lines: list[str], claim_line: int) -> int | None:
    """How many items the claim on line `claim_line` (0-based) enumerates, or `None` for none.

    Two places these corpora put the list, and both are tried because the difference is only whether
    a blank line separates it from the claim: under the claim's own last line, as in
    `DETECTOR_BACKENDS.md`'s "Two rules, documented in the file and the sidecar README:" with its two
    bullets starting on the very next line, and after the claim's paragraph, as in the tools'
    docstrings, where the sentence wraps before the items begin. The claim's paragraph is read first
    so a wrapped sentence contributes no items of its own.
    """
    end = claim_line
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    inside = _item_run(lines, claim_line + 1, stop=end + 1)
    if inside:
        return inside
    after = _item_run(lines, end + 1)
    return after or None


def _claims_in(path: str, text: str) -> list[Claim]:
    """Every stated count in one text, with the items under it.

    Pure, and the only walk there is: the corpus arm hands it a file's text and the canary hands it
    synthetic text, so the two cannot disagree about what a claim is.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    claims = []
    for number, line in enumerate(lines):
        for match in CLAIM.finditer(line):
            # The claim's own paragraph, which is what an exemption row is matched against.
            end = number
            while end + 1 < len(lines) and lines[end + 1].strip():
                end += 1
            claims.append(
                Claim(
                    path=path,
                    line=number + 1,
                    stated=NUMBER_WORDS[match.group("word")],
                    headline=match.group(0),
                    snippet=" ".join(" ".join(lines[number : end + 1]).split()),
                    items=_items_under(lines, number),
                )
            )
    return claims


def _claims(path: str) -> list[Claim]:
    return _claims_in(path, _text(path))


def _shape(claim: Claim) -> str:
    """Which judgement this claim's shape is: `list` when its items are countable, else `prose`."""
    return SHAPE_PROSE if claim.items is None else SHAPE_LIST


def _mismatch(claim: Claim) -> str | None:
    """The problem sentence for a claim whose stated count and its items disagree, or `None`.

    Two judgements, kept as separate shapes so a row written for one cannot silence the other: a list
    whose length is not the number (the undercount this guard exists for), and a claim with no list
    under it at all, which is reported as needing a row rather than as a finding - the prose case is
    unverifiable here, and the row is what keeps it from being invisible.
    """
    if claim.items is None:
        return (
            f"{claim.path}:{claim.line}: {claim.headline!r} states {claim.stated}, and the paragraphs "
            "under it open with no item marker (a table, a fence, or prose). Name it in EXEMPT with "
            "the phrase, the place and why it cannot be a list, or shape the sentence as one - an "
            "uncountable claim is a sentence every reader has to count by hand"
        )
    if claim.items != claim.stated:
        return (
            f"{claim.path}:{claim.line}: {claim.headline!r} states {claim.stated} and "
            f"{claim.items} item(s) are enumerated under it"
        )
    return None


def _exempt(claim: Claim, waived: list[Waiver] | None = None) -> str | None:
    """The reason this finding is not a finding, if a row covers it *for this claim's shape*."""
    return EXEMPT.reason(_shape(claim), claim.path, claim.snippet, waived)


def _problems() -> tuple[list[str], list[Waiver], int]:
    """Every finding across the corpus, the rows that were applied, and how many claims were read."""
    found: list[str] = []
    waived: list[Waiver] = []
    read = 0
    for path in _files():
        for claim in _claims(path):
            read += 1
            problem = _mismatch(claim)
            if problem is None:
                continue
            if _exempt(claim, waived):
                continue
            found.append(problem)
    return found, waived, read


def test_the_corpus_is_the_tools_and_the_docs():
    """The globs still find the two corpora, so a scan that reads nothing cannot pass.

    Pinned because this guard's whole boundary is its corpus: a `TOOLS_GLOB` that matched no file
    would leave every claim in the tool docstrings unwatched and still report a green run.
    """
    files = _files()
    tools = [path for path in files if path.startswith("sidecar/tools/")]
    docs = [path for path in files if path.startswith("docs/")]
    assert len(tools) > 15, f"{TOOLS_GLOB} matched {len(tools)} file(s) - the tools moved"
    assert len(docs) > 10, f"{DOCS_GLOB} matched {len(docs)} file(s) - the docs moved"
    assert "sidecar/tools/generate_version.py" in files, "the refusal-count docstring is not read"


def test_every_stated_count_matches_the_items_enumerated_under_it():
    """The arm: a claim's number is the number of items under it, or the claim says why not.

    The failure it was written for is `Four states have no answer here:` over five of them, and its
    two halves read as true one at a time - the sentence stands on its own and the list stands on its
    own - so nothing else in this suite can see it. Both judgements are reported: a list that is
    longer than the number (the undercount), and a claim whose items are not a list at all, which is
    the prose case and which asks for a row rather than being skipped.

    What this scan applied is asserted against the table's rows *per shape*, so a row that is live but
    which no claim of its shape ever reaches - a waiver nobody is using - fails here instead of
    reading as coverage.
    """
    problems, waived, read = _problems()
    assert read >= CLAIMS_READ, (
        f"the scan read {read} claim(s) where {CLAIMS_READ} were measured here: a claim shape that "
        "stopped matching, or a corpus that shrank, leaves the rest of this guard checking nothing"
    )
    for shape in SHAPES:
        rows = {row.phrase for row in EXEMPT.rows if row.shape == shape}
        applied = {waiver.phrase for waiver in waived if waiver.shape == shape}
        assert applied == rows, (
            f"this scan applied {sorted(applied) or 'no'} {shape!r} exemption(s) where the table's "
            f"{shape!r} rows are {sorted(rows)}: a row this scan consults and never reaches is a "
            "waiver nobody is using - drop it, or point it at the sentence that replaced the one it "
            "was written for"
        )
    assert not problems, (
        "stated counts that disagree with the items under them:\n"
        + "\n".join(problems)
        + waived_note(waived)
    )


def test_the_stated_count_exemptions_are_still_live():
    """An exemption is a claim too, and it is the one that could rot.

    Three ends, all but the first of them this shape's own:

    * the phrase is still in every place the row names - `tests/exemptions.py`'s rule, over this
      module's reader,
    * the row names a shape this guard produces, since a row naming another spelling excuses nothing,
    * and the phrase still sits in a claim this guard flags *for that shape*, so a row cannot outlive
      the sentence it was written for and go on covering whatever replaced it.

    The last end is what makes the table worth having rather than a list of sentiments: a row whose
    claim now counts correctly is one nobody needs, and it would sit there waiting to excuse the next
    sentence that happened to carry its words.
    """
    assert_still_present(EXEMPT, _flattened)
    for row in EXEMPT.rows:
        assert row.shape in SHAPES, (
            f"the exemption for {row.phrase!r} names {row.shape!r}, which is not a shape this guard "
            f"produces ({list(SHAPES)}) - a row no scan asks with excuses nothing"
        )
        for place in row.places:
            assert place in _files(), (
                f"the exemption for {row.phrase!r} names {place!r}, which no glob here reads"
            )
            flagged = [
                claim
                for claim in _claims(place)
                if row.phrase in claim.snippet and _shape(claim) == row.shape
            ]
            assert flagged, (
                f"the exemption for {row.phrase!r} in {place} excuses nothing any more: no claim of "
                f"the {row.shape!r} shape there carries that phrase, so the row is one nobody needs - "
                "drop it, or re-point it at the claim that replaced the one it was written for"
            )
            for other in SHAPES:
                if other == row.shape:
                    continue
                assert EXEMPT.reason(other, place, row.phrase) is None, (
                    f"the exemption for {row.phrase!r} silences the {other!r} shape, which it does not "
                    "name - a row written for a sentence with no list must not excuse a list whose "
                    "length disagrees with its number"
                )


def test_the_counter_and_its_two_judgements_still_fire():
    """The canary. What makes this guard worth having is a negative property, and none of it shows up
    in a green run - so the walk is driven over synthetic text and the failure it was written for is
    reproduced here rather than hoped for.

    The last two cases are the ones that keep the guard from being turned off: a table below a claim
    is not a list (reading one as items would flag every documented table in the repo), and the row
    for a prose claim is shape-scoped, so it cannot excuse the same sentence once its items become a
    list that disagrees with it.
    """
    five = (
        "Four things make this not a re-run:\n"
        "\n"
        "**One.** First.\n"
        "\n"
        "**Two.** Second.\n"
        "\n"
        "**Three.** Third.\n"
        "\n"
        "**Four.** Fourth.\n"
        "\n"
        "**Five.** Fifth.\n"
    )
    counted = _claims_in("synthetic.md", five)
    assert len(counted) == 1, f"expected one claim, got {[claim.headline for claim in counted]}"
    assert counted[0].items == 5, f"the walk found {counted[0].items} items under a five-item list"
    problem = _mismatch(counted[0])
    assert problem, "a list of five under `Four things` was not flagged"
    assert "Four things" in problem and "5 item(s)" in problem, problem
    fixed = _claims_in("synthetic.md", five.replace("Four things", "Five things"))
    assert _mismatch(fixed[0]) is None, "the same list under `Five things` is still a finding"

    # The bullet form, wrapped and nested lines included: five bullets with a continuation are five
    # items rather than ten, and a nested marker does not start a second one.
    bullets = (
        "Three rules behind that order:\n"
        "\n"
        " - **First.** A line\n"
        "   wrapped onto another.\n"
        "   - a nested note, not a new item\n"
        " - **Second.** Another.\n"
        " - **Third.** A third.\n"
    )
    bullet_claim = _claims_in("synthetic.md", bullets)[0]
    assert bullet_claim.items == 3, f"the bullet walk counted {bullet_claim.items} items"

    # Prose under a claim: no list, so the judgement is `prose` and the finding asks for a row.
    prose = _claims_in("synthetic.md", "Four states have no answer here, and each is reported.\n")[0]
    assert prose.items is None and _shape(prose) == SHAPE_PROSE
    assert "EXEMPT" in (_mismatch(prose) or ""), "an uncountable claim has to ask for a row"

    # A table, a fence or a wrapped continuation is not a list.
    table = _claims_in("synthetic.md", "Two things are in play:\n\n| | Numbering |\n|---|---|\n")[0]
    assert table.items is None, "a table was counted as items"
    fenced = _claims_in("synthetic.md", "Two things are listed:\n\n```\n- not an item\n- nor this\n```\n")[0]
    assert fenced.items is None, "a fenced block was counted as items"
    wrapped = _claims_in("synthetic.md", "Three things differ from v2, and none is a flag:\nso a reader\nsees a continuation.\n")[0]
    assert wrapped.items is None, "a wrapped sentence was counted as items"

    # The shape scoping, which is what keeps a prose row from covering a real mismatch.
    row = next(entry for entry in EXEMPT.rows if entry.shape == SHAPE_PROSE)
    assert EXEMPT.reason(row.shape, row.places[0], row.phrase), "a row does not answer itself"
    assert EXEMPT.reason(SHAPE_LIST, row.places[0], row.phrase) is None, (
        "a prose row silences the list shape"
    )


# --------------------------------------------------------------------------
# The rows: every claim this scan cannot count, and the one whose number is deliberately not a list's
# length.
#
# A row is `phrase`, `reason`, `shape`, `places` - `tests/exemptions.py`'s shape, shared with the other
# guards that keep such a table. `phrase` carries no number on purpose: the count in an uncountable
# sentence may change without this row being wrong, and a row that broke on a correct edit would be
# re-pointed or deleted rather than read. Each reason says *why* the items are not a list, because the
# next reader's question is what they are supposed to count instead.
# --------------------------------------------------------------------------
EXEMPT = Exemptions(
    Exemption(
        phrase="things the wheel does not ship",
        reason="a heading's items are the numbered list under it, and a paragraph explaining them sits "
        "between the two - the walk reads paragraphs, so it cannot bridge one to reach a list",
        shape=SHAPE_PROSE,
        places=("docs/DETECTOR_BACKENDS.md",),
    ),
    Exemption(
        phrase="rules for that swap",
        reason="the three are clauses of one sentence separated by semicolons, in a paragraph about "
        "the CUDA versions the same command installs",
        shape=SHAPE_PROSE,
        places=("docs/DEVELOPMENT.md",),
    ),
    Exemption(
        phrase="answers, all of them the same failure as a missing field",
        reason="the three are clauses of the next sentence (no file named, file gone, file changed), "
        "each a `Verdict` the same function returns",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/accept_v2.py",),
    ),
    Exemption(
        phrase="outcomes rather than three, and the missing one is the point",
        reason="the two are named in the clause pair that follows, and the sentence's point is the "
        "third outcome the function deliberately does not have",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/accept_v2.py",),
    ),
    Exemption(
        phrase="make a single sweep untrustworthy",
        reason="the two are the rest of that paragraph - the 404 listing route and search's eventual "
        "consistency, with the measured 1,297/1,380/1,383 sweeps - and the loop below is their answer",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/clean_v2.py",),
    ),
    Exemption(
        phrase="findings live here, and the two directions of the first are",
        reason="the three are named across the paragraph (missing, not declared, re-created), and the "
        "first two are contrasted rather than listed",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/clean_v2.py",),
    ),
    Exemption(
        phrase='kinds of "not ready"',
        reason="the two are the `FAIL` and `WARN` rows named in the sentences that follow, which is "
        "what the table below prints and what `--strict` separates",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/readiness.py",),
    ),
    Exemption(
        phrase="states, and the middle one is the one worth having",
        reason="the three are the branches named in the next sentence (a saved profile, `--defaults` "
        "on a machine that has one, and no profile at all)",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/spec_check.py",),
    ),
    Exemption(
        phrase="different things are called `WORKSPACE` in these tools",
        reason="the two are the two imports named in the same comment (`label_classes.WORKSPACE` and "
        "`workspace.WORKSPACE`), in a note about why they are aliased",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/train_model.py",),
    ),
    Exemption(
        phrase="outcomes rather than two",
        reason="the three are the branches of the loop below (passed, failed, and the class the split "
        "never asked about), and the sentence's point is the third",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/train_model.py",),
    ),
    Exemption(
        phrase="files hold the joined pair, and this reads both",
        reason="the two are named in the same sentence (the manifest the dataset tooling wrote, and "
        "the built set's own `merge_report.json`)",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/train_model.py",),
    ),
    Exemption(
        phrase="states have no answer here",
        reason="the five are clauses inside one sentence, and the tool's own refusal order is the "
        "list: each is a `Verdict` the same function returns",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/backfill_provenance.py",),
    ),
    Exemption(
        phrase="of those states are refused rather than printed",
        reason="the five are paragraphs of one docstring, each opening a class of state with a "
        "sentence about it, and the refusal messages themselves restate the count",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/generate_version.py",),
    ),
    Exemption(
        phrase="documents carry numbers that are really copies of code facts",
        reason="the three are the paths of the indented block below, which is a literal block rather "
        "than a list - and `docs_sync.SITES_BY_DOC` is their owner",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/docs_sync.py",),
    ),
    Exemption(
        phrase="sweeps, for the two ways a miss can be an inference setting",
        reason="the two are the `--conf-sweep` and `--iou-sweep` lines of the indented block below, "
        "which is a usage block rather than a list",
        shape=SHAPE_PROSE,
        places=("sidecar/tools/audit_recall.py",),
    ),
    Exemption(
        phrase="Three outcomes, not two",
        reason="the three are clauses of the next sentence (clears the floor, below it, `[SKIP]`) and "
        "the sentence's point is the contrast with the two a reader expects",
        shape=SHAPE_PROSE,
        places=("docs/MODEL_TRAINING.md",),
    ),
    Exemption(
        phrase="different numbers are in play, and they deliberately disagree",
        reason="the two are the rows of the table below, and a table is the shape this scan reads as "
        "uncountable by design",
        shape=SHAPE_PROSE,
        places=("docs/MODEL_TRAINING.md",),
    ),
    Exemption(
        phrase="the counting cannot do, both by design",
        reason="the two are clauses of the same sentence, each bolding its own name mid-sentence",
        shape=SHAPE_PROSE,
        places=("docs/POS_INTEGRATION.md",),
    ),
    Exemption(
        phrase="this print has said that people misread",
        reason="the two are clauses of the same sentence, in a note about one printed line",
        shape=SHAPE_PROSE,
        places=("docs/RUN_SHEET.md",),
    ),
    Exemption(
        phrase="the difference is an instruction rather than a style",
        reason="the three are inline code spans in one sentence (`a number`, `!`, `-`), which a rule "
        "counting them would be counting punctuation",
        shape=SHAPE_PROSE,
        places=("docs/RUN_SHEET.md",),
    ),
    Exemption(
        phrase="about how it is measured, and one about what it means",
        reason="the number is deliberately not the list's length: the sentence splits the three "
        "bullets under it into two about the measurement and one about the reading",
        shape=SHAPE_LIST,
        places=("docs/MODEL_TRAINING.md",),
    ),
)
