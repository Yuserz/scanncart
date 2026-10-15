"""The claim engine's own tests: the rules a table has to satisfy, and the report both ends of it.

Every other module tests a *claim* through the engine. This one tests the engine, for the reason
`test_wording_engine.py` gives about its own: the properties that make a claim worth having are all
negative ones - a table with no rows, a claim listed twice so one of the two readings is never looked
at, a failure that prints a count and not the failing statement, a `--strict` exit that stays zero -
and not one of them shows up in a green run of the claims themselves.

The last two tests are the ones that keep the unification from decaying: no tool may keep a private
answer to "which claims failed", and each surface that decides claims has to be reached for the
engine rather than for a copy of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from claims import Claim, Claims

#: The engine, and the module that owns it - a private copy anywhere else is the drift this exists for.
ENGINE = Path("sidecar/tools/claims.py")

#: The tools that decide claims: they report a run's verdict to a reader, so none of them may build
#: the sentence itself - that is the copy this engine exists to remove.
TOOLS = (
    "sidecar/tools/clamp_probe.py",
    "sidecar/tools/unsure_probe.py",
    "sidecar/tools/spec_check.py",
)

#: The text guards that decide claims: they report a scan's findings, so they go through the engine
#: too - but they are test modules, and *asserting* the documented sentence is what a test is for
#: ("`1 of 1 claims FAIL` is what a failure says"), so the literal scan below leaves them out.
GUARDS = (
    "sidecar/tests/test_docs_claims.py",
    "sidecar/tests/test_stated_counts.py",
)

#: Everything that decides a claim, for the wiring check.
USERS = TOOLS + GUARDS


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _table(**overrides) -> Claims:
    """A table that holds, so a test can break exactly one thing about it."""
    fields = dict(
        name="a test table",
        owner="sidecar/tools/claims.py",
        rows=(
            Claim("the first thing holds", "1 of 1"),
            Claim("the second thing holds", "2 of 2"),
        ),
    )
    fields.update(overrides)
    return Claims(**fields)


# --- the record -------------------------------------------------------------------------------


def test_a_claim_holds_unless_it_is_told_otherwise():
    """The default is the honest one: a claim is written because it was checked."""
    assert Claim("something holds", "1 of 1").holds is True
    assert Claim("something does not", "0 of 1", holds=False).holds is False


def test_a_claim_with_no_reading_or_no_statement_fails_at_construction():
    """The two ways of writing a claim nobody can check, both refused where it is written.

    A blank statement prints a row with nothing to argue with; a blank reading is the shape this
    engine exists to make impossible - a verdict with no number behind it, which is exactly how "the
    rule works" gets said about a rule that was never measured.
    """
    with pytest.raises(ValueError, match="names nothing"):
        Claim("   ", "1 of 1")
    with pytest.raises(ValueError, match="carries no reading"):
        Claim("the rule works", "  ")


# --- the table --------------------------------------------------------------------------------


def test_a_table_with_no_rows_cannot_be_built():
    """The one failure a table cannot see from the inside: with nothing in it, everything passes."""
    with pytest.raises(ValueError, match="verifies nothing"):
        _table(rows=())


def test_the_same_statement_twice_is_refused():
    """Two rows for one claim would count it as two and never look at the second reading."""
    with pytest.raises(ValueError, match="listed twice"):
        _table(
            rows=(
                Claim("the rule works", "1 of 1"),
                Claim("the rule works", "0 of 1", holds=False),
            )
        )


def test_a_table_cannot_fail_more_claims_than_it_decided():
    """`read` is how many were decided, so a table claiming two failures out of one is arithmetic
    that no scan could have produced - and it would print a verdict no reader could reconcile."""
    with pytest.raises(ValueError, match="cannot fail more"):
        _table(read=1)


def test_a_table_that_says_nothing_or_has_no_owner_is_refused():
    """Both are needed before a failure can be written: what was checked, and where the fact lives."""
    with pytest.raises(ValueError, match="no name"):
        _table(name="  ")
    with pytest.raises(ValueError, match="no owner"):
        _table(owner="")
    with pytest.raises(ValueError, match="blank word"):
        _table(kind=" ")


# --- the one verdict --------------------------------------------------------------------------


def test_the_verdict_counts_with_the_tables_own_words():
    """One sentence, and the noun and verbs are the table's - which is what lets two of them stay
    documented output (`all 5 claims hold`, `all 5 PRD targets pass`) while sharing this engine."""
    assert _table().verdict() == "all 2 claims hold"
    assert _table(kind="PRD targets", holds_verb="pass").verdict() == "all 2 PRD targets pass"
    prd = _table(
        kind="PRD targets",
        holds_verb="pass",
        fails_verb="fail",
        rows=(
            Claim("a speed", "26.8 ms", holds=False),
            Claim("a latency", "150 ms", holds=False),
            Claim("a recall", "0.94"),
        ),
    )
    assert prd.verdict() == "2 of 3 PRD targets fail: a speed; a latency"


def test_the_verdict_names_the_failures_rather_than_only_counting_them():
    """A count alone sends the reader back into the report to find out which claim moved."""
    text = _table(
        rows=(
            Claim("the first thing holds", "1 of 1"),
            Claim("the second thing does not", "0 of 2", holds=False),
        )
    ).verdict()
    assert text == "1 of 2 claims FAIL: the second thing does not"
    assert "the second thing does not" in text and "the first thing holds" not in text


def test_a_scan_that_carries_only_findings_still_reports_the_share_it_decided():
    """The guards' direction: `rows` are the findings, `read` is how many claims the scan decided -
    so their failure reads in the same words a measured run's does."""
    table = _table(read=36, rows=(Claim("'Four things' in spec_check.py", "states 4 over 5 items",
                                      holds=False),))
    assert table.decided == 36
    assert table.verdict() == "1 of 36 claims FAIL: 'Four things' in spec_check.py"


# --- the report -------------------------------------------------------------------------------


def test_the_block_pads_every_row_to_one_width_and_ends_in_the_verdict():
    """The block is what a tool prints, and the single pad is the point of computing it here: the two
    probes used to hand-tune 52 and 56 for the same job."""
    block = _table(rows=(
        Claim("short", "1 of 1"),
        Claim("a considerably longer statement", "2 of 2"),
    )).block()
    assert block[0] == "" and block[1] == "claims"
    # One pad for the table - the longest statement plus two - so a reader's eye can run down the
    # readings instead of hunting for where each one starts.
    assert block[2:4] == [
        "  [PASS] short                             1 of 1",
        "  [PASS] a considerably longer statement   2 of 2",
    ]
    assert block[4] == "" and block[5] == "  all 2 claims hold"


def test_a_table_may_render_its_own_row_when_its_reading_has_columns():
    """`spec_check` prints a measured value, a unit and the target it was compared against, and
    `docs/MODEL_TRAINING.md` pastes that row - so the layout stays the table's business."""
    table = _table(row=lambda claim: f"<{claim.statement}|{claim.reading}>")
    assert table.block()[2] == "<the first thing holds|1 of 1>"


def test_a_failure_prints_the_failing_rows_and_the_remedy():
    """What a test raises: the verdict, the reading that came out wrong, and what to do next."""
    table = _table(
        remedy="Update the claim in the same edit as the number.",
        rows=(
            Claim("the first thing holds", "1 of 1"),
            Claim("the second thing does not", "0 of 2", holds=False),
        ),
    )
    text = table.failure()
    assert text.startswith("a test table - 1 of 2 claims FAIL: the second thing does not")
    assert "[FAIL] the second thing does not" in text and "0 of 2" in text
    assert "the first thing holds" not in text
    assert text.endswith("Update the claim in the same edit as the number.")


def test_a_holding_table_says_nothing_and_raises_nothing():
    """The other end: silence is what a green run looks like, and a failure that fired on a holding
    table would make every caller's own test fail for the engine's mistake."""
    table = _table()
    assert table.holds is True
    assert table.assert_holds() is None
    assert table.decided == 2 and table.failed == ()


def test_the_strict_exit_is_the_tables_and_not_the_callers():
    """`1 if args.strict and strict_failed(...) else 0` was spelled out in three tools; the flag is
    the caller's and the answer is the table's, so only the flag is left to them."""
    assert _table().exit_code(False) == 0
    assert _table().exit_code(True) == 0
    failing = _table(rows=(Claim("a thing that does not hold", "0 of 2", holds=False),))
    assert failing.exit_code(False) == 0
    assert failing.exit_code(True) == 1


def test_a_canary_proves_each_half_can_fire():
    """The engine's own both-ends check, on the smallest table that has an end to break.

    Flipping one claim has to move every half of the report at once - the sentence, the failing list,
    the block and the exit - because a report whose halves disagree is worse than none: a reader
    would see `all 2 claims hold` above a FAIL row, or a non-zero exit from a run that printed a
    passing verdict.
    """
    holding = _table()
    broken = _table(rows=(
        Claim("the first thing holds", "1 of 1"),
        Claim("the second thing does not", "0 of 2", holds=False),
    ))
    assert holding.verdict() != broken.verdict()
    assert [row.holds for row in broken.rows] == [True, False]
    assert "[FAIL]" in broken.block()[3] and "[FAIL]" not in holding.block()[2]
    assert broken.exit_code(True) == 1 and holding.exit_code(True) == 0
    with pytest.raises(AssertionError) as caught:
        broken.assert_holds()
    assert caught.value.args[0] == broken.failure()


# --- the engine is the only one ---------------------------------------------------------------


def _defined_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _docstrings(tree: ast.Module) -> set[int]:
    """The nodes that are docstrings: the first statement of the module, a class or a function.

    They are `ast.Constant` strings like any other literal, and prose is exactly what this scan must
    not read: `spec_check.py`'s own docstring quotes the sentence it hands to the engine, and a
    guard that called that a second copy would be turned off within a week. So the docstrings are
    collected here and the literals that *do* something are what is left.
    """
    nodes: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            nodes.add(id(first.value))
    return nodes


def _string_literals(tree: ast.Module) -> list[str]:
    """Every string constant that is not a docstring - the ones that reach a reader as output."""
    prose = _docstrings(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in prose
    ]


def test_no_tool_keeps_its_own_answer_to_which_claims_failed():
    """The unification, held in place: the four copies this engine replaced cannot come back.

    Read with `ast` rather than as text, because these words also appear in prose - a docstring
    explaining the verdict, a comment about the old spelling - and a scan that flagged a sentence
    about the idea would be turned off within a week. So it is the shapes that are refused: another
    module defining `verdict` or `strict_failed`, or carrying the verdict's own sentence as a
    literal.

    Only the tools are read, and the guards are not, for the reason their own rows give: a tool
    *prints* the verdict, so a copy of it is a second implementation; a test *asserts* it, and
    `"1 of 1 claims FAIL" in report` is the pin that keeps the wording honest. The wiring check
    below covers both.
    """
    root = _repo_root()
    offenders: list[str] = []
    for relative in TOOLS:
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in sorted(_defined_functions(tree) & {"verdict", "strict_failed"}):
            offenders.append(f"{relative}: defines its own `{name}`")
        for literal in _string_literals(tree):
            if "claims hold" in literal or "claims FAIL" in literal:
                offenders.append(f"{relative}: spells the verdict itself ({literal!r})")
    assert not offenders, (
        "a claim report was copied back out of the engine: "
        "`sidecar/tools/claims.py` owns the verdict, the report, the strict exit and the failure "
        "text, and a second copy is how the five surfaces start reporting the same claim "
        "differently - route it through `Claims` instead:\n  " + "\n  ".join(offenders)
    )


def test_every_surface_that_decides_claims_is_wired_to_the_engine():
    """The other half of the same rule, and the one a green suite cannot see: a module that stopped
    importing the engine would still pass everything it has, because it would be reporting through
    its own copy - or through nothing.
    """
    root = _repo_root()
    wired, missing = [], []
    for relative in USERS:
        text = (root / relative).read_text(encoding="utf-8")
        (wired if "from claims import" in text or "import claims" in text else missing).append(relative)
    assert not missing, (
        "these decide claims without the engine: "
        f"{missing} - a surface that reports its own failures is the drift `claims.py` exists to "
        "remove, and the whole file is the argument for why"
    )
    assert len(wired) == len(USERS)

    engine = ast.parse((root / ENGINE).read_text(encoding="utf-8"))
    exported = {node.targets[0].id for node in engine.body if isinstance(node, ast.Assign)}
    assert exported == {"__all__"}, f"{ENGINE} exports something other than `__all__`: {exported}"
