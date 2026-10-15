"""Drift guard: the probes' claim tables, pinned where their measurement cannot run.

`clamp_probe.py`, `unsure_probe.py` and `spec_check.py` decide claims about *data* - that the clamp
rule catches the phantom shape at all, that the unsure rule's ceiling sits in a gap no real detection
occupies, that the PRD's five targets still hold for a trained weight. Answering them needs an
installed weight, the staged frames and (for one claim) a camera, which is why the gates that do it
are outside `make test` (`make verify-clamp`, `make verify-unsure`, and the run sheet's own
`spec_check.py` invocation).

The tables themselves are *not* what needs the hardware. `checks()` and `spec_targets()` are ordinary
functions over the measured objects, and until this guard existed nothing on a bare checkout read the
sentences they declare. So a claim could be deleted, reworded or quietly reordered and the suite
would stay green: the engine's rules are exercised by `test_claims_engine.py`, each tool's own tests
pin its verdict *word* (`all 5 claims hold`) and its fixtures, and the hardware gate is the only thing
that would have printed the missing row - which is a run nobody does every commit. The sentences are
the artifact and they had no owner. This guard is that owner, and it is the same arrangement
`test_cost_figures.py` made for the rules' measured price.

**Read as text, with `ast`, never imported.** Importing the probes would drag numpy, torch and cv2
into whatever job collects this file, and none of them is needed to see what a table declares: the
statements are literals in the builder, and what this guard pins is the declaration rather than a
measurement. Reading the source also lets the extractor be canaried over synthetic text, which is the
only way to prove it would notice a deletion at all.

**Every sentence, in order, exactly.** The pin below is an equality and not a subset, so all four
edits fail here: a claim deleted, one added, one reworded, and the order changed (the report prints
rows in this order, so a reorder is a change to every run's output). Each declared claim also has to
still be a *literal* with a reading beside it - a statement assembled from a variable cannot be
pinned, and the engine's own rule that a claim has a reading is only enforced when the tool runs.

**The refusal row is not one of the counted claims.** Both probes open their builder with the one row
their ladder-miss branch produces *instead of* the rest (`the shipped tolerance is on the ladder`),
which is why the tables below carry it and why each probe's own verdict counts the tail: five claims
for `clamp_probe`, nine for `unsure_probe`, five PRD targets for `spec_check` - whose sentence the run
sheet quotes and this guard re-renders from the tool's own kind and verbs.

What this cannot check is what the claims are *worth*: whether the rule still catches the shape, or
whether the cost it measured is still `app/acceptance.py`'s. That is the hardware gates' job, and they
fail a run whose fresh measurement disagrees. This is what runs on a bare checkout.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

#: The claim engine, which is how this guard's own failures are reported: `Claims` is also what
#: re-renders each tool's verdict from the words its declaration carries, so the sentence a run
#: prints can be checked on a checkout where the tool cannot be run.
from claims import Claim, Claims, findings

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class Probe:
    """One tool's claim table, as this guard pins it.

    `claims` is every sentence the tool can report, in the order its builder declares them - the
    refusal first for the two probes, which is why `refusals` says how many of them the tool's own
    verdict does *not* count. `verdict` is the sentence it prints when the rest hold, in the words the
    repo documents it in (`docs/RUN_SHEET.md` for `spec_check`; the tools' own tests for the probes).
    """

    #: The tool, repo-relative, spelled the way a failure quotes it.
    path: str
    #: The function that declares the table.
    function: str
    #: The record each claim is declared with, inside that function.
    record: str
    #: Every sentence, in declaration order.
    claims: tuple[str, ...]
    #: How many sentences *lead* the table as the refusal its failure branch produces instead of the
    #: claims below them.
    refusals: int
    #: The sentence the tool prints when the counted claims hold.
    verdict: str
    #: The module the claims are about, for the failure messages - the same owner the tool's own
    #: table carries, because a failure here is about the tool's declaration rather than this pin.
    owner: str
    #: The function that builds the `Claims` table, when that is not the one declaring the claims:
    #: `spec_check.py` declares its five targets in `spec_targets()` and names its verdict words in
    #: `checks()`, so the two have to be read apart rather than assumed to be the same function.
    #: Empty means `function`.
    table_function: str = ""

    @property
    def words_from(self) -> str:
        """The function the verdict's own words are declared in."""
        return self.table_function or self.function

    @property
    def counted(self) -> tuple[str, ...]:
        """The sentences the tool's own verdict counts - its claims, with the refusal left out."""
        return self.claims[self.refusals:]

    def __post_init__(self) -> None:
        # A pin that says nothing about anything, refused where it is written: `test_cost_figures`
        # and the wording engine hold their tables to the same rule.
        if not self.claims:
            raise ValueError(f"{self.path}: a probe pinned with no claims pins nothing")
        if not 0 <= self.refusals < len(self.claims):
            raise ValueError(
                f"{self.path}: {self.refusals} refusal row(s) out of {len(self.claims)} pinned "
                "claim(s) - the refusal is a row of the same table and cannot be all of it"
            )
        if f"all {len(self.counted)}" not in self.verdict:
            raise ValueError(
                f"{self.path}: the pinned verdict {self.verdict!r} does not count the "
                f"{len(self.counted)} claim(s) this pin lists below the refusal"
            )


TOOLS: tuple[Probe, ...] = (
    Probe(
        path="sidecar/tools/clamp_probe.py",
        function="checks",
        record="Claim",
        refusals=1,
        verdict="all 5 claims hold",
        owner="sidecar/app/pipeline.py (CLAMPED_EDGE_TOLERANCE)",
        claims=(
            "the shipped tolerance is on the ladder",
            "no measured real detection is inside the rule",
            "the rule catches the phantom shape at all",
            "the Pipeline drops detections for it",
            "fewer empty-counter frames show a phantom with it on",
            "the Pipeline loses no real product detection",
        ),
    ),
    Probe(
        path="sidecar/tools/unsure_probe.py",
        function="checks",
        record="Claim",
        refusals=1,
        verdict="all 9 claims hold",
        owner="sidecar/app/acceptance.py (is_unsure_phantom)",
        claims=(
            "the shipped ceiling is on the ladder",
            "the ceiling sits in a gap no real detection occupies",
            "it costs fewer real detections than the clamp rule does",
            "it catches the stored empty counter's own family",
            "no shaped detection under the ceiling escaped in the live window",
            "the live window exercised the band family",
            "its price on the real product control has not moved",
            "no detection under the ceiling survives outside both shapes",
            "the rule is on in the profile this measured",
            "the cost it measured is still the one app/acceptance.py records",
        ),
    ),
    Probe(
        path="sidecar/tools/spec_check.py",
        function="spec_targets",
        record="Target",
        refusals=0,
        verdict="all 5 PRD targets pass",
        owner="sidecar/tools/spec_check.py",
        table_function="checks",
        claims=(
            "isolated infer fps",
            "in-app pipeline fps",
            "in-app mean latency",
            "in-app p95 latency",
            "instance recall at app conf",
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class Declaration:
    """One claim as the source declares it: where, the sentence, and whether it is checkable here."""

    #: The line the declaration starts on, so a failure can point at it.
    line: int
    #: The literal statement, or `None` when it stopped being a literal this guard can pin.
    statement: str | None
    #: Whether a reading is passed beside the statement - the engine refuses one without.
    has_reading: bool
    #: The `kind`/`holds_verb`/`fails_verb` the table renders its verdict with, carried on the call
    #: that builds the table so the words can be checked without running the tool.
    kind: str = "claims"
    holds_verb: str = "hold"
    fails_verb: str = "FAIL"


def _keyword(call: ast.Call, name: str) -> str | None:
    """The literal value of one keyword argument, or `None` when it is absent or not a literal."""
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    return None


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    """One top-level function by name, or `None` - a renamed builder is reported by the caller."""
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def declared(
    source: str, *, function: str, record: str, words_function: str = ""
) -> tuple[Declaration, ...]:
    """Every claim one builder declares, in source order, read out of the text.

    Text rather than an import, so the extractor is drivable over a synthetic module in the canary
    below - and the declarations are what this guard pins, not the measurements they will be handed.
    A builder this cannot find, or a claim whose statement is no longer a literal, is reported as a
    declaration rather than skipped: a table that stopped being readable here would otherwise read as
    one with nothing wrong.

    `words_function` is where the verdict's own noun and verbs are declared, when that is a different
    function from the one declaring the claims (`spec_check.py`'s five targets are declared in
    `spec_targets()` and counted in `checks()`).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as broken:
        # A probe whose source does not parse is a broken tool rather than a drifted table, and the
        # failure should say that instead of pointing a traceback at this guard's own `ast.parse`.
        raise AssertionError(
            f"the tool this guard reads does not parse any more ({broken.msg} at line "
            f"{broken.lineno}), so its claim table cannot be read at all"
        ) from broken
    builder = _function(tree, function)
    if builder is None:
        return ()
    calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == record
    ]
    calls.sort(key=lambda node: node.lineno)
    # The words come off the last table built, which is the branch that carries the counted claims -
    # the refusal branch returns early with a one-row table of its own.
    words_fn = _function(tree, words_function or function)
    tables = (
        [
            node
            for node in ast.walk(words_fn)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Claims"
        ]
        if words_fn is not None
        else []
    )
    tables.sort(key=lambda node: node.lineno)
    words = tables[-1] if tables else None
    return tuple(
        Declaration(
            line=call.lineno,
            statement=(
                call.args[0].value
                if call.args and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
                else None
            ),
            has_reading=len(call.args) >= 2
            or any(keyword.arg == "reading" for keyword in call.keywords),
            kind=(_keyword(words, "kind") or "claims") if words else "claims",
            holds_verb=(_keyword(words, "holds_verb") or "hold") if words else "hold",
            fails_verb=(_keyword(words, "fails_verb") or "FAIL") if words else "FAIL",
        )
        for call in calls
    )


def _source(probe: Probe) -> str:
    path = REPO_ROOT / probe.path
    assert path.is_file(), (
        f"{probe.path} is gone, so the claims this guard pins have no owner left to be read from"
    )
    return path.read_text(encoding="utf-8")


def _drift(probe: Probe, declared_: tuple[Declaration, ...]) -> list[tuple[str, str]]:
    """`(where, why)` for every way this probe's table stopped saying what this pin says it says."""
    problems: list[tuple[str, str]] = []
    for declaration in declared_:
        where = f"{probe.path}:{declaration.line}"
        if declaration.statement is None:
            problems.append(
                (
                    where,
                    f"this claim's statement is not a literal any more, so nothing here can pin it - "
                    f"keep the sentence written out in the `{probe.record}(...)` call, or move "
                    "whatever it is built from into the reading",
                )
            )
        if not declaration.has_reading:
            problems.append(
                (
                    where,
                    "this claim carries no reading, so nothing decides it - the engine refuses one "
                    "at construction, but only when the tool runs, which needs the hardware",
                )
            )
    sentences = [
        declaration.statement for declaration in declared_ if declaration.statement is not None
    ]
    for sentence in probe.claims:
        if sentence not in sentences:
            problems.append(
                (
                    f"{probe.path}: the claim {sentence!r}",
                    "is gone from the table this guard pins - a deleted claim is invisible on a bare "
                    "checkout, which is why it is written down here, and a *reworded* one reads as "
                    "gone because the sentence is what a run prints",
                )
            )
    for sentence in sentences:
        if sentence not in probe.claims:
            problems.append(
                (
                    f"{probe.path}: the claim {sentence!r}",
                    "is declared and not pinned here - add it to this guard's table in the same "
                    "edit, with the sentence a run would print",
                )
            )
    if not problems and sentences != list(probe.claims):
        problems.append(
            (
                f"{probe.path}: the claims are declared in a different order",
                "the report prints rows in declaration order, so a reorder changes every run's "
                "output - reorder this pin in the same edit, or restore the order",
            )
        )
    return problems


def test_each_probe_declares_exactly_the_claims_this_guard_pins():
    """The arm: every sentence, in order, and nothing else.

    The pin is an equality, so one deleted claim, one added, one reworded and one reordered all fail
    here - and the reworded case is the one that matters most, because the sentence *is* the claim a
    reader of the run trusts, and a tool whose wording drifted is a tool whose prose no longer
    matches the number beside it. Both directions are reported, because the repairs differ: a claim
    that left the tool is a measurement nobody makes any more, and a claim that arrived without a row
    is a sentence that nothing on a bare checkout has ever read.
    """
    problems: list[tuple[str, str]] = []
    read = 0
    for probe in TOOLS:
        declared_ = declared(
            _source(probe),
            function=probe.function,
            record=probe.record,
            words_function=probe.words_from,
        )
        assert declared_, (
            f"{probe.path} has no `{probe.record}(...)` declarations in `{probe.function}` any more "
            "- did the builder or the record get renamed? A table this guard cannot find would "
            "otherwise read as one with nothing wrong"
        )
        read += 1
        problems.extend(_drift(probe, declared_))
    if problems:
        findings(
            name="a probe's claim table no longer says what this guard pins",
            owner="each tool's own `checks()`/`spec_targets()`, and this file",
            rows=problems,
            read=read,
            remedy=(
                "Fix the tool or this pin in the same edit. The claims are what the hardware gates "
                "print and fail on, so a sentence that moved is a run whose output nobody has read."
            ),
        ).assert_holds()


def test_each_tools_verdict_still_renders_the_sentence_the_repo_documents():
    """The count and the words, re-rendered from the tool's own declaration.

    The engine builds the sentence from the table's noun and verbs, so this walks the same path with
    the words read off the tool's `Claims(...)` call and the rows it counts - which is what catches a
    claim added to a table whose verdict is still asserted somewhere as `all 5 claims hold`. It is
    also the one place the run sheet's quoted sentence (`all 5 PRD targets pass`) is held to the tool
    that prints it, on a checkout where `spec_check.py` cannot run.
    """
    problems: list[tuple[str, str]] = []
    for probe in TOOLS:
        declared_ = declared(
            _source(probe),
            function=probe.function,
            record=probe.record,
            words_function=probe.words_from,
        )
        words = declared_[-1] if declared_ else None
        assert words is not None, f"{probe.path}: no declaration to read a verdict off"
        rendered = Claims(
            name=f"{probe.path}'s table",
            owner=probe.owner,
            kind=words.kind,
            holds_verb=words.holds_verb,
            fails_verb=words.fails_verb,
            rows=tuple(
                Claim(sentence, "a reading this guard does not measure")
                for sentence in probe.counted
            ),
        ).verdict()
        if rendered != probe.verdict:
            problems.append(
                (
                    f"{probe.path}: its table renders {rendered!r}",
                    f"this guard pins the documented {probe.verdict!r} - a claim added or removed "
                    "changes the count in a sentence the tools' own tests and the run sheet quote",
                )
            )
    run_sheet = (REPO_ROOT / "docs/RUN_SHEET.md").read_text(encoding="utf-8")
    for probe in TOOLS:
        if probe.path.endswith("spec_check.py"):
            if probe.verdict not in run_sheet:
                problems.append(
                    (
                        "docs/RUN_SHEET.md",
                        f"no longer quotes {probe.verdict!r}, the sentence spec_check.py prints "
                        "when its targets hold - the doc and the table are one claim about five "
                        "targets",
                    )
                )
    if problems:
        findings(
            name="a probe's verdict no longer matches the sentence the repo documents",
            owner="each tool's `Claims(...)` words, and the docs that quote them",
            rows=problems,
            read=len(TOOLS),
            remedy="Keep the count, the noun and the verbs together: they are the tool's output.",
        ).assert_holds()


def test_the_gates_these_tables_belong_to_are_still_outside_ci():
    """The premise, pinned - because this guard's whole reason is that the measurement needs hardware.

    If a probe's gate ever became a CI step, this file would be a pin beside a check that already
    runs, and the honest edit would be to fold it in rather than keep two answers. So the gates are
    asserted where they live: the Makefile's own list of targets that need local data, and the
    suite's `testpaths`, which is what keeps a `tools/*.py` module from being collected as a test.
    """
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("verify-clamp", "verify-unsure"):
        assert f"{target}:" in makefile, (
            f"the Makefile no longer defines `{target}`, the gate that re-measures one of these "
            "tables - if that measurement moved into `make test`, this guard is pinning a claim CI "
            "already checks and should be folded into whatever runs it"
        )
    ini = (REPO_ROOT / "sidecar/pytest.ini").read_text(encoding="utf-8")
    assert "testpaths = tests" in ini, (
        "the suite's testpaths changed: if it now collects `sidecar/tools/`, the probes' own "
        "modules are being imported into the run this guard exists to stand in for"
    )


# --- the canary: the extractor has to be able to see a deletion ---------------------------------

#: A synthetic builder with the same shape as a probe's: one refusal row and two counted claims, the
#: table's words on the `Claims(...)` call. Small enough to mutate by hand, which is the point -
#: the real tools cannot be broken in the suite.
FIXTURE = '''
def checks(measured):
    if measured is None:
        return Claims(name="n", owner="o", rows=(
            Claim("the shipped thing is on the ladder", "nothing to justify it", holds=False),
        ))
    return Claims(
        name="n",
        owner="o",
        kind="claims",
        holds_verb="hold",
        fails_verb="FAIL",
        rows=(
            Claim("the first claim", "1 of 1", measured.first),
            Claim("the second claim", "2 of 2", measured.second),
        ),
    )
'''

FIXTURE_PIN = Probe(
    path="sidecar/tests/test_probe_claims.py",
    function="checks",
    record="Claim",
    refusals=1,
    verdict="all 2 claims hold",
    owner="a fixture",
    claims=(
        "the shipped thing is on the ladder",
        "the first claim",
        "the second claim",
    ),
)


def test_the_canary_proves_the_extractor_sees_a_deletion_a_reword_and_a_catapulted_statement():
    """Both ends of this guard, on a fixture rather than on a probe.

    A pin that never fires is a pin nobody knows the teeth of, and the three edits it exists for are
    the ones that leave a run looking healthy: a claim deleted (the row is simply absent), a claim
    reworded (the sentence is what a run prints, so a rewrite reads as a deletion plus an arrival),
    and a claim whose statement stopped being a literal (nothing here can compare it, which has to be
    reported rather than skipped). The order is checked here too, since it is the one failure that
    keeps every sentence present.
    """
    clean = declared(FIXTURE, function="checks", record="Claim")
    assert [declaration.statement for declaration in clean] == list(FIXTURE_PIN.claims)
    assert _drift(FIXTURE_PIN, clean) == [], "the fixture's own table is reported as drifted"
    assert (clean[-1].kind, clean[-1].holds_verb, clean[-1].fails_verb) == ("claims", "hold", "FAIL")

    deleted = declared(
        FIXTURE.replace('            Claim("the second claim", "2 of 2", measured.second),\n', ""),
        function="checks",
        record="Claim",
    )
    gone = _drift(FIXTURE_PIN, deleted)
    assert gone, "a deleted claim was not reported"
    assert any("'the second claim'" in where and "gone from the table" in why for where, why in gone)

    reworded = declared(
        FIXTURE.replace("the second claim", "the second claim, reworded"),
        function="checks",
        record="Claim",
    )
    changed = _drift(FIXTURE_PIN, reworded)
    assert changed, "a reworded claim was not reported"

    reordered = declared(
        FIXTURE.replace('the first claim", "1 of 1', 'the second claim", "1 of 1')
        .replace('the second claim", "2 of 2', 'the first claim", "2 of 2'),
        function="checks",
        record="Claim",
    )
    assert _drift(FIXTURE_PIN, reordered), "a reordered table was not reported"

    built = declared(
        FIXTURE.replace('Claim("the first claim", "1 of 1", measured.first)',
                        'Claim(measured.first_name, "1 of 1", measured.first)'),
        function="checks",
        record="Claim",
    )
    unpinnable = _drift(FIXTURE_PIN, built)
    assert any("not a literal" in why for _, why in unpinnable), (
        "a statement that stopped being a literal was skipped instead of reported"
    )

    # And the engine's own half, on the fixture's table: the words it declares render the pinned
    # sentence, so the check above is about the tools and not about a re-implementation.
    words = clean[-1]
    assert (
        Claims(
            name="fixture",
            owner="a fixture",
            kind=words.kind,
            holds_verb=words.holds_verb,
            fails_verb=words.fails_verb,
            rows=tuple(Claim(sentence, "r") for sentence in FIXTURE_PIN.counted),
        ).verdict()
        == FIXTURE_PIN.verdict
    )
