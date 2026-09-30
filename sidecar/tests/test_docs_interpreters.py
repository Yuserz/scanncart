"""A documented command that runs a *bare* `python`, which cannot import this app at all.

The docs are how this project is run, and a fenced block is meant to be pasted. What a pasted line can
get wrong and no other guard can see: **a bare `python`**. The uv path these docs lead with activates
nothing, so a bare one is the *system* interpreter - which cannot import this app's dependencies at
all - and the local inference server is worse still, since it lives in its own venv precisely because
the sidecar's cannot import it. A reader who pastes `python run.py` gets a `ModuleNotFoundError` and
no hint that the command was wrong rather than the code; an agent that follows one gets the same, and
the first fix that suggests itself (pip install into the system interpreter) is the wrong one.

So: a command position inside a shell fence must name an interpreter by *path*
(`.venv/Scripts/python.exe`, `.venv-inference/bin/python`, `sidecar/.venv/bin/python`) or go through
something that does (`make`, a script with a shebang). Two things are deliberately outside that:

* **Creating the venv.** `python -m venv .venv` is the one command that legitimately needs an
  interpreter from outside the venv it is making, so it is a rule (`VENV_CREATION`) rather than an
  exemption - it will appear in the setup section of any new doc, and a reader following it is not
  being misled.
* **Prose.** This reads fences only, and the reason is in the subject matter: these docs have to be
  able to *talk about* the command they forbid - `CLAUDE.md` says a bare `python` is the system
  interpreter, this module says `python -m venv .venv` is the one legal case - so a guard that also
  read inline code would fail on the sentences explaining the rule. An inline mention is as likely to
  be a quotation as an instruction, too.

It also does not check *which* venv, or that the shell a block assumes is the one a reader has: those
are judgements the fences in this repo make explicitly and visibly (a `# Windows` / `# Linux/macOS`
pair, or §7a's `uv` setup), and a guard that guessed at them would be the kind that gets turned off.

The reading - fences, logical lines, commands, and whether a line is an invocation at all - is in
`tests/docs_fences.py`, shared with the path and option rules. This module owns the interpreter rule
and its canaries: that every instruction doc is scanned, that the corpus's own `python -m venv` lines
are still found, and that neither the positive nor the negative control has gone quiet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from tests.docs_fences import (
    INSTRUCTION_DOCS,
    REPO_ROOT,
    _commands,
    _corpus,
    _is_invocation,
    _logical,
    blocks,
)

pytestmark = pytest.mark.docs

#: The interpreters a reader could reach from `PATH`: the system one on either platform, the `py`
#: launcher, and a versioned `python3.12` - with or without `.exe`. A *path* to one is never matched,
#: because this is anchored at the start of the command: `.venv\Scripts\python.exe` is a path, and
#: the whole point of this rule is that spelling the path is what makes the command work.
BARE_INTERPRETER = re.compile(r"^(?:python(?:3(?:\.\d+)?)?|py)(?:\.exe)?(?:\s|$)")
#: The one legal use of a bare interpreter: making the venv it is about to be replaced by.
VENV_CREATION = re.compile(r"^(?:python(?:3(?:\.\d+)?)?|py)(?:\.exe)?\s+-m\s+venv(?:\s|$)")

# `(doc, phrase, reason)`, the same shape `test_docs_claims` uses, and live for the same reason: an
# exemption is a sentence somebody has to read, and `test_the_exemptions_are_still_live` fails when
# the line it covers is gone, so one cannot outlive its reason and start covering whatever replaced
# it. Empty today on purpose - the venv rule above is a rule, not a doc's excuse - and kept so the
# first legitimate one costs a reason instead of a looser pattern.
EXEMPT: tuple[tuple[str, str, str], ...] = ()

#: The corpus's own bare-interpreter usage, which is exactly the `python -m venv .venv` lines the
#: rule above legalises. Pinned so the scan is known to reach *inside* fences on real docs rather than
#: only on the synthetic ones below: a fence reader that silently stopped entering blocks would find
#: no violations and pass, which is the failure this whole guard exists to prevent.
EXEMPT_EXAMPLES = ("README.md", "sidecar/README.md", "docs/DEVELOPMENT.md")


@dataclass(frozen=True)
class Invocation:
    """One bare interpreter found in a documented shell block."""

    doc: str
    line: int
    command: str
    fence: str
    #: Legal by `VENV_CREATION`: the reader is not misled, they are making the venv.
    creating_venv: bool

    @property
    def exempt(self) -> bool:
        return self.creating_venv or self._listed_exemption() is not None

    def _listed_exemption(self) -> str | None:
        for doc, phrase, reason in EXEMPT:
            if doc == self.doc and phrase in self.command:
                return reason
        return None


def scan(text: str, doc: str = "") -> tuple[list[Invocation], int]:
    """Every bare interpreter in `text`'s shell blocks, and how many such blocks there were.

    The fence count is returned rather than derived by a caller, because it is what proves the
    walker entered a block: a file with no shell fences scanned and a file whose blocks were never
    opened look identical from the list alone.
    """
    found: list[Invocation] = []
    shell = blocks(text)
    for block in shell:
        for number, line in _logical(block.lines):
            for command in _commands(line):
                if not _is_invocation(command):
                    continue
                if BARE_INTERPRETER.match(command):
                    found.append(
                        Invocation(
                            doc=doc,
                            line=number,
                            command=command,
                            fence=block.info,
                            creating_venv=VENV_CREATION.match(command) is not None,
                        )
                    )
    return found, len(shell)


def scan_corpus() -> tuple[dict[str, list[Invocation]], dict[str, int]]:
    """`({doc: invocations}, {doc: shell fences})` over the whole corpus."""
    hits: dict[str, list[Invocation]] = {}
    fences: dict[str, int] = {}
    for path in _corpus():
        doc = path.relative_to(REPO_ROOT).as_posix()
        found, blocks_read = scan(path.read_text(encoding="utf-8"), doc)
        if found:
            hits[doc] = found
        fences[doc] = blocks_read
    return hits, fences


def test_no_documented_command_runs_a_bare_python():
    """The rule this guard exists for, over every instruction doc in the repo."""
    hits, _fences = scan_corpus()
    violations = [
        f"{doc}:{inv.line}: {inv.command}"
        for doc, found in hits.items()
        for inv in found
        if not inv.exempt
    ]
    assert not violations, (
        "a documented shell block tells the reader to run a bare `python`, which is not the "
        "interpreter these commands need - the uv path the setup sections lead with activates "
        "nothing, so a bare one is the system interpreter and cannot import the app's dependencies "
        "(the local inference server is worse: it lives in its own `.venv-inference`). Name the "
        "venv's interpreter the way the rest of the docs do:\n  " + "\n  ".join(violations)
    )


def test_the_guard_catches_every_shape_that_was_wrong():
    """The positive control: the shapes the docs actually had, each of which must fail.

    Written from the two rounds that fixed them - a plain line, a `cd` chained onto one, a platform
    label in front of one, and a prompt - so a change to the reader that stopped seeing one of them
    fails here rather than in a reader's terminal.
    """
    text = "\n".join(
        [
            "```bash",
            "python run.py",
            "cd sidecar && python -m pytest -v",
            "# Windows: python -m pytest tests/test_pipeline.py -v",
            "$ python -m pytest",
            "uv run --no-sync python -m pytest",  # not a command position: `uv` is the command
            "```",
        ]
    )
    found, fences = scan(text)

    assert fences == 1
    assert [inv.command for inv in found] == [
        "python run.py",
        "python -m pytest -v",
        "python -m pytest tests/test_pipeline.py -v",
        "python -m pytest",
    ]
    assert not any(inv.exempt for inv in found), "none of these is a legal bare interpreter"


def test_it_does_not_cry_wolf_on_the_commands_that_are_right():
    """The negative control, and the more important half: a guard that flags good lines is ignored.

    Every line here is something this repo's docs actually contain, including the two shapes that
    would break a looser pattern: `uv pip install --python .venv/bin/python` (the interpreter is an
    *argument* to uv, not the command) and the inference server's own venv, which is a different venv
    from the sidecar's and still has to name one.
    """
    text = "\n".join(
        [
            "```bash",
            ".venv/Scripts/python.exe run.py",
            ".venv/bin/python -m pytest -v",
            "python -m venv .venv",
            "# POSIX:  python -m venv .venv && source .venv/bin/activate",
            "uv venv --python 3.12 .venv",
            "uv pip install --python .venv/bin/python -r requirements.txt",
            ".venv-inference/Scripts/python.exe local_inference_server.py",
            "make sidecar-run",
            "cd sidecar && .venv/Scripts/python.exe -m pytest -v",
            "```",
            "```python",
            "python = 'not a command'",
            "```",
        ]
    )
    found, fences = scan(text)

    assert fences == 1, "the Python sample block is not a shell fence"
    assert [(inv.command, inv.exempt) for inv in found] == [
        ("python -m venv .venv", True),
        ("python -m venv .venv", True),
    ]


def test_the_same_rule_fires_on_a_real_doc():
    """The mutation goes into a real block, not only into synthetic text.

    The cases above prove the reader understands the shapes; this proves the corpus is the hand-written
    thing a reader copies from - a doc whose fences are formatted some other way (indented, opened by
    prose, inside a list) is where a reader actually gets the command.
    """
    text = (REPO_ROOT / "QUICKSTART.md").read_text(encoding="utf-8")
    assert "```bash" in text, "QUICKSTART.md no longer has a shell block to mutate"

    found, _fences = scan(text.replace("```bash", "```bash\npython run.py", 1))

    assert [inv.command for inv in found] == ["python run.py"]
    assert not found[0].exempt, "the injected command is the one this guard exists for"


def test_the_scan_reaches_the_invocations_that_are_legal_by_rule():
    """The other half of the canary, on real text: the venv rule has to keep being exercised.

    `python -m venv .venv` is the whole of this corpus's bare-interpreter usage, and it is legal by
    rule rather than by exemption - so it is also the proof that the walker is entering these fences
    at all. Synthetic text in the tests above proves the reader works; this proves it is pointed at
    the docs.
    """
    hits, _fences = scan_corpus()
    seen = {doc for doc, found in hits.items() if any(inv.exempt for inv in found)}
    assert set(EXEMPT_EXAMPLES) <= seen, (
        "the `python -m venv` lines this rule legalises were not found, which means either the "
        "scan stopped entering fences or the docs stopped showing the setup: expected "
        f"{sorted(EXEMPT_EXAMPLES)}, saw {sorted(seen)}"
    )


def test_every_instruction_doc_is_scanned():
    """The corpus canary: a doc that fell out of the walk would take its blocks with it."""
    hits, fences = scan_corpus()
    missing = [doc for doc in INSTRUCTION_DOCS if doc not in fences]
    assert not missing, (
        "these docs tell a reader what to run and were not scanned at all - a renamed or moved file "
        "leaves this guard reading nothing on that side: " + ", ".join(missing)
    )
    silent = [doc for doc in INSTRUCTION_DOCS if fences[doc] == 0]
    assert not silent, (
        "an instruction doc with no shell block at all: either its commands moved somewhere this "
        "walk does not read, or the fence syntax changed under it: " + ", ".join(silent)
    )


def test_the_exemptions_are_still_live():
    """An exemption cannot outlive the line it was written for.

    The same rule `test_docs_claims` holds its own list to, for the same reason: a phrase left behind
    after an edit keeps covering text nobody has read, and the next bare `python` written near it
    would pass. Vacuously true while `EXEMPT` is empty, and the point is that it stops being vacuous
    the moment somebody adds an entry.
    """
    for doc, phrase, reason in EXEMPT:
        assert phrase in (REPO_ROOT / doc).read_text(encoding="utf-8"), (
            f"{doc} no longer contains {phrase!r}, so this exemption ({reason}) is covering text "
            "that is not there - delete it, or point it at what replaced it"
        )
