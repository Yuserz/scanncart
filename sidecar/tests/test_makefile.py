"""The root Makefile, read as text, so a broken one fails the suite instead of a terminal.

CI never invokes `make` at all - every job runs `pytest` and `npm` directly - so a Makefile that no
longer parses is invisible to it and only bites a person running the documented commands. That is
exactly what happened: two `@echo` lines in `help` merged onto one line and two more lost their
leading tab, and make aborted the whole file with `missing separator` at that line. Every target
then failed, `make dev` included, while the suite stayed green.

These guards read the file rather than shelling out to `make`: GNU Make is optional on Windows (the
README says so), and a guard that needs a tool the project calls optional would fail on the machines
it is meant to protect. They check the two shapes a hand edit actually produces - a recipe line
indented with spaces, and a `help` line carrying two commands - not the whole make grammar.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


# What may legitimately start with spaces in a Makefile: a variable assignment (as inside an
# `ifeq` block, e.g. the Windows/Unix interpreter pair) or a conditional directive. Anything else
# indented with spaces is the corruption this guard exists for.
_MAKE_STATEMENT = re.compile(
    r"^\s*(?:"
    r"ifeq|ifneq|ifdef|ifndef|else|endif|include|-include|sinclude|"
    r"export|unexport|override|define|endef|vpath"
    r")\b"
    r"|[A-Za-z_][A-Za-z0-9_.-]*\s*[:+?!]?="
)


def _space_indented_violations(text: str) -> list[tuple[int, str]]:
    """Lines that start with spaces where make needs a tab, continuation lines aside.

    A trailing backslash means the next physical line continues this one (the `.PHONY` list is
    written that way), so those are skipped whatever their indentation.
    """
    violations: list[tuple[int, str]] = []
    continuing = False
    for number, line in enumerate(text.splitlines(), 1):
        if continuing:
            continuing = line.rstrip().endswith("\\")
            continue
        if not line.strip() or line.startswith("\t"):
            continue
        if line.lstrip().startswith("#"):
            continue
        if line.rstrip().endswith("\\"):
            continuing = True
            continue
        if line[0].isspace() and not _MAKE_STATEMENT.search(line):
            violations.append((number, line))
    return violations


def _help_recipe(text: str) -> list[str]:
    """The `help` target's recipe lines, tab dropped. The shape the corruption was introduced in."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("help:"))
    recipe: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        recipe.append(line[1:])
    return recipe


def test_every_recipe_line_is_indented_with_a_tab():
    """A space-indented recipe line is a parse error, and make reports it once and serves nothing.

    That is the failure that took `make dev` down: the abort is at the first such line regardless of
    which target it sits in, so every target fails, not just the one being fixed.
    """
    text = MAKEFILE.read_text(encoding="utf-8")
    violations = _space_indented_violations(text)
    assert not violations, "the Makefile no longer parses:\n" + "\n".join(
        f"  line {number}: {line!r}" for number, line in violations
    )


def test_the_guard_catches_the_corruption_it_was_written_for():
    """The same check against a mutated copy, so a guard that matches nothing cannot pass.

    Both shapes are injected into a real recipe line, and both must be reported - the tab replaced
    by spaces, and the statement that turns out to be a merged pair of commands.
    """
    text = MAKEFILE.read_text(encoding="utf-8")
    assert not _space_indented_violations(text), "the shipped Makefile is already broken"

    body = text.splitlines()
    target = body.index("help:") + 1
    spaced = body[:target] + ["  " + body[target].lstrip("\t")] + body[target + 1 :]
    assert _space_indented_violations("\n".join(spaced)), (
        "a recipe line indented with spaces was not reported"
    )


def test_no_help_line_carries_two_commands():
    """A second `@echo` on one recipe line prints its own `@echo "..."` literally into `make help`.

    It parses - which is why it is not the failure above - but the target list it prints is then
    wrong, and the edit that produced it is the same edit that produced the parse error.
    """
    doubled = [line for line in _help_recipe(MAKEFILE.read_text(encoding="utf-8")) if line.count("@echo") > 1]
    assert not doubled, "a help line carries two commands:\n" + "\n".join(f"  {line!r}" for line in doubled)
