"""A documented flag or verb the tool no longer has: the third way a pasted line fails.

The names inside a command drift too, and they are the half a doc quotes most freely: `--strict`,
`--yes`, `--generation v1`, and the verbs the dataset docs are written in (`clean_v2.py clean`,
`upload`, `retag`, `sanity`, `scaffold`, `wipe`). Dropping one is the same silent breakage as moving
a file - every mention still reads as an instruction - except that a flag's absence is also a
*refusal*: argparse answers `unrecognized arguments`, exits 2, and the reader is the one who finds
out.

So the options are read from the file that declares them, the way the targets are read from the
Makefile: a line's script is found (a `.py` at the head, or the one an interpreter is handed, or
`-m <module>` when that module is a file here), and every flag and every leading verb on the line is
checked against that script's own `add_argument`/`add_parser` calls. Reading text rather than
importing is deliberate - these entrypoints import torch, ultralytics and the app's runtime in their
module bodies, so a guard that imported thirteen of them would cost more than the suite it belongs to
and would fail wherever a heavy dependency is missing. The prices of that choice are stated with it:
a parser built through a shared helper is in the module that defines it, not the one that calls it;
argparse's own `-h`/`--help` are assumed (unless the file passes `add_help=False`); and a flag that
lives on a *sibling* subparser counts, so the claim is "this tool knows the option" rather than "this
particular invocation is well-formed" - measured rather than assumed: all twelve of the corpus's
(verb, flag) pairs are reachable for the verb on their line today, and attributing options per
subparser would need `parents=[...]` accounted for, which is the fragile reading that would report a
flag a shared parent declares as missing. One more is closed rather than paid: a script that declares
nothing accepts nothing, so `NO_OPTIONS` names the two entrypoints that are started and left alone,
and `test_every_documented_script_has_a_command_line` fails when a third appears - which is what a
tool losing its whole parser would look like.

The reading - fences, the script a line runs, and the flags and verbs on it - is in
`tests/docs_fences.py`, shared with the interpreter and path rules; the *paths* those same lines name
are `test_docs_paths.py`'s to judge. This module owns the parser reader, the option verdict and its
canaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import pytest

from tests.docs_fences import (
    Reference,
    _is_invocation,
    _resolved,
    _run_script,
    _script_key,
    corpus_commands,
    report,
    scan_reference_corpus,
    scan_references,
)

pytestmark = pytest.mark.docs

#: `add_argument("--strict", ...)` / `sub.add_parser("clean", ...)` - the two calls that make a
#: script's command line. Read as text rather than by importing, for the reason in the docstring.
ADD_ARGUMENT = re.compile(r"\badd_argument\s*\(")
ADD_PARSER = re.compile(r"\badd_parser\s*\(")
#: An option string, and argparse's own `-h`/`--help`, which no `add_argument` call declares.
OPTION_LITERAL = re.compile(r"['\"](-{1,2}[A-Za-z0-9][\w-]*)['\"]")
VERB_LITERAL = re.compile(r"['\"]([\w-]+)['\"]")
BUILT_IN_OPTIONS = frozenset({"-h", "--help"})

#: A script the docs run that declares no options at all, with the reason: both are started and left
#: alone, so a flag on one of them is the failure this rule exists for rather than an exemption from
#: it. `test_every_documented_script_has_a_command_line` keeps the list honest from both sides.
NO_OPTIONS = (
    ("sidecar/run.py", "started with no arguments: `SIDECAR_PORT=<n>` on stdout is the whole contract"),
    ("sidecar/local_inference_server.py", "started with no arguments; its address comes from settings"),
)

#: The option rule's own floors. Measured today: the corpus's fenced lines hand 107 options and 22
#: verbs to 15 of this repo's scripts, over 30 (script, doc) pairs. A resolver that stopped finding
#: the script on a line, or a reader that stopped seeing `add_argument`, would take all three towards
#: zero - and an empty check passes everything, which is the failure mode a floor is here to catch
#: rather than to document.
FLAG_FLOOR = 90
SUBCOMMAND_FLOOR = 18
SCRIPT_FLOOR = 12


@dataclass(frozen=True)
class Cli:
    """What one script's own argument parser answers to, read from the file that builds it.

    Two sets, because they fail the same way and are found in the same two calls: a *flag* the tool
    no longer declares (`--strict`) and a *subcommand* it no longer answers to (`clean_v2.py drop`)
    are both a documented word the tool has forgotten. `found` is the third fact, and it is the
    path rule's rather than this one's: a script that is not on disk at all has already been reported
    once, and reporting every option on the line as missing beside it is noise.
    """

    options: frozenset[str] = frozenset()
    subcommands: frozenset[str] = frozenset()
    found: bool = True

    def knows(self, reference: Reference) -> bool:
        """Whether this parser declares `reference` - a flag, or a verb if it declares any at all."""
        if reference.kind == "flag":
            return reference.target in self.options
        # A bare word on a tool with no subparsers is a positional *value* (`--weights x.pt`), not a
        # verb: argparse has no list of those to check it against, and guessing one is not this rule.
        return not self.subcommands or reference.target in self.subcommands


def _source_calls(text: str, call: re.Pattern[str]) -> Iterator[str]:
    """The inside of every `call(...)` in Python `text`, with brackets and quoted strings respected.

    Balanced-paren extraction, and quoted strings matter for the same reason they do in
    `test_server_loop.py`'s scan of uvicorn call sites: a `help="..."` string is free to contain a
    bracket, a comma or the name of an option, and a reader that ended the call there would report
    the tool's own help text as a declaration.
    """
    for match in call.finditer(text):
        index = match.end()
        depth = 1
        start = index
        while index < len(text) and depth:
            character = text[index]
            if character in "([{":
                depth += 1
            elif character in ")]}":
                depth -= 1
            elif character in "\"'":
                quote = character
                index += 1
                while index < len(text) and text[index] != quote:
                    index += 2 if text[index] == "\\" else 1
            index += 1
        yield text[start : index - 1]


def _segments(call: str) -> list[str]:
    """`call`'s arguments, split at the commas that are neither bracketed nor quoted."""
    parts: list[str] = []
    depth = 0
    current = ""
    index = 0
    while index < len(call):
        character = call[index]
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character in "\"'":
            quote = character
            current += character
            index += 1
            while index < len(call) and call[index] != quote:
                current += call[index]
                index += 2 if call[index] == "\\" else 1
            current += quote
            index += 1
            continue
        elif character == "," and depth == 0:
            parts.append(current)
            current = ""
            index += 1
            continue
        current += character
        index += 1
    parts.append(current)
    return parts


KEYWORD = re.compile(r"^\s*[A-Za-z_]\w*\s*=(?!=)")


def _positional(call: str) -> list[str]:
    """The arguments `call` passes positionally, up to its first `key=value`.

    That is where an option string lives: `add_argument("--out", type=int, help="a, b")` has one
    option and two keywords, and the difference matters because the *help text* is free to contain a
    flag's name - a reader that took every string in the call would count the tool's own prose as a
    declaration and then accept a flag the tool no longer has.
    """
    found: list[str] = []
    for segment in _segments(call):
        if KEYWORD.match(segment):
            break
        found.append(segment)
    return found


def read_cli(text: str) -> Cli:
    """The command line a script's own source declares: `(options, subcommands)`.

    One pass over the file that builds the parser, because that file is the owner of the answer - the
    same relationship `makefile_targets` has with the Makefile. Two things argparse answers to are not
    a call of its own: `-h`/`--help` come from `add_help` - added only when the file builds a parser
    at all, since `run.py` is started and left alone and answers nothing, and dropped when the file
    passes `add_help=False` - and a parser shared through a helper (`resources.add_args(ap)`) lives in
    the module that defines it rather than the one that calls it. That second one is the price of
    reading text instead of importing, and it is paid loudly: the corpus's own options are all
    declared in the files its lines name, and the day one is not, this reports it as a missing option
    rather than passing it.
    """
    options: set[str] = set()
    subcommands: set[str] = set()
    for argument in _source_calls(text, ADD_ARGUMENT):
        for segment in _positional(argument):
            options.update(OPTION_LITERAL.findall(segment))
    for parser in _source_calls(text, ADD_PARSER):
        positional = _positional(parser)
        verb = VERB_LITERAL.search(positional[0] if positional else "")
        if verb:
            subcommands.add(verb.group(1))
    builds_a_parser = "ArgumentParser" in text or "add_argument" in text
    if builds_a_parser and "add_help=False" not in text.replace(" ", ""):
        options |= BUILT_IN_OPTIONS
    return Cli(options=frozenset(options), subcommands=frozenset(subcommands))


@lru_cache(maxsize=None)
def _cli(script: str, doc: str) -> Cli:
    """`script`'s own command line, read from the file it names.

    Cached by `(script, doc)`, because the corpus runs `clean_v2.py` eighteen times and the answer is
    the same file each time - and because the *reader* is pure, so a cache here cannot hide a change
    inside an import.
    """
    path = _resolved(script, doc)
    if path is None or not path.is_file():
        return Cli(found=False)
    return read_cli(path.read_text(encoding="utf-8"))


def documented_scripts() -> set[tuple[str, str]]:
    """`{(script, doc)}` - every script of this repo the corpus's fenced lines run.

    The scripts a line *runs*, which is what the option rule checks and what the note rule skips: a
    line that runs none this repo owns (`pytest -v`, `uv pip install -r x`, `npm run build`) is
    somebody else's interface, and the tools' own flags are not this module's to judge.
    """
    found: set[tuple[str, str]] = set()
    for doc, command in corpus_commands():
        if not _is_invocation(command):
            continue
        run = _run_script(command)
        if run is not None:
            found.add((_script_key(run[0], doc), doc))
    return found


def problems(references: list[Reference], *, cli=_cli) -> list[Reference]:
    """The flag and subcommand references a reader cannot use, in the order they were found.

    Pure over the parser facts rather than reading the disk itself, so the rule can be pointed at a
    scratch set of parsers - and the path, `make` and npm verdicts are `test_docs_paths.py`'s, which
    owns those kinds.
    """
    bad: list[Reference] = []
    for reference in references:
        if reference.kind not in ("flag", "subcommand"):
            continue
        parser = cli(reference.script, reference.doc)
        # A script that is not on disk is the path rule's finding, and it has already made it.
        if not parser.found or parser.knows(reference):
            continue
        bad.append(reference)
    return bad


def test_every_documented_flag_is_one_the_tool_declares():
    """The third way a pasted line fails: an option the tool no longer has.

    A flag is half of a tool's interface and the half a doc quotes most freely - `--strict`, `--yes`,
    `--generation v1` - and dropping one is a rename that leaves every mention of it reading as an
    instruction. `argparse` answers `unrecognized arguments` and exits 2, so the reader finds out; the
    suite should find out first, because the file that declares the options is in this repo.
    """
    references, _fences = scan_reference_corpus()
    flags = [ref for ref in problems(references) if ref.kind == "flag"]
    assert not flags, (
        "a documented command passes an option the script it runs does not declare - argparse "
        "answers `unrecognized arguments` and exits, so the doc is telling a reader to run "
        "something that cannot work. Rename it in the doc, or add it to the tool's parser if the "
        "doc is the right one:\n  " + report(flags)
    )


def test_every_documented_subcommand_is_one_the_tool_declares():
    """The same rule one word along: the verb, not the flag.

    `clean_v2.py clean` / `upload` / `retag` / `sanity` / `scaffold` / `wipe` / `drop` is the shape
    the dataset docs are written in, and a renamed verb is the same silent breakage as a renamed
    flag - except that nothing about a *word* looks like a name, so it is the harder one to spot.
    """
    references, _fences = scan_reference_corpus()
    verbs = [ref for ref in problems(references) if ref.kind == "subcommand"]
    assert not verbs, (
        "a documented command asks a tool for a subcommand its parser does not register - the "
        "`add_parser` calls in that tool are the list of verbs it answers to. Rename it in the doc, "
        "or add the verb back:\n  " + report(verbs)
    )


def test_the_option_rules_reach_the_docs():
    """The canary for the option half of the scan: a resolver that stopped finding the script on a
    line, or a reader that stopped seeing `add_argument`, would take all three towards zero - and an
    empty check passes everything, which is the failure a floor is here to catch.

    The path/make/npm floors live in `test_docs_paths.py`, which owns those kinds.
    """
    references, _fences = scan_reference_corpus()
    flags = [ref for ref in references if ref.kind == "flag"]
    verbs = [ref for ref in references if ref.kind == "subcommand"]
    scripts = documented_scripts()
    assert len(flags) >= FLAG_FLOOR, (
        f"only {len(flags)} documented options found (floor {FLAG_FLOOR}): the corpus's "
        "invocations either stopped naming this repo's scripts or stopped being read, and this rule "
        "has gone quiet without saying so"
    )
    assert len(verbs) >= SUBCOMMAND_FLOOR, (
        f"only {len(verbs)} documented subcommands found (floor {SUBCOMMAND_FLOOR}) - the verbs are "
        "how the dataset docs are written, so a drop here is a resolver that has stopped finding "
        "the script a line runs"
    )
    assert len(scripts) >= SCRIPT_FLOOR, (
        f"only {len(scripts)} distinct scripts are run by the corpus's fenced lines (floor "
        f"{SCRIPT_FLOOR}), which cannot be right when the three dataset docs alone run six of them - "
        "the resolver that finds the script on a line has given up"
    )


def test_the_cli_reader_reads_the_parser_and_not_the_help_text():
    """The reader's own positive and negative controls, on source shaped like the tools'.

    The shapes that matter are the three a naive `add_argument` scrape gets wrong: a call whose
    option string is on the next line, a pair of spellings in one call (`-o`, `--overwrite`), and a
    `help=` string that mentions an option - which must not be collected, precisely because a reader
    that grabbed every string in the call would count the tool's own prose as a declaration and
    would then accept a flag the tool dropped. The fourth is argparse's own `-h`/`--help`, which no
    call declares, and the fifth is a file that says `add_help=False` and means it.
    """
    source = "\n".join(
        [
            "def main():",
            "    ap = argparse.ArgumentParser()",
            '    ap.add_argument("--dataset", default=".", help="the merged set to measure on")',
            '    ap.add_argument("--strict", action="store_true")',
            "    ap.add_argument(",
            '        "--out",',
            '        default=".",',
            '        help="where the report goes; the other tool spells it --not-an-option",',
            "    )",
            '    ap.add_argument("-o", "--overwrite", action="store_true")',
            '    ap.add_argument("--x", help="a (paren, a , comma")',
            '    sub = ap.add_subparsers(dest="cmd")',
            '    clean = sub.add_parser("clean", help="ingest")',
            '    clean.add_argument("--src")',
            '    upload = sub.add_parser("upload")',
        ]
    )
    cli = read_cli(source)

    assert cli.options == frozenset(
        {"--dataset", "--strict", "--out", "-o", "--overwrite", "--x", "--src", "-h", "--help"}
    )
    assert "--not-an-option" not in cli.options, "the help text is not the interface"
    assert cli.subcommands == frozenset({"clean", "upload"})
    assert read_cli("ap = argparse.ArgumentParser(add_help=False)\n").options == frozenset(), (
        "a parser that turns help off does not answer `--help`, and assuming it does would be the "
        "reader inventing an option"
    )


def test_the_option_rules_catch_a_flag_or_verb_that_is_gone():
    """The positive control for the option rule, over the shapes a line can take.

    Each line is one way into the resolver - a plain script, a `-m` module, an external module that
    is nobody's business, a script that declares nothing at all, and a script that is not on disk -
    and the two failures are a dropped flag and a dropped verb.
    """
    text = "\n".join(
        [
            "```bash",
            "sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean --src x --gone",
            "sidecar/tools/spec_check.py --generation v1 --strict-mode",
            ".venv/Scripts/python.exe -m annotate.human_pass --check --missing",
            ".venv/Scripts/python.exe -m pytest tests/test_docs_options.py -v",
            "sidecar/.venv/Scripts/python.exe run.py --port 9000",
            "sidecar/.venv/Scripts/python.exe sidecar/tools/gone.py --strict",
            "sidecar/tools/train_model.py v2",
            "```",
        ]
    )
    parsers = {
        "sidecar/tools/clean_v2.py": Cli(frozenset({"--src"}), frozenset({"clean", "upload"})),
        "sidecar/tools/spec_check.py": Cli(frozenset({"--generation"}), frozenset()),
        "sidecar/annotate/human_pass.py": Cli(frozenset({"--check"}), frozenset()),
        "sidecar/run.py": Cli(frozenset(), frozenset()),
        "sidecar/tools/gone.py": Cli(found=False),
        "sidecar/tools/train_model.py": Cli(frozenset({"--weights"}), frozenset()),
    }
    references, fences = scan_references(text)
    bad = problems(references, cli=lambda script, _doc: parsers.get(script, Cli()))

    assert fences == 1
    assert [(ref.kind, ref.spelled) for ref in references if ref.kind in ("flag", "subcommand")] == [
        ("subcommand", "sidecar/tools/clean_v2.py clean"),
        ("flag", "sidecar/tools/clean_v2.py --src"),
        ("flag", "sidecar/tools/clean_v2.py --gone"),
        ("flag", "sidecar/tools/spec_check.py --generation"),
        ("flag", "sidecar/tools/spec_check.py --strict-mode"),
        ("flag", "sidecar/annotate/human_pass.py --check"),
        ("flag", "sidecar/annotate/human_pass.py --missing"),
        ("flag", "sidecar/run.py --port"),
        ("flag", "sidecar/tools/gone.py --strict"),
        ("subcommand", "sidecar/tools/train_model.py v2"),
    ], "the pytest line names no script of ours, so none of its options are this rule's to check"
    assert [(ref.kind, ref.spelled) for ref in bad] == [
        ("flag", "sidecar/tools/clean_v2.py --gone"),
        ("flag", "sidecar/tools/spec_check.py --strict-mode"),
        ("flag", "sidecar/annotate/human_pass.py --missing"),
        ("flag", "sidecar/run.py --port"),
    ], (
        "a script that declares nothing accepts nothing, a script that is not on disk is the path "
        "rule's finding and not a failed option, and a bare word on a tool with no subparsers is a "
        "positional value rather than a verb"
    )


def test_every_documented_script_has_a_command_line():
    """The hole this rule would otherwise have, kept open on purpose and named.

    A script that declares no options cannot have any of its flags checked - which is the right
    answer for the two entrypoints that are started and left alone, and the wrong one for a tool that
    has lost its parser entirely (every flag on its lines would then read as fine). So every script
    the docs run has to either declare something or be listed in `NO_OPTIONS` with its reason, and
    each listing is held to both halves: still run by some doc, still declaring nothing.
    """
    pairs = documented_scripts()
    silent = {
        script
        for script, doc in pairs
        if not (_cli(script, doc).options or _cli(script, doc).subcommands)
    }
    named = {script for script, _reason in NO_OPTIONS}

    assert silent == named, (
        "a script the docs run declares no options at all, so nothing on its lines is checked and a "
        "tool that lost its parser would read as healthy. Either it is one of the entrypoints that "
        "are started and left alone (add it to NO_OPTIONS with the reason), or the parser it uses "
        "is built somewhere this reader does not look: "
        f"unlisted {sorted(silent - named)}, stale {sorted(named - silent)}"
    )
    for script, reason in NO_OPTIONS:
        assert script in {name for name, _doc in pairs}, (
            f"NO_OPTIONS lists {script} ({reason}) and no doc runs it any more - an exemption cannot "
            "outlive the line it was written for"
        )
        assert not (_cli(script, "").options or _cli(script, "").subcommands), (
            f"{script} is listed as declaring no options and now declares some - the listing is a "
            "claim about the file, and the file is what answers"
        )
