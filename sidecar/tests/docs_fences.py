"""The reader behind the three docs guards: fenced shell in, documented names out.

The docs are how this project is run: `README.md` and `QUICKSTART.md` take a fresh clone to a running
app, `docs/DEVELOPMENT.md` is the step-by-step beside them, the two package READMEs cover their own
side, and `CLAUDE.md` plus the run-desktop skill are what an agent is told to follow. Every one of
them answers "how do I run this?" with a fenced block, and a fenced block is meant to be pasted -
which is the fact the guards are built on.

Three things a pasted line can get wrong, and each has its own module with its own rules and its own
canaries:

* `test_docs_interpreters.py` - the *interpreter*: a bare `python` in command position, which is the
  system interpreter the uv path these docs lead with never activated.
* `test_docs_paths.py` - the *name*: a path, `make` target or npm script that is no longer there.
* `test_docs_options.py` - the *option*: a flag or verb the script it runs no longer declares.

What those three share is reading - and only reading. This module turns a document's text into the
things the rules have opinions about: its shell fences, the logical line each command starts on, the
commands on that line (prompts, platform labels and notes removed), whether a line is an
**invocation** or a **note**, the references on it (paths, `make` targets, npm scripts, flags, verbs),
the files those resolve to, and the argv a script's own source declares. No verdict, no floor and no
canary lives here; naming one of those is what makes the module that needs it own the answer.

Two readings are the reader's own, and both exist because the corpus is prose as well as commands:

* A trailing `\\` joins the next line onto this one - a wrapped `--dataset-dir` is part of the command
  above it, not a line of its own - and the line number kept is the first one, which is the line a
  reader sees.
* A `#` that is not a platform label starts a *note*. `# Windows: cmd` is a command behind a label;
  `# Naming rule (MODEL_TRAINING.md §8.2)` is prose about a file rather than a use of one, which is
  what keeps the latter out of the path check without a special case for tables.

The invocation/note decision itself is here because all three rules depend on it: a line is read as an
invocation when it leads with a word from `RUNNERS` or with a path, and as a note otherwise - 177 of
this corpus's 416 command-ish parts are notes, over 155 distinct leading words - and the lines that
matter are the ones a reader can paste, because a table is not pasted. A line that leads with a *path*
is still ambiguous, since a listing's rows look the same, so that one is settled by typography rather
than vocabulary: this repo aligns a listing's annotations into a column (`class_names.txt     the 7
grocery SKUs`) and draws its diagrams with arrows (`settings.json ──load at startup──► Settings`),
while a command is paths and flags on single spaces.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# The docs that tell a reader to run something. Every one has to be in the scanned corpus - that is
# what the modules' corpus canaries hold - because a corpus that quietly stopped matching a path is how
# these guards would stop guarding anything.
INSTRUCTION_DOCS = (
    "README.md",
    "QUICKSTART.md",
    "CLAUDE.md",
    "docs/DEVELOPMENT.md",
    "docs/DETECTOR_BACKENDS.md",
    "sidecar/README.md",
    "desktop/README.md",
    ".claude/skills/run-desktop/SKILL.md",
)

# Directories that never hold hand-written instructions: build output, virtualenvs, caches, and the
# scratch trees this workflow writes (`graphify-out`, `.superpowers/`, pytest's own). The same list
# `scripts/check_doc_links.py` walks the repo with, plus that second scratch tree - the reports under
# it are a session's own notes, not instructions.
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".superpowers",
    ".venv",
    ".venv-inference",
    "__pycache__",
    "dist",
    "graphify-out",
    "node_modules",
    "out",
}

# Subtrees that are a *record* rather than instructions. `docs/superpowers/` holds the plans and specs
# of sessions that ran before any of this convention existed, and they quote the commands those
# sessions used - `python -m pytest` among them. Editing history to satisfy a guard is the guard
# rewriting its own evidence, and the reader who opens a plan is not following a setup step.
HISTORY_PREFIXES = (("docs", "superpowers"),)
# The dataset workspace: tool output, not documentation. (`scripts/check_doc_links.py` skips it too.)
WORKSPACE_PREFIXES = (("sidecar", "data"),)

#: The info strings whose contents are commands. An empty one is in, because this repo writes shell
#: in bare fences as well as in ` ```bash ` - and a guard that only read the labelled ones would miss
#: every block that forgot the label.
SHELL_FENCES = {
    "",
    "bash",
    "bat",
    "cmd",
    "console",
    "ps1",
    "powershell",
    "pwsh",
    "sh",
    "shell",
    "zsh",
}

FENCE = re.compile(r"^\s*```(\S*)")
#: Every place a shell starts a new command. Good enough for the shapes these docs use (`cd x && y`,
#: a pipe, a `;` list) and deliberately not a shell grammar: a bare interpreter inside a `$(...)`
#: substitution is not a line a reader copies on its own.
COMMAND_BOUNDARY = re.compile(r"&&|\|\||;|\|")
#: `$ cmd` / `> cmd`, the prompt some blocks show.
PROMPT = re.compile(r"^(?:\$\s+|>\s+)")
#: `# Windows: cmd` / `# POSIX: cmd` - this repo's convention for showing one platform per line, so
#: the command behind the label is as documented as a bare one. The label is capped at a few words so
#: that a sentence ending in a colon does not make the rest of it look like a command.
COMMENT_LABEL = re.compile(r"^#\s*[A-Za-z][A-Za-z0-9_./ -]{0,23}:\s*")
#: Any other `#`: a note *about* the command, not a command. The two are read differently on purpose
#: - see the module docstring - and `# Naming rule (MODEL_TRAINING.md §8.2)` inside a fence is the
#: case that makes the distinction worth having.
COMMENT = re.compile(r"^#")
#: `\` at the end of a line: the command continues on the next one. The docs wrap long tool
#: invocations, and a wrapped `--dataset-dir sidecar/data/...` is an argument of the command above it
#: rather than a line that leads with a flag.
CONTINUATION = re.compile(r"\\\s*$")
#: Everything from a `#` that follows whitespace: the trailing note on a *command* line
#: (`--yes       # generate`), which is prose in the same way an unlabelled comment is.
TRAILING_COMMENT = re.compile(r"(?<=\s)#.*$")
#: `[label](./target.md)` - the target is the path a reader clicks, so it is the half that is checked
#: and the label is the half that is prose. `scripts/check_doc_links.py` is the guard for links as a
#: whole; this only has to stop a label from looking like a path.
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^()\s]+)\)")

#: The words a pasted line can *lead* with and still be a command. Open on purpose - a new tool in a
#: block should not need an edit here - and `RUNNERS_IN_USE` in `test_docs_paths.py` is the closed
#: half: the ones this corpus actually leads with, pinned by a canary there. A line that leads with
#: anything else is read as a note unless the word is a path, which is what keeps the corpus's
#: tables, trees and diagrams - 667 distinct leading tokens, against nine of these - out of the check
#: entirely.
RUNNERS = frozenset(
    {
        "Add-MpPreference",
        "bash",
        "cd",
        "chmod",
        "cmd",
        "cp",
        "curl",
        "docker",
        "export",
        "git",
        "ls",
        "make",
        "mkdir",
        "mv",
        "node",
        "npm",
        "npx",
        "pip",
        "pip3",
        "powershell",
        "pwsh",
        "py",
        "pytest",
        "python",
        "python3",
        "rm",
        "set",
        "sh",
        "source",
        "tar",
        "unzip",
        "uv",
        "uvicorn",
        "winget",
    }
)

#: What a documented path can end with and still be a *source* the repo carries: code, config, docs,
#: an interpreter. Weights, images, exports and datasets are deliberately absent - those are files
#: the docs tell a reader to *produce*, so their absence is the expected state of a fresh clone, and
#: the artifact prefixes below cover the trees they land in.
CODE_SUFFIXES = (
    ".bat",
    ".cjs",
    ".cfg",
    ".css",
    ".exe",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
)

#: The repo's top-level directories, read once from disk rather than restated: a path is accepted as
#: a path when it starts with one of these (`sidecar/tools/clamp_probe.py`, `.claude/skills/...`), and
#: deriving the list is what stops a new top-level directory from being unmentionable. Prose that
#: carries a slash (`filtering/remapping`, `265/265`, `century-tuna/far`) starts with none of them and
#: with no code suffix, which is the whole separation - no word list, no stem guessing.
TOP_LEVEL_DIRS = frozenset(
    entry.name for entry in REPO_ROOT.iterdir() if entry.is_dir() and entry.name not in SKIP_DIRS
)

#: Prefixes - matched by path *parts*, not by string - whose contents a checkout is not expected to
#: carry: the two venvs, the dataset workspace, the weight drop-in, build output, training runs. A doc
#: naming one of these is naming where something will be, so it is exempt from the existence check
#: rather than failing on a fresh clone and in CI. `ARTIFACT_EXEMPT_TODAY`, in `test_docs_paths.py`,
#: is the price, measured: the number of path references that ride on this list today.
ARTIFACT_PREFIXES = (
    (".venv",),
    (".venv-inference",),
    ("data",),
    ("desktop", "node_modules"),
    ("desktop", "out"),
    ("dist",),
    ("graphify-out",),
    ("node_modules",),
    ("out",),
    ("runs",),
    ("sidecar", ".venv"),
    ("sidecar", ".venv-inference"),
    ("sidecar", "data"),
    ("sidecar", "models"),
)

#: Characters that say "only the reader knows this value": `<workspace>`, `$HOME`, a glob, a quoted
#: string. A token carrying one is not a path this repo could be asked to have.
PLACEHOLDER_CHARS = frozenset("<>$*{}%~^\"'`…")
#: The directory-shaped half of the same idea - the docs' own stand-in for a path the reader
#: supplies, in every place they need one (`data=path/to/data.yaml`, `SIDECAR_SCRIPT=/path/to/run.py`).
PLACEHOLDER_PREFIXES = ("path/to",)
#: How a *listing* lays its annotations out: a column gap, not a word. The three file rows of the
#: cached-model tree (`class_names.txt     the 7 grocery SKUs`) are separated from the commands by
#: nothing else, and one command in the corpus has a stray double space of its own - which is why
#: this only ever decides a line that *leads with a path*, where the alternative reading is a table.
COLUMN_GAP = re.compile(r"\s{2,}|\t")
#: How a *diagram* gets drawn: box drawing, arrows, bullets. `settings.json ──load at startup──►
#: Settings` is a line of a picture, not an invocation, and there is no word-based way to say so.
DIAGRAM_GLYPH = re.compile(r"[\u2190-\u21ff\u2500-\u257f\u25a0-\u25ff]")

#: `VAR=value`. A line may lead with these (`SIDECAR_PYTHON=... SIDECAR_SCRIPT=... npm run dev`), so
#: the invocation's *word* is the first token that is not one - and the value half is still read as a
#: path, which is how `/path/to/run.py` gets skipped as a placeholder rather than as an absence.
ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: The punctuation a shell line hangs on a token: `(cd desktop && npm test)` and `"quoted values"`.
#: Deliberately *not* a leading `.` or `/` - `.venv/Scripts/python.exe` and `.claude/...` are paths,
#: and stripping a leading dot turns the first into a path that does not exist.
LEADING_PUNCTUATION = re.compile(r"^[('\"\[`]+")
TRAILING_PUNCTUATION = re.compile(r"[)\].,;:'\"`]+$")

#: A Makefile target: `name:` at the start of a line, and not `name :=` (an assignment).
MAKE_TARGET = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*)\s*:(?!=)")
PHONY = re.compile(r"^\.PHONY:\s*(.*)$", re.M)
#: `make -C desktop test`: these flags take a value, so the token after one is not the target. Nothing
#: in the corpus's fences uses them today; the table exists so that adding one cannot silently turn
#: `make -C desktop test` into a check of the target `desktop`.
MAKE_VALUE_FLAGS = frozenset({"-C", "-f", "-I", "--directory", "--file", "--makefile"})

#: `npm test` is shorthand for `npm run test`; everything else bare (`npm install`, `npm ci`,
#: `npm rebuild`) is a subcommand of npm itself and names no script.
NPM_RUN_SHORTHANDS = frozenset({"restart", "start", "stop", "test"})

#: The interpreter on a line, by its last path segment: `.venv/Scripts/python.exe`,
#: `sidecar/.venv/bin/python`, a versioned `python3.12`, and the bare ones the interpreter rule
#: rejects (their flags are still worth reading on the way past).
INTERPRETER = re.compile(r"(?:^|/)(?:python[0-9.]*|py)(?:\.exe)?$")
#: `--tolerance -0.5`: a token that starts with a dash and *reads as a number* is a flag's value.
#: Nothing in the corpus does this today; the pattern is here so that the first doc that does cannot
#: have its value reported as an option the tool dropped.
NEGATIVE_NUMBER = re.compile(r"^-\d")

#: The kinds a `Reference` can have, in the vocabulary the failure messages use. The kinds are all
#: read here because one line produces them together; which of them a module *judges* is that
#: module's business - `test_docs_paths.py` takes path/make/npm and `test_docs_options.py` the rest.
REFERENCE_KINDS = ("path", "make", "npm", "flag", "subcommand")


@dataclass(frozen=True)
class Reference:
    """One path or command name a documented shell block tells a reader to use."""

    doc: str
    line: int
    kind: str
    #: The path as the docs write it (separators normalised), or the target/script/option name.
    target: str
    fence: str
    #: For a `flag` or a `subcommand`: the script it is an option or a verb *of*. Empty for the rest,
    #: because a path is its own owner and a Make target's is the Makefile.
    script: str = ""

    @property
    def artifact(self) -> bool:
        """Under a tree a checkout is not expected to carry, so not this rule's to check."""
        return self.kind == "path" and _artifact_parts(self.target) is not None

    @property
    def spelled(self) -> str:
        """The reference the way a failure message should name it."""
        return f"{self.script} {self.target}".strip() if self.script else self.target


@dataclass(frozen=True)
class Block:
    """One shell fence: its info string and the numbered lines inside it."""

    info: str
    lines: tuple[tuple[int, str], ...]


def blocks(text: str) -> tuple[Block, ...]:
    """Every shell block in `text`, with the fence's info string held until its closer.

    Held, not toggled: ` ```json ` opens a block whose lines are not shell, and the *closer* that ends
    it is indistinguishable from an opener by sight. Reading the closer as an opener is how a walker
    starts scanning a document's prose at the wrong line - so the label lives as long as the block
    does, and a shell label is what makes a block the kind these rules read.
    """
    found: list[Block] = []
    label: str | None = None
    body: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        opener = FENCE.match(raw)
        if opener:
            if label is None:
                label = opener.group(1).lower()
                body = []
            else:
                if label in SHELL_FENCES:
                    found.append(Block(info=label, lines=tuple(body)))
                label = None
            continue
        if label is not None:
            body.append((number, raw))
    return tuple(found)


def _logical(lines: Iterable[tuple[int, str]]) -> list[tuple[int, str]]:
    """`lines` with `\\`-continued ones joined, numbered by the line the command *starts* on.

    One reading, shared by every rule: a wrapped invocation is one command, and reporting the second
    half of it as a line of its own would both mis-number the failure and hand the rules a line that
    leads with a flag. The line number kept is the first one, which is the line a reader sees.
    """
    joined: list[tuple[int, str]] = []
    buffer: list[str] = []
    first = 0
    for number, raw in lines:
        if not buffer:
            first = number
        buffer.append(raw.strip())
        if not CONTINUATION.search(raw):
            joined.append((first, " ".join(buffer).strip()))
            buffer = []
    if buffer:
        joined.append((first, " ".join(buffer).strip()))
    return joined


def _commands(line: str) -> list[str]:
    """The commands on one logical line: prompts, platform labels and notes removed.

    A `#` that is not a label ends the part - the rest is prose about the command - and a trailing
    note ends it too. Both are the same rule seen from two ends of a line, and both exist because the
    rule modules' own tests have to be able to discuss the commands they reject.
    """
    found: list[str] = []
    for part in COMMAND_BOUNDARY.split(MARKDOWN_LINK.sub(r"\1", line)):
        text = part.strip()
        if not text:
            continue
        text = PROMPT.sub("", text)
        if COMMENT.match(text):
            if not COMMENT_LABEL.match(text):
                continue
            text = COMMENT_LABEL.sub("", text)
        text = TRAILING_COMMENT.sub("", text).strip("()").strip()
        if text:
            found.append(text)
    return found


def _bare(token: str) -> str:
    """`token` without the punctuation a shell line hangs on it: quotes, brackets, commas, a period."""
    return TRAILING_PUNCTUATION.sub("", LEADING_PUNCTUATION.sub("", token))


def _invocation(command: str) -> tuple[int, str]:
    """The index and the spelling of the word that leads `command`, skipping `VAR=value` prefixes."""
    words = command.split()
    for index, token in enumerate(words):
        if ENV_ASSIGNMENT.match(token):
            continue
        return index, _bare(token)
    return len(words), ""


def _candidate(token: str) -> str | None:
    """`token` as a repo path, or None when it is not a path at all.

    The narrow half of the path rule, and the reason it does not need a list of prose words: a path
    here either carries a code-ish suffix or starts with a directory this repo has at its top level.
    Every slashed word this corpus's diagrams are made of fails both - `filtering/remapping`,
    `265/265`, `century-tuna/far`, `A/B` - while `train_model.py` and `sidecar/data/datasets/s2`
    pass, which is exactly the distinction between a line that names a file and a line that takes a
    slash.
    """
    token = _bare(token.replace("\\", "/"))
    if not token:
        return None
    if ENV_ASSIGNMENT.match(token):
        token = _bare(token.split("=", 1)[1].replace("\\", "/"))
        if not token:
            return None
    if token.startswith(("-", "/", "~")):
        return None
    if token.startswith(("http:", "https:", "git@")):
        return None
    if set(token) & PLACEHOLDER_CHARS:
        return None
    token = token.split("::")[0]  # a pytest node id: the file is ours, the node is the reader's
    while token.startswith("./"):
        token = token[2:]
    token = token.rstrip("/")
    if not token or token.startswith(PLACEHOLDER_PREFIXES):
        return None
    head = token.split("/")[0]
    if token.endswith(CODE_SUFFIXES) or ("/" in token and head in TOP_LEVEL_DIRS):
        return token
    return None


def _is_invocation(command: str) -> bool:
    """Whether `command` is a line a reader could paste, rather than a note that happens to be fenced.

    Two readings, because the corpus has two kinds of line that start with something a shell could
    run, and only one of them is a command.

    A word from `RUNNERS` is unambiguous - nothing writes `make`, `npm` or `cp` at the start of a
    table row - so those lines are commands whatever else is on them, stray spacing included.

    A line that starts with a bare *path* is the ambiguous case, because a tree's and a diagram's
    rows start with a path too. What separates them is typography rather than vocabulary: this
    repo's listings align their annotations into a column (`class_names.txt     the 7 grocery SKUs`)
    and its diagrams are drawn with arrows (`settings.json ──load at startup──► Settings`), while a
    command is a path followed by flags, values and other paths on single spaces. Measured over the
    corpus: three listing rows and one diagram line, against no command misread - which is the shape
    a rule has to have, since reading a listing as commands makes every produced file look broken.
    """
    if not command.split():
        return False
    _index, head = _invocation(command)
    if head in RUNNERS:
        return True
    if _candidate(head) is None and _artifact_parts(head) is None:
        return False
    return not (COLUMN_GAP.search(command) or DIAGRAM_GLYPH.search(command))


def _make_target(args: list[str]) -> str | None:
    """`make`'s target out of its arguments, or None when the line names only flags."""
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token in MAKE_VALUE_FLAGS:
            skip = True
            continue
        if token.startswith("-") or "=" in token:
            continue
        return _bare(token)
    return None


def _npm_script(args: list[str]) -> str | None:
    """The script `npm` is being asked for, or None when the line asks for a subcommand."""
    words = [_bare(token) for token in args if not token.startswith("-")]
    if words[:1] == ["run"]:
        return words[1] if len(words) > 1 else None
    if words[:1] and words[0] in NPM_RUN_SHORTHANDS:
        return words[0]
    return None


def _module_script(module: str) -> str | None:
    """`-m annotate.human_pass` as the file it resolves to, or None when it is somebody else's.

    Resolved by *existence* rather than by a list of names: `-m pytest`, `-m venv` and `-m pip` are
    the tools a reader already has, and the only way to tell one of those from this repo's own module
    is whether the module is a file here.
    """
    if not module:
        return None
    candidate = "sidecar/" + module.replace(".", "/") + ".py"
    return candidate if (REPO_ROOT / candidate).exists() else None


def _run_script(command: str) -> tuple[str, list[str]] | None:
    """`(the repo script this line runs, the tokens after it)`, or None when it runs none of ours.

    Three shapes, which is all this corpus has: a `.py` path at the head, an interpreter followed by
    one, and an interpreter followed by `-m <module>`. Everything else - `make`, `npm`, `node
    driver.mjs`, `pytest -v`, `uv pip install` - is another rule's business or nobody's, and returning
    None for them is what keeps their tools' own flags (`-r`, `-v`) out of the option check.
    """
    index, head = _invocation(command)
    words = [_bare(token) for token in command.split()[index:]]
    if head.endswith(".py"):
        return head, words[1:]
    if not INTERPRETER.search(head):
        return None
    rest = words[1:]
    if rest[:1] == ["-m"]:
        script = _module_script(rest[1] if len(rest) > 1 else "")
        return (script, rest[2:]) if script else None
    for offset, token in enumerate(rest):
        if token.endswith(".py"):
            return token, rest[offset + 1 :]
    return None


def _references_on(command: str, doc: str, line: int, fence: str) -> list[Reference]:
    """Every reference on one pasted command: its paths, and a target or script if it names one."""
    found: list[Reference] = []
    index, head = _invocation(command)
    args = command.split()[index + 1 :]
    if head == "make":
        target = _make_target(args)
        if target:
            found.append(Reference(doc=doc, line=line, kind="make", target=target, fence=fence))
    if head == "npm":
        script = _npm_script(args)
        if script:
            found.append(Reference(doc=doc, line=line, kind="npm", target=script, fence=fence))
    for token in command.split():
        path = _candidate(token)
        if path is not None:
            found.append(Reference(doc=doc, line=line, kind="path", target=path, fence=fence))
    run = _run_script(command)
    if run is not None:
        script, rest = run
        script = _script_key(script, doc)
        if rest and not rest[0].startswith("-"):
            found.append(
                Reference(
                    doc=doc,
                    line=line,
                    kind="subcommand",
                    target=rest[0],
                    fence=fence,
                    script=script,
                )
            )
        found.extend(
            Reference(
                doc=doc,
                line=line,
                kind="flag",
                target=token.split("=")[0],
                fence=fence,
                script=script,
            )
            for token in rest
            if token.startswith("-") and token != "--" and not NEGATIVE_NUMBER.match(token)
        )
    return found


def scan_references(text: str, doc: str = "") -> tuple[list[Reference], int]:
    """Every reference in `text`'s shell blocks, and how many such blocks there were.

    The fence count is returned rather than derived by a caller, because it is what proves the walker
    entered a block: a file with no shell fences scanned and a file whose blocks were never opened
    look identical from the list alone.
    """
    found: list[Reference] = []
    shell = blocks(text)
    for block in shell:
        for number, line in _logical(block.lines):
            for command in _commands(line):
                if not _is_invocation(command):
                    continue
                found.extend(_references_on(command, doc, number, block.info))
    return found, len(shell)


def _walk(names: Callable[[str], bool]) -> Iterator[Path]:
    """Every file under the repo those `names` accepts, in a stable order.

    Walked with the skips pruning directories rather than filtering the result: `node_modules` and
    the two venvs are the only reason a `rglob` over this repo is not instant, and descending into
    them to then throw the files away is most of what a scan here costs.
    """
    for root, dirnames, filenames in os.walk(REPO_ROOT):
        here = Path(root)
        relative = here.relative_to(REPO_ROOT).parts
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if any(relative[: len(p)] == p for p in HISTORY_PREFIXES + WORKSPACE_PREFIXES):
            dirnames[:] = []
            continue
        yield from (here / name for name in sorted(filenames) if names(name))


def _corpus() -> list[Path]:
    """Every Markdown file that is instructions, in a stable order."""
    return sorted(_walk(lambda name: name.endswith(".md")))


def scan_reference_corpus() -> tuple[list[Reference], dict[str, int]]:
    """`(every documented reference, {doc: shell fences})` over the whole corpus."""
    found: list[Reference] = []
    fences: dict[str, int] = {}
    for path in _corpus():
        doc = path.relative_to(REPO_ROOT).as_posix()
        references, blocks_read = scan_references(path.read_text(encoding="utf-8"), doc)
        found.extend(references)
        fences[doc] = blocks_read
    return found, fences


def corpus_commands() -> list[tuple[str, str]]:
    """`(doc, command)` for every part of every shell fence in the corpus.

    One walk, for the questions that are about the *lines* rather than about the docs: which are
    commands and which are notes, and which scripts they run. In one place because a second walk that
    forgot the platform labels or the continuation joining would answer a different question than the
    rules do - and a canary that walks differently from the rule it guards is worth nothing.
    """
    found: list[tuple[str, str]] = []
    for path in _corpus():
        doc = path.relative_to(REPO_ROOT).as_posix()
        for block in blocks(path.read_text(encoding="utf-8")):
            for _number, line in _logical(block.lines):
                found.extend((doc, command) for command in _commands(line))
    return found


def corpus_parts() -> tuple[list[str], list[str]]:
    """`(invocations, notes)` - the corpus's fenced parts, split by the line reader itself.

    The reader's own answer, without the reference rules on top of it, so `test_docs_paths.py` can
    hold both sides of the decision: too many invocations means a listing is being read as commands,
    and too few means the blocks a reader pastes are not being read at all.
    """
    invocations: list[str] = []
    notes: list[str] = []
    for _doc, command in corpus_commands():
        (invocations if _is_invocation(command) else notes).append(command)
    return invocations, notes


def _artifact_parts(path: str) -> tuple[str, ...] | None:
    """The `ARTIFACT_PREFIXES` entry `path` sits under, or None - matched by parts, not by string."""
    parts = tuple(path.split("/"))
    for prefix in ARTIFACT_PREFIXES:
        if parts[: len(prefix)] == prefix:
            return prefix
    return None


def _script_key(script: str, doc: str) -> str:
    """A script's identity: its repo-relative path when it is on disk, else the spelling doc used.

    The docs write the same tool three ways - `clean_v2.py` after a `cd sidecar/tools`,
    `tools/clean_v2.py` after a `cd sidecar`, `sidecar/tools/clean_v2.py` from the root - and those
    spellings are not three tools. One file per tool is what the option rule reads, and it is what a
    canary can list: a list of spellings would not notice one of them going missing, since the other
    two would still be there.
    """
    path = _resolved(script, doc)
    return path.relative_to(REPO_ROOT).as_posix() if path and path.is_file() else script


def _resolved(path: str, doc: str) -> Path | None:
    """The file a reader pasting this line would land on, or None.

    Four roots, because the prose around a block is what says where a reader is: the doc's own
    directory (`DEPLOYMENT.md` inside `docs/`), the repo root (where the setup blocks start), and the
    two package roots (which the `cd sidecar` / `cd desktop` blocks are written for). A path in none
    of them is the thing this rule exists for - a name no reader can paste from anywhere.
    """
    for base in ((REPO_ROOT / doc).parent, REPO_ROOT, REPO_ROOT / "sidecar", REPO_ROOT / "desktop"):
        candidate = base / path
        if candidate.exists():
            return candidate
    return None


def _exists(path: str, doc: str) -> bool:
    """Whether a reader pasting this line lands on a file."""
    return _resolved(path, doc) is not None


def makefile_targets(text: str) -> frozenset[str]:
    """Every target a Makefile answers to: `name:` rules plus the `.PHONY` names."""
    targets = set(MAKE_TARGET.findall(text))
    phony = PHONY.search(text.replace("\\\n", " "))
    if phony:
        targets.update(phony.group(1).split())
    return frozenset(targets)


def package_scripts(texts: Iterable[str]) -> frozenset[str]:
    """Every `scripts` key across some `package.json` texts, so a doc's `npm run x` has one answer."""
    scripts: set[str] = set()
    for text in texts:
        try:
            scripts.update(json.loads(text).get("scripts", {}))
        except (ValueError, AttributeError):
            continue
    return frozenset(scripts)


def _makefile_targets() -> frozenset[str]:
    return makefile_targets((REPO_ROOT / "Makefile").read_text(encoding="utf-8"))


def _npm_scripts() -> frozenset[str]:
    return package_scripts(
        path.read_text(encoding="utf-8")
        for path in _walk(lambda name: name == "package.json")
    )


def report(bad: Iterable[Reference]) -> str:
    """The failing references, one per line, in the shape every rule's message uses."""
    return "\n  ".join(f"{ref.doc}:{ref.line}: {ref.kind} {ref.spelled}" for ref in bad)
