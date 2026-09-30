"""A documented path or command name that is no longer *there*: the second way a pasted line fails.

Naming an interpreter by path is only half of a line that works. `sidecar/tools/clamp_probe.py` can
become `tools/clamp_probe.py`, a `make` target can be renamed, an npm script can be deleted - and
every place the old spelling was written down keeps reading as instructions. Nothing else in this
suite can see it: the tools' own tests import what exists, the `makefile` job runs the targets `make
help` lists, `scripts/check_doc_links.py` reads Markdown *links*, and the CI workflow's own steps are
*executed* - so a stale name there fails loudly by running. A doc's commands are the ones nothing
runs, so a reader is the only thing that would have noticed, and they notice by pasting it and getting
`No such file or directory`.

So a line inside a shell fence is read as an **invocation** when its first word is a command from
`RUNNERS` or a path, and then everything path-shaped on it is checked against the checkout, a `make`
target against the Makefile's own targets, and an `npm` script against a `package.json`'s. Lines that
lead with neither - or that lead with a path and turn out to be a row of a listing - are **notes**, and
none of their contents are read, which is what keeps most of this corpus's fenced text out of the
check. That line-reader decision is exercised here because this is the rule it protects: a reading
that let a listing through as commands would report every produced file as a stale one.

Four things are deliberately out of scope, each because checking it would be a guess rather than a
fact, and a guard that guesses is a guard that gets turned off:

* **A tree the repo does not carry.** `ARTIFACT_PREFIXES` - the two venvs, the dataset workspace,
  `sidecar/models`, `runs/`, build output - are where things *will* be: a weight, a staged capture, an
  export. The docs have to be able to name them, and a fresh clone is exactly the state where they do
  not exist yet. The cost is paid rather than hidden: a path inside one of those trees is not checked
  at all, so a doc naming a *file* under one (`sidecar/models/README.md`, which is tracked) is not
  held to it either. Measured today: 121 of the corpus's 229 path references sit under one of them.
* **A path the reader supplies.** `path/to/data.yaml`, `<workspace>/merged-v2`, `--out DIR`, `$HOME`,
  a glob: the docs' own stand-in for a value only the reader knows. `PLACEHOLDER_PREFIXES` and
  `PLACEHOLDER_CHARS` are what that looks like here, and they are the same reason `SIDECAR_SCRIPT=
  /path/to/run.py` is skipped twice over.
* **Whether a tool is installed.** `uv`, `docker`, `npx` and the binary `npx` would fetch are facts
  about a machine, not about this checkout, and `make verify-clamp` needs weights the repo does not
  carry. This rule is about names, and the only thing it can settle is whether the name is one this
  repo still answers to.
* **The block's working directory.** A path is accepted if it exists under the repo root, under the
  doc's own directory, or under `sidecar/`/`desktop/` - because the prose around a block is what says
  which of those a reader is in, and a path in *none* of them is the thing that cannot be right. A
  `cd` argument is therefore not checked at all: `cd ../desktop` means one thing after `cd sidecar`
  and another at the root, and adjudicating that from the fence alone is the guess this rule refuses.
  A pytest node id keeps its file half (`tests/test_pipeline.py::test_name` is checked as the file,
  since the node is the reader's to choose) and a Markdown link is checked by its *target* rather
  than its label.

The reading - fences, logical lines, the invocation/note split, the reference scan, and which files
exist - is in `tests/docs_fences.py`, shared with the interpreter and option rules. This module owns
the verdict for the three kinds it checks and the canaries that prove the reader is still deciding:
the invocation/note floors, the runner list, and the measures of what it finds and what it skips. The
*flag* and *verb* half of the reference scan is `test_docs_options.py`'s to judge.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pytest

from tests.docs_fences import (
    REFERENCE_KINDS,
    REPO_ROOT,
    RUNNERS,
    Reference,
    corpus_commands,
    corpus_parts,
    leading_word,
    makefile_targets,
    npm_scripts,
    path_exists,
    report,
    scan_reference_corpus,
    scan_references,
)

pytestmark = pytest.mark.docs

#: The runners the corpus is *known* to lead a line with, measured over the fences it reads: `cd`,
#: `curl`, `git`, `make`, `node`, `npm`, `python`, `uv` (plus the interpreter paths and the
#: `SIDECAR_PYTHON=...` prefix, which are paths or assignments rather than words). Two claims are
#: held over it - that they are all in `RUNNERS`, and that the scan still finds each of them leading
#: a line - so dropping one from the set, or a change to the walker that stops seeing line starts,
#: fails here rather than by silently checking nothing.
RUNNERS_IN_USE = ("cd", "curl", "git", "make", "node", "npm", "python", "uv")

#: How many references the rules have to find in the corpus for the scan to be believed. Measured
#: today: 229 path references, 32 `make` and 20 `npm` ones, and 121 of the paths riding on
#: `ARTIFACT_PREFIXES`. The floors sit under those - at roughly the fraction a single document's
#: worth of blocks would cost - so that a doc losing a block does not fail a healthy tree, while a
#: walker that stopped reading fences, or a candidate rule that stopped matching, does.
#: `ARTIFACT_EXEMPT_TODAY` is the other half of the same idea, for the exemption list rather than for
#: the scan: an exemption nothing rides on is a rule that has quietly stopped being applied.
PATHS_FLOOR = 190
COMMANDS_FLOOR = 44
ARTIFACT_EXEMPT_TODAY = 100
#: And the two halves of the line reader, which is the reader every rule depends on: measured today,
#: 239 invocations and 177 notes. A floor under each is what says the reader is still *deciding* -
#: reading everything would make the path rule report a listing as dozens of stale files, and reading
#: nothing would make it report nothing at all, and both are visible here first.
INVOCATION_FLOOR = 200
NOTE_FLOOR = 140


def problems(
    references: Iterable[Reference],
    *,
    targets: frozenset[str],
    scripts: frozenset[str],
    exists: Callable[[str, str], bool] = path_exists,
) -> list[Reference]:
    """The path, Make-target and npm-script references a reader cannot use, in the order found.

    Pure over its facts rather than reading the disk itself, so the rule can be pointed at a scratch
    set of targets and scripts - and so the answer to "why is this one fine?" is one of `targets`,
    `scripts`, `exists`, or the artifact list, rather than a walk. A reference of another kind is
    left to `test_docs_options.py`, which owns the flag and verb verdict.
    """
    bad: list[Reference] = []
    for reference in references:
        if reference.kind == "path":
            if reference.artifact or exists(reference.target, reference.doc):
                continue
        elif reference.kind == "make":
            if reference.target in targets:
                continue
        elif reference.kind == "npm":
            if reference.target in scripts:
                continue
        else:
            continue
        bad.append(reference)
    return bad


def test_every_pasted_path_is_one_this_checkout_has():
    """The other half of a line that works: the file it names has to still be there.

    A renamed tool, a moved module, a doc that kept the old home - none of them fail anywhere else,
    because the tests import what exists and the Makefile job runs the targets `make help` lists. The
    reader is the only thing that would find out, and by pasting it.
    """
    references, _fences = scan_reference_corpus()
    bad = problems(references, targets=makefile_targets(), scripts=npm_scripts())
    paths = [ref for ref in bad if ref.kind == "path"]
    assert not paths, (
        "a documented shell block names a path this checkout does not have, so pasting it is a "
        "`No such file or directory` with no hint that the doc is what is stale. Point it at where "
        "the file lives now - or, if the reader is meant to supply it, write it the way the other "
        "placeholders do (`path/to/...`, `<workspace>/...`). Checked against the repo root, the "
        "doc's own directory and the two package roots:\n  " + report(paths)
    )


def test_every_pasted_make_target_exists():
    """`make <target>` is a name this repo answers to, or it is a line that cannot run."""
    references, _fences = scan_reference_corpus()
    bad = problems(references, targets=makefile_targets(), scripts=npm_scripts())
    targets = [ref for ref in bad if ref.kind == "make"]
    assert not targets, (
        "a documented shell block asks for a Make target the Makefile does not have - `make` fails "
        "on it with `No rule to make target`, and `make help` is the index that lists what exists. "
        "Rename it in the doc, or add the target:\n  " + report(targets)
    )


def test_every_pasted_npm_script_exists():
    """`npm run <script>` is the same question with a `package.json` as the owner."""
    references, _fences = scan_reference_corpus()
    bad = problems(references, targets=makefile_targets(), scripts=npm_scripts())
    scripts = [ref for ref in bad if ref.kind == "npm"]
    assert not scripts, (
        "a documented shell block asks npm for a script no `package.json` declares, which npm "
        "answers with `Missing script: ...` - the desktop package's own `scripts` block is the list "
        "to check it against:\n  " + report(scripts)
    )


def test_the_reference_scan_reaches_the_docs():
    """The canary: a rule that stopped matching would find nothing and pass.

    Both halves are needed - that every kind of reference is found at all, and that they are found in
    the numbers the corpus has (the floors are stated under the measured ones at the top, so a doc
    losing a block is not a failure while a walker losing its fences is). The floor for the option
    rule's own kinds is `test_docs_options.py`'s, so a scan that lost *those* still fails somewhere
    the rule they belong to can see it. The corpus's own presence check - that every instruction doc
    is read at all, with a shell fence in it - is `test_docs_interpreters.py`'s: one walk, held to
    once, rather than the same assertion over a reader all three families share.
    """
    references, _fences = scan_reference_corpus()
    kinds = {ref.kind for ref in references}
    assert kinds == set(REFERENCE_KINDS), (
        "the scan found no reference of some kind, so that half of the rule is not being exercised "
        f"by this corpus at all: expected {sorted(REFERENCE_KINDS)}, saw {sorted(kinds)}"
    )
    paths = [ref for ref in references if ref.kind == "path"]
    commands = [ref for ref in references if ref.kind != "path"]
    assert len(paths) >= PATHS_FLOOR, (
        f"only {len(paths)} path references found where the corpus has far more (floor "
        f"{PATHS_FLOOR}) - the candidate rule or the fence walker has stopped reading the blocks a "
        "reader pastes"
    )
    assert len(commands) >= COMMANDS_FLOOR, (
        f"only {len(commands)} make/npm references found (floor {COMMANDS_FLOOR}): the docs' "
        "invocations are no longer being read, which would make the two rules above vacuous"
    )
    exempt = [ref for ref in paths if ref.artifact]
    assert len(exempt) >= ARTIFACT_EXEMPT_TODAY, (
        f"only {len(exempt)} path references ride on the artifact prefixes, where the corpus has at "
        f"least {ARTIFACT_EXEMPT_TODAY} - the exemption list or the paths it was written for have "
        "moved, and the half of the corpus it covers is now being checked by nobody"
    )


def test_the_line_reader_still_tells_commands_from_notes():
    """The claim the whole second rule rests on, held from both ends.

    A version of this that read every fenced line as a command would report each row of a listing as a
    stale file, and the one that read only the labelled `bash` blocks would miss the invocations the
    other blocks carry - so both counts are pinned, at the numbers the corpus is measured at today.
    """
    invocations, notes = corpus_parts()
    assert len(invocations) >= INVOCATION_FLOOR, (
        f"only {len(invocations)} of the corpus's fenced parts were read as commands (floor "
        f"{INVOCATION_FLOOR}), so the blocks a reader pastes are no longer being read and every "
        "reference rule above is closer to vacuous than it looks"
    )
    assert len(notes) >= NOTE_FLOOR, (
        f"only {len(notes)} were left as notes (floor {NOTE_FLOOR}) - the line reader has become "
        "permissive, which is the direction that turns a table row or a tree into a dozen stale "
        "paths; the corpus's listing rows and diagram lines are the shapes this is protecting"
    )


def test_the_runner_list_is_still_the_one_the_docs_lead_with():
    """`RUNNERS` has an open half and a measured one, and the measured one is pinned twice over.

    An open set is the only shape that does not need an edit per new tool, but an open set with
    nothing pinned can quietly stop covering the lines this corpus actually has - so every word the
    docs are known to lead with has to be in it, and has to still be *seen* leading a line.
    """
    unknown = [word for word in RUNNERS_IN_USE if word not in RUNNERS]
    assert not unknown, (
        f"{unknown} lead lines in this corpus's fences and are missing from RUNNERS, so lines that "
        "start with them are read as notes: a bare `python` behind one, or a path on one, is not "
        "checked"
    )
    seen = {leading_word(command) for _doc, command in corpus_commands()}
    unseen = [word for word in RUNNERS_IN_USE if word not in seen]
    assert not unseen, (
        f"{unseen} are declared as words the docs lead a line with and none was seen: either the "
        "docs changed and this tuple should follow them, or the walker stopped reading line starts "
        "and every rule here is now looking at nothing"
    )


def test_the_rules_catch_a_path_or_target_that_is_gone():
    """The positive control: the shapes that mean a reader is about to paste a broken line.

    Each case is one of the ways a name goes stale - a tool that moved, a target that was renamed, a
    script that was deleted, a path written for a directory that is not there - and each must be
    found by the scan *and* refused by the rule, since a rule that finds them and passes them is the
    same as no rule.
    """
    text = "\n".join(
        [
            "```bash",
            "sidecar/tools/clamp-probe.py --strict",  # a tool that was renamed
            "sidecar/.venv/Scripts/python.exe sidecar/tools/gone.py",  # a script that was deleted
            "make verify-clamp-typo",  # a target the Makefile does not have
            "make verify-unsure UNSURE_REQUIRE_BAND=1 DATA=1",
            "npm run typcheck",  # a script `package.json` does not declare
            "cd desktop && npm test",
            "docs/GONE.md",  # a doc path written from the repo root
            "```",
        ]
    )
    references, fences = scan_references(text)

    assert fences == 1
    assert [(ref.kind, ref.target) for ref in references if ref.kind not in ("flag", "subcommand")] == [
        ("path", "sidecar/tools/clamp-probe.py"),
        ("path", "sidecar/.venv/Scripts/python.exe"),
        ("path", "sidecar/tools/gone.py"),
        ("make", "verify-clamp-typo"),
        ("make", "verify-unsure"),
        ("npm", "typcheck"),
        ("npm", "test"),
        ("path", "docs/GONE.md"),
    ]
    assert [(ref.script, ref.kind, ref.target) for ref in references if ref.kind == "flag"] == [
        ("sidecar/tools/clamp-probe.py", "flag", "--strict"),
    ], "one option on one script: `cd desktop && npm test` is npm's line, and `make` takes none"
    bad = problems(
        references,
        targets=frozenset({"verify-unsure", "test"}),
        scripts=frozenset({"test"}),
        exists=lambda path, _doc: path == "sidecar/.venv/Scripts/python.exe",
    )
    assert [(ref.kind, ref.target) for ref in bad] == [
        ("path", "sidecar/tools/clamp-probe.py"),
        ("path", "sidecar/tools/gone.py"),
        ("make", "verify-clamp-typo"),
        ("npm", "typcheck"),
        ("path", "docs/GONE.md"),
    ], (
        "the venv path is an artifact exemption and the rest are names that exist - and the option "
        "on the missing script is not reported here, because the script's absence is the path rule's "
        "finding and a flag is the other module's to judge"
    )


def test_it_does_not_cry_wolf_on_tables_trees_and_placeholders():
    """The negative control, in two halves, both written from this corpus's own lines.

    The first half is what must not be *read* at all: the slashed prose of a diagram, a ratio, a tree,
    a table row, an unlabelled comment. None of them is a line a reader pastes, and treating them as
    commands is how this rule would need a list of prose words to stay quiet.

    The second half is what must not be *refused*: real invocations whose paths are artifacts, the
    reader's own values, or files that are there. The corpus resolves all of these today, and the
    distinction between the two halves is the whole reason the rule has roots, prefixes and
    placeholders rather than one existence check.
    """
    notes = "\n".join(
        [
            "```",
            "Logitech StreamCam ──USB──▶ sidecar/ (Python/FastAPI)   ──ws://127.0.0.1──▶ desktop/",
            "     OpenCV capture → YOLO11 track              live preview + overlay + item log",
            "SQLite detection log                     REST for start/stop/health/logs",
            "265/265   103/136   36/62",
            "century-tuna/far        century-tuna/mid       filtering/remapping      A/B",
            "    BEARBRAND/   CLOSE/   MID/   FAR/   NEGATIVES/",
            "Backend        (roboflow_core/inner_workflow@v1)",
            "def infer(self, frame):   # Ultralytics/PyTorch",
            "OK=$ok/10",
            "scanncart-grocery-v2.pt      v2, trained locally    <- what train_model.py installs",
            "# Naming rule (MODEL_TRAINING.md §8.2)",
            "```",
        ]
    )
    found, fences = scan_references(notes)

    assert fences == 1, "the bare block is a shell fence here; it is the labels that decide"
    assert found == [], (
        "none of these lines is a command: the diagram rows are notes, the ratios and the tree are "
        "slashed prose, and the comment is a note about a file rather than a use of one - so nothing "
        f"on them is a reference, and {[(ref.kind, ref.target) for ref in found]} was read as one"
    )

    real = "\n".join(
        [
            "```bash",
            "uv venv --python 3.12 .venv",
            "uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows",
            "SIDECAR_PYTHON=/path/to/sidecar/.venv/Scripts/python.exe SIDECAR_SCRIPT=/path/to/run.py npm run dev",
            "git clone <repo-url> scanncart",
            "npm install",
            "npm ci",
            "npm rebuild electron",
            "npx vitest run src/renderer/src/hooks/useSidecarStream.test.tsx",
            "sidecar/tools/spec_check.py --generation v1 --defaults  # see [the plan](./RUN_SHEET.md)",
            "Add-MpPreference -ExclusionPath \"C:/path/to/scanncart\"",
            "make human-pass HUMAN_PASS_ARGS=--check",
            "cd ../desktop && npm run typecheck",
            "cp /tmp/export.zip sidecar/data/datasets/",
            "docs/DETECTOR_BACKENDS.md §7a",
            "```",
        ]
    )
    references, fences = scan_references(real)
    bad = problems(references, targets=makefile_targets(), scripts=npm_scripts())

    assert fences == 1
    assert bad == [], (
        "these are the corpus's own working lines - artifact trees, the reader's paths, npm "
        f"subcommands, a doc that is there - and the rule refused {report(bad)}"
    )


def test_the_same_reference_rule_fires_on_a_real_doc():
    """The mutation goes into a real block, not only into synthetic text.

    The cases above prove the reader understands the shapes; this proves the corpus is the
    hand-written thing a reader copies from - a `make` line inside a real fence, with the target
    renamed in memory rather than in the file.
    """
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "make verify-clamp" in text, "CLAUDE.md no longer documents the clamp gate"

    references, _fences = scan_references(text.replace("make verify-clamp", "make verify-clamped"))
    bad = problems(
        references,
        targets=makefile_targets(),
        scripts=npm_scripts(),
    )

    assert [(ref.kind, ref.target) for ref in bad] == [("make", "verify-clamped")], (
        "a renamed target in a real doc has to be the only thing refused - anything else here is a "
        "rule reading a line it should have left alone"
    )
