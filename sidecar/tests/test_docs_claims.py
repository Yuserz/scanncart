"""Drift guards: the claims the docs make about the code, checked against the code.

The three docs this repo runs from (`CLAUDE.md`, `docs/RUN_SHEET.md`, `docs/MODEL_TRAINING.md`) plus
the run-desktop skill (`SKILL.md`, which describes the §12 verification procedure the other two point
at) are prose, and most of what they say cannot be checked at all. This module deliberately does not
try. What it pins is the five kinds of sentence that are really a *copy* of a code fact, because
those are the ones that survive a rename looking true:

1. **A class count.** `24 classes`, `24-output`, `the seven names`, `6 cells` - a head width, a
   roster size or a cell count. The code facts are the roster sizes per generation
   (`app.roster`) and the distance axis (`label_classes.DISTANCE_ORDER`), so a claim has to be a
   number those can produce. This is the guard for the eight-class round: the roster lost
   Palmolive, `V2_ROSTER` became v1's seven, and the docs kept reading `24` for as long as nobody
   could check them in prose.
2. **A default.** `| `conf_threshold` (default `0.5`) |` and `capture_width=1280` are statements
   about `Settings()`, parsed out of the doc and compared field by field.
3. **A floor, a ratio or a margin.** The `0.85` recall floor, the planner's `70/20/10`, and
   `CROWD_MIN=2` - each one number the code owns.
4. **A split name or an artifact name.** The splits published in `label_classes.SPLIT_NAMES`, and
   every backticked file name the docs use, which has to be something this repo's code or disk
   actually has.
5. **A contract table's own size.** `thirteen surfaces in one table` is `len(table.surfaces)`
   written out in prose, and `AGREEMENTS`'s two rows is `len(AGREEMENTS)` - adding a row to either
   table is the quietest way to make such a sentence wrong, and a table's *own* row count is the one
   thing every other guard here cannot see, since each of them checks the rows and never how many
   there are. Both counts are attributed to the table by name (the constant the guard declares it
   as), which is what makes them checkable rather than merely adjacent.

Everything asserted here is a fact about the *code*, never about the workspace: the live dataset
counts the run sheet also carries (`858/1433 decided`, `132 of 186 across 6 cells`) move with the
data and are deliberately not pinned - a guard on those would fail on a labeling session, which is
how a guard stops being read. Where a doc number looks like a claim but is not one (a diagnostic's
own count, a UI value the docs say must never appear), the exemption lives in `EXEMPT` at the foot of
this file: one row per phrase, with the reason it is allowed, the docs it was written about and the
*one claim shape* it is excused from - the row itself being `tests/exemptions.py`'s shape, shared with
the two other guards that keep such a table. A shape here is an arm *and* the pattern that matched,
because this arm has several: a phrase written for the count shape cannot silence the subset or the
roster-pair shape. `test_the_docs_guard_exemptions_are_still_live` fails when the phrase it covers is
edited away, when it stops excusing a claim its own shape still flags, or when any other shape it does
not name would be silenced by it - so an exemption cannot quietly cover new text, nor another shape's.

One more check sits beside those five: the §12 fixture the docs describe (`driver.mjs classlist`) has
to keep being the thing they say it is - a head sized by the roster x the distance axis. It is the
root of this round's stale counts, because prose describing a fixture nothing builds stays accurate
right up until the fixture stops being that.

And one check is not about a sentence at all, the tables themselves: a contract table that no doc
names and no check calls is a declaration that *reads* as maintained - its rows are validated at
construction, so it looks wired up - while nothing can fail on it and no reader was told it exists.
Which tables exist is read from the module they live in, so that arm needs no upkeep of its own.

Same shape as the other drift guards in this suite, and the same failure mode it exists for: a stale
doc reads as a healthy project, and prose is where a rename goes to be forgotten.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from functools import lru_cache
from pathlib import Path

import pytest

from app.roster import V1_ROSTER, V2_ROSTER
from app.settings import Settings
from app.settings_store import is_custom_model

import accept_v2
import clean_v2
import label_classes
import plan_split
import train_model

from tests.exemptions import Exemption, Exemptions, Waiver, assert_still_present, waived_note

pytestmark = pytest.mark.docs

REPO_ROOT = Path(__file__).resolve().parents[2]

# The docs as the repo ships them, in the order a reader meets them. `SKILL.md` is here because it
# describes the same §12 fixture `RUN_SHEET.md` and `MODEL_TRAINING.md` point at, and carried the
# same stale class count: a doc that restates another doc's claim is exactly as able to be wrong.
#
# Deliberately not `docs/CAPTURE_CHECKLIST.md`, which carries the same kind of prose but a different
# kind of number: "100 per class", "40 per cell", "350 per class" are capture *targets* and tier
# sizes, and the class-count arm cannot tell one from a roster size without being told which nouns
# mean a rate. Its roster claim ("the 8-class roster") was corrected by hand in the same round this
# guard was written; covering that file honestly means an arm that reads targets as targets.
DOCS = (
    "CLAUDE.md",
    "sidecar/app/CLAUDE.md",
    "sidecar/tools/CLAUDE.md",
    "desktop/CLAUDE.md",
    "sidecar/tests/CLAUDE.md",
    "docs/RUN_SHEET.md",
    "docs/MODEL_TRAINING.md",
    ".claude/skills/run-desktop/SKILL.md",
)

# Everything the repo spells in code, for the artifact-name check. The test files are in on purpose:
# the claim being checked is "this repo names this file somewhere", not "this module owns it". The
# hidden, vendored and workspace directories are out: `sidecar/.venv` holds every package the
# sidecar installed, and a corpus that broad accepts any name at all.
SOURCE_GLOBS = (
    "sidecar/*.py",
    "sidecar/app/**/*.py",
    "sidecar/tools/**/*.py",
    "sidecar/tests/**/*.py",
    "sidecar/annotate/**/*.py",
    "desktop/src/**/*.ts",
    "desktop/src/**/*.tsx",
    ".claude/skills/**/*.mjs",
    "Makefile",
)

# This module is not part of its own corpus: it has to write down the stale spellings it exists to
# catch (`merge-report.json` as the renamed form of `merge_report.json`), and a corpus that contains
# the checker would accept every name the checker explains.
SELF = Path(__file__).resolve()

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

# The exemptions - the phrases these docs are allowed to carry, the reason for each, and the one claim
# shape each is excused from - are in one place at the foot of this file (§7), with the check that holds
# them live and the shapes the arms produce. The arms below ask through `_exempt`, passing their own
# name and the pattern that matched, which is what keeps a phrase written for one shape from silencing
# another's.


@lru_cache(maxsize=None)
def _text(doc: str) -> str:
    return (REPO_ROOT / doc).read_text(encoding="utf-8")


def _line(doc: str, index: int) -> int:
    return _text(doc)[:index].count("\n") + 1


def _snippet(doc: str, start: int, end: int, pad: int = 30) -> str:
    text = _text(doc)
    return text[max(0, start - pad) : end + pad]


# --------------------------------------------------------------------------
# 1. class counts
# --------------------------------------------------------------------------

# The shapes the docs actually use for one: `7 classes`, `24-class`, `21 outputs`, `24 cells`,
# `seven-class`, `the seven names`, `the 24 distance-split names`. All four are anchored on a number
# or a number word immediately in front of a count noun, which is what keeps ordinary prose out
# ("one class v2 added", "three cells of that class", "v1's six raw capture archives" are not
# claims about a roster, and a guard that flagged them would be turned off within a week).
NOUN = r"(?:class|classes|output|outputs|name|names|cell|cells)"
WORDS = "|".join(NUMBER_WORDS)
COUNT_CLAIM = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<value>\d+)(?:\s+(?!per\b)[a-z]+(?:-[a-z]+)*)?[-\s](?P<noun>" + NOUN + r")\b"
    r"|(?<![A-Za-z0-9_-])(?P<word>" + WORDS + r")-(?P<wnoun>" + NOUN + r")\b(?! subset)"
    r"|\bthe (?:same )?(?P<theword>" + WORDS + r")(?: (?:allowed|product|class|distance-split))? "
    r"(?P<the_noun>names|classes|outputs|cells)\b"
)

# "the two-name subset the gate quotes from" is a count too, but of `accept_v2.GATE_SPLITS` rather
# than of a roster - so it is checked against its own owner, and excluded from the roster arm above
# (`(?! subset)`) so the two cannot both claim it.
SUBSET_CLAIM = re.compile(r"(?<![A-Za-z0-9_-])(?P<value>\d+|(?:" + WORDS + r"))-name subset\b")

# The one sentence that names both generations' rosters side by side - "v1's seven, or v2's eight" -
# is checked against each generation's *own* size, which is what makes it catch a roster that changed
# while the sentence that explains `resolve_roster` stayed the same.
ROSTER_PAIR = re.compile(
    r"(?P<a>v[12])'s (?P<a_value>" + WORDS + r")\s*,?\s*(?:or|and)\s*"
    r"(?P<b>v[12])'s (?P<b_value>" + WORDS + r")\b"
)

ROSTER_NOUNS = ("class", "classes", "output", "outputs", "name", "names")


def _reachable_counts() -> set[int]:
    """Every class count the code can produce: a generation's roster, or a head per product and
    distance (the failure `roster.class_list_problems` reports). Read from the owners rather than
    written down, so adding a class to `V2_ROSTER` moves this set with it.
    """
    rosters = {len(V1_ROSTER), len(V2_ROSTER)}
    distances = len(label_classes.DISTANCE_ORDER)
    return rosters | {size * distances for size in rosters}


def _allowed_counts(noun: str) -> set[int]:
    counts = _reachable_counts()
    if noun.startswith("cell"):
        # Tier A's gap is named in cells too, and it is its own count rather than the whole grid.
        return counts | {len(clean_v2.TIER_A_CELLS)}
    return counts


def test_every_class_count_in_the_docs_is_one_the_code_can_produce():
    """A head width, a roster size or a cell count the code cannot produce is a stale doc.

    `24 classes` is the case this exists for: it was 8 products x 3 distances while v2 still held
    Palmolive, and after the drop it is 7 x 3 - a sentence nothing in the app reads, so it stayed
    on the page through a dataset change, a retrain and a weight drop. The remedy is the doc's, not
    a setting's.

    What it skipped is collected as it goes (`waived`) and stated at the end, and carried in the failure
    message if there is one: those sentences are the ones a reader would otherwise have to know the
    table to find, and a row this scan never reaches is a waiver nobody is using.
    """
    problems = []
    checked = 0
    waived: list[Waiver] = []
    for doc in DOCS:
        text = _text(doc)
        for match in COUNT_CLAIM.finditer(text):
            raw = match.group("value") or match.group("word") or match.group("theword")
            value = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
            noun = match.group("noun") or match.group("wnoun") or match.group("the_noun")
            if value in _allowed_counts(noun):
                checked += 1
                continue
            snippet = _snippet(doc, match.start(), match.end())
            if _exempt(ARM_CLASS_COUNTS, COUNT_CLAIM, doc, snippet, waived):
                checked += 1
                continue
            problems.append(
                f"{doc}:{_line(doc, match.start())}: {match.group(0)!r} - the code's counts are "
                f"{sorted(_allowed_counts(noun))}"
            )
        for match in SUBSET_CLAIM.finditer(text):
            checked += 1
            raw = match.group("value")
            value = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
            if value != len(accept_v2.GATE_SPLITS):
                problems.append(
                    f"{doc}:{_line(doc, match.start())}: {match.group(0)!r} - the gate's splits are "
                    f"{list(accept_v2.GATE_SPLITS)}"
                )
        for match in ROSTER_PAIR.finditer(text):
            checked += 1
            sizes = {"v1": len(V1_ROSTER), "v2": len(V2_ROSTER)}
            for generation, value in (
                (match.group("a"), match.group("a_value")),
                (match.group("b"), match.group("b_value")),
            ):
                if NUMBER_WORDS[value] != sizes[generation]:
                    problems.append(
                        f"{doc}:{_line(doc, match.start())}: {match.group(0)!r} - "
                        f"{generation} has {sizes[generation]} names"
                    )
    # A guard that stops matching anything passes silently; the docs state class counts today.
    assert checked > 20, f"the class-count scan matched only {checked} claims - did the shape change?"
    # The trace, stated: every row this scan can apply was applied. A row that is live - its phrase
    # present, its claim still flagged - but which this scan never reached is a waiver nobody is using,
    # and a problem list cannot report that, because it is empty exactly when the exemptions work.
    count_shape = _shape(ARM_CLASS_COUNTS, COUNT_CLAIM)
    applied = {waiver.phrase for waiver in waived}
    rows_of_shape = {row.phrase for row in EXEMPT.rows if row.shape == count_shape}
    assert applied == rows_of_shape, (
        f"this scan applied {sorted(applied) or 'no'} exemption(s) where the {count_shape!r} rows are "
        f"{sorted(rows_of_shape)}: a row this scan consults and never reaches is a waiver nobody is "
        "using - drop it, or point it at the sentence that replaced the one it was written for"
    )
    assert not problems, (
        "stale class count(s) in the docs:\n" + "\n".join(problems) + waived_note(waived)
    )


# --------------------------------------------------------------------------
# 2. settings defaults
# --------------------------------------------------------------------------

# `| `conf_threshold` (default `0.5`) |` - the tuning table's own shape.
DEFAULT_ROW = re.compile(r"\|\s*`(?P<field>[a-z_]+)`\s*\(default\s*`(?P<value>[^`]+)`\)")
# `capture_width=1280` - a default stated inline, next to the module that holds it.
DEFAULT_ASSIGN = re.compile(r"`(?P<field>[a-z_]+)=(?P<value>[^`]+)`")

FIELDS = {field.name: field for field in dataclasses.fields(Settings)}


def _same(doc_value: str, code_value: object) -> bool:
    doc_value = doc_value.strip()
    if isinstance(code_value, bool):
        return doc_value.lower() == str(code_value).lower()
    try:
        return float(doc_value) == float(code_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return doc_value == str(code_value)


def test_settings_defaults_named_in_the_docs_match_the_settings_dataclass():
    """Both spellings of "this field defaults to X" are checked against `Settings()`.

    The inline shape (`capture_width=1280`) is the one that was actually stale: `settings.py` says
    640x480@30 - the geometry that opens reliably over USB 2.0 - while the capture guidance still
    told a shooter to record at 1280x720@60 and *claimed those were the defaults in that file*.
    """
    problems = []
    checked = 0
    for doc in DOCS:
        text = _text(doc)
        for pattern in (DEFAULT_ROW, DEFAULT_ASSIGN):
            for match in pattern.finditer(text):
                field = match.group("field")
                if field not in FIELDS:
                    continue
                code_value = getattr(Settings(), field)
                checked += 1
                if not _same(match.group("value"), code_value):
                    problems.append(
                        f"{doc}:{_line(doc, match.start())}: {field} is documented as "
                        f"{match.group('value')!r}, Settings() says {code_value!r}"
                    )
    assert checked >= 3, f"only {checked} default(s) found - did the table's shape change?"
    assert not problems, "stale setting default(s) in the docs:\n" + "\n".join(problems)


# --------------------------------------------------------------------------
# 3. the augmentation table, the recall floor, the split ratio, the crowding margin
# --------------------------------------------------------------------------


def test_the_augmentation_table_the_docs_print_is_the_trainers():
    """`RUN_SHEET.md` §6 shows the printed `augmentation:` line, and `train_model.augmentation_kwargs`
    is what prints it. One is a doc, one is the run: a doc that lists the old values is a doc telling
    an operator to expect a training run this tool would not produce.
    """
    text = _text("docs/RUN_SHEET.md")
    matches = re.findall(r"augmentation: (?P<pairs>[^\n`]+)", text)
    assert len(matches) == 1, "expected exactly one printed augmentation line in RUN_SHEET.md §6"
    documented = [pair.strip() for pair in matches[0].split(",")]
    printed = [
        f"{key}={value:g}" for key, value in train_model.augmentation_kwargs().items()
    ]
    assert documented == printed, (
        "the augmentation line in RUN_SHEET.md §6 is not what train_model.py prints:\n"
        f"  doc:   {documented}\n  code:  {printed}"
    )


def test_the_recall_floor_named_in_the_docs_is_the_trainers():
    """The floor, in both directions: no number standing next to the word "floor" may differ from
    `RECALL_FLOOR`, and a line that states a floor states the current one.

    The second half is what catches a floor that moved - a doc can keep saying the old value in a
    sentence that never says "floor" (`the recall the floor is computed over (0.918)` is a measured
    number, which is why the check is "a line mentioning both a floor and a number", not "a line
    mentioning a floor").
    """
    floor = train_model.RECALL_FLOOR
    before = re.compile(r"(?<![.\d])(\d\.\d+)\s+floor\b")
    after = re.compile(r"\bfloor\b[^.\n(:]{0,14}?(?<![.\d])(\d\.\d+)")
    problems = []
    for doc in DOCS:
        for number, line in enumerate(_text(doc).splitlines(), start=1):
            if "floor" not in line or not re.search(r"\d\.\d+", line):
                continue
            for pattern in (before, after):
                for match in pattern.finditer(line):
                    if float(match.group(1)) != floor:
                        problems.append(
                            f"{doc}:{number}: {match.group(0)!r} - the floor is {floor:g}"
                        )
            if f"{floor:g}" not in line and after.search(line):
                problems.append(f"{doc}:{number}: states a floor but not {floor:g}")
    assert f"{floor:g}" in _text("docs/RUN_SHEET.md"), "the run sheet stopped stating the recall floor"
    assert not problems, "stale recall floor in the docs:\n" + "\n".join(problems)


def test_the_split_ratio_in_the_docs_is_the_planners():
    """`70/20/10` is `plan_split.TARGET`, and the ratio is what the docs promise a capture buys.

    Only a triple of percentages that adds up to 100 counts as a ratio claim: the run sheet also
    carries split *image* counts (`863/400/243`) and frame sizes, and those are data, not policy.
    """
    expected = "/".join(f"{plan_split.TARGET[name] * 100:.0f}" for name in label_classes.SPLIT_NAMES)
    ratio = re.compile(r"(?<![\d.])(\d{1,3})/(\d{1,3})/(\d{1,3})(?![\d.])")
    problems = []
    for doc in DOCS:
        text = _text(doc)
        for match in ratio.finditer(text):
            values = [int(part) for part in match.groups()]
            if sum(values) != 100:
                continue
            if "/".join(str(value) for value in values) != expected:
                problems.append(
                    f"{doc}:{_line(doc, match.start())}: {match.group(0)!r} - the planner's split is "
                    f"{expected} over {list(label_classes.SPLIT_NAMES)}"
                )
    assert expected in _text("docs/RUN_SHEET.md"), (
        f"the run sheet does not state the {expected} split any more - it is the one the planner applies"
    )
    assert not problems, "stale split ratio in the docs:\n" + "\n".join(problems)


def test_the_crowding_margin_named_in_the_docs_is_the_acceptance_one():
    """`CROWD_MIN=2` is quoted verbatim in the run sheet's §11 table, because the whole point of the
    v2 acceptance number is "frames holding at least this many items". A doc quoting a different
    margin describes a measurement this repo does not make.
    """
    problems = []
    for doc in DOCS:
        text = _text(doc)
        for match in re.finditer(r"CROWD_MIN\s*=\s*(\d+)", text):
            if int(match.group(1)) != accept_v2.CROWD_MIN:
                problems.append(
                    f"{doc}:{_line(doc, match.start())}: CROWD_MIN={match.group(1)} - the code's is "
                    f"{accept_v2.CROWD_MIN}"
                )
    assert "CROWD_MIN=" in _text("docs/RUN_SHEET.md"), "the run sheet stopped naming CROWD_MIN"
    assert not problems, "stale crowding margin in the docs:\n" + "\n".join(problems)


# --------------------------------------------------------------------------
# 4. split names and artifact names
# --------------------------------------------------------------------------

# A path component that is meant to be a split directory. `val` is a name this project keeps bumping
# into - it is ultralytics' yaml key and the key `train_model.VALIDATION_SPLITS` uses - while the
# *directory* the dataset tools create is `valid`, so writing it into a path here sends a reader to a
# directory that no tool makes.
SPLIT_IN_PATH = re.compile(r"(?<![\w-])(?P<name>train|val|valid|test|validation|trainval|dev|eval)/")
# The pair shape, which the path scan cannot see the second half of: `train/val` writes the wrong
# split name without ever writing it before a slash.
SPLIT_PAIR = re.compile(r"(?<![\w-])(?P<a>train|valid|test)/(?P<b>val|valid|test|train)(?![\w-])")


def test_split_directories_in_the_docs_are_the_datasets_splits():
    """The split vocabulary, checked over the docs as a set: no doc may path into a split the dataset
    does not have, no pair may name one that is not a split, and between them the docs still name all
    three of them.

    `label_classes.SPLIT_NAMES` is the owner - `plan_split` assigns them, `build_dataset` creates
    them, `train_model` trains on them - so a doc written against a fourth spelling is a reader sent
    to an empty directory.
    """
    problems = []
    named: set[str] = set()
    for doc in DOCS:
        text = _text(doc)
        for match in SPLIT_IN_PATH.finditer(text):
            name = match.group("name")
            named.add(name)
            if name not in label_classes.SPLIT_NAMES:
                problems.append(
                    f"{doc}:{_line(doc, match.start())}: {name!r} is not one of "
                    f"{list(label_classes.SPLIT_NAMES)}"
                )
        for match in SPLIT_PAIR.finditer(text):
            for name in match.groups():
                if name not in label_classes.SPLIT_NAMES:
                    problems.append(
                        f"{doc}:{_line(doc, match.start())}: {match.group(0)!r} names {name!r}, "
                        f"which is not one of {list(label_classes.SPLIT_NAMES)}"
                    )
    missing = set(label_classes.SPLIT_NAMES) - named
    assert not missing, (
        f"no doc paths into {sorted(missing)} any more - the split vocabulary moved without the docs"
    )
    assert not problems, "split-name drift in the docs:\n" + "\n".join(problems)


# A backticked file name: the doc's way of saying "this artifact exists, at this path".
ARTIFACT_TOKEN = re.compile(
    r"`([A-Za-z0-9_./<>*-]+\.(?:json|yaml|yml|pt|onnx|csv|txt|db|log|jpg|png|md))`"
)

_PLACEHOLDER = re.compile(r"<[^>]*>|\*")


@lru_cache(maxsize=None)
def _source_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for pattern in SOURCE_GLOBS
        for path in sorted(REPO_ROOT.glob(pattern))
        if path.resolve() != SELF
    )


@lru_cache(maxsize=None)
def _repo_basenames() -> frozenset[str]:
    """Every file name in the repo, so a doc can name `requirements-inference.txt` the way its
    directory is written rather than the way a reader would type the path.
    """
    skip = {".git", ".venv", ".venv-inference", "node_modules", "__pycache__", "data", "out", "dist"}
    names = set()
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or skip & set(path.parts):
            continue
        names.add(path.name)
    return frozenset(names)


def _component_present(component: str, source: str) -> bool:
    """A whole component, so `models` cannot be satisfied by `models.py` inside another word."""
    return re.search(r"(?<![\w-])" + re.escape(component) + r"(?![\w-])", source) is not None


def _artifact_covers(token: str) -> bool:
    """Is this name something the repo actually has?

    Three ways to qualify, each the code's own answer rather than a copy of it: the path (or its
    basename, wherever it lives) exists on disk; `settings_store.is_custom_model` accepts it as a
    weight (the picker's own validator, so a documented `models/scanncart-grocery-v2.onnx` is legal
    even though the name is built from a template and never spelled literally); or every
    `/`-component of it is spelled somewhere in this repo's code - which is what catches a renamed
    artifact (`merge_report.json` after a rename to `merge-report.json`) and a renamed dataset
    directory (`merged-v2` -> `merged-v3`) in the mid-path positions a plain basename check misses.
    """
    if (REPO_ROOT / token).exists() or token.split("/")[-1] in _repo_basenames():
        return True
    if is_custom_model("models/" + token.split("/")[-1]):
        return True
    source = _source_text()
    components = [
        part for part in token.split("/") if part and not _PLACEHOLDER.search(part)
    ]
    return all(len(part) < 3 or _component_present(part, source) for part in components)


def test_artifact_names_in_the_docs_are_names_this_repo_has():
    """Every backticked file name in these docs has to be a name the repo has - on disk, accepted by
    the model-name validator, or spelled in the code that writes it.

    Artifact names are the quietest drift in the repo: they are spelled in prose and in strings, and
    a rename that misses a doc sends an operator to a file that is not there. `workspace.py` exists
    so the *code* has one spelling each; this is the same rule extended to the docs, where a rename
    cannot be found by a reader either.
    """
    problems = []
    seen = 0
    for doc in DOCS:
        text = _text(doc)
        for match in ARTIFACT_TOKEN.finditer(text):
            token = match.group(1)
            seen += 1
            if not _artifact_covers(token):
                problems.append(
                    f"{doc}:{_line(doc, match.start())}: {token!r} is not a file this repo has - "
                    "no path on disk, no model name the picker accepts, and no source that spells it"
                )
    assert seen > 40, f"the artifact scan found only {seen} names - did the quoting style change?"
    assert not problems, "artifact names in the docs the code does not have:\n" + "\n".join(problems)


# --------------------------------------------------------------------------
# 5. the contract tables: the size a doc states, and whether anything reads them
# --------------------------------------------------------------------------

# `thirteen surfaces in one table`, `13 surfaces in the table`, `a 13-surface table`. Two things
# keep ordinary prose out, both of them the clause the docs use for a table's size: the word `table`
# beside the count, and the plural - `the one surface an operator reads` (next to a table, even) is
# a single surface a rule is about, never a table's size, so it is not read as one. That is the same
# anchoring the class-count arm does on its count nouns.
# The number word is matched case-insensitively - a count opening a sentence is still a count - and
# read through `_number` below, which is the one place the spelling is turned into a value.
TABLE_NEAR = 40
SURFACE_COUNT = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<value>\d+|(?i:\b(?:" + WORDS + r")\b))\s+surfaces\b"
    r"|(?<![A-Za-z0-9_-])(?P<dash>\d+|(?i:\b(?:" + WORDS + r")\b))-surface\b"
)
# Which table a count is about: the *constant* the doc spells it as. That is why the size is read
# through the module rather than from a second list of table names here - a row added to the table
# moves this number with nothing to update on the guard's side, which is the point.
TABLE_REFERENCE = re.compile(r"`(?P<name>[A-Z][A-Z0-9_]+)`")
# A sentence, for attributing a count to the table it names. These docs wrap mid-sentence, so this
# breaks on whitespace rather than on line ends, and `settings.py` and `0.85` are not ends (a
# lower-case letter or a digit follows). A sentence opening on a backticked name is one.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*])")
# A count of *rows* has to be attached to the table it counts (`AGREEMENTS`'s two rows, two rows in
# `AGREEMENTS`), because unlike `surfaces` this noun is how these docs talk about every table they
# have - `one row per gate`, `the §3 rows`, `111 rows are that shape` - so a bare count of them
# cannot be told from prose, and a guard that tried would be turned off within a week.
ROW_NUMBER = r"(?P<value>\d+|(?i:\b(?:" + WORDS + r")\b)) rows?\b"
ROW_CLAIM = re.compile(
    r"`(?P<name>[A-Z][A-Z0-9_]+)`(?:'s| holds| has| lists) " + ROW_NUMBER +
    r"|(?P<value_after>\d+|(?i:\b(?:" + WORDS + r")\b)) rows? (?:in|of) "
    r"`(?P<name_after>[A-Z][A-Z0-9_]+)`"
)


def _number(raw: str) -> int:
    """One count as a value: a digit run as itself, a number word through the table above."""
    return int(raw) if raw.isdigit() else NUMBER_WORDS[raw.lower()]


@lru_cache(maxsize=None)
def _sentences(text: str) -> tuple[tuple[int, str], ...]:
    """`(offset, sentence)` pairs, so a count can be read against the table its own sentence names."""
    sentences = []
    start = 0
    for end in SENTENCE_BREAK.finditer(text):
        sentences.append((start, text[start : end.start()]))
        start = end.end()
    sentences.append((start, text[start:]))
    return tuple(sentences)


@lru_cache(maxsize=None)
def _tables() -> dict[str, object]:
    """The tables this suite declares, by the constant name the docs spell: each contract, and each
    tuple of them (`AGREEMENTS`, the table a doc counts rows of).

    Read from the module that owns them (`test_desktop_contracts.py`), and read as the objects
    rather than parsed out of its source: `len(table.surfaces)` is the fact, and a second list of
    names and numbers here would be one more copy to keep.
    """
    from tests import test_desktop_contracts as contracts
    from tests.wording import AgreementContract, WordingContract

    rows = (WordingContract, AgreementContract)
    tables: dict[str, object] = {}
    for name, value in vars(contracts).items():
        if isinstance(value, rows):
            tables[name] = value
        elif (
            isinstance(value, (tuple, list))
            and value
            and all(isinstance(row, rows) for row in value)
        ):
            tables[name] = value
    return tables


def _sizes(table: object) -> dict[str, int]:
    """What a doc may state about one table, by the noun it counts.

    A `WordingContract`'s rows *are* its surfaces, one file and its fragments each; an
    `AgreementContract` is a single row whose surfaces are the copies it compares; and a tuple of
    rows is the table a doc counts the rows of.
    """
    from tests.wording import AgreementContract

    if isinstance(table, (tuple, list)):
        return {"rows": len(table), "surfaces": sum(len(row.surfaces) for row in table)}
    if isinstance(table, AgreementContract):
        return {"rows": 1, "surfaces": len(table.surfaces)}
    return {"rows": len(table.surfaces), "surfaces": len(table.surfaces)}


def _table_size_verdict(doc: str, text: str, tables: dict[str, object]) -> tuple[int, list[str]]:
    """`(counts about a table, problems)` for one doc's text - the arm's whole body, so the canary
    below can feed it sentences rather than edit a real doc.
    """
    known = ", ".join(f"`{name}`" for name in sorted(tables)[:3])
    problems: list[str] = []
    seen = 0
    for start, sentence in _sentences(text):
        named = {name: tables[name] for name in TABLE_REFERENCE.findall(sentence) if name in tables}
        for match in SURFACE_COUNT.finditer(sentence):
            # What makes it a table's size rather than prose about one: the clause these docs use for
            # a count of a table's rows.
            near = text[
                max(0, start + match.start() - TABLE_NEAR) : start + match.end() + TABLE_NEAR
            ]
            if "table" not in near:
                continue
            seen += 1
            value = _number(match.group("value") or match.group("dash"))
            # The line number is counted from the text handed in rather than through `_line`, which
            # reads the file by name: the canary below passes sentences, not a doc on disk.
            where = f"{doc}:{text[: start + match.start()].count('\n') + 1}"
            if len(named) != 1:
                problems.append(
                    f"{where}: {match.group(0)!r} counts a table's surfaces but its sentence "
                    f"names {len(named)} table(s) of this suite, so nothing can check it against "
                    f"the code - name the one it counts (e.g. {known}), or reword it if it is not "
                    "a table's size"
                )
                continue
            name, table = next(iter(named.items()))
            surfaces = _sizes(table)["surfaces"]
            if value != surfaces:
                problems.append(f"{where}: {match.group(0)!r} - `{name}` has {surfaces} surfaces")
    # The row counts, which need no sentence to be attributed in: their table's name is attached.
    for match in ROW_CLAIM.finditer(text):
        seen += 1
        name = match.group("name") or match.group("name_after")
        value = _number(match.group("value") or match.group("value_after"))
        where = f"{doc}:{text[: match.start()].count('\n') + 1}"
        if name not in tables:
            problems.append(
                f"{where}: {match.group(0)!r} counts the rows of `{name}`, which is not a table "
                f"this suite declares - so nothing can check it (e.g. {known})"
            )
            continue
        rows = _sizes(tables[name])["rows"]
        if value != rows:
            problems.append(f"{where}: {match.group(0)!r} - `{name}` has {rows} row(s)")
    return seen, problems


def test_table_sizes_in_the_docs_match_the_tables_they_name():
    """A table's size stated in prose is `len(table.surfaces)` or `len(table)` written out, and it has
    one failure mode: a row added to the table, which leaves the sentence reading true while the
    table moved. A table's own count is the one thing no other guard in this suite looks at, since
    each of them checks the rows and never how many there are.

    The two nouns are anchored differently, because they are not equally tellable apart from prose.
    `surfaces` is rare enough that the word `table` beside the count identifies it, and then the
    count's own *sentence* has to name exactly one table of this suite - the constant the guard
    declares it as - because a number that names nothing is the shape this whole module exists to
    catch (and a count beside two tables fails too: the reader could not tell which number moved).
    `rows` is how these docs talk about every table they have, so that count has to be attached to
    the name it counts; an attached count naming something this suite does not declare fails, since
    nothing could check it.
    """
    tables = _tables()
    problems: list[str] = []
    seen = 0
    for doc in DOCS:
        found, found_problems = _table_size_verdict(doc, _text(doc), tables)
        seen += found
        problems += found_problems
    # A guard that stops matching anything passes silently, so the shape has to still be there - and
    # the count of what was *found* rather than of what was checkable, because a count that names no
    # table is a problem below and must be reported as one rather than absorbed here.
    assert seen >= 1, (
        "no doc states a contract table's size any more - if that is deliberate, this arm has "
        "nothing left to check, and the paragraph it was written for is what it should lose"
    )
    assert not problems, "table size(s) in the docs the code disagrees with:\n" + "\n".join(problems)


# Where a table's name can be referred to: the docs that describe the conventions, and the suite's
# own source - the checks that call it, the tuples that walk it, and the docstrings beside them. The
# app, the tools and the desktop are out because a contract table is a sidecar-test object (nothing
# outside `sidecar/tests/` can read one, and today nothing there names one).
REFEREE_GLOBS = ("sidecar/tests/**/*.py",)


@lru_cache(maxsize=None)
def _referee_texts() -> tuple[tuple[str, str], ...]:
    """`(relative path, text)` for everything a table's name can be referred to in, read once."""
    paths = [REPO_ROOT / doc for doc in DOCS]
    for pattern in REFEREE_GLOBS:
        paths += sorted(REPO_ROOT.glob(pattern))
    return tuple(
        (path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8", errors="replace"))
        for path in paths
        if path.is_file()
    )


def _mentions(name: str) -> list[str]:
    """Every place that names one table, other than the line that declares it.

    The declaration is excluded because a constant always spells itself there - which is exactly the
    sense in which a table can look maintained while nothing reads it. A call
    (`assert_fragments_present(NULL_CONTROL_CONTRACT)`), a walk over a tuple of rows, and a sentence
    in a doc all count, since all three are a reader or a check reaching for it.
    """
    pattern = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])")
    declaration = re.compile(r"\s*" + re.escape(name) + r"\s*(?::[^=]+)?=")
    places = []
    for relative, text in _referee_texts():
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) and not declaration.match(line):
                places.append(f"{relative}:{number}")
    return places


def test_every_declared_table_is_read_by_something():
    """No table may be declared and then referred to by nothing - not by a doc, not by a check.

    This is the failure the *other* arms cannot see, because every one of them reads the tables that
    exist: a `WordingContract` whose rows are all still phrased correctly, whose canary fires, and
    which no test ever calls is green everywhere in this file and dead in the suite. The rows being
    validated at construction is what makes it look wired up, which is the trap - so the check is
    `_mentions`, and its two remedies are the two ways a table is meant to be reached for: name it
    where the convention is written, or call it from the check it was declared for.
    """
    tables = _tables()
    assert tables, "the suite declares no contract tables any more - did `_tables()` stop scanning?"
    unread = [name for name in sorted(tables) if not _mentions(name)]
    assert not unread, (
        "these tables are declared and nothing refers to them - no doc explains them and no check "
        "calls them, so they read as maintained while nothing can fail on them (the scan reads "
        f"{', '.join(REFEREE_GLOBS)} and the docs). Name each where its convention is written, "
        "wire it into the check it was declared for, or drop it:\n  " + "\n  ".join(unread)
    )


def test_the_table_reference_scan_knows_a_name_from_a_name_it_cannot_find():
    """Both ends of that scan: names the corpus has, and a name it cannot have.        Without this the guard's own failure mode is silent - a moved glob or a renamed doc leaves every
        table looking unread at once (loud, and the message says so), but a corpus that matched a single
        stray word would keep it green while it checked nothing.
    """
    assert _mentions("AGREEMENTS"), "the reference scan cannot see a table named in a doc and a test"
    assert _mentions("NULL_CONTROL_CONTRACT"), "the scan cannot see a table a check calls"
    # A name the corpus cannot have, written in two pieces on purpose: this file is scanned too, so
    # spelling it whole here is how it would be found.
    undeclared = "A_TABLE_NOBODY_" + "EVER_DECLARED"
    assert _mentions(undeclared) == [], "the scan reported a name the suite does not declare"
    # And it matches a whole name rather than a piece of one: the declared `NULL_CONTROL_CONTRACT`
    # cut short at its last underscore is not a name, and must find nothing. Built in two pieces too.
    partial = "NULL_CONTROL" + "_"
    assert _mentions(partial) == [], "the scan is matching part of a name as a name"


def _fixture_table(rows: int):
    """A table of `rows` rows, for the canary - the smallest thing that has a size."""
    from tests.wording import WordingContract

    return WordingContract(
        name="the canary's own table",
        owner="tests/test_docs_claims.py",
        claim="a surface of the canary's table lost its wording.",
        facts="It is a fixture, stated so the engine accepts it.",
        surfaces=tuple(("sidecar/tests/CLAUDE.md", ("a fixture fragment",)) for _ in range(rows)),
        canary_fragments=("a fixture fragment",),
    )


def test_the_table_size_arm_fires_on_every_way_a_stated_size_can_be_wrong():
    """The arm's own canary, because a guard whose only proof is that it passes is a guard nobody
    knows the teeth of - and the real docs cannot be broken on purpose in the suite.

    One contract of ten rows, a second of four, and a table of two of them: the count that agrees,
    the count that is one row behind the table, a count that names no table, a count naming two, and
    the row form - in both spellings, and for a table of rows and a single row's spans.
    """
    tables = {
        "FIXTURE_CONTRACT": _fixture_table(10),
        "OTHER_CONTRACT": _fixture_table(4),
        "FIXTURE_TABLE": (_fixture_table(3), _fixture_table(3)),
    }

    def verdict(text: str):
        return _table_size_verdict("docs/MODEL_TRAINING.md", text, tables)

    # The one that holds: the sentence is a table's size, it names that table, and it equals it.
    assert verdict("The table `FIXTURE_CONTRACT` holds ten surfaces in one table.") == (1, [])
    # The failure this arm exists for: a row added to the table and the sentence left reading true.
    seen, problems = verdict("`FIXTURE_CONTRACT` is a table of eleven surfaces, per the guard.")
    assert seen == 1 and len(problems) == 1
    assert "`FIXTURE_CONTRACT` has 10 surfaces" in problems[0], problems
    # A count about a table that names none of this suite's tables: nothing to check it against.
    seen, problems = verdict("The same table holds thirteen surfaces in one table.")
    assert seen == 1 and len(problems) == 1
    assert "names 0 table(s)" in problems[0], problems
    # A count beside two tables: checkable, but not attributable to one of them.
    seen, problems = verdict(
        "`FIXTURE_CONTRACT` and `OTHER_CONTRACT` are tables of four surfaces each."
    )
    assert seen == 1 and len(problems) == 1
    assert "names 2 table(s)" in problems[0], problems
    # And the singular is prose, not a size - even with the word `table` right beside it, which is
    # what keeps `the one surface an operator reads` out of this arm.
    assert verdict("The table's one surface is the one an operator reads.") == (0, [])

    # The row form, which is anchored by the name instead: both spellings, and the table it counts
    # is the tuple of rows rather than one row of a contract.
    assert verdict("`FIXTURE_TABLE` holds two rows, one per door.") == (1, [])
    assert verdict("Two rows in `FIXTURE_TABLE` are the claims.") == (1, [])
    # One row added to the table, with the sentence left reading true - the same failure as above.
    seen, problems = verdict("`FIXTURE_TABLE`'s three rows are the claims.")
    assert seen == 1 and len(problems) == 1
    assert "`FIXTURE_TABLE` has 2 row(s)" in problems[0], problems
    # A row count is a table's size too, so the attribution is checked the same way - and a
    # contract's own row count is its surfaces, not the rows of the tuple it would be in.
    seen, problems = verdict("`NOT_A_TABLE` has two rows.")
    assert seen == 1 and len(problems) == 1
    assert "is not a table this suite declares" in problems[0], problems
    seen, problems = verdict("`FIXTURE_CONTRACT` has one row.")
    assert seen == 1 and len(problems) == 1
    assert "`FIXTURE_CONTRACT` has 10 row(s)" in problems[0], problems
    # Prose about rows in general is not a count of any table's rows, so it is not read at all.
    assert verdict("One row per gate, and the table below says which.") == (0, [])


# --------------------------------------------------------------------------
# 6. the fixture the docs describe
# --------------------------------------------------------------------------

DRIVER = ".claude/skills/run-desktop/driver.mjs"


def test_the_classlist_fixture_builds_its_head_from_the_roster_and_the_axis():
    """The docs describe §12's scratch weight as one output per product *and distance*, and the
    driver is what builds it - so the claim is checked in the place that makes it true: the fixture
    derives its product list from the generation spec, and the distances it multiplies by are
    `label_classes.DISTANCE_ORDER`.

    This is where the eight-class round actually started. The driver hard-coded eight products
    (Palmolive included), so it built an 24-output head long after the class was dropped, and the
    docs describing what it prints (`24 classes`, `24 of 24`) were *accurate* about a fixture that
    had stopped being the thing they claimed to verify. Correcting the prose alone would have left
    the next session rebuilding the same eight-product weight.
    """
    text = (REPO_ROOT / DRIVER).read_text(encoding="utf-8")
    assert "PRODUCTS = list(generations.V2.classes)" in text, (
        "the scratch fixture no longer takes its product list from the generation spec"
    )
    per_distance = re.search(r'names = \[f"\{p\} \{d\}" for p in PRODUCTS for d in \(([^)]*)\)\]', text)
    assert per_distance, "the scratch fixture no longer builds one class per product and distance"
    axis = tuple(re.findall(r'"([^"]*)"', per_distance.group(1)))
    assert axis == tuple(label_classes.DISTANCE_ORDER), (
        f"the scratch fixture multiplies the roster by {list(axis)}, not {list(label_classes.DISTANCE_ORDER)}"
    )


# --------------------------------------------------------------------------
# 7. the exemptions, in one place
# --------------------------------------------------------------------------

# The arms of this guard, by the name a row uses to say which scan step it is excused from: a name
# rather than the test function, so the table stays data and can be read without the arms' bodies,
# while every arm still has to pass its own name to `_exempt` - which is the only way in.
ARM_CLASS_COUNTS = "class counts"


@dataclasses.dataclass(frozen=True)
class Arm:
    """One arm of this guard: the test that runs its scan, and the patterns it scans the docs with.

    `test` is the function's name in this module rather than the function itself, so the table stays
    data - and so the check below can read that function's body and hold these patterns to it.
    """

    test: str
    patterns: tuple[re.Pattern[str], ...]


# What each arm scans with, so an exemption can be proved *load-bearing* and attributed to the exact
# shape it was written for: a row whose phrase no longer falls inside a claim *that pattern* would flag
# has outlived the sentence it was written for. A map that had drifted from the arm it describes would
# make that proof meaningless, so it is read off the arm rather than written down beside it - the
# patterns are held to `Arm.test`'s own body by the check at the end of this section.
ARMS: dict[str, Arm] = {
    ARM_CLASS_COUNTS: Arm(
        test="test_every_class_count_in_the_docs_is_one_the_code_can_produce",
        patterns=(COUNT_CLAIM, SUBSET_CLAIM, ROSTER_PAIR),
    ),
}

# The name each of this module's patterns is known by, read off the module rather than written down:
# a shape names a pattern's own identifier, so renaming one moves the shape with it.
PATTERN_NAMES: dict[re.Pattern[str], str] = {
    value: name for name, value in globals().items() if isinstance(value, re.Pattern)
}


def _shape(arm: str, pattern: re.Pattern[str]) -> str:
    """One claim shape of one arm, as a row names it: the arm and the pattern its scan matched on.

    This is the granularity a row is written at, and it is what keeps an exemption from spreading
    *within* an arm: a phrase that excuses a count cannot excuse the same words under another of that
    arm's patterns - the subset shape and the roster-pair shape being the two this arm also has.
    """
    return f"{arm} / {PATTERN_NAMES[pattern]}"


# Every shape this guard's arms produce, by the name a row writes - what a row's `shape` has to be one
# of, and what the liveness check below asks each row's own text through.
SHAPES: dict[str, tuple[str, re.Pattern[str]]] = {
    _shape(arm, pattern): (arm, pattern)
    for arm, spec in ARMS.items()
    for pattern in spec.patterns
}

# The row - phrase, reason, places, shape - is `tests/exemptions.py`'s, shared with the two other
# guards that keep such a table, and so is the rule that keeps it live (`assert_still_present`, called
# with this module's own reader). What is this module's: the phrase is the row's identity, so a phrase
# and the reason it is allowed travel together and the same phrase excused for the same reason in two
# docs is one row rather than two; and every row names a *shape* rather than an arm, because this arm
# matches three of them - a phrase written for one is asked for by that one alone, so adding one cannot
# leave another pattern of the same arm covering text it was not written for.
EXEMPT = Exemptions(
    Exemption(
        phrase="`0 classes` would be a claim",
        reason="a UI value the docs say must never appear, not a roster size",
        shape=_shape(ARM_CLASS_COUNTS, COUNT_CLAIM),
        places=("docs/MODEL_TRAINING.md", "desktop/CLAUDE.md"),
    ),
    Exemption(
        phrase="`note: 1 class(es)",
        reason="the tool's own diagnostic, whose count varies with the dataset rather than "
        "describing a roster",
        shape=_shape(ARM_CLASS_COUNTS, COUNT_CLAIM),
        places=("docs/MODEL_TRAINING.md",),
    ),
    Exemption(
        phrase="the 8-class roster",
        reason="history: the constant that was in the trainer before `generations.py`, described "
        "in the past tense",
        shape=_shape(ARM_CLASS_COUNTS, COUNT_CLAIM),
        places=("docs/MODEL_TRAINING.md",),
    ),
)


def _exempt(
    arm: str,
    pattern: re.Pattern[str],
    doc: str,
    snippet: str,
    waived: list[Waiver] | None = None,
) -> str | None:
    """The reason this snippet is not a claim *for the shape this pattern matched*, if a row excuses it.

    The pattern is a required argument and not only the arm, so a phrase written for one of an arm's
    claim shapes cannot silence another: an exemption is a fact about one shape, and the shape is the
    arm *and* the pattern. The shared lookup does the matching; this module adds the one rule only it
    can make - the arm has to be one of the arms it declared above, an arm that never declared itself
    reading as the same failure as a row it forgot.
    """
    assert arm in ARMS, (
        f"{arm!r} asks `_exempt` without being declared - the declared arms are {sorted(ARMS)}"
    )
    return EXEMPT.reason(_shape(arm, pattern), doc, snippet, waived)


def _functions() -> dict[str, ast.AST]:
    """This module's top-level functions by name, for the checks that read one arm's own body."""
    return {
        node.name: node
        for node in ast.parse(SELF.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _exempt_calls() -> list[tuple[str, ast.Call]]:
    """`(function, call)` for every `_exempt(...)` call in this file, read with `ast` rather than as
    text: this file explains the table in prose, so `_exempt` appears in docstrings and comments it
    must not be read from. The enclosing function comes along so a call site can be held to the arm
    whose test it sits in.
    """
    calls = []
    for node in ast.parse(SELF.read_text(encoding="utf-8")).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_exempt"
            ):
                calls.append((node.name, child))
    return calls


def _patterns_reached(
    node: ast.AST, functions: dict[str, ast.AST], through_calls: bool = True
) -> set[re.Pattern[str]]:
    """The compiled patterns one piece of code reaches for: the module's pattern names it uses, and the
    same one level into the functions it calls, because an arm scans through its helpers as often as
    in its own lines - what the table describes is a scan, not a body.
    """
    reached = {
        globals()[child.id]
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and isinstance(globals().get(child.id), re.Pattern)
    }
    if through_calls:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in functions
                and functions[child.func.id] is not node
            ):
                reached |= _patterns_reached(functions[child.func.id], functions, through_calls=False)
    return reached


def test_the_docs_guard_exemptions_are_still_live():
    """An exemption is a claim too, and it is the one that could rot - so both ends of it, and both
    ends of the *scope*: the phrase is still in every doc the row names (the shared rule, in
    `tests/exemptions.py`), the row still names a shape this guard's arms produce, that shape still
    flags a claim this phrase is what excuses (otherwise the row is one nobody needs), and the text it
    excuses is excused only for that shape.

    The last end is the one this shape exists for: an exemption written for one claim shape cannot make
    another skip text it was never written about, and that now includes another *pattern* of the same
    arm - which is what a row naming only its arm used to allow, silently, since every pattern of the
    arm matched the same row.
    """
    assert_still_present(EXEMPT, _text)
    for row in EXEMPT.rows:
        assert row.shape in SHAPES, (
            f"the exemption for {row.phrase!r} names {row.shape!r}, which is not a claim shape of "
            f"this guard ({sorted(SHAPES)}) - a row no scan asks with excuses nothing"
        )
        arm, pattern = SHAPES[row.shape]
        for doc in row.places:
            assert doc in DOCS, f"the exemption for {row.phrase!r} names {doc!r}, which no scan reads"
            excused = [
                match.group(0)
                for match in pattern.finditer(_text(doc))
                if _exempt(arm, pattern, doc, _snippet(doc, match.start(), match.end()))
            ]
            assert excused, (
                f"the exemption for {row.phrase!r} in {doc} excuses nothing any more: the "
                f"{row.shape!r} shape flags no claim whose sentence carries that phrase, so the row "
                "is one nobody needs - drop it, or re-point it at the claim that replaced the one it "
                "was written for"
            )
            # Scoped to its own shape, asked as every other shape this guard knows: the very text it
            # excuses is not excused for any of them. Every other shape rather than a stand-in, because
            # the two the class-count arm also has are exactly the ones a phrase written for this one
            # could silence without this.
            for other, (other_arm, other_pattern) in SHAPES.items():
                if other == row.shape:
                    continue
                assert _exempt(other_arm, other_pattern, doc, row.phrase) is None, (
                    f"the exemption for {row.phrase!r} silences the {other!r} shape, which it does "
                    "not name"
                )
    # An entry is a phrase, not a topic: the trainer's own diagnostic is excused where the same
    # count written as a roster readout is not.
    assert _exempt(ARM_CLASS_COUNTS, COUNT_CLAIM, "docs/MODEL_TRAINING.md", "see `note: 1 class(es)`")
    assert (
        _exempt(ARM_CLASS_COUNTS, COUNT_CLAIM, "docs/MODEL_TRAINING.md", "prints `1 classes`") is None
    )


def test_each_arm_scans_with_exactly_the_patterns_the_table_lists():
    """`ARMS` says which test runs each arm and what it scans with, and this reads the test to check
    the second half.

    That map is what an exemption's *load-bearing* end is proved against, so one that had drifted from
    the arm it describes would make that proof meaningless: it would show a phrase sitting inside a
    pattern the arm no longer uses, or miss the pattern it does. Read with `ast` from the arm's own
    body, one level into the helpers it calls, so a pattern added to an arm - or dropped from it -
    fails here instead of leaving the table quietly wrong. An arm that defines a pattern inside itself
    is one this table cannot describe at all, and it fails here too rather than reading as covered.
    """
    functions = _functions()
    by_pattern = {value: name for name, value in globals().items() if isinstance(value, re.Pattern)}
    for name, arm in ARMS.items():
        assert arm.test in functions, (
            f"{name!r} names {arm.test!r} as the test that runs it, and this module has no such "
            "function"
        )
        reached = _patterns_reached(functions[arm.test], functions)
        assert reached == set(arm.patterns), (
            f"{name!r} does not scan with the patterns ARMS lists: `{arm.test}` reaches for "
            f"{sorted(by_pattern.get(pattern, repr(pattern)) for pattern in reached)} while the "
            f"table declares {sorted(by_pattern.get(pattern, repr(pattern)) for pattern in arm.patterns)}"
            " - the two are updated together, and a pattern the test defines inside itself is not "
            "something this table can describe"
        )


def test_every_pattern_of_an_arm_has_its_own_shape():
    """One shape per (arm, pattern), which is what a row is written against.

    A spelling that dropped the pattern would put every pattern of an arm back onto one row - the exact
    spreading this arrangement exists to prevent - and it would stay green everywhere else, because the
    row would still be load-bearing for the arm. The count per arm is what makes the spelling checkable,
    and the arm in the shape is what keeps two arms' shapes apart.
    """
    for arm, spec in ARMS.items():
        shapes = sorted(shape for shape, (row_arm, _) in SHAPES.items() if row_arm == arm)
        assert len(shapes) == len(spec.patterns), (
            f"{arm!r} has {len(spec.patterns)} pattern(s) and {len(shapes)} shape(s) ({shapes}) - a "
            "shape names one arm and one pattern, so a phrase written for one of them cannot silence "
            "another"
        )
        assert all(SHAPES[shape][1] in spec.patterns for shape in shapes), (
            f"{arm!r}'s shapes are {shapes}, which do not resolve back to its own patterns"
        )


def test_every_arm_that_excuses_a_snippet_is_declared_with_the_patterns_it_scans():
    """The table and the arms that use it, held to one place: every arm asks `_exempt` from its own test
    under its own name, and the names asked that way are exactly the declared arms (the rows' shapes
    being a subset of what those arms produce, which each row above is asserted against).

    Only the arms' own tests are read here. The table's own checks ask the lookup too - as another
    shape, on purpose - and what they pass is theirs to choose. Without this the two drift in silence:
    an arm excusing snippets against a row written for another shape, a declared arm nothing asks with,
    or a call site inside one arm passing another arm's constant, which would excuse text for a scan
    the row was not written for.
    """
    by_test = {arm.test: name for name, arm in ARMS.items()}
    calls = [(function, call) for function, call in _exempt_calls() if function in by_test]
    assert calls, "no arm asks `_exempt` any more - did the call shape change?"
    for function, call in calls:
        assert call.args and isinstance(call.args[0], ast.Name), (
            f"`{function}` asks `_exempt` with something other than an arm - the arm has to be one of "
            "the constants this module declares, so it can be held to the table"
        )
        assert call.args[0].id in globals(), (
            f"`{function}` asks `_exempt` with {call.args[0].id!r}, which this module does not define"
        )
    # Through `globals()`, because what the call sites pass is the constants and what the map is keyed
    # by is what they hold.
    asked = {globals()[call.args[0].id] for _, call in calls}
    assert asked == set(ARMS), (
        f"the arms that ask are {sorted(asked)} while ARMS declares {sorted(ARMS)}: an arm that asks "
        "without being declared, or a declared arm nothing asks with, is the second place this would "
        "have to be kept in step"
    )
    wrong = {
        function: globals()[call.args[0].id]
        for function, call in calls
        if globals()[call.args[0].id] != by_test[function]
    }
    assert not wrong, (
        f"{wrong} - an arm asks under its own name, so a call site inside one arm passing another "
        "arm's constant would excuse text for a scan the row was not written for"
    )
