"""Drift guards: the claims the docs make about the code, checked against the code.

The three docs this repo runs from (`CLAUDE.md`, `docs/RUN_SHEET.md`, `docs/MODEL_TRAINING.md`) plus
the run-desktop skill (`SKILL.md`, which describes the §12 verification procedure the other two point
at) are prose, and most of what they say cannot be checked at all. This module deliberately does not
try. What it pins is the four kinds of sentence that are really a *copy* of a code fact, because
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

Everything asserted here is a fact about the *code*, never about the workspace: the live dataset
counts the run sheet also carries (`858/1433 decided`, `132 of 186 across 6 cells`) move with the
data and are deliberately not pinned - a guard on those would fail on a labeling session, which is
how a guard stops being read. Where a doc number looks like a claim but is not one (a diagnostic's
own count, a UI value the docs say must never appear), the exemption lives in `EXEMPT` below with
its reason, and `test_the_docs_guard_exemptions_are_still_live` fails when the phrase it covers is
edited away - so an exemption cannot quietly cover new text.

One more check sits beside those four: the §12 fixture the docs describe (`driver.mjs classlist`) has
to keep being the thing they say it is - a head sized by the roster x the distance axis. It is the
root of this round's stale counts, because prose describing a fixture nothing builds stays accurate
right up until the fixture stops being that.

Same shape as the other drift guards in this suite, and the same failure mode it exists for: a stale
doc reads as a healthy project, and prose is where a rename goes to be forgotten.
"""

from __future__ import annotations

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
}

# (doc, snippet, reason). The snippet must still be present - `test_the_docs_guard_exemptions_are_still_live`
# holds that end - so an exemption cannot outlive the sentence it was written for and start covering
# whatever replaces it.
EXEMPT: tuple[tuple[str, str, str], ...] = (
    (
        "docs/MODEL_TRAINING.md",
        "`0 classes` would be a claim",
        "a UI value the docs say must never appear, not a roster size",
    ),
    (
        "CLAUDE.md",
        "`0 classes` would be a claim",
        "the same UI value, stated in the Live view's paragraph",
    ),
    (
        "docs/MODEL_TRAINING.md",
        "`note: 1 class(es)",
        "the tool's own diagnostic, whose count varies with the dataset rather than describing a roster",
    ),
    (
        "docs/MODEL_TRAINING.md",
        "the 8-class roster",
        "history: the constant that was in the trainer before `generations.py`, described in the past tense",
    ),
)


@lru_cache(maxsize=None)
def _text(doc: str) -> str:
    return (REPO_ROOT / doc).read_text(encoding="utf-8")


def _line(doc: str, index: int) -> int:
    return _text(doc)[:index].count("\n") + 1


def _snippet(doc: str, start: int, end: int, pad: int = 30) -> str:
    text = _text(doc)
    return text[max(0, start - pad) : end + pad]


def _exempt(doc: str, snippet: str) -> str | None:
    """The reason this snippet is not a claim, if any exemption covers it."""
    for exempt_doc, phrase, reason in EXEMPT:
        if exempt_doc == doc and phrase in snippet:
            return reason
    return None


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
    """
    problems = []
    checked = 0
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
            if _exempt(doc, snippet):
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
    assert not problems, "stale class count(s) in the docs:\n" + "\n".join(problems)


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
# 5. the fixture the docs describe
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
# 6. the exemptions themselves
# --------------------------------------------------------------------------


def test_the_docs_guard_exemptions_are_still_live():
    """An exemption is a claim too, and it is the one that could rot: the phrase it was written for
    disappearing leaves an exemption covering text nobody read. Both ends, like every other guard
    here - the phrase is still there, and it is still the number the exemption describes.
    """
    for doc, phrase, reason in EXEMPT:
        assert phrase in _text(doc), (
            f"{doc} no longer says {phrase!r}, so the exemption ({reason}) covers nothing - "
            "drop it, or re-point it at the sentence that replaced this one"
        )
    # And the class-count exemption is bounded to what it claims: `1 class(es)` as a diagnostic's
    # own output, never a roster readout like `1 classes`.
    assert _exempt("docs/MODEL_TRAINING.md", "see `note: 1 class(es)`") is not None
