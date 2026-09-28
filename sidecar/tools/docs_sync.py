#!/usr/bin/env python
"""Rewrite the numbers the docs state from the code that owns them.

Three documents carry numbers that are really copies of code facts - the run sheet's checkpoint
values, the training doc's tables and `--val` guidance, and the run-desktop skill's §12 expectations:

    docs/RUN_SHEET.md, docs/MODEL_TRAINING.md, .claude/skills/run-desktop/SKILL.md

`test_docs_claims.py` *checks* those numbers against the code, which is how this round found the
eight-class claims that had survived a dataset change. A check is a good net and a poor guarantee:
it tells the next person that a sentence went stale, and leaves the repair to them, in prose, by
hand, after a failing test. This tool is the repair - it owns the *sites* where a number is a fact
and rewrites each one from its owner, so `make docs-sync` after a deliberate change is the whole
maintenance step, and the numbers cannot drift in the first place.

What is a site, and what is not
-------------------------------

A site is a sentence whose value is *determined* by the code: the phrase around the number fixes
which fact it reports. `CROWD_MIN=2` names the constant; `` | `conf_threshold` (default `0.5`) | ``
names the field; `the 7-name roster` and `` `… — 21 classes` with ⚠ `21 of 21 class name(s) …` ``
name the roster and the per-product-and-distance head respectively. Each site carries the group (or
groups) it owns, so the rewrite is a splice of those spans and never a rebuild of the sentence -
the prose around them is the author's and stays byte-identical.

That is also the boundary. A class count in ordinary prose - `one class v2 added`, `100 per class`,
`v1's six raw capture archives` - is not a claim about a roster, and this tool has nothing to say
about it. Neither has it anything to say about measured values (`0.918` recall on v1's test split)
or about counts of things that are not code constants (13 `PASS` lines in a driver script). Those
are left to the reader and, where they are shape claims at all, to `test_docs_claims.py`. The two
are complements: the sync owns what it can render, the guard catches what it cannot.

Fail-closed, like the rest of the toolchain: a site that matches nothing in its doc is an *error*,
not a silent skip. A site is a sentence someone decided the code owns, and a rewording that drops it
means the doc no longer says what the site claims it says - which is exactly the case where writing
the rest of the file and reporting success would be a lie.

Usage::

    cd sidecar && .venv/Scripts/python.exe tools/docs_sync.py --check   # exit 1 with a diff
    cd sidecar && .venv/Scripts/python.exe tools/docs_sync.py --write   # rewrite, print a summary

Exit codes: 0 clean (or written), 1 for a stale or unmatched site under `--check`, and 2 under
`--write` when a site did not match anything or when a document's line endings are mixed - in both
cases nothing is written at all. A half-synced file is worse than a stale one, and a doc whose
newlines cannot be put back is not this tool's to rewrite.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# The tools live on `pythonpath` in one context (pytest) and are run as scripts in the other, so the
# runtime half of the import below (`app.settings`) needs the sidecar root put there explicitly.
# Both halves are needed: the numbers are the app's defaults and the dataset tooling's constants.
SIDECAR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SIDECAR_ROOT.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

import accept_v2  # noqa: E402  (path set above)
import label_classes  # noqa: E402
import plan_split  # noqa: E402
import train_model  # noqa: E402
from app.roster import V1_ROSTER, V2_ROSTER  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.settings_store import HOT_RELOADABLE_FIELDS, RESTART_REQUIRED_FIELDS  # noqa: E402

RUN_SHEET = "docs/RUN_SHEET.md"
MODEL_TRAINING = "docs/MODEL_TRAINING.md"
SKILL = ".claude/skills/run-desktop/SKILL.md"

NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def _word(count: int) -> str:
    return NUMBER_WORDS.get(count, str(count))


def _roster(generation: str = "v2") -> int:
    """A generation's roster size. v2 by default: it is the generation the docs under construction
    are about, and the two are the same seven names today."""
    return len(V1_ROSTER) if generation == "v1" else len(V2_ROSTER)


def _head(generation: str = "v2") -> int:
    """The width of a head trained one output per product *and* distance - the failure the roster
    guard exists for, and the number a distance-split export bakes in."""
    return _roster(generation) * len(label_classes.DISTANCE_ORDER)


def _floor() -> str:
    return f"{train_model.RECALL_FLOOR:g}"


def _ratio() -> str:
    return "/".join(f"{plan_split.TARGET[name] * 100:.0f}" for name in label_classes.SPLIT_NAMES)


def _augmentation() -> str:
    return ", ".join(f"{key}={value:g}" for key, value in train_model.augmentation_kwargs().items())


SETTINGS_FIELDS = {field.name for field in dataclasses.fields(Settings)}


def _default(field: str) -> str:
    return str(getattr(Settings(), field))


def _reload_word(field: str) -> str:
    """Which kind of setting this is, in the two words the tuning table uses. Read from the same two
    sets the API serves as `hot_reloadable_fields`/`restart_required_fields`, so the doc cannot claim
    a restart for a field `PATCH` applies live."""
    return "Hot-reloadable" if field in HOT_RELOADABLE_FIELDS else "Restart-required"


@dataclass(frozen=True)
class Site:
    """One sentence whose number is a code fact.

    `pattern` must name every span this tool owns as a group; `values` answers, per match, what each
    of those spans should say. A group the callable does not return is context and is never touched.
    """

    name: str
    doc: str
    pattern: re.Pattern[str]
    values: Callable[[re.Match[str]], dict[str, str]]
    why: str

    def edits(self, text: str) -> tuple[list[tuple[int, int, str, str]], bool]:
        """(stale spans as (start, end, current, wanted), whether the site matched at all)."""
        found: list[tuple[int, int, str, str]] = []
        matched = False
        for match in self.pattern.finditer(text):
            matched = True
            for group, wanted in self.values(match).items():
                current = match.group(group)
                if current != wanted:
                    found.append((match.start(group), match.end(group), current, wanted))
        return found, matched


def _tuning_row(match: re.Match[str]) -> dict[str, str]:
    """The tuning table's two code-owned spans: the default and the reload word.

    A row for something that is not a `Settings` field is left alone: the pattern matches any row
    shaped "field (default value)", and a value the code does not hold is not this tool's to
    rewrite.
    """
    field = match.group("field")
    if field not in SETTINGS_FIELDS:
        return {}
    return {"default": _default(field), "reload": _reload_word(field)}


def _site(
    name: str,
    doc: str,
    pattern: str,
    values: Callable[[re.Match[str]], dict[str, str]],
    why: str,
) -> Site:
    return Site(name, doc, re.compile(pattern, re.MULTILINE), values, why)


# The docs this tool owns, in the order a reader meets them. The tests walk this list, so a site
# added for a fourth doc cannot be left out of the round-trip coverage.
SITES_BY_DOC: tuple[str, ...] = (RUN_SHEET, MODEL_TRAINING, SKILL)


# --------------------------------------------------------------------------
# The sites. One entry per sentence whose number the code determines.
# --------------------------------------------------------------------------

SITES: tuple[Site, ...] = (
    # ---- docs/RUN_SHEET.md -------------------------------------------------
    _site(
        "sanity class list",
        RUN_SHEET,
        r"`\[\.ok\.\] class list … (?P<count>\d+) class\(es\) defined`",
        lambda match: {"count": str(_roster("v2"))},
        "`clean_v2.py sanity` prints the count of classes the live project declares, and the "
        "project is v2's",
    ),
    _site(
        "the planner's split",
        RUN_SHEET,
        r"`plan_split` decides (?P<ratio>\d+/\d+/\d+) over the batches",
        lambda match: {"ratio": _ratio()},
        "`plan_split.TARGET` is the split the planner applies",
    ),
    _site(
        "the printed augmentation table",
        RUN_SHEET,
        r"`augmentation: (?P<table>[^`]+)`",
        lambda match: {"table": _augmentation()},
        "`train_model.augmentation_kwargs` is what the run prints and records",
    ),
    _site(
        "the pass condition",
        RUN_SHEET,
        r"at or above the (?P<floor>\d\.\d+) floor",
        lambda match: {"floor": _floor()},
        "`train_model.RECALL_FLOOR` is what `--val` fails a class against",
    ),
    _site(
        "the Test connection reading",
        RUN_SHEET,
        r"`(?P<roster>\d+) classes`, and no class warning\. A distance-split weight would read "
        r"`(?P<head>\d+) classes`",
        lambda match: {"roster": str(_roster("v1")), "head": str(_head("v1"))},
        "the shipped weight is v1's seven, and a distance-split head of it is one output per "
        "product and distance",
    ),
    _site(
        "the acceptance floor table",
        RUN_SHEET,
        r"either weight below (?P<floor>\d\.\d+) on any class",
        lambda match: {"floor": _floor()},
        "the same floor, quoted in §11's table",
    ),
    _site(
        "CROWD_MIN",
        RUN_SHEET,
        r"`CROWD_MIN=(?P<margin>\d+)`",
        lambda match: {"margin": str(accept_v2.CROWD_MIN)},
        "`accept_v2.CROWD_MIN` is the crowding the acceptance number is about",
    ),
    _site(
        "the §12 classlist expectations",
        RUN_SHEET,
        r"`… — (?P<admin>\d+) classes` with ⚠ `(?P<names>\d+) of (?P<total>\d+) class name\(s\)",
        lambda match: {
            "admin": str(_head("v2")),
            "names": str(_head("v2")),
            "total": str(_head("v2")),
        },
        "the §12 fixture is built one class per product and distance, from v2's roster",
    ),
    _site(
        "the §12 chip reading",
        RUN_SHEET,
        r"`live-class-warnings`, and `(?P<chip>\d+)` · `classes · 1 finding`",
        lambda match: {"chip": str(_head("v2"))},
        "the Live chip shows the running model's class count, which for this fixture is the head",
    ),
    _site(
        "the §12 fixture shape",
        RUN_SHEET,
        r"nc=(?P<nc>\d+)\)` \(one head output per product and distance",
        lambda match: {"nc": str(_head("v2"))},
        "`DetectionModel(..., nc=N)` is built from the roster and the distance axis",
    ),
    _site(
        "the measured floor line",
        RUN_SHEET,
        r"\| measured \| every class ≥ (?P<floor>\d\.\d+) \*\*and\*\*",
        lambda match: {"floor": _floor()},
        "§12's last table quotes the same floor",
    ),
    # ---- docs/MODEL_TRAINING.md --------------------------------------------
    _site(
        "the capture defaults",
        MODEL_TRAINING,
        r"`(?P<width_field>capture_width)=(?P<width>\d+)`, "
        r"`(?P<height_field>capture_height)=(?P<height>\d+)`, "
        r"`(?P<fps_field>capture_fps)=(?P<fps>\d+)`",
        lambda match: {
            "width": _default("capture_width"),
            "height": _default("capture_height"),
            "fps": _default("capture_fps"),
        },
        "the sentence says these are the defaults in `sidecar/app/settings.py`, so they are "
        "`Settings()`'s",
    ),
    _site(
        "the class count in the volume target",
        MODEL_TRAINING,
        r"\*\* across all (?P<count>\d+) classes",
        lambda match: {"count": str(_roster())},
        "the target is per class, over the roster §8.1 defines",
    ),
    _site(
        "the class count in the backbone advice",
        MODEL_TRAINING,
        r"With only (?P<count>\d+) classes and a few thousand images",
        lambda match: {"count": str(_roster())},
        "the same roster, quoted once more on the way to choosing `yolo11s`",
    ),
    _site(
        "the acceptance floor in the `--val` comment",
        MODEL_TRAINING,
        r"# the acceptance number: per-class recall on the test split against the "
        r"(?P<floor>\d\.\d+) floor",
        lambda match: {"floor": _floor()},
        "`--val`'s floor is `train_model.RECALL_FLOOR`",
    ),
    _site(
        "the v1 continuity check's class list",
        MODEL_TRAINING,
        r"check the ingested export's (?P<count>[a-z]+) classes and print the run",
        lambda match: {"count": _word(_roster("v1"))},
        "v1's export declares v1's roster, which is v2's seven names today",
    ),
    _site(
        "the shared floor in the v1 run",
        MODEL_TRAINING,
        r"# per-class recall on the test split, against the same (?P<floor>\d\.\d+) floor v2 is "
        r"judged by",
        lambda match: {"floor": _floor()},
        "the same constant, quoted in the v1 run's comment",
    ),
    _site(
        "the per-class pass condition",
        MODEL_TRAINING,
        r"\| \*\*Per-class recall\*\* \| ≥ (?P<floor>\d\.\d+) for",
        lambda match: {"floor": _floor()},
        "§6's acceptance table states the floor",
    ),
    _site(
        "the example verdict line",
        MODEL_TRAINING,
        r"\(`milo [\d.]+ >= (?P<floor>\d\.\d+) \(n=\d+\)`\)",
        lambda match: {"floor": _floor()},
        "the example sentence prints the floor it was judged against",
    ),
    _site(
        "the per-class table's floor",
        MODEL_TRAINING,
        r"prints per-class recall against the (?P<floor>\d\.\d+) floor, with the instance count",
        lambda match: {"floor": _floor()},
        "what `--val` prints each class against, stated beside the table it produces",
    ),
    _site(
        "the class list at every distance",
        MODEL_TRAINING,
        r"The class list is the same (?P<count>[a-z]+) names at every distance",
        lambda match: {"count": _word(_roster())},
        "distance is a tag, and the class list is the roster",
    ),
    _site(
        "the roster section heading",
        MODEL_TRAINING,
        r"^### 8\.1 Class roster — (?P<count>\d+) classes, defined once",
        lambda match: {"count": str(_roster())},
        "§8.1 is where the roster is defined",
    ),
    _site(
        "the auto-label prompt count",
        MODEL_TRAINING,
        r"the prompt for each of the (?P<count>\d+) classes",
        lambda match: {"count": str(_roster())},
        "one prompt per class in the roster",
    ),
    _site(
        "the embedded class names",
        MODEL_TRAINING,
        r"all (?P<count>\d+) class names embedded",
        lambda match: {"count": str(_roster())},
        "the ONNX carries the roster's names",
    ),
    _site(
        "the distance-split head width",
        MODEL_TRAINING,
        r"list trains (?P<head>\d+) outputs and nothing in the file says so",
        lambda match: {"head": str(_head())},
        "a class list with a distance in it bakes one output per product and distance",
    ),
    _site(
        "the roster and the head in one sentence",
        MODEL_TRAINING,
        r"not the\s+(?P<noun>\d+)-name roster \(§8\.1\) — the (?P<head>\d+)-class failure",
        lambda match: {"noun": str(_roster()), "head": str(_head())},
        "the roster and the failure a distance-split list produces, in one sentence",
    ),
    _site(
        "the distance-name failure's shape",
        MODEL_TRAINING,
        r"trains a head with (?P<head>\d+) outputs instead of (?P<roster>\d+), and every box",
        lambda match: {"head": str(_head()), "roster": str(_roster())},
        "§8.1's worked example of how a tainted class list goes wrong",
    ),
    _site(
        "the head width in the checklist",
        MODEL_TRAINING,
        r"`… far` is a\s+(?P<head>\d+)-class head, not a per-distance one",
        lambda match: {"head": str(_head())},
        "the checklist's own summary of the distance-name failure",
    ),
    _site(
        "the fixture in the v2 acceptance checklist",
        MODEL_TRAINING,
        r"exit 0\*\* against a scratch (?P<head>\d+)-output weight",
        lambda match: {"head": str(_head())},
        "the §12 fixture's head width, as MODEL_TRAINING's checklist states it",
    ),
    _site(
        "which roster a weight is judged against",
        MODEL_TRAINING,
        r"v1's (?P<v1>[a-z]+), or v2's (?P<v2>[a-z]+) when the record says so",
        lambda match: {"v1": _word(_roster("v1")), "v2": _word(_roster("v2"))},
        "`resolve_roster` judges a weight against its own generation's roster",
    ),
    _site(
        "the tuning table's defaults and reload words",
        MODEL_TRAINING,
        r"^\| `(?P<field>[a-z_]+)` \(default `(?P<default>[^`]+)`\) "
        r"\|(?P<before>[^|]*?)(?P<reload>Restart-required|Hot-reloadable)(?P<after>[^|]*)\|$",
        _tuning_row,
        "each row states `Settings()`'s default and whether a running capture picks it up live "
        "(`settings_store`'s two sets)",
    ),
    # ---- .claude/skills/run-desktop/SKILL.md -------------------------------
    _site(
        "the classlist fixture's width",
        SKILL,
        r"nc=(?P<nc>\d+)\)` \+ the (?P<names>\d+) distance-split names",
        lambda match: {"nc": str(_head("v2")), "names": str(_head("v2"))},
        "the skill states the same fixture the run sheet does",
    ),
    _site(
        "the listing's class count",
        SKILL,
        r"row shows `(?P<count>\d+) classes` plus the",
        lambda match: {"count": str(_head("v2"))},
        "the Admin listing shows the recorded class list's length",
    ),
    _site(
        "the chip's class count",
        SKILL,
        r"`stat-classes` chip reading `(?P<chip>\d+)` / `1 finding`",
        lambda match: {"chip": str(_head("v2"))},
        "the Live chip shows the running model's class count",
    ),
    _site(
        "the Live banner's roster",
        SKILL,
        r"generation — v1's (?P<v1>[a-z]+) or v2's (?P<v2>[a-z]+), the same names today",
        lambda match: {"v1": _word(_roster("v1")), "v2": _word(_roster("v2"))},
        "the same per-generation roster rule `resolve_roster` applies",
    ),
)


def _read(path: Path) -> tuple[str, str | None]:
    """(the text with LF newlines, the newline the file uses - None when it uses both).

    A rewrite must not touch a byte it does not own, and a line ending is such a byte: this repo
    stores every text file with LF (`.gitattributes`: `* text=auto eol=lf`), and Python's text-mode
    write on Windows would silently convert a whole document to CRLF - a diff of every line, for a
    change of one number. So the splice always runs on LF text and the file's own newline is put
    back on the way out.

    A file that mixes the two is reported rather than guessed at: there is no single newline to
    restore it with, and picking one would rewrite lines this tool has no business touching.
    """
    data = path.read_bytes()
    crlf = data.count(b"\r\n")
    lf = data.count(b"\n")
    if crlf and crlf != lf:
        newline: str | None = None
    else:
        newline = "\r\n" if crlf else "\n"
    return data.decode("utf-8").replace("\r\n", "\n"), newline


def _encode(text: str, newline: str) -> bytes:
    return text.replace("\n", newline).encode("utf-8")


def _docs() -> dict[str, str]:
    return {doc: _read(REPO_ROOT / doc)[0] for doc in SITES_BY_DOC}


def apply(new_docs: dict[str, str], current: dict[str, str], root: Path = REPO_ROOT) -> tuple[int, list[str]]:
    """Write the docs that changed, each in the newline it already had. ([changed], [problems]).

    All the newlines are resolved before anything is written: a document whose endings cannot be
    restored stops the whole run, because writing the others first would leave a half-synced tree
    plus a non-zero exit, which is harder to reason about than either outcome alone.
    """
    targets = {doc: text for doc, text in new_docs.items() if text != current[doc]}
    newlines: dict[str, str] = {}
    problems: list[str] = []
    for doc in targets:
        newline = _read(root / doc)[1]
        if newline is None:
            problems.append(
                f"{doc} mixes LF and CRLF line endings - fix that first; this tool will not rewrite "
                "a document whose line endings it cannot put back"
            )
        else:
            newlines[doc] = newline
    if problems:
        return 0, problems
    for doc in targets:
        (root / doc).write_bytes(_encode(targets[doc], newlines[doc]))
    return len(targets), []


def _line(text: str, index: int) -> int:
    return text[:index].count("\n") + 1


def check(docs: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """(stale sites, unmatched sites). A site that matched nothing is reported, never skipped."""
    docs = docs or _docs()
    stale: list[str] = []
    unmatched: list[str] = []
    for doc, text in docs.items():
        for site in [site for site in SITES if site.doc == doc]:
            found, matched = site.edits(text)
            if not matched:
                unmatched.append(f"{doc}: site {site.name!r} matched nothing - {site.why}")
                continue
            for start, _end, current, wanted in found:
                stale.append(
                    f"{doc}:{_line(text, start)}: {site.name}: {current!r} -> {wanted!r} "
                    f"({site.why})"
                )
    return stale, unmatched


def sync(docs: dict[str, str] | None = None) -> dict[str, str]:
    """The docs with every site rendered from the code. Pure: no writing, no filesystem writes."""
    docs = docs or _docs()
    out: dict[str, str] = {}
    for doc, text in docs.items():
        for site in [site for site in SITES if site.doc == doc]:
            found, _matched = site.edits(text)
            for start, end, _current, wanted in sorted(found, reverse=True):
                text = text[:start] + wanted + text[end:]
        out[doc] = text
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report stale sites, write nothing")
    mode.add_argument("--write", action="store_true", help="rewrite the docs from the code")
    args = parser.parse_args(argv)

    docs = _docs()
    stale, unmatched = check(docs)


    for problem in unmatched:
        print(f"[!] {problem}", file=sys.stderr)
    if unmatched:
        print(
            "a site with nothing to match is a doc that no longer says what the site claims - "
            "re-point the site, or drop it",
            file=sys.stderr,
        )
        return 2 if args.write else 1

    if not stale:
        print(f"docs-sync: {len(SITES)} site(s) over {len(docs)} doc(s) are current")
        return 0

    if args.check:
        for problem in stale:
            print(problem)
        print(f"\n{len(stale)} stale number(s) - run `make docs-sync` to rewrite them")
        return 1

    if args.write:
        changed, problems = apply(sync(docs), docs)
        if problems:
            for problem in problems:
                print(f"[!] {problem}", file=sys.stderr)
            return 2
        for problem in stale:
            print(problem)
        print(f"\nrewrote {len(stale)} number(s) across {changed} doc(s)")
        return 0

    parser.error("pass --check or --write")


if __name__ == "__main__":
    raise SystemExit(main())
