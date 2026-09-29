"""Drift guards: the rules' measured price in every file that quotes it by hand.

`app/acceptance.py` owns the three phantom rules and, since this guard exists, owns their price too
- `MEASURED_COST`, a record carrying the export population and what each rule took off it. The
problem it solves is not hypothetical. Before it, the same four numbers were typed into six files
that cannot import each other, and on the day the frame-filling default flipped they had already
come apart: `acceptance.py`'s own table said `252 of 2018` while `Settings`' comments and both
desktop hint strings still said `177 of 1936`, `10 of 1933` and `~9%` - the figures of a scratch
harness that had been deleted, contradicted by the tool the same commit added. A reader had no way
to tell which denominator was the live one, and the desktop half of it was user-facing prose. Those
three spellings are in `RETIRED` below, which is why this file may quote them and the copies it
checks may not.

So the copies are compared against the owner, in both directions:

**Every count the file quotes is the owner's.** Each copy's figures are matched as *patterns built
from the record*, never as literals retyped here, because a guard that hard-coded `252 of 2018`
would be a sixth copy of the same problem.

**No other count is attributed to that population.** Any `N of 2018` in these files has to be one of
the three rules' own counts, so a *new* stale figure fails even though nobody thought to blacklist
it - which is the direction that actually catches a re-measurement.

**The retired spellings are gone.** The denominators and shares this project really did ship are
listed as tombstones. They stay literal on purpose: their whole job is to be spellings nobody should
write again, and they are checked against the record below so that a future measurement landing on
one of them fails *here* - on the collision - rather than quietly re-adopting a number this module
once had to correct.

What this cannot check is the owner itself: whether `MEASURED_COST` still describes the weights.
That is `tools/unsure_probe.py`'s job, and it fails its own run when a fresh measurement disagrees -
which needs a GPU and the export, and so cannot live in CI. This guard is what runs on a bare
checkout, and it is why the *copies* cannot drift even when the measurement does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.acceptance import MEASURED_COST

from tests.desktop_mirrors import REPO_ROOT, TS_LIB, read_ts

pytestmark = pytest.mark.mirror

COST = MEASURED_COST
SETTINGS_PY = REPO_ROOT / "sidecar" / "app" / "settings.py"
ACCEPTANCE_PY = REPO_ROOT / "sidecar" / "app" / "acceptance.py"
ACCEPTANCE_TEST = REPO_ROOT / "sidecar" / "tests" / "test_acceptance.py"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
DEFAULTS_TS = TS_LIB / "settingsDefaults.ts"
FIELDS_TS = TS_LIB / "settingsFields.ts"

#: The spellings this project shipped before the figures had an owner. A tombstone list, not a
#: pattern: each one is a number that was wrong for the rule it was attached to, so seeing any of
#: them again means a copy was restored from an older draft. `_tombstones_quarantined` below is what
#: keeps this list from silently outliving its purpose.
RETIRED = (
    "1936",           # ground-truth-matched detections, the deleted harness's denominator
    "1933",           # ditto, in the unsure rule's own price
    "1983",           # ditto, the detections it counted
    "177 of",         # the frame-filling rule's cost, before the tool re-derived it
    "~9%",            # ditto, as a share
    "about 9%",       # ditto, as the desktop phrasing had it
    "0.5%",           # the unsure rule's price, before the tool re-derived it
)


def _prose(path: Path) -> str:
    """A file as one line of prose: comment markers off, every whitespace run collapsed.

    The figures are quoted inside wrapped paragraphs, so `252 of 2018` normally sits across a
    comment prefix and a newline, and a hint string can be wrapped by a formatter. Comparing the
    *prose* rather than the raw bytes is what makes this guard about the numbers instead of about
    where a paragraph happens to break: a rewrap must not fail it, and a stale number must.
    """
    if not path.is_file():
        raise AssertionError(f"a file this guard reads is gone: {path}")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        # Longest marker first, so `#:` does not become `:` and a TS comment does not keep its `//`.
        for marker in ("#:", "#", "//", ">", "*"):
            if stripped.startswith(marker):
                stripped = stripped[len(marker) :].lstrip()
                break
        lines.append(stripped)
    return " ".join(" ".join(lines).split())


def _of_matched(count: int) -> re.Pattern[str]:
    """`count` attributed to the export's matched population, as either copy spells it.

    `252 of 2018` and `6 of the 2018` are the same claim; only the second reads better in a
    sentence, and a guard that accepted one would either forbid the other or need the owner to
    carry a second rendering of the same number.
    """
    return re.compile(rf"\b{count}\s+of\s+(?:the\s+)?{COST.matched}\b")


def _present(fragment: str | re.Pattern[str], prose: str) -> bool:
    """Whether a file says `fragment` - a literal claim, or a pattern where the wording may vary."""
    if isinstance(fragment, re.Pattern):
        return bool(fragment.search(prose))
    return fragment in prose


def _label(fragment: str | re.Pattern[str]) -> str:
    return fragment.pattern if isinstance(fragment, re.Pattern) else fragment


#: What each copy has to say. Fragments are *built from the record* - a literal here would be the
#: sixth copy this module exists to prevent - and each entry is one sentence the file carries, so a
#: copy that quietly drops its figures fails rather than passing on an empty match.
COPIES = {
    "sidecar/app/settings.py": (
        _of_matched(COST.frame_filling),
        f"(~{COST.share(COST.frame_filling)})",
        f"it drops {COST.unsure}, or {COST.share(COST.unsure)}",
        f"{COST.clamp} ({COST.share(COST.clamp)}) and {COST.frame_filling}"
        f" ({COST.share(COST.frame_filling)}) of the same {COST.matched} matched",
        f"({COST.matched} ground-truth-matched",
    ),
    "sidecar/app/acceptance.py": (
        f"{COST.clamp} / {COST.matched}",
        f"{COST.frame_filling} / {COST.matched}",
        f"{COST.unsure} / {COST.matched}",
        _of_matched(COST.unsure),
        COST.share(COST.unsure),
    ),
    "sidecar/tests/test_acceptance.py": (
        _of_matched(COST.frame_filling),
        COST.share(COST.frame_filling),
    ),
    "desktop/src/renderer/src/lib/settingsDefaults.ts": (
        f"{COST.frame_filling} of the {COST.matched} boxes",
        f"~{COST.share(COST.frame_filling)}",
        f"{COST.unsure} of the export's {COST.matched} matched detections"
        f" ({COST.share(COST.unsure)})",
        f"{COST.clamp} ({COST.share(COST.clamp)}) and {COST.frame_filling}"
        f" (~{COST.share(COST.frame_filling)})",
    ),
    "desktop/src/renderer/src/lib/settingsFields.ts": (
        f"drops {COST.frame_filling} of {COST.matched} boxes that match their label"
        f" (about {COST.share(COST.frame_filling)})",
        f"costs {COST.unsure} of {COST.matched} real detections ({COST.share(COST.unsure)})",
    ),
    "CLAUDE.md": (
        f"{COST.frame_filling} of {COST.matched} ground-truth-matched",
        _of_matched(COST.unsure),
        COST.share(COST.unsure),
        f"the clamp rule's {COST.clamp}",
        f"the frame-filling rule's {COST.frame_filling}",
    ),
}


def _path_for(name: str) -> Path:
    if name == "CLAUDE.md":
        return CLAUDE_MD
    if name in {DEFAULTS_TS.name, FIELDS_TS.name}:
        return TS_LIB / name
    return REPO_ROOT / name


def test_every_copy_is_a_file_this_guard_can_read():
    """A renamed file must fail here, not silently stop being checked."""
    for name in COPIES:
        assert _path_for(name).is_file(), f"the copy this guard checks is gone: {name}"


def test_the_copies_quote_the_owners_figures():
    """Each file still says what the owner says, in its own wording."""
    wanted = set(COST.rules.values())
    for name, fragments in COPIES.items():
        prose = _prose(_path_for(name))
        missing = [_label(fragment) for fragment in fragments if not _present(fragment, prose)]
        assert not missing, (
            f"{name} no longer quotes the measured cost in app/acceptance.py::MEASURED_COST "
            f"(values {sorted(wanted)} of {COST.matched}); missing {missing}. Fix the prose, not "
            "the record - the record is what the tool measures against."
        )


def test_no_copy_attributes_another_count_to_the_measured_population():
    """The direction a blacklist cannot cover: a *new* figure for the same population.

    Any `N of 2018` in these files has to be one of the three rules' own counts. This is what
    catches a re-measurement that moved the numbers in the owner while some file kept an older
    one - the file's figure is then neither the owner's nor on the retired list.
    """
    allowed = set(COST.rules.values())
    pattern = re.compile(rf"\b(\d+)\s+of\s+(?:the\s+)?{COST.matched}\b")
    for name in COPIES:
        found = {int(match.group(1)) for match in pattern.finditer(_prose(_path_for(name)))}
        strays = sorted(found - allowed)
        assert not strays, (
            f"{name} attributes {strays} to the {COST.matched} ground-truth-matched detections, "
            f"but app/acceptance.py::MEASURED_COST records only {sorted(allowed)}. Either the file "
            "is stale or the record moved and this file was not updated."
        )


def test_the_retired_spellings_are_gone():
    """The figures this project shipped by mistake, in every file that once carried one."""
    for name in COPIES:
        prose = _prose(_path_for(name))
        found = [stale for stale in RETIRED if stale in prose]
        assert not found, (
            f"{name} quotes {found}, which is how the cost read before it had an owner - the "
            "frame-filling rule's price was 177 of 1936 there, not 252 of 2018. See "
            "app/acceptance.py::MEASURED_COST for the live figures."
        )


def test_tombstones_quarantined():
    """The retired list cannot collide with a live figure, or it would forbid the truth.

    Cheap, and it is the check that makes the list safe to keep: if a future measurement lands on
    one of these numbers, this fails and says so, rather than the tombstone quietly rejecting a
    correct copy. The fix then is to reword or delete the tombstone, deliberately.
    """
    live = {str(COST.matched), str(COST.frames), str(COST.predictions)}
    live |= {str(count) for count in COST.rules.values()}
    live |= {COST.share(count) for count in COST.rules.values()}
    collisions = [stale for stale in RETIRED if stale.strip("~%") in {v.strip("~%") for v in live}]
    assert not collisions, (
        f"these tombstones now describe a live figure: {collisions} - the measurement moved onto a "
        "number this list still forbids. Reword or drop the tombstone, and update "
        "app/acceptance.py::MEASURED_COST if the record is what is stale."
    )


def test_the_owner_records_the_operating_point():
    """A cost quoted without the threshold it was measured at is not reproducible.

    The module says this about the counts themselves and quotes them at `conf_threshold` 0.5: a
    phantom is a *low-confidence* detection, so the same frames produce fewer of them as the
    threshold rises. Carrying it is also what lets the tool decide whether its own run is
    comparable at all before it compares.
    """
    assert COST.conf == 0.5
    assert COST.matched <= COST.predictions <= COST.matched * 2, (
        "the matched population has to be a subset of the predictions it was matched out of"
    )
    assert COST.frames <= COST.predictions
