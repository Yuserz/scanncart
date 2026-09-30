"""One engine for the wording guards: a table of surfaces in, assertions out.

The repo's wording guards started as two hand-rolled copies in
`test_desktop_contracts.py` (the null camera control, then the roster findings).
The plumbing - a normalizer that lets one sentence survive its line wraps and
its comment markers, a surface reader that fails loudly on a renamed file, the
presence / absence / canary assertions, and the failure messages that state the
same-edit contract - was identical, so a third wording contract would have been
a third copy of it. This module is that copy, taken once: a guard is now a
`WordingContract` table plus three one-line tests.

The convention (CLAUDE.md's `test_desktop_contracts` paragraph states it too):

- **Fragments, not sentences.** A surface may be reworded freely; losing the
  fact is what fails. The failure names the file and the missing fragment, so
  the drift is found where it happened.
- **The same-edit rule.** A wording surface that is added, moved, renamed or
  reworded takes its row with it in the same edit, because the fragments live
  in comments and rendered copy that nothing compiles - the table is the only
  place the drift can be caught.
- **A retired phrase is a guard of its own, and only with a history.** A phrase
  goes in `absent` when the wording it carried was wrong (it restated a bug),
  not merely old; a guard without a history does not invent one. The absence
  scan runs over the contract's own surfaces, because those are the ones a
  maintainer edits while thinking about that exact sentence.
- **The canary proves both ends.** Its fixture carries a marker *inside* a
  fragment's span - a marker at the fixture's edges is dropped with nothing
  between it and the words, so a broken normalizer survives it - and the engine
  asserts exactly the pinned fragments matched, no more and no fewer, so a
  fixture that drifted from the surfaces fails instead of passing green.

Everything here is text-only on purpose: the surfaces are read as files, never
imported, because several of them (`.ts`, `.tsx`, `.css`, `CLAUDE.md`) have no
Python import to offer and the ones that do would drag the runtime into the
mirror subset.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The tokens a comment wrapper contributes. A wrapped TS comment repeats `//` on every line and
#: a CSS block carries `/*` and ` * `, and those markers land *inside* a sentence the fragments
#: cross - dropping them as tokens is what makes one sentence one fact wherever its lines break.
#: A marker is matched as a whole token, so `http://127.0.0.1` and `a#b` survive untouched.
_COMMENT_MARKER_TOKENS = frozenset({"//", "/*", "*/", "*", "#", "##", "<!--", "-->"})


def normalize_wording(text: str) -> str:
    """`text` as one string of single-spaced words, so a fragment survives its line wraps.

    The wording these guards pin travels in comments - TypeScript, CSS and the sidecar's own -
    which the formatter and the docstring convention wrap freely: the same sentence is one fact
    whatever width it was written at, and a reader that demanded the original line breaks would
    fail on the first reformat rather than on any real drift. Comment markers are dropped as
    whole tokens (`_COMMENT_MARKER_TOKENS`), which is what keeps a fragment readable across the
    `//` its continuation lines carry.
    """
    return " ".join(w for w in text.split() if w not in _COMMENT_MARKER_TOKENS)


def read_surface(relative: str) -> str:
    """One wording surface as text. A moved or renamed file fails here, not as an empty read."""
    from tests.desktop_mirrors import REPO_ROOT

    path = REPO_ROOT / relative
    if not path.is_file():
        raise AssertionError(f"the surface this guard reads is gone: {relative}")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class WordingContract:
    """One wording fact, its surfaces, and the sentences its failures read as.

    `surfaces` is the table: `(file, fragments)` pairs, where every fragment is a load-bearing
    phrase the file must still carry (normalized - line wraps and comment markers do not count
    against it). `owner` names the module that owns the fact, for the failure messages.
    `claim`/`facts` are the presence failure's opening sentences; `absent`/`retired_why`
    configure the absence half, which only exists when the wording was *wrong* rather than old;
    `canary_fixture`/`canary_fragments` prove the guard is wired to both of its ends.
    """

    #: Names the fact, for messages ("the retired null camera control wording came back").
    name: str
    #: The module that owns the answer, spelled the way a failure message quotes it.
    owner: str
    #: The presence failure's first sentence - what no longer "carries its wording".
    claim: str
    #: The presence failure's second sentence - what the fact actually is, at its owner.
    facts: str
    #: `(file, fragments)` pairs; every fragment must survive in that file's normalized text.
    surfaces: tuple[tuple[str, tuple[str, ...]], ...]
    #: The canary's raw fixture text (normalized here, markers and all).
    canary_fixture: str = ""
    #: The fragments the canary fixture must match, and *only* those.
    canary_fragments: tuple[str, ...] = ()
    #: Phrases that must stay gone from every surface. Empty unless the wording was retired
    #: because it was wrong - a guard without a history does not invent one.
    absent: tuple[str, ...] = ()
    #: Why a returning phrase is a regression rather than a leftover; quoted by the absence
    #: failure. Required exactly when `absent` is non-empty.
    retired_why: str = ""

    def __post_init__(self) -> None:
        if not self.surfaces:
            raise ValueError(f"{self.name}: a wording contract with no surfaces pins nothing")
        for relative, fragments in self.surfaces:
            if not fragments:
                raise ValueError(f"{self.name}: {relative} lists no fragments, so it pins nothing")
            for fragment in fragments:
                if not fragment.strip():
                    raise ValueError(f"{self.name}: {relative} has a blank fragment")
        if not self.canary_fragments:
            raise ValueError(f"{self.name}: a canary with no fragments cannot fire")
        every_fragment = {f for _, fragments in self.surfaces for f in fragments}
        unknown = set(self.canary_fragments) - every_fragment
        if unknown:
            raise ValueError(
                f"{self.name}: canary fragments that are not surface fragments "
                f"(the canary must prove the guard's own wiring): {sorted(unknown)}"
            )
        if self.absent and not self.retired_why.strip():
            raise ValueError(
                f"{self.name}: a retired phrase needs the sentence saying why its return is "
                "a regression rather than a leftover"
            )
        overlap = set(self.absent) & every_fragment
        if overlap:
            raise ValueError(
                f"{self.name}: phrases both required and forbidden - the presence and absence "
                f"halves contradict each other: {sorted(overlap)}"
            )


def assert_fragments_present(contract: WordingContract) -> None:
    """Every surface still carries its fragments, or this names each file and what went missing."""
    missing: list[str] = []
    for relative, fragments in contract.surfaces:
        flat = normalize_wording(read_surface(relative))
        missing.extend(
            f"{relative}: {fragment!r}" for fragment in fragments if fragment not in flat
        )
    if missing:
        raise AssertionError(
            f"{contract.claim} {contract.facts} Reword the surface or move the wording into "
            "whatever replaced it, and update this guard in the same edit:\n  "
            + "\n  ".join(missing)
        )


def assert_wording_stays_retired(contract: WordingContract) -> None:
    """The retired phrase stays gone from the contract's own surfaces.

    Only those surfaces are scanned, because those are the ones a maintainer edits while thinking
    about this exact sentence - repo-wide would be a judgement about every sentence in the repo.
    """
    if not contract.absent:
        raise ValueError(f"{contract.name}: no retired phrase was configured")
    stale: list[str] = []
    for relative in (relative for relative, _ in contract.surfaces):
        flat = normalize_wording(read_surface(relative))
        stale.extend(relative for phrase in contract.absent if phrase in flat)
    if stale:
        raise AssertionError(
            f"the retired {contract.name} wording came back. {contract.retired_why} "
            "Reword the new text rather than reverting to the retired sentence; the owner of "
            f"the answer is `{contract.owner}`:\n  " + "\n  ".join(stale)
        )


def assert_canary_fires(contract: WordingContract) -> None:
    """The guard's own wiring check: the canary fixture must match exactly the pinned fragments.

    Zero matches - or a pinned fragment that failed to match - means the marker-dropping the
    wrapped comment surfaces depend on is broken, or the fragment set stopped matching, and the
    presence test would pass while checking nothing. An extra match is tolerated only when it is
    a substring of a pinned fragment: surfaces routinely carry one sentence in a short and a long
    form (the null-control wording does), a fixture quoting the long form necessarily contains
    the short one, and that is the same sentence matching - not drift. Any other extra means the
    fixture drifted from the surfaces and no longer proves what it was written to. A contract
    with retired phrases also proves that half here: the phrase has to be in the fixture for the
    absence test to be able to fire at all.
    """
    flat = normalize_wording(contract.canary_fixture)
    kept = {
        fragment
        for _, fragments in contract.surfaces
        for fragment in fragments
        if fragment in flat
    }
    unpinned = [f for f in kept if f not in set(contract.canary_fragments)]
    unaccounted = [f for f in unpinned if not any(f in pinned for pinned in contract.canary_fragments)]
    if unaccounted:
        raise AssertionError(
            f"the {contract.name} canary matched fragments its pin does not cover: {unaccounted!r}. "
            "An extra match is only legitimate when it is a substring of a pinned fragment (the "
            "same sentence in a shorter form); anything else means this fixture drifted and no "
            "longer proves what it was written to"
        )
    not_fired = [f for f in contract.canary_fragments if f not in kept]
    if not_fired:
        raise AssertionError(
            f"the {contract.name} canary did not fire on {not_fired!r}: the marker-dropping the "
            "wrapped comment surfaces depend on is broken, or the fragment set stopped matching "
            "- either way the presence test would never fire"
        )
    for phrase in contract.absent:
        if phrase not in flat:
            raise AssertionError(
                f"the retired-wording half of the {contract.name} canary matched nothing: "
                f"{phrase!r} is not in the fixture, so the absence test would never fire"
            )
