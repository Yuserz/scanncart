"""One engine for the wording guards: a table of surfaces in, assertions out.

The repo's wording guards started as two hand-rolled copies in
`test_desktop_contracts.py` (the null camera control, then the roster findings).
The plumbing - a normalizer that lets one sentence survive its line wraps and
its comment markers, a surface reader that fails loudly on a renamed file, the
presence / absence / canary assertions, and the failure messages that state the
same-edit contract - was identical, so a third wording contract would have been
a third copy of it. This module is that copy, taken once: a guard is now a
`WordingContract` table plus three one-line tests.

The convention (`sidecar/tests/CLAUDE.md`'s mirror-guard section states it too):

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

A second shape rides the same engine: a claim that **several surfaces must agree**.
A `WordingContract` asks, surface by surface, whether one still carries the fact,
which is blind to the failure that matters most for these - a rewrite that keeps
every pinned fragment everywhere and still sends two surfaces to different tools.
Such a claim has no surface of its own to live on, so it gets a row instead: an
`AgreementContract` names the tokens the surfaces are agreeing *about* and the
span each one keeps its copy in, and a table of rows is walked by two one-line
tests - the claim, and a canary that drops each token from each surface in turn
and requires exactly that surface and token to be reported. That per-pair
mutation is the second shape's version of the marker-inside-the-span rule: it
proves every token is load-bearing on every surface, not just that the check ran.

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


# --- agreements: one claim, several surfaces, and the row that says they match ---------------


@dataclass(frozen=True)
class SurfaceSpan:
    """One surface's copy of a claim, and how to find the copy inside the file.

    `opens`/`closes` are anchors the file already writes - a `def`/`export function` line, and the
    next top-level one after it - rather than a regex invented per claim. That is what makes the
    check about *that region*: a token mentioned somewhere else in the file must not be able to
    satisfy a row about one definition, and a renamed definition has to fail loudly instead of
    quietly widening the search to the whole file. Both anchors are required, and a span with no
    closing anchor fails too: running to the end of the file is the same defect one step quieter.
    """

    #: The file, spelled the way a failure message quotes it.
    surface: str
    #: Where the span starts: a line the file already has, verbatim.
    opens: str
    #: Where it ends: the next anchor of the same kind, verbatim.
    closes: str
    #: What this span is called in a failure message, when one file carries more than one of them.
    #: Empty falls back to the filename, which is what a file with a single span wants.
    label: str = ""


@dataclass(frozen=True)
class AgreementContract:
    """One fact several surfaces must state the same way - the claim that *spans* them.

    `WordingContract` asks "does this surface still carry the fact?" per surface. This asks the
    other question, the one no single surface's own row can: "do these agree?" The failure it
    exists for is a rewrite that keeps every pinned fragment in every row and still sends two
    surfaces to different tools, so the claim needs a row of its own rather than a bespoke test.

    `tokens` is what agreeing means for the row: every surface's own span has to name every token.
    A token is a name, a path or a flag - the prose around it belongs to the surfaces, and a whole
    *sentence* is a `WordingContract`'s business. `canary_spans` proves both ends: the row's own
    spans as text, one per surface, which each have to carry every token, and then each token is
    dropped from each surface in turn and the check must report exactly that surface and token.
    """

    #: Names the claim, for messages ("the two doors into one record no longer agree").
    name: str
    #: The places that own either end of the claim, spelled the way a failure message quotes them.
    owner: str
    #: The agreement failure's first sentence - what no longer agrees, and why that is not one
    #: surface's own business.
    claim: str
    #: The agreement failure's second sentence - what the surfaces are supposed to be saying.
    facts: str
    #: What every surface has to name for the row to hold.
    tokens: tuple[str, ...]
    #: `SurfaceSpan` rows: the surfaces, and the region of each that carries its copy.
    surfaces: tuple[SurfaceSpan, ...]
    #: `(span name, text)` pairs - one per surface - proving the check can see drift.
    canary_spans: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(self.surfaces) < 2:
            raise ValueError(
                f"{self.name}: an agreement needs at least two surfaces - with one there is "
                "nothing it could disagree with, and the claim belongs in a WordingContract"
            )
        if not self.tokens:
            raise ValueError(f"{self.name}: an agreement with no tokens agrees about nothing")
        for token in self.tokens:
            if not token.strip():
                raise ValueError(f"{self.name}: blank token")
        for outer in self.tokens:
            for inner in self.tokens:
                if outer != inner and inner in outer:
                    raise ValueError(
                        f"{self.name}: {inner!r} is part of {outer!r}, so naming the longer one "
                        "already names the shorter - and the canary, which drops one token at a "
                        "time, could not tell which of the two went missing"
                    )
        names = [span_name(span) for span in self.surfaces]
        if len(set(names)) != len(names):
            raise ValueError(
                f"{self.name}: two spans share a name, so a failure could not say which one "
                f"drifted - give one a `label`: {names}"
            )
        for span in self.surfaces:
            if not span.opens.strip() or not span.closes.strip():
                raise ValueError(
                    f"{self.name}: {span_name(span)} needs both anchors - a span read to the end "
                    "of a file would let a mention elsewhere in it satisfy this row"
                )
        canary = dict(self.canary_spans)
        if set(canary) != set(names):
            raise ValueError(
                f"{self.name}: the canary must speak for every span and no others - "
                f"spans={sorted(names)}, canary={sorted(canary)}"
            )


def span_name(span: SurfaceSpan) -> str:
    """The name one span goes by: its own label, or the file when it is that file's only one.

    This is also the key its canary text is filed under, so it is kept short - `span_where` is what
    a failure message prints, which adds the file back for a labelled span.
    """
    return span.label or span.surface


def span_where(span: SurfaceSpan) -> str:
    """How a failure message points at one span: the label *and* the file it lives in.

    A label alone ("the `Record it now` confirmation") says which copy drifted but not where to
    find it, and the whole point of naming the surface in a failure is that the reader can open it.
    """
    return f"{span.label} ({span.surface})" if span.label else span.surface


def read_span(span: SurfaceSpan) -> str:
    """The region of one surface that carries a claim, normalized to one string of words.

    Read as text and never imported, like every other surface here, and loud in both directions: a
    missing opening anchor means the definition was renamed or moved, and a missing closing one
    means the span would silently run to the end of the file - where a token from an unrelated part
    of it could stand in for the copy this row is about.
    """
    text = read_surface(span.surface)
    start = text.find(span.opens)
    if start == -1:
        raise AssertionError(
            f"{span.surface}: {span.opens!r} is gone, so the span this row reads is gone with it - "
            f"a renamed or moved definition has to fail here rather than quietly read the file "
            f"(the owner of this agreement is `{span.surface}`'s neighbour in the row)"
        )
    end = text.find(span.closes, start + len(span.opens))
    if end == -1:
        raise AssertionError(
            f"{span.surface}: no {span.closes!r} after {span.opens!r}, so this span has no end - "
            "reading on to the end of the file would let a mention elsewhere satisfy the row"
        )
    return normalize_wording(text[start:end])


def _disagreements(
    agreement: AgreementContract, texts: dict[str, str]
) -> list[tuple[str, str]]:
    """The `(span, token)` pairs whose own text does not name the token, in table order."""
    return [
        (span_name(span), token)
        for span in agreement.surfaces
        for token in agreement.tokens
        if token not in texts[span_name(span)]
    ]


def assert_surfaces_agree(agreement: AgreementContract) -> None:
    """Every surface's own span names every token, or this names what is missing and where it is.

    The failure distinguishes the two repairs, because they are different jobs: one surface drifted,
    or the fact moved everywhere and this row is what still remembers the old shape. So a missing
    token is reported with the surfaces that still name it - and with the fact that none do.
    """
    texts = {span_name(span): read_span(span) for span in agreement.surfaces}
    where = {span_name(span): span_where(span) for span in agreement.surfaces}
    # What the *other* surfaces are called in the "still named by" note: their labels, since the
    # reader already has the file in front of them by then.
    problems = _disagreements(agreement, texts)
    if not problems:
        return
    lines = []
    for name, token in problems:
        elsewhere = [other for other, text in texts.items() if token in text]
        lines.append(
            f"{where[name]}: its own span does not name {token!r}"
            + (
                f" (still named by {', '.join(elsewhere)})"
                if elsewhere
                else " (no surface in this row names it any more)"
            )
        )
    raise AssertionError(
        f"{agreement.claim} {agreement.facts} A token is what this row agrees *about*: a surface "
        "that stops naming one has stopped making the same claim, and a fact that left every "
        "surface is this row remembering a shape nothing says any more. Update this row in the "
        "same edit as the wording, or move it into a `WordingContract` if it is no longer a claim "
        "about several surfaces:\n  " + "\n  ".join(lines)
    )


def assert_agreement_canary_fires(agreement: AgreementContract) -> None:
    """Both ends, for every span and every token in the row.

    The canary is the row's own spans as text. Each first has to carry every token, or the row would
    fail on the fixture written to prove it works; then each token is dropped from each span in turn
    and the check must report exactly that span and that token. Per *pair* rather than one mutation,
    because a detector that consulted only the first span, or only the first token, would otherwise
    stay green - which is the canary's whole job, the same reason the fragment canary puts its
    marker inside the fragment's span instead of at its edges.
    """
    texts = {name: normalize_wording(text) for name, text in agreement.canary_spans}
    unfaithful = [
        f"{name}: {token!r}"
        for name, text in texts.items()
        for token in agreement.tokens
        if token not in text
    ]
    if unfaithful:
        raise AssertionError(
            f"the {agreement.name} canary does not carry its own tokens, so this row would fail on "
            "the fixture meant to prove the check works:\n  " + "\n  ".join(unfaithful)
        )
    for span in agreement.surfaces:
        name = span_name(span)
        for token in agreement.tokens:
            drifted = {**texts, name: texts[name].replace(token, "")}
            problems = _disagreements(agreement, drifted)
            if problems != [(name, token)]:
                raise AssertionError(
                    f"the {agreement.name} check does not catch {token!r} going missing from "
                    f"{name}: it reported {problems!r}. This row would let exactly that drift "
                    "through, which is what the canary is for"
                )
