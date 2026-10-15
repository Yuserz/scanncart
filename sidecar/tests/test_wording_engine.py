"""The engine's own tests: the rules an agreement row has to satisfy, and the checks that can fail.

Every other module here tests a *claim* through the engine. This one tests the engine, because the
properties that make the claims worth having are all negative ones - a row that pins nothing, a span
that quietly reads the whole file, a canary that proves the check ran rather than that it works - and
none of them show up in a green run of the claims themselves.

Not marked `mirror`: these read the engine, not the two toolchains. The claims it carries are marked
where they live (`test_desktop_contracts.py`).
"""

from __future__ import annotations

import pytest

from tests import wording
from tests.wording import (
    AgreementContract,
    SurfaceSpan,
    assert_agreement_canary_fires,
    assert_surfaces_agree,
    read_span,
    read_surface,
    span_name,
    span_where,
)

#: A real file with two functions in it, one of which calls the other - so a token can be *in the
#: file* and outside a given span, which is the property the anchors exist for.
TWO_FUNCTIONS = "sidecar/tests/desktop_mirrors.py"

#: The two spans in that file, labelled - one file with two spans of it is exactly the shape the
#: labels exist for. The anchors are the file's own def lines and the two-blank-line break between
#: them, not a pattern invented here.
SINGULAR = "the singular comment stripper"
READER = "the reader"


def _two_spans(closes: str = "\n\n\ndef ") -> tuple[SurfaceSpan, ...]:
    return (
        SurfaceSpan(TWO_FUNCTIONS, "def strip_ts_comment", closes, label=SINGULAR),
        SurfaceSpan(TWO_FUNCTIONS, "def read_ts", closes, label=READER),
    )


def _agreement(**overrides) -> AgreementContract:
    """A row that holds, so a test can break exactly one thing about it."""
    fields = dict(
        name="a test agreement",
        owner="tests/test_wording_engine.py",
        claim="the two halves of this fixture drifted.",
        facts="They are one fact written twice.",
        tokens=("alpha", "beta"),
        surfaces=_two_spans(),
        canary_spans=(
            (SINGULAR, "strip_ts_comment and read_ts: alpha and beta"),
            (READER, "alpha, beta"),
        ),
    )
    fields.update(overrides)
    return AgreementContract(**fields)


# --- the rules a row has to satisfy ---------------------------------------------------------


def test_an_agreement_needs_something_to_disagree_with() -> None:
    """One surface has no claim in it: there is nothing for it to agree *with*, and the fact is a
    `WordingContract`'s - which is the engine's own advice in the failure."""
    with pytest.raises(ValueError, match="at least two surfaces"):
        _agreement(surfaces=(SurfaceSpan(TWO_FUNCTIONS, "def read_ts", "\n\n\ndef "),))


def test_an_agreement_without_tokens_agrees_about_nothing() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        _agreement(tokens=())
    with pytest.raises(ValueError, match="blank token"):
        _agreement(tokens=("alpha", "  "))


def test_a_token_inside_another_token_is_refused() -> None:
    """Because the canary drops one token at a time: if `alpha` is part of `alpha-two`, removing it
    takes both out and the check could not say which went missing - and naming the longer one
    already implies the shorter, so the row's second token was never doing any work."""
    with pytest.raises(ValueError, match="is part of"):
        _agreement(tokens=("alpha", "alpha-two"))


def test_two_spans_of_one_file_have_to_be_tellable_apart() -> None:
    """A failure that named the file twice for two different spans would leave the reader to guess
    which copy drifted."""
    with pytest.raises(ValueError, match="share a name"):
        _agreement(
            surfaces=(
                SurfaceSpan(TWO_FUNCTIONS, "def strip_ts_comment", "\n\n\ndef "),
                SurfaceSpan(TWO_FUNCTIONS, "def read_ts", "\n\n\ndef "),
            )
        )


def test_both_anchors_are_required() -> None:
    """A span read to the end of a file is the defect the anchors exist to prevent, so it is refused
    where it is declared rather than tolerated at read time."""
    with pytest.raises(ValueError, match="both anchors"):
        _agreement(surfaces=_two_spans(closes=""))


def test_the_canary_has_to_speak_for_every_span() -> None:
    """In both directions: a span with no canary text is a span whose half of the row is unproven,
    and a canary text for a span that is not in the row proves nothing about the row."""
    with pytest.raises(ValueError, match="must speak for every span"):
        _agreement(canary_spans=((SINGULAR, "alpha and beta"),))
    with pytest.raises(ValueError, match="must speak for every span"):
        _agreement(canary_spans=(*_agreement().canary_spans, ("a span that is not here", "alpha")))


# --- what a span reads, and what it refuses to read -----------------------------------------


def test_a_span_is_its_own_region_and_not_the_whole_file() -> None:
    """The property that makes a row about *one definition*: `strip_ts_comments` appears later in the
    same file, and a reader that took the whole file would let that mention satisfy a row whose
    surface is the singular function."""
    span = SurfaceSpan(TWO_FUNCTIONS, "def strip_ts_comment", "\n\n\ndef ")
    region = read_span(span)
    assert "strip_ts_comment(" in region
    assert "strip_ts_comments" in read_surface(TWO_FUNCTIONS), "the token must exist outside the span"
    assert "strip_ts_comments" not in region


def test_a_missing_anchor_fails_loudly_in_both_directions() -> None:
    """A renamed definition and a span whose closing anchor moved are the same defect: the row would
    read the wrong region. Neither may degrade into reading the file, which is the mistake the
    hand-rolled reader this engine replaced made (its closing anchor never matched the file's last
    export, so that span silently ran to the end of the file)."""
    with pytest.raises(AssertionError, match="is gone"):
        read_span(SurfaceSpan(TWO_FUNCTIONS, "def renamed_away", "\n\n\ndef "))
    with pytest.raises(AssertionError, match="has no end"):
        read_span(SurfaceSpan(TWO_FUNCTIONS, "def read_ts", "\n\n\ndef something_that_is_not_there"))


def test_a_span_name_falls_back_to_the_file_it_is_the_only_one_of() -> None:
    assert span_name(SurfaceSpan(TWO_FUNCTIONS, "x", "y")) == TWO_FUNCTIONS
    assert span_name(SurfaceSpan(TWO_FUNCTIONS, "x", "y", label=READER)) == READER


def test_a_labelled_span_still_says_which_file_it_is_in() -> None:
    """A label says which copy drifted; only the file says where to go and fix it, and a message
    that printed the label alone would leave the reader searching the repo for a sentence."""
    assert span_where(SurfaceSpan(TWO_FUNCTIONS, "x", "y", label=READER)) == f"{READER} ({TWO_FUNCTIONS})"
    assert span_where(SurfaceSpan(TWO_FUNCTIONS, "x", "y")) == TWO_FUNCTIONS


# --- the canary's own teeth ----------------------------------------------------------------


def test_a_faithful_canary_passes() -> None:
    assert_agreement_canary_fires(_agreement())


def test_the_canary_refuses_a_fixture_that_is_missing_its_own_tokens() -> None:
    """A canary that does not carry the tokens would fail on the fixture written to prove the check
    works, and a row would look broken instead of proving anything."""
    with pytest.raises(AssertionError, match="does not carry its own tokens"):
        assert_agreement_canary_fires(
            _agreement(
                canary_spans=(
                    (SINGULAR, "alpha and beta"),
                    (READER, "only alpha here"),
                )
            )
        )


def test_the_canary_catches_a_check_that_ignores_a_surface(monkeypatch) -> None:
    """The reason the canary mutates per *pair* rather than once: a detector that consulted only the
    first surface would satisfy a single mutation and stay green while blind to the rest of the row.

    The stub is that detector - it looks at one span and calls it the agreement.
    """
    agreement = _agreement()

    def first_span_only(row, texts):
        name = span_name(row.surfaces[0])
        return [(name, token) for token in row.tokens if token not in texts[name]]

    monkeypatch.setattr(wording, "_disagreements", first_span_only)
    with pytest.raises(AssertionError, match="does not catch"):
        assert_agreement_canary_fires(agreement)


def test_a_real_disagreement_names_the_surface_the_token_and_where_it_still_is(monkeypatch) -> None:
    """The message is the deliverable: one surface drifting and a fact that left every surface are
    different repairs, and the reader has to be able to tell which one they are looking at."""
    real = _agreement(tokens=("write_text", "strip_ts_comment"))
    monkeypatch.setattr(
        wording,
        "read_span",
        lambda span: "" if span.opens == "def read_ts" else "write_text strip_ts_comment",
    )
    with pytest.raises(AssertionError) as excinfo:
        assert_surfaces_agree(real)
    message = str(excinfo.value)
    assert "does not name 'write_text'" in message
    assert f"{READER} ({TWO_FUNCTIONS})" in message, "the failing span has to be openable"
    assert f"still named by {SINGULAR}" in message, "the reader needs the surface that still has it"


def test_a_disagreement_with_no_other_surface_left_says_so(monkeypatch) -> None:
    """The other repair: the token moved everywhere and this row is what still remembers the old
    shape. Saying it is only missing *here* would send the reader to add a name back that every
    surface has deliberately dropped."""
    real = _agreement(tokens=("write_text",))
    monkeypatch.setattr(wording, "read_span", lambda span: "")
    with pytest.raises(AssertionError) as excinfo:
        assert_surfaces_agree(real)
    assert "no surface in this row names it any more" in str(excinfo.value)
