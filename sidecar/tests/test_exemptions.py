"""The shared exemption row's own tests: every property that makes one worth having is a negative one.

A row that silences nothing, a lookup answered by a shape the row does not name, a phrase excused in a
place it was not written for, a trace that records a waiver nothing applied, a liveness rule that stops
reading one of a row's places, an empty table nobody declared empty - none of those show up in a green
run of the three guards that keep such tables,
which is the same reason `test_wording_engine.py` exists beside the wording contracts. The rows here are
fixtures: the real tables are exercised by the guards that own them.
"""

from __future__ import annotations

import pytest

from tests.exemptions import Exemption, Exemptions, Waiver, assert_still_present, waived_note


def _row(
    phrase: str = "a phrase",
    reason: str = "a reason",
    shape: str = "a shape",
    places: tuple[str, ...] = ("a/doc.md",),
) -> Exemption:
    return Exemption(phrase=phrase, reason=reason, shape=shape, places=places)


def test_a_row_that_pins_nothing_fails_at_construction():
    """A blank phrase is the worst of these - it is inside every text, so it silences a whole scan -
    and the other three leave a reader with nothing to judge: no reason, nowhere it applies, or no shape
    that could ever ask for it.
    """
    with pytest.raises(ValueError, match="no phrase"):
        _row(phrase="")
    with pytest.raises(ValueError, match="no reason"):
        _row(reason="   ")
    with pytest.raises(ValueError, match="no place"):
        _row(places=())
    with pytest.raises(ValueError, match="no shape"):
        _row(shape="")


def test_two_rows_for_one_phrase_are_refused():
    """Keyed by phrase, so the second row's reason and arm would be unreachable: the first answers and
    the other reads as a rule while covering nothing.
    """
    with pytest.raises(ValueError, match="more than one exemption"):
        Exemptions(_row(reason="the one that answers"), _row(reason="the one that does not"))


def test_a_lookup_is_answered_only_by_the_shape_and_the_place_the_row_names():
    """The three matches a row makes, one at a time: the shape it was written for, one of its places,
    and its phrase inside the text it was asked about. Any of them missing is `None` - which is what
    stops a phrase excused in one scan step, or one doc, from silencing another.
    """
    table = Exemptions(_row(phrase="a phrase", reason="why", shape="a scan step", places=("a/doc.md", "b.md")))
    assert table.reason("a scan step", "a/doc.md", "text with a phrase in it") == "why"
    assert table.reason("a scan step", "b.md", "a phrase") == "why"
    # Another shape - another arm, or another pattern of the same one - another place, or text that
    # does not carry the phrase: not excused.
    assert table.reason("another scan step", "a/doc.md", "a phrase") is None
    assert table.reason("a scan step", "c.md", "a phrase") is None
    assert table.reason("a scan step", "a/doc.md", "something else entirely") is None
    # And the shapes it excuses, and the places it is written about, are the ones its rows name.
    assert table.shapes == frozenset({"a scan step"})
    assert table.places == ("a/doc.md", "b.md")
    assert len(table) == 1 and [row.phrase for row in table.rows] == ["a phrase"]


def test_a_lookup_hands_back_a_trace_when_the_caller_collects_one():
    """The waiver a row applies is recorded where the caller asked for it, carrying the row's own reason
    - the difference between "that row is still in the table" and "this scan skipped something because
    of it". A miss records nothing, a caller that collects nothing pays nothing, and `waived_note`
    renders the collected ones for a failure message (or renders nothing, for the same empty case).
    """
    table = Exemptions(_row(phrase="a phrase", reason="why"))
    waived: list[Waiver] = []
    assert table.reason("a shape", "a/doc.md", "text with a phrase in it", waived) == "why"
    assert [str(waiver) for waiver in waived] == ["'a phrase' in a/doc.md [a shape]: why"]
    assert waived_note(waived) == "\n(waived by an exemption: 'a phrase' in a/doc.md [a shape]: why)"
    # Another shape, another place, or text the row does not name: not excused, and nothing recorded.
    assert table.reason("another shape", "a/doc.md", "a phrase", waived) is None
    assert table.reason("a shape", "other.md", "a phrase", waived) is None
    assert table.reason("a shape", "a/doc.md", "something else", waived) is None
    assert len(waived) == 1, waived
    # And a caller that only wants the verdict gets no trace to keep.
    assert table.reason("a shape", "a/doc.md", "a phrase") == "why"
    assert waived_note([]) == ""


def test_an_empty_table_passes_only_when_the_caller_says_empty_is_the_point():
    """With no rows there is nothing to check, so the rule cannot see the failure it exists for: a table
    someone cleared passes trivially and reports nothing. The declaration is what puts that state in
    front of a reader, and it is read both ways so it cannot outlive the emptiness it describes.
    """
    texts = {"a/doc.md": "a phrase lives here"}
    empty = Exemptions()
    with pytest.raises(AssertionError, match="nothing says that is deliberate"):
        assert_still_present(empty, texts.__getitem__)
    assert_still_present(empty, texts.__getitem__, designed_empty=True)
    # ...and the declaration has to go when the table grows its first row.
    grown = Exemptions(_row(phrase="a phrase", places=("a/doc.md",)))
    with pytest.raises(AssertionError, match="declared empty on purpose"):
        assert_still_present(grown, texts.__getitem__, designed_empty=True)
    assert_still_present(grown, texts.__getitem__)


def test_the_liveness_rule_fires_both_ways():
    """`assert_still_present` reads every place a row names, so one of two places losing the phrase is
    enough: an exemption left behind after an edit is a rule covering text nobody read.
    """
    live = Exemptions(_row(phrase="a phrase", reason="why", places=("a/doc.md", "b.md")))
    texts = {"a/doc.md": "before a phrase after", "b.md": "before a phrase after"}
    assert_still_present(live, texts.__getitem__)

    for missing in ("a/doc.md", "b.md"):
        edited = dict(texts, **{missing: "the sentence that replaced it"})
        with pytest.raises(AssertionError) as caught:
            assert_still_present(live, edited.__getitem__)
        # The failure names the place that lost it and the reason the row was written for.
        assert missing in str(caught.value) and "why" in str(caught.value)
