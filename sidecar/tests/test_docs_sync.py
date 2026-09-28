"""The docs sync, tested by mutating every site it claims to own.

`docs_sync.py` rewrites the numbers the docs state from the code that owns them, so the failure mode
worth guarding is not "the numbers are stale" (that is `test_docs_claims.py`, and it is a net) but
"a site stopped matching anything and reports success" - the one state where the tool rewrites
nothing, prints a summary, and leaves a stale doc behind looking maintained.

So every site is exercised here: its sentence is found in the real doc, its owned span is mutated to
a wrong value, and the tool has to (a) report it under `--check`, (b) put it back byte-for-byte under
`--write`. Deleting the sentence instead proves the other half - a site with no match is an error,
never a skip. Both run over the shipped docs, so this is also the assertion that "the docs are what
the sync would write" holds right now: `check()` over the real files is empty.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import docs_sync

pytestmark = pytest.mark.docs

REPO_ROOT = Path(__file__).resolve().parents[2]

# The wrong value to write into a span: something a person could plausibly have typed, so the
# mutation is a realistic stale doc rather than a string the site's pattern would reject.
ALTERNATIVES = {"Hot-reloadable": "Restart-required", "Restart-required": "Hot-reloadable"}


def _real_docs() -> dict[str, str]:
    return {doc: (REPO_ROOT / doc).read_text(encoding="utf-8") for doc in docs_sync.SITES_BY_DOC}


def _wrong(current: str) -> str:
    """A plausible stale value for a span, of the same shape as the real one."""
    if current in ALTERNATIVES:
        return ALTERNATIVES[current]
    if current.isdigit():
        return str(int(current) + 1)
    if re.fullmatch(r"\d+\.\d+", current):
        return f"{float(current) + 0.01:.2f}"
    if re.fullmatch(r"\d+/\d+/\d+", current):
        return "80/10/10"
    if "," in current:  # the printed augmentation table
        first, _, rest = current.partition(",")
        key, _, value = first.partition("=")
        return f"{key}={float(value) + 0.1:g},{rest}"
    if current in docs_sync.NUMBER_WORDS.values():
        spelled = {word: number for number, word in docs_sync.NUMBER_WORDS.items()}
        return docs_sync._word(spelled[current] + 1)
    raise AssertionError(f"no mutation is defined for the span {current!r}")


def test_the_docs_the_tool_owns_use_the_repo_s_line_endings():
    """`.gitattributes` says `* text=auto eol=lf`, and this tool rewrites the files it owns.

    A text-mode write on Windows puts CRLF in them, which is a diff of every line for a change of
    one number - and it is the failure this tool produced on its first run, so the bytes are
    asserted rather than the intent.
    """
    for doc in docs_sync.SITES_BY_DOC:
        data = (REPO_ROOT / doc).read_bytes()
        assert b"\r\n" not in data, f"{doc} has CRLF line endings; the repo stores LF"


def test_a_rewrite_keeps_the_line_endings_the_document_had(tmp_path):
    """LF stays LF and CRLF stays CRLF: the splice owns spans, never a document's newlines."""
    for newline in ("\n", "\r\n"):
        doc = "docs/sample.md"
        path = tmp_path / doc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(docs_sync._encode("one\ntwo\n", newline))
        changed, problems = docs_sync.apply(
            {doc: docs_sync._read(path)[0].replace("two", "three")},
            {doc: docs_sync._read(path)[0]},
            root=tmp_path,
        )
        assert (changed, problems) == (1, [])
        assert path.read_bytes() == docs_sync._encode("one\nthree\n", newline)


def test_a_document_with_mixed_line_endings_is_refused(tmp_path):
    """No single newline can be restored, so nothing is written at all."""
    doc = "docs/mixed.md"
    path = tmp_path / doc
    path.parent.mkdir(parents=True, exist_ok=True)
    original = b"one\r\ntwo\nthree\r\n"
    path.write_bytes(original)
    changed, problems = docs_sync.apply(
        {doc: "changed\n"}, {doc: docs_sync._read(path)[0]}, root=tmp_path
    )
    assert changed == 0
    assert problems and "mixes LF and CRLF" in problems[0]
    assert path.read_bytes() == original


def test_every_site_belongs_to_a_doc_the_tool_owns():
    """A site for an unlisted doc would be checked by nothing and synced by nothing."""
    assert {site.doc for site in docs_sync.SITES} == set(docs_sync.SITES_BY_DOC)


def test_the_shipped_docs_are_what_the_sync_would_write():
    """The headline property: nothing to rewrite, and every site has a sentence to rewrite."""
    stale, unmatched = docs_sync.check(_real_docs())
    assert not unmatched, "a site matched nothing:\n" + "\n".join(unmatched)
    assert not stale, "stale number(s) - run `make docs-sync`:\n" + "\n".join(stale)


def test_writing_the_shipped_docs_changes_nothing():
    """`--write` on a current tree is a no-op, byte for byte - so running it is always safe."""
    docs = _real_docs()
    assert docs_sync.sync(docs) == docs


def test_the_sync_leaves_every_other_byte_alone():
    """A rewrite is a splice of the spans it owns: prose, wrapping and punctuation are untouched.

    The check that matters for a tool that edits Markdown: mutate one site, and the only difference
    between the synced text and the original is that site's span.
    """
    docs = _real_docs()
    site = next(site for site in docs_sync.SITES if site.name == "CROWD_MIN")
    match = site.pattern.search(docs[site.doc])
    assert match, "the CROWD_MIN site should match the run sheet"
    mutated = (
        docs[site.doc][: match.start("margin")] + "9" + docs[site.doc][match.end("margin") :]
    )
    synced = docs_sync.sync({site.doc: mutated})[site.doc]
    assert synced == docs[site.doc]
    assert synced.count("\n") == mutated.count("\n")


@pytest.mark.parametrize("site", docs_sync.SITES, ids=lambda site: f"{site.doc}:{site.name}")
def test_every_site_reports_its_own_drift_and_repairs_it(site):
    """Mutate this site's span, and only this site's span: check must name it, sync must undo it."""
    docs = _real_docs()
    text = docs[site.doc]
    match = next(site.pattern.finditer(text), None)
    assert match, f"{site.name} no longer matches its doc ({site.why})"

    owned = site.values(match)
    assert owned, f"{site.name} matches but owns no span - it can never be synced"
    group = next(iter(owned))
    wrong = _wrong(match.group(group))
    mutated = text[: match.start(group)] + wrong + text[match.end(group) :]
    assert mutated != text

    stale, unmatched = docs_sync.check({site.doc: mutated})
    assert not unmatched, f"{site.name} stopped matching once its span changed"
    assert any(site.name in line for line in stale), (
        f"a stale {site.name} was not reported:\n" + "\n".join(stale)
    )
    assert docs_sync.sync({site.doc: mutated})[site.doc] == text, (
        f"the sync did not restore the {site.name} span it owns"
    )


@pytest.mark.parametrize("site", docs_sync.SITES, ids=lambda site: f"{site.doc}:{site.name}")
def test_a_site_whose_sentence_is_gone_is_an_error_not_a_skip(site):
    """Deleting the sentence a site points at is reported, because the alternative is a tool that
    quietly stops maintaining a doc while still reporting that it does.
    """
    docs = _real_docs()
    text = docs[site.doc]
    matches = list(site.pattern.finditer(text))
    assert matches, f"{site.name} no longer matches its doc"
    # Every match, not just the first: a site that owns three table rows is only unmatched when all
    # three sentences are gone.
    stripped = text
    for match in reversed(matches):
        stripped = stripped[: match.start()] + stripped[match.end() :]

    _stale, unmatched = docs_sync.check({site.doc: stripped})
    assert any(site.name in line for line in unmatched), (
        f"removing the sentence behind {site.name!r} was not reported as unmatched"
    )


def test_write_refuses_a_doc_whose_site_has_gone(monkeypatch):
    """`--write` exits 2 without writing anything: a half-synced file is worse than a stale one.

    The doc it is pointed at is mutated in memory and *also* made stale elsewhere, so a tool that
    wrote the sites it could find before failing would leave a changed file on disk.
    """
    docs = _real_docs()
    gone = next(site for site in docs_sync.SITES if site.name == "CROWD_MIN")
    stale = next(site for site in docs_sync.SITES if site.name == "the pass condition")
    text = docs[gone.doc]
    match = gone.pattern.search(text)
    broken = text[: match.start()] + text[match.end() :]
    match = stale.pattern.search(broken)
    broken = broken[: match.start("floor")] + "0.99" + broken[match.end("floor") :]
    before = (REPO_ROOT / gone.doc).read_text(encoding="utf-8")

    monkeypatch.setattr(docs_sync, "_docs", lambda: {gone.doc: broken})
    assert docs_sync.main(["--write"]) == 2
    assert (REPO_ROOT / gone.doc).read_text(encoding="utf-8") == before


def test_check_exits_zero_on_the_shipped_docs():
    assert docs_sync.main(["--check"]) == 0


def test_the_plan_split_ratio_comes_from_the_planner_not_from_the_doc():
    """Both halves of one site, to show where the value comes from: the ratio is `TARGET` x 100 over
    `SPLIT_NAMES`, so a re-ordered or re-weighted split moves the prose with it.
    """
    import label_classes
    import plan_split

    assert docs_sync._ratio() == "/".join(
        f"{plan_split.TARGET[name] * 100:.0f}" for name in label_classes.SPLIT_NAMES
    )
    assert docs_sync._head() == len(docs_sync.V2_ROSTER) * len(label_classes.DISTANCE_ORDER)
