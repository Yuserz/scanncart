"""Tests for `app/source_index.py` - the machine-local map from a weight's bytes to its dataset.

The file answers one question in the one case every other source fails in: *after the training run
is gone*. Two of its properties are load-bearing enough to pin here rather than only through the
tools that write it, because both are silent when they are wrong:

* **the key is the content.** The three v2 weights are installed as
  `-v2-stretch`/`-v2-letterbox`/`-v2-letterbox-synth` while their run directories are
  `scanncart-grocery-v2-2`/`-2-3`/`-2-4`, so a name-keyed index would be right about today's files
  and wrong the first time one is renamed - and the entry has to survive a `--force` re-install
  putting different bytes under a name it already knows.
* **a bad read costs nothing.** This is loaded on every `/api/models` and for every weight a repair
  run inspects, so a missing file, an unparseable one and one junk entry must each cost what they
  are worth - nothing - rather than the listing or the repair.

Two more are rules rather than taste, and each has a test: a blank `source` is refused *by the
writer* (an entry that answers with a blank looks like a lookup that worked while leaving the gap in
place), and the file is never a second opinion about a record that exists - that gate lives in
`app.models.installed_models` and is tested in `tests/test_models.py`.

Every test writes to `source_index.index_path()`, which `tests/conftest.py` redirects into the test's
own temporary directory: the machine's real index is never read or written by this suite.
"""

from __future__ import annotations

import json

import pytest

from app import source_index


def _weight(path, payload: bytes = b"weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _recorded(weight, source: str = "local merge merged-v2", **extra):
    """One entry, written through the writer every tool uses - never by hand."""
    return source_index.record_source(weight, source, **extra)


# --- the key is the content --------------------------------------------------------------


def test_a_weight_is_found_by_its_bytes_whatever_it_is_called(tmp_path):
    """The reason the key is a hash: an installed weight can be renamed, and a copy can arrive under
    another name, without becoming a different weight."""
    weight = _weight(tmp_path / "models" / "scanncart-grocery-v2-stretch.pt")
    _recorded(weight)

    renamed = _weight(tmp_path / "elsewhere" / "whatever.pt")
    assert source_index.lookup(renamed).source == "local merge merged-v2"


def test_different_bytes_are_a_different_weight(tmp_path):
    """The direction that matters when a `--force` re-install replaces a weight: the entry for the
    old bytes must not answer for the new file, since the two were trained from different sets."""
    _recorded(_weight(tmp_path / "a.pt", b"one"))
    assert source_index.lookup(_weight(tmp_path / "b.pt", b"two")) is None


# --- a bad read costs nothing ------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        None,  # no file at all
        "",  # empty
        "not json at all",
        "[]",  # JSON, but not an object
        '{"version": 1}',  # no entries
        '{"weights": []}',  # entries of the wrong shape
        '{"weights": {"not-a-digest": {"source": "x"}}}',  # unreachable key
    ],
)
def test_a_file_that_says_nothing_reads_as_no_index(tmp_path, body):
    """Same contract as `app/models.read_record`: this is read to *report* something, and a
    hand-edited file (plain JSON, in a gitignored directory) is a normal way for it to arrive."""
    path = tmp_path / "source_index.json"
    if body is not None:
        path.write_text(body, encoding="utf-8")
    assert source_index.read_index(path) == {}


def test_one_junk_entry_costs_only_itself(tmp_path):
    """A file with a usable entry beside unusable ones keeps the usable one: the tolerance is per
    entry rather than per file, or one bad hand edit would blank a machine's whole history."""
    weight = _weight(tmp_path / "models" / "fresh.pt", b"fresh")
    _recorded(weight)
    path = source_index.index_path()
    stored = json.loads(path.read_text(encoding="utf-8"))
    digest = next(iter(stored["weights"]))
    stored["weights"]["a" * 64] = {"source": "   "}  # says nothing
    stored["weights"]["b" * 64] = "not an object"
    stored["weights"]["c" * 64] = {"source": "unnamed but usable"}
    path.write_text(json.dumps(stored), encoding="utf-8")

    entries = source_index.read_index()
    assert entries[digest].source == "local merge merged-v2"
    assert set(entries) == {digest, "c" * 64}


# --- what an entry holds -----------------------------------------------------------------


def test_recording_writes_the_documented_shape_and_leaves_no_temporary_behind(tmp_path):
    weight = _weight(tmp_path / "models" / "w.pt", b"payload")
    written = _recorded(
        weight,
        dataset="D:/sets/merged-v2",
        run="D:/runs/scanncart-grocery-v2-2",
        generation="v2",
        now="2026-10-14T04:00:00",
    )

    body = json.loads(written.read_text(encoding="utf-8"))
    assert body["version"] == source_index.INDEX_VERSION
    (entry,) = body[source_index.WEIGHTS_KEY].values()
    assert entry == {
        "source": "local merge merged-v2",
        # The bytes' size is evidence, not a lookup key: the key is the hash above it.
        "size": len(b"payload"),
        "dataset": "D:/sets/merged-v2",
        "run": "D:/runs/scanncart-grocery-v2-2",
        "generation": "v2",
        "recorded_at": "2026-10-14T04:00:00",
    }
    # Written through a temporary and replaced, like every other record this repo writes: a
    # truncated index reads as *no* index, which is a normal state, so the loss would be silent.
    # The temporary name is therefore gone again - and it is outside the `*.json` a reader looks
    # for, so a crash mid-write cannot leave something that half-parses as the index.
    assert written.is_file()
    assert list(written.parent.glob("*.tmp")) == []


def test_a_second_write_merges_field_by_field_rather_than_replacing_the_entry(tmp_path):
    """The two writers know different halves - the trainer has the run and the set, `--install` has
    the `--version` the operator passed - so the later write has to update `source` without erasing
    the run the first one knew. And it has to update rather than append: one weight, one entry."""
    weight = _weight(tmp_path / "models" / "w.pt")
    _recorded(
        weight,
        run="D:/runs/run-2",
        dataset="D:/sets/merged-v2",
        generation="v2",
        now="2026-10-14T04:00:00",
    )
    _recorded(weight, "snc-grocery version 2", now="2026-10-14T05:00:00")

    (entry,) = source_index.read_index().values()
    # `--version` wins the field it is about - that is what the record beside the weights will say.
    assert entry.source == "snc-grocery version 2"
    # ... and the evidence the second write could not supply is still there.
    assert (entry.run, entry.dataset, entry.generation) == ("D:/runs/run-2", "D:/sets/merged-v2", "v2")
    assert entry.recorded_at == "2026-10-14T05:00:00"


def test_every_resolved_weight_gets_its_own_entry(tmp_path):
    """One file, many weights: the entry is per *bytes*, so a machine with several trained weights
    keeps them all - and the output is ordered, so a rewrite of unchanged facts is a diff-free file.

    `now` is pinned to the same second on both writes to `one.pt` because `recorded_at` is itself
    one of the facts: left to the clock, the second write stamps a fresh second and the file differs
    for a reason that has nothing to do with the serialization this asserts. That made the test fail
    whenever the two calls straddled a second boundary - which is a loaded suite, not a broken
    writer - and the failure was a two-digit diff (`T...:05` against `T...:06`) in a file whose
    ordering and field set were both correct.
    """
    _recorded(_weight(tmp_path / "one.pt", b"one"), "local merge merged-v2",
              now="2026-10-14T04:00:00")
    _recorded(_weight(tmp_path / "two.pt", b"two"), "snc-grocery version 2",
              now="2026-10-14T04:05:00")

    assert len(source_index.read_index()) == 2
    before = source_index.index_path().read_bytes()
    _recorded(_weight(tmp_path / "one.pt", b"one"), "local merge merged-v2",
              now="2026-10-14T04:00:00")
    assert source_index.index_path().read_bytes() == before


# --- the two refusals --------------------------------------------------------------------


def test_a_blank_source_is_refused_by_the_writer(tmp_path):
    """Refused here rather than stored, because a stored blank is indistinguishable from no entry
    when it is read - so the caller, which knows *why* it has no answer, is the one that reports it."""
    weight = _weight(tmp_path / "models" / "w.pt")
    for blank in ("", "   ", None, 7):
        with pytest.raises(ValueError):
            source_index.record_source(weight, blank)  # type: ignore[arg-type]
    assert source_index.read_index() == {}


def test_recording_needs_the_weight_to_exist(tmp_path):
    """The entry is the weight's identity, so a path with nothing behind it has nothing to key."""
    with pytest.raises(FileNotFoundError):
        source_index.record_source(tmp_path / "absent.pt", "local merge merged-v2")


def test_the_index_can_be_pointed_out_of_the_machine_s_data_directory(tmp_path, monkeypatch):
    """The override the suite itself runs on (`tests/conftest.py`), so the mechanism the tests
    exercise is the shipped one: resolved at call time, since a path frozen at import would ignore
    a caller that set it afterwards."""
    default = source_index.INDEX_PATH
    before = default.read_bytes() if default.is_file() else b""

    moved = tmp_path / "elsewhere" / "index.json"
    monkeypatch.setenv(source_index.ENV_VAR, str(moved))
    assert source_index.index_path() == moved
    weight = _weight(tmp_path / "w.pt")
    assert _recorded(weight) == moved
    assert source_index.lookup(weight).source == "local merge merged-v2"

    # And the default is untouched: the redirect is a redirect, not a second location. Compared
    # before and after rather than read as "empty", because that file is real machine state -
    # `train_model.py --install` on this checkout legitimately writes entries into it, so an
    # "is empty" assertion would fail on a machine that has trained something (or, as here, on one
    # a throwaway script has written a dead entry into) rather than on the redirect being wrong.
    monkeypatch.delenv(source_index.ENV_VAR)
    assert source_index.index_path() == default
    assert (default.read_bytes() if default.is_file() else b"") == before
