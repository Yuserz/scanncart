"""Which dataset trained a weight, written down while the run can still be asked.

A weight's provenance is recorded in the file beside it (`models/<stem>.json`, written by
`tools/train_model.py --install`), and that record is what the Admin Panel prints in parentheses
after a weight's name. It has two sources, and both of them can be gone by the time the record is
written: the `--version` is a number the operator has to still remember, and the derived half reads
the *run* a checkpoint came from (its `args.yaml` `data=` key, resolved to a directory that still
carries its merge report). Delete or prune the run, rebuild the set somewhere else, and the install
reports a gap - which is the state three installed weights were in, and the reason
`tools/backfill_provenance.py` exists at all. But that tool matches a weight's bytes against the run
project too, so it fails in exactly the case that needs it most: *the run is gone*.

This module is the third place, and the one that survives: a small machine-local index keyed by the
weight's own bytes, written at the moment the answer is known - when a run is being trained or
installed - so that "which dataset trained these weights" outlives the run, the workspace and the
terminal. The bytes are the key because they are the only stable identity a checkpoint has: the same
three v2 weights are installed as `-v2-stretch`/`-v2-letterbox`/`-v2-letterbox-synth` while their
run directories are `scanncart-grocery-v2-2`/`-2-3`/`-2-4`, so a name-keyed index would be right
about today's files and wrong the first time one is renamed - and a `--force` re-install can put
different bytes under a name this file already knows. It is also the identity `--val`'s measurement
carries (`weights_sha256` in `val_metrics.json`), so one weight's provenance and its score hang off
one fact rather than two that could disagree.

Why it lives in `app/`: the runtime has to *read* it - `/api/models` reports the index's answer for a
weight whose record is silent - and the tools have to *write* it, and `app/` is the only package
both halves already import from. `tools/` is deliberately never imported by the runtime
(`CLAUDE.md`), while `accept_v2.py`, `ab_live.py` and the annotator reach into `app.models` the same
way this is reached. The app itself never writes an entry: the fact comes from a training run, which
only a tool is holding.

It is **evidence, not authority.** The record beside the weights is what the app reads first, and
this answers only where that record is silent - it cannot overrule a record, and it is not a second
place to keep a fact the record already holds. That is also why it is machine-local state in
`sidecar/data/` (gitignored) rather than something committed beside the code: a `.pt` copied to
another machine carries its record and not this file, and an index that claimed otherwise would be
answering about a run that machine never ran.

Two writers, both tools, and the same entry:

* **`train_model.py`**, as a run finishes - the moment the dataset is a fact on disk. This is the
  write that makes the index worth having: it happens *before* anyone runs `--install`, so an
  install months later still has an answer after the run is pruned.
* **`backfill_provenance.py`**, when it matches a weight's bytes to a run - so a machine whose
  records predate this file catches up as it is repaired, and its repair is not needed twice.

and two readers, both of which say what they are reading:

* **`--install`**, as its last resort when neither `--version` nor the run answers, and
* **`app.models.installed_models`**, which reports the index's answer beside a weight whose record
  names no dataset, so the panel can show a fact that is known on this machine and not yet written
  down (the *`Record it now`* button's confirmation names the tool that writes it).

Nothing here raises on a bad file. This is read to *report* something - the panel loads it on every
open, and a backfill run reads it for every weight it inspects - so a missing file, a file that is
not JSON, a JSON that is not an object and one junk entry all cost what they are worth, which is
nothing, rather than the listing or the repair.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

#: `sidecar/data/` is the sidecar's machine-local state (gitignored: `data/`), which is what this
#: is - a record of the runs *this* machine trained, next to the settings and the detection log
#: rather than beside the weights, because it must not travel with a copy of a `.pt`.
INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "source_index.json"

#: The key the entries hang under. Spelled once, because the writer emits it and the reader looks
#: for it, and a rename here that missed one of the two would report an empty index rather than an
#: error - the one failure mode this file is written to avoid, since an empty index is a normal
#: state (a machine that has trained nothing yet).
WEIGHTS_KEY = "weights"

#: The environment variable that moves the index, named after `workspace.SCANNCART_DATASET_ROOT` for
#: the tools' own relocatable data. Two callers need it: a machine whose `sidecar/data/` lives
#: elsewhere, and - the reason it exists at all - the tests, which must not write the real file. A
#: suite that declared an install would otherwise leave entries in the developer's index for weights
#: that never existed, and the file is written by *every* install, not only by a test that opted in.
ENV_VAR = "SCANNCART_SOURCE_INDEX"

#: The file's own shape, so a later reader can tell an old file from a new one instead of reading a
#: field that moved as absent. Written but not enforced: an unknown version is read for what it has,
#: because refusing would turn a rollback into a blank index.
INDEX_VERSION = 1

#: What a `recorded_at` looks like - `train_model`'s own format, so the two read as one timestamp
#: family rather than two formats a reader has to know about.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

_HEX_DIGITS = frozenset("0123456789abcdef")


def index_path() -> Path:
    """The index file to read or write: `INDEX_PATH`, or `SCANNCART_SOURCE_INDEX` when it is set.

    Resolved at *call* time rather than at import, so a caller that changed the environment after
    this module loaded - every test, through the suite's own fixture - is honoured instead of a path
    frozen when the process started.
    """
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser() if override else INDEX_PATH


def weights_sha256(weights: Path) -> str:
    """The identity of a weight: the bytes of the file, hashed.

    The owner of this rule rather than a copy of it, because two things are keyed by it - `--val`'s
    measurement, which is how a stale score is kept away from a re-trained checkpoint, and this
    index's entries - and a second implementation that hashed a prefix, or read in different chunks,
    would give one file two identities. The failure would be a lookup that finds nothing, which is
    indistinguishable from a weight nothing has recorded.

    `train_model.weights_sha256` calls this, so the tools' own call sites keep the name they were
    written with and there is still one implementation of what a weight *is*.

    Raises `OSError` for a file that cannot be read, like any file read: every caller here either
    catches it or has already checked the file exists.
    """
    digest = hashlib.sha256()
    with Path(weights).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IndexEntry:
    """One weight's provenance, as this machine wrote it down.

    `source` is the name the record uses - `local merge merged-v2`, `<project> version N` - which is
    the string the panel prints and the one `backfill_provenance.py` merges in. It is *not* computed
    here: it comes from the rule that names a dataset (`train_model.dataset_source`), through the
    run that was in hand. Storing the answer rather than a pointer is the whole point of the file,
    since the directory that rule read may be gone by the time anyone asks again.

    `dataset` and `run` are the inputs that answer came from, kept as evidence rather than as a
    second way to reach it: they let a reader say which run trained these bytes and from where, and
    they are what makes a surprising name checkable by hand. A reader that re-derived `source` from
    them would lose the entry exactly when the index is needed - after those directories are gone -
    which is the mistake this file exists to fix.
    """

    sha256: str
    source: str
    size: int = 0
    dataset: str = ""
    run: str = ""
    generation: str = ""
    recorded_at: str = ""


def _read_json(path: Path) -> object:
    """`path`'s JSON, or None for every way it can fail to be there or to parse."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text(raw: object) -> str:
    """`raw` when it is a string, else `""` - the empty value meaning *not recorded*.

    Same reading as `app/models._read_class_names`: a field that is present but not a string says
    nothing, and inventing a string for it would put this program's words in a record of what a run
    said.
    """
    return raw if isinstance(raw, str) else ""


def _spelled(value: object) -> str:
    """`value` as the string the file holds: a `Path` is spelled out, never dropped.

    Not `_text`'s string-or-blank rule, which is for reading a field a person may have written,
    because the direction is the other one here: a caller holding a directory passes a `Path`, and
    that is the answer rather than a malformed field. Turning it into `""` would lose evidence with
    nothing to notice it - which is the failure mode this file exists to remove.
    """
    return "" if value is None else str(value)


def _count(raw: object) -> int:
    """`raw` as a non-negative int, else 0. `bool` is excluded: it is an `int`, and `True` would
    arrive as a size of 1 byte."""
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else 0


def _is_digest(key: object) -> bool:
    """Whether `key` could be a `weights_sha256` - 64 hex digits, case-insensitively.

    Checked rather than assumed because the key is the identity: an entry filed under a typo'd or
    truncated digest is unreachable, and keeping it would make `read_index` claim an entry for bytes
    nothing can ever look up. Lower-cased on the way in, since a hand edit may paste an uppercase
    one and `hashlib` answers lower case.
    """
    return isinstance(key, str) and len(key) == 64 and set(key.lower()) <= _HEX_DIGITS


def _entry_from(key: str, raw: object) -> IndexEntry | None:
    """One stored entry, or None when it says nothing usable. Never raises.

    A missing `source` drops the entry rather than being kept as a blank, and that is the load-bearing
    drop: the reason to read this file is that a weight's provenance is missing, so an entry that
    answers with a blank would look like a lookup that worked and leave the gap in place.
    """
    if not isinstance(raw, dict):
        return None
    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        return None
    digest = key.lower()
    return IndexEntry(
        sha256=digest,
        source=source.strip(),
        size=_count(raw.get("size")),
        dataset=_text(raw.get("dataset")),
        run=_text(raw.get("run")),
        generation=_text(raw.get("generation")),
        recorded_at=_text(raw.get("recorded_at")),
    )


def read_index(path: Path | None = None) -> dict[str, IndexEntry]:
    """Every usable entry, keyed by the bytes it describes. Never raises.

    Same contract as `app/models.read_record`, and for the same reason: this is read to *report*
    something, so a missing file, an unparseable one and one junk entry each cost what they are worth
    - nothing - rather than the Admin Panel's model list or a repair run. A hand edit is a normal way
    for a file like this to arrive (plain JSON, in a gitignored directory, written to be checkable),
    which makes the tolerance a rule rather than defensiveness.
    """
    body = _read_json(path or index_path())
    weights = body.get(WEIGHTS_KEY) if isinstance(body, dict) else None
    entries: dict[str, IndexEntry] = {}
    for key, raw in (weights if isinstance(weights, dict) else {}).items():
        if not _is_digest(key):
            continue
        entry = _entry_from(key, raw)
        if entry is not None:
            entries[key] = entry
    return entries


def lookup(weight: Path, path: Path | None = None) -> IndexEntry | None:
    """The entry for these exact bytes, or None. Never raises.

    The file is read first and the checkpoint hashed second, so a machine with no index - the
    normal state before its first training run - pays nothing at all for the question. The hash is
    the expensive half (a checkpoint is megabytes, not kilobytes), which is also why a caller should
    only ask when it needs the answer: `installed_models` asks for a weight whose *record* is silent
    and not for every weight on disk.
    """
    entries = read_index(path)
    if not entries:
        return None
    try:
        digest = weights_sha256(weight)
    except OSError:
        return None
    return entries.get(digest)


def _entry_payload(entry: IndexEntry) -> dict:
    """One entry as the dict written out, omitting fields that say nothing.

    Omitted rather than written empty, which is this repo's rule everywhere a record is written
    (`train_model.weight_record`): a field present and blank is a claim, an absent one is not, and
    it keeps a re-write of unchanged facts a byte-identical file rather than a diff.
    """
    payload: dict[str, object] = {"source": entry.source}
    if entry.size:
        payload["size"] = entry.size
    for key, value in (
        ("dataset", entry.dataset),
        ("run", entry.run),
        ("generation", entry.generation),
        ("recorded_at", entry.recorded_at),
    ):
        if value:
            payload[key] = value
    return payload


def record_source(
    weight: Path,
    source: str,
    *,
    dataset: str | Path = "",
    run: str | Path = "",
    generation: str = "",
    path: Path | None = None,
    now: str = "",
) -> Path:
    """Write `source` against these bytes, merging with whatever the file already holds.

    The one writer, shared by the two tools that can have the answer - `train_model.py` while a run
    is in hand, `backfill_provenance.py` when it has matched a record to a run - so the key, the
    indentation and the merge cannot differ between them. Same serialization rule as
    `train_model.write_record`: `json.dumps(..., indent=2)` plus a trailing newline, sorted keys, and
    a temporary file `os.replace`d into place. Through a temporary file for the same reason too, and
    one step sharper here: a truncated index reads as *no index at all*, which is a normal state, so
    the loss would be silent - and what it would lose is the only copy of a fact whose run may
    already be gone.

    **Merging per field, not replacing the entry.** The same bytes reach this twice - once when the
    run is trained (which knows the dataset directory the run read) and once when it is installed
    (which knows the `--version` the operator passed, a claim nothing on disk can produce, and it
    *should* win the `source` field, because that is what the record beside the weights will say and
    two answers to one question is how a panel and a record start disagreeing). A field this call
    cannot fill is left as it was, so the second write never erases the run the first one knew.

    Raises `ValueError` for a blank `source`: a writer that has no answer must not put a blank in the
    file, because the reader cannot tell it from no entry at all - and every caller here has the
    empty case in hand and can say so instead.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("the index records a named dataset, so `source` cannot be blank")
    if not Path(weight).is_file():
        raise FileNotFoundError(f"no weights to index at {weight}")

    target = path or index_path()
    digest = weights_sha256(weight)
    existing = read_index(target).get(digest)
    entry = IndexEntry(
        sha256=digest,
        source=source.strip(),
        size=Path(weight).stat().st_size,
        # `_spelled` on each, and `str` accepted for the path-shaped fields: the writers hold
        # directories, and a caller spelling a `Path` is stating the fact rather than failing to.
        dataset=_spelled(dataset) or (existing.dataset if existing else ""),
        run=_spelled(run) or (existing.run if existing else ""),
        generation=generation or (existing.generation if existing else ""),
        recorded_at=now or time.strftime(TIMESTAMP_FORMAT),
    )

    entries = read_index(target)
    entries[digest] = entry
    payload = {
        "version": INDEX_VERSION,
        WEIGHTS_KEY: {key: _entry_payload(entries[key]) for key in sorted(entries)},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target
