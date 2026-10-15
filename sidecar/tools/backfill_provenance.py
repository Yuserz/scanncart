#!/usr/bin/env python
"""Name the dataset an installed weight came from, for the records `--install` could not fill.

`train_model.py --install` writes `source` beside the weight - what the Admin Panel prints in
parentheses after a weight's name, and the one place "which dataset produced these weights" is
written down anywhere. It takes that from `--version` when one was passed, and otherwise derives it
from the run it is installing (`train_model.run_dataset_source`), which is how a weight trained from
a **local merge** gets an answer at all: `build_dataset.py` is not a version, so there is no number
to name and the merge is named instead. This tool is for the records written before that was true and
for the installs whose run is not at hand - three installed weights are in the first state today
(`scanncart-grocery-v2-stretch`, `-v2-letterbox`, `-v2-letterbox-synth`), and until this runs the
panel shows a name and no provenance for any of them.

**The index first, because it needs nothing but the weight.** `app/source_index.py` is where a run's
answer is written down while that run is still on disk - by `train_model.py` as a run finishes, and by
this tool when it repairs a record from one - keyed by the weight's bytes. It is read before the run
project is looked at at all, and that order is the point: matching bytes against the run project needs
the run project to still exist *and* to still hold a checkpoint identical to these weights, and a
pruned or deleted run is precisely the state a record is most likely to be empty in, since the read
that fills it happens a command later than the run itself. A weight the index can name is placed with
no workspace at all - the weights, the records and the index are the whole input.

When the index has nothing, the fact may still be recoverable, and it is somewhere else. Every run
ultralytics writes keeps the arguments it ran with (`args.yaml`), and `train_model` writes that run's
`data=` itself, from the directory it was pointed at - so the run that produced a weight *names the
dataset it trained on*, in its own file, months after the terminal it was typed into is gone. One
thing is needed to get from a weight to that file, and it is what this tool has that a bare
`--install` does not:

* **Which run.** The weight's bytes: `weights_sha256`, the same identity `--val` attaches to a
  measurement, matched against every `runs/*/weights/{best,last}.pt`. A filename cannot answer this -
  the run directories for these weights are `scanncart-grocery-v2-2`, `-2-3`, `-2-4`, because they
  were trained before `--variant` existed, while the *weights* were installed as
  `-v2-stretch`/`-v2-letterbox`/`-v2-letterbox-synth`. A rule that read the name would be right
  about these three and wrong the first time one is renamed.
* **Which dataset.** Read off the run it just matched by `train_model.run_dataset_source` - the same
  call `--install` makes when it is holding the run - so the name and the reasons it can have no
  answer are one implementation rather than two that could drift. That rule resolves the run's
  `args.yaml` `data=` key to the directory it names (the one `write_data_yaml` wrote
  `data.scanncart.yaml` into), and calls it *`local merge <directory>`* while the directory still
  carries `merge_report.json` - `workspace.MERGE_REPORT_NAME`, the provenance `build_dataset.py`
  writes, and this repo's own word for such a set (`sidecar/tools/CLAUDE.md` calls `build_dataset.py`
  "the local merge"). Its `generation` is cross-checked against the record's, because the record's
  generation is what the class list is judged against and two artifacts naming two generations is one
  of them being wrong.

**Refusing is the feature.** Four states have no answer here, and each one is reported with the
reason and left unwritten rather than filled in with something plausible: neither the index nor any
run holds these bytes (a weight copied from another machine), two runs hold them (a copied run
directory is not evidence about which one trained what), the recorded dataset directory is gone, and
a dataset directory with no merge report - which is a *generated version's* export, where the version
number exists only in the `--version` that would have been passed to `--install` and is not in the
run. Each of those prints what it looked for, so the next step is a decision and not a guess.

Two weights are deliberately **not** touched. One whose record already names its dataset, because
this is idempotent and a second run must be a no-op. And one with no record at all: writing one here
would flip `app/models.read_record`'s `recorded` flag from false to true, and that flag is what makes
the app raise its *`auto` gives letterbox* entry for a weight whose geometry nothing knows - so a
provenance write would silence a live warning about the resize requirement. `--install` is the writer
for a weight with no record, and the Admin Panel's *Record it now* button is the one for its geometry.

**Merges, never replaces**, exactly as `app/models.record_requirement` does and for the same reason:
the rest of a record is evidence only the training run could supply (what `--val` measured, the
weight's own class list, the augmentation it trained with), so a whole-file write would delete all of
it to add one field. It writes through the one serializer, `train_model.write_record`. A repair made
from a run also writes the same fact into the index (`train_model.write_source_index`), so the run is
not needed a second time - the two halves of one repair, in one command, and only ever in the writing
path: `--dry-run` leaves both files alone.

    sidecar/.venv/Scripts/python.exe sidecar/tools/backfill_provenance.py            # write
    sidecar/.venv/Scripts/python.exe sidecar/tools/backfill_provenance.py --dry-run  # print only

Reads every checkpoint under the run project to match on bytes, so it takes a few seconds; sizes are
compared before hashes, which is what keeps that to the handful of files that could match. Exit 0 when
nothing is left unplaced, 1 when a record still has an empty `source` and no run could be named for
it - so a weight whose provenance is genuinely unrecoverable fails a script rather than passing
quietly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import train_model

# `app/source_index.py` is the runtime's, and this reaches into it the same way `accept_v2.py`
# reaches into `app.models`: `sidecar/` is not on `sys.path` when this file is run as a script.
if str(train_model.SIDECAR_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(train_model.SIDECAR_ROOT))

from app import source_index  # noqa: E402 - after the path insert above, which is the point of it

# The same two directories `train_model` installs into and trains under, imported rather than
# retyped: a default here that drifted from the trainer's would report on weights nobody installed.
DEFAULT_MODELS_DIR = train_model.DEFAULT_MODELS_DIR
DEFAULT_RUN_PROJECT = train_model.DEFAULT_RUN_PROJECT

#: The suffix a weight can carry. `models/README.md` says a `.pt` **or** an `.onnx` directly in the
#: directory is selectable, and a record beside either one is the same file - so both are looked for
#: rather than only the `.pt` `train_model` installs, which would report an `.onnx` as record-less.
WEIGHT_SUFFIXES = (".pt", ".onnx")


@dataclass(frozen=True)
class Verdict:
    """What was found for one installed weight, and what to write because of it.

    `source` empty means *leave this record alone*, and `note` is the sentence that says why - so the
    printed report and the write are one decision rather than two that could disagree.

    `run`/`generation` are the run these bytes were matched to, when the answer came from one. They
    are carried out of here because the write has two halves rather than one: the record beside the
    weights, and the entry in the source index (`train_model.write_source_index`) that stops this
    machine needing that run again - a repair that had to be repeated after the run was pruned would
    be the gap this tool exists to close.
    """

    weight: Path
    record_path: Path
    source: str
    note: str
    run: Path | None = None
    generation: str = ""

    @property
    def write(self) -> bool:
        """Whether there is anything to write, which is exactly whether a dataset was named."""
        return bool(self.source)


def describe_entry(entry: source_index.IndexEntry) -> str:
    """One index entry as a sentence fragment: which run wrote it, from where, and when.

    Everything the entry carries is named, so a claim read back out of the index can be checked
    rather than trusted - the run it was written from, the dataset that run read, and the moment it
    was written. The source itself is not repeated here: that is the value being recorded, and the
    caller prints it as such beside this.
    """
    where = Path(entry.run).name if entry.run else "an unnamed run"
    source = f" from {entry.dataset}" if entry.dataset else ""
    when = f" on {entry.recorded_at}" if entry.recorded_at else ""
    return f"written when {where} trained these bytes{source}{when}"


def weights_in(models_dir: Path) -> list[tuple[Path, Path, object]]:
    """`(weight, its record path, the record)` for every selectable weight in `models_dir`.

    Driven by the *weights*, because they are what the picker lists and what the operator sees a
    name and no provenance for - a record with no weight beside it is a leftover from a deleted
    `.pt`, not a weight missing a fact, and it is not iterated here.

    The suffix rule is `app/models.installed_models`': an entry directly in the directory, not
    hidden, ending `.pt` or `.onnx`. Which is also why the record is looked up by stem rather than
    the reverse: the two share a filename, and pairing them any other way would need a rule the app
    does not have.
    """
    found: list[tuple[Path, Path, object]] = []
    if not models_dir.is_dir():
        return found
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() or entry.name.startswith("."):
            continue
        if not entry.name.endswith(WEIGHT_SUFFIXES):
            continue
        record_path = entry.with_suffix(".json")
        found.append((entry, record_path, train_model.read_json(record_path)))
    return found


def runs_holding(weight: Path, run_project: Path) -> list[Path]:
    """Every run under `run_project` holding a checkpoint byte-identical to `weight`.

    Files are sized first and hashed second: the size is a `stat` and the hash reads the whole
    checkpoint, and a run project with a dozen runs holds a dozen `best.pt`s that could not possibly
    match. What survives the size check is what gets read.

    A *set* of run directories: `best.pt` and `last.pt` of one run can be the same file, and that is
    one candidate rather than an ambiguity.
    """
    digest = train_model.weights_sha256(weight)
    size = weight.stat().st_size
    found: set[Path] = set()
    for candidate in sorted(run_project.glob("*/weights/*.pt")):
        try:
            if candidate.stat().st_size != size:
                continue
        except OSError:
            continue
        if train_model.weights_sha256(candidate) == digest:
            found.add(candidate.parent.parent)
    return sorted(found)


def verdict_for(
    weight: Path,
    record_path: Path,
    record: object,
    run_project: Path,
    index: Path | None = None,
) -> Verdict:
    """What to write for one installed weight, and the sentence explaining it."""
    if not isinstance(record, dict):
        if not record_path.is_file():
            # Not a finding this tool can fix, and deliberately not one it works around: writing a
            # record here would flip `app/models.read_record`'s `recorded` flag to true, and that
            # flag is what makes the app raise its *`auto` gives letterbox* entry for a weight whose
            # geometry nothing knows. A provenance write would silence a live warning.
            return Verdict(
                weight,
                record_path,
                "",
                "no record beside it - nothing to merge a source into, and writing one from scratch "
                "would silence the app's `auto` gives letterbox entry for a weight nothing recorded "
                "the geometry of. `--install` (or the Admin Panel's Record it now) is that writer.",
            )
        # A record that exists and will not parse is not the same as no record: a person wrote
        # something there, and this tool has no business deciding what it meant.
        return Verdict(
            weight,
            record_path,
            "",
            f"{record_path.name} does not read as a JSON object - nothing to merge into. "
            "`--install` is the writer that rebuilds it, and it needs the training run.",
        )
    current = record.get("source")
    if isinstance(current, str) and current.strip():
        return Verdict(weight, record_path, "", f"already names its dataset: {current!r}")
    generation = record.get("generation")
    generation = generation if isinstance(generation, str) else ""

    # The index first, and before the run project is asked about at all - which is the whole reason
    # it exists. Matching runs needs the run project to still exist *and* to still hold a checkpoint
    # byte-identical to these weights; this needs nothing but the weight. That was the one case this
    # tool could not serve (a pruned or deleted run, a copy made on another machine), and it is the
    # case a record is most likely to be in, since the read that would have filled it happens a
    # command later than the run itself.
    entry = source_index.lookup(weight, index)
    if entry is not None:
        return Verdict(
            weight,
            record_path,
            entry.source,
            f"the source index ({index or source_index.index_path()}) has these bytes: "
            + describe_entry(entry),
        )

    matches = runs_holding(weight, run_project)
    if not matches:
        # Both places a machine could know this have now been asked, and that is why the sentence
        # names them both: "no run holds these bytes" on its own reads as "the answer is not on this
        # machine", when what it means is that neither the run project nor the index this machine
        # keeps has ever seen them.
        return Verdict(
            weight,
            record_path,
            "",
            "no run under the run project holds these bytes, and the source index has nothing for "
            "them either, so which dataset trained them is not recoverable from here - a deleted "
            "run and a weight copied from another machine look exactly alike. Re-installing with "
            "--version records it if the run still exists elsewhere.",
        )
    if len(matches) > 1:
        return Verdict(
            weight,
            record_path,
            "",
            "two runs hold these bytes ("
            + ", ".join(run.name for run in matches)
            + ") - a copied run directory is not evidence of which one trained these weights. "
            "Delete or rename the copy, then re-run.",
        )

    run = matches[0]
    # The reading itself - the run's `data=` to a named dataset, and the reasons it can have no
    # answer - lives in `train_model`, beside the `args.yaml` reader and the record writer `--install`
    # uses. Two callers reach one rule from opposite ends (that command holds the run; this tool holds
    # only a weight's bytes) and the sentences come back unchanged, so what is printed here is the
    # same report a bare `--install` prints.
    source, note = train_model.run_dataset_source(run, generation)
    return Verdict(
        weight, record_path, source, note, run=run if source else None, generation=generation
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    ap.add_argument(
        "--run-project",
        default=str(DEFAULT_RUN_PROJECT),
        help="the runs directory a weight is traced back through (default %(default)s)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be recorded and write nothing",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models_dir = Path(args.models_dir).expanduser()
    run_project = Path(args.run_project).expanduser()

    index = source_index.index_path()
    print(f"weights: {models_dir}")
    print(f"runs:    {run_project}")
    print(f"index:   {index}")
    if not models_dir.is_dir():
        print()
        print(f"  there is no {models_dir} - pass --models-dir")
        return 2

    weights = weights_in(models_dir)
    # The run project is asked about only if something still needs it, and "needs it" is now a
    # smaller set than "has no source": the index can name a weight with nothing but its bytes. A
    # missing directory is a missing input - named rather than reported as a weight nothing matches,
    # which would send the operator looking at the weights - but it is not one when every record
    # already says where it came from *or* the index remembers the run that produced them, and
    # refusing there would make this tool unusable on the machine it was written for: a checkout
    # whose workspace has been wiped while the weights, the records and the index survive (which is
    # the state it is meant to leave behind).
    #
    # The bytes are hashed here and once more in `verdict_for`, which is the price of keeping the
    # index lookup in the function that decides the verdict rather than threading it through as an
    # argument - and it is paid only for a weight whose record names no dataset, which is the state
    # this tool is run for in the first place.
    wanted = [
        weight
        for weight, _record_path, record in weights
        if isinstance(record, dict)
        and not str(record.get("source") or "").strip()
        and source_index.lookup(weight, index) is None
    ]
    if wanted and not run_project.is_dir():
        print()
        print(f"  there is no {run_project} - nothing to match {len(wanted)} weight(s) against.")
        print("  Pass --run-project, or check the workspace is where you think it is.")
        return 2

    unplaced = 0
    written = 0
    print()
    for weight, record_path, record in weights:
        verdict = verdict_for(weight, record_path, record, run_project, index)
        print(f"{weight.name}")
        if verdict.write:
            if args.dry_run:
                print(f"  would record: source = {verdict.source!r}   ({verdict.note})")
            else:
                merged = dict(record) if isinstance(record, dict) else {}
                merged["source"] = verdict.source
                train_model.write_record(weight, merged)
                written += 1
                print(f"  recorded:     source = {verdict.source!r}   ({verdict.note})")
                print(f"                -> {verdict.record_path.name}")
                # The second half of the repair, and the half that outlives the run: the same fact
                # into the machine's source index, so `--install` and the Admin Panel can name these
                # bytes without the run project at all. Only when a *run* answered - an entry read
                # back out of the index is already there, and writing it again would only refresh a
                # timestamp nobody asked to change.
                if verdict.run is not None:
                    indexed, index_note = train_model.write_source_index(
                        weight, verdict.source, verdict.run, verdict.generation
                    )
                    if indexed is not None:
                        print(f"                index: {index_note}")
            continue
        print(f"  left alone:   {verdict.note}")
        # Only a *missing provenance* is a failure. A record already naming its dataset, or one that
        # will not parse, is a state this tool is not the fix for, and failing on them would turn an
        # idempotent re-run into a red result.
        if isinstance(record, dict) and not str(record.get("source") or "").strip():
            unplaced += 1

    print()
    if args.dry_run:
        print(f"nothing written (--dry-run); {unplaced} record(s) still without a source")
        return 1 if unplaced else 0
    print(f"{written} record(s) updated")
    if unplaced:
        print(f"{unplaced} record(s) still without a source - see the reasons above")
        return 1
    print("every record that could be named is named")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
