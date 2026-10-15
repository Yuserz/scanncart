"""Tests for `backfill_provenance.py` - naming the dataset a weight came from, after the fact.

The tool exists for the three installed weights whose records carry an empty `source` (the v2
family, trained from local merges, so there was no Roboflow version number `--install` could write
there). What it does is a chain - the weight's bytes find the run, the run's own `args.yaml` names
the dataset, the merge report names it as a build - and what this module is mostly about is the
*four* places that chain has no answer: no run holds the bytes, two do, the dataset is gone, and a
dataset directory with no merge report. Each of those has to be a refusal with a reason rather than
a plausible string, because a wrong `source` is a claim about provenance that nothing downstream can
falsify - `accept_v2` and the panel both read it as fact.

The mutation the tests check is the *record beside the weight*, never the weight: the tools install
weights, and a repair tool that could replace one would be a much larger thing than this.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

import backfill_provenance as backfill
import train_model
import workspace
from app import source_index


def _weight(models_dir: Path, stem: str, payload: bytes = b"weights", record: dict | None = None) -> Path:
    """An installed weight, with the record `--install` writes beside it.

    `payload` is what the checkpoint's bytes *are* here - the whole identity the tool matches on, so
    a test says which run produced this weight by writing the same bytes in both places.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    weight = models_dir / f"{stem}.pt"
    weight.write_bytes(payload)
    if record is not None:
        train_model.write_record(weight, record)
    return weight


def _v2_record(**over) -> dict:
    """A record in the shape three real ones have: a generation, a requirement, an empty source."""
    record = {
        "generation": "v2",
        "resize_mode": "letterbox",
        "source": "",
        "installed_at": "2026-10-07T01:58:24",
        "class_names": ["milo"],
    }
    record.update(over)
    return record


def _run(
    run_project: Path,
    name: str,
    dataset: str | Path,
    payload: bytes = b"weights",
    recorded: str | None = None,
) -> Path:
    """A finished run: the checkpoint, and the `args.yaml` ultralytics leaves beside it.

    Only `data` is written, because `data` is the one key this tool reads out of a run - the run's
    own answer to "what did you train on", which is what makes the derivation evidence rather than
    inference from a filename. `dataset` is the directory the run was *pointed at*; what a real run
    records is the yaml `write_data_yaml` wrote inside it, which is what `recorded` overrides for the
    tests about the path itself.
    """
    run = run_project / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(payload)
    if recorded is None:
        recorded = str(Path(dataset) / workspace.SCANNCART_DATA_YAML_NAME)
    (run / "args.yaml").write_text(yaml.safe_dump({"data": recorded}), encoding="utf-8")
    return run


def _dataset(root: Path, name: str, generation: str = "v2", report: bool = True) -> Path:
    """A built set: the directory, and the merge report `build_dataset.py` leaves in it.

    `report=False` is the shape a Roboflow *export* directory has instead - which is a case the tool
    has to refuse rather than call a merge, because the version number an export would need is not
    written anywhere on this side.
    """
    dataset = root / name
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / workspace.SCANNCART_DATA_YAML_NAME).write_text("names: [milo]\n", encoding="utf-8")
    if report:
        (dataset / workspace.MERGE_REPORT_NAME).write_text(
            json.dumps({"generation": generation}), encoding="utf-8"
        )
    return dataset


def _main(tmp_path: Path, *extra: str, models_dir: Path | None = None, runs: Path | None = None) -> int:
    return backfill.main(
        [
            "--models-dir",
            str(models_dir if models_dir is not None else tmp_path / "models"),
            "--run-project",
            str(runs if runs is not None else tmp_path / "runs"),
            *extra,
        ]
    )


def _source(weight: Path) -> str:
    return json.loads(weight.with_suffix(".json").read_text(encoding="utf-8"))["source"]


def _record(weight: Path) -> dict:
    return json.loads(weight.with_suffix(".json").read_text(encoding="utf-8"))


# --- the derivation ---------------------------------------------------------------------


def test_the_bytes_of_the_weight_find_the_run_and_the_run_names_the_dataset(tmp_path):
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "scanncart-grocery-v2-2", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2"


def test_the_bytes_decide_which_run_not_the_filename(tmp_path):
    """The rule the three real weights need, and the reason it has to be the bytes.

    These were trained before `--variant` existed, so the run directories are named for the
    generation with ultralytics' `-2`/`-3` suffix (`scanncart-grocery-v2-3`) while the *weights* were
    installed under a descriptive name (`scanncart-grocery-v2-letterbox`). A tool that matched on the
    name would pick the run whose name is closest here - and that run trained on something else.
    """
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-letterbox", record=_v2_record())
    _run(
        tmp_path / "runs",
        "scanncart-grocery-v2-letterbox",
        str(_dataset(tmp_path / "datasets", "merged-v2-decoy")),
        payload=b"a different checkpoint",
    )
    _run(
        tmp_path / "runs",
        "scanncart-grocery-v2-3",
        str(_dataset(tmp_path / "datasets", "merged-v2-letterbox")),
    )

    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2-letterbox"


def test_it_merges_rather_than_replaces_and_writes_the_documented_format(tmp_path):
    """`source` is one field of a file full of evidence only a training run could supply.

    A whole-file write would delete what `--val` measured and the class list the head indexes - and
    the class list is the one nothing can recover afterwards.
    """
    record = _v2_record(
        **{
            "validation": [{"split": "test", "floor": 0.85, "measured_at": "x"}],
            "unknown_key": {"kept": True},
        }
    )
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=record)
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path) == 0
    written = _record(weight)
    assert written["source"] == "local merge merged-v2"
    assert written["resize_mode"] == "letterbox"
    assert written["class_names"] == ["milo"]
    assert written["validation"] == record["validation"]
    assert written["unknown_key"] == {"kept": True}

    text = weight.with_suffix(".json").read_text(encoding="utf-8")
    assert text.startswith('{\n  "generation"')  # `write_record`'s indent, one owner
    assert text.endswith("}\n")
    assert not list(weight.parent.glob("*.tmp"))


def test_a_second_run_is_a_no_op(tmp_path):
    """Idempotent by the field it writes: the second run finds a non-empty `source` and stops."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path) == 0
    after_first = weight.with_suffix(".json").read_bytes()
    assert _main(tmp_path) == 0
    assert weight.with_suffix(".json").read_bytes() == after_first


def test_dry_run_says_what_it_would_write_and_writes_nothing(tmp_path, capsys):
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")))
    before = weight.with_suffix(".json").read_bytes()

    assert _main(tmp_path, "--dry-run") == 0
    assert weight.with_suffix(".json").read_bytes() == before
    printed = capsys.readouterr().out
    assert "would record: source = 'local merge merged-v2'" in printed


# --- the four refusals ------------------------------------------------------------------


def test_a_weight_no_run_produced_is_reported_and_left_empty(tmp_path, capsys):
    """A weight copied from another machine, or one whose run was since deleted - the same state
    from here, and neither is a licence to guess."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")), payload=b"elsewhere")

    assert _main(tmp_path) == 1
    assert _source(weight) == ""
    assert "no run under the run project holds these bytes" in capsys.readouterr().out


def test_two_runs_holding_the_same_bytes_are_refused_rather_than_guessed(tmp_path, capsys):
    """A copied run directory is not evidence about which one trained what: the tool cannot tell a
    copy from two genuinely identical checkpoints, so it says so and writes nothing."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "scanncart-grocery-v2-2", str(_dataset(tmp_path / "datasets", "merged-v2")))
    _run(tmp_path / "runs", "scanncart-grocery-v2-2-copy", str(_dataset(tmp_path / "datasets", "other")))

    assert _main(tmp_path) == 1
    assert _source(weight) == ""
    printed = capsys.readouterr().out
    assert "two runs hold these bytes" in printed
    assert "scanncart-grocery-v2-2-copy" in printed


def test_a_dataset_that_is_no_longer_on_disk_is_reported(tmp_path, monkeypatch, capsys):
    """The run still says what it trained on; the *set* is what carries the file that identifies it.

    `WORKSPACE` is pointed at an empty directory so the name fallback cannot find a set that happens
    to share the name - which is a real substitution this tool does make (the workspace is
    relocatable, so an absolute path recorded months ago is tried first and the *name* after it), and
    which `test_a_moved_workspace_is_still_found_by_the_dataset_name` pins. What is refused here is
    the case where no directory anywhere answers to the name at all.
    """
    monkeypatch.setattr(workspace, "WORKSPACE", tmp_path / "elsewhere")
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", tmp_path / "datasets" / "merged-v9")

    assert _main(tmp_path) == 1
    assert _source(weight) == ""
    printed = capsys.readouterr().out
    assert "is not on disk any more" in printed
    assert "merged-v9" in printed


def test_an_export_with_no_merge_report_is_refused_with_the_remedy(tmp_path, capsys):
    """The other dataset a run can train on, and the one this tool cannot name: a version's export
    carries no merge report, and the version number lives in the `--version` passed to `--install`
    rather than in the run."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(
        tmp_path / "runs",
        "run",
        str(_dataset(tmp_path / "datasets", "scanncart-grocery-v2", report=False)),
    )

    assert _main(tmp_path) == 1
    assert _source(weight) == ""
    assert "--install --version N" in capsys.readouterr().out


def test_a_report_and_a_record_naming_different_generations_are_refused(tmp_path, capsys):
    """The record's generation is what the class list is judged against, so two artifacts naming two
    generations is one of them being wrong - and a person has to settle which."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v1", generation="v1")))

    assert _main(tmp_path) == 1
    assert _source(weight) == ""
    printed = capsys.readouterr().out
    assert "'v1'" in printed and "'v2'" in printed


def test_a_matching_report_whose_generation_predates_the_field_is_still_placed(tmp_path):
    """Only a disagreement between two *present* values is a finding - the same reading
    `dataset_doctor` takes of a report that cannot carry a field yet."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    dataset = _dataset(tmp_path / "datasets", "merged-v2")
    (dataset / workspace.MERGE_REPORT_NAME).write_text("{}", encoding="utf-8")
    _run(tmp_path / "runs", "run", str(dataset))

    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2"


# --- the records it deliberately does not touch ---------------------------------------


def test_a_record_that_already_names_its_dataset_is_reported_and_left_alone(tmp_path, capsys):
    """No run project here, on purpose: nothing needs matching, so its absence is not an input the
    tool is missing - the same reason a wiped workspace does not fail a fully-recorded checkout."""
    weight = _weight(
        tmp_path / "models",
        "scanncart-grocery-v1",
        record=_v2_record(generation="v1", source="scanncart-grocery version 1"),
    )
    before = weight.with_suffix(".json").read_bytes()

    assert _main(tmp_path) == 0
    assert weight.with_suffix(".json").read_bytes() == before
    assert "already names its dataset: 'scanncart-grocery version 1'" in capsys.readouterr().out


def test_a_weight_with_no_record_is_reported_and_not_given_one(tmp_path, capsys):
    """Writing a record here would flip `app/models.read_record`'s `recorded` flag, and that flag is
    what makes the app raise its `auto` gives letterbox entry - so a provenance write would silence
    a live warning about the geometry."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=None)
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path) == 0
    assert not weight.with_suffix(".json").exists()
    assert "no record beside it" in capsys.readouterr().out


def test_a_record_that_will_not_parse_is_reported_and_left_alone(tmp_path, capsys):
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    weight.with_suffix(".json").write_text("{ this was half written", encoding="utf-8")
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path) == 0
    assert weight.with_suffix(".json").read_text(encoding="utf-8") == "{ this was half written"
    assert "does not read as a JSON object" in capsys.readouterr().out


# --- resolving the path a run recorded --------------------------------------------------


def test_the_recorded_data_path_is_resolved_from_either_directory_a_run_started_in(
    tmp_path, monkeypatch
):
    """One generation was trained twice from two working directories, and its runs recorded the same
    dataset as `sidecar\\data\\...` and `data\\...` - both relative, both true."""
    monkeypatch.setattr(workspace, "SIDECAR_ROOT", tmp_path / "sidecar")
    monkeypatch.setattr(workspace, "REPO_ROOT", tmp_path)
    dataset = _dataset(tmp_path / "sidecar" / "data" / "datasets", "merged-v2")

    yaml_name = workspace.SCANNCART_DATA_YAML_NAME
    relative = str(Path("data") / "datasets" / "merged-v2" / yaml_name)
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "from-sidecar", tmp_path, recorded=relative)
    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2"

    # ...and the same set again, recorded the way a run started at the repo root spells it.
    other = _weight(tmp_path / "models", "scanncart-grocery-v2-other", record=_v2_record())
    _run(
        tmp_path / "runs",
        "from-root",
        tmp_path,
        recorded=str(Path("sidecar") / "data" / "datasets" / "merged-v2" / yaml_name),
        payload=b"the second weight",
    )
    assert _main(tmp_path) == 0
    assert _source(other) == "local merge merged-v2"
    assert dataset.is_dir()


def test_a_moved_workspace_is_still_found_by_the_dataset_name(tmp_path, monkeypatch):
    """The workspace is relocatable (`SCANNCART_DATASET_ROOT`) and a set can be rebuilt elsewhere, so
    an absolute path recorded months ago is only worth trying first - not the only thing tried."""
    moved = tmp_path / "elsewhere"
    monkeypatch.setattr(workspace, "WORKSPACE", moved)
    dataset = _dataset(moved, "merged-v2")

    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", tmp_path / "gone" / "merged-v2")

    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2"
    assert dataset.is_dir()


def test_a_run_that_recorded_the_folder_instead_of_its_yaml_is_still_placed(tmp_path):
    """The two spellings name one directory, and which one a run wrote is not a fact about the set."""
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    dataset = _dataset(tmp_path / "datasets", "merged-v2")
    _run(tmp_path / "runs", "run", dataset, recorded=str(dataset))

    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2"


# --- the listing ------------------------------------------------------------------------


def test_only_weights_are_listed_so_a_stray_record_is_not_a_missing_provenance(tmp_path, capsys):
    """Driven by the weights, like the picker: a `<name>.json` whose `.pt` was deleted is a leftover,
    and reporting it as a record missing a source would send the operator after a file that is not
    part of the install."""
    models = tmp_path / "models"
    _weight(models, "scanncart-grocery-v2-stretch", record=_v2_record())
    (models / "scanncart-grocery-v2-gone.json").write_text(json.dumps(_v2_record()), encoding="utf-8")
    (models / "notes.txt").write_text("not a weight", encoding="utf-8")
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path) == 0
    printed = capsys.readouterr().out
    assert "scanncart-grocery-v2-gone" not in printed
    assert "notes.txt" not in printed


@pytest.mark.parametrize(("missing", "flag"), [("models", "--models-dir"), ("runs", "--run-project")])
def test_a_missing_directory_is_a_missing_input_rather_than_an_unplaced_weight(
    tmp_path, missing, flag, capsys
):
    """Exit 2, not 1: "I could not look" and "I looked and these weights cannot be placed" send the
    operator to two different places, and an empty run project otherwise reports every weight as
    unmatchable."""
    _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", _dataset(tmp_path / "datasets", "merged-v2"))
    (tmp_path / missing).rename(tmp_path / f"{missing}-moved")

    assert _main(tmp_path) == 2
    assert flag in capsys.readouterr().out


# --- the index, which needs no run at all --------------------------------------------


def test_the_index_names_a_weight_whose_run_project_is_gone_entirely(tmp_path, capsys):
    """The case this tool could not serve before the index existed, and the one a record is most
    likely to be empty in: the run was pruned, or the set it read was rebuilt elsewhere, so matching
    bytes has nothing to match against. The entry was written while that run was still here
    (`train_model.write_source_index`), so the weight needs neither the run project nor a workspace -
    and the run project is *absent* here on purpose, which without the entry would be the exit-2
    "I could not look" above rather than a repair.
    """
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    source_index.record_source(
        weight,
        "local merge merged-v2",
        run=tmp_path / "runs" / "scanncart-grocery-v2-2",
        dataset=tmp_path / "datasets" / "merged-v2",
        now="2026-10-14T04:00:00",
    )

    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2"
    printed = capsys.readouterr().out
    # Everything the claim rests on is named, because nothing beside the weight corroborates it: a
    # sentence read out of an index that did not say which run wrote it could not be checked at all.
    assert str(source_index.index_path()) in printed
    assert "scanncart-grocery-v2-2" in printed
    assert "merged-v2" in printed


def test_a_repair_made_from_a_run_is_written_into_the_index_so_it_is_needed_once(tmp_path):
    """One command, two writes: the record beside the weights, and the entry that stops this machine
    needing the run again. Without the second half, pruning the run would put the record back where
    it started - so this deletes the run project *and* the record's source and runs the tool again,
    which now has to place the weight with nothing on disk but the weights and the index.
    """
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "scanncart-grocery-v2-2", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path) == 0
    entry = source_index.lookup(weight)
    assert entry is not None
    assert (entry.source, Path(entry.run).name) == ("local merge merged-v2", "scanncart-grocery-v2-2")

    shutil.rmtree(tmp_path / "runs")
    train_model.write_record(weight, _v2_record())

    assert _main(tmp_path) == 0
    assert _source(weight) == "local merge merged-v2"


def test_a_dry_run_writes_no_index_entry_either(tmp_path, capsys):
    """`--dry-run` is the whole tool's promise, so it covers both halves of the repair: an entry
    written while a repair was only being *described* would be a fact about a write nobody made.
    """
    weight = _weight(tmp_path / "models", "scanncart-grocery-v2-stretch", record=_v2_record())
    _run(tmp_path / "runs", "run", str(_dataset(tmp_path / "datasets", "merged-v2")))

    assert _main(tmp_path, "--dry-run") == 0
    assert source_index.read_index(source_index.index_path()) == {}
    assert "would record: source = 'local merge merged-v2'" in capsys.readouterr().out


def test_a_record_that_already_names_its_dataset_is_never_second_guessed_by_the_index(
    tmp_path, capsys
):
    """The index answers *where the record is silent*, and the order is what makes it evidence
    rather than an override: a record naming a version stays as it is even when this machine can
    trace the same bytes to a merge - a copy of a fact, not a better one.
    """
    weight = _weight(
        tmp_path / "models",
        "scanncart-grocery-v2-stretch",
        record=_v2_record(source="snc-grocery version 2"),
    )
    source_index.record_source(weight, "local merge merged-v2")

    assert _main(tmp_path) == 0
    assert _source(weight) == "snc-grocery version 2"
    assert "already names its dataset" in capsys.readouterr().out
