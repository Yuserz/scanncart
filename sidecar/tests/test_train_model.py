"""Tests for `train_model.py`: the export check, the run verdict, and the drop-in.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
import pytest
import generate_version
import generations
import label_classes
import train_model
import workspace

# Rebuilt from the modules that own each fact, not copied by value.
V2 = generations.V2
RUN_NAME = V2.run_name
WEIGHT_NAME = V2.weight_name
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "MODEL_TRAINING.md"

from tests.dataset_tool_helpers import (
    _RESULTS_CSV,
    _fake_export,
    _finished_run,
    _report_with_distances,
)


class _Resp:
    """Just enough of an httpx response for `generate_version.main`."""

    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = {} if body is None else body

    def json(self):
        return self._body


# --------------------------------------------------------------------------
# 9. train_model - the export check, the run's verdict, and the drop-in
# --------------------------------------------------------------------------


def test_export_check_passes_on_the_roster(tmp_path):
    root = _fake_export(tmp_path / "export-v2")
    splits, problems = train_model.check_export(root)
    assert problems == []
    assert set(splits) == {"train", "valid", "test"}
    assert train_model.count_images(splits["train"]) == 3


def test_export_missing_a_class_is_a_problem_not_a_warning(tmp_path):
    """The failure this check exists for: a model trained against a short class list emits
    boxes under the wrong label with no error at all, so it has to stop before the GPU run.
    """
    names = sorted(label_classes.SLUG_TO_CLASS.values())[:-1]  # drop one
    root = _fake_export(tmp_path / "export-v2", names=names)
    _splits, problems = train_model.check_export(root)
    assert any("no class for" in p for p in problems)
    assert any(sorted(label_classes.SLUG_TO_CLASS.values())[-1] in p for p in problems)


def test_export_with_a_phantom_class_is_a_problem(tmp_path):
    root = _fake_export(tmp_path / "export-v2", names=["croutons"])
    _splits, problems = train_model.check_export(root)
    assert any("not v2 classes" in p for p in problems)


def test_export_check_reports_a_missing_split_and_an_empty_one(tmp_path):
    root = _fake_export(tmp_path / "export-v2", splits=("train", "valid"), per_split=0)
    _splits, problems = train_model.check_export(root)
    assert any("no test/images directory" in p for p in problems)
    assert any("train/images is empty" in p for p in problems)


def test_find_split_dirs_accepts_the_nested_extraction(tmp_path):
    nested = _fake_export(tmp_path / "nested", nested=True)
    assert set(train_model.find_split_dirs(nested)) == {"train", "valid", "test"}


def test_missing_export_explains_what_to_download(tmp_path, capsys):
    """The operator-facing half of the check: this is the state the repo is in today (no
    version 2 generated yet), so the message is what a reader meets first."""
    splits, problems = train_model.check_export(tmp_path / "nope")
    assert splits == {}
    assert any("no export at" in p for p in problems)

    code = train_model.main(["--export-dir", str(tmp_path / "nope")])
    out = capsys.readouterr().out
    assert code == 2
    assert "YOLOv11 PyTorch" in out  # named the way the version page names it
    assert "not at the .zip" in out


def test_a_run_refuses_while_the_pass_leaves_machine_only_decisions(tmp_path, capsys):
    """The human pass is a precondition of the *run*, not only of the acceptance measurement.

    A weight's unread boxes in `valid`/`test` do not make the rows wrong to train on - they are the
    set's frames like any other. They make every number the run produces unquotable (`test` becomes
    a measurement of the annotator) and the checkpoint itself suspect (`valid` is the split the run
    *selects* on). `accept_v2` refuses to report over that set, and finding it out there costs the
    GPU hours, the checkpoint and the weight's name - so the same verdict is asked here first.

    `yolo` fails the test if it is called at all: the refusal has to land before anything loads.
    """
    root = _fake_export(tmp_path / "export-v2")
    # `train` is tolerated by the rule and only named: the gate is about the two measured splits.
    _report_with_distances(root, [("t1.jpg", "close")], machine_only={"valid": 2, "train": 5})

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(tmp_path / "models"),
            "--yes",
        ],
        yolo=lambda *_a, **_k: pytest.fail("nothing may load while the pass gate is dirty"),
    )

    out = capsys.readouterr().out
    assert code == 2
    assert "Nothing was trained: the human pass is not finished" in out
    assert "2 machine-only decision(s) in `valid`" in out
    assert "no machine-only decisions in valid or test" not in out
    assert "--status" in out and "build_dataset" in out
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "models").exists()


def test_the_pass_gate_catches_a_clean_zero_from_before_the_pass_was_worked(tmp_path, capsys):
    """The stale-zero case, at this door as well as at `accept_v2`'s.

    `machine_only_by_split` is a measurement of one moment - the build's. A set built before any
    decision was saved reports zero in both gate splits while the annotator holds hundreds of
    unread boxes, so the counts alone read *clean*; the stamp over the `provenance.json` they were
    counted from is the half that can see the difference, which is why the gate is the report's two
    part verdict rather than a check on one field. Same teeth at the CLI: one answer, exit 2.
    """
    root = _fake_export(tmp_path / "export-v2")
    _report_with_distances(root, [], machine_only={})

    ok, lines = train_model.pass_gate(root)
    assert ok is True
    assert "clean" in lines[-1]

    # Somebody saves a decision in the annotator - the store moves on, the set's account of itself
    # does not, and the zero it reports is no longer about the state on disk.
    (tmp_path / "annotations-v2" / workspace.PROVENANCE_NAME).write_text(
        json.dumps({"milo_0001.jpg": {"machine_only": True}}), encoding="utf-8"
    )
    ok, lines = train_model.pass_gate(root)
    assert ok is False
    assert any("changed after this set was built" in line for line in lines)

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--val",
        ],
        yolo=lambda *_a, **_k: pytest.fail("nothing may load while the pass gate is dirty"),
    )
    assert code == 2
    assert "Nothing was trained" in capsys.readouterr().out


def test_a_set_with_no_merge_report_is_not_gated_and_says_so(tmp_path):
    """A Roboflow version export has no machine-only state to check - its labels came from that
    project's own annotator - so refusing there would block a path that is not dirty, only
    unanswerable. It says which of the two it is: silence would read as "the gate passed", and this
    is the one mode where the gate has not actually been asked anything.
    """
    root = _fake_export(tmp_path / "export-v2")

    ok, lines = train_model.pass_gate(root)

    assert ok is True
    assert len(lines) == 1
    assert "carries no" in lines[0] and "Not gated" in lines[0]


def test_cli_dry_run_checks_the_export_and_runs_nothing(tmp_path, capsys):
    """No --yes means no training and no file written outside the export, which is the
    only safe way to show a reviewer what the run would be."""
    root = _fake_export(tmp_path / "export-v2")
    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(tmp_path / "models"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Nothing ran" in out
    assert "yolo detect train" in out and f"model={train_model.BASE_MODEL}" in out
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "models").exists()


def test_cli_install_from_a_finished_run_copies_the_checkpoint(tmp_path, capsys):
    """The drop-in, at the CLI level: a completed run installs, and the resize_mode reminder
    comes with it - that field is what makes stretched training pay off."""
    root = _fake_export(tmp_path / "export-v2")
    run = tmp_path / "runs" / RUN_NAME
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"weights")
    (run / "results.csv").write_text(_RESULTS_CSV, encoding="utf-8")

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(tmp_path / "models"),
            "--run-dir",
            str(run),
            "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "models" / WEIGHT_NAME).read_bytes() == b"weights"
    assert "resize_mode: stretch" in out


def test_written_data_yaml_uses_absolute_paths_and_the_export_s_own_name_order(tmp_path):
    """The export's yaml is relative to wherever Roboflow expected the archive to be
    extracted. Rewriting it from absolute paths is what makes the run work from any cwd -
    and the *order* has to be the export's, because that is the model's output indexing.
    """
    import yaml

    roster = sorted(label_classes.SLUG_TO_CLASS.values())
    root = _fake_export(tmp_path / "export-v2", names=list(reversed(roster)))
    splits = train_model.find_split_dirs(root)
    written = train_model.write_data_yaml(root, splits)
    body = yaml.safe_load(written.read_text(encoding="utf-8"))

    assert body["names"] == list(reversed(roster))
    assert body["nc"] == len(roster)
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        assert Path(body[key]).is_absolute()
        assert Path(body[key]) == splits[split].resolve()
    # Written into the export, not over it: the original is the provenance record.
    assert written.name == "data.scanncart.yaml"


def _results(tmp_path, text: str = _RESULTS_CSV) -> Path:
    path = tmp_path / "results.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_results_csv_columns_are_stripped_and_rows_parsed(tmp_path):
    rows = train_model.read_results(_results(tmp_path))
    assert len(rows) == 3
    assert rows[-1]["metrics/mAP50-95(B)"] == 0.620
    assert rows[0]["epoch"] == 1.0


def test_verdict_fails_on_the_final_epoch_even_when_an_earlier_one_passed(tmp_path):
    """patience means the last epoch need not be the best one; the verdict deliberately
    quotes the end of the run, and `best_epoch` is reported beside it so the two are read
    together rather than the better number being cherry-picked.
    """
    rows = train_model.read_results(_results(tmp_path))
    passed, failed = train_model.verdict(rows)
    assert any("mAP50-95" in f for f in failed)
    # mAP50 clears its target in the same row, so the two are decided independently - and
    # the lookup must not have answered mAP50 with mAP50-95's value.
    assert any(p.startswith("mAP50(B) 0.910") for p in passed)
    assert not any("mAP50(B)" in f for f in failed)
    # The log's epoch column is 1-based (ultralytics writes `self.epoch + 1`), so the
    # 0.630 row is epoch 2 and must be reported as 2, not silently rewritten to 1.
    assert train_model.best_epoch(rows) == (2, 0.630)


def test_verdict_passes_when_both_targets_are_met(tmp_path):
    text = _RESULTS_CSV.replace("0.910, 0.620", "0.940, 0.700")
    rows = train_model.read_results(_results(tmp_path, text))
    passed, failed = train_model.verdict(rows)
    assert failed == []
    assert len(passed) == 2


def test_verdict_reports_a_missing_metric_rather_than_assuming_it():
    rows = [{"epoch": 1.0, "metrics/mAP50(B)": 0.95}]
    _passed, failed = train_model.verdict(rows)
    assert any("not in the run log" in f for f in failed)


def test_metric_matches_either_the_suffixed_or_bare_column_name():
    assert train_model.metric({"metrics/mAP50": 0.5}, "metrics/mAP50(B)") == 0.5
    assert train_model.metric({"metrics/mAP50(B)": 0.6}, "metrics/mAP50(B)") == 0.6
    assert train_model.metric({"metrics/mAP50-95(B)": 0.4}, "metrics/mAP50(B)") is None


def test_find_best_prefers_best_over_last(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"last")
    assert train_model.find_best(tmp_path).name == "last.pt"
    (weights / "best.pt").write_bytes(b"best")
    assert train_model.find_best(tmp_path).name == "best.pt"
    assert train_model.find_best(tmp_path / "empty") is None


def test_run_dir_suffixes_instead_of_overwriting_an_earlier_run(tmp_path):
    assert train_model.run_dir(tmp_path, RUN_NAME) == tmp_path / RUN_NAME
    (tmp_path / RUN_NAME).mkdir()
    assert train_model.run_dir(tmp_path, RUN_NAME) == tmp_path / f"{RUN_NAME}-2"
    (tmp_path / f"{RUN_NAME}-2").mkdir()
    assert train_model.run_dir(tmp_path, RUN_NAME) == tmp_path / f"{RUN_NAME}-3"


def test_install_copies_and_refuses_to_clobber(tmp_path):
    """The picker is keyed by filename, so an overwrite silently replaces the model a
    running app is configured with - and there is no history to roll back to.
    """
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    models = tmp_path / "models"

    target = train_model.install(src, models)
    assert target == models / WEIGHT_NAME
    assert target.read_bytes() == b"weights"

    src.write_bytes(b"newer")
    with pytest.raises(SystemExit):
        train_model.install(src, models)
    assert target.read_bytes() == b"weights"

    assert train_model.install(src, models, force=True).read_bytes() == b"newer"


def test_the_recorded_requirement_is_the_one_the_version_geometry_implies():
    """The requirement is derived from the version's preprocessing, not written out twice:
    `Stretch to 640` is what the export was generated with, and `stretch` is what the sidecar
    setting has to be. If a future generation changes the format, the mapping is the thing
    that has to change with it - and `REQUIRED_RESIZE_MODE` going `None` is how that surfaces
    instead of a confidently wrong requirement landing in the record.
    """
    fmt = generate_version.PREPROCESSING["resize"]["format"]
    assert fmt in generate_version.RESIZE_MODE_BY_FORMAT
    assert generate_version.REQUIRED_RESIZE_MODE == "stretch"
    # The mapping's *values* are the sidecar's `resize_mode` vocabulary, so the app-side test
    # (`test_models.py`) is what checks the one it produces is a value the settings PATCH
    # would accept - the contract belongs at that boundary, not in this file.
    # Roboflow's "Fill within" scales *and crops*, which the sidecar cannot reproduce - so it
    # is absent rather than mapped to the nearest-sounding value.
    assert "Fill within" not in generate_version.RESIZE_MODE_BY_FORMAT


def test_install_writes_the_record_beside_the_weights(tmp_path):
    """The requirement travels with the file, not in a `models/`-global manifest: the weight
    is the thing that gets copied to another machine, and a manifest left behind would lose
    the requirement at exactly the moment it is needed.
    """
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    models = tmp_path / "models"

    target = train_model.install(
        src, models, record=train_model.weight_record(V2, 2, "snc-grocery")
    )

    record = json.loads((models / f"{target.stem}.json").read_text(encoding="utf-8"))
    # No `model` field on purpose: the filename is the model, and a record that repeated it
    # would be a second answer that a copied file could make wrong.
    assert "model" not in record
    assert record["generation"] == V2.name
    assert record["resize_mode"] == "stretch"
    assert record["source"] == "snc-grocery version 2"
    assert record["installed_at"]
    # Beside the weight, named after it - that is the pairing the sidecar reads.
    assert target.with_suffix(".json").is_file()


def test_a_record_is_optional_so_install_still_works_alone(tmp_path):
    """`install()` is also the plain "copy this checkpoint in" path, and it has to keep
    working without a record - the reader treats a missing one as an unknown requirement."""
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    target = train_model.install(src, tmp_path / "models")
    assert target.read_bytes() == b"weights"
    assert not target.with_suffix(".json").exists()


def test_cli_install_records_and_announces_the_resize_mode(tmp_path, capsys):
    """The record is only half of it: the operator still has to set the field, so the run says
    which value and where the panel will check it."""
    root = _fake_export(tmp_path / "export-v2")
    models = tmp_path / "models"
    _finished_run(tmp_path / "runs", RUN_NAME)

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(models),
            "--version",
            "2",
            "--install",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "resize_mode: stretch" in out
    assert f"{WEIGHT_NAME[:-3]}.json" in out
    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    assert record["source"] == "snc-grocery version 2"


def test_install_creates_the_models_directory(tmp_path):
    """There was no `sidecar/models/` in this checkout at all, which is how a finished
    `best.pt` ends up never becoming a selectable model."""
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    models = tmp_path / "absent" / "models"
    assert train_model.install(src, models).is_file()
    assert models.is_dir()


def test_run_command_matches_the_hyperparameters_the_doc_quotes():
    """One source of truth for the run. Retuning either the tool or MODEL_TRAINING.md 6
    alone leaves the doc describing a run nobody performs."""
    doc = DOC.read_text(encoding="utf-8")
    for key, value in (
        ("epochs", train_model.EPOCHS),
        ("imgsz", train_model.IMGSZ),
        ("batch", train_model.BATCH),
        ("patience", train_model.PATIENCE),
    ):
        assert f"{key}={value}" in doc, f"MODEL_TRAINING.md does not quote {key}={value}"
    assert train_model.BASE_MODEL in doc

    hyper = train_model.Hyper()
    line = train_model.command_line(Path("d.yaml"), Path("runs"), RUN_NAME, hyper, "0")
    for token in (
        f"model={train_model.BASE_MODEL}",
        f"epochs={train_model.EPOCHS}",
        f"imgsz={train_model.IMGSZ}",
        f"name={RUN_NAME}",
        "device=0",
        f"workers={hyper.workers}",
    ):
        assert token in line
    # The printed command and the call that runs must not be able to disagree.
    kwargs = train_model.train_kwargs(Path("d.yaml"), Path("runs"), RUN_NAME, hyper, "0")
    assert kwargs["epochs"] == train_model.EPOCHS and kwargs["name"] == RUN_NAME
    assert kwargs["device"] == "0"


def test_the_augmentation_is_the_docs_table_and_the_run_applies_it():
    """4's table, not ultralytics' defaults - which are *not* the table: `degrees` is 0.0 there
    and `mosaic` is 1.0. A run that inherits them trains a model the doc does not describe, and
    nothing in the log says so; that is what this pins.
    """
    doc = DOC.read_text(encoding="utf-8")
    assert "±15°" in doc, "MODEL_TRAINING.md 4 no longer quotes the rotation this file uses"
    assert "±20%" in doc, "...nor the brightness range"

    defaults = train_model.augmentation_kwargs(train_model.Hyper())
    assert defaults["degrees"] == 15.0 and defaults["fliplr"] > 0 and defaults["hsv_v"] == 0.2
    # The three exclusions are explicit zeros rather than absent keys: the record has to say a
    # vertical flip was *decided against*, not that nobody remembered it.
    assert defaults["flipud"] == 0.0 and defaults["mosaic"] == 0.0 and defaults["erasing"] == 0.0

    hyper = train_model.Hyper()
    kwargs = train_model.train_kwargs(Path("d.yaml"), Path("runs"), RUN_NAME, hyper, "0")
    assert all(kwargs[key] == value for key, value in defaults.items())
    # And the printed command is the run that happens, degrees included.
    line = train_model.command_line(Path("d.yaml"), Path("runs"), RUN_NAME, hyper, "0")
    assert "degrees=15" in line and "scale=0.5" in line

    # The flags reach the run through the same dict, so a retune cannot land in one and not the
    # other.
    tuned = train_model.augmentation_kwargs(train_model.Hyper(degrees=0.0, scale=0.2))
    assert (tuned["degrees"], tuned["scale"]) == (0.0, 0.2)


def test_the_record_carries_the_size_and_augmentation_the_run_used(tmp_path):
    """`resize_mode` was the first half of "the app feeds the model the wrong geometry"; `imgsz`
    is the second. Neither is recoverable from the checkpoint, and both are silent when wrong.
    """
    record = train_model.weight_record(
        V2,
        2,
        "snc-grocery",
        imgsz=960,
        augmentation=train_model.augmentation_kwargs(train_model.Hyper(degrees=0.0)),
    )

    assert record["imgsz"] == 960
    assert record["augmentation"]["degrees"] == 0.0
    # Only the table's own keys: a caller passing an unrelated dict cannot smuggle a field in.
    assert set(record["augmentation"]) == set(train_model.AUGMENTATION_KEYS)

    # Absent rather than zero when there is nothing to say: a re-install of a run from before
    # this field existed must write the same bytes, and `0` would read as a trained size.
    bare = train_model.weight_record(V2)
    assert "imgsz" not in bare and "augmentation" not in bare


def test_the_run_reads_back_its_own_arguments_rather_than_this_commands_flags(tmp_path):
    """`--val` and `--install` are separate commands, so their `--imgsz` is whatever was typed
    this time. Ultralytics' `args.yaml` is the run's own record of what it trained with."""
    run = tmp_path / RUN_NAME
    run.mkdir()
    (run / "args.yaml").write_text(
        "imgsz: 960\ndegrees: 0.0\nscale: 0.5\nmosaic: 1.0\nname: scanncart-grocery-v2\n",
        encoding="utf-8",
    )

    args = train_model.run_args(run)
    assert args["imgsz"] == 960 and args["degrees"] == 0.0
    fallback = train_model.augmentation_kwargs(train_model.Hyper())
    assert train_model.trained_value(args, "imgsz", train_model.IMGSZ) == 960
    # A real run's keys are read off it - including the ones this file's defaults disagree with,
    # which is the whole point of asking the run - and a key it does not carry keeps this
    # command's value.
    assert train_model.trained_value(args, "mosaic", fallback["mosaic"]) == 1.0
    assert train_model.trained_value(args, "hsv_v", fallback["hsv_v"]) == fallback["hsv_v"]
    # A string is a fallback too (coercing "640" would be guessing a unit), and `True` would
    # otherwise arrive as `1.0` - a wrong number rather than a missing one.
    assert train_model.trained_value({"imgsz": "640"}, "imgsz", 960) == 960.0
    assert train_model.trained_value({"degrees": True}, "degrees", 15.0) == 15.0

    # A run predating this read, a hand-made directory and a corrupt file are all `{}`.
    assert train_model.run_args(tmp_path / "absent") == {}
    (run / "args.yaml").write_text("not: [valid: yaml: at all", encoding="utf-8")
    assert train_model.run_args(run) == {}


class _Resp:
    """Just enough of an httpx.Response for the export path: a status, a JSON body, bytes."""

    def __init__(self, status_code=200, body=None, content=b""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.content = content
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def test_export_link_reads_only_a_ready_body():
    """202-with-progress and 200-with-link are told apart by the body, because the status
    code does not distinguish "accepted" from "ready" on its own."""
    assert train_model.export_link({"ready": False, "progress": 0.4}) is None
    assert train_model.export_link({}) is None
    assert train_model.export_link(None) is None
    assert train_model.export_link({"export": {"link": "https://x/y.zip"}}) == "https://x/y.zip"
    assert train_model.export_progress({"ready": False, "progress": 0.4}) == 0.4
    assert train_model.export_progress({}) is None


def _export_zip(tmp_path) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train/images/a.jpg", b"\xff\xd8\xff")
        zf.writestr("data.yaml", "nc: 8\nnames: [a, b, c, d, e, f, g, h]\n")
    return buf.getvalue()


def test_download_export_polls_until_the_link_appears(tmp_path):
    """Three answers in order: building, building, ready. Then the archive, extracted into
    the destination - which is what makes the whole chain runnable without a click."""
    calls: list[str] = []
    payload = _export_zip(tmp_path)

    def fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith(".zip"):
            return _Resp(200, content=payload)
        if len([c for c in calls if c == url]) < 3:
            return _Resp(202, {"ready": False, "progress": 0.5})
        return _Resp(200, {"export": {"link": "https://files/x.zip"}})

    slept: list[float] = []
    dest = tmp_path / "export-v2"
    train_model.download_export(
        "snc-grocery", 2, dest, "key", get=fake_get, sleep=slept.append
    )

    assert slept == [10, 10]
    assert (dest / "train" / "images" / "a.jpg").is_file()
    assert (dest / "data.yaml").is_file()
    # The whole URL, workspace included. Asserting only the project and version is what let a
    # shadowed `WORKSPACE` (the dataset *directory*) reach the live API as a path segment: the
    # request still carried "/snc-grocery/2/" while addressing `C:\codes\...\datasets`.
    assert calls[0] == (
        f"https://api.roboflow.com/{label_classes.WORKSPACE}/snc-grocery/2/{train_model.EXPORT_FORMAT}"
    )
    assert str(train_model.DATASET_ROOT) != label_classes.WORKSPACE
    assert ":\\" not in calls[0]  # no filesystem path leaked into the URL


def test_download_export_refuses_a_bad_request_instead_of_looping(tmp_path):
    """A wrong --format or version is an error body, not a 202 that polls for 15 minutes."""
    def fake_get(url, **kwargs):
        return _Resp(404, {"error": "format not available"})

    with pytest.raises(SystemExit):
        train_model.download_export("p", 2, tmp_path / "d", "k", get=fake_get, sleep=lambda _s: None)


def test_download_export_gives_up_on_a_stuck_export(tmp_path):
    with pytest.raises(SystemExit):
        train_model.download_export(
            "p", 2, tmp_path / "d", "k",
            get=lambda url, **kw: _Resp(202, {"ready": False, "progress": 0.1}),
            sleep=lambda _s: None,
            timeout_s=0,
        )


def test_extract_zip_refuses_a_member_that_escapes_the_destination(tmp_path):
    """A zip is an untrusted input; `../../` in a member name writes outside the directory
    it was extracted into."""
    import zipfile

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(SystemExit):
        train_model.extract_zip(archive, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()


def test_cli_skips_a_download_that_is_already_there(tmp_path, capsys):
    root = _fake_export(tmp_path / "export-v2")
    code = train_model.main([
        "--export-dir", str(root), "--download", "--version", "2",
        "--run-project", str(tmp_path / "runs"), "--models-dir", str(tmp_path / "models"),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "already downloaded" in out


def test_cli_download_without_a_version_says_which_flag_is_missing(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        train_model.main(["--export-dir", str(tmp_path / "x"), "--download"])
    assert "--version" in str(excinfo.value)


def test_train_model_does_not_import_ultralytics_at_module_level():
    """Same shape as app/hardware.py's lazy torch import: the tool has to be importable,
    and its pure functions testable, without torch on the path."""
    source = Path(train_model.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^(from|import)\s+ultralytics", source, re.MULTILINE)
    assert re.search(r"^\s+from ultralytics import", source, re.MULTILINE)




def test_the_record_carries_the_sets_own_geometry_when_it_declares_one():
    """A local `fit` build is letterbox whatever the generation's default is - and a set that
    declares nothing keeps the generation's requirement."""
    assert train_model.weight_record(generations.V2, resize_mode="letterbox")["resize_mode"] == "letterbox"
    assert train_model.weight_record(generations.V2)["resize_mode"] == generations.V2.resize_mode


def _installed(models, stem, generation):
    models.mkdir(exist_ok=True)
    (models / f"{stem}.pt").write_bytes(b"w")
    (models / f"{stem}.json").write_text(json.dumps({"generation": generation}), encoding="utf-8")


def test_default_weights_finds_a_renamed_weight_by_its_record(tmp_path):
    """`--name` means `models/scanncart-grocery-v2.pt` may not exist; the tools' `--generation v2`
    then means the v2 weight the app runs, found by its record."""
    models = tmp_path / "models"
    _installed(models, "scanncart-grocery-v1", "v1")
    _installed(models, "scanncart-grocery-v2-stretch", "v2")
    assert train_model.default_weights(generations.V2, models).name == "scanncart-grocery-v2-stretch.pt"

    _installed(models, "scanncart-grocery-v2-letterbox", "v2")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"active_model": "models/scanncart-grocery-v2-letterbox.pt"}))
    assert train_model.default_weights(generations.V2, models, settings).name == (
        "scanncart-grocery-v2-letterbox.pt"
    )
    settings.write_text(json.dumps({"active_model": "yolo11n.pt"}))
    with pytest.raises(SystemExit, match="pass --weights"):
        train_model.default_weights(generations.V2, models, settings)


def test_default_weights_prefers_the_generations_own_name_when_it_exists(tmp_path):
    models = tmp_path / "models"
    _installed(models, "scanncart-grocery-v2", "v2")
    _installed(models, "scanncart-grocery-v2-letterbox", "v2")
    assert train_model.default_weights(generations.V2, models).name == "scanncart-grocery-v2.pt"


def test_install_can_name_a_second_weight_of_one_generation(tmp_path):
    source = tmp_path / "best.pt"
    source.write_bytes(b"w")
    first = train_model.install(source, tmp_path / "models", name="scanncart-grocery-v2-stretch.pt")
    second = train_model.install(source, tmp_path / "models", name="scanncart-grocery-v2-letterbox.pt")
    assert first.name != second.name and first.exists() and second.exists()
