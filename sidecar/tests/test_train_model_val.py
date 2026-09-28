"""Tests for `train_model.py --val`: the per-class number, and the two ways to get it wrong.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
import pytest
import build_dataset
import generations
import train_model
import workspace

# Rebuilt from the modules that own each fact, not copied by value.
V2 = generations.V2
RUN_NAME = V2.run_name
WEIGHT_NAME = V2.weight_name
VAL_NAME = train_model.val_name(V2)
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "MODEL_TRAINING.md"
CHECKLIST = REPO_ROOT / "docs" / "CAPTURE_CHECKLIST.md"

from tests.dataset_tool_helpers import (
    _FakeBox,
    _FakeMetrics,
    _NAMES,
    _export_with_names,
    _fake_export,
    _finished_run,
    _manifest_with_distances,
    _report_with_distances,
)


def _built_set(root: Path, per_split: dict[str, list[tuple[str, str]]]) -> Path:
    """A merged set in the shape `build_dataset.py` writes it: frames filed per split, a label
    beside each, and the per-frame `distances` map in its own `merge_report.json`.

    `per_split` is `split -> [(filename, distance)]`, so a test states the counts the reader has to
    join - the report is keyed by name and the split is a directory, and the grid's passes have to
    marry the two. `_export_with_names` supplies the frames, labels and class list; the report is
    what makes it a *built set* rather than a download.
    """
    dataset = _export_with_names(
        root, {split: [name for name, _ in frames] for split, frames in per_split.items()}
    )
    _report_with_distances(
        dataset, [(name, distance) for frames in per_split.values() for name, distance in frames]
    )
    return dataset


# --- --val: the per-class number, and the two ways to get it wrong ---------------------


def test_per_class_recall_reads_r_positionally_against_ap_class_index():
    """The bug this test exists for, in one line: `r[i]` belongs to `ap_class_index[i]`.

    Classes 1 and 2 have no ground truth in the split, so ultralytics omits them - and a
    report that zipped `names` against `box.r` would put 0.95 (milo) on century-tuna and
    then call the whole thing a pass. Nothing about that output looks wrong.
    """
    box = _FakeBox(index=[0, 3], recall=[0.90, 0.95])
    metrics = _FakeMetrics(_NAMES, box, counts=[10, 0, 0, 12])

    rows = train_model.per_class_recall(metrics)
    assert [(name, recall) for name, recall, _n in rows] == [
        ("bear-brand", 0.90),
        ("century-tuna", None),
        ("lucky-me", None),
        ("milo", 0.95),
    ]
    assert [n for _name, _r, n in rows] == [10, 0, 0, 12]


def test_a_single_scored_class_is_not_dropped_by_numpy_truthiness():
    """The live probe's find, pinned. Ultralytics answers with numpy arrays, and `box.r` for
    a split whose only scored class recalls 0.0 is `array([0.0])` - which is *falsy* because
    a one-element array is truthy by its contents. `list(box.r or [])` thus reported
    "nothing was measured" for a class that had been measured and scored zero: the loudest
    result this step can produce, discarded quietly. A fake built from Python lists cannot
    catch this - `[0.0]` is truthy - which is exactly why the probe was worth running.
    """
    np = pytest.importorskip("numpy")
    box = _FakeBox(index=np.array([0]), recall=np.array([0.0]))
    metrics = _FakeMetrics({0: "bear-brand"}, box, counts=np.array([10]))

    rows = train_model.per_class_recall(metrics)
    assert rows == [("bear-brand", 0.0, 10)]

    passed, failed, unmeasured, values = train_model.recall_report(rows)
    assert failed == ["bear-brand 0.000 < 0.85 (n=10)"]
    assert passed == [] and unmeasured == []
    assert values == {"bear-brand": 0.0}


def test_a_class_the_split_never_asked_about_is_not_a_zero():
    """A missing class has no recall, and 0.0 is not the same answer: one means go capture
    images for the split, the other means go capture images for the class."""
    box = _FakeBox(index=[0, 3], recall=[0.90, 0.95])
    metrics = _FakeMetrics(_NAMES, box, counts=[10, 0, 0, 12])
    passed, failed, unmeasured, values = train_model.recall_report(train_model.per_class_recall(metrics))

    assert passed == ["bear-brand 0.900 >= 0.85 (n=10)", "milo 0.950 >= 0.85 (n=12)"]
    assert failed == []
    assert "century-tuna" not in values and "lucky-me" not in values
    assert len(unmeasured) == 2
    assert all("no ground-truth instances" in line for line in unmeasured)


def test_recall_report_puts_a_measured_miss_under_the_floor():
    """A class the model predicted badly *was* measured, so it fails rather than skipping -
    even though it is the near-empty cell you would most expect to read as unmeasured."""
    box = _FakeBox(index=[0, 1], recall=[0.62, 0.99], map50=0.8, map_=0.6, mp=0.8, mr=0.7)
    metrics = _FakeMetrics(_NAMES, box, counts=[4, 30, 0, 0])
    passed, failed, unmeasured, values = train_model.recall_report(train_model.per_class_recall(metrics))

    assert failed == ["bear-brand 0.620 < 0.85 (n=4)"]
    assert passed == ["century-tuna 0.990 >= 0.85 (n=30)"]
    assert len(unmeasured) == 2
    assert values["bear-brand"] == 0.62


def test_the_floor_is_the_one_the_doc_states():
    """6 sets 0.85 for *every* class. The number lives in code and in the doc, and only one
    of them can be edited silently."""
    doc = DOC.read_text(encoding="utf-8")
    assert f"≥ {train_model.RECALL_FLOOR:.2f} for *every* class" in doc


def _val_block(split="test", recall=0.9, weights_hash="hash-a", floor=0.85):
    return train_model.validation_record(
        [("bear-brand", recall, 10), ("lucky-me", None, 0)],
        split,
        {"mAP50": 0.88},
        floor,
        weights_hash,
    )


def test_the_measurement_carries_the_split_and_the_floor_it_was_judged_against(tmp_path):
    """A recall on its own is not a result: `test` is the acceptance split and `valid` is the
    one training selected on, and the floor is what turns a number into a verdict. Both travel
    with the numbers, because whatever reads them later was not there when they were made.
    """
    block = _val_block()
    assert block["split"] == "test"
    assert block["floor"] == 0.85
    assert block["weights_sha256"] == "hash-a"
    assert [c["name"] for c in block["per_class"]] == ["bear-brand", "lucky-me"]
    # `None`, not 0.0: the class the split never asked about is recorded as unmeasured, so
    # the distinction survives into a file that outlives the terminal it was printed to.
    assert block["per_class"][1]["recall"] is None


def test_a_measurement_of_other_weights_is_not_attached(tmp_path):
    """The reason the hash is here. `--val` and `--install` are separate commands, and
    re-training into the same run directory replaces `best.pt` in place - so a measurement of
    the previous checkpoint is the ordinary case, not the exotic one, and attaching it would
    show an operator a score these weights never got.
    """
    path = tmp_path / train_model.VAL_METRICS_NAME
    train_model.write_val_metrics(path, _val_block(weights_hash="old-weights"))

    assert train_model.load_val_metrics(path, "old-weights")[0]["split"] == "test"
    assert train_model.load_val_metrics(path, "new-weights") == []
    # A missing or unreadable file is the same answer, for the same reason.
    assert train_model.load_val_metrics(tmp_path / "nope.json", "old-weights") == []
    (tmp_path / "corrupt.json").write_text("not json", encoding="utf-8")
    assert train_model.load_val_metrics(tmp_path / "corrupt.json", "old-weights") == []


def test_measuring_the_other_split_keeps_the_one_already_there(tmp_path):
    """`--val --split valid` after `--val` would otherwise delete the acceptance number in
    favour of the selection one - and the selection number is the flattering one, so losing
    the other would be the quietest possible way to overstate a model. `test` stays first.
    """
    path = tmp_path / train_model.VAL_METRICS_NAME
    train_model.write_val_metrics(path, _val_block(split="valid", recall=0.99))
    train_model.write_val_metrics(path, _val_block(split="test", recall=0.62))

    kept = train_model.load_val_metrics(path, "hash-a")
    assert [b["split"] for b in kept] == ["test", "valid"]
    assert kept[0]["per_class"][0]["recall"] == 0.62
    # Re-measuring a split replaces it rather than stacking a second opinion.
    train_model.write_val_metrics(path, _val_block(weights_hash="hash-b"))
    assert [b["split"] for b in train_model.load_val_metrics(path, "hash-b")] == ["test"]


def test_the_recorded_measurement_has_no_hash_and_no_missing_field(tmp_path):
    """Two deliberate omissions. The hash is stripped: the record sits beside the weight it
    describes, so a copy inside it is implied by its location and a `--force` replacement of
    the `.pt` alone would leave it wrong. And `validation` is absent - not null - when there
    are no numbers, so a re-install writes a byte-identical file.
    """
    with_numbers = train_model.weight_record(V2, 2, "snc-grocery", validation=[_val_block()])
    (block,) = with_numbers["validation"]
    assert "weights_sha256" not in block
    assert block["floor"] == 0.85 and block["split"] == "test"

    without = train_model.weight_record(V2, 2, "snc-grocery")
    assert "validation" not in without
    assert train_model.weight_record(V2, 2, "snc-grocery", validation=[]) == without


def test_the_record_carries_the_class_list_the_export_declared():
    """The second fact only this run knows. A `.pt` stores the training run, not its label set,
    so a model trained from a distance-split project predicts three labels per product and
    nothing about the file says so - writing the list down is what lets the app's listing call
    that out before the weights are ever run.
    """
    names = [f"Palmolive Naturals Bar Soap 85g {d}" for d in ("close", "mid", "far")]
    record = train_model.weight_record(V2, 2, "snc-grocery", class_names=names)
    assert record["class_names"] == names

    # Absent, not empty, when the export declared none: an empty list would read as "this model
    # predicts nothing" instead of "not recorded", and those are opposite instructions.
    assert "class_names" not in train_model.weight_record(V2, 2, "snc-grocery")
    assert "class_names" not in train_model.weight_record(V2, 2, "snc-grocery", class_names=[])


def test_the_runbook_and_the_checklist_quote_the_val_step():
    """The floor and the command that measures it have to appear together. A per-class
    number that lives only in the tool is a number nobody is asked for."""
    for doc in (DOC, CHECKLIST):
        text = doc.read_text(encoding="utf-8")
        assert "train_model.py --val" in text, f"{doc.name} does not quote the --val step"


def test_the_default_split_is_the_acceptance_one():
    """`test`, not `val`: the validation split is what training selected on, so quoting it
    back would present the selection number as an acceptance one."""
    assert train_model.DEFAULT_SPLIT == "test"
    assert train_model.DEFAULT_SPLIT in train_model.VALIDATION_SPLITS
    # The yaml key is `val` (write_data_yaml maps valid -> val), so `valid` would be rejected
    # by ultralytics with a FileNotFoundError rather than silently measured.
    assert "valid" not in train_model.VALIDATION_SPLITS
    kw = train_model.validation_kwargs(Path("d.yaml"), "test", Path("runs"), VAL_NAME)
    assert kw["split"] == "test" and kw["imgsz"] == train_model.IMGSZ
    assert kw["name"] == VAL_NAME and kw["exist_ok"] is True


def test_aggregate_metrics_reports_only_what_the_pass_produced():
    box = _FakeBox(index=[0], recall=[0.9], map50=0.91, map_=0.62, mp=0.88, mr=0.86)
    assert train_model.aggregate_metrics(_FakeMetrics(_NAMES, box)) == {
        "precision": 0.88,
        "recall": 0.86,
        "mAP50": 0.91,
        "mAP50-95": 0.62,
    }
    # A pass that answered nothing is not a pass with zeros in it.
    assert train_model.aggregate_metrics(object()) == {}
    assert train_model.per_class_recall(object()) == []


def test_validate_passes_the_split_and_data_yaml_to_the_injected_loader(tmp_path):
    """`yolo` is injectable for the same reason `download_export`'s `get`/`sleep` are: the
    pass needs a GPU, and what is worth pinning is the arguments and the answer."""
    calls: list = []
    metrics = _FakeMetrics(_NAMES, _FakeBox(index=[0, 1], recall=[0.9, 0.88]))

    class _Model:
        def __init__(self, weights):
            calls.append(("load", weights))

        def val(self, **kwargs):
            calls.append(("val", kwargs))
            return metrics

    data_yaml = tmp_path / "data.scanncart.yaml"
    returned = train_model.validate(
        tmp_path / "weights" / "best.pt",
        data_yaml,
        "test",
        tmp_path / "runs",
        VAL_NAME,
        yolo=_Model,
    )

    assert returned is metrics
    assert calls[0] == ("load", str(tmp_path / "weights" / "best.pt"))
    kind, kwargs = calls[1]
    assert kind == "val"
    assert kwargs["split"] == "test"
    assert kwargs["data"] == str(data_yaml)
    assert kwargs["imgsz"] == train_model.IMGSZ


def test_latest_run_finds_a_finished_run_where_run_dir_would_name_the_next_one(tmp_path):
    """`run_dir` answers "where would the *next* run go" - the opposite of what a bare
    `--val`/`--install` needs, which used to look for `<name>-2` and fail."""
    runs = tmp_path / "runs"
    runs.mkdir()
    assert train_model.latest_run(runs, RUN_NAME) is None
    assert train_model.resolve_run(runs, V2) == runs / RUN_NAME

    first = _finished_run(runs, RUN_NAME, mtime=1_000)
    second = _finished_run(runs, f"{RUN_NAME}-2", mtime=2_000)
    # The val output is not a run: it has no weights, so it cannot be installed or measured.
    (runs / VAL_NAME / "weights").mkdir(parents=True)

    assert train_model.latest_run(runs, RUN_NAME, VAL_NAME) == second
    # Newest, not highest-numbered: lexically `-9` sorts above `-2` and `-10` below it, so the
    # suffix cannot be ranked as a number without parsing it back out of the name.
    _finished_run(runs, f"{RUN_NAME}-9", mtime=1_500)
    assert train_model.latest_run(runs, RUN_NAME, VAL_NAME) == second
    assert train_model.resolve_run(runs, V2, str(first)) == first
    # Training still asks run_dir for the next free name, gap-filling included - that contract
    # is unchanged.
    assert train_model.resolve_run(runs, V2, training=True) == runs / f"{RUN_NAME}-3"


def test_cli_val_reports_recall_by_distance_and_writes_it_into_the_record(tmp_path, capsys):
    """The whole step, end to end: the split's number, the same recall split by distance, and
    the breakdown landing in the file `--install` carries into the weights' record.

    The fake answers differently for the `far` pass, which is what a real model does and what
    the feature exists to expose - a class that passes the split while missing most items at
    distance. Written before, this would be a stored overall number and a `far` row describing
    two different runs.
    """
    root = _export_with_names(
        tmp_path / "export",
        {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg", "middle.jpg", "edge.jpg"]},
    )
    manifest = _manifest_with_distances(
        tmp_path, [("near.jpg", "close"), ("middle.jpg", "mid"), ("edge.jpg", "far")]
    )
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)
    calls: list[str] = []

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            if kwargs["name"].endswith("-far"):
                return _FakeMetrics(
                    _NAMES, _FakeBox(index=[0], recall=[0.62], map50=0.4), counts=[10, 0, 0, 0]
                )
            return _FakeMetrics(
                _NAMES, _FakeBox(index=[0], recall=[0.95], map50=0.9), counts=[10, 0, 0, 0]
            )

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--manifest",
            str(manifest),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert calls == [
        VAL_NAME,
        f"{VAL_NAME}-close",
        f"{VAL_NAME}-mid",
        f"{VAL_NAME}-far",
    ]
    assert "the test split by distance (floor 0.85):" in out
    assert "! = below the floor" in out
    assert "below the floor at a distance: far bear-brand 0.620 (n=10)" in out

    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert [entry["distance"] for entry in block["per_distance"]] == ["close", "mid", "far"]
    assert [entry["images"] for entry in block["per_distance"]] == [1, 1, 1]
    far = block["per_distance"][2]
    assert (far["per_class"][0]["name"], far["per_class"][0]["recall"]) == ("bear-brand", 0.62)
    # The floor is on the block, not repeated per distance: the breakdown slices *this*
    # measurement, so a copy per distance could only disagree with the one they were judged on.
    assert "floor" not in far and block["floor"] == train_model.RECALL_FLOOR

    # And the same file read by the sidecar's own reader, which is what the panel uses - the
    # writer and the reader asserted against each other rather than against a fixture.
    from app.models import read_record

    record = train_model.weight_record(V2, 2, "snc-grocery", validation=[block])
    (models_dir := tmp_path / "models-with-record").mkdir()
    weights = models_dir / WEIGHT_NAME
    weights.write_bytes(b"weights")
    weights.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")

    (parsed,) = read_record(weights)["validation"]
    assert [d.distance for d in parsed.per_distance] == ["close", "mid", "far"]
    assert parsed.per_distance[2].per_class[0].recall == 0.62
    assert parsed.per_distance[2].images == 1


def test_cli_no_per_distance_skips_the_extra_passes(tmp_path, capsys):
    """Three extra passes are cheap on a test split and not free on a whole one - and never
    worth running when the answer is not wanted. The record then simply has no breakdown, which
    is the same shape an older record has."""
    root = _export_with_names(
        tmp_path / "export", {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg"]}
    )
    manifest = _manifest_with_distances(tmp_path, [("near.jpg", "close")])
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)
    calls: list[str] = []

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            return _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.95]), counts=[10, 0, 0, 0])

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(tmp_path / "models"), "--manifest", str(manifest),
            "--no-per-distance", "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0 and calls == [VAL_NAME]
    assert "by distance" not in out
    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert block["per_distance"] == []


def test_cli_val_without_a_usable_manifest_says_the_breakdown_was_skipped(tmp_path, capsys):
    """A missing or unmatched manifest is one of the two ways `--val` on this machine cannot
    split by distance (an export whose images no distance accounts for is the other). Either
    way the run must say it skipped, because a section that simply does not appear reads as
    "every distance passed"."""
    root = _export_with_names(
        tmp_path / "export", {"train": ["t.jpg"], "valid": ["v.jpg"], "test": ["near.jpg"]}
    )
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **_kwargs):
            return _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.95]), counts=[10, 0, 0, 0])

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(tmp_path / "models"),
            "--manifest", str(tmp_path / "nowhere" / "manifest.json"), "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "note: no distances for the test split" in out
    assert "skipped, not reported as clean" in out
    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert block["per_distance"] == []


def _grid_cells(out: str, name: str) -> list[str]:
    """One class's four grid cells in column order, read off the printed block.

    Positional on purpose: the whole point of the grid is that the `mid` cell is the `mid`
    pass's number, and a substring check passes happily with two columns swapped.
    """
    line = next(line for line in out.splitlines() if line.strip().startswith(f"{name} "))
    return re.findall(r"-|\d+\.\d{3} \(\d+\)!?", line.split(name, 1)[1])


def test_cli_val_grid_runs_all_three_rows_from_the_sets_own_report(tmp_path, capsys):
    """The rows this dataset exists for, driven end to end through the real CLI. On this machine
    only `close` frames were ever decided, so the `mid` and `far` columns and the pass/fail they
    carry were code paths nothing had observed - this fixture's own report spans the plan.

    The counts are **per split**: `train` holds frames at all three distances too, and each pass
    measures the split being validated only. Pinning `images` to the test split's counts is what
    proves the join - report by filename, split by directory - instead of a sum over the set.

    The cells are read positionally, and the notes are pinned empty: with a frame at every
    distance and every frame placed, there is no caveat to print - neither `mid`/`far`
    unmeasured nor a count of frames carrying no distance - and a line that only ever appears
    when something is missing is exactly the one nobody notices going wrong.
    """
    root = _built_set(
        tmp_path / "merged-v2",
        {
            "train": [("t1.jpg", "close"), ("t2.jpg", "mid"), ("t3.jpg", "far"), ("t4.jpg", "far")],
            "valid": [("v1.jpg", "close")],
            "test": [
                ("near1.jpg", "close"),
                ("near2.jpg", "close"),
                ("middle1.jpg", "mid"),
                ("middle2.jpg", "mid"),
                ("edge1.jpg", "far"),
                ("edge2.jpg", "far"),
            ],
        },
    )
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)
    calls: list[str] = []
    # A number per pass, so a row cannot be right by being the overall measurement again: the
    # two real bugs this catches are a pass that is never run and a row fed by the wrong pass.
    recalls = {f"{VAL_NAME}-close": 0.95, f"{VAL_NAME}-mid": 0.80, f"{VAL_NAME}-far": 0.62}

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            return _FakeMetrics(
                _NAMES,
                _FakeBox(index=[0], recall=[recalls.get(kwargs["name"], 0.90)], map50=0.9),
                counts=[2, 0, 0, 0],
            )

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--manifest",
            str(tmp_path / "elsewhere" / "manifest.json"),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert calls == [VAL_NAME, f"{VAL_NAME}-close", f"{VAL_NAME}-mid", f"{VAL_NAME}-far"]
    assert "11 frame(s) from the set's own merge_report.json" in out
    # The three rows, positionally: `close`/`mid`/`far` each carry their own pass's number and
    # its own `!`, and `all` is the overall pass - which a substring check cannot tell from
    # three columns showing one pass's result under three headings.
    assert _grid_cells(out, "bear-brand") == [
        "0.950 (2)",
        "0.800 (2)!",
        "0.620 (2)!",
        "0.900 (2)",
    ]
    # No notes at all: every planned distance had a pass and every test frame had a distance.
    assert "  note:" not in out
    # The miss list, which is what a reader acts on - and not the all-clear sentence, because
    # two cells here are below the floor.
    assert "below the floor at a distance:" in out
    assert "no class is below the floor at any distance" not in out
    # Worst first, both distances named - order asserted loosely because it is sorted by the
    # number (far 0.620 above mid 0.800), and a test that pins the order would fail on a tie
    # without catching anything.
    assert "mid bear-brand 0.800 (n=2)" in out
    assert "far bear-brand 0.620 (n=2)" in out

    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert [entry["distance"] for entry in block["per_distance"]] == ["close", "mid", "far"]
    assert [entry["images"] for entry in block["per_distance"]] == [2, 2, 2]


def test_cli_val_grid_says_no_distance_hides_a_miss_when_none_does(tmp_path, capsys):
    """The grid's other verdict, and the shape the real coverage run took: a test split holding
    frames at all three distances, every one of them above the floor. The block has to *say*
    that - a footer that fell silent would read as "nothing below the floor" whether or not
    anything had been measured - and the notes stay empty, because there is no caveat left once
    every distance was measured and every frame was placed.

    Read positionally for the same reason as its sibling: the three numbers here are close
    together (0.950/0.900/0.880), so a swapped pair would still satisfy a substring check.
    """
    root = _built_set(
        tmp_path / "merged-v2",
        {
            "train": [("t1.jpg", "close"), ("t2.jpg", "mid")],
            "valid": [("v1.jpg", "close")],
            "test": [
                ("near1.jpg", "close"),
                ("near2.jpg", "close"),
                ("near3.jpg", "close"),
                ("middle1.jpg", "mid"),
                ("middle2.jpg", "mid"),
                ("middle3.jpg", "mid"),
                ("edge1.jpg", "far"),
                ("edge2.jpg", "far"),
                ("edge3.jpg", "far"),
            ],
        },
    )
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)
    calls: list[str] = []
    recalls = {f"{VAL_NAME}-close": 0.95, f"{VAL_NAME}-mid": 0.90, f"{VAL_NAME}-far": 0.88}

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            name = kwargs["name"]
            # The split pass counts all nine instances; each distance pass counts its three.
            one_distance = name.rsplit("-", 1)[-1] in train_model.DISTANCE_ORDER
            return _FakeMetrics(
                _NAMES,
                _FakeBox(index=[0], recall=[recalls.get(name, 0.91)], map50=0.9),
                counts=[3, 0, 0, 0] if one_distance else [9, 0, 0, 0],
            )

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--manifest",
            str(tmp_path / "elsewhere" / "manifest.json"),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert calls == [VAL_NAME, f"{VAL_NAME}-close", f"{VAL_NAME}-mid", f"{VAL_NAME}-far"]
    assert _grid_cells(out, "bear-brand") == [
        "0.950 (3)",
        "0.900 (3)",
        "0.880 (3)",
        "0.910 (9)",
    ]
    # The all-clear sentence instead of the miss list - and the miss heading absent, so the two
    # verdicts cannot be printed at once.
    assert "no class is below the floor at any distance" in out
    assert "below the floor at a distance:" not in out
    # Full coverage, nothing to caveat: no note line at all.
    assert "  note:" not in out

    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert [entry["distance"] for entry in block["per_distance"]] == ["close", "mid", "far"]
    assert [entry["images"] for entry in block["per_distance"]] == [3, 3, 3]


def test_cli_val_says_a_distance_with_no_test_frame_rather_than_passing_over_it(tmp_path, capsys):
    """The same set with `far` gone from the report - the shape a build that never recorded the
    axis has for *every* distance. The frame is still in the set (and still in `train`), so the
    pass it would have been measured on disappears: one fewer validation, the distance named as
    unmeasured, and the frame counted as carrying no distance rather than folded into a row.
    """
    root = _built_set(
        tmp_path / "merged-v2",
        {
            "train": [("t1.jpg", "close"), ("t2.jpg", "mid"), ("t3.jpg", "far"), ("t4.jpg", "far")],
            "valid": [("v1.jpg", "close")],
            "test": [("near1.jpg", "close"), ("middle1.jpg", "mid"), ("edge1.jpg", "far")],
        },
    )
    report = json.loads((root / "merge_report.json").read_text(encoding="utf-8"))
    report["distances"] = {
        name: distance for name, distance in report["distances"].items() if distance != "far"
    }
    (root / "merge_report.json").write_text(json.dumps(report), encoding="utf-8")
    runs = tmp_path / "runs"
    run = _finished_run(runs, RUN_NAME)
    calls: list[str] = []

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **kwargs):
            calls.append(kwargs["name"])
            return _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.95]), counts=[1, 0, 0, 0])

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--manifest",
            str(tmp_path / "elsewhere" / "manifest.json"),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert calls == [VAL_NAME, f"{VAL_NAME}-close", f"{VAL_NAME}-mid"]  # no far pass
    assert "note: far: no images in the test split, so it is unmeasured" in out
    assert "1 image(s) in the test split carry no distance and are counted in the overall number only" in out
    (block,) = json.loads((run / train_model.VAL_METRICS_NAME).read_text(encoding="utf-8"))
    assert [entry["distance"] for entry in block["per_distance"]] == ["close", "mid"]


def test_cli_val_reports_per_class_recall_without_being_told_the_run(tmp_path, capsys):
    """The whole step at the CLI level: no `--run-dir`, because that is the form the docs
    quote - the run it just trained, found by being the newest one with weights."""
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    _finished_run(runs, f"{RUN_NAME}-2")

    metrics = _FakeMetrics(
        _NAMES,
        _FakeBox(index=[0, 1, 3], recall=[0.90, 0.62, 0.95], map50=0.88, map_=0.61, mp=0.9, mr=0.82),
        counts=[10, 30, 0, 12],
    )
    loaded: list[str] = []

    class _Model:
        def __init__(self, weights):
            loaded.append(weights)

        def val(self, **kwargs):
            assert kwargs["split"] == "test"
            return metrics

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            # An explicit manifest, so this test's outcome cannot depend on the dataset workspace
            # on the machine running it: no distances means the breakdown is skipped, which is
            # the state this test is about.
            "--manifest",
            str(tmp_path / "no-manifest.json"),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert loaded == [str(runs / f"{RUN_NAME}-2" / "weights" / "best.pt")]
    assert "[.ok.] bear-brand 0.900 >= 0.85 (n=10)" in out
    assert "[.ok.] milo 0.950 >= 0.85 (n=12)" in out
    assert "[WARN] century-tuna 0.620 < 0.85 (n=30)" in out
    assert "[SKIP] lucky-me: no ground-truth instances in the split" in out
    assert "target >= 0.90" in out  # the mAP50 row keeps 6's aggregate target
    assert "below the 0.85 floor" in out
    assert f"--install --generation {V2.name} --run-dir" in out
    # --val alone installs nothing.
    assert not (tmp_path / "models").exists()


def test_cli_the_numbers_from_val_reach_the_record_via_a_separate_install(tmp_path, capsys):
    """The documented sequence, run the way the docs run it: `--val` in one command,
    `--install` in another. The numbers therefore travel through a file, and this is the test
    that would fail if either end of that file stopped agreeing with the other - which would
    leave the panel showing nothing while the terminal had printed a full table.
    """
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    run = _finished_run(runs, RUN_NAME)

    metrics = _FakeMetrics(
        _NAMES,
        _FakeBox(index=[0, 1], recall=[0.62, 0.99], map50=0.88, map_=0.61, mp=0.9, mr=0.82),
        counts=[30, 12, 0, 0],
    )

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **_kwargs):
            return metrics

    measured = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models),
            "--manifest", str(tmp_path / "no-manifest.json"), "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out
    assert measured == 0
    assert str(run / train_model.VAL_METRICS_NAME) in out
    # --val alone still installs nothing: the numbers are written, not the model.
    assert not models.exists()

    installed = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models), "--install", "--version", "2",
        ],
        yolo=lambda *_a, **_k: pytest.fail("an install must not need the model runtime"),
    )
    out = capsys.readouterr().out
    assert installed == 0

    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    # The export's own class list, read off its data.yaml at this boundary rather than passed
    # in by the test: the wiring is the half that could go missing while every unit below it
    # still passes, and its absence is invisible until a bad model runs unnoticed.
    assert record["class_names"] == list(V2.classes)
    (block,) = record["validation"]
    assert block["split"] == "test" and block["floor"] == train_model.RECALL_FLOOR
    assert [(c["name"], c["recall"]) for c in block["per_class"]] == [
        ("bear-brand", 0.62),
        ("century-tuna", 0.99),
        ("lucky-me", None),
        ("milo", None),
    ]
    # `--install` reports what it picked up, so an install without numbers is visible at the
    # moment it happens rather than only when someone opens the panel.
    assert "test: 2/4 classes measured, 1 below the 0.85 floor (bear-brand)" in out
    # And the same record read by the sidecar's own reader, which is what the panel uses -
    # the writer and the reader asserted against each other, not against a fixture.
    from app.models import read_record

    (parsed,) = read_record(models / WEIGHT_NAME)["validation"]
    assert parsed.split == "test"
    assert [c.name for c in parsed.per_class if c.recall is None] == ["lucky-me", "milo"]


def test_cli_install_says_so_when_nothing_was_measured(tmp_path, capsys):
    """Silence would read as "the numbers are there and the panel will show them". The one
    thing this step must not do is leave the operator believing a score was recorded.
    """
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    _finished_run(runs, RUN_NAME)

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models), "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No measurement of *these* weights was found" in out
    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    assert "validation" not in record


def test_cli_a_measurement_of_a_replaced_checkpoint_is_not_installed(tmp_path, capsys):
    """The hash, at the CLI level. A `--val` of the previous checkpoint sits in the same run
    directory; re-training replaces `best.pt` in place. If the numbers were attached anyway,
    the panel would show a score these weights never got - and nothing about the output would
    look wrong.
    """
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    run = _finished_run(runs, RUN_NAME)

    metrics = _FakeMetrics(_NAMES, _FakeBox(index=[0], recall=[0.99]), counts=[10, 0, 0, 0])

    class _Model:
        def __init__(self, _weights):
            pass

        def val(self, **_kwargs):
            return metrics

    train_model.main(
        ["--export-dir", str(root), "--run-project", str(runs), "--val"], yolo=_Model
    )
    capsys.readouterr()
    assert train_model.load_val_metrics(
        run / train_model.VAL_METRICS_NAME, train_model.weights_sha256(run / "weights" / "best.pt")
    )

    # The checkpoint is replaced - same path, different bytes - as a re-train into the same
    # --run-dir does.
    (run / "weights" / "best.pt").write_bytes(b"a different checkpoint")

    code = train_model.main(
        [
            "--export-dir", str(root), "--run-project", str(runs),
            "--models-dir", str(models), "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No measurement of *these* weights was found" in out
    record = json.loads((models / f"{WEIGHT_NAME[:-3]}.json").read_text("utf-8"))
    assert "validation" not in record


def test_cli_val_says_nothing_ran_when_asked_for_neither_action(tmp_path, capsys):
    root = _fake_export(tmp_path / "export-v2")
    code = train_model.main(
        ["--export-dir", str(root), "--run-project", str(tmp_path / "runs")],
        yolo=lambda *_a, **_k: pytest.fail("--val was not asked for; nothing should load"),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--val to validate" in out


def test_cli_val_on_a_split_with_no_run_is_an_error_not_a_silent_pass(tmp_path):
    """No weights means no number. Reporting an empty table as success is the one outcome
    that would make `--val` worse than not having it."""
    root = _fake_export(tmp_path / "export-v2")
    with pytest.raises(SystemExit) as excinfo:
        train_model.main(
            ["--export-dir", str(root), "--run-project", str(tmp_path / "runs"), "--val"],
            yolo=lambda *_a, **_k: pytest.fail("nothing to load"),
        )
    assert "the run did not complete" in str(excinfo.value)


def test_cli_install_without_a_run_dir_installs_the_finished_run(tmp_path, capsys):
    """The docs' own `train_model.py --install` line, which used to need a `--run-dir` the
    examples did not mention."""
    root = _fake_export(tmp_path / "export-v2")
    runs = tmp_path / "runs"
    _finished_run(runs, RUN_NAME)

    code = train_model.main(
        [
            "--export-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--install",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "models" / WEIGHT_NAME).read_bytes() == b"weights"
    assert "resize_mode: stretch" in out


def test_no_weight_files_are_tracked():
    """63 MB of COCO weights were committed before the `*.pt` rule existed and have been taken
    out of the index with `git rm --cached` (the files stay on disk - ultralytics needs them
    there). Checked through git rather than by reading `.gitignore`, because that is the only
    thing that knows: an ignore rule does **not** untrack something already in the index, which
    is how three weight files stayed committed while the file that bans them sat right beside
    them. Nothing in the working tree changes if this regresses; that is the whole problem.
    """
    ls = subprocess.run(
        ["git", "ls-files", "sidecar"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    if ls.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = [p for p in ls.stdout.splitlines() if p.endswith((".pt", ".onnx"))]
    assert tracked == [], f"weights are tracked again: {tracked}"


def test_models_readme_is_tracked_and_names_the_weights_the_tool_installs(tmp_path):
    """Two files have to agree about the filename the picker will show: the tool that
    installs the weights and the README in the directory they land in. And the weights
    themselves must stay out of git while the README stays in.
    """
    models = REPO_ROOT / "sidecar" / "models"
    readme = models / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert WEIGHT_NAME in text
    assert "resize_mode: stretch" in text

    ignore = (REPO_ROOT / "sidecar" / ".gitignore").read_text(encoding="utf-8")
    # `models/*`, not `models/`: excluding the directory excludes its contents too, and
    # git does not re-include a path whose parent directory is excluded - so the negation
    # would be dead and this README silently untracked. (`git check-ignore -v` on it used
    # to report the *directory* pattern, i.e. ignored.)
    assert "models/*" in ignore
    assert "!models/README.md" in ignore


def _built_set(root: Path, per_split: dict[str, list[tuple[str, str]]]) -> Path:
    """A merged set in the shape `build_dataset.py` writes it: frames filed per split, a label
    beside each, and the per-frame `distances` map in its own `merge_report.json`.

    `per_split` is `split -> [(filename, distance)]`, so a test states the counts the reader has to
    join - the report is keyed by name and the split is a directory, and the grid's passes have to
    marry the two. `_export_with_names` supplies the frames, labels and class list; the report is
    what makes it a *built set* rather than a download.
    """
    dataset = _export_with_names(
        root, {split: [name for name, _ in frames] for split, frames in per_split.items()}
    )
    _report_with_distances(
        dataset, [(name, distance) for frames in per_split.values() for name, distance in frames]
    )
    return dataset
