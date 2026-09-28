"""Tests for `generations.py`: one trainer, two datasets, and what has to differ.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from pathlib import Path
import pytest
import annotate.store
import build_dataset
import dataset_doctor
import generations
import label_classes
import resources
import train_model

# Rebuilt from the modules that own each fact, not copied by value.
V2 = generations.V2

from tests.dataset_tool_helpers import (
    _FakeBox,
    _FakeMetrics,
    _NAMES,
    _fake_export,
    _finished_run,
)


# --------------------------------------------------------------------------
# 10. generations - one trainer, two datasets, and what has to differ
# --------------------------------------------------------------------------


def test_both_generations_declare_the_same_seven_names():
    """v1 is not a different product list: it is the same seven names v1's project declares, and
    v2 declares them too now that Palmolive is gone - which is what keeps a v1 weight's labels
    continuous with v2's and lets one merged dataset train either generation's head.

    `added_over` is asserted empty in **both** directions rather than deleted: it is the sentence
    the export check prints above the run ("which classes this generation can never predict"), and
    an empty answer is the honest one today. The mechanism is exercised by the export check's own
    tests, where a synthetic generation declares an extra class.
    """
    assert set(generations.V1.classes) == set(generations.V2.classes)
    assert len(generations.V1.classes) == 7 and len(generations.V2.classes) == 7
    assert generations.added_over(generations.V1, generations.V2) == ()
    assert generations.added_over(generations.V2, generations.V1) == ()
    # The v2 side is the tools' own copy of the roster, not a retyped list.
    assert generations.V2.classes == tuple(label_classes.SLUG_TO_CLASS.values())
    # The same seven names in a different order, which is what makes the order itself a fact worth
    # owning: every label row indexes its list by position, so a set in one order read as the other
    # is a set of boxes under neighbouring products.
    assert generations.V1.classes != generations.V2.classes
    assert generations.order_of(generations.V1.classes).name == "v1"
    assert generations.order_of(V2.classes).name == "v2"
    # Only an exact match counts: a near miss is a membership question, and answering it with the
    # closest roster would read a mislabelled set as a healthy one from the other generation.
    assert generations.order_of([]) is None
    assert generations.order_of(V2.classes[:-1]) is None
    assert generations.order_of([*V2.classes[:-1], "Palmolive Naturals Bar Soap 85g"]) is None
    # A sorted list is *v1's* order, not a default and not a mistake: v1's project declares its
    # seven alphabetically, which is why sorting a v2 roster answers v1 rather than nothing.
    assert generations.order_of(sorted(V2.classes)).name == "v1"


def test_every_dataset_writer_reads_the_order_from_the_spec(tmp_path):
    """One order per generation, and no second derivation of it. `build_dataset` writes the merged
    set's `names` and remaps v1's indices into it, and the annotator writes the `cls` column that
    the merged set copies verbatim - so both have to be *this* list, in this order. The merged order
    used to be a second `tuple(SLUG_TO_CLASS.values())` here, which agreed with `generations.V2`
    only while the two expressions did; a reorder in one of them would have made the doctor's order
    finding fire on a set whose own builder wrote it.
    """
    assert build_dataset.CANONICAL_NAMES == V2.classes
    assert build_dataset.INDEX_BY_NAME == {name: i for i, name in enumerate(V2.classes)}
    assert build_dataset.DECLARED_GENERATION is V2
    # The authoring end: rows are `cls` positions in this list, and `build_v2` copies them without
    # translating, so the annotator's order is the merged set's order by construction or not at all.
    assert annotate.store.CLASS_NAMES == V2.classes


def test_the_merged_set_records_the_generation_whose_order_it_writes(tmp_path):
    """A dataset that says which generation it is: `data.yaml` (names + the generation they are) and
    `merge_report.json`. `dataset_doctor --generation auto` reads both instead of being told, which
    is what makes judging a v1 set against v2's expectation impossible by omission."""
    from collections import Counter

    import yaml

    out = tmp_path / "merged-v2"
    for split in build_dataset.SPLIT_NAMES:
        images, labels = build_dataset.split_dirs(out, split)
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / f"{split}_0001.jpg").write_bytes(b"")
        (labels / f"{split}_0001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    splits = {split: build_dataset.split_dirs(out, split)[0] for split in build_dataset.SPLIT_NAMES}

    body = yaml.safe_load(build_dataset.write_names_yaml(out, splits).read_text(encoding="utf-8"))
    assert body["names"] == list(V2.classes)
    assert body["generation"] == "v2"
    # The reader the doctor uses, against the writer above - and a name this app has no roster for
    # is *not recorded* rather than quoted, so a record written by a later tool cannot make the
    # doctor judge the labels against an order it cannot name.
    assert train_model.read_export_generation(out) == "v2"
    (out / "data.yaml").write_text("names: [a]\ngeneration: v9\n", encoding="utf-8")
    assert train_model.read_export_generation(out) is None

    report = build_dataset.summarise(out, [], [], Counter(), 640, [])
    assert report["generation"] == "v2"
    assert report["classes"] == list(V2.classes)
    assert "in v2's order" in build_dataset.render_report(report, [], {"path": "sheet.jpg", "frames": 0})


def test_the_generation_names_are_the_ones_the_picker_looks_for():
    """A generation's name is not a label: it is the filename the Admin Panel lists and
    `settings_store.is_custom_model` validates (MODEL_TRAINING.md 8.2)."""
    assert generations.V1.weight_name == "scanncart-grocery-v1.pt"
    assert generations.V2.weight_name == "scanncart-grocery-v2.pt"
    assert generations.V1.run_name == "scanncart-grocery-v1"
    assert train_model.val_name(generations.V2) == "scanncart-grocery-v2-val"
    assert generations.DEFAULT.name == "v2"
    with pytest.raises(SystemExit, match="unknown generation"):
        generations.get("v3")


def test_a_v1_export_is_judged_against_v1s_class_list(tmp_path):
    """Why the check is per generation rather than one shared roster: an export is judged against
    the list *its own dataset declared*, so judging v1's seven against a list that expects a class
    v1 never had would refuse a set that is correct - and a check that cries wolf is how the real
    mismatch gets waved through.

    Both rosters declare the same seven names today, so the two readings agree; the mechanism is
    proven with a generation that declares an eighth, which is the state the project is in the
    moment a product is added.
    """
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    _splits, problems = train_model.check_export(root, generations.V1)
    assert problems == []

    # Read as a v2 export it is still clean: same seven names, so nothing is missing and nothing
    # is unexpected. Only a generation that declares a *different* list can disagree with it.
    _splits, v2_problems = train_model.check_export(root, generations.V2)
    assert v2_problems == []

    widened = dataclasses.replace(
        generations.V2, classes=generations.V2.classes + ("Farmer's Choice Fresh Milk",)
    )
    _splits, widened_problems = train_model.check_export(root, widened)
    assert widened_problems == ["the export has no class for: Farmer's Choice Fresh Milk"]


def test_the_export_check_says_which_classes_this_generation_can_never_predict(tmp_path, capsys, monkeypatch):
    """Not a problem - it is a property of the dataset - but it is the sentence the app will show
    about these weights, so it is better known before an hour of GPU time than after it.

    Empty today in both directions (the two generations declare the same seven), so the silence is
    asserted for both real generations and the line is exercised with the spec table widened -
    otherwise the note would be untested exactly while it is unreachable.

    The note is read off `generations.GENERATIONS` rather than off the spec it is handed ("another
    generation declares"), which is why the widened entry is installed in the table rather than
    passed in: that is the lookup the print actually performs.
    """
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    train_model.check_export(root, generations.V1)
    train_model.check_export(root, generations.V2)
    assert "can never predict" not in capsys.readouterr().out

    widened = dataclasses.replace(
        generations.V2, classes=generations.V2.classes + ("Farmer's Choice Fresh Milk",)
    )
    monkeypatch.setitem(generations.GENERATIONS, "v2", widened)
    train_model.check_export(root, generations.V1)
    assert "can never predict them: Farmer's Choice Fresh Milk" in capsys.readouterr().out


def test_v1_declares_no_distance_axis_where_v2_names_a_manifest():
    """`None` is the generation saying it has no such axis - v1's export carries no distance tags
    - which is why `--val` gives it its own sentence rather than reporting a manifest that went
    missing, and why an absent breakdown there cannot be read as "every distance passed"."""
    assert generations.V1.manifest is None
    assert generations.V2.manifest is not None
    assert generations.V2.manifest.name == "manifest.json"
    # A generation with no axis has no distances to look up, whatever it is asked for.
    assert train_model.distance_map(generations.V1.manifest) == {}


def test_the_geometry_reading_tells_stretched_frames_from_padded_ones(tmp_path):
    """Why this reading exists: `resize_mode: auto` resolving to letterbox for a stretch-trained
    `.pt` is a failure this project has already paid for, so the frames are measured rather than
    trusted from a constant. A stretched frame fills its border with content; a fitted one pads
    with a constant colour, and that is the whole signature."""
    numpy = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")

    content = numpy.random.default_rng(0).integers(40, 210, (64, 64, 3), dtype="uint8")

    def _images(name: str, pixels) -> Path:
        directory = tmp_path / name
        directory.mkdir(parents=True)
        image_module.fromarray(pixels).save(directory / "frame.jpg")
        return directory

    stretched = _images("stretched", content)
    letterboxed_pixels = numpy.zeros((64, 64, 3), dtype="uint8")
    letterboxed_pixels[16:-16] = content[16:-16]
    letterboxed = _images("letterboxed", letterboxed_pixels)

    (stretched_note,) = train_model.frame_geometry({"train": stretched}, sample=1)
    assert "no constant border, i.e. stretched" in stretched_note

    (padded_note,) = train_model.frame_geometry({"train": letterboxed}, sample=1)
    assert "constant border" in padded_note
    assert "fitting" in padded_note
    # A reading that cannot be taken is a note, never a failure: the frames are a claim about the
    # geometry, and a training run should not die because one file would not open.
    assert train_model.frame_geometry({}) == [
        "frames  no images to sample, so the geometry was not measured"
    ]


def test_the_run_is_bounded_by_the_machine_budget():
    """The trainer shares this box with the app, so its dataloader count comes from the CPU budget
    rather than ultralytics' default of eight processes, and its batch from the VRAM share rather
    than from the constant - which is what stops a run taking the machine over."""
    assert train_model.derive_workers(12) == 2
    assert train_model.derive_workers(1) == 1  # never zero workers
    assert resources.CPU_THREADS < (os.cpu_count() or 4)  # something is always left free
    budget = resources.Budget(vram_cap_gb=3.0)
    assert budget.batch_size(640) == 8
    # The same share at four times the pixels fits far less, which is the whole point of deriving
    # it: a batch that OOMs takes every other application on the card down with the run.
    assert budget.batch_size(1280) < budget.batch_size(640)
    # An explicit --batch is a ceiling, never a way past the card's share.
    assert min(16, budget.batch_size(640)) == 8
    assert min(4, budget.batch_size(640)) == 4


def test_cli_val_for_a_generation_with_no_distance_axis_says_there_is_none(tmp_path, capsys):
    """The per-class table is the whole readout for v1: no grid, and no complaint about a file
    that never existed. The distinction is the reason `manifest` is `None` rather than a path."""
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    runs = tmp_path / "runs"
    _finished_run(runs, generations.V1.run_name)

    metrics = _FakeMetrics(
        _NAMES,
        _FakeBox(index=[0], recall=[0.91], map50=0.9, map_=0.63, mp=0.9, mr=0.88),
        counts=[20, 20, 20, 20],
    )

    class _Model:
        def __init__(self, weights):
            self.weights = weights

        def val(self, **kwargs):
            return metrics

    code = train_model.main(
        [
            "--generation",
            "v1",
            "--dataset-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--val",
        ],
        yolo=_Model,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "has no distance axis" in out
    assert "no distances for the" not in out
    assert "split by distance" not in out
    # The per-class readout itself is the same one v2 gets, floor and instance counts included.
    assert "[.ok.] bear-brand 0.910 >= 0.85 (n=20)" in out


def test_a_training_run_refuses_a_set_the_doctor_has_not_passed(tmp_path, capsys):
    """The gate on this side of the seam, asserted as "the model was never even pulled".

    A six-field row is the shape the loader drops *whole frames* for, and nothing else in this tool
    looks at labels at all - the run would train, print numbers, and install a weight whose set was
    not what the command line said. `--val` goes through the same gate (`will_run`).
    """
    root = _fake_export(tmp_path / "export-v2")
    (root / "train" / "labels" / "train_0.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5\n", encoding="utf-8"
    )
    pulled: list[str] = []

    def _yolo(name):
        pulled.append(name)
        raise AssertionError("a refused set must not reach the model runtime")

    code = train_model.main(
        ["--export-dir", str(root), "--run-project", str(tmp_path / "runs"), "--yes"],
        yolo=_yolo,
    )
    out = capsys.readouterr().out

    assert code == 2
    assert pulled == []
    assert "[FAIL]" in out and "6 field(s)" in out
    assert "Nothing was trained" in out


def test_cli_train_creates_the_run_project_and_passes_the_budget_to_the_run(tmp_path, capsys):
    """The `--yes` path, which is the one this project actually runs and the one no other test
    reaches. Two things in it are load-bearing: the run project may not exist yet - the free-space
    check raises FileNotFoundError on a missing directory, which is exactly how the first v1
    training run died a second before it started - and the batch/worker/device that reach
    ultralytics have to be the ones the budget derived, not the constants.
    """
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    runs = tmp_path / "runs"  # deliberately absent
    pulled: list[str] = []
    trained: list[dict] = []

    class _Model:
        def train(self, **kwargs):
            trained.append(kwargs)
            # A checkpoint, because the tool reads one back to report on the run; a fake that
            # left it out would test only the failure path.
            weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "best.pt").write_bytes(b"weights")

    def _yolo(name):
        pulled.append(name)
        return _Model()

    code = train_model.main(
        [
            "--generation",
            "v1",
            "--dataset-dir",
            str(root),
            "--run-project",
            str(runs),
            "--models-dir",
            str(tmp_path / "models"),
            "--yes",
        ],
        yolo=_yolo,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert runs.is_dir()
    # The doctor ran and passed on the way in - the `[ok]` line is the evidence that a run is never
    # measured on a set nobody checked, in the same words `make doctor` prints it in.
    assert "[ok] the set is internally consistent" in out
    assert pulled == [train_model.BASE_MODEL]
    (kwargs,) = trained
    assert kwargs["name"] == generations.V1.run_name
    assert kwargs["epochs"] == train_model.EPOCHS
    assert kwargs["imgsz"] == train_model.IMGSZ
    # The machine's share, not the constants: a batch that OOMs on a shared card takes the other
    # applications on it down with the run.
    assert kwargs["batch"] <= train_model.BATCH
    assert kwargs["workers"] == train_model.derive_workers(resources.CPU_THREADS)
    assert kwargs["device"] in ("0", "cpu")
    assert "resource budget" in out and "GB free at" in out
    # --yes alone installs nothing: the drop-in is its own step.
    assert not (tmp_path / "models").exists()


def test_cli_v1_installs_under_v1s_own_name_with_its_own_class_list(tmp_path, capsys):
    """The drop-in is what makes a trained checkpoint selectable, and for v1 it has to be v1's
    name with v1's seven classes recorded: those weights are correct and still unable to predict
    Palmolive, and the record is where the app can see that before running them.
    """
    root = _fake_export(tmp_path / "export-v1", names=list(generations.V1.classes))
    models = tmp_path / "models"
    _finished_run(tmp_path / "runs", generations.V1.run_name)

    code = train_model.main(
        [
            "--generation",
            "v1",
            "--dataset-dir",
            str(root),
            "--run-project",
            str(tmp_path / "runs"),
            "--models-dir",
            str(models),
            "--version",
            "1",
            "--install",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert (models / "scanncart-grocery-v1.pt").read_bytes() == b"weights"
    # Nothing lands under the other generation's name: the filename is the picker's key, so a v1
    # run installing as v2 would silently replace the model the app is built towards.
    assert not (models / generations.V2.weight_name).exists()

    record = json.loads((models / "scanncart-grocery-v1.json").read_text("utf-8"))
    assert record["generation"] == "v1"
    assert record["resize_mode"] == "stretch"
    assert record["source"] == "scanncart-grocery version 1"
    assert record["class_names"] == list(generations.V1.classes)
    assert "Palmolive Naturals Bar Soap 85g" not in record["class_names"]
    assert "scanncart-grocery-v1" in out
