"""Tests for `readiness.py` - the page that answers "are we ready to train", and its two kinds of no.

The tool's whole value is that it *delegates*: every row is another module's verdict, and the one
thing it adds is the distinction between a gate that refuses a run (`--yes` exits 2, no GPU spent) and
a row that only says the dataset is unfinished. So the tests are about that distinction first, and
about delegation second - a row that re-derived a rule would pass every assertion about its own output
while being free to disagree with the command it claims to summarise, which is the failure worth a
test rather than the formatting of a table.

Real modules where they are cheap and file-sized (`accept_v2`'s stamp gate reads a file,
`train_model.pass_gate` reads a report, `dataset_doctor`'s verdict is a list) and fakes only for the
two things a test cannot have: torch, which the budget row is the sole caller of, and a labeled
annotator store, which `dataset_tool_helpers` and `test_label_progress` already build.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import dataset_doctor
import generations
import label_progress
import readiness
import resources
import train_model
import workspace
from tests.dataset_tool_helpers import _local_args, _staged_local

OK, WARN, FAIL = readiness.OK, readiness.WARN, readiness.FAIL


def _progress(**over) -> label_progress.Progress:
    """A `Progress` with the numbers a row reads, so a row can be tested without a label tree.

    Built as the dataclass rather than as a dict because that is the shape `local_progress` returns;
    a fake that is not the reader's own type would let a rename through.
    """
    summary = {
        "total": 10,
        "decided": 10,
        "tier_a": {"target": 0, "remaining": 0, "cells_under_target": 0, "cells": []},
    }
    summary.update(over.pop("summary", {}))
    cells = over.pop("cells", {})
    split_stat = over.pop("split_stat", {})
    base = dict(
        summary=summary,
        cells=cells,
        per_class={},
        split_stat=split_stat,
        sessions_out=[],
        null_names=[],
        mismatches=[],
        multi_box=[],
    )
    base.update(over)
    return label_progress.Progress(**base)


def _staged(tmp_path: Path, **over) -> Path:
    """A staged set with a manifest beside it, and optional numbers for the label rows."""
    out = tmp_path / "cleaned-v2"
    out.mkdir(exist_ok=True)
    (out / workspace.MANIFEST_NAME).write_text("[]", encoding="utf-8")
    if over:
        (out / "the-numbers.pkl").write_text("placeholder", encoding="utf-8")
    return out


def _rows_of(rows: list[readiness.Row], name: str) -> readiness.Row:
    found = [row for row in rows if row.name == name]
    assert found, f"no {name!r} row in {[row.name for row in rows]}"
    return found[0]


def _annotations(tmp_path: Path) -> Path:
    """The annotator's tree, with one decision in it - enough for a stamp to have something to hash."""
    annotations = tmp_path / "annotations-v2"
    annotations.mkdir(exist_ok=True)
    (annotations / workspace.PROVENANCE_NAME).write_text(
        json.dumps({"milo_0001.jpg": {"state": "labeled"}}), encoding="utf-8"
    )
    return annotations


def _dataset(tmp_path: Path, *, reported: bool = True, machine_only=None, sha: str | None = None) -> Path:
    """A merged set: the directory, and the merge report §6 leaves in it.

    The stamp is over a file this fixture actually writes, because `accept_v2.annotation_state_gate`
    reads that file - a made-up digest would only ever agree with a made-up reader. `sha` overrides it
    to produce the stale case, which is what a labeling session *after* a build looks like.
    """
    dataset = tmp_path / "merged-v2"
    dataset.mkdir(exist_ok=True)
    if not reported:
        return dataset
    annotations = _annotations(tmp_path)
    body = {
        "generation": generations.DEFAULT.name,
        "annotations": {
            "provenance": str(annotations / workspace.PROVENANCE_NAME),
            "sha256": sha
            or hashlib.sha256((annotations / workspace.PROVENANCE_NAME).read_bytes()).hexdigest(),
        },
        "machine_only_by_split": {} if machine_only is None else machine_only,
    }
    (dataset / workspace.MERGE_REPORT_NAME).write_text(json.dumps(body), encoding="utf-8")
    return dataset


class _Budget:
    """What `resources.measure`/`apply` hand back, with only the fields the rows read."""

    disk_reserve_gb = 20.0
    vram_cap_gb = 3.0

    def __init__(self, refuse: bool = False):
        self.refuse = refuse

    def check_disk(self, path: str) -> float:
        if self.refuse:
            # The owner's own refusal, which is a `SystemExit` - the row has to survive it to report it.
            raise SystemExit(f"refusing to write: free space on {path} is 4.0 GB and the reserve is 20 GB")
        return 54.6

    def batch_size(self, imgsz: int, per_image_mb: float = 0.0, overhead_mb: float = 0.0) -> int:
        return 8 if imgsz <= train_model.IMGSZ else 4


def _budget(monkeypatch, *, device: str = "0", refuse: bool = False) -> _Budget:
    """The machine, faked at the three functions the rows call - torch is why this is faked at all."""
    budget = _Budget(refuse=refuse)
    monkeypatch.setattr(resources, "resolve_device", lambda spec: device)
    monkeypatch.setattr(resources, "measure", lambda *a, **k: budget)
    monkeypatch.setattr(resources, "apply", lambda b, d: b)
    return budget


# --- the staged set: what is captured, labeled, placed and marked -----------------------


def test_the_staged_rows_read_the_store_through_the_tools_own_parser(tmp_path, monkeypatch):
    """The wiring, and the reason `label_progress.parse_args` exists: a second caller must not hand
    build that namespace, or a field added there stops reaching this page silently.
    """
    out = _staged(tmp_path)
    # A staged hard-negative set *beside* the staged session: `workspace.DEFAULT_EXTRAS` owns the
    # name, and the union is what keeps the frames that are not products in the count.
    negatives = tmp_path / "cleaned-negatives"
    negatives.mkdir()
    (negatives / workspace.MANIFEST_NAME).write_text("[]", encoding="utf-8")

    seen: dict = {}

    def spy(args, staged):
        seen["out"] = args.out
        seen["extras"] = list(args.extras)
        seen["project"] = args.project
        return _progress()

    monkeypatch.setattr(label_progress, "local_progress", spy)

    readiness.staged_rows(out)

    assert Path(seen["out"]) == out
    assert seen["extras"] == [str(negatives)]
    assert seen["project"] == "snc-grocery"  # the parser's default, not this file's guess


def test_a_decided_frame_with_no_split_blocks_the_run(tmp_path, monkeypatch):
    """`build_dataset.py` refuses a decided frame it cannot place, so `--yes` cannot start: this is a
    refusal, not a warning, and the whole point of the page is that the two are told apart."""
    out = _staged(tmp_path)
    monkeypatch.setattr(
        label_progress,
        "local_progress",
        lambda args, staged: _progress(split_stat={"train": [1, 1], "?": [2, 3]}),
    )

    placement = _rows_of(readiness.staged_rows(out), "placement")

    assert (placement.state, placement.blocks) == (FAIL, True)
    assert "2 decided frame(s) have no split" in placement.detail


def test_an_unlabeled_frame_is_unfinished_and_not_a_refusal(tmp_path, monkeypatch):
    """The distinction the tool exists for: a frame nobody has opened is work left, and a run over it
    would produce a number about a smaller dataset rather than no number at all."""
    out = _staged(tmp_path)
    monkeypatch.setattr(
        label_progress,
        "local_progress",
        lambda args, staged: _progress(summary={"total": 10, "decided": 7}),
    )

    labels = _rows_of(readiness.staged_rows(out), "labels")

    assert (labels.state, labels.blocks) == (WARN, False)
    assert labels.detail.startswith("7/10 decided, 3 left")


def test_unmarked_hard_negatives_say_what_excluding_them_costs(tmp_path, monkeypatch):
    """An unmarked frame is dropped from the version rather than trained as background - a silent
    loss, and the row is the only place the number appears next to the consequence."""
    out = _staged(tmp_path)
    monkeypatch.setattr(
        label_progress,
        "local_progress",
        lambda args, staged: _progress(cells={("negative", "unknown"): [4, 50], ("milo", "mid"): [3, 3]}),
    )

    row = _rows_of(readiness.staged_rows(out), "hard negatives")

    assert (row.state, row.blocks) == (WARN, False)
    assert "4/50 marked null" in row.detail and "excluded" in row.detail


def test_no_hard_negatives_at_all_is_named_as_absent(tmp_path, monkeypatch):
    out = _staged(tmp_path)
    monkeypatch.setattr(
        label_progress, "local_progress", lambda args, staged: _progress(cells={("milo", "mid"): [3, 3]})
    )

    row = _rows_of(readiness.staged_rows(out), "hard negatives")

    assert row.state == WARN and "none staged" in row.detail


def test_the_capture_gap_names_the_cells_that_are_at_zero(tmp_path, monkeypatch):
    """A cell at zero is a product the model is never taught at that distance, which no later number
    recovers - the grid prints `-` for it - so it is named rather than counted."""
    out = _staged(tmp_path)
    tier_a = {
        "target": 186,
        "remaining": 132,
        "cells_under_target": 6,
        "cells": [
            {"slug": "century_tuna_flakes_in_oil_155_grams", "distance": "far", "target": 40, "have": 0, "remaining": 40},
            {"slug": "silver_swan_sukang_puti_200ML", "distance": "mid", "target": 35, "have": 5, "remaining": 30},
        ],
    }
    monkeypatch.setattr(
        label_progress, "local_progress", lambda args, staged: _progress(summary={"tier_a": tier_a})
    )

    row = _rows_of(readiness.staged_rows(out), "capture gap")

    assert row.state == WARN and row.blocks is False
    assert "132 of 186" in row.detail
    assert "century_tuna_flakes_in_oil_155_grams @ far (0/40)" in row.detail
    assert "silver_swan" not in row.detail  # thin is not empty, and only empty is named


# --- the merged set: the pass gate, the stamp, the doctor --------------------------------


def test_the_pass_gate_row_carries_the_trainers_own_sentence(tmp_path, monkeypatch):
    """Delegation, pinned: the wording belongs to the gate, and a paraphrase here could say `clean`
    over a set the trainer refuses."""
    dataset = _dataset(tmp_path)
    monkeypatch.setattr(
        train_model,
        "pass_gate",
        lambda path: (False, ["pass gate: a sentence only the trainer says"]),
    )

    row = _rows_of(readiness.dataset_rows(dataset, "auto"), "pass gate")

    assert (row.state, row.blocks) == (FAIL, True)
    assert row.detail == "pass gate: a sentence only the trainer says"
    assert row.lines == ("pass gate: a sentence only the trainer says",)


def test_a_report_with_unread_boxes_in_the_measured_splits_fails_the_real_pass_gate(tmp_path):
    """The real gate, not a fake: `train_model.pass_gate` over a report whose `test` holds two
    machine-only decisions - which is the state `accept_v2` refuses to quote a number over."""
    dataset = _dataset(tmp_path, machine_only={"test": 2})

    row = _rows_of(readiness.dataset_rows(dataset, "auto"), "pass gate")

    assert (row.state, row.blocks) == (FAIL, True)
    assert "machine-only" in row.detail


def test_a_current_stamp_passes_and_names_its_own_evidence(tmp_path):
    """The clean answer of `annotation_state_gate` is an empty list, so the row states the file and
    digest it was checked against rather than leaving the cell blank."""
    dataset = _dataset(tmp_path)

    row = _rows_of(readiness.dataset_rows(dataset, "auto"), "merge currency")

    assert (row.state, row.blocks) == (OK, False)
    assert "provenance.json unchanged" in row.detail and "sha " in row.detail


def test_a_merge_built_before_the_last_labeling_session_blocks_the_run(tmp_path):
    """The silent one: the counts in the report describe the moment of the build, and a person opening
    the annotator afterwards changes the answer without touching the merged set."""
    dataset = _dataset(tmp_path, sha="0" * 64)

    row = _rows_of(readiness.dataset_rows(dataset, "auto"), "merge currency")

    assert (row.state, row.blocks) == (FAIL, True)
    assert "changed after this set was built" in row.detail


def test_a_set_with_no_merge_report_is_unfinished_rather_than_clean(tmp_path):
    """Not a refusal - `pass_gate` calls this same state \"not this gate's set\", because a Roboflow
    export has no provenance to read - but not a pass either: nothing records which label state the
    set came from, so a stale one is indistinguishable from a current one."""
    dataset = _dataset(tmp_path, reported=False)

    rows = readiness.dataset_rows(dataset, "auto")

    assert _rows_of(rows, "merge currency").state == WARN
    assert _rows_of(rows, "merge currency").blocks is False
    assert "no merge_report.json" in _rows_of(rows, "merge currency").detail
    assert _rows_of(rows, "pass gate").state == OK  # and the gate says so in its own words


def test_the_doctor_is_asked_about_the_generation_the_set_itself_declares(tmp_path, monkeypatch):
    """`auto` is the doctor's own resolution. A second reading of `data.yaml`/`merge_report.json` here
    would eventually disagree with the verdict this row exists to report."""
    dataset = _dataset(tmp_path)
    asked: dict = {}

    monkeypatch.setattr(
        dataset_doctor, "resolve_generation", lambda path: (generations.V2, "from its data.yaml")
    )

    def fake_check(path, generation, duplicates=True, exports=None, decided_by="as asked"):
        asked.update(generation=generation, decided_by=decided_by, duplicates=duplicates)
        return {"names": list(generation.classes), "generation": generation.name, "splits": {}, "errors": [], "warnings": []}

    monkeypatch.setattr(dataset_doctor, "check_dataset", fake_check)

    row = _rows_of(readiness.dataset_rows(dataset, "auto"), "doctor")

    assert asked["generation"] is generations.V2
    assert asked["decided_by"] == "from its data.yaml"
    assert asked["duplicates"] is True  # the scan a run performs, not a cheaper one
    assert row.state == OK and "7 class(es) in v2's order" in row.detail


def test_the_doctors_block_is_captured_and_shown_only_when_the_row_fails(tmp_path, monkeypatch, capsys):
    """The check is data, but not silent: the frame-geometry measurement prints as it goes. Captured so
    the page stays one page, and carried into the row so a failure is readable without a second command.
    """
    dataset = _dataset(tmp_path)

    def noisy(path, generation, duplicates=True, exports=None, decided_by="as asked"):
        print("  frames  640x640 x12 of 12 sampled: no constant border")
        return {"names": [], "generation": "v2", "splits": {}, "errors": [], "warnings": []}

    monkeypatch.setattr(dataset_doctor, "check_dataset", noisy)
    passing = _rows_of(readiness.dataset_rows(dataset, "auto"), "doctor")
    assert passing.state == OK
    assert "frames" not in capsys.readouterr().out  # nothing leaked into the page
    assert passing.lines == ()

    def failing(path, generation, duplicates=True, exports=None, decided_by="as asked"):
        print("  frames  no images to sample")
        return {"names": [], "generation": "v2", "splits": {}, "errors": ["no train/images directory"], "warnings": []}

    monkeypatch.setattr(dataset_doctor, "check_dataset", failing)
    broken = _rows_of(readiness.dataset_rows(dataset, "auto"), "doctor")

    assert (broken.state, broken.blocks) == (FAIL, True)
    assert broken.lines and all(line.strip() for line in broken.lines)  # no blank rows between them
    assert any("no images to sample" in line for line in broken.lines)
    assert any("no train/images directory" in line for line in broken.lines)


def test_a_doctor_refusal_is_the_gates_own_block(tmp_path, monkeypatch):
    """The failing row shows `make doctor`'s lines rather than a count of them: the findings name the
    files, and a reader should not have to run the command again to see them."""
    dataset = _dataset(tmp_path)
    monkeypatch.setattr(
        dataset_doctor,
        "check_dataset",
        lambda *a, **k: {
            "names": [],
            "generation": "v2",
            "splits": {},
            "errors": ["no train/images directory in the export"],
            "warnings": [],
        },
    )

    row = _rows_of(readiness.dataset_rows(dataset, "auto"), "doctor")

    assert (row.state, row.blocks) == (FAIL, True)
    assert any("no train/images directory" in line for line in row.lines)
    assert "make doctor" in row.detail


# --- the machine: disk and card ----------------------------------------------------------


def test_the_budget_rows_are_the_machines_own_refusal(tmp_path, monkeypatch):
    """`check_disk` refuses by raising, and the sentence it raises is the one `--yes` would print a
    second later - so the row reports it rather than dying on it."""
    _budget(monkeypatch, refuse=True)

    disk = _rows_of(readiness.budget_rows(tmp_path / "runs"), "disk")

    assert (disk.state, disk.blocks) == (FAIL, True)
    assert "the reserve is 20 GB" in disk.detail


def test_no_card_is_a_warning_and_a_card_reports_the_batch_it_would_get(tmp_path, monkeypatch):
    _budget(monkeypatch, device="0")
    ok = readiness.budget_rows(tmp_path / "runs")
    assert _rows_of(ok, "vram").state == OK
    assert "batch 8 at 640 px (4 at 960)" in _rows_of(ok, "vram").detail

    _budget(monkeypatch, device="cpu")
    rows = readiness.budget_rows(tmp_path / "runs")
    assert _rows_of(rows, "vram").state == WARN
    assert "no card is visible" in _rows_of(rows, "vram").detail


def test_the_disk_row_measures_the_nearest_real_directory(tmp_path, monkeypatch):
    """The run project does not exist until the first run creates it, and `disk_usage` raises on a
    missing path - so the measurement walks up and says which directory it used."""
    _budget(monkeypatch)
    parent = tmp_path / "datasets"
    parent.mkdir()

    row = _rows_of(readiness.budget_rows(parent / "runs"), "disk")

    assert row.state == OK and str(parent) in row.detail


# --- the verdict ------------------------------------------------------------------------


def _verdict(monkeypatch, tmp_path, **by_name) -> tuple[int, str]:
    """`main` over hand-made rows, so the assembly can be tested without a dataset behind it."""
    staged = _staged(tmp_path)
    dataset = _dataset(tmp_path)
    rows = [readiness.Row(name, state, f"{name} detail", blocks=blocks) for name, (state, blocks) in by_name.items()]
    monkeypatch.setattr(readiness, "staged_rows", lambda path: rows[:1])
    monkeypatch.setattr(readiness, "dataset_rows", lambda path, generation: rows[1:2])
    monkeypatch.setattr(readiness, "budget_rows", lambda path: rows[2:])
    code = readiness.main(
        ["--out", str(staged), "--dataset-dir", str(dataset), "--run-project", str(tmp_path / "runs")]
    )
    return code, str(dataset)


def test_a_blocking_gate_exits_nonzero_and_names_itself(tmp_path, monkeypatch, capsys):
    code, _ = _verdict(monkeypatch, tmp_path, a=(FAIL, True), b=(WARN, False))

    assert code == 1
    out = capsys.readouterr().out
    assert "NOT READY TO TRAIN: 1 gate(s) refuse this run - a" in out
    assert "`train_model.py --yes` exits 2 on these" in out


def test_warnings_alone_do_not_stop_a_run_unless_strict(tmp_path, monkeypatch, capsys):
    """The two questions, and the flag that switches which one is being asked."""
    code, _ = _verdict(monkeypatch, tmp_path, a=(OK, False), b=(WARN, False))
    assert code == 0
    printed = capsys.readouterr().out
    assert "no gate refuses this run" in printed
    assert "1 warning(s) - the run would proceed and the dataset is unfinished: b" in printed

    staged = tmp_path / "cleaned-v2"
    dataset = tmp_path / "merged-v2"
    code = readiness.main(
        [
            "--out", str(staged), "--dataset-dir", str(dataset),
            "--run-project", str(tmp_path / "runs"), "--strict",
        ]
    )
    assert code == 1
    assert "--strict: an unfinished dataset is a failure here" in capsys.readouterr().out


def test_every_row_at_target_says_so(tmp_path, monkeypatch, capsys):
    code, _ = _verdict(monkeypatch, tmp_path, a=(OK, False), b=(OK, False), c=(OK, False))

    assert code == 0
    out = capsys.readouterr().out
    assert "no gate refuses this run" in out
    assert "nothing unfinished: every row is at its target." in out


def test_the_json_verdict_is_the_table(tmp_path, monkeypatch, capsys):
    """One reader for a script and one for a person, over the same rows - a second derivation of
    "ready" is how `--json` would come to disagree with the page it is meant to be."""
    code, _ = _verdict(monkeypatch, tmp_path, a=(FAIL, True), b=(WARN, False))
    table = capsys.readouterr().out
    assert code == 1
    assert "NOT READY TO TRAIN: 1 gate(s) refuse this run - a" in table
    assert "b detail" in table

    staged = tmp_path / "cleaned-v2"
    dataset = tmp_path / "merged-v2"
    code = readiness.main(
        ["--out", str(staged), "--dataset-dir", str(dataset), "--run-project", str(tmp_path / "runs"), "--json"]
    )
    printed = capsys.readouterr().out

    assert code == 1
    body = json.loads(printed)  # nothing but the document: no header a script would have to skip
    assert body["blocking"] == ["a"] and body["warnings"] == ["b"] and body["ready"] is False


@pytest.mark.parametrize("missing", ["staged", "dataset"])
def test_a_missing_input_is_cannot_look_rather_than_not_ready(tmp_path, monkeypatch, missing, capsys):
    """Exit 2, not 1: "I could not look" and "the data is not ready" send the operator to two different
    places - and a page of `0/0 decided` over a directory that does not exist reads as finished.
    """
    staged = tmp_path / "cleaned-v2"
    dataset = tmp_path / "merged-v2"
    staged.mkdir()
    dataset.mkdir()
    (staged / workspace.MANIFEST_NAME).write_text("[]", encoding="utf-8")
    if missing == "staged":
        (staged / workspace.MANIFEST_NAME).unlink()
    else:
        dataset.rmdir()

    code = readiness.main(["--out", str(staged), "--dataset-dir", str(dataset)])

    assert code == 2
    printed = capsys.readouterr().out
    assert ("not a staged set" in printed) if missing == "staged" else ("nothing to train on yet" in printed)


def test_it_writes_nothing(tmp_path, monkeypatch):
    """Read-only by construction: the snapshot the panel reads is `label_progress.py`'s artifact, and a
    readiness check that rewrote it would make the panel's numbers depend on who looked last."""
    out = _staged(tmp_path)
    before = {path.name for path in out.iterdir()}

    monkeypatch.setattr(label_progress, "local_progress", lambda args, staged: _progress())
    readiness.staged_rows(out)

    assert {path.name for path in out.iterdir()} == before
    assert not (out / "label_progress.json").exists()


def test_the_staged_rows_read_a_real_store(tmp_path):
    """End to end over a genuine staged set and annotator tree - `_local_args` and `_staged_local` are
    the fixtures the label-progress tests use, so this pins that the parser wiring above produces the
    same read `label_progress.py` does by hand."""
    out = _staged_local(tmp_path, frames=3)
    progress = label_progress.local_progress(_local_args(out), out)

    assert progress.summary["total"] == 3 and progress.summary["decided"] == 0
