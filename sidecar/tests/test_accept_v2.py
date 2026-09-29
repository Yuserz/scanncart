"""Tests for `sidecar/tools/accept_v2.py` - the command that decides whether v2 replaces v1.

No GPU, no weights, no dataset workspace: the validation pass takes an injectable `yolo` (the same
seam `train_model.validate` has) and the counting passes take an injectable `predict` factory, so
the whole command runs against fakes and a four-frame dataset under `tmp_path`.

Three rules are asserted here rather than anywhere else:

* **A gate that cannot be evaluated is a failure.** The machine-only check reads the merge report,
  and a set whose report does not carry the field - or that carries no stamp naming the annotation
  state those counts came from, or one whose stamp no longer matches `provenance.json` - has not
  passed, it has not been asked. The one outcome this command must never produce is a number over a
  split it did not check.
* **Both weights are measured here, on the same frames.** The baseline is not a quoted number from
  v1's own export; `measure` is called for each weight with the same `data.yaml`, split and `imgsz`.
* **The crowding claim is a count.** Frames with two or more items, per weight, over the same file
  list - and `--iou-sweep` reports whether that count is the model's or the threshold's.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import accept_v2
import generations

V2 = generations.V2


# --------------------------------------------------------------------------
# Fixtures: a four-frame merged set, and two fakes
# --------------------------------------------------------------------------


def _jpeg(path: Path) -> None:
    """A real JPEG with content on its edges: `check_export` samples frames to read the geometry,
    and a solid colour would be reported as a padded (letterboxed) frame."""
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.frombuffer(os.urandom(64 * 64 * 3), dtype=np.uint8).reshape(64, 64, 3)
    Image.fromarray(pixels).save(path, quality=60)


# The distances the fixture's two `test` frames are filed at: one `mid`, one `far`, the two cells
# the crowding claim is about. A real merged set always carries this block - `build_dataset.py`
# records the distance it filed each v2 frame under - so a fixture without it is the *old* build,
# which is a case worth being able to ask for: `distances={}`.
DEFAULT_DISTANCES = {"mid": ["test_0000.jpg"], "far": ["test_0001.jpg"]}


def _dataset(
    tmp_path: Path,
    machine_only: dict | None = {"valid": 0, "test": 0},
    distances: dict | None = None,
) -> Path:
    """The shapes `build_dataset.py` writes: three splits, a data.yaml, the merge report, and the
    annotation stamp its machine-only counts came from.

    `machine_only=None` drops the counts field, which is the "cannot answer" shape the gate's own
    tests use; the stamp is written either way, because a current build writes both. The tests that
    need the pre-stamp report delete it explicitly.

    `distances=None` is `DEFAULT_DISTANCES` (one frame per claimed cell, so the end-to-end tests
    accept); `distances={}` is the set that records no distance axis at all.
    """
    dataset = tmp_path / "merged-v2"
    frames: list[str] = []
    for split, count in (("train", 2), ("valid", 1), ("test", 2)):
        (dataset / split / "labels").mkdir(parents=True, exist_ok=True)
        for index in range(count):
            name = f"{split}_{index:04d}.jpg"
            frames.append(name)
            _jpeg(dataset / split / "images" / name)
            (dataset / split / "labels" / f"{split}_{index:04d}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    import yaml

    (dataset / "data.yaml").write_text(
        yaml.safe_dump({"path": str(dataset), "nc": len(V2.classes), "names": list(V2.classes)}),
        encoding="utf-8",
    )
    # The annotator's tree, beside the staged set rather than inside the merged one - the layout
    # `build_dataset.py` reads and stamps.
    provenance = tmp_path / "annotations-v2" / "provenance.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps({name: {"state": "labeled", "rows": 1, "machine_only": False} for name in frames}),
        encoding="utf-8",
    )
    report: dict = {
        "classes": list(V2.classes),
        "splits": {},
        "annotations": {
            "provenance": str(provenance),
            "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        },
    }
    if machine_only is not None:
        report["machine_only_by_split"] = machine_only
    # `{name: distance}` as `build_dataset` writes it: one entry per v2 frame, and no entry for a
    # frame the set did not contribute (v1's export carries no tags).
    mapping = DEFAULT_DISTANCES if distances is None else distances
    report["distances"] = {
        name: distance for distance, names in mapping.items() for name in names
    }
    (dataset / "merge_report.json").write_text(json.dumps(report), encoding="utf-8")
    return dataset


def _report(dataset: Path) -> dict:
    """The set's own merge report, as `accept_v2` reads it."""
    return json.loads((dataset / "merge_report.json").read_text(encoding="utf-8"))


def _mark_machine_only(report: dict, name: str) -> None:
    """A decision saved *after* the build: the annotator rewrites `provenance.json`, while the
    merged set and its report stay exactly as they were - the state the stamp exists to notice."""
    path = Path(report["annotations"]["provenance"])
    body = json.loads(path.read_text(encoding="utf-8"))
    body[name]["machine_only"] = True
    path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")


class _FakeBox:
    """The subset of ultralytics' `Metric` the recall report reads - see `train_model.per_class_recall`
    for why `r` is positional against `ap_class_index`."""

    def __init__(self, index, recall):
        self.ap_class_index = index
        self.r = recall
        self.map50 = 0.9
        self.map = 0.7
        self.mp = 0.9
        self.mr = 0.9


class _FakeModel:
    def __init__(self, names, recalls, counts, frames):
        self.names = names
        self.recalls = recalls
        self.counts = counts
        # One entry per frame: the class names "found" on it.
        self.frames = frames
        self.val_calls: list[dict] = []
        self.predict_calls: list[dict] = []

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        return SimpleNamespace(
            names=self.names,
            box=_FakeBox(list(range(len(self.recalls))), list(self.recalls)),
            nt_per_class=list(self.counts),
        )

    def predict(self, image, **kwargs):
        self.predict_calls.append({"image": image, **kwargs})
        found = self.frames.get(Path(image).name, [])
        return [SimpleNamespace(boxes=[SimpleNamespace(cls=index) for index in found])]


class _FakeYolo:
    """`yolo(path) -> model`, with a different answer per weights file."""

    def __init__(self, models: dict[str, _FakeModel]):
        self.models = models

    def __call__(self, path):
        return self.models[Path(path).name]


def _recalls(value: float, classes: tuple[str, ...] = V2.classes) -> list[float]:
    return [value] * len(classes)


def _models(
    baseline_recall: float = 0.80,
    candidate_recall: float = 0.90,
    baseline_frames: dict | None = None,
    candidate_frames: dict | None = None,
) -> _FakeYolo:
    names = {i: name for i, name in enumerate(V2.classes)}
    counts = [4] * len(V2.classes)
    return _FakeYolo(
        {
            "v1.pt": _FakeModel(names, _recalls(baseline_recall), counts, baseline_frames or {}),
            "v2.pt": _FakeModel(names, _recalls(candidate_recall), counts, candidate_frames or {}),
        }
    )


# A second class order for the baseline, standing in for v1's: the two generations' lists are the
# same seven names in different positions, and this permutation moves index 0 onto the name the set
# declares at index 2 - so a name read through the wrong list is visible in the output rather than a
# difference nobody could see.
BASELINE_ORDER = (V2.classes[2], V2.classes[0], V2.classes[1], *V2.classes[3:])


def _weights(
    tmp_path: Path, baseline_order: tuple[str, ...] | None = None
) -> tuple[Path, Path]:
    """Two weights with the record `--install` writes beside each: the set's own class list.

    `baseline_order` is the other-order case, and the record is where both the order and its absence
    come from - a `.pt` carries neither, so a fixture without one is a weight nothing can be
    compared against (`weight_order` refuses it).
    """
    baseline = tmp_path / "v1.pt"
    candidate = tmp_path / "v2.pt"
    baseline.write_bytes(b"weights")
    candidate.write_bytes(b"weights")
    for path, order in ((baseline, baseline_order or V2.classes), (candidate, V2.classes)):
        path.with_suffix(".json").write_text(
            json.dumps({"class_names": list(order)}), encoding="utf-8"
        )
    return baseline, candidate


def _run(tmp_path: Path, yolo, **over) -> int:
    """`main()` with the flags a test cares about; the rest come from the defaults.

    The dataset is resolved before the default is built: `over.pop("dataset", _dataset(tmp_path))`
    would *evaluate the default anyway*, rebuilding the set (and overwriting its merge report) on
    every run - which silently turned a gated fixture back into a clean one here.
    """
    dataset = over.pop("dataset", None)
    baseline, candidate = over.pop("weights", None) or _weights(tmp_path)
    argv = [
        "--dataset",
        str(dataset if dataset is not None else _dataset(tmp_path)),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--run-project",
        str(tmp_path / "runs"),
    ]
    for key, value in over.items():
        flag = f"--{key.replace('_', '-')}"
        # A store_true flag takes no value, and argparse would read `True` as a stray argument.
        argv += [flag] if value is True else [flag, str(value)]
    return accept_v2.main(argv, yolo=yolo)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_the_gate_treats_a_report_that_cannot_answer_as_unverified_not_clean():
    """`recorded`'s rule, applied to a split: absence is not a pass. A report from before the
    field existed, or a hand-made set with no report at all, has not been asked."""
    ok, lines = accept_v2.machine_only_gate({})
    assert ok is False
    assert "cannot verify" in lines[0] and "machine_only_by_split" in lines[0]

    ok, lines = accept_v2.machine_only_gate({"machine_only_by_split": {"valid": 0, "test": 0}})
    assert ok is True
    assert "no machine-only decisions" in lines[0]

    # A missing split is zero only when the field itself is there: the field is the evidence.
    ok, _ = accept_v2.machine_only_gate({"machine_only_by_split": {}})
    assert ok is True


def test_the_gate_fails_on_any_unreviewed_decision_in_a_measured_split():
    ok, lines = accept_v2.machine_only_gate({"machine_only_by_split": {"train": 9, "test": 2}})
    assert ok is False
    assert any("2 machine-only decision(s) in `test`" in line for line in lines)
    # `train` is deliberately not part of the gate: those boxes are cheap.
    assert not any("train" in line for line in lines)

    ok, lines = accept_v2.machine_only_gate({"machine_only_by_split": {"test": True}})
    assert ok is False and any("unusable count" in line for line in lines)
    ok, _ = accept_v2.machine_only_gate({"machine_only_by_split": {"test": -1}})
    assert ok is False


def test_a_report_that_names_no_annotation_state_cannot_clear_a_set(tmp_path):
    """The defect this closed, kept as the regression: `machine_only_by_split` present and zero is
    *not* enough, because a report built before the annotator recorded those decisions looks
    identical - and on this very workspace that is what let a set with unreviewed boxes in
    `valid`/`test` through. With no stamp the zero cannot be checked, so the verdict is a refusal,
    and the counts half on its own still says clean - which is exactly why the hole existed.
    """
    report = _report(_dataset(tmp_path))
    del report["annotations"]

    assert accept_v2.machine_only_gate(report)[0] is True  # the check that was not enough
    ok, lines = accept_v2.machine_only_verdict(report)
    assert ok is False
    assert "cannot verify" in lines[0] and "rebuild" in lines[0]


def test_a_report_whose_annotations_moved_on_cannot_clear_the_set(tmp_path):
    """The other half, and the one a stamp alone would not catch: the report says zero, the stamps
    match nothing because a machine decision was saved in `test` *after* the build. The counts are
    a measurement of the build's moment, so the gate has to re-read the state rather than trust
    the arithmetic."""
    dataset = _dataset(tmp_path)
    report = _report(dataset)
    _mark_machine_only(report, "test_0000.jpg")

    assert accept_v2.machine_only_gate(report)[0] is True
    ok, lines = accept_v2.machine_only_verdict(report)
    assert ok is False
    assert any("changed after this set was built" in line for line in lines)


def test_a_stamp_that_still_matches_is_clean(tmp_path):
    """And the check is not "always refuse": the untouched fixture clears both halves, so the two
    refusals above are about the stamp, not about the check existing."""
    ok, lines = accept_v2.machine_only_verdict(_report(_dataset(tmp_path)))
    assert ok is True
    assert "no machine-only decisions" in lines[0]


def test_a_stamp_whose_annotations_are_gone_is_a_refusal(tmp_path):
    """The annotation tree can be moved, wiped with the staged set, or sit on a drive that is not
    attached - and in every one of those the counts beside it are uncheckable rather than clean."""
    report = _report(_dataset(tmp_path))
    Path(report["annotations"]["provenance"]).unlink()

    ok, lines = accept_v2.annotation_state_gate(report)
    assert ok is False
    assert "not readable" in lines[0]


@pytest.mark.parametrize(
    "stamp",
    [None, "provenance.json", {}, {"provenance": ""}, {"provenance": "nowhere.json"}, {"sha256": "x"}, 7],
)
def test_a_stamp_that_cannot_be_read_is_a_refusal_rather_than_a_crash(tmp_path, stamp):
    """Shapes a hand-edited, half-written or older report can carry. Every one of them is a refusal
    with a sentence, never an exception and never a pass - the file this reads is outside the set
    and nothing about it is guaranteed."""
    report = {"machine_only_by_split": {"valid": 0, "test": 0}, "annotations": stamp}
    ok, lines = accept_v2.annotation_state_gate(report)
    assert ok is False and lines


def test_the_command_refuses_a_stale_or_unstamped_report(tmp_path):
    """Through `main`, the real entry point: both shapes exit non-zero rather than printing an
    acceptance verdict, which is the whole claim this fix makes about the set on disk."""
    # (a) the pre-stamp report: present-and-zero counts, nothing naming where they came from.
    unstamped = _dataset(tmp_path / "a")
    report = _report(unstamped)
    del report["annotations"]
    (unstamped / "merge_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert _run(tmp_path / "a", _models(0.80, 0.95), dataset=unstamped) == 1

    # (b) the stamped report whose decisions moved on after the build.
    stale = _dataset(tmp_path / "b")
    _mark_machine_only(_report(stale), "test_0000.jpg")
    assert _run(tmp_path / "b", _models(0.80, 0.95), dataset=stale) == 1


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


def test_a_class_below_the_floor_and_a_class_worse_than_v1_fail_for_different_reasons():
    """The two failures are different findings: one says the model cannot do the class, the other
    says it lost ground. Collapsing them would hide which one the fix is about."""
    passed, failed, unmeasured = accept_v2.compare(
        baseline={"milo": 0.90, "tuna": 0.70, "sardines": 0.95, "odd": None},
        candidate={"milo": 0.91, "tuna": 0.88, "sardines": 0.80, "odd": None},
        floor=0.85,
        tolerance=0.02,
    )

    assert any("tuna: 0.880 >= 0.85, v1 0.700" in line for line in passed)
    # Below the floor *and* better than v1: still a failure - the floor is the bar.
    assert any("tuna" not in line for line in failed)
    assert any("sardines: 0.800 < 0.85 floor, v1 0.950" in line for line in failed)
    # A class the split never asked about joins neither list.
    assert any("odd" in line for line in unmeasured)
    assert not any("odd" in line for line in passed + failed)


def test_a_regression_inside_the_tolerance_is_not_a_regression():
    """"Worse than v1" cannot mean "0.001 worse": the floor is the bar, and two measurements of the
    same weights on the same split differ by more than nothing."""
    passed, failed, _ = accept_v2.compare(
        {"milo": 0.90}, {"milo": 0.89}, floor=0.85, tolerance=0.02
    )
    assert passed and not failed

    passed, failed, _ = accept_v2.compare(
        {"milo": 0.90}, {"milo": 0.87}, floor=0.85, tolerance=0.02
    )
    assert not passed
    assert any("regressed from v1's 0.900" in line for line in failed)


def test_a_class_v1_was_never_measured_on_is_a_pass_without_a_win():
    """Crediting the candidate with a win over a missing number is how a gate stops meaning
    anything: v1 has no measurement there to lose against."""
    passed, failed, _ = accept_v2.compare({}, {"milo": 0.90}, floor=0.85, tolerance=0.02)

    assert any("v1 not measured" in line for line in passed)
    assert not failed


def test_a_class_the_candidate_never_scored_is_unmeasured_rather_than_zero():
    """A 0 would read as a total miss, and "add images of that item" and "add captures to that
    split" are opposite instructions - the three states `--val` already keeps apart."""
    passed, failed, unmeasured = accept_v2.compare(
        {"milo": 0.90}, {"milo": None}, floor=0.85, tolerance=0.02
    )
    assert not passed and not failed
    assert "no ground-truth instances" in unmeasured[0]


# --------------------------------------------------------------------------
# The crowding claim
# --------------------------------------------------------------------------


def test_crowding_counts_frames_with_two_items_and_separates_the_same_product_case():
    """Two different questions: the crowded counter this dataset exists for, and the
    near-duplicate SKU case that logs one physical item twice (4's warning)."""
    counts = {
        "one_item.jpg": ["milo"],
        "two_items.jpg": ["milo", "safeguard"],
        "two_milo.jpg": ["milo", "milo"],
        "none.jpg": [],
    }

    block = accept_v2.crowding(counts)

    assert block["frames"] == 4
    assert block["crowded_frames"] == 2
    assert block["crowded_names"] == ["two_items.jpg", "two_milo.jpg"]
    assert block["same_product_frames"] == {"milo": 1}
    assert block["detections"] == 5  # 1 + 2 + 2 + 0


def test_the_counting_passes_use_the_same_file_list_and_the_same_thresholds():
    """A count that ran over a different list, or at a different `iou`, would not be a comparison
    - so the seam is asserted on what was asked for, not only on what came back."""
    seen: list[tuple] = []

    def predict(image, conf, iou):
        seen.append((Path(image).name, conf, iou))
        return ["milo"]

    images = [Path("a.jpg"), Path("b.jpg")]

    got = accept_v2.frame_instances(predict, images, 0.5, 0.7)

    assert got == {"a.jpg": ["milo"], "b.jpg": ["milo"]}
    assert seen == [("a.jpg", 0.5, 0.7), ("b.jpg", 0.5, 0.7)]


def test_the_sweep_note_says_whether_the_count_is_the_model_or_the_threshold():
    """This is the sentence `--iou-sweep` exists to produce: if the bucket moves with NMS iou it
    has to be quoted with it, and if it does not, a later change to the detector's NMS cannot
    rewrite the acceptance."""
    flat = {0.5: {"crowded_frames": 12}, 0.7: {"crowded_frames": 12}, 0.9: {"crowded_frames": 12}}
    moved = {0.5: {"crowded_frames": 9}, 0.7: {"crowded_frames": 12}, 0.9: {"crowded_frames": 14}}

    assert "does not move" in accept_v2.sweep_note(flat)
    assert "0.5:12" in accept_v2.sweep_note(flat)
    assert "moves" in accept_v2.sweep_note(moved)
    assert "0.9:14" in accept_v2.sweep_note(moved)


def test_the_crowding_claim_is_made_per_distance_and_the_total_cannot_answer_it():
    """The blind spot the per-distance rows exist for: a total that *rises* while the cell the
dataset was re-shot for falls. `far` loses both its crowded frames and the whole-split rule is
perfectly happy, so the claim is made on the distances and not on the total."""
    distances = {"a.jpg": "close", "b.jpg": "close", "c.jpg": "far", "d.jpg": "far"}
    baseline_counts = {"a.jpg": [], "b.jpg": [], "c.jpg": ["milo", "safeguard"], "d.jpg": ["milo", "milo"]}
    candidate_counts = {"a.jpg": ["milo", "milo"], "b.jpg": ["milo", "milo"], "c.jpg": [], "d.jpg": ["milo"]}

    baseline = accept_v2.crowd_by_distance(baseline_counts, distances)
    candidate = accept_v2.crowd_by_distance(candidate_counts, distances)
    passed, failed = accept_v2.distance_claim(baseline, candidate, ("mid", "far"), "test")

    # The rule that already existed: v2 finds more crowded frames overall, so it does not fire.
    assert accept_v2.crowding(candidate_counts)["crowded_frames"] == 2
    assert accept_v2.crowding(baseline_counts)["crowded_frames"] == 2
    assert baseline["far"]["crowded_frames"] == 2
    assert candidate["far"]["crowded_frames"] == 0
    assert passed == []
    assert any("crowding at `far`" in line and "v1 on 2" in line for line in failed)
    # And `mid` files no frame at all here, which is a different failure from losing the row: the
    # split cannot speak for the cell, so the claim was never answered.
    assert any("crowding at `mid`" in line and "not measured" in line for line in failed)


def test_a_claimed_distance_the_split_never_files_is_a_failure_rather_than_a_pass():
    """`machine_only_gate`'s rule, asked of a cell: "cannot be asked" is not "held". Every other
row can hold and the run still refuses, because the claim was made on a distance this split has
nothing to say about."""
    distances = {"a.jpg": "close", "b.jpg": "close"}
    counts = {"a.jpg": ["milo", "milo"], "b.jpg": ["milo", "safeguard"]}

    passed, failed = accept_v2.distance_claim(
        accept_v2.crowd_by_distance(counts, distances),
        accept_v2.crowd_by_distance(counts, distances),
        ("mid", "far"),
        "test",
    )

    assert passed == []
    assert len(failed) == 2
    assert all("not measured" in line and "test split files no frame" in line for line in failed)


def test_the_unattributed_row_keeps_the_per_distance_table_adding_up():
    """v1's export frames carry no distance tag (the axis was added for v2), so most of a real split
lands in `unattributed`. It is counted rather than dropped - rows that did not add up to the total
would read as a fault in the counting - and the row is never a cell, so it is never a pass."""
    counts = {"a.jpg": ["milo"], "b.jpg": ["milo", "milo"], "c.jpg": ["milo"]}
    distances = {"a.jpg": "close"}

    blocks = accept_v2.crowd_by_distance(counts, distances)

    assert set(blocks) == {"close", "unattributed"}
    assert blocks["unattributed"]["frames"] == 2
    assert sum(block["frames"] for block in blocks.values()) == len(counts)
    # The file lists group the same way, which is what lets the sweep run one row at a time.
    groups = accept_v2.group_images(list(Path(name) for name in counts), distances)
    assert [p.name for p in groups["close"]] == ["a.jpg"]
    assert {p.name for p in groups["unattributed"]} == {"b.jpg", "c.jpg"}


def test_an_unknown_claim_distance_is_refused_before_anything_is_measured():
    """A typo would otherwise be a claim *about nothing*: the unknown name has no frames, so the
run would fail with a sentence about re-shooting a cell nobody has a folder for. The axis is named
in the refusal, and `none` is the documented way to make no claim."""
    with pytest.raises(SystemExit) as caught:
        accept_v2.main(["--baseline", "a.pt", "--candidate", "b.pt", "--claim-distances", "middle"])

    assert "unknown --claim-distances" in str(caught.value)
    assert "close, mid, far" in str(caught.value)
    assert accept_v2.claim_distance_list("none") == ()
    assert accept_v2.claim_distance_list(" mid , close ") == ("mid", "close")


def test_the_acceptance_refuses_a_set_whose_v2_frames_are_all_one_distance(tmp_path, capsys):
    """The state `merged-v2` was in before the mid/far capture sessions were decided, and the reason
this table exists: every class clears its floor, the crowded count rises, and the run still refuses
- because the two cells the claim is about were never filed, so nothing above them is evidence about
the question the re-shoot was paid for."""
    crowding = {"test_0000.jpg": [0], "test_0001.jpg": [0, 1]}
    close_only = {"close": ["test_0000.jpg", "test_0001.jpg"]}

    code = _run(
        tmp_path,
        _models(0.80, 0.90, crowding, crowding),
        dataset=_dataset(tmp_path, distances=close_only),
        json=True,
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "by distance" in out and "crowding at `far`: not measured" in out
    # The verdict has to read as one: a run that says "accepted:" and exits 1 is the output that
    # would make an operator think the claim was made.
    assert not any(line.startswith("accepted:") for line in out.splitlines())

    # `--claim-distances none` is the run that is not making the claim, and it makes the same set a
    # pass: without a claim there is nothing for an absent cell to fail.
    assert _run(
        tmp_path,
        _models(0.80, 0.90, crowding, crowding),
        dataset=_dataset(tmp_path, distances={}),
        claim_distances="none",
    ) == 0


# --------------------------------------------------------------------------
# The class order
# --------------------------------------------------------------------------


def test_remap_labels_moves_the_class_column_and_nothing_else(tmp_path):
    """The whole view in miniature. The labels are read from the `images` sibling - the rule
    ultralytics reads them by, which is what lets a view move the labels without moving the frames -
    the box columns come through untouched, and a frame with no label file is written as an empty
    one rather than left out (background stays background)."""
    images = tmp_path / "test" / "images"
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _jpeg(images / name)
    labels = tmp_path / "test" / "labels"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("2 0.5 0.5 0.2 0.2\n0 0.1 0.9 0.05 0.05\n", encoding="utf-8")
    (labels / "b.txt").write_text("1 0.25 0.25 0.5 0.5\n", encoding="utf-8")

    files, rows = accept_v2.remap_labels(images, tmp_path / "view" / "labels", {0: 3, 1: 0, 2: 1})

    assert (files, rows) == (3, 3)
    view = tmp_path / "view" / "labels"
    assert view.joinpath("a.txt").read_text(encoding="utf-8") == (
        "1 0.5 0.5 0.2 0.2\n3 0.1 0.9 0.05 0.05\n"
    )
    assert view.joinpath("b.txt").read_text(encoding="utf-8") == "0 0.25 0.25 0.5 0.5\n"
    assert view.joinpath("c.txt").read_text(encoding="utf-8") == ""


def test_a_label_row_naming_a_class_the_set_does_not_declare_is_a_refusal(tmp_path):
    """A row the mapping cannot cover is a broken set, not a row to drop: passing it through would
    measure the model against boxes attributed to nobody."""
    images = tmp_path / "test" / "images"
    _jpeg(images / "a.jpg")
    labels = tmp_path / "test" / "labels"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        accept_v2.remap_labels(images, tmp_path / "view" / "labels", {0: 0})

    assert "no entry of the set's declared class list" in str(caught.value)


def test_a_weight_with_no_recorded_class_order_is_refused(tmp_path):
    """Assuming it matches the set is exactly the mistake: the run reports a number, no error
    appears anywhere, and the number is about the mapping rather than the model."""
    baseline, candidate = _weights(tmp_path)
    baseline.with_suffix(".json").write_text(json.dumps({"resize_mode": "stretch"}), encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        accept_v2.main(
            [
                "--dataset",
                str(_dataset(tmp_path)),
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--run-project",
                str(tmp_path / "runs"),
            ],
            yolo=_models(),
        )

    assert "no class order is recorded" in str(caught.value)


def test_a_weight_whose_class_list_is_not_a_reordering_is_refused(tmp_path):
    """Two lists that do not name the same products have no remap between them, and no number here
    would mean anything until they do - so both sides are printed."""
    weights = _weights(tmp_path, baseline_order=(*V2.classes[:-1], "some_other_product"))

    with pytest.raises(SystemExit) as caught:
        accept_v2.main(
            [
                "--dataset",
                str(_dataset(tmp_path)),
                "--baseline",
                str(weights[0]),
                "--candidate",
                str(weights[1]),
                "--run-project",
                str(tmp_path / "runs"),
            ],
            yolo=_models(),
        )

    message = str(caught.value)
    assert "not a reordering of the set's" in message
    assert "some_other_product" in message and V2.classes[-1] in message
    # Which side each name is on, not only that both are printed: the two directions are opposite
    # findings (this weight predicts a product the set has no ground truth for; the set holds one
    # it cannot predict) and the refusal names them under separate headings. The relation is asked
    # of `generations.class_gaps` as `class_gaps(set_order, order)`, so this is what pins the
    # argument order as well as the two sentences.
    assert "it predicts \n  some_other_product\nthat the set has no ground truth for" in message
    assert f"the set has \n  {V2.classes[-1]}\nthat it cannot predict" in message


def test_a_baseline_in_another_order_is_measured_on_a_remapped_view(tmp_path, capsys):
    """The measurement the view exists for: same weights, same frames, same boxes, and the class
    column in the order that weight's head counts in. The candidate - trained on this set - needs no
    view, so the two passes are handed different yamls and only one of them is a view."""
    import yaml

    weights = _weights(tmp_path, baseline_order=BASELINE_ORDER)
    yolo = _models(0.80, 0.90, {}, {})

    code = _run(tmp_path, yolo, weights=weights, json=True)
    out = capsys.readouterr().out
    summary = json.loads(out[out.rindex("\n{") + 1 :])

    assert code == 0
    assert "remapped into it" in out
    baseline_yaml = yolo.models["v1.pt"].val_calls[0]["data"]
    candidate_yaml = yolo.models["v2.pt"].val_calls[0]["data"]
    assert baseline_yaml != candidate_yaml
    assert summary["orders"]["baseline"]["remapped"] is True
    assert summary["orders"]["candidate"]["remapped"] is False
    assert summary["orders"]["baseline"]["order"] == list(BASELINE_ORDER)
    # The view is the same frames. Its `names` are the baseline's order, and its labels index that
    # order - the two facts that make the pass a measurement of the model rather than of the map.
    body = yaml.safe_load(Path(baseline_yaml).read_text(encoding="utf-8"))
    view = Path(baseline_yaml).parent
    assert body["names"] == list(BASELINE_ORDER)
    assert sorted(p.name for p in (view / "test" / "images").iterdir()) == [
        "test_0000.jpg",
        "test_0001.jpg",
    ]
    # `test_0000` was staged as `mid` under the set's index 0; in the baseline's order that name is
    # index `BASELINE_ORDER.index(V2.classes[0])`, which is where its box has to be for v1's head to
    # be credited with it.
    assert (view / "test" / "labels" / "test_0000.txt").read_text(encoding="utf-8") == (
        f"{BASELINE_ORDER.index(V2.classes[0])} 0.5 0.5 0.2 0.2\n"
    )


def test_the_counted_names_come_from_each_weights_own_order(tmp_path, capsys):
    """The other surface the order decides: a box's *name* is `classes[index]`, so the baseline's
    near-duplicate finding has to be read through the baseline's list. The count of frames is the
    same either way - which product it names is not, and the product is the thing being reported."""
    weights = _weights(tmp_path, baseline_order=BASELINE_ORDER)
    # The same two boxes of the same index from both weights, so the counts are equal and the only
    # thing that can differ is the name each weight's own list gives index 0.
    yolo = _models(0.80, 0.90, {"test_0000.jpg": [0, 0]}, {"test_0000.jpg": [0, 0]})

    code = _run(tmp_path, yolo, weights=weights, json=True)
    out = capsys.readouterr().out
    summary = json.loads(out[out.rindex("\n{") + 1 :])

    assert code == 0
    assert summary["crowding"]["baseline"]["same_product_frames"] == {BASELINE_ORDER[0]: 1}
    assert summary["crowding"]["candidate"]["same_product_frames"] == {V2.classes[0]: 1}
    assert BASELINE_ORDER[0] != V2.classes[0]


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_a_set_the_doctor_refuses_is_never_measured(tmp_path):
    """The strongest form of the promise: not "the verdict is discarded" but "no number exists".

    Each of these trains and validates happily, so nothing downstream would notice them - and an
    acceptance number computed over one is a measurement of the *set*, quoted as if it were the
    model's. The fake weight counts its calls, so "refused" is asserted as "never asked".
    """
    import yaml

    for label, mutate in (
        (
            "reversed class order",
            lambda d: (d / "data.yaml").write_text(
                yaml.safe_dump({"names": list(reversed(list(V2.classes)))}), encoding="utf-8"
            ),
        ),
        (
            "a label row the loader drops",
            lambda d: (d / "train" / "labels" / "train_0000.txt").write_text(
                "0 0.1 0.1 0.5 0.1 0.5\n", encoding="utf-8"
            ),
        ),
        (
            "a test frame that duplicates a train one",
            lambda d: (d / "test" / "images" / "test_0000.jpg").write_bytes(
                (d / "train" / "images" / "train_0000.jpg").read_bytes()
            ),
        ),
        (
            "a label file that is not there at all",
            lambda d: (d / "train" / "labels" / "train_0000.txt").unlink(),
        ),
    ):
        dataset = _dataset(tmp_path / label.replace(" ", "-"))
        mutate(dataset)
        yolo = _models(0.80, 0.95)

        with pytest.raises(SystemExit) as caught:
            _run(tmp_path / label.replace(" ", "-"), yolo, dataset=dataset)

        assert "doctor refused it" in str(caught.value), label
        assert yolo.models["v1.pt"].val_calls == [] and yolo.models["v2.pt"].val_calls == [], label
        assert yolo.models["v2.pt"].predict_calls == [], label


def test_the_acceptance_passes_when_the_gate_is_clean_and_v2_beats_v1(tmp_path):
    """The whole command, with fakes: both weights measured on the same split, the gate read from
    the dataset's own report, and the crowded-frame count compared."""
    frames = {"test_0000.jpg": [0, 0], "test_0001.jpg": [0]}
    baseline_frames = {"test_0000.jpg": [0], "test_0001.jpg": [0]}
    yolo = _models(0.80, 0.90, baseline_frames, frames)

    code = _run(tmp_path, yolo)

    assert code == 0
    baseline_model = yolo.models["v1.pt"]
    candidate_model = yolo.models["v2.pt"]
    # Measured on the same split, at the same size, from the same yaml.
    assert [call["split"] for call in baseline_model.val_calls] == ["test"]
    assert baseline_model.val_calls[0]["data"] == candidate_model.val_calls[0]["data"]
    assert baseline_model.val_calls[0]["imgsz"] == candidate_model.val_calls[0]["imgsz"] == 640
    # And the counting pass ran over the test split's frames, for both weights.
    assert len(candidate_model.predict_calls) == 2
    assert len(baseline_model.predict_calls) == 2


def test_the_acceptance_fails_on_an_unreviewed_decision_even_when_the_numbers_pass(tmp_path):
    """A number measured over a weight's unread boxes says something about the annotator. The
    command must not report it as evidence about v2 - and it exits non-zero either way."""
    dataset = _dataset(tmp_path, machine_only={"valid": 0, "test": 3})
    yolo = _models(0.80, 0.95)

    code = _run(tmp_path, yolo, dataset=dataset)

    assert code == 1


def test_the_acceptance_fails_when_the_crowded_frame_count_falls(tmp_path):
    """The claim this dataset exists for, as a count: v2 finding two items on *fewer* frames than
    v1 is not an improvement, whatever the per-class numbers say."""
    baseline_frames = {"test_0000.jpg": [0, 0], "test_0001.jpg": [0]}
    candidate_frames = {"test_0000.jpg": [0], "test_0001.jpg": [0]}
    yolo = _models(0.70, 0.95, baseline_frames, candidate_frames)

    code = _run(tmp_path, yolo)

    assert code == 1


def test_the_crowding_passes_can_be_skipped_and_the_iou_sweep_is_reported(tmp_path):
    """`--no-crowding` is the cheap run, and `--iou-sweep` adds one pass per value: both have to
    work without a weight or a GPU beyond the fake."""
    yolo = _models(0.80, 0.90, {}, {})
    code = _run(tmp_path, yolo, no_crowding=True)
    assert code == 0
    assert yolo.models["v1.pt"].predict_calls == []

    yolo = _models(0.80, 0.90, {}, {})
    code = _run(tmp_path, yolo, iou_sweep=True)
    assert code == 0
    # The candidate's own counting pass over the split, then one pass per sweep value *per claimed
    # distance* - each over that row's own frames, so the sweep is about the cell the count is
    # quoted for. No more expensive than sweeping the whole split was: the rows partition it.
    calls = yolo.models["v2.pt"].predict_calls
    ious = [call["iou"] for call in calls]
    assert len(calls) == 2 + len(accept_v2.IOU_SWEEP) * len(accept_v2.CLAIM_DISTANCES)
    # The operating iou is also one of the swept values, so the set is the union rather than two
    # disjoint passes - and the first pass over the list is the acceptance's own, in file order.
    assert sorted(set(ious)) == sorted({accept_v2.IOU, *accept_v2.IOU_SWEEP})
    assert [Path(call["image"]).name for call in calls[:2]] == ["test_0000.jpg", "test_0001.jpg"]
    # One frame per sweep pass, not two: each claimed distance is swept over its own row.
    assert all(
        Path(call["image"]).name in {"test_0000.jpg", "test_0001.jpg"} for call in calls[2:]
    )
    assert {Path(call["image"]).name for call in calls[2:]} == {"test_0000.jpg", "test_0001.jpg"}


def test_missing_weights_are_a_hard_error_rather_than_a_skip(tmp_path):
    """`make verify-clamp`'s rule: a skip here is a silent pass on the only command that says
    whether v2 may replace v1."""
    with pytest.raises(SystemExit) as caught:
        accept_v2.main(
            [
                "--dataset",
                str(_dataset(tmp_path)),
                "--baseline",
                str(tmp_path / "absent.pt"),
                "--candidate",
                str(tmp_path / "also-absent.pt"),
            ],
            yolo=_models(),
        )
    assert "no baseline weights" in str(caught.value)

    baseline, candidate = _weights(tmp_path)
    with pytest.raises(SystemExit) as caught:
        accept_v2.main(["--baseline", str(baseline)], yolo=_models())
    assert "--baseline and --candidate" in str(caught.value)


def test_the_measured_size_comes_from_the_candidates_own_record(tmp_path):
    """`--imgsz` defaults to what the weights recorded, not to the module constant: a model
    trained at 960 and measured at 640 is a different configuration, and the number it reports is
    not the model's."""
    baseline, candidate = _weights(tmp_path)
    candidate.with_suffix(".json").write_text(json.dumps({"imgsz": 960}), encoding="utf-8")

    assert accept_v2.imgsz_of(candidate) == 960
    assert accept_v2.imgsz_of(baseline) == 640  # no record, so the run default
    candidate.with_suffix(".json").write_text(json.dumps({"imgsz": 961}), encoding="utf-8")
    # Off the stride grid is not a size this app can be set to, so it does not get used here
    # either - the app owns that rule, and this reads it rather than restating it.
    assert accept_v2.imgsz_of(candidate) == 640


def test_the_counted_class_names_come_from_the_datasets_own_order(tmp_path):
    """The head indexes the export's order, and `check_export` compares the two lists by
    *membership* - so a generation list in another order would relabel every counted detection with
    nothing erroring."""
    import yaml

    dataset = _dataset(tmp_path)
    reversed_names = list(reversed(V2.classes))
    (dataset / "data.yaml").write_text(
        yaml.safe_dump({"names": reversed_names}), encoding="utf-8"
    )

    assert accept_v2.class_names_for(dataset, V2) == tuple(reversed_names)
    # And with no readable list at all, the generation's is the fallback rather than a crash.
    (dataset / "data.yaml").unlink()
    assert accept_v2.class_names_for(dataset, V2) == tuple(V2.classes)
