"""Tests for `build_dataset.py`: the v1 export merged with v2 local labels into one set.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import pytest
import accept_v2
import annotate.store
import build_dataset
import clean_v2
import dataset_doctor
import generations
import label_classes
import label_progress
import plan_split
import train_model
import workspace

from tests.dataset_tool_helpers import _export_with_names, _staged_local


# --------------------------------------------------------------------------
# 3b. the merge: v1's export + v2's local labels -> one set
# --------------------------------------------------------------------------


def test_a_polygon_row_is_reduced_to_the_box_ultralytics_would_see():
    """1,921 of v1's 2,111 label rows are polygons, not boxes, and ultralytics converts them at load
    time (`segments2boxes`, gated on any row having more than six fields). Reducing them here is what
    makes the merged label files readable - a polygon written through verbatim draws nonsense on the
    contact sheet, which is the one check that can see a wrong class mapping.
    """
    import build_dataset

    # v1's own `data.yaml` order, which is *not* the canonical one - the translation is by name.
    names = ["555 sardines 155grams", "Bear Brand Fortified Powdered Milk 33g"]

    box, problem, was_polygon = build_dataset.remap_row(["0", "0.5", "0.5", "0.2", "0.1"], names)
    assert (problem, was_polygon) == ("", False)
    assert box == f"{build_dataset.INDEX_BY_NAME[names[0]]} 0.5 0.5 0.2 0.1"

    # A triangle: min/max over the x's and the y's is the box ultralytics builds from the same row.
    polygon, problem, was_polygon = build_dataset.remap_row(
        ["1", "0.2", "0.4", "0.6", "0.4", "0.4", "0.8"], names
    )
    assert (problem, was_polygon) == ("", True)
    fields = polygon.split()
    assert fields[0] == str(build_dataset.INDEX_BY_NAME[names[1]])
    assert [round(float(v), 6) for v in fields[1:]] == [0.4, 0.6, 0.4, 0.4]


def test_a_row_that_cannot_be_translated_is_a_problem_and_not_a_dropped_box():
    """Silently losing a row changes what the model is trained on, which is the failure class this
    module exists to make impossible - so every unmappable shape is a refusal the build prints."""
    import build_dataset

    names = list(build_dataset.CANONICAL_NAMES)

    # A class index the source's own list does not have, and one whose name is not ours.
    assert build_dataset.remap_row(["9", "0.5", "0.5", "0.1", "0.1"], names)[1]
    assert build_dataset.remap_row(["0", "0.5", "0.5", "0.1", "0.1"], ["Palmolive"])[1]
    assert build_dataset.remap_row(["x", "0.5", "0.5", "0.1", "0.1"], names)[1]
    # Five coordinates is neither a box nor a polygon, and an odd count is not a point list at all.
    assert build_dataset.remap_row(["0", "0.5", "0.5", "0.1", "0.1", "0.2"], names)[1]
    assert build_dataset.remap_row(["0", "0.1", "0.1", "0.2", "0.2", "0.3", "0.3", "0.4"], names)[1]

    # And a blank line is simply not a row: files end with newlines.
    assert build_dataset.remap_rows("0 0.5 0.5 0.2 0.2\n\n", names) == (
        [f"{build_dataset.INDEX_BY_NAME[names[0]]} 0.5 0.5 0.2 0.2"],
        [],
        0,
    )


def test_the_translation_verdict_is_one_rule_for_both_sides():
    """One rule and one sentence, asked of v1's export and of the annotator's tree alike.

    The fatal direction is a name the source declares with no position in this dataset's order - a
    box that can be filed under no product, so the frame cannot enter the set. The other direction
    is deliberately *not* symmetric, and the asymmetry is the rule: a class this dataset declares
    that a source has no frames of is a capture gap when the source is v1's export (`check_export`
    prints it as a note about the other generation), and a tooling mismatch when the source is the
    annotator - which draws the frames, so a class it cannot draw is one no v2 frame can ever be
    labelled as. Only that side is asked with `coverage`.
    """
    names = list(build_dataset.CANONICAL_NAMES)
    short = names[:-1]
    extra = [*names, "Palmolive Naturals Bar Soap 85g"]

    assert build_dataset.translation_problem(names, source="v1") is None
    # v1's export missing a class this dataset declares: a gap, not a refusal - and the annotator's
    # list in the same shape is a refusal, because nobody there can draw it.
    assert build_dataset.translation_problem(short, source="v1") is None
    uncovered = build_dataset.translation_problem(short, source="the annotator", coverage=True)
    assert "can never draw" in uncovered and repr(names[-1]) in uncovered

    problem = build_dataset.translation_problem(extra, source="v1")
    assert problem.startswith("v1 declares class(es) this dataset does not have")
    assert "'Palmolive Naturals Bar Soap 85g'" in problem
    # The same list judged for the annotator: the same sentence with the other subject, which is what
    # "reported the same way on either side" means for the two builders.
    assert build_dataset.translation_problem(extra, source="the annotator", coverage=True) == (
        problem.replace("v1", "the annotator")
    )
    # And the annotator's own list translates into *either* generation's order, which is the freedom
    # the by-name remap bought: v1's names are the same products in another order.
    assert build_dataset.annotator_class_problem(list(generations.V1.classes)) is None


def test_the_contact_sheet_shows_every_split_not_only_the_largest(tmp_path):
    """The sheet is the gate on the one transformation nothing else can check, and it only works if
    the frames a person has *not* seen are on it: `check_export` compares class names by membership,
    so a wrongly translated index is a valid file about the wrong product."""
    from PIL import Image as PILImage

    import build_dataset

    out = tmp_path / "merged-v2"
    for split, count in (("train", 1265), ("valid", 223), ("test", 303)):
        images, labels = build_dataset.split_dirs(out, split)
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            PILImage.new("RGB", (32, 32)).save(images / f"{split}_{index:05d}.jpg", quality=40)
            (labels / f"{split}_{index:05d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    sheet = build_dataset.contact_sheet(out, tmp_path / "sheet.jpg", 30)

    assert sheet["frames"] == 30
    # Ten each, rather than 11 train / 11 valid / 8 test - the largest split must not be able to
    # spend the sheet's slack on the frames the model already trains on.
    assert sheet["per_split"] == {"train": 10, "valid": 10, "test": 10}


def test_tier_a_is_the_same_block_from_either_source():
    """One function computes it, so a local run and a Roboflow run cannot rank the same cells
    differently - which would make the panel's next-action list depend on which tool wrote the
    file rather than on the plan."""
    # A cell from the plan itself, with some of its images staged, so the block has something to
    # rank: `have` counts the images in that cell and `remaining` is the capture gap.
    product, distance = next(iter(clean_v2.TIER_A_CELLS))
    slug = clean_v2.CLASS_MAP[product]
    target = clean_v2.TIER_A_CELLS[(product, distance)]
    block = label_progress.tier_a_block({(slug, distance): [0, 4]})

    assert block["target"] == sum(clean_v2.TIER_A_CELLS.values())
    assert [c["remaining"] for c in block["cells"]] == sorted(
        (c["remaining"] for c in block["cells"]), reverse=True
    )
    cell = next(c for c in block["cells"] if c["slug"] == slug and c["distance"] == distance)
    staged = 4 if target >= 4 else target
    assert (cell["have"], cell["remaining"]) == (staged, target - staged)


def test_auto_reads_local_only_once_a_decision_exists(tmp_path):
    """The signal is a decision, not the tool's bookkeeping.

    `classes.json` is written when the annotator *starts*, so keying on it would let a machine
    where somebody opened the worklist and went back to Roboflow answer "local" - reporting
    `0 / 1,383 decided` over a project that has annotations, and replacing that project's snapshot
    with it. Both shapes of a decision count, which is why provenance is consulted too: a null
    annotation is an empty label file.
    """
    out = _staged_local(tmp_path)
    annotations = out.parent / label_progress.ANNOTATIONS_DIRNAME

    assert label_progress.local_annotations_dir(out) == annotations
    assert label_progress.local_tree_exists(annotations) is False

    # Opened, and nothing decided: still the project's answer.
    annotations.mkdir(parents=True)
    (annotations / "classes.json").write_text("{}", encoding="utf-8")
    (annotations / "provenance.json").write_text(
        json.dumps({"milo_0001.jpg": {"suggestion": {"provider": "local:v1"}}}), encoding="utf-8"
    )
    assert label_progress.local_tree_exists(annotations) is False

    # A saved box is a decision.
    (annotations / "milo_0001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert label_progress.local_tree_exists(annotations) is True

    # And so is a null, which leaves the label file empty.
    (annotations / "milo_0001.txt").write_text("", encoding="utf-8")
    (annotations / "provenance.json").write_text(
        json.dumps({"negative_0001.jpg": {"state": "null", "provider": "unknown"}}), encoding="utf-8"
    )
    assert label_progress.local_tree_exists(annotations) is True


def test_the_hard_negatives_are_found_beside_the_set_being_read(tmp_path):
    """The default extras follow `--out`/`--v2` the way `--annotations` already does.

    Anchored on a fixed workspace path instead, a run against a set staged elsewhere - a second
    attempt, a test's temporary copy - still pulled *this* workspace's 50 hard negatives into its
    worklist, its snapshot and its build: 50 frames the operator does not have, and the ones that
    set actually means left out. That is how the local-snapshot end-to-end test in
    `test_dataset_status.py` began counting 52 frames where it had staged 2.
    """
    beside = tmp_path / "second-attempt"
    beside.mkdir()

    assert workspace.default_extras(beside) == []  # nothing staged beside it yet
    (beside / "cleaned-negatives").mkdir()
    # A folder is not a set: the manifest is what a reader can actually open.
    assert workspace.default_extras(beside) == []
    (beside / "cleaned-negatives" / "manifest.json").write_text("[]", encoding="utf-8")

    # The names are the workspace-level list, so the two anchors cannot disagree about *what* the
    # extras are - only about where they were staged.
    assert workspace.default_extras(beside) == [beside / name for name in workspace.DEFAULT_EXTRAS]


def test_a_named_extra_adds_to_the_staged_negatives_rather_than_replacing_them(tmp_path):
    """`--extras` is additive: the default is a second *set*, not a fallback value.

    Every reader that takes an `--extras` used to treat an explicit value as a replacement, so
    naming a later session's set - the ordinary thing to do once `s3` exists - silently dropped the
    50 frames whose whole job is to teach the model what these products are not. Nothing downstream
    said so: the snapshot, the merge and the build all report a smaller, perfectly consistent set.
    Replacing them is still reachable, as a decision rather than as a side effect of naming a
    directory, through `include_defaults=False` (`--no-extras` on the tools).
    """
    # The container, not the set: `resolve_extras` is handed `--out`'s parent, which is where the
    # defaults sit - the same anchor `--annotations` follows.
    beside = tmp_path
    assert workspace.resolve_extras(beside, ["elsewhere"]) == [Path("elsewhere")]

    negatives = tmp_path / "cleaned-negatives"
    (negatives / "negative").mkdir(parents=True)
    (negatives / "manifest.json").write_text("[]", encoding="utf-8")
    other = tmp_path / "cleaned-v2-s3"

    # Defaults first, then what was named, and the named set is not resolved through the manifest
    # filter - the caller asked for it, and a reader that skipped a manifest-less directory would
    # report a lower total with no explanation.
    assert workspace.resolve_extras(beside, [str(other)]) == [negatives, other]
    # The replacement, said out loud.
    assert workspace.resolve_extras(beside, [str(other)], include_defaults=False) == [other]
    assert workspace.resolve_extras(beside, [], include_defaults=False) == []
    # Naming the default itself is not a way to double it: these lists are walked, and a set read
    # twice would report every frame in it twice.
    assert workspace.resolve_extras(beside, [str(negatives)]) == [negatives]
    assert workspace.resolve_extras(beside, [str(negatives), str(other), str(other)]) == [negatives, other]


def _staged_negatives(root: Path, frames: int = 2, name: str = "cleaned-negatives") -> Path:
    """The hard-negative set as `clean_v2 clean --negatives` stages it: its own directory, its own
    manifest, images off the app's camera under the `negative` pseudo-class and no distance."""
    out = root / name
    (out / "negative").mkdir(parents=True)
    entries = []
    for i in range(1, frames + 1):
        frame = f"cam0_{i:04d}.jpg"
        (out / "negative" / frame).write_bytes(b"jpeg")
        entries.append({"new_name": frame, "class": "negative", "distance": "", "session": "s2", "batch": "negative"})
    (out / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return out


def _decide(annotations: Path, names: list[str], null: bool = False) -> None:
    """One frame's label file: boxes, or the *empty* file that means "there is nothing here"."""
    annotations.mkdir(parents=True, exist_ok=True)
    for name in names:
        (annotations / f"{Path(name).stem}.txt").write_text(
            "" if null else "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )


def test_the_snapshot_still_counts_the_negatives_when_another_set_is_named(tmp_path):
    """Through `label_progress.main`: the reader the Admin Panel and the worklist both come from.

    The snapshot is the number the panel shows and the file `app/dataset_status.py` serves, so a
    set dropped from it is a set nobody knows is missing - `total` just reads lower.
    """
    out = _staged_local(tmp_path, frames=3)
    negatives = _staged_negatives(tmp_path, frames=2)
    s3 = _staged_local(tmp_path / "s3-root", frames=1, slug="bear-brand-milk")
    annotations = tmp_path / label_progress.ANNOTATIONS_DIRNAME
    _decide(annotations, ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"])
    _decide(annotations, ["cam0_0001.jpg", "cam0_0002.jpg"], null=True)
    _decide(annotations, ["bear-brand-milk_0001.jpg"])  # s3's own frame, in the one flat label tree

    assert label_progress.main(["--source", "local", "--out", str(out), "--extras", str(s3)]) == 0
    summary = json.loads((out / "label_progress.json").read_text(encoding="utf-8"))
    assert summary["total"] == 6  # 3 staged + 2 negatives + 1 named
    assert summary["null_annotations"] == 2  # the negatives were counted, not just totalled

    # The decision to leave them out is available and is a flag, not a consequence of naming s3.
    assert label_progress.main(["--source", "local", "--no-extras", "--out", str(out), "--extras", str(s3)]) == 0
    summary = json.loads((out / "label_progress.json").read_text(encoding="utf-8"))
    assert summary["total"] == 4
    assert negatives.exists()  # nothing was deleted - the run just read a smaller set


def test_a_build_that_names_a_set_still_merges_the_negatives(tmp_path, capsys):
    """Through `build_dataset.main`: the merge is where a dropped set stops being recoverable.

    The dry run reads the manifests and the split lookup and nothing else, so this is the same
    count a real build would write (`preview_v2` is `build_v2` without the writes).
    """
    out = _staged_local(tmp_path, frames=3)
    positives = [f"milo_{i:04d}.jpg" for i in (1, 2, 3)]
    negatives = [f"cam0_{i:04d}.jpg" for i in (1, 2)]
    _staged_negatives(tmp_path, frames=2)
    s3 = _staged_local(tmp_path / "s3-root", frames=1, slug="bear-brand-milk")
    s3_frame = "bear-brand-milk_0001.jpg"

    annotations = tmp_path / label_progress.ANNOTATIONS_DIRNAME
    _decide(annotations, positives)
    _decide(annotations, negatives, null=True)
    _decide(annotations, [s3_frame])  # s3's own frame, in the one flat label tree
    (out / "splits.json").write_text(
        json.dumps({name: "train" for name in [*positives, *negatives, s3_frame]}), encoding="utf-8"
    )

    def merged_train() -> int:
        """The dry run's own train count - the field, not the padding around it."""
        match = re.search(r"^\s+v2 train\s+(\d+) decided", capsys.readouterr().out, re.M)
        assert match, "the dry run printed no v2 train row"
        return int(match.group(1))

    base = ["--dry-run", "--no-v1", "--v2", str(out), "--extras", str(s3)]
    assert build_dataset.main(base) == 0
    assert merged_train() == 6  # 3 staged + 2 negatives + 1 named

    assert build_dataset.main([*base, "--no-extras"]) == 0
    assert merged_train() == 4


def _frames(*slugs: str) -> list:
    """The one attribute `v2_set_problem` reads off a frame - a whole `Frame` would be scaffolding."""
    from types import SimpleNamespace

    return [SimpleNamespace(slug=slug) for slug in slugs]


def test_a_v2_that_is_not_a_staged_set_is_named_rather_than_merged(tmp_path):
    """The verdict is about the directory, so all of its answers are pinned here.

    `--v2` is read through a *manifest*, so a wrong one contributes zero frames and no error: the
    merged set keeps v1's numbers and the seven right names, the doctor still prints `[ok]`, and one
    capture is simply absent from a set everyone believes is v2's. The v1 side has failed closed on
    a wrong directory since it was written (its export is checked split by split); this is that rule
    for the side whose evidence is a manifest.
    """
    v2 = tmp_path / "cleaned-v2"
    v2.mkdir()

    # Never staged: the manifest is what `clean_v2.py clean` writes to make a folder a set.
    problem = build_dataset.v2_set_problem(v2, [])
    assert "no manifest.json" in problem
    assert "--v1" not in problem  # nothing about it looks like an export yet

    # ...and the same directory once it does: an export has the right names in a `data.yaml`, which
    # is what makes this mistake read as a staged set until the manifest is asked for.
    (v2 / "data.yaml").write_text("names: [a, b]\n", encoding="utf-8")
    assert "downloaded export" in build_dataset.v2_set_problem(v2, [])

    # Two different accidents, so two different sentences: one has to be re-staged, the other was
    # staged and is empty.
    (v2 / "manifest.json").write_text("{not json", encoding="utf-8")
    assert "could not be read" in build_dataset.v2_set_problem(v2, [])
    (v2 / "manifest.json").write_text("[]", encoding="utf-8")
    assert "names no frame" in build_dataset.v2_set_problem(v2, [])

    # Names frames that are not on disk - which is what an empty frame list *with* a full manifest
    # means, and why the count alone cannot say which of these happened.
    (v2 / "manifest.json").write_text(json.dumps([{"new_name": "milo_0001.jpg"}]), encoding="utf-8")
    assert "names 1 frame(s)" in build_dataset.v2_set_problem(v2, [])

    # The hard-negative set, named here by a slip for `--extras` - the one shape that does have
    # frames, so the count alone would look like a perfectly good set of 50.
    problem = build_dataset.v2_set_problem(v2, _frames("negative", "negative"))
    assert "clean_v2.py clean --negatives" in problem and "--extras" in problem

    # And a staged set: nothing to say, whatever its decisions are (that is `preview_v2`'s note).
    assert build_dataset.v2_set_problem(v2, _frames("milo", "bear-brand-milk")) is None


def test_a_v2_that_is_not_a_staged_set_refuses_before_force_deletes_the_last_build(tmp_path, capsys):
    """The refusal is the point; the *order* is what keeps a typo from being destructive.

    `--force` wipes the three split directories before it merges, and it is the exact command an
    operator re-runs - so a `--v2` typo found after the wipe would leave the last good merged set
    deleted with nothing written in its place.
    """
    out = tmp_path / "merged-v2"
    (out / "train" / "images").mkdir(parents=True)
    (out / "train" / "images" / "milo_0001.jpg").write_bytes(b"jpeg")
    (out / "MERGE_REPORT.md").write_text("last build", encoding="utf-8")
    typo = tmp_path / "cleaned-v2-typo"
    typo.mkdir()

    # The dry run reports it the same way it reports every other problem: exit 2, nothing written.
    assert build_dataset.main(["--dry-run", "--no-v1", "--v2", str(typo), "--out", str(out)]) == 2
    assert "no manifest.json" in capsys.readouterr().out

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(["--no-v1", "--v2", str(typo), "--out", str(out), "--force"])
    assert "no manifest.json" in str(refused.value)
    # The last build is untouched: this is what the pre-flight ordering buys.
    assert (out / "train" / "images" / "milo_0001.jpg").is_file()
    assert (out / "MERGE_REPORT.md").read_text(encoding="utf-8") == "last build"


def test_the_negatives_named_as_v2_are_refused_as_the_wrong_side(tmp_path, capsys):
    """A real set with a real manifest, named one flag over from the one that reads it.

    Two surfaces for the same verdict: the dry run through `preview_v2` (exit 2, in the problem
    list), and a real build, which refuses in the pre-flight - a second rather than after 1,815 v1
    frames are rewritten - so `out` is not even created.
    """
    negatives = _staged_negatives(tmp_path, frames=2)
    out = tmp_path / "merged-v2"

    assert build_dataset.main(["--dry-run", "--no-v1", "--v2", str(negatives), "--out", str(out)]) == 2
    printed = capsys.readouterr().out
    assert "hard-negative" in printed and "--extras" in printed

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(["--no-v1", "--v2", str(negatives), "--out", str(out), "--force"])
    assert "not rebuilding from this --v2" in str(refused.value)
    assert "hard-negative" in str(refused.value)
    assert not out.exists()
    assert not build_dataset.staging_dir(out).exists()


def test_a_v2_whose_images_are_gone_says_so_rather_than_merging_nothing(tmp_path, capsys):
    """A manifest naming frames the disk no longer has - a workspace moved in pieces, or a set
    copied without its class folders. The frames are located *through* the manifest, so this reads
    as a set with nothing in it, which is the same silent merge by a different route."""
    v2 = _staged_local(tmp_path, frames=3)
    for i in (1, 2, 3):
        (v2 / "milo" / f"milo_{i:04d}.jpg").unlink()

    assert build_dataset.main(
        ["--dry-run", "--no-v1", "--v2", str(v2), "--out", str(tmp_path / "merged-v2")]
    ) == 2
    assert "names 3 frame(s) and not one of them could be read" in capsys.readouterr().out


def _jpeg(path: Path, box: int = 0) -> None:
    """A real, decodable JPEG - `_staged_local`'s `b"jpeg"` stands in for a *staged* frame, but a
    build rewrites the frames it merges, so a test that runs one needs bytes PIL will open.

    `box` slides a white square across the frame, because a build fingerprints every train/valid
    frame and drops test frames that look like them: two frames made by this helper at the same
    offset are duplicates of each other, and a test whose only test frame is a duplicate would
    watch the build drop it and then fail the doctor on an empty `test/`.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (64, 48), (30, 40, 50))
    ImageDraw.Draw(image).rectangle([box, box, box + 24, box + 24], fill=(240, 240, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=80)


def _staged_with_real_frames(root: Path, names: list[str], splits: dict[str, str] | None = None) -> Path:
    """A staged set the build can actually merge, with `splits.json` placing the frames it names."""
    out = root / "cleaned-v2"
    entries = []
    for index, name in enumerate(names):
        _jpeg(out / "milo" / name, box=6 * index)
        entries.append(
            {"new_name": name, "class": "milo", "distance": "mid", "session": "s2", "batch": "milo_mid"}
        )
    (out / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    if splits is not None:
        (out / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return out


def _provenance(annotations: Path, names: list[str], machine_only: tuple[str, ...] = ()) -> Path:
    """The annotator's file, in the shape `LabelStore` writes: name -> record.

    `_decide_all` writes the label rows a save leaves; this writes the provenance beside them,
    which is what a build stamps and what `machine_only` is read from. Frames named in
    `machine_only` stand for a decision that was saved under a provider's suggestion and never
    confirmed - the state the acceptance gate exists to refuse.
    """
    annotations.mkdir(parents=True, exist_ok=True)
    path = annotations / workspace.PROVENANCE_NAME
    body = {
        name: {"state": "labeled", "rows": 1, "machine_only": name in machine_only}
        for name in names
    }
    path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    return path


def _decide_all(annotations: Path, names: list[str]) -> None:
    """One box per frame, drawn under the class its manifest tags it as (`milo`, v2's order).

    Not class 0 for everything: `_staged_with_real_frames` tags every frame `milo`, and the doctor's
    tag check refuses a label drawn under any other class - so a fixture that drew one class
    regardless would fail the very build these tests are about. A correct staged set is what they
    stand for, so the drawing agrees with the tag.
    """
    index = build_dataset.INDEX_BY_NAME[label_classes.SLUG_TO_CLASS["milo"]]
    annotations.mkdir(parents=True, exist_ok=True)
    for name in names:
        (annotations / f"{Path(name).stem}.txt").write_text(
            f"{index} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )


def _previous_build(out: Path) -> None:
    """A merged set as a finished build leaves one: frames, a report, and the set's own yaml."""
    _jpeg(out / "train" / "images" / "old_train.jpg")
    (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (out / "train" / "labels" / "old_train.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (out / "data.yaml").write_text("path: somewhere/else\nnames: [milo]\n", encoding="utf-8")
    (out / "MERGE_REPORT.md").write_text("stale report", encoding="utf-8")


def _tree(directory: Path) -> dict[str, bytes]:
    """Every file under `directory`, by relative path and content, so "untouched" is one assert."""
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_a_refused_rebuild_leaves_the_previous_set_exactly_as_it_was(tmp_path, capsys):
    """`--force` assembles beside the set and swaps in only on success, so refusing costs nothing.

    The old order deleted the three split directories *before* merging, which made the command an
    operator re-runs the one that could destroy the only copy of a two-minute build: one decided
    frame with no split in `splits.json` was enough. Here the refusal happens after frames have
    already been written into the staging copy - so this is the half-built case, not the easy one.
    """
    out = tmp_path / "merged-v2"
    _previous_build(out)
    before = _tree(out)

    # One frame the plan places, one it does not: the second is what makes the build refuse, and by
    # then the first has been rewritten into the staging copy.
    v2 = _staged_with_real_frames(tmp_path, ["milo_0001.jpg", "milo_0002.jpg"], {"milo_0001.jpg": "train"})
    _decide_all(tmp_path / "annotations-v2", ["milo_0001.jpg", "milo_0002.jpg"])

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out), "--force"])
    assert "refusing to write a dataset" in str(refused.value)
    assert "no split" in capsys.readouterr().out

    assert _tree(out) == before  # byte for byte: the frames, the report, the yaml, all of it
    assert not build_dataset.staging_dir(out).exists()  # and the half-built copy is gone


def test_a_rebuild_swaps_in_wholesale_and_leaves_everything_else_alone(tmp_path, capsys):
    """The other half: what a *successful* rebuild does to the set it replaces.

    A split directory is replaced rather than merged into - a frame the new build does not write
    must not survive inside the new `train/` from the old one - while a file the build knows nothing
    about is left where it is, which is the one promise `--force` has always made. All three splits,
    because the doctor refuses a set that could not be measured on one of them.
    """
    out = tmp_path / "merged-v2"
    _previous_build(out)
    (out / "operator-notes.md").write_text("kept by hand", encoding="utf-8")

    frames = ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"]
    v2 = _staged_with_real_frames(
        tmp_path, frames, {name: split for name, split in zip(frames, ("train", "valid", "test"))}
    )
    _decide_all(tmp_path / "annotations-v2", frames)

    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out), "--force"]) == 0
    printed = capsys.readouterr().out
    assert "the doctor, on the copy that has not landed yet (judged against v2's order):" in printed
    assert "[ok] the set is internally consistent" in printed

    assert (out / "train" / "images" / "milo_0001.jpg").is_file()
    assert (out / "train" / "labels" / "milo_0001.txt").is_file()
    assert not (out / "train" / "images" / "old_train.jpg").exists()
    assert (out / "valid" / "images" / "milo_0002.jpg").is_file()
    assert (out / "test" / "images" / "milo_0003.jpg").is_file()
    assert (out / "operator-notes.md").read_text(encoding="utf-8") == "kept by hand"
    assert (out / "MERGE_REPORT.md").read_text(encoding="utf-8") != "stale report"
    assert (out / "contact_sheet.jpg").is_file()

    # Both yamls carry *absolute* paths, so neither may name the staging directory: `data.yaml` is
    # written into the staging copy with the final paths in it (it is what the doctor reads), and
    # `data.scanncart.yaml` is written after the swap for the same reason.
    for name in ("data.yaml", "data.scanncart.yaml"):
        body = (out / name).read_text(encoding="utf-8")
        assert str(out.resolve()) in body
        assert ".building" not in body

    assert not build_dataset.staging_dir(out).exists()
    assert not (tmp_path / ".merged-v2.previous").exists()  # the aside copy exists only mid-swap


def test_the_merge_report_records_each_written_frame_class(tmp_path):
    """The built set holds no manifest, so the report is the last place a frame's staged class
    exists - and `dataset_doctor.py` reads it back to refuse a label drawn under a neighbour's
    class. A frame the build did not write is not in the map: a report describing frames the set
    does not hold would be describing a different build, which is the one thing it may not do.
    """
    out = tmp_path / "merged-v2"
    frames = ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg", "milo_0004.jpg"]
    v2 = _staged_with_real_frames(
        tmp_path,
        frames,
        {
            "milo_0001.jpg": "train",
            "milo_0002.jpg": "test",
            "milo_0003.jpg": "valid",
            "milo_0004.jpg": "test",
        },
    )
    # A test frame that is a copy of a train one, so the build drops it - and its tag with it.
    # The other test frame is the one the doctor needs for its own checks to have a test split.
    (v2 / "milo" / "milo_0002.jpg").write_bytes((v2 / "milo" / "milo_0001.jpg").read_bytes())
    _decide_all(tmp_path / "annotations-v2", frames)

    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out)]) == 0
    report = json.loads((out / "merge_report.json").read_text(encoding="utf-8"))

    assert report["tags"] == {
        "milo_0001.jpg": "milo",
        "milo_0003.jpg": "milo",
        "milo_0004.jpg": "milo",
    }
    assert [d["name"] for d in report["dropped_test_duplicates"]] == ["milo_0002.jpg"]
    # The distance rides along the same way, per frame: the dropped test frame's `mid` is not in
    # the map, and this map is what `--val` reads when the staged manifest is not beside the set.
    assert report["distances"] == {
        "milo_0001.jpg": "mid",
        "milo_0003.jpg": "mid",
        "milo_0004.jpg": "mid",
    }
    # And the per-split mix the doctor prints is derived from that map - the same fact, summed
    # into the shape a coverage question can be asked of, so the two cannot disagree.
    assert report["sources"]["v2"]["distances"] == {
        "train": {"mid": 1},
        "valid": {"mid": 1},
        "test": {"mid": 1},
    }
    # The side's own count is the frames *in the set*, not the ones it handed over - the dropped
    # test frame is not v2's any more. Counting what was placed is what made a real report read
    # `v2: test 83` on one line and `test close 36` on the next.
    assert report["sources"]["v2"]["images"] == {"train": 1, "valid": 1, "test": 1}
    assert "tags:" in (out / "MERGE_REPORT.md").read_text(encoding="utf-8")


def test_the_report_stamps_the_annotation_state_it_read(tmp_path):
    """The acceptance gate has to be able to ask "are these machine-only counts still true?" - and
    the report is the only thing that travels with the set, so the build records the identity of
    the `provenance.json` it read (path plus digest), the same way `--val` records the checkpoint it
    measured. Asserted through the gate's own reader rather than against this test's arithmetic: a
    stamp only this test agreed with could still be one `accept_v2` cannot check.
    """
    out = tmp_path / "merged-v2"
    frames = ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"]
    v2 = _staged_with_real_frames(
        tmp_path, frames, dict(zip(frames, ("train", "valid", "test")))
    )
    annotations = tmp_path / "annotations-v2"
    _decide_all(annotations, frames)
    provenance = _provenance(annotations, frames)

    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out)]) == 0
    report = json.loads((out / "merge_report.json").read_text(encoding="utf-8"))

    stamp = report["annotations"]
    assert Path(stamp["provenance"]) == provenance.resolve()
    assert stamp["sha256"] == hashlib.sha256(provenance.read_bytes()).hexdigest()
    assert accept_v2.annotation_state_gate(report) == (True, [])


def test_a_build_over_labels_with_no_provenance_names_no_state(tmp_path):
    """A label tree something other than the annotator wrote has no `provenance.json`, so a build
    over it can count zero machine-only decisions and still not know that. The report then names no
    state, and the gate refuses the set rather than reading the silence as a clean one - the same
    rule as a missing `machine_only_by_split`, applied to the evidence behind it.
    """
    out = tmp_path / "merged-v2"
    frames = ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"]
    v2 = _staged_with_real_frames(
        tmp_path, frames, dict(zip(frames, ("train", "valid", "test")))
    )
    _decide_all(tmp_path / "annotations-v2", frames)

    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out)]) == 0
    report = json.loads((out / "merge_report.json").read_text(encoding="utf-8"))

    assert "annotations" not in report
    ok, lines = accept_v2.annotation_state_gate(report)
    assert ok is False and "cannot verify" in lines[0]


def test_a_decision_saved_after_the_build_leaves_the_report_unmeasurable(tmp_path):
    """The failure this exists for, through both tools: the build stamps the state it counted, a
    person then saves a machine decision in `test` (the annotator rewrites its file; the merged set
    does not change), and the gate refuses the set instead of measuring it. `accept_v2`'s own tests
    assert the exit code; this is the writer/reader contract underneath it.
    """
    out = tmp_path / "merged-v2"
    frames = ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"]
    v2 = _staged_with_real_frames(
        tmp_path, frames, dict(zip(frames, ("train", "valid", "test")))
    )
    annotations = tmp_path / "annotations-v2"
    _decide_all(annotations, frames)
    provenance = _provenance(annotations, frames)

    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out)]) == 0
    report = json.loads((out / "merge_report.json").read_text(encoding="utf-8"))
    assert report["machine_only_by_split"] == {}  # nothing unreviewed when it was built
    assert accept_v2.machine_only_verdict(report)[0] is True

    body = json.loads(provenance.read_text(encoding="utf-8"))
    body["milo_0003.jpg"]["machine_only"] = True  # saved under a suggestion, in `test`
    provenance.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")

    ok, lines = accept_v2.machine_only_verdict(report)
    assert ok is False
    assert any("changed after this set was built" in line for line in lines)


def test_a_merge_that_fails_the_doctor_never_becomes_the_set(tmp_path, capsys):
    """The gate on the staging copy, which is what makes §7's check one step earlier instead of later.

    Nothing but the doctor stops this build - every frame is decided and placed - so the failure it
    reports (no `test/` to measure on) would otherwise have been found a command later, on a merged
    set that had already replaced the previous one.
    """
    out = tmp_path / "merged-v2"
    _previous_build(out)
    before = _tree(out)
    frames = ["milo_0001.jpg", "milo_0002.jpg"]
    v2 = _staged_with_real_frames(tmp_path, frames, {frames[0]: "train", frames[1]: "valid"})
    _decide_all(tmp_path / "annotations-v2", frames)

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out), "--force"])

    printed = capsys.readouterr().out
    assert "the doctor, on the copy that has not landed yet" in printed
    assert "[FAIL] no test/images directory" in printed
    assert "did not pass the doctor" in str(refused.value)
    assert _tree(out) == before  # the previous set never moved
    assert not build_dataset.staging_dir(out).exists()


def test_a_merge_whose_gate_splits_hold_unread_boxes_never_becomes_the_set(tmp_path, capsys):
    """The first of the three doors, and the one that costs nothing to close.

    `train_model --yes/--val` and `accept_v2.py` both refuse a set whose `valid`/`test` hold a
    weight's unread boxes, so building one can only replace a set that could have been used with one
    that cannot - and `--force` is exactly the command that would have done the replacing. The
    frames to work are named in the annotator, which is why the refusal points there rather than
    offering a build-side fix.
    """
    out = tmp_path / "merged-v2"
    _previous_build(out)
    before = _tree(out)
    frames = ["milo_0001.jpg", "milo_0002.jpg"]
    v2 = _staged_with_real_frames(tmp_path, frames, {frames[0]: "train", frames[1]: "test"})
    _decide_all(tmp_path / "annotations-v2", frames)
    _provenance(tmp_path / "annotations-v2", frames, machine_only=("milo_0002.jpg",))

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out), "--force"])

    printed = capsys.readouterr().out
    assert "the human pass's gate, on the copy that has not landed yet" in printed
    assert "1 machine-only decision(s) in `test`" in printed
    assert "did not pass the human pass's gate" in str(refused.value)
    assert _tree(out) == before  # the previous set never moved
    assert not build_dataset.staging_dir(out).exists()


def test_the_declaration_is_read_back_off_the_set_before_it_can_land(tmp_path, monkeypatch):
    """Two fields of one file, written by one function - and the read-back is what makes them agree.

    `dataset_doctor --generation auto` (its default, and what `make doctor` runs) resolves which
    generation's order the labels index **from the set's own `data.yaml`**, so a set whose `names`
    are one generation's list while its `generation` names another is refused a minute after the
    build reported success. The build's own doctor run cannot see it: that run is handed the
    generation the build *meant*. Each shape below is a file no build should write, and each
    sentence names the two lists so the reader can see which one moved.
    """
    import yaml

    out = tmp_path / "merged-v2"
    staging = build_dataset.staging_dir(out)
    staging.mkdir(parents=True)
    splits = {}
    for split in label_classes.SPLIT_NAMES:
        images = build_dataset.split_dirs(out, split)[0]
        images.mkdir(parents=True)
        splits[split] = images
    declaration = staging / workspace.DATA_YAML_NAME

    # What the build writes: the two fields agree, and the lists name the same products, so the
    # translation every row goes through has a destination for each of them.
    build_dataset.write_names_yaml(out, splits, path=declaration)
    assert build_dataset.declared_order_problem(staging, out) is None

    def problem(body: dict) -> str:
        declaration.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        return build_dataset.declared_order_problem(staging, out) or ""

    v1, v2 = list(generations.V1.classes), list(generations.V2.classes)
    recorded = problem({"names": v2, "generation": "v1"})
    assert "records `v1`" in recorded and "v2's order" in recorded
    assert "v1 lists: " + ", ".join(v1) in recorded
    listed = problem({"names": v1, "generation": "v2"})
    assert "records `v2`" in listed and "v1's order" in listed
    # A permutation that is nobody's list is the other half: the names themselves are the problem.
    assert "no generation's list" in problem({"names": list(reversed(v2)), "generation": "v2"})
    # And an unrecorded or unreadable declaration is refused too - the list only means something
    # against a generation, so a set naming none could only be read by guessing.
    assert "names no generation" in problem({"nc": 7, "names": v2})
    assert "names no generation" in problem({"nc": 7, "names": v2, "generation": "v9"})
    assert "declares no class list" in problem({"nc": 7, "generation": "v2"})
    assert "declares no class list" in problem({"nc": 7})

    # A build for the *other* generation's order is legal now, and that is what the remap bought:
    # both sides are translated by name, so the declaration is a choice rather than a constraint on
    # whatever order the annotator happens to write in.
    assert problem({"names": v1, "generation": "v1"}) == ""

    def register(name: str, classes: tuple[str, ...]) -> None:
        """A generation neither of today's two is - the roster shape the next addition makes."""
        monkeypatch.setitem(
            generations.GENERATIONS,
            name,
            generations.Generation(
                name=name,
                classes=classes,
                export_dir=tmp_path / f"export-{name}",
                manifest=None,
                resize_mode="stretch",
                roboflow_project="snc-grocery",
            ),
        )

    # The second clause, on its own: a declaration that agrees with the generation it records and
    # whose two lists do not name the same products. That is a roster change rather than a labelling
    # one, and it is the state the translation cannot cover - so it is refused with both directions
    # named. Registered rather than used from disk because both of *today's* generations name the
    # annotator's seven, which is exactly why this clause is a tripwire for the next one.
    extra = generations.V2.classes + ("Palmolive Naturals Bar Soap 85g",)
    register("v3", extra)
    added = problem({"names": list(extra), "generation": "v3"})
    # The sentence is the shared rule's own, asked here with the artifact's names as the target: a
    # reader meets the same wording here and in the merge that would have refused the source list.
    assert build_dataset.annotator_class_problem(list(extra)) in added
    assert "this dataset declares class(es) the annotator can never draw" in added
    assert "Palmolive Naturals Bar Soap 85g" in added

    fewer = generations.V2.classes[:-1]
    register("v4", fewer)
    dropped = problem({"names": list(fewer), "generation": "v4"})
    assert build_dataset.annotator_class_problem(list(fewer)) in dropped
    assert "this dataset does not have" in dropped
    assert generations.V2.classes[-1] in dropped


def test_both_sides_ask_the_translation_rule_before_writing_a_frame(tmp_path, capsys, monkeypatch):
    """Asked up front on either side, by the same function, and by the dry run - not row by row.

    A source list is checked against this dataset's order before the first frame is written, so a
    refusal costs nothing and leaves no half-written side behind; and it is the same rule and the
    same sentence on both sides, so an operator who has met one refusal recognises the other.
    `build_v2`'s source is the annotator's tree (`annotate.store.CLASS_NAMES`), patched here into the
    two shapes a roster change makes: a class it can draw that this dataset has no position for, and
    a class this dataset declares that it can never draw.
    """
    import yaml

    # v1's export, declaring a name with no position here.
    v1 = _export_with_names(tmp_path / "export-v1", {"train": ["a.jpg"]})
    body = yaml.safe_load((v1 / "data.yaml").read_text(encoding="utf-8"))
    body["names"] = [*body["names"], "Palmolive Naturals Bar Soap 85g"]
    body["nc"] = len(body["names"])
    (v1 / "data.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")

    staging = tmp_path / "staging"
    side = build_dataset.build_v1(v1, staging)
    assert side.problems == [
        build_dataset.translation_problem(
            [*build_dataset.CANONICAL_NAMES, "Palmolive Naturals Bar Soap 85g"], source="v1"
        )
    ]
    assert list(staging.rglob("*")) == []  # the check runs before the frames do

    v2 = _staged_with_real_frames(tmp_path, ["milo_0001.jpg"], {"milo_0001.jpg": "train"})
    _decide_all(tmp_path / "annotations-v2", ["milo_0001.jpg"])
    out = tmp_path / "merged-v2"
    assert build_dataset.main(["--dry-run", "--v1", str(v1), "--v2", str(v2), "--out", str(out)]) == 2
    assert "v1 declares class(es) this dataset does not have" in capsys.readouterr().out

    # And the v2 side, whose list the annotator owns: a class it can draw that this dataset cannot
    # hold, which is the *same* refusal with the other subject.
    monkeypatch.setattr(
        annotate.store, "CLASS_NAMES", (*generations.V2.classes, "Palmolive Naturals Bar Soap 85g")
    )
    problem = build_dataset.annotator_class_problem()
    assert problem and "the annotator declares class(es) this dataset does not have" in problem
    staging2 = tmp_path / "staging2"
    assert build_dataset.build_v2(v2, tmp_path / "annotations-v2", staging2).problems == [problem]
    assert list(staging2.rglob("*")) == []
    # The dry run refuses it too: it is `build_v2` without the writes, so a run that reported this
    # set fine a second before the build stopped would be worse than no dry run at all.
    assert build_dataset.main(["--dry-run", "--no-v1", "--v2", str(v2), "--out", str(out)]) == 2
    assert "the annotator declares class(es) this dataset does not have" in capsys.readouterr().out

    # The reverse shape: a class this dataset declares that the annotator can never draw. Nothing is
    # untranslatable - the refusal is about coverage, which is the direction only this side is asked.
    monkeypatch.setattr(annotate.store, "CLASS_NAMES", generations.V2.classes[:-1])
    uncovered = build_dataset.annotator_class_problem()
    assert uncovered and "can never draw" in uncovered
    assert build_dataset.build_v2(v2, tmp_path / "annotations-v2", tmp_path / "staging3").problems == [
        uncovered
    ]
    assert not build_dataset.translation_problem(
        [generations.V2.classes[-1]], source="the annotator"
    )  # the same list without `coverage` is silent, which is what keeps v1 free of this refusal


def test_a_build_for_another_order_translates_v2s_rows_into_it(tmp_path, capsys, monkeypatch):
    """`DECLARED_GENERATION` is a choice, and the by-name remap is what makes it one.

    v1's rows always were translated into `CANONICAL_NAMES` by name; v2's used to be copied
    verbatim, on the argument that the annotator already writes the canonical order - so a build
    declaring any other order filed every v2 row under a neighbouring product, with the doctor and
    every measuring tool still passing, because all of them read the declaration rather than the
    rows. Both sides go through `remap_row` now, so the same edit produces a set whose v2 rows carry
    *v1's* indices and which the doctor judges cleanly as v1. The last third is the half that must
    not change: while the two lists agree the translation is the identity, so the rows written are
    the annotator's own rows byte for byte - the copy this replaced.
    """
    frames = ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"]
    v2 = _staged_with_real_frames(tmp_path, frames, dict(zip(frames, ("train", "valid", "test"))))
    annotations = tmp_path / "annotations-v2"
    # Drawn *before* the constants move: these rows are in the annotator's own order, and translating
    # them is the build's job - so the fixture must not follow the declaration.
    _decide_all(annotations, frames)

    slug = label_classes.SLUG_TO_CLASS["milo"]
    annotator_index = generations.V2.classes.index(slug)
    v1_index = generations.V1.classes.index(slug)
    assert annotator_index != v1_index  # or this test would pass on an identity remap

    # First the shipped path, unchanged: today's constants make the translation the identity.
    same = tmp_path / "merged-default"
    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(same)]) == 0
    capsys.readouterr()
    from annotate.store import box_from_row

    drawn = (annotations / "milo_0001.txt").read_text(encoding="utf-8").strip()
    assert (same / "train" / "labels" / "milo_0001.txt").read_text(encoding="utf-8").strip() == (
        box_from_row(drawn).as_row()
    )

    out = tmp_path / "merged-v2"
    monkeypatch.setattr(build_dataset, "DECLARED_GENERATION", generations.V1)
    monkeypatch.setattr(build_dataset, "CANONICAL_NAMES", generations.V1.classes)
    monkeypatch.setattr(
        build_dataset, "INDEX_BY_NAME", {name: i for i, name in enumerate(generations.V1.classes)}
    )

    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    # Judged as the generation it declares, and clean - the labels agree with the frames' tags
    # because the build *translated* them, not because the two orders happened to coincide.
    assert "judged against v1's order" in printed
    assert "[ok] the set is internally consistent" in printed
    assert train_model.read_export_generation(out) == "v1"

    assert drawn.split()[0] == str(annotator_index)  # the row the annotator wrote...
    for name, split in zip(frames, ("train", "valid", "test")):
        row = (out / split / "labels" / f"{Path(name).stem}.txt").read_text(encoding="utf-8")
        assert row.split()[0] == str(v1_index)  # ...written out as v1's position for it

    # And the measuring tools, which read the declaration: this set is one they accept as v1's and
    # refuse as v2's, which is the honest reading of a set whose rows are in v1's order.
    assert train_model.require_labels_order(generations.V1, out) == generations.V1.classes
    with pytest.raises(SystemExit) as refused:
        train_model.require_labels_order(generations.V2, out)
    assert "another class order" in str(refused.value)


def test_allow_machine_only_lands_the_reading_and_says_what_it_produced(tmp_path, capsys):
    """`--allow-unassigned`'s sibling, for the same reason that flag exists: reading a merge - the
    mix, the contact sheet, how much is still unread in `far` - is a real thing to want from a set
    that is not meant to be trained. The flag says out loud what it landed, so the refusal a run
    gives later is not a surprise.
    """
    frames = ["milo_0001.jpg", "milo_0002.jpg", "milo_0003.jpg"]
    v2 = _staged_with_real_frames(
        tmp_path, frames, {frames[0]: "train", frames[1]: "valid", frames[2]: "test"}
    )
    _decide_all(tmp_path / "annotations-v2", frames)
    _provenance(tmp_path / "annotations-v2", frames, machine_only=("milo_0002.jpg",))
    out = tmp_path / "merged-v2"

    code = build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out), "--allow-machine-only"])

    printed = capsys.readouterr().out
    assert code == 0
    assert "landing it anyway: --allow-machine-only" in printed
    assert "a run will refuse it" in printed
    assert (out / "valid" / "images" / "milo_0002.jpg").is_file()


def test_the_build_gate_asks_the_counts_and_deliberately_not_the_stamp():
    """The asymmetry with the two readers, pinned - because it looks like an omission.

    The stamp exists so a *later* reader can tell whether the counts still describe the state on
    disk; a build's counts are its own measurement of the store it has just read, so there is
    nothing stale to detect. Asking for the stamp here would refuse a set with no `provenance.json`
    at all, which is not a dirty set: it is the Roboflow-labeling route, where the labels were drawn
    in that project and there are no machine-made decisions to read. That set's report carries
    `annotations: {}`, and the run says "cannot verify" rather than "dirty" about it - the honest
    difference, and one that belongs to that door.
    """
    assert build_dataset.pass_gate({"machine_only_by_split": {"train": 9}})[0] is True
    ok, lines = build_dataset.pass_gate({"machine_only_by_split": {"valid": 4}})
    assert ok is False
    assert "4 machine-only decision(s) in `valid`" in lines[0]
    # No provenance and nothing machine-only: clean, not unverifiable.
    assert build_dataset.pass_gate({"machine_only_by_split": {}, "annotations": {}})[0] is True


def test_an_allow_unassigned_build_is_not_doctored_and_says_so(tmp_path, capsys):
    """That flag already says the set is not meant for training, so it is not judged as one.

    Judging it would mean refusing the reading the flag exists to produce - and the flag's own
    output is the report, which is the thing an operator asked for.
    """
    v2 = _staged_with_real_frames(tmp_path, ["milo_0001.jpg"], {"milo_0001.jpg": "train"})
    _decide_all(tmp_path / "annotations-v2", ["milo_0001.jpg"])
    out = tmp_path / "merged-v2"

    assert build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out), "--allow-unassigned"]) == 0
    printed = capsys.readouterr().out
    assert "not doctored: --allow-unassigned" in printed
    assert "the doctor, on the copy" not in printed
    assert (out / "train" / "images" / "milo_0001.jpg").is_file()


def _staged_frame(name: str, state: str = "labeled"):
    """The two attributes `v2_contribution_problem` reads - a whole `Frame` would be scaffolding."""
    from types import SimpleNamespace

    return SimpleNamespace(name=name, state=state)


def test_the_contribution_guard_fires_only_when_nothing_would_be_merged():
    """One rule, laid out by state: what a build would get from the v2 side, and what it says.

    The deliberate silences matter as much as the refusals. A set with no frames is
    `v2_set_problem`'s question, and a decided-but-unplaced set without the flag is the unassigned
    guard's - whose sentence names the frames and the command that fixes them, and whose advice
    (`--allow-unassigned`) a second sentence would contradict.
    """
    here = Path("cleaned-v2")

    assert build_dataset.v2_contribution_problem(here, [], {}) is None  # not a staged set at all
    # A split cannot stand in for a decision: the frame would still be skipped.
    assert build_dataset.v2_contribution_problem(
        here, [_staged_frame("a.jpg", "unlabeled")], {"a.jpg": "train"}
    )
    assert build_dataset.v2_contribution_problem(
        here, [_staged_frame("a.jpg"), _staged_frame("b.jpg")], {"a.jpg": "train"}
    ) is None  # one placed frame is a contribution

    undecided = build_dataset.v2_contribution_problem(
        here, [_staged_frame("a.jpg", "unlabeled"), _staged_frame("b.jpg", "unlabeled")], {}
    )
    assert "not one of them has a decision yet" in undecided
    assert "under --no-v1" in undecided  # the sentence has to survive the build it can also be

    # The unassigned guard's state: silent while that guard refuses, and refused once the flag has
    # turned its refusal into "every decided frame was left out".
    assert build_dataset.v2_contribution_problem(here, [_staged_frame("a.jpg")], {}) is None
    dropped = build_dataset.v2_contribution_problem(
        here, [_staged_frame("a.jpg")], {}, allow_unassigned=True
    )
    assert "`--allow-unassigned` would leave every one of them out" in dropped
    # A mixed set under the flag is the same verdict, and the sentence still accounts for both
    # halves rather than reporting a set nobody looked at.
    mixed = build_dataset.v2_contribution_problem(
        here, [_staged_frame("a.jpg"), _staged_frame("b.jpg", "unlabeled")],
        {},
        allow_unassigned=True,
    )
    assert "1 of them undecided" in mixed


def test_a_rebuild_from_an_unlabelled_set_is_refused_as_v1_alone(tmp_path, capsys):
    """The live shape of this: a staged set whose decisions live somewhere this machine cannot see.

    The manifest is full, every frame is skipped, and the merged set would be v1's frames under
    v2's name - the one failure `v2_set_problem` deliberately stays quiet about, because the
    directory really is a staged set. The dry run reads the same state and exits 2 as well, so the
    cheap check cannot say "fine" to a build that refuses.
    """
    out = tmp_path / "merged-v2"
    _previous_build(out)
    before = _tree(out)
    v2 = _staged_with_real_frames(tmp_path, ["milo_0001.jpg", "milo_0002.jpg"])

    assert build_dataset.main(["--dry-run", "--no-v1", "--v2", str(v2), "--out", str(out)]) == 2
    printed = capsys.readouterr().out
    assert "not one of them has a decision yet" in printed
    assert "2 frame(s) with no decision yet" in printed  # the count is still the readout

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(out), "--force"])
    assert "not rebuilding from this --v2" in str(refused.value)
    assert "not one of them has a decision yet" in str(refused.value)

    assert _tree(out) == before  # and it refused before the staging copy was even made
    assert not build_dataset.staging_dir(out).exists()


def test_the_contribution_count_speaks_of_the_side_not_one_directory(tmp_path, capsys):
    """The frames a build reads are `--v2`'s set *plus* the defaults beside it.

    On this workspace that is 1,383 frames in `cleaned-v2` and 50 in `cleaned-negatives`, so a
    sentence that attributed all of them to `--v2` would send an operator looking in a folder that
    does not hold the count they were just given.
    """
    v2 = _staged_with_real_frames(tmp_path, ["milo_0001.jpg", "milo_0002.jpg"])
    _staged_negatives(tmp_path, frames=1)  # staged beside it, picked up without being named

    assert build_dataset.main(
        ["--dry-run", "--no-v1", "--v2", str(v2), "--out", str(tmp_path / "merged-v2")]
    ) == 2
    assert "the v2 side holds 3 staged frame(s)" in capsys.readouterr().out


def test_an_allow_unassigned_build_cannot_drop_every_decided_frame(tmp_path, capsys):
    """The flag's reading has a floor: a set it drops entirely is not a small merged set.

    `--allow-unassigned` exists to build the frames that *are* placed without failing on the ones
    that are not. With nothing placed, the build it produces is v1's frames alone carrying a note
    about how much is still unplaced - the artifact the guard above refuses, arrived at through the
    flag that was supposed to make the build possible.
    """
    out = tmp_path / "merged-v2"
    _previous_build(out)
    before = _tree(out)
    names = ["milo_0001.jpg", "milo_0002.jpg"]
    v2 = _staged_with_real_frames(tmp_path, names)  # decided, and no splits.json anywhere
    _decide_all(tmp_path / "annotations-v2", names)

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(
            ["--no-v1", "--v2", str(v2), "--out", str(out), "--force", "--allow-unassigned"]
        )
    assert "`--allow-unassigned` would leave every one of them out" in str(refused.value)
    assert _tree(out) == before
    assert not build_dataset.staging_dir(out).exists()


def test_the_unassigned_refusal_is_not_repeated_as_a_contribution_problem(tmp_path, capsys):
    """Without the flag, one refusal with one fix - not two sentences that disagree about it.

    The unassigned guard's sentence already says these frames have no split and names the freeze
    command, and it mentions `--allow-unassigned`. A contribution sentence under it would refuse
    the very build the guard just offered.
    """
    v2 = _staged_with_real_frames(tmp_path, ["milo_0001.jpg"])  # decided, unplaced
    _decide_all(tmp_path / "annotations-v2", ["milo_0001.jpg"])

    with pytest.raises(SystemExit) as refused:
        build_dataset.main(["--no-v1", "--v2", str(v2), "--out", str(tmp_path / "merged-v2")])
    assert "refusing to write a dataset" in str(refused.value)

    printed = capsys.readouterr().out
    assert "no split" in printed
    assert "would take nothing from v2" not in printed


def test_a_leftover_staging_directory_is_cleared_and_said_so(tmp_path, capsys):
    """A killed build leaves gigabytes of JPEGs beside the set; the next one clears it out loud."""
    staging = build_dataset.staging_dir(tmp_path / "merged-v2")
    assert staging.name == ".merged-v2.building"  # beside the set, so the swap is a rename
    _jpeg(staging / "train" / "images" / "half_built.jpg")

    build_dataset.prepare_staging(staging)
    assert "clearing a leftover build directory" in capsys.readouterr().out
    assert staging.is_dir() and not any(staging.iterdir())

    # And it refuses rather than deletes when the path holds something that is not ours.
    in_the_way = tmp_path / ".other.building"
    in_the_way.write_text("not a directory this tool made", encoding="utf-8")
    with pytest.raises(SystemExit) as refused:
        build_dataset.prepare_staging(in_the_way)
    assert "in the way" in str(refused.value)
    assert in_the_way.read_text(encoding="utf-8") == "not a directory this tool made"


def test_the_annotator_worklist_takes_the_same_dial():
    """The annotator has the same trap and the same fix (`resolve_extras` owns the rule now).

    Its `main` is exercised by hand, so this pins the wiring rather than the behavior: the default
    is the empty *added* list, and `--no-extras` exists to replace the defaults on purpose.
    """
    from annotate import run as annotate_run

    parser = annotate_run.build_parser()
    assert parser.parse_args([]).extras == []
    assert parser.parse_args([]).no_extras is False
    assert parser.parse_args(["--extras", "a", "--extras", "b"]).extras == ["a", "b"]
    assert parser.parse_args(["--no-extras"]).no_extras is True


def _progress_with_splits(splits, staged: list[str]) -> label_progress.Progress:
    """A `Progress` as `roboflow_progress` leaves it, carrying only what `capture_splits` reads."""
    return label_progress.Progress(
        summary={},
        cells={},
        per_class={},
        split_stat={},
        sessions_out=[],
        null_names=[],
        mismatches=[],
        multi_box=[],
        expected={name: {"new_name": name} for name in staged},
        split_by_name=splits,
    )


def test_capture_splits_freezes_the_project_assignment_and_refuses_an_incomplete_one(tmp_path):
    """The file is written either way; the exit code is what a `make` chain reads.

    Writing first is deliberate - the map is useful to look at, and it names exactly which staged
    frames the project has never seen - while exiting 0 on an incomplete one would walk the chain
    into a build that quietly drops them. Only frames with a real split land in it: an unplaced
    project frame must not be frozen as a fourth split name.
    """
    out = tmp_path / "cleaned-v2"
    out.mkdir()

    complete = _progress_with_splits({"a.jpg": "train", "b.jpg": "test"}, ["a.jpg", "b.jpg"])
    assert label_progress.capture_splits(out, complete) == 0
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == {
        "a.jpg": "train",
        "b.jpg": "test",
    }

    # `b.jpg` is in the project on no usable split and `d.jpg` is staged after the last upload.
    incomplete = _progress_with_splits(
        {"a.jpg": "train", "b.jpg": "", "c.jpg": "none"}, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    )
    assert label_progress.capture_splits(out, incomplete) == 2
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == {"a.jpg": "train"}


def test_capture_splits_refuses_a_source_that_cannot_say(tmp_path):
    """The local source has no assignment to freeze, and an empty file it wrote anyway would read
    as "no frame is in a split" - a claim about the dataset rather than about the source.

    The refusal names the file that answers the same question (`--split-plan`, i.e. a plan
    `plan_split.py` wrote one command earlier) rather than only naming the problem, and names the
    plans already staged beside the set when there are any.
    """
    out = tmp_path / "cleaned-v2"
    out.mkdir()

    with pytest.raises(SystemExit) as bare:
        label_progress.capture_splits(out, _progress_with_splits(None, ["a.jpg"]))
    assert "--split-plan" in str(bare.value)
    assert "plan_split.py" in str(bare.value)
    assert not (out / "splits.json").exists()

    for name in ("split_plan_a.json", "split_plan_b.json"):
        (out / name).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as named:
        label_progress.capture_splits(out, _progress_with_splits(None, ["a.jpg"]))
    assert "split_plan_a.json, split_plan_b.json" in str(named.value)


def test_read_split_plan_takes_the_values_and_refuses_a_shape_that_is_not_a_map(tmp_path):
    """A plan is hand-editable JSON, so only the *containers* are checked here.

    A value that is not a split name is `capture_splits`' business (it reports it frame by frame as
    unplaced, the same reading the project's own records get) - one bad row should cost one row,
    not the file. A file that is not an object at all has no map in it to report on, so it is
    refused with the path in the message, before the progress report takes a minute to compute.
    """
    path = tmp_path / "split_plan_b.json"
    path.write_text(json.dumps({"a.jpg": "train", "b.jpg": None, "c.jpg": 1}, indent=1), encoding="utf-8")
    assert label_progress.read_split_plan(path) == {"a.jpg": "train", "b.jpg": None, "c.jpg": 1}

    with pytest.raises(SystemExit) as missing:
        label_progress.read_split_plan(tmp_path / "split_plan_c.json")
    assert "plan_split.py" in str(missing.value)

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as broken:
        label_progress.read_split_plan(path)
    assert "not readable as JSON" in str(broken.value)

    path.write_text('["a.jpg"]', encoding="utf-8")
    with pytest.raises(SystemExit) as wrong_shape:
        label_progress.read_split_plan(path)
    assert "not a frame -> split map" in str(wrong_shape.value)


def test_capture_splits_from_a_plan_fails_closed_and_keeps_a_sibling_sets_frames(tmp_path):
    """The same rule as the project route, since it is the same file: write what it knows, exit 2.

    Two readings live in the last assertion. A frame the plan never reached is unplaced - that is
    what a build refuses on - while a frame staged in a *sibling* set is kept, because
    `plan_split.py --include` plans across sets on purpose (a held-out session, the hard
    negatives) and the file at `out` is the one read for the set and its extras together.
    """
    out = tmp_path / "cleaned-v2"
    out.mkdir()
    staged = ["a.jpg", "b.jpg"]

    # The plan knows `a.jpg` only: `b.jpg` is staged and unplaced.
    assert label_progress.capture_splits(out, _progress_with_splits(None, staged), {"a.jpg": "train"}) == 2
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == {"a.jpg": "train"}

    # A value that is not a split name is one unplaced row, not a broken file.
    assert label_progress.capture_splits(
        out, _progress_with_splits(None, staged), {"a.jpg": "train", "b.jpg": "none"}
    ) == 2
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == {"a.jpg": "train"}

    # Every staged frame placed, plus one from the held-out session the plan was drawn across.
    assert label_progress.capture_splits(
        out, _progress_with_splits(None, staged), {"a.jpg": "train", "b.jpg": "test", "s3_0001.jpg": "test"}
    ) == 0
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == {
        "a.jpg": "train",
        "b.jpg": "test",
        "s3_0001.jpg": "test",
    }


def test_both_routes_write_the_same_bytes(tmp_path):
    """One file, two authorities - and nothing about the reader may depend on which wrote it.

    `LabelStore.splits` and `build_dataset` parse a single shape, so a second one would not show up
    as an error: it would show up as the annotator and the build disagreeing about a frame.
    """
    out = tmp_path / "cleaned-v2"
    out.mkdir()
    staged = ["a.jpg", "b.jpg"]
    mapping = {"a.jpg": "train", "b.jpg": "test"}

    assert label_progress.capture_splits(out, _progress_with_splits(mapping, staged)) == 0
    from_project = (out / "splits.json").read_bytes()
    assert label_progress.capture_splits(out, _progress_with_splits(None, staged), mapping) == 0
    assert (out / "splits.json").read_bytes() == from_project


def test_the_local_flow_freezes_a_plan_and_the_build_stops_refusing(tmp_path, capsys):
    """The request, end to end: a plan file in, `splits.json` out, no API anywhere.

    This is the half that used to be impossible. `--capture-splits` refuses on a local source (the
    project is where the assignment lived), so a machine that cannot reach Roboflow could stage and
    label a whole set and then watch `build_dataset.py` exit 2 on every decided frame in it - with
    the fix named as "run it while the project is still reachable", which is exactly what that
    machine cannot do. The plan already *is* the per-frame map, so freezing it needs no network.
    """
    out = _staged_local(tmp_path, frames=3)
    annotations = tmp_path / label_progress.ANNOTATIONS_DIRNAME
    annotations.mkdir()
    for i in (1, 2, 3):
        (annotations / f"milo_{i:04d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # The shape `plan_split.py` writes: one entry per staged frame, which is what makes it a map
    # rather than a suggestion.
    plan = {"milo_0001.jpg": "train", "milo_0002.jpg": "train", "milo_0003.jpg": "valid"}
    plan_path = out / "split_plan_b.json"
    plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")

    assert label_progress.main(
        ["--source", "local", "--capture-splits", "--split-plan", str(plan_path), "--out", str(out)]
    ) == 0
    printed = capsys.readouterr().out
    assert f"assignment: {plan_path}" in printed  # what was frozen, named - not inferred
    assert "splits   ->" in printed
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == plan

    # The two readers the file exists for. The annotator joins it by name...
    store = annotate.store.LabelStore(out=out, annotations=annotations, extras=[])
    assert store.splits() == plan

    # ...and the build is the one that refused: every decided frame now has a split, which is the
    # `--dry-run` reading the run sheet asks for before a real build.
    side = build_dataset.preview_v2(out, annotations, [])
    assert side.problems == []
    # `placed` is name -> split, and the dry run is exactly the frames it would write.
    decided = Counter(side.placed.values())
    assert decided["train"] == 2 and decided["valid"] == 1


def test_a_plan_the_planner_wrote_is_a_complete_map_for_the_staged_set(tmp_path):
    """Pinned planner -> freeze, rather than a hand-written dict that only looks like a plan.

    `plan_split.stratify` is what `main` dumps as `split_plan_b.json`, so this is the same object
    the tool writes; the failure this route exists to prevent is a plan that is *nearly* complete.
    """
    out = _staged_local(tmp_path, frames=6)
    batches, entries = plan_split.load_batches(out)
    plan = plan_split.stratify(batches)
    assert set(plan) == {e["new_name"] for e in entries}  # every staged frame, placed

    plan_path = out / "split_plan_b.json"
    plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    assert label_progress.main(
        ["--source", "local", "--capture-splits", "--split-plan", str(plan_path), "--out", str(out)]
    ) == 0
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == plan


def test_the_split_plan_flag_is_not_a_silent_no_op(tmp_path):
    """`--split-plan` chooses what `--capture-splits` freezes, so on its own it would do nothing."""
    out = _staged_local(tmp_path, frames=1)
    with pytest.raises(SystemExit) as refused:
        label_progress.main(["--source", "local", "--split-plan", str(out / "split_plan_b.json"), "--out", str(out)])
    assert "--capture-splits" in str(refused.value)
    assert not (out / "splits.json").exists()


def test_plan_split_plans_the_sets_staged_beside_it_by_default(tmp_path):
    """The planner reads the same sets every other reader does (`workspace.DEFAULT_EXTRAS`).

    A plan that covers `cleaned-v2` but not the hard negatives beside it reports the same-looking
    tables and then leaves 50 frames a build reads without a split - which the freeze step names as
    an exit 2, one command later, for a reason with nothing to do with the plan. Union rather than
    either/or: a later session passes `--include` for its own directory and still means the
    negatives to be planned.
    """
    out = tmp_path / "cleaned-v2"
    out.mkdir()
    assert plan_split.include_dirs(out, []) == []  # nothing staged beside it yet

    negatives = tmp_path / "cleaned-negatives"
    negatives.mkdir()
    assert plan_split.include_dirs(out, []) == []  # a folder is not a set without its manifest
    (negatives / "manifest.json").write_text("[]", encoding="utf-8")
    assert plan_split.include_dirs(out, []) == [negatives]

    other = tmp_path / "cleaned-v2-s3"
    assert plan_split.include_dirs(out, [str(other)]) == [negatives, other]
    assert plan_split.include_dirs(out, [str(negatives)]) == [negatives]  # named once, not twice


def test_the_freeze_command_points_at_the_local_writer_and_a_named_plan(tmp_path):
    """The line plan_split prints at the end is the offline route, so it has to be runnable."""
    line = plan_split.freeze_command(tmp_path, "c")
    assert "label_progress.py" in line and "--capture-splits" in line and "--split-plan" in line
    assert str(tmp_path / "split_plan_c.json") in line


def test_capture_splits_rides_along_with_a_roboflow_run(tmp_path, monkeypatch):
    """End to end through `main`: the flag has to survive the argument wiring, and the assignment
    has to come from the project's own records rather than from the staged manifest."""
    out = _staged_local(tmp_path, frames=2)

    def rec(split: str, *, labeled: bool) -> dict:
        return {
            "name": "milo_0001.jpg",
            "split": split,
            "tags": ["milo", "mid", "s2"],
            "annotations": {"count": 1, "classes": {"Milo Chocolate Drink 22g Sachet": 1}}
            if labeled
            else [],
        }

    index = {"milo_0001.jpg": rec("test", labeled=True), "milo_0002.jpg": rec("train", labeled=False)}
    monkeypatch.setattr(label_progress, "fetch_all", lambda client, key, project: index)
    monkeypatch.setattr(label_progress, "load_key", lambda project: "fake")

    assert label_progress.main(["--source", "roboflow", "--capture-splits", "--out", str(out)]) == 0
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == {
        "milo_0001.jpg": "test",
        "milo_0002.jpg": "train",
    }

    # A staged frame the project has never seen is the case the exit code exists for.
    index.pop("milo_0002.jpg")
    assert label_progress.main(["--source", "roboflow", "--capture-splits", "--out", str(out)]) == 2
    assert json.loads((out / "splits.json").read_text(encoding="utf-8")) == {"milo_0001.jpg": "test"}




def test_a_box_inside_a_larger_box_of_the_same_product_is_dropped():
    """The v1-era partial box - a logo or picture panel boxed again inside the whole item - goes, and
    the box around the whole item stays."""
    whole = "3 0.500000 0.500000 0.600000 0.800000"
    logo = "3 0.450000 0.400000 0.200000 0.150000"
    rows, dropped, problems = build_dataset.drop_nested_rows([logo, whole])
    assert rows == [whole] and dropped == 1 and problems == []


def test_nested_boxes_of_different_products_are_two_items():
    """A small item in front of a large one is two items, so the rule never crosses products."""
    large = "0 0.500000 0.500000 0.600000 0.800000"
    small = "4 0.450000 0.400000 0.200000 0.150000"
    assert build_dataset.drop_nested_rows([large, small]) == ([large, small], 0, [])


def test_side_by_side_and_overlapping_boxes_of_one_product_are_kept():
    """Two cans of the same product next to each other, or partly overlapping, are two items."""
    left = "1 0.250000 0.500000 0.300000 0.400000"
    right = "1 0.750000 0.500000 0.300000 0.400000"
    overlapping = "1 0.400000 0.500000 0.300000 0.400000"
    assert build_dataset.drop_nested_rows([left, right, overlapping]) == (
        [left, right, overlapping],
        0,
        [],
    )


def test_a_box_sticking_out_within_the_tolerance_still_counts_as_inside():
    whole = "2 0.500000 0.500000 0.400000 0.400000"  # 0.30 .. 0.70
    edge = "2 0.405000 0.500000 0.230000 0.200000"  # left edge 0.29, 1% outside
    past = "2 0.390000 0.500000 0.240000 0.200000"  # left edge 0.27, 3% outside
    assert build_dataset.drop_nested_rows([whole, edge]) == ([whole], 1, [])
    assert build_dataset.drop_nested_rows([whole, past]) == ([whole, past], 0, [])


def test_of_two_identical_boxes_one_survives():
    row = "5 0.500000 0.500000 0.300000 0.300000"
    assert build_dataset.drop_nested_rows([row, row]) == ([row], 1, [])


def test_a_box_row_whose_numbers_do_not_parse_refuses_the_frame_instead_of_the_build(tmp_path):
    """`remap_row` passes a five-field row through without reading its numbers; this rule is the
    first thing that does, so a bad one has to come back as a problem naming the file."""
    good = "0 0.5 0.5 0.2 0.2"
    bad = "0 0.5 0.5 abc 0.2"
    rows, dropped, problems = build_dataset.drop_nested_rows([good, bad])
    assert (rows, dropped) == ([good, bad], 0)
    assert problems and "abc" in problems[0]

    v1 = _export_with_names(
        tmp_path / "export-v1", {"train": ["a.jpg"], "valid": ["b.jpg"], "test": ["c.jpg"]}
    )
    (v1 / "train" / "labels" / "a.txt").write_text(f"{good}\n{bad}\n", encoding="utf-8")
    side = build_dataset.build_v1(v1, tmp_path / "staging")
    assert any(p.startswith("train/a.jpg: row is not a readable box") for p in side.problems)
    assert "a.jpg" not in side.placed


def test_the_merge_writes_one_box_per_item_and_reports_the_count(tmp_path):
    """Through `build_v1`, so the rule is on the path the set is actually written by."""
    import yaml

    v1 = _export_with_names(
        tmp_path / "export-v1", {"train": ["a.jpg"], "valid": ["b.jpg"], "test": ["c.jpg"]}
    )
    names = yaml.safe_load((v1 / "data.yaml").read_text(encoding="utf-8"))["names"]
    (v1 / "train" / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.6 0.8\n0 0.45 0.4 0.2 0.15\n", encoding="utf-8"
    )
    staging = tmp_path / "staging"
    side = build_dataset.build_v1(v1, staging)

    assert side.problems == [] and side.nested == 1
    assert any(note.startswith("[nested] 1 box(es)") for note in side.notes)
    written = (staging / "train" / "labels" / "a.txt").read_text(encoding="utf-8").splitlines()
    assert len(written) == 1
    assert build_dataset.CANONICAL_NAMES[int(written[0].split()[0])] == names[0]
