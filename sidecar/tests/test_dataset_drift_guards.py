"""Drift guards: the class names, the distance axis and the tiers in every place that spells them.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
import pytest
import accept_v2
import annotate.store
import build_dataset
import clean_v2
import dataset_doctor
import label_classes
import label_progress
import plan_split
import train_model
import workspace

# Rebuilt from the modules that own each fact, not copied by value.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "MODEL_TRAINING.md"
CHECKLIST = REPO_ROOT / "docs" / "CAPTURE_CHECKLIST.md"


# --------------------------------------------------------------------------
# 4. drift guards: the class names live in three places that must agree
# --------------------------------------------------------------------------


def _doc_class_names() -> list[str]:
    """The §8.1 roster, read out of the shipped doc."""
    text = DOC.read_text(encoding="utf-8")
    section = text.split("### 8.1", 1)[1].split("### 8.2", 1)[0]
    names = []
    for line in section.splitlines():
        # | 1 | `The Class Name` | 310 | `slug` | 106 / 23 / 22 |
        m = re.match(r"^\|\s*\d+\s*\|\s*`([^`]+)`", line)
        if m:
            names.append(m.group(1))
    return names


def test_doc_has_a_class_table_to_read():
    assert len(_doc_class_names()) == 7, "expected the 7-row roster in MODEL_TRAINING.md section 8.1"


def test_label_classes_matches_the_documented_roster():
    """The mapping must be exactly the doc's roster - no more, no fewer."""
    assert set(label_classes.SLUG_TO_CLASS.values()) == set(_doc_class_names()), (
        "label_classes.SLUG_TO_CLASS drifted from the doc's roster"
    )


def test_mapped_class_names_are_real_names_not_placeholders():
    """A label is matched by exact string, so a placeholder in the mapping would create
    a class no annotation can ever match - and an angle-bracketed name is what a placeholder
    looks like once it has been promoted from `NEW_CLASS_SLUGS` into the mapping by mistake.
    """
    for slug, name in label_classes.SLUG_TO_CLASS.items():
        assert "<" not in name and ">" not in name, f"{slug} is still a placeholder"
        assert name.strip() == name and name, f"{slug} has surrounding whitespace"


def test_every_class_slug_is_named():
    """A slug that is neither mapped nor declared reaches the images as a raw slug.

    Palmolive was the last placeholder and naming it is what emptied this dict, so an
    entry appearing here again means a class was staged without deciding its name.
    """
    assert label_classes.NEW_CLASS_SLUGS == {}, (
        "a class slug is declared but unnamed - put its real name in SLUG_TO_CLASS, or "
        "the images keep a slug no label can use"
    )


def test_clean_v2_class_map_agrees_with_label_classes():
    """clean_v2 decides the tags; label_classes decides the names. They key off the
    same slugs, so a slug renamed in one and not the other breaks continuity silently.

    `negative` is excluded deliberately: it is the §2 hard-negative *pseudo*-class, a
    tag and batch key with nothing to annotate, so it must not be a labelable class.
    """
    real_slugs = set(clean_v2.CLASS_MAP.values()) - label_classes.PSEUDO_CLASS_SLUGS
    assert real_slugs == set(label_classes.SLUG_TO_CLASS) | set(label_classes.NEW_CLASS_SLUGS)
    for pseudo in label_classes.PSEUDO_CLASS_SLUGS:
        assert pseudo not in label_classes.SLUG_TO_CLASS
        assert pseudo not in label_classes.NEW_CLASS_SLUGS


def test_the_hard_negative_pseudo_class_is_not_treated_as_a_labelable_slug():
    """`negative` is a tag and batch key with nothing to annotate. Without the exclusion,
    staging any hard negative makes label_classes report a false "unmapped slug" problem -
    and Tier C2b stages exactly those frames.
    """
    pseudo = clean_v2.CLASS_MAP["NEGATIVES"]
    assert pseudo in label_classes.PSEUDO_CLASS_SLUGS
    assert pseudo not in label_classes.SLUG_TO_CLASS
    assert pseudo not in label_classes.NEW_CLASS_SLUGS


def test_no_class_is_exempt_from_v1_continuity_today():
    """The exemption exists for a class with no v1 name to inherit, and Palmolive was the only
    one - so with it dropped this set is empty, and every mapped class must be a name v1's
    project already declares. Asserted rather than assumed: a slug left in here would be a
    class silently skipped by the continuity check, which is the check that keeps the 1,815
    already-labeled images comparable.
    """
    assert label_classes.V2_ONLY_SLUGS <= set(label_classes.SLUG_TO_CLASS)
    assert label_classes.V2_ONLY_SLUGS == set()
    assert set(label_classes.SLUG_TO_CLASS) == {
        slug for slug in clean_v2.CLASS_MAP.values() if slug not in label_classes.PSEUDO_CLASS_SLUGS
    }


def test_class_seed_bytes_come_from_the_name_not_the_position():
    """Index-derived seeds repeat across runs, so adding one class to an already-seeded
    project re-uploads bytes the workspace has seen, gets back the old (annotated) asset
    from its SHA-256 dedup, and 409s - which is the ordinary "add a class later" case, and
    is exactly how `--create-classes` failed once the project had seven.
    """
    assert label_classes._seed_jpeg("Bear Brand Fortified Powdered Milk 33g") != label_classes._seed_jpeg(
        "Milo Chocolate Drink 22g Sachet"
    )
    # Stable within a run and across processes - no hash()/PYTHONHASHSEED dependence.
    assert label_classes._seed_jpeg("MILO") == label_classes._seed_jpeg("MILO")
    # The retry nonce must actually change the bytes, or the retry is a no-op.
    assert label_classes._seed_jpeg("MILO") != label_classes._seed_jpeg("MILO", 1)


def test_the_dropped_class_is_gone_from_every_list():
    """A dropped class has to leave no list behind, and there are six that can hold one.

    This is the drift guard for the drop itself: any one of them still naming Palmolive is a
    class that reappears as an empty cell in the coverage tables (`TIER_A_CELLS`), as a class
    the capture tree lays out folders for (`CLASS_MAP`), as a head output nothing can label
    (`SLUG_TO_CLASS`), or as a finding the app reports on every capture (`roster.V2_ROSTER`).
    """
    import audit_v2

    from app.roster import V1_ROSTER, V2_ROSTER

    assert "palmolive" not in set(clean_v2.CLASS_MAP.values())
    assert "palmolive" not in label_classes.SLUG_TO_CLASS
    assert "palmolive" not in label_classes.V2_ONLY_SLUGS
    assert not [c for c in clean_v2.TIER_A_CELLS if "PALMOLIVE" in c[0].upper()]
    assert not [c for c in clean_v2.TIER_D_CELLS if "PALMOLIVE" in c[0].upper()]
    assert "palmolive" not in audit_v2.BOTTLE_SHAPED
    assert not [n for n in V1_ROSTER + V2_ROSTER if "palmolive" in n.lower()]

    # The v1 project's names are what the seven map onto, so nothing was renamed on the way out.
    assert set(label_classes.SLUG_TO_CLASS.values()) == set(clean_v2.V1_CLASSES)


def test_clean_v2_class_map_covers_every_source_folder_with_spaces():
    """The source folders are the real input contract - a rename there must not
    silently drop a class (ingest skips unmapped folders with only a report line).
    """
    for folder in ("BEARBRAND", "LUCKY ME", "MILO", "SAFEGUARD", "SARDINES", "Silver Swan", "TUNA"):
        assert folder in clean_v2.CLASS_MAP, f"source folder {folder!r} is unmapped"


def test_hard_negative_frames_get_no_invented_distance():
    """A null-annotated frame carries no box, so close/mid/far would be a value nobody
    observed - and it would surface as a real cell in the per-distance coverage report.
    """
    assert clean_v2.batch_name("negative", "") == "negative"
    assert clean_v2.tags_for("negative", "") == ["negative"]
    # The roster keeps distance-first order, because only tags[0] is sent at upload.
    assert clean_v2.tags_for("milo", "mid") == ["mid", "milo"]
    assert clean_v2.batch_name("milo", "mid") == "milo_mid"
    assert clean_v2.NEGATIVE_CLS == clean_v2.CLASS_MAP["NEGATIVES"]


def test_the_capture_session_is_a_tag_and_never_tag_zero():
    """The session has to be a tag because Roboflow's dataset search has no `batch:`
    filter - a batch can group images but cannot select them - and it has to come last
    because `upload_one` sends only `tags[0]`, which must stay the distance.
    """
    assert clean_v2.tags_for("milo", "mid", "s2") == ["mid", "milo", "s2"]
    assert clean_v2.tags_for("milo", "mid", "s2")[0] == "mid"
    # A frame with no distance still carries its session.
    assert clean_v2.tags_for("negative", "", "neg1") == ["negative", "neg1"]
    # Absent session: unchanged, so every pre-session call site keeps working.
    assert clean_v2.tags_for("milo", "mid") == ["mid", "milo"]


def test_tier_a_is_the_documented_six_cells():
    """Six, not eight: Palmolive's `close` and `mid` rows went with the class, and its `close`
    cell is the reason the class went - a target that can never be filled is worse than a
    product the app does not know.
    """
    cells = clean_v2.TIER_A_CELLS
    assert len(cells) == 6
    assert sum(cells.values()) == 186
    for product, distance in cells:
        # Every name must be one the ingest maps, or scaffold builds a tree it then skips.
        assert product in clean_v2.CLASS_MAP, f"{product!r} is not a CLASS_MAP folder"
        assert clean_v2.DISTANCE_MAP[distance.upper()] in ("close", "mid", "far")


# The Tier A table as an operator reads it: `| \`century-tuna\` | mid | 0 | **40** | ... |`.
# Anchored on the backticked slug so it cannot match the surrounding prose.
_TIER_A_ROW = re.compile(
    r"^\|\s*`(?P<slug>[a-z0-9-]+)`\s*\|\s*(?P<distance>close|mid|far)\s*\|"
    r"\s*\d+\s*\|\s*\*\*(?P<capture>\d+)\*\*\s*\|"
)
_TIER_A_TOTAL = re.compile(r"\*\*Tier A total: (\d+) images\.\*\*")


def _checklist_tier_a_section() -> str:
    text = CHECKLIST.read_text(encoding="utf-8")
    start = text.index("## Tier A")
    return text[start : text.index("## Tier B", start)]


def test_tier_a_table_in_the_checklist_matches_the_scaffold_cells():
    """Tier A's numbers live in two places that must agree: the table an operator shoots from,
    and `TIER_A_CELLS`, which `scaffold` turns into folders. A cell added to one and not the
    other is a session that gets skipped or has nowhere to land, and both fail silently -
    the same class of bug as the folder-name mapping, one level up.

    The table's `Have` column is deliberately not compared: it is a snapshot of the staged set
    and is *supposed* to move as images land, while the capture targets are the contract.
    """
    documented: dict[tuple[str, str], int] = {}
    for line in _checklist_tier_a_section().splitlines():
        match = _TIER_A_ROW.match(line)
        if match:
            documented[(match["slug"], match["distance"])] = int(match["capture"])

    expected = {
        (clean_v2.CLASS_MAP[product], distance): want
        for (product, distance), want in clean_v2.TIER_A_CELLS.items()
    }
    assert documented == expected, (
        "CAPTURE_CHECKLIST.md's Tier A table and clean_v2.TIER_A_CELLS disagree"
    )

    total = _TIER_A_TOTAL.search(_checklist_tier_a_section())
    assert total, "the Tier A section lost its stated total"
    assert int(total.group(1)) == sum(expected.values())


def test_the_progress_snapshot_takes_its_capture_plan_from_the_same_table():
    """The Admin Panel's capture gap and the folder skeleton must be the same plan.

    `label_progress.py` writes the Tier A targets into the snapshot the sidecar reads, so
    if it ever grew its own copy of them the panel would confidently point at cells that
    `scaffold` does not create - and the two would only disagree after a capture session.
    """
    assert label_progress.TIER_A_CELLS is clean_v2.TIER_A_CELLS
    assert label_progress.CLASS_MAP is clean_v2.CLASS_MAP


def test_the_scaffold_subcommand_is_wired_up(tmp_path):
    """`scaffold_tree` is only useful if the CLI reaches it. The checklist tells the operator
    to run this, so a renamed or unregistered subcommand is an instruction that errors out.
    """
    assert clean_v2.main(["scaffold", "--root", str(tmp_path), "--dry-run"]) == 0
    assert clean_v2.main(["scaffold", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "README.md").is_file()
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 4  # 6 cells, 4 products


def test_scaffold_builds_a_tree_the_ingest_accepts(tmp_path):
    """The round trip that matters. `ingest` skips an unrecognised product folder with only
    a report line, so a folder typed from the checklist ("Lucky Me") is a whole capture
    session thrown away - and you find out after shooting it.
    """
    made = clean_v2.scaffold_tree(clean_v2.TIER_A, tmp_path, current_total=1383)
    assert len(made) == 6

    report: list[str] = []
    assert clean_v2.ingest(tmp_path, report) == []  # empty cells: nothing to ingest yet
    assert not [line for line in report if "unmapped" in line], report
    # Each empty cell is named, which is how a partly-finished session gets noticed.
    assert len([line for line in report if "[gap]" in line]) == 6

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "186 images to shoot" in readme
    assert "from 1383 to 1569" in readme
    # The README must name the folders as they exist on disk, spaces and all.
    assert "`Silver Swan/` | `MID`" in readme and "`LUCKY ME/` | `MID`" in readme


def test_scaffold_omits_the_projection_when_the_staged_set_is_unknown(tmp_path):
    """A projected total baked into the file would quietly become wrong; better to say
    nothing than to state a number nobody measured."""
    clean_v2.scaffold_tree(clean_v2.TIER_A, tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "would take the staged set" not in readme


def test_scaffold_refuses_a_name_the_ingest_would_skip(tmp_path):
    bad_product = dataclasses.replace(clean_v2.TIER_A, cells={("Lucky Me", "mid"): 5})
    bad_distance = dataclasses.replace(clean_v2.TIER_A, cells={("MILO", "middleish"): 5})
    with pytest.raises(SystemExit):
        clean_v2.scaffold_tree(bad_product, tmp_path)
    with pytest.raises(SystemExit):
        clean_v2.scaffold_tree(bad_distance, tmp_path)


def test_tier_d_is_the_whole_grid_and_its_total_is_the_documented_one():
    """Derived, not hand-listed: Tier D is a re-shoot of cells earlier sessions cover, so
    every product x distance is in it by construction and there is no per-cell judgement
    that could disagree with the checklist's arithmetic."""
    products = [p for p in clean_v2.CLASS_MAP if p != "NEGATIVES"]
    assert len(products) == 7
    assert set(clean_v2.TIER_D_CELLS) == {
        (p, d) for p in products for d in clean_v2.DISTANCE_ORDER
    }
    assert len(clean_v2.TIER_D_CELLS) == 21
    assert set(clean_v2.TIER_D_CELLS.values()) == {clean_v2.TIER_D_PER_CELL}
    assert sum(clean_v2.TIER_D_CELLS.values()) == 315

    # And the checklist quotes the same total, so a retuned per-cell count cannot leave the
    # document telling a reader to shoot a different number than the scaffold lays out.
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "315" in text
    assert re.search(r"15 per cell", text)


def test_tier_d_scaffold_names_the_precondition_and_the_planning_order(tmp_path):
    """The two things that make this session different from every other: it must re-shoot
    only cells train already covers, and its split has to be planned between `clean` and
    `upload` - because a split cannot be changed by re-uploading an image afterwards."""
    made = clean_v2.scaffold_tree(clean_v2.TIER_D, tmp_path)
    assert len(made) == 21
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "315 images to shoot" in readme
    assert "Re-shoot only cells an earlier session already covers" in readme
    # `#` lines are comments inside the same bash block, so read the runnable ones.
    commands = [c for c in clean_v2.tier_commands(clean_v2.TIER_D, tmp_path) if not c.startswith("#")]
    # clean -> plan_split -> upload -> retag, in that order.
    assert len(commands) == 4
    assert " clean " in commands[0]
    assert "--holdout-session s3" in commands[1] and "--include" in commands[1]
    assert "--split-plan" in commands[2]
    assert " retag " in commands[3]
    assert "CANNOT hold out session s3" in readme  # the failure to look for

    # Tier A's session has no holdout step, for the same reason it has no acceptance role.
    a_commands = clean_v2.tier_commands(clean_v2.TIER_A, tmp_path)
    assert len(a_commands) == 3
    assert not any("--holdout-session" in c for c in a_commands)
    assert all("s2" in c for c in a_commands)


def test_the_readme_commands_are_pasteable_into_a_shell(tmp_path):
    """They are pasted, not read: the README's block is the whole runbook for a session that
    cost a day to shoot. On Windows `Path.__str__` renders `sidecar\\data\\...`, and bash eats
    that backslash as an escape - so `clean --src` resolves to a directory that does not
    exist, ingests nothing, and reports success over an empty staged set.
    """
    for plan in (clean_v2.TIER_A, clean_v2.TIER_D):
        for command in clean_v2.tier_commands(plan, tmp_path):
            assert "\\" not in command, command


def test_ingest_would_skip_the_camera_frames_so_negatives_needs_its_own_path(tmp_path):
    """The camera writes cam0_*.jpg flat into images/. Guards the reason --negatives
    exists at all: the product-tree walk has nothing to map that folder to.
    """
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "cam0_000000.jpg").write_bytes(b"\xff\xd8\xff")
    report: list[str] = []
    assert clean_v2.ingest(tmp_path, report) == []
    assert any("unmapped product folder" in line for line in report)


def test_ingest_negatives_walks_a_flat_folder_with_no_distance(tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(3):
        (tmp_path / "images" / f"cam0_{i:06d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([i]))
    (tmp_path / "images" / "notes.txt").write_text("not an image")
    (tmp_path / "images" / "empty.jpg").write_bytes(b"")

    report: list[str] = []
    found = clean_v2.ingest_negatives(tmp_path / "images", report)

    assert [f.name for f in found] == [f"cam0_{i:06d}.jpg" for i in range(3)]
    assert {f.cls for f in found} == {clean_v2.NEGATIVE_CLS}
    assert {f.distance for f in found} == {""}
    # Unusable files are reported rather than staged or silently dropped.
    assert any("zero-byte" in line for line in report)
    assert any("negatives" in line for line in report)


def test_distance_map_covers_every_spelling_the_sources_use():
    for spelling in ("CLOSE", "CLOSE-UP", "MID", "MIDDLE", "MID-SHOT", "FAR", "FAR-SHOT"):
        assert clean_v2.DISTANCE_MAP.get(spelling) in ("close", "mid", "far")
    assert clean_v2.DISTANCE_MAP["CLOSE-UP"] == "close"
    assert clean_v2.DISTANCE_MAP["MIDDLE"] == "mid"
    assert clean_v2.DISTANCE_MAP["FAR-SHOT"] == "far"


def test_sanity_expected_resize_matches_the_sidecar_default():
    """v2 is trained at whatever preprocessing the version used, but inference is pinned
    to settings.imgsz - so these two numbers must not drift apart.
    """
    settings = (REPO_ROOT / "sidecar" / "app" / "settings.py").read_text(encoding="utf-8")
    imgsz = re.search(r"^\s*imgsz:\s*int\s*=\s*(\d+)", settings, re.MULTILINE)
    assert imgsz, "could not find imgsz in settings.py"
    want_w, want_h, want_fmt = clean_v2.EXPECTED_RESIZE
    assert (want_w, want_h) == (int(imgsz.group(1)), int(imgsz.group(1)))
    assert want_fmt == "Stretch to", "the custom .onnx path resolves resize_mode 'auto' to stretch"


def test_v1_class_names_in_the_doc_are_the_ones_used_for_continuity():
    """The whole point of the roster: these are v1's names, not tidy substitutes."""
    doc = set(_doc_class_names())
    for name in label_classes.SLUG_TO_CLASS.values():
        assert name in doc
    # v1's names deliberately break Roboflow's "alphanumeric plus -" advice, because
    # 1,815 images are already labeled with them. Guard that we did not "fix" them.
    assert any(" " in n for n in label_classes.SLUG_TO_CLASS.values())


def test_the_split_names_are_defined_once_and_every_tool_reads_that_one():
    """Everything that names a split, naming the same triple: `plan_split` assigns them,
    `clean_v2` validates a plan against them, `build_dataset` creates the directories, `train_model`
    trains on them and `--val` reads them, `label_progress` prints their columns, `dataset_doctor`
    reads the mix line in that order, and `annotate/store` counts a session's spread over them. The
    identity assertions are the point: one object, every reader. The fitted pair is the same fact
    one level down - the two splits a model learns from - and is pinned here too. Sibling of the
    distance-axis guard (test_audit_recall.py).
    """
    assert clean_v2.SPLIT_NAMES is label_classes.SPLIT_NAMES
    assert plan_split.SPLIT_NAMES is label_classes.SPLIT_NAMES
    assert build_dataset.SPLIT_NAMES is label_classes.SPLIT_NAMES
    assert train_model.SPLIT_NAMES is label_classes.SPLIT_NAMES
    assert label_progress.SPLIT_NAMES is label_classes.SPLIT_NAMES
    assert dataset_doctor.SPLIT_NAMES is label_classes.SPLIT_NAMES
    assert annotate.store.SPLIT_NAMES is label_classes.SPLIT_NAMES
    assert build_dataset.FIT_SPLITS is label_classes.FIT_SPLITS
    assert dataset_doctor.FIT_SPLITS is label_classes.FIT_SPLITS
    assert set(label_classes.FIT_SPLITS) == set(label_classes.SPLIT_NAMES) - {"test"}
    # Every module in the tools tree, not only the readers above: a *new* tool that spells either
    # tuple out is the drift this guard exists for. One pattern covers both - a spelled-out
    # `SPLIT_NAMES` starts with the fitted pair - while `accept_v2.GATE_SPLITS`, the two names a
    # gate number is quoted from, is a different fact and is not matched.
    spelling = re.compile(r'''["']train["']\s*,\s*["']valid["']''')
    offenders = [
        path.name
        for path in sorted((Path(label_classes.__file__).parent).glob("*.py"))
        if path.name != "label_classes.py" and spelling.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    # The accepted-yaml-key list is a different fact, not a copy of this one: the validation split's
    # key is `val`, and a set whose folder is `valid` is one ultralytics never reads. Pinned here
    # because the two are one "unification" apart, and that edit trains a set nobody validates.
    assert train_model.VALIDATION_SPLITS == ("train", "val", "test")
    # And the gate's two names have to be names of this dataset.
    assert set(accept_v2.GATE_SPLITS) <= set(label_classes.SPLIT_NAMES)


