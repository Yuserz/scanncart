"""Tests for `push_labels.py --check` - the gate over the reverse bridge.

No network, no key, no staged data beyond a temp tree a test writes: the classifier is pure
(`classify_received`), and the mode is driven end-to-end through `main` with the search index
faked the way this suite fakes APIs everywhere else. What is worth asserting is not that a
request is formatted but the things that are silent when they are wrong:

* a **byte-twin** covers a frame the project holds under the *sibling's* name
  (negative_neg3_0014 landed as its twin 0015's record) - received, not missing, but only while
  the twin's record holds the same kind of decision as the local one;
* a twin whose record state **differs** from the local decision is missing, not received - the
  project has the bytes, but no record represents this decision, and pushing would overwrite the
  twin's with this frame's;
* a frame neither named nor twinned **exits 1 by name**, and the ok frames stay in the report,
  so the verdict reads as totals over a set rather than a pass with a footnote;
* an **empty local tree is a refusal** (exit 2), never a green pass - a bare checkout is exactly
  the silence this check exists to kill.

    sidecar/.venv/Scripts/python.exe -m pytest tests/test_push_labels_check.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import push_labels
from push_labels import classify_received

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


# --------------------------------------------------------------------------
# The classifier, pure
# --------------------------------------------------------------------------

def test_received_under_its_own_name() -> None:
    cloud = {"milo_0001": ("milo_0001.jpg", "labeled")}
    report = classify_received({"milo_0001": SHA_A}, {"milo_0001": "labeled"}, cloud)
    assert report.by_name == ["milo_0001"]
    assert (report.by_twin, report.missing, report.undecided) == ([], [], [])
    assert report.received == 1


def test_a_byte_twin_covers_a_frame_the_project_holds_under_the_sibling_name() -> None:
    """The 0014/0015 shape: two local stems, one sha256, the cloud record under one of them,
    the same null on both sides - the other is received as that twin's, not missing."""
    local_hash = {"negative_0014": SHA_A, "negative_0015": SHA_A}
    local_state = {"negative_0014": "null", "negative_0015": "null"}
    cloud = {"negative_0015": ("negative_0015.jpg", "null")}
    report = classify_received(local_hash, local_state, cloud)
    assert report.by_name == ["negative_0015"]
    assert report.by_twin == [("negative_0014", "negative_0015.jpg")]
    assert report.received == 2
    assert not report.missing


def test_a_labeled_twin_pair_covers_through_the_labeled_sibling() -> None:
    """Twins share bytes, so their decisions are expected to agree - a labeled twin missing from
    the project behind a labeled sibling is the same coverage the null case gets."""
    local_hash = {"milo_0001": SHA_A, "milo_0002": SHA_A}
    local_state = {"milo_0001": "labeled", "milo_0002": "labeled"}
    cloud = {"milo_0001": ("milo_0001.jpg", "labeled")}
    report = classify_received(local_hash, local_state, cloud)
    assert report.by_twin == [("milo_0002", "milo_0001.jpg")]
    assert not report.missing


def test_a_twin_with_a_different_decision_is_missing_the_decision_not_received() -> None:
    """Same bytes under the sibling's name, but the sibling's record holds the *other* kind of
    decision - unrepresentable: annotating this frame would overwrite the twin's record."""
    local_hash = {"milo_0001": SHA_A, "milo_0002": SHA_A}
    local_state = {"milo_0001": "labeled", "milo_0002": "null"}
    cloud = {"milo_0001": ("milo_0001.jpg", "labeled")}
    report = classify_received(local_hash, local_state, cloud)
    assert report.by_name == ["milo_0001"]
    stem, why = report.missing[0]
    assert stem == "milo_0002"
    assert "no record represents this decision" in why and "milo_0001.jpg" in why


def test_a_frame_with_no_hash_cannot_prove_a_twin() -> None:
    report = classify_received({"milo_0009": None}, {"milo_0009": "labeled"}, {})
    stem, why = report.missing[0]
    assert stem == "milo_0009"
    assert "no manifest record holds its hash" in why


def test_a_frame_neither_named_nor_twinned_is_missing() -> None:
    report = classify_received({"milo_0009": SHA_B}, {"milo_0009": "labeled"}, {})
    assert report.missing == [("milo_0009", "its bytes are in no record the project holds")]
    assert report.received == 0


def test_received_by_name_but_not_yet_decided_is_reported_not_failed() -> None:
    """The photo is under its own name and the project has not annotated it: the recovery path's
    window, and the run sheet's cloud-side readers already surface it as outstanding work."""
    report = classify_received(
        {"milo_0001": SHA_A}, {"milo_0001": "labeled"}, {"milo_0001": ("milo_0001.jpg", "unlabeled")}
    )
    assert report.undecided == ["milo_0001"]
    assert not report.missing


def test_a_gap_is_listed_beside_the_ok_frames_not_instead_of_them() -> None:
    """One gap does not hide the receipts: both halves stay in the report, so the exit-1 print
    names exactly the frames to fix and nothing else."""
    local_hash = {"milo_0001": SHA_A, "milo_0009": SHA_B}
    local_state = {"milo_0001": "labeled", "milo_0009": "labeled"}
    cloud = {"milo_0001": ("milo_0001.jpg", "labeled")}
    report = classify_received(local_hash, local_state, cloud)
    assert report.by_name == ["milo_0001"]
    assert report.missing == [("milo_0009", "its bytes are in no record the project holds")]


# --------------------------------------------------------------------------
# The mode, driven through main with the search index faked
# --------------------------------------------------------------------------

def _workspace(tmp: Path, frames: dict[str, str]) -> Path:
    """A minimal annotator tree plus the staged manifest whose sha256 the check hashes.

    `frames` is name -> sha256, where "" means a *null* label file (annotated empty) whose
    manifest hash is SHA_A by construction. Labeled frames carry the sha they are given.
    """
    ann = tmp / "annotations-v2"
    ann.mkdir()
    (ann / "classes.json").write_text(json.dumps({"names": ["x"]}), encoding="utf-8")
    manifest = []
    for name, sha in frames.items():
        (ann / f"{name}.txt").write_text("" if sha == "" else "0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        manifest.append({"new_name": f"{name}.jpg", "sha256": SHA_A if sha == "" else sha})
    staged = tmp / "cleaned-v2"
    staged.mkdir()
    (staged / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ann


def _cloud(stems: dict[str, str]) -> dict[str, dict]:
    """The search endpoint's records at the shape `state_of` reads: the *type* of `annotations`
    is what separates the three states - a dict (even empty) counts as null, a list is not yet."""
    out: dict[str, dict] = {}
    for name, state in stems.items():
        out[name] = {
            "id": "img_" + name,
            "name": name,
            "tags": [],
            "split": "train",
            "annotations": {"count": 1} if state == "labeled" else
                           ({} if state == "null" else []),
        }
    return out


def _fake_index(monkeypatch: pytest.MonkeyPatch, cloud: dict[str, dict]) -> None:
    """`main` imports `fetch_all` from `label_progress` at call time, so patching there rebinds
    where it lands - and the key env keeps `load_key` off the machine's real .env."""
    import label_progress

    monkeypatch.setattr(label_progress, "fetch_all", lambda *a, **k: dict(cloud))
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")


def _staged_roots(monkeypatch: pytest.MonkeyPatch, tmp: Path) -> None:
    """One staged root pointed at the temp tree; `find_staged` never runs (no pushes here)."""
    monkeypatch.setattr(push_labels, "STAGED_ROOTS", (tmp / "cleaned-v2",))


def test_green_when_every_local_frame_is_held_by_name(tmp_path: Path, monkeypatch, capsys) -> None:
    ann = _workspace(tmp_path, {"pushed_0001": SHA_A})
    _staged_roots(monkeypatch, tmp_path)
    _fake_index(monkeypatch, _cloud({"pushed_0001.jpg": "labeled"}))
    assert push_labels.main(["--check", "--annotations", str(ann)]) == 0
    out = capsys.readouterr().out
    assert "1 under their own name" in out and "all received" in out
    assert "FAIL" not in out


def test_exit_1_with_each_frame_named_when_the_project_never_received_it(tmp_path: Path,
                                                                        monkeypatch, capsys) -> None:
    ann = _workspace(tmp_path, {"pushed_0001": SHA_B, "lost_0002": SHA_C})
    _staged_roots(monkeypatch, tmp_path)
    _fake_index(monkeypatch, _cloud({"pushed_0001.jpg": "labeled"}))
    assert push_labels.main(["--check", "--annotations", str(ann)]) == 1
    out = capsys.readouterr().out
    fail_lines = [line for line in out.splitlines() if "FAIL" in line]
    assert len(fail_lines) == 1 and fail_lines[0].startswith("FAIL lost_0002")
    assert "never received under any name" in out


def test_the_live_projects_twin_shape_reads_green(tmp_path: Path, monkeypatch, capsys) -> None:
    """null_0002 is missing from the cloud by name, but its bytes are null_0003's record - the
    check prints it as a twin and exits 0, which is exactly how the real tree reads."""

    ann = _workspace(tmp_path, {"null_0003": "", "null_0002": ""})
    _staged_roots(monkeypatch, tmp_path)
    _fake_index(monkeypatch, _cloud({"null_0003.jpg": "null"}))
    assert push_labels.main(["--check", "--annotations", str(ann)]) == 0
    out = capsys.readouterr().out
    assert "twin null_0002: bytes identical to null_0003.jpg, which the project holds with the same decision" in out
    assert "all received" in out


def test_a_local_null_over_a_labeled_sibling_exits_1(tmp_path: Path, monkeypatch, capsys) -> None:
    """A labeled local twin over a *null*-state cloud sibling: the bytes are received but no
    record holds this decision, and pushing would overwrite the twin's interpretation."""
    ann = _workspace(tmp_path, {"labeled_0001": SHA_A, "null_0002": ""})
    # Give the twin pair one hash: null_0002's manifest sha follows SHA_A by the fixture's
    # construction, so the sibling record `labeled_0001.jpg`'s state is what decides here.
    _staged_roots(monkeypatch, tmp_path)
    _fake_index(monkeypatch, _cloud({"labeled_0001.jpg": "labeled"}))
    assert push_labels.main(["--check", "--annotations", str(ann)]) == 1
    out = capsys.readouterr().out
    fail_lines = [line for line in out.splitlines() if "FAIL" in line]
    assert len(fail_lines) == 1 and fail_lines[0].startswith("FAIL null_0002")
    assert "no record represents this decision" in out


def test_a_stem_with_no_manifest_hash_exits_1(tmp_path: Path, monkeypatch, capsys) -> None:
    """No manifest record: nothing to hash a twin with, and the push tool cannot stage the frame
    either - the same silence the classify-only rule above covers, through the mode."""
    ann = _workspace(tmp_path, {"pushed_0001": SHA_B, "lost_0002": SHA_B})
    _staged_roots(monkeypatch, tmp_path)
    (tmp_path / "cleaned-v2" / "manifest.json").write_text(
        json.dumps([{"new_name": "pushed_0001.jpg", "sha256": SHA_B}]),
        encoding="utf-8",
    )
    _fake_index(monkeypatch, _cloud({"pushed_0001.jpg": "labeled"}))
    assert push_labels.main(["--check", "--annotations", str(ann)]) == 1
    out = capsys.readouterr().out
    assert "FAIL lost_0002" in out and "no manifest record holds its hash" in out


def test_an_empty_or_absent_tree_refuses_to_pass(tmp_path: Path, monkeypatch, capsys) -> None:
    ann = tmp_path / "not-there"
    _staged_roots(monkeypatch, tmp_path)
    _fake_index(monkeypatch, {})
    assert push_labels.main(["--check", "--annotations", str(ann)]) == 2
    out = capsys.readouterr().out
    assert "refuses to pass on nothing" in out
    assert "FAIL no local decisions" in out


def test_check_refuses_the_push_only_flags(tmp_path: Path, monkeypatch, capsys) -> None:
    """`--stems` and `--any-outstanding` are push-mode selections; the gate is the whole tree.
    Argparse answers on stderr and exits 2, so the refusal's own sentence is what is asserted."""
    _fake_index(monkeypatch, {})
    _staged_roots(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as system:
        push_labels.main(["--check", "--stems", "x_0001"])
    assert system.value.code == 2
    assert "--stems" in capsys.readouterr().err
    with pytest.raises(SystemExit) as limited:
        push_labels.main(["--check", "--any-outstanding"])
    assert limited.value.code == 2
    assert "--any-outstanding is a push-mode selection" in capsys.readouterr().err
