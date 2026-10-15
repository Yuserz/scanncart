"""Tests for `clean_v2.py`: dropping a class, and the coverage preflight a held-out tier answers.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
import pytest
import clean_v2
import plan_split

from tests.dataset_tool_helpers import _batch, _entries, _frame, _two_session_batches


# --------------------------------------------------------------------------
# Dropping a class: the commands that remove the rest of its footprint
# --------------------------------------------------------------------------


def _staged_with_two_classes(tmp_path: Path) -> Path:
    """A staged set shaped like the real one: two class folders, a manifest pair, one class that
    is being dropped and one that must come through untouched.
    """
    out = tmp_path / "cleaned-v2"
    for slug, names in {
        "palmolive": ["palmolive_0001.jpg", "palmolive_0002.jpg"],
        "milo": ["milo_0001.jpg"],
    }.items():
        (out / slug).mkdir(parents=True)
        for name in names:
            (out / slug / name).write_bytes(b"jpeg")
    entries = [
        {"new_name": "palmolive_0001.jpg", "class": "palmolive", "distance": "mid"},
        {"new_name": "palmolive_0002.jpg", "class": "palmolive", "distance": "far"},
        {"new_name": "milo_0001.jpg", "class": "milo", "distance": "close"},
    ]
    (out / "manifest.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["new_name", "class", "distance"])
        writer.writeheader()
        writer.writerows(entries)
    return out


def test_drop_parks_the_staged_files_and_rewrites_both_manifests(tmp_path, capsys):
    """The one command that removes a class's footprint, run end to end with no network.

    All four halves matter and each fails quietly on its own: the JPEGs have to leave the live
    set (or the next `clean` and the split planner still count them), *both* manifests have to
    shrink (the CSV is what a person reads, the JSON is what the tools read), the other class has
    to be untouched, and the JPEGs have to be **parked rather than deleted** - the capture is real
    work, so re-shooting that product later should be an upload.
    """
    out = _staged_with_two_classes(tmp_path)

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0
    print_out = capsys.readouterr().out

    assert "2 staged image(s)" in print_out
    assert "3 -> 1 entries (2 removed)" in print_out
    # Parked, not deleted, and out of the live folders.
    assert sorted(p.name for p in (out / "dropped" / "palmolive").iterdir()) == [
        "palmolive_0001.jpg",
        "palmolive_0002.jpg",
    ]
    assert not (out / "palmolive").exists()
    # The untouched class is exactly where it was.
    assert (out / "milo" / "milo_0001.jpg").exists()

    kept = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [e["new_name"] for e in kept] == ["milo_0001.jpg"]
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["new_name"] for r in rows] == ["milo_0001.jpg"]
    # And the CSV's own header survives, so a column added to `write_manifest` cannot be dropped
    # by this command.
    assert list(rows[0]) == ["new_name", "class", "distance"]


def test_drop_dry_run_changes_nothing(tmp_path, capsys):
    """A dry run that had already moved the files would be the worst kind of flag."""
    out = _staged_with_two_classes(tmp_path)

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out), "--dry-run"]) == 0

    assert "nothing changed" in capsys.readouterr().out
    assert (out / "palmolive" / "palmolive_0001.jpg").exists()
    assert len(json.loads((out / "manifest.json").read_text(encoding="utf-8"))) == 3


def test_drop_says_so_when_the_class_is_not_staged(tmp_path, capsys):
    """Re-running the command has to be safe: idempotence is what makes it usable in a runbook."""
    out = _staged_with_two_classes(tmp_path)
    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0
    capsys.readouterr()

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0
    assert "nothing to drop" in capsys.readouterr().out


def test_a_leftover_file_in_the_dropped_folder_is_not_swept_away(tmp_path):
    """Only the names the manifest lists are moved. A file an operator put there by hand is theirs,
    and the empty-bucket cleanup must not turn into a delete of something it was never told about.
    """
    out = _staged_with_two_classes(tmp_path)
    (out / "palmolive" / "hand-added.jpg").write_bytes(b"jpeg")

    assert clean_v2.main(["drop", "--class", "palmolive", "--out", str(out)]) == 0

    assert (out / "palmolive" / "hand-added.jpg").exists()
    assert not (out / "palmolive" / "palmolive_0001.jpg").exists()


def test_only_a_drawn_box_counts_as_labeled_when_dropping_from_the_project():
    """The guard that makes `--delete-on-server` safe to run, judged the way the rest of the tools
    judge it: by the *type* of `annotations`, never by emptiness. A null-annotated hard negative is
    `{"count": 0}` - a decision nobody needs protecting from - while a positive count is somebody's
    drawn boxes, and a manifest cannot put those back.
    """
    index = {
        "drawn.jpg": {"id": "a", "annotations": {"count": 3, "classes": {"x": 3}}},
        "null.jpg": {"id": "b", "annotations": {"count": 0}},
        "unlabeled.jpg": {"id": "c", "annotations": []},
    }

    assert clean_v2.labeled_on_server(index, ["drawn.jpg", "null.jpg", "unlabeled.jpg"]) == [
        "drawn.jpg"
    ]
    # An image the project does not have is not "labeled", and not an error either - it is the
    # ordinary case for a frame that was staged and never uploaded.
    assert clean_v2.labeled_on_server(index, ["never-uploaded.jpg"]) == []


def test_with_nothing_staged_every_held_out_cell_is_a_gap():
    """True, and the reason the printer has a separate "cannot tell" branch: one alarming
    line per cell on a project that has simply not been staged yet is how a real warning
    gets trained away."""
    gaps = clean_v2.holdout_gaps(clean_v2.TIER_D, {})

    assert len(gaps) == len(clean_v2.TIER_D.cells) == 21
    assert ("century-tuna", "mid") in gaps and ("milo", "close") in gaps


def test_the_two_cells_tier_a_has_not_shot_read_as_gaps_against_the_rest_of_the_grid():
    """Today's actual state, which is the print's whole purpose: the staged set covers 19 of
    the 21 cells, so a Tier D hold-out would make exactly these two unlearnable - and they
    are two of the cells Tier A is already shooting. The next session has to land before
    `s3` can be held out.

    Two, not three: Palmolive's `close` cell was the third and left with the class, which is
    also why the grid is 21 cells rather than 24.
    """
    coverage = {
        (clean_v2.CLASS_MAP[product], distance): 15
        for (product, distance) in clean_v2.TIER_D.cells
        if (product, distance) not in {("TUNA", "mid"), ("TUNA", "far")}
    }

    assert sorted(clean_v2.holdout_gaps(clean_v2.TIER_D, coverage)) == [
        ("century-tuna", "far"),
        ("century-tuna", "mid"),
    ]


def test_a_missing_or_unreadable_manifest_is_no_coverage_not_a_crash(tmp_path):
    """Before a shoot there is nothing staged, which is the ordinary case - so a directory
    that does not exist, one with no manifest, and one holding garbage all have to read as
    "no coverage" rather than taking the scaffold down."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    assert clean_v2.staged_coverage([tmp_path / "nope", empty, broken]) == {}


def test_the_scaffold_reads_the_staged_coverage_it_reports(tmp_path, capsys):
    """End to end through the command: the real staged set is 1,383 images and the three
    cells Tier A has not shot yet are exactly the ones a Tier D hold-out would break on, so
    the report has to name them rather than pass silently."""
    staged = tmp_path / "cleaned-v2"
    staged.mkdir()
    entries = [
        {"new_name": f"{cls}_{d}_{i}.jpg", "class": cls, "distance": d}
        for cls, cells in {
            "milo": [("close", 40), ("far", 40)],
            "century-tuna": [("close", 30)],
        }.items()
        for d, n in cells
        for i in range(n)
    ]
    (staged / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    root = tmp_path / "s3-capture"

    code = clean_v2.main(
        ["scaffold", "--tier", "d", "--root", str(root), "--out", str(staged)]
    )
    out = capsys.readouterr().out

    assert code == 0
    # The folders exist, so the shoot has somewhere to file into.
    assert (root / "MILO" / "FAR").is_dir()
    assert (root / "TUNA" / "CLOSE").is_dir()
    assert "held-out coverage check" in out
    # Named as the cells a hold-out would break on...
    assert "century-tuna/far" in out and "century-tuna/mid" in out
    # ...and *not* the cells this staged set already covers, since the gap list is what the
    # shoot has to act on (<class>/<distance> is the gap list's own format, not a folder).
    assert "milo/close" not in out and "milo/far" not in out


def test_the_scaffold_says_it_cannot_tell_when_nothing_is_staged(tmp_path, capsys):
    """The case a fresh project is in, and the one where a wall of gaps would be worst: with
    nothing staged there is no coverage to compare against, so the honest answer is that the
    question cannot be answered yet - not 24 lines of `no train images` for a shoot nobody
    has done.
    """
    root = tmp_path / "s3-capture"

    code = clean_v2.main(
        ["scaffold", "--tier", "d", "--root", str(root), "--out", str(tmp_path / "unused")]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "no staged set to compare against" in out
    assert "century-tuna/mid" not in out


def test_a_cell_only_the_held_out_session_covers_is_reported_as_unlearnable():
    """The mistake the rule prevents: hold out the only session that ever shot a cell and
    the model is never taught it, so its test reading measures the absence of training.
    `main` refuses to write a plan in this state rather than warning about it."""
    batches = _two_session_batches() + [_batch("safeguard_mid_s3", "safeguard", "mid", 30, "s3")]
    plan = plan_split.holdout_plan(batches, "s3")
    s = plan_split.summarize(plan, _entries(batches), batches)

    assert s["no_train_cells"] == [("safeguard", "mid")]


def test_a_whole_batch_plan_reports_the_cells_it_leaves_unlearnable():
    """Plan A's accepted compromise, made visible. Which cells went missing is only
    actionable if they are named, and the consequence is spelled out because "no train
    images" reads like a coverage nicety rather than an unlearnable class."""
    batches = [_batch("milo_close", "milo", "close", 10), _batch("milo_far", "milo", "far", 10)]
    entries = _entries(batches)
    plan = {e["new_name"]: ("train" if e["batch"] == "milo_close" else "test") for e in entries}
    text = plan_split.render("t", plan, batches, entries, "note")

    assert "1 of 2 cell(s) have no train images" in text
    assert "`milo/far`" in text


def test_the_acceptance_report_claims_only_the_cells_the_session_shot():
    """The scope of the claim, not just its size: a test session that skipped a cell says
    nothing about that cell, and saying so is what keeps the number honest."""
    batches = _two_session_batches() + [_batch("safeguard_mid", "safeguard", "mid", 30, "s1")]
    entries = _entries(batches)
    plan = plan_split.holdout_plan(batches, "s3")
    text = plan_split.acceptance_report(plan, batches, entries, "s3")

    assert "unseen capture session" in text
    assert "Cells in the acceptance set: 2 of 3." in text
    assert "absent from the acceptance set" in text
    assert "`safeguard/mid`" in text

    # And when the session shot every cell, the claim widens to the whole grid.
    full = _two_session_batches()
    full_plan = plan_split.holdout_plan(full, "s3")
    assert "Every cell is in the acceptance set" in plan_split.acceptance_report(
        full_plan, full, _entries(full), "s3"
    )


# --------------------------------------------------------------------------
# Staging: what a frame is called on disk
# --------------------------------------------------------------------------


def _negative_shoot(root: Path, frames: int = 2) -> Path:
    """A flat folder of decodable frames, the shape `--negatives` reads.

    Distinct random pixels rather than one repeated file: the cleaner collapses near-duplicates,
    so a fixture that looks like one photo would stage a single frame and the names under test
    would be unobservable.
    """
    src = root / "PUSHCART"
    for i in range(1, frames + 1):
        _frame(src / f"frame_{i:02d}.jpg")
    return src


def test_a_later_session_cannot_reuse_an_earlier_sessions_staged_names(tmp_path):
    """The bug this prevents, stated as the collision it is: frames are numbered per class from 1
    in every output directory, so a second negatives session used to stage as
    `negative_0001.jpg` again - and every reader joins a split to a frame *by name*, so the new
    set silently inherited the first one's train/valid/test entries from `splits.json` (measured:
    the neg2 shoot reported 35/10/5 for frames nothing had planned).

    The default session's names are asserted unchanged rather than only the later ones': the
    suffix is invisible for `s1`, or every existing staged set, manifest and project filename
    would be orphaned by a fix for a collision none of them have.
    """
    src = _negative_shoot(tmp_path)
    first, neg1, neg2 = (
        tmp_path / "cleaned-negatives",
        tmp_path / "cleaned-negatives-neg1",
        tmp_path / "cleaned-negatives-neg2",
    )

    def clean(out: Path, session: list[str]) -> None:
        assert clean_v2.main(
            ["clean", "--src", str(src), "--negatives", str(src), "--out", str(out), *session]
        ) == 0

    clean(first, [])  # the documented first negatives: staged as s1, and so unsuffixed
    clean(neg1, ["--session", "neg1"])
    clean(neg2, ["--session", "neg2"])

    def staged(out: Path) -> list[str]:
        return sorted(p.name for p in (out / "negative").glob("*.jpg"))

    assert staged(first) == ["negative_0001.jpg", "negative_0002.jpg"]
    assert staged(neg1) == ["negative_neg1_0001.jpg", "negative_neg1_0002.jpg"]
    assert staged(neg2) == ["negative_neg2_0001.jpg", "negative_neg2_0002.jpg"]
    # The names a reader joins on are disjoint, which is the whole point - including between
    # two later sessions, which is what a re-shoot of the same background produces.
    assert not set(staged(first)) & set(staged(neg2))
    assert not set(staged(neg1)) & set(staged(neg2))
    # And the manifest carries what the uploader will send, so the two agree - the rename is not
    # something only the directory listing knows about.
    with (neg2 / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "kept"]
    assert {r["new_name"] for r in rows} == set(staged(neg2))
    assert {r["batch"] for r in rows} == {"negative"}
    assert {r["session"] for r in rows} == {"neg2"}


# --------------------------------------------------------------------------
# Uploading to a project the API reports as public
# --------------------------------------------------------------------------


class _FakeResponse:
    """Just enough of an httpx response for `cmd_upload`: a status, a body, text for errors."""

    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _FakeHttpx:
    """A stand-in for the httpx module that `cmd_upload` imports *inside* itself.

    The import is function-local on purpose - the tool's own "httpx is not installed" message
    depends on it - so the module cache is the only seam: patching an attribute on `clean_v2` would
    leave the `import httpx` in the function reading the real module. Every request is recorded,
    because a gate whose entire job is to send nothing has to be judged on what was actually sent
    rather than on the exit status alone.
    """

    def __init__(self, public: bool = True):
        self.public = public
        self.posts: list[dict] = []
        self.gets: list[str] = []

    def Client(self, *args, **kwargs):  # noqa: N802 - named after the httpx API it replaces
        return _FakeClient(self)


class _FakeClient:
    def __init__(self, owner: _FakeHttpx):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        self.owner.gets.append(url)
        return _FakeResponse(
            200,
            {
                "project": {
                    "type": "object-detection",
                    "images": 1343,
                    "versions": 1,
                    "public": self.owner.public,
                }
            },
        )

    def post(self, url, params=None, files=None, timeout=None):
        self.owner.posts.append(dict(params or {}))
        return _FakeResponse(200, {"id": "asset-1"})


def _staged_for_upload(tmp_path: Path, names: tuple[str, ...]) -> Path:
    """A staged set holding only what `upload` reads: each manifest entry, and the file it names."""
    out = tmp_path / "cleaned-negatives"
    (out / "negative").mkdir(parents=True)
    for name in names:
        (out / "negative" / name).write_bytes(b"jpeg")
    entries = [
        {
            "new_name": name,
            "class": "negative",
            "distance": "",
            "batch": "negative",
            "session": "neg2",
            "tags": ["negative", "neg2"],
            "status": "kept",
        }
        for name in names
    ]
    (out / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return out


def _upload_args(out: Path, *extra: str) -> list[str]:
    return [
        "upload", "--out", str(out), "--project", "snc-grocery", "--api-key", "k",
        "--session", "neg2", *extra,
    ]


def test_the_public_project_gate_names_the_flag_that_opens_it():
    """The decision itself, all four ways. A gate that cannot be opened is a wall, and one that
    opens itself is the warning the gate exists to replace - so both directions are pinned."""
    assert clean_v2.public_send_refusal("snc-grocery", public=False, acknowledged=False) == ""
    assert clean_v2.public_send_refusal("snc-grocery", public=False, acknowledged=True) == ""
    assert clean_v2.public_send_refusal("snc-grocery", public=True, acknowledged=True) == ""

    refusal = clean_v2.public_send_refusal("snc-grocery", public=True, acknowledged=False)
    assert "PUBLIC" in refusal
    assert "snc-grocery" in refusal
    # The remedy, spelled as the flag that is actually parsed - a refusal naming a flag that does
    # not exist is the failure this is written to avoid.
    assert "--allow-public" in refusal


def test_upload_refuses_a_public_project_and_sends_nothing(tmp_path, monkeypatch):
    """The whole point, measured on what left the machine: the project is public, no
    acknowledgement was given, so the run stops - after it has read the project (that is how it
    knows) and before the first POST. No resume record either, so a later run starts clean."""
    out = _staged_for_upload(tmp_path, ("negative_neg2_0001.jpg", "negative_neg2_0002.jpg"))
    fake = _FakeHttpx(public=True)
    monkeypatch.setitem(sys.modules, "httpx", fake)

    with pytest.raises(SystemExit) as exit_:
        clean_v2.main(_upload_args(out))

    assert "--allow-public" in str(exit_.value)
    assert fake.gets, "the project was never read, so this refused for the wrong reason"
    assert fake.posts == [], "a refused upload sent something"
    assert not (out / "upload_state.json").exists()


def test_upload_proceeds_once_the_public_project_is_acknowledged(tmp_path, monkeypatch):
    """And the gate opens: the same project, the same frames, with the acknowledgement - landing in
    the session's own batch, under the distance-first tag the upload endpoint can carry."""
    names = ("negative_neg2_0001.jpg", "negative_neg2_0002.jpg")
    out = _staged_for_upload(tmp_path, names)
    fake = _FakeHttpx(public=True)
    monkeypatch.setitem(sys.modules, "httpx", fake)

    assert clean_v2.main(_upload_args(out, "--allow-public")) == 0

    assert len(fake.posts) == len(names)
    assert {p["name"] for p in fake.posts} == set(names)
    assert {p["batch"] for p in fake.posts} == {"negative_neg2"}
    state = json.loads((out / "upload_state.json").read_text(encoding="utf-8"))
    assert set(state) == set(names) and all(v["ok"] for v in state.values())


def test_a_private_project_is_never_asked_to_acknowledge_anything(tmp_path, monkeypatch):
    """The gate has to be invisible where it does not apply, or every ordinary upload grows a flag
    nobody needs and the flag stops meaning anything."""
    out = _staged_for_upload(tmp_path, ("negative_neg2_0001.jpg",))
    fake = _FakeHttpx(public=False)
    monkeypatch.setitem(sys.modules, "httpx", fake)

    assert clean_v2.main(_upload_args(out)) == 0
    assert len(fake.posts) == 1


def test_a_dry_run_on_a_public_project_still_prints_its_plan(tmp_path, monkeypatch, capsys):
    """The evidence the decision needs: a dry run sends nothing, so it is how someone sees which
    batches and how many frames are at stake. Refusing to print that would leave the only way to
    inspect the plan behind the flag the plan exists to inform."""
    out = _staged_for_upload(tmp_path, ("negative_neg2_0001.jpg", "negative_neg2_0002.jpg"))
    fake = _FakeHttpx(public=True)
    monkeypatch.setitem(sys.modules, "httpx", fake)

    assert clean_v2.main(_upload_args(out, "--dry-run")) == 0
    printed = capsys.readouterr().out

    assert "batch negative_neg2" in printed and "2 to send" in printed
    assert fake.posts == []
