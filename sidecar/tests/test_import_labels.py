"""Tests for `sidecar/tools/import_labels.py` - the project's decisions pulled into the local tree.

No network, no key and no project: a fake index and a fake details route stand in for the API, and
the geometry rules are tested against synthetic annotations. What is worth asserting is not that a
request is formatted correctly but three things that are silent when they are wrong:

* the **translation** is exact - the API's `x`/`y` are a box's centre (measured against the project,
  because a corner read is still a valid-looking row, just in the wrong place on the frame), and a
  class *name* lands on the index the local roster gives it rather than the project's class order;
* a frame with **any** problem is left untouched rather than written half - a dropped box or a
  clamped one changes what the frame says, and nothing downstream could tell;
* a pull is **idempotent**: a second run cannot quietly replace a decision somebody has since made
  in the annotator, and `--force` is the deliberate way to say it should.

    sidecar/.venv/Scripts/python.exe -m pytest tests/test_import_labels.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import import_labels

# Two local class names whose positions the project's own order does *not* share, so a test that
# asserts an index is asserting the name mapping rather than a coincidence of two orders.
BY_NAME = {name: index for index, name in enumerate(import_labels.CLASS_NAMES)}
BEAR_BRAND = "Bear Brand Fortified Powdered Milk 33g"  # local index 0
MILO = "Milo Chocolate Drink 22g Sachet"  # local index 5


def _jpeg(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 50, 60)).save(path, quality=70)


def _staged(root: Path, frames: dict[str, str], size: tuple[int, int] = (80, 60)) -> Path:
    """A staged set the store reads: `<class>/<name>` images plus the manifest it locates them through."""
    out = root / "cleaned-v2"
    entries = []
    for name, slug in frames.items():
        _jpeg(out / slug / name, size)
        entries.append({"new_name": name, "class": slug, "distance": "mid", "session": "s2"})
    (out / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return out


def _annotation(labels_and_boxes: list[tuple[str, float, float, float, float]], size=(80, 60)) -> dict:
    return {
        "width": size[0],
        "height": size[1],
        "boxes": [
            {"label": label, "x": x, "y": y, "width": w, "height": h}
            for label, x, y, w, h in labels_and_boxes
        ],
    }


def _record(count: int, image_id: str = "img1") -> dict:
    """One search-index record: the *type* of `annotations` is what separates the three states."""
    if count < 0:
        return {"id": image_id, "name": "x", "annotations": [], "tags": [], "split": "train"}
    return {
        "id": image_id,
        "name": "x",
        "annotations": {"count": count, "classes": {}},
        "tags": [],
        "split": "train",
    }


class _Response:
    def __init__(self, body: object, status: int = 200) -> None:
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> object:
        return self._body


class _Details:
    """The `get` half of the client: one image's details, keyed by the id the URL ends with.

    `fail_times` makes the first N responses fail, which is what the retry has to survive - and
    `always` makes it fail every time, which is what has to stop the run with the resumable
    sentence rather than a traceback.
    """

    def __init__(self, annotations: dict[str, dict], fail_times: int = 0, always: object = None):
        self.annotations = annotations
        self.fail_times = fail_times
        self.always = always
        self.calls: list[str] = []

    def get(self, url: str, params=None, timeout=None) -> _Response:
        image_id = url.rsplit("/", 1)[-1]
        self.calls.append(image_id)
        if self.always is not None:
            return _Response(self.always, status=500)
        if len(self.calls) <= self.fail_times:
            return _Response("", status=500)
        return _Response({"image": {"annotation": self.annotations[image_id]}})


def _wire(monkeypatch, index: dict[str, dict], annotations: dict[str, dict]) -> None:
    """Everything `main` reaches the network with, faked at the module's own seams."""
    monkeypatch.setattr(import_labels, "load_key", lambda project: "fake-key")
    monkeypatch.setattr(import_labels, "fetch_all", lambda client, key, project: index)
    monkeypatch.setattr(
        import_labels,
        "fetch_image",
        lambda client, key, project, image_id, **kwargs: {"annotation": annotations[image_id]},
    )


# --------------------------------------------------------------------------
# The pure translation: pixels + class names -> the rows the trainer reads
# --------------------------------------------------------------------------


def test_a_boxes_x_and_y_are_its_centre_not_a_corner():
    """The convention, pinned rather than assumed - and it cannot be checked from the rows.

    A corner read of the same numbers produces a *valid* row (`Box.valid` checks the centre and the
    size, not the extent), so the mistake is one nobody could see afterwards: the box is simply
    somewhere else on the frame. The measurement behind this is the project's own frames: read as
    centres, every box of a sampled dozen sits inside its frame; read as corners, dozens stick out
    by up to half the image.
    """
    boxes, problems = import_labels.translate(_annotation([(MILO, 40, 30, 20, 15)]), (80, 60))

    assert problems == []
    # Centre 40/80, 30/60 with half-frame extent - which is only true if x/y are the centre.
    assert [box.as_row() for box in boxes] == ["5 0.500000 0.500000 0.250000 0.250000"]


def test_a_class_name_lands_on_the_local_index_not_the_projects_order():
    """Names, not positions: the project's class list is its own order (and its own length), while a
    label row's `cls` is a position in `label_classes`' - so the same box is a different number on
    each side, and only the name can be trusted to cross."""
    boxes, problems = import_labels.translate(
        _annotation([(BEAR_BRAND, 40, 30, 40, 30), (MILO, 20, 15, 20, 15)]), (80, 60)
    )

    assert problems == []
    assert [box.cls for box in boxes] == [BY_NAME[BEAR_BRAND], BY_NAME[MILO]] == [0, 5]


def test_a_frame_with_an_unmappable_box_is_refused_whole():
    """The Palmolive case: the project holds a class this dataset does not, and a frame carrying one
    cannot be written at all. Dropping just that box would silently change what the frame says, so
    the refusal is the frame - and the sentence names the class to remove or relabel."""
    boxes, problems = import_labels.translate(
        _annotation([(MILO, 40, 30, 20, 15), ("Palmolive Naturals Bar Soap 85g", 20, 15, 10, 10)]),
        (80, 60),
    )

    assert boxes == []  # not even the box that mapped
    assert "'Palmolive Naturals Bar Soap 85g' is not one of this dataset's 7 names" in problems[0]


def test_every_bad_box_shape_is_named():
    """A wrong coordinate or a zero extent is a row the trainer would clip or drop, and a frame that
    reads as labeled while contributing nothing is the failure class this whole toolchain refuses."""
    cases = [
        ([{"label": MILO, "x": "x", "y": 1, "width": 4, "height": 4}], "non-numeric coordinate"),
        ([{"label": MILO, "x": 40, "y": 30, "width": 0, "height": 15}], "not a row the trainer can read"),
        ([{"label": MILO, "x": 400, "y": 30, "width": 20, "height": 15}], "not a row the trainer can read"),
    ]
    for raw_boxes, expected in cases:
        boxes, problems = import_labels.translate(
            {"width": 80, "height": 60, "boxes": raw_boxes}, (80, 60)
        )
        assert boxes == [] and any(expected in problem for problem in problems), raw_boxes


def test_a_missing_or_broken_frame_size_is_a_refusal():
    for body in ({"boxes": []}, {"width": None, "height": 60, "boxes": []}, {"width": 0, "height": 0}):
        boxes, problems = import_labels.translate(body, (80, 60))
        assert boxes == [] and problems, body


def test_a_crop_is_refused_and_a_resize_is_not():
    """The one geometry mistake the numbers cannot show on their own.

    Normalized boxes survive any *uniform* resize - a re-staged 640 copy is the same photo, so the
    coordinates still point at the same part of it - while a crop moves them to a different place
    on a frame that is no longer the annotated one. Aspect ratio is what separates the two.
    """
    # Same frame, different pixel size: 160x120 is the same 4:3 photo, and the row is unchanged.
    boxes, problems = import_labels.translate(_annotation([(MILO, 80, 60, 40, 30)], size=(160, 120)), (80, 60))
    assert problems == []
    assert [box.as_row() for box in boxes] == ["5 0.500000 0.500000 0.250000 0.250000"]

    # A crop: 160x90 against a 4:3 staged file is a different frame, whatever the numbers say.
    boxes, problems = import_labels.translate(
        _annotation([(MILO, 80, 45, 40, 30)], size=(160, 90)), (80, 60)
    )
    assert boxes == []
    assert "that is a different frame's coordinates, not a resize of this one" in problems[0]


# --------------------------------------------------------------------------
# The selection: who gets pulled, and the four kinds that do not
# --------------------------------------------------------------------------


def _frames(*pairs: tuple[str, str]):
    from types import SimpleNamespace

    return [SimpleNamespace(name=name, state=state) for name, state in pairs]


def test_the_selection_is_five_buckets_and_no_sixth():
    """One rule, laid out by state - the project's side and the local side read together.

    `unmatched` is the one that matters most: the project decided a frame no staged set here holds,
    which is either a stray upload or a set re-staged under other names, and either way a pull that
    ignored it would report success over a frame nobody can merge.
    """
    index = {
        "both.jpg": _record(1),
        "remote.jpg": _record(1),
        "remote_null.jpg": _record(0),
        "undecided.jpg": _record(-1),
        "stray.jpg": _record(1),
    }
    frames = _frames(
        ("both.jpg", "labeled"),
        ("remote.jpg", "unlabeled"),
        ("remote_null.jpg", "unlabeled"),
        ("undecided.jpg", "unlabeled"),
        ("local_only.jpg", "unlabeled"),
    )

    plan = import_labels.classify(index, frames)
    assert plan["pull"] == ["remote.jpg", "remote_null.jpg"]
    assert plan["already"] == ["both.jpg"]
    assert plan["unmatched"] == ["stray.jpg"]
    assert plan["unknown"] == ["local_only.jpg"]
    assert plan["undecided"] == ["undecided.jpg"]

    # `--force` moves what is already decided here into the pull instead of replacing it silently.
    assert import_labels.classify(index, frames, force=True)["pull"] == [
        "both.jpg",
        "remote.jpg",
        "remote_null.jpg",
    ]


# --------------------------------------------------------------------------
# End to end, through `main`, with the network faked at the module's seams
# --------------------------------------------------------------------------


def test_a_pull_writes_the_rows_and_the_provenance(tmp_path, monkeypatch, capsys):
    """The whole point: a decision that lived only in the project becomes a label file the merge
    reads, with provenance that says where it came from rather than pretending it was made here."""
    v2 = _staged(tmp_path, {"milo_0001.jpg": "milo"})
    annotations = v2.parent / "annotations-v2"
    index = {"milo_0001.jpg": _record(1, "id-milo")}
    _wire(monkeypatch, index, {"id-milo": _annotation([(MILO, 40, 30, 20, 15)])})

    assert import_labels.main(["--v2", str(v2)]) == 0

    assert (annotations / "milo_0001.txt").read_text(encoding="utf-8") == (
        "5 0.500000 0.500000 0.250000 0.250000\n"
    )
    record = json.loads((annotations / "provenance.json").read_text(encoding="utf-8"))["milo_0001.jpg"]
    assert record["state"] == "labeled" and record["rows"] == 1
    assert record["provider"] == "roboflow:snc-grocery"
    # Not machine-only: the API records that an image is annotated, never who drew it, and the
    # default is the footing v1's hand-labelled Roboflow frames are already on.
    assert record["machine_only"] is False
    assert "authorship" in capsys.readouterr().out


def test_a_second_run_leaves_a_local_decision_alone(tmp_path, monkeypatch):
    """Idempotence, and whose work it protects: a person who edits an imported frame in the annotator
    keeps that edit. `--force` is the deliberate override, not the default."""
    v2 = _staged(tmp_path, {"milo_0001.jpg": "milo"})
    annotations = v2.parent / "annotations-v2"
    _wire(monkeypatch, {}, {"id-milo": _annotation([(MILO, 40, 30, 20, 15)])})
    monkeypatch.setattr(
        import_labels,
        "fetch_all",
        lambda client, key, project: {"milo_0001.jpg": _record(1, "id-milo")},
    )

    assert import_labels.main(["--v2", str(v2)]) == 0
    (annotations / "milo_0001.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")  # an edit

    assert import_labels.main(["--v2", str(v2)]) == 0
    assert (annotations / "milo_0001.txt").read_text(encoding="utf-8") == "0 0.1 0.1 0.1 0.1\n"

    assert import_labels.main(["--v2", str(v2), "--force"]) == 0
    assert (annotations / "milo_0001.txt").read_text(encoding="utf-8").startswith("5 0.5")


def test_awaiting_review_is_the_other_provenance_for_the_same_rows(tmp_path, monkeypatch):
    """The flag for labels that came from Label Assist: identical rows, different claim - and
    `machine_only` is what puts them in the annotator's review pass and in the acceptance gate's
    count, so the operator has to choose rather than have the tool choose for them."""
    v2 = _staged(tmp_path, {"milo_0001.jpg": "milo"})
    annotations = v2.parent / "annotations-v2"
    _wire(monkeypatch, {"milo_0001.jpg": _record(1, "id-milo")}, {"id-milo": _annotation([(MILO, 40, 30, 20, 15)])})

    assert import_labels.main(["--v2", str(v2), "--awaiting-review"]) == 0

    record = json.loads((annotations / "provenance.json").read_text(encoding="utf-8"))["milo_0001.jpg"]
    assert record["machine_only"] is True
    assert (annotations / "milo_0001.txt").read_text(encoding="utf-8").startswith("5 0.5")


def test_a_null_becomes_the_empty_file_a_hard_negative_needs(tmp_path, monkeypatch):
    """The 50 negatives are decisions too - an *empty* label file plus a provenance state, which is
    the local spelling of the project's `{"count": 0}` and the state a naive sweep would re-open.

    Staged the way they are in life: their own set beside the product one, which is why the pull
    reads the defaults (`cleaned-negatives`) and not only `--v2`.
    """
    v2 = _staged(tmp_path, {"milo_0001.jpg": "milo"})
    negatives = tmp_path / "cleaned-negatives"
    _jpeg(negatives / "negative" / "cam0_0001.jpg")
    (negatives / "manifest.json").write_text(
        json.dumps(
            [{"new_name": "cam0_0001.jpg", "class": "negative", "distance": "", "session": "s2"}]
        ),
        encoding="utf-8",
    )
    annotations = v2.parent / "annotations-v2"
    _wire(monkeypatch, {"cam0_0001.jpg": _record(0, "id-neg")}, {})

    assert import_labels.main(["--v2", str(v2)]) == 0

    assert (annotations / "cam0_0001.txt").read_text(encoding="utf-8") == ""
    record = json.loads((annotations / "provenance.json").read_text(encoding="utf-8"))["cam0_0001.jpg"]
    assert record["state"] == "null" and record["machine_only"] is False


def test_a_refused_frame_is_left_outstanding_and_the_exit_code_says_so(tmp_path, monkeypatch, capsys):
    """A refusal must not leave a file behind: an outstanding frame is visible work, while a written
    frame that could not be read is a label nobody drew and nothing to say so."""
    v2 = _staged(tmp_path, {"milo_0001.jpg": "milo", "milo_0002.jpg": "milo"})
    annotations = v2.parent / "annotations-v2"
    index = {"milo_0001.jpg": _record(1, "id-1"), "milo_0002.jpg": _record(1, "id-2")}
    _wire(
        monkeypatch,
        index,
        {
            "id-1": _annotation([(MILO, 40, 30, 20, 15)]),
            "id-2": _annotation([("Palmolive Naturals Bar Soap 85g", 40, 30, 20, 15)]),
        },
    )

    assert import_labels.main(["--v2", str(v2)]) == 2

    assert (annotations / "milo_0001.txt").is_file()
    assert not (annotations / "milo_0002.txt").exists()  # refused means untouched
    assert "milo_0002.jpg" in capsys.readouterr().out


def test_a_frame_at_another_size_is_pulled_with_a_note(tmp_path, monkeypatch, capsys):
    """A re-staged copy is the same photo: normalized boxes transfer, so the frame is pulled - and
    the note is kept because it is the operator's only evidence that the files were re-staged."""
    v2 = _staged(tmp_path, {"milo_0001.jpg": "milo"})  # staged at 80x60
    annotations = v2.parent / "annotations-v2"
    _wire(
        monkeypatch,
        {"milo_0001.jpg": _record(1, "id-milo")},
        {"id-milo": _annotation([(MILO, 80, 60, 40, 30)], size=(160, 120))},
    )

    assert import_labels.main(["--v2", str(v2)]) == 0
    printed = capsys.readouterr().out

    assert (annotations / "milo_0001.txt").read_text(encoding="utf-8").startswith("5 0.5")
    assert "1 frame(s) at a different size (same aspect)" in printed


def test_a_dry_run_writes_nothing_and_the_json_readout_carries_the_plan(tmp_path, monkeypatch, capsys):
    """The plan is the part worth reading before 858 frames are fetched - and the mode says what it
    does not check (box geometry needs the details route, which a dry run never calls)."""
    v2 = _staged(tmp_path, {"milo_0001.jpg": "milo", "milo_0002.jpg": "milo"})
    annotations = v2.parent / "annotations-v2"
    _wire(
        monkeypatch,
        {"milo_0001.jpg": _record(1, "id-1"), "milo_0002.jpg": _record(1, "id-2")},
        {},
    )

    assert import_labels.main(["--v2", str(v2), "--dry-run", "--json"]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["planned"] == 2 and summary["remote"] == {"labeled": 2, "null": 0, "unlabeled": 0}
    assert not list(annotations.glob("*.txt")) if annotations.is_dir() else True
    assert not (annotations / "provenance.json").exists()


def test_the_details_route_is_retried_then_says_the_pull_is_resumable():
    """One flaky response in the middle of hundreds must not send the operator back to the start,
    and a hard failure must say what is on disk rather than raise a traceback."""
    flaky = _Details({"id-1": _annotation([])}, fail_times=2)
    sleeps: list[float] = []
    image = import_labels.fetch_image(
        flaky, "key", "project", "id-1", sleep=lambda seconds: sleeps.append(seconds)
    )

    assert image == {"annotation": {"width": 80, "height": 60, "boxes": []}}
    assert len(flaky.calls) == 3 and sleeps == [1, 2]  # backoff, and only between attempts

    with pytest.raises(SystemExit) as stopped:
        import_labels.fetch_image(
            _Details({}, always="boom"), "key", "project", "id-9", sleep=lambda seconds: None
        )
    assert "re-run the command and it continues from there" in str(stopped.value)
    assert "HTTP 500" in str(stopped.value)


def test_the_pull_refuses_a_v2_that_is_not_a_staged_set(tmp_path, monkeypatch):
    """The store's frames come from a manifest, so a wrong `--v2` would report a clean pull of
    nothing; `read_v2`'s guard is reused so this tool cannot answer about a directory the build
    would refuse too."""
    empty = tmp_path / "cleaned-v2"
    empty.mkdir()
    monkeypatch.setattr(import_labels, "load_key", lambda project: "fake-key")

    with pytest.raises(SystemExit) as refused:
        import_labels.main(["--v2", str(empty)])

    assert "not pulling into this --v2" in str(refused.value)
    assert "no manifest.json" in str(refused.value)
