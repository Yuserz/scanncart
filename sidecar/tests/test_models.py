"""Selectable weights: discovery, filtering, and the route that serves them.

The point of this feature is that the picker stops being a fixed list. `settings_store`
validates model names without touching the filesystem (settings tests stay pure), so
nothing there can answer "what exists?" - `app/models.py` does, and these tests pin that
the two agree: everything listed must be something the settings PATCH would accept.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.models import (
    MODELS_DIR,
    installed_models,
    read_record,
    record_requirement,
    requirement_for,
)
from app.roster import ROSTER
from app.settings import Settings
from app.settings_store import (
    ALLOWED_MODELS,
    ALLOWED_RESIZE_MODES,
    CUSTOM_MODEL_DIR,
    CUSTOM_MODEL_SUFFIXES,
    is_custom_model,
)


def _touch(directory, name: str, data: bytes = b"\x00") -> None:
    (directory / name).write_bytes(data)


def _values(directory) -> list[str]:
    return [m.value for m in installed_models(directory)]


def _client() -> TestClient:
    return TestClient(build_app(lambda: AppState(settings=Settings(), db_path=":memory:")))


# --------------------------------------------------------------------------
# the listing
# --------------------------------------------------------------------------


def test_a_missing_directory_is_empty_not_an_error(tmp_path):
    """A packaged app with no custom weights is a normal state; erroring would make the
    picker look broken on exactly the installs that are fine."""
    assert installed_models(tmp_path / "nope") == []


def test_it_lists_both_suffixes_as_selectable_values(tmp_path):
    _touch(tmp_path, "scanncart-grocery.onnx")
    _touch(tmp_path, "scanncart-grocery-v2.pt")
    assert _values(tmp_path) == [
        f"{CUSTOM_MODEL_DIR}scanncart-grocery-v2.pt",
        f"{CUSTOM_MODEL_DIR}scanncart-grocery.onnx",
    ]


def test_everything_listed_is_something_the_validator_accepts(tmp_path):
    """The invariant that makes the picker safe to trust: a listed value can be saved.
    Otherwise an operator picks a name from the UI and gets an error about that name."""
    _touch(tmp_path, "good-v2.pt")
    _touch(tmp_path, "notes.txt")
    _touch(tmp_path, "weights.pt.bak")
    (tmp_path / "nested").mkdir()
    _touch(tmp_path / "nested", "inside.pt")
    _touch(tmp_path, ".hidden.pt")

    listed = _values(tmp_path)
    assert listed == [f"{CUSTOM_MODEL_DIR}good-v2.pt"]
    assert all(is_custom_model(v) for v in listed)


def test_a_directory_named_like_a_model_is_not_offered(tmp_path):
    """`models/weights.pt/` would pass a suffix check and fail at load."""
    (tmp_path / "weights.pt").mkdir()
    assert installed_models(tmp_path) == []


def test_the_stock_models_are_reported_separately_from_discovered_ones():
    """They are answered by different places - a constant versus a directory read - and the
    picker needs both: a stock model can be selected before it is downloaded (capture
    start fetches it), a custom one has to exist."""
    assert "yolo11n.pt" in ALLOWED_MODELS
    assert all(not name.startswith(CUSTOM_MODEL_DIR) for name in ALLOWED_MODELS)
    assert all(name.endswith(CUSTOM_MODEL_SUFFIXES) for name in ALLOWED_MODELS)


# --------------------------------------------------------------------------
# the route
# --------------------------------------------------------------------------


def test_the_route_serves_stock_and_installed_weights(monkeypatch, tmp_path):
    _touch(tmp_path, "scanncart-grocery-v2.pt")
    monkeypatch.setattr("app.main.installed_models", lambda: installed_models(tmp_path))
    body = _client().get("/api/models").json()

    assert [m["value"] for m in body["installed"]] == [
        f"{CUSTOM_MODEL_DIR}scanncart-grocery-v2.pt"
    ]
    assert "yolo11n.pt" in body["stock"]
    assert body["directory"] == str(MODELS_DIR)
    # Every value the picker will offer, in one place - so the renderer needs no mirror of
    # the stock list to render it (labels stay there; the values do not).
    assert set(body["stock"]) | {m["value"] for m in body["installed"]}


def test_the_route_is_a_read_only_get():
    """It must never be the thing that *changes* the model - that is the settings PATCH, so
    a mis-typed URL cannot swap the weights mid-session."""
    r = _client().post("/api/models", json={})
    assert r.status_code == 405


def test_the_route_reports_a_directory_even_when_nothing_is_in_it(monkeypatch):
    monkeypatch.setattr("app.main.installed_models", lambda: [])
    body = _client().get("/api/models").json()
    assert body["installed"] == []
    assert body["directory"].endswith("models"), body["directory"]


def test_the_lookup_answers_only_for_weights_that_exist(tmp_path):
    """`requirement_for` is read at capture start, so it must answer the same `None` for
    every kind of unknown rather than raising: a stock weight, a name the settings validator
    would reject, and a file that is not there. The capture path has no handler for an
    exception from here, and a missing record is not a reason a capture should not start."""
    _with_record(tmp_path, resize_mode="stretch")

    assert requirement_for("models/scanncart-grocery-v2.pt", tmp_path) == "stretch"
    assert requirement_for("yolo11n.pt", tmp_path) is None
    assert requirement_for("models/gone.pt", tmp_path) is None
    assert requirement_for("../escape.pt", tmp_path) is None
    assert requirement_for("models/nested/inside.pt", tmp_path) is None


def test_a_corrupt_record_reads_as_no_requirement_not_an_error(tmp_path):
    """Same three states as the panel, and the same answer for the unreadable one: the
    operator's own file must not be able to fail a capture with an OSError."""
    _touch(tmp_path, "hand-copied.pt")
    (tmp_path / "hand-copied.json").write_text("{not json", encoding="utf-8")
    assert requirement_for("models/hand-copied.pt", tmp_path) is None


def test_a_record_naming_a_mode_the_settings_patch_would_reject_is_ignored(tmp_path):
    """A hand-edited record must not be able to put a value on the wire that PATCH
    /api/settings would refuse - it is dropped in `read_record`, which both callers share."""
    _touch(tmp_path, "odd.pt")
    (tmp_path / "odd.json").write_text(
        json.dumps({"resize_mode": "crop"}), encoding="utf-8"
    )
    assert requirement_for("models/odd.pt", tmp_path) is None
    (only,) = installed_models(tmp_path)
    assert only.resize_mode is None
    assert only.auto_resolves_to == "letterbox"


VALIDATION = [
    {
        "split": "test",
        "floor": 0.85,
        "measured_at": "2026-09-22T10:40:00",
        "aggregates": {"precision": 0.9, "recall": 0.82, "mAP50": 0.88, "mAP50-95": 0.61},
        "per_class": [
            {"name": "bear-brand", "recall": 0.9, "instances": 10},
            {"name": "century-tuna", "recall": 0.62, "instances": 30},
            {"name": "lucky-me", "recall": None, "instances": 0},
        ],
    }
]


def test_a_recorded_measurement_is_read_back_whole(monkeypatch, tmp_path):
    """What the panel renders, at the wire: which split, against which floor, per class with
    the instance count that says whether the number means anything.
    """
    _with_record(tmp_path, resize_mode="stretch", validation=VALIDATION)
    monkeypatch.setattr("app.main.installed_models", lambda: installed_models(tmp_path))

    (only,) = installed_models(tmp_path)
    assert only.recorded is True
    (block,) = only.validation
    assert (block.split, block.floor) == ("test", 0.85)
    assert block.aggregates["mAP50"] == 0.88
    assert [(c.name, c.recall, c.instances) for c in block.per_class] == [
        ("bear-brand", 0.9, 10),
        ("century-tuna", 0.62, 30),
        ("lucky-me", None, 0),
    ]
    # And the same thing as the route serves it.
    served = _client().get("/api/models").json()["installed"][0]["validation"][0]
    assert served["split"] == "test" and served["per_class"][2]["recall"] is None


def test_an_unmeasured_class_survives_the_round_trip_as_null_not_zero(tmp_path):
    """The distinction the record exists to carry. Ultralytics answers 0.0 for a class the
    split holds no instances of, and a 0 would arrive in the panel looking like a total miss
    - sending the operator to shoot more images of an item when what is missing is captures
    in this split. `None` is what says so.
    """
    _touch(tmp_path, "x.pt")
    (tmp_path / "x.json").write_text(
        json.dumps({"validation": VALIDATION}), encoding="utf-8"
    )
    (block,) = read_record(tmp_path / "x.pt")["validation"]
    unmeasured = [c for c in block.per_class if c.recall is None]
    assert [c.name for c in unmeasured] == ["lucky-me"]
    assert unmeasured[0].instances == 0


def test_a_block_whose_floor_is_unusable_is_dropped_rather_than_defaulted(tmp_path):
    """The floor is what makes a recall interpretable, so a block without a trustworthy one
    cannot be rendered at all. Defaulting it to 0.0 would turn every class into a pass - a
    verdict that looks like a result and is the opposite of the truth.
    """
    _touch(tmp_path, "x.pt")
    for bad_floor in (None, "0.85", float("nan"), 1.5):
        (tmp_path / "x.json").write_text(
            json.dumps({"validation": [{**VALIDATION[0], "floor": bad_floor}]}), encoding="utf-8"
        )
        assert read_record(tmp_path / "x.pt")["validation"] == [], bad_floor


def test_a_corrupt_recall_is_dropped_rather_than_clamped_or_shown_raw(tmp_path):
    """No honest pass produces a recall outside 0..1 or a NaN, so a value like that is
    corrupt rather than extreme. Clamping it would be this program's invention, and showing
    it would read as the panel being broken instead of the file.
    """
    _touch(tmp_path, "x.pt")
    block = {
        **VALIDATION[0],
        "per_class": [
            {"name": "good", "recall": 0.9, "instances": 4},
            {"name": "too-high", "recall": 42.0, "instances": 4},
            {"name": "negative", "recall": -1.0, "instances": 4},
            {"name": "not-a-number", "recall": "high", "instances": 4},
            {"name": "boolean", "recall": True, "instances": 4},
            {"name": "", "recall": 0.9, "instances": 4},
        ],
    }
    (tmp_path / "x.json").write_text(json.dumps({"validation": [block]}), encoding="utf-8")

    (parsed,) = read_record(tmp_path / "x.pt")["validation"]
    assert [c.name for c in parsed.per_class] == ["good"]


DISTANCES = [
    {
        "distance": "close",
        "images": 20,
        "aggregates": {"mAP50": 0.91},
        "per_class": [
            {"name": "bear-brand", "recall": 0.97, "instances": 20},
            {"name": "milo", "recall": None, "instances": 0},
        ],
    },
    {
        "distance": "far",
        "images": 35,
        "aggregates": {"mAP50": 0.55},
        "per_class": [{"name": "bear-brand", "recall": 0.62, "instances": 30}],
    },
]


def test_a_per_distance_breakdown_is_read_back_whole(monkeypatch, tmp_path):
    """The breakdown travels, because it is the half of the measurement the per-class number
    cannot express: `bear-brand` scores 0.62 at `far` while the split as a whole looks healthy,
    and the panel has to be able to show both.
    """
    _with_record(tmp_path, resize_mode="stretch", validation=[{**VALIDATION[0], "per_distance": DISTANCES}])
    monkeypatch.setattr("app.main.installed_models", lambda: installed_models(tmp_path))

    (only,) = installed_models(tmp_path)
    (block,) = only.validation
    assert [d.distance for d in block.per_distance] == ["close", "far"]
    assert [d.images for d in block.per_distance] == [20, 35]
    assert block.per_distance[0].aggregates["mAP50"] == 0.91
    # The same three states per distance: `None` is still "this distance held no instances",
    # which is not a miss and must not arrive as one.
    assert [(c.name, c.recall, c.instances) for c in block.per_distance[0].per_class] == [
        ("bear-brand", 0.97, 20),
        ("milo", None, 0),
    ]
    # The floor is the block's, not repeated per distance: a second copy could only disagree
    # with the one these cells were judged against.
    assert set(block.per_distance[0].model_dump()) == {"distance", "images", "aggregates", "per_class"}

    served = _client().get("/api/models").json()["installed"][0]["validation"][0]
    assert [d["distance"] for d in served["per_distance"]] == ["close", "far"]
    assert served["per_distance"][1]["per_class"][0]["recall"] == 0.62


def test_a_record_without_a_breakdown_reads_as_empty_not_missing(tmp_path):
    """An older record, or one written with `--no-per-distance`. Empty, so the panel renders it
    exactly as it did before the breakdown existed - "not measured by distance" never has to be
    told apart from "this dataset has no distances".
    """
    _touch(tmp_path, "x.pt")
    (tmp_path / "x.json").write_text(json.dumps({"validation": VALIDATION}), encoding="utf-8")
    (block,) = read_record(tmp_path / "x.pt")["validation"]
    assert block.per_distance == []


def test_a_corrupt_breakdown_costs_the_breakdown_and_not_the_measurement(tmp_path):
    """The direction that matters: the breakdown is extra detail under an acceptance number, so
    a hand-edited or partly-written one must lose only itself. Dropping the block instead would
    throw away the number the operator actually came for.
    """
    _touch(tmp_path, "x.pt")
    for junk in ("nope", 5, {"far": 1}, ["nope", {}], [{"images": 4}], [{"distance": "", "per_class": DISTANCES[1]["per_class"]}]):
        block = {**VALIDATION[0], "per_distance": junk}
        (tmp_path / "x.json").write_text(json.dumps({"validation": [block]}), encoding="utf-8")
        (parsed,) = read_record(tmp_path / "x.pt")["validation"]
        assert parsed.per_distance == [], junk
        assert [c.name for c in parsed.per_class] == ["bear-brand", "century-tuna", "lucky-me"]


def test_a_breakdown_entry_with_no_usable_class_is_dropped(tmp_path):
    """A distance row with nothing renderable in it says nothing, and a panel cannot show an
    empty grid row as "measured and clean"."""
    _touch(tmp_path, "x.pt")
    block = {
        **VALIDATION[0],
        "per_distance": [
            {"distance": "close", "images": 4, "per_class": []},
            {"distance": "far", "images": 4, "per_class": [{"name": "", "recall": 0.9}]},
            {"distance": "mid", "images": "lots", "per_class": [{"name": "milo", "recall": 0.9, "instances": -3}]},
        ],
    }
    (tmp_path / "x.json").write_text(json.dumps({"validation": [block]}), encoding="utf-8")

    (parsed,) = read_record(tmp_path / "x.pt")["validation"]
    assert [d.distance for d in parsed.per_distance] == ["mid"]
    # A junk count is a zero rather than a non-number, so the row still renders.
    assert (parsed.per_distance[0].images, parsed.per_distance[0].per_class[0].instances) == (0, 0)


def test_an_unreadable_measurement_does_not_read_as_a_score_of_zero(tmp_path):
    """Three states, one of which is dangerous: a validation field that is not a list at all
    has to come back empty ("not measured") rather than as a block with nothing in it.
    """
    _touch(tmp_path, "x.pt")
    for junk in ("nope", 5, {"test": VALIDATION[0]}, ["nope"], [{}, {"split": "test"}]):
        (tmp_path / "x.json").write_text(
            json.dumps({"validation": junk}), encoding="utf-8"
        )
        assert read_record(tmp_path / "x.pt")["validation"] == [], junk


def test_a_nonsense_aggregate_is_dropped_without_taking_its_class_list(tmp_path):
    """The metrics strip is decoration next to the per-class table; a corrupt entry in it must
    not cost the numbers the operator actually came for.
    """
    _touch(tmp_path, "x.pt")
    block = {**VALIDATION[0], "aggregates": {"mAP50": 0.88, "recall": "n/a", "x": None}}
    (tmp_path / "x.json").write_text(json.dumps({"validation": [block]}), encoding="utf-8")

    (parsed,) = read_record(tmp_path / "x.pt")["validation"]
    assert parsed.aggregates == {"mAP50": 0.88}
    assert len(parsed.per_class) == 3


def test_the_response_carries_paths_and_nothing_secret():
    """It reaches the renderer, so it carries the same guarantee as the settings response:
    filenames, never credentials."""
    body = json.dumps(_client().get("/api/models").json()).lower()
    assert "api_key" not in body and "token" not in body


# --------------------------------------------------------------------------
# the record written beside a weight by `train_model.py --install`
# --------------------------------------------------------------------------
#
# The requirement has to be read from somewhere, and the weight itself cannot be it: a
# `.pt` records the training run, not the dataset geometry, and the filename is a
# convention. So the tool that installs the weights writes what it knows, and this is the
# read side. Three states matter, and only one of them is safe to act on.


def _with_record(tmp_path, name: str = "scanncart-grocery-v2.pt", **record):
    _touch(tmp_path, name)
    (tmp_path / f"{Path(name).stem}.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_a_recorded_requirement_is_reported(tmp_path):
    _with_record(tmp_path, resize_mode="stretch", source="snc-grocery version 2", model="x")
    (only,) = installed_models(tmp_path)
    assert only.value == f"{CUSTOM_MODEL_DIR}scanncart-grocery-v2.pt"
    assert only.resize_mode == "stretch"
    assert only.source == "snc-grocery version 2"


def test_weights_with_no_record_report_no_requirement_rather_than_a_guess(tmp_path):
    """`None`, not `"stretch"` and not `"auto"`: those weights were copied in by hand, so
    nothing is known about how they were trained, and a guess in this field would be
    indistinguishable from a fact in the panel that renders it. An operator setting it
    deliberately is the correct outcome.
    """
    _touch(tmp_path, "hand-copied.pt")
    (only,) = installed_models(tmp_path)
    assert only.resize_mode is None
    assert only.source == ""
    # Still selectable, still listed - the requirement is unknown, the model is not.
    assert is_custom_model(only.value)


def test_a_corrupt_or_hostile_record_cannot_break_the_route(tmp_path):
    """A hand-edited file must not be able to take down `/api/models`, which the Admin Panel
    loads on every open - and a value the settings PATCH would reject must not be presented
    as a requirement to satisfy.
    """
    _touch(tmp_path, "broken.pt")
    (tmp_path / "broken.json").write_text("not json at all", encoding="utf-8")
    _with_record(tmp_path, "wrong-mode.pt", resize_mode="banana")
    _with_record(tmp_path, "list.pt")
    (tmp_path / "list.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    by_value = {m.value.rsplit("/", 1)[-1]: m for m in installed_models(tmp_path)}
    assert set(by_value) == {"broken.pt", "wrong-mode.pt", "list.pt"}
    assert all(m.resize_mode is None for m in by_value.values())
    assert all(m.class_names == [] for m in by_value.values())
    # No class list means no finding about one. `class_list_problems([])` would say "cannot
    # predict 8 of the 8 roster classes", which is a verdict about a list nobody has seen -
    # the wrong reading for a weight whose record simply predates the field.
    assert all(m.class_warnings == [] for m in by_value.values())
    # Every field empty, but the file *is* there - which is the whole content of `recorded`,
    # and the difference between a panel that can say "nothing measured yet" and one that has
    # to stay silent. A corrupt record is still a record the tool wrote.
    assert read_record(tmp_path / "broken.pt") == {
        "recorded": True,
        "resize_mode": None,
        "source": "",
        "class_names": [],
        "validation": [],
    }
    assert read_record(tmp_path / "nothing-here.pt") == {
        "recorded": False,
        "resize_mode": None,
        "source": "",
        "class_names": [],
        "validation": [],
    }


def test_a_class_list_of_junk_reads_as_unrecorded_and_a_real_one_survives(tmp_path):
    """Two directions in one, because they are the same rule seen twice: an entry that carries no
    name is dropped, and a list where nothing survives is *unrecorded* rather than empty-but-
    recorded. That second half is what keeps the listing honest - `class_list_problems([])`
    would report all 8 roster classes missing, which is a verdict about a list nobody has seen.
    A name that does survive is reported as it stands, trimmed.
    """
    _touch(tmp_path, "junk.pt")
    (tmp_path / "junk.json").write_text(
        json.dumps({"class_names": [1, None, "", "   ", {"name": "x"}]}), encoding="utf-8"
    )
    (only,) = installed_models(tmp_path)
    assert only.class_names == []
    assert only.class_warnings == []

    _touch(tmp_path, "good.pt")
    (tmp_path / "good.json").write_text(
        json.dumps({"class_names": ["  Bear Brand Fortified Powdered Milk 33g  ", 7]}),
        encoding="utf-8",
    )
    by_value = {m.value.rsplit("/", 1)[-1]: m for m in installed_models(tmp_path)}
    assert by_value["good.pt"].class_names == ["Bear Brand Fortified Powdered Milk 33g"]


def test_the_install_step_writes_the_record_this_reader_finds(tmp_path):
    """The seam, end to end: `train_model.py --install` decides the filename and the contents,
    `app/models.py` finds the file and reads them, and *nothing* but this test would notice
    if the two stopped agreeing - a renamed record is a requirement silently reverting to
    "not recorded", with no error anywhere. Uses the tool's own writer for that reason,
    rather than a hand-built JSON that would agree with the reader by construction.
    """
    import generations
    import train_model

    best = tmp_path / "best.pt"
    best.write_bytes(b"weights")
    train_model.install(
        best,
        tmp_path / "models",
        record=train_model.weight_record(
            generations.V2, 2, "snc-grocery", class_names=list(ROSTER)
        ),
    )

    (only,) = installed_models(tmp_path / "models")
    assert only.value == f"{CUSTOM_MODEL_DIR}{generations.V2.weight_name}"
    assert only.source == "snc-grocery version 2"
    # And the value the tool records is one this app would accept as a setting.
    assert only.resize_mode in ALLOWED_RESIZE_MODES
    # The second fact only the training run knows: what these weights predict. Read back through
    # the same listing the panel loads, so a renamed or reshaped field fails here instead of
    # surfacing as an unflagged model that logs one product under three labels.
    assert only.class_names == list(ROSTER)
    assert only.class_warnings == []


def test_a_v1_weight_installs_and_the_listing_names_the_one_class_it_cannot_predict(tmp_path):
    """The other direction of the same guard, and the case this project's v1 build lands in: v1's
export declares seven classes, so a v1 weight is *correct* and still cannot predict Palmolive -
the one class v2 adds (`generations.added_over`). Nothing it does predict is wrong, which is
exactly why the listing has to say it: with no sentence, the only symptom is that one product
never appears in the item log. Uses the tool's own writer, like the seam test above.
    """
    import generations
    import train_model

    best = tmp_path / "best.pt"
    best.write_bytes(b"weights")
    train_model.install(
        best,
        tmp_path / "models",
        name=generations.V1.weight_name,
        record=train_model.weight_record(
            generations.V1, 1, "scanncart-grocery", class_names=list(generations.V1.classes)
        ),
    )

    (only,) = installed_models(tmp_path / "models")
    assert only.value == f"{CUSTOM_MODEL_DIR}{generations.V1.weight_name}"
    assert only.recorded is True
    # The geometry requirement travels with it too, and is the value the settings PATCH accepts.
    assert only.resize_mode in ALLOWED_RESIZE_MODES
    assert only.class_names == list(generations.V1.classes)

    (warning,) = only.class_warnings
    assert "cannot predict 1 of the 8 roster classes" in warning
    assert "Palmolive Naturals Bar Soap 85g" in warning
    # Neither of the other two findings: every name it predicts *is* a roster name, and none of
    # them carries a distance - which is what makes this the quiet one.
    assert "carry a distance" not in warning
    assert "not in this app's roster" not in warning


def test_the_listing_flags_a_weight_trained_per_product_and_distance(monkeypatch, tmp_path):
    """The failure this field exists for, caught from the *listing* - before the weight is
    selected, let alone run. A project whose class list was split by distance trains a head with
    an output per product-and-distance, so every box comes back under a label the roster does not
    contain; the filename and the checkpoint say nothing about it, and once such a model is
    running the app's own log fills with near-duplicates of one product.
    """
    names = [f"Palmolive Naturals Bar Soap 85g {d}" for d in ("close", "mid", "far")]
    _touch(tmp_path, "distance-split.pt")
    (tmp_path / "distance-split.json").write_text(
        json.dumps({"class_names": names}), encoding="utf-8"
    )

    (only,) = installed_models(tmp_path)
    assert only.class_names == names
    # One finding, and it is the distance one: the "cannot predict 8 of 8" sentence is true for
    # this list and deliberately suppressed, because a reader cannot act on it until the list is
    # a roster - so printing both would dilute the one that names the fix.
    assert len(only.class_warnings) == 1
    assert "distance" in only.class_warnings[0]
    # Through the route the panel actually loads, not only the function: this is the difference
    # between "the fact is known" and "the fact reaches the screen before the model does".
    monkeypatch.setattr("app.main.installed_models", lambda: installed_models(tmp_path))
    served = _client().get("/api/models").json()["installed"]
    entry = next(m for m in served if m["value"].endswith("distance-split.pt"))
    assert entry["class_names"] == names
    assert entry["class_warnings"] == only.class_warnings


# --------------------------------------------------------------------------
# recording a requirement from the app
# --------------------------------------------------------------------------
#
# The other writer of these files is `--install`, which knows the requirement because it knows
# the training run. This path exists for weights that arrived some other way, where the operator
# holds the fact - and it is what the Admin Panel's "record it now" button calls, so the merge
# rules here are what stop that button from damaging a record it did not write.


def test_recording_writes_the_requirement_beside_the_weights(tmp_path):
    _touch(tmp_path, "hand-copied.pt")
    path = record_requirement(f"{CUSTOM_MODEL_DIR}hand-copied.pt", "letterbox", tmp_path)

    assert path == tmp_path / "hand-copied.json"
    assert requirement_for(f"{CUSTOM_MODEL_DIR}hand-copied.pt", tmp_path) == "letterbox"
    # Through the reader the panel uses, so this cannot pass on a file the listing would ignore.
    (only,) = installed_models(tmp_path)
    assert (only.resize_mode, only.auto_resolves_to) == ("letterbox", "letterbox")
    assert only.recorded is True
    # And nothing left half-written: the reader treats a corrupt record as "nothing known", so a
    # crash mid-write would silently discard the requirement just recorded.
    assert list(tmp_path.glob("*.tmp")) == []


def test_recording_merges_rather_than_replacing(tmp_path):
    """The rest of a record is evidence only the training run can supply. A whole-file write
    would delete `--val`'s measurements in order to add one field."""
    _with_record(tmp_path, validation=VALIDATION, source="snc-grocery version 2")
    record_requirement(f"{CUSTOM_MODEL_DIR}scanncart-grocery-v2.pt", "letterbox", tmp_path)

    body = json.loads((tmp_path / "scanncart-grocery-v2.json").read_text(encoding="utf-8"))
    assert body["resize_mode"] == "letterbox"
    assert body["source"] == "snc-grocery version 2"
    assert body["validation"][0]["per_class"][2]["recall"] is None


def test_recording_corrects_a_requirement_that_contradicts_the_weights(tmp_path):
    """Recording twice is not an error: the second answer replaces the first, which is the
    whole point of having the button in the panel rather than only in the training tool."""
    _with_record(tmp_path, resize_mode="letterbox")
    record_requirement(f"{CUSTOM_MODEL_DIR}scanncart-grocery-v2.pt", "stretch", tmp_path)
    assert requirement_for(f"{CUSTOM_MODEL_DIR}scanncart-grocery-v2.pt", tmp_path) == "stretch"


def test_recording_over_a_corrupt_record_lands_rather_than_failing(tmp_path):
    """An unreadable file is one the reader already treats as empty, so a write that refused it
    would leave the operator with no way to fix the file from the app."""
    _touch(tmp_path, "broken.pt")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    record_requirement(f"{CUSTOM_MODEL_DIR}broken.pt", "letterbox", tmp_path)
    assert requirement_for(f"{CUSTOM_MODEL_DIR}broken.pt", tmp_path) == "letterbox"


def test_recording_refuses_a_weight_that_is_not_on_disk(tmp_path):
    """A record describes the file beside it. Writing one for a name that is not there would
    create a requirement attached to nothing - and appear to succeed."""
    with pytest.raises(FileNotFoundError):
        record_requirement(f"{CUSTOM_MODEL_DIR}absent.pt", "letterbox", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_recording_refuses_auto(tmp_path):
    """`auto` is a lookup, not a requirement - `resolve_resize_mode` ignores a recorded `auto` -
    so recording it would look like the question had been answered while `auto` guessed on."""
    _touch(tmp_path, "hand-copied.pt")
    with pytest.raises(ValueError):
        record_requirement(f"{CUSTOM_MODEL_DIR}hand-copied.pt", "auto", tmp_path)
    assert not (tmp_path / "hand-copied.json").exists()


def test_recording_refuses_a_name_the_settings_validator_rejects(tmp_path):
    """Checked before the path is used, so the restriction that keeps `active_model` inside
    `models/` is also what keeps this write there."""
    for name in ("yolo11n.pt", "../escape.pt", "models/nested/inside.pt", "models/x.jpg"):
        with pytest.raises(ValueError):
            record_requirement(name, "letterbox", tmp_path)


def test_the_record_route_answers_with_the_requirement_in_place(monkeypatch, tmp_path):
    """What the panel's button does: one POST, and the weights it was looking at come back
    carrying a requirement - so the list it renders is the result rather than a prediction."""
    _touch(tmp_path, "hand-copied.pt")
    monkeypatch.setattr("app.models.MODELS_DIR", tmp_path)
    monkeypatch.setattr("app.main.installed_models", lambda: installed_models(tmp_path))

    r = _client().post(
        "/api/models/record",
        json={"model": f"{CUSTOM_MODEL_DIR}hand-copied.pt", "resize_mode": "letterbox"},
    )
    assert r.status_code == 200, r.text
    (only,) = r.json()["installed"]
    assert (only["resize_mode"], only["auto_resolves_to"]) == ("letterbox", "letterbox")
    assert requirement_for(f"{CUSTOM_MODEL_DIR}hand-copied.pt", tmp_path) == "letterbox"


def test_the_record_route_404s_for_a_weight_that_is_not_on_disk(monkeypatch, tmp_path):
    """Refused with a reason rather than a 500, because the operator can act on it: the panel's
    model list is a directory read, and a file can go away between the two."""
    monkeypatch.setattr("app.models.MODELS_DIR", tmp_path)
    r = _client().post(
        "/api/models/record",
        json={"model": f"{CUSTOM_MODEL_DIR}absent.pt", "resize_mode": "letterbox"},
    )
    assert r.status_code == 404
    assert "nothing was written" in r.json()["detail"].lower()


def test_the_record_route_rejects_a_stock_name_and_an_auto_requirement():
    """Both are refused before any write: a stock weight is not something the app can say
    anything about, and `auto` is not a requirement to record."""
    for body in (
        {"model": "yolo11n.pt", "resize_mode": "letterbox"},
        {"model": f"{CUSTOM_MODEL_DIR}x.pt", "resize_mode": "auto"},
        {"model": f"{CUSTOM_MODEL_DIR}x.pt", "resize_mode": "crop"},
    ):
        assert _client().post("/api/models/record", json=body).status_code == 422, body


def test_the_route_reports_what_each_installed_weight_needs(monkeypatch, tmp_path):
    """The whole point, at the wire: a model and the mode it has to be run with, plus what
    `auto` would do instead - so the renderer can compare without reimplementing the rule."""
    _with_record(tmp_path, resize_mode="stretch", source="snc-grocery version 2")
    _touch(tmp_path, "legacy.onnx")
    monkeypatch.setattr("app.main.installed_models", lambda: installed_models(tmp_path))

    installed = {m["value"]: m for m in _client().get("/api/models").json()["installed"]}
    trained = installed[f"{CUSTOM_MODEL_DIR}scanncart-grocery-v2.pt"]
    legacy = installed[f"{CUSTOM_MODEL_DIR}legacy.onnx"]

    assert trained["resize_mode"] == "stretch"
    assert trained["source"] == "snc-grocery version 2"
    # `auto` honours the record, so for these weights it answers the requirement itself -
    # there is no longer a configuration in which leaving the field alone is wrong.
    assert trained["auto_resolves_to"] == "stretch"
    # The unrecorded weight is where the format heuristic still answers, and the only case
    # an operator has to be told which way it guesses. A Roboflow .onnx guesses right.
    assert legacy["resize_mode"] is None
    assert legacy["auto_resolves_to"] == "stretch"
