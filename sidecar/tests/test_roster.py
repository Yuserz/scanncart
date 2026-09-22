"""Tests for `app/roster.py` - the app's own copy of the 8-class roster.

Two halves, and the second is why this file exists at all: the roster and the distance predicate
are a **hand-mirrored contract** between the runtime (which cannot import `sidecar/tools/`) and the
dataset tools (which own the names). The repo's standing answer to that is a drift guard, so a
disagreement fails the suite instead of turning into a model that loads and labels boxes wrongly.
"""

from __future__ import annotations

import label_classes

from app.roster import ROSTER, class_list_problems, distance_tokens_in

# The 24-class list a version generated from a distance-split project declares: every product,
# three times over. This is the shape the check exists for.
DISTANCE_SPLIT = [
    f"{name} {distance}" for name in ROSTER for distance in ("close", "mid", "far")
]


# --------------------------------------------------------------------------
# The drift guard: two copies that must agree
# --------------------------------------------------------------------------


def test_the_runtime_roster_is_the_tools_roster():
    """The names are typed twice on purpose (the runtime must not import the tools), so the one
    thing that has to hold is that they are the same names - including spelling and case, which is
    what a class list is matched on."""
    assert sorted(ROSTER) == sorted(label_classes.SLUG_TO_CLASS.values())
    assert len(ROSTER) == 8


def test_the_distance_predicate_agrees_with_the_tools():
    """Same contract, second half. The tools refuse to *build* a distance-split dataset; this
    module is what notices a weight that arrived from one anyway, and a predicate that disagreed
    with theirs would flag different names than the pipeline refused to create."""
    names = DISTANCE_SPLIT + list(ROSTER) + [
        "Farmer's Choice Fresh Milk",
        "Midfield Brand Coffee",
        "Palmolive Naturals Bar Soap 85g",
        "",
    ]
    for name in names:
        assert distance_tokens_in(name) == label_classes.distance_tokens_in(name), name
    assert label_classes.DISTANCE_TOKENS == DISTANCE_TOKENS_MIRRORED


# Imported here rather than at the top so the drift guard's two halves sit together and it is
# obvious that they are the same constant.
from app.roster import DISTANCE_TOKENS as DISTANCE_TOKENS_MIRRORED  # noqa: E402


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def test_a_clean_roster_raises_nothing():
    """The false alarm this must never raise: `--install`ed weights trained from a correctly
    generated version match the roster exactly, and warning about those would train an operator to
    ignore the warning that matters."""
    assert class_list_problems(ROSTER) == []


def test_a_distance_split_class_list_is_reported_as_the_24_output_failure():
    """The failure the whole check exists for. The sentence has to name the *cause* (a class list
    split by distance) and say that no setting fixes it - retraining is the only repair."""
    problems = class_list_problems(DISTANCE_SPLIT)

    assert len(problems) == 1
    told = problems[0]
    assert "24 of 24 class name(s) carry a distance" in told
    assert "'Bear Brand Fortified Powdered Milk 33g close'" in told
    assert "one class per product-and-distance" in told
    # The repair is named, and it is not a setting: a split class list cannot be reinterpreted
    # back into 8 classes, because the head genuinely has 24 outputs.
    assert "Retrain from a version" in told
    # And only that one finding. "8 roster classes missing" is true here - a distance-split model
    # matches no roster name - but it is the same fact the other way round, and it would bury the
    # sentence that says what to do about it.
    assert not any("cannot predict" in p for p in problems)


def test_unexpected_classes_without_distances_get_their_own_sentence():
    """A project-id mix-up and a distance split are different mistakes with different fixes, so
    the distance advice must not be printed for a class that has no distance in it."""
    problems = class_list_problems(["milo", "safeguard", "some-other-project-thing"])

    assert any("not in this app's roster" in p for p in problems)
    assert not any("carry a distance" in p for p in problems)


def test_a_model_missing_roster_classes_is_reported_even_though_nothing_it_predicts_is_wrong():
    """The quiet direction. A 6-class model passes every other check - its boxes are all correctly
    labelled - and the only symptom is two products that never appear in the log."""
    problems = class_list_problems(ROSTER[:6])

    assert len(problems) == 1
    assert "cannot predict 2 of the 8 roster classes" in problems[0]
    assert "'safeguard_pure_white_60g'" in problems[0]
    assert "will simply never be logged" in problems[0]


def test_the_findings_are_total_over_junk_input():
    """It runs on whatever a weight declares, including nothing: a detector that reports no names
    at all is a probe failure to debug, not a crash in the route."""
    assert len(class_list_problems([])) == 1  # every roster class is missing
    assert class_list_problems(()) == class_list_problems([])


# --------------------------------------------------------------------------
# The route it feeds
# --------------------------------------------------------------------------


class _Detector:
    """The shape `POST /api/detector/probe` builds: `infer` plus a `names` map."""

    def __init__(self, names: list[str]):
        self.names = {i: n for i, n in enumerate(names)}
        self.closed = False

    def infer(self, _frame):
        return []

    def close(self):
        self.closed = True


def _probe_client(names: list[str], monkeypatch):
    """The app with a detector that declares `names`.

    The native branch answers one filesystem question - is the weight on disk - before it builds
    anything, and a missing file short-circuits into "ultralytics will download it on first
    start". Narrowing that lie to the weight suffixes is the same trick `test_detector_api.py`
    uses: a blanket `exists` also answers torch about its own DLL directories while being
    imported.
    """
    import os

    from fastapi.testclient import TestClient

    from app.main import build_app
    from app.settings import Settings

    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda p: True if str(p).endswith((".pt", ".onnx")) else real_exists(p),
    )
    monkeypatch.setattr(
        "app.main.load_api_key", lambda: "test-key", raising=False
    )

    class _State:
        settings = Settings()
        device = "cpu"
        detector_factory = staticmethod(lambda settings, device: _Detector(names))

    return TestClient(build_app(lambda: _State()))


def test_the_probe_reports_the_class_problem_on_the_reachable_path(monkeypatch):
    """The transport, and why it is tested: `class_warnings` is only worth anything if it reaches
    the renderer on the *reachable* path, which is the one a working model takes."""
    client = _probe_client(DISTANCE_SPLIT, monkeypatch)

    body = client.post("/api/detector/probe").json()

    assert body["reachable"] is True
    assert len(body["class_names"]) == 24
    assert body["class_warnings"] and "carry a distance" in body["class_warnings"][0]


def test_the_probe_says_nothing_about_a_matching_class_list(monkeypatch):
    body = _probe_client(list(ROSTER), monkeypatch).post("/api/detector/probe").json()

    assert body["reachable"] is True
    assert len(body["class_names"]) == 8
    assert body["class_warnings"] == []
