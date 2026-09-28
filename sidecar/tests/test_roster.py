"""Tests for `app/roster.py` - the app's own copy of the per-generation class rosters.

Two halves, and the second is why this file exists at all: the rosters and the distance predicate
are a **hand-mirrored contract** between the runtime (which cannot import `sidecar/tools/`) and the
dataset tools (which own the names). The repo's standing answer to that is a drift guard, so a
disagreement fails the suite instead of turning into a model that loads and labels boxes wrongly.

The third thing this file pins is *which* roster a weight is judged against. That used to be a live
distinction - v1's seven against v2's eight, where the eighth was Palmolive, so the v1 weights this
app runs carried a permanent "cannot predict Palmolive" finding that no action cleared. Palmolive
was dropped (no v1 name to inherit and no close captures, so its `close` cell could never be shot),
which makes the two rosters the same seven names. The machinery stays, and these tests keep
exercising it, because the *next* added class re-opens the difference immediately and the answer has
to be right before it matters rather than after.
"""

from __future__ import annotations

import pytest

import generations
import label_classes

from app.roster import (
    NEWEST,
    ROSTERS,
    V1_ROSTER,
    V2_ROSTER,
    class_list_problems,
    distance_tokens_in,
    resolve_roster,
)

# The class list a version generated from a distance-split project declares: every product, three
# times over. This is the shape the check exists for - 7 products * 3 distances = 21 outputs.
DISTANCE_SPLIT = [
    f"{name} {distance}" for name in V2_ROSTER for distance in ("close", "mid", "far")
]


# --------------------------------------------------------------------------
# The drift guard: two copies that must agree
# --------------------------------------------------------------------------


@pytest.mark.mirror
def test_the_runtime_rosters_are_the_tools_rosters():
    """The names are typed twice on purpose (the runtime must not import the tools), so the one
    thing that has to hold is that they are the same names, generation by generation - including
    spelling and case, which is what a class list is matched on.

    Per generation rather than once: the split is the whole point of this module, so a roster that
    agreed with the tools as a *union* but differed about which generation owns a name would be
    exactly the drift that matters and invisible to a one-list guard.

    Both are seven names today, and asserted as two facts rather than one: `V2_ROSTER is V1_ROSTER`
    is the implementation (one spelling, so they cannot drift), while "v2 declares exactly v1's
    seven" is the claim - and the day a class is added the second assertion is what changes.
    """
    assert sorted(V1_ROSTER) == sorted(generations.V1.classes)
    assert sorted(V2_ROSTER) == sorted(generations.V2.classes)
    assert len(V1_ROSTER) == 7
    assert len(V2_ROSTER) == 7
    assert V2_ROSTER is V1_ROSTER
    # Nothing is exempt from v1 continuity any more: the exemption existed for the class v2 added,
    # and v2 now adds nothing. Kept as the assertion rather than deleted with the constant, because
    # a future addition has to *choose* between an exemption and a real shared name.
    assert label_classes.V2_ONLY_SLUGS == set()

    # Keyed by the generation names a record carries and the tools' `--generation` accepts, so the
    # runtime's lookup and the tools' flag cannot disagree about the vocabulary.
    assert set(ROSTERS) == {generations.V1.name, generations.V2.name}
    # And the app's fallback is the generation the project is building towards, which is the tools'
    # own default rather than a second opinion about it.
    assert NEWEST == generations.DEFAULT.name


@pytest.mark.mirror
def test_the_distance_predicate_agrees_with_the_tools():
    """Same contract, second half. The tools refuse to *build* a distance-split dataset; this
    module is what notices a weight that arrived from one anyway, and a predicate that disagreed
    with theirs would flag different names than the pipeline refused to create."""
    names = DISTANCE_SPLIT + list(V1_ROSTER) + list(V2_ROSTER) + [
        "Farmer's Choice Fresh Milk",
        "Midfield Brand Coffee",
        "Safeguard Pure White 60g",
        "",
    ]
    for name in names:
        assert distance_tokens_in(name) == label_classes.distance_tokens_in(name), name
    assert label_classes.DISTANCE_TOKENS == DISTANCE_TOKENS_MIRRORED


# Imported here rather than at the top so the drift guard's two halves sit together and it is
# obvious that they are the same constant.
from app.roster import DISTANCE_TOKENS as DISTANCE_TOKENS_MIRRORED  # noqa: E402


# --------------------------------------------------------------------------
# Which roster a weight is judged against
# --------------------------------------------------------------------------


def test_each_generations_own_roster_is_clean():
    """The false alarm this must never raise, in both directions.

    v1's is the one that mattered: it is the class list of the weights this app runs, so a finding
    here is a warning on every capture that no setting and no retrain could clear. v2's is the same
    claim for the generation the datasets are being built for - and `--install`ed v2 weights must
    not be told they are missing a class either.

    Note what makes this cheap now and did not before: the two lists are identical, so a v1 head
    cannot be judged against a roster that expects a class it never had.
    """
    assert class_list_problems(V1_ROSTER) == []
    assert class_list_problems(V2_ROSTER) == []


def test_the_models_own_names_pick_the_roster_when_nothing_is_recorded():
    """No record, only what the model declares - which is all a probe on a hand-copied weight ever
    has. An exact class list *is* the evidence, with no code change when a later generation lands.

    While the two rosters are the same names, an exact match names the newest - and the honest
    thing to assert is not which label comes back but that **no verdict depends on it**: the
    sentences below differ only in the word "v1"/"v2", and the names they list are the same. That
    is the property that makes the tie harmless, and it stops being true the moment a class is
    added - which is when this test should fail and be rewritten rather than quietly pass.
    """
    first = resolve_roster(V1_ROSTER)[0]
    assert first == NEWEST
    assert resolve_roster(V2_ROSTER) == (first, ROSTERS[first])
    # A class list is a set of names, so the order a head happens to index them in cannot matter.
    assert resolve_roster(list(reversed(V2_ROSTER)))[0] == first

    short = V1_ROSTER[:5]
    assert class_list_problems(short) == class_list_problems(short, "v2")


def test_a_recorded_generation_decides_even_when_the_names_cannot():
    """The rule is the record first, and that has to hold before it is needed rather than after.

    It was live while v2 declared eight names: a v2 head that had lost a class and a complete v1
    head carried the same names and wanted opposite readings, and only the training run knew which
    it was. Today the two rosters agree, so this is a no-op in *outcome* - which is exactly why it
    is asserted now, so the mechanism is proven to work on the day a class makes it matter again.
    A generation this app has no roster for is not evidence; it is a file written by a later tool,
    and the names are then the only thing left to judge.
    """
    assert resolve_roster(V1_ROSTER, "v2") == ("v2", V2_ROSTER)
    assert resolve_roster(V1_ROSTER, "v1") == ("v1", V1_ROSTER)
    assert resolve_roster(V2_ROSTER, "v1") == ("v1", V1_ROSTER)  # the record is the authority
    assert resolve_roster(V1_ROSTER, "v9") == (NEWEST, V2_ROSTER)  # unknown -> fall back to names


def test_a_partial_list_is_held_to_the_generation_it_is_closest_to():
    """A head that lost a class, or one carrying a label from another project, is still recognisably
    one generation's roster: fewest names outside it first, then fewest of its own names missing.

    With one roster to be close to, every partial list resolves to it - and the count in the
    sentence is what carries the meaning, not the generation word. A stray name from another
    project is still told apart from a missing one; that split is asserted in the verdict tests.
    """
    assert resolve_roster(V1_ROSTER[:6]) == (NEWEST, V2_ROSTER)
    assert resolve_roster(list(V2_ROSTER) + ["some-other-project-thing"])[0] == NEWEST
    assert resolve_roster(list(V1_ROSTER) + ["some-other-project-thing"])[0] == NEWEST


def test_a_list_nobody_can_place_is_held_to_the_newest_roster():
    """Nothing to be close to, so the app's current expectation is the only yardstick. Distinct
    from the empty case only in emphasis - both are "unknown", and the honest answer for both is
    the roster the project is building towards. Callers gate on the empty list; this is what
    happens if one does not."""
    assert resolve_roster([]) == (NEWEST, V2_ROSTER)
    assert resolve_roster([], "v1") == ("v1", V1_ROSTER)  # a record still beats a blank list


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def test_a_distance_split_class_list_is_reported_as_the_many_output_failure():
    """The failure the whole check exists for. The sentence has to name the *cause* (a class list
    split by distance) and say that no setting fixes it - retraining is the only repair.

    The count is read off the roster rather than typed: seven products at three distances is 21
    outputs, and it was 24 with the eighth product, so a literal here would have to be edited
    alongside every roster change - which is the drift this module is about.
    """
    problems = class_list_problems(DISTANCE_SPLIT)
    expected = len(V2_ROSTER) * 3

    assert len(problems) == 1
    told = problems[0]
    assert f"{expected} of {expected} class name(s) carry a distance" in told
    assert f"'{V2_ROSTER[0]} close'" in told
    assert "one class per product-and-distance" in told
    # The repair is named, and it is not a setting: a split class list cannot be reinterpreted back
    # into 7 classes, because the head genuinely has one output per product-and-distance.
    assert "Retrain from a version" in told
    # And only that one finding. "7 roster classes missing" is true here - a distance-split model
    # matches no roster name - but it is the same fact the other way round, and it would bury the
    # sentence that says what to do about it.
    assert not any("cannot predict" in p for p in problems)


def test_unexpected_classes_without_distances_get_their_own_sentence():
    """A project-id mix-up and a distance split are different mistakes with different fixes, so
    the distance advice must not be printed for a class that has no distance in it. The sentence
    names the roster it was judged against, since "the roster" is not one list in general."""
    problems = class_list_problems(["milo", "safeguard", "some-other-project-thing"])

    assert any("roster" in p and "some-other-project-thing" in p for p in problems)
    assert not any("carry a distance" in p for p in problems)


def test_a_model_missing_roster_classes_is_reported_even_though_nothing_it_predicts_is_wrong():
    """The quiet direction: a head that cannot predict two of its classes passes every other check
    - its boxes are all correctly labelled - and the only symptom is those products never appearing
    in the log. The count is the *roster's*, which is why a hardcoded number would be wrong the
    moment the roster changes.
    """
    problems = class_list_problems(V2_ROSTER[:5], "v2")

    assert len(problems) == 1
    assert f"cannot predict 2 of the {len(V2_ROSTER)} v2 roster classes" in problems[0]
    assert f"'{V2_ROSTER[5]}'" in problems[0]
    assert "will simply never be logged" in problems[0]


def test_a_v1_head_missing_a_class_is_measured_against_the_full_roster():
    """The same finding on the generation the app runs, and the reason the count is read off the
    roster: the sentence names whichever classes are absent rather than a remembered one."""
    problems = class_list_problems(V1_ROSTER[:6])

    assert len(problems) == 1
    assert f"cannot predict 1 of the {len(V1_ROSTER)}" in problems[0]
    assert f"'{V1_ROSTER[6]}'" in problems[0]


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
    assert len(body["class_names"]) == len(V2_ROSTER) * 3
    assert body["class_warnings"] and "carry a distance" in body["class_warnings"][0]


def test_the_probe_says_nothing_about_a_matching_class_list(monkeypatch):
    body = _probe_client(list(V2_ROSTER), monkeypatch).post("/api/detector/probe").json()

    assert body["reachable"] is True
    assert len(body["class_names"]) == len(V2_ROSTER)
    assert body["class_warnings"] == []


def test_the_probe_says_nothing_about_the_v1_weights_this_app_runs(monkeypatch):
    """The case the roster split exists for, through the route rather than the function: the
    installed v1 weight declares seven names, and the probe - which has only the loaded model's
    vocabulary to go on, since a stock weight may have no record at all - must recognise them as
    its own rather than report a missing class. Pressing Test connection on the shipping model used
    to produce that finding, and still would if the rosters were ever declared separately again."""
    body = _probe_client(list(V1_ROSTER), monkeypatch).post("/api/detector/probe").json()

    assert body["reachable"] is True
    assert len(body["class_names"]) == 7
    assert body["class_warnings"] == []
