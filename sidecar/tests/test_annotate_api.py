"""Tests for `annotate/app.py` - the loop, its two refusals, and the one thing it never writes.

The annotator is an authoring tool with an HTTP surface, so this file is written the way the
sidecar's own app tests are: `TestClient` over `build_annotate_app(state)`, fakes injected at the
two seams the app already has (`detector_factory` for the local weight, `hosted_post` for the
network), and everything on disk under `tmp_path`.

Three contracts are asserted here rather than anywhere else, because this is the layer that could
break them:

* **Suggest is not save.** `POST /suggest` records a proposal in `provenance.suggestion` and writes
  no label file, so a frame nobody looked at is still outstanding work. `machine_only` is then a
  fact about the files (the saved rows compared against the recorded suggestion) instead of a claim
  about what a user clicked.
* **The two refusals are 409s, not 500s.** A box on a hard negative and a null on a product frame
  are the two annotations that teach the model something false; the store refuses them and the
  route reports that as a client error.
* **A name from the URL never leaves the staged set.** The image route is the one place a query
  parameter reaches the filesystem.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from annotate.app import AnnotateState, Weight, build_annotate_app, choose_weights, encode_for_hosted
from annotate.providers import HOSTED_PROVIDERS
from annotate.store import CLASS_NAMES, CLASS_SLUGS, Box, LabelStore


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _jpeg(path: Path, width: int = 48, height: int = 32) -> None:
    """A real JPEG, because `read_frame` decodes with OpenCV - a byte-string placeholder would make
    the suggest route's own behavior untestable."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def _staged(tmp_path: Path) -> LabelStore:
    """The staged shapes `clean_v2.py` writes: one product cell, one `far` frame, and the negatives
    in a set of their own."""
    out = tmp_path / "cleaned-v2"
    entries = [
        {"new_name": "milo_0001.jpg", "class": "milo", "distance": "mid", "session": "s2"},
        {"new_name": "milo_0002.jpg", "class": "milo", "distance": "mid", "session": "s2"},
        {"new_name": "safeguard_0001.jpg", "class": "safeguard", "distance": "far", "session": "s2"},
    ]
    for entry in entries:
        _jpeg(out / entry["class"] / entry["new_name"])
    (out / "manifest.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")

    negatives = tmp_path / "cleaned-negatives"
    (negatives / "negative").mkdir(parents=True)
    _jpeg(negatives / "negative" / "negative_0001.jpg")
    (negatives / "manifest.json").write_text(
        json.dumps([{"new_name": "negative_0001.jpg", "class": "negative", "distance": ""}]),
        encoding="utf-8",
    )
    return LabelStore(out=out, annotations=tmp_path / "annotations-v2", extras=[negatives])


def _weight(name: str = "scanncart-grocery-v2.pt") -> Weight:
    return Weight(
        name=name,
        path=Path("models") / name,
        resize_mode="stretch",
        class_names=tuple(CLASS_NAMES),
    )


class FakeDetector:
    def __init__(self, detections):
        self.detections = list(detections)

    def infer(self, frame):
        return list(self.detections)


def _factory(detections=()):
    def build(weights, device, conf, imgsz, resize_mode="letterbox"):
        return FakeDetector(detections)

    return build


def _detection(cls, x1=0.25, y1=0.25, x2=0.75, y2=0.75):
    return SimpleNamespace(cls=cls, conf=0.9, box=(x1, y1, x2, y2))


def _client(tmp_path: Path, **kwargs) -> tuple[TestClient, AnnotateState]:
    store = _staged(tmp_path)
    state = AnnotateState(store=store, weight=kwargs.pop("weight", _weight()), **kwargs)
    return TestClient(build_annotate_app(state)), state


def _env_with(name: str) -> object:
    return lambda key: "sk-secret" if key == name else ""


# --------------------------------------------------------------------------
# Config, worklist, frame
# --------------------------------------------------------------------------


def test_config_names_the_classes_the_weight_and_which_providers_have_keys(tmp_path):
    """Everything the page needs to draw itself, and nothing it would have to guess: the class
    order the label rows index, what the suggestions will come from, and which second opinions this
    machine could offer at all."""
    client, _ = _client(
        tmp_path, hosted_name="openai", hosted_post=lambda *a: (200, {}), env=_env_with("OPENAI_API_KEY")
    )

    body = client.get("/api/config").json()

    assert [c["slug"] for c in body["classes"]] == list(CLASS_SLUGS)
    assert [c["name"] for c in body["classes"]] == list(CLASS_NAMES)
    assert [c["index"] for c in body["classes"]] == list(range(len(CLASS_NAMES)))
    assert [c["pseudo"] for c in body["classes"]] == [slug == "negative" for slug in CLASS_SLUGS]
    assert body["weight"]["name"] == "scanncart-grocery-v2.pt"
    assert body["weight"]["resize_mode"] == "stretch"
    assert body["hosted"]["name"] == "openai"
    assert body["hosted"]["available"] == ["openai"]
    assert body["hosted"]["keys"]["openai"] == HOSTED_PROVIDERS["openai"]["env"]


def test_the_worklist_is_the_manifest_read_filtered_and_limited(tmp_path):
    """The manifest is the worklist: a frame parked by `clean_v2 drop` is not in it and so is not
    work, and the order puts the outstanding `far` frames - the axis this dataset exists for -
    first."""
    client, _ = _client(tmp_path)

    body = client.get("/api/frames").json()

    assert body["counts"]["total"] == 4
    assert body["counts"]["unlabeled"] == 4
    assert body["counts"]["loaded"] == 4
    assert [f["name"] for f in body["frames"]] == [
        "safeguard_0001.jpg",  # outstanding, far
        "milo_0001.jpg",
        "milo_0002.jpg",
        "negative_0001.jpg",  # no distance: last of the outstanding frames
    ]
    assert body["frames"][0]["cell"] == "safeguard|far"

    filtered = client.get("/api/frames", params={"state": "unlabeled", "limit": 2}).json()
    assert [f["name"] for f in filtered["frames"]] == ["safeguard_0001.jpg", "milo_0001.jpg"]
    assert filtered["counts"]["loaded"] == 2

    cell = client.get("/api/frames", params={"cell": "safeguard|far"}).json()
    assert [f["name"] for f in cell["frames"]] == ["safeguard_0001.jpg"]


def test_the_split_and_distance_filters_narrow_the_worklist_by_the_stores_own_fields(tmp_path):
    """The two facts `clean_v2` and `plan_split` leave beside the frames, narrowable server-side:
    the page sends tokens and never decides which frames match, so a filtered list here and the
    checklist's sections cannot mean two different things.

    `unknown` is the store's own spelling for a value that was never recorded (`Frame.cell` uses it
    for a hard negative's distance). It has to be a *token* rather than an empty parameter, because
    an empty one means "do not filter" - without it the negatives, and every frame no plan has
    reached, would be impossible to ask for.
    """
    client, state = _client(tmp_path)
    (state.store.out / "splits.json").write_text(
        json.dumps(
            {
                "milo_0001.jpg": "train",
                "milo_0002.jpg": "test",
                "safeguard_0001.jpg": "valid",
            }
        ),
        encoding="utf-8",
    )

    # A comma list is a set of values: the two splits the acceptance gate asks about, in one request,
    # in the worklist's own order (`far` first), not in the order they were asked for.
    gate = client.get("/api/frames", params={"split": "test,valid"}).json()
    assert [f["name"] for f in gate["frames"]] == ["safeguard_0001.jpg", "milo_0002.jpg"]
    assert [f["split"] for f in gate["frames"]] == ["valid", "test"]

    only_test = client.get("/api/frames", params={"split": "test"}).json()
    assert [f["name"] for f in only_test["frames"]] == ["milo_0002.jpg"]
    far = client.get("/api/frames", params={"distance": "far"}).json()
    assert [f["name"] for f in far["frames"]] == ["safeguard_0001.jpg"]
    # The negative has neither a distance nor a split entry, and is reachable by either axis.
    unplaced = client.get("/api/frames", params={"split": "unknown"}).json()
    assert [f["name"] for f in unplaced["frames"]] == ["negative_0001.jpg"]
    both = client.get("/api/frames", params={"distance": "unknown", "state": "unlabeled"}).json()
    assert [f["name"] for f in both["frames"]] == ["negative_0001.jpg"]

    # The counts stay the *whole* set's: a header recalculated from a filtered list would report
    # 100% done the moment a pass started. `loaded` is the one number that describes the filter.
    assert both["counts"]["total"] == 4 and both["counts"]["loaded"] == 1


def test_the_draw_view_is_the_stores_draw_list_and_a_filter_narrows_it(tmp_path):
    """`?draw=1` is `store.draw_worklist` - the same call `make human-pass` renders as the
    checklist's draw section - rather than a predicate spelled out in the route.

    The suggestion has to be *recorded* for a frame to be in it: the section's claim is that the
    weight was asked and found nothing, which is evidence about the model, while a frame nobody
    asked about is outstanding work of a different kind.
    """
    client, state = _client(tmp_path)
    (state.store.out / "splits.json").write_text(
        json.dumps({"safeguard_0001.jpg": "valid", "milo_0002.jpg": "valid"}),
        encoding="utf-8",
    )
    state.store.record_suggestion("safeguard_0001.jpg", [], "local:v1")  # asked, found nothing
    state.store.record_suggestion("milo_0002.jpg", [Box(0, 0.5, 0.5, 0.2, 0.2)], "local:v1")

    body = client.get("/api/frames", params={"draw": 1}).json()

    assert body["draw"] is True
    assert [f["name"] for f in body["frames"]] == ["safeguard_0001.jpg"]
    # `milo_0002` is `mid`, and the weight proposed a box for it - two reasons it is not draw work.
    # Nothing is saved, so nothing is machine-only either: the two views are disjoint here.
    assert client.get("/api/frames", params={"review": 1}).json()["frames"] == []

    # A view selects and a filter narrows it: nothing here replaces the ordinary worklist.
    assert len(client.get("/api/frames").json()["frames"]) == 4
    kept = client.get("/api/frames", params={"draw": 1, "distance": "far"}).json()
    assert [f["name"] for f in kept["frames"]] == ["safeguard_0001.jpg"]
    empty = client.get("/api/frames", params={"draw": 1, "distance": "mid"}).json()
    assert empty["frames"] == [] and empty["counts"]["loaded"] == 0


def test_the_nulls_view_is_the_stores_null_list_of_undecided_negatives(tmp_path):
    """`?nulls=1` is `store.null_worklist` - the same call `make human-pass` renders as the
    checklist's null section - so the 15 frames that section lists are one click in the page.

    Two things scope it, and both are the store's own rule rather than a predicate here: the gate
    splits (a negative in `train` costs the acceptance number nothing) and `state == "unlabeled"` -
    an empty label file *is* the decision that there is no item, so a marked frame is finished work,
    not still outstanding.
    """
    client, state = _client(tmp_path)

    def nulls_view(params: dict | None = None) -> dict:
        return client.get("/api/frames", params={"nulls": 1, **(params or {})}).json()

    for split in ("train", ""):
        (state.store.out / "splits.json").write_text(
            json.dumps({"negative_0001.jpg": split}), encoding="utf-8"
        )
        assert nulls_view()["frames"] == []

    (state.store.out / "splits.json").write_text(
        json.dumps({"negative_0001.jpg": "valid"}), encoding="utf-8"
    )
    body = nulls_view()
    assert body["nulls"] is True
    # The views are exclusive: this list is not the review one, and the flags say which is on.
    assert body["review"] is False and body["draw"] is False
    assert [f["name"] for f in body["frames"]] == ["negative_0001.jpg"]
    assert body["frames"][0]["cell"] == "negative|unknown"
    # A filter narrows it like any other list - the negatives carry no distance, so they are the
    # `unknown` bucket, and they are the whole of it.
    assert [f["name"] for f in nulls_view({"distance": "unknown", "state": "unlabeled"})["frames"]] == [
        "negative_0001.jpg"
    ]

    state.store.write("negative_0001.jpg", [], null=True)
    assert nulls_view()["frames"] == []
    # And the ordinary worklist is untouched: nothing here selects *instead* of it.
    assert len(client.get("/api/frames").json()["frames"]) == 4


def test_the_worklist_reports_the_acceptance_gate_live_and_clears_it_as_the_pass_is_worked(tmp_path):
    """The finish line, on screen: how many machine-only decisions remain in the two splits a run
    refuses on, and what the review count falls to once they are clean.

    It is asserted against the *document's* number (`human_pass.Checklist.gate`) rather than against
    a figure typed here: the page and the checklist answering "is the gate clear" differently is the
    failure this readout could introduce, and one call (`store.gate_frames`) is what prevents it.
    It is reported in every view, not only in review mode - the far frames get drawn first.
    """
    from annotate import human_pass

    client, state = _client(tmp_path)
    (state.store.out / "splits.json").write_text(
        json.dumps(
            {
                "milo_0001.jpg": "train",
                "milo_0002.jpg": "test",
                "safeguard_0001.jpg": "valid",
            }
        ),
        encoding="utf-8",
    )
    box = [Box(0, 0.5, 0.5, 0.2, 0.2)]
    for name in ("milo_0001.jpg", "milo_0002.jpg", "safeguard_0001.jpg"):
        state.store.record_suggestion(name, box, "local:v1")
        state.store.write(name, box)  # saved unchanged: a machine's, unread

    body = client.get("/api/frames").json()

    assert body["gate"] == {
        "splits": ["test", "valid"],
        "pending": 2,
        "remaining": 1,
        "by_split": {"test": 1, "valid": 1},
        "clear": False,
    }
    items = human_pass.checklist(
        state.store.frames(),
        state.store.splits(),
        state.store.provenance(),
        out=state.store.out,
        annotations=state.store.annotations,
        extras=tuple(state.store.extras),
    )
    assert items.gate == body["gate"]["pending"] == 2
    assert items.remaining == body["gate"]["remaining"] == 1

    # The chip's jump is this request: the review pass narrowed to the pair - and it has to serve
    # exactly the frames the chip counted, or clicking a number would land somewhere else.
    jumped = client.get("/api/frames", params={"review": 1, "split": ",".join(body["gate"]["splits"])}).json()
    assert [f["name"] for f in jumped["frames"]] == ["milo_0002.jpg", "safeguard_0001.jpg"]
    assert len(jumped["frames"]) == body["gate"]["pending"] == 2

    # Confirming the two in the gate splits clears it; the `train` one is tolerated and only counted.
    for name in ("milo_0002.jpg", "safeguard_0001.jpg"):
        state.store.write(name, box, confirm=True)
    after = client.get("/api/frames", params={"draw": 1}).json()["gate"]
    assert after["clear"] is True and after["pending"] == 0 and after["remaining"] == 1
    assert after["by_split"] == {"test": 0, "valid": 0}


def test_config_offers_the_filters_with_the_tokens_the_query_takes(tmp_path):
    """The options are built server-side so the store's spellings - including `unknown` for a value
    that was never recorded - reach the page instead of being retyped in JavaScript, and so a
    filter can be widened back: options computed from the *loaded* frames would lose the values the
    current filter filtered away."""
    client, _ = _client(tmp_path)

    body = client.get("/api/config").json()

    # The gate pair first, as one entry: that is how the pass asks about it (the parameter takes a
    # comma list) and what the header's gate chip selects, so the chip is not a query of its own.
    assert body["splits"][0] == {"value": "test,valid", "label": "test/valid (the gate)"}
    assert [entry["value"] for entry in body["splits"]] == [
        "test,valid",
        "test",
        "valid",
        "train",
        "unknown",
    ]
    assert body["splits"][-1]["label"] == "unassigned"
    assert [entry["value"] for entry in body["distances"]] == ["far", "mid", "close", "unknown"]
    assert body["distances"][-1]["label"] == "unrecorded"
    # Then the two individually, which is also the order the pass works them - so the dropdown is
    # the worklist's priority, not an alphabet.
    assert body["splits"][1:3] == [{"value": "test", "label": "test"}, {"value": "valid", "label": "valid"}]


def test_a_frame_carries_its_boxes_its_state_and_its_provenance(tmp_path):
    """The overlay is positioned from the percentages the server computed, so the wire shape is
    asserted: a UI that re-derived `left` from `cx - w/2` would be the second copy of the same
    arithmetic this repo keeps drift guards for."""
    client, state = _client(tmp_path)
    state.store.record_suggestion("milo_0001.jpg", [Box(0, 0.5, 0.5, 0.4, 0.2)], "local:v2")
    state.store.write("milo_0001.jpg", [Box(cls=1, cx=0.5, cy=0.5, w=0.4, h=0.2)])

    body = client.get("/api/frame/milo_0001.jpg").json()

    assert body["frame"]["state"] == "labeled"
    assert body["frame"]["boxes"] == 1
    assert body["frame"]["url"] == "/api/frame/milo_0001.jpg/image"
    assert body["boxes"] == [
        {
            "cls": 1,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.4,
            "h": 0.2,
            "left": 30.0,
            "top": 40.0,
            "width": 40.0,
            "height": 20.0,
        }
    ]
    # Kept apart: what is on disk versus what was proposed. Showing the suggestion as provenance
    # is how a reviewer ends up unable to tell the two questions apart.
    assert "suggestion" not in body["provenance"]
    assert body["suggestion"]["provider"] == "local:v2"


# --------------------------------------------------------------------------
# Suggest writes a proposal and no label
# --------------------------------------------------------------------------


def test_suggest_records_a_proposal_and_writes_no_label_file(tmp_path):
    """A frame that has only been suggested is still outstanding work - and that is what makes
    `machine_only` (decided later, at save, against these recorded rows) a fact about the files."""
    client, state = _client(tmp_path, detector_factory=_factory([_detection(CLASS_NAMES[4])]))

    body = client.post("/api/frame/milo_0001.jpg/suggest").json()

    assert body["provider"] == "local:scanncart-grocery-v2"
    assert [box["cls"] for box in body["boxes"]] == [4]
    assert state.store.state_of("milo_0001.jpg") == "unlabeled"
    assert state.store.read_boxes("milo_0001.jpg") == []
    recorded = state.store.provenance()["milo_0001.jpg"]["suggestion"]["provider"]
    assert recorded == "local:scanncart-grocery-v2"


def test_saving_the_suggestion_unchanged_is_machine_only_and_editing_it_is_not(tmp_path):
    """The whole point of the split, end to end through the two routes and the summary the panel
    reads: 100% decided must not be able to mean 500 frames of unread machine boxes."""
    client, _ = _client(tmp_path, detector_factory=_factory([_detection(CLASS_NAMES[0])]))

    suggested = client.post("/api/frame/milo_0001.jpg/suggest").json()["boxes"]
    saved = client.post(
        "/api/frame/milo_0001.jpg/labels",
        json={"boxes": [{k: box[k] for k in ("cls", "cx", "cy", "w", "h")} for box in suggested]},
    )
    assert saved.status_code == 200
    assert saved.json()["provenance"]["machine_only"] is True

    summary = client.get("/api/summary").json()
    assert (summary["total"], summary["decided"], summary["pseudo"], summary["reviewed"]) == (4, 1, 1, 0)
    assert summary["pseudo_by_cell"] == {"milo|mid": 1}

    edited = dict(suggested[0])
    edited["cx"] = 0.51
    again = client.post(
        "/api/frame/milo_0001.jpg/labels",
        json={"boxes": [{k: edited[k] for k in ("cls", "cx", "cy", "w", "h")}]},
    )
    assert again.json()["provenance"]["machine_only"] is False
    assert client.get("/api/summary").json()["reviewed"] == 1


def test_confirming_a_suggestion_through_the_route_clears_it_from_the_review_count(tmp_path):
    """The review pass has to be able to *finish*: agreeing with a weight's boxes leaves the rows
    unchanged, so without an explicit confirmation the frame stays machine-only and the gate (zero
    in valid/test) can never be satisfied - the pass would keep offering work it cannot clear."""
    client, state = _client(tmp_path)
    box = {"cls": 2, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    state.store.record_suggestion("milo_0001.jpg", [Box(2, 0.5, 0.5, 0.2, 0.2)], "local:v1")
    saved = client.post("/api/frame/milo_0001.jpg/labels", json={"boxes": [box]})
    assert saved.status_code == 200
    assert saved.json()["provenance"]["machine_only"] is True
    assert client.get("/api/summary").json()["pseudo"] == 1

    confirmed = client.post(
        "/api/frame/milo_0001.jpg/labels", json={"boxes": [box], "confirmed": True}
    )

    assert confirmed.status_code == 200
    assert confirmed.json()["provenance"]["machine_only"] is False
    assert confirmed.json()["frame"]["machine_only"] is False
    # The number the panel drives to zero, and the list the review pass walks, both follow.
    assert client.get("/api/summary").json()["pseudo"] == 0
    assert client.get("/api/frames", params={"review": 1}).json()["frames"] == []

    # A hand-drawn frame can be confirmed too, and it changes nothing: it was never machine-only.
    hand = client.post("/api/frame/milo_0002.jpg/labels", json={"boxes": [box], "confirmed": True})
    assert hand.json()["provenance"]["machine_only"] is False
    assert hand.json()["provenance"]["provider"] == "unknown"


def test_the_review_pass_serves_only_unreviewed_machine_work_worst_split_first(tmp_path):
    """`?review=1` is the second pass over the same set, and the ordering is the whole point: the
    gate is "no machine-only decisions in `valid` or `test`", so a pass that walked the worklist in
    labeling order would spend its first hour on `train` boxes that cost nothing.

    The counts stay the *whole* set's, because a header that recomputed itself from the filtered
    list would report 100% done the moment a review pass started.
    """
    client, state = _client(tmp_path)
    (state.store.out / "splits.json").write_text(
        json.dumps(
            {
                "milo_0001.jpg": "train",
                "milo_0002.jpg": "test",
                "safeguard_0001.jpg": "valid",
            }
        ),
        encoding="utf-8",
    )
    suggested = [Box(0, 0.5, 0.5, 0.2, 0.2)]
    for name in ("safeguard_0001.jpg", "milo_0002.jpg"):
        state.store.record_suggestion(name, suggested, "local:v1")
        state.store.write(name, suggested)  # saved unchanged: still a machine's, unreviewed
    state.store.write("milo_0001.jpg", suggested)  # drawn here: not review work

    body = client.get("/api/frames", params={"review": 1}).json()

    assert body["review"] is True
    assert [f["name"] for f in body["frames"]] == ["milo_0002.jpg", "safeguard_0001.jpg"]
    assert all(f["machine_only"] for f in body["frames"])
    # The header still counts the whole set, and `pseudo` is the number the pass drives to zero.
    assert body["counts"]["total"] == 4 and body["counts"]["pseudo"] == 2

    # And the ordinary worklist is unchanged: this flag selects, it does not replace.
    plain = client.get("/api/frames").json()
    assert len(plain["frames"]) == 4 and plain["review"] is False


def test_suggest_with_no_weight_configured_is_a_note_not_an_error(tmp_path):
    """A machine with no installed weight still has to be able to open frames; the page shows the
    reason rather than an exception."""
    client, _ = _client(tmp_path, weight=None, weight_note="no weights in models/")

    body = client.post("/api/frame/milo_0001.jpg/suggest").json()

    assert body["provider"] == "none"
    assert body["note"] == "no weights in models/"
    assert body["boxes"] == []


def test_the_hosted_provider_is_asked_where_local_found_nothing_and_capped_after_one_call(tmp_path):
    """Local first is a cost rule, so it is counted rather than trusted: with a cap of one, the
    second frame local could not help with is proposed by local (nothing) instead of spending
    another call - and the response says the cap is what stopped it."""
    posted: list[str] = []

    def post(url, headers, payload, timeout):
        posted.append(url)
        answer = json.dumps({"objects": [{"class": CLASS_NAMES[2], "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]})
        return 200, {"choices": [{"message": {"content": answer}}]}

    client, state = _client(
        tmp_path,
        detector_factory=_factory(()),
        hosted_name="openai",
        hosted_post=post,
        max_hosted_calls=1,
        env=_env_with("OPENAI_API_KEY"),
    )

    first = client.post("/api/frame/milo_0001.jpg/suggest").json()
    assert first["provider"] == "hosted:openai"
    assert first["note"].startswith("local found nothing;")
    assert first["hosted_capped"] is True
    assert len(posted) == 1

    second = client.post("/api/frame/milo_0002.jpg/suggest").json()
    assert second["provider"].startswith("local:")
    assert second["note"].endswith("no hosted provider is configured")
    assert second["hosted_capped"] is True
    assert len(posted) == 1
    assert state.hosted_calls == 1

    # And the second frame's suggestion was still recorded - a capped run is a degraded run, not a
    # broken one.
    assert state.store.provenance()["milo_0002.jpg"]["suggestion"]["provider"].startswith("local:")


def test_a_hosted_suggestion_reaches_the_live_transport_with_no_post_seam(tmp_path, monkeypatch):
    """The production wiring, which no other test here exercises: `annotate/run.py` builds the state
    without a `post`, so the route has to fall through to the real network. Injection is for tests -
    a default of `None` made the one frame the cap exists to spend a 500 instead of a suggestion."""
    import annotate.providers as providers

    calls: list[str] = []

    def fake(url, headers, payload, timeout):
        calls.append(url)
        answer = json.dumps(
            {"objects": [{"class": CLASS_NAMES[2], "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]}
        )
        return 200, {"choices": [{"message": {"content": answer}}]}

    monkeypatch.setattr(providers, "http_post", fake)
    client, state = _client(
        tmp_path,
        detector_factory=_factory(()),
        hosted_name="openai",
        env=_env_with("OPENAI_API_KEY"),
    )

    body = client.post("/api/frame/milo_0001.jpg/suggest").json()

    assert body["provider"] == "hosted:openai"
    assert [box["cls"] for box in body["boxes"]] == [2]
    assert body["note"].startswith("local found nothing;")
    assert state.hosted_calls == 1
    assert calls == [HOSTED_PROVIDERS["openai"]["url"]]


def test_a_hosted_failure_does_not_lose_the_frame(tmp_path):
    """The local proposal is already in hand when the hosted call fails, so a bad key or a rate
    limit must surface as a 5xx for *that* frame - the worklist survives it."""
    def post(url, headers, payload, timeout):
        return 401, {"error": "bad key sk-secret"}

    client, _ = _client(
        tmp_path,
        detector_factory=_factory(()),
        hosted_name="openai",
        hosted_post=post,
        env=_env_with("OPENAI_API_KEY"),
    )

    with pytest.raises(RuntimeError):
        client.post("/api/frame/milo_0001.jpg/suggest")

    assert [f["name"] for f in client.get("/api/frames").json()["frames"]]


# --------------------------------------------------------------------------
# The two refusals, and a malformed payload
# --------------------------------------------------------------------------


def test_a_null_on_a_hard_negative_is_saved_and_a_box_on_it_is_a_409(tmp_path):
    """The hard negatives are the only frames that can be null-annotated, and they are the only
    frames a box is refused on. Both directions train the model wrong."""
    client, state = _client(tmp_path)

    saved = client.post("/api/frame/negative_0001.jpg/labels", json={"boxes": [], "null": True})
    assert saved.status_code == 200
    assert saved.json()["provenance"]["state"] == "null"
    # Not attributed and not machine-drawn: the page sends no provider, and a null is a human's
    # assertion whatever a suggestion happened to say.
    assert saved.json()["provenance"]["provider"] == "unknown"
    assert saved.json()["provenance"]["machine_only"] is False
    assert state.store.state_of("negative_0001.jpg") == "null"

    refused = client.post(
        "/api/frame/negative_0001.jpg/labels",
        json={"boxes": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]},
    )
    assert refused.status_code == 409
    assert "carries no box by definition" in refused.json()["detail"]


def test_a_null_on_a_product_frame_is_a_409(tmp_path):
    """An empty label on a product frame teaches the head that the product is not in the picture -
    the one annotation mistake that actively trains the model wrong."""
    client, state = _client(tmp_path)

    refused = client.post("/api/frame/milo_0001.jpg/labels", json={"boxes": [], "null": True})

    assert refused.status_code == 409
    assert "not a hard negative" in refused.json()["detail"]
    assert state.store.state_of("milo_0001.jpg") == "unlabeled"


def test_an_out_of_frame_box_is_a_409_and_a_malformed_payload_is_a_422(tmp_path):
    """Two different client errors, and the distinction is worth keeping: 409 is "that annotation
    is wrong for this frame" (the operator can act on it), 422 is "that is not a box at all"."""
    client, state = _client(tmp_path)

    outside = client.post(
        "/api/frame/milo_0001.jpg/labels",
        json={"boxes": [{"cls": 0, "cx": 1.4, "cy": 0.5, "w": 0.2, "h": 0.2}]},
    )
    assert outside.status_code == 409
    assert "out-of-frame" in outside.json()["detail"]

    assert client.post("/api/frame/milo_0001.jpg/labels", json={"boxes": "0 0.5 0.5 0.2 0.2"}).status_code == 422
    assert client.post("/api/frame/milo_0001.jpg/labels", json={"boxes": [{"cls": 0}]}).status_code == 422
    assert state.store.state_of("milo_0001.jpg") == "unlabeled"


def test_a_class_index_outside_the_roster_is_a_409_rather_than_a_silent_row(tmp_path):
    """`cls` is a number, so a bad one writes a label that means a different product - refused at
    the same boundary as the other two, because the operator can fix it."""
    client, _ = _client(tmp_path)

    refused = client.post(
        "/api/frame/milo_0001.jpg/labels",
        json={"boxes": [{"cls": len(CLASS_NAMES), "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]},
    )

    assert refused.status_code == 409


# --------------------------------------------------------------------------
# Paths: the image route is the one place a URL reaches the filesystem
# --------------------------------------------------------------------------


def test_the_image_route_serves_a_staged_frame(tmp_path):
    client, _ = _client(tmp_path)

    response = client.get("/api/frame/milo_0001.jpg/image")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content[:2] == b"\xff\xd8"  # a JPEG


def test_the_image_route_refuses_a_name_that_leaves_the_staged_set(tmp_path):
    (tmp_path / "secret.jpg").write_bytes(b"not for the browser")

    client, _ = _client(tmp_path)

    for name in ("../secret.jpg", "../../secret.jpg", "/etc/passwd", "..%2Fsecret.jpg"):
        response = client.get(f"/api/frame/{urllib.parse.quote(name, safe='')}/image")
        assert response.status_code == 404, name


def test_a_frame_outside_every_manifest_is_a_404_on_every_route(tmp_path):
    client, _ = _client(tmp_path, detector_factory=_factory(()))

    assert client.get("/api/frame/nope.jpg").status_code == 404
    assert client.get("/api/frame/nope.jpg/image").status_code == 404
    assert client.post("/api/frame/nope.jpg/suggest").status_code == 404
    assert client.post("/api/frame/nope.jpg/labels", json={"boxes": []}).status_code == 404


# --------------------------------------------------------------------------
# Which weight may suggest
# --------------------------------------------------------------------------


def _models_dir(tmp_path: Path, names, resize_mode="stretch") -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "scanncart-grocery-v2.pt").write_bytes(b"weights")
    (models / "scanncart-grocery-v2.json").write_text(
        json.dumps({"resize_mode": resize_mode, "class_names": list(names)}), encoding="utf-8"
    )
    return models


def test_choose_weights_accepts_a_weight_whose_recorded_classes_are_this_datasets(tmp_path):
    """Selection is by recorded class list rather than by filename, because the recommendation
    carries an *index* into this dataset's order."""
    weight, note = choose_weights(_models_dir(tmp_path, CLASS_NAMES))

    assert note == ""
    assert weight is not None
    assert weight.resize_mode == "stretch"
    assert weight.class_names == tuple(CLASS_NAMES)


def test_choose_weights_refuses_a_weight_whose_recorded_classes_are_not_this_datasets(tmp_path):
    """A weight that predicts a different list would propose boxes that get saved under classes
    that mean something else - the failure the annotator is the *first* place to be able to stop."""
    models = _models_dir(tmp_path, CLASS_NAMES[:-1])

    weight, note = choose_weights(models)

    assert weight is None
    assert CLASS_NAMES[-1] in note


def test_choose_weights_honours_an_explicit_request_and_explains_a_bad_one(tmp_path):
    """`--weights` is how an operator overrides the pick, so a request that cannot be used has to
    say which weight and why instead of quietly falling back to another one."""
    models = _models_dir(tmp_path, CLASS_NAMES)

    chosen, note = choose_weights(models, requested="scanncart-grocery-v2.pt")
    assert chosen is not None and note == ""

    _, missing = choose_weights(models, requested="something-else.pt")
    assert "not in" in missing


def test_a_missing_weights_directory_is_a_note_not_a_crash(tmp_path):
    """The annotator labels frames with no weight at all - the hard negatives in particular need
    nobody's help."""
    weight, note = choose_weights(tmp_path / "models")

    assert weight is None
    assert "no weights directory" in note


# --------------------------------------------------------------------------
# What leaves the machine
# --------------------------------------------------------------------------


def test_the_image_sent_to_a_hosted_provider_is_downscaled_and_still_a_jpeg():
    """Downscaling is free in the coordinates - a uniform scale keeps every normalized box where it
    was - and it is what stops a 3 MB staged frame becoming a 3 MB request per frame."""
    import cv2

    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    encoded = encode_for_hosted(frame)

    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == 1024
    assert encoded[:2] == b"\xff\xd8"


def test_a_small_frame_is_sent_as_it_is(tmp_path):
    """No upscaling, no re-framing: the provider sees the same picture the local weight was given,
    so a difference between their answers cannot be a difference in input."""
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    import cv2

    decoded = cv2.imdecode(np.frombuffer(encode_for_hosted(frame), dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded.shape[:2] == (32, 48)


def test_a_frame_that_cannot_be_decoded_is_a_415_and_the_others_still_work(tmp_path):
    """A corrupt file in the set is a per-frame problem: the worklist and the rest of the run must
    survive it."""
    client, state = _client(tmp_path, detector_factory=_factory(()))
    (state.store.out / "milo" / "milo_0001.jpg").write_bytes(b"not a jpeg")

    broken = client.post("/api/frame/milo_0001.jpg/suggest")

    assert broken.status_code == 415
    assert client.post("/api/frame/milo_0002.jpg/suggest").status_code == 200
