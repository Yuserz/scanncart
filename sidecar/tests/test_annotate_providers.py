"""Tests for `annotate/providers.py` - the two providers, and the order between them.

Three things are asserted here that are not features but rules, because breaking either of them
would be invisible in the annotator and expensive in the dataset:

* **A suggestion is not a label.** Every test drives `LocalWeights`/`HostedVision` directly; nothing
  in this module can write, and `record_suggestion` is the caller's job.
* **Local first, hosted only where local found nothing.** The order is a cost rule (money and
  egress), and it is asserted by counting the fake provider's calls rather than by reading a note.
* **A suggestion carries an index into *this* dataset's classes.** A local weight is asked by name
  and answered with an index (`Detection.cls` is a name); a hosted model is told the class list and
  answered with a name. Both go through `parse_class_name`, so a weight whose own order differs is
  usable instead of silently wrong - which is the failure `train_model.check_export` cannot see.

No network, no key, no GPU: the detector comes through the same factory injection point the
sidecar's own tests use, and every hosted call goes through a fake `post`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from annotate.providers import (
    HOSTED_PROVIDERS,
    HostedVision,
    LocalWeights,
    build_hosted,
    clamp_box,
    hosted_available,
    parse_class_name,
    suggest_for,
)
from annotate.store import CLASS_NAMES, CLASS_SLUGS, Box


class FakeDetector:
    """Stands in for `app.inference.YoloDetector`: one `infer()` per frame, no weights."""

    def __init__(self, detections):
        self.detections = list(detections)
        self.frames = 0

    def infer(self, frame):
        self.frames += 1
        return list(self.detections)


def _detection(cls, x1=0.1, y1=0.1, x2=0.4, y2=0.5):
    return SimpleNamespace(cls=cls, conf=0.9, box=(x1, y1, x2, y2))


def _factory(detections=(), calls=None):
    def build(weights, device, conf, imgsz, resize_mode="letterbox"):
        if calls is not None:
            calls.append(
                {
                    "weights": weights,
                    "device": device,
                    "conf": conf,
                    "imgsz": imgsz,
                    "resize_mode": resize_mode,
                }
            )
        return FakeDetector(detections)

    return build


class FakeHosted:
    """A `HostedVision`-shaped provider that counts its calls and never touches a network."""

    def __init__(self, boxes=(), note="2 proposed by fake:model", provider="hosted:fake"):
        self.boxes = list(boxes)
        self.note = note
        self.provider = provider
        self.calls = 0

    def suggest(self, frame_bytes, media_type="image/jpeg"):
        self.calls += 1
        return SimpleNamespace(boxes=list(self.boxes), provider=self.provider, note=self.note)


BUYER = "2 x 2"
# The BGR frame itself is never inspected by these fakes, but the shape is what the endpoint
# passes through, so a placeholder with the right signature keeps the intent readable.
FRAME = [[[0, 0, 0]]]


# --------------------------------------------------------------------------
# Geometry and names
# --------------------------------------------------------------------------


def test_clamp_pulls_a_box_inside_the_frame_and_a_degenerate_one_does_not_survive():
    """Model output is not trusted geometry: one coordinate at `-0.02` would be written to the
    label file and silently clipped by the trainer, leaving a frame that reads as labeled while
    contributing a box nobody can see."""
    box = clamp_box(0, cx=0.01, cy=0.5, w=0.1, h=0.2)

    assert box.cx - box.w / 2 == 0.0  # the left edge landed on the frame
    assert box.w == pytest.approx(0.06)
    assert box.valid()

    outside = clamp_box(0, cx=-0.5, cy=0.5, w=0.1, h=0.2)
    assert not outside.valid()  # nothing left of it, so it never reaches disk


def test_a_class_name_maps_to_the_dataset_index_and_nothing_else_does():
    """Exact, slug, and case-insensitive - and never an invented index. The local weight is the
    one that cannot make this mistake (it predicts an index); a hosted model is told the names."""
    assert parse_class_name(CLASS_NAMES[3]) == 3
    assert parse_class_name(CLASS_SLUGS[3]) == 3
    assert parse_class_name(CLASS_NAMES[3].upper()) == 3
    assert parse_class_name("soup") is None
    assert parse_class_name("") is None
    assert parse_class_name(None) is None


# --------------------------------------------------------------------------
# Local weights
# --------------------------------------------------------------------------


def test_a_weight_whose_recorded_classes_differ_is_refused_rather_than_used():
    """The roster guard, failing closed. A weight is offered boxes only if its recorded class list
    *is* this dataset's list, because a suggestion carries an index: a head with 21 outputs would
    propose boxes that get saved under a class the labels do not have."""
    with pytest.raises(ValueError, match="different class list"):
        LocalWeights("models/scanncart-grocery-v9.pt", class_names=list(CLASS_NAMES)[:3])


def test_a_suggestion_translates_the_weights_own_label_into_this_datasets_index():
    """`Detection.cls` is a class *name*, so the translation is the point: the weight's own order
    is irrelevant, and a label outside the roster is dropped and reported rather than indexed."""
    detections = [_detection(CLASS_NAMES[5]), _detection("soup")]
    local = LocalWeights(
        "models/scanncart-grocery-v2.pt",
        detector_factory=_factory(detections),
        class_names=list(CLASS_NAMES),
    )

    proposed = local.suggest(FRAME)

    assert [box.cls for box in proposed.boxes] == [5]
    assert proposed.provider == "local:scanncart-grocery-v2"
    assert "ignored labels outside this roster: soup" in proposed.note


def test_the_detector_is_built_with_the_recorded_geometry_not_a_default():
    """The suggestion is the box the *running app* would draw, so the detector is built the way the
    capture path builds it: the record's `resize_mode`, the run's `imgsz` and confidence floor.
    Suggesting at a guessed geometry is how an annotator gets weak boxes and blames the weights."""
    calls: list[dict] = []
    local = LocalWeights(
        "models/scanncart-grocery-v1.pt",
        resize_mode="stretch",
        conf=0.4,
        imgsz=512,
        device="cuda",
        detector_factory=_factory((), calls),
        class_names=list(CLASS_NAMES),
    )

    local.suggest(FRAME)
    local.suggest(FRAME)  # built once, reused: a weight load is ~1s and a session opens hundreds

    assert calls == [
        {
            "weights": "models/scanncart-grocery-v1.pt",
            "device": "cuda",
            "conf": 0.4,
            "imgsz": 512,
            "resize_mode": "stretch",
        }
    ]


def test_a_box_the_weight_draws_with_no_survivable_extent_is_dropped():
    """A clamped box that lost its width is not a box, and the frame it came from reads as
    outstanding rather than as labeled with an invisible row."""
    local = LocalWeights(
        "models/w.pt",
        detector_factory=_factory([_detection(CLASS_NAMES[0], x1=-0.4, y1=0.2, x2=-0.2, y2=0.4)]),
        class_names=list(CLASS_NAMES),
    )

    assert local.suggest(FRAME).boxes == []


# --------------------------------------------------------------------------
# Hosted vision models
# --------------------------------------------------------------------------


def _hosted(name="openai", **kwargs):
    return HostedVision(name, "sk-secret", kwargs.pop("post", lambda *a: (200, {})), **kwargs)


def test_each_vendor_envelope_is_read_back_to_the_same_boxes():
    """One prompt, three shapes. The envelopes are written out per vendor on purpose - the
    difference is not cosmetic - so each one is read back to the same normalized box."""
    objects = [{"class": CLASS_NAMES[2], "x": 0.25, "y": 0.5, "w": 0.5, "h": 0.25}]
    answer = json.dumps({"objects": objects})
    bodies = {
        "openai": {"choices": [{"message": {"content": answer}}]},
        "gemini": {"candidates": [{"content": {"parts": [{"text": answer}]}}]},
        "anthropic": {"content": [{"type": "text", "text": answer}]},
    }

    for name, body in bodies.items():
        proposed = _hosted(name)._parse(body)
        assert [box.cls for box in proposed.boxes] == [2], name
        assert proposed.boxes[0].cx == pytest.approx(0.5), name
        assert proposed.boxes[0].cy == pytest.approx(0.625), name
        assert proposed.provider == f"hosted:{name}"


@pytest.mark.parametrize(
    "text",
    [
        "```json\n{\"objects\": [{\"class\": \"%s\", \"x\": 0, \"y\": 0, \"w\": 1, \"h\": 1}]}\n```",
        "Sure, here it is: {\"objects\": [{\"class\": \"%s\", \"x\": 0, \"y\": 0, \"w\": 1, \"h\": 1}]}",
    ],
)
def test_an_answer_wrapped_in_prose_or_a_code_fence_is_still_read(text):
    """The wrapper is the part that varies; a model that prefixes its JSON with a sentence is
    answering correctly. Structural, like `roboflow.find_predictions`."""
    proposed = _hosted()._parse({"choices": [{"message": {"content": text % CLASS_NAMES[1]}}]})

    assert [box.cls for box in proposed.boxes] == [1]


def test_a_hosted_label_outside_our_classes_is_named_rather_than_dropped_in_silence():
    """\"It found something, under a name we do not have\" is a different fact from \"it found
    nothing\", and it is the ordinary failure: a general vision model has its own vocabulary."""
    answer = json.dumps({"objects": [{"class": "a bottle of milk", "x": 0, "y": 0, "w": 1, "h": 1}]})

    proposed = _hosted()._parse({"choices": [{"message": {"content": answer}}]})

    assert proposed.boxes == []
    assert "ignored labels outside this roster: a bottle of milk" in proposed.note


def test_an_http_error_raises_without_the_key_in_the_message():
    """The rule `credentials.py` and `roboflow.py` both keep: a key reaches no log line, no
    response body and no traceback."""
    def post(url, headers, payload, timeout):
        return 401, {"error": "bad key sk-secret"}

    with pytest.raises(RuntimeError) as caught:
        _hosted(post=post).suggest(b"jpeg")

    assert "sk-secret" not in str(caught.value)
    assert "***" in str(caught.value)


def test_the_provider_sends_the_class_list_and_never_the_key_in_the_query():
    """The prompt is the reason a hosted model can help at all: it is *told* this dataset's
    classes, so it can answer about a product no local weight has a name for."""
    seen: dict = {}

    def post(url, headers, payload, timeout):
        seen.update({"url": url, "headers": headers, "payload": payload})
        return 200, {"choices": [{"message": {"content": "{\"objects\": []}"}}]}

    _hosted(post=post).suggest(b"jpeg", media_type="image/jpeg")

    prompt = seen["payload"]["messages"][0]["content"][0]["text"]
    assert all(name in prompt for name in CLASS_NAMES)
    assert "data:image/jpeg;base64," in seen["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    assert seen["headers"]["Authorization"] == "Bearer sk-secret"


def test_a_missing_key_means_the_provider_is_simply_not_offered():
    """The annotator runs with no key at all - local weights label the frames - so a key drives
    only which second opinion exists, never whether the tool works."""
    assert hosted_available(getter=lambda name: "") == []
    assert hosted_available(getter=lambda name: "k" if name == "OPENAI_API_KEY" else "") == [
        "openai"
    ]

    assert build_hosted("openai", lambda *a: (200, {}), getter=lambda name: "") is None
    built = build_hosted("openai", lambda *a: (200, {}), getter=lambda name: "sk")
    assert built is not None and built.name == "openai"
    # An unknown provider name is not a crash, it is nothing to offer.
    assert build_hosted("nope", lambda *a: (200, {}), getter=lambda name: "sk") is None


def test_the_default_transport_is_the_live_one_not_a_test_seam(monkeypatch):
    """`annotate/run.py` passes no `post`, so the default has to be the network.

    With the default at `None`, every hosted suggestion answered `TypeError: 'NoneType' object is
    not callable` inside the request - a second opinion offered by `--hosted`, printed in the
    startup line and counted against `--max-hosted-calls`, then failing on the one frame it was
    asked about. The suite could not see it, because every other test here injects `post`.
    """
    import annotate.providers as providers

    assert build_hosted("openai", None, getter=lambda name: "sk")._post is providers.http_post

    seen: dict = {}

    def fake(url, headers, payload, timeout):
        seen.update({"url": url, "timeout": timeout})
        return 200, {"choices": [{"message": {"content": '{"objects": []}'}}]}

    monkeypatch.setattr(providers, "http_post", fake)
    proposed = build_hosted("openai", None, getter=lambda name: "sk").suggest(b"jpeg")

    assert proposed.provider == "hosted:openai"
    assert seen["url"] == HOSTED_PROVIDERS["openai"]["url"]


def test_a_transport_failure_becomes_a_sentence_with_no_key_in_it():
    """Gemini's key is a query parameter, so its own `ConnectError` would print the key into a log.
    The seam is where that is turned into a sentence instead - the same rule the HTTP-error path
    above keeps, applied to the failure that has no status code to attach it to."""

    def post(url, headers, payload, timeout):
        raise OSError(f"connection failed for {url}")

    with pytest.raises(RuntimeError) as caught:
        _hosted("gemini", post=post).suggest(b"jpeg")

    assert "sk-secret" not in str(caught.value)
    assert "could not be reached" in str(caught.value)


def test_the_hosted_client_uses_the_vendors_own_env_var_names():
    """So an operator with a key already exported does not have to learn a second name."""
    assert HOSTED_PROVIDERS["openai"]["env"] == "OPENAI_API_KEY"
    assert HOSTED_PROVIDERS["gemini"]["env"] == "GEMINI_API_KEY"
    assert HOSTED_PROVIDERS["anthropic"]["env"] == "ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------
# The order: local first, hosted only where local found nothing
# --------------------------------------------------------------------------


def test_local_answers_alone_and_the_hosted_provider_is_never_called():
    """The cost rule. Local is free, offline and fast, so a frame it can help with spends no
    money and sends no store imagery anywhere."""
    hosted = FakeHosted()
    local = LocalWeights(
        "models/w.pt",
        detector_factory=_factory([_detection(CLASS_NAMES[0])]),
        class_names=list(CLASS_NAMES),
    )

    proposed = suggest_for(FRAME, local, hosted, b"jpeg")

    assert [box.cls for box in proposed.boxes] == [0]
    assert proposed.provider.startswith("local:")
    assert hosted.calls == 0


def test_the_hosted_provider_is_asked_only_where_local_found_nothing():
    """The one case a second opinion is worth something: a free offline model could not help, and
    the frame is exactly the `far`/cluttered case this dataset exists for."""
    hosted = FakeHosted(boxes=[Box(0, 0.5, 0.5, 0.2, 0.2)])
    local = LocalWeights("models/w.pt", detector_factory=_factory(()), class_names=list(CLASS_NAMES))

    proposed = suggest_for(FRAME, local, hosted, b"jpeg")

    assert hosted.calls == 1
    assert proposed.provider == "hosted:fake"
    assert [box.cls for box in proposed.boxes] == [0]
    assert proposed.note.startswith("local found nothing;")


def test_local_finding_nothing_with_no_hosted_provider_says_exactly_that():
    """A frame nobody suggested is still a frame to label, and the note has to distinguish
    \"asked and got nothing\" from \"nothing was asked\"."""
    local = LocalWeights("models/w.pt", detector_factory=_factory(()), class_names=list(CLASS_NAMES))

    proposed = suggest_for(FRAME, local, None)

    assert proposed.boxes == []
    assert "no hosted provider is configured" in proposed.note


def test_a_hosted_provider_on_its_own_is_asked_rather_than_refused():
    """With no usable local weight the ordering rule has nothing to consult, and refusing would
    mean an annotator that cannot suggest at all on a machine with no installed weight."""
    hosted = FakeHosted(boxes=[Box(1, 0.5, 0.5, 0.2, 0.2)])

    assert [b.cls for b in suggest_for(FRAME, None, hosted, b"jpeg").boxes] == [1]
    assert hosted.calls == 1

    with pytest.raises(ValueError, match="needs the encoded image"):
        suggest_for(FRAME, None, hosted)


def test_nothing_configured_proposes_nothing_and_says_so():
    proposed = suggest_for(FRAME, None, None)

    assert proposed.boxes == []
    assert proposed.provider == "none"
    assert "no provider is configured" in proposed.note
