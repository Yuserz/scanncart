"""Phase 3 coverage for app/roboflow.py — no network, fake transport only."""

import httpx
import pytest

from app.roboflow import (
    BACKOFF_CAP_S,
    RoboflowAuthError,
    RoboflowError,
    RoboflowTimeout,
    RoboflowUnavailable,
    WorkflowClient,
    find_predictions,
    first_result,
    workflow_url,
)

PRED = {"x": 10.0, "y": 20.0, "width": 4.0, "height": 6.0, "confidence": 0.9, "class_name": "can"}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, raises_value_error=False):
        self.status_code = status_code
        self._payload = payload
        self._raises = raises_value_error

    def json(self):
        if self._raises:
            raise ValueError("not json")
        return self._payload


class FakeClient:
    """Replays a scripted sequence of responses/exceptions, recording calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.closed = False

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        item = self.script.pop(0) if self.script else FakeResponse(200, [{}])
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


def make_client(script=(), **kw):
    fake = FakeClient(script)
    slept = []
    client = WorkflowClient(
        api_url=kw.pop("api_url", "https://serverless.roboflow.com"),
        workspace=kw.pop("workspace", "ws"),
        workflow_id=kw.pop("workflow_id", "wf"),
        api_key=kw.pop("api_key", "secret-key"),
        client_factory=lambda _t: fake,
        sleep=slept.append,
        **kw,
    )
    return client, fake, slept


# --- url building --------------------------------------------------------


def test_workflow_url_shape():
    assert workflow_url("https://a.com", "ws", "wf") == "https://a.com/ws/workflows/wf"


def test_workflow_url_strips_trailing_slash():
    assert workflow_url("https://a.com/", "ws", "wf") == "https://a.com/ws/workflows/wf"


def test_local_url_is_the_only_difference_for_self_hosted():
    assert workflow_url("http://127.0.0.1:9001", "ws", "wf") == (
        "http://127.0.0.1:9001/ws/workflows/wf"
    )


# --- auth ----------------------------------------------------------------


def test_key_is_sent_as_a_bearer_header():
    client, fake, _ = make_client([FakeResponse(200, [{"out": {"predictions": [PRED]}}])])
    client.run("BASE64")
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer secret-key"


def test_key_never_appears_in_url_or_body():
    client, fake, _ = make_client([FakeResponse(200, [{}])])
    client.run("BASE64")
    call = fake.calls[0]
    assert "secret-key" not in call["url"]
    assert "secret-key" not in str(call["json"])


def test_no_auth_header_when_key_is_absent():
    client, fake, _ = make_client([FakeResponse(200, [{}])], api_key=None)
    client.run("BASE64")
    assert "Authorization" not in fake.calls[0]["headers"]


def test_repr_does_not_leak_the_key():
    client, _, _ = make_client()
    assert "secret-key" not in repr(client)
    assert "set" in repr(client)


# --- request shape -------------------------------------------------------


def test_image_is_sent_as_base64_input():
    client, fake, _ = make_client([FakeResponse(200, [{}])])
    client.run("BASE64")
    assert fake.calls[0]["json"] == {
        "inputs": {"image": {"type": "base64", "value": "BASE64"}}
    }


def test_parameters_are_merged_into_inputs():
    client, fake, _ = make_client([FakeResponse(200, [{}])])
    client.run("BASE64", parameters={"confidence": 0.4})
    inputs = fake.calls[0]["json"]["inputs"]
    assert inputs["confidence"] == 0.4
    assert inputs["image"]["type"] == "base64"


# --- error mapping -------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_raise_and_do_not_retry(status):
    client, fake, _ = make_client([FakeResponse(status)] * 3)
    with pytest.raises(RoboflowAuthError):
        client.run("B")
    assert len(fake.calls) == 1


def test_404_names_the_misconfigured_fields():
    client, _, _ = make_client([FakeResponse(404)])
    with pytest.raises(RoboflowError, match="roboflow_workspace"):
        client.run("B")


def test_other_4xx_raises_without_retry():
    client, fake, _ = make_client([FakeResponse(422)] * 3)
    with pytest.raises(RoboflowError):
        client.run("B")
    assert len(fake.calls) == 1


def test_connect_error_maps_to_unavailable_and_hints_at_the_server():
    err = httpx.ConnectError("refused")
    client, _, _ = make_client([err, err, err])
    with pytest.raises(RoboflowUnavailable, match="inference server start"):
        client.run("B")


def test_timeout_maps_to_timeout_error():
    err = httpx.ReadTimeout("slow")
    client, _, _ = make_client([err, err, err])
    with pytest.raises(RoboflowTimeout):
        client.run("B")


def test_non_json_body_raises():
    client, _, _ = make_client([FakeResponse(200, raises_value_error=True)])
    with pytest.raises(RoboflowError, match="non-JSON"):
        client.run("B")


def test_generic_http_error_is_wrapped():
    client, _, _ = make_client([httpx.HTTPError("boom")] * 3)
    with pytest.raises(RoboflowError):
        client.run("B")


# --- retries -------------------------------------------------------------


def test_retries_then_succeeds():
    client, fake, _ = make_client(
        [httpx.ReadTimeout("x"), FakeResponse(200, [{"o": {"predictions": [PRED]}}])]
    )
    assert client.run("B") == {"o": {"predictions": [PRED]}}
    assert len(fake.calls) == 2


def test_attempt_count_is_retries_plus_one():
    client, fake, _ = make_client([httpx.ReadTimeout("x")] * 5, max_retries=2)
    with pytest.raises(RoboflowTimeout):
        client.run("B")
    assert len(fake.calls) == 3


def test_zero_retries_means_one_attempt():
    client, fake, _ = make_client([httpx.ReadTimeout("x")] * 3, max_retries=0)
    with pytest.raises(RoboflowTimeout):
        client.run("B")
    assert len(fake.calls) == 1


def test_5xx_is_retried():
    client, fake, _ = make_client([FakeResponse(500), FakeResponse(200, [{}])])
    client.run("B")
    assert len(fake.calls) == 2


def test_backoff_grows_and_is_capped():
    client, _, slept = make_client([httpx.ReadTimeout("x")] * 6, max_retries=5)
    with pytest.raises(RoboflowTimeout):
        client.run("B")
    assert slept == sorted(slept)
    assert all(s <= BACKOFF_CAP_S for s in slept)
    assert len(slept) == 5


def test_first_attempt_does_not_sleep():
    client, _, slept = make_client([FakeResponse(200, [{}])])
    client.run("B")
    assert slept == []


# --- response unwrapping -------------------------------------------------


def test_first_result_from_bare_list():
    assert first_result([{"a": 1}, {"b": 2}]) == {"a": 1}


def test_first_result_from_outputs_wrapper():
    assert first_result({"outputs": [{"a": 1}]}) == {"a": 1}


def test_first_result_from_empty_list():
    assert first_result([]) == {}
    assert first_result({"outputs": []}) == {}


def test_first_result_from_plain_dict():
    assert first_result({"a": 1}) == {"a": 1}


def test_first_result_from_garbage():
    assert first_result(None) == {}
    assert first_result("nope") == {}


# --- prediction discovery ------------------------------------------------


def test_finds_predictions_under_an_arbitrary_output_name():
    assert find_predictions({"my_custom_output": {"predictions": [PRED]}}) == [PRED]


def test_finds_a_bare_prediction_list():
    assert find_predictions([PRED]) == [PRED]


def test_finds_predictions_nested_two_levels_deep():
    assert find_predictions({"a": {"b": {"predictions": [PRED]}}}) == [PRED]


def test_finds_predictions_under_double_nesting():
    assert find_predictions({"out": {"predictions": {"predictions": [PRED]}}}) == [PRED]


def test_empty_predictions_returns_empty_not_a_deeper_match():
    """A model that detected nothing must not fall through to some other list."""
    payload = {"detections": {"predictions": []}, "other": {"predictions": [PRED]}}
    assert find_predictions(payload["detections"]) == []


def test_missing_predictions_returns_empty():
    assert find_predictions({"label": "nothing here"}) == []


def test_base64_image_output_is_never_traversed():
    huge = "A" * 10000
    assert find_predictions({"image": huge, "out": {"predictions": [PRED]}}) == [PRED]


def test_a_string_payload_is_ignored():
    assert find_predictions("x" * 100) == []


def test_deeply_buried_predictions_are_not_searched_forever():
    payload = {}
    node = payload
    for _ in range(20):
        node["next"] = {}
        node = node["next"]
    node["predictions"] = [PRED]
    assert find_predictions(payload) == []


def test_non_prediction_lists_are_not_mistaken_for_detections():
    assert find_predictions({"classes": ["a", "b"], "counts": [1, 2]}) == []


def test_requires_all_box_keys():
    partial = [{"x": 1, "y": 2}]
    assert find_predictions({"out": partial}) == []


# --- lifecycle -----------------------------------------------------------


def test_close_closes_the_underlying_client():
    client, fake, _ = make_client()
    client.close()
    assert fake.closed is True


def test_close_tolerates_a_client_without_close():
    class Bare:
        def post(self, *a, **k):
            return FakeResponse(200, [{}])

    c = WorkflowClient("https://a", "w", "f", client_factory=lambda _t: Bare())
    c.close()


def test_url_property_is_exposed():
    client, _, _ = make_client()
    assert client.url.endswith("/ws/workflows/wf")
