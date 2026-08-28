"""HTTP client for Roboflow Workflows.

Talks to either Roboflow's serverless endpoint or a self-hosted inference
server — they differ only by base URL, so one client covers the `cloud_api`
and `local_api` backends. See docs/DETECTOR_BACKENDS.md.

Uses `httpx` (already a sidecar dependency) rather than `inference-sdk`: no new
dependency, direct control of timeout and retry, and no Python-version ceiling
(`inference` caps at <3.13 while the sidecar targets 3.12).

The API key is passed in the `Authorization: Bearer` header only — never in the
query string or request body — and is never logged, repr'd, or raised in an
error message.
"""

import time
from typing import Any, Callable

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_RETRIES = 2
# Backoff must stay well under track_expiry_s, or a retry storm on one frame
# outlives the tracks it was meant to update.
BACKOFF_BASE_S = 0.2
BACKOFF_CAP_S = 2.0

_BOX_KEYS = ("x", "y", "width", "height")
_MAX_SEARCH_DEPTH = 6


class RoboflowError(Exception):
    """Base for every Roboflow client failure."""


class RoboflowAuthError(RoboflowError):
    """401/403 — missing, invalid, or unauthorized API key."""


class RoboflowUnavailable(RoboflowError):
    """Could not connect. Usually the local inference server isn't running."""


class RoboflowTimeout(RoboflowError):
    """Exceeded the request timeout on every attempt."""


def _httpx():
    import httpx

    return httpx


def _default_client_factory(timeout_s: float):
    return _httpx().Client(timeout=timeout_s)


def workflow_url(api_url: str, workspace: str, workflow_id: str) -> str:
    return f"{api_url.rstrip('/')}/{workspace}/workflows/{workflow_id}"


def _looks_like_predictions(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, dict) for v in value)
        and all(k in value[0] for k in _BOX_KEYS)
    )


def find_predictions(payload: Any, _depth: int = 0) -> list[dict]:
    """Locate the detection list in a workflow response.

    Output names are chosen by whoever built the Workflow, so they are
    discovered structurally rather than hard-coded. Strings are never
    traversed or retained — image-shaped outputs come back as base64 blobs of
    hundreds of KB and must not be held or logged.
    """
    if _depth > _MAX_SEARCH_DEPTH or isinstance(payload, (str, bytes)):
        return []
    if _looks_like_predictions(payload):
        return payload
    if isinstance(payload, dict):
        # An explicit `predictions` key wins, including when it is empty —
        # "the model found nothing" must not fall through to a deeper match.
        inner = payload.get("predictions")
        if isinstance(inner, list) and (not inner or _looks_like_predictions(inner)):
            return inner
        if isinstance(inner, dict):
            found = find_predictions(inner, _depth + 1)
            if found:
                return found
        for key, value in payload.items():
            if key == "predictions" or isinstance(value, (str, bytes)):
                continue
            found = find_predictions(value, _depth + 1)
            if found:
                return found
        return []
    if isinstance(payload, list):
        for item in payload:
            found = find_predictions(item, _depth + 1)
            if found:
                return found
    return []


def find_image_size(payload: Any, _depth: int = 0) -> tuple[int, int] | None:
    """The `image: {width, height}` block a workflow reports alongside its
    predictions, describing the frame the coordinates are relative to
    (`coordinates_system: "own"`). Returns None when absent, in which case the
    caller falls back to the dimensions it actually transmitted.
    """
    if _depth > _MAX_SEARCH_DEPTH or isinstance(payload, (str, bytes)):
        return None
    if isinstance(payload, dict):
        image = payload.get("image")
        if isinstance(image, dict):
            w, h = image.get("width"), image.get("height")
            if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                return w, h
        for key, value in payload.items():
            if isinstance(value, (str, bytes)):
                continue
            found = find_image_size(value, _depth + 1)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_image_size(item, _depth + 1)
            if found:
                return found
    return None


def first_result(body: Any) -> Any:
    """Unwrap a workflow response to the first image's output dict.

    The endpoint returns one entry per input image, either as a bare list or
    wrapped in an `outputs` key depending on version.
    """
    if isinstance(body, dict):
        outputs = body.get("outputs")
        if isinstance(outputs, list):
            return outputs[0] if outputs else {}
        return body
    if isinstance(body, list):
        return body[0] if body else {}
    return {}


class WorkflowClient:
    """One Roboflow Workflow, callable with a base64 JPEG.

    `client_factory` and `sleep` are injection seams so tests run against a
    fake transport with no network and no real backoff delay.
    """

    def __init__(
        self,
        api_url: str,
        workspace: str,
        workflow_id: str,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client_factory: Callable[[float], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = workflow_url(api_url, workspace, workflow_id)
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._sleep = sleep
        factory = client_factory or _default_client_factory
        self._client = factory(timeout_s)

    def __repr__(self) -> str:
        # Never let the key reach a log line or a traceback.
        return f"WorkflowClient(url={self._url!r}, api_key={'set' if self._api_key else 'unset'})"

    @property
    def url(self) -> str:
        return self._url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def run(self, image_b64: str, parameters: dict | None = None) -> Any:
        """POST one base64 JPEG and return the first image's output dict.

        Retries only on timeout, connection failure, and 5xx — never on 4xx,
        which will fail identically however many times it is sent.
        """
        httpx = _httpx()
        inputs: dict[str, Any] = {"image": {"type": "base64", "value": image_b64}}
        if parameters:
            inputs.update(parameters)
        body = {"inputs": inputs}

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S))
            try:
                response = self._client.post(self._url, json=body, headers=self._headers())
            except httpx.TimeoutException as exc:
                last_error = RoboflowTimeout(
                    f"Roboflow request timed out after {self._timeout_s}s"
                )
                _ = exc
                continue
            except httpx.ConnectError:
                last_error = RoboflowUnavailable(
                    f"Could not connect to {self._url}. For local_api, is "
                    "`inference server start` running?"
                )
                continue
            except httpx.HTTPError as exc:
                last_error = RoboflowError(f"Roboflow request failed: {type(exc).__name__}")
                continue

            status = response.status_code
            if status in (401, 403):
                raise RoboflowAuthError(
                    f"Roboflow rejected the API key (HTTP {status}). Check "
                    "ROBOFLOW_API_KEY in sidecar/.env."
                )
            if status == 404:
                raise RoboflowError(
                    f"Workflow not found (HTTP 404) at {self._url}. Check "
                    "roboflow_workspace and roboflow_workflow_id."
                )
            if 400 <= status < 500:
                raise RoboflowError(f"Roboflow rejected the request (HTTP {status}).")
            if status >= 500:
                last_error = RoboflowError(f"Roboflow server error (HTTP {status}).")
                continue

            try:
                return first_result(response.json())
            except ValueError:
                raise RoboflowError("Roboflow returned a non-JSON response.") from None

        raise last_error or RoboflowError("Roboflow request failed.")

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()
