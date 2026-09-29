"""What the app *does* with the monitor: when it starts, where it points, and who gets told.

`tests/test_inference_health.py` covers the verdict; this covers the wiring, which is the half that
can be wrong while every unit test passes - a monitor nobody starts, a startup that watches a URL
the settings do not name, or a transition that reaches the log but not a connected client. The probe
is injected everywhere (`AppState.inference_probe`), so nothing here opens a socket, and the monitor
comes from `AppState.inference_monitor_factory` with the real timings replaced: the app owns *which*
probe and *which* callback, the test owns the clock.

`TestClient(app)` without a context manager does not run the lifespan, which is why the handshake
test below can assert a stored verdict with nothing running, and why the ones that are about the
watch say `with` explicitly.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from app.inference_health import (
    OK,
    UNKNOWN,
    UNRESPONSIVE,
    InferenceHealthMonitor,
    InferenceStatus,
)
from app.main import AppState, build_app
from app.settings import Settings

LOCAL = "http://127.0.0.1:9001"


def settings(**over) -> Settings:
    s = Settings()
    for key, value in over.items():
        setattr(s, key, value)
    return s


class _Probe:
    """A probe that answers the same thing every time and remembers what it was asked.

    `release` is the seam that makes a transition land when the test wants it to: left unset, the
    probe blocks until the test sets it, so a verdict can be made to arrive *after* a client has
    connected rather than during startup, where no client would be there to receive it.
    """

    def __init__(self, answer: tuple[bool, str] = (True, "")) -> None:
        self.answer = answer
        self.urls: list[str] = []
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, url: str) -> tuple[bool, str]:
        with self._lock:
            self.urls.append(url)
        self.release.wait(5)
        return self.answer

    def wait_for(self, url: str, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if url in self.urls:
                    return True
            time.sleep(0.01)
        return False


def _factory(*, interval_s: float = 0.01, failures: int = 1) -> Callable[..., InferenceHealthMonitor]:
    """The app's monitor with a clock a test can wait out (`monitor_for`'s shape, other timings)."""

    def build(probe, on_change):
        return InferenceHealthMonitor(
            probe, on_change, interval_s=interval_s, failures=failures
        )

    return build


def _state(tmp_path, probe: _Probe, **over) -> AppState:
    return AppState(
        settings=settings(**over),
        settings_path=str(tmp_path / "settings.json"),
        inference_probe=probe,
        inference_monitor_factory=_factory(),
    )


def test_the_handshake_carries_the_verdict_a_late_client_missed(tmp_path):
    """The monitor reports transitions, so a client that connects afterwards has only this."""
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)
    state.inference_health = InferenceStatus(
        backend="local_api", url=LOCAL, state=UNRESPONSIVE, detail="ConnectError: refused"
    )

    client = TestClient(build_app(lambda: state))
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "status"
        message = ws.receive_json()

    assert message["type"] == "inference"
    assert message["state"] == UNRESPONSIVE
    assert message["backend"] == "local_api"
    assert message["url"] == LOCAL
    assert message["detail"] == "ConnectError: refused"


def test_the_handshake_says_unknown_for_a_native_backend(tmp_path):
    """`unknown` is what a client renders nothing for, and it is the ordinary answer here."""
    probe = _Probe()
    state = _state(tmp_path, probe)

    client = TestClient(build_app(lambda: state))
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        message = ws.receive_json()

    assert message["type"] == "inference"
    assert message["state"] == UNKNOWN
    assert message["url"] == ""
    assert probe.urls == []


def test_the_lifespan_starts_the_watch_and_stops_it(tmp_path):
    probe = _Probe()
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)

    assert state.inference_monitor is None
    with TestClient(build_app(lambda: state)):
        assert state.inference_monitor is not None, "the app started without watching anything"
        assert probe.wait_for(LOCAL), "the watch never probed the configured URL"
    assert state.inference_monitor is None, "the monitor outlived the app"


def test_a_native_backend_watches_nothing_at_startup(tmp_path):
    probe = _Probe()
    state = _state(tmp_path, probe)

    with TestClient(build_app(lambda: state)):
        time.sleep(0.1)
        assert probe.urls == [], "a native backend must not call a URL nobody configured"


def test_a_verdict_reaches_a_connected_client(tmp_path):
    """The broadcast half: the monitor is what notices, the client is what needs telling."""
    probe = _Probe(answer=(False, "ConnectError: refused"))
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)

    with TestClient(build_app(lambda: state)) as client:
        with client.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["type"] == "status"
            assert ws.receive_json()["state"] == UNKNOWN, "the probe is still in the air"
            # ...which is what `release` is for: the answer lands now, with a client watching.
            probe.release.set()
            message = ws.receive_json()

    assert message["type"] == "inference"
    assert message["state"] == UNRESPONSIVE
    assert message["detail"] == "ConnectError: refused"
    assert state.inference_health.state == UNRESPONSIVE


def test_changing_the_backend_moves_what_is_watched(tmp_path):
    """A patch that moves the backend or its URL has to move the watch, or the notice is about the
    server the app no longer calls."""
    probe = _Probe()
    state = _state(tmp_path, probe)

    other = "http://127.0.0.1:9333"
    with TestClient(build_app(lambda: state)) as client:
        assert probe.urls == []
        response = client.patch(
            "/api/settings?persist=false",
            json={"detector_backend": "local_api", "local_api_url": other},
        )
        assert response.status_code == 200, response.text
        assert probe.wait_for(other), f"the watch never moved to {other}"
        assert state.inference_health.url == other

        # ...and back to native, which calls nothing: no further probes.
        assert client.patch("/api/settings?persist=false", json={"detector_backend": "native"}).status_code == 200
        seen = len(probe.urls)
        time.sleep(0.1)
        assert len(probe.urls) == seen, "a native backend kept probing an endpoint"
        assert state.inference_health.state == UNKNOWN
        assert state.inference_health.url == ""


def test_a_recovering_server_is_reported_to_clients_too(tmp_path):
    """Both directions, and the count restarted in between - `retarget` is not the only path."""
    probe = _Probe(answer=(False, "ConnectError: refused"))
    state = _state(tmp_path, probe, detector_backend="local_api", local_api_url=LOCAL)

    with TestClient(build_app(lambda: state)) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            assert ws.receive_json()["state"] == UNKNOWN
            probe.release.set()
            assert ws.receive_json()["state"] == UNRESPONSIVE

            probe.answer = (True, "")
            probe.release.clear()
            probe.release.set()
            message = ws.receive_json()

    assert message["state"] == OK
    assert state.inference_health.state == OK
