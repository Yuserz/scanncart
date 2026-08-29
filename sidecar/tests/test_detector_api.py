"""Phase 4 wiring: detector_factory selection, /api/detector/probe, and the
error mapping on capture start. No network — the client is always faked."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.inference import RoboflowRemoteDetector, YoloDetector
from app.main import AppState, backend_url, build_app
from app.roboflow import (
    RoboflowAuthError,
    RoboflowError,
    RoboflowTimeout,
    RoboflowUnavailable,
)
from app.settings import Settings


@pytest.fixture(autouse=True)
def _no_real_key(monkeypatch):
    """Never read the developer's real sidecar/.env during tests."""
    monkeypatch.setattr(main, "load_api_key", lambda: "test-key")


def settings(**over):
    s = Settings()
    for k, v in over.items():
        setattr(s, k, v)
    return s


# --- backend_url ---------------------------------------------------------


def test_backend_url_for_local():
    assert backend_url(settings(detector_backend="local_api")) == "http://127.0.0.1:9001"


def test_backend_url_for_cloud():
    assert backend_url(settings(detector_backend="cloud_api")).startswith("https://serverless")


# --- detector_factory ----------------------------------------------------


def test_native_backend_builds_a_yolo_detector(monkeypatch):
    built = {}
    monkeypatch.setattr(
        main, "YoloDetector", lambda *a, **k: built.setdefault("d", object())
    )
    d = main._default_detector_factory(settings(detector_backend="native"), "cpu")
    assert d is built["d"]


@pytest.mark.parametrize("backend", ["local_api", "cloud_api"])
def test_remote_backends_build_a_roboflow_detector(backend):
    d = main._default_detector_factory(settings(detector_backend=backend), "cpu")
    assert isinstance(d, RoboflowRemoteDetector)
    d.close()


def test_remote_detector_gets_a_tracker():
    """The workflow has no tracking block, so a tracker is mandatory."""
    d = main._default_detector_factory(settings(detector_backend="cloud_api"), "cpu")
    assert d._tracker is not None
    d.close()


def test_tracker_expiry_matches_the_pipeline():
    d = main._default_detector_factory(
        settings(detector_backend="cloud_api", track_expiry_s=4.0), "cpu"
    )
    assert d._tracker._expiry_s == 4.0
    d.close()


def test_cloud_without_a_key_fails_fast(monkeypatch):
    monkeypatch.setattr(main, "load_api_key", lambda: None)
    with pytest.raises(RoboflowAuthError, match="ROBOFLOW_API_KEY"):
        main._default_detector_factory(settings(detector_backend="cloud_api"), "cpu")


def test_local_without_a_key_is_allowed(monkeypatch):
    """A self-hosted server may not require auth at all."""
    monkeypatch.setattr(main, "load_api_key", lambda: None)
    d = main._default_detector_factory(settings(detector_backend="local_api"), "cpu")
    assert isinstance(d, RoboflowRemoteDetector)
    d.close()


# --- probe endpoint ------------------------------------------------------


def client_for(tmp_path, detector_factory=None, **over):
    state = AppState(
        settings=settings(**over),
        settings_path=str(tmp_path / "s.json"),
        db_path=str(tmp_path / "t.db"),
        api_key_probe=lambda: True,
    )
    if detector_factory is not None:
        state.detector_factory = detector_factory
    return TestClient(build_app(lambda: state))


class FakeDetector:
    def __init__(self, raises=None, names=None):
        self.raises = raises
        self.names = names or {}
        self.closed = False

    def infer(self, frame):
        if self.raises:
            raise self.raises
        return []

    def close(self):
        self.closed = True


def test_probe_native_reports_reachable(tmp_path):
    r = client_for(tmp_path, detector_backend="native").post("/api/detector/probe").json()
    assert r["backend"] == "native"
    assert r["reachable"] is True


def test_probe_native_flags_missing_weights(tmp_path):
    r = (
        client_for(tmp_path, detector_backend="native", active_model="yolo11x.pt")
        .post("/api/detector/probe")
        .json()
    )
    assert "not on disk" in r["detail"] or "present" in r["detail"]


def test_probe_remote_success_reports_latency_and_classes(tmp_path):
    fake = FakeDetector(names={3: "century_tuna_flakes_in_oil_155_grams"})
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="cloud_api")
        .post("/api/detector/probe")
        .json()
    )
    assert r["reachable"] is True
    assert r["latency_ms"] is not None
    assert r["class_names"] == ["century_tuna_flakes_in_oil_155_grams"]


def test_probe_closes_the_detector_it_built(tmp_path):
    fake = FakeDetector()
    client_for(tmp_path, lambda s, d: fake, detector_backend="cloud_api").post(
        "/api/detector/probe"
    )
    assert fake.closed is True


def test_probe_reports_unreachable_rather_than_erroring(tmp_path):
    fake = FakeDetector(raises=RoboflowUnavailable("server down"))
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="local_api")
        .post("/api/detector/probe")
        .json()
    )
    assert r["reachable"] is False
    assert "server down" in r["detail"]


def test_probe_surfaces_an_auth_failure(tmp_path):
    fake = FakeDetector(raises=RoboflowAuthError("bad key"))
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="cloud_api")
        .post("/api/detector/probe")
        .json()
    )
    assert r["reachable"] is False
    assert "bad key" in r["detail"]


# --- capture start error mapping -----------------------------------------


class FakeSource:
    def open(self):
        pass

    def close(self):
        pass

    def read(self):
        return None


def _raising_factory(exc):
    def factory(settings, device):
        raise exc

    return factory


@pytest.mark.parametrize(
    "exc,status",
    [
        (RoboflowAuthError("no key"), 401),
        (RoboflowUnavailable("refused"), 503),
        (RoboflowTimeout("slow"), 504),
        (RoboflowError("weird"), 502),
    ],
)
def test_capture_start_maps_roboflow_errors(tmp_path, exc, status):
    state = AppState(
        settings=settings(detector_backend="cloud_api"),
        settings_path=str(tmp_path / "s.json"),
        db_path=str(tmp_path / "t.db"),
        api_key_probe=lambda: True,
    )
    state.source_factory = lambda s: FakeSource()
    state.detector_factory = _raising_factory(exc)
    r = TestClient(build_app(lambda: state)).post("/api/capture/start")
    assert r.status_code == status
    assert str(exc) in r.json()["detail"]


def test_failed_start_leaves_state_idle(tmp_path):
    state = AppState(
        settings=settings(detector_backend="cloud_api"),
        settings_path=str(tmp_path / "s.json"),
        db_path=str(tmp_path / "t.db"),
        api_key_probe=lambda: True,
    )
    state.source_factory = lambda s: FakeSource()
    state.detector_factory = _raising_factory(RoboflowUnavailable("down"))
    c = TestClient(build_app(lambda: state))
    c.post("/api/capture/start")
    assert c.get("/api/health").json()["state"] == "idle"


def test_failed_start_closes_the_camera(tmp_path):
    """Otherwise a retry hits a camera already held open by the failed attempt."""
    opened = FakeSource()
    opened.closed = False
    opened.close = lambda: setattr(opened, "closed", True)
    state = AppState(
        settings=settings(detector_backend="cloud_api"),
        settings_path=str(tmp_path / "s.json"),
        db_path=str(tmp_path / "t.db"),
        api_key_probe=lambda: True,
    )
    state.source_factory = lambda s: opened
    state.detector_factory = _raising_factory(RoboflowUnavailable("down"))
    TestClient(build_app(lambda: state)).post("/api/capture/start")
    assert opened.closed is True
