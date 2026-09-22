"""Phase 4 wiring: detector_factory selection, /api/detector/probe, and the
error mapping on capture start. No network — the client is always faked."""

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.inference import RoboflowRemoteDetector, YoloDetector
from app.main import AppState, backend_url, build_app
from app.settings_store import CUSTOM_MODEL_SUFFIXES
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


def _weights_on_disk(monkeypatch, present: bool = True) -> None:
    """Answer the probe's one filesystem question, and nobody else's.

    A blanket `lambda _p: True` also answers *torch*, which asks `os.path.exists` about its own DLL
    directories while being imported (`AppState.__post_init__` → `resolve_device`) and calls
    `os.add_dll_directory` on whatever says yes — including a conda-shaped `.venv/Library/bin`
    that does not exist here. That made these tests pass only when an earlier test in the same
    process had already imported torch, and fail with a `FileNotFoundError` inside torch when this
    file was run on its own. Narrowing the lie to the weight suffixes keeps the intent — the
    weights are (or are not) there — without holding an opinion about the rest of the filesystem.
    """
    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda p: present if str(p).endswith(CUSTOM_MODEL_SUFFIXES) else real_exists(p),
    )


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


def _capture_yolo(monkeypatch):
    """A stand-in for YoloDetector that records how it was constructed."""
    captured: dict = {}

    def fake(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(main, "YoloDetector", fake)
    return captured


def test_native_backend_applies_a_recorded_resize_requirement(monkeypatch):
    """The end of the chain, and the point of the whole record: `auto` reaches the detector as
    the geometry these weights were trained with, not as the letterbox their `.pt` suffix
    suggests. Before this, the requirement was only *displayed* - the app flagged the mismatch
    and left the operator to go and set the field."""
    captured = _capture_yolo(monkeypatch)
    monkeypatch.setattr(main, "requirement_for", lambda name: "stretch")

    main._default_detector_factory(
        settings(detector_backend="native", active_model="models/x.pt", resize_mode="auto"),
        "cpu",
    )
    assert captured["model"] == "models/x.pt"
    assert captured["resize_mode"] == "stretch"


def test_native_backend_without_a_record_still_letterboxes_a_pt(monkeypatch):
    """A hand-copied `.pt` has nothing recorded, so the heuristic answers - unchanged, and the
    reason it has to stay: `None` is the normal case for anything not installed by the tool."""
    captured = _capture_yolo(monkeypatch)
    monkeypatch.setattr(main, "requirement_for", lambda name: None)

    main._default_detector_factory(
        settings(detector_backend="native", active_model="models/x.pt", resize_mode="auto"),
        "cpu",
    )
    assert captured["resize_mode"] == "letterbox"


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
    """A `Detector`, and deliberately nothing more — no `last_geometry`, since that is a
    remote-detector extra rather than part of the protocol the pipeline depends on."""

    def __init__(self, raises=None, names=None, provider=None, geometry=None, records_shape=False):
        self.raises = raises
        self.names = names or {}
        self.provider = provider
        self.closed = False
        self.infer_calls = 0
        self.last_geometry = geometry
        self.shape = None
        self._records_shape = records_shape

    def infer(self, frame):
        self.infer_calls += 1
        if self._records_shape:
            self.shape = frame.shape
        if self.raises:
            raise self.raises
        return []

    def close(self):
        self.closed = True


def test_probe_native_builds_and_measures_the_detector(tmp_path, monkeypatch):
    """The native probe does what the remote one already did: build the
    detector through the factory, warm the session, measure a real inference
    and report what it found — latency, the 7 SKUs, and the provider."""
    _weights_on_disk(monkeypatch)
    fake = FakeDetector(names={3: "milo"}, provider="CUDAExecutionProvider")
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="native")
        .post("/api/detector/probe")
        .json()
    )
    assert r["backend"] == "native"
    assert r["reachable"] is True
    assert r["latency_ms"] is not None
    assert r["class_names"] == ["milo"]
    assert r["provider"] == "CUDAExecutionProvider"
    # One warmup call plus one measured call.
    assert fake.infer_calls == 2
    assert fake.closed is True


def test_probe_native_missing_weights_short_circuits_without_building(tmp_path, monkeypatch):
    """A missing file is reported as a future download, never probed — the
    probe must not trigger a download or a network hit."""
    _weights_on_disk(monkeypatch, present=False)
    built = []

    def factory(settings, device):
        built.append(1)
        return FakeDetector()

    r = (
        client_for(tmp_path, factory, detector_backend="native", active_model="yolo11x.pt")
        .post("/api/detector/probe")
        .json()
    )
    assert r["reachable"] is True
    assert "not on disk" in r["detail"]
    assert built == []


def test_probe_native_load_failure_is_unreachable(tmp_path, monkeypatch):
    """A model that fails to load or infer surfaces in the probe response,
    not as a failed capture start."""
    _weights_on_disk(monkeypatch)
    fake = FakeDetector(raises=RuntimeError("corrupt weights"))
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="native")
        .post("/api/detector/probe")
        .json()
    )
    assert r["reachable"] is False
    assert "corrupt weights" in r["detail"]


def test_probe_native_load_failure_still_closes_the_detector(tmp_path, monkeypatch):
    _weights_on_disk(monkeypatch)
    fake = FakeDetector(raises=RuntimeError("boom"))
    client_for(tmp_path, lambda s, d: fake, detector_backend="native").post(
        "/api/detector/probe"
    )
    assert fake.closed is True


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


def test_probe_remote_reports_what_it_sent_and_what_the_workflow_answered(tmp_path):
    """The remote analogue of `resize_mode_resolved`: the geometry a live frame is sent at, beside
    the geometry the workflow's coordinates are relative to. Both come from the round trip that
    just happened, so the panel compares two observations rather than a number it derived."""
    from app.inference import RemoteGeometry

    fake = FakeDetector(geometry=RemoteGeometry(sent=(640, 360), reported=(640, 640)))
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="local_api")
        .post("/api/detector/probe")
        .json()
    )
    assert r["reachable"] is True
    assert r["sent_size"] == [640, 360]
    assert r["reported_size"] == [640, 640]


def test_probe_remote_reports_a_silent_workflow_as_silent(tmp_path):
    """A response with no size block leaves `reported_size` null — which the panel has to render
    differently from a match, because it means the coordinates were *assumed* to be relative to
    the frame sent rather than observed to be."""
    from app.inference import RemoteGeometry

    fake = FakeDetector(geometry=RemoteGeometry(sent=(640, 360), reported=None))
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="local_api")
        .post("/api/detector/probe")
        .json()
    )
    assert r["sent_size"] == [640, 360]
    assert r["reported_size"] is None


def test_probe_reports_no_remote_geometry_for_a_backend_that_never_transmits(tmp_path):
    """`native`'s geometry is a settings fact, already on `SettingsResponse`; a probe that invented
    a pair of sizes for it would be answering a question nobody asked in this response."""
    fake = FakeDetector()
    r = (
        client_for(tmp_path, lambda s, d: fake, detector_backend="native")
        .post("/api/detector/probe")
        .json()
    )
    assert r["sent_size"] is None and r["reported_size"] is None


def test_probe_remote_sends_a_frame_shaped_like_a_real_capture(tmp_path):
    """A 64x64 square could not show an aspect mismatch at all, which is the whole point of asking
    about geometry — so the probe has to send the frame capture would send, through the same
    downscale. Pinned by shape here; the sizes it produces are `transmit_size`'s business."""
    fake = FakeDetector(records_shape=True)
    client_for(
        tmp_path,
        lambda s, d: fake,
        detector_backend="local_api",
        capture_width=1280,
        capture_height=720,
    ).post("/api/detector/probe")
    assert fake.shape == (720, 1280, 3)


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
