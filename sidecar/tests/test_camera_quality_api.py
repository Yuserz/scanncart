import numpy as np
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.schemas import Detection


class _Src:
    width, height, fps = 128, 96, 30.0
    measured_fps = 28.0

    def open(self): return True
    def latest(self): return (1, np.full((96, 128, 3), 130, dtype=np.uint8))
    def release(self): pass


class _Det:
    names = {0: "milo"}
    def infer(self, frame):
        return [Detection(track_id=1, cls="milo", conf=0.9, box=(0.1, 0.1, 0.2, 0.2))]


def _client():
    state = AppState(source_factory=lambda s: _Src(),
                     detector_factory=lambda s, d: _Det(), db_path=":memory:")
    return TestClient(build_app(lambda: state)), state


def test_quality_is_unavailable_before_capture_starts():
    client, _ = _client()
    body = client.get("/api/camera/quality").json()
    assert body["available"] is False


def test_quality_reports_metrics_and_verdicts_while_running():
    client, _ = _client()
    with client:
        client.post("/api/capture/start")
        body = client.get("/api/camera/quality").json()
        client.post("/api/capture/stop")

    assert body["available"] is True
    assert 129 <= body["brightness"] <= 131
    assert body["verdicts"]["brightness"] == "ok"      # 130 is the target
    assert body["capture_fps"] == 28.0


def test_a_dark_frame_is_reported_as_low():
    class _Dark(_Src):
        def latest(self): return (1, np.full((96, 128, 3), 23, dtype=np.uint8))

    state = AppState(source_factory=lambda s: _Dark(),
                     detector_factory=lambda s, d: _Det(), db_path=":memory:")
    client = TestClient(build_app(lambda: state))
    with client:
        client.post("/api/capture/start")
        body = client.get("/api/camera/quality").json()
        client.post("/api/capture/stop")

    assert body["verdicts"]["brightness"] == "low"
