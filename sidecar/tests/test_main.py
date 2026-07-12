import numpy as np
from fastapi.testclient import TestClient
from app.main import build_app, AppState
from app.schemas import Detection
from app.settings import Settings


class _StubSource:
    width = 128
    height = 96
    fps = 30.0

    def open(self):
        return True

    def latest(self):
        return (1, np.full((96, 128, 3), 50, dtype=np.uint8))

    def read(self):
        return np.full((96, 128, 3), 50, dtype=np.uint8)

    def release(self):
        pass


class _StubDetector:
    names = {0: "banana"}

    def infer(self, frame):
        return [Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]


def _make_client():
    state = AppState(
        settings=Settings(),
        source_factory=lambda settings: _StubSource(),
        detector_factory=lambda settings, device: _StubDetector(),
        db_path=":memory:",
    )
    return TestClient(build_app(lambda: state)), state


def test_health_reports_idle_and_model():
    client, _ = _make_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["active_model"] == "yolo11n.pt"
    assert body["device"] in ("cpu", "cuda")


def test_start_then_stop_transitions_state():
    client, _ = _make_client()
    assert client.post("/api/capture/start").json()["state"] == "running"
    assert client.get("/api/health").json()["state"] == "running"
    assert client.post("/api/capture/stop").json()["state"] == "idle"


def test_cross_origin_requests_get_cors_headers():
    # The renderer's origin (Vite dev server port, or a packaged app's
    # file:// origin) never matches this server's http://127.0.0.1:<port>
    # origin, so without CORS headers the browser blocks the renderer from
    # reading the response body even though the request succeeds server-side.
    client, _ = _make_client()
    r = client.get("/api/settings", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"


def test_websocket_receives_a_frame_after_start():
    client, _ = _make_client()
    client.post("/api/capture/start")
    with client.websocket_connect("/ws/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        assert msg["detections"][0]["cls"] == "banana"
    client.post("/api/capture/stop")


def test_system_info_reports_accelerator():
    client, _ = _make_client()
    r = client.get("/api/system-info")
    assert r.status_code == 200
    assert r.json()["accelerator"] in {"cuda", "integrated", "cpu"}
