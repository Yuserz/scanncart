import numpy as np
from fastapi.testclient import TestClient
from app.main import build_app, AppState
from app.schemas import Detection
from app.settings import Settings

from tests import next_frame


class _StubSource:
    width, height, fps = 128, 96, 30.0

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


def _client():
    state = AppState(
        settings=Settings(),
        source_factory=lambda s: _StubSource(),
        detector_factory=lambda s, d: _StubDetector(),
        db_path=":memory:",
    )
    return TestClient(build_app(lambda: state))


def test_logs_empty_before_any_session():
    r = _client().get("/api/logs")
    assert r.status_code == 200
    assert r.json() == {"session_id": None, "events": []}


def test_logs_report_current_session_events_after_a_run():
    client = _client()
    client.post("/api/capture/start")
    # Pull a frame so the pipeline records at least one detection.
    with client.websocket_connect("/ws/stream") as ws:
        next_frame(ws)
    client.post("/api/capture/stop")

    body = client.get("/api/logs").json()
    assert body["session_id"] == 1
    assert len(body["events"]) == 1
    ev = body["events"][0]
    assert ev["track_id"] == 1
    assert ev["class_name"] == "banana"
    assert ev["max_conf"] == 0.9
    assert ev["left_at"] is not None   # capture/stop resolved the open track


def test_logs_since_filters_out_tracks_that_left_before_it():
    state = AppState(
        settings=Settings(),
        source_factory=lambda s: _StubSource(),
        detector_factory=lambda s, d: _StubDetector(),
        db_path=":memory:",
    )
    store = state.logging_store
    sid = store.start_session("m", "cpu")
    store.record_detection(sid, 1, "banana", 0.9, ts=1.0)
    store.resolve_left(sid, 1, ts=2.0)
    store.record_detection(sid, 2, "banana", 0.9, ts=3.0)
    client = TestClient(build_app(lambda: state))

    assert [e["track_id"] for e in client.get("/api/logs").json()["events"]] == [1, 2]
    body = client.get("/api/logs", params={"since": 2.5}).json()
    assert body["session_id"] == sid
    assert [e["track_id"] for e in body["events"]] == [2]


def test_logs_since_rejects_a_non_number():
    assert _client().get("/api/logs", params={"since": "soon"}).status_code == 422
