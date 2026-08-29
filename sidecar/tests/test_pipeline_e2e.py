"""End-to-end capture pipeline test using fakes — no camera, no GPU, no network.

Verifies the full flow: POST /api/capture/start → WebSocket frames →
detection logging in SQLite → POST /api/capture/stop, all through the real
FastAPI app with injected fakes.
"""

import numpy as np
from fastapi.testclient import TestClient

from app.hardware import HardwareInfo
from app.main import AppState, build_app
from app.schemas import Detection
from app.settings import Settings


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFrameSource:
    """Always returns a frame from latest() — like the real camera thread."""

    width = 128
    height = 96
    fps = 30.0

    def __init__(self):
        # Counts frames handed to the pipeline, so a skip test can assert the
        # ratio it actually cares about instead of a magic call-count bound.
        self.pulls = 0

    def open(self):
        return True

    def latest(self):
        self.pulls += 1
        return (1, np.full((96, 128, 3), 50, dtype=np.uint8))

    def read(self):
        return np.full((96, 128, 3), 50, dtype=np.uint8)

    def release(self):
        pass


class _FakeDetector:
    """Always returns the same detections — like the StubDetector in test_main.

    Tracks call count for assertions.  When ``sequence`` is provided, the
    detector returns items from it in order and falls back to ``default``
    once exhausted — useful for tests that need to observe drain behaviour.
    """

    _DEFAULT_DETS = [
        Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4)),
    ]

    def __init__(
        self,
        default: list[Detection] | None = None,
        sequence: list[list[Detection]] | None = None,
    ):
        self._default = default if default is not None else self._DEFAULT_DETS
        self._sequence = sequence or []
        self._i = 0
        self.names = {0: "banana", 1: "apple", 2: "milk"}
        self.calls = 0

    def infer(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self._i < len(self._sequence):
            dets = self._sequence[self._i]
            self._i += 1
            return dets
        return self._default


def _fake_hardware() -> HardwareInfo:
    return HardwareInfo(
        cpu_count=8, ram_gb=16.0, cuda_available=False, accelerator="cpu"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    settings: Settings | None = None,
    **detector_kw,
) -> tuple[TestClient, _FakeDetector]:
    """Wire up the full app with fakes and return (client, detector).

    Extra kwargs are forwarded to ``_FakeDetector`` (e.g. ``sequence=``).
    """
    if settings is None:
        settings = Settings()
    detector = _FakeDetector(**detector_kw)

    source = _FakeFrameSource()
    state = AppState(
        settings=settings,
        source_factory=lambda s: source,
        detector_factory=lambda s, d: detector,
        db_path=":memory:",
        hardware_prober=_fake_hardware,
    )
    client = TestClient(build_app(lambda: state))
    client.source = source
    return client, detector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptureE2E:
    """Full capture lifecycle through the REST + WebSocket interface."""

    def test_start_produces_frame_with_jpeg_and_detections(self):
        """Start capture → receive a WebSocket frame → verify it has a
        non-empty JPEG, detection data, and stats."""
        dets = [
            Detection(track_id=None, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4)),
            Detection(track_id=None, cls="apple", conf=0.7, box=(0.5, 0.5, 0.8, 0.9)),
        ]
        client, _ = _make_client(default=dets)

        r = client.post("/api/capture/start")
        assert r.status_code == 200
        assert r.json()["state"] == "running"

        with client.websocket_connect("/ws/stream") as ws:
            msg = ws.receive_json()

        assert msg["type"] == "frame"
        assert msg["seq"] >= 1
        # JPEG payload is non-empty base64
        assert isinstance(msg["jpeg"], str)
        assert len(msg["jpeg"]) > 100
        # Detections passed through
        assert len(msg["detections"]) == 2
        assert msg["detections"][0]["cls"] == "banana"
        assert msg["detections"][1]["cls"] == "apple"
        # Stats present
        assert "infer_fps" in msg["stats"]
        assert "capture_fps" in msg["stats"]
        assert "latency_ms" in msg["stats"]

        client.post("/api/capture/stop")

    def test_start_stop_records_session(self):
        """Each start/stop pair creates a session visible via GET /api/logs."""
        client, _ = _make_client(default=[])

        client.post("/api/capture/start")
        logs = client.get("/api/logs").json()
        assert logs["session_id"] is not None
        client.post("/api/capture/stop")

    def test_detections_logged_with_track_ids(self):
        """When the detector returns track_ids, they appear in the log."""
        dets = [
            Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4)),
        ]
        client, _ = _make_client(default=dets)

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # consume the frame
        client.post("/api/capture/stop")

        logs = client.get("/api/logs").json()
        assert len(logs["events"]) == 1
        assert logs["events"][0]["track_id"] == 1
        assert logs["events"][0]["class_name"] == "banana"
        assert logs["events"][0]["confidence"] == 0.9

    def test_max_conf_tracked_across_frames(self):
        """If the same track_id appears multiple times, max_conf is the max."""
        dets1 = [
            Detection(track_id=5, cls="milk", conf=0.6, box=(0.1, 0.1, 0.4, 0.4)),
        ]
        dets2 = [
            Detection(track_id=5, cls="milk", conf=0.95, box=(0.1, 0.1, 0.4, 0.4)),
        ]
        client, _ = _make_client(default=[], sequence=[dets1, dets2])

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.receive_json()
        client.post("/api/capture/stop")

        logs = client.get("/api/logs").json()
        assert len(logs["events"]) == 1
        assert logs["events"][0]["max_conf"] == 0.95

    def test_no_detections_yields_empty_log(self):
        """Frames with no detections produce no log entries."""
        client, _ = _make_client(default=[])

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.receive_json()
        client.post("/api/capture/stop")

        logs = client.get("/api/logs").json()
        assert logs["events"] == []

    def test_multiple_tracks_logged_independently(self):
        """Different track_ids produce separate log rows."""
        dets = [
            Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.1, 0.3, 0.3)),
            Detection(track_id=2, cls="apple", conf=0.8, box=(0.5, 0.5, 0.8, 0.8)),
            Detection(track_id=3, cls="milk", conf=0.7, box=(0.2, 0.6, 0.5, 0.9)),
        ]
        client, _ = _make_client(default=dets)

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
        client.post("/api/capture/stop")

        logs = client.get("/api/logs").json()
        assert len(logs["events"]) == 3
        classes = {e["class_name"] for e in logs["events"]}
        assert classes == {"banana", "apple", "milk"}

    def test_frame_skip_reduces_inference_calls(self):
        """With infer_frame_skip=1, only every other frame is inferred."""
        settings = Settings(infer_frame_skip=1)
        dets = [
            Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4)),
        ]
        client, detector = _make_client(default=dets, settings=settings)

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            # Receive frames — pipeline only delivers ones that pass inference
            for _ in range(2):
                msg = ws.receive_json()
                assert msg["type"] == "frame"
        client.post("/api/capture/stop")

        # Assert the ratio, not an absolute call count. The pipeline keeps
        # running between the WS closing and the stop landing, so how many
        # extra iterations it gets is pure scheduling luck — the old
        # `calls < 4` bound tripped roughly one run in eight.
        assert detector.calls >= 1
        pulls = client.source.pulls
        assert pulls >= 2
        # skip=1 infers every other pulled frame; allow one for the frame
        # in flight when stop landed.
        assert detector.calls <= pulls // 2 + 1

    def test_stop_resolves_open_tracks(self):
        """When capture stops, open tracks get left_at set."""
        dets = [
            Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4)),
        ]
        client, _ = _make_client(default=dets)

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
        client.post("/api/capture/stop")

        logs = client.get("/api/logs").json()
        assert len(logs["events"]) == 1
        assert logs["events"][0]["left_at"] is not None

    def test_settings_reflect_detector_backend(self):
        """Settings endpoint returns detector backend fields."""
        client, _ = _make_client(default=[])
        r = client.get("/api/settings")
        assert r.status_code == 200
        body = r.json()
        assert body["active_model"] == "yolo11n.pt"
        assert body["detector_backend"] == "native"
        assert "hot_reloadable_fields" in body
        assert "restart_required_fields" in body

    def test_health_reports_model_and_device(self):
        """Health endpoint reflects the active model and device."""
        client, _ = _make_client(default=[])
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "idle"
        assert body["active_model"] == "yolo11n.pt"

    def test_capture_start_stop_lifecycle(self):
        """Verify the full state transitions: idle → running → idle."""
        client, _ = _make_client(default=[])

        assert client.get("/api/health").json()["state"] == "idle"
        assert client.post("/api/capture/start").json()["state"] == "running"
        assert client.get("/api/health").json()["state"] == "running"
        assert client.post("/api/capture/stop").json()["state"] == "idle"
        assert client.get("/api/health").json()["state"] == "idle"
