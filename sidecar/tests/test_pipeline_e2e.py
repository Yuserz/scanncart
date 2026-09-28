"""End-to-end capture pipeline test using fakes — no camera, no GPU, no network.

Verifies the full flow: POST /api/capture/start → WebSocket frames →
detection logging in SQLite → POST /api/capture/stop, all through the real
FastAPI app with injected fakes.
"""

import numpy as np
from fastapi.testclient import TestClient

from tests import next_frame

from app.hardware import HardwareInfo
from app.main import AppState, build_app
from app.roster import V2_ROSTER
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
    ``names`` is the class list it declares — the thing a roster verdict and a
    ``class_allowlist`` are about — and defaults to the fake classes above.
    """

    _DEFAULT_DETS = [
        Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4)),
    ]

    def __init__(
        self,
        default: list[Detection] | None = None,
        sequence: list[list[Detection]] | None = None,
        names: dict[int, str] | None = None,
    ):
        self._default = default if default is not None else self._DEFAULT_DETS
        self._sequence = sequence or []
        self._i = 0
        self.names = dict(names) if names is not None else {0: "banana", 1: "apple", 2: "milk"}
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


def _roster_names() -> dict[int, str]:
    """v2's roster in the shape a detector declares it (`YoloDetector.names`): index -> name."""
    return dict(enumerate(V2_ROSTER))


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
            msg = next_frame(ws)

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
            next_frame(ws)  # pull one frame, so the detection is logged
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
            # Two frames, because the point is that the same track's confidence is *tracked*
            # across them. Counting messages instead of frames would now consume the handshake
            # status as one of the two and measure a single frame.
            next_frame(ws)
            next_frame(ws)
        client.post("/api/capture/stop")

        logs = client.get("/api/logs").json()
        assert len(logs["events"]) == 1
        assert logs["events"][0]["max_conf"] == 0.95

    def test_no_detections_yields_empty_log(self):
        """Frames with no detections produce no log entries."""
        client, _ = _make_client(default=[])

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            next_frame(ws)
            next_frame(ws)
        client.post("/api/capture/stop")

        logs = client.get("/api/logs").json()
        assert logs["events"] == []

    def test_a_feed_that_matches_nothing_streams_and_logs_nothing(self):
        """The empty-counter case on the one path that can produce it: the model reports boxes and
        every one of them is off the class allowlist, so nothing that matches is left.

        Its promise is that the preview keeps running - an operator still has to aim the camera at
        the counter - while the item log stays empty and the session is still recorded. Driven
        through the real surface: the WS stream and the SQLite-backed `/api/logs`. The roster is
        what "matches" means here: the detector declares v2's seven, the dropped boxes name none
        of them, and the stream says which names were in force.
        """
        dets = [
            Detection(track_id=1, cls="person", conf=0.93, box=(0.1, 0.2, 0.3, 0.4)),
            Detection(track_id=2, cls="tv", conf=0.81, box=(0.4, 0.4, 0.7, 0.7)),
        ]
        client, _ = _make_client(
            default=dets,
            settings=Settings(class_allowlist=list(V2_ROSTER)),
            names=_roster_names(),
        )

        frames: list[dict] = []
        announced = None
        states: set[str] = set()
        with client.websocket_connect("/ws/stream") as ws:
            # Connected *before* start, so the class-list broadcast reaches this client on the
            # stream rather than being folded into a handshake it did not take.
            assert ws.receive_json()["state"] == "idle"  # the handshake, before any capture
            assert client.post("/api/capture/start").json()["state"] == "running"
            assert client.get("/api/health").json()["state"] == "running"
            # A session ran, even though nothing in it matched: the log being empty is a fact
            # about the feed, not a missing session.
            assert client.get("/api/logs").json()["session_id"] is not None
            # Bounded rather than "receive until satisfied": a broadcast that stops arriving has
            # to fail the assertions below, not hang the suite.
            for _ in range(50):
                msg = ws.receive_json()
                if msg["type"] == "frame":
                    frames.append(msg)
                elif msg["type"] == "status":
                    states.add(msg.get("state") or "")
                    if msg.get("class_names"):
                        announced = msg["class_names"]
                if len(frames) >= 2 and announced:
                    break

        assert announced == sorted(V2_ROSTER)
        assert "error" not in states  # an empty feed is not an error, and must not become one
        for msg in frames:
            assert msg["detections"] == []
            # The preview is the behavior, not a side effect: a rendered JPEG with no boxes, on
            # every frame, so the operator can see the counter they are aiming at.
            assert isinstance(msg["jpeg"], str) and len(msg["jpeg"]) > 100
            assert msg["stats"]["suppressed"] == 0

        assert client.post("/api/capture/stop").json()["state"] == "idle"
        logs = client.get("/api/logs").json()
        assert logs["session_id"] is not None
        assert logs["events"] == []
        # And the app is still serving afterwards, rather than wedged on the empty feed.
        assert client.get("/api/health").json()["state"] == "idle"

    def test_the_default_allowlist_filters_nothing_not_even_a_non_roster_class(self):
        """The default is *no* class filter: the app draws and logs what the model reports, roster
        or not. That is why the empty case above has to set the allowlist to reach its state - and
        pinning it here is what keeps a future "auto-filter to the roster" from silently changing
        every existing capture, which the class-list verdict warns about but does not do.
        """
        dets = [Detection(track_id=1, cls="person", conf=0.91, box=(0.1, 0.2, 0.3, 0.4))]
        client, _ = _make_client(default=dets, names=_roster_names())

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            msg = next_frame(ws)
        client.post("/api/capture/stop")

        assert [d["cls"] for d in msg["detections"]] == ["person"]
        events = client.get("/api/logs").json()["events"]
        assert [(e["class_name"], e["confidence"]) for e in events] == [("person", 0.91)]

    def test_a_v2_roster_class_passes_the_allowlist_that_empties_the_non_roster_feed(self):
        """The other half of the filter, so a filter that dropped *everything* cannot pass the
        empty-feed test: a name the roster does carry is streamed and logged under the same
        allowlist, and logged under its own name rather than remapped to a roster position.
        """
        name = V2_ROSTER[0]
        dets = [Detection(track_id=7, cls=name, conf=0.88, box=(0.2, 0.2, 0.6, 0.8))]
        client, _ = _make_client(
            default=dets,
            settings=Settings(class_allowlist=list(V2_ROSTER)),
            names=_roster_names(),
        )

        client.post("/api/capture/start")
        with client.websocket_connect("/ws/stream") as ws:
            msg = next_frame(ws)
        client.post("/api/capture/stop")

        assert [d["cls"] for d in msg["detections"]] == [name]
        events = client.get("/api/logs").json()["events"]
        assert [(e["class_name"], e["confidence"]) for e in events] == [(name, 0.88)]

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
            next_frame(ws)
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
                next_frame(ws)
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
            next_frame(ws)
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
        assert body["active_model"] == Settings().active_model
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
        assert body["active_model"] == Settings().active_model

    def test_capture_start_stop_lifecycle(self):
        """Verify the full state transitions: idle → running → idle."""
        client, _ = _make_client(default=[])

        assert client.get("/api/health").json()["state"] == "idle"
        assert client.post("/api/capture/start").json()["state"] == "running"
        assert client.get("/api/health").json()["state"] == "running"
        assert client.post("/api/capture/stop").json()["state"] == "idle"
        assert client.get("/api/health").json()["state"] == "idle"
