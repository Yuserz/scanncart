import numpy as np
from fastapi.testclient import TestClient

from tests import next_frame
from app.main import build_app, AppState
from app.roster import V1_ROSTER, V2_ROSTER
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


class _DetectorDeclaring:
    """A detector that predicts exactly the class names it was handed.

    The realistic shape for this: a detector's `names` are only knowable after an inference (a
    native one fills them from the first result on purpose, so reading them at construction cannot
    build a second ONNX session), which is why the report waits inside the pipeline rather than
    being asked for at start.
    """

    def __init__(self, names: list[str]):
        self.names = {i: n for i, n in enumerate(names)}

    def infer(self, frame):
        first = self.names[0]
        return [Detection(track_id=1, cls=first, conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]

    def close(self):
        pass


class _DyingDetector:
    """Stands in for any failure on the pipeline thread - the wedged camera, a dead inference
    server - which the pipeline reports through `on_error` rather than raising into the void.

    It fails on a timer rather than on the first call, which is what the real thing does: the
    StreamCam wedge takes ~3 s to declare itself, so the start handler has long finished by the
    time the death is reported. Raising instantly would instead race that handler and test a
    lifecycle the hardware cannot produce.
    """

    names = {0: "banana"}

    def infer(self, frame):
        import time

        time.sleep(0.2)
        raise RuntimeError("Camera 0 stopped delivering frames for 3s")


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _dying_client(detectors=None):
    pool = iter(detectors or [_DyingDetector()])
    state = AppState(
        settings=Settings(),
        source_factory=lambda settings: _StubSource(),
        detector_factory=lambda settings, device: next(pool),
        db_path=":memory:",
        hardware_prober=_fake_hardware,
    )
    return TestClient(build_app(lambda: state)), state


def _fake_hardware():
    from app.hardware import HardwareInfo

    return HardwareInfo(cpu_count=8, ram_gb=16.0, cuda_available=False, accelerator="cpu")


def _make_client(detector=None):
    """The app with a stub source and, by default, the stub detector.

    `detector` exists because the class list a model declares is now part of the running state:
    `_StubDetector` predicts `banana`, which is not in this app's roster, so any test that wants to
    say something about a *clean* class list has to supply a detector that predicts the roster.
    """
    factory = (lambda settings, device: detector) if detector is not None else _stub_detector
    state = AppState(
        settings=Settings(),
        source_factory=lambda settings: _StubSource(),
        detector_factory=factory,
        db_path=":memory:",
        hardware_prober=_fake_hardware,
    )
    return TestClient(build_app(lambda: state)), state


def _stub_detector(settings=None, device=None):
    return _StubDetector()


def test_health_reports_idle_and_model():
    client, _ = _make_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["active_model"] == Settings().active_model
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
        msg = next_frame(ws)
        assert msg["detections"][0]["cls"] == "banana"
    client.post("/api/capture/stop")


def test_the_class_list_reaches_a_client_that_is_already_watching():
    """The broadcast, as distinct from the handshake, and the reason it is unconditional.

    A renderer that is open when a capture starts never sees the handshake the server sends to a
    *new* connection, so the names have to arrive on the stream as they are learned - otherwise the
    stats strip's class chip would populate only after a reload. This is the path the earlier "a
    clean list is not news" rule skipped, and it is exactly the healthy case the chip exists for.
    """
    client, _ = _make_client(detector=_DetectorDeclaring(list(V2_ROSTER)))

    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["state"] == "idle"
        client.post("/api/capture/start")
        # Frames come interleaved with status messages, so the search is for the message rather
        # than for the next one to arrive.
        reported = None
        for _ in range(50):
            msg = ws.receive_json()
            if msg["type"] == "status" and msg["class_names"]:
                reported = msg
                break
        assert reported is not None, "no status message carried the class list"
        assert reported["class_names"] == sorted(V2_ROSTER)
        # Reported with an empty verdict, which is the whole point: the count is the news.
        assert reported["class_warnings"] == []
    client.post("/api/capture/stop")


def test_a_new_websocket_is_told_the_state_it_connected_into():
    """The handshake, which is what a reloaded renderer has to go on.

    Until this existed the only status messages a client ever received were its own start/stop
    responses and pipeline errors, so a renderer that connected to an already-running capture
    showed `idle` - a Start button over streaming frames - and skipped the log recovery it gates
    on `running`. The state is read at handshake time, so it describes the capture the client is
    actually joining.
    """
    client, _ = _make_client()
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json() == {
            "type": "status",
            "state": "idle",
            "detail": "",
            # Nothing is loaded, so nothing is known about a class list. An empty list of names is
            # "not known yet" - never "predicts nothing" - and it is what keeps the stats strip's
            # class chip absent rather than showing a zero.
            "class_names": [],
            # Empty here means "no finding", which is the same thing a clean list reports - see
            # `class_list_problems`.
            "class_warnings": [],
        }

    client.post("/api/capture/start")
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["state"] == "running"
        # And the snapshot does not crowd out the stream behind it.
        assert next_frame(ws)["detections"][0]["cls"] == "banana"
    client.post("/api/capture/stop")

    # After the stop, a later connection is told idle again rather than the stale state.
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["state"] == "idle"


def test_a_client_joining_a_run_with_a_bad_class_list_is_told_about_it():
    """The runtime half of the roster guard, and why it is on the *stream* rather than the probe.

    A probe is a check somebody has to think to press; a 24-class weight is a fact about the model
    that is running. The pipeline is where that becomes knowable (the detector's `names` are filled
    from an inference, not from construction), so the verdict is broadcast when it is learned and
    replayed in the handshake - which is the only way a client that connects *after* that moment
    can learn it, exactly as `last_error` covers a capture that died before a client arrived.
    """
    names = [f"{n} {d}" for n in ("milo", "safeguard") for d in ("close", "mid", "far")]
    client, state = _make_client(detector=_DetectorDeclaring(names))
    assert client.post("/api/capture/start").json()["state"] == "running"
    assert state.class_warnings, "the pipeline should have reported the class list by now"

    with client.websocket_connect("/ws/stream") as ws:
        snapshot = ws.receive_json()

    assert len(snapshot["class_warnings"]) == 1
    assert "carry a distance" in snapshot["class_warnings"][0]
    assert "one class per product-and-distance" in snapshot["class_warnings"][0]
    # The names ride with the verdict, not instead of it: the count is a readout in its own right,
    # and the strip can only show it if the list it counts arrives.
    assert snapshot["class_names"] == sorted(names)
    client.post("/api/capture/stop")

    # And a stop clears it: nothing is loaded, so a stale warning would describe a model that is
    # no longer there.
    assert (state.class_names, state.class_warnings) == ([], [])
    with client.websocket_connect("/ws/stream") as ws:
        snapshot = ws.receive_json()
    assert snapshot["class_warnings"] == [] and snapshot["class_names"] == []


def test_a_clean_class_list_reports_its_names_and_no_warnings():
    """Two claims, and they are different ones.

    The verdict is empty - which is the false alarm the once-only report must never raise, since a
    detector that has not inferred yet has `names == {}` and reporting that as "this model predicts
    none of the 8 roster classes" would fire on every healthy capture. And the *names* are still
    reported, because the stats strip's class count has to appear on a healthy capture too; the
    earlier "a clean list is not news" rule was written when the warning banner was the only
    consumer, and it would leave the count at zero on exactly the captures that are fine.
    """
    client, state = _make_client(detector=_DetectorDeclaring(list(V2_ROSTER)))

    assert client.post("/api/capture/start").json()["state"] == "running"
    assert state.class_names == sorted(V2_ROSTER)
    assert state.class_warnings == []

    with client.websocket_connect("/ws/stream") as ws:
        snapshot = ws.receive_json()

    assert snapshot["class_names"] == sorted(V2_ROSTER)
    assert snapshot["class_warnings"] == []
    client.post("/api/capture/stop")


def test_the_running_weights_recorded_generation_decides_its_roster(monkeypatch):
    """The wiring the per-generation split hangs on, at the point it matters: the *runtime* judge
    consults the record.

    A v2 head that lost Palmolive declares exactly v1's seven names, so its vocabulary alone
    cannot say whether the missing class is a fault or the correct list for the generation - the
    record is the only side that knows (`roster.resolve_roster`). Here a detector declares v1's
    seven and the record claims v2, which is that head exactly: the finding has to appear.
    Monkeypatched rather than read off disk, because the real `models/` directory is state this
    suite must not assume.
    """
    monkeypatch.setattr("app.main.generation_for", lambda *a, **k: "v2")
    client, state = _make_client(detector=_DetectorDeclaring(list(V1_ROSTER)))

    assert client.post("/api/capture/start").json()["state"] == "running"
    assert state.class_names == sorted(V1_ROSTER)
    (warning,) = state.class_warnings
    assert "cannot predict 1 of the 8 v2 roster classes" in warning
    assert "Palmolive Naturals Bar Soap 85g" in warning


def test_a_v1_weights_own_record_leaves_its_seven_classes_alone():
    """The same wiring, the other way - and the case the app is actually in: the installed v1
    weight's record says `generation: v1`, so its seven names are judged against v1's roster and
    nothing is reported. Held to v2 it produced a Palmolive finding on every capture."""
    client, state = _make_client(detector=_DetectorDeclaring(list(V1_ROSTER)))

    assert client.post("/api/capture/start").json()["state"] == "running"
    assert state.class_names == sorted(V1_ROSTER)
    assert state.class_warnings == []
    client.post("/api/capture/stop")


def test_a_capture_that_died_is_explained_to_the_client_that_arrives_later():
    """The stored reason, sent on connect.

    The error status explains the death to whoever is connected when it happens, and nobody else
    can: teardown leaves `state` at `idle`, which is exactly what a capture that was never started
    looks like. So a reload after a dead capture used to show a frozen preview and no reason for
    it - the one case where the operator most needs the sentence.
    """
    client, state = _dying_client()
    assert client.post("/api/capture/start").json()["state"] == "running"
    # Wait for the pipeline thread to report and tear down, so this is the *after* state.
    assert _wait_until(lambda: state.last_error is not None and state.state == "idle")

    with client.websocket_connect("/ws/stream") as ws:
        snapshot = ws.receive_json()
    assert snapshot["type"] == "status"
    assert snapshot["state"] == "idle"
    assert snapshot["detail"].startswith("Capture stopped:")
    assert "stopped delivering frames" in snapshot["detail"]


def test_starting_a_new_capture_clears_the_stored_reason():
    """The stored reason describes a capture that is gone. Once another one is running it is
    history, and leaving it would put the old failure's explanation over a working capture the
    next time anyone reloaded."""
    client, state = _dying_client([_DyingDetector(), _StubDetector()])
    assert client.post("/api/capture/start").json()["state"] == "running"
    assert _wait_until(lambda: state.last_error is not None)

    assert client.post("/api/capture/start").json()["state"] == "running"
    with client.websocket_connect("/ws/stream") as ws:
        snapshot = ws.receive_json()
    assert snapshot["state"] == "running"
    assert snapshot["detail"] == ""
    client.post("/api/capture/stop")


def test_system_info_reports_accelerator():
    client, _ = _make_client()
    r = client.get("/api/system-info")
    assert r.status_code == 200
    assert r.json()["accelerator"] in {"cuda", "integrated", "cpu"}
