from app.schemas import Detection, Stats, FrameMessage, StatusMessage, HealthResponse


def test_frame_message_serializes():
    msg = FrameMessage(
        type="frame",
        ts=1720598400.123,
        seq=42,
        jpeg="Zm9v",
        detections=[Detection(track_id=1, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))],
        stats=Stats(infer_fps=22.4, capture_fps=60.0, latency_ms=88.0),
    )
    dumped = msg.model_dump()
    assert dumped["type"] == "frame"
    assert dumped["detections"][0]["cls"] == "banana"
    assert dumped["detections"][0]["box"] == (0.1, 0.2, 0.3, 0.4)
    assert dumped["stats"]["infer_fps"] == 22.4


def test_detection_allows_null_track_id():
    d = Detection(track_id=None, cls="apple", conf=0.5, box=(0.0, 0.0, 1.0, 1.0))
    assert d.track_id is None


def test_status_message_default_detail():
    s = StatusMessage(type="status", state="running")
    assert s.detail == ""


def test_health_response_fields():
    h = HealthResponse(state="idle", active_model="yolo11n.pt", device="cpu")
    assert h.active_model == "yolo11n.pt"
