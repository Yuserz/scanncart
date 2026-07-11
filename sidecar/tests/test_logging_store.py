from app.logging_store import LoggingStore, EventRow


def _store():
    # ":memory:" keeps one connection alive for the store's lifetime.
    return LoggingStore(":memory:")


def test_start_session_returns_incrementing_ids():
    s = _store()
    a = s.start_session("yolo11n.pt", "cpu")
    b = s.start_session("yolo11n.pt", "cpu")
    assert a == 1 and b == 2
    assert s.current_session_id() == 2


def test_record_detection_inserts_one_row_per_track():
    s = _store()
    sid = s.start_session("yolo11n.pt", "cpu")
    s.record_detection(sid, 7, "banana", 0.80, ts=100.0)
    s.record_detection(sid, 7, "banana", 0.95, ts=100.2)  # update
    s.record_detection(sid, 7, "banana", 0.60, ts=100.4)  # lower, ignored for max
    rows = s.query_events(sid)
    assert len(rows) == 1
    r = rows[0]
    assert r.confidence == 0.80          # frozen at first sighting
    assert r.entered_at == 100.0         # frozen at first sighting
    assert r.max_conf == 0.95            # best seen
    assert r.left_at is None


def test_resolve_left_sets_left_at_once():
    s = _store()
    sid = s.start_session("yolo11n.pt", "cpu")
    s.record_detection(sid, 3, "apple", 0.9, ts=10.0)
    s.resolve_left(sid, 3, ts=12.5)
    s.resolve_left(sid, 3, ts=99.0)      # must not overwrite an already-set left_at
    rows = s.query_events(sid)
    assert rows[0].left_at == 12.5


def test_query_events_scoped_to_session_and_ordered():
    s = _store()
    s1 = s.start_session("m", "cpu")
    s.record_detection(s1, 1, "banana", 0.9, ts=5.0)
    s2 = s.start_session("m", "cpu")
    s.record_detection(s2, 2, "apple", 0.9, ts=2.0)
    s.record_detection(s2, 3, "orange", 0.9, ts=1.0)
    assert [r.track_id for r in s.query_events(s1)] == [1]
    assert [r.track_id for r in s.query_events(s2)] == [3, 2]  # ordered by entered_at


def test_current_session_id_none_when_empty():
    assert _store().current_session_id() is None
