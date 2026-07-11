from app.pipeline import Pipeline
from app.settings import Settings
from app.schemas import Detection


class FakeStore:
    def __init__(self):
        self.records = []   # (session_id, track_id, cls, conf, ts)
        self.resolved = []  # (session_id, track_id, ts)

    def record_detection(self, session_id, track_id, cls, conf, ts):
        self.records.append((session_id, track_id, cls, conf, ts))

    def resolve_left(self, session_id, track_id, ts):
        self.resolved.append((session_id, track_id, ts))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ScriptedSource:
    width = 128
    height = 96
    fps = 30.0

    def latest(self):
        import numpy as np
        return (1, np.full((96, 128, 3), 50, dtype=np.uint8))


class ScriptedDetector:
    """Returns whatever detection list is queued for the next infer() call."""
    names = {0: "banana"}

    def __init__(self, script):
        self._script = list(script)

    def infer(self, frame):
        return self._script.pop(0) if self._script else []


def _pipe(script, store, clock, expiry=1.5):
    return Pipeline(
        ScriptedSource(),
        ScriptedDetector(script),
        Settings(track_expiry_s=expiry),
        on_message=lambda m: None,
        logging_store=store,
        session_id=42,
        track_expiry_s=expiry,
        clock=clock,
    )


def test_new_track_is_recorded_once_per_frame():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det, det], store, clock)
    pipe.process_once()
    pipe.process_once()
    assert [r[1] for r in store.records] == [5, 5]   # recorded each frame; store dedups
    assert store.resolved == []                       # still present, not resolved


def test_untracked_detection_is_not_logged():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=None, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det], store, clock)
    pipe.process_once()
    assert store.records == []


def test_track_is_resolved_after_expiry():
    store, clock = FakeStore(), FakeClock()
    seen = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([seen, [], []], store, clock, expiry=1.0)
    clock.t = 0.0
    pipe.process_once()          # track 5 seen at t=0
    clock.t = 0.5
    pipe.process_once()          # empty; 0.5s gap < expiry → not resolved
    assert store.resolved == []
    clock.t = 2.0
    pipe.process_once()          # empty; 2.0s since last-seen > expiry → resolved
    assert store.resolved == [(42, 5, 0.0)]   # left_at = last-seen time


def test_resolve_open_tracks_flushes_remaining():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=8, cls="apple", conf=0.9, box=(0, 0, 0.5, 0.5))]
    pipe = _pipe([det], store, clock)
    clock.t = 3.0
    pipe.process_once()
    pipe.resolve_open_tracks()
    assert store.resolved == [(42, 8, 3.0)]
