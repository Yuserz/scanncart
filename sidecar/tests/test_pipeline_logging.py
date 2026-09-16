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


def _pipe(script, store, clock, expiry=1.5, confirm=2):
    return Pipeline(
        ScriptedSource(),
        ScriptedDetector(script),
        Settings(track_expiry_s=expiry, track_confirm_hits=confirm),
        on_message=lambda m: None,
        logging_store=store,
        session_id=42,
        clock=clock,
    )


def test_track_requires_two_consecutive_hits_by_default():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det, det, det], store, clock)
    pipe.process_once()   # first sighting -> pending, not logged
    assert store.records == []
    pipe.process_once()   # second consecutive sighting -> confirmed
    assert [r[1] for r in store.records] == [5]
    pipe.process_once()   # still present -> logged each frame (store dedups)
    assert [r[1] for r in store.records] == [5, 5]
    assert store.resolved == []                       # still present, not resolved


def test_single_frame_phantom_is_never_logged():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det, [], []], store, clock, expiry=10.0)
    pipe.process_once()
    pipe.process_once()  # phantom gone before confirming
    pipe.process_once()
    assert store.records == []
    assert store.resolved == []  # never logged -> never resolved either


def test_confirm_hits_gap_resets_counter():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det, [], det, det], store, clock, expiry=10.0)
    pipe.process_once()  # hit 1
    pipe.process_once()  # gone -> pending cleared
    pipe.process_once()  # re-appears -> hit 1 again (fresh)
    assert store.records == []
    pipe.process_once()  # hit 2 -> confirmed
    assert [r[1] for r in store.records] == [5]


def test_confirm_hits_of_one_logs_first_sighting():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det, det], store, clock, confirm=1)
    pipe.process_once()
    assert [r[1] for r in store.records] == [5]


def test_class_allowlist_drops_disallowed_classes():
    store, clock = FakeStore(), FakeClock()
    settings = Settings(track_confirm_hits=1, class_allowlist=["banana"])
    keep = Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))
    drop = Detection(track_id=6, cls="person", conf=0.95, box=(0, 0, 0.5, 0.5))
    pipe = Pipeline(
        ScriptedSource(),
        ScriptedDetector([[keep, drop], [keep, drop]]),
        settings,
        on_message=lambda m: None,
        logging_store=store,
        session_id=42,
        clock=clock,
    )
    pipe.process_once()
    pipe.process_once()
    assert [r[2] for r in store.records] == ["banana", "banana"]


def test_class_allowlist_is_live_reloaded():
    store, clock = FakeStore(), FakeClock()
    settings = Settings(track_confirm_hits=1)  # empty allowlist = keep all
    det = Detection(track_id=5, cls="person", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))
    pipe = Pipeline(
        ScriptedSource(),
        ScriptedDetector([[det], [det]]),
        settings,
        on_message=lambda m: None,
        logging_store=store,
        session_id=42,
        clock=clock,
    )
    pipe.process_once()
    assert [r[2] for r in store.records] == ["person"]
    settings.class_allowlist = ["banana"]  # mutate in place, as a live PATCH would
    pipe.process_once()
    assert [r[2] for r in store.records] == ["person"]  # nothing new logged


def test_empty_class_allowlist_keeps_everything():
    store, clock = FakeStore(), FakeClock()
    det = Detection(track_id=5, cls="person", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))
    pipe = _pipe([[det]], store, clock, confirm=1)
    pipe.process_once()
    assert [r[2] for r in store.records] == ["person"]


def test_untracked_detection_is_not_logged():
    store, clock = FakeStore(), FakeClock()
    det = [Detection(track_id=None, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([det], store, clock)
    pipe.process_once()
    assert store.records == []


def test_track_is_resolved_after_expiry():
    store, clock = FakeStore(), FakeClock()
    seen = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = _pipe([seen, [], []], store, clock, expiry=1.0, confirm=1)
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
    pipe = _pipe([det], store, clock, confirm=1)
    clock.t = 3.0
    pipe.process_once()
    pipe.resolve_open_tracks()
    assert store.resolved == [(42, 8, 3.0)]


def test_track_expiry_s_is_live_reloaded_from_settings():
    # settings.track_expiry_s is mutated in place mid-run (as a real settings
    # PATCH would do) — the pipeline must pick it up on the very next frame,
    # not just at construction time.
    store, clock = FakeStore(), FakeClock()
    settings = Settings(track_expiry_s=10.0, track_confirm_hits=1)
    seen = [Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))]
    pipe = Pipeline(
        ScriptedSource(),
        ScriptedDetector([seen, [], []]),
        settings,
        on_message=lambda m: None,
        logging_store=store,
        session_id=42,
        clock=clock,
    )
    clock.t = 0.0
    pipe.process_once()          # track 5 seen at t=0
    settings.track_expiry_s = 1.0   # shrink expiry in place after construction
    clock.t = 0.5
    pipe.process_once()          # 0.5s gap < new 1.0s expiry → not yet resolved
    assert store.resolved == []
    clock.t = 2.0
    pipe.process_once()          # 2.0s gap > new 1.0s expiry → resolved
    assert store.resolved == [(42, 5, 0.0)]
