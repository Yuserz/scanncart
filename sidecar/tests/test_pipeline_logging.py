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


#: The box the empty counter actually produced at 0.957 conf, and the shape the frame-clamp rule
#: suppresses (see `app/pipeline.py::CLAMPED_EDGE_TOLERANCE` for how the boundary was chosen).
_PHANTOM = (0.0002, 0.0001, 0.9996, 1.0)


def test_a_phantom_never_becomes_an_item_log_row():
    """The whole point of the suppression, stated where it is actually felt.

    The phantom held for 16 consecutive frames in the live capture, so it is a *persistent track*:
    dropping it at the detector's output is what stops one keystroke of work turning into one row in
    the item log per session. The row is the user-visible artifact, so the assertion belongs here
    rather than only on the frame message.
    """
    store, clock = FakeStore(), FakeClock()
    phantom = [Detection(track_id=7, cls="Bear Brand Fortified Powdered Milk 33g", conf=0.957,
                         box=_PHANTOM)]
    pipe = _pipe([phantom, phantom, phantom], store, clock)

    for _ in range(3):
        clock.t += 0.1
        pipe.process_once()

    assert store.records == []
    # Nothing was created, so nothing has to be resolved later either: a suppressed detection
    # should leave no trace at all, not a track that opens and then expires.
    assert store.resolved == []


def test_a_real_detection_beside_a_phantom_is_still_logged():
    """The filter drops a *box*, not a frame.

    A crowded counter is exactly where the phantom and real items coexist, so a rule that discarded
    the whole frame's output on seeing one clamped box would throw away the items that matter.
    """
    store, clock = FakeStore(), FakeClock()
    mixed = [
        Detection(track_id=7, cls="Bear Brand Fortified Powdered Milk 33g", conf=0.957, box=_PHANTOM),
        Detection(track_id=8, cls="Milo Chocolate Drink 22g Sachet", conf=0.91,
                  box=(0.2, 0.3, 0.5, 0.7)),
    ]
    pipe = _pipe([mixed], store, clock)
    pipe.process_once()

    assert [(r[1], r[2]) for r in store.records] == [(8, "Milo Chocolate Drink 22g Sachet")]


def test_turning_the_suppression_off_logs_the_phantom_again():
    """The escape hatch, on the path an operator would use it: if a real item really does fill the
    frame, turning the rule off has to put its row back."""
    store, clock = FakeStore(), FakeClock()
    settings = Settings(suppress_clamped_detections=False)
    pipe = Pipeline(
        ScriptedSource(),
        ScriptedDetector([[Detection(track_id=7, cls="banana", conf=0.957, box=_PHANTOM)]]),
        settings,
        on_message=lambda m: None,
        logging_store=store,
        session_id=42,
        clock=clock,
    )
    pipe.process_once()

    assert [r[1] for r in store.records] == [7]


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


def test_track_expiry_s_is_live_reloaded_from_settings():
    # settings.track_expiry_s is mutated in place mid-run (as a real settings
    # PATCH would do) — the pipeline must pick it up on the very next frame,
    # not just at construction time.
    store, clock = FakeStore(), FakeClock()
    settings = Settings(track_expiry_s=10.0)
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


def test_class_allowlist_drops_disallowed_classes():
    store, clock = FakeStore(), FakeClock()
    keep = Detection(track_id=5, cls="banana", conf=0.9, box=(0.1, 0.2, 0.3, 0.4))
    drop = Detection(track_id=6, cls="person", conf=0.95, box=(0, 0, 0.5, 0.5))
    pipe = Pipeline(
        ScriptedSource(),
        ScriptedDetector([[keep, drop], [keep, drop]]),
        Settings(class_allowlist=["banana"]),
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
    settings = Settings()  # empty allowlist = keep all
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
