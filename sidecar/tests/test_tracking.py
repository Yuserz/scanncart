import pytest

from app.schemas import Detection
from app.tracking import DEFAULT_IOU_THRESHOLD, IouTracker, iou


def det(cls="can", box=(0.1, 0.1, 0.3, 0.3), conf=0.9):
    return Detection(track_id=None, cls=cls, conf=conf, box=box)


# --- iou() ---------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    assert iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_touching_edges_is_zero():
    # Shared edge, no area — must not be counted as overlap.
    assert iou((0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5)) == 0.0


def test_iou_half_overlap():
    # Two unit-ish boxes overlapping on exactly half their area:
    # intersection 0.5, union 1.5 -> 1/3
    assert iou((0.0, 0.0, 1.0, 1.0), (0.5, 0.0, 1.5, 1.0)) == pytest.approx(1 / 3)


def test_iou_degenerate_box_is_zero():
    assert iou((0.2, 0.2, 0.2, 0.2), (0.0, 0.0, 1.0, 1.0)) == 0.0


# --- id assignment -------------------------------------------------------


def test_assigns_incrementing_ids_to_new_detections():
    t = IouTracker()
    out = t.assign([det(box=(0.0, 0.0, 0.2, 0.2)), det(box=(0.7, 0.7, 0.9, 0.9))], now=0.0)
    assert [d.track_id for d in out] == [1, 2]


def test_same_item_keeps_its_id_across_frames():
    t = IouTracker()
    first = t.assign([det(box=(0.10, 0.10, 0.30, 0.30))], now=0.0)
    # Small jitter, still heavily overlapping.
    second = t.assign([det(box=(0.11, 0.11, 0.31, 0.31))], now=0.1)
    assert first[0].track_id == second[0].track_id == 1
    assert t.active_count == 1


def test_moved_far_enough_becomes_a_new_track():
    t = IouTracker()
    t.assign([det(box=(0.0, 0.0, 0.2, 0.2))], now=0.0)
    out = t.assign([det(box=(0.7, 0.7, 0.9, 0.9))], now=0.1)
    assert out[0].track_id == 2


def test_original_detection_is_not_mutated():
    t = IouTracker()
    d = det()
    out = t.assign([d], now=0.0)
    assert d.track_id is None
    assert out[0].track_id == 1
    assert out[0].cls == d.cls and out[0].box == d.box and out[0].conf == d.conf


# --- class awareness -----------------------------------------------------


def test_track_never_migrates_between_classes():
    """Same position, different class -> new track, not a silent relabel."""
    t = IouTracker()
    t.assign([det(cls="noodles-chicken", box=(0.1, 0.1, 0.3, 0.3))], now=0.0)
    out = t.assign([det(cls="noodles-beef", box=(0.1, 0.1, 0.3, 0.3))], now=0.1)
    assert out[0].track_id == 2


def test_overlapping_different_classes_get_separate_ids():
    t = IouTracker()
    out = t.assign(
        [
            det(cls="can", box=(0.1, 0.1, 0.4, 0.4)),
            det(cls="box", box=(0.1, 0.1, 0.4, 0.4)),
        ],
        now=0.0,
    )
    assert out[0].track_id != out[1].track_id


# --- greedy matching -----------------------------------------------------


def test_one_track_is_claimed_by_only_one_detection():
    t = IouTracker()
    t.assign([det(box=(0.10, 0.10, 0.30, 0.30))], now=0.0)
    # Two candidates overlap the single existing track; the better one keeps
    # id 1, the other must get a fresh id rather than duplicating it.
    out = t.assign(
        [
            det(box=(0.18, 0.18, 0.38, 0.38)),  # weaker overlap
            det(box=(0.11, 0.11, 0.31, 0.31)),  # stronger overlap
        ],
        now=0.1,
    )
    ids = [d.track_id for d in out]
    assert ids[1] == 1
    assert ids[0] != 1
    assert len(set(ids)) == 2


def test_matching_is_deterministic_for_tied_scores():
    t = IouTracker()
    t.assign([det(box=(0.0, 0.0, 0.2, 0.2)), det(box=(0.5, 0.5, 0.7, 0.7))], now=0.0)
    a = IouTracker()
    a.assign([det(box=(0.0, 0.0, 0.2, 0.2)), det(box=(0.5, 0.5, 0.7, 0.7))], now=0.0)
    frame = [det(box=(0.0, 0.0, 0.2, 0.2)), det(box=(0.5, 0.5, 0.7, 0.7))]
    assert [d.track_id for d in t.assign(frame, now=0.1)] == [
        d.track_id for d in a.assign(frame, now=0.1)
    ]


# --- expiry --------------------------------------------------------------


def test_track_survives_a_gap_shorter_than_expiry():
    t = IouTracker(expiry_s=1.5)
    t.assign([det()], now=0.0)
    out = t.assign([det()], now=1.4)
    assert out[0].track_id == 1


def test_track_expires_after_expiry_and_gets_a_new_id():
    t = IouTracker(expiry_s=1.5)
    t.assign([det()], now=0.0)
    out = t.assign([det()], now=1.6)
    assert out[0].track_id == 2


def test_expiry_boundary_is_inclusive():
    """Exactly at expiry_s the track is still alive (strict > drops it)."""
    t = IouTracker(expiry_s=1.5)
    t.assign([det()], now=0.0)
    assert t.assign([det()], now=1.5)[0].track_id == 1


def test_expired_tracks_are_dropped_from_memory():
    t = IouTracker(expiry_s=1.0)
    t.assign([det()], now=0.0)
    assert t.active_count == 1
    t.assign([], now=5.0)
    assert t.active_count == 0


def test_empty_frame_does_not_expire_a_fresh_track():
    t = IouTracker(expiry_s=1.5)
    t.assign([det()], now=0.0)
    t.assign([], now=0.5)
    assert t.assign([det()], now=1.0)[0].track_id == 1


# --- misc ----------------------------------------------------------------


def test_threshold_is_configurable():
    # Corner overlap of these two boxes is IoU ~0.032.
    loose = IouTracker(iou_threshold=0.01)
    loose.assign([det(box=(0.0, 0.0, 0.2, 0.2))], now=0.0)
    # Weakly-overlapping box matches under a loose threshold...
    assert loose.assign([det(box=(0.15, 0.15, 0.35, 0.35))], now=0.1)[0].track_id == 1

    strict = IouTracker(iou_threshold=0.9)
    strict.assign([det(box=(0.10, 0.10, 0.30, 0.30))], now=0.0)
    # ...and does not under a strict one.
    assert strict.assign([det(box=(0.15, 0.15, 0.35, 0.35))], now=0.1)[0].track_id == 2


def test_reset_clears_tracks_and_restarts_ids():
    t = IouTracker()
    t.assign([det(), det(box=(0.6, 0.6, 0.8, 0.8))], now=0.0)
    t.reset()
    assert t.active_count == 0
    assert t.assign([det()], now=0.1)[0].track_id == 1


def test_uses_injected_clock_when_now_is_omitted():
    ticks = iter([0.0, 0.1])
    t = IouTracker(clock=lambda: next(ticks))
    assert t.assign([det()])[0].track_id == 1
    assert t.assign([det()])[0].track_id == 1


def test_empty_input_returns_empty_list():
    assert IouTracker().assign([], now=0.0) == []


def test_default_threshold_is_exposed():
    assert 0.0 < DEFAULT_IOU_THRESHOLD < 1.0
