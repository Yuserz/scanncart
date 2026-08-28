"""Phase 4: RoboflowRemoteDetector, wiring, and /api/detector/probe.

The REAL_RESPONSE fixture is the verbatim payload captured from the live
workflow in Phase 0 (docs/DETECTOR_BACKENDS.md §0) — parsing is tested against
what the endpoint actually returns, not what we assumed it would.
"""

import copy

import numpy as np
import pytest

from app.inference import RoboflowRemoteDetector
from app.roboflow import RoboflowAuthError, RoboflowError, find_image_size
from app.tracking import IouTracker

REAL_RESPONSE = {
    "predictions": {
        "image": {"width": 480, "height": 640},
        "predictions": [
            {
                "width": 456.0,
                "height": 444.0,
                "x": 229.0,
                "y": 416.0,
                "confidence": 0.9426124691963196,
                "class_id": 3,
                "class": "century_tuna_flakes_in_oil_155_grams",
                "detection_id": "7d58050c-4dfa-475e-bebd-f775156ae6b9",
                "parent_id": "image",
            }
        ],
    }
}


class FakeClient:
    def __init__(self, response=None, raises=None):
        self.response = response if response is not None else copy.deepcopy(REAL_RESPONSE)
        self.raises = raises
        self.sent = []
        self.closed = False

    def run(self, image_b64, parameters=None):
        self.sent.append(image_b64)
        if self.raises:
            raise self.raises
        return self.response

    def close(self):
        self.closed = True


def frame(h=640, w=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make(response=None, raises=None, **kw):
    client = FakeClient(response, raises)
    kw.setdefault("tracker", IouTracker())
    return RoboflowRemoteDetector(client, **kw), client


def resp_with(**over):
    r = copy.deepcopy(REAL_RESPONSE)
    r["predictions"]["predictions"][0].update(over)
    return r


# --- parsing the real payload --------------------------------------------


def test_parses_the_real_response():
    det, _ = make()
    out = det.infer(frame())
    assert len(out) == 1
    assert out[0].cls == "century_tuna_flakes_in_oil_155_grams"
    assert out[0].conf == pytest.approx(0.9426, abs=1e-3)


def test_converts_centre_xywh_to_normalized_xyxy():
    det, _ = make()
    x1, y1, x2, y2 = det.infer(frame())[0].box
    # x 229 ± 228 over width 480; y 416 ± 222 over height 640
    assert x1 == pytest.approx((229 - 228) / 480, abs=1e-4)
    assert y1 == pytest.approx((416 - 222) / 640, abs=1e-4)
    assert x2 == pytest.approx((229 + 228) / 480, abs=1e-4)
    assert y2 == pytest.approx((416 + 222) / 640, abs=1e-4)


def test_all_box_values_stay_normalized():
    det, _ = make()
    assert all(0.0 <= v <= 1.0 for v in det.infer(frame())[0].box)


def test_out_of_frame_boxes_are_clamped():
    det, _ = make(resp_with(x=10.0, y=10.0, width=400.0, height=400.0))
    x1, y1, _, _ = det.infer(frame())[0].box
    assert x1 == 0.0 and y1 == 0.0


def test_reference_dims_come_from_the_response_not_the_frame():
    """Server echoes the dims it saw; trust those over our own."""
    det, _ = make()
    # A frame of a totally different shape must not change the box.
    a = det.infer(frame(100, 100))[0].box
    b = det.infer(frame(640, 480))[0].box
    assert a == b


def test_falls_back_to_sent_dims_when_image_meta_is_absent():
    r = {"predictions": {"predictions": [dict(REAL_RESPONSE["predictions"]["predictions"][0])]}}
    det, _ = make(r, infer_size=640)
    assert det.infer(frame(640, 480))[0].box[0] >= 0.0


def test_find_image_size_reads_the_meta_block():
    assert find_image_size(REAL_RESPONSE) == (480, 640)


def test_find_image_size_returns_none_when_absent():
    assert find_image_size({"predictions": {"predictions": []}}) is None


# --- class label ---------------------------------------------------------


def test_reads_the_class_field():
    det, _ = make()
    assert det.infer(frame())[0].cls == "century_tuna_flakes_in_oil_155_grams"


def test_falls_back_to_class_name_spelling():
    p = copy.deepcopy(REAL_RESPONSE)
    p["predictions"]["predictions"][0].pop("class")
    p["predictions"]["predictions"][0]["class_name"] = "milo"
    det, _ = make(p)
    assert det.infer(frame())[0].cls == "milo"


def test_prediction_without_any_label_is_dropped():
    p = copy.deepcopy(REAL_RESPONSE)
    p["predictions"]["predictions"][0].pop("class")
    det, _ = make(p)
    assert det.infer(frame()) == []


def test_names_is_populated_from_observed_classes():
    det, _ = make()
    det.infer(frame())
    assert det.names == {3: "century_tuna_flakes_in_oil_155_grams"}


# --- confidence filtering (client-side; workflow takes no parameters) ----


def test_filters_below_the_threshold():
    det, _ = make(conf=0.99)
    assert det.infer(frame()) == []


def test_keeps_at_or_above_the_threshold():
    det, _ = make(conf=0.9)
    assert len(det.infer(frame())) == 1


def test_threshold_is_applied_per_prediction():
    p = copy.deepcopy(REAL_RESPONSE)
    low = dict(p["predictions"]["predictions"][0])
    low["confidence"] = 0.10
    low["x"] = 50.0
    p["predictions"]["predictions"].append(low)
    det, _ = make(p, conf=0.5)
    assert len(det.infer(frame())) == 1


# --- malformed predictions -----------------------------------------------


@pytest.mark.parametrize("missing", ["x", "y", "width", "height", "confidence"])
def test_prediction_missing_a_geometry_field_is_dropped(missing):
    p = copy.deepcopy(REAL_RESPONSE)
    p["predictions"]["predictions"][0].pop(missing)
    det, _ = make(p)
    assert det.infer(frame()) == []


def test_non_numeric_geometry_is_dropped_not_raised():
    det, _ = make(resp_with(x="left"))
    assert det.infer(frame()) == []


def test_empty_predictions_yields_no_detections():
    det, _ = make({"predictions": {"image": {"width": 1, "height": 1}, "predictions": []}})
    assert det.infer(frame()) == []


def test_garbage_response_yields_no_detections():
    det, _ = make({"unexpected": "shape"})
    assert det.infer(frame()) == []


# --- tracking ------------------------------------------------------------


def test_assigns_a_track_id_since_the_workflow_has_no_tracker():
    det, _ = make()
    assert det.infer(frame())[0].track_id == 1


def test_same_item_keeps_its_id_across_frames():
    det, _ = make()
    assert det.infer(frame())[0].track_id == det.infer(frame())[0].track_id


def test_response_tracker_id_wins_when_present():
    """Adding a tracking block upstream later needs no code change here."""
    det, _ = make(resp_with(tracker_id=77))
    assert det.infer(frame())[0].track_id == 77


def test_without_a_tracker_track_id_stays_none():
    det, _ = make(tracker=None)
    assert det.infer(frame())[0].track_id is None


def test_empty_frame_still_ages_tracks():
    """An item leaving frame must expire, not hold its slot indefinitely."""
    tracker = IouTracker(expiry_s=1.0, clock=iter([0.0, 5.0, 5.0]).__next__)
    det, client = make(tracker=tracker)
    det.infer(frame())
    assert tracker.active_count == 1
    client.response = {"predictions": {"predictions": []}}
    det.infer(frame())
    assert tracker.active_count == 0


# --- transmission --------------------------------------------------------


def test_large_frames_are_downscaled_before_transmit():
    import base64

    import cv2

    det, client = make(infer_size=640)
    det.infer(np.zeros((1080, 1920, 3), dtype=np.uint8))
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(client.sent[0]), np.uint8), cv2.IMREAD_COLOR
    )
    assert max(decoded.shape[:2]) == 640


def test_small_frames_are_not_upscaled():
    import base64

    import cv2

    det, client = make(infer_size=640)
    det.infer(np.zeros((100, 120, 3), dtype=np.uint8))
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(client.sent[0]), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape[:2] == (100, 120)


def test_lower_jpeg_quality_produces_a_smaller_payload():
    """Quality 95 was 128 KB in Phase 0; 80 is the pipeline's own default."""
    noisy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    hi, hi_c = make(jpeg_quality=95)
    lo, lo_c = make(jpeg_quality=80)
    hi.infer(noisy)
    lo.infer(noisy)
    assert len(lo_c.sent[0]) < len(hi_c.sent[0])


def test_client_errors_propagate():
    det, _ = make(raises=RoboflowError("boom"))
    with pytest.raises(RoboflowError):
        det.infer(frame())


def test_close_closes_the_client():
    det, client = make()
    det.close()
    assert client.closed is True
