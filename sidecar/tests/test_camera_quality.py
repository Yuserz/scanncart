import numpy as np

from app.camera_quality import BRIGHTNESS_MIN, FrameQuality, frame_quality


def test_a_uniform_dark_frame_reports_its_brightness():
    # The 2026-08-29 counter frame measured 23/255 and detected nothing.
    frame = np.full((64, 64, 3), 23, dtype=np.uint8)
    q = frame_quality(frame)
    assert isinstance(q, FrameQuality)
    assert 22 <= q.brightness <= 24
    assert q.brightness < BRIGHTNESS_MIN


def test_a_flat_frame_has_no_contrast_and_no_sharpness():
    q = frame_quality(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert q.contrast < 1.0
    assert q.sharpness < 1.0


def test_an_edged_frame_is_sharper_than_a_blurred_one():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, 32:] = 255
    import cv2

    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    assert frame_quality(frame).sharpness > frame_quality(blurred).sharpness
