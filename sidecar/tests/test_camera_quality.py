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


def test_focus_drift_is_zero_for_a_steady_scene():
    from app.camera_quality import focus_drift

    assert focus_drift([100.0, 100.0, 100.0]) == 0.0


def test_focus_drift_flags_a_hunting_autofocus():
    from app.camera_quality import focus_drift

    # The StreamCam swung 6 -> 31 on a static scene.
    assert focus_drift([6.0, 31.0, 8.0, 29.0]) > 0.25


def test_focus_drift_of_too_few_samples_is_zero():
    from app.camera_quality import focus_drift

    assert focus_drift([]) == 0.0
    assert focus_drift([12.0]) == 0.0


def test_measure_fps_counts_successful_reads_per_second():
    from app.camera_quality import measure_fps

    now = {"t": 0.0}
    reads = {"n": 0}

    def clock():
        return now["t"]

    def read():
        reads["n"] += 1
        now["t"] += 0.05  # 20 fps
        return True

    fps = measure_fps(read, seconds=1.0, clock=clock, sleep=lambda s: None)
    assert 19.0 <= fps <= 21.0


def test_measure_fps_ignores_failed_reads():
    from app.camera_quality import measure_fps

    now = {"t": 0.0}
    n = {"i": 0}

    def clock():
        return now["t"]

    def read():
        n["i"] += 1
        now["t"] += 0.05
        return n["i"] % 2 == 0  # half the reads fail

    fps = measure_fps(read, seconds=1.0, clock=clock, sleep=lambda s: None)
    assert 9.0 <= fps <= 11.0
