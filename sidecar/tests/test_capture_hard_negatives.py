"""Tests for tools/capture_hard_negatives.py.

Follows the suite convention: everything runs against fakes — no real
camera is opened. FakeCap mimics the cv2.VideoCapture surface the tool
uses (isOpened/read/release).
"""

import numpy as np
import pytest

from tools.capture_hard_negatives import run, _next_index, _source_label, _write_data_yaml


class FakeCap:
    """Mimics cv2.VideoCapture: yields frames in order, then keeps failing."""

    def __init__(self, frames, fail_after=True):
        self._frames = list(frames)
        self._i = 0
        self._fail_after = fail_after
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self._i < len(self._frames):
            frame = self._frames[self._i]
            self._i += 1
            return True, frame
        return (False, None) if self._fail_after else (False, None)

    def release(self):
        self.released = True


class ClosedCap:
    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


def _frame(w=64, h=48):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_run_saves_frames_with_empty_labels(tmp_path):
    cap = FakeCap([_frame(), _frame(), _frame()])
    saved = run(
        0, out=str(tmp_path), interval=0.0, max_frames=2,
        capture_factory=lambda src: cap,
    )
    assert saved == 2
    assert cap.released
    images = sorted((tmp_path / "images").glob("*.jpg"))
    labels = sorted((tmp_path / "labels").glob("*.txt"))
    assert [p.name for p in images] == ["cam0_000000.jpg", "cam0_000001.jpg"]
    assert [p.name for p in labels] == ["cam0_000000.txt", "cam0_000001.txt"]
    for lbl in labels:
        assert lbl.read_text() == ""  # empty label == YOLO background image


def test_run_respects_interval(tmp_path):
    ticks = iter([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cap = FakeCap([_frame() for _ in range(6)])
    saved = run(
        0, out=str(tmp_path), interval=1.0, max_frames=2,
        capture_factory=lambda src: cap, now_fn=lambda: next(ticks),
    )
    assert saved == 2  # frames at t=1.0 and t=2.0; t=0.5/1.5 are too soon


def test_run_skips_failed_reads_then_continues(tmp_path):
    class FlakyCap(FakeCap):
        def __init__(self, frames):
            super().__init__(frames)
            self._failures = 0

        def read(self):
            if self._failures < 2:  # fail the first two reads (before consuming)
                self._failures += 1
                return False, None
            if self._i < len(self._frames):
                frame = self._frames[self._i]
                self._i += 1
                return True, frame
            return False, None

    cap = FlakyCap([_frame(), _frame()])
    saved = run(
        0, out=str(tmp_path), interval=0.0, max_frames=2,
        capture_factory=lambda src: cap,
    )
    assert saved == 2


def test_run_raises_when_source_dies(tmp_path):
    cap = FakeCap([])  # never produces a frame
    with pytest.raises(RuntimeError, match="stopped producing"):
        run(0, out=str(tmp_path), interval=0.0, max_frames=5,
            capture_factory=lambda src: cap)


def test_run_raises_on_unopenable_source(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        run(9, out=str(tmp_path), capture_factory=lambda src: ClosedCap())


def test_run_rejects_bad_args(tmp_path):
    cap = FakeCap([_frame()])
    with pytest.raises(ValueError):
        run(0, out=str(tmp_path), interval=-1.0, max_frames=1,
            capture_factory=lambda src: cap)
    with pytest.raises(ValueError):
        run(0, out=str(tmp_path), interval=1.0, max_frames=0,
            capture_factory=lambda src: cap)


def test_run_continues_numbering_after_previous_run(tmp_path):
    cap = FakeCap([_frame() for _ in range(3)])
    run(0, out=str(tmp_path), interval=0.0, max_frames=2,
        capture_factory=lambda src: cap)
    cap2 = FakeCap([_frame()])
    saved = run(0, out=str(tmp_path), interval=0.0, max_frames=1,
                capture_factory=lambda src: cap2)
    assert saved == 1
    assert (tmp_path / "images" / "cam0_000002.jpg").exists()


def test_run_writes_data_yaml_with_background_note(tmp_path):
    cap = FakeCap([_frame()])
    run(0, out=str(tmp_path), interval=0.0, max_frames=1,
        capture_factory=lambda src: cap)
    yaml_text = (tmp_path / "data.yaml").read_text()
    assert "train: images" in yaml_text
    assert "background" in yaml_text  # warns against training on it alone


def test_write_data_yaml_is_idempotent(tmp_path):
    _write_data_yaml(tmp_path)
    _write_data_yaml(tmp_path)
    assert (tmp_path / "data.yaml").read_text().count("train: images") == 1


def test_source_label_for_camera_and_video():
    assert _source_label(0) == "cam0"
    assert _source_label("0") == "cam0"
    assert _source_label("checkout footage.mp4") == "checkout_footage"


def test_next_index_scans_existing_files(tmp_path):
    (tmp_path / "cam0_000005.jpg").touch()
    (tmp_path / "cam0_000002.jpg").touch()
    assert _next_index(tmp_path, "cam0") == 6
    assert _next_index(tmp_path, "cam1") == 0
