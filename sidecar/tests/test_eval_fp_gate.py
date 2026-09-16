"""Tests for tools/eval_fp_gate.py — all fakes, no model weights needed."""

import numpy as np
import pytest

from tools.eval_fp_gate import GateReport, run
from app.schemas import Detection


def _write_image(path, w=32, h=24):
    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


class ScriptedGateDetector:
    """Yields queued detections per infer() call, in file order."""

    names = {0: "safeguard"}

    def __init__(self, script):
        self._script = list(script)

    def infer(self, frame):
        return self._script.pop(0) if self._script else []


def _factory(script):
    def make(model_path, device, conf, imgsz):
        assert conf == 0.1  # scan threshold forwarded
        return ScriptedGateDetector(script)
    return make


def test_passes_when_nothing_detected(tmp_path):
    for i in range(2):
        _write_image(tmp_path / f"f{i}.jpg")
    report = run("m.pt", str(tmp_path), conf=0.1, detector_factory=_factory([]))
    assert report.passed
    assert report.frames_checked == 2
    assert report.max_conf == 0.0


def test_flags_any_detection_by_default(tmp_path):
    _write_image(tmp_path / "a.jpg")
    det = Detection(track_id=1, cls="safeguard", conf=0.8, box=(0.1, 0.2, 0.3, 0.4))
    report = run("m.pt", str(tmp_path), conf=0.1, detector_factory=_factory([[det]]))
    assert not report.passed
    assert len(report.violations) == 1
    assert report.violations[0].image == "a.jpg"
    assert report.violations[0].cls == "safeguard"
    assert report.violations[0].conf == 0.8


def test_class_filter_only_flags_listed_classes(tmp_path):
    _write_image(tmp_path / "a.jpg")
    dets = [
        Detection(track_id=1, cls="bottle", conf=0.9, box=(0, 0, 0.1, 0.1)),
        Detection(track_id=2, cls="safeguard", conf=0.7, box=(0, 0, 0.2, 0.2)),
    ]
    report = run(
        "m.pt", str(tmp_path), conf=0.1, classes=["safeguard"],
        detector_factory=_factory([dets]),
    )
    assert len(report.violations) == 1
    assert report.violations[0].cls == "safeguard"


def test_unreadable_image_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not a jpeg")
    _write_image(tmp_path / "good.jpg")
    report = run("m.pt", str(tmp_path), conf=0.1, detector_factory=_factory([]))
    assert report.frames_checked == 1
    assert report.frames_skipped == 1
    assert report.passed


def test_missing_dir_raises():
    with pytest.raises(NotADirectoryError):
        run("m.pt", "Z:/no/such/dir", detector_factory=_factory([]))


def test_empty_dir_raises(tmp_path):
    with pytest.raises(RuntimeError, match="zero frames"):
        run("m.pt", str(tmp_path), detector_factory=_factory([]))


def test_ignores_non_image_files(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "labels").mkdir()
    _write_image(tmp_path / "f.jpg")
    report = run("m.pt", str(tmp_path), conf=0.1, detector_factory=_factory([]))
    assert report.frames_checked == 1  # only f.jpg, not notes.txt or labels/


def test_gate_report_defaults():
    r = GateReport()
    assert r.passed
    assert r.max_conf == 0.0
