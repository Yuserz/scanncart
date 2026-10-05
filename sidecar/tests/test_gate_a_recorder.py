"""Tests for tools/gate_a_recorder.py.

Verifies trial roster generation, argument parsing, frame stamping, dry-run mode,
and simulated video recording against mock VideoCapture doubles.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.gate_a_recorder import (
    build_trial_roster,
    parse_args,
    record_clip,
    run,
    stamp_frame,
)


class FakeCap:
    """Mock cv2.VideoCapture returning synthetic frames."""

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
        # If exhausted, loop last frame or fail
        if self._frames and not self._fail_after:
            return True, self._frames[-1]
        return False, None

    def release(self):
        self.released = True


def _blank_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_build_trial_roster_exact_count():
    trials = build_trial_roster()
    assert len(trials) == 90

    deposits = [t for t in trials if t.kind == "deposit"]
    removals = [t for t in trials if t.kind == "removal"]
    no_transfers = [t for t in trials if t.kind == "no_transfer"]

    assert len(deposits) == 30
    assert len(removals) == 30
    assert len(no_transfers) == 30

    # Ensure IDs match DEP-01..30, REM-01..30, NOT-01..30
    assert deposits[0].trial_id == "DEP-01"
    assert deposits[-1].trial_id == "DEP-30"
    assert removals[0].trial_id == "REM-01"
    assert removals[-1].trial_id == "REM-30"
    assert no_transfers[0].trial_id == "NOT-01"
    assert no_transfers[-1].trial_id == "NOT-30"


def test_stamp_frame_preserves_dimensions_and_draws_text():
    frame = _blank_frame(640, 480)
    stamped = stamp_frame(
        frame=frame,
        trial_id="DEP-01",
        timestamp_iso="2026-10-02T12:00:00Z",
        frame_idx=15,
        fps=30.0,
        elapsed_s=0.5,
    )
    assert stamped.shape == frame.shape
    # Top banner pixels should not be completely black
    assert np.any(stamped[:35, :, :] > 0)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.source == "0"
    assert args.session == "s1"
    assert args.out == "data/gate_a_footage"
    assert args.duration == 10.0
    assert not args.stamp
    assert args.fps == 30.0
    assert args.trial is None
    assert not args.dry_run


def test_dry_run_does_not_open_cap(capsys):
    args = parse_args(["--dry-run", "--trial", "DEP-01"])
    ret = run(args, cap=None)
    assert ret == 0
    captured = capsys.readouterr()
    assert "[DRY RUN] Planned trials:" in captured.out
    assert "[DEP-01]" in captured.out


def test_record_clip_with_fake_cap(tmp_path):
    frames = [_blank_frame() for _ in range(5)]
    cap = FakeCap(frames, fail_after=True)
    trial = build_trial_roster()[0]
    out_file = tmp_path / "test_dep_01.mp4"

    # Use simulated clock steps
    clock = [0.0]

    def fake_time():
        clock[0] += 0.033
        return clock[0]

    def fake_sleep(_amt):
        pass

    record = record_clip(
        cap=cap,
        out_path=out_file,
        trial=trial,
        duration_s=0.15,
        target_fps=30.0,
        time_func=fake_time,
        sleep_func=fake_sleep,
    )

    assert out_file.exists()
    assert record["trial_id"] == "DEP-01"
    assert record["frame_count"] > 0
    assert record["resolution"] == [320, 240]


def test_run_session_writes_manifest(tmp_path):
    frames = [_blank_frame() for _ in range(10)]
    cap = FakeCap(frames, fail_after=False)

    out_dir = tmp_path / "footage"
    args = parse_args(["--trial", "DEP-01", "--out", str(out_dir), "--duration", "0.05", "--fps", "30.0"])

    ret = run(args, cap=cap)
    assert ret == 0
    assert cap.released

    manifest_path = out_dir / "s1" / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["session"] == "s1"
    assert len(data["trials"]) == 1
    assert data["trials"][0]["trial_id"] == "DEP-01"
    assert (out_dir / "s1" / "DEP-01.mp4").exists()


def _bright_frame(w=320, h=240):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _record(tmp_path, *, stamp=False, step=0.033, frames=10, duration=0.15, fps=30.0):
    cap = FakeCap([_bright_frame() for _ in range(frames)], fail_after=True)
    out_file = tmp_path / "clip.mp4"
    clock = [0.0]

    def fake_time():
        clock[0] += step
        return clock[0]

    record = record_clip(
        cap=cap, out_path=out_file, trial=build_trial_roster()[0], duration_s=duration,
        target_fps=fps, time_func=fake_time, sleep_func=lambda _s: None, stamp=stamp,
    )
    return out_file, record


def _first_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert ok
    return frame, fps


def test_frames_are_written_clean_by_default(tmp_path):
    """The clips are the replay corpus: a banner would be pixels the live camera never shows."""
    out_file, record = _record(tmp_path)
    frame, _ = _first_frame(out_file)
    assert record["stamped"] is False
    # The top band is the camera's mid-grey, not the banner's near-black (mp4v is lossy, so a margin).
    assert abs(float(frame[:30].mean()) - 128) < 10


def test_stamp_burns_the_banner_in(tmp_path):
    out_file, record = _record(tmp_path, stamp=True)
    frame, _ = _first_frame(out_file)
    assert record["stamped"] is True
    assert float(frame[:15].mean()) < 80


def test_a_slow_camera_is_retimed_to_the_rate_it_delivered(tmp_path):
    """15 fps into a 30 fps container would play every interaction back at double speed."""
    # Two clock reads per frame at 0.033 s each: ~15 fps delivered against a 30 fps target.
    out_file, record = _record(tmp_path, frames=20, duration=0.6)
    assert record["container_fps"] != 30.0
    assert abs(record["container_fps"] - record["fps"]) < 0.01
    _, fps = _first_frame(out_file)
    assert abs(fps - record["container_fps"]) < 0.5
    assert not list(tmp_path.glob(".*retime*")), "the temp file must not be left behind"


def test_a_camera_on_target_keeps_the_container_rate(tmp_path):
    # One clock read per frame would be needed for exactly 30 fps; a step of half the interval gives
    # two reads per frame at 1/60 s each, i.e. 30 fps delivered.
    out_file, record = _record(tmp_path, step=1 / 60, frames=20, duration=0.5)
    assert record["container_fps"] == 30.0
    _, fps = _first_frame(out_file)
    assert abs(fps - 30.0) < 0.5
