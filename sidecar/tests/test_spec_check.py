"""Tests for `spec_check.py`: the PRD targets, and the instrument each is read from.

Split out of `test_dataset_tools.py` along its section banners; the fixtures this and the
other split modules share live in `tests/dataset_tool_helpers.py`. Nothing here touches the
dataset workspace - the tools read it lazily, from inside `main()`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
import pytest
import audit_recall
import spec_check

# Rebuilt from the modules that own each fact, not copied by value.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "MODEL_TRAINING.md"


# ---------------------------------------------------------------------------
# spec_check: the PRD's targets, and the two instruments each is read from
# ---------------------------------------------------------------------------


def test_the_summary_takes_its_p95_as_the_nearest_rank_frame():
    """A latency budget is a promise about frames, so its p95 should be one of them.

    Nearest rank - `ceil(0.95 n)` - puts a 20-frame run's p95 at its 19th value. An
    interpolating percentile would land between the 19th and 20th, which on a run with one slow
    frame is the difference between reporting the typical bad frame and reporting the worst one;
    the worst one is already the `max` column beside it, so the interpolation would make p95 and
    max the same number and lose the distribution.
    """
    summary = spec_check.Summary.of([float(v) for v in range(1, 21)])
    assert summary.n == 20
    assert summary.p50 == 10.5  # median of 1..20
    assert summary.p95 == 19.0  # rank ceil(0.95 * 20) = 19 -> the 19th value
    assert summary.largest == 20.0
    # A single frame still summarizes to itself, which is what makes --limit 1 usable while
    # wiring the tool up rather than a crash nobody expects from a diagnostic.
    assert spec_check.Summary.of([7.5]).p95 == 7.5


def test_a_summary_of_nothing_is_an_error_not_zero_milliseconds():
    """A loop that timed no frames reads as infinitely fast, so it has to fail loudly.

    `1000.0 / 0.0` would raise anyway, but only once something divides by it; the summary is
    where the emptiness is known, and a tool whose speed verdict comes from a mean of nothing is
    exactly the "passed because it did not measure" failure this check exists to avoid.
    """
    with pytest.raises(ValueError):
        spec_check.Summary.of([])


def test_a_latency_target_is_a_ceiling_and_a_speed_target_is_a_floor():
    """The direction *is* the claim: a latency compared with `>=` passes at any speed."""
    faster = spec_check.Target("fps", 40.0, 30.0, "fps", True, "x")
    slower = spec_check.Target("fps", 20.0, 30.0, "fps", True, "x")
    assert faster.ok and not slower.ok

    under = spec_check.Target("latency", 100.0, 150.0, "ms", False, "x")
    over = spec_check.Target("latency", 200.0, 150.0, "ms", False, "x")
    assert under.ok and not over.ok


def test_the_targets_are_the_prd_s_own_numbers_not_the_last_measurement():
    """Pinned to the document, so a slow day cannot become a looser target."""
    assert spec_check.TARGET_FPS == 30.0
    assert spec_check.TARGET_LATENCY_MS == 150.0
    assert spec_check.TARGET_ACCURACY == 0.90
    prd = (REPO_ROOT / "docs" / "PRD.md").read_text(encoding="utf-8")
    for phrase in ("30 fps", "150 ms", "90%"):
        assert phrase in prd


def test_a_pipeline_can_clear_the_latency_ceiling_and_still_miss_the_fps_floor():
    """Why both instruments are reported instead of the flattering one.

    40 ms a frame is comfortably inside the 150 ms promise and is only 25 fps, so a check that
    read latency alone would pass a pipeline that cannot hold 30. The reverse shape is what the
    `isolated` figure covers: a detector that clears 30 fps on its own inside a pipeline that
    does not - which is why the detector's own timing is reported beside the app's.
    """
    targets = spec_check.spec_targets(
        spec_check.Summary.of([20.0]),  # 50 fps isolated: fine
        spec_check.Summary.of([40.0]),  # 25 fps in-app: not
        0.95,
    )
    by_label = {t.label: t for t in targets}
    assert len(targets) == 5
    assert by_label["isolated infer fps"].ok
    assert not by_label["in-app pipeline fps"].ok
    assert by_label["in-app mean latency"].ok
    # Each reading says which clause it came from, since a failure is re-read against the PRD.
    assert {t.source for t in targets} == {"PRD 5, 7", "PRD 6, 7", "PRD 7"}


def test_the_target_summary_all_passes_reads_as_one_sentence():
    """The line a reader stops at, and the empty failure list it depends on."""
    targets = spec_check.spec_targets(
        spec_check.Summary.of([20.0]), spec_check.Summary.of([26.0]), 0.918
    )
    assert spec_check.failed(targets) == ()
    assert spec_check.target_summary(targets) == "all 5 PRD targets pass"


def test_the_target_summary_names_the_failures_rather_than_only_counting_them():
    """"2 of 5 failed" alone sends the reader back through the block above to find out which.

    Accurate recall with a slow pipeline: the count is 3, and all three are speed - a shape that
    reads as "the model is fine, this machine's not", which is only visible if they are named.
    """
    targets = spec_check.spec_targets(
        spec_check.Summary.of([20.0]), spec_check.Summary.of([200.0]), 0.95
    )
    text = spec_check.target_summary(targets)
    assert text.startswith("3 of 5 PRD targets fail")
    assert "in-app pipeline fps" in text
    assert "in-app mean latency" in text and "in-app p95 latency" in text
    assert "instance recall" not in text  # the one that passed is not in the list


def test_the_record_path_is_the_weights_stem_beside_it():
    """The tool reads the same record pair `--install` writes, not a second naming rule."""
    assert spec_check.record_path("models/scanncart-grocery-v1.pt").name == (
        "scanncart-grocery-v1.json"
    )


def test_a_missing_or_unreadable_record_is_not_an_error(tmp_path):
    """A hand-trained weight has no record, and a check that refused to run without one would be
    a check nobody runs on the weights most in need of checking."""
    assert spec_check.recorded_aggregates(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert spec_check.recorded_aggregates(broken) is None
    # A record that exists but holds no test measurement is the same answer as no record.
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"validation": []}), encoding="utf-8")
    assert spec_check.recorded_aggregates(empty) is None


def test_the_record_s_most_recent_test_block_is_the_one_quoted(tmp_path):
    """The number printed beside today's has to be the latest one taken, not the first.

    A weight can be measured more than once - a re-run after a fix, a later `--val` on more
    data - and quoting the oldest block would compare today's speed against a stale accuracy and
    read as a regression in the model.
    """
    record = tmp_path / "w.json"
    record.write_text(
        json.dumps({"validation": [
            {"split": "valid", "aggregates": {"recall": 0.50}},
            {"split": "test", "aggregates": {"recall": 0.90}},
            {"split": "test", "aggregates": {"recall": 0.94}},
        ]}),
        encoding="utf-8",
    )
    assert spec_check.recorded_aggregates(record)["recall"] == 0.94


def test_the_cycling_source_never_runs_out_and_never_reports_failure():
    """The timing loop runs to the frame count, not until the source dries up.

    A source that returned None would quietly shorten the measurement while the report still
    claimed the requested frame count - the mean would describe fewer frames than the header
    says, and nothing on screen would show it.
    """

    class _Frame:
        shape = (2, 2, 3)

    frames = [_Frame(), _Frame()]
    source = spec_check.Cycling(frames)
    assert source.failure is None
    seen = [source.latest()[1] for _ in range(5)]
    assert len(seen) == 5 and all(s is not None for s in seen)
    # 2 frames, 5 calls: it cycles rather than stopping or repeating the last one.
    assert [s is frames[0] for s in seen] == [True, False, True, False, True]


def test_spec_check_does_not_import_a_gpu_library_at_module_level():
    """Same shape as the other two tools: the accounting is testable with no torch and no cv2.

    The suite imports every tool in this file, so a module-level `import cv2` or `torch` would
    make the dataset-tool tests need the GPU stack - the thing they exist to not need.
    """
    source = Path(spec_check.__file__).read_text(encoding="utf-8")
    for module in ("cv2", "torch", "ultralytics"):
        assert not re.search(rf"^(from|import)\s+{module}", source, re.MULTILINE), module
    # Deferred rather than absent: frames come from `cv2`, and the two loops are the app's own
    # detector and pipeline - which is what pulls torch in, one level down.
    assert re.search(r"^\s+import cv2", source, re.MULTILINE)
    assert re.search(r"^\s+from app\.inference import", source, re.MULTILINE)
    assert re.search(r"^\s+from app\.pipeline import", source, re.MULTILINE)


def test_the_configuration_label_tells_apart_the_three_ways_to_get_settings(tmp_path):
    """`--defaults` and "this machine was never configured" give the same numbers for opposite
    reasons, and a report that conflates them reads as a machine nobody set up."""
    path = tmp_path / "settings.json"
    assert "as built" in spec_check.configuration_label(False, path, False)
    assert "as built" in spec_check.configuration_label(False, path, True)  # --defaults wins
    assert "not found" in spec_check.configuration_label(True, path, False)
    label = spec_check.configuration_label(True, path, True)
    assert str(path) in label and "not found" not in label


def test_spec_check_reads_the_settings_file_the_app_reads():
    """Absolute, not relative: `load_settings` resolves a relative path against the process cwd,
    which is the repo root by hand and the sidecar dir under Electron - two different profiles
    from one command. The tool names the file outright so both start-ups measure the same one.
    """
    assert spec_check.SETTINGS_PATH.is_absolute()
    assert spec_check.SETTINGS_PATH.name == "settings.json"
    assert spec_check.SETTINGS_PATH.parent.name == "data"
    # And the reload it feeds, rather than a second reader of its own.
    source = Path(spec_check.__file__).read_text(encoding="utf-8")
    assert re.search(r"^\s+from app\.settings_store import load_settings", source, re.MULTILINE)


def test_spec_check_puts_the_sidecar_root_on_the_path_so_it_runs_as_a_script():
    """The one dataset tool that imports `app`, and the failure is a bare traceback.

    Run as a script, `tools/` is what Python puts on the path - not the sidecar root - so the
    `from app.inference import ...` inside `main()` cannot resolve. The suite does not catch
    this on its own, because pytest.ini adds the sidecar root for it: the tool passes under test
    and dies for whoever follows the doc. A `subprocess` run would be the real check, but it
    drags torch into this file, which is exactly what these tests stay away from.
    """
    source = Path(spec_check.__file__).read_text(encoding="utf-8")
    assert re.search(r"^sys\.path\.[a-z]+\(str\(SIDECAR_ROOT\)\)", source, re.MULTILINE)
    # And the import it exists for, so removing one without the other is caught.
    assert re.search(r"^\s+from app\.[a-z]+ import", source, re.MULTILINE)


def test_spec_check_times_the_app_s_two_loops_and_warms_up_the_detector():
    """The instruments are named, and the warm-up is real.

    The first infers build the CUDA context and the predictor. Counted, they would report the
    cost of starting up as the cost of running - which on this host is most of the first
    second's worth of frames and would fail the latency target for a reason that is not the
    pipeline.
    """
    assert "YoloDetector.infer" in spec_check.__doc__
    assert "Pipeline.process_once" in spec_check.__doc__
    assert 0 < spec_check.WARMUP < spec_check.TIMING_FRAMES
    # Nearest-rank p95, stated as a constant rather than a literal inside the formula.
    assert spec_check.P95 == 0.95


def test_the_training_doc_points_at_the_spec_check():
    """The PRD's targets are only checkable by someone who knows the command exists.

    `--val` and `audit_recall.py` both report accuracy, so a reader of the acceptance section
    would reasonably conclude accuracy is all there is to clear - and then ship a weight that
    cannot hold 30 fps.
    """
    text = DOC.read_text(encoding="utf-8")
    assert "spec_check.py" in text
    floor_at = text.index('### What \"good\" looks like')
    section = text[floor_at : text.index("## 7. Integrating", floor_at)]
    assert "spec_check.py" in section
    # And the command that gates on it, since a check nobody can fail is a report.
    assert "--strict" in section
