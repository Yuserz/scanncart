#!/usr/bin/env python
"""Do the PRD's inference targets hold for a trained weight, through the app's own code?

The PRD promises three things about detection on this host, and each is a number with a
threshold rather than an impression:

    PRD 5    real-time detection   >= 30 fps processing with YOLO11
    PRD 6    performance           end-to-end latency < 150 ms (local)
    PRD 7    success metrics       all three, plus >= 90% accuracy on common grocery items

`--val` and `audit_recall.py` answer *how accurate* a weight is, per class and per distance.
Neither answers whether the app clears those three, which is a different question: a weight can
sit above every recall floor and still be too slow to run at 30 fps, and a pipeline can be fast
enough and still spend 40 ms a frame inside a JPEG encode nobody was timing.

Four things make this not a re-run of either tool:

**Measured through the app, not around it.** Timing runs `YoloDetector.infer` and the real
`Pipeline.process_once`, against real product photos, at the app's shipped `imgsz`, confidence
and `preview_height`. The per-frame cost of a live capture is detection + tracking + the
allowlist filter + logging + the JPEG preview encode + building the `FrameMessage`, so
re-implementing "detect one image" would measure a pipeline that does not exist.

**Two timings, because they answer different questions.** The `isolated` figure is what
DETECTOR_BACKENDS.md's table quotes; the `in-app` figure is wall clock around the whole frame.
The pipeline's own `latency_ms` stat is a *third* number and the smallest of the three: it is
`t1 - t0` around `detector.infer`, so it stops before the encode and undercounts what a frame
really costs. All three are printed so none is mistaken for another.

**Accuracy at the app's operating point.** Recall here is per-instance at `conf_threshold`
(0.5), matched class-aware by IoU >= 0.5 - the recall a Live view experiences. Ultralytics'
validator instead picks each class's best-F1 threshold, which can be an operating point no
operator could run, so its per-class recall and this total are allowed to differ. The tool
prints both and says which is which rather than resolving them.

**Frames at the geometry the measurement needs.** Timing frames are resized to the configured
capture size, because resolution is part of the cost - the stretch resize inside `infer`, the
NMS, the encode all scale with it, and an empty frame would understate all three by leaving them
nothing to do. Frames for accuracy are left at the size the weights were trained on. A frame
that is not in front of the lens cannot be measured: this is the closest thing to the live
reading that needs no camera.

**The configuration measured is this machine's, and it says so.** By default the settings come
from `data/settings.json` - what the app on this box actually runs - rather than from
`Settings()`'s hardcoded defaults, because those are two different pipelines and the difference
is not cosmetic: the shipped defaults capture 640x480 at `imgsz` 640, while a saved profile can
run 960 and a different capture size, and a frame costs what the profile says it costs. The
report names the file it read, and `--defaults` measures the shipped set instead - which is what
you want when comparing machines or a fresh install.

    python sidecar/tools/spec_check.py --generation v1
    python sidecar/tools/spec_check.py --generation v1 --limit 40 --strict
    python sidecar/tools/spec_check.py --generation v1 --defaults   # shipped settings

Needs no camera and no network. Reads the ignored dataset workspace; the per-class and
per-crowding breakdown lives in `audit_recall.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import resources  # must precede numpy/torch: sets OMP/MKL thread limits

import audit_recall
from generations import DEFAULT, GENERATIONS, Generation
from generations import get as generation_for
from workspace import SIDECAR_ROOT

# This is the one dataset tool that drives `app` code, and run as a script Python puts `tools/`
# on the path - not the sidecar root - so `from app.inference import ...` in `main()` would raise
# ModuleNotFoundError for whoever followed the doc. The test suite never sees it, because
# pytest.ini adds the sidecar root for it: the tool works under test and dies by hand.
sys.path.append(str(SIDECAR_ROOT))

# The PRD's own numbers. Not tunable by flag on purpose: a target that can be moved is not a
# target, and a run that needed a looser one would be reporting a failure it should report.
TARGET_FPS = 30.0
TARGET_LATENCY_MS = 150.0
TARGET_ACCURACY = 0.90

# Where each target is written down, so a failure names the clause to re-read. Section 5 states
# its requirement and section 7 restates it as a success metric, which is why two carry a pair.
PRD_REALTIME = "PRD 5, 7"
PRD_LATENCY = "PRD 6, 7"
PRD_ACCURACY = "PRD 7"

# The app's own settings file, resolved absolutely: `settings_store.load_settings` takes a
# relative path and reads it against the process cwd, which is the repo root when a person runs
# this by hand and the sidecar dir when Electron spawns it. Naming the file outright is the only
# way the tool reads the same one whichever way it was started.
SETTINGS_PATH = SIDECAR_ROOT / "data" / "settings.json"

# How many labelled test frames to time over. Enough that one slow frame cannot move the mean,
# few enough that a full check stays a couple of minutes on this machine.
TIMING_FRAMES = 80
# Frames discarded before timing starts: the first infers build the CUDA context and the
# predictor, and counting them would report the cost of starting up as the cost of running.
WARMUP = 6
# The 95th percentile as nearest-rank: ceil(0.95 n), so a 20-frame run's p95 is its 19th value
# rather than a number between two frames. A latency budget is a promise about frames, so the
# statistic it is checked against should be one of them.
P95 = 0.95


@dataclass(frozen=True)
class Summary:
    """A latency distribution, in milliseconds, over one instrumented loop."""

    n: int
    mean: float
    p50: float
    p95: float
    largest: float

    @classmethod
    def of(cls, values: list[float]) -> "Summary":
        """Summarize a non-empty list of per-frame milliseconds."""
        if not values:
            raise ValueError("no frames timed - nothing to summarize")
        ordered = sorted(values)
        rank = max(1, math.ceil(P95 * len(ordered)))
        return cls(
            n=len(values),
            mean=statistics.mean(values),
            p50=statistics.median(values),
            p95=ordered[rank - 1],
            largest=ordered[-1],
        )

    def line(self) -> str:
        """`mean 24.1  p50 23.7  p95 33.8  max 38.9` - the reading, one row."""
        return (
            f"mean {self.mean:7.1f}  p50 {self.p50:7.1f}  "
            f"p95 {self.p95:7.1f}  max {self.largest:7.1f}  ms"
        )


def fps(mean_ms: float) -> float:
    """Frames per second from a per-frame mean, which is how both speed targets read."""
    return 1000.0 / mean_ms if mean_ms else 0.0


def configuration_label(use_saved: bool, path: Path, exists: bool) -> str:
    """Where the measured configuration came from - a claim the report has to carry.

    Three states, and the middle one is the one worth having: `--defaults` on a machine that has
    a profile, and no profile at all, produce the same numbers for opposite reasons. A reader
    deciding whether a result describes their app needs to know which they are looking at, and
    a report that says only "shipped defaults" reads as a machine that was never configured.
    """
    if not use_saved:
        return "shipped defaults  (Settings(), as built)"
    if exists:
        return f"saved settings    ({path})"
    return f"shipped defaults  ({path} not found - this machine has no saved profile)"


@dataclass(frozen=True)
class Target:
    """One PRD threshold next to the number measured for it, and whether it cleared."""

    label: str
    measured: float
    target: float
    unit: str
    higher_is_better: bool
    source: str
    digits: int = 1

    @property
    def ok(self) -> bool:
        """`>=` for a floor (fps, accuracy), `<` for a ceiling (latency).

        The comparison is written out rather than passed in as a callable so that the two
        directions are visible at every call site: a latency target checked with `>=` reads as
        passing at any speed.
        """
        return (
            self.measured >= self.target
            if self.higher_is_better
            else self.measured < self.target
        )

    def line(self) -> str:
        """`[PASS] in-app mean latency   26.8 ms  < 150 (PRD 6, 7)` - the claim, one row."""
        mark = "PASS" if self.ok else "FAIL"
        cmp_ = ">=" if self.higher_is_better else "<"
        value = f"{self.measured:.{self.digits}f}"
        return (
            f"  [{mark}] {self.label:<34}{value:>8} {self.unit:<5}"
            f" {cmp_} {self.target:g}   ({self.source})"
        )


def spec_targets(
    isolated: Summary, in_app: Summary, accuracy: float
) -> tuple[Target, ...]:
    """The PRD's five readings: two speeds, two latencies, one accuracy.

    Two speeds and two latencies because each pair is a different instrument, and the tool's
    whole job is not to let them be collapsed: `isolated` is the detector's own cost and
    `in-app` is what a frame costs the sidecar, while the mean is the typical frame and the p95
    is the bad one - a pipeline can hold 26 ms on average and still stutter at 40.
    """
    return (
        Target("isolated infer fps", fps(isolated.mean), TARGET_FPS, "fps",
               True, PRD_REALTIME),
        Target("in-app pipeline fps", fps(in_app.mean), TARGET_FPS, "fps",
               True, PRD_REALTIME),
        Target("in-app mean latency", in_app.mean, TARGET_LATENCY_MS, "ms",
               False, PRD_LATENCY),
        Target("in-app p95 latency", in_app.p95, TARGET_LATENCY_MS, "ms",
               False, PRD_LATENCY),
        Target("instance recall at app conf", accuracy, TARGET_ACCURACY, "ratio",
               True, PRD_ACCURACY, digits=3),
    )


def failed(targets: tuple[Target, ...]) -> tuple[Target, ...]:
    """The targets that did not clear, in the order they are reported."""
    return tuple(t for t in targets if not t.ok)


def target_summary(targets: tuple[Target, ...]) -> str:
    """One sentence for the whole check - the line a reader stops at.

    Names the failures rather than only counting them: "2 of 5 failed" on its own sends someone
    back through the block above, and on a machine where only the GPU changed the two that
    failed are usually the same two.
    """
    bad = failed(targets)
    if not bad:
        return f"all {len(targets)} PRD targets pass"
    names = ", ".join(f"{t.label} ({t.measured:.{t.digits}f} vs {t.target:g})" for t in bad)
    return f"{len(bad)} of {len(targets)} PRD targets fail: {names}"


# ---------------------------------------------------------------------------
# The frame source: real frames, replayed, so a timing loop can run to length
# ---------------------------------------------------------------------------


class Cycling:
    """A frame source that hands back real product photos in a loop.

    Implements the two things `Pipeline` asks of a source: `latest()` returning `(seq, frame)`
    or `None`, and a `failure` attribute it checks every frame. Real frames rather than
    generated ones because every step under measurement depends on there being content: an
    empty frame gives NMS nothing to suppress, the encode nothing to compress and the log
    nothing to write, so it would time the parts of the pipeline that never scale.
    """

    failure = None

    def __init__(self, frames: list, fps: float = 60.0):
        self._frames = frames
        self._i = 0
        self.fps = fps
        self.height, self.width = frames[0].shape[0], frames[0].shape[1]

    def latest(self):
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return self._i, frame

    def release(self) -> None:
        """A `Pipeline` teardown may call this; there is nothing to release."""


# ---------------------------------------------------------------------------
# The impure layer: the model, the pipeline, and the clock
# ---------------------------------------------------------------------------


@dataclass
class PipelineTiming:
    """The in-app loop's wall clock, next to the two stats the pipeline reports for itself."""

    wall_ms: list[float] = field(default_factory=list)
    reported_latency_ms: list[float] = field(default_factory=list)
    reported_fps: list[float] = field(default_factory=list)

    @property
    def summary(self) -> Summary:
        return Summary.of(self.wall_ms)

    @property
    def reported(self) -> Summary:
        return Summary.of(self.reported_latency_ms) if self.reported_latency_ms else None


def load_frames(generation: Generation, split: str, limit: int, size: tuple[int, int]) -> list:
    """Real labelled frames from the split, resized to the configured capture geometry.

    Labelled frames rather than any file in the split: an unlabelled one is a frame the
    accuracy pass cannot score, and using one for timing would put the two halves of the report
    on different image sets for no reason. Resized to the capture size, because that is the
    shape a live capture delivers and the resize is a real per-frame cost - and it is the
    *configured* size, since the same weight at 640x480 and at 1280x720 are different amounts
    of work and only one of them is the app.
    """
    import cv2

    paths = audit_recall.frame_paths(generation, split, limit)
    frames = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frames.append(cv2.resize(frame, size))
    if not frames:
        raise SystemExit(
            f"no readable frames in {generation.export_dir / split} - the split is empty or "
            "unreadable, so there is nothing to time"
        )
    return frames


def measure_detection(detector, frames: list, warmup: int = WARMUP) -> list[float]:
    """Wall-clock ms per frame for `YoloDetector.infer` alone - the isolated figure."""
    import time

    for frame in frames[:warmup]:
        detector.infer(frame)
    out: list[float] = []
    for frame in frames:
        t0 = time.perf_counter()
        detector.infer(frame)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def measure_pipeline(pipeline, frames: list) -> PipelineTiming:
    """Wall-clock ms per frame for the real `Pipeline.process_once`, plus its own stats.

    One pass, so both the wall clock and the reported stats describe the same frames. A second
    pass for the stats would be cheaper to write and would answer about a different set of
    frames, which is how a "the stat is lower than the real cost" finding turns into a rumour.
    """
    import time

    timing = PipelineTiming()
    for _ in frames:
        t0 = time.perf_counter()
        message = pipeline.process_once()
        timing.wall_ms.append((time.perf_counter() - t0) * 1000.0)
        if message is not None:
            timing.reported_latency_ms.append(message["stats"]["latency_ms"])
            timing.reported_fps.append(message["stats"]["infer_fps"])
    return timing


def measure_accuracy(
    generation: Generation,
    weights: str,
    split: str,
    conf: float,
    imgsz: int,
    resize_mode: str,
    device: str,
    limit: int | None = None,
) -> audit_recall.RecallReport:
    """Instance recall at the app's operating point, through `audit_recall`'s own matcher.

    Reused rather than reimplemented: that module's matching layer is the one with tests for
    class-awareness, one-to-one pairing and both label formats, and a second implementation
    here would be a second place for the label-format bug to live.
    """
    records = audit_recall.collect(
        generation, weights, split, conf, imgsz, resize_mode, device, limit
    )
    return audit_recall.measure(records, generation.classes, conf)


def record_path(weights: str) -> Path:
    """`models/<stem>.json` beside the weights - the record `--install` writes."""
    return Path(weights).with_suffix(".json")


def recorded_aggregates(record: Path) -> dict | None:
    """The record's most recent `test` measurement's aggregates, or None if there is none.

    The most recent rather than the first: a weight can be measured more than once, and the
    number worth printing beside today's is the last one taken.
    """
    if not record.exists():
        return None
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    validations = data.get("validation") or []
    for block in reversed(validations):
        if block.get("split") == "test" and block.get("aggregates"):
            return block["aggregates"]
    return None


def report_lines(
    *,
    weights: str,
    generation: Generation,
    split: str,
    frames: int,
    settings,
    configuration: str,
    device: str,
    isolated: Summary,
    in_app: Summary,
    timing: PipelineTiming,
    accuracy: float,
    targets: tuple[Target, ...],
    aggregates: dict | None,
) -> list[str]:
    """The whole reading as lines, so `main` is a print and the tests can read the number."""
    size = Path(weights).stat().st_size / 1e6
    out = [
        f"weights    {Path(weights).name}   ({size:.1f} MB)",
        f"dataset    {generation.export_dir}",
        f"split      {split}   generation {generation.name}   "
        f"{len(generation.classes)} classes   {frames} frames timed",
        f"config     {configuration}",
        f"device     {audit_recall.device_label(device)}   "
        f"capture {settings.capture_width}x{settings.capture_height}   imgsz {settings.imgsz}   "
        f"conf {settings.conf_threshold}   preview_height {settings.preview_height}   "
        f"frame_skip {settings.infer_frame_skip}",
        "",
        f"isolated   YoloDetector.infer over {isolated.n} frames, wall clock",
        f"           {isolated.line()}",
        "",
        f"in-app     Pipeline.process_once over {in_app.n} frames, wall clock - "
        "what a frame costs the sidecar",
        f"           {in_app.line()}",
    ]
    reported = timing.reported
    if reported is not None:
        out.append(
            f"           the pipeline's own latency_ms stat: {reported.line()}"
            "   <- detection only, the encode happens after"
        )
        if timing.reported_fps:
            out.append(f"           its infer_fps stat, last frame: {timing.reported_fps[-1]:.1f}")
    out += [
        "",
        f"accuracy   per-instance recall at the app's conf {audit_recall.DEFAULT_CONF} "
        "(matched class-aware, IoU >= 0.5)",
        f"           {accuracy:.3f}",
    ]
    if aggregates:
        out.append(
            f"           the record's --val aggregate recall {aggregates.get('recall', 0):.3f}"
            f"   mAP50 {aggregates.get('mAP50', 0):.3f}"
            f"   mAP50-95 {aggregates.get('mAP50-95', 0):.3f}   <- the validator's own"
            " best-F1 threshold, a different operating point"
        )
    else:
        out.append(
            "           no measurement recorded beside these weights, so --val has nothing"
            " to compare against (train_model.py --install records one)"
        )
    out += ["", "PRD targets"]
    out += [t.line() for t in targets]
    out.append("")
    out.append(f"  {target_summary(targets)}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Check the PRD's fps, latency and accuracy targets for a trained weight."
    )
    ap.add_argument(
        "--generation", choices=sorted(GENERATIONS), default=DEFAULT.name,
        help="whose dataset, class list and resize_mode to measure with",
    )
    ap.add_argument("--weights", default=None,
                    help="default: models/<generation>.pt as installed")
    ap.add_argument("--split", default=audit_recall.DEFAULT_SPLIT, help="train / valid / test")
    ap.add_argument("--limit", type=int, default=TIMING_FRAMES,
                    help=f"how many frames to time and score over (default {TIMING_FRAMES})")
    ap.add_argument("--defaults", action="store_true",
                    help="measure Settings()'s shipped defaults instead of this machine's "
                         f"saved profile ({SETTINGS_PATH})")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any PRD target fails, for use as a gate")
    ap.add_argument("--max-use-percent", type=int, default=resources.USE_PERCENT)
    ap.add_argument("--max-vram-percent", type=int, default=resources.VRAM_PERCENT)
    ap.add_argument("--disk-reserve-gb", type=float, default=resources.DISK_RESERVE_GB)
    ap.add_argument("--hard-vram-cap", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generation = generation_for(args.generation)
    weights = args.weights or str(SIDECAR_ROOT / "models" / generation.weight_name)
    if not Path(weights).exists():
        raise SystemExit(f"no such weight: {weights}")

    budget = resources.measure(args.max_use_percent, args.disk_reserve_gb, args.max_vram_percent)
    device = resources.resolve_device(args.device)
    budget = resources.apply(budget, device, args.hard_vram_cap)

    from app.inference import YoloDetector
    from app.pipeline import Pipeline
    from app.settings import Settings
    from app.settings_store import load_settings

    # The app's profile when there is one, its defaults when there is not: a check that described
    # a pipeline this machine does not run would pass a configuration nobody is using.
    saved = args.defaults is False and SETTINGS_PATH.exists()
    settings = load_settings(str(SETTINGS_PATH)) if saved else Settings()
    configuration = configuration_label(not args.defaults, SETTINGS_PATH, SETTINGS_PATH.exists())
    size = (settings.capture_width, settings.capture_height)
    frames = load_frames(generation, args.split, args.limit, size)

    detector = YoloDetector(
        weights, device=device, conf=settings.conf_threshold, imgsz=settings.imgsz,
        resize_mode=generation.resize_mode,
    )
    try:
        isolated = Summary.of(measure_detection(detector, frames))
        # The pipeline takes over the same detector and the same settings the sidecar would
        # build, so the in-app number is the app's number rather than a third configuration.
        messages: list[dict] = []
        pipeline = Pipeline(Cycling(frames), detector, settings, on_message=messages.append)
        timing = measure_pipeline(pipeline, frames)
    finally:
        detector.close()

    accuracy = measure_accuracy(
        generation, weights, args.split, settings.conf_threshold, settings.imgsz,
        generation.resize_mode, device,
    )
    targets = spec_targets(isolated, Summary.of(timing.wall_ms), accuracy.recall)

    for line in report_lines(
        weights=weights, generation=generation, split=args.split, frames=len(frames),
        settings=settings, configuration=configuration, device=device, isolated=isolated,
        in_app=Summary.of(timing.wall_ms), timing=timing, accuracy=accuracy.recall,
        targets=targets, aggregates=recorded_aggregates(record_path(weights)),
    ):
        print(line)

    print()
    print(
        f"resource envelope: {budget.cpu_threads} threads, RAM cap {budget.ram_cap_gb:.1f} GB, "
        f"VRAM cap {budget.vram_cap_gb:.1f} GB"
    )
    print("Per-class and per-crowding recall, and the conf/IoU sweeps: audit_recall.py")
    return 1 if args.strict and failed(targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
