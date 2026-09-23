#!/usr/bin/env python
"""Is the frame-clamp rule the right one, and does the suppression actually work?

`app/pipeline.py` drops detections whose box is pinned to all four frame edges, because these
weights produce exactly that shape on an empty counter and each one becomes a phantom row in the
item log. Two things about that are claims about *data* rather than about code, and the test suite
can check neither - it has no weights and no frames:

  * **the tolerance.** 0.01 is not a taste. It is a comparison between three populations: what the
    empty-counter negatives produce, what real product frames produce, and what the weights were
    taught. The table below is that comparison, and it is the entire justification for the
    constant - `CLAMPED_EDGE_TOLERANCE`'s own docstring is a summary of this output.
  * **the suppression.** That switching it on removes phantoms *and* costs no real detection -
    measured through the app's own `Pipeline.process_once`, which is where the filter lives, and
    not by calling the detector directly, which would skip the code under test.

Both need only data already on disk, so both are reproducible, and this is the reproduction.
`docs/CAPTURE_CHECKLIST.md` (Tier C2) quotes its numbers; before this existed those numbers came
from scratch scripts that were not tracked, so the doc cited a measurement with no way to re-run
it.

    python sidecar/tools/clamp_probe.py
    python sidecar/tools/clamp_probe.py --generation v1 --strict     # gate: exit 1 on a failure

**Everything here goes through the Pipeline, including the sweep.** That is the one structural
decision worth explaining. The obvious shape is a bare detector pass over the frames for the
distributions, then a separate pipeline pass for the end-to-end numbers - and it costs the same
inferences while letting the two halves describe different sets of frames. Instead each
population is swept twice, once with the filter off and once with it on, and the *off* pass is both
the source of the distributions and the control the on pass is compared against. The table and the
end-to-end block are therefore readings of one visit to each frame.

**The measurement is one number, not a boolean.** `worst_edge(box)` is how far a box sits from the
nearest frame edge *on its worst side*: `max(x1, y1, 1-x2, 1-y2)`. A box the model wanted larger
than the image comes back clamped, so all four of those are ~0; a centred object is near 0.5. Every
row of the table is a threshold on that one number, which is what makes this a sweep rather than a
sequence of guesses - and the populations are printed as distributions first, because two
populations that overlapped completely would make any threshold arbitrary, and that is worth seeing
rather than hiding behind a chosen value.

**It reads the tolerance the app ships rather than its own copy.** `app.pipeline` is imported
inside `main()`, so the row the report marks is the rule that is running. A tool holding a second
`0.01` would go on justifying a number nothing uses.

**The two settings that would silently change the answer are forced.** `infer_frame_skip` is pinned
to 0, because a profile with a skip set would have this visit every (skip+1)th frame and report the
fraction of the frames it happened to look at. `class_allowlist` is emptied, because a profile
narrowed to some classes would make the real population smaller than the frames it came from and
turn a phantom of an excluded class into a non-observation. Both are stated here rather than
inherited quietly: they are the difference between measuring the app and measuring a configuration.

No camera, no network. Reads the ignored dataset workspace.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import resources  # must precede numpy/torch: sets OMP/MKL thread limits

import audit_recall
from generations import DEFAULT, GENERATIONS, Generation
from generations import get as generation_for
from workspace import SIDECAR_ROOT

# This tool drives `app` code, and run as a script Python puts `tools/` on the path - not the
# sidecar root - so `from app.pipeline import ...` in `main()` would raise ModuleNotFoundError for
# whoever followed the doc. The test suite never sees it, because pytest.ini adds the sidecar root:
# the tool works under test and dies by hand. Same shim as spec_check.py.
sys.path.append(str(SIDECAR_ROOT))

# The ladder the shipped tolerance is judged against. It brackets the constant on **both** sides on
# purpose: the looser rows are the argument, because what 0.02 would buy and what it would cost is
# the reason 0.01 is where it is. A ladder that stopped at the shipped value could only show the
# cheaper direction, which is the direction nobody needs convincing about.
SWEEP_TOLERANCES = (0.002, 0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120)

#: The stored empty-counter frames - Tier C2b's hard negatives, and the set the defect was found
#: on. Absolute, so the tool reads the same directory however it was started.
NEGATIVES_DIR = SIDECAR_ROOT / "data" / "datasets" / "cleaned-negatives" / "negative"

#: The control population: frames that genuinely hold an item, from the weights' own export.
REAL_SPLIT = "train"
REAL_SAMPLE = 60

#: The app's own settings file, resolved absolutely: `settings_store.load_settings` reads a
#: relative path against the process cwd, which is the repo root when a person runs this by hand
#: and the sidecar dir when Electron spawns it. Naming the file outright is the only way the tool
#: reads the same one whichever way it was started. Same reasoning as spec_check.py.
SETTINGS_PATH = SIDECAR_ROOT / "data" / "settings.json"

#: How many instances a class needs in the phantom breakdown before it is worth naming. A single
#: stray detection is not a pattern, and the doc's point - that the phantoms are not one product -
#: is made by the classes that appear repeatedly.
MIN_PHANTOM_CLASS = 3

Box = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# The pure layer: the rule, the sweep, and every number the report prints
# ---------------------------------------------------------------------------


def worst_edge(box: Box) -> float:
    """How far a box sits from the nearest frame edge on its worst side. 0 is fully clamped.

    The maximum, not the minimum or a count of pinned sides. A real close-up overflows the frame on
    the sides the object leaves through, so its *best* side is often pinned too, and counting pinned
    sides would confuse it with a clamped prediction. What separates them is the worst side: a
    genuinely partial box always has one edge well inside the frame.

    Equivalent to the app's own `is_clamped_to_frame` at any given tolerance - `max(...) <= t` holds
    exactly when all four sides are within `t` - and the tests pin that equivalence, so this tool
    cannot drift into justifying a rule the pipeline does not apply.
    """
    x1, y1, x2, y2 = box
    return max(x1, y1, 1.0 - x2, 1.0 - y2)


@dataclass(frozen=True)
class RuleRow:
    """One candidate tolerance and what it costs on all three populations."""

    tolerance: float
    caught: int
    phantoms: int
    real_lost: int
    real: int
    labels_rejected: int
    labels: int

    @property
    def rejected_share(self) -> float:
        return self.labels_rejected / self.labels if self.labels else 0.0

    @property
    def is_free(self) -> bool:
        """Prices nothing: it catches something and loses no *measured* real detection."""
        return self.real_lost == 0 and self.caught > 0


@dataclass(frozen=True)
class Pass:
    """One population swept through `Pipeline.process_once` once, at one setting.

    `boxes` is what the pass saw *after* the filter, so the off pass is the raw model output and
    the on pass is what would reach the overlay and the item log. Keeping both is what lets the
    table and the end-to-end block describe the same frames.
    """

    frames: int
    fired: int
    detections: int
    suppressed: int
    boxes: tuple[Box, ...]
    classes: tuple[tuple[str, int], ...]

    @property
    def clean(self) -> int:
        """Frames that came back with nothing - what an operator reads as an empty counter."""
        return self.frames - self.fired


def sweep_tolerances(
    shipped: float, ladder: tuple[float, ...] = SWEEP_TOLERANCES
) -> tuple[float, ...]:
    """The ladder plus the shipped tolerance, sorted.

    The table exists to justify one specific number, so a table that happened to omit it would
    print a standing for a rule nothing runs - a failure mode worth designing out, since the
    constant and this file are edited by different people at different times.
    """
    return tuple(sorted({*ladder, round(float(shipped), 6)}))


def rule_table(
    phantoms: tuple[Box, ...],
    real: tuple[Box, ...],
    labels: tuple[Box, ...],
    tolerances: tuple[float, ...],
) -> list[RuleRow]:
    """Every candidate tolerance scored against all three populations. Pure, so it is testable."""
    return [
        RuleRow(
            tolerance=tolerance,
            caught=sum(1 for box in phantoms if worst_edge(box) <= tolerance),
            phantoms=len(phantoms),
            real_lost=sum(1 for box in real if worst_edge(box) <= tolerance),
            real=len(real),
            labels_rejected=sum(1 for box in labels if worst_edge(box) <= tolerance),
            labels=len(labels),
        )
        for tolerance in tolerances
    ]


def row_for(rows: list[RuleRow], tolerance: float) -> RuleRow | None:
    """The row standing for `tolerance`, or None if the ladder never reached it."""
    for row in rows:
        if abs(row.tolerance - tolerance) < 1e-9:
            return row
    return None


def loosest_free_row(rows: list[RuleRow]) -> RuleRow | None:
    """How far the rule could be loosened and still lose no measured real detection.

    Read as a bound on the evidence, not as advice. This is the *observed* cost on one sample of
    real frames, so the loosest free row is where the measurement runs out - not a recommendation
    to move the constant there, which would spend the whole margin on a few phantoms.
    """
    free = [row for row in rows if row.is_free]
    return max(free, key=lambda row: row.tolerance) if free else None


def next_looser_row(rows: list[RuleRow], tolerance: float) -> RuleRow | None:
    """The nearest candidate above `tolerance` - what relaxing the rule would actually cost."""
    looser = [row for row in rows if row.tolerance > tolerance + 1e-9]
    return min(looser, key=lambda row: row.tolerance) if looser else None


def distribution(edges: list[float]) -> str:
    """min / median / max of a population's worst-edge readings, as one readable line."""
    if not edges:
        return "no detections"
    ordered = sorted(edges)
    return f"min {ordered[0]:.4f}   median {statistics.median(ordered):.4f}   max {ordered[-1]:.4f}"


def phantom_classes(pass_: Pass, minimum: int = MIN_PHANTOM_CLASS) -> list[tuple[str, int]]:
    """Classes the off pass detected on the empty counter, most frequent first.

    Here because the actionable half of the finding is *which products*, and the answer is not
    obvious: these weights produce the clamped shape mostly on one class, so the intuitive fix
    (re-shoot the product the live session tripped over) would not have removed it. Restricted to
    the classes that recur, since a lone detection is noise rather than a pattern.
    """
    return [(name, n) for name, n in pass_.classes if n >= minimum]


@dataclass(frozen=True)
class Check:
    """One checkable statement about the shipped rule, with the reading behind it."""

    ok: bool
    claim: str
    reading: str

    def line(self) -> str:
        return f"  [{'PASS' if self.ok else 'FAIL'}] {self.claim:<52} {self.reading}"


def checks(
    off: Pass, on: Pass, real_off: Pass, real_on: Pass, shipped: RuleRow | None
) -> tuple[Check, ...]:
    """The statements the report stands behind, each with the number that decides it.

    A table alone would let a tolerance that had begun eating real detections print a middle column
    nobody reads. These are what `--strict` gates on, and every one of them is a sentence that must
    be true about the rule as shipped - including the two that separate "the filter works" from
    "the filter did nothing", which are the same reading on a badly chosen day.
    """
    if shipped is None:
        return (
            Check(
                False,
                "the shipped tolerance is on the ladder",
                "the table cannot justify a rule it does not contain",
            ),
        )

    return (
        Check(
            shipped.real_lost == 0,
            "no measured real detection is inside the rule",
            f"{shipped.real_lost} of {shipped.real} product detections",
        ),
        Check(
            shipped.caught > 0,
            "the rule catches the phantom shape at all",
            f"{shipped.caught} of {shipped.phantoms} empty-counter detections",
        ),
        Check(
            on.suppressed > 0,
            "the Pipeline drops detections for it",
            f"{on.suppressed} dropped over {on.frames} frames",
        ),
        Check(
            on.fired < off.fired,
            "fewer empty-counter frames show a phantom with it on",
            f"{on.fired} vs {off.fired} of {off.frames} frames",
        ),
        Check(
            real_on.detections == real_off.detections,
            "the Pipeline loses no real product detection",
            f"{real_off.detections - real_on.detections} lost of {real_off.detections}",
        ),
    )


def verdict(checks_: tuple[Check, ...]) -> str:
    """One sentence for the whole block, naming the failures rather than only counting them."""
    bad = [c for c in checks_ if not c.ok]
    if not bad:
        return f"all {len(checks_)} claims hold"
    return f"{len(bad)} of {len(checks_)} claims FAIL: " + "; ".join(c.claim for c in bad)


def strict_failed(checks_: tuple[Check, ...]) -> bool:
    """Whether a `--strict` run should exit non-zero."""
    return any(not c.ok for c in checks_)


# ---------------------------------------------------------------------------
# The frame source: the real files, one at a time
# ---------------------------------------------------------------------------


class Fixed:
    """A frame source holding whichever real frame the caller last put in it.

    The sweep has to walk a directory through the real `Pipeline` and there is no camera to give
    it - so this stands in for `CameraCapture`, implementing the two things `Pipeline` asks of a
    source: `latest()` returning `(seq, frame)`, and a `failure` attribute it checks every frame.
    Real frames rather than synthetic ones because the defect being measured *is* image content:
    an empty frame produces no phantom, so generated input would report a clean bill of health
    about a defect that needs product packaging in front of the lens to appear.
    """

    failure = None
    fps = 60.0

    def __init__(self):
        self._seq = 0
        self._frame = None

    def set(self, frame) -> None:
        self._seq += 1
        self._frame = frame

    def latest(self):
        return (self._seq, self._frame)


# ---------------------------------------------------------------------------
# The impure layer: files, frames, and the model
# ---------------------------------------------------------------------------


def labelled_frames(generation: Generation, split: str, limit: int) -> list[Path]:
    """Frames from `split` that carry at least one box, sampled evenly across the split.

    Empties are excluded explicitly, and that exclusion is the whole reason this is not
    `audit_recall.frame_paths` (which accepts any frame with a label *file*, including an empty
    one): the hard negatives were uploaded to this same project, so a frame with no boxes is a
    *negative* - the other population. Letting one into the control group would answer a different
    question than the one asked, and would do it quietly.

    A step rather than the first N, because an export is ordered by capture and the front of it is
    one session's lighting. "Real product frames" should mean the variety, not the beginning.
    """
    images, labels = audit_recall.split_dirs(generation, split)
    if not images.is_dir():
        raise SystemExit(f"no such split: {images} (download or generate the export first)")
    boxed = [
        path
        for path in sorted(images.glob("*.jpg"))
        if audit_recall.parse_labels((labels / f"{path.stem}.txt").read_text(encoding="utf-8"))
    ]
    if not boxed:
        raise SystemExit(f"{images} holds no labelled frames - there is no control population")
    step = max(1, len(boxed) // limit)
    return boxed[::step][:limit]


def training_labels(generation: Generation, split: str) -> tuple[Box, ...]:
    """Every labelled box in the split - the third population, and the one already on disk.

    It is the only population here that says what the weights were *taught*, which is what makes
    the last column readable: those are boxes that may legitimately look clamped, and their count
    is the honest price of the rule rather than a second opinion about the same frames.
    """
    _, labels = audit_recall.split_dirs(generation, split)
    if not labels.is_dir():
        raise SystemExit(f"no such split: {labels} (download or generate the export first)")
    out: list[Box] = []
    for path in sorted(labels.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        out.extend(instance.box for instance in audit_recall.parse_labels(text))
    return tuple(out)


def negative_frames(directory: Path) -> list[Path]:
    """The stored empty-counter frames, in a fixed order so two runs visit them identically."""
    if not directory.is_dir():
        raise SystemExit(
            f"no such directory: {directory}\n"
            "The stored empty-counter negatives are what this measures; without them there is no "
            "phantom population to compare against.\n"
            f"Expected: {NEGATIVES_DIR} (see CAPTURE_CHECKLIST.md, Tier C2)"
        )
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"{directory} holds no .jpg frames")
    return paths


def read_frames(paths: list[Path], cv2) -> list:
    """The frames themselves, refusing to carry on past an unreadable one.

    Skipping a frame that will not decode would shrink a denominator with no trace, and a phantom
    rate computed over an unknown number of frames is exactly the kind of number this tool exists
    to avoid - so this names the file and stops.
    """
    out = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"unreadable frame: {path}")
        out.append(image)
    return out


def sweep(frames: list, source: Fixed, pipeline) -> Pass:
    """Walk one population through `Pipeline.process_once` and count what came out.

    `suppressed` is accumulated from the frame message rather than inferred from a before/after
    difference, because that stat is the field the Live view shows - reading the same number the
    operator reads is what makes this a check on the shipped path rather than a parallel one.
    """
    fired = detections = suppressed = 0
    boxes: list[Box] = []
    classes: dict[str, int] = {}
    for frame in frames:
        source.set(frame)
        message = pipeline.process_once()
        if message is None:
            continue
        rows = message["detections"]
        detections += len(rows)
        suppressed += message["stats"]["suppressed"]
        if rows:
            fired += 1
        for row in rows:
            boxes.append(tuple(row["box"]))
            classes[row["cls"]] = classes.get(row["cls"], 0) + 1
    return Pass(
        frames=len(frames),
        fired=fired,
        detections=detections,
        suppressed=suppressed,
        boxes=tuple(boxes),
        classes=tuple(sorted(classes.items(), key=lambda kv: (-kv[1], kv[0]))),
    )


def probe(
    negatives: list, real: list, settings, source: Fixed, pipeline
) -> tuple[Pass, Pass, Pass, Pass]:
    """Both populations, filter off then on, through one live Pipeline.

    One pipeline and one `Settings` instance are reused and only the flag is flipped, because that
    is the live-reload path the app actually uses: mutating a setting in place is what
    `PATCH /api/settings` does, and a second object would test a start-up configuration nobody
    runs mid-session. It also makes the comparison sound by construction - if the flip did not take
    effect, the on pass would simply reproduce the off pass, which the claims catch.

    Off first, because the off pass *is* the raw model output: it is both the control and the source
    of the distributions the table is built from.
    """
    settings.suppress_clamped_detections = False
    off = sweep(negatives, source, pipeline)
    real_off = sweep(real, source, pipeline)

    settings.suppress_clamped_detections = True
    on = sweep(negatives, source, pipeline)
    real_on = sweep(real, source, pipeline)

    return off, on, real_off, real_on


def report_lines(
    *,
    generation: Generation,
    weights: str,
    split: str,
    device: str,
    tolerance: float,
    imgsz: int,
    rows: list[RuleRow],
    off: Pass,
    on: Pass,
    real_off: Pass,
    real_on: Pass,
    labels: int,
    conf: float,
    shipped_conf: float,
    checks_: tuple[Check, ...],
) -> list[str]:
    """The whole report, pure and printable - one function so a caller cannot print half of it."""
    shipped = row_for(rows, tolerance)
    free = loosest_free_row(rows)
    looser = next_looser_row(rows, tolerance)

    out = [
        f"weights    {weights}",
        f"dataset    {generation.export_dir}",
        f"split      {split}   generation {generation.name}   {len(generation.classes)} classes",
        f"negatives  {off.frames} frames   {NEGATIVES_DIR}",
        f"device     {audit_recall.device_label(device)}   tolerance {tolerance}",
        "           (tolerance read from app.pipeline.CLAMPED_EDGE_TOLERANCE)",
        f"operating  conf>={conf}   imgsz {imgsz}   every number below is Pipeline.process_once",
    ]
    # Spelled out rather than left to be inferred from the line above, because this is the one
    # setting the populations are *not* stable across: a phantom is a low-confidence detection, so
    # a profile at a higher threshold measures fewer of them and makes the defect look smaller.
    # A number quoted from the published table was measured at a particular threshold, and the two
    # numbers being different is the finding - not a footnote.
    if abs(conf - shipped_conf) > 1e-9:
        out.append(
            f"           this profile runs conf>={conf}, while Settings() ships "
            f"{shipped_conf} - the counts scale with it, so quote which one a number came from"
        )
    out += [
        "",
        "the rule, on two populations measured against each other",
        "  worst-edge distance from the nearest frame edge, on a box's worst side (0 = clamped)",
        f"  phantoms  ({len(off.boxes)} detections on {off.frames} empty-counter frames)",
        f"    {distribution([worst_edge(b) for b in off.boxes])}",
        f"  real      ({len(real_off.boxes)} detections on {real_off.frames} product frames)",
        f"    {distribution([worst_edge(b) for b in real_off.boxes])}",
        "",
    ]

    if free is not None:
        out.append(
            f"  separated below {free.tolerance}: the loosest tolerance that still loses no "
            "measured real detection"
        )
    else:
        out.append("  NO tolerance on this ladder is free - the rule is costing real detections")
    if looser is not None and shipped is not None:
        out.append(
            f"  loosening to {looser.tolerance} would catch "
            f"{looser.caught - shipped.caught} more phantom(s) and lose {looser.real_lost} "
            "real detection(s)"
        )
    out.append("")

    out.append(
        f"  {'tolerance':>10} {'phantoms caught':>18} {'real lost':>12} {'labels rejected':>18}"
    )
    for row in rows:
        current = shipped is not None and abs(row.tolerance - shipped.tolerance) < 1e-9
        mark = "  <-- shipped" if current else ""
        out.append(
            f"  {row.tolerance:>10.3f} {row.caught:>10}/{row.phantoms:<7} "
            f"{row.real_lost:>6}/{row.real:<5} "
            f"{row.labels_rejected:>8}/{row.labels:<8} ({row.rejected_share:>6.2%}){mark}"
        )
    out.append("")
    out.append(
        f"  the last column is the honest cost: {labels} labelled boxes in {split}, and the "
        "share of them"
    )
    out.append("  that touch all four edges - an item that fills the frame is the case the rule")
    out.append("  can mistake for a phantom, which is why the suppression is a setting with an")
    out.append("  off switch")

    out += [
        "",
        "through the real Pipeline.process_once, the setting flipped in place (no restart)",
        f"  {'':<34}{'filter off':>14}{'filter on':>12}",
        f"  {'empty-counter frames, phantom shown':<34}{off.fired:>8}/{off.frames:<5}"
        f"{on.fired:>6}/{on.frames:<5}",
        f"  {'empty-counter detections':<34}{off.detections:>14}{on.detections:>12}"
        f"   ({on.suppressed} dropped by the filter)",
        f"  {'product frames, detections':<34}{real_off.detections:>14}{real_on.detections:>12}"
        f"   over {real_off.frames} frames",
    ]

    named = phantom_classes(off)
    if named:
        out.append("")
        out.append(f"  phantoms by class on the empty counter (>= {MIN_PHANTOM_CLASS} detections):")
        for name, n in named:
            out.append(f"    {name:<52}{n:>4}")
        out.append(
            "    the defect is mostly one or two classes, so re-shooting the product a live session"
        )
        out.append("    happened to trip over would not have removed it")

    out += ["", "claims", *[c.line() for c in checks_], "", f"  {verdict(checks_)}"]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Check the frame-clamp tolerance and the phantom suppression against real data."
    )
    ap.add_argument(
        "--generation", choices=sorted(GENERATIONS), default=DEFAULT.name,
        help="whose class list, resize_mode and export to measure against",
    )
    ap.add_argument("--weights", default=None, help="default: models/<generation>.pt as installed")
    ap.add_argument("--negatives", default=str(NEGATIVES_DIR),
                    help="the stored empty-counter frames (default: Tier C2b's hard negatives)")
    ap.add_argument("--split", default=REAL_SPLIT, help="where the control frames come from")
    ap.add_argument("--real-sample", type=int, default=REAL_SAMPLE,
                    help=f"how many product frames to use as the control (default {REAL_SAMPLE})")
    ap.add_argument("--conf", type=float, default=None,
                    help="detection confidence to measure at; default is the app's own profile"
                         " (Settings.conf_threshold). The populations are not stable across it -"
                         " a phantom is a low-confidence detection, so a higher threshold hides"
                         " them and makes the defect look smaller than it is.")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any claim fails, for use as a gate")
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

    import cv2

    from app.inference import YoloDetector
    from app.pipeline import CLAMPED_EDGE_TOLERANCE, Pipeline
    from app.settings import Settings
    from app.settings_store import load_settings

    # The app's own profile when there is one, its defaults when there is not - `conf_threshold` and
    # `imgsz` are the operating point every detection here is produced at, so measuring the shipped
    # defaults on a machine that runs something else would describe a pipeline nobody uses.
    settings = load_settings(str(SETTINGS_PATH)) if SETTINGS_PATH.exists() else Settings()
    if args.conf is not None:
        settings.conf_threshold = args.conf
    # Forced, for the two reasons the module docstring gives: a frame skip would report the
    # fraction of frames this happened to look at, and an allowlist would shrink the real population
    # by class - which turns a phantom of an excluded class into a non-observation.
    settings.infer_frame_skip = 0
    settings.class_allowlist = []

    negatives = read_frames(negative_frames(Path(args.negatives)), cv2)
    real = read_frames(labelled_frames(generation, args.split, args.real_sample), cv2)

    detector = YoloDetector(
        weights, device=device, conf=settings.conf_threshold, imgsz=settings.imgsz,
        resize_mode=generation.resize_mode,
    )
    source = Fixed()
    pipeline = Pipeline(source, detector, settings, on_message=lambda _m: None)
    try:
        off, on, real_off, real_on = probe(negatives, real, settings, source, pipeline)
    finally:
        detector.close()

    labels = training_labels(generation, args.split)
    rows = rule_table(off.boxes, real_off.boxes, labels, sweep_tolerances(CLAMPED_EDGE_TOLERANCE))
    checks_ = checks(off, on, real_off, real_on, row_for(rows, CLAMPED_EDGE_TOLERANCE))

    for line in report_lines(
        generation=generation, weights=weights, split=args.split, device=device,
        tolerance=CLAMPED_EDGE_TOLERANCE, imgsz=settings.imgsz, rows=rows, off=off, on=on,
        real_off=real_off, real_on=real_on, labels=len(labels),
        conf=settings.conf_threshold, shipped_conf=Settings().conf_threshold, checks_=checks_,
    ):
        print(line)

    print()
    print(
        f"resource envelope: {budget.cpu_threads} threads, RAM cap {budget.ram_cap_gb:.1f} GB, "
        f"VRAM cap {budget.vram_cap_gb:.1f} GB"
    )
    print("The rule itself, and where it is applied: app/pipeline.py (CLAMPED_EDGE_TOLERANCE)")
    print("Per-class recall, and the conf/IoU sweeps: audit_recall.py")
    return 1 if args.strict and strict_failed(checks_) else 0


if __name__ == "__main__":
    raise SystemExit(main())
