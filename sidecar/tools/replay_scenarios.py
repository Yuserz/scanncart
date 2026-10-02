#!/usr/bin/env python
"""The counting-accuracy corpus (spec §7.1): replay the recorded scenarios, write the fixtures.

The model answers "which products are in this frame, and where". Whether the *cart* is right is
something `desktop/src/main/cartState.ts` infers from how detections persist over time — the dwell,
the settle window, concurrent tracks — and that inference is what a checkout actually depends on.
Per-class recall (`train_model.py --val`, `RECALL_FLOOR`) does not answer it: a product seen in 85%
of frames can still be counted perfectly, because the windows average over many frames, and one seen
in 99% of frames can still be counted wrong when two identical items touch and come back as one box.

So the inference gets its own corpus: short videos of scripted counters, shot with the real camera
where it will be mounted, each beside a script listing what the cart should hold at each moment.
This tool replays a video through the app's own `Pipeline` and writes the resulting track log as a
fixture the desktop suite scores with no camera and no GPU.

    sidecar/.venv/Scripts/python.exe sidecar/tools/replay_scenarios.py
    sidecar/.venv/Scripts/python.exe sidecar/tools/replay_scenarios.py --only 03_two_identical

**The replay is deterministic because `Pipeline` already takes its clock.** `clock: Callable[[], float]`
was there for the tests, and `_log_detections` timestamps every event through it — so time here is
the *video's* own frame timestamps, and two runs over one file produce the same log. Nothing is
monkeypatched, and no wall clock reaches an event.

**It does not score.** The derivation has exactly one implementation, `cartState.ts`; a Python copy
in this file would be a second definition of the dwell/settle rule, free to disagree with the one the
app runs, and a corpus scored by the wrong rule reads as a healthy model. The scoring is
`desktop/src/main/cartState.scenarios.test.ts`, which loads these fixtures and asserts each
checkpoint. This tool's whole output is the evidence that test reads.

**One detector per scenario, thrown away after it.** `YoloDetector` calls `track(..., persist=True)`,
which keeps the tracker's state between calls — so a detector shared across recordings would carry a
track id (and the class identity attached to it) from one video into the next, and the second
scenario's events would begin with the first one's tracks still open. Model load per video costs a
few seconds; a contaminated fixture costs the corpus.

**`infer_frame_skip` is forced to 0, and only that.** A profile with a skip set would have this visit
every (skip+1)th frame and record a log of the fraction it happened to look at. The preview settings
are also reset (`preview_mirror` off, a small `preview_height`) for cost alone: the preview is
rendered *after* `_log_detections`, so it cannot change what an event says — which is the test that
separates a knob like this from the frame skip. Everything else is the app's own settings, because
they decide what the app logs: `conf_threshold`, `track_expiry_s`, the three suppressions and
`class_allowlist` are recorded into the fixture as provenance rather than overridden here.

**The corpus is not in git; the fixtures are.** Videos and scripts live in the gitignored workspace
(`sidecar/data/scenarios/`), which is why this is not part of `make test` and fails loudly when the
corpus is absent instead of skipping. The fixtures are small JSON, they are tracked, and they are
what `npm test` scores on a fresh clone.

The contract for both files — the script a human writes and the fixture this writes — is in
`desktop/src/main/__fixtures__/scenarios/README.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from workspace import SIDECAR_ROOT

# Root-relative, like the other tools: `app.*` is importable only once the sidecar root is on the
# path, and the import is deferred to `main()` so this module can be imported by a test that must
# not drag torch, ultralytics or cv2 into the suite.
sys.path.append(str(SIDECAR_ROOT))

REPO_ROOT = SIDECAR_ROOT.parent
DEFAULT_CORPUS = SIDECAR_ROOT / "data" / "scenarios"
DEFAULT_OUT = REPO_ROOT / "desktop" / "src" / "main" / "__fixtures__" / "scenarios"

#: What a recorded scenario can be. The value is only ever handed to `cv2.VideoCapture`, but naming
#: them keeps a `.txt` beside a script from being read as a video and reported as a decode failure.
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

#: The keys a script may carry. Unknown ones are an error rather than ignored: `checkpoint:` for
#: `checkpoints:` would otherwise mean "this script asserts nothing", which is a corpus that passes
#: by being empty.
SCRIPT_KEYS = frozenset({"name", "description", "bind_at_s", "asserted_with", "checkpoints"})
CHECKPOINT_KEYS = frozenset({"t_s", "cart"})
ASSERTED_KEYS = frozenset({"commit_dwell_s", "remove_settle_s", "min_commit_conf"})


class ScriptError(ValueError):
    """A scenario script that cannot be trusted to mean what it says."""


@dataclass(frozen=True)
class Checkpoint:
    """One moment of a scenario: what the cart should hold at `t_s` seconds into the video."""

    t_s: float
    cart: dict[str, int]


@dataclass(frozen=True)
class ScenarioScript:
    """A recorded scenario's written ground truth, read from `<name>.json` beside its video."""

    name: str
    description: str
    bind_at_s: float
    checkpoints: tuple[Checkpoint, ...]
    asserted_with: dict[str, float]


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScriptError(f"{where}: expected a number, got {value!r}")
    return float(value)


def load_script(path: Path) -> ScenarioScript:
    """Parse one scenario script, rejecting anything whose meaning is not pinned down.

    Every rejection here is a shape that would otherwise *pass*: a misspelled `checkpoints` key, a
    checkpoint whose `t_s` is a string, a quantity of 0 (which the derivation expresses by leaving
    the class out, so it is a mistake rather than a state), or two checkpoints answering for the same
    moment. A script that loads is one a failure message can be trusted about.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScriptError(f"{path.name}: not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScriptError(f"{path.name}: expected a JSON object")

    unknown = sorted(set(raw) - SCRIPT_KEYS)
    if unknown:
        raise ScriptError(f"{path.name}: unknown key(s) {unknown}; allowed: {sorted(SCRIPT_KEYS)}")

    bind_at = _number(raw.get("bind_at_s"), f"{path.name}: bind_at_s")

    checkpoints_raw = raw.get("checkpoints")
    if not isinstance(checkpoints_raw, list) or not checkpoints_raw:
        raise ScriptError(
            f"{path.name}: 'checkpoints' must be a non-empty list — a script that asserts nothing "
            "would pass by being empty"
        )

    seen: set[float] = set()
    checkpoints: list[Checkpoint] = []
    for index, entry in enumerate(checkpoints_raw):
        where = f"{path.name}: checkpoints[{index}]"
        if not isinstance(entry, dict):
            raise ScriptError(f"{where}: expected an object")
        extra = sorted(set(entry) - CHECKPOINT_KEYS)
        if extra:
            raise ScriptError(f"{where}: unknown key(s) {extra}; allowed: {sorted(CHECKPOINT_KEYS)}")
        t_s = _number(entry.get("t_s"), f"{where}.t_s")
        if t_s in seen:
            raise ScriptError(f"{where}.t_s: {t_s} answers for a moment another checkpoint already does")
        seen.add(t_s)
        cart = entry.get("cart")
        if not isinstance(cart, dict):
            raise ScriptError(f"{where}.cart: expected an object of class -> quantity")
        counted: dict[str, int] = {}
        for class_name, quantity in cart.items():
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
                raise ScriptError(
                    f"{where}.cart[{class_name!r}]: expected a whole number >= 1; an absent class "
                    "is expressed by leaving it out"
                )
            counted[str(class_name)] = quantity
        checkpoints.append(Checkpoint(t_s=t_s, cart=counted))

    asserted_raw = raw.get("asserted_with", {})
    if not isinstance(asserted_raw, dict):
        raise ScriptError(f"{path.name}: asserted_with must be an object")
    extra = sorted(set(asserted_raw) - ASSERTED_KEYS)
    if extra:
        raise ScriptError(
            f"{path.name}: asserted_with has unknown key(s) {extra}; allowed: {sorted(ASSERTED_KEYS)}"
        )
    asserted = {
        key: _number(value, f"{path.name}: asserted_with.{key}")
        for key, value in asserted_raw.items()
    }

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ScriptError(f"{path.name}: description must be a string")
    name = raw.get("name", path.stem)
    if not isinstance(name, str) or not name:
        raise ScriptError(f"{path.name}: name must be a non-empty string")

    return ScenarioScript(
        name=name,
        description=description,
        bind_at_s=bind_at,
        checkpoints=tuple(sorted(checkpoints, key=lambda c: c.t_s)),
        asserted_with=asserted,
    )


def video_for(script_path: Path) -> Path | None:
    """The video recorded for this script, or None — matched by stem, first suffix wins."""
    for suffix in VIDEO_SUFFIXES:
        candidate = script_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def scripts_in(corpus: Path) -> list[Path]:
    """Every scenario script in the corpus, in a fixed order so two runs visit them identically."""
    return sorted(corpus.glob("*.json"))


def event_document(event, session_id: int) -> dict:
    """One stored track as the desktop's `TrackEvent` expects it (camelCase at the boundary).

    The keys match `cartState.ts` rather than the database's, because this is the shape the test
    reads: `entered_at`/`left_at` are the sidecar's names, and a fixture that kept them would need a
    translation step in the suite, which is one more place for two spellings of one fact.
    """
    return {
        "sessionId": session_id,
        "trackId": event.track_id,
        "className": event.class_name,
        "maxConf": event.max_conf,
        "enteredAt": event.entered_at,
        "leftAt": event.left_at,
    }


def fixture_document(
    script: ScenarioScript,
    events: list[dict],
    *,
    video: Path,
    weights: str,
    device: str,
    pipeline_settings: dict,
    frames: int,
    duration_s: float,
) -> dict:
    """The fixture for one scenario: the script's ground truth and the replayed track log.

    Both halves travel together on purpose. The checkpoints are a copy of the script's rather than a
    reference to it, because the script lives in the gitignored workspace and the fixture is what CI
    scores — a fixture that pointed at a file a fresh clone does not have would be a test that
    silently asserts nothing.
    """
    document = {
        "name": script.name,
        "description": script.description,
        "video": video.name,
        "weights": weights,
        "device": device,
        "bind_at_s": script.bind_at_s,
        "frames": frames,
        "duration_s": duration_s,
        "pipeline": pipeline_settings,
        "checkpoints": [
            {"t_s": checkpoint.t_s, "cart": checkpoint.cart} for checkpoint in script.checkpoints
        ],
        "events": events,
    }
    if script.asserted_with:
        # Provenance, not an input: the test derives under the app's *current* settings (that is what
        # makes tuning them a test run), and this is what the person who wrote the checkpoints
        # assumed. A failing checkpoint names it, so a mismatch between the two is readable rather
        # than mysterious.
        document["asserted_with"] = script.asserted_with
    return document


def write_fixture(out_dir: Path, fixture: dict) -> Path:
    """Write `<name>.json`, creating the directory — the file is a record, so it is overwritten."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fixture['name']}.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The impure layer: files, frames, and the model
# ---------------------------------------------------------------------------


class ReplayClock:
    """The video's timeline, as something `Pipeline` can call.

    A mutable second-hand rather than a callable reading cv2: the pipeline is handed this object and
    asks it for `now` whenever it timestamps an event, and the replay moves it to each frame's own
    timestamp just before that frame is processed. That is the whole determinism story — nothing
    reads the wall clock, so the log is a function of the file.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class VideoSource:
    """A frame source holding whichever frame the replay last decoded.

    `Pipeline` asks a source for `latest() -> (seq, frame) | None`, an optional `failure`, and
    `fps`/`measured_fps` for its readout — that is the whole protocol (`CameraCapture` and the
    tools' own doubles agree on it). Real frames rather than synthetic ones, for the same reason
    `clamp_probe` uses them: the corpus is about what the weights do with *product packaging in
    front of the lens*, and a generated frame would report a clean bill of health about a scene
    nothing was ever placed in.
    """

    failure = None

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self._seq = 0
        self._frame = None

    def set(self, frame) -> None:
        self._seq += 1
        self._frame = frame

    def latest(self):
        return (self._seq, self._frame)

    def release(self) -> None:
        """Nothing to release — the decoder is owned by the replay, not by this double."""


def replay(
    video: Path,
    script: ScenarioScript,
    settings,
    store,
    *,
    cv2,
    Pipeline,
    detector,
    device: str,
):
    """Walk one video through the app's Pipeline at the video's own timestamps.

    Returns `(events, frames, duration_s)`. Called directly rather than through `Pipeline.start()`,
    because the thread would sleep and race the clock; `process_once` is the same code the app runs,
    one frame at a time.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video} (is it a video cv2 can decode?)")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    clock = ReplayClock()
    source = VideoSource(fps)
    session_id = store.start_session(script.name, device)
    pipeline = Pipeline(
        source,
        detector,
        settings,
        on_message=lambda _msg: None,
        logging_store=store,
        session_id=session_id,
        clock=clock,
    )

    frames = 0
    duration_s = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # The frame's own timestamp, in seconds: the container's timeline is the one the script
            # author watched while writing `t_s`, so checkpoints and events are the same clock.
            duration_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            clock.now = duration_s
            source.set(frame)
            pipeline.process_once()
            frames += 1
    finally:
        cap.release()
    # The same close capture-stop performs: everything still open is resolved at its last-seen
    # time, so a fixture's last item is not left reading as present forever.
    pipeline.resolve_open_tracks()

    rows = store.query_events(session_id)
    events = [event_document(row, session_id) for row in rows]
    return events, frames, duration_s


def pipeline_provenance(settings, resize_mode: str) -> dict:
    """The sidecar settings that shaped the events, recorded so a fixture can be read honestly.

    All of these decide what the app *logs* — the confidence floor, when a track expires, which
    shapes and classes are declined, and the geometry the boxes were normalized against — so a
    fixture whose answers look wrong is diagnosable from the file alone.
    """
    return {
        "conf_threshold": settings.conf_threshold,
        "imgsz": settings.imgsz,
        "track_expiry_s": settings.track_expiry_s,
        "infer_frame_skip": settings.infer_frame_skip,
        "resize_mode": resize_mode,
        "suppress_clamped_detections": settings.suppress_clamped_detections,
        "suppress_frame_filling_detections": settings.suppress_frame_filling_detections,
        "suppress_unsure_phantoms": settings.suppress_unsure_phantoms,
        "class_allowlist": list(settings.class_allowlist),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Replay the recorded counter scenarios and write the desktop fixtures."
    )
    ap.add_argument(
        "--corpus", default=str(DEFAULT_CORPUS),
        help="directory holding <name>.json scripts beside their videos (default: the workspace)",
    )
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="where the fixtures are written (default: desktop/src/main/__fixtures__/scenarios)",
    )
    ap.add_argument(
        "--model", default=None,
        help="weights to replay with (default: the app's settings.json active_model)",
    )
    ap.add_argument("--conf", type=float, default=None, help="default: the app's conf_threshold")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--only", default=None, help="replay just the scenarios whose name contains this")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        raise SystemExit(
            f"no such directory: {corpus}\n"
            "The scenario corpus is recorded, not checked in — shoot the scripts in spec §7.1 with "
            "the counter camera and put each video beside its <name>.json script.\n"
            f"Expected: {DEFAULT_CORPUS}"
        )
    scripts = [path for path in scripts_in(corpus) if args.only is None or args.only in path.stem]
    if not scripts:
        raise SystemExit(
            f"{corpus} holds no scenario scripts"
            + (f" matching {args.only!r}" if args.only else "")
            + ".\nEach scenario is a <name>.json script beside its recorded video; the format is "
            "documented in desktop/src/main/__fixtures__/scenarios/README.md."
        )

    # Loaded before any heavy import, so a missing corpus or script is reported as itself rather
    # than as an import error — and so the suite can exercise these paths without cv2 or torch.
    scripts = [(path, load_script(path)) for path in scripts]
    missing = [(path, script) for path, script in scripts if video_for(path) is None]
    if missing:
        names = ", ".join(f"{path.stem} (want one of {', '.join(VIDEO_SUFFIXES)})" for path, _ in missing)
        raise SystemExit(f"no video beside: {names}\nA script without its recording asserts nothing.")

    import resources  # must precede numpy/torch: sets OMP/MKL thread limits
    import cv2

    from app.inference import YoloDetector
    from app.logging_store import LoggingStore
    from app.models import requirement_for
    from app.pipeline import Pipeline
    from app.settings import Settings
    from app.settings_store import DEFAULT_SETTINGS_PATH, load_settings, resolve_resize_mode

    # The tools' own resolver, with the budget applied from the same reading: this is a batch job,
    # not the app, and it should leave the machine usable while it runs.
    device = resources.resolve_device(args.device)
    resources.apply(resources.measure(), device, False)

    # The app's own profile when there is one, defaults when there is not: the corpus is about the
    # pipeline that ships, so replaying at some other operating point would measure a configuration.
    settings = load_settings(str(DEFAULT_SETTINGS_PATH)) if DEFAULT_SETTINGS_PATH.exists() else Settings()
    if args.conf is not None:
        settings.conf_threshold = args.conf
    weights = args.model or settings.active_model
    if not Path(weights).exists():
        raise SystemExit(f"no such weight: {weights} (pass --model, or install one first)")
    settings.infer_frame_skip = 0
    settings.preview_mirror = False
    settings.preview_height = 64
    resize_mode = resolve_resize_mode(settings.resize_mode, weights, requirement_for(weights))
    provenance = pipeline_provenance(settings, resize_mode)

    store = LoggingStore(":memory:")
    out_dir = Path(args.out)
    try:
        for path, script in scripts:
            video = video_for(path)
            assert video is not None  # checked above, before the heavy imports
            detector = YoloDetector(
                weights, device=device, conf=settings.conf_threshold, imgsz=settings.imgsz,
                resize_mode=resize_mode,
            )
            try:
                events, frames, duration_s = replay(
                    video, script, settings, store,
                    cv2=cv2, Pipeline=Pipeline, detector=detector, device=device,
                )
            finally:
                detector.close()
            fixture = fixture_document(
                script, events, video=video, weights=weights, device=device,
                pipeline_settings=provenance, frames=frames, duration_s=duration_s,
            )
            written = write_fixture(out_dir, fixture)
            print(
                f"{script.name}: {frames} frames over {duration_s:.1f}s, {len(events)} tracks, "
                f"{len(script.checkpoints)} checkpoint(s) -> {written}"
            )
    finally:
        store.close()

    print()
    print("scored by: desktop/src/main/cartState.scenarios.test.ts (npx vitest run from desktop/)")
    print("The rule itself, and its one implementation: desktop/src/main/cartState.ts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
