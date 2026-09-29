"""Capture hard-negative frames for retraining (the producer for Tier C2b's `NEGATIVES/`).

False positives like a face firing as `safeguard` are fixed at the root by teaching the model that
such scenes contain *nothing* worth detecting. In YOLO, an image whose label file is empty (or
missing) is treated as pure background: it contributes no positive loss and actively suppresses
detections on whatever appears in it.

This tool saves frames from the checkout camera (or any video file) into the folder layout
`docs/CAPTURE_CHECKLIST.md` §Tier C2b describes:

    <out>/images/<source>_<index>.jpg   saved frames
    <out>/labels/<source>_<index>.txt   EMPTY label files (background)
    <out>/data.yaml                     declaration for a background-only set

Run it while standing in the exact pose/scene that produces the false positive (a face leaning over
the checkout counter, close to the lens). Shoot variety: different angles, distances, lighting. Aim
for roughly 5-10% of your total dataset size — more than that hurts recall on real products.

Usage (run from sidecar/):
    .venv/Scripts/python.exe tools/capture_hard_negatives.py --source 0
    .venv/Scripts/python.exe tools/capture_hard_negatives.py --source 0 --interval 1.0 --max-frames 300
    .venv/Scripts/python.exe tools/capture_hard_negatives.py --source footage.mp4

The frames are then v2 data like any other shoot, not a folder to copy into a set by hand: stage
them with `clean_v2.py clean --negatives <out>/images --out <staged>`, upload them as their own
session batch (`--session negN`), and mark each one with the annotator's null tool (N) — an unmarked
frame is silently excluded from the version. `docs/CAPTURE_CHECKLIST.md` §Tier C2b carries the two
commands.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.camera import _default_capture  # noqa: E402
from workspace import DATA_YAML_NAME  # noqa: E402  (the artifact name the readers look for)

DEFAULT_OUT = "data/datasets/hard_negatives"


def _source_label(source) -> str:
    """Stable filename prefix: 'cam0' for camera indices, video stem otherwise."""
    s = str(source)
    if s.isdigit():
        return f"cam{s}"
    return Path(s).stem.replace(" ", "_")


def _next_index(out_images: Path, stem: str) -> int:
    """Continue numbering after any previous run so re-runs never overwrite."""
    highest = -1
    for p in out_images.glob(f"{stem}_*.jpg"):
        suffix = p.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def run(
    source,
    out: str = DEFAULT_OUT,
    interval: float = 2.0,
    max_frames: int = 200,
    capture_factory=_default_capture,
    now_fn=time.monotonic,
) -> int:
    """Capture up to max_frames frames, spaced >= interval seconds apart.

    Returns the number of frames saved. Raises RuntimeError if the source fails to open or dies
    repeatedly mid-capture. The capture factory defaults to the app's own `_default_capture`, so the
    device opens over the same backend the running app uses (MSMF on Windows, which is where this
    camera's 60 fps lives) — a negatives set shot through a different OpenCV backend is a set of
    frames from a camera state the model never sees at runtime.
    """
    if interval < 0:
        raise ValueError("interval cannot be negative")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    out_images = Path(out) / "images"
    out_labels = Path(out) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stem = _source_label(source)
    index = _next_index(out_images, stem)

    cap = capture_factory(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open source: {source!r}")

    saved = 0
    last_save = now_fn()
    failures = 0
    try:
        while saved < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 30:
                    raise RuntimeError(
                        "Source stopped producing frames "
                        f"(camera unplugged or video exhausted); saved {saved} frames."
                    )
                time.sleep(0.05)
                continue
            failures = 0
            now = now_fn()
            if now - last_save < interval:
                continue
            last_save = now
            name = f"{stem}_{index:06d}"
            cv2.imwrite(str(out_images / f"{name}.jpg"), frame)
            # Empty label file == YOLO background image.
            (out_labels / f"{name}.txt").touch()
            index += 1
            saved += 1
            print(f"saved {name}.jpg ({saved}/{max_frames})", flush=True)
    finally:
        cap.release()

    _write_data_yaml(Path(out))
    return saved


def _write_data_yaml(out: Path) -> None:
    # Written under `workspace.DATA_YAML_NAME` rather than a literal, so the name is the one the
    # readers look for. `as_posix()` because the path is rendered into a file another program reads:
    # a Windows backslash path is what a shell eats as an escape, and this project has already paid
    # for that once in the scaffold's README commands.
    (out / DATA_YAML_NAME).write_text(
        "# Background/hard-negative images only: every label file is empty.\n"
        "# Ultralytics treats empty-label images as pure background during\n"
        "# training, which suppresses false positives on these scenes.\n"
        "# Merge into your main dataset (or train alongside it) - do NOT train\n"
        "# on this folder alone: it defines no classes, so it is not a model.\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: placeholder\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save camera/video frames as YOLO background images "
        "(empty labels) for false-positive retraining."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file. Default: 0",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output dataset dir. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between saved frames. Default: 2.0",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Stop after this many frames. Default: 200",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    try:
        saved = run(source, out=args.out, interval=args.interval, max_frames=args.max_frames)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out_path = Path(args.out).resolve()
    print(
        f"\nDone: {saved} background frames in {out_path / 'images'} "
        f"(empty labels in {out_path / 'labels'})."
    )
    print("Next: stage them as their own set and mark each frame with the annotator's null tool -")
    print("  clean_v2.py clean --negatives <out>/images --out <staged>  (then upload --session negN)")
    print("docs/CAPTURE_CHECKLIST.md §Tier C2b carries both commands and the marking step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
