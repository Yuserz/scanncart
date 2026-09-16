"""Capture hard-negative frames for YOLO retraining.

False positives like a face firing as 'safeguard' are fixed at the root by
teaching the model that such scenes contain *nothing* worth detecting. In
YOLO, an image whose label file is empty (or missing) is treated as pure
background: it contributes no positive loss and actively suppresses
detections on whatever appears in it.

This tool saves frames from the checkout camera (or any video file) into an
Ultralytics-ready folder:

    <out>/images/<source>_<index>.jpg   saved frames
    <out>/labels/<source>_<index>.txt   EMPTY label files (background)
    <out>/data.yaml                     dataset manifest for training

Run it while standing in the exact pose/scene that produces the false
positive (e.g. face leaning over the checkout counter, close to the lens).
Shoot variety: different angles, distances, lighting. Aim for roughly 5-10%
of your total dataset size in background images — more than that hurts
recall on real products.

Usage (run from sidecar/):
    .venv/Scripts/python.exe tools/capture_hard_negatives.py --source 0
    .venv/Scripts/python.exe tools/capture_hard_negatives.py --source 0 --interval 1.0 --max-frames 300
    .venv/Scripts/python.exe tools/capture_hard_negatives.py --source footage.mp4

Then merge the folder into your training data (append the images/labels dirs
or copy the files into your existing dataset split) and retrain.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.camera import _default_capture  # noqa: E402

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

    Returns the number of frames saved. Raises RuntimeError if the source
    fails to open or dies repeatedly mid-capture.
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
    (out / "data.yaml").write_text(
        "# Background/hard-negative images only: every label file is empty.\n"
        "# Ultralytics treats empty-label images as pure background during\n"
        "# training, which suppresses false positives on these scenes.\n"
        "# Merge into your main dataset (or train alongside it) - do NOT train\n"
        "# on this folder alone: it defines no classes, so it is not a model.\n"
        f"path: {out.resolve()}\n"
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
        "--source", default="0",
        help="Camera index (e.g. 0) or path to a video file. Default: 0",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output dataset dir. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Seconds between saved frames. Default: 2.0",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200,
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
    print("Next: pose in the scene that triggers the false positive while this runs,")
    print("then merge these images/labels into your training dataset and retrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
