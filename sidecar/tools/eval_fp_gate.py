"""False-positive regression gate: block a model from shipping if it detects
anything on known-background frames.

Companion to capture_hard_negatives.py. After a retrain, run the candidate
weights over a folder of known-FP frames (e.g. your face at the checkout
counter, saved by the negatives tool) BEFORE deploying them. Any detection
matching the gate is a regression: exit code 1, so it can double as a CI
step.

Usage (run from sidecar/):
    .venv/Scripts/python.exe tools/eval_fp_gate.py \
        --model data/custom/scanncart-grocery-1.pt \
        --frames data/datasets/hard_negatives/images

    # Only fail on a specific confused class:
    .venv/Scripts/python.exe tools/eval_fp_gate.py --model ... --frames ... \
        --classes safeguard

    # Sweep below the operating threshold to catch weak FPs early:
    .venv/Scripts/python.exe tools/eval_fp_gate.py --model ... --frames ... --conf 0.1

Exit codes: 0 = pass, 1 = violations found (gate blocked), 2 = setup error.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import YoloDetector  # noqa: E402
from app.schemas import Detection  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Violation:
    image: str
    cls: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass
class GateReport:
    frames_checked: int = 0
    frames_skipped: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def max_conf(self) -> float:
        return max((v.conf for v in self.violations), default=0.0)


def _default_detector_factory(model_path, device, conf, imgsz):
    return YoloDetector(model_path, device=device, conf=conf, imgsz=imgsz)


def _list_images(frames_dir: Path) -> list[Path]:
    return sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run(
    model_path: str,
    frames_dir: str,
    device: str = "cpu",
    conf: float = 0.25,
    imgsz: int = 960,
    classes: list[str] | None = None,
    detector_factory=None,
) -> GateReport:
    """Run the model over every image in frames_dir and report violations.

    A violation is a detection whose class is in `classes` (or ANY detection
    when `classes` is None — known-FP frames should be pure background).
    `conf` is the scan threshold; set it BELOW the model's operating
    threshold so weak regressions are caught before users ever see them.
    """
    if detector_factory is None:
        detector_factory = _default_detector_factory

    frames = Path(frames_dir)
    if not frames.is_dir():
        raise NotADirectoryError(f"frames dir not found: {frames_dir}")
    images = _list_images(frames)
    if not images:
        raise RuntimeError(
            f"no images found in {frames_dir} — a gate over zero frames proves nothing"
        )

    detector = detector_factory(model_path, device, conf, imgsz)
    report = GateReport()
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            report.frames_skipped += 1
            continue
        report.frames_checked += 1
        for d in detector.infer(frame):  # type: ignore[attr-defined]
            if classes is not None and d.cls not in classes:
                continue
            report.violations.append(
                Violation(image=image.name, cls=d.cls, conf=d.conf, box=d.box)
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate a model on known false-positive frames: fail if it "
        "detects anything it shouldn't."
    )
    parser.add_argument("--model", required=True, help="Path to the .pt weights to check")
    parser.add_argument("--frames", required=True,
                        help="Directory of known-background frames (see "
                        "tools/capture_hard_negatives.py)")
    parser.add_argument("--device", default="cpu", help="cpu/cuda. Default: cpu")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Scan threshold — keep BELOW the operating conf so weak "
                        "false positives are caught early. Default: 0.25")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Inference size; match production settings. Default: 960")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class names to fail on. Default: any "
                        "detection at all (frames must be pure background)")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None

    try:
        report = run(
            args.model, args.frames, device=args.device, conf=args.conf,
            imgsz=args.imgsz, classes=classes,
        )
    except (NotADirectoryError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"frames checked: {report.frames_checked}"
          + (f" (skipped {report.frames_skipped} unreadable)" if report.frames_skipped else ""))
    if report.passed:
        print(f"PASS — no detections across {report.frames_checked} known-background frames")
        return 0

    print(f"FAIL — {len(report.violations)} detection(s) on known-background frames "
          f"(max conf {report.max_conf:.3f}):")
    for v in sorted(report.violations, key=lambda x: -x.conf):
        print(f"  {v.image}: {v.cls} @ {v.conf:.3f} box={tuple(round(b, 3) for b in v.box)}")
    print("\nBlock deployment: retrain with more hard negatives or raise conf_threshold.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
