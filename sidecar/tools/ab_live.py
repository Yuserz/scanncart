"""Compare two installed weights on the same live camera frames.

The test splits are phone photos (4:3, ~4000x3000); the app runs on the StreamCam (16:9, 1280x720).
A model's score on the first says little about which weight is better on the second, and that is the
question a stretch-versus-letterbox choice turns on. So this records the camera once and runs both
weights over the *identical* frames, each with its own recorded geometry and size, through the app's
own detector and acceptance rules - the only comparison where the frames cannot be the difference.

    record   a window on the screen walks through each product and an empty basket, saving frames
             from the camera the app uses (`app.camera.CameraCapture`, auto exposure on, as the app
             opens it). Stop the app's capture first: one process can hold the camera.
    compare  runs each weight over the recording and prints, per segment, how often the expected
             product was shown to the operator, how often something else was, and the confidence.

There are no drawn boxes here, so "found" means the product the segment was recorded for appeared
among the detections the app would keep - which is what a cart cares about - and "wrong" means any
other class did. The empty segment has no right answer: every detection on it is a phantom.

    .venv/Scripts/python.exe tools/ab_live.py record --out data/ab-live
    .venv/Scripts/python.exe tools/ab_live.py compare --frames data/ab-live --a models/scanncart-grocery-v2-stretch.pt --b models/scanncart-grocery-v2-letterbox.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIDECAR = HERE.parent
if str(SIDECAR) not in sys.path:
    sys.path.insert(0, str(SIDECAR))

EMPTY = "empty"
# Segment names are the record's own class names, so a segment can be matched to a detection
# without a mapping table; `empty` is the basket with nothing in it.
DEFAULT_SECONDS = 12.0
SAVE_EVERY_S = 0.1


def roster() -> list[str]:
    from app.roster import V2_ROSTER

    return list(V2_ROSTER)


def record(out: Path, seconds: float, width: int, height: int, fps: int, index: int) -> int:
    import cv2

    from app.camera import CameraCapture

    out.mkdir(parents=True, exist_ok=True)
    camera = CameraCapture(index, width, height, fps, auto_exposure=True)
    if not camera.open():
        print(f"camera {index} did not open - is the app's capture still running?")
        return 2
    segments = [EMPTY, *roster()]
    window = "ab_live - record"
    try:
        # Let auto exposure settle before anything is kept.
        settle_until = time.monotonic() + 3.0
        while time.monotonic() < settle_until:
            got = camera.latest()
            if got is not None:
                cv2.imshow(window, _banner(got[1], "Getting the camera ready...", ""))
            cv2.waitKey(30)
        for segment in segments:
            target = out / segment
            target.mkdir(exist_ok=True)
            prompt = "Clear the basket - nothing in view" if segment == EMPTY else f"Show: {segment}"
            hint = "" if segment == EMPTY else "move it close, then mid, then far, turning it slowly"
            # A short get-ready pause per segment, so the first frames are not a hand mid-swap.
            ready_until = time.monotonic() + 4.0
            while time.monotonic() < ready_until:
                got = camera.latest()
                if got is not None:
                    left = ready_until - time.monotonic()
                    cv2.imshow(window, _banner(got[1], f"NEXT - {prompt}", f"starting in {left:.0f}s"))
                cv2.waitKey(30)
            end = time.monotonic() + seconds
            last_save = 0.0
            saved = 0
            while time.monotonic() < end:
                got = camera.latest()
                now = time.monotonic()
                if got is not None:
                    _, frame = got
                    if now - last_save >= SAVE_EVERY_S:
                        cv2.imwrite(str(target / f"{saved:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        saved += 1
                        last_save = now
                    cv2.imshow(window, _banner(frame, f"RECORDING - {prompt}", f"{end - now:.0f}s  {hint}"))
                if cv2.waitKey(1) & 0xFF == 27:
                    print("stopped with Esc")
                    return 1
            print(f"  {segment}: {saved} frames")
        (out / "meta.json").write_text(
            json.dumps({"width": width, "height": height, "fps": fps, "segments": segments}, indent=1),
            encoding="utf-8",
        )
        print(f"recorded into {out}")
        return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def _banner(frame, title: str, sub: str):
    import cv2

    shown = frame.copy()
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(shown, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(shown, sub, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
    return shown


def weight_setup(path: str) -> tuple[str, int]:
    """The geometry and size the weight was trained at, from the record beside it."""
    from app.models import read_record

    record = read_record(Path(path))
    return record.get("resize_mode") or "letterbox", int(record.get("imgsz") or 640)


def compare(frames: Path, weights: list[str], conf: float, device: str) -> int:
    import cv2

    from app.acceptance import accept_detections
    from app.inference import YoloDetector
    from app.settings import Settings

    defaults = Settings()
    segments = [d.name for d in sorted(frames.iterdir()) if d.is_dir()]
    if not segments:
        print(f"no recorded segments under {frames}")
        return 2
    results: dict[str, dict[str, dict]] = {}
    for weight in weights:
        mode, imgsz = weight_setup(weight)
        print(f"{weight}: resize_mode {mode}, imgsz {imgsz}, conf {conf}")
        per: dict[str, dict] = {}
        for segment in segments:
            # A fresh detector per segment: the tracker's memory of one product must not carry
            # into the next segment's first frames.
            detector = YoloDetector(weight, device=device, conf=conf, imgsz=imgsz, resize_mode=mode)
            files = sorted((frames / segment).glob("*.jpg"))
            found = wrong = any_shown = 0
            confs: list[float] = []
            started = time.perf_counter()
            for f in files:
                frame = cv2.imread(str(f))
                decision = accept_detections(
                    detector.infer(frame),
                    class_allowlist=defaults.class_allowlist,
                    suppress_clamped=defaults.suppress_clamped_detections,
                    suppress_frame_filling=defaults.suppress_frame_filling_detections,
                    suppress_unsure=defaults.suppress_unsure_phantoms,
                )
                shown = decision.accepted
                any_shown += bool(shown)
                hits = [d.conf for d in shown if d.cls == segment]
                if hits:
                    found += 1
                    confs.append(max(hits))
                if any(d.cls != segment for d in shown):
                    wrong += 1
            elapsed = time.perf_counter() - started
            per[segment] = {
                "frames": len(files),
                "found": found,
                "wrong": wrong,
                "any": any_shown,
                "mean_conf": round(sum(confs) / len(confs), 3) if confs else None,
                "ms_per_frame": round(1000 * elapsed / max(1, len(files)), 1),
            }
            detector.close()
        results[weight] = per

    names = [Path(w).stem for w in weights]
    print()
    print("per segment: found = frames where the recorded product was shown; wrong = frames where")
    print("another class was shown; on `empty` every shown detection is a phantom")
    header = f"{'segment':42s}" + "".join(f"{n[-24:]:>34s}" for n in names)
    print(header)
    for segment in segments:
        cells = []
        for weight in weights:
            r = results[weight][segment]
            if segment == EMPTY:
                cells.append(f"phantom {r['any']}/{r['frames']}")
            else:
                c = f" c{r['mean_conf']:.2f}" if r["mean_conf"] is not None else ""
                cells.append(f"found {r['found']}/{r['frames']} wrong {r['wrong']}{c}")
        print(f"{segment[:42]:42s}" + "".join(f"{c:>34s}" for c in cells))
    print()
    for weight, name in zip(weights, names):
        per = results[weight]
        product = [s for s in segments if s != EMPTY]
        frames_total = sum(per[s]["frames"] for s in product)
        found = sum(per[s]["found"] for s in product)
        wrong = sum(per[s]["wrong"] for s in product)
        ms = sum(per[s]["ms_per_frame"] * per[s]["frames"] for s in segments) / max(
            1, sum(per[s]["frames"] for s in segments)
        )
        phantom = per.get(EMPTY, {}).get("any")
        print(
            f"{name}: found {found}/{frames_total} ({found / max(1, frames_total):.1%}), "
            f"wrong {wrong}, empty-basket phantoms {phantom}, {ms:.1f} ms/frame"
        )
    out = frames / "ab_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    rec = sub.add_parser("record", help="record a guided clip per product from the camera")
    rec.add_argument("--out", default="data/ab-live")
    rec.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    rec.add_argument("--width", type=int, default=1280)
    rec.add_argument("--height", type=int, default=720)
    rec.add_argument("--fps", type=int, default=60)
    rec.add_argument("--camera", type=int, default=0)
    cmp_ = sub.add_parser("compare", help="run two weights over the same recording")
    cmp_.add_argument("--frames", default="data/ab-live")
    cmp_.add_argument("--a", required=True)
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--conf", type=float, default=0.8)
    cmp_.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.verb == "record":
        return record(Path(args.out), args.seconds, args.width, args.height, args.fps, args.camera)
    return compare(Path(args.frames), [args.a, args.b], args.conf, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
