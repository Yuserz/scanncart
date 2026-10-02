"""CLI tool to guide and record the 90 Gate A physical trials directly from the camera.

Presents trials in run-sheet order (DEP-01..30, REM-01..30, NOT-01..30), draws timestamp
and trial info watermarks on recorded frames, and writes video clips alongside a session
manifest JSON for audit and playback.

Usage:
    .venv/Scripts/python.exe tools/gate_a_recorder.py --help
    .venv/Scripts/python.exe tools/gate_a_recorder.py --session s1 --actor "Actor A" --light L1
    .venv/Scripts/python.exe tools/gate_a_recorder.py --trial DEP-01 --duration 5.0 --out data/gate_a_footage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.camera import _default_capture  # noqa: E402


@dataclass(frozen=True)
class TrialDefinition:
    trial_id: str
    kind: str  # "deposit" | "removal" | "no_transfer"
    sku: str | None
    actor: str
    light: str
    scenario: str
    instructions: str


def build_trial_roster() -> list[TrialDefinition]:
    """Build the canonical 90-trial sequence specified in docs/GATE_A_RUN_SHEET.md."""
    trials: list[TrialDefinition] = []

    # 1. 30 Deposits (DEP-01 to DEP-30)
    dep_skus = [
        "Bear Brand Sachet",
        "Lucky Me Pancit",
        "555 Sardines Can",
        "Century Tuna Can",
        "Silver Swan Bottle",
        "Milo Sachet",
        "Safeguard Bar Soap",
    ]

    dep_meta = [
        ("A", "L1", "Left / Front", dep_skus[0]),
        ("A", "L1", "Right / Side", dep_skus[1]),
        ("A", "L1", "Right / Front", dep_skus[2]),
        ("A", "L1", "Left / Side", dep_skus[3]),
        ("A", "L1", "Right / Front", dep_skus[4]),
        ("A", "L1", "Left / Front", dep_skus[5]),
        ("A", "L1", "Right / Side", dep_skus[6]),
        ("A", "L2", "Right / Front", dep_skus[0]),
        ("A", "L2", "Left / Side", dep_skus[1]),
        ("A", "L2", "Right / Side", dep_skus[2]),
        ("A", "L2", "Left / Front", dep_skus[3]),
        ("A", "L2", "Right / Front", dep_skus[4]),
        ("A", "L3", "Left / Side", dep_skus[5]),
        ("A", "L3", "Right / Front", dep_skus[6]),
        ("A", "L3", "Right / Side", dep_skus[3]),
        ("B", "L1", "Right / Front", dep_skus[0]),
        ("B", "L1", "Left / Front", dep_skus[1]),
        ("B", "L1", "Right / Side", dep_skus[2]),
        ("B", "L1", "Left / Side", dep_skus[3]),
        ("B", "L1", "Right / Front", dep_skus[4]),
        ("B", "L1", "Left / Front", dep_skus[5]),
        ("B", "L1", "Right / Side", dep_skus[6]),
        ("B", "L2", "Left / Front", dep_skus[0]),
        ("B", "L2", "Right / Side", dep_skus[1]),
        ("B", "L2", "Left / Side", dep_skus[2]),
        ("B", "L3", "Right / Front", dep_skus[3]),
        ("B", "L3", "Left / Front", dep_skus[4]),
        ("B", "L3", "Right / Side", dep_skus[5]),
        ("B", "L3", "Left / Side", dep_skus[6]),
        ("B", "L3", "Right / Front", "Bear Brand (Over Stack)"),
    ]

    for idx, (actor, light, approach, sku) in enumerate(dep_meta, start=1):
        tid = f"DEP-{idx:02d}"
        trials.append(
            TrialDefinition(
                trial_id=tid,
                kind="deposit",
                sku=sku,
                actor=actor,
                light=light,
                scenario=f"Deposit: {sku} ({approach})",
                instructions="Present 1s outside, move into opening, release on base, withdraw hand, hold 1s.",
            )
        )

    # 2. 30 Removals (REM-01 to REM-30)
    rem_meta = [
        ("A", "L1", "Right / Top", dep_skus[0]),
        ("A", "L1", "Left / Front", dep_skus[1]),
        ("A", "L1", "Right / Side", dep_skus[2]),
        ("A", "L1", "Left / Top", dep_skus[3]),
        ("A", "L1", "Right / Neck", dep_skus[4]),
        ("A", "L1", "Left / Side", dep_skus[5]),
        ("A", "L1", "Right / Top", dep_skus[6]),
        ("A", "L2", "Left / Side", dep_skus[0]),
        ("A", "L2", "Right / Front", dep_skus[1]),
        ("A", "L2", "Left / Top", dep_skus[2]),
        ("A", "L2", "Right / Side", dep_skus[3]),
        ("A", "L2", "Left / Neck", dep_skus[4]),
        ("A", "L3", "Right / Top", dep_skus[5]),
        ("A", "L3", "Left / Side", dep_skus[6]),
        ("A", "L3", "Right / Front", dep_skus[1]),
        ("B", "L1", "Right / Side", dep_skus[0]),
        ("B", "L1", "Left / Top", dep_skus[1]),
        ("B", "L1", "Right / Front", dep_skus[2]),
        ("B", "L1", "Left / Side", dep_skus[3]),
        ("B", "L1", "Right / Neck", dep_skus[4]),
        ("B", "L1", "Left / Front", dep_skus[5]),
        ("B", "L1", "Right / Top", dep_skus[6]),
        ("B", "L2", "Left / Side", dep_skus[2]),
        ("B", "L2", "Right / Top", dep_skus[3]),
        ("B", "L2", "Left / Front", dep_skus[4]),
        ("B", "L3", "Right / Side", dep_skus[5]),
        ("B", "L3", "Left / Top", dep_skus[6]),
        ("B", "L3", "Right / Front", dep_skus[0]),
        ("B", "L3", "Left / Side", dep_skus[1]),
        ("B", "L3", "Right / Bottom", "Century Tuna (Under Stack)"),
    ]

    for idx, (actor, light, approach, sku) in enumerate(rem_meta, start=1):
        tid = f"REM-{idx:02d}"
        trials.append(
            TrialDefinition(
                trial_id=tid,
                kind="removal",
                sku=sku,
                actor=actor,
                light=light,
                scenario=f"Removal: {sku} ({approach})",
                instructions="Grasp item visible inside, lift through opening, clear rim into outside zone, hold 1s.",
            )
        )

    # 3. 30 No-Transfers (NOT-01 to NOT-30)
    not_meta = [
        ("A", "L1", "N1: Outside Presentation", "Present Bear Brand in counter zone, hold 2s, withdraw"),
        ("A", "L1", "N1: Outside Presentation", "Present Milo sachet near rim, tilt label, withdraw"),
        ("B", "L2", "N1: Outside Presentation", "Present 555 Sardines with two hands, rotate, lower away"),
        ("B", "L3", "N1: Outside Presentation", "Present Century Tuna, tap counter edge, withdraw"),
        ("A", "L3", "N1: Outside Presentation", "Present Safeguard bar box 5 cm above rim, pull back"),
        ("A", "L1", "N2: Hover & Retract", "Lower Lucky Me into rim plane, hover 3s, lift back out"),
        ("A", "L2", "N2: Hover & Retract", "Lower Silver Swan into opening, hover 2s, retract to side"),
        ("B", "L1", "N2: Hover & Retract", "Lower Century Tuna into basket volume, hesitate, withdraw"),
        ("B", "L2", "N2: Hover & Retract", "Lower Milo, wave gently inside opening, withdraw"),
        ("B", "L3", "N2: Hover & Retract", "Lower Safeguard box, touch floor without letting go, lift"),
        ("A", "L1", "N3: Reach-Only", "Reach open right hand into empty basket, withdraw"),
        ("A", "L2", "N3: Reach-Only", "Reach left hand into basket with 2 items, touch floor, out"),
        ("B", "L1", "N3: Reach-Only", "Reach both hands into basket, cup hands together, out"),
        ("B", "L2", "N3: Reach-Only", "Reach hand in, point at items, pull back out"),
        ("A", "L3", "N3: Reach-Only", "Reach hand in rapidly, pause 2s at bottom, withdraw"),
        ("A", "L1", "N4: Rearrangement", "Shift Lucky Me from left basket corner to right corner"),
        ("A", "L2", "N4: Rearrangement", "Lift 555 Sardines 5 cm inside basket, place back down"),
        ("B", "L1", "N4: Rearrangement", "Stack Milo sachet on top of Safeguard box inside basket"),
        ("B", "L2", "N4: Rearrangement", "Slide Century Tuna along bottom floor 10 cm"),
        ("B", "L3", "N4: Rearrangement", "Stand Silver Swan bottle upright from side orientation"),
        ("A", "L1", "N5: Occluded Crossing Abort", "Lower Milo behind wrist/sleeve, break rim, pull back"),
        ("A", "L2", "N5: Occluded Crossing Abort", "Lower Tuna while forearm blocks camera view, retract"),
        ("B", "L1", "N5: Occluded Crossing Abort", "Lower Bear Brand behind tilted palm, reverse out"),
        ("B", "L2", "N5: Occluded Crossing Abort", "Lower Safeguard under sleeve cuff, abort at opening"),
        ("A", "L3", "N5: Occluded Crossing Abort", "Lower 555 Sardines while body leans over rim, reverse out"),
        ("A", "L1", "N6: Occlusion of Rest Item", "Cover deposited Milo with flat palm for 4s, withdraw"),
        ("A", "L2", "N6: Occlusion of Rest Item", "Cover Safeguard box with paper sheet for 5s, remove sheet"),
        ("B", "L1", "N6: Occlusion of Rest Item", "Place hand over Lucky Me pouch for 3s, withdraw hand"),
        ("B", "L2", "N6: Occlusion of Rest Item", "Drop cloth over items in basket, wait 5s, lift cloth"),
        ("B", "L3", "N6: Occlusion of Rest Item", "Rest forearm across basket opening for 4s, withdraw"),
    ]

    for idx, (actor, light, scenario, instructions) in enumerate(not_meta, start=1):
        tid = f"NOT-{idx:02d}"
        trials.append(
            TrialDefinition(
                trial_id=tid,
                kind="no_transfer",
                sku=None,
                actor=actor,
                light=light,
                scenario=scenario,
                instructions=instructions,
            )
        )

    return trials


def stamp_frame(
    frame: np.ndarray,
    trial_id: str,
    timestamp_iso: str,
    frame_idx: int,
    fps: float,
    elapsed_s: float,
) -> np.ndarray:
    """Draw a readable watermark banner on the frame for human review verification."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent top banner (height 36px)
    banner_h = min(40, int(h * 0.08))
    cv2.rectangle(annotated, (0, 0), (w, banner_h), (20, 20, 20), -1)

    text_left = f"{trial_id}  |  {timestamp_iso}"
    text_right = f"F:{frame_idx:04d}  {elapsed_s:05.2f}s  {fps:4.1f}fps"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55 if w >= 1280 else 0.45
    color = (240, 240, 240)
    thickness = 1

    cv2.putText(annotated, text_left, (10, banner_h - 12), font, scale, color, thickness, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(text_right, font, scale, thickness)
    cv2.putText(annotated, text_right, (w - rw - 10, banner_h - 12), font, scale, color, thickness, cv2.LINE_AA)

    return annotated


def record_clip(
    cap: Any,
    out_path: Path,
    trial: TrialDefinition,
    duration_s: float,
    target_fps: float = 30.0,
    time_func: Callable[[], float] = time.time,
    sleep_func: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Record a single video clip for the given trial definition."""
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        raise RuntimeError("Failed to read frame from camera")

    h, w = first_frame.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, target_fps, (w, h))

    start_time = time_func()
    frame_count = 0
    t0_iso = datetime.now(timezone.utc).isoformat()

    try:
        # Stamp and write first frame
        stamped = stamp_frame(first_frame, trial.trial_id, t0_iso, frame_count, target_fps, 0.0)
        writer.write(stamped)
        frame_count += 1

        interval = 1.0 / target_fps
        next_deadline = start_time + interval

        while True:
            now = time_func()
            elapsed = now - start_time
            if elapsed >= duration_s:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            stamp_iso = datetime.now(timezone.utc).isoformat()
            stamped = stamp_frame(frame, trial.trial_id, stamp_iso, frame_count, target_fps, elapsed)
            writer.write(stamped)
            frame_count += 1

            # Sleep to match target FPS
            now_after_write = time_func()
            delay = next_deadline - now_after_write
            if delay > 0:
                sleep_func(delay)
            next_deadline += interval

    finally:
        writer.release()

    total_time = max(0.001, time_func() - start_time)
    actual_fps = frame_count / total_time

    return {
        "trial_id": trial.trial_id,
        "kind": trial.kind,
        "sku": trial.sku,
        "actor": trial.actor,
        "light": trial.light,
        "scenario": trial.scenario,
        "file_name": out_path.name,
        "file_path": str(out_path.as_posix()),
        "frame_count": frame_count,
        "duration_s": round(total_time, 3),
        "fps": round(actual_fps, 2),
        "resolution": [w, h],
        "recorded_at": t0_iso,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate A recording tool for physical add/remove trials.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="OpenCV camera index (e.g. 0) or video file path.",
    )
    parser.add_argument(
        "--out",
        default="data/gate_a_footage",
        help="Output directory for recorded clips and manifest (default: data/gate_a_footage).",
    )
    parser.add_argument(
        "--session",
        default="s1",
        help="Recording session ID (e.g. s1, s2).",
    )
    parser.add_argument(
        "--trial",
        help="Optional single trial ID to record (e.g. DEP-01, REM-15, NOT-04). If omitted, runs interactive playlist.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Target duration per clip in seconds (default: 6.0).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target recording FPS (default: 30.0).",
    )
    parser.add_argument(
        "--actor",
        help="Filter trials to a specific actor (e.g. A or B).",
    )
    parser.add_argument(
        "--light",
        help="Filter trials to a specific lighting condition (e.g. L1, L2, L3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned trial list without opening camera or recording video.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, cap: Any = None) -> int:
    trials = build_trial_roster()

    if args.trial:
        trials = [t for t in trials if t.trial_id.upper() == args.trial.upper()]
        if not trials:
            print(f"Error: Unknown trial ID '{args.trial}'. Valid range: DEP-01..30, REM-01..30, NOT-01..30.")
            return 2

    if args.actor:
        trials = [t for t in trials if t.actor.upper() == args.actor.upper()]

    if args.light:
        trials = [t for t in trials if t.light.upper() == args.light.upper()]

    if not trials:
        print("No trials match the specified filters.")
        return 0

    out_dir = Path(args.out) / args.session
    print(f"Gate A Recorder: {len(trials)} trial(s) queued for session '{args.session}' -> {out_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Planned trials:")
        for t in trials:
            sku_label = f" | SKU: {t.sku}" if t.sku else ""
            print(f"  [{t.trial_id}] ({t.kind}) Actor {t.actor}, Light {t.light}{sku_label} -> {t.instructions}")
        return 0

    if cap is None:
        source_val: int | str = int(args.source) if args.source.isdigit() else args.source
        cap = _default_capture(source_val)
        if not cap.isOpened():
            print(f"Error: Failed to open camera source '{args.source}'")
            return 1

    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "session": args.session,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trials": [],
    }
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    existing_trial_ids = {entry.get("trial_id") for entry in manifest.get("trials", [])}

    try:
        for idx, trial in enumerate(trials, start=1):
            clip_path = out_dir / f"{trial.trial_id}.mp4"
            print("\n" + "=" * 60)
            print(f"Trial {idx}/{len(trials)}: [{trial.trial_id}] {trial.scenario}")
            print(f"  Actor: {trial.actor} | Light: {trial.light} | Duration: {args.duration}s")
            print(f"  Instructions: {trial.instructions}")
            print(f"  Destination:  {clip_path.name}")
            print("=" * 60)

            # In interactive CLI, give a brief 2-second countdown before filming
            if not os.environ.get("CI"):
                print("Recording starting in 2 seconds... Prepare item.")
                time.sleep(2.0)

            print("🔴 RECORDING...")
            clip_record = record_clip(
                cap=cap,
                out_path=clip_path,
                trial=trial,
                duration_s=args.duration,
                target_fps=args.fps,
            )
            print(f"✔ Saved: {clip_path.name} ({clip_record['frame_count']} frames, {clip_record['fps']} fps)")

            # Update manifest
            if trial.trial_id in existing_trial_ids:
                manifest["trials"] = [t for t in manifest["trials"] if t.get("trial_id") != trial.trial_id]
            manifest["trials"].append(clip_record)
            existing_trial_ids.add(trial.trial_id)

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

    finally:
        cap.release()

    print(f"\nGate A recording complete. Manifest updated at {manifest_path}")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
