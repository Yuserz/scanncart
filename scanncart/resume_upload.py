"""Resume-upload cleaned SCANnCART images to Roboflow.

Ground truth for what's already uploaded comes from the project's batch
listing (authoritative). A local state file (.freebuff/upload-state.json)
records individual image uploads as they succeed so re-runs skip them, and
Roboflow's server-side dedup (duplicate:true) makes re-attempts safe.

Usage:
    python resume_upload.py [--cls CLASS] [--dry-run]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
CLEANED_DIR = SCRIPT_DIR / "cleaned"
STATE_DIR = SCRIPT_DIR.parent / ".freebuff"
STATE_FILE = STATE_DIR / "upload-state.json"

TARGETS = {
    "555-sardines": 300,
    "bear-brand-milk": 324,
    "century-tuna": 300,
    "lucky-me-pancit": 300,
    "safeguard": 300,
    "silver-swan-vinegar": 300,
}

load_dotenv(SCRIPT_DIR / ".env")


def normalize_batch_name(name: str) -> str:
    """Map a Roboflow batch name back to a class key."""
    n = name.strip()
    if n.startswith("scanncart-"):
        n = n[len("scanncart-"):]
    n = n.split(" - Auto Label")[0].strip()
    return n


def get_batch_counts(project) -> dict:
    batches = project.get_batches().get("batches", [])
    counts = {}
    for b in batches:
        cls = normalize_batch_name(b.get("name", ""))
        if cls in TARGETS:
            counts[cls] = counts.get(cls, 0) + b.get("images", 0)
    return counts


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def upload_image(api_key, project_id, img_path, cls, session) -> tuple:
    """Return (ok, duplicate) — ok False means it failed and must be retried."""
    url = f"https://api.roboflow.com/dataset/{project_id}/upload"
    params = {
        "api_key": api_key,
        "name": img_path.name,
        "split": "train",
        "batch": f"scanncart-{cls}",
    }
    last_err = None
    for attempt in range(4):
        try:
            with open(img_path, "rb") as f:
                resp = session.post(
                    url, params=params, files={"file": (img_path.name, f, "image/jpeg")}, timeout=180
                )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") or data.get("duplicate"):
                    return True, bool(data.get("duplicate"))
                last_err = f"unexpected payload: {resp.text[:200]}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    return False, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", "-c", help="Upload specific class only")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be uploaded")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    project_id = os.getenv("ROBOFLOW_PROJECT_ID")
    if not api_key or not project_id:
        print("Error: Check .env for ROBOFLOW_API_KEY and ROBOFLOW_PROJECT_ID")
        return 1

    from roboflow import Roboflow

    print("Connecting to Roboflow...")
    project = Roboflow(api_key=api_key).workspace().project(project_id)
    counts = get_batch_counts(project)

    classes = {c: TARGETS[c] for c in TARGETS}
    if args.cls:
        if args.cls not in TARGETS:
            print(f"Unknown class. Use: {', '.join(TARGETS)}")
            return 1
        classes = {args.cls: TARGETS[args.cls]}

    print("\n=== Current Status ===")
    for cls, target in TARGETS.items():
        u = counts.get(cls, 0)
        print(f"  {cls}: {u}/{target} ({'OK' if u >= target else f'MISSING {target-u}'})")

    state = load_state()
    session = requests.Session()

    print("\n=== Uploading ===")
    grand_ok = 0
    grand_dup = 0
    grand_fail = 0
    for cls, target in classes.items():
        if counts.get(cls, 0) >= target:
            print(f"  {cls}: already complete, skipping")
            continue
        cls_dir = CLEANED_DIR / cls
        if not cls_dir.exists():
            print(f"  {cls}: folder missing, skipping")
            continue
        images = sorted(cls_dir.glob("*.jpg"))
        todo = [img for img in images if img.name not in state.get(cls, {})]
        print(f"  {cls}: {len(todo)} to upload")
        if args.dry_run:
            continue
        ok = dup = fail = 0
        for i, img in enumerate(todo, 1):
            uploaded, is_dup = upload_image(api_key, project_id, img, cls, session)
            if uploaded:
                state.setdefault(cls, {})[img.name] = is_dup
                if is_dup:
                    dup += 1
                else:
                    ok += 1
            else:
                fail += 1
                print(f"    FAILED: {img.name}")
            if i % 25 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)} (new={ok}, dup={dup}, fail={fail})")
                save_state(state)
        grand_ok += ok
        grand_dup += dup
        grand_fail += fail

    print(f"\n=== Done ===")
    print(f"  new: {grand_ok}, duplicate: {grand_dup}, failed: {grand_fail}")
    print(f"  state saved to {STATE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())