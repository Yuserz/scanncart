"""
Upload cleaned SCANnCART images to Roboflow for labeling.

Usage:
  1. Set ROBOFLOW_API_KEY and ROBOFLOW_PROJECT_ID in .env
  2. Create an Object Detection project on Roboflow
  3. Run: python upload_to_roboflow.py

Credentials are loaded from .env file (see .env.example).
"""
import argparse
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Load .env from script directory
load_dotenv(Path(__file__).resolve().parent / ".env")


CLEANED_DIR = Path(__file__).resolve().parent / "cleaned"

CLASS_NAMES = [
    "555-sardines",
    "bear-brand-milk",
    "century-tuna",
    "lucky-me-pancit",
    "safeguard",
    "silver-swan-vinegar",
]


def upload_single_image(api_key, project_id, img_path, cls_name, batch_name):
    """Upload a single image to Roboflow via REST API."""
    url = f"https://api.roboflow.com/dataset/{project_id}/upload"
    params = {
        "api_key": api_key,
        "name": img_path.name,
        "split": "train",
        "batch": batch_name or f"scanncart-{cls_name}",
    }
    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        resp = requests.post(url, params=params, files=files)
    return resp.status_code == 200


def upload_class_images(api_key, project_id, cls_name, batch_name=None, workers=5):
    """Upload all images from a class folder to Roboflow."""
    cls_dir = CLEANED_DIR / cls_name
    if not cls_dir.exists():
        print(f"  [SKIP] {cls_dir} not found")
        return 0

    images = sorted(cls_dir.glob("*.jpg"))
    if not images:
        print(f"  [SKIP] No images in {cls_dir}")
        return 0

    print(f"\n  Uploading {len(images)} images for class '{cls_name}'...")

    uploaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                upload_single_image,
                api_key, project_id, img_path, cls_name,
                batch_name or f"scanncart-{cls_name}",
            ): img_path
            for img_path in images
        }
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                if future.result():
                    uploaded += 1
                    if uploaded % 50 == 0:
                        print(f"    ... {uploaded}/{len(images)} uploaded")
                else:
                    failed += 1
            except Exception as e:
                failed += 1

    print(f"  Done: {uploaded}/{len(images)} uploaded, {failed} failed for '{cls_name}'")
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Upload SCANnCART images to Roboflow")
    parser.add_argument("--api-key", default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY in .env)")
    parser.add_argument("--project", default=None,
                        help="Roboflow project ID (or set ROBOFLOW_PROJECT_ID in .env)")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Specific classes to upload (default: all)")
    parser.add_argument("--batch", default=None,
                        help="Batch name for grouping uploads")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel upload workers (default: 5)")
    args = parser.parse_args()

    # Load from .env if not provided via CLI
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    project_id = args.project or os.getenv("ROBOFLOW_PROJECT_ID")

    if not api_key or not project_id:
        print("Error: Missing credentials!")
        print("\nOption 1: Set in .env file (recommended):")
        print("  Edit scanncart/.env and add your ROBOFLOW_API_KEY and ROBOFLOW_PROJECT_ID")
        print("\nOption 2: Pass via command line:")
        print("  python upload_to_roboflow.py --api-key YOUR_KEY --project YOUR_PROJECT")
        return

    print(f"Connected to Roboflow")
    print(f"Project: {project_id}")

    classes_to_upload = args.classes or CLASS_NAMES
    print(f"\nUploading {len(classes_to_upload)} classes...")
    print(f"Source: {CLEANED_DIR.resolve()}")

    total = 0
    for cls in classes_to_upload:
        count = upload_class_images(api_key, project_id, cls, args.batch, args.workers)
        total += count

    print(f"\n{'=' * 50}")
    print(f"Total uploaded: {total} images")
    print(f"{'=' * 50}")
    print(f"\nNext steps:")
    print(f"  1. Go to https://app.roboflow.com/{project_id}/annotate")
    print(f"  2. Open images and use Label Assist (magic wand) to auto-label")
    print(f"  3. Review and correct the auto-generated bounding boxes")
    print(f"  4. Generate a dataset version")
    print(f"  5. Export as YOLO format and download")
    print(f"  6. Train locally with: yolo detect train data=YOUR_DATA.yaml model=yolo11n.pt")


if __name__ == "__main__":
    main()
