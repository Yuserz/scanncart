"""
Deploy trained SCANnCART model to the sidecar.

Usage:
  python deploy_model.py                          # Deploy to default slot (grocery-v1)
  python deploy_model.py --name grocery-v2        # Deploy to named slot
  python deploy_model.py --source path/to/best.pt # Deploy from custom path

Models are deployed to sidecar/data/custom/<name>.pt
"""
import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "scanncart" / "yolo11-grocery" / "weights" / "best.pt"
CUSTOM_DIR = REPO_ROOT / "sidecar" / "data" / "custom"


def main():
    parser = argparse.ArgumentParser(description="Deploy trained model to sidecar")
    parser.add_argument("--source", type=Path, default=None,
                        help="Path to best.pt (default: scanncart/yolo11-grocery/weights/best.pt)")
    parser.add_argument("--name", default="grocery-v1",
                        help="Model slot name (default: grocery-v1)")
    args = parser.parse_args()

    source = args.source or DEFAULT_SOURCE
    target = CUSTOM_DIR / f"{args.name}.pt"

    if not source.exists():
        print(f"Error: Model not found at {source}")
        print("\nTrain first:")
        print("  cd scanncart && python train_yolo11.py --data dataset/data.yaml")
        print("\nOr specify a custom path:")
        print("  python scanncart/deploy_model.py --source path/to/best.pt")
        sys.exit(1)

    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

    size_mb = source.stat().st_size / (1024 * 1024)
    print(f"Deploying model: {args.name}")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Size: {size_mb:.1f} MB")

    shutil.copy2(source, target)

    print(f"\nDone! Model deployed to {target}")
    print(f"\nTo use it:")
    print(f"  1. Start the SCANnCART desktop app")
    print(f"  2. Open Admin Panel")
    print(f"  3. Select 'data/custom/{args.name}.pt' from the Model dropdown")
    print(f"  4. Start capture")

    # List all deployed custom models
    print(f"\nAll custom models in {CUSTOM_DIR}:")
    for f in sorted(CUSTOM_DIR.glob("*.pt")):
        size = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
