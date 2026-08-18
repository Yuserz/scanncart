"""
Train a custom YOLO11 model on labeled SCANnCART data.

Usage:
  1. Label your images in Roboflow and export as YOLO format
  2. Extract the exported ZIP to scanncart/dataset/
  3. Run: python train_yolo11.py --data scanncart/dataset/data.yaml

Or use Roboflow's hosted training (no GPU needed):
  1. In Roboflow, go to Versions > Create New Version
  2. Click "Train Model" and select YOLO11
"""
import argparse
from pathlib import Path


def train_local(data_yaml, model, epochs, imgsz, batch, device):
    """Train locally using Ultralytics CLI."""
    from ultralytics import YOLO

    print(f"\nTraining YOLO11 with:")
    print(f"  Data: {data_yaml}")
    print(f"  Base model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print()

    # Load a pretrained YOLO11 model
    yolo = YOLO(model)

    # Train the model
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="scanncart",
        name="yolo11-grocery",
        exist_ok=True,
        patience=20,          # early stopping patience
        save=True,
        save_period=10,       # save checkpoint every 10 epochs
        verbose=True,
        seed=42,
        # Augmentation (beyond what's in your dataset)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Optimization
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
    )

    print(f"\nTraining complete!")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"\nTo validate: yolo detect val data={data_yaml} model={results.save_dir}/weights/best.pt")
    print(f"To predict: yolo detect predict model={results.save_dir}/weights/best.pt source=YOUR_IMAGE")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on SCANnCART data")
    parser.add_argument("--data", required=True,
                        help="Path to data.yaml (YOLO format)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base model (default: yolo11n.pt for speed, yolo11s.pt or yolo11m.pt for accuracy)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, reduce if GPU OOM)")
    parser.add_argument("--device", default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU (default: 0)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Make sure you've exported your Roboflow dataset and extracted it.")
        print("Expected structure:")
        print("  scanncart/dataset/")
        print("    train/images/  train/labels/")
        print("    valid/images/  valid/labels/")
        print("    data.yaml")
        return

    train_local(
        data_yaml=data_path,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
