# SCANnCART Dataset Pipeline

Build custom YOLO11 grocery detection models from product photos.

## Quick Start

```bash
# 1. Add raw photos to scanncart/raw/ (one .zip per product)

# 2. Clean + augment
python clean_dataset.py

# 3. Upload to Roboflow (get API key from app.roboflow.com/settings/api)
python upload_to_roboflow.py --api-key YOUR_KEY --project YOUR_PROJECT

# 4. Label in Roboflow web UI (use Label Assist for auto-labeling)

# 5. Export from Roboflow -> extract to scanncart/dataset/

# 6. Train
python train_yolo11.py --data dataset/data.yaml

# 7. Deploy to sidecar
python deploy_model.py
```

## Adding a New Product Class

1. Collect 250+ photos -> zip -> place in `scanncart/raw/`
2. Add class prefix to `clean_dataset.py`:
   ```python
   CLASS_PREFIXES = [
       # ... existing ...
       ("Coca-Cola", "coca-cola-330ml"),  # ADD THIS
   ]
   ```
3. Re-run: `python clean_dataset.py`
4. Upload, label, export, train, deploy (see above)

## Scripts

| Script | Purpose |
|--------|---------|
| `clean_dataset.py` | Clean + augment raw images -> `cleaned/` |
| `upload_to_roboflow.py` | Batch upload to Roboflow |
| `train_yolo11.py` | Train YOLO11 locally |
| `deploy_model.py` | Copy model to sidecar |

## Directory Layout

```
scanncart/
├── raw/              # Your raw .zip files
├── cleaned/          # Cleaned + augmented images
├── dataset/          # Labeled dataset (after Roboflow export)
├── yolo11-grocery/   # Training results
│   └── weights/best.pt
└── DATASET_GUIDE.md  # Full documentation
```

## Full Guide

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for complete documentation.
