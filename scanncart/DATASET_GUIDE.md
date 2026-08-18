# SCANnCART Dataset Pipeline Guide

Complete guide for building and maintaining custom YOLO11 grocery detection models.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Adding a New Product Class](#adding-a-new-product-class)
4. [Step 1: Collect Raw Images](#step-1-collect-raw-images)
5. [Step 2: Clean the Dataset](#step-2-clean-the-dataset)
6. [Step 3: Label with Roboflow](#step-3-label-with-roboflow)
7. [Step 4: Train the Model](#step-4-train-the-model)
8. [Step 5: Deploy to Sidecar](#step-5-deploy-to-sidecar)
9. [Adding More Classes Later](#adding-more-classes-later)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The SCANnCART dataset pipeline transforms raw product photos into a trained YOLO11 object detection model:

```
Raw Photos (.jpg/.heic)
    ↓ clean_dataset.py
Cleaned Images (deduped, validated, augmented)
    ↓ Roboflow Label Assist
Labeled Dataset (YOLO format: images + .txt bounding boxes)
    ↓ train_yolo11.py
Trained Model (best.pt)
    ↓ deploy_model.py
Sidecar Integration (custom-grocery.pt)
```

---

## Directory Structure

```
scanncart/
├── raw/                          # YOUR RAW PHOTOS (one .zip per product)
│   ├── 555-sardines.zip
│   ├── bear-brand-milk.zip
│   └── ...
│
├── cleaned/                      # OUTPUT: cleaned + augmented images
│   ├── 555-sardines/            #   300 images per class
│   ├── bear-brand-milk/         #   324 images (raw had more)
│   ├── century-tuna/            #   300 images
│   ├── lucky-me-pancit/         #   300 images
│   ├── safeguard/               #   300 images
│   ├── silver-swan-vinegar/     #   300 images
│   └── data.yaml                #   YOLO class definitions
│
├── dataset/                      # OUTPUT: labeled dataset (after Roboflow export)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── yolo11-grocery/              # OUTPUT: training results
│   └── weights/
│       ├── best.pt              #   ← your trained model
│       └── last.pt
│
├── clean_dataset.py             # Script: clean + augment raw images
├── upload_to_roboflow.py        # Script: batch upload to Roboflow
├── train_yolo11.py              # Script: train YOLO11 locally
├── deploy_model.py              # Script: copy best.pt to sidecar
└── README.md                    # This file
```

---

## Adding a New Product Class

### Quick Start (5 minutes)

```bash
# 1. Add raw photos to scanncart/raw/
#    Create a .zip file named after the product:
#    e.g., "Coca-Cola 330ml.zip"

# 2. Add class mapping to clean_dataset.py
#    Edit the CLASS_PREFIXES list:
CLASS_PREFIXES = [
    # ... existing classes ...
    ("Coca-Cola", "coca-cola-330ml"),  # ADD THIS
]

# 3. Re-run the cleaning script
cd scanncart
python clean_dataset.py

# 4. Upload ALL classes to Roboflow (including new one)
python upload_to_roboflow.py --api-key YOUR_KEY --project YOUR_PROJECT

# 5. Label the new class in Roboflow
#    - New images will appear in the upload batch
#    - Use Label Assist to auto-label
#    - Review and correct

# 6. Export updated dataset from Roboflow
#    - Generate new version with ALL classes
#    - Download YOLO format ZIP

# 7. Extract to scanncart/dataset/

# 8. Re-train the model
python train_yolo11.py --data dataset/data.yaml --epochs 100

# 9. Deploy updated model
python deploy_model.py
```

### Detailed Steps for New Classes

#### A. Prepare Raw Images

1. **Collect 250-350 photos** of the new product from multiple angles
2. **Organize**: One folder per product, named clearly
3. **Zip it**: Create a .zip file with the product name
4. **Place in**: `scanncart/raw/`

**Photo Tips:**
- Vary angles (front, back, side, top, angled)
- Vary lighting (bright, dim, natural, artificial)
- Include cluttered backgrounds (realistic checkout scenarios)
- Capture different packaging states (sealed, opened, partially visible)

#### B. Update the Cleaning Script

Edit `scanncart/clean_dataset.py`:

```python
# Find this section near the top:
CLASS_PREFIXES = [
    ("555 (Sardines in Tomato Sauce)", "555-sardines"),
    ("Bear Brand Fortified Powdered Milk", "bear-brand-milk"),
    ("Century Tuna", "century-tuna"),
    ("Lucky Me Pancit Canton", "lucky-me-pancit"),
    ("Safeguard (Pure White)", "safeguard"),
    ("SILVER SWAN", "silver-swan-vinegar"),
    # ADD YOUR NEW CLASS HERE:
    ("Coca-Cola", "coca-cola-330ml"),  # prefix matches folder name inside zip
]
```

**Important**: The `prefix` must match text inside the zip's folder name (not the zip filename). For example, if your zip contains `Coca-Cola 330ml/IMG_001.jpg`, use `"Coca-Cola"` as the prefix.

#### C. Re-run Cleaning

```bash
cd scanncart
python clean_dataset.py
```

This will:
- Extract all zips (including the new one)
- Convert any HEIC images to JPG
- Remove exact duplicates
- Validate all images
- Augment each class to 300+ images
- Output to `scanncart/cleaned/`

#### D. Upload to Roboflow

```bash
python upload_to_roboflow.py \
  --api-key YOUR_API_KEY \
  --project YOUR_PROJECT_ID
```

This uploads ALL classes in `cleaned/`, including the new one.

#### E. Label in Roboflow

1. Go to your Roboflow project
2. New images appear in upload batches
3. Click images → use **Label Assist** (magic wand) for auto-labeling
4. Draw/adjust bounding boxes around each product
5. **Tip**: Label all visible instances in each image

#### F. Export and Train

1. In Roboflow: **Versions** → **Create New Version**
2. Apply preprocessing (Auto-Orient, Resize to 640×640)
3. **Generate** → **Export Dataset** → **YOLO v8** format
4. Download ZIP → extract to `scanncart/dataset/`
5. Train:

```bash
python train_yolo11.py \
  --data dataset/data.yaml \
  --model yolo11n.pt \
  --epochs 100
```

#### G. Deploy

```bash
python deploy_model.py
```

Restart the SCANnCART desktop app and select the new model.

---

## Step 1: Collect Raw Images

### Requirements per Class

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Images | 100 | 250-350 |
| Angles | 3 | 5-8 |
| Lighting | 2 | 3-4 |
| Backgrounds | 2 | 3-5 |

### Image Quality

- **Resolution**: At least 640×480 (higher is better)
- **Format**: JPG preferred; HEIC will be converted automatically
- **Blur**: Avoid heavily blurred images
- **Occlusion**: Include partially visible products (realistic)

### Naming Convention

```
scanncart/raw/
├── 555-sardines.zip           # Contains "555 (Sardines)/IMG_*.jpg"
├── bear-brand-milk.zip        # Contains "Bear Brand Milk/IMG_*.HEIC"
├── coca-cola-330ml.zip        # Contains "Coca-Cola 330ml/IMG_*.jpg"
└── ...
```

---

## Step 2: Clean the Dataset

The cleaning script handles:

1. **Extraction**: Unzips all archives to per-class folders
2. **HEIC Conversion**: iPhone photos → JPG (requires `pillow-heif`)
3. **Deduplication**: Removes exact byte-identical copies (cross-zip and within-zip)
4. **Validation**: Rejects corrupt, too-small, or too-large images
5. **EXIF Orientation**: Bakes rotation into pixels (OpenCV ignores EXIF)
6. **Augmentation**: Generates synthetic variants to reach 300+ per class

### Augmentations Applied

Each image gets 1-3 random augmentations:

| Augmentation | What It Does |
|--------------|--------------|
| Horizontal flip | Mirror image (safe for most products) |
| Rotation ±5°/±10° | Slight tilt variation |
| Brightness ±25% | Lighting variation |
| Contrast ±25% | Lighting variation |
| Sharpness variation | Focus variation |
| Gaussian blur | Simulates slight defocus |
| JPEG compression | Simulates quality loss |
| Random crop+rezoom | Framing variation |
| Color jitter | Saturation variation |

### Running the Script

```bash
cd scanncart
python clean_dataset.py
```

**Output:**
```
scanncart/cleaned/
├── 555-sardines/     (300 images)
├── bear-brand-milk/  (324 images)
├── century-tuna/     (300 images)
├── lucky-me-pancit/  (300 images)
├── safeguard/        (300 images)
├── silver-swan-vinegar/ (300 images)
└── data.yaml
```

---

## Step 3: Label with Roboflow

### Setup

1. Create account at [app.roboflow.com](https://app.roboflow.com)
2. Create **Object Detection** project
3. Copy API key from Settings → API

### Upload

```bash
python upload_to_roboflow.py \
  --api-key YOUR_API_KEY \
  --project YOUR_PROJECT_ID
```

### Labeling Workflow

1. **Open image** in Roboflow Annotate
2. **Click magic wand** (Label Assist)
3. **Choose SAM** (Segment Anything Model)
4. **Click products** to auto-generate bounding boxes
5. **Adjust boxes** to fit tightly around products
6. **Move to next image** (arrow keys)
7. **Repeat** until all images are labeled

### Labeling Tips

- **Tight boxes**: Draw closely around each product
- **All instances**: Label every visible product in each image
- **Consistent style**: Include full product with visible label text
- **Handle occlusion**: Still label partially hidden products
- **Quality > Speed**: Better labels = better model

### Export

1. Go to **Versions** → **Create New Version**
2. Preprocessing:
   - ✅ Auto-Orient
   - Resize to 640×640
3. **Generate** → **Export Dataset** → **YOLO v8**
4. Download ZIP → extract to `scanncart/dataset/`

---

## Step 4: Train the Model

### Prerequisites

- Labeled dataset in `scanncart/dataset/`
- GPU recommended (NVIDIA with CUDA)
- Ultralytics installed: `pip install ultralytics`

### Training Command

```bash
cd scanncart
python train_yolo11.py \
  --data dataset/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

### Model Options

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| `yolo11n.pt` | Fast | Good | ~2GB | Development, iteration |
| `yolo11s.pt` | Medium | Better | ~3GB | Balanced |
| `yolo11m.pt` | Slow | Best | ~5GB | Production |

**Recommendation**: Start with `yolo11n.pt` for fast iteration, then upgrade to `yolo11s.pt` or `yolo11m.pt` for production.

### Training Output

```
scanncart/yolo11-grocery/
├── weights/
│   ├── best.pt          # ← Use this one
│   └── last.pt
├── results.csv
├── confusion_matrix.png
├── results.png          # Training curves
└── ...
```

### Monitoring Training

- Watch `results.png` for loss curves
- Check confusion matrix for class-specific issues
- Early stopping triggers if no improvement for 20 epochs

---

## Step 5: Deploy to Sidecar

### Deploy Script

```bash
python deploy_model.py
```

This copies `best.pt` to `sidecar/data/custom-grocery.pt`.

### Manual Deploy

```bash
cp scanncart/yolo11-grocery/weights/best.pt sidecar/data/custom-grocery.pt
```

### Use in App

1. Start SCANnCART desktop app
2. Open **Admin Panel**
3. Select **"data/custom-grocery.pt"** from Model dropdown
4. Click **Save**
5. Go to **Live View** → **Start**

### Verify Detection

- Products should appear in the item log
- Bounding boxes should appear on the preview
- Check confidence scores are reasonable (>0.5)

---

## Adding More Classes Later

### Workflow Summary

```
1. Collect 250+ photos of new product
2. Zip and place in scanncart/raw/
3. Add class prefix to clean_dataset.py
4. Run: python clean_dataset.py
5. Run: python upload_to_roboflow.py
6. Label new images in Roboflow
7. Export updated dataset
8. Run: python train_yolo11.py
9. Run: python deploy_model.py
10. Restart app and select updated model
```

### Important Notes

- **Re-train from scratch**: YOLO11 doesn't support incremental learning; you must retrain with all classes
- **More classes = more data**: Each class needs 250+ images; 10 classes × 300 images = 3,000 images total
- **Balance classes**: Aim for similar image counts per class (within 2× of each other)
- **Update data.yaml**: Roboflow export includes all classes automatically

### Adding a Class: Checklist

- [ ] Collected 250+ images of new product
- [ ] Images cover multiple angles and lighting
- [ ] Zipped and placed in `scanncart/raw/`
- [ ] Added class prefix to `clean_dataset.py`
- [ ] Ran `clean_dataset.py` successfully
- [ ] Uploaded all classes to Roboflow
- [ ] Labeled all new images in Roboflow
- [ ] Exported updated dataset (YOLO format)
- [ ] Extracted to `scanncart/dataset/`
- [ ] Trained model with all classes
- [ ] Deployed with `deploy_model.py`
- [ ] Verified detection in SCANnCART app

---

## Troubleshooting

### "No mapping for folder" Warning

The class prefix in `clean_dataset.py` doesn't match the folder name inside your zip.

**Fix**: Check the zip contents and update the prefix:
```bash
unzip -l yourfile.zip | head -5  # See folder name
```

### Low Detection Accuracy

- **More data**: Add more images (aim for 300+ per class)
- **Better labels**: Tighter bounding boxes, label all instances
- **More epochs**: Try `--epochs 200`
- **Larger model**: Use `yolo11s.pt` or `yolo11m.pt`
- **Higher imgsz**: Try `--imgsz 960`

### Training Too Slow

- **Reduce batch size**: `--batch 8` or `--batch 4`
- **Use smaller model**: `yolo11n.pt`
- **Reduce imgsz**: `--imgsz 480`
- **Train on CPU**: `--device cpu` (much slower)

### Model Not Detected in App

- Check file exists: `ls sidecar/data/custom-grocery.pt`
- Check model name in Admin Panel matches exactly
- Restart the app after deploying

### Out of VRAM

- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 480`
- Use smaller model: `yolo11n.pt`
- Train on CPU: `--device cpu`

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `clean_dataset.py` | Clean + augment raw images | `python clean_dataset.py` |
| `upload_to_roboflow.py` | Batch upload to Roboflow | `python upload_to_roboflow.py --api-key KEY --project ID` |
| `train_yolo11.py` | Train YOLO11 model | `python train_yolo11.py --data data.yaml` |
| `deploy_model.py` | Copy model to sidecar | `python deploy_model.py` |

---

## Class Naming Convention

Use lowercase, hyphenated names:

| ❌ Bad | ✅ Good |
|--------|---------|
| CocaCola330ml | coca-cola-330ml |
| Bear Brand Milk | bear-brand-milk |
| 555 Sardines | 555-sardines |

---

## Model File Sizes

| Model | Parameters | File Size | Inference Speed |
|-------|-----------|-----------|-----------------|
| yolo11n.pt | 2.6M | ~5 MB | ~5 ms |
| yolo11s.pt | 9.4M | ~18 MB | ~10 ms |
| yolo11m.pt | 20.1M | ~40 MB | ~20 ms |
| Custom (6 classes) | ~2.6M | ~5 MB | ~5 ms |

---

## Future Improvements

Potential enhancements to the pipeline:

1. **Incremental Training**: Fine-tune existing model with new classes
2. **Auto-labeling**: Use existing model to pre-label new images
3. **Active Learning**: Flag low-confidence predictions for review
4. **Class Balancing**: Auto-augment underrepresented classes
5. **Model Compression**: Export to ONNX/TensorRT for faster inference
6. **Edge Deployment**: Optimize for Jetson/Raspberry Pi

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review Ultralytics docs: [docs.ultralytics.com](https://docs.ultralytics.com)
- Roboflow docs: [docs.roboflow.com](https://docs.roboflow.com)

---

**Last Updated**: August 2026
**Maintainer**: SCANnCART Team
