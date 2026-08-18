"""SCANnCART dataset cleaner + augmenter for YOLO11 custom training. v2"""
import hashlib
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ExifTags

ZIP_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(__file__).resolve().parent / "cleaned"
TARGET_COUNT = 300
MIN_DIM = 64
MAX_DIM = 8192
random.seed(42)
np.random.seed(42)

# Prefix-based matching for inner folder names
CLASS_PREFIXES = [
    ("555 (Sardines in Tomato Sauce)", "555-sardines"),
    ("Bear Brand Fortified Powdered Milk", "bear-brand-milk"),
    ("Century Tuna", "century-tuna"),
    ("Lucky Me Pancit Canton", "lucky-me-pancit"),
    ("Safeguard (Pure White)", "safeguard"),
    ("SILVER SWAN", "silver-swan-vinegar"),
]


def _resolve_class(inner_folder):
    lower = inner_folder.lower()
    for prefix, cls_name in CLASS_PREFIXES:
        if prefix.lower() in lower:
            return cls_name
    return None


# --- Augmentations ---
def aug_flip_h(img):
    return ImageOps.mirror(img)

def aug_rotate_5(img):
    return img.rotate(random.uniform(-5, 5), resample=Image.BICUBIC, fillcolor=(0, 0, 0))

def aug_rotate_10(img):
    return img.rotate(random.uniform(-10, 10), resample=Image.BICUBIC, fillcolor=(0, 0, 0))

def aug_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))

def aug_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))

def aug_sharpness(img):
    return ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.8))

def aug_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

def aug_jpeg_compress(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=random.randint(50, 90))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def aug_crop_rezoom(img):
    w, h = img.size
    ratio = random.uniform(0.85, 0.95)
    nw, nh = int(w * ratio), int(h * ratio)
    left = random.randint(0, w - nw)
    top = random.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)

def aug_color_jitter(img):
    return ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))


AUGMENTATIONS = [
    aug_flip_h, aug_rotate_5, aug_rotate_10,
    aug_brightness, aug_contrast, aug_sharpness,
    aug_blur, aug_jpeg_compress, aug_crop_rezoom, aug_color_jitter,
]


def apply_exif_orientation(img):
    try:
        exif = img.getexif()
        if not exif:
            return img
        for tag_id, val in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "Orientation":
                if val == 3:
                    img = img.rotate(180, expand=True)
                elif val == 6:
                    img = img.rotate(270, expand=True)
                elif val == 8:
                    img = img.rotate(90, expand=True)
                elif val in (2, 4, 5, 7):
                    img = ImageOps.mirror(img)
                break
    except Exception:
        pass
    return img


def heic_to_jpg(path):
    try:
        import pillow_heif
        heif = pillow_heif.read_heif(str(path))
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw", heif.mode, 0, 1)
        img = apply_exif_orientation(img)
        jpg_path = path.with_suffix(".jpg")
        img.convert("RGB").save(jpg_path, "JPEG", quality=95)
        path.unlink()
        return jpg_path
    except Exception as e:
        print(f"  [WARN] HEIC conversion failed: {path.name} - {e}")
        return None


def is_valid_image(img):
    w, h = img.size
    return min(w, h) >= MIN_DIM and max(w, h) <= MAX_DIM


def random_augment(img):
    n = random.randint(1, 3)
    for fn in random.sample(AUGMENTATIONS, k=n):
        img = fn(img)
    return img


def run():
    OUT_DIR.mkdir(exist_ok=True)

    # ---- Step 1: Extract all zips (flattened) ----
    print("\nStep 1: Extracting zips...\n")
    seen_hashes_per_class = defaultdict(set)
    class_files = defaultdict(set)

    for zf_path in sorted(ZIP_DIR.glob("*.zip")):
        with zipfile.ZipFile(zf_path) as zf:
            members = [m for m in zf.namelist() if not m.startswith(".") and "__MACOSX" not in m]
            if not members:
                continue
            inner_folder = members[0].split("/")[0]
            cls_name = _resolve_class(inner_folder)
            if not cls_name:
                print(f"  [WARN] No mapping for '{inner_folder}' - skipping {zf_path.name}")
                continue
            dest = OUT_DIR / cls_name
            dest.mkdir(exist_ok=True)

            added = 0
            skipped = 0
            for member in members:
                if member.endswith("/"):
                    continue
                data = zf.read(member)
                h = hashlib.sha256(data).hexdigest()
                # Skip exact byte-duplicates across zips during extraction
                if h in seen_hashes_per_class[cls_name]:
                    skipped += 1
                    continue
                seen_hashes_per_class[cls_name].add(h)

                fname = Path(member).name
                target = dest / fname
                # Handle name collision within same zip
                if target.exists():
                    base = Path(fname).stem
                    ext = Path(fname).suffix
                    target = dest / f"{base}_{hashlib.md5(data[:1024]).hexdigest()[:6]}{ext}"
                target.write_bytes(data)
                class_files[cls_name].add(target)
                added += 1

            print(f"  {zf_path.name}: {added} added, {skipped} cross-zip duplicates skipped")

    # ---- Step 2: Convert HEIC -> JPG ----
    print("\nStep 2: Converting HEIC -> JPG...\n")
    for cls in list(class_files.keys()):
        heic_files = [f for f in class_files[cls] if f.suffix.lower() in (".heic", ".heif")]
        if heic_files:
            print(f"  {cls}: {len(heic_files)} HEIC files")
            for hf in sorted(heic_files):
                new_path = heic_to_jpg(hf)
                class_files[cls].discard(hf)
                if new_path:
                    class_files[cls].add(new_path)
                    print(f"    OK {hf.name} -> {new_path.name}")

    # ---- Step 3: Validate + re-save as clean JPEG ----
    print("\nStep 3: Validating & normalising...\n")
    final_files = {}
    for cls in sorted(class_files.keys()):
        good = []
        bad_count = 0
        for p in sorted(class_files[cls]):
            if p.suffix.lower() in (".heic", ".heif"):
                continue
            try:
                raw = p.read_bytes()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img = apply_exif_orientation(img)
                if not is_valid_image(img):
                    bad_count += 1
                    p.unlink()
                    continue
                img.save(p, "JPEG", quality=95)
                good.append(p)
            except Exception:
                bad_count += 1
                try:
                    p.unlink()
                except Exception:
                    pass
        final_files[cls] = good
        print(f"  {cls}: {len(good)} valid / {bad_count} invalid")

    # ---- Step 4: Augment to >= TARGET_COUNT ----
    print(f"\nStep 4: Augmenting to >= {TARGET_COUNT} per class...\n")
    summary = {}
    for cls in sorted(final_files.keys()):
        originals = final_files[cls]
        count = len(originals)
        need = TARGET_COUNT - count
        cls_dir = OUT_DIR / cls
        augmented = 0
        if need > 0:
            print(f"  {cls}: {count} originals -> augmenting {need}...")
            for i in range(need):
                try:
                    img = Image.open(random.choice(originals)).convert("RGB")
                    aug = random_augment(img)
                    aug.save(cls_dir / f"aug_{i:04d}.jpg", "JPEG", quality=95)
                    augmented += 1
                except Exception as e:
                    print(f"    [WARN] Aug failed: {e}")
        else:
            print(f"  {cls}: {count} originals (no augmentation needed)")
        final_total = len(list(cls_dir.glob("*.jpg")))
        summary[cls] = {"originals": count, "augmented": augmented, "total": final_total}
        print(f"  -> {cls}: {final_total} total\n")

    # ---- Step 5: Rename all files sequentially ----
    print("Step 5: Renaming files...\n")
    for cls in sorted(summary.keys()):
        cls_dir = OUT_DIR / cls
        all_files = sorted(cls_dir.glob("*.jpg"))
        non_aug = sorted([f for f in all_files if not f.name.startswith("aug_")])
        aug_files = sorted([f for f in all_files if f.name.startswith("aug_")])
        idx = 1
        for f in non_aug + aug_files:
            new_name = f"{cls}_{idx:04d}.jpg"
            if f.name != new_name:
                f.rename(cls_dir / new_name)
            idx += 1
        final_count = len(list(cls_dir.glob("*.jpg")))
        summary[cls]["total"] = final_count
        print(f"  {cls}: {final_count} files")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DATASET CLEANING SUMMARY")
    print("=" * 60)
    hdr = "{:<30} {:>10} {:>10} {:>8}".format("Class", "Original", "Augmented", "Total")
    print(hdr)
    print("-" * 60)
    total_all = 0
    for cls in sorted(summary.keys()):
        s = summary[cls]
        row = "{:<30} {:>10} {:>10} {:>8}".format(cls, s["originals"], s["augmented"], s["total"])
        print(row)
        total_all += s["total"]
    print("-" * 60)
    tot = "{:<30} {:>10} {:>10} {:>8}".format("TOTAL", "", "", total_all)
    print(tot)
    print("=" * 60)
    print(f"\nOutput: {OUT_DIR.resolve()}")

    class_names = sorted(summary.keys())
    lines = ["# SCANnCART YOLO11 dataset", f"nc: {len(class_names)}", "names:"]
    for i, cls in enumerate(class_names):
        lines.append(f"  {i}: {cls}")
    (OUT_DIR / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"\nDone! data.yaml written to {OUT_DIR / 'data.yaml'}")
    print("\nNext steps:")
    print("  1. Label images (Roboflow, CVAT, or labelImg)")
    print("  2. Split into train/val")
    print("  3. Update data.yaml")
    print("  4. Train: yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    run()
