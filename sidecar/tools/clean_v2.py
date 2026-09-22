#!/usr/bin/env python
"""Clean the SCANnCART "CLOSE MID FAR" capture set and stage it for Roboflow.

Two phases, both non-destructive to the source tree:

  clean   ingest -> normalise to upright JPEG -> drop unusable/duplicate frames
          -> stage as <class>/<class>_NNNN.jpg -> write manifest + report
  upload  push the kept images to a Roboflow project with per-session batches,
          resumable via upload_state.json
  retag   add the tags and distance metadata the upload endpoint cannot carry,
          idempotently, for images that are already in the project
  sanity  read the project back and check it is set up for the capture
          checklist: class list (including a distance that has crept into a
          class name, which is blocking), version preprocessing, tag/manifest
          agreement, and the verdict on hard negatives (CAPTURE_CHECKLIST.md)

Why each step exists:
  * HEIC is not a format Roboflow ingests, so every image is re-encoded to JPEG.
  * Phone shots carry EXIF orientation; stored sideways they teach the model a
    rotation it will never see from the fixed StreamCam mount, so the pixels are
    transposed once, here.
  * EXIF also carries GPS on some of these files. Re-encoding drops it.
  * The source has `- Copy` / `(1)` twins and true frame-level near-duplicates.
    Left in, one physical item gets counted many times and train/val leak into
    each other, so they are collapsed before the manifest is written.
  * Distance is metadata, not a class: the class stays the SKU (so labels stay
    continuous with the v1 project) and close/mid/far rides along as a tag and
    as the batch name, which is also the group key for a by-session split.
  * The upload endpoint takes only ONE tag, whatever the docs imply: a repeated
    multipart field collapses to its last value server-side, so asking for
    [distance, class] silently kept only the class and dropped the distance -
    the one tag that carries information the filename does not. Upload therefore
    sends a single tag, and `retag` adds the rest through the metadata endpoint,
    which also writes distance/class/batch as image metadata.

Usage (from the repo root, with the sidecar venv for Pillow + pillow-heif):

    sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py sanity
    sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py clean
    sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload --dry-run
    sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload

Source images live outside the repo. The cleaned copy, the manifest and the
credentials file all live in the dataset *workspace* - sidecar/data/datasets/ by
default, which .gitignore excludes - not next to this file, so the tools can be
tracked without dragging gigabytes of JPEGs or an API key in with them. See
workspace.py, or override the location with SCANNCART_DATASET_ROOT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import resources  # must precede numpy: sets OMP/MKL thread limits before init
from workspace import DEFAULT_OUT, ENV_PATH  # workspace lives outside the repo tree

from PIL import Image, ImageOps
import numpy as np

try:  # HEIC/HEIF is the one format Pillow needs help with.
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - reported per-file instead of crashing
    pillow_heif = None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_SRC = Path(r"C:\Users\yusri\Downloads\Downloads\PRODUCTS\CLOSE MID FAR")
# DEFAULT_OUT and ENV_PATH come from workspace.py.
#
# Decoding is the only heavy step, and it happens in a thread pool. Derive the
# pool from the shared budget instead of hardcoding 8, which is what made an
# earlier run look like it had taken the machine over.
DEFAULT_WORKERS = resources.CPU_THREADS
STAGE_DIRNAME = "_stage-v2"
CACHE_NAME = "_fingerprints-v2.json"

JPEG_QUALITY = 92
IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff"}

# Folder name -> class name. v1 project names are reused verbatim so labels stay
# continuous; a rename here also has to happen in the Roboflow class list.
CLASS_MAP = {
    "BEARBRAND": "bear-brand-milk",
    "LUCKY ME": "lucky-me-pancit",
    "MILO": "milo",
    "PALMOLIVE": "palmolive",
    "SAFEGUARD": "safeguard",
    "SARDINES": "555-sardines",
    "Silver Swan": "silver-swan-vinegar",
    "TUNA": "century-tuna",
    # Pseudo-class for §2's hard negatives: frames with no roster product in them
    # (empty counter, hands, bags, a phone). There is nothing to annotate, so the
    # name is only a tag, a filename prefix and a batch key - see the capture
    # checklist for why these may not enter a version without extra steps.
    "NEGATIVES": "negative",
}

# The hard-negative pseudo-class, named once so `--negatives` cannot drift from the
# folder mapping above.
NEGATIVE_CLS = CLASS_MAP["NEGATIVES"]

# Distance folder name -> tag. The source uses three different spellings.
DISTANCE_MAP = {
    "CLOSE": "close",
    "CLOSE-UP": "close",
    "MID": "mid",
    "MIDDLE": "mid",
    "MID-SHOT": "mid",
    "FAR": "far",
    "FAR-SHOT": "far",
}

DISTANCE_ORDER = ["close", "mid", "far"]
SPLIT_NAMES = ("train", "valid", "test")

# The first capture. Every unsuffixed batch, every manifest written before sessions were
# recorded, and every retag that did not name one has meant this session - so it is the
# default rather than an empty value. Absent is not the same as unrecorded: the split
# planner reasons about sessions, and "unknown" would have to be treated as its own
# session, which would invent a second one out of a missing field.
DEFAULT_SESSION = "s1"

# Checklist Tier A: the eight cells that are empty or too thin to measure, and how many
# images each needs. Keyed by (SOURCE FOLDER NAME, distance) - the folder name as it must
# appear on disk, not the slug - so `scaffold` cannot create a folder that `ingest` would
# then skip as unmapped. Three of these names contain a space, which is exactly where a
# hand-made tree goes wrong: a typo costs a whole capture session, silently.
TIER_A_CELLS: dict[tuple[str, str], int] = {
    ("TUNA", "mid"): 40,
    ("TUNA", "far"): 40,
    ("PALMOLIVE", "close"): 40,
    ("Silver Swan", "mid"): 35,
    ("LUCKY ME", "far"): 27,
    ("PALMOLIVE", "mid"): 27,
    ("SARDINES", "mid"): 23,
    ("LUCKY ME", "mid"): 21,
}

# Tier D: the held-out acceptance session (see CAPTURE_CHECKLIST.md). It is a *re-shoot*
# of cells earlier sessions already cover, so it is derived rather than hand-listed: every
# product x every distance, same count. That is the whole point of the tier - a cell only
# this session covers would have no train images and be unlearnable - so there is no
# per-cell judgement to encode here, and nothing to drift out of step with the roster.
TIER_D_PER_CELL = 15
TIER_D_CELLS: dict[tuple[str, str], int] = {
    (product, distance): TIER_D_PER_CELL
    for product in CLASS_MAP
    if product != "NEGATIVES"
    for distance in DISTANCE_ORDER
}


@dataclass(frozen=True)
class TierPlan:
    """One capture tier: what to shoot, and how the resulting session has to be filed.

    The filing is the part worth typing once. Each tier is its own capture session with
    its own staged set and its own batch suffix, and Tier D additionally has to plan the
    acceptance split *before* uploading, because the split is set at upload time and an
    image's split cannot be changed afterwards by re-uploading it.
    """

    key: str
    title: str
    cells: dict[tuple[str, str], int]
    session: str
    note: str
    out: str
    holdout: bool = False


TIER_A = TierPlan(
    key="a",
    title="Tier A capture",
    cells=TIER_A_CELLS,
    session="s2",
    note=(
        "Every cell here is below its target, and none is empty by accident: these are the\n"
        "cells where a per-class x per-distance reading is currently impossible."
    ),
    out=f"{DEFAULT_OUT}-s2",
)

TIER_D = TierPlan(
    key="d",
    title="Tier D capture - the held-out acceptance session (s3)",
    cells=TIER_D_CELLS,
    session="s3",
    note=(
        "**Re-shoot only cells an earlier session already covers.** Holding a session out\n"
        "removes its frames from `train`, so a cell whose only coverage is this session has\n"
        "no train images, and its test reading would measure the absence of training rather\n"
        "than the model. Shoot Tier A first if you have not, and shoot this on a different\n"
        "day from the earlier sessions - that is the variable being isolated."
    ),
    out=f"{DEFAULT_OUT}-s3",
    holdout=True,
)

# Dedup thresholds. Hamming distance is over a 64-bit dhash; MSE is over a
# 16x16 grey thumbnail and only ever tightens a match the hash already found.
# Defaults; overridable per run on the CLI. The dedup evidence table in the
# report shows where the drops actually land, because a threshold that cuts
# through the middle of a smooth distribution is discarding real variation.
#
# These started at 4 / 24 and were tightened after the evidence table showed a
# smooth decay with no gap - at 4/24 the cut was eating 124 real frames, most of
# them in the thin far/mid buckets this dataset exists to fix. At 2 / 8 the MSE
# gate is the binding constraint: 53 pairs whose hashes matched to within two
# bits were kept because their thumbnails genuinely differ, which is exactly the
# "same product, different angle" case we must not delete.
NEAR_HAMMING = 2
NEAR_MSE = 8.0
# Cross-bucket collisions are the same photo filed under two classes (or two
# distances), i.e. a labelling conflict. Stricter, because merging two genuinely
# different products would silently delete real data.
CONFLICT_HAMMING = 2
CONFLICT_MSE = 8.0


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


@dataclass
class SourceImage:
    """One candidate image, still encoded, with where it came from."""

    origin: str  # "file" or "zip"
    source: str  # absolute path on disk
    entry: str  # zip member name, or "" for loose files
    name: str  # basename, for reporting
    data: bytes  # raw encoded bytes
    cls: str
    distance: str

    @property
    def label(self) -> str:
        return f"{self.source}!{self.entry}" if self.entry else str(Path(self.source).name) + "/" + self.name


def _read_zip_images(path: Path, cls: str, distance: str, report: list[str]) -> list[SourceImage]:
    """A distance folder was actually a zip archive (Silver Swan/CLOSE, 661 MB)."""
    out: list[SourceImage] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir() or "__MACOSX" in info.filename:
                    continue
                if Path(info.filename).suffix.lower() not in IMAGE_EXTS:
                    continue
                with zf.open(info) as fh:
                    out.append(
                        SourceImage(
                            origin="zip",
                            source=str(path),
                            entry=info.filename,
                            name=Path(info.filename).name,
                            data=fh.read(),
                            cls=cls,
                            distance=distance,
                        )
                    )
        report.append(f"[zip] {path.name} -> {len(out)} images ({cls}/{distance})")
    except zipfile.BadZipFile as exc:
        report.append(f"[zip] FAILED {path}: {exc}")
    return out


def ingest(src: Path, report: list[str]) -> list[SourceImage]:
    """Walk <product>/<distance>/ and yield every candidate image."""
    images: list[SourceImage] = []
    if not src.is_dir():
        raise SystemExit(f"source folder not found: {src}")

    for product_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        cls = CLASS_MAP.get(product_dir.name)
        if cls is None:
            report.append(f"[skip] unmapped product folder: {product_dir.name}")
            continue

        for sub in sorted(product_dir.iterdir()):
            if sub.is_file():
                # A distance folder saved as an archive: name drives the distance.
                distance = DISTANCE_MAP.get(sub.name.upper())
                if distance is None and sub.suffix == "":
                    report.append(f"[warn] unrecognised file in {product_dir.name}: {sub.name}")
                    continue
                if sub.suffix.lower() == ".zip" or sub.stat().st_size > 50_000_000:
                    if distance:
                        images += _read_zip_images(sub, cls, distance, report)
                    continue
                continue

            distance = DISTANCE_MAP.get(sub.name.upper())
            if distance is None:
                report.append(f"[warn] unmapped distance folder: {product_dir.name}/{sub.name}")
                continue

            found = 0
            for f in sorted(sub.rglob("*")):
                if not f.is_file():
                    continue
                # A nested zip inside a distance folder still counts as that distance.
                if f.suffix.lower() == ".zip":
                    images += _read_zip_images(f, cls, distance, report)
                    continue
                if f.suffix.lower() not in IMAGE_EXTS:
                    report.append(f"[warn] skipping non-image: {f}")
                    continue
                if f.stat().st_size == 0:
                    report.append(f"[warn] zero-byte file: {f}")
                    continue
                images.append(
                    SourceImage(
                        origin="file",
                        source=str(f),
                        entry="",
                        name=f.name,
                        data=f.read_bytes(),
                        cls=cls,
                        distance=distance,
                    )
                )
                found += 1
            if found == 0 and distance:
                report.append(f"[gap] {product_dir.name}/{sub.name} is empty ({cls}/{distance})")

    return images


def ingest_negatives(path: Path, report: list[str]) -> list[SourceImage]:
    """§2's hard-negative frames, which have no `<product>/<distance>/` tree to walk.

    These come off the app's own camera (`cam0_*.jpg`, the deployment view) rather than
    being filed by hand, so there is no product folder to name them after and no
    distance to record. They are staged under the `negative` pseudo-class with an empty
    distance, which is what keeps them from inventing a distance cell in the coverage
    report - there is nothing to annotate in them at all, only to mark as background.
    """
    if not path.is_dir():
        raise SystemExit(f"negatives folder not found: {path}")
    out: list[SourceImage] = []
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        if f.stat().st_size == 0:
            report.append(f"[warn] zero-byte negative: {f}")
            continue
        out.append(
            SourceImage(
                origin="file",
                source=str(f),
                entry="",
                name=f.name,
                data=f.read_bytes(),
                cls=NEGATIVE_CLS,
                distance="",
            )
        )
    report.append(f"[negatives] {path} -> {len(out)} frame(s) ({NEGATIVE_CLS}, no distance)")
    return out


# --------------------------------------------------------------------------
# Normalise + fingerprint
# --------------------------------------------------------------------------


@dataclass
class Record:
    idx: int
    src: SourceImage
    staged: str = ""
    width: int = 0
    height: int = 0
    bytes: int = 0
    sha256: str = ""
    dhash: int = 0
    thumb: np.ndarray | None = None
    lap_var: float = 0.0
    status: str = "kept"
    reason: str = ""
    dup_of: str = ""
    dup_hamming: int = -1
    dup_mse: float = -1.0
    new_name: str = ""

    @property
    def bucket(self) -> tuple[str, str]:
        return (self.src.cls, self.src.distance)


def dhash64(im: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: robust to scale/compression, cheap to compare."""
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def grey_thumb(im: Image.Image, size: int = 16) -> np.ndarray:
    return np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.int16)


def laplacian_var(im: Image.Image) -> float:
    """Blur proxy. Reported, never used to delete (safe-clean mode)."""
    w = 512
    h = max(1, round(w * im.height / im.width))
    g = np.asarray(im.convert("L").resize((w, h), Image.LANCZOS), dtype=np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def normalise(data: bytes) -> tuple[Image.Image, str]:
    """Decode to an upright RGB image. Returns (image, soft-note).

    draft() first: on a 3,060x4,080 JPEG, libjpeg can decode at 1/8 scale, which
    is several times cheaper than a full decode. The result is identical after
    the resize that follows, so the fingerprints are unaffected.
    """
    note = ""
    im = Image.open(io.BytesIO(data))
    im.draft("RGB", (2048, 2048))  # fast path; no-op for formats/cases it can't help
    if im.mode not in ("RGB", "L"):
        note = f"converted from {im.mode}"
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im, note


def source_key(src: SourceImage) -> str:
    if src.origin == "zip":
        return f"zip:{src.source}:{src.entry}:{hashlib.sha1(src.data).hexdigest()[:16]}"
    p = Path(src.source)
    st = p.stat()
    return f"file:{src.source}:{st.st_size}:{int(st.st_mtime)}"


def process_one(
    idx: int, src: SourceImage, stage: Path, report: list[str], lock: threading.Lock
) -> Record:
    rec = Record(idx=idx, src=src)
    try:
        im, note = normalise(src.data)
    except Exception as exc:  # noqa: BLE001 - any decode failure means unusable
        rec.status = "dropped"
        rec.reason = f"undecodable ({type(exc).__name__}: {exc})"
        with lock:
            report.append(f"[drop] cannot decode {src.label}: {type(exc).__name__}: {exc}")
        return rec

    rec.width, rec.height = im.size
    rec.dhash = dhash64(im)
    rec.thumb = grey_thumb(im)
    rec.lap_var = laplacian_var(im)

    staged = stage / f"{idx:05d}.jpg"
    im.save(staged, "JPEG", quality=JPEG_QUALITY, optimize=True)  # no exif= -> GPS dropped
    rec.staged = str(staged)
    rec.bytes = staged.stat().st_size
    rec.sha256 = hashlib.sha256(staged.read_bytes()).hexdigest()
    if note and rec.width < 200:
        with lock:
            report.append(f"[note] tiny after decode: {src.label} {rec.width}x{rec.height} ({note})")
    return rec


# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.int32) - b.astype(np.int32)
    return float((diff * diff).mean())


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedup(records: list[Record], notes: list[str], log: list[str]) -> list[tuple[int, float]]:
    """Collapse frame-level near-duplicates, then cross-bucket conflicts.

    Ordering is deterministic (class, distance, source name) and within a
    cluster the largest/sharplest frame wins, so repeated runs keep the same
    images rather than shuffling which twin survives.
    """
    live = [r for r in records if r.status == "kept"]
    live.sort(key=lambda r: (r.src.cls, r.src.distance, r.src.name.lower(), r.idx))
    dup_pairs: list[tuple[int, float]] = []

    def note_pair(rec: Record, rep: Record, kind: str) -> None:
        hamming = _hamming(rec.dhash, rep.dhash)
        mse = _mse(rec.thumb, rep.thumb)
        rec.dup_hamming, rec.dup_mse = hamming, round(mse, 2)
        dup_pairs.append((hamming, mse))
        log.append(f"[{kind}] {rec.src.cls}/{rec.src.distance} {rec.src.name} == {rep.src.name} "
                   f"(hamming={hamming} mse={mse:.2f})")

    # Pass 1 - within the same (class, distance) bucket: burst frames. The
    # later twin is dropped, which is what "- Copy" and "(1)" files are.
    reps: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for rec in live:
        match = None
        for rep in reps[rec.bucket]:
            if _hamming(rec.dhash, rep.dhash) <= NEAR_HAMMING and _mse(rec.thumb, rep.thumb) < NEAR_MSE:
                match = rep
                break
        if match is None:
            reps[rec.bucket].append(rec)
        else:
            # Keep whichever frame is bigger on disk; it is the less-degraded copy.
            note_pair(rec, match, "dup:near")
            if rec.bytes > match.bytes:
                match.status, match.reason, match.dup_of = "dropped", "near-duplicate", rec.src.name
                match.dup_hamming, match.dup_mse = rec.dup_hamming, rec.dup_mse
                rec.dup_of = match.src.name
                reps[rec.bucket].remove(match)
                reps[rec.bucket].append(rec)
            else:
                rec.status = "dropped"
                rec.reason = "near-duplicate"
                rec.dup_of = match.src.name

    # Pass 2 - across buckets: the same photo filed under two classes or two
    # distances is a labelling conflict, not extra data.
    survivors = [r for r in live if r.status == "kept"]
    for i, a in enumerate(survivors):
        if a.status != "kept":
            continue
        for b in survivors[i + 1 :]:
            if b.status != "kept" or b.bucket == a.bucket:
                continue
            if _hamming(a.dhash, b.dhash) <= CONFLICT_HAMMING and _mse(a.thumb, b.thumb) < CONFLICT_MSE:
                b.status = "dropped"
                b.reason = "cross-bucket-duplicate"
                b.dup_of = a.src.name
                note_pair(b, a, "dup:conflict")
                notes.append(
                    f"[conflict] {b.src.cls}/{b.src.distance} {b.src.name} is the same photo as "
                    f"{a.src.cls}/{a.src.distance} {a.src.name} - dropped the later one, check your filing"
                )
    return dup_pairs


# --------------------------------------------------------------------------
# Stage + report
# --------------------------------------------------------------------------


def stage(records: list[Record], out: Path, report: list[str]) -> list[Record]:
    counters: Counter[str] = Counter()
    kept: list[Record] = []
    for rec in records:
        if rec.status != "kept":
            continue
        bucket = out / rec.src.cls
        bucket.mkdir(parents=True, exist_ok=True)
        counters[rec.src.cls] += 1
        name = f"{rec.src.cls}_{counters[rec.src.cls]:04d}.jpg"
        dest = bucket / name
        # Copy, not move: the staged file is what the fingerprint cache points
        # at, and keeping it lets a re-run with different dedup thresholds skip
        # the 3-minute decode entirely.
        shutil.copyfile(rec.staged, dest)
        rec.new_name = name
        rec.staged = str(dest)
        kept.append(rec)
    return kept


def write_manifest(
    kept: list[Record], all_records: list[Record], out: Path, session: str = DEFAULT_SESSION
) -> None:
    fields = [
        "new_name",
        "class",
        "distance",
        "batch",
        "session",
        "tags",
        "width",
        "height",
        "bytes",
        "sha256",
        "lap_var",
        "source_file",
        "zip_entry",
        "status",
        "reason",
        "dup_of",
        "dup_hamming",
        "dup_mse",
    ]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rec in sorted(all_records, key=lambda r: (r.src.cls, r.src.distance, r.new_name or r.src.name)):
            w.writerow(
                {
                    "new_name": rec.new_name,
                    "class": rec.src.cls,
                    "distance": rec.src.distance,
                    "batch": batch_name(rec.src.cls, rec.src.distance),
                    "session": session,
                    "tags": " ".join(tags_for(rec.src.cls, rec.src.distance, session)),
                    "width": rec.width,
                    "height": rec.height,
                    "bytes": rec.bytes,
                    "sha256": rec.sha256,
                    "lap_var": round(rec.lap_var, 1),
                    "source_file": rec.src.source,
                    "zip_entry": rec.src.entry,
                    "status": rec.status,
                    "reason": rec.reason,
                    "dup_of": rec.dup_of,
                    "dup_hamming": rec.dup_hamming,
                    "dup_mse": rec.dup_mse,
                }
            )
    (out / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "new_name": r.new_name,
                    "class": r.src.cls,
                    "distance": r.src.distance,
                    "batch": batch_name(r.src.cls, r.src.distance),
                    "session": session,
                    "tags": tags_for(r.src.cls, r.src.distance, session),
                    "source": r.src.source,
                    "zip_entry": r.src.entry,
                    "sha256": r.sha256,
                    "bytes": r.bytes,
                    "status": r.status,
                    "reason": r.reason,
                    "dup_of": r.dup_of,
                }
                for r in kept
            ],
            indent=1,
        ),
        encoding="utf-8",
    )


def batch_name(cls: str, distance: str) -> str:
    """`<class>_<distance>`, or just the class for frames that have no distance.

    Hard negatives are the one thing here with no distance to name. A null-annotated
    frame carries no box, so close/mid/far would be a value nobody observed - and it
    would then appear as a real cell in the per-distance coverage report.
    """
    return f"{cls}_{distance}" if distance else cls


def batch_for_session(batch: str, session: str) -> str:
    """The batch name an upload creates for a session: `milo_mid`, or `milo_mid_s2`.

    The first session has no suffix, because that is what every pre-session upload
    produced - and it is also the rule that stops a later session merging into the
    first one's batch, which would silently collapse two sessions into one split key.

    One function, used by `upload` and `retag` and read by `plan_split`, because the
    batch *is* the split unit: a planner that names batches differently from the
    uploader would assign splits to batches that never exist.
    """
    return batch if session in ("", DEFAULT_SESSION) else f"{batch}_{session}"


def tags_for(cls: str, distance: str, session: str = "") -> list[str]:
    """Tags an image should end up with, distance first (see upload_one).

    Order matters because only the first is sent at upload, and `retag` fills in the
    rest: distance first keeps the tag the filename does not already encode. A frame
    with no distance ends up with the pseudo-class first instead.

    The capture session goes last, and it is a *tag* rather than only metadata because
    of a gap in the platform: Roboflow's dataset search has a `tag:` filter and no
    `batch:` filter, so a batch can group images but cannot select them. Without this
    tag there is no query for "everything from session 2", and no way to check whether
    train and test share a session - which is the one thing the split rule in
    MODEL_TRAINING.md §8.3 is trying to prevent. (Checked against the documented filter
    list: tag, filename, split, class, date, like-image, job, min/max width, height and
    annotation counts. No batch.)

    It cannot ride the upload itself: only one tag survives that call, and it has to be
    the distance. `retag` is what writes this one.
    """
    tags = [distance, cls] if distance else [cls]
    if session:
        tags.append(session)
    return tags


def tier_commands(plan: TierPlan, root: Path) -> list[str]:
    """The exact commands that turn a shot tier into an uploaded session.

    Order matters and is the part most easily got wrong. `plan_split` reads the *staged*
    manifest, not the project, so for a held-out session it has to run after `clean` (which
    writes that manifest) and before `upload` (which is the only chance to set a split -
    re-uploading an image later returns `{"duplicate": true}` and leaves its split alone).

    `--include` is what makes the holdout check meaningful: the staged sets are one per
    session, so a planner reading only s3's would see every cell as s3-only and refuse a
    plan that is in fact fine, because s1 covers those cells in another directory.
    """
    py = "sidecar/.venv/Scripts/python.exe"
    tool = "sidecar/tools/clean_v2.py"
    # POSIX separators in every path, because these lines are meant to be pasted into a shell
    # and on Windows `Path.__str__` renders `sidecar\data\...` - where bash eats the backslash
    # as an escape (`sidecar\d` -> `sidecard`). A `clean --src` that resolves to nothing
    # ingests nothing, which is the silent post-shoot failure this file exists to prevent.
    src = Path(root).as_posix()
    out = Path(plan.out).as_posix()
    base = Path(DEFAULT_OUT).as_posix()
    lines = [
        f"{py} {tool} clean --src {src} --session {plan.session} --out {out}",
    ]
    if plan.holdout:
        lines += [
            f"{py} sidecar/tools/plan_split.py --out {base} --include {out} "
            f"--holdout-session {plan.session}",
            f"# must NOT print `CANNOT hold out session {plan.session}` - if it does, one of the",
            "# cells above is covered only by this session; shoot it into an earlier session first",
            f"{py} {tool} upload --out {out} --session {plan.session} "
            f"--split-plan {base}/split_plan_c.json",
        ]
    else:
        lines.append(f"{py} {tool} upload --out {out} --session {plan.session}")
    lines.append(f"{py} {tool} retag --out {out} --session {plan.session}")
    return lines


def scaffold_tree(plan: TierPlan, root: Path, current_total: int | None = None) -> list[Path]:
    """Create the `<PRODUCT>/<DISTANCE>/` skeleton for a tier's cells, plus a README.

    Every name comes from CLASS_MAP/DISTANCE_MAP, so the tree that gets created is by
    construction one the ingest will accept - the failure this prevents is a folder typed
    from the checklist ("LUCKY ME" vs "Lucky Me") being skipped as an unmapped product,
    which the operator only discovers after shooting.
    """
    made: list[Path] = []
    for (product, distance), _want in sorted(plan.cells.items()):
        if CLASS_MAP.get(product) is None:
            raise SystemExit(f"scaffold: {product!r} is not a CLASS_MAP folder name")
        if DISTANCE_MAP.get(distance.upper()) is None:
            raise SystemExit(f"scaffold: {distance!r} is not a known distance")
        folder = root / product / distance.upper()
        folder.mkdir(parents=True, exist_ok=True)
        made.append(folder)

    lines = [
        f"# {plan.title}",
        "",
        "Folder names here are the ingest contract - do not rename them.",
        "",
        "| folder | distance | target images | class it becomes |",
        "|---|---|---:|---|",
    ]
    for (product, distance), want in sorted(plan.cells.items()):
        lines.append(f"| `{product}/` | `{distance.upper()}` | {want} | `{CLASS_MAP[product]}` |")
    total = sum(plan.cells.values())
    lines += ["", f"**{total} images to shoot.**"]
    if current_total:
        # Only claimed when the staged set could actually be counted; a number baked into
        # this file would start lying the moment the set changed.
        lines.append(
            f"That would take the staged set from {current_total} to {current_total + total}."
        )
    lines += [
        "",
        plan.note,
        "",
        "Shoot at the same rig height and re-mark the close/mid/far positions before starting,",
        "then:",
        "",
        "```bash",
        *tier_commands(plan, root),
        "```",
        "",
        "See docs/CAPTURE_CHECKLIST.md for what to shoot in each cell.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return made


def write_report(
    src: Path,
    kept: list[Record],
    all_records: list[Record],
    out: Path,
    notes: list[str],
    dup_pairs: list[tuple[int, float]],
) -> str:
    total = len(all_records)
    dropped = [r for r in all_records if r.status != "kept"]
    reasons = Counter(r.reason.split(" (")[0] for r in dropped)

    lines = [
        "# v2 dataset clean report",
        "",
        f"Source: `{src}`",
        f"Kept **{len(kept)}** of {total} candidate images ({len(dropped)} dropped).",
        "",
        "## Kept per class and distance",
        "",
        "| class | close | mid | far | total |",
        "|---|---|---|---|---|",
    ]
    grid: dict[str, Counter[str]] = defaultdict(Counter)
    for r in kept:
        grid[r.src.cls][r.src.distance] += 1
    # The distance axis belongs to the roster. Hard negatives have no distance and get
    # their own row, so they cannot pad a close/mid/far column with frames nobody placed.
    roster = {c: row for c, row in grid.items() if c != NEGATIVE_CLS}
    for cls in sorted(roster):
        row = roster[cls]
        lines.append(
            f"| `{cls}` | {row['close']} | {row['mid']} | {row['far']} | {sum(row.values())} |"
        )
    if NEGATIVE_CLS in grid:
        lines.append(
            f"| `{NEGATIVE_CLS}` (background, no distance) | — | — | — | "
            f"{sum(grid[NEGATIVE_CLS].values())} |"
        )
    lines += [
        f"| **total** | **{sum(c['close'] for c in roster.values())}** | "
        f"**{sum(c['mid'] for c in roster.values())}** | **{sum(c['far'] for c in roster.values())}** | "
        f"**{len(kept)}** |",
        "",
        "## Dropped",
        "",
    ]
    if reasons:
        for reason, n in reasons.most_common():
            lines.append(f"- `{reason}`: {n}")
    else:
        lines.append("- nothing")
    lines += ["", "## Sharpness by bucket (laplacian variance, higher = sharper)", ""]
    lap: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in kept:
        lap[(r.src.cls, r.src.distance)].append(r.lap_var)
    lines += ["| class | distance | n | p10 | median |", "|---|---|---|---|---|"]
    for key in sorted(lap):
        vals = sorted(lap[key])
        p10 = vals[max(0, int(0.10 * (len(vals) - 1)))]
        med = vals[len(vals) // 2]
        lines.append(f"| `{key[0]}` | {key[1] or '—'} | {len(vals)} | {p10:.0f} | {med:.0f} |")

    # Distance cells only. Reporting `negative`/close, /mid and /far as three empty
    # buckets would be three false capture instructions for frames that are never
    # supposed to hold a distance - the opposite of what this section is for.
    thin = [
        f"- `{cls}`/{d}: {grid[cls][d]} images"
        for cls in sorted(grid)
        if cls != NEGATIVE_CLS
        for d in DISTANCE_ORDER
        if grid[cls][d] < 40
    ]
    lines += ["", "## Thin buckets (< 40 images) - capture more here", ""]
    lines += thin or ["- none"]

    # Did the near-duplicate cut land in an obvious gap between "same frame" and
    # "same product, different photo"? If a wall of drops sits at the threshold,
    # the threshold is eating real variation and wants lowering.
    if dup_pairs:
        ham = Counter(h for h, _ in dup_pairs)
        lines += ["", "## Dedup evidence", "", "| hamming | dropped pairs |", "|---|---|"]
        for h in sorted(ham):
            lines.append(f"| {h} | {ham[h]} |")
        tight = sum(1 for h, m in dup_pairs if h >= NEAR_HAMMING and m >= NEAR_MSE / 2)
        max_mse = max(m for _, m in dup_pairs)
        lines += [
            "",
            f"Max MSE across dropped pairs: {max_mse:.2f} (limit {NEAR_MSE}). "
            f"{tight} pair(s) sit at the hamming limit with a loose thumbnail match.",
        ]

    lines += ["", "## Notes", ""]
    lines += [f"- {n}" for n in notes] or ["- none"]
    lines += ["", f"Per-image detail: `{out.name}/clean_log.txt` and `manifest.csv`."]
    text = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------
# Clean entrypoint
# --------------------------------------------------------------------------


def staged_coverage(dirs: Iterable[Path]) -> Counter[tuple[str, str]]:
    """(class, distance) -> images, across staged sets, for the held-out coverage check.

    Tolerant of a missing or unreadable manifest on purpose: this answers a question
    *before* a shoot, and a set that has not been staged yet is the ordinary case rather
    than an error to report.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for directory in dirs:
        manifest = Path(directory).expanduser() / "manifest.json"
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for e in entries:
            counts[(e.get("class"), e.get("distance"))] += 1
    return counts


def holdout_gaps(
    plan: TierPlan, coverage: Mapping[tuple[str, str], int]
) -> list[tuple[str, str]]:
    """A held-out tier's cells that have no coverage outside the session itself.

    These are the cells that make `plan_split --holdout-session` refuse (exit 2, nothing
    written): holding a session out removes its frames from `train`, so a cell only that
    session shot has no train images, and its test reading would measure the absence of
    training rather than the model. Asked here, when the folders are being laid out,
    because both fixes are decisions about a camera - shoot the cell into an earlier
    session, or drop it from this shoot - and the alternative is discovering it after
    shooting the session.

    The zero test is deliberately the planner's own (see `plan_split.summarize`'s
    `no_train_cells`), and a test in sidecar/tests/test_dataset_tools.py holds the two
    together rather than trusting that they stay equal. A non-held-out tier has no gaps to
    report: Tier A *is* the fix for them, so it would name its own cells as problems.
    """
    if not plan.holdout:
        return []
    return [
        (CLASS_MAP[product], distance)
        for (product, distance) in sorted(plan.cells)
        if coverage.get((CLASS_MAP[product], distance), 0) == 0
    ]


def _print_holdout_gaps(plan: TierPlan, gaps: list[tuple[str, str]], coverage: Counter, sources: list[Path]) -> None:
    """Report what the held-out session would leave unlearnable, or that it cannot tell.

    The "cannot tell" case is its own message rather than an empty list padded out to
    "every cell is a gap": with nothing staged there is no answer yet, and 24 alarming
    lines would train the operator to ignore the one that matters.
    """
    if not plan.holdout:
        return
    print()
    if not coverage:
        print(f"held-out coverage check: no staged set to compare against (looked in {', '.join(str(s) for s in sources)})")
        print(f"  Nothing staged means nothing can cover {plan.session}'s cells yet, so this cannot")
        print("  say whether the hold-out will plan. Stage the earlier sessions first.")
        return
    if not gaps:
        print(f"held-out coverage check: every one of these {len(plan.cells)} cell(s) already has")
        print(f"  images outside {plan.session}, so holding it out leaves them all learnable.")
        return
    print(f"held-out coverage check: {len(gaps)} of these {len(plan.cells)} cell(s) have NO images")
    print(f"outside {plan.session}, so shooting them only into this session leaves them with no")
    print("train images - `plan_split --holdout-session` will refuse the plan (exit 2):")
    for cls, distance in gaps:
        print(f"    {cls}/{distance}")
    print("  Shoot these into an earlier session (they are Tier A's own cells) or drop them from")
    print(f"this shoot. Everything else here is safe to re-shoot into {plan.session}.")


def cmd_scaffold(args: argparse.Namespace) -> int:
    plan = TIER_A if args.tier == "a" else TIER_D
    root = Path(args.root).expanduser()
    # Read before writing anything: the answer decides what to shoot, and it is the same
    # answer whether this is a dry run or the real thing.
    sources = [Path(args.out).expanduser()] + [Path(p).expanduser() for p in args.include]
    coverage = staged_coverage(sources)
    gaps = holdout_gaps(plan, coverage)
    if args.dry_run:
        print(f"--dry-run: would create {len(plan.cells)} folder(s) under {root}")
        for (product, distance), want in sorted(plan.cells.items()):
            print(f"  {product}/{distance.upper()}/  -> {want} images")
        _print_holdout_gaps(plan, gaps, coverage, sources)
        return 0
    # The projection in the README is only printed when the staged set can be counted.
    current: int | None = None
    manifest = Path(args.out).expanduser() / "manifest.json"
    if manifest.exists():
        try:
            current = len(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            current = None

    made = scaffold_tree(plan, root, current)
    print(f"created {len(made)} folder(s) under {root}")
    for folder in made:
        print(f"  {folder.relative_to(root)}")
    print(f"README -> {root / 'README.md'}")
    print()
    # Count-aware: Tier D has a planning step between clean and upload, because its split
    # has to be planned before the images are uploaded (a split cannot be set afterwards).
    n = len([c for c in tier_commands(plan, root) if not c.startswith("#")])
    print(f"Put the images in those folders, then run the {n} commands in the README.")
    _print_holdout_gaps(plan, gaps, coverage, sources)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    global NEAR_HAMMING, NEAR_MSE
    NEAR_HAMMING, NEAR_MSE = args.near_hamming, args.near_mse

    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    stage_dir = out / STAGE_DIRNAME

    if pillow_heif is None:
        print("WARNING: pillow_heif missing - HEIC files will fail to decode", file=sys.stderr)

    budget = resources.measure(resources.USE_PERCENT, args.disk_reserve_gb)
    print(budget.describe())
    print(f"decode pool: {args.workers} threads")

    notes: list[str] = []
    log: list[str] = []
    images = ingest(src, notes)
    if args.negatives:
        neg_dir = Path(args.negatives).expanduser()
        negatives = ingest_negatives(neg_dir, notes)
        images += negatives
        print(f"ingested {len(negatives)} hard-negative frame(s) from {neg_dir}")
    print(f"ingested {len(images)} candidate images from {src}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Optionally cache fingerprints and staged JPEGs across runs, which is what
    # makes trying a different dedup threshold cheap. Off by default now: it holds
    # ~2.5 GB on a drive that is 86% full, which is too much to spend silently on
    # a convenience. Pass --keep-cache to opt in.
    cache_path = out / CACHE_NAME
    cache: dict[str, dict] = {}
    if args.cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}

    records: list[Record] = []
    lock = threading.Lock()
    done = hits = 0
    t0 = time.time()

    for i, img in enumerate(images):
        rec = None
        if args.cache:
            entry = cache.get(source_key(img))
            if entry and Path(entry.get("staged", "")).exists():
                rec = Record(
                    idx=i,
                    src=img,
                    staged=entry["staged"],
                    width=entry["width"],
                    height=entry["height"],
                    bytes=entry["bytes"],
                    sha256=entry.get("sha256", ""),
                    dhash=entry["dhash"],
                    thumb=np.asarray(entry["thumb"], dtype=np.int16),
                    lap_var=entry["lap_var"],
                )
                hits += 1
        if rec is None:
            rec = process_one(i, img, stage_dir, notes, lock)
            if args.cache and rec.status == "kept":
                cache[source_key(img)] = {
                    "staged": rec.staged,
                    "width": rec.width,
                    "height": rec.height,
                    "bytes": rec.bytes,
                    "sha256": rec.sha256,
                    "dhash": rec.dhash,
                    "thumb": rec.thumb.tolist(),
                    "lap_var": rec.lap_var,
                }
        records.append(rec)
        done += 1
        if done % 200 == 0:
            print(f"  prepared {done}/{len(images)} ({hits} from cache, {time.time() - t0:.0f}s)")
    records.sort(key=lambda r: r.idx)
    print(f"prepared {len(records)} images in {time.time() - t0:.0f}s ({hits} from cache)")
    if args.cache:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    dup_pairs = dedup(records, notes, log)
    n_dup = sum(1 for r in records if r.status != "kept")
    print(f"dropped {n_dup} unusable/duplicate images")

    for cls_dir in out.iterdir():
        if cls_dir.is_dir() and cls_dir.name != STAGE_DIRNAME:
            for f in cls_dir.glob("*.jpg"):
                f.unlink()

    # Worst case the final copy is as big as everything staged.
    staged_bytes = sum(f.stat().st_size for f in stage_dir.glob("*.jpg"))
    budget.check_disk(str(out), extra_bytes=staged_bytes)
    kept = stage(records, out, notes)
    write_manifest(kept, records, out, args.session)
    (out / "clean_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    if args.cache:
        stage_gb = sum(f.stat().st_size for f in stage_dir.glob("*.jpg")) / 1e9
        notes.append(f"[cache] keeping {stage_dir.name} ({stage_gb:.1f} GB) for fast threshold re-runs")
    else:
        shutil.rmtree(stage_dir, ignore_errors=True)
        cache_path.unlink(missing_ok=True)

    notes.append(f"[threshold] near-dup cut: hamming <= {NEAR_HAMMING} and mse < {NEAR_MSE}")
    text = write_report(src, kept, records, out, notes, dup_pairs)

    print()
    print(text)
    print(f"staged -> {out}")
    print(f"manifest -> {out / 'manifest.csv'}")
    return 0


# --------------------------------------------------------------------------
# Roboflow upload
# --------------------------------------------------------------------------


def load_credentials(args: argparse.Namespace) -> tuple[str, str]:
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    project = args.project or os.environ.get("ROBOFLOW_PROJECT_ID", "")
    if (not api_key or not project) and ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("ROBOFLOW_API_KEY=") and not api_key:
                api_key = line.split("=", 1)[1].strip()
            elif line.startswith("ROBOFLOW_PROJECT_ID=") and not project:
                project = line.split("=", 1)[1].strip()
    if not api_key:
        raise SystemExit(f"no Roboflow API key (pass --api-key or set it in {ENV_PATH})")
    if not project:
        raise SystemExit(f"no Roboflow project id (pass --project or set it in {ENV_PATH})")
    return api_key, project


def upload_one(
    client: "httpx.Client",
    api_key: str,
    project: str,
    name: str,
    path: Path,
    batch: str,
    tags: list[str],
    split: str = "",
) -> dict:
    """One image.

    `split` may be a single name or, from --split-plan, a per-image choice; it is
    only sent when non-empty so the default (everything on train) is unchanged.

    Every field except the file goes in the querystring, which is how the docs
    list them ("Querystring parameters accepted by the API"). That is not
    cosmetic: as a multipart form field, `batch` is silently ignored and the
    image lands in the default "Uploaded via API" batch. `name` and `tag` do
    survive as form fields, which is what makes the failure quiet - the names
    come back right, so only the batches look wrong afterwards.

    `split` is left off by default: the batch is the session key and Roboflow's
    generator splits by batch, so pre-assigning train here would fight the
    70/20/10-by-session rule from MODEL_TRAINING.md §8.3, which also records the
    per-class and per-distance coverage checks the batch split needs.

    Only the first tag is sent. A repeated field keeps its last value
    server-side (Starlette's form accessor returns the last entry), so passing
    [distance, class] would drop the distance and keep only the class - which the
    filename already encodes. `retag` adds the rest through the metadata API.
    """
    url = f"https://api.roboflow.com/dataset/{project}/upload"
    query: dict = {"api_key": api_key, "name": name, "batch": batch}
    if tags:
        query["tag"] = tags[0]
    if split:
        query["split"] = split
    for attempt in range(5):
        try:
            resp = client.post(
                url,
                params=query,
                files={"file": (name, path.read_bytes(), "image/jpeg")},
                timeout=120.0,
            )
        except Exception as exc:  # noqa: BLE001 - network hiccup, retry
            if attempt == 4:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(2**attempt)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == 4:
                return {"ok": False, "error": f"HTTP {resp.status_code} after retries"}
            time.sleep(2**attempt)
            continue

        try:
            body = resp.json()
        except ValueError:
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        if isinstance(body, dict) and body.get("error"):
            err = body["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return {"ok": False, "error": msg}
        return {"ok": True, "id": (body or {}).get("id", "")}
    return {"ok": False, "error": "exhausted retries"}


def delete_images(client: "httpx.Client", api_key: str, workspace: str, project: str, ids: list[str], chunk: int = 200) -> int:
    """Delete images by id. Returns the number of ids successfully submitted."""
    done = 0
    for i in range(0, len(ids), chunk):
        resp = client.request(
            "DELETE",
            f"https://api.roboflow.com/{workspace}/{project}/images",
            params={"api_key": api_key},
            headers={"Content-Type": "application/json"},
            content=json.dumps({"images": ids[i : i + chunk]}),
        )
        if resp.status_code == 204:
            done += len(ids[i : i + chunk])
        else:
            raise SystemExit(f"delete chunk at {i} failed: HTTP {resp.status_code} {resp.text[:200]}")
    return done


def cmd_wipe(args: argparse.Namespace) -> int:
    """Remove every image from the project. Exists because the `batch` bug left
    1,383 images in the wrong batch, and a duplicate upload will not re-batch an
    image that is already there - so a clean re-upload has to start empty."""
    try:
        import httpx
    except ImportError:
        raise SystemExit("httpx is not installed - run this with the sidecar venv python")

    api_key, project = load_credentials(args)
    if not args.yes and not args.dry_run:
        raise SystemExit("refusing to delete a project's images without --yes")

    with httpx.Client(timeout=180) as client:
        index = fetch_index(client, api_key, args.workspace, project)
        ids = [rec["id"] for rec in index.values()]
        print(f"project {project}: {len(ids)} images to delete")
        if args.dry_run:
            print("--dry-run: nothing deleted")
            return 0
        deleted = delete_images(client, api_key, args.workspace, project, ids)
        print(f"deleted {deleted}")

        # Do not trust one re-sweep. The search index is eventually consistent and
        # also *drops* records mid-flight: right after deleting 1,383 images a
        # single sweep reported 198 remaining, a second reported 1, and the next
        # reported 0 - so a one-shot check both cries wolf and hides stragglers.
        # Poll until the count settles instead.
        remaining: dict[str, dict] = {}
        for attempt in range(8):
            remaining = fetch_index(client, api_key, args.workspace, project, attempts=2)
            if not remaining:
                break
            time.sleep(5)
        print(f"images remaining: {len(remaining)}")
        if remaining:
            print(f"WARNING: project not empty - {len(remaining)} image(s) left, re-run to catch stragglers")
            print("NOTE: the project's `unannotated` counter does not decrement on delete, so it is")
            print("      not a usable emptiness signal here; the search index is.")
            return 1
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    try:
        import httpx
    except ImportError:
        raise SystemExit("httpx is not installed - run this with the sidecar venv python")

    out = Path(args.out).expanduser()
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} - run `clean` first")

    api_key, project = load_credentials(args)

    # Fail before the first upload rather than halfway through 1,383: a typo'd
    # project id or a classification project would reject every image.
    with httpx.Client(timeout=60) as probe:
        info = probe.get(f"https://api.roboflow.com/{args.workspace}/{project}", params={"api_key": api_key})
        if info.status_code != 200:
            raise SystemExit(f"project {args.workspace}/{project} not reachable: HTTP {info.status_code} {info.text[:200]}")
        body = info.json()
        if "error" in body:
            raise SystemExit(f"project {project}: {body['error']}")
        # The project endpoint wraps the payload in a "project" key.
        meta = body.get("project", body)
        print(f"project type: {meta.get('type')}  images: {meta.get('images')}  versions: {meta.get('versions')}")
        if meta.get("type") != "object-detection":
            raise SystemExit(f"project {project} is {meta.get('type')!r}, expected 'object-detection'")
        if meta.get("public"):
            print(
                "WARNING: project is PUBLIC - these store captures are world-readable.\n"
                "         Public is the free plan's default and is not a per-project toggle you\n"
                "         can flip here; a private dataset needs a paid plan (Roboflow's own\n"
                "         docs: 'on paid plans, data is private by default'). Uploading is\n"
                "         still what puts the images in front of a labeler, so decide now."
            )

    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if args.limit:
        entries = entries[: args.limit]

    # A split plan is per-image, not per-batch: plan_split.py's stratified plan
    # splits one (class, distance) batch across train/valid/test, which a
    # per-batch flag could not express. Note that applying a plan to a project
    # that already has these images needs a wipe first - a re-upload of identical
    # bytes returns {"duplicate": true} and leaves the old split in place.
    plan: dict[str, str] = {}
    if args.split_plan:
        plan_path = Path(args.split_plan).expanduser()
        if not plan_path.exists():
            raise SystemExit(f"no split plan at {plan_path} - run plan_split.py first")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        unknown = [s for s in set(plan.values()) if s not in SPLIT_NAMES]
        if unknown:
            raise SystemExit(f"split plan has values that are not splits: {unknown}")
        covered = sum(1 for e in entries if e["new_name"] in plan)
        print(f"split plan: {plan_path}  ({covered} of {len(entries)} images covered)")
        if covered < len(entries):
            print(f"WARNING: {len(entries) - covered} image(s) are not in the plan and will go to the default split")

    def split_for(entry: dict) -> str:
        return plan.get(entry["new_name"], args.split)

    state_path = out / "upload_state.json"
    if args.restart and state_path.exists():
        state_path.unlink()
        print("restarting: previous upload state cleared")
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    todo = [e for e in entries if state.get(e["new_name"], {}).get("ok") is not True]

    print(f"project   : {project}")
    print(f"api key   : ****{api_key[-4:]}")
    # One place for the batch name, used by both the preview and the send, so a
    # dry run cannot promise different batches from the ones that get created.
    def batch_for(entry: dict) -> str:
        return batch_for_session(entry["batch"], args.session or DEFAULT_SESSION)

    print(f"images    : {len(entries)} in manifest, {len(entries) - len(todo)} already uploaded, {len(todo)} to send")
    if args.session:
        note = (
            "no suffix, this is session 1"
            if args.session == DEFAULT_SESSION
            else f"batches suffixed _{args.session}"
        )
        print(f"session   : {args.session} ({note})")
    batches = Counter(batch_for(e) for e in todo)
    for b, n in sorted(batches.items()):
        print(f"  batch {b:32} {n}")
    if plan:
        per_split = Counter(split_for(e) for e in todo)
        print("  " + "  ".join(f"{s}: {per_split.get(s, 0)}" for s in SPLIT_NAMES))

    if args.dry_run:
        print("\n--dry-run: nothing sent")
        return 0

    ok = fail = 0
    lock = threading.Lock()
    t0 = time.time()

    def work(entry: dict, client=None) -> None:
        nonlocal ok, fail
        path = out / entry["class"] / entry["new_name"]
        if not path.exists():
            with lock:
                fail += 1
            state[entry["new_name"]] = {"ok": False, "error": "staged file missing"}
            return
        # A new capture session must not merge into the previous session's batch,
        # or the batch stops being a session key and the split in §8.3 loses its
        # meaning. --session s2 uploads into <class>_<distance>_s2 instead.
        res = upload_one(
            client, api_key, project, entry["new_name"], path, batch_for(entry), entry["tags"], split_for(entry)
        )
        with lock:
            state[entry["new_name"]] = res
            if res["ok"]:
                ok += 1
            else:
                fail += 1
                print(f"  FAILED {entry['new_name']}: {res['error']}")
            # Checkpoint so an interrupted run resumes instead of restarting.
            if (ok + fail) % 50 == 0:
                state_path.write_text(json.dumps(state), encoding="utf-8")
                print(f"  {ok} uploaded, {fail} failed, {len(todo) - ok - fail} left ({time.time() - t0:.0f}s)")

    with httpx.Client() as client:
        # httpx.Client is thread-safe; one connection pool for the whole run.
        from functools import partial

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(partial(work, client=client), todo))

    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"\nuploaded {ok}, failed {fail} in {time.time() - t0:.0f}s")
    print(f"state -> {state_path}")
    if fail:
        print("re-run the same command to retry only the failures")
        return 1
    return 0


# --------------------------------------------------------------------------
# Retag / metadata
# --------------------------------------------------------------------------


def _index_sweep(client: "httpx.Client", api_key: str, workspace: str, project: str) -> tuple[dict[str, dict], int]:
    """One paginated pass. Returns (name -> record, total reported by server)."""
    out: dict[str, dict] = {}
    total = 0
    offset = 0
    while True:
        resp = client.post(
            f"https://api.roboflow.com/{workspace}/{project}/search",
            params={"api_key": api_key},
            json={"limit": 500, "offset": offset, "fields": ["name", "tags", "split"]},
        )
        body = resp.json()
        if "results" not in body:
            raise SystemExit(f"search failed: {str(body)[:200]}")
        results = body["results"]
        total = body.get("total", total)
        for rec in results:
            out[rec["name"]] = rec
        offset += len(results)
        if not results or offset >= total:
            break
    return out, total


def fetch_index(
    client: "httpx.Client", api_key: str, workspace: str, project: str, attempts: int = 6
) -> dict[str, dict]:
    """name -> search record, paginated, converged.

    Two things make a single sweep untrustworthy. The route that lists a
    project's images directly is not available on this plan (404), so search is
    the only way to read back what landed. And search is eventually consistent:
    three back-to-back sweeps of the same 1,383-image project returned 1,297,
    1,380 and 1,383 records. Since a short sweep is indistinguishable from
    "those images were never uploaded", the sweep is repeated and unioned until
    it reaches the server's own total - otherwise retag would silently skip
    real images and report them as missing.
    """
    merged: dict[str, dict] = {}
    total = 0
    for attempt in range(attempts):
        sweep, total = _index_sweep(client, api_key, workspace, project)
        merged.update(sweep)
        if total and len(merged) >= total:
            break
        if attempt < attempts - 1:
            time.sleep(3)
    if total and len(merged) < total:
        print(f"WARNING: read back {len(merged)} of {total} images after {attempts} sweeps")
    return merged


def poll_task(client: "httpx.Client", api_key: str, url: str, timeout_s: int = 300) -> list:
    """The batch endpoint is async (202 + task url). Wait for it, surface errors."""
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        body = client.get(url, params={"api_key": api_key}).json()
        last = body if isinstance(body, dict) else {}
        status = str(last.get("status", last.get("state", ""))).lower()
        if status in ("done", "completed", "complete", "failed", "error") or last.get("progress") == 1:
            break
        time.sleep(3)
    else:
        return [{"error": "task polling timed out"}]
    for key in ("errors", "failures", "failed"):
        if last.get(key):
            return list(last[key])
    if str(last.get("status", "")).lower() in ("failed", "error"):
        return [{"error": str(last)[:400]}]
    return []


def tags_wanted(entry: dict, session: str) -> list[str]:
    """The full tag set an image should carry, de-duped with order kept.

    Used by both the update and the read-back, so verification cannot end up checking a
    different set from the one it applied - which is how the session tag would go
    unchecked. Manifest tags are unioned in rather than replaced, so a hand-added tag
    survives.
    """
    return list(
        dict.fromkeys(tags_for(entry["class"], entry["distance"], session) + list(entry["tags"] or []))
    )


def cmd_retag(args: argparse.Namespace) -> int:
    try:
        import httpx
    except ImportError:
        raise SystemExit("httpx is not installed - run this with the sidecar venv python")

    out = Path(args.out).expanduser()
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} - run `clean` first")

    api_key, project = load_credentials(args)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    # Resolved once, so the tag and the metadata cannot disagree: an unlabelled retag is
    # session 1, which is what the metadata already assumed.
    session = args.session or DEFAULT_SESSION

    with httpx.Client(timeout=180) as client:
        index = fetch_index(client, api_key, args.workspace, project)
        print(f"project {project}: {len(index)} images indexed")

        missing_on_server = [e["new_name"] for e in entries if e["new_name"] not in index]
        updates = []
        for e in entries:
            rec = index.get(e["new_name"])
            if rec is None:
                continue
            have = set(rec.get("tags") or [])
            need = [t for t in tags_wanted(e, session) if t not in have]
            if need:
                updates.append(
                    {
                        "imageId": rec["id"],
                        "addTags": need,
                        "metadata": {
                            "distance": e["distance"],
                            "sku": e["class"],
                            "batch": batch_for_session(e["batch"], session),
                            "session": session,
                        },
                    }
                )

        by_tag = Counter(t for u in updates for t in u["addTags"])
        print(f"images needing tags: {len(updates)} of {len(entries)}")
        for tag, n in by_tag.most_common():
            print(f"  +{tag:24} {n}")
        if missing_on_server:
            print(f"WARNING: {len(missing_on_server)} manifest image(s) not found in {project}:")
            for n in missing_on_server[:10]:
                print(f"  {n}")

        if args.dry_run:
            print("\n--dry-run: nothing sent")
            return 0
        if not updates:
            print("nothing to do")

        errors: list = []
        for i in range(0, len(updates), 1000):
            chunk = updates[i : i + 1000]
            resp = client.post(
                f"https://api.roboflow.com/{args.workspace}/images/metadata",
                params={"api_key": api_key},
                json={"updates": chunk},
            )
            if resp.status_code == 202:
                task = resp.json()
                print(f"  chunk {i // 1000 + 1}: {len(chunk)} images queued...")
                errors += poll_task(client, api_key, task["url"])
            elif resp.status_code == 200:
                print(f"  chunk {i // 1000 + 1}: {len(chunk)} images applied synchronously")
            else:
                errors.append({"chunk": i // 1000 + 1, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"})

    for e in errors[:10]:
        print(f"  ERROR: {e}")

    # Read back rather than trust the 202s - the whole point of retag is that the
    # upload path was quietly lossy once already.
    with httpx.Client(timeout=180) as client:
        index = fetch_index(client, api_key, args.workspace, project)
    counts: Counter[str] = Counter()
    still_missing = Counter()
    for e in entries:
        have = set((index.get(e["new_name"]) or {}).get("tags") or [])
        for t in tags_wanted(e, session):
            if t in have:
                counts[t] += 1
            else:
                still_missing[t] += 1
    print("\nverified tags (image counts):")
    for tag, n in counts.most_common():
        print(f"  {tag:26} {n}")
    if still_missing:
        for tag, n in still_missing.most_common():
            print(f"  STILL MISSING {tag}: {n}")
        return 1
    print(f"\nall {len(entries)} images carry their intended tags")
    return 0


# --------------------------------------------------------------------------
# Pre-capture sanity check
# --------------------------------------------------------------------------

# The 8 canonical class names, exactly as MODEL_TRAINING.md §8.1 records them.
# Rows 1-7 are v1's names copied verbatim - that is the whole reason labels stay
# continuous with the 1,815 images already annotated in `scanncart-grocery`.
# Row 8 (Palmolive) is genuinely new, so the real name is still to be chosen.
#
# These are CLASS names, not the upload tags. The tags (`bear-brand-milk`, ...)
# are per-image metadata and are already on all 1,383 images; the class list is
# project-wide, set by hand on Settings -> Classes, and read-only from here.
V1_CLASSES = [
    "Bear Brand Fortified Powdered Milk 33g",
    "lucky_me_pancit_canton_calamansi_flavor",
    "555 sardines 155grams",
    "century_tuna_flakes_in_oil_155_grams",
    "silver_swan_sukang_puti_200ML",
    "Milo Chocolate Drink 22g Sachet",
    "safeguard_pure_white_60g",
]
# The sidecar feeds every frame at this size (settings.imgsz) and
# resolve_resize_mode() resolves `auto` to `stretch` for a custom .onnx, so a
# version generated with a different preprocessing trains at a different scale
# than inference uses. v1 used 640x640 Stretch to.
# Imported rather than retyped: this is the geometry `generate_version.py` actually sends,
# and a second copy here could disagree with it - which would make `sanity` approve a
# version the generator would never produce (or worse, reject the one it does).
from generate_version import EXPECTED_RESIZE  # noqa: E402  (kept with its consumers below)

# The distance-as-class predicate, shared rather than re-spelled here: `label_classes.py`
# checks the live project with it and `train_model.check_export` checks a generated version
# with it, so all three verdicts on the same class name have to come from the same words.
from label_classes import distance_tokens_in  # noqa: E402

# Settled by reading Roboflow's docs rather than guessing: the mechanism exists.
# Kept as one string so the checklist and this command cannot drift apart.
NULL_ANNOTATION_VERDICT = """\
Hard negatives (checklist Tier C2b) - the route EXISTS, and it is a UI action.

  A background image is a *null annotation*: an image deliberately marked as
  containing no object of interest. You mark it in the annotator with the
  Mark Null tool - the empty-set button in the right-hand toolbar, keyboard
  shortcut N - and it then counts as annotated, so it is eligible to enter a
  generated version. A plain unannotated image is not: Roboflow excludes it and
  counts it under `unannotated`.

  The trap is the 'include images without annotations' toggle at version
  generation: it sweeps in EVERY unreviewed image, not just the ones you meant.
  Mark the ~30 frames individually with the null tool instead of using it.

  This cannot be done from the upload API. The API exposes no null flag, and
  the annotate endpoint rejects an empty annotation file ("Unrecognized
  annotation format") because it sniffs the body to infer the format. For 30
  frames, marking them by hand in the UI is the sane route.

  Verifying it worked: a null annotation IS distinguishable from an unlabeled
  image, but only by the *type* of the search API's `annotations` field - not by
  its emptiness, and not by the project's `unannotated` counter (which does not
  decrement on delete and cannot be trusted as either a total or a work count):

      annotations: []                        -> not labeled yet, counted unannotated
      annotations: {"count": 0, ...}          -> null annotation, counted annotated
      annotations: {"count": n>0, ...}        -> labeled

  So `count: 0` inside an object is a *decision someone made*, while no object at
  all is outstanding work. Confirmed against the v1 project: all 1,516 images
  return an object, 16 of them with count 0, and v1 reports `unannotated: 0`.

  Practical upshot for C2b: `label_progress.py` lists null annotations, so the
  check is "did my 30 pure-background frames register as nulls", not a guess."""


def _looks_like_a_session(tag: str) -> bool:
    """`s2`, `neg1`, `session3` - enough to nudge an operator at a session tag the manifest
    does not know about. Deliberately loose: it is a prompt to look, not a parser, and a
    false positive is one line of output.
    """
    tag = tag.lower()
    for prefix in ("session", "s", "neg"):
        tail = tag[len(prefix) :]
        if tail.isdigit() and tag.startswith(prefix):
            return True
    return False


def _trashed_versions(client: "httpx.Client", api_key: str, workspace: str, project: str) -> list[dict]:
    """Versions sitting in the workspace Trash. They keep their numbers, so they change
    what the next generation will be called."""
    resp = client.get(f"https://api.roboflow.com/{workspace}/trash", params={"api_key": api_key})
    if resp.status_code != 200:
        return []
    items = resp.json().get("items") or []
    return [i for i in items if i.get("type") == "version" and i.get("parentUrl") == project]


def _emit(results: list[tuple[str, str, str]]) -> None:
    """Print the check list, aligned, so a long detail never hides the verdict."""
    for level, label, detail in results:
        mark = {"ok": ".ok.", "warn": "WARN", "fail": "FAIL"}[level]
        print(f"[{mark}] {label}")
        for line in detail.splitlines() if detail else []:
            print(f"        {line}")


def hard_negative_verdict(on_server: int, staged_here: int) -> tuple[str, str, str]:
    """Whether the set has hard negatives, judged from the project rather than the manifest.

    The hard negatives are their own set with their own manifest (`cleaned-negatives`), so
    counting this manifest's entries reported "0 staged" for a project that already had
    all 50 uploaded - a warning worse than silence, because the work it sends you to do
    had already been done.

    `staged_here` is still reported, because "none here" is the *expected* state, and
    reading it as a gap is exactly the mistake this function exists to prevent.
    """
    status = "ok" if on_server else "warn"
    title = f"hard negatives: {on_server} in the project"
    if staged_here:
        title += f", {staged_here} staged here"
    if on_server:
        detail = f"{on_server} frame(s) in the project carry the `{NEGATIVE_CLS}` tag"
        detail += (
            f"; {staged_here} also sit in this manifest (unusual - they are their own set)"
            if staged_here
            else "; they come from their own staged set, so this manifest holding none is expected"
        )
        detail += (
            ". Each needs a null annotation (not just an upload) to enter a version - "
            "`label_progress.py` reports how many are marked"
        )
    else:
        detail = (
            "none in the project - shoot them (CAPTURE_CHECKLIST Tier C2b) or point "
            "`clean --negatives <dir>` at existing frames; see the verdict below"
        )
    return status, title, detail


def class_list_rows(classes: object) -> list[tuple[str, str, str]]:
    """The class list's sanity rows: pure, so the verdict is testable without the network.

    Two independent failures live here, and they send an operator to opposite actions. A
    **missing** name is work not done yet (create it, then label). A name carrying a
    **distance** is work done in the wrong shape: those classes have to be deleted and their
    annotations moved onto the product class, and further labeling only adds to what has to be
    moved - which is why this one is blocking rather than a warning.

    Both are checked here, at the point the checklist already gates a shoot on (`sanity`,
    MODEL_TRAINING.md section 9), rather than being left to the two consumers of the same
    predicate's other two consumers: `label_classes.py` sees the live project when asked, and
    `train_model.check_export` sees a version only after one has been generated - i.e. after a
    version number is already spent.
    """
    names = {str(n) for n in (classes or [])}
    if not names:
        return [(
            "fail",
            "no classes defined yet",
            "set the 8 names from MODEL_TRAINING.md section 8.1 on Settings -> Classes and\n"
            "turn on Lock Classes BEFORE labeling. Renaming a class later rewrites every\n"
            "annotation that used it and deleting one deletes them - neither is\n"
            "reversible. The upload tags on the images are NOT the class list.",
        )]

    missing = [c for c in V1_CLASSES if c not in names]
    extra = sorted(n for n in names if n not in V1_CLASSES)
    palmolive = sorted(n for n in names if "palmolive" in n.lower())
    detail = f"{len(names)} class(es) defined"
    if missing:
        detail += f"; missing {len(missing)} v1 name(s):\n  " + "\n  ".join(missing)
    if palmolive:
        detail += f"; palmolive class: {palmolive[0]}"
    else:
        detail += "; no palmolive class yet (the one genuinely new class)"
    if extra:
        detail += f"; unexpected: {', '.join(extra)}"
    rows: list[tuple[str, str, str]] = [("warn" if missing else "ok", "class list", detail)]

    tainted = {n: words for n, words in ((n, distance_tokens_in(n)) for n in sorted(names)) if words}
    if tainted:
        rows.append((
            "fail",
            f"{len(tainted)} class name(s) carry a distance: "
            + ", ".join(repr(n) for n in tainted),
            "distance is a tag on the image and a cell in the coverage tables - never a class - so\n"
            "these split one product into three and train one head output per product-and-distance\n"
            "(24 instead of 8). Every box then comes back under a name the app's own roster does\n"
            "not contain, and nothing errors, because a class *name* records none of this.\n"
            "Fix the Classes tab, MOVE the annotations onto the product class (a class-list edit\n"
            "alone orphans the boxes), then regenerate the version - a version number cannot be\n"
            "reused. `label_classes.py` and `train_model.check_export` run the same check.",
        ))
    return rows


def cmd_sanity(args: argparse.Namespace) -> int:
    """Verify the project is set up to receive what the checklist says to shoot.

    This runs BEFORE a capture session on purpose. Every check below is cheap and
    every one of them is a thing that is expensive to discover later: a missing
    class list, or a preprocessing size that does not match inference, is
    recoverable only by relabeling or by regenerating the version.
    """
    try:
        import httpx
    except ImportError:
        raise SystemExit("httpx is not installed - run this with the sidecar venv python")

    api_key, project = load_credentials(args)
    session = args.session or DEFAULT_SESSION
    results: list[tuple[str, str, str]] = []

    print("Roboflow pre-capture sanity check")
    print(f"  workspace : {args.workspace}")
    print(f"  project   : {project}")
    print(f"  session   : {session}")
    print(f"  api key   : ****{api_key[-4:]}")
    print()

    with httpx.Client(timeout=60) as client:
        resp = client.get(f"https://api.roboflow.com/{args.workspace}/{project}", params={"api_key": api_key})
        if resp.status_code != 200:
            raise SystemExit(f"project not reachable: HTTP {resp.status_code} {resp.text[:300]}")
        body = resp.json()
        if "error" in body:
            raise SystemExit(f"{project}: {body['error']}")
        meta = body.get("project", body)
        versions = body.get("versions") or []

        ptype = meta.get("type")
        results.append((
            "ok" if ptype == "object-detection" else "fail",
            f"project type is object-detection  ({ptype})",
            "" if ptype == "object-detection" else "only object-detection yields boxes + track_id, which is all the sidecar consumes",
        ))

        if meta.get("public"):
            results.append((
                "warn",
                "project is PUBLIC",
                "these are store captures of a real shop floor, and public is the free plan's "
                "default - there is no toggle to flip, a private dataset needs a paid plan",
            ))

        # ---- images / versions ----
        annotated = meta.get("images", 0)
        unannotated = meta.get("unannotated", 0)
        results.append((
            "ok",
            f"images: {annotated} annotated, {unannotated} unannotated, {len(versions)} version(s)",
            "`annotated` excludes the unannotated, so a freshly-uploaded set reads as 0 images\n"
            "`unannotated` does NOT decrement on delete, so it is not an emptiness signal",
        ))

        # A trashed version still occupies its number, and this project already burned
        # one empty version by accident - so report the numbering the next generation
        # will actually get, rather than letting it be discovered later.
        trashed = _trashed_versions(client, api_key, args.workspace, project)
        if trashed:
            # Live versions carry a full path id (`ws/proj/2`); trashed ones carry just
            # the number, so both have to be reduced to their trailing segment.
            def _vnum(v: dict) -> int:
                tail = str(v.get("id", "")).rsplit("/", 1)[-1]
                return int(tail) if tail.isdigit() else 0

            trashed_nums = [_vnum(v) for v in trashed]
            nxt = max([_vnum(v) for v in versions] + trashed_nums + [0]) + 1
            results.append((
                "warn",
                f"{len(trashed)} version(s) in Trash: {', '.join(str(n) for n in trashed_nums)}",
                f"a trashed version still holds its number, so the next generation is version {nxt}.\n"
                "Remove it permanently from the Trash in the UI if the numbering matters;\n"
                "otherwise remember the weight filename - scanncart-grocery-v2.* - is what counts.",
            ))

        # ---- preprocessing: must match what inference feeds ----
        # The project's `preprocessing` is empty until a version carries settings, and
        # it is chosen when you generate one (`sanity` cannot pre-set it). So an empty
        # block means "not decided yet", which is a different thing from "decided wrong"
        # - saying the latter would put a warning on a project that is simply early.
        pre = meta.get("preprocessing") or {}
        want_w, want_h, want_fmt = EXPECTED_RESIZE
        resize = pre.get("resize") or {}
        w, h, fmt = resize.get("width"), resize.get("height"), resize.get("format")

        if not pre:
            results.append((
                "warn",
                "preprocessing not set yet - no version carries it",
                f"it is set when a version is generated, and it is not revisitable afterwards.\n"
                f"Generate it with the reviewed settings rather than by clicking:\n"
                f"  python sidecar/tools/generate_version.py --dry-run\n"
                f"  python sidecar/tools/generate_version.py --yes\n"
                f"That sends auto-orient on and resize {want_fmt} {want_w}x{want_h}. The sidecar\n"
                f"infers at settings.imgsz, so any other size trains at a scale inference never\n"
                f"uses - and for a locally trained .pt the geometry has to travel with the\n"
                f"weights, which is what train_model.py --install records beside them (resize_mode\n"
                f"'auto' then honours it; the format heuristic alone would answer *letterbox*).",
            ))
        else:
            matches = (w, h, fmt) == (want_w, want_h, want_fmt)
            detail = f"version preprocessing: {fmt} {w}x{h}"
            if not matches:
                detail += (
                    f", expected {want_fmt} {want_w}x{want_h}"
                    "\nthe sidecar infers at settings.imgsz and resolves resize_mode 'auto' to"
                    "\n'stretch' for a custom .onnx, so training at another size puts inference"
                    "\nat a scale the model never saw. Set 640x640 Stretch to, matching v1."
                )
            results.append(("ok" if matches else "warn", "version preprocessing matches inference", detail))

        auto_orient = pre.get("auto-orient")
        if pre:
            results.append((
                "ok" if auto_orient else "warn",
                f"auto-orient is {'on' if auto_orient else 'off'}",
                ""
                if auto_orient
                else "our clean step already bakes orientation into the pixels, so off is tolerable - but v1 had it on",
            ))

        # ---- class list: the thing that is expensive to fix after labeling ----
        # Including the distance words, because a class named `palmolive close` is the one
        # class-list problem that labeling cannot fix afterwards (see `class_list_rows`).
        results.extend(class_list_rows(meta.get("classes")))

        # ---- what actually landed: tags and batches ----
        index = fetch_index(client, api_key, args.workspace, project)
        tag_counts: Counter[str] = Counter()
        for rec in index.values():
            for t in rec.get("tags") or []:
                tag_counts[t] += 1

        # Seeding a class uploads a throwaway annotated image; probing did the same. If
        # one failed to clean up it sits in the project as an unannotated image and can
        # reach a version, so surface it rather than let it be counted as a real frame.
        leaked = [
            r for r in index.values() if str(r.get("name", "")).startswith(("__class_seed_", "__probe", "__global"))
        ]
        if leaked:
            results.append((
                "warn",
                f"{len(leaked)} tooling seed image(s) left in the project: {', '.join(sorted(str(r.get('name')) for r in leaked))[:120]}",
                "these came from creating classes / probing. Delete them by id (the images\n"
                "endpoint takes a list) so they cannot enter a generated version.",
            ))

    # ---- against the local manifest, which is the source of truth for intent ----
    manifest = Path(args.out).expanduser() / "manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        # Compared against the *wanted* set, not the manifest's own list, so the session
        # tag is checked too. That matters because `plan_split` reads the session from the
        # manifest and assumes the tags agree - a session that was recorded locally but
        # never stamped on the images is the one way that assumption goes wrong silently.
        want_tags = Counter(t for e in entries for t in tags_wanted(e, session))
        drift = {t: (want_tags[t], tag_counts.get(t, 0)) for t in want_tags if want_tags[t] != tag_counts.get(t, 0)}
        # Tags that exist on the server but were never intended: a renamed session, or a
        # hand-added tag, shows up here rather than in the drift above.
        stray = {t: n for t, n in tag_counts.items() if t not in want_tags}
        detail = f"manifest has {len(entries)} images; server has {len(index)}"
        if drift:
            detail += "\ntag count drift (manifest vs server):\n" + "\n".join(
                f"  {t}: {a} vs {b}" for t, (a, b) in sorted(drift.items())
            )
        results.append((
            "ok" if not drift and len(entries) == len(index) else "warn",
            "local manifest agrees with the server",
            detail,
        ))

        # The session is half local and half server-side, so it gets its own line: the
        # planner reads it from the manifest and assumes the tags agree, and this is the
        # only place that checks the two against each other.
        session_detail = (
            f"images carry the session tag `{session}`"
            if tag_counts.get(session)
            else f"NO image carries the session tag `{session}` - run `retag --session {session}`"
        )
        for tag, n in sorted(stray.items()):
            if _looks_like_a_session(tag):
                session_detail += (
                    f"\n`{tag}` is on {n} image(s) but this manifest records none of them. "
                    "A separate set is expected here (the hard negatives are their own "
                    "manifest); frames that belong to *this* set will be planned as "
                    f"`{session}`."
                )
        results.append((
            "ok" if tag_counts.get(session) else "warn",
            f"capture session: {session}",
            session_detail,
        ))

        d_counts = Counter(e["distance"] for e in entries)
        results.append((
            "ok",
            "distance split of the staged set",
            "  " + "  ".join(f"{d}: {d_counts.get(d, 0)}" for d in DISTANCE_ORDER),
        ))

        results.append(
            hard_negative_verdict(
                tag_counts.get(NEGATIVE_CLS, 0),
                sum(1 for e in entries if e["class"] == NEGATIVE_CLS),
            )
        )
    else:
        results.append((
            "warn",
            "no local manifest to compare against",
            f"expected {manifest} - run `clean` first if the set should already be uploaded",
        ))

    _emit(results)

    print()
    print(NULL_ANNOTATION_VERDICT)

    fails = sum(1 for level, _, _ in results if level == "fail")
    warns = sum(1 for level, _, _ in results if level == "warn")
    print()
    if fails:
        print(f"{fails} blocking problem(s) and {warns} warning(s) - fix the blocking ones before shooting")
        return 1
    print(f"no blocking problems, {warns} warning(s)")
    return 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scaffold", help="create the capture folder skeleton for a tier")
    sc.add_argument("--root", required=True, help="where to create <PRODUCT>/<DISTANCE>/ folders")
    sc.add_argument("--out", default=str(DEFAULT_OUT), help="staged set to size the projection against")
    sc.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="DIR",
        help="another staged set to count as coverage outside the held-out session (repeatable)",
    )
    sc.add_argument(
        "--tier",
        default="a",
        choices=["a", "d"],
        help="which tier's cells to lay out ('d' is the held-out acceptance session)",
    )
    sc.add_argument("--dry-run", action="store_true")
    sc.set_defaults(func=cmd_scaffold)

    c = sub.add_parser("clean", help="normalise, dedupe and stage the capture set")
    c.add_argument("--src", default=str(DEFAULT_SRC))
    c.add_argument("--out", default=str(DEFAULT_OUT))
    c.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=(
            "capture session, written into the manifest and the image tags "
            f"(default {DEFAULT_SESSION} - the first capture). Change it for a new session, or "
            "the split planner treats the new frames as the same session as the first."
        ),
    )
    c.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"default {DEFAULT_WORKERS} (20%% of {os.cpu_count()} cores)")
    c.add_argument("--disk-reserve-gb", type=float, default=resources.DISK_RESERVE_GB)
    c.add_argument("--near-hamming", type=int, default=NEAR_HAMMING, help="dhash bit slack for a near-duplicate")
    c.add_argument("--near-mse", type=float, default=NEAR_MSE, help="16x16 thumbnail MSE limit")
    c.add_argument(
        "--negatives",
        default="",
        help="folder of hard-negative frames (no <product>/<distance>/ tree) staged as the `negative` pseudo-class",
    )
    c.add_argument(
        "--keep-cache",
        dest="cache",
        action="store_true",
        default=False,
        help="keep the ~2.5 GB staging dir so a threshold re-run skips the decodes",
    )
    c.set_defaults(func=cmd_clean)

    u = sub.add_parser("upload", help="push the staged images to Roboflow")
    u.add_argument("--out", default=str(DEFAULT_OUT))
    u.add_argument("--project", default="")
    u.add_argument("--api-key", default="")
    u.add_argument("--workers", type=int, default=4)
    u.add_argument("--workspace", default="yusri-caloyloy")
    u.add_argument(
        "--session",
        default="",
        help="suffix so a new capture session gets its own batches (e.g. s2 -> milo_mid_s2)",
    )
    u.add_argument(
        "--split",
        default="",
        help="train/valid/test; leave empty and split by batch at dataset generation instead",
    )
    u.add_argument(
        "--split-plan",
        default="",
        help="JSON from plan_split.py mapping image name -> split; overrides --split per image",
    )
    u.add_argument("--limit", type=int, default=0, help="upload only the first N (smoke test)")
    u.add_argument("--restart", action="store_true", help="ignore upload_state.json and resend everything")
    u.add_argument("--dry-run", action="store_true")
    u.set_defaults(func=cmd_upload)

    w = sub.add_parser("wipe", help="delete every image in the project (clean re-upload)")
    w.add_argument("--project", default="")
    w.add_argument("--api-key", default="")
    w.add_argument("--workspace", default="yusri-caloyloy")
    w.add_argument("--yes", action="store_true")
    w.add_argument("--dry-run", action="store_true")
    w.set_defaults(func=cmd_wipe)

    t = sub.add_parser("retag", help="add the tags/metadata the upload endpoint drops")
    t.add_argument("--out", default=str(DEFAULT_OUT))
    t.add_argument("--project", default="")
    t.add_argument("--api-key", default="")
    t.add_argument("--workspace", default="yusri-caloyloy")
    t.add_argument(
        "--session",
        default="",
        help="capture session, recorded as a tag and in image metadata (default s1)",
    )
    t.add_argument("--dry-run", action="store_true")
    t.set_defaults(func=cmd_retag)

    s = sub.add_parser("sanity", help="check the project is ready to receive a capture session")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--project", default="")
    s.add_argument("--api-key", default="")
    s.add_argument("--workspace", default="yusri-caloyloy")
    s.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"capture session the staged set should be tagged with (default {DEFAULT_SESSION})",
    )
    s.set_defaults(func=cmd_sanity)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
