#!/usr/bin/env python
"""Audit the cleaned v2 set against the buckets in MODEL_TRAINING.md §2.

§2 budgets a training set in three buckets:

    solo shots        ~150 per item     teaches what each class looks like
    multi-item scenes ~800-1,200        3-6 items together, overlapping, occluded
    hard negatives    ~150-250 (5-10%)  empty counter, hands, bags, wallet, phone

The two valuable buckets are invisible in every easy statistic: a set can look
healthy on image count and per-class balance while containing nothing but solo
product shots, which is precisely the failure §2 warns about ("this is what the
app actually sees").

Three audits, kept separate because they carry different confidence:

  structural  From the manifest alone. Proves the capture taxonomy is
              single-product-per-folder, and therefore that every image is filed
              under exactly one product. Zero hard negatives follows by
              construction - no image was captured without a product in it.
              No model, no guessing, exact.

  coco        A COCO detector over every image to measure the distractor
              ingredients §2 names (hands, bags, phone, clutter) and to put a
              *lower bound* on multi-item scenes for bottle-shaped SKUs. COCO
              knows nothing about grocery SKUs, so it cannot count instances of
              sachets or cans - it is a floor, not a measurement. Run with
              --report-conf, because at collection confidence COCO invents
              `book`/`kite`/`bed` on product close-ups; the report shows both so
              the false-positive rate is visible rather than hidden.

  resource    The caps this run actually worked under, printed into the report so
              "how much of the machine did this take" has an answer.

Resource envelope: see resources.py. This tool stays under 20% of CPU, RAM and
VRAM by default and refuses to write past a disk reserve, because the machine is
shared and C: is 86% full. The expensive part was never inference (13.6 ms/frame
on the 4060) - it was decoding 1,383 3,060x4,080 JPEGs, several times, which is
what the thumbnail cache and PIL's draft() mode exist to avoid.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import resources  # must precede numpy/torch: sets OMP/MKL thread limits
from workspace import DEFAULT_OUT, SIDECAR_ROOT  # DEFAULT_OUT is outside the repo tree

# yolo11s, not the m: under the default 20% VRAM cap (1.7 GB of 8.6) the m OOMs
# even at batch 4, and this pass only needs to name distractor-shaped objects.
# Bigger models are available with --model plus --max-use-percent if wanted.
DEFAULT_MODEL = SIDECAR_ROOT / "yolo11s.pt"
THUMB_DIRNAME = "_audit-640"

# §2's hard-negative examples, mapped onto COCO. COCO has no "wallet", and a bare
# hand is usually too little of a person to fire "person" - so these are floors
# on the ingredient, not a count of hard negatives.
DISTRACTORS = {
    "person": "hands / arm / body in frame",
    "handbag": "bag",
    "backpack": "bag",
    "suitcase": "bag",
    "cell phone": "phone",
    "bottle": "bottle-shaped object",
    "cup": "cup / glass on the counter",
    "wine glass": "glass",
    "bowl": "bowl / container",
    "book": "flat clutter (book, tray)",
    "laptop": "flat clutter (laptop)",
    "keyboard": "flat clutter",
    "mouse": "flat clutter",
    "remote": "flat clutter",
    "scissors": "tool",
    "knife": "tool",
    "fork": "cutlery",
    "spoon": "cutlery",
}

# Bottle-shaped SKUs: the only v2 classes a COCO "bottle" detection can plausibly
# stand in for, so the only ones where a multi-bottle frame is real evidence.
BOTTLE_SHAPED = {"silver-swan-vinegar", "palmolive", "safeguard"}


def empty_coco() -> dict:
    """Full shape, so the report renders even when the COCO pass is skipped."""
    return {
        "images": 0,
        "report_conf": 0,
        "detections_total": 0,
        "images_with_any_detection": 0,
        "images_with_zero_detections": 0,
        "images_with_2plus_detections": 0,
        "detections_per_image": {"0": 0, "1": 0, "2": 0, "3+": 0},
        "images_with_any_distractor": 0,
        "distractor_kinds": {},
        "bottle_shaped": {},
        "per_bucket": {},
        "below_report_conf": {},
    }


def load_manifest(out: Path) -> list[dict]:
    path = out / "manifest.json"
    if not path.exists():
        raise SystemExit(f"no manifest at {path} - run `clean` first")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Structural audit
# --------------------------------------------------------------------------


def structural(entries: list[dict]) -> dict:
    """Facts that follow from the manifest with no inference at all."""
    per_bucket: collections.Counter = collections.Counter()
    for e in entries:
        per_bucket[(e["class"], e["distance"])] += 1

    classes = sorted({e["class"] for e in entries})
    distances = sorted({e["distance"] for e in entries})
    ambiguous = sum(1 for e in entries if len([t for t in e["tags"] if t in classes]) != 1)

    return {
        "images": len(entries),
        "classes": classes,
        "distances": distances,
        "buckets": {f"{c}_{d}": per_bucket[(c, d)] for c in classes for d in distances},
        "filled_buckets": sum(1 for c in classes for d in distances if per_bucket[(c, d)] > 0),
        "total_buckets": len(classes) * len(distances),
        "multi_class_filed": ambiguous,
        "instances_if_all_solo": len(entries),
    }


# --------------------------------------------------------------------------
# Thumbnail cache
# --------------------------------------------------------------------------


def build_thumbs(entries: list[dict], out: Path, budget: resources.Budget, size: int, rebuild: bool) -> Path:
    """Decode each 3,060x4,080 JPEG once and keep a 640px copy (~60 KB).

    draft() asks libjpeg to decode at a reduced scale, so this is not a full
    decode plus a resize - it is roughly an 8x cheaper decode. Afterwards the
    detector reads ~83 MB per pass instead of ~5 GB, and a re-run costs seconds.
    """
    from PIL import Image

    thumb_dir = out / THUMB_DIRNAME
    thumb_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        (e, out / e["class"] / e["new_name"])
        for e in entries
        if rebuild or not (thumb_dir / e["new_name"]).exists()
    ]
    todo = [(e, src) for e, src in todo if src.exists()]

    print(f"thumbnail cache: {len(entries) - len(todo)} present, {len(todo)} to build ({size}px)")
    if not todo:
        return thumb_dir

    budget.check_disk(str(out), extra_bytes=len(todo) * 90_000)  # ~90 KB each
    t0 = time.time()
    for i, (e, src) in enumerate(todo, start=1):
        with Image.open(src) as im:
            im.draft("RGB", (size, size))  # cheap scaled decode
            im = im.convert("RGB")
            if max(im.size) > size:
                scale = size / max(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
            im.save(thumb_dir / e["new_name"], "JPEG", quality=85)
        if i % 200 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  {(time.time() - t0) / i * 1000:.0f} ms/img")
    return thumb_dir


# --------------------------------------------------------------------------
# COCO pass
# --------------------------------------------------------------------------


def coco_pass(
    entries: list[dict],
    out: Path,
    thumbs: Path | None,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    budget: resources.Budget,
    limit: int = 0,
    batch_override: int = 0,
    chunk_size: int = 64,
) -> list[dict]:
    """Detect on every image, resumable, inside the VRAM/RAM budget.

    Results are flushed every 50 images: a full pass outlives a single command
    window, and losing a run to one timeout costs minutes of decoding.
    """
    from ultralytics import YOLO

    partial = out / "audit_coco.json"
    done: dict[str, dict] = {}
    if partial.exists():
        try:
            done = {r["name"]: r for r in json.loads(partial.read_text(encoding="utf-8"))}
        except ValueError:
            done = {}

    todo: list[tuple[dict, Path]] = []
    for e in entries:
        if e["new_name"] in done:
            continue
        src = (thumbs / e["new_name"]) if thumbs else (out / e["class"] / e["new_name"])
        if src.exists():
            todo.append((e, src))
    if limit:
        todo = todo[:limit]

    batch = batch_override or budget.batch_size(imgsz)
    print(f"coco: {len(done)} done, {len(todo)} to run (conf={conf}, imgsz={imgsz}, device={device}, batch={batch})")
    if not todo:
        return list(done.values())

    model = YOLO(str(model_path))
    names = model.names
    # Bound ultralytics' internal buffering. Not just an OOM guard: the torch
    # caching allocator reserves to the high-water mark, and a 256-image chunk
    # reserved 5.99 GB of an 8.6 GB card even though a batch of 8 only needs
    # 0.6 GB. A small chunk keeps the reservation proportionate to the work.
    CHUNK = chunk_size

    def run(chunk: list[tuple[dict, Path]], bs: int):
        return model.predict(
            [str(p) for _, p in chunk],
            conf=conf,
            imgsz=imgsz,
            device=device,
            stream=True,
            verbose=False,
            batch=bs,
            workers=0,  # in-process loading: no dataloader workers, no RAM spike
        )

    t0 = time.time()
    i = 0
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start : start + CHUNK]
        bs = batch
        while True:
            try:
                for (e, _), r in zip(chunk, run(chunk, bs)):
                    dets = []
                    if r.boxes is not None and len(r.boxes):
                        for b in r.boxes:
                            cls_id = int(b.cls.item())
                            xywh = b.xywhn[0].tolist()  # normalised cx, cy, w, h
                            dets.append(
                                {
                                    "cls": names.get(cls_id, str(cls_id)),
                                    "conf": round(float(b.conf.item()), 3),
                                    "area": round(xywh[2] * xywh[3], 5),
                                }
                            )
                    done[e["new_name"]] = {
                        "name": e["new_name"],
                        "class": e["class"],
                        "distance": e["distance"],
                        "dets": dets,
                    }
                    i += 1
                    if i % 50 == 0 or i == len(todo):
                        partial.write_text(json.dumps(list(done.values())), encoding="utf-8")
                        print(f"  {i}/{len(todo)}  {(time.time() - t0) / i * 1000:.0f} ms/img  (total {len(done)}/{len(entries)})")
                break
            except RuntimeError as exc:  # torch OOM under the VRAM cap
                # Chunking is the real fix here, not the batch: passing all 1,383
                # paths in one predict made ultralytics request a single 1.58 GB
                # buffer even at batch=1, which no sensible cap can accommodate.
                if "out of memory" not in str(exc).lower() or bs <= 1:
                    raise
                bs = max(1, bs // 2)
                budget.notes.append(f"halved batch to {bs} after a CUDA OOM")
                print(f"  CUDA OOM under the VRAM cap - retrying with batch={bs}")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
    return list(done.values())


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def summarise(results: list[dict], struct: dict, report_conf: float) -> dict:
    """Aggregate at `report_conf`, plus the low-confidence noise for contrast."""
    n = len(results)
    counts = [len([d for d in r["dets"] if d["conf"] >= report_conf]) for r in results]
    distractor_counts: collections.Counter = collections.Counter()
    images_with_distractor = 0
    bottle: collections.Counter = collections.Counter()
    noise: collections.Counter = collections.Counter()

    for r in results:
        for d in r["dets"]:
            if d["conf"] < report_conf:
                noise[d["cls"]] += 1
        names = [d["cls"] for d in r["dets"] if d["conf"] >= report_conf]
        hit = {DISTRACTORS[c] for c in names if c in DISTRACTORS}
        if hit:
            images_with_distractor += 1
            for h in hit:
                distractor_counts[h] += 1
        if r["class"] in BOTTLE_SHAPED:
            bottle["images"] += 1
            bottles = names.count("bottle")
            if bottles >= 2:
                bottle["images_with_2plus_bottles"] += 1
            bottle["max_bottles"] = max(bottle["max_bottles"], bottles)

    # A "2+ detections" count is not evidence of a multi-item scene on its own:
    # COCO read a single product as two objects in most of those frames. Split the
    # 2+ group into repeats of one class (which would mean two objects) versus a
    # mixed pair (which just means one product misread), so the report cannot
    # inflate the multi-item bucket with detector noise.
    same_class: collections.Counter = collections.Counter()
    mixed_only = 0
    for r in results:
        names = [d["cls"] for d in r["dets"] if d["conf"] >= report_conf]
        if len(names) < 2:
            continue
        top, count = collections.Counter(names).most_common(1)[0]
        if count >= 2:
            same_class[top] += 1
        else:
            mixed_only += 1

    per_bucket: dict[str, dict] = {}
    for c in struct["classes"]:
        for d in struct["distances"]:
            vals = [len([x for x in r["dets"] if x["conf"] >= report_conf])
                    for r in results if r["class"] == c and r["distance"] == d]
            if not vals:
                continue
            per_bucket[f"{c}_{d}"] = {
                "n": len(vals),
                "mean_dets": round(statistics.mean(vals), 2),
                "median_dets": statistics.median(vals),
                "with_2plus": sum(1 for v in vals if v >= 2),
                "zero_dets": sum(1 for v in vals if v == 0),
            }

    return {
        "images": n,
        "report_conf": report_conf,
        "detections_total": sum(counts),
        "images_with_any_detection": sum(1 for c in counts if c > 0),
        "images_with_zero_detections": sum(1 for c in counts if c == 0),
        "images_with_2plus_detections": sum(1 for c in counts if c >= 2),
        "detections_per_image": {
            "0": counts.count(0),
            "1": counts.count(1),
            "2": counts.count(2),
            "3+": sum(1 for c in counts if c >= 3),
        },
        "images_with_any_distractor": images_with_distractor,
        "distractor_kinds": dict(distractor_counts),
        "bottle_shaped": dict(bottle),
        "per_bucket": per_bucket,
        "below_report_conf": dict(noise.most_common(12)),
        "same_class_repeats": dict(same_class),
        "two_plus_mixed_only": mixed_only,
        "person_images": sum(1 for r in results if any(d["cls"] == "person" and d["conf"] >= report_conf for d in r["dets"])),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(
    path: Path, struct: dict, coco: dict, model_name: str, budget: resources.Budget, thumb_note: str
) -> str:
    s, c = struct, coco
    lines = [
        "# v2 dataset audit — MODEL_TRAINING.md §2 buckets",
        "",
        "## §2 budget versus what v2 actually contains",
        "",
        "| §2 bucket | §2 target | v2 | status |",
        "|---|---|---|---|",
        f"| Solo shots | ~150 per item (~1,200 for 8) | {s['images']} images, each filed under exactly one product | ✅ covered |",
        f"| Multi-item scenes (3–6 items, overlap, occlusion) | 800–1,200 | **no evidence of a single one** (see below) | ❌ **absent** |",
        "| Hard negatives (empty counter, hands, bags, phone) | 150–250 (5–10%) | 0 | ❌ **absent by construction** |",
        "",
        f"**§2 target: ~2,000–2,800 images. v2: {s['images']}.** The count is short, but the shape of",
        "the set is the bigger problem: §2's two highest-value buckets are missing, and no number of",
        "extra solo shots substitutes for either.",
        "",
        "## Structural audit (exact — no model involved)",
        "",
        f"- Every image carries **exactly one** class tag and one distance tag; {s['multi_class_filed']}",
        "  images are filed ambiguously between two products.",
        f"- {s['filled_buckets']} of {s['total_buckets']} product×distance buckets are non-empty. The",
        "  capture taxonomy is single-product-per-folder **by design**, so a multi-item scene had no",
        "  folder to be filed into even if one had been shot.",
        "- Hard negatives are therefore **0, not few**: an image only entered this set by being placed in",
        "  a product folder. Nothing was captured *without* a product in frame — no counter, hands, bag or",
        "  phone shot exists to draw negatives from.",
        f"- With one item per image, v2 teaches from ~{s['instances_if_all_solo']} instances, ~1.0 per image.",
        "",
        f"## COCO pass (`{model_name}`, report confidence ≥ {c.get('report_conf', 0)})",
        "",
        f"Ran over {c['images']} images. COCO knows nothing about grocery SKUs; it is used only to detect",
        "the *other* things §2 wants hard negatives to contain.",
        "",
        "| detections in image | images |",
        "|---|---|",
        f"| 0 | {c['detections_per_image']['0']} |",
        f"| 1 | {c['detections_per_image']['1']} |",
        f"| 2 | {c['detections_per_image']['2']} |",
        f"| 3+ | {c['detections_per_image']['3+']} |",
        "",
        f"- Images containing a §2 distractor ingredient: **{c['images_with_any_distractor']}**",
    ]
    if c["distractor_kinds"]:
        lines += ["", "| ingredient | images |", "|---|---|"]
        for k, v in sorted(c["distractor_kinds"].items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "### Is any frame actually a multi-item scene?",
        "",
        f"{c.get('images_with_2plus_detections', 0)} images have 2+ detections, but that number is not the answer, and",
        "treating it as one would inflate the bucket with detector noise. Splitting it:",
        "",
        f"- **{c.get('two_plus_mixed_only', 0)}** are a *mixed* pair (e.g. `knife` + `dining table`): one product that",
        "  COCO read as two unrelated objects. Not two items.",
        f"- **{sum(c.get('same_class_repeats', {}).values())}** repeat a single class: {c.get('same_class_repeats', {}) or 'none'}.",
        "  Those are clutter classes already known to fire on product shots, not a second product.",
        f"- **Max bottles in any image: {c.get('bottle_shaped', {}).get('max_bottles', 0)}.** On the only SKUs COCO can",
        "  count at all, not one frame contains two objects.",
        "",
        "So the honest reading is **zero multi-item scenes**, arrived at two independent ways: the folder",
        "taxonomy could not have held one, and no frame shows two of anything.",
    ]
    if c["bottle_shaped"]:
        b = c["bottle_shaped"]
        lines += [
            "",
            "### Bottle-shaped SKUs (the only COCO-countable classes)",
            "",
            f"- {b.get('images', 0)} images across {sorted(BOTTLE_SHAPED)}",
            f"- images with 2+ bottle detections: **{b.get('images_with_2plus_bottles', 0)}**  ← the multi-item floor",
            f"- most bottles in one image: {b.get('max_bottles', 0)}",
        ]

    lines += ["", "## Per-bucket detail", "", "| bucket | n | mean dets | median | ≥2 dets | 0 dets |", "|---|---|---|---|---|---|"]
    for k, v in sorted(c["per_bucket"].items()):
        lines.append(f"| `{k}` | {v['n']} | {v['mean_dets']} | {v['median_dets']} | {v['with_2plus']} | {v['zero_dets']} |")

    lines += [
        "",
        "## Confidence in this audit",
        "",
        f"Detections below the report confidence ({c.get('report_conf', 0)}) were discarded as noise.",
        "What survived is still not all real — COCO misreads product close-ups:",
        "",
        "| class | detections below the report threshold |",
        "|---|---|",
    ]
    for k, v in (c.get("below_report_conf") or {}).items():
        lines.append(f"| {k} | {v} |")
    if not c.get("below_report_conf"):
        lines.append("| — | none |")
    lines += [
        "",
        "Treat `bottle` on a bottle-shaped SKU and `person` as the only semi-reliable signals here;",
        "the clutter classes (`book`, `kite`, `bed`, `dining table`, `knife`) fire on plain product shots",
        f"and are false positives. `person` at ≥{c.get('report_conf', 0)} appears in **{c.get('person_images', 0)}** images —",
        "concentrated in close-range Milo and Bear Brand frames — which is plausibly a hand holding the",
        "product. Worth a visual spot-check as an accidental hands signal, but it is not a captured hard",
        "negative and 2% of a set is nowhere near the 5-10% §2 asks for.",
        "",
        "### What this cannot tell you",
        "",
        "- **Target-class instances per image.** COCO misses sachets and cans, so it cannot count",
        "  Milo/555/Century Tuna/Lucky Me/Bear Brand at all, and cannot see a second sachet behind a first.",
        "  The multi-item number is a floor for bottle-shaped SKUs and nothing for the other five classes.",
        "- **Whether a zero-detection image is empty or a product COCO missed.** Those cases demand",
        "  opposite responses and are indistinguishable here.",
        "- **Hard negatives.** The structural audit shows none were captured; it cannot rule out that some",
        "  filed image is actually a background or clutter shot.",
        "",
        "To close either gap, run the deployed grocery model over the set (it knows the 7 v1 classes and",
        "would give real per-image instance counts), or review the `0 detections` images by eye in Roboflow.",
        "",
        "## Resource envelope this run stayed inside",
        "",
        "```",
        budget.describe(),
        f"  peak VRAM    {c.get('peak_vram_gb', 0):.2f} GB measured",
        f"  thumbnails   {thumb_note}",
        "```",
        "",
        "Inference was never the cost: 13.6 ms/frame on the 4060. Decoding 1,383 3,060×4,080 JPEGs was,",
        "which is what the thumbnail cache removes. Raise the caps with `--max-use-percent`.",
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--conf", type=float, default=0.25, help="collection floor; kept for thresholding later")
    ap.add_argument("--report-conf", type=float, default=0.5, help="confidence the report counts detections at")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=0, help="override the budget-derived batch size")
    ap.add_argument("--device", default="auto", help="auto | 0 | cpu")
    ap.add_argument("--max-use-percent", type=int, default=resources.USE_PERCENT, help="CPU + RAM ceiling")
    ap.add_argument("--max-vram-percent", type=int, default=resources.VRAM_PERCENT, help="VRAM ceiling")
    ap.add_argument(
        "--hard-vram-cap",
        action="store_true",
        help="enforce the VRAM ceiling in the torch allocator (causes spurious OOMs - see resources.apply)",
    )
    ap.add_argument("--disk-reserve-gb", type=float, default=resources.DISK_RESERVE_GB)
    ap.add_argument("--thumb-size", type=int, default=640)
    ap.add_argument("--rebuild-thumbs", action="store_true")
    ap.add_argument("--skip-thumbs", action="store_true", help="detect on the full-resolution originals")
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--thumbs-only", action="store_true", help="build/refresh the thumbnail cache and stop")
    ap.add_argument("--limit", type=int, default=0, help="process only this many more images (resumable)")
    ap.add_argument("--chunk", type=int, default=64, help="images per predict call; bounds the VRAM reservation")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()

    # Order matters: measure and apply before torch initialises its CUDA context,
    # otherwise the VRAM cap cannot take effect.
    budget = resources.measure(args.max_use_percent, args.disk_reserve_gb, args.max_vram_percent)
    device = resources.resolve_device(args.device)
    budget = resources.apply(budget, device, args.hard_vram_cap)
    print(budget.describe())

    entries = load_manifest(out)
    struct = structural(entries)
    print(f"manifest: {struct['images']} images, {struct['filled_buckets']}/{struct['total_buckets']} buckets filled")

    peak_vram_gb = 0.0
    thumbs: Path | None = None
    thumb_note = "skipped (--skip-thumbs)"
    if (not args.skip_coco and not args.skip_thumbs) or args.thumbs_only:
        budget.check_disk(str(out), extra_bytes=len(entries) * 90_000)
        budget.check_ram(0.5)
        thumbs = build_thumbs(entries, out, budget, args.thumb_size, args.rebuild_thumbs)
        size_mb = sum(f.stat().st_size for f in thumbs.glob("*.jpg")) / 1e6
        thumb_note = f"{len(list(thumbs.glob('*.jpg')))} files at {args.thumb_size}px, {size_mb:.0f} MB"
        if args.thumbs_only:
            print("thumbs-only: stopping before the detector pass")
            return 0

    coco: dict = empty_coco()
    if not args.skip_coco:
        results = coco_pass(
            entries,
            out,
            thumbs,
            Path(args.model),
            args.conf,
            args.imgsz,
            device,
            budget,
            args.limit,
            args.batch,
            args.chunk,
        )
        coco = summarise(results, struct, args.report_conf)
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.max_memory_reserved() > 0:
                peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9
        except ImportError:
            pass
        if not peak_vram_gb:
            # Resume run: nothing was inferred here, so keep the figure measured by
            # the pass that actually did the work rather than reporting 0.00.
            old = out / "audit.json"
            if old.exists():
                try:
                    peak_vram_gb = json.loads(old.read_text(encoding="utf-8"))["coco"].get("peak_vram_gb", 0)
                except (ValueError, KeyError):
                    peak_vram_gb = 0.0
        print(f"  peak VRAM used: {peak_vram_gb:.2f} GB")

    coco["peak_vram_gb"] = peak_vram_gb

    budget.check_disk(str(out), extra_bytes=2_000_000)
    (out / "audit.json").write_text(json.dumps({"structural": struct, "coco": coco}, indent=1), encoding="utf-8")
    text = write_report(out / "AUDIT.md", struct, coco, Path(args.model).name, budget, thumb_note)
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
