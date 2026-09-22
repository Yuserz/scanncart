#!/usr/bin/env python
"""Pin the v1 class names onto the v2 images before labeling starts.

WHY THIS EXISTS
---------------
MODEL_TRAINING.md §8.1 says v2 must reuse v1's class names *verbatim*, so the two
sets stay mergeable and comparable. v1's names are inconsistently formatted -
`555 sardines 155grams` and `century_tuna_flakes_in_oil_155_grams` and
`silver_swan_sukang_puti_200ML` are three different conventions for the same idea -
so a human applying them to 1,383 images will drift. This tool removes the drift.

WHAT IT CANNOT DO (verified, not assumed)
-----------------------------------------
1. It cannot rename the upload tag to the class name. The tag API rejects any tag
   containing a space:

       {"error": {"message": "Invalid tag",
                  "hint": "Valid characters: a-z A-Z 0-9 -_:/.[]<>{}@, length 1-64",
                  "invalid": "Bear Brand Fortified Powdered Milk 33g"}}

   Three of the seven v1 names contain spaces, so a tag cannot carry them. Tags
   keep the short slug (`bear-brand-milk`), which is also what makes the
   `tag:` search filters and the batch keys work.

2. It does NOT rename a class. Once a class exists it is renamed or deleted on the
   project's Settings -> Classes tab, which has no API (Roboflow docs, "Set Dataset
   Classes") - and renaming rewrites every annotation that used it, so it is a
   deliberate, manual, one-way action.

   It CAN, however, create classes: `--create-classes`. That route is not documented
   as such, and the obvious probe says it does not work - annotating an image and
   re-reading the project 55 seconds later still showed `classes: {}`. It works, but
   the class list refreshes much later than that. Verified end to end: uploading a
   throwaway image annotated with `Bear Brand Fortified Powdered Milk 33g` produced
   that class in the project, and it survived deleting the image (`{"Bear Brand
   Fortified Powdered Milk 33g": 0}` - a zero-image class, which v1 also carries).
   So `--create-classes` seeds each class from a temporary annotated image, deletes
   the image, and then polls, because the refresh is slow and not immediate.

WHAT IT DOES
------------
* `class_name` metadata on every image: the exact v1 name. Metadata has no
  character restriction (verified: the spaced name round-trips intact), so this is
  where the authoritative name lives. It travels with the data - any later export,
  audit or script can map an image to its class with no slug lookup table.
* A validated mapping, checked against the live v1 project rather than trusted:
  if a name in this file does not exist in `scanncart-grocery`, that is a
  continuity break and the tool says so.
* The class list to paste into Settings -> Classes, in a fixed order.

    sidecar/.venv/Scripts/python.exe sidecar/tools/label_classes.py            # report
    sidecar/.venv/Scripts/python.exe sidecar/tools/label_classes.py --apply    # write metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import httpx

from workspace import DEFAULT_OUT, ENV_PATH  # workspace lives outside the repo tree

# The Roboflow workspace slug. Not to be confused with the dataset workspace in
# workspace.py - this one is a Roboflow account namespace, that one is a directory.
WORKSPACE = "yusri-caloyloy"
V1_PROJECT = "scanncart-grocery"

# slug (what is on the images) -> exact class name (what labels must use).
# Row order is deliberate: it is the order to create the classes in, which becomes
# the order the class selector lists them, which is what makes keyboard labelling
# predictable. Keep it stable.
#
# Rows 1-7 are v1's names verbatim, so v2 stays mergeable and comparable with the
# 1,815 images already labeled in `scanncart-grocery`. Row 8 is the one class v2
# adds - Palmolive - which had no v1 name to inherit. It is written in v1's own
# style (`<brand>_<product>_<size>`, cf. `safeguard_pure_white_60g`) so the roster
# reads as one list rather than seven inherited names plus an odd one out.
SLUG_TO_CLASS: dict[str, str] = {
    "bear-brand-milk": "Bear Brand Fortified Powdered Milk 33g",
    "lucky-me-pancit": "lucky_me_pancit_canton_calamansi_flavor",
    "555-sardines": "555 sardines 155grams",
    "century-tuna": "century_tuna_flakes_in_oil_155_grams",
    "silver-swan-vinegar": "silver_swan_sukang_puti_200ML",
    "milo": "Milo Chocolate Drink 22g Sachet",
    "safeguard": "safeguard_pure_white_60g",
    "palmolive": "Palmolive Naturals Bar Soap 85g",
}

# Slugs for classes that are *declared but not yet named*. Empty, and deliberately
# kept: Palmolive was the last one and now has a real name above, but the next new
# class needs the same treatment. A placeholder is fine as a *label* here because it
# is only ever printed at the operator - it is never written to an image, since
# pinning a name that is about to change is worse than carrying the slug.
NEW_CLASS_SLUGS: dict[str, str] = {}

# Slugs that are batch and tag keys rather than labelable classes. `negative` is
# §2's hard-negative pseudo-class (see clean_v2.CLASS_MAP): those frames carry no
# annotation by definition, so there is no class name to resolve for them and
# reporting them as unmapped would be a false alarm the moment any are staged.
PSEUDO_CLASS_SLUGS = {"negative"}

# Slugs whose class is new in v2, so v1 mutual continuity is not a property to check -
# v1 has no Palmolive. Without this the continuity check reports the one class that is
# supposed to be missing from v1 as a continuity break, which would train the operator
# to ignore the check that guards the other seven.
V2_ONLY_SLUGS = {"palmolive"}

# The distance words, as they appear in a class *name* when something has gone wrong.
#
# Distance is a **tag** on the image and a **cell** in the reports (MODEL_TRAINING.md 8.1, 8.3) -
# never a category. A class named `palmolive close` splits one product into three classes, and the
# trained head comes back with 24 outputs instead of 8: every box is then labelled something the
# app's own roster does not contain, and nothing errors, because a class *name* records none of
# this. That is the whole reason this is a check rather than a convention.
#
# Matched on whole tokens, so a legitimate name cannot trip it: "Farmer's Choice" tokenises to
# `farmer`, which is not `far`. A drift guard in tests/test_dataset_tools.py keeps this covering
# every spelling `clean_v2.DISTANCE_MAP` accepts, so a new folder spelling cannot slip past.
DISTANCE_TOKENS = frozenset({"close", "closeup", "mid", "middle", "far", "near", "distance"})


def distance_tokens_in(name: str) -> list[str]:
    """The distance words a class name carries, if any. Pure, so it can be tested directly.

    Shared by the two places that can see a wrong class list: this file's check of the live
    project, and `train_model.check_export`, which reads the list a *version* was generated with.
    Both need to say the same thing about the same name.
    """
    tokens = re.split(r"[^a-z0-9]+", str(name).lower())
    return sorted({t for t in tokens if t in DISTANCE_TOKENS})


def load_key(project: str) -> str:
    key = __import__("os").environ.get("ROBOFLOW_API_KEY", "")
    if not key and ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("ROBOFLOW_API_KEY=") and not key:
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit(f"no Roboflow API key (set ROBOFLOW_API_KEY or put it in {ENV_PATH})")
    return key


def get_project(client: httpx.Client, key: str, project: str) -> dict:
    r = client.get(f"https://api.roboflow.com/{WORKSPACE}/{project}", params={"api_key": key})
    if r.status_code != 200:
        raise SystemExit(f"{project} not reachable: HTTP {r.status_code} {r.text[:200]}")
    body = r.json()
    if "error" in body:
        raise SystemExit(f"{project}: {body['error']}")
    return body.get("project", body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--project", default="snc-grocery")
    ap.add_argument("--apply", action="store_true", help="write class_name metadata")
    ap.add_argument(
        "--create-classes",
        action="store_true",
        help="seed any missing class via a throwaway annotated image, then delete it",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} - run clean_v2.py clean first")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    by_name = {e["new_name"]: e for e in entries}
    key = load_key(args.project)

    problems: list[str] = []

    with httpx.Client(timeout=120) as client:
        # ---- 1. is the mapping actually still v1's roster? ----
        v1 = get_project(client, key, V1_PROJECT)
        live_v1_names = set(v1.get("classes") or {})
        inherited = {s: n for s, n in SLUG_TO_CLASS.items() if s not in V2_ONLY_SLUGS}
        new_to_v2 = sorted(n for s, n in SLUG_TO_CLASS.items() if s in V2_ONLY_SLUGS)
        missing_from_v1 = [n for n in inherited.values() if n not in live_v1_names]
        print(f"v1 reference ({V1_PROJECT}): {len(live_v1_names)} classes")
        if missing_from_v1:
            problems.append(
                "these names are not in the v1 project, so reusing them does not preserve continuity: "
                + ", ".join(missing_from_v1)
            )
        else:
            print(f"  all {len(inherited)} inherited names exist in v1 - labels will stay continuous")
        if new_to_v2:
            print(
                "  new to v2, so v1 continuity does not apply: " + ", ".join(new_to_v2)
            )

        # ---- 2. does every image carry a slug we can resolve? ----
        slugs = Counter(e["class"] for e in entries)
        unmapped = sorted(
            s
            for s in slugs
            if s not in SLUG_TO_CLASS and s not in NEW_CLASS_SLUGS and s not in PSEUDO_CLASS_SLUGS
        )
        if unmapped:
            problems.append(f"images tagged with an unmapped class slug: {', '.join(unmapped)}")
        for slug in sorted(s for s in slugs if s in NEW_CLASS_SLUGS):
            print(
                f"  note: {slugs[slug]} image(s) tagged `{slug}` are the new v2 class - "
                f"name it in v1's style ({NEW_CLASS_SLUGS[slug]}) before creating the class list"
            )

        # ---- 3. create any missing classes, then assess what the project has ----
        # Creation runs BEFORE the assessment on purpose: judging the class list first
        # and then creating them would leave a stale "missing class" problem behind
        # and report failure on a run that succeeded.
        want = set(SLUG_TO_CLASS.values())
        have = set(get_project(client, key, args.project).get("classes") or {})
        print(f"\nv2 project ({args.project}): {len(have)} classes defined")

        # ---- 3a. is any class a distance in disguise? ----
        # Checked before the create/assess step because it changes what to *do*: a project with
        # `palmolive close` in it has to have that class deleted and its annotations moved, not
        # another class added. Left to the extra-classes check in train_model this surfaces only
        # after a version has been generated - i.e. after the mistake is expensive.
        tainted = {name: distance_tokens_in(name) for name in sorted(have)}
        tainted = {name: words for name, words in tainted.items() if words}
        if tainted:
            for name, words in tainted.items():
                print(f"  distance in a class name: {name!r} ({', '.join(words)})")
            problems.append(
                "the class list has distances baked into it ("
                + ", ".join(f"{name!r}" for name in tainted)
                + ") - distance is a tag, not a class: fix the class list, move any annotations "
                "onto the product class, and regenerate the version, or the trained head has "
                "one output per product-and-distance instead of one per product"
            )
        else:
            print(f"  no class name carries a distance ({len(DISTANCE_TOKENS)} word(s) checked)")

        if args.create_classes:
            absent = [n for n in SLUG_TO_CLASS.values() if n not in have]
            if args.dry_run:
                print(f"--dry-run: would create {len(absent)} class(es): " + ", ".join(absent) if absent else "--dry-run: nothing to create")
            elif not absent:
                print("all v1 class names already exist - nothing to create")
            else:
                print(f"creating {len(absent)} missing class(es) via throwaway annotated images...")
                create_classes(client, key, args.project, absent)
                have = _poll_classes(client, key, args.project, want)
                still = sorted(want - have)
                if still:
                    print(f"  not visible yet after polling: {', '.join(still)}")
                    print("  the class list refreshes slowly - re-run without --create-classes to confirm")
                else:
                    print("  all mapped classes now present")

        absent = sorted(want - have)
        if not have:
            problems.append(
                "no classes defined in the v2 project - run with --create-classes, then turn on "
                "Lock Classes on Settings -> Classes BEFORE labeling"
            )
        elif absent:
            problems.append("missing from the v2 class list: " + ", ".join(absent))

        # ---- 4. the class list to paste in ----
        print("\nclass list for Settings -> Classes, in this order:")
        print("-" * 66)
        for i, (_slug, name) in enumerate(SLUG_TO_CLASS.items(), start=1):
            print(f"{i:2}. {name}")
        # Declared-but-unnamed classes come last, flagged so nobody creates a
        # placeholder as if it were a name. No-op while NEW_CLASS_SLUGS is empty.
        for i, (_slug, placeholder) in enumerate(NEW_CLASS_SLUGS.items(), start=len(SLUG_TO_CLASS) + 1):
            print(f"{i:2}. {placeholder}   <- naming still open, do not create yet")
        print("-" * 66)
        print("order matters: it is the order the class selector lists them, which is what")
        print("makes keyboard labelling predictable.")

        # ---- 5. write class_name metadata ----
        if args.apply and not args.dry_run:
            index = _image_ids(client, key, args.project)
            updates = []
            for name, e in by_name.items():
                cls = SLUG_TO_CLASS.get(e["class"])
                if not cls or name not in index:
                    continue
                updates.append(
                    {"imageId": index[name], "metadata": {"class_name": cls, "class_slug": e["class"]}}
                )
            print(f"\nwriting class_name metadata to {len(updates)} images...")
            errors = _push_metadata(client, key, updates)
            for err in errors[:10]:
                print(f"  ERROR: {err}")
            if errors:
                problems.append(f"{len(errors)} metadata update(s) failed")
            else:
                sample = updates[0]["imageId"]
                got = client.get(
                    f"https://api.roboflow.com/{WORKSPACE}/{args.project}/images/{sample}",
                    params={"api_key": key},
                ).json().get("image", {})
                print(f"  read back one image -> metadata.class_name = {got.get('metadata', {}).get('class_name')!r}")
        elif args.apply:
            print("\n--dry-run: metadata not written")

    # ---- 6. coverage summary, so labeling can be planned ----
    print("\nimages per class in v2 (slug -> class name):")
    for slug, cls in SLUG_TO_CLASS.items():
        print(f"  {slug:22} {slugs.get(slug, 0):5}   {cls}")

    print()
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        return 1
    print("class naming checks out")
    return 0


# A tiny flat JPEG is enough: the class is registered from the annotation, not the
# pixels, and the image is deleted immediately afterwards so it never reaches a
# version.
#
# The bytes MUST differ per class. Roboflow deduplicates at the workspace level by
# SHA-256, so identical seed images collapse into one asset - and then every
# annotation after the first fails with 409 "Image was already annotated", which is
# exactly what happened on the first attempt: only one of six classes got created.
#
# The colour MUST be derived from the class NAME, never from the loop index. An
# index-based seed repeats across runs (run 1's seed 1 is run 2's seed 1), so adding a
# single new class to a project that already seeded one uploaded bytes the workspace
# had seen, got the older - already annotated - asset back from its dedup, and 409'd.
# That is the common case: the first run creates six classes and only later does
# someone add the seventh. The nonce lets a retry past a stale asset of the same name.
def _seed_jpeg(cls: str, nonce: int = 0) -> bytes:
    try:
        import io

        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is required to seed classes - run this with the sidecar venv python")
    digest = hashlib.sha256(f"{cls}:{nonce}".encode("utf-8")).digest()
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (digest[0], digest[1], digest[2])).save(buf, "JPEG", quality=60)
    return buf.getvalue()


def _purge_seed_batch(client: httpx.Client, key: str, project: str) -> int:
    """Delete any leftover seed images. A failed annotate must not leave junk in the
    project - the first buggy run left five behind, all counted as unannotated."""
    ids: list[str] = []
    offset = 0
    while True:
        body = client.post(
            f"https://api.roboflow.com/{WORKSPACE}/{project}/search",
            params={"api_key": key},
            json={"limit": 500, "offset": offset, "fields": ["name"]},
        ).json()
        results = body.get("results") or []
        ids += [r["id"] for r in results if r["name"].startswith("__class_seed_")]
        offset += len(results)
        if not results or offset >= body.get("total", 0):
            break
    if not ids:
        return 0
    client.request(
        "DELETE",
        f"https://api.roboflow.com/{WORKSPACE}/{project}/images",
        params={"api_key": key},
        headers={"Content-Type": "application/json"},
        content=json.dumps({"images": ids}),
    )
    return len(ids)


def _voc_xml(name: str, cls: str, w: int = 640, h: int = 480) -> str:
    return (
        f"<annotation><folder></folder><filename>{name}</filename><path>{name}</path>"
        f"<source><database>roboflow.com</database></source>"
        f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"
        f"<segmented>0</segmented>"
        f"<object><name>{cls}</name><pose>Unspecified</pose><truncated>0</truncated>"
        f"<difficult>0</difficult><occluded>0</occluded>"
        f"<bndbox><xmin>1</xmin><xmax>{w - 1}</xmax><ymin>1</ymin><ymax>{h - 1}</ymax></bndbox>"
        f"</object></annotation>"
    )


def create_classes(client: httpx.Client, key: str, project: str, names: list[str]) -> None:
    """Register each class by annotating a throwaway image, then delete the image.

    'Annotating' means: upload a byte-unique seed image, POST a VOC XML naming the
    class to /dataset/<project>/annotate/<image_id>, then DELETE the image. The class
    survives the delete - that is the whole point, and it is verified rather than
    assumed.
    """
    stale = _purge_seed_batch(client, key, project)
    if stale:
        print(f"  cleared {stale} leftover seed image(s) from an earlier run")

    for i, cls in enumerate(names, start=1):
        name = f"__class_seed_{i}__.jpg"
        # Two attempts: a 409 means the workspace dedup handed back an asset that
        # already carries an annotation, so a fresh nonce is needed. Retrying here
        # rather than aborting keeps `--create-classes` self-healing when an earlier
        # run's seed is still in the workspace (a re-run is the normal case).
        for nonce in (0, 1):
            up = client.post(
                f"https://api.roboflow.com/dataset/{project}/upload",
                params={"api_key": key, "name": name, "batch": "__class_seed__"},
                files={"file": (name, _seed_jpeg(cls, nonce), "image/jpeg")},
            )
            if up.status_code != 200 or not up.json().get("id"):
                print(f"  [{i}/{len(names)}] {cls}: upload failed HTTP {up.status_code} {up.text[:120]}")
                break
            image_id = up.json()["id"]
            ann = client.post(
                f"https://api.roboflow.com/dataset/{project}/annotate/{image_id}",
                params={"api_key": key},
                content=_voc_xml(name, cls).encode("utf-8"),
                headers={"Content-Type": "text/xml"},
            )
            if ann.status_code == 409:
                if nonce == 0:
                    print(f"  [{i}/{len(names)}] {cls}: seed asset was already annotated - retrying with a fresh seed")
                    continue
                print(
                    f"  [{i}/{len(names)}] {cls}: collided twice - create this one by hand on "
                    "Settings -> Classes"
                )
                break
            if ann.status_code != 200:
                print(f"  [{i}/{len(names)}] {cls}: annotate failed HTTP {ann.status_code} {ann.text[:120]}")
                break
            client.request(
                "DELETE",
                f"https://api.roboflow.com/{WORKSPACE}/{project}/images",
                params={"api_key": key},
                headers={"Content-Type": "application/json"},
                content=json.dumps({"images": [image_id]}),
            )
            print(f"  [{i}/{len(names)}] seeded {cls}")
            break

    left = _purge_seed_batch(client, key, project)
    if left:
        print(f"  cleaned up {left} seed image(s) that failed to annotate")


def _poll_classes(client: httpx.Client, key: str, project: str, want: set[str], seconds: int = 240) -> set[str]:
    """The class list is not updated synchronously with the annotation - it was still
    empty 55 s after an accepted annotation. Poll rather than reporting a false gap."""
    import time

    deadline = time.time() + seconds
    have: set[str] = set()
    while time.time() < deadline:
        have = set(get_project(client, key, project).get("classes") or {})
        if want <= have:
            return have
        time.sleep(15)
    return have


def _image_ids(client: httpx.Client, key: str, project: str) -> dict[str, str]:
    """name -> image id, converged. The search route drops records from large pages
    while its index catches up (1,297 / 1,380 / 1,383 on three consecutive sweeps),
    so a single pass cannot be trusted to have seen every image."""
    import time

    merged: dict[str, str] = {}
    total = 0
    for attempt in range(6):
        offset = 0
        while True:
            body = client.post(
                f"https://api.roboflow.com/{WORKSPACE}/{project}/search",
                params={"api_key": key},
                json={"limit": 500, "offset": offset, "fields": ["name"]},
            ).json()
            results = body.get("results")
            if results is None:
                raise SystemExit(f"search failed: {str(body)[:200]}")
            total = body.get("total", total)
            for rec in results:
                merged[rec["name"]] = rec["id"]
            offset += len(results)
            if not results or offset >= total:
                break
        if total and len(merged) >= total:
            break
        if attempt < 5:
            time.sleep(3)
    if total and len(merged) < total:
        print(f"  WARNING: indexed {len(merged)} of {total} images after 6 sweeps")
    return merged


def _push_metadata(client: httpx.Client, key: str, updates: list[dict]) -> list:
    """Batch endpoint is async (202 + task url) - the single endpoint is not, and
    1,383 x 2 calls is worse than polling."""
    import time

    errors: list = []
    for i in range(0, len(updates), 1000):
        chunk = updates[i : i + 1000]
        resp = client.post(
            f"https://api.roboflow.com/{WORKSPACE}/images/metadata",
            params={"api_key": key},
            json={"updates": chunk},
        )
        if resp.status_code not in (200, 202):
            errors.append(f"chunk {i // 1000 + 1}: HTTP {resp.status_code} {resp.text[:200]}")
            continue
        print(f"  chunk {i // 1000 + 1}: {len(chunk)} images queued")
        if resp.status_code == 202:
            url = resp.json().get("url")
            deadline = time.time() + 300
            while url and time.time() < deadline:
                body = client.get(url, params={"api_key": key}).json()
                if str(body.get("status", "")).lower() in ("done", "completed", "complete", "failed", "error"):
                    if str(body.get("status", "")).lower() in ("failed", "error"):
                        errors.append(f"chunk {i // 1000 + 1}: {str(body)[:200]}")
                    break
                time.sleep(3)
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
