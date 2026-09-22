#!/usr/bin/env python
"""Generate the Roboflow dataset version non-interactively, then verify what it got.

Why this is a script and not a few clicks: a version's preprocessing is a **one-shot,
irreversible** decision. It bakes the geometry every training image is stored at, it
cannot be edited afterwards (a new set of settings means a new version number), and it is
the single biggest silent accuracy lever in the whole pipeline. `sanity` used to report
that there was no API for it; there is - `POST /{workspace}/{project}/generate` - so the
settings live here, in code, reviewable and re-runnable, instead of in someone's memory.

The geometry, and why it is `Stretch to`:

| | x scale (1280 -> 640) | y scale (720 -> 640) | aspect |
|---|---|---|---|
| `Stretch to` | 0.500 | **0.889** | distorted |
| `Fit within` (letterbox) | 0.500 | 0.500 | preserved |

Both downscale the horizontal axis by the same 0.5, because 1280 is the binding dimension
- but stretch keeps 89% of the vertical scale where letterboxing throws 44% of the canvas
away as padding. Every object therefore occupies ~1.8x more vertical pixels under stretch
at no horizontal cost. The frames where that matters are the `far` ones, which is the axis
this dataset exists to add (see MODEL_TRAINING.md 8.3), and it is how v1's shipped model
was trained (`yusri-caloyloy/scanncart-grocery/1`: `auto-orient` + `Stretch to 640`, mAP
98.21).

The thing that has to travel with the geometry: a model trained on this version has to be
*run* stretched, and `resolve_resize_mode`'s format heuristic answers letterbox for a local
`.pt` - so without the record it sees everything at the letterboxed scale, the exact mismatch
the numbers above are about. That is why `REQUIRED_RESIZE_MODE` below is derived from
`PREPROCESSING` rather than retyped, and why `train_v2.py --install` writes it beside the
weights: `resize_mode: auto` then honours it (`app.models.requirement_for`). `--verify`
prints the reminder with the version it checked, and the Admin Panel's model entry carries it
too.

    # what would be sent, without generating anything
    sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --dry-run

    # generate (consumes a version number - only once the annotation is done)
    sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --yes

    # read a version back and compare its settings against this file
    sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --verify 2
"""

from __future__ import annotations

import argparse
import json

import httpx

from label_classes import WORKSPACE, distance_tokens_in, load_key

API = "https://api.roboflow.com"


# The single source of truth. Everything downstream reads this - the tool, the guard it
# prints, and the tests - so the version that gets generated is the one described here.
PREPROCESSING: dict = {
    "auto-orient": True,
    "resize": {"width": 640, "height": 640, "format": "Stretch to"},
}
# Empty on purpose and checked as such: augmentation applies to the train split only, and
# faking volume with it is the trap MODEL_TRAINING.md 4 names. Report real counts instead.
AUGMENTATION: dict = {}

# 640 must match settings.imgsz, which is what the sidecar infers at. A version generated
# at another size trains a model at a scale inference never uses.
EXPECTED_SIZE = 640
# The same three facts as a tuple, because `sanity` compares an existing version against
# them and imports this rather than keeping a second copy that could drift from what this
# tool actually sends.
EXPECTED_RESIZE = (
    PREPROCESSING["resize"]["width"],
    PREPROCESSING["resize"]["height"],
    PREPROCESSING["resize"]["format"],
)

# Roboflow's resize formats translated into the sidecar's `resize_mode` vocabulary
# (`ALLOWED_RESIZE_MODES`). Two entries, and only the two that mean the same thing on both
# sides: Roboflow's "Fill within" scales and *crops*, which the sidecar cannot do, so it is
# deliberately absent rather than mapped to the nearest-sounding value.
RESIZE_MODE_BY_FORMAT = {"Stretch to": "stretch", "Fit within": "letterbox"}
# What a model trained on this version's export has to be run with - the requirement that
# travels with the weights and that the Admin Panel checks the setting against. Derived, not
# retyped, so it cannot disagree with the geometry above; it is `None` if the format stops
# being one of the two, which is how a future geometry surfaces instead of being guessed at.
REQUIRED_RESIZE_MODE: str | None = RESIZE_MODE_BY_FORMAT.get(PREPROCESSING["resize"]["format"])

# Preprocessing steps that must NOT be present, and why each one is a defect rather than a
# preference. `filter-null` is the dangerous one: it drops null-annotated images, and the
# null annotations are the entire hard-negative set (Tier C2b) - set it to 50% and half the
# background frames the model was supposed to learn from quietly leave the version.
FORBIDDEN_STEPS: dict[str, str] = {
    "filter-null": (
        "drops null-annotated images, which are the hard negatives (Tier C2b) - they are "
        "annotated deliberately and must enter the version"
    ),
    "filter-by-tag": "would silently narrow the set to whatever the tag filter says",
    "isolate": "crops each box into a classification dataset - wrong project type",
    "tile": "changes what one training example is; not wanted for counter-scale frames",
    "static-crop": "crops a fixed region of every frame, discarding the distances' framing",
}


def version_settings() -> dict:
    """The exact request body. Deep-copied so a caller cannot mutate the constants."""
    return json.loads(json.dumps({"preprocessing": PREPROCESSING, "augmentation": AUGMENTATION}))


def compare_preprocessing(actual: object) -> list[str]:
    """Differences between what a version actually used and what this file specifies.

    Written to compare against a *response*, so it is total: a version generated by hand,
    or by an older version of this file, produces a list rather than an exception. The
    point is to make a silent difference loud, since the preprocessing is not revisitable.
    """
    problems: list[str] = []
    if not isinstance(actual, dict):
        return [f"no preprocessing recorded on the version (got {type(actual).__name__})"]

    if actual.get("auto-orient") is not True:
        problems.append(f"auto-orient is {actual.get('auto-orient')!r}, expected True")

    resize = actual.get("resize")
    if not isinstance(resize, dict):
        problems.append("no resize step - the sidecar infers at 640, so training must match")
    else:
        for axis in ("width", "height"):
            if resize.get(axis) != EXPECTED_SIZE:
                problems.append(f"resize {axis} is {resize.get(axis)!r}, expected {EXPECTED_SIZE}")
        fmt = resize.get("format")
        if fmt != PREPROCESSING["resize"]["format"]:
            problems.append(
                f"resize format is {fmt!r}, expected {PREPROCESSING['resize']['format']!r} "
                "('Fit within' keeps the aspect but hands 44% of the canvas to padding, "
                "which costs the far cells the most)"
            )

    for step, why in FORBIDDEN_STEPS.items():
        if step in actual:
            problems.append(f"preprocessing includes {step!r}: {why}")
    return problems


def describe(settings: dict) -> str:
    """The settings as a human reads them, for the dry run and the receipt."""
    prep = settings["preprocessing"]
    resize = prep.get("resize", {})
    return (
        f"auto-orient: {prep.get('auto-orient')}  ·  "
        f"resize: {resize.get('format')} {resize.get('width')}x{resize.get('height')}  ·  "
        f"augmentation: {'none' if not settings['augmentation'] else settings['augmentation']}"
    )


def classes_with_distance(classes: object) -> dict[str, list[str]]:
    """Project class names that carry a distance word, with the words that matched each one.

    Pure, so the refusal below is testable without the network - and it exists because this
    is the *last* point at which the mistake is free. Distance is a tag on the image and a
    cell in the coverage tables, never a category (MODEL_TRAINING.md 8.1): a version
    generated from a class list of `... close` / `... mid` / `... far` declares 24 classes,
    so the trained head gets one output per product-and-distance and every box comes back
    under a name the app's own roster does not contain - with no error anywhere, because a
    class *name* records none of this.

    The other three checks see it at a different moment: `clean_v2.class_list_rows` (what
    `sanity` prints) and `label_classes.py` both look at the live project, and
    `train_v2.check_export` looks at a version that already exists. Only this one can refuse
    *before* the number is spent, which is why it fails closed rather than warning.
    """
    found = {str(name): distance_tokens_in(name) for name in (classes or [])}
    return {name: words for name, words in found.items() if words}


def next_version(project_body: dict) -> int | None:
    """The number the next generation will take.

    Worth computing rather than assuming: a trashed version still holds its number, so
    "we deleted v1" does not mean the next generation is v1. v2's project has one version
    in the Trash, which is why the next is 2 rather than 1.
    """
    versions = project_body.get("versions")
    if not isinstance(versions, list) or not versions:
        return None
    numbers = []
    for entry in versions:
        vid = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(vid, str) and vid.rsplit("/", 1)[-1].isdigit():
            numbers.append(int(vid.rsplit("/", 1)[-1]))
    return max(numbers) + 1 if numbers else None


def project(project_id: str, key: str) -> dict:
    r = httpx.get(f"{API}/{WORKSPACE}/{project_id}", params={"api_key": key}, timeout=60)
    body = r.json()
    if r.status_code != 200:
        raise SystemExit(f"could not read project {project_id}: {body.get('error', r.status_code)}")
    return body


def version(project_id: str, number: int, key: str) -> dict:
    r = httpx.get(f"{API}/{WORKSPACE}/{project_id}/{number}", params={"api_key": key}, timeout=60)
    body = r.json()
    if r.status_code != 200:
        raise SystemExit(f"could not read version {number}: {body.get('error', r.status_code)}")
    # The single-version route nests it under "version"; the project route returns the same
    # object inline in its `versions` list. Both shapes are handled rather than trusting one.
    return body.get("version") if isinstance(body.get("version"), dict) else body


def cmd_verify(args: argparse.Namespace, key: str) -> int:
    v = version(args.project, args.verify, key)
    print(f"version {args.verify} of {args.project}")
    print(f"  images   : {v.get('images')}")
    splits = v.get("splits") or {}
    if isinstance(splits, dict):
        print("  splits   : " + ", ".join(f"{k} {splits[k]}" for k in sorted(splits)))
    print(f"  created  : {v.get('created')}")
    print(f"  exporting: {v.get('generating')} (false means the version is built)")
    print()
    print("preprocessing it actually used:")
    print(f"  {describe({'preprocessing': v.get('preprocessing') or {}, 'augmentation': v.get('augmentation') or {}})}")
    print()
    problems = compare_preprocessing(v.get("preprocessing"))
    if v.get("augmentation"):
        problems.append(f"augmentation is not empty: {v['augmentation']}")
    if problems:
        print("MISMATCH - this version is not what generate_version.py specifies:")
        for p in problems:
            print(f"  - {p}")
        print()
        print("Preprocessing cannot be edited on an existing version; the fix is a new version.")
        return 2
    print("matches generate_version.py.")
    print()
    print_reminders()
    return 0


def print_reminders() -> None:
    print("Next, and both are easy to miss:")
    print()
    print("  1. Export the version (YOLOv11 PyTorch) and train it - MODEL_TRAINING.md 6:")
    print("       yolo detect train model=yolo11s.pt data=<export>/data.yaml \\")
    print("         epochs=100 imgsz=640 batch=16 patience=25 name=scanncart-grocery-v2")
    print("     Target: mAP50 >= 0.90 and recall >= 0.85 for EVERY class.")
    print()
    print("  2. check the weights are run at this version's geometry - resize_mode: stretch.")
    print("     Train_v2.py --install records that requirement beside the weights, and `auto`")
    print("     now honours it, so the default is correct. It is still worth confirming in the")
    print("     Admin Panel's Model field: an explicit `letterbox` overrides the record and")
    print("     presents every object at 0.56x the canvas this version trained at - a silent")
    print("     accuracy loss, worst on the far cells.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="snc-grocery")
    ap.add_argument("--dry-run", action="store_true", help="print the request body, generate nothing")
    ap.add_argument(
        "--yes",
        action="store_true",
        help="actually generate; without it, the command only reports what it would do",
    )
    ap.add_argument("--verify", type=int, default=0, metavar="N", help="read version N back and compare it")
    ap.add_argument("--json", action="store_true", help="emit the request body and exit")
    args = ap.parse_args(argv)

    settings = version_settings()

    if args.json:
        print(json.dumps(settings, indent=1))
        return 0

    if args.verify:
        return cmd_verify(args, load_key(args.project))

    print(f"version settings -> {describe(settings)}")
    print()

    if args.dry_run or not args.yes:
        print("request body for POST /{workspace}/{project}/generate:")
        print(json.dumps(settings, indent=1))
        print()
        if not args.dry_run:
            print("Nothing was generated. Re-run with --yes to generate, or --dry-run to see this again.")
        print()
        print("Notes:")
        print("  - Only *annotated* images enter a version, so generate this after labeling (and")
        print("    after the Tier C2b frames are marked null - an unmarked one is excluded).")
        print("  - A version number is consumed and cannot be reused; a trashed version keeps its")
        print("    number, so this project's next generation is 2 even though v1 is in the Trash.")
        return 0

    key = load_key(args.project)
    body = project(args.project, key)

    # The class list this version would bake in, checked before the POST because the cost of
    # getting it wrong is a version number (which cannot be reused) plus a training run on a
    # head with an output per product-and-distance. Fail closed: nothing is generated.
    tainted = classes_with_distance((body.get("project") or body).get("classes"))
    if tainted:
        print(f"refusing to generate: {len(tainted)} class name(s) carry a distance")
        for name, words in sorted(tainted.items()):
            print(f"  {name!r} ({', '.join(words)})")
        print()
        print("Distance is a tag on the image and a cell in the coverage tables - never a class")
        print("(MODEL_TRAINING.md 8.1). A version generated from this list declares one class per")
        print("product-and-distance, so the trained head gets 24 outputs instead of 8 and every")
        print("box comes back under a name the app's own roster does not contain - with no error,")
        print("because a class *name* records none of this.")
        print()
        print("Fix the class list on Settings -> Classes, MOVE the annotations onto the product")
        print("class (a class-list edit alone orphans the boxes), then re-run this command. No")
        print("version was generated, so no version number was spent.")
        return 2

    expected = next_version(body)
    if expected is not None:
        print(f"project {args.project} is at version {expected - 1}; this generation will be {expected}")

    r = httpx.post(f"{API}/{WORKSPACE}/{args.project}/generate", params={"api_key": key}, json=settings, timeout=120)
    try:
        out = r.json()
    except ValueError:
        raise SystemExit(f"generation request failed ({r.status_code}): {r.text[:200]}")
    if r.status_code != 200:
        raise SystemExit(f"generation failed: {out.get('error', json.dumps(out)[:200])}")

    number = out.get("version")
    print(f"generating version {number}: {out.get('message', '')}".rstrip())
    print()
    print("Read it back once the build finishes:")
    print(f"  sidecar/.venv/Scripts/python.exe sidecar/tools/generate_version.py --verify {number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
