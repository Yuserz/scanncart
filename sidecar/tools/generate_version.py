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
`PREPROCESSING` rather than retyped, and why `train_model.py --install` writes it beside the
weights: `resize_mode: auto` then honours it (`app.models.requirement_for`). `--verify`
prints the reminder with the version it checked, and the Admin Panel's model entry carries it
too.

**Everything the POST would bake in is read and printed first**, because a version number is
one-shot: the class list in the order the export will index it, the geometry being sent, and how many
of the project's images carry a decision - which is what the version will contain, since only
annotated images enter one (a null annotation is a decision, and `filter-null` is forbidden below).

Five of those states are refused rather than printed. A class name carrying a **distance** would
train one output per product-and-distance. A project holding the roster's names in **another order**
is refused because the export declares this project's order - so the version would bake a head whose
indices mean a different product than the roster says, and nothing in the chain sees that as an
error: `train_model.check_export` compares names by *membership*, and the dataset doctor refuses the
export only once it exists, by which time the number is spent. (The order cannot be set by API -
MODEL_TRAINING.md 8.1: Settings -> Classes, and it follows the order the classes were created in -
so the tool prints both lists and sends the operator to the one place that can fix it.) A project
declaring a class the **roster does not** is refused too - `generations.class_gaps`' `source_only`
side, and a *re-created* class is exactly that case, which is why the live project's eighth one reads
as this refusal rather than as a note under an ok row. A version declares every class its project
holds, so the export would carry an output the app's roster has no name for, on every capture, for
the life of the weights. A project where **nothing is annotated** is refused, because that version
would freeze an empty set. And a **background batch holding a frame with no null annotation** is
refused, because a hard negative
enters a version only once it is marked null - an unmarked one is excluded with no error, and being
invisible in the class and distance tables (there is nothing to draw either way) it is the one
absence no other readout in this chain can show.

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


def class_order(classes: object) -> list[str] | None:
    """A project's `classes` field as the ordered list a version's export will declare, or `None`.

    A *position* is what the export's `names` list is written from, and therefore what every label
    row of every image in the version is written against - so a mapping is ranked by its value and
    never read in dict order, which is insertion order and not the thing being checked. A list is
    accepted as well, for a route whose shape is already ordered (`read_export_names` handles both
    for the same reason). Indices are read through `int()`, so a route that numbers them as strings
    still ranks.

    **A mapping whose values are not a complete set of positions is not an order, and reads as
    `None`.** The live project route answers `name -> annotation count` - `snc-grocery` says 150, 79,
    282 - so ranking those values put the class with the fewest boxes first. Ranking any
    integer-valued mapping meant that the moment a project's names *did* match its roster,
    `order_mismatch` compared the roster against an order assembled from box counts and refused a
    project whose order was never read at all; the same laxity is what would have let a genuinely
    reordered project pass. The values must be exactly `0..n-1` before this claims to have read an
    order.

    Checked against the live API before being written this way, and the order a version bakes is
    not exposed by this route *or* by any other: the project body carries `classes` (counts, as
    above) and `colors` (a UI palette), and `colors` is falsified as the answer rather than merely
    suspect - `scanncart-grocery`'s export on disk declares the roster's order while its `colors`
    order differs, so reading the palette would be a guess that happens to be right for
    `snc-grocery` alone. It is a palette and the export's order is not a palette.

    `None` therefore means **no order was readable**, and the caller says so rather than inventing
    one: two of these lists in the wrong order is exactly the mistake this check exists for, so a
    guess here *is* the defect. The caller reads the class *names* off the mapping's keys whether or
    not an order could be ranked, so a map that cannot be ranked still reports its membership.
    """
    if isinstance(classes, list):
        return [str(name) for name in classes]
    if isinstance(classes, dict):
        if not classes:
            return []
        try:
            ranked = sorted((int(index), str(name)) for name, index in classes.items())
        except (TypeError, ValueError):
            return None
        if [index for index, _name in ranked] != list(range(len(ranked))):
            return None  # values that are per-class counts, not positions - see the docstring
        return [name for _index, name in ranked]
    return None


def declaring_generation(project_id: str):
    """The generation whose class list `project_id` is supposed to declare, or None.

    Resolved from the project name rather than from a flag: `generations.py` is where each
    generation's Roboflow project is written down, so the project *is* the statement of which roster
    it holds, and a flag could only be told to disagree with it. A project no generation names gets
    no order check (said out loud, not skipped silently): judging it against a roster chosen for it
    would be this tool's invention.

    Imported inside the function because `generations.py` imports `REQUIRED_RESIZE_MODE` from this
    module - a module-level import would be a cycle - and because the spec is only needed on the
    `--yes` path, where a project is read at all.
    """
    import generations

    for generation in generations.GENERATIONS.values():
        if generation.roboflow_project == project_id:
            return generation
    return None


def order_mismatch(names: list[str], generation) -> list[str] | None:
    """The order `names` should be declared in, when they are the roster's names in another one.

    Pure, and narrow on purpose: a **different set** of names is `label_classes.py`'s finding (it
    reads the live project against the same roster) and a version that lost a class is
    `train_model.check_export`'s, so this speaks only where its own remedy - reorder Settings →
    Classes, generate again - is the right one. Returning the expected list rather than a sentence
    is what lets the caller print both orders side by side: the fix is a drag, and a diff of two
    seven-name lists is the instruction.
    """
    if not names or list(names) == list(generation.classes):
        return None
    if sorted(names) != sorted(generation.classes):
        return None
    return list(generation.classes)


def project_readout(project: str, key: str) -> tuple[dict[str, int] | None, list[dict], str | None]:
    """What a version of `project` would contain, or the reason it could not be read.

    Only annotated images enter a version, and a null annotation is a decision (the hard negatives,
    which `FORBIDDEN_STEPS` refuses to filter out) - so the number of frames carrying a decision is
    the size of the set about to be frozen, and it is the number an operator has to see before
    spending the version number to freeze it.

    Read through `label_progress`, which owns the three-state rule (`state_of`) and the retrying
    search sweep, rather than a second look at the API's `annotations` field here - and both answers
    come off **one** sweep, because the hard negatives are what `unmarked_background` refuses on and
    a second read could be looking at a different project.

    A failed read is a *note*, not a dead run: this is a readout, and the two things that make the
    tool fail closed are about the class list, which the caller has already read. `SystemExit` is
    what the search route raises when it will not answer at all - the same exception the POST below
    raises its own failures with.
    """
    import label_progress  # deferred: the dataset tooling is only needed on the --yes path

    try:
        index = label_progress.project_index(project, key)
    except SystemExit as exc:
        return None, [], str(exc)
    return (
        label_progress.project_states(project, key, index),
        label_progress.background_batches(index),
        None,
    )


def unmarked_background(rows: list[dict]) -> list[dict]:
    """The background batches a version would drop frames from, in the order they were read.

    Pure, so the refusal is testable without the network. The rule is `undecided > 0` rather than
    "none of the batch is marked": a batch with 50 of its 52 frames marked and two forgotten loses
    exactly those two, silently and by the same mechanism, so "some were marked" is not the state
    worth passing. Every one of them is a frame the model was supposed to learn background from and
    would never see - and the negative set is not decoration, it is the measured fix for the
    empty-counter false positive (MODEL_TRAINING.md 2).

    Scoped to the background batches on purpose. An unlabeled frame elsewhere in the project is a
    legitimate intermediate - a partial version is a real move - and it is *visible*: the readout
    prints it and the coverage grid has a cell for it. A background frame has neither, which is why
    this one fails closed.
    """
    return [row for row in rows if row.get("undecided")]


def describe_background(rows: list[dict]) -> str:
    """The project's hard negatives as the one line the readout prints about them."""
    if not rows:
        return "none in the project (no frame carries the hard-negative class tag)"
    images = sum(row["images"] for row in rows)
    nulls = sum(row["null"] for row in rows)
    undecided = sum(row["undecided"] for row in rows)
    batches = f"{len(rows)} batch" + ("es" if len(rows) > 1 else "")
    text = f"{images} frame(s) in {batches} - {nulls} marked null"
    return text if not undecided else f"{text}, {undecided} with no decision"


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
    `train_model.check_export` looks at a version that already exists. Only this one can refuse
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


def inspect_project(args: argparse.Namespace, settings: dict, key: str, generate: bool) -> int:
    """Read the project, print everything a generation would bake in, and POST only if `generate`.

    The body of `--yes`, with the POST made optional, so `--check` shows the *same* readout: what
    the command prints and what it sends must not be two code paths, or the readout would be a
    description of something else. The refusals return 2 and spend nothing, and there are five of
    them: a class list carrying a distance, the roster's names in another order, a class the roster
    does not declare, a background batch holding an unmarked frame, and a project with nothing
    annotated. The module docstring above says what each one is about.
    """
    body = project(args.project, key)
    # The project's own block, which is where the class list lives on both routes. Kept apart from
    # `body`: the version count `next_version` reads sits on the outer object.
    project_body = body.get("project") if isinstance(body.get("project"), dict) else body

    # The class list this version would bake in, checked before the POST because the cost of
    # getting it wrong is a version number (which cannot be reused) plus a training run on a
    # head with an output per product-and-distance. Fail closed: nothing is generated.
    tainted = classes_with_distance(project_body.get("classes"))
    if tainted:
        print(f"refusing to generate: {len(tainted)} class name(s) carry a distance")
        for name, words in sorted(tainted.items()):
            print(f"  {name!r} ({', '.join(words)})")
        print()
        print("Distance is a tag on the image and a cell in the coverage tables - never a class")
        print("(MODEL_TRAINING.md 8.1). A version generated from this list declares one class per")
        print("product-and-distance, so the trained head gets 21 outputs instead of 7 and every")
        print("box comes back under a name the app's own roster does not contain - with no error,")
        print("because a class *name* records none of this.")
        print()
        print("Fix the class list on Settings -> Classes, MOVE the annotations onto the product")
        print("class (a class-list edit alone orphans the boxes), then re-run this command. No")
        print("version was generated, so no version number was spent.")
        return 2

    # The same names in another order, also before the POST - and for the same reason: the export
    # declares this project's order, so the version would bake a head whose indices mean a different
    # product than the roster says, with nothing between here and the doctor noticing.
    generation = declaring_generation(args.project)
    declared = project_body.get("classes")
    # The declared *order* and the class *names* are two different facts, and only the first is
    # conditional: the project route may carry no positions at all (it answers a name -> count map),
    # but its keys still name every class. Read separately so an unrankable map reports its
    # membership instead of reading as "none declared" - and so the order check cannot be handed a
    # list assembled from box counts.
    names = class_order(declared)
    member_names = (
        list(names) if names else [str(name) for name in declared] if isinstance(declared, dict) else []
    )
    if generation is not None and names:
        expected_order = order_mismatch(names, generation)
        if expected_order:
            print(
                f"refusing to generate: {args.project} declares {len(names)} classes in the wrong "
                "order"
            )
            print("  declared : " + ", ".join(names))
            print(
                f"  expected : "
                + ", ".join(expected_order)
                + f"   ({generation.name}'s order - generations.py)"
            )
            print()
            print("A version's export declares the classes in this project's order, and a trained")
            print("model's outputs are in its dataset's declared order - so a version generated from")
            print("this list trains a head whose indices mean another product than the roster says.")
            print("Nothing on the way catches it: train_model.py's export check compares names by")
            print("membership, and the dataset doctor refuses the export only once it exists - after")
            print("the version number, which cannot be reused, has been spent.")
            print()
            print("Fix the order on Settings -> Classes (drag the classes; the API cannot set it - the")
            print("order follows the order they were created in), then re-run this command. No version")
            print("was generated, so no version number was spent.")
            return 2

    # What this project's class list is against the roster, in two directions that are not the
    # same finding and do not get the same treatment. A name the roster does **not** declare is
    # drift into a state no version can be right in, so it is refused outright below - and a class
    # that came back after being dropped (a *re-created* one, which is how the live project's
    # eighth class reads) is exactly that case, which is why it is no longer a note under an ok row.
    # A roster class the project has not created yet is work not done - created on the Classes tab
    # and labeled afterwards - so that direction stays a loud report rather than a refusal.
    wanted = list(generation.classes) if generation is not None else []
    # Imported here rather than at the top because `generations.py` imports this module
    # (`REQUIRED_RESIZE_MODE`), which is why `declaring_generation` above does the same.
    import generations

    # The relation is `generations.class_gaps`' - the one owner of every class-list difference in
    # this tree - and the headings below are this caller's own words.
    gaps = generations.class_gaps(member_names, wanted)
    extra = list(gaps.source_only)
    missing = list(gaps.target_only)

    # Refused before the POST for the reason the order check above is: this list bakes one output
    # per class, so a version generated from it carries a head output the app's roster cannot name
    # - on every capture, for the life of the weights. And it is not an export anything downstream
    # will take either: `train_model.check_export` compares names by membership and refuses this
    # one by name at the first run. Both of those are loud and both are one version number too
    # late, because a number cannot be reused.
    if generation is not None and extra:
        print(
            f"refusing to generate: {args.project} declares {len(extra)} class(es) the roster "
            "does not"
        )
        print("  extra    : " + ", ".join(extra))
        print(
            f"  expected : {len(wanted)} class(es) - {generation.name}'s roster and nothing else "
            "(`generations.py`)"
        )
        print()
        print("A version declares every class its project holds, so this one would bake an extra")
        print("output into the export. The app's roster has no name for it, so every box the head")
        print("emits under it comes back under a label nothing in the app can resolve - with no")
        print("error, because nothing about a class *name* says the app has no output for it.")
        print()
        recreated = [n for n in extra if "palmolive" in n.lower()]
        if recreated:
            print(f"{', '.join(recreated)} was dropped from this roster and is present again, so")
            print("it was *re-created* after being dropped. Deleting a class takes its annotations")
            print("with it, so a re-created one is empty - and it arrives last in the creation order,")
            print("which is the order a version's export declares. `sanity` reports this as a")
            print("blocking problem too, at the other end of the chain: before a shoot instead of")
            print("before a version.")
            print()
        print("Delete it on Settings -> Classes (moving its boxes onto a product class first if it")
        print("has any - a class-list edit alone leaves them under a name nothing can resolve), then")
        print("re-run this command. No version was generated, so no version number was spent.")
        return 2

    # Everything the POST would bake in, in one read and before the number is spent: the class
    # list in the order the export will index it, the geometry being sent, and how many of the
    # project's images carry a decision - which is what the version will contain, because only
    # annotated images enter one. The refusals above come first: when the class list is the
    # problem, what is worth printing is the list, not a table around it.
    print(f"project {args.project}")
    if member_names:
        if generation is None:
            print(f"  classes  : {len(member_names)}")
        elif missing:
            # Only this direction reaches here: an unexpected name is refused above, before the
            # readout, so "N extra" would be a count that can never be anything but zero.
            print(
                f"  classes  : {len(member_names)} - {generation.name} declares {len(wanted)} "
                f"({len(missing)} missing)"
            )
        elif names:
            print(f"  classes  : {len(names)}, in {generation.name}'s order")
        else:
            print(
                f"  classes  : {len(member_names)} - {generation.name}'s names, order unverified"
            )
        for position, name in enumerate(member_names, start=1):
            print(f"    {position:>2}  {name}")
        if missing:
            print("  missing  : " + ", ".join(missing))
    else:
        print("  classes  : none declared (`label_classes.py` is what reports that state)")
    if generation is None:
        print(f"  order    : not checked - no generation in generations.py uses {args.project!r}")
    elif not member_names:
        print("  order    : not checked - the project declares no classes to rank")
    elif names is None:
        print("  order    : not verifiable here - this route answers a name -> annotation count map,")
        print("             not name -> index, so there is no order to read. Ranking those counts as")
        print("             positions is what used to refuse a correctly ordered project.")
        print("             The order a version declares is readable off its export afterwards - its")
        print("             data.yaml's `names`, which `dataset_doctor --dataset <export>` judges against")
        print("             the roster - so generation is the read, and the doctor is the verdict.")
        print("             If it comes back wrong, fix it on Settings -> Classes and generate again:")
        print("             the number this one spent cannot be reused.")
    print(f"  settings : {describe(settings)}")

    counts, background, note = project_readout(args.project, key)
    entering = 0
    if counts is None:
        print(f"  images   : not counted ({note})")
        print("             only annotated frames enter a version, so the set it freezes is the")
        print("             annotated one - `label_progress.py` reads the same project in full")
    else:
        entering = counts["labeled"] + counts["null"]
        print(
            f"  images   : {counts['images']} in the project - {counts['labeled']} labeled, "
            f"{counts['null']} marked null, {counts['unlabeled']} not looked at yet"
        )
        print(
            f"  entering : {entering} frame(s) - only annotated frames enter a version, and a null"
            " is a decision rather than an outstanding one"
        )
        # Printed on every run, the clean case included: a batch whose frames were all marked has to
        # read as *checked*, not as never looked at - and this is the only line that can show the
        # difference, since the frames themselves have nothing to draw either way.
        print(f"  negatives: {describe_background(background)}")
    expected = next_version(body)
    if expected is not None:
        print(
            f"  version  : this generation will be {expected} (the project is at {expected - 1} - "
            "trashed versions keep their numbers)"
        )

    unmarked = unmarked_background(background)
    if unmarked:
        bare = sum(row["undecided"] for row in unmarked)
        print()
        print(f"refusing to generate: {bare} hard-negative frame(s) with no null annotation")
        for row in unmarked:
            print(
                f"  {row['batch']}: {row['images']} frame(s) - {row['null']} marked null, "
                f"{row['labeled']} labeled, {row['undecided']} not looked at yet"
            )
        print()
        print("A hard negative enters a version only once it is marked null: it has nothing to draw,")
        print("so an unmarked one is not background to Roboflow - it is an image nobody decided about,")
        print("and it is left out of the version with no error anywhere. Nothing else here can see")
        print("that either: in the class and distance tables it looks exactly like a marked one, and")
        print("`sanity` counts the frames carrying the tag rather than the marks on them.")
        print()
        print("Mark each of them with the annotator's null tool (N) - `label_progress.py` reports how")
        print("many are decided - then re-run this command.")
        print()
        print("Nothing was generated, so no version number was spent.")
        return 2

    if counts is not None and not entering:
        print()
        print("refusing to generate: nothing in this project is annotated, so the version would")
        print("freeze an empty set - and a version number cannot be reused. `label_progress.py`")
        print("lists what is outstanding, from this same project or from the local labels.")
        print("Nothing was generated, so no version number was spent.")
        return 2
    if generation is not None and missing:
        print(
            f"  [!] this is not {generation.name}'s list: {len(missing)} of its class(es) are not "
            "in the project, and a version only declares the classes it is given"
        )
        print("      (`label_classes.py --create-classes` creates a missing one, and its frames are")
        print("      then shot and labeled; `train_model.check_export` refuses the export of a")
        print("      version that lacks a roster class, naming it.)")
    if counts is not None and counts["unlabeled"]:
        print(
            f"  [!] {counts['unlabeled']} frame(s) have no decision yet, so they will NOT enter "
            "this version"
        )
        print("      (an unmarked frame is excluded, not treated as background) - and a version is")
        print("      not revisitable. `label_progress.py` names the cells they are in.")

    if not generate:
        # The same read, stopping one step short of the number: what is printed above is what
        # `--yes` would send, so a review and a generation cannot describe different things.
        print()
        print("Nothing was generated (--check): the readout above is what --yes would send. No")
        print("version number was spent.")
        return 0
    print()

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="snc-grocery")
    ap.add_argument("--dry-run", action="store_true", help="print the request body, generate nothing")
    ap.add_argument(
        "--check",
        action="store_true",
        help=(
            "read the project and print everything a generation would bake in - the class list in "
            "order, the geometry, how many images carry a decision, and the hard negatives - "
            "generating nothing"
        ),
    )
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

    if args.check or args.yes:
        # One read for both: `--check` stops after the readout, `--yes` sends the same thing.
        return inspect_project(args, settings, load_key(args.project), generate=args.yes)

    print(f"version settings -> {describe(settings)}")
    print()
    print("request body for POST /{workspace}/{project}/generate:")
    print(json.dumps(settings, indent=1))
    print()
    if not args.dry_run:
        print("Nothing was generated. Re-run with --yes to generate, or --dry-run to see this again.")
    print()
    print("Notes:")
    print("  - Only *annotated* images enter a version, so generate this after labeling (and")
    print("    after the Tier C2b frames are marked null: an unmarked one is excluded, and a")
    print("    background batch holding one is refused rather than generated).")
    print("  - A version number is consumed and cannot be reused; a trashed version keeps its")
    print("    number, so this project's next generation is 2 even though v1 is in the Trash.")
    print("  - `--check` reads the project and prints the whole decision - the class list in order,")
    print("    the geometry, and how many images carry a decision - spending nothing; `--yes` prints")
    print("    that same readout and then spends the number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
