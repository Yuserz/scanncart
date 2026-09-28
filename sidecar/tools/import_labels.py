#!/usr/bin/env python
"""Pull the decisions already made in the Roboflow project into the local annotation tree.

WHY THIS EXISTS
---------------
The merge reads the annotator's tree (`annotations-v2/<stem>.txt` plus `provenance.json`), never
the project. So a decision that exists only in Roboflow is invisible to `build_dataset.py`: the
frame reads as outstanding work, the merged set is built without it, and nothing says so - which
is the silence `v2_contribution_problem` refuses on when *no* frame has a decision, arriving one
frame at a time instead of all at once.

This is the bridge, in the direction the project -> local one. It reads the project's annotated
images, resolves each against the staged sets, and writes the label file (or the empty file that
is a null) and its provenance into the local tree. Afterwards `label_progress.py --source local`
reports the numbers the project reports, and the merge takes those frames instead of skipping
them.

WHY NOT AN EXPORT
-----------------
A dataset export is the other route to the same boxes, and it is the wrong one here:

* it goes through a **version**, and a version is a one-shot decision - `generate_version.py`
  refuses a project whose class list carries a distance word, the POST spends a version number,
  and the preprocessing it freezes can never be revised. Pulling labels must not require that;
* a version holds the frames annotated *at generation time*, so a pull through one reports
  yesterday's decisions;
* the export's folder names are the project's split assignment, and the split belongs to
  `splits.json` here (`label_progress.py --capture-splits`). A second source of it is how the
  acceptance number stops meaning what it says.

The per-image route is exact instead: one GET per image returns its boxes in the uploaded
image's own pixel space, plus the class name each box was drawn as.

HOW A BOX BECOMES A LOCAL ROW
-----------------------------
`GET /{workspace}/{project}/images/{id}` answers with

    "annotation": {"boxes": [{"label": ..., "x": ..., "y": ..., "width": ..., "height": ...}],
                   "width": ..., "height": ...}

in pixels, where `x`/`y` are the box's **centre**. Measured, not assumed: read as a centre, every
box of a sampled dozen frames sits inside its own frame; read as top-left corners, dozens stick
out by up to half the image. `label` is the project's class *name* - the same strings
`label_classes.SLUG_TO_CLASS` holds - so it maps onto the `cls` column by name rather than by the
project's class order, which is a different order from this dataset's.

Normalized rows are `(x/width, y/height, w/width, h/height)`, and a uniform resize commutes with
that division. So the local file and the uploaded one only have to be the same *photo*: a size
mismatch (a re-staged 640 copy) is a note, while an **aspect** mismatch is a refusal - that is a
crop, and the coordinates describe a frame the staged file no longer shows.

WHAT IT REFUSES
---------------
* **A box label this dataset does not have.** The project currently declares eight classes to the
  dataset's seven, so a frame carrying the extra one cannot be written - and the fix is in the
  project (remove the class, relabel the frame), not here. The frame is refused whole: dropping
  the one unmappable box would silently change what the frame says.
* **A frame whose stored geometry is not the staged file's** (different aspect), for the reason
  above.
* **A box that is not a valid local row** - a non-numeric coordinate, zero extent, a centre
  outside the frame. `LabelStore.write` refuses these too; clamping them would be this tool's
  invention of a label nobody drew.
* **A null on a product frame**, which the store refuses by name (an empty label on a product
  frame teaches the head that the product is not in the frame). The store's own sentence is
  reported.

Nothing partial is ever written: a frame with any problem is left untouched, so a refusal is
always visible as an outstanding frame rather than as a quietly smaller label.

WHAT IT CANNOT SAY, AND THE FLAG FOR THAT
-----------------------------------------
Roboflow records *that* an image is annotated, never *who* drew it: a box a person drew and a box
they accepted from Label Assist come back identical. So the pulled rows are recorded with
provider `roboflow:<project>` and are not counted as machine-only - which is the same footing
v1's 1,815 hand-labelled Roboflow frames are on, and a claim the API cannot check either way.
`--awaiting-review` is the other side of it: the same boxes are recorded as a provider's, unread,
which puts every imported frame into the annotator's review pass (test split first) and keeps
`accept_v2`'s machine-only gate meaningful. Only the operator knows which of the two they did.

Re-running is cheap and idempotent: a frame that already has a local decision is left alone (said
out loud and counted; `--force` replaces it), so an interrupted pull is resumed by running it
again, and nothing is ever pulled twice by accident.

    sidecar/.venv/Scripts/python.exe sidecar/tools/import_labels.py --dry-run
    sidecar/.venv/Scripts/python.exe sidecar/tools/import_labels.py
    sidecar/.venv/Scripts/python.exe sidecar/tools/import_labels.py --awaiting-review
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import httpx

# `annotate.store` lives under `sidecar/`, which is not on `sys.path` when this file is run as a
# script (`tools/` is what lands there) - the same plumbing `build_dataset.read_v2` does. Module
# scope rather than inside a function, unlike there, because this tool *is* the bridge into the
# annotator's tree: the direction is this file writing the local rows, so importing its `Box` and
# class list is the intended dependency, not one to keep at arm's length.
HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from annotate.store import CLASS_NAMES, Box  # noqa: E402  (after the path fix above)
from build_dataset import DEFAULT_V2, read_v2  # noqa: E402
from generations import V2  # noqa: E402
from label_classes import load_key  # noqa: E402
from label_progress import ANNOTATIONS_DIRNAME, fetch_all, state_of  # noqa: E402
from label_progress import WORKSPACE as ROBOFLOW_WORKSPACE  # noqa: E402
from workspace import resolve_extras  # noqa: E402

API = "https://api.roboflow.com"
# How far two aspect ratios may differ and still be "the same frame". A uniform resize keeps the
# ratio exactly (a 4080x3060 original and its 640x480 copy are both 4:3), while a crop moves it -
# and a crop is exactly what this is protecting against. Loose enough for integer rounding.
ASPECT_TOLERANCE = 0.01
# A detail fetch is retried rather than fatal: the pull is resumable, and one flaky response in
# the middle of 858 frames should not send the operator back to the start.
DETAIL_ATTEMPTS = 3
PROGRESS_EVERY = 50
# How many names of each exception kind the report prints before it stops listing.
SHOW = 5

# The class name -> `cls` mapping the rows are written with. Names rather than the project's own
# class order, because the two orders differ and the label is the only thing that crosses.
INDEX_BY_NAME: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}


def translate(
    annotation: dict, local_size: tuple[int, int], index_by_name: dict[str, int] | None = None
) -> tuple[list[Box], list[str]]:
    """One image's API annotation as local rows, or the reasons it cannot be written.

    Pure: the API's shape in, `Box`es out, so the geometry rules are testable without a project,
    a store or a network. **A frame is refused whole**: `boxes` is empty whenever `problems` is
    not, enforced here rather than left to the caller, because a partial list next to a problem is
    exactly the shape a caller writes by accident - and dropping the one unmappable box would
    silently change what the frame says.

    `local_size` is the staged file's own `(width, height)`, read by the caller: the check it
    feeds is the one thing here that needs the disk, and keeping it outside makes this function
    the same call for a synthetic annotation in a test and for a live one.
    """
    index_by_name = INDEX_BY_NAME if index_by_name is None else index_by_name
    try:
        width, height = float(annotation["width"]), float(annotation["height"])
    except (KeyError, TypeError, ValueError):
        return [], ["the annotation carries no usable frame size"]
    if width <= 0 or height <= 0:
        return [], [f"the annotation's frame size is {width:g}x{height:g}"]

    aspect, local_aspect = width / height, local_size[0] / local_size[1]
    if abs(aspect - local_aspect) / max(aspect, local_aspect) > ASPECT_TOLERANCE:
        return [], [
            f"the annotated frame is {width:g}x{height:g} (aspect {aspect:.3f}) and the staged "
            f"file is {local_size[0]}x{local_size[1]} (aspect {local_aspect:.3f}) - that is a "
            "different frame's coordinates, not a resize of this one"
        ]

    boxes: list[Box] = []
    problems: list[str] = []
    for raw in annotation.get("boxes") or []:
        label = str((raw or {}).get("label") or "")
        index = index_by_name.get(label)
        if index is None:
            problems.append(
                f"class {label!r} is not one of this dataset's {len(CLASS_NAMES)} names - the "
                "frame cannot be expressed without it"
            )
            continue
        try:
            x, y = float(raw["x"]), float(raw["y"])
            w, h = float(raw["width"]), float(raw["height"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"a box with a non-numeric coordinate: {str(raw)[:100]}")
            continue
        box = Box(index, x / width, y / height, w / width, h / height)
        if not box.valid():
            problems.append(
                f"a {label!r} box that is not a row the trainer can read: {box.as_row()}"
            )
            continue
        boxes.append(box)
    return ([] if problems else boxes), problems


def classify(index: dict[str, dict], frames: list, force: bool = False) -> dict[str, list[str]]:
    """Who gets pulled, and the four kinds of frame that do not.

    Pure: `frames` is the store's own list (each with a `name` and a `state`), and `index` is
    `label_progress.fetch_all`'s name -> record, so the whole selection rule is testable without
    a project or a workspace.

    * `pull` - the project holds a decision and this frame has none locally (or `--force`);
    * `already` - both sides have one, and the local one is left alone;
    * `unmatched` - the project decided a frame no staged set here holds: a stray upload, or a set
      that has since been re-staged under other names;
    * `unknown` - staged here and not in the project at all;
    * `undecided` - staged and in the project, with no decision there yet. Outstanding work, and
      the one list that is expected to be large.

    Sorted, so a run's report reads the same twice; the caller re-orders `pull` by the store's own
    worklist order when it walks it.
    """
    staged = {frame.name: frame for frame in frames}
    pull: list[str] = []
    already: list[str] = []
    unmatched: list[str] = []
    undecided: list[str] = []
    for name, record in index.items():
        if state_of(record)[0] == "unlabeled":
            if name in staged:
                undecided.append(name)
            continue
        frame = staged.get(name)
        if frame is None:
            unmatched.append(name)
        elif frame.state != "unlabeled" and not force:
            already.append(name)
        else:
            pull.append(name)
    unknown = [name for name in staged if name not in index]
    return {
        "pull": sorted(pull),
        "already": sorted(already),
        "unmatched": sorted(unmatched),
        "unknown": sorted(unknown),
        "undecided": sorted(undecided),
    }


def fetch_image(
    client: httpx.Client,
    key: str,
    project: str,
    image_id: str,
    attempts: int = DETAIL_ATTEMPTS,
    sleep=time.sleep,
) -> dict:
    """One image's details, retried, as the `image` object.

    A hard failure is a `SystemExit` that says the pull is resumable rather than a traceback: the
    frames written before this point are on disk, and re-running skips them (`classify`'s
    `already`), so the operator's next move is to run the command again.
    """
    url = f"{API}/{ROBOFLOW_WORKSPACE}/{project}/images/{image_id}"
    last = "no attempt was made"
    for attempt in range(attempts):
        try:
            response = client.get(url, params={"api_key": key}, timeout=120)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                body = response.json()
                image = body.get("image") if isinstance(body, dict) else None
                if isinstance(image, dict):
                    return image
                last = f"no `image` in the response ({str(body)[:120]})"
            else:
                last = f"HTTP {response.status_code} {response.text[:160]}"
        if attempt < attempts - 1:
            sleep(2**attempt)
    raise SystemExit(
        f"could not read image {image_id} ({last}). The frames pulled before this one are on "
        "disk - re-run the command and it continues from there"
    )


def _local_size(path: Path) -> tuple[int, int]:
    """The staged file's pixel size, which is what the API's coordinates are checked against."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _listing(preamble: str, names: list[str]) -> str:
    shown = ", ".join(names[:SHOW]) + (" ..." if len(names) > SHOW else "")
    return f"{preamble}: {shown}" if names else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2", default=str(DEFAULT_V2), help="the staged set whose frames are pulled into")
    ap.add_argument(
        "--annotations",
        default="",
        help="where the local labels live (default: <v2>/../annotations-v2, as `make annotate` uses)",
    )
    ap.add_argument(
        "--extras",
        action="append",
        default=[],
        metavar="DIR",
        help="another staged set whose frames count as part of this one (repeatable) - adds to the "
        "hard negatives beside --v2, the same union the annotator and the build read",
    )
    ap.add_argument(
        "--no-extras",
        action="store_true",
        help="pull for the set named with --v2 only, not the staged hard negatives beside it",
    )
    ap.add_argument("--project", default=V2.roboflow_project, help="the Roboflow project to read")
    ap.add_argument(
        "--awaiting-review",
        action="store_true",
        help="record the pulled boxes as a provider's, unread - they then join the annotator's "
        "review pass (test split first) and `accept_v2`'s machine-only gate counts them",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="replace frames that already have a local decision (default: leave them alone)",
    )
    ap.add_argument(
        "--limit", type=int, default=0, metavar="N", help="stop after attempting N frames (a trial)"
    )
    ap.add_argument("--dry-run", action="store_true", help="report what would be pulled and stop")
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = ap.parse_args(argv)

    v2_dir = Path(args.v2).expanduser()
    annotations = (
        Path(args.annotations).expanduser() if args.annotations else v2_dir.parent / ANNOTATIONS_DIRNAME
    )
    extras = resolve_extras(v2_dir.parent, args.extras, include_defaults=not args.no_extras)
    store, frames, problem = read_v2(v2_dir, annotations, extras)
    if problem:
        raise SystemExit(f"not pulling into this --v2: {problem}")
    staged = {frame.name: frame for frame in frames}

    provider = f"roboflow:{args.project}"
    key = load_key(args.project)
    pulled = {"labeled": 0, "null": 0}
    refusals: list[str] = []
    notes: list[str] = []
    summary: dict = {
        "project": args.project,
        "v2": str(v2_dir),
        "annotations": str(annotations),
        "staged": len(frames),
        "provider": provider,
    }

    # Everything the network prints goes through one redirection: `fetch_all`'s incomplete-index
    # warning is the one stdout writer, and a `--json` caller has to be able to parse stdout.
    stream = sys.stderr if args.json else sys.stdout
    with contextlib.redirect_stdout(stream):
        with httpx.Client(timeout=180) as client:
            index = fetch_all(client, key, args.project)
            plan = classify(index, frames, args.force)
            states: dict[str, int] = {"labeled": 0, "null": 0, "unlabeled": 0}
            for record in index.values():
                states[state_of(record)[0]] += 1

            if not args.json:
                print(f"project  {args.project} ({ROBOFLOW_WORKSPACE})")
                print(f"v2       {v2_dir}  ({len(frames)} staged frame(s))")
                print(
                    f"remote   {len(index)} image(s): {states['labeled']} labeled, "
                    f"{states['null']} null, {states['unlabeled']} with no decision yet"
                )
                print(
                    f"local    {len(plan['already'])} frame(s) already decided here "
                    "(--force replaces them)"
                )
                print(
                    f"pull     {len(plan['pull'])} frame(s)"
                    + (f" (--limit {args.limit})" if args.limit else "")
                )
                for key_name, preamble in (
                    ("unmatched", "in the project, not staged here"),
                    ("unknown", "staged here, not in the project"),
                ):
                    line = _listing(preamble, plan[key_name])
                    if line:
                        print(f"  ... {line}")

            summary.update(
                {
                    "remote": states,
                    "already": len(plan["already"]),
                    "planned": len(plan["pull"]),
                    "unmatched": plan["unmatched"],
                    "unknown": plan["unknown"],
                    "undecided": len(plan["undecided"]),
                }
            )

            if args.dry_run:
                if not args.json:
                    for name in plan["pull"][:SHOW]:
                        print(f"    would pull {name}")
                    print()
                    print(
                        "no files written (--dry-run) - and this mode reads the index only, so "
                        "box geometry is not checked until the real run"
                    )

            # The dry run stops at the plan: the index is all it reads, and the loop below is the
            # half that fetches coordinates and writes. Emptying the worklist rather than wrapping
            # the loop keeps one copy of the walk.
            order = {frame.name: position for position, frame in enumerate(frames)}
            worklist = (
                [] if args.dry_run else sorted(plan["pull"], key=lambda name: order.get(name, 0))
            )
            attempts_left = args.limit if args.limit else len(worklist)
            for name in worklist:
                if attempts_left <= 0:
                    break
                attempts_left -= 1
                record = index[name]
                if state_of(record)[0] == "null":
                    try:
                        store.write(name, [], null=True, provider=provider)
                    except ValueError as exc:
                        refusals.append(f"{name}: {exc}")
                        continue
                    pulled["null"] += 1
                else:
                    image_id = str(record.get("id") or "")
                    if not image_id:
                        refusals.append(
                            f"{name}: the search index returned no image id, and the details "
                            "route needs one"
                        )
                        continue
                    annotation = fetch_image(client, key, args.project, image_id).get("annotation")
                    if not isinstance(annotation, dict):
                        refusals.append(
                            f"{name}: the project counted boxes on it but its details carry no "
                            "annotation"
                        )
                        continue
                    local_size = _local_size(staged[name].image)
                    if local_size != (annotation.get("width"), annotation.get("height")):
                        notes.append(
                            f"{name}: the annotation is {annotation.get('width')}x"
                            f"{annotation.get('height')} and the staged file is "
                            f"{local_size[0]}x{local_size[1]} - same aspect, so the normalized "
                            "boxes transfer"
                        )
                    boxes, problems = translate(annotation, local_size)
                    if problems:
                        refusals.append(f"{name}: {problems[0]}")
                        continue
                    if not boxes:
                        refusals.append(
                            f"{name}: the project counts {state_of(record)[1]} box(es) and its "
                            "annotation holds none"
                        )
                        continue
                    try:
                        if args.awaiting_review:
                            store.record_suggestion(name, boxes, provider)
                        store.write(name, boxes, provider=provider)
                    except ValueError as exc:
                        refusals.append(f"{name}: {exc}")
                        continue
                    pulled["labeled"] += 1
                done = pulled["labeled"] + pulled["null"]
                if done % PROGRESS_EVERY == 0 and done:
                    print(f"  pulled {done}/{len(worklist)} ...")

    if args.dry_run:
        if args.json:
            json.dump(summary, sys.stdout, indent=1)
        return 0

    written = pulled["labeled"] + pulled["null"]
    summary.update({"pulled": pulled, "refusals": refusals, "notes": len(notes)})
    if args.limit:
        summary["stopped_at_limit"] = written < len(plan["pull"])

    if args.json:
        json.dump(summary, sys.stdout, indent=1)
        return 2 if refusals else 0

    print()
    print(f"pulled {written} frame(s) from {args.project} into {annotations}")
    print(f"  {pulled['labeled']} labeled, {pulled['null']} null")
    if plan["already"]:
        print(f"  ... {len(plan['already'])} left alone (already decided here; --force replaces)")
    if plan["undecided"]:
        print(f"  ... {len(plan['undecided'])} staged frame(s) have no decision in the project yet")
    if plan["unknown"]:
        print(f"  ... {_listing('staged here, not in the project', plan['unknown'])}")
    if plan["unmatched"]:
        print(f"  ... {_listing('decided in the project, not staged here', plan['unmatched'])}")
    if notes:
        print(f"  ... {len(notes)} frame(s) at a different size (same aspect):")
        for line in notes[:SHOW]:
            print(f"      {line}")
    if refusals:
        print()
        print(f"  ! {len(refusals)} frame(s) refused (left untouched):")
        for line in refusals[:SHOW]:
            print(f"      {line}")
        if len(refusals) > SHOW:
            print(f"      ... and {len(refusals) - SHOW} more")
    print()
    print(
        "authorship: the API records that an image is annotated, not who drew it. These are "
        f"recorded as the project's decisions (provider `{provider}`)"
        + (
            " - with --awaiting-review, as a provider's unread boxes, so the annotator's review "
            "pass covers them instead."
            if args.awaiting_review
            else "; `--awaiting-review` is for labels that came from Label Assist and have not "
            "been read yet."
        )
    )
    print()
    print("  label_progress.py --source local   (the snapshot the panel reads, from these labels)")
    print("  build_dataset.py --dry-run         (what a merge would take; it refuses until splits exist)")
    return 2 if refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
