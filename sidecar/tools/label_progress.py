#!/usr/bin/env python
"""Report labeling progress for the v2 set: what is done, and what is wrong.

Answers the two questions that matter while labeling 1,383 images:

  1. How much is left, per class and per distance? A total percentage hides the
     thing that actually stalls a project - one distance bucket nobody has reached.
  2. Which images are labeled with the WRONG class? Every image carries its intended
     class as a tag (or in the manifest), and the labels say what was actually drawn
     on it, so the two can be compared. A mismatch is either a typo or a genuine
     mislabel, and both are worth catching at 100 images rather than after training.

TWO SOURCES, ONE SNAPSHOT
-------------------------
`--source roboflow` reads the project over the API. `--source local` reads the
annotator's own store (`sidecar/annotate/`) - the frames, their YOLO txt labels and
their provenance - with no network and no key at all, which is what keeps this tool
usable after the machine goes offline. Both write the *same* `label_progress.json`,
because the desktop Admin Panel reads that file rather than either source, and a
second snapshot format would mean a second reader in app code.

`--source auto` (the default) picks local when the annotator has run against the set
and Roboflow otherwise. The signal is the annotator's own bookkeeping file
(`annotations-v2/classes.json`), not the labels: a set with nothing decided yet is
still a set being labeled locally, and asking Roboflow about it would report progress
from a project that no longer holds those frames.

The local source is the more precise of the two where they overlap - a drawn class is
an index rather than an API's name summary, and a null annotation is an empty file
rather than the *type* of a JSON field - and it is the only one that can say how many
decided frames carry a machine's unread boxes (`awaiting_review`, and where they
landed). The Roboflow source deliberately leaves those keys out instead of writing a
zero, because it cannot know: its tags record who uploaded a frame, not who drew it.

How state is detected in the Roboflow source - and this is the whole trick, because
the three states are distinguished only by the *type* of `annotations`, not by a flag:

| `annotations` | Meaning | Counted by Roboflow as |
|---|---|---|
| `[]` (empty **list**) | not labeled yet | `unannotated` |
| `{"count": 0, "classes": {}}` (**dict**) | null annotation - deliberately nothing in frame | annotated |
| `{"count": n, "classes": {...}}` (**dict**) | labeled | annotated |

So a truthiness test is wrong twice over (`[]` is falsy, and so is `{"count": 0}`),
and the presence of a dict is exactly what separates "marked as background" from
"nobody has looked at it yet". Verified on v1: all 1,516 images return a dict, 16 of
them with `count: 0`, and the project reports `unannotated: 0` - i.e. those 16 are
registered null annotations (hard negatives), not unfinished work. The local source
keeps the same three states as three states of a file (absent / empty / rows), which is
`annotate/store.py`'s contract.

Note: the project-scoped search endpoint ignores query filters (`max-annotations:0`
returns annotated images and an unchanged total), so everything is filtered here.

ONE FILE, TWO AUTHORITIES
-------------------------
`--capture-splits` writes `<out>/splits.json` (frame -> train|valid|test), the file the
local annotator and `build_dataset.py` both read. The assignment it freezes comes from
the project while the API is reachable, or from a **plan file** (`--split-plan`, i.e. one
of `plan_split.py`'s `split_plan_{a,b,c}.json`). The second one is what makes the chain
local: the plan already decided every frame's side of the train/test line, so nothing
about the split needs a project or a key at all - which is why this step no longer has a
deadline. Both routes are validated the same way against the staged manifests, because a
map that misses a frame is one the build refuses.

    sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py
    sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py --json
    sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py --source local
    sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py --capture-splits \
      --split-plan sidecar/data/datasets/cleaned-v2/split_plan_b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from clean_v2 import CLASS_MAP, TIER_A_CELLS
from label_classes import (
    DISTANCE_ORDER,
    NEW_CLASS_SLUGS,
    PSEUDO_CLASS_SLUGS,
    SPLIT_NAMES,
    SLUG_TO_CLASS,
    WORKSPACE,
    load_key,
    tag_mismatch,
)
from workspace import (
    DEFAULT_OUT,
    MANIFEST_NAME,
    PROVENANCE_NAME,
    SPLITS_NAME,
    resolve_extras,
)  # workspace lives outside the repo tree

# The directory the annotator writes its labels into, relative to the staged set. Mirrors
# `annotate/run.py`'s default so a run of either tool finds the other's work.
ANNOTATIONS_DIRNAME = "annotations-v2"


def fetch_all(client: httpx.Client, key: str, project: str) -> dict[str, dict]:
    """name -> record, converged.

    The search route drops records while its index catches up (1,297 / 1,380 / 1,383 on
    three consecutive sweeps of the same project), so a single pass would under-report
    progress *and* report finished images as missing.
    """
    merged: dict[str, dict] = {}
    total = 0
    for attempt in range(8):
        offset = 0
        while True:
            body = client.post(
                f"https://api.roboflow.com/{WORKSPACE}/{project}/search",
                params={"api_key": key},
                json={"limit": 500, "offset": offset, "fields": ["name", "annotations", "tags", "split"]},
            ).json()
            results = body.get("results")
            if results is None:
                raise SystemExit(f"search failed: {str(body)[:200]}")
            total = body.get("total", total)
            for rec in results:
                merged[rec["name"]] = rec
            offset += len(results)
            if not results or offset >= total:
                break
        if total and len(merged) >= total:
            break
        if attempt < 7:
            time.sleep(3)
    if total and len(merged) < total:
        print(f"WARNING: indexed {len(merged)} of {total} images - progress below is incomplete")
    return merged


def state_of(rec: dict) -> tuple[str, int]:
    """('unlabeled' | 'null' | 'labeled', box count).

    Only the type of `annotations` separates the three: a list means the image has no
    annotation record at all, while a dict with `count: 0` means it was deliberately
    marked as containing nothing. Treating that dict as "no boxes = not labeled" would
    silently re-open every hard negative in the project.
    """
    ann = rec.get("annotations")
    if not isinstance(ann, dict):
        return "unlabeled", 0
    n = int(ann.get("count") or 0)
    return ("labeled" if n > 0 else "null"), n


def project_states(project: str, key: str) -> dict[str, int]:
    """How many images the project holds, and how many of them carry a *decision*.

    The counts `generate_version.py` prints before it spends a version number, because they are
    what a version contains: only annotated images enter one, and a null annotation is a decision
    (the hard negatives) rather than outstanding work. Read here rather than there because the
    three states are this module's own reading - `state_of` above, verified against v1's project -
    and a second interpretation of the API's `annotations` field in another tool is how the panel
    and the preflight would come to disagree about the same project.

    Raises `SystemExit` when the search route will not answer: the caller decides whether that is
    fatal, and every caller is already talking to this project over the network.
    """
    with httpx.Client(timeout=180) as client:
        index = fetch_all(client, key, project)
    states = Counter(state_of(record)[0] for record in index.values())
    return {
        "images": len(index),
        "labeled": states["labeled"],
        "null": states["null"],
        "unlabeled": states["unlabeled"],
    }


def annotated_classes(rec: dict) -> Counter:
    """The class actually drawn on the image, which is what catches a mislabel."""
    ann = rec.get("annotations")
    if isinstance(ann, dict):
        return Counter(ann.get("classes") or {})
    return Counter()


def display_name(slug: str) -> str:
    """How a slug is shown to a person.

    `negative` is the one slug that is not a class to label: those frames are §2's
    hard negatives, which carry no annotation by definition. Naming it after the
    slug would put a row in the class table that looks like a ninth class to label,
    so it is spelled out instead - and the parens keep it sorting apart from the
    real class names.
    """
    if slug in PSEUDO_CLASS_SLUGS:
        return f"({slug}: background frames, nothing to draw)"
    return SLUG_TO_CLASS.get(slug, NEW_CLASS_SLUGS.get(slug, slug))


def slug_of(rec: dict) -> str:
    """The class slug from the tags. Distance tags are the other tag present.

    Pseudo-class slugs count here: without that, a hard-negative frame's tag resolves
    to nothing and the frame lands in an `unknown` bucket, which reads as a filing bug
    rather than as the deliberate background set it is.
    """
    tags = rec.get("tags") or []
    for t in tags:
        if t in SLUG_TO_CLASS or t in NEW_CLASS_SLUGS or t in PSEUDO_CLASS_SLUGS:
            return t
    return ""


def distance_of(rec: dict) -> str:
    for t in rec.get("tags") or []:
        if t in DISTANCE_ORDER:
            return t
    return ""


def session_of(rec: dict) -> str:
    """The capture session from the tags, or "" when the image carries none.

    Read from the tags rather than from the manifest on purpose: the hard negatives live
    in their own manifest, so a manifest lookup would leave all 50 of them looking like
    no session at all. Every tag the uploader writes is one of three things - class,
    distance, session - so the session is whatever is left over. More than one leftover
    tag is possible only if someone tagged an image by hand in the UI, and that is
    joined rather than dropped: an image's provenance is not this tool's to guess.

    "" is the ordinary case for images uploaded before the session tag existed, and it is
    reported as its own row rather than folded into s1. Assuming it was s1 is true here
    and now, false later, and invisible either way.
    """
    known = set(SLUG_TO_CLASS) | set(NEW_CLASS_SLUGS) | set(PSEUDO_CLASS_SLUGS) | set(DISTANCE_ORDER)
    extra = sorted(t for t in (rec.get("tags") or []) if t not in known)
    return "+".join(extra)


def session_rows(session_stat: dict[str, dict[str, list[int]]]) -> list[dict]:
    """The capture-session view, as the snapshot carries it.

    Pure, so the sidecar's reader can be tested against the writer's actual output
    rather than against a hand-written copy of it that would agree only by accident.
    `splits` is the count of splits a session appears in, which is the leak signal: more
    than one means train and test share a rig state, a day and a lighting setup, so a
    test reading is held-out frames rather than an estimate on an unseen session.

    An untagged session is named rather than dropped. "" would render as a blank row and
    read as a bug in the panel; assuming the images were s1 is true here, false later,
    and invisible either way.
    """
    rows = []
    for name in sorted(session_stat):
        per_split = session_stat[name]
        rows.append(
            {
                "name": name or "(no session tag)",
                "train": per_split.get("train", [0, 0])[1],
                "valid": per_split.get("valid", [0, 0])[1],
                "test": per_split.get("test", [0, 0])[1],
                "decided": sum(v[0] for v in per_split.values()),
                "total": sum(v[1] for v in per_split.values()),
                "splits": sum(1 for v in per_split.values() if v[1]),
            }
        )
    return rows


def tier_a_block(cells: dict[tuple[str, str], list[int]]) -> dict:
    """Tier A's capture gap, carried into the snapshot so the desktop Admin Panel can show
    it without app code holding any opinion about the capture plan.

    The targets are `clean_v2.TIER_A_CELLS` - the machine-readable copy of the checklist
    table that `scaffold` also builds folders from, so the panel and the folder skeleton
    cannot disagree - and `have` is how many images the source holds in that cell.
    Ordered most-remaining-first: the question this answers is "what do I shoot next".

    Shared by both sources so a local run and a Roboflow run cannot rank the same cells
    differently, which would make the panel's next-action list depend on which tool wrote
    the file.
    """
    tier_a_cells = [
        {
            "slug": CLASS_MAP[product],
            "distance": distance,
            "target": want,
            "have": cells.get((CLASS_MAP[product], distance), [0, 0])[1],
            "decided": cells.get((CLASS_MAP[product], distance), [0, 0])[0],
        }
        for (product, distance), want in TIER_A_CELLS.items()
    ]
    for cell in tier_a_cells:
        cell["remaining"] = max(0, cell["target"] - cell["have"])
    tier_a_cells.sort(key=lambda c: (-c["remaining"], c["slug"], c["distance"]))
    return {
        "target": sum(TIER_A_CELLS.values()),
        "remaining": sum(c["remaining"] for c in tier_a_cells),
        "cells_under_target": sum(1 for c in tier_a_cells if c["remaining"] > 0),
        "cells": tier_a_cells,
    }


def class_names_block(cells: dict[tuple[str, str], list[int]]) -> dict:
    """slug -> display name. Travels in the snapshot on purpose.

    The reader (`sidecar/app/dataset_status.py`) is app code and must not import these
    tools, so without this it would have to carry its own copy of the roster and drift.
    """
    slugs = {s for s, _ in cells} | set(SLUG_TO_CLASS)
    return {slug: display_name(slug) for slug in sorted(slugs)}


@dataclass
class Progress:
    """What both sources produce: the snapshot, plus what the console report prints from.

    One shape for two writers is the point. The panel reads the *file*, so a local run and
    a Roboflow run have to leave the same keys behind - and the report below is written
    once, so a number cannot be computed one way for the terminal and another way for the
    snapshot.
    """

    summary: dict
    cells: dict[tuple[str, str], list[int]]
    per_class: dict[str, list[int]]
    split_stat: dict[str, list[int]]
    sessions_out: list[dict]
    null_names: list[str]
    mismatches: list[tuple[str, str, Counter]]
    multi_box: list[tuple[str, int]]
    expected: dict[str, dict] = field(default_factory=dict)
    # name -> the split the project holds it in, for the staged names the project knows. Only
    # the Roboflow source can fill this: it is the one place the project's assignment is readable,
    # and `--capture-splits` freezes it beside the staged set. `None` means "this source cannot
    # say", which is also what makes the flag refuse rather than write an empty file - unless a
    # `--split-plan` file answers the same question locally (see `capture_splits`).
    split_by_name: dict[str, str] | None = None


def roboflow_progress(args, out: Path, expected: dict[str, dict]) -> Progress:
    """Read the project over the API, and require a key to do it.

    Everything here needs the network, which is why it is a function of its own: the
    local source below answers the same questions from disk, and the two must not share
    anything that only one of them can compute.
    """
    key = load_key(args.project)
    with httpx.Client(timeout=180) as client:
        index = fetch_all(client, key, args.project)

    total = 0
    done = 0
    nulls = 0
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # (slug, dist) -> [done, total]
    split_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # session -> split -> [done, total]. This is what makes the panel able to answer the
    # question the split plan cannot: which capture session each split's frames came from,
    # and therefore whether the test number is a genuinely unseen session or only
    # held-out frames of one the model trained on.
    session_stat: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    mismatches: list[tuple[str, str, Counter]] = []
    multi_box: list[tuple[str, int]] = []
    null_names: list[str] = []

    for name, rec in index.items():
        slug = slug_of(rec) or (expected.get(name, {}).get("class") or "unknown")
        dist = distance_of(rec) or (expected.get(name, {}).get("distance") or "unknown")
        state, n = state_of(rec)
        # A null annotation counts as *done*: it is a decision someone made and it will
        # enter the version. Reporting it as outstanding would mean chasing work that is
        # already finished.
        is_done = state != "unlabeled"
        total += 1
        done += is_done
        nulls += state == "null"
        if state == "null":
            null_names.append(name)
        cells[(slug, dist)][0] += is_done
        cells[(slug, dist)][1] += 1
        per_class[slug][0] += is_done
        per_class[slug][1] += 1
        split = rec.get("split") or "?"
        split_stat[split][0] += is_done
        split_stat[split][1] += 1
        sess = session_stat[session_of(rec)]
        sess[split][0] += is_done
        sess[split][1] += 1

        if is_done:
            got = annotated_classes(rec)
            # The rule is `label_classes.tag_mismatch`'s, shared with the local source below and
            # with `dataset_doctor.py`: only a wrong *class* is a mismatch. A background frame, a
            # hard negative and a slug nothing knows all answer empty there rather than each
            # needing a case here. (A box drawn on a null frame is C2a's "product plus clutter"
            # material - a different bucket, not a mistake.)
            if tag_mismatch(slug, got):
                mismatches.append((name, slug, got))
            if n > 1:
                multi_box.append((name, n))

    tier_a = tier_a_block(cells)
    sessions_out = session_rows(session_stat)
    summary = {
        "project": args.project,
        # Which writer produced this file. The two sources agree on every key they both
        # set, and they are not the same measurement: a frame the annotator has not
        # reached yet is outstanding locally, while a Roboflow project cannot see a frame
        # nobody uploaded. An operator looking at a number has to be able to tell which
        # one they are reading.
        "source": "roboflow",
        # Freshness matters more than usual here: the desktop Admin Panel renders this
        # snapshot instead of calling Roboflow, so it has to be able to say how old it is.
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": total,
        "decided": done,
        "null_annotations": nulls,
        "percent": round(100 * done / total, 1) if total else 0,
        "by_class": {k: v for k, v in sorted(per_class.items())},
        "by_cell": {f"{k[0]}|{k[1]}": v for k, v in sorted(cells.items())},
        "by_split": dict(split_stat),
        "sessions": sessions_out,
        "mismatches": len(mismatches),
        # Capture gap, as opposed to labeling work: these cells need the camera, not a
        # box drawn. Kept apart from the per-class table because the two are different
        # actions and a single "0%" would conflate them.
        "tier_a": tier_a,
        "classes": class_names_block(cells),
        # Deliberately absent: `pseudo` / `machine_only_by_split` / `reviewed`. Roboflow
        # records who uploaded a frame, not who drew its boxes, so this source has nothing
        # to say about unreviewed machine work - and writing a zero would say "none", which
        # is a different and false claim. `dataset_status` reads absence as unknown.
    }
    return Progress(
        summary=summary,
        cells=cells,
        per_class=per_class,
        split_stat=split_stat,
        sessions_out=sessions_out,
        null_names=null_names,
        mismatches=mismatches,
        multi_box=multi_box,
        expected=expected,
        # Restricted to the staged names on purpose: this file is read as "which split is this
        # staged frame in", and an entry for an image this machine does not have is a stale fact
        # that would only ever be wrong (`build_dataset` joins it by name).
        split_by_name={name: str(index[name].get("split") or "") for name in expected if name in index},
    )


def local_annotations_dir(out: Path, explicit: str = "") -> Path:
    """Where the annotator keeps its labels: a tree beside the staged set, by default.

    The same rule `annotate/run.py` applies, spelled here rather than imported so this tool
    still runs if the annotator package is missing - the Roboflow source has no dependency
    on it at all.
    """
    return Path(explicit).expanduser() if explicit else out.parent / ANNOTATIONS_DIRNAME


def local_tree_exists(annotations: Path) -> bool:
    """Whether the local annotator holds a *decision* about this set yet.

    Not "was the tool ever opened": `classes.json` is written when it starts, so a machine where
    somebody glanced at the worklist and went back to the project would answer *local* and report
    `0 / 1,383 decided` - over a project that has annotations. That reads as a catastrophe and is
    wrong twice over, because it also replaces a good project snapshot with it.

    The signals are the two shapes a decision leaves: a label file with rows, or a provenance
    record that carries a state (which is how a **null** is saved - an empty file, and the reason
    provenance has to be consulted rather than only the labels).
    """
    try:
        body = json.loads((annotations / PROVENANCE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = {}
    if isinstance(body, dict) and any(
        isinstance(record, dict) and record.get("state") for record in body.values()
    ):
        return True
    try:
        candidates = sorted(annotations.glob("*.txt"))
    except OSError:  # a missing tree, or one that is not a directory
        return False
    for path in candidates:
        try:
            if path.read_text(encoding="utf-8").strip():
                return True
        except OSError:
            continue
    return False


def local_progress(args, out: Path) -> Progress:
    """The same snapshot, from the annotator's store: no network, no key, no Roboflow project.

    This is what keeps the tooling usable after the machine goes offline. Everything the
    remote path reads off the project has a local equivalent, and the local ones are the
    *better* evidence where they overlap:

    * the class actually drawn on a frame is a label row rather than an API's name summary;
    * the split comes from `splits.json`, which `--capture-splits` writes either from the
      project (captured before going offline) or from a `--split-plan` file, both beside the
      staged set;
    * the session comes from the manifest, which records it for every staged frame including
      the hard negatives;
    * and provenance (`machine_only`) exists at all, which is what makes `awaiting_review`
      and `machine_only_by_split` reportable - the acceptance gate wants zero of the latter
      in valid/test.

    The price is stated rather than hidden: this source sees only what is *staged*. A cell it
    reports as short may simply not have been staged yet, which is exactly the right reading
    for a Tier A capture gap (the staged set is what will be trained on) and the wrong one for
    "the project holds images this machine does not" - that is the Roboflow source's job.
    """
    # Imported here and not at module scope: the annotator package is a sibling of this one,
    # not a dependency of it, and the Roboflow source must keep running without it.
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from annotate.store import CLASS_NAMES, LabelStore, summary as store_summary

    annotations = local_annotations_dir(out, args.annotations)
    extras = [Path(p).expanduser() for p in args.extras]
    store = LabelStore(out=out, annotations=annotations, extras=extras)
    frames = store.frames()
    splits = store.splits()

    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    split_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    null_names: list[str] = []
    mismatches: list[tuple[str, str, Counter]] = []
    multi_box: list[tuple[str, int]] = []

    for frame in frames:
        is_done = frame.state != "unlabeled"
        # `unknown` for a frame with no distance, the spelling the annotation store uses for
        # the cell key as well - one spelling, so the panel's worklist cannot end up with two
        # buckets for the hard negatives.
        cells[(frame.slug, frame.distance or "unknown")][0] += is_done
        cells[(frame.slug, frame.distance or "unknown")][1] += 1
        per_class[frame.slug][0] += is_done
        per_class[frame.slug][1] += 1
        # `?` is the untagged-split bucket, matching what the remote source gets from the API
        # for an image no plan has reached. A frame with no `splits.json` entry is not a fourth
        # split; it is a frame the planner has not placed yet, and saying so is the point.
        split = splits.get(frame.name) or "?"
        split_stat[split][0] += is_done
        split_stat[split][1] += 1
        if frame.state == "null":
            null_names.append(frame.name)

        if frame.state == "labeled":
            rows = store.read_boxes(frame.name)
            if len(rows) > 1:
                multi_box.append((frame.name, len(rows)))
            drawn = Counter(
                CLASS_NAMES[row.cls] for row in rows if 0 <= row.cls < len(CLASS_NAMES)
            )
            # Same rule as the remote source, and now literally the same code: a box on a hard
            # negative is C2a's material and is not this check's job.
            if tag_mismatch(frame.slug, drawn):
                mismatches.append((frame.name, frame.slug, drawn))

    body = store_summary(frames, splits)
    summary = {
        "project": args.project,
        "source": "local",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": body["total"],
        "decided": body["decided"],
        "null_annotations": body["null_annotations"],
        "percent": round(100 * body["decided"] / body["total"], 1) if body["total"] else 0,
        "by_class": body["by_class"],
        "by_cell": body["by_cell"],
        "by_split": {k: list(v) for k, v in sorted(split_stat.items())},
        "sessions": body["sessions"],
        "mismatches": len(mismatches),
        "tier_a": tier_a_block(cells),
        "classes": class_names_block(cells),
        # The three the remote source cannot answer. `pseudo` is what the panel shows as
        # "awaiting review": decisions a machine made that nobody has looked at, which a
        # percentage of "decided" would otherwise count as finished work.
        "pseudo": body["pseudo"],
        "reviewed": body["reviewed"],
        "machine_only_by_split": body["machine_only_by_split"],
    }
    return Progress(
        summary=summary,
        cells=cells,
        per_class=per_class,
        split_stat=split_stat,
        sessions_out=body["sessions"],
        null_names=null_names,
        mismatches=mismatches,
        multi_box=multi_box,
        expected={
            e["new_name"]: e
            for directory in [out, *extras]
            for e in _manifest_entries(directory)
            if e.get("new_name")
        },
    )


def _manifest_entries(out: Path) -> list[dict]:
    """The staged manifest's entries, tolerantly - this is used for the report's counts only."""
    try:
        entries = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return entries if isinstance(entries, list) else []


def read_split_plan(path: Path) -> dict[str, object]:
    """A plan file read back as the assignment to freeze: name -> split, in `plan_split.py`'s shape.

    Tolerant about the *values* and strict about the *shape*. A value that is not one of the three
    split names is reported frame by frame as unplaced (the same reading the project's own records
    get), because a plan is hand-editable JSON and a typo in one row should cost one row rather
    than the file - `capture_splits` decides that, not this. A file that is not a JSON object at
    all is a different thing: there is no map in it to report on, so it is refused here, before the
    progress report takes a minute to compute on a machine where nothing can be written anyway.
    """
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        raise SystemExit(
            f"no split plan at {path} - run plan_split.py first (its `--out` is this `--out`):\n"
            "  it writes split_plan_a.json, split_plan_b.json and, with --holdout-session, split_plan_c.json"
        )
    except ValueError as exc:
        raise SystemExit(f"{path} is not readable as JSON: {exc}")
    if not isinstance(body, dict):
        raise SystemExit(f"{path} is a JSON {type(body).__name__}, not a frame -> split map")
    return {str(name): split for name, split in body.items()}


def capture_splits(out: Path, progress: Progress, known: dict[str, object] | None = None) -> int:
    """Freeze the split assignment into `<out>/splits.json`.

    Two authorities, and which one applies is a fact about what this run can read rather than a
    preference:

    * **the project** - `progress.split_by_name`, which `--source roboflow` fills. Roboflow
      assigned a split at upload, so it is the authority while the API is up. It is also the one
      project fact that cannot be recovered afterwards (a fresh export only carries the frames
      annotated on the server), which is what used to give this step a deadline.
    * **a plan** - `known`, which `--split-plan` reads out of one of `plan_split.py`'s
      `split_plan_{a,b,c}.json` files. The plan already decided every frame's side of the
      train/test line, so freezing it is what lets the local chain (annotate -> build -> train ->
      accept) run with no API and no key. Nothing on this path touches the network.

    The file is the point, because every downstream reader joins on it: the local annotator (the
    panel shows which side of the train/test line a frame is on) and `build_dataset.py` (a decided
    frame the planner never placed cannot be put in a split without inventing one, and the
    acceptance number depends on which side it lands). Checking the assignment against the staged
    set is therefore not a formality: a plan that quietly misses frames moves the acceptance
    number with no error anywhere else.

    Entries for frames outside `progress.expected` survive when they come from a plan (they cannot
    arise from the project, whose map is restricted to the staged names). That is deliberate:
    `plan_split.py --include` plans *across* staged sets - a held-out session, the hard negatives -
    and the file at `out` is the one read for the set and its extras together, since
    `LabelStore.splits` returns the first file it can parse rather than merging them.

    Fails closed, and still writes what it knows. That pairing is deliberate: a partial map is
    useful to look at (it says exactly which staged frames the assignment never reached), while
    exiting 0 on it would let a `make` chain walk into a build that quietly drops those frames.
    Writing first means the operator can inspect the file the exit code is complaining about.
    """
    if known is None:
        if progress.split_by_name is None:
            # Named rather than generic: the answer to this refusal is a specific file the operator
            # almost always already has (plan_split.py writes it one command earlier), so say which
            # ones are there instead of making them go looking.
            plans = sorted(p.name for p in out.glob("split_plan_*.json"))
            raise SystemExit(
                "--capture-splits has no assignment to freeze: the local source has no project to "
                "read one from, so pass --split-plan <file> - a split plan\n"
                + (
                    f"  plan(s) beside the staged set: {', '.join(plans)}"
                    if plans
                    else f"  no split_plan_*.json in {out} - run plan_split.py first"
                )
            )
        known = progress.split_by_name
    wanted = {name: split for name, split in known.items() if split in SPLIT_NAMES}
    unplaced = {name: split for name, split in known.items() if split not in SPLIT_NAMES}
    unseen = sorted(set(progress.expected) - set(known))

    path = out / SPLITS_NAME
    path.write_text(json.dumps(wanted, indent=1, sort_keys=True), encoding="utf-8")
    counts = Counter(wanted.values())
    print(
        f"splits   -> {path}  ({len(wanted)} frame(s): "
        + ", ".join(f"{split} {counts[split]}" for split in SPLIT_NAMES if counts[split])
        + ")"
    )
    if unplaced:
        print(f"  {len(unplaced)} staged frame(s) have no usable split in the assignment:")
        for name, split in sorted(unplaced.items())[:5]:
            print(f"    {name}: {split!r}")
    if unseen:
        print(f"  {len(unseen)} staged frame(s) are not in the assignment at all:")
        for name in unseen[:5]:
            print(f"    {name}")
        if not unplaced:
            print("    (a frame staged after the plan was drawn, or after the last upload - either")
            print("     way the assignment has to reach it before a build will take it)")
    if unplaced or unseen:
        print("  the assignment is incomplete, so a build from it would refuse those frames")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--project", default="snc-grocery")
    ap.add_argument(
        "--source",
        choices=("auto", "roboflow", "local"),
        default="auto",
        help="where the numbers come from: the Roboflow project, or the annotator's local "
        "labels (auto picks local when the annotator has been used against this set)",
    )
    ap.add_argument(
        "--annotations",
        default="",
        help="where the local labels live (default: <workspace>/annotations-v2, as `make annotate` uses)",
    )
    ap.add_argument(
        "--extras",
        action="append",
        default=[],
        metavar="DIR",
        help="another staged set to include (repeatable). Adds to the staged hard negatives beside "
        "--out, which are their own set with their own manifest - pass --no-extras to mean only "
        "what you name",
    )
    ap.add_argument(
        "--no-extras",
        action="store_true",
        help="count only the sets named with --extras, not the staged hard negatives beside --out",
    )
    ap.add_argument(
        "--capture-splits",
        action="store_true",
        help="also write <out>/splits.json (frame -> train|valid|test), the file the annotator and "
        "build_dataset.py read: from the project while it is reachable, or from --split-plan for "
        "the offline route",
    )
    ap.add_argument(
        "--split-plan",
        default="",
        metavar="FILE",
        help="freeze the assignment from this plan file instead of the project (plan_split.py's "
        "split_plan_{a,b,c}.json). Needs no API or key at all, which is how splits.json gets "
        "written on a machine that cannot reach Roboflow",
    )
    ap.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    plan_path = Path(args.split_plan).expanduser() if args.split_plan else None
    if plan_path is not None and not args.capture_splits:
        raise SystemExit(
            "--split-plan chooses the assignment --capture-splits freezes, so pass --capture-splits too"
        )
    # Read (and validated) here rather than at the write: a plan that is not a frame -> split map is
    # worth saying so about before the progress report takes a minute to compute.
    plan_map = read_split_plan(plan_path) if plan_path is not None else None
    annotations = local_annotations_dir(out, args.annotations)
    use_local = args.source == "local" or (args.source == "auto" and local_tree_exists(annotations))
    if args.source == "auto":
        reasons = {
            True: f"local labels found at {annotations}",
            False: f"no local label tree at {annotations} - reading the Roboflow project",
        }
        print(f"source: {'local' if use_local else 'roboflow'} ({reasons[use_local]})")

    # Resolved after `--out`, because the default is the set staged *beside* it (see
    # `workspace.resolve_extras`). Written back onto `args` as well, since `local_progress` reads
    # the same list off it - the one place in this file that does, so the two cannot disagree.
    #
    # A union rather than either/or: the hard negatives are a second *set*, so naming one more
    # directory must not silently drop the 50 frames that teach the model what the products are
    # not - which is what a snapshot built without them is missing, with no row saying so.
    # `--no-extras` is how a run that means only what it names says so.
    extras = resolve_extras(out.parent, args.extras, include_defaults=not args.no_extras)
    args.extras = [str(p) for p in extras]
    if use_local:
        progress = local_progress(args, out)
    else:
        # The staged sets *and* their siblings, because the project holds the negatives too and
        # `--capture-splits` has to be able to say which split they are in - they are in `train`
        # by construction (`batch_for_session` names their own batch), and a frame the project
        # knows but this manifest does not would otherwise look like a frame that was never
        # uploaded.
        expected: dict[str, dict] = {}
        for directory in [out, *extras]:
            for e in _manifest_entries(directory):
                expected[e["new_name"]] = e
        progress = roboflow_progress(args, out, expected)

    summary = progress.summary
    cells = progress.cells
    per_class = progress.per_class
    split_stat = progress.split_stat
    sessions_out = progress.sessions_out
    expected = progress.expected
    null_names = progress.null_names
    mismatches = progress.mismatches
    multi_box = progress.multi_box
    tier_a = summary["tier_a"]
    total = int(summary["total"])
    done = int(summary["decided"])
    nulls = int(summary["null_annotations"])

    # Before the `--json` early return, so the split capture happens whichever output the caller
    # asked for - it is a separate artifact from the snapshot, and silently skipping it because
    # the progress was printed instead of written would lose the one thing that cannot be
    # re-read later.
    capture_code = 0
    if args.capture_splits:
        if plan_path is not None:
            print(f"assignment: {plan_path} (a split plan - the project is not read)")
        capture_code = capture_splits(out, progress, plan_map)

    if args.json:
        print(json.dumps(summary, indent=1))
        return capture_code

    # Written on every run, not only with --json, because the desktop Admin Panel reads
    # this file rather than calling Roboflow or the annotator - which is what keeps the app
    # offline-safe and the API key out of the runtime. See sidecar/app/dataset_status.py.
    snapshot = out / "label_progress.json"
    snapshot.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    pct = 100 * done / total if total else 0
    print(f"labeling progress: {done}/{total} decided ({pct:.1f}%)")
    if nulls:
        print(f"  of which {nulls} are null annotations (marked as deliberately empty)")
    if "pseudo" in summary:
        awaiting = int(summary["pseudo"])
        print(
            f"  {awaiting} decision(s) are a machine's boxes nobody has reviewed"
            if awaiting
            else "  nothing awaiting review - every decision is human work"
        )
    print(f"images in project: {total}" + (f"  (manifest: {len(expected)})" if expected else ""))
    print()

    slugs = sorted({s for s, _ in cells} | set(SLUG_TO_CLASS) | set(NEW_CLASS_SLUGS))
    w = max(len(display_name(s)) for s in slugs) if slugs else 10
    header = f"{'class':<{w}}  " + "  ".join(f"{d:>11}" for d in DISTANCE_ORDER) + f"  {'total':>11}"
    print(header)
    print("-" * len(header))
    for slug in slugs:
        label = display_name(slug)
        row = []
        for d in DISTANCE_ORDER:
            dn, dt = cells.get((slug, d), [0, 0])
            row.append(f"{'-':>11}" if dt == 0 else f"{dn:>5}/{dt:<5}")
        cn, ct = per_class.get(slug, [0, 0])
        tot = f"{'-':>11}" if ct == 0 else f"{cn:>5}/{ct:<5}"
        print(f"{label:<{w}}  " + "  ".join(row) + f"  {tot}")
    print()
    print("per split:")
    for s in SPLIT_NAMES:
        if s in split_stat:
            sn, st = split_stat[s]
            print(f"  {s:<6} {sn:>5}/{st:<5}  ({100 * sn / st:.0f}%)")
    print()

    # Same reading as plan_split's session table, from the project instead of the
    # manifest: it is the one line that decides how much the eventual test number is
    # worth, so it is printed here rather than left to the panel to infer.
    if sessions_out:
        w = max(len(s["name"]) for s in sessions_out)
        print("per capture session (images, not annotations):")
        for s in sessions_out:
            spread = "  ".join(
                f"{sp} {s[sp]:>5}" if s[sp] else f"{sp} {'—':>5}" for sp in SPLIT_NAMES
            )
            print(f"  {s['name']:<{w}}  {spread}   {s['splits']} split(s)")
        leaked = [s["name"] for s in sessions_out if s["splits"] > 1]
        if leaked:
            print(
                f"  note: {', '.join(leaked)} appear(s) in more than one split, so train and test "
                "share a capture session - the test number is held-out frames, not an unseen session"
            )
        else:
            print("  note: every session sits in exactly one split, so test is an unseen session")
        print()

    # The next action, not just the status: the largest still-empty cell is where a
    # labeling session should start, because a partial pass over big classes leaves the
    # small buckets at zero and the per-distance reading unusable.
    pending = sorted(
        ((dt - dn, slug, d) for (slug, d), (dn, dt) in cells.items() if dn < dt),
        reverse=True,
    )
    if pending:
        print(f"{len(pending)} cell(s) still have unlabeled images; largest first:")
        for left, slug, d in pending[:8]:
            dn, dt = cells[(slug, d)]
            name = display_name(slug)
            if slug in PSEUDO_CLASS_SLUGS:
                # It is outstanding work, so it belongs on this list - but the action is
                # "mark null", not "draw a box". Sending a labeling pass at 50 empty
                # frames is exactly the wasted session this list exists to prevent.
                print(f"  {left:>5} left  {name}  -> mark each null (N), do not draw")
            else:
                print(f"  {left:>5} left  {name} @ {d}  ({dn}/{dt})")
    else:
        print("every cell is fully labeled")

    # Pseudo-classes are excluded: `negative` has no distance *by design*, so counting
    # its empty distance cells here would report a capture gap for the one bucket that
    # is not supposed to have any.
    missing_cells = [
        s
        for s in slugs
        if s not in PSEUDO_CLASS_SLUGS
        and sum(cells.get((s, d), [0, 0])[1] for d in DISTANCE_ORDER) == 0
    ]
    if missing_cells:
        print()
        print("class(es) with NO images at all - capture gap, not a labeling gap:")
        for s in missing_cells:
            print(f"  {display_name(s)}")

    # Finer-grained than the list above: Tier A is per (class, distance) with a target,
    # so it also names the cells that exist but are too thin to measure.
    if tier_a["target"]:
        print()
        if tier_a["remaining"]:
            print(
                f"Tier A capture gap: {tier_a['remaining']} of {tier_a['target']} images still to "
                f"shoot across {tier_a['cells_under_target']} cell(s) - see docs/CAPTURE_CHECKLIST.md"
            )
            for c in tier_a["cells"]:
                if c["remaining"]:
                    print(
                        f"  {c['remaining']:>5} to shoot  {display_name(c['slug'])} @ {c['distance']}"
                        f"  ({c['have']}/{c['target']})"
                    )
        else:
            print(
                f"Tier A complete: every one of the {len(tier_a['cells'])} cell(s) has reached "
                f"its target ({tier_a['target']} images)"
            )

    if null_names:
        print()
        print(f"null annotations ({nulls}) - these enter the version as pure background:")
        for n in sorted(null_names)[:10]:
            print(f"  {n}")
        if len(null_names) > 10:
            print(f"  ... and {len(null_names) - 10} more")

    if mismatches:
        print()
        print(f"{len(mismatches)} image(s) labeled with a class that does not match their tag:")
        for name, slug, got in mismatches[:15]:
            want = SLUG_TO_CLASS.get(slug, slug)
            found = ", ".join(f"{c} x{n}" for c, n in got.items())
            print(f"  {name}: expected {want!r}, found {found}")
        if len(mismatches) > 15:
            print(f"  ... and {len(mismatches) - 15} more")
    else:
        print()
        print("no class mismatches")

    if multi_box:
        print()
        print(f"{len(multi_box)} image(s) have more than one box (multi-item scenes): {len(multi_box)}")

    report = out / "LABEL_PROGRESS.md"
    lines = [
        "# v2 labeling progress",
        "",
        f"Generated by `sidecar/tools/label_progress.py` from the **{summary['source']}** source. "
        f"{done}/{total} annotated ({pct:.1f}%).",
        "",
        "| Class | " + " | ".join(DISTANCE_ORDER) + " | total |",
        "|-------|" + "|".join(["---:"] * (len(DISTANCE_ORDER) + 1)) + "|",
    ]
    for slug in slugs:
        label = display_name(slug)
        cells_md = []
        for d in DISTANCE_ORDER:
            dn, dt = cells.get((slug, d), [0, 0])
            cells_md.append("—" if dt == 0 else f"{dn}/{dt}")
        cn, ct = per_class.get(slug, [0, 0])
        lines.append(f"| {label} | " + " | ".join(cells_md) + f" | {cn}/{ct} |")
    lines += ["", "## Per split", "", "| Split | Done | Total |", "|-------|---:|---:|"]
    for s in SPLIT_NAMES:
        if s in split_stat:
            sn, st = split_stat[s]
            lines.append(f"| {s} | {sn} | {st} |")
    lines += ["", "## Per capture session", "", "| Session | train | valid | test | Splits |", "|---------|------:|------:|-----:|-------:|"]
    for s in sessions_out:
        lines.append(
            f"| {s['name']} | {s['train']} | {s['valid']} | {s['test']} | {s['splits']} |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- null annotations (deliberate background): {nulls}",
        f"- class mismatches: {len(mismatches)}",
        f"- multi-box images: {len(multi_box)}",
    ]
    if "pseudo" in summary:
        lines += [
            f"- decisions awaiting review (machine-drawn, untouched): {summary['pseudo']}",
            f"- reviewed decisions: {summary['reviewed']}",
        ]
        split_counts = summary.get("machine_only_by_split") or {}
        if split_counts:
            lines += [
                "",
                "Machine-only decisions per split (the acceptance gate wants none in valid/test):",
                "",
                *[f"- {split}: {n}" for split, n in sorted(split_counts.items())],
            ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print(f"report   -> {report}")
    print(f"snapshot -> {snapshot}")
    return capture_code


if __name__ == "__main__":
    raise SystemExit(main())
