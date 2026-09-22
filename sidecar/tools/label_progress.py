#!/usr/bin/env python
"""Report labeling progress for the v2 set: what is done, and what is wrong.

Answers the two questions that matter while labeling 1,383 images:

  1. How much is left, per class and per distance? A total percentage hides the
     thing that actually stalls a project - one distance bucket nobody has reached.
  2. Which images are labeled with the WRONG class? Every image carries its intended
     class as a tag, and Roboflow reports the class actually drawn on it, so the two
     can be compared. A mismatch is either a typo or a genuine mislabel, and both are
     worth catching at 100 images rather than after training.

How state is detected - and this is the whole trick, because the three states are
distinguished only by the *type* of `annotations`, not by a flag:

| `annotations` | Meaning | Counted by Roboflow as |
|---|---|---|
| `[]` (empty **list**) | not labeled yet | `unannotated` |
| `{"count": 0, "classes": {}}` (**dict**) | null annotation - deliberately nothing in frame | annotated |
| `{"count": n, "classes": {...}}` (**dict**) | labeled | annotated |

So a truthiness test is wrong twice over (`[]` is falsy, and so is `{"count": 0}`),
and the presence of a dict is exactly what separates "marked as background" from
"nobody has looked at it yet". Verified on v1: all 1,516 images return a dict, 16 of
them with `count: 0`, and the project reports `unannotated: 0` - i.e. those 16 are
registered null annotations (hard negatives), not unfinished work.

Note: the project-scoped search endpoint ignores query filters (`max-annotations:0`
returns annotated images and an unchanged total), so everything is filtered here.

    sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py
    sidecar/.venv/Scripts/python.exe sidecar/tools/label_progress.py --json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from clean_v2 import CLASS_MAP, TIER_A_CELLS
from label_classes import (
    NEW_CLASS_SLUGS,
    PSEUDO_CLASS_SLUGS,
    SLUG_TO_CLASS,
    WORKSPACE,
    load_key,
)
from workspace import DEFAULT_OUT  # workspace lives outside the repo tree

DISTANCES = ("close", "mid", "far")


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
        if t in DISTANCES:
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
    known = set(SLUG_TO_CLASS) | set(NEW_CLASS_SLUGS) | set(PSEUDO_CLASS_SLUGS) | set(DISTANCES)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--project", default="snc-grocery")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    manifest = out / "manifest.json"
    expected: dict[str, dict] = {}
    if manifest.exists():
        for e in json.loads(manifest.read_text(encoding="utf-8")):
            expected[e["new_name"]] = e

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
            # A background frame has no class to be right or wrong about, so there is
            # nothing to compare against. (A box drawn on one is C2a's "product plus
            # clutter" material, which is a different bucket rather than a mistake, so
            # it is deliberately not reported here as a mismatch.)
            want = None if slug in PSEUDO_CLASS_SLUGS else SLUG_TO_CLASS.get(slug)
            got = annotated_classes(rec)
            # Only a wrong *class* is a problem. Extra boxes on a multi-item scene are
            # intended, and an unlabeled sibling in the frame is not this check's job.
            # A null (count 0) has no classes, so it can never reach this branch.
            if want and any(c and c != want for c in got):
                mismatches.append((name, slug, got))
            if n > 1:
                multi_box.append((name, n))

    # Tier A's capture gap, carried into the snapshot so the desktop Admin Panel can show
    # it without app code holding any opinion about the capture plan. The targets are
    # `clean_v2.TIER_A_CELLS` - the machine-readable copy of the checklist table that
    # `scaffold` also builds folders from, so the panel and the folder skeleton cannot
    # disagree - and `have` is what the project actually holds in that cell.
    #
    # Ordered most-remaining-first: the question this answers is "what do I shoot next".
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
    tier_a = {
        "target": sum(TIER_A_CELLS.values()),
        "remaining": sum(c["remaining"] for c in tier_a_cells),
        "cells_under_target": sum(1 for c in tier_a_cells if c["remaining"] > 0),
        "cells": tier_a_cells,
    }

    sessions_out = session_rows(session_stat)

    summary = {
        "project": args.project,
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
        # The slug -> class-name mapping travels in the snapshot on purpose. The reader
        # (sidecar/app/dataset_status.py) is app code and must not import these tools, so
        # without this it would have to carry its own copy of the roster and drift.
        "classes": {
            slug: display_name(slug)
            for slug in sorted({s for s, _ in cells} | set(SLUG_TO_CLASS))
        },
    }
    if args.json:
        print(json.dumps(summary, indent=1))
        return 0

    # Written on every run, not only with --json, because the desktop Admin Panel reads
    # this file rather than calling Roboflow - which is what keeps the app offline-safe
    # and the API key out of the runtime. See sidecar/app/dataset_status.py.
    snapshot = out / "label_progress.json"
    snapshot.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    pct = 100 * done / total if total else 0
    print(f"labeling progress: {done}/{total} decided ({pct:.1f}%)")
    if nulls:
        print(f"  of which {nulls} are null annotations (marked as deliberately empty)")
    print(f"images in project: {total}" + (f"  (manifest: {len(expected)})" if expected else ""))
    print()

    slugs = sorted({s for s, _ in cells} | set(SLUG_TO_CLASS) | set(NEW_CLASS_SLUGS))
    w = max(len(display_name(s)) for s in slugs) if slugs else 10
    header = f"{'class':<{w}}  " + "  ".join(f"{d:>11}" for d in DISTANCES) + f"  {'total':>11}"
    print(header)
    print("-" * len(header))
    for slug in slugs:
        label = display_name(slug)
        row = []
        for d in DISTANCES:
            dn, dt = cells.get((slug, d), [0, 0])
            row.append(f"{'-':>11}" if dt == 0 else f"{dn:>5}/{dt:<5}")
        cn, ct = per_class.get(slug, [0, 0])
        tot = f"{'-':>11}" if ct == 0 else f"{cn:>5}/{ct:<5}"
        print(f"{label:<{w}}  " + "  ".join(row) + f"  {tot}")
    print()
    print("per split:")
    for s in ("train", "valid", "test"):
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
                f"{sp} {s[sp]:>5}" if s[sp] else f"{sp} {'—':>5}" for sp in ("train", "valid", "test")
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
        and sum(cells.get((s, d), [0, 0])[1] for d in DISTANCES) == 0
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
        f"Generated by `sidecar/tools/label_progress.py`. {done}/{total} annotated ({pct:.1f}%).",
        "",
        "| Class | " + " | ".join(DISTANCES) + " | total |",
        "|-------|" + "|".join(["---:"] * (len(DISTANCES) + 1)) + "|",
    ]
    for slug in slugs:
        label = display_name(slug)
        cells_md = []
        for d in DISTANCES:
            dn, dt = cells.get((slug, d), [0, 0])
            cells_md.append("—" if dt == 0 else f"{dn}/{dt}")
        cn, ct = per_class.get(slug, [0, 0])
        lines.append(f"| {label} | " + " | ".join(cells_md) + f" | {cn}/{ct} |")
    lines += ["", "## Per split", "", "| Split | Done | Total |", "|-------|---:|---:|"]
    for s in ("train", "valid", "test"):
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
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print(f"report   -> {report}")
    print(f"snapshot -> {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
