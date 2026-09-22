#!/usr/bin/env python
"""Plan the train/valid/test split for the v2 set and report its coverage.

MODEL_TRAINING.md §8.3 says to split **by capture batch**, because a batch is one
(capture session, product, distance) bucket and its frames are near-identical: cut
one across splits and the model validates on frames it trained on. That explains
every constraint below.

Two plans, because the batch structure cannot satisfy the split rule alone:

  A  strict-by-batch   whole batches only. Honours §8.3 exactly. Cannot put a class
                       in more than one split when that class has a single batch.
  B  stratified        splits each (class, distance) cell 70/20/10 by image. Every
                       class and distance lands in every split, at near-exact
                       proportions. Costs session isolation - §8.3 argues this is
                       defensible *because* the dedup pass already removed the
                       near-duplicate leakage that session-splitting guards against.

And one plan that only exists once you have shot a second session:

  C  held-out session  `--holdout-session sN` puts one whole capture session in
                       `test` and splits the remainder train/valid. This is the only
                       one of the three whose test set is a session train never saw,
                       which is what makes the acceptance number honest rather than
                       optimistic - A and B both draw test from the same session their
                       train came from. It refuses to produce a plan if holding the
                       session out would leave any (class, distance) cell with no
                       train images: a cell train never saw cannot be learned, so its
                       test reading would measure the absence of training rather than
                       the model. That is satisfiable only when the held-out session
                       re-shoots cells the other sessions already cover, which is why
                       Tier D in CAPTURE_CHECKLIST.md is a re-shoot by design.

Read-only: this computes a plan and writes it to JSON/Markdown. Applying it means
re-uploading with `upload --split-plan`, and since a duplicate upload does NOT move
an image's existing split (verified: returns {"duplicate": true} and leaves the
image on train), the project has to be wiped and re-uploaded first. `--commands`
prints that exact sequence.

    sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py
    sidecar/.venv/Scripts/python.exe sidecar/tools/plan_split.py --commands
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from zlib import crc32

from clean_v2 import DEFAULT_SESSION, batch_for_session
from workspace import DEFAULT_OUT  # the workspace lives outside the repo tree

SPLITS = ("train", "valid", "test")
TARGET = {"train": 0.70, "valid": 0.20, "test": 0.10}

# Cost weights. Size dominates because 70/20/10 is the rule; the distance axis is
# next, because valid that is entirely `mid` and test entirely `far` makes the
# per-distance numbers unreadable - which is the axis this whole dataset exists for.
W_SIZE = 40.0
W_DIST_MISSING = 1.5
W_DIST_MIRROR = 2.0
W_CLASS_MISSING = 0.6
# A capture session spread across splits: the frames share a rig state, a day and a
# lighting setup, so train/valid/test then share more than their subject. Priced below
# the size rule (which is the hard 70/20/10 requirement) but above the distance and
# class terms, because it is a correctness problem rather than a coverage nicety.
# Constant when every batch is one session, which is why adding this did not move the
# already-applied plan - see the session section it prints.
W_SESSION_SPLIT = 1.2


class Batch:
    """One upload batch: a (capture session, class, distance) bucket.

    The session is part of the identity, not decoration. `upload --session s2` gives a
    second session its own batch (`milo_mid_s2`), so two sessions of the same cell are
    two batches - and grouping them by the manifest's `batch` alone would merge them
    back into one split key, which is exactly the leak the session is carried to avoid.
    """

    __slots__ = ("name", "cls", "distance", "session", "n", "images")

    def __init__(self, name: str, cls: str, distance: str, session: str, images: list[str]):
        self.name = name
        self.cls = cls
        self.distance = distance
        self.session = session
        self.n = len(images)
        self.images = images


def load_batches(out: Path, include: tuple[Path, ...] = ()) -> tuple[list[Batch], list[dict]]:
    """The batches of the staged set at `out`, plus any `include` directories.

    `include` exists because each capture session is staged into its own directory, so the
    sessions a plan is *about* are not all in one manifest. A held-out-session plan is the
    case that needs it: reading only s3's staged set would show every cell as covered by
    s3 alone and refuse a plan that is in fact fine, because s1 - which covers those cells
    in train - lives in a sibling directory. Reading them together is what makes "is this
    cell still learnable once s3 is held out" a real question rather than a misread one.
    """
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} - run `clean` first")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    for extra in include:
        extra_manifest = Path(extra) / "manifest.json"
        if not extra_manifest.exists():
            raise SystemExit(f"no manifest at {extra_manifest} (passed via --include)")
        entries = entries + json.loads(extra_manifest.read_text(encoding="utf-8"))
    # Manifests written before sessions were recorded carry no `session`, and every image
    # in them came from the first capture - which is what DEFAULT_SESSION names. That is
    # an assumption, but a checkable one: the tags on the images say the same thing, and
    # `sanity` reads them back.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        session = e.get("session") or DEFAULT_SESSION
        grouped[batch_for_session(e["batch"], session)].append(e)
    batches = [
        Batch(
            name,
            rows[0]["class"],
            rows[0]["distance"],
            rows[0].get("session") or DEFAULT_SESSION,
            [r["new_name"] for r in rows],
        )
        for name, rows in sorted(grouped.items())
    ]
    return batches, entries


# --------------------------------------------------------------------------
# Plan A: strict by-batch
# --------------------------------------------------------------------------


def evaluate(assign: dict[str, str], batches: list[Batch], total: int) -> tuple[float, dict]:
    size: Counter[str] = Counter()
    dist: dict[str, Counter] = {s: Counter() for s in SPLITS}
    cls_seen: dict[str, set] = {s: set() for s in SPLITS}
    session_splits: dict[str, set] = defaultdict(set)
    for b in batches:
        s = assign[b.name]
        size[s] += b.n
        dist[s][b.distance] += b.n
        cls_seen[s].add(b.cls)
        session_splits[b.session].add(s)

    global_dist: Counter[str] = Counter()
    for b in batches:
        global_dist[b.distance] += b.n

    cost = 0.0
    for s in SPLITS:
        # 1. how far this split is from its target share of the whole set
        cost += W_SIZE * abs(size[s] / total - TARGET[s])
        # 2 + 3. distance coverage and mirroring
        for d in ("close", "mid", "far"):
            if dist[s][d] == 0:
                cost += W_DIST_MISSING
            cost += W_DIST_MIRROR * abs(dist[s][d] / max(size[s], 1) - global_dist[d] / total)

    all_cls = {b.cls for b in batches}
    for s in SPLITS:
        cost += W_CLASS_MISSING * len(all_cls - cls_seen[s])

    # 4. a session split across N splits pays N-1 times. With one session this is a
    # constant, so it cannot perturb a plan it cannot improve; with two it is what makes
    # the search keep a held-out session whole.
    for splits in session_splits.values():
        cost += W_SESSION_SPLIT * (len(splits) - 1)

    return cost, {"size": size, "dist": dist, "cls": cls_seen, "session_splits": session_splits}


def anneal(batches: list[Batch], total: int, seed: int = 7, iters: int = 240_000) -> dict[str, str]:
    """Deterministic anneal. 3^21 assignments is too many to enumerate, and the cost
    surface is mostly flat, so this searches with restarts and keeps the best."""
    rng = random.Random(seed)
    best: dict[str, str] = {}
    best_cost = math.inf
    for _ in range(6):
        assign = {b.name: rng.choice(SPLITS) for b in batches}
        cost, _ = evaluate(assign, batches, total)
        for i in range(iters):
            t = max(0.02, 1.0 - i / iters)
            b = rng.choice(batches)
            old = assign[b.name]
            new = rng.choice(SPLITS)
            if new == old:
                continue
            assign[b.name] = new
            ncost, _ = evaluate(assign, batches, total)
            if ncost <= cost or rng.random() < math.exp((cost - ncost) / max(t, 1e-6)):
                cost = ncost
            else:
                assign[b.name] = old
        if cost < best_cost:
            best_cost, best = cost, dict(assign)
    return best


# --------------------------------------------------------------------------
# Plan B: stratified by (class, distance) cell
# --------------------------------------------------------------------------


def _split_counts(n: int, valid_share: float, test_share: float) -> tuple[int, int, int]:
    """How a cell of `n` images divides, never starving train.

    Shared by the stratified and held-out plans so the two cannot drift apart on the
    one rule both depend on. `test_share` of 0 means "this cell sends nothing to
    test" (the held-out plan's case, where test is a whole session instead), and it
    also suppresses the give-a-tiny-cell-one-test-image grant, which would otherwise
    invent a test frame out of a cell that asked for none.
    """
    n_valid = max(1, round(n * valid_share)) if n >= 3 else (1 if n >= 2 else 0)
    n_test = max(1, round(n * test_share)) if test_share > 0 and n >= 5 else 0
    if n_valid + n_test >= n:  # never starve train
        n_valid = max(1, min(n_valid, n - 1))
        n_test = max(0, min(n_test, n - n_valid - 1 if n - n_valid > 1 else 0))
    return n - n_valid - n_test, n_valid, n_test


def _interleaved(names: list[str], counts: tuple[int, int, int], seed: str) -> dict[str, str]:
    """Assign a cell's images to splits, shuffled by a per-batch seed.

    Interleave rather than slice: the manifest is already near-duplicate-free, but
    consecutive frames are still the most alike, so spreading them is the cheaper way
    to keep any residue out of the same split. The seed is `zlib.crc32`, not builtin
    `hash()`: string hashing is salted per process, so the plan would come out
    different on every run and stop being a plan.
    """
    n_train, n_valid, n_test = counts
    order = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
    rng = random.Random(crc32(seed.encode()))
    rng.shuffle(order)
    return dict(zip(names, order))


def stratify(batches: list[Batch]) -> dict[str, str]:
    """Split each cell's images 70/20/10, granting valid and test at least one image
    whenever the cell can afford it - a cell of 5 that sends 1 to valid and 0 to test
    is a cell dimension that cannot be measured at all."""
    plan: dict[str, str] = {}
    for b in batches:
        names = sorted(b.images)
        plan.update(_interleaved(names, _split_counts(len(names), TARGET["valid"], TARGET["test"]), b.name))
    return plan


# --------------------------------------------------------------------------
# Plan C: one whole capture session held out
# --------------------------------------------------------------------------


def holdout_plan(batches: list[Batch], session: str) -> dict[str, str]:
    """Put one entire capture session in `test`; split the rest train/valid.

    The whole session, not a sample of it: a single frame of the held-out session
    landing in train re-introduces exactly the shared rig state, lighting and day that
    makes a same-session test optimistic, and it does so invisibly - one leaked frame
    looks like nothing in a 1,400-image set.

    The remainder splits train/valid at the ratio those two targets imply (0.2/0.9),
    because the 10% that normally goes to test is already spoken for by the session.
    """
    plan: dict[str, str] = {}
    valid_share = TARGET["valid"] / (TARGET["train"] + TARGET["valid"])
    for b in batches:
        names = sorted(b.images)
        if b.session == session:
            plan.update(dict.fromkeys(names, "test"))
        else:
            plan.update(_interleaved(names, _split_counts(len(names), valid_share, 0.0), b.name))
    return plan


def acceptance_report(plan: dict[str, str], batches: list[Batch], entries: list[dict], session: str) -> str:
    """The held-out plan's verdict: is this test set actually an unseen session?

    Written as its own section rather than folded into `render` because the question is
    different in kind. `render` asks whether a split is balanced and covered; this asks
    whether the number you are about to quote means what you want it to mean.
    """
    s = summarize(plan, entries, batches)
    by_name = {e["new_name"]: e for e in entries}
    test_cells: Counter[tuple[str, str]] = Counter()
    train_cells: Counter[tuple[str, str]] = Counter()
    for name, split in plan.items():
        e = by_name[name]
        if split == "test":
            test_cells[(e["class"], e["distance"])] += 1
        elif split == "train":
            train_cells[(e["class"], e["distance"])] += 1

    all_cells = {(e["class"], e["distance"]) for e in entries}
    covered = sorted(test_cells)
    uncovered = sorted(all_cells - set(covered))
    lines = [
        f"### Plan C - acceptance run, session `{session}` held out whole",
        "",
        f"All {s['size']['test']} images of capture session `{session}` are `test`; the other "
        f"{s['size']['train'] + s['size']['valid']} images split {s['size']['train']} train / "
        f"{s['size']['valid']} valid. Train and valid therefore share their session(s) - that "
        f"costs model *selection* some of its independence, and it is the price of a test set "
        f"that has none of it. The number to quote is the `test` one.",
        "",
        f"Cells in the acceptance set: {len(covered)} of {len(all_cells)}.",
        "",
        "| Cell | test | train (learnable from) |",
        "|------|-----:|----------------------:|",
    ]
    for cls, distance in covered:
        lines.append(f"| `{cls}/{distance}` | {test_cells[(cls, distance)]} | {train_cells[(cls, distance)]} |")
    lines.append("")
    if uncovered:
        lines.append(
            f"**{len(uncovered)} cell(s) are absent from the acceptance set** - the estimate says "
            "nothing about "
            + ", ".join(f"`{c}/{d}`" for c, d in uncovered)
            + " at all. Shooting those cells into the held-out session is what widens the claim; "
            "leaving them out narrows the sentence you are entitled to write, it does not weaken it."
        )
    else:
        lines.append(
            "**Every cell is in the acceptance set**, so the quoted test number covers the whole "
            "class x distance grid rather than a convenient subset of it."
        )
    lines.append("")
    lines.append(
        "Protocol: generate the version from this plan (train on `train`, ignore `valid` for "
        "anything but selection), then quote `test` as **performance on an unseen capture "
        "session** - and say the session, not just the number, because that is what makes it "
        "unseen. Do not re-shoot, top up or re-upload the held-out session after seeing the "
        "result; a test set is spent the moment it is tuned against."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------


def summarize(plan: dict[str, str], entries: list[dict], batches: list[Batch]) -> dict:
    by_name = {e["new_name"]: e for e in entries}
    total = len(entries)
    size: Counter[str] = Counter()
    dist: dict[str, Counter] = {s: Counter() for s in SPLITS}
    cls: dict[str, Counter] = {s: Counter() for s in SPLITS}
    batch_ids: dict[str, set] = {s: set() for s in SPLITS}
    # Which (class, distance) cells a split can actually speak about. A cell with no
    # train images is one the model was never taught, so its valid/test reading
    # measures the absence of training rather than the model - a different failure from
    # "no images in this split", which is what the missing_dist/missing_cls terms above
    # already cover.
    cell_ids: dict[str, set] = {s: set() for s in SPLITS}
    for name, s in plan.items():
        e = by_name[name]
        size[s] += 1
        dist[s][e["distance"]] += 1
        cls[s][e["class"]] += 1
        cell_ids[s].add((e["class"], e["distance"]))
        batch_ids[s].add(batch_for_session(e["batch"], e.get("session") or DEFAULT_SESSION))

    # A batch that appears in more than one split is session leakage - the thing
    # §8.3's by-batch rule exists to prevent. Count it so the trade is visible.
    straddling = 0
    for b in batches:
        seen = {plan[n] for n in b.images}
        if len(seen) > 1:
            straddling += 1

    # The same question one level up, and the one that actually matters: a *session* in
    # more than one split means train and test share a rig state, a day and a lighting
    # setup. Per-image counts, because that is what the leakage is measured in.
    session_splits: dict[str, set] = defaultdict(set)
    session_size: dict[str, Counter] = defaultdict(Counter)
    for b in batches:
        for n in b.images:
            session_splits[b.session].add(plan[n])
            session_size[b.session][plan[n]] += 1

    all_cls = sorted({e["class"] for e in entries})
    all_dist = ["close", "mid", "far"]
    missing_cls = {s: [c for c in all_cls if cls[s][c] == 0] for s in SPLITS}
    missing_dist = {s: [d for d in all_dist if dist[s][d] == 0] for s in SPLITS}

    all_cells = {(e["class"], e["distance"]) for e in entries}
    return {
        "total": total,
        "size": size,
        "dist": dist,
        "cls": cls,
        "batch_ids": batch_ids,
        "straddling": straddling,
        "session_splits": session_splits,
        "session_size": session_size,
        "n_sessions": len({b.session for b in batches}),
        "missing_cls": missing_cls,
        "missing_dist": missing_dist,
        "all_cls": all_cls,
        "all_dist": all_dist,
        "all_cells": all_cells,
        "no_train_cells": sorted(all_cells - cell_ids["train"]),
    }


def render(title: str, plan: dict[str, str], batches: list[Batch], entries: list[dict], note: str) -> str:
    s = summarize(plan, entries, batches)
    total = s["total"]
    lines = [f"### {title}", "", note, ""]

    lines.append("| Split | Images | Share | Target |")
    lines.append("|-------|-------:|------:|-------:|")
    for sp in SPLITS:
        n = s["size"][sp]
        lines.append(f"| {sp} | {n} | {n / total:.1%} | {TARGET[sp]:.0%} |")
    lines.append("")

    lines.append("Per-distance coverage (this is the axis v2 exists to add):")
    lines.append("")
    lines.append("| Split | " + " | ".join(s["all_dist"]) + " |")
    lines.append("|-------|" + "|".join(["---:"] * len(s["all_dist"])) + "|")
    for sp in SPLITS:
        cells = []
        for d in s["all_dist"]:
            n = s["dist"][sp][d]
            cells.append("**0**" if n == 0 else str(n))
        lines.append(f"| {sp} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("Per-class coverage:")
    lines.append("")
    lines.append("| Class | " + " | ".join(SPLITS) + " |")
    lines.append("|-------|" + "|".join(["---:"] * len(SPLITS)) + "|")
    for c in s["all_cls"]:
        cells = []
        for sp in SPLITS:
            n = s["cls"][sp][c]
            cells.append("**0**" if n == 0 else str(n))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")

    warns = []
    for sp in SPLITS:
        if s["missing_dist"][sp]:
            warns.append(f"**{sp} has no {', '.join(s['missing_dist'][sp])} images** - that distance is unmeasurable in {sp}")
        if s["missing_cls"][sp]:
            warns.append(f"**{sp} has no {', '.join(s['missing_cls'][sp])} images** - "
                         f"{len(s['missing_cls'][sp])} class(es) unmeasurable in {sp}")
    if s["straddling"]:
        warns.append(
            f"{s['straddling']} of {len(batches)} batches straddle splits (session leakage). "
            "Tolerable here only because the dedup pass already removed near-duplicates."
        )
    # Named rather than counted: the count is a policy trade the size rule already made,
    # but *which* cells went missing is only actionable if you can read the list. This is
    # why it is a warning on every plan instead of a constraint inside the search - a
    # whole-batch plan over a single session cannot satisfy it, so constraining the
    # search would silently trade 70/20/10 away to buy a number it cannot reach.
    if s["no_train_cells"]:
        named = ", ".join(f"`{c}/{d}`" for c, d in s["no_train_cells"][:6])
        more = f" and {len(s['no_train_cells']) - 6} more" if len(s["no_train_cells"]) > 6 else ""
        warns.append(
            f"**{len(s['no_train_cells'])} of {len(s['all_cells'])} cell(s) have no train images** "
            f"({named}{more}) - the model is never taught them, so a valid/test reading for one of "
            "them measures the absence of training, not the model. Only a plan that splits "
            "*within* a cell (Plan B), or a held-out session that re-shoots them (Plan C), fixes it."
        )

    lines.append("Capture sessions in this plan:")
    lines.append("")
    lines.append("| Session | " + " | ".join(SPLITS) + " | Splits |")
    lines.append("|---------|" + "|".join(["---:"] * len(SPLITS)) + "|--------:|")
    for sess in sorted(s["session_splits"]):
        counts = s["session_size"][sess]
        cells = [str(counts[sp]) if counts[sp] else "—" for sp in SPLITS]
        lines.append(f"| {sess} | " + " | ".join(cells) + f" | {len(s['session_splits'][sess])} |")
    lines.append("")
    leaked = sorted(sess for sess, sps in s["session_splits"].items() if len(sps) > 1)
    if not leaked:
        lines.append("**No session crosses a split** - every test frame is from a session train never saw.")
    elif s["n_sessions"] == 1 and leaked == sorted(s["session_splits"]):
        lines.append(
            f"**Every batch is capture session {leaked[0]}, so train, valid and test share one. "
            "Unavoidable with a single session, and the honest reading of these numbers is "
            "\"held-out frames\", not \"an unseen session\".** Shoot a later session over cells "
            "train already covers and this section will say the opposite."
        )
    else:
        lines.append(
            f"**Session(s) {', '.join(leaked)} appear in more than one split** - those frames share "
            "a rig state across train and test. Each session's own batches are intact; it is the "
            "across-split sharing that is the leak."
        )
    lines.append("")
    lines.append("**Warnings**" if warns else "**No coverage warnings.**")
    lines.append("")
    for w in warns:
        lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="DIR",
        help="another staged set to plan across, e.g. a later session's (repeatable)",
    )
    ap.add_argument("--commands", action="store_true", help="also print how to apply a plan")
    ap.add_argument(
        "--holdout-session",
        default=None,
        metavar="NAME",
        help="plan the acceptance run from one held-out capture session (e.g. s3); "
        "refuses if holding it out would leave a cell with no train images",
    )
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    batches, entries = load_batches(out, tuple(Path(p).expanduser() for p in args.include))
    total = len(entries)

    sessions = Counter(b.session for b in batches)
    print(f"{len(batches)} batches, {total} images, {len(sessions)} capture session(s)\n")
    for sess, n in sorted(sessions.items()):
        print(f"  session {sess}: {n} batch(es), {sum(b.n for b in batches if b.session == sess)} images")
    print()

    # ---- feasibility limits, stated before any optimising ----
    per_class_batches = Counter(b.cls for b in batches)
    singles = [c for c, n in per_class_batches.items() if n == 1]
    biggest = max(batch.n for batch in batches)

    out_lines = [
        "# v2 split plan",
        "",
        "Generated by `sidecar/tools/plan_split.py`. Read alongside "
        "[MODEL_TRAINING.md](./MODEL_TRAINING.md) §8.3.",
        "",
        "## Feasibility limits, before optimising",
        "",
        f"- A batch is one `(capture session, class, distance)` bucket, so it is the smallest unit "
        f"the by-batch rule allows. The largest is **{biggest} images ({biggest / total:.1%} of the "
        f"set)**, while `test` targets {TARGET['test']:.0%} ({round(total * TARGET['test'])} images): "
        f"no whole-batch assignment can hit 70/20/10 exactly.",
    ]
    if singles:
        out_lines.append(
            f"- {len(singles)} class(es) have only one batch, so under a strict by-batch rule they "
            f"can appear in exactly one split: {', '.join('`' + c + '`' for c in singles)}."
        )
    out_lines.append("")

    # ---- Plan A ----
    assign_a = anneal(batches, total)
    plan_a = {n: assign_a[b.name] for b in batches for n in b.images}
    a = summarize(plan_a, entries, batches)
    print("PLAN A (strict by-batch)")
    for sp in SPLITS:
        d = a["dist"][sp]
        print(f"  {sp:6} {a['size'][sp]:5} ({a['size'][sp] / total:5.1%})  close {d['close']:4} mid {d['mid']:4} far {d['far']:4}"
              f"   missing classes: {len(a['missing_cls'][sp])}")
    print()

    # ---- Plan B ----
    plan_b = stratify(batches)
    b = summarize(plan_b, entries, batches)
    print("PLAN B (stratified per cell)")
    for sp in SPLITS:
        d = b["dist"][sp]
        print(f"  {sp:6} {b['size'][sp]:5} ({b['size'][sp] / total:5.1%})  close {d['close']:4} mid {d['mid']:4} far {d['far']:4}"
              f"   missing classes: {len(b['missing_cls'][sp])}")
    print()

    out_lines.append(render(
        "Plan A - strict by-batch",
        plan_a, batches, entries,
        "Honours §8.3 exactly: every image of a batch lands in the same split, so no "
        "capture session spans train and valid. The price is coverage.",
    ))
    out_lines.append(render(
        "Plan B - stratified per (class, distance) cell",
        plan_b, batches, entries,
        "Every class and distance appears in every split at near-exact proportions. "
        "Costs session isolation within a cell.",
    ))

    # ---- batch-level view of A, since that is what a human assigns in the UI ----
    out_lines.append("## Plan A as batch assignments")
    out_lines.append("")
    out_lines.append(
        "A batch is exactly one `(capture session, class, distance)` bucket, and all three are already "
        "tags on every image in it - so the query in the last column selects precisely that batch. That "
        "makes Plan A applicable *without* wiping: paste a query, Select all matching, then Change "
        "Dataset Split from the bulk actions menu. The session tag is in the query for the same reason "
        "it exists: without it, a cell shot twice would match both sessions' frames and the split would "
        "cross a session."
    )
    out_lines.append("")
    out_lines.append("| Batch | Session | Class | Distance | Images | Split | Search query |")
    out_lines.append("|-------|---------|-------|----------|-------:|-------|--------------|")
    for batch in batches:
        query = f"tag:{batch.cls} tag:{batch.distance}"
        if batch.session:
            query += f" tag:{batch.session}"
        out_lines.append(
            f"| `{batch.name}` | {batch.session} | {batch.cls} | {batch.distance} | {batch.n} | "
            f"{assign_a[batch.name]} | `{query}` |"
        )
    out_lines.append("")

    # ---- recommendation ----
    a_gaps = sum(len(v) for v in a["missing_cls"].values()) + sum(len(v) for v in a["missing_dist"].values())
    b_gaps = sum(len(v) for v in b["missing_cls"].values()) + sum(len(v) for v in b["missing_dist"].values())
    out_lines.append("## Recommendation")
    out_lines.append("")
    out_lines.append(
        f"Plan A leaves **{a_gaps} unmeasurable class/distance slots**; Plan B leaves **{b_gaps}**. "
        "Plan B is the one to use **for evaluating distance robustness**, which is what v2 is for - "
        "under Plan A the `mid` and `far` readings are confounded with the split itself, so a weak "
        "mid-distance result cannot be told apart from an unlucky batch assignment."
    )
    out_lines.append("")
    out_lines.append(
        "Keep Plan A in reserve for a **final acceptance run**: it is the more honest estimate of "
        "performance on a genuinely new capture session, which is the number to quote if the "
        "capstone asks how the model does on unseen data."
    )
    out_lines.append("")
    out_lines.append(
        "Neither plan fixes the set's real imbalance. Both mirror the global "
        f"close/mid/far mix ({sum(1 for e in entries if e['distance'] == 'close')}/"
        f"{sum(1 for e in entries if e['distance'] == 'mid')}/"
        f"{sum(1 for e in entries if e['distance'] == 'far')}), so `mid` stays thin in every split - "
        "`valid` gets a few dozen mid images, which is enough to see a collapse and not enough for a "
        "tight estimate. Only the capture work in the checklist (Tier B) raises that floor; a split "
        "can redistribute coverage but it cannot create it."
    )
    out_lines.append("")

    # ---- Plan C: a whole capture session held out, when one is nominated ----
    written = [out / "split_plan_a.json", out / "split_plan_b.json"]
    plan_c: dict[str, str] | None = None
    if args.holdout_session:
        held = [b for b in batches if b.session == args.holdout_session]
        if not held:
            print()
            print(
                f"no batches in capture session {args.holdout_session!r} - known session(s): "
                f"{', '.join(sorted(sessions))}"
            )
            print(
                "Shoot it first (docs/CAPTURE_CHECKLIST.md, Tier D), then re-run to plan the "
                "acceptance run."
            )
            return 2
        plan_c = holdout_plan(batches, args.holdout_session)
        c = summarize(plan_c, entries, batches)
        # The pin, enforced where it is enforceable. Holding out a session can only leave
        # a cell unlearnable by *having covered* that cell and nothing else covering it -
        # which is a capture mistake (a cell shot only into the held-out session), not a
        # planning one, so the fix is named rather than worked around.
        if c["no_train_cells"]:
            print()
            print(
                f"CANNOT hold out session {args.holdout_session}: it is the only coverage of "
                f"{len(c['no_train_cells'])} cell(s), so holding it out leaves them with no train "
                f"images and nothing for the model to learn:"
            )
            for cls, distance in c["no_train_cells"]:
                print(f"    {cls}/{distance}")
            print()
            print(
                "Either shoot those cells into an earlier session (so train covers them and the "
                "held-out session only re-shoots them), or drop them from the held-out shoot. "
                "An acceptance number that spans a cell the model never saw is not a measurement."
            )
            return 2
        out_lines.append(acceptance_report(plan_c, batches, entries, args.holdout_session))
        written.append(out / "split_plan_c.json")
        print()
        print(f"PLAN C (held-out session {args.holdout_session})")
        for sp in SPLITS:
            d = c["dist"][sp]
            print(f"  {sp:6} {c['size'][sp]:5} ({c['size'][sp] / total:5.1%})  close {d['close']:4} mid {d['mid']:4} far {d['far']:4}"
                  f"   missing classes: {len(c['missing_cls'][sp])}")
        print(
            f"  acceptance set: {len({(e['class'], e['distance']) for e in entries if plan_c[e['new_name']] == 'test'})}"
            f" of {len(c['all_cells'])} cells, and every one of them has train images"
        )

    report = out / "SPLIT_PLAN.md"
    report.write_text("\n".join(out_lines), encoding="utf-8")
    (out / "split_plan_a.json").write_text(json.dumps(plan_a, indent=1), encoding="utf-8")
    (out / "split_plan_b.json").write_text(json.dumps(plan_b, indent=1), encoding="utf-8")
    if plan_c is not None:
        (out / "split_plan_c.json").write_text(json.dumps(plan_c, indent=1), encoding="utf-8")
    print(f"plan A: {a_gaps} unmeasurable slots, plan B: {b_gaps}")
    # Stated on the console for both plans, because it is the one number here that says
    # whether a plan is *learnable*, not merely balanced.
    print(
        f"cells with no train images: A = {len(a['no_train_cells'])} of {len(a['all_cells'])}, "
        f"B = {len(b['no_train_cells'])} of {len(b['all_cells'])}"
    )
    # The session verdict belongs on the console too: it decides how to read every number
    # above, and a report nobody opens should not be the only place it is said.
    if len(sessions) == 1:
        only = next(iter(sessions))
        print(
            f"sessions: one ({only}) - train/valid/test share a capture session, so these are "
            f"held-out frames, not an unseen session"
        )
    else:
        print(f"sessions: {', '.join(sorted(sessions))} - see the session table in the report")
    print(f"report  -> {report}")
    print("plans   -> " + ", ".join(str(p) for p in written))

    if args.commands:
        print()
        print("HOW TO APPLY")
        print()
        print("There is no split API, and an existing image's split cannot be changed by")
        print("re-uploading it - identical bytes come back as {\"duplicate\": true} and the")
        print("old split stands. So there are two routes, and they are not equivalent:")
        print()
        print("Plan A - non-destructive, one bulk action per batch in the UI:")
        print("  A batch is one (session, class, distance) bucket and all three are tags, so")
        print("  `tag:X tag:Y tag:sN` selects exactly that batch. For each row of 'Plan A as")
        print("  batch assignments' in SPLIT_PLAN.md: paste the query, Select all matching,")
        print("  then Change Dataset Split from the bulk actions menu. Nothing is deleted.")
        print("  The session tag is what keeps this exact: a cell shot twice would otherwise")
        print("  match both sessions and put one capture in two splits.")
        print()
        print("Plan B - stratified, needs a wipe and re-upload (the only route, since it")
        print("  varies the split *within* a batch, which no tag query can express):")
        print()
        print("  sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py wipe --yes")
        print(f"  sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py upload --split-plan {out / 'split_plan_b.json'}")
        print(f"  sidecar/.venv/Scripts/python.exe sidecar/tools/clean_v2.py retag")
        print(f"  sidecar/.venv/Scripts/python.exe sidecar/tools/label_classes.py --apply")
        print()
        print("  The last step is not optional: a wipe deletes every image, so the class_name")
        print("  metadata has to be rewritten afterwards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
