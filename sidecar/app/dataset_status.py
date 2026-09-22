"""Read the labeling-progress snapshot the dataset tooling writes. No network, no key.

The desktop Admin Panel shows how far labeling has got, and the obvious implementation
— have the sidecar call Roboflow — is the wrong one: it would put an API key
requirement, a dataset project id and an internet dependency into a product whose
stated promise is that everything runs locally. So the numbers are produced by the
tracked tooling (`sidecar/tools/label_progress.py`, which already talks to Roboflow and
already owns the roster) and written to a JSON snapshot next to its report. This module
only reads that file.

The consequence is worth being explicit about rather than hiding: the snapshot is
exactly as fresh as the last time someone ran the tool. That is why `generated_at` and
`age_seconds` are part of the response and the panel renders them, instead of
presenting the numbers as live.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

# The sidecar's own copy of the dataset workspace path. It deliberately does NOT import
# sidecar/tools/workspace.py: the tools are development code that a packaged sidecar
# does not ship, so app code must not depend on them. The two are kept honest by a test
# (tests/test_dataset_status.py) rather than by an import.
DATASET_WORKSPACE = Path(__file__).resolve().parents[1] / "data" / "datasets"
SNAPSHOT_PATH = DATASET_WORKSPACE / "cleaned-v2" / "label_progress.json"

DISTANCES = ("close", "mid", "far")
SPLITS = ("train", "valid", "test")

# The pseudo-classes: batch and tag keys rather than labelable products, so their frames carry no
# distance and no box. `negative` is §2's hard-negative set, and it is the one row of the backlog
# where the work is *not* drawing: those frames are marked with the annotator's null tool, which is
# what makes them enter a version at all (an unmarked one is excluded, and the project's
# `unannotated` count cannot tell the two apart).
#
# It mirrors `label_classes.PSEUDO_CLASS_SLUGS`, and the app deliberately keeps its own copy: a
# packaged sidecar does not ship `sidecar/tools/`, so app code must not import it. A test in
# tests/test_dataset_status.py keeps the two from drifting, exactly as the workspace path does.
BACKGROUND_SLUGS = ("negative",)


@dataclass
class DatasetStatus:
    """Labeling progress as of the last tool run.

    `available=False` is a normal state, not an error — it means nobody has run
    label_progress.py yet on this machine (or the snapshot was removed), and the panel
    says so rather than showing a misleading zero.
    """

    available: bool
    snapshot_path: str
    generated_at: str | None = None
    age_seconds: int | None = None
    project: str | None = None
    total: int = 0
    decided: int = 0
    percent: float = 0.0
    null_annotations: int = 0
    mismatches: int = 0
    classes: list[dict] = field(default_factory=list)
    by_distance: dict[str, list[int]] = field(default_factory=dict)
    by_split: dict[str, list[int]] = field(default_factory=dict)
    # Which capture session each split's frames came from. This is the one part of the
    # snapshot that says how much the eventual test number is worth: a session appearing
    # in more than one split means train and test share a rig state, a day and a lighting
    # setup, so the reading is held-out frames rather than an unseen session.
    sessions: list[dict] = field(default_factory=list)
    # The *capture* gap (Tier A), which is a different action from the labeling gap: these
    # cells are waiting for a photo, not a box. Flattened rather than nested to match the
    # rest of this dataclass, which main.py hands to a pydantic model with `**asdict`.
    tier_a_target: int = 0
    tier_a_remaining: int = 0
    tier_a_cells_under_target: int = 0
    tier_a_cells: list[dict] = field(default_factory=list)
    # The *labeling* gap, one entry per cell with work left, biggest first — the list to work
    # through rather than a table to read. Same counts as `classes`, arranged by the question a
    # labeling session actually asks ("what next?") instead of by class name, and each row carries
    # its own distance because a cell is what gets labeled in one sitting.
    labeling_backlog: list[dict] = field(default_factory=list)


def _pair(value: object) -> list[int]:
    """The tool records counts as `[decided, total]`.

    Anything else normalises to zeros rather than raising: a snapshot is a convenience
    artifact written by a separate program, so a shape change there must not be able to
    take the app's Admin Panel down.
    """
    if not isinstance(value, list) or len(value) != 2:
        return [0, 0]
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return [0, 0]


def _int(value: object) -> int:
    """A count, or 0. Same reasoning as `_pair`: the snapshot is written by another
    program, so a shape change there must degrade this panel, not crash it.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _tier_a(raw: object) -> dict:
    """Tier A's capture gap, or zeros when the snapshot predates it.

    A snapshot written before this block existed is the ordinary case on a machine that
    has not re-run the tool, so "missing" reads as empty rather than as an error.
    """
    empty: dict = {"target": 0, "remaining": 0, "cells_under_target": 0, "cells": []}
    if not isinstance(raw, dict):
        return empty
    cells = []
    for cell in raw.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        distance = str(cell.get("distance") or "")
        if distance not in DISTANCES:
            continue
        cells.append(
            {
                "slug": str(cell.get("slug") or ""),
                "distance": distance,
                "target": _int(cell.get("target")),
                "have": _int(cell.get("have")),
                "decided": _int(cell.get("decided")),
                "remaining": _int(cell.get("remaining")),
            }
        )
    return {
        "target": _int(raw.get("target")),
        "remaining": _int(raw.get("remaining")),
        "cells_under_target": _int(raw.get("cells_under_target")),
        "cells": cells,
    }


def _sessions(raw: object) -> list[dict]:
    """The capture-session spread, or [] when the snapshot predates it.

    `splits` is derived from the three counts rather than read from the file, so the flag
    the panel colours on can never disagree with the numbers printed next to it. The
    file's own `splits` counts every split an image sits in, including a `?` from a
    record with no split at all, and that difference is exactly the kind of invisible
    mismatch this reader exists to avoid.
    """
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        entry: dict = {"name": name}
        for split in SPLITS:
            entry[split] = _int(item.get(split))
        entry["decided"] = _int(item.get("decided"))
        entry["total"] = _int(item.get("total"))
        entry["splits"] = sum(1 for split in SPLITS if entry[split])
        out.append(entry)
    return out


def _backlog(by_cell: dict, by_class: dict, names: dict) -> list[dict]:
    """The cells still waiting for a box, most-remaining first.

    Derived here rather than written by the tool, because the snapshot already carries every
    number this needs (`by_cell` is keyed `"<slug>|<distance>"` with `[decided, total]`) and a
    second list in the file could only disagree with the first.

    Cells that are finished are dropped: a worklist is what is *left*, and a row of `242/242`
    would push the work off the screen as the labeling gets closer to done. Ties break on the
    class name then the distance, so the order is stable between runs — a list that reshuffles
    on every refresh is one nobody can keep their place in.

    The background pseudo-class comes from `by_class` instead: its frames carry no distance, so
    they have no cell to sit in, and leaving them out would be the one omission with a cost (an
    unmarked frame never enters the version, and nothing else in this panel would say so).
    """
    names = names if isinstance(names, dict) else {}
    rows: list[dict] = []
    for key, value in (by_cell if isinstance(by_cell, dict) else {}).items():
        slug, _, distance = str(key).partition("|")
        if distance not in DISTANCES:
            # No distance means no cell — the pseudo-classes live in `by_class`, handled below.
            continue
        decided, total = _pair(value)
        remaining = max(0, total - decided)
        if remaining == 0:
            continue
        rows.append(
            {
                "slug": slug,
                "name": names.get(slug) or slug,
                "distance": distance,
                "decided": decided,
                "total": total,
                "remaining": remaining,
                "background": False,
            }
        )
    for slug in BACKGROUND_SLUGS:
        decided, total = _pair(by_class.get(slug))
        remaining = max(0, total - decided)
        if remaining == 0:
            continue
        rows.append(
            {
                "slug": slug,
                "name": names.get(slug) or slug,
                # Empty rather than a distance someone observed: these frames are staged without
                # one on purpose, and printing a distance would invent a cell.
                "distance": "",
                "decided": decided,
                "total": total,
                "remaining": remaining,
                "background": True,
            }
        )
    rows.sort(key=lambda r: (-r["remaining"], str(r["name"]).lower(), r["distance"]))
    return rows


def _age_seconds(stamp: str | None) -> int | None:
    """Seconds since the snapshot was written, or None if the stamp is unusable.

    The tool stamps local time with no offset, which is unambiguous here because the
    sidecar and the tool always run on the same machine.

    Clock, minus stamp. Written the other way round this still looks plausible and
    still returns a number - but `max(0, ...)` then clamps *every* past stamp to 0, so
    the panel reports any aged snapshot as just-written. That is the one reading this
    field exists to give, so the order is pinned by a test that builds its stamp from
    the clock rather than from a literal.
    """
    if not stamp:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            # Clamped because a stamp in the future (a skewed clock, a hand-edited
            # file) is a reason to say "now", not to render a negative age.
            return max(0, int(time.time() - time.mktime(time.strptime(stamp, fmt))))
        except ValueError:
            continue
    return None


def load_dataset_status(path: Path | None = None) -> DatasetStatus:
    snapshot = Path(path) if path is not None else SNAPSHOT_PATH
    if not snapshot.exists():
        return DatasetStatus(available=False, snapshot_path=str(snapshot))
    try:
        raw = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A half-written or corrupt file reads as "no snapshot", not a crash. The tool
        # rewrites it on its next run, and this panel is never load-bearing.
        return DatasetStatus(available=False, snapshot_path=str(snapshot))
    if not isinstance(raw, dict):
        return DatasetStatus(available=False, snapshot_path=str(snapshot))

    by_class = raw.get("by_class") if isinstance(raw.get("by_class"), dict) else {}
    names = raw.get("classes") if isinstance(raw.get("classes"), dict) else {}

    # Derive the distance axis here so the renderer stays dumb. `by_cell` is keyed
    # "<slug>|<distance>" by the tool.
    by_distance: dict[str, list[int]] = {d: [0, 0] for d in DISTANCES}
    by_cell = raw.get("by_cell") if isinstance(raw.get("by_cell"), dict) else {}
    for key, value in by_cell.items():
        distance = str(key).split("|")[-1]
        if distance not in DISTANCES:
            continue
        done, total = _pair(value)
        by_distance[distance][0] += done
        by_distance[distance][1] += total

    by_split: dict[str, list[int]] = {}
    raw_splits = raw.get("by_split") if isinstance(raw.get("by_split"), dict) else {}
    for split in SPLITS:
        if split in raw_splits:
            by_split[split] = _pair(raw_splits[split])

    classes = [
        {
            "slug": slug,
            "name": names.get(slug) or str(slug),
            "decided": _pair(counts)[0],
            "total": _pair(counts)[1],
        }
        for slug, counts in sorted(by_class.items(), key=lambda kv: str(names.get(kv[0], kv[0])))
    ]

    tier_a = _tier_a(raw.get("tier_a"))

    stamp = raw.get("generated_at")
    return DatasetStatus(
        available=True,
        snapshot_path=str(snapshot),
        generated_at=stamp if isinstance(stamp, str) else None,
        age_seconds=_age_seconds(stamp if isinstance(stamp, str) else None),
        project=raw.get("project") if isinstance(raw.get("project"), str) else None,
        total=int(raw.get("total") or 0),
        decided=int(raw.get("decided") or 0),
        percent=float(raw.get("percent") or 0.0),
        null_annotations=int(raw.get("null_annotations") or 0),
        mismatches=int(raw.get("mismatches") or 0),
        classes=classes,
        by_distance=by_distance,
        by_split=by_split,
        sessions=_sessions(raw.get("sessions")),
        labeling_backlog=_backlog(by_cell, by_class, names),
        tier_a_target=tier_a["target"],
        tier_a_remaining=tier_a["remaining"],
        tier_a_cells_under_target=tier_a["cells_under_target"],
        tier_a_cells=tier_a["cells"],
    )
