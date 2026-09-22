#!/usr/bin/env python
"""Which generation a training run is for: its dataset, its class list, and its names.

The trainer used to be v2's, with the export directory, the 8-class roster and the
`scanncart-grocery-v2` names baked in as module constants. Training v1 then meant either editing
that file or copying it - and a copy is how two generations' "same" checks drift apart, which is
the failure the roster guard exists to catch in the first place. So: one spec per generation, and
the trainer reads it.

    v1  seven classes, from the export of Roboflow's `scanncart-grocery` version 1
    v2  eight classes, from the export of `snc-grocery` version N (`generate_version.py`)

Three fields differ for a reason worth keeping:

**classes.** A trained model's outputs are in whatever order its dataset declared, so an export is
judged against *its own generation's* names. Judging v1 against the 8-name roster would refuse to
train a set that is correct: Palmolive Naturals Bar Soap 85g is the class v2 adds, and v1 never
had it. `app/roster.py` keeps the same split from the runtime's side ("rows 1-7 are v1's names
copied exactly ... and Palmolive is the one class v2 adds"), so this side's v1 list is a strict
subset of its v2 list. The honest consequence is that a v1 weight cannot predict Palmolive at all -
which is what the app reports about it (`roster.class_list_problems`, the third finding), and why
`train_model.check_export` says so out loud rather than passing it in silence.

**manifest.** Distance is a Roboflow *tag* and a YOLO export carries none, so the per-distance
breakdown reads a filename -> distance map from the dataset tooling's manifest. v2's set is staged
in the three distances and has one; v1 predates the tagging and has no distance axis at all. `None`
therefore means "this generation has no such axis" - a different statement from "the manifest went
missing", and the one that gets its own sentence, because a section that silently vanishes reads as
"every distance passed".

**resize_mode.** The requirement a locally trained `.pt` must be run with, recorded beside the
weights because nothing else can know it: a checkpoint stores the training run, not the dataset
geometry, the filename is a convention, and `auto`'s format heuristic answers the *wrong* geometry
for these weights. v2's comes from `generate_version.REQUIRED_RESIZE_MODE`, because that block *is*
the preprocessing that produced its export. v1's is frozen rather than derived from the same
constant: its version already exists and will never be regenerated, so binding it to a value that
describes v2 would let a later change to v2's preprocessing silently rewrite v1's requirement. It
was measured from the export - all 1,815 frames are 640x640 with no constant border, i.e. stretched
rather than fitted with padding - and `check_export` re-measures the frames in hand on every run,
so the claim is checked against the data rather than trusted from this comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from generate_version import REQUIRED_RESIZE_MODE
from label_classes import SLUG_TO_CLASS
from workspace import DEFAULT_OUT
from workspace import WORKSPACE as DATASET_ROOT


@dataclass(frozen=True)
class Generation:
    """One generation's dataset, the classes it must declare, and the names its artifacts carry."""

    name: str
    classes: tuple[str, ...]
    export_dir: Path
    # Where each image's distance is recorded, or None when the set has no distance axis.
    manifest: Path | None
    resize_mode: str
    # The Roboflow project a version would be generated from, and where `--download` fetches it.
    roboflow_project: str

    @property
    def weight_name(self) -> str:
        """`models/<this>.pt` - the drop-in's filename, and the picker's key (MODEL_TRAINING.md 8.2)."""
        return f"scanncart-grocery-{self.name}.pt"

    @property
    def run_name(self) -> str:
        """The ultralytics run directory under the run project, and the `--val` run's stem."""
        return f"scanncart-grocery-{self.name}"

    @property
    def label(self) -> str:
        """How this generation is named in a sentence, e.g. for the record's `source`."""
        return f"{self.roboflow_project} ({self.name})"


# The tools' copy of the 8-name roster, derived rather than retyped: `label_classes.SLUG_TO_CLASS`
# is what the dataset pipeline already uses for continuity, and the drift that matters - against
# the runtime's `app/roster.py` - is held in step by `tests/test_roster.py` on that copy.
V2_CLASSES: tuple[str, ...] = tuple(SLUG_TO_CLASS.values())

# v1's seven, in the order its Roboflow project declares them. Retyped rather than derived as
# "the roster minus Palmolive": a set difference is right only while that subtraction happens to
# name the one class v2 added, and a re-spelling in one file would then silently change what this
# dataset is expected to declare - the exact class of failure a hand-copied contract needs to fail
# loudly on instead.
V1_CLASSES: tuple[str, ...] = (
    "555 sardines 155grams",
    "Bear Brand Fortified Powdered Milk 33g",
    "Milo Chocolate Drink 22g Sachet",
    "century_tuna_flakes_in_oil_155_grams",
    "lucky_me_pancit_canton_calamansi_flavor",
    "safeguard_pure_white_60g",
    "silver_swan_sukang_puti_200ML",
)

V1 = Generation(
    name="v1",
    classes=V1_CLASSES,
    # The hand-downloaded export, ingested here as-is (`data.yaml`, three splits, no tags). Named
    # after the project rather than `export-v1`, which is where `--download` would put a fresh one.
    export_dir=DATASET_ROOT / "scanncart-grocery-v1",
    manifest=None,
    resize_mode="stretch",
    roboflow_project="scanncart-grocery",
)

V2 = Generation(
    name="v2",
    classes=V2_CLASSES,
    export_dir=DATASET_ROOT / "export-v2",
    manifest=DEFAULT_OUT / "manifest.json",
    resize_mode=REQUIRED_RESIZE_MODE,
    roboflow_project="snc-grocery",
)

GENERATIONS: dict[str, Generation] = {g.name: g for g in (V1, V2)}

# The generation whose names the tool's own defaults carry: `--install` with nothing said installs
# v2, which is the generation the app is being built towards.
DEFAULT = V2


def get(name: str) -> Generation:
    """The spec for `name`, or a SystemExit naming the ones that exist.

    Exits rather than raising a KeyError because every caller is a CLI path: an unknown
    `--generation` is a typing mistake, and a traceback would not say which values are real.
    """
    try:
        return GENERATIONS[name]
    except KeyError:
        raise SystemExit(
            f"unknown generation {name!r} - known: {', '.join(sorted(GENERATIONS))}"
        ) from None


def added_over(earlier: Generation, later: Generation) -> tuple[str, ...]:
    """Classes `later` declares that `earlier` does not - Palmolive, for v1 -> v2.

    Used to say *which* classes a weight can never predict, rather than only that it cannot
    predict all of them: the fact is per-class, and "Palmolive" is the actionable half of it.
    """
    have = set(earlier.classes)
    return tuple(name for name in later.classes if name not in have)
