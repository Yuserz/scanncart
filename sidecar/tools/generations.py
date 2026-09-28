#!/usr/bin/env python
"""Which generation a training run is for: its dataset, its class list, and its names.

The trainer used to be v2's, with the export directory, the 8-class roster and the
`scanncart-grocery-v2` names baked in as module constants. Training v1 then meant either editing
that file or copying it - and a copy is how two generations' "same" checks drift apart, which is
the failure the roster guard exists to catch in the first place. So: one spec per generation, and
the trainer reads it.

    v1  seven classes, from the export of Roboflow's `scanncart-grocery` version 1
    v2  the same seven, from the merged local dataset (`build_dataset.py`)

Three fields differ for a reason worth keeping:

**classes.** A trained model's outputs are in whatever order its dataset declared, so an export is
judged against *its own generation's* names - which is a per-generation list rather than one shared
roster because the two sets were eight-versus-seven until Palmolive was dropped. Judging v1 against
a list that had Palmolive in it would refuse to train a set that is correct, and judging v2 against
v1's seven would pass a head that had lost a class. They are the same seven names today (`v1`'s list
is spelled out below and `v2`'s is derived from `label_classes.SLUG_TO_CLASS`, which is where the
continuity with the 1,815 v1 images actually lives), so the two entries agree by construction rather
than by luck - and the moment a product is added they stop agreeing, which is exactly the case this
dimension exists for. `app/roster.py` keeps the same per-generation split from the runtime's side.

The list is **ordered**, and that order is a fact this file owns rather than one anything
re-derives. A label row's `cls` column is a *position* in it, so two datasets declaring the same
names in different orders are two different labelings - which is not hypothetical here: v1 and v2
declare the same seven names and differ only in order. So `build_dataset.CANONICAL_NAMES` reads
`V2.classes` instead of a second `tuple(SLUG_TO_CLASS.values())` that agreed only while the two
expressions did, the merged set records which generation it was built for, and `dataset_doctor`
asks `order_of` which generation a set's own declared names index rather than assuming the one it
was told - because judging a v1 set against v2's order is a verdict about the wrong expectation,
and it names a fix that would corrupt the labels.

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
from workspace import DEFAULT_OUT, MANIFEST_NAME
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
# "the roster minus one class": a set difference is right only while that subtraction happens to
# name the class the other generation adds, and a re-spelling in one file would then silently change
# what this dataset is expected to declare - the exact class of failure a hand-copied contract needs
# to fail loudly on instead. (v2's list below *is* derived from the shared slug mapping, because for
# v2 the mapping itself is the contract - the same table the uploader tags images from.)
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
    manifest=DEFAULT_OUT / MANIFEST_NAME,
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


def order_of(names) -> Generation | None:
    """The generation whose class list is exactly `names`, **in that order** - or None.

    The question "which generation's order do these labels index?", asked of a dataset's own
    declared names. Only an exact match counts: the two generations declare the same seven names
    today and differ only in order, so this is the one read that tells them apart - and a near miss
    (one name short, one name renamed) is a *membership* question, which `check_export` judges.
    Answering "closest roster" here would put a set of labels back in the wrong generation's mouth,
    which is the failure this exists to prevent.
    """
    wanted = tuple(names)
    for generation in GENERATIONS.values():
        if wanted == generation.classes:
            return generation
    return None


def added_over(earlier: Generation, later: Generation) -> tuple[str, ...]:
    """Classes `later` declares that `earlier` does not.

    Used to say *which* classes a weight can never predict, rather than only that it cannot
    predict all of them: the fact is per-class, and the name is the actionable half of it. Empty
    for v1 -> v2 today, because both declare the same seven names (Palmolive was the entry here
    until it was dropped) - which is the honest reading, not a broken check: there is no class a
    v1 weight is missing that a v2 one knows.
    """
    have = set(earlier.classes)
    return tuple(name for name in later.classes if name not in have)
