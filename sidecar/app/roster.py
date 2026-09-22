"""The 8 product classes the app expects a weight to predict, and what to say when it doesn't.

Two facts about this app make this module necessary rather than decorative.

**The class list is a property of the *weights*, and nothing else knows it.** A `.pt` or `.onnx`
records no roster, the filename is only a convention, and the export that produced it is gone by
the time someone runs it. So a model trained against a project whose class list had distances in
it - `palmolive close` / `palmolive mid` / `palmolive far` instead of one `palmolive` - loads
without complaint and predicts 24 classes. Every box then comes back under a label the roster does
not contain, the item log fills with near-duplicates of one product, and *nothing errors*: a class
name records none of this. (The dataset tools refuse to *create* such a set - `clean_v2.py
sanity`, `label_classes.py`, `generate_version.py --yes`, `train_model.check_export` - which is the
other half of the same guard. This is the half that catches a weight arriving from anywhere else.)

**It cannot import the tools that own the roster.** `sidecar/tools/` is deliberately outside the
runtime - the PRD's offline promise is that the product needs no network, no API key and no
dataset pipeline - so `label_classes.SLUG_TO_CLASS` is not available here. The names are therefore
a second copy, and a **drift guard in `sidecar/tests/test_roster.py`** asserts this list equals the
tools' roster and that the predicate here agrees with `label_classes.distance_tokens_in` on the
same names. That is this repo's standing pattern for a contract two sides must share by hand
(the settings mirrors, the WS schemas): one copy is not possible, so disagreement has to fail the
suite rather than surface as a silent misreading.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# The 8 canonical class names, verbatim from MODEL_TRAINING.md 8.1 (which is the source of truth
# for the spelling; rows 1-7 are v1's names copied exactly so labels stay continuous, and
# Palmolive is the one class v2 adds). Order is the selector's order there, and irrelevant here.
ROSTER: tuple[str, ...] = (
    "Bear Brand Fortified Powdered Milk 33g",
    "lucky_me_pancit_canton_calamansi_flavor",
    "555 sardines 155grams",
    "century_tuna_flakes_in_oil_155_grams",
    "silver_swan_sukang_puti_200ML",
    "Milo Chocolate Drink 22g Sachet",
    "safeguard_pure_white_60g",
    "Palmolive Naturals Bar Soap 85g",
)

# The distance words, as they appear in a class *name* when something has gone wrong. Distance is
# a tag on the image and a cell in the coverage reports - never a category - so a name carrying
# one means the head was trained per product-and-distance.
#
# Whole tokens, so a legitimate name cannot trip it: "Farmer's Choice" tokenises to `farmer`, not
# `far`. Kept in step with `label_classes.DISTANCE_TOKENS` by the drift guard.
DISTANCE_TOKENS = frozenset({"close", "closeup", "mid", "middle", "far", "near", "distance"})


def distance_tokens_in(name: str) -> list[str]:
    """The distance words a class name carries, if any."""
    tokens = re.split(r"[^a-z0-9]+", str(name).lower())
    return sorted({t for t in tokens if t in DISTANCE_TOKENS})


def class_list_problems(names: Iterable[str]) -> list[str]:
    """What is wrong with this weight's class list, as sentences an operator can act on.

    Pure and total, so it is tested against hand-written class lists rather than a loaded model.
    Three findings, worst first, because the order is also the reading order in the panel:

    1. **A distance in a name** - the 24-output failure. Reported with the word that matched, since
       "which class" is not the question here; "why is that a problem" is.
    2. **Names outside the roster** - any other stray output, e.g. a project id mix-up. Told
       separately from (1) so advice about distance is not given for a class that has none.
    3. **Roster names missing** - the other direction, and the quieter one: a model that can only
       ever predict 6 of the 8 products passes every check above, because nothing it predicted was
       ever wrong.
    """
    have = [str(n) for n in names]
    problems: list[str] = []

    tainted = {n: words for n, words in ((n, distance_tokens_in(n)) for n in have) if words}
    if tainted:
        problems.append(
            f"{len(tainted)} of {len(have)} class name(s) carry a distance, so this model "
            "predicts one class per product-and-distance instead of one per product: "
            + ", ".join(f"{n!r} ({'/'.join(w)})" for n, w in sorted(tainted.items()))
            + ". Distance is a tag on the training image (MODEL_TRAINING.md 8.1), never a class - "
            "so this is a project whose class list was split by distance. Retrain from a version "
            "generated with the 8 product names; no setting here fixes it."
        )

    unexpected = sorted(n for n in have if n not in ROSTER and n not in tainted)
    if unexpected:
        problems.append(
            f"{len(unexpected)} class(es) are not in this app's roster: "
            + ", ".join(repr(n) for n in unexpected)
            + ". Their boxes are drawn and logged - nothing filters them out by default - but the "
            "item log will not match what this app is built to show."
        )

    # Deliberately silent when a distance is present, even though it is technically true (a
    # distance-split model matches no roster name at all). It would be the same fact in the other
    # direction, and a reader cannot act on "8 missing" until the list is a roster - so on the one
    # input where the distance sentence matters most, the second sentence would only dilute it.
    missing = [] if tainted else [n for n in ROSTER if n not in set(have)]
    if missing:
        problems.append(
            f"this model cannot predict {len(missing)} of the 8 roster classes: "
            + ", ".join(repr(n) for n in missing)
            + ". Nothing it *does* predict is wrong, which is why this is easy to miss; those "
            "products will simply never be logged."
        )

    return problems
