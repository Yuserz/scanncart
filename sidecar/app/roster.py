"""The product classes the app expects a weight to predict, and what to say when it doesn't.

**One roster per generation, and the weight decides which one applies.** v1's export declares seven
classes; v2's declares eight, which is those same seven with Palmolive added. Neither number is a
fact about the app - it belongs to the dataset that trained the head - so a single fixed list is
wrong in both directions. Held to v2's eight, the v1 weights *this app runs today* carry a
permanent "cannot predict Palmolive" finding that no action clears, and a warning that cannot be
acted on is how an operator learns to ignore the ones that can. Held to v1's seven, a v2 head
that lost a class reads as healthy. `resolve_roster` picks by the evidence available: the record's
`generation` when one was written beside the weight, the model's own class list otherwise.

Two facts about this app make this module necessary rather than decorative.

**The class list is a property of the *weights*, and nothing else knows it.** A `.pt` or `.onnx`
records no roster, the filename is only a convention, and the export that produced it is gone by
the time someone runs it. So a model trained against a project whose class list had distances in
it - `palmolive close` / `palmolive mid` / `palmolive far` instead of one `palmolive` - loads
without complaint and predicts 24 classes. Every box then comes back under a label the roster does
not contain, the item log fills with near-duplicates of one product, and *nothing errors*: a class
name records none of this. (The dataset tools refuse to *create* such a set - `clean_v2.py
sanity`, `label_classes.py`, `generate_version.py --yes`, `train_model.check_export` - which is
the other half of the same guard. This is the half that catches a weight arriving from anywhere
else.)

**It cannot import the tools that own the roster.** `sidecar/tools/` is deliberately outside the
runtime - the PRD's offline promise is that the product needs no network, no API key and no
dataset pipeline - so `label_classes.SLUG_TO_CLASS` and the per-generation lists in
`generations.py` are not available here. The names are therefore a second copy, per generation,
and a **drift guard in `sidecar/tests/test_roster.py`** asserts each of these lists equals the
tools' list for the same generation and that the predicate here agrees with
`label_classes.distance_tokens_in` on the same names. That is this repo's standing pattern for a
contract two sides must share by hand (the settings mirrors, the WS schemas): one copy is not
possible, so disagreement has to fail the suite rather than surface as a silent misreading.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# v1's seven, verbatim from the export of Roboflow's `scanncart-grocery` version 1 - the weights
# this app was built on and still runs. Order is that project's class order, which is what a
# trained head indexes by; nothing here depends on it, since every comparison is a set.
V1_ROSTER: tuple[str, ...] = (
    "Bear Brand Fortified Powdered Milk 33g",
    "lucky_me_pancit_canton_calamansi_flavor",
    "555 sardines 155grams",
    "century_tuna_flakes_in_oil_155_grams",
    "silver_swan_sukang_puti_200ML",
    "Milo Chocolate Drink 22g Sachet",
    "safeguard_pure_white_60g",
)

# v2's eight. Written as v1's seven plus the one class v2 adds, which is the convention itself
# (MODEL_TRAINING.md 8.1: rows 1-7 are v1's names copied exactly so labels stay continuous, and
# Palmolive is the addition) rather than a coincidence this file would have to keep re-checking.
# A typo in either name fails the drift guard against the tools' own two lists.
V2_ROSTER: tuple[str, ...] = V1_ROSTER + ("Palmolive Naturals Bar Soap 85g",)

# Keyed by the `generation` a record carries (`train_model.weight_record` writes it) and by the
# generation names the dataset tools use, so a record and this table cannot disagree about what a
# generation is called without the drift guard noticing.
ROSTERS: dict[str, tuple[str, ...]] = {"v1": V1_ROSTER, "v2": V2_ROSTER}

# What a class list that matches nothing is held to, and the one line to change when a later
# generation lands: the newest roster is the app's current expectation, not a fact about v2.
NEWEST = "v2"

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


def resolve_roster(
    names: Iterable[str], generation: str | None = None
) -> tuple[str, tuple[str, ...]]:
    """Which generation's roster `names` is judged against, and that roster.

    Four rules, in order, each resting on weaker evidence than the one before:

    1. **A recorded generation, when it names a roster this app knows.** The record is written by
       the training run (`--install`), so it is the one place the fact exists - and it is the only
       thing that can settle the case the vocabulary cannot. v2's eight minus Palmolive *is* v1's
       seven, so a v2-trained head that cannot predict Palmolive is indistinguishable from v1's
       weights by its class list alone; those want opposite readings, and only the record has the
       answer. Passing the generation is optional: the panel has a record to read, the probe on a
       hand-copied weight may not.
    2. **A class list that is exactly a known roster.** Whatever a record says or does not say, a
       model that predicts exactly v1's seven *is* a v1 model - which is what keeps the installed
       v1 weight clean even with no record beside it, and what makes every future generation work
       here without an edit: its own list is the evidence.
    3. **The closest roster**, fewest names outside it first and then fewest of its names missing.
       A partial list - a head that lost a class, or one carrying a label from another project -
       is still recognisably one generation's roster.
    4. **The newest**, which covers both a list that matches nothing (a stock COCO model, say) and
       an empty one: no names is not evidence of an older generation, and the app's current
       expectation is the honest yardstick for a model nobody can place. Callers gate on the empty
       list anyway - `class_list_problems([])` is a verdict about a class list nobody has seen.

    Pure and total, so it is tested against hand-written lists rather than a loaded model.
    """
    have = [str(n) for n in names]
    if generation in ROSTERS:
        return generation, ROSTERS[generation]
    if not have:
        return NEWEST, ROSTERS[NEWEST]
    seen = set(have)
    exact = [name for name, roster in ROSTERS.items() if seen == set(roster)]
    if exact:
        # Newest wins a tie, and a tie here is two generations declaring the same names.
        name = max(exact, key=lambda name: list(ROSTERS).index(name))
        return name, ROSTERS[name]
    ranked = min(
        ROSTERS.items(),
        key=lambda item: (
            len(seen - set(item[1])),                      # names this roster does not know
            len(set(item[1]) - seen),                      # roster names the model lacks
            -list(ROSTERS).index(item[0]),                 # then the newest
        ),
    )
    return ranked[0], ranked[1]


def class_list_problems(
    names: Iterable[str], generation: str | None = None
) -> list[str]:
    """What is wrong with this weight's class list, as sentences an operator can act on.

    Judged against the roster of the generation the weight belongs to (`resolve_roster`), because
    "cannot predict N of the roster" is only meaningful relative to the generation that declared
    it: v1's seven are complete for v1, and v2's eight are what a v2 head is held to.

    Pure and total, so it is tested against hand-written class lists rather than a loaded model.
    Three findings, worst first, because the order is also the reading order in the panel:

    1. **A distance in a name** - the 24-output failure. Reported with the word that matched, since
       "which class" is not the question here; "why is that a problem" is.
    2. **Names outside the roster** - any other stray output, e.g. a project id mix-up. Told
       separately from (1) so advice about distance is not given for a class that has none.
    3. **Roster names missing** - the other direction, and the quieter one: a model that can only
       ever predict 6 of 8 products passes every check above, because nothing it predicted was
       ever wrong.
    """
    have = [str(n) for n in names]
    generation, roster = resolve_roster(have, generation)
    problems: list[str] = []

    tainted = {n: words for n, words in ((n, distance_tokens_in(n)) for n in have) if words}
    if tainted:
        problems.append(
            f"{len(tainted)} of {len(have)} class name(s) carry a distance, so this model "
            "predicts one class per product-and-distance instead of one per product: "
            + ", ".join(f"{n!r} ({'/'.join(w)})" for n, w in sorted(tainted.items()))
            + ". Distance is a tag on the training image (MODEL_TRAINING.md 8.1), never a class - "
            "so this is a project whose class list was split by distance. Retrain from a version "
            "generated with one class per product; no setting here fixes it."
        )

    unexpected = sorted(n for n in have if n not in roster and n not in tainted)
    if unexpected:
        problems.append(
            f"{len(unexpected)} class(es) are not in this app's {generation} roster: "
            + ", ".join(repr(n) for n in unexpected)
            + ". Their boxes are drawn and logged - nothing filters them out by default - but the "
            "item log will not match what this app is built to show."
        )

    # Deliberately silent when a distance is present, even though it is technically true (a
    # distance-split model matches no roster name at all). It would be the same fact in the other
    # direction, and a reader cannot act on "8 missing" until the list is a roster - so on the one
    # input where the distance sentence matters most, the second sentence would only dilute it.
    missing = [] if tainted else [n for n in roster if n not in set(have)]
    if missing:
        problems.append(
            f"this model cannot predict {len(missing)} of the {len(roster)} {generation} roster "
            "classes: "
            + ", ".join(repr(n) for n in missing)
            + ". Nothing it *does* predict is wrong, which is why this is easy to miss; those "
            "products will simply never be logged."
        )

    return problems
