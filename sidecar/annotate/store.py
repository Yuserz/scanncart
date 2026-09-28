"""The annotation app's own on-disk store: where the work lives, and the three states.

`sidecar/annotate/` is an **authoring tool**, not part of the product. The packaged app never
imports it, it is never started by `desktop/src/main`, and the sidecar's own `run.py` does not know
it exists. That is what lets it import from both sides of the line this repo otherwise keeps - the
runtime's detector (`app/inference.py`) for suggestions and the dataset tools' roster
(`label_classes.py`) for the class list - without weakening either. Nothing in `app/` imports this
package, and nothing here is on the packaging path.

Two things about that boundary are deliberate:

**The class list is not copied.** `label_classes.SLUG_TO_CLASS` already exists, `roster.py` already
mirrors it by hand with a drift guard, and a *third* copy for the annotator would be a third chance
to disagree about a name that labels are matched against exactly. So this imports it.

**The labels are YOLO txt, because that is what the trainer eats.** The dataset build is then a
directory copy rather than a translation, and there is no annotation format whose semantics have to
be re-derived later. Rows are `cls cx cy w h`, normalized, with `cls` indexing the canonical class
order - the same order `label_classes.SLUG_TO_CLASS` declares, which is also what
`generations.V2_CLASSES` and `app/roster.V2_ROSTER` hold.

### The three states, and why an empty file is not an absent one

| state | on disk | meaning |
|---|---|---|
| not labeled | **no label file** | outstanding work |
| deliberately empty | **an empty `.txt`** | a null annotation: "there is no item here", a decision |
| labeled | `.txt` with rows | boxes |

This is the local form of the distinction the Roboflow path had to read off the *type* of an API
field (`annotations: []` vs `{"count": 0}` vs `{"count": n}`), and it is the one a naive sweep
destroys: "the file is empty, so it is not labeled" re-opens the 50 hard negatives, whose whole
purpose is to be training images with no boxes. Nulls are also the frames that make the
false-positive half of the acceptance numbers measurable, so losing them costs evidence, not just
tidiness.

Provenance is the second half. `provenance.json` records per image which provider proposed the
boxes that are on disk and whether anyone touched them, which is what keeps "a machine drew this and
nobody looked" separable from "someone reviewed it" *without* the Roboflow tag the cloud path used.
It is written by `write()`, i.e. **on save and never on suggestion** - a frame that has only been
suggested is still outstanding work.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from label_classes import DISTANCE_ORDER, SLUG_TO_CLASS, SPLIT_NAMES
from workspace import MANIFEST_NAME, PROVENANCE_NAME, SPLITS_NAME

# The canonical class order. Index in this tuple is the `cls` column of every label row, so it is
# also what `data.yaml`'s `names` has to be written in - an order mismatch here is the failure
# `train_model.check_export` cannot see (it compares names by membership).
CLASS_NAMES: tuple[str, ...] = tuple(SLUG_TO_CLASS.values())
CLASS_SLUGS: tuple[str, ...] = tuple(SLUG_TO_CLASS)
SLUG_BY_NAME: dict[str, str] = {name: slug for slug, name in SLUG_TO_CLASS.items()}
CLASS_INDEX: dict[str, int] = {slug: i for i, slug in enumerate(CLASS_SLUGS)}

# The pseudo-class: a tag and a batch key with nothing to annotate. Its frames are the null
# annotations, so they are the one "class" whose every label file must be empty.
PSEUDO_CLASS_SLUGS = {"negative"}

# How a *missing* value is named rather than a value: the distance of a hard negative, the split of
# a frame no plan has reached yet. `Frame.cell` already spells a missing distance this way (it is
# the word the Roboflow writer uses for one), and a filter parameter adopts the same token because
# an empty parameter has to keep meaning "do not filter at all" - which would otherwise make the
# negatives, and every unplanned frame, impossible to ask for.
UNRECORDED = "unknown"

FRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")

# The provenance of the boxes that are on disk. `unknown` is a real value rather than a fallback
# to "human": a label file written by an older tool, or by hand outside this app, must not be
# counted as machine-drawn - and `machine_only` is what the acceptance gate reads.
PROVIDER_HUMAN = "human"
PROVIDER_UNKNOWN = "unknown"

# The order a **review pass** walks the worklist in, which is the acceptance gate's own priority
# (`build_dataset`/`accept_v2` require zero machine-only decisions in `valid` and `test`): unread
# boxes in `test` make the acceptance number a measurement of the annotator, so they go first,
# then `valid`; `train` is cheap until those two are clean. Unplaced frames are last *and named* -
# no split entry yet is unplanned work rather than a fourth split.
REVIEW_SPLIT_ORDER: tuple[str, ...] = ("test", "valid", "train", "")

# The two splits a human pass is *about*: the pair `build_dataset`/`accept_v2` require to be clear
# of machine-only decisions before the acceptance number measures the model rather than the
# annotator. Named as the gate names it (`accept_v2.GATE_SPLITS`, which a drift guard pins this
# against), and deliberately not derived from `label_classes.SPLIT_NAMES` - the three splits are
# the axis, this pair is a verdict about two of them.
GATE_SPLITS: tuple[str, ...] = ("test", "valid")

# The distance axis in the order a pass should work it: farthest first, because that is where v1
# finds least and where the per-distance evidence is thinnest. Reversed from `DISTANCE_ORDER`
# rather than spelled again, so a distance the axis gains is ranked here by existing.
DISTANCE_WORK_ORDER: tuple[str, ...] = tuple(reversed(DISTANCE_ORDER))


def review_rank(split: str) -> int:
    """Where one split sits in a review pass. A split nobody planned for sorts after unplaced."""
    return (
        REVIEW_SPLIT_ORDER.index(split)
        if split in REVIEW_SPLIT_ORDER
        else len(REVIEW_SPLIT_ORDER)
    )


def verdict_priority(frame: Frame) -> tuple[int, int]:
    """How much the v2 verdict rests on one frame, most first: its distance, then its crowding.

    Both are places the acceptance evidence is thin. A `far`/`mid` frame is an instance the
    per-distance grid can quote, and `far` is where v1 finds nothing at all - the axis v2 exists
    for. Within a distance, a frame holding two or more boxes comes first, because the crowded
    counter is counted over exactly those frames. A `close` frame with one box is the ordinary
    case: it sorts last, and is still in the pass.
    """
    distance = (
        DISTANCE_WORK_ORDER.index(frame.distance)
        if frame.distance in DISTANCE_WORK_ORDER
        else len(DISTANCE_WORK_ORDER)
    )
    return (distance, 0 if frame.boxes >= 2 else 1)


def review_worklist(frames: list[Frame], splits: dict[str, str]) -> list[Frame]:
    """The decisions a weight made that nobody has looked at, in the order that matters.

    Pure, and a *selection* rather than a sort: everything already reviewed is dropped, because a
    review pass that walked the whole set again would be the labeling pass over again. `test` first,
    then `valid`; within a split the frames the verdict is thinnest at (`verdict_priority`), then
    the cells in the worklist's own order (`frames()`), so the pass reads the same way the labeling
    pass did.
    """
    return sorted(
        (frame for frame in frames if frame.machine_only),
        key=lambda frame: (
            review_rank(splits.get(frame.name, "")),
            *verdict_priority(frame),
            frame.cell,
            frame.name,
        ),
    )


def suggested_nothing(record: dict) -> bool:
    """Whether a provider was *asked* about this frame and proposed no boxes.

    The strict reading of a recorded suggestion, and the one the draw list needs: an empty list is a
    proposal of nothing, while a frame with no `suggestion` key at all was never put to a provider
    and says nothing about the model. A truthiness test reads those two as the same thing, which is
    the mistake this function exists to not make - the difference between evidence about the model
    and outstanding work is exactly the difference between them.
    """
    suggestion = record.get("suggestion")
    return isinstance(suggestion, dict) and not (suggestion.get("boxes") or [])


def draw_worklist(frames: list[Frame], splits: dict[str, str], provenance: dict) -> list[Frame]:
    """The far frames in the gate splits a weight was asked about and found nothing in.

    `far` because that is the axis this dataset exists for and the one the per-distance grid quotes,
    and the gate splits because drawing a `train` frame is real work with no effect on the number
    the pass is running to make checkable. Kept in the order given (the store's own), so the list
    reads down the app's list rather than in some order of its own.

    The second half of the human pass, and served two ways from here on purpose: `human_pass.py`
    prints it as the checklist's draw section and `GET /api/frames?draw=1` hands it to the page, so
    "which frames is the pass about" has one answer rather than one per reader.
    """
    return [
        frame
        for frame in frames
        if frame.distance == "far"
        and splits.get(frame.name, "") in GATE_SPLITS
        and frame.state == "unlabeled"
        and suggested_nothing(provenance.get(frame.name) or {})
    ]


def null_worklist(frames: list[Frame], splits: dict[str, str]) -> list[Frame]:
    """The hard negatives in the gate splits that have no null yet.

    `state == "unlabeled"` rather than "no boxes": an empty label file is the *decision* that there
    is no item here, which is the whole reason the negatives are a staged set of their own, so a
    negative already marked null is finished work and must leave the list.
    """
    return [
        frame
        for frame in frames
        if frame.slug in PSEUDO_CLASS_SLUGS
        and splits.get(frame.name, "") in GATE_SPLITS
        and frame.state == "unlabeled"
    ]


def gate_frames(frames: list[Frame], splits: dict[str, str]) -> list[Frame]:
    """The frames in `frames` that sit in the two splits the acceptance gate measures.

    The gate's pair is `GATE_SPLITS` - `accept_v2`'s own, pinned equal by a test - and this is the
    one place the annotator turns a list of frames into "the ones that block the number". Filtering
    the review worklist with it is the count `make human-pass` prints in its header, and the gate
    readout `GET /api/frames` serves is the same call, so the page's finish line and the document's
    cannot be two numbers that disagree - and neither can the two places inside the document.

    Order-preserving, so a caller can still list them the way the pass walks them.
    """
    return [frame for frame in frames if splits.get(frame.name, "") in GATE_SPLITS]


def filter_values(raw: str) -> frozenset[str] | None:
    """One filter parameter's worth of values, or None when nothing was asked for.

    Comma-separated rather than repeated, because the two splits a pass works are asked about
    *together* (`test,valid`) and a single-value parameter could not say that.
    """
    wanted = {part.strip() for part in (raw or "").split(",") if part.strip()}
    return frozenset(wanted) if wanted else None


def matches_filter(wanted: frozenset[str] | None, value: str) -> bool:
    """Whether a frame's own value passes a filter. A missing value is `UNRECORDED`, not `''`."""
    return wanted is None or (value or UNRECORDED) in wanted


@dataclass(frozen=True)
class Box:
    """One box, in the units a YOLO row carries: class index plus a normalized centre and size."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def as_row(self) -> str:
        return f"{int(self.cls)} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def as_dict(self) -> dict:
        """The wire shape: normalized values *and* the four CSS percentages the page draws with.

        The percentages are computed here rather than in the browser on purpose. The alternative is
        a second copy of the same arithmetic in JavaScript (`left = (cx - w/2) * 100`), which is the
        exact shape of mirror this repo keeps drift guards for - and a guard is a weaker answer than
        not having the copy. So the page assigns numbers it was handed, and the one implementation
        of "where does this box go" is tested by the suite that already runs.
        """
        return {
            "cls": int(self.cls),
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h,
            **self.as_percent(),
        }

    def as_percent(self) -> dict:
        """The box as CSS percentages of the frame, which is how the overlay is positioned."""
        return {
            "left": (self.cx - self.w / 2) * 100,
            "top": (self.cy - self.h / 2) * 100,
            "width": self.w * 100,
            "height": self.h * 100,
        }

    def valid(self) -> bool:
        """Every value a number, inside the frame, and with a real extent.

        Checked before anything is written, because a normalized coordinate outside 0..1 is not a
        label that "looks wrong" later - it is a box the trainer silently clips or drops, and the
        frame then reads as labeled while contributing nothing.
        """
        try:
            values = [float(self.cx), float(self.cy), float(self.w), float(self.h)]
        except (TypeError, ValueError):
            return False
        if not all(0.0 <= v <= 1.0 for v in values):
            return False
        return bool(self.w > 0 and self.h > 0) and 0 <= int(self.cls) < len(CLASS_NAMES)


def box_from_row(row: str) -> Box | None:
    """One label line, or None when it is not a box at all."""
    parts = row.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return None
    return Box(cls=cls, cx=cx, cy=cy, w=w, h=h)


@dataclass(frozen=True)
class Frame:
    """One image the tool can open, and what is known about it."""

    name: str
    slug: str
    distance: str
    session: str
    image: Path
    label: Path
    state: str          # "unlabeled" | "null" | "labeled"
    boxes: int
    machine_only: bool  # a provider drew these and nobody touched them
    provider: str

    @property
    def cell(self) -> str:
        """`slug|distance`, the key the coverage tables use (`label_progress.by_cell`).

        `unknown` for the distance rather than an empty string: that is the spelling the Roboflow
        writer already uses for a frame with no distance (the hard negatives), and the panel keys
        its worklist off this string - so a local snapshot that spelled the same cell differently
        would split one bucket into two.
        """
        return f"{self.slug}|{self.distance or UNRECORDED}"


@dataclass(frozen=True)
class Counts:
    """What a set of frames adds up to. One shape, so the panel and the app cannot disagree."""

    total: int
    labeled: int
    nulls: int
    unlabeled: int
    machine_only: int

    @property
    def decided(self) -> int:
        """Frames carrying a decision - boxes *or* a deliberate null."""
        return self.labeled + self.nulls

    @property
    def reviewed(self) -> int:
        """Decisions a human has touched, i.e. everything a machine did not draw unattended."""
        return self.decided - self.machine_only


def counts(frames: list[Frame]) -> Counts:
    return Counts(
        total=len(frames),
        labeled=sum(1 for f in frames if f.state == "labeled"),
        nulls=sum(1 for f in frames if f.state == "null"),
        unlabeled=sum(1 for f in frames if f.state == "unlabeled"),
        machine_only=sum(1 for f in frames if f.machine_only),
    )


class LabelStore:
    """The staged images plus the annotation tree beside them.

    Two roots, deliberately not one. The images stay exactly as the uploader staged them (every
    reader that predates this app - `plan_split`, `staged_coverage`, `audit_v2` - globs those
    directories for images and a stray `.txt` in them is a change none of those tools asked for),
    while every file this app writes lands in a tree of its own:

        <annotations>/<name>.txt            the labels, in the trainer's own format
        <annotations>/provenance.json       who drew what, and what is still machine-only
        <annotations>/classes.json          the class order the `cls` column indexes

    One flat directory rather than a `<slug>/` subdirectory *for the labels*: a staged filename
    already begins with its class (`milo_0007.jpg`), the names are unique across every set by
    construction, and a flat tree is what makes "which frames have a decision" a single `listdir`
    rather than a walk. The images keep the class folders `clean_v2.stage` gave them - `located()`
    is what joins the two, reading the folder off the manifest rather than globbing for it.

    The manifest is the worklist: it is what the dataset tools already wrote, it carries each
    frame's class, distance and session, and reading it (rather than walking the filesystem) is
    what keeps a frame that was parked by `clean_v2 drop` from showing up as work.
    """

    def __init__(self, out: Path, annotations: Path, extras: list[Path] | None = None):
        self.out = Path(out)
        self.annotations = Path(annotations)
        # De-duplicated, and never the staged set itself. Both cases are reachable now that a
        # *default* extras list exists: a caller may add the same directory the default already
        # names, and `--out` can point at what is also passed as an extra. A set listed twice would
        # be walked twice, so every frame in it would appear twice in the worklist and be counted
        # twice in `build_dataset`'s per-split report - a doubling nothing would flag, since the
        # second write of the same frame lands on the same filename.
        seen = {self.out.expanduser().resolve()}
        kept: list[Path] = []
        for candidate in extras or []:
            resolved = Path(candidate).expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            kept.append(Path(candidate))  # the path as given: `/api/config` prints it
        self.extras = kept

    # -- paths -----------------------------------------------------------------

    def manifest_paths(self) -> list[Path]:
        """The staged set's manifest, then each extra's - the order is precedence for a name."""
        return [self.out / MANIFEST_NAME, *[p / MANIFEST_NAME for p in self.extras]]

    def _manifests(self) -> list[tuple[Path, list[dict]]]:
        """Each readable manifest with the directory it sits in, in precedence order.

        The root travels with the entries because a staged set is not flat: `clean_v2.stage` files
        an image under its class (`<out>/<slug>/<name>`) and `upload` reads it back that way, so
        finding one is a two-part question - which set, then which class folder - and answering
        the second half only from the manifest is what keeps this from globbing.
        """
        out: list[tuple[Path, list[dict]]] = []
        for manifest in self.manifest_paths():
            try:
                body = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(body, list):
                out.append((manifest.parent, [e for e in body if isinstance(e, dict)]))
        return out

    def located(self) -> dict[str, Path]:
        """name -> staged image, for every frame the manifests describe.

        Built in one pass rather than looked up per frame, because `frames()` walks all 500-odd
        entries and a per-name manifest scan would turn that into a filesystem crawl. `setdefault`
        gives the first set that has the name precedence, matching `manifest_paths()`'s order.
        """
        found: dict[str, Path] = {}
        for root, entries in self._manifests():
            base = root.expanduser().resolve()
            for entry in entries:
                name = Path(str(entry.get("new_name") or "")).name
                if not name:
                    continue
                slug = str(entry.get("class") or "")
                for candidate in ([root / slug / name] if slug else []) + [root / name]:
                    resolved = candidate.expanduser().resolve()
                    if resolved.is_file() and resolved.is_relative_to(base):
                        found.setdefault(name, resolved)
                        break
        return found

    def image_path(self, name: str) -> Path:
        """Where `name`'s image is, refusing anything that is not inside a staged set.

        A query parameter that reaches the filesystem has to be resolved before it is trusted:
        `../../` and an absolute path both look like a name to `Path`. Two guards, and both are
        load-bearing - the name is reduced to its leaf, so no query string can walk anywhere, and
        every candidate is `resolve()`d and prefix-checked against the root it came from, so a
        symlink inside the workspace cannot point out of it.
        """
        found = self.located().get(Path(name).name)
        return found if found is not None else Path()

    def label_path(self, name: str) -> Path:
        """Where `name`'s label file goes: the image's stem, `.txt`.

        The extension is replaced rather than appended, because ultralytics pairs `foo.jpg` with
        `foo.txt` by stem - a file called `foo.jpg.txt` is one the trainer silently ignores, which
        would leave every frame reading as labeled while the dataset build copies no labels at all.
        """
        return self.annotations / Path(name).with_suffix(".txt").name

    # -- reading ---------------------------------------------------------------

    def _entries(self) -> list[dict]:
        out: list[dict] = []
        for _, entries in self._manifests():
            out += [e for e in entries if e.get("new_name")]
        return out

    def provenance(self) -> dict:
        path = self.annotations / PROVENANCE_NAME
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return body if isinstance(body, dict) else {}

    def read_boxes(self, name: str) -> list[Box]:
        """The boxes on disk. An empty list covers both "no file" and "an empty file" - the state
        is `state_of`, not this, because conflating them is the one mistake this store exists to
        prevent.
        """
        path = self.label_path(name)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        return [box for box in (box_from_row(line) for line in text.splitlines()) if box]

    def state_of(self, name: str) -> str:
        path = self.label_path(name)
        if not path.is_file():
            return "unlabeled"
        return "null" if not path.read_text(encoding="utf-8").strip() else "labeled"

    def frame(self, name: str) -> Frame | None:
        for frame in self.frames():
            if frame.name == name:
                return frame
        return None

    def frames(self) -> list[Frame]:
        """Every staged frame, most-outstanding-first then by name.

        The order is the worklist: a `far` frame the local weight found nothing in is the one a
        human can actually help with, and a session that starts at the top spends its first hour
        on the frames that move the numbers.
        """
        provenance = self.provenance()
        located = self.located()
        out: list[Frame] = []
        for entry in self._entries():
            name = Path(str(entry["new_name"])).name
            image = located.get(name, Path())
            if not image.is_file():
                continue
            label = self.label_path(name)
            record = provenance.get(name) or {}
            out.append(
                Frame(
                    name=name,
                    slug=str(entry.get("class") or ""),
                    distance=str(entry.get("distance") or ""),
                    session=str(entry.get("session") or ""),
                    image=image,
                    label=label,
                    state=self.state_of(name),
                    boxes=len(self.read_boxes(name)),
                    machine_only=bool(record.get("machine_only")),
                    provider=str(record.get("provider") or PROVIDER_UNKNOWN),
                )
            )

        def sort_key(frame: Frame) -> tuple:
            # Distance first: `far` is the axis this dataset exists for, and within a distance the
            # cell with the least work done is the one to open. Undone cells rank above partly-done
            # ones because "nobody opened this" is the state that stalls a project.
            return (
                0 if frame.state == "unlabeled" else 1,
                0 if frame.distance == "far" else 1 if frame.distance == "mid" else 2,
                frame.cell,
                frame.name,
            )

        return sorted(out, key=sort_key)

    # -- writing ---------------------------------------------------------------

    def record_suggestion(self, name: str, boxes: list[Box], provider: str) -> None:
        """Remember what a provider proposed, so a later save can tell whether it was touched.

        Written on *suggestion*, and deliberately not the same field as the label's provenance:
        this is evidence about boxes that are not on disk yet (`machine_only` stays false until a
        save), and keeping the two apart is what makes "accepted the suggestion unchanged" and
        "drew it by hand" distinguishable at all.
        """
        body = self.provenance()
        record = dict(body.get(name) or {})
        record["suggestion"] = {
            "provider": provider,
            "at": _now(),
            "boxes": [box.as_row() for box in boxes],
        }
        body[name] = record
        self._write_provenance(body)

    def write(
        self,
        name: str,
        boxes: list[Box],
        *,
        null: bool = False,
        provider: str = PROVIDER_UNKNOWN,
        confirm: bool = False,
    ) -> dict:
        """Save one frame's annotation - boxes, or a deliberate null - and its provenance.

        `null=True` writes an **empty** file: the decision that there is no item here. It is
        rejected for a frame whose class is a real product (an empty label there is a mistake that
        trains the head against itself) and it is how the hard negatives enter the dataset at all.

        `machine_only` is decided here rather than sent by the caller: the saved rows are compared
        against the suggestion this store last recorded, and only an *unchanged* copy of a
        provider's boxes counts as machine-drawn. A client bug cannot then mark reviewed work as
        unreviewed, or the other way round, and the acceptance gate that reads this field is
        reading a fact about the files instead of a claim about them.

        `confirm=True` is the one thing a review pass needs and the comparison above cannot
        express: a person looked at a weight's boxes, agreed with them, and saved them *without
        touching them* - which is the ordinary outcome of reviewing, and would otherwise leave the
        frame machine-only for ever (the rows still equal the suggestion, so nothing in the files
        changed). It is an assertion only a person makes, like a null: it never alters the rows,
        and it is recorded with a timestamp rather than used to rewrite the comparison. Once
        confirmed, a frame stays reviewed - an unchanged re-save cannot put it back in the gate's
        count - and editing the boxes makes it reviewed the usual way.
        """
        slug = ""
        for entry in self._entries():
            if Path(str(entry.get("new_name"))).name == name:
                slug = str(entry.get("class") or "")
                break
        if slug in PSEUDO_CLASS_SLUGS and boxes:
            raise ValueError(
                f"{name} is a hard negative ({slug}) - it carries no box by definition; "
                "save it as a null instead"
            )
        if slug not in PSEUDO_CLASS_SLUGS and null:
            raise ValueError(
                f"{name} is {slug or 'a staged frame'}, not a hard negative - a null annotation "
                "here would teach the model that item is not in this frame"
            )

        bad = [box.as_row() for box in boxes if not box.valid()]
        if bad:
            raise ValueError(f"refusing to write out-of-frame boxes: {bad[:3]}")

        path = self.label_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [] if null else [box.as_row() for box in boxes]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        os.replace(tmp, path)

        body = self.provenance()
        record = dict(body.get(name) or {})
        suggestion = record.get("suggestion") or {}
        suggested_by = str(suggestion.get("provider") or "")
        unchanged = [str(r) for r in suggestion.get("boxes") or []] == rows
        # Read before the update, so a frame reviewed in an earlier session stays reviewed: the
        # field is monotone in that direction (a confirmation only ever *removes* the machine's
        # authorship) while an edit removes it the ordinary way, by making `unchanged` false.
        already_confirmed = bool(record.get("confirmed_at"))
        record.update(
            {
                "state": "null" if null else "labeled",
                "rows": len(rows),
                # The suggestion's provider only stands for the saved rows when there *is* one: an
                # empty recorded suggestion and a saved null compare equal (`[] == []`), and
                # crediting a provider that proposed nothing would attribute a human's decision.
                "provider": suggested_by if (unchanged and suggested_by) else provider,
                # A null is never machine-only, whatever was recorded: "there is no item here" is
                # an assertion only a person makes (`providers.py` will not mark one), so a
                # provider that also found nothing is agreement, not authorship.
                "machine_only": bool(
                    unchanged and suggested_by and rows and not (confirm or already_confirmed)
                ),
                "saved_at": _now(),
                "image_sha256": _sha256(self.image_path(name)),
            }
        )
        if confirm:
            record["confirmed_at"] = _now()
        body[name] = record
        self._write_provenance(body)
        return record

    def _write_provenance(self, body: dict) -> None:
        self.annotations.mkdir(parents=True, exist_ok=True)
        path = self.annotations / PROVENANCE_NAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def write_classes(self) -> Path:
        """Write the class order the label rows index. Refusing to guess is the point: `cls` is a
        number, so a dataset whose `names` are in another order relabels every frame silently.
        """
        self.annotations.mkdir(parents=True, exist_ok=True)
        path = self.annotations / "classes.json"
        path.write_text(
            json.dumps(
                {
                    "names": list(CLASS_NAMES),
                    "slugs": list(CLASS_SLUGS),
                    "written_at": _now(),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return path

    # -- summaries -------------------------------------------------------------

    def splits(self) -> dict[str, str]:
        """name -> split, from the file `build_dataset.py` captures before going offline.

        Absence is not an error: the annotator works without it (the split is only needed to keep
        the acceptance number honest at *training* time), and a missing file reads as no split
        rather than as a guess the plan forbids.
        """
        for candidate in [self.out / SPLITS_NAME, *[p / SPLITS_NAME for p in self.extras]]:
            try:
                body = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(body, dict):
                return {str(k): str(v) for k, v in body.items()}
        return {}


def summary(frames: list[Frame], splits: dict[str, str] | None = None) -> dict:
    """The snapshot pieces the desktop panel reads, computed from the local store.

    Deliberately the same keys `label_progress.py` writes from the Roboflow side (`total`,
    `decided`, `null_annotations`, `by_class`, `by_cell`, `sessions`) so `app/dataset_status.py`
    cannot tell the two apart - plus the two this path adds, `pseudo` and `machine_only_by_split`,
    which are what stop a panel reporting 100% while 500 frames carry a machine's unread boxes.

    The session rows are the reader's own shape, not the file's: `dataset_status._sessions` builds
    `train`/`valid`/`test`/`decided`/`total` and derives `splits` from the first three, so using
    the same spelling here means the panel has one reader for two writers rather than a branch.

    Pure: frames in, dict out, so the arithmetic is testable without a workspace.
    """
    splits = splits or {}
    by_class: dict[str, list[int]] = {}
    by_cell: dict[str, list[int]] = {}
    pseudo_by_cell: dict[str, int] = {}
    sessions: dict[str, dict[str, int]] = {}
    machine_by_split: dict[str, int] = {}

    for frame in frames:
        decided = 1 if frame.state != "unlabeled" else 0
        for bucket, key in ((by_class, frame.slug), (by_cell, frame.cell)):
            counts_ = bucket.setdefault(key, [0, 0])
            counts_[0] += decided
            counts_[1] += 1
        split = splits.get(frame.name, "")
        if frame.machine_only:
            pseudo_by_cell[frame.cell] = pseudo_by_cell.get(frame.cell, 0) + 1
            machine_by_split[split] = machine_by_split.get(split, 0) + 1

        session = frame.session or "(no session tag)"
        record = sessions.setdefault(
            session, {"train": 0, "valid": 0, "test": 0, "decided": 0, "total": 0}
        )
        record["decided"] += decided
        record["total"] += 1
        # Only the three real splits land in the counts. A frame with no entry in `splits.json`
        # yet is not a fourth split, and `dataset_status._sessions` derives its own `splits` from
        # exactly these three - so a `?` bucket here would make the local snapshot the one file
        # whose session flag the panel has to correct.
        if split in record:
            record[split] += 1

    totals = counts(frames)
    return {
        "total": totals.total,
        "decided": totals.decided,
        "null_annotations": totals.nulls,
        "reviewed": totals.reviewed,
        "pseudo": totals.machine_only,
        "by_class": {k: v for k, v in sorted(by_class.items())},
        "by_cell": {k: v for k, v in sorted(by_cell.items())},
        "pseudo_by_cell": {k: v for k, v in sorted(pseudo_by_cell.items())},
        "machine_only_by_split": {k: v for k, v in sorted(machine_by_split.items())},
        "sessions": [
            {
                "name": name,
                **record,
                "splits": sum(1 for s in SPLIT_NAMES if record[s]),
            }
            for name, record in sorted(sessions.items())
        ],
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sha256(path: Path) -> str:
    """The image's digest, recorded so a re-encode cannot silently invalidate a label.

    Empty when there is no file: provenance about a frame whose image vanished is worth keeping,
    and an absent digest says that rather than pretending the bytes were checked.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""
