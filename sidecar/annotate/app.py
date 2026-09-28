"""The annotator's HTTP surface: a worklist, a frame, and one save - plus suggestions.

One app, one browser tab, no bundler. `build_annotate_app(store=..., ...)` is the whole interface,
modelled on `app/main.py`'s `build_app(state_factory)` so the test style is the one the repo already
uses (`TestClient`, fakes injected, no camera, no GPU, no network).

Five endpoints carry the loop:

    GET  /api/frames                 the worklist, most-outstanding-first - or one of the three views
    GET  /api/frame/<name>/image     the JPEG
    GET  /api/frame/<name>           boxes + state + provenance for that frame
    POST /api/frame/<name>/suggest   a proposal from the local weight (hosted only if it found none)
    POST /api/frame/<name>/labels    save boxes, or save a deliberate null

Four decisions in here are worth knowing before changing anything.

**Suggest is not save.** The suggestion endpoint writes only to `provenance.suggestion`; the label
file is written by the save endpoint, by a person. That split is what makes "a machine drew this and
nobody looked" a fact about the filesystem (`provenance.machine_only`) rather than a claim about
what a user clicked.

**The weights are chosen by roster, not by filename.** `choose_weights` accepts a weight only if its
recorded `class_names` are this dataset's classes, because a weight that predicts a different list
would propose boxes under classes that mean something else when saved as an index. The name is a
convention; the recorded list is evidence.

**A null is a product of the negative set only.** `store.write` enforces it, and the endpoint does
not second-guess: an empty file on a *product* frame teaches the head that the product is absent,
which is the one annotation mistake that actively trains the model wrong.

**A view is the store's own selector, and the filters narrow it.** `?review=1`, `?draw=1` and
`?nulls=1` are `store.review_worklist`/`draw_worklist`/`null_worklist` rather than predicates written
out in this file, because all three are also what `make human-pass` prints - the checklist's review,
draw and null sections: two implementations of "which frames is this pass about" is how a page and a
document come to disagree, and neither of them looks wrong on its own. `split=`/`distance=`/`state=`/
`cell=` then narrow whichever view is active, and the counts in the response stay the whole set's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .providers import (
    HOSTED_PROVIDERS,
    HostedVision,
    LocalWeights,
    build_hosted,
    hosted_available,
    suggest_for,
)
from .store import (
    CLASS_NAMES,
    CLASS_SLUGS,
    DISTANCE_WORK_ORDER,
    PSEUDO_CLASS_SLUGS,
    PROVIDER_UNKNOWN,
    REVIEW_SPLIT_ORDER,
    UNRECORDED,
    Box,
    GATE_SPLITS,
    LabelStore,
    draw_worklist,
    filter_values,
    gate_frames,
    matches_filter,
    null_worklist,
    review_worklist,
    summary,
)

STATIC_DIR = Path(__file__).parent / "static"

# What a hosted call is allowed to send: the longest side, in pixels, of the image that leaves the
# machine. Downscaling is free in the coordinates - a uniform scale keeps every normalized box
# where it was, which is the same argument `RoboflowRemoteDetector._encode` makes - and it is what
# keeps a 3 MB staged JPEG from becoming a 3 MB request per frame.
HOSTED_MAX_SIDE = 1024
HOSTED_QUALITY = 85

DEFAULT_MAX_HOSTED_CALLS = 100


def read_frame(path: Path, flipper=None):
    """The frame as the detector wants it: BGR, straight off disk.

    `cv2.imread` (not PIL) because `YoloDetector.infer` is handed an OpenCV frame by the capture
    path, so a suggestion made here is made on the same kind of input the app would produce.
    """
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(status_code=415, detail=f"could not decode {path.name} as an image")
    return frame


def encode_for_hosted(frame) -> bytes:
    """A JPEG that costs as little as it can while keeping every box's position.

    `frame` is the BGR array the local weight was given, so the two providers are looking at the
    same pixels - one at full size, one downscaled - and a difference between their answers cannot
    be a difference in input.
    """
    import cv2

    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > HOSTED_MAX_SIDE:
        scale = HOSTED_MAX_SIDE / float(longest)
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), HOSTED_QUALITY])
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode the frame for the provider")
    return bytes(buf.tobytes())


# --------------------------------------------------------------------------
# Which weight may suggest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Weight:
    """A local weight that is allowed to suggest, and the geometry it needs."""

    name: str
    path: Path
    resize_mode: str
    class_names: tuple[str, ...]


def choose_weights(models_dir: Path, requested: str = "") -> tuple[Weight | None, str]:
    """The weight to suggest with, and a sentence saying why when there is none.

    Selection is by **recorded class list**, with a deliberate preference for a weight whose list is
    this dataset's classes. A weight is only usable if its names map onto the dataset's exactly: the
    suggestion carries an *index*, so a weight trained on a different list would write rows that
    mean a different product. Nothing here guesses from the filename (`models/README.md` is explicit
    that the name is a convention).

    Read through `app.models` rather than by listing the directory, so the annotator sees what the
    app's own *Weights on disk* list shows - including each weight's recorded `resize_mode`, which is
    the geometry the suggestions have to be produced at to be worth anything.
    """
    from app.models import installed_models

    if not Path(models_dir).is_dir():
        return None, f"no weights directory at {models_dir}"

    entries = installed_models(Path(models_dir))
    wanted = set(CLASS_NAMES)

    def usable(entry) -> bool:
        return bool(entry.class_names) and set(entry.class_names) == wanted

    if requested:
        for entry in entries:
            if Path(entry.value).name == Path(requested).name:
                if not usable(entry):
                    return None, (
                        f"{requested} is not recorded as predicting this dataset's classes "
                        f"({len(entry.class_names)} recorded) - suggestions from it would be saved "
                        "under the wrong product"
                    )
                return (
                    Weight(
                        name=Path(entry.value).name,
                        path=Path(models_dir) / Path(entry.value).name,
                        resize_mode=str(entry.resize_mode or "stretch"),
                        class_names=tuple(entry.class_names),
                    ),
                    "",
                )
        return None, f"{requested} is not in {models_dir}"

    for entry in entries:
        if usable(entry):
            return (
                Weight(
                    name=Path(entry.value).name,
                    path=Path(models_dir) / Path(entry.value).name,
                    resize_mode=str(entry.resize_mode or "stretch"),
                    class_names=tuple(entry.class_names),
                ),
                "",
            )
    if entries:
        # Each weight's own findings are repeated here, because "none of them matches" is not
        # actionable and "this one cannot predict safeguard_pure_white_60g" is - it is the same
        # sentence the Admin Panel's *Weights on disk* list shows, from the same place.
        detail = "; ".join(
            f"{Path(entry.value).name}: {'; '.join(entry.class_warnings) or 'no recorded class list'}"
            for entry in entries
        )
        return None, (
            f"none of the installed weights is recorded as predicting this dataset's "
            f"{len(CLASS_NAMES)} classes ({detail}) - pass --weights, or record the requirement "
            "for one in the Admin Panel"
        )
    return None, f"no weights in {models_dir}"


@dataclass
class AnnotateState:
    """Everything the routes close over. One instance, no globals - the sidecar's own shape."""

    store: LabelStore
    weight: Weight | None = None
    weight_note: str = ""
    conf: float = 0.25
    imgsz: int = 640
    device: str = "cpu"
    hosted_name: str = ""
    max_hosted_calls: int = DEFAULT_MAX_HOSTED_CALLS
    detector_factory: object = None
    hosted_post: object = None
    env: object = None
    hosted_calls: int = 0
    local_calls: int = 0
    last_note: str = ""
    counters: dict = field(default_factory=dict)

    # Built lazily and cached: a weight load is ~1 s and an annotator session opens hundreds of
    # frames, while a *hosted* client is just headers and is rebuilt per call so a rotated key
    # needs no restart.
    def local(self) -> LocalWeights | None:
        if self.weight is None:
            return None
        if "local" not in self.counters:
            self.counters["local"] = LocalWeights(
                str(self.weight.path),
                resize_mode=self.weight.resize_mode,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                detector_factory=self.detector_factory,
                class_names=self.weight.class_names,
            )
        return self.counters["local"]

    def hosted(self) -> HostedVision | None:
        if not self.hosted_name:
            return None
        if self.hosted_calls >= self.max_hosted_calls:
            return None
        getter = self.env or _env
        return build_hosted(self.hosted_name, self.hosted_post, getter=getter)

    def hosted_capped(self) -> bool:
        return bool(self.hosted_name) and self.hosted_calls >= self.max_hosted_calls


def _env(name: str) -> str:
    from app.credentials import env_value

    return env_value(name)


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


def build_annotate_app(state: AnnotateState) -> FastAPI:
    app = FastAPI(title="SCANnCART Annotator")
    app.state.annotate = state

    @app.get("/api/config")
    def config() -> dict:
        getter = state.env or _env
        return {
            "classes": [
                {
                    "index": index,
                    "slug": slug,
                    "name": CLASS_NAMES[index],
                    "pseudo": slug in PSEUDO_CLASS_SLUGS,
                }
                for index, slug in enumerate(CLASS_SLUGS)
            ],
            "images": str(state.store.out),
            "annotations": str(state.store.annotations),
            "extras": [str(p) for p in state.store.extras],
            # The filters the page offers, as the token it sends and the word to show. Built here
            # rather than in JavaScript so the store's own spellings reach it instead of being
            # retyped there - including `unknown` for "nothing was recorded" (`store.UNRECORDED`),
            # which is how a frame with no distance or no split is asked for. Ordered the way a
            # pass works them: the two splits the acceptance gate asks about first, then the
            # distances from farthest, because that is where the evidence is thinnest.
            "splits": [
                # The pair the acceptance gate measures, offered as one entry because that is how the
                # pass asks about it (`split=test,valid` is a set of values, which is why the
                # parameter takes a comma list at all) - and because the header's gate chip
                # selects exactly this. Built from `GATE_SPLITS`, so the option a person can pick
                # by hand and the pair the gate is defined as cannot come apart.
                {"value": ",".join(GATE_SPLITS), "label": f"{'/'.join(GATE_SPLITS)} (the gate)"},
                *(
                    {"value": s or UNRECORDED, "label": s or "unassigned"}
                    for s in REVIEW_SPLIT_ORDER
                ),
            ],
            "distances": [
                {"value": d or UNRECORDED, "label": d or "unrecorded"}
                for d in (*DISTANCE_WORK_ORDER, "")
            ],
            "weight": (
                {
                    "name": state.weight.name,
                    "resize_mode": state.weight.resize_mode,
                    "conf": state.conf,
                    "imgsz": state.imgsz,
                }
                if state.weight
                else None
            ),
            "weight_note": state.weight_note,
            "hosted": {
                "name": state.hosted_name or "",
                "available": hosted_available(getter),
                "calls": state.hosted_calls,
                "max_calls": state.max_hosted_calls,
                "keys": {name: spec["env"] for name, spec in HOSTED_PROVIDERS.items()},
            },
        }

    @app.get("/api/frames")
    def frames(
        state_filter: str = Query("", alias="state"),
        cell: str = "",
        split: str = "",
        distance: str = "",
        limit: int = 0,
        review: int = 0,
        draw: int = 0,
        nulls: int = 0,
    ) -> dict:
        # Three named views over the one set, then three narrowing filters, in that order.
        #
        # `review=1` is the second pass over the same set: the frames a weight drew and nobody has
        # looked at, most-costly-split-first (`store.review_worklist`). `draw=1` and `nulls=1` are the
        # pass's other two halves - the `far` frames in `test`/`valid` a weight was asked about and
        # found nothing in (`store.draw_worklist`), and the hard negatives in those splits that carry
        # no null yet (`store.null_worklist`). All three are the same calls `make human-pass` renders
        # as the checklist's sections, so the page and the document cannot come out with different
        # lists. They select *before* the filters below, so `split=`/`distance=`/`state=`/`cell=`
        # narrow a view rather than replacing it - and the counts stay the *whole* set's, because a
        # header that recalculated itself from the filtered list would report 100% done the moment a
        # review pass started. The views are exclusive; the order here is which one wins if a client
        # asks for two, and the page only ever sets one.
        every = state.store.frames()
        splits = state.store.splits()
        if draw:
            all_frames = draw_worklist(every, splits, state.store.provenance())
        elif nulls:
            all_frames = null_worklist(every, splits)
        elif review:
            all_frames = review_worklist(every, splits)
        else:
            all_frames = every
        # A comma-separated list, so the pair a pass works (`split=test,valid`) is one request. An
        # empty parameter means "no filter"; `unknown` names the frames with nothing recorded.
        wanted_splits = filter_values(split)
        wanted_distances = filter_values(distance)
        selected = [
            f
            for f in all_frames
            if (not state_filter or f.state == state_filter)
            and (not cell or f.cell == cell)
            and matches_filter(wanted_splits, splits.get(f.name, ""))
            and matches_filter(wanted_distances, f.distance)
        ]
        if limit > 0:
            selected = selected[:limit]
        # The gate readout, computed on *every* request rather than only in the review view: its
        # whole point is that the number is visible while the operator is working the pass, and the
        # pass is not always in review mode (the far frames get drawn first). The count is
        # `store.gate_frames` over the review worklist - the same call the checklist's header is
        # rendered from - so the finish line on screen and the one in the document are one number.
        review_list = review_worklist(every, splits)
        inside = gate_frames(review_list, splits)
        pending_by_split: dict[str, int] = {}
        for frame in inside:
            split = splits.get(frame.name, "")
            pending_by_split[split] = pending_by_split.get(split, 0) + 1
        counts = summary(every)
        return {
            "gate": {
                "splits": list(GATE_SPLITS),
                "pending": len(inside),
                # What the whole-set review count falls to once the gate splits are clean: the
                # stopping point the checklist's header prints, so the page can say "work it down
                # to this" instead of only "this many left".
                "remaining": len(review_list) - len(inside),
                "by_split": {split: pending_by_split.get(split, 0) for split in GATE_SPLITS},
                "clear": not inside,
            },
            # `unlabeled` is added here rather than in `summary()`, which is shaped to be written
            # out as the snapshot `app/dataset_status.py` reads - this one is a page readout, and
            # "how much is left" is a derived number the file has no use for.
            "counts": counts
            | {
                "loaded": len(selected),
                "unlabeled": counts["total"] - counts["decided"],
            },
            "review": bool(review),
            "draw": bool(draw),
            "nulls": bool(nulls),
            "frames": [frame_payload(f, splits.get(f.name, "")) for f in selected],
        }

    @app.get("/api/summary")
    def summary_route() -> dict:
        return summary(state.store.frames(), state.store.splits())

    @app.get("/api/frame/{name}")
    def frame(name: str) -> dict:
        found = state.store.frame(name)
        if found is None:
            raise HTTPException(status_code=404, detail=f"{name} is not in a staged manifest")
        record = (state.store.provenance().get(name) or {})
        # The split travels with the frame because it decides whether these labels are part of the
        # acceptance number at all: a frame in `test` that a machine drew makes the test reading a
        # measurement of the annotator, and that is worth seeing while labeling rather than
        # discovering from a summary afterwards. Empty when `splits.json` has not been captured
        # yet - never a guess.
        split = state.store.splits().get(name, "")
        return {
            "frame": frame_payload(found, split),
            "split": split,
            "boxes": [box.as_dict() for box in state.store.read_boxes(name)],
            "provenance": {k: v for k, v in record.items() if k != "suggestion"},
            "suggestion": record.get("suggestion") or {},
        }

    @app.get("/api/frame/{name}/image")
    def image(name: str) -> Response:
        path = state.store.image_path(name)
        # `.is_file()` rather than truthiness: an empty `Path()` is `Path(".")`, which is truthy,
        # and `FileResponse` on it is a 500 from inside starlette rather than the 404 this means.
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no image for {name}")
        media = {
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(path.suffix.lower(), "image/jpeg")
        return FileResponse(path, media_type=media)

    @app.post("/api/frame/{name}/suggest")
    def suggest(name: str) -> dict:
        found = state.store.frame(name)
        if found is None:
            raise HTTPException(status_code=404, detail=f"{name} is not in a staged manifest")
        local = state.local()
        hosted = state.hosted()  # None when none is configured *or* the run's cap is reached
        if local is None and hosted is None:
            return {
                "boxes": [],
                "provider": "none",
                "note": state.weight_note or "no provider is configured",
                "hosted_capped": state.hosted_capped(),
            }

        frame_array = read_frame(found.image)
        encoded = encode_for_hosted(frame_array) if hosted is not None else None
        proposed = suggest_for(frame_array, local, hosted, encoded)

        # Counted from *what ran*, not from what was configured: the ordering rule decides which
        # provider answers a given frame, and a counter that guessed it would make the cap (and the
        # cost it exists to bound) wrong by exactly the frames where local found something.
        state.local_calls += 1 if local is not None else 0
        state.hosted_calls += 1 if proposed.provider.startswith("hosted") else 0
        state.store.record_suggestion(name, proposed.boxes, proposed.provider)
        return {
            "boxes": [box.as_dict() for box in proposed.boxes],
            "provider": proposed.provider,
            "note": proposed.note,
            "hosted_capped": state.hosted_capped(),
        }

    @app.post("/api/frame/{name}/labels")
    def save(name: str, payload: dict) -> JSONResponse:
        found = state.store.frame(name)
        if found is None:
            raise HTTPException(status_code=404, detail=f"{name} is not in a staged manifest")
        raw_boxes = payload.get("boxes")
        if not isinstance(raw_boxes, list):
            raise HTTPException(status_code=422, detail="`boxes` must be a list")
        boxes = []
        for item in raw_boxes:
            if not isinstance(item, dict):
                raise HTTPException(status_code=422, detail="each box must be an object")
            try:
                boxes.append(
                    Box(
                        cls=int(item["cls"]),
                        cx=float(item["cx"]),
                        cy=float(item["cy"]),
                        w=float(item["w"]),
                        h=float(item["h"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"malformed box: {item}") from None
        try:
            # `provider` defaults to `unknown`, never to `human`: the page does not claim who drew
            # what (the store decides `machine_only` by comparing the rows to its own recorded
            # suggestion), and a label written outside this app must not read as hand-drawn work.
            record = state.store.write(
                name,
                boxes,
                null=bool(payload.get("null")),
                provider=str(payload.get("provider") or PROVIDER_UNKNOWN),
                # A person's assertion that they checked a weight's boxes and they are right, which
                # is the ordinary outcome of reviewing and the one thing the store's own comparison
                # cannot see (the rows are unchanged by definition). `null`'s sibling: only a
                # person makes it, and the page sends it - never a provider.
                confirm=bool(payload.get("confirmed")),
            )
        except ValueError as exc:
            # A refusal, not a failure: the two guards `store.write` enforces are the two mistakes
            # that teach the model something false (a box on a hard negative, a null on a product).
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return JSONResponse(
            {
                "frame": frame_payload(
                    state.store.frame(name), state.store.splits().get(name, "")
                ),
                "provenance": record,
            }
        )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> Response:
            return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    return app


def frame_payload(frame, split: str = "") -> dict:
    return {
        "name": frame.name,
        "slug": frame.slug,
        "cell": frame.cell,
        "distance": frame.distance,
        # Carried on the row because the worklist can be filtered by it (`?split=test`) and because
        # it is the fact the pass is *for*: a frame in `test` a machine drew makes the acceptance
        # number a measurement of the annotator. `""` when no plan has reached this frame.
        "split": split,
        "session": frame.session,
        "state": frame.state,
        "boxes": frame.boxes,
        "machine_only": frame.machine_only,
        "provider": frame.provider,
        "url": f"/api/frame/{frame.name}/image",
    }


__all__ = [
    "AnnotateState",
    "Weight",
    "build_annotate_app",
    "choose_weights",
    "encode_for_hosted",
    "read_frame",
]
