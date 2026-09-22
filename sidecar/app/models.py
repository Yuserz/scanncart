"""Which weights are on disk and therefore selectable. One directory read, no torch.

`settings_store` validates model *names* and deliberately never touches the filesystem, so
it can never answer "what could I select?" - this is the other half of that split.

It exists because the picker used to be a fixed list in the renderer: a newly trained
`models/scanncart-grocery-v2.pt` was valid to the API (`is_custom_model()` accepts any
`.pt`/`.onnx` directly under `models/`) and impossible to select in the UI, so the one
moment the model changed was the one moment the app needed a code edit. Reporting the
directory fixes that once, for every future generation.

It also reads the **record** each weight may carry - `models/<stem>.json`, written by
`tools/train_v2.py --install`, or by `record_requirement` below when the operator is the only
source of the fact - because neither the requirement a model has to be *run* with nor the class
list it predicts is recoverable from the weight or from its name. That second fact is what lets
`installed_models` report `class_list_problems` for a weight before it ever runs: a checkpoint
from a distance-split project has 24 outputs and nothing in its file says so. See `InstalledModel`'s docstring in
`app.schemas` for why that field is a fact rather than a guess, and note the three states it
can be in: recorded, absent, or unreadable - all three reported distinctly, since only the
first one is safe to act on. The record carries `--val`'s measured recall as well, so what a
model *scored* travels with it and the panel can show it without a second lookup. `requirement_for` is the same record read for the *capture*
path rather than the panel: it is what `resolve_resize_mode` consults, so a recorded
requirement is honoured by `auto` and not just displayed next to it.

Two deliberately small erasures: a missing directory is an empty list rather than an error
(a packaged app with no custom weights is a normal state, not a fault), and names are
returned exactly as `active_model` stores them (`models/<name>`) so the value the picker
offers is the value the validator accepts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from app.roster import class_list_problems
from app.schemas import ClassRecall, DistanceRecall, InstalledModel, ValidationRecord
from app.settings_store import (
    ALLOWED_RESIZE_MODES,
    CUSTOM_MODEL_DIR,
    CUSTOM_MODEL_SUFFIXES,
    is_custom_model,
    resolve_resize_mode,
)

# Resolved from the package, not the process's cwd, so a sidecar started from elsewhere
# still finds the models the app was configured with. This matches what inference resolves,
# because the Electron main process spawns the sidecar with sidecar/ as its cwd.
MODELS_DIR = Path(__file__).resolve().parents[1] / CUSTOM_MODEL_DIR.rstrip("/")


def _finite_number(value: object) -> float | None:
    """`value` as a finite float, or None.

    `bool` is excluded deliberately: it is an `int` in Python, so `True` would otherwise be
    read as a recall of 1.000. A NaN or an infinity has to be rejected too — one arithmetic
    accident away from rendering as "recall nan", which reads like the panel is broken rather
    than like the file is.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_classes(raw: object) -> list[ClassRecall]:
    """One `per_class` list, dropping anything that says nothing. Never raises.

    Shared by a split's record and by every distance breakdown under it, so the two cannot
    apply different rules to the same rows - a distance cell that was dropped here but kept
    there would be a breakdown that does not add up to the number above it.
    """
    classes: list[ClassRecall] = []
    for entry in raw if isinstance(raw, list) else []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        raw_recall = entry.get("recall")
        recall = None if raw_recall is None else _finite_number(raw_recall)
        if raw_recall is not None and (recall is None or not 0.0 <= recall <= 1.0):
            continue
        instances = entry.get("instances")
        classes.append(
            ClassRecall(
                name=name,
                recall=recall,
                instances=(
                    instances
                    if isinstance(instances, int)
                    and not isinstance(instances, bool)
                    and instances >= 0
                    else 0
                ),
            )
        )
    return classes


def _read_distances(raw: object) -> list[DistanceRecall]:
    """The per-distance breakdown, on the same rules as the split above it.

    The floor is **not** repeated in each entry: the distances slice *this* measurement, so a
    second copy of the floor could only disagree with the one they were judged against, and the
    panel reads it off the block these hang under. An entry whose distance is missing, or that
    carries no measurable class, says nothing a panel can render and goes.

    The direction that matters here is the never-thrown one: a breakdown is extra detail, so a
    corrupt one must cost the detail and not the acceptance number beside it.
    """
    out: list[DistanceRecall] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        distance = entry.get("distance")
        if not isinstance(distance, str) or not distance.strip():
            continue
        classes = _read_classes(entry.get("per_class"))
        if not classes:
            continue
        images = entry.get("images")
        raw_aggregates = entry.get("aggregates")
        aggregates: dict[str, float] = {}
        if isinstance(raw_aggregates, dict):
            for key, value in raw_aggregates.items():
                number = _finite_number(value)
                if isinstance(key, str) and number is not None:
                    aggregates[key] = number
        out.append(
            DistanceRecall(
                distance=distance,
                images=(
                    images
                    if isinstance(images, int) and not isinstance(images, bool) and images >= 0
                    else 0
                ),
                aggregates=aggregates,
                per_class=classes,
            )
        )
    return out


def _read_class_names(raw: object) -> list[str]:
    """The `class_names` a record carries, dropping anything that is not a usable name.

    Same contract as the rest of the module - a hand-edited record must not be able to take down
    `/api/models`, which the Admin Panel loads on every open - with one rule of its own: a list
    that is absent, or that holds no non-blank string, answers the *empty* list, which is the
    module's word for "not recorded". That is deliberate rather than lossy. The empty list is
    exactly what `class_list_problems` must not be handed, since it would read as a model that
    predicts none of the 8 roster classes; the caller gates on it, so "nothing is known" stays
    silent instead of being reported as a verdict about a list nobody has seen.

    Names are **not** validated against the roster here. This reads a fact; judging it is
    `roster.class_list_problems`, called by `installed_models`, so the one place that decides a
    class list is wrong is the one place the app-side and probe-side findings come from.
    """
    if not isinstance(raw, list):
        return []
    return [name.strip() for name in raw if isinstance(name, str) and name.strip()]


def _read_validation(raw: object) -> list[ValidationRecord]:
    """`--val`'s recorded measurements, dropping anything malformed rather than raising.

    Same contract as the rest of this module — a hand-edited record must not be able to take
    down `/api/models`, which the Admin Panel loads on every open — but two drops here are
    rules rather than defensiveness:

    * A block whose **floor** is missing or not a real number is dropped whole. The floor is
      what makes a recall interpretable, and defaulting it to 0.0 would turn every class into
      a pass — a plausible-looking verdict that is the opposite of the truth.
    * A **recall** that is present but not in 0..1 is dropped, not clamped. No honest pass
      produces one, so the number is corrupt, and a clamped value would be this program's
      invention rather than the measurement's.

    What is emphatically *not* dropped is a class with `recall: null`: that is how "the split
    held no instances of it" was recorded, which is a fact about the run rather than a gap in
    the file. A class entry with no name, or a block with no measurable class in it, says
    nothing a panel can render, so those go.
    """
    if not isinstance(raw, list):
        return []
    records: list[ValidationRecord] = []
    for block in raw:
        if not isinstance(block, dict):
            continue
        split = block.get("split")
        floor = _finite_number(block.get("floor"))
        if not isinstance(split, str) or not split.strip() or floor is None:
            continue
        if not 0.0 <= floor <= 1.0:
            continue

        classes = _read_classes(block.get("per_class"))
        if not classes:
            continue

        raw_aggregates = block.get("aggregates")
        aggregates: dict[str, float] = {}
        if isinstance(raw_aggregates, dict):
            for key, value in raw_aggregates.items():
                number = _finite_number(value)
                if isinstance(key, str) and number is not None:
                    aggregates[key] = number
        measured_at = block.get("measured_at")
        records.append(
            ValidationRecord(
                split=split,
                floor=floor,
                measured_at=measured_at if isinstance(measured_at, str) else "",
                aggregates=aggregates,
                per_class=classes,
                per_distance=_read_distances(block.get("per_distance")),
            )
        )
    return records


def read_record(weights: Path) -> dict:
    """What is known about one weight from the JSON beside it. Never raises.

    A missing record and a corrupt one both answer the same values, because they mean the
    same thing to the caller — nothing is known — and a hand-edited file must not be able to
    take down `/api/models`, which the Admin Panel loads on every open. `resize_mode` is
    validated against `ALLOWED_RESIZE_MODES` for the same reason: a value the settings PATCH
    would reject must not be presented as a requirement to satisfy.

    `recorded` is the one field that is about the *file* rather than its contents, and it is
    load-bearing rather than diagnostic. "Installed by the tool, nothing measured yet" and
    "copied into `models/` by hand, nothing known" leave every other field identical — empty
    requirement, no numbers — but they are different situations with different remedies, and
    the panel has to choose between telling the operator to run `--val` and telling them
    nothing. Only the filesystem can distinguish them, so it answers here.

    `validation` is what `--val` measured, per split. Empty list means "not measured", which
    is distinct from a measurement that scored zero.

    `class_names` is the weight's own class list as `tools/train_v2.py --install` recorded it,
    and the empty list means "not recorded" rather than "predicts nothing" - a hand-copied
    weight, or a record written before this field existed. This is the read that lets a bad
    weight be caught from the listing instead of only once it runs: a model trained from a
    distance-split project predicts 24 classes, which nothing about its filename or its
    checkpoint says.
    """
    record_path = weights.with_suffix(".json")
    recorded = record_path.is_file()
    try:
        body = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = None
    if not isinstance(body, dict):
        body = {}
    mode = body.get("resize_mode")
    if mode is not None and mode not in ALLOWED_RESIZE_MODES:
        mode = None
    return {
        "recorded": recorded,
        "resize_mode": mode,
        "source": body.get("source") if isinstance(body.get("source"), str) else "",
        "class_names": _read_class_names(body.get("class_names")),
        "validation": _read_validation(body.get("validation")),
    }


def requirement_for(active_model: str, directory: Path | None = None) -> str | None:
    """The `resize_mode` recorded beside `active_model`, or None when there is none.

    This is what makes `auto` honour a weight's requirement instead of merely displaying
    it: `resolve_resize_mode` is passed this and prefers it to its format heuristic, so the
    geometry the weights were trained with is used whether or not an operator went back and
    set the field by hand.

    It lives here, beside `read_record`, and not in `settings_store` — that module is pure
    and never touches the filesystem, so the fact is looked up by the caller and handed to
    the rule as an argument. One read of one small JSON file, at capture start.

    Never raises, and answers None for every kind of "unknown" alike — a stock weight, a
    name the settings validator would reject, a file that is not there, and a record that is
    missing or corrupt. All of them mean the same thing to `resolve_resize_mode`, which then
    falls back to its own rule, and a hand-edited record must not be able to fail a capture.
    """
    if not isinstance(active_model, str) or not is_custom_model(active_model):
        return None
    root = Path(directory) if directory is not None else MODELS_DIR
    weights = root / active_model.replace("\\", "/")[len(CUSTOM_MODEL_DIR):]
    if not weights.is_file():
        return None
    return read_record(weights).get("resize_mode")


def record_requirement(
    active_model: str, resize_mode: str, directory: Path | None = None
) -> Path:
    """Write `resize_mode` into the record beside these weights, and return that record's path.

    The other writer of these files is `tools/train_v2.py --install`, which knows the requirement
    because it knows the version the weights were trained from. This one is for the case that
    path cannot cover: weights that reached `models/` some other way — a copy from another
    machine, a stock checkpoint, a checkpoint whose training run was never recorded — where the
    operator is the only source of the fact. `POST /api/models/record` is its route, and the
    Admin Panel's button is its caller.

    It **merges rather than replaces**. The rest of a record is evidence only the training run
    can supply — which dataset version, what `--val` measured, and the weight's own class list —
    so a whole-file write here would delete all of it in order to add one field, and the class
    list is the one that cannot be recovered afterwards (`--install` is the only writer that ever
    knew it). Keys it does not recognise are preserved for the same reason: this function knows
    about `resize_mode` and nothing else.

    Written to a temporary file and `os.replace`d, like `settings_store.save_settings`: a crash
    mid-write must not leave a half-written record, because the reader treats a corrupt record as
    "nothing is known" and would silently drop the requirement just recorded.

    Raises `FileNotFoundError` when there is no such weight, `ValueError` for a mode the reader
    would ignore (`auto` is not a requirement, it is a lookup — recording it would produce a
    record that changes nothing while looking like it fixed something), and `ValueError` for a
    name `is_custom_model` rejects, which is checked before the path is used.
    """
    if resize_mode not in ALLOWED_RESIZE_MODES or resize_mode == "auto":
        raise ValueError(f"resize_mode must be one of {sorted(ALLOWED_RESIZE_MODES - {'auto'})}")
    if not is_custom_model(active_model):
        raise ValueError(f"model must be a .pt/.onnx directly under {CUSTOM_MODEL_DIR}")
    root = Path(directory) if directory is not None else MODELS_DIR
    weights = root / active_model.replace("\\", "/")[len(CUSTOM_MODEL_DIR):]
    if not weights.is_file():
        raise FileNotFoundError(f"no weights at {active_model}")

    record_path = weights.with_suffix(".json")
    try:
        body = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = None
    merged = body if isinstance(body, dict) else {}
    merged["resize_mode"] = resize_mode

    tmp_path = record_path.with_name(record_path.name + ".tmp")
    tmp_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, record_path)
    return record_path


def installed_models(directory: Path | None = None) -> list[InstalledModel]:
    """The selectable custom weights, sorted, as `models/<name>` plus what is known of each.

    Filtered through `is_custom_model` rather than by suffix alone: offering something the
    settings validator would reject means an operator can pick a value that fails to save,
    with an error about a name they just chose from a list.
    """
    root = Path(directory) if directory is not None else MODELS_DIR
    if not root.is_dir():
        return []
    installed: list[InstalledModel] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() or entry.name.startswith("."):
            continue
        if not entry.name.endswith(CUSTOM_MODEL_SUFFIXES):
            continue
        value = f"{CUSTOM_MODEL_DIR}{entry.name}"
        if not is_custom_model(value):
            continue
        record = read_record(entry)
        required = record.get("resize_mode")
        # Judged here rather than in the renderer, for the same reason `auto_resolves_to` is:
        # the roster is a rule (`app/roster.py`) and a second copy in TypeScript could only
        # disagree with this one about which names are wrong. Gated on a non-empty list, because
        # an empty one is "not recorded" and `class_list_problems([])` would turn that silence
        # into "predicts none of the 8" - a finding about a class list nobody has seen.
        names = record.get("class_names") or []
        installed.append(
            InstalledModel(
                value=value,
                resize_mode=required,
                class_names=names,
                class_warnings=class_list_problems(names) if names else [],
                # Resolved *through* the requirement, not beside it: `auto` now honours a
                # record, so for a recorded weight this answers the requirement itself. What
                # it is still for is the unrecorded case — the one situation where `auto`
                # really is a guess, and the operator has to be told which way it guesses.
                auto_resolves_to=resolve_resize_mode("auto", value, required),
                source=record.get("source", ""),
                recorded=bool(record.get("recorded")),
                validation=record.get("validation") or [],
            )
        )
    return installed
