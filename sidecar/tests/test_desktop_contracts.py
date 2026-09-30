"""The hand-mirrored contracts between the two toolchains, checked in both directions.

`CLAUDE.md` has said for a while that the WS message contract and the settings contract are
"duplicated by hand ... so keep them in sync manually". That was a convention with nothing behind
it: the defaults and the allowed vocabularies had guards (`tests/test_settings.py`), and everything
else - the frame and status shapes the renderer reads every second, the REST response models, a
per-backend numeric floor, the experimental-model set, the fixture the desktop's own tests assert
against, and the class counts printed in the model labels - was kept in step by remembering.

Every guard here is a pair, read twice: the sidecar's own declaration, and the desktop's restatement
of it. Drift in either direction fails, and so does a field that changed from required to optional
on one side only, because each of those is the same mistake from a different angle: the two halves
of one program disagreeing about what is on the wire.

The subset runs as its own CI job (`pytest -m mirror`), which is what the `pytestmark` below is for
- and the marker is the convention now, so a new hand-mirrored pair gets one rather than a mention
in a document.
"""

from __future__ import annotations

import re
from typing import get_args

import pytest

from app import schemas
from app.roster import ROSTERS, V2_ROSTER
from app.settings import Settings
from app.settings_store import (
    EXPERIMENTAL_MODELS,
    MIN_REMOTE_TRACK_EXPIRY_S,
    MIN_TRACK_EXPIRY_S_BY_BACKEND,
)

from tests.desktop_mirrors import (
    REPO_ROOT,
    TS_LIB,
    TS_TEST,
    assert_same_fields,
    python_fields,
    read_ts,
    strip_ts_comments,
    ts_array_items,
    ts_entries,
    ts_export_rhs,
    ts_fields,
    ts_object_array,
    ts_record,
    ts_scalar,
    ts_value,
)

pytestmark = pytest.mark.mirror

WS_TS = TS_LIB / "ws.ts"
API_TS = TS_LIB / "api.ts"
FIELDS_TS = TS_LIB / "settingsFields.ts"
FAKES_TS = TS_TEST / "fakes.ts"

# The process boundary, which is not a file the two toolchains share but a set of names they have
# to agree about: the handshake `run.py` prints, the variable the desktop puts its pid in, the IPC
# channel between the main process and the renderer, and the interpreter both the desktop and the
# Makefile launch the sidecar with.
RUN_PY = REPO_ROOT / "sidecar" / "run.py"
SPAWN_TS = REPO_ROOT / "desktop" / "src" / "main" / "sidecar.ts"
MAIN_TS = REPO_ROOT / "desktop" / "src" / "main" / "index.ts"
PRELOAD_TS = REPO_ROOT / "desktop" / "src" / "preload" / "index.ts"
PRELOAD_DTS = REPO_ROOT / "desktop" / "src" / "preload" / "index.d.ts"
RENDERER_APP_TS = REPO_ROOT / "desktop" / "src" / "renderer" / "src" / "App.tsx"
RENDERER_HEALTH_TS = (
    REPO_ROOT / "desktop" / "src" / "renderer" / "src" / "hooks" / "useSidecarHealth.ts"
)
MAKEFILE = REPO_ROOT / "Makefile"

# The calibration probe's own copy of the camera-control bounds. It is in `app/`, next to the API's,
# and it is not imported here - see the guard's docstring for why.
CAMERA_CAPS_PY = REPO_ROOT / "sidecar" / "app" / "camera_caps.py"

# Where the desktop renamed a mirror. `Stats` is the sidecar's model; the renderer calls it
# `FrameStats` because it hangs off a frame message, and the pair is listed once rather than the
# rename being special-cased inside the comparison.
RENAMED = {"Stats": "FrameStats"}

# Everything on the WebSocket, read once: the field pairs below and the dispatch branches are two
# facts about the same list, and a type present in one and not the other is exactly the defect this
# guard is for.
STREAM_MESSAGES = (
    (schemas.Detection, "Detection"),
    (schemas.Stats, "FrameStats"),
    (schemas.FrameMessage, "FrameMessage"),
    (schemas.StatusMessage, "StatusMessage"),
    (schemas.InferenceMessage, "InferenceMessage"),
)


def test_the_stream_protocol_mirrors_the_sidecar():
    """The message shapes - the contract with the least room for drift.

    A field renamed here is not a compile error anywhere: the sidecar sends `latency_ms`, the
    renderer reads `latency`, and every reading is silently `undefined` on a live preview. That is
    the failure this pair exists for, and it is why `cls`/`track_id`/`detail` are compared as names
    rather than as anything the TypeScript side could infer.
    """
    source = read_ts(WS_TS)
    for model, ts_name in STREAM_MESSAGES:
        assert_same_fields(
            f"the stream protocol ({model.__name__} vs ws.ts's {ts_name})",
            python_fields(model),
            ts_fields(source, ts_name),
        )
    # And every declared *message* is also dispatched. A type with no branch is not a compile error
    # either: the sidecar sends it, `JSON.parse` succeeds, and the renderer drops it on the floor.
    # For the inference notice that is the whole feature - the one thing that can explain a
    # `local_api` run where the camera works and nothing is ever detected, discarded in silence.
    # Read from the model's own `Literal` rather than restated, so a renamed `type` fails here too.
    dispatch = _flat(source)
    for model, _ in STREAM_MESSAGES:
        field = model.model_fields.get("type")
        if field is None:
            continue  # `Detection` and `Stats` are payloads, not envelopes.
        literals = get_args(field.annotation)
        assert literals, f"{model.__name__}.type is no longer a Literal, so nothing can read it"
        for name in literals:
            assert f"msg.type === '{name}'" in dispatch, (
                f"ws.ts declares a {name!r} message and never dispatches it, so the sidecar's "
                "message is parsed and thrown away"
            )


def test_the_settings_contracts_mirror_the_sidecar():
    """`SettingsPayload` and what the panel adds to it, including which fields may be null.

    `camera_brightness`/`camera_exposure`/`camera_autofocus`/`camera_focus` are the null-able ones -
    `null` means "this app imposes no value" - so a `null` that stopped being declared on one side
    would be the difference between "this control was never set" and a value.
    """
    source = read_ts(API_TS)
    assert_same_fields(
        "the settings payload (SettingsPayload)",
        python_fields(schemas.SettingsPayload),
        ts_fields(source, "SettingsPayload"),
    )
    # `SettingsResponse extends SettingsPayload` on the desktop; the parent's fields are merged in by
    # `ts_fields`, and on the sidecar's side by Pydantic's own inheritance.
    assert_same_fields(
        "the settings response (SettingsResponse)",
        python_fields(schemas.SettingsResponse),
        ts_fields(source, "SettingsResponse"),
    )
    # And the structured entry the panel renders with a button beside it.
    assert_same_fields(
        "the unrecorded resize entry (UnrecordedResizeMode)",
        python_fields(schemas.UnrecordedResizeMode),
        ts_fields(source, "UnrecordedResizeMode"),
    )


# The tokens a comment wrapper contributes. A wrapped TS comment repeats `//` on every line and
# a CSS block carries `/*` and ` * `, and those markers land *inside* a sentence the fragments
# cross - dropping them as tokens is what makes one sentence one fact wherever its lines break.
#: A marker is matched as a whole token, so `http://127.0.0.1` and `a#b` survive untouched.
_COMMENT_MARKER_TOKENS = frozenset({"//", "/*", "*/", "*", "#", "##", "<!--", "-->"})


def _normalized_text(text: str) -> str:
    """`text` as one string of single-spaced words, so a fragment survives its line wraps.

    The wording these tests pin travels in comments - TypeScript, CSS and the sidecar's own -
    which the formatter and the docstring convention wrap freely: the same sentence is one fact
    whatever width it was written at, and a reader that demanded the original line breaks would
    fail on the first reformat rather than on any real drift. Comment markers are dropped as
    whole tokens (`_COMMENT_MARKER_TOKENS`), which is what keeps a fragment readable across the
    `//` its continuation lines carry.
    """
    return " ".join(w for w in text.split() if w not in _COMMENT_MARKER_TOKENS)


def _one_wording_surface(relative: str) -> str:
    """One wording surface as text. A moved or renamed file fails here, not as an empty read."""
    path = REPO_ROOT / relative
    if not path.is_file():
        raise AssertionError(f"the surface this guard reads is gone: {relative}")
    return path.read_text(encoding="utf-8")


#: What a null camera control means, and where a maintainer can read it. `_undo_for` in
#: `app/camera.py` owns the answer - measured on this hardware, only autofocus has an automatic
#: mode the app can name (`CAP_PROP_AUTOFOCUS=1`), and the getter lies about the rest
#: (`get(CAP_PROP_BRIGHTNESS)` answered 0.0 over a normal picture, and writing that "restored"
#: value back turned the image black), so a reset stops this app writing the control and the
#: device keeps the last value it was given. Every surface that *describes* that behaviour, on
#: either side of the wire, has to keep saying it: the wording is the operator's only explanation
#: of what Revert does, and it lives in comments the compiler and the formatter never look at —
#: except on the tuning card, where `settingsFields.ts`'s field hints render it, so there the
#: wording is the explanation rather than a note beside the code. A rename can survive that
#: silence; a reword that drifts back toward "auto" cannot.
NULL_CONTROL_SURFACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "sidecar/app/camera.py",
        (
            "Autofocus is the whole list",
            "reading it back would pin focus on the way to undoing the drag",
            "leaves the device holding the last value it was",
        ),
    ),
    (
        "sidecar/app/settings.py",
        (
            'None means "this app imposes no value"',
            "(see CameraCapture._with_restores)",
        ),
    ),
    (
        "sidecar/app/settings_store.py",
        ('None ("this app imposes no value")',),
    ),
    (
        "sidecar/app/schemas.py",
        ('cannot restore "this app imposes no value"',),
    ),
    (
        "sidecar/tests/test_live_settings.py",
        (
            "None means 'this app imposes no value', and telling the camera is the",
            "What the device then does about it is camera.py's half",
        ),
    ),
    (
        "CLAUDE.md",
        (
            "A queued `None` is **not** a skip",
            "the device keeps the last value it was given until something writes it again",
        ),
    ),
    (
        "desktop/src/renderer/src/lib/api.ts",
        (
            'null means "this app imposes no value"',
            "(CameraCapture._undo_for)",
            "cannot travel as a value",
        ),
    ),
    # The four camera-control hints the tuning card renders — the one surface the
    # operator reads without opening an editor, so here the wording is the
    # explanation rather than a note beside the code. Each control's own claim is
    # pinned: unset imposes no value and the device keeps what it was given
    # (brightness, focus), unset stops the writing and nothing hands it back
    # (exposure), unset *is* the hand-back (autofocus) — the one control whose
    # hint must not drift toward the no-hand-back sentence.
    (
        "desktop/src/renderer/src/lib/settingsFields.ts",
        (
            "Unset means this app imposes no value",
            "the device keeps the last value it was given",
            "Unset stops this app writing the control",
            "no automatic mode the app can name hands it back",
            "Unset gives the lens back its own automatic mode",
        ),
    ),
    (
        "desktop/src/renderer/src/components/CameraTuning.tsx",
        (
            'placeholder="unset"',
            '"unset", not "auto": for three of these four controls the',
            "For autofocus the sidecar hands the control back to the lens",
            "CameraCapture's `_undo_for`",
        ),
    ),
    (
        "desktop/src/renderer/src/components/CameraTuning.css",
        (
            '"unset" — the app is not setting this control',
            "Not \"auto\": only autofocus can be handed back to the device's own mode",
        ),
    ),
    (
        "desktop/src/renderer/src/components/CameraTuning.test.tsx",
        (
            'reads "unset" — not "auto" — for a control the sidecar is not setting',
            'reverts a control back to "this app imposes no value"',
        ),
    ),
)

#: The sentence this wording replaced. It promised the camera an automatic mode only autofocus
#: has, which is the exact claim the fix was written to stop making - so it is not just unused,
#: it is wrong, and coming back is a regression rather than a leftover.
RETIRED_NULL_WORDING = ("leave the camera alone",)


def test_the_null_control_wording_survives_on_every_surface():
    """Every surface that says what a null camera control means still says what it is.

    The semantics have one owner (`_undo_for`) and eleven tellers, and the tellers are comments
    and rendered copy (the tuning card's field hints): nothing fails to compile when one is
    deleted or reworded toward "auto", so the drift this pins is the quiet kind - the card keeps
    working while its explanation starts promising a restoration the camera cannot do. The fragments are the load-bearing phrases, not the whole
    sentences: rewriting a surface is fine, losing the fact is not. A fragment that no longer
    fits what the code does should be updated in the same edit as the code, which is the
    contract the failure message below states.
    """
    missing: list[str] = []
    for relative, fragments in NULL_CONTROL_SURFACES:
        flat = _normalized_text(_one_wording_surface(relative))
        missing.extend(
            f"{relative}: {fragment!r}" for fragment in fragments if fragment not in flat
        )
    assert not missing, (
        "a surface that describes what a null camera control means no longer carries its "
        "wording. The semantics live in `sidecar/app/camera.py::_undo_for` - a queued None "
        "stops this app writing the control: autofocus handed back to the lens, the other "
        "three left holding whatever the device was last given, because nothing on the device "
        "says what that was. Reword the surface or move the wording into whatever replaced "
        "it, and update this guard in the same edit:\n  " + "\n  ".join(missing)
    )


def test_the_retired_null_wording_stays_retired():
    """The retired phrase promised a restoration the camera cannot do; it stays gone.

    "leave the camera alone" predates the fix that measured the getter lying
    (`get(CAP_PROP_BRIGHTNESS)` answered 0.0 over a normal picture, and writing the read-back
    value turned the image black), so unlike an ordinary old spelling it is not neutral if it
    returns - it restates the bug. Only the surfaces the wording guard reads are scanned,
    because those are the ones a maintainer edits while thinking about this exact sentence.
    """
    stale: list[str] = []
    for relative, _fragments in NULL_CONTROL_SURFACES:
        flat = _normalized_text(_one_wording_surface(relative))
        stale.extend(relative for phrase in RETIRED_NULL_WORDING if phrase in flat)
    assert not stale, (
        "the retired null-control wording came back. It promised the camera an automatic "
        "mode only autofocus has - measured, the other three controls keep whatever this app "
        "last wrote, and neither a capture restart nor a reopen gives it back. Reword the new "
        "text rather than reverting to the sentence that promised the reset would work; the "
        "owner of the answer is `sidecar/app/camera.py::_undo_for`:\n  " + "\n  ".join(stale)
    )


def test_the_null_wording_guard_is_wired_to_both_of_its_ends():
    """The guard's own canary: a fragment set that matched nothing, or an absence check that
    matched everything, would let the tests above pass while checking nothing.

    The fixture is text shaped like the surfaces - one sentence carrying a kept fragment, one
    carrying the retired phrase - and both halves have to fire on it. This is the same reason
    `test_docs_paths.py` floors its invocation and note counts: a reader that stopped deciding
    is invisible from its verdicts alone.
    """
    fixture = _normalized_text(
        'the old sentence promised null means "leave the camera alone", but the reset stops '
        "this app writing the control, and the device keeps the last value it was given "
        "until something writes it again"
    )
    kept = [
        fragment
        for _relative, fragments in NULL_CONTROL_SURFACES
        for fragment in fragments
        if fragment in fixture
    ]
    assert "the device keeps the last value it was given until something writes it again" in kept, (
        "the kept-wording half of this guard matched nothing: the surfaces could all go quiet "
        "and the presence test above would never fire"
    )
    assert "leave the camera alone" in fixture, (
        "the retired-wording half of this guard matched nothing: the old sentence could come "
        "back and the absence test above would never fire"
    )


def test_the_rest_response_models_mirror_the_sidecar():
    """The remaining response models, one pair each.

    These are read by the panel rather than every second, so a drift shows up as a blank field or a
    row that never renders instead of a broken preview - quieter, and therefore later.
    """
    source = read_ts(API_TS)
    for model, ts_name in (
        (schemas.HealthResponse, "HealthResponse"),
        (schemas.InferenceStatusPayload, "InferenceStatusPayload"),
        (schemas.LogEvent, "LogEvent"),
        (schemas.LogsResponse, "LogsResponse"),
        (schemas.CameraInfo, "CameraInfo"),
        (schemas.CamerasResponse, "CamerasResponse"),
        (schemas.CameraQualityResponse, "CameraQualityResponse"),
        (schemas.DetectorProbeResponse, "DetectorProbeResponse"),
        (schemas.SystemInfoResponse, "SystemInfoResponse"),
        (schemas.PresetInfo, "PresetInfo"),
        (schemas.PresetsResponse, "PresetsResponse"),
        (schemas.ClassRecall, "ClassRecall"),
        (schemas.DistanceRecall, "DistanceRecall"),
        (schemas.ValidationRecord, "ValidationRecord"),
        (schemas.InstalledModel, "InstalledModel"),
        (schemas.ModelsResponse, "ModelsResponse"),
        (schemas.ControlSupportPayload, "CameraControlSupport"),
        (schemas.CameraProfileResponse, "CameraProfileResponse"),
        (schemas.StoredProfileResponse, "StoredProfileResponse"),
        (schemas.DatasetClassProgress, "DatasetClassProgress"),
        (schemas.DatasetTierACell, "DatasetTierACell"),
        (schemas.DatasetSessionProgress, "DatasetSessionProgress"),
        (schemas.DatasetBacklogCell, "DatasetBacklogCell"),
        (schemas.DatasetStatusResponse, "DatasetStatusResponse"),
    ):
        ts_name = RENAMED.get(model.__name__, ts_name)
        assert_same_fields(
            f"{model.__name__} vs api.ts's {ts_name}",
            python_fields(model),
            ts_fields(source, ts_name),
        )


def test_the_per_backend_track_expiry_floor_mirrors_the_sidecar():
    """The one *numeric* mirror: below it, a slow round trip outlives the tracker's memory.

    The desktop states the floor to warn an operator before they set a value the sidecar's own
    warning would then complain about - so a floor that drifted would put two different numbers on
    the same screen, one of them telling the operator to do the thing the other refuses.
    """
    source = read_ts(FIELDS_TS)
    desktop_floors = {
        key: ts_value(value) for key, value in ts_record(source, "MIN_TRACK_EXPIRY_S_BY_BACKEND").items()
    }
    assert desktop_floors == MIN_TRACK_EXPIRY_S_BY_BACKEND, (
        "the per-backend track-expiry floors have drifted: "
        f"desktop={desktop_floors} sidecar={MIN_TRACK_EXPIRY_S_BY_BACKEND}"
    )
    assert ts_scalar(source, "MIN_REMOTE_TRACK_EXPIRY_S") == MIN_REMOTE_TRACK_EXPIRY_S, (
        "the conservative fallback for an unlisted backend has drifted: "
        f"desktop={ts_scalar(source, 'MIN_REMOTE_TRACK_EXPIRY_S')} "
        f"sidecar={MIN_REMOTE_TRACK_EXPIRY_S}"
    )
    # The accessor is the desktop's own wiring rather than a mirrored value, so it is read
    # textually: a fallback replaced by a literal would put a second number on screen that the
    # comparisons above could not see, since it would no longer be the exported constant.
    function = re.search(r"export function minTrackExpiryS\b.*?\n\}", source, re.DOTALL)
    assert function is not None, "the desktop no longer exports minTrackExpiryS"
    assert "MIN_TRACK_EXPIRY_S_BY_BACKEND" in function.group(0), (
        "minTrackExpiryS no longer reads the mirrored map"
    )
    assert "MIN_REMOTE_TRACK_EXPIRY_S" in function.group(0), (
        "minTrackExpiryS no longer falls back to the mirrored constant"
    )


def test_the_experimental_model_set_mirrors_the_sidecar():
    """`EXPERIMENTAL_MODELS` decides which weights get the "calibrated for YOLO11" warning.

    Both sides call it a mirror in a comment. A name in one and not the other is either a warning
    nobody sees or a warning nobody can act on.
    """
    source = read_ts(FIELDS_TS)
    literals, names = ts_entries(ts_array_items(source, "EXPERIMENTAL_MODELS"))
    assert not names, "EXPERIMENTAL_MODELS should be literals"
    assert literals == set(EXPERIMENTAL_MODELS), (
        "the experimental-model set has drifted: "
        f"desktop={sorted(literals)} sidecar={sorted(EXPERIMENTAL_MODELS)}"
    )


def test_the_desktop_fixture_roster_mirrors_the_sidecar_roster():
    """The names the desktop's own tests assert against, which are a copy of the roster.

    A fixture is a kind of mirror that fails quietly in a different way: it is what the panel's
    tests are written against, so a stale fixture keeps a stale panel green. This one carried
    Palmolive - a class dropped from the dataset - while the runtime roster had moved on, and the
    drift was invisible because nothing compared the two.
    """
    source = read_ts(FAKES_TS)
    literals, names = ts_entries(ts_array_items(source, "ROSTER_NAMES"))
    assert not names, "ROSTER_NAMES should be literals"
    assert literals == set(V2_ROSTER), (
        "the desktop's fixture roster has drifted from the runtime's: "
        f"desktop={sorted(literals)} sidecar={sorted(V2_ROSTER)}"
    )


def _read(path):
    """A file this guard reads, failing loudly rather than comparing two empty strings."""
    assert path.is_file(), f"the file this guard reads is gone: {path}"
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """A TypeScript file with its line breaks collapsed and its comments dropped.

    So a match does not pin the formatting prettier chose (this repo's interfaces are wrapped by
    the formatter, and a guard that broke on a re-wrap would be a guard people delete) and cannot
    be satisfied by a comment saying what the code used to do.
    """
    return re.sub(r"\s+", " ", strip_ts_comments(text))


def _one(pattern: str, text: str, what: str) -> str:
    match = re.search(pattern, text)
    assert match is not None, f"could not find {what} - searched for {pattern!r}"
    return match.group(1)


def _panel_entries() -> dict[str, dict[str, str]]:
    """`SETTINGS_FIELDS` keyed by the setting each entry edits, as written.

    A duplicate key fails rather than shadowing: two entries for one setting means one of them is
    invisible in the panel while still looking wired up in the file.
    """
    entries = ts_object_array(read_ts(FIELDS_TS), "SETTINGS_FIELDS")
    keys = [ts_value(entry["key"]) for entry in entries]
    assert len(set(keys)) == len(keys), f"the panel edits a setting twice: {sorted(keys)}"
    return {ts_value(entry["key"]): entry for entry in entries}


def _api_numeric_bounds() -> dict[str, dict[str, object]]:
    """The numeric bounds `SettingsUpdateRequest` declares, per field.

    Read off the model rather than off its source: `ge`/`le`/`gt`/`lt` land in the field's metadata
    as pydantic's own bound objects, so this is the same information the API rejects a PATCH with
    rather than a second reading of it.

    `min_length` is deliberately not collected - a string's minimum length is not a range the panel
    offers, so counting it would fail on `roboflow_workspace` for no reason.
    """
    bounds: dict[str, dict[str, object]] = {}
    for name, field in schemas.SettingsUpdateRequest.model_fields.items():
        low: float | None = None
        high: float | None = None
        low_exclusive = high_exclusive = False
        for bound in field.metadata:
            if getattr(bound, "ge", None) is not None:
                low = bound.ge
            elif getattr(bound, "gt", None) is not None:
                low, low_exclusive = bound.gt, True
            if getattr(bound, "le", None) is not None:
                high = bound.le
            elif getattr(bound, "lt", None) is not None:
                high, high_exclusive = bound.lt, True
        if low is not None or high is not None:
            bounds[name] = {
                "low": low,
                "high": high,
                "low_exclusive": low_exclusive,
                "high_exclusive": high_exclusive,
            }
    assert bounds, "no numeric bounds were read from SettingsUpdateRequest at all"
    return bounds


def test_the_panel_edits_exactly_the_settings_the_sidecar_declares():
    """The panel's field list against the runtime's, so neither can gain or lose one alone.

    A field on the panel's side only is a control that PATCHes a name nothing recognises: the
    operator moves it, the response echoes the old value back, and the two screens disagree about
    a setting that was never stored. A field on the sidecar's side only is a setting nobody can
    reach. Neither is an error anywhere - the PATCH is simply applied to what it did recognise -
    which is why the lists are compared whole rather than spot-checked.
    """
    panel = set(_panel_entries())
    runtime = set(Settings.__dataclass_fields__)
    assert panel == runtime, (
        "the panel's field list and the runtime's settings have drifted:\n"
        f"  the panel edits, the sidecar does not have: {sorted(panel - runtime)}\n"
        f"  the sidecar has, the panel does not edit: {sorted(runtime - panel)}"
    )
    # And the wire model is that same list, so the three spellings of a setting cannot disagree.
    assert set(schemas.SettingsPayload.model_fields) == runtime, (
        "SettingsPayload and Settings have drifted: "
        f"{sorted(set(schemas.SettingsPayload.model_fields) ^ runtime)}"
    )


def test_the_panel_field_bounds_mirror_the_api():
    """Every numeric bound, both ways, because the two sides mean different things by it.

    The panel's `min`/`max` are what the operator can type; the API's `ge`/`le` are what a save
    accepts. A panel bound looser than the API's is a control that offers a value the save then
    rejects, and one tighter silently withholds range the app can use. An explicit `gt` is the one
    asymmetry allowed: `track_expiry_s` cannot be offered as a slider minimum of zero (the API
    refuses it), so the rule there is that the panel's minimum is strictly above it.
    """
    bounds = _api_numeric_bounds()
    panel = _panel_entries()
    problems: list[str] = []
    for key, entry in panel.items():
        has_min, has_max = "min" in entry, "max" in entry
        api = bounds.get(key)
        if not (has_min or has_max):
            if api is not None:
                problems.append(
                    f"{key}: the API bounds it ({api['low']}..{api['high']}"
                    f"{', exclusive' if api['low_exclusive'] or api['high_exclusive'] else ''})"
                    " and the panel offers no range at all"
                )
            continue
        if api is None:
            problems.append(
                f"{key}: the panel offers a range the API does not bound, so both endpoints are "
                "this file's invention"
            )
            continue
        if ts_value(entry["type"]) != "number":
            problems.append(
                f"{key}: bounded but rendered as a {ts_value(entry['type'])}, where a min/max is "
                "inert"
            )
        low = ts_value(entry["min"]) if has_min else None
        high = ts_value(entry["max"]) if has_max else None
        if api["low"] is not None:
            if api["low_exclusive"]:
                if low is None or not low > api["low"]:
                    problems.append(
                        f"{key}: the API requires more than {api['low']} and the panel's minimum "
                        f"is {low}"
                    )
            elif low != api["low"]:
                problems.append(
                    f"{key}: the API accepts from {api['low']} and the panel starts at {low}"
                )
        if api["high"] is not None:
            if api["high_exclusive"]:
                if high is None or not high < api["high"]:
                    problems.append(
                        f"{key}: the API requires less than {api['high']} and the panel's maximum "
                        f"is {high}"
                    )
            elif high != api["high"]:
                problems.append(
                    f"{key}: the API accepts to {api['high']} and the panel ends at {high}"
                )
    for key in sorted(set(bounds) - set(panel)):
        problems.append(f"{key}: the API bounds it and the panel does not edit it at all")
    assert not problems, "the panel's field bounds have drifted from the API's:\n  " + "\n  ".join(
        problems
    )


def test_the_camera_control_ranges_have_one_owner():
    """The probe must not declare its own copy of the bounds it has to stay inside.

    It used to: three `*_RANGE` constants in `camera_caps.py` mirroring `SettingsUpdateRequest`'s
    `ge`/`le`, which the previous version of this file compared as text (the module imports `cv2`,
    so the fast mirror job cannot import it). They live in `app/settings_store.py` now, where the
    API's field bounds and the panel's slider bounds read them too - so the pair is gone rather
    than checked, and the only thing left to fail on is a copy creeping back. That copy is worth
    failing on because of what it costs: a probe sweeping past a bound spends its probes measuring
    a value `calibrate` then recommends, which the operator's own save rejects.
    """
    redeclared = re.findall(
        r"^(EXPOSURE|BRIGHTNESS|FOCUS)_RANGE\s*=", _read(CAMERA_CAPS_PY), re.MULTILINE
    )
    assert not redeclared, (
        f"camera_caps.py declares {redeclared} again - import them from `app.settings_store` "
        "(which is also where `SettingsUpdateRequest` and the panel get them) rather than "
        "restating numbers the probe has to stay inside"
    )


def test_the_spawn_handshake_mirrors_the_sidecar():
    """The names that cross the process boundary, in both directions.

    `desktop/src/main` spawns `sidecar/run.py` and then knows only what it is told: the port
    arrives as one printed line, the app's own pid as one environment variable, and the renderer
    asks for the port - and, since the failure it cannot see for itself, for whether the sidecar is
    answering at all - over two IPC channels. Neither side can import the other and nothing
    type-checks the pair, so a rename on one side fails in the quietest way this repo has: the
    supervisor waits for a line that never comes while a perfectly healthy sidecar streams beside
    it, and the window sits on "waiting for the sidecar".

    Each name is read from the side that owns it and compared to the side that has to match, so
    the failure says which of the two moved rather than printing two files.
    """
    run_py = _read(RUN_PY)
    supervisor = _flat(_read(SPAWN_TS))
    main = _flat(_read(MAIN_TS))
    preload = _flat(_read(PRELOAD_TS))
    makefile = _read(MAKEFILE)

    # The handshake line: what `run.py` prints versus what the supervisor's regex parses.
    printed = _one(r'print\(f"([A-Z_]+=)', run_py, "the handshake line run.py prints")
    parsed = _one(r"const PORT_RE = /([A-Z_]+=)\(\\d\+\)/", supervisor, "the port regex")
    assert printed == parsed, (
        f"run.py prints {printed!r} and the desktop parses {parsed!r} - the app would never learn "
        "its sidecar's port"
    )
    # And the digits really are the port: a regex that matched something else would pass above.
    assert "onPort?.(Number(m[1]))" in supervisor, "the supervisor no longer reports the parsed port"

    # The parent-pid variable: the desktop puts its pid in it, the sidecar reads it.
    set_by_desktop = _one(
        r"process\.env, ([A-Z_]+): String\(process\.pid\)",
        supervisor,
        "the parent-pid variable the supervisor sets",
    )
    read_by_sidecar = _one(
        r'os\.environ\.get\("([A-Z_]+)"\)', run_py, "the parent-pid variable run.py reads"
    )
    assert set_by_desktop == read_by_sidecar, (
        f"the desktop sets {set_by_desktop!r} and the sidecar reads {read_by_sidecar!r} - the "
        "watchdog would keep a sidecar alive after the app is gone"
    )

    # The IPC chain: main serves each channel, the preload calls every one of them, the types
    # declare the accessors, and the renderer is what uses them.
    #
    # The channels are read as a *set*, not as the first match. `re.search` here is what a second
    # channel would slip past, and a channel served by nothing that calls it is the same defect as
    # a renamed one from the renderer's side - it just fails without a name to grep for.
    channels = re.findall(r"ipcMain\.handle\('([^']+)'", main)
    assert channels, "the main process no longer serves an IPC channel"
    for channel in channels:
        assert f"invoke('{channel}')" in preload, (
            f"the main process serves {channel!r} and the preload does not call it"
        )

    accessor = _one(
        r"([A-Za-z_][A-Za-z0-9_]*): \(\): Promise<number \| null>",
        preload,
        "the preload's port accessor",
    )
    assert re.search(rf"\b{accessor}\b", _read(PRELOAD_DTS)), (
        f"preload/index.d.ts does not declare {accessor!r}, so the renderer's call is untyped"
    )
    assert f".api.{accessor}()" in _flat(_read(RENDERER_APP_TS)), (
        f"the renderer never calls window.api.{accessor}() - nothing asks for the port"
    )

    # The health pair, the same four steps. Its absence is worse than the port's: an app whose
    # sidecar has stopped answering looks exactly like a working one that has not been used yet.
    health = _one(
        r"([A-Za-z_][A-Za-z0-9_]*): \(\): Promise<SidecarHealth>",
        preload,
        "the preload's health accessor",
    )
    assert re.search(rf"\b{health}\b", _read(PRELOAD_DTS)), (
        f"preload/index.d.ts does not declare {health!r}, so the renderer has to restate the union"
    )
    assert f"window.api.{health}()" in _flat(_read(RENDERER_HEALTH_TS)), (
        f"nothing calls window.api.{health}() - the health state reaches no screen"
    )

    # Where the sidecar lives: the desktop resolves it and the Makefile launches the same thing.
    sidecar_dir = _one(r"SIDECAR_DIR\s*:=\s*(\S+)", makefile, "the Makefile's sidecar directory")
    resolved_dir = _one(r"join\(app\.getAppPath\(\), '\.\.', '([^']+)'\)", main, "the sidecar dir")
    assert sidecar_dir == resolved_dir, (
        f"the Makefile runs the sidecar in {sidecar_dir!r} and the desktop spawns it in "
        f"{resolved_dir!r}"
    )

    script = _one(
        r"SIDECAR_SCRIPT'\] \|\| join\(sidecarDir, '([^']+)'\)",
        main,
        "the sidecar entry point the desktop spawns",
    )
    assert script in makefile, f"the desktop spawns {script!r} and the Makefile runs something else"
    assert (REPO_ROOT / sidecar_dir / script).is_file(), (
        f"the entry point both sides name does not exist: {sidecar_dir}/{script}"
    )

    venv_paths = re.findall(r"SIDECAR_VENV_PY\s*:=\s*(\S+)", makefile)
    assert venv_paths, "the Makefile no longer names the sidecar's interpreter"
    tokens = set(re.findall(r"'([^']*)'", main))
    for path in venv_paths:
        parts = path.split("/")
        assert all(part in tokens for part in parts), (
            f"the Makefile launches the sidecar with {path!r} and the desktop resolves a different "
            f"interpreter - its path is built from {sorted(tokens)} in main/index.ts"
        )


def test_the_model_labels_state_their_roster_s_class_count():
    """A label is prose, except for the one number in it that a reader cannot check by hand.

    `'SCANnCART grocery v2 (custom, 8 SKUs)'` is read as a fact about the selected weights - how many
    products this model can name - and it is the only place an operator sees that number before
    running anything. It is also the number a dropped class changes, which is exactly the kind of
    edit that updates the roster and forgets the sentence describing it.
    """
    source = read_ts(FIELDS_TS)
    for record, label in (
        ("MODEL_LABELS", "MODEL_LABELS"),
        ("MODEL_SPEC_HINTS", "MODEL_SPEC_HINTS"),
    ):
        for model, text in ts_record(source, record).items():
            entry = ts_value(text)
            generation = re.search(r"\bv(\d)\b", entry)
            count = re.search(r"\(?(\d+) SKUs?\)?", entry)
            if count is None:
                continue
            assert generation is not None, (
                f"{label}[{model!r}] states a class count without saying which generation it "
                f"belongs to, so nothing can check it: {entry!r}"
            )
            name = f"v{generation.group(1)}"
            assert name in ROSTERS, (
                f"{label}[{model!r}] names generation {name!r}, which this app has no roster for"
            )
            assert int(count.group(1)) == len(ROSTERS[name]), (
                f"{label}[{model!r}] says {count.group(1)} SKUs but this app's {name} roster holds "
                f"{len(ROSTERS[name])}: {sorted(ROSTERS[name])}"
            )
