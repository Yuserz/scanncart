import re
from dataclasses import fields
from pathlib import Path

from app.settings import Settings, resolve_device
from app.settings_store import (
    ALLOWED_BACKENDS,
    ALLOWED_DEVICES,
    ALLOWED_MODELS,
    ALLOWED_RESIZE_MODES,
    REMOTE_BACKENDS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_DEFAULTS = (
    REPO_ROOT / "desktop" / "src" / "renderer" / "src" / "lib" / "settingsDefaults.ts"
)


# --- the desktop mirror of these defaults ----------------------------------
#
# `settingsDefaults.ts` restates every default below by hand, and "Restore Defaults" PATCHes the
# whole object into the running sidecar — so a drifted value is not a cosmetic problem: it is
# accepted as long as it is in bounds, silently becomes the running configuration, and no test
# anywhere reads both files. That is the gap these four tests close.
#
# The parser is deliberately strict. A guard that answers `None` for a literal it does not
# recognise would agree with Python on the four fields whose default *is* `None`, so a mirror that
# had stopped being readable at all would still look pinned.


def _strip_ts_comment(line: str) -> str:
    """`line` without its `//` comment, ignoring a `//` inside a quoted string.

    `line.split("//")[0]` is the tempting version and it corrupts `'http://127.0.0.1:9001'` — a
    value really in this file — into `'http:`. The local API URL is the one default whose entire
    meaning is its scheme and port, so that shortcut would mangle the value it exists to protect.
    """
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "/" and line[i : i + 2] == "//":
            return line[:i]
        i += 1
    return line


def _ts_value(token: str):
    """One TypeScript literal as the Python value it mirrors, or a loud failure."""
    if token == "null":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "[]":
        return []
    if len(token) >= 2 and token[0] in "'\"" and token[-1] == token[0]:
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        raise ValueError(
            f"settingsDefaults.ts holds a literal this guard cannot read: {token!r}. Extend "
            "_ts_value to cover it — leaving it unread would let the mirror drift unchecked, "
            "which is the one thing this file exists to prevent."
        ) from None


def _defaults_body(text: str) -> list[str]:
    """The lines inside `DEFAULT_SETTINGS`'s object literal, braces excluded."""
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if re.match(r"\s*export const DEFAULT_SETTINGS\b", ln)),
        None,
    )
    if start is None:
        raise AssertionError("settingsDefaults.ts no longer exports DEFAULT_SETTINGS")

    depth = 0
    opened = False
    body: list[str] = []
    for line in lines[start:]:
        depth += line.count("{") - line.count("}")
        if not opened:
            if "{" in line:
                opened = True
            continue
        if depth == 0:
            break  # the closing brace of the literal
        body.append(line)
    if not opened:
        raise AssertionError("DEFAULT_SETTINGS is not an object literal")
    return body


def parse_desktop_defaults(text: str) -> dict:
    """`DEFAULT_SETTINGS` as a dict. Pure, so the parser can be tested without the file."""
    out: dict = {}
    for raw in _defaults_body(text):
        line = _strip_ts_comment(raw).strip()
        if not line:
            continue
        # The key separator is the *first* colon: several values are URLs and carry their own
        # (`'http://127.0.0.1:9001'`), which rsplit or a naive split would cut in half.
        key, sep, value = line.partition(":")
        if not sep:
            raise AssertionError(f"not a key/value line in DEFAULT_SETTINGS: {raw!r}")
        out[key.strip()] = _ts_value(value.strip().rstrip(",").strip())
    return out


def _defaults_diff(actual: dict, mirrored: dict) -> str:
    only_ts = sorted(set(mirrored) - set(actual))
    only_python = sorted(set(actual) - set(mirrored))
    drifted = sorted(k for k in set(actual) & set(mirrored) if actual[k] != mirrored[k])

    parts = []
    if only_ts:
        parts.append(f"in settingsDefaults.ts but not in Settings: {only_ts}")
    if only_python:
        parts.append(f"in Settings but missing from settingsDefaults.ts: {only_python}")
    for key in drifted:
        parts.append(f"{key}: Settings={actual[key]!r} ts={mirrored[key]!r}")
    return "the desktop defaults mirror has drifted from Settings:\n  " + "\n  ".join(parts)


def test_the_desktop_defaults_mirror_matches_settings():
    """Every default, both directions: a value that drifted, a key added to one side only.

    Values are read off a real `Settings()` rather than restated here, which is what makes this
    different from `test_settings_defaults` above — that one re-asserts a handful of literals, so
    it cannot see the desktop file at all and would pass with the mirror deleted.
    """
    mirrored = parse_desktop_defaults(DESKTOP_DEFAULTS.read_text(encoding="utf-8"))
    actual = {f.name: getattr(Settings(), f.name) for f in fields(Settings)}

    assert mirrored == actual, _defaults_diff(actual, mirrored)


def test_the_defaults_parser_keeps_a_url_whole():
    """The `//` trap, pinned: the parser must not treat a URL's scheme as a comment."""
    parsed = parse_desktop_defaults(
        "export const DEFAULT_SETTINGS: SettingsPayload = {\n"
        "  local_api_url: 'http://127.0.0.1:9001',\n"
        "  cloud_api_url: 'https://serverless.roboflow.com'\n"
        "  // a real comment, which must go\n"
        "  active_model: 'models/x.onnx'\n"
        "}\n"
    )

    assert parsed == {
        "local_api_url": "http://127.0.0.1:9001",
        "cloud_api_url": "https://serverless.roboflow.com",
        "active_model": "models/x.onnx",
    }


def test_the_defaults_parser_rejects_a_literal_it_cannot_read():
    """The property that keeps the guard honest, rather than merely green.

    Returning `None` for an unreadable literal would agree with Python on every field whose
    default *is* `None`, so a mirror this parser had silently stopped understanding would still
    read as pinned. Failing here instead means the guard can only pass by reading the file.
    """
    import pytest

    with pytest.raises(ValueError, match="cannot read"):
        parse_desktop_defaults(
            "export const DEFAULT_SETTINGS: SettingsPayload = {\n"
            "  active_model: someHelper(),\n"
            "}\n"
        )


def test_the_defaults_parser_notices_a_renamed_object():
    """A rename must fail as \"I cannot find it\", never as \"nothing to compare\" — otherwise the
    guard quietly stops guarding the day someone renames the export."""
    import pytest

    with pytest.raises(AssertionError, match="DEFAULT_SETTINGS"):
        parse_desktop_defaults("export const OTHER: SettingsPayload = {\n  a: 1\n}\n")


def test_settings_defaults():
    s = Settings()
    # The locally trained grocery model, run in-process — see
    # docs/MODEL_TRAINING.md and docs/DETECTOR_BACKENDS.md §1a.
    assert s.active_model == "models/scanncart-grocery.pt"
    # USB-2.0-safe first-run settings.
    assert s.capture_width == 640
    assert s.capture_height == 480
    assert s.capture_fps == 30
    assert s.conf_threshold == 0.5
    assert s.imgsz == 640
    # `auto`, not `letterbox`: it reads the requirement recorded beside the selected weights, which
    # is the only thing that can tell a `.pt` trained on a `Stretch to` version from one trained
    # letterboxed. See `resolve_resize_mode` / `resize_guess`.
    assert s.resize_mode == "auto"
    assert s.infer_frame_skip == 0
    assert s.device == "auto"
    assert s.preview_height == 720
    assert s.camera_brightness is None
    assert s.camera_exposure is None
    assert s.camera_autofocus is None
    assert s.camera_focus is None


def _fake_torch(monkeypatch, cuda_available: bool):
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_resolve_device_cpu_always_passthrough(monkeypatch):
    # "cpu" is honored even when a CUDA GPU is available.
    _fake_torch(monkeypatch, cuda_available=True)
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_and_auto_use_gpu_when_available(monkeypatch):
    _fake_torch(monkeypatch, cuda_available=True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") == "cuda"


def test_resolve_device_forced_cuda_falls_back_when_unavailable(monkeypatch):
    # A stale/forced "cuda" must NOT be returned when torch can't use it,
    # otherwise capture crashes at start on a CPU-only torch install.
    _fake_torch(monkeypatch, cuda_available=False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("auto") == "cpu"


# --- the desktop mirror of the sidecar's vocabularies ----------------------
#
# `settingsDefaults.ts` is not the only file the desktop keeps in step by hand: `settingsFields.ts`
# restates the model list, the three resize modes, the devices and the two backend sets, and the
# picker renders straight from it. Nothing on the desktop side can import `settings_store.py` to
# check any of that, so a value changed on one side and not the other lands silently — and this is
# the surface a merge writes into from both directions at once, because the model default and the
# resize vocabulary are each spelled twice.
#
# The guards live here, reading the desktop files, for the same reason the defaults mirror does:
# the sidecar is the side that can see both.
#
# Two things they deliberately do *not* pin. The **field ranges** (`SETTINGS_FIELDS`' min/max):
# drift there produces a rejected PATCH the operator sees, not a silently different configuration.
# And `MODEL_LABELS`, which is prose about weights the sidecar has no opinion on.

FIELDS_TS = REPO_ROOT / "desktop" / "src" / "renderer" / "src" / "lib" / "settingsFields.ts"


def _ts_export_rhs(source: str, name: str) -> str:
    """The right-hand side of `export const NAME = ...`, an array literal returned whole.

    Whole, and with the type annotation and any trailing `as const` dropped, so a list written
    across lines reads exactly like a one-line one. Failing loudly on an absent name is the point:
    a guard that answered `""` would compare two empty sets and pass.
    """
    match = re.search(rf"^export const {re.escape(name)}\b[^=]*= *(.*)$", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"settingsFields.ts no longer exports {name!r}")
    rhs = match.group(1)
    if not rhs.startswith("["):
        return rhs.strip()
    tail = source[match.start(1) :]
    depth = 0
    for i, ch in enumerate(tail):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return tail[: i + 1]
    raise AssertionError(f"{name!r} is an unterminated array literal")


def _ts_array_items(source: str, name: str) -> list[str]:
    """The items of `export const NAME = [...]`, in order and as written.

    As written, because one of these lists holds an identifier (`CUSTOM_MODEL`) rather than a
    literal; evaluating it here would mean re-implementing the file instead of reading it.
    """
    rhs = _ts_export_rhs(source, name)
    if not (rhs.startswith("[") and rhs.endswith("]")):
        raise AssertionError(f"{name!r} is not an array literal")
    return [item.strip() for item in rhs[1:-1].split(",") if item.strip()]


def _ts_entries(items: list[str]) -> tuple[set[str], set[str]]:
    """`items` split into string literals (as Python values) and bare identifiers.

    A third kind of entry raises rather than being dropped: an unrecognised item silently left out
    of both sets is how a guard stops covering a line nobody looks at.
    """
    literals: set[str] = set()
    names: set[str] = set()
    for item in items:
        if item[:1] in ("'", '"'):
            literals.add(_ts_value(item))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item):
            names.add(item)
        else:
            raise AssertionError(f"an entry this guard cannot read: {item!r}")
    return literals, names


def test_the_desktop_model_list_mirror_matches_the_sidecar():
    """The picker's list, the sidecar's validator, and the shipped default, as one fact.

    The default is the load-bearing half: the desktop's constant and `Settings().active_model` are
    the same name written twice, so a change that moved one without the other would render a picker
    starting on a model the sidecar is not running. The stock names are compared as sets, so adding
    or dropping a weight is caught from either side.
    """
    source = FIELDS_TS.read_text(encoding="utf-8")
    custom = _ts_value(_ts_export_rhs(source, "CUSTOM_MODEL"))
    items = _ts_array_items(source, "ALLOWED_MODELS")
    literals, names = _ts_entries(items)

    assert names == {"CUSTOM_MODEL"}, "the desktop list names the custom model by constant"
    assert items[0] == "CUSTOM_MODEL", "the custom model is the default, so it stays first"
    assert custom == Settings().active_model, (
        f"the desktop's CUSTOM_MODEL is {custom!r} but the sidecar defaults to "
        f"{Settings().active_model!r}"
    )
    assert literals - {custom} == set(ALLOWED_MODELS), (
        "the desktop's model list and the sidecar's ALLOWED_MODELS have drifted: only in the "
        f"desktop {sorted(literals - {custom} - set(ALLOWED_MODELS))}, only in the sidecar "
        f"{sorted(set(ALLOWED_MODELS) - literals)}"
    )


def test_the_desktop_resize_vocabulary_mirror_matches_the_sidecar():
    """The three modes, the order the picker offers them in, and the mode it starts on.

    Ordering is a decision rather than a value: `letterbox` is listed first because it is what the
    checkout view wants, while the default stays `auto` because that is what reads the requirement
    recorded beside the weights. Both are asserted because collapsing them is the mistake this
    pairing invites — and `auto` is the half a "prefer letterbox" tidy-up would quietly delete.
    """
    source = FIELDS_TS.read_text(encoding="utf-8")
    items = _ts_array_items(source, "ALLOWED_RESIZE_MODES")
    literals, names = _ts_entries(items)

    assert not names, "every resize mode should be written as a literal"
    assert literals == ALLOWED_RESIZE_MODES, (
        f"the resize modes have drifted: desktop={sorted(literals)} "
        f"sidecar={sorted(ALLOWED_RESIZE_MODES)}"
    )
    assert _ts_value(items[0]) == "letterbox", "the aspect-preserving choice is offered first"
    assert Settings().resize_mode == "auto", (
        "the default must stay `auto`: it is what reads the geometry recorded beside the weights, "
        "and a fixed default runs a `Stretch to`-trained `.pt` at the wrong scale with no error"
    )


def test_the_desktop_device_and_backend_vocabularies_mirror_the_sidecar():
    """Devices, backends, and which backends are remote.

    The last one decides whether the UI offers the API-key field and the per-backend track-expiry
    floor, so a backend that left that set on one side only would be rendered as a local one.
    """
    source = FIELDS_TS.read_text(encoding="utf-8")
    for name, expected in (
        ("ALLOWED_DEVICES", ALLOWED_DEVICES),
        ("ALLOWED_BACKENDS", ALLOWED_BACKENDS),
        ("REMOTE_BACKENDS", REMOTE_BACKENDS),
    ):
        literals, names = _ts_entries(_ts_array_items(source, name))
        assert not names, f"{name} should be written as literals"
        assert literals == expected, (
            f"{name} has drifted from the sidecar: desktop={sorted(literals)} "
            f"sidecar={sorted(expected)}"
        )


def test_the_vocabulary_parser_fails_loudly_on_a_renamed_export():
    """The property that keeps these guards honest: a renamed or removed export has to fail as
    "I cannot find it", never as "both lists are empty" — and an entry the parser does not
    understand has to fail too, rather than being dropped out of the comparison."""
    import pytest

    source = FIELDS_TS.read_text(encoding="utf-8")
    with pytest.raises(AssertionError, match="no longer exports"):
        _ts_array_items(source, "ALLOWED_RESIZE_MODES_RENAMED")
    with pytest.raises(AssertionError, match="cannot read"):
        _ts_entries(["{ ...spread }"])


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    # Simulate torch missing -> must fall back to cpu, never crash.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_device("auto") == "cpu"
    # A forced "cuda" with no torch at all must also degrade, not crash.
    assert resolve_device("cuda") == "cpu"
