"""Readers for the desktop files that restate a sidecar fact by hand.

The two toolchains share no schema generation. `desktop/src/renderer/src/lib/*.ts` restates the
sidecar's settings defaults, its allowed vocabularies, the WS message shapes, the REST response
shapes, the roster and a per-backend numeric floor - and nothing on the TypeScript side can import
Python to check any of it. So the guards live on the sidecar's side and read these files, and the
parsing they need lives here rather than in whichever test file grew it first, the same arrangement
`dataset_tool_helpers.py` has for the tools.

Every reader fails loudly. A guard that answered `""` or `None` for a name or a literal it could not
find would compare two empty things and pass - and silence is exactly what a mirror guard exists to
break.

Scoped on purpose to what a hand-written interface is: names, optionality (`?`) and nullability
(`| null`). Type *shapes* are not compared - `tuple[float, float, float, float]` against
`[number, number, number, number]` would mean writing a TypeScript type checker, and a field whose
name and optionality agree but whose shape does not is a mistake the compiler on the other side
catches at the use site.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
TS_RENDERER = REPO_ROOT / "desktop" / "src" / "renderer" / "src"
TS_LIB = TS_RENDERER / "lib"
TS_TEST = TS_RENDERER / "test"


def read_ts(path: Path) -> str:
    """A desktop source file as text. A missing file fails here, not as an empty read."""
    if not path.is_file():
        raise AssertionError(f"the desktop file this guard reads is gone: {path}")
    return path.read_text(encoding="utf-8")


# --- TypeScript literals ----------------------------------------------------


def strip_ts_comment(line: str) -> str:
    """`line` without its `//` comment, ignoring a `//` inside a quoted string.

    `line.split("//")[0]` is the tempting version and it corrupts `'http://127.0.0.1:9001'` - a
    value really in these files - into `'http:`. The local API URL is the one default whose entire
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


def strip_ts_comments(text: str) -> str:
    return "\n".join(strip_ts_comment(line) for line in text.splitlines())


def ts_value(token: str) -> Any:
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
            f"a desktop literal this guard cannot read: {token!r}. Extend `ts_value` to cover it - "
            "leaving it unread would let the mirror drift unchecked, which is the one thing these "
            "guards exist to prevent."
        ) from None


def bracket_delta(line: str, depth: int) -> int:
    """How `line` changes the bracket depth, for the running `depth` before it.

    `>` closes only at depth > 0: an arrow function type (`=> void`) would otherwise take the count
    negative and swallow every separator after it.
    """
    delta = 0
    for ch in line:
        if ch in "([{<":
            delta += 1
        elif ch in ")]}>" and depth + delta > 0:
            delta -= 1
    return delta


def interface_members(body: str) -> list[str]:
    """The members of an interface body, one string each.

    Neither a comma nor a newline is the separator: prettier is configured without semicolons, so
    these interfaces are written as one member per line with no punctuation between them at all, and
    a reader that split on commas alone read exactly one field per interface and called the rest
    missing. A member therefore ends where the next declaration begins at depth 0 - which is also
    what keeps a type wrapped across lines in one piece.
    """
    members: list[str] = []
    current: list[str] = []
    depth = 0
    for raw in body.splitlines():
        line = strip_ts_comment(raw)
        if not line.strip():
            continue
        starts_member = depth == 0 and re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\??\s*:", line)
        if starts_member and current:
            members.append("\n".join(current).strip())
            current = []
        current.append(line)
        depth += bracket_delta(line, depth)
        if depth == 0 and line.rstrip().endswith((",", ";")):
            members.append("\n".join(current).strip())
            current = []
    if current:
        members.append("\n".join(current).strip())
    return [member for member in members if member]


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """`text` split on `sep` where bracket depth is 0 and no string is open.

    Two things have to be tracked, and both are load-bearing here. Depth, so `Record<string,
    number>` stays whole (and `>` only closes at depth > 0, because `=>` in a function type would
    otherwise unbalance the count and swallow every separator after it). Quotes, because these are
    lists of *prose*: `MODEL_SPEC_HINTS` says "Runs on torch, so a CUDA GPU is the fast path", and
    a reader that split on that comma reported the rest of the sentence as a member it could not
    read.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            current.append(ch)
            if ch == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "'\"`":
            quote = ch
            current.append(ch)
        else:
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>" and depth > 0:
                depth -= 1
            if ch == sep and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        i += 1
    parts.append("".join(current))
    return parts


def ts_export_rhs(source: str, name: str) -> str:
    """The right-hand side of `[export] const NAME = ...`, a literal returned whole.

    Whole (and with the type annotation and any trailing `as const` dropped), so a list written
    across lines reads exactly like a one-line one. Comments are stripped first, `//` inside a
    quoted string excepted: several of these declarations are annotated line by line (`MODEL_SPEC_
    HINTS` above all), and a reader that took the comment text for a value reported a member it
    "cannot read" on a file that is perfectly well formed.

    `export` is optional and leading whitespace is allowed so the same reader can be pointed at a
    literal that is not a top-level declaration - which is how `ts_object_array` reads the members
    of a one-line object it has just lifted out of a list.

    Failing loudly on an absent name is the point: a guard that answered `""` would compare two
    empty sets and pass.
    """
    match = re.search(
        rf"^\s*(?:export\s+)?const {re.escape(name)}\b[^=]*= *(.*)$", source, re.MULTILINE
    )
    if match is None:
        raise AssertionError(f"the desktop no longer exports {name!r}")
    rhs = strip_ts_comment(match.group(1)).strip()
    if rhs[:1] not in "[{":
        return rhs
    body = strip_ts_comments(source[match.start(1) :])
    opener, closer = body[0], {"[": "]", "{": "}"}[body[0]]
    depth = 0
    for i, ch in enumerate(body):
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return body[: i + 1]
    raise AssertionError(f"{name!r} is an unterminated literal")


def ts_scalar(source: str, name: str) -> Any:
    """`export const NAME = <literal>` as a Python value."""
    rhs = ts_export_rhs(source, name)
    if rhs[:1] in "[{":
        raise AssertionError(f"{name!r} is not a scalar")
    return ts_value(rhs)


def ts_array_items(source: str, name: str) -> list[str]:
    """The items of `export const NAME = [...]`, in order and as written.

    As written, because some of these lists hold an identifier (`CUSTOM_MODEL`) rather than a
    literal; evaluating it here would mean re-implementing the file instead of reading it.
    """
    rhs = ts_export_rhs(source, name)
    if not (rhs.startswith("[") and rhs.endswith("]")):
        raise AssertionError(f"{name!r} is not an array literal")
    return [item.strip() for item in split_top_level(rhs[1:-1]) if item.strip()]


def ts_object_array(source: str, name: str) -> list[dict[str, str]]:
    """`export const NAME = [{...}, {...}]` as `[{member: value text}, ...]`, in order.

    The shape `SETTINGS_FIELDS` is written in: one object per entry, one member per line, no
    separators - prettier is configured without semicolons, and a comma after the last member is
    not what these files look like either. Each item is read by `ts_record` on a synthetic
    one-literal source, so the same parsing (and the same loud failures) serve a lone object and
    one nested in a list rather than two nearly-equal implementations drifting apart.
    """
    out: list[dict[str, str]] = []
    for index, item in enumerate(ts_array_items(source, name)):
        text = strip_ts_comments(item).strip()
        if not (text.startswith("{") and text.endswith("}")):
            raise AssertionError(f"an item of {name!r} is not an object literal: {item!r}")
        fake = f"{name}_{index}"
        out.append(ts_record(f"export const {fake} = {text}", fake))
    if not out:
        raise AssertionError(f"{name!r} parsed as an empty list of objects")
    return out


def ts_entries(items: list[str]) -> tuple[set[str], set[str]]:
    """`items` split into string literals (as Python values) and bare identifiers.

    A third kind of entry raises rather than being dropped: an unrecognised item silently left out
    of both sets is how a guard stops covering a line nobody looks at.
    """
    literals: set[str] = set()
    names: set[str] = set()
    for item in items:
        if item[:1] in ("'", '"'):
            literals.add(ts_value(item))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item):
            names.add(item)
        else:
            raise AssertionError(f"an entry this guard cannot read: {item!r}")
    return literals, names


def ts_record(source: str, name: str) -> dict[str, str]:
    """`export const NAME: Record<...> = { ... }` as `{key: value text}`, keys resolved.

    A computed key (`[CUSTOM_MODEL]`) is resolved to the scalar that constant holds, because that is
    the value the object really has - and a guard comparing the *text* `"[CUSTOM_MODEL]"` against a
    model path would never match anything.
    """
    rhs = ts_export_rhs(source, name)
    if not (rhs.startswith("{") and rhs.endswith("}")):
        raise AssertionError(f"{name!r} is not an object literal")
    out: dict[str, str] = {}
    for item in split_top_level(rhs[1:-1]):
        entry = item.strip()
        if not entry:
            continue
        key, sep, value = entry.partition(":")
        if not sep:
            raise AssertionError(f"a member of {name!r} this guard cannot read: {entry!r}")
        key = key.strip()
        if key.startswith("[") and key.endswith("]"):
            key = str(ts_scalar(source, key[1:-1].strip()))
        out[key] = value.strip()
    return out


# --- TypeScript interfaces --------------------------------------------------


def ts_interface(source: str, name: str) -> tuple[dict[str, str], str | None]:
    """`export interface NAME [extends P] { ... }` as `({field: type text}, parent)`.

    Optionality stays in the key (`key?`) and nullability in the type text, so both are read off the
    declaration rather than inferred. The body runs to the line that closes the interface in column
    zero, which is how every interface in these files is written - that is what makes this a reader
    for the declarations here rather than a half-parser for TypeScript.
    """
    opener = chr(0x7B)  # `{`, kept out of the f-string below
    match = re.search(
        rf"^export interface {re.escape(name)}\b([^{opener}]*){opener}", source, re.MULTILINE
    )
    if match is None:
        raise AssertionError(f"the desktop no longer declares interface {name!r}")
    parent_match = re.search(r"extends\s+([A-Za-z_][A-Za-z0-9_]*)", match.group(1))
    parent = parent_match.group(1) if parent_match else None

    rest = source[match.end() :]
    end = re.search(r"^\}", rest, re.MULTILINE)
    if end is None:
        raise AssertionError(f"interface {name!r} is unterminated")
    fields: dict[str, str] = {}
    for entry in interface_members(rest[: end.start()]):
        key, sep, type_text = entry.partition(":")
        if not sep:
            raise AssertionError(f"a member of {name!r} this guard cannot read: {entry!r}")
        fields[key.strip()] = type_text.strip().rstrip(",;").strip()
    if not fields:
        raise AssertionError(f"interface {name!r} parsed as having no members at all")
    return fields, parent


def ts_fields(source: str, name: str) -> dict[str, tuple[bool, bool]]:
    """`{field: (optional, may_be_null)}` for an interface, parents merged in."""
    raw, parent = ts_interface(source, name)
    merged = ts_fields(source, parent) if parent else {}
    for key, type_text in raw.items():
        merged[key.rstrip("?")] = (key.endswith("?"), re.search(r"\bnull\b", type_text) is not None)
    return merged


# --- Pydantic models --------------------------------------------------------


def python_fields(model: type[BaseModel]) -> dict[str, tuple[bool, bool]]:
    """`{field: (has_a_default, may_be_null)}` for a Pydantic model, inherited fields included.

    The first element is not compared - see `assert_same_fields` for why - but it is reported in the
    message when nullability is what drifted, because "may be null" and "has a default" are the two
    facts a reader of the failure needs to tell apart.
    """
    return {
        name: (not field.is_required(), type(None) in get_args(field.annotation))
        for name, field in model.model_fields.items()
    }


def assert_same_fields(
    label: str,
    python_side: dict[str, tuple[bool, bool]],
    ts_side: dict[str, tuple[bool, bool]],
) -> None:
    """Both directions of the field *names*, and both directions of "may be null".

    Which of the two sides moved, and how, is the whole diagnosis - so the failure says that rather
    than printing two dicts.

    Optionality (`?` on the TypeScript side) is deliberately *not* compared. In this codebase it
    means "a sidecar that predates this field sends no such key" - the renderer's tolerance for an
    older process, which is a fact about history rather than about the contract both sides hold
    now, and the sidecar serializes every field it declares. `LogEvent.left_at` is the pair that
    makes the distinction concrete: it is `number | null` and *required* in TypeScript because it
    has been on the wire from the start and carries `null` while a track is open, whereas
    `Stats.suppressed` is optional because it is newer than the wire. Reading "has a default" as
    "may be omitted" would fail on the first and pass the second for the wrong reason - so names and
    nullability are compared, and this is written down rather than left as a subtlety.
    """
    problems: list[str] = []
    for name in sorted(set(python_side) - set(ts_side)):
        problems.append(f"{name} is declared by the sidecar and missing from the desktop")
    for name in sorted(set(ts_side) - set(python_side)):
        problems.append(f"{name} is declared by the desktop and missing from the sidecar")
    for name in sorted(set(python_side) & set(ts_side)):
        py_default, py_null = python_side[name]
        ts_optional, ts_null = ts_side[name]
        if py_null != ts_null:
            problems.append(
                f"{name}: the sidecar {'allows null' if py_null else 'never sends null'} "
                f"({'has a default' if py_default else 'required'}), the desktop "
                f"{'allows null' if ts_null else 'does not'} "
                f"({'marked optional' if ts_optional else 'required'})"
            )
    assert not problems, f"{label} has drifted:\n  " + "\n  ".join(problems)
