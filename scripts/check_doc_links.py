#!/usr/bin/env python3
"""Fail when a Markdown file links to a relative path that does not exist.

The repo's docs cross-reference each other by relative path
(``docs/DEVELOPMENT.md``, ``../sidecar/README.md``, ...). Renaming or moving a
file silently leaves those pointers dangling, and nothing else in the toolchain
notices: the docs are not imported, so a broken link is only found by a reader
who clicks it. This walks every tracked Markdown file and checks each relative
target resolves.

Only internal, relative links are checked. Absolute URLs (``http://``,
``https://``, ``mailto:``, ``tel:``) and pure anchors (``#section``) are skipped
on purpose - the point is to catch broken cross-references, not to make the
build depend on the network.

Usage::

    python scripts/check_doc_links.py            # check the repo
    python scripts/check_doc_links.py a.md b.md  # check specific files

Exit code is 0 when every link resolves, 1 with a report otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that hold generated or vendored content, never hand-written docs.
SKIP_DIRS = {
    ".git",
    "node_modules",
    "out",
    "dist",
    ".venv",
    ".venv-inference",
    "graphify-out",
    "__pycache__",
}

# Whole subtrees that are generated data, not docs: the dataset workspace holds
# tool output (split plans, manifests) that the tools write, so a stale link in
# one of its files is not this repo's to fix.
SKIP_PREFIXES = (("sidecar", "data"),)

# [text](target) and ![alt](target), plus reference definitions: [label]: target
INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REF_DEF_RE = re.compile(r"^\[[^\]]+\]:\s*(\S+)", re.MULTILINE)

# Schemes and prefixes that are not repo-relative paths.
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "ftp://")


def iter_markdown_files(paths: list[str]) -> list[Path]:
    """Return the Markdown files to scan: explicit paths, or the whole repo."""
    if paths:
        return [Path(p).resolve() for p in paths]
    found: list[Path] = []
    for path in sorted(REPO_ROOT.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel_parts = path.relative_to(REPO_ROOT).parts
        if any(rel_parts[: len(prefix)] == prefix for prefix in SKIP_PREFIXES):
            continue
        found.append(path)
    return found


def targets_in(text: str) -> list[str]:
    """Every link target in a Markdown document, in source order."""
    targets = INLINE_LINK_RE.findall(text)
    targets += REF_DEF_RE.findall(text)
    return targets


def is_internal(target: str) -> bool:
    """True when a link target is a repo-relative path worth checking."""
    if not target or target.startswith("#"):
        return False
    if target.startswith(EXTERNAL_PREFIXES):
        return False
    if target.startswith("/") or target.startswith("~"):
        return False  # site-absolute, not repo-relative
    return True


def check_file(path: Path) -> list[str]:
    """Return one message per broken link in ``path`` (empty when all good)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover - defensive
        return [f"{path}: could not read ({exc})"]

    problems: list[str] = []
    for target in targets_in(text):
        if not is_internal(target):
            continue
        # Strip any #anchor and ?query before resolving the filesystem path.
        rel = target.split("#", 1)[0].split("?", 1)[0]
        if not rel:
            continue  # anchor-only after the fragment was stripped
        candidate = (path.parent / rel).resolve()
        if not candidate.exists():
            problems.append(f"{path.relative_to(REPO_ROOT)}: broken link -> {target}")
    return problems


def main(argv: list[str]) -> int:
    files = iter_markdown_files(argv)
    if not files:
        print("check_doc_links: no Markdown files found", file=sys.stderr)
        return 1

    problems: list[str] = []
    for path in files:
        problems.extend(check_file(path))

    if problems:
        print("Broken internal documentation links:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            f"\n{len(problems)} broken link(s) across {len(files)} file(s).",
            file=sys.stderr,
        )
        return 1

    print(f"check_doc_links: {len(files)} Markdown file(s), all internal links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
