"""PostToolUse hook: flag when a change likely invalidates the architecture docs.

Reads the hook payload on stdin. If the edited file belongs to a layer the
architecture docs describe, emits additionalContext naming the specific
sections to re-check. Silent for everything else.

Docs kept in sync:  docs/ARCHITECTURE.md
                    docs/architecture.drawio          (Layout C — swimlanes)
                    docs/architecture-layout-a.drawio (Layout A — pipeline)
"""

import json
import posixpath
import sys

# (path prefix, what it affects) — first match wins, so order matters.
RULES: list[tuple[str, str]] = [
    ("sidecar/app/pipeline.py",
     "S3 processing chain + S3.1 AI model (thread hand-off, frame skip, track lifecycle)"),
    ("sidecar/app/inference.py",
     "S3.1 AI model — detector/tracker choice is stated there (YOLO11 + BoT-SORT)"),
    ("sidecar/app/camera.py",
     "S3 processing chain — capture thread and the size-1 latest-frame buffer"),
    ("sidecar/app/logging_store.py",
     "S6 data model — sessions / detection_events schema is mirrored in the docs"),
    ("sidecar/app/schemas.py",
     "S5 interfaces — the REST/WS contract surface"),
    ("sidecar/app/main.py",
     "S5 interfaces — route list; S4 runtime flow if start/stop changed"),
    ("sidecar/app/settings",
     "S7 configuration — hot-reloadable vs restart-required split"),
    ("sidecar/app/presets.py",
     "S3.1 AI model — the model tier table (n/s/m/l/x by preset)"),
    ("sidecar/app/hardware.py",
     "S7 configuration — hardware probing feeds preset recommendation"),
    ("sidecar/run.py",
     "S2 containers + startup handshake strip (port discovery is drawn in both .drawio files)"),
    ("desktop/src/main/sidecar.ts",
     "S2 containers + S7 lifecycle — spawn/port handshake"),
    ("desktop/src/main/index.ts",
     "S7 lifecycle — app startup/quit ordering"),
    ("desktop/src/preload/",
     "S2 containers — the preload bridge is documented as the ONLY main-renderer channel"),
    ("desktop/src/renderer/src/lib/",
     "S5 interfaces — REST/WS client contract"),
    ("desktop/src/renderer/src/hooks/",
     "S2 containers + S4 runtime flow — state wiring and item-log dedupe"),
    ("desktop/src/renderer/src/views/",
     "S2 containers — Live View / Admin Panel responsibilities"),
    # tooling / dependency changes
    ("sidecar/requirements.txt",
     "S8 design decisions — a dependency change may change the stated stack"),
    ("desktop/package.json",
     "S8 design decisions — a dependency change may change the stated stack"),
    ("Makefile",
     "S8 design decisions / CLAUDE.md commands"),
]

DOCS = (
    "docs/ARCHITECTURE.md",
    "docs/architecture.drawio",
    "docs/architecture-layout-a.drawio",
)


def relevant(rel: str) -> str | None:
    for prefix, affects in RULES:
        if rel.startswith(prefix):
            return affects
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not path:
        return 0

    rel = path.replace("\\", "/")
    marker = "/scanncart/"
    idx = rel.lower().rfind(marker)
    if idx != -1:
        rel = rel[idx + len(marker):]
    rel = posixpath.normpath(rel).lstrip("./")

    # Editing the docs themselves is the fix, not the trigger.
    if rel in DOCS:
        return 0

    affects = relevant(rel)
    if affects is None:
        return 0

    msg = (
        f"Architecture-doc check: you changed `{rel}`, which the architecture docs "
        f"describe. Re-read the relevant part of docs/ARCHITECTURE.md and update it if "
        f"this change altered the described behavior.\n"
        f"Likely affected: {affects}\n"
        f"If the change is drawn in the diagrams (a box, an arrow, or a label), also update "
        f"docs/architecture.drawio and docs/architecture-layout-a.drawio. "
        f"If nothing architectural changed, do nothing and do not mention this."
    )

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": msg,
            },
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
