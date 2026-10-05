---
name: codebase-notes
description: Load SCANnCART's detailed module notes for one area on demand — the sidecar runtime, the dataset tools, the desktop app, or the guard/mirror tests. Use when about to change code in an area and the overview in the root CLAUDE.md is not enough, when asked why a module works the way it does, or when the user types /codebase-notes with an area name (app, tools, desktop, tests, or all).
---

# SCANnCART codebase notes

The root `CLAUDE.md` is a short overview. The module-by-module notes (what each module owns, its
invariants, the measurements behind its defaults, and why the guards exist) live beside the code in
four files. Claude Code loads one automatically the first time work touches its folder; this skill
loads one by name, before any file there has been opened.

## Which file to read

| Area (argument) | File | Read it when |
| --- | --- | --- |
| `app` | `sidecar/app/CLAUDE.md` | Touching the Python runtime: capture lifecycle, camera controls and auto exposure, detection filters, pipeline, settings and hot reload, models/roster, the event loop |
| `tools` | `sidecar/tools/CLAUDE.md` | Touching the dataset tools: clean/plan/build/doctor/train/accept, class-order rules, the workspace |
| `desktop` | `desktop/CLAUDE.md` | Touching the Electron app: sidecar supervision, the self-checkout loop and deposit/removal rules, the views and hooks |
| `tests` | `sidecar/tests/CLAUDE.md` | Touching a guard, a mirrored contract, a wording contract, the Makefile, or a doc's shell commands; or a `mirror`/`docs` test failed |

## How to use it

1. Map the argument (or, with none, the files the task is about) to a row above. `all` means every
   row.
2. Read that file in full with the Read tool. They are long single-line paragraphs, one per module;
   use Grep on the module's name first when only one module matters.
3. Treat what it says as the current design, and keep it true: a change to a module's behaviour
   updates its paragraph in the same edit. Several paragraphs are pinned by tests
   (`test_desktop_contracts.py` wording contracts, `test_cost_figures.py`, `test_docs_claims.py`),
   so run `make test` (or the sidecar `-m "mirror or docs"` subset) after editing one.
