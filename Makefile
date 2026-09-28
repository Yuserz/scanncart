##
## SCANnCART — root Makefile
##
## Wraps the desktop (Electron/npm) and sidecar (Python/uv) toolchains.
## Run `make` or `make help` to list targets.
##
## Windows note: this Makefile assumes a POSIX shell (Git Bash / WSL / MSYS).
## Install GNU Make via `winget install GnuWin32.Make`, or run these targets
## from within WSL or a Git Bash shell that has `make` on PATH.
##
## `test` is CI's target and stays fake-only, so it needs no data. `verify-clamp`,
## `annotate`, `human-pass`, `doctor` and `accept-v2` are the ones that do - see the
## notes above them.
##

SIDECAR_DIR := sidecar
DESKTOP_DIR := desktop

ifeq ($(OS),Windows_NT)
  SIDECAR_VENV_PY := .venv/Scripts/python.exe
else
  SIDECAR_VENV_PY := .venv/bin/python
endif

.DEFAULT_GOAL := help

.PHONY: help install dev test docs-check build lint format typecheck clean \
        verify-clamp doctor annotate human-pass accept-v2 \
        sidecar-setup sidecar-run sidecar-test \
        desktop-install desktop-dev desktop-start desktop-test desktop-test-watch \
        desktop-build desktop-build-win desktop-build-mac desktop-build-linux \
        desktop-lint desktop-format desktop-typecheck

help:
	@echo "SCANnCART - available targets"
	@echo ""
	@echo "  install              install desktop deps + set up sidecar venv"
	@echo "  dev                  run the desktop app in dev mode (spawns the sidecar)"
	@echo "  test                 run desktop + sidecar test suites"
	@echo "  docs-check           check that internal documentation links resolve"
	@echo "  verify-clamp         re-check the frame-clamp claims (local data, not CI)"
	@echo "  annotate             label a staged session locally in the browser (local data, not CI)"
	@echo "  human-pass           render the human pass's checklist from the annotator's store (local data, not CI)"
	@echo "                       HUMAN_PASS_ARGS=--check verifies it, =--status exits nonzero while the gate is dirty"
	@echo "  doctor               check the merged set before training it (local data, not CI)"
	@echo "  accept-v2            measure v2 against v1 on the merged set's test split (local data, not CI)"
	@echo "  build                typecheck + build the desktop app"
	@echo "  lint                 lint the desktop app"
	@echo "  format               format the desktop app"
	@echo "  typecheck            typecheck the desktop app (node + web)"
	@echo "  clean                remove desktop node_modules/out/dist and sidecar .venv"
	@echo ""
	@echo "  sidecar-setup        create sidecar/.venv and install requirements (uses uv if available)"
	@echo "  sidecar-run          run the sidecar standalone (prints SIDECAR_PORT=<n>)"
	@echo "  sidecar-test         run the sidecar pytest suite"
	@echo ""
	@echo "  desktop-install      npm install in desktop/"
	@echo "  desktop-dev          electron-vite dev (full app; needs sidecar-setup first)"
	@echo "  desktop-start        electron-vite preview"
	@echo "  desktop-test         vitest run (headless, fakes only)"
	@echo "  desktop-test-watch   vitest watch mode"
	@echo "  desktop-build        typecheck + electron-vite build"
	@echo "  desktop-build-win    build + package for Windows"
	@echo "  desktop-build-mac    build + package for macOS"
	@echo "  desktop-build-linux  build + package for Linux"
	@echo "  desktop-lint         eslint --cache"
	@echo "  desktop-format       prettier --write"
	@echo "  desktop-typecheck    tsc --noEmit (node + web)"

## --- aggregate targets ---

install: desktop-install sidecar-setup

dev: desktop-dev

test: desktop-test sidecar-test

## --- documentation ---

# Every relative link between the tracked Markdown files must resolve, so a
# rename or move cannot quietly leave a cross-reference dangling. Standard
# library only, so it runs before anything is installed - but it still needs an
# interpreter, and on Windows `python` is often absent, so the sidecar venv's is
# used there (path is root-relative here, unlike SIDECAR_VENV_PY). CI runs the
# script with its own python3 in the `docs-links` job. Not part of `test`.
ifeq ($(OS),Windows_NT)
  DOCS_PYTHON ?= $(SIDECAR_DIR)/.venv/Scripts/python.exe
else
  DOCS_PYTHON ?= python3
endif

docs-check:
	$(DOCS_PYTHON) scripts/check_doc_links.py

## --- verification gates that need data the repo does not carry ---

# The frame-clamp claims in docs/CAPTURE_CHECKLIST.md are measured against v1's installed weights
# and the staged Tier C2 negatives. Both inputs are gitignored on purpose - weights are build
# outputs (`test_no_weight_files_are_tracked` keeps them so) and the dataset workspace is 6.6 GB -
# so CI cannot run this: `ubuntu-latest` checks out the repo and has neither. The tool therefore
# fails loudly rather than skipping when they are missing, because a skip here is a silent pass on
# the one check that says the suppression still works on real frames and still costs nothing.
#
# `--conf` is pinned instead of taking the value from data/settings.json, so the gate is
# reproducible and means the same thing on every machine: the claims are about the rule and these
# weights, and 0.5 is the operating point the published table was measured at. Point it at another
# generation or threshold with `make verify-clamp CLAMP_GEN=v2 CLAMP_CONF=0.7` - at a high enough
# threshold the empty counter produces nothing, the rule catches nothing, and it correctly fails.
CLAMP_GEN ?= v1
CLAMP_CONF ?= 0.5

verify-clamp:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) tools/clamp_probe.py \
		--generation $(CLAMP_GEN) --conf $(CLAMP_CONF) --strict

# The local labeler (`sidecar/annotate/`): a browser app over a staged set, writing into the
# workspace's `annotations-v2/`. Its own `--out` default is `workspace.DEFAULT_OUT` (so
# `SCANNCART_DATASET_ROOT` moves it with everything else), which is why this passes no path by
# default - override the whole command line with `make annotate ANNOTATE_ARGS="--out … --hosted …"`.
ANNOTATE_ARGS ?=

annotate:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) -m annotate.run $(ANNOTATE_ARGS)

# The human pass's checklist (`<workspace>/_human_pass.md`), *rendered* from the store rather than
# kept by hand. Every number in it and every name in its three frame lists is read back off the
# annotator's own selectors, so a copy that has stopped matching the set is detectable instead of
# merely wrong: run this before sitting down to work the pass, and again after a session.
# `make human-pass HUMAN_PASS_ARGS=--check` writes nothing and exits 1 with the diff when the file on
# disk is no longer what the store says - the form anything scriptable wants.
# `make human-pass HUMAN_PASS_ARGS=--status` is its sibling: the three sections' counts, and a
# nonzero exit while `test`/`valid` still hold a machine-only decision - the acceptance gate asked as
# a question. The tool exits 1; make reports any failed recipe as 2, so test for nonzero.
HUMAN_PASS_ARGS ?=

human-pass:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) -m annotate.human_pass $(HUMAN_PASS_ARGS)

# Run before training, not after: the failures it looks for - a class list in the wrong order, a
# label row nothing can read, a frame drawn under a class other than the one it was staged as, a test
# frame that duplicates a train one, a merge report left over from an earlier build - all produce a
# dataset that trains happily and measures the wrong thing. It also prints the v2 distance mix, and
# warns when a distance the model would train on has no test frame (nothing can measure it) or when a
# distance the plan follows is in no split at all (the capture gap). Also the
# same local data `accept-v2` needs (the merged set), so it fails loudly when the set is absent rather
# than reporting on nothing. Point it at another set or generation with
# `make doctor DOCTOR_DATASET=... DOCTOR_GEN=v1`.
#
# This target is the reading, not the enforcement: `train_model.py --yes/--val` and `accept_v2.py`
# run the same check inline and refuse a set that fails it, so a number cannot be produced over a set
# nobody doctored. Run it by hand to see the whole report before spending GPU time.
#
# The generation is `auto`: the doctor reads it off the set (the merge report, then the order its own
# `data.yaml` declares) rather than taking this target's word for it, so pointing the dataset at v1's
# export without also naming v1 cannot fail the set for being in v1's order. Override with
# `DOCTOR_GEN=v1` only to ask a deliberate what-if.
DOCTOR_DATASET ?= data/datasets/merged-v2
DOCTOR_GEN ?= auto

doctor:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) tools/dataset_doctor.py \
		--dataset $(DOCTOR_DATASET) --generation $(DOCTOR_GEN)

# The acceptance gate for v2: both weights measured on the merged set's own test split, per-class
# against 6's floor, the crowded counter compared, and `machine_only` required to be zero in
# valid/test. Needs the same local data CI does not carry (two installed weights and the merged
# set), so it fails loudly rather than skipping - and it is a *verdict*: exit 1 with the reasons.
# `ACCEPT_BASELINE`/`ACCEPT_CANDIDATE` point it at other generations.
ACCEPT_BASELINE ?= models/scanncart-grocery-v1.pt
ACCEPT_CANDIDATE ?= models/scanncart-grocery-v2.pt
ACCEPT_SPLIT ?= test

accept-v2:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) tools/accept_v2.py \
		--baseline $(ACCEPT_BASELINE) --candidate $(ACCEPT_CANDIDATE) \
		--split $(ACCEPT_SPLIT) --iou-sweep

build: desktop-build

lint: desktop-lint

format: desktop-format

typecheck: desktop-typecheck

clean:
	rm -rf $(DESKTOP_DIR)/node_modules $(DESKTOP_DIR)/out $(DESKTOP_DIR)/dist
	rm -rf $(SIDECAR_DIR)/.venv

## --- sidecar (Python / FastAPI / YOLO11) ---

sidecar-setup:
	cd $(SIDECAR_DIR) && \
	if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.12 .venv && \
		uv pip install --python $(SIDECAR_VENV_PY) -r requirements.txt; \
	else \
		python -m venv .venv && \
		$(SIDECAR_VENV_PY) -m pip install -r requirements.txt; \
	fi

sidecar-run:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) run.py

sidecar-test:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) -m pytest -v

## --- desktop (Electron / React / TypeScript) ---

desktop-install:
	cd $(DESKTOP_DIR) && npm install

desktop-dev:
	cd $(DESKTOP_DIR) && npm run dev

desktop-start:
	cd $(DESKTOP_DIR) && npm run start

desktop-test:
	cd $(DESKTOP_DIR) && npm test

desktop-test-watch:
	cd $(DESKTOP_DIR) && npm run test:watch

desktop-build:
	cd $(DESKTOP_DIR) && npm run build

desktop-build-win:
	cd $(DESKTOP_DIR) && npm run build:win

desktop-build-mac:
	cd $(DESKTOP_DIR) && npm run build:mac

desktop-build-linux:
	cd $(DESKTOP_DIR) && npm run build:linux

desktop-lint:
	cd $(DESKTOP_DIR) && npm run lint

desktop-format:
	cd $(DESKTOP_DIR) && npm run format

desktop-typecheck:
	cd $(DESKTOP_DIR) && npm run typecheck
