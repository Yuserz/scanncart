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
## `test` is CI's target and stays fake-only, so it needs no data. `verify-clamp`
## is the one target that does - see the note above it.
##

SIDECAR_DIR := sidecar
DESKTOP_DIR := desktop

ifeq ($(OS),Windows_NT)
  SIDECAR_VENV_PY := .venv/Scripts/python.exe
else
  SIDECAR_VENV_PY := .venv/bin/python
endif

.DEFAULT_GOAL := help

.PHONY: help install dev test build lint format typecheck clean \
        verify-clamp \
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
	@echo "  verify-clamp         re-check the frame-clamp claims (local data, not CI)"
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
