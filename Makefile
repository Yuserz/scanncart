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

SIDECAR_DIR := sidecar
DESKTOP_DIR := desktop

ifeq ($(OS),Windows_NT)
  SIDECAR_VENV_PY := .venv/Scripts/python.exe
else
  SIDECAR_VENV_PY := .venv/bin/python
endif

.DEFAULT_GOAL := help

.PHONY: help install dev test build lint format typecheck clean \
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
