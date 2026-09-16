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

# Model deployment / FP-gate knobs (override on the command line, e.g.
#   make deploy-model MODEL=path/to/best.pt MODEL_SLOT=grocery-v2)
# The gate's default policy (any detection = violation) assumes a custom
# grocery-only model: it has no 'person' class, so a person hit on the
# face-at-counter frames IS the false positive we're hunting. Stock COCO
# models (yolo11n.pt etc.) legitimately detect the person in these frames and
# will always fail the gate — narrow them with FP_CLASSES=safeguard, etc.
MODEL ?= scanncart/yolo11-grocery/weights/best.pt
MODEL_SLOT ?= grocery-v1
FP_FRAMES ?= $(SIDECAR_DIR)/data/datasets/hard_negatives/images
# Scan ~2x below the 0.5 operating threshold: deep enough to catch weak
# regressions early, high enough to stay above the sub-0.2 noise floor.
FP_CONF ?= 0.25
FP_CLASSES ?=
NEG_SOURCE ?= 0
NEG_FRAMES ?= 60
NEG_INTERVAL ?= 2.0

ifeq ($(OS),Windows_NT)
  SIDECAR_VENV_PY := .venv/Scripts/python.exe
else
  SIDECAR_VENV_PY := .venv/bin/python
endif

.DEFAULT_GOAL := help

.PHONY: help install dev test build lint format typecheck clean \
        sidecar-setup sidecar-run sidecar-test \
        capture-negatives fp-gate deploy-model \
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
	@echo "  capture-negatives    save background frames from the camera (NEG_SOURCE, NEG_FRAMES, NEG_INTERVAL)"
	@echo "  fp-gate              run the FP regression gate on a model (MODEL, FP_FRAMES, FP_CONF, FP_CLASSES)"
	@echo "  deploy-model         FP-gate a model, then deploy it to a custom slot (MODEL, MODEL_SLOT)"
	@echo "                       bypass the gate once with SKIP_FP_GATE=1"
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

## --- model deployment (FP gate + custom slot) ---

capture-negatives:
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) tools/capture_hard_negatives.py \
		--source $(NEG_SOURCE) --max-frames $(NEG_FRAMES) --interval $(NEG_INTERVAL)

fp-gate:
ifeq ($(SKIP_FP_GATE),1)
	@echo "SKIP_FP_GATE=1 -- deploying WITHOUT the false-positive gate"
else
	@test -f "$(MODEL)" || { \
		echo "error: model not found: $(MODEL)"; \
		echo "  point MODEL at the candidate .pt, e.g. make fp-gate MODEL=scanncart/yolo11-grocery/weights/best.pt"; \
		exit 1; \
	}
	@[ -d "$(FP_FRAMES)" ] || { \
		echo "error: FP frames dir not found: $(FP_FRAMES)"; \
		echo "  capture some first: make capture-negatives"; \
		echo "  (a gate over zero frames proves nothing; bypass once with SKIP_FP_GATE=1 make deploy-model)"; \
		exit 1; \
	}
	@echo "== FP gate: $(MODEL) vs $(FP_FRAMES) (conf >= $(FP_CONF)) =="
	cd $(SIDECAR_DIR) && $(SIDECAR_VENV_PY) tools/eval_fp_gate.py \
		--model $(abspath $(MODEL)) --frames $(abspath $(FP_FRAMES)) --conf $(FP_CONF) \
		$(if $(FP_CLASSES),--classes $(FP_CLASSES))
	@echo "== FP gate passed =="
endif

deploy-model: fp-gate
	$(SIDECAR_DIR)/$(SIDECAR_VENV_PY) scanncart/deploy_model.py \
		--source $(MODEL) --name $(MODEL_SLOT)
	@echo ""
	@echo "Next: select 'data/custom/$(MODEL_SLOT).pt' in the Admin Panel, then start capture."

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
