# SCANnCART quickstart

The short version — a fresh clone to a running app. For the full guide (manual
equivalents of every command, GPU/Roboflow extras, and the environment quirks
that bite on a fresh machine) read
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## You need

- **Node.js 20+** and npm — for the Electron/React desktop app.
- **Python exactly 3.12** — for the sidecar. Newer interpreters have no
  installable `torch`/`numpy` wheels. `uv venv --python 3.12` fetches one for you.
- **GNU Make** — optional. On Windows use Git Bash/WSL, or
  `winget install GnuWin32.Make`; every `make` target has a manual equivalent in
  the full guide.
- UV — optional but recommended.

## Run it

```bash
git clone <repo-url> scanncart && cd scanncart
make install    # desktop npm install + sidecar venv
make test       # confirm both toolchains work (no camera needed)
make dev        # launch the app — it spawns the sidecar for you
```

`make test` runs entirely against fakes, so it needs no camera, GPU, or API
key — it is the fastest way to know your machine is set up correctly. `make dev`
starts electron-vite; the Electron main process launches the sidecar and you
should land on the **Live** view. Click **Start** to begin capture.

Two things worth expecting on first launch:

- The default grocery model (`sidecar/models/scanncart-grocery-v1.pt` + its `.json`) is not
  in git: copy it in, or pick a stock model such as `yolo11n.pt` in **Admin → Model**,
  which downloads on the first capture start.
- A Logitech StreamCam can take ~37 s to open and set its capture mode. That is
  the device, not a hang — `/api/health` keeps answering throughout.

## Without `make`

```bash
# sidecar
cd sidecar
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows
uv pip install --python .venv/bin/python -r requirements.txt           # Linux/macOS

# desktop
cd ../desktop && npm install
```

## Where to go next

- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — the step-by-step guide.
- [`sidecar/README.md`](sidecar/README.md) / [`desktop/README.md`](desktop/README.md)
  — per-package detail.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit together.
- [`CLAUDE.md`](CLAUDE.md) — module-by-module map and full command reference.
