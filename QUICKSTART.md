# SCANnCART quickstart

The short version — a fresh clone to a running app. For the full guide (manual
equivalents of every command, GPU/Roboflow extras, and the environment quirks
that bite on a fresh machine) read
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## You need

- **Node.js 20+** and npm — for the Electron/React desktop app.
- **Python exactly 3.12** — for the sidecar. Newer interpreters have no
  installable `torch`/`numpy` wheels. `uv venv --python 3.12` fetches one for you.
- **Visual C++ runtime** (Windows only) — PyTorch's DLLs need it.
- **GNU Make** — optional. On Windows it works from any terminal once Git is
  installed (the Makefile runs its recipes in Git's `sh.exe`); install it with
  `winget install ezwinports.make`. Every `make` target has a manual equivalent in
  the full guide.
- Optional extras the setup does not install: an NVIDIA driver (with one installed, the
  setup puts in the GPU builds of torch on its own) and Docker (only for the `local_api`
  detector backend).
- UV — optional but recommended.

Don't have them? One script installs whatever is missing, then the project's
own dependencies (safe to re-run; `--check` / `-Check` only reports):

```bash
bash scripts/setup.sh                                          # Linux/macOS
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1     # Windows (or double-click setup.cmd)
```

On Windows the whole thing is two double-clicks: `setup.cmd`, then `start.cmd`
to launch. Run `setup.cmd` **before** any `make` command (it is what installs
Make), open a new terminal after it, and keep the folder outside OneDrive. The model weights are committed, so the
clone has them; `-ModelsFrom <folder>` on `setup.cmd` copies in any that are not committed yet.

## Run it

```bash
git clone <repo-url> scanncart && cd scanncart
make install    # desktop npm ci + sidecar venv + model weights
make test       # confirm both toolchains work (no camera needed)
make dev        # launch the app — it spawns the sidecar for you
```

`make test` runs entirely against fakes, so it needs no camera, GPU, or API
key — it is the fastest way to know your machine is set up correctly. `make dev`
starts electron-vite; the Electron main process launches the sidecar and you
should land on the **Live** view. Click **Start** to begin capture.

Two things worth expecting on first launch:

- The default grocery model (`sidecar/models/scanncart-grocery-v2.1.pt` + its `.json`) is
  committed, along with the other installed grocery weights, so a clone or `git pull` has it.
  A stock model such as `yolo11n.pt` (**Admin → Model**) downloads on the first capture start.
- A Logitech StreamCam can take ~37 s to open and set its capture mode. That is
  the device, not a hang — `/api/health` keeps answering throughout.

## Without `make`

```bash
# sidecar
cd sidecar
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.lock.txt   # Windows
uv pip install --python .venv/bin/python -r requirements.lock.txt           # Linux/macOS

# desktop
cd ../desktop && npm install
```

## Where to go next

- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — the step-by-step guide.
- [`sidecar/README.md`](sidecar/README.md) / [`desktop/README.md`](desktop/README.md)
  — per-package detail.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit together.
- [`CLAUDE.md`](CLAUDE.md) — module-by-module map and full command reference.
