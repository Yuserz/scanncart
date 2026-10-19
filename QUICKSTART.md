# SCANnCART quickstart

The short version — a fresh clone to a running app. For the full guide (manual
equivalents of every command, GPU/Roboflow extras, and the environment quirks
that bite on a fresh machine) read
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## New Windows device (the short path)

1. Get the code: `git clone <repo-url> C:\scanncart` — or download the ZIP from GitHub and
   unzip it. Keep it **outside OneDrive** (a synced folder breaks `node_modules` and the venv).
2. Double-click **`setup.cmd`**. It installs everything missing (Node, uv + Python 3.12, the
   VC++ runtime, Git, Make, CUDA torch when an NVIDIA driver is present), the app's own
   dependencies and the model weights, then puts a **SCANnCART** shortcut on the desktop.
   Safe to re-run; `setup.cmd -Check` only reports what is missing.
3. Plug the Logitech StreamCam into a **USB 3.0** port (blue, or marked SS) and close NVIDIA
   Broadcast / Logi apps that might hold the camera.
4. Double-click **SCANnCART** on the desktop (or `start.cmd`), then press **Start**.

What a fresh install defaults to: capture **1920x1080 @ 60 fps** (the StreamCam's best mode)
and self-checkout cart mode **Basket**. If the camera never delivers frames on a machine with
only USB 2.0, set 1280x720 or 640x480 @ 30 in **Admin → Camera & Capture** and replug the camera. If
Windows Defender kills the sidecar on start, add the repo folder as a Defender exclusion.

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
