#!/usr/bin/env bash
#
# SCANnCART one-command setup for Linux and macOS.
#
# Installs the system tools this repo needs (the list below), then installs both toolchains - the
# desktop's npm packages and the sidecar's Python 3.12 venv - exactly as `make install` would.
# Re-running it is safe: a tool that is already there at a good enough version is left alone.
#
#   bash scripts/setup.sh              # tools + project install
#   bash scripts/setup.sh --tools-only # system tools only
#   bash scripts/setup.sh --check      # report what is missing, change nothing
#   bash scripts/setup.sh --no-models  # skip downloading the model weights
#   bash scripts/setup.sh --test       # ... and run `make test` at the end
#
# Windows: use scripts/setup.ps1 instead (or setup.cmd at the repo root).
#
# Every tool this repo uses, and why (README.md "Prerequisites" is the human version of this list;
# scripts/setup.ps1 carries the same list - change both together).
#
# Installed by this script:
#   node+npm 20+           required - the Electron/React desktop app (CI uses 20)
#   uv       any           required - creates the sidecar venv and fetches Python 3.12, no sudo
#   python   exactly 3.12  required - the sidecar; installed *by uv*, the system python is untouched
#   Python packages        required - requirements.lock.txt (exact versions) into sidecar/.venv
#   npm packages           required - package-lock.json into desktop/node_modules (Electron, React)
#   git      any           recommended - clone/pull the repo
#   make     any           optional - the root Makefile that wraps both toolchains
#
# Not installed by this script (optional, see the docs):
#   CUDA torch + onnxruntime-gpu   NVIDIA acceleration - docs/DEVELOPMENT.md section 6
#   Docker                         only for the local_api detector backend (docs/DETECTOR_BACKENDS.md)
#   a camera                       live capture only; the test suites need none

set -euo pipefail

NODE_MAJOR_MIN=20
NODE_MAJOR_INSTALL=22   # the LTS fetched when the distro's node is missing or too old
PYTHON_VERSION=3.12

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE=all
RUN_TESTS=0
FETCH_MODELS=1
# The app's default model (sidecar/app/settings.py); its .json record carries the stretch geometry.
DEFAULT_MODEL=scanncart-grocery-v2.1
for arg in "$@"; do
  case "$arg" in
    --tools-only) MODE=tools ;;
    --check) MODE=check ;;
    --test) RUN_TESTS=1 ;;
    --no-models) FETCH_MODELS=0 ;;
    -h|--help) awk 'NR>2 && !/^#/ {exit} NR>2 {sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m      %s\n' "$*"; }
miss() { printf '  \033[33mmissing\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# uv's installer drops it in ~/.local/bin (or ~/.cargo/bin on older ones); make sure this shell
# sees it straight away instead of only after a new login.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  have sudo && SUDO="sudo"
fi

OS="$(uname -s)"
PM=""
case "$OS" in
  Darwin) PM=brew ;;
  Linux)
    for candidate in apt-get dnf yum pacman zypper apk; do
      if have "$candidate"; then PM="$candidate"; break; fi
    done
    ;;
  *) die "unsupported OS '$OS' - on Windows run scripts/setup.ps1" ;;
esac

pm_install() {
  case "$PM" in
    brew)    brew install "$@" ;;
    apt-get) $SUDO apt-get update -y && $SUDO apt-get install -y "$@" ;;
    dnf)     $SUDO dnf install -y "$@" ;;
    yum)     $SUDO yum install -y "$@" ;;
    pacman)  $SUDO pacman -Sy --noconfirm --needed "$@" ;;
    zypper)  $SUDO zypper --non-interactive install "$@" ;;
    apk)     $SUDO apk add --no-cache "$@" ;;
    *)       die "no supported package manager found; install $* by hand" ;;
  esac
}

node_major() { node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0; }

# --- what is there -------------------------------------------------------------------------------

say "Checking system tools"
need_git=0; need_node=0; need_uv=0; need_make=0
if have git;  then ok "git $(git --version | awk '{print $3}')"; else miss git; need_git=1; fi
if have node && [ "$(node_major)" -ge "$NODE_MAJOR_MIN" ] && have npm; then
  ok "node $(node --version), npm $(npm --version)"
else
  if have node; then miss "node $NODE_MAJOR_MIN+ (found $(node --version))"; else miss "node $NODE_MAJOR_MIN+"; fi
  need_node=1
fi
if have uv;   then ok "$(uv --version)"; else miss uv; need_uv=1; fi
if have make; then ok "$(make --version | head -n1)"; else miss make; need_make=1; fi

if [ "$MODE" = check ]; then
  if have uv && uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
    ok "python $PYTHON_VERSION (via uv)"
  else
    miss "python $PYTHON_VERSION (uv fetches it during setup)"
  fi
  if [ -x "$REPO_ROOT/sidecar/.venv/bin/python" ]; then ok "sidecar/.venv"; else miss "sidecar/.venv"; fi
  if [ -d "$REPO_ROOT/desktop/node_modules" ]; then ok "desktop/node_modules"; else miss "desktop/node_modules"; fi
  if [ -f "$REPO_ROOT/sidecar/models/$DEFAULT_MODEL.pt" ]; then
    ok "default model $DEFAULT_MODEL.pt"
  else
    miss "default model $DEFAULT_MODEL.pt (setup downloads it from the models release)"
  fi
  exit 0
fi

# --- install what is missing ---------------------------------------------------------------------

if [ "$PM" = brew ] && ! have brew; then
  die "Homebrew is required on macOS - install it from https://brew.sh and re-run"
fi

if [ $need_git -eq 1 ] || [ $need_make -eq 1 ]; then
  say "Installing git/make"
  if [ "$PM" = brew ]; then
    # Both ship with the Xcode Command Line Tools; brew covers whatever they did not.
    xcode-select -p >/dev/null 2>&1 || xcode-select --install || true
    if ! have git; then brew install git; fi
    if ! have make; then brew install make; fi
  else
    pkgs=()
    [ $need_git -eq 1 ] && pkgs+=(git)
    [ $need_make -eq 1 ] && pkgs+=(make)
    pm_install "${pkgs[@]}"
  fi
fi

if [ $need_node -eq 1 ]; then
  say "Installing Node.js $NODE_MAJOR_MIN+"
  case "$PM" in
    brew)   brew install "node@$NODE_MAJOR_INSTALL" && brew link --overwrite --force "node@$NODE_MAJOR_INSTALL" ;;
    pacman) pm_install nodejs npm ;;
    apk)    pm_install nodejs npm ;;
    apt-get)
      # Ubuntu/Debian's own nodejs is usually far older than 20, so go straight to NodeSource.
      pm_install ca-certificates curl gnupg
      curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR_INSTALL}.x" | $SUDO -E bash -
      $SUDO apt-get install -y nodejs
      ;;
    dnf|yum)
      pm_install nodejs npm || true
      if ! have node || [ "$(node_major)" -lt "$NODE_MAJOR_MIN" ]; then
        curl -fsSL "https://rpm.nodesource.com/setup_${NODE_MAJOR_INSTALL}.x" | $SUDO bash -
        $SUDO "$PM" install -y nodejs
      fi
      ;;
    zypper) pm_install "nodejs$NODE_MAJOR_INSTALL" "npm$NODE_MAJOR_INSTALL" || pm_install nodejs npm ;;
  esac
  hash -r
  { have node && [ "$(node_major)" -ge "$NODE_MAJOR_MIN" ]; } \
    || die "Node.js $NODE_MAJOR_MIN+ is still not on PATH - install it from https://nodejs.org and re-run"
  ok "node $(node --version)"
fi

if [ $need_uv -eq 1 ]; then
  say "Installing uv"
  if [ "$PM" = brew ]; then
    brew install uv
  else
    have curl || pm_install curl
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  hash -r
  have uv || die "uv installed but is not on PATH - open a new terminal and re-run"
  ok "$(uv --version)"
fi

say "Installing Python $PYTHON_VERSION (managed by uv)"
uv python install "$PYTHON_VERSION"

if [ "$MODE" = tools ]; then say "System tools ready"; exit 0; fi

# --- project install (what `make install` does) --------------------------------------------------

# `npm ci`, not `npm install`: it installs exactly the lockfile and never writes it. npm 10 (Node
# 22's) rewrites a lockfile npm 11 wrote, so `npm install` left a fresh clone with a modified file.
say "Installing desktop dependencies (npm ci - Electron's download can take a while)"
(cd "$REPO_ROOT/desktop" && npm ci)

say "Creating sidecar/.venv with Python $PYTHON_VERSION and installing requirements.lock.txt"
(
  cd "$REPO_ROOT/sidecar"
  [ -x .venv/bin/python ] || uv venv --python "$PYTHON_VERSION" .venv
  uv pip install --python .venv/bin/python -r requirements.lock.txt
  # Prove the venv loads what the sidecar loads now, rather than at the first app start.
  say "Checking the sidecar can import its libraries"
  .venv/bin/python -c "import torch, cv2, ultralytics, fastapi, onnxruntime, lap; print('  ok      torch', torch.__version__, '/ opencv', cv2.__version__, '/ ultralytics', ultralytics.__version__)"
)

# The installed weights are committed under sidecar/models/; any the checkout lacks are fetched
# from the assets of the repo's `models` GitHub Release by the same script `make models` and
# setup.ps1 run. It keeps any weight already here, and a missing
# release or network only warns.
if [ $FETCH_MODELS -eq 1 ]; then
  say "Downloading the model weights from the repo's GitHub release"
  (cd "$REPO_ROOT" && sidecar/.venv/bin/python scripts/fetch_models.py)
fi
if [ ! -f "$REPO_ROOT/sidecar/models/$DEFAULT_MODEL.pt" ]; then
  printf '\033[33mwarning:\033[0m %s\n' \
    "The default model, sidecar/models/$DEFAULT_MODEL.pt, is not here - the app starts, but has no grocery detector." \
    "It is committed in git - run \`git pull\` - or attach it and its .json to the repo's \`models\` release and re-run," \
    "copy them into sidecar/models/ by hand, or pick a stock model such as yolo11n.pt in Admin > Model."
elif [ ! -f "$REPO_ROOT/sidecar/models/$DEFAULT_MODEL.json" ]; then
  printf '\033[33mwarning:\033[0m %s\n' \
    "sidecar/models/$DEFAULT_MODEL.json is missing: without it 'auto' guesses the frame geometry wrong for this weight."
fi

if [ $RUN_TESTS -eq 1 ]; then
  say "Running both test suites (fakes only - no camera needed)"
  (cd "$REPO_ROOT" && make test)
fi

say "Done. Next:"
echo "    make test   # confirm both toolchains work (no camera needed)"
echo "    make dev    # launch the app"
