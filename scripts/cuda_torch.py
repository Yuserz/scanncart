"""Swap the sidecar venv's CPU torch/onnxruntime for the NVIDIA CUDA builds, when there is a driver.

A plain `pip install -r requirements.txt` pulls **CPU** torch even on a CUDA machine (requirements.txt
does not pin torch, and PyPI's Windows wheel is the CPU one). The sidecar trusts
`torch.cuda.is_available()` for everything - the device it runs on, and the hardware the Admin panel
reports and recommends a preset from - so an RTX laptop set up that way ran on its CPU and was
offered a CPU preset. `setup.cmd` / `scripts/setup.ps1` and `make sidecar-setup` (so `make install`)
both run this, so there is one copy of the rule (docs/DEVELOPMENT.md section 6, as commands):

* no NVIDIA driver (`nvidia-smi` not on PATH): keep the CPU builds and change nothing;
* torch and torchvision are replaced together from PyTorch's CUDA index (one alone fails on import
  with "torchvision::nms does not exist"), at the exact versions sidecar/requirements.lock.txt pins
  (unpinned, each machine got the newest its index carried that day - cu124's is 2.6.0, five
  releases behind the dev machine), and only when the venv's torch is not that exact CUDA build
  already, so a re-run does not download ~2.5 GB again;
* the CUDA tag follows the card: cu128's builds carry kernels for sm_75 (Turing) and newer only, so
  an older card (GTX 10xx, MX) gets cu126, which still builds them - the wrong one installs fine,
  reports `is_available()`, and fails on the first kernel;
* onnxruntime is swapped for -gpu on **every** run, never installed beside it: the two share an
  import name and break each other, and the requirements install puts the CPU build back on a
  re-run (removing it deletes the shared directory, so -gpu is reinstalled after);
* a GPU torch then cannot run a kernel on only warns - the app still runs, on the CPU
  (app/settings.py's resolve_device makes the same check) - unless `--require`.

Standard library only, so it runs in a venv that has just been created. Run it with the venv's own
interpreter (Windows: `sidecar\\.venv\\Scripts\\python.exe scripts\\cuda_torch.py`); `--cpu` keeps the
CPU builds, `--require` makes a missing driver or an unusable GPU a failure.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_REQUIREMENTS = REPO_ROOT / "sidecar" / "requirements-cuda.txt"
LOCK = REPO_ROOT / "sidecar" / "requirements.lock.txt"
#: The PyTorch CUDA wheel tag for Turing (sm_75) and newer, Blackwell included; CUDA 12.8 with
#: cuDNN 9 is what requirements-cuda.txt's onnxruntime-gpu binds to.
DEFAULT_CUDA_TAG = "cu128"
#: For cards older than Turing, which cu128 no longer builds kernels for. Also CUDA 12 / cuDNN 9.
LEGACY_CUDA_TAG = "cu126"
TURING = (7, 5)

_IS_TARGET_BUILD = "import sys, torch; sys.exit(0 if torch.__version__ == sys.argv[1] else 1)"
# A kernel, not `is_available()`: that is True for a card the build has no kernels for.
_GPU_CHECK = (
    "import sys, torch\n"
    "try:\n"
    "    ok = torch.cuda.is_available() and (torch.ones(8, device='cuda') * 2).sum().item() == 16\n"
    "except Exception as e:\n"
    "    print('  torch cannot run on the GPU:', e); sys.exit(1)\n"
    "print('  ok      CUDA', torch.version.cuda, torch.cuda.get_device_name(0)) if ok "
    "else print('  torch cannot see the GPU'); sys.exit(0 if ok else 1)"
)


def locked_version(name: str, lock: Path = LOCK) -> str:
    """The version requirements.lock.txt pins `name` to (the CPU build's; CUDA builds add +cuXXX)."""
    for line in lock.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}=="):
            return line.split("==", 1)[1].split(";", 1)[0].strip()
    raise SystemExit(f"error: {name} is not pinned in {lock}")


def compute_capability(run: Callable[..., subprocess.CompletedProcess]) -> tuple[int, int] | None:
    """The first GPU's compute capability from nvidia-smi, or None when it cannot say."""
    try:
        out = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                  capture_output=True, text=True)
        major, minor = out.stdout.strip().splitlines()[0].strip().split(".")
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - an old driver without the field, or no output at all
        return None


def cuda_tag_for(cap: tuple[int, int] | None) -> str:
    return LEGACY_CUDA_TAG if cap is not None and cap < TURING else DEFAULT_CUDA_TAG


def install(
    python: str,
    *,
    cuda_tag: str | None = None,
    require: bool = False,
    cpu: bool = False,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    say: Callable[[str], None] = lambda m: print(m, flush=True),
) -> int:
    """0 when the venv ends up right for this machine (GPU builds, or CPU with no driver)."""
    if cpu:
        say("==> --cpu: keeping the CPU builds of torch and onnxruntime")
        return 0
    if not which("nvidia-smi"):
        if require:
            say("error: the GPU builds need an NVIDIA driver (nvidia-smi was not found). "
                "Install it from nvidia.com, or run without --require.")
            return 1
        say("==> No NVIDIA driver (nvidia-smi) - keeping the CPU builds of torch and onnxruntime")
        return 0
    if not which("uv"):
        say("warning: an NVIDIA driver is installed, but uv is needed to swap in the CUDA builds of "
            "torch - install it (winget install astral-sh.uv) and re-run. The app runs on the CPU "
            "until then.")
        return 1 if require else 0

    def step(what: str, cmd: list[str]) -> bool:
        if run(cmd).returncode != 0:
            say(f"error: {what} failed")
            return False
        return True

    if cuda_tag is None:
        cap = compute_capability(run)
        cuda_tag = cuda_tag_for(cap)
        if cap is not None:
            say(f"  ok      NVIDIA GPU, compute capability {cap[0]}.{cap[1]} - using the {cuda_tag} build")
    torch_v, vision_v = locked_version("torch"), locked_version("torchvision")
    target = f"{torch_v}+{cuda_tag}"
    if run([python, "-c", _IS_TARGET_BUILD, target]).returncode == 0:
        say(f"  ok      torch is already {target}")
    else:
        say(f"==> NVIDIA driver found - installing torch {target}; --cpu keeps the CPU build")
        if not step("CUDA torch", [
            "uv", "pip", "install", "--python", python,
            f"torch=={torch_v}", f"torchvision=={vision_v}",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
            "--reinstall-package", "torch", "--reinstall-package", "torchvision",
        ]):
            return 1
    say("==> Swapping onnxruntime for onnxruntime-gpu")
    if not step("removing CPU onnxruntime", ["uv", "pip", "uninstall", "--python", python, "onnxruntime"]):
        return 1
    if not step("onnxruntime-gpu", [
        "uv", "pip", "install", "--python", python, "-r", str(CUDA_REQUIREMENTS),
        "--reinstall-package", "onnxruntime-gpu",
    ]):
        return 1
    if run([python, "-c", _GPU_CHECK]).returncode != 0:
        if require:
            say(f"error: torch cannot use the NVIDIA GPU - is the driver new enough for {cuda_tag}?"
                " (pass --cuda-tag to pick another build)")
            return 1
        say("warning: torch cannot use the NVIDIA GPU; the app will run on the CPU. "
            "Update the driver (nvidia.com), then re-run the setup.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable,
                        help="the venv interpreter to install into (default: this one)")
    parser.add_argument("--cuda-tag", default=None,
                        help=f"PyTorch CUDA wheel tag (default: {DEFAULT_CUDA_TAG}, or "
                             f"{LEGACY_CUDA_TAG} for a card older than Turing)")
    parser.add_argument("--require", action="store_true",
                        help="fail when there is no NVIDIA driver or torch cannot use the GPU")
    parser.add_argument("--cpu", action="store_true", help="keep the CPU builds")
    args = parser.parse_args(argv)
    return install(args.python, cuda_tag=args.cuda_tag, require=args.require, cpu=args.cpu)


if __name__ == "__main__":
    sys.exit(main())
