"""Swap the sidecar venv's CPU torch/onnxruntime for the NVIDIA CUDA builds, when there is a driver.

A plain `pip install -r requirements.txt` pulls **CPU** torch even on a CUDA machine (requirements.txt
does not pin torch, and PyPI's Windows wheel is the CPU one). The sidecar trusts
`torch.cuda.is_available()` for everything - the device it runs on, and the hardware the Admin panel
reports and recommends a preset from - so an RTX laptop set up that way ran on its CPU and was
offered a CPU preset. `setup.cmd` / `scripts/setup.ps1` and `make sidecar-setup` (so `make install`)
both run this, so there is one copy of the rule (docs/DEVELOPMENT.md section 6, as commands):

* no NVIDIA driver (`nvidia-smi` not on PATH): keep the CPU builds and change nothing;
* torch and torchvision are replaced together from PyTorch's CUDA index (one alone fails on import
  with "torchvision::nms does not exist"), and only when torch is not a CUDA build already, so a
  re-run does not download ~2.5 GB again;
* onnxruntime is swapped for -gpu on **every** run, never installed beside it: the two share an
  import name and break each other, and the requirements install puts the CPU build back on a
  re-run (removing it deletes the shared directory, so -gpu is reinstalled after);
* a GPU torch then cannot use only warns - the app still runs, on the CPU - unless `--require`.

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
#: The PyTorch CUDA wheel tag; cu124 pairs with requirements-cuda.txt's onnxruntime-gpu.
DEFAULT_CUDA_TAG = "cu124"

_IS_CUDA_BUILD = "import sys, torch; sys.exit(0 if torch.version.cuda else 1)"
_GPU_CHECK = (
    "import sys, torch; ok = torch.cuda.is_available(); "
    "print('  ok      CUDA', torch.version.cuda, torch.cuda.get_device_name(0)) if ok "
    "else print('  torch cannot see the GPU'); sys.exit(0 if ok else 1)"
)


def install(
    python: str,
    *,
    cuda_tag: str = DEFAULT_CUDA_TAG,
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

    if run([python, "-c", _IS_CUDA_BUILD]).returncode == 0:
        say("  ok      torch is already a CUDA build")
    else:
        say(f"==> NVIDIA driver found - installing CUDA torch ({cuda_tag}); --cpu keeps the CPU build")
        if not step("CUDA torch", [
            "uv", "pip", "install", "--python", python, "torch", "torchvision",
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
            say(f"error: torch cannot use the NVIDIA GPU - is the driver new enough for {cuda_tag}?")
            return 1
        say("warning: torch cannot use the NVIDIA GPU; the app will run on the CPU. "
            "Update the driver (nvidia.com), then re-run the setup.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable,
                        help="the venv interpreter to install into (default: this one)")
    parser.add_argument("--cuda-tag", default=DEFAULT_CUDA_TAG, help="PyTorch CUDA wheel tag")
    parser.add_argument("--require", action="store_true",
                        help="fail when there is no NVIDIA driver or torch cannot use the GPU")
    parser.add_argument("--cpu", action="store_true", help="keep the CPU builds")
    args = parser.parse_args(argv)
    return install(args.python, cuda_tag=args.cuda_tag, require=args.require, cpu=args.cpu)


if __name__ == "__main__":
    sys.exit(main())
