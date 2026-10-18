"""`scripts/cuda_torch.py`: the NVIDIA builds of torch and onnxruntime, on any machine with a driver.

A plain `pip install -r requirements.txt` pulls CPU torch even on a CUDA machine. That is how an RTX
laptop (12 cores, 8 GB, RTX 4050) set up with `make install` ran on its CPU and was recommended a CPU
preset: only `setup.cmd` swapped the GPU builds in. Both now call this one script, so there is one
copy of the rule. Run against an injected `which` and `run`, never the real installer.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("cuda_torch", REPO_ROOT / "scripts" / "cuda_torch.py")
cuda_torch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cuda_torch)

PY = "C:/repo/sidecar/.venv/Scripts/python.exe"


class _Machine:
    """A fake PATH plus a fake process runner. `torch_cuda` is whether the venv's torch is already
    a CUDA build; `gpu_ok` whether the final check sees the GPU; `fail` a command word to fail on."""

    def __init__(self, *, driver=True, uv=True, torch_cuda=False, gpu_ok=True, fail=None):
        self.tools = {"nvidia-smi": driver, "uv": uv}
        self.torch_cuda = torch_cuda
        self.gpu_ok = gpu_ok
        self.fail = fail
        self.ran: list[list[str]] = []

    def which(self, name):
        return f"C:/bin/{name}.exe" if self.tools.get(name) else None

    def run(self, cmd, **_kwargs):
        self.ran.append(list(cmd))
        text = " ".join(cmd)
        code = 0
        if "torch.version.cuda" in text:
            code = 0 if self.torch_cuda else 1
        elif "torch.cuda.is_available" in text:
            code = 0 if self.gpu_ok else 1
        elif self.fail and self.fail in text:
            code = 1
        return subprocess.CompletedProcess(cmd, code)

    def installs(self):
        return [c for c in self.ran if c[:3] in (["uv", "pip", "install"], ["uv", "pip", "uninstall"])]


def _install(machine, **kwargs):
    said: list[str] = []
    code = cuda_torch.install(PY, which=machine.which, run=machine.run, say=said.append, **kwargs)
    return code, "\n".join(said)


def test_a_machine_with_an_nvidia_driver_gets_cuda_torch_and_onnxruntime_gpu():
    machine = _Machine(torch_cuda=False)
    code, said = _install(machine)
    assert code == 0
    torch_install = next(c for c in machine.installs() if "torch" in c)
    assert "https://download.pytorch.org/whl/cu124" in torch_install
    assert {"torch", "torchvision"} <= set(torch_install)
    assert ["uv", "pip", "uninstall", "--python", PY, "onnxruntime"] in machine.installs()
    assert any("requirements-cuda.txt" in " ".join(c) for c in machine.installs())
    assert "NVIDIA" in said


def test_a_rerun_keeps_a_cuda_torch_but_still_redoes_the_onnxruntime_swap():
    # The requirements install puts the CPU onnxruntime back beside -gpu on every re-run, and the
    # two share an import name - so the swap is not skippable, while a 2.5 GB torch re-download is.
    machine = _Machine(torch_cuda=True)
    code, said = _install(machine)
    assert code == 0
    assert not any("torchvision" in c for c in machine.installs())
    assert ["uv", "pip", "uninstall", "--python", PY, "onnxruntime"] in machine.installs()
    assert "already a CUDA build" in said


def test_no_driver_keeps_the_cpu_builds_and_touches_nothing():
    machine = _Machine(driver=False)
    code, said = _install(machine)
    assert code == 0
    assert machine.ran == []
    assert "CPU" in said


def test_require_makes_a_missing_driver_a_failure():
    code, said = _install(_Machine(driver=False), require=True)
    assert code == 1
    assert "nvidia-smi" in said


def test_cpu_opts_out_even_with_a_driver():
    machine = _Machine(driver=True)
    code, _ = _install(machine, cpu=True)
    assert code == 0
    assert machine.ran == []


def test_a_gpu_torch_cannot_use_only_warns_when_found_automatically():
    # The app still runs - on the CPU, as it would have without this step - so a setup that found
    # the GPU on its own does not fail over it. Asked for with --require, it does.
    assert _install(_Machine(gpu_ok=False))[0] == 0
    assert _install(_Machine(gpu_ok=False), require=True)[0] == 1


def test_a_failed_install_step_fails_the_setup():
    # Half a swap (onnxruntime removed, -gpu never installed) is a broken venv, not a CPU fallback.
    code, said = _install(_Machine(fail="requirements-cuda.txt"))
    assert code == 1
    assert "failed" in said


def test_without_uv_it_says_how_to_get_the_gpu_rather_than_guessing_at_pip():
    machine = _Machine(uv=False)
    code, said = _install(machine)
    assert code == 0
    assert machine.installs() == []
    assert "uv" in said
