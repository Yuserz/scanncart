"""The sidecar entrypoint's environment guard.

Ultralytics "AutoUpdate" pip-installs at import time. On a machine where torch
reports CUDA it replaced our pinned CPU `onnxruntime` with `onnxruntime-gpu`,
which advertises CUDAExecutionProvider and then dies on the first frame with
"no data transfer registered" (its cublasLt64_13.dll is not installed). That
took capture down mid-session, so the guard is load-bearing.
"""

import os
import re
from pathlib import Path

RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def test_importing_run_disables_ultralytics_autoinstall():
    import run  # noqa: F401 - imported for its module-level side effect

    assert os.environ["YOLO_AUTOINSTALL"] == "false"


def test_the_guard_is_set_before_anything_imports_ultralytics():
    """Order matters: ultralytics reads the flag at import time, so setting it
    after `from app.main import ...` would be too late."""
    source = RUN_PY.read_text(encoding="utf-8")
    guard = source.index("YOLO_AUTOINSTALL")
    app_import = source.index("from app.main import")

    assert guard < app_import


def test_the_guard_does_not_override_an_explicit_setting():
    """setdefault, not assignment — someone debugging can still turn it on."""
    source = RUN_PY.read_text(encoding="utf-8")
    assert re.search(r"environ\.setdefault\(\s*[\"']YOLO_AUTOINSTALL", source)
