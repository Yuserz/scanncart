"""Roboflow inference server, run natively — no Docker.

`inference server start` (the documented path in docs/DETECTOR_BACKENDS.md)
pulls and runs a Docker image. This module builds the same FastAPI app the
image serves, so the `local_api` backend works on a machine with no Docker.

Deliberately kept out of `app/`: it imports the `inference` package, which
lives in its own venv (`.venv-inference`) because its pins on numpy/opencv
conflict with the sidecar's ultralytics stack. The sidecar never imports this
module — it only talks to it over HTTP at `local_api_url`.

Run:
    .venv-inference/Scripts/python.exe local_inference_server.py

Reads ROBOFLOW_API_KEY from the environment or sidecar/.env — the server needs
it to pull the private grocery model's weights from Roboflow the first time a
workflow runs. After that the weights are cached on disk and the server keeps
working offline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
PORT = int(os.environ.get("INFERENCE_PORT", "9001"))


def _load_key() -> None:
    """Reuse the sidecar's own .env loader so both processes read one file."""
    from app.credentials import ROBOFLOW_API_KEY_ENV, load_api_key

    if os.environ.get(ROBOFLOW_API_KEY_ENV):
        return
    key = load_api_key()
    if key:
        os.environ[ROBOFLOW_API_KEY_ENV] = key
        print(f"{ROBOFLOW_API_KEY_ENV} loaded from .env", flush=True)
    else:
        print(
            f"WARNING: no {ROBOFLOW_API_KEY_ENV}. The server starts, but fetching "
            "the private grocery model will fail with 401.",
            flush=True,
        )


# HttpInterface unconditionally mounts the landing-page bundle from
# "./inference/landing/out/..." — a path relative to the working directory that
# only exists inside the Docker image (the wheel does not ship the built
# assets). Stub the directories so the mount succeeds; the landing UI 404s,
# which is irrelevant — the sidecar only ever calls the workflow endpoint.
_LANDING_DIRS = (
    "inference/landing/out/static",
    "inference/landing/out/_next/static",
)
RUNTIME_ROOT = os.environ.get(
    "INFERENCE_RUNTIME_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".inference-runtime"),
)


def _prepare_runtime_root() -> None:
    for rel in _LANDING_DIRS:
        os.makedirs(os.path.join(RUNTIME_ROOT, rel), exist_ok=True)


def build_app():
    """The same wiring docker/config/cpu_http.py uses inside the image."""
    # Import before the chdir: RUNTIME_ROOT contains an `inference/` directory,
    # and resolving the real package first means it can never be shadowed.
    from inference.core.interfaces.http.http_api import HttpInterface
    from inference.core.managers.base import ModelManager
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
    from inference.core.registries.roboflow import RoboflowModelRegistry
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    _prepare_runtime_root()
    os.chdir(RUNTIME_ROOT)

    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    manager = WithFixedSizeCache(
        ModelManager(model_registry=registry),
        max_size=int(os.environ.get("MAX_ACTIVE_MODELS", "8")),
    )
    return HttpInterface(manager).app


_load_key()
app = build_app()


if __name__ == "__main__":
    import uvicorn

    print(f"Roboflow inference server (no Docker) on http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
