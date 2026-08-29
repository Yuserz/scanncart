import os
import socket

# Ultralytics "AutoUpdate" pip-installs what it thinks it needs at import time.
# On a machine where torch reports CUDA it swaps our pinned CPU `onnxruntime`
# for `onnxruntime-gpu`, which then advertises CUDAExecutionProvider and dies
# on the first frame with "no data transfer registered" because its CUDA/cuDNN
# runtime is not installed (cublasLt64_13.dll missing). That took capture down
# mid-session. The sidecar pins its own dependencies; nothing may change them
# underneath it at runtime.
#
# Set before importing anything that pulls in ultralytics.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from app.main import build_app  # noqa: E402 - must follow the env guard above


def pick_port(preferred: int) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", preferred))
        s.close()
        return preferred
    except OSError:
        s.close()
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.bind(("127.0.0.1", 0))
        port = s2.getsockname()[1]
        s2.close()
        return port


def main() -> None:
    import uvicorn
    port = pick_port(8765)
    print(f"SIDECAR_PORT={port}", flush=True)
    uvicorn.run(build_app(), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
