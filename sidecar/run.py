import os
import socket
import time

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


def bind_port(preferred: int) -> socket.socket:
    """Bind and listen, and hand back the live socket.

    Deliberately not "pick a port, close it, let uvicorn bind it again".
    Announcing a port nothing is listening on yet is a race the renderer
    loses: main/index.ts forwards SIDECAR_PORT the instant it is printed, the
    AppShell mounts, and the settings hook's one-shot load lands on a port
    that is not accepting — a bare "Failed to fetch" with no retry behind it.
    Handing uvicorn a socket that is already listening means the kernel
    queues that connection in the backlog instead of refusing it, so an early
    request waits rather than failing.

    It also closes the window where another process took the port between our
    test bind and uvicorn's real one.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", preferred))
    except OSError:
        s.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
    s.listen(128)
    return s


def watch_parent(poll_s: float = 2.0) -> None:
    """Exit when whoever launched us is gone.

    Electron kills the sidecar on `before-quit`, but that never runs if the app
    crashes or is force-killed — the sidecar is then orphaned and keeps its
    camera, its port and its threads forever. One such orphan was measured
    holding all 12 cores at 100%, which dragged live inference from 56 ms to
    500 ms (17.8 fps to 2.0) for the app that *was* running. They also
    accumulate silently, since each one is invisible until you go looking.

    Which process to watch is set by SIDECAR_PARENT_PID, because the immediate
    parent is not the app: `.venv/Scripts/python.exe` on Windows is a shim that
    re-execs the real uv-managed interpreter, so the sidecar's parent is that
    shim. Electron passes its own pid; without the variable (a manual launch)
    we fall back to the immediate parent, which is right for a shell.

    The target is identified by (pid, create_time): a bare pid can be recycled
    by an unrelated process, and exiting on that would be worse than the leak.
    Started as a daemon thread so it never blocks shutdown; uses os._exit
    because uvicorn owns the main thread and will not return on its own.
    """
    import threading

    import psutil

    try:
        declared = os.environ.get("SIDECAR_PARENT_PID")
        target = psutil.Process(int(declared)) if declared else psutil.Process(os.getpid()).parent()
        if target is None:
            return
        pid, started = target.pid, target.create_time()
    except Exception:  # noqa: BLE001 - never let the watchdog stop the sidecar
        return

    def watch() -> None:
        while True:
            time.sleep(poll_s)
            try:
                if not psutil.pid_exists(pid) or psutil.Process(pid).create_time() != started:
                    print("[sidecar] parent process is gone — exiting", flush=True)
                    os._exit(0)
            except psutil.NoSuchProcess:
                os._exit(0)
            except Exception:  # noqa: BLE001 - a transient probe failure is not a reason to die
                pass

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    import uvicorn
    watch_parent()
    sock = bind_port(8765)
    port = sock.getsockname()[1]
    print(f"SIDECAR_PORT={port}", flush=True)
    config = uvicorn.Config(build_app(), host="127.0.0.1", port=port)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()
