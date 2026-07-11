import asyncio
import queue
from dataclasses import dataclass, field
from typing import Callable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.settings import Settings, resolve_device
from app.pipeline import Pipeline
from app.schemas import HealthResponse
from app.camera import CameraCapture
from app.inference import YoloDetector


def _default_source_factory(settings: Settings):
    return CameraCapture(
        settings.camera_index, settings.capture_width,
        settings.capture_height, settings.capture_fps,
    )


def _default_detector_factory(settings: Settings, device: str):
    return YoloDetector(settings.active_model, device=device, conf=settings.conf_threshold)


class WSManager:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=4)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        # Bind the serving loop lazily at handshake time. This runs on the
        # asyncio loop that actually serves requests, so the pipeline thread's
        # submit() can hand frames back to it — and it works with a bare
        # TestClient (which does not run startup handlers).
        self._loop = asyncio.get_running_loop()
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    def submit(self, message: dict) -> None:
        # Called from the pipeline thread; hand off to the event loop.
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            return
        coro = self._drain()
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            # Loop is shutting down between the is_running() check and now;
            # close the coroutine so it is not left un-awaited.
            coro.close()

    async def _drain(self) -> None:
        while not self._queue.empty():
            msg = self._queue.get_nowait()
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws)


@dataclass
class AppState:
    settings: Settings = field(default_factory=Settings)
    source_factory: Callable = _default_source_factory
    detector_factory: Callable = _default_detector_factory
    ws_manager: WSManager = field(default_factory=WSManager)
    pipeline: Pipeline | None = None
    state: str = "idle"
    device: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = resolve_device(self.settings.device)


def build_app(state_factory: Callable[[], AppState] = AppState) -> FastAPI:
    app = FastAPI(title="SCANnCART Sidecar")
    state = state_factory()

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            state=state.state,
            active_model=state.settings.active_model,
            device=state.device,
        )

    @app.post("/api/capture/start")
    async def start():
        if state.state != "running":
            source = state.source_factory(state.settings)
            if hasattr(source, "open"):
                source.open()
            detector = state.detector_factory(state.settings, state.device)
            state.pipeline = Pipeline(
                source, detector, state.settings,
                on_message=state.ws_manager.submit,
            )
            state.pipeline.start()
            state.state = "running"
        return {"state": state.state}

    @app.post("/api/capture/stop")
    async def stop():
        if state.pipeline is not None:
            state.pipeline.stop()
            state.pipeline = None
        state.state = "idle"
        return {"state": state.state}

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        await state.ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            state.ws_manager.disconnect(ws)

    return app
