import asyncio
import queue
from dataclasses import dataclass, field
from typing import Callable
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from app.settings import Settings, resolve_device
from app.settings_store import (
    HOT_RELOADABLE_FIELDS,
    RESTART_REQUIRED_FIELDS,
    compute_warnings,
    load_settings,
    save_settings,
)
from app.hardware import HardwareInfo, probe_hardware
from app.presets import PRESETS, recommend_preset
from app.pipeline import Pipeline
from app.schemas import (
    ApplyPresetRequest,
    HealthResponse,
    LogEvent,
    LogsResponse,
    PresetInfo,
    PresetsResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    SystemInfoResponse,
)
from app.camera import CameraCapture
from app.inference import YoloDetector
from app.logging_store import LoggingStore


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
    settings: Settings | None = None
    settings_path: str = "data/settings.json"
    source_factory: Callable = _default_source_factory
    detector_factory: Callable = _default_detector_factory
    ws_manager: WSManager = field(default_factory=WSManager)
    pipeline: Pipeline | None = None
    state: str = "idle"
    device: str = ""
    db_path: str = "data/scanncart.db"
    logging_store: LoggingStore | None = None
    session_id: int | None = None
    hardware_info: HardwareInfo | None = None
    # Injection seam so tests can supply a fake instead of the real probe
    # (which shells out to PowerShell and reads torch.cuda).
    hardware_prober: Callable[[], HardwareInfo] = probe_hardware

    def __post_init__(self):
        if self.settings is None:
            self.settings = load_settings(self.settings_path)
        if not self.device:
            self.device = resolve_device(self.settings.device)
        if self.logging_store is None:
            self.logging_store = LoggingStore(self.db_path)


def _settings_response(state: "AppState") -> SettingsResponse:
    return SettingsResponse(
        active_model=state.settings.active_model,
        camera_index=state.settings.camera_index,
        capture_width=state.settings.capture_width,
        capture_height=state.settings.capture_height,
        capture_fps=state.settings.capture_fps,
        conf_threshold=state.settings.conf_threshold,
        infer_frame_skip=state.settings.infer_frame_skip,
        device=state.settings.device,
        preview_height=state.settings.preview_height,
        track_expiry_s=state.settings.track_expiry_s,
        hot_reloadable_fields=sorted(HOT_RELOADABLE_FIELDS),
        restart_required_fields=sorted(RESTART_REQUIRED_FIELDS),
        warnings=compute_warnings(state.settings, state.state),
    )


def _apply_settings_patch(state: "AppState", patch: dict) -> SettingsResponse:
    if state.state == "running":
        locked = set(patch) & RESTART_REQUIRED_FIELDS
        if locked:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot change {sorted(locked)} while capture is running; stop capture first.",
            )
    for key, value in patch.items():
        setattr(state.settings, key, value)
    if "device" in patch:
        state.device = resolve_device(state.settings.device)
    save_settings(state.settings, state.settings_path)
    return _settings_response(state)


def build_app(state_factory: Callable[[], AppState] = AppState) -> FastAPI:
    app = FastAPI(title="SCANnCART Sidecar")
    # The renderer's origin varies by mode (Vite dev server port, or a
    # packaged app's file:// origin) and this server only ever binds to
    # 127.0.0.1 as a locally-spawned child process, so allow any origin
    # rather than hand-tracking renderer origins here.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = state_factory()

    # Guard the first probe so concurrent callers (AdminPanel mount fires
    # /api/system-info and /api/presets together) don't each spawn a probe;
    # double-checked against the cache so subsequent calls skip the lock.
    hw_lock = asyncio.Lock()

    async def _get_hardware() -> HardwareInfo:
        if state.hardware_info is None:
            async with hw_lock:
                if state.hardware_info is None:
                    state.hardware_info = await run_in_threadpool(state.hardware_prober)
        return state.hardware_info

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            state=state.state,
            active_model=state.settings.active_model,
            device=state.device,
        )

    @app.get("/api/settings", response_model=SettingsResponse)
    async def get_settings():
        return _settings_response(state)

    @app.patch("/api/settings", response_model=SettingsResponse)
    async def update_settings(body: SettingsUpdateRequest):
        patch = body.model_dump(exclude_none=True)
        return _apply_settings_patch(state, patch)

    @app.get("/api/system-info", response_model=SystemInfoResponse)
    async def system_info():
        hw = await _get_hardware()
        return SystemInfoResponse(
            cpu_count=hw.cpu_count,
            ram_gb=hw.ram_gb,
            cuda_available=hw.cuda_available,
            accelerator=hw.accelerator,
            gpu_name=hw.gpu_name,
            gpu_vram_gb=hw.gpu_vram_gb,
            recommended_preset=recommend_preset(hw),
        )

    @app.get("/api/presets", response_model=PresetsResponse)
    async def presets():
        hw = await _get_hardware()
        return PresetsResponse(
            presets=[
                PresetInfo(name=p.name, label=p.label, description=p.description, settings=p.settings)
                for p in PRESETS.values()
            ],
            recommended=recommend_preset(hw),
        )

    @app.post("/api/settings/preset", response_model=SettingsResponse)
    async def apply_preset(body: ApplyPresetRequest):
        preset = PRESETS.get(body.name)
        if preset is None:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {body.name}")
        return _apply_settings_patch(state, preset.settings)

    @app.post("/api/capture/start")
    async def start():
        if state.state != "running":
            source = state.source_factory(state.settings)
            if hasattr(source, "open"):
                source.open()
            detector = state.detector_factory(state.settings, state.device)
            state.session_id = state.logging_store.start_session(
                state.settings.active_model, state.device
            )
            state.pipeline = Pipeline(
                source, detector, state.settings,
                on_message=state.ws_manager.submit,
                logging_store=state.logging_store,
                session_id=state.session_id,
            )
            state.pipeline.start()
            state.state = "running"
        return {"state": state.state}

    @app.post("/api/capture/stop")
    async def stop():
        if state.pipeline is not None:
            state.pipeline.stop()
            state.pipeline.resolve_open_tracks()
            state.pipeline = None
        if state.session_id is not None:
            state.logging_store.end_session(state.session_id)
            state.session_id = None
        state.state = "idle"
        return {"state": state.state}

    @app.get("/api/logs", response_model=LogsResponse)
    async def logs():
        sid = state.logging_store.current_session_id()
        if sid is None:
            return LogsResponse(session_id=None, events=[])
        events = [
            LogEvent(
                track_id=r.track_id,
                class_name=r.class_name,
                confidence=r.confidence,
                max_conf=r.max_conf,
                entered_at=r.entered_at,
                left_at=r.left_at,
            )
            for r in state.logging_store.query_events(sid)
        ]
        return LogsResponse(session_id=sid, events=events)

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        await state.ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            state.ws_manager.disconnect(ws)

    return app
