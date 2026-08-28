import asyncio
import queue

import numpy as np
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
from app.credentials import has_api_key, load_api_key
from app.hardware import HardwareInfo, probe_hardware
from app.presets import PRESETS, recommend_preset
from app.pipeline import Pipeline
from app.roboflow import (
    RoboflowAuthError,
    RoboflowError,
    RoboflowTimeout,
    RoboflowUnavailable,
    WorkflowClient,
)
from app.tracking import IouTracker
from app.schemas import (
    ApplyPresetRequest,
    HealthResponse,
    LogEvent,
    LogsResponse,
    PresetInfo,
    DetectorProbeResponse,
    PresetsResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    SystemInfoResponse,
)
from app.camera import CameraCapture
from app.inference import RoboflowRemoteDetector, YoloDetector
from app.logging_store import LoggingStore


def _default_source_factory(settings: Settings):
    return CameraCapture(
        settings.camera_index, settings.capture_width,
        settings.capture_height, settings.capture_fps,
    )


def backend_url(settings: Settings) -> str:
    return (
        settings.local_api_url
        if settings.detector_backend == "local_api"
        else settings.cloud_api_url
    )


def _default_detector_factory(settings: Settings, device: str):
    if settings.detector_backend == "native":
        return YoloDetector(
            settings.active_model, device=device,
            conf=settings.conf_threshold, imgsz=settings.imgsz,
        )
    api_key = load_api_key()
    if api_key is None and settings.detector_backend == "cloud_api":
        # Fail here with an actionable message rather than as a 401 mid-capture.
        raise RoboflowAuthError(
            "No Roboflow API key. Set ROBOFLOW_API_KEY in sidecar/.env (see .env.example)."
        )
    client = WorkflowClient(
        api_url=backend_url(settings),
        workspace=settings.roboflow_workspace,
        workflow_id=settings.roboflow_workflow_id,
        api_key=api_key,
        timeout_s=settings.remote_timeout_s,
        max_retries=settings.remote_max_retries,
    )
    return RoboflowRemoteDetector(
        client,
        infer_size=settings.remote_infer_size,
        conf=settings.conf_threshold,
        # The workflow has no tracking block, so this is the only source of
        # stable track ids. Expiry matches the pipeline's so the two agree on
        # when a track is gone.
        tracker=IouTracker(expiry_s=settings.track_expiry_s),
    )
    )


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
    # Same seam for the API key, so settings tests never touch sidecar/.env.
    # Only ever reports presence — the key itself stays inside this process.
    api_key_probe: Callable[[], bool] = has_api_key

    def __post_init__(self):
        if self.settings is None:
            self.settings = load_settings(self.settings_path)
        if not self.device:
            self.device = resolve_device(self.settings.device)
        if self.logging_store is None:
            self.logging_store = LoggingStore(self.db_path)


def _settings_response(state: "AppState") -> SettingsResponse:
    api_key_present = state.api_key_probe()
    return SettingsResponse(
        active_model=state.settings.active_model,
        camera_index=state.settings.camera_index,
        capture_width=state.settings.capture_width,
        capture_height=state.settings.capture_height,
        capture_fps=state.settings.capture_fps,
        conf_threshold=state.settings.conf_threshold,
        imgsz=state.settings.imgsz,
        infer_frame_skip=state.settings.infer_frame_skip,
        device=state.settings.device,
        preview_height=state.settings.preview_height,
        track_expiry_s=state.settings.track_expiry_s,
        detector_backend=state.settings.detector_backend,
        roboflow_workspace=state.settings.roboflow_workspace,
        roboflow_workflow_id=state.settings.roboflow_workflow_id,
        local_api_url=state.settings.local_api_url,
        cloud_api_url=state.settings.cloud_api_url,
        remote_infer_size=state.settings.remote_infer_size,
        remote_timeout_s=state.settings.remote_timeout_s,
        remote_max_retries=state.settings.remote_max_retries,
        hot_reloadable_fields=sorted(HOT_RELOADABLE_FIELDS),
        restart_required_fields=sorted(RESTART_REQUIRED_FIELDS),
        warnings=compute_warnings(state.settings, state.state, api_key_present),
        roboflow_api_key_present=api_key_present,
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

    _ERROR_STATUS = {
        RoboflowAuthError: 401,
        RoboflowUnavailable: 503,
        RoboflowTimeout: 504,
    }

    def _http_from_roboflow(exc: RoboflowError) -> HTTPException:
        for exc_type, status in _ERROR_STATUS.items():
            if isinstance(exc, exc_type):
                return HTTPException(status_code=status, detail=str(exc))
        return HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/detector/probe", response_model=DetectorProbeResponse)
    async def probe_detector():
        """Validate the selected backend before the user hits Start, so a bad
        URL or missing key surfaces in the Admin Panel rather than as a failed
        capture."""
        backend = state.settings.detector_backend
        if backend == "native":
            import os

            model = state.settings.active_model
            found = os.path.exists(model)
            return DetectorProbeResponse(
                backend=backend,
                reachable=True,
                detail=(
                    f"{model} present."
                    if found
                    else f"{model} not on disk; ultralytics will download it on first start."
                ),
            )

        def _run_probe():
            import time as _time

            detector = state.detector_factory(state.settings, state.device)
            try:
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                started = _time.perf_counter()
                detector.infer(frame)
                elapsed = (_time.perf_counter() - started) * 1000.0
                return elapsed, sorted(str(v) for v in detector.names.values())
            finally:
                closer = getattr(detector, "close", None)
                if callable(closer):
                    closer()

        try:
            latency_ms, class_names = await run_in_threadpool(_run_probe)
        except RoboflowError as exc:
            return DetectorProbeResponse(
                backend=backend, reachable=False, detail=str(exc)
            )
        return DetectorProbeResponse(
            backend=backend,
            reachable=True,
            detail=f"Reached {backend_url(state.settings)}",
            latency_ms=round(latency_ms, 1),
            class_names=class_names,
        )

    @app.post("/api/capture/start")
    async def start():
        if state.state != "running":
            source = state.source_factory(state.settings)
            if hasattr(source, "open"):
                source.open()
            try:
                detector = state.detector_factory(state.settings, state.device)
            except RoboflowError as exc:
                closer = getattr(source, "close", None)
                if callable(closer):
                    closer()
                raise _http_from_roboflow(exc) from None
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
