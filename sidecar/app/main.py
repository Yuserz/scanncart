import asyncio
import os
import queue
import threading

import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Callable
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from app.settings import Settings, resolve_device
from app.settings_store import (
    is_custom_model,
    resolve_resize_mode,
    HOT_RELOADABLE_FIELDS,
    RESTART_REQUIRED_FIELDS,
    compute_warnings,
    load_settings,
    save_settings,
)
from app.camera_quality import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    SHARPNESS_MIN,
    frame_quality,
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
    CameraInfo,
    CameraProfileResponse,
    CameraQualityResponse,
    CamerasResponse,
    HealthResponse,
    LogEvent,
    LogsResponse,
    PresetInfo,
    DetectorProbeResponse,
    PresetsResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    StatusMessage,
    StoredProfileResponse,
    SystemInfoResponse,
)
from app.camera import CameraCapture
from app.camera_caps import CameraProfile, calibrate, device_key_for
from app.camera_profiles import load_profiles, save_profile
from app.cameras import CameraDevice, list_cameras, list_device_names, name_for_index
from app.inference import RoboflowRemoteDetector, YoloDetector
from app.logging_store import LoggingStore


def _default_source_factory(settings: Settings):
    return CameraCapture(
        settings.camera_index, settings.capture_width,
        settings.capture_height, settings.capture_fps,
        brightness=settings.camera_brightness,
        exposure=settings.camera_exposure,
        autofocus=settings.camera_autofocus,
        focus=settings.camera_focus,
    )


def _resolve_camera_name(state: "AppState") -> str:
    """Best-effort device name for the calibration device_key.

    Deliberately re-queries `state.camera_namer()` rather than reading
    `state.cameras`: the cached list is None until /api/cameras has been
    called at least once (calibration must work without that ever
    happening), and even when populated it can be stale relative to what is
    plugged in right now. The ~550 ms PowerShell call is cheap next to the
    seconds calibration already takes to sample frames.

    Uses the same positional index -> name convention as list_cameras (via
    name_for_index) rather than a second one, and never raises: an empty or
    "Camera N" fallback device_key is far better than a failed calibration.
    """
    try:
        names = state.camera_namer()
        return name_for_index(state.settings.camera_index, names)
    except Exception:  # noqa: BLE001 - naming must never block calibration
        return f"Camera {state.settings.camera_index}"


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
            resize_mode=resolve_resize_mode(settings.resize_mode, settings.active_model),
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
        # stable track ids. Read expiry live rather than snapshotting it:
        # track_expiry_s is hot-reloadable and Pipeline re-reads it every call,
        # so a snapshot here desynchronised the two as soon as an operator
        # raised it mid-capture — which the Admin Panel actively tells them to
        # do for a remote backend.
        tracker=IouTracker(
            expiry_s=settings.track_expiry_s,
            expiry_provider=lambda: settings.track_expiry_s,
        ),
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
    # Held so teardown can release them: the frame source owns an OpenCV device
    # and a thread, the remote detector an httpx connection pool.
    source: object | None = None
    detector: object | None = None
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
    # And for camera enumeration, which shells out to PowerShell and opens
    # every device. Cached because probing is slow and cannot run while
    # capture holds the camera.
    camera_lister: Callable[[], list[CameraDevice]] = list_cameras
    cameras: list[CameraDevice] | None = None
    # Windows' device names at the time of the last scan. Re-reading them costs
    # ~550 ms (one PowerShell call, opens nothing) against ~30 s for a full
    # scan, so it is the cheap way to notice a camera plugged in after startup.
    camera_namer: Callable[[], list[str]] = list_device_names
    camera_signature: list[str] | None = None
    # Serializes capture teardown between the HTTP handler and the pipeline
    # thread's error handler.
    teardown_lock: threading.Lock = field(default_factory=threading.Lock)
    # Injection seam, like camera_lister: tests supply a profile instead of
    # opening a device. calibrate() itself is Task 11 — this route only
    # measures-and-reports, never applies, so tests can exercise the review
    # step without a real camera.
    calibrator: Callable[[], CameraProfile] | None = None
    last_profile: CameraProfile | None = None
    # Marks the camera as exclusively held by an in-flight calibration (~80s).
    # `state.state == "running"` already refuses calibration during capture,
    # but nothing said the reverse until this: without it, /api/capture/start
    # and /api/cameras (its rescan path) could open or probe the same device
    # a calibration is mid-measurement on. Set for the duration of the
    # calibrate request and always cleared in a finally, so an exception
    # cannot strand it true.
    calibrating: bool = False

    def __post_init__(self):
        if self.settings is None:
            self.settings = load_settings(self.settings_path)
        if not self.device:
            self.device = resolve_device(self.settings.device)
        if self.logging_store is None:
            self.logging_store = LoggingStore(self.db_path)
        if self.calibrator is None:
            self.calibrator = lambda: calibrate(
                self.settings.camera_index,
                self.settings.capture_width,
                self.settings.capture_height,
                device_name=_resolve_camera_name(self),
                # Gates the exposure recommendation relative to what the
                # operator actually configured, not an absolute floor — see
                # camera_derive.derive_camera_settings.
                target_fps=self.settings.capture_fps,
            )


def _release(obj: object, *names: str) -> None:
    """Call the first of `names` that exists. Frame sources expose `release()`
    and detectors `close()`, and neither is guaranteed — a `FakeFrameSource` in
    a test has no pool to free."""
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            return


def _teardown_capture(state: "AppState", join_thread: bool = True) -> None:
    """Return to idle and free every resource capture acquired.

    Two callers race here: the HTTP stop handler and the pipeline thread's own
    error handler. Claim the resources under a lock and clear them in one step,
    so whichever arrives second gets Nones and no-ops instead of tripping over
    half-torn-down state (it used to raise AttributeError on a None pipeline).

    The join happens *outside* the lock deliberately. The error handler runs on
    the pipeline thread, so holding the lock across a join would deadlock: the
    HTTP caller would wait on a thread that is itself waiting for the lock.
    `join_thread=False` is that in-thread path — it must never join itself.
    """
    with state.teardown_lock:
        pipeline = state.pipeline
        source = state.source
        detector = state.detector
        session_id = state.session_id
        state.pipeline = None
        state.source = None
        state.detector = None
        state.session_id = None
        state.state = "idle"

    if pipeline is not None:
        if join_thread:
            pipeline.stop()
        else:
            pipeline.is_running = False
        pipeline.resolve_open_tracks()
    # Order matters: the detector's client can be mid-request until the thread
    # is done, so release only after the pipeline has stopped.
    if detector is not None:
        _release(detector, "close")
    if source is not None:
        _release(source, "release", "close")
    if session_id is not None:
        state.logging_store.end_session(session_id)


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
        resize_mode=state.settings.resize_mode,
        infer_frame_skip=state.settings.infer_frame_skip,
        device=state.settings.device,
        preview_height=state.settings.preview_height,
        preview_max_fps=state.settings.preview_max_fps,
        track_expiry_s=state.settings.track_expiry_s,
        detector_backend=state.settings.detector_backend,
        roboflow_workspace=state.settings.roboflow_workspace,
        roboflow_workflow_id=state.settings.roboflow_workflow_id,
        local_api_url=state.settings.local_api_url,
        cloud_api_url=state.settings.cloud_api_url,
        remote_infer_size=state.settings.remote_infer_size,
        remote_timeout_s=state.settings.remote_timeout_s,
        remote_max_retries=state.settings.remote_max_retries,
        camera_brightness=state.settings.camera_brightness,
        camera_exposure=state.settings.camera_exposure,
        camera_autofocus=state.settings.camera_autofocus,
        camera_focus=state.settings.camera_focus,
        hot_reloadable_fields=sorted(HOT_RELOADABLE_FIELDS),
        restart_required_fields=sorted(RESTART_REQUIRED_FIELDS),
        warnings=compute_warnings(state.settings, state.state, api_key_present),
        roboflow_api_key_present=api_key_present,
    )


# settings key -> CameraCapture.set_controls keyword.
_CAMERA_CONTROL_KEYS = {
    "camera_brightness": "brightness",
    "camera_exposure": "exposure",
    "camera_autofocus": "autofocus",
    "camera_focus": "focus",
}


def _push_live_settings(state: "AppState", patch: dict) -> None:
    """Hand hot-reloadable changes to the objects that already exist.

    Pipeline re-reads infer_frame_skip/preview_*/track_expiry_s from settings
    itself, but the camera and detector hold their own copies, so those two
    need telling. Both lookups go through getattr: with capture stopped there
    is no source or detector at all, and a source need not implement
    set_controls (FakeFrameSource does not).
    """
    controls = {
        kw: patch[key] for key, kw in _CAMERA_CONTROL_KEYS.items() if key in patch
    }
    if controls:
        set_controls = getattr(state.source, "set_controls", None)
        if callable(set_controls):
            set_controls(**controls)
    if "conf_threshold" in patch:
        set_conf = getattr(state.detector, "set_conf", None)
        if callable(set_conf):
            set_conf(patch["conf_threshold"])


def _apply_settings_patch(
    state: "AppState", patch: dict, persist: bool = True
) -> SettingsResponse:
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
    _push_live_settings(state, patch)
    if persist:
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
    async def update_settings(body: SettingsUpdateRequest, persist: bool = True):
        """persist=false applies the change without writing settings.json.

        The Live tab's tuning card uses it so a slider drag reaches the camera
        immediately without every intermediate value becoming the config the
        app boots with. POST /api/settings/save commits what is in memory.
        """
        patch = body.model_dump(exclude_none=True)
        # exclude_none drops nulls, so "set this back to null" has to travel
        # as an explicit list of names — see SettingsUpdateRequest.reset_fields.
        for name in patch.pop("reset_fields", []):
            patch[name] = None
        return _apply_settings_patch(state, patch, persist=persist)

    @app.post("/api/settings/save", response_model=SettingsResponse)
    async def save_current_settings():
        """Persist the in-memory settings, including anything applied with
        persist=false. Writes the whole Settings object — see the design
        doc's 'Save is global' tradeoff."""
        save_settings(state.settings, state.settings_path)
        return _settings_response(state)

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
        patch = dict(preset.settings)
        # Presets pick a stock model size for the machine (yolo11n/s/m). That
        # is meaningless for a custom model — there is only the one — and
        # applying it would silently swap the grocery model for generic COCO
        # weights, which is the whole point of the app. Keep the custom model
        # and let the preset tune everything else.
        if is_custom_model(state.settings.active_model):
            patch.pop("active_model", None)
        return _apply_settings_patch(state, patch)

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

    @app.get("/api/cameras", response_model=CamerasResponse)
    async def get_cameras(rescan: bool = False):
        """Enumerate capture devices so the Admin Panel can show names rather
        than bare indices.

        Scanning opens every device, which measured ~30 s under contention, so
        the result is cached and only re-scanned when `rescan=true`. It is also
        refused outright while capture holds a device.
        """

        def _response(probed: bool, detail: str) -> CamerasResponse:
            return CamerasResponse(
                cameras=[CameraInfo(**vars(c)) for c in (state.cameras or [])],
                probed=probed,
                detail=detail,
            )

        if state.state == "running":
            return _response(False, "Capture is running — stop it to rescan for cameras.")

        if state.calibrating:
            # A rescan opens every device in turn; hitting the one calibration
            # is mid-measurement on used to break early (probe_index fails
            # closed) and overwrite state.cameras with a truncated list.
            return _response(False, "Calibration is in progress — wait for it to finish to rescan.")

        if state.cameras is not None and not rescan:
            # Cheap hotplug check: compare Windows' device names against the
            # ones present at the last scan. Without this a camera plugged in
            # after startup stayed invisible until someone pressed Rescan,
            # because every caller got the cache back.
            try:
                current = await run_in_threadpool(state.camera_namer)
            except Exception:  # noqa: BLE001 - never fail the request over this
                current = state.camera_signature
            if current == state.camera_signature:
                return _response(False, f"{len(state.cameras)} camera(s), from the last scan.")

        state.camera_signature = await run_in_threadpool(state.camera_namer)
        state.cameras = await run_in_threadpool(state.camera_lister)
        return _response(True, f"Found {len(state.cameras)} camera(s).")

    @app.get("/api/camera/quality", response_model=CameraQualityResponse)
    async def camera_quality():
        """Live image metrics. Reads the pipeline's newest frame rather than
        opening the device, so it works while capture holds it."""
        source = state.source
        if state.state != "running" or source is None:
            return CameraQualityResponse(
                available=False, detail="Start capture to measure the image."
            )
        got = source.latest()
        if got is None:
            return CameraQualityResponse(available=False, detail="No frame yet.")

        q = await run_in_threadpool(frame_quality, got[1])
        fps = float(getattr(source, "measured_fps", 0.0))
        target_fps = float(state.settings.capture_fps)
        return CameraQualityResponse(
            available=True,
            brightness=round(q.brightness, 1),
            contrast=round(q.contrast, 1),
            sharpness=round(q.sharpness, 1),
            capture_fps=round(fps, 1),
            target_fps=target_fps,
            verdicts={
                "brightness": "low" if q.brightness < BRIGHTNESS_MIN
                else "high" if q.brightness > BRIGHTNESS_MAX else "ok",
                "sharpness": "low" if q.sharpness < SHARPNESS_MIN else "ok",
                # Relative to what was actually requested, not a fixed
                # threshold: capture_fps is user-configurable (1-120) and the
                # shipped low_end preset asks for 15, so a hardcoded minimum
                # falsely flagged hardware hitting exactly its own target.
                "capture_fps": "low" if fps < target_fps * 0.8 else "ok",
            },
            detail="",
        )

    def _profiles_path() -> str:
        # Sibling of settings_path rather than camera_profiles.py's hardcoded
        # default, so tests pointing settings_path at tmp_path never touch the
        # real data/ directory.
        return os.path.join(
            os.path.dirname(state.settings_path) or ".", "camera_profiles.json"
        )

    @app.get("/api/camera/profile", response_model=StoredProfileResponse)
    async def get_camera_profile():
        """The stored calibration for the camera currently configured.

        This is what tells the tuning card which controls the device honours,
        and it is why a calibration survives an app restart.
        """
        key = device_key_for(
            _resolve_camera_name(state),
            state.settings.camera_index,
            state.settings.capture_width,
            state.settings.capture_height,
        )
        profile = load_profiles(_profiles_path()).get(key)
        if profile is None:
            return StoredProfileResponse(profile=None)
        return StoredProfileResponse(profile=CameraProfileResponse(**asdict(profile)))

    @app.post("/api/camera/calibrate", response_model=CameraProfileResponse)
    async def calibrate_camera():
        """Measure the camera and return a recommendation. Applies nothing —
        the operator reviews it first (review-first design)."""
        if state.state == "running":
            raise HTTPException(
                status_code=409,
                detail="Stop capture before calibrating; the camera is exclusive.",
            )
        if state.calibrating:
            raise HTTPException(
                status_code=409,
                detail="Calibration is already in progress for this camera.",
            )
        if state.calibrator is None:
            raise HTTPException(status_code=503, detail="No calibrator configured.")
        state.calibrating = True
        try:
            profile = await run_in_threadpool(state.calibrator)
        finally:
            # Always cleared, even on a raised exception, so a failed
            # calibration cannot strand the camera permanently exclusive.
            state.calibrating = False
        state.last_profile = profile
        save_profile(profile, _profiles_path())
        return CameraProfileResponse(**asdict(profile))

    @app.post("/api/camera/profile/apply", response_model=SettingsResponse)
    async def apply_camera_profile():
        if state.last_profile is None:
            raise HTTPException(status_code=404, detail="Calibrate the camera first.")
        return _apply_settings_patch(state, state.last_profile.recommended)

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
        if state.calibrating:
            # Calibration holds the device exclusively for ~80s (camera_caps.
            # calibrate). Starting capture underneath it would open the same
            # device twice: capture fails, or calibration measures a
            # contended device and writes a bogus profile to disk.
            raise HTTPException(
                status_code=409,
                detail="Calibration is in progress; the camera is exclusive. Wait for it to finish.",
            )
        if state.state != "running":
            def _acquire():
                """Opening a camera blocks — measured 43.5 s for a Logitech
                StreamCam (~9.5 s to open plus ~18.7 s for the 1080p mode-set,
                plus detector setup). Inline on the event loop that froze the
                whole sidecar: /api/health stopped answering and the renderer's
                WebSocket could not even complete its handshake."""
                source = state.source_factory(state.settings)
                if hasattr(source, "open"):
                    source.open()
                try:
                    return source, state.detector_factory(state.settings, state.device)
                except RoboflowError:
                    # `source.open()` already started the capture thread, and
                    # frame sources expose release(), not close() — the old
                    # getattr(source, "close") never matched, so every failed
                    # start leaked the camera device and its thread.
                    _release(source, "release", "close")
                    raise

            try:
                source, detector = await run_in_threadpool(_acquire)
            except RoboflowError as exc:
                raise _http_from_roboflow(exc) from None
            state.session_id = state.logging_store.start_session(
                state.settings.active_model, state.device
            )
            state.source = source
            state.detector = detector

            def _on_pipeline_error(exc: Exception) -> None:
                # Called on the pipeline thread, so teardown must not join it.
                state.ws_manager.submit(
                    StatusMessage(
                        type="status",
                        state="error",
                        detail=f"Capture stopped: {exc}",
                    ).model_dump()
                )
                _teardown_capture(state, join_thread=False)

            state.pipeline = Pipeline(
                source, detector, state.settings,
                on_message=state.ws_manager.submit,
                logging_store=state.logging_store,
                session_id=state.session_id,
                on_error=_on_pipeline_error,
            )
            state.pipeline.start()
            state.state = "running"
        return {"state": state.state}

    @app.post("/api/capture/stop")
    async def stop():
        # Signal first, join second. Ending the loop has to happen inline or
        # the thread keeps inferring for however long the threadpool dispatch
        # takes; the join then goes off the event loop because it can sit in a
        # remote retry for timeout*(retries+1) — ~15.6 s on the defaults —
        # which inline would freeze health polling and every WS send.
        if state.pipeline is not None:
            state.pipeline.signal_stop()
        await run_in_threadpool(_teardown_capture, state)
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
