import asyncio
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Callable
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from app.settings import Settings, resolve_device
from app.settings_store import (
    ALLOWED_MODELS,
    DEFAULT_SETTINGS_PATH,
    is_custom_model,
    resize_guess,
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
from app.roster import class_list_problems
from app.roboflow import (
    RoboflowAuthError,
    RoboflowError,
    RoboflowTimeout,
    RoboflowUnavailable,
    WorkflowClient,
)
from app.tracking import IouTracker
from app.dataset_status import load_dataset_status
from app.models import (
    MODELS_DIR,
    generation_for,
    imgsz_for,
    installed_models,
    record_requirement,
    requirement_for,
)
from app.schemas import (
    ApplyPresetRequest,
    CameraInfo,
    InferenceMessage,
    InferenceStatusPayload,
    CameraProfileResponse,
    CameraQualityResponse,
    CamerasResponse,
    DatasetStatusResponse,
    HealthResponse,
    ModelsResponse,
    LogEvent,
    LogsResponse,
    PresetInfo,
    RecordResizeModeRequest,
    DetectorProbeResponse,
    PresetsResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    StatusMessage,
    StoredProfileResponse,
    SystemInfoResponse,
    UnrecordedResizeMode,
)
from app.camera import CameraCapture
from app.camera_caps import CameraProfile, calibrate, device_key_for
from app.camera_profiles import load_profiles, save_profile
from app.cameras import CameraDevice, list_cameras, list_device_names, name_for_index
from app.inference import RoboflowRemoteDetector, YoloDetector
from app.inference_health import (
    OK,
    UNRESPONSIVE,
    InferenceHealthMonitor,
    InferenceStatus,
    local_server_command,
    monitor_for,
    probe_url,
)
from app.logging_store import LoggingStore


def _default_source_factory(settings: Settings):
    return CameraCapture(
        settings.camera_index, settings.capture_width,
        settings.capture_height, settings.capture_fps,
        brightness=settings.camera_brightness,
        exposure=settings.camera_exposure,
        autofocus=settings.camera_autofocus,
        focus=settings.camera_focus,
        auto_exposure=settings.camera_auto_exposure,
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


def _inference_payload(
    status: InferenceStatus, command: str | None = None
) -> InferenceStatusPayload:
    """The verdict's body, as both surfaces spell it.

    One builder for the stream message and the health read, so neither can describe the state, the
    endpoint or the failure differently from the other. The two differ in exactly one way, and
    deliberately: `age_seconds` is computed here, when the payload is built, so the pushed copy
    carries the age of the change and the polled copy carries the age of the most recent probe.

    `command` is passed in rather than resolved here: it is a fact about this machine's layout, not
    about the verdict, and the callers are the ones holding the app state that owns it.
    """
    return InferenceStatusPayload(
        backend=status.backend,
        url=status.url,
        state=status.state,
        detail=status.detail,
        local_server_command=command,
        age_seconds=status.age_seconds(),
    )


def _local_server_command(state: "AppState", status: InferenceStatus) -> str | None:
    """The remedy to go with a verdict, but only where this app is the one that can name it.

    Gated on `local_api` for the reason `retarget` blanks the URL off a native target: a payload
    carrying a command for a backend that does not use it would be the notice telling an operator to
    start a server this configuration never calls. The factory rather than the answer is held on the
    app state, so the file it reads is read per payload - a venv created while the app is running is
    then picked up by the notice that sent the operator to make it.
    """
    if status.backend != "local_api":
        return None
    return state.local_server_command_factory()


def _inference_read(state: "AppState", status: InferenceStatus) -> InferenceStatusPayload | None:
    """The verdict for a *read*, or None when there is no server to watch.

    `None` on a blank URL, which is how `retarget` spells "nothing to watch" (`native`, or a remote
    backend with no endpoint configured) - and the panel needs that told apart from `unknown`, which
    is a configured endpoint that has not been asked yet. The message path keeps the blank URL
    instead: a client receives it on the handshake and renders nothing for an `unknown` state
    either way, so there is nothing there for the absence to add.
    """
    if not status.url:
        return None
    return _inference_payload(status, _local_server_command(state, status))


def _current_inference_status(state: "AppState") -> InferenceStatus:
    """The freshest verdict there is: the running monitor's, else the last one stored.

    The two cannot disagree about the verdict - the monitor replaces its copy only when it reports
    the replacement, so the stored one is a prefix of the same history. They differ in exactly one
    field, which is why this exists: `checked_at` is re-stamped on *every* probe, including the ones
    that merely re-confirm a standing verdict and therefore tell no client anything. Reading the
    stored copy instead would date a server that has been down all afternoon from the moment it went
    down, when it was in fact checked seconds ago. The monitor is None only where nothing is
    watching (no lifespan, or after shutdown), and then the stored verdict is all there is.
    """
    monitor = state.inference_monitor
    if monitor is not None:
        return monitor.status
    return state.inference_health


def _inference_message(state: "AppState", status: InferenceStatus) -> InferenceMessage:
    """The one spelling of the message, used by the handshake and by every transition.

    A second construction at the handshake would be a second place to forget a field, and the
    handshake is exactly where a forgotten field stays invisible: the transition path is the one a
    test reaches by watching a server go down.
    """
    payload = _inference_payload(status, _local_server_command(state, status))
    return InferenceMessage(type="inference", **payload.model_dump())


def _report_inference_health(state: "AppState", status: InferenceStatus) -> None:
    """Store a verdict, tell every connected client, and write it to this process's log.

    Called on the monitor's thread for a verdict and on the request thread for a retarget, which is
    what `WSManager.submit` is for: it hands the message to the event loop whichever thread is
    calling.

    The print is for the reason `run.py` prints the loop it serves on: when a `local_api` capture
    detects nothing, the first question is whether the server was ever there, and this process's
    stdout - which the desktop forwards as `[sidecar] …` - is where that can be answered without
    instrumenting anything.
    """
    state.inference_health = status
    state.ws_manager.submit(_inference_message(state, status).model_dump())
    if status.state == UNRESPONSIVE:
        print(
            f"[sidecar] the {status.backend} server at {status.url} is not answering: "
            f"{status.detail}",
            flush=True,
        )
    elif status.state == OK:
        print(f"[sidecar] the {status.backend} server at {status.url} is answering", flush=True)
    elif status.url:
        # A retarget reports `unknown` (`inference_health` explains why), and a line naming the
        # endpoint now being watched is what turns that silence into something visible.
        print(f"[sidecar] watching the {status.backend} server at {status.url}", flush=True)


def _start_inference_watch(state: "AppState") -> None:
    """Start watching the endpoint the configured backend calls, if it calls one at all.

    The backend decides: `native` runs the weights in this process, so there is nothing to watch
    and no request is ever made - the monitor's `retarget` enforces that, for a remote backend with
    no URL configured too.
    """
    monitor = state.inference_monitor_factory(
        state.inference_probe,
        lambda status: _report_inference_health(state, status),
    )
    monitor.retarget(state.settings.detector_backend, backend_url(state.settings))
    monitor.start()
    state.inference_monitor = monitor


def _stop_inference_watch(state: "AppState") -> None:
    monitor, state.inference_monitor = state.inference_monitor, None
    if monitor is not None:
        monitor.stop()


def _retarget_inference_watch(state: "AppState") -> None:
    """Follow a settings change to the endpoint being watched (a no-op when it has not moved)."""
    monitor = state.inference_monitor
    if monitor is not None:
        monitor.retarget(state.settings.detector_backend, backend_url(state.settings))


def _default_detector_factory(settings: Settings, device: str):
    if settings.detector_backend == "native":
        if is_custom_model(settings.active_model) and not os.path.exists(
            settings.active_model
        ):
            # ultralytics would raise a bare FileNotFoundError here, or try to
            # fetch the name as a URL. A custom model never auto-downloads — it
            # is a file the operator must place — so fail with the action to
            # take instead of a 500 traceback. Reached at capture start and via
            # Test Connection; both surface the detail to the user.
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{settings.active_model} is not on disk — capture cannot start. "
                    "Copy the model file into sidecar/models/ (for the grocery model, "
                    "see docs/DETECTOR_BACKENDS.md §1a), then retry Test Connection."
                ),
            )
        # `requirement_for` is the record read (`models/<stem>.json`), which is what makes
        # `resize_mode: auto` correct for a locally trained model rather than merely
        # detectable as wrong. Read once here, at capture start, since resize_mode is
        # restart-required — the detector holds the resolved mode for its lifetime.
        return YoloDetector(
            settings.active_model, device=device,
            conf=settings.conf_threshold, imgsz=settings.imgsz,
            resize_mode=resolve_resize_mode(
                settings.resize_mode,
                settings.active_model,
                requirement_for(settings.active_model),
            ),
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
    settings_path: str = DEFAULT_SETTINGS_PATH
    source_factory: Callable = _default_source_factory
    detector_factory: Callable = _default_detector_factory
    ws_manager: WSManager = field(default_factory=WSManager)
    pipeline: Pipeline | None = None
    # Held so teardown can release them: the frame source owns an OpenCV device
    # and a thread, the remote detector an httpx connection pool.
    source: object | None = None
    detector: object | None = None
    state: str = "idle"
    # Why the last capture ended without being asked to, or None. Recorded when the pipeline
    # reports it and cleared when a new capture starts, because it exists for clients that were not
    # there: by the time one connects, `state` is `idle` again (teardown runs on the way
    # out) and nothing else on the wire distinguishes a capture that died from one that was never
    # started. See the handshake in the `stream` route.
    last_error: str | None = None
    # What is wrong with the *running* model's class list (`app/roster.py`), or empty. Stored for
    # the same reason `last_error` is: the client that started the capture gets this in its own
    # start response, but a renderer that connects *while* one runs - a reload, a second window -
    # has no other way to learn it, and this is a fact about the process rather than about the
    # request. Cleared with the rest of the runtime in `_teardown_capture`: nothing is loaded then,
    # so a stale warning would describe a model that is no longer there.
    class_warnings: list[str] = field(default_factory=list)
    # The running model's own class names, the input to that judgement and a readout in its own
    # right (`StatusMessage.class_names`). Stored and cleared with `class_warnings`, and for the
    # same reason - a reloaded renderer has to be able to show which model it is watching, and the
    # count is part of what the Live view's stats strip reports.
    class_names: list[str] = field(default_factory=list)
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
    # Whether the server the selected backend calls is answering (`app/inference_health.py`), and
    # the last thing a client was told about it. Stored for the same reason `class_warnings` is:
    # the monitor reports transitions, so a client that connects afterwards has no other way to
    # learn the current verdict - and the handshake replays this field rather than re-probing.
    inference_health: InferenceStatus = field(default_factory=InferenceStatus)
    # Injection seam, like `camera_lister`: a test supplies a probe that answers without a socket.
    inference_probe: Callable[[str], tuple[bool, str]] = probe_url
    # And the monitor itself, so a test can keep its interval and failure count out of the picture
    # (`monitor_for` is the app's wiring: which probe, which callback).
    inference_monitor_factory: Callable[..., InferenceHealthMonitor] = monitor_for
    inference_monitor: InferenceHealthMonitor | None = None
    # How an operator would start the local backend's server on this machine (`app/inference_health`),
    # which the notices render as the remedy. Read per payload rather than resolved once, because the
    # venv this names can be created while the app is running - and the notice that prompted that is
    # still on screen, so it has to stop sending the operator to the setup step the moment it exists.
    # `None` from the callable means this checkout has no such venv to run.
    local_server_command_factory: Callable[[], str | None] = local_server_command

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
        state.class_warnings = []
        state.class_names = []

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


def _models_response() -> ModelsResponse:
    """The selectable weights, as both `/api/models` and `POST /api/models/record` report them.

    One builder rather than two: the record's entire answer *is* this list, and a second
    construction of it could only drift from the one the panel reads on load.
    """
    return ModelsResponse(
        stock=sorted(ALLOWED_MODELS), installed=installed_models(), directory=str(MODELS_DIR)
    )


def _settings_response(state: "AppState") -> SettingsResponse:
    api_key_present = state.api_key_probe()
    # One lookup, two consumers: the warning check and the resolved-geometry readout below have to
    # agree about what these weights require, and a second `requirement_for` could only ever be a
    # second read of the same file.
    requirement = requirement_for(state.settings.active_model)
    # The other half of the same fact, read from the same file: `resize_mode` says how the frame
    # is fitted to the square, `imgsz` says how big the square is. Two lookups rather than one
    # because the two are independent - a weight may have recorded either, both, or neither - and
    # `compute_warnings` is where the second becomes a sentence.
    trained_imgsz = imgsz_for(state.settings.active_model)
    # The third consumer of the same requirement, and the reason it is a *structured* entry: this
    # case has a remedy the panel can perform (`POST /api/models/record`), so it carries the model
    # to write for, the mode to write, and the sentence explaining why. `compute_warnings` no
    # longer reports it — one situation, one place on screen, and the place that can answer it.
    guess = resize_guess(state.settings, requirement)
    return SettingsResponse(
        active_model=state.settings.active_model,
        camera_index=state.settings.camera_index,
        capture_width=state.settings.capture_width,
        capture_height=state.settings.capture_height,
        capture_fps=state.settings.capture_fps,
        conf_threshold=state.settings.conf_threshold,
        imgsz=state.settings.imgsz,
        resize_mode=state.settings.resize_mode,
        # Native only, because only the native branch resizes the frame with these weights: a
        # remote backend sends it to a workflow that holds its own model. Resolved from the same
        # requirement this function already read, so the readout and the warning cannot disagree
        # about what these weights need — and it is the same call `_default_detector_factory`
        # makes, so what the Live view prints is the geometry the detector was built with.
        resize_mode_resolved=(
            resolve_resize_mode(
                state.settings.resize_mode, state.settings.active_model, requirement
            )
            if state.settings.detector_backend == "native"
            else None
        ),
        infer_frame_skip=state.settings.infer_frame_skip,
        device=state.settings.device,
        preview_height=state.settings.preview_height,
        preview_max_fps=state.settings.preview_max_fps,
        preview_mirror=state.settings.preview_mirror,
        suppress_clamped_detections=state.settings.suppress_clamped_detections,
        suppress_frame_filling_detections=(
            state.settings.suppress_frame_filling_detections
        ),
        suppress_unsure_phantoms=state.settings.suppress_unsure_phantoms,
        track_expiry_s=state.settings.track_expiry_s,
        class_allowlist=list(state.settings.class_allowlist),
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
        camera_auto_exposure=state.settings.camera_auto_exposure,
        hot_reloadable_fields=sorted(HOT_RELOADABLE_FIELDS),
        restart_required_fields=sorted(RESTART_REQUIRED_FIELDS),
        warnings=compute_warnings(
            state.settings, state.state, api_key_present, requirement, trained_imgsz
        ),
        unrecorded_resize_mode=(
            UnrecordedResizeMode(
                model=state.settings.active_model,
                resize_mode=guess.mode,
                warning=guess.warning,
                remedy=guess.remedy,
            )
            if guess is not None
            else None
        ),
        roboflow_api_key_present=api_key_present,
    )


# settings key -> CameraCapture.set_controls keyword.
_CAMERA_CONTROL_KEYS = {
    "camera_brightness": "brightness",
    "camera_exposure": "exposure",
    "camera_autofocus": "autofocus",
    "camera_focus": "focus",
    "camera_auto_exposure": "auto_exposure",
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
    # Not a live setting in the `_push_live_settings` sense - no running pipeline picks it up - but
    # the monitor is watching a *URL*, and a patch that moved the backend or its URL has to move
    # what is being watched or the notice would describe a server this app no longer calls.
    _retarget_inference_watch(state)
    if persist:
        save_settings(state.settings, state.settings_path)
    return _settings_response(state)


def build_app(state_factory: Callable[[], AppState] = AppState) -> FastAPI:
    state = state_factory()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Start the one thing here that watches a *foreign* process, and stop it on the way out.

        The monitor is the only work in this app that runs without a request or a capture to drive
        it, and it is deliberately not started lazily on the first WebSocket connection: an
        inference server that was never started is the ordinary `local_api` failure, and the
        operator should be able to see that before they press Start - with the window closed, from
        the process's own log.

        `TestClient(app)` without a context manager does not run this, which is what keeps the rest
        of the suite from making requests on a timer; the tests that are *about* the watch use
        `with TestClient(app)` and an injected probe.
        """
        _start_inference_watch(state)
        try:
            yield
        finally:
            _stop_inference_watch(state)

    app = FastAPI(title="SCANnCART Sidecar", lifespan=lifespan)
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
            # Read here rather than pushed, because this endpoint is polled: the age of a standing
            # verdict is only honest if it is recomputed, and a client has no way to know the
            # monitor went on probing after the message it last received.
            inference=_inference_read(state, _current_inference_status(state)),
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
            model = state.settings.active_model
            if not os.path.exists(model):
                # Keep the old cheap answer for the missing case: the probe
                # must never trigger a download or a network hit. Stock names
                # (yolo11n.pt…) auto-download on first start; a custom model
                # under models/ never does, so say where it goes instead.
                if is_custom_model(model):
                    detail = (
                        f"{model} is not on disk. Copy the model file into sidecar/models/ "
                        "(the Roboflow-exported grocery ONNX — see "
                        "docs/DETECTOR_BACKENDS.md §1a), then probe again. Ultralytics "
                        "only auto-downloads its own stock weights."
                    )
                else:
                    detail = f"{model} not on disk; ultralytics will download it on first start."
                return DetectorProbeResponse(
                    backend=backend,
                    reachable=True,
                    detail=detail,
                )

            def _run_native_probe():
                import time as _time

                detector = state.detector_factory(state.settings, state.device)
                try:
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)
                    # Warm the session first: the first infer builds the ONNX
                    # session / loads the weights (~7 s for ultralytics, more
                    # for a CUDA session), which is exactly the cold hit a
                    # pre-start probe exists to move out of capture start.
                    detector.infer(frame)
                    started = _time.perf_counter()
                    detector.infer(frame)
                    elapsed = (_time.perf_counter() - started) * 1000.0
                    return (
                        elapsed,
                        sorted(str(v) for v in detector.names.values()),
                        getattr(detector, "provider", None),
                    )
                finally:
                    closer = getattr(detector, "close", None)
                    if callable(closer):
                        closer()

            try:
                latency_ms, class_names, provider = await run_in_threadpool(
                    _run_native_probe
                )
            except Exception as exc:  # noqa: BLE001 - a load failure is a probe failure
                return DetectorProbeResponse(
                    backend=backend, reachable=False, detail=str(exc)
                )
            detail = f"{model} loaded."
            if provider:
                detail += f" Inference on {provider}."
            return DetectorProbeResponse(
                backend=backend,
                reachable=True,
                detail=detail,
                latency_ms=round(latency_ms, 1),
                class_names=class_names,
                # Judged here rather than in the renderer: the names came off the loaded model in
                # this process, so this is the only place they are known at all. The record's
                # generation goes with them for the same reason `requirement_for` does - these are
                # the local weights, whose record is the only thing that can say whether a missing
                # class is a fault (a v2 head) or the correct list for the generation (v1's).
                class_warnings=class_list_problems(
                    class_names, generation_for(state.settings.active_model)
                ),
                provider=provider,
            )

        def _run_probe():
            import time as _time

            detector = state.detector_factory(state.settings, state.device)
            try:
                # Shaped like a real capture, not a 64x64 square: the geometry is half of what this
                # probe is for, and a square frame cannot show an aspect mismatch at all. The
                # detector downscales it by the same rule capture uses, so the size reported below
                # is the size a live frame would actually be sent at.
                frame = np.zeros(
                    (state.settings.capture_height, state.settings.capture_width, 3),
                    dtype=np.uint8,
                )
                started = _time.perf_counter()
                detector.infer(frame)
                elapsed = (_time.perf_counter() - started) * 1000.0
                # A record of the round trip that just happened. Only the remote detectors keep
                # one — the native path's geometry is a settings fact (`resize_mode_resolved`)
                # rather than something a call has to discover — so a detector without it leaves
                # both fields null, which is the honest answer for a backend that never transmits.
                geometry = getattr(detector, "last_geometry", None)
                return (
                    elapsed,
                    sorted(str(v) for v in detector.names.values()),
                    geometry.sent if geometry else None,
                    geometry.reported if geometry else None,
                )
            finally:
                closer = getattr(detector, "close", None)
                if callable(closer):
                    closer()

        try:
            latency_ms, class_names, sent_size, reported_size = await run_in_threadpool(
                _run_probe
            )
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
            # Same check as the native branch, on the classes the workflow actually reports - a
            # remote workflow's model is the one that can be swapped without touching this app,
            # so this is the branch where the roster is least under our control.
            #
            # No generation here, unlike the native branch: a record describes the `.pt` beside
            # it, and under a remote backend no such file is what running. Handing a workflow a
            # local weight's generation would hold it to a roster nothing about it claims, so the
            # classes it reports are left to name their own roster (`resolve_roster`).
            class_warnings=class_list_problems(class_names),
            sent_size=sent_size,
            reported_size=reported_size,
        )

    # One start at a time. Health reads `idle` for the whole of a start that is still opening the
    # camera (seconds on a StreamCam), and the desktop's POS loop - which keeps capture running -
    # calls start whenever it sees `idle`. Without this a second start passed the `!= "running"`
    # check, acquired a second camera handle, detector and pipeline beside the first, and when one
    # of them failed its error teardown closed the *other* pipeline's detector mid-inference
    # ("'NoneType' object has no attribute 'names'"). Under the lock the second caller waits for
    # the first and then sees `running`, so it acquires nothing.
    start_lock = asyncio.Lock()

    @app.post("/api/capture/start")
    async def start():
        async with start_lock:
            return await _start_capture()

    async def _start_capture():
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
                WebSocket could not even complete its handshake.

                The detector factory (which imports ultralytics, ~7 s) is
                started in a background thread *before* the camera opens so
                the two costs overlap instead of stacking. Overlapping them
                means either half can fail while the other is still building,
                so both are resolved before anything is returned and whichever
                one succeeded is released — otherwise a camera that fails to
                open strands a fully loaded model, holding VRAM until GC."""
                with ThreadPoolExecutor(max_workers=1) as pool:
                    detector_fut = pool.submit(
                        state.detector_factory, state.settings, state.device,
                    )
                    source: object | None = None
                    source_exc: BaseException | None = None
                    try:
                        source = state.source_factory(state.settings)
                        if hasattr(source, "open") and source.open() is False:
                            # CameraCapture.open() returns False — it does not
                            # raise — when the device cannot be opened at all
                            # (unplugged, claimed by another app, or a camera
                            # index that points at nothing). Treat that as the
                            # same failure as a raised exception: an actionable
                            # error instead of reporting "running" with a feed
                            # that never delivers a frame.
                            source_exc = HTTPException(
                                status_code=503,
                                detail=(
                                    f"Camera {state.settings.camera_index} could not be "
                                    "opened. Check it is plugged in and not in use by "
                                    "another app; if you have several cameras, try a "
                                    "different camera_index in Admin → Settings."
                                ),
                            )
                    except BaseException as exc:  # noqa: BLE001 - re-raised below
                        source_exc = exc
                    # Always resolved, never abandoned: the pool's shutdown
                    # waits for this call anyway, so skipping result() would
                    # only lose the object, not the cost of building it.
                    try:
                        detector = detector_fut.result()
                    except BaseException:
                        if source is not None:
                            # `source.open()` already started the capture
                            # thread, and frame sources expose release(),
                            # not close().
                            _release(source, "release", "close")
                        raise
                    if source_exc is not None:
                        _release(detector, "close")
                        raise source_exc
                    return source, detector

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
                detail = f"Capture stopped: {exc}"
                # Kept as well as broadcast, and kept *here* rather than in teardown: this status
                # explains the death to whoever is connected at the time, and only a stored copy
                # can explain it to a client that connects afterwards - the state alone cannot,
                # since teardown leaves `idle` behind either way.
                state.last_error = detail
                state.ws_manager.submit(
                    StatusMessage(type="status", state="error", detail=detail).model_dump()
                )
                _teardown_capture(state, join_thread=False)

            def _on_class_list(names: list[str]) -> None:
                """Called from the pipeline thread the first time the model's classes are knowable.

                Reported rather than merely stored, because the point is that a 24-class weight
                cannot run *unnoticed*: a client already watching gets a status message, and the
                handshake covers whoever connects later.

                Broadcast **unconditionally** now, which reverses an earlier rule ("a clean list is
                not news"). It was not news while the warning banner was the only consumer - an
                empty `class_warnings` said as much - but the names are also a readout: the Live
                view's stats strip shows how many classes the running model has, and the count of a
                *clean* list is exactly the number an operator wants to see beside `24`. Silence
                here would leave the chip absent on every healthy capture. Once per capture, so
                the cost is one small message.
                """
                state.class_names = list(names)
                state.class_warnings = class_list_problems(
                    names,
                    # Which roster these are judged against, on the same terms as the probe: the
                    # local weights' own generation, and nothing when a workflow is what runs.
                    generation_for(state.settings.active_model)
                    if state.settings.detector_backend == "native"
                    else None,
                )
                state.ws_manager.submit(
                    StatusMessage(
                        type="status",
                        state="running",
                        class_names=list(state.class_names),
                        class_warnings=list(state.class_warnings),
                    ).model_dump()
                )

            state.pipeline = Pipeline(
                source, detector, state.settings,
                on_message=state.ws_manager.submit,
                logging_store=state.logging_store,
                session_id=state.session_id,
                on_error=_on_pipeline_error,
                on_class_list=_on_class_list,
            )
            # A capture that is starting now supersedes the last one's obituary: the stored reason
            # describes something that is no longer the case, and leaving it would put the previous
            # failure's explanation over a working capture the moment anyone reloads.
            #
            # Cleared *before* the thread starts rather than after. The error handler runs on that
            # thread, so a detector that fails on its first frame can record its own reason before
            # the next two lines run — and clearing afterwards would erase the explanation for the
            # capture that just died. A start that never gets this far (a bad API key, an
            # unopenable device) raises in `_acquire` above and deliberately leaves the stored
            # reason alone, since that failure is reported in its own response.
            state.last_error = None
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

    @app.get("/api/models", response_model=ModelsResponse)
    async def models():
        # Answers "what weights could I select, and what do they need?". It reads the
        # models/ directory rather than asking the detector, because the question is about
        # what exists, not what loads - probing each candidate would import torch per file.
        # Off the event loop like the snapshot read, so a slow disk cannot stall the health
        # poll. The per-model records it picks up are the reason the Admin Panel can flag a
        # resize_mode mismatch instead of leaving it to be remembered.
        return await run_in_threadpool(_models_response)

    @app.post("/api/models/record", response_model=ModelsResponse)
    async def record_model_requirement(body: RecordResizeModeRequest):
        """Write the geometry these weights were trained with into the record beside them.

        The Admin Panel's one-click remedy for SettingsResponse.unrecorded_resize_mode: the
        operator attests to a fact only they hold (what these weights were trained on), and
        `auto` stops guessing for them. The other writer of these files is
        `tools/train_model.py --install`, which knows the requirement from the training run; this
        route exists for the weights that path never installed.

        Allowed while capture runs, unlike the restart-required settings: what it writes is by
        construction the mode `auto` had already resolved to (`unrecorded_resize_mode` is only
        produced for that case), so the geometry the running detector was built with is
        unchanged and the readout cannot start disagreeing with the detector. Refusing here would
        remove the button at the one moment an operator is looking at the warning.

        The answer is the refreshed weights list, because that list is where the change shows -
        these weights gain a requirement and their `auto_resolves_to` becomes it. Returning the
        list rather than an acknowledgement means the panel reads the result instead of
        predicting it.
        """
        try:
            await run_in_threadpool(record_requirement, body.model, body.resize_mode)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No weights at {body.model} to record a requirement beside; nothing was "
                    "written. Pick a model that is on disk."
                ),
            )
        return await run_in_threadpool(_models_response)

    @app.get("/api/dataset/status", response_model=DatasetStatusResponse)
    async def dataset_status():
        # Reads one small local JSON file - no network, no credentials, no AppState.
        # Still off the event loop, so a slow disk cannot stall the /api/health poll
        # the renderer runs on a timer.
        return DatasetStatusResponse(**asdict(await run_in_threadpool(load_dataset_status)))

    @app.get("/api/logs", response_model=LogsResponse)
    async def logs(since: float | None = None):
        # `since` (sidecar wall-clock seconds, the same clock as `entered_at`/`left_at`) drops tracks
        # that had already left by then. Omitted, the whole current session is returned, which is
        # what the renderer's reconnect recovery wants; the POS integration's poller passes its bind
        # time so an all-day capture is not re-read in full every second.
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
            for r in state.logging_store.query_events(sid, since=since)
        ]
        return LogsResponse(session_id=sid, events=events)

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        await state.ws_manager.connect(ws)
        # Tell the client what it is looking at before it has to ask. Until this existed the only
        # status messages any client received were its own start/stop responses and pipeline
        # errors, so a renderer connecting to an *already running* capture - a reload, a second
        # window - sat on its default "idle": a toolbar offering Start over streaming frames, and
        # the renderer's /api/logs recovery, which is gated on "running", never firing. Read from
        # the same field /api/health reports, at handshake time.
        #
        # Built before the `try` so a bug constructing it raises here rather than being swallowed
        # as a dead client.
        # The stored reason the last capture ended, when there is one. This is the only field that
        # can tell a client joining *after* a capture died what happened to it: `state` is `idle`
        # there - that is what the sidecar is - so without the detail a reloaded renderer sat in
        # front of a frozen preview with no explanation. Empty in every ordinary case (never
        # started, still running, stopped by hand), which is the client's cue that there is nothing
        # to explain.
        # `class_names`/`class_warnings` ride along for the same reason the detail does: a client
        # joining a capture already in progress cannot observe the moment the class list became
        # known, and "the model you are running predicts 24 classes" is exactly the kind of thing a
        # reloaded renderer must not have to guess from the labels going past. The names travel with
        # the verdict, not instead of it, so the chip in the stats strip is populated on a reload
        # for a *clean* model too - where no warning is ever broadcast and the count is the point.
        snapshot = StatusMessage(
            type="status",
            state=state.state,
            detail=state.last_error or "",
            class_names=list(state.class_names),
            class_warnings=list(state.class_warnings),
        ).model_dump()
        # ...and whether the server its detections would come from is answering. Sent on every
        # connect and then only on a change, so this is the client's read of a state that is kept
        # by a monitor running in the background: without it, a window opened while `local_api`'s
        # server is down would show a live preview and no detections, with nothing on screen saying
        # why. `unknown` is the ordinary answer for `native`, and a client renders nothing for it.
        try:
            await ws.send_json(snapshot)
            await ws.send_json(
                _inference_message(state, _current_inference_status(state)).model_dump()
            )
            while True:
                await ws.receive_text()
        except Exception:  # noqa: BLE001
            # `_drain`'s rule, for `_drain`'s reason: any failure on a socket in this route means
            # the client is gone - Starlette answers RuntimeError once it is closed, uvicorn can
            # raise its own disconnect type - and which one it was is not worth a traceback.
            state.ws_manager.disconnect(ws)

    return app
