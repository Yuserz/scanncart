import json
import os
import threading
from dataclasses import fields
from typing import Any

from app.settings import Settings

DEFAULT_SETTINGS_PATH = "data/settings.json"

# YOLO26 is an experimental lane: YoloDetector loads it fine (ultralytics
# >= 8.4), but presets and tuning guidance are calibrated for YOLO11, so
# selecting one surfaces a soft warning via compute_warnings(). Mirrored by
# hand in desktop settingsFields.ts (EXPERIMENTAL_MODELS / MODEL_SPEC_HINTS).
EXPERIMENTAL_MODELS = {"yolo26n.pt", "yolo26s.pt", "yolo26m.pt"}
ALLOWED_MODELS = {
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
} | EXPERIMENTAL_MODELS
ALLOWED_DEVICES = {"auto", "cpu", "cuda"}
ALLOWED_RESIZE_MODES = {"auto", "letterbox", "stretch"}

# A custom model is any weights file under sidecar/models/. That covers the
# Roboflow-exported grocery model, whose .onnx the local inference server
# caches and which `YOLO()` loads directly (see docs/DETECTOR_BACKENDS.md §1a).
#
# `active_model` is operator-supplied and reaches `YOLO(path)`, so it is
# constrained rather than free-form: one fixed directory, a known extension,
# no absolute paths and no traversal. Validation deliberately never touches
# the filesystem — settings tests stay pure, and a missing file is reported by
# POST /api/detector/probe instead.
CUSTOM_MODEL_DIR = "models/"
CUSTOM_MODEL_SUFFIXES = (".onnx", ".pt")


def is_custom_model(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if not normalized.startswith(CUSTOM_MODEL_DIR) or ".." in normalized:
        return False
    name = normalized[len(CUSTOM_MODEL_DIR):]
    # A real filename directly in models/: a non-empty stem, a known suffix,
    # and no nested directory to walk into.
    return (
        name.endswith(CUSTOM_MODEL_SUFFIXES)
        and "/" not in name
        and len(name.rsplit(".", 1)[0]) > 0
    )


def resolve_resize_mode(mode: str, active_model: str) -> str:
    """"auto" means: match how this model was trained.

    A custom model under models/ is a Roboflow export, and Roboflow's default
    preprocessing is "Stretch to". The stock YOLO weights are letterbox-trained.
    """
    if mode != "auto":
        return mode
    return "stretch" if is_custom_model(active_model) else "letterbox"


def is_allowed_model(value: object) -> bool:
    return isinstance(value, str) and (value in ALLOWED_MODELS or is_custom_model(value))
# Detector backends. "native" runs weights in-process; "local_api" talks to a
# Roboflow inference server on this machine; "cloud_api" talks to Roboflow's
# serverless endpoint. Mirrored by hand in desktop settingsFields.ts.
ALLOWED_BACKENDS = {"native", "local_api", "cloud_api"}
REMOTE_BACKENDS = {"local_api", "cloud_api"}

# Settings that can be set back to None ("leave the camera alone"). Only the
# device controls: every other field has a non-optional type.
RESETTABLE_FIELDS = {
    "camera_brightness",
    "camera_exposure",
    "camera_autofocus",
    "camera_focus",
}

# Fields the running Pipeline re-reads from `settings` every frame/track update,
# or that _push_live_settings hands to the open camera and detector, so
# mutating them in place takes effect without stopping capture. Everything
# else is baked into source/detector objects at /api/capture/start time.
HOT_RELOADABLE_FIELDS = {
    "infer_frame_skip",
    "preview_height",
    "preview_max_fps",
    "track_expiry_s",
    # Read per inference call (YoloDetector passes it to track(); the remote
    # detector filters responses against it), so a setter is all it needs.
    "conf_threshold",
    # Plain cap.set() writes against the open handle — see
    # CameraCapture.set_controls, which defers them to the capture thread.
    "camera_brightness",
    "camera_exposure",
    "camera_autofocus",
    "camera_focus",
}
RESTART_REQUIRED_FIELDS = {
    "active_model",
    "camera_index",
    "capture_width",
    "capture_height",
    "capture_fps",
    "imgsz",
    "resize_mode",
    "device",
    # Every backend field is baked into the detector object at
    # /api/capture/start time, so none of them can be swapped mid-session.
    "detector_backend",
    "roboflow_workspace",
    "roboflow_workflow_id",
    "local_api_url",
    "cloud_api_url",
    "remote_infer_size",
    "remote_timeout_s",
    "remote_max_retries",
}

_COMMON_CAPTURE_MODES = {(640, 480), (1280, 720), (1920, 1080)}

# A remote round trip must not outlast the tracker's memory of a track, or a
# stationary item expires between calls and is logged twice. The safe floor is
# therefore per-backend, because the two remote backends are an order of
# magnitude apart: measured 600-3250 ms against the serverless endpoint, but
# ~90 ms warm against a local inference server on the same PC. Holding
# local_api to the cloud's 5 s floor would delay every "item left" by 5 s for
# no reason.
MIN_TRACK_EXPIRY_S_BY_BACKEND = {
    "local_api": 2.0,
    "cloud_api": 5.0,
}
# Kept as the conservative fallback for any backend not listed above.
MIN_REMOTE_TRACK_EXPIRY_S = 5.0


def min_track_expiry_s(backend: str) -> float:
    """The lowest track_expiry_s that is safe for `backend`."""
    return MIN_TRACK_EXPIRY_S_BY_BACKEND.get(backend, MIN_REMOTE_TRACK_EXPIRY_S)

_lock = threading.Lock()


def _valid_field(name: str, value: Any) -> bool:
    if name == "active_model":
        return is_allowed_model(value)
    if name == "resize_mode":
        return isinstance(value, str) and value in ALLOWED_RESIZE_MODES
    if name == "device":
        return isinstance(value, str) and value in ALLOWED_DEVICES
    if name == "camera_index":
        return isinstance(value, int) and 0 <= value <= 8
    if name == "capture_width":
        return isinstance(value, int) and 160 <= value <= 3840
    if name == "capture_height":
        return isinstance(value, int) and 120 <= value <= 2160
    if name == "capture_fps":
        return isinstance(value, int) and 1 <= value <= 120
    if name == "conf_threshold":
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
    if name == "imgsz":
        # YOLO requires the inference size to be a multiple of the model stride
        # (32); ultralytics silently rounds otherwise, so reject up front.
        return isinstance(value, int) and 320 <= value <= 1920 and value % 32 == 0
    if name == "infer_frame_skip":
        return isinstance(value, int) and 0 <= value <= 30
    if name == "preview_max_fps":
        return isinstance(value, int) and 0 <= value <= 120
    if name == "preview_height":
        return isinstance(value, int) and 120 <= value <= 1080
    if name == "track_expiry_s":
        return isinstance(value, (int, float)) and 0.0 < value <= 30.0
    if name == "detector_backend":
        return isinstance(value, str) and value in ALLOWED_BACKENDS
    if name in ("roboflow_workspace", "roboflow_workflow_id"):
        return isinstance(value, str) and bool(value.strip())
    if name in ("local_api_url", "cloud_api_url"):
        return isinstance(value, str) and value.startswith(("http://", "https://"))
    if name == "remote_infer_size":
        return isinstance(value, int) and 128 <= value <= 1920
    if name == "remote_timeout_s":
        return isinstance(value, (int, float)) and 0.1 <= value <= 60.0
    if name == "remote_max_retries":
        return isinstance(value, int) and 0 <= value <= 5
    if name in ("camera_brightness", "camera_exposure", "camera_focus"):
        return value is None or isinstance(value, (int, float))
    if name == "camera_autofocus":
        return value is None or isinstance(value, bool)
    return False


def load_settings(path: str = DEFAULT_SETTINGS_PATH) -> Settings:
    """Load settings from disk, overlaying only known/valid fields onto the
    hardcoded defaults. Never raises: a missing, corrupt, or hand-edited-wrong
    file just falls back to defaults (per-field, not all-or-nothing)."""
    settings = Settings()
    with _lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return settings
    if not isinstance(data, dict):
        return settings
    valid_names = {f.name for f in fields(Settings)}
    for key, value in data.items():
        if key in valid_names and _valid_field(key, value):
            setattr(settings, key, value)
    return settings


def save_settings(settings: Settings, path: str = DEFAULT_SETTINGS_PATH) -> None:
    """Atomic write: write to a temp file then os.replace() into place, so a
    crash mid-write never leaves a half-written file that fails to load."""
    payload = {f.name: getattr(settings, f.name) for f in fields(Settings)}
    with _lock:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)


def compute_warnings(
    settings: Settings, state: str, api_key_present: bool | None = None
) -> list[str]:
    """Soft warnings surfaced in SettingsResponse. `api_key_present` is passed
    in rather than read here so tests stay off the filesystem; None means
    "look it up"."""
    warnings: list[str] = []
    if settings.detector_backend in REMOTE_BACKENDS:
        if api_key_present is None:
            from app.credentials import has_api_key

            api_key_present = has_api_key()
        if not api_key_present:
            warnings.append(
                f"{settings.detector_backend} needs a Roboflow API key, but none is set. "
                "Add ROBOFLOW_API_KEY to sidecar/.env (see .env.example); capture will "
                "fail to start without it."
            )
        # If a round trip outlasts track_expiry_s, the tracker forgets a
        # stationary item between calls and re-issues its id — one physical
        # item, several log rows. The floor is per-backend; see above.
        floor = min_track_expiry_s(settings.detector_backend)
        if settings.track_expiry_s < floor:
            warnings.append(
                f"track_expiry_s={settings.track_expiry_s}s is too low for "
                f"{settings.detector_backend}: a slow round trip will expire a stationary "
                f"item and log it twice. Use at least {floor}s."
            )
    if settings.detector_backend == "cloud_api":
        warnings.append(
            "cloud_api sends every inference frame to Roboflow's servers: it requires a "
            "live internet connection, bills per inference, and contradicts the PRD's "
            "offline guarantee. Intended for demos, not deployment."
        )
        if settings.cloud_api_url.startswith("http://"):
            warnings.append("cloud_api_url should be https — plain http is rejected upstream.")
    if settings.detector_backend == "local_api":
        warnings.append(
            f"local_api expects a Roboflow inference server reachable at "
            f"{settings.local_api_url} (`inference server start`). Use Test Connection "
            "before starting capture."
        )
    if settings.active_model in EXPERIMENTAL_MODELS:
        warnings.append(
            f"{settings.active_model} is experimental: presets and tuning guidance "
            "are calibrated for YOLO11. Requires ultralytics >= 8.4 in the sidecar "
            "environment; weights auto-download on the first capture start (needs "
            "internet once)."
        )
    if state == "running":
        locked = ", ".join(sorted(RESTART_REQUIRED_FIELDS))
        warnings.append(f"Capture is running — {locked} require stopping capture first.")
    if settings.imgsz > 960:
        warnings.append(
            "imgsz above 960 sharply raises inference latency; small/fast-moving "
            "objects detect better at higher imgsz, but confirm the live infer fps "
            "still keeps up after starting capture."
        )
    if (settings.capture_width, settings.capture_height) not in _COMMON_CAPTURE_MODES:
        warnings.append(
            "Camera may not support the requested resolution; cheap webcams often "
            "silently clamp to their nearest supported mode. Verify via the live "
            "preview after starting capture."
        )
    if settings.capture_fps > 0:
        seconds_between_inferences = (settings.infer_frame_skip + 1) / settings.capture_fps
        if seconds_between_inferences >= settings.track_expiry_s * 0.5:
            warnings.append(
                "infer_frame_skip is high relative to track_expiry_s; tracks may be "
                "marked 'left' between inferences. Consider raising track_expiry_s or "
                "lowering infer_frame_skip."
            )
    return warnings
