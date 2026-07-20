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

# Fields the running Pipeline re-reads from `settings` every frame/track update,
# so mutating them in place takes effect without stopping capture. Everything
# else is baked into source/detector objects at /api/capture/start time.
HOT_RELOADABLE_FIELDS = {"infer_frame_skip", "preview_height", "track_expiry_s"}
RESTART_REQUIRED_FIELDS = {
    "active_model",
    "camera_index",
    "capture_width",
    "capture_height",
    "capture_fps",
    "conf_threshold",
    "imgsz",
    "device",
}

_COMMON_CAPTURE_MODES = {(640, 480), (1280, 720), (1920, 1080)}

_lock = threading.Lock()


def _valid_field(name: str, value: Any) -> bool:
    if name == "active_model":
        return isinstance(value, str) and value in ALLOWED_MODELS
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
    if name == "preview_height":
        return isinstance(value, int) and 120 <= value <= 1080
    if name == "track_expiry_s":
        return isinstance(value, (int, float)) and 0.0 < value <= 30.0
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


def compute_warnings(settings: Settings, state: str) -> list[str]:
    warnings: list[str] = []
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
