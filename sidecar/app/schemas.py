from typing import Literal
from pydantic import BaseModel, Field, field_validator

from app.settings_store import (
    ALLOWED_BACKENDS,
    ALLOWED_DEVICES,
    ALLOWED_MODELS,
    ALLOWED_RESIZE_MODES,
    CUSTOM_MODEL_DIR,
    RESETTABLE_FIELDS,
    is_allowed_model,
)


class Detection(BaseModel):
    track_id: int | None
    cls: str
    conf: float
    box: tuple[float, float, float, float]


class Stats(BaseModel):
    infer_fps: float
    capture_fps: float
    latency_ms: float


class FrameMessage(BaseModel):
    type: Literal["frame"]
    ts: float
    seq: int
    jpeg: str
    detections: list[Detection]
    stats: Stats


class StatusMessage(BaseModel):
    type: Literal["status"]
    state: str
    detail: str = ""


class HealthResponse(BaseModel):
    state: str
    active_model: str
    device: str


class LogEvent(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    max_conf: float
    entered_at: float
    left_at: float | None = None


class LogsResponse(BaseModel):
    session_id: int | None = None
    events: list[LogEvent] = []


class SettingsPayload(BaseModel):
    active_model: str
    camera_index: int
    capture_width: int
    capture_height: int
    capture_fps: int
    conf_threshold: float
    imgsz: int
    resize_mode: str
    infer_frame_skip: int
    device: str
    preview_height: int
    preview_max_fps: int
    track_expiry_s: float
    detector_backend: str
    roboflow_workspace: str
    roboflow_workflow_id: str
    local_api_url: str
    cloud_api_url: str
    remote_infer_size: int
    remote_timeout_s: float
    remote_max_retries: int
    camera_brightness: float | None
    camera_exposure: float | None
    camera_autofocus: bool | None
    camera_focus: float | None


class SettingsResponse(SettingsPayload):
    hot_reloadable_fields: list[str]
    restart_required_fields: list[str]
    warnings: list[str] = []
    # Whether a Roboflow API key is configured. The key itself must never be
    # serialized — this response goes straight to the renderer.
    roboflow_api_key_present: bool = False


class SettingsUpdateRequest(BaseModel):
    active_model: str | None = None
    camera_index: int | None = Field(default=None, ge=0, le=8)
    capture_width: int | None = Field(default=None, ge=160, le=3840)
    capture_height: int | None = Field(default=None, ge=120, le=2160)
    capture_fps: int | None = Field(default=None, ge=1, le=120)
    conf_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    imgsz: int | None = Field(default=None, ge=320, le=1920)
    resize_mode: str | None = None
    infer_frame_skip: int | None = Field(default=None, ge=0, le=30)
    device: str | None = None
    preview_height: int | None = Field(default=None, ge=120, le=1080)
    preview_max_fps: int | None = Field(default=None, ge=0, le=120)
    track_expiry_s: float | None = Field(default=None, gt=0.0, le=30.0)
    detector_backend: str | None = None
    roboflow_workspace: str | None = Field(default=None, min_length=1)
    roboflow_workflow_id: str | None = Field(default=None, min_length=1)
    local_api_url: str | None = None
    cloud_api_url: str | None = None
    remote_infer_size: int | None = Field(default=None, ge=128, le=1920)
    remote_timeout_s: float | None = Field(default=None, ge=0.1, le=60.0)
    remote_max_retries: int | None = Field(default=None, ge=0, le=5)
    # Bounds mirror settingsFields.ts's min/max for these controls. They are
    # generous because the meaningful range is device-specific; they exist to
    # reject nonsense, not to encode one camera's scale. Note that calibration
    # applies its recommendation through _apply_settings_patch directly and so
    # is not validated here.
    camera_brightness: float | None = Field(default=None, ge=0.0, le=255.0)
    # Windows exposure is log2 seconds: -6 is 1/64 s, 0 is one full second.
    camera_exposure: float | None = Field(default=None, ge=-13.0, le=0.0)
    camera_autofocus: bool | None = None
    camera_focus: float | None = Field(default=None, ge=0.0, le=1023.0)

    # exclude_none=True means a patch can never send a field back to null, so
    # without this Revert cannot restore "leave the camera alone" — which is
    # the default state of all four controls, and therefore the saved baseline
    # on a fresh install. Restricted to those four because they are the only
    # settings whose type admits None; nulling imgsz would break capture.
    reset_fields: list[str] | None = None

    @field_validator("reset_fields")
    @classmethod
    def _validate_reset_fields(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            unknown = set(v) - RESETTABLE_FIELDS
            if unknown:
                raise ValueError(f"reset_fields must be a subset of {sorted(RESETTABLE_FIELDS)}")
        return v

    @field_validator("detector_backend")
    @classmethod
    def _validate_detector_backend(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_BACKENDS:
            raise ValueError(f"detector_backend must be one of {sorted(ALLOWED_BACKENDS)}")
        return v

    @field_validator("local_api_url", "cloud_api_url")
    @classmethod
    def _validate_api_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    @field_validator("active_model")
    @classmethod
    def _validate_active_model(cls, v: str | None) -> str | None:
        if v is not None and not is_allowed_model(v):
            raise ValueError(
                f"active_model must be one of {sorted(ALLOWED_MODELS)}, "
                f"or a custom .onnx/.pt under {CUSTOM_MODEL_DIR}"
            )
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_DEVICES:
            raise ValueError(f"device must be one of {sorted(ALLOWED_DEVICES)}")
        return v

    @field_validator("resize_mode")
    @classmethod
    def _validate_resize_mode(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {sorted(ALLOWED_RESIZE_MODES)}")
        return v

    @field_validator("imgsz")
    @classmethod
    def _validate_imgsz(cls, v: int | None) -> int | None:
        if v is not None and v % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32 (the YOLO model stride)")
        return v


class CameraInfo(BaseModel):
    """One enumerated capture device. `width`/`height` are what the device
    actually opened at — the operator's check that `name` was paired with the
    right index, since that pairing is positional. See app/cameras.py."""
    index: int
    name: str
    width: int
    height: int


class CamerasResponse(BaseModel):
    cameras: list[CameraInfo] = []
    # Probing opens each device, so it is skipped while capture holds one.
    # False means `cameras` is a cached or empty list, not a fresh scan.
    probed: bool = True
    detail: str = ""


class CameraQualityResponse(BaseModel):
    """Live image metrics, for the setup wizard's readout."""
    available: bool = False
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    capture_fps: float = 0.0
    target_fps: float = 0.0
    verdicts: dict[str, str] = {}
    detail: str = ""


class DetectorProbeResponse(BaseModel):
    """Result of checking the selected backend before capture starts."""
    backend: str
    reachable: bool
    detail: str = ""
    latency_ms: float | None = None
    class_names: list[str] = []


class ApplyPresetRequest(BaseModel):
    name: str


class ControlSupportPayload(BaseModel):
    brightness: bool = False
    exposure: bool = False
    gain: bool = False
    focus: bool = False
    autofocus: bool = False


class CameraProfileResponse(BaseModel):
    device_key: str
    backend: str
    width: int
    height: int
    fps_auto_exposure: float
    fps_capped_exposure: float
    controls: ControlSupportPayload
    recommended: dict = {}
    measured_at: float = 0.0
    # Evidence per control: value, metric, baseline, reached, probes. Mirrors
    # CameraProfile.measured — this model is built with **asdict(profile) and
    # Pydantic drops unknown keys silently, so a field missing here vanishes
    # between the sidecar and the UI with no error raised.
    measured: dict = {}
    # 0 means the profile predates the sweep, which the card must report
    # differently from "this camera responded to nothing".
    sweep_version: int = 0


class StoredProfileResponse(BaseModel):
    """The saved profile for the camera currently configured, if any.

    `profile` is null for a camera that has never been calibrated — a normal
    state the UI renders, not an error, which is why this is a 200 rather
    than the 404 POST /api/camera/profile/apply returns.
    """
    profile: CameraProfileResponse | None = None


class HardwareInfo(BaseModel):
    cpu_count: int
    ram_gb: float
    cuda_available: bool
    accelerator: Literal["cuda", "integrated", "cpu"] = "cpu"
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None


class SystemInfoResponse(HardwareInfo):
    recommended_preset: str


class PresetInfo(BaseModel):
    name: str
    label: str
    description: str
    settings: dict


class PresetsResponse(BaseModel):
    presets: list[PresetInfo]
    recommended: str
