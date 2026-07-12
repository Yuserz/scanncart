from typing import Literal
from pydantic import BaseModel, Field, field_validator

from app.settings_store import ALLOWED_DEVICES, ALLOWED_MODELS


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
    infer_frame_skip: int
    device: str
    preview_height: int
    track_expiry_s: float


class SettingsResponse(SettingsPayload):
    hot_reloadable_fields: list[str]
    restart_required_fields: list[str]
    warnings: list[str] = []


class SettingsUpdateRequest(BaseModel):
    active_model: str | None = None
    camera_index: int | None = Field(default=None, ge=0, le=8)
    capture_width: int | None = Field(default=None, ge=160, le=3840)
    capture_height: int | None = Field(default=None, ge=120, le=2160)
    capture_fps: int | None = Field(default=None, ge=1, le=120)
    conf_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    infer_frame_skip: int | None = Field(default=None, ge=0, le=30)
    device: str | None = None
    preview_height: int | None = Field(default=None, ge=120, le=1080)
    track_expiry_s: float | None = Field(default=None, gt=0.0, le=30.0)

    @field_validator("active_model")
    @classmethod
    def _validate_active_model(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_MODELS:
            raise ValueError(f"active_model must be one of {sorted(ALLOWED_MODELS)}")
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_DEVICES:
            raise ValueError(f"device must be one of {sorted(ALLOWED_DEVICES)}")
        return v


class ApplyPresetRequest(BaseModel):
    name: str


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
