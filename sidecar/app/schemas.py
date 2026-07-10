from typing import Literal
from pydantic import BaseModel


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
