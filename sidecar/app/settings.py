from dataclasses import dataclass


@dataclass
class Settings:
    active_model: str = "yolo11n.pt"
    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 60
    conf_threshold: float = 0.5
    infer_frame_skip: int = 0
    device: str = "auto"
    preview_height: int = 720
    track_expiry_s: float = 1.5


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
