from dataclasses import dataclass, field

from app.hardware import HardwareInfo


@dataclass
class Preset:
    name: str
    label: str
    description: str
    # Partial patch applied the same way a PATCH /api/settings body is —
    # never includes camera_index, which is a physical-topology fact about
    # this machine, not a performance tier.
    settings: dict = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    "low_end": Preset(
        name="low_end",
        label="Low-end / CPU-only laptop",
        description="No dedicated GPU, 4 or fewer cores, or under 8GB RAM. Prioritizes smoothness over accuracy.",
        settings=dict(
            active_model="yolo11n.pt",
            device="cpu",
            capture_width=640,
            capture_height=480,
            capture_fps=15,
            infer_frame_skip=2,
            conf_threshold=0.5,
            preview_height=480,
            track_expiry_s=2.5,
        ),
    ),
    "mid_range": Preset(
        name="mid_range",
        label="Mid-range / integrated or entry GPU",
        description="6 or more cores and at least 8GB RAM, or a modest GPU. Balanced accuracy and framerate.",
        settings=dict(
            active_model="yolo11s.pt",
            device="auto",
            capture_width=1280,
            capture_height=720,
            capture_fps=30,
            infer_frame_skip=1,
            conf_threshold=0.5,
            preview_height=720,
            track_expiry_s=2.0,
        ),
    ),
    "high_end": Preset(
        name="high_end",
        label="High-end / discrete GPU",
        description="Discrete GPU with at least 4GB VRAM. Maximizes accuracy and framerate.",
        settings=dict(
            active_model="yolo11m.pt",
            device="cuda",
            capture_width=1920,
            capture_height=1080,
            capture_fps=60,
            infer_frame_skip=0,
            conf_threshold=0.45,
            preview_height=720,
            track_expiry_s=1.5,
        ),
    ),
}


def recommend_preset(hw: HardwareInfo) -> str:
    if hw.cuda_available and (hw.gpu_vram_gb or 0) >= 4:
        return "high_end"
    if hw.cpu_count >= 6 and hw.ram_gb >= 8:
        return "mid_range"
    return "low_end"
