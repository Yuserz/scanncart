from dataclasses import dataclass


@dataclass
class Settings:
    # The Roboflow-exported grocery model, run in-process. It is the only
    # default that both detects the actual SKUs and keeps the PRD's offline
    # promise: measured 51 ms on CPU alone vs ~100 ms for the same model over
    # local_api, because that 100 ms is an HTTP round trip, not inference.
    # See docs/DETECTOR_BACKENDS.md §1a for how the file gets to models/.
    active_model: str = "models/scanncart-grocery.onnx"
    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 60
    conf_threshold: float = 0.5
    imgsz: int = 640
    # How a frame is fitted to `imgsz` before detection, and it must match how
    # the model was trained. Ultralytics letterboxes (pads to square, keeping
    # aspect); Roboflow's export config for the grocery model records
    # "resize": {"format": "Stretch to"}. On a 1280x720 frame letterboxing
    # leaves 140px bars and uses only 56% of the 640x640 canvas, so every
    # object arrives at ~56% of its training pixel area. "auto" picks stretch
    # for a custom Roboflow export and letterbox for the stock YOLO weights,
    # which are themselves letterbox-trained.
    resize_mode: str = "auto"
    infer_frame_skip: int = 0
    device: str = "auto"
    preview_height: int = 720
    # Preview frames per second. The preview thread fills the gaps between
    # inferences with this cadence, so the image stays smooth even when
    # inference runs at ~10 fps. Costs one JPEG encode each (~12 ms at 720p),
    # which competes with inference on a CPU-bound machine — lower it, or set
    # 0 to emit only on inference as before.
    preview_max_fps: int = 30
    track_expiry_s: float = 1.5

    # Which detector implementation backs capture. "native" runs the weights in
    # this process (the only backend that satisfies the PRD's offline promise);
    # the two remote backends differ only by URL. See docs/DETECTOR_BACKENDS.md.
    detector_backend: str = "native"
    roboflow_workspace: str = "yusri-caloyloy"
    roboflow_workflow_id: str = "scanncart-grocery-vscanncart-grocery-1-yolo11n-t1-logic"
    local_api_url: str = "http://127.0.0.1:9001"
    cloud_api_url: str = "https://serverless.roboflow.com"
    # Frames are downscaled to this longest edge before transmit. YOLO11 infers
    # at 640 regardless, so sending full 1080p frames is pure bandwidth waste.
    remote_infer_size: int = 640
    remote_timeout_s: float = 5.0
    remote_max_retries: int = 2


def resolve_device(pref: str) -> str:
    # "cpu" is always honored. "auto" and an explicit "cuda" both want the GPU,
    # but only when torch can actually use it — otherwise fall back to "cpu" so a
    # stale or forced "cuda" (e.g. persisted before a CPU-only torch install)
    # never crashes capture at start on a machine without a CUDA-enabled torch.
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
