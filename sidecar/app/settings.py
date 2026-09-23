from dataclasses import dataclass, field


@dataclass
class Settings:
    # The Roboflow-exported grocery model, run in-process. It is the only
    # default that both detects the actual SKUs and keeps the PRD's offline
    # promise: measured 51 ms on CPU alone vs ~100 ms for the same model over
    # local_api, because that 100 ms is an HTTP round trip, not inference.
    # See docs/DETECTOR_BACKENDS.md §1a for how the file gets to models/.
    active_model: str = "models/scanncart-grocery.onnx"
    camera_index: int = 0
    # 640x480@30 opens and streams reliably over USB 2.0; the StreamCam's
    # 1080p60 needs USB 3.0 and a failed mode switch there can wedge the MSMF
    # driver until the camera is physically replugged (see camera.py).
    capture_width: int = 640
    capture_height: int = 480
    capture_fps: int = 30
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
    # Whether the preview is mirrored left to right. On by default because a counter view that
    # moves the way a mirror does is easier to aim at while holding a product, and off is what
    # an operator wants when they need to read a label or a barcode the right way round.
    #
    # Preview-only, and that placement is the whole point: capture, inference, the logging store
    # and any dataset frames keep the true orientation, because the weights were trained on
    # ordinary photographs and a mirrored input asks the model to read reversed text and mirrored
    # brand marks — exactly what identifies a sachet. The overlay follows this setting rather
    # than a second rule of its own, so turning it off puts the boxes back on the items in the
    # same frame. Hot-reloadable: `Pipeline` reads it at each emit, so the toggle takes effect
    # without stopping capture.
    preview_mirror: bool = True
    track_expiry_s: float = 1.5
    # Drop detections whose box is pinned to all four frame edges — the shape a prediction takes
    # when the model wanted something larger than the image. This is a measured defect in the
    # grocery weights rather than a precaution: 19 of the 25 detections the 50 stored empty-counter
    # negatives produced are exactly that shape, against 0 of 60 real product frames, and a live
    # capture logged one as a persistent phantom item at 0.957 confidence.
    #
    # On by default because those weights are what the app presently runs. It is a setting, and not
    # a rule baked into the detector, because ~1.4% of the weights' own training labels touch all
    # four edges: an item that genuinely fills the frame is the case this can mistake for a phantom,
    # and the escape hatch has to exist. Expect it to become the wrong default once v2 ships with
    # the hard negatives in it, at which point the phantoms stop and only that risk remains.
    # Hot-reloadable, so it can be turned off while watching the item log.
    suppress_clamped_detections: bool = True
    # Class names to KEEP (post-inference); empty list = keep everything.
    # Narrows detection to checkout-relevant classes; hot-reloadable.
    class_allowlist: list[str] = field(default_factory=list)

    # Device controls. None means "leave the camera alone", so behaviour is
    # unchanged until calibration proposes values. The StreamCam's automatic
    # focus and exposure track faces; a counter has none, which is why locked
    # manual values suit this app.
    camera_brightness: float | None = None
    camera_exposure: float | None = None
    camera_autofocus: bool | None = None
    camera_focus: float | None = None

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
