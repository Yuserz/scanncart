from dataclasses import dataclass, field


@dataclass
class Settings:
    # The locally trained grocery model (see docs/MODEL_TRAINING.md), run
    # in-process. It is the only default that both detects the actual SKUs and
    # keeps the PRD's offline promise. Trained 2026-09-24 from the dataset
    # export to mAP50-95 0.944; measured level with the Roboflow ONNX export
    # on CUDA (~18 ms isolated, ~40 fps in-app) while letterbox-native and
    # retrainable. The ONNX export remains selectable (models/scanncart-grocery.onnx).
    # See docs/DETECTOR_BACKENDS.md §1a for the backend comparison.
    active_model: str = "models/scanncart-grocery-v1.pt"
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
    # the model was trained. `auto` means "match how these weights were trained":
    # it reads the requirement recorded beside them (`models/<stem>.json`, written
    # by `train_model.py --install` or by the Admin Panel's record button) and only
    # falls back to the format heuristic when nothing was recorded — `stretch` for
    # a custom `.onnx` (a Roboflow export, which that export configures as "Stretch
    # to"), `letterbox` for a custom `.pt`. That heuristic is a guess about weights
    # nobody recorded anything about, and the app reports it as one
    # (`SettingsResponse.unrecorded_resize_mode`); a recorded requirement is a fact
    # and is used silently, which is what makes `auto` the right default rather
    # than merely the safe-looking one. An explicit `letterbox`/`stretch` remains
    # the operator overriding all of that, and is always honoured.
    #
    # The picker lists `letterbox` first and calls `stretch` experimental: preserving
    # package proportions is the checkout default, and stretch warps the frame.
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
    # Drop a detection pinned to exactly three frame edges - the shape the all-four-edges rule
    # leaves behind. A measured defect of the same weights, not a precaution: with nothing placed in
    # front of the camera, 20 of 20 live frames returned one box covering ~96% of the image as Bear
    # Brand at 0.90-0.94, pinned to the top, right and bottom edges and stopping ~5% short of the
    # left - five times the tolerance the clamp rule above tests at.
    #
    # On by default, and the price is measured rather than left as a side effect. It costs real
    # detections, because the two populations overlap: over the whole v1 export it also drops 252 of
    # 2018 ground-truth-matched detections (~12%), and the closest of those to a phantom is a real
    # `lucky_me_pancit_canton_calamansi_flavor` pouch at area 0.972 with three edges inside 1%,
    # confidence 0.966, held right up to the lens - which is how this project's own capture
    # protocol shoots a product. The cost is paid knowingly, in
    # exchange for the defect: the phantom is what a fresh install shows the operator on an empty
    # counter, item-log row and all, and a lever nobody flips is not a fix. The rule that removes it
    # for good is the model's - train the staged hard negatives in, which is what v2 exists for -
    # and this is the one that works until then, so it ships on.
    #
    # Hot-reloadable, so it can be turned off the moment an item held close stops registering -
    # which is the failure mode to expect, and why the Live view's suppressed count is visible
    # while it happens.
    suppress_frame_filling_detections: bool = True
    # Drop a frame-spanning box, or a wide band along a top/bottom edge, that the model is not
    # confident about (`acceptance.is_unsure_phantom`, `PHANTOM_CONF_CEILING` 0.85). The two shape
    # rules above are expressed in pinned frame edges, and the model has two more ways to say
    # "something is here" on an empty counter that pin only one or two: a bottom band at 0.50-0.79
    # (measured live: 862 detections over 1500 frames, and 10 item-log rows in 25 seconds) and, under
    # the geometry v1's record requires, a frame-spanning box at 0.50-0.83 (1024 over 1500 frames).
    # Neither is reachable by shape alone, because both shapes are also real products when the model
    # is sure of them.
    #
    # On by default, and the price is measured: over the whole v1 export (2018 ground-truth-matched
    # detections) it drops 6, or 0.30%, and every one of those is a large box at 0.719-0.839 - the
    # rule's whole statement is that a big box the model is unsure about is not an item. That is
    # cheaper than either shape rule above, 36 (1.8%) and 252 (12%) of the same 2018 matched, and it
    # is what makes an empty counter actually log nothing.
    #
    # Hot-reloadable, for the same reason as the others: the failure mode is an item the operator
    # can see and the model is unsure of, and the escape hatch has to work while capture runs.
    suppress_unsure_phantoms: bool = True
    # Class names to KEEP (post-inference); empty list = keep everything.
    # Narrows detection to checkout-relevant classes; hot-reloadable.
    class_allowlist: list[str] = field(default_factory=list)

    # Device controls. None means "this app imposes no value", so behaviour is
    # unchanged until calibration proposes values — and on the device that is
    # not a no-op: a control the app has already written is handed back when
    # the setting returns to None (see CameraCapture._with_restores). The
    # StreamCam's automatic focus and exposure track faces; a counter has none,
    # which is why locked manual values suit this app.
    camera_brightness: float | None = None
    camera_exposure: float | None = None
    camera_autofocus: bool | None = None
    camera_focus: float | None = None
    # The app's own auto-exposure (CameraCapture's AutoExposure): it keeps the shutter at the
    # longest the capture fps allows and moves brightness to hold the picture's level, so the
    # view follows the room's light without the framerate loss the camera's built-in automatic
    # exposure causes (measured 12 fps). While on, camera_brightness/camera_exposure are only
    # its starting point. Hot-reloadable.
    camera_auto_exposure: bool = True
    # The one trade auto exposure may make in a room too dark for the framerate's shutter: one stop
    # longer (30 fps at a 60 fps setting) before it gives up and reports `too_dark`. Off by default,
    # because 60 fps is the promise and a lamp is the better fix; tracking still works at 30.
    # Hot-reloadable; switching it off puts a lengthened shutter straight back.
    camera_auto_exposure_slow: bool = False

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
