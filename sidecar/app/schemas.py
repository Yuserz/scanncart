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
    is_custom_model,
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
    # The class names the model that is *running* predicts, as its own vocabulary reports them,
    # reported once capture is far enough along to know them. Empty means "not known yet" - a
    # detector that has not inferred has `names == {}` on purpose - and never "predicts nothing",
    # which is the reading the client must not take from an empty list.
    #
    # Carried because the count is a readout in its own right, not only the input to a verdict: the
    # Live view's stats strip shows how many classes the running model has, which is how an operator
    # sees `24` where they expected `8` even in the case where every name happens to be on the
    # roster. `class_warnings` is the judgement of these names (`app/roster.py`); the two travel
    # together so a client never has to explain one without the other.
    class_names: list[str] = []
    # What is wrong with that class list, judged by the same module. Empty means either nothing is
    # wrong or nothing is known yet - the two are the same to a client that has no better source, and
    # the honest default for a message that must not cry wolf before the first frame.
    #
    # Carried on the status protocol rather than only on the probe because "the app you are running
    # has a 24-class model" is a fact about the capture, and a probe is a check somebody has to
    # think to press. The difference from the probe's field of the same name is *when*: this one
    # describes a capture in progress, the probe describes a model that has not been started.
    class_warnings: list[str] = []


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
    class_allowlist: list[str]
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


class UnrecordedResizeMode(BaseModel):
    """Weights whose training geometry nothing recorded, and the sentence saying so.

    The one resize case with a remedy the app can perform in place, which is why it is a
    structured entry and not a string in `warnings`: `POST /api/models/record` writes the
    requirement into `models/<stem>.json`, and a button has to know which weight and which mode
    to write. Everything the panel needs to offer that is here, `warning` included — the renderer
    holds no copy of the sentence, so the button and the reason for it cannot describe different
    situations.

    `resize_mode` is the mode to record, which is `auto`'s answer for this weight (letterbox —
    `settings_store.resize_guess` only produces this entry for that mode) and therefore the
    geometry the running detector was already built with. Recording it cannot change what the
    model does; it converts a fallback into a stated fact, and the warning it replaces is about
    the *proof*, not about the value.

    `warning` and `remedy` are separate because two views render this entry and only one of them
    holds the button: the Live view surfaces the assumption while a capture runs — that is where
    the weak `far` detections show up — and its only route to the fix is naming it. Both views
    print both sentences verbatim, so neither can describe a situation, or a control, the other
    view is not showing.
    """

    model: str
    resize_mode: str
    warning: str
    remedy: str


class SettingsResponse(SettingsPayload):
    hot_reloadable_fields: list[str]
    restart_required_fields: list[str]
    warnings: list[str] = []
    # Whether a Roboflow API key is configured. The key itself must never be
    # serialized — this response goes straight to the renderer.
    roboflow_api_key_present: bool = False
    # What `resize_mode` actually resolves to for the selected weights, `auto` included: the
    # geometry the local detector is fitted to, from the same `resolve_resize_mode` call
    # `_default_detector_factory` makes with the same arguments — so it is the running
    # configuration rather than a re-derivation of it. Reported rather than left to the renderer
    # because the renderer *cannot* derive it: `auto`'s answer depends on the weights' record
    # (and on the file format for weights with none), neither of which is in this response.
    #
    # `None` when the backend does not run these weights at all: `local_api`/`cloud_api` send the
    # frame to a workflow that holds its own model and resizes to `remote_infer_size` server-side,
    # so `active_model` and `resize_mode` are not in the inference path. The Live view reads that
    # absence as "nothing to say about local weights" and drops the readout, which is why the
    # distinction is `None` rather than an empty mode.
    resize_mode_resolved: str | None = None
    # The geometry `auto` *assumes* because nothing is recorded beside these weights, with the
    # sentences that explain the risk and the mode a one-click record would write. `None`
    # whenever there is no assumption to report — a record, an explicit mode, stock weights, a
    # `.onnx`, or a remote backend that never resizes with these weights at all. Computed by
    # `settings_store.resize_guess` from the same requirement this response already read, so the
    # entry and its absence are one answer rather than two.
    unrecorded_resize_mode: UnrecordedResizeMode | None = None


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
    class_allowlist: list[str] | None = None
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

    @field_validator("class_allowlist")
    @classmethod
    def _validate_class_allowlist(cls, v: list[str] | None) -> list[str] | None:
        # Mirror _valid_field in settings_store: no empty/whitespace entries.
        if v is not None and any(c.strip() == "" for c in v):
            raise ValueError("class_allowlist entries must be non-empty class names")
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


class ClassRecall(BaseModel):
    """One class's recall in the split a validation pass measured.

    `recall` is `None` when the split held no ground-truth instances of the class - the same
    three-state rule the tool's own report uses, and for the same reason. Ultralytics answers
    0.0 there, and 0.0 reads as a total miss, when the truth is that the split never asked:
    "add images of that item" and "add captures to that split" are opposite instructions.
    """

    name: str
    recall: float | None = None
    instances: int = 0


class DistanceRecall(BaseModel):
    """One distance's slice of a split's validation pass: the same classes, a different image set.

    `distance` is a **capture** property, not a class: the class list is the same eight names at
    every distance, and this exists because the per-class floor is computed over all of them
    mixed together. A class whose test frames happen to be mostly `close` can clear 0.85 while
    being unable to find the item at `far`, and the per-class number cannot say so.

    `images` is how many frames this pass ran on, which is what makes the number readable: a
    `far` recall of 1.000 over 3 images and one over 60 are different claims, and the per-class
    `instances` beside each row is a count of *boxes*, not of frames.

    `floor` is deliberately absent - the block these hang under carries the one they were judged
    against, and a copy here could only disagree with it.
    """

    distance: str
    images: int = 0
    aggregates: dict[str, float] = Field(default_factory=dict)
    per_class: list[ClassRecall] = Field(default_factory=list)


class ValidationRecord(BaseModel):
    """What one split's validation pass measured, as recorded beside the weights.

    `split` is carried in the block rather than used as a key because it is what the number
    *means*: `test` is the acceptance split, and `valid` is the one training selected on, so
    the same 0.95 is a different claim depending on which one it came from. The panel labels
    every block, which it could not do from a bare list of numbers.

    `below_floor` is deliberately **not** a field. It is `recall < floor`, derived at render
    time: a stored list could disagree with the numbers beside it, and there would be no way
    to tell which of the two was right.

    `per_distance` is empty rather than absent when a measurement has no breakdown - an older
    record, or a run with `--no-per-distance` - so the panel renders those exactly as it did
    before, and "not measured by distance" never has to be distinguished from "no distances in
    this dataset".
    """

    split: str = ""
    floor: float = 0.0
    measured_at: str = ""
    aggregates: dict[str, float] = Field(default_factory=dict)
    per_class: list[ClassRecall] = Field(default_factory=list)
    per_distance: list[DistanceRecall] = Field(default_factory=list)


class InstalledModel(BaseModel):
    """One weight on disk, plus what is known about running it.

    `resize_mode` is the **requirement** — what these weights were trained to expect — and it
    is `None` when no record was found beside them. Not defaulted to `"auto"`, and deliberately
    not inferred from the filename: a guess and a record would be indistinguishable in the
    panel, and the value of showing it at all is that it is a fact the operator can act on.

    `recorded` and `validation` answer the same question from two sides: whether these weights
    were installed *by the tool* (so there is something they should have been measured with)
    and what that measurement was. A hand-copied weight is false/empty, which is not a fault -
    there is simply nothing to report, and the panel says nothing rather than telling the
    operator to run a command that does not apply to a file it did not install.

    `class_names`/`class_warnings` are the second fact only the record can carry. A `.pt` stores
    the training run, not its label set, so a model whose project class list was split by
    distance (one product in `close`/`mid`/`far`) predicts 24 classes and says nothing about it -
    the failure is a box under a label the app has no use for, on every frame, with no error. The
    probe already catches it at the moment someone tests the connection; recording the names lets
    the listing catch it at the moment the panel opens.

    `auto_resolves_to` is what `auto` gives for this weight, reported rather than left for the
    renderer to reimplement. `auto` is a rule (`settings_store.resolve_resize_mode`), and since
    that rule now *honours* the recorded requirement, this equals `resize_mode` whenever there
    is a record. It earns its place in the unrecorded case, which is the only one where `auto`
    is still a guess rather than a lookup — there it answers the format heuristic (letterbox
    for a `.pt`, stretch for a custom `.onnx`), and the operator needs to see which way that
    guess falls before they trust it.
    """

    value: str
    resize_mode: str | None = None
    auto_resolves_to: str = ""
    # Free text: which dataset version the weights came from, "" when unknown.
    source: str = ""
    # The class list these weights predict, as `train_v2.py --install` recorded it beside them.
    # Empty means *not recorded* - a hand-copied weight, or a record written before the field
    # existed - and never "predicts nothing": `roster.class_list_problems([])` would read as all
    # 8 roster classes missing, so an empty list reports no findings rather than a verdict about
    # a list nothing has seen.
    class_names: list[str] = Field(default_factory=list)
    # What is wrong with `class_names`, judged in the sidecar against the 8-class roster
    # (`app/roster.py`) - the same sentences `DetectorProbeResponse.class_warnings` carries, for
    # the same reason: the names are a property of the weights and this is the only process that
    # can read them. The difference is *when*: this is served by the listing, so a weight whose
    # head was trained per product-and-distance is visible before it is selected, let alone
    # before it runs and logs one item under three labels.
    class_warnings: list[str] = Field(default_factory=list)
    # Whether a record file exists beside these weights at all. Not diagnostic: a weight with
    # no record and one whose record predates `--val` leave every other field empty, and the
    # only useful response to one of those is a command the operator can run.
    recorded: bool = False
    # What `--val` measured, one entry per split, empty when nothing was recorded. A list
    # rather than a single block because the two splits answer different questions and both
    # are worth keeping - see `ValidationRecord`.
    validation: list[ValidationRecord] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    """Weights the operator can select, discovered rather than hardcoded.

    `stock` is the built-in list (the ones that auto-download) and `installed` is what is
    actually on disk under `models/`. They are separate because they are answered by
    different places — the list is a constant in `settings_store`, the directory is a
    filesystem read in `app/models.py` — and the picker needs both: an operator should be
    able to pick a stock model that is not downloaded yet (starting capture downloads it),
    but only a custom model that exists.

    A model that is selected but absent from both lists is still offered by the UI: a
    broken configuration has to be visible to be fixable, and `POST /api/detector/probe`
    is what explains it.
    """

    stock: list[str] = Field(default_factory=list)
    installed: list[InstalledModel] = Field(default_factory=list)
    directory: str = ""


class RecordResizeModeRequest(BaseModel):
    """Body of `POST /api/models/record`: which weights, and the geometry to record beside them.

    `resize_mode` admits only the two real modes. `auto` is rejected deliberately, and not for
    tidiness: it is a *lookup*, not a requirement — `resolve_resize_mode` ignores a recorded
    `auto` — so a record naming it would look like the question had been answered while `auto`
    went on guessing as before.
    """

    model: str
    resize_mode: str

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        if not is_custom_model(v):
            raise ValueError(f"model must be a .pt/.onnx directly under {CUSTOM_MODEL_DIR}")
        return v

    @field_validator("resize_mode")
    @classmethod
    def _validate_recorded_resize_mode(cls, v: str) -> str:
        if v not in ALLOWED_RESIZE_MODES or v == "auto":
            raise ValueError(
                f"resize_mode must be one of {sorted(ALLOWED_RESIZE_MODES - {'auto'})}"
            )
        return v


class DetectorProbeResponse(BaseModel):
    """Result of checking the selected backend before capture starts.

    `provider` is native-only: the onnxruntime execution provider or torch
    device actually backing inference, so an operator can see a silent CPU
    fallback before starting capture.

    `sent_size`/`reported_size` are the remote answer to the question
    `resize_mode_resolved` answers for the native one — which geometry does the model actually see?
    `sent_size` is what the probe transmitted (a capture-shaped frame downscaled by the same rule
    capture uses), `reported_size` is what the workflow said the frame was, and `None` means its
    response carried no size block at all, so the detector fell back to assuming the coordinates
    are relative to what it sent. That is a third state, not a synonym for "they agreed": an
    operator can act on "the workflow re-frames", and on "the workflow told us nothing", but not
    on a silence presented as a match.

    Both are None for `native`, whose geometry is already on `SettingsResponse`
    (`resize_mode_resolved`) and is not something a call has to discover.
    """
    backend: str
    reachable: bool
    detail: str = ""
    latency_ms: float | None = None
    class_names: list[str] = []
    # What is wrong with those names, judged against the app's 8-class roster (`app/roster.py`).
    # Empty means the weight's own class list matches - which is the only way the app can know:
    # a `.pt` records no roster, the filename is a convention, and the export that produced it is
    # long gone. A weight trained from a distance-split project predicts 24 classes and would
    # otherwise run silently, logging one product under three labels.
    class_warnings: list[str] = []
    provider: str | None = None
    sent_size: tuple[int, int] | None = None
    reported_size: tuple[int, int] | None = None


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


class DatasetClassProgress(BaseModel):
    """One class's labeling state.

    The display name arrives inside the snapshot rather than from a table here: the
    roster belongs to the dataset tooling, and the sidecar holding its own copy is
    exactly the kind of hand-synced duplication CLAUDE.md warns about.
    """

    slug: str
    name: str
    decided: int
    total: int


class DatasetTierACell(BaseModel):
    """One Tier A capture cell — what it still needs from the camera.

    This is the capture gap, not the labeling gap: nothing on this row is waiting for a box
    to be drawn, it is waiting for photos to be taken. `target` is the checklist's number and
    `have` is what the project actually holds, both carried in the snapshot rather than
    recomputed here — the capture plan belongs to the dataset tooling, exactly like the
    roster does.
    """

    slug: str
    distance: str
    target: int = 0
    have: int = 0
    decided: int = 0
    remaining: int = 0


class DatasetSessionProgress(BaseModel):
    """One capture session's image spread across the splits.

    `splits` is the leak signal, not decoration: a session appearing in more than one
    split means train and test share a rig state, a day and a lighting setup, which is
    what makes a test number "held-out frames of a session the model trained on" rather
    than an estimate on an unseen session. It is the same reading the split planner
    prints, taken from the project's own tags instead of the manifest.
    """

    name: str
    train: int = 0
    valid: int = 0
    test: int = 0
    decided: int = 0
    total: int = 0
    splits: int = 0


class DatasetBacklogCell(BaseModel):
    """One cell still waiting for a label, with what is left of it.

    The unit of work: a cell is one class at one distance, which is what gets labeled in a
    sitting. `remaining` is `total - decided` computed by the reader, so the count the panel
    shows and the bar/order derived from it cannot disagree with the pair beside them.

    `background` marks the hard-negative pseudo-class, whose frames are marked with the null tool
    rather than drawn on — and whose `distance` is therefore empty, because they are staged
    without one. It is a flag rather than inferred from an empty distance in the renderer: the
    reason these rows are different is a fact about the dataset, not about a string.
    """

    slug: str
    name: str
    distance: str = ""
    decided: int = 0
    total: int = 0
    remaining: int = 0
    background: bool = False


class DatasetStatusResponse(BaseModel):
    """Labeling progress, read from the local snapshot written by
    `sidecar/tools/label_progress.py`.

    Two deliberate choices: `available: false` is a 200 rather than a 404, because
    "nobody has run the tool on this machine yet" is the state this panel starts in and
    the UI renders it as instructions; and `age_seconds` is part of the payload because
    the numbers are only as fresh as the last tool run. The sidecar never calls Roboflow
    for this - see app/dataset_status.py for why.
    """

    available: bool
    snapshot_path: str
    generated_at: str | None = None
    age_seconds: int | None = None
    project: str | None = None
    total: int = 0
    decided: int = 0
    percent: float = 0.0
    null_annotations: int = 0
    mismatches: int = 0
    classes: list[DatasetClassProgress] = Field(default_factory=list)
    by_distance: dict[str, list[int]] = Field(default_factory=dict)
    by_split: dict[str, list[int]] = Field(default_factory=dict)
    sessions: list[DatasetSessionProgress] = Field(default_factory=list)
    # Same counts as `classes`, ordered as work (most remaining first, finished cells dropped)
    # rather than by name. A separate field instead of a sorted `classes` because the two answer
    # different questions — "how is each class doing" and "what do I label next" — and the panel
    # replaced the first with the second rather than reordering one list under both meanings.
    labeling_backlog: list[DatasetBacklogCell] = Field(default_factory=list)
    tier_a_target: int = 0
    tier_a_remaining: int = 0
    tier_a_cells_under_target: int = 0
    tier_a_cells: list[DatasetTierACell] = Field(default_factory=list)
