// REST client for the SCANnCART sidecar. The renderer talks to the sidecar
// directly over localhost HTTP; see the Phase 2 plan for the contract (spec §4.2).

export interface HealthResponse {
  state: string
  active_model: string
  device: string
}

export interface StateResponse {
  state: string
}

export interface LogEvent {
  track_id: number
  class_name: string
  confidence: number
  max_conf: number
  entered_at: number
  left_at: number | null
}

export interface LogsResponse {
  session_id: number | null
  events: LogEvent[]
}

// Mirrors sidecar/app/settings.py::Settings 1:1 — keep in sync by hand, same
// as the WS message contract (see CLAUDE.md's testing conventions note).
export interface SettingsPayload {
  active_model: string
  camera_index: number
  capture_width: number
  capture_height: number
  capture_fps: number
  conf_threshold: number
  imgsz: number
  resize_mode: string
  infer_frame_skip: number
  device: string
  preview_height: number
  preview_max_fps: number
  track_expiry_s: number
  class_allowlist: string[]
  detector_backend: string
  roboflow_workspace: string
  roboflow_workflow_id: string
  local_api_url: string
  cloud_api_url: string
  remote_infer_size: number
  remote_timeout_s: number
  remote_max_retries: number
  // Device controls. null means "leave the camera alone" — calibration owns
  // these, they are not hand-editable in the settings form.
  camera_brightness: number | null
  camera_exposure: number | null
  camera_autofocus: boolean | null
  camera_focus: number | null
}

// Weights whose training geometry nothing recorded, with the sentence saying so and the mode a
// one-click record would write. A structured entry rather than a line in `warnings` because it is
// the one resize case the app can settle by itself: `recordResizeMode` writes
// `models/<stem>.json`, so the panel needs the weight and the mode, and a button cannot read
// either out of prose.
//
// Mirrors sidecar/app/schemas.py::UnrecordedResizeMode. `null` whenever there is no assumption to
// report — a record, an explicit mode, stock weights, a `.onnx`, or a remote backend that never
// resizes with these weights at all.
export interface UnrecordedResizeMode {
  model: string
  // What recording would write, which is `auto`'s answer for these weights — i.e. the geometry
  // already running. Recording it cannot change what the model does; it turns the fallback into a
  // stated fact.
  resize_mode: string
  // The diagnosis, from the sidecar, so neither view holds a copy of it that could describe a
  // different situation than the entry beside it.
  warning: string
  // How to stop assuming, from the same place and for the same reason. Separate from `warning`
  // because two views render this entry and only one of them (the Admin Panel) holds the button
  // that performs the fix: the Live view surfaces the assumption while capture runs, and prose
  // addressed to "the button below" would point at a control that view does not have.
  remedy: string
}

export interface SettingsResponse extends SettingsPayload {
  hot_reloadable_fields: string[]
  restart_required_fields: string[]
  warnings: string[]
  // Presence only — the sidecar never sends the key itself.
  roboflow_api_key_present: boolean
  // What `resize_mode` resolves to for the selected weights, `auto` already resolved — from the
  // same `resolve_resize_mode` call the detector factory makes, so it is the geometry that is
  // running rather than a re-derivation of it here. The Live view prints it because the setting
  // alone does not answer the question: `auto` depends on the weights' record, and on the file
  // format for weights with none, neither of which the renderer holds.
  //
  // `null` when the backend does not run these weights at all — a remote backend hands the frame
  // to a workflow with its own model — and the view reads that absence as "no local weights to
  // describe" rather than printing a geometry for a resize that never happens.
  resize_mode_resolved: string | null
  // The geometry `auto` *assumes* because nothing is recorded beside the selected weights, or
  // `null`. Non-null is what the Admin panel's "record it now" button is for, and the entry
  // carries everything that button needs.
  unrecorded_resize_mode: UnrecordedResizeMode | null
}

// One enumerated capture device. width/height are what the device actually
// opened at — the operator's check that `name` landed on the right index,
// since the sidecar pairs names to indices positionally.
export interface CameraInfo {
  index: number
  name: string
  width: number
  height: number
}

export interface CamerasResponse {
  cameras: CameraInfo[]
  // False means the list is cached, not a fresh scan: probing opens each
  // device, so it is skipped while capture holds one.
  probed: boolean
  detail: string
}

// Result of POST /api/detector/probe: checks the selected backend is usable
// before the user starts capture. `provider` is native-only: the onnxruntime
// execution provider or torch device actually backing inference (e.g.
// "CUDAExecutionProvider" vs "CPUExecutionProvider"), so a silent CPU
// fallback is visible before capture starts.
export interface DetectorProbeResponse {
  backend: string
  reachable: boolean
  detail: string
  latency_ms: number | null
  class_names: string[]
  // What is wrong with `class_names`, judged in the sidecar against the 8-class roster
  // (`app/roster.py`) — empty means the weight's own class list matches. A weight trained from a
  // distance-split project predicts 24 classes and would otherwise run silently, logging one
  // product under three labels; an empty list here is the *only* place that fact can be known,
  // because a `.pt` records no roster and the export that produced it is long gone.
  class_warnings: string[]
  provider: string | null
  // The remote answer to the question `resize_mode_resolved` answers for native weights: which
  // geometry does the model actually see? `sent_size` is what this probe transmitted (a
  // capture-shaped frame, downscaled by the rule capture uses), and `reported_size` is what the
  // workflow said the frame was — the canvas its coordinates are relative to.
  //
  // `reported_size: null` is the third state, and not a synonym for "they agreed": the response
  // carried no dimensions, so the detector fell back to assuming the coordinates are relative to
  // what it sent. An operator can act on "the workflow re-frames" and on "the workflow told us
  // nothing"; a silence rendered as a match is the one reading that misleads.
  //
  // Both are null for `native`, whose geometry is already on `SettingsResponse`. Mirrors
  // sidecar/app/schemas.py::DetectorProbeResponse.
  sent_size: [number, number] | null
  reported_size: [number, number] | null
}

// Mirrors sidecar/app/schemas.py::SettingsUpdateRequest. `reset_fields` names
// settings to set back to null; it exists because the sidecar drops nulls from
// a patch (exclude_none), so "leave the camera alone" cannot travel as a value.
// Only the four camera controls are resettable — see RESETTABLE_FIELDS.
export type SettingsUpdate = Partial<SettingsPayload> & {
  reset_fields?: (keyof SettingsPayload)[]
}

// One class's labeling progress. `name` is the exact class name to create in
// Roboflow, and it arrives inside the snapshot rather than from settingsFields.ts:
// the dataset roster belongs to the tooling, and the sidecar deliberately carries
// no copy of it to drift out of date.
export interface DatasetClassProgress {
  slug: string
  name: string
  decided: number
  total: number
}

// One Tier A capture cell: the *capture* gap rather than the labeling gap. Nothing on
// this row is waiting for a box to be drawn, it is waiting for photos. `target` is the
// checklist's number and `have` is what the project holds, both carried in the snapshot —
// the capture plan belongs to the dataset tooling, like the roster, and the sidecar keeps
// no second copy of it to drift.
export interface DatasetTierACell {
  slug: string
  distance: string
  target: number
  have: number
  decided: number
  remaining: number
}

// One capture session's image spread across the splits. `splits` is the leak signal:
// a session in more than one split means train and test share a rig state, a day and a
// lighting setup, so a test reading is held-out frames of a session the model trained
// on rather than an estimate on an unseen one. Derived by the sidecar from the three
// counts, so it cannot disagree with the numbers rendered beside it.
export interface DatasetSessionProgress {
  name: string
  train: number
  valid: number
  test: number
  decided: number
  total: number
  splits: number
}

// One cell still waiting for a label: one class at one distance, which is the unit of
// work — a cell is what gets labeled in one sitting.
//
// `remaining` is `total - decided`, computed by the sidecar, so the count shown and the
// bar drawn from it cannot disagree with the pair beside them.
//
// `background` marks the hard-negative pseudo-class, whose frames are marked with the
// annotator's null tool rather than drawn on — the one row where "label this" means the
// opposite of what it means everywhere else. It is a flag rather than a test on an empty
// `distance`, because the reason these rows differ is a fact about the dataset and not
// about a string: `distance` is empty only because those frames are staged without one.
// Mirrors sidecar/app/schemas.py::DatasetBacklogCell.
export interface DatasetBacklogCell {
  slug: string
  name: string
  distance: string
  decided: number
  total: number
  remaining: number
  background: boolean
}

// Mirrors sidecar/app/schemas.py::DatasetStatusResponse.
//
// `available: false` is a normal 200, not an error: it means nobody has run
// sidecar/tools/label_progress.py on this machine yet, so the panel renders the
// instructions instead of a misleading 0%.
//
// There is deliberately no polling for this. The snapshot only changes when the tool
// is run by hand, so a timer would fetch identical bytes forever; the panel offers an
// explicit refresh instead. `age_seconds` is what keeps that honest — the numbers are
// only as new as the last tool run, and the UI says so rather than implying "live".
export interface DatasetStatusResponse {
  available: boolean
  snapshot_path: string
  generated_at: string | null
  age_seconds: number | null
  project: string | null
  total: number
  decided: number
  percent: number
  null_annotations: number
  mismatches: number
  classes: DatasetClassProgress[]
  // The worklist: the same counts as `classes`, ordered as work (most remaining first) with
  // finished cells dropped, and with a distance axis the per-class view cannot express. The
  // panel renders this instead of a per-class table — the two answer different questions
  // ("how is each class doing" vs. "what do I label next") and this is the one a labeling
  // session asks. `classes` is still sent because the Tier A rows name a class by slug, and
  // this is the only place that translates one.
  labeling_backlog: DatasetBacklogCell[]
  by_distance: Record<string, number[]>
  by_split: Record<string, number[]>
  sessions: DatasetSessionProgress[]
  tier_a_target: number
  tier_a_remaining: number
  tier_a_cells_under_target: number
  tier_a_cells: DatasetTierACell[]
}

// One class's recall in the split a validation pass measured.
//
// `recall` is `null` when the split held no ground-truth instances of the class — not 0, which
// would read as a total miss. The two call for opposite work (more images of that item vs. more
// captures in that split), and the sidecar's own report keeps them apart for the same reason.
// Mirrors sidecar/app/schemas.py::ClassRecall.
export interface ClassRecall {
  name: string
  recall: number | null
  instances: number
}

// What one split's validation pass measured, as recorded beside the weights.
//
// `split` names what the number means — `test` is the acceptance split, `valid` is the one
// training selected on — which is why it is carried in the block rather than used as a key.
// Below-floor classes are *not* listed here: that is `recall < floor`, derived where it is
// rendered, so a stored list cannot disagree with the numbers beside it.
//
// One distance's slice of a split's validation pass: the same classes, a different image set.
//
// `distance` is a *capture* property, not a class — the class list is the same names at every
// distance. This exists because the per-class floor is computed over all of them mixed together,
// so a class whose test frames happen to be mostly `close` can clear the floor while being unable
// to find the item at `far`.
//
// `images` is how many frames the pass ran on, which is what makes the number readable; the
// `instances` on each class row counts *boxes*, not frames. There is deliberately no `floor`:
// the block these hang under carries the one they were judged against.
export interface DistanceRecall {
  distance: string
  images: number
  aggregates: Record<string, number>
  per_class: ClassRecall[]
}

// Mirrors sidecar/app/schemas.py::ValidationRecord.
export interface ValidationRecord {
  split: string
  floor: number
  measured_at: string
  aggregates: Record<string, number>
  per_class: ClassRecall[]
  // Empty rather than absent when a measurement has no breakdown — an older record, or a run
  // with `--no-per-distance` — so those render exactly as they did before, and "not measured by
  // distance" never has to be told apart from "this dataset has no distances".
  per_distance: DistanceRecall[]
}

// One weight that exists under sidecar/models/, plus what is known about running it.
//
// `resize_mode` is the *requirement* — what these weights were trained to expect — and it is
// `null` when nothing was recorded beside them. It is never inferred here: a guess would be
// indistinguishable from a fact in the UI that is meant to be acted on. `auto_resolves_to`
// is what the sidecar's `auto` gives for this weight, reported so this side can compare the
// two without holding a second copy of the resolution rule. Since that rule honours the
// record, it equals `resize_mode` whenever there is one; it earns its place in the unrecorded
// case, where `auto` is still a format-based guess and the operator needs to see which way it
// falls before trusting it.
//
// Mirrors sidecar/app/schemas.py::InstalledModel.
export interface InstalledModel {
  value: string
  resize_mode: string | null
  auto_resolves_to: string
  source: string
  // The class list these weights predict, as recorded beside them by `train_model.py --install`.
  // Empty means *not recorded* — a hand-copied weight, or a record written before the field
  // existed — and never "predicts nothing", so an empty list produces no findings below.
  class_names: string[]
  // What is wrong with `class_names`, judged in the sidecar against the 8-class roster, the same
  // sentences `DetectorProbeResponse.class_warnings` carries. The difference is *when*: these come
  // with the listing, so a weight whose head was trained per product-and-distance is visible
  // before it is selected — let alone before it runs and logs one item under three labels. The
  // sentences are rendered verbatim; the roster is not mirrored here on purpose.
  class_warnings: string[]
  // Whether a record file exists beside these weights at all. It separates "installed by the
  // tool, nothing measured yet" (fixable, by running `--val`) from "copied into models/ by
  // hand, nothing known" (nothing to report) — two states whose other fields are identical,
  // and only one of which is worth a message.
  recorded: boolean
  // What `--val` measured on this weight, one entry per split, empty when nothing was
  // recorded. An empty list is "not measured" — which is the normal state for a hand-copied
  // weight or a baseline kept from an earlier generation — and never a score of zero.
  validation: ValidationRecord[]
}

// Weights the operator can select. `stock` are the built-in models that auto-download on
// first use; `installed` is what actually exists under sidecar/models/. The picker needs
// both because they behave differently: a stock model can be chosen before it is downloaded,
// an installed one has to be on disk.
//
// Mirrors sidecar/app/schemas.py::ModelsResponse. The built-in *labels* stay in
// settingsFields.ts — this is the values, so a newly trained model needs no renderer edit.
export interface ModelsResponse {
  stock: string[]
  installed: InstalledModel[]
  directory: string
}

export interface CameraQualityResponse {
  available: boolean
  brightness: number
  contrast: number
  sharpness: number
  capture_fps: number
  target_fps: number
  verdicts: Record<string, string>
  detail: string
}

// Which physical device controls the camera actually accepted during
// calibration (a StreamCam commonly lacks gain/focus control, for instance).
// Mirrors sidecar/app/camera_caps.py::ControlSupport. `autofocus` is measured
// by probe_autofocus, not asked of the driver.
export interface CameraControlSupport {
  brightness: boolean
  exposure: boolean
  gain: boolean
  focus: boolean
  autofocus: boolean
}

// One entry of CameraProfile.measured: what the sweep found for one control.
// `baseline` is the metric before the sweep, so the card can show the
// improvement rather than a bare number.
export interface MeasuredControl {
  value: number
  metric: number
  baseline: number
  reached: boolean
  probes: number
}

// Mirrors sidecar/app/schemas.py::CameraProfileResponse. Applies nothing on
// its own — POST /api/camera/calibrate only measures; the operator reviews
// fps_auto_exposure vs fps_capped_exposure (the evidence) before choosing to
// apply `recommended` via POST /api/camera/profile/apply.
export interface CameraProfileResponse {
  device_key: string
  backend: string
  width: number
  height: number
  fps_auto_exposure: number
  fps_capped_exposure: number
  controls: CameraControlSupport
  recommended: Record<string, unknown>
  measured_at: number
  measured: Record<string, MeasuredControl>
  // 0 = calibrated before levels were measured, not "nothing is supported".
  sweep_version: number
}

// Mirrors sidecar/app/schemas.py::StoredProfileResponse. `profile` is null
// for a camera that has never been calibrated — a normal state, not an error.
export interface StoredProfileResponse {
  profile: CameraProfileResponse | null
}

export interface SystemInfoResponse {
  cpu_count: number
  ram_gb: number
  cuda_available: boolean
  accelerator: 'cuda' | 'integrated' | 'cpu'
  gpu_name: string | null
  gpu_vram_gb: number | null
  recommended_preset: string
}

export interface PresetInfo {
  name: string
  label: string
  description: string
  settings: SettingsUpdate
}

export interface PresetsResponse {
  presets: PresetInfo[]
  recommended: string
}

export interface ApiClient {
  health(): Promise<HealthResponse>
  start(): Promise<StateResponse>
  stop(): Promise<StateResponse>
  getLogs(): Promise<LogsResponse>
  getSettings(): Promise<SettingsResponse>
  // persist=false applies the change to the running camera/detector without
  // writing settings.json — the Live tab's tuning card drags sliders through
  // this, then commits once via saveSettings().
  updateSettings(patch: SettingsUpdate, persist?: boolean): Promise<SettingsResponse>
  saveSettings(): Promise<SettingsResponse>
  // The stored calibration for the currently configured camera, or
  // { profile: null } if it has never been calibrated.
  getCameraProfile(): Promise<StoredProfileResponse>
  getSystemInfo(): Promise<SystemInfoResponse>
  getPresets(): Promise<PresetsResponse>
  applyPreset(name: string): Promise<SettingsResponse>
  probeDetector(): Promise<DetectorProbeResponse>
  // rescan re-opens every device (slow); omit it to take the cached list.
  getCameras(rescan?: boolean): Promise<CamerasResponse>
  getCameraQuality(): Promise<CameraQualityResponse>
  // Measures the camera; applies nothing. 409 while capture is running (the
  // camera is exclusive) — the caller disables the button in that state.
  calibrateCamera(): Promise<CameraProfileResponse>
  // Applies the most recently calibrated profile's `recommended` patch. 404
  // if nothing has been calibrated yet.
  applyCameraProfile(): Promise<SettingsResponse>
  // Dataset labeling progress, read from a local snapshot. Never hits Roboflow
  // and needs no API key — see sidecar/app/dataset_status.py.
  getDatasetStatus(): Promise<DatasetStatusResponse>
  // Which weights exist and what each one needs, so the Model picker is not a fixed list
  // and can say when the resize_mode setting does not match. Never triggers a download
  // (unlike selecting a stock model) and never probes a model.
  getModels(): Promise<ModelsResponse>
  // Writes `resize_mode` into the record beside these weights — the operator attesting to how
  // they were trained, so `auto` stops guessing for them. Answers with the refreshed weights
  // list, which is where the change shows; 404 if there are no such weights on disk, 422 for a
  // stock name or an `auto` requirement.
  recordResizeMode(model: string, resizeMode: string): Promise<ModelsResponse>
}

export function createApiClient(port: number): ApiClient {
  const base = `http://127.0.0.1:${port}/api`

  async function request<T>(
    path: string,
    method: 'GET' | 'POST' | 'PATCH',
    body?: unknown
  ): Promise<T> {
    const init: RequestInit | undefined =
      method === 'GET'
        ? undefined
        : {
            method,
            ...(body !== undefined
              ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
              : {})
          }
    const res = await fetch(`${base}${path}`, init)
    if (!res.ok) {
      // The sidecar puts the actionable half of every refusal in `detail` —
      // "Calibration is in progress; the camera is exclusive", "Cannot change
      // [...] while capture is running; stop capture first". Throwing the bare
      // status discarded exactly the sentence that tells an operator what to
      // do, leaving them with "failed: 409". FastAPI's own validation errors
      // put a list there instead, which is for us, not them — fall back.
      let detail: unknown
      try {
        detail = (await res.json())?.detail
      } catch {
        // Not a JSON body; the status line is all there is.
      }
      if (typeof detail === 'string' && detail !== '') throw new Error(detail)
      throw new Error(`sidecar ${method} ${path} failed: ${res.status}`)
    }
    return (await res.json()) as T
  }

  return {
    health: () => request<HealthResponse>('/health', 'GET'),
    start: () => request<StateResponse>('/capture/start', 'POST'),
    stop: () => request<StateResponse>('/capture/stop', 'POST'),
    getLogs: () => request<LogsResponse>('/logs', 'GET'),
    getSettings: () => request<SettingsResponse>('/settings', 'GET'),
    updateSettings: (patch, persist = true) =>
      request<SettingsResponse>(`/settings${persist ? '' : '?persist=false'}`, 'PATCH', patch),
    saveSettings: () => request<SettingsResponse>('/settings/save', 'POST'),
    getCameraProfile: () => request<StoredProfileResponse>('/camera/profile', 'GET'),
    getSystemInfo: () => request<SystemInfoResponse>('/system-info', 'GET'),
    getPresets: () => request<PresetsResponse>('/presets', 'GET'),
    applyPreset: (name) => request<SettingsResponse>('/settings/preset', 'POST', { name }),
    probeDetector: () => request<DetectorProbeResponse>('/detector/probe', 'POST'),
    getCameras: (rescan) =>
      request<CamerasResponse>(`/cameras${rescan ? '?rescan=true' : ''}`, 'GET'),
    getCameraQuality: () => request<CameraQualityResponse>('/camera/quality', 'GET'),
    calibrateCamera: () => request<CameraProfileResponse>('/camera/calibrate', 'POST'),
    applyCameraProfile: () => request<SettingsResponse>('/camera/profile/apply', 'POST'),
    getDatasetStatus: () => request<DatasetStatusResponse>('/dataset/status', 'GET'),
    getModels: () => request<ModelsResponse>('/models', 'GET'),
    recordResizeMode: (model, resizeMode) =>
      request<ModelsResponse>('/models/record', 'POST', {
        model,
        resize_mode: resizeMode
      })
  }
}
