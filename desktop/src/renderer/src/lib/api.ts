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

export interface SettingsResponse extends SettingsPayload {
  hot_reloadable_fields: string[]
  restart_required_fields: string[]
  warnings: string[]
  // Presence only — the sidecar never sends the key itself.
  roboflow_api_key_present: boolean
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
// before the user starts capture.
export interface DetectorProbeResponse {
  backend: string
  reachable: boolean
  detail: string
  latency_ms: number | null
  class_names: string[]
}

// Mirrors sidecar/app/schemas.py::SettingsUpdateRequest. `reset_fields` names
// settings to set back to null; it exists because the sidecar drops nulls from
// a patch (exclude_none), so "leave the camera alone" cannot travel as a value.
// Only the four camera controls are resettable — see RESETTABLE_FIELDS.
export type SettingsUpdate = Partial<SettingsPayload> & {
  reset_fields?: (keyof SettingsPayload)[]
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
export interface CameraControlSupport {
  brightness: boolean
  exposure: boolean
  gain: boolean
  focus: boolean
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
    applyCameraProfile: () => request<SettingsResponse>('/camera/profile/apply', 'POST')
  }
}
