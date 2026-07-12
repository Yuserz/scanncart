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
  infer_frame_skip: number
  device: string
  preview_height: number
  track_expiry_s: number
}

export interface SettingsResponse extends SettingsPayload {
  hot_reloadable_fields: string[]
  restart_required_fields: string[]
  warnings: string[]
}

export type SettingsUpdate = Partial<SettingsPayload>

export interface SystemInfoResponse {
  cpu_count: number
  ram_gb: number
  cuda_available: boolean
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
  updateSettings(patch: SettingsUpdate): Promise<SettingsResponse>
  getSystemInfo(): Promise<SystemInfoResponse>
  getPresets(): Promise<PresetsResponse>
  applyPreset(name: string): Promise<SettingsResponse>
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
    updateSettings: (patch) => request<SettingsResponse>('/settings', 'PATCH', patch),
    getSystemInfo: () => request<SystemInfoResponse>('/system-info', 'GET'),
    getPresets: () => request<PresetsResponse>('/presets', 'GET'),
    applyPreset: (name) => request<SettingsResponse>('/settings/preset', 'POST', { name })
  }
}
