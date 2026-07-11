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

export interface ApiClient {
  health(): Promise<HealthResponse>
  start(): Promise<StateResponse>
  stop(): Promise<StateResponse>
  getLogs(): Promise<LogsResponse>
}

export function createApiClient(port: number): ApiClient {
  const base = `http://127.0.0.1:${port}/api`

  async function request<T>(path: string, method: 'GET' | 'POST'): Promise<T> {
    const res = await fetch(`${base}${path}`, method === 'GET' ? undefined : { method })
    if (!res.ok) {
      throw new Error(`sidecar ${method} ${path} failed: ${res.status}`)
    }
    return (await res.json()) as T
  }

  return {
    health: () => request<HealthResponse>('/health', 'GET'),
    start: () => request<StateResponse>('/capture/start', 'POST'),
    stop: () => request<StateResponse>('/capture/stop', 'POST'),
    getLogs: () => request<LogsResponse>('/logs', 'GET')
  }
}
