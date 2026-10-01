// Main-process wiring for the POS integration: the real dependencies behind the orchestrator
// (sidecar REST on 127.0.0.1, pushcart-web over the configured base URL), the config store, and
// the "Test connection" read the Admin Panel makes. Electron lives here rather than in
// `posSession.ts`, which stays injectable and testable.

import { BrowserWindow } from 'electron'
import {
  DEFAULT_TRACK_EXPIRY_S,
  insecureBaseUrlWarning,
  isPosEnabled,
  PosConfigStore,
  type PosConfig
} from './posConfig'
import {
  PosConflictError,
  PosSessionOrchestrator,
  PosTransportError,
  type PosConnectionResult,
  type PosSessionDeps,
  type PosState,
  type PosSyncPayload,
  type PosSyncResponse,
  type RemoteSession,
  type SidecarLogsResponse
} from './posSession'

export const POS_STATE_CHANNEL = 'pos:state'

const SIDECAR_TIMEOUT_MS = 6000
const WEBAPP_TIMEOUT_MS = 8000

function trimSlash(url: string): string {
  return url.replace(/\/+$/, '')
}

export class PosController {
  private readonly store: PosConfigStore
  private readonly orchestrator: PosSessionOrchestrator
  /** `null` until the orchestrator reports something: a disabled integration has no state to show. */
  private state: PosState | null = null

  constructor(
    private readonly getPort: () => number | null,
    userDataDir: string
  ) {
    this.store = new PosConfigStore(userDataDir)
    this.orchestrator = new PosSessionOrchestrator(this.buildDeps())
  }

  private buildDeps(): PosSessionDeps {
    return {
      getConfig: () => this.store.load(),
      getHealth: () => this.sidecarJson<{ state: string }>('/api/health'),
      startCapture: () => this.sidecarStartCapture(),
      getLogs: (since) => this.getLogs(since),
      getRemoteSession: (stationId) => this.getRemoteSession(stationId),
      postSync: (payload) => this.postSync(payload),
      emit: (state) => this.setState(state)
    }
  }

  // ---- lifecycle ----
  async start(): Promise<void> {
    await this.orchestrator.start()
  }

  stop(): void {
    this.orchestrator.stop()
  }

  async refreshConfig(): Promise<void> {
    await this.orchestrator.refreshConfig()
  }

  // ---- IPC surface ----
  getState(): PosState | null {
    return this.state
  }

  async getConfig(): Promise<PosConfig> {
    return this.store.load()
  }

  async saveConfig(patch: Partial<PosConfig>): Promise<PosConfig> {
    const trackExpiryS = await this.readTrackExpiryS().catch(() => DEFAULT_TRACK_EXPIRY_S)
    const saved = await this.store.save(patch, trackExpiryS)
    await this.orchestrator.refreshConfig()
    return saved
  }

  async testConnection(): Promise<PosConnectionResult> {
    const config = await this.store.load()
    const warning = insecureBaseUrlWarning(config.posBaseUrl)
    const trackExpiryS = await this.readTrackExpiryS().catch(() => null)

    const base = {
      cartCode: null as string | null,
      sessionRef: null as string | null,
      warning,
      trackExpiryS
    }

    if (!isPosEnabled(config)) {
      return {
        ...base,
        ok: false,
        status: null,
        message: 'Set the base URL, secret and station id first.'
      }
    }

    let response: Response
    try {
      response = await fetch(
        `${trimSlash(config.posBaseUrl)}/api/pos/session?station_id=${encodeURIComponent(config.stationId)}`,
        {
          headers: { 'x-pos-token': config.posSecret },
          signal: AbortSignal.timeout(WEBAPP_TIMEOUT_MS)
        }
      )
    } catch (error) {
      return {
        ...base,
        ok: false,
        status: null,
        message: `Unreachable: ${error instanceof Error ? error.message : String(error)}`
      }
    }

    if (response.status === 401) {
      return {
        ...base,
        ok: false,
        status: 401,
        message: 'Secret rejected — check POS_INGEST_SECRET on the server.'
      }
    }
    if (response.status === 404) {
      return {
        ...base,
        ok: false,
        status: 404,
        message: 'Station not registered — add it on the pushcart-web admin POS screen.'
      }
    }
    if (!response.ok) {
      return {
        ...base,
        ok: false,
        status: response.status,
        message: `Unexpected response (${response.status}).`
      }
    }

    const data =
      ((await response.json().catch(() => null)) as { data?: Record<string, unknown> } | null)
        ?.data ?? null

    if (trackExpiryS !== null && config.commitDwellS <= trackExpiryS) {
      return {
        ...base,
        ok: false,
        status: 200,
        message: `commitDwellS (${config.commitDwellS}s) must exceed the sidecar's track_expiry_s (${trackExpiryS}s).`
      }
    }

    return {
      ...base,
      ok: true,
      status: 200,
      message: data
        ? 'Connected — a session is open at this counter.'
        : 'Connected — no session open right now.',
      cartCode: (data?.cart_code as string | undefined) ?? null,
      sessionRef: (data?.session_ref as string | undefined) ?? null
    }
  }

  // ---- sidecar ----
  private async sidecarJson<T>(path: string, init?: RequestInit): Promise<T> {
    const port = this.getPort()
    if (port === null) throw new Error('sidecar is not running')
    const response = await fetch(`http://127.0.0.1:${port}${path}`, {
      ...init,
      signal: AbortSignal.timeout(SIDECAR_TIMEOUT_MS)
    })
    if (!response.ok) throw new Error(`sidecar ${path} failed (${response.status})`)
    return (await response.json()) as T
  }

  private async sidecarStartCapture(): Promise<{ ok: boolean; status: number }> {
    const port = this.getPort()
    if (port === null) throw new Error('sidecar is not running')
    const response = await fetch(`http://127.0.0.1:${port}/api/capture/start`, {
      method: 'POST',
      signal: AbortSignal.timeout(SIDECAR_TIMEOUT_MS)
    })
    return { ok: response.ok, status: response.status }
  }

  private async getLogs(since: number): Promise<SidecarLogsResponse> {
    const body = await this.sidecarJson<{
      session_id: number | null
      events: Array<{
        track_id: number
        class_name: string
        max_conf: number
        entered_at: number
        left_at: number | null
      }>
    }>(`/api/logs?since=${encodeURIComponent(String(since))}`)
    return {
      sessionId: body.session_id ?? null,
      events: (body.events ?? []).map((event) => ({
        trackId: event.track_id,
        className: event.class_name,
        maxConf: event.max_conf,
        enteredAt: event.entered_at,
        leftAt: event.left_at
      }))
    }
  }

  private async readTrackExpiryS(): Promise<number> {
    const body = await this.sidecarJson<{ track_expiry_s?: number }>('/api/settings')
    return typeof body.track_expiry_s === 'number' ? body.track_expiry_s : DEFAULT_TRACK_EXPIRY_S
  }

  // ---- webapp ----
  private async getRemoteSession(stationId: string): Promise<RemoteSession | null> {
    const config = await this.store.load()
    const response = await fetch(
      `${trimSlash(config.posBaseUrl)}/api/pos/session?station_id=${encodeURIComponent(stationId)}`,
      {
        headers: { 'x-pos-token': config.posSecret },
        signal: AbortSignal.timeout(WEBAPP_TIMEOUT_MS)
      }
    )

    if (response.status === 409) throw new PosConflictError('session conflict')
    if (!response.ok) throw new PosTransportError(`session poll failed (${response.status})`)

    const data = ((await response.json()) as { data?: Record<string, unknown> | null }).data
    if (!data) return null

    return {
      sessionRef: String(data.session_ref),
      cartId: String(data.cart_id),
      cartCode: String(data.cart_code ?? ''),
      cartStatus: String(data.cart_status ?? '')
    }
  }

  private async postSync(payload: PosSyncPayload): Promise<PosSyncResponse> {
    const config = await this.store.load()
    const response = await fetch(`${trimSlash(config.posBaseUrl)}/api/pos/sync`, {
      method: 'POST',
      headers: { 'x-pos-token': config.posSecret, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(WEBAPP_TIMEOUT_MS)
    })

    if (response.status === 409) throw new PosConflictError('session closed or cart paid')
    if (!response.ok) throw new PosTransportError(`sync failed (${response.status})`)

    const data = ((await response.json()) as { data?: PosSyncResponse }).data
    return data ?? {}
  }

  // ---- state push ----
  private setState(state: PosState | null): void {
    this.state = state
    for (const window of BrowserWindow.getAllWindows()) {
      window.webContents.send(POS_STATE_CHANNEL, state)
    }
  }
}
