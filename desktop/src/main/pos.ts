// Main-process wiring for the POS integration: the real dependencies behind the orchestrator
// (sidecar REST on 127.0.0.1, pushcart-web over the configured base URL), the config store, and
// the "Test connection" read the Admin Panel makes. Electron lives here rather than in
// `posSession.ts`, which stays injectable and testable.

import { BrowserWindow } from 'electron'
import {
  fetchRemoteSession,
  postCartSync,
  probePosSession,
  type PosSessionProbe
} from './posClient'
import {
  DEFAULT_TRACK_EXPIRY_S,
  insecureBaseUrlWarning,
  isPosEnabled,
  PosConfigStore,
  type PosConfig
} from './posConfig'
import {
  PosSessionOrchestrator,
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

    const probe: PosSessionProbe = await probePosSession(
      config.posBaseUrl,
      config.posSecret,
      config.stationId
    )

    if (probe.unreachable !== null) {
      return {
        ...base,
        ok: false,
        status: null,
        message: `Unreachable: ${probe.unreachable}`
      }
    }

    if (probe.status === 401) {
      return {
        ...base,
        ok: false,
        status: 401,
        message: 'Secret rejected — check POS_INGEST_SECRET on the server.'
      }
    }
    if (probe.status === 404) {
      return {
        ...base,
        ok: false,
        status: 404,
        message: 'Station not registered — add it on the pushcart-web admin POS screen.'
      }
    }
    if (probe.status === null || probe.status >= 400) {
      return {
        ...base,
        ok: false,
        status: probe.status,
        message: `Unexpected response (${probe.status}).`
      }
    }

    const data = probe.data

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

  // ---- webapp: the routes themselves live in `posClient.ts`, so they are drivable over a socket
  // in `posRoutes.integration.test.ts`; here they are only given the stored connection details.
  private async getRemoteSession(stationId: string): Promise<RemoteSession | null> {
    const config = await this.store.load()
    return fetchRemoteSession(config.posBaseUrl, config.posSecret, stationId)
  }

  private async postSync(payload: PosSyncPayload): Promise<PosSyncResponse> {
    const config = await this.store.load()
    return postCartSync(config.posBaseUrl, config.posSecret, payload)
  }

  // ---- state push ----
  private setState(state: PosState | null): void {
    this.state = state
    for (const window of BrowserWindow.getAllWindows()) {
      window.webContents.send(POS_STATE_CHANNEL, state)
    }
  }
}
