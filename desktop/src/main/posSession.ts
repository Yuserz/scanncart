// The POS orchestrator (spec §5.3): keeps capture running, binds to whatever session the
// pushcart-web tablet opened on this station, and posts the derived desired cart state.
//
// Capture is always on rather than started per customer — opening the StreamCam takes ~37 s, and
// binding is a timestamp (`bindAt`) instead, so the previous customer's tracks never enter a new
// cart. Every dependency is injected so the whole loop runs against fakes with fake timers.

import { deriveCartState, type CartEntry, type CartStateConfig, type TrackEvent } from './cartState'
import { isPosEnabled, type PosConfig } from './posConfig'

export type PosPhase = 'unbound' | 'warming_up' | 'bound' | 'error'

export interface PosState {
  phase: PosPhase
  cartCode: string | null
  syncedItemCount: number
  lastSyncAgeS: number | null
  error: string | null
}

/** A sidecar `/api/logs` event, camel-cased at the boundary. */
export interface SidecarLogEvent {
  trackId: number
  className: string
  maxConf: number
  enteredAt: number
  leftAt: number | null
}

export interface SidecarLogsResponse {
  sessionId: number | null
  events: SidecarLogEvent[]
}

export interface RemoteSession {
  sessionRef: string
  cartId: string
  cartCode: string
  cartStatus: string
}

export interface PosSyncItem {
  class_name: string
  quantity: number
  max_confidence: number
}

export interface PosSyncPayload {
  session_ref: string
  station_id: string
  items: PosSyncItem[]
}

export interface PosSyncResponse {
  cart_totals?: { item_count: number; subtotal: number }
}

/** The Admin Panel's "Test connection" result. */
export interface PosConnectionResult {
  ok: boolean
  status: number | null
  message: string
  cartCode: string | null
  sessionRef: string | null
  warning: string | null
  trackExpiryS: number | null
}

/** The webapp answered 409: the session is over (closed or the cart is paid). */
export class PosConflictError extends Error {}
/** The webapp could not be reached, or answered with an error that is not a 409. */
export class PosTransportError extends Error {}

export interface PosSessionDeps {
  getConfig: () => Promise<PosConfig>
  getHealth: () => Promise<{ state: string }>
  startCapture: () => Promise<{ ok: boolean; status: number }>
  getLogs: (since: number) => Promise<SidecarLogsResponse>
  getRemoteSession: (stationId: string) => Promise<RemoteSession | null>
  postSync: (payload: PosSyncPayload) => Promise<PosSyncResponse>
  /**
   * The state to show, or `null` for "there is no integration to report" — the feature is off
   * (spec §5.1: an empty URL, secret or station id disables it). `null` is deliberately not the
   * same as an `unbound` phase with nothing in the cart: one means nobody turned this on, the
   * other means someone did and no session is open yet. The Live view renders the first as
   * nothing, the same way it renders a native backend's absent inference verdict.
   */
  emit: (state: PosState | null) => void
}

/** Heartbeat cadence while bound, so the webapp sees liveness even with no change (ms). */
export const SYNC_HEARTBEAT_MS = 15000
/** How often the capture state is checked (ms). */
export const CAPTURE_HEALTH_POLL_MS = 2000
/** Ceiling on the retry backoff while the webapp is unreachable, so a long outage still probes (ms). */
export const RETRY_BACKOFF_CAP_MS = 30_000

const nowSeconds = (): number => Date.now() / 1000

function sameSnapshot(a: Map<string, CartEntry>, b: Map<string, CartEntry>): boolean {
  if (a.size !== b.size) return false
  for (const [key, value] of a) {
    const other = b.get(key)
    if (!other || other.quantity !== value.quantity) return false
  }
  return true
}

export class PosSessionOrchestrator {
  private stopped = true
  private timer: ReturnType<typeof setTimeout> | null = null
  private cfg: PosConfig | null = null

  private bound = false
  private sessionRef: string | null = null
  private cartCode: string | null = null
  private bindAt = 0

  /** Sidecar-session-scoped track cache, keyed `${sessionId}:${trackId}`. */
  private cache = new Map<string, TrackEvent>()
  private logsSessionId: number | null = null
  /** Sidecar seconds of the last successful logs fetch, used to close tracks abandoned by a restart. */
  private lastLogsAtS = 0
  private lastSent = new Map<string, CartEntry>()
  private lastSyncAtMs = 0
  private syncedItemCount = 0

  /** Sidecar-time seconds when capture was (re)started by this orchestrator, for the warm-up guard. */
  private captureRunningSince: number | null = null
  private captureRunning = false
  private restartTried = false

  private lastHealthMs = 0
  private lastSessionPollMs = 0
  private error: string | null = null

  /**
   * Consecutive ticks whose webapp call failed, and whether the tick now running did. This is spec
   * §5.3's "retry with backoff": a webapp that stays unreachable is retried on a doubling interval
   * rather than at the poll cadence, so an outage does not become a request flood. A clean tick
   * resets it to zero, so the cadence comes straight back once the webapp answers.
   */
  private consecutiveFailures = 0
  private tickFailed = false

  /**
   * Whether the tick chain is scheduled or running. It is what keeps *one* chain: `start()` runs
   * once, when the sidecar's port arrives, so a config the Admin Panel saves afterwards has to be
   * able to start the loop itself (`refreshConfig`) without a second chain appearing beside the
   * one already going.
   */
  private looping = false

  constructor(private readonly deps: PosSessionDeps) {}

  private cartCfg(): CartStateConfig {
    const cfg = this.cfg as PosConfig
    return {
      commitDwellS: cfg.commitDwellS,
      removeSettleS: cfg.removeSettleS,
      minCommitConf: cfg.minCommitConf
    }
  }

  async start(): Promise<void> {
    this.stopped = false
    this.cfg = await this.deps.getConfig()
    if (!isPosEnabled(this.cfg)) {
      this.emit()
      return
    }
    if (!this.looping) {
      this.looping = true
      this.schedule(0)
    }
  }

  stop(): void {
    this.stopped = true
    this.looping = false
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
  }

  /** Re-read configuration after the Admin Panel saves it, without dropping the current binding. */
  async refreshConfig(): Promise<void> {
    this.cfg = await this.deps.getConfig()
    this.emit()
    // Turning the feature on at runtime starts the loop here. Nothing else would: the app calls
    // `start()` exactly once, when the sidecar's port arrives, so a station configured afterwards
    // would otherwise be saved, emitted as enabled, and then sit idle until the next launch.
    if (!this.stopped && !this.looping && isPosEnabled(this.cfg)) {
      this.looping = true
      this.schedule(0)
    }
  }

  private schedule(delayMs: number): void {
    if (this.stopped) return
    this.timer = setTimeout(() => {
      this.timer = null
      void this.tick()
    }, delayMs)
  }

  private async tick(): Promise<void> {
    if (this.stopped || !this.cfg || !isPosEnabled(this.cfg)) {
      // Nothing to poll, or the feature was switched off while the chain was running: end it here
      // rather than ticking against a webapp the settings no longer name, and let `refreshConfig`
      // start a new chain if it comes back on.
      this.looping = false
      return
    }
    this.tickFailed = false
    try {
      await this.ensureCapture()
      if (this.bound) {
        if (Date.now() - this.lastSessionPollMs >= this.cfg.sessionPollMs) {
          this.lastSessionPollMs = Date.now()
          await this.pollSession()
        }
        if (this.bound) await this.syncOnce()
      } else {
        this.lastSessionPollMs = Date.now()
        await this.pollSession()
      }
    } catch (error) {
      this.tickFailed = true
      this.setError(error instanceof Error ? error.message : String(error))
    }
    if (this.stopped) return
    // A webapp failure is the only thing that backs the interval off — capture trouble is handled
    // by `ensureCapture`'s own throttle, and a 409 is a session that is over rather than a retry.
    this.consecutiveFailures = this.tickFailed ? this.consecutiveFailures + 1 : 0
    this.schedule(this.retryDelay())
  }

  /**
   * The delay before the next tick: the poll cadence normally, doubling per consecutive failure
   * while the webapp is unreachable, capped so a long outage still probes rather than going silent.
   */
  private retryDelay(): number {
    const cfg = this.cfg as PosConfig
    const base = this.bound ? cfg.logsPollMs : cfg.unboundPollMs
    if (this.consecutiveFailures === 0) return base
    return Math.min(base * 2 ** this.consecutiveFailures, RETRY_BACKOFF_CAP_MS)
  }

  /** Make sure the camera is up; try exactly one auto-restart when it is not. */
  private async ensureCapture(): Promise<void> {
    const nowMs = Date.now()
    if (this.captureRunning && nowMs - this.lastHealthMs < CAPTURE_HEALTH_POLL_MS) return
    this.lastHealthMs = nowMs

    const health = await this.deps.getHealth()
    if (health.state === 'running' || health.state === 'starting') {
      this.captureRunning = true
      this.restartTried = false
      return
    }

    // Any other state while POS is enabled means capture is down (a pipeline error tears down to idle).
    this.captureRunning = false
    if (this.restartTried) {
      this.setError('capture is down — corrections happen on the tablet')
      return
    }

    const result = await this.deps.startCapture()
    if (result.status === 409) {
      // Calibration is in progress; wait for it rather than counting it as the failed restart.
      return
    }
    this.restartTried = true
    if (result.ok) {
      // Blind for the restart: hold the cart at its last counts until the camera is warm again.
      this.captureRunningSince = nowSeconds()
      this.captureRunning = true
      this.error = null
      this.emit()
    } else {
      this.setError('capture failed to restart')
    }
  }

  private async pollSession(): Promise<void> {
    if (!this.cfg) return
    let remote: RemoteSession | null
    try {
      remote = await this.deps.getRemoteSession(this.cfg.stationId)
    } catch (error) {
      this.tickFailed = true
      this.setError(error instanceof Error ? error.message : String(error))
      return
    }

    if (!remote || remote.cartStatus === 'paid') {
      if (this.bound) this.unbind()
      return
    }

    if (!this.bound || remote.sessionRef !== this.sessionRef) {
      this.bind(remote)
    }
  }

  private bind(remote: RemoteSession): void {
    this.bound = true
    this.sessionRef = remote.sessionRef
    this.cartCode = remote.cartCode
    this.bindAt = nowSeconds()
    this.cache.clear()
    this.lastSent.clear()
    this.logsSessionId = null
    // The heartbeat clock starts at binding, so an empty first snapshot is not posted as if
    // the 15 s had already elapsed.
    this.lastSyncAtMs = Date.now()
    this.syncedItemCount = 0
    this.error = null
    this.emit()
  }

  /** Stop syncing but leave capture running (the customer tapped Finish). */
  private unbind(): void {
    this.bound = false
    this.sessionRef = null
    this.cartCode = null
    this.cache.clear()
    this.lastSent.clear()
    this.logsSessionId = null
    this.syncedItemCount = 0
    this.emit()
  }

  private isWarmingUp(now: number): boolean {
    if (!this.captureRunning) return true
    if (this.captureRunningSince === null) return false
    const cfg = this.cartCfg()
    return now < this.captureRunningSince + cfg.commitDwellS + cfg.removeSettleS
  }

  /**
   * One sync attempt per tick, and the tick's own loop is what keeps a request in flight from being
   * overlapped: `tick` awaits this before scheduling the next one, and a change that arrives while
   * a post is out is picked up by re-deriving from the cache on the next pass rather than by
   * starting a second request. That is the spec's "a change replaces the queued snapshot" read
   * literally — the queued snapshot is the cache, which the next derivation always reflects.
   */
  private async syncOnce(): Promise<void> {
    if (!this.cfg || !this.sessionRef) return
    const logSince = this.bindAt
    const logs = await this.deps.getLogs(logSince)
    const fetchedAtS = nowSeconds()

    if (logs.sessionId !== null) {
      // A restart changes the sidecar session. Tracks the old session still had open are closed
      // at the last time the sidecar confirmed them, so they expire through the settle window
      // instead of counting forever (the restart itself is ~37 s of blindness).
      if (this.logsSessionId !== null && logs.sessionId !== this.logsSessionId) {
        const closedAt = this.lastLogsAtS || fetchedAtS
        for (const cached of this.cache.values()) {
          if (cached.sessionId !== logs.sessionId && cached.leftAt === null) {
            cached.leftAt = closedAt
          }
        }
        // The camera restarted: warm up from now, so the blind gap cannot empty the cart.
        this.captureRunningSince = fetchedAtS
      }
      this.logsSessionId = logs.sessionId
    }
    this.lastLogsAtS = fetchedAtS

    const sessionId = this.logsSessionId ?? 0
    for (const event of logs.events) {
      this.cache.set(`${sessionId}:${event.trackId}`, {
        sessionId,
        trackId: event.trackId,
        className: event.className,
        maxConf: event.maxConf,
        enteredAt: event.enteredAt,
        leftAt: event.leftAt
      })
    }

    const now = nowSeconds()
    const derived = deriveCartState([...this.cache.values()], now, this.bindAt, this.cartCfg())
    const warmingUp = this.isWarmingUp(now)

    const snapshot = new Map<string, CartEntry>()
    const classes = new Set<string>([...derived.keys()])
    if (warmingUp) for (const key of this.lastSent.keys()) classes.add(key)

    for (const className of classes) {
      const fromDerived = derived.get(className)
      const held = this.lastSent.get(className)
      const quantity = warmingUp
        ? Math.max(fromDerived?.quantity ?? 0, held?.quantity ?? 0)
        : (fromDerived?.quantity ?? 0)
      if (quantity <= 0) continue
      const maxConfidence = Math.max(
        fromDerived?.maxConfidence ?? 0,
        warmingUp ? (held?.maxConfidence ?? 0) : 0
      )
      snapshot.set(className, { quantity, maxConfidence })
    }

    const changed = !sameSnapshot(snapshot, this.lastSent)
    const due = Date.now() - this.lastSyncAtMs >= SYNC_HEARTBEAT_MS
    if (!changed && !due) return

    await this.postSync(snapshot)
  }

  private async postSync(snapshot: Map<string, CartEntry>): Promise<void> {
    if (!this.cfg || !this.sessionRef) return
    const items: PosSyncItem[] = [...snapshot].map(([className, entry]) => ({
      class_name: className,
      quantity: entry.quantity,
      max_confidence: entry.maxConfidence
    }))

    try {
      const response = await this.deps.postSync({
        session_ref: this.sessionRef,
        station_id: this.cfg.stationId,
        items
      })
      this.lastSent = snapshot
      this.lastSyncAtMs = Date.now()
      this.syncedItemCount =
        response.cart_totals?.item_count ?? items.reduce((sum, item) => sum + item.quantity, 0)
      this.error = null
      this.emit()
    } catch (error) {
      if (error instanceof PosConflictError) {
        // A 409 means the session is over: Finish won, or the cart is paid.
        this.unbind()
        return
      }
      // Webapp unreachable: keep the last state and surface the error. The tick is marked failed
      // so the loop retries on the backoff rather than at the poll cadence.
      this.tickFailed = true
      this.setError(error instanceof Error ? error.message : String(error))
    }
  }

  private setError(message: string): void {
    this.error = message
    this.emit()
  }

  private emit(): void {
    // A disabled integration reports `null` rather than an idle-looking state, so the Live view
    // shows no panel for a feature nobody turned on (see `PosSessionDeps.emit`).
    if (!this.cfg || !isPosEnabled(this.cfg)) {
      this.deps.emit(null)
      return
    }
    this.deps.emit({
      phase: this.currentPhase(),
      cartCode: this.cartCode,
      syncedItemCount: this.syncedItemCount,
      lastSyncAgeS: this.lastSyncAtMs ? (Date.now() - this.lastSyncAtMs) / 1000 : null,
      error: this.error
    })
  }

  private currentPhase(): PosPhase {
    if (this.error) return 'error'
    if (!this.bound) return 'unbound'
    return this.isWarmingUp(nowSeconds()) ? 'warming_up' : 'bound'
  }
}
