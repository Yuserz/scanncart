// The POS orchestrator (spec §5.3): keeps capture running, binds to whatever session the
// pushcart-web tablet opened on this station, and posts the derived desired cart state —
// merged with the session's cumulative floor, because D2's automatic removal is suspended until
// the next model can tell a taken item from a lost track.
//
// Capture is always on rather than started per customer — opening the StreamCam takes ~37 s, and
// binding is a timestamp (`bindAt`) instead, so the previous customer's tracks never enter a new
// cart. Every dependency is injected so the whole loop runs against fakes with fake timers.

import type { ReviewItem } from './basketLedger'
import {
  deriveCartState,
  deriveLiveCounts,
  type CartEntry,
  type CartStateConfig,
  type TrackEvent
} from './cartState'
import { isPosEnabled, type CartMode, type PosConfig } from './posConfig'
import type { BasketReadout } from './transferStream'

export type PosPhase = 'unbound' | 'warming_up' | 'bound' | 'error'

export interface PosState {
  phase: PosPhase
  cartCode: string | null
  syncedItemCount: number
  lastSyncAgeS: number | null
  error: string | null
  /**
   * The deadline of the standing retry backoff, or null when the tick cadence is healthy. A
   * deadline rather than seconds-remaining, so a push is a transition (backoff began, backoff
   * ended, next attempt moved) rather than a message every second; the renderer counts it down
   * locally. It is set only at the tick tail, only while `consecutiveFailures` stands above zero.
   */
  retryAtMs: number | null
  /**
   * When pushcart-web last answered this station (a session poll or a sync that came back), or null
   * when it never has since launch. An `unbound` phase alone cannot say "ready": it is also what the
   * loop reports before its first request has returned, so for the first seconds against a server
   * that is down the Admin Panel read "Ready" over a connection nothing had proven.
   */
  lastContactMs: number | null
  /** Which derivation the posted cart follows (`posConfig.cartMode`). */
  cartMode: CartMode
  /**
   * The transfer ledger's view, or null when no basket tracker is wired. In `counter` mode this
   * is the shadow readout — what the basket *would* post — shown beside the posted count so the
   * two can be compared before anyone switches modes.
   */
  basket: BasketReadout | null
}

/**
 * The basket tracker as the orchestrator sees it (`transferStream.BasketTracker`). Bound and
 * unbound with the session in both modes, so the shadow ledger covers exactly the customer the
 * counter is posting for; only `basket` mode posts its snapshot.
 */
export interface BasketPort {
  bind(sessionRef: string): Promise<void>
  unbind(): void
  snapshot(): Map<string, CartEntry>
  pendingReview(): ReviewItem[]
  readout(): BasketReadout
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
  /**
   * "The camera has lost sight of this item" (spec §3.2). Sent only when true — with the floor
   * holding every posted quantity, the numbers alone cannot say "still on the counter" from
   * "taken, and the floor keeps it": this flag is what turns the server's silent hold into a
   * tablet badge instead of a row that reads as silently deleted. Cleared by sending no field
   * once the camera sees the item again; the server stamps the transition, not every repeat.
   */
  lost?: boolean
}

/** One class's cumulative floor entry, as persisted for a restart. */
export interface PosFloorItem {
  className: string
  quantity: number
  maxConfidence: number
}

/** The floor of one POS session, saved so an app restart cannot un-count verified items. */
export interface PosFloor {
  sessionRef: string
  items: PosFloorItem[]
}

export interface PosSyncPayload {
  session_ref: string
  station_id: string
  items: PosSyncItem[]
  /**
   * Basket mode only: observed interactions the camera could not resolve. Above zero, the tablet
   * shows a review notice and refuses Finish — the cart may be missing a deposit or holding an
   * item that went back out. Absent in counter mode, which has no such notion.
   */
  pending_review?: number
  review_reasons?: string[]
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
   * The persisted cumulative floor, keyed by `session_ref`. The floor is memory-only by nature,
   * so an app restart mid-session would drop it and the next snapshot could empty the cart; the
   * store restores what this exact session already verified. Optional and failure-tolerant: a
   * floor that cannot be read or written costs the restart protection, never the sync.
   */
  floorStore?: {
    load(): Promise<PosFloor | null>
    save(floor: PosFloor): Promise<void>
    clear(): Promise<void>
  }
  /**
   * The state to show, or `null` for "there is no integration to report" — the feature is off
   * (spec §5.1: an empty URL, secret or station id disables it). `null` is deliberately not the
   * same as an `unbound` phase with nothing in the cart: one means nobody turned this on, the
   * other means someone did and no session is open yet. The Live view renders the first as
   * nothing, the same way it renders a native backend's absent inference verdict.
   */
  emit: (state: PosState | null) => void
  /** The basket tracker; optional so the counter-only orchestrator (and its tests) need none. */
  basket?: BasketPort
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
    // `lost` is part of the snapshot: losing sight of an item is a change worth posting even
    // though the floor keeps the quantity identical.
    if (!other || other.quantity !== value.quantity || other.lost !== value.lost) return false
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
  /**
   * The cumulative per-class floor of the bound session: the largest quantity ever posted for a
   * class. D2's automatic removal is suspended until the removal-capable model ships, so a
   * verified count never drops — the camera can only add to it. Cleared with the rest of the
   * session state in `bind`/`unbind`, so a new cart never inherits the previous customer's
   * counts. Customer corrections still work: an A6 override makes the webapp ignore the camera
   * for that product entirely.
   */
  private cumulative = new Map<string, CartEntry>()
  private lastSyncAtMs = 0
  private syncedItemCount = 0
  /** Review items in the last posted basket snapshot, so a change in review alone is posted. */
  private lastReviewSent = -1

  /** Sidecar-time seconds when capture was (re)started by this orchestrator, for the warm-up guard. */
  private captureRunningSince: number | null = null
  private captureRunning = false
  private restartTried = false

  private lastHealthMs = 0
  private lastSessionPollMs = 0
  private error: string | null = null
  /**
   * Who set `error`. A successful webapp call clears only a webapp's error: the session poll runs
   * every tick whether or not capture is up, and a webapp that started answering again says
   * nothing about a camera that is still down.
   */
  private errorSource: 'webapp' | 'capture' | 'sidecar' | null = null

  /**
   * Consecutive ticks whose webapp call failed, and whether the tick now running did. This is spec
   * §5.3's "retry with backoff": a webapp that stays unreachable is retried on a doubling interval
   * rather than at the poll cadence, so an outage does not become a request flood. A clean tick
   * resets it to zero, so the cadence comes straight back once the webapp answers.
   */
  private consecutiveFailures = 0
  private tickFailed = false
  /**
   * When the next tick is scheduled, while a backoff is standing. Kept beside the timeout so the
   * emitted state can carry the deadline; cleared as soon as a tick succeeds, so it never claims a
   * retry that is not coming.
   */
  private retryAtMs: number | null = null
  /** See `PosState.lastContactMs`. */
  private lastContactMs: number | null = null

  /**
   * Whether the tick chain is scheduled or running. It is what keeps *one* chain: `start()` runs
   * once, when the sidecar's port arrives, so a config the Admin Panel saves afterwards has to be
   * able to start the loop itself (`refreshConfig`) without a second chain appearing beside the
   * one already going.
   */
  private looping = false

  constructor(private readonly deps: PosSessionDeps) {}

  /** The basket tracker's readout changed: push it, without waiting for a POS transition. */
  notifyBasketChanged(): void {
    this.emit()
  }

  private basketMode(): boolean {
    return this.cfg?.cartMode === 'basket' && this.deps.basket !== undefined
  }

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
    // A fresh configuration is not a failure; a deadline a previous configuration scheduled would
    // otherwise be reported under the new one's name.
    this.retryAtMs = null
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
      // The catch is reached only by failures this method owns directly — the health probe, the
      // logs fetch, a capture start that threw. A webapp failure sets its own source inside
      // `pollSession`/`postSync` and never gets here.
      this.errorSource = 'sidecar'
      this.setError(error instanceof Error ? error.message : String(error))
    }
    if (this.stopped) return
    // Anything that fails the tick backs the interval off — the webapp unreachable, or the sidecar
    // not answering (a `getHealth` throw). What does *not* back off is the capture trouble
    // `ensureCapture` handles without throwing (a 409 calibration, its one failed restart), and a
    // 409 from the webapp is a session that is over rather than a retry.
    const wasBackingOff = this.retryAtMs !== null
    this.consecutiveFailures = this.tickFailed ? this.consecutiveFailures + 1 : 0
    const delay = this.retryDelay()
    this.retryAtMs = this.consecutiveFailures > 0 ? Date.now() + delay : null
    // Emits are transitions, and this one is: the backoff began, or it ended (possibly having
    // moved when a still-failing tick pushes the deadline out again).
    if (this.retryAtMs !== null || wasBackingOff) this.emit()
    this.schedule(delay)
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
      this.errorSource = 'capture'
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
      this.errorSource = null
      this.emit()
    } else {
      this.errorSource = 'capture'
      this.setError('capture failed to restart')
    }
  }

  private async pollSession(): Promise<void> {
    if (!this.cfg) return
    let remote: RemoteSession | null
    try {
      remote = await this.deps.getRemoteSession(this.cfg.stationId)
      this.noteContact()
    } catch (error) {
      this.tickFailed = true
      this.errorSource = 'webapp'
      this.setError(error instanceof Error ? error.message : String(error))
      return
    }

    if (!remote || remote.cartStatus === 'paid') {
      if (this.bound) this.unbind()
      this.clearTransportError()
      return
    }

    if (!this.bound || remote.sessionRef !== this.sessionRef) {
      await this.bind(remote)
      return
    }

    // Bound to the same session and healthy.
    this.clearTransportError()
  }

  private async bind(remote: RemoteSession): Promise<void> {
    this.bound = true
    this.sessionRef = remote.sessionRef
    this.cartCode = remote.cartCode
    this.bindAt = nowSeconds()
    this.cache.clear()
    this.lastSent.clear()
    this.cumulative.clear()
    this.logsSessionId = null
    // The heartbeat clock starts at binding, so an empty first snapshot is not posted as if
    // the 15 s had already elapsed.
    this.lastSyncAtMs = Date.now()
    this.syncedItemCount = 0
    this.lastReviewSent = -1
    this.error = null
    this.errorSource = null
    try {
      await this.deps.basket?.bind(remote.sessionRef)
    } catch {
      // The shadow ledger failing to bind must not stop the counter from serving the customer.
    }
    // Restore the persisted floor for this exact session: an app restart mid-session must not
    // un-count what was already verified (the camera can be blind to items already bagged). A
    // floor saved for any other session is stale and is dropped, not served.
    try {
      const saved = (await this.deps.floorStore?.load()) ?? null
      if (saved && saved.sessionRef === remote.sessionRef) {
        for (const item of saved.items) {
          this.cumulative.set(item.className, {
            quantity: item.quantity,
            maxConfidence: item.maxConfidence
          })
        }
      } else if (saved) {
        await this.deps.floorStore?.clear()
      }
    } catch {
      // A floor that cannot be read must not stop the bind; the camera remains source of truth.
    }
    this.emit()
  }

  /** Stop syncing but leave capture running (the customer tapped Finish). */
  private unbind(): void {
    this.bound = false
    this.sessionRef = null
    this.cartCode = null
    this.cache.clear()
    this.lastSent.clear()
    this.cumulative.clear()
    this.logsSessionId = null
    this.syncedItemCount = 0
    this.lastReviewSent = -1
    this.deps.basket?.unbind()
    // The session is over: a saved floor for it must not survive to be restored by mistake.
    void this.deps.floorStore?.clear().catch(() => {})
    this.emit()
  }

  /**
   * The session poll answered, which proves the webapp *and* the sidecar are reachable — the tick
   * runs its health probe first, and a throw there never reaches the poll. So an error recorded by
   * either transport is over; a capture error is not, because a webapp that started answering says
   * nothing about a camera that is still down.
   */
  private clearTransportError(): void {
    if (this.error && this.errorSource !== 'capture') {
      this.error = null
      this.errorSource = null
      this.emit()
    }
  }

  private isWarmingUp(now: number): boolean {
    // The ledger is not derived from what is visible, so there is nothing to warm up: a blind
    // camera leaves it exactly where it was, which is the hold the warm-up exists to provide.
    if (this.basketMode()) return false
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
    const basket = this.deps.basket
    if (this.basketMode() && basket) {
      // Basket mode posts the ledger: no logs, no warm-up, no floor. A confirmed removal is the
      // only thing that lowers a count, so there is nothing for the floor to protect against.
      const snapshot = basket.snapshot()
      const review = basket.pendingReview()
      const changed =
        !sameSnapshot(snapshot, this.lastSent) || review.length !== this.lastReviewSent
      const due = Date.now() - this.lastSyncAtMs >= SYNC_HEARTBEAT_MS
      if (changed || due) await this.postSync(snapshot, review)
      return
    }
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
    const cached = [...this.cache.values()]
    const derived = deriveCartState(cached, now, this.bindAt, this.cartCfg())
    // What the camera's raw track log can still account for, uncommitted — the mirror of
    // `derived`'s reluctant view. Below the quantity we are posting, the camera has lost the item.
    const live = deriveLiveCounts(cached, now, this.bindAt, this.cartCfg())
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

    // D2 suspended: the posted quantity is a floor for the rest of the session. `deriveCartState`
    // still answers "what is on the counter now" — including the dips of a track swap or a taken
    // item — and this merge is what keeps the billing side from following it down until a model
    // can distinguish the two.
    for (const [className, floor] of this.cumulative) {
      const current = snapshot.get(className)
      if (!current || current.quantity < floor.quantity) {
        snapshot.set(className, {
          quantity: floor.quantity,
          maxConfidence: Math.max(floor.maxConfidence, current?.maxConfidence ?? 0)
        })
      }
    }

    // The lost flag rides on the *final* quantities, floor-merged included: the camera's raw
    // count is compared against what will actually be posted. Warm-up is the exception — the
    // camera is blind by construction there, so no new flag is raised, and one already standing
    // is carried rather than dropped, so the badge cannot flap across a capture restart.
    //
    // A capture whose one restart has failed is not warming up, it is dead: warm-up would never
    // end, and the sync heartbeat keeps the tablet from showing *camera offline*, so without this
    // an unstaffed counter would show a healthy cart the camera can no longer see. Every held item
    // is lost until capture comes back.
    const captureDead = !this.captureRunning && this.restartTried
    if (captureDead) {
      for (const entry of snapshot.values()) entry.lost = true
    } else if (warmingUp) {
      for (const [className, entry] of snapshot) {
        if (this.lastSent.get(className)?.lost) entry.lost = true
      }
    } else {
      for (const [className, entry] of snapshot) {
        if ((live.get(className) ?? 0) < entry.quantity) entry.lost = true
      }
    }

    const changed = !sameSnapshot(snapshot, this.lastSent)
    const due = Date.now() - this.lastSyncAtMs >= SYNC_HEARTBEAT_MS
    if (!changed && !due) return

    await this.postSync(snapshot)
  }

  private async postSync(snapshot: Map<string, CartEntry>, review?: ReviewItem[]): Promise<void> {
    if (!this.cfg || !this.sessionRef) return
    const items: PosSyncItem[] = [...snapshot].map(([className, entry]) => {
      const item: PosSyncItem = {
        class_name: className,
        quantity: entry.quantity,
        max_confidence: entry.maxConfidence
      }
      // Absent when false, so the wire shape of a healthy item is unchanged.
      if (entry.lost) item.lost = true
      return item
    })
    const payload: PosSyncPayload = {
      session_ref: this.sessionRef,
      station_id: this.cfg.stationId,
      items
    }
    if (review !== undefined) {
      payload.pending_review = review.length
      payload.review_reasons = review.map((r) => r.reason)
    }

    try {
      const response = await this.deps.postSync(payload)
      this.noteContact()
      this.lastSent = snapshot
      if (review !== undefined) this.lastReviewSent = review.length
      let floorGrew = false
      for (const [className, entry] of review === undefined
        ? snapshot
        : new Map<string, CartEntry>()) {
        const floor = this.cumulative.get(className)
        if (!floor || entry.quantity > floor.quantity) floorGrew = true
        this.cumulative.set(className, {
          quantity: Math.max(floor?.quantity ?? 0, entry.quantity),
          maxConfidence: Math.max(floor?.maxConfidence ?? 0, entry.maxConfidence)
        })
      }
      // Persist the floor only when it grew: the heartbeat cadence must not cost a disk write
      // every 15 s, and the file is only the restart seed, not the cart's record.
      if (floorGrew) {
        try {
          await this.deps.floorStore?.save({
            sessionRef: this.sessionRef,
            items: [...this.cumulative].map(([className, entry]) => ({
              className,
              quantity: entry.quantity,
              maxConfidence: entry.maxConfidence
            }))
          })
        } catch {
          // The webapp already holds the counts; the file is only how a restart learns them.
        }
      }
      this.lastSyncAtMs = Date.now()
      this.syncedItemCount =
        response.cart_totals?.item_count ?? items.reduce((sum, item) => sum + item.quantity, 0)
      this.error = null
      this.errorSource = null
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
      this.errorSource = 'webapp'
      this.setError(error instanceof Error ? error.message : String(error))
    }
  }

  /** pushcart-web answered. The first answer is a transition worth telling an open window about. */
  private noteContact(): void {
    const first = this.lastContactMs === null
    this.lastContactMs = Date.now()
    if (first) this.emit()
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
      error: this.error,
      retryAtMs: this.retryAtMs,
      lastContactMs: this.lastContactMs,
      cartMode: this.cfg.cartMode,
      basket: this.deps.basket?.readout() ?? null
    })
  }

  private currentPhase(): PosPhase {
    if (this.error) return 'error'
    if (!this.bound) return 'unbound'
    return this.isWarmingUp(nowSeconds()) ? 'warming_up' : 'bound'
  }
}
