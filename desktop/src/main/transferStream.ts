// The basket tracker: the sidecar's frame stream → the transfer machine → the basket ledger.
//
// It runs in the main process, beside the POS orchestrator, on its own WebSocket to the
// sidecar (the renderer has one too; the sidecar serves any number). Three rules decide what
// counts as evidence:
// - **Fresh only.** `fresh: false` frames are preview fill-ins repeating the last boxes; feeding
//   them would let one inference satisfy "two observations per side" by itself.
// - **True orientation.** Boxes reflected for a mirrored preview are reflected back
//   (`frameBox`), so "the cart is at the left edge" means the camera's left, not the mirror's.
// - **Tracked only.** A detection with no `track_id` cannot be followed along a path, so it is
//   no evidence of a transfer at all.
//
// It also owns the two checks that need the stream rather than a single event: the camera going
// blind (no fresh frame for `blindAfterS`, or capture leaving `running`), which the ledger turns
// into review when it lasts, and the empty-basket baseline at bind (#16) — a product already
// sitting in the cart band when a session starts is flagged, because the ledger begins at zero
// and that item would otherwise never be billed. "Already sitting" means its track was *first
// seen* inside the cart band: a product the customer carries in through the opening during those
// first seconds was first seen outside, is a deposit, and must not block Finish as a leftover.

import { BasketLedger, type LedgerStore, type ReviewItem } from './basketLedger'
import type { CartEntry } from './cartState'
import {
  frameBox,
  layoutKey,
  layoutRegions,
  type ZoneLayout,
  type ZonePreset
} from './transferGeometry'
import {
  DEFAULT_TRANSFER_CONFIG,
  TransferStateMachine,
  regionOf,
  type Candidate,
  type Region,
  type TransferStateConfig
} from './transferState'

/**
 * How many tracks' first regions are remembered before stale ones are pruned, and how long a
 * track may go unseen before it counts as stale. The map only answers "where did this track
 * start", which nothing asks once a track has been gone for this long.
 */
const FIRST_REGION_LIMIT = 256
const FIRST_REGION_STALE_S = 30

/** The subset of a sidecar frame message this reads. */
export interface StreamFrame {
  type: 'frame'
  ts: number
  seq: number
  fresh?: boolean
  mirrored?: boolean
  detections: Array<{
    track_id: number | null
    cls: string
    conf: number
    box: [number, number, number, number]
  }>
}

interface WSLike {
  onopen: (() => void) | null
  onmessage: ((e: { data: unknown }) => void) | null
  onclose: (() => void) | null
  onerror: ((e: unknown) => void) | null
  close(): void
}

export interface BasketReadout {
  /** In-flight candidates (shadow readout). */
  candidates: number
  /** Units the ledger holds. */
  itemCount: number
  items: Array<{ className: string; quantity: number }>
  review: ReviewItem[]
  blind: boolean
}

/**
 * What the basket test screen shows: the ledger's readout, the zones it is judging under (to draw
 * them over the preview), and who the ledger is bound to. Separate from `PosState` because that is
 * `null` whenever the POS integration is off, and a desk test has no pushcart-web at all.
 */
export interface BasketViewState {
  readout: BasketReadout
  layout: ZoneLayout
  /** A desk practice run is bound: deposits and removals count with no tablet session. */
  practice: boolean
  /** A real customer session is bound (practice cannot start while one is). */
  customerBound: boolean
}

export interface BasketTrackerOptions {
  store?: LedgerStore
  transfer?: TransferStateConfig
  /** Seconds without a fresh frame before the camera counts as blind. */
  blindAfterS?: number
  /** A blind interval shorter than this is not worth a review (a dropped frame or two). */
  blindReviewAfterS?: number
  /** Seconds after bind during which a product in the cart band means the basket was not empty. */
  baselineS?: number
  now?: () => number
  wsFactory?: (url: string) => WSLike
  reconnectDelayMs?: number
  onChange?: (readout: BasketReadout) => void
}

/** A bare preset is a band layout: that is what every caller passed before outlines existed. */
export function asLayout(zones: ZonePreset | ZoneLayout): ZoneLayout {
  return 'mode' in zones ? zones : { mode: 'bands', ...zones }
}

export function zonesKey(p: ZonePreset): string {
  return layoutKey({ mode: 'bands', ...p })
}

export class BasketTracker {
  readonly ledger: BasketLedger
  private machine: TransferStateMachine
  private layout: ZoneLayout
  private zones: string
  private lastFreshT: number | null = null
  private captureRunning = true
  private baselineUntil: number | null = null
  private baselineFlagged = false
  /**
   * Where each track was first seen, and when it was last seen. Kept across a bind on purpose: a
   * leftover's track usually began before the customer tapped Start, and "first seen inside" is
   * exactly what makes it a leftover. Cleared when capture stops, because the sidecar's tracker
   * starts its ids over and a reused id would inherit another item's history.
   */
  private firstRegion = new Map<number, { region: Region; lastT: number }>()
  private ws: WSLike | null = null
  private closed = true
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private readonly now: () => number

  constructor(
    zones: ZonePreset | ZoneLayout,
    private readonly opts: BasketTrackerOptions = {}
  ) {
    this.ledger = new BasketLedger(opts.store)
    this.machine = new TransferStateMachine(
      layoutRegions(asLayout(zones)),
      opts.transfer ?? DEFAULT_TRANSFER_CONFIG
    )
    this.layout = asLayout(zones)
    this.zones = layoutKey(this.layout)
    this.now = opts.now ?? (() => Date.now() / 1000)
  }

  /** The scanner's live `conf_threshold`, so the basket infers nothing the operator cut. */
  setConfThreshold(value: number): void {
    this.machine.setConfThreshold(value)
  }

  /** New zones (bands or drawn outlines) were saved. In-flight candidates survive. */
  setZones(zones: ZonePreset | ZoneLayout): void {
    const layout = asLayout(zones)
    const key = layoutKey(layout)
    if (key === this.zones) return
    this.machine.setRegions(layoutRegions(layout))
    this.layout = layout
    this.zones = key
  }

  /** The layout the basket is judging under, for drawing it over the preview. */
  zoneLayout(): ZoneLayout {
    return this.layout
  }

  /** Whether a session (a customer's, or a desk practice run) is bound to the ledger. */
  boundSessionRef(): string | null {
    return this.ledger.sessionRef
  }

  // ---- the port the POS orchestrator uses ----

  async bind(sessionRef: string): Promise<void> {
    const fresh = this.ledger.sessionRef !== sessionRef
    await this.ledger.bind(sessionRef, this.now())
    if (fresh) {
      this.machine.reset()
      this.baselineUntil = this.now() + (this.opts.baselineS ?? 3)
      this.baselineFlagged = false
    }
    this.changed()
  }

  unbind(): void {
    this.ledger.unbind()
    this.machine.reset()
    this.baselineUntil = null
    this.changed()
  }

  snapshot(): Map<string, CartEntry> {
    return this.ledger.snapshot()
  }

  pendingReview(): ReviewItem[] {
    return this.ledger.pendingReview()
  }

  resolveReview(id?: string): void {
    this.ledger.resolveReview(id)
    this.changed()
  }

  readout(): BasketReadout {
    const items = [...this.ledger.snapshot()].map(([className, e]) => ({
      className,
      quantity: e.quantity
    }))
    return {
      candidates: this.machine.snapshot().length,
      itemCount: items.reduce((n, i) => n + i.quantity, 0),
      items,
      review: this.ledger.pendingReview(),
      blind: this.ledger.isBlind()
    }
  }

  candidates(): Candidate[] {
    return this.machine.snapshot()
  }

  // ---- the stream ----

  /** Feed one sidecar frame message. Public so tests (and a replay tool) can drive it directly. */
  onFrame(msg: StreamFrame): void {
    if (msg.fresh === false) return
    const t = msg.ts
    let dirty = false
    if (this.lastFreshT === null || this.ledger.isBlind()) {
      this.ledger.markSeeing(t, this.opts.blindReviewAfterS ?? 3)
      dirty = true
    }
    this.lastFreshT = t

    const liveIds = new Set<number>()
    for (const d of msg.detections) if (d.track_id !== null) liveIds.add(d.track_id)

    for (const d of msg.detections) {
      if (d.track_id === null) continue
      const box = frameBox(d.box, msg.mirrored === true)
      const observation = {
        seq: msg.seq,
        t,
        trackId: d.track_id,
        className: d.cls,
        conf: d.conf,
        box
      }
      // A fresh id for an item the tracker just lost (fast motion) carries on its path, and its
      // first region with it, so the baseline does not mistake it for a leftover either.
      const replaced = this.machine.stitch(observation, liveIds)
      if (replaced !== null) {
        const entry = this.firstRegion.get(replaced)
        if (entry) this.firstRegion.set(d.track_id, entry)
      }
      const first = this.noteRegion(d.track_id, regionOf(this.machine.getRegions(), box), t)

      if (this.baselineUntil !== null && t <= this.baselineUntil && !this.baselineFlagged) {
        if (first === 'inside' && d.conf >= this.machine.getConfThreshold()) {
          this.ledger.flag(d.cls, `the basket was not empty at Start (${d.cls} already inside)`, t)
          this.baselineFlagged = true
          dirty = true
        }
      }

      const out = this.machine.observe(observation)
      for (const e of out.events) {
        this.ledger.apply(e, this.zones)
        dirty = true
      }
      if (out.removed?.review) {
        this.ledger.flag(out.removed.className, out.removed.reason, t)
        dirty = true
      }
    }
    if (this.baselineUntil !== null && t > this.baselineUntil) this.baselineUntil = null

    for (const x of this.machine.sweep(t)) {
      if (x.review) {
        this.ledger.flag(x.className, `${x.className} stopped being seen mid-transfer`, t)
        dirty = true
      }
    }
    if (dirty || this.machine.snapshot().length > 0) this.changed()
  }

  /** Capture state from the stream's status messages: anything but running is blind. */
  onStatus(state: string): void {
    const running = state === 'running'
    if (running === this.captureRunning) return
    this.captureRunning = running
    if (!running) {
      this.machine.reset()
      this.firstRegion.clear()
      this.ledger.markBlind(this.lastFreshT ?? this.now())
      this.changed()
    }
  }

  /** Periodic check for a stream that went quiet without a status (a hung sidecar). */
  tick(): void {
    const blindAfter = this.opts.blindAfterS ?? 2
    if (
      this.lastFreshT !== null &&
      this.now() - this.lastFreshT > blindAfter &&
      !this.ledger.isBlind()
    ) {
      this.machine.reset()
      this.ledger.markBlind(this.lastFreshT)
      this.changed()
    }
  }

  /**
   * Open the stream, asking `getPort` for the sidecar's port on **every** attempt. A restarted
   * sidecar can come back on another port (`run.py` falls back to a free one when 8765 is taken),
   * and a socket that kept redialling the first one would leave the basket blind for the rest of
   * the run while every REST call — which reads the port live — went on working.
   */
  connect(getPort: () => number | null): void {
    this.closed = false
    const factory =
      this.opts.wsFactory ??
      ((u: string) =>
        new (globalThis as unknown as { WebSocket: new (u: string) => WSLike }).WebSocket(u))
    const retry = (): void => {
      if (this.closed || this.reconnectTimer !== null) return
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null
        open()
      }, this.opts.reconnectDelayMs ?? 1000)
    }
    const open = (): void => {
      if (this.closed) return
      const port = getPort()
      if (port === null) {
        // No sidecar yet (or it is between restarts): ask again shortly rather than giving up.
        retry()
        return
      }
      this.ws = factory(`ws://127.0.0.1:${port}/ws/stream`)
      this.ws.onmessage = (e) => {
        if (typeof e.data !== 'string') return
        let msg: { type?: string; state?: string }
        try {
          msg = JSON.parse(e.data)
        } catch {
          return
        }
        if (msg?.type === 'frame') this.onFrame(msg as StreamFrame)
        else if (msg?.type === 'status' && typeof msg.state === 'string') this.onStatus(msg.state)
      }
      this.ws.onopen = null
      this.ws.onerror = () => {}
      this.ws.onclose = retry
    }
    open()
  }

  close(): void {
    this.closed = true
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = null
    this.ws?.close()
    this.ws = null
  }

  /** Record a sighting and return the region this track was first seen in. */
  private noteRegion(trackId: number, region: Region, t: number): Region {
    const known = this.firstRegion.get(trackId)
    if (known) {
      known.lastT = t
      return known.region
    }
    if (this.firstRegion.size >= FIRST_REGION_LIMIT) {
      for (const [id, entry] of this.firstRegion) {
        if (t - entry.lastT > FIRST_REGION_STALE_S) this.firstRegion.delete(id)
      }
    }
    this.firstRegion.set(trackId, { region, lastT: t })
    return region
  }

  private changed(): void {
    this.opts.onChange?.(this.readout())
  }
}
