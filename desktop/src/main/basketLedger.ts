// The basket ledger: the per-SKU quantities the camera vouches for in `basket` cart mode
// (CART_TRANSFER_SPEC §4), and the journal of confirmed transfers that produced them.
//
// The counter mode (`cartState.ts`) derives the cart from what is *visible*, which is why it
// cannot remove anything safely: an item hidden under another one looks the same as an item
// taken out. Here only a confirmed transfer changes a quantity — a deposit +1, a removal −1 —
// and disappearing changes nothing, so an item can be covered for the rest of the session and
// still be paid for.
//
// Four rules carry it:
// - **Once per event.** An event's id is derived from its own facts (session, track, kind,
//   completion time), so a replay — a restart re-reading the journal, a duplicated stream
//   message — is recognised and applied zero more times.
// - **Never negative.** A removal of something the ledger does not hold is not a −1; it is
//   an observed interaction nobody can resolve automatically, so it becomes a review item.
// - **Review is a state, not a log line.** Unresolved interactions (identity conflicts, a
//   track that died mid-path, two items at once, a blind camera) stay listed until resolved,
//   and `pendingReview()` is what blocks Finish on the tablet.
// - **Keyed by session.** Restored only for the same `session_ref`, so a new customer never
//   inherits the previous one's basket; a finished session's ledger is cleared.

import { promises as fs } from 'fs'
import { join } from 'path'

import type { CartEntry } from './cartState'
import type { TransferEvent } from './transferState'

export interface LedgerEvent {
  id: string
  className: string
  kind: 'inbound' | 'outbound'
  completedAt: number
  trackId: number
  completionConf: number
  /** Which zone layout the event was judged under (`zoneKey`), for the evidence trail. */
  zones: string
}

export interface ReviewItem {
  id: string
  /** The product involved, when the interaction named one. */
  className: string | null
  reason: string
  at: number
}

export interface LedgerItem {
  className: string
  quantity: number
  maxConfidence: number
}

export interface BasketLedgerState {
  sessionRef: string
  items: LedgerItem[]
  journal: LedgerEvent[]
  review: ReviewItem[]
  /** When the camera stopped producing fresh evidence, or null while it is observing. */
  blindSince: number | null
}

export type ApplyResult =
  | { status: 'applied'; event: LedgerEvent }
  | { status: 'duplicate' }
  | { status: 'review'; review: ReviewItem }
  | { status: 'unbound' }

export interface LedgerStore {
  load(): Promise<BasketLedgerState | null>
  save(state: BasketLedgerState): Promise<void>
  clear(): Promise<void>
}

export function eventId(sessionRef: string, e: TransferEvent): string {
  return `${sessionRef}:${e.trackId}:${e.kind}:${e.completedAt.toFixed(3)}`
}

export class BasketLedger {
  private state: BasketLedgerState | null = null
  private seen = new Set<string>()
  private reviewSeq = 0

  constructor(private readonly store?: LedgerStore) {}

  get sessionRef(): string | null {
    return this.state?.sessionRef ?? null
  }

  /**
   * Bind to a session. A saved ledger for the *same* session is restored (an app restart
   * mid-session); one for any other session is stale and dropped. A restored ledger comes
   * back with a review item, because whatever happened while the app was down was not seen.
   */
  async bind(sessionRef: string, now: number): Promise<void> {
    if (this.state?.sessionRef === sessionRef) return
    let restored: BasketLedgerState | null = null
    try {
      const saved = (await this.store?.load()) ?? null
      if (saved && saved.sessionRef === sessionRef) restored = saved
      else if (saved) await this.store?.clear()
    } catch {
      // A ledger that cannot be read costs the restart protection, never the bind.
    }
    this.state = restored ?? { sessionRef, items: [], journal: [], review: [], blindSince: null }
    this.seen = new Set(this.state.journal.map((e) => e.id))
    this.reviewSeq = this.state.review.length
    if (restored)
      this.addReview(null, 'the app restarted during this session; check the basket', now)
    this.persist()
  }

  /** The session is over: forget it and remove the saved copy. */
  unbind(): void {
    this.state = null
    this.seen.clear()
    const store = this.store
    // Chained after any pending save, or a late write would resurrect the finished session.
    if (store) this.writes = this.writes.then(() => store.clear()).catch(() => {})
  }

  /** Apply one confirmed transfer, at most once. */
  apply(e: TransferEvent, zones: string): ApplyResult {
    const s = this.state
    if (!s) return { status: 'unbound' }
    const id = eventId(s.sessionRef, e)
    if (this.seen.has(id)) return { status: 'duplicate' }
    this.seen.add(id)

    const item = s.items.find((i) => i.className === e.className)
    if (e.kind === 'outbound' && (!item || item.quantity <= 0)) {
      const review = this.addReview(
        e.className,
        `removal of ${e.className}, which is not in the basket`,
        e.completedAt
      )
      return { status: 'review', review }
    }

    const event: LedgerEvent = {
      id,
      className: e.className,
      kind: e.kind,
      completedAt: e.completedAt,
      trackId: e.trackId,
      completionConf: e.completionConf,
      zones
    }
    s.journal.push(event)
    if (e.kind === 'inbound') {
      if (item) {
        item.quantity += 1
        item.maxConfidence = Math.max(item.maxConfidence, e.completionConf)
      } else {
        s.items.push({ className: e.className, quantity: 1, maxConfidence: e.completionConf })
      }
    } else if (item) {
      item.quantity -= 1
      if (item.quantity === 0) s.items = s.items.filter((i) => i !== item)
    }
    this.persist()
    return { status: 'applied', event }
  }

  /** Record an observed interaction the machine could not resolve. */
  flag(className: string | null, reason: string, at: number): ReviewItem | null {
    if (!this.state) return null
    const r = this.addReview(className, reason, at)
    this.persist()
    return r
  }

  /** The camera stopped producing fresh evidence. Idempotent while blind. */
  markBlind(now: number): void {
    if (!this.state || this.state.blindSince !== null) return
    this.state.blindSince = now
    this.persist()
  }

  /**
   * Fresh evidence again. A blind interval longer than `graceS` becomes a review item: the
   * ledger is kept as it was (nothing is guessed about the blind interval), but someone has
   * to confirm nothing went in or out while the camera could not see.
   */
  markSeeing(now: number, graceS: number): void {
    const s = this.state
    if (!s || s.blindSince === null) return
    const blindFor = now - s.blindSince
    if (blindFor > graceS) {
      this.addReview(null, `the camera could not see the basket for ${Math.round(blindFor)} s`, now)
    }
    s.blindSince = null
    this.persist()
  }

  /** Resolve one review item (staff confirmed the basket), or all of them. */
  resolveReview(id?: string): void {
    const s = this.state
    if (!s) return
    s.review = id === undefined ? [] : s.review.filter((r) => r.id !== id)
    this.persist()
  }

  pendingReview(): ReviewItem[] {
    return [...(this.state?.review ?? [])]
  }

  isBlind(): boolean {
    return this.state?.blindSince != null
  }

  /** The desired cart: the shape `posSession` posts as a snapshot. */
  snapshot(): Map<string, CartEntry> {
    const out = new Map<string, CartEntry>()
    for (const i of this.state?.items ?? []) {
      if (i.quantity > 0)
        out.set(i.className, { quantity: i.quantity, maxConfidence: i.maxConfidence })
    }
    return out
  }

  journal(): LedgerEvent[] {
    return [...(this.state?.journal ?? [])]
  }

  private addReview(className: string | null, reason: string, at: number): ReviewItem {
    const s = this.state as BasketLedgerState
    this.reviewSeq += 1
    const r: ReviewItem = { id: `r${this.reviewSeq}`, className, reason, at }
    s.review.push(r)
    return r
  }

  private writes: Promise<void> = Promise.resolve()

  /**
   * Saves are chained, never concurrent: two overlapping writes through one temp file could
   * rename an older copy over a newer one. A failed write costs restart protection only.
   */
  private persist(): void {
    if (!this.state || !this.store) return
    const store = this.store
    const copy: BasketLedgerState = JSON.parse(JSON.stringify(this.state))
    this.writes = this.writes.then(() => store.save(copy)).catch(() => {})
  }

  /** Resolves once every save requested so far has finished (tests, quit). */
  flushed(): Promise<void> {
    return this.writes
  }
}

// ---------------------------------------------------------------------------
// The file store
// ---------------------------------------------------------------------------

export const BASKET_LEDGER_FILENAME = 'basket-ledger.json'

/** `basket-ledger.json` beside `pos.json`; takes the directory so it is testable on a temp dir. */
export class BasketLedgerFileStore implements LedgerStore {
  constructor(private readonly dir: string) {}

  private get path(): string {
    return join(this.dir, BASKET_LEDGER_FILENAME)
  }

  /** A missing or corrupt file reads as nothing saved; it must never crash the sync loop. */
  async load(): Promise<BasketLedgerState | null> {
    try {
      const raw = JSON.parse(await fs.readFile(this.path, 'utf8')) as Partial<BasketLedgerState>
      if (typeof raw.sessionRef !== 'string' || !Array.isArray(raw.items)) return null
      const items = raw.items.filter(
        (i) =>
          i != null &&
          typeof i.className === 'string' &&
          Number.isInteger(i.quantity) &&
          i.quantity > 0
      )
      const journal = Array.isArray(raw.journal)
        ? raw.journal.filter((e) => e != null && typeof e.id === 'string')
        : []
      const review = Array.isArray(raw.review)
        ? raw.review.filter((r) => r != null && typeof r.id === 'string')
        : []
      const blindSince = typeof raw.blindSince === 'number' ? raw.blindSince : null
      return { sessionRef: raw.sessionRef, items, journal, review, blindSince }
    } catch {
      return null
    }
  }

  /** Atomic write (temp file + rename): a half-written ledger must not read as an empty basket. */
  async save(state: BasketLedgerState): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true })
    const tmp = `${this.path}.tmp`
    await fs.writeFile(tmp, `${JSON.stringify(state, null, 2)}\n`, 'utf8')
    await fs.rename(tmp, this.path)
  }

  async clear(): Promise<void> {
    await fs.rm(this.path, { force: true })
  }
}
