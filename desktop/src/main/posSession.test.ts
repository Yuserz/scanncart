// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ApplyResult } from './basketLedger'
import { DEFAULT_POS_CONFIG, type PosConfig } from './posConfig'
import {
  PosConflictError,
  PosSessionOrchestrator,
  PosTransportError,
  RETRY_BACKOFF_CAP_MS,
  SYNC_HEARTBEAT_MS,
  type PosFloor,
  type PosSessionDeps,
  type PosState,
  type PosSyncPayload,
  type RemoteSession,
  type SidecarLogEvent,
  type SidecarLogsResponse
} from './posSession'

const BASE_MS = 1_000_000_000_000
const BASE_S = BASE_MS / 1000

const CFG = {
  ...DEFAULT_POS_CONFIG,
  posBaseUrl: 'http://localhost:3000',
  posSecret: 'secret',
  stationId: 'counter-1',
  commitDwellS: 3,
  removeSettleS: 10,
  minCommitConf: 0.6
}

/**
 * The states the loop emits must be total: every consumer of the push (the Live panel's retry
 * countdown among them) reads the fields it needs off the object rather than checking for them, so
 * a field forgotten here would arrive as `undefined` at the renderer instead of failing here.
 */
function assertPosStateShape(state: PosState | null): void {
  if (state === null) return
  // Present and defined — `null` is a legitimate value for several of these (no cart, no error,
  // no standing retry), and it is *absence* the push must never carry.
  for (const key of [
    'phase',
    'cartCode',
    'syncedItemCount',
    'lastSyncAgeS',
    'error',
    'retryAtMs'
  ]) {
    expect(state).toHaveProperty(key)
    expect((state as unknown as Record<string, unknown>)[key]).not.toBeUndefined()
  }
  expect(state.retryAtMs === null || typeof state.retryAtMs === 'number').toBe(true)
}

const SESSION: RemoteSession = {
  sessionRef: 'scanncart-counter-1-1',
  cartId: 'cart-1',
  cartCode: 'CODE-1',
  cartStatus: 'active'
}

function event(overrides: Partial<SidecarLogEvent> & { enteredAt: number }): SidecarLogEvent {
  return {
    trackId: 1,
    className: 'soda',
    maxConf: 0.9,
    leftAt: null,
    ...overrides
  }
}

interface Harness {
  state: {
    /** Mutable, so a test can save a new configuration the way the Admin Panel does. */
    cfg: PosConfig
    health: string
    logs: SidecarLogsResponse
    remote: RemoteSession | null
    startCaptureResult: { ok: boolean; status: number }
    syncQueue: Array<() => void>
    /** When set, every sync rejects with it — for the persistent-outage backoff test. */
    syncError: Error | null
    /** When set, every session poll rejects with it — the unbound failure path. */
    sessionError: Error | null
    /** The persisted floor the injected store serves, and how often it was cleared. */
    savedFloor: PosFloor | null
    floorClears: number
  }
  calls: { logs: number[]; syncs: PosSyncPayload[]; starts: number; states: (PosState | null)[] }
  deps: PosSessionDeps
}

function harness(): Harness {
  const state = {
    cfg: CFG as PosConfig,
    health: 'running',
    logs: { sessionId: 1 as number | null, events: [] as SidecarLogEvent[] },
    remote: SESSION as RemoteSession | null,
    startCaptureResult: { ok: true, status: 200 },
    syncQueue: [] as Array<() => void>,
    syncError: null as Error | null,
    sessionError: null as Error | null,
    savedFloor: null as PosFloor | null,
    floorClears: 0
  }
  const calls = {
    logs: [] as number[],
    syncs: [] as PosSyncPayload[],
    starts: 0,
    states: [] as (PosState | null)[]
  }

  const deps: PosSessionDeps = {
    getConfig: async () => state.cfg,
    getHealth: async () => ({ state: state.health }),
    startCapture: async () => {
      calls.starts += 1
      if (state.startCaptureResult.ok) state.health = 'running'
      return state.startCaptureResult
    },
    getLogs: async (since) => {
      calls.logs.push(since)
      return { sessionId: state.logs.sessionId, events: state.logs.events }
    },
    getRemoteSession: async () => {
      if (state.sessionError) throw state.sessionError
      return state.remote
    },
    postSync: async (payload) => {
      calls.syncs.push(payload)
      if (state.syncError) throw state.syncError
      const behaviour = state.syncQueue.shift()
      if (behaviour) behaviour()
      return {
        cart_totals: {
          item_count: payload.items.reduce((sum, item) => sum + item.quantity, 0),
          subtotal: 0
        }
      }
    },
    floorStore: {
      load: async () => state.savedFloor,
      save: async (floor) => {
        state.savedFloor = {
          sessionRef: floor.sessionRef,
          items: floor.items.map((item) => ({ ...item }))
        }
      },
      clear: async () => {
        state.savedFloor = null
        state.floorClears += 1
      }
    },
    emit: (next) => {
      assertPosStateShape(next)
      calls.states.push(next)
    }
  }

  return { state, calls, deps }
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(BASE_MS)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('PosSessionOrchestrator', () => {
  it('starts capture when idle and leaves it running', async () => {
    const h = harness()
    h.state.health = 'idle'
    h.state.remote = null
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(10_000)

    expect(h.calls.starts).toBe(1)
    expect(h.calls.states.at(-1)?.phase).toBe('unbound')
    orch.stop()
  })

  it('binds to the session and excludes tracks from before bindAt', async () => {
    const h = harness()
    h.state.logs.events = [
      event({ trackId: 1, className: 'leftover', enteredAt: BASE_S - 50 }),
      event({ trackId: 2, className: 'fresh', enteredAt: BASE_S + 1 })
    ]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)

    const last = h.calls.syncs.at(-1)
    expect(last?.items.map((i) => i.class_name)).toEqual(['fresh'])
    expect(last?.items[0].quantity).toBe(1)
    orch.stop()
  })

  it('fetches logs with since=bindAt every time', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(6_000)

    expect(h.calls.logs.length).toBeGreaterThan(1)
    expect(new Set(h.calls.logs)).toEqual(new Set([BASE_S]))
    orch.stop()
  })

  it('syncs on change and then only on the 15 s heartbeat', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)
    const afterFirst = h.calls.syncs.length
    expect(afterFirst).toBeGreaterThanOrEqual(1)

    // No change for a few seconds: no extra sync.
    await vi.advanceTimersByTimeAsync(4_000)
    expect(h.calls.syncs.length).toBe(afterFirst)

    // The heartbeat arrives anyway.
    await vi.advanceTimersByTimeAsync(SYNC_HEARTBEAT_MS)
    expect(h.calls.syncs.length).toBeGreaterThan(afterFirst)
    orch.stop()
  })

  it('unbinds when the cart is paid, without stopping capture', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)

    h.state.remote = { ...SESSION, cartStatus: 'paid' }
    await vi.advanceTimersByTimeAsync(6_000)

    expect(h.calls.states.at(-1)?.phase).toBe('unbound')
    expect(h.calls.starts).toBe(0) // capture was already running and stayed running
    orch.stop()
  })

  it('surfaces a transport error and recovers on the next sync', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    h.state.syncQueue = [
      () => {
        throw new PosTransportError('webapp unreachable')
      }
    ]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.calls.states.at(-1)?.phase).toBe('error')

    await vi.advanceTimersByTimeAsync(2_000)
    expect(h.calls.states.at(-1)?.phase).toBe('bound')
    expect(h.calls.states.at(-1)?.error).toBeNull()
    orch.stop()
  })

  it('backs the retry off while the webapp stays unreachable, and resets on recovery', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    h.state.syncError = new PosTransportError('webapp unreachable')
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(30_000)

    // A fixed 1 s cadence would have made ~26 attempts in 30 s; the backoff makes far fewer.
    const attempts = h.calls.syncs.length
    expect(attempts).toBeGreaterThanOrEqual(3)
    expect(attempts).toBeLessThan(8)
    expect(h.calls.states.at(-1)?.phase).toBe('error')

    // The webapp comes back: the next attempt succeeds and the cadence resets to the poll interval.
    h.state.syncError = null
    await vi.advanceTimersByTimeAsync(40_000)
    expect(h.calls.states.at(-1)?.phase).toBe('bound')
    expect(h.calls.states.at(-1)?.error).toBeNull()
    orch.stop()
  })

  it('carries the retry deadline in the state it emits, and clears it on recovery', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    h.state.syncError = new PosTransportError('webapp unreachable')
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(2_000)

    // The first sync attempt lands once the track has committed (enteredAt + commitDwellS).
    await orch.start()
    await vi.advanceTimersByTimeAsync(4_500)

    const errored = h.calls.states.filter((s) => s?.phase === 'error')
    expect(errored.length).toBeGreaterThanOrEqual(1)
    // The deadline is a fact only the loop knows: the delay it scheduled for the next attempt.
    expect(errored.at(-1)?.retryAtMs).toBeGreaterThan(Date.now())
    const firstDeadline = errored.at(-1)!.retryAtMs as number
    expect(firstDeadline).toBeLessThanOrEqual(Date.now() + RETRY_BACKOFF_CAP_MS)

    // A still-failing tick pushes the deadline out; the emit between failures is what a countdown
    // on an already-open window reads, since nothing else re-renders the line.
    h.calls.states.length = 0
    await vi.advanceTimersByTimeAsync(10_000)
    const moved = h.calls.states.filter((s) => s?.retryAtMs != null)
    expect(moved.length).toBeGreaterThanOrEqual(1)
    expect(moved.at(-1)!.retryAtMs).toBeGreaterThan(firstDeadline)

    // Recovery: the cadence comes back, and the state stops claiming a retry is scheduled.
    h.state.syncError = null
    await vi.advanceTimersByTimeAsync(60_000)
    expect(h.calls.states.at(-1)?.retryAtMs).toBeNull()
    expect(h.calls.states.at(-1)?.phase).toBe('bound')
    expect(h.calls.states.at(-1)?.error).toBeNull()
    orch.stop()
  })

  it('emits the cleared deadline even when the success emits nothing new', async () => {
    // The recovery emit must reach the panel on a window that is already open: without it, the
    // 'retrying in Ns' line would outlive the failure it belongs to. This walks the *unbound*
    // failure path (the session poll, whose success emits nothing), so the tick tail is the only
    // place the cleared deadline can come from.
    const h = harness()
    h.state.remote = null
    h.state.sessionError = new PosTransportError('webapp unreachable')
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(2_000)
    expect(h.calls.states.at(-1)?.retryAtMs).not.toBeNull()
    expect(h.calls.states.at(-1)?.phase).toBe('error')

    h.state.sessionError = null
    await vi.advanceTimersByTimeAsync(60_000)
    expect(h.calls.states.at(-1)?.retryAtMs).toBeNull()
    expect(h.calls.states.at(-1)?.phase).toBe('unbound')
    orch.stop()
  })

  it('unbinds on a sync conflict (Finish won)', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    h.state.syncQueue = [
      () => {
        throw new PosConflictError('session_closed')
      }
    ]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)

    expect(h.calls.states.at(-1)?.phase).toBe('unbound')
    orch.stop()
  })

  it('restarts capture once and holds counts through the warm-up', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])

    const syncsBefore = h.calls.syncs.length
    // The camera dies and comes back with a new sidecar session and no events yet.
    h.state.health = 'idle'
    h.state.logs = { sessionId: 2, events: [] }
    h.state.startCaptureResult = { ok: true, status: 200 }
    await vi.advanceTimersByTimeAsync(3_000)

    expect(h.calls.starts).toBe(1)
    // Warming up: the derived state is empty but the held count is not removed.
    expect(h.calls.states.at(-1)?.phase).toBe('warming_up')
    expect(h.calls.syncs.length).toBe(syncsBefore)

    // Once the camera is warm and still sees nothing, the verified count holds anyway: D2's
    // automatic removal is suspended until the removal-capable model ships, so the cart never
    // follows the camera down to zero.
    await vi.advanceTimersByTimeAsync(20_000)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])
    orch.stop()
  })

  it('never lowers a posted quantity when the track ends (D2 suspended)', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])

    // The item is taken away: the track ends and the settle window passes with nothing there.
    h.state.logs.events = [event({ enteredAt: BASE_S + 1, leftAt: BASE_S + 5 })]
    await vi.advanceTimersByTimeAsync(SYNC_HEARTBEAT_MS + 12_000)

    // The posted count was verified, so it is a floor: the next heartbeat still carries it and
    // `pos_reconcile` never sees the row leave the snapshot.
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])
    orch.stop()
  })

  it('adds a second unit to the floor, and holds it when the counter empties', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])

    // A second soda is placed and commits: the count goes up, and its confidence is the best seen.
    h.state.logs.events = [
      event({ enteredAt: BASE_S + 1 }),
      event({ trackId: 2, enteredAt: BASE_S + 9, maxConf: 0.95 })
    ]
    await vi.advanceTimersByTimeAsync(9_000)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 2, max_confidence: 0.95 }
    ])

    // Both are taken away: the settle window empties the counter, the floor holds the last
    // verified count for the rest of the session.
    h.state.logs.events = [
      event({ enteredAt: BASE_S + 1, leftAt: BASE_S + 14 }),
      event({ trackId: 2, enteredAt: BASE_S + 9, maxConf: 0.95, leftAt: BASE_S + 15 })
    ]
    await vi.advanceTimersByTimeAsync(SYNC_HEARTBEAT_MS + 12_000)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 2, max_confidence: 0.95 }
    ])
    orch.stop()
  })

  it('starts a rebound session at zero, not at the previous cart', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])

    // The customer finishes; the desktop unbinds on the paid status…
    h.state.remote = { ...SESSION, cartStatus: 'paid' }
    await vi.advanceTimersByTimeAsync(6_000)
    expect(h.calls.states.at(-1)?.phase).toBe('unbound')

    // …and the next customer opens a session: the floor is gone, so nothing of the previous
    // cart can leak into the new one even though the track events are still in the log.
    h.state.remote = { ...SESSION, sessionRef: 'scanncart-counter-1-2' }
    await vi.advanceTimersByTimeAsync(SYNC_HEARTBEAT_MS + 6_000)
    const rebound = h.calls.syncs.filter((s) => s.session_ref === 'scanncart-counter-1-2')
    expect(rebound.length).toBeGreaterThanOrEqual(1)
    for (const sync of rebound) {
      expect(sync.items).toEqual([])
    }
    orch.stop()
  })

  it('restores the persisted floor after an app restart mid-session', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const first = new PosSessionOrchestrator(h.deps)
    await first.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])
    first.stop()

    // The app restarts: a new orchestrator, the same session still open, and the camera blind to
    // the item (the customer is holding it). The saved floor must come back, or the next empty
    // snapshot would reconcile the verified row out of the cart.
    h.state.logs = { sessionId: 1, events: [] }
    const second = new PosSessionOrchestrator(h.deps)
    await second.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])
    second.stop()
  })

  it('does not serve a floor saved for a different session', async () => {
    const h = harness()
    h.state.savedFloor = {
      sessionRef: 'scanncart-counter-1-0',
      items: [{ className: 'soda', quantity: 3, maxConfidence: 0.9 }]
    }
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(SYNC_HEARTBEAT_MS + 4_000)

    // The stale floor was dropped at bind, not counted into the new cart.
    expect(h.state.floorClears).toBeGreaterThanOrEqual(1)
    for (const sync of h.calls.syncs) {
      expect(sync.items).toEqual([])
    }
    orch.stop()
  })

  it('clears the persisted floor when the session ends', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)
    expect(h.state.savedFloor).not.toBeNull()

    h.state.remote = { ...SESSION, cartStatus: 'paid' }
    await vi.advanceTimersByTimeAsync(6_000)
    expect(h.calls.states.at(-1)?.phase).toBe('unbound')
    expect(h.state.savedFloor).toBeNull()
    orch.stop()
  })

  it('reports nothing and polls nothing while the feature is off', async () => {
    const h = harness()
    h.state.cfg = { ...CFG, posBaseUrl: '', posSecret: '', stationId: '' }
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(30_000)

    // One `null`, from `start()`, and no loop behind it: an empty URL/secret/station id disables
    // the integration (§5.1), so there is nothing to poll and nothing to report.
    expect(h.calls.states).toEqual([null])
    expect(h.calls.logs).toEqual([])
    expect(h.calls.starts).toBe(0)
    orch.stop()
  })

  it('starts the loop when a saved configuration enables it', async () => {
    const h = harness()
    h.state.cfg = { ...CFG, posBaseUrl: '', posSecret: '', stationId: '' }
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    // The Admin Panel's Save re-reads the configuration; the app only ever calls `start()` once,
    // when the sidecar's port arrives, so this is the path that has to start the loop.
    h.state.cfg = CFG
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    await orch.refreshConfig()
    await vi.advanceTimersByTimeAsync(4_200)

    expect(h.calls.states.at(-1)?.phase).toBe('bound')
    expect(h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
    ])
    orch.stop()
  })

  it('ends the loop when the feature is switched off, and keeps it restartable', async () => {
    const h = harness()
    h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(4_200)

    h.state.cfg = { ...CFG, posBaseUrl: '' }
    await orch.refreshConfig()
    expect(h.calls.states.at(-1)).toBeNull()

    // Switched off: the chain stops rather than polling a webapp the settings no longer name. The
    // capture itself keeps running — this is not a capture stop.
    const fetches = h.calls.logs.length
    await vi.advanceTimersByTimeAsync(30_000)
    expect(h.calls.logs.length).toBe(fetches)

    // And back on, the same entry point starts a fresh chain.
    h.state.cfg = CFG
    await orch.refreshConfig()
    await vi.advanceTimersByTimeAsync(2_000)
    expect(h.calls.states.at(-1)?.phase).toBe('bound')
    orch.stop()
  })

  it('tries only one restart, then reports the failure', async () => {
    const h = harness()
    h.state.health = 'idle'
    h.state.remote = null
    h.state.startCaptureResult = { ok: false, status: 500 }
    const orch = new PosSessionOrchestrator(h.deps)

    await orch.start()
    await vi.advanceTimersByTimeAsync(10_000)

    expect(h.calls.starts).toBe(1)
    expect(h.calls.states.at(-1)?.phase).toBe('error')
    orch.stop()
  })
})

describe('PosSessionOrchestrator — basket mode', () => {
  /** A basket port backed by a plain ledger, driven by the test instead of a camera. */
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- a test harness whose shape is its own return
  async function withBasket(mode: 'counter' | 'basket') {
    const { BasketLedger } = await import('./basketLedger')
    const ledger = new BasketLedger()
    const bound: string[] = []
    let unbinds = 0
    const h = harness()
    h.state.cfg = { ...CFG, cartMode: mode }
    h.deps.basket = {
      bind: async (ref) => {
        bound.push(ref)
        await ledger.bind(ref, BASE_S)
      },
      unbind: () => {
        unbinds += 1
        ledger.unbind()
      },
      snapshot: () => ledger.snapshot(),
      pendingReview: () => ledger.pendingReview(),
      readout: () => ({
        candidates: 0,
        itemCount: [...ledger.snapshot().values()].reduce((n, e) => n + e.quantity, 0),
        items: [],
        review: ledger.pendingReview(),
        blind: false
      })
    }
    const deposit = (className: string, at: number, trackId = 1): ApplyResult =>
      ledger.apply(
        { kind: 'inbound', className, completedAt: at, trackId, completionConf: 0.9 },
        'z'
      )
    const removal = (className: string, at: number, trackId = 1): ApplyResult =>
      ledger.apply(
        { kind: 'outbound', className, completedAt: at, trackId, completionConf: 0.9 },
        'z'
      )
    return { h, ledger, bound, unbinds: () => unbinds, deposit, removal }
  }

  it('posts the ledger, and a confirmed removal lowers the cart', async () => {
    const b = await withBasket('basket')
    const orch = new PosSessionOrchestrator(b.h.deps)
    await orch.start()
    await vi.advanceTimersByTimeAsync(1_500)
    expect(b.bound).toEqual([SESSION.sessionRef])

    b.deposit('soda', BASE_S + 1)
    b.deposit('soda', BASE_S + 2, 2)
    await vi.advanceTimersByTimeAsync(1_500)
    expect(b.h.calls.syncs.at(-1)?.items).toEqual([
      { class_name: 'soda', quantity: 2, max_confidence: 0.9 }
    ])

    b.removal('soda', BASE_S + 3, 2)
    await vi.advanceTimersByTimeAsync(1_500)
    expect(b.h.calls.syncs.at(-1)?.items[0].quantity).toBe(1)
    // Basket mode never reads the counter's logs.
    expect(b.h.calls.logs).toEqual([])
    orch.stop()
  })

  it('carries review in the payload, and a review change alone is posted', async () => {
    const b = await withBasket('basket')
    const orch = new PosSessionOrchestrator(b.h.deps)
    await orch.start()
    await vi.advanceTimersByTimeAsync(1_500)
    expect(b.h.calls.syncs.at(-1)?.pending_review).toBe(0)

    const before = b.h.calls.syncs.length
    b.ledger.flag('soda', 'two items crossed the opening at once', BASE_S + 1)
    await vi.advanceTimersByTimeAsync(1_500)
    expect(b.h.calls.syncs.length).toBeGreaterThan(before)
    expect(b.h.calls.syncs.at(-1)).toMatchObject({
      pending_review: 1,
      review_reasons: ['two items crossed the opening at once']
    })
    orch.stop()
  })

  it('counter mode keeps posting the counter, with the ledger as a shadow readout', async () => {
    const b = await withBasket('counter')
    b.h.state.logs.events = [event({ enteredAt: BASE_S + 1 })]
    const orch = new PosSessionOrchestrator(b.h.deps)
    await orch.start()
    await vi.advanceTimersByTimeAsync(500)
    b.deposit('tuna', BASE_S + 1)
    orch.notifyBasketChanged()
    await vi.advanceTimersByTimeAsync(3_700)

    const last = b.h.calls.syncs.at(-1)
    expect(last?.items.map((i) => i.class_name)).toEqual(['soda'])
    expect(last).not.toHaveProperty('pending_review')
    expect(b.h.calls.states.at(-1)?.basket?.itemCount).toBe(1)
    expect(b.bound).toEqual([SESSION.sessionRef])
    orch.stop()
  })

  it('unbinds the ledger when the session ends', async () => {
    const b = await withBasket('basket')
    const orch = new PosSessionOrchestrator(b.h.deps)
    await orch.start()
    await vi.advanceTimersByTimeAsync(1_500)
    b.h.state.remote = { ...SESSION, cartStatus: 'paid' }
    await vi.advanceTimersByTimeAsync(6_000)
    expect(b.unbinds()).toBe(1)
    orch.stop()
  })
})
