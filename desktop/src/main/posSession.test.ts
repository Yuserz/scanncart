// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_POS_CONFIG, type PosConfig } from './posConfig'
import {
  PosConflictError,
  PosSessionOrchestrator,
  PosTransportError,
  SYNC_HEARTBEAT_MS,
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
    syncError: null as Error | null
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
    getRemoteSession: async () => state.remote,
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
    emit: (next) => calls.states.push(next)
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

    // Once the camera is warm and still sees nothing, the item is finally removed.
    await vi.advanceTimersByTimeAsync(20_000)
    expect(h.calls.syncs.at(-1)?.items).toEqual([])
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
