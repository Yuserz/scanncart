// @vitest-environment node
// The POS routes over a socket, in two modes (docs/POS_SMOKE_TEST.md §3; spec §4.2/§7).
//
// **The in-repo stand-in** (`posWebappStandIn.ts`) runs always, in `npm test`. It is the mode that can
// be *controlled*, so the smoke test's lifecycle rows live here: a dead host and its backoff
// countdown, the host coming back with no duplicate rows, a session that ends under the desktop.
//
// **A real pushcart-web** runs when `POS_E2E_BASE_URL` is set — `make verify-pos-routes` is how. It
// runs the same protocol checks against the actual routes, which is the only thing that can catch the
// stand-in drifting from what it mirrors: a header, a path, a status code, a validation rule. Nothing
// else can, because the stand-in is only as faithful as the pushcart-web files it was written from.
// It needs a running server and a registered station, so it is out of `make test`, and the target
// fails loudly when the three variables are absent rather than reporting a green skip.
//
// The unit suites fake `postSync`/`getRemoteSession` at the dependency boundary, so they cannot see
// the layer between them and the wire. That layer is what this file exercises — the real transport
// (`posClient.ts`) over HTTP, with the real orchestrator (`posSession.ts`) on top of it.

import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { fetchRemoteSession, postCartSync, probePosSession } from './posClient'
import { DEFAULT_POS_CONFIG, type PosConfig } from './posConfig'
import {
  PosConflictError,
  PosSessionOrchestrator,
  PosTransportError,
  type PosState,
  type SidecarLogEvent
} from './posSession'
import { PosWebappStandIn, type StandInSession } from './posWebappStandIn'

const STATION = 'counter-1'
const SECRET = 'pos-secret-under-test'
/** Never valid anywhere: the 401 check must not depend on the host's configuration. */
const BAD_SECRET = 'definitely-not-the-secret'
/** A port nothing listens on. The transport's own failure needs no host under test. */
const DEAD_HOST = 'http://127.0.0.1:1'

const SESSION: StandInSession = {
  session_ref: 'scanncart-counter-1-1',
  cart_id: 'cart-1',
  cart_code: 'CODE-1',
  cart_status: 'active'
}

/** Every stand-in a test started, torn down after it whether the test passed or threw. */
const running: PosWebappStandIn[] = []

afterEach(async () => {
  for (const standIn of running.splice(0)) await standIn.stop()
})

async function serve(stationIds: string[] = [STATION]): Promise<PosWebappStandIn> {
  const standIn = new PosWebappStandIn(SECRET, stationIds)
  if (stationIds.includes(STATION)) standIn.openSession(STATION, SESSION)
  await standIn.start()
  running.push(standIn)
  return standIn
}

async function startStandIn(stationIds: string[]): Promise<PosWebappStandIn> {
  const standIn = new PosWebappStandIn(SECRET, stationIds)
  await standIn.start()
  running.push(standIn)
  return standIn
}

async function waitFor(predicate: () => boolean, label: string, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    if (predicate()) return
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label}`)
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
}

/** What a protocol check needs from whichever host it is pointed at. */
interface PosTarget {
  label: string
  baseUrl: string
  /** A secret valid on this host. */
  secret: string
  /** A station id this host knows. */
  knownStationId: string
  /** A station id this host does not know. */
  unknownStationId: string
  /** A session_ref that is not open on this host. */
  deadSessionRef: string
}

/** A freshly started stand-in, with its station registered and no session open. */
async function standInTarget(): Promise<PosTarget> {
  const standIn = await startStandIn([STATION])
  return {
    label: 'the in-repo stand-in',
    baseUrl: standIn.url,
    secret: SECRET,
    knownStationId: STATION,
    unknownStationId: 'counter-not-registered',
    deadSessionRef: 'scanncart-counter-1-not-open'
  }
}

/**
 * The real host, from the environment — or null when none is configured.
 *
 * A base URL with no secret or station id is an error rather than "not configured". A half-set
 * environment would run the suite against a host it cannot authenticate to and report 401s as the
 * route's behaviour, which is the green-that-checks-nothing the whole opt-in exists to avoid.
 */
function realTargetFromEnvironment(): PosTarget | null {
  const baseUrl = process.env.POS_E2E_BASE_URL
  if (!baseUrl) return null

  const secret = process.env.POS_E2E_SECRET
  const stationId = process.env.POS_E2E_STATION_ID
  if (!secret || !stationId) {
    throw new Error(
      'POS_E2E_BASE_URL is set but POS_E2E_SECRET and/or POS_E2E_STATION_ID are not: a fidelity run ' +
        'against a host it cannot authenticate to would check nothing. Set all three — see ' +
        '`make verify-pos-routes`.'
    )
  }

  return {
    label: 'a real pushcart-web',
    baseUrl: baseUrl.replace(/\/+$/, ''),
    secret,
    knownStationId: stationId,
    // Overridable, because the one a fresh clone would pick might, by accident, be registered.
    unknownStationId: process.env.POS_E2E_UNKNOWN_STATION_ID ?? 'scanncart-station-not-registered',
    deadSessionRef: process.env.POS_E2E_DEAD_SESSION_REF ?? 'scanncart-session-not-open'
  }
}

interface ProtocolCheck {
  name: string
  run: (target: PosTarget) => Promise<void>
}

/**
 * The checks both hosts must answer identically. Everything here is something the *contract* promises
 * and a real pushcart-web can be asked without owning its database: the auth header, the station's
 * existence, the snapshot validation, and which status means "the session is over".
 *
 * Deliberately absent, because they need state only the stand-in can be given: a 500, a cart that
 * reconciles, a session that ends mid-run. Those are the next describe down.
 */
const PROTOCOL_CHECKS: ProtocolCheck[] = [
  {
    name: 'reaches the session route with a valid token',
    run: async (target) => {
      const probe = await probePosSession(target.baseUrl, target.secret, target.knownStationId)
      expect(probe.unreachable).toBeNull()
      expect(probe.status).toBe(200)
      // `data: null` is "no session open" and an object is one — both are healthy, and a real host
      // may be either depending on whether a customer is mid-shop.
      if (probe.data !== null) {
        expect(typeof probe.data.session_ref).toBe('string')
        expect(typeof probe.data.cart_id).toBe('string')
      }
    }
  },
  {
    name: 'reads a wrong secret as 401, for the poll and for Test connection',
    run: async (target) => {
      expect(await probePosSession(target.baseUrl, BAD_SECRET, target.knownStationId)).toEqual({
        status: 401,
        data: null,
        unreachable: null
      })
      await expect(
        fetchRemoteSession(target.baseUrl, BAD_SECRET, target.knownStationId)
      ).rejects.toBeInstanceOf(PosTransportError)
    }
  },
  {
    name: 'reads an unregistered station as 404',
    run: async (target) => {
      expect(
        (await probePosSession(target.baseUrl, target.secret, target.unknownStationId)).status
      ).toBe(404)
      await expect(
        fetchRemoteSession(target.baseUrl, target.secret, target.unknownStationId)
      ).rejects.toBeInstanceOf(PosTransportError)
    }
  },
  {
    name: 'reads a snapshot the route refuses to parse as 400, not as a conflict',
    run: async (target) => {
      // Validation runs before the reconcile, so this needs no session — which is what makes it
      // askable of a host whose carts nobody controls.
      await expect(
        postCartSync(target.baseUrl, target.secret, {
          session_ref: target.deadSessionRef,
          station_id: target.knownStationId,
          items: [
            { class_name: 'soda', quantity: 1, max_confidence: 0.9 },
            { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
          ]
        })
      ).rejects.toBeInstanceOf(PosTransportError)
    }
  },
  {
    name: 'reads a snapshot for a session that is not open as a conflict',
    run: async (target) => {
      await expect(
        postCartSync(target.baseUrl, target.secret, {
          session_ref: target.deadSessionRef,
          station_id: target.knownStationId,
          items: [{ class_name: 'soda', quantity: 1, max_confidence: 0.9 }]
        })
      ).rejects.toBeInstanceOf(PosConflictError)
    }
  },
  {
    name: 'refuses a negative review count as 400 (basket mode), before any session lookup',
    run: async (target) => {
      const refused = postCartSync(target.baseUrl, target.secret, {
        session_ref: target.deadSessionRef,
        station_id: target.knownStationId,
        items: [],
        pending_review: -1
      })
      await expect(refused).rejects.toBeInstanceOf(PosTransportError)
      await expect(refused).rejects.toThrow(/400/)
    }
  },
  {
    name: 'refuses a lost flag that is not a boolean as 400, not as a reconcile 500',
    run: async (target) => {
      // pos_reconcile casts `lost` with ::boolean, which throws on a value Postgres cannot read
      // as one; the route has to refuse it before the RPC sees it.
      const refused = postCartSync(target.baseUrl, target.secret, {
        session_ref: target.deadSessionRef,
        station_id: target.knownStationId,
        items: [
          {
            class_name: 'soda',
            quantity: 1,
            max_confidence: 0.9,
            lost: 'abc' as unknown as boolean
          }
        ]
      })
      await expect(refused).rejects.toBeInstanceOf(PosTransportError)
      await expect(refused).rejects.toThrow(/400/)
    }
  },
  {
    name: 'reads a review sync for a session that is not open as a conflict',
    run: async (target) => {
      await expect(
        postCartSync(target.baseUrl, target.secret, {
          session_ref: target.deadSessionRef,
          station_id: target.knownStationId,
          items: [{ class_name: 'soda', quantity: 1, max_confidence: 0.9 }],
          pending_review: 1,
          review_reasons: ['two items crossed the opening at once']
        })
      ).rejects.toBeInstanceOf(PosConflictError)
    }
  },
  {
    name: 'reads a wrong secret on the sync route as a transport failure too',
    run: async (target) => {
      await expect(
        postCartSync(target.baseUrl, BAD_SECRET, {
          session_ref: target.deadSessionRef,
          station_id: target.knownStationId,
          items: []
        })
      ).rejects.toBeInstanceOf(PosTransportError)
    }
  }
]

describe('the POS routes over the wire — the in-repo stand-in', () => {
  let target: PosTarget
  beforeEach(async () => {
    target = await standInTarget()
  })

  it.each(PROTOCOL_CHECKS)('$name', async ({ run }) => {
    await run(target)
  })
})

const realTarget = realTargetFromEnvironment()

describe.skipIf(realTarget === null)('the POS routes over the wire — a real pushcart-web', () => {
  it.each(PROTOCOL_CHECKS)('$name', async ({ run }) => {
    await run(realTarget as PosTarget)
  })
})

describe('the in-repo stand-in: what only it can be asked', () => {
  it('carries the token and station id the route authenticates', async () => {
    const standIn = await serve()

    await fetchRemoteSession(standIn.url, SECRET, STATION)

    const [request] = standIn.requests
    expect(request.method).toBe('GET')
    expect(request.path).toBe('/api/pos/session')
    expect(request.stationId).toBe(STATION)
    expect(request.token).toBe(SECRET)
  })

  it('round-trips a real snapshot and reads cart_totals back', async () => {
    const standIn = await serve()
    standIn.prices.soda = 2.5

    const response = await postCartSync(standIn.url, SECRET, {
      session_ref: SESSION.session_ref,
      station_id: STATION,
      items: [{ class_name: 'soda', quantity: 2, max_confidence: 0.91 }]
    })

    expect(standIn.requests.at(-1)?.payload).toEqual({
      session_ref: SESSION.session_ref,
      station_id: STATION,
      items: [{ class_name: 'soda', quantity: 2, max_confidence: 0.91 }]
    })
    expect(response.cart_totals).toEqual({ item_count: 1, subtotal: 5 })
    expect(standIn.cartItems('cart-1')).toEqual({ soda: 2 })
  })

  it('reads no session as a healthy null, not an error', async () => {
    const standIn = await startStandIn([STATION])
    expect(await fetchRemoteSession(standIn.url, SECRET, STATION)).toBeNull()
  })

  it('relays a snapshot it rejects, and injects a 500 the routes cannot be asked to produce', async () => {
    const standIn = await serve()

    await expect(
      postCartSync(standIn.url, SECRET, {
        session_ref: SESSION.session_ref,
        station_id: STATION,
        items: [
          { class_name: 'soda', quantity: 1, max_confidence: 0.9 },
          { class_name: 'soda', quantity: 1, max_confidence: 0.9 }
        ]
      })
    ).rejects.toBeInstanceOf(PosTransportError)
    // The body left the client: the desktop surfaces the refusal rather than quietly preventing it.
    expect(standIn.requests.at(-1)?.payload?.items).toHaveLength(2)

    standIn.setMode('server_error')
    await expect(
      postCartSync(standIn.url, SECRET, {
        session_ref: SESSION.session_ref,
        station_id: STATION,
        items: []
      })
    ).rejects.toBeInstanceOf(PosTransportError)
  })
})

describe('a host that is not there', () => {
  it('fails as transport, not conflict, when nothing is listening', async () => {
    await expect(fetchRemoteSession(DEAD_HOST, SECRET, STATION)).rejects.toBeInstanceOf(
      PosTransportError
    )
    expect((await probePosSession(DEAD_HOST, SECRET, STATION)).unreachable).toBeTruthy()
  })
})

interface SessionHarness {
  cfg: PosConfig
  states: (PosState | null)[]
  /** Wall-clock ms of each session poll, so a test can see the retry cadence rather than assume it. */
  attempts: number[]
  /**
   * The retry delay each emit announced (`retryAtMs - now`), so the backoff can be asserted exactly
   * rather than inferred from timer spacing — which is jitter on a loaded CI box.
   */
  retryDelays: number[]
  /** Put another product on the counter, as the sidecar would report it. */
  addItem: (className: string) => void
  orchestrator: PosSessionOrchestrator
}

const stateOf = (harness: SessionHarness): PosState | null =>
  harness.states.length > 0 ? harness.states[harness.states.length - 1] : null

/**
 * The real orchestrator over the real transport, with only the sidecar faked — which is what makes
 * this an integration test rather than a second unit suite. The poll intervals are small so an
 * outage of a few hundred milliseconds is several ticks; `itemClass` makes the fake sidecar report
 * one committed track, stamped on its read so its commit time does not chase `now`.
 */
function sessionOver(standIn: PosWebappStandIn, itemClass?: string): SessionHarness {
  const cfg: PosConfig = {
    ...DEFAULT_POS_CONFIG,
    posBaseUrl: standIn.url,
    posSecret: SECRET,
    stationId: STATION,
    commitDwellS: 0.1,
    removeSettleS: 0.1,
    minCommitConf: 0.5,
    unboundPollMs: 20,
    sessionPollMs: 20,
    logsPollMs: 20
  }
  const states: (PosState | null)[] = []
  const attempts: number[] = []
  const retryDelays: number[] = []
  const tracks: SidecarLogEvent[] = []
  const pending: string[] = itemClass ? [itemClass] : []
  let nextTrackId = 1
  const addItem = (className: string): void => {
    pending.push(className)
  }

  const orchestrator = new PosSessionOrchestrator({
    getConfig: async () => cfg,
    getHealth: async () => ({ state: 'running' }),
    startCapture: async () => ({ ok: true, status: 200 }),
    getLogs: async () => {
      // Stamped at the read, not at construction: `syncOnce` only fetches logs while bound, so this
      // timestamp is always after `bindAt`. A track entered before the binding is the previous
      // customer's and is excluded by design — the harness must not fake its way past that.
      const now = Date.now() / 1000
      for (const className of pending.splice(0)) {
        tracks.push({ trackId: nextTrackId, className, maxConf: 0.9, enteredAt: now, leftAt: null })
        nextTrackId += 1
      }
      return { sessionId: 1, events: tracks }
    },
    getRemoteSession: (stationId) => {
      attempts.push(Date.now())
      return fetchRemoteSession(cfg.posBaseUrl, cfg.posSecret, stationId)
    },
    postSync: (payload) => postCartSync(cfg.posBaseUrl, cfg.posSecret, payload),
    emit: (state) => {
      states.push(state)
      const retryAtMs = state?.retryAtMs
      if (typeof retryAtMs === 'number') retryDelays.push(retryAtMs - Date.now())
    }
  })

  return { cfg, states, attempts, retryDelays, addItem, orchestrator }
}

describe("the smoke test's §3 rows a machine can hold", () => {
  it('row 1 — a dead pushcart-web surfaces the error and a backing-off retry, not a flood', async () => {
    const standIn = await serve()
    const harness = sessionOver(standIn, 'soda')
    await harness.orchestrator.start()

    await waitFor(() => standIn.cartItems('cart-1').soda === 1, 'the first snapshot to reconcile')
    expect(harness.states.some((state) => state?.phase === 'bound')).toBe(true)

    harness.attempts.length = 0
    await standIn.stop()

    // A fixed window of outage: at the un-backed-off cadence this is the poll interval several times
    // over, and backing off is what keeps the attempts to a handful.
    await new Promise((resolve) => setTimeout(resolve, 500))
    expect(harness.attempts.length).toBeGreaterThanOrEqual(2)
    expect(harness.attempts.length).toBeLessThanOrEqual(10)

    const state = stateOf(harness)
    expect(state?.phase).toBe('error')
    expect(state?.error).toBeTruthy()
    // The panel's "retrying in Ns" is this deadline, handed to the renderer to count down locally.
    expect(typeof state?.retryAtMs).toBe('number')
    expect(state!.retryAtMs!).toBeGreaterThan(0)

    // Backoff, asserted on the delay the loop chose rather than on timer spacing: consecutive
    // failures must widen the interval instead of holding the poll cadence.
    const delays = harness.retryDelays.filter((delay) => delay > 0)
    expect(delays.length).toBeGreaterThanOrEqual(2)
    expect(delays[1]).toBeGreaterThan(delays[0] * 1.5)

    harness.orchestrator.stop()
  })

  it('row 2 — pushcart-web back converges the cart, leaves no duplicate rows, and clears the countdown', async () => {
    const standIn = await serve()
    const harness = sessionOver(standIn, 'soda')
    await harness.orchestrator.start()
    await waitFor(() => standIn.cartItems('cart-1').soda === 1, 'the first snapshot to reconcile')

    const port = standIn.port
    await standIn.stop()
    await waitFor(() => stateOf(harness)?.error != null, 'the outage to be reported')

    await standIn.start(port)
    await waitFor(
      () => stateOf(harness)?.error === null && stateOf(harness)?.retryAtMs === null,
      'recovery'
    )
    await waitFor(() => stateOf(harness)?.phase === 'bound', 'the panel to read bound again')

    // Nothing was lost and nothing was duplicated across the outage: the cart is where it was.
    expect(standIn.cartItems('cart-1')).toEqual({ soda: 1 })
    const before = standIn.requests.filter((request) => request.path === '/api/pos/sync').length

    // The next change converges the cart, one row per class however many snapshots arrived.
    harness.addItem('century_tuna')
    await waitFor(
      () => standIn.cartItems('cart-1').century_tuna === 1,
      'the next snapshot to reconcile'
    )
    expect(standIn.cartItems('cart-1')).toEqual({ soda: 1, century_tuna: 1 })
    const after = standIn.requests.filter((request) => request.path === '/api/pos/sync').length
    expect(after).toBeGreaterThan(before)

    harness.orchestrator.stop()
  })

  it('row 8 tail — a cart that ends under the desktop unbinds it, without an error', async () => {
    const standIn = await serve()
    const harness = sessionOver(standIn)
    await harness.orchestrator.start()
    await waitFor(() => stateOf(harness)?.phase === 'bound', 'the panel to bind')

    // The customer tapped Finish on the tablet: pushcart-web answers with a paid cart, which is the
    // desktop's cue to stop syncing while capture keeps running.
    standIn.openSession(STATION, { ...SESSION, cart_status: 'paid' })

    await waitFor(() => stateOf(harness)?.phase === 'unbound', 'the desktop to unbind')
    expect(stateOf(harness)?.cartCode).toBeNull()
    expect(stateOf(harness)?.error).toBeNull()

    harness.orchestrator.stop()
  })

  it('a 409 from the reconcile unbinds rather than retrying', async () => {
    // The poll/sync race: the session is still reported open, and the snapshot that follows it is
    // refused. Retrying that is a flood; the desktop ends the binding instead.
    const standIn = await serve()
    const harness = sessionOver(standIn, 'soda')
    await harness.orchestrator.start()
    await waitFor(() => standIn.cartItems('cart-1').soda === 1, 'the first snapshot to reconcile')

    standIn.setMode('conflict')
    // A change is what makes the desktop post at all: an unchanged snapshot is not re-sent, which is
    // why the race needs the cart to move to be observable.
    harness.addItem('century_tuna')
    await waitFor(
      () => harness.states.some((state) => state?.phase === 'unbound'),
      'the 409 to unbind'
    )
    expect(harness.states.some((state) => state?.phase === 'error')).toBe(false)

    harness.orchestrator.stop()
  })

  it('row 7 — a rotated secret and an unregistered station each surface as an error', async () => {
    const rotated = await serve()
    rotated.setSecret('rotated-on-the-server')
    const wrongSecret = sessionOver(rotated)
    await wrongSecret.orchestrator.start()
    await waitFor(() => stateOf(wrongSecret)?.phase === 'error', 'the 401 to surface')
    expect(stateOf(wrongSecret)?.error).toContain('401')
    wrongSecret.orchestrator.stop()

    const unregistered = await startStandIn([])
    const wrongStation = sessionOver(unregistered)
    await wrongStation.orchestrator.start()
    await waitFor(() => stateOf(wrongStation)?.phase === 'error', 'the 404 to surface')
    expect(stateOf(wrongStation)?.error).toContain('404')
    wrongStation.orchestrator.stop()
  })
})
