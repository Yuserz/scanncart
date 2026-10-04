// @vitest-environment node
// The counting-accuracy corpus (spec §7.1): score the recorded scenarios against the cart they were
// supposed to produce.
//
// The model says which products are in a frame; whether the *cart* ends up right is something
// `cartState.ts` infers from how those detections persist. That inference is what a checkout depends
// on, and this is the only test that checks it against ground truth rather than against hand-written
// events.
//
// What runs in CI and what does not is worth being precise about, because a gate that silently
// checks nothing is worse than none:
//
//   * **Always:** the loader, the validation and the scoring against a temporary fixture written by
//     the test itself. That is the wiring, and it runs on a bare checkout.
//   * **When a corpus has been recorded:** one test per fixture, asserting the cart at every
//     checkpoint. Until then the directory holds no `.json` and the suite says so out loud rather
//     than skipping in silence.
//
// Two kinds of scenario, scored by the rule each one is about. A **counter** scenario (items set
// down and taken back) is scored through `deriveCartState`, from the track log. A **basket** scenario
// (items moved through the opening into or out of the cart band) carries its zone preset and the
// recorded frame stream, and is scored through the real `BasketTracker` — the transfer machine and
// the ledger, fed the same fresh frames the app's own WebSocket would have delivered — because a
// deposit is a path through regions, which a track log cannot describe.
//
// The fixtures are committed (they are small JSON); the videos and scripts are not, because they
// live in the sidecar's gitignored workspace. They are produced by
// `sidecar/tools/replay_scenarios.py` — see `__fixtures__/scenarios/README.md` for both formats.

import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { deriveCartState, type TrackEvent } from './cartState'
import { DEFAULT_POS_CONFIG } from './posConfig'
import type { ZonePreset } from './transferGeometry'
import { DEFAULT_TRANSFER_CONFIG } from './transferState'
import { BasketTracker, type StreamFrame } from './transferStream'

const here = dirname(fileURLToPath(import.meta.url))
const FIXTURE_DIR = join(here, '__fixtures__', 'scenarios')
const README = join(FIXTURE_DIR, 'README.md')

/**
 * The settings the checkpoints are scored under: **the app's current defaults**, not the values
 * recorded beside a fixture. That is what makes tuning `commitDwellS`/`removeSettleS`/`minCommitConf`
 * a test run instead of a trip to the counter — the events are the recording, and the rule applied
 * to them is the one the app ships today. The fixture's `asserted_with` is what the person who wrote
 * the checkpoints assumed, and it is printed when one fails.
 */
const CFG = {
  commitDwellS: DEFAULT_POS_CONFIG.commitDwellS,
  removeSettleS: DEFAULT_POS_CONFIG.removeSettleS,
  minCommitConf: DEFAULT_POS_CONFIG.minCommitConf
}

interface Checkpoint {
  t_s: number
  cart: Record<string, number>
  /** Basket only: the review items the ledger should hold. Absent means not asserted. */
  review?: number
}

/** One recorded fresh frame, on the video's timeline (`replay_scenarios.stream_frame`). */
type RecordedFrame = Pick<StreamFrame, 'ts' | 'seq' | 'detections'>

interface ScenarioFixture {
  name: string
  description: string
  video: string
  weights: string
  device: string
  bind_at_s: number
  frames: number
  duration_s: number
  pipeline: Record<string, unknown>
  asserted_with?: Record<string, number>
  checkpoints: Checkpoint[]
  events: TrackEvent[]
  /** Present on a basket scenario: the zones the video was shot with, and its frame stream. */
  basket?: ZonePreset
  stream?: RecordedFrame[]
  /** Basket only: what the ledger already holds when the clip starts (a removal clip's item). */
  initial_cart?: Record<string, number>
}

/**
 * The event list, checked rather than trusted.
 *
 * This is the one validation that matters here, and it is not defensive programming: a fixture whose
 * fields were camel-cased differently (`entered_at`, `max_conf`) would produce `undefined` timestamps,
 * `deriveCartState` would count nothing, and a checkpoint expecting an empty cart would **pass** —
 * a green run describing a corpus that is no longer being read. So a fixture that does not carry the
 * events it claims is an error, and a checkpoint that expects an empty cart is checked against a log
 * that was actually parsed.
 */
function checkEvents(value: unknown, where: string): TrackEvent[] {
  if (!Array.isArray(value)) throw new Error(`${where}: "events" must be an array`)
  return value.map((raw, index) => {
    const at = `${where}: events[${index}]`
    if (typeof raw !== 'object' || raw === null) throw new Error(`${at}: expected an object`)
    const event = raw as Record<string, unknown>
    for (const key of ['sessionId', 'trackId', 'className', 'maxConf', 'enteredAt']) {
      if (event[key] === undefined) throw new Error(`${at}: missing "${key}"`)
    }
    for (const key of ['sessionId', 'trackId', 'maxConf', 'enteredAt']) {
      if (typeof event[key] !== 'number') throw new Error(`${at}: "${key}" must be a number`)
    }
    if (typeof event.className !== 'string') throw new Error(`${at}: "className" must be a string`)
    if (event.leftAt !== null && typeof event.leftAt !== 'number') {
      throw new Error(`${at}: "leftAt" must be null or a number`)
    }
    return event as unknown as TrackEvent
  })
}

/**
 * The frame stream, checked for the same reason the events are: a frame whose `ts` were missing would
 * be fed at `undefined`, nothing would ever transfer, and a checkpoint expecting an empty basket
 * would pass against a stream nobody read.
 */
function checkStream(value: unknown, where: string): RecordedFrame[] {
  if (!Array.isArray(value)) throw new Error(`${where}: a basket scenario needs a "stream" array`)
  return value.map((raw, index) => {
    const at = `${where}: stream[${index}]`
    const frame = raw as Record<string, unknown>
    if (typeof frame?.ts !== 'number') throw new Error(`${at}: "ts" must be a number`)
    if (typeof frame.seq !== 'number') throw new Error(`${at}: "seq" must be a number`)
    if (!Array.isArray(frame.detections)) throw new Error(`${at}: "detections" must be an array`)
    frame.detections.forEach((d: Record<string, unknown>, j: number) => {
      const dat = `${at}.detections[${j}]`
      if (d.track_id !== null && typeof d.track_id !== 'number') {
        throw new Error(`${dat}: "track_id" must be null or a number`)
      }
      if (typeof d.cls !== 'string') throw new Error(`${dat}: "cls" must be a string`)
      if (typeof d.conf !== 'number') throw new Error(`${dat}: "conf" must be a number`)
      const box = d.box as unknown[]
      if (!Array.isArray(box) || box.length !== 4 || box.some((v) => typeof v !== 'number')) {
        throw new Error(`${dat}: "box" must be four numbers`)
      }
    })
    return frame as unknown as RecordedFrame
  })
}

/** Every fixture in `dir`, in name order. Throws on a file that is not a usable fixture. */
function loadFixtures(dir: string = FIXTURE_DIR): ScenarioFixture[] {
  if (!existsSync(dir)) return []
  const files = readdirSync(dir)
    .filter((name) => name.endsWith('.json'))
    .sort()
  return files.map((file) => {
    const raw = JSON.parse(readFileSync(join(dir, file), 'utf8')) as Record<string, unknown>
    const where = file
    notFalsy(raw.name, `${where}: "name"`)
    notFalsy(raw.bind_at_s, `${where}: "bind_at_s"`)
    const checkpoints = raw.checkpoints as Checkpoint[] | undefined
    if (!Array.isArray(checkpoints) || checkpoints.length === 0) {
      throw new Error(`${where}: "checkpoints" must be a non-empty list`)
    }
    return {
      ...(raw as unknown as ScenarioFixture),
      checkpoints: checkpoints.map((checkpoint, index) => {
        if (typeof checkpoint?.t_s !== 'number') {
          throw new Error(`${where}: checkpoints[${index}].t_s must be a number`)
        }
        if (typeof checkpoint.cart !== 'object' || checkpoint.cart === null) {
          throw new Error(`${where}: checkpoints[${index}].cart must be an object`)
        }
        if (checkpoint.review !== undefined && raw.basket === undefined) {
          throw new Error(
            `${where}: checkpoints[${index}].review is only meaningful in a basket scenario`
          )
        }
        return checkpoint
      }),
      initial_cart: checkInitialCart(raw.initial_cart, raw.basket !== undefined, where),
      events: checkEvents(raw.events, where),
      ...(raw.basket !== undefined ? { stream: checkStream(raw.stream, where) } : {})
    }
  })
}

/**
 * A basket fixture's starting inventory. Checked because a misspelled quantity would seed nothing,
 * and the removal it was written for would then score as review — a failure that reads as the
 * machine's fault rather than the fixture's.
 */
function checkInitialCart(
  value: unknown,
  basket: boolean,
  where: string
): Record<string, number> | undefined {
  if (value === undefined) return undefined
  if (!basket) throw new Error(`${where}: "initial_cart" is only meaningful in a basket scenario`)
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${where}: "initial_cart" must be an object of class -> quantity`)
  }
  for (const [className, quantity] of Object.entries(value)) {
    if (!Number.isInteger(quantity) || (quantity as number) < 1) {
      throw new Error(`${where}: initial_cart[${className}] must be a whole number >= 1`)
    }
  }
  return value as Record<string, number>
}

function notFalsy(value: unknown, where: string): void {
  if (value === undefined) throw new Error(`${where} is missing`)
}

/** The cart `deriveCartState` holds at one checkpoint, as the plain object the script wrote. */
function cartAt(fixture: ScenarioFixture, t_s: number): Record<string, number> {
  const derived = deriveCartState(fixture.events, t_s, fixture.bind_at_s, CFG)
  const cart: Record<string, number> = {}
  for (const [className, entry] of derived) cart[className] = entry.quantity
  return cart
}

/**
 * How far before a seeded clip's first frame its session is bound. Longer than the tracker's
 * empty-basket window (`baselineS`, 3 s by default), so items the clip starts with are the
 * session's own rather than "the basket was not empty at Start".
 */
const SEED_LEAD_S = 60

interface BasketCheckpointResult {
  cart: Record<string, number>
  review: number
  reasons: string[]
}

/**
 * Walk a basket fixture's stream through a real `BasketTracker`, reading the ledger at each
 * checkpoint. One pass in time order, as the app would see it: frames before `bind_at_s` still reach
 * the machine (the tracker's socket is open whether or not a customer is), and the session binds at
 * `bind_at_s` exactly, which is when the empty-basket baseline starts looking.
 *
 * The confidence floor is the scanner's own `conf_threshold` the fixture was replayed at, because
 * that is the value `PosController` pushes into the tracker; everything else the transfer machine
 * uses is the app's current default, so tuning it is a test run like the counter's settings.
 */
async function basketCheckpoints(fixture: ScenarioFixture): Promise<BasketCheckpointResult[]> {
  const preset = fixture.basket as ZonePreset
  const conf = fixture.pipeline?.conf_threshold
  let clock = 0
  const tracker = new BasketTracker(preset, {
    now: () => clock,
    transfer: {
      ...DEFAULT_TRANSFER_CONFIG,
      confThreshold: typeof conf === 'number' ? conf : DEFAULT_TRANSFER_CONFIG.confThreshold
    }
  })
  const stream = [...(fixture.stream ?? [])].sort((a, b) => a.ts - b.ts)
  const checkpoints = [...fixture.checkpoints].sort((a, b) => a.t_s - b.t_s)

  let bound = false
  const bindIfDue = async (t: number): Promise<void> => {
    if (bound || t < fixture.bind_at_s) return
    clock = fixture.bind_at_s
    await tracker.bind(`scenario:${fixture.name}`)
    bound = true
  }

  // A clip that starts mid-session: the customer bound earlier and these items went in then. Bind
  // before the first frame, far enough back that the empty-basket window is already over, and put
  // the items in through the ledger's own `apply` as confirmed deposits — the path a real earlier
  // deposit took — so nothing about the ledger's state is constructed by hand.
  if (fixture.initial_cart && Object.keys(fixture.initial_cart).length > 0) {
    const firstT = Math.min(fixture.bind_at_s, stream[0]?.ts ?? fixture.bind_at_s)
    clock = firstT - SEED_LEAD_S
    await tracker.bind(`scenario:${fixture.name}`)
    bound = true
    let seq = 0
    for (const [className, quantity] of Object.entries(fixture.initial_cart)) {
      for (let unit = 0; unit < quantity; unit++) {
        seq += 1
        const applied = tracker.ledger.apply(
          {
            kind: 'inbound',
            className,
            completionConf: 1,
            completedAt: clock + seq / 1000,
            trackId: -seq
          },
          'seed'
        )
        if (applied.status !== 'applied') {
          throw new Error(`${fixture.name}: seeding ${className} was ${applied.status}`)
        }
      }
    }
  }

  const results: BasketCheckpointResult[] = []
  let next = 0
  for (const checkpoint of checkpoints) {
    while (next < stream.length && stream[next].ts <= checkpoint.t_s) {
      const frame = stream[next++]
      await bindIfDue(frame.ts)
      clock = frame.ts
      tracker.onFrame({ type: 'frame', fresh: true, mirrored: false, ...frame })
    }
    await bindIfDue(checkpoint.t_s)
    const cart: Record<string, number> = {}
    for (const [className, entry] of tracker.snapshot()) cart[className] = entry.quantity
    const review = tracker.pendingReview()
    results.push({ cart, review: review.length, reasons: review.map((r) => r.reason) })
  }
  return results
}

const fixtures = loadFixtures()

describe('scenario scoring (spec §7.1)', () => {
  it('scores a fixture end to end, so the wiring is exercised with no corpus recorded', () => {
    // Written here rather than committed into the corpus directory on purpose: a synthetic fixture
    // sitting beside the real ones would make CI green for a corpus that does not exist yet, which
    // is the "measurement wired to nothing" this test file is meant to avoid. It is a temp file, its
    // only job is to run the loader and the derivation, and its numbers are chosen to fail if the
    // rule changed: one track committed after a 3 s dwell.
    const dir = mkdtempSync(join(tmpdir(), 'scenarios-'))
    const fixture = {
      name: 'inline',
      bind_at_s: 0,
      checkpoints: [{ t_s: 110, cart: { soda: 1 } }],
      events: [
        { sessionId: 1, trackId: 1, className: 'soda', maxConf: 0.9, enteredAt: 100, leftAt: null }
      ]
    }
    writeFileSync(join(dir, 'inline.json'), JSON.stringify(fixture), 'utf8')

    const [loaded] = loadFixtures(dir)
    expect(loaded.name).toBe('inline')
    expect(cartAt(loaded, 110)).toEqual({ soda: 1 })
    // Before the dwell the item has not committed: this is what makes the case depend on the app's
    // own `commitDwellS` rather than on the fixture's numbers.
    expect(cartAt(loaded, 102.9)).toEqual({})
    expect(CFG.commitDwellS).toBe(DEFAULT_POS_CONFIG.commitDwellS)
  })

  it('refuses a fixture whose events are not in the desktop vocabulary', () => {
    // The failure this exists for is a *pass*: with `entered_at` instead of `enteredAt` every
    // timestamp is undefined, nothing counts, and a checkpoint expecting an empty cart goes green.
    const dir = mkdtempSync(join(tmpdir(), 'scenarios-bad-'))
    writeFileSync(
      join(dir, 'bad.json'),
      JSON.stringify({
        name: 'bad',
        bind_at_s: 0,
        checkpoints: [{ t_s: 10, cart: {} }],
        events: [
          { sessionId: 1, trackId: 1, className: 'soda', maxConf: 0.9, entered_at: 1, leftAt: null }
        ]
      }),
      'utf8'
    )
    expect(() => loadFixtures(dir)).toThrow(/enteredAt/)
  })

  /**
   * One item carried from the top of the frame down into the bottom cart band, starting at
   * `startS`, at 10 fps, in small steps so no region sees a jump; it reaches the cart band 1.0 s
   * in and holds there for 1.5 s. Checkpoints just before the path starts and after it settles.
   */
  function depositFixture(startS: number, bindAtS: number): ScenarioFixture {
    const detections = (y: number): RecordedFrame['detections'] => [
      { track_id: 7, cls: 'century_tuna', conf: 0.9, box: [0.4, y - 0.05, 0.6, y + 0.05] }
    ]
    const stream: RecordedFrame[] = []
    let seq = 0
    for (let i = 0; i <= 14; i++) {
      stream.push({ ts: startS + i / 10, seq: ++seq, detections: detections(0.15 + i * 0.05) })
    }
    for (let i = 0; i <= 15; i++) {
      stream.push({ ts: startS + 1.5 + i / 10, seq: ++seq, detections: detections(0.85) })
    }
    const dir = mkdtempSync(join(tmpdir(), 'scenarios-basket-'))
    writeFileSync(
      join(dir, 'deposit.json'),
      JSON.stringify({
        name: 'deposit',
        bind_at_s: bindAtS,
        pipeline: { conf_threshold: 0.5 },
        basket: { cartEdge: 'bottom', insideFraction: 0.35, openingFraction: 0.2 },
        checkpoints: [
          { t_s: startS - 0.1, cart: {}, review: 0 },
          { t_s: startS + 3.5, cart: { century_tuna: 1 }, review: 0 }
        ],
        events: [],
        stream
      }),
      'utf8'
    )
    return loadFixtures(dir)[0]
  }

  it('scores a basket fixture end to end through the transfer machine and the ledger', async () => {
    // Started well after the empty-basket window (bind + 3 s): one deposit, nothing in review.
    const results = await basketCheckpoints(depositFixture(5, 1))
    expect(results.map((r) => r.cart)).toEqual([{}, { century_tuna: 1 }])
    expect(results.map((r) => r.review)).toEqual([0, 0])
  })

  /** A tuna lifted from the bottom cart band up and out, held clear for 1.5 s, starting at 5 s. */
  function removalFixture(initialCart?: Record<string, number>): ScenarioFixture {
    const detections = (y: number): RecordedFrame['detections'] => [
      { track_id: 9, cls: 'century_tuna', conf: 0.9, box: [0.4, y - 0.05, 0.6, y + 0.05] }
    ]
    const stream: RecordedFrame[] = []
    let seq = 0
    for (let i = 0; i <= 14; i++) {
      stream.push({ ts: 5 + i / 10, seq: ++seq, detections: detections(0.85 - i * 0.05) })
    }
    for (let i = 0; i <= 15; i++) {
      stream.push({ ts: 6.5 + i / 10, seq: ++seq, detections: detections(0.15) })
    }
    const dir = mkdtempSync(join(tmpdir(), 'scenarios-removal-'))
    writeFileSync(
      join(dir, 'removal.json'),
      JSON.stringify({
        name: 'removal',
        bind_at_s: 0,
        pipeline: { conf_threshold: 0.5 },
        basket: { cartEdge: 'bottom', insideFraction: 0.35, openingFraction: 0.2 },
        ...(initialCart ? { initial_cart: initialCart } : {}),
        checkpoints: [
          { t_s: 4.9, cart: initialCart ?? {}, review: 0 },
          { t_s: 8.5, cart: {}, review: 0 }
        ],
        events: [],
        stream
      }),
      'utf8'
    )
    return loadFixtures(dir)[0]
  }

  it('scores a removal clip from its initial cart', async () => {
    // The item is held when the clip starts, so lifting it out is a confirmed removal: the basket
    // goes from one tuna to empty, and the empty-basket check never sees it (the seeded session
    // bound long before the first frame).
    const results = await basketCheckpoints(removalFixture({ century_tuna: 1 }))
    expect(results.map((r) => r.cart)).toEqual([{ century_tuna: 1 }, {}])
    expect(results.map((r) => r.review)).toEqual([0, 0])
  })

  it('scores the same removal clip as review without a seed, which is why the seed exists', async () => {
    const results = await basketCheckpoints(removalFixture())
    expect(results[1].cart).toEqual({})
    expect(results[1].review).toBe(1)
    expect(results[1].reasons[0]).toMatch(/not in the basket/)
  })

  it('refuses an initial cart on a counter fixture, or one with a bad quantity', () => {
    const dir = mkdtempSync(join(tmpdir(), 'scenarios-seed-bad-'))
    const counter = { name: 'c', bind_at_s: 0, checkpoints: [{ t_s: 1, cart: {} }], events: [] }
    writeFileSync(
      join(dir, 'c.json'),
      JSON.stringify({ ...counter, initial_cart: { soda: 1 } }),
      'utf8'
    )
    expect(() => loadFixtures(dir)).toThrow(/basket scenario/)
    writeFileSync(
      join(dir, 'c.json'),
      JSON.stringify({
        ...counter,
        basket: { cartEdge: 'bottom', insideFraction: 0.35, openingFraction: 0.2 },
        stream: [],
        initial_cart: { soda: 0 }
      }),
      'utf8'
    )
    expect(() => loadFixtures(dir)).toThrow(/whole number/)
  })

  it('reports a deposit made inside the empty-basket window as review (current behaviour)', async () => {
    // The baseline check flags any product seen in the cart band during the first 3 s after bind,
    // including one that just arrived through the opening on a tracked path. Pinned here so a quick
    // first deposit's review is a known, visible behaviour rather than a surprise in a recording —
    // and so the test changes, on purpose, if the baseline learns to skip a tracked arrival.
    const results = await basketCheckpoints(depositFixture(2, 1))
    expect(results[1].cart).toEqual({ century_tuna: 1 })
    expect(results[1].review).toBe(1)
    expect(results[1].reasons[0]).toMatch(/not empty at Start/)
  })

  it('refuses a basket fixture without a readable stream', () => {
    // The failure is again a pass: no stream means no transfer, and an empty-basket checkpoint goes
    // green against frames nobody read.
    const dir = mkdtempSync(join(tmpdir(), 'scenarios-basket-bad-'))
    const base = {
      name: 'bad',
      bind_at_s: 0,
      basket: { cartEdge: 'bottom', insideFraction: 0.35, openingFraction: 0.2 },
      checkpoints: [{ t_s: 10, cart: {} }],
      events: []
    }
    writeFileSync(join(dir, 'bad.json'), JSON.stringify(base), 'utf8')
    expect(() => loadFixtures(dir)).toThrow(/stream/)
    writeFileSync(
      join(dir, 'bad.json'),
      JSON.stringify({ ...base, stream: [{ t: 1, seq: 1, detections: [] }] }),
      'utf8'
    )
    expect(() => loadFixtures(dir)).toThrow(/"ts"/)
  })

  it('refuses a review count on a counter fixture', () => {
    const dir = mkdtempSync(join(tmpdir(), 'scenarios-review-'))
    writeFileSync(
      join(dir, 'counter.json'),
      JSON.stringify({
        name: 'c',
        bind_at_s: 0,
        checkpoints: [{ t_s: 1, cart: {}, review: 0 }],
        events: []
      }),
      'utf8'
    )
    expect(() => loadFixtures(dir)).toThrow(/basket scenario/)
  })

  it('documents the corpus contract in the fixture directory it reads', () => {
    // Pinned so the directory is a real target rather than an empty folder: the README is the
    // contract for both files, and it is where a new scenario's author is sent.
    expect(existsSync(FIXTURE_DIR)).toBe(true)
    const readme = readFileSync(README, 'utf8')
    expect(readme).toContain('replay_scenarios.py')
    expect(readme).toContain('make replay-scenarios')
    expect(readme).toContain('checkpoints')
  })
})

if (fixtures.length === 0) {
  describe('the recorded corpus', () => {
    it('has not been recorded yet, which is a state rather than a skip', () => {
      // No fixtures is the corpus' real state today: it needs a camera and a scripted counter. The
      // tests above already prove the scoring path works, so this is not a hole — and when the
      // first fixture lands, this block disappears and the per-fixture assertions take its place.
      expect(fixtures).toEqual([])
      expect(existsSync(FIXTURE_DIR)).toBe(true)
    })
  })
} else {
  const counterFixtures = fixtures.filter((fixture) => fixture.basket === undefined)
  const basketFixtures = fixtures.filter((fixture) => fixture.basket !== undefined)

  describe.each(basketFixtures.map((fixture) => [fixture.name, fixture] as const))(
    'basket scenario %s',
    (_name, fixture) => {
      it('holds the scripted basket at every checkpoint', async () => {
        const results = await basketCheckpoints(fixture)
        const checkpoints = [...fixture.checkpoints].sort((a, b) => a.t_s - b.t_s)
        checkpoints.forEach((checkpoint, index) => {
          const got = results[index]
          expect(got.cart, `cart at ${checkpoint.t_s}s`).toEqual(checkpoint.cart)
          if (checkpoint.review !== undefined) {
            // The reasons ride in the message: a review count alone does not say *which* path the
            // machine refused to trust, and that is the thing tuning needs to see.
            expect(got.review, `review at ${checkpoint.t_s}s: ${JSON.stringify(got.reasons)}`).toBe(
              checkpoint.review
            )
          }
        })
      })
    }
  )

  describe.each(counterFixtures.map((fixture) => [fixture.name, fixture] as const))(
    'scenario %s',
    (_name, fixture) => {
      it.each(fixture.checkpoints.map((checkpoint) => [checkpoint.t_s, checkpoint] as const))(
        'holds the scripted cart at t=%ss',
        (_t, checkpoint) => {
          const derived = cartAt(fixture, checkpoint.t_s)
          const context = fixture.asserted_with
            ? ` (checkpoints written assuming ${JSON.stringify(fixture.asserted_with)})`
            : ''
          expect(derived, `at ${checkpoint.t_s}s${context}`).toEqual(checkpoint.cart)
        }
      )
    }
  )
}
