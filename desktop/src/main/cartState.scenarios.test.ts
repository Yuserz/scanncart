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
}

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
        return checkpoint
      }),
      events: checkEvents(raw.events, where)
    }
  })
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
  describe.each(fixtures.map((fixture) => [fixture.name, fixture] as const))(
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
