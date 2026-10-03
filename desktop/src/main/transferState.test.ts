// @vitest-environment node
// Synthetic track sequences through the transfer state machine.
//
// No camera, no sidecar, no pixels: each test feeds `Observation`s shaped like the ones the
// stream forwards (fresh per-inference boxes in TRANSFER_FRAME units) over the default
// preset — cart at the bottom 35% of the frame, opening the 20% above it, outside the rest —
// and asserts what the spec §Gate-B(3) requires: confirmed only after an observed, held
// completion, once per physical transfer, with reversals and identity conflicts routed to
// abort/review instead of the ledger. The `#n` tags are the plan's events table.

import { describe, it, expect, beforeEach } from 'vitest'
import {
  TransferStateMachine,
  DEFAULT_TRANSFER_CONFIG,
  regionOf,
  type MachineOutput,
  type Observation,
  type TransferEvent
} from './transferState'
import {
  DEFAULT_ZONE_PRESET,
  frameBox,
  presetRegions,
  zonePresetProblems
} from './transferGeometry'

const regions = presetRegions(DEFAULT_ZONE_PRESET)
const cfg = { ...DEFAULT_TRANSFER_CONFIG } // 2 per side, 1 s hold, scanner threshold 0.5, slop 150

// Box centres for each band (box height 60, so y is the centre minus 30).
const OUT = 200
const OPEN = 550
const IN = 820

let seq = 0
let t = 0
beforeEach(() => {
  seq = 0
  t = 0
})

/** One observation 0.2 s after the previous (5 fresh inferences a second). */
function obs(
  o: { cy: number; x?: number } & Partial<Omit<Observation, 'box' | 'seq' | 't'>>
): Observation {
  seq += 1
  t += 0.2
  const { cy, x = 460, ...rest } = o
  return {
    seq,
    t,
    trackId: 7,
    className: 'soda',
    conf: 0.9,
    box: { x, y: cy - 30, w: 80, h: 60 },
    ...rest
  }
}

/** `n` observations at one band — a hold of (n - 1) × 0.2 s. */
const at = (cy: number, n: number, extra: Partial<Observation> = {}): Observation[] =>
  Array.from({ length: n }, () => obs({ cy, ...extra }))

interface Fed {
  events: TransferEvent[]
  outputs: MachineOutput[]
  removed: NonNullable<MachineOutput['removed']>[]
}

function feed(m: TransferStateMachine, list: Observation[]): Fed {
  const events: TransferEvent[] = []
  const outputs: MachineOutput[] = []
  for (const o of list) {
    const out = m.observe(o)
    events.push(...out.events)
    outputs.push(out)
  }
  return { events, outputs, removed: outputs.flatMap((o) => (o.removed ? [o.removed] : [])) }
}

const deposit = (extra: Partial<Observation> = {}): Observation[] => [
  ...at(OUT, 2, extra),
  ...at(OPEN, 1, extra),
  ...at(IN, 6, extra)
]
const removal = (extra: Partial<Observation> = {}): Observation[] => [
  ...at(IN, 2, extra),
  ...at(OPEN, 1, extra),
  ...at(OUT, 6, extra)
]

describe('zone preset', () => {
  it('cart at the bottom: bands stack outside → opening → inside downward', () => {
    expect(zonePresetProblems(DEFAULT_ZONE_PRESET)).toEqual([])
    const r = (cy: number): string => regionOf(regions, { x: 460, y: cy - 30, w: 80, h: 60 })
    expect([r(OUT), r(OPEN), r(IN)]).toEqual(['outside', 'opening', 'inside'])
  })

  it('every edge produces usable, non-overlapping bands', () => {
    for (const cartEdge of ['bottom', 'top', 'left', 'right'] as const) {
      const p = presetRegions({ ...DEFAULT_ZONE_PRESET, cartEdge })
      expect(Object.values(p).every((b) => b.w > 0 && b.h > 0)).toBe(true)
    }
  })

  it('refuses a preset that leaves no outside band', () => {
    expect(
      zonePresetProblems({ cartEdge: 'bottom', insideFraction: 0.6, openingFraction: 0.4 })
    ).not.toEqual([])
    expect(() =>
      presetRegions({ cartEdge: 'bottom', insideFraction: 0.6, openingFraction: 0.4 })
    ).toThrow()
  })

  it('undoes the preview mirror with the sidecar rule', () => {
    const t0 = frameBox([0.1, 0.2, 0.3, 0.4], false)
    expect([t0.x, t0.y, t0.w, t0.h].map((v) => Math.round(v))).toEqual([100, 200, 200, 200])
    const m = frameBox([0.7, 0.2, 0.9, 0.4], true)
    expect(m.x).toBeCloseTo(100)
    expect(m.w).toBeCloseTo(200)
  })
})

describe('#1 deposit', () => {
  it('commits after outside×2 → opening → inside held for 1 s', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { events } = feed(m, deposit())
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ kind: 'inbound', className: 'soda', completionConf: 0.9 })
  })

  it('does not commit before the endpoint hold has elapsed', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { events } = feed(m, [...at(OUT, 2), ...at(OPEN, 1), ...at(IN, 4)]) // 0.6 s inside
    expect(events).toEqual([])
    expect(m.snapshot()[0]?.phase).toBe('inbound_pending')
  })

  it('does not commit on a single outside sighting', () => {
    const m = new TransferStateMachine(regions, cfg)
    expect(feed(m, [...at(OUT, 1), ...at(OPEN, 1), ...at(IN, 6)]).events).toEqual([])
  })

  it('a deposit hidden before the hold completes is reviewed, not committed', () => {
    const m = new TransferStateMachine(regions, cfg)
    feed(m, [...at(OUT, 2), ...at(OPEN, 1), ...at(IN, 2)])
    const expired = m.sweep(t + cfg.candidateTimeoutS + 0.1)
    expect(expired).toEqual([expect.objectContaining({ phase: 'inbound_pending', review: true })])
  })
})

describe('#2 removal', () => {
  it('commits after inside×2 → opening → outside held clear for 1 s', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { events } = feed(m, removal())
    expect(events).toHaveLength(1)
    expect(events[0].kind).toBe('outbound')
  })
})

describe('#3 two identical units', () => {
  it('two separate transfers are two events', () => {
    const m = new TransferStateMachine(regions, cfg)
    const first = feed(m, deposit({ trackId: 1 })).events
    const second = feed(m, deposit({ trackId: 2 })).events
    expect([...first, ...second].map((e) => e.kind)).toEqual(['inbound', 'inbound'])
  })
})

describe('#4 remove, then put back (same track)', () => {
  it('deposit → removal → deposit is three events, never a duplicate', () => {
    const m = new TransferStateMachine(regions, cfg)
    const kinds = [
      ...feed(m, deposit()).events,
      ...feed(m, at(IN, 4)).events, // rests in the cart: no re-fire
      ...feed(m, [...at(OPEN, 1), ...at(OUT, 6)]).events, // taken out
      ...feed(m, [...at(OPEN, 1), ...at(IN, 6)]).events // put back
    ].map((e) => e.kind)
    expect(kinds).toEqual(['inbound', 'outbound', 'inbound'])
  })
})

describe('#5 #7 no-transfer families', () => {
  it('outside presentation never crosses', () => {
    const m = new TransferStateMachine(regions, cfg)
    expect(feed(m, at(OUT, 8)).events).toEqual([])
  })

  it('hover at the opening then back out is aborted without review, and a retry still counts', () => {
    const m = new TransferStateMachine(regions, cfg)
    const hover = feed(m, [...at(OUT, 2), ...at(OPEN, 2), ...at(OUT, 1)])
    expect(hover.events).toEqual([])
    expect(hover.removed).toEqual([expect.objectContaining({ phase: 'aborted', review: false })])
    expect(feed(m, [...at(OUT, 1), ...at(OPEN, 1), ...at(IN, 6)]).events).toHaveLength(1)
  })

  it('rearranging inside the cart is no event', () => {
    const m = new TransferStateMachine(regions, cfg)
    expect(
      feed(m, [obs({ cy: IN }), obs({ cy: IN + 40 }), obs({ cy: IN - 40 }), obs({ cy: IN })]).events
    ).toEqual([])
  })

  it('lifting an item to the opening and setting it back down is no event', () => {
    const m = new TransferStateMachine(regions, cfg)
    expect(feed(m, [...at(IN, 2), ...at(OPEN, 2), ...at(IN, 4)]).events).toEqual([])
  })
})

describe('a dip back over the opening (review finding)', () => {
  it('a deposit lifted back over the opening and set down again is one +1 and no review', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { events, removed } = feed(m, [
      ...at(OUT, 2),
      ...at(OPEN, 1),
      ...at(IN, 2), // lowered in, not yet held
      ...at(OPEN, 1), // lifted back over the rim
      ...at(IN, 6) // set down and left
    ])
    expect(events.map((e) => e.kind)).toEqual(['inbound'])
    expect(removed).toEqual([])
  })

  it('the hold restarts after the dip, so it cannot complete early', () => {
    const m = new TransferStateMachine(regions, cfg)
    // 0.8 s inside, dip, then only 0.6 s back inside: neither stretch is a full 1 s hold.
    const { events } = feed(m, [
      ...at(OUT, 2),
      ...at(OPEN, 1),
      ...at(IN, 5),
      ...at(OPEN, 1),
      ...at(IN, 4)
    ])
    expect(events).toEqual([])
  })

  it('a removal that dips back over the opening is one −1 and no review', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { events, removed } = feed(m, [
      ...at(IN, 2),
      ...at(OPEN, 1),
      ...at(OUT, 2),
      ...at(OPEN, 1),
      ...at(OUT, 6)
    ])
    expect(events.map((e) => e.kind)).toEqual(['outbound'])
    expect(removed).toEqual([])
  })

  it('an item resting on the cart/opening boundary after a deposit never reviews or re-fires', () => {
    const m = new TransferStateMachine(regions, cfg)
    expect(feed(m, deposit()).events).toHaveLength(1)
    const jitter = Array.from({ length: 12 }, (_, i) => obs({ cy: i % 2 === 0 ? OPEN : IN }))
    const { events, removed } = feed(m, jitter)
    expect(events).toEqual([])
    expect(removed.filter((r) => r.review)).toEqual([])
  })

  it('jumping from outside straight into the cart is not ordered evidence of the path', () => {
    const m = new TransferStateMachine(regions, cfg)
    expect(feed(m, [...at(OUT, 2), ...at(IN, 6)]).events).toEqual([])
  })
})

describe('#9 identity conflicts', () => {
  it('a class change mid-path goes to review and commits nothing', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { events, removed } = feed(m, [...at(OUT, 2), obs({ cy: OPEN, className: 'water' })])
    expect(events).toEqual([])
    expect(removed[0]).toMatchObject({ review: true, className: 'soda' })
    expect(removed[0].reason).toContain('class changed')
  })

  it('a jump within one region goes to review', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { removed } = feed(m, [obs({ cy: OUT, x: 50 }), obs({ cy: OUT, x: 800 })])
    expect(removed[0]?.reason).toContain('box jumped')
  })
})

describe('#10 stuck mid-path', () => {
  it('a candidate that stops at the opening expires for review', () => {
    const m = new TransferStateMachine(regions, cfg)
    feed(m, [...at(OUT, 2), ...at(OPEN, 1)])
    const expired = m.sweep(t + cfg.candidateTimeoutS + 0.1)
    expect(expired).toEqual([
      expect.objectContaining({ phase: 'inbound_pending', review: true, className: 'soda' })
    ])
    expect(m.snapshot()).toEqual([])
  })
})

describe('#12 two items at once', () => {
  it('a completion while another track was in the opening goes to review', () => {
    const m = new TransferStateMachine(regions, cfg)
    feed(m, at(OUT, 2, { trackId: 1 }))
    feed(m, at(OUT, 2, { trackId: 2 }))
    feed(m, [obs({ cy: OPEN, trackId: 1 }), obs({ cy: OPEN, trackId: 2 })])
    const { events, removed } = feed(m, at(IN, 6, { trackId: 1 }))
    expect(events).toEqual([])
    expect(removed[0]?.reason).toContain('two items')
  })
})

describe('#14 confidence follows the scanner threshold', () => {
  it('below the scanner threshold never enters the machine', () => {
    const m = new TransferStateMachine(regions, cfg) // the sidecar default, 0.5
    expect(feed(m, deposit({ conf: 0.3 })).events).toEqual([])
    expect(m.snapshot()).toEqual([])
  })

  it('at 0.9 on the scanner, a 0.85 deposit is not inferred and a 0.92 one is', () => {
    const m = new TransferStateMachine(regions, cfg)
    m.setConfThreshold(0.9)
    expect(feed(m, deposit({ conf: 0.85, trackId: 1 })).events).toEqual([])
    expect(feed(m, deposit({ conf: 0.92, trackId: 2 })).events).toHaveLength(1)
  })

  it('one frame under the threshold on the path is simply not a sighting', () => {
    const m = new TransferStateMachine(regions, cfg)
    m.setConfThreshold(0.9)
    // Only one outside sighting clears 0.9, so there is no origin evidence to commit on.
    const path = [
      obs({ cy: OUT, conf: 0.95 }),
      obs({ cy: OUT, conf: 0.6 }),
      ...at(OPEN, 1),
      ...at(IN, 6)
    ]
    expect(feed(m, path).events).toEqual([])
  })

  it('a lowered threshold is honoured too, with no second floor of its own', () => {
    const m = new TransferStateMachine(regions, cfg)
    m.setConfThreshold(0.3)
    expect(feed(m, deposit({ conf: 0.35 })).events).toHaveLength(1)
  })

  it('never writes through to the shared default', () => {
    new TransferStateMachine(regions).setConfThreshold(0.95)
    expect(DEFAULT_TRANSFER_CONFIG.confThreshold).toBe(0.5)
  })

  it('ignores a value that is not a confidence', () => {
    const m = new TransferStateMachine(regions, cfg)
    m.setConfThreshold(Number.NaN)
    m.setConfThreshold(4)
    expect(m.getConfThreshold()).toBe(0.5)
  })
})

describe('first seen mid-path', () => {
  it('appearing in the opening and landing inside is reviewed, never committed', () => {
    const m = new TransferStateMachine(regions, cfg)
    const { events, removed } = feed(m, [...at(OPEN, 2), ...at(IN, 6)])
    expect(events).toEqual([])
    expect(removed[0]).toMatchObject({ review: true })
    expect(removed[0].reason).toContain('without origin evidence')
  })
})

describe('reset', () => {
  it('clears in-flight candidates', () => {
    const m = new TransferStateMachine(regions, cfg)
    feed(m, [...at(OUT, 2), ...at(OPEN, 1)])
    m.reset()
    expect(m.snapshot()).toEqual([])
  })

  it('accepts new regions without losing candidates', () => {
    const m = new TransferStateMachine(regions, cfg)
    feed(m, at(OUT, 2))
    m.setRegions(presetRegions({ ...DEFAULT_ZONE_PRESET, insideFraction: 0.3 }))
    expect(m.snapshot()).toHaveLength(1)
  })
})
