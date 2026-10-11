// @vitest-environment node
import { describe, it, expect } from 'vitest'
import {
  deriveCartState,
  deriveLiveCounts,
  type CartStateConfig,
  type TrackEvent
} from './cartState'

const cfg: CartStateConfig = { commitDwellS: 3, removeSettleS: 10, minCommitConf: 0.6 }
const BIND = 0

let nextId = 1
function ev(o: Partial<TrackEvent> & { className: string; enteredAt: number }): TrackEvent {
  return {
    sessionId: 1,
    trackId: nextId++,
    maxConf: 0.9,
    leftAt: null,
    ...o
  }
}

const qty = (events: TrackEvent[], now: number, bindAt = BIND): number =>
  deriveCartState(events, now, bindAt, cfg).get('soda')?.quantity ?? 0

describe('deriveCartState', () => {
  it('commits a track only after the dwell', () => {
    const events = [ev({ className: 'soda', enteredAt: 100 })]
    // Before entered_at + commitDwellS the track is not committed.
    expect(qty(events, 102.9)).toBe(0)
    expect(qty(events, 103)).toBe(1)
  })

  it('never counts a track shorter than the dwell (flicker)', () => {
    const events = [ev({ className: 'soda', enteredAt: 100, leftAt: 101 })]
    expect(qty(events, 200)).toBe(0)
  })

  it('holds a count through a dip shorter than the settle window', () => {
    // One item, removed at t=120. Within the settle window it still reads as present.
    const events = [ev({ className: 'soda', enteredAt: 100, leftAt: 120 })]
    expect(qty(events, 125)).toBe(1)
    // Once the settle window has passed clean of the interval, it drops.
    expect(qty(events, 135)).toBe(0)
  })

  it('keeps quantity 1 through a track swap (gap shorter than the settle window)', () => {
    const events = [
      ev({ className: 'soda', enteredAt: 100, leftAt: 130 }),
      ev({ className: 'soda', enteredAt: 131 })
    ]
    // Across the whole window the count is 1; it never reads 2 (no overlap) or 0 (the gap is
    // covered by the settle window).
    for (const now of [130, 132, 134, 140]) {
      expect(qty(events, now)).toBe(1)
    }
  })

  it('drops then re-adds when the gap exceeds the settle window', () => {
    const events = [
      ev({ className: 'soda', enteredAt: 100, leftAt: 120 }),
      ev({ className: 'soda', enteredAt: 140 })
    ]
    expect(qty(events, 115)).toBe(1)
    expect(qty(events, 135)).toBe(0) // removed and gone
    expect(qty(events, 160)).toBe(1) // set back down later
  })

  it('counts two concurrent tracks as quantity 2, then 1 after one leaves', () => {
    const events = [
      ev({ className: 'soda', enteredAt: 100 }),
      ev({ className: 'soda', enteredAt: 101, leftAt: 111 })
    ]
    expect(qty(events, 112)).toBe(2)
    expect(qty(events, 125)).toBe(1)
  })

  it('excludes leftovers from before the session bound', () => {
    const leftover = ev({ className: 'soda', enteredAt: 100 })
    expect(qty([leftover], 210, 200)).toBe(0)
    // A track that starts after binding does count, even beside the leftover.
    const fresh = ev({ className: 'soda', enteredAt: 205 })
    expect(qty([leftover, fresh], 215, 200)).toBe(1)
  })

  it('respects minCommitConf as an inclusive floor', () => {
    const low = ev({ className: 'soda', enteredAt: 100, maxConf: 0.59 })
    const atFloor = ev({ className: 'soda', enteredAt: 100, maxConf: 0.6 })
    expect(qty([low], 110)).toBe(0)
    expect(qty([atFloor], 110)).toBe(1)
  })

  it('reports the highest confidence among the tracks that count', () => {
    const events = [
      ev({ className: 'soda', enteredAt: 100, maxConf: 0.71 }),
      ev({ className: 'soda', enteredAt: 101, maxConf: 0.93 })
    ]
    const entry = deriveCartState(events, 115, BIND, cfg).get('soda')
    expect(entry).toEqual({ quantity: 2, maxConfidence: 0.93 })
  })

  it('combines events from two sidecar sessions (a capture restart)', () => {
    const events = [
      ev({ sessionId: 1, className: 'soda', enteredAt: 300, leftAt: 340 }),
      ev({ sessionId: 2, className: 'soda', enteredAt: 345 })
    ]
    // The old session's track was closed on teardown; the new one carries the item.
    expect(qty(events, 360)).toBe(1)
  })

  it('returns an empty map when nothing is on the counter', () => {
    expect(deriveCartState([], 100, BIND, cfg).size).toBe(0)
  })
})

const live = (events: TrackEvent[], now: number, bindAt = BIND): number =>
  deriveLiveCounts(events, now, bindAt, cfg).get('soda') ?? 0

describe('deriveLiveCounts', () => {
  it('counts an open track immediately, with no commit dwell', () => {
    const events = [ev({ className: 'soda', enteredAt: 100 })]
    // The committed view still waits until 103; the raw view is the camera's eyes, which see the
    // item the moment the track exists.
    expect(qty(events, 101)).toBe(0)
    expect(live(events, 100)).toBe(1)
    expect(live(events, 101)).toBe(1)
  })

  it('stops counting a track `commitDwellS` after it ends', () => {
    const events = [ev({ className: 'soda', enteredAt: 100, leftAt: 120 })]
    // Within the dwell the track still counts as seen (smoothing)…
    expect(live(events, 122.9)).toBe(1)
    // …and exactly dwell seconds after left_at the camera has lost it.
    expect(live(events, 123)).toBe(0)
    expect(live(events, 125)).toBe(0)
  })

  it('never drops to 0 across a track swap shorter than the dwell', () => {
    const events = [
      ev({ className: 'soda', enteredAt: 100, leftAt: 130 }),
      ev({ className: 'soda', enteredAt: 131 })
    ]
    // A 1 s gap between tracks must not flash the lost state — that is what the dwell buys.
    for (const now of [130.5, 131, 132, 133.4]) {
      expect(live(events, now)).toBe(1)
    }
  })

  it('reads 0 in a gap longer than the dwell, then recovers', () => {
    const events = [
      ev({ className: 'soda', enteredAt: 100, leftAt: 130 }),
      ev({ className: 'soda', enteredAt: 133.5 })
    ]
    expect(live(events, 133.2)).toBe(0)
    expect(live(events, 133.6)).toBe(1)
  })

  it('never counts a zero-length flicker', () => {
    const events = [ev({ className: 'soda', enteredAt: 100, leftAt: 100 })]
    expect(live(events, 100)).toBe(0)
  })

  it('applies the same bind and confidence filters as the committed view', () => {
    const leftover = ev({ className: 'soda', enteredAt: 100 })
    expect(live([leftover], 210, 200)).toBe(0)
    const low = ev({ className: 'soda', enteredAt: 205, maxConf: 0.59 })
    expect(live([leftover, low], 210, 200)).toBe(0)
    const fresh = ev({ className: 'soda', enteredAt: 205, maxConf: 0.6 })
    expect(live([leftover, fresh], 210, 200)).toBe(1)
  })

  it('peaks across the window, so a just-ended track still vouches for its class', () => {
    const events = [
      ev({ className: 'soda', enteredAt: 100 }),
      ev({ className: 'soda', enteredAt: 101, leftAt: 111 })
    ]
    // Both open at 110 → 2; one closes at 111, and the window still remembers 2 until 114.
    expect(live(events, 110)).toBe(2)
    expect(live(events, 112)).toBe(2)
    expect(live(events, 114.1)).toBe(1)
  })

  it('returns an empty map when the camera sees nothing', () => {
    expect(deriveLiveCounts([], 100, BIND, cfg).size).toBe(0)
  })
})
