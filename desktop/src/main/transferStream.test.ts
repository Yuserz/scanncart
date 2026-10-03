// @vitest-environment node
// The basket tracker end to end on synthetic frame messages: the sidecar's wire shape in,
// the ledger's quantities and review list out.

import { describe, expect, it, vi } from 'vitest'

import { DEFAULT_ZONE_PRESET } from './transferGeometry'
import { BasketTracker, type StreamFrame } from './transferStream'

// Normalized centres for the default bands (cart = bottom 35%, opening the 20% above).
const OUT = 0.2
const OPEN = 0.55
const IN = 0.82

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- a test harness whose shape is its own return
function harness() {
  let now = 1000
  const tracker = new BasketTracker(DEFAULT_ZONE_PRESET, { now: () => now })
  let seq = 0
  const frame = (
    cy: number | null,
    o: Partial<StreamFrame> & { cls?: string; track?: number | null; conf?: number } = {}
  ): StreamFrame => {
    seq += 1
    now += 0.2
    const { cls = 'soda', track = 7, conf = 0.9, ...rest } = o
    return {
      type: 'frame',
      ts: now,
      seq,
      fresh: true,
      mirrored: false,
      detections:
        cy === null
          ? []
          : [{ track_id: track, cls, conf, box: [0.45, cy - 0.03, 0.55, cy + 0.03] }],
      ...rest
    }
  }
  const run = (cy: number, n: number, o: Parameters<typeof frame>[1] = {}): void => {
    for (let i = 0; i < n; i++) tracker.onFrame(frame(cy, o))
  }
  const deposit = (o: Parameters<typeof frame>[1] = {}): void => {
    run(OUT, 2, o)
    run(OPEN, 1, o)
    run(IN, 6, o)
  }
  const removal = (o: Parameters<typeof frame>[1] = {}): void => {
    run(IN, 2, o)
    run(OPEN, 1, o)
    run(OUT, 6, o)
  }
  const qty = (): Record<string, number> =>
    Object.fromEntries([...tracker.snapshot()].map(([k, v]) => [k, v.quantity]))
  return {
    tracker,
    frame,
    run,
    deposit,
    removal,
    qty,
    advance: (s: number) => {
      now += s
    }
  }
}

describe('basket tracker', () => {
  it('a deposit then a removal leaves the basket empty', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.run(OUT, 1, { track: null }) // past the baseline window with nothing inside
    h.advance(4)
    h.deposit()
    expect(h.qty()).toEqual({ soda: 1 })
    h.removal()
    expect(h.qty()).toEqual({})
    expect(h.tracker.pendingReview()).toEqual([])
  })

  it('#8 a deposited item that is hidden stays in the basket', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.advance(4)
    h.deposit()
    for (let i = 0; i < 50; i++) h.tracker.onFrame(h.frame(null)) // 10 s of nothing visible
    expect(h.qty()).toEqual({ soda: 1 })
  })

  it('ignores preview fill-in frames', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.advance(4)
    h.run(OUT, 2, { fresh: false })
    h.run(OPEN, 1, { fresh: false })
    h.run(IN, 6, { fresh: false })
    expect(h.qty()).toEqual({})
  })

  it('undoes the preview mirror before judging the cart edge', async () => {
    const h = harness()
    await h.tracker.setZones({ ...DEFAULT_ZONE_PRESET, cartEdge: 'left' })
    await h.tracker.bind('s1')
    h.advance(4)
    // True geometry: the item moves from the right (outside) to the left (inside). Mirrored on
    // the wire, the boxes arrive on the opposite side — the tracker must still see a deposit.
    const at = (x: number, n: number): void => {
      for (let i = 0; i < n; i++) {
        const f = h.frame(0.5, { mirrored: true })
        f.detections[0].box = [1 - (x + 0.03), 0.47, 1 - (x - 0.03), 0.53]
        h.tracker.onFrame(f)
      }
    }
    at(0.8, 2)
    at(0.45, 1)
    at(0.15, 6)
    expect(h.qty()).toEqual({ soda: 1 })
  })

  it('#16 a product already in the cart band at Start is flagged', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.run(IN, 2)
    expect(h.tracker.pendingReview()[0].reason).toContain('not empty at Start')
  })

  it('#15 capture stopping keeps the basket and asks for review when it comes back', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.advance(4)
    h.deposit()
    h.tracker.onStatus('idle')
    h.advance(40)
    h.tracker.onStatus('running')
    h.run(OUT, 1, { track: null })
    expect(h.qty()).toEqual({ soda: 1 })
    expect(h.tracker.pendingReview().some((r) => r.reason.includes('could not see'))).toBe(true)
  })

  it('a stream that goes quiet is blind after the timeout', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.run(OUT, 1, { track: null })
    h.advance(5)
    h.tracker.tick()
    expect(h.tracker.readout().blind).toBe(true)
  })

  it('#11 removing what was never deposited is review, not a negative count', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.advance(4)
    h.removal({ cls: 'tuna' })
    expect(h.qty()).toEqual({})
    expect(h.tracker.pendingReview()[0].reason).toContain('not in the basket')
  })

  it('infers nothing under the scanner threshold it was given', async () => {
    const h = harness()
    h.tracker.setConfThreshold(0.9)
    await h.tracker.bind('s1')
    h.advance(4)
    h.deposit({ conf: 0.8 })
    expect(h.qty()).toEqual({})
    h.deposit({ conf: 0.95, track: 8 })
    expect(h.qty()).toEqual({ soda: 1 })
  })

  it('the empty-basket check uses the scanner threshold too', async () => {
    const h = harness()
    h.tracker.setConfThreshold(0.9)
    await h.tracker.bind('s1')
    h.run(IN, 2, { conf: 0.7 }) // below the scanner's cut: not evidence of anything
    expect(h.tracker.pendingReview()).toEqual([])
  })

  it('redials the port the sidecar has now, not the one it had at start (review finding)', async () => {
    vi.useFakeTimers()
    try {
      const sockets: Array<{ url: string; onclose: (() => void) | null }> = []
      let port: number | null = 8765
      const tracker = new BasketTracker(DEFAULT_ZONE_PRESET, {
        reconnectDelayMs: 100,
        wsFactory: (url) => {
          const ws = {
            url,
            onopen: null,
            onmessage: null,
            onclose: null,
            onerror: null,
            close: () => {}
          }
          sockets.push(ws)
          return ws
        }
      })
      tracker.connect(() => port)
      expect(sockets.map((s) => s.url)).toEqual(['ws://127.0.0.1:8765/ws/stream'])

      // The sidecar restarts on another port: nothing answers for a moment, then 8766 does.
      port = null
      sockets[0].onclose?.()
      await vi.advanceTimersByTimeAsync(100)
      expect(sockets).toHaveLength(1) // no port yet: it waits rather than dialling a dead one
      port = 8766
      await vi.advanceTimersByTimeAsync(100)
      expect(sockets.at(-1)?.url).toBe('ws://127.0.0.1:8766/ws/stream')

      tracker.close()
      sockets.at(-1)?.onclose?.()
      await vi.advanceTimersByTimeAsync(500)
      expect(sockets).toHaveLength(2) // closed means closed
    } finally {
      vi.useRealTimers()
    }
  })

  it('a new session starts from an empty basket', async () => {
    const h = harness()
    await h.tracker.bind('s1')
    h.advance(4)
    h.deposit()
    h.tracker.unbind()
    await h.tracker.bind('s2')
    expect(h.qty()).toEqual({})
  })
})
