// @vitest-environment node
import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  HEALTH_FAILURES_BEFORE_UNRESPONSIVE,
  SidecarHealthMonitor,
  type SidecarHealth
} from './sidecarHealth'

// Collects the transitions a monitor reports, so a test can assert on the sequence rather than
// only the final state — "says nothing while the answers keep coming" is a claim about silence.
function recorder(): { seen: SidecarHealth[]; onChange: (h: SidecarHealth) => void } {
  const seen: SidecarHealth[] = []
  return { seen, onChange: (h) => seen.push(h) }
}

/** A probe whose answer the test decides, recording the ports it was asked about. */
function recording(answer: (port: number, call: number) => boolean): {
  probe: (port: number) => Promise<boolean>
  ports: number[]
} {
  const ports: number[] = []
  return {
    ports,
    probe: (port) => {
      const call = ports.length
      ports.push(port)
      return Promise.resolve(answer(port, call))
    }
  }
}

// Probes are scheduled one interval *after* the previous one finishes, so probe k lands at
// (k - 1) * TICK. The `-1` below is what keeps a boundary case on the side of "has not happened
// yet", which is the difference the assertions are about.
const TICK = 2000

afterEach(() => {
  vi.useRealTimers()
})

describe('SidecarHealthMonitor', () => {
  it('is silent about startup and only reports the first answer', async () => {
    vi.useFakeTimers()
    const p = recording(() => true)
    const { seen, onChange } = recorder()
    const monitor = new SidecarHealthMonitor({ probe: p.probe, intervalMs: TICK, onChange })

    monitor.start(8765)
    expect(monitor.health).toBe('starting')

    // Four successful probes: the operator is told the sidecar is up once, not every two seconds.
    await vi.advanceTimersByTimeAsync(TICK * 4 - 1)

    expect(p.ports).toHaveLength(4)
    expect(monitor.health).toBe('ok')
    expect(seen).toEqual(['ok'])
  })

  it('goes unresponsive only after several consecutive failures', async () => {
    vi.useFakeTimers()
    const p = recording(() => false)
    const { seen, onChange } = recorder()
    const monitor = new SidecarHealthMonitor({ probe: p.probe, intervalMs: TICK, onChange })

    monitor.start(8765)
    await vi.advanceTimersByTimeAsync(TICK * (HEALTH_FAILURES_BEFORE_UNRESPONSIVE - 1) - 1)
    expect(monitor.health).toBe('starting')

    await vi.advanceTimersByTimeAsync(TICK)
    expect(monitor.health).toBe('unresponsive')
    expect(seen).toEqual(['unresponsive'])
  })

  it('clears the count when an answer arrives, so a hiccup is not an alarm', async () => {
    vi.useFakeTimers()
    // Two failures, a success, then one more failure — never three in a row.
    const answers = [false, false, true, false]
    const p = recording(() => answers.shift() ?? true)
    const { seen, onChange } = recorder()
    const monitor = new SidecarHealthMonitor({ probe: p.probe, intervalMs: TICK, onChange })

    monitor.start(8765)
    await vi.advanceTimersByTimeAsync(TICK * 6)

    expect(monitor.health).toBe('ok')
    // Never unresponsive: the sequence above is exactly the shape a single hiccup takes.
    expect(seen).toEqual(['ok'])
  })

  it('counts from the end of the previous probe, so slow probes cannot stack', async () => {
    vi.useFakeTimers()
    let inFlight = 0
    let peak = 0
    const monitor = new SidecarHealthMonitor({
      probe: async () => {
        inFlight += 1
        peak = Math.max(peak, inFlight)
        await new Promise((r) => setTimeout(r, TICK * 3))
        inFlight -= 1
        return true
      },
      intervalMs: TICK
    })

    monitor.start(8765)
    await vi.advanceTimersByTimeAsync(TICK * 12)

    expect(peak).toBe(1)
    expect(monitor.health).toBe('ok')
  })

  it('treats a probe that throws as a failure without dying on it', async () => {
    vi.useFakeTimers()
    const monitor = new SidecarHealthMonitor({
      probe: () => Promise.reject(new Error('socket exploded')),
      intervalMs: TICK
    })

    monitor.start(8765)
    await vi.advanceTimersByTimeAsync(TICK * HEALTH_FAILURES_BEFORE_UNRESPONSIVE)

    expect(monitor.health).toBe('unresponsive')
  })

  it('moves to a restarted sidecar\u2019s port and starts over', async () => {
    vi.useFakeTimers()
    const p = recording((port) => port === 9000)
    const { seen, onChange } = recorder()
    const monitor = new SidecarHealthMonitor({ probe: p.probe, intervalMs: TICK, onChange })

    monitor.start(8765)
    await vi.advanceTimersByTimeAsync(TICK * (HEALTH_FAILURES_BEFORE_UNRESPONSIVE - 1) + 1)
    expect(monitor.health).toBe('unresponsive')

    monitor.start(9000)
    // The stale verdict is dropped immediately: the port it described is not the one being asked now.
    expect(monitor.health).toBe('starting')

    await vi.advanceTimersByTimeAsync(1)
    expect(monitor.health).toBe('ok')
    expect(p.ports).toEqual([8765, 8765, 8765, 9000])
    expect(seen).toEqual(['unresponsive', 'starting', 'ok'])
  })

  it('ignores an answer that arrives for the port it has moved on from', async () => {
    vi.useFakeTimers()
    const pending: ((ok: boolean) => void)[] = []
    const monitor = new SidecarHealthMonitor({
      probe: () => new Promise<boolean>((resolve) => pending.push(resolve)),
      intervalMs: TICK
    })

    monitor.start(8765)
    await vi.advanceTimersByTimeAsync(0)
    expect(pending).toHaveLength(1)

    // The child is replaced while its health request is still open.
    monitor.start(9000)
    await vi.advanceTimersByTimeAsync(0)
    expect(pending).toHaveLength(2)

    // The answer for the old port arrives late and says everything is fine.
    pending[0](true)
    await vi.advanceTimersByTimeAsync(0)
    expect(monitor.health).toBe('starting')
  })

  it('stops asking once stopped', async () => {
    vi.useFakeTimers()
    const p = recording(() => true)
    const monitor = new SidecarHealthMonitor({ probe: p.probe, intervalMs: TICK })

    monitor.start(8765)
    await vi.advanceTimersByTimeAsync(0)
    expect(p.ports).toHaveLength(1)

    monitor.stop()
    await vi.advanceTimersByTimeAsync(TICK * 10)
    expect(p.ports).toHaveLength(1)
  })

  it('is forgiving by default, because a false alarm costs more than a late one', () => {
    // Telling an operator to restart a working app is the failure mode to avoid, so the balance is
    // written down and a tightening has to be deliberate.
    expect(HEALTH_FAILURES_BEFORE_UNRESPONSIVE).toBeGreaterThanOrEqual(2)
  })
})
