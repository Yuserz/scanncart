// Watches for the failure the process lifecycle cannot see: a sidecar child that is still alive —
// still in `tasklist`, still holding its camera — while nothing answers on its port.
//
// `SidecarSupervisor` reports `exit` and `main/index.ts` treated the port it printed as the proxy
// for "healthy", so both were blind to it. The renderer cannot supply the signal either: its
// WebSocket is *established* over the same loopback and stays established, which is why this looked
// like a frozen app with every fresh request hanging behind it. (`sidecar/app/loops.py` fixes the
// cause we found — Windows' proactor accept path closing the listening socket and never re-arming —
// but nothing on this side can assume a future cause away, and the shape of the failure is what
// this notices.)
//
// So the main process asks: one `GET /api/health` every couple of seconds, and a state that changes
// only when the answer does. Electron-free and timer-injectable, the same shape as
// `singleInstance.ts`, so it is unit-testable without a child process.
//
// The policy numbers are deliberately forgiving. A false alarm here tells an operator to restart a
// working app, which is worse than noticing a dead one a few seconds later: three consecutive
// failures, each with its own deadline, is roughly five seconds against a socket that refuses
// instantly and a dozen against one that has stopped answering altogether.

export type SidecarHealth =
  // Not asked yet: no port, or the first probe has not answered. Deliberately not an error —
  // startup is the one moment a silent sidecar is expected, and a banner there would be noise on
  // every launch.
  | 'starting'
  | 'ok'
  // Consecutive probes failed. Covers both shapes the operator sees as one: a child that is alive
  // and not listening, and a child that is gone.
  | 'unresponsive'

/** How long a single health request may take before it counts as a failure. */
export const HEALTH_TIMEOUT_MS = 2000
/** The gap between probes, measured from the end of the previous one, so they cannot stack. */
export const HEALTH_INTERVAL_MS = 2000
/** Consecutive failures before the state becomes `unresponsive`. */
export const HEALTH_FAILURES_BEFORE_UNRESPONSIVE = 3

export interface HealthMonitorOptions {
  /** Whether the sidecar on `port` answered. Must not throw; the monitor treats a throw as a no. */
  probe: (port: number) => Promise<boolean>
  intervalMs?: number
  failuresBeforeUnresponsive?: number
  /** Called on every *transition*, never for a state that has not changed. */
  onChange?: (health: SidecarHealth) => void
}

export class SidecarHealthMonitor {
  private readonly opts: HealthMonitorOptions
  private readonly intervalMs: number
  private readonly failuresBefore: number
  private timer: ReturnType<typeof setTimeout> | null = null
  private port: number | null = null
  private failures = 0
  private state: SidecarHealth = 'starting'

  constructor(opts: HealthMonitorOptions) {
    this.opts = opts
    this.intervalMs = opts.intervalMs ?? HEALTH_INTERVAL_MS
    this.failuresBefore = opts.failuresBeforeUnresponsive ?? HEALTH_FAILURES_BEFORE_UNRESPONSIVE
  }

  /** The current state, for a renderer that joins later (it reads it over IPC). */
  get health(): SidecarHealth {
    return this.state
  }

  /**
   * Begin probing `port`. Called when the sidecar reports a port, so it also covers a *restart*:
   * the previous port is dropped and the state goes back to `starting` rather than staying on
   * whatever the process that just died was.
   */
  start(port: number): void {
    this.cancel()
    this.port = port
    this.failures = 0
    this.set('starting')
    this.schedule(0)
  }

  /** Stop probing. The state stays where it was — the caller owns what is on screen after this. */
  stop(): void {
    this.cancel()
    this.port = null
  }

  private cancel(): void {
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
  }

  private schedule(delayMs: number): void {
    if (this.port === null) return
    this.timer = setTimeout(() => {
      this.timer = null
      void this.tick()
    }, delayMs)
  }

  private async tick(): Promise<void> {
    const port = this.port
    if (port === null) return
    let ok: boolean
    try {
      ok = await this.opts.probe(port)
    } catch {
      // A probe that throws is a probe that did not answer. Never let the monitor itself die on
      // one — that is exactly the state this exists to report.
      ok = false
    }
    // A `start()` may have landed while the probe was in flight; its port, not this one, is what
    // the next probe must be about, and its failure count must not be charged one of ours.
    if (this.port !== port) return
    if (ok) {
      this.failures = 0
      this.set('ok')
    } else {
      this.failures += 1
      if (this.failures >= this.failuresBefore) this.set('unresponsive')
    }
    this.schedule(this.intervalMs)
  }

  private set(next: SidecarHealth): void {
    if (next === this.state) return
    this.state = next
    this.opts.onChange?.(next)
  }
}
