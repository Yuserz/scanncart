// @vitest-environment node
import { describe, it, expect, vi } from 'vitest'
import { SidecarSupervisor } from './sidecar'

class FakeChild {
  stdoutCbs: Array<(c: Buffer) => void> = []
  stderrCbs: Array<(c: Buffer) => void> = []
  exitCbs: Array<(code: number | null) => void> = []
  killed = false
  stdout = {
    on: (e: string, cb: (c: Buffer) => void): void => {
      if (e === 'data') this.stdoutCbs.push(cb)
    }
  }
  stderr = {
    on: (e: string, cb: (c: Buffer) => void): void => {
      if (e === 'data') this.stderrCbs.push(cb)
    }
  }
  on(e: string, cb: (code: number | null) => void): void {
    if (e === 'exit') this.exitCbs.push(cb)
  }
  kill(): void {
    this.killed = true
  }
  emitStdout(s: string): void {
    this.stdoutCbs.forEach((cb) => cb(Buffer.from(s)))
  }
  emitStderr(s: string): void {
    this.stderrCbs.forEach((cb) => cb(Buffer.from(s)))
  }
  emitExit(code: number | null): void {
    this.exitCbs.forEach((cb) => cb(code))
  }
}

const makeSupervisor = (
  overrides = {}
): {
  sup: SidecarSupervisor
  child: FakeChild
  onPort: ReturnType<typeof vi.fn>
  onExit: ReturnType<typeof vi.fn>
} => {
  const child = new FakeChild()
  const onPort = vi.fn()
  const onExit = vi.fn()
  const sup = new SidecarSupervisor({
    spawnFn: () => child as never,
    pythonPath: 'python3',
    scriptPath: '/x/run.py',
    onPort,
    onExit,
    ...overrides
  })
  return { sup, child, onPort, onExit }
}

describe('SidecarSupervisor', () => {
  it('parses SIDECAR_PORT from a stdout line', () => {
    const { sup, child, onPort } = makeSupervisor()
    sup.start()
    child.emitStdout('SIDECAR_PORT=8765\n')
    expect(onPort).toHaveBeenCalledWith(8765)
  })

  it('reassembles a port line split across stdout chunks', () => {
    const { sup, child, onPort } = makeSupervisor()
    sup.start()
    child.emitStdout('some log\nSIDECAR_PO')
    child.emitStdout('RT=9001\n')
    expect(onPort).toHaveBeenCalledWith(9001)
  })

  it('ignores unrelated stdout without emitting a port', () => {
    const { sup, child, onPort } = makeSupervisor()
    sup.start()
    child.emitStdout('INFO: Uvicorn running\n')
    expect(onPort).not.toHaveBeenCalled()
  })

  it('forwards the sidecar’s own stdout lines to a sink, port line excluded', () => {
    // Without a sink these lines were read and dropped: `EVENT_LOOP=` reached a log only when a
    // human ran the sidecar by hand, never when the app launched it.
    const onStdoutLine = vi.fn()
    const { sup, child, onPort } = makeSupervisor({ onStdoutLine })
    sup.start()
    child.emitStdout('SIDECAR_PORT=8765\n')
    // CRLF, as Windows writes it: the \r must not ride into the log.
    child.emitStdout('EVENT_LOOP=_WindowsSelectorEventLoop\r\n')
    expect(onPort).toHaveBeenCalledWith(8765)
    expect(onStdoutLine).toHaveBeenCalledTimes(1)
    expect(onStdoutLine).toHaveBeenCalledWith('EVENT_LOOP=_WindowsSelectorEventLoop')
  })

  it('forwards only whole stdout lines, after reassembling a split one', () => {
    const onStdoutLine = vi.fn()
    const { sup, child } = makeSupervisor({ onStdoutLine })
    sup.start()
    child.emitStdout('EVENT_LO')
    expect(onStdoutLine).not.toHaveBeenCalled()
    child.emitStdout('OP=_WindowsSelectorEventLoop\n')
    expect(onStdoutLine).toHaveBeenCalledWith('EVENT_LOOP=_WindowsSelectorEventLoop')
  })

  it('forwards the watchdog line even when the port never arrived', () => {
    // The launch that dies before the handshake is exactly when stdout matters most.
    const onStdoutLine = vi.fn()
    const { sup, child, onPort } = makeSupervisor({ onStdoutLine })
    sup.start()
    child.emitStdout('[sidecar] parent process is gone — exiting\n')
    expect(onPort).not.toHaveBeenCalled()
    expect(onStdoutLine).toHaveBeenCalledWith('[sidecar] parent process is gone — exiting')
  })

  it('drains stderr and forwards it to onStderr (prevents pipe-buffer deadlock)', () => {
    const onStderr = vi.fn()
    const { sup, child } = makeSupervisor({ onStderr })
    sup.start()
    // Attaching a 'data' listener is what keeps the pipe flowing; assert it happened.
    expect(child.stderrCbs.length).toBe(1)
    child.emitStderr('INFO: Uvicorn running\n')
    expect(onStderr).toHaveBeenCalledWith('INFO: Uvicorn running\n')
  })

  it('stop() kills the child', () => {
    const { sup, child } = makeSupervisor()
    sup.start()
    sup.stop()
    expect(child.killed).toBe(true)
  })

  it('calls onExit for an unexpected exit but not after stop()', () => {
    const a = makeSupervisor()
    a.sup.start()
    a.child.emitExit(1)
    expect(a.onExit).toHaveBeenCalledWith(1)

    const b = makeSupervisor()
    b.sup.start()
    b.sup.stop()
    b.child.emitExit(0)
    expect(b.onExit).not.toHaveBeenCalled()
  })

  it('tells the sidecar which pid to watch, so it cannot outlive us', () => {
    // before-quit does not run on a crash or force-kill, and an orphaned
    // sidecar keeps its camera, port and threads. One was measured holding all
    // 12 cores, dragging live inference from 56 ms to 500 ms.
    let seen: { cwd?: string; env?: NodeJS.ProcessEnv } | undefined
    const sup = new SidecarSupervisor({
      spawnFn: (_c, _a, options) => {
        seen = options
        return new FakeChild() as never
      },
      pythonPath: 'py',
      scriptPath: 'run.py',
      cwd: '.',
      onPort: () => {},
      onExit: () => {}
    })
    sup.start()

    expect(seen?.env?.SIDECAR_PARENT_PID).toBe(String(process.pid))
    // ...without dropping the rest of the environment.
    expect(Object.keys(seen?.env ?? {}).length).toBeGreaterThan(1)
  })
})
