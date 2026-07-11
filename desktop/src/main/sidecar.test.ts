// @vitest-environment node
import { describe, it, expect, vi } from 'vitest'
import { SidecarSupervisor } from './sidecar'

class FakeChild {
  stdoutCbs: Array<(c: Buffer) => void> = []
  exitCbs: Array<(code: number | null) => void> = []
  killed = false
  stdout = {
    on: (e: string, cb: (c: Buffer) => void): void => {
      if (e === 'data') this.stdoutCbs.push(cb)
    }
  }
  stderr = { on: (): void => {} }
  on(e: string, cb: (code: number | null) => void): void {
    if (e === 'exit') this.exitCbs.push(cb)
  }
  kill(): void {
    this.killed = true
  }
  emitStdout(s: string): void {
    this.stdoutCbs.forEach((cb) => cb(Buffer.from(s)))
  }
  emitExit(code: number | null): void {
    this.exitCbs.forEach((cb) => cb(code))
  }
}

function makeSupervisor(overrides = {}) {
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
})
