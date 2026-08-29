// @vitest-environment node
import { describe, it, expect, vi } from 'vitest'
import { handleSecondInstance, type FocusableWindow } from './singleInstance'

function fakeWindow(state: { minimized?: boolean; visible?: boolean } = {}): FocusableWindow & {
  calls: string[]
} {
  const calls: string[] = []
  let minimized = state.minimized ?? false
  let visible = state.visible ?? true
  return {
    calls,
    isMinimized: () => minimized,
    restore: () => {
      minimized = false
      calls.push('restore')
    },
    isVisible: () => visible,
    show: () => {
      visible = true
      calls.push('show')
    },
    focus: () => calls.push('focus')
  }
}

function deps(
  over: Partial<Parameters<typeof handleSecondInstance>[0]> = {}
): Parameters<typeof handleSecondInstance>[0] & { createWindow: ReturnType<typeof vi.fn> } {
  return {
    getWindow: () => null,
    createWindow: vi.fn(),
    getSidecarPort: () => 8765,
    restartSidecar: vi.fn(),
    ...over
  } as never
}

describe('handleSecondInstance', () => {
  it('focuses the existing window instead of opening another', () => {
    const win = fakeWindow()
    const d = deps({ getWindow: () => win })

    handleSecondInstance(d)

    expect(win.calls).toEqual(['focus'])
    expect(d.createWindow).not.toHaveBeenCalled()
  })

  it('restores a minimized window before focusing it', () => {
    // focus() will not raise a minimized window on its own.
    const win = fakeWindow({ minimized: true })

    handleSecondInstance(deps({ getWindow: () => win }))

    expect(win.calls).toEqual(['restore', 'focus'])
  })

  it('shows a hidden window before focusing it', () => {
    const win = fakeWindow({ visible: false })

    handleSecondInstance(deps({ getWindow: () => win }))

    expect(win.calls).toEqual(['show', 'focus'])
  })

  it('creates a window when the old one was closed', () => {
    const d = deps({ getWindow: () => null })

    handleSecondInstance(d)

    expect(d.createWindow).toHaveBeenCalledOnce()
  })

  it('leaves a healthy sidecar alone', () => {
    const d = deps({ getWindow: () => fakeWindow(), getSidecarPort: () => 8765 })

    handleSecondInstance(d)

    expect(d.restartSidecar).not.toHaveBeenCalled()
  })

  it('restarts a sidecar that never reported a port', () => {
    // There is no auto-restart, so a dead sidecar leaves the renderer polling
    // for a port forever. Relaunching is what a user tries; make it repair the
    // instance they already have.
    const d = deps({ getWindow: () => fakeWindow(), getSidecarPort: () => null })

    handleSecondInstance(d)

    expect(d.restartSidecar).toHaveBeenCalledOnce()
  })

  it('revives the sidecar even when the window also had to be recreated', () => {
    const d = deps({ getWindow: () => null, getSidecarPort: () => null })

    handleSecondInstance(d)

    expect(d.createWindow).toHaveBeenCalledOnce()
    expect(d.restartSidecar).toHaveBeenCalledOnce()
  })
})
