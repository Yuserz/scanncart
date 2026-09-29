import { describe, it, expect, vi, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useSidecarHealth } from './useSidecarHealth'
import type { SidecarHealth } from '../../../main/sidecarHealth'

// The bridge itself, not injected deps: this file is what pins the *default* path, so a rename on
// the preload side shows up here rather than only in the app.
function stubBridge(opts: {
  read?: () => Promise<SidecarHealth>
  subscribe?: (cb: (health: SidecarHealth) => void) => () => void
}): { push: (health: SidecarHealth) => void; unsubscribed: () => boolean } {
  let listener: ((health: SidecarHealth) => void) | null = null
  vi.stubGlobal('api', {
    getSidecarPort: async () => 8765,
    getSidecarHealth: opts.read ?? (async () => 'ok' as SidecarHealth),
    onSidecarHealth: (cb: (health: SidecarHealth) => void) => {
      listener = cb
      return opts.subscribe
        ? opts.subscribe(cb)
        : () => {
            listener = null
          }
    }
  })
  return { push: (health) => listener?.(health), unsubscribed: () => listener === null }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useSidecarHealth', () => {
  it('starts at "starting" rather than at a verdict', async () => {
    stubBridge({})
    const { result } = renderHook(() => useSidecarHealth())

    // Nothing has been asked yet. Rendering a warning here would put one on every healthy launch.
    expect(result.current).toBe('starting')
    // ...and then the main process's answer lands, which also keeps this test from leaving an
    // update to settle into whichever test runs next.
    await waitFor(() => expect(result.current).toBe('ok'))
  })

  it('takes the state the main process already has', async () => {
    stubBridge({ read: async () => 'unresponsive' })
    const { result } = renderHook(() => useSidecarHealth())

    await waitFor(() => expect(result.current).toBe('unresponsive'))
  })

  it('follows a transition that arrives while the window is open', async () => {
    const bridge = stubBridge({})
    const { result } = renderHook(() => useSidecarHealth())
    await waitFor(() => expect(result.current).toBe('ok'))

    act(() => bridge.push('unresponsive'))
    expect(result.current).toBe('unresponsive')
  })

  it('leaves the state alone when the read fails, instead of inventing an alarm', async () => {
    stubBridge({ read: () => Promise.reject(new Error('ipc is gone')) })
    const { result } = renderHook(() => useSidecarHealth())

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    // An IPC failure is not evidence about the sidecar; the main process is the one that says so.
    expect(result.current).toBe('starting')
  })

  it('unsubscribes when the window goes away', async () => {
    const bridge = stubBridge({})
    const { result, unmount } = renderHook(() => useSidecarHealth())
    await waitFor(() => expect(result.current).toBe('ok'))

    unmount()

    expect(bridge.unsubscribed()).toBe(true)
  })
})
