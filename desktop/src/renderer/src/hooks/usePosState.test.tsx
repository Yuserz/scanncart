import { describe, it, expect, vi, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { usePosState } from './usePosState'
import type { PosState } from '../../../main/posSession'

const STATE: PosState = {
  phase: 'bound',
  cartCode: 'CODE-1',
  syncedItemCount: 2,
  lastSyncAgeS: 1,
  error: null,
  retryAtMs: null,
  lastContactMs: null,
  cartMode: 'counter',
  basket: null
}

function stubBridge(opts: {
  read?: () => Promise<PosState | null>
  subscribe?: (cb: (state: PosState) => void) => () => void
}): { push: (state: PosState) => void; unsubscribed: () => boolean } {
  let listener: ((state: PosState) => void) | null = null
  vi.stubGlobal('api', {
    getPosState: opts.read ?? (async () => null),
    onPosState: (cb: (state: PosState) => void) => {
      listener = cb
      return opts.subscribe
        ? opts.subscribe(cb)
        : () => {
            listener = null
          }
    }
  })
  return { push: (state) => listener?.(state), unsubscribed: () => listener === null }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('usePosState', () => {
  it('starts at null (nothing reported yet) rather than at a verdict', async () => {
    stubBridge({})
    const { result } = renderHook(() => usePosState())

    expect(result.current).toBeNull()
    await waitFor(() => expect(result.current).toBeNull())
  })

  it('takes the state the main process already has', async () => {
    stubBridge({ read: async () => STATE })
    const { result } = renderHook(() => usePosState())

    await waitFor(() => expect(result.current).toEqual(STATE))
  })

  it('follows a transition while the window is open', async () => {
    const bridge = stubBridge({ read: async () => STATE })
    const { result } = renderHook(() => usePosState())
    await waitFor(() => expect(result.current).toEqual(STATE))

    act(() => bridge.push({ ...STATE, phase: 'unbound', cartCode: null }))
    expect(result.current?.phase).toBe('unbound')
  })

  it('leaves the state alone when the read fails', async () => {
    stubBridge({ read: () => Promise.reject(new Error('ipc gone')) })
    const { result } = renderHook(() => usePosState())

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(result.current).toBeNull()
  })

  it('unsubscribes when the window goes away', async () => {
    const bridge = stubBridge({ read: async () => STATE })
    const { result, unmount } = renderHook(() => usePosState())
    await waitFor(() => expect(result.current).toEqual(STATE))

    unmount()
    expect(bridge.unsubscribed()).toBe(true)
  })
})
