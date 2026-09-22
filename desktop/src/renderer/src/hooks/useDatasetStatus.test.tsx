import { describe, it, expect, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useDatasetStatus } from './useDatasetStatus'
import type { ApiClient } from '../lib/api'
import { makeDeps, datasetStatus } from '../test/fakes'

// Mirrors useSidecarSettings.test.tsx: the api client is injected, never mocked at
// module level, so these stay honest about the calls the hook actually makes.
function depsWith(overrides: Partial<ApiClient>): {
  apiFactory: (port: number) => ApiClient
  retryDelayMs: number
} {
  const { deps, api } = makeDeps(overrides)
  void api
  return {
    apiFactory: deps.apiFactory as (port: number) => ApiClient,
    retryDelayMs: 10
  }
}

describe('useDatasetStatus', () => {
  it('reads the snapshot once and exposes it', async () => {
    const status = datasetStatus()
    const apiFactory = vi.fn(
      () => ({ getDatasetStatus: vi.fn(async () => status) }) as unknown as ApiClient
    )

    const { result } = renderHook(() => useDatasetStatus(8765, { apiFactory }))

    await waitFor(() => expect(result.current.status).not.toBeNull())
    expect(result.current.status?.total).toBe(1383)
    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
    expect(apiFactory).toHaveBeenCalledWith(8765)
  })

  it('treats an unavailable snapshot as a state, not an error', async () => {
    // This is the normal first-run answer: nobody has run the tool yet. It must not
    // surface as a failure, or the panel would shout at an operator who has done
    // nothing wrong.
    const apiFactory = vi.fn(
      () =>
        ({
          getDatasetStatus: vi.fn(async () =>
            datasetStatus({ available: false, total: 0, decided: 0 })
          )
        }) as unknown as ApiClient
    )

    const { result } = renderHook(() => useDatasetStatus(8765, { apiFactory }))

    await waitFor(() => expect(result.current.status).not.toBeNull())
    expect(result.current.status?.available).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('does not poll — only the explicit refresh refetches', async () => {
    // The sidecar serves a file that changes only when the tool is run by hand, so a
    // timer would refetch identical bytes forever. Guarded here because "add an
    // interval later" is exactly the optimisation someone would reach for.
    const getDatasetStatus = vi.fn(async () => datasetStatus())
    const apiFactory = vi.fn(() => ({ getDatasetStatus }) as unknown as ApiClient)

    const { result } = renderHook(() => useDatasetStatus(8765, { apiFactory }))

    await waitFor(() => expect(result.current.status).not.toBeNull())
    expect(getDatasetStatus).toHaveBeenCalledTimes(1)

    await new Promise((r) => setTimeout(r, 60))
    expect(getDatasetStatus).toHaveBeenCalledTimes(1)
  })

  it('refresh refetches', async () => {
    const getDatasetStatus = vi.fn(async () => datasetStatus())
    const apiFactory = vi.fn(() => ({ getDatasetStatus }) as unknown as ApiClient)

    const { result } = renderHook(() => useDatasetStatus(8765, { apiFactory }))
    await waitFor(() => expect(result.current.status).not.toBeNull())

    // Wrapped in act because refresh() sets state: awaiting it bare leaves the update
    // unwrapped, which React reports as a warning rather than a failure.
    await act(async () => {
      await result.current.refresh()
    })
    expect(getDatasetStatus).toHaveBeenCalledTimes(2)
    expect(result.current.loading).toBe(false)
  })

  it('retries while the sidecar is still starting, then gives up gracefully', async () => {
    const getDatasetStatus = vi.fn(async () => {
      throw new Error('connection refused')
    })

    const { result } = renderHook(() => useDatasetStatus(8765, depsWith({ getDatasetStatus })))

    await waitFor(() => expect(result.current.error).toBe('connection refused'), { timeout: 2000 })
    expect(result.current.status).toBeNull()
    expect(result.current.loading).toBe(false)
    expect(getDatasetStatus.mock.calls.length).toBeGreaterThan(1)
  })
})
