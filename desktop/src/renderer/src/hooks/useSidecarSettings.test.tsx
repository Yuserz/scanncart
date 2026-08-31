import { DEFAULT_SETTINGS } from '../lib/settingsDefaults'
import { describe, it, expect, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useSidecarSettings } from './useSidecarSettings'
import { baseSettings, makeDeps } from '../test/fakes'

describe('useSidecarSettings', () => {
  it('loads settings, system info, and presets on mount', async () => {
    const { deps } = makeDeps()
    const { result } = renderHook(() => useSidecarSettings(8765, deps))

    expect(result.current.loading).toBe(true)

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.settings?.active_model).toBe('yolo11n.pt')
    expect(result.current.systemInfo?.recommended_preset).toBe('mid_range')
    expect(result.current.presets).toHaveLength(1)
    expect(result.current.recommended).toBe('mid_range')
  })

  it('polls health() and exposes captureState', async () => {
    const { deps } = makeDeps({
      health: vi.fn(async () => ({ state: 'running', active_model: 'yolo11n.pt', device: 'cpu' }))
    })
    const { result } = renderHook(() => useSidecarSettings(8765, deps))
    await waitFor(() => expect(result.current.captureState).toBe('running'))
  })

  it('update() calls updateSettings and merges the response', async () => {
    const { deps, api } = makeDeps()
    const { result } = renderHook(() => useSidecarSettings(8765, deps))
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.update({ infer_frame_skip: 3 })
    })

    expect(api.updateSettings).toHaveBeenCalledWith({ infer_frame_skip: 3 })
    expect(result.current.settings?.infer_frame_skip).toBe(3)
    expect(result.current.saving).toBe(false)
  })

  it('applyPreset() calls applyPreset and merges the response', async () => {
    const { deps, api } = makeDeps()
    const { result } = renderHook(() => useSidecarSettings(8765, deps))
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.applyPreset('high_end')
    })

    expect(api.applyPreset).toHaveBeenCalledWith('high_end')
    expect(result.current.settings?.active_model).toBe('high_end.pt')
  })

  it('restoreDefaults() calls updateSettings with the hardcoded defaults', async () => {
    const { deps, api } = makeDeps()
    const { result } = renderHook(() => useSidecarSettings(8765, deps))
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.restoreDefaults()
    })

    expect(api.updateSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        active_model: DEFAULT_SETTINGS.active_model,
        capture_width: 1280
      })
    )
  })

  it('retries the initial load if the sidecar is not reachable yet', async () => {
    let calls = 0
    const { deps } = makeDeps({
      getSettings: vi.fn(async () => {
        calls += 1
        if (calls < 3) throw new Error('sidecar GET /settings failed: ECONNREFUSED')
        return baseSettings()
      })
    })
    const { result } = renderHook(() => useSidecarSettings(8765, deps))

    // Fails at least once (settings still null) before eventually succeeding.
    await waitFor(() => expect(result.current.settings?.active_model).toBe('yolo11n.pt'))
    expect(calls).toBeGreaterThanOrEqual(3)
  })

  it('refresh() re-fetches on demand', async () => {
    const { deps, api } = makeDeps()
    const { result } = renderHook(() => useSidecarSettings(8765, deps))
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.refresh()
    })

    expect(api.getSettings).toHaveBeenCalledTimes(2)
  })

  it('a rejected update() surfaces an error and rethrows', async () => {
    const { deps } = makeDeps({
      updateSettings: vi.fn(async () => {
        throw new Error('sidecar PATCH /api/settings failed: 409')
      })
    })
    const { result } = renderHook(() => useSidecarSettings(8765, deps))
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await expect(result.current.update({ active_model: 'yolo11s.pt' })).rejects.toThrow('409')
    })

    expect(result.current.error).toMatch(/409/)
    expect(result.current.saving).toBe(false)
  })

  it('polls for a camera plugged in after startup', async () => {
    // refreshCameras only ran on mount, so a camera plugged in later stayed
    // invisible until the user pressed Rescan. The sidecar makes each poll
    // cheap (device names only) and re-scans only when the set changed.
    const { api } = makeDeps()
    renderHook(() => useSidecarSettings(8765, { apiFactory: () => api, cameraPollMs: 40 }))

    await waitFor(() => expect(vi.mocked(api.getCameras).mock.calls.length).toBeGreaterThan(2))
  })
})
