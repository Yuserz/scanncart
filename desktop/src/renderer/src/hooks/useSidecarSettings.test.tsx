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

describe('hosting inside LiveView', () => {
  it('does not poll health when the host already owns capture state', async () => {
    // LiveView drives capture through useSidecarStream. A second poller here
    // would give the same view two answers to "are we running".
    let healthCalls = 0
    const { deps } = makeDeps({
      health: vi.fn(async () => {
        healthCalls++
        return { state: 'idle', active_model: 'm', device: 'cpu' }
      })
    })

    renderHook(() => useSidecarSettings(9000, { ...deps, pollHealth: false }))
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50))
    })

    expect(healthCalls).toBe(0)
  })

  it('keeps reading camera quality even with health polling off', async () => {
    // The two share one timer. The tuning card turns health off because
    // LiveView owns capture state, but its whole readout is quality.
    let qualityCalls = 0
    const { deps } = makeDeps({
      getCameraQuality: vi.fn(async () => {
        qualityCalls++
        return {
          available: true,
          brightness: 128,
          contrast: 40,
          sharpness: 90,
          capture_fps: 29,
          target_fps: 30,
          verdicts: {},
          detail: ''
        }
      })
    })

    const { result } = renderHook(() => useSidecarSettings(9000, { ...deps, pollHealth: false }))
    await waitFor(() => expect(result.current.cameraQuality).not.toBeNull())

    expect(qualityCalls).toBeGreaterThan(0)
  })

  it('does not enumerate cameras for a consumer that never reads them', async () => {
    // Enumerating opens every device (~30 s). The card has no camera list.
    let cameraCalls = 0
    const { deps } = makeDeps({
      getCameras: vi.fn(async () => {
        cameraCalls++
        return { cameras: [], probed: true, detail: '' }
      })
    })

    const { result } = renderHook(() => useSidecarSettings(9000, { ...deps, pollCameras: false }))
    await waitFor(() => expect(result.current.settings).not.toBeNull())

    expect(cameraCalls).toBe(0)
    // Nothing else clears the initial true, and a stuck spinner reads as a
    // scan which never finishes.
    expect(result.current.camerasLoading).toBe(false)
  })

  it('still polls health by default, for the Admin panel', async () => {
    let healthCalls = 0
    const { deps } = makeDeps({
      health: vi.fn(async () => {
        healthCalls++
        return { state: 'idle', active_model: 'm', device: 'cpu' }
      })
    })

    renderHook(() => useSidecarSettings(9000, deps))
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50))
    })

    expect(healthCalls).toBeGreaterThan(0)
  })
})

describe('live apply and the saved baseline', () => {
  it('applies without persisting', async () => {
    const calls: [unknown, boolean | undefined][] = []
    const { deps } = makeDeps({
      updateSettings: vi.fn(async (patch, persist) => {
        calls.push([patch, persist])
        return baseSettings(patch)
      })
    })

    const { result } = renderHook(() => useSidecarSettings(9000, deps))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => {
      await result.current.liveUpdate({ conf_threshold: 0.9 })
    })

    expect(calls).toEqual([[{ conf_threshold: 0.9 }, false]])
  })

  it('leaves the saved baseline untouched while tuning', async () => {
    // The regression this exists to catch: a live PATCH returns a fresh
    // settings object, and treating that as "saved" makes every slider tick
    // look committed, so Revert has nothing to go back to.
    const { deps } = makeDeps({
      updateSettings: vi.fn(async (patch) => baseSettings(patch))
    })

    const { result } = renderHook(() => useSidecarSettings(9000, deps))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => {
      await result.current.liveUpdate({ conf_threshold: 0.9 })
    })

    expect(result.current.settings?.conf_threshold).toBe(0.9)
    expect(result.current.savedSettings?.conf_threshold).toBe(baseSettings().conf_threshold)
  })

  it('moves the baseline on save', async () => {
    const { deps } = makeDeps({
      updateSettings: vi.fn(async (patch) => baseSettings(patch)),
      saveSettings: vi.fn(async () => baseSettings({ conf_threshold: 0.9 }))
    })

    const { result } = renderHook(() => useSidecarSettings(9000, deps))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => {
      await result.current.liveUpdate({ conf_threshold: 0.9 })
    })
    await act(async () => {
      await result.current.save()
    })

    expect(result.current.savedSettings?.conf_threshold).toBe(0.9)
  })

  it('moves the baseline on an ordinary persisting update too', async () => {
    const { deps } = makeDeps({
      updateSettings: vi.fn(async (patch) => baseSettings(patch))
    })

    const { result } = renderHook(() => useSidecarSettings(9000, deps))
    await waitFor(() => expect(result.current.settings).not.toBeNull())
    await act(async () => {
      await result.current.update({ imgsz: 960 })
    })

    expect(result.current.savedSettings?.imgsz).toBe(960)
  })
})

describe('stored profile', () => {
  it('loads the saved calibration on mount', async () => {
    const profile = {
      device_key: 'StreamCam:0:1280x720',
      backend: 'MSMF',
      width: 1280,
      height: 720,
      fps_auto_exposure: 29.9,
      fps_capped_exposure: 30.8,
      controls: { brightness: true, exposure: true, gain: false, focus: false },
      recommended: {},
      measured_at: 1
    }
    const { deps } = makeDeps({ getCameraProfile: vi.fn(async () => ({ profile })) })

    const { result } = renderHook(() => useSidecarSettings(9000, deps))
    await waitFor(() => expect(result.current.storedProfile).not.toBeNull())

    expect(result.current.storedProfile?.controls.focus).toBe(false)
  })

  it('leaves it null for an uncalibrated camera', async () => {
    const { deps } = makeDeps({ getCameraProfile: vi.fn(async () => ({ profile: null })) })

    const { result } = renderHook(() => useSidecarSettings(9000, deps))
    await waitFor(() => expect(result.current.settings).not.toBeNull())

    expect(result.current.storedProfile).toBeNull()
  })
})
