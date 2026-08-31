import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createApiClient,
  type ApiClient,
  type CameraInfo,
  type CameraProfileResponse,
  type CameraQualityResponse,
  type DetectorProbeResponse,
  type PresetInfo,
  type SettingsResponse,
  type SettingsUpdate,
  type SystemInfoResponse
} from '../lib/api'
import { DEFAULT_SETTINGS } from '../lib/settingsDefaults'

export interface SettingsDeps {
  apiFactory?: (port: number) => ApiClient
  healthPollMs?: number
  cameraPollMs?: number
  retryDelayMs?: number
  // False when a host view already owns capture state (LiveView drives it
  // through useSidecarStream). Two pollers in one view means two answers to
  // "are we running", and they disagree during a start or stop. It suppresses
  // ONLY the health call — the camera-quality read shares that timer and the
  // tuning card depends on it.
  pollHealth?: boolean
  // False for a consumer that never reads `cameras`. Enumerating opens every
  // device (~30 s), so the tuning card must not trigger it on mount.
  pollCameras?: boolean
}

export interface SidecarSettings {
  settings: SettingsResponse | null
  systemInfo: SystemInfoResponse | null
  presets: PresetInfo[]
  recommended: string | null
  captureState: string
  loading: boolean
  saving: boolean
  error: string | null
  refresh: () => Promise<void>
  update: (patch: SettingsUpdate) => Promise<SettingsResponse>
  applyPreset: (name: string) => Promise<SettingsResponse>
  restoreDefaults: () => Promise<SettingsResponse>
  probe: () => Promise<DetectorProbeResponse>
  cameras: CameraInfo[]
  // Stopping from Admin: the restart-required fields are edited here, so
  // sending the user to Live view just to unblock the form is a dead end.
  stopCapture: () => Promise<void>
  startCapture: () => Promise<void>
  stopping: boolean
  refreshCameras: (rescan?: boolean) => Promise<void>
  camerasLoading: boolean
  probing: boolean
  probeResult: DetectorProbeResponse | null
  cameraQuality: CameraQualityResponse | null
  calibrate: () => Promise<CameraProfileResponse>
  calibrating: boolean
  profile: CameraProfileResponse | null
  applyProfile: () => Promise<SettingsResponse>
  // The saved calibration for the configured camera, or null if it has never
  // been calibrated. Says which controls the device actually honours.
  storedProfile: CameraProfileResponse | null
  // Applies to the running camera/detector without writing settings.json.
  liveUpdate: (patch: SettingsUpdate) => Promise<void>
  // Commits whatever is in memory, including live-applied values.
  save: () => Promise<SettingsResponse>
  // The last persisted settings. Distinct from `settings`, which tracks the
  // live values — the difference between the two is "unsaved changes".
  savedSettings: SettingsResponse | null
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

// Wires the settings/system-info/presets REST endpoints into React state.
// captureState is polled via the existing /api/health endpoint (reused, not
// a second WebSocket) so Save can be gated on capture being stopped, the
// same `running` check LiveView already makes off useSidecarStream.
export function useSidecarSettings(port: number, deps: SettingsDeps = {}): SidecarSettings {
  const apiFactory = deps.apiFactory ?? createApiClient
  const healthPollMs = deps.healthPollMs ?? 2000
  // Each poll costs the sidecar one ~550 ms device-name query, and only
  // triggers the expensive scan when the device set actually changed.
  const cameraPollMs = deps.cameraPollMs ?? 15000
  const retryDelayMs = deps.retryDelayMs ?? 1000
  const shouldPollHealth = deps.pollHealth ?? true
  const shouldPollCameras = deps.pollCameras ?? true

  const [settings, setSettings] = useState<SettingsResponse | null>(null)
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null)
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [recommended, setRecommended] = useState<string | null>(null)
  const [captureState, setCaptureState] = useState('idle')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [camerasLoading, setCamerasLoading] = useState(true)
  const [stopping, setStopping] = useState(false)
  const [probing, setProbing] = useState(false)
  const [probeResult, setProbeResult] = useState<DetectorProbeResponse | null>(null)
  const [cameraQuality, setCameraQuality] = useState<CameraQualityResponse | null>(null)
  const [calibrating, setCalibrating] = useState(false)
  const [profile, setProfile] = useState<CameraProfileResponse | null>(null)
  const [storedProfile, setStoredProfile] = useState<CameraProfileResponse | null>(null)
  const [savedSettings, setSavedSettings] = useState<SettingsResponse | null>(null)

  const apiRef = useRef<ApiClient | null>(null)

  // Returns whether the load succeeded, so the mount effect below can retry
  // (the sidecar's port is reported to the renderer before uvicorn has
  // necessarily finished starting up — see ws.ts's auto-reconnect for the
  // same race on the WebSocket side).
  const load = useCallback(async (): Promise<boolean> => {
    const api = apiRef.current
    if (!api) return false
    setLoading(true)
    setError(null)
    try {
      const [s, sys, p, prof] = await Promise.all([
        api.getSettings(),
        api.getSystemInfo(),
        api.getPresets(),
        // Cheap: reads one small JSON file, opens no device.
        api.getCameraProfile()
      ])
      setSettings(s)
      setSavedSettings(s)
      setSystemInfo(sys)
      setPresets(p.presets)
      setRecommended(p.recommended)
      setStoredProfile(prof.profile)
      return true
    } catch (e) {
      setError(errorMessage(e))
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  const refreshCameras = useCallback(async (rescan = false): Promise<void> => {
    // `camerasLoading` already starts true, so only an explicit rescan flips
    // it back — setting it unconditionally would be a synchronous setState
    // inside the mount effect below.
    if (rescan) setCamerasLoading(true)
    try {
      const r = await apiRef.current!.getCameras(rescan)
      setCameras(r.cameras ?? [])
    } catch {
      setCameras([])
    } finally {
      setCamerasLoading(false)
    }
  }, [])

  useEffect(() => {
    const api = apiFactory(port)
    apiRef.current = api

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    const attemptLoad = (): void => {
      void load().then((ok) => {
        if (!cancelled && !ok) {
          retryTimer = setTimeout(attemptLoad, retryDelayMs)
        }
      })
    }
    attemptLoad()

    // `camerasLoading` starts true and nothing else would ever clear it for a
    // consumer that opts out of camera polling — that would leave a spinner
    // that reads as a scan which never finishes. Deferred (rather than called
    // directly here) for the same reason the camera scan below is: a
    // synchronous setState inside the effect body would cascade a render.
    if (!shouldPollCameras) {
      queueMicrotask(() => {
        if (!cancelled) setCamerasLoading(false)
      })
    }

    // Deferred off the effect body: the probe's setState would otherwise run
    // synchronously here and cascade a render. Also lets the settings load
    // reach the sidecar first, since the scan can hold it for ~30 s.
    const cameraTimer = shouldPollCameras ? setTimeout(() => void refreshCameras(), 0) : null

    // Cheap enough (a few frame stats, no device I/O) to piggyback on the
    // same interval as health rather than run its own timer.
    const pollHealth = async (): Promise<void> => {
      // Skipped when a host view already owns capture state. The quality read
      // below is NOT skipped — it shares this timer and the tuning card needs
      // it regardless of who owns capture state.
      if (shouldPollHealth) {
        try {
          const h = await api.health()
          if (!cancelled) setCaptureState(h.state)
        } catch {
          // Sidecar not reachable yet; keep the last known capture state.
        }
      }
      try {
        const q = await api.getCameraQuality()
        if (!cancelled) setCameraQuality(q)
      } catch {
        // Sidecar not reachable yet; keep the last known reading.
      }
    }
    void pollHealth()
    const healthTimer = setInterval(() => void pollHealth(), healthPollMs)

    // Notice a camera plugged in after startup. The sidecar compares device
    // names (cheap) and only re-scans when they changed, so this is not the
    // 30 s scan on a timer. It skips itself while capture holds a device.
    const cameraTicker = shouldPollCameras
      ? setInterval(() => {
          if (!cancelled) void refreshCameras()
        }, cameraPollMs)
      : null
    return () => {
      cancelled = true
      if (retryTimer !== null) clearTimeout(retryTimer)
      if (cameraTimer !== null) clearTimeout(cameraTimer)
      clearInterval(healthTimer)
      if (cameraTicker !== null) clearInterval(cameraTicker)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    port,
    apiFactory,
    healthPollMs,
    retryDelayMs,
    cameraPollMs,
    shouldPollHealth,
    shouldPollCameras
  ])

  const update = useCallback(async (patch: SettingsUpdate): Promise<SettingsResponse> => {
    setSaving(true)
    setError(null)
    try {
      const r = await apiRef.current!.updateSettings(patch)
      setSettings(r)
      setSavedSettings(r)
      return r
    } catch (e) {
      setError(errorMessage(e))
      throw e
    } finally {
      setSaving(false)
    }
  }, [])

  const applyPreset = useCallback(async (name: string): Promise<SettingsResponse> => {
    setSaving(true)
    setError(null)
    try {
      const r = await apiRef.current!.applyPreset(name)
      setSettings(r)
      setSavedSettings(r)
      return r
    } catch (e) {
      setError(errorMessage(e))
      throw e
    } finally {
      setSaving(false)
    }
  }, [])

  const restoreDefaults = useCallback(
    async (): Promise<SettingsResponse> => update(DEFAULT_SETTINGS),
    [update]
  )

  const refresh = useCallback(async (): Promise<void> => {
    await load()
  }, [load])

  // Enumerating opens every camera device, so it is a deliberate action and
  // a no-op while capture holds one (the sidecar returns its cached list).
  // A failure leaves `cameras` empty, which falls the field back to a plain
  // index input rather than blanking the form.
  // Deliberately does NOT re-read settings. AdminPanel clears its unsaved
  // `draft` whenever the settings object changes identity, so reloading here
  // would discard the very edit the user stopped capture in order to make.
  // The restart-required lock keys off `captureState`, which this sets.
  const stopCapture = useCallback(async (): Promise<void> => {
    setStopping(true)
    setError(null)
    try {
      setCaptureState((await apiRef.current!.stop()).state)
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setStopping(false)
    }
  }, [])

  const startCapture = useCallback(async (): Promise<void> => {
    setStopping(true)
    setError(null)
    try {
      setCaptureState((await apiRef.current!.start()).state)
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setStopping(false)
    }
  }, [])

  // An unreachable backend is a normal answer here, not an error: the sidecar
  // returns reachable:false rather than a non-2xx, so only a transport failure
  // lands in the catch.
  const probe = useCallback(async (): Promise<DetectorProbeResponse> => {
    setProbing(true)
    setProbeResult(null)
    try {
      const r = await apiRef.current!.probeDetector()
      setProbeResult(r)
      return r
    } catch (e) {
      const failed = {
        backend: 'unknown',
        reachable: false,
        detail: errorMessage(e),
        latency_ms: null,
        class_names: []
      }
      setProbeResult(failed)
      return failed
    } finally {
      setProbing(false)
    }
  }, [])

  // Applies nothing itself — review-first: the operator sees fps_auto_exposure
  // vs fps_capped_exposure (the evidence) and the recommended patch before
  // choosing to apply it via applyProfile(). A 409 (capture running) surfaces
  // through `error` like any other API failure.
  const calibrate = useCallback(async (): Promise<CameraProfileResponse> => {
    setCalibrating(true)
    setError(null)
    try {
      const r = await apiRef.current!.calibrateCamera()
      setProfile(r)
      return r
    } catch (e) {
      setError(errorMessage(e))
      throw e
    } finally {
      setCalibrating(false)
    }
  }, [])

  // Applies the most recently calibrated profile's `recommended` patch, then
  // reloads settings so the form reflects the applied values.
  const applyProfile = useCallback(async (): Promise<SettingsResponse> => {
    setSaving(true)
    setError(null)
    try {
      const r = await apiRef.current!.applyCameraProfile()
      await load()
      return r
    } catch (e) {
      setError(errorMessage(e))
      throw e
    } finally {
      setSaving(false)
    }
  }, [load])

  // Deliberately does not touch savedSettings: this is an uncommitted
  // experiment, and the gap between `settings` and `savedSettings` is exactly
  // what the tuning card reports as unsaved changes.
  const liveUpdate = useCallback(async (patch: SettingsUpdate): Promise<void> => {
    setError(null)
    try {
      setSettings(await apiRef.current!.updateSettings(patch, false))
    } catch (e) {
      setError(errorMessage(e))
    }
  }, [])

  const save = useCallback(async (): Promise<SettingsResponse> => {
    setSaving(true)
    setError(null)
    try {
      const r = await apiRef.current!.saveSettings()
      setSettings(r)
      setSavedSettings(r)
      return r
    } catch (e) {
      setError(errorMessage(e))
      throw e
    } finally {
      setSaving(false)
    }
  }, [])

  return {
    settings,
    systemInfo,
    presets,
    recommended,
    captureState,
    loading,
    saving,
    error,
    refresh,
    update,
    applyPreset,
    restoreDefaults,
    probe,
    probing,
    probeResult,
    cameras,
    refreshCameras,
    camerasLoading,
    stopCapture,
    startCapture,
    stopping,
    cameraQuality,
    calibrate,
    calibrating,
    profile,
    applyProfile,
    storedProfile,
    liveUpdate,
    save,
    savedSettings
  }
}
