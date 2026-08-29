import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createApiClient,
  type ApiClient,
  type CameraInfo,
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
      const [s, sys, p] = await Promise.all([
        api.getSettings(),
        api.getSystemInfo(),
        api.getPresets()
      ])
      setSettings(s)
      setSystemInfo(sys)
      setPresets(p.presets)
      setRecommended(p.recommended)
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

    // Deferred off the effect body: the probe's setState would otherwise run
    // synchronously here and cascade a render. Also lets the settings load
    // reach the sidecar first, since the scan can hold it for ~30 s.
    const cameraTimer = setTimeout(() => void refreshCameras(), 0)

    const pollHealth = async (): Promise<void> => {
      try {
        const h = await api.health()
        if (!cancelled) setCaptureState(h.state)
      } catch {
        // Sidecar not reachable yet; keep the last known capture state.
      }
    }
    void pollHealth()
    const healthTimer = setInterval(() => void pollHealth(), healthPollMs)

    // Notice a camera plugged in after startup. The sidecar compares device
    // names (cheap) and only re-scans when they changed, so this is not the
    // 30 s scan on a timer. It skips itself while capture holds a device.
    const cameraTicker = setInterval(() => {
      if (!cancelled) void refreshCameras()
    }, cameraPollMs)
    return () => {
      cancelled = true
      if (retryTimer !== null) clearTimeout(retryTimer)
      clearTimeout(cameraTimer)
      clearInterval(healthTimer)
      clearInterval(cameraTicker)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [port, apiFactory, healthPollMs, retryDelayMs, cameraPollMs])

  const update = useCallback(async (patch: SettingsUpdate): Promise<SettingsResponse> => {
    setSaving(true)
    setError(null)
    try {
      const r = await apiRef.current!.updateSettings(patch)
      setSettings(r)
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
    stopping
  }
}
