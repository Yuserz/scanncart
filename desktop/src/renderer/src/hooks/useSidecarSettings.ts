import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createApiClient,
  type ApiClient,
  type PresetInfo,
  type SettingsResponse,
  type SettingsUpdate,
  type SystemInfoResponse
} from '../lib/api'
import { DEFAULT_SETTINGS } from '../lib/settingsDefaults'

export interface SettingsDeps {
  apiFactory?: (port: number) => ApiClient
  healthPollMs?: number
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
  const retryDelayMs = deps.retryDelayMs ?? 1000

  const [settings, setSettings] = useState<SettingsResponse | null>(null)
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null)
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [recommended, setRecommended] = useState<string | null>(null)
  const [captureState, setCaptureState] = useState('idle')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
    return () => {
      cancelled = true
      if (retryTimer !== null) clearTimeout(retryTimer)
      clearInterval(healthTimer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [port, apiFactory, healthPollMs, retryDelayMs])

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
    restoreDefaults
  }
}
