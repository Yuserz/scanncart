import { useCallback, useEffect, useRef, useState } from 'react'
import { createApiClient, type ApiClient, type DatasetStatusResponse } from '../lib/api'

export interface DatasetStatusDeps {
  apiFactory?: (port: number) => ApiClient
  retryDelayMs?: number
}

export interface DatasetStatus {
  status: DatasetStatusResponse | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

// Reads dataset labeling progress for the Admin Panel.
//
// Deliberately polled *once*, not on a timer: the sidecar serves a snapshot file that
// only changes when someone runs sidecar/tools/label_progress.py by hand, so an
// interval would refetch identical bytes forever and imply a liveness the data does not
// have. `refresh` is exposed for the button instead, and the panel renders
// `age_seconds` so the operator can see how old the reading is.
export function useDatasetStatus(port: number, deps: DatasetStatusDeps = {}): DatasetStatus {
  const apiFactory = deps.apiFactory ?? createApiClient
  const retryDelayMs = deps.retryDelayMs ?? 1000

  const [status, setStatus] = useState<DatasetStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const apiRef = useRef<ApiClient | null>(null)
  // Guards against setting state after unmount. The retry loop below is what makes this
  // necessary rather than theoretical: a sidecar that is slow to start keeps a request
  // in flight, and its resolution can land after the panel is gone (which React reports
  // as an unwrapped act() update in tests).
  const aliveRef = useRef(true)

  // Returns success so the mount effect can retry: the renderer learns the sidecar's
  // port before uvicorn has necessarily finished starting, the same race ws.ts's
  // auto-reconnect covers. A failure here is not worth an error banner on its own —
  // the panel simply stays in its "no snapshot" state.
  const load = useCallback(async (): Promise<boolean> => {
    const api = apiRef.current
    if (!api || !aliveRef.current) return false
    setLoading(true)
    setError(null)
    try {
      const next = await api.getDatasetStatus()
      if (!aliveRef.current) return false
      setStatus(next)
      return true
    } catch (e) {
      if (!aliveRef.current) return false
      setError(errorMessage(e))
      return false
    } finally {
      if (aliveRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const api = apiFactory(port)
    apiRef.current = api
    aliveRef.current = true

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

    return () => {
      cancelled = true
      aliveRef.current = false
      if (retryTimer !== null) clearTimeout(retryTimer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [port, apiFactory, retryDelayMs])

  const refresh = useCallback(async (): Promise<void> => {
    await load()
  }, [load])

  return { status, loading, error, refresh }
}
