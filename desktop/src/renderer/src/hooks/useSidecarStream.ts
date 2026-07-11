import { useCallback, useEffect, useRef, useState } from 'react'
import { createApiClient, type ApiClient } from '../lib/api'
import {
  createStreamClient,
  type FrameMessage,
  type StatusMessage,
  type StreamClient,
  type StreamClientOptions
} from '../lib/ws'

export interface LoggedItem {
  track_id: number
  cls: string
  conf: number
  ts: number
}

export interface StreamDeps {
  apiFactory?: (port: number) => ApiClient
  streamFactory?: (opts: StreamClientOptions) => StreamClient
}

export interface SidecarStream {
  frame: FrameMessage | null
  statusState: string
  connected: boolean
  items: LoggedItem[]
  start: () => Promise<void>
  stop: () => Promise<void>
}

// Wires the REST + WebSocket clients into React state. Detections are deduped
// in-memory by track_id for the session (one row per item). On WS open, the
// persisted /api/logs rows seed items + the dedup set, but only to recover a
// session that is actually running (a reconnect mid-capture) — a fresh
// launch or idle app leaves the log empty, and start() resets it for a new
// capture session.
export function useSidecarStream(port: number, deps: StreamDeps = {}): SidecarStream {
  const apiFactory = deps.apiFactory ?? createApiClient
  const streamFactory = deps.streamFactory ?? createStreamClient

  const [frame, setFrame] = useState<FrameMessage | null>(null)
  const [statusState, setStatusState] = useState<string>('idle')
  const [connected, setConnected] = useState(false)
  const [items, setItems] = useState<LoggedItem[]>([])

  const apiRef = useRef<ApiClient | null>(null)
  const seenRef = useRef<Set<number>>(new Set())
  const statusRef = useRef<string>('idle')

  const setStatus = useCallback((s: string): void => {
    statusRef.current = s
    setStatusState(s)
  }, [])

  useEffect(() => {
    const api = apiFactory(port)
    apiRef.current = api
    seenRef.current = new Set()
    let cancelled = false

    const seedFromLogs = async (): Promise<void> => {
      try {
        const res = await api.getLogs()
        if (cancelled) return
        // Only recover the log for a session that is actually running (reconnect
        // recovery). On a fresh launch / idle the sidecar's most-recent session is
        // stale, so the log stays empty until the user starts a new capture.
        if (statusRef.current !== 'running' || res.session_id == null) return
        const seeded: LoggedItem[] = []
        for (const e of res.events) {
          if (seenRef.current.has(e.track_id)) continue
          seenRef.current.add(e.track_id)
          seeded.push({ track_id: e.track_id, cls: e.class_name, conf: e.max_conf, ts: e.entered_at })
        }
        if (seeded.length > 0) setItems((prev) => [...prev, ...seeded])
      } catch {
        // /api/logs unavailable (sidecar not ready): keep the live-only log.
      }
    }

    const onFrame = (msg: FrameMessage): void => {
      setFrame(msg)
      const fresh: LoggedItem[] = []
      for (const d of msg.detections) {
        if (d.track_id == null || seenRef.current.has(d.track_id)) continue
        seenRef.current.add(d.track_id)
        fresh.push({ track_id: d.track_id, cls: d.cls, conf: d.conf, ts: msg.ts })
      }
      if (fresh.length > 0) setItems((prev) => [...prev, ...fresh])
    }

    const onStatus = (msg: StatusMessage): void => setStatus(msg.state)

    const client = streamFactory({
      port,
      onFrame,
      onStatus,
      onOpen: () => {
        setConnected(true)
        void seedFromLogs()
      },
      onClose: () => setConnected(false)
    })
    client.connect()

    return () => {
      cancelled = true
      client.close()
    }
  }, [port, apiFactory, streamFactory])

  const start = useCallback(async (): Promise<void> => {
    seenRef.current = new Set()
    setItems([])
    const r = await apiRef.current!.start()
    setStatus(r.state)
  }, [setStatus])

  const stop = useCallback(async (): Promise<void> => {
    const r = await apiRef.current!.stop()
    setStatus(r.state)
  }, [setStatus])

  return { frame, statusState, connected, items, start, stop }
}
