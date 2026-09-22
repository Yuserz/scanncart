import { useCallback, useEffect, useRef, useState } from 'react'
import { createApiClient, type ApiClient } from '../lib/api'
import {
  createStreamClient,
  type FrameMessage,
  type StatusMessage,
  type StreamClient,
  type StreamClientOptions
} from '../lib/ws'
import type { SettingsDeps } from './useSidecarSettings'

export interface LoggedItem {
  track_id: number
  cls: string
  conf: number
  ts: number
}

export interface StreamDeps {
  apiFactory?: (port: number) => ApiClient
  streamFactory?: (opts: StreamClientOptions) => StreamClient
  // Passed through to CameraTuning; never used by the stream hook itself.
  // Lives here so LiveView keeps taking exactly one `deps` prop.
  settingsDeps?: SettingsDeps
}

export interface SidecarStream {
  frame: FrameMessage | null
  statusState: string
  // The class names the running model declares, as the sidecar reports them. Empty means nothing is
  // known yet — a detector has no vocabulary until its first inference — and never "predicts
  // nothing", which is what keeps the stats strip's class chip absent rather than reading zero.
  // Set from every status message (including the handshake, which is how a reloaded renderer learns
  // it mid-capture) and cleared when a capture starts or stops, because neither one has a loaded
  // model to describe yet.
  classNames: string[]
  // What is wrong with the class list of the model that is actually running, as the sidecar judged
  // it against its 8-class roster. Empty means either nothing is wrong or nothing is known yet,
  // which are the same to this side of the wire — and the distinction the sidecar keeps, since it
  // is the process that knows when a detector has inferred. Set from the same messages as
  // `classNames`, and cleared with them for the same reason.
  classWarnings: string[]
  // Last error from the sidecar: a failed start/stop, or a capture that died
  // mid-session (the pipeline reports that as a status message with a detail).
  error: string | null
  clearError: () => void
  connected: boolean
  items: LoggedItem[]
  start: () => Promise<void>
  stop: () => Promise<void>
}

// Wires the REST + WebSocket clients into React state. Detections are deduped
// in-memory by track_id for the session (one row per item). The persisted
// /api/logs rows seed items + the dedup set, but only to recover a session that
// is actually running (a reconnect — or a reload — mid-capture); a fresh launch
// or idle app leaves the log empty, and start() resets it for a new capture
// session.
//
// The seeding is triggered twice per connection, and it needs both. The sidecar
// sends the current capture state as the first message on `/ws/stream`, so on a
// reload mid-capture the state arrives *after* `onOpen` — seeding only there would
// ask /api/logs while the hook still believed it was idle and skip the recovery
// entirely. An open socket's own status is therefore the second trigger, and a
// latch keeps it to one attempt per connection.
export function useSidecarStream(port: number, deps: StreamDeps = {}): SidecarStream {
  const apiFactory = deps.apiFactory ?? createApiClient
  const streamFactory = deps.streamFactory ?? createStreamClient

  const [frame, setFrame] = useState<FrameMessage | null>(null)
  const [statusState, setStatusState] = useState<string>('idle')
  const [classNames, setClassNames] = useState<string[]>([])
  const [classWarnings, setClassWarnings] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
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

    // Once per connection. Not a correctness guard — `seenRef` already keeps a duplicate
    // row out — but the two triggers below fire on the same reconnect, and one /api/logs
    // call is the honest number of times to ask.
    let attemptedSeed = false
    const seedFromLogs = async (): Promise<void> => {
      if (attemptedSeed) return
      try {
        const res = await api.getLogs()
        if (cancelled) return
        // Only recover the log for a session that is actually running (reconnect
        // recovery). On a fresh launch / idle the sidecar's most-recent session is
        // stale, so the log stays empty until the user starts a new capture. Not
        // latched on this path: at `onOpen` the state may simply not have arrived yet.
        if (statusRef.current !== 'running' || res.session_id == null) return
        attemptedSeed = true
        const recovered: LoggedItem[] = []
        for (const e of res.events) {
          if (seenRef.current.has(e.track_id)) continue
          seenRef.current.add(e.track_id)
          recovered.push({
            track_id: e.track_id,
            cls: e.class_name,
            conf: e.max_conf,
            ts: e.entered_at
          })
        }
        if (recovered.length > 0) setItems((prev) => [...prev, ...recovered])
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

    const onStatus = (msg: StatusMessage): void => {
      // `setStatus` writes the ref synchronously, so the seed below sees the state this
      // message just reported rather than the one it replaced.
      setStatus(msg.state)
      // Both replaced rather than merged, like the state above: the sidecar always sends its
      // current answer, and `[]` is a real answer (nothing wrong, or not known yet). A merge would
      // keep a warning alive after the model it described had been replaced, and would leave a
      // stale count beside it — the two describe one model and have to change together.
      setClassNames(msg.class_names ?? [])
      setClassWarnings(msg.class_warnings ?? [])
      // A status that carries a detail is reporting something that went wrong, in either of the
      // two shapes the sidecar has for it: `state: 'error'` when the capture thread dies
      // mid-session (e.g. the inference server went away), and the same detail on the *handshake*
      // when the capture it describes died before this client connected — there the state reads
      // `idle`, because that is what the sidecar is, and the detail is the stored reason. Dropping
      // it left a frozen preview with no explanation, and on a reload it left a dead capture
      // looking exactly like one that was never started.
      //
      // Gating on `state === 'error'` would therefore lose the recovered case, and gating on the
      // state at all is redundant: `detail` is only ever an error explanation. Dismissing is not
      // remembered across a reconnect — the sidecar still has the same reason, and this is the
      // banner's own Dismiss button, not a setting.
      if (msg.detail) setError(msg.detail)
      // The handshake status: if it says a capture is already running, this connection is
      // joining one, so recover its log the same way a later reconnect would.
      if (msg.state === 'running') void seedFromLogs()
    }

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
  }, [port, apiFactory, streamFactory, setStatus])

  // start/stop swallow their rejection deliberately: the sidecar returns a
  // real status for a missing API key (401), an unreachable server (503) or a
  // timeout (504), and the caller renders `error`. Rethrowing here only
  // produced an unhandled rejection and a silently reset button.
  const start = useCallback(async (): Promise<void> => {
    seenRef.current = new Set()
    setItems([])
    setError(null)
    // Cleared before the request: this capture has not loaded a model yet, and the previous
    // capture's finding described a model that is about to be replaced. The sidecar's status
    // message is what fills it in, once it has an inference to read the classes from.
    setClassNames([])
    setClassWarnings([])
    try {
      const r = await apiRef.current!.start()
      setStatus(r.state)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [setStatus])

  const stop = useCallback(async (): Promise<void> => {
    try {
      const r = await apiRef.current!.stop()
      setStatus(r.state)
      // Nothing is loaded after a stop, so a finding about the last model would be stale in the
      // one place it is most likely to be read (the log sitting there afterwards).
      setClassNames([])
      setClassWarnings([])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [setStatus])

  const clearError = useCallback((): void => setError(null), [])

  return {
    frame,
    statusState,
    connected,
    items,
    start,
    stop,
    error,
    clearError,
    classNames,
    classWarnings
  }
}
