// WebSocket stream client for the SCANnCART sidecar (spec §4.1).
// The renderer connects directly to ws://127.0.0.1:<port>/ws/stream and
// auto-reconnects, so it tolerates the sidecar not being ready yet.

export interface Detection {
  track_id: number | null
  cls: string
  conf: number
  box: [number, number, number, number]
}

export interface FrameStats {
  infer_fps: number
  capture_fps: number
  latency_ms: number
}

export interface FrameMessage {
  type: 'frame'
  ts: number
  seq: number
  jpeg: string
  detections: Detection[]
  stats: FrameStats
}

export interface StatusMessage {
  type: 'status'
  state: string
  detail?: string
}

export type StreamMessage = FrameMessage | StatusMessage

// Minimal structural type so a fake or the global WebSocket both satisfy it.
interface WSLike {
  onopen: (() => void) | null
  onmessage: ((e: { data: unknown }) => void) | null
  onclose: (() => void) | null
  onerror: ((e: unknown) => void) | null
  close(): void
}

export interface StreamClientOptions {
  port: number
  onFrame?: (msg: FrameMessage) => void
  onStatus?: (msg: StatusMessage) => void
  onOpen?: () => void
  onClose?: () => void
  reconnectDelayMs?: number
  wsFactory?: (url: string) => WSLike
}

export interface StreamClient {
  connect(): void
  close(): void
}

export function createStreamClient(opts: StreamClientOptions): StreamClient {
  const url = `ws://127.0.0.1:${opts.port}/ws/stream`
  const delay = opts.reconnectDelayMs ?? 1000
  const factory = opts.wsFactory ?? ((u: string) => new WebSocket(u) as unknown as WSLike)

  let ws: WSLike | null = null
  let closedByUser = false
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  function handleMessage(data: unknown): void {
    if (typeof data !== 'string') return
    let msg: StreamMessage
    try {
      msg = JSON.parse(data) as StreamMessage
    } catch {
      return // ignore malformed frames rather than crashing the stream
    }
    if (msg.type === 'frame') opts.onFrame?.(msg)
    else if (msg.type === 'status') opts.onStatus?.(msg)
  }

  function scheduleReconnect(): void {
    if (closedByUser || reconnectTimer !== null) return
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      open()
    }, delay)
  }

  function open(): void {
    ws = factory(url)
    ws.onopen = () => opts.onOpen?.()
    ws.onmessage = (e) => handleMessage(e.data)
    ws.onerror = () => {
      /* onclose follows; reconnect handled there */
    }
    ws.onclose = () => {
      opts.onClose?.()
      scheduleReconnect()
    }
  }

  return {
    connect(): void {
      closedByUser = false
      open()
    },
    close(): void {
      closedByUser = true
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
      ws?.close()
      ws = null
    }
  }
}
