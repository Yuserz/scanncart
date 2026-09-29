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
  // Detections this frame's inference produced and the sidecar's frame-clamp filter dropped
  // (settings.suppress_clamped_detections). Optional because the field is newer than the wire:
  // a sidecar that predates it sends no such key, and `undefined` has to read as "nothing
  // suppressed" rather than as a missing reading.
  suppressed?: number
  // Whether this frame's inference was refused by a server that is up but at capacity (HTTP
  // 503/429). The sidecar drops that frame rather than the capture, so an empty frame here is
  // *not* evidence of an empty counter — which is the reading a log with nothing in it invites.
  // Optional for the same reason as `suppressed`: a sidecar that predates the field sends no such
  // key, and `undefined` has to read as "not shed" rather than as a missing reading.
  shed?: boolean
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
  // The class names the *running* model predicts, empty until the sidecar knows them. Empty means
  // "not known yet" - a detector has no vocabulary until its first inference - and never "predicts
  // nothing", which is why the stats strip's class chip is absent rather than zero.
  //
  // It arrives on the status protocol because it is a fact about the capture rather than about a
  // request, and it is sent for a clean model too: the count itself is a readout, not only the
  // input to a verdict. The handshake replays it, so a renderer that connects mid-capture is not
  // left guessing from the labels going by.
  class_names?: string[]
  // What is wrong with that class list (`app/roster.py` on the sidecar side), empty when there is
  // nothing wrong or nothing is known yet. Judged by the sidecar and rendered as it arrives — and
  // against the roster of the running weight's own generation, which is a fact only the sidecar
  // has: the roster is not mirrored on this side, so the sentences are the only description of it.
  class_warnings?: string[]
}

// Whether the server the selected backend calls is answering (`app/inference_health.py` on the
// sidecar side). Its own message type rather than two more fields on StatusMessage, because it is a
// fact about the *configured backend* rather than about a capture: it is equally true before one is
// started, survives a start/stop, and is the one thing on this wire that can explain a `local_api`
// run where the camera works, the preview streams and nothing is ever detected.
//
// Sent on every connect (so a window opened mid-session is not left to infer it from silence) and
// then only when `state` changes.
export interface InferenceMessage {
  type: 'inference'
  backend: string
  url: string
  // `unknown` is not "fine" and not "broken": it means this backend has no server to watch
  // (`native`), or the first probe has not landed yet. Nothing is rendered for it — a verdict that
  // has not been reached must not be drawn as one.
  state: 'unknown' | 'ok' | 'unresponsive'
  // Why, in the endpoint's own words, when `state` is 'unresponsive'. Empty otherwise.
  detail: string
  // Seconds since the last probe, as of the moment this message was built. Same meaning as the copy
  // /api/health carries (the same payload, the same live reading). Nothing on this side renders it:
  // the Admin Panel is where the age is shown, and it reads it from the polled copy, which is fresh
  // by construction rather than aging from a push. `null` means no probe has happened at all.
  age_seconds?: number | null
}

export type StreamMessage = FrameMessage | StatusMessage | InferenceMessage

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
  onInference?: (msg: InferenceMessage) => void
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
    // JSON.parse can yield null/non-objects; guard before reading .type.
    if (typeof msg !== 'object' || msg === null) return
    if (msg.type === 'frame') opts.onFrame?.(msg)
    else if (msg.type === 'status') opts.onStatus?.(msg)
    else if (msg.type === 'inference') opts.onInference?.(msg)
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
