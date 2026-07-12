import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createStreamClient } from './ws'
import type { FrameMessage, StatusMessage } from './ws'

class FakeWS {
  static instances: FakeWS[] = []
  url: string
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: ((e: unknown) => void) | null = null
  readyState = 0
  constructor(url: string) {
    this.url = url
    FakeWS.instances.push(this)
  }
  close(): void {
    this.readyState = 3
    this.onclose?.()
  }
  // eslint-disable-next-line @typescript-eslint/no-empty-function -- intentional no-op stub
  send(): void {}
  // --- test helpers ---
  emitOpen(): void {
    this.readyState = 1
    this.onopen?.()
  }
  emitJSON(obj: unknown): void {
    this.onmessage?.({ data: JSON.stringify(obj) })
  }
  emitRaw(data: string): void {
    this.onmessage?.({ data })
  }
  emitClose(): void {
    this.readyState = 3
    this.onclose?.()
  }
}

const frame: FrameMessage = {
  type: 'frame',
  ts: 1,
  seq: 5,
  jpeg: 'AAAA',
  detections: [{ track_id: 1, cls: 'banana', conf: 0.9, box: [0.1, 0.2, 0.3, 0.4] }],
  stats: { infer_fps: 20, capture_fps: 60, latency_ms: 80 }
}

describe('createStreamClient', () => {
  beforeEach(() => {
    FakeWS.instances = []
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('connects to the correct ws URL', () => {
    const c = createStreamClient({ port: 8765, wsFactory: (u) => new FakeWS(u) as never })
    c.connect()
    expect(FakeWS.instances[0].url).toBe('ws://127.0.0.1:8765/ws/stream')
  })

  it('routes frame messages to onFrame', () => {
    const onFrame = vi.fn()
    const c = createStreamClient({ port: 1, onFrame, wsFactory: (u) => new FakeWS(u) as never })
    c.connect()
    FakeWS.instances[0].emitOpen()
    FakeWS.instances[0].emitJSON(frame)
    expect(onFrame).toHaveBeenCalledWith(frame)
  })

  it('routes status messages to onStatus', () => {
    const onStatus = vi.fn()
    const status: StatusMessage = { type: 'status', state: 'camera_lost', detail: 'unplugged' }
    const c = createStreamClient({ port: 1, onStatus, wsFactory: (u) => new FakeWS(u) as never })
    c.connect()
    FakeWS.instances[0].emitJSON(status)
    expect(onStatus).toHaveBeenCalledWith(status)
  })

  it('ignores malformed JSON without throwing', () => {
    const onFrame = vi.fn()
    const c = createStreamClient({ port: 1, onFrame, wsFactory: (u) => new FakeWS(u) as never })
    c.connect()
    expect(() => FakeWS.instances[0].emitRaw('{not json')).not.toThrow()
    expect(onFrame).not.toHaveBeenCalled()
  })

  it('auto-reconnects after an unexpected drop', () => {
    const c = createStreamClient({
      port: 1,
      reconnectDelayMs: 500,
      wsFactory: (u) => new FakeWS(u) as never
    })
    c.connect()
    expect(FakeWS.instances).toHaveLength(1)
    FakeWS.instances[0].emitClose() // server drop
    vi.advanceTimersByTime(500)
    expect(FakeWS.instances).toHaveLength(2)
  })

  it('close() stops further reconnects', () => {
    const c = createStreamClient({
      port: 1,
      reconnectDelayMs: 500,
      wsFactory: (u) => new FakeWS(u) as never
    })
    c.connect()
    c.close()
    vi.advanceTimersByTime(2000)
    expect(FakeWS.instances).toHaveLength(1)
  })
})
