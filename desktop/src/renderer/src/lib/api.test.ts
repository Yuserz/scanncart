import { describe, it, expect, vi, beforeEach } from 'vitest'
import { createApiClient } from './api'

function mockFetchOnce(body: unknown, ok = true, status = 200): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok,
      status,
      json: async () => body
    }))
  )
}

describe('createApiClient', () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
  })

  it('health() GETs /api/health and returns parsed JSON', async () => {
    mockFetchOnce({ state: 'idle', active_model: 'yolo11n.pt', device: 'cpu' })
    const api = createApiClient(8765)
    const h = await api.health()
    expect(h.active_model).toBe('yolo11n.pt')
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/health')
    // GET has no explicit method or method 'GET'
    expect(init?.method ?? 'GET').toBe('GET')
  })

  it('start() POSTs /api/capture/start', async () => {
    mockFetchOnce({ state: 'running' })
    const api = createApiClient(9000)
    const r = await api.start()
    expect(r.state).toBe('running')
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:9000/api/capture/start')
    expect(init?.method).toBe('POST')
  })

  it('stop() POSTs /api/capture/stop', async () => {
    mockFetchOnce({ state: 'idle' })
    const api = createApiClient(8765)
    const r = await api.stop()
    expect(r.state).toBe('idle')
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/capture/stop')
    expect(init?.method).toBe('POST')
  })

  it('rejects with a useful error on non-OK response', async () => {
    mockFetchOnce({ detail: 'boom' }, false, 500)
    const api = createApiClient(8765)
    await expect(api.health()).rejects.toThrow(/500/)
  })
})
