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

  it('getLogs() GETs /api/logs and returns parsed JSON', async () => {
    mockFetchOnce({
      session_id: 3,
      events: [
        {
          track_id: 1,
          class_name: 'banana',
          confidence: 0.8,
          max_conf: 0.91,
          entered_at: 100.0,
          left_at: null
        }
      ]
    })
    const api = createApiClient(8765)
    const r = await api.getLogs()
    expect(r.session_id).toBe(3)
    expect(r.events[0].class_name).toBe('banana')
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/logs')
    expect(init?.method ?? 'GET').toBe('GET')
  })

  it('getSettings() GETs /api/settings and returns parsed JSON', async () => {
    mockFetchOnce({
      active_model: 'yolo11n.pt',
      camera_index: 0,
      capture_width: 1280,
      capture_height: 720,
      capture_fps: 60,
      conf_threshold: 0.5,
      imgsz: 640,
      infer_frame_skip: 0,
      device: 'auto',
      preview_height: 720,
      track_expiry_s: 1.5,
      hot_reloadable_fields: ['infer_frame_skip'],
      restart_required_fields: ['active_model'],
      warnings: []
    })
    const api = createApiClient(8765)
    const r = await api.getSettings()
    expect(r.active_model).toBe('yolo11n.pt')
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/settings')
    expect(init?.method ?? 'GET').toBe('GET')
  })

  it('updateSettings() PATCHes /api/settings with a JSON body', async () => {
    mockFetchOnce({ infer_frame_skip: 2 })
    const api = createApiClient(8765)
    await api.updateSettings({ infer_frame_skip: 2 })
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/settings')
    expect(init?.method).toBe('PATCH')
    expect(init?.headers).toMatchObject({ 'Content-Type': 'application/json' })
    expect(JSON.parse(init?.body as string)).toEqual({ infer_frame_skip: 2 })
  })

  it('getSystemInfo() GETs /api/system-info', async () => {
    mockFetchOnce({
      cpu_count: 8,
      ram_gb: 16,
      cuda_available: false,
      gpu_name: null,
      gpu_vram_gb: null,
      recommended_preset: 'mid_range'
    })
    const api = createApiClient(8765)
    const r = await api.getSystemInfo()
    expect(r.recommended_preset).toBe('mid_range')
    const [url] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/system-info')
  })

  it('getPresets() GETs /api/presets', async () => {
    mockFetchOnce({ presets: [], recommended: 'low_end' })
    const api = createApiClient(8765)
    const r = await api.getPresets()
    expect(r.recommended).toBe('low_end')
    const [url] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/presets')
  })

  it('applyPreset() POSTs /api/settings/preset with the preset name', async () => {
    mockFetchOnce({ active_model: 'yolo11s.pt' })
    const api = createApiClient(8765)
    await api.applyPreset('mid_range')
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8765/api/settings/preset')
    expect(init?.method).toBe('POST')
    expect(JSON.parse(init?.body as string)).toEqual({ name: 'mid_range' })
  })
})
