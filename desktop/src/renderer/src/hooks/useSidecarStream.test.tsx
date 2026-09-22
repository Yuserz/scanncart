import { describe, it, expect, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useSidecarStream, type StreamDeps } from './useSidecarStream'
import type { StreamClientOptions, FrameMessage } from '../lib/ws'
import type { ApiClient, LogsResponse } from '../lib/api'

function frameWith(dets: FrameMessage['detections']): FrameMessage {
  return {
    type: 'frame',
    ts: 123,
    seq: 1,
    jpeg: 'AAAA',
    detections: dets,
    stats: { infer_fps: 1, capture_fps: 1, latency_ms: 1 }
  }
}

const makeDeps = (
  logs: LogsResponse
): {
  deps: StreamDeps
  opts: () => StreamClientOptions
  // Exposed so a test can count the /api/logs calls. Which attempts happened is the behaviour
  // under test for the reconnect case, and it is not visible from the items alone when the two
  // triggers can each produce the same result.
  api: ApiClient
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
} => {
  let opts: StreamClientOptions | null = null
  const start = vi.fn(async () => ({ state: 'running' }))
  const stop = vi.fn(async () => ({ state: 'idle' }))
  const api: ApiClient = {
    health: vi.fn(),
    start,
    stop,
    getLogs: vi.fn(async () => logs),
    getSettings: vi.fn(),
    updateSettings: vi.fn(),
    getSystemInfo: vi.fn(),
    getPresets: vi.fn(),
    applyPreset: vi.fn(),
    probeDetector: vi.fn(),
    getCameras: vi.fn(async () => ({
      cameras: [{ index: 0, name: 'Fake Cam', width: 1280, height: 720 }],
      probed: true,
      detail: ''
    })),
    getCameraQuality: vi.fn(async () => ({
      available: false,
      brightness: 0,
      contrast: 0,
      sharpness: 0,
      capture_fps: 0,
      target_fps: 0,
      verdicts: {},
      detail: ''
    })),
    calibrateCamera: vi.fn(),
    applyCameraProfile: vi.fn(),
    saveSettings: vi.fn(),
    getCameraProfile: vi.fn(async () => ({ profile: null })),
    // Required by ApiClient; never called from this view.
    getDatasetStatus: vi.fn(),
    getModels: vi.fn(),
    recordResizeMode: vi.fn()
  }
  const deps: StreamDeps = {
    apiFactory: () => api,
    streamFactory: (o: StreamClientOptions) => {
      opts = o
      return { connect: vi.fn(), close: vi.fn() }
    }
  }
  return { deps, opts: () => opts!, api, start, stop }
}

describe('useSidecarStream reconciliation', () => {
  it('does not seed items from /api/logs when idle (fresh launch)', async () => {
    const { deps, opts } = makeDeps({
      session_id: 1,
      events: [
        {
          track_id: 7,
          class_name: 'banana',
          confidence: 0.8,
          max_conf: 0.9,
          entered_at: 100,
          left_at: null
        }
      ]
    })
    const { result } = renderHook(() => useSidecarStream(8765, deps))

    act(() => opts().onOpen?.())

    // Give any in-flight getLogs() promise a chance to resolve; items must
    // stay empty because the hook is idle, not running.
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(result.current.items).toHaveLength(0)
  })

  it('seeds items from /api/logs on reconnect while running, then merges live frames without duplicates', async () => {
    const { deps, opts } = makeDeps({
      session_id: 1,
      events: [
        {
          track_id: 7,
          class_name: 'banana',
          confidence: 0.8,
          max_conf: 0.9,
          entered_at: 100,
          left_at: null
        }
      ]
    })
    const { result } = renderHook(() => useSidecarStream(8765, deps))

    // Session is running (e.g. reported by a status message on the socket).
    act(() => opts().onStatus?.({ type: 'status', state: 'running' }))
    // Reconnect: socket re-opens mid-session.
    act(() => opts().onOpen?.())

    // Seeded from the persisted log because a session is running.
    await waitFor(() => expect(result.current.items).toHaveLength(1))
    expect(result.current.items[0]).toMatchObject({ track_id: 7, cls: 'banana' })

    // A live frame for the already-seeded track must not duplicate it.
    act(() =>
      opts().onFrame?.(
        frameWith([{ track_id: 7, cls: 'banana', conf: 0.95, box: [0, 0, 0.5, 0.5] }])
      )
    )
    expect(result.current.items).toHaveLength(1)

    // A live frame for a new track appends.
    act(() =>
      opts().onFrame?.(frameWith([{ track_id: 8, cls: 'apple', conf: 0.7, box: [0, 0, 0.5, 0.5] }]))
    )
    expect(result.current.items).toHaveLength(2)
  })

  it('seeds the log when the handshake status says a capture is already running', async () => {
    // The reload sequence, which is the order that matters now: a fresh renderer opens the socket,
    // and the sidecar's *first* message tells it a capture is already in flight.
    //
    // The open-time attempt is flushed to completion *before* that message arrives, because
    // otherwise this test passes either way: `getLogs()` is a real request and the status can beat
    // it home, leaving the first attempt to find `running` and seed by luck. Relying on that would
    // mean the recovery worked only when /api/logs was slow, so here the attempt is made to give up
    // first — which is the case that needs the second trigger.
    const { deps, opts, api } = makeDeps({
      session_id: 1,
      events: [
        {
          track_id: 7,
          class_name: 'banana',
          confidence: 0.8,
          max_conf: 0.9,
          entered_at: 100,
          left_at: null
        }
      ]
    })
    const { result } = renderHook(() => useSidecarStream(8765, deps))

    act(() => opts().onOpen?.())
    // A macrotask hop, so every microtask behind that attempt has run rather than "probably has".
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(result.current.items).toHaveLength(0)

    // No prior state: the handshake message is what carries it.
    act(() => opts().onStatus?.({ type: 'status', state: 'running' }))

    await waitFor(() => expect(result.current.items).toHaveLength(1))
    expect(result.current.items[0]).toMatchObject({ track_id: 7, cls: 'banana' })
    // Two asks, and the second one is the point: the first gave up while the state was unknown.
    expect(api.getLogs).toHaveBeenCalledTimes(2)
  })

  it('does not seed when the handshake status says idle', async () => {
    // The other half of that rule, and the one that would be a regression rather than a missed
    // recovery: a fresh launch must not be handed the sidecar's most recent — finished — session,
    // which would put a previous customer's items in the log before anything was scanned.
    const { deps, opts } = makeDeps({
      session_id: 4,
      events: [
        {
          track_id: 9,
          class_name: 'milo',
          confidence: 0.8,
          max_conf: 0.9,
          entered_at: 100,
          left_at: null
        }
      ]
    })
    const { result } = renderHook(() => useSidecarStream(8765, deps))

    act(() => opts().onOpen?.())
    act(() => opts().onStatus?.({ type: 'status', state: 'idle' }))

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(result.current.items).toHaveLength(0)
  })

  it('a new start() resets the log', async () => {
    const { deps, opts, start } = makeDeps({ session_id: null, events: [] })
    const { result } = renderHook(() => useSidecarStream(8765, deps))

    // Put some items into the list via a running status + a live frame.
    act(() => opts().onStatus?.({ type: 'status', state: 'running' }))
    act(() =>
      opts().onFrame?.(
        frameWith([{ track_id: 1, cls: 'banana', conf: 0.9, box: [0, 0, 0.5, 0.5] }])
      )
    )
    await waitFor(() => expect(result.current.items).toHaveLength(1))

    await act(async () => {
      await result.current.start()
    })

    expect(start).toHaveBeenCalledTimes(1)
    expect(result.current.items).toHaveLength(0)
  })
})
