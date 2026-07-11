import { describe, it, expect, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useSidecarStream } from './useSidecarStream'
import type { StreamClientOptions, FrameMessage } from '../lib/ws'
import type { LogsResponse } from '../lib/api'

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

function makeDeps(logs: LogsResponse) {
  let opts: StreamClientOptions | null = null
  const start = vi.fn(async () => ({ state: 'running' }))
  const stop = vi.fn(async () => ({ state: 'idle' }))
  const deps = {
    apiFactory: () => ({
      health: vi.fn(),
      start,
      stop,
      getLogs: vi.fn(async () => logs)
    }),
    streamFactory: (o: StreamClientOptions) => {
      opts = o
      return { connect: vi.fn(), close: vi.fn() }
    }
  }
  return { deps, opts: () => opts!, start, stop }
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
      opts().onFrame?.(
        frameWith([{ track_id: 8, cls: 'apple', conf: 0.7, box: [0, 0, 0.5, 0.5] }])
      )
    )
    expect(result.current.items).toHaveLength(2)
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
