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
  const deps = {
    apiFactory: () => ({
      health: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      getLogs: vi.fn(async () => logs)
    }),
    streamFactory: (o: StreamClientOptions) => {
      opts = o
      return { connect: vi.fn(), close: vi.fn() }
    }
  }
  return { deps, opts: () => opts! }
}

describe('useSidecarStream reconciliation', () => {
  it('seeds items from /api/logs on open, then merges live frames without duplicates', async () => {
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
    // Seeded from the persisted log.
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
})
