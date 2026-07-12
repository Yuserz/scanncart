import { describe, it, expect, vi } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LiveView } from './LiveView'
import type { StreamDeps } from '../hooks/useSidecarStream'
import type { StreamClientOptions, FrameMessage } from '../lib/ws'

function frameWith(dets: FrameMessage['detections']): FrameMessage {
  return {
    type: 'frame',
    ts: 123,
    seq: 1,
    jpeg: 'AAAA',
    detections: dets,
    stats: { infer_fps: 22.4, capture_fps: 60, latency_ms: 88 }
  }
}

const makeHarness = (): {
  deps: StreamDeps
  opts: () => StreamClientOptions
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
} => {
  let captured: StreamClientOptions | null = null
  const start = vi.fn(async () => ({ state: 'running' }))
  const stop = vi.fn(async () => ({ state: 'idle' }))
  const deps = {
    apiFactory: () => ({
      health: vi.fn(),
      start,
      stop,
      getLogs: vi.fn(async () => ({ session_id: null, events: [] })),
      getSettings: vi.fn(),
      updateSettings: vi.fn(),
      getSystemInfo: vi.fn(),
      getPresets: vi.fn(),
      applyPreset: vi.fn()
    }),
    streamFactory: (opts: StreamClientOptions) => {
      captured = opts
      return { connect: vi.fn(), close: vi.fn() }
    }
  }
  return { deps, opts: () => captured!, start, stop }
}

describe('LiveView', () => {
  it('renders a detection box, item-log row, and stats from a frame', () => {
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    act(() => {
      h.opts().onOpen?.()
      h.opts().onFrame?.(
        frameWith([{ track_id: 1, cls: 'banana', conf: 0.9, box: [0.1, 0.2, 0.3, 0.4] }])
      )
    })
    expect(screen.getByTestId('conn')).toHaveTextContent('connected')
    expect(screen.getAllByTestId('det-box')).toHaveLength(1)
    expect(screen.getByTestId('det-box')).toHaveTextContent('banana 90%')
    expect(screen.getByTestId('item-log')).toHaveTextContent('banana (90%)')
    expect(screen.getByTestId('stat-infer-fps')).toHaveTextContent('22.4')
    expect(screen.getByTestId('stat-capture-fps')).toHaveTextContent('60')
    expect(screen.getByTestId('stat-latency')).toHaveTextContent('88')
    expect(screen.getByTestId('stat-tracked')).toHaveTextContent('1')
  })

  it('does not add a duplicate item-log row for a repeated track_id', () => {
    const h = makeHarness()
    render(<LiveView port={1} deps={h.deps} />)
    const det = [
      {
        track_id: 7,
        cls: 'apple',
        conf: 0.8,
        box: [0, 0, 0.5, 0.5] as [number, number, number, number]
      }
    ]
    act(() => {
      h.opts().onFrame?.(frameWith(det))
      h.opts().onFrame?.(frameWith(det))
    })
    expect(screen.getByTestId('item-log').querySelectorAll('li')).toHaveLength(1)
  })

  it('Start button calls the REST start endpoint', async () => {
    const h = makeHarness()
    render(<LiveView port={1} deps={h.deps} />)
    await userEvent.click(screen.getByRole('button', { name: 'Start' }))
    expect(h.start).toHaveBeenCalledTimes(1)
  })
})
