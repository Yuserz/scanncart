import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { BasketTestView } from './BasketTestView'
import type { StreamDeps } from '../hooks/useSidecarStream'
import type { ApiClient } from '../lib/api'
import type { FrameMessage, StreamClientOptions } from '../lib/ws'
import type { BasketViewState } from '../../../main/transferStream'
import { DEFAULT_POS_CONFIG, type PosConfig } from '../../../main/posConfig'
import { DEFAULT_ZONE_PRESET } from '../../../main/transferGeometry'

function harness(mirrored = false): {
  streamDeps: StreamDeps
  frame: () => void
  status: (state: string, detail: string) => void
} {
  let captured: StreamClientOptions | null = null
  // Every API member the stream hook might call answers with an empty object; this screen only
  // needs the frames, which arrive through the stream client below.
  const api = new Proxy({}, { get: () => vi.fn(async () => ({ events: [] })) }) as ApiClient
  return {
    streamDeps: {
      apiFactory: () => api,
      streamFactory: (opts) => {
        captured = opts
        return { connect: vi.fn(), close: vi.fn() }
      }
    },
    frame: () => {
      const msg: FrameMessage = {
        type: 'frame',
        ts: 1,
        seq: 1,
        jpeg: 'AAAA',
        detections: [],
        stats: { infer_fps: 10, capture_fps: 30, latency_ms: 20, suppressed: 0 },
        fresh: true,
        mirrored
      } as FrameMessage
      act(() => {
        captured!.onOpen?.()
        captured!.onFrame?.(msg)
      })
    },
    status: (state, detail) => {
      act(() => {
        captured!.onStatus?.({ type: 'status', state, detail } as never)
      })
    }
  }
}

function basketState(overrides: Partial<BasketViewState> = {}): BasketViewState {
  return {
    readout: { candidates: 0, itemCount: 0, items: [], review: [], blind: false },
    layout: { mode: 'bands', ...DEFAULT_ZONE_PRESET },
    practice: false,
    customerBound: false,
    ...overrides
  }
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- a test double whose mock types are its own return
function bridgeWith(state: BasketViewState, config: PosConfig = DEFAULT_POS_CONFIG) {
  return {
    getPosConfig: vi.fn(async () => config),
    savePosConfig: vi.fn(async (patch: Partial<PosConfig>) => ({ ...config, ...patch })),
    startBasketPractice: vi.fn(async () => state),
    stopBasketPractice: vi.fn(async () => state),
    resolvePosReview: vi.fn(async () => {})
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('BasketTestView', () => {
  it('draws the saved bands over the preview once a frame arrives', async () => {
    const h = harness()
    const state = basketState()
    render(
      <BasketTestView
        port={8765}
        streamDeps={h.streamDeps}
        basketDeps={{ read: async () => state, subscribe: () => () => {} }}
        bridge={bridgeWith(state)}
      />
    )
    h.frame()
    await waitFor(() => expect(screen.getByTestId('zone-inside')).toBeInTheDocument())
    expect(screen.getByTestId('zone-inside').getAttribute('points')).toBe(
      '0.0000,0.6500 1.0000,0.6500 1.0000,1.0000 0.0000,1.0000'
    )
    expect(screen.getByTestId('zone-opening')).toBeInTheDocument()
  })

  it('stores a click on a mirrored preview un-mirrored, and saves drawn outlines', async () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 100,
      height: 100,
      right: 100,
      bottom: 100,
      x: 0,
      y: 0,
      toJSON: () => ({})
    })
    const h = harness(true)
    const state = basketState()
    const bridge = bridgeWith(state)
    render(
      <BasketTestView
        port={8765}
        streamDeps={h.streamDeps}
        basketDeps={{ read: async () => state, subscribe: () => () => {} }}
        bridge={bridge}
      />
    )
    h.frame()
    await waitFor(() => expect(screen.getByLabelText(/B · Drawn outlines/)).toBeInTheDocument())
    fireEvent.click(screen.getByLabelText(/B · Drawn outlines/))

    // Inside: three clicks on the preview's left side, which is the camera's right side.
    fireEvent.click(screen.getByTestId('bt-draw-inside'))
    const preview = screen.getByTestId('bt-preview')
    for (const [x, y] of [
      [10, 60],
      [40, 60],
      [40, 90]
    ]) {
      fireEvent.click(preview, { clientX: x, clientY: y })
    }
    fireEvent.click(screen.getByTestId('bt-draw-opening'))
    for (const [x, y] of [
      [5, 40],
      [45, 40],
      [45, 58]
    ]) {
      fireEvent.click(preview, { clientX: x, clientY: y })
    }
    expect(screen.getByText('unsaved')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('bt-save-zones'))
    await waitFor(() => expect(bridge.savePosConfig).toHaveBeenCalledTimes(1))
    const patch = bridge.savePosConfig.mock.calls[0][0] as Partial<PosConfig>
    expect(patch.zoneMode).toBe('drawn')
    // x = 0.10 on a mirrored screen is x = 0.90 in the camera's own frame.
    expect(patch.drawnInside?.[0].x).toBeCloseTo(0.9)
    expect(patch.drawnInside?.[0].y).toBeCloseTo(0.6)
    expect(patch.drawnOpening).toHaveLength(3)
  })

  it('starts a practice run, and shows the basket and its review reasons', async () => {
    const h = harness()
    let push: (s: BasketViewState) => void = () => {}
    const idle = basketState()
    const bridge = bridgeWith(idle)
    render(
      <BasketTestView
        port={8765}
        streamDeps={h.streamDeps}
        basketDeps={{
          read: async () => idle,
          subscribe: (cb) => {
            push = cb
            return () => {}
          }
        }}
        bridge={bridge}
      />
    )
    await waitFor(() => expect(screen.getByTestId('bt-practice-start')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('bt-practice-start'))
    expect(bridge.startBasketPractice).toHaveBeenCalledTimes(1)

    act(() =>
      push(
        basketState({
          practice: true,
          readout: {
            candidates: 1,
            itemCount: 2,
            items: [{ className: 'century_tuna', quantity: 2 }],
            review: [
              { id: 'r1', className: null, reason: 'two items crossed the opening at once', at: 3 }
            ],
            blind: false
          }
        })
      )
    )
    expect(screen.getByTestId('bt-practice-on')).toBeInTheDocument()
    expect(screen.getByTestId('bt-count')).toHaveTextContent('2')
    expect(screen.getByTestId('bt-items')).toHaveTextContent('century_tuna')
    expect(screen.getByTestId('bt-review')).toHaveTextContent(
      'two items crossed the opening at once'
    )
  })

  it('says practice is off while a customer session owns the basket', async () => {
    const h = harness()
    const state = basketState({ customerBound: true })
    render(
      <BasketTestView
        port={8765}
        streamDeps={h.streamDeps}
        basketDeps={{ read: async () => state, subscribe: () => () => {} }}
        bridge={bridgeWith(state)}
      />
    )
    await waitFor(() => expect(screen.getByTestId('bt-customer')).toBeInTheDocument())
    expect(screen.queryByTestId('bt-practice-start')).not.toBeInTheDocument()
  })

  it('shows why a capture stopped instead of only "error"', async () => {
    const h = harness()
    const state = basketState()
    render(
      <BasketTestView
        port={8765}
        streamDeps={h.streamDeps}
        basketDeps={{ read: async () => state, subscribe: () => () => {} }}
        bridge={bridgeWith(state)}
      />
    )
    h.frame()
    h.status('error', "Capture stopped: 'NoneType' object has no attribute 'names'")
    expect(screen.getByTestId('bt-error')).toHaveTextContent("Capture stopped: 'NoneType'")
    fireEvent.click(screen.getByText('Dismiss'))
    expect(screen.queryByTestId('bt-error')).not.toBeInTheDocument()
  })

  it('marks a point from the first click, lets it be dragged, and undoes the drag', async () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 100,
      height: 100,
      right: 100,
      bottom: 100,
      x: 0,
      y: 0,
      toJSON: () => ({})
    })
    const h = harness()
    const state = basketState()
    const bridge = bridgeWith(state)
    render(
      <BasketTestView
        port={8765}
        streamDeps={h.streamDeps}
        basketDeps={{ read: async () => state, subscribe: () => () => {} }}
        bridge={bridge}
      />
    )
    h.frame()
    await waitFor(() => expect(screen.getByLabelText(/B · Drawn outlines/)).toBeInTheDocument())
    fireEvent.click(screen.getByLabelText(/B · Drawn outlines/))
    fireEvent.click(screen.getByTestId('bt-draw-inside'))
    const preview = screen.getByTestId('bt-preview')

    // One click draws no outline, so the marker is the only sign the click landed.
    fireEvent.click(preview, { clientX: 20, clientY: 60 })
    const first = screen.getByTestId('bt-handle-inside-0')
    expect(first).toHaveTextContent('1')
    expect(first.style.left).toBe('20%')
    expect(first.style.top).toBe('60%')

    // Clicking a marker selects it for dragging; it must not add a second point beneath it.
    fireEvent.click(first, { clientX: 20, clientY: 60 })
    expect(screen.queryByTestId('bt-handle-inside-1')).not.toBeInTheDocument()

    fireEvent.click(preview, { clientX: 80, clientY: 60 })
    fireEvent.click(preview, { clientX: 80, clientY: 90 })
    expect(screen.getByTestId('bt-handle-inside-2')).toBeInTheDocument()

    // Drag point 1 from (20, 60) to (30, 50): it follows the pointer, and the drag is one step.
    fireEvent.pointerDown(first, { clientX: 20, clientY: 60, pointerId: 1 })
    fireEvent.pointerMove(first, { clientX: 25, clientY: 55, pointerId: 1 })
    fireEvent.pointerMove(first, { clientX: 30, clientY: 50, pointerId: 1 })
    fireEvent.pointerUp(first, { clientX: 30, clientY: 50, pointerId: 1 })
    expect(screen.getByTestId('bt-handle-inside-0').style.left).toBe('30%')
    expect(screen.getByTestId('bt-handle-inside-0').style.top).toBe('50%')

    fireEvent.click(screen.getByTestId('bt-undo'))
    expect(screen.getByTestId('bt-handle-inside-0').style.left).toBe('20%')
    fireEvent.click(screen.getByTestId('bt-redo'))
    expect(screen.getByTestId('bt-handle-inside-0').style.left).toBe('30%')
    // Ctrl+Z is the same step.
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(screen.getByTestId('bt-handle-inside-0').style.left).toBe('20%')
    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    expect(screen.getByTestId('bt-handle-inside-0').style.left).toBe('30%')

    // Delete takes the whole outline away, and Undo brings it back.
    fireEvent.click(screen.getByTestId('bt-delete-inside'))
    expect(screen.queryByTestId('bt-handle-inside-0')).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId('bt-undo'))
    expect(screen.getByTestId('bt-handle-inside-2')).toBeInTheDocument()
  })
})
