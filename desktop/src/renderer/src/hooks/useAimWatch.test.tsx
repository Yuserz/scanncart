import { describe, it, expect, vi, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { AIM_QUIET_S, describeQuietS, useAimWatch } from './useAimWatch'
import type { FrameMessage } from '../lib/ws'

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

afterEach(() => {
  vi.useRealTimers()
})

describe('useAimWatch', () => {
  it('says nothing while the capture is idle, however long it has been boxless', () => {
    vi.useFakeTimers()
    const { result, rerender } = renderHook(
      (props: { running: boolean; frame: FrameMessage | null }) =>
        useAimWatch(props.running, props.frame),
      { initialProps: { running: false, frame: null as FrameMessage | null } }
    )
    act(() => {
      vi.advanceTimersByTime(3 * AIM_QUIET_S * 1000)
    })
    rerender({ running: false, frame: frameWith([]) })
    expect(result.current.warn).toBe(false)
    expect(result.current.quietS).toBe(0)
  })

  it('stays quiet inside the twenty-second window the smoke test grants, then warns', () => {
    vi.useFakeTimers()
    const { result, rerender } = renderHook(
      (props: { running: boolean; frame: FrameMessage | null }) =>
        useAimWatch(props.running, props.frame),
      { initialProps: { running: false, frame: null as FrameMessage | null } }
    )
    rerender({ running: true, frame: frameWith([]) })
    act(() => {
      vi.advanceTimersByTime((AIM_QUIET_S - 1) * 1000)
    })
    expect(result.current.warn).toBe(false)

    act(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(result.current.warn).toBe(true)
    expect(result.current.quietS).toBeGreaterThanOrEqual(AIM_QUIET_S)
  })

  it('never warns once a box has been seen, however long the capture runs', () => {
    vi.useFakeTimers()
    const { result, rerender } = renderHook(
      (props: { running: boolean; frame: FrameMessage | null }) =>
        useAimWatch(props.running, props.frame),
      { initialProps: { running: false, frame: null as FrameMessage | null } }
    )
    rerender({ running: true, frame: frameWith([]) })
    // One box, ever, is the latch: between customers the counter is legitimately empty for
    // minutes, and the banner must survive that rather than nag all day.
    rerender({
      running: true,
      frame: frameWith([{ track_id: 1, cls: 'banana', conf: 0.9, box: [0.1, 0.2, 0.3, 0.4] }])
    })
    act(() => {
      vi.advanceTimersByTime(10 * AIM_QUIET_S * 1000)
    })
    expect(result.current.warn).toBe(false)
  })

  it('clears on the first box and stays cleared afterwards', () => {
    vi.useFakeTimers()
    const { result, rerender } = renderHook(
      (props: { running: boolean; frame: FrameMessage | null }) =>
        useAimWatch(props.running, props.frame),
      { initialProps: { running: false, frame: null as FrameMessage | null } }
    )
    rerender({ running: true, frame: frameWith([]) })
    act(() => {
      vi.advanceTimersByTime((AIM_QUIET_S + 1) * 1000)
    })
    expect(result.current.warn).toBe(true)

    rerender({
      running: true,
      frame: frameWith([{ track_id: 2, cls: 'banana', conf: 0.8, box: [0, 0, 0.5, 0.5] }])
    })
    expect(result.current.warn).toBe(false)
    act(() => {
      vi.advanceTimersByTime(5 * AIM_QUIET_S * 1000)
    })
    expect(result.current.warn).toBe(false)
  })

  it('re-arms on a capture restart: the boxes of the last run say nothing about this one', () => {
    vi.useFakeTimers()
    const { result, rerender } = renderHook(
      (props: { running: boolean; frame: FrameMessage | null }) =>
        useAimWatch(props.running, props.frame),
      { initialProps: { running: false, frame: null as FrameMessage | null } }
    )
    // Run one: verified by a box, so it never warns.
    rerender({ running: true, frame: frameWith([]) })
    rerender({
      running: true,
      frame: frameWith([{ track_id: 1, cls: 'banana', conf: 0.9, box: [0.1, 0.2, 0.3, 0.4] }])
    })
    // Stop and start: a new capture is a new pre-flight.
    rerender({ running: false, frame: null })
    rerender({ running: true, frame: frameWith([]) })
    act(() => {
      vi.advanceTimersByTime((AIM_QUIET_S + 1) * 1000)
    })
    expect(result.current.warn).toBe(true)
  })

  it('checks on a mount that lands on an already-running capture (a reload re-verifies)', () => {
    vi.useFakeTimers()
    // One frame object, as the stream hook holds: the latch keys on message identity, so a
    // fresh object every render would never settle.
    const frame = frameWith([])
    const { result } = renderHook(() => useAimWatch(true, frame))
    act(() => {
      vi.advanceTimersByTime((AIM_QUIET_S + 1) * 1000)
    })
    expect(result.current.warn).toBe(true)
  })
})

describe('describeQuietS', () => {
  it('renders seconds below a minute, minutes+seconds below an hour, hours+minutes above', () => {
    expect(describeQuietS(0)).toBe('0s')
    expect(describeQuietS(47)).toBe('47s')
    expect(describeQuietS(250)).toBe('4m 10s')
    expect(describeQuietS(3900)).toBe('1h 05m')
  })
})
