import { useEffect, useState } from 'react'
import type { FrameMessage } from '../lib/ws'

/**
 * The Live view's aim pre-flight: a capture that has produced no box at all cannot tell an empty
 * counter from a camera pointed at a curtain — both are identical to every filter this app runs,
 * because they are identical to the detector. What the app can do is refuse to let the ambiguity
 * pass silently: after a boxless window, `warn` says so, and the only thing that clears it is
 * evidence — the first box this capture produces.
 *
 * The window is twenty seconds, because that is the norm docs/POS_SMOKE_TEST.md §0 already
 * grants an empty counter ("Twenty seconds with nothing placed leaves the item log empty"): the
 * warning must not fire inside the ritual the doc describes, only once the ritual has had its
 * chance and the operator is looking at a stream that has never once been right.
 *
 * The latch, the arm and the ticker are three separate decisions:
 *
 * - **The latch** (`sawBox`) is deliberately one-way for the life of a capture. Between
 *   customers the counter is legitimately empty for minutes; the banner must survive that, or it
 *   would nag all day and train the operator to ignore it. It resets only when the capture does.
 * - **The arm** runs on every capture transition, including a mount that lands on an
 *   already-running capture (a reload re-checks: the fresh window knows nothing about the boxes
 *   an earlier one saw, and pretending otherwise would let a curtain hide behind a reload).
 * - **The ticker** runs only while the answer can still change — boxless and running — so a
 *   verified or stopped stream schedules nothing.
 */
export const AIM_QUIET_S = 20

export interface AimWatch {
  /** True only while a capture runs and has never yet produced a box past the quiet window. */
  warn: boolean
  /** Seconds since this capture (re)started — the number the banner quotes. */
  quietS: number
}

export function useAimWatch(running: boolean, frame: FrameMessage | null): AimWatch {
  // Seconds since this capture (re)started. A counter the interval below owns rather than a
  // timestamp read off the clock, so the render stays pure: re-renders (StrictMode's included)
  // replay the same adjustments against the same state and cost nothing.
  const [quietS, setQuietS] = useState(0)
  const [sawBox, setSawBox] = useState(false)
  // The two previous-value slots the render-time adjustments below compare against — one per
  // input whose *change* carries meaning (the capture's lifecycle, the newest frame).
  const [prevRunning, setPrevRunning] = useState(running)
  const [lastFrame, setLastFrame] = useState<FrameMessage | null>(frame)

  // Arm on a capture transition and latch on a boxy frame — both during render, which React
  // documents for exactly this case ("storing information from previous renders") and this view
  // already uses for the overlay hold. Order matters inside the body: the arm runs first, so a
  // boxy frame arriving in the same render as a restart latches onto the NEW run instead of
  // being erased by it, while the identity guard keeps the old run's last frame from
  // re-latching after a restart. Mounting onto a running capture arms at zero the same way —
  // the initial states above are that mount's arm.
  if (prevRunning !== running) {
    setPrevRunning(running)
    setQuietS(0)
    setSawBox(false)
  }
  if (frame !== lastFrame) {
    setLastFrame(frame)
    if (frame?.detections.length) setSawBox(true)
  }

  const boxless = running && !sawBox
  useEffect(() => {
    if (!boxless) return
    const timer = setInterval(() => setQuietS((s) => s + 1), 1000)
    return () => clearInterval(timer)
  }, [boxless])

  return {
    warn: boxless && quietS >= AIM_QUIET_S,
    quietS
  }
}

/** The banner's one number, human-sized: `47s`, `4m 10s`, `1h 05m`. */
export function describeQuietS(totalS: number): string {
  const h = Math.floor(totalS / 3600)
  const m = Math.floor((totalS % 3600) / 60)
  const s = totalS % 60
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`
  return `${s}s`
}
