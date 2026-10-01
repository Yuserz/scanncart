// Derive what is on the counter now from the sidecar's track log (spec §5.2).
//
// Quantity is the number of a class's tracks *open at the same time*, not the number of tracks:
// the tracker routinely ends one track and starts another for the same physical item (a hand
// passes over it, the customer lifts it and puts it back, detection drops for longer than
// `track_expiry_s`). Counting tracks over a window would read that one item as two every time.
//
// All timestamps are the sidecar's `time.time()` seconds; the desktop reads `now` from the same
// machine's clock, so the two are directly comparable.

export interface TrackEvent {
  /** The sidecar session the track belongs to, so events from a restart can be passed together. */
  sessionId: number
  trackId: number
  className: string
  maxConf: number
  enteredAt: number
  leftAt: number | null
}

export interface CartStateConfig {
  commitDwellS: number
  removeSettleS: number
  minCommitConf: number
}

export interface CartEntry {
  quantity: number
  maxConfidence: number
}

interface Interval {
  start: number
  end: number
  maxConf: number
}

/** Number of half-open intervals [start, end) covering `t`. */
function countAt(intervals: Interval[], t: number): number {
  let count = 0
  for (const interval of intervals) {
    if (interval.start <= t && t < interval.end) count += 1
  }
  return count
}

/**
 * The class's committed intervals: a track counts over `[entered_at + commitDwellS, end]`, where
 * `end` is `left_at`, or `now` while the track is still open. A track whose end comes before its
 * commit time never counts — that is the flicker filter.
 */
function committedIntervals(
  events: TrackEvent[],
  bindAt: number,
  cfg: CartStateConfig
): Interval[] {
  const intervals: Interval[] = []
  for (const event of events) {
    if (event.enteredAt < bindAt) continue
    if (event.maxConf < cfg.minCommitConf) continue
    const start = event.enteredAt + cfg.commitDwellS
    // An open track counts from its commit time onwards; `now` only bounds the settle window.
    const end = event.leftAt ?? Infinity
    if (end < start) continue
    intervals.push({ start, end, maxConf: event.maxConf })
  }
  return intervals
}

/**
 * The largest committed count over `[now - removeSettleS, now]`. The step function only changes at
 * an interval's start or end, so the maximum is attained at the window start, just after a start,
 * or just before an end.
 */
function settledQuantity(intervals: Interval[], now: number, removeSettleS: number): number {
  const windowStart = now - removeSettleS
  let best = countAt(intervals, windowStart)
  best = Math.max(best, countAt(intervals, now))

  for (const interval of intervals) {
    if (interval.start > windowStart && interval.start <= now) {
      best = Math.max(best, countAt(intervals, interval.start))
    }
    // Just before the interval ends the count is still one higher; at the exact end it is not.
    if (interval.end > windowStart && interval.end <= now) {
      best = Math.max(best, countAt(intervals, interval.end - 1e-9))
    }
  }

  return best
}

/**
 * `deriveCartState(events, now, bindAt, cfg)` — the map of class -> {quantity, maxConfidence} for
 * everything committed to the counter since the session bound. Classes with quantity 0 are absent,
 * so an empty map means "nothing on the counter".
 */
export function deriveCartState(
  events: TrackEvent[],
  now: number,
  bindAt: number,
  cfg: CartStateConfig
): Map<string, CartEntry> {
  const byClass = new Map<string, TrackEvent[]>()
  for (const event of events) {
    const list = byClass.get(event.className)
    if (list) list.push(event)
    else byClass.set(event.className, [event])
  }

  const out = new Map<string, CartEntry>()

  for (const [className, classEvents] of byClass) {
    const intervals = committedIntervals(classEvents, bindAt, cfg)
    if (intervals.length === 0) continue

    const quantity = settledQuantity(intervals, now, cfg.removeSettleS)
    if (quantity <= 0) continue

    // The confidence of what still counts: the highest max_conf among intervals overlapping the
    // settle window (the same tracks the quantity is derived from).
    const windowStart = now - cfg.removeSettleS
    let maxConfidence = 0
    for (const interval of intervals) {
      if (interval.start <= now && interval.end > windowStart) {
        maxConfidence = Math.max(maxConfidence, interval.maxConf)
      }
    }

    out.set(className, { quantity, maxConfidence })
  }

  return out
}
