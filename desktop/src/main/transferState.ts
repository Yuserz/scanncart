// Transfer state machine for basket add/remove detection (CART_TRANSFER_SPEC §Gate-B).
//
// A deterministic, testable core that turns a stream of fresh per-inference detections into
// confirmed transfer events: a deposit (+1) or a removal (−1) of one unit of one product.
// It is wired to the cart only through `basketLedger.ts`, and the ledger only reaches the
// customer's cart when the POS config's `cartMode` is `basket` — spec §Gate-B(4) requires
// shadow-mode validation against human inventory before that.
//
// The spec's rules this implements, and where each lives:
//
// - "Require stable identity across multiple fresh observations and ordered
//   origin/transition/destination evidence. An initial rule is two fresh observations
//   per side plus visible completion" -> `minObservationsPerSide`, `endpointHoldS` and the
//   region-order checks in `observe()`.
// - "Candidate states: observed → inbound/outbound pending → confirmed, aborted or
//   uncertain" -> `TransferPhase`.
// - "Commit only after observed completion, once per physical transfer" -> a confirmed
//   track restarts its history *at its destination* (`settleAt`), so the same direction
//   cannot fire again until the item has genuinely gone back the other way.
// - "Historic maximum confidence or a raw tracker ID alone is insufficient" -> every
//   transition consults only observations recorded now, every one of them — the completing
//   frame included — must clear the scanner's own `conf_threshold`, and a class change or a
//   jump within a region breaks
//   the association (review) rather than being trusted because the ID matched.
// - "Excluded: simultaneous exchanges / bundles" -> a completion while another track was in
//   the opening at the same moment goes to review (`concurrentOpeningS`).
//
// Regions are pure geometry policy: the caller supplies rect regions (outside / opening /
// inside) in true (unmirrored) frame coordinates and this module never touches pixels. A
// box is in a region when its centre is. For v1 the regions come from a preset of bands
// (`transferGeometry.presetRegions`); drawn polygons are a later version that changes only
// how `regionOf` answers.

import { center, pointIn, validateRegions, type Box } from './transferGeometry'

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/** The three regions of the marked loading path. */
export type Region = 'outside' | 'opening' | 'inside'

/** The deterministic states a tracked candidate moves through (spec §Gate-B(3)). */
export type TransferPhase =
  | 'observed' // item seen, direction not yet established
  | 'inbound_pending' // outside → opening seen; waiting on inside evidence + completion
  | 'outbound_pending' // inside → opening seen; waiting on outside evidence + completion
  | 'confirmed_inbound'
  | 'confirmed_outbound'
  | 'aborted' // reversed before completion; drops out without a ledger effect
  | 'uncertain' // ambiguous evidence (identity conflict, incompatible association)

/** One fresh detection observation (never a reused preview box — spec §Gate-B). */
export interface Observation {
  /** Monotonic frame/observation sequence number (the sidecar's, or a local counter). */
  seq: number
  /** Observation time in seconds (same clock as everything else in the desktop). */
  t: number
  /** Tracker identity from the sidecar. */
  trackId: number
  className: string
  conf: number
  /** Detection box in true (unmirrored) frame coordinates. */
  box: Box
}

/** A completed transfer the state machine is prepared to vouch for. */
export interface TransferEvent {
  kind: 'inbound' | 'outbound'
  className: string
  /** Confidence of the observation that completed the transfer (fresh, not historic max). */
  completionConf: number
  /** Completion time (the observation that satisfied the endpoint rule). */
  completedAt: number
  /** The tracker id the candidate travelled under (diagnostics, not identity proof). */
  trackId: number
}

export interface TransferStateConfig {
  /**
   * Spec §Gate-B(2): "two fresh observations per side plus visible completion". Each
   * side of the path (origin and destination) must see the candidate this many times.
   */
  minObservationsPerSide: number
  /**
   * Seconds a candidate may sit without a fresh observation before it is dropped.
   * A candidate that vanishes mid-path becomes `uncertain`, never `confirmed`.
   */
  candidateTimeoutS: number
  /**
   * The scanner's `conf_threshold` (the sidecar setting the operator tunes): an observation
   * below it is no evidence at all, on any step of the path, the completing one included.
   * Deliberately one number with one owner rather than a second threshold here — set it to
   * 0.9 on the scanner and the basket infers nothing under 0.9. The sidecar already drops
   * detections below it; this is the same cut applied again so the rule holds by itself
   * (a replay, an older sidecar), and it is kept in step at runtime with `setConfThreshold`.
   */
  confThreshold: number
  /**
   * Maximum box displacement between consecutive observations *within one region*
   * before the association is judged broken (in the caller's coordinate units).
   */
  maxSideSlop: number
  /**
   * Spec §2's endpoint hold: after reaching the destination the candidate must stay
   * there, still observed, for this many seconds before the transfer completes. This is
   * the release proxy for a deposit (the item rests in the cart band) and the clearance
   * proxy for a removal (the item stays out of it). 0 completes on arrival.
   */
  endpointHoldS: number
  /**
   * Two different tracks in the opening within this many seconds of each other are a
   * simultaneous exchange or bundle — excluded from automatic commits (spec §2), so the
   * completing candidate goes to review instead.
   */
  concurrentOpeningS: number
}

export const DEFAULT_TRANSFER_CONFIG: TransferStateConfig = {
  minObservationsPerSide: 2,
  candidateTimeoutS: 5,
  // The sidecar's default `conf_threshold` (`settings.py`), until the live value is read.
  confThreshold: 0.5,
  maxSideSlop: 150,
  endpointHoldS: 1.0,
  concurrentOpeningS: 1.0
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

export interface Regions {
  outside: Box
  opening: Box
  inside: Box
}

export function regionOf(regions: Regions, box: Box): Region {
  const c = center(box)
  if (pointIn(c, regions.inside)) return 'inside'
  if (pointIn(c, regions.opening)) return 'opening'
  // Anything not in the cart band or the opening is outside the basket.
  return 'outside'
}

// ---------------------------------------------------------------------------
// Per-candidate machine
// ---------------------------------------------------------------------------

/** Everything the machine reports about one in-flight candidate. */
export interface Candidate {
  trackId: number
  className: string
  phase: TransferPhase
  /** Fresh observations seen in the origin region (outside for inbound, inside for outbound). */
  originObs: number
  /** Fresh observations seen in the destination region since arriving there. */
  destObs: number
  /** The last box seen, for association and drift checks. */
  lastBox: Box
  lastSeq: number
  lastT: number
}

export interface Removal {
  phase: TransferPhase
  /** The spec requires a human look; the ledger records it as pending review. */
  review: boolean
  reason: string
  className: string
  trackId: number
}

export interface MachineOutput {
  /** Events completed on this observation (0 or 1; "once per physical transfer"). */
  events: TransferEvent[]
  /** Candidate state after the observation, for diagnostics and shadow-mode readouts. */
  candidate?: Candidate
  /**
   * The candidate was dropped (aborted or uncertain). This module never touches the
   * ledger; when `review` is set the ledger holds it as an unresolved interaction.
   */
  removed?: Removal
}

const REGION_ORDER_IN: Region[] = ['outside', 'opening', 'inside']
const REGION_ORDER_OUT: Region[] = ['inside', 'opening', 'outside']

/**
 * Classify the visited-region sequence against one direction. `incomplete` when the path
 * has not started from that direction's origin, or has started but not reached the end;
 * `reversed` when it started and then went back; `ordered` when it reached the end in order.
 */
export function classifyVisit(
  visited: Region[],
  direction: 'inbound' | 'outbound'
): 'ordered' | 'reversed' | 'incomplete' {
  const expected = direction === 'inbound' ? REGION_ORDER_IN : REGION_ORDER_OUT
  if (visited.length === 0 || visited[0] !== expected[0]) return 'incomplete'
  let matched = 0
  for (const r of visited) {
    if (matched < expected.length && r === expected[matched]) matched += 1
    else if (r === expected[Math.max(0, matched - 1)])
      continue // same region again
    else return 'reversed'
  }
  return matched === expected.length ? 'ordered' : 'incomplete'
}

interface TrackHistory {
  visited: Region[]
  counts: Record<Region, number>
  /** When the candidate arrived in its direction's destination (endpoint hold). */
  destSince: number | null
  /** When this track was last observed in the opening (concurrent-exchange check). */
  lastOpeningT: number | null
  /** Last time the track was observed at all, for forgetting abandoned tracks. */
  lastT: number
}

function historyAt(region: Region, count: number, t: number): TrackHistory {
  const counts: Record<Region, number> = { outside: 0, opening: 0, inside: 0 }
  counts[region] = count
  return { visited: [region], counts, destSince: null, lastOpeningT: null, lastT: t }
}

type Expired = { trackId: number; className: string; phase: TransferPhase; review: boolean }

// ---------------------------------------------------------------------------
// The machine
// ---------------------------------------------------------------------------

export class TransferStateMachine {
  private candidates = new Map<number, Candidate>()
  private history = new Map<number, TrackHistory>()
  private regions: Regions

  private cfg: TransferStateConfig

  constructor(regions: Regions, cfg: TransferStateConfig = DEFAULT_TRANSFER_CONFIG) {
    this.regions = TransferStateMachine.checked(regions)
    // A copy, so `setConfThreshold` can never write through to the shared default.
    this.cfg = { ...cfg }
  }

  /** Follow the scanner's `conf_threshold` when the operator changes it (hot-reloadable). */
  setConfThreshold(value: number): void {
    if (Number.isFinite(value) && value >= 0 && value <= 1) this.cfg.confThreshold = value
  }

  getConfThreshold(): number {
    return this.cfg.confThreshold
  }

  private static checked(regions: Regions): Regions {
    const problems = validateRegions(regions)
    if (problems.length > 0) throw new Error(`invalid regions: ${problems.join('; ')}`)
    return regions
  }

  /** Update regions (e.g. after a recalibration) without losing candidates. */
  setRegions(regions: Regions): void {
    this.regions = TransferStateMachine.checked(regions)
  }

  getRegions(): Regions {
    return this.regions
  }

  /** Candidates currently tracked, for shadow-mode readouts. */
  snapshot(): Candidate[] {
    return [...this.candidates.values()]
  }

  /** Another track seen in the opening close to `t` — a simultaneous exchange. */
  private otherInOpening(trackId: number, t: number): boolean {
    for (const [id, h] of this.history) {
      if (id === trackId || h.lastOpeningT === null) continue
      if (Math.abs(t - h.lastOpeningT) <= this.cfg.concurrentOpeningS) return true
    }
    return false
  }

  private remember(
    obs: Observation,
    phase: TransferPhase,
    originObs: number,
    destObs: number
  ): Candidate {
    const cand: Candidate = {
      trackId: obs.trackId,
      className: obs.className,
      phase,
      originObs,
      destObs,
      lastBox: obs.box,
      lastSeq: obs.seq,
      lastT: obs.t
    }
    this.candidates.set(obs.trackId, cand)
    return cand
  }

  /**
   * Feed one fresh observation. Returns the event it completed (at most one) and the
   * candidate's resulting state, or the reason it was dropped.
   */
  observe(obs: Observation): MachineOutput {
    // Below the scanner's own conf_threshold is no evidence at all.
    if (obs.conf < this.cfg.confThreshold) return { events: [] }

    const region = regionOf(this.regions, obs.box)
    const existing = this.candidates.get(obs.trackId)
    const drop = (
      phase: TransferPhase,
      review: boolean,
      reason: string,
      restart: boolean
    ): MachineOutput => {
      this.candidates.delete(obs.trackId)
      if (restart) this.history.set(obs.trackId, historyAt(region, 1, obs.t))
      else this.history.delete(obs.trackId)
      return {
        events: [],
        removed: {
          phase,
          review,
          reason,
          className: existing?.className ?? obs.className,
          trackId: obs.trackId
        }
      }
    }

    // Identity quality first. A class change is an identity conflict no matter what; a box
    // displacement only is when it happens without leaving the region — crossing the
    // opening legitimately covers a large distance between observations.
    if (existing) {
      const drift =
        Math.abs(existing.lastBox.x - obs.box.x) + Math.abs(existing.lastBox.y - obs.box.y)
      if (obs.className !== existing.className) {
        return drop(
          'uncertain',
          existing.phase !== 'observed' || region === 'opening',
          `class changed ${existing.className} → ${obs.className} mid-path`,
          false
        )
      }
      if (region === regionOf(this.regions, existing.lastBox) && drift > this.cfg.maxSideSlop) {
        return drop('uncertain', true, `box jumped ${Math.round(drift)}px within ${region}`, false)
      }
    }

    const h = this.history.get(obs.trackId) ?? historyAt(region, 0, obs.t)
    this.history.set(obs.trackId, h)
    h.lastT = obs.t
    if (h.visited[h.visited.length - 1] !== region) h.visited.push(region)
    // Counted over observations, not region changes: two sightings in the origin count even
    // when the candidate never left that region between them.
    h.counts[region] += 1
    if (region === 'opening') h.lastOpeningT = obs.t

    // First seen mid-path: there is no origin evidence, so nothing it does can commit. When
    // it lands on a side it is re-anchored there (it is now an item on that side), and the
    // landing is reported for review — it may be a deposit or removal the camera half-saw.
    if (h.visited[0] === 'opening' && region !== 'opening') {
      const phase = existing?.phase ?? 'observed'
      this.candidates.delete(obs.trackId)
      this.history.set(obs.trackId, historyAt(region, 1, obs.t))
      return {
        events: [],
        removed: {
          phase,
          review: true,
          reason: `appeared in the opening and moved ${region} without origin evidence`,
          className: obs.className,
          trackId: obs.trackId
        }
      }
    }

    const inboundVisit = classifyVisit(h.visited, 'inbound')
    const outboundVisit = classifyVisit(h.visited, 'outbound')

    // A reversal of a path that had started: the item went back the way it came. Not a
    // transfer, and not worth a human look — it restarts from where the item is now, so a
    // hover that went back outside can still be deposited properly on the next attempt.
    if (inboundVisit === 'reversed' || outboundVisit === 'reversed') {
      return drop('aborted', false, `reversal after ${h.visited.join('→')}`, true)
    }

    const direction: 'inbound' | 'outbound' = h.visited[0] === 'inside' ? 'outbound' : 'inbound'
    const visit = direction === 'inbound' ? inboundVisit : outboundVisit
    const originRegion: Region = direction === 'inbound' ? 'outside' : 'inside'
    const destRegion: Region = direction === 'inbound' ? 'inside' : 'outside'
    const originObs = h.counts[originRegion]
    const arrived = visit === 'ordered' && region === destRegion
    if (arrived && h.destSince === null) h.destSince = obs.t
    const destObs = arrived ? h.counts[destRegion] : 0

    const originOk = originObs >= this.cfg.minObservationsPerSide
    // A small tolerance so a hold sampled at exactly the interval is not lost to float error.
    const held = h.destSince !== null && obs.t - h.destSince >= this.cfg.endpointHoldS - 1e-6
    const pendingPhase: TransferPhase =
      direction === 'inbound' ? 'inbound_pending' : 'outbound_pending'

    if (arrived && originOk && destObs >= this.cfg.minObservationsPerSide && held) {
      if (this.otherInOpening(obs.trackId, h.lastOpeningT ?? obs.t)) {
        return drop('uncertain', true, 'two items crossed the opening at once', true)
      }
      // Restart the history at the destination: the item now rests there and has been seen
      // there enough to be a valid origin, which is what lets "remove, then put back" work on
      // a track the tracker keeps following — and the same direction cannot re-fire, because
      // its origin is the side the item has just left.
      this.candidates.delete(obs.trackId)
      this.history.set(obs.trackId, historyAt(destRegion, destObs, obs.t))
      return {
        events: [
          {
            kind: direction,
            className: obs.className,
            completionConf: obs.conf,
            completedAt: obs.t,
            trackId: obs.trackId
          }
        ]
      }
    }

    const phase: TransferPhase = h.visited.length > 1 ? pendingPhase : 'observed'
    return { events: [], candidate: this.remember(obs, phase, originObs, destObs) }
  }

  /**
   * Time-based sweep: expire candidates that stopped producing fresh observations and
   * report the ones that died mid-path (the spec's `uncertain` family — a vanishing
   * candidate is *not* evidence of completion). Call periodically with `now`.
   */
  sweep(now: number): Expired[] {
    const expired: Expired[] = []
    for (const [trackId, cand] of this.candidates) {
      if (now - cand.lastT > this.cfg.candidateTimeoutS) {
        this.candidates.delete(trackId)
        this.history.delete(trackId)
        expired.push({
          trackId,
          className: cand.className,
          phase: cand.phase,
          review: cand.phase !== 'observed'
        })
      }
    }
    // A settled track the tracker abandoned stops vouching for its last region.
    for (const [trackId, h] of this.history) {
      if (!this.candidates.has(trackId) && now - h.lastT > this.cfg.candidateTimeoutS)
        this.history.delete(trackId)
    }
    return expired
  }

  /** Forgets all in-flight state (capture restart, session rebind). */
  reset(): void {
    this.candidates.clear()
    this.history.clear()
  }
}
