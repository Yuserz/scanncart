// Pure geometry helpers for the transfer regions (CART_TRANSFER_SPEC §2).
//
// Split from transferState.ts so the state machine stays about *evidence* and this file
// stays about *pixels*: boxes, containment, and the calibration contract. Everything
// here is pure and total — no camera, no settings store, no clock.

/** A rectangle in true (unmirrored) frame coordinates, in pixels. */
export interface Box {
  /** Left edge. */
  x: number
  /** Top edge. */
  y: number
  /** Width (positive). */
  w: number
  /** Height (positive). */
  h: number
}

export interface Point {
  x: number
  y: number
}

/** True when `p` lies inside (inclusive edges) the rect `r`. */
export function pointIn(p: Point, r: Box): boolean {
  return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h
}

/** Centre point of a box. */
export function center(b: Box): Point {
  return { x: b.x + b.w / 2, y: b.y + b.h / 2 }
}

/**
 * Smallest axis-aligned rect containing both `a` and `b` — the "did the box jump?"
 * measure in the state machine is the Manhattan distance between centres, which this
 * module does not own; it only supplies the boxes.
 */
export function union(a: Box, b: Box): Box {
  const x = Math.min(a.x, b.x)
  const y = Math.min(a.y, b.y)
  return {
    x,
    y,
    w: Math.max(a.x + a.w, b.x + b.w) - x,
    h: Math.max(a.y + a.h, b.y + b.h) - y
  }
}

/** Fraction of `inner`'s area covered by `outer` (0..1). */
export function coverageFraction(inner: Box, outer: Box): number {
  const ix = Math.max(inner.x, outer.x)
  const iy = Math.max(inner.y, outer.y)
  const ix2 = Math.min(inner.x + inner.w, outer.x + outer.w)
  const iy2 = Math.min(inner.y + inner.h, outer.y + outer.h)
  const iw = Math.max(0, ix2 - ix)
  const ih = Math.max(0, iy2 - iy)
  const innerArea = inner.w * inner.h
  if (innerArea <= 0) return 0
  return (iw * ih) / innerArea
}

/**
 * Validate a calibrated region set before the machine trusts it. Returns a list of
 * problems; an empty list means the regions are usable. Overlap between regions is
 * *allowed* (the opening naturally overlaps both sides on a real cart) but zero-size
 * or inverted regions are not.
 */
export function validateRegions(
  regions: Record<string, Box> | { outside: Box; opening: Box; inside: Box }
): string[] {
  const problems: string[] = []
  for (const [name, r] of Object.entries(regions)) {
    if (
      !Number.isFinite(r.x) ||
      !Number.isFinite(r.y) ||
      !Number.isFinite(r.w) ||
      !Number.isFinite(r.h)
    ) {
      problems.push(`${name}: non-finite coordinate`)
      continue
    }
    if (r.w <= 0 || r.h <= 0) problems.push(`${name}: width/height must be positive`)
  }
  return problems
}

// ---------------------------------------------------------------------------
// The v1 zone preset: bands instead of drawn polygons
// ---------------------------------------------------------------------------

/**
 * The coordinate frame the transfer machine works in: the sidecar's 0–1 boxes scaled to
 * 1000 units a side, so slop and region numbers read as tenths of a percent of the frame
 * whatever the capture resolution is.
 */
export const TRANSFER_FRAME = 1000

/** Which edge of the frame the cart is at. A deposit moves toward it, a removal away. */
export type CartEdge = 'bottom' | 'top' | 'left' | 'right'
export const CART_EDGES: readonly CartEdge[] = ['bottom', 'top', 'left', 'right']

export interface ZonePreset {
  cartEdge: CartEdge
  /** Share of the frame, measured from the cart edge, that is inside the cart (0–1). */
  insideFraction: number
  /** Share of the frame just past the inside band that is the opening (0–1). */
  openingFraction: number
}

export const DEFAULT_ZONE_PRESET: ZonePreset = {
  cartEdge: 'bottom',
  insideFraction: 0.35,
  openingFraction: 0.2
}

/** Problems with a preset; empty means usable. */
export function zonePresetProblems(p: ZonePreset): string[] {
  const problems: string[] = []
  if (!CART_EDGES.includes(p.cartEdge))
    problems.push(`cartEdge must be one of ${CART_EDGES.join(', ')}`)
  for (const [name, v] of [
    ['insideFraction', p.insideFraction],
    ['openingFraction', p.openingFraction]
  ] as const) {
    if (!Number.isFinite(v) || v <= 0 || v >= 1) problems.push(`${name} must be between 0 and 1`)
  }
  if (p.insideFraction + p.openingFraction >= 1) {
    problems.push('insideFraction + openingFraction must leave room for the outside band')
  }
  return problems
}

/**
 * The three bands for a preset, in `TRANSFER_FRAME` units and true (unmirrored)
 * orientation. With the default, the bottom 35% is the cart, the next 20% up is the
 * opening and the rest is outside: an item coming in from the top or a side and moving
 * down into the bottom band is a deposit; the reverse is a removal.
 */
export function presetRegions(p: ZonePreset = DEFAULT_ZONE_PRESET): {
  outside: Box
  opening: Box
  inside: Box
} {
  const problems = zonePresetProblems(p)
  if (problems.length > 0) throw new Error(`invalid zone preset: ${problems.join('; ')}`)
  const F = TRANSFER_FRAME
  const a = p.insideFraction * F
  const b = p.openingFraction * F
  const rest = F - a - b
  switch (p.cartEdge) {
    case 'bottom':
      return {
        inside: { x: 0, y: F - a, w: F, h: a },
        opening: { x: 0, y: F - a - b, w: F, h: b },
        outside: { x: 0, y: 0, w: F, h: rest }
      }
    case 'top':
      return {
        inside: { x: 0, y: 0, w: F, h: a },
        opening: { x: 0, y: a, w: F, h: b },
        outside: { x: 0, y: a + b, w: F, h: rest }
      }
    case 'left':
      return {
        inside: { x: 0, y: 0, w: a, h: F },
        opening: { x: a, y: 0, w: b, h: F },
        outside: { x: a + b, y: 0, w: rest, h: F }
      }
    case 'right':
      return {
        inside: { x: F - a, y: 0, w: a, h: F },
        opening: { x: F - a - b, y: 0, w: b, h: F },
        outside: { x: 0, y: 0, w: rest, h: F }
      }
  }
}

/**
 * A sidecar box — normalized `(x1, y1, x2, y2)` — as a `Box` in `TRANSFER_FRAME` units and
 * true orientation. `mirrored` undoes the preview reflection with the sidecar's own rule
 * (`pipeline.mirrored_detections`: `(1 - x2, y1, 1 - x1, y2)`), which is its own inverse.
 */
export function frameBox(box: readonly [number, number, number, number], mirrored: boolean): Box {
  const [y1, y2] = [box[1], box[3]]
  const [x1, x2] = mirrored ? [1 - box[2], 1 - box[0]] : [box[0], box[2]]
  const F = TRANSFER_FRAME
  return { x: x1 * F, y: y1 * F, w: Math.max(0, x2 - x1) * F, h: Math.max(0, y2 - y1) * F }
}
