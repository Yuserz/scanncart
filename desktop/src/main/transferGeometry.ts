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

/** A closed outline, its vertices in order (the last joins the first). */
export interface Polygon {
  points: Point[]
}

/**
 * One region's shape: a band from a preset (`Box`) or an outline someone drew (`Polygon`). The
 * state machine asks only "is this point in it", so the two are interchangeable to everything
 * past `pointInZone`.
 */
export type Zone = Box | Polygon

export function isPolygon(zone: Zone): zone is Polygon {
  return Array.isArray((zone as Polygon).points)
}

/** True when `p` lies inside (inclusive edges) the rect `r`. */
export function pointIn(p: Point, r: Box): boolean {
  return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h
}

/** Even-odd ray cast: true when `p` lies inside the outline `pts`. */
export function pointInPolygon(p: Point, pts: readonly Point[]): boolean {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i]
    const b = pts[j]
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside
    }
  }
  return inside
}

/** True when `p` lies in the zone, whichever shape it has. */
export function pointInZone(p: Point, zone: Zone): boolean {
  return isPolygon(zone) ? pointInPolygon(p, zone.points) : pointIn(p, zone)
}

/** Area enclosed by an outline (shoelace), always positive. */
export function polygonArea(pts: readonly Point[]): number {
  let twice = 0
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    twice += (pts[j].x + pts[i].x) * (pts[j].y - pts[i].y)
  }
  return Math.abs(twice) / 2
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
  regions: Record<string, Zone> | { outside: Zone; opening: Zone; inside: Zone }
): string[] {
  const problems: string[] = []
  for (const [name, r] of Object.entries(regions)) {
    if (isPolygon(r)) {
      if (r.points.length < 3) problems.push(`${name}: an outline needs at least 3 points`)
      else if (r.points.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y))) {
        problems.push(`${name}: non-finite coordinate`)
      } else if (polygonArea(r.points) <= 0) problems.push(`${name}: the outline encloses no area`)
      continue
    }
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

// ---------------------------------------------------------------------------
// Drawn zones: outlines instead of bands
// ---------------------------------------------------------------------------

/**
 * How the zones are laid out. `bands` (spot A in the cart layout: the camera under the handle,
 * looking *across* the basket) is the default, and a preset of three bands from one edge
 * describes that picture. `drawn` (spot B: a camera above the basket looking *down* into it) is
 * for a picture where the rim is a ring rather than a line, so the inside and the opening are
 * outlines someone draws on the preview; everything outside both is "outside".
 */
export type ZoneMode = 'bands' | 'drawn'
export const ZONE_MODES: readonly ZoneMode[] = ['bands', 'drawn']

/**
 * The two outlines of a drawn layout, as normalized 0–1 points in **true** (unmirrored) frame
 * orientation — the same frame the sidecar's boxes are in once `frameBox` undoes the preview
 * mirror. Normalized rather than in `TRANSFER_FRAME` units so the stored config means the same
 * thing whatever the capture resolution or the frame unit becomes.
 */
export interface DrawnZones {
  inside: Point[]
  opening: Point[]
}

/** A zone layout: a band preset (spot A) or drawn outlines (spot B). */
export type ZoneLayout = ({ mode: 'bands' } & ZonePreset) | ({ mode: 'drawn' } & DrawnZones)

/**
 * An outline smaller than this share of the frame is almost certainly a stray click rather than a
 * basket: a box's centre would land in it by accident or never at all.
 */
export const MIN_DRAWN_AREA = 0.005

/** Problems with drawn outlines; empty means usable. */
export function drawnZoneProblems(d: DrawnZones): string[] {
  const problems: string[] = []
  for (const [name, pts] of [
    ['inside', d.inside],
    ['opening', d.opening]
  ] as const) {
    if (!Array.isArray(pts) || pts.length < 3) {
      problems.push(`the ${name} outline needs at least 3 points`)
      continue
    }
    if (pts.some((p) => !Number.isFinite(p?.x) || !Number.isFinite(p?.y))) {
      problems.push(`the ${name} outline has a point that is not a number`)
      continue
    }
    if (pts.some((p) => p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) {
      problems.push(`the ${name} outline has a point outside the picture`)
      continue
    }
    if (polygonArea(pts) < MIN_DRAWN_AREA) {
      problems.push(`the ${name} outline is too small to be a basket region`)
    }
  }
  return problems
}

function scaled(points: readonly Point[]): Polygon {
  return { points: points.map((p) => ({ x: p.x * TRANSFER_FRAME, y: p.y * TRANSFER_FRAME })) }
}

/**
 * The regions for a layout, in `TRANSFER_FRAME` units. A drawn layout's "outside" is the whole
 * frame: `regionOf` asks inside first and opening second, so outside is simply everything left.
 */
export function layoutRegions(layout: ZoneLayout): { outside: Zone; opening: Zone; inside: Zone } {
  if (layout.mode === 'bands') return presetRegions(layout)
  const problems = drawnZoneProblems(layout)
  if (problems.length > 0) throw new Error(`invalid drawn zones: ${problems.join('; ')}`)
  return {
    inside: scaled(layout.inside),
    opening: scaled(layout.opening),
    outside: { x: 0, y: 0, w: TRANSFER_FRAME, h: TRANSFER_FRAME }
  }
}

/** Problems with any layout; empty means usable. */
export function layoutProblems(layout: ZoneLayout): string[] {
  return layout.mode === 'bands' ? zonePresetProblems(layout) : drawnZoneProblems(layout)
}

/**
 * A stable identity for a layout: equal layouts give equal keys, so a save that changed nothing
 * does not reset the machine, and the ledger's evidence trail names the geometry it judged under.
 */
export function layoutKey(layout: ZoneLayout): string {
  if (layout.mode === 'bands') {
    return `preset:${layout.cartEdge}:${layout.insideFraction}:${layout.openingFraction}`
  }
  const pts = (ps: Point[]): string =>
    ps.map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`).join(';')
  return `drawn:${pts(layout.inside)}|${pts(layout.opening)}`
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
