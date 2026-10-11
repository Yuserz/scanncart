// Drawing the basket's zones over the live preview, and turning clicks on it into outline points.
//
// The zones live in **true** frame orientation (the sidecar's boxes, unmirrored) and the preview is
// usually mirrored (`preview_mirror`, on by default). So every point crosses the mirror twice: once
// on the way to the screen (`toScreen`) and once on the way back from a click (`fromScreen`). The
// rule is the sidecar's own (`x -> 1 - x`), which is its own inverse, and it lives in one place here
// so a drawn outline cannot land reflected from where it was drawn.
//
// The geometry itself is the main process's (`transferGeometry.ts`), imported rather than restated:
// a band layout drawn here is exactly the regions the basket judges under.

import {
  presetRegions,
  TRANSFER_FRAME,
  type CartEdge,
  type Point,
  type ZoneLayout
} from '../../../main/transferGeometry'
import type { PosConfig } from '../../../main/posConfig'

/** A layout's two outlines, normalized 0–1, true orientation. */
export interface ZoneOutlines {
  inside: Point[]
  opening: Point[]
}

function rect(box: { x: number; y: number; w: number; h: number }): Point[] {
  const F = TRANSFER_FRAME
  const x1 = box.x / F
  const y1 = box.y / F
  const x2 = (box.x + box.w) / F
  const y2 = (box.y + box.h) / F
  return [
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 }
  ]
}

/**
 * The outlines a layout describes. A band layout becomes two rectangles from the same
 * `presetRegions` the basket uses; a drawn one is its points as stored. An invalid band preset
 * yields no outlines rather than throwing, so a half-typed fraction never blanks the screen.
 */
export function layoutOutlines(layout: ZoneLayout): ZoneOutlines {
  if (layout.mode === 'drawn') return { inside: layout.inside, opening: layout.opening }
  try {
    const r = presetRegions(layout)
    return {
      inside: rect(r.inside as { x: number; y: number; w: number; h: number }),
      opening: rect(r.opening as { x: number; y: number; w: number; h: number })
    }
  } catch {
    return { inside: [], opening: [] }
  }
}

/** A true-orientation point as it appears on a (possibly mirrored) preview. */
export function toScreen(p: Point, mirrored: boolean): Point {
  return mirrored ? { x: 1 - p.x, y: p.y } : p
}

/** A click on a (possibly mirrored) preview, back in true orientation. Same rule; it is its own inverse. */
export function fromScreen(p: Point, mirrored: boolean): Point {
  return toScreen(p, mirrored)
}

/** An SVG `points` attribute for an outline in a 0–1 viewBox, mirrored for the screen. */
export function svgPoints(points: readonly Point[], mirrored: boolean): string {
  return points
    .map((p) => toScreen(p, mirrored))
    .map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`)
    .join(' ')
}

/** Which drawn outline an edit is about. */
export type OutlineName = 'inside' | 'opening'

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v))

/** The outline with one corner moved to `p` (clamped to the picture). */
export function moveCorner(points: readonly Point[], index: number, p: Point): Point[] {
  return points.map((q, i) => (i === index ? { x: clamp01(p.x), y: clamp01(p.y) } : q))
}

/**
 * Undo/redo over whole states: `past` is what Undo returns to (newest last), `future` what Redo
 * returns to (next first). Pure, so the editor's history is tested without a screen.
 */
export interface History<T> {
  past: T[]
  present: T
  future: T[]
}

export function historyOf<T>(present: T): History<T> {
  return { past: [], present, future: [] }
}

/** A new state the operator made: the old one becomes undoable, and Redo is forgotten. */
export function commit<T>(h: History<T>, next: T, from: T = h.present): History<T> {
  if (JSON.stringify(next) === JSON.stringify(from)) return { ...h, present: next }
  return { past: [...h.past, from], present: next, future: [] }
}

export function undo<T>(h: History<T>): History<T> {
  if (h.past.length === 0) return h
  return {
    past: h.past.slice(0, -1),
    present: h.past[h.past.length - 1],
    future: [h.present, ...h.future]
  }
}

export function redo<T>(h: History<T>): History<T> {
  if (h.future.length === 0) return h
  return { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) }
}

/**
 * Where a click landed on an element, normalized 0–1 and clamped to it, from the click's client
 * coordinates and the element's bounding box. Clamped because a click on the very edge can round
 * a hair outside, and an outline point outside the picture is refused on save.
 */
export function clickPoint(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number }
): Point | null {
  if (rect.width <= 0 || rect.height <= 0) return null
  const clamp = (v: number): number => Math.min(1, Math.max(0, v))
  return {
    x: clamp((clientX - rect.left) / rect.width),
    y: clamp((clientY - rect.top) / rect.height)
  }
}

const EDGE_WORDS: Record<CartEdge, string> = {
  bottom: 'bottom',
  top: 'top',
  left: 'left side',
  right: 'right side'
}

/** The zone setup in one line, from the saved config (the Admin Panel's POS section). */
export function zoneSummary(config: PosConfig): string {
  if (config.zoneMode === 'drawn') {
    const n = (pts: unknown[]): string => `${pts.length} point${pts.length === 1 ? '' : 's'}`
    return `B · Drawn outlines — inside ${n(config.drawnInside)}, opening ${n(config.drawnOpening)}`
  }
  const pct = (v: number): string => `${Math.round(v * 100)}%`
  return `A · Bands — basket at the ${EDGE_WORDS[config.cartEdge]} of the picture, inside ${pct(
    config.insideFraction
  )}, opening ${pct(config.openingFraction)}`
}
