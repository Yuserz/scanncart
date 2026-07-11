// Pure geometry for drawing detection boxes over the preview image.
//
// Boxes arrive normalized 0-1. They MUST be scaled against the *actual rendered
// image dimensions* (dispW/dispH), not the container — the JPEG letterboxes
// inside its box under object-fit, so scaling to the container misaligns boxes.
// Callers pass the measured displayed image size explicitly.

import type { Detection } from './ws'

export interface PixelRect {
  x: number
  y: number
  w: number
  h: number
}

export interface LabeledRect extends PixelRect {
  label: string
}

export function boxToPixels(
  box: [number, number, number, number],
  dispW: number,
  dispH: number
): PixelRect {
  const [x1, y1, x2, y2] = box
  return {
    x: x1 * dispW,
    y: y1 * dispH,
    w: (x2 - x1) * dispW,
    h: (y2 - y1) * dispH
  }
}

export interface PercentRect {
  left: number
  top: number
  width: number
  height: number
}

// Percentages relative to a wrapper sized to the rendered image. Used by the
// DOM overlay (positioned elements), which is inherently letterbox-safe because
// the wrapper IS the image bounds.
export function boxToPercent(box: [number, number, number, number]): PercentRect {
  const [x1, y1, x2, y2] = box
  return {
    left: x1 * 100,
    top: y1 * 100,
    width: (x2 - x1) * 100,
    height: (y2 - y1) * 100
  }
}

export function layoutDetections(
  detections: Detection[],
  dispW: number,
  dispH: number
): LabeledRect[] {
  return detections.map((d) => ({
    ...boxToPixels(d.box, dispW, dispH),
    label: `${d.cls} ${Math.round(d.conf * 100)}%`
  }))
}
