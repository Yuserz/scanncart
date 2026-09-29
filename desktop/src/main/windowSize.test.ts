// @vitest-environment node
import { describe, it, expect } from 'vitest'
import { MIN_SCALE, TABLET, tabletWindowBounds } from './windowSize'

describe('tabletWindowBounds', () => {
  it('opens on the tablet canvas and cannot be grown past it', () => {
    const bounds = tabletWindowBounds()

    expect(bounds.width).toBe(1024)
    expect(bounds.height).toBe(768)
    // The maximum IS the opening size: a tablet-sized app, so there is no larger size to offer.
    expect(bounds.maxWidth).toBe(bounds.width)
    expect(bounds.maxHeight).toBe(bounds.height)
    expect(bounds.maxWidth).toBe(TABLET.width)
    expect(bounds.maxHeight).toBe(TABLET.height)
  })

  it('keeps the canvas ratio at the floor', () => {
    const { width, height, minWidth, minHeight } = tabletWindowBounds()

    // The floor is the canvas scaled, not a second pair of numbers: a rectangle of the same shape
    // cannot favour one axis, so the layout that fits at the floor also fits at the canvas.
    expect(minWidth / minHeight).toBeCloseTo(width / height, 6)
  })

  it('never asks the layout for a size it was not designed at', () => {
    const { minWidth, minHeight, maxWidth, maxHeight } = tabletWindowBounds()

    // The old floor, and the point of standardizing: the floor may not go below the size the CSS
    // was written against, and the ceiling may not go below the floor.
    expect(minWidth).toBeGreaterThanOrEqual(720)
    expect(minHeight).toBeGreaterThanOrEqual(480)
    expect(minWidth).toBeLessThanOrEqual(maxWidth)
    expect(minHeight).toBeLessThanOrEqual(maxHeight)
    // 900px is where the Live view stops drawing the feed and the rail side by side, so a floor at
    // or above it would mean the stacked layout (and its item-log cap) is unreachable.
    expect(minWidth).toBeLessThanOrEqual(900)
  })

  it('moves every bound together when the canvas changes', () => {
    // The standard is one fact, so the next tablet is a one-line change — and the floor cannot be
    // left behind at the old canvas's size, which is what four hand-written numbers allowed.
    const larger = tabletWindowBounds({ width: 1280, height: 800 })

    expect(larger).toEqual({
      width: 1280,
      height: 800,
      minWidth: Math.round(1280 * MIN_SCALE),
      minHeight: Math.round(800 * MIN_SCALE),
      maxWidth: 1280,
      maxHeight: 800
    })
    expect(larger.minWidth).toBeGreaterThan(tabletWindowBounds().minWidth)
  })
})
