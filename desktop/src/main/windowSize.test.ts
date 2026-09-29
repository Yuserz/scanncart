// @vitest-environment node
import { describe, it, expect } from 'vitest'
import {
  MIN_HEIGHT,
  MIN_WIDTH,
  MIN_WIDTH_SCALE,
  TABLET,
  TUNING_BAND_MIN_HEIGHT,
  TUNING_BAND_MIN_WIDTH,
  tabletWindowBounds
} from './windowSize'

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

  it('holds the width floor at a fraction of the canvas', () => {
    // Width degrades smoothly — the rail keeps its `clamp()` minimum and the feed column takes
    // what is left — so this one really is a fraction, and a wider canvas moves it.
    expect(MIN_WIDTH).toBe(Math.round(TABLET.width * MIN_WIDTH_SCALE))
    expect(tabletWindowBounds().minWidth).toBe(768)
  })

  it('sets the height floor from the console budget, not from the canvas ratio', () => {
    // The one thing the old floor got wrong: it scaled both axes by 0.75, which gave 576 — 78px
    // short of what the console needs with the band shut (116px of chrome and padding, the
    // collapsed band's 132px at the narrow widths, a 12px gap and a 356px first row for the rail's
    // stats strip, flags line and four log rows), plus the title bar. Height runs out all at once,
    // so it is measured, not scaled.
    const shut = { chrome: 116, bandCollapsed: 132, gap: 12, firstRow: 356 }
    const content = Object.values(shut).reduce((a, b) => a + b, 0)

    expect(MIN_HEIGHT).toBeGreaterThanOrEqual(content + 38) // + the title bar
    expect(MIN_HEIGHT).not.toBe(Math.round(TABLET.height * MIN_WIDTH_SCALE))
    expect(MIN_HEIGHT).toBeLessThan(TABLET.height)
  })

  it('lets the band open exactly where the window standard says the console has room', () => {
    // The threshold the camera band reads (`CameraTuning`), so "the band is shut below the canvas"
    // and "the canvas is the maximum width" are one fact rather than two numbers that agree today.
    expect(TUNING_BAND_MIN_WIDTH).toBe(TABLET.width)
    expect(TUNING_BAND_MIN_WIDTH).toBeGreaterThanOrEqual(MIN_WIDTH)
  })

  it('holds the band shut if the window is too short for its row as well as too narrow', () => {
    // The threshold is a pair, because the band costs width *and* height: 116 of chrome, the band's
    // own row, a 12px gap and the rail's 356, which is 702 of content. Opening it below this would
    // take the item log's rows away on a window that has no room to lose them — and it is a content
    // height, because that is the number `matchMedia` compares.
    const openBudget = 116 + 218 + 12 + 356

    expect(TUNING_BAND_MIN_HEIGHT).toBeGreaterThanOrEqual(openBudget)
    // Reachable: the band can open at the canvas, and the floor is low enough that a window can be
    // short without the band being the thing that breaks.
    expect(TUNING_BAND_MIN_HEIGHT).toBeLessThanOrEqual(TABLET.height)
    expect(MIN_HEIGHT).toBeLessThan(TUNING_BAND_MIN_HEIGHT)
  })

  it('never asks the layout for a size it was not designed at', () => {
    const { minWidth, minHeight, maxWidth, maxHeight } = tabletWindowBounds()

    // The old floor, and the point of standardizing: the floor may not go below the size the CSS
    // was written against, and the ceiling may not go below the floor.
    expect(minWidth).toBeGreaterThanOrEqual(720)
    expect(minHeight).toBeGreaterThanOrEqual(480)
    expect(minWidth).toBeLessThanOrEqual(maxWidth)
    expect(minHeight).toBeLessThanOrEqual(maxHeight)
    // Nothing stacks below a breakpoint any more — the band closes itself and the grid holds — so
    // the floor is free to sit where the preview still has a column beside the rail.
    expect(minWidth).toBeLessThan(TUNING_BAND_MIN_WIDTH)
  })

  it('moves the width floor with the canvas and cannot cross its own ceiling', () => {
    // The standard is one fact, so the next tablet is a one-line change — and the floor cannot be
    // left behind at the old canvas's size, which is what four hand-written numbers allowed.
    const larger = tabletWindowBounds({ width: 1280, height: 800 })

    expect(larger).toEqual({
      width: 1280,
      height: 800,
      minWidth: Math.round(1280 * MIN_WIDTH_SCALE),
      // The height floor is a budget for this layout rather than a fraction of whatever canvas
      // arrives, so a taller canvas does not raise it. (It is also below the canvas, which is what
      // keeps a wider window from inheriting a floor above its own ceiling.)
      minHeight: MIN_HEIGHT,
      maxWidth: 1280,
      maxHeight: 800
    })
    expect(larger.minWidth).toBeGreaterThan(tabletWindowBounds().minWidth)

    // A canvas smaller than this layout's height budget clamps to itself rather than opening below
    // its own floor, and the width floor never crosses the ceiling either: below 900px the band is
    // shut, so a narrow window is a layout this console does fit — see the band thresholds.
    const small = tabletWindowBounds({ width: 640, height: 480 })

    expect(small.minWidth).toBe(Math.round(640 * MIN_WIDTH_SCALE))
    expect(small.minHeight).toBe(480)
    expect(small.minWidth).toBeLessThanOrEqual(small.maxWidth)
  })
})
