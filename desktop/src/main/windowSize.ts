// The app window's size, standardized to one tablet canvas instead of four loose numbers.
//
// What this replaces: `width: 1000, height: 760, minWidth: 720, minHeight: 480`, hard-coded in
// `createWindow`. Four unrelated values, so the app opened at a size nothing had chosen on purpose
// and could be dragged down to a floor nothing had laid out for. Nothing pinned the relationship
// between them either — the opening size and the floor could cross, and a bound written by hand can
// disagree with the CSS it exists to serve.
//
// The canvas is a 1024x768 (4:3) tablet in landscape, which is the layout the Live view is drawn
// for: the feed and the rail (stats strip, item log) side by side, with the camera band under both.
// So the window opens on the canvas and cannot be grown past it — `maxWidth`/`maxHeight` are the
// canvas, deliberately: this is a tablet-sized app, and a window stretched across a desktop monitor
// is a layout nothing here was designed or tested against (`--rail-min` would simply get the space).
//
// The floor is not a scale of the canvas on both axes, because the two axes fail differently and
// were measured rather than chosen:
//
//   * Width is a ratio, because the layout degrades smoothly: the rail takes its `clamp()` minimum
//     (280px) and the feed column keeps the rest. Three quarters of the canvas (768) is where the
//     preview still has a column worth watching next to it.
//   * Height is a *budget*, because it runs out all at once. With the band shut, the console needs
//     116px of chrome and padding around the body (nav, toolbar, padding, the gap above the body),
//     the band, a 12px gap, and a first row tall enough for the rail's own content — stats strip,
//     flags line and the item log's 4-row floor, 356px. That first row is what sets the floor: at
//     670 the rail fits without scrolling at both widths the window allows.
//
// The band's own threshold is the pair that makes those numbers hold: below either one it closes
// (`TUNING_BAND_MIN_WIDTH`/`TUNING_BAND_MIN_HEIGHT`), so the narrow sizes never have to fit its
// columns and the short ones never have to fit its row. What is left at the floor — preview, stats,
// flags, item log, and a one-line camera readout — fits with room to spare.
//
// Electron-free and pure, the same shape as `singleInstance.ts` and `sidecarHealth.ts`, so the
// bounds can be asserted without launching a window.

export interface WindowBounds {
  /** The opening size, and the largest the window may be. */
  width: number
  height: number
  minWidth: number
  minHeight: number
  maxWidth: number
  maxHeight: number
}

/** The canvas everything else is derived from: a 1024x768 (4:3) tablet in landscape. */
export const TABLET = { width: 1024, height: 768 } as const

/**
 * What the window's frame and title bar cost: 16px of width, 39px of height.
 *
 * This module states two kinds of size and they are not interchangeable. The canvas and the floors
 * are *window* sizes - what `BrowserWindow` is given - while the band's two thresholds are *content*
 * sizes, which is what the page's `matchMedia` and `innerWidth` report. Reading one as the other is
 * how the band came to be shut on the canvas: `TABLET.width` is 1024 and the content there is 1008.
 * The numbers are measured (a 1024x768 window draws a 1008x729 viewport on Windows, with the window
 * options `createWindow` uses) rather than computed, because the frame is the OS's, not ours.
 */
export const CONTENT_CHROME = { width: 16, height: 39 } as const

/**
 * The content area a window of this size draws - the renderer's own viewport.
 *
 * Only the band's thresholds are stated in these units, so nothing in the app calls this: it exists
 * so the tests can say *the band opens at the canvas and is shut at every corner of the floor* as
 * arithmetic, instead of comparing a window size against a viewport one and passing by luck. The
 * measured version of the same claim is `make verify-live-layout`, which asks a real engine for the
 * viewport rather than deriving it.
 */
export function contentSize(window: { width: number; height: number }): {
  width: number
  height: number
} {
  return {
    width: window.width - CONTENT_CHROME.width,
    height: window.height - CONTENT_CHROME.height
  }
}

/** How much of the canvas's width the window may shrink to — three quarters, 768px. */
export const MIN_WIDTH_SCALE = 0.75

// The shortest window the console fits in — the budget above, not a scale of the canvas — and the
// *narrow* sizes are the binding case: below 900px the collapsed band's readout goes full-width and
// stands 132px rather than 116 (measured), so the budget there is 116 + 132 + 12 + 356 = 616 of
// content, or a 654px window. 670 is that with slack enough to absorb a wrapped label. At three
// quarters of the canvas the same budget would be short by 78px, which is how the layout came to
// stack and scroll there.
export const MIN_HEIGHT = 670

/**
 * The width at which the camera band has room for its columns, and so opens by default.
 *
 * A *content* width, unlike every other number in this file — the renderer's viewport is the
 * window's content area, so this is a size `contentSize` produces rather than one a `BrowserWindow`
 * is given. The first version of this said `TABLET.width` (1024), which is a window size: at the
 * canvas the content is ~1008, so the comparison never held and the band would have been shut on the
 * one window it was drawn for. The columns need ~966px of body width (measured; the page's own
 * padding takes the rest), which is ~998 of content, rounded to 1000 — deliberately under the
 * canvas's content width, not equal to the window's.
 */
export const TUNING_BAND_MIN_WIDTH = 1000

/**
 * The content height at which the band's row, the gap under it and the rail's own content all fit:
 * 218 + 12 + 356 of body, plus the 116 of chrome and padding above it, is 702 — and a few pixels
 * of slack over that, because the band's row grows when one of its hints appears. A *content*
 * height, like the width above and unlike the floor below: `matchMedia` compares the renderer's
 * viewport, which is the window's content area (`contentSize`).
 */
export const TUNING_BAND_MIN_HEIGHT = 705

/** The narrowest the window may be, before a smaller canvas (or a guard) applies. */
export const MIN_WIDTH = Math.round(TABLET.width * MIN_WIDTH_SCALE)

/**
 * The four bounds a `BrowserWindow` needs, all read off one canvas.
 *
 * The width floor follows the canvas; the height floor is the measured budget above. Both are
 * clamped by their own ceiling, so a canvas smaller than this layout's needs can never produce a
 * floor above the size it opens at — the relationship the four hand-written numbers did not have.
 */
export function tabletWindowBounds(
  canvas: { width: number; height: number } = TABLET
): WindowBounds {
  return {
    width: canvas.width,
    height: canvas.height,
    // The width floor follows the canvas (a wider tablet gets a wider floor, and neither can cross
    // the size it opens at); the height floor is this layout's measured budget, so a taller canvas
    // does not raise it.
    minWidth: Math.min(canvas.width, Math.round(canvas.width * MIN_WIDTH_SCALE)),
    minHeight: Math.min(canvas.height, MIN_HEIGHT),
    maxWidth: canvas.width,
    maxHeight: canvas.height
  }
}
