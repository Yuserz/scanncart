// The app window's size, standardized to one tablet canvas instead of four loose numbers.
//
// What this replaces: `width: 1000, height: 760, minWidth: 720, minHeight: 480`, hard-coded in
// `createWindow`. Four unrelated values, so the app opened at a size nothing had chosen on purpose
// and could be dragged down to a floor nothing had laid out for. Nothing pinned the relationship
// between them either — the opening size and the floor could cross, and a bound written by hand can
// disagree with the CSS it exists to serve.
//
// The canvas is a 1024x768 (4:3) tablet in landscape, which is the layout the Live view is drawn
// for: the feed and the rail (stats strip, camera tuning, item log) side by side, with the stacked
// layout below the 900px breakpoint the CSS documents. So the window opens on the canvas and cannot
// be grown past it — `maxWidth`/`maxHeight` are the canvas, deliberately: this is a tablet-sized
// app, and a window stretched across a desktop monitor is a layout nothing here was designed or
// tested against (`--rail-min` would simply get the space).
//
// The floor is derived from the same canvas rather than chosen again: three quarters of it per axis
// (768x576), which keeps the canvas's own ratio and lands on the tablet's portrait *width*, so the
// smallest window is the one where the layout switches to its stacked form rather than a size it
// has never been asked to draw. That floor is deliberately *above* the old 720x480 — nothing that
// worked there stops working, and the app no longer offers a size below the breakpoint it is
// designed to stack at.
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

/** How much of the canvas the window may shrink to, per axis — three quarters, on both. */
export const MIN_SCALE = 0.75

/**
 * The four bounds a `BrowserWindow` needs, all read off one canvas.
 *
 * Derived rather than listed so the standard is one fact: another canvas (a different tablet, or a
 * larger one) moves the opening size, both maxima and both minima together, and cannot produce a
 * floor above the ceiling or an opening size outside it.
 */
export function tabletWindowBounds(
  canvas: { width: number; height: number } = TABLET,
  scale: number = MIN_SCALE
): WindowBounds {
  return {
    width: canvas.width,
    height: canvas.height,
    minWidth: Math.round(canvas.width * scale),
    minHeight: Math.round(canvas.height * scale),
    maxWidth: canvas.width,
    maxHeight: canvas.height
  }
}
