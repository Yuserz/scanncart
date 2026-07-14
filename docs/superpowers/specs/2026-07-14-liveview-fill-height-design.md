# SCANnCART — Live View fills the window height — Design

**Date:** 2026-07-14
**Status:** Approved (pending spec review)
**Depends on:** `desktop/src/renderer/src/views/LiveView.tsx`+`.css`.

## Problem

On a tall window, the Live view's content row (`.live-body`) hugs its natural
content height: the preview wrapper is width-driven (`max-width: 1280px`,
height from the image's aspect ratio) and the side rail's item log is capped
at `max-height: 200px`. Everything sits in the top half of the window and the
rest is dead background (see 2026-07-14 screenshots). The Admin panel is not
affected — it scrolls and fills naturally.

## Goals

- The preview grows to use available window height as well as width, without
  distorting or cropping, and without regressing the "respect the real
  capture aspect ratio" fix (commit `8a75139`).
- The detection overlay stays 1:1 aligned with the image at any size.
- The side rail uses the full column height (item log grows instead of being
  capped at 200px; it still scrolls internally when full).
- No regression of the stacked `<900px` layout's dead-space fix
  (commit `9e69b95`) — stacked mode keeps natural content height.

## Non-goals

- No new panels or features in the freed space (explicitly decided against).
- No changes to hooks, WS/REST contracts, or the Admin panel.

## Design

CSS-first, with one small JSX touch to learn the frame's real aspect ratio.

1. **Height chain (row layout only):** `.live-body` gets `flex: 1;
   min-height: 0` so it claims the window height left under the toolbar.
   In the `<900px` stacked media query it stays `flex: none` (natural
   height), preserving the existing stacked-layout behavior.
2. **Preview sizing:** `.preview-wrapper` is sized by an explicit
   `aspect-ratio` (CSS variable, see 3) constrained by both `max-width: 100%`
   and `max-height: 100%` inside a flex `feed-col` (`min-width: 0;
   min-height: 0`). The browser resolves the largest box that fits both
   constraints at that ratio. `preview-img` becomes `width: 100%;
   height: 100%` filling the wrapper — the wrapper box *is* the image box, so
   the `inset: 0` overlay stays aligned exactly as today. The `max-width:
   1280px` cap is dropped: the preview may upscale beyond the JPEG's encoded
   size on large windows; acceptable for a live preview.
3. **Real aspect ratio, not hardcoded:** the wrapper's `aspect-ratio` comes
   from a CSS variable (e.g. `--preview-ar`) set from the rendered image's
   `naturalWidth`/`naturalHeight` in the img's `onLoad` handler (React state
   in `LiveView`, inline style on the wrapper). Until the first frame loads
   (idle placeholder), it defaults to `16 / 9` — same as today's
   placeholder-only fallback.
4. **Letterboxing:** when the window shape doesn't match the frame's ratio,
   the leftover space in `feed-col` is split around the preview
   (`justify-content: center; align-items: center` on the flex `feed-col`),
   giving a deliberate video-player look instead of a dead slab below.
5. **Side rail:** `.item-log` drops `max-height: 200px` in favor of
   `flex: 1; min-height: 0` inside the already-`flex: 1` `.log-card`, so the
   log fills the rail's remaining height and scrolls internally. In stacked
   mode the 200px cap is kept (the rail has no height budget to fill there).

## Testing

- Existing Vitest suites must pass unchanged — all `data-testid`s and the
  JSX structure (minus the new `onLoad` and inline style) are preserved.
- Add one renderer test: after the img fires `load` with fake
  `naturalWidth`/`naturalHeight` (jsdom reports 0/0, so define them on the
  element), the wrapper's inline `aspect-ratio` style reflects the frame's
  ratio; before any load it is the 16/9 default.
- Manual check (dev run): wide window → preview fills height, rail full
  height; tall/narrow window → preview centered with letterbox split;
  `<900px` width → stacked layout unchanged.
