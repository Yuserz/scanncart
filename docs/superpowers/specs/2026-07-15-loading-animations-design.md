# SCANnCART — Loading animations for module/model waits — Design

**Date:** 2026-07-15
**Status:** Approved (discussed in-session)
**Depends on:** `desktop/src/renderer/src/App.tsx`, `views/LiveView.tsx`,
`views/AdminPanel.tsx` and their CSS; new `components/Spinner.tsx`.

## Problem

Three waits in the app show static, unstyled text with no indication that
anything is happening:

1. **App boot** (`App.tsx`): "Starting sidecar…" bare text for the 5–15+ s
   the Python sidecar spends importing torch/ultralytics (module loading).
2. **Capture start** (`LiveView`): `POST /api/capture/start` blocks while
   the YOLO model loads — and on first use of a model, downloads weights
   (tens of seconds for yolo26). The Start button gives zero feedback while
   the request is in flight, then the preview shows static "Waiting for
   frames…" until the first inference lands.
3. **Admin panel**: "Loading settings…" bare text (brief; included for
   consistency).

## Design

### `components/Spinner.tsx` + `Spinner.css`

A dependency-free CSS ring: `<span class="spinner" aria-hidden="true">`,
2px border in `var(--border)` with `border-top-color: var(--accent)`,
`border-radius: 50%`, 0.8 s linear infinite rotation. `size?: number` prop
(px, default 16) sets width/height via inline style. Purely decorative —
the accompanying text carries the meaning, so `aria-hidden` and no role.

### App boot screen (`App.tsx` + new `App.css`)

`.app-waiting` becomes a full-viewport centered column on the theme
background: `<Spinner size={28} />`, "Starting sidecar…" (existing copy,
kept so `App.test.tsx`'s text query still matches), and a dim sub-line
"loading Python runtime and model libraries". `App.css` imported from
`App.tsx`.

### LiveView start/stop feedback

- New local state `pending: 'start' | 'stop' | null` in `LiveView` (not the
  hook — purely presentational). The click handler sets it, awaits the
  hook's `start()`/`stop()`, and clears it in a `finally`.
- Button: disabled while pending; label becomes "Starting…"/"Stopping…"
  with a small inline spinner. `aria-label` keeps Start/Stop.
- Preview placeholder (only rendered when there is no frame):
  - `pending === 'start'`: spinner + "Loading model…" + dim sub-line
    "first use of a model downloads its weights (one time)".
  - running (post-start, pre-first-frame): spinner + "Waiting for frames…".
  - idle: current static "Waiting for frames…" text, unchanged.

### Admin panel

The `loading && !settings` branch gets a spinner next to the existing
"Loading settings…" text.

## Testing

- `LiveView.test.tsx`: with a `start()` that resolves on command
  (controllable promise), clicking Start shows a disabled "Starting…"
  button and the "Loading model…" placeholder; after resolution the button
  reads Stop. A stop-pending equivalent is covered by the same mechanism.
- `App.test.tsx`: existing "Starting sidecar" text assertion keeps passing;
  add a check that the boot screen renders the spinner.
- `AdminPanel.test.tsx`: never-resolving `getSettings` renders the spinner
  alongside "Loading settings…".
- Spinner itself is CSS animation — not unit-testable in jsdom beyond
  presence; visual check via the app driver once the sidecar can run again
  (currently blocked by the Defender native-DLL issue).

**Verified 2026-07-15 (second machine, no Defender interference — 10/10
`import torch,cv2,ultralytics` succeeded):** built the app and drove it with
a throwaway Playwright `_electron` script. Confirmed the boot screen mounts
past the sidecar-import wait quickly, the Start button shows a disabled
"Starting…" state with a spinner, the Live preview shows the spinner +
"Loading model… / first use of a model downloads its weights (one time)"
placeholder while `/api/capture/start` is in flight, and the Admin panel
shows the spinner next to "Loading settings…" during the initial fetch.
(No physical camera on this machine, so capture start itself errors out
after the model loads — expected and out of scope for this feature.)

## Non-goals

- No fake progress stages/percentages — the sidecar reports no import or
  download progress, so any staged copy would be invented.
- No skeleton screens.
- No sidecar/protocol changes.
