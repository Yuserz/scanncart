# SCANnCART — Desktop UI/UX redesign (Live View + Admin Panel) — Design

**Date:** 2026-07-13
**Status:** Approved (pending spec review)
**Depends on:** `desktop/src/renderer/src/views/LiveView.tsx`+`.css`, `desktop/src/renderer/src/views/AdminPanel.tsx`+`.css`, `desktop/src/renderer/src/components/AppShell.tsx`+`.css`.

## Problem

The current renderer UI is functional but visually flat: one dark background
color, unstyled buttons, no card grouping, ad-hoc font sizes. Two concrete pain
points called out:

- **Live View** has no visual hierarchy between the video feed, performance
  stats, and the session item log — everything is plain text stacked in a
  column.
- **Admin Panel** renders its 9 settings fields as one flat vertical list
  (label + input + hint, repeated), with no grouping by purpose.

This is a prototype primarily viewed by a developer/operator at a desk
(monitoring during a demo), not a checkout-floor kiosk — so the redesign
should favor information density and visible diagnostics over large touch
targets.

## Goals

- Give both views real visual hierarchy: card-based grouping, a consistent
  type scale, and one shared dark/technical theme.
- Live View: surface performance stats and the item log as always-visible,
  clearly separated panels alongside the feed (no more digging through a flat
  stack).
- Admin Panel: cluster the 9 fields into purpose-based groups instead of one
  flat list; condense the hardware info and presets sections.
- Preserve all existing behavior, data flow, and `data-testid` hooks — this is
  a visual/structural pass, not a functional change.

## Non-goals

- No changes to `useSidecarStream`/`useSidecarSettings` hook logic, the WS/REST
  contracts, or item-log dedup behavior.
- No new features (e.g. item quantity aggregation, log filtering/search,
  timestamps in the item log) — out of scope for this pass.
- No changes to sidecar (Python) code.
- No routing library or new state-management dependency — `AppShell`'s
  existing `useState` view switch is kept.

## Design

### 1. Shared theme

New `desktop/src/renderer/src/styles/theme.css`, imported once (e.g. from
`App.tsx` or `main.tsx`), defining CSS custom properties consumed by both
views and `AppShell`:

```css
:root {
  --bg: #0b0d10;
  --panel: #14171b;
  --border: #262a30;
  --text: #e7e9ea;
  --dim: #8a9099;
  --accent: #2dd4a7;      /* refined teal */
  --accent-ink: #08130e;  /* text color on accent-filled surfaces */
  --danger-bg: #3a1418;
  --danger-fg: #ff8a8a;
  --danger-border: #5a2228;
  --font-ui: 'Inter', -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'Cascadia Code', 'SF Mono', Consolas, monospace;
}
```

Type scale (replacing today's ad-hoc sizes): 20px/600 for stat numbers,
15px/600 for section titles, 13px/500 body, 12.5px/500 field labels and log
rows, 11px/600 uppercase group headers (`letter-spacing: 0.08em`), 10.5px
italic for field hints. Numeric values (fps, latency, confidence %, all
number-type field inputs) use `--font-mono`; everything else uses
`--font-ui`.

A reusable `.card` class (`background: var(--panel); border: 1px solid
var(--border); border-radius: 8px;`) backs the stat card, item-log card,
admin group cards, and preset cards, so the four don't each hand-roll their
own box styling.

If Inter/JetBrains Mono aren't already available as web fonts bundled with
the app, fall back to the system stack already listed — no new font-loading
infra is introduced for this pass.

### 2. `AppShell` nav

Restyle the Live/Admin buttons as a pill-style segmented toggle (active tab:
`var(--panel)` background, `var(--text)`; inactive: transparent, `var(--dim)`).
Structure (`useState<View>`, two buttons) is unchanged — CSS only.

### 3. `LiveView` — sidebar dashboard layout

Restructure the JSX into: nav (from AppShell) → toolbar row → body row.

- **Toolbar row:** status indicator (dot + "Running"/"Idle" using `--accent`
  when running), connection state (`connected`/`disconnected`), and the
  Start/Stop button pushed right (danger-styled when it reads "Stop").
- **Body row:** `display: flex` — feed column (`flex: 2`, unchanged
  `preview-wrapper`/overlay logic) and a right rail (`flex: 1`, `min-width:
  230px`) containing two stacked `.card`s:
  - **Performance card:** 2×2 mono stat grid — infer fps, capture fps,
    latency ms, tracked count (tracked count = number of currently-open
    entries; derive from existing `items`/`frame.detections` data already in
    `useSidecarStream`, no new hook state).
  - **Item log card:** header with a live count badge (`items.length`), below
    it the existing scrollable list restyled as compact row chips (item name
    left, confidence in accent-colored mono right). Same underlying `items`
    array and dedup logic — only the row markup/styling changes.

All existing `data-testid`s (`state`, `conn`, `overlay`, `det-box`,
`stats`, `item-log`) are preserved on their (possibly relocated) elements.

### 4. `AdminPanel` — grouped sections

- **Header:** title + capture-state pill (existing `data-testid="capture-state"`
  kept), restyled only.
- **Hardware line:** condense the current `<ul>` of 3 items into a single
  row of inline label/value pairs (CPU, RAM, GPU) — same
  `systemInfo`/`describeGpu` data, no logic change. `data-testid="hardware-info"`
  kept on the containing element.
- **Presets:** same cards, restyled with `.card`, recommended preset gets an
  accent border + small "Recommended" badge (already exists, restyled).
- **Field groups:** replace the single flat `SETTINGS_FIELDS.map` render with
  three grouped sections, each a `.card` with an 11px uppercase header and a
  2-column CSS grid of fields:
  - **Model & Device** — `active_model`, `device`
  - **Camera & Capture** — `camera_index`, `capture_width`, `capture_height`,
    `capture_fps`, `preview_height`
  - **Detection & Tracking** — `conf_threshold`, `infer_frame_skip`,
    `track_expiry_s`

  Grouping is a new `const SETTINGS_GROUPS: { label: string; keys:
  (keyof SettingsPayload)[] }[]` in `settingsFields.ts`, consumed by
  `AdminPanel` in place of the flat `SETTINGS_FIELDS` iteration (existing
  `SETTINGS_FIELDS` array stays as the per-field metadata lookup — groups
  just reference its keys). Per-field restart/live badge logic, the custom
  device GPU/CPU toggle, hint text, and validation are all unchanged; only
  the grouping/markup/CSS around them changes. Hint text is visually
  demoted to a smaller italic caption (still always visible — no
  tooltips/hover-reveal, since this is a low-frequency admin screen where
  hiding guidance behind hover adds friction without saving meaningful space).
- **Actions row:** Save (accent-filled) / Restore Defaults (outline) —
  same handlers, restyled.

### 5. Testing

- Existing Vitest suites (`LiveView`, `AdminPanel`, `AppShell` if any) should
  pass unmodified since `data-testid` attributes are preserved.
- If any test queries structure that changes (e.g. item-log rows moving from
  `<li>` to a `<div>`-based chip), update that test's query alongside the
  markup change — don't leave a test asserting stale structure.
- No new test *behavior* is required (no new hook logic), but add/adjust
  assertions if the grouped `SETTINGS_GROUPS` rendering changes how
  `SETTINGS_FIELDS`-driven tests locate a given field's input.

## Open questions / follow-ups (explicitly deferred)

- Web font bundling (Inter/JetBrains Mono) vs. system-font fallback is left
  to implementation-time discovery of what's already available in the
  Electron renderer; not blocking this design.
