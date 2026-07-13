# SCANnCART Desktop UI/UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle `LiveView`, `AdminPanel`, and `AppShell` into a cohesive dark/technical theme with real visual hierarchy (card grouping, a consistent type scale), per `docs/superpowers/specs/2026-07-13-scanncart-ui-redesign-design.md`, without changing any hook logic, WS/REST contracts, or item-log/settings behavior.

**Architecture:** Pure CSS + JSX-structure changes in `desktop/src/renderer/src/`. A new shared `theme.css` defines CSS custom properties (colors, mono font stack) consumed by `AppShell.css`, `LiveView.css`, and `AdminPanel.css`. `LiveView.tsx` gains a sidebar-dashboard layout (feed + right rail of stat/log cards). `AdminPanel.tsx` gains a new `SETTINGS_GROUPS` data structure (in `settingsFields.ts`) that clusters the existing 9 fields into three named groups, rendered as separate cards instead of one flat list.

**Tech Stack:** React + TypeScript (renderer), Vitest + Testing Library (existing test setup), plain CSS (no new dependencies).

## Global Constraints

- Theme colors (exact hex, from the approved mockup): `--bg:#0b0d10`, `--panel:#14171b`, `--border:#262a30`, `--text:#e7e9ea`, `--dim:#8a9099`, `--accent:#2dd4a7`, `--accent-ink:#08130e`, `--danger-bg:#3a1418`, `--danger-fg:#ff8a8a`, `--danger-border:#5a2228`.
- Numeric/stat values use `--font-mono: 'JetBrains Mono', 'Cascadia Code', 'SF Mono', Consolas, monospace`. UI text keeps the app's existing body font stack (`base.css` already lists Inter first) — no new font-loading infrastructure.
- All existing `data-testid` values must survive unchanged: `state`, `conn`, `overlay`, `det-box`, `item-log`, `capture-state`, `hardware-info`, `apply-preset-<name>`, `save-settings`, `restore-defaults`, `admin-error`, `retry-load`, `restart-warning`, `device-toggle`, `device-gpu-note`, `server-warnings`, `nav-live`, `nav-admin`.
- Item-log rows stay `<li>` elements inside a `data-testid="item-log"` `<ul>` (an existing test does `querySelectorAll('li')`).
- No changes to `useSidecarStream`, `useSidecarSettings`, `lib/api.ts`, `lib/ws.ts`, `lib/overlay.ts`, or any sidecar (Python) file.
- `data-testid="stats"` is intentionally restructured (stat tiles instead of one text line) — the one test asserting on its old text format is updated as part of Task 2, not left broken.

---

### Task 1: Shared theme CSS variables

**Files:**
- Create: `desktop/src/renderer/src/assets/theme.css`
- Modify: `desktop/src/renderer/src/main.tsx`

**Interfaces:**
- Produces: CSS custom properties `--bg`, `--panel`, `--border`, `--text`, `--dim`, `--accent`, `--accent-ink`, `--danger-bg`, `--danger-fg`, `--danger-border`, `--font-mono` on `:root`, plus a reusable `.card` class (`background: var(--panel); border: 1px solid var(--border); border-radius: 8px;`). Consumed by Tasks 2–5.

This task is pure CSS with no testable logic — no test step, per the plan's own convention of not contriving fake tests for styling-only changes. Verify by running the app (Task 5's manual check covers this) and by `npm run typecheck`/`npm test` still passing (no behavior touched).

- [ ] **Step 1: Create the theme file**

```css
/* desktop/src/renderer/src/assets/theme.css */
:root {
  --bg: #0b0d10;
  --panel: #14171b;
  --border: #262a30;
  --text: #e7e9ea;
  --dim: #8a9099;
  --accent: #2dd4a7;
  --accent-ink: #08130e;
  --danger-bg: #3a1418;
  --danger-fg: #ff8a8a;
  --danger-border: #5a2228;
  --font-mono:
    'JetBrains Mono',
    'Cascadia Code',
    'SF Mono',
    Consolas,
    monospace;
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}
```

- [ ] **Step 2: Import it in `main.tsx`**

```tsx
import './assets/main.css'
import './assets/theme.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
```

- [ ] **Step 3: Run the full desktop test suite to confirm nothing broke**

Run: `cd desktop && npm test`
Expected: PASS (same pass count as before this change — this step only added an unused-so-far CSS file and its import).

- [ ] **Step 4: Commit**

```bash
git add desktop/src/renderer/src/assets/theme.css desktop/src/renderer/src/main.tsx
git commit -m "style(desktop): add shared dark/technical theme CSS variables"
```

---

### Task 2: `LiveView` sidebar-dashboard layout

**Files:**
- Modify: `desktop/src/renderer/src/views/LiveView.tsx`
- Modify: `desktop/src/renderer/src/views/LiveView.css`
- Modify: `desktop/src/renderer/src/views/LiveView.test.tsx`

**Interfaces:**
- Consumes: `useSidecarStream(port, deps)` (unchanged) returning `{ frame, statusState, connected, items, start, stop }`; `boxToPercent` (unchanged); theme variables from Task 1.
- Produces: same public `LiveViewProps` (`port`, `deps`); no new exports.

The stats strip changes from one text line per metric (`"infer 22.4 fps"`) to a 2×2 grid of labeled stat tiles (`data-testid="stat-infer-fps"`, `stat-capture-fps`, `stat-latency`, `stat-tracked`). This is a deliberate structural change per the approved design, so the existing test assertion on the old text format is updated first (TDD: update the test to describe the new desired behavior, watch it fail against old code, then implement).

- [ ] **Step 1: Update the stats assertion in the test to the new tile structure**

In `desktop/src/renderer/src/views/LiveView.test.tsx`, replace this line inside the first `it(...)` block:

```tsx
    expect(screen.getByTestId('stats')).toHaveTextContent('infer 22.4 fps')
```

with:

```tsx
    expect(screen.getByTestId('stat-infer-fps')).toHaveTextContent('22.4')
    expect(screen.getByTestId('stat-capture-fps')).toHaveTextContent('60')
    expect(screen.getByTestId('stat-latency')).toHaveTextContent('88')
    expect(screen.getByTestId('stat-tracked')).toHaveTextContent('1')
```

(`60` and `88` come from the existing `frameWith()` helper's `stats: { infer_fps: 22.4, capture_fps: 60, latency_ms: 88 }`; `1` is the single detection passed to `frameWith(...)` in that test, matching the new "tracked" tile = current frame's detection count.)

- [ ] **Step 2: Run the test file to verify it fails against the current implementation**

Run: `cd desktop && npx vitest run src/renderer/src/views/LiveView.test.tsx`
Expected: FAIL — `Unable to find an element by: [data-testid="stat-infer-fps"]` (the tile doesn't exist yet).

- [ ] **Step 3: Rewrite `LiveView.tsx` with the sidebar layout**

```tsx
import { type JSX } from 'react'
import { useSidecarStream, type StreamDeps } from '../hooks/useSidecarStream'
import { boxToPercent } from '../lib/overlay'
import './LiveView.css'

export interface LiveViewProps {
  port: number
  deps?: StreamDeps
}

export function LiveView({ port, deps }: LiveViewProps): JSX.Element {
  const { frame, statusState, connected, items, start, stop } = useSidecarStream(port, deps)
  const running = statusState === 'running'
  const stats = frame?.stats
  const trackedCount = frame?.detections.length ?? 0

  return (
    <div className="live-view">
      <div className="live-toolbar">
        <span className={`status-dot${running ? ' running' : ''}`} aria-hidden="true" />
        <span className="state" data-testid="state">
          {statusState}
        </span>
        <span className="conn" data-testid="conn">
          {connected ? 'connected' : 'disconnected'}
        </span>
        <button
          className={running ? 'btn-stop' : 'btn-start'}
          onClick={running ? stop : start}
          aria-label={running ? 'Stop' : 'Start'}
        >
          {running ? 'Stop' : 'Start'}
        </button>
      </div>

      <div className="live-body">
        <div className="feed-col">
          <div className="preview-wrapper">
            {frame ? (
              <img
                className="preview-img"
                alt="live preview"
                src={`data:image/jpeg;base64,${frame.jpeg}`}
              />
            ) : (
              <div className="preview-placeholder">Waiting for frames…</div>
            )}
            <div className="overlay" data-testid="overlay">
              {frame?.detections
                .filter((d) => d.box)
                .map((d, i) => {
                  const p = boxToPercent(d.box)
                  return (
                    <div
                      key={d.track_id ?? `d${i}`}
                      className="det-box"
                      data-testid="det-box"
                      style={{
                        position: 'absolute',
                        left: `${p.left}%`,
                        top: `${p.top}%`,
                        width: `${p.width}%`,
                        height: `${p.height}%`
                      }}
                    >
                      <span className="det-label">
                        {d.cls} {Math.round(d.conf * 100)}%
                      </span>
                    </div>
                  )
                })}
            </div>
          </div>
        </div>

        <div className="side-rail">
          <div className="card stats-card">
            <h4>Performance</h4>
            <div className="stats-strip" data-testid="stats">
              {stats ? (
                <>
                  <div className="stat-tile" data-testid="stat-infer-fps">
                    <b>{stats.infer_fps.toFixed(1)}</b>
                    <small>infer fps</small>
                  </div>
                  <div className="stat-tile" data-testid="stat-capture-fps">
                    <b>{stats.capture_fps.toFixed(0)}</b>
                    <small>capture fps</small>
                  </div>
                  <div className="stat-tile" data-testid="stat-latency">
                    <b>{stats.latency_ms.toFixed(0)}</b>
                    <small>latency ms</small>
                  </div>
                  <div className="stat-tile" data-testid="stat-tracked">
                    <b>{trackedCount}</b>
                    <small>tracked</small>
                  </div>
                </>
              ) : (
                <span>no stats yet</span>
              )}
            </div>
          </div>

          <div className="card log-card">
            <h4>
              Item log <span className="log-count">({items.length})</span>
            </h4>
            <ul className="item-log" data-testid="item-log">
              {items.map((it) => (
                <li className="log-row" key={it.track_id}>
                  <span className="log-cls">{it.cls}</span>{' '}
                  <span className="log-conf">({Math.round(it.conf * 100)}%)</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
```

Note the `{' '}` between the two spans in the item-log row — this preserves the exact `"banana (90%)"` text content the dedup test checks (`toHaveTextContent('banana (90%)')`), since JSX would otherwise collapse the newline-indented whitespace between them to nothing.

- [ ] **Step 4: Rewrite `LiveView.css` for the new structure**

```css
.live-view {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  color: var(--text);
  background: var(--bg);
}

.live-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--dim);
}

.status-dot.running {
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent);
}

.live-toolbar .state {
  font-size: 12px;
  font-weight: 500;
  color: var(--dim);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.live-toolbar .conn {
  font-size: 12px;
  color: var(--dim);
}

.live-toolbar button {
  margin-left: auto;
  padding: 6px 18px;
  font-weight: 600;
  font-size: 12.5px;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text);
}

.live-toolbar .btn-stop {
  background: var(--danger-bg);
  color: var(--danger-fg);
  border-color: var(--danger-border);
}

.live-body {
  display: flex;
  gap: 14px;
}

.feed-col {
  flex: 2;
  min-width: 0;
}

/* Wrapper shrink-wraps the image so percentage-positioned boxes stay aligned
   regardless of container size. align-self: flex-start is essential: without it
   the flex column's default align-items: stretch widens the wrapper to the full
   window, making the inset:0 overlay wider than the (max 1280px) image and
   drifting boxes right of their objects once the window exceeds the image width. */
.preview-wrapper {
  position: relative;
  display: inline-block;
  align-self: flex-start;
  max-width: 100%;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
}

.preview-img {
  display: block;
  max-width: 100%;
  height: auto;
}

.preview-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 640px;
  max-width: 100%;
  aspect-ratio: 16 / 9;
  color: var(--dim);
}

.overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.det-box {
  border: 2px solid var(--accent);
  box-sizing: border-box;
}

.det-label {
  position: absolute;
  top: -18px;
  left: -2px;
  background: var(--accent);
  color: var(--accent-ink);
  font-size: 11px;
  font-weight: 700;
  padding: 1px 4px;
  white-space: nowrap;
}

.side-rail {
  flex: 1;
  min-width: 230px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.side-rail .card {
  padding: 12px 14px;
}

.side-rail h4 {
  margin: 0 0 10px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
  font-weight: 600;
}

.stats-strip {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 8px;
}

.stat-tile {
  font-family: var(--font-mono);
}

.stat-tile b {
  display: block;
  font-size: 20px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}

.stat-tile small {
  color: var(--dim);
  font-size: 11px;
  font-weight: 500;
  font-family: initial;
}

.log-card {
  display: flex;
  flex-direction: column;
  flex: 1;
}

.log-count {
  color: var(--dim);
  font-weight: 500;
  text-transform: none;
  letter-spacing: 0;
}

.item-log {
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 12.5px;
}

.log-row {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 6px;
  padding: 7px 9px;
  font-weight: 500;
}

.log-conf {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent);
}
```

- [ ] **Step 5: Run the test file to verify it passes**

Run: `cd desktop && npx vitest run src/renderer/src/views/LiveView.test.tsx`
Expected: PASS (all 3 tests).

- [ ] **Step 6: Run the full desktop suite to catch any cross-file regression**

Run: `cd desktop && npm test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add desktop/src/renderer/src/views/LiveView.tsx desktop/src/renderer/src/views/LiveView.css desktop/src/renderer/src/views/LiveView.test.tsx
git commit -m "feat(desktop): sidebar-dashboard layout for LiveView with stat tiles"
```

---

### Task 3: `AppShell` nav restyle

**Files:**
- Modify: `desktop/src/renderer/src/components/AppShell.css`

**Interfaces:**
- Consumes: theme variables from Task 1. `AppShell.tsx` markup/logic is untouched (same `View` state, same `nav-live`/`nav-admin` test ids and `aria-pressed`).
- Produces: nothing new consumed by later tasks — purely visual.

No test changes: `AppShell.test.tsx` only asserts on `aria-pressed` and text content, both unaffected by CSS.

- [ ] **Step 1: Rewrite `AppShell.css`**

```css
.app-shell {
  display: flex;
  flex-direction: column;
  background: var(--bg);
  min-height: 100vh;
}

.app-nav {
  display: flex;
  gap: 4px;
  padding: 8px 16px;
  border-bottom: 1px solid var(--border);
}

.app-nav button {
  padding: 6px 14px;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--dim);
  cursor: pointer;
  font-weight: 600;
  font-size: 13px;
}

.app-nav button.active {
  color: var(--text);
  background: var(--panel);
}
```

- [ ] **Step 2: Run the `AppShell` test file to confirm no regression**

Run: `cd desktop && npx vitest run src/renderer/src/components/AppShell.test.tsx`
Expected: PASS (both tests).

- [ ] **Step 3: Commit**

```bash
git add desktop/src/renderer/src/components/AppShell.css
git commit -m "style(desktop): restyle AppShell nav as a segmented pill toggle"
```

---

### Task 4: `SETTINGS_GROUPS` data structure

**Files:**
- Modify: `desktop/src/renderer/src/lib/settingsFields.ts`

**Interfaces:**
- Consumes: existing `SettingsPayload` type, existing `SETTINGS_FIELDS: FieldMeta[]`.
- Produces: `export interface FieldGroup { label: string; keys: (keyof SettingsPayload)[] }` and `export const SETTINGS_GROUPS: FieldGroup[]` — consumed by Task 5's `AdminPanel.tsx`.

This is a plain data addition (no branching logic), so there's no meaningful unit to TDD in isolation; Task 5's existing `AdminPanel.test.tsx` suite (which already exercises every field via `getByLabelText`) is the test for whether the grouping is wired correctly, per that task's steps.

- [ ] **Step 1: Append the group definition to `settingsFields.ts`**

Add after the existing `SETTINGS_FIELDS` array (end of file):

```ts
export interface FieldGroup {
  label: string
  keys: (keyof SettingsPayload)[]
}

export const SETTINGS_GROUPS: FieldGroup[] = [
  { label: 'Model & Device', keys: ['active_model', 'device'] },
  {
    label: 'Camera & Capture',
    keys: ['camera_index', 'capture_width', 'capture_height', 'capture_fps', 'preview_height']
  },
  {
    label: 'Detection & Tracking',
    keys: ['conf_threshold', 'infer_frame_skip', 'track_expiry_s']
  }
]
```

- [ ] **Step 2: Typecheck**

Run: `cd desktop && npm run typecheck`
Expected: PASS (no errors — `SETTINGS_GROUPS` isn't consumed yet, so this only checks the new code itself is well-typed).

- [ ] **Step 3: Commit**

```bash
git add desktop/src/renderer/src/lib/settingsFields.ts
git commit -m "feat(desktop): add SETTINGS_GROUPS clustering for the admin form"
```

---

### Task 5: `AdminPanel` grouped-sections layout

**Files:**
- Modify: `desktop/src/renderer/src/views/AdminPanel.tsx`
- Modify: `desktop/src/renderer/src/views/AdminPanel.css`

**Interfaces:**
- Consumes: `useSidecarSettings` (unchanged), `SETTINGS_FIELDS` + new `SETTINGS_GROUPS` from Task 4, theme variables from Task 1.
- Produces: same public `AdminPanelProps`; no new exports. All existing `data-testid`s preserved (`capture-state`, `hardware-info`, `apply-preset-<name>`, `save-settings`, `restore-defaults`, `admin-error`, `retry-load`, `restart-warning`, `device-toggle`, `device-gpu-note`, `server-warnings`).

No test changes expected: every `AdminPanel.test.tsx` assertion targets a `data-testid`, `getByLabelText`, or a text substring that this task preserves verbatim — only the surrounding grouping/markup/CSS changes. Steps below run the existing suite before and after to prove that.

- [ ] **Step 1: Run the existing `AdminPanel` suite as a pre-change baseline**

Run: `cd desktop && npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: PASS (all tests, current implementation) — establishes the baseline this task must not break.

- [ ] **Step 2: Rewrite `AdminPanel.tsx`**

Replace the whole file with:

```tsx
import { useState, type JSX } from 'react'
import { useSidecarSettings, type SettingsDeps } from '../hooks/useSidecarSettings'
import type {
  SettingsPayload,
  SettingsResponse,
  SettingsUpdate,
  SystemInfoResponse
} from '../lib/api'
import { SETTINGS_FIELDS, SETTINGS_GROUPS, type FieldMeta } from '../lib/settingsFields'
import './AdminPanel.css'

export interface AdminPanelProps {
  port: number
  deps?: SettingsDeps
}

// Human-readable one-liner for the "This machine" GPU row.
function describeGpu(si: SystemInfoResponse): string {
  if (si.accelerator === 'cuda') {
    return `${si.gpu_name ?? 'CUDA GPU'} (${si.gpu_vram_gb?.toFixed(1) ?? '?'} GB VRAM) — GPU acceleration available`
  }
  if (si.accelerator === 'integrated') {
    // A CUDA-capable NVIDIA card that torch can't use is almost always a
    // missing CUDA torch build, not an APU — say so rather than mislabeling a
    // discrete GPU as integrated.
    if (si.gpu_name?.toLowerCase().includes('nvidia')) {
      return `${si.gpu_name} — GPU detected but CUDA is unavailable (install a CUDA build of torch to accelerate); running on CPU`
    }
    return `Integrated graphics: ${si.gpu_name ?? 'unknown'} (APU) — no CUDA acceleration, runs on CPU`
  }
  return 'No GPU detected — CPU only'
}

export function AdminPanel({ port, deps }: AdminPanelProps): JSX.Element {
  const {
    settings,
    systemInfo,
    presets,
    recommended,
    captureState,
    loading,
    saving,
    error,
    update,
    applyPreset,
    restoreDefaults,
    refresh
  } = useSidecarSettings(port, deps)

  // Holds only *unsaved* edits; reset whenever the server-confirmed settings
  // object changes identity (initial load, or after a successful
  // save/preset/restore-defaults). Adjusted during render rather than in a
  // useEffect, per React's guidance on resetting state when a prop changes.
  const [settingsAtLastReset, setSettingsAtLastReset] = useState<SettingsResponse | null>(null)
  const [draft, setDraft] = useState<SettingsUpdate>({})
  if (settings !== settingsAtLastReset) {
    setSettingsAtLastReset(settings)
    setDraft({})
  }

  const running = captureState === 'running'

  if (loading && !settings) {
    return (
      <div className="admin-panel">
        <p>Loading settings…</p>
      </div>
    )
  }

  if (!settings) {
    return (
      <div className="admin-panel">
        <p className="admin-error" data-testid="admin-error">
          Could not load settings from the sidecar{error ? `: ${error}` : ''}. Retrying
          automatically — this is expected for a few seconds right after launch while the sidecar
          finishes starting up.
        </p>
        <button onClick={() => void refresh()} data-testid="retry-load">
          Retry now
        </button>
      </div>
    )
  }

  const valueOf = (key: keyof SettingsPayload): string | number => {
    const draftVal = draft[key]
    return (draftVal !== undefined ? draftVal : settings[key]) as string | number
  }

  const setField = (key: keyof SettingsPayload, value: string | number): void => {
    setDraft((prev) => ({ ...prev, [key]: value }) as SettingsUpdate)
  }

  const pendingFields = Object.keys(draft) as (keyof SettingsPayload)[]
  const pendingRestartFields = pendingFields.filter((f) =>
    settings.restart_required_fields.includes(f)
  )
  const blockedByRunning = running && pendingRestartFields.length > 0
  const canSave = pendingFields.length > 0 && !blockedByRunning

  const handleSave = async (): Promise<void> => {
    if (!canSave) return
    await update(draft)
  }

  const handleApplyPreset = async (name: string): Promise<void> => {
    if (running) return
    await applyPreset(name)
  }

  const handleRestoreDefaults = async (): Promise<void> => {
    if (running) return
    await restoreDefaults()
  }

  const renderField = (field: FieldMeta): JSX.Element => {
    const isRestartField = settings.restart_required_fields.includes(field.key)
    const value = valueOf(field.key)

    if (field.key === 'device') {
      const gpuAvailable = systemInfo?.accelerator === 'cuda'
      const isCpu = value === 'cpu' || !gpuAvailable
      return (
        <div className="admin-field" key={field.key}>
          <div className="admin-field-label">
            <span className="admin-field-labeltext">{field.label}</span>
            <span className={`badge ${isRestartField ? 'restart' : 'live'}`}>
              {isRestartField ? 'restart required' : 'live'}
            </span>
          </div>
          <div className="device-toggle" data-testid="device-toggle">
            <label>
              <input
                type="radio"
                name="device"
                value="gpu"
                checked={!isCpu}
                disabled={!gpuAvailable}
                onChange={() => setField('device', 'auto')}
              />
              GPU (recommended)
            </label>
            <label>
              <input
                type="radio"
                name="device"
                value="cpu"
                checked={isCpu}
                onChange={() => setField('device', 'cpu')}
              />
              CPU only
            </label>
          </div>
          {!gpuAvailable && (
            <p className="field-hint" data-testid="device-gpu-note">
              No CUDA GPU on this machine — integrated/APU can&apos;t accelerate; running on CPU.
            </p>
          )}
          <p className="field-hint">{field.hint}</p>
        </div>
      )
    }

    return (
      <div className="admin-field" key={field.key}>
        <div className="admin-field-label">
          <label htmlFor={field.key}>{field.label}</label>
          <span className={`badge ${isRestartField ? 'restart' : 'live'}`}>
            {isRestartField ? 'restart required' : 'live'}
          </span>
        </div>
        {field.type === 'select' ? (
          <select
            id={field.key}
            value={String(value)}
            onChange={(e) => setField(field.key, e.target.value)}
          >
            {field.options?.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={field.key}
            type="number"
            value={value}
            min={field.min}
            max={field.max}
            step={field.step}
            onChange={(e) => {
              const n = e.target.valueAsNumber
              if (!Number.isNaN(n)) setField(field.key, n)
            }}
          />
        )}
        <p className="field-hint">{field.hint}</p>
      </div>
    )
  }

  return (
    <div className="admin-panel">
      <header className="admin-header">
        <h2>Admin Settings</h2>
        <span className="state" data-testid="capture-state">
          {captureState}
        </span>
      </header>

      <section className="admin-hardware" data-testid="hardware-info">
        {systemInfo ? (
          <>
            <span>CPU cores: {systemInfo.cpu_count}</span>
            <span>RAM: {systemInfo.ram_gb.toFixed(1)} GB</span>
            <span>GPU: {describeGpu(systemInfo)}</span>
          </>
        ) : (
          <span>Detecting hardware…</span>
        )}
      </section>

      <section className="admin-presets">
        <div className="preset-cards">
          {presets.map((p) => (
            <div
              key={p.name}
              className={`preset-card${p.name === recommended ? ' recommended' : ''}`}
            >
              <div className="preset-card-header">
                <strong>{p.label}</strong>
                {p.name === recommended && (
                  <span className="badge accent">Recommended for this machine</span>
                )}
              </div>
              <p>{p.description}</p>
              <button
                disabled={running || saving}
                onClick={() => void handleApplyPreset(p.name)}
                data-testid={`apply-preset-${p.name}`}
              >
                Apply preset
              </button>
            </div>
          ))}
        </div>
        {running && <p className="admin-warning">Stop capture to apply a preset.</p>}
      </section>

      <div className="admin-groups">
        {SETTINGS_GROUPS.map((group) => (
          <section className="admin-group" key={group.label}>
            <h4>{group.label}</h4>
            <div className="admin-grid">
              {group.keys.map((key) => {
                const field = SETTINGS_FIELDS.find((f) => f.key === key)
                return field ? renderField(field) : null
              })}
            </div>
          </section>
        ))}
      </div>

      {settings.warnings.length > 0 && (
        <ul className="admin-warnings" data-testid="server-warnings">
          {settings.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      {blockedByRunning && (
        <p className="admin-warning" data-testid="restart-warning">
          Stop capture to change: {pendingRestartFields.join(', ')}.
        </p>
      )}

      {error && (
        <p className="admin-error" data-testid="admin-error">
          {error}
        </p>
      )}

      <div className="admin-actions">
        <button
          className="btn-primary"
          disabled={!canSave || saving}
          onClick={() => void handleSave()}
          data-testid="save-settings"
        >
          Save
        </button>
        <button
          className="btn-outline"
          disabled={running || saving}
          onClick={() => void handleRestoreDefaults()}
          data-testid="restore-defaults"
        >
          Restore Defaults
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Run the `AdminPanel` suite to confirm it still passes**

Run: `cd desktop && npx vitest run src/renderer/src/views/AdminPanel.test.tsx`
Expected: PASS (all tests — same set as the Step 1 baseline).

- [ ] **Step 4: Rewrite `AdminPanel.css`**

```css
.admin-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  color: var(--text);
  background: var(--bg);
  max-width: 760px;
  font-size: 13px;
}

.admin-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.admin-header h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
}

.admin-header .state {
  font-size: 11px;
  font-weight: 600;
  color: var(--dim);
  background: rgba(255, 255, 255, 0.05);
  padding: 3px 9px;
  border-radius: 20px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.admin-hardware {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--dim);
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}

.preset-cards {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.preset-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  flex: 1 1 200px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  position: relative;
}

.preset-card.recommended {
  border-color: var(--accent);
}

.preset-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.preset-card-header strong {
  font-size: 12.5px;
  font-weight: 600;
}

.preset-card p {
  font-size: 11px;
  color: var(--dim);
  flex: 1;
}

.preset-card button {
  align-self: flex-start;
  font-size: 11px;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 5px 10px;
  border-radius: 6px;
  cursor: pointer;
}

.preset-card button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.badge {
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.1);
  color: var(--dim);
}

.badge.live {
  background: var(--accent);
  color: var(--accent-ink);
}

.badge.restart {
  background: rgba(255, 100, 100, 0.15);
  color: #ffb4b4;
}

.badge.accent {
  background: var(--accent);
  color: var(--accent-ink);
  position: absolute;
  top: -8px;
  right: 10px;
}

.admin-groups {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.admin-group {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
}

.admin-group h4 {
  margin: 0 0 12px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
  font-weight: 600;
}

.admin-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px 20px;
}

.admin-field {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.admin-field-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12.5px;
  font-weight: 500;
}

.admin-field input,
.admin-field select {
  width: 100%;
  box-sizing: border-box;
  background: #0e1114;
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 8px;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: 12.5px;
}

.device-toggle {
  display: flex;
  gap: 14px;
  font-size: 12.5px;
}

.field-hint {
  font-size: 10.5px;
  color: var(--dim);
  font-style: italic;
  margin: 0;
  line-height: 1.3;
}

.admin-warnings {
  list-style: disc;
  margin: 0;
  padding-left: 20px;
  font-size: 12px;
  color: #ffb020;
}

.admin-warning {
  font-size: 12px;
  color: #ffb020;
  margin: 0;
}

.admin-error {
  font-size: 12px;
  color: #ff6b6b;
  margin: 0;
}

.admin-actions {
  display: flex;
  gap: 10px;
}

.admin-actions button {
  padding: 8px 16px;
  font-weight: 600;
  font-size: 12.5px;
  border-radius: 6px;
  cursor: pointer;
}

.admin-actions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.admin-actions .btn-primary {
  background: var(--accent);
  border: none;
  color: var(--accent-ink);
}

.admin-actions .btn-outline {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--dim);
}
```

- [ ] **Step 5: Run the full desktop suite**

Run: `cd desktop && npm test`
Expected: PASS (every test file, including `App.test.tsx` and `AppShell.test.tsx` which mount `AdminPanel`/`LiveView` indirectly).

- [ ] **Step 6: Typecheck and lint**

Run: `cd desktop && npm run typecheck && npm run lint`
Expected: PASS, no errors.

- [ ] **Step 7: Commit**

```bash
git add desktop/src/renderer/src/views/AdminPanel.tsx desktop/src/renderer/src/views/AdminPanel.css
git commit -m "feat(desktop): grouped-sections layout for AdminPanel"
```

---

### Task 6: Manual verification pass

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Ensure the sidecar venv is set up (one-time)**

Run: `cd sidecar && python -m pytest -v` (confirms the sidecar test environment is functional; the redesign doesn't touch this side, but `make dev` spawns it).
Expected: PASS (existing suite, untouched by this plan).

- [ ] **Step 2: Launch the full app in dev mode**

Run: `cd desktop && npm run dev`
Expected: Electron window opens showing the Live view with the new sidebar layout (feed left, Performance + Item log cards right, pill nav at top).

- [ ] **Step 3: Exercise the Live view**

In the running app: click Start, confirm the status dot turns accent-teal and reads "running", the Stop button turns red-tinted, the Performance card's four tiles populate with live numbers, and detected items appear as rows in the Item log card with the confidence in teal mono. Click Stop and confirm it returns to idle styling.

- [ ] **Step 4: Exercise the Admin view**

Click the Admin nav tab. Confirm the three group cards (Model & Device / Camera & Capture / Detection & Tracking) render all 9 fields with visible restart/live badges, the hardware line and preset cards read correctly, and Save/Restore Defaults behave as before (Save disabled until a field changes; blocked with a warning if a restart-required field changes while capture is running).

- [ ] **Step 5: Report back**

Note any visual issues found (e.g. overflow, misaligned grid, contrast problems) so they can be fixed in a follow-up commit before considering this plan done. If everything matches the approved mockups, the redesign is complete.
