import { useState, type JSX } from 'react'
import { useDatasetStatus } from '../hooks/useDatasetStatus'
import { useSidecarSettings, type SettingsDeps } from '../hooks/useSidecarSettings'
import type {
  SettingsPayload,
  SettingsResponse,
  SettingsUpdate,
  SystemInfoResponse
} from '../lib/api'
import { recordedConfirmation, resizeModeMismatch } from '../lib/resizeMode'
import {
  ALLOWED_BACKENDS,
  ALLOWED_MODELS,
  BACKEND_HINTS,
  BACKEND_LABELS,
  EXPERIMENTAL_MODELS,
  MODEL_LABELS,
  minTrackExpiryS,
  MODEL_SPEC_HINTS,
  REMOTE_BACKENDS,
  SETTINGS_FIELDS,
  SETTINGS_GROUPS,
  type FieldMeta
} from '../lib/settingsFields'
import { Spinner } from '../components/Spinner'
import './AdminPanel.css'

// "bottle, cup" → ["bottle", "cup"]. Blank entries are dropped so a trailing
// comma never puts an empty class name on the wire.
function parseList(text: string): string[] {
  return text
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s !== '')
}

export interface AdminPanelProps {
  port: number
  deps?: SettingsDeps
}

// "2 min ago" rather than a raw timestamp, because the only thing that matters about
// this reading is how stale it is - the panel is showing a snapshot, not live state.
function describeAge(seconds: number | null): string {
  if (seconds === null) return 'age unknown'
  if (seconds < 90) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 48) return `${hours} h ago`
  return `${Math.round(hours / 24)} d ago`
}

// "640×360" — a size pair as an operator reads a resolution, not as [w, h].
function fmtSize([width, height]: [number, number]): string {
  return `${width}×${height}`
}

// What the remote probe's two sizes amount to. A union rather than a bag of fields so that
// `state: 'unreported'` and `reported: null` cannot come apart — the render switches on the state
// and reads the sizes, and the one state with nothing reported is the one that has no size to
// read. See the comment where it is computed for why that state is not "they agreed".
type ProbeGeometry =
  | { state: 'unreported'; sent: [number, number]; reported: null; aspectChanged: false }
  | {
      state: 'same' | 'resized'
      sent: [number, number]
      reported: [number, number]
      aspectChanged: boolean
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
    refresh,
    probe,
    probing,
    probeResult,
    cameras,
    refreshCameras,
    camerasLoading,
    models,
    installed,
    recordRequirement,
    recording,
    savedSettings,
    stopCapture,
    startCapture,
    stopping
  } = useSidecarSettings(port, deps)

  // Its own hook, not part of useSidecarSettings: that one polls the sidecar on
  // timers, and this reads a snapshot file that only changes when the dataset tool is
  // run by hand. `deps.apiFactory` is forwarded so tests inject the same fake client.
  const dataset = useDatasetStatus(port, deps?.apiFactory ? { apiFactory: deps.apiFactory } : {})

  // Holds only *unsaved* edits; reset whenever the server-confirmed settings
  // object changes identity (initial load, or after a successful
  // save/preset/restore-defaults). Adjusted during render rather than in a
  // useEffect, per React's guidance on resetting state when a prop changes.
  const [settingsAtLastReset, setSettingsAtLastReset] = useState<SettingsResponse | null>(null)
  const [draft, setDraft] = useState<SettingsUpdate>({})
  // Raw text behind the comma-separated (list) fields. The draft holds the
  // parsed array, but the input has to show exactly what was typed: parsing on
  // every keystroke eats the comma as soon as it is typed, so "bottle, cup"
  // collapsed to "bottlecupp".
  const [listText, setListText] = useState<Record<string, string>>({})
  const [justSaved, setJustSaved] = useState(false)
  if (settings !== settingsAtLastReset) {
    setSettingsAtLastReset(settings)
    setDraft({})
    setListText({})
  }

  const running = captureState === 'running'

  // The Tier A rows name a class by slug, and the snapshot's own roster is the only place
  // that translates one, so build the lookup from it rather than duplicating names here.
  const datasetClassNames = new Map(
    (dataset.status?.classes ?? []).map((c) => [c.slug, c.name] as const)
  )
  // The Model picker is the built-in list plus whatever the sidecar found under
  // sidecar/models/, plus the saved value if it is in neither. Those last two are what stop
  // this being a fixed list: a locally trained `.pt` becomes selectable the moment it is
  // copied in, with no renderer edit. The saved value is always offered even when the file
  // is gone — a configuration that points at nothing has to be visible to be diagnosable,
  // and `POST /api/detector/probe` is what explains it.
  const savedModel = String(savedSettings?.active_model ?? '')
  const modelOptions = Array.from(
    new Set([
      ...models,
      ...(ALLOWED_MODELS as readonly string[]),
      ...(savedModel ? [savedModel] : [])
    ])
  )

  const datasetSessions = dataset.status?.sessions ?? []
  // The predicate is deliberately narrower than the `splits` count beside it: what spoils
  // a test number is a session sitting on *both sides of the train/test line*, because
  // those frames share a rig state, a day and a lighting setup. A session spread across
  // train and valid only is the ordinary remainder split - it costs model selection some
  // independence, which is a different (and much smaller) problem, and flagging it here
  // would call the acceptance number into question when it is not actually in question.
  const testSessions = datasetSessions.filter((s) => s.train > 0 && s.test > 0)
  // What is left to label, biggest first, as the sidecar ordered it. The panel does not sort
  // this itself: the order is a claim about priority, and two places producing it is two
  // places that can disagree.
  const backlog = dataset.status?.labeling_backlog ?? []

  if (loading && !settings) {
    return (
      <div className="admin-panel">
        <p className="admin-loading">
          <Spinner /> Loading settings…
        </p>
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

  const valueOf = (key: keyof SettingsPayload): string | number | string[] => {
    const draftVal = draft[key]
    return (draftVal !== undefined ? draftVal : settings[key]) as string | number | string[]
  }

  const setField = (key: keyof SettingsPayload, value: string | number | string[]): void => {
    setJustSaved(false)
    setDraft((prev) => ({ ...prev, [key]: value }) as SettingsUpdate)
  }

  const pendingFields = Object.keys(draft) as (keyof SettingsPayload)[]
  const pendingRestartFields = pendingFields.filter((f) =>
    settings.restart_required_fields.includes(f)
  )
  const blockedByRunning = running && pendingRestartFields.length > 0
  const selectedBackend = String(valueOf('detector_backend'))
  const backendIsRemote = REMOTE_BACKENDS.includes(selectedBackend)
  // The Roboflow URL/workspace fields are noise when running native weights.
  const visibleGroups = SETTINGS_GROUPS.filter(
    (g) => g.home === 'admin' && (g.label !== 'Roboflow API backends' || backendIsRemote)
  )
  const canSave = pendingFields.length > 0 && !blockedByRunning
  const selectedCamera = cameras.find((c) => c.index === Number(valueOf('camera_index')))

  // The sidecar emits one warning that lists every restart-required field by
  // name whenever capture runs. It is 16 items of comma-separated prose, and
  // the per-field badges plus the inline restart warning already say it
  // better and only for the fields actually being edited.
  const visibleWarnings = settings.warnings.filter((w) => !w.startsWith('Capture is running —'))

  // The one resize warning with a remedy this app can perform itself, which is why the sidecar
  // sends it as an entry rather than as another line in the list above: the sentence and the
  // button have to travel together, and the button needs the weight and the mode.
  // `?? null` rather than the field alone: a payload without it (an older sidecar, a partial
  // fixture) has to read as "nothing to record", which costs a button — whereas `undefined`
  // reaching the render below costs the whole panel.
  const unrecorded = settings.unrecorded_resize_mode ?? null
  // Whether the record landed is read off the *weights list*, not remembered here. A weight that
  // had no requirement now has one — the same fact the server would report, and one that cannot
  // go stale the way a local "just recorded" flag would.
  const recorded =
    unrecorded !== null ? installed.find((m) => m.value === unrecorded.model) : undefined
  const requirementRecorded = recorded?.resize_mode != null

  // The remote probe's geometry, as something the render can switch on. Three states, and only one
  // of them is a problem: the workflow agreeing with what was sent, re-framing it, or saying
  // nothing at all — the last being the one that must not read as agreement, since the sidecar
  // then *assumes* the coordinates are relative to what it sent.
  //
  // `native` reports neither size (its geometry is a settings fact, already shown as
  // `resize_mode_resolved`), and an unreachable backend reports nothing to compare, so both fall
  // through to no line at all rather than an empty one.
  const probeGeometry = ((): ProbeGeometry | null => {
    if (!probeResult?.reachable || !probeResult.sent_size) return null
    const sent = probeResult.sent_size
    const reported = probeResult.reported_size
    if (!reported)
      return { state: 'unreported', sent, reported: null, aspectChanged: false } as const
    return {
      state: reported[0] === sent[0] && reported[1] === sent[1] ? 'same' : 'resized',
      sent,
      reported,
      // Cross-multiplied rather than divided: an aspect comparison through floats would call
      // 640x360 and 6x3.375 different sizes of the same shape.
      aspectChanged: reported[0] * sent[1] !== reported[1] * sent[0]
    }
  })()

  const handleSave = async (): Promise<void> => {
    if (!canSave) return
    await update(draft)
    setJustSaved(true)
  }

  // Most settings are restart-required, so the common path is: stop, save,
  // start again — three actions across two views. Doing it in one keeps the
  // edit, which is why stopCapture must not re-read settings.
  const handleSaveAndRestart = async (): Promise<void> => {
    if (pendingFields.length === 0) return
    await stopCapture()
    await update(draft)
    setJustSaved(true)
    await startCapture()
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
    // What the selected weights were trained to expect, when a record beside them says so.
    // The comparison uses the *pending* value, because that is what a save would apply, and the
    // rule itself lives in `lib/resizeMode.ts` so this and the Live view's banner cannot drift
    // apart about what "mismatched" means. `auto` cannot be the wrong choice (`auto_resolves_to`
    // honours the record); what is left for this to catch is an explicit value that contradicts
    // the weights, which is the only way to get the geometry wrong from the settings.
    const record = installed.find((m) => m.value === String(value))
    const mismatch = resizeModeMismatch(String(valueOf('resize_mode')), String(value), installed)

    if (field.key === 'device') {
      const gpuAvailable = systemInfo?.accelerator === 'cuda'
      const isCpu = value === 'cpu' || !gpuAvailable
      return (
        <div className="admin-field" key={field.key}>
          <div className="admin-field-label">
            <span className="admin-field-labeltext">{field.label}</span>
            {!isRestartField && <span className="badge live">applies instantly</span>}
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
          {/* Only the exceptions are badged. 14 of 16 fields are
              restart-required, so badging those made the badge meaningless;
              the few that apply instantly are the surprising ones. The
              restart-required set is still enforced server-side and surfaced
              in the warning above the actions. */}
          {!isRestartField && <span className="badge live">applies instantly</span>}
        </div>
        {field.key === 'camera_index' && cameras.length === 0 && running ? (
          // Scanning opens every device, so the sidecar refuses while capture
          // holds one. Say that, rather than silently degrading to a bare
          // index box the user has to guess at.
          <div className="camera-picker" data-testid="camera-locked">
            <input
              id={field.key}
              type="number"
              value={value}
              min={field.min}
              max={field.max}
              onChange={(e) => {
                const n = e.target.valueAsNumber
                if (!Number.isNaN(n)) setField(field.key, n)
              }}
            />
            <span className="field-hint">Stop capture to detect camera names.</span>
          </div>
        ) : field.key === 'camera_index' && cameras.length === 0 && camerasLoading ? (
          // Opening every device is slow (~30 s — the StreamCam alone takes
          // ~28 s to open and switch mode), so say so rather than showing a
          // bare index box that silently becomes a dropdown later.
          <div className="camera-picker" data-testid="camera-scanning">
            <Spinner />
            <span className="field-hint">Detecting cameras…</span>
          </div>
        ) : field.key === 'camera_index' && cameras.length > 0 ? (
          <div className="camera-picker">
            <select
              id={field.key}
              value={String(value)}
              onChange={(e) => setField(field.key, Number(e.target.value))}
              data-testid="camera-select"
            >
              {/* A saved index with no matching device must stay selectable,
                  or the form would silently rewrite the user's setting. */}
              {!cameras.some((c) => c.index === Number(value)) && (
                <option value={String(value)}>{`${value} — not detected`}</option>
              )}
              {cameras.map((c) => (
                <option key={c.index} value={c.index}>
                  {`${c.index} — ${c.name}`}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="btn-outline btn-small"
              onClick={() => void refreshCameras(true)}
              disabled={running}
              data-testid="rescan-cameras"
              title={running ? 'Stop capture to rescan' : 'Re-detect connected cameras'}
            >
              Rescan
            </button>
          </div>
        ) : field.type === 'text' ? (
          <input
            id={field.key}
            type="text"
            value={String(value)}
            onChange={(e) => setField(field.key, e.target.value)}
          />
        ) : field.type === 'list' ? (
          <input
            id={field.key}
            type="text"
            value={listText[field.key] ?? (Array.isArray(value) ? value.join(', ') : String(value))}
            placeholder="e.g. bottle, cup"
            onChange={(e) => {
              setListText((prev) => ({ ...prev, [field.key]: e.target.value }))
              setField(field.key, parseList(e.target.value))
            }}
          />
        ) : field.type === 'select' ? (
          <select
            id={field.key}
            value={String(value)}
            onChange={(e) => setField(field.key, e.target.value)}
          >
            {(field.key === 'active_model' ? modelOptions : (field.options ?? [])).map((opt) => (
              <option key={opt} value={opt}>
                {EXPERIMENTAL_MODELS.includes(opt)
                  ? `${opt} (experimental)`
                  : (MODEL_LABELS[opt] ?? opt)}
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
        {field.key === 'camera_index' && selectedCamera && (
          <p className="field-hint" data-testid="camera-resolution">
            Opens at {selectedCamera.width}×{selectedCamera.height} — check this matches the camera
            you expect.
          </p>
        )}
        {field.key === 'active_model' && MODEL_SPEC_HINTS[String(value)] && (
          <p className="field-hint experimental" data-testid="model-spec-hint">
            {MODEL_SPEC_HINTS[String(value)]}
          </p>
        )}
        {field.key === 'active_model' && mismatch && (
          // Only reachable by overriding the record by hand now, so the fix is named rather
          // than implied: the operator has said something the weights disagree with, and the
          // way back is the value that defers to them.
          <p className="field-hint mismatch" data-testid="model-resize-mismatch">
            {`These weights need `}
            <code>resize_mode: {mismatch.required}</code>
            {` — the form has ${mismatch.mode}, which presents every object at the wrong scale.`}
          </p>
        )}
        {field.key === 'active_model' &&
          record &&
          record.recorded &&
          (record.validation.length > 0 ? (
            // What the model *scored*, beside what it needs. The record is the only place this
            // can come from: a measurement lives in a terminal log otherwise, and the operator
            // choosing between generations is the person who has to read it.
            <div className="model-validation" data-testid="model-validation">
              <p className="installed-models-title">What these weights scored</p>
              {record.validation.map((block) => {
                const measured = block.per_class.filter((c) => c.recall !== null)
                const below = measured.filter((c) => (c.recall as number) < block.floor)
                return (
                  <div
                    className="model-validation-split"
                    key={block.split}
                    data-testid={`model-validation-${block.split}`}
                  >
                    <p className="model-validation-head">
                      <code>{block.split}</code>
                      {` split, floor ${block.floor.toFixed(2)} — `}
                      {below.length === 0
                        ? `${measured.length} measured, all at or above the floor`
                        : `${below.length} of ${measured.length} below the floor`}
                    </p>
                    {Object.keys(block.aggregates).length > 0 && (
                      <p
                        className="model-validation-aggregates"
                        data-testid="model-validation-metrics"
                      >
                        {Object.entries(block.aggregates)
                          .map(([name, value]) => `${name} ${value.toFixed(3)}`)
                          .join(' · ')}
                      </p>
                    )}
                    <ul className="model-validation-classes">
                      {block.per_class.map((c) => (
                        <li
                          key={c.name}
                          className={
                            c.recall === null
                              ? 'unmeasured'
                              : c.recall < block.floor
                                ? 'below'
                                : 'ok'
                          }
                          data-testid={`model-validation-class-${c.name}`}
                        >
                          {c.name}{' '}
                          {c.recall === null
                            ? // Named rather than skipped: ultralytics answers 0.000 here, and a
                              // zero would send the operator after images of a class the split
                              // simply never asked about.
                              '— not measured (the split holds no instances of it)'
                            : `${c.recall.toFixed(3)} (n=${c.instances})`}
                        </li>
                      ))}
                    </ul>
                    {/* The same recall split by distance. The split above mixes all three, so a
                        class whose frames happen to be mostly `close` can read as healthy while
                        missing the item at `far` — and `far` is the bucket this dataset exists
                        to fix, which makes this the one number the per-class list cannot state.
                        Rows come from the split's own classes, so a class a distance never
                        scored still appears, as a dash: its absence there *is* the finding. */}
                    {block.per_distance.length > 0 && (
                      <div className="model-validation-distances">
                        <p className="model-validation-subhead">
                          By distance — a class can clear the floor on the split while failing it up
                          close or far away.
                        </p>
                        <table
                          className="model-validation-grid"
                          data-testid="model-validation-by-distance"
                        >
                          <thead>
                            <tr>
                              <th>Class</th>
                              {block.per_distance.map((d) => (
                                <th key={d.distance}>
                                  {d.distance}
                                  <span className="model-validation-images">{d.images} img</span>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {block.per_class.map((c) => (
                              <tr key={c.name}>
                                <td>{c.name}</td>
                                {block.per_distance.map((d) => {
                                  const cell = d.per_class.find((x) => x.name === c.name)
                                  // Two different absences, one rendering: a distance that
                                  // scored this class and held no instances of it, and a
                                  // distance with no frames of it at all. Either way it is
                                  // not a miss, and `0.000` would say it was.
                                  const state =
                                    cell === undefined || cell.recall === null
                                      ? 'unmeasured'
                                      : cell.recall < block.floor
                                        ? 'below'
                                        : 'ok'
                                  return (
                                    <td
                                      key={d.distance}
                                      className={state}
                                      data-testid={`model-validation-${d.distance}-${c.name}`}
                                    >
                                      {cell === undefined || cell.recall === null
                                        ? '—'
                                        : cell.recall.toFixed(3)}
                                    </td>
                                  )
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {(() => {
                          // Derived here rather than stored, for the same reason the split's own
                          // count is: a recorded list of misses could disagree with the numbers
                          // printed beside it, and nothing could say which of the two was right.
                          const misses = block.per_distance.flatMap((d) =>
                            d.per_class
                              .filter((c) => c.recall !== null && c.recall < block.floor)
                              .map((c) => `${d.distance} ${c.name} ${c.recall!.toFixed(3)}`)
                          )
                          return misses.length > 0 ? (
                            <p
                              className="model-validation-distance-misses"
                              data-testid="model-validation-distance-misses"
                            >
                              Below the floor at a distance: {misses.join(' · ')}
                            </p>
                          ) : (
                            <p
                              className="model-validation-distance-misses"
                              data-testid="model-validation-distance-misses"
                            >
                              No class is below the floor at any distance — including the ones the
                              mean over all distances would have hidden.
                            </p>
                          )
                        })()}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ) : (
            // Installed by the tool, `--val` not run. Gated on `recorded` rather than on the
            // empty list alone: a weight copied into `models/` by hand has no numbers either,
            // and telling its owner to run `--val` would be advice about someone else's file.
            <p className="field-hint" data-testid="model-validation-missing">
              No measured score is recorded beside these weights — <code>train_model.py --val</code>{' '}
              writes the per-class recall that appears here.
            </p>
          ))}
        {field.key === 'active_model' && installed.length > 0 && (
          <div className="installed-models" data-testid="installed-models">
            <p className="installed-models-title">Weights on disk</p>
            <ul>
              {installed.map((m) => (
                <li key={m.value} data-testid={`installed-model-${m.value}`}>
                  <code>{m.value}</code>
                  {m.resize_mode ? (
                    <>
                      {' needs '}
                      <code>resize_mode: {m.resize_mode}</code>
                      {' — '}
                      <code>auto</code>
                      {' uses it'}
                    </>
                  ) : (
                    // Two different unknowns must not read as one. Nothing was recorded, so
                    // `auto` falls back to a rule about the file format - which may be the
                    // wrong guess for a hand-copied weight, and a guess is the one thing this
                    // block must not present as a fact.
                    <>
                      {` — no recorded requirement; `}
                      <code>auto</code>
                      {` gives ${m.auto_resolves_to}`}
                    </>
                  )}
                  {m.source ? <span className="installed-models-source"> ({m.source})</span> : null}
                  {m.class_names.length > 0 && (
                    <span className="installed-models-classes">{` — ${m.class_names.length} classes`}</span>
                  )}
                  {/* What those classes *are*, which the count cannot say — and the reason the
                      sidecar records them beside the weight. A model trained from a project whose
                      class list was split by distance predicts three labels per product; the
                      filename and the checkpoint say nothing about it, so without this the only
                      way to notice is to run it and watch one item log three times. The sidecar
                      judged the names against the roster because it is the roster's owner, so its
                      sentences are rendered rather than re-derived here. */}
                  {m.class_warnings.map((warning) => (
                    <p
                      className="admin-error"
                      data-testid={`installed-model-class-warning-${m.value}`}
                      key={warning.slice(0, 40)}
                    >
                      ⚠ {warning}
                    </p>
                  ))}
                </li>
              ))}
            </ul>
          </div>
        )}
        <p className="field-hint">{field.hint}</p>
      </div>
    )
  }

  const stopButton = (testId: string): JSX.Element => (
    <button
      className="btn-outline btn-small"
      disabled={stopping}
      onClick={() => void stopCapture()}
      data-testid={testId}
    >
      {stopping ? <Spinner /> : null} Stop capture
    </button>
  )

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

      <section className="admin-dataset" data-testid="dataset-progress">
        <h4>
          Dataset labeling
          {dataset.status?.available === true && (
            <button
              className="admin-dataset-refresh"
              disabled={dataset.loading}
              onClick={() => void dataset.refresh()}
              data-testid="refresh-dataset"
            >
              {dataset.loading ? 'Refreshing…' : 'Refresh'}
            </button>
          )}
        </h4>
        {dataset.status === null ? (
          <p data-testid="dataset-pending">
            {dataset.loading ? 'Reading labeling snapshot…' : 'Labeling snapshot unavailable.'}
          </p>
        ) : !dataset.status.available ? (
          <p className="admin-hint" data-testid="dataset-unavailable">
            No snapshot yet. Run <code>sidecar/tools/label_progress.py</code> to write one — the
            sidecar reads that file, so this needs no API key and no network.
          </p>
        ) : (
          <>
            <p className="admin-dataset-summary" data-testid="dataset-summary">
              <strong>
                {dataset.status.decided} / {dataset.status.total}
              </strong>{' '}
              labeled ({dataset.status.percent.toFixed(1)}%) in {dataset.status.project}
            </p>
            <p className="admin-hint" data-testid="dataset-freshness">
              snapshot from {dataset.status.generated_at ?? 'an unknown time'} (
              {describeAge(dataset.status.age_seconds)})
              {dataset.status.null_annotations > 0 &&
                ` · ${dataset.status.null_annotations} null (background)`}
              {dataset.status.mismatches > 0 &&
                ` · ${dataset.status.mismatches} class mismatch(es)`}
            </p>
            {/* The worklist rather than a per-class table. A cell is one class at one
                distance, so this carries a whole axis the class view could not - and the
                one that stalls a project is a distance bucket nobody opened, not a class
                nobody started. Ranked and with finished cells dropped, so the top row is
                the next thing to open in Roboflow and the list shrinks as it is worked
                through, instead of showing 242/242 rows that are no longer work. */}
            {backlog.length > 0 && (
              <p className="admin-hint" data-testid="dataset-backlog-summary">
                Worklist — {backlog.length} cell(s) left, biggest first. A cell is one class at one
                distance, which is what gets labeled in a sitting.
              </p>
            )}
            {backlog.length > 0 ? (
              <ol className="admin-backlog" data-testid="dataset-backlog">
                {backlog.map((row, i) => (
                  <li
                    key={`${row.slug}|${row.distance}`}
                    className={
                      row.background ? 'admin-backlog-row is-background' : 'admin-backlog-row'
                    }
                    data-testid="dataset-backlog-row"
                  >
                    <span className="admin-backlog-rank">{i + 1}</span>
                    <span className="admin-backlog-cell">
                      <span className="admin-backlog-name">{row.name}</span>
                      {/* The distance is what makes this a cell rather than a class; the
                          background row has none, and says so in words instead of leaving
                          the slot looking broken. */}
                      <span className="admin-backlog-where">
                        {row.background ? 'background frames' : row.distance}
                      </span>
                    </span>
                    <span className="admin-backlog-count">
                      <span className="admin-backlog-left" data-testid="dataset-backlog-left">
                        {row.remaining}
                      </span>{' '}
                      left
                      <span className="admin-backlog-of">
                        {row.decided}/{row.total}
                      </span>
                    </span>
                    <span className="admin-backlog-bar" aria-hidden="true">
                      <span
                        className="admin-backlog-fill"
                        style={{
                          width: `${row.total > 0 ? Math.round((row.decided / row.total) * 100) : 0}%`
                        }}
                      />
                    </span>
                    {row.background && (
                      <span className="admin-backlog-null" data-testid="dataset-backlog-null">
                        mark each one <strong>null</strong> (N) — do not draw
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            ) : (
              <p className="admin-hint" data-testid="dataset-backlog-done">
                Nothing left to label — every cell is complete.
              </p>
            )}
            <p className="admin-hint" data-testid="dataset-distance">
              by distance:{' '}
              {Object.entries(dataset.status.by_distance)
                .map(([distance, counts]) => `${distance} ${counts[0]}/${counts[1]}`)
                .join(' · ')}
            </p>
            {/* Which split each session's frames ended up in, because it decides how the
                test number may be quoted. Hidden when the snapshot predates the field, so
                an old snapshot renders as it did before rather than as an empty table. */}
            {datasetSessions.length > 0 && (
              <div className="admin-dataset-sessions" data-testid="dataset-sessions">
                <table className="admin-dataset-table" data-testid="dataset-session-rows">
                  <thead>
                    <tr>
                      <th>Session</th>
                      <th>train</th>
                      <th>valid</th>
                      <th>test</th>
                      <th>Splits</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasetSessions.map((s) => (
                      <tr
                        key={s.name}
                        className={s.train > 0 && s.test > 0 ? 'dataset-session-shared' : undefined}
                      >
                        <td>{s.name}</td>
                        <td>{s.train || '—'}</td>
                        <td>{s.valid || '—'}</td>
                        <td>{s.test || '—'}</td>
                        <td>{s.splits}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="admin-hint" data-testid="dataset-session-verdict">
                  {testSessions.length > 0 ? (
                    <>
                      {testSessions.map((s) => s.name).join(', ')} appear(s) in both train and test,
                      so the test number is held-out frames of a session the model trained on, not
                      an unseen-session estimate. A held-out session that re-shoots covered cells is
                      what fixes it (Tier D).
                    </>
                  ) : (
                    <>
                      No session sits on both sides of the train/test line, so test is genuinely an
                      unseen capture session.
                    </>
                  )}
                </p>
              </div>
            )}
            {/* Tier A is deliberately a separate block from the class table: this is the
                capture gap, so a cell sitting here at 0/40 is waiting for the camera, not
                for a label. Folding it into "0% labeled" is what makes a capture session
                look like a labeling backlog. */}
            {dataset.status.tier_a_target > 0 && (
              <div className="admin-dataset-tier-a" data-testid="dataset-tier-a">
                <p className="admin-hint" data-testid="dataset-tier-a-summary">
                  {dataset.status.tier_a_remaining > 0 ? (
                    <>
                      capture gap (Tier A): <strong>{dataset.status.tier_a_remaining}</strong> of{' '}
                      {dataset.status.tier_a_target} images still to shoot across{' '}
                      {dataset.status.tier_a_cells_under_target} cell(s)
                    </>
                  ) : (
                    <>
                      Tier A complete: all {dataset.status.tier_a_cells.length} cell(s) have reached
                      their target ({dataset.status.tier_a_target} images)
                    </>
                  )}
                </p>
                {dataset.status.tier_a_remaining > 0 && (
                  <ul className="admin-dataset-gaps" data-testid="dataset-tier-a-cells">
                    {dataset.status.tier_a_cells
                      .filter((c) => c.remaining > 0)
                      .map((c) => (
                        <li key={`${c.slug}|${c.distance}`}>
                          <span className="dataset-gap-count">{c.remaining} to shoot</span>{' '}
                          <span className="dataset-gap-class">
                            {datasetClassNames.get(c.slug) ?? c.slug}
                          </span>{' '}
                          @ {c.distance}{' '}
                          <span className="dataset-gap-have">
                            ({c.have}/{c.target})
                          </span>
                        </li>
                      ))}
                  </ul>
                )}
              </div>
            )}
          </>
        )}
        {dataset.error !== null && <p className="admin-warning">{dataset.error}</p>}
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

      <section className="admin-backend" data-testid="backend-picker">
        <h4>Detector backend</h4>
        <div className="backend-toggle">
          {ALLOWED_BACKENDS.map((b) => (
            <label key={b} className={selectedBackend === b ? 'selected' : ''}>
              <input
                type="radio"
                name="detector_backend"
                value={b}
                checked={selectedBackend === b}
                onChange={() => setField('detector_backend', b)}
              />
              {BACKEND_LABELS[b] ?? b}
            </label>
          ))}
        </div>
        <p className="field-hint">{BACKEND_HINTS[selectedBackend]}</p>

        {backendIsRemote && !settings.roboflow_api_key_present && (
          <p className="admin-warning" data-testid="missing-api-key">
            No Roboflow API key found. Add <code>ROBOFLOW_API_KEY</code> to{' '}
            <code>sidecar/.env</code> (see <code>.env.example</code>) — capture will not start
            without it.
          </p>
        )}

        {backendIsRemote &&
          Number(valueOf('track_expiry_s')) < minTrackExpiryS(selectedBackend) && (
            <p className="admin-warning" data-testid="expiry-warning">
              Track expiry is {String(valueOf('track_expiry_s'))}s. A slow API round trip can
              outlast that and log one item twice — use at least {minTrackExpiryS(selectedBackend)}s
              with {BACKEND_LABELS[selectedBackend]}.
            </p>
          )}

        <div className="backend-actions">
          <button
            className="btn-outline"
            disabled={probing || pendingFields.length > 0}
            onClick={() => void probe()}
            data-testid="test-connection"
          >
            {probing ? <Spinner /> : null} Test connection
          </button>
          {pendingFields.length > 0 && (
            <span className="field-hint">Save your changes before testing.</span>
          )}
        </div>

        {probeResult && (
          <p
            className={probeResult.reachable ? 'probe-ok' : 'admin-error'}
            data-testid="probe-result"
          >
            {probeResult.reachable ? '✓' : '✗'} {probeResult.detail}
            {probeResult.latency_ms !== null && ` — ${probeResult.latency_ms} ms`}
            {probeResult.class_names.length > 0 && ` — ${probeResult.class_names.length} classes`}
          </p>
        )}

        {/* What those classes *are*, which the count above cannot say. The sidecar judged them
            against the roster of these weights' own generation because it is the only process
            that has them (they come off the loaded model), so this renders its sentences rather
            than re-deriving them: one bad class list has one explanation, and a second copy here
            could contradict it. */}
        {probeResult?.reachable &&
          probeResult.class_warnings.map((warning) => (
            <p className="admin-error" data-testid="probe-class-warning" key={warning.slice(0, 40)}>
              ⚠ {warning}
            </p>
          ))}

        {/* The remote geometry, reported rather than judged: two sizes and a silence, and the
            silence must not read as agreement. Nothing here can be derived in the renderer — the
            workflow's canvas is a fact about a round trip only the sidecar saw — so this renders
            what the probe found and says what each state does and does not establish. */}
        {probeGeometry ? (
          <p
            className={`field-hint ${probeGeometry.state === 'resized' ? 'mismatch' : ''}`}
            data-testid="probe-geometry"
            data-state={probeGeometry.state}
          >
            {probeGeometry.state === 'same' && (
              <>
                {`Workflow geometry: ${fmtSize(probeGeometry.reported)} — the size the app sent, so `}
                {'the workflow passes the frame through. Whatever geometry these weights apply '}
                {'inside it is not visible from here.'}
              </>
            )}
            {probeGeometry.state === 'resized' && (
              <>
                {`Workflow geometry: ${fmtSize(probeGeometry.reported)}, but the app sends `}
                {fmtSize(probeGeometry.sent)}
                {'. The workflow re-frames the image before the model sees it, so '}
                <code>remote_infer_size</code>
                {' is not the geometry these weights run at'}
                {probeGeometry.aspectChanged
                  ? '. The aspect ratio changes too, so it is stretching or padding — check that ' +
                    'matches how these weights were trained: padding presents small and far ' +
                    'objects well under their training scale.'
                  : ' (the aspect ratio is kept, so objects keep their shape — the model sees a ' +
                    'different number of pixels than the app sent).'}
              </>
            )}
            {probeGeometry.state === 'unreported' && (
              <>
                {'The workflow reported no image size for the '}
                {fmtSize(probeGeometry.sent)}
                {
                  ' frame sent, so the app assumes its detections are relative to that size. Nothing '
                }
                {
                  'here confirms it — a workflow that re-frames without saying so puts boxes in the '
                }
                {'wrong place, and this is the state that cannot rule that out.'}
              </>
            )}
          </p>
        ) : null}
      </section>

      <div className="admin-groups">
        {visibleGroups.map((group) => (
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

      {(visibleWarnings.length > 0 || unrecorded !== null) && (
        <ul className="admin-warnings" data-testid="server-warnings">
          {unrecorded !== null && (
            <li
              className={requirementRecorded ? 'resolved' : 'actionable'}
              data-testid="unrecorded-resize-mode"
            >
              {requirementRecorded ? (
                // The remedy, confirmed — and confirmed from the sidecar's own answer rather than
                // from the click, so a write that landed somewhere else could not look like this.
                //
                // The sentence is shared with the Live view's copy of the same banner, which is
                // the other place this button's outcome has to be describable: one write, one
                // description. `recorded?.resize_mode` is the mode the *listing* now carries,
                // never the one the click asked for.
                <span>{recordedConfirmation(String(recorded?.resize_mode))}</span>
              ) : (
                <>
                  <span>{unrecorded.warning}</span>
                  {/* The remedy sentence comes from the sidecar and the Live view renders the same
                      one without a button under it, so this is the same advice in both views
                      rather than two descriptions of one situation. */}
                  <span>{unrecorded.remedy}</span>
                  <button
                    className="btn-outline btn-small"
                    disabled={recording}
                    onClick={() => void recordRequirement(unrecorded.model, unrecorded.resize_mode)}
                    data-testid="record-resize-mode"
                  >
                    {recording ? <Spinner /> : null} Record it now: {unrecorded.resize_mode}
                  </button>
                </>
              )}
            </li>
          )}
          {visibleWarnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      {blockedByRunning && (
        <p className="admin-warning" data-testid="restart-warning">
          These need a capture restart: {pendingRestartFields.join(', ')}. Use{' '}
          <strong>Save &amp; restart capture</strong> below.
        </p>
      )}

      {error && (
        <p className="admin-error" data-testid="admin-error">
          {error}
        </p>
      )}

      <div className="admin-actions" data-testid="admin-actions">
        <span className="admin-actions-status" data-testid="pending-count">
          {pendingFields.length > 0
            ? `${pendingFields.length} unsaved change${pendingFields.length === 1 ? '' : 's'}`
            : justSaved
              ? 'Saved'
              : 'No changes'}
        </span>
        {running && stopButton('stop-capture-inline')}
        {blockedByRunning && (
          <button
            className="btn-primary btn-small"
            disabled={stopping || saving}
            onClick={() => void handleSaveAndRestart()}
            data-testid="save-and-restart"
          >
            {stopping || saving ? <Spinner /> : null} Save &amp; restart capture
          </button>
        )}
        <button
          className="btn-primary"
          disabled={!canSave || saving}
          onClick={() => void handleSave()}
          data-testid="save-settings"
        >
          {saving ? <Spinner /> : null} Save
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
