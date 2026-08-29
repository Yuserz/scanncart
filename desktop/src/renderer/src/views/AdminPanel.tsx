import { useState, type JSX } from 'react'
import { useSidecarSettings, type SettingsDeps } from '../hooks/useSidecarSettings'
import type {
  SettingsPayload,
  SettingsResponse,
  SettingsUpdate,
  SystemInfoResponse
} from '../lib/api'
import {
  ALLOWED_BACKENDS,
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
    refresh,
    probe,
    probing,
    probeResult,
    cameras,
    refreshCameras,
    camerasLoading,
    stopCapture,
    startCapture,
    stopping,
    cameraQuality
  } = useSidecarSettings(port, deps)

  // Holds only *unsaved* edits; reset whenever the server-confirmed settings
  // object changes identity (initial load, or after a successful
  // save/preset/restore-defaults). Adjusted during render rather than in a
  // useEffect, per React's guidance on resetting state when a prop changes.
  const [settingsAtLastReset, setSettingsAtLastReset] = useState<SettingsResponse | null>(null)
  const [draft, setDraft] = useState<SettingsUpdate>({})
  const [justSaved, setJustSaved] = useState(false)
  if (settings !== settingsAtLastReset) {
    setSettingsAtLastReset(settings)
    setDraft({})
  }

  const running = captureState === 'running'

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

  const valueOf = (key: keyof SettingsPayload): string | number => {
    const draftVal = draft[key]
    return (draftVal !== undefined ? draftVal : settings[key]) as string | number
  }

  const setField = (key: keyof SettingsPayload, value: string | number): void => {
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
    (g) => g.label !== 'Roboflow API backends' || backendIsRemote
  )
  const canSave = pendingFields.length > 0 && !blockedByRunning
  const selectedCamera = cameras.find((c) => c.index === Number(valueOf('camera_index')))

  // The sidecar emits one warning that lists every restart-required field by
  // name whenever capture runs. It is 16 items of comma-separated prose, and
  // the per-field badges plus the inline restart warning already say it
  // better and only for the fields actually being edited.
  const visibleWarnings = settings.warnings.filter((w) => !w.startsWith('Capture is running —'))

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
        ) : field.type === 'select' ? (
          <select
            id={field.key}
            value={String(value)}
            onChange={(e) => setField(field.key, e.target.value)}
          >
            {field.options?.map((opt) => (
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
      </section>

      {cameraQuality?.available && (
        <section className="camera-quality" data-testid="camera-quality">
          <h4>Live image</h4>
          <div className="quality-row">
            {[
              [
                'Brightness',
                cameraQuality.brightness,
                cameraQuality.verdicts.brightness,
                '110–160'
              ],
              [
                'Sharpness',
                cameraQuality.sharpness,
                cameraQuality.verdicts.sharpness,
                'higher is sharper'
              ],
              [
                'Capture fps',
                cameraQuality.capture_fps,
                cameraQuality.verdicts.capture_fps,
                `≥ ${Math.round(cameraQuality.target_fps * 0.8)}`
              ]
            ].map(([label, value, verdict, hint]) => (
              <div key={String(label)} className="quality-metric">
                <span className="quality-label">{label}</span>
                <span
                  className={verdict === 'ok' ? 'quality-value' : 'quality-value bad'}
                  data-testid={verdict === 'ok' ? 'quality-ok' : 'quality-low'}
                >
                  {value}
                </span>
                <span className="field-hint">{hint}</span>
              </div>
            ))}
          </div>
        </section>
      )}

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

      {visibleWarnings.length > 0 && (
        <ul className="admin-warnings" data-testid="server-warnings">
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
