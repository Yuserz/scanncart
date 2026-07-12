import { useState, type JSX } from 'react'
import { useSidecarSettings, type SettingsDeps } from '../hooks/useSidecarSettings'
import type { SettingsPayload, SettingsResponse, SettingsUpdate } from '../lib/api'
import { SETTINGS_FIELDS } from '../lib/settingsFields'
import './AdminPanel.css'

export interface AdminPanelProps {
  port: number
  deps?: SettingsDeps
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

  return (
    <div className="admin-panel">
      <header className="admin-header">
        <h2>Admin Settings</h2>
        <span className="state" data-testid="capture-state">
          {captureState}
        </span>
      </header>

      <section className="admin-hardware" data-testid="hardware-info">
        <h3>This machine</h3>
        {systemInfo ? (
          <ul>
            <li>CPU cores: {systemInfo.cpu_count}</li>
            <li>RAM: {systemInfo.ram_gb.toFixed(1)} GB</li>
            <li>
              GPU:{' '}
              {systemInfo.accelerator === 'cuda'
                ? `${systemInfo.gpu_name ?? 'CUDA GPU'} (${systemInfo.gpu_vram_gb?.toFixed(1) ?? '?'} GB VRAM) — GPU acceleration available`
                : systemInfo.accelerator === 'integrated'
                  ? `Integrated graphics: ${systemInfo.gpu_name ?? 'unknown'} (APU) — no CUDA acceleration, runs on CPU`
                  : 'No GPU detected — CPU only'}
            </li>
          </ul>
        ) : (
          <p>Detecting hardware…</p>
        )}
      </section>

      <section className="admin-presets">
        <h3>Presets</h3>
        <div className="preset-cards">
          {presets.map((p) => (
            <div
              key={p.name}
              className={`preset-card${p.name === recommended ? ' recommended' : ''}`}
            >
              <div className="preset-card-header">
                <strong>{p.label}</strong>
                {p.name === recommended && (
                  <span className="badge">Recommended for this machine</span>
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

      <section className="admin-form">
        <h3>Settings</h3>
        {SETTINGS_FIELDS.map((field) => {
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
                    No CUDA GPU on this machine — integrated/APU can&apos;t accelerate; running on
                    CPU.
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
        })}
      </section>

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
          disabled={!canSave || saving}
          onClick={() => void handleSave()}
          data-testid="save-settings"
        >
          Save
        </button>
        <button
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
