import { useEffect, useRef, useState, type JSX } from 'react'
import { useSidecarSettings, type SettingsDeps } from '../hooks/useSidecarSettings'
import type { SettingsPayload, SettingsUpdate } from '../lib/api'
import { SETTINGS_FIELDS, SETTINGS_GROUPS, type FieldMeta } from '../lib/settingsFields'
import { Spinner } from './Spinner'
import './CameraTuning.css'

export interface CameraTuningProps {
  port: number
  running: boolean
  start: () => Promise<void>
  stop: () => Promise<void>
  cameraName: string
  deps?: SettingsDeps
  // Trailing-edge debounce for slider writes; overridden to 0 in tests.
  debounceMs?: number
}

// Which measured control support gates which settings field. `gain` is
// measured by calibration but no setting exposes it, so it is absent here.
const SUPPORT_KEY: Partial<Record<keyof SettingsPayload, 'brightness' | 'exposure' | 'focus'>> = {
  camera_brightness: 'brightness',
  camera_exposure: 'exposure',
  camera_focus: 'focus'
}

const LIVE_GROUPS = SETTINGS_GROUPS.filter((g) => g.home === 'live')

export function CameraTuning({
  port,
  running,
  cameraName,
  deps,
  debounceMs: debounceMsProp
}: CameraTuningProps): JSX.Element {
  // LiveView owns capture state and passes it as `running`, so this instance
  // must not poll health too. It has no use for the camera list either, and
  // enumerating opens every device. Quality still reads — see useSidecarSettings.
  const {
    settings,
    storedProfile,
    cameraQuality,
    loading,
    liveUpdate,
    save,
    savedSettings,
    saving
  } = useSidecarSettings(port, {
    ...deps,
    pollHealth: deps?.pollHealth ?? false,
    pollCameras: deps?.pollCameras ?? false
  })
  const [open, setOpen] = useState(false)

  const debounceMs = debounceMsProp ?? 150
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  // A drag emits dozens of change events; each would otherwise be a full
  // PATCH. Trailing edge, keyed per field so two sliders do not cancel each
  // other.
  const applyDebounced = (key: keyof SettingsPayload, value: number | boolean): void => {
    const existing = timers.current.get(key)
    if (existing) clearTimeout(existing)
    timers.current.set(
      key,
      setTimeout(() => {
        timers.current.delete(key)
        void liveUpdate({ [key]: value } as SettingsUpdate)
      }, debounceMs)
    )
  }

  useEffect(
    () => () => {
      for (const t of timers.current.values()) clearTimeout(t)
    },
    []
  )

  // Unsaved changes are the gap between the live settings and the last
  // persisted ones — never a local draft, because a live PATCH returns a
  // fresh settings object that would otherwise look committed.
  const dirtyKeys =
    settings && savedSettings
      ? LIVE_GROUPS.flatMap((g) => g.keys).filter((k) => settings[k] !== savedSettings[k])
      : []

  const valueOf = (key: keyof SettingsPayload): string | number | boolean =>
    (settings?.[key] ?? '') as string | number | boolean

  // A control is off-limits when calibration measured the device ignoring it.
  // With no profile there is no evidence either way, so everything stays
  // enabled — an unsupported control simply does nothing.
  const unsupported = (key: keyof SettingsPayload): boolean => {
    const support = SUPPORT_KEY[key]
    if (!support || !storedProfile) return false
    return storedProfile.controls[support] === false
  }

  const renderField = (field: FieldMeta): JSX.Element => {
    // Writing a focus value while autofocus is on is meaningless: the device
    // immediately hunts away from it.
    const autofocusOn = settings?.camera_autofocus === true
    const disabled = unsupported(field.key) || (field.key === 'camera_focus' && autofocusOn)

    return (
      <div className="tuning-field" key={field.key}>
        <label htmlFor={`tune-${field.key}`}>{field.label}</label>
        {field.type === 'boolean' ? (
          <input
            id={`tune-${field.key}`}
            type="checkbox"
            checked={valueOf(field.key) === true}
            disabled={disabled}
            onChange={(e) => applyDebounced(field.key, e.target.checked)}
          />
        ) : (
          <input
            id={`tune-${field.key}`}
            type="range"
            value={Number(valueOf(field.key)) || 0}
            min={field.min}
            max={field.max}
            step={field.step}
            disabled={disabled}
            onChange={(e) => {
              const n = e.target.valueAsNumber
              if (!Number.isNaN(n)) applyDebounced(field.key, n)
            }}
          />
        )}
        {unsupported(field.key) && (
          <p className="field-hint" data-testid="unsupported-control">
            This camera ignored it during calibration.
          </p>
        )}
        <p className="field-hint">{field.hint}</p>
      </div>
    )
  }

  return (
    <div className="card tuning-card" data-testid="camera-tuning">
      <h4>
        <button
          type="button"
          className="tuning-toggle"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? '▾' : '▸'} Camera tuning
        </button>
        <span className="tuning-camera">{cameraName}</span>
      </h4>

      {loading && !settings && (
        <p className="field-hint">
          <Spinner /> Loading…
        </p>
      )}

      {cameraQuality?.available && (
        <div className="quality-row" data-testid="tuning-quality">
          {[
            ['Brightness', cameraQuality.brightness, cameraQuality.verdicts.brightness],
            ['Sharpness', cameraQuality.sharpness, cameraQuality.verdicts.sharpness],
            ['Capture fps', cameraQuality.capture_fps, cameraQuality.verdicts.capture_fps]
          ].map(([label, value, verdict]) => {
            const failing = verdict !== 'ok'
            return (
              <div key={String(label)} className="quality-metric">
                <span className="quality-label">{label}</span>
                <span
                  className={failing ? 'quality-value bad' : 'quality-value'}
                  data-testid={failing ? 'quality-low' : 'quality-ok'}
                >
                  {/* Colour alone fails a colourblind operator reading a
                      readout whose whole job is flagging a bad number. */}
                  {failing && (
                    <span className="quality-flag" aria-hidden="true">
                      ▲{' '}
                    </span>
                  )}
                  {value}
                  {failing && <span className="sr-only"> — outside the expected range</span>}
                </span>
              </div>
            )
          })}
        </div>
      )}

      {settings &&
        LIVE_GROUPS.map((group) => (
          <section className="tuning-group" key={group.label} hidden={!open}>
            <h5>{group.label}</h5>
            {group.keys.map((key) => {
              const field = SETTINGS_FIELDS.find((f) => f.key === key)
              return field ? renderField(field) : null
            })}
          </section>
        ))}

      {dirtyKeys.length > 0 && (
        <div className="tuning-actions">
          <span data-testid="tuning-dirty">
            {dirtyKeys.length} unsaved change{dirtyKeys.length === 1 ? '' : 's'}
          </span>
          <button
            className="btn-primary btn-small"
            disabled={saving}
            data-testid="tuning-save"
            onClick={() => void save()}
          >
            {saving ? <Spinner /> : null} Save
          </button>
          <button
            className="btn-outline btn-small"
            data-testid="tuning-revert"
            onClick={() => {
              if (!savedSettings) return
              // A saved value of null means "leave the camera alone", which is
              // the default for all four controls — so on a fresh install this
              // is the common case, not an edge one. exclude_none on the
              // sidecar drops nulls from a patch, so they travel by name in
              // reset_fields instead.
              const restore = dirtyKeys.filter((k) => savedSettings[k] !== null)
              const reset = dirtyKeys.filter((k) => savedSettings[k] === null)
              const patch: SettingsUpdate = Object.fromEntries(
                restore.map((k) => [k, savedSettings[k]])
              ) as SettingsUpdate
              if (reset.length > 0) patch.reset_fields = reset
              void liveUpdate(patch)
            }}
          >
            Revert
          </button>
        </div>
      )}

      {!running && (
        <p className="field-hint" data-testid="tuning-idle">
          Start capture to see changes take effect.
        </p>
      )}
    </div>
  )
}
