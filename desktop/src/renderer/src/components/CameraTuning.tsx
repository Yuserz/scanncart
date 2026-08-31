import { useState, type JSX } from 'react'
import { useSidecarSettings, type SettingsDeps } from '../hooks/useSidecarSettings'
import type { SettingsPayload } from '../lib/api'
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

export function CameraTuning({ port, running, cameraName, deps }: CameraTuningProps): JSX.Element {
  // LiveView owns capture state and passes it as `running`, so this instance
  // must not poll health too. It has no use for the camera list either, and
  // enumerating opens every device. Quality still reads — see useSidecarSettings.
  const { settings, storedProfile, cameraQuality, loading } = useSidecarSettings(port, {
    ...deps,
    pollHealth: deps?.pollHealth ?? false,
    pollCameras: deps?.pollCameras ?? false
  })
  const [open, setOpen] = useState(false)

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
            readOnly
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
            readOnly
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

      {!running && (
        <p className="field-hint" data-testid="tuning-idle">
          Start capture to see changes take effect.
        </p>
      )}
    </div>
  )
}
