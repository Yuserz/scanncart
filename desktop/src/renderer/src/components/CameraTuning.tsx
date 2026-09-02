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
  // Optional: the card resolves its own name from the stored profile when the
  // host does not know the device's.
  cameraName?: string
  deps?: SettingsDeps
  // Called when the card takes the camera exclusively, and again when it
  // gives it back. The host owns Start/Stop and would otherwise offer a
  // button the sidecar can only refuse.
  onCameraBusy?: (busy: boolean) => void
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
  start,
  stop,
  cameraName,
  deps,
  onCameraBusy,
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
    saving,
    calibrate,
    calibrating,
    profile,
    applyProfile,
    error
  } = useSidecarSettings(port, {
    ...deps,
    pollHealth: deps?.pollHealth ?? false,
    pollCameras: deps?.pollCameras ?? false
  })
  const [open, setOpen] = useState(false)

  // Calibrate needs a visible target, and "Camera 0" is what cameras.py
  // exists to replace. The device list is not an option here — enumerating
  // opens every device (~30 s) and the sidecar refuses it while capture holds
  // one — but the stored profile's device_key already carries the name, for
  // free, from a request the card makes anyway. The key is
  // `{name}:{index}:{W}x{H}` (camera_caps.device_key_for), so the name is
  // everything before the last two segments; a name containing a colon
  // survives that, which splitting from the left would not.
  const deviceKey = storedProfile?.device_key
  const nameFromKey = deviceKey?.split(':').slice(0, -2).join(':')
  const resolvedCameraName = settings ? nameFromKey || `Camera ${settings.camera_index}` : ''

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

  // `calibrating` covers only the sweep itself, but the sequence below awaits
  // a stop before it and a start after it. Gating on `calibrating` alone left
  // the button live through both windows, so a second click launched a whole
  // second overlapping stop/calibrate/start.
  const [busy, setBusy] = useState(false)

  // The gate is shown while capture is still running on purpose: the
  // operator frames the item using the live feed. Stopping first would ask
  // them to aim at a picture they cannot see.
  const [gateOpen, setGateOpen] = useState(false)

  // The hook records the failure in `error` and rethrows, so callers that
  // sequence around it (handleCalibrate) can react. A click handler has
  // nothing to react with, and `void` on a rejecting promise is an unhandled
  // rejection — so drop it here, after the banner already has it.
  const reported = (p: Promise<unknown>): void => {
    void p.catch(() => {})
  }

  // The sidecar refuses to calibrate while capture holds the camera, and the
  // sweep releases and reopens the device twice (~60-90 s on a StreamCam).
  // Driving the lifecycle here means the operator presses one button instead
  // of learning that constraint.
  // Tell the host across the whole sequence, not just the sweep: the camera
  // is unavailable from the moment we stop capture until we have started it
  // again.
  useEffect(() => {
    onCameraBusy?.(busy)
  }, [busy, onCameraBusy])

  const handleCalibrate = async (): Promise<void> => {
    if (busy) return
    setBusy(true)
    await stop()
    try {
      await calibrate()
    } catch {
      // Reported through the hook's `error`, which the card renders below;
      // the restart in `finally` runs either way so a failed sweep never
      // strands the feed off.
    } finally {
      await start()
      setBusy(false)
    }
  }

  // A stored profile from before the sweep existed. Its empty `recommended`
  // means "we never looked", not "this camera responded to nothing" — and
  // the fix for one is re-calibrating while the fix for the other is a
  // different camera.
  const profileIsStale = storedProfile?.sweep_version === 0

  // null means "leave the camera alone" — the default for all four camera
  // controls — so it must stay distinguishable from a real number all the way
  // to the input, never collapsing to 0.
  const numericOf = (key: keyof SettingsPayload): number | null => {
    const raw = settings?.[key]
    return raw === null || raw === undefined ? null : Number(raw)
  }

  const midpoint = (field: FieldMeta): number => ((field.min ?? 0) + (field.max ?? 0)) / 2

  const renderField = (field: FieldMeta): JSX.Element => {
    const numeric = numericOf(field.key)
    // Writing a focus value while autofocus is on is meaningless: the device
    // immediately hunts away from it.
    const autofocusOn = settings?.camera_autofocus === true
    const disabled = unsupported(field.key) || (field.key === 'camera_focus' && autofocusOn)

    return (
      <div className="tuning-field" key={field.key}>
        <div className="tuning-field-label">
          <label htmlFor={`tune-${field.key}`}>{field.label}</label>
          {/* Focusable so the hint is reachable without a pointer, and
              described-by rather than labelled-by so a screen reader reads
              the hint itself instead of the word "Info". */}
          <span
            className="field-info"
            tabIndex={0}
            role="note"
            aria-describedby={`hint-${field.key}`}
          >
            ?
            <span className="field-tooltip" id={`hint-${field.key}`}>
              {field.hint}
            </span>
          </span>
          {field.type !== 'boolean' && (
            <span className="tuning-value-wrap">
              {/* Editable, not a readout: exposure is 14 discrete stops and
                  focus runs 0-1023, so landing on one by dragging is luck —
                  and one stop of exposure is a doubling of shutter time. */}
              <input
                type="number"
                className="tuning-value"
                data-testid={`value-${field.key}`}
                aria-label={`${field.label} value`}
                value={numeric ?? ''}
                placeholder="auto"
                min={field.min}
                max={field.max}
                step={field.step}
                disabled={disabled}
                onChange={(e) => {
                  const n = e.target.valueAsNumber
                  if (Number.isNaN(n)) return
                  // Clamp rather than let a fat-fingered digit become a 422
                  // from the sidecar's own ge/le validation.
                  const lo = field.min ?? n
                  const hi = field.max ?? n
                  applyDebounced(field.key, Math.min(hi, Math.max(lo, n)))
                }}
              />
              {field.unit && (
                <span className="tuning-unit" data-testid={`unit-${field.key}`}>
                  {field.unit}
                </span>
              )}
            </span>
          )}
        </div>
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
            // An unset control sits mid-range and dimmed rather than on a
            // bound. `Number(null) || 0` used to put brightness on its
            // minimum and exposure on its MAXIMUM — a one-second shutter —
            // both of which read as settings that are in force when the
            // sidecar is in fact leaving the camera alone.
            className={numeric === null ? 'is-unset' : undefined}
            value={numeric ?? midpoint(field)}
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
            Ignored during calibration.
          </p>
        )}
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
        <span className="tuning-camera">{cameraName ?? resolvedCameraName}</span>
      </h4>

      {/* Every failure this card can produce — a rejected live write, a save,
          a calibration sweep — arrives here. It sits outside the collapsible
          body on purpose: a sweep failing while the card is shut would
          otherwise be invisible, and LiveView's own banner reads a different
          hook. */}
      {error !== null && (
        <p className="tuning-error" role="alert" data-testid="tuning-error">
          {error}
        </p>
      )}

      <div className="tuning-body">
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
              <div className="tuning-grid">
                {group.keys.map((key) => {
                  const field = SETTINGS_FIELDS.find((f) => f.key === key)
                  return field ? renderField(field) : null
                })}
              </div>
            </section>
          ))}

        <section className="tuning-group" hidden={!open}>
          <h5>Calibration</h5>
          {profileIsStale && !busy && (
            <p className="field-hint" data-testid="tuning-stale-profile">
              This camera was calibrated before we measured control levels. Re-calibrate to get
              recommended values.
            </p>
          )}
          <button
            type="button"
            className="btn-outline btn-small"
            disabled={calibrating || busy}
            data-testid="tuning-calibrate"
            onClick={() => setGateOpen(true)}
            title="Stops capture, measures the camera, then starts again"
          >
            {calibrating ? <Spinner /> : null} Calibrate camera
          </button>

          {gateOpen && !busy && (
            <div className="tuning-gate" data-testid="tuning-scene-gate">
              <p className="field-hint">
                Place a typical item where it will be scanned, under the lighting you&apos;ll use.
                Use the feed to frame it. The feed stops for about 90 seconds.
              </p>
              {!running && (
                <p className="field-hint">
                  Start the feed first if you want to see what you are framing.
                </p>
              )}
              <div className="tuning-gate-actions">
                <button
                  type="button"
                  className="btn-primary btn-small"
                  data-testid="tuning-scene-ready"
                  onClick={() => {
                    setGateOpen(false)
                    void handleCalibrate()
                  }}
                >
                  Ready
                </button>
                <button
                  type="button"
                  className="btn-outline btn-small"
                  data-testid="tuning-scene-cancel"
                  onClick={() => setGateOpen(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {busy && (
            <div className="field-hint" data-testid="tuning-phases">
              <p>Measuring camera — about 90 seconds. The feed resumes when it finishes.</p>
              {/* Static, not live: the phases run in one blocking sidecar
                  call, and streaming progress would need a transport this
                  card does not have. Naming them still beats a bare
                  spinner. */}
              <ol className="tuning-phase-list">
                <li>exposure</li>
                <li>focus</li>
                <li>brightness</li>
                <li>confirming framerate</li>
              </ol>
            </div>
          )}

          {profile && !calibrating && (
            <div className="tuning-profile" data-testid="tuning-profile">
              <p className="field-hint">
                Measured {profile.fps_auto_exposure} fps on automatic exposure,{' '}
                {profile.fps_capped_exposure} fps with it capped.
              </p>
              {/* Reachable exactly when a focus sweep was possible (the
                  profile was actually swept, and the device supports both
                  focus and the autofocus lock) but came back with no
                  camera_focus evidence — a flat sharpness curve, meaning
                  nothing was in frame. Distinct from "nothing recommended"
                  below: this device may still have plenty else to show. */}
              {profile.sweep_version >= 1 &&
                profile.controls.focus &&
                profile.controls.autofocus &&
                !profile.measured?.camera_focus && (
                  <p className="field-hint" data-testid="tuning-no-focus-peak">
                    Couldn&apos;t find a focus peak — was an item in view? Re-run calibration with
                    the item in frame.
                  </p>
                )}
              {Object.keys(profile.recommended).length === 0 ? (
                <p className="field-hint" data-testid="tuning-no-recommendation">
                  Nothing to change: this camera ignored every control we can set.
                </p>
              ) : (
                <>
                  <ul data-testid="tuning-evidence">
                    {Object.entries(profile.recommended).map(([k, v]) => {
                      const m = profile.measured?.[k]
                      return (
                        <li key={k}>
                          {k}: {String(v)}
                          {m && (
                            <span className="field-hint">
                              {' '}
                              — {m.metric} (was {m.baseline}){m.reached ? '' : ', best available'}
                            </span>
                          )}
                        </li>
                      )
                    })}
                  </ul>
                  <button
                    className="btn-primary btn-small"
                    disabled={saving}
                    data-testid="tuning-apply-profile"
                    onClick={() => reported(applyProfile())}
                  >
                    {saving ? <Spinner /> : null} Apply these settings
                  </button>
                </>
              )}
            </div>
          )}
        </section>
      </div>

      {dirtyKeys.length > 0 && (
        <div className="tuning-actions">
          <span data-testid="tuning-dirty">
            {dirtyKeys.length} unsaved change{dirtyKeys.length === 1 ? '' : 's'}
          </span>
          <button
            className="btn-primary btn-small"
            disabled={saving}
            data-testid="tuning-save"
            onClick={() => reported(save())}
          >
            {saving ? <Spinner /> : null} Save
          </button>
          <button
            className="btn-outline btn-small"
            data-testid="tuning-revert"
            onClick={() => {
              if (!savedSettings) return
              // A saved value of null means "leave the camera alone", which
              // is the default for all four controls — so on a fresh install
              // this is the common case, not an edge one. exclude_none on the
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
