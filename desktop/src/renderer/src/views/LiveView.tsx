import { useState, type CSSProperties, type JSX } from 'react'
import { useSidecarStream, type StreamDeps } from '../hooks/useSidecarStream'
import { useActiveWeights } from '../hooks/useActiveWeights'
import type { ClassRecall, UnrecordedResizeMode, ValidationRecord } from '../lib/api'
import { boxToPercent } from '../lib/overlay'
import { recordedConfirmation } from '../lib/resizeMode'
import { Spinner } from '../components/Spinner'
import { CameraTuning } from '../components/CameraTuning'
import './LiveView.css'

export interface LiveViewProps {
  port: number
  deps?: StreamDeps
}

// The measured score as the stats strip shows it: what `--val` recorded, or a dash that says
// nothing was measured.
//
// Three decisions in here are rules rather than formatting. The split is **named in the label**
// rather than assumed: `test` is the acceptance number and `valid` is the split training selected
// on, so a `valid` recall labelled just "recall" would overstate the model. Below-floor classes are
// **counted, not listed** (a tile has no room for the names, and the count is what fails the
// acceptance criterion), with the names in the title, so the strip never implies the *mean* is
// what was judged. And a class the split holds no instances of is `recall: null`, which is not a
// miss — including it would count a data gap as a model failure, the same distinction the Admin
// Panel's score block draws.
function recallReadout(measured: ValidationRecord | null): {
  value: string
  label: string
  title: string
  warn: boolean
  unmeasured: boolean
} {
  if (measured === null) {
    return {
      value: '—',
      label: 'recall not measured',
      title:
        'No measured score is recorded beside these weights, so how well they score is not known. ' +
        'Run `train_model.py --val` to measure them; the Admin Panel shows the result once recorded.',
      warn: false,
      unmeasured: true
    }
  }
  const below = measured.per_class.filter(
    (c): c is ClassRecall & { recall: number } => c.recall !== null && c.recall < measured.floor
  )
  const floor = measured.floor.toFixed(2)
  const verdict =
    below.length === 0
      ? `Every class the split holds is at or above the ${floor} floor.`
      : `Below the ${floor} floor: ${below.map((c) => `${c.name} ${c.recall.toFixed(3)}`).join(', ')}.`
  const mean = measured.aggregates.recall
  if (typeof mean !== 'number') {
    // A block with per-class numbers and no recall aggregate cannot render a mean, and inventing
    // one from the class list would be a different statistic.
    return {
      value: '—',
      label: 'recall not measured',
      title: `The ${measured.split} split was measured, but no recall aggregate was recorded. ${verdict}`,
      warn: false,
      unmeasured: true
    }
  }
  return {
    value: `${Math.round(mean * 100)}%`,
    label:
      below.length > 0
        ? `${measured.split} recall · ${below.length} below floor`
        : `${measured.split} recall`,
    title:
      `Mean recall over the classes the ${measured.split} split holds, judged per class against a ` +
      `${floor} floor. ${verdict}`,
    warn: below.length > 0,
    unmeasured: false
  }
}

// The requirement chip: what the record beside these weights says they were trained at, and — when
// nothing recorded it — that the mode on screen is a guess.
//
// The banners above say the second case at length, but only while a capture runs, and this is a
// property of the *weights*: taking it away while idle would hide it exactly when someone is
// deciding whether to start. Two states, and the difference is what the operator can act on — a
// record is evidence, an assumption is a gap in what is known. That is the same distinction the
// recall tile beside it draws for a measurement nobody recorded, which is why both use the dim
// style rather than the amber one: neither is a bad value, both are missing proof.
function requirementReadout(
  required: string | null,
  unrecorded: UnrecordedResizeMode | null
): { value: string; label: string; title: string; assumed: boolean } | null {
  if (unrecorded !== null) {
    return {
      value: unrecorded.resize_mode,
      label: 'requirement (assumed)',
      // The sidecar's own sentence, so the chip and the banner cannot describe different risks.
      title: unrecorded.warning,
      assumed: true
    }
  }
  if (required === null) return null
  return {
    value: required,
    label: 'requirement (recorded)',
    title:
      `A record beside these weights says they were trained at ${required}, which is what ` +
      'resize_mode: auto honours. Setting resize_mode explicitly overrides the record instead.',
    assumed: false
  }
}

// The class-list chip: how many classes the *running* model has, and whether they are the names this
// app is built for.
//
// It sits beside the banner above rather than instead of it, on the same division the geometry tiles
// draw: the banner is the **problem**, present only when there is one and in the sidecar's own words,
// while the chip is the **state**, present whenever the model's classes are known — including the
// healthy case the banner is silent about. Without it the count is invisible on a good model and
// shows up only inside a sentence about distances on a bad one, so `24` where `8` was expected is not
// legible as a number.
//
// Three things here are rules rather than formatting. The count is a tile value, because the number
// is what an operator compares against their own model's count — 7 for a v1 weight, 8 for v2 — and
// the sidecar is the side that knows which generation the running weight belongs to. The verdict is
// a **count of findings**
// rather than a direction (`wrong` / `missing`), because the sidecar's third finding is the quiet one
// — a model that simply cannot predict some roster names — and a word like "mismatch" would be false
// for it. And an empty list renders **nothing**: before the first inference the sidecar knows no
// names, so a `0` would be a claim about a model that has not spoken yet.
function classListReadout(
  names: string[],
  warnings: string[]
): { value: string; label: string; title: string; warn: boolean } | null {
  if (names.length === 0) return null
  const listed = `Predicts ${names.join(', ')}.`
  if (warnings.length === 0) {
    return {
      value: String(names.length),
      label: 'classes · roster ok',
      title:
        `This model declares ${names.length} classes, and every one of them is a name in this ` +
        `app's roster. ${listed}`,
      warn: false
    }
  }
  return {
    value: String(names.length),
    label: `classes · ${warnings.length} finding${warnings.length === 1 ? '' : 's'}`,
    // The sidecar's sentences, so the chip and the banner cannot describe one model differently.
    title: `${warnings.join(' ')} ${listed}`,
    warn: true
  }
}

export function LiveView({ port, deps }: LiveViewProps): JSX.Element {
  const {
    frame,
    statusState,
    connected,
    items,
    start,
    stop,
    error,
    clearError,
    classNames,
    classWarnings
  } = useSidecarStream(port, deps)
  const running = statusState === 'running'
  // The weights the configured model points at: the geometry they run at, what they scored, and
  // whether the settings contradict their record. Read here as well as in the Admin Panel's Model
  // field, because this is the screen where the consequences show up — a mismatch does not fail
  // anything, it just makes every detection weaker than the model can do, and the operator
  // watching the feed is the one who can see that and cannot see why.
  //
  // Unlike the banner above, the readout is **not** gated on capture running: it is a fact about
  // the weights rather than about the frames, and both fields behind it are restart-required, so
  // it is the same before and after Start. Hiding it while idle would take it away exactly when
  // someone is deciding whether to start.
  const weights = useActiveWeights(port, deps?.settingsDeps)
  const resizeMismatch = weights?.mismatch ?? null
  // The running model's vocabulary, as the sidecar reported it on the stream. A fact about the
  // *capture* rather than about the settings above it, which is why it is not gated on the backend
  // the way the geometry readout is: a remote workflow declares its classes too, and there is no
  // local model description to hide behind.
  const classes = classListReadout(classNames, classWarnings)
  // The softer half of the same subject: the settings do not contradict the weights, nothing
  // recorded what they were trained to expect, so `auto` is guessing from the file format. Worth
  // the same banner because the consequence is the same one — weaker detections, worst on `far` —
  // and because nothing else on this screen can say so: the mismatch rule is silent by
  // construction (there is no record to contradict) and the field that would fix it lives in
  // Admin. Mutually exclusive with the mismatch above, so they never stack: one needs a record to
  // disagree with, the other needs the absence of one.
  const unrecorded = weights?.unrecorded ?? null
  // What the record says, when there is one — the other half of the pair above, and the mode the
  // confirmation below names (read back off the listing rather than remembered from the click).
  const required = weights?.required ?? null
  const requirement = requirementReadout(required, unrecorded)
  // A write from this view landed, and the sidecar's refreshed answer is what cleared the banner —
  // so the slot it vacated says what happened instead of going silently blank, which on a live
  // feed is indistinguishable from a click that did nothing.
  const justRecorded = weights?.justRecorded ?? false
  const recall = recallReadout(weights?.measured ?? null)
  const geometryTitle =
    weights === null
      ? ''
      : resizeMismatch !== null
        ? `These weights need resize_mode: ${resizeMismatch.required} and the settings have ` +
          `${resizeMismatch.mode}, so every object reaches the model at a different scale than it ` +
          'was trained on.'
        : `The geometry every frame is resized to before inference: resize_mode is ` +
          `${weights.setting}, which resolves to ${weights.geometry} for these weights.`
  const stats = frame?.stats
  const trackedCount = frame?.detections.length ?? 0
  // Absent on a sidecar that predates the field, which reads as "nothing suppressed" — the same
  // reading as a clean frame, and the honest one: there is no evidence to the contrary.
  const suppressed = stats?.suppressed ?? 0
  // Real decoded frame size, read on img load — drives the wrapper's
  // aspect-ratio and fit-to-column sizing in CSS (falls back to 16/9
  // while idle). Same-value updates bail out, so per-frame loads are free.
  const [frameSize, setFrameSize] = useState<{ w: number; h: number } | null>(null)
  const previewStyle: CSSProperties | undefined = frameSize
    ? ({ '--preview-w': `${frameSize.w}`, '--preview-h': `${frameSize.h}` } as CSSProperties)
    : undefined
  // Purely presentational: start/stop block on the sidecar (model load,
  // possibly a one-time weight download), so surface that wait in the UI.
  const [pending, setPending] = useState<'start' | 'stop' | null>(null)
  // Calibration holds the camera exclusively, so the sidecar 409s any start
  // during it. Offering the button anyway turned a known constraint into an
  // error banner the operator had to interpret.
  const [cameraBusy, setCameraBusy] = useState(false)

  const handleToggle = async (): Promise<void> => {
    const action = running ? stop : start
    setPending(running ? 'stop' : 'start')
    try {
      await action()
    } finally {
      setPending(null)
    }
  }

  return (
    <div className="live-view">
      {error !== null && (
        <div className="live-error" role="alert" data-testid="live-error">
          <span>{error}</span>
          <button className="btn-outline btn-small" onClick={clearError} aria-label="Dismiss error">
            Dismiss
          </button>
        </div>
      )}
      {/* What the *running* model actually predicts, when that is not this app's roster.
          Deliberately not dismissible and deliberately not gated on a probe: the sidecar reports
          it from the first inference that knows the model's classes, so it appears on its own, in
          the view where the consequence (boxes logged under labels the app does not know) is
          happening. A 24-class weight - one trained per product-and-distance - is the case this
          exists for; so is a stock COCO model, which simply cannot predict any of the roster's
          products.
          The sentences come from the sidecar rather than being composed here: it is the only side
          that has the class names, and one finding must have one description. */}
      {classWarnings.length > 0 && (
        <div className="live-warning" role="status" data-testid="live-class-warnings">
          {classWarnings.map((warning) => (
            <span key={warning.slice(0, 40)}>{warning}</span>
          ))}
        </div>
      )}
      {/* Only while capture runs, and not dismissible. The claim is about the frames and the
          items being produced right now - which is not true of an idle view, where the settings
          are editable a click away in Admin and the same field says the same thing. Dismissing
          it would hide the reason the numbers below are worse than the model, which is the one
          fact this banner exists to supply. */}
      {running && resizeMismatch !== null && (
        <div className="live-warning" role="status" data-testid="live-resize-mismatch">
          <span>
            <b>These weights are running at the wrong geometry.</b> {`They need `}
            <code>resize_mode: {resizeMismatch.required}</code>
            {` and the settings have `}
            <code>{resizeMismatch.mode}</code>
            {`, so every object reaches the model at a different scale than it was trained on — `}
            {`the detections and the item log below are weaker than this model can do. Stop `}
            {`capture, set `}
            <code>resize_mode</code>
            {` to `}
            <code>auto</code>
            {` (which uses the recorded requirement) or to `}
            <code>{resizeMismatch.required}</code>
            {`, then start again.`}
          </span>
        </div>
      )}
      {/* Gated on capture running for the same reason as the banner above: the claim is about the
          frames and the item log on screen. While idle the Admin Panel's Model field says it
          already — with the button that answers it — and repeating it here would be a second
          place to read one situation, the one without the remedy.

          Not dismissible either. The operator can see the weaker detections and cannot see why,
          and this is the only thing on this screen that supplies the why. */}
      {running && unrecorded !== null && (
        <div className="live-warning has-action" role="status" data-testid="live-assumed-geometry">
          <span>
            <b>These weights are running at an assumed geometry.</b> {unrecorded.warning}{' '}
            {unrecorded.remedy}
          </span>
          {/* The same write the Admin Panel's button performs, on the screen where the cost of the
              assumption is visible. Safe to offer mid-capture — the mode written is the one `auto`
              already resolved to, so the detector is not disturbed — which is exactly why it does
              not need the view switch it used to. */}
          <button
            className="btn-outline btn-small"
            disabled={weights?.recording ?? false}
            onClick={() => void weights?.record()}
            data-testid="live-record-resize-mode"
          >
            {weights?.recording ? <Spinner size={12} /> : null} Record it now:{' '}
            {unrecorded.resize_mode}
          </button>
        </div>
      )}
      {/* Gated on `running` like the banner it replaces: this is the answer to a warning that only
          exists while a capture runs, and an idle view would otherwise carry a green line about a
          write nobody can connect to anything on screen. The chip in the strip is where the fact
          lives once the capture stops. */}
      {running && justRecorded && unrecorded === null && required !== null && (
        <div className="live-confirm" role="status" data-testid="live-recorded">
          {recordedConfirmation(required)}
        </div>
      )}
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
          onClick={() => void handleToggle()}
          disabled={pending !== null || cameraBusy}
          aria-label={running ? 'Stop' : 'Start'}
          title={cameraBusy ? 'Calibration is using the camera' : undefined}
        >
          {pending !== null ? (
            <>
              <Spinner size={12} />
              {pending === 'start' ? 'Starting…' : 'Stopping…'}
            </>
          ) : running ? (
            'Stop'
          ) : (
            'Start'
          )}
        </button>
      </div>

      <div className="live-body">
        <div className="feed-col">
          <div className="preview-wrapper" data-testid="preview-wrapper" style={previewStyle}>
            {frame ? (
              <img
                className="preview-img"
                alt="live preview"
                src={`data:image/jpeg;base64,${frame.jpeg}`}
                onLoad={(e) => {
                  const { naturalWidth: w, naturalHeight: h } = e.currentTarget
                  if (w > 0 && h > 0) {
                    setFrameSize((prev) => (prev && prev.w === w && prev.h === h ? prev : { w, h }))
                  }
                }}
              />
            ) : (
              <div className="preview-placeholder" data-testid="preview-placeholder">
                {pending === 'start' ? (
                  <>
                    <Spinner size={22} />
                    <span>Loading model…</span>
                    <small>first use of a model downloads its weights (one time)</small>
                  </>
                ) : running ? (
                  <>
                    <Spinner size={22} />
                    <span>Waiting for frames…</span>
                  </>
                ) : (
                  'Waiting for frames…'
                )}
              </div>
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
            <h4>Model &amp; performance</h4>
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
                  {/* Rendered only when it fires, unlike `tracked` above. A count of 0 IS the
                      healthy reading here, and a tile that always showed 0 would be one nobody
                      looks at by the time it says 1. Absence therefore means "no phantom this
                      frame", and the setting's own state is legible where it is set, in the
                      tuning card's checkbox. */}
                  {suppressed > 0 && (
                    <div
                      className="stat-tile warn"
                      data-testid="stat-suppressed"
                      title={
                        'Detections whose box was pinned to all four frame edges, dropped before ' +
                        'they reached the item log. These weights produce that shape on an empty ' +
                        'counter. If a real item that fills the frame stops being detected, turn ' +
                        'off “Drop frame-edge phantoms” in Camera tuning.'
                      }
                    >
                      <b>{suppressed}</b>
                      <small>suppressed</small>
                    </div>
                  )}
                </>
              ) : (
                // Renamed from "no stats yet": with the weights readout below always present, the
                // claim is about the *frames*, and the old wording read as a contradiction beside
                // two numbers that are on screen.
                <span>no frames yet</span>
              )}
              {/* The weights readout: what geometry these weights run at, what the record says they
                  were trained at (or that nothing recorded it), and what they scored. Rendered
                  whenever the backend actually runs them — `geometry` is `null` for a remote
                  backend, whose workflow holds its own model and resizes server-side, so there is
                  nothing local to describe and no score of *these* weights to show. */}
              {weights !== null && weights.geometry !== null && (
                <>
                  <div
                    className={`stat-tile${resizeMismatch !== null ? ' warn' : ''}`}
                    data-testid="stat-geometry"
                    title={geometryTitle}
                  >
                    <b>{weights.geometry}</b>
                    <small>{weights.setting === 'auto' ? 'geometry (auto)' : 'geometry'}</small>
                  </div>
                  {requirement !== null && (
                    <div
                      className={`stat-tile${requirement.assumed ? ' unmeasured' : ''}`}
                      data-testid="stat-requirement"
                      title={requirement.title}
                    >
                      <b>{requirement.value}</b>
                      <small>{requirement.label}</small>
                    </div>
                  )}
                  <div
                    className={`stat-tile${recall.warn ? ' warn' : ''}${
                      recall.unmeasured ? ' unmeasured' : ''
                    }`}
                    data-testid="stat-recall"
                    title={recall.title}
                  >
                    <b>{recall.value}</b>
                    <small>{recall.label}</small>
                  </div>
                </>
              )}
              {/* The running model's class count and roster verdict. Outside the gate above on
                  purpose: the tiles there describe the *configured* weights (restart-required
                  settings, readable while idle), while this one describes what the sidecar actually
                  loaded, and a remote backend has classes but no local geometry. */}
              {classes !== null && (
                <div
                  className={`stat-tile${classes.warn ? ' warn' : ''}`}
                  data-testid="stat-classes"
                  title={classes.title}
                >
                  <b>{classes.value}</b>
                  <small>{classes.label}</small>
                </div>
              )}
            </div>
          </div>

          <CameraTuning
            port={port}
            running={running}
            start={start}
            stop={stop}
            onCameraBusy={setCameraBusy}
            deps={deps?.settingsDeps}
          />

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
