import { useState, type CSSProperties, type JSX } from 'react'
import { useSidecarStream, type StreamDeps } from '../hooks/useSidecarStream'
import { boxToPercent } from '../lib/overlay'
import { Spinner } from '../components/Spinner'
import { CameraTuning } from '../components/CameraTuning'
import './LiveView.css'

export interface LiveViewProps {
  port: number
  deps?: StreamDeps
}

export function LiveView({ port, deps }: LiveViewProps): JSX.Element {
  const { frame, statusState, connected, items, start, stop, error, clearError } = useSidecarStream(
    port,
    deps
  )
  const running = statusState === 'running'
  const stats = frame?.stats
  const trackedCount = frame?.detections.length ?? 0
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
