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
