import { type JSX } from 'react'
import { useSidecarStream, type StreamDeps } from '../hooks/useSidecarStream'
import { boxToPercent } from '../lib/overlay'

export interface LiveViewProps {
  port: number
  deps?: StreamDeps
}

export function LiveView({ port, deps }: LiveViewProps): JSX.Element {
  const { frame, statusState, connected, items, start, stop } = useSidecarStream(port, deps)
  const running = statusState === 'running'
  const stats = frame?.stats

  return (
    <div className="live-view">
      <div className="live-toolbar">
        <button onClick={running ? stop : start} aria-label={running ? 'Stop' : 'Start'}>
          {running ? 'Stop' : 'Start'}
        </button>
        <span className="state" data-testid="state">
          {statusState}
        </span>
        <span className="conn" data-testid="conn">
          {connected ? 'connected' : 'disconnected'}
        </span>
      </div>

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

      <div className="stats-strip" data-testid="stats">
        {stats ? (
          <>
            <span>infer {stats.infer_fps.toFixed(1)} fps</span>
            <span>capture {stats.capture_fps.toFixed(0)} fps</span>
            <span>latency {stats.latency_ms.toFixed(0)} ms</span>
          </>
        ) : (
          <span>no stats yet</span>
        )}
      </div>

      <ul className="item-log" data-testid="item-log">
        {items.map((it) => (
          <li key={it.track_id}>
            {it.cls} ({Math.round(it.conf * 100)}%)
          </li>
        ))}
      </ul>
    </div>
  )
}
