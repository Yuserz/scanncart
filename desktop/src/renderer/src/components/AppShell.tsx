import { useState, type JSX } from 'react'
import { LiveView } from '../views/LiveView'
import { AdminPanel } from '../views/AdminPanel'
import { BasketTestView } from '../views/BasketTestView'
import { useSidecarHealth, type SidecarHealthDeps } from '../hooks/useSidecarHealth'
import './AppShell.css'

export type View = 'live' | 'admin' | 'basket'

export interface AppShellProps {
  port: number
  // Injectable for tests; defaults to the preload bridge.
  healthDeps?: SidecarHealthDeps
}

export function AppShell({ port, healthDeps }: AppShellProps): JSX.Element {
  const [view, setView] = useState<View>('live')
  const health = useSidecarHealth(healthDeps)

  return (
    <div className="app-shell">
      <nav className="app-nav">
        <button
          data-testid="nav-live"
          aria-pressed={view === 'live'}
          className={view === 'live' ? 'active' : ''}
          onClick={() => setView('live')}
        >
          Live
        </button>
        <button
          data-testid="nav-admin"
          aria-pressed={view === 'admin'}
          className={view === 'admin' ? 'active' : ''}
          onClick={() => setView('admin')}
        >
          Admin
        </button>
        <button
          data-testid="nav-basket"
          aria-pressed={view === 'basket'}
          className={view === 'basket' ? 'active' : ''}
          onClick={() => setView('basket')}
        >
          Basket test
        </button>
      </nav>
      {/* Above the views rather than inside one, because it is not about either of them: with no
          sidecar answering, both panels fail — the Live view on frames and the item log, the Admin
          panel on every read and write — and the reason has to be legible in one place that does
          not depend on either view's own request succeeding.
          Deliberately not dismissible: it describes a condition that is still true, and every
          other reading on screen is either stale or a failed fetch that this is the explanation
          for. The recovery is named because nothing in this window can perform it: the port the
          renderer holds is only good while its sidecar is listening.
          The self-side-car sentence is not decoration. "Alive but silent" is the case that was
          invisible — the process stays in `tasklist` with nothing on its port — and an operator
          who has seen that shape once can recognise it from the top of a process list. */}
      {health === 'unresponsive' && (
        <div className="sidecar-notice" role="alert" data-testid="sidecar-unresponsive">
          <span>
            <b>The sidecar is not answering.</b> Its process may still be running — a sidecar that
            stays alive with nothing listening is the shape this notices — but either way this
            window cannot reach it, so the preview, the item log and every Admin action will fail
            until it is restarted.
          </span>
          <span>Quit SCANnCART and start it again to recover.</span>
        </div>
      )}
      {view === 'live' && <LiveView port={port} />}
      {view === 'admin' && <AdminPanel port={port} />}
      {view === 'basket' && <BasketTestView port={port} />}
    </div>
  )
}
