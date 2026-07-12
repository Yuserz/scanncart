import { useState, type JSX } from 'react'
import { LiveView } from '../views/LiveView'
import { AdminPanel } from '../views/AdminPanel'
import './AppShell.css'

export type View = 'live' | 'admin'

export interface AppShellProps {
  port: number
}

export function AppShell({ port }: AppShellProps): JSX.Element {
  const [view, setView] = useState<View>('live')

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
      </nav>
      {view === 'live' ? <LiveView port={port} /> : <AdminPanel port={port} />}
    </div>
  )
}
