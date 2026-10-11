import { useEffect, useState, type JSX } from 'react'
import { AppShell } from './components/AppShell'
import { Spinner } from './components/Spinner'
import './App.css'

export interface AppProps {
  // Injectable for tests; defaults to the preload bridge.
  getPort?: () => Promise<number | null>
  pollMs?: number
}

function App({ getPort, pollMs = 500 }: AppProps = {}): JSX.Element {
  const resolvePort = getPort ?? ((): Promise<number | null> => window.api.getSidecarPort())
  const [port, setPort] = useState<number | null>(null)
  // Outside the Electron shell — this dev URL opened in a plain browser — there is no preload
  // bridge, so `window.api` is undefined and port discovery can never succeed. An eternal
  // "Starting sidecar…" spinner there misreports a running sidecar as still loading; the truth
  // is that this screen only works inside the app.
  const inShell = getPort != null || (typeof window !== 'undefined' && window.api != null)

  useEffect(() => {
    if (!inShell) return
    let active = true
    let timer: ReturnType<typeof setTimeout>
    const tick = async (): Promise<void> => {
      let p: number | null = null
      try {
        p = await resolvePort()
      } catch {
        // Transient IPC failure: keep polling rather than stalling forever.
      }
      if (!active) return
      if (p != null) setPort(p)
      else timer = setTimeout(tick, pollMs)
    }
    tick()
    return () => {
      active = false
      clearTimeout(timer)
    }
    // resolvePort is stable per mount; intentionally not re-running on identity change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pollMs])

  if (!inShell) {
    return (
      <div className="app-waiting">
        <p>Open the SCANnCART desktop app</p>
        <small>
          This screen reaches the sidecar through the app itself. A browser tab at this address has
          no shell to ask, so there is nothing to connect to here.
        </small>
      </div>
    )
  }
  if (port == null) {
    return (
      <div className="app-waiting">
        <Spinner size={28} />
        <p>Starting sidecar…</p>
        <small>loading Python runtime and model libraries</small>
      </div>
    )
  }
  return <AppShell port={port} />
}

export default App
