import { useEffect, useState, type JSX } from 'react'
import { AppShell } from './components/AppShell'

export interface AppProps {
  // Injectable for tests; defaults to the preload bridge.
  getPort?: () => Promise<number | null>
  pollMs?: number
}

function App({ getPort, pollMs = 500 }: AppProps = {}): JSX.Element {
  const resolvePort = getPort ?? ((): Promise<number | null> => window.api.getSidecarPort())
  const [port, setPort] = useState<number | null>(null)

  useEffect(() => {
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

  if (port == null) {
    return <div className="app-waiting">Starting sidecar…</div>
  }
  return <AppShell port={port} />
}

export default App
