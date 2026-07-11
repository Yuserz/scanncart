import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import App from './App'

// jsdom has no WebSocket; stub a no-op so LiveView's stream hook can mount.
class NoopWS {
  onopen: (() => void) | null = null
  onmessage: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close(): void {}
  send(): void {}
}

describe('App', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', NoopWS as never)
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('shows a waiting state until the sidecar port resolves', async () => {
    const getPort = vi.fn().mockResolvedValue(null)
    render(<App getPort={getPort} pollMs={10} />)
    expect(screen.getByText(/Starting sidecar/i)).toBeInTheDocument()
  })

  it('mounts the Live View once a port is available', async () => {
    const getPort = vi.fn().mockResolvedValue(8765)
    render(<App getPort={getPort} pollMs={10} />)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /start/i })).toBeInTheDocument()
    })
  })
})
