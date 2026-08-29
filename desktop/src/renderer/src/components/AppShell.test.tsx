import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppShell } from './AppShell'

// jsdom has no WebSocket; stub a no-op so LiveView's stream hook can mount.
class NoopWS {
  onopen: (() => void) | null = null
  onmessage: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  // eslint-disable-next-line @typescript-eslint/no-empty-function -- intentional no-op stub
  close(): void {}
  // eslint-disable-next-line @typescript-eslint/no-empty-function -- intentional no-op stub
  send(): void {}
}

function bodyForUrl(url: string): unknown {
  if (url.includes('/api/settings/preset') || url.includes('/api/settings')) {
    return {
      active_model: 'yolo11n.pt',
      camera_index: 0,
      capture_width: 1280,
      capture_height: 720,
      capture_fps: 60,
      conf_threshold: 0.5,
      imgsz: 640,
      infer_frame_skip: 0,
      device: 'auto',
      preview_height: 720,
      track_expiry_s: 1.5,
      hot_reloadable_fields: ['infer_frame_skip'],
      restart_required_fields: ['active_model'],
      warnings: []
    }
  }
  if (url.includes('/api/system-info')) {
    return {
      cpu_count: 8,
      ram_gb: 16,
      cuda_available: false,
      gpu_name: null,
      gpu_vram_gb: null,
      recommended_preset: 'mid_range'
    }
  }
  if (url.includes('/api/presets')) {
    return { presets: [], recommended: 'mid_range' }
  }
  if (url.includes('/api/cameras')) {
    return {
      cameras: [{ index: 0, name: 'Fake Cam', width: 1280, height: 720 }],
      probed: true,
      detail: ''
    }
  }
  if (url.includes('/api/health')) {
    return { state: 'idle', active_model: 'yolo11n.pt', device: 'cpu' }
  }
  return {}
}

describe('AppShell', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', NoopWS as never)
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => ({
        ok: true,
        status: 200,
        json: async () => bodyForUrl(url)
      }))
    )
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('defaults to the Live view', async () => {
    render(<AppShell port={8765} />)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /start/i })).toBeInTheDocument()
    })
    expect(screen.getByTestId('nav-live')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('nav-admin')).toHaveAttribute('aria-pressed', 'false')
  })

  it('switches to the Admin view on nav click', async () => {
    const user = userEvent.setup()
    render(<AppShell port={8765} />)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /start/i })).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('nav-admin'))

    expect(screen.getByTestId('nav-admin')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('nav-live')).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByText(/Admin Settings/i)).toBeInTheDocument()
  })
})
