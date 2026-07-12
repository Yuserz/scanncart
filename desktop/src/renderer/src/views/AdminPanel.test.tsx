import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AdminPanel } from './AdminPanel'
import type { SettingsDeps } from '../hooks/useSidecarSettings'
import type { ApiClient, SettingsResponse } from '../lib/api'

function baseSettings(overrides: Partial<SettingsResponse> = {}): SettingsResponse {
  return {
    active_model: 'yolo11n.pt',
    camera_index: 0,
    capture_width: 1280,
    capture_height: 720,
    capture_fps: 60,
    conf_threshold: 0.5,
    infer_frame_skip: 0,
    device: 'auto',
    preview_height: 720,
    track_expiry_s: 1.5,
    hot_reloadable_fields: ['infer_frame_skip', 'preview_height', 'track_expiry_s'],
    restart_required_fields: [
      'active_model',
      'camera_index',
      'capture_width',
      'capture_height',
      'capture_fps',
      'conf_threshold',
      'device'
    ],
    warnings: [],
    ...overrides
  }
}

function makeDeps(
  captureState: string,
  overrides: Partial<ApiClient> = {}
): { deps: SettingsDeps; api: ApiClient } {
  const api: ApiClient = {
    health: vi.fn(async () => ({ state: captureState, active_model: 'yolo11n.pt', device: 'cpu' })),
    start: vi.fn(),
    stop: vi.fn(),
    getLogs: vi.fn(),
    getSettings: vi.fn(async () => baseSettings()),
    updateSettings: vi.fn(async (patch) => baseSettings(patch)),
    getSystemInfo: vi.fn(async () => ({
      cpu_count: 8,
      ram_gb: 16,
      cuda_available: false,
      accelerator: 'cpu' as const,
      gpu_name: null,
      gpu_vram_gb: null,
      recommended_preset: 'mid_range'
    })),
    getPresets: vi.fn(async () => ({
      presets: [
        { name: 'low_end', label: 'Low-end', description: 'weak machine', settings: {} },
        { name: 'mid_range', label: 'Mid-range', description: 'balanced', settings: {} }
      ],
      recommended: 'mid_range'
    })),
    applyPreset: vi.fn(async (name) => baseSettings({ active_model: `${name}.pt` })),
    ...overrides
  }
  return { deps: { apiFactory: () => api, healthPollMs: 10_000 }, api }
}

describe('AdminPanel', () => {
  it('renders fetched settings, hardware info, and presets', async () => {
    const { deps } = makeDeps('idle')
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('capture-state')).toHaveTextContent('idle'))
    expect(screen.getByTestId('hardware-info')).toHaveTextContent('CPU cores: 8')
    expect(screen.getByTestId('hardware-info')).toHaveTextContent('No GPU detected')
    expect(screen.getByText(/Recommended for this machine/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/Model/i)).toHaveValue('yolo11n.pt')
  })

  it('editing and saving a hot-reloadable field while running succeeds', async () => {
    const { deps, api } = makeDeps('running')
    const user = userEvent.setup()
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('capture-state')).toHaveTextContent('running'))

    const frameSkip = screen.getByLabelText(/Frame skip/i)
    await user.clear(frameSkip)
    await user.type(frameSkip, '3')

    const saveButton = screen.getByTestId('save-settings')
    await waitFor(() => expect(saveButton).toBeEnabled())
    await user.click(saveButton)

    await waitFor(() => expect(api.updateSettings).toHaveBeenCalledWith({ infer_frame_skip: 3 }))
  })

  it('editing a restart-required field while running blocks Save with a warning', async () => {
    const { deps } = makeDeps('running')
    const user = userEvent.setup()
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('capture-state')).toHaveTextContent('running'))

    const confInput = screen.getByLabelText(/Confidence threshold/i)
    await user.clear(confInput)
    await user.type(confInput, '0.7')

    await waitFor(() => expect(screen.getByTestId('restart-warning')).toBeInTheDocument())
    expect(screen.getByTestId('restart-warning')).toHaveTextContent('conf_threshold')
    expect(screen.getByTestId('save-settings')).toBeDisabled()
  })

  it('applying a preset calls applyPreset with its name', async () => {
    const { deps, api } = makeDeps('idle')
    const user = userEvent.setup()
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('apply-preset-mid_range')).toBeInTheDocument())
    await user.click(screen.getByTestId('apply-preset-mid_range'))

    await waitFor(() => expect(api.applyPreset).toHaveBeenCalledWith('mid_range'))
  })

  it('Restore Defaults calls updateSettings with the hardcoded defaults', async () => {
    const { deps, api } = makeDeps('idle')
    const user = userEvent.setup()
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('restore-defaults')).toBeEnabled())
    await user.click(screen.getByTestId('restore-defaults'))

    await waitFor(() =>
      expect(api.updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({ active_model: 'yolo11n.pt', capture_width: 1280 })
      )
    )
  })

  it('Restore Defaults is disabled while capture is running', async () => {
    const { deps } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('restore-defaults')).toBeDisabled())
  })

  it('shows a retrying message (not a dead end) when the sidecar is not reachable yet, and recovers', async () => {
    let calls = 0
    const { deps } = makeDeps('idle', {
      getSettings: vi.fn(async () => {
        calls += 1
        if (calls < 2) throw new Error('sidecar GET /settings failed: ECONNREFUSED')
        return baseSettings()
      })
    })
    deps.retryDelayMs = 10
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('admin-error')).toBeInTheDocument())
    expect(screen.getByTestId('admin-error')).toHaveTextContent(/retrying/i)
    expect(screen.getByTestId('retry-load')).toBeInTheDocument()

    // Auto-retry succeeds without any user action.
    await waitFor(() => expect(screen.getByLabelText(/Model/i)).toBeInTheDocument())
  })

  it('labels a CUDA machine as GPU-acceleration-available', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: true,
        accelerator: 'cuda' as const,
        gpu_name: 'NVIDIA GeForce RTX 4060',
        gpu_vram_gb: 8,
        recommended_preset: 'high_end'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)
    await waitFor(() =>
      expect(screen.getByTestId('hardware-info')).toHaveTextContent(
        /NVIDIA GeForce RTX 4060.*GPU acceleration available/
      )
    )
  })

  it('labels an integrated GPU as APU without CUDA', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: false,
        accelerator: 'integrated' as const,
        gpu_name: 'AMD Radeon(TM) Graphics',
        gpu_vram_gb: null,
        recommended_preset: 'low_end'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)
    await waitFor(() =>
      expect(screen.getByTestId('hardware-info')).toHaveTextContent(
        /Integrated graphics: AMD Radeon\(TM\) Graphics \(APU\)/
      )
    )
  })

  it('labels a discrete NVIDIA card without CUDA as a missing-CUDA-torch case, not an APU', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: false,
        accelerator: 'integrated' as const,
        gpu_name: 'NVIDIA GeForce RTX 4060',
        gpu_vram_gb: null,
        recommended_preset: 'mid_range'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)
    await waitFor(() =>
      expect(screen.getByTestId('hardware-info')).toHaveTextContent(
        /NVIDIA GeForce RTX 4060 — GPU detected but CUDA is unavailable/
      )
    )
    // Must NOT mislabel a discrete NVIDIA card as an APU.
    expect(screen.getByTestId('hardware-info')).not.toHaveTextContent(/\(APU\)/)
  })

  it('device toggle: GPU is selectable and default on a CUDA machine, and stores auto', async () => {
    const { deps, api } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: true,
        accelerator: 'cuda' as const,
        gpu_name: 'NVIDIA GeForce RTX 4060',
        gpu_vram_gb: 8,
        recommended_preset: 'high_end'
      }))
    })
    const user = userEvent.setup()
    render(<AdminPanel port={8765} deps={deps} />)

    const gpu = await screen.findByLabelText(/GPU \(recommended\)/i)
    const cpu = screen.getByLabelText(/CPU only/i)
    expect(gpu).toBeEnabled()
    expect(gpu).toBeChecked() // stored device 'auto' shows as GPU

    // Switch to CPU, then save persists 'cpu'.
    await user.click(cpu)
    const saveButton = screen.getByTestId('save-settings')
    await waitFor(() => expect(saveButton).toBeEnabled())
    await user.click(saveButton)
    await waitFor(() => expect(api.updateSettings).toHaveBeenCalledWith({ device: 'cpu' }))
  })

  it('device toggle: back-compat stored device "cuda" shows as GPU on a CUDA machine', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: true,
        accelerator: 'cuda' as const,
        gpu_name: 'NVIDIA GeForce RTX 4060',
        gpu_vram_gb: 8,
        recommended_preset: 'high_end'
      })),
      getSettings: vi.fn(async () => baseSettings({ device: 'cuda' }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const gpu = await screen.findByLabelText(/GPU \(recommended\)/i)
    const cpu = screen.getByLabelText(/CPU only/i)
    expect(gpu).toBeChecked()
    expect(gpu).toBeEnabled()
    expect(cpu).not.toBeChecked()
  })

  it('device toggle: GPU is disabled and CPU forced when no CUDA GPU', async () => {
    const { deps } = makeDeps('idle', {
      getSystemInfo: vi.fn(async () => ({
        cpu_count: 8,
        ram_gb: 16,
        cuda_available: false,
        accelerator: 'integrated' as const,
        gpu_name: 'AMD Radeon(TM) Graphics',
        gpu_vram_gb: null,
        recommended_preset: 'low_end'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const gpu = await screen.findByLabelText(/GPU \(recommended\)/i)
    expect(gpu).toBeDisabled()
    expect(screen.getByLabelText(/CPU only/i)).toBeChecked()
    expect(screen.getByTestId('device-gpu-note')).toBeInTheDocument()
  })
})
