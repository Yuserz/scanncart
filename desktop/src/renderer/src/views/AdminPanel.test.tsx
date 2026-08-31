import { DEFAULT_SETTINGS } from '../lib/settingsDefaults'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AdminPanel } from './AdminPanel'
import type { SettingsDeps } from '../hooks/useSidecarSettings'
import type { ApiClient, SettingsResponse } from '../lib/api'
import { baseSettings as sharedBaseSettings, makeDeps as makeSharedDeps } from '../test/fakes'

// AdminPanel edits every restart-required field, so its fake settings need
// the full list rather than the shared helper's minimal two-field stand-in.
const ADMIN_RESTART_REQUIRED_FIELDS = [
  'active_model',
  'camera_index',
  'capture_width',
  'capture_height',
  'capture_fps',
  'conf_threshold',
  'imgsz',
  'device'
]

function baseSettings(overrides: Partial<SettingsResponse> = {}): SettingsResponse {
  return sharedBaseSettings({
    restart_required_fields: ADMIN_RESTART_REQUIRED_FIELDS,
    ...overrides
  })
}

function makeDeps(
  captureState: string,
  overrides: Partial<ApiClient> = {}
): { deps: SettingsDeps; api: ApiClient } {
  const { deps, api } = makeSharedDeps({
    health: vi.fn(async () => ({ state: captureState, active_model: 'yolo11n.pt', device: 'cpu' })),
    start: vi.fn(async () => ({ state: 'running' })),
    stop: vi.fn(async () => ({ state: 'idle' })),
    getSettings: vi.fn(async () => baseSettings()),
    updateSettings: vi.fn(async (patch) => baseSettings(patch)),
    getPresets: vi.fn(async () => ({
      presets: [
        { name: 'low_end', label: 'Low-end', description: 'weak machine', settings: {} },
        { name: 'mid_range', label: 'Mid-range', description: 'balanced', settings: {} }
      ],
      recommended: 'mid_range'
    })),
    applyPreset: vi.fn(async (name) => baseSettings({ active_model: `${name}.pt` })),
    applyCameraProfile: vi.fn(async () => baseSettings()),
    saveSettings: vi.fn(async () => baseSettings()),
    ...overrides
  })
  return { deps: { ...deps, healthPollMs: 10_000 }, api }
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

  it('editing a restart-required field while running blocks Save with a warning', async () => {
    const { deps } = makeDeps('running')
    const user = userEvent.setup()
    render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('capture-state')).toHaveTextContent('running'))

    const imgszInput = screen.getByLabelText(/Inference size/i)
    await user.clear(imgszInput)
    await user.type(imgszInput, '960')

    await waitFor(() => expect(screen.getByTestId('restart-warning')).toBeInTheDocument())
    expect(screen.getByTestId('restart-warning')).toHaveTextContent('imgsz')
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
        expect.objectContaining({
          active_model: DEFAULT_SETTINGS.active_model,
          capture_width: 1280
        })
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

  it('shows a spinner while settings are loading', () => {
    const { deps } = makeDeps('idle', {
      getSettings: vi.fn(() => new Promise<never>(() => {})) // never resolves
    })
    const { container } = render(<AdminPanel port={8765} deps={deps} />)

    expect(screen.getByText(/Loading settings/i)).toBeInTheDocument()
    expect(container.querySelector('.spinner')).not.toBeNull()
  })

  it('yolo26 options are labeled experimental and show a hardware spec hint when selected', async () => {
    const { deps } = makeDeps('idle', {
      getSettings: vi.fn(async () => baseSettings({ active_model: 'yolo26n.pt' }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const model = await screen.findByLabelText(/Model/i)
    expect(model).toHaveValue('yolo26n.pt')
    expect(screen.getByRole('option', { name: 'yolo26n.pt (experimental)' })).toBeInTheDocument()
    expect(screen.getByTestId('model-spec-hint')).toHaveTextContent(/CPU|GPU/)
  })

  it('supported yolo11 models get no experimental spec hint', async () => {
    const { deps } = makeDeps('idle')
    render(<AdminPanel port={8765} deps={deps} />)

    await screen.findByLabelText(/Model/i)
    expect(screen.getByRole('option', { name: 'yolo11n.pt' })).toBeInTheDocument()
    expect(screen.queryByTestId('model-spec-hint')).not.toBeInTheDocument()
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

  it('names each camera in a dropdown instead of showing a bare index', async () => {
    const { deps } = makeDeps('idle', {
      getCameras: vi.fn(async () => ({
        cameras: [
          { index: 0, name: 'USB2.0 HD UVC WebCam', width: 1280, height: 720 },
          { index: 1, name: 'Logitech StreamCam', width: 1920, height: 1080 }
        ],
        probed: true,
        detail: 'Found 2 camera(s).'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const select = await screen.findByTestId('camera-select')
    expect(select).toHaveTextContent('1 — Logitech StreamCam')
    expect(select).toHaveTextContent('0 — USB2.0 HD UVC WebCam')
    // The resolution is what tells the operator the name landed on the right
    // index, so it still has to be visible — but on the hint line, where it
    // is not truncated by the narrow two-column select. It reflects the
    // *selected* camera, which the fixture leaves at index 0.
    expect(screen.getByTestId('camera-resolution')).toHaveTextContent('1280×720')

    await userEvent.selectOptions(select, '1')
    expect(screen.getByTestId('camera-resolution')).toHaveTextContent('1920×1080')
  })

  it('keeps a saved camera index selectable when no device matches it', async () => {
    // Otherwise the form would silently rewrite the user's saved setting to
    // whatever happened to be plugged in.
    const { deps } = makeDeps('idle', {
      getSettings: vi.fn(async () => baseSettings({ camera_index: 7 })),
      getCameras: vi.fn(async () => ({
        cameras: [{ index: 0, name: 'Only Cam', width: 640, height: 480 }],
        probed: true,
        detail: ''
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const select = await screen.findByTestId('camera-select')
    expect(select).toHaveValue('7')
    expect(select).toHaveTextContent('7 — not detected')
  })

  it('falls back to a plain index input when no camera could be enumerated', async () => {
    const { deps } = makeDeps('idle', {
      getCameras: vi.fn(async () => ({ cameras: [], probed: true, detail: '' }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    await screen.findByTestId('hardware-info')
    expect(screen.queryByTestId('camera-select')).not.toBeInTheDocument()
  })

  it('disables the camera rescan while capture is running', async () => {
    // Probing opens every device, which would fight the running pipeline.
    const { deps } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)

    expect(await screen.findByTestId('rescan-cameras')).toBeDisabled()
  })

  it('offers Stop capture in the action bar while running', async () => {
    // The bar is sticky, so unlike the header it cannot scroll out of view.
    const { deps } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)

    const bar = await screen.findByTestId('admin-actions')
    expect(within(bar).getByTestId('stop-capture-inline')).toBeEnabled()
  })

  it('hides Stop capture when idle', async () => {
    const { deps } = makeDeps('idle')
    render(<AdminPanel port={8765} deps={deps} />)

    await screen.findByTestId('hardware-info')
    expect(screen.queryByTestId('stop-capture-inline')).not.toBeInTheDocument()
  })

  it('puts Stop capture inside the restart-required warning', async () => {
    // The warning used to say "stop capture to change X" while offering no way
    // to do it — the user had to leave for the Live view and come back.
    const { deps } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)

    const model = await screen.findByLabelText(/Model/i)
    await userEvent.selectOptions(model, 'yolo11s.pt')

    const warning = await screen.findByTestId('restart-warning')
    expect(warning).toHaveTextContent('active_model')
    // The warning explains; the sticky bar carries the action, so the two are
    // not duplicated on screen.
    expect(within(warning).queryByRole('button')).not.toBeInTheDocument()
    const bar = screen.getByTestId('admin-actions')
    expect(within(bar).getByTestId('save-and-restart')).toBeInTheDocument()
  })

  it('stopping capture keeps unsaved edits', async () => {
    // Stop must not re-read settings: AdminPanel clears `draft` whenever the
    // settings object changes identity, so a reload here would discard the
    // very edit the user stopped capture in order to make.
    const { deps, api } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)

    const model = await screen.findByLabelText(/Model/i)
    await userEvent.selectOptions(model, 'yolo11s.pt')
    await userEvent.click(screen.getByTestId('stop-capture-inline'))

    expect(api.stop).toHaveBeenCalled()
    await waitFor(() => expect(screen.getByLabelText(/Model/i)).toHaveValue('yolo11s.pt'))
  })

  it('Save & restart does stop -> save -> start in one action', async () => {
    // Most settings are restart-required, so this is the common path; doing it
    // by hand is three actions across two views.
    const { deps, api } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)

    const model = await screen.findByLabelText(/Model/i)
    await userEvent.selectOptions(model, 'yolo11s.pt')
    await userEvent.click(await screen.findByTestId('save-and-restart'))

    await waitFor(() => expect(api.start).toHaveBeenCalled())
    expect(api.stop).toHaveBeenCalled()
    expect(api.updateSettings).toHaveBeenCalledWith({ active_model: 'yolo11s.pt' })
    const order = [
      vi.mocked(api.stop).mock.invocationCallOrder[0],
      vi.mocked(api.updateSettings).mock.invocationCallOrder[0],
      vi.mocked(api.start).mock.invocationCallOrder[0]
    ]
    expect(order).toEqual([...order].sort((a, b) => a - b))
  })

  it('does not offer Save & restart when nothing is pending', async () => {
    const { deps } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)

    await screen.findByTestId('hardware-info')
    expect(screen.queryByTestId('save-and-restart')).not.toBeInTheDocument()
  })

  it('badges only the fields that apply instantly, not the restart-required ones', async () => {
    // 14 of 16 fields are restart-required, so badging those made the badge
    // carry no information. The exceptions are the surprising ones.
    const { deps } = makeDeps('idle')
    render(<AdminPanel port={8765} deps={deps} />)

    await screen.findByTestId('hardware-info')
    expect(screen.queryByText(/restart required/i)).not.toBeInTheDocument()
    expect(screen.getAllByText(/applies instantly/i).length).toBeGreaterThan(0)
  })

  it('hides the server warning that lists every restart-required field', async () => {
    // 16 items of comma-separated prose the badges and inline warning cover.
    const { deps } = makeDeps('running', {
      getSettings: vi.fn(async () =>
        baseSettings({
          warnings: [
            'Capture is running — active_model, camera_index, capture_fps require stopping capture first.',
            'cloud_api sends every inference frame to Roboflow.'
          ]
        })
      )
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const list = await screen.findByTestId('server-warnings')
    expect(list).toHaveTextContent('cloud_api sends every inference frame')
    expect(list).not.toHaveTextContent('require stopping capture first')
  })

  it('counts unsaved changes in the action bar', async () => {
    const { deps } = makeDeps('idle')
    render(<AdminPanel port={8765} deps={deps} />)

    expect(await screen.findByTestId('pending-count')).toHaveTextContent('No changes')

    await userEvent.selectOptions(await screen.findByLabelText(/Model/i), 'yolo11s.pt')
    expect(screen.getByTestId('pending-count')).toHaveTextContent('1 unsaved change')
  })

  it('confirms a save, and clears the confirmation on the next edit', async () => {
    // Saving used to just go quiet, leaving no signal it had worked.
    const { deps } = makeDeps('idle')
    render(<AdminPanel port={8765} deps={deps} />)

    await userEvent.selectOptions(await screen.findByLabelText(/Model/i), 'yolo11s.pt')
    await userEvent.click(screen.getByTestId('save-settings'))
    await waitFor(() => expect(screen.getByTestId('pending-count')).toHaveTextContent('Saved'))

    await userEvent.selectOptions(screen.getByLabelText(/Model/i), 'yolo11m.pt')
    expect(screen.getByTestId('pending-count')).not.toHaveTextContent('Saved')
  })

  it('shows live image quality with a warning when the frame is too dark', async () => {
    // Brightness 23/255 was the real cause of "detection is broken".
    const { deps } = makeDeps('running', {
      getCameraQuality: vi.fn(async () => ({
        available: true,
        brightness: 23,
        contrast: 27,
        sharpness: 4.8,
        capture_fps: 12,
        target_fps: 60,
        verdicts: { brightness: 'low', sharpness: 'low', capture_fps: 'low' },
        detail: ''
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const panel = await screen.findByTestId('camera-quality')
    expect(panel).toHaveTextContent('23')
    expect(within(panel).getAllByTestId('quality-low').length).toBeGreaterThan(0)
  })

  it('marks a failing metric with more than colour, for a colourblind operator', async () => {
    // This panel's whole purpose is making a bad reading visible, so a
    // failing metric must be identifiable without relying on hue: a visible
    // symbol plus text a screen reader can announce.
    const { deps } = makeDeps('running', {
      getCameraQuality: vi.fn(async () => ({
        available: true,
        brightness: 23,
        contrast: 27,
        sharpness: 200,
        capture_fps: 60,
        target_fps: 60,
        verdicts: { brightness: 'low', sharpness: 'ok', capture_fps: 'ok' },
        detail: ''
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    const panel = await screen.findByTestId('camera-quality')
    const failing = within(panel).getAllByTestId('quality-low')
    expect(failing).toHaveLength(1)
    // A symbol is present in the rendered text, not just a CSS colour class.
    expect(failing[0].textContent).toMatch(/[^\d\s]/)
    // And an accessible name conveys the failing state to a screen reader.
    expect(failing[0]).toHaveTextContent(/outside the expected range/i)

    const passing = within(panel).getAllByTestId('quality-ok')
    for (const el of passing) {
      expect(el).not.toHaveTextContent(/outside the expected range/i)
    }
  })

  it('shows the calibration result with its measured evidence, and applies on request', async () => {
    const { deps, api } = makeDeps('idle')
    render(<AdminPanel port={8765} deps={deps} />)

    await userEvent.click(await screen.findByTestId('calibrate-camera'))

    const card = await screen.findByTestId('calibration-result')
    expect(card).toHaveTextContent('30.3') // the measured evidence
    expect(card).toHaveTextContent('12.3')

    await userEvent.click(within(card).getByTestId('apply-profile'))
    expect(api.applyCameraProfile).toHaveBeenCalled()
  })

  it('disables calibration while capture is running', async () => {
    const { deps } = makeDeps('running')
    render(<AdminPanel port={8765} deps={deps} />)
    expect(await screen.findByTestId('calibrate-camera')).toBeDisabled()
  })

  it('disables Apply while a request is in flight, preventing a double-submit', async () => {
    const applyGate: { resolve: (() => void) | null } = { resolve: null }
    const { deps, api } = makeDeps('idle', {
      applyCameraProfile: vi.fn(
        () =>
          new Promise<SettingsResponse>((resolve) => {
            applyGate.resolve = () => resolve(baseSettings())
          })
      )
    })
    render(<AdminPanel port={8765} deps={deps} />)

    await userEvent.click(await screen.findByTestId('calibrate-camera'))
    const card = await screen.findByTestId('calibration-result')
    const applyButton = within(card).getByTestId('apply-profile')

    await userEvent.click(applyButton)
    expect(applyButton).toBeDisabled()
    expect(api.applyCameraProfile).toHaveBeenCalledTimes(1)

    applyGate.resolve?.()
    await waitFor(() => expect(applyButton).not.toBeDisabled())
  })

  it('shows an explanatory message and hides Apply when the camera has nothing to recommend', async () => {
    const { deps } = makeDeps('idle', {
      calibrateCamera: vi.fn(async () => ({
        device_key: 'Fake Cam:0:1280x720',
        backend: 'msmf',
        width: 1280,
        height: 720,
        fps_auto_exposure: 12.3,
        fps_capped_exposure: 12.3,
        controls: { brightness: false, exposure: false, gain: false, focus: false },
        recommended: {},
        measured_at: 1
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    await userEvent.click(await screen.findByTestId('calibrate-camera'))
    const card = await screen.findByTestId('calibration-result')
    expect(card).toHaveTextContent(/No settings to change/i)
    expect(within(card).queryByTestId('apply-profile')).not.toBeInTheDocument()
  })

  it('hides the quality readout when capture is not running', async () => {
    const { deps } = makeDeps('idle', {
      getCameraQuality: vi.fn(async () => ({
        available: false,
        brightness: 0,
        contrast: 0,
        sharpness: 0,
        capture_fps: 0,
        target_fps: 0,
        verdicts: {},
        detail: 'Start capture to measure the image.'
      }))
    })
    render(<AdminPanel port={8765} deps={deps} />)

    await screen.findByTestId('hardware-info')
    expect(screen.queryByTestId('camera-quality')).not.toBeInTheDocument()
  })
})
