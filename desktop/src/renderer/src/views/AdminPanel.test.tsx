import { DEFAULT_SETTINGS } from '../lib/settingsDefaults'
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AdminPanel } from './AdminPanel'
import type { SettingsDeps } from '../hooks/useSidecarSettings'
import type { ApiClient, SettingsResponse } from '../lib/api'
import {
  baseSettings as sharedBaseSettings,
  datasetStatus,
  installedUnrecorded,
  installedV2,
  installedV2DistanceSplit,
  installedV2Unmeasured,
  makeDeps as makeSharedDeps,
  unrecordedResizeMode,
  VALIDATION_V2
} from '../test/fakes'

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

  it('renders the class allowlist as a text field in Admin and saves a parsed array', async () => {
    // This one lives in Admin only: the tuning card renders numeric sliders,
    // so a field placed on the live side gets a range input bound to a string
    // array instead of an editable list.
    const { deps, api } = makeDeps('idle')
    const user = userEvent.setup()
    const { container } = render(<AdminPanel port={8765} deps={deps} />)

    await waitFor(() => expect(screen.getByTestId('capture-state')).toHaveTextContent('idle'))

    const input = container.querySelector<HTMLInputElement>('#class_allowlist')
    expect(input).toHaveAttribute('type', 'text')

    await user.type(input!, 'bottle, cup')
    // The text survives verbatim — only the draft holds the parsed array, so
    // the comma is not eaten before the next name starts.
    expect(input).toHaveValue('bottle, cup')

    await user.click(screen.getByTestId('save-settings'))
    await waitFor(() =>
      expect(api.updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({ class_allowlist: ['bottle', 'cup'] })
      )
    )
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
          capture_width: 640
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
    // Must outlast waitFor's ~50 ms polling, or the transient error panel
    // (visible only between the failed load and the auto-retry) is skipped
    // entirely and this flakes.
    deps.retryDelayMs = 200
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

  describe('after the tuning fields moved to Live', () => {
    it('no longer renders the relocated fields', async () => {
      const { deps } = makeDeps('idle')
      render(<AdminPanel port={8765} deps={deps} />)
      await screen.findByTestId('hardware-info')
      for (const key of [
        'conf_threshold',
        'infer_frame_skip',
        'preview_height',
        'preview_max_fps',
        'track_expiry_s'
      ]) {
        expect(document.getElementById(key)).toBeNull()
      }
    })

    it('no longer renders the calibration card or the quality readout', async () => {
      const { deps } = makeDeps('idle')
      render(<AdminPanel port={8765} deps={deps} />)
      await screen.findByTestId('hardware-info')
      expect(screen.queryByTestId('calibrate-camera')).toBeNull()
      expect(screen.queryByTestId('camera-quality')).toBeNull()
    })

    it('still renders the fields that need a restart', async () => {
      const { deps } = makeDeps('idle')
      render(<AdminPanel port={8765} deps={deps} />)
      await screen.findByTestId('hardware-info')
      expect(document.getElementById('capture_width')).not.toBeNull()
      expect(document.getElementById('imgsz')).not.toBeNull()
    })
  })

  describe('dataset labeling progress', () => {
    it('points at the tool instead of showing zeros when no snapshot exists', async () => {
      // The default fake is the real first-run answer (`available: false`). Rendering
      // "0 / 0 labeled" there would read as "nothing is done" rather than "nobody has
      // looked yet", which is the opposite instruction for the operator.
      const { deps } = makeDeps('idle')
      render(<AdminPanel port={8765} deps={deps} />)

      const panel = await screen.findByTestId('dataset-unavailable')
      expect(panel).toHaveTextContent(/sidecar\/tools\/label_progress\.py/)
      expect(screen.queryByTestId('dataset-summary')).toBeNull()
      expect(screen.queryByTestId('dataset-backlog')).toBeNull()
    })

    it('renders per-class progress and says how stale the reading is', async () => {
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () =>
          datasetStatus({ generated_at: '2026-09-22T04:35:09', age_seconds: 120 })
        )
      })
      render(<AdminPanel port={8765} deps={deps} />)

      expect(await screen.findByTestId('dataset-summary')).toHaveTextContent('40 / 1383')
      expect(screen.getByTestId('dataset-summary')).toHaveTextContent('2.9%')
      // Freshness is not decoration: the numbers are a snapshot, and the panel must
      // not present them as live.
      expect(screen.getByTestId('dataset-freshness')).toHaveTextContent('2 min ago')
      expect(screen.getByTestId('dataset-freshness')).toHaveTextContent('2026-09-22T04:35:09')

      // The panel shows the worklist, not a per-class table: the same counts as work rather
      // than as a report. The distance axis is the part a class table cannot express, and a
      // distance bucket nobody opened is what actually stalls a project.
      const rows = within(screen.getByTestId('dataset-backlog')).getAllByRole('listitem')
      expect(rows).toHaveLength(5)
      expect(rows[0]).toHaveTextContent('silver_swan_sukang_puti_200ML')
      expect(rows[0]).toHaveTextContent('close')
      expect(rows[0]).toHaveTextContent('242 left')
      expect(rows[0]).toHaveTextContent('0/242')
      expect(screen.getByTestId('dataset-backlog-summary')).toHaveTextContent('5 cell(s) left')
    })

    it('orders the worklist biggest first, as the sidecar sent it', async () => {
      // The order is a claim about priority, and only one place may make it. The panel renders
      // the list in the order it arrived rather than sorting it, so this asserts the arrival
      // order survives - re-sorting here would be a second opinion about what to do next.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () => datasetStatus())
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const rows = within(await screen.findByTestId('dataset-backlog')).getAllByRole('listitem')
      expect(rows.map((r) => within(r).getByTestId('dataset-backlog-left').textContent)).toEqual([
        '242',
        '234',
        '154',
        '150',
        '50'
      ])
      // The two halves of a row come from one pair of counts, so a row that had 30/184 labeled
      // must derive 154 left rather than repeat the total.
      expect(rows[2]).toHaveTextContent('30/184')
      // Ranked, so "work down the list" is an instruction a person can follow.
      expect(rows[0]).toHaveTextContent(/^1/)
      expect(rows[4]).toHaveTextContent(/^5/)
    })

    it('flags the hard negatives as null-marking work, not drawing work', async () => {
      // The one row where "label this" means the opposite: these frames must be marked null
      // (N). Left unannotated they are silently excluded from the version, so the instruction
      // has to be on the row rather than in a doc nobody has open.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () => datasetStatus())
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const rows = within(await screen.findByTestId('dataset-backlog')).getAllByRole('listitem')
      const background = rows[4]
      expect(background).toHaveTextContent('(negative: background frames, nothing to draw)')
      // A distance of "" would leave the slot looking broken, so it says what these frames are.
      expect(background).toHaveTextContent('background frames')
      expect(within(background).getByTestId('dataset-backlog-null')).toHaveTextContent('null')
      // And exactly one row carries that instruction - a second one would mean the flag is
      // being applied to a class that does get boxes.
      expect(screen.getAllByTestId('dataset-backlog-null')).toHaveLength(1)
    })

    it('says the worklist is empty once nothing is left to label', async () => {
      // A finished backlog must not render a stale list of 0-remaining rows, and must not
      // read as "the panel is broken" either.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () => datasetStatus({ labeling_backlog: [] }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      expect(await screen.findByTestId('dataset-backlog-done')).toHaveTextContent(
        'Nothing left to label'
      )
      expect(screen.queryByTestId('dataset-backlog')).toBeNull()
      expect(screen.queryByTestId('dataset-backlog-summary')).toBeNull()
    })

    it('surfaces null annotations and class mismatches when they exist', async () => {
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () => datasetStatus({ null_annotations: 5, mismatches: 2 }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const freshness = await screen.findByTestId('dataset-freshness')
      expect(freshness).toHaveTextContent('5 null (background)')
      expect(freshness).toHaveTextContent('2 class mismatch(es)')
    })

    it('refreshes on demand rather than polling', async () => {
      const getDatasetStatus = vi.fn(async () => datasetStatus())
      const { deps } = makeDeps('idle', { getDatasetStatus })
      const user = userEvent.setup()
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('dataset-summary')
      expect(getDatasetStatus).toHaveBeenCalledTimes(1)

      await user.click(screen.getByTestId('refresh-dataset'))
      await waitFor(() => expect(getDatasetStatus).toHaveBeenCalledTimes(2))
    })

    it('does not offer a refresh before there is anything to refresh', async () => {
      const { deps } = makeDeps('idle')
      render(<AdminPanel port={8765} deps={deps} />)
      await screen.findByTestId('dataset-unavailable')
      expect(screen.queryByTestId('refresh-dataset')).toBeNull()
    })
  })

  describe('the model picker', () => {
    it('offers weights the sidecar discovered, with no renderer edit', async () => {
      // The point of the endpoint: a locally trained model becomes selectable the moment it
      // is copied into sidecar/models/. Before this, the picker was a fixed list, so the one
      // moment the model changed was the one moment the app needed a code change.
      const { deps } = makeDeps('idle', {
        getModels: vi.fn(async () => ({
          stock: ['yolo11n.pt'],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const select = (await screen.findByLabelText(/Model/i)) as HTMLSelectElement
      const values = Array.from(select.options).map((o) => o.value)
      expect(values).toContain('models/scanncart-grocery-v2.pt')
      expect(values).toContain('yolo11n.pt')
      // Labelled for a person, and named for the 8 classes v2 adds rather than v1's 7.
      const label = Array.from(select.options).find(
        (o) => o.value === 'models/scanncart-grocery-v2.pt'
      )?.textContent
      expect(label).toContain('v2')
      expect(label).toContain('8 SKUs')
    })

    it('lists each installed weight with the resize_mode it requires', async () => {
      // The requirement is a fact recorded beside the weights at install time, not something
      // the panel can infer from a filename - so it is shown for whatever is on disk, which
      // is what makes a hand-copied weight as visible as the one the tool installed.
      const { deps } = makeDeps('idle', {
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2(), installedUnrecorded()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const list = await screen.findByTestId('installed-models')
      expect(list).toHaveTextContent('models/scanncart-grocery-v2.pt')
      expect(list).toHaveTextContent('resize_mode: stretch')
      expect(list).toHaveTextContent('snc-grocery version 2')
      // Two unknowns are not one: nothing recorded is said out loud, because `auto` is not a
      // safe default for weights whose training geometry nobody wrote down.
      expect(list).toHaveTextContent('no recorded requirement')
      // And the other fact the record carries: what these weights predict. Shown as a count
      // for the installed one, absent for the hand-copied weight - where the class list is as
      // unknown as the geometry, not zero.
      expect(list).toHaveTextContent('8 classes')
      expect(
        within(screen.getByTestId('installed-model-models/hand-copied.pt')).queryByText(/classes/)
      ).toBeNull()
    })

    it('flags a weight whose recorded class list is not the roster, before it runs', async () => {
      // The listing is the earliest place this can be said. A project whose class list was split
      // by distance trains one head output per product-and-distance, so every box comes back
      // under a label the app has no use for and the item log fills with near-duplicates of one
      // product - with nothing erroring, and nothing in the filename or the checkpoint to hint
      // at it. Until the names are recorded the only way to find out is to run it.
      const { deps } = makeDeps('idle', {
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2DistanceSplit(), installedUnrecorded()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const warning = await screen.findByTestId(
        'installed-model-class-warning-models/scanncart-grocery-v2.pt'
      )
      // The sidecar's own sentence, rendered rather than re-worded: it is the process holding
      // the roster, and a second phrasing here could tell a different story about one list.
      expect(warning).toHaveTextContent('carry a distance')
      expect(warning).toHaveTextContent('Retrain')
      // The hand-copied weight beside it is silent, because no class list was recorded for it -
      // an unknown list is not a wrong one, and a finding about it would be an invention.
      expect(screen.queryByTestId('installed-model-class-warning-models/hand-copied.pt')).toBeNull()
    })

    it('does not flag the reserved word, because `auto` now uses the requirement', async () => {
      // This used to be the warning's main case: `auto` was the tempting answer and the wrong
      // one, resolving to letterbox for a locally trained .pt. The sidecar honours the record
      // now, so the panel has to stop calling the correct configuration a mistake - a warning
      // here would tell an operator to override the weights with a value that breaks them.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/scanncart-grocery-v2.pt',
            resize_mode: 'auto'
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('installed-models')
      expect(screen.queryByTestId('model-resize-mismatch')).toBeNull()
      expect(await screen.findByTestId('installed-models')).toHaveTextContent('auto uses it')
    })

    it('shows what the selected weights scored, per class, on the split they were measured on', async () => {
      // The record is the only place this can come from: the numbers otherwise live in a
      // terminal log from a command run once, and the person choosing between generations is
      // the person who has to read them.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/scanncart-grocery-v2.pt',
            resize_mode: 'auto'
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const block = await screen.findByTestId('model-validation')
      // The split is named, because the same 0.95 means something different on `valid`.
      expect(within(block).getByTestId('model-validation-test')).toHaveTextContent('test')
      expect(block).toHaveTextContent('floor 0.85')
      expect(block).toHaveTextContent('1 of 3 below the floor')
      expect(block).toHaveTextContent('mAP50 0.880')
      // Per class, with the count that says whether the number means anything.
      expect(screen.getByTestId('model-validation-class-century-tuna')).toHaveTextContent(
        '0.620 (n=30)'
      )
      expect(screen.getByTestId('model-validation-class-milo')).toHaveTextContent('0.950 (n=12)')
    })

    it('names an unmeasured class instead of printing a zero for it', async () => {
      // Ultralytics answers 0.000 for a class the split holds no instances of, and a zero
      // reads as a total miss - which would send the operator after images of `lucky-me` when
      // what is missing is captures *in this split*. The two are opposite instructions.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/scanncart-grocery-v2.pt' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const cell = await screen.findByTestId('model-validation-class-lucky-me')
      expect(cell).toHaveTextContent('not measured')
      expect(cell).not.toHaveTextContent('0.000')
      expect(cell).toHaveClass('unmeasured')
      // The classes that were measured are marked as such, so `unmeasured` is a statement
      // about the split rather than about how the list renders by default.
      expect(screen.getByTestId('model-validation-class-bear-brand')).toHaveClass('ok')
      expect(screen.getByTestId('model-validation-class-century-tuna')).toHaveClass('below')
    })

    it('shows the same recall per distance, so a far miss cannot hide behind the split', async () => {
      // The reason the breakdown exists. `bear-brand` scores 0.900 on the split as a whole - a
      // pass - while scoring 0.620 at `far`, and the split's number is a mean over distances
      // that cannot say so. `far` is the bucket this dataset exists to fix.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/scanncart-grocery-v2.pt' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const grid = await screen.findByTestId('model-validation-by-distance')
      // Each distance names the frame count it ran on, so 1.000 over three images cannot read
      // as a stronger claim than 0.900 over sixty.
      expect(grid).toHaveTextContent('close')
      expect(grid).toHaveTextContent('34 img')
      expect(grid).toHaveTextContent('far')
      expect(grid).toHaveTextContent('28 img')

      // The pass and the miss, sitting in the same row.
      expect(screen.getByTestId('model-validation-class-bear-brand')).toHaveClass('ok')
      expect(screen.getByTestId('model-validation-close-bear-brand')).toHaveTextContent('0.970')
      const farCell = screen.getByTestId('model-validation-far-bear-brand')
      expect(farCell).toHaveTextContent('0.620')
      expect(farCell).toHaveClass('below')
      // And the miss is named below the table, because a colour alone is not a to-do list.
      expect(screen.getByTestId('model-validation-distance-misses')).toHaveTextContent(
        'Below the floor at a distance: far bear-brand 0.620'
      )
    })

    it('prints a dash where a distance held no instances, not a zero', async () => {
      // Two ways a cell can be absent, one rendering: `lucky-me` is scored at `close` and the
      // split holds none of it there (`recall: null`), and `far` has no frames of it at all.
      // Neither is a miss, and a 0.000 would send the operator after images of the item when
      // what is missing is captures at that distance.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/scanncart-grocery-v2.pt' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const grid = await screen.findByTestId('model-validation-by-distance')
      expect(grid.textContent).not.toContain('0.000')
      for (const cell of ['model-validation-close-lucky-me', 'model-validation-far-lucky-me']) {
        expect(screen.getByTestId(cell)).toHaveTextContent('—')
        expect(screen.getByTestId(cell)).toHaveClass('unmeasured')
      }
      // Cells a distance *did* score carry their own marking - 0.88 clears the floor and 0.71
      // does not - so the dash is a statement about the split rather than about how this table
      // renders by default.
      expect(screen.getByTestId('model-validation-close-century-tuna')).toHaveClass('ok')
      expect(screen.getByTestId('model-validation-far-century-tuna')).toHaveClass('below')
    })

    it('says nothing is below the floor per distance when nothing is', async () => {
      // The mirror of the miss line. A line that only ever appears when something is wrong
      // leaves "all clear" and "not measured" looking the same.
      const passes = {
        ...VALIDATION_V2,
        per_distance: [
          {
            distance: 'far',
            images: 28,
            aggregates: { mAP50: 0.51 },
            per_class: [{ name: 'bear-brand', recall: 0.9, instances: 8 }]
          }
        ]
      }
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/scanncart-grocery-v2.pt' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2({ validation: [passes] })],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('model-validation-by-distance')
      const line = screen.getByTestId('model-validation-distance-misses')
      expect(line).toHaveTextContent('No class is below the floor at any distance')
      expect(line).not.toHaveTextContent('Below the floor at a distance')
      expect(screen.getByTestId('model-validation-far-bear-brand')).toHaveClass('ok')
    })

    it('renders no distance grid for a record that has none', async () => {
      // An older record, or one written with `--no-per-distance`. The split's own numbers must
      // still render in full: the breakdown is extra detail, not a precondition for the score.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/scanncart-grocery-v2.pt' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2({ validation: [{ ...VALIDATION_V2, per_distance: [] }] })],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      expect(await screen.findByTestId('model-validation-test')).toHaveTextContent(
        '1 of 3 below the floor'
      )
      expect(screen.getByTestId('model-validation-class-milo')).toHaveTextContent('0.950')
      expect(screen.queryByTestId('model-validation-by-distance')).toBeNull()
    })

    it('says a record exists but has no score, rather than rendering nothing', async () => {
      // Installed by the tool, `--val` never run. Silence here would be indistinguishable
      // from a panel that failed to render the numbers, and there is a command that fixes it.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/scanncart-grocery-v2.pt' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2Unmeasured()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      expect(await screen.findByTestId('model-validation-missing')).toHaveTextContent(
        'train_model.py --val'
      )
      expect(screen.queryByTestId('model-validation')).toBeNull()
    })

    it('shows no score for a weight nothing was recorded about', async () => {
      // No record means no requirement *and* no numbers. The block cannot say "not measured"
      // on behalf of a file it knows nothing about - only the record can say that.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () => baseSettings({ active_model: 'models/hand-copied.pt' })),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('installed-models')
      expect(screen.queryByTestId('model-validation')).toBeNull()
      expect(screen.queryByTestId('model-validation-missing')).toBeNull()
    })

    it('shows what `auto` gives for weights nobody recorded anything about', async () => {
      // The unrecorded case is where `auto` is still a guess, so the guess is named rather
      // than left to be discovered - and it is a different sentence from the recorded one,
      // because "needs stretch" and "will use letterbox" must not read alike.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/hand-copied.pt', resize_mode: 'auto' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const list = await screen.findByTestId('installed-models')
      expect(list).toHaveTextContent('no recorded requirement')
      expect(list).toHaveTextContent('auto gives letterbox')
    })

    it('offers to record the assumption, with the sentence it answers', async () => {
      // The one resize warning with a remedy the app can perform itself. The sentence and the
      // button arrive together in one structured entry, so the label cannot describe a
      // different mode than the sentence above it.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/hand-copied.pt',
            resize_mode: 'auto',
            unrecorded_resize_mode: unrecordedResizeMode()
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const entry = await screen.findByTestId('unrecorded-resize-mode')
      expect(entry).toHaveTextContent('has no record of the geometry')
      expect(screen.getByTestId('record-resize-mode')).toHaveTextContent('Record it now: letterbox')
    })

    it('writes the requirement for the weights the warning names', async () => {
      const record = vi.fn(async () => ({
        stock: [],
        installed: [installedUnrecorded({ resize_mode: 'letterbox', recorded: true })],
        directory: 'models/'
      }))
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/hand-copied.pt',
            resize_mode: 'auto',
            unrecorded_resize_mode: unrecordedResizeMode()
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded()],
          directory: 'models/'
        })),
        recordResizeMode: record
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const user = userEvent.setup()
      await user.click(await screen.findByTestId('record-resize-mode'))
      // The mode the sidecar named, not one derived here: it is what `auto` had already
      // resolved to, which is why recording cannot change what the detector does.
      await waitFor(() => expect(record).toHaveBeenCalledWith('models/hand-copied.pt', 'letterbox'))
    })

    it('confirms by reading the weights list back rather than by trusting the click', async () => {
      // The confirmation is derived from the refreshed `/api/models` answer, so it shows the
      // requirement the file now carries - not the one that was asked for. A write that landed
      // somewhere else would leave this warning in place, which is the honest outcome.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/hand-copied.pt',
            resize_mode: 'auto',
            unrecorded_resize_mode: unrecordedResizeMode()
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded()],
          directory: 'models/'
        })),
        recordResizeMode: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded({ resize_mode: 'letterbox', recorded: true })],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const user = userEvent.setup()
      await user.click(await screen.findByTestId('record-resize-mode'))

      await waitFor(() =>
        expect(screen.getByTestId('unrecorded-resize-mode')).toHaveTextContent('Recorded')
      )
      const entry = screen.getByTestId('unrecorded-resize-mode')
      expect(entry).not.toHaveTextContent('has no record of the geometry')
      expect(screen.queryByTestId('record-resize-mode')).toBeNull()
      // And the listing agrees, because it is where the confirmation came from.
      expect(screen.getByTestId('installed-models')).toHaveTextContent('auto uses it')
    })

    it('keeps the warning and the button when the write is refused', async () => {
      // Recording is a write to a file that may have gone away since the listing was read. A
      // refusal must leave the warning exactly as it was - an acknowledgement rendered on
      // failure would claim the assumption had been answered when it had not.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/hand-copied.pt',
            resize_mode: 'auto',
            unrecorded_resize_mode: unrecordedResizeMode()
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded()],
          directory: 'models/'
        })),
        recordResizeMode: vi.fn(async () => {
          throw new Error('No weights at models/hand-copied.pt; nothing was written.')
        })
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const user = userEvent.setup()
      await user.click(await screen.findByTestId('record-resize-mode'))

      expect(await screen.findByTestId('admin-error')).toHaveTextContent('nothing was written')
      expect(screen.getByTestId('unrecorded-resize-mode')).toHaveTextContent(
        'has no record of the geometry'
      )
      expect(screen.getByTestId('record-resize-mode')).toBeInTheDocument()
    })

    it('offers nothing to record once a requirement is recorded', async () => {
      // `unrecorded_resize_mode` is null whenever there is no assumption to report, and the
      // button's presence is that field alone - so a recorded weight, an explicit mode, stock
      // weights and a remote backend all render nothing rather than an action with no object.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/scanncart-grocery-v2.pt', resize_mode: 'auto' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('installed-models')
      expect(screen.queryByTestId('unrecorded-resize-mode')).toBeNull()
      expect(screen.queryByTestId('record-resize-mode')).toBeNull()
    })

    it('flags an explicit value too, and stops once it matches', async () => {
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/scanncart-grocery-v2.pt',
            resize_mode: 'letterbox'
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      const { unmount } = render(<AdminPanel port={8765} deps={deps} />)

      expect(await screen.findByTestId('model-resize-mismatch')).toHaveTextContent(
        'the form has letterbox'
      )
      unmount()

      // Setting the field is the fix, and the warning has to go away when it is applied -
      // and use the *pending* value while it is still unsaved, not the saved one.
      const pending = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({
            active_model: 'models/scanncart-grocery-v2.pt',
            resize_mode: 'letterbox'
          })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedV2()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={pending.deps} />)
      fireEvent.change(await screen.findByLabelText(/Frame fitting/i), {
        target: { value: 'stretch' }
      })
      expect(screen.queryByTestId('model-resize-mismatch')).toBeNull()
    })

    it('does not invent a requirement for weights that have none recorded', async () => {
      // A hand-copied .pt: the panel says what it knows and nothing more. Claiming `stretch`
      // here would be indistinguishable from a record, which is the one thing this block
      // exists to distinguish.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () =>
          baseSettings({ active_model: 'models/hand-copied.pt', resize_mode: 'auto' })
        ),
        getModels: vi.fn(async () => ({
          stock: [],
          installed: [installedUnrecorded()],
          directory: 'models/'
        }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('installed-models')
      expect(screen.queryByTestId('model-resize-mismatch')).toBeNull()
    })

    it('keeps the requirement out of the way when nothing is installed', async () => {
      const { deps } = makeDeps('idle')
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByLabelText(/Model/i)
      expect(screen.queryByTestId('installed-models')).toBeNull()
      expect(screen.queryByTestId('model-resize-mismatch')).toBeNull()
    })

    it('keeps a selected model selectable even when its file is missing', async () => {
      // A config pointing at nothing has to be visible to be diagnosable - silently
      // dropping the option would rewrite the field's displayed value instead.
      const { deps } = makeDeps('idle', {
        getSettings: vi.fn(async () => baseSettings({ active_model: 'models/gone.pt' })),
        getModels: vi.fn(async () => ({ stock: [], installed: [], directory: 'models/' }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const select = (await screen.findByLabelText(/Model/i)) as HTMLSelectElement
      expect(Array.from(select.options).map((o) => o.value)).toContain('models/gone.pt')
      expect(select.value).toBe('models/gone.pt')
    })
  })

  describe('capture sessions in the dataset section', () => {
    it('shows each session with its split spread and what the shared one costs', async () => {
      // The block answers one question: is the eventual test number an unseen session, or
      // held-out frames of a session the model trained on? With one capture it is the
      // second, and the panel has to say so rather than let the number travel unqualified.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () => datasetStatus())
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const rows = within(await screen.findByTestId('dataset-session-rows')).getAllByRole('row')
      expect(rows).toHaveLength(3) // header + the two sessions
      expect(rows[1]).toHaveTextContent('s1')
      expect(rows[1]).toHaveTextContent('969')
      expect(rows[1]).toHaveTextContent('278')
      expect(rows[1]).toHaveTextContent('136')
      expect(rows[1]).toHaveClass('dataset-session-shared')
      expect(rows[2]).toHaveTextContent('neg1')
      // An empty split renders as a dash, not a 0 - a zero would read as "no images at
      // all" rather than "this session sent none there".
      expect(rows[2]).toHaveTextContent('—')

      const verdict = screen.getByTestId('dataset-session-verdict')
      expect(verdict).toHaveTextContent('s1')
      expect(verdict).toHaveTextContent('held-out frames of a session the model trained on')
    })

    it('calls test an unseen session once no session is on both sides of the line', async () => {
      // The state Tier C buys: s3 exists only in test, and s1 - now spread over train and
      // valid - is the ordinary remainder rather than a leak. It must not be flagged,
      // which is what keeps the warning worth reading when it does appear.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () =>
          datasetStatus({
            sessions: [
              { name: 's1', train: 969, valid: 278, test: 0, decided: 0, total: 1247, splits: 2 },
              { name: 's3', train: 0, valid: 0, test: 136, decided: 0, total: 136, splits: 1 }
            ]
          })
        )
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const rows = within(await screen.findByTestId('dataset-session-rows')).getAllByRole('row')
      expect(rows[1]).not.toHaveClass('dataset-session-shared')
      expect(screen.getByTestId('dataset-session-verdict')).toHaveTextContent(
        'test is genuinely an unseen capture session'
      )
    })

    it('hides the block when the snapshot predates the sessions field', async () => {
      // A snapshot written by an older tool is the ordinary case on a machine that has not
      // re-run it, so it renders as it did before rather than as an empty table.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () => datasetStatus({ sessions: [] }))
      })
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('dataset-summary')
      expect(screen.queryByTestId('dataset-sessions')).toBeNull()
    })
  })

  describe('Tier A capture gap', () => {
    it('names the cells still waiting for the camera, not for a label', async () => {
      // The distinction this block exists for: a Tier A cell at 0/40 is not "0% labeled",
      // it is unshot. Reporting both as backlog is what wastes a capture session.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () => datasetStatus())
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const summary = await screen.findByTestId('dataset-tier-a-summary')
      expect(summary).toHaveTextContent('186')
      expect(summary).toHaveTextContent('of 253 images')
      expect(summary).toHaveTextContent('8 cell(s)')

      const cells = within(screen.getByTestId('dataset-tier-a-cells')).getAllByRole('listitem')
      expect(cells).toHaveLength(8)
      // Most-remaining first. The fixture's roster only carries milo and safeguard, so a
      // Tier A class falls back to its slug here - which is the fallback working, and the
      // next test covers the name path with a slot the roster does know.
      expect(cells[0]).toHaveTextContent('40 to shoot')
      expect(cells[0]).toHaveTextContent('century-tuna')
      expect(cells[0]).toHaveTextContent('(0/40)')
      expect(cells[7]).toHaveTextContent('2 to shoot')
      expect(cells[7]).toHaveTextContent('(19/21)')
    })

    it('shows the class by name when the snapshot roster has one', async () => {
      // The name has to come from the snapshot rather than a second roster in the renderer,
      // so this pins that the lookup actually happens.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () =>
          datasetStatus({
            tier_a_target: 40,
            tier_a_remaining: 40,
            tier_a_cells_under_target: 1,
            tier_a_cells: [
              { slug: 'milo', distance: 'mid', target: 40, have: 0, decided: 0, remaining: 40 }
            ]
          })
        )
      })
      render(<AdminPanel port={8765} deps={deps} />)

      const cells = within(await screen.findByTestId('dataset-tier-a-cells')).getAllByRole(
        'listitem'
      )
      expect(cells).toHaveLength(1)
      expect(cells[0]).toHaveTextContent('Milo Chocolate Drink 22g Sachet')
      expect(cells[0]).toHaveTextContent('@ mid')
    })

    it('says the tier is done instead of showing an empty list', async () => {
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () =>
          datasetStatus({
            tier_a_remaining: 0,
            tier_a_cells_under_target: 0,
            tier_a_cells: [
              { slug: 'milo', distance: 'mid', target: 27, have: 27, decided: 0, remaining: 0 }
            ]
          })
        )
      })
      render(<AdminPanel port={8765} deps={deps} />)

      expect(await screen.findByTestId('dataset-tier-a-summary')).toHaveTextContent(
        'Tier A complete'
      )
      expect(screen.queryByTestId('dataset-tier-a-cells')).toBeNull()
    })

    it('stays out of the way when the snapshot predates the gap block', async () => {
      // An older snapshot is the ordinary case on a machine that has not re-run the tool,
      // and it must not render a "0 of 0 images" capture gap.
      const { deps } = makeDeps('idle', {
        getDatasetStatus: vi.fn(async () =>
          datasetStatus({ tier_a_target: 0, tier_a_remaining: 0, tier_a_cells: [] })
        )
      })
      render(<AdminPanel port={8765} deps={deps} />)

      await screen.findByTestId('dataset-summary')
      expect(screen.queryByTestId('dataset-tier-a')).toBeNull()
    })
  })

  describe('the remote probe geometry', () => {
    // `resize_mode_resolved` answers "which geometry runs?" for weights that run here. These are
    // the remote half: what the app transmits, and what the workflow says the frame was. The
    // workflow's answer is a fact only a round trip can supply, so the panel renders the pair and
    // flags the one state that means the model is not being fed what the app sent.
    function remoteProbe(
      sent: [number, number] | null,
      reported: [number, number] | null
    ): { deps: SettingsDeps; api: ApiClient } {
      return makeDeps('idle', {
        getSettings: vi.fn(async () => baseSettings({ detector_backend: 'local_api' })),
        probeDetector: vi.fn(async () => ({
          backend: 'local_api',
          reachable: true,
          detail: 'Reached http://127.0.0.1:9001',
          latency_ms: 120,
          class_names: [],
          class_warnings: [],
          provider: null,
          sent_size: sent,
          reported_size: reported
        }))
      })
    }

    async function testConnection(deps: SettingsDeps): Promise<void> {
      const user = userEvent.setup()
      render(<AdminPanel port={8765} deps={deps} />)
      await user.click(await screen.findByTestId('test-connection'))
    }

    it('says a pass-through workflow is a pass-through, and no more', async () => {
      // The frame arrived as sent, which is what this can establish. It cannot establish that the
      // model's own preprocessing is right — that happens inside the workflow — and claiming it
      // would be the remote version of the bug the local resize warning exists to prevent.
      await testConnection(remoteProbe([640, 360], [640, 360]).deps)

      const line = await screen.findByTestId('probe-geometry')
      expect(line).toHaveAttribute('data-state', 'same')
      expect(line).toHaveTextContent('passes the frame through')
      expect(line).toHaveTextContent('not visible from here')
      expect(line).not.toHaveTextContent('stretching')
    })

    it('flags a workflow that re-frames the image, and names the aspect it changed', async () => {
      // The remote geometry mismatch: 640x360 goes in and a square canvas is what the model works
      // on, so `remote_infer_size` is not the geometry these weights run at. Amber, because it is
      // something to act on — and the aspect is called out because that is the part that decides
      // whether objects are warped or merely resampled.
      await testConnection(remoteProbe([640, 360], [640, 640]).deps)

      const line = await screen.findByTestId('probe-geometry')
      expect(line).toHaveAttribute('data-state', 'resized')
      expect(line).toHaveClass('mismatch')
      expect(line).toHaveTextContent('640×640')
      expect(line).toHaveTextContent('640×360')
      expect(line).toHaveTextContent('stretching or padding')
    })

    it('distinguishes a resize that keeps the aspect from one that warps it', async () => {
      // A uniform downscale is a detail question, not a geometry one: objects keep their shape, so
      // the operator does not need the "is it stretch or pad?" question raised for it.
      await testConnection(remoteProbe([640, 360], [320, 180]).deps)

      const line = await screen.findByTestId('probe-geometry')
      expect(line).toHaveAttribute('data-state', 'resized')
      expect(line).toHaveTextContent('aspect ratio is kept')
      expect(line).not.toHaveTextContent('stretching or padding')
    })

    it('does not render a silent workflow as agreement', async () => {
      // The state nothing can rule out: no size block came back, so the sidecar falls back to what
      // it sent. That is an assumption, and a panel that showed "640×360" here would be presenting
      // it as an observation.
      await testConnection(remoteProbe([640, 360], null).deps)

      const line = await screen.findByTestId('probe-geometry')
      expect(line).toHaveAttribute('data-state', 'unreported')
      expect(line).toHaveTextContent('reported no image size')
      expect(line).toHaveTextContent('assumes')
      expect(line).not.toHaveTextContent('passes the frame through')
    })

    it('renders nothing for a backend that does not transmit a frame', async () => {
      // `native` reports neither size, and its geometry is already on the settings response. An
      // empty line here would suggest a check happened.
      await testConnection(remoteProbe(null, null).deps)

      await screen.findByTestId('probe-result')
      expect(screen.queryByTestId('probe-geometry')).toBeNull()
    })
  })

  describe('the class list a weight actually predicts', () => {
    // The backend is remote only to reach the Test connection button (the native path probes at
    // start instead); what is under test here is rendering what the sidecar said about the
    // classes, which is the same sentence for either backend.
    const weights = (classNames: string[], classWarnings: string[]): SettingsDeps =>
      makeDeps('idle', {
        getSettings: vi.fn(async () => baseSettings({ detector_backend: 'local_api' })),
        probeDetector: vi.fn(async () => ({
          backend: 'local_api',
          reachable: true,
          detail: 'Fake model loaded.',
          latency_ms: 12,
          class_names: classNames,
          class_warnings: classWarnings,
          provider: null,
          sent_size: null,
          reported_size: null
        }))
      }).deps

    async function testConnection(deps: SettingsDeps): Promise<void> {
      const user = userEvent.setup()
      render(<AdminPanel port={1} deps={deps} />)
      await user.click(await screen.findByTestId('test-connection'))
    }

    it('shows what is wrong with the class list, not only how many classes there are', async () => {
      // The count alone is the trap: a 24-class weight reads as "24 classes", which looks like a
      // bigger, better model rather than one trained per product-and-distance.
      await testConnection(
        weights(
          ['milo', 'milo close', 'milo mid', 'milo far'],
          [
            '3 of 4 class name(s) carry a distance, so this model predicts one class per product-and-distance.'
          ]
        )
      )

      const warnings = await screen.findAllByTestId('probe-class-warning')
      expect(warnings).toHaveLength(1)
      expect(warnings[0]).toHaveTextContent('carry a distance')
      expect(warnings[0]).toHaveTextContent('one class per product-and-distance')
    })

    it("renders the sidecar's sentence rather than composing one", async () => {
      // One write, one description: the sidecar is the only process that has the model's own class
      // list, so the panel must not paraphrase it into a second, possibly contradicting, claim.
      const sentence =
        'this model cannot predict 2 of the 8 roster classes: something the panel cannot know'
      await testConnection(weights(['milo'], [sentence]))

      expect(await screen.findByTestId('probe-class-warning')).toHaveTextContent(sentence)
    })

    it('stays quiet when the class list matches the roster', async () => {
      await testConnection(weights(['milo', 'safeguard'], []))

      await screen.findByTestId('probe-result')
      expect(screen.queryAllByTestId('probe-class-warning')).toHaveLength(0)
    })
  })
})
