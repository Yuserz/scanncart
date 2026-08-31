import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ApiClient, CameraProfileResponse } from '../lib/api'
import { baseSettings, makeDeps } from '../test/fakes'
import { CameraTuning } from './CameraTuning'

// Reuse the shape the hook tests already establish.
const PROFILE: CameraProfileResponse = {
  device_key: 'StreamCam:0:1280x720',
  backend: 'MSMF',
  width: 1280,
  height: 720,
  fps_auto_exposure: 29.9,
  fps_capped_exposure: 30.8,
  controls: { brightness: true, exposure: true, gain: false, focus: false },
  recommended: {},
  measured_at: 1
}

function renderCard(overrides: Partial<ApiClient> = {}): ReturnType<typeof render> {
  const { deps } = makeDeps({
    getCameraProfile: async () => ({ profile: PROFILE }),
    getCameraQuality: async () => ({
      available: true,
      brightness: 128,
      contrast: 40,
      sharpness: 90,
      capture_fps: 29.4,
      target_fps: 30,
      verdicts: { brightness: 'ok', sharpness: 'ok', capture_fps: 'ok' },
      detail: ''
    }),
    ...overrides
  })
  return render(
    <CameraTuning
      port={9000}
      running={true}
      start={async () => {}}
      stop={async () => {}}
      cameraName="Logitech StreamCam"
      debounceMs={0}
      deps={{ ...deps, pollHealth: false, pollCameras: false }}
    />
  )
}

describe('CameraTuning', () => {
  it('names the camera it is tuning', async () => {
    renderCard()
    // The camera is chosen in Admin, so the card has to say which one this is.
    expect(await screen.findByText(/Logitech StreamCam/)).toBeInTheDocument()
  })

  it('disables a control the device does not honour', async () => {
    renderCard()
    // Calibration measured focus support as false; a slider that does
    // nothing is worse than one that is visibly unavailable.
    await waitFor(() => expect(screen.getByLabelText('Focus')).toBeDisabled())
  })

  it('enables a control the device does honour', async () => {
    renderCard()
    await waitFor(() => expect(screen.getByLabelText('Brightness')).toBeEnabled())
  })

  it('disables focus while autofocus is on regardless of support', async () => {
    renderCard({
      getCameraProfile: async () => ({
        profile: { ...PROFILE, controls: { ...PROFILE.controls, focus: true } }
      }),
      getSettings: async () => baseSettings({ camera_autofocus: true })
    })
    await waitFor(() => expect(screen.getByLabelText('Focus')).toBeDisabled())
  })

  it('shows the live quality readout', async () => {
    renderCard()
    expect(await screen.findByTestId('tuning-quality')).toHaveTextContent('29.4')
  })

  it('flags a failing metric by more than colour', async () => {
    renderCard({
      getCameraQuality: async () => ({
        available: true,
        brightness: 23,
        contrast: 10,
        sharpness: 12,
        capture_fps: 4.1,
        target_fps: 30,
        verdicts: { brightness: 'low', sharpness: 'low', capture_fps: 'low' },
        detail: ''
      })
    })
    await waitFor(() => expect(screen.getAllByTestId('quality-low').length).toBeGreaterThan(0))
  })

  it('leaves every control enabled for an uncalibrated camera', async () => {
    // No profile means no evidence either way; unsupported controls simply
    // do nothing, which the quality readout makes visible.
    renderCard({ getCameraProfile: async () => ({ profile: null }) })
    await waitFor(() => expect(screen.getByLabelText('Focus')).toBeEnabled())
  })
})
