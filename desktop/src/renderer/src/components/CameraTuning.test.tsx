import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { ApiClient, CameraProfileResponse } from '../lib/api'
import { baseSettings, makeDeps } from '../test/fakes'
import { CameraTuning } from './CameraTuning'

// The brief's snippets reference a bare `BASE_SETTINGS` — this is that,
// built from the same fake the rest of the suite (and makeDeps' default
// getSettings) uses, so overrides here stay consistent with renderCard.
const BASE_SETTINGS = baseSettings()

// Reuse the shape the hook tests already establish.
const PROFILE: CameraProfileResponse = {
  device_key: 'StreamCam:0:1280x720',
  backend: 'MSMF',
  width: 1280,
  height: 720,
  fps_auto_exposure: 29.9,
  fps_capped_exposure: 30.8,
  controls: { brightness: true, exposure: true, gain: false, focus: false },
  recommended: { camera_exposure: -6 },
  measured_at: 1
}

function renderCard(overrides: Partial<ApiClient> = {}, running = true): ReturnType<typeof render> {
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
      running={running}
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

describe('applying and committing', () => {
  it('coalesces a drag into one request', async () => {
    const calls: unknown[] = []
    renderCard({
      updateSettings: async (patch: unknown) => {
        calls.push(patch)
        return { ...BASE_SETTINGS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))

    const slider = screen.getByLabelText('Brightness')
    fireEvent.change(slider, { target: { value: '100' } })
    fireEvent.change(slider, { target: { value: '150' } })
    fireEvent.change(slider, { target: { value: '180' } })

    // debounceMs is 0 in tests, but the trailing edge still collapses the
    // burst to one write — a real drag emits dozens.
    await waitFor(() => expect(calls.length).toBe(1))
    expect(calls[0]).toEqual({ camera_brightness: 180 })
  })

  it('does not persist while tuning', async () => {
    const persists: (boolean | undefined)[] = []
    renderCard({
      updateSettings: async (patch: unknown, persist?: boolean) => {
        persists.push(persist)
        return { ...BASE_SETTINGS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })

    await waitFor(() => expect(persists).toEqual([false]))
  })

  it('reports unsaved changes and clears them on save', async () => {
    let saved = false
    renderCard({
      updateSettings: async (patch: unknown) => ({ ...BASE_SETTINGS, ...(patch as object) }),
      saveSettings: async () => {
        saved = true
        return { ...BASE_SETTINGS, camera_brightness: 180 }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })

    await screen.findByTestId('tuning-dirty')
    await userEvent.click(screen.getByTestId('tuning-save'))

    expect(saved).toBe(true)
    await waitFor(() => expect(screen.queryByTestId('tuning-dirty')).toBeNull())
  })

  it('reverts to the last saved values', async () => {
    // A non-null saved baseline, so this exercises the "restore the previous
    // value" path — distinct from the null/"leave the camera alone" path
    // covered below.
    const SAVED_BRIGHTNESS = 100
    const calls: unknown[] = []
    renderCard({
      getSettings: async () => ({ ...BASE_SETTINGS, camera_brightness: SAVED_BRIGHTNESS }),
      updateSettings: async (patch: unknown) => {
        calls.push(patch)
        return { ...BASE_SETTINGS, camera_brightness: SAVED_BRIGHTNESS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })
    await screen.findByTestId('tuning-dirty')

    await userEvent.click(screen.getByTestId('tuning-revert'))

    await waitFor(() => expect(calls.at(-1)).toEqual({ camera_brightness: SAVED_BRIGHTNESS }))
  })

  it('reverts a control back to "leave the camera alone"', async () => {
    // All four controls default to null, so this is the fresh-install path:
    // tune brightness for the first time, then Revert.
    const calls: unknown[] = []
    renderCard({
      getSettings: async () => ({ ...BASE_SETTINGS, camera_brightness: null }),
      updateSettings: async (patch: unknown) => {
        calls.push(patch)
        return { ...BASE_SETTINGS, ...(patch as object) }
      }
    })
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })
    await screen.findByTestId('tuning-dirty')

    await userEvent.click(screen.getByTestId('tuning-revert'))

    await waitFor(() => expect(calls.at(-1)).toEqual({ reset_fields: ['camera_brightness'] }))
  })

  it('offers nothing to save when nothing changed', async () => {
    renderCard()
    await screen.findByLabelText('Brightness')
    expect(screen.queryByTestId('tuning-dirty')).toBeNull()
  })

  // Coverage debt: an earlier task deleted an Admin test asserting that
  // editing a hot-reloadable field while capture is RUNNING succeeds. After
  // the settings moved, nothing in Admin was hot-reloadable any more, so
  // there was nowhere to retarget it — that behaviour now lives here. The
  // tests above assert the PATCH shape; this one proves the live edit
  // actually round-trips end to end against a running pipeline.
  it('applies a live edit against a running pipeline and reflects the new value', async () => {
    renderCard(
      {
        updateSettings: async (patch: unknown) => ({ ...BASE_SETTINGS, ...(patch as object) })
      },
      true // running
    )
    await screen.findByLabelText('Brightness')
    await userEvent.click(screen.getByRole('button', { name: /Camera tuning/ }))

    const slider = screen.getByLabelText('Brightness') as HTMLInputElement
    expect(slider).toBeEnabled()
    fireEvent.change(slider, { target: { value: '180' } })

    await waitFor(() => expect(slider.value).toBe('180'))
    // The live PATCH landed, and the running camera should not show the idle hint.
    expect(screen.queryByTestId('tuning-idle')).toBeNull()
  })
})

describe('calibration from the live tab', () => {
  function renderWithLifecycle(overrides: Partial<ApiClient> = {}, order: string[] = []): string[] {
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => {
        order.push('calibrate')
        return PROFILE
      },
      ...overrides
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {
          order.push('start')
        }}
        stop={async () => {
          order.push('stop')
        }}
        cameraName="Logitech StreamCam"
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    return order
  }

  it('stops, calibrates, then restarts', async () => {
    // The camera is exclusive during a sweep, so the sidecar 409s while
    // capture runs. The operator should not have to know that.
    const order = renderWithLifecycle()
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    await waitFor(() => expect(order).toEqual(['stop', 'calibrate', 'start']))
  })

  it('restarts capture even when the sweep fails', async () => {
    // Otherwise a failed calibration leaves the operator staring at a dark
    // feed with no idea why.
    const order: string[] = []
    renderWithLifecycle(
      {
        calibrateCamera: async () => {
          order.push('calibrate')
          throw new Error('camera busy')
        }
      },
      order
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    await waitFor(() => expect(order).toEqual(['stop', 'calibrate', 'start']))
  })

  it('shows the measured evidence before applying anything', async () => {
    renderWithLifecycle()
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    // Review-first: the operator sees the two framerates and the proposed
    // patch, and chooses.
    expect(await screen.findByTestId('tuning-profile')).toHaveTextContent('29.9')
    expect(screen.getByTestId('tuning-apply-profile')).toBeInTheDocument()
  })
})

describe('failures the card must not swallow', () => {
  it('shows the error when a calibration sweep fails', async () => {
    // The card holds its own useSidecarSettings instance, so its `error` is
    // local to it — LiveView's banner reads useSidecarStream and would never
    // show this. Without rendering it here, a failed sweep is silent: the
    // feed stops and restarts and nothing says why.
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => {
        throw new Error('camera busy')
      }
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    expect(await screen.findByTestId('tuning-error')).toHaveTextContent('camera busy')
  })

  it('shows the error even while the card is collapsed', async () => {
    // The card starts shut. An operator who never expands it still needs to
    // learn that the sweep they launched failed.
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      saveSettings: async () => {
        throw new Error('disk full')
      }
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    // Dirty the card without expanding it: the sliders live in the
    // collapsible body, so open it, change one, then shut it again.
    await screen.findByLabelText('Brightness')
    const toggle = screen.getByRole('button', { name: /Camera tuning/ })
    await userEvent.click(toggle)
    fireEvent.change(screen.getByLabelText('Brightness'), { target: { value: '180' } })
    await screen.findByTestId('tuning-dirty')
    await userEvent.click(toggle)
    await userEvent.click(screen.getByTestId('tuning-save'))

    const banner = await screen.findByTestId('tuning-error')
    expect(banner).toHaveTextContent('disk full')
    expect(banner).toHaveAttribute('role', 'alert')
  })

  it('ignores a second click during the stop and start either side of a sweep', async () => {
    // `calibrating` covers only the sweep. The stop before it and the start
    // after it are awaited too, and a click landing in either window used to
    // launch a whole second overlapping sequence.
    const order: string[] = []
    let releaseStop: () => void = () => {}
    const stopped = new Promise<void>((r) => {
      releaseStop = r
    })
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => {
        order.push('calibrate')
        return PROFILE
      }
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {
          order.push('start')
        }}
        stop={async () => {
          order.push('stop')
          await stopped
        }}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    const button = await screen.findByTestId('tuning-calibrate')
    await userEvent.click(button)
    // Still inside the awaited stop() — the sweep has not begun.
    expect(order).toEqual(['stop'])
    expect(button).toBeDisabled()

    fireEvent.click(button)
    releaseStop()

    await waitFor(() => expect(order).toEqual(['stop', 'calibrate', 'start']))
  })
})

describe('a camera that responds to nothing', () => {
  it('says so instead of offering an empty patch to apply', async () => {
    const { deps } = makeDeps({
      getCameraProfile: async () => ({ profile: PROFILE }),
      calibrateCamera: async () => ({ ...PROFILE, recommended: {} })
    })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    await userEvent.click(await screen.findByTestId('tuning-calibrate'))

    expect(await screen.findByTestId('tuning-no-recommendation')).toBeInTheDocument()
    expect(screen.queryByTestId('tuning-apply-profile')).toBeNull()
  })
})

describe('naming the camera being tuned', () => {
  it('reads the device name off the stored profile rather than showing an index', async () => {
    // Calibrate needs a visible target. Enumerating devices is not available
    // here (it opens every camera, and the sidecar refuses it during
    // capture), but the profile the card already fetches carries the name.
    const { deps } = makeDeps({ getCameraProfile: async () => ({ profile: PROFILE }) })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    expect(await screen.findByText('StreamCam')).toBeInTheDocument()
  })

  it('falls back to the index for a camera that has never been calibrated', async () => {
    const { deps } = makeDeps({ getCameraProfile: async () => ({ profile: null }) })
    render(
      <CameraTuning
        port={9000}
        running={true}
        start={async () => {}}
        stop={async () => {}}
        debounceMs={0}
        deps={{ ...deps, pollHealth: false, pollCameras: false }}
      />
    )
    expect(await screen.findByText('Camera 0')).toBeInTheDocument()
  })
})

describe('field hints', () => {
  it('exposes each hint to assistive tech instead of hiding it behind hover', async () => {
    // The hint used to be plain visible text. Moving it into a hover tooltip
    // put it out of reach of the keyboard and, with an aria-label over the
    // top, out of reach of a screen reader too.
    renderCard()
    const brightness = await screen.findByLabelText('Brightness')
    const describedBy = brightness
      .closest('.tuning-field')
      ?.querySelector('[aria-describedby]')
      ?.getAttribute('aria-describedby')

    expect(describedBy).toBe('hint-camera_brightness')
    expect(document.getElementById(describedBy!)).toHaveTextContent('amplifies noise')
  })
})
