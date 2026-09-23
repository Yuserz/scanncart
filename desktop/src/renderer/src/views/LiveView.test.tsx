import { describe, it, expect, vi } from 'vitest'
import { render, screen, act, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LiveView } from './LiveView'
import type { StreamDeps } from '../hooks/useSidecarStream'
import type { StreamClientOptions, FrameMessage } from '../lib/ws'
import type { ApiClient, CameraProfileResponse, InstalledModel, SettingsResponse } from '../lib/api'
import {
  baseSettings,
  installedUnrecorded,
  installedV2,
  installedV2Unmeasured,
  makeDeps,
  ROSTER_NAMES,
  unrecordedResizeMode
} from '../test/fakes'

const PROFILE: CameraProfileResponse = {
  device_key: 'Fake Cam:0:1280x720',
  backend: 'MSMF',
  width: 1280,
  height: 720,
  fps_auto_exposure: 30,
  fps_capped_exposure: 30,
  controls: { brightness: true, exposure: true, gain: false, focus: true, autofocus: true },
  recommended: {},
  measured_at: 1,
  measured: {},
  sweep_version: 1
}

// The weights facts the readout renders come from `/api/settings` + `/api/models`, through the
// same settings deps `CameraTuning` uses. `resize_mode_resolved` is stated rather than derived:
// the sidecar resolves it (`resolve_resize_mode` has one home), so a test that means to exercise
// the readout has to say what the running geometry is.
function readoutDeps(
  settings: Partial<SettingsResponse>,
  installed: InstalledModel[]
): StreamDeps['settingsDeps'] {
  const { deps } = makeDeps({
    getSettings: vi.fn(async () => baseSettings(settings)),
    getModels: vi.fn(async () => ({ stock: [], installed, directory: 'models/' }))
  })
  return { ...deps, pollHealth: false, pollCameras: false }
}

function frameWith(
  dets: FrameMessage['detections'],
  stats: Partial<FrameMessage['stats']> = {}
): FrameMessage {
  return {
    type: 'frame',
    ts: 123,
    seq: 1,
    jpeg: 'AAAA',
    detections: dets,
    stats: { infer_fps: 22.4, capture_fps: 60, latency_ms: 88, ...stats }
  }
}

const makeHarness = (): {
  deps: StreamDeps
  opts: () => StreamClientOptions
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
} => {
  let captured: StreamClientOptions | null = null
  const start = vi.fn(async () => ({ state: 'running' }))
  const stop = vi.fn(async () => ({ state: 'idle' }))
  const deps = {
    apiFactory: () => ({
      health: vi.fn(),
      start,
      stop,
      getLogs: vi.fn(async () => ({ session_id: null, events: [] })),
      getSettings: vi.fn(),
      updateSettings: vi.fn(),
      getSystemInfo: vi.fn(),
      getPresets: vi.fn(),
      applyPreset: vi.fn(),
      probeDetector: vi.fn(),
      getCameras: vi.fn(async () => ({
        cameras: [{ index: 0, name: 'Fake Cam', width: 1280, height: 720 }],
        probed: true,
        detail: ''
      })),
      getCameraQuality: vi.fn(async () => ({
        available: false,
        brightness: 0,
        contrast: 0,
        sharpness: 0,
        capture_fps: 0,
        target_fps: 0,
        verdicts: {},
        detail: ''
      })),
      calibrateCamera: vi.fn(),
      applyCameraProfile: vi.fn(),
      saveSettings: vi.fn(),
      getCameraProfile: vi.fn(async () => ({ profile: null })),
      // Required by ApiClient; never called from this view.
      getDatasetStatus: vi.fn(),
      getModels: vi.fn(),
      recordResizeMode: vi.fn()
    }),
    streamFactory: (opts: StreamClientOptions) => {
      captured = opts
      return { connect: vi.fn(), close: vi.fn() }
    }
  }
  return { deps, opts: () => captured!, start, stop }
}

describe('LiveView', () => {
  it('renders a detection box, item-log row, and stats from a frame', () => {
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    act(() => {
      h.opts().onOpen?.()
      h.opts().onFrame?.(
        frameWith([{ track_id: 1, cls: 'banana', conf: 0.9, box: [0.1, 0.2, 0.3, 0.4] }])
      )
    })
    expect(screen.getByTestId('conn')).toHaveTextContent('connected')
    expect(screen.getAllByTestId('det-box')).toHaveLength(1)
    expect(screen.getByTestId('det-box')).toHaveTextContent('banana 90%')
    expect(screen.getByTestId('item-log')).toHaveTextContent('banana (90%)')
    expect(screen.getByTestId('stat-infer-fps')).toHaveTextContent('22.4')
    expect(screen.getByTestId('stat-capture-fps')).toHaveTextContent('60')
    expect(screen.getByTestId('stat-latency')).toHaveTextContent('88')
    expect(screen.getByTestId('stat-tracked')).toHaveTextContent('1')
  })

  it('shows the suppressed count when the sidecar drops a frame-edge phantom', () => {
    // The suppression removes a row from the item log, so this chip is the only evidence it
    // happened at all: without it a working filter is indistinguishable from a model that never
    // had the defect, and an operator whose real item was dropped has nothing to notice.
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    act(() => {
      h.opts().onOpen?.()
      h.opts().onFrame?.(frameWith([], { suppressed: 2 }))
    })

    expect(screen.getByTestId('stat-suppressed')).toHaveTextContent('2')
    expect(screen.getByTestId('stat-suppressed')).toHaveTextContent('suppressed')
  })

  it('leaves the suppressed tile off when nothing was dropped', () => {
    // Zero is the healthy reading, so a tile showing it would be one nobody looks at by the time
    // it says 1 — and the setting's own state is legible in the tuning card either way.
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    act(() => {
      h.opts().onOpen?.()
      h.opts().onFrame?.(frameWith([], { suppressed: 0 }))
    })

    expect(screen.getByTestId('stats')).toBeInTheDocument()
    expect(screen.queryByTestId('stat-suppressed')).not.toBeInTheDocument()
  })

  it('treats a sidecar that omits the field as nothing suppressed', () => {
    // The field is newer than the wire, so an older sidecar sends no such key; `undefined` has to
    // read as "no phantom" rather than as a tile stuck at a falsy value.
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    const frame = frameWith([])
    delete frame.stats.suppressed
    act(() => {
      h.opts().onOpen?.()
      h.opts().onFrame?.(frame)
    })

    expect(screen.queryByTestId('stat-suppressed')).not.toBeInTheDocument()
  })

  it('does not add a duplicate item-log row for a repeated track_id', () => {
    const h = makeHarness()
    render(<LiveView port={1} deps={h.deps} />)
    const det = [
      {
        track_id: 7,
        cls: 'apple',
        conf: 0.8,
        box: [0, 0, 0.5, 0.5] as [number, number, number, number]
      }
    ]
    act(() => {
      h.opts().onFrame?.(frameWith(det))
      h.opts().onFrame?.(frameWith(det))
    })
    expect(screen.getByTestId('item-log').querySelectorAll('li')).toHaveLength(1)
  })

  it('sets the preview aspect ratio from the loaded frame dimensions', () => {
    const h = makeHarness()
    render(<LiveView port={1} deps={h.deps} />)
    const wrapper = screen.getByTestId('preview-wrapper')
    // no frame loaded yet: no override, CSS falls back to its 16/9 default
    expect(wrapper.style.getPropertyValue('--preview-w')).toBe('')
    expect(wrapper.style.getPropertyValue('--preview-h')).toBe('')

    act(() => {
      h.opts().onFrame?.(frameWith([]))
    })
    const img = screen.getByAltText('live preview') as HTMLImageElement
    // jsdom reports 0x0 for natural dimensions; define the decoded size
    Object.defineProperty(img, 'naturalWidth', { value: 640 })
    Object.defineProperty(img, 'naturalHeight', { value: 480 })
    fireEvent.load(img)

    expect(wrapper.style.getPropertyValue('--preview-w')).toBe('640')
    expect(wrapper.style.getPropertyValue('--preview-h')).toBe('480')
  })

  it('Start button calls the REST start endpoint', async () => {
    const h = makeHarness()
    render(<LiveView port={1} deps={h.deps} />)
    await userEvent.click(screen.getByRole('button', { name: 'Start' }))
    expect(h.start).toHaveBeenCalledTimes(1)
  })

  it('shows a loading state while start() is in flight, then clears it', async () => {
    const h = makeHarness()
    let resolveStart!: (v: { state: string }) => void
    h.start.mockImplementation(() => new Promise<{ state: string }>((res) => (resolveStart = res)))
    render(<LiveView port={1} deps={h.deps} />)

    await userEvent.click(screen.getByRole('button', { name: 'Start' }))

    // In flight: button disabled and spinning, placeholder explains the model load.
    const button = screen.getByRole('button', { name: 'Start' })
    expect(button).toBeDisabled()
    expect(button).toHaveTextContent(/Starting…/)
    expect(screen.getByTestId('preview-placeholder')).toHaveTextContent(/Loading model/)
    expect(screen.getByTestId('preview-placeholder')).toHaveTextContent(/downloads its weights/)

    await act(async () => {
      resolveStart({ state: 'running' })
    })
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled()
    // Running but no frame yet: waiting-for-frames copy with a spinner.
    expect(screen.getByTestId('preview-placeholder')).toHaveTextContent(/Waiting for frames/)
    expect(screen.getByTestId('preview-placeholder').querySelector('.spinner')).not.toBeNull()
  })

  it('shows the sidecar detail when capture dies mid-session', async () => {
    // The pipeline thread reports its own death as a status message carrying a
    // detail. Dropping that detail left a frozen preview and no explanation.
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    act(() => {
      h.opts().onStatus?.({
        type: 'status',
        state: 'error',
        detail: 'Capture stopped: server went away'
      })
    })

    expect(screen.getByTestId('live-error')).toHaveTextContent('Capture stopped: server went away')
  })

  it('shows the reason the last capture died when the handshake carries it', async () => {
    // The other half of the same field. A client that connects *after* a capture died is told
    // `idle` and the stored reason, because `idle` alone is exactly what a capture that was never
    // started looks like — so the detail is the only thing standing between a reload and a frozen
    // preview with nothing to explain it.
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    act(() => {
      h.opts().onStatus?.({
        type: 'status',
        state: 'idle',
        detail: 'Capture stopped: Camera 0 stopped delivering frames for 3s'
      })
    })

    expect(screen.getByTestId('live-error')).toHaveTextContent('stopped delivering frames')
    // The reason is not a claim about the state: nothing is running, and the toolbar says so.
    expect(screen.getByTestId('state')).toHaveTextContent('idle')
  })

  it('surfaces a failed Start instead of swallowing the rejection', async () => {
    // The sidecar answers a missing key with 401, an unreachable server with
    // 503 and a timeout with 504. Those used to become unhandled rejections:
    // the spinner vanished and the button silently reset with nothing shown.
    const h = makeHarness()
    h.start.mockRejectedValueOnce(new Error('503: local_api server unreachable'))
    render(<LiveView port={8765} deps={h.deps} />)

    await userEvent.click(screen.getByLabelText('Start'))

    expect(await screen.findByTestId('live-error')).toHaveTextContent('unreachable')
  })

  it('dismisses the error banner', async () => {
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'error', detail: 'boom' })
    })

    await userEvent.click(screen.getByLabelText('Dismiss error'))

    expect(screen.queryByTestId('live-error')).not.toBeInTheDocument()
  })

  it('shows the camera tuning card in the side rail', async () => {
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    expect(await screen.findByTestId('camera-tuning')).toBeInTheDocument()
  })

  describe('the class list the running model declares', () => {
    const warning =
      '12 of 12 class name(s) carry a distance, so this model predicts one class per ' +
      'product-and-distance instead of one per product: …'

    it('shows what the sidecar said about the running model, in the view where it bites', async () => {
      // The runtime half of the roster guard. It arrives on a status message with no error in it -
      // the capture is running fine, which is exactly why the banner cannot be error-styled or
      // gated on `state === 'error'`.
      const h = makeHarness()
      render(<LiveView port={8765} deps={h.deps} />)

      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_warnings: [warning]
        })
      })

      expect(await screen.findByTestId('live-class-warnings')).toHaveTextContent(
        'one class per product-and-distance'
      )
    })

    it("renders the sidecar's sentence rather than composing one", async () => {
      // One write, one description: the sidecar is the only process that has the loaded model's
      // class names, so a second wording here could only contradict it.
      const sentence =
        'this model cannot predict 2 of the 8 v2 roster classes: something only the sidecar knows'
      const h = makeHarness()
      render(<LiveView port={8765} deps={h.deps} />)

      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_warnings: [sentence]
        })
      })

      expect(await screen.findByTestId('live-class-warnings')).toHaveTextContent(sentence)
    })

    it('clears it when the finding no longer applies', async () => {
      // A later status carrying nothing is the sidecar saying the model is fine now (or that a
      // previous model's finding no longer describes what runs). Merging instead of replacing
      // would leave a stale warning about a model that is gone.
      const h = makeHarness()
      render(<LiveView port={8765} deps={h.deps} />)

      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_warnings: [warning]
        })
      })
      await screen.findByTestId('live-class-warnings')

      act(() => {
        h.opts().onStatus?.({ type: 'status', state: 'running', detail: '', class_warnings: [] })
      })

      expect(screen.queryByTestId('live-class-warnings')).not.toBeInTheDocument()
    })

    it('stays out of the way of a clean capture', async () => {
      const h = makeHarness()
      render(<LiveView port={8765} deps={h.deps} />)

      act(() => {
        h.opts().onStatus?.({ type: 'status', state: 'running', detail: '', class_warnings: [] })
      })

      expect(screen.queryByTestId('live-class-warnings')).not.toBeInTheDocument()
    })

    it('shows the class count and the roster verdict as a chip in the stats strip', async () => {
      // The banner above is silent on a healthy model, which is exactly why the count needs a tile
      // of its own: `8` where an operator expects 8 is the reassuring case, and it has to be
      // visible without a problem to carry it.
      const h = makeHarness()
      render(<LiveView port={8765} deps={h.deps} />)

      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_names: ROSTER_NAMES,
          class_warnings: []
        })
      })

      const chip = await screen.findByTestId('stat-classes')
      expect(chip).toHaveTextContent('8')
      expect(chip).toHaveTextContent('classes · roster ok')
      expect(chip).not.toHaveClass('warn')
      // A tile cannot carry the names, so the title does — including which 8 they are.
      expect(chip).toHaveAttribute('title', expect.stringContaining(ROSTER_NAMES[0]))
    })

    it('counts the findings on the chip when the list is not the roster', async () => {
      // A distance-split project: 24 outputs where the roster has 8. The number is the whole point
      // of this tile — the banner says what is wrong, and only the chip says *how many* classes the
      // head came back with.
      const names = ROSTER_NAMES.flatMap((n) => ['close', 'mid', 'far'].map((d) => `${n} ${d}`))
      const h = makeHarness()
      render(<LiveView port={8765} deps={h.deps} />)

      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_names: names,
          class_warnings: [warning]
        })
      })

      const chip = await screen.findByTestId('stat-classes')
      expect(chip).toHaveTextContent('24')
      expect(chip).toHaveTextContent('1 finding')
      expect(chip).toHaveClass('warn')
      // The sidecar's own sentence in the title, so the chip and the banner cannot describe one
      // model differently.
      expect(chip).toHaveAttribute('title', expect.stringContaining(warning))

      // Counted, not described: a finding worded as "mismatch" would be false for the third one
      // (a model that simply cannot predict some roster names), so the label says how many there
      // are and leaves what they are to the sentences.
      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_names: names,
          class_warnings: [warning, 'second finding']
        })
      })
      expect(screen.getByTestId('stat-classes')).toHaveTextContent('2 findings')
    })

    it('has no chip until the sidecar has read a model\u2019s classes, and loses it when they change', async () => {
      // Before the first inference the sidecar knows no names, and `0 classes` would be a claim about
      // a model that has not spoken yet. The same rule clears the chip when a later status describes
      // a different model: the count and the verdict come from one report and have to change with it.
      const h = makeHarness()
      render(<LiveView port={8765} deps={h.deps} />)

      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_names: ROSTER_NAMES,
          class_warnings: []
        })
      })
      await screen.findByTestId('stat-classes')

      act(() => {
        h.opts().onStatus?.({
          type: 'status',
          state: 'running',
          detail: '',
          class_names: [],
          class_warnings: []
        })
      })

      expect(screen.queryByTestId('stat-classes')).toBeNull()
    })
  })

  it('gives the tuning card the same capture state the toolbar uses', async () => {
    // One owner: LiveView drives capture through useSidecarStream, and the
    // card is told. A second health poller would disagree mid start/stop.
    const h = makeHarness()
    render(<LiveView port={8765} deps={h.deps} />)
    await screen.findByTestId('camera-tuning')
    expect(screen.getByTestId('tuning-idle')).toBeInTheDocument()
  })

  it('warns while running when the settings contradict the weights’ record', async () => {
    // The mismatch does not fail anything - it makes every detection weaker than the model can
    // do, silently. The operator watching the feed is the one who can see that and cannot see
    // why, which is what this banner supplies.
    const h = makeHarness()
    const { deps: settingsDeps } = makeDeps({
      getSettings: vi.fn(async () =>
        baseSettings({
          active_model: installedV2().value,
          resize_mode: 'letterbox'
        })
      ),
      getModels: vi.fn(async () => ({
        stock: [],
        installed: [installedV2()],
        directory: 'models/'
      }))
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)

    // Idle first: the claim is about the frames and the log in front of you, so it must not be
    // made before there are any. (This is also the case the Admin Panel's field already covers,
    // where the setting can actually be changed.)
    //
    // Waiting for the tuning card to finish loading is what makes this assertion mean something:
    // both hooks read the same endpoints through the same client in the same commit, so by the
    // time its loaded state is on screen, this verdict has been computed too. Asserted straight
    // after `render()` it would pass whether or not the gate exists, because the fetch has not
    // resolved yet and "no warning" is then just "nothing known".
    await screen.findByTestId('tuning-idle')
    expect(screen.queryByTestId('live-resize-mismatch')).toBeNull()

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })

    const warning = await screen.findByTestId('live-resize-mismatch')
    // Both halves named, because "wrong geometry" without the two values is not actionable.
    expect(warning).toHaveTextContent('resize_mode: stretch')
    expect(warning).toHaveTextContent('letterbox')
    // And the way out, since the field is not editable from this view.
    expect(warning).toHaveTextContent(/stop capture/i)
  })

  it('says nothing while running on `auto`, because `auto` uses the record', async () => {
    // The control for the check above, and the acceptance case for the whole feature: `auto` is
    // what the docs tell you to leave the field on, and the sidecar honours the recorded
    // requirement, so a banner here would tell the operator to fix a correct configuration.
    const h = makeHarness()
    const { deps: settingsDeps } = makeDeps({
      getSettings: vi.fn(async () =>
        baseSettings({ active_model: installedV2().value, resize_mode: 'auto' })
      ),
      getModels: vi.fn(async () => ({
        stock: [],
        installed: [installedV2()],
        directory: 'models/'
      }))
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)
    await screen.findByTestId('camera-tuning')

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    await waitFor(() => expect(screen.getByTestId('state')).toHaveTextContent('running'))
    expect(screen.queryByTestId('live-resize-mismatch')).toBeNull()
  })

  it('says nothing while running when the setting spells the requirement out', async () => {
    const h = makeHarness()
    const { deps: settingsDeps } = makeDeps({
      getSettings: vi.fn(async () =>
        baseSettings({ active_model: installedV2().value, resize_mode: 'stretch' })
      ),
      getModels: vi.fn(async () => ({
        stock: [],
        installed: [installedV2()],
        directory: 'models/'
      }))
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)
    await screen.findByTestId('camera-tuning')

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    await waitFor(() => expect(screen.getByTestId('state')).toHaveTextContent('running'))
    expect(screen.queryByTestId('live-resize-mismatch')).toBeNull()
  })

  it('stays out of the way when the weights carry no requirement', async () => {
    // A hand-copied `.onnx` baseline has no record, so nothing is required of it and no setting
    // could clear a warning about it. Silent is the only correct state.
    const h = makeHarness()
    const { deps: settingsDeps } = makeDeps({
      getSettings: vi.fn(async () =>
        baseSettings({ active_model: installedUnrecorded().value, resize_mode: 'stretch' })
      ),
      getModels: vi.fn(async () => ({
        stock: [],
        installed: [installedUnrecorded()],
        directory: 'models/'
      }))
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    await waitFor(() => expect(screen.getByTestId('state')).toHaveTextContent('running'))
    expect(screen.queryByTestId('live-resize-mismatch')).toBeNull()
  })

  it('does not warn when the weights could not be read at all', async () => {
    // A warning that cannot be computed is not a warning. An unreachable sidecar is already
    // visible in the toolbar's connection readout; inventing a geometry warning on top of it
    // would send the operator to fix a setting that is not the problem.
    const h = makeHarness()
    const { deps: settingsDeps } = makeDeps({
      getSettings: vi.fn(async () => baseSettings({ active_model: installedV2().value })),
      getModels: vi.fn(async () => {
        throw new Error('sidecar down')
      })
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    await waitFor(() => expect(screen.getByTestId('state')).toHaveTextContent('running'))
    expect(screen.queryByTestId('live-resize-mismatch')).toBeNull()
  })

  it('warns while running that the geometry is assumed, when nothing recorded it', async () => {
    // The other half of the same subject. Nothing here contradicts anything — the settings are on
    // `auto`, which is the documented answer — so the mismatch rule beside this one is silent by
    // construction, and without this banner the assumption reaches the feed unremarked.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedUnrecorded().value,
        resize_mode: 'auto',
        resize_mode_resolved: 'letterbox',
        unrecorded_resize_mode: unrecordedResizeMode({ model: installedUnrecorded().value })
      },
      [installedUnrecorded()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    // Idle first, and for the same reason the mismatch check waits for the tuning card: both hooks
    // read these endpoints through the same client in the same commit, so its loaded state is
    // proof this verdict has been computed. Asserted straight after `render()`, "no banner" would
    // only mean "nothing known yet", and the test would pass with the gate removed.
    await screen.findByTestId('tuning-idle')
    expect(screen.queryByTestId('live-assumed-geometry')).toBeNull()

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })

    const banner = await screen.findByTestId('live-assumed-geometry')
    // The sidecar's diagnosis, verbatim.
    expect(banner).toHaveTextContent('assumes letterbox')
    expect(banner).toHaveTextContent('has no record of the geometry')
    // And its remedy, which has to be usable from a view with no button in it — the same sentence
    // the Admin Panel renders above the one-click record. A banner that only said "fix this in
    // Admin" would leave the operator without the fact that the field itself is not the fix.
    expect(banner).toHaveTextContent(/Admin Panel/)
  })

  it('records the requirement the entry names, and says what landed', async () => {
    // The Admin Panel's button, on the screen where the cost of the assumption is visible. The
    // model and the mode come from the sidecar's entry, never from the button's own label, so no
    // view can offer to write something the sidecar did not name.
    const h = makeHarness()
    let landed = false
    const record = vi.fn(async () => ({
      stock: [],
      installed: [installedUnrecorded({ resize_mode: 'letterbox' })],
      directory: 'models/'
    }))
    const { deps: settingsDeps } = makeDeps({
      getSettings: vi.fn(async () =>
        baseSettings({
          active_model: installedUnrecorded().value,
          resize_mode: 'auto',
          resize_mode_resolved: 'letterbox',
          unrecorded_resize_mode: landed
            ? null
            : unrecordedResizeMode({ model: installedUnrecorded().value })
        })
      ),
      getModels: vi.fn(async () => ({
        stock: [],
        installed: [installedUnrecorded(landed ? { resize_mode: 'letterbox' } : {})],
        directory: 'models/'
      })),
      recordResizeMode: record
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    const banner = await screen.findByTestId('live-assumed-geometry')
    // The requirement chip already says "assumed" — the two render the same fact at different
    // lengths and the banner is the one with the answer attached.
    expect(await screen.findByTestId('stat-requirement')).toHaveTextContent('requirement (assumed)')
    expect(banner).toHaveTextContent('Record it now: letterbox')

    landed = true
    await userEvent.click(screen.getByTestId('live-record-resize-mode'))

    await waitFor(() => expect(record).toHaveBeenCalledWith('models/hand-copied.pt', 'letterbox'))
    // The banner goes because the *sidecar* stopped reporting the assumption — the re-read is what
    // clears it, not the click — and the slot it vacated says what happened rather than going
    // blank, which on a live feed is indistinguishable from a button that did nothing.
    await waitFor(() => expect(screen.queryByTestId('live-assumed-geometry')).toBeNull())
    const confirm = await screen.findByTestId('live-recorded')
    expect(confirm).toHaveTextContent('Recorded')
    // The mode is read back off the refreshed listing rather than echoed from the click.
    expect(confirm).toHaveTextContent('resize_mode: letterbox')
    // And the chip now carries the record, which is the same outcome seen without the banner.
    expect(screen.getByTestId('stat-requirement')).toHaveTextContent('requirement (recorded)')
  })

  it('keeps the warning when the record does not land', async () => {
    // A write that failed leaves the situation exactly as it was, and the warning is the honest
    // outcome: it still has the button that can be clicked again. Replacing it with an error
    // banner would take away the sentence that says what the button is for.
    const h = makeHarness()
    const { deps: settingsDeps } = makeDeps({
      getSettings: vi.fn(async () =>
        baseSettings({
          active_model: installedUnrecorded().value,
          resize_mode: 'auto',
          resize_mode_resolved: 'letterbox',
          unrecorded_resize_mode: unrecordedResizeMode({ model: installedUnrecorded().value })
        })
      ),
      getModels: vi.fn(async () => ({
        stock: [],
        installed: [installedUnrecorded()],
        directory: 'models/'
      })),
      recordResizeMode: vi.fn(async () => {
        throw new Error('sidecar down')
      })
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    await screen.findByTestId('live-assumed-geometry')

    await userEvent.click(screen.getByTestId('live-record-resize-mode'))

    // The button coming back is the observable end of the in-flight state — and the warning is
    // still above it.
    await waitFor(() => expect(screen.getByTestId('live-record-resize-mode')).toBeEnabled())
    expect(screen.getByTestId('live-assumed-geometry')).toBeInTheDocument()
    expect(screen.queryByTestId('live-recorded')).toBeNull()
  })

  it('shows what the record says these weights were trained at', async () => {
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedV2().value,
        resize_mode: 'auto',
        resize_mode_resolved: 'stretch'
      },
      [installedV2()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    const tile = await screen.findByTestId('stat-requirement')
    expect(tile).toHaveTextContent('stretch')
    expect(tile).toHaveTextContent('requirement (recorded)')
    // Neutral, not dim: a record is evidence, and the dim style means "nothing recorded" — the
    // distinction the recall tile beside it draws for a measurement.
    expect(tile.className).not.toContain('unmeasured')
  })

  it('marks the requirement as assumed when nothing recorded it, while idle', async () => {
    // The banner's fact without the banner. It has to be visible here: the banner is gated on a
    // running capture, this is a property of the weights, and idle is exactly when someone is
    // deciding whether to start.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedUnrecorded().value,
        resize_mode: 'auto',
        resize_mode_resolved: 'letterbox',
        unrecorded_resize_mode: unrecordedResizeMode({ model: installedUnrecorded().value })
      },
      [installedUnrecorded()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    // Idle, so the banner is absent by design — which is what makes this the only place it shows.
    await screen.findByTestId('tuning-idle')
    expect(screen.queryByTestId('live-assumed-geometry')).toBeNull()

    const tile = await screen.findByTestId('stat-requirement')
    expect(tile).toHaveTextContent('letterbox')
    expect(tile).toHaveTextContent('requirement (assumed)')
    expect(tile.className).toContain('unmeasured')
    // The explanation is the sidecar's own sentence, so the chip and the banner cannot describe
    // different risks.
    expect(tile.getAttribute('title')).toContain('assumes letterbox')
  })

  it('says nothing while running when a record names the geometry `auto` resolves to', async () => {
    // The control, and the one that keeps this from being a re-render of the geometry tile: the
    // visible configuration is *identical* to the test above — `auto`, resolving to letterbox —
    // and the only difference is that a record says why. There is nothing assumed, so there is
    // nothing to warn about, which is what makes the entry the thing that drives the banner
    // rather than the resolved mode.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedV2().value,
        resize_mode: 'auto',
        resize_mode_resolved: 'letterbox',
        unrecorded_resize_mode: null
      },
      [installedV2({ resize_mode: 'letterbox', auto_resolves_to: 'letterbox' })]
    )
    render(<LiveView port={8765} deps={h.deps} />)
    await screen.findByTestId('camera-tuning')

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    await waitFor(() => expect(screen.getByTestId('state')).toHaveTextContent('running'))
    await screen.findByTestId('stat-geometry')
    expect(screen.queryByTestId('live-assumed-geometry')).toBeNull()
  })

  it('reads a payload without the entry as nothing assumed', async () => {
    // An older sidecar, or a fixture written before the field existed. `?? null` is what keeps
    // that from rendering `undefined.warning` and taking the view down for a missing *warning* —
    // the failure mode would be worse than the silence.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedUnrecorded().value,
        resize_mode: 'auto',
        resize_mode_resolved: 'letterbox',
        unrecorded_resize_mode: undefined
      },
      [installedUnrecorded()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    act(() => {
      h.opts().onStatus?.({ type: 'status', state: 'running', detail: '' } as never)
    })
    await waitFor(() => expect(screen.getByTestId('state')).toHaveTextContent('running'))
    await screen.findByTestId('stat-geometry')
    expect(screen.queryByTestId('live-assumed-geometry')).toBeNull()
  })

  it('shows the running model’s geometry and its measured recall in the stats strip', async () => {
    // The two numbers behind the detections on screen: the geometry these weights run at and what
    // `--val` measured them at. Neither is anywhere else in the Live view, and the person judging
    // the feed is the one who needs them.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedV2().value,
        resize_mode: 'auto',
        resize_mode_resolved: 'stretch'
      },
      [installedV2()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    // Deliberately idle - no status message, so capture is not running. Unlike the mismatch
    // banner, the readout is a fact about the weights rather than about the frames, and both
    // fields behind it are restart-required, so it must be on screen before Start too.
    const geometry = await screen.findByTestId('stat-geometry')
    // `auto` is the setting the docs tell you to leave alone, so the tile names the geometry it
    // resolves to *and* where it came from: "stretch" on its own would hide that nobody set it.
    expect(geometry).toHaveTextContent('stretch')
    expect(geometry).toHaveTextContent('geometry (auto)')
    expect(geometry).not.toHaveClass('warn')

    const recall = screen.getByTestId('stat-recall')
    // The value is the mean, and the label names the split it was measured on: `valid` is the one
    // training selected on, so an unlabelled "recall" would be an overstatement waiting to happen.
    expect(recall).toHaveTextContent('82%')
    expect(recall).toHaveTextContent('test recall')
    // The acceptance criterion is per class, so amber means "a class is below the floor" - which
    // the label states in words, because a bare amber number would read as the mean having failed.
    expect(recall).toHaveTextContent('1 below floor')
    expect(recall).toHaveClass('warn')
  })

  it('names the geometry that will actually run when the setting contradicts the record', async () => {
    // The strip must report what the detector does, not what the operator should have chosen:
    // letterbox is wrong for these weights, and it is also the truth about the running config.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedV2().value,
        resize_mode: 'letterbox',
        resize_mode_resolved: 'letterbox'
      },
      [installedV2()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    const geometry = await screen.findByTestId('stat-geometry')
    expect(geometry).toHaveTextContent('letterbox')
    expect(geometry).toHaveClass('warn')
    // A tile cannot carry the explanation, so the title does - naming the requirement to satisfy.
    expect(geometry).toHaveAttribute('title', expect.stringContaining('resize_mode: stretch'))
  })

  it('says a weight was never measured instead of showing a zero', async () => {
    // Installed by the tool with no `--val` yet. 0% would be a score nobody earned, and it would
    // send the operator after a model problem when the gap is a missing measurement.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      { active_model: installedV2Unmeasured().value, resize_mode_resolved: 'stretch' },
      [installedV2Unmeasured()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    const recall = await screen.findByTestId('stat-recall')
    expect(recall).toHaveTextContent('—')
    expect(recall).toHaveTextContent('recall not measured')
    expect(recall).toHaveClass('unmeasured')
    expect(recall).not.toHaveClass('warn')
    // The geometry is still known - the two facts come from different places and only one of them
    // is missing, which is why the readout is not all-or-nothing.
    expect(screen.getByTestId('stat-geometry')).toHaveTextContent('stretch')
  })

  it('says nothing about weights a remote backend does not run', async () => {
    // `local_api`/`cloud_api` send the frame to a workflow that holds its own model and resizes
    // server-side, so neither the recorded geometry nor the recorded score describes what is
    // running. The sidecar answers `null` for that, and the strip prints nothing rather than a
    // geometry for a resize that never happens.
    const h = makeHarness()
    h.deps.settingsDeps = readoutDeps(
      {
        active_model: installedV2().value,
        detector_backend: 'local_api',
        resize_mode_resolved: null
      },
      [installedV2()]
    )
    render(<LiveView port={8765} deps={h.deps} />)

    // The tuning card's loaded state is the wait: both hooks read the same endpoints through the
    // same client in the same commit, so by then the readout has been computed. Asserted straight
    // after render() this would pass because the fetch had not resolved yet.
    await screen.findByTestId('tuning-idle')
    expect(screen.queryByTestId('stat-geometry')).toBeNull()
    expect(screen.queryByTestId('stat-recall')).toBeNull()
  })

  it('still names the geometry when the weights list could not be read', async () => {
    // The geometry travels in the settings response, so losing `/api/models` costs the score and
    // the mismatch verdict but not the whole readout. That asymmetry is the reason the two are
    // fetched separately rather than one call that can only fail as a unit.
    const h = makeHarness()
    const { deps } = makeDeps({
      getSettings: vi.fn(async () =>
        baseSettings({ active_model: installedV2().value, resize_mode_resolved: 'stretch' })
      ),
      getModels: vi.fn(async () => {
        throw new Error('sidecar down')
      })
    })
    h.deps.settingsDeps = { ...deps, pollHealth: false, pollCameras: false }
    render(<LiveView port={8765} deps={h.deps} />)

    expect(await screen.findByTestId('stat-geometry')).toHaveTextContent('stretch')
    expect(screen.getByTestId('stat-recall')).toHaveTextContent('recall not measured')
  })

  it('will not offer Start while calibration is holding the camera', async () => {
    // The sidecar answers a start during a sweep with a 409, so the button
    // was offering an action that could only fail. It stayed enabled through
    // the whole stop -> calibrate -> start sequence.
    const h = makeHarness()
    let releaseSweep: () => void = () => {}
    // The card builds its own client from settingsDeps — LiveView's own
    // apiFactory drives the stream hook and never reaches it.
    const { deps: settingsDeps } = makeDeps({
      calibrateCamera: vi.fn(
        () =>
          new Promise((resolve) => {
            releaseSweep = () => resolve(PROFILE)
          })
      ) as ApiClient['calibrateCamera']
    })
    h.deps.settingsDeps = { ...settingsDeps, pollHealth: false, pollCameras: false }
    const { container } = render(<LiveView port={8765} deps={h.deps} />)
    await screen.findByTestId('camera-tuning')
    // By position, not by label: the sequence restarts capture, so the same
    // button reads "Stop" by the time it is handed back.
    const toggle = (): HTMLButtonElement =>
      container.querySelector('.live-toolbar button') as HTMLButtonElement
    expect(toggle()).toBeEnabled()

    await userEvent.click(screen.getByTestId('tuning-calibrate'))
    // The staged-scene gate sits in front of the sweep now — confirm through
    // it before the stop/calibrate/start sequence (and the busy toggle) begins.
    await userEvent.click(await screen.findByTestId('tuning-scene-ready'))
    await waitFor(() => expect(toggle()).toBeDisabled())

    await act(async () => {
      releaseSweep()
    })
    await waitFor(() => expect(toggle()).toBeEnabled())
  })
})
