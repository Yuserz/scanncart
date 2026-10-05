import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { PosAdminSection } from './PosAdminSection'
import { zoneSummary } from '../lib/zones'
import type { PosState } from '../../../main/posSession'
import { DEFAULT_POS_CONFIG } from '../../../main/posConfig'

const CONFIG = {
  ...DEFAULT_POS_CONFIG,
  posBaseUrl: 'http://192.168.1.20:3000',
  posSecret: 'secret',
  stationId: 'counter-1'
}

let saved: unknown = null
let result: unknown = null

function stubBridge(): void {
  vi.stubGlobal('api', {
    getPosConfig: async () => CONFIG,
    savePosConfig: async (patch: unknown) => {
      saved = patch
      return { ...CONFIG, ...(patch as object) }
    },
    testPosConnection: async () => result
  })
}

afterEach(() => {
  vi.unstubAllGlobals()
  saved = null
  result = null
})

describe('PosAdminSection', () => {
  const posState = (
    state: PosState | null
  ): { read: () => Promise<PosState | null>; subscribe: () => () => void } => ({
    read: async () => state,
    subscribe: () => () => {}
  })

  it('loads the saved configuration into the form', async () => {
    stubBridge()
    render(<PosAdminSection />)

    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))
    expect(screen.getByTestId('pos-station-id')).toHaveValue('counter-1')
    expect(screen.getByTestId('pos-cartMode-counter').querySelector('input')).toBeChecked()
  })

  it('enables Save only once something changed, and Discard puts the form back', async () => {
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))

    expect(screen.getByText('Save self-checkout settings')).toBeDisabled()
    fireEvent.change(screen.getByTestId('pos-station-id'), { target: { value: 'counter-2' } })
    expect(screen.getByText('Save self-checkout settings')).toBeEnabled()
    expect(screen.getByText('Unsaved self-checkout changes')).toBeInTheDocument()

    fireEvent.click(screen.getByText('Discard'))
    expect(screen.getByTestId('pos-station-id')).toHaveValue('counter-1')
    expect(screen.getByText('Save self-checkout settings')).toBeDisabled()
  })

  it('saves the edited config, including numbers and the cart mode', async () => {
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))

    fireEvent.change(screen.getByTestId('pos-station-id'), { target: { value: 'counter-2' } })
    fireEvent.change(screen.getByTestId('pos-commitDwellS'), { target: { value: '4' } })
    fireEvent.click(screen.getByTestId('pos-cartMode-basket').querySelector('input')!)
    fireEvent.click(screen.getByText('Save self-checkout settings'))

    await waitFor(() => expect(saved).not.toBeNull())
    expect(saved).toMatchObject({ stationId: 'counter-2', commitDwellS: 4, cartMode: 'basket' })
    await waitFor(() => expect(screen.getByTestId('pos-save-message')).toHaveTextContent('Saved.'))
  })

  it('never writes the zones, which belong to the Basket test tab', async () => {
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))

    fireEvent.change(screen.getByTestId('pos-station-id'), { target: { value: 'counter-2' } })
    fireEvent.click(screen.getByText('Save self-checkout settings'))

    await waitFor(() => expect(saved).not.toBeNull())
    for (const key of [
      'zoneMode',
      'cartEdge',
      'insideFraction',
      'openingFraction',
      'drawnInside',
      'drawnOpening'
    ]) {
      expect(saved as object).not.toHaveProperty(key)
    }
    expect(screen.getByTestId('pos-zones')).toHaveTextContent('Basket test tab')
  })

  it('summarises either zone layout in one line', () => {
    expect(zoneSummary(CONFIG)).toBe(
      'A · Bands — basket at the bottom of the picture, inside 35%, opening 20%'
    )
    expect(
      zoneSummary({
        ...CONFIG,
        zoneMode: 'drawn',
        drawnInside: [
          { x: 0.1, y: 0.1 },
          { x: 0.2, y: 0.1 },
          { x: 0.2, y: 0.2 }
        ],
        drawnOpening: []
      })
    ).toBe('B · Drawn outlines — inside 3 points, opening 0 points')
  })

  it("shows a save refusal in the main process's own words", async () => {
    vi.stubGlobal('api', {
      getPosConfig: async () => CONFIG,
      savePosConfig: async () => {
        throw new Error(
          "Error invoking remote method 'pos:save-config': Error: commitDwellS (1s) must exceed the sidecar's track_expiry_s (1.5s)."
        )
      }
    })
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))
    fireEvent.change(screen.getByTestId('pos-commitDwellS'), { target: { value: '1' } })
    fireEvent.click(screen.getByText('Save self-checkout settings'))

    await waitFor(() =>
      expect(screen.getByTestId('pos-save-message')).toHaveTextContent(
        "commitDwellS (1s) must exceed the sidecar's track_expiry_s (1.5s)."
      )
    )
    expect(screen.getByTestId('pos-save-message')).not.toHaveTextContent('Error invoking')
  })

  it('reports the connection test, including the TLS warning', async () => {
    result = {
      ok: true,
      status: 200,
      message: 'Connected — no session open right now.',
      cartCode: null,
      sessionRef: null,
      warning:
        'posBaseUrl is plain http:// and not localhost — the POS secret would travel in clear text. Use https://.',
      trackExpiryS: 1.5
    }
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))

    fireEvent.click(screen.getByText('Test connection'))

    await waitFor(() =>
      expect(screen.getByTestId('pos-test-result')).toHaveTextContent('Connected')
    )
    expect(screen.getByTestId('pos-test-result')).toHaveTextContent('HTTP 200')
    expect(screen.getByTestId('pos-tls-warning')).toHaveTextContent('https://')
  })

  it("warns inline when the dwell is not longer than the sidecar's track expiry", async () => {
    result = {
      ok: true,
      status: 200,
      message: 'Connected — no session open right now.',
      cartCode: null,
      sessionRef: null,
      warning: null,
      trackExpiryS: 1.5
    }
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))
    fireEvent.click(screen.getByText('Test connection'))
    await waitFor(() => expect(screen.getByTestId('pos-test-result')).toBeInTheDocument())

    expect(screen.queryByTestId('pos-dwell-warning')).not.toBeInTheDocument()
    fireEvent.change(screen.getByTestId('pos-commitDwellS'), { target: { value: '1.5' } })
    expect(screen.getByTestId('pos-dwell-warning')).toHaveTextContent('1.5 s')
  })

  it('reveals the secret only on request', async () => {
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-secret')).toHaveValue('secret'))

    expect(screen.getByTestId('pos-secret')).toHaveAttribute('type', 'password')
    fireEvent.click(screen.getByText('Show'))
    expect(screen.getByTestId('pos-secret')).toHaveAttribute('type', 'text')
    fireEvent.click(screen.getByText('Hide'))
    expect(screen.getByTestId('pos-secret')).toHaveAttribute('type', 'password')
  })

  it('says Off when the connection is not configured', async () => {
    vi.stubGlobal('api', { getPosConfig: async () => ({ ...CONFIG, posSecret: '' }) })
    render(<PosAdminSection posStateDeps={posState(null)} />)
    await waitFor(() => expect(screen.getByTestId('pos-status')).toHaveTextContent('Off'))
    expect(screen.getByTestId('pos-status')).toHaveAttribute('data-tone', 'off')
  })

  it('reports the live state: ready, a customer session, or an error', async () => {
    const base: PosState = {
      phase: 'unbound',
      cartCode: null,
      syncedItemCount: 0,
      lastSyncAgeS: null,
      error: null,
      retryAtMs: null,
      lastContactMs: 1_000,
      cartMode: 'counter',
      basket: null
    }
    stubBridge()
    // No session and pushcart-web has not answered yet: connecting, not ready.
    const first = render(
      <PosAdminSection posStateDeps={posState({ ...base, lastContactMs: null })} />
    )
    await waitFor(() => expect(screen.getByTestId('pos-status')).toHaveTextContent('Connecting…'))
    expect(screen.getByTestId('pos-status')).not.toHaveTextContent('Ready')
    first.unmount()

    stubBridge()
    const { unmount } = render(<PosAdminSection posStateDeps={posState(base)} />)
    await waitFor(() => expect(screen.getByTestId('pos-status')).toHaveTextContent('Ready'))
    unmount()

    stubBridge()
    const bound = render(
      <PosAdminSection
        posStateDeps={posState({ ...base, phase: 'bound', cartCode: 'abcdef1234' })}
      />
    )
    await waitFor(() =>
      expect(screen.getByTestId('pos-status')).toHaveTextContent('Customer session open')
    )
    expect(screen.getByTestId('pos-status')).toHaveTextContent('Cart abcdef12')
    bound.unmount()

    stubBridge()
    render(
      <PosAdminSection
        posStateDeps={posState({ ...base, phase: 'error', error: 'fetch failed' })}
      />
    )
    await waitFor(() => expect(screen.getByTestId('pos-status')).toHaveTextContent('Error'))
    expect(screen.getByTestId('pos-status')).toHaveTextContent('fetch failed')
    expect(screen.getByTestId('pos-status')).toHaveAttribute('data-tone', 'bad')
  })
})
