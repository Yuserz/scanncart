import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { PosAdminSection } from './PosAdminSection'
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
  it('loads the saved configuration into the form', async () => {
    stubBridge()
    render(<PosAdminSection />)

    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))
    expect(screen.getByTestId('pos-station-id')).toHaveValue('counter-1')
  })

  it('saves the edited config, including numbers', async () => {
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-base-url')).toHaveValue(CONFIG.posBaseUrl))

    fireEvent.change(screen.getByTestId('pos-station-id'), { target: { value: 'counter-2' } })
    fireEvent.change(screen.getByTestId('pos-commitDwellS'), { target: { value: '4' } })
    fireEvent.click(screen.getByText('Save POS settings'))

    await waitFor(() => expect(saved).not.toBeNull())
    expect(saved).toMatchObject({ stationId: 'counter-2', commitDwellS: 4 })
  })

  it('saves the cart mode, the cart edge and the band sizes', async () => {
    stubBridge()
    render(<PosAdminSection />)
    await waitFor(() => expect(screen.getByTestId('pos-cartMode')).toHaveValue('counter'))

    fireEvent.change(screen.getByTestId('pos-cartMode'), { target: { value: 'basket' } })
    fireEvent.change(screen.getByTestId('pos-cartEdge'), { target: { value: 'left' } })
    fireEvent.change(screen.getByTestId('pos-insideFraction'), { target: { value: '0.3' } })
    fireEvent.click(screen.getByText('Save POS settings'))

    await waitFor(() => expect(saved).not.toBeNull())
    expect(saved).toMatchObject({ cartMode: 'basket', cartEdge: 'left', insideFraction: 0.3 })
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
    expect(screen.getByTestId('pos-tls-warning')).toHaveTextContent('https://')
  })
})
