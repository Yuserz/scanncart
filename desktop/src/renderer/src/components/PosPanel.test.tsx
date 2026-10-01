import { describe, it, expect, vi, afterEach } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import { PosPanel } from './PosPanel'
import type { PosState } from '../../../main/posSession'

const BOUND: PosState = {
  phase: 'bound',
  cartCode: 'abcdef1234567890',
  syncedItemCount: 3,
  lastSyncAgeS: 12,
  error: null
}

type Listener = (state: PosState) => void

function stubBridge(initial: PosState | null): {
  push: (state: PosState) => void
  unsubscribed: () => boolean
} {
  let listener: Listener | null = null
  vi.stubGlobal('api', {
    getPosState: async () => initial,
    onPosState: (cb: Listener) => {
      listener = cb
      return () => {
        listener = null
      }
    }
  })
  return { push: (state) => listener?.(state), unsubscribed: () => listener === null }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('PosPanel', () => {
  it('renders nothing while the feature is off', async () => {
    stubBridge(null)
    const { container } = render(<PosPanel />)

    await waitFor(() => expect(screen.queryByTestId('pos-panel')).toBeNull())
    // Nothing at all, rather than an empty card: the feature being off is not a state to read.
    expect(container.firstChild).toBeNull()
  })

  it('reads out the binding, the cart and the item count', async () => {
    stubBridge(BOUND)
    render(<PosPanel />)

    await waitFor(() => expect(screen.getByTestId('pos-phase')).toHaveTextContent('syncing'))
    // Truncated: the rail is 300px and the code is not something an operator reads off the desktop
    // — it is what the tablet displays, and the prefix is enough to tell two carts apart.
    expect(screen.getByTestId('pos-cart')).toHaveTextContent('abcdef12')
    expect(screen.getByTestId('pos-cart').textContent).toHaveLength(8)
    expect(screen.getByTestId('pos-items')).toHaveTextContent('3')
    expect(screen.getByTestId('pos-age')).toHaveTextContent('12s ago')
    expect(screen.queryByTestId('pos-error')).toBeNull()
  })

  it('says a sync has never happened rather than showing a zero age', async () => {
    stubBridge({
      ...BOUND,
      phase: 'unbound',
      cartCode: null,
      syncedItemCount: 0,
      lastSyncAgeS: null
    })
    render(<PosPanel />)

    await waitFor(() => expect(screen.getByTestId('pos-phase')).toHaveTextContent('no session'))
    expect(screen.getByTestId('pos-age')).toHaveTextContent('never')
    // No cart yet, so no cart row at all — a placeholder would name a cart that does not exist.
    expect(screen.queryByTestId('pos-cart')).toBeNull()
  })

  it('follows a transition while the window is open', async () => {
    const bridge = stubBridge(BOUND)
    render(<PosPanel />)
    await waitFor(() => expect(screen.getByTestId('pos-phase')).toHaveTextContent('syncing'))

    act(() => bridge.push({ ...BOUND, phase: 'warming_up', syncedItemCount: 4 }))

    expect(screen.getByTestId('pos-phase')).toHaveTextContent('warming up')
    expect(screen.getByTestId('pos-items')).toHaveTextContent('4')
  })

  it('shows a sync error beside the state it broke', async () => {
    stubBridge({ ...BOUND, phase: 'error', error: 'webapp unreachable' })
    render(<PosPanel />)

    await waitFor(() =>
      expect(screen.getByTestId('pos-error')).toHaveTextContent('webapp unreachable')
    )
    expect(screen.getByTestId('pos-phase')).toHaveTextContent('error')
  })

  it('renders nothing when the bridge has no POS methods', async () => {
    // An older preload (or a harness that stubs only what it needs) has no `getPosState`; the
    // readout degrades to nothing instead of taking the Live view down with it.
    vi.stubGlobal('api', {})
    render(<PosPanel />)

    await waitFor(() => expect(screen.queryByTestId('pos-panel')).toBeNull())
  })
})
