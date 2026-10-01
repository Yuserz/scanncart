import type { JSX } from 'react'
import { usePosState } from '../hooks/usePosState'

const PHASE_LABELS: Record<string, string> = {
  unbound: 'no session',
  warming_up: 'warming up',
  bound: 'syncing',
  error: 'error'
}

function describeAge(seconds: number | null): string {
  if (seconds === null) return 'never'
  if (seconds < 90) return `${Math.round(seconds)}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  return `${Math.round(minutes / 60)}h ago`
}

// A read-only readout of the POS integration: whether the desktop is bound to a counter session,
// which cart, how many items it has synced, and any error. Nothing here controls the coupling —
// the tablet's Start is what binds it — so there is nothing to click.
export function PosPanel(): JSX.Element | null {
  const state = usePosState()

  if (state === null) return null

  return (
    <div className="card pos-card" data-testid="pos-panel">
      <h4>Self-checkout</h4>
      <div className="pos-rows">
        <div className="pos-row">
          <small>status</small>
          <b data-testid="pos-phase">{PHASE_LABELS[state.phase] ?? state.phase}</b>
        </div>
        {state.cartCode && (
          <div className="pos-row">
            <small>cart</small>
            <b data-testid="pos-cart">{state.cartCode.slice(0, 8)}</b>
          </div>
        )}
        <div className="pos-row">
          <small>items synced</small>
          <b data-testid="pos-items">{state.syncedItemCount}</b>
        </div>
        <div className="pos-row">
          <small>last sync</small>
          <b data-testid="pos-age">{describeAge(state.lastSyncAgeS)}</b>
        </div>
      </div>
      {state.error && (
        <p className="pos-error" data-testid="pos-error">
          {state.error}
        </p>
      )}
    </div>
  )
}
