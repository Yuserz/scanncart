import { useEffect, useState } from 'react'
import type { PosState } from '../../../main/posSession'

export interface PosStateDeps {
  // Injectable for tests; the defaults are the preload bridge. Module-level functions rather than
  // inline arrows, so the effect's dependency list is stable across renders. `null` is part of both
  // signatures: the push carries it when the feature is switched off, exactly as the read does.
  read?: () => Promise<PosState | null>
  subscribe?: (cb: (state: PosState | null) => void) => () => void
}

// The bridge is additive across the process boundary: a renderer carrying the POS panel can run
// against a preload that predates it (the two halves are versioned apart), and the layout harness
// stubs the app's preload down to the handful of methods it needs. Reading through an optional view
// is what keeps a readout that cannot be answered from taking the Live view down with it — a window
// with no POS methods reports nothing, which the panel already renders as nothing.
type PosBridge = {
  getPosState?: () => Promise<PosState | null>
  onPosState?: (cb: (state: PosState | null) => void) => () => void
}

function posBridge(): PosBridge {
  // `window.api` itself is absent on a window whose preload never ran — which is what a bare test
  // harness and the layout harness both are — so the whole read is optional, not just the method.
  return (window.api ?? {}) as PosBridge
}

function readPosState(): Promise<PosState | null> {
  return posBridge().getPosState?.() ?? Promise.resolve(null)
}

function subscribePosState(cb: (state: PosState | null) => void): () => void {
  return posBridge().onPosState?.(cb) ?? (() => {})
}

// The POS integration's state, as the main process reports it. `null` means there is nothing to
// report — the feature is off (an empty URL, secret or station id disables it, spec §5.1), or
// nothing has been emitted yet — and the panel renders nothing for it, the same rule the inference
// verdict follows and for the same reason: a permanent notice about a feature nobody turned on is
// what that rule exists to stop. The main process owns the loop, so this reads rather than derives,
// with a subscription and a read because either alone loses a transition.
export function usePosState(deps: PosStateDeps = {}): PosState | null {
  const read = deps.read ?? readPosState
  const subscribe = deps.subscribe ?? subscribePosState
  const [state, setState] = useState<PosState | null>(null)

  useEffect(() => {
    let active = true
    const unsubscribe = subscribe((next) => {
      if (active) setState(next)
    })
    void read()
      .then((next) => {
        if (active) setState(next)
      })
      .catch(() => {
        // An IPC failure is not a verdict about the integration; leave the state alone.
      })
    return () => {
      active = false
      unsubscribe()
    }
  }, [read, subscribe])

  return state
}
