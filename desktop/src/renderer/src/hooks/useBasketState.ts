import { useEffect, useState } from 'react'
import type { BasketViewState } from '../../../main/transferStream'

export interface BasketStateDeps {
  // Injectable for tests; the defaults are the preload bridge. Module-level functions so the
  // effect's dependency list is stable across renders, as in `usePosState`.
  read?: () => Promise<BasketViewState | null>
  subscribe?: (cb: (state: BasketViewState) => void) => () => void
}

// Optional all the way down for the same reason `usePosState` is: the bridge is additive across
// the process boundary, and a window whose preload predates these methods (or has none) must show
// "not available" rather than take the screen down.
type BasketBridge = {
  getBasketState?: () => Promise<BasketViewState | null>
  onBasketState?: (cb: (state: BasketViewState) => void) => () => void
  startBasketPractice?: () => Promise<BasketViewState>
  stopBasketPractice?: () => Promise<BasketViewState | null>
  resolvePosReview?: (id?: string) => Promise<void>
}

export function basketBridge(): BasketBridge {
  return (window.api ?? {}) as BasketBridge
}

function readBasketState(): Promise<BasketViewState | null> {
  return basketBridge().getBasketState?.() ?? Promise.resolve(null)
}

function subscribeBasketState(cb: (state: BasketViewState) => void): () => void {
  return basketBridge().onBasketState?.(cb) ?? (() => {})
}

// The basket ledger as the main process reports it: a read for a window that mounts late and a
// subscription for one already open, because either alone loses a transition. `null` until the
// main process answers, which is also what a window with no basket bridge stays at.
export function useBasketState(deps: BasketStateDeps = {}): BasketViewState | null {
  const read = deps.read ?? readBasketState
  const subscribe = deps.subscribe ?? subscribeBasketState
  const [state, setState] = useState<BasketViewState | null>(null)

  useEffect(() => {
    let active = true
    const unsubscribe = subscribe((next) => {
      if (active) setState(next)
    })
    void read()
      .then((next) => {
        if (active && next) setState(next)
      })
      .catch(() => {
        // An IPC failure says nothing about the basket; keep what we have.
      })
    return () => {
      active = false
      unsubscribe()
    }
  }, [read, subscribe])

  return state
}
