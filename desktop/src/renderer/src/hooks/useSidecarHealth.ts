import { useEffect, useState } from 'react'
import type { SidecarHealth } from '../../../main/sidecarHealth'

export interface SidecarHealthDeps {
  // Injectable for tests; the defaults are the preload bridge. Module-level functions rather than
  // inline arrows, so the effect's dependency list is stable across renders.
  read?: () => Promise<SidecarHealth>
  subscribe?: (cb: (health: SidecarHealth) => void) => () => void
}

function readHealth(): Promise<SidecarHealth> {
  return window.api.getSidecarHealth()
}

function subscribeHealth(cb: (health: SidecarHealth) => void): () => void {
  return window.api.onSidecarHealth(cb)
}

// Whether the sidecar is answering at all, as the *main process* sees it.
//
// Deliberately not derived here from what this window can observe. The renderer's own signals are
// established rather than live: `ws.ts`'s socket stays open across the failure that matters (a
// sidecar that keeps running with nothing listening), and a hung `fetch` looks exactly like a slow
// one. The main process owns the child, so it is the only side that can answer "is it there", and
// it answers with one state that a renderer renders rather than re-derives.
//
// `starting` is the initial state and is not an error: at launch nothing has been asked yet, and a
// warning during a healthy startup would be the one thing that teaches an operator to ignore it.
export function useSidecarHealth(deps: SidecarHealthDeps = {}): SidecarHealth {
  const read = deps.read ?? readHealth
  const subscribe = deps.subscribe ?? subscribeHealth
  const [health, setHealth] = useState<SidecarHealth>('starting')

  useEffect(() => {
    let active = true
    // Both a subscription and a read, because each one alone loses a transition. A window that
    // mounts between a change and its broadcast would sit on the stale value until the *next*
    // change — and a sidecar that stays broken never sends one — while a subscription-only version
    // would miss the state that was already true when it arrived.
    const unsubscribe = subscribe((next) => {
      if (active) setHealth(next)
    })
    void read()
      .then((next) => {
        if (active) setHealth(next)
      })
      .catch(() => {
        // A failed read says nothing about the sidecar — it is an IPC failure — so the state is
        // left alone rather than turned into an alarm the main process never reported.
      })
    return () => {
      active = false
      unsubscribe()
    }
  }, [read, subscribe])

  return health
}
