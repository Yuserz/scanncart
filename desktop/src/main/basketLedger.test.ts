// @vitest-environment node
import { mkdtemp, readFile, rm, writeFile } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { BASKET_LEDGER_FILENAME, BasketLedger, BasketLedgerFileStore } from './basketLedger'
import type { TransferEvent } from './transferState'

const ev = (
  kind: 'inbound' | 'outbound',
  className = 'soda',
  completedAt = 10,
  trackId = 1
): TransferEvent => ({
  kind,
  className,
  completedAt,
  trackId,
  completionConf: 0.9
})

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'ledger-'))
})
// Every ledger a test makes is tracked, so cleanup waits for its last save: on Windows a write still
// landing in the directory makes the removal fail, which surfaced as a flaky test under load.
const ledgers: BasketLedger[] = []
function ledger(store?: BasketLedgerFileStore): BasketLedger {
  const l = new BasketLedger(store)
  ledgers.push(l)
  return l
}

afterEach(async () => {
  await Promise.all(ledgers.splice(0).map((l) => l.flushed()))
  await rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })
})

describe('basket ledger', () => {
  it('deposits add and removals subtract, per product', async () => {
    const l = ledger()
    await l.bind('s1', 0)
    l.apply(ev('inbound', 'soda', 1), 'z')
    l.apply(ev('inbound', 'soda', 2, 2), 'z')
    l.apply(ev('inbound', 'tuna', 3, 3), 'z')
    l.apply(ev('outbound', 'soda', 4, 2), 'z')
    expect(Object.fromEntries([...l.snapshot()].map(([k, v]) => [k, v.quantity]))).toEqual({
      soda: 1,
      tuna: 1
    })
  })

  it('applies an event once however many times it arrives', async () => {
    const l = ledger()
    await l.bind('s1', 0)
    expect(l.apply(ev('inbound'), 'z').status).toBe('applied')
    expect(l.apply(ev('inbound'), 'z').status).toBe('duplicate')
    expect(l.snapshot().get('soda')?.quantity).toBe(1)
  })

  it('a removal of something not in the basket is review, never a negative count', async () => {
    const l = ledger()
    await l.bind('s1', 0)
    const r = l.apply(ev('outbound', 'tuna'), 'z')
    expect(r.status).toBe('review')
    expect(l.snapshot().size).toBe(0)
    expect(l.pendingReview()).toHaveLength(1)
  })

  it('a hidden item is still in the basket: nothing but a removal takes it out', async () => {
    const l = ledger()
    await l.bind('s1', 0)
    l.apply(ev('inbound'), 'z')
    l.markBlind(20)
    l.markSeeing(21, 5) // a short occlusion is not a blind interval worth reviewing
    expect(l.snapshot().get('soda')?.quantity).toBe(1)
    expect(l.pendingReview()).toEqual([])
  })

  it('a long blind interval keeps the basket and asks for review', async () => {
    const l = ledger()
    await l.bind('s1', 0)
    l.apply(ev('inbound'), 'z')
    l.markBlind(20)
    l.markSeeing(60, 5)
    expect(l.snapshot().get('soda')?.quantity).toBe(1)
    expect(l.pendingReview()[0].reason).toContain('could not see')
    l.resolveReview()
    expect(l.pendingReview()).toEqual([])
  })

  it('does nothing while unbound', () => {
    const l = ledger()
    expect(l.apply(ev('inbound'), 'z').status).toBe('unbound')
    expect(l.flag('soda', 'x', 0)).toBeNull()
  })
})

describe('basket ledger persistence', () => {
  it('an app restart in the same session restores the basket and flags review', async () => {
    const a = ledger(new BasketLedgerFileStore(dir))
    await a.bind('s1', 0)
    a.apply(ev('inbound'), 'z')
    await a.flushed()

    const b = ledger(new BasketLedgerFileStore(dir))
    await b.bind('s1', 100)
    expect(b.snapshot().get('soda')?.quantity).toBe(1)
    expect(b.pendingReview().some((r) => r.reason.includes('restarted'))).toBe(true)
    // The restored journal still deduplicates.
    expect(b.apply(ev('inbound'), 'z').status).toBe('duplicate')
  })

  it('a new session never inherits the previous basket', async () => {
    const a = ledger(new BasketLedgerFileStore(dir))
    await a.bind('s1', 0)
    a.apply(ev('inbound'), 'z')
    await a.flushed()

    const b = ledger(new BasketLedgerFileStore(dir))
    await b.bind('s2', 100)
    expect(b.snapshot().size).toBe(0)
    expect(b.pendingReview()).toEqual([])
  })

  it('unbind removes the saved copy', async () => {
    const a = ledger(new BasketLedgerFileStore(dir))
    await a.bind('s1', 0)
    a.apply(ev('inbound'), 'z')
    await a.flushed()
    a.unbind()
    await new Promise((r) => setTimeout(r, 20))
    await expect(readFile(join(dir, BASKET_LEDGER_FILENAME), 'utf8')).rejects.toThrow()
  })

  it('a corrupt file reads as nothing saved', async () => {
    await writeFile(join(dir, BASKET_LEDGER_FILENAME), '{nope', 'utf8')
    expect(await new BasketLedgerFileStore(dir).load()).toBeNull()
  })
})
