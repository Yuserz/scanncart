// @vitest-environment node
// Does this repo still encode pushcart-web's POS contract? (spec §7)
//
// This is the half that runs on a bare checkout. Its other half is `scripts/check-pos-contract.mjs`,
// which reads the contract out of a real pushcart-web checkout and keeps `posContract.json` fresh —
// it cannot run in CI, because the other checkout is not here. So the recorded file is the seam: the
// script keeps it honest about pushcart-web, and this keeps `posClient.ts` and the stand-in honest
// about the file.
//
// What it catches that the integration suite cannot: both sides of that suite were written together
// from the same reading, so the header name, the paths and the status codes could all move at once
// and it would still pass. Nothing else in this repo knows what pushcart-web actually answers with.
//
// Read as text, not parsed. Every fact here is a literal in the file that owns it, and the
// alternative — importing and inspecting the modules — would restate their structure in a test that
// is free to disagree with it. The price is that the patterns below describe a line's shape, so a
// reformat that splits one is a failure to re-read rather than a real drift; they are written to
// tolerate wrapping where it costs nothing (a `\s*` between the tokens).

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const here = dirname(fileURLToPath(import.meta.url))
const text = (name: string): string => readFileSync(join(here, name), 'utf8')

interface PosContract {
  auth_header: string
  outcomes: Record<string, string>
  routes: Record<string, string>
  statuses: Record<string, number>
}

const contract = JSON.parse(text('posContract.json')) as PosContract
const client = text('posClient.ts')
const standIn = text('posWebappStandIn.ts')

const { routes, statuses } = contract

describe("the modules still encode pushcart-web's POS contract", () => {
  it('names the routes pushcart-web serves, with the methods it exports', () => {
    expect(Object.keys(routes)).toEqual(['/api/pos/session', '/api/pos/sync'])

    for (const [path, method] of Object.entries(routes)) {
      expect(client, `posClient.ts no longer names ${path}`).toContain(path)
      expect(standIn, `the stand-in no longer answers ${path}`).toContain(path)
      // The stand-in dispatches on both, so a path that moved without its method moving is caught.
      expect(standIn, `the stand-in no longer answers ${method} ${path}`).toMatch(
        new RegExp(`url\\.pathname === '${path}' && req\\.method === '${method}'`)
      )
      // `fetch` defaults to GET, so only a method it does not default to is written down.
      if (method !== 'GET') {
        expect(client, `posClient.ts no longer sends ${method}`).toMatch(
          new RegExp(`method:\\s*'${method}'`)
        )
      }
    }
  })

  it('sends the auth header pushcart-web reads', () => {
    expect(client).toMatch(new RegExp(`'${contract.auth_header}':\\s*secret`))
    expect(standIn, 'the stand-in no longer reads the header it authenticates with').toContain(
      contract.auth_header
    )
  })

  it('maps the conflict status to a conflict, and every other non-ok status to transport', () => {
    // The contract is where the number comes from: if pushcart-web stopped answering 409 for a
    // session that is over, this fails here rather than as a mystifying unbound cart.
    expect(statuses.conflictRequestResponse).toBe(409)

    expect(client).toMatch(
      new RegExp(`status === ${statuses.conflictRequestResponse}\\)\\s*throw new PosConflictError`)
    )
    expect((client.match(/throw new PosConflictError/g) ?? []).length).toBe(2)
    // Both routes — the session poll and the reconcile — answer anything else as transport.
    expect(
      (client.match(/if \(!response\.ok\)\s*throw new PosTransportError/g) ?? []).length,
      'every non-ok status that is not the conflict must be a transport failure'
    ).toBe(2)
  })

  it('answers the statuses pushcart-web answers', () => {
    // The ones the desktop's mapping depends on: success, the snapshot validation, the secret, the
    // station, the conflict, and the internal error the panel has to survive.
    const answered = [
      'successResponse',
      'validationErrorNextResponse',
      'unauthorizedResponse',
      'notFoundResponse',
      'conflictRequestResponse',
      'generalErrorResponse'
    ]

    for (const helper of answered) {
      const status = statuses[helper]
      expect(typeof status, `the contract no longer records ${helper}`).toBe('number')
      expect(standIn, `the stand-in no longer answers ${status} (${helper})`).toMatch(
        new RegExp(`respond\\(res, ${status},`)
      )
    }
  })

  it('records why a 409 means the session is over rather than a retry', () => {
    // The desktop's whole conflict rule rests on this: these codes are conflict outcomes, so the
    // number it treats as "the session ended" is the one pushcart-web uses for them.
    expect(contract.outcomes.session_closed).toBe('conflictRequestResponse')
    expect(contract.outcomes.cart_paid).toBe('conflictRequestResponse')
    expect(statuses.conflictRequestResponse).toBe(409)
  })
})

// The third claim, and the only one about the *other* copy of this contract: pushcart-web derives
// and records the same surface from its own routes, and `scripts/check-pos-contract.mjs` compares
// the two records so a change recorded on one side only cannot leave both repos green while they
// describe different routes. That comparison needs the sibling checkout, so it cannot run here — but
// its *shape* can be pinned, and that is where it could quietly narrow: a fifth key added to
// `posContract.json` that the comparison does not list is a fact the two repos may disagree about
// unobserved.
const script = readFileSync(join(here, '..', '..', 'scripts', 'check-pos-contract.mjs'), 'utf8')

describe('the two recorded copies of the contract are compared key by key', () => {
  it('compares every key of the record except the provenance note, and no others', () => {
    const listed = /const COMPARED = \[([^\]]*)\]/.exec(script)?.[1] ?? ''
    const compared = listed
      .split(',')
      .map((part) => part.trim().replace(/^'|'$/g, ''))
      .filter(Boolean)

    const keys = Object.keys(contract)
    expect(keys).toContain('note')
    expect([...compared].sort()).toEqual([...keys.filter((key) => key !== 'note')].sort())
    expect([...compared].sort()).toEqual(['auth_header', 'outcomes', 'routes', 'statuses'])
  })

  it('reads that copy from the checkout it already has, and says why `note` is left out', () => {
    expect(script).toContain("join(pushcart, 'scripts', 'pos-contract.json')")
    expect(script, 'the exclusion needs its reason beside it, not just the omission').toContain(
      '`note` is deliberately not compared'
    )
  })
})
