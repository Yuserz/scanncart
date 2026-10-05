// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { unreachableMessage, WEBAPP_TIMEOUT_MS } from './posClient'

describe('unreachableMessage', () => {
  it("names the host and the deadline for a timeout, not fetch's own wording", () => {
    const timeout = Object.assign(new Error('The operation was aborted due to timeout'), {
      name: 'TimeoutError'
    })
    expect(unreachableMessage('http://localhost:3000/api/pos/session?station_id=a', timeout)).toBe(
      `pushcart-web at http://localhost:3000 did not answer within ${WEBAPP_TIMEOUT_MS / 1000} s`
    )
  })

  it('names the errno for a refused connection', () => {
    const refused = Object.assign(new TypeError('fetch failed'), {
      cause: { code: 'ECONNREFUSED' }
    })
    expect(unreachableMessage('http://192.168.1.20:3000/api/pos/sync', refused)).toBe(
      'pushcart-web at http://192.168.1.20:3000 could not be reached (ECONNREFUSED)'
    )
  })

  it("falls back to the error's own message when there is no errno", () => {
    expect(unreachableMessage('not a url', new Error('boom'))).toBe(
      'pushcart-web at not a url could not be reached (boom)'
    )
  })
})
