// The pushcart-web transport (spec §4.2, §7): the two desktop-facing POS routes, plus the read the
// Admin Panel's "Test connection" makes.
//
// Electron-free on purpose. `pos.ts` holds the Electron wiring and the config store; this module is
// the part that talks to the wire, so `posRoutes.integration.test.ts` can drive it against a real
// HTTP server and exercise the request shape (the `x-pos-token` header, the JSON snapshot) and the
// status-code mapping over a socket rather than through a stubbed `fetch`.

import {
  PosConflictError,
  PosTransportError,
  type PosSyncPayload,
  type PosSyncResponse,
  type RemoteSession
} from './posSession'

/** Ceiling on a pushcart-web request: the LAN host answers in milliseconds, the internet may not. */
export const WEBAPP_TIMEOUT_MS = 8000

export function trimSlash(url: string): string {
  return url.replace(/\/+$/, '')
}

/** pushcart-web's session route, with the station id encoded into its query. */
export function sessionUrl(baseUrl: string, stationId: string): string {
  return `${trimSlash(baseUrl)}/api/pos/session?station_id=${encodeURIComponent(stationId)}`
}

/** pushcart-web's reconcile route. */
export function syncUrl(baseUrl: string): string {
  return `${trimSlash(baseUrl)}/api/pos/sync`
}

/**
 * `fetch` with the connection failure turned into the type the contract promises.
 *
 * A refused connection or a DNS failure is a plain `TypeError` out of `fetch`, but every caller here
 * reasons in terms of `PosTransportError` ("pushcart-web could not be reached") — so without this the
 * one failure the smoke test spends a whole row on would arrive as a `TypeError: fetch failed` with
 * nothing to catch it by. The message names the host and the reason - the errno for a refused
 * connection, the deadline for a timeout - because it is shown to an operator as it is: `fetch`'s own
 * "The operation was aborted due to timeout" or "fetch failed" says neither what nor where.
 */
async function request(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init)
  } catch (error) {
    throw new PosTransportError(unreachableMessage(url, error))
  }
}

/** Why a request never got an answer, in words that name the host. */
export function unreachableMessage(url: string, error: unknown): string {
  let host = url
  try {
    host = new URL(url).origin
  } catch {
    // An unparseable URL is still the best name for where the request went.
  }
  const err = error as { name?: string; message?: string; cause?: { code?: string } } | null
  if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
    return `pushcart-web at ${host} did not answer within ${WEBAPP_TIMEOUT_MS / 1000} s`
  }
  const code = err?.cause?.code
  return `pushcart-web at ${host} could not be reached (${code ?? err?.message ?? String(error)})`
}

export interface PosSessionProbe {
  /** The HTTP status, or null when the host could not be reached at all. */
  status: number | null
  /** The `data` object; null when the response carried none, or when it was not ok. */
  data: Record<string, unknown> | null
  /** The connection failure's own message, or null when a response arrived. */
  unreachable: string | null
}

/**
 * "Test connection": ask pushcart-web for this station's session once and report what came back.
 *
 * Never throws — the panel renders the result, so a failure is a value rather than an exception the
 * call site has to catch. Only a successful response is parsed: a 401/404/500 body is the status's
 * to describe, not the panel's.
 */
export async function probePosSession(
  baseUrl: string,
  secret: string,
  stationId: string
): Promise<PosSessionProbe> {
  let response: Response
  try {
    response = await request(sessionUrl(baseUrl, stationId), {
      headers: { 'x-pos-token': secret },
      signal: AbortSignal.timeout(WEBAPP_TIMEOUT_MS)
    })
  } catch (error) {
    return {
      status: null,
      data: null,
      unreachable: error instanceof Error ? error.message : String(error)
    }
  }

  if (!response.ok) return { status: response.status, data: null, unreachable: null }

  const body = (await response.json().catch(() => null)) as {
    data?: Record<string, unknown>
  } | null
  return { status: response.status, data: body?.data ?? null, unreachable: null }
}

/**
 * The session poll. `null` is "no session open" — a healthy answer, not an error. A 409 is the
 * webapp saying the session is over; any other non-ok status is a transport failure.
 */
export async function fetchRemoteSession(
  baseUrl: string,
  secret: string,
  stationId: string
): Promise<RemoteSession | null> {
  const response = await request(sessionUrl(baseUrl, stationId), {
    headers: { 'x-pos-token': secret },
    signal: AbortSignal.timeout(WEBAPP_TIMEOUT_MS)
  })

  if (response.status === 409) throw new PosConflictError('session conflict')
  if (!response.ok) throw new PosTransportError(`session poll failed (${response.status})`)

  const data = ((await response.json()) as { data?: Record<string, unknown> | null }).data
  if (!data) return null

  return {
    sessionRef: String(data.session_ref),
    cartId: String(data.cart_id),
    cartCode: String(data.cart_code ?? ''),
    cartStatus: String(data.cart_status ?? '')
  }
}

/**
 * The desired-cart snapshot. `409` means the session is over — Finish won, or the cart is paid — and
 * is the one failure that is a *state* rather than a transport problem; anything else not ok is
 * transport.
 */
export async function postCartSync(
  baseUrl: string,
  secret: string,
  payload: PosSyncPayload
): Promise<PosSyncResponse> {
  const response = await request(syncUrl(baseUrl), {
    method: 'POST',
    headers: { 'x-pos-token': secret, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(WEBAPP_TIMEOUT_MS)
  })

  if (response.status === 409) throw new PosConflictError('session closed or cart paid')
  if (!response.ok) throw new PosTransportError(`sync failed (${response.status})`)

  const data = ((await response.json()) as { data?: PosSyncResponse }).data
  return data ?? {}
}
