// A local stand-in for pushcart-web's desktop-facing POS routes, for the integration test.
//
// It is deliberately not a mock of `fetch`. The test drives the real transport (`posClient.ts`) and
// the real orchestrator (`posSession.ts`) over a loopback socket, so the request shape and the
// status-code mapping are exercised the way the shop exercises them. The contract it mirrors is
// pushcart-web's own: `app/api/pos/session/route.ts`, `app/api/pos/sync/route.ts`, its `validate()`,
// and the `pos_reconcile` function's `{results, cart_totals}` body — the `x-pos-token` header, 401
// on a bad secret, 404 for an unregistered station, 400 for a malformed snapshot, 409 for a session
// that is over, and a reconcile keyed by cart so a repeated snapshot cannot duplicate a row.
//
// What it does not model, on purpose: Supabase, the class->product map, overrides, stock, prices
// beyond a plain table. A scenario that needs one of those is a scenario for the real checkout and a
// person — this is the part of the smoke test a machine can hold. See `docs/POS_SMOKE_TEST.md` §3.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'

/** One open session, as pushcart-web's session route projects it onto the wire. */
export interface StandInSession {
  session_ref: string
  cart_id: string
  cart_code: string
  cart_status: string
}

/** One request the stand-in answered, kept so a test can assert what actually went over the socket. */
export interface StandInRequest {
  method: string
  path: string
  stationId: string | null
  token: string | null
  payload: {
    session_ref: string
    station_id: string
    items: Array<{ class_name: string; quantity: number; max_confidence?: number }>
    pending_review?: unknown
    review_reasons?: unknown
  } | null
}

/**
 * `server_error` is an injected 500. `conflict` is the poll/sync race that is real in the shop:
 * the session ends (Finish, or a cancel) between the session poll that still saw it open and the
 * snapshot that follows — so the poll keeps reporting it while the reconcile refuses the sync.
 */
export type StandInMode = 'ok' | 'server_error' | 'conflict'

// `Connection: close` on every response: undici otherwise pools a keep-alive socket, which makes
// `stop()` hang until the pool idles and lets a restarted server inherit a dead connection.
const respond = (res: ServerResponse, status: number, body: unknown): void => {
  res.writeHead(status, { 'Content-Type': 'application/json', Connection: 'close' })
  res.end(JSON.stringify(body))
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(chunk as Buffer)
  return Buffer.concat(chunks).toString('utf8')
}

/** pushcart-web's `validate()`, verbatim in its refusals and in the order it finds them. */
function validate(payload: StandInRequest['payload']): string | null {
  if (!payload || typeof payload !== 'object') return 'invalid body'
  if (typeof payload.session_ref !== 'string' || !payload.session_ref) return 'session_ref required'
  if (typeof payload.station_id !== 'string' || !payload.station_id) return 'station_id required'
  if (!Array.isArray(payload.items)) return 'items must be an array'
  if (payload.items.length > 200) return 'at most 200 items'

  const seen = new Set<string>()
  for (const item of payload.items) {
    if (!item || typeof item.class_name !== 'string' || !item.class_name) {
      return 'class_name required'
    }
    if (!Number.isInteger(item.quantity) || item.quantity < 1) {
      return 'quantity must be an integer >= 1'
    }
    if (seen.has(item.class_name)) return `duplicate class_name: ${item.class_name}`
    seen.add(item.class_name)
  }
  // Basket mode's review state, refused in pushcart-web's words and order.
  if (payload.pending_review !== undefined) {
    const n = payload.pending_review
    if (typeof n !== 'number' || !Number.isInteger(n) || n < 0) {
      return 'pending_review must be an integer >= 0'
    }
  }
  if (payload.review_reasons !== undefined) {
    const r = payload.review_reasons
    if (
      !Array.isArray(r) ||
      r.length > 50 ||
      r.some((x) => typeof x !== 'string' || x.length > 300)
    ) {
      return 'review_reasons must be at most 50 strings'
    }
  }
  return null
}

export class PosWebappStandIn {
  private server: Server | null = null
  private mode: StandInMode = 'ok'
  private boundPort = 0

  /** Registered stations and their open session, if any: an unregistered id is a 404. */
  private readonly stations = new Map<string, StandInSession | null>()
  /** Committed camera rows per cart, keyed by class — which is what makes a re-sync idempotent. */
  private readonly items = new Map<string, Map<string, number>>()
  /** Unit prices by class, for the `cart_totals.subtotal` the reconcile reports. */
  readonly prices: Record<string, number> = {}

  /** Everything the stand-in was asked, in order. */
  readonly requests: StandInRequest[] = []

  constructor(
    private secret: string,
    stationIds: string[] = []
  ) {
    for (const id of stationIds) this.stations.set(id, null)
  }

  get url(): string {
    return `http://127.0.0.1:${this.boundPort}`
  }

  get port(): number {
    return this.boundPort
  }

  setSecret(secret: string): void {
    this.secret = secret
  }

  /** `server_error` is the only injected fault: the rest are expressed as state (see the docs). */
  setMode(mode: StandInMode): void {
    this.mode = mode
  }

  registerStation(id: string): void {
    this.stations.set(id, null)
  }

  openSession(stationId: string, session: StandInSession): void {
    this.stations.set(stationId, session)
  }

  /** No session open — a healthy `data: null`, not an error. */
  closeSession(stationId: string): void {
    this.stations.set(stationId, null)
  }

  /** The committed camera rows for a cart: what the reconcile left, one row per class. */
  cartItems(cartId: string): Record<string, number> {
    return Object.fromEntries(this.items.get(cartId) ?? [])
  }

  /**
   * Listen on `port` (0 = pick a free one) and answer the routes. Re-listening on a port this
   * stand-in just gave up is how a test simulates the host coming back, so a bind that loses the
   * race to the OS' own cleanup is retried rather than failing the run.
   */
  async start(port = 0): Promise<number> {
    for (let attempt = 0; ; attempt += 1) {
      try {
        await this.listen(port)
        break
      } catch (error) {
        if (attempt >= 4) throw error
        await new Promise((resolve) => setTimeout(resolve, 50))
      }
    }
    return this.boundPort
  }

  private listen(port: number): Promise<void> {
    const server = createServer((req, res) => {
      void this.handle(req, res).catch(() => {
        if (!res.headersSent) respond(res, 500, { error: 'stand-in failure' })
      })
    })
    return new Promise<void>((resolve, reject) => {
      server.once('error', (error) => {
        server.close()
        reject(error)
      })
      server.listen(port, '127.0.0.1', () => {
        this.server = server
        this.boundPort = (server.address() as AddressInfo).port
        resolve()
      })
    })
  }

  /** Stop listening: from the desktop's side, the host is gone and requests are refused. */
  async stop(): Promise<void> {
    const server = this.server
    this.server = null
    if (!server) return
    await new Promise<void>((resolve) => {
      server.close(() => resolve())
      // `close()` alone waits for open connections; a client mid-request must not hold the test up.
      server.closeAllConnections()
    })
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', this.url)
    const header = req.headers['x-pos-token']
    const token = Array.isArray(header) ? header[0] : (header ?? null)

    let payload: StandInRequest['payload'] = null
    if (req.method === 'POST') {
      const raw = await readBody(req)
      try {
        payload = JSON.parse(raw) as StandInRequest['payload']
      } catch {
        payload = null
      }
    }

    this.requests.push({
      method: req.method ?? '',
      path: url.pathname,
      stationId: url.searchParams.get('station_id'),
      token,
      payload
    })

    // pushcart-web treats an unset secret as "nobody is authorized", not as "anyone is".
    if (!this.secret || token !== this.secret) {
      respond(res, 401, { error: 'unauthorized' })
      return
    }

    if (url.pathname === '/api/pos/session' && req.method === 'GET') {
      const stationId = url.searchParams.get('station_id')
      if (!stationId) {
        respond(res, 400, { error: 'station_id required' })
        return
      }
      if (!this.stations.has(stationId)) {
        respond(res, 404, { error: 'unknown_station' })
        return
      }
      const session = this.stations.get(stationId) ?? null
      respond(res, 200, {
        message: session ? 'Successfully fetched session' : 'Unbound',
        data: session
      })
      return
    }

    if (url.pathname === '/api/pos/sync' && req.method === 'POST') {
      if (this.mode === 'server_error') {
        respond(res, 500, { error: 'stand-in failure' })
        return
      }
      const invalid = validate(payload)
      if (invalid !== null) {
        respond(res, 400, { error: invalid })
        return
      }
      if (this.mode === 'conflict') {
        respond(res, 409, { error: 'session_closed' })
        return
      }
      this.reconcile(res, payload as NonNullable<StandInRequest['payload']>)
      return
    }

    respond(res, 404, { error: 'not_found' })
  }

  /** `pos_reconcile`'s essence: find the open session, refuse if it is over, else diff the snapshot. */
  private reconcile(res: ServerResponse, payload: NonNullable<StandInRequest['payload']>): void {
    const open = [...this.stations.values()].find(
      (session) => session !== null && session.session_ref === payload.session_ref
    )
    if (!open) {
      respond(res, 409, { error: 'session_closed' })
      return
    }
    if (open.cart_status === 'paid') {
      respond(res, 409, { error: 'cart_paid' })
      return
    }

    const rows = this.items.get(open.cart_id) ?? new Map<string, number>()
    this.items.set(open.cart_id, rows)

    const results: Array<{ class_name: string; status: string; low_stock_warning: boolean }> = []
    const desired = new Set<string>()
    for (const item of payload.items) {
      desired.add(item.class_name)
      results.push({
        class_name: item.class_name,
        status: rows.has(item.class_name) ? 'updated' : 'added',
        low_stock_warning: false
      })
    }
    for (const className of rows.keys()) {
      if (!desired.has(className)) {
        results.push({ class_name: className, status: 'removed', low_stock_warning: false })
      }
    }

    // Full-snapshot reconcile: the rows become exactly the snapshot, which is what makes a repeated
    // sync a no-op instead of a second row.
    rows.clear()
    for (const item of payload.items) rows.set(item.class_name, item.quantity)

    const subtotal = [...rows].reduce((sum, [name, qty]) => sum + qty * (this.prices[name] ?? 0), 0)

    respond(res, 200, {
      message: 'Successfully synced cart',
      data: {
        results,
        cart_totals: { item_count: rows.size, subtotal }
      }
    })
  }
}
