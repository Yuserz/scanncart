// Persistent configuration for the POS integration (spec §5.1).
//
// `pos.json` lives in Electron's userData directory; the store takes the directory rather than
// importing `app`, so it is unit-testable against a temp dir. An empty base URL, secret or
// station id means the whole feature is disabled.

import { promises as fs } from 'fs'
import { join } from 'path'

import {
  CART_EDGES,
  DEFAULT_ZONE_PRESET,
  zonePresetProblems,
  type CartEdge
} from './transferGeometry'

/**
 * How the cart is derived. `counter` is the original rule (`cartState.ts`: what is visible,
 * with the posted count as a floor). `basket` posts the transfer ledger (`basketLedger.ts`):
 * a confirmed deposit adds, a confirmed removal subtracts, and hiding changes nothing. The
 * ledger runs in both modes, so `counter` is also the basket's shadow mode.
 */
export type CartMode = 'counter' | 'basket'
export const CART_MODES: readonly CartMode[] = ['counter', 'basket']

export interface PosConfig {
  /** Whichever host serves pushcart-web (`http://192.168.1.20:3000` or an https cloud host). */
  posBaseUrl: string
  /** Shared secret for the `x-pos-token` header; matches pushcart-web's POS_INGEST_SECRET. */
  posSecret: string
  /** The id of a row in pushcart-web's `stations` table. Configured, never generated. */
  stationId: string
  /** Seconds a track must persist before it commits an item. Must exceed the sidecar's track_expiry_s. */
  commitDwellS: number
  /** Seconds a lower count must hold before the cart drops an item. */
  removeSettleS: number
  /** Minimum highest-confidence for a track to be eligible. */
  minCommitConf: number
  /** Session poll interval while unbound (ms). */
  unboundPollMs: number
  /** Session poll interval while bound (ms). */
  sessionPollMs: number
  /** `/api/logs` poll interval while bound (ms). */
  logsPollMs: number
  /** Which derivation the cart follows; `counter` until the basket passes its rehearsal. */
  cartMode: CartMode
  /** The edge of the camera frame the cart is at; a deposit moves toward it. */
  cartEdge: CartEdge
  /** Share of the frame, from the cart edge, that is inside the cart (0–1). */
  insideFraction: number
  /** Share of the frame just past the inside band that is the opening (0–1). */
  openingFraction: number
}

export const POS_CONFIG_FILENAME = 'pos.json'

/** The sidecar's default `track_expiry_s`, used when the live value cannot be read. */
export const DEFAULT_TRACK_EXPIRY_S = 1.5

export const DEFAULT_POS_CONFIG: PosConfig = {
  posBaseUrl: '',
  posSecret: '',
  stationId: '',
  commitDwellS: 3,
  removeSettleS: 10,
  minCommitConf: 0.6,
  unboundPollMs: 1000,
  sessionPollMs: 5000,
  logsPollMs: 1000,
  cartMode: 'counter',
  cartEdge: DEFAULT_ZONE_PRESET.cartEdge,
  insideFraction: DEFAULT_ZONE_PRESET.insideFraction,
  openingFraction: DEFAULT_ZONE_PRESET.openingFraction
}

/** The feature is on only when all three connection fields are set. */
export function isPosEnabled(config: PosConfig): boolean {
  return Boolean(config.posBaseUrl && config.posSecret && config.stationId)
}

/**
 * §2.1's TLS rule: once the base URL leaves the LAN the secret header must travel over HTTPS.
 * Returns a message when it would not, or null when there is nothing to warn about.
 */
export function insecureBaseUrlWarning(baseUrl: string): string | null {
  if (!baseUrl) return null
  let url: URL
  try {
    url = new URL(baseUrl)
  } catch {
    return 'posBaseUrl is not a valid URL.'
  }
  if (url.protocol !== 'http:') return null
  const host = url.hostname.toLowerCase()
  if (host === 'localhost' || host === '127.0.0.1' || host === '::1' || host === '[::1]') {
    return null
  }
  return 'posBaseUrl is plain http:// and not localhost — the POS secret would travel in clear text. Use https://.'
}

export class PosConfigError extends Error {}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

/** Coerce a parsed object to a total PosConfig, falling back to defaults per field. */
function sanitize(raw: unknown): PosConfig {
  const source = (raw ?? {}) as Partial<Record<keyof PosConfig, unknown>>
  const out: PosConfig = { ...DEFAULT_POS_CONFIG }

  if (typeof source.posBaseUrl === 'string') out.posBaseUrl = source.posBaseUrl
  if (typeof source.posSecret === 'string') out.posSecret = source.posSecret
  if (typeof source.stationId === 'string') out.stationId = source.stationId
  if (CART_MODES.includes(source.cartMode as CartMode)) out.cartMode = source.cartMode as CartMode
  if (CART_EDGES.includes(source.cartEdge as CartEdge)) out.cartEdge = source.cartEdge as CartEdge

  const numeric: Array<[keyof PosConfig, (n: number) => boolean]> = [
    ['commitDwellS', (n) => n > 0],
    ['removeSettleS', (n) => n >= 0],
    ['minCommitConf', (n) => n >= 0 && n <= 1],
    ['unboundPollMs', (n) => n > 0],
    ['sessionPollMs', (n) => n > 0],
    ['logsPollMs', (n) => n > 0],
    ['insideFraction', (n) => n > 0 && n < 1],
    ['openingFraction', (n) => n > 0 && n < 1]
  ]
  for (const [key, valid] of numeric) {
    const value = source[key]
    if (isFiniteNumber(value) && valid(value)) {
      // Each key is a number-valued field in the interface; the cast keeps the loop single.
      ;(out[key] as number) = value
    }
  }
  if (zonePresetProblems(out).length > 0) {
    out.insideFraction = DEFAULT_ZONE_PRESET.insideFraction
    out.openingFraction = DEFAULT_ZONE_PRESET.openingFraction
  }

  return out
}

export class PosConfigStore {
  constructor(private readonly dir: string) {}

  private get path(): string {
    return join(this.dir, POS_CONFIG_FILENAME)
  }

  /** A missing or corrupt file degrades to defaults rather than crashing startup. */
  async load(): Promise<PosConfig> {
    try {
      const raw = await fs.readFile(this.path, 'utf8')
      return sanitize(JSON.parse(raw))
    } catch {
      return { ...DEFAULT_POS_CONFIG }
    }
  }

  /**
   * Merge a patch onto the stored config and persist it. Refuses a `commitDwellS` that does not
   * exceed the sidecar's `track_expiry_s` — the §5.2 invariant that keeps a track swap from
   * reading as two items.
   */
  async save(
    patch: Partial<PosConfig>,
    trackExpiryS: number = DEFAULT_TRACK_EXPIRY_S
  ): Promise<PosConfig> {
    const current = await this.load()
    const next = sanitize({ ...current, ...patch })

    if (next.commitDwellS <= trackExpiryS) {
      throw new PosConfigError(
        `commitDwellS (${next.commitDwellS}s) must exceed the sidecar's track_expiry_s (${trackExpiryS}s).`
      )
    }

    // Judged on the merge *before* sanitizing: `sanitize` quietly restores default bands for a
    // stored file whose pair leaves no outside band, which is right for a read and wrong for a
    // save — an operator who typed that pair has to be told, not overruled.
    const asked = { ...current, ...patch }
    const zoneProblems = zonePresetProblems({
      cartEdge: next.cartEdge,
      insideFraction: Number(asked.insideFraction),
      openingFraction: Number(asked.openingFraction)
    })
    if (zoneProblems.length > 0) throw new PosConfigError(`Zones: ${zoneProblems.join('; ')}.`)

    await fs.mkdir(this.dir, { recursive: true })
    await fs.writeFile(this.path, `${JSON.stringify(next, null, 2)}\n`, 'utf8')
    return next
  }
}
