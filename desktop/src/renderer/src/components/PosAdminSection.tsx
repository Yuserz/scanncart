import { useEffect, useState, type JSX } from 'react'
import type { CartMode, PosConfig } from '../../../main/posConfig'
import type { CartEdge } from '../../../main/transferGeometry'
import type { PosConnectionResult } from '../../../main/posSession'

interface NumericField {
  key: keyof PosConfig
  label: string
  step: number
}

const NUMERIC_FIELDS: NumericField[] = [
  { key: 'commitDwellS', label: 'Commit dwell (s)', step: 0.5 },
  { key: 'removeSettleS', label: 'Remove settle (s)', step: 0.5 },
  { key: 'minCommitConf', label: 'Min commit confidence', step: 0.05 },
  { key: 'unboundPollMs', label: 'Unbound poll (ms)', step: 100 },
  { key: 'sessionPollMs', label: 'Session poll (ms)', step: 500 },
  { key: 'logsPollMs', label: 'Logs poll (ms)', step: 100 },
  { key: 'insideFraction', label: 'Cart band (share of frame)', step: 0.05 },
  { key: 'openingFraction', label: 'Opening band (share of frame)', step: 0.05 }
]

const CART_MODE_LABELS: Record<CartMode, string> = {
  counter: 'Counter — what is visible (basket shown as shadow)',
  basket: 'Basket — confirmed deposits and removals'
}

const CART_EDGE_LABELS: Record<CartEdge, string> = {
  bottom: 'Bottom of the frame',
  top: 'Top of the frame',
  left: 'Left of the frame (camera’s left)',
  right: 'Right of the frame (camera’s right)'
}

function numberOr(value: string, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

// The same optional read `usePosState` does, for the same reason: the bridge is additive across the
// process boundary, so a window whose preload predates the POS methods must report that it could not
// load the configuration rather than taking the Admin Panel down. Both buttons stay disabled while
// it is null, so nothing tries to test or save a config that was never read.
type PosConfigBridge = { getPosConfig?: () => Promise<PosConfig> }

function loadConfig(): Promise<PosConfig | null> {
  // `window.api` itself is absent on a window whose preload never ran — which is what a bare test
  // harness and the layout harness both are — so the whole read is optional, not just the method.
  const bridge = (window.api ?? {}) as PosConfigBridge
  return bridge.getPosConfig?.() ?? Promise.resolve(null)
}

// The Admin Panel's POS integration section (spec §5.5). The config and the connection test both
// live in the main process: the renderer cannot reach the pushcart-web host itself. Empty
// URL/secret/station id means the feature is off, which is a state rather than an error.
export function PosAdminSection(): JSX.Element {
  const [config, setConfig] = useState<PosConfig | null>(null)
  const [baseUrl, setBaseUrl] = useState('')
  const [secret, setSecret] = useState('')
  const [stationId, setStationId] = useState('')
  const [numbers, setNumbers] = useState<Record<string, string>>({})
  const [cartMode, setCartMode] = useState<CartMode>('counter')
  const [cartEdge, setCartEdge] = useState<CartEdge>('bottom')
  const [message, setMessage] = useState<string | null>(null)
  const [result, setResult] = useState<PosConnectionResult | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void loadConfig()
      .then((loaded) => {
        if (loaded === null) {
          setMessage('Could not load the POS configuration.')
          return
        }
        setConfig(loaded)
        setBaseUrl(loaded.posBaseUrl)
        setSecret(loaded.posSecret)
        setStationId(loaded.stationId)
        setCartMode(loaded.cartMode ?? 'counter')
        setCartEdge(loaded.cartEdge ?? 'bottom')
        const draft: Record<string, string> = {}
        for (const field of NUMERIC_FIELDS) draft[field.key] = String(loaded[field.key])
        setNumbers(draft)
      })
      .catch(() => setMessage('Could not load the POS configuration.'))
  }, [])

  const onSave = (): void => {
    if (!config) return
    setBusy(true)
    setMessage(null)
    const patch: Partial<PosConfig> = {
      posBaseUrl: baseUrl.trim(),
      posSecret: secret.trim(),
      stationId: stationId.trim(),
      cartMode,
      cartEdge
    }
    for (const field of NUMERIC_FIELDS) {
      ;(patch[field.key] as number) = numberOr(
        numbers[field.key] ?? '',
        config[field.key] as number
      )
    }
    void window.api
      .savePosConfig(patch)
      .then((saved) => {
        setConfig(saved)
        setMessage('Saved. Empty URL, secret or station id disables the integration.')
      })
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : String(error))
      })
      .finally(() => setBusy(false))
  }

  const onTest = (): void => {
    setBusy(true)
    setResult(null)
    setMessage(null)
    void window.api
      .testPosConnection()
      .then((r) => setResult(r))
      .catch((error: unknown) =>
        setResult({
          ok: false,
          status: null,
          message: error instanceof Error ? error.message : String(error),
          cartCode: null,
          sessionRef: null,
          warning: null,
          trackExpiryS: null
        })
      )
      .finally(() => setBusy(false))
  }

  return (
    <section className="admin-backend" data-testid="pos-admin">
      <h4>Self-checkout (POS integration)</h4>
      <p className="field-hint">
        The tablet opens the session; the desktop binds to it. Leave the URL, secret or station id
        empty to turn the feature off.
      </p>
      <p className="field-hint">
        Basket mode adds an item when it crosses from outside, through the opening band, into the
        cart band and rests there for a second, and removes it on the reverse path. Hiding an item
        never removes it. Keep Counter until the basket readout on the Live view matches the real
        basket over a rehearsal.
      </p>

      <div className="admin-grid">
        <label>
          <span>pushcart-web base URL</span>
          <input
            type="text"
            data-testid="pos-base-url"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="http://192.168.1.20:3000"
          />
        </label>
        <label>
          <span>POS secret</span>
          <input
            type="password"
            data-testid="pos-secret"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
          />
        </label>
        <label>
          <span>Station id (from pushcart-web)</span>
          <input
            type="text"
            data-testid="pos-station-id"
            value={stationId}
            onChange={(e) => setStationId(e.target.value)}
          />
        </label>
        <label>
          <span>Cart mode</span>
          <select
            data-testid="pos-cartMode"
            value={cartMode}
            onChange={(e) => setCartMode(e.target.value as CartMode)}
          >
            {(Object.keys(CART_MODE_LABELS) as CartMode[]).map((m) => (
              <option key={m} value={m}>
                {CART_MODE_LABELS[m]}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Cart is at (deposit moves toward it)</span>
          <select
            data-testid="pos-cartEdge"
            value={cartEdge}
            onChange={(e) => setCartEdge(e.target.value as CartEdge)}
          >
            {(Object.keys(CART_EDGE_LABELS) as CartEdge[]).map((edge) => (
              <option key={edge} value={edge}>
                {CART_EDGE_LABELS[edge]}
              </option>
            ))}
          </select>
        </label>
        {NUMERIC_FIELDS.map((field) => (
          <label key={field.key}>
            <span>{field.label}</span>
            <input
              type="number"
              step={field.step}
              data-testid={`pos-${field.key}`}
              value={numbers[field.key] ?? ''}
              onChange={(e) => setNumbers((prev) => ({ ...prev, [field.key]: e.target.value }))}
            />
          </label>
        ))}
      </div>

      <div className="admin-actions">
        <button type="button" onClick={onSave} disabled={busy || !config}>
          Save POS settings
        </button>
        <button type="button" onClick={onTest} disabled={busy || !config}>
          Test connection
        </button>
      </div>

      {message && (
        <p className="admin-warning" data-testid="pos-save-message">
          {message}
        </p>
      )}

      {result && (
        <div data-testid="pos-test-result">
          <p className={result.ok ? 'field-hint' : 'admin-warning'}>
            {result.ok ? '✓ ' : '✕ '}
            {result.message}
            {result.status !== null ? ` (${result.status})` : ''}
          </p>
          {result.cartCode && <p className="field-hint">Cart {result.cartCode.slice(0, 8)}</p>}
          {result.warning && (
            <p className="admin-warning" data-testid="pos-tls-warning">
              {result.warning}
            </p>
          )}
        </div>
      )}
    </section>
  )
}
