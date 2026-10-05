import { useEffect, useMemo, useState, type JSX } from 'react'
import type { CartMode, PosConfig } from '../../../main/posConfig'
import type { PosConnectionResult } from '../../../main/posSession'
import { usePosState, type PosStateDeps } from '../hooks/usePosState'
import { zoneSummary } from '../lib/zones'
import './PosAdminSection.css'

// The Admin Panel's self-checkout section (spec §5.5). The config and the connection test both live
// in the main process: the renderer cannot reach the pushcart-web host itself, and the secret must
// not travel into a page beyond this form. Empty URL/secret/station id means the feature is off,
// which is a state rather than an error.
//
// Laid out as the operator works through it: is it on and connected (the header), how it connects,
// what the cart follows, how strict the counting is, and - folded away - the polling a person
// rarely needs. The zones are *summarised* here and edited on the Basket test tab, which draws them
// over the live picture; editing bands here as well used to leave two editors for one setting, and
// only one of them could show (or set) a drawn outline.

interface NumberField {
  key: keyof PosConfig
  label: string
  unit: string
  hint: string
  step: number
  min: number
  max?: number
}

/** How strict the counter-mode count is: the three numbers an operator might tune. */
const COUNTING_FIELDS: NumberField[] = [
  {
    key: 'commitDwellS',
    label: 'Time before an item is added',
    unit: 's',
    hint: 'How long an item must stay in view before it joins the cart. Must be longer than the sidecar’s track expiry.',
    step: 0.5,
    min: 0.5
  },
  {
    key: 'removeSettleS',
    label: 'Time before an item is removed',
    unit: 's',
    hint: 'How long a lower count must hold before the cart drops an item. Automatic removal is suspended, so this only smooths over camera dropouts.',
    step: 0.5,
    min: 0
  },
  {
    key: 'minCommitConf',
    label: 'Minimum confidence',
    unit: '0–1',
    hint: 'A track must reach this confidence at least once to count.',
    step: 0.05,
    min: 0,
    max: 1
  }
]

/** Polling cadence: correct by default, folded under Advanced. */
const POLLING_FIELDS: NumberField[] = [
  {
    key: 'unboundPollMs',
    label: 'Check for a new customer every',
    unit: 'ms',
    hint: 'While no session is open.',
    step: 100,
    min: 100
  },
  {
    key: 'sessionPollMs',
    label: 'Check the open session every',
    unit: 'ms',
    hint: 'While a customer is shopping; this is how fast Finish is noticed.',
    step: 500,
    min: 500
  },
  {
    key: 'logsPollMs',
    label: 'Read the camera’s tracks every',
    unit: 'ms',
    hint: 'While a customer is shopping (counter mode).',
    step: 100,
    min: 100
  }
]

const NUMBER_FIELDS = [...COUNTING_FIELDS, ...POLLING_FIELDS]

const CART_MODES: Array<{ mode: CartMode; title: string; body: string }> = [
  {
    mode: 'counter',
    title: 'Counter',
    body: 'The cart is what the camera sees, with each posted count kept as a floor. The basket readout runs beside it as a shadow for comparison.'
  },
  {
    mode: 'basket',
    title: 'Basket',
    body: 'The cart is what the camera confirmed: an item crossing into the basket adds it, crossing out removes it, and hiding it changes nothing.'
  }
]

// The same optional read `usePosState` does, for the same reason: the bridge is additive across the
// process boundary, so a window whose preload predates the POS methods must report that it could not
// load the configuration rather than taking the Admin Panel down. Every action stays disabled while
// the config is null, so nothing tries to test or save a config that was never read.
type PosConfigBridge = {
  getPosConfig?: () => Promise<PosConfig>
  savePosConfig?: (patch: Partial<PosConfig>) => Promise<PosConfig>
  testPosConnection?: () => Promise<PosConnectionResult>
}

function bridge(): PosConfigBridge {
  // `window.api` itself is absent on a window whose preload never ran — which is what a bare test
  // harness and the layout harness both are — so the whole read is optional, not just the method.
  return (window.api ?? {}) as PosConfigBridge
}

function errorText(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)
  // Electron prefixes a rejected invoke with its own wrapper; the main process's sentence is the rest.
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '')
}

interface Draft {
  baseUrl: string
  secret: string
  stationId: string
  cartMode: CartMode
  numbers: Record<string, string>
}

function draftOf(config: PosConfig): Draft {
  const numbers: Record<string, string> = {}
  for (const field of NUMBER_FIELDS) numbers[field.key] = String(config[field.key])
  return {
    baseUrl: config.posBaseUrl,
    secret: config.posSecret,
    stationId: config.stationId,
    cartMode: config.cartMode ?? 'counter',
    numbers
  }
}

type Tone = 'off' | 'ok' | 'live' | 'warn' | 'bad'

export interface PosAdminSectionProps {
  /** Injectable for tests; defaults to the preload bridge, like the Live view's POS panel. */
  posStateDeps?: PosStateDeps
}

export function PosAdminSection({ posStateDeps }: PosAdminSectionProps = {}): JSX.Element {
  const [config, setConfig] = useState<PosConfig | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [notice, setNotice] = useState<{ tone: 'ok' | 'bad'; text: string } | null>(null)
  const [result, setResult] = useState<PosConnectionResult | null>(null)
  const [busy, setBusy] = useState<'save' | 'test' | null>(null)
  const [showSecret, setShowSecret] = useState(false)
  const live = usePosState(posStateDeps)

  useEffect(() => {
    const read = bridge().getPosConfig?.() ?? Promise.resolve(null)
    void read
      .then((loaded) => {
        if (!loaded) {
          setLoadError('Could not load the POS configuration.')
          return
        }
        setConfig(loaded)
        setDraft(draftOf(loaded))
      })
      .catch(() => setLoadError('Could not load the POS configuration.'))
  }, [])

  const dirty = useMemo(
    () =>
      config !== null &&
      draft !== null &&
      JSON.stringify(draft) !== JSON.stringify(draftOf(config)),
    [config, draft]
  )

  const enabled = !!config && !!config.posBaseUrl && !!config.posSecret && !!config.stationId

  // The header's status: the saved configuration decides on/off, the running loop decides the rest.
  const status: { tone: Tone; label: string; detail?: string } = !config
    ? { tone: 'off', label: loadError ? 'Unavailable' : 'Loading…' }
    : !enabled
      ? {
          tone: 'off',
          label: 'Off',
          detail: 'Add the server address, secret and station to turn it on.'
        }
      : live?.phase === 'error'
        ? { tone: 'bad', label: 'Error', detail: live.error ?? undefined }
        : live?.phase === 'bound' || live?.phase === 'warming_up'
          ? {
              tone: 'live',
              label:
                live.phase === 'warming_up'
                  ? 'Customer session · warming up'
                  : 'Customer session open',
              detail: live.cartCode ? `Cart ${live.cartCode.slice(0, 8)}` : undefined
            }
          : live?.phase === 'unbound' && live.lastContactMs !== null
            ? {
                tone: 'ok',
                label: 'Ready',
                detail: 'Waiting for a customer to tap Start on the tablet.'
              }
            : // No session, but pushcart-web has not answered yet either: not "Ready" until it has.
              {
                tone: 'warn',
                label: 'Connecting…',
                detail: config.posBaseUrl ? `Reaching ${config.posBaseUrl}` : undefined
              }

  // The dwell rule, checked against the sidecar's live track expiry once a test has read it.
  const dwell = Number(draft?.numbers.commitDwellS)
  const expiry = result?.trackExpiryS ?? null
  const dwellTooShort = expiry !== null && Number.isFinite(dwell) && dwell <= expiry

  const setField = (patch: Partial<Draft>): void => setDraft((d) => (d ? { ...d, ...patch } : d))
  const setNumber = (key: string, value: string): void =>
    setDraft((d) => (d ? { ...d, numbers: { ...d.numbers, [key]: value } } : d))

  const onSave = (): void => {
    const save = bridge().savePosConfig
    if (!config || !draft || !save) return
    setBusy('save')
    setNotice(null)
    // The zone fields are deliberately absent: they are the Basket test tab's to change, and a save
    // here must not write back whatever this form happened to load.
    const patch: Partial<PosConfig> = {
      posBaseUrl: draft.baseUrl.trim(),
      posSecret: draft.secret.trim(),
      stationId: draft.stationId.trim(),
      cartMode: draft.cartMode
    }
    for (const field of NUMBER_FIELDS) {
      const parsed = Number(draft.numbers[field.key])
      ;(patch[field.key] as number) = Number.isFinite(parsed)
        ? parsed
        : (config[field.key] as number)
    }
    void save(patch)
      .then((saved) => {
        setConfig(saved)
        setDraft(draftOf(saved))
        const on = !!saved.posBaseUrl && !!saved.posSecret && !!saved.stationId
        setNotice({
          tone: 'ok',
          text: on
            ? 'Saved.'
            : 'Saved. Self-checkout is off until the address, secret and station are all set.'
        })
      })
      .catch((error: unknown) => setNotice({ tone: 'bad', text: errorText(error) }))
      .finally(() => setBusy(null))
  }

  const onTest = (): void => {
    const test = bridge().testPosConnection
    if (!test) return
    setBusy('test')
    setResult(null)
    void test()
      .then((r) => setResult(r))
      .catch((error: unknown) =>
        setResult({
          ok: false,
          status: null,
          message: errorText(error),
          cartCode: null,
          sessionRef: null,
          warning: null,
          trackExpiryS: null
        })
      )
      .finally(() => setBusy(null))
  }

  const numberInput = (field: NumberField): JSX.Element => (
    <label className="pos-field" key={field.key}>
      <span className="pos-field-label">{field.label}</span>
      <span className="pos-input-unit">
        <input
          type="number"
          id={`pos-${field.key}`}
          data-testid={`pos-${field.key}`}
          step={field.step}
          min={field.min}
          max={field.max}
          value={draft?.numbers[field.key] ?? ''}
          onChange={(e) => setNumber(field.key, e.target.value)}
        />
        <span className="pos-unit">{field.unit}</span>
      </span>
      <span className="pos-field-hint">{field.hint}</span>
    </label>
  )

  return (
    <section className="pos-admin" data-testid="pos-admin" aria-labelledby="pos-admin-h">
      <header className="pos-admin-head">
        <div className="pos-head-row">
          <h4 id="pos-admin-h">Self-checkout</h4>
          <div className="pos-status" data-testid="pos-status" data-tone={status.tone}>
            <span className="pos-dot" aria-hidden="true" />
            <span>
              <b>{status.label}</b>
              {status.detail && <small>{status.detail}</small>}
            </span>
          </div>
        </div>
        <p className="pos-sub">
          Links this station to pushcart-web. The customer taps Start on the tablet; this desktop
          follows that session and fills the cart from the camera.
        </p>
      </header>

      {loadError && <p className="pos-notice bad">{loadError}</p>}

      <div className="pos-card">
        <h5>Connection</h5>
        <div className="pos-grid">
          <label className="pos-field wide">
            <span className="pos-field-label">pushcart-web address</span>
            <input
              type="text"
              id="pos-base-url"
              data-testid="pos-base-url"
              value={draft?.baseUrl ?? ''}
              onChange={(e) => setField({ baseUrl: e.target.value })}
              placeholder="https://pushcart.example.com or http://192.168.1.20:3000"
              spellCheck={false}
            />
          </label>
          <label className="pos-field">
            <span className="pos-field-label">Shared secret</span>
            <span className="pos-input-unit">
              <input
                type={showSecret ? 'text' : 'password'}
                id="pos-secret"
                data-testid="pos-secret"
                value={draft?.secret ?? ''}
                onChange={(e) => setField({ secret: e.target.value })}
                spellCheck={false}
                autoComplete="off"
              />
              <button
                type="button"
                className="pos-ghost"
                aria-pressed={showSecret}
                onClick={() => setShowSecret((s) => !s)}
              >
                {showSecret ? 'Hide' : 'Show'}
              </button>
            </span>
            <span className="pos-field-hint">POS_INGEST_SECRET from pushcart-web’s settings.</span>
          </label>
          <label className="pos-field">
            <span className="pos-field-label">Station</span>
            <input
              type="text"
              id="pos-station-id"
              data-testid="pos-station-id"
              value={draft?.stationId ?? ''}
              onChange={(e) => setField({ stationId: e.target.value })}
              spellCheck={false}
            />
            <span className="pos-field-hint">The station id from pushcart-web’s station list.</span>
          </label>
        </div>
        <div className="pos-row">
          <button
            type="button"
            className="pos-btn"
            onClick={onTest}
            disabled={busy !== null || !config}
            title={dirty ? 'Tests the saved settings; save your changes first.' : undefined}
          >
            {busy === 'test' ? 'Testing…' : 'Test connection'}
          </button>
          {dirty && <span className="pos-dim">Tests the saved settings — save first.</span>}
        </div>
        {result && (
          <div
            className={`pos-result ${result.ok ? 'ok' : 'bad'}`}
            data-testid="pos-test-result"
            role="status"
          >
            <p className="pos-result-head">
              <span aria-hidden="true">{result.ok ? '✓' : '✕'}</span> {result.message}
              {result.status !== null && <span className="pos-dim"> · HTTP {result.status}</span>}
            </p>
            {(result.cartCode || result.trackExpiryS !== null) && (
              <p className="pos-dim">
                {result.cartCode && <>Open cart {result.cartCode.slice(0, 8)}. </>}
                {result.trackExpiryS !== null && <>Sidecar track expiry {result.trackExpiryS} s.</>}
              </p>
            )}
            {result.warning && (
              <p className="pos-warn" data-testid="pos-tls-warning">
                {result.warning}
              </p>
            )}
          </div>
        )}
      </div>

      <div className="pos-card">
        <h5>What the cart follows</h5>
        <div className="pos-modes" role="radiogroup" aria-label="Cart mode">
          {CART_MODES.map(({ mode, title, body }) => (
            <label
              key={mode}
              className={`pos-mode${draft?.cartMode === mode ? ' selected' : ''}`}
              data-testid={`pos-cartMode-${mode}`}
            >
              <input
                type="radio"
                name="pos-cart-mode"
                id={`pos-cart-mode-${mode}`}
                value={mode}
                checked={draft?.cartMode === mode}
                onChange={() => setField({ cartMode: mode })}
                disabled={!draft}
              />
              <span>
                <b>{title}</b>
                <small>{body}</small>
              </span>
            </label>
          ))}
        </div>
        <p className="pos-zones" data-testid="pos-zones">
          <span className="pos-dim">Zones:</span> {config ? zoneSummary(config) : '…'}
          <span className="pos-dim"> — set and draw them on the Basket test tab.</span>
        </p>
      </div>

      <div className="pos-card">
        <h5>Counting rules</h5>
        <div className="pos-grid">{COUNTING_FIELDS.map(numberInput)}</div>
        {dwellTooShort && (
          <p className="pos-warn" data-testid="pos-dwell-warning">
            Time before an item is added ({dwell} s) must be longer than the sidecar’s track expiry
            ({expiry} s), or one item can be counted twice. Save will refuse it.
          </p>
        )}
        <details className="pos-advanced">
          <summary>Advanced: polling</summary>
          <div className="pos-grid">{POLLING_FIELDS.map(numberInput)}</div>
        </details>
      </div>

      <footer className="pos-foot">
        <button
          type="button"
          className="pos-btn primary"
          onClick={onSave}
          disabled={busy !== null || !config || !dirty}
        >
          {busy === 'save' ? 'Saving…' : 'Save self-checkout settings'}
        </button>
        <button
          type="button"
          className="pos-btn"
          onClick={() => {
            if (config) setDraft(draftOf(config))
            setNotice(null)
          }}
          disabled={busy !== null || !dirty}
        >
          Discard
        </button>
        {dirty && !notice && <span className="pos-dim">Unsaved self-checkout changes</span>}
        {notice && (
          <p className={`pos-notice ${notice.tone}`} data-testid="pos-save-message" role="status">
            {notice.text}
          </p>
        )}
      </footer>
    </section>
  )
}
