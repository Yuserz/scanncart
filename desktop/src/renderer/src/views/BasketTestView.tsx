// The basket test screen: try deposits and removals at a desk, with any webcam and no tablet.
//
// Three things the Live view cannot do, which is why this is its own screen rather than another
// card in Live's fixed-height rail (whose one-screen promise `verify-live-layout` measures):
// - **Practice.** The basket ledger only counts once a customer session binds it, and a desk has no
//   pushcart-web. A practice run binds the ledger locally, so a real deposit counts with nothing
//   else running; a customer session binding later simply replaces it.
// - **See the zones.** The zones are drawn over the live preview, mirrored the way the preview is,
//   so "is my basket in the inside band" is answered by looking rather than by arithmetic.
// - **Choose and draw them.** Bands (spot A: the camera under the handle, looking across the
//   basket - the default) or drawn outlines (spot B: a camera looking down into it). A drawn
//   outline is placed by clicking its corners on the picture; every point shows as a marker the
//   moment it is placed and can be dragged afterwards, with Undo/Redo over every zone edit. Edits
//   show on the preview before they are saved, and Save goes through the same POS config the
//   Admin Panel writes, so the basket judges under exactly what is on screen.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type JSX,
  type MouseEvent,
  type PointerEvent as ReactPointerEvent
} from 'react'
import { useSidecarStream, type StreamDeps } from '../hooks/useSidecarStream'
import { basketBridge, useBasketState, type BasketStateDeps } from '../hooks/useBasketState'
import {
  clickPoint,
  commit,
  fromScreen,
  historyOf,
  layoutOutlines,
  moveCorner,
  redo,
  svgPoints,
  toScreen,
  undo,
  type History,
  type OutlineName
} from '../lib/zones'
import {
  CART_EDGES,
  drawnZoneProblems,
  zonePresetProblems,
  type CartEdge,
  type Point,
  type ZoneLayout,
  type ZoneMode
} from '../../../main/transferGeometry'
import type { PosConfig } from '../../../main/posConfig'
import './BasketTestView.css'

/** The zone fields of the POS config: what this screen edits. */
type ZoneDraft = Pick<
  PosConfig,
  'zoneMode' | 'cartEdge' | 'insideFraction' | 'openingFraction' | 'drawnInside' | 'drawnOpening'
>

export interface ZoneConfigBridge {
  getPosConfig?: () => Promise<PosConfig>
  savePosConfig?: (patch: Partial<PosConfig>) => Promise<PosConfig>
}

export interface BasketTestViewProps {
  port: number
  // Injectable for tests; the defaults are the live stream and the preload bridge.
  streamDeps?: StreamDeps
  basketDeps?: BasketStateDeps
  bridge?: ReturnType<typeof basketBridge> & ZoneConfigBridge
}

const EDGE_LABELS: Record<CartEdge, string> = {
  bottom: 'Bottom of the picture',
  top: 'Top of the picture',
  left: 'Left side',
  right: 'Right side'
}

function draftOf(config: PosConfig): ZoneDraft {
  return {
    zoneMode: config.zoneMode,
    cartEdge: config.cartEdge,
    insideFraction: config.insideFraction,
    openingFraction: config.openingFraction,
    drawnInside: config.drawnInside,
    drawnOpening: config.drawnOpening
  }
}

function layoutOfDraft(d: ZoneDraft): ZoneLayout {
  return d.zoneMode === 'drawn'
    ? { mode: 'drawn', inside: d.drawnInside, opening: d.drawnOpening }
    : {
        mode: 'bands',
        cartEdge: d.cartEdge,
        insideFraction: d.insideFraction,
        openingFraction: d.openingFraction
      }
}

const OUTLINE_KEY: Record<OutlineName, 'drawnInside' | 'drawnOpening'> = {
  inside: 'drawnInside',
  opening: 'drawnOpening'
}

const OUTLINE_LABEL: Record<OutlineName, string> = {
  inside: 'Inside of the basket',
  opening: 'Opening (rim)'
}

/** A point being dragged, and the draft as it was when the drag began (one undo step). */
interface Drag {
  which: OutlineName
  index: number
  start: ZoneDraft
}

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA')
}

function draftProblems(d: ZoneDraft): string[] {
  return d.zoneMode === 'drawn'
    ? drawnZoneProblems({ inside: d.drawnInside, opening: d.drawnOpening })
    : zonePresetProblems(d)
}

export function BasketTestView({
  port,
  streamDeps,
  basketDeps,
  bridge: bridgeProp
}: BasketTestViewProps): JSX.Element {
  const stream = useSidecarStream(port, streamDeps)
  const basket = useBasketState(basketDeps)
  const bridge = (bridgeProp ?? window.api ?? {}) as ReturnType<typeof basketBridge> &
    ZoneConfigBridge

  const [saved, setSaved] = useState<ZoneDraft | null>(null)
  // The zone draft with its undo/redo history. Every discrete edit goes through `change`; a point
  // drag updates the draft live and becomes one history step when it ends.
  const [hist, setHist] = useState<History<ZoneDraft> | null>(null)
  const draft = hist?.present ?? null
  const [drawing, setDrawing] = useState<OutlineName | null>(null)
  const dragRef = useRef<Drag | null>(null)

  const change = useCallback((next: ZoneDraft): void => {
    setHist((h) => (h ? commit(h, next) : historyOf(next)))
  }, [])
  const doUndo = useCallback((): void => setHist((h) => (h ? undo(h) : h)), [])
  const doRedo = useCallback((): void => setHist((h) => (h ? redo(h) : h)), [])

  // Ctrl+Z / Ctrl+Y (and Ctrl+Shift+Z), except while typing in a field, where the browser's own
  // undo for that text is what the operator means.
  useEffect(() => {
    function onKey(e: KeyboardEvent): void {
      if (!(e.ctrlKey || e.metaKey) || isTyping(e.target)) return
      const key = e.key.toLowerCase()
      if (key === 'z' && !e.shiftKey) {
        e.preventDefault()
        doUndo()
      } else if (key === 'y' || (key === 'z' && e.shiftKey)) {
        e.preventDefault()
        doRedo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [doUndo, doRedo])
  const [zoneNote, setZoneNote] = useState<string | null>(null)
  const [practiceNote, setPracticeNote] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const imgRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    let active = true
    void bridge
      .getPosConfig?.()
      .then((config) => {
        if (!active || !config) return
        setSaved(draftOf(config))
        setHist(historyOf(draftOf(config)))
      })
      .catch(() => setZoneNote('The zone settings could not be read.'))
    return () => {
      active = false
    }
    // The bridge is read once per mount; a new bridge object every render must not refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const mirrored = stream.frame?.mirrored === true
  // The preview shows the draft, so an edit is visible before it is saved. Before the config has
  // loaded, the layout the basket is actually using stands in.
  const shownLayout: ZoneLayout | null = draft ? layoutOfDraft(draft) : (basket?.layout ?? null)
  const outlines = useMemo(
    () => (shownLayout ? layoutOutlines(shownLayout) : { inside: [], opening: [] }),
    [shownLayout]
  )
  const dirty = draft !== null && saved !== null && JSON.stringify(draft) !== JSON.stringify(saved)
  const problems = draft ? draftProblems(draft) : []

  const capturing = stream.statusState === 'running'

  /** A pointer position on the picture, in true (unmirrored) orientation. */
  function pointerPoint(e: { clientX: number; clientY: number }): Point | null {
    if (!imgRef.current) return null
    const p = clickPoint(e.clientX, e.clientY, imgRef.current.getBoundingClientRect())
    return p ? fromScreen(p, mirrored) : null
  }

  function onPreviewClick(event: MouseEvent<HTMLDivElement>): void {
    if (!drawing || !draft) return
    const point = pointerPoint(event)
    if (!point) return
    const key = OUTLINE_KEY[drawing]
    change({ ...draft, [key]: [...draft[key], point] })
  }

  function beginDrag(e: ReactPointerEvent<HTMLElement>, which: OutlineName, index: number): void {
    if (!draft) return
    e.preventDefault()
    e.stopPropagation()
    e.currentTarget.setPointerCapture?.(e.pointerId)
    dragRef.current = { which, index, start: draft }
  }

  function dragTo(e: ReactPointerEvent<HTMLElement>): void {
    const drag = dragRef.current
    if (!drag) return
    const p = pointerPoint(e)
    if (!p) return
    const key = OUTLINE_KEY[drag.which]
    const moved = moveCorner(drag.start[key], drag.index, p)
    // Live, without a history step per pixel: the whole drag is one step when it ends.
    setHist((h) => (h ? { ...h, present: { ...h.present, [key]: moved } } : h))
  }

  function endDrag(): void {
    const drag = dragRef.current
    if (!drag) return
    dragRef.current = null
    setHist((h) => (h ? commit(h, h.present, drag.start) : h))
  }

  function deleteOutline(which: OutlineName): void {
    if (!draft) return
    if (drawing === which) setDrawing(null)
    change({ ...draft, [OUTLINE_KEY[which]]: [] })
  }

  async function saveZones(): Promise<void> {
    if (!draft || !bridge.savePosConfig) return
    setBusy(true)
    setZoneNote(null)
    try {
      const next = await bridge.savePosConfig(draft)
      setSaved(draftOf(next))
      // Saving is not an edit: what is on screen stays, and so does the way back through it.
      setHist((h) => (h ? { ...h, present: draftOf(next) } : historyOf(draftOf(next))))
      setDrawing(null)
      setZoneNote('Zones saved. The basket now judges under them.')
    } catch (error) {
      // The main process names the problem (`Zones: …` / `Drawn zones: …`); show it as it is.
      const message = error instanceof Error ? error.message : String(error)
      setZoneNote(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''))
    } finally {
      setBusy(false)
    }
  }

  async function startPractice(): Promise<void> {
    setPracticeNote(null)
    try {
      await bridge.startBasketPractice?.()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setPracticeNote(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''))
    }
  }

  const readout = basket?.readout ?? null

  return (
    <div className="basket-test" data-testid="basket-test">
      <div className="bt-stage">
        <div className="bt-toolbar">
          <span className={`status-dot ${capturing ? 'running' : ''}`} aria-hidden="true" />
          <span className="bt-state">{capturing ? 'camera running' : stream.statusState}</span>
          {capturing ? (
            <button data-testid="bt-stop" onClick={() => void stream.stop()}>
              Stop camera
            </button>
          ) : (
            <button data-testid="bt-start" className="primary" onClick={() => void stream.start()}>
              Start camera
            </button>
          )}
          {drawing && (
            <span className="bt-drawing" data-testid="bt-drawing">
              Click the corners of the{' '}
              {drawing === 'inside' ? 'inside of the basket' : 'opening (rim)'} on the picture; drag
              a point to move it
            </span>
          )}
        </div>

        {/* The sidecar's own sentence for a capture that died, as the Live view shows it. Without
            it this screen showed only "error" over a frozen last frame, which reads as a camera
            that is struggling to connect rather than one that stopped for a stated reason. */}
        {stream.error && (
          <div className="bt-banner" role="alert" data-testid="bt-error">
            <span>{stream.error}</span>
            <button onClick={stream.clearError}>Dismiss</button>
          </div>
        )}

        <div
          className={`bt-preview ${drawing ? 'drawing' : ''}`}
          data-testid="bt-preview"
          onClick={onPreviewClick}
          onPointerMove={dragTo}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          {stream.frame ? (
            <img
              ref={imgRef}
              className="bt-img"
              alt="live preview with basket zones"
              src={`data:image/jpeg;base64,${stream.frame.jpeg}`}
            />
          ) : (
            <div className="bt-placeholder">
              {capturing
                ? 'Waiting for frames…'
                : 'Start the camera to see the zones on the picture.'}
            </div>
          )}
          {stream.frame && (
            <svg
              className="bt-overlay"
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
              data-testid="bt-overlay"
              aria-hidden="true"
            >
              {outlines.opening.length >= 2 && (
                <polygon
                  className="z-opening"
                  points={svgPoints(outlines.opening, mirrored)}
                  data-testid="zone-opening"
                />
              )}
              {outlines.inside.length >= 2 && (
                <polygon
                  className="z-inside"
                  points={svgPoints(outlines.inside, mirrored)}
                  data-testid="zone-inside"
                />
              )}
              {stream.frame.detections.map((d, i) => (
                <rect
                  key={`${d.track_id ?? 'x'}-${i}`}
                  className="det"
                  x={d.box[0]}
                  y={d.box[1]}
                  width={Math.max(0, d.box[2] - d.box[0])}
                  height={Math.max(0, d.box[3] - d.box[1])}
                />
              ))}
            </svg>
          )}
          {/* What each box is, beside it: the product, the confidence and the track it is followed
              under. HTML rather than SVG text, because the overlay is stretched to the picture
              (`preserveAspectRatio="none"`) and text drawn in it would be stretched with it. It
              lets clicks through, so drawing over a box still places a point. */}
          {stream.frame && stream.frame.detections.length > 0 && (
            <div className="bt-labels" data-testid="bt-labels">
              {stream.frame.detections.map((d, i) => (
                <span
                  key={`${d.track_id ?? 'x'}-${i}`}
                  className={`bt-label${d.box[1] < 0.06 ? ' inside-box' : ''}`}
                  data-testid="bt-label"
                  style={{ left: `${d.box[0] * 100}%`, top: `${d.box[1] * 100}%` }}
                  title={d.cls}
                >
                  {d.cls} {Math.round(d.conf * 100)}%
                  {d.track_id !== null && <span className="bt-label-id"> #{d.track_id}</span>}
                </span>
              ))}
            </div>
          )}
          {/* Every drawn point as a marker, numbered in the order it was placed, from the first
              click - a single point draws no outline, so without these a click looked like it did
              nothing. Draggable whenever the drawn layout is chosen, so a misplaced point is
              moved rather than redrawn. */}
          {stream.frame && draft?.zoneMode === 'drawn' && (
            <div className="bt-handles">
              {(['opening', 'inside'] as const).flatMap((which) =>
                draft[OUTLINE_KEY[which]].map((p, i) => {
                  const s = toScreen(p, mirrored)
                  return (
                    <span
                      key={`${which}-${i}`}
                      role="slider"
                      aria-label={`${OUTLINE_LABEL[which]} point ${i + 1}`}
                      aria-valuetext={`${Math.round(p.x * 100)}%, ${Math.round(p.y * 100)}%`}
                      className={`bt-handle ${which}${drawing === which ? ' active' : ''}`}
                      data-testid={`bt-handle-${which}-${i}`}
                      style={{ left: `${s.x * 100}%`, top: `${s.y * 100}%` }}
                      onPointerDown={(e) => beginDrag(e, which, i)}
                      // A click on a point must not also add a point under it.
                      onClick={(e) => e.stopPropagation()}
                    >
                      {i + 1}
                    </span>
                  )
                })
              )}
            </div>
          )}
        </div>
        <p className="bt-legend">
          <span className="key inside" /> inside the basket
          <span className="key opening" /> opening (rim)
          <span className="key outside" /> everything else is outside
        </p>
      </div>

      <aside className="bt-rail">
        <section className="bt-card" aria-labelledby="bt-practice-h">
          <h3 id="bt-practice-h">Test basket</h3>
          {basket === null ? (
            <p className="dim">The basket is not available until the sidecar is running.</p>
          ) : basket.customerBound ? (
            <p className="dim" data-testid="bt-customer">
              A customer session is open on this station, so practice is off. The counts below are
              that customer&apos;s basket.
            </p>
          ) : basket.practice ? (
            <div className="bt-row">
              <span className="bt-pill on" data-testid="bt-practice-on">
                practice running
              </span>
              <button data-testid="bt-practice-restart" onClick={() => void startPractice()}>
                Empty and restart
              </button>
              <button
                data-testid="bt-practice-stop"
                onClick={() => void bridge.stopBasketPractice?.()}
              >
                End
              </button>
            </div>
          ) : (
            <div className="bt-row">
              <button
                className="primary"
                data-testid="bt-practice-start"
                onClick={() => void startPractice()}
              >
                Start practice
              </button>
              <span className="dim">Counts deposits and removals with no tablet.</span>
            </div>
          )}
          {practiceNote && <p className="bt-error">{practiceNote}</p>}

          {readout && (basket?.practice || basket?.customerBound) && (
            <div className="bt-readout" data-testid="bt-readout">
              <div className="bt-stats">
                <span>
                  <b data-testid="bt-count">{readout.itemCount}</b> in basket
                </span>
                <span>
                  <b>{readout.candidates}</b> in motion
                </span>
                {readout.blind && <span className="bt-warn">camera blind</span>}
              </div>
              {readout.items.length > 0 ? (
                <ul className="bt-items" data-testid="bt-items">
                  {readout.items.map((item) => (
                    <li key={item.className}>
                      <span>{item.className}</span>
                      <b>× {item.quantity}</b>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="dim">
                  Empty. Move a product from outside, across the rim, into the basket and hold it
                  there for a second.
                </p>
              )}
              {readout.review.length > 0 && (
                <div className="bt-review" data-testid="bt-review">
                  <p>
                    <b>Needs review ({readout.review.length})</b>: the camera saw something it could
                    not settle.
                  </p>
                  <ul>
                    {readout.review.map((r) => (
                      <li key={r.id}>{r.reason}</li>
                    ))}
                  </ul>
                  <button onClick={() => void bridge.resolvePosReview?.()}>Clear review</button>
                </div>
              )}
            </div>
          )}
        </section>

        <section className="bt-card" aria-labelledby="bt-zones-h">
          <h3 id="bt-zones-h">Zones {dirty && <span className="bt-pill unsaved">unsaved</span>}</h3>
          {draft === null ? (
            <p className="dim">Loading the zone settings…</p>
          ) : (
            <>
              <div className="bt-modes" role="radiogroup" aria-label="Zone layout">
                {(['bands', 'drawn'] as ZoneMode[]).map((mode) => (
                  <label key={mode} className={draft.zoneMode === mode ? 'checked' : ''}>
                    <input
                      type="radio"
                      name="zone-mode"
                      id={`zone-mode-${mode}`}
                      value={mode}
                      checked={draft.zoneMode === mode}
                      onChange={() => {
                        change({ ...draft, zoneMode: mode })
                        setDrawing(null)
                      }}
                    />
                    <span>
                      <b>{mode === 'bands' ? 'A · Bands' : 'B · Drawn outlines'}</b>
                      <small>
                        {mode === 'bands'
                          ? 'Camera under the handle, looking across the basket.'
                          : 'Camera above the basket, looking down into it.'}
                      </small>
                    </span>
                  </label>
                ))}
              </div>

              {draft.zoneMode === 'bands' ? (
                <div className="bt-fields">
                  <label htmlFor="zone-edge">
                    Basket is at the
                    <select
                      id="zone-edge"
                      value={draft.cartEdge}
                      onChange={(e) => change({ ...draft, cartEdge: e.target.value as CartEdge })}
                    >
                      {CART_EDGES.map((edge) => (
                        <option key={edge} value={edge}>
                          {EDGE_LABELS[edge]}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label htmlFor="zone-inside">
                    Inside band, % of picture
                    <input
                      id="zone-inside"
                      type="number"
                      min={5}
                      max={90}
                      step={1}
                      value={Math.round(draft.insideFraction * 100)}
                      onChange={(e) =>
                        change({ ...draft, insideFraction: Number(e.target.value) / 100 })
                      }
                    />
                  </label>
                  <label htmlFor="zone-opening">
                    Opening band, % of picture
                    <input
                      id="zone-opening"
                      type="number"
                      min={5}
                      max={90}
                      step={1}
                      value={Math.round(draft.openingFraction * 100)}
                      onChange={(e) =>
                        change({ ...draft, openingFraction: Number(e.target.value) / 100 })
                      }
                    />
                  </label>
                </div>
              ) : (
                <div className="bt-draw">
                  {(['inside', 'opening'] as const).map((which) => {
                    const pts = which === 'inside' ? draft.drawnInside : draft.drawnOpening
                    return (
                      <div className="bt-draw-row" key={which}>
                        <span>
                          {which === 'inside' ? 'Inside of the basket' : 'Opening (rim)'}
                          <small className="dim">
                            {' '}
                            · {pts.length} point{pts.length === 1 ? '' : 's'}
                          </small>
                        </span>
                        <button
                          data-testid={`bt-draw-${which}`}
                          aria-pressed={drawing === which}
                          className={drawing === which ? 'primary' : ''}
                          disabled={!stream.frame}
                          onClick={() => setDrawing(drawing === which ? null : which)}
                        >
                          {drawing === which ? 'Done' : 'Draw'}
                        </button>
                        <button
                          data-testid={`bt-delete-${which}`}
                          disabled={pts.length === 0}
                          onClick={() => deleteOutline(which)}
                        >
                          Delete
                        </button>
                      </div>
                    )
                  })}
                  <p className="dim">
                    Press Draw, then click the corners on the picture. Each point shows as a
                    numbered marker; drag one to move it. Draw the opening as a ring around the
                    inside outline, or as the strip along the rim. Anything outside both outlines
                    counts as outside.
                  </p>
                </div>
              )}

              {problems.length > 0 && (
                <ul className="bt-problems" data-testid="bt-problems">
                  {problems.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              )}
              <div className="bt-row">
                <button
                  className="primary"
                  data-testid="bt-save-zones"
                  disabled={!dirty || busy || problems.length > 0}
                  onClick={() => void saveZones()}
                >
                  Save zones
                </button>
                <button
                  disabled={!dirty || busy}
                  onClick={() => {
                    if (saved) change(saved)
                    setDrawing(null)
                  }}
                >
                  Revert
                </button>
                <span className="bt-spacer" />
                <button
                  data-testid="bt-undo"
                  disabled={!hist || hist.past.length === 0}
                  onClick={doUndo}
                  title="Undo (Ctrl+Z)"
                >
                  Undo
                </button>
                <button
                  data-testid="bt-redo"
                  disabled={!hist || hist.future.length === 0}
                  onClick={doRedo}
                  title="Redo (Ctrl+Y)"
                >
                  Redo
                </button>
              </div>
              {zoneNote && <p className="bt-note">{zoneNote}</p>}
            </>
          )}
        </section>
      </aside>
    </div>
  )
}
