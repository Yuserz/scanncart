// The app's preload, stubbed down to the few things the renderer asks for while drawing the Live
// view.
//
// `App` waits for a sidecar port before it draws any view at all, so the harness has to answer
// `getSidecarPort` — with a port nothing is listening on: the layout is what is being measured, and
// no sidecar (or camera, or venv) should decide whether this check can run. Everything else here is
// the real renderer: the real built bundle, the real stylesheet, a real Chromium at a real size.
//
// It also answers `measure`, which the harness asks for once per size, because `ipcRenderer` here
// has full access to the page's DOM and no content-security policy in the way.
const { ipcRenderer } = require('electron')

const NOTHING_LISTENING = 9

window.api = {
  getSidecarPort: () => Promise.resolve(NOTHING_LISTENING),
  getSidecarHealth: () => Promise.resolve('starting'),
  onSidecarHealth: () => () => {},
  // The POS readout is part of the rail this promise is made about, and it is only on screen when
  // the integration is configured — so it is answered as *bound* rather than left absent. A layout
  // measured with the feature off would be the one layout that never carries the card, and the
  // card is the only thing it could push the item log's floor past.
  getPosState: () =>
    Promise.resolve({
      phase: 'bound',
      cartCode: 'ABCD1234',
      syncedItemCount: 4,
      lastSyncAgeS: 7,
      error: null
    }),
  onPosState: () => () => {}
}
window.electron = { process: { versions: process.versions } }

function box(el) {
  if (!el) return null
  const rect = el.getBoundingClientRect()
  return {
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    client: el.clientHeight,
    scroll: el.scrollHeight
  }
}

function measure() {
  const pick = (selector) => document.querySelector(selector)
  const toggle = pick('.tuning-toggle')
  return {
    // The renderer's own viewport: the *content* area, which is what `matchMedia` and the layout
    // both see, and what the standard's band thresholds are stated in.
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    mounted: pick('.live-view') !== null,
    view: box(pick('.live-view')),
    rail: box(pick('.side-rail')),
    logCard: box(pick('.log-card')),
    preview: box(pick('.preview-wrapper')),
    band: {
      expanded: toggle === null ? null : toggle.getAttribute('aria-expanded') === 'true',
      height: box(pick('.tuning-card')) === null ? null : box(pick('.tuning-card')).height,
      // The collapse is React setting `hidden` on each group, so with no sidecar (and therefore no
      // settings) the *fields* are absent for a reason that has nothing to do with the layout:
      // these two count the attribute, and `fields` is reported as a number the reader can sanity
      // check rather than one this check rules on when it is 0.
      groups: document.querySelectorAll('.tuning-group').length,
      groupsHidden: [...document.querySelectorAll('.tuning-group')].filter((group) =>
        group.hasAttribute('hidden')
      ).length,
      fields: [...document.querySelectorAll('.tuning-field')].filter(
        (field) => field.getClientRects().length > 0
      ).length
    }
  }
}

ipcRenderer.on('measure', () => ipcRenderer.send('measured', measure()))
