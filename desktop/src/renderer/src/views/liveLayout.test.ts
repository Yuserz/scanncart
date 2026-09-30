// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import {
  MIN_HEIGHT,
  TUNING_BAND_MIN_HEIGHT,
  TUNING_BAND_MIN_WIDTH,
  contentSize,
  tabletWindowBounds
} from '../../../main/windowSize'

// The Live tab's one-screen promise, as far as a test with no layout engine can check it.
//
// jsdom computes no geometry — `scrollHeight` there is 0 for everything — so the promise cannot be
// asserted by rendering. What *can* be asserted is the set of CSS facts the promise is made of, and
// those are exactly the ones a careless edit would remove: the body's height budget (a grid whose
// first row has a floor, so a shortage scrolls the rail rather than the page), the rail's own
// scroll, the item log's floor, the band's collapse actually collapsing, and *no second layout* —
// the stacked mode that made the page scroll 802px at 768x576 is what this replaced.
//
// The measurements themselves live in `desktop/scripts/check-live-layout.mjs`
// (`make verify-live-layout`), which draws the real renderer in Electron at every size the standard
// allows and fails if the page scrolls. This file is the cheap net under it: it runs in the suite, on
// a bare checkout, so the structural regressions do not have to wait for someone to run the real
// one - and its last block is the net under *that*, since a measurement wired to nothing is the hand
// check it replaced.

const here = dirname(fileURLToPath(import.meta.url))
const LIVE_CSS = readFileSync(join(here, 'LiveView.css'), 'utf8')
const TUNING_CSS = readFileSync(join(here, '../components/CameraTuning.css'), 'utf8')

/** Comments mention selectors too (and one of them describes the rule below it), so they go first. */
function withoutComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, '')
}

/** The declarations of one rule, matched on its whole selector (so `.live-body` misses `> .x`). */
function rule(css: string, selector: string): string | null {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = new RegExp(`(?:^|[},])\\s*${escaped}\\s*\\{`).exec(css)
  if (match === null) return null
  const start = match.index + match[0].length
  const end = css.indexOf('}', start)
  return css.slice(start, end)
}

function declaration(body: string | null, property: string): string | null {
  if (body === null) return null
  const match = new RegExp(`(?:^|;)\\s*${property}\\s*:\\s*([^;]+)`).exec(body)
  return match === null ? null : match[1].trim()
}

/** Every reason this stylesheet would let the page scroll at a window the standard allows. */
function problems(liveCss: string): string[] {
  const found: string[] = []
  liveCss = withoutComments(liveCss)

  // A second layout is the thing that broke the promise: the stacked mode kept natural content
  // height and needed 1296px of it in 494px at the floor. `LiveView` has one layout, and the band
  // closes itself to fit it; a media query here means a size that has to be measured separately.
  if (/@media/.test(liveCss)) {
    found.push('a media query: the body has one layout, and a second one is unmeasured')
  }

  const view = rule(liveCss, '.live-view')
  if (declaration(view, 'overflow-y') !== 'auto') {
    found.push('.live-view must be `overflow-y: auto`, so a shortage scrolls rather than clips')
  }

  const body = rule(liveCss, '.live-body')
  if (declaration(body, 'display') !== 'grid') found.push('.live-body must be the grid')
  if (declaration(body, 'flex') !== '1') found.push('.live-body must claim the height left over')
  // The first row's floor is what keeps the page fixed: with the track able to shrink to 200px, a
  // short window gives the rail a shorter box (which scrolls its own rows) instead of growing the
  // page. A bare `1fr` would let the band's row push it instead.
  if (declaration(body, 'grid-template-rows') !== 'minmax(200px, 1fr) auto') {
    found.push(".live-body rows must keep the first row's 200px floor and let the band size itself")
  }
  if (!/clamp\(/.test(declaration(body, 'grid-template-columns') ?? '')) {
    found.push('.live-body must keep the rail a clamp()ed track and give the rest to the preview')
  }

  const rail = rule(liveCss, '.side-rail')
  if (declaration(rail, 'overflow-y') !== 'auto') {
    found.push('.side-rail must scroll itself: that is where a height shortage goes')
  }
  if (declaration(rail, 'min-height') !== '0') {
    found.push(
      '.side-rail needs `min-height: 0`, or a flex/grid item refuses to shrink and overflows'
    )
  }

  const log = rule(liveCss, '.log-card')
  // The number the standard's height budget is built on (`windowSize.ts`): four rows of item log.
  if (declaration(log, 'min-height') !== '108px') {
    found.push(".log-card must keep its 108px floor — the standard's budget assumes it")
  }

  return found
}

/** The same, for the band: it collapses, and the collapse is spelled out in CSS. */
function tuningProblems(tuningCss: string): string[] {
  const found: string[] = []
  tuningCss = withoutComments(tuningCss)

  // React collapses the band by setting `hidden` on each group, and `display: none` from the UA
  // sheet loses to the group's own `display: flex` — author rules beat UA ones whatever the
  // specificity. Without this rule the toggle swapped its arrow and left all 13 controls on screen,
  // which is a band that never gave its 130-odd pixels back to the item log.
  if (declaration(rule(tuningCss, '.tuning-group[hidden]'), 'display') !== 'none') {
    found.push('.tuning-group[hidden] must be `display: none`, or the collapse never happens')
  }
  // The band is laid out at two widths (four columns, then a row per group): both are heights the
  // standard has to fit, and the second is the one at the floor when the operator opens it there.
  if (!/@media \(max-width: 900px\)/.test(tuningCss)) {
    found.push("the band's narrow arrangement is missing: its columns need the canvas width")
  }

  return found
}

describe('the Live tab can fit one window', () => {
  it('keeps the CSS facts the one-screen promise is made of', () => {
    expect(problems(LIVE_CSS)).toEqual([])
    expect(tuningProblems(TUNING_CSS)).toEqual([])
  })

  it('catches a second layout, so the check is not vacuous', () => {
    // The exact shape this replaced, planted back into the stylesheet it was deleted from.
    const planted =
      LIVE_CSS +
      '\n@media (max-width: 900px) {\n  .live-body { display: flex; flex-direction: column; }\n}\n'

    expect(problems(planted)).toContainEqual(expect.stringContaining('media query'))
  })

  it('catches a body that would grow the page instead of scrolling the rail', () => {
    const planted = LIVE_CSS.replace(
      'grid-template-rows: minmax(200px, 1fr) auto;',
      'grid-template-rows: 1fr auto;'
    )

    expect(planted).not.toBe(LIVE_CSS) // the replace found its target
    expect(problems(planted)).toContainEqual(expect.stringContaining('200px floor'))
  })

  it('catches a collapse that does not collapse', () => {
    const planted = TUNING_CSS.replace(
      '.tuning-group[hidden] {\n  display: none;\n}',
      '.tuning-group[hidden] {\n  display: flex;\n}'
    )

    expect(tuningProblems(planted)).toContainEqual(expect.stringContaining('collapse'))
  })

  it('holds the standard the measurements were taken against', () => {
    const bounds = tabletWindowBounds()

    // Two thresholds, one collapse: the band needs the canvas's width for its columns and a
    // content height its own row plus the rail's rows fit in (`windowSize.ts`), and the window
    // floor has to be low enough that the *shut* band is what the smaller windows see.
    //
    // Both ends are read as the content areas they draw, because that is what `matchMedia` compares
    // - the window is 16px wider and 39px taller than the viewport. Stating this against the window
    // sizes is the bug this pins: 1000 against 1024 holds by luck, and a threshold set to the canvas
    // itself would shut the band on the one window it was drawn for.
    const canvas = contentSize({ width: bounds.maxWidth, height: bounds.maxHeight })
    const floor = contentSize({ width: bounds.minWidth, height: bounds.minHeight })

    expect(TUNING_BAND_MIN_WIDTH).toBeLessThanOrEqual(canvas.width)
    expect(TUNING_BAND_MIN_HEIGHT).toBeLessThanOrEqual(canvas.height)
    expect(floor.width).toBeLessThan(TUNING_BAND_MIN_WIDTH)
    expect(floor.height).toBeLessThan(TUNING_BAND_MIN_HEIGHT)
    expect(MIN_HEIGHT).toBeLessThan(TUNING_BAND_MIN_HEIGHT)

    // The floor, against the budget the file states: chrome and padding around the body, the
    // collapsed band at its narrow (132px) form, the gap, the rail's stats strip + flags + 108px
    // log, and the title bar. A floor below this is a window where the rail scrolls for no reason.
    const budget = [116, 132, 12, 356, 38]

    expect(MIN_HEIGHT).toBeGreaterThanOrEqual(budget.reduce((a, b) => a + b, 0))
  })
})

// The measurement above is the promise; these guard its wiring, which is the part that rots
// silently. A renamed script, a target dropped from `make help`, or a step deleted from a workflow
// leaves a one-screen promise measured by nobody while every test in this file still passes - the
// same failure the sidecar's `verify-clamp`/`verify-unsure` guards exist for.
const REPO_ROOT = join(here, '../../../../..')
const CHECK = join(REPO_ROOT, 'desktop', 'scripts', 'check-live-layout.mjs')

/**
 * One target's recipe: its own lines and the tab-continued ones, stopping at the first that is not.
 * The leading tab is make's, not the recipe's, so it is dropped.
 */
function recipe(source: string, target: string): string {
  const lines = source.split('\n')
  const start = lines.findIndex((line) => line.startsWith(`${target}:`))
  if (start === -1) return ''
  const taken: string[] = []
  for (const line of lines.slice(start + 1)) {
    if (!line.startsWith('\t')) break
    taken.push(line.slice(1))
  }
  return taken.join('\n')
}

describe('the one-screen check is wired up where it has to be', () => {
  const makefile = readFileSync(join(REPO_ROOT, 'Makefile'), 'utf8')

  it('reads a recipe correctly, so the assertions below can fail', () => {
    const planted = [
      'help:',
      '\t@echo hi',
      'verify-live-layout: desktop-build',
      '\tcd desktop && node scripts/planted.mjs',
      '',
      'other:',
      '\t@echo other'
    ].join('\n')

    expect(recipe(planted, 'verify-live-layout')).toBe('cd desktop && node scripts/planted.mjs')
    // A target whose name is a prefix of another's is not that other target, and the stop rule is
    // what keeps one recipe from swallowing the next.
    expect(recipe(planted, 'help')).toBe('@echo hi')
    expect(recipe(planted, 'verify')).toBe('')
  })

  it('keeps the target real, phony, listed, and out of the suite that has no display', () => {
    expect(recipe(makefile, 'verify-live-layout')).toContain('check-live-layout.mjs')
    // Phony, or a directory of that name on PATH would shadow it and the target would no-op.
    expect(makefile.slice(makefile.indexOf('.PHONY'), makefile.indexOf('help:'))).toContain(
      'verify-live-layout'
    )
    // Named in the help output, the only index of the targets.
    expect(makefile).toContain('"  verify-live-layout')
    // And not a prerequisite of `test`, which CI runs everywhere including where there is no
    // display: the measurement is reachable on its own.
    const testLine = makefile.split('\n').find((line) => line.startsWith('test:')) ?? ''
    expect(testLine.split(':')[1].split(/\s+/)).not.toContain('verify-live-layout')
  })

  it('names the target in the docs that tell an operator what to run', () => {
    expect(readFileSync(join(REPO_ROOT, 'CLAUDE.md'), 'utf8')).toContain('make verify-live-layout')
    expect(readFileSync(join(REPO_ROOT, 'README.md'), 'utf8')).toContain('make verify-live-layout')
  })

  it('runs it in CI, which is the whole point of not measuring it by hand', () => {
    const ci = readFileSync(join(REPO_ROOT, '.github/workflows/ci.yml'), 'utf8')

    // The whole command, not the two names: the step's own comment explains why it needs a display,
    // so asserting on `xvfb-run` alone would still pass with the step reduced to a bare `node` —
    // which is exactly what it was before this was tightened.
    expect(ci).toContain('xvfb-run -a node scripts/check-live-layout.mjs')
  })

  it('takes its sizes from the standard, so a new canvas moves the measurement', () => {
    const check = readFileSync(CHECK, 'utf8')

    // The four sizes are the canvas and the floor's two axes, read from the module that owns them
    // rather than typed in: a literal size here is how a measurement quietly stops describing the
    // standard it reports on.
    expect(check).toContain("'../src/main/windowSize.ts'")
    expect(check).toContain('width: TABLET.width')
    expect(check).toContain('height: TABLET.height')
    expect(check).toContain('width: MIN_WIDTH')
    expect(check).toContain('height: MIN_HEIGHT')
  })

  it('measures a built renderer, and fails loudly when there is none to measure', () => {
    const check = readFileSync(CHECK, 'utf8')

    expect(check).toContain("'out', 'renderer', 'index.html'")
    // Exit 2 and a sentence naming the fix: a check that skips itself is a check that passes, and a
    // checkout with no `make build` behind it is the one state this can be run in by mistake.
    expect(check).toContain('run `make build` first')
    expect(check).toMatch(/process\.exit\(2\)/)
  })
})
