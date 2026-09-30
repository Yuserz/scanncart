// Does the Live tab still fit one screen at every size the window standard allows?
//
// The console's whole premise is one screen with no page scroll: preview and rail beside each other,
// the camera band under them, the item log scrolling inside its own card. That is a claim about
// geometry, so this measures it — the built renderer, in Electron, at the canvas and at both corners
// of the floor — instead of arguing it from the stylesheet. The structural half (the CSS facts the
// claim rests on) is `src/renderer/src/views/liveLayout.test.ts`, which runs in the normal suite;
// this is the half that needs a browser engine.
//
// It reads the sizes from the code that owns them, so changing the canvas moves the check with it,
// and it fails loudly rather than skipping when there is nothing to measure: a built renderer is
// required (`make build`), and the harness draws it with no sidecar, no camera and no network —
// which is exactly what a checkout has.
//
// Run it with `make verify-live-layout`, or directly:
//     cd desktop && node scripts/check-live-layout.mjs
import { createRequire } from 'node:module'
import { existsSync } from 'node:fs'
import { spawn } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  MIN_HEIGHT,
  MIN_WIDTH,
  TABLET,
  TUNING_BAND_MIN_HEIGHT,
  TUNING_BAND_MIN_WIDTH
} from '../src/main/windowSize.ts'

const here = dirname(fileURLToPath(import.meta.url))
const desktop = join(here, '..')
const renderer = join(desktop, 'out', 'renderer', 'index.html')
const electron = createRequire(import.meta.url)('electron')

if (!existsSync(renderer)) {
  // Fail loudly: a check that skips itself is a check that passes.
  console.error(`check-live-layout: no built renderer at ${renderer}`)
  console.error('check-live-layout: run `make build` first (the target does it for you).')
  process.exit(2)
}

// The canvas and the four corners of the floor. Each corner is one threshold: the wide-but-short
// window is where the band's *height* rule has to close it and the narrow-but-tall one is where its
// *width* rule does, and those are the two sizes where a single-condition rule would scroll.
const sizes = [
  { label: 'canvas', width: TABLET.width, height: TABLET.height },
  { label: 'floor', width: MIN_WIDTH, height: MIN_HEIGHT },
  { label: 'floor, canvas width', width: TABLET.width, height: MIN_HEIGHT },
  { label: 'floor height, canvas width', width: MIN_WIDTH, height: TABLET.height }
]

// CI is the one place this runs without a usable Chromium sandbox - a hosted runner's AppArmor
// profile blocks unprivileged user namespaces, and a container's `/dev/shm` is smaller than the
// shared-memory arena Chromium maps by default. Neither is a security decision here: the only thing
// the window ever loads is this checkout's own built renderer, over `file://`. Named rather than
// assumed, so the local run keeps both defaults.
const CI_FLAGS = ['--no-sandbox', '--disable-dev-shm-usage']

function electronArgs(requested) {
  return [
    ...(process.env.CI ? CI_FLAGS : []),
    join(here, 'layout-harness.cjs'),
    JSON.stringify(requested),
    renderer
  ]
}

function runHarness(requested) {
  return new Promise((resolve, reject) => {
    const child = spawn(electron, electronArgs(requested), { cwd: desktop })
    let out = ''
    let err = ''
    child.stdout.on('data', (chunk) => (out += chunk))
    child.stderr.on('data', (chunk) => (err += chunk))
    child.on('error', reject)
    child.on('exit', (code) => {
      const line = out.split('\n').find((l) => l.startsWith('LAYOUT_RESULTS '))
      if (line === undefined) {
        reject(new Error(`the harness exited ${code} without a measurement\n${err.trim()}`))
        return
      }
      resolve(JSON.parse(line.slice('LAYOUT_RESULTS '.length)))
    })
  })
}

const results = await runHarness(sizes)

const problems = []
const rows = []

for (const result of results) {
  const pageScrolls = result.view.scroll > result.view.client
  const pageOverflow = result.view.scroll - result.view.client
  const railOverflow = result.rail.scroll - result.rail.client
  // The two rules are a pair, and this is the expectation they produce — computed from what the
  // renderer *reports* about its own viewport, not from the size that was asked for.
  const bandShouldBeOpen =
    result.innerWidth >= TUNING_BAND_MIN_WIDTH && result.innerHeight >= TUNING_BAND_MIN_HEIGHT

  if (pageScrolls) {
    problems.push(
      `${result.label}: the page scrolls by ${pageOverflow}px ` +
        `(${result.view.scroll} of content in ${result.view.client} of view)`
    )
  }
  if (result.logCard.height < 108) {
    problems.push(
      `${result.label}: the item log is ${result.logCard.height}px tall, under its 108px floor`
    )
  }
  if (result.band.expanded !== bandShouldBeOpen) {
    problems.push(
      `${result.label}: the camera band is ${result.band.expanded ? 'open' : 'shut'} at ` +
        `${result.innerWidth}x${result.innerHeight} of content, where the standard says ` +
        `${bandShouldBeOpen ? 'open' : 'shut'}`
    )
  }
  // What "open" means, without needing the sidecar this harness deliberately does not run: the
  // groups React gates carry (or do not carry) `hidden`. A number of controls between 1 and 12 is a
  // real failure — the band shows all 13 or none — while 0 is this environment having no settings.
  if (bandShouldBeOpen && result.band.groupsHidden !== 0) {
    problems.push(
      `${result.label}: the band should be open but ${result.band.groupsHidden} of its ` +
        `${result.band.groups} groups are collapsed`
    )
  }
  if (!bandShouldBeOpen && result.band.groupsHidden === 0) {
    problems.push(`${result.label}: the band should be shut but none of its groups are collapsed`)
  }
  if (result.band.fields > 0 && result.band.fields !== 13) {
    problems.push(
      `${result.label}: ${result.band.fields} of 13 controls are visible with the band open`
    )
  }

  rows.push([
    result.label + ` (${result.width}x${result.height})`,
    `${result.innerWidth}x${result.innerHeight}`,
    pageScrolls ? `scrolls ${pageOverflow}px` : 'none',
    railOverflow > 0 ? `scrolls ${railOverflow}px` : 'none',
    `${result.logCard.height}px`,
    result.band.expanded ? `open, ${result.band.fields || 13}/13` : `shut, ${result.band.height}px`,
    `${result.band.groups - result.band.groupsHidden}/${result.band.groups} groups`,
    `${result.preview.width}x${result.preview.height}`
  ])
}

const header = [
  'window',
  'content',
  'page',
  'rail',
  'item log',
  'camera band',
  'groups shown',
  'preview'
]
const widths = header.map((h, i) => Math.max(h.length, ...rows.map((row) => row[i].length)))
const line = (cells) => cells.map((cell, i) => cell.padEnd(widths[i])).join('  ')

console.log('Live tab geometry, in the built renderer')
console.log(`standard: canvas ${TABLET.width}x${TABLET.height}, floor ${MIN_WIDTH}x${MIN_HEIGHT}`)
console.log(
  `band opens at ${TUNING_BAND_MIN_WIDTH}px of content width and ${TUNING_BAND_MIN_HEIGHT}px of ` +
    'content height'
)
console.log('')
console.log(line(header))
console.log(line(widths.map((w) => '-'.repeat(w))))
for (const row of rows) console.log(line(row))
console.log('')

if (problems.length > 0) {
  console.error('the Live tab does not fit one screen:')
  for (const problem of problems) console.error(`  - ${problem}`)
  process.exit(1)
}

console.log('the Live tab fits one screen at every size the standard allows (no page scroll)')
