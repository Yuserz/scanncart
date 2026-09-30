// Draw the built renderer at each size it is handed and report what the layout did.
//
// Run by `check-live-layout.mjs`, not by hand: the driver owns the sizes (it reads them from
// `src/main/windowSize.ts`) and the assertions, this file owns the window. It is Electron rather
// than a browser because the numbers a layout check is about are the ones a *real* engine computes
// — jsdom reports 0 for every `scrollHeight`, which is the whole reason the promise was measured by
// hand until this existed.
//
// Usage: electron scripts/layout-harness.cjs '<json sizes>' <built renderer index.html>
const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('node:path')

const sizes = JSON.parse(process.argv[2] ?? '[]')
const renderer = process.argv[3]

if (sizes.length === 0 || !renderer) {
  console.error('layout-harness: expected <sizes json> <renderer index.html>')
  app.exit(2)
}

function measureAt(win, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('layout-harness: the renderer did not answer `measure`')),
      timeoutMs
    )
    ipcMain.once('measured', (_event, measured) => {
      clearTimeout(timer)
      resolve(measured)
    })
    win.webContents.send('measure')
  })
}

async function main() {
  const win = new BrowserWindow({
    width: sizes[0].width,
    height: sizes[0].height,
    // Never shown: this is a measurement, and a window appearing and resizing itself on someone's
    // desktop is a worse way to find that out than reading the table it prints. Layout, media
    // queries and `clientHeight` are all computed for an unshown window.
    show: false,
    // The app's own window options, because they are what the content area is: a visible menu bar
    // costs 27px of height, and the band's height threshold is close enough to the canvas that
    // those 27px decide whether it opens. A harness that draws a *different* window would measure a
    // layout nobody runs.
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'layout-harness-preload.cjs'),
      // The preload writes `window.api` itself, which only reaches the page in one world.
      contextIsolation: false,
      // An unshown window is a hidden one, and Chromium defers a hidden window's layout: without
      // this, a resize is not applied to the page until the next time it is drawn, which is never.
      // The first measurement is taken long after loading and so passed anyway, which is exactly
      // how this looked like a working check while four sizes came back with one window's numbers.
      backgroundThrottling: false
    }
  })

  await win.loadFile(renderer)

  // The renderer mounts its view once it has a port, so the first measurement is a poll: an unmounted
  // page reports no `.live-view`, and that is not a layout failure — it is too early.
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if ((await measureAt(win)).mounted) break
    await new Promise((resolve) => setTimeout(resolve, 100))
  }

  const results = []
  for (const size of sizes) {
    win.setBounds({ ...win.getBounds(), width: size.width, height: size.height })
    // A resize settles over a frame or two: the grid re-lays out, then `matchMedia` fires and the
    // band re-renders. Give it long enough that this is a measurement rather than a race.
    await new Promise((resolve) => setTimeout(resolve, 400))
    results.push({ ...size, ...(await measureAt(win)) })
  }

  console.log('LAYOUT_RESULTS ' + JSON.stringify(results))
  app.exit(0)
}

app.disableHardwareAcceleration()
app.whenReady().then(main)
