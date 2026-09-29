import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { SidecarSupervisor } from './sidecar'
import { HEALTH_TIMEOUT_MS, SidecarHealthMonitor, type SidecarHealth } from './sidecarHealth'
import { handleSecondInstance } from './singleInstance'
import { tabletWindowBounds } from './windowSize'

// Resolve where the Python sidecar lives. Defaults assume the repo layout
// (desktop/ and sidecar/ side by side) and the sidecar's local venv; override
// with SIDECAR_PYTHON / SIDECAR_SCRIPT for packaged builds or custom setups.
function resolveSidecarPaths(): { python: string; script: string; cwd: string } {
  const sidecarDir = join(app.getAppPath(), '..', 'sidecar')
  const venvPython =
    process.platform === 'win32'
      ? join(sidecarDir, '.venv', 'Scripts', 'python.exe')
      : join(sidecarDir, '.venv', 'bin', 'python')
  return {
    python: process.env['SIDECAR_PYTHON'] || venvPython,
    script: process.env['SIDECAR_SCRIPT'] || join(sidecarDir, 'run.py'),
    cwd: sidecarDir
  }
}

let sidecarPort: number | null = null
let supervisor: SidecarSupervisor | null = null
let mainWindow: BrowserWindow | null = null
// The last thing the main process knows about whether the sidecar is answering. Starts at
// `starting` because that is true: nothing has been asked yet, and a banner at launch would be
// noise on every healthy start.
let sidecarHealth: SidecarHealth = 'starting'
let healthMonitor: SidecarHealthMonitor | null = null

// Whether the sidecar on `port` answers its own health route. A refused connection rejects
// immediately, which is the case this exists for (a process that is alive with nothing listening),
// and a request that hangs is what the deadline is for.
async function probeSidecar(port: number): Promise<boolean> {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/api/health`, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS)
    })
    return res.ok
  } catch {
    return false
  }
}

// One health state, two read paths, because either alone loses a transition: every window is told
// about a change as it happens, and a window that mounts later reads the current value over IPC
// instead of waiting for the next one (which may never come — a sidecar that stays broken sends no
// further transitions).
function setSidecarHealth(health: SidecarHealth): void {
  sidecarHealth = health
  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send('sidecar:health', health)
  }
}

function startSidecar(): void {
  const { python, script, cwd } = resolveSidecarPaths()
  // A restart must not leave the previous monitor probing a port that is about to change.
  healthMonitor?.stop()
  healthMonitor = new SidecarHealthMonitor({ probe: probeSidecar, onChange: setSidecarHealth })
  supervisor = new SidecarSupervisor({
    spawnFn: (command, args, options) => {
      const cp = spawn(command, args, options)
      // Guard against ENOENT (bad python path) crashing the main process.
      cp.on('error', (e) => console.error('[sidecar] spawn error:', e.message))
      return cp
    },
    pythonPath: python,
    scriptPath: script,
    cwd,
    onPort: (port) => {
      sidecarPort = port
      console.log(`[sidecar] ready on port ${port}`)
      healthMonitor?.start(port)
    },
    onExit: (code) => {
      console.error(`[sidecar] exited unexpectedly (code ${code})`)
      sidecarPort = null
      // No probing left to do — the answer is known, and reporting it now rather than after three
      // failed probes is the difference between "the app is broken" and "the app is broken, and
      // it says why". `onExit` is suppressed for a stop() we asked for (`before-quit`,
      // the second-instance repair), so a clean shutdown does not flash this.
      healthMonitor?.stop()
      healthMonitor = null
      setSidecarHealth('unresponsive')
    },
    onStderr: (text) => process.stderr.write(text),
    // The sidecar's own stdout prints, which are the ones an operator needs and no log carried:
    // `run.py` announces `EVENT_LOOP=…` right after the port (see `app/loops.py`), and that line is
    // how you tell the proactor wedge apart from a second problem without instrumenting the
    // machine. Dropping it here left the choice visible everywhere except the log of the app that
    // runs it. `console.log` rather than `process.stderr.write` (where uvicorn's own lines go):
    // these are the sidecar's app-level messages, and the `[sidecar]` prefix is what keeps them
    // legible beside the framework's.
    onStdoutLine: (line) => console.log(`[sidecar] ${line}`)
  })
  supervisor.start()
}

function createWindow(): void {
  // Create the browser window at the tablet standard — one canvas that supplies the opening size
  // and both bounds, rather than four numbers here that can disagree with the layout
  // (`windowSize.ts`).
  mainWindow = new BrowserWindow({
    ...tabletWindowBounds(),
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// Raise the window this app already has, and revive it if its sidecar died.
// The logic lives in singleInstance.ts so it is testable without Electron.
function focusExistingInstance(): void {
  handleSecondInstance({
    getWindow: () => mainWindow,
    createWindow,
    getSidecarPort: () => sidecarPort,
    restartSidecar: () => {
      console.log('[sidecar] no port on second-instance — restarting it')
      supervisor?.stop()
      supervisor = null
      startSidecar()
    }
  })
}

// A second launch must never reach startSidecar(): two sidecars race for port
// 8765, for data/settings.json and for the same SQLite file, and the loser
// silently ends up on a different port. Take the lock before anything is
// spawned, and hand the user the instance they already have.
if (!app.requestSingleInstanceLock()) {
  console.log('[app] another instance is already running — focusing it')
  app.quit()
} else {
  app.on('second-instance', () => focusExistingInstance())

  // This method will be called when Electron has finished
  // initialization and is ready to create browser windows.
  // Some APIs can only be used after this event occurs.
  app.whenReady().then(() => {
    // Set app user model id for windows
    electronApp.setAppUserModelId('com.scanncart.app')

    // Default open or close DevTools by F12 in development
    // and ignore CommandOrControl + R in production.
    // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
    app.on('browser-window-created', (_, window) => {
      optimizer.watchWindowShortcuts(window)
    })

    // Renderer asks for the sidecar port; returns null until the sidecar reports it.
    ipcMain.handle('sidecar:port', () => sidecarPort)
    // ...and for whether it is answering, which the push above keeps current from then on.
    ipcMain.handle('sidecar:health', () => sidecarHealth)

    startSidecar()
    createWindow()

    app.on('activate', function () {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
  })
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// Ensure the sidecar child is terminated with the app.
app.on('before-quit', () => {
  healthMonitor?.stop()
  healthMonitor = null
  supervisor?.stop()
  supervisor = null
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
