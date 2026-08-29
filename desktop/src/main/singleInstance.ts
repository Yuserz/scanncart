// What a second launch should do to the instance that already exists.
//
// Extracted from index.ts so it can be tested without Electron: index.ts wires
// the real BrowserWindow and supervisor in, the same way sidecar.ts takes an
// injected spawnFn.

/** The subset of BrowserWindow this needs. */
export interface FocusableWindow {
  isMinimized(): boolean
  restore(): void
  isVisible(): boolean
  show(): void
  focus(): void
}

export interface SecondInstanceDeps {
  /** The existing window, or null if it was closed. */
  getWindow: () => FocusableWindow | null
  createWindow: () => void
  /** The sidecar's port, or null when it never reported one / has died. */
  getSidecarPort: () => number | null
  restartSidecar: () => void
}

/**
 * Raise the existing window, and revive its sidecar if that died.
 *
 * The revive half matters because `main/index.ts` has no sidecar auto-restart:
 * if the sidecar dies, the renderer polls for a port forever and the app is
 * stuck. Relaunching is what a user naturally tries at that point, and before
 * the single-instance lock that just produced a second, equally broken
 * instance. Now it repairs the one they have.
 */
export function handleSecondInstance(deps: SecondInstanceDeps): void {
  const window = deps.getWindow()
  if (window === null) {
    deps.createWindow()
  } else {
    // Order matters: a minimized window must be restored before focus() will
    // actually raise it.
    if (window.isMinimized()) window.restore()
    if (!window.isVisible()) window.show()
    window.focus()
  }

  if (deps.getSidecarPort() === null) {
    deps.restartSidecar()
  }
}
