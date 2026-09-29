import { ElectronAPI } from '@electron-toolkit/preload'
import type { SidecarHealth } from '../main/sidecarHealth'

// The vocabulary is the main process's (`src/main/sidecarHealth.ts`, which produces it); this file
// declares the boundary that carries it, so the renderer's reading is typed from the one definition
// rather than from a second copy of the union.
export interface SidecarApi {
  getSidecarPort: () => Promise<number | null>
  getSidecarHealth: () => Promise<SidecarHealth>
  onSidecarHealth: (cb: (health: SidecarHealth) => void) => () => void
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: SidecarApi
  }
}
