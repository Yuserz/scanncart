import { ElectronAPI } from '@electron-toolkit/preload'
import type { SidecarHealth } from '../main/sidecarHealth'
import type { PosConfig } from '../main/posConfig'
import type { PosConnectionResult, PosState } from '../main/posSession'
import type { BasketViewState } from '../main/transferStream'

// The vocabulary is the main process's (`src/main/sidecarHealth.ts`, which produces it); this file
// declares the boundary that carries it, so the renderer's reading is typed from the one definition
// rather than from a second copy of the union.
export interface SidecarApi {
  getSidecarPort: () => Promise<number | null>
  getSidecarHealth: () => Promise<SidecarHealth>
  onSidecarHealth: (cb: (health: SidecarHealth) => void) => () => void
  getPosState: () => Promise<PosState | null>
  getPosConfig: () => Promise<PosConfig>
  savePosConfig: (patch: Partial<PosConfig>) => Promise<PosConfig>
  testPosConnection: () => Promise<PosConnectionResult>
  resolvePosReview: (id?: string) => Promise<void>
  onPosState: (cb: (state: PosState | null) => void) => () => void
  getBasketState: () => Promise<BasketViewState | null>
  startBasketPractice: () => Promise<BasketViewState>
  stopBasketPractice: () => Promise<BasketViewState | null>
  onBasketState: (cb: (state: BasketViewState) => void) => () => void
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: SidecarApi
  }
}
