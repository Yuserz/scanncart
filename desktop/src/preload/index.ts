import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type { SidecarHealth } from '../main/sidecarHealth'
import type { PosConfig } from '../main/posConfig'
import type { PosConnectionResult, PosState } from '../main/posSession'
import type { BasketViewState } from '../main/transferStream'

// The channel the main process pushes POS state on, mirroring `sidecar:health`.
const POS_STATE_CHANNEL = 'pos:state'
// ...and the basket test screen's state, pushed the same way.
const BASKET_STATE_CHANNEL = 'basket:state'

// Custom APIs for renderer. Hands the renderer the sidecar port (null until the
// sidecar has reported it); the renderer then connects directly over WS/HTTP.
const api = {
  getSidecarPort: (): Promise<number | null> => ipcRenderer.invoke('sidecar:port'),
  // Whether the sidecar is answering at all. The renderer cannot see this for itself: its
  // WebSocket is *established* over the same loopback and stays established when the sidecar stops
  // listening, which is why a dead sidecar looked like a frozen app rather than a dead one. The
  // main process owns the child, so the main process asks.
  getSidecarHealth: (): Promise<SidecarHealth> => ipcRenderer.invoke('sidecar:health'),
  // Every transition, as it happens. Returns the unsubscribe instead of leaving the listener to be
  // removed by name elsewhere: the effect that registers it is the only thing that knows when it is
  // finished with it.
  onSidecarHealth: (cb: (health: SidecarHealth) => void): (() => void) => {
    const listener = (_event: IpcRendererEvent, health: SidecarHealth): void => cb(health)
    ipcRenderer.on('sidecar:health', listener)
    return () => {
      ipcRenderer.removeListener('sidecar:health', listener)
    }
  },
  // The POS integration. Its config and its connection test go through the main process because
  // the renderer cannot reach the pushcart-web host directly (CORS), and its state is pushed the
  // same way sidecar health is — a read for a window that mounts late, a push for one already open.
  getPosState: (): Promise<PosState | null> => ipcRenderer.invoke('pos:get-state'),
  getPosConfig: (): Promise<PosConfig> => ipcRenderer.invoke('pos:get-config'),
  savePosConfig: (patch: Partial<PosConfig>): Promise<PosConfig> =>
    ipcRenderer.invoke('pos:save-config', patch),
  testPosConnection: (): Promise<PosConnectionResult> => ipcRenderer.invoke('pos:test-connection'),
  // Staff looked at the basket and confirmed it: clear one review item by id, or all of them.
  resolvePosReview: (id?: string): Promise<void> => ipcRenderer.invoke('pos:resolve-review', id),
  // `null` is a state this channel really carries - the feature being switched off, or nothing
  // configured - rather than a value that only ever appears in a read, so it is part of the
  // signature instead of something the renderer has to know to expect.
  onPosState: (cb: (state: PosState | null) => void): (() => void) => {
    const listener = (_event: IpcRendererEvent, state: PosState | null): void => cb(state)
    ipcRenderer.on(POS_STATE_CHANNEL, listener)
    return () => {
      ipcRenderer.removeListener(POS_STATE_CHANNEL, listener)
    }
  },
  // The basket test screen: the ledger's readout and the zones it judges under, available with
  // the POS integration off, plus a desk practice session that binds the ledger with no tablet.
  // `null` from the read means the controller is not up yet (no sidecar port).
  getBasketState: (): Promise<BasketViewState | null> => ipcRenderer.invoke('basket:get-state'),
  startBasketPractice: (): Promise<BasketViewState> => ipcRenderer.invoke('basket:practice-start'),
  stopBasketPractice: (): Promise<BasketViewState | null> =>
    ipcRenderer.invoke('basket:practice-stop'),
  onBasketState: (cb: (state: BasketViewState) => void): (() => void) => {
    const listener = (_event: IpcRendererEvent, state: BasketViewState): void => cb(state)
    ipcRenderer.on(BASKET_STATE_CHANNEL, listener)
    return () => {
      ipcRenderer.removeListener(BASKET_STATE_CHANNEL, listener)
    }
  }
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.api = api
}
