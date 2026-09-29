import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type { SidecarHealth } from '../main/sidecarHealth'

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
