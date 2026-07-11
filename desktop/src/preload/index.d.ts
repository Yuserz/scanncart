import { ElectronAPI } from '@electron-toolkit/preload'

export interface SidecarApi {
  getSidecarPort: () => Promise<number | null>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: SidecarApi
  }
}
