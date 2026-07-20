// Mirrors sidecar/app/settings.py::Settings' hardcoded defaults 1:1 — kept in
// sync by hand, same tradeoff already accepted for the WS message contract
// (see CLAUDE.md's testing conventions note). Used by "Restore Defaults" so
// the desktop doesn't need a dedicated reset route on the sidecar.
import type { SettingsPayload } from './api'

export const DEFAULT_SETTINGS: SettingsPayload = {
  active_model: 'yolo11n.pt',
  camera_index: 0,
  capture_width: 1280,
  capture_height: 720,
  capture_fps: 60,
  conf_threshold: 0.5,
  imgsz: 640,
  infer_frame_skip: 0,
  device: 'auto',
  preview_height: 720,
  track_expiry_s: 1.5
}
