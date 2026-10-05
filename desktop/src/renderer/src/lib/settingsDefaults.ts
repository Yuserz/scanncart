// Mirrors sidecar/app/settings.py::Settings' hardcoded defaults 1:1 — kept in
// sync by hand, same tradeoff already accepted for the WS message contract
// (see CLAUDE.md's testing conventions note). Used by "Restore Defaults" so
// the desktop doesn't need a dedicated reset route on the sidecar.
import type { SettingsPayload } from './api'

export const DEFAULT_SETTINGS: SettingsPayload = {
  active_model: 'models/scanncart-grocery-v1.pt',
  camera_index: 0,
  // Mirrors settings.py: USB-2.0-safe mode; 1080p60 needs USB 3.0 (see there).
  capture_width: 640,
  capture_height: 480,
  capture_fps: 30,
  conf_threshold: 0.5,
  imgsz: 640,
  // `auto` reads the geometry recorded beside the selected weights; the picker lists
  // `letterbox` first because that is what the checkout view wants. Mirrors settings.py.
  resize_mode: 'auto',
  infer_frame_skip: 0,
  device: 'auto',
  preview_height: 720,
  preview_max_fps: 30,
  preview_mirror: true,
  suppress_clamped_detections: true,
  // On, like the clamp rule above, and for the same measured reason: without it a fresh install
  // logs Bear Brand on an empty counter. It costs real detections (measured 252 of the 2018 boxes
  // that match their label across the whole v1 export, ~12%), which is the price of that default.
  suppress_frame_filling_detections: true,
  // On, and the cheapest of the three: it drops 6 of the export's 2018 matched detections (0.30%)
  // where the two shape rules above cost 36 (1.8%) and 252 (~12%), and it is what stops an empty
  // counter logging items.
  suppress_unsure_phantoms: true,
  track_expiry_s: 1.5,
  class_allowlist: [],
  detector_backend: 'native',
  roboflow_workspace: 'yusri-caloyloy',
  roboflow_workflow_id: 'scanncart-grocery-vscanncart-grocery-1-yolo11n-t1-logic',
  local_api_url: 'http://127.0.0.1:9001',
  cloud_api_url: 'https://serverless.roboflow.com',
  remote_infer_size: 640,
  remote_timeout_s: 5.0,
  remote_max_retries: 2,
  camera_brightness: null,
  camera_exposure: null,
  camera_autofocus: null,
  camera_focus: null,
  camera_auto_exposure: true
}
