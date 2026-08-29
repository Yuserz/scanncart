// Mirrors sidecar/app/settings.py::Settings' hardcoded defaults 1:1 — kept in
// sync by hand, same tradeoff already accepted for the WS message contract
// (see CLAUDE.md's testing conventions note). Used by "Restore Defaults" so
// the desktop doesn't need a dedicated reset route on the sidecar.
import type { SettingsPayload } from './api'

export const DEFAULT_SETTINGS: SettingsPayload = {
  active_model: 'models/scanncart-grocery.onnx',
  camera_index: 0,
  capture_width: 1280,
  capture_height: 720,
  capture_fps: 60,
  conf_threshold: 0.5,
  imgsz: 640,
  infer_frame_skip: 0,
  device: 'auto',
  preview_height: 720,
  track_expiry_s: 1.5,
  detector_backend: 'native',
  roboflow_workspace: 'yusri-caloyloy',
  roboflow_workflow_id: 'scanncart-grocery-vscanncart-grocery-1-yolo11n-t1-logic',
  local_api_url: 'http://127.0.0.1:9001',
  cloud_api_url: 'https://serverless.roboflow.com',
  remote_infer_size: 640,
  remote_timeout_s: 5.0,
  remote_max_retries: 2
}
