// Client-side mirror of sidecar/app/settings_store.py's ALLOWED_MODELS /
// ALLOWED_DEVICES + field ranges, kept in sync by hand (same tradeoff as the
// WS message contract — see CLAUDE.md's testing conventions note). The
// server's response is still the source of truth for which fields are
// hot-reloadable vs restart-required; this file only drives form rendering.
import type { SettingsPayload } from './api'

// The Roboflow-exported grocery model, run in-process. Listed first because
// it is the default and the only model that detects the actual SKUs; the
// stock YOLO weights below are generic COCO. Mirrors the sidecar's
// CUSTOM_MODEL_DIR convention — any .onnx/.pt under sidecar/models/ is valid,
// this is just the one we ship with.
export const CUSTOM_MODEL = 'models/scanncart-grocery.onnx'

// A raw path is not a label. Anything not listed falls back to its own name.
export const MODEL_LABELS: Record<string, string> = {
  [CUSTOM_MODEL]: 'SCANnCART grocery (custom, 7 SKUs)'
}

export const ALLOWED_MODELS = [
  CUSTOM_MODEL,
  'yolo11n.pt',
  'yolo11s.pt',
  'yolo11m.pt',
  'yolo11l.pt',
  'yolo11x.pt',
  'yolo26n.pt',
  'yolo26s.pt',
  'yolo26m.pt'
] as const
export const ALLOWED_DEVICES = ['auto', 'cpu', 'cuda'] as const

// Mirrors the sidecar's ALLOWED_BACKENDS. 'native' runs the weights in the
// sidecar process; the two remote backends call a Roboflow Workflow over HTTP
// and differ only by base URL. See docs/DETECTOR_BACKENDS.md.
export const ALLOWED_BACKENDS = ['native', 'local_api', 'cloud_api'] as const
export const REMOTE_BACKENDS: readonly string[] = ['local_api', 'cloud_api']

export const BACKEND_LABELS: Record<string, string> = {
  native: 'Native (local weights)',
  local_api: 'Self-hosted API',
  cloud_api: 'Roboflow Cloud'
}

export const BACKEND_HINTS: Record<string, string> = {
  native:
    'Runs the model inside the sidecar. Works fully offline and is the only backend that meets the "no cloud" requirement. Needs the weights file on disk. Speed is GPU-bound, not backend-bound — a GTX 1050 Ti measured ~40 ms/frame, about the same as a local API server on the same PC.',
  local_api:
    'Calls a Roboflow inference server running on this same PC. Stays offline, costs nothing, and measured ~90 ms warm, but adds a second process and an HTTP hop per inference. See docs/DETECTOR_BACKENDS.md to run it without Docker.',
  cloud_api:
    "Calls Roboflow's serverless endpoint. Needs no GPU and no setup, but requires internet, bills per inference, and measured 600-3250 ms per call — demo use, not deployment."
}

// Below this, a slow remote round trip can outlast the tracker's memory and
// re-issue a stationary item's track id, logging one item twice. The floor is
// per-backend because the two are an order of magnitude apart: ~90 ms warm
// against a local inference server, 600-3250 ms against the cloud. Mirrors the
// sidecar's MIN_TRACK_EXPIRY_S_BY_BACKEND.
export const MIN_TRACK_EXPIRY_S_BY_BACKEND: Record<string, number> = {
  local_api: 2.0,
  cloud_api: 5.0
}
// Conservative fallback for a backend not listed above.
export const MIN_REMOTE_TRACK_EXPIRY_S = 5.0

export function minTrackExpiryS(backend: string): number {
  return MIN_TRACK_EXPIRY_S_BY_BACKEND[backend] ?? MIN_REMOTE_TRACK_EXPIRY_S
}

// YOLO26 is offered as an experimental lane: the detector loads it fine, but
// presets and tuning guidance are calibrated for YOLO11. Mirrors the
// sidecar's EXPERIMENTAL_MODELS set.
export const EXPERIMENTAL_MODELS: readonly string[] = ['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt']

// Per-model hardware guidance shown under the Model field while an
// experimental model is selected.
export const MODEL_SPEC_HINTS: Record<string, string> = {
  'yolo26n.pt':
    'Experimental — lightest YOLO26. Needs roughly yolo11n-class hardware: a modern 4-core CPU and 8 GB RAM. Its NMS-free design typically runs faster than yolo11n on CPU. Weights auto-download on first capture start (internet needed once).',
  'yolo26s.pt':
    'Experimental — needs a strong CPU (8+ cores) or an entry CUDA GPU (≥2 GB VRAM) to hold ~30 fps. Weights auto-download on first capture start (internet needed once).',
  'yolo26m.pt':
    'Experimental — needs a discrete CUDA GPU (≥4 GB VRAM); CPU-only machines will fall behind in real time. Weights auto-download on first capture start (internet needed once).'
}

export interface FieldMeta {
  key: keyof SettingsPayload
  label: string
  hint: string
  type: 'select' | 'number' | 'text'
  options?: readonly string[]
  min?: number
  max?: number
  step?: number
}

export const SETTINGS_FIELDS: FieldMeta[] = [
  {
    key: 'active_model',
    label: 'Model',
    hint: 'The grocery model detects your actual SKUs; the yolo* weights are generic COCO and will not. Larger stock models (n → x) are more accurate but slower — match this to your hardware.',
    type: 'select',
    options: ALLOWED_MODELS
  },
  {
    key: 'device',
    label: 'Device',
    hint: 'GPU uses your NVIDIA (CUDA) GPU for faster inference; CPU runs on the processor. GPU is the default when a CUDA GPU is present, and disabled otherwise.',
    // Rendered as a custom GPU/CPU toggle in AdminPanel (the `field.key === 'device'`
    // branch), so `type` here is inert and no `options` list drives its UI.
    type: 'select'
  },
  {
    key: 'camera_index',
    label: 'Camera',
    // Rendered as a dropdown of detected devices when the sidecar can
    // enumerate them; falls back to this number input when it cannot.
    hint: 'Which camera device to use. The resolution shown is what the device opened at — check it matches the camera you expect.',
    type: 'number',
    min: 0,
    max: 8,
    step: 1
  },
  {
    key: 'capture_width',
    label: 'Capture width (px)',
    hint: 'Resolution requested from the webcam. Higher is sharper but slower and uses more USB bandwidth.',
    type: 'number',
    min: 160,
    max: 3840,
    step: 1
  },
  {
    key: 'capture_height',
    label: 'Capture height (px)',
    hint: 'Resolution requested from the webcam. Higher is sharper but slower and uses more USB bandwidth.',
    type: 'number',
    min: 120,
    max: 2160,
    step: 1
  },
  {
    key: 'capture_fps',
    label: 'Capture FPS',
    hint: 'Frames per second requested from the webcam.',
    type: 'number',
    min: 1,
    max: 120,
    step: 1
  },
  {
    key: 'conf_threshold',
    label: 'Confidence threshold',
    hint: 'Minimum confidence to report a detection. Higher means fewer false positives, but may miss partially obscured items.',
    type: 'number',
    min: 0,
    max: 1,
    step: 0.05
  },
  {
    key: 'imgsz',
    label: 'Inference size (px)',
    hint: 'Size each frame is scaled to before detection (square, multiple of 32). Bigger sees small and fast-moving items better — the key lever for catching thrown objects — but raises latency. 640 is the default; 960 is a good accuracy step on a discrete GPU.',
    type: 'number',
    min: 320,
    max: 1920,
    step: 32
  },
  {
    key: 'infer_frame_skip',
    label: 'Frame skip',
    hint: "Skip N frames between inferences. Higher means less CPU/GPU load but staler tracking — pair with a larger track expiry so items aren't marked 'left' between inferences.",
    type: 'number',
    min: 0,
    max: 30,
    step: 1
  },
  {
    key: 'preview_height',
    label: 'Preview height (px)',
    hint: 'Resolution of the JPEG preview streamed to this UI — purely visual, does not affect detection accuracy.',
    type: 'number',
    min: 120,
    max: 1080,
    step: 1
  },
  {
    key: 'detector_backend',
    label: 'Detector backend',
    hint: 'Where inference runs. Native is the target for deployment; the API backends exist so the custom Roboflow model can be used without downloading its weights.',
    type: 'select',
    options: ALLOWED_BACKENDS
  },
  {
    key: 'local_api_url',
    label: 'Self-hosted API URL',
    hint: 'Where the local Roboflow inference server is listening. Start it with `inference server start`.',
    type: 'text'
  },
  {
    key: 'cloud_api_url',
    label: 'Cloud API URL',
    hint: "Roboflow's serverless endpoint. Must be https.",
    type: 'text'
  },
  {
    key: 'roboflow_workspace',
    label: 'Roboflow workspace',
    hint: 'Workspace slug that owns the workflow.',
    type: 'text'
  },
  {
    key: 'roboflow_workflow_id',
    label: 'Roboflow workflow ID',
    hint: 'Workflow slug (not the document id) to run for each frame.',
    type: 'text'
  },
  {
    key: 'remote_infer_size',
    label: 'Transmit size (px)',
    hint: 'Frames are downscaled to this longest edge before being sent. The model infers at 640 regardless, so larger values only cost bandwidth.',
    type: 'number',
    min: 128,
    max: 1920,
    step: 32
  },
  {
    key: 'remote_timeout_s',
    label: 'Request timeout (seconds)',
    hint: 'How long to wait for one inference before giving up. Measured cloud calls ranged 0.6-3.3 s.',
    type: 'number',
    min: 0.1,
    max: 60,
    step: 0.5
  },
  {
    key: 'remote_max_retries',
    label: 'Max retries',
    hint: 'Retries on timeout or server error only — never on an auth or bad-request failure. Each retry adds latency, so keep this low.',
    type: 'number',
    min: 0,
    max: 5,
    step: 1
  },
  {
    key: 'track_expiry_s',
    label: 'Track expiry (seconds)',
    hint: "How long a track can go undetected before it's logged as 'left'. Too low drops items during brief occlusion or frame skips; too high keeps stale items lingering.",
    type: 'number',
    min: 0.1,
    max: 30,
    step: 0.5
  }
]

export interface FieldGroup {
  label: string
  keys: (keyof SettingsPayload)[]
}

export const SETTINGS_GROUPS: FieldGroup[] = [
  { label: 'Model & Device', keys: ['active_model', 'device'] },
  {
    label: 'Roboflow API backends',
    keys: [
      'local_api_url',
      'cloud_api_url',
      'roboflow_workspace',
      'roboflow_workflow_id',
      'remote_infer_size',
      'remote_timeout_s',
      'remote_max_retries'
    ]
  },
  {
    label: 'Camera & Capture',
    keys: ['camera_index', 'capture_width', 'capture_height', 'capture_fps', 'preview_height']
  },
  {
    label: 'Detection & Tracking',
    keys: ['conf_threshold', 'imgsz', 'infer_frame_skip', 'track_expiry_s']
  }
]
