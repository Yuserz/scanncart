// Client-side mirror of sidecar/app/settings_store.py's ALLOWED_MODELS /
// ALLOWED_DEVICES + field ranges, kept in sync by hand (same tradeoff as the
// WS message contract — see CLAUDE.md's testing conventions note). The
// server's response is still the source of truth for which fields are
// hot-reloadable vs restart-required; this file only drives form rendering.
import type { SettingsPayload } from './api'

export const ALLOWED_MODELS = [
  'yolo11n.pt',
  'yolo11s.pt',
  'yolo11m.pt',
  'yolo11l.pt',
  'yolo11x.pt',
  'yolo26n.pt',
  'yolo26s.pt',
  'yolo26m.pt',
  // Custom trained models — deploy via: python scanncart/deploy_model.py --name <name>
  'data/custom/grocery-v1.pt',
  'data/custom/grocery-v2.pt',
  'data/custom/experiment-1.pt',
  'data/custom/experiment-2.pt',
  'data/custom/experiment-3.pt',
] as const
export const ALLOWED_DEVICES = ['auto', 'cpu', 'cuda'] as const

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
    'Experimental — needs a discrete CUDA GPU (≥4 GB VRAM); CPU-only machines will fall behind in real time. Weights auto-download on first capture start (internet needed once).',
  'data/custom/grocery-v1.pt':
    'Custom model slot 1 — deploy via: python scanncart/deploy_model.py --name grocery-v1',
  'data/custom/grocery-v2.pt':
    'Custom model slot 2 — deploy via: python scanncart/deploy_model.py --name grocery-v2',
  'data/custom/experiment-1.pt':
    'Experiment slot 1 — deploy via: python scanncart/deploy_model.py --name experiment-1',
  'data/custom/experiment-2.pt':
    'Experiment slot 2 — deploy via: python scanncart/deploy_model.py --name experiment-2',
  'data/custom/experiment-3.pt':
    'Experiment slot 3 — deploy via: python scanncart/deploy_model.py --name experiment-3'
}

export interface FieldMeta {
  key: keyof SettingsPayload
  label: string
  hint: string
  type: 'select' | 'number'
  options?: readonly string[]
  min?: number
  max?: number
  step?: number
}

export const SETTINGS_FIELDS: FieldMeta[] = [
  {
    key: 'active_model',
    label: 'Model',
    hint: 'Larger models (n → x) are more accurate but slower. Match this to your hardware — an underpowered machine will fall behind on frames with a large model.',
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
    label: 'Camera index',
    hint: 'Which camera device to use, if more than one is connected.',
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
    label: 'Camera & Capture',
    keys: ['camera_index', 'capture_width', 'capture_height', 'capture_fps', 'preview_height']
  },
  {
    label: 'Detection & Tracking',
    keys: ['conf_threshold', 'imgsz', 'infer_frame_skip', 'track_expiry_s']
  }
]
