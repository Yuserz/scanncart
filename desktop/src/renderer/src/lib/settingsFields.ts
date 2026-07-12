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
  'yolo11x.pt'
] as const
export const ALLOWED_DEVICES = ['auto', 'cpu', 'cuda'] as const

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
    type: 'select',
    options: ALLOWED_DEVICES
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
