import { vi } from 'vitest'
import type { SettingsDeps } from '../hooks/useSidecarSettings'
import type { ApiClient, SettingsResponse } from '../lib/api'

export function baseSettings(overrides: Partial<SettingsResponse> = {}): SettingsResponse {
  return {
    active_model: 'yolo11n.pt',
    camera_index: 0,
    capture_width: 1280,
    capture_height: 720,
    capture_fps: 60,
    conf_threshold: 0.5,
    imgsz: 640,
    resize_mode: 'auto',
    infer_frame_skip: 0,
    device: 'auto',
    preview_height: 720,
    preview_max_fps: 30,
    track_expiry_s: 1.5,
    detector_backend: 'ultralytics',
    roboflow_workspace: '',
    roboflow_workflow_id: '',
    local_api_url: '',
    cloud_api_url: '',
    remote_infer_size: 640,
    remote_timeout_s: 10,
    remote_max_retries: 3,
    camera_brightness: null,
    camera_exposure: null,
    camera_autofocus: null,
    camera_focus: null,
    hot_reloadable_fields: ['infer_frame_skip', 'preview_height', 'track_expiry_s'],
    restart_required_fields: ['active_model', 'device'],
    warnings: [],
    roboflow_api_key_present: false,
    ...overrides
  }
}

export function makeDeps(overrides: Partial<ApiClient> = {}): {
  deps: SettingsDeps
  api: ApiClient
} {
  const api: ApiClient = {
    health: vi.fn(async () => ({ state: 'idle', active_model: 'yolo11n.pt', device: 'cpu' })),
    start: vi.fn(),
    stop: vi.fn(),
    getLogs: vi.fn(),
    getSettings: vi.fn(async () => baseSettings()),
    updateSettings: vi.fn(async (patch) => baseSettings(patch)),
    getSystemInfo: vi.fn(async () => ({
      cpu_count: 8,
      ram_gb: 16,
      cuda_available: false,
      accelerator: 'cpu' as const,
      gpu_name: null,
      gpu_vram_gb: null,
      recommended_preset: 'mid_range'
    })),
    getPresets: vi.fn(async () => ({
      presets: [{ name: 'mid_range', label: 'Mid', description: 'd', settings: {} }],
      recommended: 'mid_range'
    })),
    applyPreset: vi.fn(async (name) => baseSettings({ active_model: `${name}.pt` })),
    getCameras: vi.fn(async () => ({
      cameras: [{ index: 0, name: 'Fake Cam', width: 1280, height: 720 }],
      probed: true,
      detail: ''
    })),
    probeDetector: vi.fn(async () => ({
      backend: 'ultralytics',
      reachable: true,
      detail: 'ok',
      latency_ms: 10,
      class_names: ['banana']
    })),
    getCameraQuality: vi.fn(async () => ({
      available: false,
      brightness: 0,
      contrast: 0,
      sharpness: 0,
      capture_fps: 0,
      target_fps: 0,
      verdicts: {},
      detail: ''
    })),
    calibrateCamera: vi.fn(async () => ({
      device_key: 'Fake Cam:0:1280x720',
      backend: 'msmf',
      width: 1280,
      height: 720,
      fps_auto_exposure: 12.3,
      fps_capped_exposure: 30.3,
      controls: { brightness: true, exposure: true, gain: false, focus: false },
      recommended: { camera_exposure: -6, camera_brightness: 180 },
      measured_at: 1
    })),
    applyCameraProfile: vi.fn(async () => baseSettings()),
    saveSettings: vi.fn(async () => baseSettings()),
    getCameraProfile: vi.fn(async () => ({ profile: null })),
    ...overrides
  }
  return { deps: { apiFactory: () => api, healthPollMs: 10_000, retryDelayMs: 10 }, api }
}
