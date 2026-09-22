import { vi } from 'vitest'
import type { SettingsDeps } from '../hooks/useSidecarSettings'
import type {
  ApiClient,
  DatasetStatusResponse,
  InstalledModel,
  SettingsResponse,
  UnrecordedResizeMode,
  ValidationRecord
} from '../lib/api'

// A populated labeling snapshot, for tests that want the panel past its "no snapshot"
// state. Counts are shaped like the real ones (1,383 images, 969/278/136 by split).
export function datasetStatus(
  overrides: Partial<DatasetStatusResponse> = {}
): DatasetStatusResponse {
  return {
    available: true,
    snapshot_path: 'sidecar/data/datasets/cleaned-v2/label_progress.json',
    generated_at: '2026-09-22T04:35:09',
    age_seconds: 120,
    project: 'snc-grocery',
    total: 1383,
    decided: 40,
    percent: 2.9,
    null_annotations: 0,
    mismatches: 0,
    classes: [
      { slug: 'milo', name: 'Milo Chocolate Drink 22g Sachet', decided: 10, total: 142 },
      { slug: 'safeguard', name: 'safeguard_pure_white_60g', decided: 30, total: 279 }
    ],
    by_distance: { close: [30, 1015], mid: [10, 152], far: [0, 216] },
    by_split: { train: [30, 969], valid: [7, 278], test: [3, 136] },
    // The sessions the project actually holds, as the tool reads them off the image
    // tags: `s1` is the single capture so far and therefore unavoidable across all three
    // splits, `neg1` is the hard negatives in train alone. Shaped like the real snapshot
    // so a test can ask the question the block exists for - is the test number an unseen
    // session or held-out frames of one the model trained on.
    sessions: [
      { name: 's1', train: 969, valid: 278, test: 136, decided: 0, total: 1383, splits: 3 },
      { name: 'neg1', train: 50, valid: 0, test: 0, decided: 0, total: 50, splits: 1 }
    ],
    // The worklist, in the order the sidecar produces it: most-remaining first, and the
    // background pseudo-class last because it is the smallest. Shaped like the real
    // 2026-09-22 snapshot's head (the biggest cells are the `close` ones), with the
    // hard-negative row included because it is the one row whose instruction is the
    // opposite of every other row's - mark each frame null rather than draw on it.
    labeling_backlog: [
      {
        slug: 'silver-swan-vinegar',
        name: 'silver_swan_sukang_puti_200ML',
        distance: 'close',
        decided: 0,
        total: 242,
        remaining: 242,
        background: false
      },
      {
        slug: 'lucky-me-pancit',
        name: 'lucky_me_pancit_canton_calamansi_flavor',
        distance: 'close',
        decided: 0,
        total: 234,
        remaining: 234,
        background: false
      },
      {
        slug: 'safeguard',
        name: 'safeguard_pure_white_60g',
        distance: 'close',
        decided: 30,
        total: 184,
        remaining: 154,
        background: false
      },
      {
        slug: 'century-tuna',
        name: 'century_tuna_flakes_in_oil_155_grams',
        distance: 'close',
        decided: 0,
        total: 150,
        remaining: 150,
        background: false
      },
      {
        slug: 'negative',
        name: '(negative: background frames, nothing to draw)',
        // Empty rather than a distance someone observed: these frames are staged without
        // one on purpose, so a distance here would invent a cell that does not exist.
        distance: '',
        decided: 0,
        total: 50,
        remaining: 50,
        background: true
      }
    ],
    // Tier A's real shape (the tool's 2026-09-22 snapshot): 8 cells, 186 of 253 to shoot,
    // ordered most-remaining-first and internally consistent — target and remaining are the
    // sums of these cells, so a test comparing the list against the summary line is real.
    // Two cells are empty, which is what the block exists to surface.
    tier_a_target: 253,
    tier_a_remaining: 186,
    tier_a_cells_under_target: 8,
    tier_a_cells: [
      { slug: 'century-tuna', distance: 'far', target: 40, have: 0, decided: 0, remaining: 40 },
      { slug: 'century-tuna', distance: 'mid', target: 40, have: 0, decided: 0, remaining: 40 },
      { slug: 'palmolive', distance: 'close', target: 40, have: 0, decided: 0, remaining: 40 },
      {
        slug: 'silver-swan-vinegar',
        distance: 'mid',
        target: 35,
        have: 5,
        decided: 0,
        remaining: 30
      },
      {
        slug: 'lucky-me-pancit',
        distance: 'far',
        target: 27,
        have: 13,
        decided: 0,
        remaining: 14
      },
      { slug: 'palmolive', distance: 'mid', target: 27, have: 13, decided: 0, remaining: 14 },
      { slug: '555-sardines', distance: 'mid', target: 23, have: 17, decided: 0, remaining: 6 },
      { slug: 'lucky-me-pancit', distance: 'mid', target: 21, have: 19, decided: 0, remaining: 2 }
    ],
    ...overrides
  }
}

// The other first-run answer: nobody has run the dataset tooling on this machine yet.
// `available: false` is a 200, not an error, and a panel test has to opt in to progress by
// overriding the fake above rather than getting numbers for free.
export function datasetUnavailable(): DatasetStatusResponse {
  return {
    available: false,
    snapshot_path: 'sidecar/data/datasets/cleaned-v2/label_progress.json',
    generated_at: null,
    age_seconds: null,
    project: null,
    total: 0,
    decided: 0,
    percent: 0,
    null_annotations: 0,
    mismatches: 0,
    classes: [],
    labeling_backlog: [],
    by_distance: {},
    by_split: {},
    sessions: [],
    tier_a_target: 0,
    tier_a_remaining: 0,
    tier_a_cells_under_target: 0,
    tier_a_cells: []
  }
}

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
    // The sidecar's own answer for the default weights: `yolo11n.pt` has no record, so `auto`
    // falls to the format heuristic, which letterboxes. It is deliberately *not* derived from
    // `resize_mode` here — `resolve_resize_mode` has one home (the sidecar), so a test that
    // overrides `resize_mode` to an explicit mode has to say what geometry that mode runs.
    resize_mode_resolved: 'letterbox',
    infer_frame_skip: 0,
    device: 'auto',
    preview_height: 720,
    preview_max_fps: 30,
    track_expiry_s: 1.5,
    class_allowlist: [],
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
    // Nothing is assumed for the default stock weights: `auto` on `yolo11n.pt` is a known answer,
    // not a guess. A test that means to exercise the record-it-now button opts in by overriding
    // this, the same way the installed-weights block does.
    unrecorded_resize_mode: null,
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
      class_names: ['banana'],
      class_warnings: [],
      provider: 'CPUExecutionProvider',
      // The native default reports neither size: its geometry is already on the settings response,
      // so a test that means to exercise the remote geometry line has to opt in by overriding this
      // (the same way the installed-weights block does).
      sent_size: null,
      reported_size: null
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
      controls: { brightness: true, exposure: true, gain: false, focus: false, autofocus: true },
      recommended: { camera_exposure: -6, camera_brightness: 180 },
      measured_at: 1,
      measured: {},
      sweep_version: 1
    })),
    applyCameraProfile: vi.fn(async () => baseSettings()),
    saveSettings: vi.fn(async () => baseSettings()),
    getCameraProfile: vi.fn(async () => ({ profile: null })),
    // Defaults to "no snapshot", which is what a machine that has never run the
    // dataset tooling actually returns - so a test that means to exercise the panel
    // has to opt in by overriding this, rather than getting progress for free.
    getDatasetStatus: vi.fn(async () => datasetUnavailable()),
    // Empty, which is the truthful default on a machine where nothing has been trained
    // locally yet - so the picker shows the built-in list and nothing else. A test that
    // means to exercise the installed-weights block has to opt in by overriding this.
    getModels: vi.fn(async () => ({ stock: [], installed: [], directory: 'models/' })),
    recordResizeMode: vi.fn(async () => ({ stock: [], installed: [], directory: 'models/' })),
    ...overrides
  }
  return { deps: { apiFactory: () => api, healthPollMs: 10_000, retryDelayMs: 10 }, api }
}

// A measured `test` split, with the three outcomes in it: a pass, a miss, and a class the split
// held no instances of. The last one is why this cannot be a flat list of numbers - `null` and
// `0` have to stay distinguishable all the way to the pixels.
export const VALIDATION_V2: ValidationRecord = {
  split: 'test',
  floor: 0.85,
  measured_at: '2026-09-22T10:40:00',
  aggregates: { precision: 0.9, recall: 0.82, mAP50: 0.88, 'mAP50-95': 0.61 },
  per_class: [
    { name: 'bear-brand', recall: 0.9, instances: 10 },
    { name: 'century-tuna', recall: 0.62, instances: 30 },
    { name: 'lucky-me', recall: null, instances: 0 },
    { name: 'milo', recall: 0.95, instances: 12 }
  ],
  // The same four classes measured at two distances, holding all four cell states in one
  // fixture: a pass, a miss at `far` (0.62, under the floor while the split as a whole reads
  // 0.900), a class the distance held no instances of (`null`), and a class the distance did not
  // score at all (absent - no frames of it out there). The last two both render as a dash, and
  // both are why `` is not `0.000`.
  per_distance: [
    {
      distance: 'close',
      images: 34,
      aggregates: { mAP50: 0.93 },
      per_class: [
        { name: 'bear-brand', recall: 0.97, instances: 12 },
        { name: 'century-tuna', recall: 0.88, instances: 20 },
        { name: 'lucky-me', recall: null, instances: 0 },
        { name: 'milo', recall: 0.96, instances: 10 }
      ]
    },
    {
      distance: 'far',
      images: 28,
      aggregates: { mAP50: 0.51 },
      per_class: [
        { name: 'bear-brand', recall: 0.62, instances: 8 },
        { name: 'century-tuna', recall: 0.71, instances: 14 }
      ]
    }
  ]
}

// The 8 class names a v2 weight predicts, as `--install` records them. A fixture mirror like the
// other hand-kept pairs in this repo: the renderer never judges these names (the sidecar does and
// sends sentences), so only the count and the presence matter here.
export const ROSTER_NAMES = [
  'Bear Brand Fortified Powdered Milk 33g',
  'lucky_me_pancit_canton_calamansi_flavor',
  '555 sardines 155grams',
  'century_tuna_flakes_in_oil_155_grams',
  'silver_swan_sukang_puti_200ML',
  'Milo Chocolate Drink 22g Sachet',
  'safeguard_pure_white_60g',
  'Palmolive Naturals Bar Soap 85g'
]

// What `train_v2.py --install` records beside the weights, as the sidecar reports it.
// `auto_resolves_to` equals the requirement because the sidecar's `auto` honours the record, so
// leaving the field alone uses `stretch` and nothing is flagged. Only an explicit value can
// contradict the weights, which is the state both the Admin field and the Live banner warn about.
export function installedV2(overrides: Partial<InstalledModel> = {}): InstalledModel {
  return {
    value: 'models/scanncart-grocery-v2.pt',
    resize_mode: 'stretch',
    auto_resolves_to: 'stretch',
    source: 'snc-grocery version 2',
    class_names: ROSTER_NAMES,
    class_warnings: [],
    recorded: true,
    validation: [VALIDATION_V2],
    ...overrides
  }
}

// A weight whose project class list was split by distance: one product in `close`/`mid`/`far`,
// so 24 outputs and every box under a label the roster does not contain. The names and the
// sentence are what the sidecar reports; the panel renders them as they arrive.
export function installedV2DistanceSplit(): InstalledModel {
  return installedV2({
    class_names: ['Palmolive Naturals Bar Soap 85g close', 'Palmolive Naturals Bar Soap 85g mid'],
    class_warnings: [
      '2 of 2 class name(s) carry a distance, so this model predicts one class per ' +
        'product-and-distance instead of one per product: ' +
        "'Palmolive Naturals Bar Soap 85g close' (close), 'Palmolive Naturals Bar Soap 85g " +
        "mid' (mid). Distance is a tag on the training image (MODEL_TRAINING.md 8.1), never a " +
        'class - so this is a project whose class list was split by distance. Retrain from a ' +
        'version generated with the 8 product names; no setting here fixes it.'
    ]
  })
}

// Installed by the tool, `--val` never run: a record, no numbers.
export function installedV2Unmeasured(): InstalledModel {
  return installedV2({ validation: [] })
}

// What the sidecar reports for a `.pt` with nothing recorded beside it while the setting is
// `auto`: the geometry it falls back to, plus the two sentences (diagnosis and remedy) and the
// mode a one-click record would write. This is the state the Admin panel's button exists for, and
// the state the Live view reports while a capture runs.
export function unrecordedResizeMode(
  overrides: Partial<UnrecordedResizeMode> = {}
): UnrecordedResizeMode {
  return {
    model: 'models/hand-copied.pt',
    resize_mode: 'letterbox',
    warning:
      'models/hand-copied.pt has no record of the geometry it was trained with, so ' +
      'resize_mode=auto assumes letterbox.',
    remedy:
      'Record it if you know these weights are letterbox-trained — `train_v2.py --install` ' +
      'records it from the training run, and the Admin Panel records the same fact from what ' +
      'you know.',
    ...overrides
  }
}

// Copied in by hand: no record at all, so nothing is known - not even a requirement.
//
// Overridable because the same weight is the one a test records a requirement for: after the
// button is clicked the sidecar answers with this entry carrying the mode it just wrote, and the
// panel's confirmation is read from exactly that field.
export function installedUnrecorded(overrides: Partial<InstalledModel> = {}): InstalledModel {
  return {
    value: 'models/hand-copied.pt',
    resize_mode: null,
    auto_resolves_to: 'letterbox',
    source: '',
    // Nothing recorded beside it, so no class list is known either - and "nothing known" must
    // not render as a finding about the names.
    class_names: [],
    class_warnings: [],
    recorded: false,
    validation: [],
    ...overrides
  }
}
