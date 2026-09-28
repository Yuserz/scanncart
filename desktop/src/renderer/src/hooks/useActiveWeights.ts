import { useCallback, useEffect, useState } from 'react'
import {
  createApiClient,
  type InstalledModel,
  type SettingsResponse,
  type UnrecordedResizeMode,
  type ValidationRecord
} from '../lib/api'
import { resizeModeMismatch, type ResizeMismatch } from '../lib/resizeMode'
import type { SettingsDeps } from './useSidecarSettings'

// What is known about the weights the configured model points at: the geometry they run at, what
// `--val` measured on them, whether the settings contradict their record, whether anything
// recorded the geometry at all, and — when there is a record — what it says.
//
// Five facts, one read of `/api/settings` + `/api/models`, because they have to come from the
// same instant. A geometry read from one snapshot and a warning from another could describe two
// different configurations on the same screen — and an amber tile beside a silent banner is
// exactly the kind of contradiction nobody can debug.
//
// Deliberately not a second `useSidecarSettings`: that hook also loads system info, presets and a
// camera profile, two of which are device queries, and `CameraTuning` in the same view already
// fires it. This reads only what the question needs, once, with no polling — the answer can change
// only when someone edits a restart-required field or drops a weights file in, and neither happens
// mid-run. The one exception is the write below, which changes the answer on purpose and re-reads
// both endpoints for exactly that reason.
//
// Only *saved* settings are consulted. `active_model` and `resize_mode` are restart-required, so
// the saved value is what the running detector was built from; there is no draft here to be ahead
// of it, unlike the Admin Panel, which has to compare against what a Save would apply.
export interface WeightsFacts {
  // The mode the local detector is fitted to for these weights, `auto` already resolved by the
  // sidecar from the same call the detector factory makes. `null` when the backend runs someone
  // else's model: a remote backend hands the frame to a workflow that holds its own weights and
  // resizes server-side, so there is no local resize to describe. Callers read that absence as
  // "these weights are not what runs" rather than printing a geometry for a resize that never
  // happens.
  geometry: string | null
  // The setting that produced `geometry`, echoed so a caller can say where it came from ("from
  // auto") without keeping its own copy of the argument.
  setting: string
  // What `--val` measured, preferring the acceptance split. `null` when nothing was recorded — a
  // hand-copied weight, a stock model, or a run whose `--val` has not happened yet — which is not
  // a score of zero, and is shown as "not measured" rather than as a number nobody earned.
  measured: ValidationRecord | null
  // The saved `resize_mode` against the weights' recorded requirement.
  mismatch: ResizeMismatch | null
  // The assumption behind `auto`, when there is one to report: nothing recorded the geometry
  // these weights were trained at, so the letterbox they fall back to is a rule about file
  // formats rather than a fact. `null` in every other case, including a record, an explicit mode,
  // stock weights, and a remote backend.
  //
  // Carried here rather than fetched by a second hook because it arrives on the same settings
  // response the geometry does — the same instant, and the same `null`-when-native-only gate
  // that `resize_guess` applies. It is not a *mismatch*: the settings do not contradict anything,
  // there is nothing to contradict, which is why the comparison next door is silent here and the
  // Live view has to report it itself.
  unrecorded: UnrecordedResizeMode | null
  // What the record beside these weights says they were trained at, or `null` when nothing
  // recorded it. Read from the same listing the comparison above uses — the record is the
  // *evidence*, and it is what makes the letterbox an answer rather than an assumption.
  //
  // The pair with `unrecorded` is exhaustive and ordered: a record (this) and the absence of one
  // that `resize_guess` judged worth reporting (that) cannot both be set, and neither being set
  // means the assumption is not one this app can report — stock weights, whose geometry is
  // known, or a `.onnx`, whose heuristic lands elsewhere.
  required: string | null
}

export interface ActiveWeights extends WeightsFacts {
  // Records the requirement the entry names, then re-reads. The entry itself is the remedy — the
  // model to write for and the mode to write — so nothing here takes arguments: a caller with a
  // different model in mind is looking at a different set of weights.
  record: () => Promise<void>
  // In flight. The button that called this is disabled while it is true, because the write is a
  // file on disk and a second click would only rewrite the same fact.
  recording: boolean
  // A write from *this* view landed, and the sidecar's refreshed answers are the reason the
  // assumption is gone. Deliberately not a mode: the mode is `required` above, read back off the
  // refreshed listing, which is the same fact the Admin Panel's acknowledgement reads and cannot
  // be a claim about a write that went somewhere else.
  justRecorded: boolean
}

// The measurement to report from a weight's record, or null when there is none.
//
// `test` first because that is the acceptance split, and the other recorded split (`valid`) is the
// one training *selected on* — the flattering number. Falls back to whatever is there rather than
// reporting nothing, since the block carries its own `split` and every caller labels the number
// with it: a `valid` recall shown as `valid` is honest, and hiding a measurement that exists is
// not an improvement.
function measuredOn(record: InstalledModel | undefined): ValidationRecord | null {
  const blocks = record?.validation ?? []
  return blocks.find((b) => b.split === 'test') ?? blocks[0] ?? null
}

// One settings response + one weights listing, turned into the facts above — the single
// construction, used by the initial read and by the re-read after a write. Two copies of this
// could answer the geometry from one snapshot and the warning from another, which is the state
// this hook exists to avoid.
function factsOf(settings: SettingsResponse, installed: InstalledModel[]): WeightsFacts {
  const record = installed.find((m) => m.value === settings.active_model)
  return {
    geometry: settings.resize_mode_resolved,
    setting: settings.resize_mode,
    measured: measuredOn(record),
    mismatch: resizeModeMismatch(settings.resize_mode, settings.active_model, installed),
    // `?? null` rather than the field alone: a payload without it (an older sidecar, a partial
    // fixture) has to read as "nothing assumed", which costs a banner — whereas `undefined`
    // reaching the render would cost the view.
    unrecorded: settings.unrecorded_resize_mode ?? null,
    required: record?.resize_mode ?? null
  }
}

const NO_MODELS: InstalledModel[] = []

export function useActiveWeights(port: number, deps: SettingsDeps = {}): ActiveWeights | null {
  const apiFactory = deps.apiFactory ?? createApiClient
  const [facts, setFacts] = useState<WeightsFacts | null>(null)
  const [recording, setRecording] = useState(false)
  const [justRecorded, setJustRecorded] = useState(false)

  // `apiFactory` in the deps rather than a ref, exactly as `useSidecarSettings` does it: the
  // default is a module-level function and the injected ones in tests are stable, so this reads
  // once per mount instead of re-running the effect on every render.
  useEffect(() => {
    let cancelled = false
    const load = async (): Promise<void> => {
      const api = apiFactory(port)
      try {
        const [settings, models] = await Promise.all([
          api.getSettings(),
          // Same tolerance as before: this feeds readouts, so a failure here must not take the
          // Live view down with it. The geometry comes from settings, which are already in hand,
          // so losing `/api/models` costs the score and the warnings but not the whole readout.
          api.getModels().catch(() => ({ stock: [], installed: NO_MODELS, directory: '' }))
        ])
        if (cancelled) return
        setFacts(factsOf(settings, models.installed ?? NO_MODELS))
      } catch {
        // Left at `null` — nothing known. An unreachable sidecar is already visible in the
        // toolbar's connection readout, and a geometry or a score that could not be read must not
        // be invented: either one would send the operator after a setting that is not the problem.
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [port, apiFactory])

  // The one write this hook performs, and the reason it re-reads instead of trusting an answer.
  //
  // The write itself is the same route the Admin Panel's button uses, with the model and the mode
  // taken from the sidecar's *entry* rather than from a caller — so the two views cannot ask for
  // different records, and neither can offer a mode the sidecar did not name.
  //
  // The banner clears because the next settings response no longer reports the assumption, not
  // because a click happened: if the write fails, or lands somewhere this hook cannot read back,
  // the entry is still what the sidecar reports and the warning stays with its button. When the
  // settings re-read and the write disagree, there is nothing to show — so nothing is shown, and
  // `justRecorded` stays false along with it.
  const record = useCallback(async (): Promise<void> => {
    const entry = facts?.unrecorded ?? null
    if (entry === null || recording) return
    setRecording(true)
    try {
      const api = apiFactory(port)
      const models = await api.recordResizeMode(entry.model, entry.resize_mode)
      const settings = await api.getSettings()
      setFacts(factsOf(settings, models.installed ?? NO_MODELS))
      setJustRecorded(true)
    } catch {
      // Deliberately silent, and deliberately without a partial update: the warning above the
      // button is the honest outcome of a write that did not land, and replacing it with an error
      // banner would take away the sentence that says what the button is for. The same rule the
      // Admin Panel keeps — a failed write keeps its warning.
    } finally {
      setRecording(false)
    }
  }, [apiFactory, port, facts, recording])

  return facts === null ? null : { ...facts, record, recording, justRecorded }
}
