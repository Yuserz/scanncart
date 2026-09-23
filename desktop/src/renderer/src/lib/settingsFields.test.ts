import { describe, it, expect } from 'vitest'
import { DEFAULT_SETTINGS } from './settingsDefaults'
import { SETTINGS_FIELDS, SETTINGS_GROUPS } from './settingsFields'

//: Keys that render somewhere other than a `SETTINGS_GROUPS` group, on purpose, with where.
//:
//: `detector_backend` needs radio buttons, a per-backend hint, the API-key and track-expiry
//: warnings and the Test Connection button, so AdminPanel gives it its own section
//: (`data-testid="backend-picker"`) instead of a row. Listing it here is what keeps "rendered
//: outside a group" a decision rather than an oversight — and it is not a formality: a check that
//: demanded `detector_backend` be grouped would fail on correct code, and the natural fix for a
//: failing test is to delete it.
const BESPOKE_RENDERERS: readonly string[] = ['detector_backend']

describe('field placement', () => {
  it('gives every group a home', () => {
    for (const g of SETTINGS_GROUPS) {
      expect(['live', 'admin']).toContain(g.home)
    }
  })

  it('places every field in exactly one group', () => {
    const seen = SETTINGS_GROUPS.flatMap((g) => g.keys)
    expect(new Set(seen).size).toBe(seen.length)
  })

  it('describes every grouped key', () => {
    // A key in a group with no FieldMeta renders as nothing at all.
    for (const key of SETTINGS_GROUPS.flatMap((g) => g.keys)) {
      expect(SETTINGS_FIELDS.find((f) => f.key === key)).toBeDefined()
    }
  })

  it('renders every settings key somewhere — a group or a known bespoke section', () => {
    // The other half of the invariant this file states at the top: CLAUDE.md promises a field
    // "cannot appear in both or neither", but only *both* was ever checked (the duplicate test
    // below). Nothing checked that a key appears at all, so one added to `SettingsPayload` and to
    // `SETTINGS_FIELDS` yet forgotten in `SETTINGS_GROUPS` renders in neither view, silently, with
    // the whole suite green.
    //
    // `DEFAULT_SETTINGS` is the honest key list to check against rather than a second hand-written
    // array: it is annotated `SettingsPayload`, so a payload field missing from it is already a
    // typecheck error. The two lists cannot drift, which is the only reason this test can be
    // trusted as a completeness check.
    const grouped = new Set<string>(SETTINGS_GROUPS.flatMap((g) => g.keys))
    const unaccounted = Object.keys(DEFAULT_SETTINGS).filter(
      (key) => !grouped.has(key) && !BESPOKE_RENDERERS.includes(key)
    )

    expect(
      unaccounted,
      `not in any group and not in BESPOKE_RENDERERS, so nothing renders it: ${unaccounted}`
    ).toEqual([])
  })

  it('keeps the bespoke-rendered keys out of the groups', () => {
    // Otherwise a key is rendered twice — a radio picker and a group row for one setting — and the
    // duplicate check above cannot see it, because that only looks *within* the groups.
    const grouped = SETTINGS_GROUPS.flatMap((g) => g.keys) as string[]

    for (const key of BESPOKE_RENDERERS) {
      expect(grouped, `${key} is rendered by a bespoke section as well`).not.toContain(key)
    }
  })

  it('keeps the fields that need a device reopen in admin', () => {
    const liveKeys = SETTINGS_GROUPS.filter((g) => g.home === 'live').flatMap((g) => g.keys)
    for (const key of ['camera_index', 'capture_width', 'capture_height', 'capture_fps', 'imgsz']) {
      expect(liveKeys).not.toContain(key)
    }
  })

  it('puts the five tunable fields on live', () => {
    const liveKeys = SETTINGS_GROUPS.filter((g) => g.home === 'live').flatMap((g) => g.keys)
    for (const key of [
      'conf_threshold',
      'camera_brightness',
      'camera_exposure',
      'camera_autofocus',
      'camera_focus'
    ]) {
      expect(liveKeys).toContain(key)
    }
  })

  it('puts only slider-renderable fields on live', () => {
    // The tuning card draws a range input or a checkbox and nothing else, so a
    // text/list/select field placed on live renders as a slider bound to a
    // non-numeric value. class_allowlist shipped that way once.
    const renderable = ['number', 'boolean']
    for (const group of SETTINGS_GROUPS.filter((g) => g.home === 'live')) {
      for (const key of group.keys) {
        const field = SETTINGS_FIELDS.find((f) => f.key === key)
        expect(renderable, `${key} sits in the live group "${group.label}"`).toContain(field?.type)
      }
    }
  })

  it('keeps the class allowlist in admin', () => {
    const liveKeys = SETTINGS_GROUPS.filter((g) => g.home === 'live').flatMap((g) => g.keys)
    expect(liveKeys).not.toContain('class_allowlist')
  })

  it('warns about the exposure framerate trap in the hint', () => {
    const exposure = SETTINGS_FIELDS.find((f) => f.key === 'camera_exposure')
    expect(exposure?.hint).toMatch(/fps|framerate/i)
  })
})
