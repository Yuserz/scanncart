import { describe, it, expect } from 'vitest'
import { SETTINGS_FIELDS, SETTINGS_GROUPS } from './settingsFields'

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
