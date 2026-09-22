import { describe, it, expect } from 'vitest'
import { resizeModeMismatch } from './resizeMode'
import { installedUnrecorded, installedV2 } from '../test/fakes'

describe('resizeModeMismatch', () => {
  it('is silent when the settings already say what the weights need', () => {
    expect(resizeModeMismatch('stretch', installedV2().value, [installedV2()])).toBeNull()
  })

  it('is silent on `auto`, which cannot be wrong', () => {
    // The reason this function does not simply compare the setting against the requirement: the
    // sidecar's `auto` honours the record, so `auto` is resolved *through* the requirement. A
    // naive `mode !== required` would flag the one configuration the docs tell you to use.
    expect(resizeModeMismatch('auto', installedV2().value, [installedV2()])).toBeNull()
  })

  it('reports both halves of an explicit contradiction', () => {
    // `required` alone would leave the caller to re-supply the setting, and the two are only
    // meaningful together - this is what a banner says out loud.
    const mismatch = resizeModeMismatch('letterbox', installedV2().value, [installedV2()])
    expect(mismatch).toEqual({ required: 'stretch', mode: 'letterbox' })
  })

  it('is silent for weights with no record, because nothing is required of them', () => {
    // A hand-copied `.pt` is not "mismatched" with anything. Inventing a requirement here would
    // put a warning in front of an operator about weights whose training geometry nobody wrote
    // down - and no setting could clear it.
    expect(
      resizeModeMismatch('letterbox', 'models/hand-copied.pt', [installedUnrecorded()])
    ).toBeNull()
    expect(
      resizeModeMismatch('stretch', 'models/hand-copied.pt', [installedUnrecorded()])
    ).toBeNull()
  })

  it('is silent for a selected model that is not on disk at all', () => {
    // A stored path pointing at nothing has to stay visible in the picker (that is how a broken
    // config gets diagnosed), and it must not produce a geometry warning on the way.
    expect(resizeModeMismatch('stretch', 'models/gone.pt', [installedV2()])).toBeNull()
  })

  it('matches the record by exact picker value, not by filename', () => {
    // The values the picker offers are `models/<name>`, which is what `active_model` stores. A
    // comparison against a bare filename would silently find no record and warn about nothing.
    const installed = [installedV2()]
    expect(resizeModeMismatch('letterbox', 'scanncart-grocery-v2.pt', installed)).toBeNull()
  })
})
