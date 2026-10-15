import { describe, it, expect } from 'vitest'
import { recordedConfirmation, resizeModeMismatch } from './resizeMode'
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

describe('recordedConfirmation', () => {
  it('names the mode that landed, read back rather than echoed from the click', () => {
    const sentence = recordedConfirmation('letterbox', 'local merge merged-v2')
    expect(sentence).toContain('resize_mode: letterbox')
    expect(sentence).toContain('travels with them')
  })

  it('says nothing about provenance when the record names its dataset', () => {
    // Two unknowns must not be one: a record that came from `--install` carries the dataset, and
    // a confirmation that mentioned `source` anyway would train the operator to ignore the clause
    // on the records where it is the only thing standing between them and an empty field.
    expect(recordedConfirmation('letterbox', 'local merge merged-v2')).not.toContain('source')
    expect(recordedConfirmation('letterbox', 'snc-grocery version 2')).not.toContain('source')
  })

  it('reports the field this button cannot write, and both repairs for it', () => {
    // The button attests to the geometry and knows nothing about training runs, so the record it
    // writes names no dataset — and an empty `source` renders as nothing at all in the Model list,
    // no parentheses after the name. Without this sentence the gap is silent on both screens,
    // which is the state the `--install` door's own note exists to end.
    const sentence = recordedConfirmation('letterbox', '')
    expect(sentence).toContain('`source` is left empty')
    expect(sentence).toContain("parentheses after a weight's name")
    // The two writers, the same pair the CLI names: a generated version's number, and the tool
    // that names a local merge from an already-recorded weight.
    expect(sentence).toContain('`train_model.py --install`')
    expect(sentence).toContain('backfill_provenance.py')
    // And the file the CLI's own note names in the same state: with an empty index as well, the
    // sentence says which file was asked, because "no run holds these bytes" and "nowhere was
    // looked" send an operator to two different places.
    expect(sentence).toContain('nothing for them either')
    expect(sentence).toContain('source_index.json')
  })

  it('names the index as the route when the machine already knows the dataset', () => {
    // The case the file exists for: the run that could derive the dataset is gone, and the index
    // was written while it was still here. The gap is still real — the record names nothing — but
    // the remedy is now one command that needs no training run, which is a different sentence.
    const sentence = recordedConfirmation('letterbox', '', 'local merge merged-v2')
    expect(sentence).toContain('`source` is left empty')
    expect(sentence).toContain('`sidecar/data/source_index.json` does name it')
    expect(sentence).toContain('local merge merged-v2')
    expect(sentence).toContain('backfill_provenance.py')
    expect(sentence).not.toContain('nothing for them either')
    expect(sentence).not.toContain('--install')
  })

  it('says nothing about the index for a record that already names its dataset', () => {
    // The index answers only where the record is silent, so the sentence must not mention it at
    // all here — including when the sidecar happens to have an entry for the same bytes, which
    // would otherwise read as a second opinion about a fact that is already written down.
    for (const indexed of ['', 'local merge merged-v2']) {
      const sentence = recordedConfirmation('letterbox', 'snc-grocery version 2', indexed)
      expect(sentence).not.toContain('source_index')
      expect(sentence).not.toContain('is left empty')
    }
  })
})
