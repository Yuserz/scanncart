import type { InstalledModel } from './api'

// A `resize_mode` setting that contradicts the weights' recorded requirement.
//
// `mode` is the setting that was compared (echoed back so a caller can name it without keeping
// its own copy of the argument), `required` is what the record beside the weights says.
export interface ResizeMismatch {
  required: string
  mode: string
}

// The mode the selected weights require when the setting contradicts it, else `null`.
//
// `auto` is resolved through the sidecar's own answer (`auto_resolves_to`) rather than by
// reimplementing `resolve_resize_mode` here. Since that rule *honours* the recorded requirement,
// `auto` cannot itself be the mismatch — which is why the only way to reach a result from here is
// an explicit value that disagrees with the weights.
//
// One function, two views: the Admin Panel passes the *pending* value so the warning clears as
// the fix is typed, and the Live view passes the running one. Two copies of this comparison would
// eventually disagree about what "mismatched" means, and the two screens would contradict each
// other about the same run.
export function resizeModeMismatch(
  mode: string,
  selectedModel: string,
  installed: InstalledModel[]
): ResizeMismatch | null {
  const record = installed.find((m) => m.value === selectedModel)
  const required = record?.resize_mode ?? null
  // No record means no requirement — a hand-copied weight is not "mismatched" with anything, and
  // inventing one would make the warning appear for weights whose training geometry is unknown.
  if (!required) return null
  const effective = mode === 'auto' && record ? record.auto_resolves_to : mode
  return effective === required ? null : { required, mode }
}

// What both views say once a record lands, for the same reason the comparison above is shared: the
// same write described twice would eventually be described differently, and the two screens would
// disagree about whether it had happened.
//
// It is prose rather than a flag because the acknowledgement is not a claim about the *value* — the
// assumption and the record name the same mode, which is why recording cannot change what the
// detector does. What is settled is that the weights now carry a requirement instead of relying on
// a rule about file formats, and that the record travels with them.
//
// The mode is an argument rather than read from the caller's own state so a call site cannot pass
// the mode it *asked* for while the sidecar recorded something else. Both callers read it back off
// the refreshed weights listing.
//
// `source` is the record's own provenance as that same listing reports it now — which dataset the
// weights came from, `''` when nothing names one. It is a second argument because this write
// *cannot* fill it: the button attests to the geometry, and only a training run ever knew the
// dataset. An acknowledgement that stopped at "recorded" would leave the operator believing the
// record is complete, and an empty `source` renders as nothing at all in the Model list — no
// parentheses after the name — so the gap would be silent on both screens.
//
// `indexed` is the third answer, read off the same listing: what this machine's source index says
// trained these bytes, which the sidecar reports only where the record is silent. It changes the
// sentence because it changes the remedy — the gap is one `backfill_provenance.py` run wide rather
// than waiting on a training run that may not exist any more — and it is the one case where this
// confirmation can say a missing fact is *recoverable* instead of only missing. The index is
// written while a run is still on disk (`app/source_index.py`), so it is what survives a pruned
// run, which is the case both of the record's own sources fail in.
//
// The two repairs are the ones the CLI's `--install` note names (`--version` for a generated
// version, `backfill_provenance.py` for a local merge), because both doors write this one field
// and must not describe it differently. `backfill_provenance.py` is reachable *because* this
// button wrote a record: it leaves a weight with no record alone, since writing one would silence
// the app's own geometry warning.
export function recordedConfirmation(mode: string, source: string, indexed = ''): string {
  const recorded =
    `Recorded — these weights now carry resize_mode: ${mode}, so auto uses it and this warning is ` +
    'answered for good: the record sits beside the weights, so it travels with them.'
  if (source) return recorded
  const gap =
    ' Its `source` is left empty, though: nothing in it names the dataset these weights came ' +
    "from — the field the Model list shows in parentheses after a weight's name — because this " +
    'button records the geometry and knows nothing about training runs.'
  if (indexed) {
    return (
      recorded +
      gap +
      // Assembled from plain literals rather than one template: the path is markdown-style code in
      // an operator-facing sentence, and a template literal would have to escape every backtick.
      ' The index at `sidecar/data/source_index.json` does name it, though (' +
      indexed +
      '), so `sidecar/tools/backfill_provenance.py` records it from there — that file is written ' +
      'while the run is still on disk, so no training run has to still be around.'
    )
  }
  return (
    recorded +
    gap +
    ' `train_model.py --install` writes it from the run that trained them (`--version N` when ' +
    'that run was a generated version), and `sidecar/tools/backfill_provenance.py` names a local ' +
    "merge by matching the weights' bytes to it. The index at `sidecar/data/source_index.json` has " +
    'nothing for them either, so the run is the only thing left that can name it.'
  )
}
