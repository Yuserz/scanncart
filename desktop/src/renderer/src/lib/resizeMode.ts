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
export function recordedConfirmation(mode: string): string {
  return (
    `Recorded — these weights now carry resize_mode: ${mode}, so auto uses it and this warning is ` +
    'answered for good: the record sits beside the weights, so it travels with them.'
  )
}
