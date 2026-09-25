# v1's recapture is stopped — why, and what carries forward

**Decision (2026-09-26).** The v1 weight line (`scanncart-grocery`, seven classes) will not be
re-shot and retrained. It stays as shipped. The plan for it lives on the `v1-local-build` branch and
is kept there as a record rather than an intention:
`docs/superpowers/plans/2026-09-26-v1-varied-size-recapture.md` (branch tip `4b15716`).

## What was true, and is still true

The case for the recapture was measured, not assumed, and none of the measurements expire:

- **The recall gap is crowding, not size.** `audit_recall.py` against v1's 327-frame test split:
  **265/265 (100%)** of single-object instances found, **103/136 (75.7%)** of instances in frames
  holding two or more. Not one solo instance was missed, so the aggregate the 0.85 floor is
  computed over is entirely the crowded bucket pulling it down.
- **An empty counter produces a detection.** 25 of 50 stored empty-counter frames fired at
  `conf 0.5`; **19 of those 25 were clamped full-frame boxes**, against **0 of 60** real product
  frames. It is not one class misbehaving (Milo 16, Bear Brand 6, century tuna 3), and the runtime
  `suppress_clamped_detections` filter stops 19 of the 25 — the other 6 sit 0.0124–0.0286 from the
  nearest edge, inside the band real detections occupy, so the rule cannot be tightened without
  discarding real items.
- **Framing is not the lever.** Letterbox vs stretch changes the pixels an object is given, not
  what the weights learned. v1's export is 1,815 frames all at 640×640 with no constant border,
  i.e. stretched, which is why `resize_mode: auto` honours the recorded requirement on that branch
  and stays there.

## Why it was stopped anyway

The measured defect is real; the cost of fixing *this* one stopped being worth it:

1. **Any fix is data, and data is a session.** A capture sitting, a labelling sitting, a version
   generation and a full retrain — with the crowded shots as the only bucket that moves a number,
   because solo coverage is already complete.
2. **The dataset workspace is gone from the machine.** `sidecar/data/datasets/` — the frozen v1
   export, the manifests, the tools' own `.env` — no longer exists here, and v1's Roboflow version
   is recorded as being in Trash (`clean_v2.py sanity`'s warning). So the pipeline starts with a
   re-ingest that may not be possible, and *that* setup is the expensive part rather than the
   shooting.
3. **Added images arrive as a new generation.** A Roboflow version is frozen: the number is spent,
   the geometry is baked, the class list is fixed. The retrain was therefore never a v1 patch — it
   was a refresh that would land as a new version of a project, which is the v2 path with an extra
   step.
4. **v1 is one class short by construction.** Its seven names cannot include Palmolive, so its head
   can never predict everything the roster knows. A model trained now would be born superseded.

## What carries forward

- **On `main`:** the letterbox-default decision with stretch kept as an explicit experimental
  choice, the Live overlay labelling every box with its decoded pixel size, and the docs and
  settings that agree with both (`d3c726d`, restored from the stashed main-branch WIP). The root
  `data/` runtime log is ignored rather than tracked (`30dd76d`).
- **On `v1-local-build`** (pushed, tip `4b15716`): the auditor's `--dataset-dir`/`--manifest`
  overrides and `--size-histogram`, and the recapture plan itself.
- **The tools, which outlive the model.** `audit_recall.py`'s crowding split is what turned "the
  model misses things" into "the model misses *the second* thing" — a finding that decides between
  more solo shots (useless here) and more crowded scenes (the fix). `--size-histogram` derives a
  shoot list from an export's labels alone, with no weight and no GPU. Both are weight-agnostic.
- **Four findings for whoever rebuilds a dataset**, v2 or otherwise: shoot crowded scenes first and
  put the null-marked negatives in the *same* sitting (a dropped negative set leaves no trace, while
  the phantom it fixes ships); the negative defect is a property of border-pinned training labels
  rather than of one class; a runtime clamp filter masks the loudest phantoms and cannot reach the
  rest; and no framing change substitutes for any of it.

## What does not carry forward

- The plan's shoot list and its per-tier counts — they describe a set that will not be shot.
- `auto` as v1's default. That was a v1-specific choice, because v1's ONNX was trained stretched and
  the recorded requirement makes `auto` correct for it. `main` now defaults to letterbox, and
  stretch is opt-in and labelled experimental.
