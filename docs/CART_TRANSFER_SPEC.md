# Plan: camera-first cart additions and removals

Status: **Implemented in shadow mode; not yet measured on real footage** · updated 2026-10-03

Shipped on the desktop: the transfer state machine (`desktop/src/main/transferState.ts`), the zone layouts (bands from one frame edge, `transferGeometry.presetRegions`, the default for a camera under the handle; or drawn outlines, `layoutRegions`, for a camera looking down into the basket), the persistent ledger and review list (`basketLedger.ts`) and the stream tracker that feeds them from fresh, unmirrored sidecar frames (`transferStream.ts`; the sidecar marks `fresh`/`mirrored` on every frame message). The ledger runs on every session as a **shadow** beside the counter; `cartMode: basket` in `pos.json` makes it the posted cart, and that switch waits on Gate C. The pushcart-web half is in (migration `20261003120000_pos_basket_automation.sql`): customers cannot add or remove (no tablet buttons, `403` routes, restrictive RLS on a cart in an open session), the sync's `pending_review` is stored on the session and refuses Finish, and `POS_STAFF_PIN` enables a decrease-only staff removal as the fallback. The whole loop — synthetic frames through the tracker, ledger and orchestrator into a live pushcart-web, read back as the tablet reads it — was run once against the local stack. The §3 gates are still the go/no-go: nothing here has been run against Gate A footage.

## 1. Goal and existing behavior

Demonstrate a stationary, supervised basket that identifies a product, adds it after a completed deposit, retains it while hidden, and removes it after a completed retrieval. Prefer the existing camera and model; add training or hardware only for a demonstrated failure.

The current scanner is different:

| Current counter snapshot | Proposed basket ledger |
|---|---|
| `desktop/src/main/cartState.ts` counts concurrent committed tracks; `posSession.ts` sends full desired snapshots | Confirmed inward/outward events change persistent per-SKU quantities |
| Defaults in `posConfig.ts`: 3 s addition dwell, 10 s removal settle | Completion evidence, not disappearance, authorizes a change |
| A sustained lower visible count eventually removes an item | Concealment/reappearance alone never changes inventory |

`/api/logs` contains track entry/leave and class/confidence, not trajectories. The [scenario fixture README](../desktop/src/main/__fixtures__/scenarios/README.md) reports no real recorded fixtures; synthetic tests do not prove camera accuracy. The proposed ledger and review behavior below are requirements, not shipped capabilities.

**Recommendation:** keep the seven-class grocery detector for identity; start with tracking plus a deterministic transfer state machine for direction/completion. Do not create `SKU-add`/`SKU-remove` classes or treat a hand gesture as a transaction. A temporal model is a later option, not the starting dependency.

## 2. Constrained demo setup and completion rule

- Existing StreamCam, local PC/native weights and customer UI; fixed oblique camera view of a marked loading opening, cart stationary.
- Operator verifies an empty basket before Start. One product unit at a time, label readable, deliberate motion with a readable pause. Identical units pass separately.
- Mark outside, transition and inside regions in true/unmirrored coordinates; reflect them only for the preview. Recalibrate after mount/resolution changes.
- Keep the landing area visible until deposit completion. Products may be hidden **after** confirmation. For removal, expose the product inside before retrieving it through the marked opening.

A 2D region crossing is not proof of crossing the basket rim in depth. The selected view must show these endpoints:

| Outcome | Observable completion for this demo |
|---|---|
| Deposit | Identified product travels outside → opening → inside, is visibly released onto the basket support, and remains supported while the hand separates/withdraws |
| Removal | Identified product travels inside → opening → outside, clears the basket, and stays wholly outside in the presentation area |
| No transfer | Hovering, outside presentation, empty-hand reach, internal rearrangement, temporary occlusion or reversal before either endpoint |

Use a **provisional one-second endpoint hold** during capture/rehearsal: after release for deposit, after clearance for removal. Completion time is the end of that hold. Tune on development footage if necessary, then freeze it before acceptance. (Development value: 0.4 s since the sidecar reached 50–60 fresh inferences a second — `DEFAULT_TRANSFER_CONFIG` — not yet frozen.) A product held inside without visible release is not a completed deposit. Retrieval may remain in the customer's hand; it need not be released outside. Returning an item after completed retrieval is a new deposit, not an aborted removal.

Missing/ambiguous endpoint evidence means no automatic commit. These rules require neither a particular hand model nor a claim that YOLO currently detects release: Gate B must demonstrate that the chosen visual cues can be implemented reliably.

**Excluded:** moving carts, free throws, bundles, opaque bags, simultaneous exchanges and loading outside the marked opening. Gentle drops may be supported only after separate visibility and acceptance trials. For an **observed** unresolved interaction, retain inventory and flag review; a completely invisible interaction cannot reliably trigger an alert. Disclose these restrictions in the capstone presentation.

## 3. Execution checklist

All counts and thresholds below are **proposed demo gates**, not achieved results. Do not buy sensors or start action-model training before the first gate.

### Gate A — human visibility, ordinary video only

Record the actual scanner camera view using an ordinary video recording method; no trajectory recorder, new detector code or instrumented cart is required. Keep several seconds before/after each interaction and preserve original video rather than only annotated preview screenshots. Any independent observer/reference view supplies ground truth, not evidence available to the scanner.

| Pilot trials | Coverage |
|---|---|
| 30 deposits + 30 removals | All seven SKUs, at least two people and two setup sessions; intended orientations, both hand approaches and partially filled basket |
| 30 no-transfers | Five each: outside presentation, hover, reach-only, rearrangement, partial crossing/reversal, occlusion/reappearance |

For each clip, a human reviewer marks SKU readability, ordered path, release/clearance, hold completion time, actual inventory before/after and any blind interval. Have a second reviewer resolve ambiguous clips. Save the mount photograph/sketch, clip IDs and verdicts.

**Pass:** every supported transfer's path and completion are distinguishable from no-transfer footage, and at least 57/60 transfers have a readable SKU. List all failures; unreadable samples are not silently removed. This justifies attempting the baseline, **not camera-only feasibility**. If completion is hidden, re-aim or narrow the path and repeat the pilot before software work. If identity alone fails, retain those clips for Gate B diagnosis.

The ready-to-print trial execution sheet, per-clip scorecard tables, and review rubric are in [GATE_A_RUN_SHEET.md](./GATE_A_RUN_SHEET.md).


### Gate B — detector/tracking proof, then integration

This is future implementation work, not part of the planning pass:

1. Add fresh per-inference evidence capture and a transfer state machine. Start with existing grocery weights; use pilot/development footage to tune association, identity stability, endpoint cues and thresholds.
2. Require stable identity across multiple fresh observations and ordered origin/transition/destination evidence. An initial rule is two fresh observations per side plus visible completion; adjust on development data and freeze before Gate C. Historic maximum confidence or a raw tracker ID alone is insufficient.
3. Candidate states: observed → inbound/outbound pending → confirmed, aborted or uncertain. Commit **only after observed completion**, once per physical transfer. Stitch fragmented tracks conservatively; incompatible associations require review.
4. Run the baseline in shadow mode against human inventory before connecting it to the customer cart. Then verify full-snapshot forwarding, manual overrides and recovery as described in §4.

Evidence capture must distinguish source frame/sequence, acquisition time when available, inference completion and UI update. `sidecar/app/pipeline.py` reuses old boxes in `emit_preview`; those preview emissions are not fresh observations. Current WS timestamps are emission/completion times, not guaranteed camera acquisition times. Record class/confidence, boxes and suppression decisions: filters may reject genuine close transfer views. Replay at every recorded frame is not proof of performance at the live inference cadence; measure fresh observations and gaps under actual load.

Do not collect a large training set until errors are categorized:

| Failure | Next action |
|---|---|
| Human cannot see completion/direction | Change view/path; training cannot recover absent evidence |
| SKU visible but detector misses/misclassifies it | Consider grocery fine-tuning using product-box-labelled video frames and existing static controls |
| Good identity but fragmented tracks/duplicate direction decisions | Fix association/state logic first |
| Observable interactions still defeat that baseline | Consider a separate temporal model on video clips or trajectories; compare on untouched recordings |

For temporal training, preserve continuous clips and label outcome, SKU/quantity, start/completion times, initial/final inventory and occlusion/reversal flags. Hand labels are needed only if the selected auxiliary model uses them. Keep all frames/clips from one recording in one split; hold out sessions and people. Pilot counts are not a sufficient training-set guarantee. If time/data are insufficient, constrain the demo rather than train on acceptance footage.

### Gate C — frozen held-out rehearsal

Use a new session/day and at least one participant absent from tuning. Freeze camera placement, model and thresholds. Record every attempt, including misses, abstentions and retries.

| Trial | Proposed pass condition |
|---|---|
| 35 deposits + 35 removals, five per SKU per direction | At least 34/35 correct per direction; no premature, wrong-SKU/direction, duplicate or negative-count commits |
| 30 no-transfers, same six categories as Gate A | Zero ledger changes; include 30 s concealment after confirmed deposit |
| Five consecutive scripted shopping sessions | Exact intermediate and final per-SKU quantities without manual corrections; include two identical units, remove one, remove/re-add, rearrange and hide |
| At least three probes each: bundle, throw, hidden crossing | No invented SKU/quantity/direction; review for observed unresolved interactions, explicitly record invisible blind spots |
| Camera interruption/restart; sync retry and browser reload; interaction after manual override; Finish while pending | No duplicate/silent loss of confirmed state; blind interval requires review, override is preserved, pending interaction blocks Finish |

**Scoring:** match one committed event to one ground-truth event only in **[completion, completion + 5 s]**, using the frozen endpoint rule in §2. A commit before completion is a false event and fails the gate, even if the item is subsequently deposited. Require correct SKU, direction and quantity together. Duplicates/unmatched commits are false events; misses/abstentions count against automatic success. A retry does not erase its original failure.

Report median/p95 completion-to-correct-customer-UI latency; proposed p95 budget is **≤5 s** for successful events, with misses separately included in the success denominator. Record event/UI times on a common clock or document the clock mapping. State whether session trials overlap the event sample; never count overlapping events twice.

Archive footage, ground truth, frozen configuration/model/calibration, fresh-inference evidence, events, UI timing and failure counts. After fixes, use a new acceptance set. This small rehearsal supports only the guided demo—not a 99% reliability claim or unrestricted store deployment.

## 4. Minimal ledger and integration requirements

- Sidecar owns transfer candidates, persistent per-SKU ledger and confirmed-event journal. Save event ID/sequence, session, SKU/quantity, direction, completion/commit time and evidence/model/calibration reference before forwarding; retries cannot apply an event twice.
- Electron `posSession.ts` binds the active shopping session and sends the **full desired ledger snapshot** to pushcart-web. Never send event deltas into its current snapshot-reconciliation route.
- pushcart-web remains sole owner of prices, stock, customer corrections and orders. Finish remains order-without-payment under [POS_INTEGRATION_SPEC.md](POS_INTEGRATION_SPEC.md).
- Hidden/reappearing items do not change the ledger. Two units require two transfers; removal of an absent SKU or ambiguous quantity needs review. Preserve inventory across camera loss/restart; do not guess actions during the blind interval.
- Start establishes the verified empty baseline/checkpoint. Restart resumes it but requires review before automatic operation continues. Do not replace current counter semantics implicitly; basket transfer is an explicit opt-in mode.
- Existing tablet edits override a SKU for the rest of the session. Mark that SKU manually managed: stop applying automatic transfers to its sensor ledger and let the server correction prevail. Observed later transfers prompt review. Resume automation only after explicit quantity reconciliation/checkpoint reset; do not silently replay deferred events or infer the customer's corrected baseline.
- Finish waits for pending/uncertain **observed** interactions to be resolved. Show confirmed updates separately from detector boxes. Manual review cannot certify completely unobserved actions automatically.

## 5. Conditional hardware fallback

Preference: existing camera/PC → fixed mount and observable path → optional useful second view → weight-assisted stationary platform **only if needed**. If the required workflow remains unobservable or hardware is impractical, narrow the demo or use honestly labelled counter occupancy with manual corrections. Weight is never compulsory merely because a vision trial fails.

Weight can corroborate stable pre/post mass increases/decreases after release and settling; impact peaks, hand pressure, vibration and drift are not transfers. It cannot identify a hidden SKU, resolve equal-mass swaps or guarantee bundle quantities. The measured load must pass through the sensing structure without bypass. A transfer tray measures tray interactions, not total basket inventory.

**Already documented:** [PRD.md](PRD.md) specifies StreamCam, local PC/YOLO, Python and Electron, and excludes scale/ESP32 integration from the current prototype. [DEPLOYMENT.md](DEPLOYMENT.md) lists mounts, power, optional display/weight sensor and possible edge nodes; it does not supply an approved scale BOM. ESP32-CAM there is a camera-node option, not a selected scale controller.

**New recommendations if weight is justified:**

| Component | What must be checked before selection |
|---|---|
| Load cell(s), rigid isolated basket/platform and mounting | Tare + maximum load, off-center loading, dimensions, impact margin and mechanical load path |
| Suitable load-cell ADC/amplifier; HX711 is one prototype candidate | Actual board datasheet: bridge/channel compatibility, supported sampling modes, noise, latency and supply/logic levels; no promised gram-level accuracy |
| ESP32 acquisition board and USB serial to existing PC | Timestamp/connection behavior, regulated supply, grounding and logic compatibility; no Wi-Fi requirement |
| Overload stops, enclosure, secured wiring/strain relief | Mechanical protection and safe operation |
| Known calibration masses | Repeatability, corner/load errors, tare and drift; compare measured noise with the lightest SKU |

Half-bridge corner sensors need correct bridge/combinator wiring; multiple full bridges may need separate channels or a suitable summing arrangement. Determine rating, ADC count, pins and costs from the actual design and verified datasheets—not this proposal. A battery/dock or IMU is deferred for a stationary demo.

If selected, record synchronized stable weight windows with the same video trials. Compare fusion against vision-only: agreement may support a visible SKU transfer; disagreement or mass change without visible identity requires review. Sensor disconnect/saturation is unknown, never zero mass. Do not assume statistically independent evidence or automatic confidence improvement.

## 6. Handoff and deferred work

**Next action:** choose the actual basket/mount and marked path, disclose the label-facing/one-item pace, then record Gate A with ordinary video. No instrumented recorder is needed for that human visibility decision.

Camera-only feasibility is **unproven**: no transfer footage, inference trials or scale measurements were performed for this plan. Fresh evidence capture, release/clearance recognition, persistent basket ledger and pending review remain future work. A successful Gate A justifies Gate B; only the frozen detector/integration rehearsal can support the guided camera-only demo claim.

Deferred: unrestricted mobile shopping, fleet/edge/cloud design, store-grade reliability targets, large action-model training and detailed hardware procurement. This planning document does not authorize implementation, training, purchases or production rollout.
