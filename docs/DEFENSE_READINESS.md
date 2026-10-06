# Defense readiness audit

Audited 2026-10-05 against [PRD.md](./PRD.md), [POS_INTEGRATION_SPEC.md](./POS_INTEGRATION_SPEC.md)
and [CART_TRANSFER_SPEC.md](./CART_TRANSFER_SPEC.md), on the defense machine (RTX 4060, Logitech
StreamCam, local pushcart-web on Supabase).

## Verdict

**Ready to defend as a working, integrated prototype; not ready to claim measured deposit/removal
accuracy.** Every component is built, wired end to end and covered by automated tests, and the
live system meets the speed and latency targets. What is missing is evidence on a real cart: the
cart mount does not exist yet, so the transfer acceptance gates (A–C) have not been run. The app
runs the retrained **v2** model (retrained 2026-10-06 on cleaned labels), which beats v1 on the
same test images overall, per product and at every distance.

Say it that way at the defense: the claims below marked **Met** are measured; the ones marked
**Implemented, unmeasured** are designed, tested on synthetic data and demonstrable at a desk, but
have no field accuracy number.

## Requirement trace

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| F1 | ≥ 30 analysed frames/s | **Met** | Live: capture 60–61 fps, analysis 46–59 fps, 14–19 ms per frame (2026-10-05 soak samples, v1); v2: capture 62 fps, analysis ~45 fps (2026-10-06) |
| F2 | Boxes, names, confidence on screen | **Met** | Live view overlay; Basket test labels each box with product, confidence and track id |
| F3 | +1 on deposit, −1 on removal, hidden items stay billed | **Implemented, unmeasured** | State machine + ledger; 557+ desktop tests incl. 60 fps fast-hand and id-switch cases; desk practice mode. No cart footage (Gates A–C not run) |
| F4 | Ambiguity → review, Finish blocked | **Met** (integration) | Desktop review rules; pushcart-web `pos_finish` refuses while `pending_review > 0`; `scripts/pos-e2e.sh` against the local stack; observed live on 2026-10-05 (a desk item produced "removal of … not in the basket", Finish disabled) |
| F5 | Auto-bind on Start, unbind on Finish | **Met** | Live run 2026-10-05: tablet *Start shopping* → desktop bound to `cart-1` within the 1 s poll |
| F6 | Sync on change + heartbeat, no duplicates on retry | **Met** | Immediate post after a confirmed change (orchestrator test); `pos_reconcile` idempotency (stand-in + e2e); 409/backoff tests |
| F7 | SQLite track log | **Met** | `logging_store.py` + tests; `scanncart.db` |
| F8 | Camera disconnect / sidecar stop explained | **Met** | 3 s camera-failure deadline → error banner; main-process health monitor → *sidecar unresponsive* notice |
| F9 | Start/stop, live tuning | **Met** | Live view; hot-reloadable settings incl. confidence, filters, camera controls, auto exposure |
| — | Staff correction (decrease-only) with a changeable staff code | **Met** | pushcart-web *Staff* → code → *Remove 1*; an admin sets/changes the code on POS Mapping (bcrypt hash in `pos_settings`, admin-only database functions; `POS_STAFF_PIN` is the fallback). Database checks 2026-10-05: non-admin refused, bad format refused, only the right code matches. The admin card's click-through is still to be done by hand |
| — | Admin screens closed to customers | **Met** | `scripts/pos-admin-routes-e2e.mjs` (pushcart-web): an anonymous tablet session and a non-admin user are refused (403) on all 11 POS admin calls, an admin is allowed — 29/29 on 2026-10-05 |
| N1 | Camera-to-screen < 150 ms | **Met (estimate)** | ≤ 17 ms frame wait + 19 ms analysis + 17 ms delivery to a WebSocket client + ~17 ms paint ≈ 70 ms. Component sum, not a photon-to-photon measurement |
| N2 | Transfer → tablet ≤ 5 s (p95) | **Implemented, unmeasured** | Design: confirmation ~0.5 s after landing, posted immediately (~0.25 s round trip locally). Gate C would measure it |
| N3 | ≥ 2 h continuous | **Partly measured** | Two runs (34 and 80+ min): scanner up throughout, never restarted, memory flat, connections 4–15. In two runs capture latched from 60 to ~30 fps at minute 18–23 (once with NVIDIA Broadcast closed); the scanner kept running but the camera's cause is unconfirmed (see Soak result) |
| N4 | Detection with no internet | **Met** | Native backend, local weights; only the pushcart-web hop uses the LAN |
| N5 | Modular, tested without hardware | **Met** | 560 desktop (Vitest) + 1,717 sidecar (pytest) tests on fakes; CI; POS contract check against pushcart-web's source |

## Model metrics

The app runs **`scanncart-grocery-v2-stretch`** (YOLO11s, 100 epochs, 640×640 stretched), retrained
2026-10-06 on a rebuilt merged set (2,134 train / 494 valid / 277 test images). The rebuild cleaned
the labels: the old v1-era photos carried a second, partial box inside the box around an item (the
"555" logo on a sardines can, the noodle picture on a Lucky Me pouch), which taught the model to
draw both and scored a correct one-box detection as a miss. The merge now keeps one box per item
(`build_dataset.drop_nested_rows`, v1 photos only — 292 boxes dropped, every one checked against its
frame); the labels re-annotated in Roboflow were pulled in and the photos deleted there left out.

All three weights measured by `make accept-v2` on the same 277 held-out test images (279 boxes),
each in its own class order, at conf 0.5:

| Metric | Target | v1 | v2 (first, 10-06) | **v2 (running)** | Status |
| --- | --- | --- | --- | --- | --- |
| mAP50 | ≥ 90% | 89.8% | 98.1% | **99.4%** | **Met** |
| Precision | — | 89.9% | 95.3% | **99.7%** | — |
| Recall | — | 91.3% | 99.2% | **99.6%** | — |
| mAP50–95 | — | 85.2% | 91.1% | **93.2%** | — |
| Per-class recall ≥ 0.85 | every SKU | 6 of 7 (Milo 0.63) | 7 of 7 | **7 of 7** (lowest Safeguard 0.975) | **Met** |
| Recall at close / mid / far (`--val`) | ≥ 0.85 | — | — | **every class ≥ 0.91 at every distance** (small cells: 46 / 14 / 17 frames) | **Met** (indicative) |
| Frames with a doubled detection | — | 37 | 33 | **0** | — |

These numbers are on the cleaned labels, so they are not comparable with the earlier table that
had v2 at 96.7% and Lucky Me at 0.794 — that Lucky Me figure was the partial boxes, not the model.

**The acceptance gate still reports "not accepted"**, for a reason that is about the test images,
not the model: its crowding claim needs test frames holding two or more items at mid and far, and
there are none (the only two crowded test frames are stacked Safeguard bars at close). With no crowd
to find it reports "not measured", which the gate treats as a failure. The check itself now scores
against the labels, so a model is no longer rewarded for doubled detections.

**Known weakness — phantoms on a blank or empty view.** With no product in frame, the new weight
often reports one whole-frame box (mostly 555 Sardines): 44 of 50 stored empty-counter photos at
conf 0.8 (the first v2: 12), and 217 of 227 frames of a live 30 s window with the camera facing a
blank grey surface. The app's frame-edge filter (on by default) removed every one of them — the item
log stayed empty — so the app behaves correctly, but it depends on that filter. Training with
empty-basket photos from the cart's view is the fix.

v1 stays installed, one click away in Admin → Model. The first v2 is kept out of the picker, with its record, in `sidecar/data/datasets/runs/scanncart-grocery-v2/installed-record/` — copy both files back into `sidecar/models/` to offer it again.

### Stretch or letterbox: an experiment in progress (2026-10-06)

The stretched v2 squashes each frame to a square; the phone photos it learned from are 4:3 and the
camera is 16:9, so a product is squashed by a different amount live than in training. A second v2,
**`scanncart-grocery-v2-letterbox`**, is trained on the same frames kept at their own shape (v1's
originals re-fetched from Roboflow, plus 261 Safeguard originals matched from the raw capture zips),
at `imgsz` 960 and run letterboxed. The two are compared on the same test frames and, because no
test frame comes from the StreamCam, on the same recorded camera clips (`tools/ab_live.py`); the
winner becomes the shipped model. On held-out photos cropped to 16:9 the stretch weights still
scored above 0.85 on every frame, so the geometry did not explain the live tracking drops by itself.

### Fixes from the same day

- **Tracking dropped on weak frames.** The operator's threshold was passed to the tracker, which
  then never saw the weak boxes it uses to carry a track through a dip, so one frame under the
  threshold ended a track and the item returned with a new id. The tracker now sees boxes down to
  0.1 and the display holds a shown item down to 0.25 (it needs the threshold to appear).
- **Grey picture and low fps in a dark room.** Auto exposure now reports "too dark for the camera
  at this frame rate" instead of painting the picture grey; an opt-in *Allow 30 fps when dark*
  trades frame rate for light; the Exposure slider warns when a manual value costs frame rate.
- **`imgsz` follows the model**: switching weights in Admin sets the size they were trained at.
- **v1 frames with an empty label file are left out of the build**: all 17 were products nobody
  boxed, which trained the model that the back of a pack is nothing.

## Open risks for the defense day

| Risk | Seen | Mitigation |
| --- | --- | --- |
| Capture latches from 60 to ~30 fps after ~18–23 min of running; **cause not yet confirmed** | 2026-10-05, twice: once with NVIDIA Broadcast running, once with it closed. In that state a −9 shutter still gives ~30 fps (so it is not exposure), the camera ignores some exposure writes, and a stream reopen — even by a separate process — inherits it | Test the suspects: Logitech Options+ (its agent runs and can set the camera's low-light compensation), the camera's firmware low-light mode, USB power/bandwidth (another port; unplug/replug). Keep Broadcast's autostart off. Before the demo, check the Capture fps tile reads ≈ 60 |
| The scanner refused requests (“Exceeded concurrency limit”, > 64 open connections) once, then the app closed | 2026-10-05, during repeated dev-mode reloads of the Basket test screen; idle connection count is 4–8 | The soak watches the connection count; avoid hot-reloading during the demo (run a normal build) |
| Products on the desk inside the *inside* zone trigger review | 2026-10-05 | Clear the inside zone before Start; press **Basket checked** if it fires |
| The self-checkout config defaults to **Counter** mode | by design | Set **Basket** in Admin → Self-checkout before the demo (done on this machine) |
| Four leftover test stations in the local database | 2026-10-05 | A `cart-1` station sorts first and is what the desktop is set to |
| Low light lowers detection confidence | — | Auto exposure holds picture brightness at 60 fps; keep the demo area lit |
| **Restore Defaults** in Admin puts the model back to v1 (the code default) | by design until the default is changed | Do not press it on the demo machine, or reselect `scanncart-grocery-v2` after |
| v2 reports a whole-frame phantom (mostly 555 Sardines) on a blank or empty view; only the frame-edge filter keeps it out of the log | 2026-10-06 | Keep **Drop frame-edge phantoms** on; empty-basket photos from the cart + retrain fixes it |
| Two items stacked on top of each other (seen with Safeguard bars) are counted as one | 2026-10-06 | Demo with items side by side |

## Not done (state plainly if asked)

1. **Gates A–C** of the transfer spec: no recorded cart footage, so no deposit/removal accuracy number.
2. **v2 acceptance**: every per-class and distance number passes, but `make accept-v2` cannot
   measure crowding at mid/far — the test set has no multi-item frame there. Needs a few test
   captures with 2+ items at mid and far.
3. **Physical cart mount**: all transfer testing is at a desk with practice mode.
4. **Empty-basket negatives** from the cart's view are not in training (the cause of the phantom
   weakness above).
6. **Dataset clean-up in Roboflow**, in progress: low-quality photos are being removed from
   `snc-grocery` (Lucky Me 0144 done 2026-10-06; the rest next session). Each removal needs a
   re-pull (`import_labels.py --force`), a rebuild and a retrain to reach the model.
5. **Payment**: Finish creates the order and deducts stock; no payment is taken.

## Demo-day checklist

1. Close NVIDIA Broadcast (and turn off its autostart), Discord and anything else that can hold the camera.
2. Start Docker Desktop → `npx supabase start` and `npm run dev` in pushcart-web.
3. Start SCANnCART; on **Live**, confirm Capture fps ≈ 60 and the model is `scanncart-grocery-v2-stretch`
   with geometry `stretch` (Admin → `resize_mode` on `auto`).
4. Admin → Self-checkout: station `cart-1`, mode **Basket**, *Test connection* answers 200.
5. Basket test: zones match the table/basket in the picture; clear products out of the inside zone.
6. Tablet: open `/customer/guest/scan-start`, tap **Start shopping**; the desktop shows *Customer session open*.
7. Deposit one product slowly through the opening and leave it; it appears in about half a second.
   Remove it the same way. Show a two-at-once move going to review and **Basket checked** clearing it.
8. Finish: the order is created and the tablet resets.

## Soak result (N3)

Run 2026-10-05, one sample every 30 s from a separate client (health, a 30-frame WebSocket
sample, the scanner's memory and its open connections); stopped at 34 min when the session ended.

| Window | Capture fps | Analysed fps | Analysis ms | Delivery ms | Scanner memory | Connections |
| --- | --- | --- | --- | --- | --- | --- |
| 0–18 min | 60–62 | 44–52 | 16.6–20.0 | 15.6–19.7 | 1,488–1,499 MB | 4–11 |
| 18–34 min | 27–30 | 29–32 | 16.3–20.0 | 8.2–18.2 | 1,450–1,497 MB | 6–11 |

The scanner never stopped answering, memory did not grow, and connections did not pile up. The
drop at minute 18 was not the scanner process: with the app closed, a bare OpenCV capture of the
same camera also got 27–30 fps, and a capture-only test writing brightness five times a second
ruled out the app's auto exposure (30 fps with and without the writes). NVIDIA Broadcast was
running and was first suspected, but a **second run with Broadcast closed** (started 2026-10-05
20:55) dropped the same way at minute 23, and a −9 shutter still gave ~30 fps — so the camera
latches into a ~30 fps mode for a reason not yet found (see the risk table). In both runs the
scanner's processes never restarted and connections stayed at 4–15. **Owed:** the cause, then a
clean 2-hour run.
