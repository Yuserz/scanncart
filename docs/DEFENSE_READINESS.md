# Defense readiness audit

Audited 2026-10-05 against [PRD.md](./PRD.md), [POS_INTEGRATION_SPEC.md](./POS_INTEGRATION_SPEC.md)
and [CART_TRANSFER_SPEC.md](./CART_TRANSFER_SPEC.md), on the defense machine (RTX 4060, Logitech
StreamCam, local pushcart-web on Supabase).

## Verdict

**Ready to defend as a working, integrated prototype; not ready to claim measured deposit/removal
accuracy.** Every component is built, wired end to end and covered by automated tests, and the
live system meets the speed and latency targets. What is missing is evidence on a real cart: the
cart mount does not exist yet, so the transfer acceptance gates (A–C) have not been run. The app
runs the retrained **v2** model (trained 2026-10-05/06), which is better than v1 on the same test
images overall and at distance, with one product (Lucky Me) that regressed.

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

The app runs **`scanncart-grocery-v2`** (YOLO11s, 100 epochs, 640×640 stretched, trained on the
merged set's 2,210 images). Both models measured by `make accept-v2` on the same 272 held-out test
images (325 boxes), each in its own class order, at conf 0.5:

| Metric | Target | v1 | **v2 (running)** | Status |
| --- | --- | --- | --- | --- |
| mAP50 | ≥ 90% | 90.7% | **96.7%** | **Met** |
| Precision | — | 96.1% | **99.1%** | — |
| Recall | — | 88.0% | **94.6%** | — |
| mAP50–95 | — | 86.5% | **89.7%** | — |
| Per-class recall ≥ 0.85 | every SKU | 4 of 7 | **6 of 7** — Lucky Me 0.794 | **Not met** (one SKU) |
| Recall at mid / far distance (`--val`) | ≥ 0.85 | far 17.6% (conf 0.5, earlier probe) | **100% at every distance, every class** (small cells: 14 mid, 17 far) | **Met** (indicative) |

Per class, v1 → v2: Bear Brand 0.92 → 1.00, Milo 0.63 → 1.00, 555 Sardines 0.75 → 0.86, Century
Tuna 1.00 → 1.00, Safeguard 0.93 → 0.97, Silver Swan 0.99 → 1.00, **Lucky Me 0.95 → 0.79**. v2 gets
every distance-tagged Lucky Me frame right; its misses are among the older v1-era photos.

**The acceptance gate (`make accept-v2`) did not pass**, for two reasons, and v2 was switched in by
decision anyway (2026-10-06):

1. **Lucky Me regressed** (above) — real; the fix is more Lucky Me photos and a retrain.
2. **"Fewer crowded frames found than v1"** — a flaw in the check, not in v2. It counts frames where
   a model *reports* two or more items without comparing with the labels. The labels hold **0**
   truly crowded frames at mid (v1 reported 3, v2 0) and **2** at close (v1 6, v2 1), so v1's lead
   is its own duplicate detections. The check should count only frames that truly hold two items.

v1 stays installed and is one click away in Admin → Model.

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
| Lucky Me is v2's weakest product (0.79) | 2026-10-06 | Demo with the other six first; more Lucky Me photos + retrain fixes it |

## Not done (state plainly if asked)

1. **Gates A–C** of the transfer spec: no recorded cart footage, so no deposit/removal accuracy number.
2. **v2 acceptance**: v2 is trained and running, but it has not passed `make accept-v2` (Lucky Me
   regressed; the crowding check needs fixing to compare against the labels).
3. **Physical cart mount**: all transfer testing is at a desk with practice mode.
4. **Empty-basket negatives** from the cart's view are not in training.
5. **Payment**: Finish creates the order and deducts stock; no payment is taken.

## Demo-day checklist

1. Close NVIDIA Broadcast (and turn off its autostart), Discord and anything else that can hold the camera.
2. Start Docker Desktop → `npx supabase start` and `npm run dev` in pushcart-web.
3. Start SCANnCART; on **Live**, confirm Capture fps ≈ 60 and the model is `scanncart-grocery-v1`.
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
