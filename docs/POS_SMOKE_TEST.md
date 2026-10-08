# Go-live smoke test (SCANnCART ↔ pushcart-web)

> The one run that needs both halves and real objects on a real counter. The unit suites prove each
> side on its own; only this says the two agree — the tablet ends up showing exactly what the camera
> saw on the counter, with nobody typing anything.
>
> Related: [POS_INTEGRATION.md](./POS_INTEGRATION.md) (setup, the operator-less flow, troubleshooting)
> · [POS_INTEGRATION_SPEC.md](./POS_INTEGRATION_SPEC.md) §7 (the acceptance this list walks) and §6
> (the edge cases it exercises).

Run it before the shop goes live, and again after anything that moves a number the counting depends
on: a new weight, changed dwell/settle/confidence settings, or a change to pushcart-web's POS routes.
Two people — one at the tablet, one watching the desktop — is the easiest split; one runner can do it
with the tablet page on a phone and the desktop in front of them.

## 0. Before the first run

| Check | Where | What it has to say |
| --- | --- | --- |
| Migration applied; secret and idle window in the environment | pushcart-web | `POS_INGEST_SECRET` is the same value the desktop holds; `POS_IDLE_CANCEL_MINUTES` is set (15 by default) |
| Station registered per counter; every roster class mapped | pushcart-web admin → POS mapping | This counter's station id exists, and no class reads as unmapped |
| URL, secret, station id | SCANnCART Admin Panel → **Self-checkout** | *Test connection* reports **Connected** — not `401` (secret mismatch) or `404` (station not registered) |
| Weights and the class list | SCANnCART Admin Panel → **Detector backend** | *Test connection* reads `7 classes`, with no class warning |
| Dwell above the sidecar's expiry | Admin Panel → **Self-checkout** | **Save settings** is accepted; a refusal that names `track_expiry_s` is the rule working, not a bug |
| Aim | the camera | The frame is the marked counter area and nothing else — no shelves, no queue, no bagging area. The logs this integration reads carry no box positions, so the aim is the only zone filter there is. The Live view checks along with you: a capture that has run twenty seconds without producing a single box raises a banner asking for the hand-wave (wave where items go — a box should appear within a couple of seconds), and it stays up until a box is seen, so a camera left pointing at a wall cannot sit unnoticed between customers |
| Capture | Live view | Streaming, with the counter **empty** |
| Empty-counter quiet | Live view item log | Twenty seconds with nothing placed leaves the item log empty. That is the phantom suppressions holding; a row here is a false add before the first customer |
| Panel vocabulary | Live view | The *Self-checkout* panel appears only while the feature is on, and reads `no session` before the first *Start* |

Reset between runs: tap **Finish** (or cancel the session from the admin screen's open-sessions list),
sign the tablet out, empty the counter, and wait for the Live panel to read `no session` again.

## 1. The happy path

One customer, in this order. The "expect" column is what makes the run a pass.

| # | Do | Expect |
| --- | --- | --- |
| 1 | Tablet: **Start shopping** | Lands on the live cart, empty. **Counter busy** here means a session is already open on this station — cancel it, or wait out `POS_IDLE_CANCEL_MINUTES` |
| 2 | Watch the desktop | The panel flips `no session` → `syncing` within about a second, showing the cart code the tablet shows, `0` items |
| 3 | Place one product and leave it | After ~3 s it is on the tablet, quantity 1, total = its price. **Record the add latency** (placed → in the cart); it should land near the dwell |
| 4 | Place three different products, one at a time | Each appears with the right quantity, and the total is their sum |
| 5 | Take one away | It leaves the tablet after ~10 s and the total drops. **Record the remove latency** (taken away → gone); it should land near the settle window. In counter mode the count **holds** here instead (the floor, see POS_INTEGRATION.md): the row stays, and about one dwell (~3 s) after the item leaves it shows *Camera lost this item — still in your cart*; staff's PIN removal is the correction. Put the item back and the badge clears with the quantity unchanged; in basket mode lift it out through the opening and it leaves after the 1 s clearance — record what you saw either way |
| 6 | Lift an item and set it back down (or pass a hand over it for 1–3 s) | Quantity stays 1 throughout — never 2, never 0. This is the track swap the settle window exists for |
| 7 | Two identical products, apart; then touching | Apart: quantity 2 after the dwell. Touching or overlapping **may read as 1** — that under-count is known and parked (spec §7.1). Correct it on the tablet, and write down what you saw |
| 8 | Tap **Finish** with items still on the counter | The order is created, stock is decremented, the cart is `paid` and the session `completed`; the tablet signs out and returns to the start screen; the desktop **unbinds on its own** (the panel reads `no session`) with capture **still running** — the preview keeps streaming |
| 9 | Leave one of those items where it lies, then **Start** as the next customer | The new cart is **empty**, even though the camera can still see the item: it entered before the new binding and belongs to nobody |

Step 8 continues in the database, which is where the four facts the desktop cannot see live:

| Row | What to find |
| --- | --- |
| `orders` | One row for the anonymous customer, with the cart's lines and VAT computed from the active rate |
| `products.stock_quantity` | Reduced by each line's quantity |
| `carts.status` | `paid` |
| `station_sessions` | `completed` for that `session_ref` — and a sync naming that ref now answers `409` |

Then run one more customer end to end with a different mix of products. A single clean run is a
coincidence; the second one is the one that says the first was not.

## 2. The correction surface

These are the paths that decide whether a wrong count is *fixable* at the counter.

| Do | Expect |
| --- | --- |
| Look for a way to edit the cart as the customer | There is none: the tablet shows quantities only, and no *−*, *+* or *×* |
| *Staff* → wrong PIN → *Remove 1* | *Wrong staff PIN*; nothing changes |
| *Staff* → the staff code → *Remove 1* on a camera-detected item, then leave the item on the counter | The quantity drops by one and stays there. Later snapshots report it `overridden` and never re-add it |
| Admin → POS Mapping → *Staff code* → set a new code; on the tablet try the old code, then the new one | The old code is refused as *Wrong staff PIN*; the new one works at once, with no restart. The card reads *Set here* and never shows the code |
| (basket mode) Make two items cross the opening together | The tablet shows *Staff check needed* and Finish is disabled; **Basket checked** on the desktop clears it within a sync |
| Place an item whose class is not mapped to a product | The tablet shows *N items not recognized — please ask staff*; nothing enters the cart. Staff add it by hand as a manual row, and later snapshots leave that row alone |
| Check the shrinkage trail | `pos_sync_log` rows with `kind = staff_edit` are the staff removals from this run — the only record that a camera-detected item was taken off by hand |

## 3. The failure paths worth running at go-live

Each of these is one run's worth of setup and a great deal of confidence.

The rows that are only about the desktop's two POS routes — a dead host and its retry countdown, the
host coming back, a session that ends under the desktop, and a rotated secret or unregistered station
— are also an automated integration test. `desktop/src/main/posRoutes.integration.test.ts` drives the
real transport and orchestrator over a loopback stand-in for pushcart-web's `/api/pos/session` and
`/api/pos/sync`, so a route, a header or a status code that moves fails `npm test` rather than
surprising someone at the counter. It is a stand-in for pushcart-web's *routes*, not its database: no
Supabase, no mapping table, no stock. Because a stand-in is only as faithful as the pushcart-web files
it was written from, the same protocol checks — auth, an unregistered station, snapshot validation, a
session that is not open — can be pointed at a real server instead: `make verify-pos-routes` wants
`POS_E2E_BASE_URL`, `POS_E2E_SECRET` and `POS_E2E_STATION_ID`, and says so loudly rather than
reporting a green skip when they are missing. And because both sides of those tests were written from
one reading of pushcart-web, a header or a path that moved in *both* would pass them green:
`make verify-pos-contract` re-reads pushcart-web's route source and fails if the contract this repo
records has gone stale, with `posContract.test.ts` checking the modules against that record in
`npm test`. The rest of this table needs the shop — a camera, a
tablet, the database and a person — and this list does not replace it.

| Break it like this | Expect |
| --- | --- |
| Stop pushcart-web (or point the desktop at a dead port) with items in the cart | The panel shows the sync error **and `retrying in Ns`**, counting down; the retry interval backs off instead of flooding. The tablet keeps showing the cart and **Finish still works** — it reads the database, not the desktop |
| Start pushcart-web again | The next snapshot converges the cart, no duplicate rows appear, and the countdown disappears |
| Stop the camera mid-session (Admin Panel, or unplug it) | The desktop restarts capture once — an operator's *Stop capture* looks exactly like a crash from the desktop's side, which is why it comes back; that is expected here. While it is blind the panel reads `warming up` and the counts are **held** — nothing leaves the cart. The tablet can still finish. If the restart fails, the panel says capture is down and the cart stays at its last counts |
| Set **Commit dwell** to 1 and save | Save is refused, naming `track_expiry_s` — one item can never read as two |
| Start a second session while one is open (reload the tablet and Start again) | **Counter busy** inside the idle window — the protection for a customer who only stepped away |
| Give a cart more of a product than its remaining stock, then Finish | Finish is refused with the items named, and **nothing is written**: the cart stays active and stock is unchanged. Fix it, Finish again, and it succeeds |
| Point the station id at something unregistered, or paste the wrong secret | *Test connection* reports `404` or `401`, each naming what to fix |

## 4. Pass or fail

A run **passes** when:

- every Finish's cart holds exactly the products and quantities that were on the counter;
- nothing entered the cart with an empty counter;
- every miss has an explanation — a known limitation (items touching or stacked, one product confused
  with another) or a correction made on the tablet;
- the desktop unbinds on Finish, and capture is still running afterwards;
- the next customer's *Start* carries nothing over from the previous one.

A run **fails** on any of: a duplicate camera row for one product, an order whose lines or stock do
not match the cart, a desktop that stays bound after Finish, capture stopped after Finish, or an
unexplained add with nothing on the counter. A failed run goes back to tuning (`commitDwellS`,
`removeSettleS`, `minCommitConf`), to the camera's aim, or to retraining — not to the shop.

The numeric go-live gate — final-cart exact match on a recorded scenario corpus — is spec §7.1's
counting-accuracy harness. That work is recommended and parked; it is not what this checklist does,
and this checklist does not replace it.

## 5. Record the run

Keep the sheet with the release notes so the next run has something to compare against.

| Field | Value |
| --- | --- |
| Date, shop, counter / station id | 2026-10-07, home bench, station `cart-1` (basket mode) |
| Models in use (desktop weight, pushcart-web revision) | `scanncart-grocery-v2-letterbox-synth.pt` (installed 2026-10-07); pushcart-web with the `pos_camera_lost` migration |
| Dwell / settle / min confidence | 3 s / 10 s / 0.6 (removal still suspended — no automatic removal) |
| Add latency, remove latency (observed) | not measured this run — no item ever reached the cart (step 3 failed) |
| Steps 1–9, per step | **1–2 pass**: tablet *Start shopping* → fresh empty cart `639b644b`, desktop bound within a poll, empty heartbeats flowing. **3 fail — camera aim**: the live frame watches a curtain and a clothes rack, not the counter; a 10 s WS sample carried 0 detections in ~350 frames, so step 3 could not be attempted. Steps 4–9 not run live — gated on 3. **Automated equivalent of the webapp half passed the same day**: `pushcart-web scripts/pos-e2e.sh` against the live local stack — **116/116** — walking sync add, the camera-lost stamp/repeat/clear cycle, summed classes, unmapped class, staff removal (`overridden`, `staff_edit` logged), review blocking Finish, short stock, the full Finish chain (order row, cart `paid`, session `completed`, stock deducted) and a sync on the closed session → 409, plus both idle-window mechanisms: the poll sweep (a session backdated past the window is cancelled by the desktop's own `GET /api/pos/session` poll — no Start needed — which answers `unbound`; a fresh session survives the same poll; a session just inside the window survives too) and the on-Start cancel the spec always had (backdated `last_activity_at` → next Start cancels the abandoned session and opens a fresh one; sync on the cancelled ref → 409), plus the daily shopper label (`Shopper_NNNN` carried by the open response and the desktop poll, drawn previous+1 whatever ran before, reset to `0001` on a new Manila day, and distinct under two parallel Starts), plus the admin screen's Basket checked (customer clear → 403; the desktop poll carries `review_cleared_at`; a review re-post whose newest interaction predates the clear loses to it; a newer interaction raises the review again). The desktop's half over the real transport: `make verify-pos-routes` against the live server — **28/28** |
| §2 correction surface | **pass (automated)**: staff remove right/wrong PIN, over-remove refused, `overridden` afterwards, customer edits refused 403 with the camera count untouched, direct customer writes refused by RLS — all in the 92 above; the tablet's own screens were not driven |
| §3 failure paths | **pass (automated)**: dead host + backoff, host back, session ended under the desktop (`posRoutes.integration.test.ts` 28/28, real transport); review-blocked finish, short-stock finish, busy counter, idle-window cancel, post-finish 409 (the 92 above). **Camera unplugged mid-session — driven on the rig 2026-10-07** (below) |
| Verdict, and who ran it | **Webapp + desktop halves verified end to end by automation; the camera half is the only thing still unproven — blocked at step 3 because the camera is not looking at the work surface** (plain aim problem, not the focus/scale one from 2026-10-04: the frame is sharp, it just frames the wrong scene). Physical ask before the next live run: aim the camera down at the basket zone so a can placed there fills roughly half the frame height with clear space above and below it, camera ~0.5–1 m away at eye level with the basket opening. Also fixed this run: a server bug that looked like a broken rig (see DEFENSE_READINESS risk table — websocket message compression starved the sidecar loop). Runner: Codebuff agent + shop operator for the physical aim |
| §3 camera-unplug path (2026-10-07, driven on the rig) | StreamCam's USB video interface disabled mid-session with the desktop bound. **Found and fixed a gap**: on this MSMF/Windows stack the vanished device's `read()` keeps *succeeding* with one stale black frame at ~1 fps, so the 3-second failed-read deadline in `camera.py` never fired — `/api/health` said `running` for 5+ minutes, no WS error, the preview re-encoded a frozen black frame ~57 msg/s, and the tablet showed a normal live cart (its liveness signal is the sync heartbeat, which kept flowing; no "Camera offline"). **Fix**: a delivered-rate stall guard in `CameraCapture._loop` — below 5 fps over a 5 s window, sustained 5 s, is the same "camera went away" failure (ordinary hitches and the 60→30 fps latch sit far above it); regression-tested against a faked 1 fps black stream and a 30 fps healthy one (`test_camera_capture.py`), then **replayed against the real device**: the guard fired, capture tore down with the clear "frozen … restart capture (or the app)" message, and the desktop's recovery began re-opening capture itself (19 auto `capture/start` attempts while the device stayed disabled — a restart loop while the camera is gone, converging the moment it returns; each attempt ≈10 s so the load is negligible). The failure path a customer or operator sees is now: frozen preview ≤ ~10 s → capture stops with a named error → desktop retries → self-heals on replug. |
