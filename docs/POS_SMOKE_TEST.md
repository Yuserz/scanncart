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
| Migration applied; secret and idle window in the environment | pushcart-web | `POS_INGEST_SECRET` is the same value the desktop holds; `POS_IDLE_CANCEL_MINUTES` is set (5 by default) |
| Station registered per counter; every roster class mapped | pushcart-web admin → POS mapping | This counter's station id exists, and no class reads as unmapped |
| URL, secret, station id | SCANnCART Admin Panel → **Self-checkout** | *Test connection* reports **Connected** — not `401` (secret mismatch) or `404` (station not registered) |
| Weights and the class list | SCANnCART Admin Panel → **Detector backend** | *Test connection* reads `7 classes`, with no class warning |
| Dwell above the sidecar's expiry | Admin Panel → **Self-checkout** | **Save settings** is accepted; a refusal that names `track_expiry_s` is the rule working, not a bug |
| Aim | the camera | The frame is the marked counter area and nothing else — no shelves, no queue, no bagging area. The logs this integration reads carry no box positions, so the aim is the only zone filter there is |
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
| Date, shop, counter / station id | |
| Models in use (desktop weight, pushcart-web revision) | |
| Dwell / settle / min confidence | |
| Add latency, remove latency (observed) | |
| Steps 1–9, per step | pass / fail, with what was seen |
| §2 correction surface | pass / fail |
| §3 failure paths | pass / fail |
| Verdict, and who ran it | |
