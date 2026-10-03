# Self-checkout integration (SCANnCART ↔ pushcart-web)

SCANnCART is the **sensor**; **pushcart-web** (a separate checkout, Next.js + Supabase) is the **sole
owner of money-relevant state**.
The camera infers what is on the counter, the desktop posts that as *desired state* to pushcart-web,
and pushcart-web reconciles its own `cart_items` rows to match. No cashier confirms anything, and
nothing here charges a card: **Finish** creates an order, stock deducts through pushcart-web's own
trigger, and payment stays out of scope.

Two machines, one hop:

| Where | What it does |
| --- | --- |
| Counter tablet (a browser) | Runs the customer-facing pages *inside pushcart-web*. Starts the session, shows the live cart and total, takes removals and quantity edits, and finishes. |
| SCANnCART desktop (Electron) | Keeps the camera running, binds to the tablet's session, derives the cart from what the tracker saw, and syncs it. Shows a read-only POS panel (Live view) and the integration's settings (Admin Panel). |
| pushcart-web host | Holds the cart, the order and the stock. Reachable from the desktop over the LAN or the internet. |

The spec this implements is [POS_INTEGRATION_SPEC.md](POS_INTEGRATION_SPEC.md) — §4 is the
pushcart-web half, §5 this repo's. To bring the two halves up together for the first time, or before
the shop goes live, the ordered run is [POS_SMOKE_TEST.md](./POS_SMOKE_TEST.md).

## 1. What runs where

- **Capture is always on while the feature is enabled.** The StreamCam takes ~37 s to open, so the
  desktop makes sure capture is `running` and leaves it running between customers. Binding is a
  *timestamp* instead: a track counts for this customer only if it entered after the session bound.
- **The tablet opens the session, the desktop binds to it.** Nobody types anything at the counter.
- **Corrections happen on the tablet.** The camera's guess is the starting point, not the last word:
  a removal or a quantity change there wins for the rest of the session.

## 2. Set up pushcart-web

1. Apply the migration `supabase/migrations/*_pos_self_checkout.sql`. It adds `stations`,
   `product_class_map`, `station_sessions`, `station_session_overrides` and `pos_sync_log`, the
   `cart_items.session_ref` column, and the five functions (`pos_reconcile` for the desktop;
   `pos_open_session`, `pos_my_session`, `pos_customer_edit`, `pos_finish` for the tablet).
2. Put two variables in the environment pushcart-web runs with (its README documents both):

   ```json
   { "POS_INGEST_SECRET": "<32+ hex>", "POS_IDLE_CANCEL_MINUTES": 5 }
   ```

   `POS_INGEST_SECRET` is what the desktop sends as the `x-pos-token` header; `POS_IDLE_CANCEL_MINUTES`
   is how long an abandoned session blocks the counter before the next customer's *Start* cancels it
   (default 5).
3. In the admin screen at `/admin/<userId>/pos-mapping`, **register a station per counter** and
   **map every class slug the model can predict to a product**. Both matter:

   - The station's `id` is what an operator pastes into SCANnCART (§3). An unregistered station
     answers `404` to everything, including *Test connection*.
   - A class with no mapping is reported as `unmapped` on every sync and is **never inserted**. The
     camera cannot add what nobody has mapped, and the tablet tells the customer to ask staff.

   The same screen lists the recent unmapped classes (with one-click mapping), the open sessions
   (with a *Cancel* for a stuck one) and the customer-edit log — the record of items a customer
   removed, which is the only shrinkage trail this design has.

## 3. Set up SCANnCART

Admin Panel → **Self-checkout (POS integration)**:

| Field | Meaning |
| --- | --- |
| pushcart-web base URL | Whichever host runs pushcart-web: `http://192.168.1.20:3000` for Docker on the shop LAN, `https://pushcart.example.com` for cloud hosting. No trailing path. |
| POS secret | The same value as pushcart-web's `POS_INGEST_SECRET`. |
| Station id | The `id` of the station registered for **this** counter (§2 step 3). |
| Commit dwell (s) | How long an item must be seen before it counts. Default 3. |
| Remove settle (s) | How long a lower count must hold before an item is removed. Default 10. **Suspended 2026-10-03:** until the removal-capable model ships, no automatic removal happens — a verified count is a floor for the rest of the session, and this window only carries the count through camera dropouts. |
| Min commit confidence | Confidence below which a detection never counts. Default 0.6, to be tuned at the counter. |
| Unbound poll (ms) | How often the desktop asks for a session while unbound. Default 1000. |
| Session poll (ms) | The same, while bound. Default 5000. |
| Logs poll (ms) | How often the sidecar's tracks are read while bound. Default 1000. |
| Cart mode | `Counter` (default): the cart is what the camera sees, with the posted count as a floor. `Basket`: the cart is the transfer ledger — a confirmed deposit adds, a confirmed removal subtracts, hiding changes nothing. See §4a. |
| Cart is at | Which edge of the camera frame the cart is at (the camera's own edge, not the mirrored preview's). A deposit moves toward it. Default bottom. |
| Cart band / Opening band | The share of the frame, from that edge, that is inside the cart (default 0.35) and the opening just past it (default 0.2). The rest is outside. Save refuses a pair that leaves no outside band. |

**An empty URL, secret or station id turns the feature off** — a state, not an error: the Live view
then shows no POS panel at all, and capture behaves exactly as it did before. The configuration is
stored in the app's user-data directory as `pos.json`.

**Save settings** writes it and starts the integration immediately (no restart — the menu that runs
the loop is started here if it was off). Two rules are enforced as it saves:

- **Commit dwell must exceed the sidecar's `track_expiry_s`** (1.5 s by default). The sidecar waits
  that long before declaring a track gone, so a shorter dwell would let a replacement track count
  while the old one is still open, and one item would read as two. Save refuses such a value and
  names the two numbers.
- **The secret travels in clear text over plain `http://`.** *Test connection* warns when the base
  URL is `http://` and not localhost. On the LAN that is a decision; over the internet it should be
  `https://`.

**Test connection** asks pushcart-web for this station's session once and reports what came back:
the HTTP status, whether a session is open right now, the cart code, the TLS warning, and the
sidecar's live `track_expiry_s` — so the dwell rule is checked against the value that is actually
running rather than the default.

## 4. The flow, with no operator

1. The customer taps **Start shopping** on the tablet. The tablet signs out anyone left over from the
   previous customer, signs in anonymously (which creates the cart), picks this counter's station and
   opens a session on it.
2. The desktop binds: within `unboundPollMs` it sees a new `session_ref` and marks the moment as its
   beginning. Items already on the counter were there before that moment and belong to nobody.
3. **Placing an item.** The tracker sees it; after `commitDwellS` of continuous presence it counts,
   and the desktop posts the new snapshot. Rows appear on the tablet with a running total.
4. **Two of the same product** placed apart count as two. **One item picked up and put back down**
   stays at one: the tracker ends the old track and starts a new one, and the gap between them is
   shorter than the settle window.
5. **Taking an item away.** The count drops, and it is removed from the cart after `removeSettleS`
   of the lower count holding — so a hand passing over an item does not remove it.
   **Suspended 2026-10-03:** the next model is not ready to tell a taken item from a lost track,
   so the desktop holds every verified count for the rest of the session (the camera can only add
   to it, and the count survives an app restart — the floor is kept beside `pos.json`, keyed to
   the session; the tablet's Remove button, an override, is still the way out). This step
   describes the behavior the removal model restores.
6. **The customer disagrees.** Removing or changing a quantity on the tablet is recorded as an
   override, and the camera stops managing that product for the rest of the session (it will not
   re-add it, however long it sits there). If a second unit is then placed, the customer increments
   the quantity on the tablet.
7. **Finish.** The tablet creates the order in one transaction: stock is checked first (an order for
   more than is in stock is refused, naming the items, and staff resolve it), then the order, the VAT
   and the stock deduction that pushcart-web's trigger performs. The session becomes `completed`, the
   cart `paid`.
8. The desktop's next sync gets a `409` and it unbinds: it stops syncing and **keeps capture
   running**, so the next customer's *Start* binds in about a second. The tablet signs out and
   returns to the start screen.
9. **The next customer's Start does not pick up the previous customer's leftovers**, because they
   entered before the new binding.

If SCANnCART is off, or the camera has died, the tablet keeps working: the customer still sees their
items, can still edit them, and can still finish. They just add the rest by hand (staff add those as
manual rows, which the camera never touches).

## 4a. Basket mode (camera-only add and remove)

Basket mode replaces "what is visible" with **confirmed transfers**, which is what makes automatic removal safe: an item covered by another one is still in the basket, because only a confirmed removal takes it out.

- **Deposit (+1):** the item is seen at least twice outside, crosses the opening band, and is seen at least twice in the cart band over a one-second hold. Every sighting on the path has to clear the scanner's own **confidence threshold** (`conf_threshold`, Camera tuning) — set it to 0.9 and nothing under 0.9 counts, here or in the empty-basket check. The basket has no threshold of its own; it re-reads the scanner's every few seconds.
- **Removal (−1):** the same path in reverse, ending with the item held clear of the cart band for a second.
- **No change:** showing an item outside, hovering at the opening and pulling back, rearranging inside the cart, an empty hand reaching in, or an item hidden after it was deposited.
- **Needs review (cart unchanged):** the class changes mid-path, two items cross the opening at once, an item stops mid-path, an item appears in the opening with no origin, a removal of something the ledger does not hold, a product already in the cart band at Start, an app restart mid-session, or the camera seeing nothing for more than a few seconds. Review is sent to pushcart-web as `pending_review`; staff clear it with **Basket checked** on the Live view.

The ledger runs in counter mode too, as a shadow, so it can be compared with the real basket before switching. Keep `Counter` until a rehearsal (`CART_TRANSFER_SPEC.md` Gate C) shows the shadow matching. The ledger is kept beside `pos.json` as `basket-ledger.json`, keyed to the session.

## 5. What the desktop shows

**Live view** — a read-only *Self-checkout* panel, present only while the feature is on:

| Row | Meaning |
| --- | --- |
| status | `no session` (waiting for the tablet), `warming up` (the camera restarted and counts are held), `syncing` (bound), or `error` |
| cart | The first 8 characters of the cart code — the same one the tablet displays |
| items synced | What pushcart-web currently totals for this cart |
| last sync | How long ago the last accepted snapshot was posted |
| basket / basket (shadow) | Units the transfer ledger holds — the posted cart in basket mode, a comparison in counter mode — with `camera blind` while no fresh frames arrive |
| needs review | The unresolved interactions, with a **Basket checked** button that clears them once staff have looked |

A sync failure appears under those rows as red text (the same condition the banner above reports,
kept here because this panel is where its detail still is). While the desktop is backing off from an
unreachable webapp, that error is followed by `retrying in Ns`, counting down to the next attempt;
the line goes away when the attempt runs and the state stops claiming a retry. The only control on
the panel is **Basket checked**: the tablet's *Start* is what binds the desktop.

**Admin Panel** — the settings in §3.

## 6. Troubleshooting

| Symptom | Cause | What to do |
| --- | --- | --- |
| *Test connection* → `401` | The desktop's POS secret is not pushcart-web's `POS_INGEST_SECRET`. | Copy the value again on both sides. |
| *Test connection* → `404` | The station id is not in pushcart-web's `stations` table. | Register it on the admin POS screen and paste its `id`. |
| *Test connection* → `409` / the desktop binds to nothing | No session is open on this station. The desktop is waiting for the table's *Start*. | Tap *Start shopping* on the tablet. |
| Save refuses the commit dwell | `commitDwellS <= track_expiry_s`, which would let one item count as two. | Raise the dwell or lower `track_expiry_s` in Camera tuning. |
| Live panel shows `error: … 409` | The session is over — the customer tapped **Finish**, or staff cancelled it. | Nothing. It unbinds on its own; the next *Start* binds again. |
| Live panel shows a sync error / *Test connection* says unreachable | pushcart-web is not reachable from this machine (down, wrong host, WAN). | Check the base URL; the desktop retries with backoff and keeps the last counts. |
| The tablet says *counter busy* | Another session is open on that station and was active less than `POS_IDLE_CANCEL_MINUTES` ago. | Wait for it to go idle, or cancel it on the admin screen's open-sessions list. |
| The tablet says *N items not recognized* | Those classes are not mapped to products. | Map them on the admin POS screen. Nothing reaches the cart until then. |
| The tablet says *camera offline* | The desktop has not synced recently. | The customer finishes manually and asks staff to add what the camera missed. |
| An item is in the cart that the customer never put there | A detection counted (`minCommitConf`, `commitDwellS`). | Remove it on the tablet — that is permanent for this session. |
| An item on the counter is missing from the cart | The camera missed it, or the item is stacked/touching another. | Staff add it as a manual row; the camera never touches manual rows. |
| *Test connection* warns about `http://` | The secret would travel in clear text. | Use `https://` unless this is localhost. |

Two things the counting cannot do, both by design: **stacked or overlapping items under-count** (one
box, not two), and **a product the model confuses with another** is counted under the wrong name.
Both are visible on the tablet while the item is still on the counter, and "ask staff" is the answer
to each.

## 7. Payloads

`GET /api/pos/session?station_id=<id>` (desktop → pushcart-web, `x-pos-token`), answering `404` for
an unregistered station and data `null` when none is open:

```json
{ "data": { "session_ref": "scanncart-<station>-<epochms>",
            "cart_id": "…", "cart_code": "…", "cart_status": "active" } }
```

`POST /api/pos/sync` (desktop → pushcart-web) is a full snapshot of what is on the counter now — the
same snapshot twice is a no-op, and duplicate rows are impossible because pushcart-web reconciles
inside one transaction:

```json
{ "session_ref": "scanncart-…", "station_id": "counter-1",
  "items": [ { "class_name": "safeguard_pure_white_60g", "quantity": 2, "max_confidence": 0.91 } ] }
```

Each item's `status` comes back as `added`, `updated`, `removed`, `unmapped`, `overridden` or
`warned` (`warned` = in the cart, but more than pushcart-web's stock says it has; Finish is where
that is enforced).

What the desktop reads from the sidecar is unchanged from a normal capture — `GET /api/logs`, with
the optional `?since=<seconds>` filter that drops tracks which had left before that time. It is what
keeps the per-sync payload the size of one customer's session however long capture has been up.
