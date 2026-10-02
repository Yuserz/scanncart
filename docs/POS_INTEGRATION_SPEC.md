# Spec: SCANnCART → pushcart-web Automatic Self-Checkout Integration

Status: **Ready for implementation** · Date: 2026-10-01
Repos: **SCANnCART** (this checkout, Electron desktop) + **pushcart-web** (`C:\codes\pushcart-web`, Next.js + Supabase POS/inventory).
Reviewer note: file paths, routes, table/enum names were verified against both checkouts on 2026-10-01. pushcart-web uses `yarn` (its `yarn.lock` is authoritative; there is no `package-lock.json`).

---

## 1. Product intent

No cashier. A customer places items on the counter; SCANnCART's camera infers them and the items **appear automatically in the customer's cart** with a running total on a customer-facing screen (tablet at the counter, mobile-friendly). The customer reviews, removes anything wrong, taps **Finish** — the existing order flow creates the order and the existing `on_order_created_deduct_stock` trigger deducts inventory. SCANnCART is the sensor; pushcart-web remains the sole owner of money-relevant state.

**Product decisions — LOCKED by the client:**
1. Commit is **automatic** — no cashier confirms items before they enter the cart.
2. A new customer-side UI (tablet/mobile) shows the live cart + total; checkout happens there.
3. **Deployment model: the tablet UI is part of pushcart-web** — same Next.js app, same Supabase DB, one hosting (Docker on the shop LAN or cloud). No separate tablet backend, no tablet↔POS sync problem: the tablet *is* a pushcart-web client. Only the SCANnCART desktop is a separate machine (see §2.1).

**Decisions taken by default (OPEN FOR REVIEW, flipped by editing this line):**
- D1 **Cart binding: station-paired tablet.** Customer taps *Start shopping* on the counter tablet → webapp creates the cart (existing anonymous sign-in) and opens a session tagged to that counter's `station_id`; the SCANnCART PC auto-binds by polling. No typing, no QR decoding.
- D2 **Item removal: automatic on track-leave** (with a settle delay so a detector dropout doesn't yank a real item), plus manual per-item remove buttons on the tablet as the correction surface.
- D3 **Checkout scope: order without payment.** Finish creates the order (stock deducts via trigger); actual payment integration is parked.
- D4 **Session trigger: customer taps Start/Finish.** Start binds the desktop to the session (from that moment its tracks count) and Finish unbinds it. Capture itself runs all the time while the POS feature is enabled (§5.3), because opening the camera takes ~37 s. A session abandoned mid-way is cancelled by the **next** customer's Start once it has been idle for `POS_IDLE_CANCEL_MINUTES` (§3.3). There is no background timeout.
- D5 **Stock policy: warn while scanning, block at Finish.** A mapped product whose `stock_quantity` is below the cart quantity still enters the cart, with a `low_stock_warning` flag the tablet displays. Finish refuses such a cart and names the items (§3.3), and staff resolve it. Allowing the order is not possible without changing the existing schema: `products.stock_quantity` has `CHECK (stock_quantity >= 0)`, and the existing `deduct_inventory_on_order` trigger subtracts the cart quantity on order insert. So an order for more than the stock fails the insert. Selling into negative stock would mean changing that constraint and trigger for the existing POS too, which is out of scope.

**Recorded constraint change:** the PRD's "everything local, no network" promise applies to *detection* and is preserved — the sidecar never touches a network. The desktop↔webapp hop is new and requires the webapp reachable from the desktop. Accepted implicitly by the client's request; reviewers should re-confirm.

## 2. Architecture & deployment

```
 Counter tablet (browser)          SCANnCART desktop (Electron)         pushcart-web host (Next.js + Supabase)
┌────────────────────────┐        ┌──────────────────────────┐         ┌─────────────────────────────────┐
│ Start shopping →       │        │ Main process:            │  poll   │ GET  /api/pos/session?station=… │
│ creates cart + station │        │  posSession.ts poll loop │◀═══════▶│ POST /api/pos/sync (reconcile)  │
│ session (cookie auth)  │        │  cartState.ts derivation │  sync   │ POST /api/protected/… (tablet)  │
│ Live cart page:        │        │  (sidecar REST + /logs)  │         │  reconcile cart_items to match  │
│  items · total · remove│◀──────▶│ Renderer: Live view POS  │         │  provenance: cart_items.session │
│  Finish → order        │        │  panel (status readout)  │         │  _ref = this session's ref      │
└────────────────────────┘        └──────────┬───────────────┘         └─────────────────────────────────┘
                                             │ localhost WS + REST (UNCHANGED)
                                    ┌────────▼─────────┐
                                    │ sidecar (Python) │ — stays 100% offline; one additive change
                                    └──────────────────┘
```

### 2.1 Deployment model (client-confirmed)
- **One hosting:** the tablet UI ships inside pushcart-web — pages under `app/customer/...`, cookie-authed, reading/writing the same Supabase DB the POS uses. Works unchanged whether pushcart-web runs in Docker on the shop LAN or on cloud hosting (Vercel/host + Supabase cloud).
- **The desktop points at one URL:** `posBaseUrl` in SCANnCART's config is whichever host runs pushcart-web (`http://192.168.1.20:3000` for LAN Docker, `https://pushcart.example.com` for cloud). Same routes, same `POS_INGEST_SECRET` on both sides. The sync payload is a tiny JSON snapshot (≤200 items), so 1 s polling is fine even over WAN.
- **TLS:** once `posBaseUrl` leaves the LAN (cloud hosting), the secret header must travel over HTTPS — noted in setup docs and validated in "Test connection" (warn on `http://` non-localhost).
- **Tablet independence:** the tablet page must not depend on the desktop being online to render — it reads cart state from the DB like any other page. If SCANnCART is down, the customer still sees their items and can finish manually.

Binding constraints:
- **A1 — sidecar network-free, and changed in only one additive way.** The Electron main process reads committed state from the sidecar's `GET /api/logs` and can `POST /api/capture/start|stop` (it already spawns the sidecar and knows the port). It uses these to keep capture running and to restart it, not to start and stop capture per customer (§5.3). The one sidecar change is an optional `?since=<ts>` filter on `/api/logs` (**already implemented**: `LoggingStore.query_events(since=)`, tested in `tests/test_logging_store.py` and `tests/test_logs_api.py`). It drops tracks that had left before `ts`, and without it the response is unchanged. The detection pipeline, suppressions and WS protocol stay as they are.
- **A2 — pushcart-web is the sole writer of carts/orders/stock.** SCANnCART only sends *desired state*; the webapp decides writes.
- **A3 — auth split.** Desktop routes live under `/api/pos/*` (outside the `/api/protected/:path*` cookie matcher in `proxy/auth-middleware/auth-middleware.ts` — desktop has no browser session) and authenticate with an `x-pos-token` header vs. pushcart-web's `POS_INGEST_SECRET` env var. Tablet routes stay under `/api/protected/*` with the existing Supabase cookie auth. If the middleware matcher ever broadens, `/api/pos/*` must stay exempt or token-checked first.
- **A4 — desired-state reconciliation, not events.** The desktop sends a full snapshot of "items on the counter now"; the webapp diffs it against the cart's camera-sourced rows. Re-sending the same snapshot is a no-op. Duplicate rows are prevented by the database, not by the payload: reconcile runs as one Postgres transaction that locks the session row (§3.2), and a partial unique index allows only one camera row per product per cart. Retries, overlapping requests and network blips therefore cannot double-add. This replaces any idempotency-key scheme.
- **A5 — provenance.** Camera-managed `cart_items` rows carry `session_ref`. Reconciliation only ever creates, updates or deletes rows whose `session_ref` equals the active session's. Manual rows (`session_ref IS NULL`) are never touched by the camera.
- **A6 — the customer's edit wins over the camera.** Once a product has been edited on the tablet (quantity changed or removed), the camera stops managing that product for the rest of the session (§3.3). Otherwise the next snapshot would undo the edit, and the tablet and camera would keep overwriting each other every heartbeat.

## 3. Contracts (pushcart-web endpoints)

All desktop-facing routes check `x-pos-token`; all tablet routes are cookie-authed under `/api/protected`.

### 3.1 `GET /api/pos/session?station_id=<id>` (desktop; token)
Returns the open session for that station, `unbound` (200, `data: null`) when none, or `404` for an unknown station:

```json
{ "data": { "session_ref": "scanncart-<station>-<epochms>",
            "cart_id": "…", "cart_code": "…", "cart_status": "active" } }
```

`cart_status` is the live `carts.status` value, which is constrained to `unpaid | paid | active`. (The `cart_status` enum type in the initial migration, which includes `completed`, is not what `carts.status` uses.) The desktop watches for `paid` and then unbinds (§5.3).

### 3.2 `POST /api/pos/sync` (desktop; token) — the heart of the integration

```json
{ "session_ref": "scanncart-…", "station_id": "…",
  "items": [ { "class_name": "safeguard_pure_white_60g", "quantity": 2, "max_confidence": 0.91 } ] }
```

The route validates the payload in TypeScript (`app/api/model/pos_sync.ts`). Everything after validation happens in **one Postgres function**, `pos_reconcile(p_session_ref text, p_station_id text, p_items jsonb) returns jsonb`, called with `supabase.rpc`. Doing the read, diff and write as separate supabase-js calls is what allows two overlapping syncs, or a sync racing Finish, to double-insert a row or write into an already-ordered cart. Inside the function:
1. *(route)* Validate: at most 200 items; `class_name` an exact slug; `quantity` ≥ 1; the same class twice → 400.
2. `SELECT … FROM station_sessions WHERE session_ref = p_session_ref FOR UPDATE`. A missing or non-`open` session, or a `station_id` that doesn't match, returns `{error:'session_closed'}`, which the route maps to `409`. Then check the cart's status in the same transaction: `paid` → `409 Cart already paid`. This lock is what serializes reconcile against Finish and against tablet edits. Both of those take the same row lock first (§3.3).
3. Map slugs to products through `product_class_map` **only**. There is no fallback to matching `products.name`: a slug like `safeguard_pure_white_60g` almost never equals a display name, and when it does the match can be ambiguous. Unmapped classes are reported per item as `unmapped` and never inserted.
4. **Several classes may map to one product** (for example, two packagings of the same SKU). Their quantities are **summed** into one desired quantity for that product.
5. **Products the camera no longer manages.** Skip, and report as `overridden`, any product that has a row in `station_session_overrides` for this session, or a manual row (`session_ref IS NULL`) in this cart (A5, A6).
6. Compare the desired map `{product_id: qty}` with the existing `cart_items` rows `WHERE cart_id = … AND session_ref = p_session_ref`:
   - product desired but no row → `INSERT … ON CONFLICT (cart_id, product_id) WHERE session_ref IS NOT NULL DO UPDATE SET quantity = EXCLUDED.quantity`;
   - both exist → update the quantity if it differs;
   - row exists but product not desired → **delete** (the camera says it's gone). Manual rows are never touched (A5).
7. A mapped product with `stock_quantity` below the desired quantity is still applied, flagged `low_stock_warning: true` (D5). Finish is where the stock check actually happens.
8. Append the payload and results to `pos_sync_log` in the same transaction. If this sync **changed** any `cart_items` row, also set `station_sessions.last_activity_at = now()`. A no-op sync (the 15 s heartbeat, an unchanged snapshot) does not count as activity, because a walked-away customer's items left on the counter would otherwise keep the session alive indefinitely.
9. Return `{ results: [{class_name, status: added|updated|removed|unmapped|overridden|warned}], cart_totals: {item_count, subtotal} }`, which the route wraps as `200 { data: … }`. The subtotal is computed from `products.price` on the server, so the tablet and the sensor can't disagree.

### 3.3 Tablet routes (`/api/protected`, cookie auth)
**Which database role these routes run as.** `createClient()` (`config/index.ts`) builds the Supabase client from the request's cookies. On a tablet route the cookie holds the anonymous customer's session, so every query runs as the `authenticated` role under RLS, **not** with the service key. The desktop routes send no cookie, so they do run with the service key. As a result, every tablet-side read or write of the new tables goes through a `SECURITY DEFINER` function. Each of those functions **checks that the caller owns the cart** (`carts.customer_id = auth.uid()`) before doing anything; any other caller gets `{error:'forbidden'}`, which the route maps to `403`. Without that check, any signed-in customer could edit or finish any `cart_id`.

**Starting a session takes two calls,** because the middleware rejects every `/api/protected/*` request that has no auth cookie, and a new customer has none yet:
1. The existing `POST /api/auth` with body `{"type":"customer-sign-in"}` (`signInCustomer`): signs in anonymously, sets the cookie, and creates the cart with `status: 'active'`. This is unchanged. The station-session route does **not** create a second cart.
2. `POST /api/protected/station-session` — body `{station_id, cart_id}`. Calls `pos_open_session(p_station_id, p_cart_id)`, which:
   - checks the caller owns the cart and the cart is `active`;
   - checks the station exists (`stations`, §4.1): unknown → `404`;
   - **handles an abandoned session:** locks any `open` session on this station (`FOR UPDATE`). If its `last_activity_at` is older than `p_idle_cancel_minutes`, it sets it to `cancelled` with `ended_at = now()` and continues. Otherwise it returns `{error:'busy'}` → `409` ("counter busy"). The route passes `p_idle_cancel_minutes` from the `POS_IDLE_CANCEL_MINUTES` env var (default **5**). The cancelled session's cart stays `active` with no order, so nothing is charged, and the desktop unbinds on its next session poll and binds to the new `session_ref`. The abandoned customer's items still on the counter are not carried over, because they entered before the new `bindAt` (§5.2);
   - inserts `station_sessions {station_id, cart_id, session_ref, status:'open', last_activity_at: now()}`.

   Returns `{session_ref, cart_id, cart_code}`, where `cart_code` is `carts.code_token` (also in §3.1). A partial unique index allows **one open session per station** (`CREATE UNIQUE INDEX … ON station_sessions(station_id) WHERE status = 'open'`). That index is the backstop: two tablets racing for one station still cannot both open a session, and the loser gets `409` ("counter busy").
- `GET /api/protected/station-session` calls `pos_my_session()`, which returns the caller's own open session (matched on `auth.uid()`), or nothing. The tablet uses it after a page reload.
- `GET /api/protected/stations` returns `[{id, name}]` for the station picker. `stations` has a read-only `SELECT` policy for `authenticated`, because station names are not sensitive. The picker runs after step 1, so the customer already has a cookie.
- Reuse the existing `GET /api/protected/cart_items/[id]/[status]` for the live item list (read-only).
- **Customer edits go through their own route, not the generic cart-items `PUT`/`DELETE`.** `PUT`/`DELETE /api/protected/station-session/items/[productId]` (body `{cart_id, quantity}` for `PUT`) calls `pos_customer_edit(p_cart_id, p_product_id, p_quantity)`. That function checks the caller owns the cart, takes the session's `FOR UPDATE` lock, sets `last_activity_at = now()`, upserts `station_session_overrides (session_ref, product_id)`, and then:
  - `quantity > 0`: turns that product's camera row into a manual row with the new quantity (`session_ref = NULL`), or inserts a manual row if there was none;
  - `quantity = 0` / `DELETE`: deletes the product's rows. The override is what remembers the removal, since no row is left.

  It appends `{kind:'customer_edit', product_id, from_qty, to_qty}` to `pos_sync_log`. The generic routes would edit the row without recording an override, so the next sync would put the item straight back (A6).
- `POST /api/protected/station-session/finish` — body `{cart_id}`. Calls `pos_finish(p_cart_id)`, which in **one transaction**:
  - checks the caller owns the cart, locks the open session (`FOR UPDATE`), and checks the cart is `active`;
  - **checks stock before writing anything:** if any `cart_items.quantity > products.stock_quantity`, it returns `{error:'insufficient_stock', items:[{product_id, name, in_cart, in_stock}]}` without writing anything. The route maps this to `409`, and the tablet shows "These items need staff: …". Without this check the order insert would fail on the `stock_quantity >= 0` constraint (D5) with a generic 500;
  - computes `subtotal` from `cart_items × products.price`, then `vat_amount = round(subtotal × rate / 100, 2)` using the active `vat_rates.rate` (a percentage: `12.00` means 12%), and `total_amount = subtotal + vat_amount`. This is the same rule as the POS helper `app/user/[userId]/pos/helpers/calculateVat.ts`;
  - inserts the `orders` row with `user_id = carts.user_id`, the anonymous customer. The existing `on_order_created_deduct_stock` trigger fires on this insert. It deducts the stock **and sets the cart to `paid` itself**, so `pos_finish` does not need to update the cart;
  - marks the session `completed`.

  It returns the order id. **It does not reuse `addOrders`**, which marks the cart `paid` *before* inserting the order as two separate calls with no transaction. If that insert failed, the cart would be paid with no order, and every later sync would get a 409. (The POS used to add `rate` to the total as a flat amount, so ₱12 instead of 12%. That was fixed in pushcart-web on 2026-10-01; orders created before the fix keep their stored `vat_amount`.)
- **The shared tablet signs out after every customer.** After Finish succeeds, the confirmation screen calls the existing `signOut` and returns to `scan-start` after a short countdown. `scan-start` also signs out any session left over (an abandoned customer) before creating a new one. Without this, the anonymous customer's cookie stays in the browser and the next customer starts out logged in as them.

### 3.4 Error map (desktop routes)
401 `unauthorizedResponse` (bad token) · 400 `validationErrorNextResponse` · 404 `notFoundResponse` (station not in `stations`) · 409 `conflictRequestResponse` (session closed, cart paid) · 500 `generalErrorResponse`. Tablet routes use the same helpers, plus **403** when the caller does not own the cart and **409** "counter busy" for a second open session on a station. Success helpers return `{ message, data }` where `data` is an **object** — do not copy the `response.data.data as Carts[]` array assumption from `getCarts`.

## 4. pushcart-web changes

1. **Migration** `supabase/migrations/<ts>_pos_self_checkout.sql`:
   - `stations (id TEXT PK, name TEXT NOT NULL, created_at, updated_at)` + `set_updated_at` trigger. This is the station registry. Admins create stations on the mapping screen (§4.5), and each SCANnCART desktop is configured with one station's `id` (§5.1). It has a `SELECT` policy for `authenticated` (the tablet's station picker) and no write policies.
   - `product_class_map (class_slug TEXT PK, product_id UUID FK→products ON DELETE CASCADE, note TEXT, created_at, updated_at)` + `set_updated_at` trigger (helper already exists).
   - `station_sessions (id UUID PK, station_id TEXT NOT NULL FK→stations, session_ref TEXT NOT NULL UNIQUE, cart_id UUID FK→carts ON DELETE CASCADE NOT NULL, status TEXT CHECK (status IN ('open','completed','cancelled')) DEFAULT 'open', created_at, last_activity_at TIMESTAMPTZ NOT NULL DEFAULT now(), ended_at)` + the partial unique index from §3.3.
   - `ALTER TABLE cart_items ADD COLUMN session_ref TEXT NULL;` + index on `(cart_id, session_ref)` (reconcile filters on it) + `CREATE UNIQUE INDEX cart_items_camera_product ON cart_items(cart_id, product_id) WHERE session_ref IS NOT NULL` (at most one camera row per product per cart, which is what reconcile's `ON CONFLICT` relies on).
   - `station_session_overrides (session_ref TEXT FK→station_sessions(session_ref) ON DELETE CASCADE, product_id UUID FK→products, created_at, PRIMARY KEY (session_ref, product_id))`.
   - Functions, split by who calls them, because the two kinds of route run as different database roles (§3.3):
     - **Desktop:** `pos_reconcile` (§3.2) is `SECURITY INVOKER` and only the service role may call it (`REVOKE EXECUTE … FROM PUBLIC, anon, authenticated`). The desktop routes send no cookie, so they run with the service key.
     - **Tablet:** `pos_open_session`, `pos_my_session`, `pos_customer_edit` and `pos_finish` (§3.3) are `SECURITY DEFINER` with `SET search_path = public`. The first statement of each checks that the caller owns the cart (`carts.customer_id = auth.uid()`), or matches on `auth.uid()` in `pos_my_session`'s case. Grant: `REVOKE EXECUTE … FROM PUBLIC, anon; GRANT EXECUTE … TO authenticated`.
   - RLS: no policies on the new tables apart from the `stations` read. The tablet reaches the others only through the `SECURITY DEFINER` functions above, and the desktop with the service key. `cart_items.session_ref` inherits the existing table policy. *Known gap, pre-existing:* `cart_items`' existing policies let any `authenticated` user write any row (`WITH CHECK (true)`). The generic cart-items routes are not used by the tablet (§3.3), but tightening those policies is out of scope here.
   - *Note:* "server-route-only" holds only while the Supabase secret key stays out of the browser. **Done on 2026-10-01:** the key was renamed from `NEXT_PUBLIC_SUPABASE_SERVICE_SECRET_KEY` to the server-only `SUPABASE_SECRET_KEY` (`config/index.ts`, `config/updateSession.ts`, `Makefile`, `scripts/cleanup-anonymous-users.mjs`, local `.env`). The unused `NEXT_PUBLIC_SUPABASE_SERVICE_ROLE_KEY` passthrough was removed from `next.config.ts`. A production build was checked: no `sb_secret_` string in `.next/static` or `.next/server/app`. **Every deployed environment (Docker env file, Vercel project settings) must rename the variable too,** or the server client will refuse to start.
2. **`app/api/model/pos_sync.ts`** — `getSessionForStation`, plus thin `rpc()` wrappers for the five functions. It follows the `app/api/model/*.ts` pattern (`createClient()` from `@/config`, response helpers) and maps each function's `{error}` codes to the §3.4 helpers.
3. **Routes**: `app/api/pos/session/route.ts`, `app/api/pos/sync/route.ts` (token check inline, both reading `process.env.POS_INGEST_SECRET`), `app/api/protected/station-session/route.ts` (`POST` + `GET`), `app/api/protected/stations/route.ts`, `app/api/protected/station-session/items/[productId]/route.ts`, `app/api/protected/station-session/finish/route.ts`.
4. **Customer tablet UI** `app/customer/[userId]/[cartId]/scan/page.tsx`: live item list (poll cart_items every ~2 s or reuse any existing polling hook in `services/cart/states`), per-item remove and quantity buttons that call the station-session items route (§3.3), so an edit records an override, never the generic cart-items routes. A running total computed with the same rule `pos_finish` uses. **Finish** button → finish route → confirmation screen → sign out → back to `scan-start` (§3.3). Plus `app/customer/[userId]/scan-start/page.tsx`. *Start shopping* runs: sign out any leftover session → `POST /api/auth {type:'customer-sign-in'}` (anonymous sign-in, which creates the cart) → station picker from `GET /api/protected/stations` (v1 auto-picks when there is only one) → `POST /api/protected/station-session {station_id, cart_id}` → redirect to the scan page. On `409` it shows "counter busy". Styling consistent with the existing customer shop page. Mobile/tablet layout first. If the desktop stops syncing, the scan page shows a "camera offline — ask staff to add items" notice, and the item list, edits and Finish keep working (§2.1). The running total reads the active rate from the existing `GET /api/protected/vat/all`, which now returns only the active rate.
5. **Admin mapping screen** `app/admin/[userId]/pos-mapping/page.tsx` + `services/pos/pos.services.ts`:
   - CRUD on `stations` through `/api/protected/pos-stations` (+`[id]`). The station `id` shown here is what an operator pastes into the SCANnCART Admin Panel (§5.5).
   - CRUD on `product_class_map` (joined to products) through protected routes `/api/protected/pos-mapping` (+`[id]`). Several slugs may map to one product (§3.2 step 4).
   - A "recent unmapped classes" panel reading recent sync results from the audit table (below), with one-click "map to…" prefill.
   - An **open sessions** list with a *Cancel* button that sets a stuck session to `cancelled` (§6, abandoned tablet).
   - A **customer edits** list (`pos_sync_log` rows with `kind = 'customer_edit'`) for reviewing shrinkage.
6. **Audit:** `pos_sync_log (id UUID PK, session_ref TEXT, kind TEXT CHECK (kind IN ('sync','customer_edit')), payload jsonb, results jsonb, created_at)`. One row is appended per sync and per customer edit, inside the same transaction as the change. It powers the unmapped-classes panel and debugging. It is also the shrinkage trail: under D1–D3 a customer can remove a camera-detected item and finish without paying for it, and `customer_edit` rows are the only record that this happened. No policies.
7. `.env`: `POS_INGEST_SECRET=<32+ hex>` and, optionally, `POS_IDLE_CANCEL_MINUTES=5` (repo has no `.env.example`; document both in its README).

## 5. SCANnCART (this repo) changes — in `desktop/`, plus the already-done `/api/logs?since=` (A1)

1. **`desktop/src/main/posConfig.ts`** — reads/writes `pos.json` in `app.getPath('userData')`: `{ posBaseUrl, posSecret, stationId, commitDwellS=3, removeSettleS=10, minCommitConf=0.6, unboundPollMs=1000, sessionPollMs=5000, logsPollMs=1000 }`. `minCommitConf` is a starting value to be tuned during the joint smoke test (§7). `stationId` is **configured, not generated**: it must be the `id` of a row in pushcart-web's `stations` table (§4.1), copied from its admin screen. A `404` from "Test connection" means the station isn't registered. Saving refuses `commitDwellS ≤ track_expiry_s` (the §5.2 invariant). Empty `posBaseUrl`/`posSecret`/`stationId` = feature disabled. Async getters/setters; unit-tested against a temp dir. "Test connection" warns when `posBaseUrl` is plain `http://` and not localhost (TLS rule, §2.1).
2. **`desktop/src/main/cartState.ts` (pure, heavily tested)** — `deriveCartState(events: TrackEvent[], now, bindAt, cfg) → Map<class, {quantity, maxConfidence}>`, where `TrackEvent` is a sidecar `LogEvent` keyed by `(sidecarSessionId, track_id)` so events from more than one sidecar session can be passed together (see §5.3, restarts). All timestamps are the sidecar's `time.time()` seconds, and the desktop reads `now` from the same machine's clock, so the two are directly comparable.

   **Quantity is the number of tracks open at the same time, not the number of tracks.** The sidecar's tracker routinely ends one track and starts another for the same physical item: a hand passes over it, the customer lifts it and puts it back, or detection drops for longer than `track_expiry_s` (1.5 s by default). Counting tracks across a time window would show that one item as 2 every time this happens. The rule:
   - **Eligible track:** `entered_at ≥ bindAt` (items already on the counter when the session bound belong to nobody) and `max_conf ≥ minCommitConf`.
   - **Committed interval:** a track counts over `[entered_at + commitDwellS, end]`, where `end` is `left_at`, or `now` while `left_at` is null. A track whose `end` comes before `entered_at + commitDwellS` never counts (this filters out detector flicker).
   - **Concurrency:** `c(t)` = the number of a class's committed intervals that cover time `t`.
   - **Quantity:** `quantity(class) = max over t in [now − removeSettleS, now] of c(t)`. A count goes up once the extra track has been on the counter for `commitDwellS`, and goes down only after the lower count has held for a full `removeSettleS`. When the tracker swaps one track for another, the old interval ends at the item's last-seen time (the sidecar backdates `left_at` to it) and the new interval starts `commitDwellS` later. The gap between them is shorter than the settle window, so the item never reads as 2 and never drops out of the cart.
   - **Invariant: `commitDwellS > track_expiry_s` (+ one inference frame).** While the sidecar waits to declare a track expired, its `left_at` is still null and the track reads as open until `now`. A replacement track becomes committed only after `commitDwellS`, which is later than the old track's expiry, so the two never overlap. `posConfig` refuses to save a value that breaks this rule, and "Test connection" also checks it against the sidecar's live `track_expiry_s` from `GET /api/settings`.
   - `maxConfidence` = the highest `max_conf` among the class's tracks that count at `now`.
   - Classes with `quantity = 0` are left out, so an empty map means "nothing on the counter".

   This is the check that replaces the cashier: the sidecar's `conf_threshold`, then `minCommitConf`, then the dwell, then the settle window.
3. **`desktop/src/main/posSession.ts`** — the orchestrator (own loop, fake-timer tests).
   - **Capture is always on, not started per customer.** When the POS feature is enabled, the orchestrator makes sure capture is `running` (`POST /api/capture/start` if it is idle) and leaves it running between customers. Starting capture per customer would cost the StreamCam's ~37 s device open on every *Start shopping*. It would also sync the previous customer's tracks if capture had been left running, because `/api/logs` returns the whole current sidecar session. Binding is a timestamp instead (`bindAt`), and the `entered_at ≥ bindAt` rule in §5.2 separates one customer from the next.
   - **Session polling.** Poll `GET /api/pos/session?station_id=…` every `unboundPollMs` (1000 ms) while unbound, so items placed right after *Start shopping* are not filtered out by a late `bindAt`, and every `sessionPollMs` (5000 ms) while bound. When a new `session_ref` appears, bind and set `bindAt = now`. When the server reports no session, the session closed, or the cart `paid`: unbind and stop syncing, but do **not** stop capture. A final sync that comes back 409 means the session is over.
   - **Sync loop while bound.** Every `logsPollMs`, fetch `/api/logs?since=<bindAt>` (see *Log growth* below) and merge it into a per-binding event cache keyed by `(session_id, track_id)`. Then run `deriveCartState` and `POST /api/pos/sync` only when the derived map changed, plus a 15 s heartbeat. Allow only one sync request in flight: a change that arrives during a request replaces the queued snapshot instead of starting a second request. The cache is cleared on unbind.
   - **Capture restarts during a session.** A restart creates a new sidecar session, and `/api/logs` from then on returns only that session's events. The cache still holds the earlier session's tracks, which `resolve_open_tracks()` closed with `left_at` set to their last-seen time. The camera is blind for the length of the restart (~37 s), which is longer than the settle window, so the derived state would drop to zero. To stop that, a **warm-up guard** applies until capture has been running continuously for `commitDwellS + removeSettleS`: the snapshot sent is the per-class `max(lastSent, derived)`. Counts can go up during warm-up but cannot go down. The same guard applies whenever capture is not `running`. No snapshot is ever derived from a capture that isn't observing the counter.
   - **Failure policy.** Webapp unreachable: retry with backoff, keep the last state, and surface `pos_sync_error` to the renderer. Capture died: the main process has no WebSocket, so it learns this from `GET /api/health`. Any `state` other than `running` (or `starting`) while POS is enabled means capture is down, because a pipeline error tears down to `idle`. Reuse `SidecarHealthMonitor`'s probe loop or poll alongside it. Then try **one** auto-restart. A `409` from `/api/capture/start` means calibration is in progress: wait for it to finish rather than counting it as the failed restart. If that fails, stay bound with the warm-up guard holding the cart at its last counts, and show the warning. Corrections then happen on the tablet.
   - **Log growth.** Capture runs all day, so the sidecar session grows by one row per track. The sync loop therefore always fetches `/api/logs?since=<bindAt>`. That returns only the tracks still present at or after bind time, which is everything `deriveCartState` can count (eligibility already requires `entered_at ≥ bindAt`). The payload is the size of one customer's session however long capture has been up. No capture recycle is needed. Use the response's `session_id` to detect a sidecar session change (a restart) and to key the cache.
   - Emits `pos:state` (`unbound | warming_up | bound | error`, cart code, synced item count, last sync age) to the renderer via webContents (pattern: `sidecar:health`).
4. **Preload** (`desktop/src/preload/index.ts` + `index.d.ts`): `pos:get-state` (invoke), `pos:get-config`/`pos:save-config` (invoke), `pos:on-state` (event listener, mirroring `onSidecarHealth`). Expose nothing else. `sidecar/tests/test_desktop_contracts.py` already requires every `ipcMain.handle` channel to be called by the preload, so the new channels are covered automatically. Update CLAUDE.md's statement that the port and health are the *only* bridges, in the same change.
5. **Renderer:**
   - Live view: small **POS panel** in the stats/side area — bound/unbound, cart code, item count synced, last sync age, sync errors. Read-only readout.
   - Admin Panel: **POS integration** section — base URL, secret, station id (a text input pasted from pushcart-web's station list), dwell/settle/poll inputs, "Test connection" button hitting `GET /api/pos/session` once and reporting (including the http/https warning).
6. **Docs** `docs/POS_INTEGRATION.md`: setup (webapp URL, secret, station pairing, LAN-vs-cloud deployment note from §2.1), the operator-less flow, troubleshooting (401 secret mismatch · 409 session closed/cart paid · `unmapped` items need admin mapping · sync error = webapp unreachable). JSON fences only — no shell commands in fenced blocks — so the repo's doc-fence test readers stay green.

## 6. Edge cases the implementer must handle (test-backed)

- **Flicker:** a track shorter than the dwell never counts, and a committed count that dips for less than the settle window never goes down. (Unit tests in `cartState.test.ts`.)
- **Track swap on the same item:** a hand passes over the item, or the customer lifts it and puts it back, so the tracker ends the old track and starts a new one. The quantity stays 1 the whole time, never 2 and never 0 (§5.2). Tests cover a gap shorter than the settle window, a gap longer than it (the item is removed and then re-added), and the `commitDwellS > track_expiry_s` boundary.
- **Two of the same product:** two tracks open at the same time count as quantity 2. If one is taken away, the quantity drops to 1 after the settle window.
- **Leftovers from the previous customer:** tracks with `entered_at < bindAt` are excluded. If the tracker swaps one of those leftovers' tracks after binding, the new track counts. The tablet's remove button is how that gets corrected.
- **Camera dies mid-session:** capture auto-restarts once. The cached events from the old sidecar session plus the warm-up guard keep every count where it was, so nothing is removed while the camera is blind. If the restart fails, the cart stays at its last counts and the tablet still shows them.
- **Webapp dies mid-session:** backoff retries; on reconnect the next snapshot converges the cart (A4) — no duplicate rows possible.
- **Two tablets / double Start:** partial unique index rejects the second open session per station; tablet shows "counter busy".
- **Finish while the camera still sees items:** Finish wins. The session becomes `completed` and the cart `paid`, the next desktop sync gets 409, and the desktop unbinds (capture keeps running). Rows already in the cart stay as ordered.
- **Manual rows:** reconcile never touches `session_ref IS NULL` rows (A5), and it also skips their product entirely, so the camera never adds a second row next to a staff add.
- **Customer edits a camera-detected item on the tablet:** the product moves to manual with an override (§3.3). Later snapshots report it as `overridden` and leave it alone, even if the camera still sees it. A removed item is not re-added. If the customer later puts a second unit of an overridden product on the counter, the camera won't add it, and the customer increments the quantity on the tablet.
- **Concurrent writers:** two overlapping syncs, a sync racing Finish, or a sync racing a tablet edit all take the same session row lock and run one after another. A sync that arrives after Finish sees `paid` and gets a 409. It never writes into an ordered cart.
- **Not enough stock at Finish:** `pos_finish` refuses with `409 insufficient_stock` and the list of items. Nothing is written, so the cart stays `active` and the session `open`. Staff fix it (adjust the item or restock) and the customer taps Finish again.
- **Order insert fails at Finish:** the transaction rolls back. The cart stays `active`, the session stays `open`, and the customer can try again.
- **Abandoned tablet:** the next *Start shopping* signs out the leftover browser session. If the abandoned station session has been idle for `POS_IDLE_CANCEL_MINUTES` (default 5), `pos_open_session` cancels it and starts the new one (§3.3). If it was active more recently, the new customer sees "counter busy", which protects a customer who only stepped away. Activity means a sync that changed the cart, or a customer edit; heartbeats don't count. Staff can still cancel a session at any time from the admin screen's open-sessions list.
- **Unmapped class, or an item the camera missed:** an unmapped class is reported on each sync, shown on the admin panel, and never inserted. The tablet shows "N items not recognized — please ask staff". **v1 has no customer-side product picker**: a staff member adds the item manually (a manual row, A5). A customer-side picker is parked (§9).
- **Cart ownership:** a customer calling a tablet route with someone else's `cart_id` gets `403`, and nothing is read or written.
- **Unregistered station:** the desktop's "Test connection" and session poll get `404`. The tablet picker never offers that station.
- **Cloud-hosted webapp:** desktop polls over WAN — keep payloads snapshot-small; no behavior change otherwise.

## 7. Testing & acceptance

- **pushcart-web** (no test runner configured): clean `yarn build` + `yarn lint`; `scripts/pos-e2e.sh` (curl) walking the happy path: seed a `stations` row → `/api/auth` anonymous sign-in (captures cookie and `cart_id`) → `POST /api/protected/station-session {station_id, cart_id}` → token sync snapshot ×2 (adds, then no-op) → snapshot with an item removed (verifies delete) → customer edit on the tablet route, then re-send a snapshot containing that item (verifies `overridden`, not re-added) → finish → verify the `orders` row, the `stock_quantity` decrement, cart `paid`, session `completed`, and that a sync now gets 409. Also: wrong token → 401; second Start within the idle window → 409; second Start after the idle window (set `last_activity_at` back in the test) → old session `cancelled`, new one `open`, and a sync on the old `session_ref` → 409; a heartbeat-only sync does not move `last_activity_at`; a second anonymous customer (own cookie) calling edit/finish on the first customer's `cart_id` → 403 with the cart unchanged; an unknown `station_id` → 404 on both the desktop and tablet routes; unmapped class reported; two classes mapped to one product are summed; two identical syncs sent in parallel (`&` + `wait`) leave exactly one row per product; a forced order-insert failure leaves the cart `active`; a cart holding more of a product than its `stock_quantity` gets `409 insufficient_stock` at Finish, with no order written and stock unchanged.
- **SCANnCART**: vitest — `cartState.test.ts` (the critical pure logic: dwell, settle, concurrent-track counting, track swaps, `bindAt` exclusion, `minCommitConf`, events from two sidecar sessions together), `posSession.test.ts` (fake timers: ensure capture is running → bind sets `bindAt` → sync-on-change with only one sync in flight → paid → unbind with capture still running; webapp-down backoff; capture death → restart once → warm-up guard holds counts; every logs fetch carries `since=bindAt`), `posConfig.test.ts` (temp dir), `posRoutes.integration.test.ts` (the desktop's two POS routes over real HTTP against a local stand-in for pushcart-web — the smoke test's §3 rows a machine can hold: a dead host's backoff countdown, the host coming back with no duplicate rows, a 409 that ends the binding, and the 401/404 the panel names; `make verify-pos-routes` points the same protocol checks at a real pushcart-web, since the stand-in is only as faithful as the routes it was written from; `posContract.test.ts` pins both modules to a contract recorded from pushcart-web's own route source, and `make verify-pos-contract` re-reads that source so the record cannot go stale; pushcart-web guards that same surface in its own CI, deriving it from its own routes so a PR that moves a route, the `x-pos-token` header or a status code fails where the change is made rather than only here), Admin Panel POS-section tests. `npm run typecheck` + `npm test` + `npm run lint` + `make docs-sync-check` green.
- **Joint smoke** (§8 step 3): tablet Start → items placed on the real counter, with the StreamCam aimed at it, appear on tablet with totals → item taken back disappears after settle → item lifted and set back down stays at quantity 1 → Finish → order exists, stock decremented, SCANnCART unbinds on its own (capture stays running) → the next customer's *Start* does not pick up items left on the counter.

### 7.1 Counting accuracy: proving add and remove work — *recommendation, not in v1 scope*

> **Status: recommended, not required.** Nothing below blocks implementing §4–§5 or passing §7's checks. It is the advised way to prove, before a real shop relies on it, that the cart ends up right. Adopting it (or any part of it) is a separate decision for the client and the team.

The model only answers "which products are in this frame, and where". Adding and removing an item is something §5.2 *infers* from how detections persist over time, so that inference must be measured on its own. The model's per-class recall (`--val`, `RECALL_FLOOR`) does not answer "did the customer's cart end up right". A product the model sees in 85% of frames can still be counted perfectly, because the dwell and settle windows average over many frames. A product the model sees in 99% of frames can still be counted wrong, when two identical items touch and come back as one box.

**What to measure.** The number that matters is **final-cart exact match**: at Finish, does the cart hold exactly the products and quantities that were on the counter? Report alongside it, per scenario: false adds (in the cart, never placed), false removes (placed and still there, but dropped from the cart), add latency (placed → in the cart; this should land near `commitDwellS`) and remove latency (taken away → gone; near `removeSettleS`).

**Scenario corpus (record once, replay forever).** Record short videos with the real StreamCam, mounted where it will be at the counter, each following a written script. The ground truth is a list of checkpoints `{t_s, cart: {class: qty}}` saying what the cart should hold at each moment, with dwell and settle latency allowed for. Minimum set:
1. One item placed, left there, Finish.
2. Three different products placed one by one.
3. Two identical items, apart; then two identical items **touching**.
4. Place three, take one back.
5. Lift an item and put it back (the track-swap case, §5.2).
6. A hand passes over an item for 1–3 s (occlusion).
7. Items **stacked** or overlapping. This is expected to under-count; the run documents how badly.
8. An item held above the counter and taken away without being set down.
9. The customer's own things only: phone, wallet, bag, receipt. Nothing may be added.
10. Previous customer's leftover still on the counter at *Start* (§6).
11. Each product in the roster at least once, in more than one orientation (label facing up, sideways, back).

**Two harnesses, one corpus:**
- **Replay through the real pipeline** — `sidecar/tools/replay_scenarios.py` (new; it lives in `tools/` and is not part of the runtime). It feeds each video through `Pipeline` with the installed weights, a frame source that reads the file, and a clock driven by the video's timestamps, so a replay is deterministic. It writes the resulting `LogEvent`s to `desktop/src/main/__fixtures__/scenarios/<name>.json` beside the script's checkpoints. Re-run it whenever the weights change (v2): it is the regression test for the *model's* effect on counting.
- **Score the counting logic** — `cartState.scenarios.test.ts` loads every fixture, runs `deriveCartState` at each checkpoint time, and asserts the cart. It runs in `npm test` with no camera or GPU, so tuning `commitDwellS`/`removeSettleS`/`minCommitConf` is a test run, not a trip to the counter. A real fixture captures the tracker's actual habits (track swaps, short flickers), which hand-written events miss.

**Suggested go-live gate (if adopted; client to confirm the numbers):** final-cart exact match on **≥ 95%** of the corpus, **zero** false adds in scenario 9, and every miss explained by a scenario the counter guidance below rules out (stacking). A run below the gate goes back to tuning or retraining, not to the shop.

**Counter setup that makes counting tractable** (the cheap mitigations, before any hardware):
- **The counter is the frame.** Aim and crop the camera so its view is the marked counter area and nothing else: no shelves, no queue, no bagging area. `/api/logs` carries no box positions, so the desktop cannot ignore part of the frame. The camera's aim is the only zone filter.
- **One layer, apart, labels up.** Signage at the counter, and the same instruction on the tablet's scan page. This turns scenarios 3 (touching) and 7 (stacked) from model problems into instructions.
- **The customer is the last check.** The tablet shows the live cart with counts (§4.4), so a miss is visible while the item is still on the counter, and "ask staff" covers whatever the camera cannot fix (§6).

**When that is not enough:** a load cell under the counter (parked in §9) independently checks *how much* is there. The camera says which products and the scale says whether the total weight agrees. A count the camera got wrong (stacked or touching duplicates) disagrees with the weight, and Finish can then ask for staff. That is the standard design for unattended checkouts, and it is the upgrade path if the gate cannot be met with the camera alone.

#### 7.2 Weight sensor: what it replaces and what it does not — *recommendation*

A scale checks **quantity**, not **identity**. Adding one removes some of the reasons to retrain the model, but not the main one.

What a scale fixes, with no retraining:
- **Phantom items on an empty counter.** v1 has reported Bear Brand at 0.90–0.94 with nothing placed (`acceptance.py`). An empty counter weighs 0 g, so the scale rejects these outright. That covers much of what v2's hard-negative frames are for. It could also let the frame-filling filter (`suppress_frame_filling_detections`) be turned off: today that filter drops 252 of 2018 real detections (~12%, `tools/unsure_probe.py`) to suppress this phantom.
- **Wrong quantities.** Two touching identical items detected as one, or stacked items: the camera says one, the weight says two, and the mismatch is caught.
- **Removals.** A drop in weight confirms the camera's remove, independently of the tracker.

What a scale cannot fix:
- **Telling products apart.** If the model calls one product another, the scale catches it only when the two products' weights differ by more than the scale's tolerance. Products of similar weight (for example, two 155 g cans) look identical to it.
- **A missed product.** The scale sees weight it cannot account for but cannot say which product it is. The only response is "ask staff". A weak model plus a scale still works, but staff get called often, which defeats an unattended counter.

**Whether to retrain therefore depends on how well the model tells the products apart at the counter, not on the scale.** Two checks decide it:
1. **Per-class recall at the counter's distance.** Run `train_model.py --val` on frames shot at the real counter. The counter is a fixed, fairly close distance, so v2's `far` bucket may matter less here. Any class under `RECALL_FLOOR` (0.85) needs more data whether or not a scale is fitted.
2. **Product weights.** Weigh one unit of each roster product. Any pair within the scale's tolerance plus normal packaging variation cannot be separated by weight, so the model alone has to separate them.

If the model already identifies every roster product reliably at the counter, a scale can stand in for the hard-negative retrain and fix the quantity failures. If the model confuses products, the scale only turns those confusions into staff calls, and retraining is still needed. The §7.1 recordings answer this with numbers: they show whether each failure is an identity error or a count error.

*If adopted, the integration point is pushcart-web, not the sidecar:* a `products.unit_weight_g` column, the scale's reading sent with each sync (or read by `pos_finish`), and a tolerance check before the order is created. The details are out of scope until the client decides to buy the hardware.

## 8. Build order (two-agent split, for whoever implements this)

1. **Agent W (pushcart-web):** §4.1 migration (tables, indexes, the five functions) → §4.2 model+routes → §4.4 tablet pages → §4.5 admin screen → e2e script (§7).
2. **Agent S (SCANnCART):** §5.1 config → §5.2 cartState (pure logic + tests first) → §5.3 orchestrator → §5.4 IPC → §5.5 UI → docs.
3. §7 joint smoke last. The halves parallelize against §3's contracts; only the smoke needs both.
4. *Optional, recommended (§7.1):* record the scenario corpus and add the replay and scoring harness. It needs only the camera and the current weights, so it can start in parallel at any point.

## 9. Parked (explicitly out of scope)
Counting-accuracy harness and go-live gate (§7.1, recommended) · Payment integration (GCash/Maya) · customer-side product picker for items the camera missed (§6) · tightening `cart_items`' existing permissive RLS (§4.1) · QR-to-camera binding · background idle timeout (abandoned sessions are cleared on the next Start instead, §3.3) · multi-counter orchestration beyond `station_id` · weight-sensor cross-check (PRD §future; see §7.2 for what it would and would not replace) · retry/backoff tuning beyond §5.3's simple policy.
