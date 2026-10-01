# Spec: SCANnCART → pushcart-web Automatic Self-Checkout Integration

Status: **Draft for review** · Date: 2026-10-01
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
- D4 **Session trigger: customer taps Start/Finish.** Binding starts capture automatically; Finish ends it. (Idle auto-timeout is a parked alternative.)
- D5 **Zero-stock policy: warn, don't block** — a mapped product at `stock_quantity = 0` still enters the cart with a `low_stock_warning` flag the tablet displays; stock is only enforced by business rules at checkout.

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
                                    │ sidecar (Python) │ — stays 100% offline, ZERO changes
                                    └──────────────────┘
```

### 2.1 Deployment model (client-confirmed)
- **One hosting:** the tablet UI ships inside pushcart-web — pages under `app/customer/...`, cookie-authed, reading/writing the same Supabase DB the POS uses. Works unchanged whether pushcart-web runs in Docker on the shop LAN or on cloud hosting (Vercel/host + Supabase cloud).
- **The desktop points at one URL:** `posBaseUrl` in SCANnCART's config is whichever host runs pushcart-web (`http://192.168.1.20:3000` for LAN Docker, `https://pushcart.example.com` for cloud). Same routes, same `POS_INGEST_SECRET` on both sides. The sync payload is a tiny JSON snapshot (≤200 items), so 1 s polling is fine even over WAN.
- **TLS:** once `posBaseUrl` leaves the LAN (cloud hosting), the secret header must travel over HTTPS — noted in setup docs and validated in "Test connection" (warn on `http://` non-localhost).
- **Tablet independence:** the tablet page must not depend on the desktop being online to render — it reads cart state from the DB like any other page. If SCANnCART is down, the customer still sees their items and can finish manually.

Binding constraints:
- **A1 — sidecar untouched and network-free.** The Electron main process reads committed state from the sidecar's existing `GET /api/logs` and can `POST /api/capture/start|stop` (it already spawns the sidecar and knows the port). No sidecar code changes; detection pipeline, suppressions, and WS protocol stay as-is.
- **A2 — pushcart-web is the sole writer of carts/orders/stock.** SCANnCART only sends *desired state*; the webapp decides writes.
- **A3 — auth split.** Desktop routes live under `/api/pos/*` (outside the `/api/protected/:path*` cookie matcher in `proxy/auth-middleware.ts` — desktop has no browser session) and authenticate with an `x-pos-token` header vs. pushcart-web's `POS_INGEST_SECRET` env var. Tablet routes stay under `/api/protected/*` with the existing Supabase cookie auth. If the middleware matcher ever broadens, `/api/pos/*` must stay exempt or token-checked first.
- **A4 — desired-state reconciliation, not events.** The desktop sends a full snapshot of "items on the counter now"; the webapp diffs it against the cart's camera-sourced rows. Re-sending the same snapshot is a no-op, so retries and network blips can never double-add. This replaces any idempotency-key scheme.
- **A5 — provenance.** Camera-managed `cart_items` rows carry `session_ref`; reconciliation only ever creates/updates/deletes rows whose `session_ref` equals the active session's. Manual rows (`session_ref IS NULL`) are never touched by the camera.

## 3. Contracts (pushcart-web endpoints)

All desktop-facing routes check `x-pos-token`; all tablet routes are cookie-authed under `/api/protected`.

### 3.1 `GET /api/pos/session?station_id=<id>` (desktop; token)
Returns the open session for that station, `unbound` (200, `data: null`) when none, or `404` for an unknown station:

```json
{ "data": { "session_ref": "scanncart-<station>-<epochms>",
            "cart_id": "…", "cart_code": "…", "cart_status": "active" } }
```

`cart_status` reflects the live carts row — the desktop watches for it to become `paid`/`completed` to end capture.

### 3.2 `POST /api/pos/sync` (desktop; token) — the heart of the integration

```json
{ "session_ref": "scanncart-…", "station_id": "…",
  "items": [ { "class_name": "safeguard_pure_white_60g", "quantity": 2, "max_confidence": 0.91 } ] }
```

Server reconcile (in `app/api/model/pos_sync.ts`):
1. Validate (items ≤ 200; `class_name` exact slug; `quantity` ≥ 1; duplicate class → 400).
2. Load the open `station_sessions` row by `session_ref`; missing or not open → `409 conflictRequestResponse` (session closed; desktop stops syncing). Cart must be `active`/`unpaid` — `paid` → `409 Cart already paid`.
3. Map slugs → products via `product_class_map`, fallback exact match on `products.name`; unmapped → per-item `unmapped`, never inserted.
4. Desired map `{product_id: qty}` vs existing `cart_items` rows `where session_ref = this ref`:
   - desired row absent → insert with `session_ref` set;
   - both → update quantity if different;
   - existing not in desired → **delete** (the camera says it's gone). Manual rows untouched (A5).
5. `stock_quantity = 0` on a mapped product → still applied, flagged `low_stock_warning: true` (D5).
6. Respond `200 { data: { results: [{class_name, status: added|updated|removed|unmapped|warned}], cart_totals: {item_count, subtotal} } }` — subtotal computed from `products.price` server-side so the tablet and the sensor can't disagree.

### 3.3 Tablet routes (`/api/protected`, cookie auth)
- `POST /api/protected/station-session` — body `{station_id}`. Creates the cart via the existing anonymous sign-in flow (`signInCustomer` semantics) **plus** a `station_sessions` row `{station_id, cart_id, session_ref, status:'open'}`; returns `{session_ref, cart_id, cart_code}`. Partial unique index: **one open session per station** (`CREATE UNIQUE INDEX … ON station_sessions(station_id) WHERE status = 'open'`).
- `GET` variant returns the tablet's own open session after a page reload.
- Reuse existing `GET /api/protected/cart_items/[id]/[status]` for the live item list and the cart-items `PUT`/`DELETE` routes for manual quantity/remove.
- `POST /api/protected/station-session/finish` — body `{cart_id}`. Validates an open session, creates the order via the **existing** `addOrders` model (which sets cart `paid` and fires the stock trigger), then marks the session `completed`. Returns the order id.

### 3.4 Error map (desktop routes)
401 `unauthorizedResponse` (bad token) · 400 `validationErrorNextResponse` · 404 `notFoundResponse` (no session / no station) · 409 `conflictRequestResponse` (session closed, cart paid) · 500 `generalErrorResponse`. Success helpers return `{ message, data }` where `data` is an **object** — do not copy the `response.data.data as Carts[]` array assumption from `getCarts`.

## 4. pushcart-web changes

1. **Migration** `supabase/migrations/<ts>_pos_self_checkout.sql`:
   - `product_class_map (class_slug TEXT PK, product_id UUID FK→products ON DELETE CASCADE, note TEXT, created_at, updated_at)` + `set_updated_at` trigger (helper already exists).
   - `station_sessions (id UUID PK, station_id TEXT NOT NULL, session_ref TEXT NOT NULL UNIQUE, cart_id UUID FK→carts ON DELETE CASCADE NOT NULL, status TEXT CHECK (status IN ('open','completed','cancelled')) DEFAULT 'open', created_at, ended_at)` + the partial unique index from §3.3.
   - `ALTER TABLE cart_items ADD COLUMN session_ref TEXT NULL;` + index on it (reconcile queries filter on it).
   - RLS: no policies on the new tables (server-route-only, same posture as existing internal tables); `cart_items.session_ref` inherits existing table policy.
2. **`app/api/model/pos_sync.ts`** — `getSessionForStation`, `reconcileCartItems`, following the `app/api/model/*.ts` pattern (`createClient()` from `@/config`, response helpers). Reconcile algorithm exactly as §3.2.
3. **Routes**: `app/api/pos/session/route.ts`, `app/api/pos/sync/route.ts` (token check inline, both reading `process.env.POS_INGEST_SECRET`), `app/api/protected/station-session/route.ts`, `app/api/protected/station-session/finish/route.ts`.
4. **Customer tablet UI** `app/customer/[userId]/[cartId]/scan/page.tsx`: live item list (poll cart_items every ~2 s or reuse any existing polling hook in `services/cart/states`), per-item remove + qty buttons, running total (subtotal + VAT from the existing `vat_rates` usage in orders), **Finish** button → finish route → confirmation screen. Plus `app/customer/[userId]/scan-start/page.tsx`: station picker (v1: single station / auto-pick the only one) → creates session → redirects to the scan page. Styling consistent with the existing customer shop page. Mobile/tablet layout first; scan pages render read-only if the camera side is offline (§2.1).
5. **Admin mapping screen** `app/admin/[userId]/pos-mapping/page.tsx` + `services/pos/pos.services.ts`: CRUD `product_class_map` (joined to products) via protected routes `/api/protected/pos-mapping` (+`[id]`); a "recent unmapped classes" panel reading recent sync results (audit table below) with one-click "map to…" prefill.
6. **Audit:** `pos_sync_log (id UUID PK, session_ref TEXT, payload jsonb, results jsonb, created_at)` — one append per sync; powers the unmapped-classes panel and debugging. No policies.
7. `.env`: `POS_INGEST_SECRET=<32+ hex>` (repo has no `.env.example`; document in its README).

## 5. SCANnCART (this repo) changes — all in `desktop/`, sidecar untouched

1. **`desktop/src/main/posConfig.ts`** — reads/writes `pos.json` in `app.getPath('userData')`: `{ posBaseUrl, posSecret, stationId, commitDwellS=3, removeSettleS=10, sessionPollMs=5000, logsPollMs=1000 }`. `stationId` is a generated UUID on first run. Empty `posBaseUrl`/`posSecret` = feature disabled. Async getters/setters; unit-tested against a temp dir. "Test connection" warns when `posBaseUrl` is plain `http://` and not localhost (TLS rule, §2.1).
2. **`desktop/src/main/cartState.ts` (pure, heavily tested)** — `deriveCartState(events: LogEvent[], now, cfg) → { items: Map<class, {quantity, maxConfidence}> }` from the sidecar's `/api/logs` events: a track counts only once `now - entered_at ≥ commitDwellS` (kills detector flicker), keeps counting while open **and** for `removeSettleS` after `left_at` (kills remove/re-add bounce), then drops. `quantity` = number of qualifying tracks per class; `max_confidence` = max of theirs. This is the "verification" layer that replaced the cashier: confidence threshold (already a setting) + dwell + settle.
3. **`desktop/src/main/posSession.ts`** — the orchestrator (own loop, fake-timer tests):
   - Poll `GET /api/pos/session?station_id=…`. **No session:** if capture is running, stop it (sidecar REST); idle otherwise. **Session opened:** if capture idle, `POST /api/capture/start`. **Cart `paid`/`completed` or session closed:** stop capture; a final sync may 409 — treat as done.
   - While bound: every `logsPollMs`, fetch `/api/logs`, run `deriveCartState`, and `POST /api/pos/sync` **only when the derived state changed** (plus a 15 s heartbeat).
   - Failure policy: webapp unreachable → retry with backoff, keep last state, surface `pos_sync_error` to the renderer; capture died (sidecar `error` status) → attempt **one** auto-restart, else keep bound and show the warning (cart keeps its items; correction happens on the tablet).
   - Emits `pos:state` to the renderer via webContents (pattern: `sidecar:health`).
4. **Preload** (`desktop/src/preload/index.ts` + `index.d.ts`): `pos:get-state` (invoke), `pos:get-config`/`pos:save-config` (invoke), `pos:on-state` (event listener, mirroring `onSidecarHealth`). Expose nothing else.
5. **Renderer:**
   - Live view: small **POS panel** in the stats/side area — bound/unbound, cart code, item count synced, last sync age, sync errors. Read-only readout.
   - Admin Panel: **POS integration** section — base URL, secret, station id (display + regenerate), dwell/settle/poll inputs, "Test connection" button hitting `GET /api/pos/session` once and reporting (including the http/https warning).
6. **Docs** `docs/POS_INTEGRATION.md`: setup (webapp URL, secret, station pairing, LAN-vs-cloud deployment note from §2.1), the operator-less flow, troubleshooting (401 secret mismatch · 409 session closed/cart paid · `unmapped` items need admin mapping · sync error = webapp unreachable). JSON fences only — no shell commands in fenced blocks — so the repo's doc-fence test readers stay green.

## 6. Edge cases the implementer must handle (test-backed)

- **Flicker:** a track dropping for < dwell never commits; a committed track dropping < settle never removes. (Unit tests in `cartState.test.ts`.)
- **Camera dies mid-session:** capture auto-restart once; cart keeps last synced items; tablet still shows them; no phantom removals (reconcile only fires on successful derivation).
- **Webapp dies mid-session:** backoff retries; on reconnect the next snapshot converges the cart (A4) — no duplicate rows possible.
- **Two tablets / double Start:** partial unique index rejects the second open session per station; tablet shows "counter busy".
- **Finish while camera still sees items:** Finish wins — session → `completed`, cart → `paid`; the next desktop sync gets 409 and the desktop stops capture. Leftover rows stay as ordered.
- **Manual rows:** `session_ref IS NULL` rows are invisible to reconcile (A5) — a staff add is never deleted by the camera.
- **Unmapped class:** reported per-sync, visible in the admin panel, never inserted; tablet shows a subtle "N items not recognized" note.
- **Cloud-hosted webapp:** desktop polls over WAN — keep payloads snapshot-small; no behavior change otherwise.

## 7. Testing & acceptance

- **pushcart-web** (no test runner configured): clean `yarn build` + `yarn lint`; `scripts/pos-e2e.sh` (curl) walking the happy path: `/api/auth` anonymous sign-in (captures cookie) → create station session → token sync snapshot ×2 (adds, then no-op) → snapshot with an item removed (verifies delete) → finish → verify `orders` row + `stock_quantity` decrement + cart `paid` + sync now 409. Also: wrong token 401, second Start 409, unmapped class reported.
- **SCANnCART**: vitest — `cartState.test.ts` (dwell/settle/counting: the critical pure logic), `posSession.test.ts` (fake timers: bind→start capture→sync-on-change→paid→stop; webapp-down backoff; capture-death restart-once), `posConfig.test.ts` (temp dir), Admin Panel POS-section tests. `npm run typecheck` + `npm test` + `npm run lint` + `make docs-sync-check` green.
- **Joint smoke** (§8 step 3): tablet Start → items dropped on counter (fake frame source in dev) appear on tablet with totals → item taken back disappears after settle → Finish → order exists, stock decremented, SCANnCART stops capture on its own.

## 8. Build order (two-agent split, for whoever implements this)

1. **Agent W (pushcart-web):** §4.1 migration → §4.2 model+routes → §4.4 tablet pages → §4.5 admin screen → e2e script (§7).
2. **Agent S (SCANnCART):** §5.1 config → §5.2 cartState (pure logic + tests first) → §5.3 orchestrator → §5.4 IPC → §5.5 UI → docs.
3. §7 joint smoke last. The halves parallelize against §3's contracts; only the smoke needs both.

## 9. Parked (explicitly out of scope)
Payment integration (GCash/Maya) · QR-to-camera binding · idle auto-timeout sessions · multi-counter orchestration beyond `station_id` · weight-sensor cross-check (PRD §future) · retry/backoff tuning beyond §5.3's simple policy.
