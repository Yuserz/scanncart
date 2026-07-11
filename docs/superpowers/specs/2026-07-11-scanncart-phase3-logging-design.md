# SCANnCART Phase 3 — Logging — Design

**Date:** 2026-07-11
**Status:** Approved for planning
**Parent spec:** [`2026-07-11-scanncart-prototype-design.md`](./2026-07-11-scanncart-prototype-design.md) (§5 data model, §10 phase 3)

This is a **delta** on the approved master spec. It records the Phase 3-specific
decisions made in brainstorming; everything not restated here follows the master
spec unchanged.

---

## 1. Scope

Phase 3 = **"sessions + dedup events + item-log panel"** (master spec §10.3).

**In scope**

- Persist capture **sessions** and deduplicated **detection_events** to SQLite
  (schema per master spec §5).
- A **minimal** `GET /api/logs` returning the current (most-recent) session's
  events — just enough for the UI item log to reconcile.
- Desktop: reconcile the in-memory item log against `/api/logs` so it survives a
  WebSocket reconnect.

**Deferred to Phase 4 (Admin)**

- Full `GET /api/logs` filtering/paging (`from`/`to`/`cls`/`limit`/`offset`),
  `DELETE /api/logs`, CSV export.
- `app_settings` table + settings persistence.
- The admin Logs tab UI.

---

## 2. Sidecar

### 2.1 `logging_store.py` (new — sole DB writer)

Owns `sidecar/data/scanncart.db`. Creates the `sessions` and `detection_events`
tables (master spec §5) on first open. **DB path is injectable** so tests use a
temp file.

Concurrency: `capture/start` (async request thread) opens a session while the
**pipeline thread** writes events. Use a single connection opened
`check_same_thread=False`, with all writes/reads serialized behind a
`threading.Lock`.

Public API:

| Function | Behavior |
|---|---|
| `start_session(model_name, device) -> session_id` | Insert a `sessions` row (`started_at=now`, `ended_at=NULL`). |
| `end_session(session_id)` | Set `ended_at=now`. |
| `record_detection(session_id, track_id, cls, conf, ts)` | **Insert-or-update.** First sighting inserts a row (`confidence=conf`, `entered_at=ts`, `max_conf=conf`). Later sightings bump `max_conf=max(max_conf, conf)`; `confidence` and `entered_at` stay frozen at first sighting. |
| `resolve_left(session_id, track_id, ts)` | Set `left_at=ts` for that event. |
| `query_events(session_id) -> list[EventRow]` | Ordered by `entered_at`; for the read endpoint. |
| `current_session_id() -> int \| None` | Most-recent session (for the default `/api/logs`). |

Dedup key is `(session_id, track_id)`.

### 2.2 Dedup wired into the pipeline

The `Pipeline` receives an optional logging hook (`logging_store` + active
`session_id`) so persistence runs per processed frame without entangling the
inference path. It maintains an in-memory `dict[track_id -> last_seen_ts]`:

- **New track_id** → `record_detection` (insert) + add to map.
- **Seen track_id** → `record_detection` (update `max_conf`) + refresh last-seen.
- **Expiry sweep** each frame: any track whose last-seen is older than
  **`track_expiry_s` (default 1.5s, in `settings.py`)** → `resolve_left` + drop
  from the map. Time-based (not frame-count) so it is robust to variable CPU
  inference FPS and brief occlusions.
- Detections with `track_id is None` are streamed but **not logged** (no stable
  identity to dedup on).

`main.py`:

- `POST /api/capture/start` → `session_id = start_session(active_model, device)`;
  pass it to the `Pipeline`.
- `POST /api/capture/stop` → resolve `left_at` for all still-open tracks, then
  `end_session(session_id)`.

### 2.3 `GET /api/logs` (minimal)

Returns the current session's events (no filters/paging). New Pydantic models in
`schemas.py`:

```jsonc
// LogsResponse
{
  "session_id": 7,
  "events": [
    { "track_id": 12, "class_name": "banana", "confidence": 0.83,
      "max_conf": 0.91, "entered_at": 1720598400.1, "left_at": 1720598403.4 }
  ]
}
```

If no session exists yet, return `{ "session_id": null, "events": [] }`.

---

## 3. Desktop

- **REST client (`lib/api.ts`):** add `getLogs(): Promise<LogsResponse>` hitting
  `GET /api/logs`, with matching TS types (`LogEvent`, `LogsResponse`).
- **`useSidecarStream`:** on WS `onOpen`, `await getLogs()` and seed both `items`
  and the `seenRef` dedup set from the returned rows (mapping `class_name`→`cls`,
  `entered_at`→`ts`). Then the live WS stream keeps appending new `track_id`s as
  today, skipping any already seeded. This restores the log after a reconnect
  instead of resetting it.
- **`LiveView.tsx`:** unchanged — still renders `items`.

Reconcile order note: seed from `/api/logs` **before** processing live frames so a
track already in the DB is not double-counted.

---

## 4. Testing

**Sidecar (pytest)**

- `logging_store`: schema init; `start/end_session`; `record_detection`
  insert-then-update (one row per `(session, track_id)`; `max_conf` keeps the
  best; `confidence`/`entered_at` frozen); `resolve_left`; `query_events` order.
- Pipeline dedup: fake frame source + scripted detections → new tracks insert,
  repeats update, a track that stops appearing past `track_expiry_s` gets
  `left_at`, and `capture/stop` resolves the rest.
- `GET /api/logs` via FastAPI `TestClient` (empty → `session_id: null`; after a
  run → expected rows).

**Desktop (vitest + RTL)**

- `getLogs()` REST client (success + non-OK error).
- `useSidecarStream`: on open, seeds `items` from a fake `/api/logs`; a
  subsequent live frame with an already-seeded `track_id` does **not** duplicate;
  a new `track_id` appends.

---

## 5. Non-goals / caveats

- SQLite is the **sole writer** from the sidecar; the DB is not accessed from
  Electron/renderer (UI reads only via REST).
- Track IDs are unique per session only; the `session_id` scoping prevents
  cross-run collisions (master spec §5).
- No migration framework — schema is created idempotently on open; acceptable for
  a single-file prototype DB.
