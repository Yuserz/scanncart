# SCANnCART — Product Requirements (prototype)

> Scope: the **prototype as built for the capstone defense**, updated 2026-10-05.
> Future work (edge hardware, cloud, more SKUs) lives in [DEPLOYMENT.md](./DEPLOYMENT.md);
> how each requirement stands today, with its evidence, is [DEFENSE_READINESS.md](./DEFENSE_READINESS.md).

## 1. Overview

SCANnCART is a **camera on a shopping cart** that recognises grocery products as the customer puts
them in or takes them out, and keeps the cart's bill on the store's self-checkout tablet in step —
no barcode scanning, no cashier.

- A **Logitech StreamCam** rides on the cart by the handle, looking over the basket.
- A **Python sidecar** on one PC runs **YOLO11** (Ultralytics) detection + BoT-SORT tracking on
  every camera frame.
- An **Electron + React desktop app** shows the live picture, decides which movements are deposits
  and removals, and keeps the basket's ledger.
- **pushcart-web** (a separate Next.js + Supabase app) owns the cart rows, prices, stock and orders,
  and runs the customer's tablet.

Detection is local: the camera, the model and the ledger never need the internet. The only network
hop is desktop → pushcart-web, on the shop LAN.

**What changed from the first PRD (2026-07):** the "counter scanner on one PC, no network" became a
cart with camera-only add/remove, a basket ledger and a self-checkout integration
([POS_INTEGRATION_SPEC.md](./POS_INTEGRATION_SPEC.md), [CART_TRANSFER_SPEC.md](./CART_TRANSFER_SPEC.md)).
The detector, the sidecar/desktop split, SQLite logging and the local-only detection promise are
unchanged.

## 2. Objectives

1. Recognise the store's products in real time from one camera.
2. Turn a product going **into** the basket into +1 on the bill, and **out of** it into −1, with no
   manual editing by the customer.
3. Never guess: an ambiguous movement holds the bill unchanged and asks staff to check.
4. Let the customer start, watch the total, and finish on a tablet.
5. Log every tracked item locally for review.

## 3. Scope

### In scope (built)

| Area | What is in |
| --- | --- |
| Camera | Logitech StreamCam, USB, 1280×720 at 60 fps (MJPG, Media Foundation), app-side auto exposure |
| Model | YOLO11s fine-tuned on 7 grocery SKUs: `scanncart-grocery-v2` (running; 96.7% mAP50 on the held-out test split), `scanncart-grocery-v1` kept installed |
| Detection | Ultralytics `track()` (BoT-SORT), confidence cutoff, three shape filters for phantom boxes |
| Cart logic | Basket ledger driven by a deposit/removal state machine over three zones (outside / opening / inside) |
| Customer UI | pushcart-web tablet pages: Start shopping, live read-only cart and total, Finish |
| Staff UI | Desktop: Live view, Admin Panel, Basket test (zones + practice), Camera tuning; tablet: staff-PIN removal |
| Data | SQLite track log on the PC; carts, orders and stock in pushcart-web's Supabase |

### Out of scope (see DEPLOYMENT.md)

Payment capture (Finish creates the order and deducts stock, no card is charged) · edge hardware on
the cart · weight sensors · cloud sync and analytics · products beyond the 7 trained · more than one
station per tablet.

## 4. Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Camera | Logitech StreamCam via OpenCV (MSMF) | 60 fps at 720p in MJPG; DirectShow caps at ~15 fps |
| Model | YOLO11s, Ultralytics/PyTorch, CUDA | Best accuracy/latency on an RTX 4060 for this SKU set |
| Inference service | Python sidecar (FastAPI), spawned by Electron | Ultralytics is Python-only; keeps CV work out of Node |
| Desktop ↔ sidecar | localhost WebSocket (frames) + REST (control) | Push for continuous frames, pull for occasional calls |
| Desktop | Electron + React + TypeScript | Web UI toolchain; main process hosts the basket logic |
| Desktop ↔ pushcart-web | HTTPS/HTTP, `x-pos-token` shared secret, full-snapshot sync | Idempotent: a repeated snapshot never duplicates an item |
| Checkout | pushcart-web (Next.js + Supabase/Postgres) | Owns money-relevant state; database policies lock the cart to the camera |
| Local log | SQLite | Single file, zero config |

## 5. Functional requirements

| ID | Requirement |
| --- | --- |
| F1 | Detect and track the 7 SKUs at ≥ 30 fps (frames analysed per second). |
| F2 | Show bounding boxes, product names and confidence on the live view (and on the Basket test screen). |
| F3 | Add one unit on a confirmed deposit and remove one on a confirmed removal; hidden items stay billed. |
| F4 | Send ambiguous movements to review (bill unchanged) and block Finish until staff clear it. |
| F5 | Bind to the tablet's session automatically on *Start shopping*; unbind on Finish. |
| F6 | Sync the whole cart to pushcart-web as soon as it changes, plus a heartbeat; recover from outages without duplicates. |
| F7 | Log every track (product, best confidence, entered/left) in SQLite. |
| F8 | Survive a camera disconnect and a sidecar stop with a clear on-screen explanation. |
| F9 | Start/stop capture; tune camera and detection settings live. |

## 6. Non-functional requirements

| ID | Requirement |
| --- | --- |
| N1 | Camera-to-screen latency under 150 ms. |
| N2 | A confirmed transfer reaches the tablet within 5 s at the 95th percentile (spec Gate C). |
| N3 | Run continuously for ≥ 2 hours without a restart. |
| N4 | Detection works with no internet connection. |
| N5 | Modular, tested code: every component testable against fakes, no camera/GPU/network in CI. |

## 7. Success metrics

| Metric | Target |
| --- | --- |
| Analysed frames per second | ≥ 30 fps |
| Camera-to-screen latency | < 150 ms |
| Per-class recall on held-out images | ≥ 0.85 for every SKU (project floor) |
| Overall detection accuracy | ≥ 90% (mAP50) |
| Deposit / removal correctness (Gate C) | ≥ 34/35 per direction; zero ledger changes on 30 no-transfer trials |
| Continuous operation | ≥ 2 hours |
