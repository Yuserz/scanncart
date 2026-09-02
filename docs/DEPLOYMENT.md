# 🚀 SCANnCART – Deployment & Future Phases

> Everything here is **out of scope for the current prototype** ([PRD.md](./PRD.md)).
> This document captures the deployment vision so it isn't lost, and is picked up only after the local prototype is validated.

## 1. Deployment Goals
- Detect and classify items **fast enough to keep checkout queues moving**.
- Support both **gentle placement** and **tossed items** without losing accuracy.
- **Centralize inference** to reduce hardware cost per cart.
- Enable **scalable deployment** across multiple carts in a store.
- Deploy onto a store's **existing cart fleet** rather than requiring it to buy new carts.

---

## 2. Centralized Architecture (Cart → Server)

| Component | Role | Technology |
|-----------|------|------------|
| **Central Server (“Brain”)** | Runs YOLO11 inference and manages API endpoints | Python + FastAPI + Ultralytics YOLO11 + GPU |
| **Cart Unit (“Edge Node”)** | Captures video and sends frames to server | ESP32-CAM / Pi Zero 2 W / Mini PC |
| **Network Layer** | Wireless communication within the store | Wi-Fi 6 mesh / LAN |
| **Data Flow** | Camera → Edge Node → API → Server → Response → UI | REST / WebSocket |
| **UI & Dashboard** | Displays detections and analytics | Electron / React |

```
[Cart: camera + edge node] ──Wi-Fi──► [Central server: YOLO11 + GPU] ──► [UI / dashboard]
        (many carts)                          (shared brain)
```

---

## 3. Retrofit Deployment (Cart Hardware Model)

SCANnCART is deployed as a **retrofit kit** — an add-on module mounted onto the store's
existing pushcarts. This is the deliberate business choice, and it is the standard industry
term for this model; use it consistently in the paper, pitch deck, and any store-facing material.

### 3.1 The two industry models

| Model | What the store buys | Examples |
|-------|--------------------|----------|
| **Purpose-built smart cart** | An entire new cart; the store replaces its fleet | Amazon Dash Cart, Instacart Caper Cart |
| **Retrofit / clip-on kit** ✅ | A device that mounts onto carts the store already owns | Shopic — and **SCANnCART** |

### 3.2 Vocabulary

| Term | Emphasis | Use it when |
|------|----------|-------------|
| **Retrofit kit** | Upgrading equipment the store already owns | Default term — headline, abstract, pitch |
| **Smart cart retrofit module** | The unit is one component of a larger system | Architecture and BOM sections |
| **Clip-on / bolt-on device** | The physical attachment method | Describing installation and mounting |
| **Cart-mounted unit / edge node** | Its role in the software architecture | Diagrams, API and network discussion |
| **Aftermarket upgrade** | Sold separately from the cart itself | Cost and procurement discussion |

Avoid "smart cart" on its own for SCANnCART — it implies the purpose-built model and
undersells the retrofit advantage.

### 3.3 Why retrofit, for a grocery business

- **Low switching cost.** No fleet replacement; the store keeps the carts it already paid for.
- **Incremental rollout.** Equip a handful of carts, measure, then scale — no all-or-nothing purchase.
- **Reversible.** A kit can be unmounted and moved to another cart, or removed entirely after a pilot.
- **Maintenance is per-module.** A failed unit is swapped out; the cart stays in service.
- **Fits the centralized architecture** (§2) — the kit stays cheap because inference lives on the shared server.

### 3.4 What ships in a kit

| Item | Purpose |
|------|---------|
| Camera (StreamCam-class or embedded module) | Captures the cart basket |
| Edge node (ESP32-CAM / Pi Zero 2 W / mini PC) | Frame capture and transmit |
| Mount / bracket | Clamps to the cart frame or basket rim; fits standard cart tubing |
| Power (battery pack + charging dock) | Untethered operation for a full shopping trip |
| Optional display / tablet | Shows the running item log to the shopper |
| Optional weight sensor | Cross-checks detections (see §4) |

### 3.5 Retrofit-specific open questions

These are unresolved and must be answered before a store pilot:

- **Cart variation.** Basket geometry and tube diameter differ per store/supplier; the bracket
  needs to tolerate that range or ship in variants.
- **Camera placement.** Field of view must cover the basket opening without blocking loading.
- **Battery life.** A kit must survive a full trading day (or a shift) between charges, plus a
  practical way to recharge a fleet.
- **Theft and tamper resistance.** Kits are on carts that leave the store floor.
- **Environmental.** Vibration, knocks, water, and cleaning routines.
- **Installation time per cart.** Determines the labour cost of a rollout.

---

## 4. Future Phases (Out-of-Scope for Prototype)
- Weight sensors and ESP32 integration.
- Cloud synchronization and analytics dashboard.
- Mobile app control interface (monitor-only prototype may be tested first).
- Embedded deployment (Jetson Nano / Pi 5).
- **Docker / Docker Compose** containerization for portable, modular deployment.
- Retrofit kit industrial design: bracket variants, enclosure, and battery/charging dock.

---

## 5. Deployment Non-Functional Targets
- **Performance:** Round-trip inference latency < 300 ms (centralized).
- **Scalability:** Support many carts against one shared inference server.
- **Portability:** Dockerized services for cross-platform deployment.
- **Retrofit fit:** One bracket design mounts to the majority of a pilot store's existing carts.

## 6. Deployment Success Metrics
- Round-trip inference < 300 ms (centralized).
- Successful deployment in a grocery store pilot.
- Stable multi-cart operation against a single server.
- Kit installed on an unmodified store cart with no tools beyond a hand wrench, within a target install time.

---
