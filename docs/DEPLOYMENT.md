# 🚀 SCANnCART – Deployment & Future Phases

> Everything here is **out of scope for the current prototype** ([PRD.md](./PRD.md)).
> This document captures the deployment vision so it isn't lost, and is picked up only after the local prototype is validated.

## 1. Deployment Goals
- Detect and classify items **fast enough to keep checkout queues moving**.
- Support both **gentle placement** and **tossed items** without losing accuracy.
- **Centralize inference** to reduce hardware cost per cart.
- Enable **scalable deployment** across multiple carts in a store.

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

## 3. Future Phases (Out-of-Scope for Prototype)
- Weight sensors and ESP32 integration.
- Cloud synchronization and analytics dashboard.
- Mobile app control interface (monitor-only prototype may be tested first).
- Embedded deployment (Jetson Nano / Pi 5).
- **Docker / Docker Compose** containerization for portable, modular deployment.

---

## 4. Deployment Non-Functional Targets
- **Performance:** Round-trip inference latency < 300 ms (centralized).
- **Scalability:** Support many carts against one shared inference server.
- **Portability:** Dockerized services for cross-platform deployment.

## 5. Deployment Success Metrics
- Round-trip inference < 300 ms (centralized).
- Successful deployment in a grocery store pilot.
- Stable multi-cart operation against a single server.

---
