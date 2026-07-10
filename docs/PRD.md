# 📘 SCANnCART YOLO11 – Vision‑Only Prototype PRD

> Scope of this document: the **local prototype** we are building now.
> Future/centralized deployment, edge hardware, and containerization live in [DEPLOYMENT.md](./DEPLOYMENT.md).

## 1. Overview
SCANnCART is a **capstone system** designed for grocery stores.
The prototype uses **YOLO11** and a **Logitech StreamCam (1080p @ 60fps)** to detect and classify grocery items in real time on a single PC.

**Context (prototype):**
- The cart acts as a **smart scanner**, detecting items as they are placed or tossed in.
- All capture, inference, UI, and logging run **locally on one PC** — no server, no network.

---

## 2. Objectives
- ✅ Detect and classify grocery items quickly and accurately.
- ✅ Handle varied item placement (gentle or tossed).
- ✅ Provide real‑time visual feedback via the UI.
- ✅ Log every detection locally for review.

---

## 3. Scope
### In‑Scope (Now)
- Camera input via Logitech StreamCam (USB, 1080p @ 60fps).
- YOLO11 inference on PC via Ultralytics.
- Python sidecar service that owns the camera + inference.
- UI visualization (Electron + React).
- Local database logging (SQLite).

### Out‑of‑Scope (see DEPLOYMENT.md)
- Centralized server / “brain” inference.
- Edge hardware (ESP32-CAM, Pi Zero, Jetson, mini PC).
- Weight sensors and ESP32 integration.
- Cloud sync and analytics dashboard.
- Mobile app control interface.
- Docker containerization.

---

## 4. Tech Stack (Recommended)

| Layer | Choice | Notes |
|-------|--------|-------|
| **Camera** | Logitech StreamCam (USB) | 1080p @ 60fps; captured in Python via OpenCV. |
| **Model** | YOLO11 (Ultralytics, PyTorch) | Object detection + classification. |
| **Inference service** | Python sidecar | Owns the camera and runs YOLO11; spawned by Electron. |
| **UI ↔ sidecar transport** | localhost **WebSocket** | Pushes detections / annotated frames in real time. |
| **UI** | Electron + React | Live feed, bounding boxes, item log, start/stop. |
| **Database** | SQLite | Local detection log. |

**Why this stack:**
- Ultralytics YOLO11 is Python-only, so a **Python sidecar** keeps the heavy work (camera + inference) out of the Electron/Node process.
- A **WebSocket** (push) fits continuous ~30fps streaming far better than REST polling — lower latency, no request-per-frame overhead.
- **SQLite** (over TinyDB) is a single-file, zero-config, well-supported store that scales cleanly if the schema grows.
- **Electron + React** (over PyQt6) gives a familiar web UI toolchain and matches the future dashboard, so UI work carries forward.

### 4.1 Architecture
```
Logitech StreamCam
        │  (USB)
        ▼
Python sidecar  ──►  OpenCV capture  ──►  YOLO11 (Ultralytics)
        │                                        │
        │  localhost WebSocket (detections + annotated frames)
        ▼                                        ▼
Electron + React UI  ◄──────────────────  SQLite (detection log)
```
Electron spawns the Python sidecar on startup and connects to it over a localhost WebSocket. The sidecar captures frames, runs inference, writes detections to SQLite, and streams results to the UI.

---

## 5. Functional Requirements
- **Real‑Time Detection:** ≥ 30 fps processing with YOLO11.
- **UI Feedback:** Bounding boxes, item names, confidence scores.
- **Logging:** Timestamp, item ID, confidence stored in SQLite.
- **Error Handling:** Graceful fallback for camera disconnects and sidecar restarts.
- **Controls:** Start/stop capture and an item-list view in the UI.

---

## 6. Non-Functional Requirements
- **Performance:** End-to-end latency < 150 ms (local).
- **Reliability:** Continuous operation for ≥ 2 hours without restart.
- **Usability:** Simple UI with start/stop and item-list view.
- **Maintainability:** Modular codebase with clear separation of concerns.

---

## 7. Success Metrics
- Stable detection at ≥ 30 fps.
- End-to-end latency < 150 ms (local).
- ≥ 90% accuracy on common grocery items.
- Smooth UI performance with minimal latency.

---
