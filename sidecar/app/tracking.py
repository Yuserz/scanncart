"""Local object tracking for detector backends that don't supply track ids.

`YoloDetector` gets stable `track_id`s from Ultralytics' `.track(persist=True)`.
Remote backends only do when the Roboflow Workflow itself contains a tracking
block, so this fills the gap: a small greedy IoU tracker with no extra
dependency.

Deliberately simple. Checkout items are placed and then sit still, so
frame-to-frame boxes overlap heavily and greedy IoU matching is sufficient. A
motion-model tracker (ByteTrack et al.) is built for fast, occluding, crossing
motion that does not happen on a counter, and costs a heavy dependency tree for
the privilege.

Matching is class-aware: a track never migrates between classes. That matters
for the visually-similar-SKU trap in MODEL_TRAINING.md §4 — without it, a model
flickering between two flavours of the same product would keep one track id and
silently relabel it mid-session.
"""

import time
from dataclasses import dataclass
from typing import Callable

from app.schemas import Detection

Box = tuple[float, float, float, float]

DEFAULT_IOU_THRESHOLD = 0.3


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two xyxy boxes. Degenerate boxes score 0.0."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class _Track:
    track_id: int
    cls: str
    box: Box
    last_seen: float


class IouTracker:
    """Assigns stable `track_id`s to detections across calls.

    `expiry_s` should match the pipeline's `track_expiry_s`. If the tracker
    forgets a track sooner than the pipeline does, the next sighting gets a
    fresh id and the item logs twice; if it forgets later, a returning item is
    silently merged into a track the pipeline already resolved as "left".
    """

    def __init__(
        self,
        expiry_s: float = 1.5,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._expiry_s = expiry_s
        self._iou_threshold = iou_threshold
        self._clock = clock
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def assign(self, detections: list[Detection], now: float | None = None) -> list[Detection]:
        """Return copies of `detections` with `track_id` populated."""
        if now is None:
            now = self._clock()
        self._expire(now)

        # Score every same-class detection/track pair above threshold, then take
        # them highest-IoU-first so the most confident pairing wins the contest
        # for a shared track. Ties break on index to stay deterministic.
        candidates = [
            (score, det_idx, track_id)
            for det_idx, det in enumerate(detections)
            for track_id, track in self._tracks.items()
            if track.cls == det.cls
            and (score := iou(det.box, track.box)) >= self._iou_threshold
        ]
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        matched: dict[int, int] = {}
        claimed_tracks: set[int] = set()
        for _score, det_idx, track_id in candidates:
            if det_idx in matched or track_id in claimed_tracks:
                continue
            matched[det_idx] = track_id
            claimed_tracks.add(track_id)

        out: list[Detection] = []
        for det_idx, det in enumerate(detections):
            track_id = matched.get(det_idx)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, det.cls, det.box, now)
            else:
                track = self._tracks[track_id]
                track.box = det.box
                track.last_seen = now
            out.append(det.model_copy(update={"track_id": track_id}))
        return out

    def reset(self) -> None:
        """Forget every track and restart ids. Call between capture sessions."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def _expire(self, now: float) -> None:
        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > self._expiry_s]
        for tid in stale:
            del self._tracks[tid]
