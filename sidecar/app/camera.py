import math
import threading
import time
from collections import deque
from typing import Protocol
import cv2
import numpy as np


def _undo_for(name: str) -> float | bool | None:
    """What to write to hand one control back, or None if nothing can be.

    Autofocus is the whole list, because it is the only one of the four whose
    automatic mode the app can *name* — `CAP_PROP_AUTOFOCUS=1` is a mode rather
    than a value, so the app never has to know what the lens was doing.
    Measured on this hardware, its getter answers 0 while the lens is in that
    mode, so reading it back would pin focus on the way to undoing the drag.

    The other three have no such mode to ask for. Brightness has no automatic
    mode in OpenCV at all, and this repo has never measured the constants
    `CAP_PROP_AUTO_EXPOSURE` takes on this backend — so the only undo left would
    be the value the device held before the app wrote it, and this device will
    not say what that was. Asking the *streaming* handle is what was tried
    first, and it is worth not repeating: measured, `get(CAP_PROP_BRIGHTNESS)`
    answered 0.0 while the picture was a normal mid-grey, and writing that
    "restored" value back turned the picture black. A value the app cannot know
    is not one it gets to write, so a reset of those three stops this app
    writing the control and leaves the device holding the last value it was
    given — which is also where a capture restart leaves it, since reopening
    does not clear a control this app wrote (measured).
    """
    return True if name == "autofocus" else None


def _with_restores(changes: dict) -> dict:
    """Resolve each queued change into the write it means, and drop the rest.

    `None` in a queued change means the setting went back to "this app imposes
    nothing". That used to be left as a skip, which reads as harmless and is
    not: the device is still holding whatever the app last wrote, so the field
    said "auto", the settings said null, and the picture stayed exactly where
    the drag put it. Autofocus is handed back to the lens (`_undo_for`); the
    other three resolve to no write at all.
    """
    writes: dict = {}
    for name, value in changes.items():
        undo = value if value is not None else _undo_for(name)
        if undo is not None:
            writes[name] = undo
    return writes


def exposure_cap_for(fps: float) -> int:
    """The longest exposure (log2 seconds) that still lets the camera deliver `fps`.

    A shutter of 2^e seconds caps delivery at 1/2^e frames a second, so -6 is the longest
    that keeps 60 fps (measured: -5 held this StreamCam near 30). Same arithmetic as
    `camera_caps.exposure_ceiling`, kept here because that module imports this one's peers
    and the capture thread needs the number without importing the calibration module.
    """
    if fps <= 0:
        return 0
    return int(max(-13, min(0, math.floor(-math.log2(fps)))))


class AutoExposure:
    """Software auto-exposure that never costs framerate.

    The StreamCam's own automatic exposure lengthens the shutter in indoor light: measured at
    1280x720, it settled at 12 fps with a washed-out picture (mean luminance 206). So the shutter
    stays at the framerate's cap and **brightness** does the adjusting — at -6 it moved mean
    luminance from 19 (brightness 64) to 152 (255) with capture holding 60 fps. Only a scene too
    bright for the lowest useful brightness shortens the shutter, one stop at a time, and the
    shutter is never lengthened past the cap.

    Pure: it is handed a luminance and returns the writes to make, so it is tested without a
    device. The capture thread owns the device and does the writing.
    """

    TARGET = 130.0  # camera_quality.BRIGHTNESS_TARGET, the level calibration aims for
    DEADBAND = 12.0
    # Luminance moved ~0.5-0.8 per brightness unit on this camera; half of the inverse damps the
    # step so the loop settles instead of ringing.
    GAIN = 0.8
    MAX_STEP = 32.0
    BRIGHTNESS_MIN = 40.0  # below this the picture crushes; shorten the shutter instead
    BRIGHTNESS_MAX = 255.0
    INTERVAL_S = 0.2
    # A shutter change takes a few frames to show; judging sooner would step twice.
    SETTLE_S = 0.6

    def __init__(self, fps: float, brightness: float | None, exposure: float | None) -> None:
        self.cap = exposure_cap_for(fps)
        self.brightness = 128.0 if brightness is None else float(brightness)
        self.exposure = float(self.cap if exposure is None else min(exposure, self.cap))
        self._next_t = 0.0

    def start_writes(self) -> dict:
        return {"exposure": self.exposure, "brightness": self.brightness}

    def update(self, luminance: float, now: float) -> dict:
        """The control writes this reading calls for (empty when nothing should change)."""
        if now < self._next_t:
            return {}
        self._next_t = now + self.INTERVAL_S
        err = self.TARGET - luminance
        if abs(err) <= self.DEADBAND:
            return {}
        step = max(-self.MAX_STEP, min(self.MAX_STEP, err * self.GAIN))
        wanted = self.brightness + step
        if err < 0 and wanted < self.BRIGHTNESS_MIN and self.exposure > -13:
            # Too bright even near the floor: one stop shorter halves the light.
            self.exposure -= 1
            self.brightness = min(self.BRIGHTNESS_MAX, self.brightness * 1.5)
            self._next_t = now + self.SETTLE_S
            return {"exposure": self.exposure, "brightness": self.brightness}
        if err > 0 and wanted > self.BRIGHTNESS_MAX and self.exposure < self.cap:
            # Too dark at full brightness, and a stop of shutter is still free.
            self.exposure += 1
            self.brightness = max(self.BRIGHTNESS_MIN, self.brightness / 1.5)
            self._next_t = now + self.SETTLE_S
            return {"exposure": self.exposure, "brightness": self.brightness}
        clamped = max(self.BRIGHTNESS_MIN, min(self.BRIGHTNESS_MAX, wanted))
        if clamped == self.brightness:
            return {}
        self.brightness = clamped
        return {"brightness": self.brightness}


def mean_luminance(frame: np.ndarray) -> float:
    """Mean brightness of a frame, sampled sparsely: the loop needs a level, not detail."""
    sample = frame[::8, ::8]
    if sample.ndim == 3:
        # BGR → luma weights, without a colour conversion of the whole frame.
        b, g, r = sample[..., 0], sample[..., 1], sample[..., 2]
        return float((0.114 * b + 0.587 * g + 0.299 * r).mean())
    return float(sample.mean())


class LatestFrameBuffer:
    """Thread-safe size-1 buffer where the newest frame always wins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: tuple[int, np.ndarray] | None = None

    def put(self, seq: int, frame: np.ndarray) -> None:
        with self._lock:
            self._item = (seq, frame)

    def get(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            return self._item


def _default_capture(index):
    # Pin the Media Foundation backend on Windows: it delivers the StreamCam's
    # full 60 fps at 1080p, whereas OpenCV's DirectShow path caps around 15 fps
    # for the same mode. Fall back to OpenCV's auto backend if MSMF can't open
    # the device (e.g. a non-Windows host or a camera with no MSMF driver).
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


class FrameSource(Protocol):
    width: int
    height: int
    fps: float

    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def release(self) -> None: ...


class FakeFrameSource:
    """Test double that yields the provided frames in order, then None."""

    def __init__(self, frames: list[np.ndarray], fps: float = 30.0) -> None:
        self._frames = frames
        self._i = 0
        h, w = (frames[0].shape[0], frames[0].shape[1]) if frames else (0, 0)
        self.width = w
        self.height = h
        self.fps = fps

    def open(self) -> None:
        self._i = 0

    def read(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def release(self) -> None:
        pass


class CameraCapture:
    """Owns an OpenCV device and runs a background capture thread."""

    def __init__(
        self, index, width, height, fps, cap_factory=_default_capture,
        brightness: float | None = None, exposure: float | None = None,
        autofocus: bool | None = None, focus: float | None = None,
        auto_exposure: bool = False,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = float(fps)
        self._cap_factory = cap_factory
        # None means "this app imposes no value" — see Settings.camera_brightness
        # et al. At open that writes nothing; live it means hand the control
        # back, which takes an actual write — see `_with_restores`. The
        # StreamCam's automatic focus/exposure track faces, which a checkout
        # counter never has, so locked manual values suit this app.
        self._brightness = brightness
        self._exposure = exposure
        self._autofocus = autofocus
        self._focus = focus
        # While on, `AutoExposure` owns brightness and exposure: the manual values are only its
        # starting point, and come back into force when it is switched off.
        self._auto_exposure = auto_exposure
        self._ae: AutoExposure | None = None
        # Control changes queued by another thread, drained by _loop between
        # reads. cv2.VideoCapture is not thread-safe, so set() must never be
        # called from the FastAPI request thread while read() is in flight.
        self._controls_lock = threading.Lock()
        self._pending_controls: dict[str, float | bool | None] = {}
        # Set when a live control write is refused by the device. Not a
        # capture failure — the stream keeps running — so it stays separate
        # from `failure`.
        self.control_error: str | None = None
        self._cap = None
        self._buffer = LatestFrameBuffer()
        self._thread = None
        self._running = False
        self._seq = 0
        self.is_open = False
        # A device can be invalidated while open — unplugged, taken by another
        # process, or suspended by USB power management. read() then fails
        # instantly and forever. Retrying flat out burned a core and wrote
        # ~23,000 OpenCV warnings to stderr in one session while the app sat
        # there looking like it was running.
        self.failure: str | None = None
        self._consecutive_failures = 0
        self._failing_since: float | None = None
        # Timestamps of recent successful reads, for the measured rate. The
        # requested fps is a request; this is what arrived.
        self._read_times: deque[float] = deque(maxlen=120)

    def _current_controls(self) -> dict:
        return {
            "autofocus": self._autofocus,
            "focus": self._focus,
            "brightness": self._brightness,
            "exposure": self._exposure,
        }

    @staticmethod
    def _write_controls(cap, controls: dict) -> None:
        """Write device controls in dependency order.

        Concrete values only: `_with_restores` has already resolved a reset
        into the value that undoes it, so a None here is a skip rather than a
        change. Autofocus goes first: a focus value written while autofocus is
        on is immediately hunted away from.
        """
        if controls.get("autofocus") is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if controls["autofocus"] else 0)
        if controls.get("focus") is not None:
            cap.set(cv2.CAP_PROP_FOCUS, controls["focus"])
        if controls.get("brightness") is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, controls["brightness"])
        if controls.get("exposure") is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, controls["exposure"])

    def set_controls(self, **changes) -> None:
        """Queue control changes for the capture thread.

        Accepts brightness/exposure/autofocus/focus. Updating a dict rather
        than appending to a queue coalesces a fast slider drag to its newest
        value, so the capture thread never works through a backlog.

        A value of None is not "skip": it means stop this app writing the
        control. Autofocus is the one control that can then be handed *back* to
        the device's own mode, and `_undo_for` writes it; the other three have
        no mode to ask for, so a reset of those merely stops the writing — the
        device keeps the value it was last given.
        """
        with self._controls_lock:
            self._pending_controls.update(changes)

    def _drain_controls(self) -> None:
        with self._controls_lock:
            if not self._pending_controls:
                return
            changes = self._pending_controls
            self._pending_controls = {}
        # Merge onto the instance fields so a later reopen replays them.
        for name, value in changes.items():
            setattr(self, f"_{name}", value)
        ae_switch = changes.pop("auto_exposure", None)
        # Every value in one write, so the autofocus-before-focus order holds
        # across a batch that moves both.
        writes = _with_restores(changes)
        if ae_switch is True and self._ae is None:
            self._ae = AutoExposure(self.fps, self._brightness, self._exposure)
            writes.update(self._ae.start_writes())
        elif ae_switch is False and self._ae is not None:
            # Hand the two controls back to the operator's own values, where there are any.
            self._ae = None
            for name in ("brightness", "exposure"):
                value = getattr(self, f"_{name}")
                if value is not None:
                    writes[name] = value
        elif self._ae is not None:
            # A manual drag while auto is on is its next starting point, not a write.
            writes.pop("brightness", None)
            writes.pop("exposure", None)
        self._write_safely(writes)

    def _write_safely(self, writes: dict) -> None:
        if not writes:
            return
        # Deliberately outside the lock: cap.set() can block on some
        # backends, and holding the lock across it would stall the caller.
        #
        # Swallowed rather than raised: this runs on the capture thread, which
        # has no handler above it. A backend that rejects one value would
        # otherwise kill the thread mid-loop, leaving `failure` unset and
        # `_running` true — the feed freezes with nothing to explain it. A
        # control that will not take is not worth the stream.
        try:
            self._write_controls(self._cap, writes)
        except Exception as exc:  # noqa: BLE001 - see above
            self.control_error = f"Camera {self.index} rejected {sorted(writes)}: {exc}"

    def open(self) -> bool:
        self._cap = self._cap_factory(self.index)
        # MJPG first, then the size. The StreamCam reaches 60 fps at 720p/1080p only in MJPG; left
        # to choose, MSMF negotiated the uncompressed format and delivered 29 fps at a 60 fps setting
        # whatever the exposure. The order matters because MSMF picks its media type when the size
        # is set. A camera with no MJPG mode ignores the request and keeps its own format.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        controls = self._current_controls()
        if self._auto_exposure:
            self._ae = AutoExposure(self.fps, self._brightness, self._exposure)
            controls.update(self._ae.start_writes())
        self._write_controls(self._cap, controls)
        if not self._cap.isOpened():
            self.is_open = False
            return False
        self.is_open = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    # A stalled read is normal for a frame or two; a device that has gone away
    # never recovers. Give up on a deadline rather than a retry count, so the
    # time a user stares at a frozen image does not depend on the backoff.
    FAILURE_TIMEOUT_S = 3.0
    MAX_FAILURE_BACKOFF_S = 0.2

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                now = time.monotonic()
                if self._failing_since is None:
                    self._failing_since = now
                self._consecutive_failures += 1
                if now - self._failing_since >= self.FAILURE_TIMEOUT_S:
                    self.failure = (
                        f"Camera {self.index} stopped delivering frames for "
                        f"{self.FAILURE_TIMEOUT_S:.0f}s ({self._consecutive_failures} "
                        "attempts) — it may have been unplugged, suspended, or taken "
                        "by another program."
                    )
                    self._running = False
                    return
                # Back off instead of spinning: the first few failures retry
                # promptly, a dead device settles at 5 reads/second.
                time.sleep(min(0.005 * self._consecutive_failures, self.MAX_FAILURE_BACKOFF_S))
                continue
            self._consecutive_failures = 0
            self._failing_since = None
            self._drain_controls()
            if self._ae is not None:
                self._write_safely(self._ae.update(mean_luminance(frame), time.monotonic()))
            self._read_times.append(time.monotonic())
            self._seq += 1
            self._buffer.put(self._seq, frame)

    MEASURED_FPS_WINDOW_S = 1.0

    @property
    def measured_fps(self) -> float:
        """Frames delivered per second over the last second, 0.0 until known.

        A plain count over a fixed window — NOT (count-1)/span between the
        first and last sample. That span-based formula blows up whenever
        samples land close together (startup, or a burst right before a
        stall): two frames 1ms apart reports ~1000 fps for a full second,
        which is exactly wrong for a readout whose job is to catch a
        starved camera. A fixed-window count has no such edge case — it
        ramps up honestly from 0 and decays to 0 during a stall.
        """
        now = time.monotonic()
        # Snapshot before iterating: the capture thread appends to
        # _read_times concurrently, and deque raises "deque mutated during
        # iteration" if a mutation lands mid-comprehension.
        snapshot = list(self._read_times)
        recent = [t for t in snapshot if now - t <= self.MEASURED_FPS_WINDOW_S]
        return len(recent) / self.MEASURED_FPS_WINDOW_S

    def latest(self):
        return self._buffer.get()

    def read(self):
        got = self._buffer.get()
        return None if got is None else got[1]

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self.is_open = False
