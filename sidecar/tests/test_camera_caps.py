import cv2
import numpy as np
import pytest

from app import camera_caps
from app.camera_caps import (
    FOCUS_RELATIVE_THRESHOLD,
    ControlSupport,
    calibrate,
    exposure_ceiling,
    probe_controls,
    sweep_controls,
)
from app.camera_quality import FrameQuality


class _Cap:
    """A device where only the named controls actually change the image —
    the others accept the write and do nothing, exactly like MSMF's GAIN."""

    def __init__(self, effective):
        self.effective = effective
        self.level = 40.0

    def set(self, prop, value):
        if prop in self.effective:
            self.level = 180.0
        return True   # always True, as OpenCV does

    def get(self, prop):
        return 0.0    # the getter lies; the probe must ignore it


def test_a_control_counts_only_when_the_image_changes():
    """Setting exposure on MSMF changed delivered fps while get() returned the
    old value. Support is proven by the image, never by a getter."""
    cap = _Cap({cv2.CAP_PROP_BRIGHTNESS})
    support = probe_controls(cap, lambda: np.full((8, 8, 3), int(cap.level), dtype=np.uint8))

    assert support.brightness is True
    assert support.gain is False
    assert support.exposure is False


def test_a_device_that_ignores_everything_supports_nothing():
    cap = _Cap(set())
    support = probe_controls(cap, lambda: np.full((8, 8, 3), int(cap.level), dtype=np.uint8))

    assert support == ControlSupport(brightness=False, exposure=False, gain=False, focus=False)


class _FocusProbeCap:
    """Cap whose only meaningful state is whether FOCUS has been set yet.

    Driving the canned sharpness off `focus_set` (rather than off call order
    or count) keeps these tests independent of probe_controls's internal
    sampling details — they only need to know that a sharpness reading was
    taken before vs. after the CAP_PROP_FOCUS set.
    """

    def __init__(self):
        self.focus_set = False

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOCUS:
            self.focus_set = True
        return True

    def get(self, prop):
        return 0.0


def _probe_focus(monkeypatch, before_sharpness, after_sharpness):
    cap = _FocusProbeCap()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)  # content is irrelevant; frame_quality is mocked below

    def fake_frame_quality(_frame):
        sharpness = after_sharpness if cap.focus_set else before_sharpness
        return FrameQuality(brightness=100.0, contrast=0.0, sharpness=sharpness)

    monkeypatch.setattr(camera_caps, "frame_quality", fake_frame_quality)
    return probe_controls(cap, lambda: frame)


def test_focus_small_absolute_change_on_sharp_frame_is_not_supported(monkeypatch):
    """Sharpness (Laplacian variance) runs from ~5 on a soft frame to hundreds
    on a focused one (camera_quality.py). A fixed absolute delta is noise-level
    against that range, so a small absolute jump (5) on an already-sharp frame
    (500) must NOT count as support."""
    before = 500.0
    after = before + 5.0
    assert after - before < before * FOCUS_RELATIVE_THRESHOLD  # sanity: below the relative bar

    support = _probe_focus(monkeypatch, before_sharpness=before, after_sharpness=after)

    assert support.focus is False


def test_focus_proportionally_large_change_is_supported(monkeypatch):
    """A change that is a large fraction of the pre-set sharpness counts as
    supported, scaling with the baseline rather than a fixed absolute delta."""
    before = 500.0
    after = before * (1 + FOCUS_RELATIVE_THRESHOLD + 0.05)  # comfortably over the relative bar

    support = _probe_focus(monkeypatch, before_sharpness=before, after_sharpness=after)

    assert support.focus is True


def test_focus_probe_survives_a_dropped_frame_after_focus_is_set():
    """The focus probe reads immediately after cap.set(CAP_PROP_FOCUS, ...),
    which is exactly when a refocusing UVC camera is most likely to drop a
    frame. read_frame() returning None there must degrade focus to
    unsupported (an unmeasurable control is exactly an unsupported one),
    never raise cv2.error out of frame_quality(None, ...)."""
    cap = _FocusProbeCap()

    def read_frame():
        # Once focus has been set, simulate the dropped read: (False, None)
        # from cv2, already translated to None by the caller's read_frame
        # convention.
        return None if cap.focus_set else np.zeros((4, 4, 3), dtype=np.uint8)

    support = probe_controls(cap, read_frame)  # must not raise

    assert support.focus is False


def test_calibrate_measures_both_exposure_modes():
    """12 fps on auto, 30 with exposure capped — the difference is the whole
    reason to calibrate."""
    class _Cap:
        def __init__(self): self.capped = False
        def isOpened(self): return True
        def set(self, prop, value):
            import cv2
            if prop == cv2.CAP_PROP_EXPOSURE: self.capped = True
            return True
        def get(self, prop): return 0.0
        def read(self):
            import numpy as np
            return True, np.full((8, 8, 3), 40, dtype=np.uint8)
        def release(self): pass

    cap = _Cap()
    profile = calibrate(1, 1280, 720, open_device=lambda i, b: cap,
                        device_name="Fake Cam", sample_seconds=0.05)

    assert profile.device_key.startswith("Fake Cam")
    assert profile.fps_auto_exposure > 0
    assert profile.fps_capped_exposure > 0
    assert isinstance(profile.recommended, dict)


class _ReopenCap:
    """Minimal working camera double for the reopen-contract test below."""

    def isOpened(self): return True

    def set(self, prop, value): return True

    def get(self, prop): return 0.0

    def read(self):
        return True, np.full((8, 8, 3), 40, dtype=np.uint8)

    def release(self): pass


def test_calibrate_reopens_device_after_destructive_probe():
    """probe_controls is destructive by design (module docstring): it leaves
    brightness/exposure/gain/focus pinned to whatever probe value last stuck,
    and its docstring names calibrate() as the caller responsible for
    reopening the device afterward. If calibrate() skips that reopen, every
    calibrated camera silently keeps whatever values probing happened to
    set — nothing else here would fail. Pin the reopen by counting how many
    times the injected open_device factory is actually called: once to open
    for probing, once again afterward for a clean handle."""
    opened = []

    def open_device(index, backend):
        cap = _ReopenCap()
        opened.append(cap)
        return cap

    calibrate(1, 1280, 720, open_device=open_device,
              device_name="Fake Cam", sample_seconds=0.01)

    assert len(opened) >= 2, (
        "calibrate() must release and reopen the device after probe_controls, "
        "not keep using the same (now-dirtied) handle"
    )


class _FlakyCap:
    """A device that drops roughly every third read — e.g. a refocusing UVC
    camera stalling right after CAP_PROP_FOCUS is set. Exercises the full
    calibrate() pipeline (measure_fps, _brightness, probe_controls) against
    intermittent (False, None) reads, not just the focus branch in
    isolation."""

    def __init__(self):
        self.reads = 0

    def isOpened(self): return True

    def set(self, prop, value): return True

    def get(self, prop): return 0.0

    def read(self):
        self.reads += 1
        if self.reads % 3 == 0:
            return False, None
        return True, np.full((8, 8, 3), 40, dtype=np.uint8)

    def release(self): pass


def test_calibrate_completes_when_reads_are_intermittently_dropped():
    """An ~80s calibration must not be discarded as an HTTP 500 over a single
    dropped read. calibrate() must complete and hand back a real profile even
    when the underlying device flakes throughout."""
    cap = _FlakyCap()

    profile = calibrate(1, 1280, 720, open_device=lambda i, b: cap,
                        device_name="Fake Cam", sample_seconds=0.05)

    assert isinstance(profile.controls, ControlSupport)
    assert profile.fps_auto_exposure >= 0
    assert profile.fps_capped_exposure >= 0


class _BackendAwareCap:
    """MSMF refuses to open this device; the auto backend succeeds — mirrors
    camera.py's _default_capture fallback contract."""

    def __init__(self, backend):
        self._opened = backend != cv2.CAP_MSMF

    def isOpened(self): return self._opened

    def set(self, prop, value): return True

    def get(self, prop): return 0.0

    def read(self):
        return True, np.full((8, 8, 3), 40, dtype=np.uint8)

    def release(self): pass


def test_calibrate_falls_back_to_auto_backend_when_msmf_wont_open():
    """Compare camera.py's _default_capture: try MSMF, then fall back to
    OpenCV's auto backend rather than handing back — and later probing — an
    unopened handle."""
    opened_backends = []

    def open_device(index, backend):
        opened_backends.append(backend)
        return _BackendAwareCap(backend)

    profile = calibrate(1, 1280, 720, open_device=open_device,
                        device_name="Fake Cam", sample_seconds=0.01)

    assert cv2.CAP_MSMF in opened_backends
    assert cv2.CAP_ANY in opened_backends
    assert isinstance(profile.controls, ControlSupport)


class _NeverOpensCap:
    """Neither backend can open this device at all."""

    def isOpened(self): return False

    def set(self, prop, value): return True

    def get(self, prop): return 0.0

    def read(self): return False, None

    def release(self): pass


def test_calibrate_skips_probing_gracefully_when_the_device_wont_open():
    """No isOpened() check used to mean calibrate() would plow ahead calling
    .set()/.read() on a handle that was never open. It must instead fail
    clearly and early, not deep inside a getter/setter."""
    def open_device(index, backend):
        return _NeverOpensCap()

    with pytest.raises(RuntimeError):
        calibrate(1, 1280, 720, open_device=open_device,
                  device_name="Fake Cam", sample_seconds=0.01)


class _AutofocusCap:
    """A device that either honours CAP_PROP_AUTOFOCUS=0 or ignores it.

    When it honours the lock, sharpness holds still. When it ignores it, the
    lens keeps hunting and sharpness wanders — which is exactly what
    focus_drift measures.
    """

    def __init__(self, honours_lock: bool):
        self.honours_lock = honours_lock
        self.locked = False
        self._tick = 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_AUTOFOCUS and self.honours_lock:
            self.locked = value == 0
        return True

    def get(self, prop):
        return 0.0

    def sharpness_now(self) -> float:
        self._tick += 1
        if self.locked:
            return 80.0
        # Hunting: stdev/mean well above FOCUS_DRIFT_MAX.
        return 80.0 if self._tick % 2 else 20.0


def _sharp_reader(cap):
    """A frame whose Laplacian variance tracks cap.sharpness_now()."""
    def read():
        level = cap.sharpness_now()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        # A checkerboard's variance scales with its amplitude, giving
        # frame_quality a real gradient to measure.
        frame[::2, ::2] = int(level)
        return frame
    return read


def test_autofocus_counts_as_supported_when_the_lock_actually_holds():
    cap = _AutofocusCap(honours_lock=True)
    assert camera_caps.probe_autofocus(cap, _sharp_reader(cap)) is True


def test_a_device_that_ignores_the_autofocus_lock_is_not_supported():
    """The StreamCam hunts for faces. If the lock does not take, any focus
    value we measure is one the lens will wander off within seconds."""
    cap = _AutofocusCap(honours_lock=False)
    assert camera_caps.probe_autofocus(cap, _sharp_reader(cap)) is False


def test_autofocus_is_unsupported_when_frames_cannot_be_read():
    """An unmeasurable control is an unsupported one, never a crash — the
    same rule the focus probe already follows."""
    cap = _AutofocusCap(honours_lock=True)
    assert camera_caps.probe_autofocus(cap, lambda: None) is False


def test_probe_controls_reports_autofocus_alongside_the_rest():
    cap = _Cap({cv2.CAP_PROP_BRIGHTNESS})
    support = probe_controls(cap, lambda: np.full((8, 8, 3), int(cap.level), dtype=np.uint8))

    assert hasattr(support, "autofocus")


def test_calibrate_threads_target_fps_into_derive_camera_settings(monkeypatch):
    """The exposure gate in derive_camera_settings needs the operator's
    configured capture rate to be relative rather than an absolute floor
    (camera_derive.py) — calibrate() is the one place that has it, so it
    must actually pass it through."""
    import app.camera_derive as camera_derive

    captured = {}

    def fake_derive(profile, measured_brightness, near_conf=None, far_conf=None,
                     imgsz=640, target_fps=None):
        captured["target_fps"] = target_fps
        return {}

    monkeypatch.setattr(camera_derive, "derive_camera_settings", fake_derive)

    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0.0
        def read(self):
            return True, np.full((8, 8, 3), 40, dtype=np.uint8)
        def release(self): pass

    calibrate(1, 1280, 720, open_device=lambda i, b: _Cap(),
              device_name="Fake Cam", sample_seconds=0.01, target_fps=15.0)

    assert captured["target_fps"] == 15.0


def test_the_exposure_ceiling_is_arithmetic_not_measured():
    """Exposure is log2 seconds, so value e caps delivery at 1/2^e fps. The
    default capture_fps of 30 gives a floor of 24, and -5 is 1/32 s -> 32 fps
    — the longest exposure that still clears it. No probe is spent finding
    this cliff by falling off it."""
    assert exposure_ceiling(24.0) == -5
    assert exposure_ceiling(30.0) == -5
    assert exposure_ceiling(12.0) == -4


def test_the_exposure_ceiling_stays_inside_the_settable_range():
    """SettingsUpdateRequest bounds camera_exposure to -13..0."""
    assert exposure_ceiling(0.0) == 0
    assert exposure_ceiling(100000.0) == -13


class _SweepCap:
    """A device with a known response curve for each control.

    brightness rises with exposure and with brightness; sharpness peaks at
    focus 400. Deliberately not the real physics — just monotone and unimodal
    respectively, which is all the searches assume.
    """

    def __init__(self, support_focus=True, support_exposure=True, support_brightness=True):
        self.exposure = -13.0
        self.brightness = 0.0
        self.focus = 0.0
        self.writes: list[int] = []
        self.support = dict(
            focus=support_focus, exposure=support_exposure, brightness=support_brightness
        )

    def set(self, prop, value):
        self.writes.append(prop)
        if prop == cv2.CAP_PROP_EXPOSURE and self.support["exposure"]:
            self.exposure = value
        elif prop == cv2.CAP_PROP_BRIGHTNESS and self.support["brightness"]:
            self.brightness = value
        elif prop == cv2.CAP_PROP_FOCUS and self.support["focus"]:
            self.focus = value
        return True

    def get(self, prop):
        return 0.0

    def level(self) -> float:
        # 0 at exposure -13, 130 at -7, clipped to a byte.
        return max(0.0, min(255.0, (self.exposure + 13) * 21.7 + self.brightness * 0.4))

    def sharp(self) -> float:
        return 100.0 - ((self.focus - 400) / 100.0) ** 2


# Gain/baseline for rendering cap.sharp() into the fake frame below. cap.sharp()
# only spans ~61..100 and is locally almost flat within one FOCUS_STEP of its
# peak (it is a parabola's vertex) — embedded directly as a pixel value, the
# uint8 cast that frame_quality's cv2.cvtColor performs rounds that whole
# neighbourhood down to the same integer, turning the peak into a >150-wide
# plateau search_for_peak cannot resolve. Rescaling the swing above its floor
# by 10x before truncating keeps adjacent focus steps distinguishable near the
# peak without saturating the frame elsewhere in the range.
# The exact multiplier isn't load-bearing: swept in 0.5 steps, gain 9.0-12.0
# lands search_for_peak exactly on focus 400; 6.0-8.5 lands on 384, still
# inside the test's +/-32 assertion; <=5 or >=13 flips the result to 336 and
# fails it. 10.0 sits in the middle of that ~[6, 12] working band, not at
# its edge.
_FOCUS_GAIN = 10.0
_FOCUS_GAIN_FLOOR = 60.0  # just under cap.sharp()'s minimum (~61.19), so the
# gained amplitude never goes negative.


def _sweep_reader(cap):
    def read():
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        level = cap.level()
        # +half/-half rather than a single modulated patch: the pair's mean
        # stays pinned to `level` (cancels out) as long as neither
        # level+half nor level-half clips against [0, 255]. Near the
        # exposure floor it does clip — e.g. at exposure -9 (level=86.8)
        # with focus untouched (half=120 constant), level-half=-33.2 clips
        # to 0 and the measured brightness comes back ~94.5 instead of the
        # unbiased 86.8, a ~8-unit bias upward. The exposure tests tolerate
        # this because the biased reading is still far enough below
        # target=130 to land on the same side of the tolerance band (and
        # so take the same branch) as the unbiased value would.
        half = _FOCUS_GAIN * (cap.sharp() - _FOCUS_GAIN_FLOOR) / 2.0
        frame[:] = int(max(0.0, min(255.0, level)))
        # Modulate a quarter of pixels up and a different quarter down so
        # Laplacian variance tracks cap.sharp() while the mean stays at level.
        frame[::2, ::2] = int(max(0.0, min(255.0, level + half)))
        frame[1::2, 1::2] = int(max(0.0, min(255.0, level - half)))
        return frame
    return read


def test_the_sweep_finds_an_exposure_that_hits_the_brightness_target():
    cap = _SweepCap(support_focus=False)
    support = ControlSupport(exposure=True, autofocus=True)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert "camera_exposure" in measured
    assert measured["camera_exposure"]["reached"] is True
    # -7 lands on the target; the ceiling for a 24 fps floor is -5, so the
    # search had room without needing to be clamped.
    assert measured["camera_exposure"]["value"] == -7


def test_the_sweep_never_proposes_an_exposure_that_would_break_the_fps_floor():
    """A dark room the shutter cannot fix. Without the ceiling the search
    walks to -1 and caps the camera at 2 fps."""
    cap = _SweepCap(support_focus=False)
    cap.level = lambda: 20.0
    support = ControlSupport(exposure=True, autofocus=True)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert measured["camera_exposure"]["value"] <= -5
    assert measured["camera_exposure"]["reached"] is False


def test_the_sweep_finds_the_focus_with_the_highest_sharpness():
    cap = _SweepCap(support_exposure=False, support_brightness=False)
    support = ControlSupport(focus=True, autofocus=True)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert abs(measured["camera_focus"]["value"] - 400) <= 32
    assert measured["camera_focus"]["baseline"] < measured["camera_focus"]["metric"]


def test_a_control_the_device_ignores_is_never_swept():
    """probe_controls already established support. Sweeping a deaf control
    spends probes to discover a flat line we already knew about."""
    cap = _SweepCap()
    support = ControlSupport(brightness=False, exposure=False, focus=False, autofocus=False)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert measured == {}
    assert cap.writes == []


def test_focus_is_not_swept_when_the_autofocus_lock_will_not_take():
    """The lens would hunt straight off whatever we measured, so the probes
    are wasted and the answer would be confidently wrong."""
    cap = _SweepCap()
    support = ControlSupport(focus=True, exposure=True, autofocus=False)
    measured = sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert "camera_focus" not in measured
    assert "camera_exposure" in measured


def test_the_sweep_writes_the_autofocus_lock_before_any_focus_value():
    """Order is a dependency, not a preference: a value written under a live
    autofocus is hunted away from immediately."""
    cap = _SweepCap()
    support = ControlSupport(focus=True, autofocus=True)
    sweep_controls(cap, _sweep_reader(cap), support, fps_floor=24.0)

    assert cap.writes.index(cv2.CAP_PROP_AUTOFOCUS) < cap.writes.index(cv2.CAP_PROP_FOCUS)


def test_a_control_whose_frames_never_arrive_is_omitted_rather_than_scored():
    """Scoring an unreadable candidate 0 would poison a peak search into
    choosing whichever value happened to read successfully."""
    cap = _SweepCap()
    support = ControlSupport(focus=True, autofocus=True)
    measured = sweep_controls(cap, lambda: None, support, fps_floor=24.0)

    assert measured == {}


def test_a_failed_probe_does_not_take_down_the_other_controls():
    """sweep_controls wraps exposure, focus, and brightness in three separate
    try/except ProbeUnavailable blocks, not one around the whole function.

    Sweep order is exposure -> focus -> brightness. Asserting only that
    exposure (which runs BEFORE focus) survives a focus failure doesn't
    discriminate: a single try/except wrapping the whole function would
    also leave exposure's already-written result sitting in `measured`
    when focus's exception is caught, and `return measured` would hand
    back the same dict. The only way to prove the three guards are
    independent is to check a control that runs AFTER the failing one —
    brightness must still get its own, real, correct value, which only
    happens if focus's ProbeUnavailable was swallowed by its own handler
    and execution fell through to the brightness block rather than
    jumping straight to `return measured`.

    Fails only the focus probe by going dark exclusively right after
    CAP_PROP_FOCUS is written; exposure and brightness use the normal
    reader throughout."""
    cap = _SweepCap()
    support = ControlSupport(exposure=True, focus=True, brightness=True, autofocus=True)
    base = _sweep_reader(cap)

    def flaky_reader():
        if cap.writes and cap.writes[-1] == cv2.CAP_PROP_FOCUS:
            return None
        return base()

    measured = sweep_controls(cap, flaky_reader, support, fps_floor=24.0)

    assert "camera_focus" not in measured
    assert "camera_exposure" in measured
    assert measured["camera_exposure"]["value"] == -7
    # camera_brightness only appears with a real, correctly-searched value if
    # execution reached its block at all — the collapse this test targets
    # would jump straight from focus's exception to `return measured`,
    # omitting this key entirely rather than getting its value wrong.
    assert "camera_brightness" in measured
    assert measured["camera_brightness"]["value"] == 64
    assert measured["camera_brightness"]["reached"] is True
