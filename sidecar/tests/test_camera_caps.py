import cv2
import numpy as np

from app import camera_caps
from app.camera_caps import FOCUS_RELATIVE_THRESHOLD, ControlSupport, calibrate, probe_controls
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
