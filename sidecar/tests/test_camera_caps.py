import cv2
import numpy as np

from app import camera_caps
from app.camera_caps import FOCUS_RELATIVE_THRESHOLD, ControlSupport, probe_controls
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
