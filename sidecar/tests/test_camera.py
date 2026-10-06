import threading
import time

import cv2
import numpy as np
import pytest

from app.camera import CameraCapture, LatestFrameBuffer, FakeFrameSource


def _frame(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val, dtype=np.uint8)


def test_buffer_returns_none_when_empty():
    buf = LatestFrameBuffer()
    assert buf.get() is None


def test_buffer_newest_wins():
    buf = LatestFrameBuffer()
    buf.put(1, _frame(10))
    buf.put(2, _frame(20))
    seq, frame = buf.get()
    assert seq == 2
    assert frame[0, 0, 0] == 20


def test_fake_frame_source_yields_then_none():
    src = FakeFrameSource([_frame(1), _frame(2)], fps=30.0)
    src.open()
    assert src.read()[0, 0, 0] == 1
    assert src.read()[0, 0, 0] == 2
    assert src.read() is None
    assert src.fps == 30.0
    src.release()


# --- a device that goes away ---------------------------------------------


class _FailingCap:
    """Opens fine, then never yields a frame — an invalidated device."""

    def __init__(self, fail_after=0):
        self.reads = 0
        self._fail_after = fail_after

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        self.reads += 1
        if self.reads <= self._fail_after:
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return False, None

    def release(self):
        pass


def test_capture_gives_up_on_a_device_that_stopped_delivering():
    """It used to retry ~200x/second forever, burning a core and writing
    ~23,000 OpenCV warnings while the app looked like it was running."""
    cap = _FailingCap()
    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: cap)
    c.FAILURE_TIMEOUT_S = 0.3  # the real 3 s deadline, shortened for the test
    c.open()
    deadline = time.time() + 5
    while time.time() < deadline and not c.failure:
        time.sleep(0.02)
    c.release()

    assert c.failure is not None
    assert "stopped delivering frames" in c.failure
    # Backed off rather than spinning: ~200/s unthrottled would be far more.
    assert cap.reads < 100


def test_a_transient_read_failure_does_not_kill_capture():
    # One bad read among good ones must not take the camera down.
    class _Flaky(_FailingCap):
        def read(self):
            self.reads += 1
            if self.reads % 10 == 0:
                return False, None
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    cap = _Flaky()
    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: cap)
    c.open()
    time.sleep(0.3)
    failure = c.failure
    c.release()

    assert failure is None


def test_pipeline_reports_a_dead_camera_instead_of_freezing():
    """Otherwise capture stays 'running' with a frozen image and no reason."""
    from app.pipeline import Pipeline
    from app.settings import Settings

    class _DeadSource:
        width, height, fps = 128, 96, 30.0
        failure = "Camera 0 stopped delivering frames after 150 attempts."

        def latest(self):
            return None

    pipe = Pipeline(_DeadSource(), None, Settings(), on_message=lambda m: None)
    with pytest.raises(RuntimeError, match="stopped delivering frames"):
        pipe.process_once()


# --- measured delivery rate ------------------------------------------------


def test_capture_reports_the_rate_it_actually_delivers():
    """capture_fps used to report the *requested* value: the UI showed 60
    while the camera delivered 12, hiding a 5x shortfall all session."""
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self):
            time.sleep(0.01)  # ~100 fps ceiling
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open()
    time.sleep(1.2)
    rate = c.measured_fps
    c.release()

    assert rate > 10.0          # it is measuring something real
    assert rate < 300.0         # and not nonsense


def test_measured_fps_does_not_inflate_on_burst_then_stall():
    """A burst of frames close together followed by a stall used to report
    an inflated rate: (n-1)/span blows up when span is tiny. Two frames 1ms
    apart within the last second, with nothing more recent (i.e. a stall
    right after the burst), must report a plain count over the window (2.0),
    not ~1000."""
    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: object())
    now = time.monotonic()
    c._read_times.append(now - 0.5)
    c._read_times.append(now - 0.499)  # 1ms after the previous sample

    rate = c.measured_fps

    assert rate == pytest.approx(2.0)


def test_measured_fps_is_zero_before_any_frame():
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self): return False, None
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    assert c.measured_fps == 0.0


def test_open_applies_configured_controls():
    """Auto exposure and face-tracking autofocus are wrong for a counter:
    the StreamCam's smart AF/AE follows faces, and there is no face here."""
    sets = []

    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value):
            sets.append((prop, value)); return True
        def get(self, prop): return 0
        def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap(),
                      brightness=180.0, exposure=-6.0, autofocus=False, focus=30.0)
    c.open(); c.release()

    assert (cv2.CAP_PROP_BRIGHTNESS, 180.0) in sets
    assert (cv2.CAP_PROP_EXPOSURE, -6.0) in sets
    assert (cv2.CAP_PROP_AUTOFOCUS, 0) in sets
    assert (cv2.CAP_PROP_FOCUS, 30.0) in sets


def test_unset_controls_are_left_alone():
    """None means 'do not touch', so existing behaviour is unchanged."""
    sets = []

    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value):
            sets.append(prop); return True
        def get(self, prop): return 0
        def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open(); c.release()

    assert cv2.CAP_PROP_BRIGHTNESS not in sets
    assert cv2.CAP_PROP_EXPOSURE not in sets


def test_measured_fps_survives_concurrent_reads():
    """measured_fps used to iterate _read_times directly while the capture
    thread appended to it, which can raise 'deque mutated during iteration'.
    Hammer both from separate threads and confirm no exception surfaces."""
    class _Cap:
        def isOpened(self): return True
        def set(self, prop, value): return True
        def get(self, prop): return 0
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self): pass

    c = CameraCapture(0, 640, 480, 30, cap_factory=lambda i: _Cap())
    c.open()

    errors: list[BaseException] = []

    def _hammer():
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                c.measured_fps
            except BaseException as exc:  # noqa: BLE001 - pin the race, not a specific type
                errors.append(exc)
                return

    readers = [threading.Thread(target=_hammer) for _ in range(4)]
    for t in readers:
        t.start()
    for t in readers:
        t.join()
    c.release()

    assert errors == []


# --- live control changes -------------------------------------------------


class _RecordingCap:
    """Opens fine, yields frames forever, and records every set() with the
    name of the thread that made it.

    Deliberately has no `get`: nothing in this class may read a control back,
    so a `get` here would turn a stray read into an AttributeError rather than
    a value the app could act on. See `_undo_for` for why.
    """

    def __init__(self):
        self.sets: list[tuple[int, object, str]] = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value, threading.current_thread().name))
        return True

    def read(self):
        time.sleep(0.001)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _wrote(cap, prop, value) -> bool:
    return any(p == prop and v == value for p, v, _ in cap.sets)


def _wait_for(predicate, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_control_writes_happen_on_the_capture_thread():
    """cv2.VideoCapture is not thread-safe and _loop is calling read() on a
    background thread, so a set() issued from the caller's thread would race
    it. The write must be deferred to the thread that owns the handle."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=140.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 140.0))
        writers = {t for p, _, t in cap.sets if p == cv2.CAP_PROP_BRIGHTNESS}
        assert threading.current_thread().name not in writers
    finally:
        src.release()


def test_set_controls_coalesces_a_fast_drag():
    """A slider drag emits dozens of values. Only the newest matters, and
    applying every one would stall reads behind a queue of set() calls."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=100.0)
        src.set_controls(brightness=110.0)
        src.set_controls(brightness=120.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 120.0))
        written = [v for p, v, _ in cap.sets if p == cv2.CAP_PROP_BRIGHTNESS]
        assert 110.0 not in written
    finally:
        src.release()


def test_controls_set_live_survive_a_reopen():
    """A restart (resolution change, say) rebuilds the handle. Values tuned
    live must come back with it, or a restart silently reverts them."""
    caps = []

    def factory(index):
        cap = _RecordingCap()
        caps.append(cap)
        return cap

    src = CameraCapture(0, 4, 4, 30, cap_factory=factory)
    src.open()
    src.set_controls(brightness=140.0)
    assert _wait_for(lambda: _wrote(caps[0], cv2.CAP_PROP_BRIGHTNESS, 140.0))
    src.release()

    src.open()
    src.release()
    assert _wrote(caps[1], cv2.CAP_PROP_BRIGHTNESS, 140.0)


def test_autofocus_is_written_before_focus_when_set_live():
    """Same ordering open() has always used: a focus value written while
    autofocus is on is immediately hunted away from."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(focus=30.0, autofocus=False)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_FOCUS, 30.0))
        props = [p for p, _, _ in cap.sets]
        assert props.index(cv2.CAP_PROP_AUTOFOCUS) < props.index(cv2.CAP_PROP_FOCUS)
    finally:
        src.release()


def test_a_reset_of_a_control_with_no_undo_leaves_the_device_holding_it():
    """Revert clears the *setting*, and that is all it can do here: nothing on
    this device will say what a control held before the app wrote it, so the
    reset stops the app writing it rather than inventing a value to put back —
    which is what `_undo_for` explains, and what a stray `get` would have
    turned into a wrong write on real hardware. `_RecordingCap` has no `get`,
    so a read would raise instead."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=180.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 180.0))

        src.set_controls(brightness=None)

        def _wrote_brightness_twice() -> bool:
            return len([p for p, _, _ in cap.sets if p == cv2.CAP_PROP_BRIGHTNESS]) > 1

        assert not _wait_for(_wrote_brightness_twice, timeout=0.2)
        assert src._thread.is_alive()
        assert src.failure is None
    finally:
        src.release()


def test_a_reset_the_device_never_saw_does_not_touch_it():
    """A control this app never wrote has nothing to un-write, so the reset is
    a settings change and no device call at all."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        before = list(cap.sets)
        src.set_controls(brightness=None)
        assert not _wait_for(lambda: cap.sets != before, timeout=0.2)
    finally:
        src.release()


def test_resetting_autofocus_hands_focus_back_to_the_lens():
    """The one control whose automatic mode the app can ask for by name — and
    the one where reading the device would have been actively wrong, since
    `CAP_PROP_AUTOFOCUS` answers 0 while the lens is in auto (measured)."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(autofocus=False)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_AUTOFOCUS, 0))

        src.set_controls(autofocus=None)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_AUTOFOCUS, 1))
    finally:
        src.release()


def test_a_batch_that_resets_one_control_and_sets_another_writes_both():
    """Revert sends one patch for every dirty field, so an autofocus hand-back
    and a brightness change arrive together and must go out in the order
    `_write_controls` enforces."""
    cap = _RecordingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=180.0, autofocus=None)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_AUTOFOCUS, 1))
        assert _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 180.0)
    finally:
        src.release()


def test_a_rejected_control_write_does_not_kill_the_capture_thread():
    """_loop has no handler above it. A backend that raises on set() would
    otherwise end the thread mid-loop with `failure` unset and `_running`
    still true — the feed freezes and nothing says why. Losing one control
    is not worth losing the stream."""

    class _RefusingCap(_RecordingCap):
        def set(self, prop, value):
            if prop == cv2.CAP_PROP_BRIGHTNESS and value == 999.0:
                raise RuntimeError("backend refused brightness")
            return super().set(prop, value)

    cap = _RefusingCap()
    src = CameraCapture(0, 4, 4, 30, cap_factory=lambda i: cap)
    src.open()
    try:
        src.set_controls(brightness=999.0)
        assert _wait_for(lambda: src.control_error is not None)
        assert "brightness" in src.control_error
        assert src.failure is None

        # Still delivering frames, and still accepting later writes.
        src.set_controls(exposure=-6.0)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_EXPOSURE, -6.0))
        assert src._thread.is_alive()
    finally:
        src.release()


def test_open_asks_for_mjpg_before_the_resolution():
    """The StreamCam reaches 60 fps at 720p/1080p only in MJPG; left to choose, MSMF negotiated the
    uncompressed format and the camera delivered 29 fps at a 60 fps setting whatever the exposure.
    The format has to be asked for before the size, because MSMF picks a media type when the size is
    set. A camera without MJPG ignores the request and keeps its own format."""
    cap = _RecordingCap()
    src = CameraCapture(0, 1280, 720, 60, cap_factory=lambda i: cap)
    src.open()
    try:
        props = [p for p, _, _ in cap.sets]
        assert _wrote(cap, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_WIDTH)
    finally:
        src.release()


# ---- auto exposure -------------------------------------------------------------------------

from app.camera import AutoExposure, exposure_cap_for, mean_luminance  # noqa: E402


def test_the_exposure_cap_is_the_longest_shutter_the_framerate_allows():
    assert exposure_cap_for(60) == -6
    assert exposure_cap_for(30) == -5
    assert exposure_cap_for(0) == 0


def _settle(ae: AutoExposure, scene, steps: int = 200) -> float:
    """Run the loop against a simulated camera: luminance = scene(brightness, exposure)."""
    # The clock carries on across calls, as the capture thread's does.
    t = _CLOCK.get(id(ae), 0.0)
    lum = scene(ae.brightness, ae.exposure)
    for _ in range(steps):
        t += 0.05
        ae.update(lum, t)
        lum = scene(ae.brightness, ae.exposure)
    _CLOCK[id(ae)] = t
    return lum


_CLOCK: dict[int, float] = {}


def _room(light: float):
    # Roughly the StreamCam at -6: ~0.8 luminance per brightness unit, doubling per stop.
    return lambda b, e: min(255.0, max(0.0, (b * 0.8 - 50) * light * 2 ** (e + 6)))


def _sunlit(light: float):
    # Brightness as a pure gain, so a strong light overexposes even at the brightness floor.
    return lambda b, e: min(255.0, b * 0.6 * light * 2 ** (e + 6))


def test_a_dim_room_is_brought_to_the_target_by_brightness_alone_at_60_fps():
    ae = AutoExposure(60, brightness=None, exposure=None)
    lum = _settle(ae, _room(0.8))
    assert abs(lum - AutoExposure.TARGET) <= AutoExposure.DEADBAND + 4
    assert ae.exposure == -6  # the shutter never went past the 60 fps cap


def test_a_very_dark_room_never_lengthens_the_shutter_past_the_cap():
    ae = AutoExposure(60, brightness=128, exposure=-6)
    _settle(ae, _room(0.1))
    assert ae.exposure == -6
    assert ae.brightness == AutoExposure.BRIGHTNESS_MAX


def test_a_bright_scene_shortens_the_shutter_instead_of_crushing_brightness():
    ae = AutoExposure(60, brightness=200, exposure=-6)
    # Sunlit: even at the lowest useful brightness the picture would be blown out.
    lum = _settle(ae, _sunlit(6.0), steps=400)
    assert ae.exposure < -6
    assert ae.brightness >= AutoExposure.BRIGHTNESS_MIN
    assert abs(lum - AutoExposure.TARGET) <= AutoExposure.DEADBAND + 10


def test_it_comes_back_up_to_the_cap_when_the_light_drops_again():
    ae = AutoExposure(60, brightness=200, exposure=-6)
    _settle(ae, _sunlit(6.0), steps=400)
    _settle(ae, _sunlit(0.8), steps=400)
    assert ae.exposure == -6


def test_inside_the_deadband_nothing_is_written():
    ae = AutoExposure(60, brightness=128, exposure=-6)
    assert ae.update(AutoExposure.TARGET + 5, 1.0) == {}


def test_a_manual_exposure_longer_than_the_cap_is_clamped_at_start():
    ae = AutoExposure(60, brightness=128, exposure=-2)
    assert ae.start_writes()["exposure"] == -6


def test_mean_luminance_reads_a_bgr_frame():
    assert mean_luminance(np.full((16, 16, 3), 100, dtype=np.uint8)) == pytest.approx(100, abs=0.5)


class _SceneCap(_RecordingCap):
    """A camera whose picture brightness follows the brightness control, like the StreamCam."""

    def __init__(self):
        super().__init__()
        self.brightness = 128.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_BRIGHTNESS:
            self.brightness = float(value)
        return super().set(prop, value)

    def read(self):
        time.sleep(0.002)
        level = int(max(0, min(255, self.brightness * 0.8 - 50)))
        # A scene with detail in it (8-pixel blocks +-20 around the level, so the mean is the level
        # and the contrast is a lit room's - blocks, because `frame_levels` samples every 8th
        # pixel), since a flat frame is what `AutoExposure` now reads as too dark to brighten.
        rows, cols = np.indices((16, 16))
        sign = np.where((rows // 8 + cols // 8) % 2 == 0, 20, -20)
        frame = np.clip(level + sign, 0, 255).astype(np.uint8)
        return True, np.repeat(frame[:, :, None], 3, axis=2)


def test_capture_with_auto_exposure_drives_the_picture_to_the_target():
    cap = _SceneCap()
    src = CameraCapture(0, 16, 16, 60, cap_factory=lambda i: cap, auto_exposure=True)
    src.open()
    try:
        assert _wait_for(lambda: abs(cap.brightness * 0.8 - 50 - AutoExposure.TARGET) <= 15, 5.0)
        assert _wrote(cap, cv2.CAP_PROP_EXPOSURE, -6.0)
        assert all(t != threading.current_thread().name for p, _, t in cap.sets
                   if p == cv2.CAP_PROP_BRIGHTNESS and _ != 128.0)
    finally:
        src.release()


def test_switching_auto_exposure_off_restores_the_manual_values():
    cap = _SceneCap()
    src = CameraCapture(0, 16, 16, 60, cap_factory=lambda i: cap,
                        brightness=90, exposure=-7, auto_exposure=True)
    src.open()
    try:
        assert _wait_for(lambda: cap.brightness > 150, 5.0)
        src.set_controls(auto_exposure=False)
        assert _wait_for(lambda: cap.brightness == 90.0)
        assert _wrote(cap, cv2.CAP_PROP_EXPOSURE, -7)
        settled = len(cap.sets)
        time.sleep(0.5)
        assert len(cap.sets) == settled  # the loop stopped writing
    finally:
        src.release()


def test_a_manual_drag_while_auto_is_on_is_not_written():
    cap = _SceneCap()
    src = CameraCapture(0, 16, 16, 60, cap_factory=lambda i: cap, auto_exposure=True)
    src.open()
    try:
        assert _wait_for(lambda: cap.brightness > 150, 5.0)
        src.set_controls(brightness=10)
        time.sleep(0.3)
        assert not _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, 10)
    finally:
        src.release()


# ---- too dark: a state, not a brightness ----------------------------------------------------

from app.camera import frame_levels  # noqa: E402

GREY_CARD = 2.3  # measured contrast of the grey picture the old loop produced in a dark room
LIT = 28.0  # an ordinary lit scene measured 24-33


def _tick(ae: AutoExposure, lum: float, contrast: float, t: list[float]) -> dict:
    t[0] += 1.0  # past both the interval and the settle time
    return ae.update(lum, t[0], contrast)


def test_a_room_too_dark_for_the_shutter_is_reported_rather_than_greyed_out():
    """The 04:30 measurement: at -6 the picture was black, and holding the target by brightness
    painted it a flat grey (contrast 2.3). The loop now backs brightness off and says so."""
    ae = AutoExposure(60, brightness=215, exposure=-6)
    t = [0.0]
    assert _tick(ae, 100, GREY_CARD, t) == {"brightness": AutoExposure.BRIGHTNESS_NEUTRAL}
    assert ae.too_dark
    # Black and flat at neutral brightness: nothing more to write, and no climbing back to grey.
    assert _tick(ae, 1, 2.4, t) == {}
    assert ae.too_dark and ae.brightness == AutoExposure.BRIGHTNESS_NEUTRAL
    assert ae.exposure == -6  # the 60 fps shutter was kept


def test_light_coming_back_clears_the_state_and_brightness_works_again():
    ae = AutoExposure(60, brightness=128, exposure=-6)
    t = [0.0]
    _tick(ae, 1, 2.4, t)
    assert ae.too_dark
    writes = _tick(ae, 70, LIT, t)
    assert not ae.too_dark
    assert writes.get("brightness", 0) > 128


def test_a_dark_but_detailed_picture_is_still_brightened():
    """Flatness is what decides it, not darkness: a dim picture with detail in it is the case
    brightness exists for."""
    ae = AutoExposure(60, brightness=128, exposure=-6)
    t = [0.0]
    writes = _tick(ae, 60, LIT, t)
    assert writes.get("brightness", 0) > 128 and not ae.too_dark


def test_without_a_contrast_reading_the_loop_behaves_as_before():
    ae = AutoExposure(60, brightness=128, exposure=-6)
    assert ae.update(1, 1.0) == {"brightness": 128 + AutoExposure.MAX_STEP}
    assert not ae.too_dark


def test_the_slow_trade_spends_one_stop_before_giving_up_and_no_more():
    ae = AutoExposure(60, brightness=128, exposure=-6, allow_slow=True)
    t = [0.0]
    assert ae.slow_cap == -5
    assert _tick(ae, 1, 2.4, t) == {"exposure": -5}
    assert not ae.too_dark  # not given up yet: the slower shutter has not been judged
    assert _tick(ae, 3, 5.0, t) == {}  # still too dark at 30 fps
    assert ae.too_dark and ae.exposure == -5


def test_the_slow_trade_comes_back_to_the_cap_only_with_light_to_spare():
    ae = AutoExposure(60, brightness=128, exposure=-6, allow_slow=True)
    t = [0.0]
    _tick(ae, 1, 2.4, t)
    assert ae.exposure == -5
    # On target at 30 fps: halving the light would undo it, so it stays.
    _tick(ae, AutoExposure.TARGET, LIT, t)
    assert ae.exposure == -5
    # Twice the target's floor: one stop shorter still reaches it.
    assert _tick(ae, 240, LIT, t) == {"exposure": -6}


def test_the_slow_trade_tries_the_cap_again_when_light_returns_gradually():
    """At 30 fps the brightness loop holds the picture near the target, so the 'twice the target'
    reading that proves there is light to spare never appears in an ordinarily lit room. The loop
    tries the cap on a timer instead: with light back it stays there."""
    ae = AutoExposure(60, brightness=128, exposure=-6, allow_slow=True)
    t = [0.0]
    _tick(ae, 1, 2.4, t)
    assert ae.exposure == -5
    # Light comes back gradually: on target at 30 fps, never near 236.
    for _ in range(int(AutoExposure.SLOW_RETRY_S)):
        _tick(ae, AutoExposure.TARGET, LIT, t)
    assert ae.exposure == -6  # tried the cap
    _tick(ae, AutoExposure.TARGET - 5, LIT, t)  # lit and in the deadband at 60 fps
    assert ae.exposure == -6


def test_the_slow_trade_steps_back_if_the_room_is_still_dark_at_the_retry():
    ae = AutoExposure(60, brightness=128, exposure=-6, allow_slow=True)
    t = [0.0]
    _tick(ae, 1, 2.4, t)
    for _ in range(int(AutoExposure.SLOW_RETRY_S)):
        _tick(ae, 3, 5.0, t)
    assert ae.exposure == -6  # the retry
    assert _tick(ae, 1, 2.4, t) == {"exposure": -5}  # still dark: back to 30 fps


def test_switching_auto_exposure_off_takes_the_slow_shutter_back_when_none_was_set():
    cap = _DarkCap()
    src = CameraCapture(0, 16, 16, 60, cap_factory=lambda i: cap, auto_exposure=True,
                        auto_exposure_slow=True)
    src.open()
    try:
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_EXPOSURE, -5.0), 5.0)
        src.set_controls(auto_exposure=False)
        assert _wait_for(
            lambda: [v for p, v, _ in cap.sets if p == cv2.CAP_PROP_EXPOSURE][-1] == -6.0, 5.0
        )
    finally:
        src.release()


def test_turning_the_slow_trade_off_puts_the_shutter_straight_back():
    ae = AutoExposure(60, brightness=128, exposure=-6, allow_slow=True)
    t = [0.0]
    _tick(ae, 1, 2.4, t)
    assert ae.set_allow_slow(False) == {"exposure": -6}
    assert ae.set_allow_slow(False) == {}


def test_frame_levels_tells_a_flat_frame_from_a_detailed_one():
    flat = np.full((32, 32, 3), 100, dtype=np.uint8)
    rows, cols = np.indices((32, 32))
    checker = np.where((rows // 8 + cols // 8) % 2 == 0, 130, 70).astype(np.uint8)
    detailed = np.repeat(checker[:, :, None], 3, axis=2)
    assert frame_levels(flat) == pytest.approx((100, 0), abs=0.5)
    mean, contrast = frame_levels(detailed)
    assert mean == pytest.approx(100, abs=0.5) and contrast > AutoExposure.CONTRAST_FLOOR


class _DarkCap(_RecordingCap):
    """A camera in a dark room: a flat, nearly black picture whatever is written to it."""

    def read(self):
        time.sleep(0.002)
        return True, np.full((16, 16, 3), 2, dtype=np.uint8)


def test_capture_reports_too_dark_only_while_auto_exposure_is_judging():
    cap = _DarkCap()
    src = CameraCapture(0, 16, 16, 60, cap_factory=lambda i: cap, brightness=215, auto_exposure=True)
    src.open()
    try:
        assert _wait_for(lambda: src.too_dark, 5.0)
        assert _wrote(cap, cv2.CAP_PROP_BRIGHTNESS, AutoExposure.BRIGHTNESS_NEUTRAL)
        src.set_controls(auto_exposure=False)
        assert _wait_for(lambda: not src.too_dark, 5.0)
    finally:
        src.release()


def test_capture_turns_the_slow_trade_on_live():
    cap = _DarkCap()
    src = CameraCapture(0, 16, 16, 60, cap_factory=lambda i: cap, auto_exposure=True)
    src.open()
    try:
        assert _wait_for(lambda: src.too_dark, 5.0)
        src.set_controls(auto_exposure_slow=True)
        assert _wait_for(lambda: _wrote(cap, cv2.CAP_PROP_EXPOSURE, -5.0), 5.0)
        src.set_controls(auto_exposure_slow=False)
        assert _wait_for(lambda: [v for p, v, _ in cap.sets if p == cv2.CAP_PROP_EXPOSURE][-1] == -6.0, 5.0)
    finally:
        src.release()
