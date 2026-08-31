"""Settings that take effect on a running pipeline. Every test uses fakes —
no camera, GPU or network, per the suite's convention."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, build_app
from app.schemas import Detection
from app.settings_store import HOT_RELOADABLE_FIELDS, RESTART_REQUIRED_FIELDS


class _Src:
    """Frame source that records the controls pushed to it."""

    width, height, fps = 64, 48, 30.0
    measured_fps = 29.0

    def __init__(self):
        self.controls: dict = {}

    def open(self):
        return True

    def latest(self):
        return (1, np.full((48, 64, 3), 130, dtype=np.uint8))

    def read(self):
        return np.full((48, 64, 3), 130, dtype=np.uint8)

    def set_controls(self, **changes):
        self.controls.update(changes)

    def release(self):
        pass


class _Det:
    names = {0: "milo"}

    def __init__(self):
        self.conf = 0.5

    def infer(self, frame):
        return [Detection(track_id=1, cls="milo", conf=0.9, box=(0.1, 0.1, 0.2, 0.2))]

    def set_conf(self, value):
        self.conf = float(value)


@pytest.fixture
def running(tmp_path):
    """A client with capture started against fakes, exposing the live
    source and detector so a test can assert what reached them."""
    src, det = _Src(), _Det()
    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        source_factory=lambda s: src,
        detector_factory=lambda s, d: det,
    )
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        yield client, state, src, det
        client.post("/api/capture/stop")


# --- the field sets ------------------------------------------------------


LIVE_FIELDS = {
    "conf_threshold",
    "camera_brightness",
    "camera_exposure",
    "camera_autofocus",
    "camera_focus",
}


def test_the_five_tunable_fields_are_hot_reloadable():
    assert LIVE_FIELDS <= HOT_RELOADABLE_FIELDS


def test_they_are_no_longer_restart_required():
    assert LIVE_FIELDS.isdisjoint(RESTART_REQUIRED_FIELDS)


def test_fields_that_need_a_reopen_are_still_restart_required():
    """The reopen path is ~30 s on a StreamCam and cannot be avoided."""
    assert {"camera_index", "capture_width", "capture_height", "capture_fps"} <= (
        RESTART_REQUIRED_FIELDS
    )


# --- pushing to the running pipeline -------------------------------------


def test_a_camera_control_patch_reaches_the_open_device(running):
    client, _, src, _ = running
    r = client.patch("/api/settings", json={"camera_brightness": 180.0})

    assert r.status_code == 200
    assert src.controls["brightness"] == 180.0


def test_autofocus_reaches_the_device_as_a_bool(running):
    client, _, src, _ = running
    client.patch("/api/settings", json={"camera_autofocus": False})

    assert src.controls["autofocus"] is False


def test_a_conf_patch_reaches_the_running_detector(running):
    client, _, _, det = running
    client.patch("/api/settings", json={"conf_threshold": 0.8})

    assert det.conf == 0.8


def test_camera_controls_no_longer_409_while_running(running):
    client, _, _, _ = running
    r = client.patch("/api/settings", json={"camera_exposure": -6.0})

    assert r.status_code == 200


def test_a_restart_required_field_still_409s_while_running(running):
    client, _, _, _ = running
    r = client.patch("/api/settings", json={"capture_width": 640})

    assert r.status_code == 409
    assert "stop capture" in r.json()["detail"].lower()


def test_patching_while_idle_touches_no_device(tmp_path):
    """With capture stopped there is no source or detector; the setattr is
    the whole job and must not raise."""
    state = AppState(settings_path=str(tmp_path / "settings.json"), db_path=":memory:")
    with TestClient(build_app(lambda: state)) as client:
        r = client.patch("/api/settings", json={"camera_brightness": 180.0})

    assert r.status_code == 200
    assert state.settings.camera_brightness == 180.0


def test_a_source_without_set_controls_is_tolerated(tmp_path):
    """FakeFrameSource and any future source need not implement it."""

    class _Bare:
        width, height, fps = 64, 48, 30.0
        measured_fps = 29.0

        def open(self):
            return True

        def latest(self):
            return (1, np.zeros((48, 64, 3), dtype=np.uint8))

        def release(self):
            pass

    state = AppState(
        settings_path=str(tmp_path / "settings.json"),
        db_path=":memory:",
        source_factory=lambda s: _Bare(),
        detector_factory=lambda s, d: _Det(),
    )
    with TestClient(build_app(lambda: state)) as client:
        client.post("/api/capture/start")
        r = client.patch("/api/settings", json={"camera_brightness": 180.0})
        client.post("/api/capture/stop")

    assert r.status_code == 200
