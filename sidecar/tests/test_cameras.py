"""Camera enumeration. Every test runs against fakes — no real device is ever
opened, matching the convention in the rest of the suite."""

import cv2
import pytest

from app.cameras import CameraDevice, list_cameras, probe_index
from app.main import AppState, build_app
from fastapi.testclient import TestClient


class FakeCap:
    """Stands in for cv2.VideoCapture. `size` None means the device is absent."""

    def __init__(self, size):
        self._size = size
        self.released = False

    def isOpened(self):
        return self._size is not None

    def set(self, prop, value):
        return True

    def get(self, prop):
        if self._size is None:
            return 0
        w, h = self._size
        return float(w if prop == cv2.CAP_PROP_FRAME_WIDTH else h)

    def release(self):
        self.released = True


def factory_for(sizes):
    """cap_factory over a dict of index -> (w, h)."""
    made = {}

    def factory(index):
        cap = FakeCap(sizes.get(index))
        made[index] = cap
        return cap

    factory.made = made
    return factory


# --- probe_index ---------------------------------------------------------


def test_probe_index_reports_the_size_the_device_opened_at():
    assert probe_index(0, factory_for({0: (1920, 1080)})) == (1920, 1080)


def test_probe_index_returns_none_for_an_absent_device():
    assert probe_index(3, factory_for({0: (1280, 720)})) is None


def test_probe_index_releases_the_device_even_when_absent():
    factory = factory_for({})
    probe_index(0, factory)
    assert factory.made[0].released is True


def test_probe_index_releases_the_device_on_success():
    factory = factory_for({0: (1280, 720)})
    probe_index(0, factory)
    assert factory.made[0].released is True


def test_probe_index_treats_a_zero_size_as_unusable():
    # An opened-but-broken device reports 0x0; it must not reach the dropdown.
    assert probe_index(0, factory_for({0: (0, 0)})) is None


def test_probe_index_never_raises_when_the_factory_explodes():
    def boom(index):
        raise OSError("device busy")

    assert probe_index(0, boom) is None


# --- list_cameras --------------------------------------------------------


def test_list_cameras_pairs_names_onto_indices_in_order():
    cams = list_cameras(
        name_lister=lambda: ["USB2.0 HD UVC WebCam", "Logitech StreamCam"],
        cap_factory=factory_for({0: (1280, 720), 1: (1920, 1080)}),
    )
    assert cams == [
        CameraDevice(index=0, name="USB2.0 HD UVC WebCam", width=1280, height=720),
        CameraDevice(index=1, name="Logitech StreamCam", width=1920, height=1080),
    ]


def test_list_cameras_stops_at_the_first_gap():
    # Indices are dense; a gap means the end, and probing past it wastes seconds.
    cams = list_cameras(
        name_lister=lambda: [],
        cap_factory=factory_for({0: (1280, 720), 2: (1920, 1080)}),
    )
    assert [c.index for c in cams] == [0]


def test_list_cameras_falls_back_to_a_generic_name_when_windows_names_fewer():
    cams = list_cameras(
        name_lister=lambda: ["Only One"],
        cap_factory=factory_for({0: (1280, 720), 1: (1920, 1080)}),
    )
    assert [c.name for c in cams] == ["Only One", "Camera 1"]


def test_list_cameras_survives_a_failing_name_lister():
    # No PowerShell / non-Windows must degrade to indices, not crash the panel.
    def boom():
        raise OSError("no powershell")

    cams = list_cameras(name_lister=boom, cap_factory=factory_for({0: (640, 480)}))
    assert [c.name for c in cams] == ["Camera 0"]


def test_list_cameras_returns_empty_when_no_device_opens():
    assert list_cameras(name_lister=lambda: ["Ghost"], cap_factory=factory_for({})) == []


def test_list_cameras_respects_max_index():
    cams = list_cameras(
        max_index=2,
        name_lister=lambda: [],
        cap_factory=factory_for({0: (1, 1), 1: (1, 1), 2: (1, 1)}),
    )
    assert [c.index for c in cams] == [0, 1]


# --- GET /api/cameras ----------------------------------------------------


@pytest.fixture
def client():
    devices = [
        CameraDevice(index=0, name="USB2.0 HD UVC WebCam", width=1280, height=720),
        CameraDevice(index=1, name="Logitech StreamCam", width=1920, height=1080),
    ]
    state = AppState(camera_lister=lambda: list(devices))
    app = build_app(lambda: state)
    with TestClient(app) as c:
        c.state = state
        yield c


def test_cameras_endpoint_returns_named_devices(client):
    body = client.get("/api/cameras").json()
    assert body["probed"] is True
    assert [(c["index"], c["name"]) for c in body["cameras"]] == [
        (0, "USB2.0 HD UVC WebCam"),
        (1, "Logitech StreamCam"),
    ]


def test_cameras_endpoint_reports_the_probed_resolution(client):
    # The operator's check that a name landed on the right index.
    cams = client.get("/api/cameras").json()["cameras"]
    assert (cams[1]["width"], cams[1]["height"]) == (1920, 1080)


def test_cameras_endpoint_does_not_probe_while_capture_is_running(client):
    client.get("/api/cameras")  # populate the cache while idle
    calls = []
    client.state.camera_lister = lambda: calls.append(1) or []
    client.state.state = "running"

    body = client.get("/api/cameras").json()
    assert calls == []  # opening a device mid-capture would fight the pipeline
    assert body["probed"] is False
    assert "stop it to rescan" in body["detail"].lower()
    # The cached list still populates the dropdown.
    assert [c["index"] for c in body["cameras"]] == [0, 1]


def test_cameras_endpoint_is_empty_when_running_before_any_scan(client):
    client.state.state = "running"
    body = client.get("/api/cameras").json()
    assert body["cameras"] == []
    assert body["probed"] is False


def test_cameras_endpoint_caches_between_calls(client):
    calls = []
    client.state.camera_lister = lambda: calls.append(1) or [
        CameraDevice(index=0, name="Cam", width=640, height=480)
    ]
    client.get("/api/cameras")
    body = client.get("/api/cameras").json()
    # Scanning opens every device (~30 s under contention) — once is enough.
    assert len(calls) == 1
    assert body["probed"] is False
    assert "last scan" in body["detail"]
    assert len(body["cameras"]) == 1


def test_cameras_endpoint_rescans_on_request(client):
    calls = []
    client.state.camera_lister = lambda: calls.append(1) or []
    client.get("/api/cameras")
    client.get("/api/cameras?rescan=true")
    assert len(calls) == 2


def test_cameras_endpoint_refuses_to_rescan_while_running(client):
    client.get("/api/cameras")
    calls = []
    client.state.camera_lister = lambda: calls.append(1) or []
    client.state.state = "running"
    client.get("/api/cameras?rescan=true")
    assert calls == []
