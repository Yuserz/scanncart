"""Settings-layer coverage for the pluggable detector backends.

See docs/DETECTOR_BACKENDS.md. Phase 2 only — the detector implementations
themselves land in Phase 4.
"""

import pytest
from pydantic import ValidationError

from app.schemas import SettingsResponse, SettingsUpdateRequest
from app.settings import Settings
from app.settings_store import (
    ALLOWED_BACKENDS,
    REMOTE_BACKENDS,
    RESTART_REQUIRED_FIELDS,
    _valid_field,
    compute_warnings,
    load_settings,
    save_settings,
)

BACKEND_FIELDS = [
    "detector_backend",
    "roboflow_workspace",
    "roboflow_workflow_id",
    "local_api_url",
    "cloud_api_url",
    "remote_infer_size",
    "remote_timeout_s",
    "remote_max_retries",
]


# --- defaults ------------------------------------------------------------


def test_defaults_to_native_backend():
    assert Settings().detector_backend == "native"


def test_default_backend_is_allowed():
    assert Settings().detector_backend in ALLOWED_BACKENDS


def test_remote_backends_are_a_subset_of_allowed():
    assert REMOTE_BACKENDS < ALLOWED_BACKENDS
    assert "native" not in REMOTE_BACKENDS


def test_default_infer_size_matches_yolo_imgsz():
    # The model resizes to 640 internally; sending more is wasted bandwidth.
    assert Settings().remote_infer_size == 640


@pytest.mark.parametrize("name", BACKEND_FIELDS)
def test_every_backend_field_requires_restart(name):
    """All eight are baked into the detector at capture-start, so none can be
    hot-swapped on a running pipeline."""
    assert name in RESTART_REQUIRED_FIELDS


# --- field validation ----------------------------------------------------


@pytest.mark.parametrize("backend", sorted(ALLOWED_BACKENDS))
def test_valid_backends_accepted(backend):
    assert _valid_field("detector_backend", backend)


@pytest.mark.parametrize("bad", ["", "cloud", "NATIVE", None, 1, True])
def test_invalid_backends_rejected(bad):
    assert not _valid_field("detector_backend", bad)


@pytest.mark.parametrize("field", ["roboflow_workspace", "roboflow_workflow_id"])
@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_blank_identifiers_rejected(field, bad):
    assert not _valid_field(field, bad)


@pytest.mark.parametrize("field", ["local_api_url", "cloud_api_url"])
@pytest.mark.parametrize("good", ["http://127.0.0.1:9001", "https://serverless.roboflow.com"])
def test_urls_accepted(field, good):
    assert _valid_field(field, good)


@pytest.mark.parametrize("field", ["local_api_url", "cloud_api_url"])
@pytest.mark.parametrize("bad", ["127.0.0.1:9001", "ftp://x", "", None])
def test_urls_without_scheme_rejected(field, bad):
    assert not _valid_field(field, bad)


@pytest.mark.parametrize("good", [128, 640, 1920])
def test_infer_size_in_range_accepted(good):
    assert _valid_field("remote_infer_size", good)


@pytest.mark.parametrize("bad", [127, 1921, 0, -1, 640.0, "640"])
def test_infer_size_out_of_range_rejected(bad):
    assert not _valid_field("remote_infer_size", bad)


@pytest.mark.parametrize("good", [0.1, 5.0, 60.0, 5])
def test_timeout_in_range_accepted(good):
    assert _valid_field("remote_timeout_s", good)


@pytest.mark.parametrize("bad", [0.0, 0.09, 60.1, "5"])
def test_timeout_out_of_range_rejected(bad):
    assert not _valid_field("remote_timeout_s", bad)


@pytest.mark.parametrize("good", [0, 2, 5])
def test_retries_in_range_accepted(good):
    assert _valid_field("remote_max_retries", good)


@pytest.mark.parametrize("bad", [-1, 6, 2.0, "2"])
def test_retries_out_of_range_rejected(bad):
    assert not _valid_field("remote_max_retries", bad)


# --- persistence ---------------------------------------------------------


def test_backend_fields_round_trip(tmp_path):
    path = str(tmp_path / "settings.json")
    s = Settings()
    s.detector_backend = "local_api"
    s.local_api_url = "http://127.0.0.1:9999"
    s.remote_infer_size = 512
    s.remote_timeout_s = 2.5
    s.remote_max_retries = 0
    save_settings(s, path)

    loaded = load_settings(path)
    assert loaded.detector_backend == "local_api"
    assert loaded.local_api_url == "http://127.0.0.1:9999"
    assert loaded.remote_infer_size == 512
    assert loaded.remote_timeout_s == 2.5
    assert loaded.remote_max_retries == 0


def test_corrupt_backend_falls_back_to_default_without_losing_siblings(tmp_path):
    path = str(tmp_path / "settings.json")
    path_obj = tmp_path / "settings.json"
    path_obj.write_text(
        '{"detector_backend": "wormhole", "remote_infer_size": 512}', encoding="utf-8"
    )
    loaded = load_settings(path)
    assert loaded.detector_backend == "native"   # rejected field -> default
    assert loaded.remote_infer_size == 512       # valid sibling still applied


# --- request validation --------------------------------------------------


def test_update_request_accepts_a_valid_backend():
    assert SettingsUpdateRequest(detector_backend="cloud_api").detector_backend == "cloud_api"


def test_update_request_rejects_an_unknown_backend():
    with pytest.raises(ValidationError):
        SettingsUpdateRequest(detector_backend="wormhole")


def test_update_request_rejects_a_schemeless_url():
    with pytest.raises(ValidationError):
        SettingsUpdateRequest(local_api_url="127.0.0.1:9001")


def test_update_request_rejects_blank_workspace():
    with pytest.raises(ValidationError):
        SettingsUpdateRequest(roboflow_workspace="")


def test_update_request_omitting_backend_fields_is_fine():
    assert SettingsUpdateRequest(conf_threshold=0.6).detector_backend is None


# --- the API key must never be serialized --------------------------------


def test_settings_response_has_no_key_field():
    """Only presence is ever exposed; the value must not have a field at all."""
    assert "roboflow_api_key_present" in SettingsResponse.model_fields
    leaky = [
        n for n in SettingsResponse.model_fields
        if "key" in n.lower() and n != "roboflow_api_key_present"
    ]
    assert leaky == []


def test_update_request_cannot_set_an_api_key():
    assert "roboflow_api_key" not in SettingsUpdateRequest.model_fields


def test_settings_dataclass_holds_no_secret():
    assert not [f for f in vars(Settings()) if "key" in f.lower()]


# --- warnings ------------------------------------------------------------


def _warn(backend, api_key_present=True, **over):
    s = Settings()
    s.detector_backend = backend
    for k, v in over.items():
        setattr(s, k, v)
    return " ".join(compute_warnings(s, "idle", api_key_present))


def test_native_backend_warns_about_neither_key_nor_cost():
    text = _warn("native", api_key_present=False)
    assert "ROBOFLOW_API_KEY" not in text
    assert "bills per inference" not in text


@pytest.mark.parametrize("backend", sorted(REMOTE_BACKENDS))
def test_remote_backend_without_key_warns(backend):
    assert "ROBOFLOW_API_KEY" in _warn(backend, api_key_present=False)


@pytest.mark.parametrize("backend", sorted(REMOTE_BACKENDS))
def test_remote_backend_with_key_does_not_warn_about_key(backend):
    assert "ROBOFLOW_API_KEY" not in _warn(backend, api_key_present=True)


def test_cloud_backend_warns_about_cost_and_offline_guarantee():
    text = _warn("cloud_api")
    assert "bills per inference" in text
    assert "offline guarantee" in text


def test_cloud_backend_warns_when_url_is_not_https():
    assert "https" in _warn("cloud_api", cloud_api_url="http://serverless.roboflow.com")


def test_cloud_backend_with_https_has_no_scheme_warning():
    assert "plain http is rejected" not in _warn("cloud_api")


def test_local_backend_names_the_url_to_start_a_server_on():
    text = _warn("local_api", local_api_url="http://127.0.0.1:9001")
    assert "http://127.0.0.1:9001" in text
    assert "inference server start" in text


def test_local_backend_does_not_warn_about_billing():
    assert "bills per inference" not in _warn("local_api")


def test_running_capture_still_locks_the_new_fields():
    warnings = compute_warnings(Settings(), "running", True)
    assert any("detector_backend" in w for w in warnings)


# --- remote latency vs track expiry --------------------------------------


@pytest.mark.parametrize("backend", sorted(REMOTE_BACKENDS))
def test_remote_backend_warns_when_track_expiry_is_too_low(backend):
    """Reproduced live in Phase 0: a 3250 ms round trip expired a stationary
    item under the default 1.5 s and re-issued its track id."""
    assert "too low" in _warn(backend, track_expiry_s=1.5)


@pytest.mark.parametrize("backend", sorted(REMOTE_BACKENDS))
def test_remote_backend_accepts_a_generous_track_expiry(backend):
    assert "too low" not in _warn(backend, track_expiry_s=6.0)


def test_native_backend_is_fine_with_the_default_expiry():
    assert "too low" not in _warn("native", track_expiry_s=1.5)


def test_the_minimum_is_above_observed_cloud_latency():
    from app.settings_store import MIN_REMOTE_TRACK_EXPIRY_S

    assert MIN_REMOTE_TRACK_EXPIRY_S >= 3.25
