import json

import pytest

from app.settings import Settings
from app.settings_store import (
    HOT_RELOADABLE_FIELDS,
    RESTART_REQUIRED_FIELDS,
    _valid_field,
    compute_warnings,
    load_settings,
    resize_guess,
    resolve_resize_mode,
    save_settings,
)


def test_load_settings_missing_file_returns_defaults(tmp_path):
    settings = load_settings(str(tmp_path / "missing.json"))
    assert settings == Settings()


def test_load_settings_corrupt_json_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json", encoding="utf-8")
    settings = load_settings(str(path))
    assert settings == Settings()


def test_load_settings_overlays_valid_fields(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"active_model": "yolo11s.pt", "conf_threshold": 0.7}), encoding="utf-8")
    settings = load_settings(str(path))
    assert settings.active_model == "yolo11s.pt"
    assert settings.conf_threshold == 0.7
    assert settings.capture_width == 640  # untouched fields keep defaults


def test_load_settings_falls_back_per_field_on_invalid_value(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"conf_threshold": "oops", "active_model": "yolo11m.pt"}), encoding="utf-8"
    )
    settings = load_settings(str(path))
    assert settings.conf_threshold == 0.5  # invalid -> default, not a crash
    assert settings.active_model == "yolo11m.pt"  # valid field still applied


def test_imgsz_is_restart_required():
    assert "imgsz" in RESTART_REQUIRED_FIELDS


def test_valid_field_imgsz_accepts_stride_multiples():
    assert _valid_field("imgsz", 640)
    assert _valid_field("imgsz", 960)


def test_valid_field_imgsz_rejects_non_stride_and_out_of_range():
    assert not _valid_field("imgsz", 641)  # not a multiple of 32
    assert not _valid_field("imgsz", 160)  # below the 320 floor
    assert not _valid_field("imgsz", 2048)  # above the 1920 ceiling
    assert not _valid_field("imgsz", 640.0)  # must be an int


def test_compute_warnings_high_imgsz():
    warnings = compute_warnings(Settings(imgsz=1280), "idle")
    assert any("imgsz above 960" in w for w in warnings)


def test_compute_warnings_default_imgsz_no_warning():
    warnings = compute_warnings(Settings(imgsz=640), "idle")
    assert not any("imgsz" in w for w in warnings)


def test_load_settings_ignores_unknown_keys(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"not_a_real_field": 123}), encoding="utf-8")
    settings = load_settings(str(path))
    assert settings == Settings()


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "settings.json"
    original = Settings(active_model="yolo11l.pt", device="cuda", infer_frame_skip=2)
    save_settings(original, str(path))
    loaded = load_settings(str(path))
    assert loaded == original


def test_save_settings_creates_parent_dir(tmp_path):
    path = tmp_path / "a" / "b" / "settings.json"
    save_settings(Settings(), str(path))
    assert path.exists()


def test_compute_warnings_running_locks_restart_fields():
    warnings = compute_warnings(Settings(), "running")
    assert any("stopping capture" in w for w in warnings)


def test_compute_warnings_idle_no_running_warning():
    warnings = compute_warnings(Settings(), "idle")
    assert not any("stopping capture" in w for w in warnings)


def test_compute_warnings_uncommon_resolution():
    warnings = compute_warnings(Settings(capture_width=800, capture_height=600), "idle")
    assert any("resolution" in w for w in warnings)


def test_compute_warnings_common_resolution_no_warning():
    warnings = compute_warnings(Settings(capture_width=1280, capture_height=720), "idle")
    assert not any("resolution" in w for w in warnings)


def test_compute_warnings_frame_skip_vs_expiry():
    settings = Settings(infer_frame_skip=10, capture_fps=15, track_expiry_s=1.0)
    warnings = compute_warnings(settings, "idle")
    assert any("infer_frame_skip" in w for w in warnings)


def test_compute_warnings_low_frame_skip_no_expiry_warning():
    settings = Settings(infer_frame_skip=0, capture_fps=60, track_expiry_s=1.5)
    warnings = compute_warnings(settings, "idle")
    assert not any("infer_frame_skip" in w for w in warnings)


def test_compute_warnings_experimental_model():
    warnings = compute_warnings(Settings(active_model="yolo26n.pt"), "idle")
    assert any("experimental" in w for w in warnings)


def test_compute_warnings_supported_model_no_experimental_warning():
    warnings = compute_warnings(Settings(), "idle")
    assert not any("experimental" in w for w in warnings)


def test_save_then_load_round_trips_experimental_model(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(Settings(active_model="yolo26m.pt"), str(path))
    assert load_settings(str(path)).active_model == "yolo26m.pt"


# --- resolve_resize_mode -------------------------------------------------


def test_auto_resolves_custom_onnx_to_stretch():
    """A Roboflow export records "Stretch to" preprocessing."""
    assert resolve_resize_mode("auto", "models/scanncart-grocery.onnx") == "stretch"


def test_auto_resolves_custom_pt_to_letterbox():
    """A locally trained .pt is letterbox-trained; stretching it would shrink
    objects below their training scale. Design doc 2026-09-04 §C."""
    assert resolve_resize_mode("auto", "models/scanncart-grocery.pt") == "letterbox"


def test_auto_resolves_stock_weights_to_letterbox():
    assert resolve_resize_mode("auto", "yolo11n.pt") == "letterbox"


def test_explicit_modes_win_over_auto():
    assert resolve_resize_mode("letterbox", "models/scanncart-grocery.onnx") == "letterbox"
    assert resolve_resize_mode("stretch", "yolo11n.pt") == "stretch"


def test_a_recorded_requirement_beats_the_format_heuristic():
    """The whole reason the record exists. A `.pt` trained on a Roboflow `Stretch to` version
    needs stretch, and the heuristic - which knows only that local training letterboxes -
    answers the opposite. Without this, leaving `resize_mode` on its default presents every
    object at 0.56x the canvas the model trained on: no error, only weaker `far` detections.
    """
    assert resolve_resize_mode("auto", "models/scanncart-grocery-v2.pt", "stretch") == "stretch"
    # And the reverse, so the record is consulted rather than stretch being special-cased.
    assert resolve_resize_mode("auto", "models/legacy.onnx", "letterbox") == "letterbox"


def test_an_explicit_mode_still_beats_a_recorded_requirement():
    """Order of authority: the operator is allowed to override, and the panel is what tells
    them they are contradicting the weights. Silently ignoring the setting would be worse
    than a wrong one - there would be no way to run a model knowingly off-spec."""
    assert resolve_resize_mode("letterbox", "models/x.pt", "stretch") == "letterbox"


def test_without_a_record_the_heuristic_still_answers():
    """`None` is the normal case for a hand-copied weight and for every stock model, so this
    path stays load-bearing."""
    assert resolve_resize_mode("auto", "models/scanncart-grocery-v2.pt") == "letterbox"
    assert resolve_resize_mode("auto", "models/scanncart-grocery.onnx", None) == "stretch"


def test_a_nonsense_requirement_is_ignored_rather_than_returned():
    """`read_record` already drops a mode the settings PATCH would reject, but a value that
    reached here another way must not become the mode frames are fitted to."""
    assert resolve_resize_mode("auto", "models/x.pt", "crop") == "letterbox"
    # `auto` is not an answer, it is the question.
    assert resolve_resize_mode("auto", "models/x.pt", "auto") == "letterbox"


# --- native onnx CPU-fallback warning ------------------------------------


def test_onnx_on_cuda_without_the_gpu_runtime_warns(monkeypatch):
    """device resolves via torch, but a .onnx runs through onnxruntime; with
    the CPU wheel installed (the requirements.txt default) inference silently
    falls back to CPU with only an ultralytics log line."""
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: False)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.onnx",
                 device="cuda"),
        "idle",
    )
    assert any("onnxruntime-gpu" in w for w in warnings)


def test_onnx_on_cuda_with_the_gpu_runtime_does_not_warn(monkeypatch):
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: True)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.onnx",
                 device="cuda"),
        "idle",
    )
    assert not any("onnxruntime-gpu" in w for w in warnings)


def test_onnx_on_cpu_never_warns(monkeypatch):
    """CPU is a deliberate choice, not a fallback."""
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: False)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.onnx",
                 device="cpu"),
        "idle",
    )
    assert not any("onnxruntime-gpu" in w for w in warnings)


def test_a_pt_model_never_triggers_the_onnx_warning(monkeypatch):
    """A .pt runs on torch, whose CUDA support is independent of onnxruntime."""
    monkeypatch.setattr("app.settings_store._cuda_provider_available", lambda: False)
    warnings = compute_warnings(
        Settings(detector_backend="native", active_model="models/scanncart-grocery.pt",
                 device="cuda"),
        "idle",
    )
    assert not any("onnxruntime-gpu" in w for w in warnings)


# --- custom .pt + stretch warning ----------------------------------------


def test_custom_pt_with_explicit_stretch_warns():
    warnings = compute_warnings(
        Settings(active_model="models/scanncart-grocery.pt", resize_mode="stretch"), "idle"
    )
    assert any("letterbox-trained" in w for w in warnings)


def test_custom_pt_with_auto_does_not_trigger_the_stretch_warning():
    """`auto` never forces a stretch, so there is nothing for this warning to be about - it is
    `resize_guess` below, a structured entry rather than a line in this list, that covers the
    `auto` case."""
    settings = Settings(active_model="models/scanncart-grocery.pt", resize_mode="auto")
    assert not any("letterbox-trained" in w for w in compute_warnings(settings, "idle"))
    assert resize_guess(settings) is not None


def test_a_recorded_stretch_requirement_silences_the_letterbox_warning():
    """Otherwise the feature contradicts itself: `--install` writes `stretch` beside the
    weights, the operator configures exactly that, and the app answers by calling it a
    mistake and telling them to undo the one thing that makes the model usable."""
    settings = Settings(active_model="models/scanncart-grocery-v2.pt", resize_mode="stretch")
    warnings = compute_warnings(settings, "idle", None, "stretch")
    assert not any("letterbox-trained" in w for w in warnings)
    # And through `auto` too - the default path, which is the one that must be right.
    warnings = compute_warnings(
        Settings(active_model="models/scanncart-grocery-v2.pt", resize_mode="auto"),
        "idle",
        None,
        "stretch",
    )
    assert not any("letterbox-trained" in w for w in warnings)


def test_a_recorded_letterbox_requirement_still_warns_on_a_forced_stretch():
    """The record is consulted, not merely trusted to silence: it says letterbox, so stretch
    is still the operator overriding the weights."""
    warnings = compute_warnings(
        Settings(active_model="models/x.pt", resize_mode="stretch"), "idle", None, "letterbox"
    )
    assert any("letterbox-trained" in w for w in warnings)


def test_custom_onnx_with_stretch_does_not_warn():
    """Stretch is exactly right for a Roboflow export."""
    warnings = compute_warnings(
        Settings(active_model="models/scanncart-grocery.onnx", resize_mode="stretch"),
        "idle",
    )
    assert not any("letterbox-trained" in w for w in warnings)


def test_stock_weights_never_warn_about_stretch():
    warnings = compute_warnings(
        Settings(active_model="yolo11n.pt", resize_mode="stretch"), "idle"
    )
    assert not any("letterbox-trained" in w for w in warnings)


# --- unrecorded .pt: the assumed geometry, and the remedy beside it -------
#
# `auto` + an unrecorded custom `.pt` lands on letterbox because that is the usual case, not
# because anything knows it here. Nothing could flag it as *wrong* — no setting fixes a missing
# record — so the panel's comparison stays silent, and the guess would be reported nowhere.
#
# It is reported by `resize_guess`, which is not a warning string: the entry carries the mode a
# one-click record would write, because this is the one resize case the app itself can settle.


def test_unrecorded_custom_pt_on_auto_reports_the_assumed_geometry():
    guess = resize_guess(
        Settings(active_model="models/scanncart-grocery.pt", resize_mode="auto")
    )
    assert guess is not None
    assert guess.mode == "letterbox"


def test_the_guess_names_what_it_assumes_and_the_command_that_removes_it():
    """The sentences travel with the mode because both views render them beside a control that
    writes that mode. Both remedies are named: the tool's (`--install`, which records the fact
    from the training run) and the operator's."""
    guess = resize_guess(
        Settings(active_model="models/scanncart-grocery.pt", resize_mode="auto")
    )
    assert "assumes letterbox" in guess.warning
    assert "has no record of the geometry" in guess.warning
    assert "--install" in guess.remedy


def test_the_diagnosis_and_the_remedy_are_different_sentences():
    """Two views render this entry and only one of them has the button, so the prose is split
    where the views diverge: the diagnosis is true wherever it is read, and the remedy has to be
    usable in a view that cannot perform it.

    The assertion that matters is the negative one. Prose addressed to "below" is prose that only
    works in the view with the button under it — the Live view, which is where the weak `far`
    detections are actually being watched, would be pointing at a control it does not have.
    """
    guess = resize_guess(
        Settings(active_model="models/scanncart-grocery.pt", resize_mode="auto")
    )
    assert guess.remedy not in guess.warning
    assert "Record it below" not in guess.warning
    assert "below" not in guess.warning
    assert "below" not in guess.remedy
    # And the remedy names where the other route is, since the view that reads it may not have a
    # button at all.
    assert "Admin Panel" in guess.remedy


def test_compute_warnings_does_not_repeat_the_guess():
    """One situation, one place on screen. The panel renders `resize_guess` with a remedy, so a
    copy in the flat list would be the same warning twice - and the copy without a fix is the one
    an operator would read first."""
    settings = Settings(active_model="models/scanncart-grocery.pt", resize_mode="auto")
    assert resize_guess(settings) is not None
    assert not any("no record of the geometry" in w for w in compute_warnings(settings, "idle"))


def test_a_recorded_requirement_silences_the_assumed_geometry_warning():
    """A record means there is no assumption to report, whichever mode it names."""
    for required in ("letterbox", "stretch"):
        guess = resize_guess(
            Settings(active_model="models/scanncart-grocery-v2.pt", resize_mode="auto"),
            required,
        )
        assert guess is None, required


def test_an_explicit_letterbox_is_a_decision_not_an_assumption():
    guess = resize_guess(
        Settings(active_model="models/scanncart-grocery.pt", resize_mode="letterbox")
    )
    assert guess is None


def test_a_remote_backend_never_reports_an_assumed_geometry():
    """The field decides nothing for a workflow holding its own model, so an entry naming it
    would send the operator to change a setting that is not in the inference path."""
    for backend in ("local_api", "cloud_api"):
        guess = resize_guess(
            Settings(
                detector_backend=backend,
                active_model="models/scanncart-grocery.pt",
                resize_mode="auto",
            )
        )
        assert guess is None, backend


def test_stock_weights_do_not_report_an_assumed_geometry():
    """Letterbox is not a guess for `yolo11n.pt` — it is what the weights were trained with,
    and there is no record to be missing."""
    assert resize_guess(Settings(active_model="yolo11n.pt", resize_mode="auto")) is None


def test_a_custom_onnx_does_not_report_an_assumed_geometry():
    """The `.onnx` heuristic lands on stretch, so this letterbox entry is not about it; a
    Roboflow export is a known shape rather than an unrecorded checkpoint."""
    assert (
        resize_guess(Settings(active_model="models/scanncart-grocery.onnx", resize_mode="auto"))
        is None
    )


def test_valid_field_class_allowlist():
    assert _valid_field("class_allowlist", [])
    assert _valid_field("class_allowlist", ["bottle", "cup"])
    assert not _valid_field("class_allowlist", "bottle")  # must be a list
    assert not _valid_field("class_allowlist", ["bottle", ""])
    assert not _valid_field("class_allowlist", ["bottle", 3])


def test_class_allowlist_is_hot_reloadable():
    assert "class_allowlist" in HOT_RELOADABLE_FIELDS


def test_valid_field_preview_mirror():
    """Only a real bool. `isinstance(True, int)` is True in Python, so a numeric branch would
    have let 1 and 0 through from a hand-edited file and stood in for the two states."""
    assert _valid_field("preview_mirror", True)
    assert _valid_field("preview_mirror", False)
    assert not _valid_field("preview_mirror", 1)
    assert not _valid_field("preview_mirror", 0)
    assert not _valid_field("preview_mirror", "true")
    assert not _valid_field("preview_mirror", None)


def test_preview_mirror_is_hot_reloadable():
    """It is read at each emit, so the checkbox has to apply without stopping capture."""
    assert "preview_mirror" in HOT_RELOADABLE_FIELDS
    assert "preview_mirror" not in RESTART_REQUIRED_FIELDS


def test_save_then_load_round_trips_preview_mirror(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(Settings(preview_mirror=False), str(path))
    assert load_settings(str(path)).preview_mirror is False


def test_valid_field_suppress_clamped_detections():
    assert _valid_field("suppress_clamped_detections", True)
    assert _valid_field("suppress_clamped_detections", False)
    assert not _valid_field("suppress_clamped_detections", 1)
    assert not _valid_field("suppress_clamped_detections", "false")
    assert not _valid_field("suppress_clamped_detections", None)


def test_suppress_clamped_detections_is_hot_reloadable():
    """It is read per inference, so an operator watching a suppressed item can turn it off and see
    that item on the next frame - not after a stop and start."""
    assert "suppress_clamped_detections" in HOT_RELOADABLE_FIELDS
    assert "suppress_clamped_detections" not in RESTART_REQUIRED_FIELDS


def test_save_then_load_round_trips_suppress_clamped_detections(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(Settings(suppress_clamped_detections=False), str(path))
    assert load_settings(str(path)).suppress_clamped_detections is False


def test_save_then_load_round_trips_class_allowlist(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(Settings(class_allowlist=["bottle", "cup"]), str(path))
    assert load_settings(str(path)).class_allowlist == ["bottle", "cup"]
