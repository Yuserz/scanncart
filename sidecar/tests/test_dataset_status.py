"""The labeling-progress snapshot: reading it, degrading safely, and staying in sync.

The whole point of this feature is that the sidecar does NOT call Roboflow, so the
tests here must never need network or an API key - they write a snapshot file, exactly
as the tool would, and assert the sidecar reads it.

There is also a cross-toolchain drift guard at the bottom. `app/dataset_status.py`
keeps its own copy of the workspace path on purpose (a packaged sidecar does not ship
`sidecar/tools/`, so app code must not import it), and this is what keeps the copy from
rotting.
"""

import datetime
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import label_classes  # from sidecar/tools, on pythonpath via pytest.ini
import label_progress  # from sidecar/tools, on pythonpath via pytest.ini
import workspace  # from sidecar/tools, on pythonpath via pytest.ini
from app.dataset_status import DISTANCES, SNAPSHOT_PATH, load_dataset_status
from app.main import AppState, build_app
from app.settings import Settings


def _snapshot(**over) -> dict:
    data = {
        "project": "snc-grocery",
        "generated_at": "2026-09-22T04:35:09",
        "total": 1383,
        "decided": 40,
        "null_annotations": 3,
        "percent": 2.9,
        "mismatches": 1,
        "by_class": {"milo": [10, 142], "safeguard": [30, 279]},
        "by_cell": {"milo|close": [5, 65], "milo|mid": [5, 34], "milo|far": [0, 43]},
        "by_split": {"train": [30, 969], "valid": [7, 278], "test": [3, 136]},
        "sessions": [
            {"name": "s1", "train": 969, "valid": 278, "test": 136, "decided": 0, "total": 1383, "splits": 3},
            {"name": "neg1", "train": 50, "valid": 0, "test": 0, "decided": 0, "total": 50, "splits": 1},
        ],
        "classes": {"milo": "Milo Chocolate Drink 22g Sachet", "safeguard": "safeguard_pure_white_60g"},
        "tier_a": {
            "target": 253,
            "remaining": 186,
            "cells_under_target": 2,
            "cells": [
                {"slug": "century-tuna", "distance": "far", "target": 40, "have": 0, "decided": 0, "remaining": 40},
                {"slug": "milo", "distance": "mid", "target": 27, "have": 13, "decided": 0, "remaining": 14},
            ],
        },
    }
    data.update(over)
    return data


def _write(tmp_path: Path, data) -> Path:
    path = tmp_path / "label_progress.json"
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def _client() -> TestClient:
    return TestClient(build_app(lambda: AppState(settings=Settings(), db_path=":memory:")))


# --------------------------------------------------------------------------
# absent / unreadable snapshots are a state, not an error
# --------------------------------------------------------------------------


def test_a_missing_snapshot_is_unavailable_not_an_exception(tmp_path):
    status = load_dataset_status(tmp_path / "nope.json")
    assert status.available is False
    assert status.total == 0
    assert status.snapshot_path.endswith("nope.json")


def test_a_corrupt_snapshot_reads_as_unavailable(tmp_path):
    """The tool rewrites this file; a half-written one must not take the panel down."""
    path = _write(tmp_path, "{not json")
    assert load_dataset_status(path).available is False


def test_a_snapshot_of_the_wrong_shape_reads_as_unavailable(tmp_path):
    path = _write(tmp_path, [1, 2, 3])
    assert load_dataset_status(path).available is False


# --------------------------------------------------------------------------
# a real snapshot
# --------------------------------------------------------------------------


def test_a_snapshot_exposes_totals_and_staleness(tmp_path):
    status = load_dataset_status(_write(tmp_path, _snapshot()))
    assert status.available is True
    assert (status.total, status.decided, status.percent) == (1383, 40, 2.9)
    assert status.null_annotations == 3
    assert status.mismatches == 1
    assert status.project == "snc-grocery"
    # Freshness is part of the payload, because the numbers are only as new as the run.
    assert status.generated_at == "2026-09-22T04:35:09"
    assert status.age_seconds is not None
    assert status.age_seconds >= 0


def test_the_age_runs_forwards(tmp_path):
    """An aged stamp has to read as an age, not as freshness.

    Subtracting the other way round is invisible under `max(0, ...)`: every stamp in the
    past clamps to 0, so the panel calls a three-hour-old snapshot current. The stamp is
    built from the clock rather than written as a literal, because a literal would age
    into exactly the case this is testing.
    """
    then = datetime.datetime.now() - datetime.timedelta(hours=3)
    status = load_dataset_status(
        _write(tmp_path, _snapshot(generated_at=then.strftime("%Y-%m-%dT%H:%M:%S")))
    )
    assert status.age_seconds is not None
    assert 3 * 3600 - 60 <= status.age_seconds <= 3 * 3600 + 60


def test_a_stamp_ahead_of_the_clock_clamps_to_zero_rather_than_going_negative(tmp_path):
    """The rule that makes the clamp above worth having, tested from the other side."""
    ahead = datetime.datetime.now() + datetime.timedelta(minutes=5)
    status = load_dataset_status(
        _write(tmp_path, _snapshot(generated_at=ahead.strftime("%Y-%m-%dT%H:%M:%S")))
    )
    assert status.age_seconds == 0


def test_an_unparseable_timestamp_gives_age_none_rather_than_a_wrong_number(tmp_path):
    status = load_dataset_status(_write(tmp_path, _snapshot(generated_at="whenever")))
    assert status.available is True
    assert status.age_seconds is None


def test_classes_carry_their_display_names_and_sort_by_them(tmp_path):
    """The roster travels in the snapshot so the sidecar needs no copy of it."""
    status = load_dataset_status(_write(tmp_path, _snapshot()))
    assert [c["name"] for c in status.classes] == [
        "Milo Chocolate Drink 22g Sachet",
        "safeguard_pure_white_60g",
    ]
    assert status.classes[0] == {"slug": "milo", "name": "Milo Chocolate Drink 22g Sachet", "decided": 10, "total": 142}


def test_the_distance_axis_is_derived_from_the_cells(tmp_path):
    """The renderer stays dumb: by_cell is keyed `<slug>|<distance>` and is summed here."""
    status = load_dataset_status(_write(tmp_path, _snapshot()))
    assert status.by_distance["close"] == [5, 65]
    assert status.by_distance["mid"] == [5, 34]
    assert status.by_distance["far"] == [0, 43]
    assert set(status.by_distance) == set(DISTANCES)


def test_unexpected_shapes_normalise_to_zero_rather_than_raising(tmp_path):
    """A snapshot is written by a different program, so a shape change there must not be
    able to break the app."""
    data = _snapshot(by_class={"milo": "oops", "x": [1]}, by_cell={"milo|close": 7, "junk": [1, 2]})
    status = load_dataset_status(_write(tmp_path, data))
    assert status.available is True
    assert all(c["total"] == 0 for c in status.classes)
    assert status.by_distance["close"] == [0, 0]


# --------------------------------------------------------------------------
# Tier A: the capture gap, which is not the labeling gap
# --------------------------------------------------------------------------


def test_tier_a_carries_the_capture_gap(tmp_path):
    status = load_dataset_status(_write(tmp_path, _snapshot()))
    assert status.tier_a_target == 253
    assert status.tier_a_remaining == 186
    assert status.tier_a_cells_under_target == 2
    assert status.tier_a_cells[0] == {
        "slug": "century-tuna",
        "distance": "far",
        "target": 40,
        "have": 0,
        "decided": 0,
        "remaining": 40,
    }


def test_a_snapshot_without_tier_a_reads_as_no_gap_rather_than_failing(tmp_path):
    """An older snapshot is the ordinary case on a machine that has not re-run the tool,
    so the panel must fall quiet instead of erroring or inventing a 0-of-0 gap."""
    data = _snapshot()
    del data["tier_a"]
    status = load_dataset_status(_write(tmp_path, data))
    assert status.available is True
    assert (status.tier_a_target, status.tier_a_remaining) == (0, 0)
    assert status.tier_a_cells == []


def test_tier_a_normalises_junk_cells_instead_of_raising(tmp_path):
    """Same reasoning as `by_class`: another program writes this file."""
    data = _snapshot(
        tier_a={
            "target": "lots",
            "cells": [
                "not a cell",
                {"slug": "milo", "distance": "sideways", "target": 5},
                {"slug": "milo", "distance": "mid", "target": "5", "have": None},
            ],
        }
    )
    status = load_dataset_status(_write(tmp_path, data))
    assert status.available is True
    assert status.tier_a_target == 0  # "lots" is not a count, and not a crash either
    # The sideways distance is dropped - the renderer keys on close/mid/far.
    assert len(status.tier_a_cells) == 1
    assert status.tier_a_cells[0]["target"] == 5  # "5" is a count, so it survives
    assert status.tier_a_cells[0]["have"] == 0  # None is not


# --------------------------------------------------------------------------
# the capture-session spread
# --------------------------------------------------------------------------


def test_the_session_spread_reads_back_with_its_split_counts(tmp_path):
    status = load_dataset_status(_write(tmp_path, _snapshot()))
    assert [s["name"] for s in status.sessions] == ["s1", "neg1"]
    assert status.sessions[0]["train"] == 969
    assert status.sessions[0]["valid"] == 278
    assert status.sessions[0]["test"] == 136
    assert status.sessions[0]["splits"] == 3


def test_the_split_count_is_derived_from_the_counts_not_taken_from_the_file(tmp_path):
    """The flag the panel colours on must not be able to disagree with the three numbers
    printed beside it, so it is recomputed here. A file claiming 1 split while holding
    frames in three is exactly the kind of stale number this reader exists to catch."""
    data = _snapshot(
        sessions=[
            {"name": "s1", "train": 969, "valid": 278, "test": 136, "total": 1383, "splits": 1}
        ]
    )
    assert load_dataset_status(_write(tmp_path, data)).sessions[0]["splits"] == 3


def test_a_snapshot_without_sessions_reads_as_empty(tmp_path):
    """Written by a tool older than this field - the ordinary case on a machine that has
    not re-run it, so it reads as absent rather than as an error."""
    data = _snapshot()
    del data["sessions"]
    status = load_dataset_status(_write(tmp_path, data))
    assert status.available is True
    assert status.sessions == []


def test_a_malformed_session_row_is_dropped_not_crashed(tmp_path):
    data = _snapshot(
        sessions=[
            "not a row",
            {"train": 10},  # no name
            {"name": "", "train": 10},  # an empty name is not a session
            {"name": "s2", "train": "many", "valid": None, "test": 4},
        ]
    )
    sessions = load_dataset_status(_write(tmp_path, data)).sessions
    assert len(sessions) == 1
    assert sessions[0]["name"] == "s2"
    assert sessions[0]["train"] == 0  # "many" is not a count, and not a crash either
    assert sessions[0]["splits"] == 1  # only `test` has images


def test_the_sidecar_reads_what_the_tool_writes(tmp_path):
    """Cross-toolchain drift guard, offline and end to end.

    The tool's own projection goes into the sidecar's reader, so a renamed key or a
    changed shape fails here instead of silently rendering an empty table - the failure
    mode CLAUDE.md calls out for this repo's hand-synced contracts. `session_rows` is
    pure precisely so this needs no project, no network and no snapshot on disk.
    """
    rows = label_progress.session_rows(
        {
            "s1": {"train": [0, 969], "valid": [0, 278], "test": [0, 136]},
            "neg1": {"train": [0, 50]},
            "": {"train": [0, 5]},
        }
    )
    status = load_dataset_status(_write(tmp_path, _snapshot(sessions=rows)))
    assert [s["name"] for s in status.sessions] == ["(no session tag)", "neg1", "s1"]
    assert status.sessions[2]["train"] == 969 and status.sessions[2]["splits"] == 3
    assert status.sessions[0]["splits"] == 1


# --------------------------------------------------------------------------
# the backlog: the same counts, ordered as work
# --------------------------------------------------------------------------


def test_the_backlog_is_what_is_left_biggest_first(tmp_path):
    """A worklist, not a table: the row at the top is the one to label next."""
    status = load_dataset_status(_write(tmp_path, _snapshot()))
    assert [(r["slug"], r["distance"], r["remaining"]) for r in status.labeling_backlog] == [
        ("milo", "close", 60),
        ("milo", "far", 43),
        ("milo", "mid", 29),
    ]


def test_a_finished_cell_leaves_the_worklist(tmp_path):
    """242/242 is not work. Leaving it in would push the actual work off the screen as a
    labeling session gets closer to done - the opposite of what a worklist is for."""
    data = _snapshot(by_cell={"milo|close": [65, 65], "milo|far": [0, 43]})
    status = load_dataset_status(_write(tmp_path, data))
    assert [r["distance"] for r in status.labeling_backlog] == ["far"]


def test_the_backlog_order_is_stable_between_runs(tmp_path):
    """Equal-sized cells must not swap places on every refresh: whoever is working down the
    list would lose their place. Ties break on name, then distance."""
    first = _snapshot(by_cell={"b|close": [0, 10], "a|far": [0, 10], "a|close": [0, 10]})
    second = _snapshot(by_cell={"a|close": [0, 10], "b|close": [0, 10], "a|far": [0, 10]})
    order = lambda data: [(r["name"], r["distance"]) for r in load_dataset_status(_write(tmp_path, data)).labeling_backlog]  # noqa: E731
    assert order(first) == order(second) == [("a", "close"), ("a", "far"), ("b", "close")]


def test_the_background_frames_are_a_row_like_any_other(tmp_path):
    """They have no cell - no distance, and no box to draw - but dropping them would be the
    one omission with a cost, because an unmarked frame never enters a version."""
    status = load_dataset_status(_write(tmp_path, _snapshot(by_class={"milo": [10, 142], "negative": [0, 50]})))
    row = next(r for r in status.labeling_backlog if r["slug"] == "negative")
    assert row["background"] is True
    assert row["distance"] == ""  # empty, not a guess: these frames are staged without one
    assert row["remaining"] == 50


def test_a_backlog_row_never_reports_negative_work(tmp_path):
    """A snapshot claiming more decided than total is nonsense; it must not sort first with a
    negative remainder."""
    data = _snapshot(by_cell={"milo|close": [99, 65], "milo|mid": [1, 5]})
    status = load_dataset_status(_write(tmp_path, data))
    # The over-counted cell reads as finished, not as -34 work that would sort it to the top;
    # the cell with real work is then the only row, and it still carries its own count.
    assert [(r["distance"], r["remaining"]) for r in status.labeling_backlog] == [("mid", 4)]


def test_a_snapshot_without_cells_has_an_empty_backlog(tmp_path):
    """An older snapshot, one a machine has not re-run the tool to produce, reads as "nothing
    to work through" rather than as an error."""
    status = load_dataset_status(_write(tmp_path, _snapshot(by_cell={}, by_class={})))
    assert status.available is True
    assert status.labeling_backlog == []


# --------------------------------------------------------------------------
# the route
# --------------------------------------------------------------------------


def test_the_route_reports_unavailable_without_a_snapshot(monkeypatch, tmp_path):
    """A 200, because an unrun tool is the state this panel starts in — not an error
    the renderer would have to special-case."""
    monkeypatch.setattr("app.dataset_status.SNAPSHOT_PATH", tmp_path / "none.json")
    r = _client().get("/api/dataset/status")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["classes"] == []


def test_the_route_serves_a_snapshot_and_needs_no_network(monkeypatch, tmp_path):
    monkeypatch.setattr("app.dataset_status.SNAPSHOT_PATH", _write(tmp_path, _snapshot()))
    body = _client().get("/api/dataset/status").json()
    assert body["available"] is True
    assert body["total"] == 1383
    assert body["by_split"]["valid"] == [7, 278]
    assert body["by_distance"]["far"] == [0, 43]
    assert [c["slug"] for c in body["classes"]] == ["milo", "safeguard"]
    # The capture gap rides along, so the panel can separate "shoot this" from "label this".
    assert body["tier_a_target"] == 253
    assert body["tier_a_remaining"] == 186
    assert [c["slug"] for c in body["tier_a_cells"]] == ["century-tuna", "milo"]
    # And the session spread, which is how the panel can say whether the eventual test
    # number is an unseen session or held-out frames of one the model trained on.
    assert [s["name"] for s in body["sessions"]] == ["s1", "neg1"]
    assert body["sessions"][0]["splits"] == 3
    assert body["sessions"][1]["splits"] == 1
    # And the worklist, which is the same counts arranged by the question a labeling session asks.
    assert body["labeling_backlog"][0]["slug"] == "milo"
    assert body["labeling_backlog"][0]["distance"] == "close"
    assert body["labeling_backlog"][0]["remaining"] == 60


def test_the_route_does_not_expose_a_roboflow_key(monkeypatch, tmp_path):
    """The snapshot is on disk and could carry anything; the response is a fixed shape,
    so a key accidentally written into it cannot leak through this endpoint."""
    monkeypatch.setattr(
        "app.dataset_status.SNAPSHOT_PATH",
        _write(tmp_path, _snapshot(api_key="secret-key-value", roboflow_api_key="secret-key-value")),
    )
    body = _client().get("/api/dataset/status").text
    assert "secret-key-value" not in body


# --------------------------------------------------------------------------
# drift guard: the sidecar's copy of the workspace path
# --------------------------------------------------------------------------


def test_the_snapshot_path_matches_where_the_tool_writes_it():
    """Fails the moment the workspace moves without the sidecar following.

    This is the repo's standing risk (CLAUDE.md: hand-synced contracts) made checkable.
    `app/dataset_status.py` deliberately does not import `sidecar/tools/workspace.py`,
    so nothing but this test stops the two from diverging.
    """
    expected = workspace.DEFAULT_OUT / "label_progress.json"
    assert Path(SNAPSHOT_PATH) == expected, (
        "the sidecar and sidecar/tools/workspace.py disagree about where the snapshot lives"
    )


def test_the_background_slug_list_matches_the_tooling():
    """The second hand-synced copy in that module, guarded the same way.

    `BACKGROUND_SLUGS` decides which rows the worklist flags as "mark null, do not draw". If
    the tooling ever renames `negative`, the sidecar would stop flagging it - and the row would
    read as an ordinary cell, which is the one instruction a labeler must not follow here.
    """
    from app.dataset_status import BACKGROUND_SLUGS

    assert set(BACKGROUND_SLUGS) == set(label_classes.PSEUDO_CLASS_SLUGS)
