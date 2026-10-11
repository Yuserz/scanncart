"""Tests for `replay_scenarios.py`: the scenario corpus' script format and its fixture shape.

Everything here is the *pure* half of the tool — script parsing, video pairing, and the fixture
document — because the other half needs what the corpus itself needs: a recorded video and installed
weights. That half is `make replay-scenarios`, which fails loudly when the data is absent.

What matters to test is narrower than it looks. The failure modes this file guards are the ones that
would make a fixture *pass*: a misspelled `checkpoints` key that reads as "this script asserts
nothing", a checkpoint whose quantity is a string that compares unequal to everything, a script whose
video is missing so its events are simply absent from the corpus. A corpus that silently shrinks is
how a counting-accuracy gate stops meaning anything, so every one of those is a rejection here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import replay_scenarios
from replay_scenarios import Checkpoint, ScenarioScript, ScriptError


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


VALID = {
    "bind_at_s": 1.0,
    "checkpoints": [
        {"t_s": 6.0, "cart": {"safeguard_pure_white_60g": 1}},
        {"t_s": 20.0, "cart": {}},
    ],
}


def test_a_valid_script_parses_into_its_checkpoints(tmp_path: Path) -> None:
    script = replay_scenarios.load_script(_write(tmp_path, "01_one_item", VALID))

    assert script.name == "01_one_item"  # the file stem, when the script does not name itself
    assert script.bind_at_s == 1.0
    assert [c.t_s for c in script.checkpoints] == [6.0, 20.0]
    assert script.checkpoints[0].cart == {"safeguard_pure_white_60g": 1}
    # An empty cart is a real checkpoint — "nothing on the counter" is a claim — so it is kept.
    assert script.checkpoints[1].cart == {}


def test_checkpoints_come_back_in_time_order(tmp_path: Path) -> None:
    payload = dict(VALID, checkpoints=list(reversed(VALID["checkpoints"])))
    script = replay_scenarios.load_script(_write(tmp_path, "ordered", payload))
    assert [c.t_s for c in script.checkpoints] == [6.0, 20.0]


@pytest.mark.parametrize(
    "payload, expected",
    [
        # A typo'd key must not read as "asserts nothing".
        ({"bind_at_s": 1.0, "checkpoint": []}, "unknown key"),
        # No checkpoints at all is the empty-corpus failure, spelled out.
        ({"bind_at_s": 1.0, "checkpoints": []}, "non-empty list"),
        ({"bind_at_s": "start", "checkpoints": []}, "expected a number"),
        ({"bind_at_s": 1.0}, "'checkpoints' must be a non-empty list"),
        (dict(VALID, checkpoints=[{"t_s": "6", "cart": {}}]), "expected a number"),
        (dict(VALID, checkpoints=[{"t_s": 6.0, "carts": {}}]), "unknown key"),
        # 0 is a mistake rather than a state: an absent class is expressed by leaving it out.
        (dict(VALID, checkpoints=[{"t_s": 6.0, "cart": {"soda": 0}}]), "whole number >= 1"),
        (dict(VALID, checkpoints=[{"t_s": 6.0, "cart": {"soda": "one"}}]), "whole number >= 1"),
        # Two answers for one moment is a script that cannot be reported on.
        (
            dict(VALID, checkpoints=[{"t_s": 6.0, "cart": {}}, {"t_s": 6.0, "cart": {}}]),
            "another checkpoint already does",
        ),
        (dict(VALID, asserted_with={"dwell": 3}), "unknown key"),
    ],
)
def test_a_script_that_cannot_be_trusted_is_rejected(tmp_path: Path, payload: dict, expected: str) -> None:
    with pytest.raises(ScriptError) as excinfo:
        replay_scenarios.load_script(_write(tmp_path, "bad", payload))
    assert expected in str(excinfo.value)


def test_a_script_that_is_not_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ScriptError) as excinfo:
        replay_scenarios.load_script(path)
    assert "not valid JSON" in str(excinfo.value)


def test_the_video_is_found_by_stem_whatever_the_container(tmp_path: Path) -> None:
    script = _write(tmp_path, "03_two_identical", VALID)
    assert replay_scenarios.video_for(script) is None  # a script with no recording asserts nothing
    (tmp_path / "03_two_identical.mov").write_bytes(b"")
    found = replay_scenarios.video_for(script)
    assert found is not None and found.name == "03_two_identical.mov"


def test_scripts_are_listed_in_a_fixed_order(tmp_path: Path) -> None:
    for name in ("02_b", "01_a", "03_c"):
        _write(tmp_path, name, VALID)
    assert [p.stem for p in replay_scenarios.scripts_in(tmp_path)] == ["01_a", "02_b", "03_c"]


def test_an_event_is_mapped_to_the_desktops_vocabulary() -> None:
    class Row:
        track_id = 7
        class_name = "safeguard_pure_white_60g"
        max_conf = 0.91
        entered_at = 12.5
        left_at = None

    assert replay_scenarios.event_document(Row(), session_id=1) == {
        "sessionId": 1,
        "trackId": 7,
        "className": "safeguard_pure_white_60g",
        "maxConf": 0.91,
        "enteredAt": 12.5,
        "leftAt": None,
    }


def test_the_fixture_carries_the_ground_truth_with_the_events(tmp_path: Path) -> None:
    """Both halves travel together: the script lives in the workspace, the fixture is what CI reads."""
    script = replay_scenarios.load_script(_write(tmp_path, "07_stacked", dict(VALID, asserted_with={
        "commit_dwell_s": 3, "remove_settle_s": 10, "min_commit_conf": 0.6,
    })))
    document = replay_scenarios.fixture_document(
        script,
        [{"sessionId": 1, "trackId": 1, "className": "soda", "maxConf": 0.9, "enteredAt": 2.0, "leftAt": None}],
        video=tmp_path / "07_stacked.mp4",
        weights="models/scanncart-grocery.pt",
        device="cpu",
        pipeline_settings={"conf_threshold": 0.5},
        frames=900,
        duration_s=30.0,
    )

    assert document["name"] == "07_stacked"
    assert document["checkpoints"] == [
        {"t_s": 6.0, "cart": {"safeguard_pure_white_60g": 1}},
        {"t_s": 20.0, "cart": {}},
    ]
    assert document["bind_at_s"] == 1.0
    assert document["frames"] == 900
    assert document["pipeline"] == {"conf_threshold": 0.5}
    # The operating point the checkpoints were written against, so a failure message can name it.
    assert document["asserted_with"]["commit_dwell_s"] == 3
    # Round-trips, because the desktop suite reads this file and nothing else.
    assert json.loads(json.dumps(document))["events"][0]["className"] == "soda"


def test_a_fixture_is_written_as_its_own_name(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "scenarios"
    written = replay_scenarios.write_fixture(out, {"name": "01_one_item", "events": []})
    assert written == out / "01_one_item.json"
    assert written.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(written.read_text(encoding="utf-8"))["name"] == "01_one_item"


def test_a_missing_corpus_fails_loudly_before_anything_heavy(tmp_path: Path) -> None:
    """The gate's contract, and the reason these checks precede the cv2/torch imports: a corpus that
    is not there must be reported as itself, not skipped, and not as an ImportError from a tool that
    never got to look."""
    with pytest.raises(SystemExit) as excinfo:
        replay_scenarios.main(["--corpus", str(tmp_path / "nope")])
    assert "no such directory" in str(excinfo.value)


def test_an_empty_corpus_is_not_a_silent_pass(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        replay_scenarios.main(["--corpus", str(tmp_path), "--out", str(tmp_path / "out")])
    assert "holds no scenario scripts" in str(excinfo.value)


def test_a_script_without_its_recording_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "01_one_item", VALID)
    with pytest.raises(SystemExit) as excinfo:
        replay_scenarios.main(["--corpus", str(tmp_path), "--out", str(tmp_path / "out")])
    message = str(excinfo.value)
    assert "no video beside: 01_one_item" in message
    # Nothing was written: a half-corpus is worse than none, because the suite scores what it finds.
    assert not (tmp_path / "out").exists()


BASKET = {
    "bind_at_s": 1.0,
    "basket": {"cart_edge": "bottom", "inside_fraction": 0.35, "opening_fraction": 0.2},
    "checkpoints": [
        {"t_s": 8.0, "cart": {"century_tuna": 1}, "review": 0},
        {"t_s": 16.0, "cart": {}},
    ],
}


def test_a_basket_script_carries_its_zones_and_review_counts(tmp_path: Path) -> None:
    script = replay_scenarios.load_script(_write(tmp_path, "11_deposit_one", BASKET))

    assert script.basket == replay_scenarios.BasketZones("bottom", 0.35, 0.2)
    assert script.checkpoints[0].review == 0
    # Unasserted is not zero: a checkpoint that says nothing about review checks nothing about it.
    assert script.checkpoints[1].review is None
    assert script.initial_cart is None


def test_a_removal_script_starts_from_its_initial_cart(tmp_path: Path) -> None:
    """A removal clip begins with the item already in the basket: without the seed the scorer's
    ledger is empty, and taking an item out is a review rather than a removal."""
    payload = dict(BASKET, initial_cart={"century_tuna": 1})
    script = replay_scenarios.load_script(_write(tmp_path, "12_remove_one", payload))
    assert script.initial_cart == {"century_tuna": 1}
    document = replay_scenarios.fixture_document(
        script, [], video=tmp_path / "12_remove_one.mp4", weights="w.pt", device="cpu",
        pipeline_settings={}, frames=0, duration_s=0.0, stream=[],
    )
    assert document["initial_cart"] == {"century_tuna": 1}


@pytest.mark.parametrize(
    "payload, expected",
    [
        (dict(VALID, initial_cart={"century_tuna": 1}), "only a basket scenario"),
        (dict(BASKET, initial_cart={"century_tuna": 0}), "whole number >= 1"),
        (dict(BASKET, initial_cart=["century_tuna"]), "expected an object"),
    ],
)
def test_an_initial_cart_that_cannot_be_trusted_is_rejected(
    tmp_path: Path, payload: dict, expected: str
) -> None:
    with pytest.raises(ScriptError) as excinfo:
        replay_scenarios.load_script(_write(tmp_path, "bad", payload))
    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    "payload, expected",
    [
        # A review count on a counter scenario would be scored against nothing.
        (dict(VALID, checkpoints=[{"t_s": 6.0, "cart": {}, "review": 0}]), "only a basket scenario"),
        (dict(BASKET, checkpoints=[{"t_s": 6.0, "cart": {}, "review": -1}]), "whole number >= 0"),
        (dict(BASKET, checkpoints=[{"t_s": 6.0, "cart": {}, "review": True}]), "whole number >= 0"),
        (dict(BASKET, basket={"cart_edge": "front", "inside_fraction": 0.3, "opening_fraction": 0.2}), "cart_edge"),
        (dict(BASKET, basket={"cart_edge": "top", "inside_fraction": 0.6, "opening_fraction": 0.4}), "sum to less than 1"),
        (dict(BASKET, basket={"cart_edge": "top", "inside_fraction": 0, "opening_fraction": 0.2}), "in (0, 1)"),
        (dict(BASKET, basket={"cart_edge": "top", "inside": 0.3, "opening_fraction": 0.2}), "unknown key"),
    ],
)
def test_a_basket_script_that_cannot_be_trusted_is_rejected(
    tmp_path: Path, payload: dict, expected: str
) -> None:
    with pytest.raises(ScriptError) as excinfo:
        replay_scenarios.load_script(_write(tmp_path, "bad", payload))
    assert expected in str(excinfo.value)


def test_the_cart_edges_are_the_desktops() -> None:
    """The scorer hands `cart_edge` straight to `presetRegions`, which throws on an edge it does not
    know — read the TypeScript list rather than trusting a copy to stay in step."""
    import re

    source = (replay_scenarios.REPO_ROOT / "desktop" / "src" / "main" / "transferGeometry.ts").read_text(
        encoding="utf-8"
    )
    match = re.search(r"export const CART_EDGES[^=]*=\s*\[([^\]]*)\]", source)
    assert match, "CART_EDGES is no longer a literal array in transferGeometry.ts"
    assert tuple(re.findall(r"'(\w+)'", match.group(1))) == replay_scenarios.CART_EDGES


def test_a_stream_frame_is_on_the_videos_clock_and_drops_fill_ins() -> None:
    message = {
        "type": "frame", "ts": 1_759_000_000.0, "seq": 12, "fresh": True, "mirrored": False,
        "detections": [
            {"track_id": 3, "cls": "century_tuna", "conf": 0.912345, "box": (0.1, 0.212345, 0.3, 0.4)},
        ],
    }
    frame = replay_scenarios.stream_frame(message, 4.56789)
    # The wall-clock `ts` is replaced by the video's: the checkpoints were written on that timeline.
    assert frame == {
        "ts": 4.5679, "seq": 12,
        "detections": [{"track_id": 3, "cls": "century_tuna", "conf": 0.9123, "box": [0.1, 0.2123, 0.3, 0.4]}],
    }
    # A preview fill-in repeats the last boxes; recording it would count one sighting twice.
    assert replay_scenarios.stream_frame(dict(message, fresh=False), 4.6) is None


def test_a_basket_fixture_carries_the_zones_and_the_stream(tmp_path: Path) -> None:
    script = replay_scenarios.load_script(_write(tmp_path, "11_deposit_one", BASKET))
    stream = [{"ts": 2.0, "seq": 1, "detections": []}]
    document = replay_scenarios.fixture_document(
        script, [], video=tmp_path / "11_deposit_one.mp4", weights="w.pt", device="cpu",
        pipeline_settings={"conf_threshold": 0.5}, frames=1, duration_s=2.0, stream=stream,
    )
    assert document["basket"] == {"cartEdge": "bottom", "insideFraction": 0.35, "openingFraction": 0.2}
    assert document["stream"] == stream
    assert document["checkpoints"][0] == {"t_s": 8.0, "cart": {"century_tuna": 1}, "review": 0}
    assert "review" not in document["checkpoints"][1]
    # A basket scenario with no stream would score every checkpoint against an empty ledger.
    with pytest.raises(ValueError):
        replay_scenarios.fixture_document(
            script, [], video=tmp_path / "x.mp4", weights="w.pt", device="cpu",
            pipeline_settings={}, frames=0, duration_s=0.0,
        )


def test_a_counter_fixture_has_no_stream(tmp_path: Path) -> None:
    script = replay_scenarios.load_script(_write(tmp_path, "01_one_item", VALID))
    document = replay_scenarios.fixture_document(
        script, [], video=tmp_path / "01_one_item.mp4", weights="w.pt", device="cpu",
        pipeline_settings={}, frames=0, duration_s=0.0,
    )
    assert "basket" not in document and "stream" not in document


def test_the_flags_the_docs_use_are_the_flags_this_tool_declares() -> None:
    """A guard against the docs and the parser drifting apart — `test_docs_options.py` reads the
    parser, and this pins the names the README and the Makefile pass."""
    args = replay_scenarios.parse_args([])
    assert args.corpus == str(replay_scenarios.DEFAULT_CORPUS)
    assert args.out == str(replay_scenarios.DEFAULT_OUT)
    assert args.model is None and args.conf is None and args.only is None
    assert args.device == "auto"
    assert replay_scenarios.DEFAULT_OUT.name == "scenarios"
    assert replay_scenarios.DEFAULT_OUT.parent.name == "__fixtures__"
