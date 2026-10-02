# Scenario fixtures — the counting-accuracy corpus

The model answers "which products are in this frame". Whether the **cart** ends up right is
something `cartState.ts` infers from how those detections persist over time — the dwell, the settle
window, concurrent tracks — and that inference is what a checkout actually depends on. This
directory is where it gets measured against ground truth instead of against hand-written events.

The design is spec §7.1. There are two halves, and only one of them is in git.

**The corpus is recorded, not checked in.** Short videos of a scripted counter, shot with the real
StreamCam where it will be mounted, each beside a written script saying what the cart should hold at
each moment. They live in the sidecar's gitignored workspace (`sidecar/data/scenarios/`), because
they are camera output and weigh what camera output weighs.

**The fixtures are checked in.** `sidecar/tools/replay_scenarios.py` replays a video through the
app's own `Pipeline` and writes the resulting track log here, as `<name>.json`. They are small JSON,
so they are tracked, and they are what `npm test` scores on a fresh clone — no camera, no GPU, no
corpus.

## The script a human writes

One `<name>.json` beside each video of the same stem (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`). This
one lives in the corpus directory and is **not** committed. `t_s` is seconds into the video, and the
`cart` at each checkpoint is the whole expected cart, not a delta — an empty object means "nothing
on the counter", which is a real claim.

```json
{
  "name": "04_place_three_take_one",
  "description": "Three different products placed, then one taken back.",
  "bind_at_s": 1.0,
  "asserted_with": { "commit_dwell_s": 3, "remove_settle_s": 10, "min_commit_conf": 0.6 },
  "checkpoints": [
    { "t_s": 6.0, "cart": { "safeguard_pure_white_60g": 1 } },
    { "t_s": 12.0, "cart": { "safeguard_pure_white_60g": 1, "century_tuna": 1 } },
    { "t_s": 20.0, "cart": { "safeguard_pure_white_60g": 1, "century_tuna": 1, "lucky_me": 1 } },
    { "t_s": 30.0, "cart": { "safeguard_pure_white_60g": 1, "century_tuna": 1 } }
  ]
}
```

A quantity is a whole number **≥ 1**: an absent class is expressed by leaving it out, so a `0` is a
mistake rather than a state. An unknown key is an error, not ignored — a typo'd `checkpoint:` would
otherwise mean "this script asserts nothing", which is a corpus that passes by being empty.

`asserted_with` is optional provenance: the `commitDwellS`/`removeSettleS`/`minCommitConf` the
checkpoints were written assuming. The scoring runs under the app's **current** settings, so this
record is not an input — it is what makes a failing checkpoint read as "the app disagrees with the
assumption this was written for" rather than as a mystery.

## The fixture the replay writes

Written here as `<name>.json`, never by hand. `events` are the stored tracks in the desktop's own
vocabulary (`enteredAt`, not the sidecar's `entered_at`), and `checkpoints` is a copy of the script's,
so the ground truth travels with the log instead of pointing at a file a fresh clone does not have.
`pipeline` records the settings that shaped the events, so a wrong answer is diagnosable from the
file alone.

```json
{
  "name": "04_place_three_take_one",
  "description": "Three different products placed, then one taken back.",
  "video": "04_place_three_take_one.mp4",
  "weights": "models/scanncart-grocery.pt",
  "device": "cpu",
  "bind_at_s": 1.0,
  "frames": 900,
  "duration_s": 30.1,
  "pipeline": {
    "conf_threshold": 0.5,
    "imgsz": 640,
    "track_expiry_s": 1.5,
    "infer_frame_skip": 0,
    "resize_mode": "letterbox",
    "suppress_clamped_detections": true,
    "suppress_frame_filling_detections": true,
    "suppress_unsure_phantoms": true,
    "class_allowlist": []
  },
  "asserted_with": { "commit_dwell_s": 3, "remove_settle_s": 10, "min_commit_conf": 0.6 },
  "checkpoints": [
    { "t_s": 6.0, "cart": { "safeguard_pure_white_60g": 1 } },
    { "t_s": 30.0, "cart": { "safeguard_pure_white_60g": 1, "century_tuna": 1 } }
  ],
  "events": [
    {
      "sessionId": 1,
      "trackId": 3,
      "className": "safeguard_pure_white_60g",
      "maxConf": 0.91,
      "enteredAt": 4.2,
      "leftAt": null
    }
  ]
}
```

## Producing a fixture

From the repo root, with the corpus recorded into `sidecar/data/scenarios/` and the weights
installed:

```bash
make replay-scenarios
```

While iterating on one scenario, replay just it by name:

```bash
make replay-scenarios REPLAY_ARGS=--only 04_place_three_take_one
```

The replay is deterministic: the tool drives `Pipeline`'s own injectable clock from the video's
frame timestamps, so two runs over one file produce the same log. Nothing is monkeypatched, and no
wall clock reaches an event. Re-run it whenever the weights change (v2) — this is the regression
test for the *model's* effect on counting.

`make replay-scenarios` needs data the repo does not carry (a recorded corpus plus an installed
weight), so it is not part of `make test` and not in CI, and it **fails loudly** when the corpus is
absent rather than skipping.

## Scoring

`cartState.scenarios.test.ts` loads every fixture here, runs `deriveCartState` at each checkpoint
time under the app's current `commitDwellS`/`removeSettleS`/`minCommitConf`, and asserts the cart.
That runs in `npm test` with no camera or GPU, which is what makes tuning those three settings a
test run instead of a trip to the counter.

A real fixture captures the tracker's actual habits — track swaps, short flickers, two touching
items — which hand-written events miss. **There are no fixtures yet.** The directory holds no
`.json`, so the suite runs its loader and scoring against a temporary fixture written by the test
itself, and says out loud that the corpus has not been recorded. It is a state, not a skip.
