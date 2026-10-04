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

`cartState.scenarios.test.ts` loads every fixture here. A counter fixture is scored by running `deriveCartState` at each checkpoint
time under the app's current `commitDwellS`/`removeSettleS`/`minCommitConf`, and asserts the cart.
That runs in `npm test` with no camera or GPU, which is what makes tuning those three settings a
test run instead of a trip to the counter.

## Basket scenarios (deposit and removal)

A counter scenario is scored from the track log, which says when a product was seen and nothing
about *where*. A basket deposit is a path — outside, through the opening, held inside the cart band —
so a basket scenario is scored from the frames themselves. A script becomes one by declaring the zone
preset the camera was set up with when the video was shot:

```json
{
  "name": "21_deposit_then_remove",
  "description": "One tuna carried in through the opening, then lifted back out.",
  "bind_at_s": 1.0,
  "basket": { "cart_edge": "bottom", "inside_fraction": 0.35, "opening_fraction": 0.2 },
  "checkpoints": [
    { "t_s": 9.0, "cart": { "century_tuna": 1 }, "review": 0 },
    { "t_s": 18.0, "cart": {}, "review": 0 }
  ]
}
```

`basket` is the zone preset the scene was shot with — the same three values as the Admin Panel's
cart zones — and not the app's current one, because it is a fact about where the camera pointed.
`review` (basket only, optional) is how many review items the ledger should hold at that moment: a
right cart with a spurious review is still a failed checkout, since the tablet refuses Finish while
one stands, so most basket checkpoints should say `0`. A checkpoint without it does not assert it.

A **removal clip** starts with the item already in the basket, so it says what the basket holds when
the clip begins:

```json
{
  "name": "REM-04",
  "bind_at_s": 0.0,
  "basket": { "cart_edge": "bottom", "inside_fraction": 0.35, "opening_fraction": 0.2 },
  "initial_cart": { "century_tuna": 1 },
  "checkpoints": [
    { "t_s": 2.5, "cart": { "century_tuna": 1 }, "review": 0 },
    { "t_s": 9.0, "cart": {}, "review": 0 }
  ]
}
```

`initial_cart` (basket only) is a session that began before the recording: the scorer binds before the
clip's first frame — long enough before that the empty-basket window is over — and puts those items
in through the ledger's own `apply` as confirmed deposits. Without it the ledger starts empty, and
lifting an item out scores as review ("removal of … which is not in the basket"). A removal clip
needs `initial_cart`, a deposit clip with an empty basket does not, and a deposit into a partly
filled basket lists what was already there.

The replay then also writes the fixture's `basket` (camel-cased, as `ZonePreset`) and `stream`: every
fresh frame message, stamped with the **video's** timestamp instead of the app's wall clock, with
boxes and confidences rounded to four digits (and `initial_cart`, copied as written). Preview fill-ins (`fresh: false`) are not recorded, for
the reason `BasketTracker` ignores them. The scorer feeds that stream through a real `BasketTracker` —
the transfer machine and the ledger the app runs — binding at `bind_at_s`, and reads the ledger and
its review list at each checkpoint. The scanner's `conf_threshold` is the one the replay ran at; the
transfer machine's other thresholds are the app's current defaults, so tuning them is a test run.

Two limits worth knowing before trusting a green run:

- **The replay infers every frame.** Live, the machine sees the inference cadence the PC sustains,
  which is fewer fresh observations per crossing. A path that passes here with frames to spare can
  still miss at the counter; `docs/CART_TRANSFER_SPEC.md` asks for fresh-observation counts under real
  load for this reason.
- **The empty-basket check covers the first 3 s after bind.** It flags any product seen in the cart
  band during that window, including one the customer just deposited through the opening, so a
  scenario that deposits within 3 s of `bind_at_s` expects `review: 1`. Leave a few seconds after
  Start unless that case is the point of the scenario.

A minimum basket shot list, alongside the counter list in spec §7.1:

1. One item deposited through the opening and released inside.
2. Three different products deposited one at a time.
3. Deposit, then the same item lifted back out through the opening (removal).
4. Remove, then put back (one track, both directions).
5. An item moved toward the opening and withdrawn without entering (nothing may change).
6. A hand hovering in the opening with nothing in it (nothing may change).
7. An item already in the basket at Start (expects `review: 1`, the baseline).
8. A product changed mid-path, or two items crossing the opening together (expects review, never a
   guessed count).

A real fixture captures the tracker's actual habits — track swaps, short flickers, two touching
items — which hand-written events miss. **There are no fixtures yet.** The directory holds no
`.json`, so the suite runs its loader and scoring against a temporary fixture written by the test
itself, and says out loud that the corpus has not been recorded. It is a state, not a skip.
