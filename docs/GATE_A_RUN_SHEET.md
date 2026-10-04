# Gate A recording run sheet and human review rubric

> Field trial run sheet, printable scorecards, and scoring rubric for the human visibility gate
> specified in [CART_TRANSFER_SPEC.md](./CART_TRANSFER_SPEC.md) §3.
>
> Verifies that grocery item additions, removals, and non-transfer interactions are physically
> observable from the single oblique camera angle before software development or hardware purchases.

---

## 1. Overview and pass criteria

Gate A proves whether a human reviewer watching video recorded from the intended camera angle can
reliably identify products, observe transfer trajectories, and distinguish real deposits and
retrievals from non-transfer interactions.

| Trial family | Count | Composition | Proposed pass criterion |
|---|---|---|---|
| **Deposits** | 30 | All 7 catalog SKUs, ≥2 actors, 3 lighting variations, both approach directions | ≥29/30 correctly identified & confirmed with valid endpoint hold (≥95%) |
| **Removals** | 30 | All 7 catalog SKUs, ≥2 actors, 3 lighting variations, inside-to-outside paths | ≥29/30 correctly identified & confirmed with valid endpoint hold (≥95%) |
| **No-transfers** | 30 | 5 trials across each of 6 edge-case categories | 30/30 correctly rejected (0 false transfers allowed) |
| **Total** | **90** | **45 per actor across 2 sessions** | **Overall SKU readability ≥57/60 on transfers, 0 false commits** |

**Crucial rule:** If a human reviewer cannot discern whether an item was released or cleared from
the video, automated vision algorithms cannot be expected to succeed either. If Gate A fails due to
line-of-sight occlusion, adjust camera elevation, angle, or basket opening tape markers and restart.

---

## 2. Hardware and physical setup

### 2.1 Camera and framing

- **Camera:** Logitech StreamCam mounted on an overhead/side rigid arm (45°–60° oblique angle).
- **Resolution and FPS:** 1080p (1920×1080) at 30 or 60 fps (unmirrored native feed).
- **Recording method:** Standard MP4/MKV video recording. You can use the dedicated helper script
  `sidecar/tools/gate_a_recorder.py` to prompt each trial and save clips directly with a JSON
  manifest. *Do not use preview overlay screenshots or lossy downscaled streams.* Clips are saved
  **clean** (no watermark) because they double as the replay corpus (`replay_scenarios.py`); pass
  `--stamp` only for a watermarked audit copy. A clip whose camera delivered a different rate than
  `--fps` is re-timed to the delivered rate so it plays back at real speed. Keep the default 10 s:
  hold still for about 3 s before the interaction, since the basket flags anything seen inside during
  its first 3 s.

```bash
# Preview planned trials without opening camera:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_recorder.py --dry-run

# Record session s1 for Actor A under lighting L1:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_recorder.py --session s1 --actor A --light L1

# Record a single specific trial:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_recorder.py --trial DEP-01 --duration 10.0
```

Reviewers grade the recorded clips with the companion tool `sidecar/tools/gate_a_reviewer.py`, which
plays each clip back, walks the §4.1 checklist, and writes per-reviewer CSV scorecards plus the §6
summary block:

```bash
# Grade session s1 as reviewer R1 (clips play in a window; SPACE pause, LEFT/RIGHT seek, Q grade):
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R1

# Regrade one trial (replaces the previous row in the scorecard):
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R1 --trial DEP-01

# Print the Gate A verdict from an existing scorecard without playback:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --summary

# Import a transcribed paper scorecard (typed from the printed §5 tables) directly:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R1 --from-csv paper.csv
```

The import file needs a header row naming `trial_id`, the five indicator columns and —
optionally — `notes`, in any order:

```text
trial_id,sku_legible,direction_correct,endpoint_reached,hold_valid,no_ambiguity,notes
DEP-01,x,1,yes,checked,1,clear label on the paper sheet
NOT-01,,y,y,y,y,endpoint correctly withheld
```

Marks may be `y`/`n`, `1`/`0`, `x`/blank or `yes`/`no` — whatever the typist finds fastest.
Verdicts are derived from the marks, never read from the file; `graded_at` is stamped at import
(the transcription moment). A row is **rejected**, never guessed at, when its trial is not in the
manifest, when it names a trial twice, when a mark is unrecognizable, or when its cell count does
not match the header (a short row would slide every cell after the gap onto the wrong column).
Rejected rows print with their line number and the run exits nonzero, while the valid rows still
land on the scorecard — an incomplete transcription is visible, not silent. `--from-csv -` reads
the same format from stdin, and combining it with `--trial` is refused.

In the scorecard CSV, `endpoint_reached` for a **no-transfer** row means "the endpoint was correctly
withheld" — a `0` there is a false commit, which is how the summary counts them. Deposits and removals
use the column literally (endpoint observed = `1`). Each row also carries a `graded_at` UTC timestamp,
and the scorecard is an **append-only grade log**: a regrade adds a row rather than replacing one,
`graded_at` orders the history, and every summary, resume, and import decision reads the *latest*
grade per trial. The `--summary` block prints the most recent grade's time, and `--history` prints
each trial's full timeline (regrades included, oldest first — `--trial <ID>` narrows it to one trial):

```bash
# Print each trial's grade timeline from the scorecard:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R1 --history
```

`--history` cannot be combined with `--summary` or `--from-csv`; grades written before the timestamp
column existed render as `(unstamped)` and sort oldest.

Once a second reviewer has also graded the session, `--merge R2` compares the two scorecards trial
by trial — the *latest* grade per trial on each side, so a regrade on either side is what the merge
reads — and classifies every manifest trial as agreed or disputed. Each disputed trial is written
to an adjudication worksheet carrying both reviewers' marks side by side plus blank adjudication
columns, and the run exits nonzero so a scripted merge notices that adjudication is pending. The
worksheet is deliberately never overwritten: marks already entered in it would be lost, so
adjudicate it (or delete it) before re-merging.

```bash
# Compare R1's and R2's scorecards and write the disputed trials to the worksheet:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R1 --merge R2

# The adjudicator grades the disputed clips, fills the worksheet's five indicator columns,
# and imports it back the same way a paper scorecard is imported:
sidecar/.venv/Scripts/python.exe sidecar/tools/gate_a_reviewer.py --out data/gate_a_footage --session s1 --reviewer R2 --from-csv data/gate_a_footage/s1/adjudication_R1_R2.csv
```

The worksheet's tail is the import header from above, with both reviewers' verdicts and marks
alongside it, so the adjudicator fills the same five indicator columns the §5 paper tables carry.
Rows left unfilled are **rejected** at import — a mark nobody entered is never guessed at — and the
imported grades land on the adjudicator's scorecard as the latest per trial. A merge never modifies
a reviewer's scorecard: it reads both and writes only the worksheet, and a re-merge after
adjudication still reports the two reviewers' (real) disagreement, because the disputed set is a
worklist for the §4 adjudication rather than a verdict. `--merge` cannot be combined with
`--summary`, `--history`, `--from-csv`, or `--trial`, and the two reviewer ids must differ.

Every grading prompt names its indicator key — `[3y / 3n] > ` — and the expected answer is that key
plus the verdict (`3n`), so an answer is always attached to the indicator it judges. A bare `y`/`n`
still answers the question on screen, but the key is what keeps an answer from silently sliding onto
the wrong row when the SKU question is skipped for a no-transfer: a prefix naming a different
indicator than the one displayed is refused with a hint, never applied.

- **Coordinate regions:** Use bright contrast tape (e.g. green/blue vinyl) to mark the basket boundary:
  1. `OUTSIDE`: Presentation counter / customer approach zone.
  2. `OPENING`: Top perimeter / rim plane of the basket basket.
  3. `INSIDE`: Bottom interior loading area / landing floor.
- **Reference clock:** A visible digital stopwatch or on-screen wall clock readable to 0.1 s.

### 2.2 Product catalog (all 7 SKUs)

Every deposit and removal block must cycle across all 7 catalog items:

1. `BEAR_BRAND`: Bear Brand Fortified Powdered Milk 33g (sachet)
2. `LUCKY_ME`: Lucky Me Pancit Canton Calamansi 80g (pouch)
3. `555_SARDINES`: 555 Sardines in Tomato Sauce 155g (can)
4. `CENTURY_TUNA`: Century Tuna Flakes in Oil 155g (can)
5. `SILVER_SWAN`: Silver Swan Sukang Puti 200ml (bottle)
6. `MILO`: Milo Chocolate Drink 22g (sachet)
7. `SAFEGUARD`: Safeguard Pure White 60g (bar soap box)

### 2.3 Environmental conditions

Record across two independent sessions (Session 1 and Session 2) and three lighting setups:

- **L1 (Standard ambient):** Standard supermarket/fluorescent ceiling illumination (300–500 lux).
- **L2 (Dim / directional):** Reduced ceiling light or single-side illumination (150–200 lux).
- **L3 (Glaze / high contrast):** Bright overhead beam or specular reflection on foil pouches / glossy cans (600–900 lux).

---

## 3. Protocol and completion rules

Reviewers evaluate recorded clips against the completion definitions from [CART_TRANSFER_SPEC.md](./CART_TRANSFER_SPEC.md) §2:

```
Deposit:  [OUTSIDE] ──▶ [OPENING] ──▶ [INSIDE] ──▶ [RELEASE ON SUPPORT] ──▶ [1.0s HOLD / HAND WITHDRAWN]
Removal:  [INSIDE]  ──▶ [OPENING] ──▶ [OUTSIDE] ──▶ [CLEAR RIM]        ──▶ [1.0s HOLD OUTSIDE]
```

### 3.1 Deposit trial rules
1. Actor presents the item in `OUTSIDE` zone with brand label facing toward camera for 1 second.
2. Item is transported smoothly through the marked `OPENING` into `INSIDE`.
3. Item is physically released onto the basket floor or on top of previously settled items.
4. Hand releases grip and completely separates / withdraws outside the opening.
5. Item remains stationary on basket support for ≥1.0 second (provisional hold time).

### 3.2 Removal trial rules
1. Basket contains item(s); target item must be visible from camera before grasp.
2. Actor grasps target item and lifts it up through the marked `OPENING`.
3. Item completely clears the perimeter rim into the `OUTSIDE` zone.
4. Item is held steady in the `OUTSIDE` zone for ≥1.0 second.
5. Releasing outside is optional; keeping the item in hand outside the basket is valid.

### 3.3 No-transfer trial categories (5 trials each)
- **N1 — Outside Presentation / Cancel:** Item presented in front of camera, moved near opening, but never crosses opening; withdrawn.
- **N2 — Hovering / Hesitation:** Item lowered into opening plane, hovered without release for 2–3 seconds, then lifted back outside.
- **N3 — Empty-Hand Reach:** Customer reaches into basket, touches items, rearranges nothing, and withdraws empty hand.
- **N4 — Internal Rearrangement:** Item lifted inside basket, repositioned within basket floor, never crosses opening rim to outside.
- **N5 — Occluded Partial Crossing & Abort:** Item lowered toward basket while occluded by customer body/sleeve, reversed before release.
- **N6 — Temporary Hand Occlusion:** Hand covers product already at rest in basket for 3–5 seconds, then withdraws without moving item.

---

## 4. Human review rubric & scoring formula

Each recorded trial clip is evaluated independently by Reviewer 1 (R1). Any ambiguous or disputed clip
is adjudicated by Reviewer 2 (R2) — `--merge R2` in §2.1 is the tooling that flags exactly the
disputed set and hands it to the adjudicator as a worksheet.

### 4.1 Clip evaluation checklist

For every clip, the reviewer fills out 6 discrete binary indicators:

1. **SKU Legible (0 or 1):** Could a human unambiguously identify the product SKU from label text/graphics during the trial?
2. **Direction Correct (0 or 1):** Was the motion trajectory distinctly observed as Inward, Outward, or Internal/None?
3. **Endpoint Reached (0 or 1):**
   - For Deposit: Did the item come to rest supported inside the basket, with hand fully detached?
   - For Removal: Did the item fully clear the top perimeter tape boundary to the outside?
   - For No-Transfer: Did the item correctly refrain from reaching either terminal state?
4. **Hold Duration Valid (0 or 1):** Did the item remain stationary in the completion zone for ≥1.0 s?
5. **No Visual Ambiguity (0 or 1):** Was the interaction free of blocking occlusions (e.g. elbow, torso, hair) during critical transitions?
6. **Verdict (PASS or FAIL):** All 5 indicators above must be 1 to score PASS.

### 4.2 Scoring equations

$$\text{SKU Readability Rate} = \frac{\sum \text{SKU Legible in Transfers}}{60} \ge 95\% \quad (57 / 60)$$

$$\text{Deposit Accuracy} = \frac{\sum \text{Deposit PASS}}{30} \ge 96.7\% \quad (29 / 30)$$

$$\text{Removal Accuracy} = \frac{\sum \text{Removal PASS}}{30} \ge 96.7\% \quad (29 / 30)$$

$$\text{False Commit Rate (No-Transfer)} = \frac{\sum \text{No-Transfer False Claims}}{30} = 0.0\% \quad (0 / 30)$$

---

## 5. Printable trial run sheet

*Instructions for field crew: Print this section. Record actor initials, lighting condition, camera height, and trial results as clips are filmed.*

**Session ID:** `____________________`  
**Date & Time:** `____________________`  
**Location:** `____________________`  
**Camera Setup:** Logitech StreamCam · Height: `_______ cm` · Oblique Tilt: `_______ °`  
**Actors:** Actor A: `________________` (Initials: `____`) · Actor B: `________________` (Initials: `____`)  
**Lighting Check:** L1 Ambient: `[ ]` · L2 Dim: `[ ]` · L3 Glaze/Reflection: `[ ]`  

### 5.1 Part 1: Item Deposits (30 Trials)

| Clip ID | Actor | Light | Hand / Approach | SKU Under Test | SKU Legible? | Released on Base? | 1.0s Hold? | Reviewer Verdict | Notes / Occlusions |
|---|---|---|---|---|:---:|:---:|:---:|:---:|---|
| `DEP-01` | A | L1 | Left / Front | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-02` | A | L1 | Right / Side | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-03` | A | L1 | Right / Front | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-04` | A | L1 | Left / Side | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-05` | A | L1 | Right / Front | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-06` | A | L1 | Left / Front | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-07` | A | L1 | Right / Side | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-08` | A | L2 | Right / Front | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-09` | A | L2 | Left / Side | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-10` | A | L2 | Right / Side | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-11` | A | L2 | Left / Front | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-12` | A | L2 | Right / Front | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-13` | A | L3 | Left / Side | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-14` | A | L3 | Right / Front | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-15` | A | L3 | Right / Side | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-16` | B | L1 | Right / Front | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-17` | B | L1 | Left / Front | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-18` | B | L1 | Right / Side | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-19` | B | L1 | Left / Side | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-20` | B | L1 | Right / Front | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-21` | B | L1 | Left / Front | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-22` | B | L1 | Right / Side | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-23` | B | L2 | Left / Front | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-24` | B | L2 | Right / Side | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-25` | B | L2 | Left / Side | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-26` | B | L3 | Right / Front | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-27` | B | L3 | Left / Front | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-28` | B | L3 | Right / Side | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-29` | B | L3 | Left / Side | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `DEP-30` | B | L3 | Right / Front | Bear Brand (Over Stack) | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | Deposited on top of items |

---

### 5.2 Part 2: Item Removals (30 Trials)

| Clip ID | Actor | Light | Hand / Approach | SKU Under Test | SKU Legible? | Cleared Rim? | 1.0s Hold? | Reviewer Verdict | Notes / Grip Concealment |
|---|---|---|---|---|:---:|:---:|:---:|:---:|---|
| `REM-01` | A | L1 | Right / Top | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-02` | A | L1 | Left / Front | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-03` | A | L1 | Right / Side | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-04` | A | L1 | Left / Top | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-05` | A | L1 | Right / Neck | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-06` | A | L1 | Left / Side | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-07` | A | L1 | Right / Top | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-08` | A | L2 | Left / Side | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-09` | A | L2 | Right / Front | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-10` | A | L2 | Left / Top | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-11` | A | L2 | Right / Side | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-12` | A | L2 | Left / Neck | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-13` | A | L3 | Right / Top | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-14` | A | L3 | Left / Side | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-15` | A | L3 | Right / Front | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-16` | B | L1 | Right / Side | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-17` | B | L1 | Left / Top | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-18` | B | L1 | Right / Front | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-19` | B | L1 | Left / Side | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-20` | B | L1 | Right / Neck | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-21` | B | L1 | Left / Front | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-22` | B | L1 | Right / Top | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-23` | B | L2 | Left / Side | 555 Sardines Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-24` | B | L2 | Right / Top | Century Tuna Can | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-25` | B | L2 | Left / Front | Silver Swan Bottle | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-26` | B | L3 | Right / Side | Milo Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-27` | B | L3 | Left / Top | Safeguard Bar Soap | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-28` | B | L3 | Right / Front | Bear Brand Sachet | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-29` | B | L3 | Left / Side | Lucky Me Pancit | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | |
| `REM-30` | B | L3 | Right / Bottom | Century Tuna (Under Stack) | `[ ]` | `[ ]` | `[ ]` | `PASS / FAIL` | Lifted from under stacked pouch |

---

### 5.3 Part 3: No-Transfer Edge Cases (30 Trials)

*Goal: Verify that non-transfer interactions are unmistakably distinct from commits.*

| Clip ID | Actor | Light | Scenario Description | Interaction Sequence | Endpoint Reached? | Reviewer Verdict | Notes |
|---|---|---|---|---|:---:|:---:|---|
| `NOT-01` | A | L1 | N1: Outside Presentation | Present Bear Brand in counter zone, hold 2s, withdraw | `NO / YES` | `PASS / FAIL` | Must not claim Inbound |
| `NOT-02` | A | L1 | N1: Outside Presentation | Present Milo sachet near rim, tilt label, withdraw | `NO / YES` | `PASS / FAIL` | Must not cross rim |
| `NOT-03` | B | L2 | N1: Outside Presentation | Present 555 Sardines with two hands, rotate, lower away | `NO / YES` | `PASS / FAIL` | Must not trigger deposit |
| `NOT-04` | B | L3 | N1: Outside Presentation | Present Century Tuna, tap counter edge, withdraw | `NO / YES` | `PASS / FAIL` | Must stay in Outside zone |
| `NOT-05` | A | L3 | N1: Outside Presentation | Present Safeguard bar box 5 cm above rim, pull back | `NO / YES` | `PASS / FAIL` | No crossing |
| `NOT-06` | A | L1 | N2: Hover & Retract | Lower Lucky Me into rim plane, hover 3s, lift back out | `NO / YES` | `PASS / FAIL` | No release occurred |
| `NOT-07` | A | L2 | N2: Hover & Retract | Lower Silver Swan into opening, hover 2s, retract to side | `NO / YES` | `PASS / FAIL` | No release occurred |
| `NOT-08` | B | L1 | N2: Hover & Retract | Lower Century Tuna into basket volume, hesitate, withdraw | `NO / YES` | `PASS / FAIL` | Hand never detached |
| `NOT-09` | B | L2 | N2: Hover & Retract | Lower Milo, wave gently inside opening, withdraw | `NO / YES` | `PASS / FAIL` | Hand never detached |
| `NOT-10` | B | L3 | N2: Hover & Retract | Lower Safeguard box, touch floor without letting go, lift | `NO / YES` | `PASS / FAIL` | Continuous grip retained |
| `NOT-11` | A | L1 | N3: Reach-Only | Reach open right hand into empty basket, withdraw | `NO / YES` | `PASS / FAIL` | No SKU present |
| `NOT-12` | A | L2 | N3: Reach-Only | Reach left hand into basket with 2 items, touch floor, out | `NO / YES` | `PASS / FAIL` | No items moved |
| `NOT-13` | B | L1 | N3: Reach-Only | Reach both hands into basket, cup hands together, out | `NO / YES` | `PASS / FAIL` | No items moved |
| `NOT-14` | B | L2 | N3: Reach-Only | Reach hand in, point at items, pull back out | `NO / YES` | `PASS / FAIL` | Hand empty throughout |
| `NOT-15` | A | L3 | N3: Reach-Only | Reach hand in rapidly, pause 2s at bottom, withdraw | `NO / YES` | `PASS / FAIL` | Hand empty throughout |
| `NOT-16` | A | L1 | N4: Rearrangement | Shift Lucky Me from left basket corner to right corner | `NO / YES` | `PASS / FAIL` | Never crossed rim out |
| `NOT-17` | A | L2 | N4: Rearrangement | Lift 555 Sardines 5 cm inside basket, place back down | `NO / YES` | `PASS / FAIL` | Stays wholly inside |
| `NOT-18` | B | L1 | N4: Rearrangement | Stack Milo sachet on top of Safeguard box inside basket | `NO / YES` | `PASS / FAIL` | Internal re-position |
| `NOT-19` | B | L2 | N4: Rearrangement | Slide Century Tuna along bottom floor 10 cm | `NO / YES` | `PASS / FAIL` | Horizontal shift only |
| `NOT-20` | B | L3 | N4: Rearrangement | Stand Silver Swan bottle upright from side orientation | `NO / YES` | `PASS / FAIL` | Orientation change only |
| `NOT-21` | A | L1 | N5: Occluded Crossing Abort | Lower Milo behind wrist/sleeve, break rim, pull back | `NO / YES` | `PASS / FAIL` | Unresolved / aborted |
| `NOT-22` | A | L2 | N5: Occluded Crossing Abort | Lower Tuna while forearm blocks camera view, retract | `NO / YES` | `PASS / FAIL` | Obscured; no release |
| `NOT-23` | B | L1 | N5: Occluded Crossing Abort | Lower Bear Brand behind tilted palm, reverse out | `NO / YES` | `PASS / FAIL` | No visible release |
| `NOT-24` | B | L2 | N5: Occluded Crossing Abort | Lower Safeguard under sleeve cuff, abort at opening | `NO / YES` | `PASS / FAIL` | Inward unconfirmed |
| `NOT-25` | A | L3 | N5: Occluded Crossing Abort | Lower 555 Sardines while body leans over rim, reverse out | `NO / YES` | `PASS / FAIL` | Partial path; aborted |
| `NOT-26` | A | L1 | N6: Occlusion of Rest Item | Cover deposited Milo with flat palm for 4s, withdraw | `NO / YES` | `PASS / FAIL` | Must not decrement |
| `NOT-27` | A | L2 | N6: Occlusion of Rest Item | Cover Safeguard box with paper sheet for 5s, remove sheet | `NO / YES` | `PASS / FAIL` | Concealment ≠ removal |
| `NOT-28` | B | L1 | N6: Occlusion of Rest Item | Place hand over Lucky Me pouch for 3s, withdraw hand | `NO / YES` | `PASS / FAIL` | Concealment ≠ removal |
| `NOT-29` | B | L2 | N6: Occlusion of Rest Item | Drop cloth over items in basket, wait 5s, lift cloth | `NO / YES` | `PASS / FAIL` | Inventory stays intact |
| `NOT-30` | B | L3 | N6: Occlusion of Rest Item | Rest forearm across basket opening for 4s, withdraw | `NO / YES` | `PASS / FAIL` | No item disturbed |

---

## 6. Summary score card & sign-off

| Evaluation Metric | Target Floor | Actual Count | Pass / Fail |
|---|---|---|:---:|
| **Transfers with Legible SKU** | $\ge 57 / 60 \ (95.0\%)$ | `____ / 60` | `PASS / FAIL` |
| **Deposit Trials Validated** | $\ge 29 / 30 \ (96.7\%)$ | `____ / 30` | `PASS / FAIL` |
| **Removal Trials Validated** | $\ge 29 / 30 \ (96.7\%)$ | `____ / 30` | `PASS / FAIL` |
| **No-Transfer False Commits** | $= 0 / 30 \ (0.0\%)$ | `____ / 30` | `PASS / FAIL` |
| **Overall Gate A Verdict** | **All 4 metrics pass** | — | **`GO / NO-GO`** |

### Reviewer Sign-Off

- **Lead Reviewer 1:** `________________________` Date: `____________` Signature: `________________`
- **Secondary Reviewer 2:** `________________________` Date: `____________` Signature: `________________`

**Disposition if NO-GO:**
- If *SKU Legibility* failed: Check camera focus / exposure ceiling, or increase presentation dwell.
- If *Endpoint Confirmation* failed: Camera elevation is too shallow or basket rim tape is ambiguous. Elevate mount and re-shoot.
- If *No-Transfer False Commits* occurred: The boundary definition is leaking. Adjust opening plane markers and re-run.
