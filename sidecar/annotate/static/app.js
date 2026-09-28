// The annotator's front end. Plain ES modules, no build step - `make annotate` starts the server
// and this is what it serves.
//
// Three things it deliberately does not do:
//
// * **No geometry.** Every box arrives with `left/top/width/height` already as percentages, so the
//   page assigns numbers rather than computing them. A `(cx - w/2) * 100` here would be a second
//   copy of the mapping the labels themselves use, and the two could disagree in a way nothing
//   would catch.
// * **No persistence of its own.** Nothing is written until Save is pressed; suggestions live on
//   the server (`provenance.suggestion`) so "a machine drew this and nobody looked" stays a fact
//   about the files rather than a claim about a click.
// * **No automatic nulls.** `n` writes an empty label file, which is the statement "there is no
//   item here". Only a person makes it - a model's silence is not that statement.

const state = {
  config: null,
  frames: [],
  index: 0,
  name: "",
  boxes: [],
  cls: 0,
  dirty: false,
  // The second pass over the same set: only the decisions a weight made that nobody has looked
  // at, ordered by the split they land in (`store.review_worklist`). The server does the
  // selection and the ordering - this is one flag, not a second copy of the rule.
  review: false,
  // The pass's other halves: the `far` frames in `test`/`valid` a weight was asked about and found
  // nothing in (`store.draw_worklist`), and the hard negatives in those splits with no null yet
  // (`store.null_worklist`) - the lists `make human-pass` prints as the checklist's draw and null
  // sections, served from the same calls so the page and the document cannot disagree. The three
  // views are exclusive; each button turns the other two off.
  draw: false,
  nulls: false,
  // The three narrowing filters, sent as query parameters and matched on the server against the
  // store's own fields. Empty means "any" - the server distinguishes that from the `unknown` token,
  // which names the frames with nothing recorded (the hard negatives, the unplanned ones).
  split: "",
  distance: "",
  filterState: "",
  // The last gate readout, kept so the chip's click can name the pair it is about without asking
  // the server again: it is the same list `/api/config` builds the split option from.
  gate: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
}

function note(text) {
  $("note").textContent = text || "";
}

// -- worklist ------------------------------------------------------------------

async function loadConfig() {
  state.config = await api("/api/config");
  const { weight, classes, hosted } = state.config;

  $("weight").textContent = weight
    ? `suggesting from ${weight.name} · ${weight.resize_mode} · conf ${weight.conf}`
    : `no local weight${state.config.weight_note ? ` — ${state.config.weight_note}` : ""}`;

  $("classes").innerHTML = "";
  classes.forEach((entry) => {
    const button = document.createElement("button");
    button.dataset.testid = `class-${entry.slug}`;
    button.dataset.cls = String(entry.index);
    button.textContent = `${entry.index + 1} · ${entry.slug}`;
    button.title = entry.name;
    button.addEventListener("click", () => selectClass(entry.index));
    $("classes").appendChild(button);
  });

  const available = hosted.available.length ? hosted.available.join(", ") : "none configured";
  $("hosted").textContent = hosted.name
    ? `second opinion ${hosted.name} · ${hosted.calls}/${hosted.max_calls} calls · keys: ${available}`
    : "second opinion off · nothing leaves this machine for suggestions";

  // The options come from the server, not from the frames that happen to be loaded: a list built
  // from what is on screen would lose the values the current filter filtered away, so a filter
  // could be narrowed but never widened back.
  fillFilter("filter-split", "any split", state.config.splits);
  fillFilter("filter-distance", "any distance", state.config.distances);
}

function fillFilter(id, anyLabel, entries) {
  const select = $(id);
  select.innerHTML = "";
  [{ value: "", label: anyLabel }, ...(entries || [])].forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.value;
    option.textContent = entry.label;
    select.appendChild(option);
  });
}

function query() {
  const params = new URLSearchParams();
  if (state.review) params.set("review", "1");
  if (state.draw) params.set("draw", "1");
  if (state.nulls) params.set("nulls", "1");
  if (state.split) params.set("split", state.split);
  if (state.distance) params.set("distance", state.distance);
  if (state.filterState) params.set("state", state.filterState);
  const text = params.toString();
  return text ? `?${text}` : "";
}

async function loadFrames() {
  const body = await api(`/api/frames${query()}`);
  state.frames = body.frames;
  renderProgress(body.counts);
  renderGate(body.gate);
  const list = $("worklist");
  list.innerHTML = "";
  state.frames.forEach((frame, index) => {
    const row = document.createElement("button");
    row.dataset.testid = `frame-${frame.name}`;
    row.className = `row state-${frame.state}${frame.machine_only ? " machine" : ""}`;
    row.innerHTML =
      // The split rides in the cell line because it is the fact the pass is for - a frame in `test`
      // a machine drew makes the acceptance number a measurement of the annotator.
      `<span class="cell">${frame.cell}${frame.split ? ` · ${frame.split}` : ""}</span>` +
      `<span class="name">${frame.name}</span>` +
      `<span class="state">${frame.state}${frame.machine_only ? " · machine" : ""}` +
      `${frame.boxes ? ` · ${frame.boxes}` : ""}</span>`;
    row.addEventListener("click", () => open(index));
    list.appendChild(row);
  });
}

function renderProgress(counts) {
  // The key names are the snapshot's, not this page's: `null_annotations` is what
  // `/api/frames` puts in `counts` (see `annotate/store.summary`), and `pseudo` is the
  // decisions a weight drew that nobody has reviewed. Guessing at the spelling here produced
  // an "undefined null" readout, which is why it is spelled out in a comment.
  const done = counts.decided || 0;
  const percent = counts.total ? Math.round((100 * done) / counts.total) : 0;
  // `loaded` is the filtered count, so it is appended only when something is narrowing the list -
  // unfiltered it says nothing the total does not.
  const narrowed = filters().length || state.review || state.draw || state.nulls;
  const showing = narrowed ? ` · showing ${counts.loaded}` : "";
  $("progress").textContent =
    `${done}/${counts.total} decided (${percent}%) · ${counts.unlabeled} left · ` +
    `${counts.null_annotations} null · ${counts.pseudo} machine-only awaiting review${showing}`;
  // The count is on the button because the gate is a *number* to drive to zero, and the pass is
  // what clears it. It reads the whole set's count even while the list is filtered, so the button
  // does not go quiet the moment the review list empties. Draw and Nulls carry no count of their
  // own: their sizes are the checklist's numbers, and a second copy here could only disagree.
  $("review").textContent = `Review (u) · ${counts.pseudo || 0}`;
  $("review").classList.toggle("active", state.review);
  $("draw").classList.toggle("active", state.draw);
  $("nulls").classList.toggle("active", state.nulls);
}

function renderGate(gate) {
  // The finish line, and the one readout that is deliberately not recomputed here: `pending`,
  // `remaining` and the per-split split of it all come from the server, off the same call the
  // checklist's header is rendered from. Working the pass moves it - every save reloads this list.
  state.gate = gate;
  const pair = gate.splits.join("/");
  const chip = $("gate");
  // Disabled rather than hidden once it is clear: it is still the answer to "is the gate clean",
  // and a chip that vanished would read as a readout that stopped working.
  chip.disabled = gate.clear;
  if (gate.clear) {
    chip.textContent = gate.remaining
      ? `gate: clear (${gate.remaining} machine-only left outside ${pair})`
      : "gate: clear - nothing is machine-only";
    chip.title = `The acceptance gate: train_model and accept_v2 refuse a set whose ${pair} hold a weight's unread boxes. Nothing is left there.`;
  } else {
    const perSplit = gate.splits.map((split) => `${split} ${gate.by_split[split] || 0}`).join(" / ");
    chip.textContent = `gate: ${gate.pending} machine-only in ${pair} (${perSplit}) - stop at ${gate.remaining}`;
    chip.title = `The acceptance gate: train_model and accept_v2 refuse a set whose ${pair} hold a weight's unread boxes. Click to open those ${gate.pending} in the review pass, narrowed to ${pair}: the review count falls to ${gate.remaining} when they are clean - the other ${gate.remaining} are in train, where unread boxes cost nothing.`;
  }
  chip.classList.toggle("clear", gate.clear);
}

// The chip's click: the work the number is about, in one step. The pair is selected in the split
// filter rather than sent as a query of its own, because that option is built from the same
// `GATE_SPLITS` on the server - picking it by hand and pressing the chip are the same request.
async function showGate() {
  const gate = state.gate;
  if (!gate || gate.clear) return;
  state.review = true;
  state.draw = false;
  state.nulls = false;
  // Read back off the select: if the option were ever missing, the list would widen to every split
  // rather than silently disagreeing with what the chip says it is showing.
  $("filter-split").value = gate.splits.join(",");
  state.split = $("filter-split").value;
  await refresh(
    () =>
      `${state.frames.length} machine-only decision(s) in ${gate.splits.join("/")} - the splits a run refuses on. Check each one's boxes, then Confirm (f) if they are right, or edit and Save`
  );
}

// -- the view and the filters ---------------------------------------------------

// The three filters as words, for the notes: the wording comes off the `<option>`s, which came from
// `/api/config`, so `unknown` reads as `unrecorded` or `unassigned` rather than as a token.
function filters() {
  const parts = [];
  if (state.split) parts.push(`split ${labelOf("filter-split")}`);
  if (state.distance) parts.push(`distance ${labelOf("filter-distance")}`);
  if (state.filterState) parts.push(`${state.filterState} only`);
  return parts;
}

function labelOf(id) {
  const select = $(id);
  return select.value ? select.selectedOptions[0].textContent : "";
}

function describe() {
  const parts = filters();
  return `${state.frames.length} frame(s)${parts.length ? ` · ${parts.join(" · ")}` : ""}`;
}

function emptyNote() {
  if (state.review) return "nothing is awaiting review";
  if (state.draw) return "nothing left to draw in those splits";
  if (state.nulls) return "every hard negative in those splits is marked";
  if (filters().length) return "no frame matches these filters";
  return "the worklist is empty";
}

// Reload the list and open its first frame. One path for a view toggle and a filter change, because
// both are "the list changed under the cursor" - and the note is built *after* the load so it can
// count the frames that came back.
async function refresh(buildText) {
  await loadFrames();
  if (!state.frames.length) {
    note(emptyNote());
    return;
  }
  await open(0);
  note(typeof buildText === "function" ? buildText() : buildText || describe());
}

const backToWorklist = () => (filters().length ? describe() : "back to the whole worklist");

// One view at a time: `review`, `draw` and `nulls` are the checklist's three sections, so turning
// one on is also turning the other two off - two at once would be a list that is not any of them.
// Pressing the active one returns to the ordinary worklist, filters intact.
async function setView(view) {
  state.review = view === "review" ? !state.review : false;
  state.draw = view === "draw" ? !state.draw : false;
  state.nulls = view === "nulls" ? !state.nulls : false;
  await refresh(() => {
    if (state.review) {
      return `${state.frames.length} unreviewed decision(s), worst split first, thinnest cells within it - check the boxes, then Confirm (f) if they are right or edit and Save`;
    }
    if (state.draw) {
      return `${state.frames.length} far frame(s) in test/valid a weight was asked about and found nothing in - draw every item you can see, then Save`;
    }
    if (state.nulls) {
      return `${state.frames.length} hard negative(s) with no null yet - press N on each, one keypress per frame`;
    }
    return backToWorklist();
  });
}

const toggleReview = () => setView("review");
const toggleDraw = () => setView("draw");
const toggleNulls = () => setView("nulls");

// -- the frame -----------------------------------------------------------------

async function open(index) {
  if (index < 0 || index >= state.frames.length) return;
  state.index = index;
  const frame = state.frames[index];
  const body = await api(`/api/frame/${frame.name}`);
  state.name = frame.name;
  state.boxes = body.boxes.map((b) => ({ ...b }));
  state.dirty = false;
  $("image").src = `${frame.url}?t=${Date.now()}`;
  $("frame-info").textContent =
    `${frame.cell} · ${frame.name}` +
    // Split and session are different facts and both matter: the split decides whether this
    // frame's labels are part of the acceptance number at all (test must be human work),
    // and the session is which capture it came from.
    ` · split ${body.split || "unassigned"}` +
    ` · session ${frame.session || "unrecorded"}` +
    (body.provenance.provider ? ` · last save by ${body.provenance.provider}` : "");
  drawOverlay();
  [...$("worklist").children].forEach((el, i) => el.classList.toggle("active", i === index));
}

function drawOverlay() {
  const overlay = $("overlay");
  overlay.innerHTML = "";
  state.boxes.forEach((box, i) => {
    const element = document.createElement("div");
    element.className = "box";
    element.dataset.testid = `box-${i}`;
    element.dataset.cls = String(box.cls);
    element.style.left = `${box.left}%`;
    element.style.top = `${box.top}%`;
    element.style.width = `${box.width}%`;
    element.style.height = `${box.height}%`;
    element.innerHTML = `<span>${state.config.classes[box.cls]?.slug ?? box.cls}</span>`;
    overlay.appendChild(element);
  });
}

function toBox(rect, x0, y0, x1, y1) {
  // Clamped to the image, which is the one place the page has to be careful: a drag that leaves
  // the picture (easy - the pointer is not captured, and the overlay is exactly the image) would
  // otherwise produce a coordinate outside 0..1, and the server refuses those outright rather
  // than writing a box the trainer would silently clip. The intent of such a drag is "the box I
  // can see", so clamping is the faithful reading of it. This mirrors `providers.clamp_box`,
  // which does the same for a model's coordinates.
  const clamp = (value, limit) => Math.max(0, Math.min(limit, value));
  const left = clamp(Math.min(x0, x1), rect.width);
  const top = clamp(Math.min(y0, y1), rect.height);
  const width = clamp(Math.max(x0, x1), rect.width) - left;
  const height = clamp(Math.max(y0, y1), rect.height) - top;
  return {
    cls: state.cls,
    cx: (left + width / 2) / rect.width,
    cy: (top + height / 2) / rect.height,
    w: width / rect.width,
    h: height / rect.height,
  };
}

function beginDrag(event) {
  const image = $("image");
  const rect = image.getBoundingClientRect();
  const startX = event.clientX - rect.left;
  const startY = event.clientY - rect.top;
  let preview = null;

  const move = (moveEvent) => {
    const box = toBox(rect, startX, startY, moveEvent.clientX - rect.left, moveEvent.clientY - rect.top);
    if (!preview) {
      preview = document.createElement("div");
      preview.className = "box drawing";
      $("overlay").appendChild(preview);
    }
    preview.style.left = `${(box.cx - box.w / 2) * 100}%`;
    preview.style.top = `${(box.cy - box.h / 2) * 100}%`;
    preview.style.width = `${box.w * 100}%`;
    preview.style.height = `${box.h * 100}%`;
  };

  const up = (upEvent) => {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    if (preview) preview.remove();
    const box = toBox(rect, startX, startY, upEvent.clientX - rect.left, upEvent.clientY - rect.top);
    if (box.w < 0.005 || box.h < 0.005) return; // a click, not a box
    // Rounded to the same precision the label file carries, so a second save of an untouched box
    // produces a byte-identical file.
    state.boxes.push({
      cls: box.cls,
      cx: round(box.cx),
      cy: round(box.cy),
      w: round(box.w),
      h: round(box.h),
      ...percent(box),
    });
    state.dirty = true;
    drawOverlay();
    note(`drew ${state.config.classes[box.cls].slug} — press Save to write it`);
  };

  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
  event.preventDefault();
}

const round = (value) => Math.round(value * 1e6) / 1e6;
const percent = (box) => ({
  left: (box.cx - box.w / 2) * 100,
  top: (box.cy - box.h / 2) * 100,
  width: box.w * 100,
  height: box.h * 100,
});

// -- actions -------------------------------------------------------------------

function selectClass(index) {
  state.cls = index;
  [...$("classes").children].forEach((el) =>
    el.classList.toggle("active", Number(el.dataset.cls) === index)
  );
}

async function suggest() {
  try {
    const body = await api(`/api/frame/${state.name}/suggest`, { method: "POST" });
    if (!body.boxes.length) {
      note(`${body.note || "nothing proposed"}${body.hosted_capped ? " · hosted cap reached" : ""}`);
      return;
    }
    state.boxes = body.boxes.map((b) => ({ ...b }));
    state.dirty = true;
    drawOverlay();
    note(`${body.note} — review, then Save. Nothing is written until you do.`);
  } catch (error) {
    note(String(error.message));
  }
}

function clear() {
  state.boxes = [];
  state.dirty = true;
  drawOverlay();
  note("cleared");
}

async function save(nullAnnotation = false, confirmed = false) {
  try {
    const body = await api(`/api/frame/${state.name}/labels`, {
      method: "POST",
      body: JSON.stringify({
        boxes: nullAnnotation ? [] : state.boxes,
        null: nullAnnotation,
        // The one thing a review pass needs and an ordinary save cannot say: these boxes are a
        // weight's, a person checked them, and they are right *unchanged*. Without it, agreeing
        // with a model leaves the frame machine-only for ever - the rows still equal the
        // suggestion, so nothing in the files could tell.
        confirmed,
      }),
    });
    state.dirty = false;
    const index = state.index;
    await loadFrames();
    // Clamped, because a save can shorten the list: a frame stops being machine-only the moment
    // its boxes are checked, so in a review pass it leaves the list under the cursor. Re-opening
    // that index is what makes the next unreviewed frame arrive without a keypress; the clamp is
    // what stops the empty-list case from being an error.
    await open(Math.min(index, Math.max(0, state.frames.length - 1)));
    note(
      `saved ${body.provenance.rows} box(es) by ${body.provenance.provider}` +
        (body.provenance.machine_only
          ? " — still machine-only: edit a box to make it yours, or press Confirm (f) to say it is right"
          : "")
    );
  } catch (error) {
    note(String(error.message));
  }
}

// -- wiring --------------------------------------------------------------------

function next(step) {
  open(state.index + step);
}

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT") return;
  const key = event.key;
  if (key >= "1" && key <= "9") {
    const index = Number(key) - 1;
    if (index < (state.config?.classes.length ?? 0)) selectClass(index);
    return;
  }
  const map = {
    n: () => save(true),
    s: () => suggest(),
    c: () => clear(),
    a: () => suggest(),
    u: () => toggleReview(),
    d: () => toggleDraw(),
    m: () => toggleNulls(),
    f: () => save(false, true),
    Enter: () => save(false),
    ArrowRight: () => next(1),
    ArrowLeft: () => next(-1),
  };
  if (key === "Backspace" || key === "Delete") {
    state.boxes.pop();
    state.dirty = true;
    drawOverlay();
    return;
  }
  const action = map[key];
  if (action) {
    event.preventDefault();
    action();
  }
});

// Drag on the image to draw a box in the selected class. Wired here rather than inside
// `beginDrag` so the listener exists as soon as the page does - the overlay is the thing under
// the pointer (the boxes are `pointer-events: none`), and it is exactly the image's rectangle.
$("overlay").addEventListener("mousedown", (event) => {
  if (event.button === 0) beginDrag(event);
});

document.addEventListener("click", (event) => {
  const action = event.target.dataset?.act;
  if (!action) return;
  if (action === "review") toggleReview();
  if (action === "draw") toggleDraw();
  if (action === "nulls") toggleNulls();
  if (action === "gate") showGate();
  if (action === "confirm") save(false, true);
  if (action === "suggest" || action === "accept") suggest();
  if (action === "clear") clear();
  if (action === "null") save(true);
  if (action === "save") save(false);
  if (action === "next") next(1);
  if (action === "prev") next(-1);
});

// Any filter change reloads from the server. All three are read at once rather than each control
// mapping to its own field, so a fourth filter cannot be added and forgotten in one of the two
// places.
["filter-split", "filter-distance", "filter-state"].forEach((id) => {
  $(id).addEventListener("change", () => {
    state.split = $("filter-split").value;
    state.distance = $("filter-distance").value;
    state.filterState = $("filter-state").value;
    refresh();
  });
});

(async function main() {
  await loadConfig();
  selectClass(0);
  await loadFrames();
  if (state.frames.length) await open(0);
})();
