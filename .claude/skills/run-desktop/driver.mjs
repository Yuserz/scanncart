// Playwright driver for the SCANnCART Electron desktop app (Windows host).
//
// Usage (from repo root, after `cd desktop && npm run build`):
//   node .claude/skills/run-desktop/driver.mjs smoke      # launch, screenshot Live + Admin
//   node .claude/skills/run-desktop/driver.mjs capture    # full Start -> frames -> Stop flow
//   node .claude/skills/run-desktop/driver.mjs allowlist  # class allowlist field -> sidecar round-trip
//   node .claude/skills/run-desktop/driver.mjs dataset    # dataset labeling panel <-> sidecar snapshot
//   node .claude/skills/run-desktop/driver.mjs models     # installed weights + resize_mode check <-> sidecar
//   node .claude/skills/run-desktop/driver.mjs probe      # Test Connection against a stub workflow
//
// Modes that assert print PASS/FAIL per check and exit non-zero if any fail,
// so they can gate a change. `allowlist` restores the sidecar's original
// setting before exiting, including when a check throws.
//
// Screenshots land in <repo>/.claude/skills/run-desktop/shots/ (gitignored),
// override with SCREENSHOT_DIR.
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';

const SKILL_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SKILL_DIR, '..', '..', '..');
const APP_DIR = path.join(REPO_ROOT, 'desktop');

// playwright-core is a devDependency of desktop/ — resolve it from there, a
// bare import fails when this script runs from outside desktop/node_modules.
const require = createRequire(path.join(APP_DIR, 'package.json'));
const { _electron: electron } = require('playwright-core');

const SHOT_DIR = process.env.SCREENSHOT_DIR || path.join(SKILL_DIR, 'shots');
fs.mkdirSync(SHOT_DIR, { recursive: true });
const electronBin = path.join(APP_DIR, 'node_modules', 'electron', 'dist', 'electron.exe');

const mode = process.argv[2] || 'smoke';
if (!['smoke', 'capture', 'allowlist', 'dataset', 'models', 'classlist', 'probe'].includes(mode)) {
  console.error(
    `unknown mode '${mode}' — use: smoke | capture | allowlist | dataset | models | classlist | probe`
  );
  process.exit(1);
}

// Screenshots are diagnostics, not assertions. This machine has GPU resets and
// intermittent compositor stalls (see SKILL.md), and an unguarded screenshot that
// throws takes every PASS/FAIL line below it down with it — which is how a flaky
// capture turns into a false report that the feature is broken. Retry once with
// animations disabled (the loading ring spins forever, and a screenshot waits for
// the page to settle), then warn and carry on.
const shot = async (page, name) => {
  const f = path.join(SHOT_DIR, `${name}.png`);
  let why = '';
  for (const opts of [{}, { animations: 'disabled' }]) {
    try {
      await page.screenshot({ path: f, timeout: 20_000, ...opts });
      console.log('screenshot:', f);
      return;
    } catch (err) {
      why = String(err?.message ?? err).split('\n')[0];
    }
  }
  console.log(`screenshot FAILED (${why}) — continuing, this is not an assertion`);
};

// Assertions for modes that verify behaviour rather than just launching.
// Collected rather than thrown, so one failure still runs the remaining checks
// (and the settings restore in `finally`) — the process exits non-zero at the
// end, which is what makes a mode usable as a gate.
const failures = [];
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(label);
};

// Launch with APP_DIR (not out/main/index.js): package.json "main" points at
// the built output, and app.getAppPath() must be desktop/ for the main process
// to resolve ../sidecar for the venv python + run.py.
//
// Retry loop: the sidecar has no auto-restart, and if its Python dies during
// native DLL imports (this machine intermittently kills processes that way —
// LiveKernelEvent 141 GPU resets in the event log), the renderer polls for a
// port forever and nav-live never appears.
let app = null;
let page = null;
for (let attempt = 1; attempt <= 4; attempt++) {
  app = await electron.launch({ executablePath: electronBin, args: [APP_DIR], cwd: APP_DIR, timeout: 30_000 });
  page = await app.firstWindow();
  try {
    await page.waitForSelector('[data-testid="nav-live"]', { timeout: 45_000 });
    console.log(`attempt ${attempt}: AppShell mounted — sidecar port received`);
    break;
  } catch {
    console.log(`attempt ${attempt}: sidecar never reported a port (likely died on import) — relaunching`);
    await app.close().catch(() => {});
    app = null;
    if (attempt === 4) throw new Error('sidecar failed to start in 4 attempts — check the machine, see SKILL.md gotchas');
  }
}

const stateText = () => page.textContent('[data-testid="state"]');
const connText = () => page.textContent('[data-testid="conn"]');
// For *diagnostic* lines only, which run precisely when something is wrong: a log line that throws
// on the condition it exists to describe turns a reported failure into a crashed run.
const readText = (sel) => page.textContent(sel, { timeout: 2_000 }).catch(() => null);
const bodyText = () =>
  page
    .evaluate(() => (document.body?.innerText ?? '').slice(0, 400))
    .catch(() => '(unreadable)');
await page.waitForTimeout(2_000);
// Diagnostic reads, not assertions: a run that launches into an unexpected view should say so and
// carry on, rather than dying here with nothing on screen recorded.
console.log(
  'state:',
  await readText('[data-testid="state"]'),
  '| ws:',
  await readText('[data-testid="conn"]'),
  '| nav:',
  await page
    .evaluate(() => ({
      live: document.querySelector('[data-testid="nav-live"]')?.getAttribute('aria-pressed'),
      admin: document.querySelector('[data-testid="nav-admin"]')?.getAttribute('aria-pressed')
    }))
    .catch(() => '(unreadable)'),
  '| page:',
  JSON.stringify(await bodyText())
);
await shot(page, `${mode}-01-live-idle`);

if (mode === 'smoke') {
  // Admin panel — proves the sidecar REST API is answering.
  await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
  await page.waitForSelector('[data-testid="hardware-info"]', { timeout: 15_000 });
  const hw = await page.evaluate(() => document.querySelector('[data-testid="hardware-info"]').innerText);
  console.log('hardware info:\n' + hw);
  await shot(page, 'smoke-02-admin');
}

if (mode === 'allowlist') {
  // The class allowlist is an Admin-only field: the Live tuning card renders
  // numeric sliders, so a list field grouped on the live side draws as a range
  // input bound to a string array (that shipped once). This is the wiring check
  // the unit tests cannot make — real keystrokes through the real renderer
  // landing on the real sidecar.
  const port = await page.evaluate(() => window.api.getSidecarPort());
  const api = (p, init) =>
    page.evaluate(
      ([pt, i]) =>
        fetch(`http://127.0.0.1:${pt}${i.path}`, i).then(async (r) => ({
          status: r.status,
          body: await r.json()
        })),
      [port, { path: p, ...init }]
    );
  // The save is a round-trip through the WS-backed store, so poll rather than
  // guess at a fixed delay.
  const waitForAllowlist = async (expected) => {
    for (let i = 0; i < 24; i++) {
      const body = (await api('/api/settings')).body;
      if (JSON.stringify(body.class_allowlist) === JSON.stringify(expected)) return body;
      await page.waitForTimeout(250);
    }
    return (await api('/api/settings')).body;
  };

  const original = (await api('/api/settings')).body.class_allowlist;
  console.log('sidecar port:', port, '| original class_allowlist:', JSON.stringify(original));

  try {
    // Expand the tuning card before looking: its contents are `hidden` rather
    // than unmounted, so asserting against the collapsed card would pass for
    // the wrong reason the day it becomes conditional.
    await page.evaluate(() => document.querySelector('.tuning-toggle')?.click());
    await page.waitForTimeout(300);
    check(
      'allowlist is absent from the Live tuning card',
      !(await page.evaluate(() => !!document.querySelector('#tune-class_allowlist')))
    );

    await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
    await page.waitForSelector('#class_allowlist', { timeout: 15_000 });
    const meta = await page.evaluate(() => {
      const el = document.querySelector('#class_allowlist');
      el.scrollIntoView({ block: 'center' });
      return { type: el.type, placeholder: el.placeholder };
    });
    check(
      'Admin renders it as a text input',
      meta.type === 'text',
      `type=${meta.type} placeholder=${JSON.stringify(meta.placeholder)}`
    );
    await shot(page, 'allowlist-01-admin-field');

    await page.click('#class_allowlist');
    await page.keyboard.press('Control+A');
    await page.keyboard.type('bottle, cup', { delay: 30 });
    const typed = await page.evaluate(() => document.querySelector('#class_allowlist').value);
    // Parsing on every keystroke used to eat the comma before the next name
    // began, so this arrived at the sidecar as ["bottlecupp"].
    check('typing keeps multi-name text verbatim', typed === 'bottle, cup', JSON.stringify(typed));
    await shot(page, 'allowlist-02-typed');

    await page.evaluate(() => document.querySelector('[data-testid="save-settings"]').click());
    const saved = await waitForAllowlist(['bottle', 'cup']);
    await shot(page, 'allowlist-03-saved');

    check(
      'Save reaches the sidecar as a parsed array',
      JSON.stringify(saved.class_allowlist) === JSON.stringify(['bottle', 'cup']),
      JSON.stringify(saved.class_allowlist)
    );
    check(
      'sidecar treats it as hot-reloadable',
      saved.hot_reloadable_fields?.includes('class_allowlist') === true
    );
    check(
      'sidecar does not require a restart for it',
      saved.restart_required_fields?.includes('class_allowlist') !== true
    );
  } finally {
    // Never leave this machine on the test allowlist, even if a check threw.
    await api('/api/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ class_allowlist: original })
    });
    const restored = await waitForAllowlist(original);
    check(
      'original allowlist restored',
      JSON.stringify(restored.class_allowlist) === JSON.stringify(original),
      JSON.stringify(restored.class_allowlist)
    );
  }
}

if (mode === 'dataset') {
  // The Admin Panel's dataset section is a read-only render of a local snapshot file.
  // The join worth checking is that what is ON SCREEN equals the sidecar's own JSON:
  // a unit test can see either half of that but not both. Also asserts the section
  // exists at all, since a render error there takes the whole Admin view down.
  await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
  // AdminPanel early-returns a spinner ("Loading settings…") or a retry view while
  // /api/settings is still in flight, and neither contains this section — nor
  // hardware-info. Waiting on one selector therefore reports "the panel is missing"
  // for what is really "the panel has not loaded yet". Wait for it to settle, and
  // say which state it settled in.
  // Generous on purpose: the panel must wait for the SAME Promise.all that includes
  // /api/system-info, which lazily imports torch on its first call. Each driver run
  // spawns a fresh sidecar, so that import is cold every time and can take tens of
  // seconds on this machine.
  await page
    .waitForSelector(
      '[data-testid="dataset-progress"], [data-testid="admin-error"], .admin-loading',
      { timeout: 120_000 }
    )
    .catch(() => {});
  // Settled means "no longer the spinner", whatever it settled on.
  await page
    .waitForSelector('.admin-loading', { state: 'detached', timeout: 120_000 })
    .catch(() => {});
  const panelState = await page.evaluate(() => {
    if (document.querySelector('[data-testid="dataset-progress"]')) return 'ready';
    if (document.querySelector('[data-testid="admin-error"]')) return 'error';
    return 'loading';
  });
  if (panelState !== 'ready') {
    console.log('body text:\n' + (await page.evaluate(() => document.body.innerText)).slice(0, 800));
  }
  check('admin panel settled into its loaded state', panelState === 'ready', `panel is '${panelState}'`);

  const panelText = await page.evaluate(
    () => document.querySelector('[data-testid="dataset-progress"]')?.innerText ?? ''
  );
  console.log('dataset panel:\n' + panelText);
  await shot(page, 'dataset-01-admin');

  const port = await page.evaluate(() => window.api.getSidecarPort());
  const status = await page.evaluate(
    (p) => fetch(`http://127.0.0.1:${p}/api/dataset/status`).then((r) => r.json()),
    port
  );
  console.log(
    'sidecar says:',
    JSON.stringify({
      available: status.available,
      decided: status.decided,
      total: status.total,
      classes: status.classes.length,
      age_seconds: status.age_seconds,
      tier_a_target: status.tier_a_target,
      tier_a_remaining: status.tier_a_remaining,
      tier_a_cells_under_target: status.tier_a_cells_under_target,
      sessions: (status.sessions ?? []).map((s) => `${s.name}:${s.train}/${s.valid}/${s.test}`)
    })
  );

  check('dataset section rendered', panelText.trim().length > 0);
  if (panelState !== 'ready') {
    // Nothing below can be judged from a panel that never loaded; the check above
    // already failed and named the state, so stop rather than pile on.
  } else if (!status.available) {
    // The first-run state: nobody has run the tool on this machine. Showing "0 / 0
    // labeled" here would read as "nothing is done" rather than "nothing has looked".
    check(
      'no snapshot reads as instructions, not as zeros',
      panelText.includes('label_progress.py') && !panelText.includes('0 / 0'),
      panelText.split('\n').slice(0, 2).join(' / ')
    );
  } else {
    check(
      'panel totals match the sidecar',
      panelText.includes(`${status.decided} / ${status.total}`),
      `expected "${status.decided} / ${status.total}" on screen`
    );
    // The labeling worklist. Checked cell for cell AND in order, because the order is the
    // information: row 1 is the next cell to open in Roboflow, and the right rows in the
    // wrong order send someone at the wrong cell first. The sidecar ranks it, the panel
    // renders it as it arrived, and this is the only place both halves are visible.
    const backlog = status.labeling_backlog ?? [];
    const onScreen = await page.evaluate(() => {
      const list = document.querySelector('[data-testid="dataset-backlog"]');
      if (!list) return null;
      return [...list.querySelectorAll('li')].map((li) => ({
        rank: li.querySelector('.admin-backlog-rank')?.textContent?.trim() ?? '',
        name: li.querySelector('.admin-backlog-name')?.textContent?.trim() ?? '',
        where: li.querySelector('.admin-backlog-where')?.textContent?.trim() ?? '',
        left: li.querySelector('[data-testid="dataset-backlog-left"]')?.textContent?.trim() ?? '',
        counts: li.querySelector('.admin-backlog-of')?.textContent?.trim() ?? '',
        nullRule: li.querySelector('[data-testid="dataset-backlog-null"]') !== null
      }));
    });
    console.log(
      'worklist on screen:\n' +
        (onScreen ?? [])
          .map((r) => `${r.rank} ${r.name} ${r.where} — ${r.left} left (${r.counts})${r.nullRule ? ' [null]' : ''}`)
          .join('\n')
    );
    await shot(page, 'dataset-02-worklist');
    check(
      'one row per cell still waiting for a label',
      onScreen !== null && onScreen.length === backlog.length,
      `${onScreen?.length ?? 'no list'} row(s) vs ${backlog.length} cell(s) left`
    );
    // Names plus distance, in order: the pair identifies a cell, so this is the whole ordering
    // claim rather than a count that a reshuffle would still satisfy.
    const expectedOrder = backlog.map(
      (c) => `${c.name}@${c.background ? 'background frames' : c.distance}`
    );
    check(
      'ranked the way the sidecar ranked them, biggest first',
      JSON.stringify((onScreen ?? []).map((r) => `${r.name}@${r.where}`)) === JSON.stringify(expectedOrder),
      `screen ${JSON.stringify((onScreen ?? []).map((r) => `${r.name}@${r.where}`).slice(0, 3))}… vs sidecar ${JSON.stringify(expectedOrder.slice(0, 3))}…`
    );
    check(
      'and numbered, so "work down the list" is followable',
      (onScreen ?? []).every((r, i) => r.rank === String(i + 1)),
      JSON.stringify((onScreen ?? []).map((r) => r.rank))
    );
    // `left` is derived by the sidecar from the pair printed beside it, so both halves being
    // read off one row is what proves the number on screen is the one it computed.
    check(
      'each row carries its own remaining count and the pair it came from',
      (onScreen ?? []).every((r, i) => r.left === String(backlog[i].remaining) && r.counts === `${backlog[i].decided}/${backlog[i].total}`),
      JSON.stringify((onScreen ?? [])[0] ?? null)
    );
    // The one row whose instruction is the opposite of every other row's: those frames are
    // marked null, not drawn on. Flagging a product cell this way would cost a labeling session.
    const flagged = (onScreen ?? []).filter((r) => r.nullRule).length;
    const background = backlog.filter((c) => c.background).length;
    check(
      'only the hard negatives carry the null-marking rule',
      flagged === background,
      `${flagged} flagged vs ${background} background cell(s)`
    );
    // A finished backlog is the one state this machine cannot reach, so it is checked by what
    // the panel does rather than by setting it up: no list, and the empty line instead.
    if (backlog.length > 0) {
      check(
        'and the worklist summary counts the cells that are left',
        panelText.includes(`${backlog.length} cell(s) left`),
        `expected "${backlog.length} cell(s) left" on screen`
      );
    } else {
      check(
        'an empty worklist says so instead of listing nothing',
        onScreen === null && panelText.includes('Nothing left to label'),
        panelText.split('\n').slice(-2).join(' / ')
      );
    }
    // A fresh snapshot says "(just now)" rather than "... ago", so match the whole
    // shape: the line must name the snapshot's timestamp AND qualify its age.
    check(
      'staleness is stated, so the numbers cannot read as live',
      /snapshot from .*\((just now|[^)]*ago|age unknown|unknown time)\)/.test(panelText),
      panelText.match(/snapshot from[^\n]*/)?.[0] ?? '(no freshness line)'
    );

    // The Tier A capture gap. In the app rather than only in the doc, because a cell with
    // no images in it is a camera job and reading it as labeling backlog is what wastes a
    // capture session.
    if (status.tier_a_target > 0) {
      const gap = await page.evaluate(() => {
        const summary = document.querySelector('[data-testid="dataset-tier-a-summary"]');
        const list = document.querySelector('[data-testid="dataset-tier-a-cells"]');
        return {
          summary: summary?.innerText ?? null,
          cells: list ? list.querySelectorAll('li').length : -1
        };
      });
      check('capture gap section rendered', gap.summary !== null, gap.summary ?? '(absent)');
      check(
        'capture gap totals match the sidecar',
        gap.summary !== null &&
          gap.summary.includes(String(status.tier_a_remaining)) &&
          gap.summary.includes(`of ${status.tier_a_target} images`),
        `expected "${status.tier_a_remaining} of ${status.tier_a_target} images" on screen`
      );
      check(
        'one row per cell still under target',
        gap.cells === status.tier_a_cells_under_target,
        `${gap.cells} row(s) vs ${status.tier_a_cells_under_target} under target`
      );
    } else {
      const block = await page.evaluate(
        () => document.querySelector('[data-testid="dataset-tier-a"]') !== null
      );
      check('no capture gap block when the snapshot carries none', block === false);
    }

    // The capture-session spread, checked the same way: on-screen must equal what the
    // sidecar served. This block is what tells an operator whether the test number they
    // are about to quote is an unseen session or held-out frames of a session the model
    // trained on, so a silent mismatch here is a wrong claim in a report.
    if ((status.sessions ?? []).length > 0) {
      const sessions = await page.evaluate(() => {
        const t = document.querySelector('[data-testid="dataset-session-rows"]');
        const rows = t ? [...t.querySelectorAll('tbody tr')] : null;
        return {
          rows: rows ? rows.map((r) => r.innerText.replace(/\s+/g, ' ').trim()) : null,
          shared: t ? t.querySelectorAll('tbody tr.dataset-session-shared').length : -1,
          verdict: document.querySelector('[data-testid="dataset-session-verdict"]')?.innerText ?? null
        };
      });
      check(
        'one row per capture session in the snapshot',
        sessions.rows?.length === status.sessions.length,
        `${sessions.rows?.length ?? 'no table'} row(s) vs ${status.sessions.length} session(s)`
      );
      // Exact, cell for cell: the counts on screen are the whole point of the block, and
      // a row that renders a 0 as a dash is the difference between "none here" and
      // "nothing at all".
      const expected = status.sessions.map(
        (s) => `${s.name} ${s.train || '\u2014'} ${s.valid || '\u2014'} ${s.test || '\u2014'} ${s.splits}`
      );
      check(
        'every session row matches the sidecar exactly',
        JSON.stringify(sessions.rows) === JSON.stringify(expected),
        `screen ${JSON.stringify(sessions.rows)} vs sidecar ${JSON.stringify(expected)}`
      );
      const straddling = status.sessions.filter((s) => s.train > 0 && s.test > 0);
      check(
        'only sessions on both sides of the train/test line are marked',
        sessions.shared === straddling.length,
        `${sessions.shared} marked vs ${straddling.length} straddling`
      );
      check(
        'the verdict says which reading the test number supports',
        straddling.length
          ? straddling.every((s) => (sessions.verdict ?? '').includes(s.name))
          : /unseen capture session/.test(sessions.verdict ?? ''),
        sessions.verdict?.split('\n')[0] ?? '(absent)'
      );
    } else {
      const block = await page.evaluate(
        () => document.querySelector('[data-testid="dataset-sessions"]') !== null
      );
      check('no session block when the snapshot carries none', block === false);
    }
  }
}

if (mode === 'models') {
  // The Model field's "what each installed weight needs" block, and the mismatch warning it
  // feeds. The join worth checking is the same one `dataset` checks: what is ON SCREEN equals
  // the sidecar's own /api/models JSON. A unit test can see either half of that but not both,
  // and the failure this guards against — a requirement nobody sees until the `far` cells are
  // weak — is invisible in exactly that gap.
  //
  // Nothing is installed on this machine yet (the v2 weights do not exist), so the mode makes
  // a scratch weight and removes it in `finally`. The record is written by
  // `train_model.weight_record` rather than by hand here: the filename-and-shape pairing between
  // that tool and `app/models.py` is precisely what is at stake, and a driver that invented
  // its own JSON would agree with the reader by construction.
  const SIDECAR = path.join(REPO_ROOT, 'sidecar');
  const MODELS = path.join(SIDECAR, 'models');
  const python = process.env.SIDECAR_PYTHON || path.join(SIDECAR, '.venv', 'Scripts', 'python.exe');
  const SCRATCH = 'driver-scratch';
  const weightPath = path.join(MODELS, `${SCRATCH}.pt`);
  const recordPath = path.join(MODELS, `${SCRATCH}.json`);

  let port = null;
  let original = null;
  try {
    fs.mkdirSync(MODELS, { recursive: true });
    // The block reads the directory listing and the record, never the weights, so the bytes
    // only have to exist. Copied from the stock checkpoint when it is there so nothing in the
    // picker is obviously fake; it may not be, since those weights are untracked and only
    // arrive on first capture.
    const stock = path.join(SIDECAR, 'yolo11n.pt');
    if (fs.existsSync(stock)) fs.copyFileSync(stock, weightPath);
    else fs.writeFileSync(weightPath, Buffer.alloc(1024));
    // The measurement is written by `validation_record` too - the real writer for the block -
    // with the three outcomes in it: a pass, a miss, and a class the split held no instances
    // of. The last one is why the block cannot be a flat list of numbers, and the panel has to
    // render `null` differently from `0.0`, so a scratch record without it would not exercise
    // the thing most likely to be wrong.
    const record = JSON.parse(
      execFileSync(
        python,
        [
          '-c',
          `import json, train_model, generations
rows = [
    ("bear-brand", 0.9, 10),
    ("century-tuna", 0.62, 30),
    ("lucky-me", None, 0),
    ("milo", 0.95, 12),
]
block = train_model.validation_record(
    rows, "test",
    {"precision": 0.9, "recall": 0.82, "mAP50": 0.88, "mAP50-95": 0.61},
    train_model.RECALL_FLOOR,
)
# The per-distance breakdown, built by the same helpers the real runs use
# (class_rows, and rows shaped like per_class_recall's). Two distances, because the
# grid's whole claim is a comparison: one where a class misses the floor while the
# split reads as a pass, and one where a class is unmeasured (None, which has to
# render as a dash rather than a zero).
block["per_distance"] = [
    {
        "distance": "close",
        "images": 34,
        "aggregates": {"mAP50": 0.93},
        "per_class": train_model.class_rows(
            [("bear-brand", 0.97, 12), ("century-tuna", 0.88, 20), ("lucky-me", None, 0)]
        ),
    },
    {
        "distance": "far",
        "images": 28,
        "aggregates": {"mAP50": 0.51},
        "per_class": train_model.class_rows([("bear-brand", 0.55, 8)]),
    },
]
print(json.dumps(train_model.weight_record(generations.V2, 2, "snc-grocery", validation=[block])))`
        ],
        { cwd: path.join(SIDECAR, 'tools'), encoding: 'utf8' }
      )
        .trim()
        .split('\n')
        .pop()
    );
    fs.writeFileSync(recordPath, JSON.stringify(record, null, 2) + '\n');
    console.log('scratch record written by train_model.weight_record:', JSON.stringify(record));

    port = await page.evaluate(() => window.api.getSidecarPort());
    const api = (p, init) =>
      page.evaluate(
        ([pt, i]) =>
          fetch(`http://127.0.0.1:${pt}${i.path}`, i).then(async (r) => ({
            status: r.status,
            body: await r.json()
          })),
        [port, { path: p, ...init }]
      );
    const patch = (body) =>
      api('/api/settings?persist=false', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

    const listed = (await api('/api/models')).body.installed;
    console.log('sidecar installed:', JSON.stringify(listed));
    const mine = listed.find((m) => m.value === `models/${SCRATCH}.pt`);
    check('the sidecar reports the scratch weight', !!mine, JSON.stringify(mine));
    check(
      'with the requirement the tool recorded',
      mine?.resize_mode === 'stretch',
      `resize_mode=${JSON.stringify(mine?.resize_mode)}`
    );
    check(
      'and that `auto` answers with that same requirement',
      mine?.auto_resolves_to === mine?.resize_mode,
      `auto_resolves_to=${JSON.stringify(mine?.auto_resolves_to)}`
    );

    // The state that used to be silently wrong: select the weights, leave resize_mode on
    // `auto`. `persist=false` so this never reaches the machine's settings.json, and the
    // whole thing is reverted in `finally` anyway.
    original = (await api('/api/settings')).body;
    const patched = await patch({ active_model: `models/${SCRATCH}.pt`, resize_mode: 'auto' });
    check('the sidecar accepted the patch', patched.status === 200, `HTTP ${patched.status}`);

    // AdminPanel mounts on this click, so it reads the settings *after* the patch — no reload.
    await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
    await page
      .waitForSelector('#resize_mode', { timeout: 120_000 })
      .catch(() => {})
      .then(() => page.waitForSelector('.admin-loading', { state: 'detached', timeout: 120_000 }))
      .catch(() => {});

    const block = await page.evaluate(() => {
      const el = document.querySelector('[data-testid="installed-models"]');
      return el ? el.innerText : null;
    });
    console.log('installed block:\n' + block);
    await shot(page, 'models-01-admin');
    check('the panel lists what is on disk', block !== null);
    check(
      'naming the weight and the mode it needs',
      !!block && block.includes(`models/${SCRATCH}.pt`) && block.includes('resize_mode: stretch'),
      JSON.stringify(block)
    );
    check(
      'and where it came from',
      !!block && block.includes(record.source ?? 'snc-grocery version 2'),
      JSON.stringify(block)
    );
    // Renderer-only text: the sidecar cannot make this pass, so it also proves the bundle is
    // the one just built. The `auto` check below would otherwise pass against a stale
    // renderer too, since its old formula re-derived the answer from `auto_resolves_to`.
    check(
      'and says `auto` will use it',
      !!block && /auto\s+uses it/.test(block),
      JSON.stringify(block)
    );

    // The measurement travels the whole way: `train_model.weight_record(2, 'snc-grocery', [block])`
    // wrote it beside the scratch weight, the sidecar read it back, and the panel renders it.
    // Nothing here re-states a number the writer did not produce.
    check(
      'the sidecar reports the measured recall',
      mine?.validation?.[0]?.split === 'test' && mine?.validation?.[0]?.per_class?.length > 0,
      JSON.stringify(mine?.validation)
    );
    const scored = await page.evaluate(
      () => document.querySelector('[data-testid="model-validation"]')?.innerText ?? null
    );
    console.log('validation block:\n' + scored);
    await shot(page, 'models-01b-validation');
    check('the panel renders a score block for the selected weight', scored !== null);
    check(
      'naming the split and the floor it was judged against',
      !!scored && scored.includes('test') && scored.includes('floor 0.85'),
      JSON.stringify(scored)
    );
    check(
      'per class, with the count beside each number',
      !!scored && /milo 0\.950 \(n=12\)/.test(scored) && /century-tuna 0\.620/.test(scored),
      JSON.stringify(scored)
    );
    // The distinction the whole block exists for: a class the split never asked about must not
    // render as a zero. Ultralytics answers 0.0 for it, and 0.0 sends the operator after
    // images of an item when what is missing is captures in that split.
    check(
      'and naming an unmeasured class instead of printing a zero for it',
      !!scored && scored.includes('lucky-me') && scored.includes('not measured'),
      JSON.stringify(scored)
    );
    check(
      'with no zero anywhere near it',
      !!scored && !scored.includes('lucky-me' + ' 0.000'),
      JSON.stringify(scored)
    );

    // The class x distance grid. Same join as everything else in this mode - what the panel
    // renders equals what the sidecar served - and the one the per-class list cannot make: the
    // split's own number is a mean over distances, so a class can pass it while missing the
    // floor at `far`, which is the bucket this dataset exists to fix.
    const distances = await page.evaluate(() => {
      const grid = document.querySelector('[data-testid="model-validation-by-distance"]');
      if (!grid) return null;
      const cells = {};
      for (const cell of grid.querySelectorAll('td[data-testid]')) {
        cells[cell.getAttribute('data-testid')] = {
          text: cell.innerText.trim(),
          state: cell.className
        };
      }
      return {
        head: grid.querySelector('thead')?.innerText.replace(/\s+/g, ' ').trim() ?? '',
        cells,
        misses:
          document.querySelector('[data-testid="model-validation-distance-misses"]')?.innerText ??
          null
      };
    });
    console.log('by distance:\n' + JSON.stringify(distances, null, 1));
    await shot(page, 'models-01c-by-distance');
    check('the panel renders the per-distance breakdown', distances !== null);
    // The frame count per column, so 1.000 over three images cannot read as a stronger claim
    // than 0.900 over sixty - and it is the sidecar's number, not one the renderer guessed.
    check(
      'naming each distance and how many frames it ran on',
      !!distances && /close 34 img/.test(distances.head) && /far 28 img/.test(distances.head),
      distances?.head ?? '(no grid)'
    );
    // The claim the feature exists for: the same class, passing on the split and failing at a
    // distance, side by side.
    check(
      'a far miss beside the split number that hides it',
      distances?.cells['model-validation-far-bear-brand']?.text === '0.550' &&
        distances?.cells['model-validation-far-bear-brand']?.state.includes('below') &&
        distances?.cells['model-validation-close-bear-brand']?.text === '0.970',
      JSON.stringify(distances?.cells['model-validation-far-bear-brand'] ?? null)
    );
    check(
      'and names it as a to-do rather than leaving it to a colour',
      /Below the floor at a distance: far bear-brand 0\.550/.test(distances?.misses ?? ''),
      distances?.misses ?? '(no line)'
    );
    // Two kinds of absent cell, one rendering: a distance that scored a class and held no
    // instances of it, and a distance with no frames of it at all.
    check(
      'a dash for a distance that held no instances, and no zero',
      distances?.cells['model-validation-close-lucky-me']?.text === '—' &&
        distances?.cells['model-validation-far-lucky-me']?.text === '—' &&
        distances?.cells['model-validation-close-lucky-me']?.state.includes('unmeasured') &&
        !JSON.stringify(distances?.cells).includes('0.000'),
      JSON.stringify(distances?.cells['model-validation-close-lucky-me'] ?? null)
    );

    const readMismatch = () =>
      page.evaluate(
        () => document.querySelector('[data-testid="model-resize-mismatch"]')?.innerText ?? null
      );

    // The acceptance criterion for this feature: leaving the field alone is *correct* now, so
    // there is nothing to warn about. This check is the inverse of what it used to be, and
    // that is the whole point - the app told the operator to go and fix a configuration that
    // was already right.
    const mismatchText = await readMismatch();
    console.log('mismatch warning on `auto`:', JSON.stringify(mismatchText));
    check('leaving it on `auto` is no longer flagged', mismatchText === null, JSON.stringify(mismatchText));

    // The only way to get the geometry wrong from the panel now, so it has to still be caught.
    await page.selectOption('#resize_mode', 'letterbox');
    await page.waitForTimeout(200);
    const overridden = await readMismatch();
    console.log('mismatch warning on an explicit override:', JSON.stringify(overridden));
    check(
      'an explicit value contradicting the record is flagged',
      !!overridden && overridden.includes('letterbox') && overridden.includes('stretch'),
      JSON.stringify(overridden)
    );
    await shot(page, 'models-02-overridden');

    // Clearing it is the check's own control: a warning that never went away would have
    // passed the check above for the wrong reason.
    await page.selectOption('#resize_mode', 'stretch');
    await page.waitForTimeout(200);
    const cleared = await readMismatch();
    check('choosing the recorded requirement clears it', cleared === null, JSON.stringify(cleared));
    await shot(page, 'models-03-matched');

    // The Live view's readout: the same two facts, in the strip where the operator is looking
    // while a capture runs. It is deliberately *not* gated on capture running, which is what makes
    // it checkable on a machine with no camera - and the first thing it proves is the whole `auto`
    // story end to end: the sidecar resolved the setting against the record, and the renderer
    // printed that answer instead of the word "auto".
    //
    // The saved setting is still `auto` here (the draft edits above never left the Admin form), so
    // this is the unmismatched reading.
    await page.evaluate(() => document.querySelector('[data-testid="nav-live"]').click());
    await page.waitForSelector('[data-testid="state"]');
    await page.waitForSelector('[data-testid="stat-geometry"]', { timeout: 30_000 });
    const readout = await page.evaluate(() => ({
      geometry: document.querySelector('[data-testid="stat-geometry"]')?.innerText ?? null,
      geometryWarn:
        document.querySelector('[data-testid="stat-geometry"]')?.classList.contains('warn') ?? null,
      recall: document.querySelector('[data-testid="stat-recall"]')?.innerText ?? null
    }));
    console.log('live readout:', JSON.stringify(readout));
    await shot(page, 'models-04a-live-readout');
    check(
      'the Live strip prints the geometry `auto` resolves to, and says it came from `auto`',
      !!readout.geometry && readout.geometry.includes('stretch') && readout.geometry.includes('auto'),
      JSON.stringify(readout.geometry)
    );
    check(
      'and the measured recall beside it, with the split that was measured',
      !!readout.recall && readout.recall.includes('82%') && readout.recall.includes('test recall'),
      JSON.stringify(readout.recall)
    );
    check(
      'and does not flag a geometry the record agrees with',
      readout.geometryWarn === false,
      `warn=${readout.geometryWarn}`
    );

    // The Live view's copy of the warning, gated on capture actually running - and the one
    // property of that banner this machine can check without a camera (its *appearance* needs a
    // running capture and is covered in LiveView.test.tsx).
    //
    // The saved setting is put into genuine disagreement with the record first. It has to be: the
    // draft edits above never leave the Admin form, so before this patch the only mismatch was in
    // an unsaved draft and the check below passed because there was nothing to warn about, not
    // because of the gate.
    const mismatched = await patch({ resize_mode: 'letterbox' });
    check(
      'the sidecar accepted a saved setting that contradicts the record',
      mismatched.status === 200,
      `HTTP ${mismatched.status}`
    );
    // Remount the Live view rather than waiting: the readout reads the settings once on mount and
    // does not poll (the answer can only change when someone edits a restart-required field), so a
    // fresh mount is what the new reading actually needs.
    await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
    await page.waitForSelector('[data-testid="nav-live"]');
    await page.evaluate(() => document.querySelector('[data-testid="nav-live"]').click());
    await page.waitForSelector('[data-testid="state"]');
    await page.waitForFunction(
      () =>
        (document.querySelector('[data-testid="stat-geometry"]')?.innerText ?? '').includes(
          'letterbox'
        ),
      null, // `waitForFunction(fn, arg, options)`: the arg slot is not the options slot.
      { timeout: 30_000 }
    );
    const liveWarning = await page.evaluate(
      () => document.querySelector('[data-testid="live-resize-mismatch"]')?.innerText ?? null
    );
    const liveState = await page.evaluate(
      () => document.querySelector('[data-testid="state"]')?.textContent ?? ''
    );
    const liveGeometryWarn = await page.evaluate(
      () =>
        document.querySelector('[data-testid="stat-geometry"]')?.classList.contains('warn') ?? null
    );
    console.log(
      'live mismatch:',
      JSON.stringify({ liveState, liveWarning, liveGeometryWarn })
    );
    await shot(page, 'models-04-live-idle');
    check(
      'the Live view stays quiet for a mismatch while capture is idle',
      liveState !== 'running' && liveWarning === null,
      `state=${liveState} warning=${JSON.stringify(liveWarning)}`
    );
    // The banner is gated and the readout is not, so the same mismatch is visible in the strip
    // while the banner is silent. Without this the check above could pass with the tile also
    // hidden - i.e. with the readout quietly dropping the very state it exists to report.
    check(
      'while the strip still names the geometry that is running, flagged',
      liveGeometryWarn === true,
      `geometryWarn=${liveGeometryWarn}`
    );

    // The *unrecorded* case, which is the one nothing else here can reach: take the record away
    // and `auto` falls back to the format heuristic, so the geometry becomes an assumption. No
    // panel check can call that a mismatch — a mismatch needs a record to contradict — so the
    // sidecar has to report the assumption itself, and this is the check that says it does.
    //
    // The appear/clear pair is the control again: a warning that is simply always there would
    // pass the first half. `detector_backend: native` is patched in explicitly because this
    // warning is native-only, and reading it off the machine's saved backend would make the
    // check depend on a setting the check is not about.
    const backToAuto = await patch({
      active_model: `models/${SCRATCH}.pt`,
      resize_mode: 'auto',
      detector_backend: 'native'
    });
    check(
      'the sidecar accepted a native patch on `auto`',
      backToAuto.status === 200,
      `HTTP ${backToAuto.status}`
    );
    const readServerWarnings = async () => {
      // The panel reads the settings once on mount, so a fresh mount is what a new warning needs.
      await page.evaluate(() => document.querySelector('[data-testid="nav-live"]').click());
      await page.waitForSelector('[data-testid="nav-admin"]');
      await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
      await page.waitForSelector('.admin-loading', { state: 'detached', timeout: 30_000 });
      return page.evaluate(
        () => document.querySelector('[data-testid="server-warnings"]')?.innerText ?? ''
      );
    };

    const withRecord = await readServerWarnings();
    check(
      'a recorded requirement draws no assumed-geometry warning',
      !withRecord.includes('has no record of the geometry'),
      JSON.stringify(withRecord)
    );

    fs.rmSync(recordPath, { force: true });
    const withoutRecord = await readServerWarnings();
    console.log('warnings without a record:', JSON.stringify(withoutRecord));
    await shot(page, 'models-05-unrecorded');
    check(
      'removing the record makes the sidecar report the assumed geometry in the panel',
      withoutRecord.includes('has no record of the geometry') &&
        withoutRecord.includes('assumes letterbox'),
      JSON.stringify(withoutRecord)
    );
    // The remedy is a *separate* field on the entry, and this is the check that it survives the
    // wire into the panel: the sentence naming `--install` is the remedy half, which the entry
    // carries because the Live view renders it too - in a view that has no button to put the
    // advice next to.
    check(
      'and names the command that removes the assumption',
      withoutRecord.includes('--install'),
      JSON.stringify(withoutRecord)
    );

    // The Live view's copy of this warning. Its *appearance* needs a running capture, so it is
    // covered in LiveView.test.tsx; what is checkable on this machine is the gate - the same
    // assumption, the same screen, capture idle, and no banner. It has to be read here, with the
    // record still missing: the button below writes the record back, and this state cannot be
    // produced again afterwards without deleting it a second time.
    const readLiveAssumed = async () => {
      // A fresh mount, because the readout reads the settings once and does not poll.
      await page.evaluate(() => document.querySelector('[data-testid="nav-live"]').click());
      await page.waitForSelector('[data-testid="state"]');
      await page.waitForFunction(
        () => document.querySelector('[data-testid="stat-geometry"]') !== null,
        null, // `waitForFunction(fn, arg, options)`: the arg slot is not the options slot.
        { timeout: 30_000 }
      );
      return page.evaluate(() => ({
        state: document.querySelector('[data-testid="state"]')?.textContent ?? '',
        geometry: document.querySelector('[data-testid="stat-geometry"]')?.innerText ?? '',
        // The strip's copy of the same fact, which is *not* gated on a capture running - the one
        // half of this feature a machine with no camera can check in both its states.
        requirement: document.querySelector('[data-testid="stat-requirement"]')?.innerText ?? null,
        requirementAssumed:
          document
            .querySelector('[data-testid="stat-requirement"]')
            ?.classList.contains('unmeasured') ?? null,
        banner: document.querySelector('[data-testid="live-assumed-geometry"]')?.innerText ?? null
      }));
    };

    const liveIdleAssumed = await readLiveAssumed();
    console.log('live assumed geometry, idle:', JSON.stringify(liveIdleAssumed));
    await shot(page, 'models-05b-live-idle');
    check(
      'the Live view stays quiet for an assumed geometry while capture is idle',
      liveIdleAssumed.state !== 'running' && liveIdleAssumed.banner === null,
      `state=${liveIdleAssumed.state} banner=${JSON.stringify(liveIdleAssumed.banner)}`
    );
    // The banner is gated and the readout is not, so the assumption is still visible in the strip
    // while the banner is silent. Without this the check above would pass with the readout also
    // hidden - i.e. with nothing loaded, rather than with a gate working.
    check(
      'while the strip still names the geometry those weights run at',
      liveIdleAssumed.geometry.includes('letterbox'),
      JSON.stringify(liveIdleAssumed.geometry)
    );
    // The requirement chip: the assumption itself, on screen while the banner is gated off. Dim
    // rather than amber, because nothing recorded it is a gap in what is known - the same
    // distinction the recall tile draws beside it.
    check(
      'and marks the requirement as assumed, unflagged by colour',
      !!liveIdleAssumed.requirement &&
        liveIdleAssumed.requirement.includes('letterbox') &&
        liveIdleAssumed.requirement.includes('assumed') &&
        liveIdleAssumed.requirementAssumed === true,
      JSON.stringify(liveIdleAssumed.requirement)
    );

    // Back to the panel, which is where the button lives — the entry is read from Admin because
    // that is the view that can answer it, and the Live view above was only ever proving the gate.
    await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
    await page.waitForSelector('.admin-loading', { state: 'detached', timeout: 30_000 });

    // The remedy, in the real app: the panel's own button writes the record the sidecar asked
    // for. Nothing else in the panel knows letterbox is the answer — the mode comes from the
    // sidecar's entry — so this is also the check that the entry and the button are the same
    // fact rather than two renderings of it. The write goes through the real route to a real
    // file, which is the half no renderer test can reach.
    const fromPanel = await page.evaluate(async () => {
      const button = document.querySelector('[data-testid="record-resize-mode"]');
      if (!button) return { clicked: false, text: '' };
      button.click();
      // Waited on the *entry* changing rather than on the click having happened: a confirmation
      // rendered from the click alone would pass this too.
      const deadline = Date.now() + 20_000;
      const read = () =>
        document.querySelector('[data-testid="server-warnings"]')?.innerText ?? '';
      while (Date.now() < deadline) {
        if (read().includes('Recorded')) break;
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      return {
        clicked: true,
        text: read(),
        weights: document.querySelector('[data-testid="installed-models"]')?.innerText ?? ''
      };
    });
    console.log('after recording from the panel:', JSON.stringify(fromPanel.text));
    await shot(page, 'models-06-recorded');
    check('the panel offers a record-it-now button for a guessed geometry', fromPanel.clicked);
    check(
      'and clicking it writes the requirement the sidecar named',
      fs.existsSync(recordPath) &&
        JSON.parse(fs.readFileSync(recordPath, 'utf8')).resize_mode === 'letterbox',
      fs.existsSync(recordPath) ? fs.readFileSync(recordPath, 'utf8') : 'no record written'
    );
    check(
      'and the warning is answered in place',
      fromPanel.text.includes('Recorded') &&
        !fromPanel.text.includes('has no record of the geometry'),
      JSON.stringify(fromPanel.text)
    );
    // The acknowledgement is read off the refreshed listing, so this is the same write seen from
    // the other side of the panel: the weights now carry a requirement and `auto` uses it.
    check(
      'and the weights list says `auto` uses it now',
      fromPanel.weights.includes('resize_mode: letterbox') && fromPanel.weights.includes('auto uses it'),
      JSON.stringify(fromPanel.weights)
    );
    check('and the machine settings were not touched by it', (await api('/api/settings')).body.active_model === `models/${SCRATCH}.pt`);

    fs.writeFileSync(recordPath, JSON.stringify(record, null, 2) + '\n');
    const withRecordAgain = await readServerWarnings();
    check(
      'and putting the tool\'s record back clears it again',
      !withRecordAgain.includes('has no record of the geometry'),
      JSON.stringify(withRecordAgain)
    );

    // The chip in its other state, and the opposite of the reading above: the record exists, so
    // the strip names it and nothing is assumed anywhere on screen. That is the pair - the same
    // two facts, recorded and assumed, both readable while capture is idle.
    const liveRecorded = await readLiveAssumed();
    console.log('live requirement, recorded:', JSON.stringify(liveRecorded));
    await shot(page, 'models-05c-live-recorded');
    check(
      'the strip names the recorded requirement, unflagged',
      !!liveRecorded.requirement &&
        liveRecorded.requirement.includes('stretch') &&
        liveRecorded.requirement.includes('recorded') &&
        liveRecorded.requirementAssumed === false,
      JSON.stringify(liveRecorded.requirement)
    );
    check(
      'with no assumption anywhere on the view',
      liveRecorded.banner === null && liveRecorded.state !== 'running',
      `state=${liveRecorded.state} banner=${JSON.stringify(liveRecorded.banner)}`
    );
  } finally {
    // Never leave scratch weights where the picker can offer them, and never leave this
    // machine's settings changed — even if a check threw.
    fs.rmSync(weightPath, { force: true });
    fs.rmSync(recordPath, { force: true });
    check(
      'scratch weight removed',
      !fs.existsSync(weightPath) && !fs.existsSync(recordPath)
    );
    if (port && original) {
      const restored = await page
        .evaluate(
          ([pt, i]) =>
            fetch(`http://127.0.0.1:${pt}${i.path}`, i).then(async (r) => ({
              status: r.status,
              body: await r.json()
            })),
          [
            port,
            {
              path: '/api/settings?persist=false',
              method: 'PATCH',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                active_model: original.active_model,
                resize_mode: original.resize_mode
              })
            }
          ]
        )
        .catch(() => null);
      check(
        'settings put back',
        restored?.body?.active_model === original.active_model &&
          restored?.body?.resize_mode === original.resize_mode
      );
    }
  }
}

if (mode === 'classlist') {
  // The two halves of the roster guard, against a weight that really is non-roster: the class list
  // **recorded** beside it (flagged by the Admin Panel's listing, before anything runs) and the
  // verdict the running app broadcasts (**the banner in the Live view**, which needs a live
  // capture). Neither can be checked from the unit tests: they can render the banner from a fake
  // status or serve `/api/models` from a temp directory, but not put a real non-roster checkpoint
  // through ultralytics in one process and read the real renderer's response to it.
  //
  // The scratch weight is a **genuine 24-output checkpoint** built by ultralytics itself, not a
  // stock `.pt` with a renamed class list. That distinction is the whole point: the failure being
  // verified is a *head* trained per product-and-distance, so the thing has to produce 24-output
  // predictions whose indices resolve in the names dict the app indexes by class id
  // (`normalize_detections`), or the run would fail on a KeyError instead of reporting a class
  // list. `DetectionModel(cfg, nc=24)` is the same constructor `yolo11n.pt` came from; the weights
  // are untrained, which is fine — this check is about the vocabulary, not about boxes.
  const SIDECAR = path.join(REPO_ROOT, 'sidecar');
  const MODELS = path.join(SIDECAR, 'models');
  const python = process.env.SIDECAR_PYTHON || path.join(SIDECAR, '.venv', 'Scripts', 'python.exe');
  const SCRATCH = 'driver-scratch-nonroster';
  const weightPath = path.join(MODELS, `${SCRATCH}.pt`);
  const recordPath = path.join(MODELS, `${SCRATCH}.json`);

  let port = null;
  let original = null;
  try {
    fs.mkdirSync(MODELS, { recursive: true });
    // Both writers are the real ones: `DetectionModel(nc=24)` for the weight, `weight_record` for
    // the record — a hand-built JSON would agree with the reader by construction, which is exactly
    // what this mode is supposed to be able to disconfirm.
    const record = JSON.parse(
      execFileSync(
        python,
        [
          '-c',
          `import json, torch
from ultralytics.nn.tasks import DetectionModel
import train_model, generations

PRODUCTS = [
    "Bear Brand Fortified Powdered Milk 33g",
    "lucky_me_pancit_canton_calamansi_flavor",
    "555 sardines 155grams",
    "century_tuna_flakes_in_oil_155_grams",
    "silver_swan_sukang_puti_200ML",
    "Milo Chocolate Drink 22g Sachet",
    "safeguard_pure_white_60g",
    "Palmolive Naturals Bar Soap 85g",
]
# One class per product *and distance* - the mistake MODEL_TRAINING 8.1 exists to prevent.
names = [f"{p} {d}" for p in PRODUCTS for d in ("close", "mid", "far")]
model = DetectionModel("yolo11n.yaml", nc=len(names), verbose=False)
model.names = {i: n for i, n in enumerate(names)}
torch.save({"model": model, "date": "driver-scratch", "version": ""}, r"${weightPath}")
print(json.dumps(train_model.weight_record(generations.V2, 2, "snc-grocery", class_names=names)))`
        ],
        { cwd: path.join(SIDECAR, 'tools'), encoding: 'utf8' }
      )
        .trim()
        .split('\n')
        .pop()
    );
    fs.writeFileSync(recordPath, JSON.stringify(record, null, 2) + '\n');
    console.log(
      'scratch weight:',
      `${fs.statSync(weightPath).size} bytes`,
      '| record:',
      JSON.stringify({ class_names: record.class_names?.length, resize_mode: record.resize_mode })
    );

    port = await page.evaluate(() => window.api.getSidecarPort());
    const api = (p, init) =>
      page.evaluate(
        ([pt, i]) =>
          fetch(`http://127.0.0.1:${pt}${i.path}`, i).then(async (r) => ({
            status: r.status,
            body: await r.json()
          })),
        [port, { path: p, ...init }]
      );
    const patch = (body) =>
      api('/api/settings?persist=false', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
    const settings = (await api('/api/settings')).body;
    console.log(
      'capture settings in force:',
      JSON.stringify({
        width: settings.capture_width,
        height: settings.capture_height,
        fps: settings.capture_fps,
        device: settings.device
      })
    );

    // ---- half one: the listing, which needs no capture ------------------------------------
    const listed = (await api('/api/models')).body.installed;
    const mine = listed.find((m) => m.value === `models/${SCRATCH}.pt`);
    check('the sidecar lists the scratch weight', !!mine, JSON.stringify(mine?.value));
    check(
      'with the class list recorded beside it',
      mine?.class_names?.length === 24,
      `class_names=${mine?.class_names?.length}`
    );
    check(
      'and the roster verdict judged from the listing, before it runs',
      mine?.class_warnings?.length === 1 && mine.class_warnings[0].includes('carry a distance'),
      JSON.stringify(mine?.class_warnings)
    );

    await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
    await page
      .waitForSelector('#active_model', { timeout: 120_000 })
      .catch(() => {})
      .then(() => page.waitForSelector('.admin-loading', { state: 'detached', timeout: 120_000 }))
      .catch(() => {});
    await page
      .waitForSelector(`[data-testid="installed-model-${mine.value}"]`, { timeout: 30_000 })
      .catch(() => {});
    const listedOnScreen = await readText(`[data-testid="installed-model-${mine.value}"]`);
    const warningOnScreen = await readText(
      `[data-testid="installed-model-class-warning-${mine.value}"]`
    );
    console.log('listing on screen:', JSON.stringify(listedOnScreen));
    check(
      'the panel shows the weight with its class count',
      !!listedOnScreen && listedOnScreen.includes('24 classes'),
      JSON.stringify(listedOnScreen)
    );
    check(
      'and the sidecar\u2019s own sentence beside it',
      !!warningOnScreen && warningOnScreen.includes('carry a distance'),
      JSON.stringify(warningOnScreen)
    );
    await shot(page, 'classlist-01-admin-listing');

    // ---- half two: the banner, which needs the pipeline to have inferred -------------------
    original = settings;
    const patched = await patch({
      active_model: `models/${SCRATCH}.pt`,
      resize_mode: 'auto'
    });
    check('the sidecar accepted the patch', patched.status === 200, `HTTP ${patched.status}`);

    await page.evaluate(() => document.querySelector('[data-testid="nav-live"]').click());
    await page.waitForTimeout(1_000);
    await page.evaluate(() => document.querySelector('button[aria-label="Start"]').click());
    console.log('clicked Start — the scratch weight loads instead of a stock one');
    const outcome = await page
      .waitForFunction(
        () => {
          if (document.querySelector('img.preview-img')) return 'frames';
          if (document.querySelector('[data-testid="live-error"]')?.innerText) return 'error';
          return null;
        },
        null,
        { timeout: 120_000 }
      )
      .then((handle) => handle.jsonValue())
      .catch(() => null);
    console.log('after Start:', outcome ?? 'neither a frame nor a reason after 120s');
    await shot(page, 'classlist-02-after-start');

    if (outcome === 'frames') {
      const seen = await page
        .waitForSelector('[data-testid="live-class-warnings"]', { timeout: 30_000 })
        .then(() => true)
        .catch(() => false);
      const banner = await readText('[data-testid="live-class-warnings"]');
      const chip = await readText('[data-testid="stat-classes"]');
      const chipClass = await page
        .evaluate(() => document.querySelector('[data-testid="stat-classes"]')?.className ?? null)
        .catch(() => null);
      // Text can be present in the DOM and still not be *on screen* (a zero-height box, a hidden
      // ancestor), which is the one thing `textContent` cannot tell apart — the reason this mode
      // takes screenshots at all. Laid out and painted, as numbers: this is the assertion the
      // images are there to corroborate by eye.
      const painted = (sel) =>
        page.evaluate((s) => {
          const el = document.querySelector(s);
          if (!el) return null;
          const r = el.getBoundingClientRect();
          const style = getComputedStyle(el);
          return {
            w: Math.round(r.width),
            h: Math.round(r.height),
            onScreen: r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth,
            shown:
              style.display !== 'none' &&
              style.visibility !== 'hidden' &&
              Number(style.opacity) > 0
          };
        }, sel);
      const bannerPaint = await painted('[data-testid="live-class-warnings"]').catch(() => null);
      const chipPaint = await painted('[data-testid="stat-classes"]').catch(() => null);
      console.log(
        'banner:', JSON.stringify(banner),
        '| chip:', JSON.stringify(chip),
        '| chip class:', JSON.stringify(chipClass)
      );
      console.log('banner box:', JSON.stringify(bannerPaint), '| chip box:', JSON.stringify(chipPaint));
      check('the running app reports the class list on screen', seen, JSON.stringify(banner));
      check(
        'and the banner is laid out and painted, not just present in the DOM',
        !!bannerPaint &&
          bannerPaint.w > 0 &&
          bannerPaint.h > 0 &&
          bannerPaint.onScreen &&
          bannerPaint.shown,
        JSON.stringify(bannerPaint)
      );
      check(
        'as is the chip',
        !!chipPaint && chipPaint.w > 0 && chipPaint.h > 0 && chipPaint.onScreen && chipPaint.shown,
        JSON.stringify(chipPaint)
      );
      check(
        'naming the distance the model was trained per product-and-distance',
        !!banner && banner.includes('carry a distance') && banner.includes('24 of 24'),
        JSON.stringify(banner)
      );
      check(
        'and the stats strip carries the count the banner cannot show',
        !!chip && chip.includes('24') && chip.includes('finding') && /\bwarn\b/.test(chipClass ?? ''),
        JSON.stringify(chip)
      );
      await shot(page, 'classlist-03-banner-and-chip');
    }

    // The stop has to happen before the settings are put back: `active_model` is restart-required,
    // so the restore would be rejected with a 409 while a capture holds it.
    const stopBtn = await page.$('button[aria-label="Stop"]');
    if (stopBtn) {
      await page.evaluate(() => document.querySelector('button[aria-label="Stop"]').click());
      await page
        .waitForFunction(
          () => document.querySelector('[data-testid="state"]')?.textContent !== 'running',
          { timeout: 30_000 }
        )
        .catch(() => console.log('WARNING: state did not leave running within 30s'));
    }
  } finally {
    // Never leave a deliberately broken weight where the picker can offer it, and never leave this
    // machine's settings changed — even if a check threw.
    fs.rmSync(weightPath, { force: true });
    fs.rmSync(recordPath, { force: true });
    check('scratch weight removed', !fs.existsSync(weightPath) && !fs.existsSync(recordPath));
    if (port && original) {
      const restored = await page
        .evaluate(
          ([pt, i]) =>
            fetch(`http://127.0.0.1:${pt}${i.path}`, i).then(async (r) => ({
              status: r.status,
              body: await r.json()
            })),
          [
            port,
            {
              path: '/api/settings?persist=false',
              method: 'PATCH',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                active_model: original.active_model,
                resize_mode: original.resize_mode
              })
            }
          ]
        )
        .catch(() => null);
      check(
        'settings put back',
        restored?.body?.active_model === original.active_model &&
          restored?.body?.resize_mode === original.resize_mode,
        `active_model=${restored?.body?.active_model}`
      );
    }
  }
}

if (mode === 'probe') {
  // `local_api`'s whole point is that it is *some* self-hosted HTTP endpoint, and this machine has
  // no API key and no inference server running — so the only way to exercise the path for real is
  // to be the server. It answers the payload shape captured from the live workflow in Phase 0
  // (docs/DETECTOR_BACKENDS.md §0) and can be told, between probes, to report a different canvas
  // or no canvas at all. That is the one thing the unit tests cannot do: they fake the detector,
  // so they never put a real HTTP response through `find_image_size` and into the panel.
  let stubSize = { width: 640, height: 360 };
  const stub = http.createServer((req, res) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      const predictions = { predictions: [] };
      if (stubSize) predictions.image = stubSize;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ predictions }));
    });
  });
  await new Promise((resolve) => stub.listen(0, '127.0.0.1', resolve));
  const stubPort = stub.address().port;
  console.log(`stub workflow listening on 127.0.0.1:${stubPort}`);

  let port = null;
  let original = null;
  try {
    port = await page.evaluate(() => window.api.getSidecarPort());
    const api = (p, init) =>
      page.evaluate(
        ([pt, i]) =>
          fetch(`http://127.0.0.1:${pt}${i.path}`, i).then(async (r) => ({
            status: r.status,
            body: await r.json()
          })),
        [port, { path: p, ...init }]
      );
    const patch = (body) =>
      api('/api/settings?persist=false', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

    original = (await api('/api/settings')).body;
    // A 16:9 capture with a 640-px transmit limit, so the sent size is 640x360 — the geometry a real
    // capture would use, and a shape a square probe frame could never have shown.
    const patched = await patch({
      detector_backend: 'local_api',
      local_api_url: `http://127.0.0.1:${stubPort}`,
      roboflow_workspace: 'stub-workspace',
      roboflow_workflow_id: 'stub-workflow',
      capture_width: 1280,
      capture_height: 720,
      remote_infer_size: 640
    });
    check('the sidecar accepted a local_api patch', patched.status === 200, `HTTP ${patched.status}`);

    // AdminPanel reads the settings on mount, so this picks up the patch — and the button is only
    // offered for a remote backend, which is itself part of what is being checked.
    await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
    await page.waitForSelector('[data-testid="test-connection"]', { timeout: 30_000 });

    const probeAndWait = async (marker) => {
      await page.evaluate(() => document.querySelector('[data-testid="test-connection"]').click());
      // Waited on the *content* of the line rather than on the click: the element survives from the
      // previous probe, so waiting for its existence would pass on the previous state's text.
      await page.waitForFunction(
        (m) => (document.querySelector('[data-testid="probe-geometry"]')?.innerText ?? '').includes(m),
        marker,
        { timeout: 30_000 }
      );
      return page.evaluate(() => {
        const el = document.querySelector('[data-testid="probe-geometry"]');
        return { state: el?.getAttribute('data-state'), text: el?.innerText ?? '' };
      });
    };

    const same = await probeAndWait('passes the frame through');
    console.log('probe geometry (workflow echoes the sent size):', JSON.stringify(same));
    await shot(page, 'probe-01-same');
    check('a workflow that echoes the sent size is reported as a pass-through', same.state === 'same');
    // The sent size is the proof that the probe reproduces capture's geometry: 1280x720 reduced to
    // a 640-px longest side, which the real `_encode` rule decides, not the panel.
    check(
      'and the app side of the pair is what a real capture would send',
      same.text.includes('640×360'),
      JSON.stringify(same.text)
    );

    stubSize = { width: 640, height: 640 };
    const resized = await probeAndWait('stretching or padding');
    console.log('probe geometry (workflow re-frames to a square canvas):', JSON.stringify(resized));
    await shot(page, 'probe-02-resized');
    check('a workflow that re-frames the image is flagged', resized.state === 'resized', JSON.stringify(resized));
    check(
      'and both sizes are named, so the operator can see which is which',
      resized.text.includes('640×640') && resized.text.includes('640×360'),
      JSON.stringify(resized.text)
    );

    stubSize = null;
    const silent = await probeAndWait('reported no image size');
    console.log('probe geometry (workflow reports no size block):', JSON.stringify(silent));
    await shot(page, 'probe-03-unreported');
    check(
      'a silent workflow is reported as unverified rather than as a match',
      silent.state === 'unreported' && !silent.text.includes('passes the frame through'),
      JSON.stringify(silent)
    );
  } finally {
    await new Promise((resolve) => stub.close(resolve));
    // `persist=false` throughout, and the original values put back — the machine's saved config is
    // none of this mode's business.
    if (port && original) {
      const restored = await page
        .evaluate(
          ([pt, i]) =>
            fetch(`http://127.0.0.1:${pt}${i.path}`, i).then(async (r) => ({
              status: r.status,
              body: await r.json()
            })),
          [
            port,
            {
              path: '/api/settings?persist=false',
              method: 'PATCH',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                detector_backend: original.detector_backend,
                local_api_url: original.local_api_url,
                capture_width: original.capture_width,
                capture_height: original.capture_height,
                remote_infer_size: original.remote_infer_size
              })
            }
          ]
        )
        .catch(() => null);
      check(
        'settings put back',
        restored?.body?.detector_backend === original.detector_backend &&
          restored?.body?.capture_width === original.capture_width
      );
    }
  }
}

if (mode === 'capture') {
  await page.evaluate(() => document.querySelector('button[aria-label="Start"]').click());
  console.log('clicked Start — first model load can take a while');
  // Whichever arrives first: a frame, or the sidecar naming why there is none. Waiting only for the
  // frame made this mode spend two minutes and then report "NO FRAMES" as a *non*-failure, which on
  // a wedged camera (this machine's standing condition — see SKILL.md) hid the one thing worth
  // reading: the reason the capture stopped.
  const outcome = await page
    .waitForFunction(
      () => {
        if (document.querySelector('img.preview-img')) return 'frames';
        if (document.querySelector('[data-testid="live-error"]')?.innerText) return 'error';
        return null;
      },
      null,
      { timeout: 120_000 }
    )
    .then((handle) => handle.jsonValue())
    .catch(() => null);
  const frames = outcome === 'frames';
  console.log(
    'after Start:',
    outcome ?? 'neither a frame nor a reason after 120s',
    '| state:',
    await readText('[data-testid="state"]'),
    '| ws:',
    await readText('[data-testid="conn"]')
  );
  await shot(page, 'capture-02-after-start');
  if (frames) console.log('frames are streaming');

  if (frames) {
    await page.waitForTimeout(8_000);
    console.log('stats:', JSON.stringify(await page.evaluate(
      () => document.querySelector('[data-testid="stats"]')?.innerText ?? '(no stats)')));
    console.log('item log:', JSON.stringify(await page.evaluate(() => {
      const ul = document.querySelector('[data-testid="item-log"]');
      return ul ? { items: ul.children.length, text: ul.innerText.slice(0, 300) } : null;
    })));
    await shot(page, 'capture-03-running');

    // A reload while capture runs — the case the handshake status exists for. The renderer has
    // no memory of the Start it already did, so the only way it can know is the state the
    // sidecar sends on connect. Before that, the reloaded page sat on `idle` with frames
    // streaming past it and offered Start over a running capture.
    let resumed = false;
    try {
      await page.reload();
      // `null` then options: this signature is (fn, arg, options), and passing the options
      // object second silently leaves the default 30s in place.
      await page.waitForFunction(
        () => document.querySelector('[data-testid="state"]')?.textContent === 'running',
        null,
        { timeout: 60_000 }
      );
      resumed = true;
    } catch (e) {
      console.log('reload check failed:', String(e).slice(0, 200));
    }
    check(
      'a reloaded renderer learns capture is running from the socket',
      resumed,
      `state=${await stateText()}`
    );
    await shot(page, 'capture-03b-reloaded');
  }

  // The other branch, and on a wedged camera the only one this mode can take: the reason, then the
  // same reason again on the other side of a reload. That second half is the only place it can be
  // checked for real — `idle` alone is exactly what a capture that was never started looks like,
  // so without the reason surviving the reconnect a dead capture is indistinguishable from an idle
  // app. Guarded on there being a *death* to recover from: a start that failed outright (an
  // unreachable API, a missing key) never reaches the pipeline and stores nothing.
  if (!frames) {
    const reason = await readText('[data-testid="live-error"]')
    console.log('reason on screen:', JSON.stringify(reason))
    if (reason !== null && reason.includes('Capture stopped')) {
      check('a capture that stopped streaming is explained on screen', true, JSON.stringify(reason))
      let after = null
      let pageText = ''
      try {
        await page.reload()
        await page.waitForFunction(
          () => !!document.querySelector('[data-testid="live-error"]')?.innerText,
          null,
          { timeout: 60_000 }
        )
        after = await page.evaluate(
          () => document.querySelector('[data-testid="live-error"]')?.innerText ?? ''
        )
      } catch (e) {
        console.log('reload-after-death check failed:', String(e).slice(0, 200))
      }
      if (after === null) pageText = await bodyText()
      console.log('after reload — error banner:', JSON.stringify(after), '| page:', JSON.stringify(pageText))
      await shot(page, 'capture-02b-reload-after-death')
      check(
        'and a reloaded renderer is told the same reason from the handshake',
        !!after && after.includes('Capture stopped'),
        JSON.stringify(after)
      )
    } else {
      // Not a death: the start itself failed, which is reported in its own error status and is not
      // this feature. Said out loud rather than silently skipped.
      console.log('no dead capture to recover from — start did not get as far as the pipeline')
    }
  }

  const stopBtn = await page.$('button[aria-label="Stop"]');
  if (stopBtn) {
    await page.evaluate(() => document.querySelector('button[aria-label="Stop"]').click());
    await page.waitForFunction(
      () => document.querySelector('[data-testid="state"]')?.textContent !== 'running',
      { timeout: 30_000 }
    ).catch(() => console.log('WARNING: state did not leave running within 30s'));
    await page.waitForTimeout(1_500);
    console.log('after stop — state:', await stateText(), '| ws:', await connText());
    await shot(page, 'capture-04-stopped');
  }
}

await app.close();
console.log('app closed cleanly');

if (failures.length > 0) {
  console.error(`\n${failures.length} check(s) failed:`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
