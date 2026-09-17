// Playwright driver for the SCANnCART Electron desktop app (Windows host).
//
// Usage (from repo root, after `cd desktop && npm run build`):
//   node .claude/skills/run-desktop/driver.mjs smoke      # launch, screenshot Live + Admin
//   node .claude/skills/run-desktop/driver.mjs capture    # full Start -> frames -> Stop flow
//   node .claude/skills/run-desktop/driver.mjs allowlist  # class allowlist field -> sidecar round-trip
//
// Modes that assert print PASS/FAIL per check and exit non-zero if any fail,
// so they can gate a change. `allowlist` restores the sidecar's original
// setting before exiting, including when a check throws.
//
// Screenshots land in <repo>/.claude/skills/run-desktop/shots/ (gitignored),
// override with SCREENSHOT_DIR.
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

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
if (!['smoke', 'capture', 'allowlist'].includes(mode)) {
  console.error(`unknown mode '${mode}' — use: smoke | capture | allowlist`);
  process.exit(1);
}

const shot = async (page, name) => {
  const f = path.join(SHOT_DIR, `${name}.png`);
  await page.screenshot({ path: f });
  console.log('screenshot:', f);
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
await page.waitForTimeout(2_000);
console.log('state:', await stateText(), '| ws:', await connText());
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

if (mode === 'capture') {
  await page.evaluate(() => document.querySelector('button[aria-label="Start"]').click());
  console.log('clicked Start — first model load can take a while');
  let frames = false;
  try {
    await page.waitForSelector('img.preview-img', { timeout: 120_000 });
    frames = true;
    console.log('frames are streaming');
  } catch {
    console.log('NO FRAMES after 120s — state:', await stateText(), '| ws:', await connText());
  }
  await shot(page, 'capture-02-after-start');

  if (frames) {
    await page.waitForTimeout(8_000);
    console.log('stats:', JSON.stringify(await page.evaluate(
      () => document.querySelector('[data-testid="stats"]')?.innerText ?? '(no stats)')));
    console.log('item log:', JSON.stringify(await page.evaluate(() => {
      const ul = document.querySelector('[data-testid="item-log"]');
      return ul ? { items: ul.children.length, text: ul.innerText.slice(0, 300) } : null;
    })));
    await shot(page, 'capture-03-running');
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
