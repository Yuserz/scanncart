// Case-by-case tests against the running app. Companion to driver.mjs.
//
//   node .claude/skills/run-desktop/case-tests.mjs <A|B|D|E>
//
// A: local_api happy path      - Start, frames, item log, Stop
// B: restart-required gating   - try switching backend while capture runs
// D: server down at Start      - does the 503 reach the user?
// E: server killed mid-capture - does the pipeline survive / report?
//
// Case D expects the caller to have stopped the inference server first;
// case E kills it from inside the run.
import * as fs from 'node:fs';
import * as path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const SKILL_DIR = path.dirname(fileURLToPath(import.meta.url));
const APP_DIR = path.resolve(SKILL_DIR, '..', '..', '..', 'desktop');
const require = createRequire(path.join(APP_DIR, 'package.json'));
const { _electron: electron } = require('playwright-core');

const SHOT_DIR = process.env.SCREENSHOT_DIR || path.join(SKILL_DIR, 'shots');
fs.mkdirSync(SHOT_DIR, { recursive: true });
const electronBin = path.join(APP_DIR, 'node_modules', 'electron', 'dist', 'electron.exe');
const CASE = (process.argv[2] || 'A').toUpperCase();

const log = (...a) => console.log(...a);
const shot = async (p, n) => {
  const f = path.join(SHOT_DIR, 'case-' + CASE + '-' + n + '.png');
  await p.screenshot({ path: f });
  log('  shot:', f);
};

function killInferenceServer() {
  try {
    const out = execSync('netstat -ano -p TCP | findstr ":9001" | findstr LISTENING', { encoding: 'utf8' });
    const pids = [...new Set(out.trim().split(/\r?\n/).map((l) => l.trim().split(/\s+/).pop()))];
    for (const pid of pids) {
      execSync('taskkill /F /PID ' + pid, { stdio: 'ignore' });
      log('  killed inference server pid', pid);
    }
    return pids.length > 0;
  } catch {
    log('  no listener on 9001 to kill');
    return false;
  }
}

let app = null;
let page = null;
for (let attempt = 1; attempt <= 4; attempt++) {
  app = await electron.launch({ executablePath: electronBin, args: [APP_DIR], cwd: APP_DIR, timeout: 30000 });
  page = await app.firstWindow();
  try {
    await page.waitForSelector('[data-testid="nav-live"]', { timeout: 45000 });
    break;
  } catch {
    await app.close().catch(() => {});
    if (attempt === 4) throw new Error('sidecar never started');
  }
}

const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(String(e)));
page.on('console', (m) => {
  if (m.type() === 'error') pageErrors.push('console: ' + m.text());
});

// The preload exposes the sidecar port; use it to read server-side truth
// rather than only what the UI chose to render.
const port = await page.evaluate(() => window.api.getSidecarPort());
const api = async (p, init) =>
  page.evaluate(
    async ([u, i]) => {
      try {
        const res = await fetch(u, i);
        return { status: res.status, body: await res.text() };
      } catch (e) {
        return { status: 0, body: String(e) };
      }
    },
    ['http://127.0.0.1:' + port + p, init || {}]
  );
const health = async () => JSON.parse((await api('/api/health')).body);
const ui = async () =>
  page.evaluate(() => ({
    state: document.querySelector('[data-testid="state"]')?.textContent,
    conn: document.querySelector('[data-testid="conn"]')?.textContent,
    frames: !!document.querySelector('img.preview-img'),
    items: document.querySelector('[data-testid="item-log"]')?.children.length ?? 0,
    errorsShown: [...document.querySelectorAll('.admin-error, .live-error, [role="alert"]')].map((n) =>
      n.innerText.trim()
    )
  }));

log('\n=== CASE ' + CASE + ' === sidecar port ' + port);
log('backend:', JSON.parse((await api('/api/settings')).body).detector_backend);
log('initial health:', JSON.stringify(await health()));

if (CASE === 'A' || CASE === 'B' || CASE === 'E') {
  log('\n-- pressing Start --');
  await page.evaluate(() => document.querySelector('button[aria-label="Start"]').click());
  let framed = false;
  try {
    await page.waitForSelector('img.preview-img', { timeout: 120000 });
    framed = true;
  } catch {
    log('  NO FRAMES within 120s');
  }
  log('  frames streaming:', framed);
  log('  health:', JSON.stringify(await health()));
  await shot(page, '01-started');
}

if (CASE === 'A') {
  await page.waitForTimeout(10000);
  log('  after 10s ->', JSON.stringify(await ui()));
  log('  stats:', await page.evaluate(() => document.querySelector('[data-testid="stats"]')?.innerText.replace(/\s+/g, ' ')));
  await shot(page, '02-running');

  log('\n-- pressing Stop --');
  const t = Date.now();
  await page.evaluate(() => document.querySelector('button[aria-label="Stop"]').click());
  await page
    .waitForFunction(() => document.querySelector('[data-testid="state"]')?.textContent !== 'running', { timeout: 40000 })
    .catch(() => log('  WARNING: still "running" after 40s'));
  log('  stop took ' + (Date.now() - t) + ' ms; health: ' + JSON.stringify(await health()));
  log('  log rows:', JSON.parse((await api('/api/logs')).body).events.length);
  await shot(page, '03-stopped');
}

if (CASE === 'B') {
  log('\n-- attempting backend switch while capture runs --');
  const r = await api('/api/settings', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ detector_backend: 'native' })
  });
  log('  PATCH detector_backend -> HTTP ' + r.status + ': ' + r.body.slice(0, 200));

  await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
  await page.waitForSelector('[data-testid="backend-picker"]', { timeout: 15000 });
  const saveDisabled = await page.evaluate(
    () => document.querySelector('[data-testid="save-settings"]')?.disabled
  );
  log('  Save button disabled while running:', saveDisabled);
  await shot(page, '01-admin-while-running');
  await api('/api/capture/stop', { method: 'POST' });
}

if (CASE === 'D') {
  log('\n-- Start with the inference server DOWN --');
  const before = await ui();
  await page.evaluate(() => document.querySelector('button[aria-label="Start"]').click());
  await page.waitForTimeout(20000);
  const after = await ui();
  log('  ui before:', JSON.stringify(before));
  log('  ui after :', JSON.stringify(after));
  log('  health   :', JSON.stringify(await health()));
  await shot(page, '01-start-with-server-down');
}

if (CASE === 'E') {
  log('\n-- killing the inference server MID-capture --');
  killInferenceServer();
  await page.waitForTimeout(30000);
  log('  ui    :', JSON.stringify(await ui()));
  log('  health:', JSON.stringify(await health()));
  await shot(page, '02-after-server-killed');

  log('\n-- pressing Start again (is it a no-op?) --');
  const r = await api('/api/capture/start', { method: 'POST' });
  log('  POST /api/capture/start -> HTTP ' + r.status + ': ' + r.body.slice(0, 200));

  log('\n-- pressing Stop --');
  const t = Date.now();
  const s = await api('/api/capture/stop', { method: 'POST' });
  log('  POST /api/capture/stop -> HTTP ' + s.status + ' in ' + (Date.now() - t) + ' ms');
  log('  health:', JSON.stringify(await health()));
  await shot(page, '03-after-stop');
}

log('\npage errors total:', pageErrors.length ? pageErrors : '(none)');
await app.close();
log('case ' + CASE + ' done');
