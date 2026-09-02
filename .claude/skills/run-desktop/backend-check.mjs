// Verifies the detector-backend picker and the Test-connection probe in the
// Admin Panel. Companion to driver.mjs; same launch pattern and gotchas.
//
//   node .claude/skills/run-desktop/backend-check.mjs
//
// Expects a detector backend already selected in sidecar/data/settings.json.
// For local_api, the inference server must be running on local_api_url —
// see docs/DETECTOR_BACKENDS.md §7a for the Docker-free setup.
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const SKILL_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SKILL_DIR, '..', '..', '..');
const APP_DIR = path.join(REPO_ROOT, 'desktop');
const require = createRequire(path.join(APP_DIR, 'package.json'));
const { _electron: electron } = require('playwright-core');

const SHOT_DIR = process.env.SCREENSHOT_DIR || path.join(SKILL_DIR, 'shots');
fs.mkdirSync(SHOT_DIR, { recursive: true });
const electronBin = path.join(APP_DIR, 'node_modules', 'electron', 'dist', 'electron.exe');

const shot = async (page, name) => {
  const f = path.join(SHOT_DIR, `${name}.png`);
  await page.screenshot({ path: f });
  console.log('screenshot:', f);
};

let app = null, page = null;
for (let attempt = 1; attempt <= 4; attempt++) {
  app = await electron.launch({ executablePath: electronBin, args: [APP_DIR], cwd: APP_DIR, timeout: 30_000 });
  page = await app.firstWindow();
  try {
    await page.waitForSelector('[data-testid="nav-live"]', { timeout: 45_000 });
    console.log(`attempt ${attempt}: AppShell mounted — sidecar port received`);
    break;
  } catch {
    console.log(`attempt ${attempt}: sidecar never reported a port — relaunching`);
    await app.close().catch(() => {});
    app = null;
    if (attempt === 4) throw new Error('sidecar failed to start in 4 attempts — see SKILL.md gotchas');
  }
}

await page.evaluate(() => document.querySelector('[data-testid="nav-admin"]').click());
await page.waitForSelector('[data-testid="backend-picker"]', { timeout: 15_000 });

const selected = await page.evaluate(() => {
  const el = document.querySelector('[data-testid="backend-picker"] input:checked');
  return el ? el.value : '(none)';
});
console.log('selected backend:', selected);

const warns = await page.evaluate(() =>
  [...document.querySelectorAll('[data-testid="backend-picker"] .admin-warning')].map((n) => n.innerText.trim()));
console.log('backend warnings:', warns.length ? warns : '(none)');
await shot(page, 'backend-01-picker');

// Probe. reachable:false is a normal answer, so the element appears either way.
await page.evaluate(() => document.querySelector('[data-testid="test-connection"]').click());
console.log('clicked Test connection — a cold model load can take ~5 s');
try {
  await page.waitForSelector('[data-testid="probe-result"]', { timeout: 120_000 });
  const r = await page.evaluate(() => {
    const el = document.querySelector('[data-testid="probe-result"]');
    return { text: el.innerText.trim(), ok: el.className.includes('probe-ok') };
  });
  console.log(`probe: ${r.ok ? 'REACHABLE' : 'FAILED'} — ${r.text}`);
} catch {
  console.log('NO PROBE RESULT after 120s');
}
await shot(page, 'backend-02-probe');

await app.close();
console.log('app closed cleanly');
