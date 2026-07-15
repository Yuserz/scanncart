// Playwright driver for the SCANnCART Electron desktop app (Windows host).
//
// Usage (from repo root, after `cd desktop && npm run build`):
//   node .claude/skills/run-desktop/driver.mjs smoke     # launch, screenshot Live + Admin
//   node .claude/skills/run-desktop/driver.mjs capture   # full Start -> frames -> Stop flow
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
if (!['smoke', 'capture'].includes(mode)) {
  console.error(`unknown mode '${mode}' — use: smoke | capture`);
  process.exit(1);
}

const shot = async (page, name) => {
  const f = path.join(SHOT_DIR, `${name}.png`);
  await page.screenshot({ path: f });
  console.log('screenshot:', f);
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
