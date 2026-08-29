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

// Launch
let app = null;
let page = null;
for (let attempt = 1; attempt <= 4; attempt++) {
  app = await electron.launch({ executablePath: electronBin, args: [APP_DIR], cwd: APP_DIR, timeout: 30_000 });
  page = await app.firstWindow();
  try {
    await page.waitForSelector('[data-testid="nav-live"]', { timeout: 45_000 });
    console.log(`attempt ${attempt}: AppShell mounted`);
    break;
  } catch {
    console.log(`attempt ${attempt}: sidecar never reported a port — relaunching`);
    await app.close().catch(() => {});
    app = null;
    if (attempt === 4) throw new Error('sidecar failed to start in 4 attempts');
  }
}

await page.waitForTimeout(2_000);
console.log('state:', await page.textContent('[data-testid="state"]'));
console.log('conn:', await page.textContent('[data-testid="conn"]'));

// Click Start
console.log('Clicking Start...');
await page.evaluate(() => document.querySelector('button[aria-label="Start"]').click());

// Wait a bit and check state
for (let i = 1; i <= 10; i++) {
  await page.waitForTimeout(3_000);
  try {
    const state = await page.textContent('[data-testid="state"]');
    const conn = await page.textContent('[data-testid="conn"]');
    const hasPreview = await page.$('img.preview-img');
    const hasPlaceholder = await page.$('[data-testid="preview-placeholder"]');
    console.log(`[${i*3}s] state=${state} conn=${conn} preview=${!!hasPreview} placeholder=${!!hasPlaceholder}`);
    
    // Check for console errors
    const errors = await page.evaluate(() => {
      return (window.__consoleErrors || []).join('\n');
    });
    if (errors) console.log('Console errors:', errors);
    
    if (hasPreview) {
      console.log('Frames are streaming!');
      await shot(page, 'capture-debug-frames');
      break;
    }
  } catch (e) {
    console.log(`[${i*3}s] Error reading state: ${e.message}`);
    await shot(page, `capture-debug-error-${i}`);
  }
}

await shot(page, 'capture-debug-final');
await app.close();
console.log('done');
