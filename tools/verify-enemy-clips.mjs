/** Do the skeletons' semantic attack states resolve to real clips, and do their hit events fire? */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5237', '--strictPort', '--host', '127.0.0.1'],
  { cwd: '/home/user/d2rim', stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((r) => {
  server.stdout.on('data', (c) => String(c).includes('Local:') && r());
  setTimeout(r, 12000);
});
const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: 128, height: 72 } });
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));
await page.goto('http://127.0.0.1:5237/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 20);

const out = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const dir = d2.ctx.services.get('ai.director');
  const e = dir.enemies[0];
  const g = e.graph;
  const table = g ? Object.fromEntries(g.clipTable) : null;
  const wanted = ['attack', 'attack.slice', 'attack.heavy', 'attack.stab', 'attack.spin', 'hit', 'death'];
  const resolved = {};
  for (const w of wanted) resolved[w] = table ? (table[w] ?? null) : 'NO GRAPH';

  // Play each attack directly and watch for the authored hit markers.
  const events = [];
  const off = g.onAnimationEvent((ev) => events.push(`${ev.action}:${ev.name}@${ev.normalizedTime.toFixed(2)}`));
  g.playAction('attack', {
    layer: 'full',
    events: [
      { name: 'hit.open', at: 0.34 },
      { name: 'hit.close', at: 0.56 },
    ],
  });
  for (let i = 0; i < 80; i++) await d2.engine.stepFrames(1);
  off();

  return {
    variant: e.profile.variant,
    clipCount: g ? g.clipNames.length : 0,
    resolved,
    hasAttackClips: g ? g.clipNames.filter((n) => /attack/i.test(n)).slice(0, 12) : [],
    events,
  };
});
console.log(JSON.stringify(out, null, 1));
console.log('pageErrors:', errs);
await browser.close();
server.kill();
process.exit(0);
