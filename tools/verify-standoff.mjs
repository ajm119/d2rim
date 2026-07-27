/** Re-check: do skeletons now hold a stand-off instead of burrowing into the player? */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5235', '--strictPort', '--host', '127.0.0.1'],
  { cwd: '/home/user/d2rim', stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((r) => {
  server.stdout.on('data', (c) => String(c).includes('Local:') && r());
  setTimeout(r, 12000);
});
const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: 256, height: 144 } });
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));
await page.goto('http://127.0.0.1:5235/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);

const out = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const V = d2.three.Vector3;
  const svc = d2.ctx.services;
  const p = svc.get('character.player');
  const dir = svc.get('ai.director');
  const combat = svc.get('combat');
  const rows = [];
  for (let i = 0; i <= 640; i++) {
    await d2.engine.stepFrames(1);
    if (i % 40 === 0) {
      rows.push({
        f: i,
        hp: +combat.vitals.health.value.toFixed(1),
        e: dir.enemies
          .filter((x) => x.alive)
          .map((x) => ({
            s: x.state,
            d: +x.footPosition(new V()).distanceTo(p.position).toFixed(2),
          })),
      });
    }
  }
  return rows;
});
for (const r of out) {
  console.log(r.f, 'hp', r.hp, '|', r.e.map((e) => `${e.s.slice(0, 5)}:${e.d}`).join(' '));
}
console.log('pageErrors:', errs);
await browser.close();
server.kill();
process.exit(0);
