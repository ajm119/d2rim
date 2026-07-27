/** At what separation does an enemy swing actually connect with the player? */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5236', '--strictPort', '--host', '127.0.0.1'],
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
await page.goto('http://127.0.0.1:5236/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);

await page.evaluate(() => {
  const d2 = window.__d2rim;
  const V = d2.three.Vector3;
  const svc = d2.ctx.services;
  const p = svc.get('character.player');
  const dir = svc.get('ai.director');
  window.__hits = [];
  window.__handReach = [];
  d2.ctx.events.on('combat:hit', (e) => {
    const near = dir.enemies
      .filter((x) => x.alive)
      .map((x) => x.footPosition(new V()).distanceTo(p.position))
      .sort((a, b) => a - b)[0];
    window.__hits.push({
      target: e.targetLabel,
      result: e.outcome.result,
      total: e.outcome.total,
      separation: +(near ?? -1).toFixed(2),
    });
  });
  // Every frame, record how close the nearest attacking skeleton's right hand
  // gets to the player. Blade tip = hand + profile reach along the forearm.
  window.__sampleHands = () => {
    for (const e of dir.enemies) {
      if (!e.alive || (e.state !== 'attack' && e.state !== 'telegraph')) continue;
      let hand = null;
      e.object.traverse((n) => {
        if (hand === null && n.name.toLowerCase().replace(/[^a-z0-9]/g, '') === 'handr') hand = n;
      });
      if (hand === null) continue;
      const w = new V();
      hand.getWorldPosition(w);
      window.__handReach.push({
        state: e.state,
        handToPlayer: +w.distanceTo(p.position).toFixed(3),
        reach: e.profile.reach,
        sep: +e.footPosition(new V()).distanceTo(p.position).toFixed(2),
      });
    }
  };
});

// Let a skeleton arrive, then walk the player straight into it.
await page.evaluate(async () => {
  for (let i = 0; i < 420; i++) {
    await window.__d2rim.engine.stepFrames(1);
    window.__sampleHands();
  }
});
await page.keyboard.down('w');
const walked = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const V = d2.three.Vector3;
  const p = d2.ctx.services.get('character.player');
  const dir = d2.ctx.services.get('ai.director');
  const combat = d2.ctx.services.get('combat');
  const rows = [];
  for (let i = 0; i < 420; i++) {
    await d2.engine.stepFrames(1);
    window.__sampleHands();
    if (i % 40 === 0) {
      const near = dir.enemies
        .filter((x) => x.alive)
        .map((x) => x.footPosition(new V()).distanceTo(p.position))
        .sort((a, b) => a - b)[0];
      rows.push({ i, hp: +combat.vitals.health.value.toFixed(1), near: +(near ?? -1).toFixed(2) });
    }
  }
  return rows;
});
await page.keyboard.up('w');
const res = await page.evaluate(() => ({
  walkedHits: window.__hits,
  handSamples: window.__handReach.length,
  minHandToPlayer: window.__handReach.length
    ? Math.min(...window.__handReach.map((h) => h.handToPlayer))
    : null,
  minSepDuringAttack: window.__handReach.length
    ? Math.min(...window.__handReach.map((h) => h.sep))
    : null,
  reach: window.__handReach[0]?.reach ?? null,
}));
console.log(JSON.stringify({ walked, ...res }, null, 1));
console.log('pageErrors:', errs);
await browser.close();
server.kill();
process.exit(0);
