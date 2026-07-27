/** Where is the enemy's blade during its own hit window, relative to the player capsule? */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5238', '--strictPort', '--host', '127.0.0.1'],
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
await page.goto('http://127.0.0.1:5238/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 20);

const out = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const T = d2.three;
  const svc = d2.ctx.services;
  const p = svc.get('character.player');
  const phys = svc.get('physics.world');
  const dir = svc.get('ai.director');
  const combat = svc.get('combat');
  const e = dir.enemies[0];

  // Stand the player just inside the skeleton's reach and let it come to us.
  const ef = e.footPosition(new T.Vector3());
  const g = phys.groundHeight(ef.x + 1.1, ef.z) ?? ef.y;
  p.teleport(ef.x + 1.1, g + 0.1, ef.z);

  const bone = (root, want) => {
    let found = null;
    root.traverse((n) => {
      if (found === null && n.name.toLowerCase().replace(/[^a-z0-9]/g, '') === want) found = n;
    });
    return found;
  };
  const hand = bone(e.object, 'handr');
  const elbow = bone(e.object, 'lowerarmr');
  if (hand === null || elbow === null) return { error: 'no arm bones', names: [] };

  let windowOpen = false;
  const off = e.graph.onAnimationEvent((ev) => {
    if (ev.name === 'hit.open') windowOpen = true;
    if (ev.name === 'hit.close') windowOpen = false;
  });

  const h = new T.Vector3();
  const el = new T.Vector3();
  const tip = new T.Vector3();
  const foot = new T.Vector3();
  const samples = [];
  let openFrames = 0;
  const hits = [];
  d2.ctx.events.on('combat:hit', (ev) =>
    hits.push({ target: ev.targetLabel, result: ev.outcome.result, total: ev.outcome.total }),
  );

  for (let i = 0; i < 260; i++) {
    await d2.engine.stepFrames(1);
    if (!windowOpen) continue;
    openFrames++;
    hand.getWorldPosition(h);
    elbow.getWorldPosition(el);
    tip.copy(h).sub(el);
    if (tip.lengthSq() > 1e-8) tip.normalize().multiplyScalar(e.profile.reach).add(h);
    else tip.copy(h);
    combat.self.footPosition(foot);
    const r = combat.self.hitRadius;
    const hh = combat.self.hitHeight;
    // Distance from the blade segment (elbow -> tip) to the player capsule axis.
    const base = new T.Vector3(foot.x, foot.y + r, foot.z);
    const top = new T.Vector3(foot.x, foot.y + hh - r, foot.z);
    let best = Infinity;
    for (let k = 0; k <= 24; k++) {
      const q = new T.Vector3().lerpVectors(el, tip, k / 24);
      const t = Math.max(
        0,
        Math.min(
          1,
          new T.Vector3().subVectors(q, base).dot(new T.Vector3().subVectors(top, base)) /
            Math.max(1e-6, new T.Vector3().subVectors(top, base).lengthSq()),
        ),
      );
      const axis = new T.Vector3().lerpVectors(base, top, t);
      best = Math.min(best, q.distanceTo(axis));
    }
    samples.push({
      sep: +e.footPosition(new T.Vector3()).distanceTo(p.position).toFixed(2),
      bladeGap: +(best - r - 0.13).toFixed(3),
      tipY: +tip.y.toFixed(2),
      footY: +foot.y.toFixed(2),
      state: e.state,
    });
  }
  off();
  return {
    variant: e.profile.variant,
    reach: e.profile.reach,
    playerHitRadius: combat.self.hitRadius,
    playerHitHeight: +combat.self.hitHeight.toFixed(2),
    openFrames,
    hits,
    minBladeGap: samples.length ? Math.min(...samples.map((s) => s.bladeGap)) : null,
    samples: samples.slice(0, 30),
  };
});
console.log(JSON.stringify(out, null, 1));
console.log('pageErrors:', errs);
await browser.close();
server.kill();
process.exit(0);
