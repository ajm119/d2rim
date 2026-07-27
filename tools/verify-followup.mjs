/** Focused follow-up: natural AI approach, camera arm, foot-plant series, real wall collision. */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import sharp from 'sharp';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const OUT = process.argv[2] ?? '/tmp/verify2';
const W = 320;
const H = 180;
mkdirSync(OUT, { recursive: true });
const t0 = Date.now();
const mark = (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5234', '--strictPort', '--host', '127.0.0.1'],
  { cwd: '/home/user/d2rim', stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((r) => {
  server.stdout.on('data', (c) => String(c).includes('Local:') && r());
  setTimeout(r, 12000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: W, height: H } });
const boot = [];
page.on('console', (m) => boot.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => boot.push(`pageerror: ${e.message}`));
await page.goto('http://127.0.0.1:5234/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
mark('ready');

const report = {};
report.bootLog = boot.filter((l) => /calibrat|CameraRig|stride|FootIK|warn|error|combat|ai/i.test(l));

await page.evaluate(() => {
  const d2 = window.__d2rim;
  const svc = () => d2.ctx.services;
  const P = () => svc().get('character.player');
  let bones = null;
  const findBones = () => {
    const o = {};
    P().object.traverse((n) => {
      const k = n.name.toLowerCase().replace(/[^a-z0-9]/g, '');
      if (k === 'footl' || k === 'footr') o[k] = n;
    });
    return o;
  };
  const rig = () => d2.engine.getModule('character.cameraRig');
  window.__f = {
    snap() {
      const p = P();
      const a = p.animation;
      if (!bones) bones = findBones();
      const V = d2.three.Vector3;
      const wp = (n) => {
        const v = new V();
        n.getWorldPosition(v);
        return [+v.x.toFixed(4), +v.y.toFixed(4), +v.z.toFixed(4)];
      };
      return {
        pos: p.position.toArray().map((v) => +v.toFixed(4)),
        speed: +p.speed.toFixed(3),
        state: a?.state,
        phase: +(a?.phase ?? 0).toFixed(3),
        cycleRate: +(a?.cycleRate ?? 0).toFixed(3),
        arm: rig() ? +rig().armLength.toFixed(3) : null,
        camDist: +d2.ctx.camera.position.distanceTo(p.position).toFixed(3),
        fl: bones.footl ? wp(bones.footl) : null,
        fr: bones.footr ? wp(bones.footr) : null,
      };
    },
    gaits() {
      const p = P();
      return { gait: p.gait, height: +p.height.toFixed(3), strides: Object.fromEntries([...(p.animation?.strideTable ?? [])].map(([k, v]) => [k, +v.toFixed(3)])) };
    },
    enemySnap() {
      const V = d2.three.Vector3;
      const p = P();
      return svc()
        .get('ai.director')
        .enemies.map((e) => {
          const f = e.footPosition(new V());
          return {
            id: e.id,
            state: e.state,
            hp: +e.health.toFixed(1),
            alive: e.alive,
            x: +f.x.toFixed(3),
            z: +f.z.toFixed(3),
            y: +f.y.toFixed(3),
            d: +f.distanceTo(p.position).toFixed(2),
          };
        });
    },
  };
});

await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 30);
report.gaits = await page.evaluate(() => window.__f.gaits());
mark('gaits');

/* --- 1. Do the naturally spawned skeletons close on a standing player? --- */
report.naturalAI = await page.evaluate(async () => {
  const out = [];
  for (let i = 0; i <= 600; i++) {
    if (i % 30 === 0) out.push({ f: i, e: window.__f.enemySnap() });
    await window.__d2rim.engine.stepFrames(1);
  }
  return out;
});
mark('natural AI done');

/* --- 2. Camera arm + foot plant while walking ---------------------------- */
await page.keyboard.down('w');
report.walkSeries = await page.evaluate(async () => {
  const out = [];
  for (let i = 0; i < 90; i++) {
    await window.__d2rim.engine.stepFrames(1);
    out.push(window.__f.snap());
  }
  return out;
});
await page.keyboard.up('w');
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 5);
mark('walk series done');

/* --- 3. Terrain following through real engine frames --------------------- */
report.terrain = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const p = d2.ctx.services.get('character.player');
  const phys = d2.ctx.services.get('physics.world');
  const out = [];
  for (let i = 0; i < 180; i++) {
    await d2.engine.stepFrames(1);
    if (i % 20 === 0) {
      const g = phys.groundHeight(p.position.x, p.position.z);
      out.push({
        i,
        y: +p.position.y.toFixed(3),
        ground: g === null ? null : +g.toFixed(3),
        gap: g === null ? null : +(p.position.y - g).toFixed(3),
        grounded: p.grounded,
      });
    }
  }
  return out;
});
mark('terrain done');

/* --- 4. Wall collision against a real prop collider ---------------------- */
report.wall = await page.evaluate(() => {
  const d2 = window.__d2rim;
  const T = d2.three;
  const p = d2.ctx.services.get('character.player');
  const phys = d2.ctx.services.get('physics.world');
  const records = phys.colliders.filter((c) => /ruinedWall/i.test(c.label ?? ''));
  if (records.length === 0) return { error: 'no ruinedWall colliders', labels: phys.colliders.slice(0, 8).map((c) => c.label) };
  // Pick the tallest course near the middle of the wall.
  const rec = records[Math.floor(records.length / 2)];
  const t = rec.collider.translation();
  const half = rec.collider.halfExtents ? rec.collider.halfExtents() : null;
  const centre = new T.Vector3(t.x, t.y, t.z);
  const results = [];
  for (const dir of [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ]) {
    const start = new T.Vector3(centre.x + dir[0] * 4, 0, centre.z + dir[1] * 4);
    const g = phys.groundHeight(start.x, start.z) ?? centre.y;
    p.controller.setPosition(new T.Vector3(start.x, g + 0.15, start.z));
    let minDist = Infinity;
    let blocked = 0;
    for (let i = 0; i < 200; i++) {
      const v = new T.Vector3(centre.x - p.controller.position.x, 0, centre.z - p.controller.position.z);
      if (v.lengthSq() > 1e-6) v.normalize().multiplyScalar(2.5);
      const r = p.controller.move(v, 1 / 60);
      if (r.blocked) blocked++;
      const d = Math.hypot(r.position.x - centre.x, r.position.z - centre.z);
      minDist = Math.min(minDist, d);
    }
    // Does the capsule end up overlapping any prop collider?
    const overlap = phys.overlapSphere(p.controller.position, p.controller.dimensions.radius);
    results.push({
      dir,
      minDistanceToBoxCentre: +minDist.toFixed(3),
      blockedFrames: blocked,
      overlapsProp: overlap ? (overlap.label ?? overlap.kind) : null,
    });
  }
  return {
    colliderCount: records.length,
    chosenLabel: rec.label,
    centre: centre.toArray().map((v) => +v.toFixed(3)),
    halfExtents: half ? [half.x, half.y, half.z].map((v) => +v.toFixed(3)) : null,
    capsuleRadius: +p.controller.dimensions.radius.toFixed(3),
    results,
  };
});
mark('wall done');

const shot = async (label, scale = 2) => {
  const f = await page.evaluate(
    async ([w, h]) => {
      const ctx = window.__d2rim.ctx;
      const s = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, w, h);
      let b = '';
      for (let i = 0; i < s.pixels.length; i += 0x8000)
        b += String.fromCharCode.apply(null, s.pixels.subarray(i, i + 0x8000));
      return { width: s.width, height: s.height, data: btoa(b) };
    },
    [W * scale, H * scale],
  );
  await sharp(Buffer.from(f.data, 'base64'), {
    raw: { width: f.width, height: f.height, channels: 4 },
  })
    .png()
    .toFile(`${OUT}/${label}.png`);
  mark(`shot ${label}`);
};

/* --- 5. Observer-camera poses, to judge the animation without the rig ---- */
await page.evaluate(() => {
  const d2 = window.__d2rim;
  const p = d2.ctx.services.get('character.player');
  const T = d2.three;
  window.__observe = () => {
    const c = d2.ctx.camera;
    c.position.set(p.position.x + 3.0, p.position.y + 1.2, p.position.z + 3.0);
    c.lookAt(new T.Vector3(p.position.x, p.position.y + 0.85, p.position.z));
    c.updateMatrixWorld(true);
  };
});
await page.keyboard.down('w');
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 14);
await page.evaluate(() => window.__observe());
await shot('obs-walk-a');
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 8);
await page.evaluate(() => window.__observe());
await shot('obs-walk-b');
await page.keyboard.up('w');
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 45);
await page.evaluate(() => window.__observe());
await shot('obs-idle');
await page.evaluate(() => window.__d2rim.ctx.services.get('combat').press('light'));
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 13);
await page.evaluate(() => window.__observe());
await shot('obs-attack');

report.pageErrors = boot.filter((l) => l.startsWith('pageerror'));
writeFileSync(`${OUT}/report.json`, JSON.stringify(report, null, 2));
console.log('WROTE report');
await browser.close();
server.kill();
process.exit(0);
