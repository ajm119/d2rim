/**
 * tools/verify-camera-arm.mjs
 *
 * Measures the third-person spring arm at every zone spawn, in a real browser,
 * and says what the arm's sphere cast hit when it collapsed.
 *
 * The bug this was written for: a cold boot into the Rogue Encampment put the
 * camera 0.38 m behind the player, so the Barbarian filled half the opening
 * frame. A unit test cannot see it — the collapse needs the real collider set,
 * which only exists once a zone has built into a live physics world.
 *
 * Usage: `node tools/verify-camera-arm.mjs [outDir] [--shots] [--frames=N]`
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

import { chromium } from 'playwright';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const OUT = args[0] ?? '/tmp/camera-arm';
const SHOTS = process.argv.includes('--shots');
const FRAMES = Number(
  (process.argv.find((a) => a.startsWith('--frames=')) ?? '--frames=90').split('=')[1],
);
const ONLY = (process.argv.find((a) => a.startsWith('--only=')) ?? '--only=')
  .split('=')[1]
  .split(',')
  .filter(Boolean);
/**
 * The shortest arm any spawn may produce, metres.
 *
 * Matches `READABLE_ARM` in `tests/character.cameraRig.test.ts`. At 1 m the
 * Barbarian's shoulders are most of a 55°-FOV frame; the opening shot of a zone
 * is exactly where that is least affordable.
 */
const READABLE_ARM = 1.0;
const failures = [];
const PORT = 5251;
const WIDTH = 1280;
const HEIGHT = 720;

mkdirSync(OUT, { recursive: true });

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'],
  { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((resolve) => {
  server.stdout.on('data', (chunk) => {
    if (String(chunk).includes('Local:')) resolve();
  });
  setTimeout(resolve, 15_000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });
page.setDefaultNavigationTimeout(240_000);
page.setDefaultTimeout(240_000);
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

const goto = async (query) => {
  logs.length = 0;
  const url = `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&fade=0&${query}`;
  process.stdout.write(`\n  ...loading ${url}\n`);
  const startedAt = Date.now();
  await page.goto(url, { waitUntil: 'load' });
  await page.evaluate(() => window.__d2rim.ready);
  process.stdout.write(`  ...ready in ${((Date.now() - startedAt) / 1000).toFixed(1)}s\n`);
};

const step = async (n) => {
  const startedAt = Date.now();
  await page.evaluate((count) => window.__d2rim.engine.stepFrames(count), n);
  process.stdout.write(`  ...${n} frames in ${((Date.now() - startedAt) / 1000).toFixed(1)}s\n`);
};

/**
 * Read the arm, and re-run the arm's own sphere cast by hand so the obstacle
 * has a name. Reporting "the arm is 0.38 m" identifies a symptom; reporting
 * "the arm is 0.38 m because it hit the prop `camp.tent.0`" identifies a bug.
 */
const probe = () =>
  page.evaluate(() => {
    const d2 = window.__d2rim;
    const ctx = d2.ctx;
    const THREE = d2.three;
    const rig = ctx.services.tryGet('character.cameraRig');
    const player = ctx.services.tryGet('character.player');
    const physics = ctx.services.tryGet('physics.world');
    const pos = player?.object?.position ?? player?.position ?? { x: 0, y: 0, z: 0 };
    const cam = ctx.camera.position;

    // Rebuild the arm's own query from the rig's public state.
    const yaw = player?.yaw ?? 0;
    const pitch = player?.pitch ?? 0;
    const cosPitch = Math.cos(pitch);
    const forward = new THREE.Vector3(
      -Math.sin(yaw) * cosPitch,
      Math.sin(pitch),
      -Math.cos(yaw) * cosPitch,
    );
    const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    const pivotHeight = rig?.pivotHeight ?? player?.height * 0.86;
    const origin = new THREE.Vector3(pos.x, pos.y + pivotHeight, pos.z).addScaledVector(right, 0.42);
    const direction = forward.clone().multiplyScalar(-1);

    const hits = [];
    if (physics?.ready) {
      for (const [label, opts] of [
        ['camera-layer', { layers: 1 << 6, solidOnly: true }],
        ['all-layers', { layers: 0xffff, solidOnly: false }],
      ]) {
        const hit = physics.sphereCast(origin, direction, 0.24, 6, opts);
        hits.push({
          query: label,
          distance: hit === null ? null : hit.distance,
          point: hit === null ? null : { x: hit.point.x, y: hit.point.y, z: hit.point.z },
          record:
            hit?.record == null
              ? null
              : { kind: hit.record.kind, label: hit.record.label, sensor: hit.collider.isSensor() },
        });
      }
    }

    // Every collider within 3 m of the probe origin, so a hit at distance 0
    // (already overlapping) can still be attributed.
    const near = [];
    for (const record of physics?.colliders ?? []) {
      const t = record.collider.translation();
      const d = Math.hypot(t.x - origin.x, t.y - origin.y, t.z - origin.z);
      if (d < 3.5) {
        near.push({
          kind: record.kind,
          label: record.label,
          sensor: record.collider.isSensor(),
          shape: record.collider.shape?.type ?? -1,
          halfExtents: record.collider.halfExtents?.() ?? null,
          radius: record.collider.radius?.() ?? null,
          at: { x: +t.x.toFixed(2), y: +t.y.toFixed(2), z: +t.z.toFixed(2) },
          distance: +d.toFixed(2),
        });
      }
    }
    near.sort((a, b) => a.distance - b.distance);

    return {
      zoneId: d2.zones.activeId,
      arm: rig?.armLength ?? null,
      restArm: rig?.restLength ?? null,
      minArm: rig?.minLength ?? null,
      obstruction: rig?.obstruction ?? null,
      mode: rig?.mode ?? null,
      player: { x: +pos.x.toFixed(2), y: +pos.y.toFixed(2), z: +pos.z.toFixed(2), yaw: +yaw.toFixed(3) },
      camera: { x: +cam.x.toFixed(2), y: +cam.y.toFixed(2), z: +cam.z.toFixed(2) },
      separation: +Math.hypot(cam.x - pos.x, cam.y - (pos.y + pivotHeight), cam.z - pos.z).toFixed(2),
      probeOrigin: { x: +origin.x.toFixed(2), y: +origin.y.toFixed(2), z: +origin.z.toFixed(2) },
      hits,
      near: near.slice(0, 12),
      colliders: physics?.colliders.length ?? 0,
    };
  });

const shoot = async (name) => {
  if (!SHOTS) return;
  const frame = await page.evaluate(
    async ([w, h]) => {
      const ctx = window.__d2rim.ctx;
      const shot = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, w, h);
      let binary = '';
      const bytes = new Uint8Array(shot);
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    },
    [WIDTH, HEIGHT],
  );
  const file = join(OUT, `${name}.png`);
  writeFileSync(file, Buffer.from(frame, 'base64'));
  process.stdout.write(`  ...wrote ${file}\n`);
};

const results = {};

const measure = async (label, query, { travel } = {}) => {
  if (ONLY.length > 0 && !ONLY.includes(label)) return;
  await goto(query);
  if (travel !== undefined) {
    await page.evaluate(async (t) => {
      await window.__d2rim.zones.travelTo(t.zone, t.entry);
    }, travel);
  }
  // Frame 1: the very first thing the player sees on a cold boot.
  await step(1);
  const first = await probe();
  await shoot(`${label}-frame1`);
  await step(FRAMES - 1);
  const settled = await probe();
  await shoot(`${label}-settled`);
  results[label] = { first, settled };
  console.log(
    `\n== ${label} ==\n` +
      `  frame 1   arm ${String(first.arm?.toFixed(2))} m  separation ${first.separation} m\n` +
      `  frame ${FRAMES}  arm ${String(settled.arm?.toFixed(2))} m  separation ${settled.separation} m\n` +
      `  rest ${String(settled.restArm?.toFixed?.(2))} m  obstruction ${String(settled.obstruction)}\n` +
      `  player ${JSON.stringify(settled.player)}  camera ${JSON.stringify(settled.camera)}\n` +
      `  hits ${JSON.stringify(settled.hits, null, 2)}\n` +
      `  nearest colliders ${JSON.stringify(settled.near.slice(0, 8), null, 2)}`,
  );

  // The assertion the whole tool exists for, on the frame that matters most:
  // the first one a player sees. A camera that recovers by frame 90 has already
  // shown them the inside of the Barbarian's head.
  for (const [when, sample] of [
    ['frame 1', first],
    [`frame ${FRAMES}`, settled],
  ]) {
    const ok = (sample.arm ?? 0) >= READABLE_ARM;
    console.log(
      `${ok ? 'PASS' : 'FAIL'}  ${label} ${when} arm ${String(sample.arm?.toFixed(2))} m ` +
        `>= ${READABLE_ARM} m`,
    );
    if (!ok) failures.push(`${label} ${when}`);
  }
};

await measure('encampment-cold-boot', 'zone=encampment');
await measure('bloodmoor', 'zone=bloodMoor');
await measure('den', 'zone=denOfEvil');
await measure('encampment-from-gate', 'zone=encampment', {
  travel: { zone: 'encampment', entry: 'gate' },
});
await measure('bloodmoor-from-camp', 'zone=encampment', {
  travel: { zone: 'bloodMoor', entry: 'from-camp' },
});
await measure('den-from-moor', 'zone=bloodMoor', {
  travel: { zone: 'denOfEvil', entry: 'cave-mouth' },
});

writeFileSync(join(OUT, 'report.json'), JSON.stringify(results, null, 2));
console.log(`\nwrote ${join(OUT, 'report.json')}`);

await browser.close();
server.kill('SIGTERM');
if (failures.length > 0) {
  console.error(`\n${failures.length} arm check(s) failed: ${failures.join(', ')}`);
  process.exit(1);
}
console.log('\nall arm checks passed');
process.exit(0);
