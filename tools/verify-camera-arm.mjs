/**
 * tools/verify-camera-arm.mjs
 *
 * Measures the third-person spring arm at every zone spawn, in a real browser,
 * and says what the arm's sphere cast hit when it collapsed.
 *
 * The bug this was written for: a cold boot into the Rogue Encampment put the
 * camera 0.25 m behind the player's shoulder, so the Barbarian filled half the
 * opening frame. No unit test can see it — the collapse needs the real collider
 * set, which only exists once a zone has built into a live physics world.
 *
 * ### One boot, then travel
 *
 * Every measurement used to reload the page, and on this container's software
 * rasteriser a cold boot costs minutes: six measurements did not fit in half an
 * hour. Booting once and using `zones.travelTo` for the rest measures the same
 * thing — the arm cares about the collider set, and travel rebuilds it in full
 * — for a fraction of the time. The cold boot is still measured, because the
 * cold boot is the frame the bug was reported on.
 *
 * Usage:
 *   node tools/verify-camera-arm.mjs [outDir] [--shots] [--frames=N]
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

import { chromium } from 'playwright';
import sharp from 'sharp';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';
import { guardImageFile } from './capture/frame-guard.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const OUT = args[0] ?? '/tmp/camera-arm';
const SHOTS = process.argv.includes('--shots');
const FRAMES = Number(
  (process.argv.find((a) => a.startsWith('--frames=')) ?? '--frames=24').split('=')[1],
);
const PORT = Number((process.argv.find((a) => a.startsWith('--port=')) ?? '--port=5251').split('=')[1]);
const WIDTH = 1280;
const HEIGHT = 720;

/**
 * The shortest arm any spawn may produce, metres.
 *
 * Matches `READABLE_ARM` in `tests/character.cameraRig.test.ts`. At 1 m the
 * Barbarian's shoulders are most of a 55°-FOV frame, and the opening shot of a
 * zone is exactly where that is least affordable.
 */
const READABLE_ARM = 1.0;
const failures = [];

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
page.setDefaultNavigationTimeout(300_000);
page.setDefaultTimeout(300_000);
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

const say = (line) => process.stdout.write(`${line}\n`);

const goto = async (query) => {
  logs.length = 0;
  const url = `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&fade=0&enemies=0&${query}`;
  say(`  ...loading ${url}`);
  const startedAt = Date.now();
  await page.goto(url, { waitUntil: 'load' });
  await page.evaluate(() => window.__d2rim.ready);
  say(`  ...ready in ${((Date.now() - startedAt) / 1000).toFixed(1)}s`);
};

const step = async (n) => {
  const startedAt = Date.now();
  await page.evaluate((count) => window.__d2rim.engine.stepFrames(count), n);
  say(`  ...${n} frames in ${((Date.now() - startedAt) / 1000).toFixed(1)}s`);
};

/**
 * Read the arm, and re-run the arm's own sphere cast by hand so the obstacle
 * has a name. Reporting "the arm is 0.25 m" identifies a symptom; reporting
 * "the arm is 0.25 m because it started inside `camp.torchposts`" identifies a
 * bug.
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

    const yaw = player?.yaw ?? 0;
    const pitch = player?.pitch ?? 0;
    const cosPitch = Math.cos(pitch);
    const forward = new THREE.Vector3(
      -Math.sin(yaw) * cosPitch,
      Math.sin(pitch),
      -Math.cos(yaw) * cosPitch,
    );
    const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    const pivotHeight = rig?.pivotHeight ?? (player?.height ?? 1.8) * 0.86;
    const origin = new THREE.Vector3(pos.x, pos.y + pivotHeight, pos.z).addScaledVector(right, 0.42);
    const direction = forward.clone().multiplyScalar(-1);

    const hits = [];
    if (physics?.ready) {
      for (const [label, opts] of [
        ['camera-layer', { layers: 1 << 6, solidOnly: true }],
        ['everything', { layers: 0xffff, solidOnly: false }],
      ]) {
        const hit = physics.sphereCast(origin, direction, 0.24, 6, opts);
        hits.push({
          query: label,
          distance: hit === null ? null : Number(hit.distance.toFixed(3)),
          record:
            hit?.record == null
              ? null
              : { kind: hit.record.kind, label: hit.record.label, sensor: hit.collider.isSensor() },
        });
      }
    }

    // Every collider whose centre is within 4 m of the probe origin, so a hit
    // at distance 0 — the sweep starting already inside something — can still
    // be attributed to an object with a name.
    const near = [];
    for (const record of physics?.colliders ?? []) {
      const t = record.collider.translation();
      const d = Math.hypot(t.x - origin.x, t.y - origin.y, t.z - origin.z);
      if (d > 4) continue;
      let extents = null;
      try {
        const half = record.collider.halfExtents?.();
        extents = half == null ? null : [+half.x.toFixed(2), +half.y.toFixed(2), +half.z.toFixed(2)];
      } catch {
        extents = null;
      }
      near.push({
        kind: record.kind,
        label: record.label,
        sensor: record.collider.isSensor(),
        halfExtents: extents,
        at: { x: +t.x.toFixed(2), y: +t.y.toFixed(2), z: +t.z.toFixed(2) },
        distance: +d.toFixed(2),
      });
    }
    near.sort((a, b) => a.distance - b.distance);

    return {
      zoneId: d2.zones.activeId,
      arm: rig?.armLength ?? null,
      restArm: rig?.restLength ?? null,
      minArm: rig?.minLength ?? null,
      obstruction: rig?.obstruction ?? null,
      player: {
        x: +pos.x.toFixed(2),
        y: +pos.y.toFixed(2),
        z: +pos.z.toFixed(2),
        yaw: +yaw.toFixed(3),
      },
      camera: { x: +cam.x.toFixed(2), y: +cam.y.toFixed(2), z: +cam.z.toFixed(2) },
      separation: +Math.hypot(
        cam.x - pos.x,
        cam.y - (pos.y + pivotHeight),
        cam.z - pos.z,
      ).toFixed(2),
      hits,
      near: near.slice(0, 10),
      colliders: physics?.colliders.length ?? 0,
    };
  });

const shoot = async (name) => {
  if (!SHOTS) return null;
  const startedAt = Date.now();
  // `captureFrame` hands back tightly packed RGBA bytes, not an encoded image —
  // writing them straight to a `.png` produces a file no decoder will open, and
  // the failure is silent. sharp does the encoding, and `guardImageFile` then
  // decodes it back off disk, because a capture tool that can write a blank or
  // unreadable frame is worse than one that does not capture at all.
  const frame = await page.evaluate(
    async ([w, h]) => {
      const ctx = window.__d2rim.ctx;
      const shot = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, w, h);
      let binary = '';
      for (let i = 0; i < shot.pixels.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, shot.pixels.subarray(i, i + 0x8000));
      }
      return { width: shot.width, height: shot.height, data: btoa(binary) };
    },
    [WIDTH, HEIGHT],
  );
  const file = join(OUT, `${name}.png`);
  await sharp(Buffer.from(frame.data, 'base64'), {
    raw: { width: frame.width, height: frame.height, channels: 4 },
  })
    .png()
    .toFile(file);
  const guard = await guardImageFile(file);
  say(
    `  ...wrote ${file} in ${((Date.now() - startedAt) / 1000).toFixed(1)}s ` +
      `(${guard.ok ? 'not blank' : `BLANK: ${guard.failures.join('; ')}`})`,
  );
  if (!guard.ok) failures.push(`${name} capture is blank`);
  return file;
};

const results = {};

const measure = async (label) => {
  // Frame 1: literally the first thing a player sees.
  await step(1);
  const first = await probe();
  await step(FRAMES - 1);
  const settled = await probe();
  await shoot(label);
  results[label] = { first, settled };

  say(
    `\n== ${label} (${settled.zoneId}) ==\n` +
      `  frame 1    arm ${String(first.arm?.toFixed(2))} m   separation ${first.separation} m\n` +
      `  frame ${FRAMES}   arm ${String(settled.arm?.toFixed(2))} m   separation ${settled.separation} m\n` +
      `  rest ${String(settled.restArm?.toFixed?.(2) ?? settled.restArm)} m   ` +
      `floor ${String(settled.minArm?.toFixed?.(2) ?? settled.minArm)} m   ` +
      `obstruction ${String(settled.obstruction)}\n` +
      `  player ${JSON.stringify(settled.player)}\n` +
      `  camera ${JSON.stringify(settled.camera)}\n` +
      `  probe hits ${JSON.stringify(settled.hits)}\n` +
      `  nearest colliders ${JSON.stringify(settled.near.slice(0, 6))}`,
  );

  // The assertion the tool exists for, on both the first frame and the settled
  // one. A camera that recovers by frame 24 has already shown the player the
  // inside of the Barbarian's head.
  for (const [when, sample] of [
    ['frame 1', first],
    [`frame ${FRAMES}`, settled],
  ]) {
    const ok = (sample.arm ?? 0) >= READABLE_ARM;
    say(
      `${ok ? 'PASS' : 'FAIL'}  ${label} ${when}: arm ${String(sample.arm?.toFixed(2))} m ` +
        `>= ${READABLE_ARM} m`,
    );
    if (!ok) failures.push(`${label} ${when}`);
  }
};

const travel = async (zone, entry) => {
  say(`\n  ...travelling to ${zone}/${entry}`);
  const startedAt = Date.now();
  await page.evaluate(
    async (t) => {
      await window.__d2rim.zones.travelTo(t.zone, t.entry);
    },
    { zone, entry },
  );
  say(`  ...arrived in ${((Date.now() - startedAt) / 1000).toFixed(1)}s`);
};

await goto('zone=encampment');
await measure('encampment-cold-boot');

await travel('bloodMoor', 'from-camp');
await measure('bloodmoor-from-camp');

await travel('denOfEvil', 'cave-mouth');
await measure('den-cave-mouth');

await travel('encampment', 'gate');
await measure('encampment-gate');

await travel('bloodMoor', 'from-den');
await measure('bloodmoor-from-den');

writeFileSync(join(OUT, 'report.json'), JSON.stringify(results, null, 2));
say(`\nwrote ${join(OUT, 'report.json')}`);

await browser.close();
server.kill('SIGTERM');
if (failures.length > 0) {
  console.error(`\n${failures.length} arm check(s) failed: ${failures.join(', ')}`);
  process.exit(1);
}
say('\nall arm checks passed');
process.exit(0);
