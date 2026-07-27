/**
 * Independent verification harness. Written from scratch by the reviewing
 * agent; shares no logic with tools/scratch-drive.mjs or tools/combat-drive.mjs.
 *
 *   node verify.mjs [outDir]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import sharp from 'sharp';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const OUT = process.argv[2] ?? '/tmp/verify';
const W = 320;
const H = 180;
mkdirSync(OUT, { recursive: true });
const report = {};
const t0 = Date.now();
const mark = (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5233', '--strictPort', '--host', '127.0.0.1'],
  { cwd: '/home/user/d2rim', stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((r) => {
  server.stdout.on('data', (c) => String(c).includes('Local:') && r());
  setTimeout(r, 12000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: W, height: H } });
const logs = [];
page.on('console', (m) => m.type() === 'error' && logs.push(`console.error: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

await page.goto('http://127.0.0.1:5233/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
mark('ready');

/* Install my own probes. */
await page.evaluate(() => {
  const d2 = window.__d2rim;
  const svc = () => d2.ctx.services;
  const P = () => svc().get('character.player');

  const findBones = () => {
    const out = {};
    P().object.traverse((o) => {
      const n = o.name.toLowerCase().replace(/[^a-z0-9]/g, '');
      if (n === 'footl' || n === 'footr' || n === 'handr' || n === 'head') out[n] = o;
    });
    return out;
  };
  let bones = null;

  window.__v = {
    probe() {
      const p = P();
      const a = p.animation;
      if (!bones) bones = findBones();
      const V = d2.three.Vector3;
      const wp = (o) => {
        if (!o) return null;
        const v = new V();
        o.getWorldPosition(v);
        return [v.x, v.y, v.z];
      };
      return {
        pos: [p.position.x, p.position.y, p.position.z],
        vel: [p.velocity.x, p.velocity.y, p.velocity.z],
        speed: p.speed,
        grounded: p.grounded,
        gait: p.gait,
        state: a?.state ?? null,
        phase: a?.phase ?? null,
        cycleRate: a?.cycleRate ?? null,
        weights: a ? Object.fromEntries(Object.entries(a.weights).filter(([, w]) => w > 0.01)) : {},
        footl: wp(bones.footl),
        footr: wp(bones.footr),
        handr: wp(bones.handr),
        head: wp(bones.head),
      };
    },
    // Full pose fingerprint: every bone's position in body-local space.
    pose() {
      const p = P();
      const V = d2.three.Vector3;
      const inv = p.object.matrixWorld.clone().invert();
      const out = [];
      p.object.traverse((o) => {
        if (!o.isBone) return;
        const v = new V();
        o.getWorldPosition(v).applyMatrix4(inv);
        out.push([o.name, +v.x.toFixed(4), +v.y.toFixed(4), +v.z.toFixed(4)]);
      });
      out.sort((a, b) => (a[0] < b[0] ? -1 : 1));
      return out;
    },
    teleport(x, y, z) {
      P().teleport(x, y, z);
    },
    setYaw(y) {
      const p = P();
      // yaw is read-only; drive it through the same path the mouse uses.
      if ('yaw' in p && Object.getOwnPropertyDescriptor(Object.getPrototypeOf(p), 'yaw')?.set) {
        p.yaw = y;
        return true;
      }
      return false;
    },
    world() {
      const out = { walls: [], colliderKinds: {} };
      d2.ctx.scene.traverse((o) => {
        if (o.name === 'ruinedWall') {
          const b = new d2.three.Box3().setFromObject(o);
          out.walls.push({ min: b.min.toArray(), max: b.max.toArray() });
        }
      });
      const phys = svc().get('physics.world');
      for (const c of phys.colliders) out.colliderKinds[c.kind] = (out.colliderKinds[c.kind] ?? 0) + 1;
      out.colliderCount = phys.colliders.length;
      out.steps = phys.steps;
      return out;
    },
    enemies() {
      const d = svc().get('ai.director');
      return d.enemies.map((e) => ({
        id: e.id,
        state: e.state,
        hp: +e.health.toFixed(2),
        alive: e.alive,
        pos: [e.footPosition(new d2.three.Vector3()).x, 0, e.footPosition(new d2.three.Vector3()).z],
        dist: +e
          .footPosition(new d2.three.Vector3())
          .distanceTo(P().position)
          .toFixed(2),
      }));
    },
    combat() {
      const c = svc().get('combat');
      return {
        move: c.moveId,
        window: c.hitWindowOpen,
        chain: c.combo.chainIndex,
        hp: +c.vitals.health.value.toFixed(2),
        stamina: +c.stamina.toFixed(1),
        blocking: c.blocking,
      };
    },
    press(kind) {
      svc().get('combat').press(kind);
    },
    timeScale() {
      return d2.ctx.time.scale;
    },
  };
});

const step = (n) => page.evaluate((k) => window.__d2rim.engine.stepFrames(k), n);
const probe = () => page.evaluate(() => window.__v.probe());
const shot = async (label) => {
  const f = await page.evaluate(
    async ([w, h]) => {
      const ctx = window.__d2rim.ctx;
      const s = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, w, h);
      let b = '';
      for (let i = 0; i < s.pixels.length; i += 0x8000)
        b += String.fromCharCode.apply(null, s.pixels.subarray(i, i + 0x8000));
      return { width: s.width, height: s.height, data: btoa(b) };
    },
    [W, H],
  );
  await sharp(Buffer.from(f.data, 'base64'), {
    raw: { width: f.width, height: f.height, channels: 4 },
  })
    .png()
    .toFile(`${OUT}/${label}.png`);
  mark(`shot ${label}`);
};

/* ---------------- A. spawn, grounding, no fall-through ------------------- */
await step(20);
const spawnState = await probe();
report.world = await page.evaluate(() => window.__v.world());
await step(120); // 2 s of standing still
const settled = await probe();
report.A_spawn = {
  spawnPos: spawnState.pos.map((v) => +v.toFixed(3)),
  spawnGrounded: spawnState.grounded,
  after120Frames: settled.pos.map((v) => +v.toFixed(3)),
  yDrift: +(settled.pos[1] - spawnState.pos[1]).toFixed(4),
  grounded: settled.grounded,
  idleState: settled.state,
  idleWeights: settled.weights,
};
mark('A done');

/* ---------------- B. walk / run, foot-slide measurement ------------------- */
const drive = async (keys, frames) => {
  for (const k of keys) await page.keyboard.down(k);
  const samples = await page.evaluate(async (n) => {
    const out = [];
    for (let i = 0; i < n; i++) {
      await window.__d2rim.engine.stepFrames(1);
      out.push(window.__v.probe());
    }
    return out;
  }, frames);
  for (const k of keys) await page.keyboard.up(k);
  await step(2);
  return samples;
};

const analyse = (s) => {
  const dt = 1 / 60;
  let bodyTotal = 0;
  const plantSlide = [];
  const bodySpeeds = [];
  for (let i = 1; i < s.length; i++) {
    const a = s[i - 1];
    const b = s[i];
    const bd = Math.hypot(b.pos[0] - a.pos[0], b.pos[2] - a.pos[2]);
    bodyTotal += bd;
    if (!a.footl || !b.footl || !a.footr || !b.footr) continue;
    if (bd / dt < 0.15) continue; // only measure while actually moving
    bodySpeeds.push(bd / dt);
    const dl = Math.hypot(b.footl[0] - a.footl[0], b.footl[2] - a.footl[2]);
    const dr = Math.hypot(b.footr[0] - a.footr[0], b.footr[2] - a.footr[2]);
    // The planted foot is the lower one; its world motion IS the slide.
    const planted = b.footl[1] <= b.footr[1] ? dl : dr;
    plantSlide.push(planted / dt);
  }
  const mean = (x) => (x.length ? x.reduce((p, c) => p + c, 0) / x.length : 0);
  const meanBody = mean(bodySpeeds);
  const meanSlide = mean(plantSlide);
  const sorted = [...plantSlide].sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
  return {
    frames: s.length,
    displacement: +bodyTotal.toFixed(3),
    meanGroundSpeed: +meanBody.toFixed(3),
    meanPlantedFootWorldSpeed: +meanSlide.toFixed(3),
    medianPlantedFootWorldSpeed: +median.toFixed(3),
    slideRatioMean: +(meanSlide / (meanBody || 1)).toFixed(3),
    slideRatioMedian: +(median / (meanBody || 1)).toFixed(3),
    states: [...new Set(s.map((x) => x.state))],
    meanCycleRate: +mean(s.map((x) => x.cycleRate ?? 0)).toFixed(3),
    endWeights: s[s.length - 1].weights,
  };
};

const walkS = await drive(['w'], 70);
report.B_walk = analyse(walkS);
await shot('walk');

const runS = await drive(['w', 'Shift'], 80);
report.B_sprint = analyse(runS);
await shot('sprint');

const strafeS = await drive(['d'], 55);
report.B_strafe = analyse(strafeS);
await shot('strafe');

const backS = await drive(['s'], 45);
report.B_back = analyse(backS);
mark('B done');

/* pose fingerprints: idle vs walk vs sprint vs strafe */
const poseOf = async (keys, frames) => {
  for (const k of keys) await page.keyboard.down(k);
  await step(frames);
  const p = await page.evaluate(() => window.__v.pose());
  for (const k of keys) await page.keyboard.up(k);
  await step(4);
  return p;
};
const poses = {};
poses.idle = await poseOf([], 30);
poses.walk = await poseOf(['w'], 25);
poses.sprint = await poseOf(['w', 'Shift'], 40);
poses.strafe = await poseOf(['d'], 25);
const poseDiff = (a, b) => {
  let max = 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.hypot(a[i][1] - b[i][1], a[i][2] - b[i][2], a[i][3] - b[i][3]);
    max = Math.max(max, d);
    sum += d;
  }
  return { maxBoneDelta: +max.toFixed(4), meanBoneDelta: +(sum / a.length).toFixed(4) };
};
report.C_poseDeltas = {
  boneCount: poses.idle.length,
  idle_vs_walk: poseDiff(poses.idle, poses.walk),
  idle_vs_sprint: poseDiff(poses.idle, poses.sprint),
  walk_vs_sprint: poseDiff(poses.walk, poses.sprint),
  walk_vs_strafe: poseDiff(poses.walk, poses.strafe),
};
// T-pose check: in a T-pose both hands are far lateral and level with the head.
report.C_notTPose = await page.evaluate(() => {
  const p = window.__v.probe();
  return { handr: p.handr, head: p.head, footl: p.footl, footr: p.footr };
});
await shot('idle');
mark('C done');

/* ---------------- D. collision against the ruined wall ------------------- */
const wall = report.world.walls[0];
if (wall) {
  const cx = (wall.min[0] + wall.max[0]) / 2;
  const cz = (wall.min[2] + wall.max[2]) / 2;
  report.D_wall = { box: wall, centre: [cx, cz] };
  // Approach along +X and along +Z from 4 m out, whichever axis is thin.
  const runs = [];
  for (const [dx, dz, key] of [
    [-4, 0, 'x'],
    [0, -4, 'z'],
    [4, 0, 'x-'],
    [0, 4, 'z-'],
  ]) {
    await page.evaluate(
      ([x, z]) => {
        const p = window.__d2rim.ctx.services.get('character.player');
        const phys = window.__d2rim.ctx.services.get('physics.world');
        const h = phys.groundHeight(x, z) ?? 0;
        p.teleport(x, h + 0.2, z);
      },
      [cx + dx, cz + dz],
    );
    await step(20);
    const before = await probe();
    // Drive straight at the wall centre by aiming the controller: instead of
    // the keyboard (whose direction depends on camera yaw), push the capsule
    // directly through the physics controller for a clean, axis-aligned test.
    const res = await page.evaluate(
      ([tx, tz, frames]) => {
        const p = window.__d2rim.ctx.services.get('character.player');
        const cc = p.controller;
        const T = window.__d2rim.three;
        const trace = [];
        for (let i = 0; i < frames; i++) {
          const dir = new T.Vector3(tx - cc.position.x, 0, tz - cc.position.z);
          if (dir.lengthSq() > 1e-6) dir.normalize().multiplyScalar(3.0);
          const r = cc.move(dir, 1 / 60);
          trace.push({
            p: [+r.position.x.toFixed(3), +r.position.y.toFixed(3), +r.position.z.toFixed(3)],
            blocked: r.blocked,
          });
        }
        return trace;
      },
      [cx, cz, 120],
    );
    const last = res[res.length - 1];
    const blockedFrames = res.filter((r) => r.blocked).length;
    const distToCentre = Math.hypot(last.p[0] - cx, last.p[2] - cz);
    runs.push({
      approach: key,
      from: [+before.pos[0].toFixed(2), +before.pos[2].toFixed(2)],
      end: last.p,
      blockedFrames,
      finalDistanceToWallCentre: +distToCentre.toFixed(3),
      insideWallBox:
        last.p[0] > wall.min[0] &&
        last.p[0] < wall.max[0] &&
        last.p[2] > wall.min[2] &&
        last.p[2] < wall.max[2] &&
        last.p[1] < wall.max[1],
    });
  }
  report.D_wall.runs = runs;
}
mark('D done');

/* ---------------- E. terrain following + no fall-through ----------------- */
report.E_terrain = await page.evaluate(async () => {
  const p = window.__d2rim.ctx.services.get('character.player');
  const phys = window.__d2rim.ctx.services.get('physics.world');
  const T = window.__d2rim.three;
  const cc = p.controller;
  const out = [];
  // Long traverse across the moor; assert the capsule tracks the heightfield.
  const spot = phys.findClearSpot(0, 0, 0.4, 1.8, 6) ?? new T.Vector3(0, 0, 0);
  cc.setPosition(new T.Vector3(spot.x, spot.y + 0.5, spot.z));
  for (let i = 0; i < 600; i++) {
    const r = cc.move(new T.Vector3(1.8, 0, 0.6), 1 / 60);
    if (i % 60 === 0) {
      const ground = phys.groundHeight(r.position.x, r.position.z);
      out.push({
        i,
        y: +r.position.y.toFixed(3),
        ground: ground === null ? null : +ground.toFixed(3),
        gap: ground === null ? null : +(r.position.y - ground).toFixed(3),
        grounded: r.grounded,
        slopeDeg: +((Math.acos(Math.min(1, r.groundNormal.y)) * 180) / Math.PI).toFixed(1),
      });
    }
  }
  return out;
});
mark('E done');

/* ---------------- F. combat: scripted kill ------------------------------- */
report.F_combat = await (async () => {
  // Put the player back on flat-ish ground and stage a duel.
  await page.evaluate(() => {
    const d2 = window.__d2rim;
    const svc = d2.ctx.services;
    const p = svc.get('character.player');
    const phys = svc.get('physics.world');
    const dir = svc.get('ai.director');
    const T = d2.three;
    const spot = phys.findClearSpot(2, 2, 0.5, 1.9, 8) ?? new T.Vector3(2, 0, 2);
    p.teleport(spot.x, spot.y + 0.1, spot.z);
    // Move every skeleton far away, then bring one back in front of the hero.
    dir.enemies.forEach((e, i) => e.teleport(spot.x + 120 + i * 4, spot.y, spot.z + 120));
  });
  await step(30);
  const fwd = await page.evaluate(() => {
    const d2 = window.__d2rim;
    const p = d2.ctx.services.get('character.player');
    const f = p.forward(new d2.three.Vector3());
    return [f.x, f.y, f.z];
  });
  const staged = await page.evaluate(
    ([fx, fz]) => {
      const d2 = window.__d2rim;
      const svc = d2.ctx.services;
      const p = svc.get('character.player');
      const phys = svc.get('physics.world');
      const dir = svc.get('ai.director');
      const e = dir.enemies[0];
      const tx = p.position.x + fx * 1.5;
      const tz = p.position.z + fz * 1.5;
      const h = phys.groundHeight(tx, tz) ?? p.position.y;
      e.skipSpawn?.();
      e.teleport(tx, h, tz);
      return { id: e.id, hp: e.health, state: e.state, pos: [tx, h, tz], playerPos: p.position.toArray() };
    },
    [fwd[0], fwd[2]],
  );
  await step(10);
  const hp0 = (await page.evaluate(() => window.__v.enemies()))[0];
  await shot('combat-faceoff');

  // One light press, then watch the window and the hp.
  await page.evaluate(() => window.__v.press('light'));
  const timeline = await page.evaluate(async () => {
    const out = [];
    for (let i = 0; i < 60; i++) {
      await window.__d2rim.engine.stepFrames(1);
      const c = window.__v.combat();
      const e = window.__v.enemies()[0];
      out.push({ f: i, move: c.move, win: c.window, hp: e.hp, ts: +window.__v.timeScale().toFixed(3) });
    }
    return out;
  });
  await shot('combat-swing');
  const firstDamage = timeline.find((t) => t.hp < hp0.hp);
  const windowFrames = timeline.filter((t) => t.win).length;
  const hitStopFrames = timeline.filter((t) => t.ts < 0.95).length;

  // Now mash until dead or 900 frames.
  const kill = await page.evaluate(async () => {
    const trace = [];
    for (let i = 0; i < 420; i++) {
      if (i % 10 === 0) window.__v.press('light');
      await window.__d2rim.engine.stepFrames(1);
      const e = window.__v.enemies()[0];
      if (i % 5 === 0) trace.push({ f: i, hp: e?.hp ?? null, state: e?.state ?? 'gone' });
      if (!e || !e.alive) return { deathFrame: i, trace };
    }
    return { deathFrame: null, trace };
  });
  const deathFrame = kill.deathFrame;
  const swings = kill.trace.filter((t, i, a) => i === 0 || t.hp !== a[i - 1].hp);
  const after = await page.evaluate(() => window.__v.enemies());
  await shot('combat-death');
  return {
    stagedAt: staged.pos.map((v) => +v.toFixed(2)),
    enemyStartHp: hp0.hp,
    enemyStartState: hp0.state,
    firstDamageFrame: firstDamage ? firstDamage.f : null,
    hpAfterFirstSwing: firstDamage ? firstDamage.hp : null,
    hitWindowOpenFrames: windowFrames,
    hitStopFrames,
    minTimeScale: Math.min(...timeline.map((t) => t.ts)),
    movesSeen: [...new Set(timeline.map((t) => t.move).filter(Boolean))],
    deathFrame,
    finalEnemies: after.map((e) => ({ state: e.state, hp: e.hp, alive: e.alive })),
    swings,
  };
})();
mark('F done');

/* ---------------- G. enemy AI: perceive, chase, attack, hurt ------------- */
report.G_ai = await (async () => {
  const setup = await page.evaluate(() => {
    const d2 = window.__d2rim;
    const svc = d2.ctx.services;
    const p = svc.get('character.player');
    const phys = svc.get('physics.world');
    const dir = svc.get('ai.director');
    const T = d2.three;
    const live = dir.enemies.filter((e) => e.alive);
    // One live skeleton, 9 m in front of the hero, unalerted.
    const f = p.forward(new T.Vector3());
    const tx = p.position.x + f.x * 9;
    const tz = p.position.z + f.z * 9;
    const h = phys.groundHeight(tx, tz) ?? p.position.y;
    live.forEach((e, i) => {
      if (i === 0) {
        e.skipSpawn?.();
        e.teleport(tx, h, tz);
      } else e.teleport(p.position.x + 200, h, p.position.z + 200);
    });
    return { count: live.length, at: [+tx.toFixed(2), +tz.toFixed(2)], hero: p.position.toArray() };
  });
  if (setup.count === 0) return { skipped: 'no live enemies left after the kill test' };
  await step(10);
  const track = await page.evaluate(async () => {
    const out = [];
    for (let i = 0; i < 480; i++) {
      await window.__d2rim.engine.stepFrames(1);
      if (i % 5 === 0) {
        const e = window.__v.enemies().filter((x) => x.alive)[0] ?? null;
        const c = window.__v.combat();
        out.push({ f: i, state: e?.state ?? 'gone', dist: e?.dist ?? null, playerHp: c.hp });
      }
    }
    return out;
  });
  const hpStart = track[0].playerHp;
  await shot('ai-chase');
  const states = [...new Set(track.map((t) => t.state))];
  const dists = track.map((t) => t.dist).filter((d) => d !== null);
  const hpEnd = track[track.length - 1].playerHp;
  return {
    setup,
    statesSeen: states,
    startDistance: dists[0],
    minDistance: dists.length ? +Math.min(...dists).toFixed(2) : null,
    playerHp: { start: hpStart, end: hpEnd, lost: +(hpStart - hpEnd).toFixed(2) },
    firstChaseFrame: track.find((t) => t.state === 'chase')?.f ?? null,
    firstTelegraphFrame: track.find((t) => t.state === 'telegraph')?.f ?? null,
    firstAttackFrame: track.find((t) => t.state === 'attack')?.f ?? null,
    trace: track.filter((_, i) => i % 6 === 0),
  };
})();
mark('G done');

report.pageErrors = logs.slice(0, 20);
writeFileSync(`${OUT}/report.json`, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
await browser.close();
server.kill();
process.exit(0);
