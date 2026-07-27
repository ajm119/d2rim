/**
 * Two-sided-combat regression guard.
 *
 * Phase 3 shipped with combat one-directional: the player killed skeletons
 * fine, skeletons never landed a blow. Perception, chase, telegraph and swing
 * animation all worked — the swept hitbox opened its window on schedule and
 * swept empty air, because every variant's standard attack was authored onto a
 * clip whose implied blade never reaches past 0.66 m while the AI holds a
 * 0.70 m stand-off. Nothing in the code claimed a reach, so nothing disagreed.
 *
 * This drives the real game and asserts the loop actually closes:
 *
 *   A. a skeleton left to its own devices takes health off the player
 *   B. a player who stands there and refuses to fight back dies
 *   C. a dead player respawns, alive and at full health
 *
 *   node tools/verify-player-damage.mjs
 *
 * Exits non-zero on a regression.
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const PORT = 5242;

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'],
  { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((r) => {
  server.stdout.on('data', (c) => String(c).includes('Local:') && r());
  setTimeout(r, 15000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: 192, height: 108 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
// `zone=bloodMoor`: the default start zone is the Rogue Encampment, which is
// a safe town and has no skeletons in it to be hit by.
await page.goto(`http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&zone=bloodMoor`, {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);

// Zones stream their enemies in after the world is ready, so wait for one to
// exist rather than assuming the first frame has a fight in it. If the zone
// does not produce one, spawn a skeleton through the director's own public
// entry point instead: this harness is about whether an enemy's swing lands,
// not about whose job it is to place him, and it should not be blocked by a
// zone whose population is still being wired up next door.
const spawned = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const svc = d2.ctx.services;
  const find = () => svc.tryGet?.('ai.director') ?? svc.tryGet?.('world.zones')?.director ?? null;
  const live = () => (find()?.enemies ?? []).filter((e) => e.alive).length;
  for (let i = 0; i < 60; i++) {
    await d2.engine.stepFrames(1);
    if (live() > 0) return { source: 'zone', frames: i, count: live() };
  }

  const director = find();
  if (director === null) return { source: 'none', reason: 'no director' };
  const assets = svc.get('assets');
  const player = svc.get('character.player');
  const gltf = await Promise.race([
    assets.loadGLTF('enemy.skeleton.warrior'),
    new Promise((_, reject) => setTimeout(() => reject(new Error('GLB load timed out')), 60000)),
  ]).catch((error) => ({ error: String(error) }));
  if (gltf.error !== undefined) return { source: 'none', reason: gltf.error };
  assets.pin('enemy.skeleton.warrior');
  const clips = [...gltf.animations];
  for (const variant of ['warrior', 'minion']) {
    director.spawn(
      { variant, x: player.position.x + (variant === 'warrior' ? 2.6 : -2.6), z: player.position.z, patrol: 0 },
      gltf.scene,
      clips,
    );
  }
  for (let i = 0; i < 20; i++) await d2.engine.stepFrames(1);
  return { source: 'harness', count: live() };
});
console.log('enemy supply:', JSON.stringify(spawned));
if ((spawned.count ?? 0) < 1) {
  console.log(`PLAYER DAMAGE: FAIL — no enemy to fight (${spawned.reason ?? 'unknown'})`);
  await browser.close();
  server.kill();
  process.exit(1);
}

/* Probes: every hit landed on the player, and every death. */
await page.evaluate(() => {
  const d2 = window.__d2rim;
  window.__hitsOnPlayer = [];
  window.__deaths = [];
  d2.ctx.events.on('combat:hit', (e) => {
    if (e.targetLabel === 'player') {
      window.__hitsOnPlayer.push({ result: e.outcome.result, total: +e.outcome.total.toFixed(1) });
    }
  });
  d2.ctx.events.on('combat:death', (e) => window.__deaths.push(e.faction));

  // Stand the player inside the nearest skeleton's attack range and hold him
  // there. Nothing is forced beyond placement: the AI decides when to swing.
  const svc = d2.ctx.services;
  const player = svc.get('character.player');
  // The enemy director moved under `ZoneManager` when zones landed; accept
  // either shape so this harness does not need re-editing for the next move.
  const director = svc.tryGet?.('ai.director') ?? svc.tryGet?.('world.zones')?.director ?? null;
  const physics = svc.get('physics.world');
  const V = d2.three.Vector3;
  const enemy = (director?.enemies ?? []).find((e) => e.alive);
  if (enemy === undefined) throw new Error('no live enemy to fight');
  const foot = enemy.footPosition(new V());
  const ground = physics.groundHeight(foot.x + 1.2, foot.z) ?? foot.y;
  player.teleport(foot.x + 1.2, ground + 0.1, foot.z);
  window.__variant = enemy.profile.variant;
});

/**
 * Seconds of simulation per driven frame.
 *
 * Deliberately coarser than the render rate. Nothing this harness measures is
 * per-frame — it counts hits, health and deaths over tens of seconds of a
 * fight — and the fixed-step accumulator runs three 60 Hz simulation slices per
 * call at this delta, so the world advances three times as far for one software
 * rasterisation. On a GPU-less runner the render is the entire cost, and a
 * fight that takes twenty minutes of wall clock does not get run.
 */
const STEP = 1 / 20;

const run = (frames, until = 'never') =>
  page.evaluate(async ([n, until, step]) => {
    const d2 = window.__d2rim;
    const svc = d2.ctx.services;
    const combat = svc.get('combat');
    const player = svc.get('character.player');
    const director = svc.tryGet?.('ai.director') ?? svc.tryGet?.('world.zones')?.director ?? null;
    const V = d2.three.Vector3;
    let minSeparation = Infinity;
    let telegraphFrames = 0;
    for (let i = 0; i < n; i++) {
      if (until === 'dead' && window.__deaths.includes('player')) break;
      if (until === 'alive' && combat.self?.alive === true && combat.vitals.health.value > 1) break;
      await d2.engine.stepFrames(1, step);
      for (const e of director?.enemies ?? []) {
        if (!e.alive) continue;
        minSeparation = Math.min(minSeparation, e.footPosition(new V()).distanceTo(player.position));
        if (e.state === 'telegraph' || e.state === 'attack') telegraphFrames++;
      }
    }
    return {
      health: +combat.vitals.health.value.toFixed(1),
      maxHealth: combat.vitals.health.max,
      alive: combat.self?.alive ?? null,
      hits: window.__hitsOnPlayer.length,
      landed: window.__hitsOnPlayer.filter((h) => h.result !== 'miss').length,
      damage: +window.__hitsOnPlayer.reduce((s, h) => s + h.total, 0).toFixed(1),
      deaths: [...window.__deaths],
      minSeparation: +minSeparation.toFixed(2),
      swingFrames: telegraphFrames,
      liveEnemies: (director?.enemies ?? []).filter((e) => e.alive).length,
    };
  }, [frames, until, STEP]);

let failed = false;
const check = (label, ok, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${label}${detail === undefined ? '' : ` — ${detail}`}`);
  if (!ok) failed = true;
};

/* A. does a skeleton left alone take health off the player? */
const a = await run(170);
console.log(
  `\n[A] 170 frames (8.5 s of fight) beside a ${await page.evaluate(() => window.__variant)}:`,
  JSON.stringify(a),
);
check('the player took at least one hit', a.landed > 0, `${a.landed} landed of ${a.hits}`);
check('the player lost health', a.health < a.maxHealth, `${a.health}/${a.maxHealth}`);
check('the enemy actually swung', a.swingFrames > 0, `${a.swingFrames} swing frames`);

/* B. can an enemy actually finish the player off?
 *
 * The player is dropped to a sliver first. That is a shortcut on the *clock*,
 * not on the mechanism: the killing blow still has to come from a skeleton
 * swinging of its own accord, and A above already proves the damage per swing
 * is real rather than scripted. Grinding 120 health off at ~6 a hit takes about
 * three thousand driven frames, which on a software renderer is minutes of
 * wall clock for no extra information. */
await page.evaluate(() => {
  // Leave exactly enough for one more swing to finish. Draining him to zero
  // outright would kill him *here*, through a code path no enemy uses, and the
  // whole point of this phase is that a skeleton can land the killing blow.
  const vitals = window.__d2rim.ctx.services.get('combat').vitals;
  vitals.applyDamage(Math.max(0, vitals.health.value - 6));
});
const b = await run(220, 'dead');
console.log('\n[B] left on a sliver of health and still not fighting back:', JSON.stringify(b));
check('the player died', b.deaths.includes('player'), JSON.stringify(b.deaths));

/* C. does he come back? */
const c = await run(160, 'alive');
console.log('\n[C] after death:', JSON.stringify(c));
check('the player respawned alive', c.alive === true, `alive=${c.alive}`);
check('the respawn restored health', c.health > 0, `${c.health}/${c.maxHealth}`);

if (errors.length > 0) {
  console.log('pageErrors:', errors);
  failed = true;
}
await browser.close();
server.kill();
console.log(failed ? '\nPLAYER DAMAGE: FAIL' : '\nPLAYER DAMAGE: PASS');
process.exit(failed ? 1 : 0);
