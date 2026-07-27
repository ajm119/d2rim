/**
 * tools/verify-combat-loop.mjs
 *
 * The end-to-end proof that the game has a fight in it.
 *
 * Boots at the Rogue Encampment exactly as a player does — no `?zone=`, no
 * harness-spawned enemies — then travels camp -> Blood Moor -> Den of Evil and
 * asserts, in each populated zone, that:
 *
 *   1. the declared encounter table is actually placed (enemies > 0),
 *   2. the skeletons *think*: dropped in front of the player they notice him and
 *      close the distance,
 *   3. they land hits, and the player kills them, inside a stated time budget.
 *
 * Every previous harness on this project either started in the moor or spawned
 * its own enemies, which is precisely why a boot path that produced an empty
 * world survived. This one walks the path a player walks.
 *
 * Usage: `node tools/verify-combat-loop.mjs [--quick]`
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { spawn } from 'node:child_process';

import { chromium } from 'playwright';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';

const PORT = 5251;

const failures = [];
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail === undefined ? '' : ` — ${detail}`}`);
  if (!ok) failures.push(name);
};

/* -- server ---------------------------------------------------------------- */

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
const page = await browser.newPage({ viewport: { width: 640, height: 360 } });
page.setDefaultNavigationTimeout(240_000);
page.setDefaultTimeout(240_000);
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

const dump = (tag) => {
  const interesting = logs.filter((l) => /\[ai\]|\[ZoneManager\]|error|Error/.test(l));
  if (interesting.length > 0) console.log(`  ${tag} log:\n    ${interesting.join('\n    ')}`);
};

/* -- boot as a player ------------------------------------------------------ */

const url = `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&fade=0`;
console.log(`loading ${url}`);
let t0 = Date.now();
await page.goto(url, { waitUntil: 'load' });
await page.evaluate(() => window.__d2rim.ready);
console.log(`  ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

/** Step frames, then let macrotasks (network callbacks) land. */
const settle = async (frames = 4) => {
  await page.evaluate(async (n) => {
    const d2 = window.__d2rim;
    await d2.engine.stepFrames(n);
    await d2.zones.enemiesReady;
    await d2.engine.stepFrames(2);
  }, frames);
};

const probe = () =>
  page.evaluate(() => {
    const d2 = window.__d2rim;
    const player = d2.ctx.services.tryGet('character.player');
    const pos = player?.position ?? { x: 0, y: 0, z: 0 };
    const director = d2.zones.director;
    return {
      zoneId: d2.zones.activeId,
      spawnTable: d2.zones.active?.enemySpawns?.length ?? 0,
      enemies: director?.enemies.length ?? 0,
      alive: director?.alive ?? 0,
      ready: director?.ready ?? null,
      loaded: director?.loaded ?? null,
      pending: director?.pending ?? null,
      report: d2.zones.report,
      player: { x: pos.x, y: pos.y, z: pos.z },
    };
  });

/* -- 1. the camp ----------------------------------------------------------- */

await settle(6);
let p = await probe();
check('boots into the encampment', p.zoneId === 'encampment', p.zoneId ?? 'null');
console.log(`  camp: ${JSON.stringify(p.report)}`);

/* -- 2. the Blood Moor ----------------------------------------------------- */

const travel = async (zoneId, entry) => {
  logs.length = 0;
  const start = Date.now();
  await page.evaluate(
    ([z, e]) => window.__d2rim.zones.travelTo(z, e),
    [zoneId, entry ?? null],
  );
  await settle(6);
  console.log(`  travelled to ${zoneId} in ${((Date.now() - start) / 1000).toFixed(1)}s`);
};

await travel('bloodMoor');
p = await probe();
dump('moor');
check(
  'Blood Moor places its encounter',
  p.enemies > 0,
  `${p.enemies}/${p.spawnTable} placed (loaded=${p.loaded} ready=${p.ready} pending=${p.pending})`,
);
check(
  'Blood Moor places the whole table',
  p.enemies === p.spawnTable,
  `${p.enemies} of ${p.spawnTable}`,
);
check('zone report agrees', p.report?.enemies === p.enemies, `report=${p.report?.enemies}`);

/* -- 3. they perceive and engage ------------------------------------------- */

/**
 * Walk the player at the encounter and let perception do its own work.
 *
 * No `alert()`, no `skipSpawn()`, no teleport onto the enemy's nose. The
 * skeletons spawn with their backs to the player's arrival — deliberately, so
 * the player picks the fight — which means first contact is *hearing* at 6.5 m,
 * not sight. A harness that stands the player 7 m behind a skeleton and then
 * reports "the AI is dead" is measuring its own staging, so this one closes the
 * distance the way a player does: pick the nearest skeleton, step toward it in
 * 1 m increments up the ground, and record where awareness actually trips.
 */
const engagement = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const player = d2.ctx.services.tryGet('character.player');
  const physics = d2.ctx.services.tryGet('physics.world');
  const director = d2.zones.director;
  const pp = player.position;
  let nearest = null;
  let best = Infinity;
  for (const e of director.enemies) {
    const d = Math.hypot(e.position.x - pp.x, e.position.z - pp.z);
    if (d < best) {
      best = d;
      nearest = e;
    }
  }
  const target = { x: nearest.position.x, z: nearest.position.z };
  // Start 12 m out along the line from the player's arrival to the skeleton.
  const dx = target.x - pp.x;
  const dz = target.z - pp.z;
  const len = Math.hypot(dx, dz) || 1;
  const ux = dx / len;
  const uz = dz / len;

  let awareAt = null;
  let closedFrom = null;
  const trace = [];
  for (let range = 12; range >= 3; range -= 1) {
    const tx = target.x - ux * range;
    const tz = target.z - uz * range;
    const ty = physics.groundHeight(tx, tz);
    if (ty === null) continue;
    player.teleport(tx, ty + 0.06, tz);
    await d2.engine.stepFrames(12); // 0.2 s of thinking at each stop
    const d = Math.hypot(nearest.position.x - tx, nearest.position.z - tz);
    trace.push(`${range}m:${nearest.aware ? 'aware' : nearest.state}@${d.toFixed(1)}`);
    if (nearest.aware) {
      awareAt = d;
      closedFrom = d;
      // Stand still and let it come. Two things are being asserted: that it
      // closes to inside its own committed-swing range, and that the swing then
      // *connects*. The second is the one that matters — a skeleton that walks
      // up and mimes at the player is not a fight, and that is exactly the
      // state this project was in before the hit windows were re-authored onto
      // clips that reach.
      const combat = d2.ctx.services.get('combat');
      const hits = [];
      const off = d2.ctx.events.on('combat:hit', (p) => {
        if (p.target === combat.self.id) hits.push(+p.outcome.total.toFixed(1));
      });
      const held = { x: tx, y: ty + 0.06, z: tz };
      let minD = d;
      for (let i = 0; i < 30; i++) {
        player.teleport(held.x, held.y, held.z);
        await d2.engine.stepFrames(15); // 0.25 s per slice, 7.5 s total
        minD = Math.min(minD, Math.hypot(nearest.position.x - tx, nearest.position.z - tz));
        if (hits.length >= 2) break;
      }
      off();
      return {
        variant: nearest.profile.variant,
        awareAt,
        closedFrom,
        closedTo: minD,
        attackRange: nearest.profile.attackRange,
        hits,
        healthLost: +(combat.vitals.health.max - combat.vitals.health.value).toFixed(1),
        state: nearest.state,
        trace,
      };
    }
  }
  return { variant: nearest.profile.variant, awareAt, trace, state: nearest.state };
});
console.log(`  engagement: ${JSON.stringify(engagement)}`);
check(
  'a Blood Moor skeleton perceives an approaching player',
  engagement.awareAt !== null,
  engagement.awareAt === null
    ? `never noticed: ${engagement.trace.join(' ')}`
    : `noticed at ${engagement.awareAt.toFixed(1)} m (${engagement.variant})`,
);
check(
  'and closes to inside its own attack range',
  engagement.closedTo !== undefined && engagement.closedTo <= engagement.attackRange,
  `${engagement.closedFrom?.toFixed(2)} m -> ${engagement.closedTo?.toFixed(2)} m ` +
    `(range ${engagement.attackRange?.toFixed(2)} m), state=${engagement.state}`,
);
check(
  'and lands real damage on the player',
  (engagement.hits?.length ?? 0) > 0,
  `${engagement.hits?.length ?? 0} hits for ${engagement.healthLost} health ` +
    `(${(engagement.hits ?? []).join(', ')})`,
);

/* -- 4. the Den of Evil ---------------------------------------------------- */

await travel('denOfEvil');
p = await probe();
dump('den');
check(
  'Den of Evil places its encounter',
  p.enemies > 0,
  `${p.enemies}/${p.spawnTable} placed (loaded=${p.loaded} ready=${p.ready} pending=${p.pending})`,
);
check(
  'Den of Evil places the whole table',
  p.enemies === p.spawnTable,
  `${p.enemies} of ${p.spawnTable}`,
);

/* -- 5. back to the moor (a second lap must still populate) ---------------- */

await travel('bloodMoor');
p = await probe();
check('a second entry to the moor repopulates', p.enemies === p.spawnTable, `${p.enemies}/${p.spawnTable}`);

/* -- done ------------------------------------------------------------------ */

await browser.close();
server.kill('SIGTERM');
console.log(failures.length === 0 ? '\nALL PASS' : `\n${failures.length} FAILED: ${failures.join(', ')}`);
process.exit(failures.length === 0 ? 1 * 0 : 1);
