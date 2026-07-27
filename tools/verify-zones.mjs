/**
 * tools/verify-zones.mjs
 *
 * Drives the three Act I zones in a real browser and asserts the things that
 * only break in a real browser.
 *
 * Three classes of check, and each exists because of a specific way a zone
 * system fails silently:
 *
 * 1. **Standing.** Load each zone, step frames, and assert the player is
 *    grounded, is within a few centimetres of the surface the physics world
 *    reports under him, and has not fallen. A zone whose colliders were built
 *    but never `syncQueries()`-ed looks perfect and drops the player through the
 *    floor, and nothing in a unit test can see that.
 * 2. **Leaking.** Travel encampment -> moor -> den -> moor -> encampment and
 *    compare collider counts, scene-graph size and the renderer's own geometry
 *    and texture counters against the baseline. A transition that leaks does not
 *    fail; it degrades, over minutes, until the session is unusable.
 * 3. **Looking.** Capture each zone and run the blank-frame guard over it, so
 *    "the zone loaded" cannot be satisfied by an empty grey plane.
 *
 * Usage: `node tools/verify-zones.mjs [outDir] [--quick] [--no-shots] [--shots-only]
 *         [--quality=low|high]`
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { mkdirSync } from 'node:fs';
import { spawn } from 'node:child_process';

import { chromium } from 'playwright';
import sharp from 'sharp';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';
import { guardImageFile } from './capture/frame-guard.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const QUICK = process.argv.includes('--quick');
const SHOTS = !process.argv.includes('--no-shots');
/** Capture and check the three zones, then stop — skips travel and leak checks. */
const SHOTS_ONLY = process.argv.includes('--shots-only');
const OUT = args[0] ?? '/tmp/zones';
const PORT = 5247;
const WIDTH = 1280;
const HEIGHT = 720;
const WARMUP = QUICK ? 8 : 24;
/**
 * `low` by default. This container rasterises in software on four cores, so the
 * full-quality tier costs minutes per frame and measures nothing useful; every
 * assertion here is about geometry, physics and bookkeeping, none of which the
 * quality tier changes. Pass `--quality=high` when the captures are the point.
 */
const QUALITY = (process.argv.find((a) => a.startsWith('--quality=')) ?? '--quality=low').split('=')[1];

mkdirSync(OUT, { recursive: true });

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
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });
// Playwright's 30 s default is a desktop-browser assumption. Booting a zone here
// means compiling every node material on a software rasteriser with four cores
// already saturated, and a navigation timeout looks exactly like a hang.
page.setDefaultNavigationTimeout(180_000);
page.setDefaultTimeout(180_000);
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

/* -- helpers --------------------------------------------------------------- */

const goto = async (query) => {
  logs.length = 0;
  const url = `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=${QUALITY}&fade=0&${query}`;
  process.stdout.write(`  ...loading ${url}\n`);
  const startedAt = Date.now();
  await page.goto(url, { waitUntil: 'load' });
  await page.evaluate(() => window.__d2rim.ready);
  process.stdout.write(`  ...ready in ${((Date.now() - startedAt) / 1000).toFixed(1)}s\n`);
};

/**
 * Wait for the active zone's enemies to finish loading.
 *
 * `stepFrames` is a tight async loop that only ever yields *microtasks* — on
 * WebGL2 `renderer.render` returns void, so awaiting it does not give the event
 * loop a turn. Network callbacks are macrotasks, so an `EnemyDirector` that
 * started loading its skeleton GLBs on frame 1 cannot finish until the stepping
 * stops. Asserting on the enemy count straight after `stepFrames` therefore
 * measures the loader, not the zone.
 */
const settleEnemies = async () => {
  const needed = await page.evaluate(() => window.__d2rim.zones.active?.enemySpawns?.length ?? 0);
  if (needed === 0) return;
  await page
    .waitForFunction(() => window.__d2rim.zones.director?.ready === true, null, { timeout: 60_000 })
    .catch(() => process.stdout.write('  ...enemies did not settle within 60s\n'));
  await step(2);
};

/** Step frames and say how long it took; on SwiftShader this is the slow part. */
const step = async (n) => {
  const startedAt = Date.now();
  await page.evaluate((count) => window.__d2rim.engine.stepFrames(count), n);
  process.stdout.write(`  ...${n} frames in ${((Date.now() - startedAt) / 1000).toFixed(1)}s\n`);
};

/** Everything the assertions need, read out of the page in one round trip. */
const probe = () =>
  page.evaluate(() => {
    const d2 = window.__d2rim;
    const ctx = d2.ctx;
    const physics = ctx.services.tryGet('physics.world');
    const player = ctx.services.tryGet('character.player');
    const zones = d2.zones;
    const info = ctx.renderer.three.info ?? { memory: {} };
    const pos = player?.position ?? { x: 0, y: 0, z: 0 };
    // Two answers to "where is the floor", because they fail differently.
    //
    // `surface` is the zone's own height function — the authority the visual
    // mesh and the physics heightfield are both built from, and therefore the
    // right thing to test "has the player fallen through the world" against.
    // `ground` is the physics *query*, which is what gameplay code actually
    // calls; it is reported alongside, with a probe 1 m off, because a null
    // there over solid ground is a real defect even when the player is standing
    // correctly (see the entry-point comment in `scene/RogueEncampment`).
    const ground = physics?.ready ? physics.groundHeight(pos.x, pos.z) : null;
    const groundOffset = physics?.ready ? physics.groundHeight(pos.x + 1, pos.z + 1) : null;
    const field = zones.active?.field ?? null;
    const surface = field?.heightAt !== undefined ? field.heightAt(pos.x, pos.z) : null;
    let zoneMeshes = 0;
    let zoneTris = 0;
    const root = zones.active?.root ?? null;
    root?.traverse((object) => {
      if (!object.isMesh) return;
      zoneMeshes++;
      const index = object.geometry.getIndex();
      const position = object.geometry.getAttribute('position');
      const verts = index !== null ? index.count : (position?.count ?? 0);
      zoneTris += (verts / 3) * (object.isInstancedMesh ? object.count : 1);
    });
    const colliderKinds = {};
    for (const record of physics?.colliders ?? []) {
      colliderKinds[record.kind] = (colliderKinds[record.kind] ?? 0) + 1;
    }
    return {
      zoneId: zones.activeId,
      report: zones.report,
      registered: zones.registered,
      colliders: physics?.colliders.length ?? 0,
      colliderKinds,
      portals: (zones.active?.portals ?? []).map((p) => p.id),
      entryPoints: (zones.active?.entryPoints ?? []).map((e) => e.id),
      enemies: zones.director?.enemies.length ?? 0,
      alive: zones.director?.alive ?? 0,
      player: { x: pos.x, y: pos.y, z: pos.z, grounded: player?.grounded ?? false },
      ground,
      groundOffset,
      surface,
      sceneChildren: ctx.scene.children.length,
      sceneChildNames: ctx.scene.children.map((child) => child.name || child.type),
      // The zone-owned collider set, which is what `ZoneManager` is responsible
      // for reclaiming. `character` is excluded deliberately and the exclusion
      // is itself a finding — see the leak assertions below.
      zoneOwnedColliders:
        (colliderKinds.terrain ?? 0) + (colliderKinds.prop ?? 0) + (colliderKinds.trigger ?? 0),
      characterColliders: colliderKinds.character ?? 0,
      zoneMeshes,
      zoneTris: Math.round(zoneTris),
      geometries: info.memory?.geometries ?? -1,
      textures: info.memory?.textures ?? -1,
      lights: ctx.services.tryGet('render.lighting')?.stats ?? null,
    };
  });

const shoot = async (label, pose) => {
  if (!SHOTS) return;
  if (pose !== undefined) await page.evaluate(pose);
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
  const file = `${OUT}/${label}.png`;
  await sharp(Buffer.from(frame.data, 'base64'), {
    raw: { width: frame.width, height: frame.height, channels: 4 },
  })
    .png()
    .toFile(file);
  const guard = await guardImageFile(file);
  check(`${label}: frame is not blank`, guard.ok, guard.ok ? file : guard.failures.join('; '));
  return file;
};

/* -- 1. each zone loads, is solid, and holds the player --------------------- */

/**
 * Camera poses per zone. Chosen to answer the question the capture exists to
 * answer — "does this look like a camp / a cave" — rather than to be pretty.
 */
const POSES = {
  encampment: () => {
    const { camera } = window.__d2rim.ctx;
    camera.position.set(21, 13.5, 27);
    camera.lookAt(0, 2.2, 0);
    camera.fov = 48;
    camera.updateProjectionMatrix();
  },
  bloodMoor: () => {
    const d2 = window.__d2rim;
    const { camera } = d2.ctx;
    const field = d2.scene.field;
    // 26 m back and 8 m up, so the cave mouth reads as a feature cut into the
    // west bank rather than as the inside of a boulder.
    camera.position.set(-2, field.heightAt(-2, 14) + 8, 14);
    camera.lookAt(-21, field.heightAt(-21, -4) + 2, -4);
    camera.fov = 52;
    camera.updateProjectionMatrix();
  },
  denOfEvil: () => {
    const d2 = window.__d2rim;
    const { camera } = d2.ctx;
    const zone = d2.zones.active;
    const player = d2.ctx.services.get('character.player');
    const p = player.position;
    // Over the player's shoulder, angled *down*, and close.
    //
    // Close matters: the generator guarantees a 4 m minimum passage width, so an
    // arm 6.5 m behind the player is very likely inside rock — which is exactly
    // what the previous pose did, and the capture came back as two wall faces
    // with a black slot between them. The real `CameraRig` sphere-casts and
    // would have pulled in; a hand-placed capture camera has to do it itself.
    // 2.6 m back and 1.9 m up stays inside the guaranteed corridor.
    camera.position.set(p.x, p.y + 1.9, p.z + 2.6);
    camera.lookAt(p.x, p.y + 0.5, p.z - 8);
    camera.fov = 68;
    camera.updateProjectionMatrix();
    void zone;
  },
};

const baselines = {};

for (const zone of ['encampment', 'bloodMoor', 'denOfEvil']) {
  console.log(`\n=== ${zone} ===`);
  await goto(`zone=${zone}`);
  await step(WARMUP);
  await settleEnemies();
  const state = await probe();
  baselines[zone] = state;

  check(`${zone}: is the active zone`, state.zoneId === zone, state.zoneId);
  check(`${zone}: all three zones registered`, state.registered.length === 3, state.registered.join(','));
  check(`${zone}: built colliders`, state.colliders > 20, `${state.colliders} colliders`);
  check(
    `${zone}: has a trigger volume per portal`,
    (state.colliderKinds.trigger ?? 0) >= state.portals.length && state.portals.length > 0,
    `${state.colliderKinds.trigger ?? 0} triggers for ${state.portals.length} portals (${state.portals.join(', ')})`,
  );
  check(
    `${zone}: declares entry points`,
    state.entryPoints.length > 0,
    state.entryPoints.join(', '),
  );
  check(
    `${zone}: player is grounded`,
    state.player.grounded === true,
    `y=${state.player.y.toFixed(2)} grounded=${state.player.grounded}`,
  );
  check(
    `${zone}: player is not falling through the world`,
    state.surface !== null && Math.abs(state.player.y - state.surface) < 0.6,
    `player y=${state.player.y.toFixed(2)} surface=${state.surface === null ? 'none' : state.surface.toFixed(2)}`,
  );
  check(
    `${zone}: the physics ground query answers under the player`,
    state.ground !== null,
    `groundHeight=${state.ground === null ? 'none' : state.ground.toFixed(2)} ` +
      `(1 m off: ${state.groundOffset === null ? 'none' : state.groundOffset.toFixed(2)})`,
  );
  check(
    `${zone}: player is inside the zone, not at the origin by accident`,
    Number.isFinite(state.player.x) && state.player.y > -20,
    `(${state.player.x.toFixed(1)}, ${state.player.y.toFixed(1)}, ${state.player.z.toFixed(1)})`,
  );
  check(
    `${zone}: geometry was actually built`,
    state.zoneMeshes > 3 && state.zoneTris > 5000,
    `${state.zoneMeshes} meshes, ${state.zoneTris} triangles`,
  );
  if (zone === 'encampment') {
    check(`${zone}: is a safe zone (no enemies)`, state.enemies === 0, `${state.enemies} enemies`);
  } else {
    check(`${zone}: spawned enemies`, state.enemies > 0, `${state.enemies} enemies`);
  }

  console.log(
    `      report: ${JSON.stringify(state.report)}\n` +
      `      colliders by kind: ${JSON.stringify(state.colliderKinds)}\n` +
      `      lights: ${JSON.stringify(state.lights)}\n` +
      `      renderer: ${state.geometries} geometries, ${state.textures} textures`,
  );

  await shoot(zone, POSES[zone]);
  const errors = logs.filter((l) => l.startsWith('pageerror') || l.startsWith('error:'));
  check(`${zone}: no page errors`, errors.length === 0, errors.slice(0, 3).join(' | '));
}

if (SHOTS_ONLY) {
  await browser.close();
  server.kill();
  console.log(`\n${failures.length === 0 ? 'ALL PASS' : `${failures.length} FAILED`} (shots only)`);
  for (const failure of failures) console.log(`  - ${failure}`);
  process.exit(failures.length === 0 ? 0 : 1);
}

/* -- 2. the den's generated layout is sane in the running game -------------- */

console.log('\n=== den layout ===');
await goto('zone=denOfEvil');
await step(6);
const den = await page.evaluate(() => {
  const zone = window.__d2rim.zones.active;
  const layout = zone.layout;
  const physics = window.__d2rim.ctx.services.get('physics.world');
  // Sample the generated spawn points against the live physics world: every one
  // of them must be somewhere a body can actually stand.
  let standable = 0;
  for (const spawn of layout.spawnPoints) {
    if (physics.findClearSpot(spawn.x, spawn.z, 0.4, 1.8, 3) !== null) standable++;
  }
  return {
    seed: layout.seed,
    attempt: layout.attempt,
    floorCells: layout.floorCells,
    chambers: layout.chambers.length,
    spawns: layout.spawnPoints.length,
    standable,
    maxDepth: layout.maxDepth,
    objective: zone.objective,
    deepest: layout.deepestChamber.depthRatio,
  };
});
check('den: generated a cave', den.floorCells > 500, `${den.floorCells} walkable cells`);
check('den: found chambers', den.chambers >= 2, `${den.chambers} chambers`);
check(
  'den: every spawn point is standable in the live physics world',
  den.standable === den.spawns,
  `${den.standable}/${den.spawns}`,
);
check('den: objective is deep', den.deepest >= 0.7, `depthRatio ${den.deepest.toFixed(2)}`);
console.log(`      ${JSON.stringify(den)}`);

/* -- 3. transitions do not leak --------------------------------------------- */

console.log('\n=== transitions ===');
await goto('zone=encampment');
await step(10);
const before = await probe();

const travel = async (zoneId, entry) => {
  await page.evaluate(
    async ([z, e]) => {
      await window.__d2rim.zones.travelTo(z, e);
    },
    [zoneId, entry ?? null],
  );
  await step(QUICK ? 3 : 8);
  await settleEnemies();
  return probe();
};

const toMoor = await travel('bloodMoor', 'from-camp');
check('travel: arrived on the moor', toMoor.zoneId === 'bloodMoor', toMoor.zoneId);
check(
  'travel: player is grounded after arriving on the moor',
  toMoor.player.grounded && toMoor.surface !== null && Math.abs(toMoor.player.y - toMoor.surface) < 0.6,
  `y=${toMoor.player.y.toFixed(2)} surface=${toMoor.surface?.toFixed(2)}`,
);
check(
  'travel: arrived away from the portal it came through',
  toMoor.portals.length === 2,
  toMoor.portals.join(', '),
);

const toDen = await travel('denOfEvil', 'cave-mouth');
check('travel: arrived in the den', toDen.zoneId === 'denOfEvil', toDen.zoneId);
check(
  'travel: player is grounded after arriving in the den',
  toDen.player.grounded && toDen.surface !== null && Math.abs(toDen.player.y - toDen.surface) < 0.6,
  `y=${toDen.player.y.toFixed(2)} surface=${toDen.surface?.toFixed(2)}`,
);
check('travel: den populated itself', toDen.enemies > 0, `${toDen.enemies} enemies`);

const backToMoor = await travel('bloodMoor', 'from-den');
check('travel: back on the moor', backToMoor.zoneId === 'bloodMoor', backToMoor.zoneId);

const home = await travel('encampment', 'gate');
check('travel: back in the camp', home.zoneId === 'encampment', home.zoneId);
check(
  'travel: player is grounded back in the camp',
  home.player.grounded && home.surface !== null && Math.abs(home.player.y - home.surface) < 0.6,
  `y=${home.player.y.toFixed(2)} surface=${home.surface?.toFixed(2)}`,
);

/* -- the leak assertions ----------------------------------------------------
 *
 * Scoped to what `ZoneManager` actually owns, because two of the four things
 * that grow across a round trip are not leaks and one is somebody else's:
 *
 * - **zone colliders** (terrain, prop, trigger) are the manager's, tracked by
 *   the before/after snapshot around `buildColliders`. These must return to
 *   baseline exactly. No tolerance.
 * - **character colliders** grow by one per despawned enemy, and the cause is
 *   `physics/CharacterController.dispose`: it calls `world.removeRigidBody`,
 *   which drops the collider inside Rapier, but never
 *   `PhysicsWorld.removeCollider`, so the `#records` map keeps a record pointing
 *   at a freed collider. Reported rather than asserted — the fix belongs in
 *   `src/physics`, and `PhysicsWorld` exposes no way to drop a record without
 *   also asking Rapier to remove an already-removed collider.
 * - **renderer geometries and textures** grow on *first* visit to a zone and
 *   should not grow again, because `AssetManager` pins shared models. So the
 *   test is a second round trip with a zero delta, not an absolute bound on the
 *   first — an absolute bound cannot tell cache warm-up from a leak.
 */
check(
  'leak: zone-owned colliders return to baseline exactly',
  home.zoneOwnedColliders === before.zoneOwnedColliders,
  `${before.zoneOwnedColliders} -> ${home.zoneOwnedColliders}`,
);
console.log(
  `      NOTE character-collider records ${before.characterColliders} -> ` +
    `${home.characterColliders} (physics/CharacterController.dispose does not ` +
    'unregister its record; see the comment above)',
);
const newChildren = home.sceneChildNames.filter(
  (name, i) => before.sceneChildNames.indexOf(name) === -1 || i >= before.sceneChildNames.length,
);
check(
  'leak: scene graph returns to baseline',
  home.sceneChildren <= before.sceneChildren,
  `${before.sceneChildren} -> ${home.sceneChildren} children; new: ${newChildren.join(', ') || 'none'}`,
);

/* A second lap. Whatever the first lap warmed, the second must not grow. */
await travel('bloodMoor', 'from-camp');
await travel('denOfEvil', 'cave-mouth');
await travel('bloodMoor', 'from-den');
const lap2 = await travel('encampment', 'gate');
check(
  'leak: renderer geometry count is flat on a second round trip',
  lap2.geometries <= home.geometries,
  `lap1 ${home.geometries} -> lap2 ${lap2.geometries} (baseline ${before.geometries})`,
);
check(
  'leak: renderer texture count is flat on a second round trip',
  lap2.textures <= home.textures,
  `lap1 ${home.textures} -> lap2 ${lap2.textures} (baseline ${before.textures})`,
);
check(
  'leak: zone-owned colliders are flat on a second round trip',
  lap2.zoneOwnedColliders === home.zoneOwnedColliders,
  `${home.zoneOwnedColliders} -> ${lap2.zoneOwnedColliders}`,
);
check(
  'leak: enemies from other zones do not survive',
  home.enemies === 0 && lap2.enemies === 0,
  `${home.enemies} then ${lap2.enemies} enemies in the camp`,
);
check(
  'leak: no stale trigger volumes',
  (lap2.colliderKinds.trigger ?? 0) === lap2.portals.length,
  `${lap2.colliderKinds.trigger ?? 0} triggers, ${lap2.portals.length} portals`,
);
console.log(
  `      baseline ${JSON.stringify({
    colliders: before.colliders,
    children: before.sceneChildren,
    geometries: before.geometries,
    textures: before.textures,
  })}\n      after    ${JSON.stringify({
    colliders: home.colliders,
    children: home.sceneChildren,
    geometries: home.geometries,
    textures: home.textures,
  })}`,
);

/* -- 4. the portal prompt --------------------------------------------------- */

console.log('\n=== portals ===');
const prompt = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const zone = d2.zones.active;
  const player = d2.ctx.services.get('character.player');
  const portal = zone.portals[0];
  player.teleport(portal.position.x, portal.position.y + 0.1, portal.position.z);
  await d2.engine.stepFrames(3);
  const system = d2.engine.getModule('world.portals');
  return { text: system.promptText, id: system.current?.id ?? null, portal: portal.id };
});
check(
  'portal: standing in the volume raises a prompt',
  prompt.id === prompt.portal && typeof prompt.text === 'string',
  `${prompt.id}: ${JSON.stringify(prompt.text)}`,
);
check(
  'portal: the prompt names the interact key and the destination',
  typeof prompt.text === 'string' && prompt.text.startsWith('Press E to') && prompt.text.length > 14,
  prompt.text,
);

const cleared = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const player = d2.ctx.services.get('character.player');
  player.teleport(0, 1, 0);
  await d2.engine.stepFrames(3);
  return d2.engine.getModule('world.portals').promptText;
});
check('portal: the prompt clears when the player leaves', cleared === null, String(cleared));

/* -- done ------------------------------------------------------------------- */

await browser.close();
server.kill();

console.log(`\n${failures.length === 0 ? 'ALL PASS' : `${failures.length} FAILED`}`);
for (const failure of failures) console.log(`  - ${failure}`);
process.exit(failures.length === 0 ? 0 : 1);
