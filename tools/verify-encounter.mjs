/**
 * tools/verify-encounter.mjs
 *
 * The two things that have to be true for this game to have a fight in it, and
 * neither of which any unit test can see.
 *
 * ### 1. There is an encounter at all
 *
 * `BLOOD_MOOR_SPAWNS` declares six skeletons. Walking into the moor from the
 * encampment has to produce six live, hostile skeletons that notice the player
 * and close on him. Booting straight into the moor has to produce the same six.
 * The two paths are genuinely different — on the boot path `ZoneManager` loads
 * the zone before `CombatSystem` has registered, so placement is deferred — and
 * the deferred one was broken.
 *
 * ### 2. The fight is worth having
 *
 * Balance measured rather than asserted from the design doc. Four scenarios,
 * all driven by the real AI and the real combo machine:
 *
 * | scenario                        | what it reports                        |
 * |---------------------------------|----------------------------------------|
 * | one minion vs a passive player  | damage per second, projected TTK       |
 * | a pair of minions vs the same   | ditto — a group must be much worse     |
 * | one warrior vs the same         | ditto, for the heavy hitter            |
 * | player vs a minion, and a warrior | swings and seconds to kill           |
 *
 * The intended numbers, the reasoning, and what is actually measured:
 *
 *  - **A lone minion kills a passive player in 20–40 s.** Long enough that
 *    dying to one skeleton means you ignored it; short enough that it is a
 *    threat rather than scenery.
 *  - **A pair does it in under 22 s, and much faster than twice as fast.**
 *    Getting surrounded has to be the thing that kills you.
 *  - **The Barbarian kills a minion in a handful of swings**, a couple of
 *    seconds. "A few hits, not twenty."
 *  - **A warrior is tougher but not a different genre**: more swings than a
 *    minion, no more than fourteen.
 *
 * | scenario                     | before      | after          |
 * |------------------------------|-------------|----------------|
 * | one minion -> passive player | 1.72 dps, 70 s | 5.44 dps, 22.1 s |
 * | two minions -> same          | 2.92 dps, 41 s | 12.41 dps, 9.7 s |
 * | one warrior -> same          | 0.25 dps, 480 s | 4.94 dps, 24.3 s |
 * | player -> minion             | 8 swings, 4.0 s | 6 swings, 2.25 s |
 * | player -> warrior            | 16 swings, 10.2 s | 9 swings, 5.42 s |
 *
 * The warrior column is the one worth looking at twice. It was not slightly
 * out of balance, it was *harmless* — one landed hit in sixteen seconds beside
 * a player who never moved — because it committed to swings from 1.85 m with an
 * axe that reaches 0.84 m. The balance was tuned against a system that dealt no
 * damage, and nothing had ever felt the numbers.
 *
 * Frame stepping is the entire cost here — this container rasterises in
 * software — so every loop runs inside the page and the simulation is advanced
 * five 60 Hz slices per rendered frame (`STEP`, and `maxSubSteps` is 5). The
 * simulation is unchanged by that; only the number of pictures drawn is.
 *
 *   node tools/verify-encounter.mjs [--quick]
 *
 * Assumes `dist/` is current; run `npm run build` first. Exits non-zero on a
 * regression.
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';

const QUICK = process.argv.includes('--quick');
const PORT = 5248;
/**
 * Seconds of simulation per rendered frame.
 *
 * `Engine` runs at most `maxSubSteps` (5) fixed 60 Hz slices per frame, so
 * 5/60 is the largest delta that is still simulated in full. Anything larger
 * silently drops simulation time and would make every number here a lie.
 */
const STEP = 5 / 60;

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
const page = await browser.newPage({ viewport: { width: 160, height: 90 } });
page.setDefaultNavigationTimeout(180_000);
page.setDefaultTimeout(300_000);
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

const started = Date.now();
const mark = (label) => console.log(`  [${((Date.now() - started) / 1000).toFixed(0)}s] ${label}`);

/* -------------------------------------------------------------------------- */
/* 1. the encounter exists                                                    */
/* -------------------------------------------------------------------------- */

console.log('\n=== the moor is populated ===');
await page.goto(`http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&fade=0&zone=encampment`, {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
mark('booted into the encampment');

const camp = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  await d2.engine.stepFrames(4);
  return { zone: d2.zones.activeId, enemies: d2.zones.director?.enemies.length ?? 0 };
});
check('camp: the hub is a safe zone', camp.zone === 'encampment' && camp.enemies === 0, JSON.stringify(camp));

// Travel exactly as walking through the portal does.
const arrival = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  await d2.zones.travelTo('bloodMoor', 'from-camp');
  await d2.zones.enemiesReady;
  await d2.engine.stepFrames(4);
  const director = d2.zones.director;
  const combat = d2.ctx.services.get('combat');
  const enemies = director?.enemies ?? [];
  return {
    zone: d2.zones.activeId,
    declared: d2.zones.active?.enemySpawns?.length ?? 0,
    placed: enemies.length,
    alive: director?.alive ?? 0,
    reported: d2.zones.report?.enemies ?? -1,
    ready: director?.ready ?? null,
    // Hostile means registered with the combat system on the enemy side; an
    // enemy that exists but is not a target cannot be fought.
    hostile: combat.targets.hostileTo('player').length,
    variants: enemies.map((e) => e.profile.variant),
  };
});
mark('travelled to the Blood Moor');
console.log('  arrival:', JSON.stringify(arrival));
check('travel: arrived on the moor', arrival.zone === 'bloodMoor', arrival.zone);
check(
  'travel: every declared spawn is placed',
  arrival.placed === arrival.declared && arrival.declared === 6,
  `${arrival.placed} of ${arrival.declared}`,
);
check('travel: all six are alive', arrival.alive === arrival.placed, `${arrival.alive} alive`);
check(
  'travel: all six are hostile combat targets',
  arrival.hostile === arrival.placed,
  `${arrival.hostile} hostile`,
);
check(
  'travel: the load report agrees with the live count',
  arrival.reported === arrival.placed,
  `report ${arrival.reported} vs live ${arrival.placed}`,
);

/**
 * Do they notice him, and do they come?
 *
 * The player is put in the middle of the arc rather than teleported on top of
 * one of them, and nothing else is forced: perception, the behaviour tree and
 * the steering all have to do their own work. Two things are measured because
 * they fail separately — an enemy can become `aware` and never move (a broken
 * steer) or close on the player without ever noticing him (a broken perception
 * with a fallback).
 */
const engagement = await page.evaluate(async ([step, seconds]) => {
  const d2 = window.__d2rim;
  const V = d2.three.Vector3;
  const director = d2.zones.director;
  const player = d2.ctx.services.get('character.player');
  const physics = d2.ctx.services.get('physics.world');

  // The centre of the declared arc, on standable ground.
  const centre = { x: 2.5, z: -7 };
  const spot = physics.findClearSpot(centre.x, centre.z, 0.42, player.height, 6);
  if (spot !== null) player.teleport(spot.x, spot.y + 0.06, spot.z);
  await d2.engine.stepFrames(2, step);

  const distance = (e) => e.footPosition(new V()).distanceTo(player.position);
  const first = director.enemies.map(distance);
  let everAware = 0;
  const engaged = new Set();
  const frames = Math.round(seconds / step);
  for (let i = 0; i < frames; i++) {
    await d2.engine.stepFrames(1, step);
    for (const e of director.enemies) {
      if (!e.alive) continue;
      if (e.aware) everAware++;
      if (['chase', 'telegraph', 'attack'].includes(e.state)) engaged.add(e.id);
    }
  }
  const last = director.enemies.map(distance);
  return {
    closed: first.filter((d, i) => last[i] < d - 0.5).length,
    engaged: engaged.size,
    awareFrames: everAware,
    nearest: +Math.min(...last).toFixed(2),
    states: director.enemies.map((e) => e.state),
    playerHealth: +d2.ctx.services.get('combat').vitals.health.value.toFixed(1),
  };
}, [STEP, QUICK ? 4 : 8]);
mark('drove the engagement');
console.log('  engagement:', JSON.stringify(engagement));
check(
  'engage: at least two skeletons noticed the player and acted',
  engagement.engaged >= 2,
  `${engagement.engaged} engaged, ${engagement.awareFrames} aware-frames`,
);
check(
  'engage: they actually closed the distance',
  engagement.closed >= 2,
  `${engagement.closed} closed by more than half a metre`,
);
check(
  'engage: somebody got within striking distance',
  engagement.nearest < 3,
  `nearest ${engagement.nearest} m`,
);

/* -------------------------------------------------------------------------- */
/* 2. balance                                                                 */
/* -------------------------------------------------------------------------- */

/** Stage helpers, installed once. */
await page.evaluate(() => {
  const d2 = window.__d2rim;
  const V = d2.three.Vector3;
  const svc = d2.ctx.services;

  window.__stage = {
    /** Park everyone far away and reset the player to full health, unhurt. */
    reset() {
      const director = d2.zones.director;
      const player = svc.get('character.player');
      const physics = svc.get('physics.world');
      const spot = physics.findClearSpot(0, 0, 0.42, player.height, 8) ?? { x: 0, y: 2, z: 0 };
      player.teleport(spot.x, spot.y + 0.06, spot.z);
      for (const e of director.enemies) {
        e.teleport(e.position.x + 400, e.position.y, e.position.z + 400);
      }
      const combat = svc.get('combat');
      combat.vitals.health.restore(combat.vitals.health.max);
      // Hit stop scales `ctx.time`, and every number below is per second of
      // simulation. Left on, a scenario would report a slower fight the harder
      // it hit, which is exactly backwards.
      d2.engine.getModule('combat.feedback')?.setHitStop(false);
      return { x: spot.x, y: spot.y, z: spot.z };
    },
    /** Bring `n` enemies of `variant` back and ring them round the player. */
    ring(variant, n, radius) {
      const director = d2.zones.director;
      const player = svc.get('character.player');
      const physics = svc.get('physics.world');
      const picked = director.enemies.filter((e) => e.alive && e.profile.variant === variant).slice(0, n);
      picked.forEach((e, i) => {
        const angle = (i / Math.max(1, picked.length)) * Math.PI * 2;
        const x = player.position.x + Math.sin(angle) * radius;
        const z = player.position.z + Math.cos(angle) * radius;
        const y = physics.groundHeight(x, z) ?? player.position.y;
        e.skipSpawn();
        e.teleport(x, y, z);
        e.alert();
      });
      return picked.length;
    },
    /** Everything a scenario reads, in one place. */
    snap() {
      const combat = svc.get('combat');
      const director = d2.zones.director;
      return {
        health: +combat.vitals.health.value.toFixed(2),
        maxHealth: combat.vitals.health.max,
        elapsed: d2.ctx.time.elapsed,
        alive: director.enemies.filter((e) => e.alive).length,
      };
    },
    nearest() {
      const director = d2.zones.director;
      const player = svc.get('character.player');
      let best = null;
      let bestD = Infinity;
      for (const e of director.enemies) {
        if (!e.alive) continue;
        const d = e.footPosition(new V()).distanceTo(player.position);
        if (d < bestD) {
          bestD = d;
          best = e;
        }
      }
      return best;
    },
  };

  window.__hits = [];
  d2.ctx.events.on('combat:hit', (e) => {
    if (e.targetLabel === 'player') window.__hits.push(+e.outcome.total.toFixed(2));
  });
});

/**
 * How hard a pack of `n` `variant`s hits a player who does not fight back.
 *
 * Health is *not* topped up during the run and regeneration is left on, because
 * both are part of the answer: a skeleton that cannot out-damage 1.6 hp/s of
 * passive regeneration is not a threat no matter what its damage roll says.
 */
const incoming = async (variant, n, seconds) => {
  const out = await page.evaluate(async ([variant, n, seconds, step]) => {
    const d2 = window.__d2rim;
    window.__stage.reset();
    await d2.engine.stepFrames(2, step);
    const staged = window.__stage.ring(variant, n, 1.6);
    window.__hits.length = 0;
    const start = window.__stage.snap();
    let swings = 0;
    let lowest = start.health;
    const frames = Math.round(seconds / step);
    for (let i = 0; i < frames; i++) {
      await d2.engine.stepFrames(1, step);
      const snap = window.__stage.snap();
      lowest = Math.min(lowest, snap.health);
      for (const e of d2.zones.director.enemies) {
        if (e.state === 'attack') swings++;
      }
      if (snap.health <= 0) break;
    }
    const end = window.__stage.snap();
    return {
      staged,
      simSeconds: +(end.elapsed - start.elapsed).toFixed(2),
      lost: +(start.health - end.health).toFixed(1),
      lowest: +lowest.toFixed(1),
      maxHealth: end.maxHealth,
      landed: window.__hits.filter((h) => h > 0).length,
      hitEvents: window.__hits.length,
      meanHit: +(
        window.__hits.filter((h) => h > 0).reduce((s, h) => s + h, 0) /
        Math.max(1, window.__hits.filter((h) => h > 0).length)
      ).toFixed(2),
      swingFrames: swings,
    };
  }, [variant, n, seconds, STEP]);
  out.dps = +(out.lost / Math.max(0.01, out.simSeconds)).toFixed(2);
  out.ttk = out.dps > 0 ? +(out.maxHealth / out.dps).toFixed(1) : Infinity;
  return out;
};

/** How long the Barbarian needs to put one `variant` down. */
const outgoing = async (variant, seconds) => {
  return page.evaluate(async ([variant, seconds, step]) => {
    const d2 = window.__d2rim;
    const combat = d2.ctx.services.get('combat');
    window.__stage.reset();
    await d2.engine.stepFrames(2, step);
    const staged = window.__stage.ring(variant, 1, 1.35);
    const target = window.__stage.nearest();
    if (target === null) return { staged, error: 'nothing to fight' };
    const swings = [];
    const off = d2.ctx.events.on('combat:swing', (p) => swings.push(p.moveId));
    const start = window.__stage.snap();
    const startHealth = target.health;
    const frames = Math.round(seconds / step);
    let used = frames;
    for (let i = 0; i < frames; i++) {
      combat.press('light');
      await d2.engine.stepFrames(1, step);
      if (!target.alive) {
        used = i + 1;
        break;
      }
    }
    off();
    const end = window.__stage.snap();
    return {
      staged,
      variant,
      killed: !target.alive,
      startHealth: +startHealth.toFixed(1),
      endHealth: +target.health.toFixed(1),
      simSeconds: +(end.elapsed - start.elapsed).toFixed(2),
      swings: swings.length,
      frames: used,
    };
  }, [variant, seconds, STEP]);
};

console.log('\n=== balance: what a skeleton does to a passive player ===');
const solo = await incoming('minion', 1, QUICK ? 10 : 16);
mark('one minion');
console.log('  one minion :', JSON.stringify(solo));
const pack = await incoming('minion', 2, QUICK ? 8 : 12);
mark('a pair of minions');
console.log('  two minions:', JSON.stringify(pack));
const warrior = await incoming('warrior', 1, QUICK ? 10 : 16);
mark('one warrior');
console.log('  one warrior:', JSON.stringify(warrior));

check(
  'incoming: a lone minion actually connects',
  solo.landed >= 2,
  `${solo.landed} landed hits in ${solo.simSeconds}s, mean ${solo.meanHit}`,
);
check(
  'incoming: a lone minion kills a passive player in 18-45 s',
  solo.ttk >= 18 && solo.ttk <= 45,
  `${solo.dps} dps -> ${solo.ttk}s (was ~90s)`,
);
check(
  'incoming: a single hit is worth noticing (>= 5% of the pool)',
  solo.meanHit >= solo.maxHealth * 0.05,
  `${solo.meanHit} of ${solo.maxHealth}`,
);
check(
  'incoming: a warrior hits harder than a minion',
  warrior.meanHit > solo.meanHit,
  `${warrior.meanHit} vs ${solo.meanHit}`,
);
check(
  'incoming: two of them are much worse than one',
  pack.dps >= solo.dps * 1.6,
  `${pack.dps} vs ${solo.dps} dps`,
);
check(
  'incoming: a pair kills a passive player inside 22 s',
  pack.ttk <= 22,
  `${pack.dps} dps -> ${pack.ttk}s`,
);

console.log('\n=== balance: what the Barbarian does back ===');
const vsMinion = await outgoing('minion', QUICK ? 10 : 14);
mark('player vs minion');
console.log('  vs minion :', JSON.stringify(vsMinion));
const vsWarrior = await outgoing('warrior', QUICK ? 14 : 20);
mark('player vs warrior');
console.log('  vs warrior:', JSON.stringify(vsWarrior));

check('outgoing: the minion dies', vsMinion.killed === true, JSON.stringify(vsMinion));
check(
  'outgoing: a minion goes down in 3-9 swings',
  vsMinion.swings >= 3 && vsMinion.swings <= 9,
  `${vsMinion.swings} swings in ${vsMinion.simSeconds}s`,
);
check(
  'outgoing: a minion goes down in under 6 s of swinging',
  vsMinion.simSeconds <= 6,
  `${vsMinion.simSeconds}s`,
);
check('outgoing: the warrior dies', vsWarrior.killed === true, JSON.stringify(vsWarrior));
check(
  'outgoing: a warrior is tougher but not a different genre (<= 14 swings)',
  vsWarrior.swings > vsMinion.swings && vsWarrior.swings <= 14,
  `${vsWarrior.swings} swings vs ${vsMinion.swings}`,
);

/* -- the two-sided summary, which is the number a designer actually reads --- */
console.log(
  '\ntime to kill:\n' +
    `  skeleton -> player   one minion ${solo.ttk}s | two minions ${pack.ttk}s | one warrior ${warrior.ttk}s\n` +
    `  player -> skeleton   minion ${vsMinion.simSeconds}s (${vsMinion.swings} swings) | ` +
    `warrior ${vsWarrior.simSeconds}s (${vsWarrior.swings} swings)`,
);

/* -------------------------------------------------------------------------- */
/* 3. the character sheet reaches the dice                                    */
/* -------------------------------------------------------------------------- */

/**
 * The seam this closes.
 *
 * `CombatSystem` used to resolve every player swing against its own
 * `PLAYER_OFFENSE` constant, and the RPG layer compensated by sending the
 * *difference* between the character's real offence and that constant as a
 * second damage packet. Damage therefore worked and nothing else did: attack
 * rating from gear could not rescue a base miss, because the roll had already
 * happened against the constant; and defence, resistances and block were
 * sheet-only, because incoming blows resolved against `PLAYER_DEFENSE_BASE`.
 *
 * Both are now read through `rpg.offense` / `rpg.defense`, and both halves are
 * measured here against `AttackOutcome.hitChance` — the chance the roll was
 * actually made against, reported by the damage model itself, so this cannot be
 * satisfied by a number that merely exists on a sheet somewhere.
 *
 * Skipped, not failed, when the RPG layer is absent: combat has to keep working
 * standalone and this file is the encounter harness, not the RPG one.
 */
console.log('\n=== the character sheet reaches the dice ===');
const sheet = await page.evaluate(() => {
  const d2 = window.__d2rim;
  const svc = d2.ctx.services;
  const rpg = svc.tryGet('rpg');
  const combat = svc.get('combat');
  if (rpg === undefined) return { present: false };

  // 1. Combat is reading the sheet, not the constant.
  const adopted =
    JSON.stringify(combat.playerOffense()) === JSON.stringify(rpg.offense()) &&
    combat.self.defense.defense === rpg.defense().defense;

  const beforeOffense = combat.playerOffense();
  const beforeDefense = combat.self.defense;

  // 2. Equip something with modifiers, through the game's own drop table.
  const loot = svc.get('rpg.loot');
  let equipped = null;
  for (let seed = 1; seed < 400 && equipped === null; seed++) {
    for (const item of loot.onEnemyDeath(4000 + seed, 'warrior#0').items) {
      if (!rpg.character.canEquip(item)) continue;
      if ((item.mods.attackRating ?? 0) <= 0 && (item.mods.defense ?? 0) <= 0) continue;
      rpg.character.acquire(item);
      if (rpg.character.equip(item).equipped) {
        equipped = { name: item.name, quality: item.quality, mods: item.mods };
        break;
      }
    }
  }
  loot.clear();

  return {
    present: true,
    adopted,
    equipped,
    beforeAr: beforeOffense.attackRating,
    afterAr: combat.playerOffense().attackRating,
    beforeDef: beforeDefense.defense,
    afterDef: combat.self.defense.defense,
    // Poise is combat's, not the sheet's; it must survive the adoption.
    poise: combat.self.defense.poise,
    resistances: combat.self.defense.resistances ?? {},
  };
});

if (!sheet.present) {
  console.log('  no RPG layer registered; skipping (combat must work standalone)');
} else {
  console.log('  sheet:', JSON.stringify(sheet));
  check('sheet: combat resolves the player against the character sheet', sheet.adopted === true);
  check(
    'sheet: gear that grants attack rating or defence was equipped',
    sheet.equipped !== null,
    sheet.equipped === null ? 'none found in 400 drops' : JSON.stringify(sheet.equipped.mods),
  );
  check(
    "sheet: combat's poise stays combat's after adoption",
    sheet.poise === 8,
    `poise ${sheet.poise}`,
  );

  // 3. The rolls themselves. `hitChance` is the number the die was compared
  //    against, so a change in it is proof the stat entered the roll rather
  //    than being applied to its result.
  const rolls = await page.evaluate(async ([step]) => {
    const d2 = window.__d2rim;
    const svc = d2.ctx.services;
    const rpg = svc.get('rpg');
    const combat = svc.get('combat');

    const seen = { player: [], enemy: [] };
    const off = d2.ctx.events.on('combat:hit', (e) => {
      (e.targetLabel === 'player' ? seen.player : seen.enemy).push(+e.outcome.hitChance.toFixed(4));
    });

    /** Swing at a staged minion until a roll is reported, and return it. */
    const swingAt = async (label) => {
      window.__stage.reset();
      await d2.engine.stepFrames(2, step);
      window.__stage.ring('minion', 1, 1.35);
      seen.enemy.length = 0;
      seen.player.length = 0;
      for (let i = 0; i < 90 && seen.enemy.length < 2; i++) {
        combat.press('light');
        await d2.engine.stepFrames(1, step);
      }
      // The enemy is swinging back on its own the whole time.
      for (let i = 0; i < 120 && seen.player.length < 1; i++) {
        await d2.engine.stepFrames(1, step);
      }
      return { label, onEnemy: seen.enemy[0] ?? null, onPlayer: seen.player[0] ?? null };
    };

    const geared = await swingAt('geared');
    const gearedAr = rpg.offense().attackRating;
    const gearedDef = rpg.defense().defense;

    // Strip everything off and re-roll. Bare-handed is the honest control: it
    // is the same code path with different numbers on the sheet.
    for (const item of [...rpg.character.equipment.items()]) {
      const slot = rpg.character.equipment.slotOf(item);
      if (slot !== null) rpg.character.unequip(slot);
    }
    const bare = await swingAt('bare');

    off();
    return {
      geared,
      bare,
      gearedAr,
      bareAr: rpg.offense().attackRating,
      gearedDef,
      bareDef: rpg.defense().defense,
    };
  }, [STEP]);
  mark('drove the geared and bare rolls');
  console.log('  rolls:', JSON.stringify(rolls));

  check(
    "sheet: gear changes the player's real chance to hit",
    rolls.geared.onEnemy !== null &&
      rolls.bare.onEnemy !== null &&
      rolls.geared.onEnemy > rolls.bare.onEnemy,
    `geared ${rolls.geared.onEnemy} vs bare ${rolls.bare.onEnemy}`,
  );
  check(
    "sheet: armour changes the enemy's chance to hit the player",
    rolls.geared.onPlayer !== null &&
      rolls.bare.onPlayer !== null &&
      rolls.geared.onPlayer < rolls.bare.onPlayer,
    `geared ${rolls.geared.onPlayer} vs bare ${rolls.bare.onPlayer}`,
  );
}

if (errors.length > 0) {
  console.log('pageErrors:', errors.slice(0, 5));
  failures.push('page errors');
}

await browser.close();
server.kill();
console.log(`\n${failures.length === 0 ? 'ENCOUNTER: ALL PASS' : `ENCOUNTER: ${failures.length} FAILED`}`);
for (const failure of failures) console.log(`  - ${failure}`);
process.exit(failures.length === 0 ? 0 : 1);
