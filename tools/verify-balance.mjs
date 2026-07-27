/**
 * tools/verify-balance.mjs
 *
 * Measures how long a fight takes, in both directions, and asserts a budget.
 *
 * The numbers in `CombatSystem` and `SKELETON_PROFILES` were tuned against a
 * system that dealt no damage at all, so nothing in this project has ever felt
 * them. This harness is the instrument: it drives the real combo state machine
 * and the real hit resolution at 60 Hz, with hit stop disabled (it only costs
 * renders and it distorts wall-clock-to-simulated-time), and reports:
 *
 *   - **Outgoing.** Swings and simulated seconds for the Barbarian to kill each
 *     skeleton variant from full health, point blank.
 *   - **Incoming.** Simulated seconds for a stationary, non-blocking player to
 *     be killed by one skeleton and by a pack of three.
 *   - **Providers.** That `rpg.offense` / `rpg.defense` are load-bearing:
 *     equipping attack rating moves the real hit chance the swing rolls
 *     against, and equipping defence/resistance reduces the damage a scripted
 *     skeleton blow actually applies.
 *
 * Intended time-to-kill, and the budget asserted below:
 *
 * | matchup                     | target        | asserted      |
 * |-----------------------------|---------------|---------------|
 * | Barbarian -> minion         | 3-5 swings    | 2-6 swings    |
 * | Barbarian -> warrior        | 5-8 swings    | 4-10 swings   |
 * | one skeleton -> player      | 12-25 s       | 8-32 s        |
 * | three skeletons -> player   | 5-11 s        | 3-14 s        |
 *
 * A fight the player wins in two swings has no shape; one he wins in twelve is
 * an errand. A single skeleton must be survivable long enough to be read and
 * answered; three must be a genuine threat on the order of ten seconds, which
 * is what makes the Blood Moor's arc of six a fight rather than a queue.
 *
 * Usage: `node tools/verify-balance.mjs [--json out.json]`
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';

import { chromium } from 'playwright';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';

const PORT = 5252;
const JSON_OUT = (() => {
  const i = process.argv.indexOf('--json');
  return i === -1 ? null : process.argv[i + 1];
})();

const failures = [];
const results = {};
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
const page = await browser.newPage({ viewport: { width: 480, height: 270 } });
page.setDefaultNavigationTimeout(240_000);
page.setDefaultTimeout(240_000);
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

const url = `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&fade=0&zone=bloodMoor`;
console.log(`loading ${url}`);
await page.goto(url, { waitUntil: 'load' });
await page.evaluate(() => window.__d2rim.ready);
await page.evaluate(async () => {
  const d2 = window.__d2rim;
  await d2.engine.stepFrames(2);
  await d2.zones.enemiesReady;
  await d2.engine.stepFrames(4);
  // Hit stop scales `ctx.time`, which would make simulated seconds a lie and
  // costs 16x the renders per simulated second on a software rasteriser.
  d2.engine.getModule('combat.feedback')?.setHitStop(false);
});

/* -- shared page helpers --------------------------------------------------- */

await page.evaluate(() => {
  const d2 = window.__d2rim;
  const T = d2.three;
  const s = () => d2.ctx.services;
  const DT = 1 / 60;

  window.__bal = {
    DT,
    combat: () => s().get('combat'),
    director: () => s().get('ai.director'),
    player: () => s().get('character.player'),

    /** Move every enemy far away so a duel is a duel. */
    clearField(except = []) {
      const keep = new Set(except);
      for (const e of this.director().enemies) {
        if (keep.has(e)) continue;
        e.skipSpawn?.();
        e.teleport(e.position.x + 400, e.position.y, e.position.z + 400);
      }
    },

    /** A live enemy of `variant`, restored to full health, or null. */
    pick(variant) {
      const e = this.director().enemies.find(
        (x) => x.alive && x.profile.variant === variant && x.health >= x.profile.maxHealth * 0.999,
      );
      return e ?? null;
    },

    /** Stand `enemy` `distance` metres in front of the player, awake. */
    stage(enemy, distance) {
      const player = this.player();
      const forward = player.forward(new T.Vector3());
      const spot = player.position.clone().addScaledVector(forward, distance);
      enemy.skipSpawn?.();
      enemy.teleport(spot.x, spot.y, spot.z);
      enemy.alert();
    },

    /**
     * Restore the player to full.
     *
     * `Pool.value` is getter-only, so this has to go through `revive`/`refill`
     * — assigning to it throws in a module's strict mode, which is exactly the
     * kind of harness bug that reads as a game bug.
     */
    healPlayer() {
      this.combat().vitals.revive();
    },
  };
});

/* -- 1. incoming: how dangerous is a skeleton? -----------------------------
 *
 * Measured first, and that ordering is load-bearing: the outgoing measurement
 * leaves three corpses in the moor, and a pack assembled out of corpses is not
 * a pack. */

/**
 * Stand still and take it. The player never blocks, never swings and never
 * moves; the skeletons are re-planted at their stand-off each frame so that
 * knockback and pursuit do not turn a damage measurement into a chase.
 */
const survival = (count) =>
  page.evaluate(async (n) => {
    const b = window.__bal;
    const d2 = window.__d2rim;
    const T = d2.three;
    const director = b.director();
    const combat = b.combat();
    const player = b.player();

    // Revive everything so a second measurement is not fought by corpses.
    for (const e of director.enemies) e.vitals?.revive?.();
    const pack = director.enemies.filter((e) => e.alive).slice(0, n);
    if (pack.length < n) return null;
    b.clearField(pack);
    b.healPlayer();

    const anchor = player.position.clone();
    const ring = pack.map((_, i) => (i / n) * Math.PI * 2);
    for (let i = 0; i < pack.length; i++) {
      const a = ring[i];
      pack[i].skipSpawn?.();
      pack[i].teleport(anchor.x + Math.sin(a) * 1.1, anchor.y, anchor.z + Math.cos(a) * 1.1);
      pack[i].alert();
    }
    await d2.engine.stepFrames(2);

    const hits = [];
    const off = d2.ctx.events.on('combat:hit', (p) => {
      if (p.target === combat.self.id) hits.push(+p.outcome.total.toFixed(1));
    });

    const startHealth = combat.vitals.health.value;
    let frames = 0;
    // Three fixed steps per render. Everything an enemy decides happens in
    // `fixedUpdate`, which still runs at exactly 60 Hz; only the *sweep* in
    // `lateUpdate` samples less often, and it sweeps from the previous pose so
    // the arc is still covered. This exists because a software rasteriser makes
    // a rendered frame the only expensive part of a simulated second, and a
    // measurement nobody can afford to run is a measurement nobody runs.
    const STRIDE = 3 / 60;
    while (combat.vitals.health.value > 0 && frames < 420) {
      for (let i = 0; i < pack.length; i++) {
        const a = ring[i];
        pack[i].teleport(anchor.x + Math.sin(a) * 1.1, anchor.y, anchor.z + Math.cos(a) * 1.1);
      }
      player.teleport(anchor.x, anchor.y, anchor.z);
      await d2.engine.stepFrames(1, STRIDE);
      frames++;
    }
    off();
    const seconds = frames * STRIDE;
    return {
      count: n,
      startHealth: +startHealth.toFixed(1),
      endHealth: +combat.vitals.health.value.toFixed(1),
      seconds: +seconds.toFixed(2),
      landed: hits.length,
      mean: hits.length === 0 ? 0 : +(hits.reduce((a, x) => a + x, 0) / hits.length).toFixed(1),
      dps: +((startHealth - combat.vitals.health.value) / seconds).toFixed(2),
      hitsPerSecond: +(hits.length / seconds).toFixed(2),
    };
  }, count);

for (const count of [1, 3]) {
  const r = await survival(count);
  results[`survive.${count}`] = r;
  if (r === null) {
    check(`${count} skeleton(s) threaten the player`, false, 'not enough live skeletons');
    continue;
  }
  console.log(`  vs ${count}: ${JSON.stringify(r)}`);
}

const solo = results['survive.1'];
const pack = results['survive.3'];
check(
  'one skeleton kills a passive player in 8-32 s',
  solo !== null && solo.endHealth <= 0 && solo.seconds >= 8 && solo.seconds <= 32,
  `${solo?.seconds}s, ${solo?.landed} hits, ${solo?.mean} mean, ${solo?.dps} dps`,
);
check(
  'three skeletons kill a passive player in 3-14 s',
  pack !== null && pack.endHealth <= 0 && pack.seconds >= 3 && pack.seconds <= 14,
  `${pack?.seconds}s, ${pack?.landed} hits, ${pack?.mean} mean, ${pack?.dps} dps`,
);
check(
  'a pack is meaningfully worse than a single skeleton',
  solo !== null && pack !== null && pack.dps > solo.dps * 1.6,
  `${solo?.dps} dps -> ${pack?.dps} dps`,
);

/* -- 2. outgoing: how long does the Barbarian need? ------------------------ */

/**
 * Swap the registered offence/defence providers for the ones that were there
 * before the class combat floor existed — the raw character sheet.
 *
 * This is what makes a before/after measurement affordable: both states are
 * measured in one browser session, against the same build, the same enemies and
 * the same combat RNG, so the difference between them is the change and nothing
 * else. Running two sessions and comparing would fold in a fresh RNG stream and
 * a different set of surviving skeletons.
 *
 * The replacement is exact, not an approximation: with no skill armed the old
 * `RpgSystem.offense()` reduced to `character.offense(PLAYER_OFFENSE.criticalChance)`
 * and the old `defense()` to `character.defense({ poise: 3 })`.
 */
const setProviders = (mode) =>
  page.evaluate((m) => {
    const d2 = window.__d2rim;
    const s = d2.ctx.services;
    const rpg = s.get('rpg');
    s.unregister('rpg.offense');
    s.unregister('rpg.defense');
    if (m === 'sheet') {
      s.register('rpg.offense', { offense: () => rpg.character.offense(0.09) });
      s.register('rpg.defense', { defense: () => rpg.character.defense({ poise: 3 }) });
    } else {
      s.register('rpg.offense', { offense: () => rpg.offense() });
      s.register('rpg.defense', { defense: () => rpg.defense() });
    }
    const o = s.get('rpg.offense').offense();
    const d = s.get('rpg.defense').defense();
    return {
      mode: m,
      attackRating: o.attackRating,
      damage: o.damage.physical,
      blockChance: +(d.blockChance ?? 0).toFixed(2),
      physicalReduction: +(d.physicalReduction ?? 0).toFixed(3),
      flatReduction: d.flatReduction ?? 0,
    };
  }, mode);

/**
 * Swing for a fixed window and report what came off the target.
 *
 * A fixed window rather than "swing until dead", because the *before* state
 * does not reach dead inside any budget this container can afford, and a
 * measurement that times out is not a measurement. Twelve simulated seconds is
 * about ten swings, which is already past the point where a fight has stopped
 * being one.
 *
 * The player is re-planted in front of the enemy every frame so a knockback
 * cannot end the measurement by walking the target out of reach — this measures
 * the damage budget, not the pursuit.
 */
const killTime = (variant) =>
  page.evaluate(async (v) => {
    const b = window.__bal;
    const d2 = window.__d2rim;
    const T = d2.three;
    const enemy = b.pick(v);
    if (enemy === null) return null;
    b.clearField([enemy]);
    b.stage(enemy, 1.4);
    await d2.engine.stepFrames(2);

    const player = b.player();
    const combat = b.combat();
    const start = enemy.health;
    const hits = [];
    let swings = 0;
    const off = d2.ctx.events.on('combat:swing', () => swings++);
    const offHit = d2.ctx.events.on('combat:hit', (p) => {
      if (p.attacker === combat.self.id) hits.push(+p.outcome.total.toFixed(1));
    });

    let frames = 0;
    const STRIDE = 3 / 60;
    const anchor = player.position.clone();
    while (enemy.alive && frames < 240) {
      combat.press('light');
      // Re-plant the *player* only, and only when he has actually drifted.
      //
      // An earlier version of this loop teleported both combatants every single
      // frame to "hold the duel geometry", and it measured 27 swings with **0
      // landed**. That was the harness, not the game: `Hitbox.track` resolves a
      // contact by sweeping the blade from its previous pose to its current
      // one, and teleporting the wielder every frame makes every one of those
      // sweeps a degenerate zero-length segment. The staging suppressed exactly
      // the mechanism it was supposed to be measuring — the same shape of
      // defect as a foot-plant test whose fixture only feeds it the one input
      // where a broken implementation agrees with a correct one.
      //
      // The enemy is left alone entirely. Knockback pushing it out of reach is
      // a real part of the fight, and it comes back on its own.
      if (player.position.distanceTo(anchor) > 0.35) {
        player.teleport(anchor.x, anchor.y, anchor.z);
      }
      await d2.engine.stepFrames(1, STRIDE);
      frames++;
    }
    void T;
    off();
    offHit();
    return {
      variant: v,
      maxHealth: start,
      alive: enemy.alive,
      removed: +(start - enemy.health).toFixed(1),
      seconds: +(frames * (3 / 60)).toFixed(2),
      swings,
      landed: hits.length,
      hits,
      mean: hits.length === 0 ? 0 : +(hits.reduce((a, x) => a + x, 0) / hits.length).toFixed(1),
    };
  }, variant);

// The sheet-only state first: it leaves its targets alive, so the "after" pass
// still has fresh skeletons of every variant to work on.
for (const mode of ['sheet', 'floor']) {
  const stats = await setProviders(mode);
  results[`providers.${mode}`] = stats;
  console.log(`  providers[${mode}]: ${JSON.stringify(stats)}`);
  for (const variant of mode === 'sheet' ? ['minion'] : ['minion', 'warrior', 'rogue']) {
    const r = await killTime(variant);
    results[`kill.${mode}.${variant}`] = r;
    if (r === null) {
      check(`${mode}: Barbarian fights a ${variant}`, false, 'no fresh target of that variant');
      continue;
    }
    console.log(`  ${mode}/${variant}: ${JSON.stringify(r)}`);
  }
}

const before = results['kill.sheet.minion'];
const minion = results['kill.floor.minion'];
const warrior = results['kill.floor.warrior'];
const rogue = results['kill.floor.rogue'];

console.log(
  `\n  minion, sheet-only:  ${before?.landed} landed / ${before?.swings} swings, ` +
    `${before?.mean} mean, ${before?.removed}/${before?.maxHealth} hp removed in ${before?.seconds}s ` +
    `(${before?.alive ? 'still alive' : 'dead'})`,
);
console.log(
  `  minion, class floor: ${minion?.landed} landed / ${minion?.swings} swings, ` +
    `${minion?.mean} mean, ${minion?.removed}/${minion?.maxHealth} hp removed in ${minion?.seconds}s ` +
    `(${minion?.alive ? 'still alive' : 'dead'})\n`,
);

check(
  'the sheet-only Barbarian could NOT put a minion down in 12 s',
  before !== null && before.alive === true,
  `${before?.removed}/${before?.maxHealth} hp in ${before?.seconds}s`,
);
check('Barbarian kills a minion', minion !== null && !minion.alive, `${minion?.seconds}s`);
check('Barbarian kills a warrior', warrior !== null && !warrior.alive, `${warrior?.seconds}s`);
check('Barbarian kills a rogue', rogue !== null && !rogue.alive, `${rogue?.seconds}s`);
check(
  'a minion dies in 2-6 landed hits',
  minion !== null && minion.landed >= 2 && minion.landed <= 6,
  `${minion?.landed} landed of ${minion?.swings} swings, ${minion?.mean} mean vs ${minion?.maxHealth} hp`,
);
check(
  'a warrior dies in 4-10 landed hits',
  warrior !== null && warrior.landed >= 4 && warrior.landed <= 10,
  `${warrior?.landed} landed of ${warrior?.swings} swings, ${warrior?.mean} mean vs ${warrior?.maxHealth} hp`,
);
check(
  'the class floor is the difference, and it is large',
  before !== null && minion !== null && minion.removed > before.removed * 2,
  `${before?.removed} hp removed -> ${minion?.removed} hp removed in the same 12 s`,
);

/* -- 3. the rpg.offense / rpg.defense providers are load-bearing ----------- */

/**
 * The whole chain, end to end, in one page call:
 *
 *   item mods -> `Character` sheet -> `rpg.offense`/`rpg.defense` provider
 *   -> `CombatSystem.playerOffense()`/`characterDefense()` -> the actual roll.
 *
 * The items are authored as literals rather than pulled out of
 * `generateItem`, because the generator has its own unit tests and what is on
 * trial here is the *plumbing* — the link that was broken, and that a sheet-only
 * stat looks exactly like a working one from the character screen.
 *
 * Damage is measured by firing identical, always-hitting probe blows through
 * `CombatSystem.resolve` — the same entry point a skeleton's blade uses — and
 * averaging what `Vitals` actually lost.
 */
const providers = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const T = d2.three;
  const s = d2.ctx.services;
  const combat = s.get('combat');
  const rpg = s.tryGet('rpg');
  if (rpg === undefined) return { present: false };
  const character = rpg.character;
  const self = combat.self;

  const item = (over) => ({
    uid: `probe-${over.baseId}-${Math.random()}`,
    quality: 'magic',
    itemLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    affixes: [],
    width: 1,
    height: 1,
    value: 1,
    maxDurability: 0,
    durability: 0,
    seed: 1,
    twoHanded: false,
    ...over,
  });

  /** Average damage applied by `n` identical unavoidable 20-point blows. */
  const volley = (n) => {
    const offense = {
      level: 3,
      attackRating: 10_000,
      damage: { physical: { min: 20, max: 20 } },
      criticalChance: 0,
      criticalMultiplier: 2,
      alwaysHits: true,
      unblockable: true,
    };
    let total = 0;
    for (let i = 0; i < n; i++) {
      const before = combat.vitals.health.value;
      combat.resolve(null, self, {
        source: null,
        offense,
        move: { id: 'probe', damageScale: 1, knockback: 0, staggerScale: 0, unblockable: true },
        point: self.footPosition(new T.Vector3()).clone(),
        normal: new T.Vector3(0, 0, 1),
        direction: new T.Vector3(0, 0, -1),
      });
      total += before - combat.vitals.health.value;
      combat.vitals.revive();
    }
    return +(total / n).toFixed(2);
  };

  /** Hit chance the swing rolls, against a real skeleton minion's defence. */
  const target = s.get('ai.director').enemies.find((e) => e.profile.variant === 'minion');
  const dd = target.profile.defense;
  const chanceOf = (o) => {
    const ar = Math.max(0, o.attackRating);
    const gear = ar + dd.defense <= 0 ? 1 : ar / (ar + dd.defense);
    const levels = (2 * Math.max(1, o.level)) / (Math.max(1, o.level) + Math.max(1, dd.level));
    return +Math.min(0.95, Math.max(0.05, gear * levels)).toFixed(4);
  };

  const before = combat.playerOffense();
  const beforeDefense = combat.characterDefense();
  const beforeDamage = volley(30);
  // Read *now*, before anything is equipped. Sampling this in the returned
  // object literal compared a pre-equip swing against a post-equip sheet and
  // reported a disagreement that did not exist.
  const sheetAr = rpg.offense().attackRating;

  const ring = item({ baseId: 'probe-ring', name: 'Probe Ring of Precision', category: 'ring', slot: 'ring', mods: { attackRating: 300 } });
  character.acquire(ring);
  const equippedRing = character.equip(ring).equipped;
  character.touch();
  const afterAr = combat.playerOffense();

  // `damageReduction` / `damageReducedPercent`, not `damageReduced` — the
  // earlier spelling was not a `ModifierKey` at all, so the plate carried no
  // mitigation and the check that followed could only ever fail.
  const plate = item({ baseId: 'probe-body', name: 'Probe Plate of the Whale', category: 'armor', slot: 'body', defense: 240, mods: { defense: 240, resistAll: 40, damageReduction: 4, damageReducedPercent: 0.08 } });
  character.acquire(plate);
  const equippedPlate = character.equip(plate).equipped;
  character.touch();
  const afterDefense = combat.characterDefense();
  const afterDamage = volley(30);

  const slim = (d) =>
    d === null
      ? null
      : {
          defense: d.defense,
          physicalReduction: +(d.physicalReduction ?? 0).toFixed(3),
          flatReduction: d.flatReduction ?? 0,
          fire: d.resistances?.fire ?? 0,
          cold: d.resistances?.cold ?? 0,
        };

  return {
    present: true,
    offenseKeyRegistered: s.has('rpg.offense'),
    defenseKeyRegistered: s.has('rpg.defense'),
    sheetAr,
    swingAr: before.attackRating,
    constantAr: 150,
    equippedRing,
    equippedPlate,
    arBefore: before.attackRating,
    arAfter: afterAr.attackRating,
    hitBefore: chanceOf(before),
    hitAfter: chanceOf(afterAr),
    defBefore: slim(beforeDefense),
    defAfter: slim(afterDefense),
    damageBefore: beforeDamage,
    damageAfter: afterDamage,
    rawDamage: 20,
  };
});
console.log(`  providers: ${JSON.stringify(providers)}`);
results.providers = providers;

check('the RPG layer is loaded', providers.present === true);
check('rpg.offense is registered', providers.offenseKeyRegistered === true);
check('rpg.defense is registered', providers.defenseKeyRegistered === true);
check(
  'the swing resolves against the character sheet, not the PLAYER_OFFENSE constant',
  providers.swingAr === providers.sheetAr,
  `swing AR ${providers.swingAr}, sheet AR ${providers.sheetAr}, constant ${providers.constantAr}`,
);
check(
  'equipping +attack rating raises the AR the swing rolls with',
  providers.equippedRing === true && providers.arAfter > providers.arBefore,
  `${providers.arBefore} -> ${providers.arAfter}`,
);
check(
  "and therefore raises the player's real hit chance",
  providers.hitAfter > providers.hitBefore + 0.02,
  `${(providers.hitBefore * 100).toFixed(1)}% -> ${(providers.hitAfter * 100).toFixed(1)}% vs a minion`,
);
check(
  'incoming hits resolve against the sheet defence',
  providers.defBefore !== null,
  JSON.stringify(providers.defBefore),
);
// Two separate claims, because defence rating and damage reduction are two
// separate mechanisms and conflating them is how "armour does nothing" hides:
// in Diablo II a defence *rating* buys a lower chance to be hit and does not
// touch the damage of a blow that lands, while `damageReduction` and
// `damageReducedPercent` reduce what a landed blow applies. The probe blow is
// `alwaysHits`, so it isolates the second.
check(
  'the class floor alone already mitigates a landed blow',
  providers.damageBefore < providers.rawDamage,
  `raw ${providers.rawDamage} -> ${providers.damageBefore} bare ` +
    `(5% + 1 flat from BARBARIAN_ARMS)`,
);
check(
  'armour reduction is not sheet-only: it cuts damage actually applied',
  providers.damageAfter < providers.damageBefore,
  `${providers.damageBefore} bare -> ${providers.damageAfter} armoured`,
);
check(
  'armour defence rating reaches the sheet the roll uses',
  (providers.defAfter?.defense ?? 0) > (providers.defBefore?.defense ?? 0),
  `${providers.defBefore?.defense} -> ${providers.defAfter?.defense}`,
);

/* -- done ------------------------------------------------------------------ */

if (JSON_OUT !== null) writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));
await browser.close();
server.kill('SIGTERM');
console.log(
  failures.length === 0 ? '\nALL PASS' : `\n${failures.length} FAILED: ${failures.join(', ')}`,
);
process.exit(failures.length === 0 ? 0 : 1);
