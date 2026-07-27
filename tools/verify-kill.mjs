/**
 * tools/verify-kill.mjs
 *
 * Can the Barbarian win a fight? Measured in the running game.
 *
 * Every combat number this project had measured before this harness existed was
 * damage *taken*: how long a passive player survives one skeleton, and how long
 * he survives three. Nothing had ever established the other direction — that a
 * player swinging an axe in the real game, at a skeleton that is moving and
 * fighting back, kills it. A combat loop proven lethal in exactly one direction
 * is not a game, and `tools/verify-balance.mjs` says so in as many words.
 *
 * ### The staging problem, and why this harness does not have it
 *
 * The obvious way to hold a duel still is to teleport both fighters to fixed
 * marks every frame. It does not work, and the way it fails is instructive:
 * `WeaponHitbox.track` resolves a contact by sweeping the blade from its
 * *previous* pose to its current one, so teleporting the wielder every frame
 * makes every sweep a zero-length segment. The first attempt at this
 * measurement did exactly that and recorded 27 swings and 0 landed hits — it
 * suppressed the mechanism it was there to measure.
 *
 * The opposite mistake is to leave both parties free and hope. Then the
 * skeleton sidesteps out of the swing arc, the harness has no way to turn the
 * player to follow it, and the Barbarian swings at where the skeleton was.
 *
 * What a real player does in that situation is *move the mouse*. So that is
 * what this harness does:
 *
 *   - **Aim** is fed as pointer pixels through `Input.nudgePointer`, the same
 *     accumulator a `mousemove` writes to. `PlayerController` folds it into the
 *     look angles, the body turns to follow at its own `turnRate`, and the
 *     camera rig follows — untouched game code, all of it. The per-frame turn
 *     is capped at {@link TURN_LIMIT} rad/s so the harness cannot aim faster
 *     than a hand could. Critically, a rotation is **not** a teleport: the
 *     blade keeps moving between frames, so the sweep stays a real sweep.
 *   - **Movement** is real `keydown`/`keyup` events for `KeyW`, dispatched to
 *     the window, closing to weapon range and holding there.
 *   - **Attacks** are real `mousedown`/`mouseup` on the canvas, which is what
 *     `Mouse0` is bound to. The harness probes that this works before it
 *     measures anything, and falls back to `CombatSystem.press` — the exact
 *     call the input handler makes one line later — only if the DOM route is
 *     dead, and says which one it used.
 *
 * The only teleport is the one that stages the encounter, once, before the
 * clock starts. After that both parties act.
 *
 * ### What it measures
 *
 *   - Swings, landed hits and simulated seconds for the Barbarian to kill each
 *     skeleton variant, plus the health he has left when it drops.
 *   - The same four fights with the best weapon the drop table will produce, to
 *     show gear moves the offence the swing resolves against and not only the
 *     character sheet.
 *   - A three-skeleton pack fight, fought for real in both directions, with the
 *     starting weapon put back on first.
 *
 * In that order, and the order is load-bearing: `EnemyDirector` splices a
 * corpse out of its list once the corpse has finished sinking, so a fight long
 * enough for that to happen permanently reduces the roster. Duels are short and
 * revive their target immediately; the pack fight is neither, so it goes last.
 *
 * Usage: `node tools/verify-kill.mjs [--json out.json]`
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';

import { chromium } from 'playwright';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';

const PORT = 5254;
const JSON_OUT = (() => {
  const i = process.argv.indexOf('--json');
  return i === -1 ? null : process.argv[i + 1];
})();

/**
 * Seconds of simulated time per stepped frame.
 *
 * The fixed-step half of the engine still runs at exactly 60 Hz inside each
 * step, so every decision an enemy makes and every tick of the combo machine
 * happens at the authored rate. Only the *sweep*, which lives in `lateUpdate`,
 * samples once per stepped frame — and because it sweeps from the previous pose
 * the arc between samples is still covered. This exists because a rendered
 * frame is the expensive part on a software rasteriser, and a measurement
 * nobody can afford to run is a measurement nobody runs.
 */
const STRIDE = 2 / 60;

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
const page = await browser.newPage({ viewport: { width: 320, height: 180 } });
page.setDefaultNavigationTimeout(300_000);
page.setDefaultTimeout(300_000);
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

const url = `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&fade=0&zone=bloodMoor`;

/**
 * Reload the page for a fresh Blood Moor and a fresh level-1 character.
 *
 * ### Why every phase gets its own boot
 *
 * The moor's roster is six skeletons — two minions, two warriors, **one** rogue
 * and **one** mage — and a dead one does not come back. `Vitals.revive()` looks
 * like it should bring a corpse back and does not: it restores the health pool,
 * but `EnemyBase` keeps its own state machine, and a revived corpse is still in
 * `state === 'dead'`, so it goes on sinking and `EnemyDirector` culls it 4.6 s
 * after it fell. `alive` reads true the whole time, which is why the effect is
 * so hard to see — an earlier run of this harness watched the rogue and a
 * warrior evaporate mid-measurement and reported "no fresh target of that
 * variant", which reads exactly like a broken spawn table.
 *
 * So the roster is treated as consumable, and each phase starts from a full
 * one. This also makes the magic-weapon phase a clean A/B: the character is
 * level 1 again, with the same attributes and the same skills, and the weapon
 * is the only thing that differs. Without the reload the fourth duel was fought
 * by a character two levels up from the first, and every number in it was
 * confounded.
 */
const boot = async (label) => {
  console.log(`\n[boot] ${label}`);
  await page.goto(url, { waitUntil: 'load' });
  await page.evaluate(() => window.__d2rim.ready);
  await page.evaluate(async () => {
    const d2 = window.__d2rim;
    await d2.engine.stepFrames(2);
    await d2.zones.enemiesReady;
    await d2.engine.stepFrames(4);
    // Hit stop scales `ctx.time`, which would make simulated seconds a lie and
    // costs many times the renders per simulated second on a software rasteriser.
    d2.engine.getModule('combat.feedback')?.setHitStop(false);
  });
  await installDriver();
  const probe = await probeAttackRoute();
  console.log(`[boot] roster ${JSON.stringify(probe.census)}, attack route ${probe.route}`);
  return probe;
};

/* -- the driver, installed in the page ------------------------------------- */

const installDriver = () =>
  page.evaluate(
  ({ stride, turnLimit }) => {
    const d2 = window.__d2rim;
    const T = d2.three;
    const s = () => d2.ctx.services;

    const wrap = (a) => {
      const w = (a + Math.PI) % (Math.PI * 2);
      return (w < 0 ? w + Math.PI * 2 : w) - Math.PI;
    };

    window.__duel = {
      STRIDE: stride,
      TURN_LIMIT: turnLimit,
      /** How the harness presses attack. Decided by `probeAttackRoute`. */
      attackRoute: 'unknown',
      combat: () => s().get('combat'),
      director: () => s().get('ai.director'),
      player: () => s().get('character.player'),
      rpg: () => s().tryGet('rpg'),
      canvas: () => document.querySelector('canvas'),

      /* -- input, as a player would produce it ----------------------------- */

      key(code, down) {
        window.dispatchEvent(
          new KeyboardEvent(down ? 'keydown' : 'keyup', { code, bubbles: true }),
        );
      },

      /** A real click on the canvas, which is what `Mouse0` is bound to. */
      clickAttack(button = 0) {
        const canvas = this.canvas();
        if (canvas === null) return;
        canvas.dispatchEvent(new MouseEvent('mousedown', { button, bubbles: true }));
        window.dispatchEvent(new MouseEvent('mouseup', { button, bubbles: true }));
      },

      /**
       * Turn the camera toward `point` by feeding pointer pixels, capped at a
       * human turn rate. Returns the residual aim error in radians.
       */
      aimAt(point, dt) {
        const player = this.player();
        const from = player.position;
        const dx = point.x - from.x;
        const dz = point.z - from.z;
        if (dx * dx + dz * dz < 1e-8) return 0;
        const len = Math.hypot(dx, dz);
        // Camera convention: forward is `(-sin(yaw), 0, -cos(yaw))`.
        const want = Math.atan2(-dx / len, -dz / len);
        const err = wrap(want - player.yaw);
        const step = Math.sign(err) * Math.min(Math.abs(err), this.TURN_LIMIT * dt);
        const sensitivity = 0.0022; // PlayerController DEFAULTS.mouseSensitivity
        d2.ctx.input.nudgePointer(-step / sensitivity, 0);
        return wrap(err - step);
      },

      /** Chest-height aim point of an enemy. */
      chest(enemy) {
        const out = enemy.footPosition(new T.Vector3());
        out.y += enemy.hitHeight * 0.55;
        return out;
      },

      planarDistance(a, b) {
        return Math.hypot(a.x - b.x, a.z - b.z);
      },

      /* -- staging ---------------------------------------------------------- */

      /** Move every other enemy far away so a duel is a duel. */
      clearField(keepList) {
        const keep = new Set(keepList);
        for (const e of this.director().enemies) {
          if (keep.has(e)) continue;
          e.skipSpawn?.();
          e.teleport(e.position.x + 400, e.position.y, e.position.z + 400);
        }
      },

      /** How many skeletons are still in the director's list, by variant. */
      census() {
        const out = {};
        for (const e of this.director().enemies) {
          const v = e.profile.variant;
          out[v] = (out[v] ?? 0) + (e.alive ? 1 : 0);
        }
        return out;
      },

      /** A live, full-health enemy of `variant`, or null. */
      pick(variant) {
        return (
          this.director().enemies.find(
            (x) =>
              x.alive &&
              x.profile.variant === variant &&
              x.health >= x.profile.maxHealth * 0.999,
          ) ?? null
        );
      },

      healPlayer() {
        this.combat().vitals.revive();
      },

      /** Release everything the harness might be holding. */
      releaseAll() {
        this.key('KeyW', false);
        this.key('KeyS', false);
        this.key('KeyA', false);
        this.key('KeyD', false);
      },
    };
  },
    { stride: STRIDE, turnLimit: 8.0 },
  );

/* -- does a real click reach the combo machine? ---------------------------- */

const probeAttackRoute = () =>
  page.evaluate(async () => {
    const d = window.__duel;
    const d2 = window.__d2rim;
    let swings = 0;
    const off = d2.ctx.events.on('combat:swing', () => swings++);
    d.clickAttack(0);
    await d2.engine.stepFrames(2, 1 / 60);
    off();
    d.attackRoute = swings > 0 ? 'dom-click' : 'combat.press';
    return { swings, route: d.attackRoute, census: d.census() };
  });

/* -- one duel -------------------------------------------------------------- */

/**
 * Fight `variant` to the death and report the fight.
 *
 * Staging is a single teleport before the clock starts. From then on the player
 * closes with `KeyW`, aims with pointer pixels and mashes attack; the skeleton
 * does whatever its behaviour tree decides, including backing off and circling.
 */
const duel = (variant, opts = {}) =>
  page.evaluate(
    async ({ v, stride, maxFrames, holdRange }) => {
      const d = window.__duel;
      const d2 = window.__d2rim;
      const T = d2.three;

      const enemy = d.pick(v);
      if (enemy === null) return null;
      d.clearField([enemy]);
      d.healPlayer();
      d.releaseAll();

      // The one teleport: stand the skeleton three metres in front, awake.
      const player = d.player();
      const forward = player.forward(new T.Vector3());
      const spot = player.position.clone().addScaledVector(forward, 3);
      enemy.skipSpawn?.();
      enemy.teleport(spot.x, spot.y, spot.z);
      enemy.alert();
      await d2.engine.stepFrames(2, stride);

      const combat = d.combat();
      let swings = 0;
      const dealt = [];
      const taken = [];
      const off = d2.ctx.events.on('combat:swing', (p) => {
        if (p.attacker === combat.self.id) swings++;
      });
      const offHit = d2.ctx.events.on('combat:hit', (p) => {
        if (p.attacker === combat.self.id && p.target === enemy.id) {
          dealt.push(+p.outcome.total.toFixed(1));
        } else if (p.target === combat.self.id) {
          taken.push(+p.outcome.total.toFixed(1));
        }
      });

      const startHealth = enemy.health;
      const playerStart = combat.vitals.health.value;
      let frames = 0;
      let holding = false;
      let aimErrorSum = 0;
      let closest = Infinity;

      while (enemy.alive && combat.vitals.health.value > 0 && frames < maxFrames) {
        const target = d.chest(enemy);
        aimErrorSum += Math.abs(d.aimAt(target, stride));

        const gap = d.planarDistance(player.position, enemy.position);
        closest = Math.min(closest, gap);
        // Close, then hold. A real player does not walk through the enemy.
        const wantForward = gap > holdRange;
        if (wantForward !== holding) {
          d.key('KeyW', wantForward);
          holding = wantForward;
        }

        // Mash. The combo machine buffers a press for 0.38 s and refuses one
        // it cannot use, so pressing every frame is a mashed button, not a
        // multiplier on swing count.
        if (d.attackRoute === 'dom-click') d.clickAttack(0);
        else combat.press('light');

        await d2.engine.stepFrames(1, stride);
        frames++;
      }
      off();
      offHit();
      d.releaseAll();

      const seconds = frames * stride;
      const sum = (xs) => xs.reduce((a, x) => a + x, 0);
      const out = {
        variant: v,
        maxHealth: +startHealth.toFixed(1),
        killed: !enemy.alive,
        enemyHealthLeft: +Math.max(0, enemy.health).toFixed(1),
        removed: +(startHealth - Math.max(0, enemy.health)).toFixed(1),
        seconds: +seconds.toFixed(2),
        swings,
        landed: dealt.length,
        accuracy: swings === 0 ? 0 : +(dealt.length / swings).toFixed(2),
        meanHit: dealt.length === 0 ? 0 : +(sum(dealt) / dealt.length).toFixed(1),
        playerStart: +playerStart.toFixed(1),
        playerLeft: +Math.max(0, combat.vitals.health.value).toFixed(1),
        playerDied: combat.vitals.health.value <= 0,
        hitsTaken: taken.length,
        damageTaken: +sum(taken).toFixed(1),
        meanAimError: +((aimErrorSum / Math.max(1, frames)) * (180 / Math.PI)).toFixed(2),
        closestApproach: +closest.toFixed(2),
        timedOut: frames >= maxFrames,
      };
      out.census = d.census();
      return out;
    },
    { v: variant, stride: opts.stride ?? STRIDE, maxFrames: opts.maxFrames ?? 480, holdRange: opts.holdRange ?? 1.2 },
  );

/* -- a pack fight, fought for real in both directions ---------------------- */

/**
 * Three skeletons at once, with the player fighting back.
 *
 * The player is given exactly one piece of footwork, because a pack fight that
 * is just "stand in the middle and mash" is not a test of whether the encounter
 * is winnable *with skill* — it is a test of whether it is winnable without
 * any. The rule is the first thing anyone learns fighting more than one enemy:
 * **do not let them surround you.** When two or more are inside `crowdRange`
 * *and* the player is hurt, back off (`KeyS`) until they string out, then
 * re-engage the nearest. Everything else is the same aim-and-mash as a duel.
 *
 * The health condition is load-bearing and was added after a run without it.
 * Retreating on crowding alone made the player back away for a third of the
 * fight at full health, which is not skill, it is stalling: he took 28 damage
 * in thirty seconds, killed two of three, and the measurement expired with the
 * third skeleton alive and the player *above* his starting health. A stalemate
 * answers neither "can he win" nor "is it dangerous". A player who gives ground
 * only when he is actually losing it produces a fight with an outcome.
 */
const packFight = (count, opts = {}) =>
  page.evaluate(
    async ({ n, stride, maxFrames, holdRange, crowdRange }) => {
      const d = window.__duel;
      const d2 = window.__d2rim;
      const T = d2.three;

      const pack = d.director().enemies.filter((e) => e.alive).slice(0, n);
      if (pack.length < n) return null;
      d.clearField(pack);
      d.healPlayer();
      d.releaseAll();

      // One teleport: an arc in front of the player, four metres out, awake.
      const player = d.player();
      const anchor = player.position.clone();
      const facing = Math.atan2(player.forward(new T.Vector3()).x, player.forward(new T.Vector3()).z);
      for (let i = 0; i < pack.length; i++) {
        const a = facing + (i - (n - 1) / 2) * 0.5;
        pack[i].skipSpawn?.();
        pack[i].teleport(anchor.x + Math.sin(a) * 4, anchor.y, anchor.z + Math.cos(a) * 4);
        pack[i].alert();
      }
      await d2.engine.stepFrames(2, stride);

      const combat = d.combat();
      let swings = 0;
      let landed = 0;
      let taken = 0;
      let damageTaken = 0;
      let retreatFrames = 0;
      const off = d2.ctx.events.on('combat:swing', (p) => {
        if (p.attacker === combat.self.id) swings++;
      });
      const offHit = d2.ctx.events.on('combat:hit', (p) => {
        if (p.attacker === combat.self.id) landed++;
        else if (p.target === combat.self.id) {
          taken++;
          damageTaken += p.outcome.total;
        }
      });

      const startHealth = combat.vitals.health.value;
      let frames = 0;
      let forward = false;
      let back = false;
      const kills = [];
      const offDeath = d2.ctx.events.on('combat:death', (p) => {
        if (p.faction !== 'player') kills.push(p.label);
      });

      while (frames < maxFrames && combat.vitals.health.value > 0) {
        const live = pack.filter((e) => e.alive);
        if (live.length === 0) break;

        // Nearest live target.
        let nearest = live[0];
        let best = Infinity;
        let crowd = 0;
        for (const e of live) {
          const gap = d.planarDistance(player.position, e.position);
          if (gap < crowdRange) crowd++;
          if (gap < best) {
            best = gap;
            nearest = e;
          }
        }

        d.aimAt(d.chest(nearest), stride);

        const hurt = combat.vitals.health.value < startHealth * 0.65;
        const wantBack = crowd >= 2 && hurt;
        const wantForward = !wantBack && best > holdRange;
        if (wantBack !== back) {
          d.key('KeyS', wantBack);
          back = wantBack;
        }
        if (wantForward !== forward) {
          d.key('KeyW', wantForward);
          forward = wantForward;
        }
        if (wantBack) retreatFrames++;

        if (d.attackRoute === 'dom-click') d.clickAttack(0);
        else combat.press('light');

        await d2.engine.stepFrames(1, stride);
        frames++;
      }
      off();
      offHit();
      offDeath();
      d.releaseAll();

      const alive = pack.filter((e) => e.alive).length;
      return {
        count: n,
        won: alive === 0 && combat.vitals.health.value > 0,
        playerDied: combat.vitals.health.value <= 0,
        enemiesLeft: alive,
        kills: kills.length,
        seconds: +(frames * stride).toFixed(2),
        swings,
        landed,
        hitsTaken: taken,
        damageTaken: +damageTaken.toFixed(1),
        playerStart: +startHealth.toFixed(1),
        playerLeft: +Math.max(0, combat.vitals.health.value).toFixed(1),
        retreatFraction: +(retreatFrames / Math.max(1, frames)).toFixed(2),
        timedOut: frames >= maxFrames,
      };
    },
    {
      n: count,
      stride: opts.stride ?? STRIDE,
      maxFrames: opts.maxFrames ?? 1500,
      holdRange: opts.holdRange ?? 1.2,
      crowdRange: opts.crowdRange ?? 2.4,
    },
  );

const VARIANTS = ['minion', 'warrior', 'rogue', 'mage'];

/** Fight every variant once, on the roster this boot came up with. */
const duelAll = async (prefix) => {
  for (const variant of VARIANTS) {
    const r = await duel(variant);
    results[`${prefix}.${variant}`] = r;
    if (r === null) {
      check(`${prefix}: a ${variant} is available to fight`, false, 'none on the roster');
      continue;
    }
    console.log(`  ${variant}: ${JSON.stringify(r)}`);
  }
  if (JSON_OUT !== null) writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));
};

/* -- 1. a starting Barbarian, one on one ----------------------------------- */

results.boot1 = await boot('phase 1 — duels, starting kit');
await duelAll('duel');

/* -- 2. the same fights with the best weapon the drop table will give ------ */

/**
 * Equip the best magic weapon the drop table will produce, and report what the
 * swing now carries.
 *
 * *Best*, not first. Run 2 took the first magic weapon that dropped — a Short
 * Sword of the Fox, whose only affix was `+1 dexterity` — and equipping it moved
 * the swing's physical damage from 10-19 to 9-20. That is a real number and it
 * proves the plumbing, but it is a *worse* weapon than the starting kit on the
 * low end, so it cannot show that gear changes offence in practice. Rolling a
 * few hundred drops and keeping the one with the highest resulting damage is
 * what a player does over an hour of play, compressed.
 */
const equipMagicWeapon = () =>
  page.evaluate(() => {
    const d = window.__duel;
    const rpg = d.rpg();
    if (rpg === undefined) return null;
    // Items come out of the loot system's own drop table — the path the game
    // itself uses — rather than by importing the generator, which is not
    // reachable by name from a production bundle.
    const loot = window.__d2rim.ctx.services.get('rpg.loot');
    const before = d.combat().playerOffense();
    // Remember what the character started the game holding, so the pack fight
    // afterwards is fought with the starting kit and not with whatever this
    // function happened to find. Stashed as a live reference, because
    // `Character.equip` takes the item itself and does not require it to be in
    // the bag — and the bag is about to be used as a scratch space.
    const starting = rpg.character.equipment.get('weapon') ?? null;
    d.startingWeapon = starting;

    /*
     * Empty the bag after every attempt.
     *
     * `Character.equip` puts whatever it displaced back into the inventory, and
     * **refuses the equip entirely** if the displaced item will not fit — losing
     * an item to a full bag is the one failure a player never forgives, so the
     * whole operation is rolled back. Six hundred rolls would fill a ten-by-four
     * grid long before the loop ended, and every equip after that point would
     * silently return `occupied`. The items being dropped here are scratch, not
     * the player's; the two that matter are held as references above and below.
     */
    const drain = () => {
      for (const item of [...rpg.character.inventory.items]) rpg.character.inventory.remove(item);
    };

    let best = null;
    let bestMax = before.damage.physical.max;
    for (let seed = 1; seed < 600; seed++) {
      const rolled = loot.onEnemyDeath(4000 + seed, 'warrior#0');
      for (const item of rolled.items) {
        if (item.slot !== 'weapon' || item.quality === 'normal') continue;
        if (!rpg.character.canEquip(item)) continue;
        if (!rpg.character.equip(item).equipped) continue;
        rpg.character.touch();
        const candidate = d.combat().playerOffense().damage.physical.max;
        if (candidate > bestMax) {
          bestMax = candidate;
          best = item;
        } else {
          const back = best ?? starting;
          if (back !== null) rpg.character.equip(back);
        }
        drain();
      }
    }
    if (best !== null) rpg.character.equip(best);
    drain();
    loot.clear();
    rpg.character.touch();
    const after = d.combat().playerOffense();
    return {
      equipped:
        best === null ? null : { name: best.name, quality: best.quality, mods: best.mods },
      startingWeapon: starting === null ? null : starting.name,
      before: { min: before.damage.physical.min, max: before.damage.physical.max, ar: before.attackRating },
      after: { min: after.damage.physical.min, max: after.damage.physical.max, ar: after.attackRating },
    };
  });

results.boot2 = await boot('phase 2 — duels, best rolled weapon');
const gear = await equipMagicWeapon();
results.gear = gear;
console.log(`  gear: ${JSON.stringify(gear)}`);
await duelAll('magic');

/* -- 3. a three-skeleton pack, starting gear ------------------------------- */

// A fresh boot rather than an unequip: the character comes back level 1 with
// the Hand Axe already on, which is exactly the state this fight is asking
// about — can a *new* player survive a pack.
results.boot3 = await boot('phase 3 — three at once, starting kit');
const pack = await packFight(3);
results['pack.base'] = pack;
console.log(`  pack: ${JSON.stringify(pack)}`);

if (JSON_OUT !== null) writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));

await browser.close();
server.kill('SIGTERM');

/* -- the budget ------------------------------------------------------------ */

/*
 * Asserted rather than reported, which is the whole point of this file: the
 * project spent its entire life able to say how fast the player *dies* and
 * unable to say whether he can win. These bounds are wide — this is a fight
 * against a moving, retreating enemy, and swing counts scatter — but they are
 * bounds. A regression that makes the axe stop connecting, or that turns a
 * skeleton into a health sponge, now fails a check instead of being noticed by
 * whoever next plays the game.
 *
 * | matchup            | measured (starting gear) | asserted   |
 * |--------------------|--------------------------|------------|
 * | minion             | see run output           | <= 12 s    |
 * | warrior            |                          | <= 20 s    |
 * | rogue              |                          | <= 16 s    |
 * | mage               |                          | <= 14 s    |
 * | three at once      |                          | won, and not trivially |
 */

console.log('\n-- verdict --');

const TTK_BUDGET = { minion: 12, warrior: 20, rogue: 16, mage: 14 };
/*
 * The floor is 0.5 s and not something respectable-looking like 2 s, because a
 * *time* floor is the wrong instrument for "the fight has shape". Crits are a
 * 9% roll at double damage; two of them in a row put a 46-point minion down in
 * a second, and that is the damage table working, not a bug. The claim worth
 * asserting is the one below it — that a skeleton takes **at least two
 * connecting blows** — which says the same thing about shape and does not
 * depend on which way the dice fell. The floor here only catches a nonsense
 * zero-length result.
 */
const LOWER = 0.5;

for (const variant of VARIANTS) {
  const r = results[`duel.${variant}`];
  const budget = TTK_BUDGET[variant];
  check(
    `a Barbarian kills a skeleton ${variant} in ${LOWER}-${budget} s of real fighting`,
    r !== null && r !== undefined && r.killed === true && r.seconds <= budget && r.seconds >= LOWER,
    r === null || r === undefined
      ? 'no fight'
      : `${r.seconds}s, ${r.swings} swings, ${r.landed} landed, ` +
        `${r.playerLeft}/${r.playerStart} hp left`,
  );
  check(
    `and lands ${variant} kills in 2-9 connecting blows`,
    r !== null && r !== undefined && r.landed >= 2 && r.landed <= 9,
    `${r?.landed} landed for ${r?.maxHealth} hp, mean ${r?.meanHit}`,
  );
}

check(
  'the scripted aim actually tracks the target',
  VARIANTS.every((v) => (results[`duel.${v}`]?.meanAimError ?? 99) < 5),
  VARIANTS.map((v) => `${v} ${results[`duel.${v}`]?.meanAimError}°`).join(', '),
);

check(
  'a magic weapon was rolled and equipped',
  gear !== null && gear.equipped !== null,
  gear?.equipped === null || gear === null
    ? 'no magic weapon dropped in 900 rolls'
    : `${gear.equipped.quality} ${gear.equipped.name} ${JSON.stringify(gear.equipped.mods)}`,
);
check(
  'and it raised the physical damage the swing actually carries',
  gear !== null && gear.after.max > gear.before.max,
  `${gear?.before.min}-${gear?.before.max} -> ${gear?.after.min}-${gear?.after.max}`,
);

/*
 * Offence in practice, and a note on what is *not* asserted here.
 *
 * `gear.before` / `gear.after` above are read from `CombatSystem.playerOffense()`
 * — the live object the swing resolves against, not the character screen — so
 * "the weapon changed the offence the running game uses" is asserted directly.
 *
 * What is only *reported* is the mean damage of a connecting blow across the
 * duels, and the reason is sample size rather than doubt. A duel lands three or
 * four blows; criticals are a 9% roll at double damage. One crit moves a
 * four-sample mean by more than the entire difference between a Hand Axe and a
 * War Axe, so asserting an inequality on it would be asserting a coin flip.
 * What *is* asserted is the robust consequence: with the better weapon on, all
 * four skeletons still die.
 */
const meanOf = (prefix) => {
  const xs = VARIANTS.map((v) => results[`${prefix}.${v}`]).filter(
    (r) => r !== null && r !== undefined && r.landed > 0,
  );
  if (xs.length === 0) return 0;
  return +(xs.reduce((a, r) => a + r.meanHit, 0) / xs.length).toFixed(1);
};
const baseMean = meanOf('duel');
const magicMean = meanOf('magic');
check(
  'the Barbarian still kills all four variants with the rolled weapon on',
  VARIANTS.every((v) => results[`magic.${v}`]?.killed === true),
  VARIANTS.map(
    (v) => `${v} ${results[`magic.${v}`]?.killed === true ? results[`magic.${v}`].seconds + 's' : 'NOT KILLED'}`,
  ).join(', '),
);
console.log(
  `  mean damage per landed blow: ${baseMean} (starting kit) -> ${magicMean} (rolled weapon) ` +
    `— reported, not asserted; see the note above this check`,
);
check(
  'every phase started from a full six-skeleton roster',
  [results.boot1, results.boot2, results.boot3].every(
    (b) => Object.values(b?.census ?? {}).reduce((a, x) => a + x, 0) === 6,
  ),
  [results.boot1, results.boot2, results.boot3]
    .map((b) => JSON.stringify(b?.census ?? null))
    .join(' '),
);

check(
  'a player can win a three-skeleton pack fight',
  pack !== null && pack.won === true,
  pack === null
    ? 'not enough skeletons'
    : `${pack.won ? 'won' : pack.playerDied ? 'DIED' : 'timed out'} in ${pack.seconds}s, ` +
      `${pack.playerLeft}/${pack.playerStart} hp left, ${pack.kills} killed, ` +
      `${pack.damageTaken} damage taken`,
);
check(
  'and the pack is a real threat rather than a queue',
  pack !== null && pack.damageTaken >= 30 && pack.playerLeft < pack.playerStart * 0.9,
  `${pack?.damageTaken} damage taken, ${pack?.playerLeft}/${pack?.playerStart} hp left`,
);
check(
  'a pack costs the player more than a single skeleton',
  pack !== null &&
    results['duel.warrior'] !== null &&
    pack.damageTaken > (results['duel.minion']?.damageTaken ?? 0),
  `${results['duel.minion']?.damageTaken} (one minion) vs ${pack?.damageTaken} (three)`,
);

if (JSON_OUT !== null) writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));

console.log('\n-- table (starting gear) --');
console.log('  variant  hp   swings  landed  mean   seconds  hp left');
for (const variant of VARIANTS) {
  const r = results[`duel.${variant}`];
  if (r === null || r === undefined) continue;
  console.log(
    `  ${variant.padEnd(8)} ${String(r.maxHealth).padEnd(4)} ${String(r.swings).padEnd(7)} ` +
      `${String(r.landed).padEnd(7)} ${String(r.meanHit).padEnd(6)} ${String(r.seconds).padEnd(8)} ` +
      `${r.playerLeft}/${r.playerStart}`,
  );
}

console.log(
  failures.length === 0 ? '\nALL PASS' : `\n${failures.length} FAILED: ${failures.join(', ')}`,
);
process.exit(failures.length === 0 ? 0 : 1);
