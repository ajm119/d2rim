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
 *   - The same fight with a magic weapon equipped, to show gear moves offence
 *     in practice and not only on the character sheet.
 *   - A three-skeleton pack fight, fought for real in both directions.
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
 * nobody can afford to run is a measurement nobody runs. `--stride60` re-runs
 * one duel at a true 60 Hz as a control.
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
console.log(`loading ${url}`);
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

/* -- the driver, installed in the page ------------------------------------- */

await page.evaluate(
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

      reviveAll() {
        for (const e of this.director().enemies) e.vitals?.revive?.();
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

const route = await page.evaluate(async () => {
  const d = window.__duel;
  const d2 = window.__d2rim;
  let swings = 0;
  const off = d2.ctx.events.on('combat:swing', () => swings++);
  d.clickAttack(0);
  await d2.engine.stepFrames(2, 1 / 60);
  off();
  d.attackRoute = swings > 0 ? 'dom-click' : 'combat.press';
  return { swings, route: d.attackRoute };
});
console.log(`  attack route: ${route.route} (probe fired ${route.swings} swing(s))`);
results.attackRoute = route;

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

      d.reviveAll();
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
      return {
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
    },
    { v: variant, stride: opts.stride ?? STRIDE, maxFrames: opts.maxFrames ?? 480, holdRange: opts.holdRange ?? 1.5 },
  );

/* -- 1. bare fists of a starting Barbarian, one on one --------------------- */

const VARIANTS = ['minion', 'warrior', 'rogue', 'mage'];
console.log('\n-- duels, starting gear --');
for (const variant of VARIANTS) {
  const r = await duel(variant);
  results[`duel.${variant}`] = r;
  if (r === null) {
    check(`a ${variant} is available to fight`, false, 'no full-health target of that variant');
    continue;
  }
  console.log(`  ${variant}: ${JSON.stringify(r)}`);
}

if (JSON_OUT !== null) writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));

/* -- 2. the same fight with a magic weapon --------------------------------- */

/** Equip a rolled magic weapon and report what the swing now carries. */
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
    let equipped = null;
    for (let seed = 1; seed < 900 && equipped === null; seed++) {
      const rolled = loot.onEnemyDeath(4000 + seed, 'warrior#0');
      for (const item of rolled.items) {
        if (item.slot !== 'weapon' || item.quality === 'normal') continue;
        if (!rpg.character.canEquip(item)) continue;
        rpg.character.acquire(item);
        if (!rpg.character.equip(item).equipped) continue;
        equipped = { name: item.name, quality: item.quality, mods: item.mods };
        break;
      }
    }
    loot.clear();
    rpg.character.touch();
    const after = d.combat().playerOffense();
    return {
      equipped,
      before: { min: before.damage.physical.min, max: before.damage.physical.max, ar: before.attackRating },
      after: { min: after.damage.physical.min, max: after.damage.physical.max, ar: after.attackRating },
    };
  });

console.log('\n-- duels, magic weapon --');
const gear = await equipMagicWeapon();
results.gear = gear;
console.log(`  gear: ${JSON.stringify(gear)}`);

for (const variant of VARIANTS) {
  const r = await duel(variant);
  results[`magic.${variant}`] = r;
  if (r === null) {
    check(`a ${variant} is available for the magic-weapon duel`, false, 'none left');
    continue;
  }
  console.log(`  ${variant} (magic): ${JSON.stringify(r)}`);
}

if (JSON_OUT !== null) writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));

await browser.close();
server.kill('SIGTERM');
console.log(JSON.stringify(results, null, 2));
process.exit(failures.length === 0 ? 0 : 1);
