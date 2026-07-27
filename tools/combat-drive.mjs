/**
 * Scripted combat encounter, driven against the real built game. Not part of
 * the build.
 *
 * Boots `dist/` under `vite preview` with `?autostart=0`, teleports one
 * skeleton into the Barbarian's face, drives the real combo machine through
 * `combat.press`, and asserts that the enemy takes damage and dies. Frames are
 * read back around the impact so the swing pose, the sparks and the hit flash
 * can be looked at rather than assumed.
 *
 * Every stepping loop runs *inside* the page: this container rasterises in
 * software, and a Playwright round trip per frame costs more than the frame.
 *
 *   node tools/combat-drive.mjs [outDir]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import sharp from 'sharp';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
/** `--quick` stops after the impact shots: enough to re-check the look. */
const QUICK = process.argv.includes('--quick');
const OUT = args[0] ?? '/tmp/combat';
const WIDTH = 512;
const HEIGHT = 288;
mkdirSync(OUT, { recursive: true });

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5201', '--strictPort', '--host', '127.0.0.1'],
  { stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((resolve) => {
  server.stdout.on('data', (chunk) => {
    if (String(chunk).includes('Local:')) resolve();
  });
  setTimeout(resolve, 10000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

const started = Date.now();
const mark = (label) => console.log(`[${((Date.now() - started) / 1000).toFixed(1)}s] ${label}`);

await page.goto('http://127.0.0.1:5201/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
mark('ready');

/**
 * Helpers installed in the page. `snapshot` is the state dump every check reads
 * and `stageDuel` is the deterministic setup: one skeleton, at a known offset,
 * already aware, with everything else moved out of the moor.
 */
await page.evaluate(() => {
  const d2 = window.__d2rim;
  const T = d2.three;
  const services = () => d2.ctx.services;

  window.__combat = {
    snapshot() {
      const combat = services().get('combat');
      const director = services().get('ai.director');
      const player = services().get('character.player');
      const feedback = d2.engine.getModule('combat.feedback');
      return {
        timeScale: d2.ctx.time.scale,
        move: combat.moveId,
        window: combat.hitWindowOpen,
        chainIndex: combat.combo.chainIndex,
        playerHealth: +combat.vitals.health.value.toFixed(1),
        playerStamina: +combat.stamina.toFixed(1),
        playerActions: (player.animation?.activeActions ?? []).map(
          (a) => `${a.action}:${a.clip}@${a.normalizedTime.toFixed(2)}`,
        ),
        trauma: +(feedback?.trauma ?? 0).toFixed(3),
        sparks: feedback?.liveSparks ?? 0,
        shake: +(feedback?.shakeOffset.length() ?? 0).toFixed(4),
        enemies: director.enemies.map((e) => ({
          id: e.id,
          v: e.profile.variant,
          st: e.state,
          hp: +e.health.toFixed(1),
          d: +e.position.distanceTo(player.position).toFixed(2),
          aware: e.aware,
          clips: (e.animationDebug ?? []).slice(0, 2),
        })),
      };
    },

    /** Put one live skeleton `distance` metres in front of the hero. */
    stageDuel(distance) {
      const director = services().get('ai.director');
      const player = services().get('character.player');
      const enemy = director.enemies.find((e) => e.alive);
      if (enemy === undefined) return null;
      const forward = player.forward(new T.Vector3());
      const spot = player.position.clone().addScaledVector(forward, distance);
      enemy.skipSpawn();
      enemy.teleport(spot.x, spot.y, spot.z);
      enemy.alert();
      for (const other of director.enemies) {
        if (other === enemy) continue;
        other.teleport(other.position.x + 120, other.position.y, other.position.z + 120);
      }
      window.__combat.duelId = enemy.id;
      return { id: enemy.id, variant: enemy.profile.variant, hp: enemy.health };
    },

    enemy(id) {
      return services()
        .get('ai.director')
        .enemies.find((e) => e.id === id);
    },

    /** Frame the duel from the side so the swing is not behind the hero. */
    stageCamera(distance = 4.2, height = 1.9, side = 1) {
      const player = services().get('character.player');
      const enemy = window.__combat.enemy(window.__combat.duelId);
      const camera = d2.ctx.camera;
      const forward = player.forward(new T.Vector3());
      const right = new T.Vector3(forward.z, 0, -forward.x).multiplyScalar(side);
      const focus = (enemy?.position ?? player.position)
        .clone()
        .add(player.position)
        .multiplyScalar(0.5);
      focus.y += 1.05;
      camera.position
        .copy(focus)
        .addScaledVector(right, distance * 0.85)
        .addScaledVector(forward, -distance * 0.25);
      camera.position.y = focus.y + height * 0.35;
      camera.lookAt(focus);
      camera.updateMatrixWorld(true);
    },
  };
});

const shot = async (label, cinematic = true) => {
  if (cinematic) await page.evaluate(() => window.__combat.stageCamera());
  const frame = await page.evaluate(
    async ([w, h]) => {
      const ctx = window.__d2rim.ctx;
      const s = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, w, h);
      let binary = '';
      for (let i = 0; i < s.pixels.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, s.pixels.subarray(i, i + 0x8000));
      }
      return { width: s.width, height: s.height, data: btoa(binary) };
    },
    [WIDTH, HEIGHT],
  );
  await sharp(Buffer.from(frame.data, 'base64'), {
    raw: { width: frame.width, height: frame.height, channels: 4 },
  })
    .png()
    .toFile(`${OUT}/${label}.png`);
  mark(`shot ${label}`);
};

const failures = [];
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail === undefined ? '' : ` — ${detail}`}`);
  if (!ok) failures.push(name);
};

/* -- settle -------------------------------------------------------------- */

await page.evaluate(() => window.__d2rim.engine.stepFrames(20));
let snap = await page.evaluate(() => window.__combat.snapshot());
console.log('spawned:', JSON.stringify(snap.enemies));
check('skeletons spawned', snap.enemies.length >= 4, `${snap.enemies.length} enemies`);
check('the player starts alive', snap.playerHealth > 0, `${snap.playerHealth} hp`);

const staged = await page.evaluate(() => window.__combat.stageDuel(1.5));
check('staged a duel', staged !== null, JSON.stringify(staged));
await page.evaluate(() => window.__d2rim.engine.stepFrames(2));
await shot('01-face-off');

/* -- one swing ------------------------------------------------------------ */

const swing = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const combat = d2.ctx.services.get('combat');
  const enemy = window.__combat.enemy(window.__combat.duelId);
  const before = enemy.health;

  combat.press('light');
  const trace = [];
  let sawWindow = false;
  let peakSparks = 0;
  let minScale = 1;
  let impactFrame = -1;
  let peakTrauma = 0;

  for (let i = 0; i < 120; i++) {
    await d2.engine.stepFrames(1);
    const feedback = d2.engine.getModule('combat.feedback');
    if (combat.hitWindowOpen) sawWindow = true;
    peakSparks = Math.max(peakSparks, feedback?.liveSparks ?? 0);
    peakTrauma = Math.max(peakTrauma, feedback?.trauma ?? 0);
    minScale = Math.min(minScale, d2.ctx.time.scale);
    if (i % 4 === 0) {
      trace.push(`${i}:${combat.moveId ?? '-'}${combat.hitWindowOpen ? '*' : ''}/${enemy.health}`);
    }
    if (impactFrame === -1 && enemy.health < before) impactFrame = i;
    // Keep sampling for a few frames past the impact: the burst is emitted from
    // `combat:hit` during lateUpdate, and the particle count is only recomputed
    // by the next frame's update, so breaking on the impact frame measures the
    // state one frame before the sparks exist.
    if (impactFrame !== -1 && i >= impactFrame + 4) break;
  }
  return {
    before,
    after: enemy.health,
    sawWindow,
    peakSparks,
    peakTrauma: +peakTrauma.toFixed(3),
    minScale: +minScale.toFixed(3),
    impactFrame,
    trace,
    enemyState: enemy.state,
  };
});
console.log('swing:', JSON.stringify(swing));
check('the authored damage window opened', swing.sawWindow);
check('the swing dealt damage', swing.after < swing.before, `${swing.before} -> ${swing.after}`);
check('impact sparks were emitted', swing.peakSparks > 0, `${swing.peakSparks} particles`);
check('hit stop slowed the clock', swing.minScale < 0.5, `min time scale ${swing.minScale}`);
check('camera trauma accumulated', swing.peakTrauma > 0, `peak trauma ${swing.peakTrauma}`);
await shot('02-impact');

// A couple of frames on so the sparks have spread and the shake has moved.
await page.evaluate(() => window.__d2rim.engine.stepFrames(3));
snap = await page.evaluate(() => window.__combat.snapshot());
console.log('after impact:', JSON.stringify(snap));
check('the swing is playing a real attack clip', snap.playerActions.length > 0, snap.playerActions.join(','));
await shot('03-after-impact');

if (QUICK) {
  writeFileSync(`${OUT}/console.log`, logs.join('\n'));
  console.log(failures.length === 0 ? '\nALL CHECKS PASSED' : `\nFAILED: ${failures.join(', ')}`);
  await browser.close();
  server.kill();
  process.exit(failures.length === 0 ? 0 : 1);
}

/* -- combo chain ---------------------------------------------------------- */

// Hit stop has been verified; from here on it only costs frames. A 16x
// slowdown on every impact means sixteen times the renders to reach the same
// simulated state, and this container rasterises in software.
await page.evaluate(() => window.__d2rim.engine.getModule('combat.feedback').setHitStop(false));

const chain = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const combat = d2.ctx.services.get('combat');
  const seen = [];
  const clips = [];
  const off = d2.ctx.events.on('combat:swing', (p) => seen.push(p.moveId));
  const player = d2.ctx.services.get('character.player');
  for (let i = 0; i < 200; i++) {
    combat.press('light');
    await d2.engine.stepFrames(1);
    for (const action of player.animation?.activeActions ?? []) {
      if (action.action.startsWith('attack') && !clips.includes(action.clip)) clips.push(action.clip);
    }
    if (seen.length >= 5) break;
  }
  off();
  return { seen, clips };
});
console.log('chain:', chain.seen.join(' -> '));
console.log('clips used:', chain.clips.join(', '));
check(
  'the combo chained through distinct moves',
  new Set(chain.seen).size >= 3,
  chain.seen.join(','),
);
check(
  'each link plays a different authored clip',
  new Set(chain.clips).size >= 3,
  chain.clips.join(','),
);

/* -- kill ----------------------------------------------------------------- */

const kill = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const combat = d2.ctx.services.get('combat');
  // A *fresh* skeleton: the one from the chain phase is already a corpse, and a
  // kill check that passes because the target was dead before the swing proves
  // nothing at all.
  const staged = window.__combat.stageDuel(1.5);
  if (staged === null) return { error: 'no live enemy left' };
  const enemy = window.__combat.enemy(window.__combat.duelId);
  // Bring it down to roughly one swing's worth so the loop is short; the
  // killing blow itself still goes through the whole hitbox and damage path.
  enemy.vitals.applyDamage(Math.max(0, enemy.health - 9));
  const before = enemy.health;
  for (let i = 0; i < 120; i++) {
    combat.press('light');
    await d2.engine.stepFrames(1);
    if (!enemy.alive) return { staged, before, after: enemy.health, state: enemy.state, frames: i };
  }
  return { staged, before, after: enemy.health, state: enemy.state, frames: -1 };
});
console.log('kill:', JSON.stringify(kill));
check(
  'the killing blow was actually swung',
  kill.before > 0 && kill.frames > 0,
  `started at ${kill.before} hp, died after ${kill.frames} frames`,
);
check('the skeleton died', kill.state === 'dead', JSON.stringify(kill));
await page.evaluate(() => window.__d2rim.engine.stepFrames(20));
await shot('04-death');

/* -- the enemy hits back --------------------------------------------------- */

const revenge = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const combat = d2.ctx.services.get('combat');
  const staged = window.__combat.stageDuel(1.4);
  if (staged === null) return { error: 'no live enemy left' };
  const enemy = window.__combat.enemy(window.__combat.duelId);
  const before = combat.vitals.health.value;
  const states = [];
  let telegraphFrame = -1;
  for (let i = 0; i < 200; i++) {
    await d2.engine.stepFrames(1);
    if (states[states.length - 1] !== enemy.state) states.push(enemy.state);
    if (telegraphFrame === -1 && enemy.state === 'telegraph') telegraphFrame = i;
    if (combat.vitals.health.value < before) {
      return {
        staged,
        before,
        after: combat.vitals.health.value,
        states,
        telegraphFrame,
        hitFrame: i,
        enemyState: enemy.state,
      };
    }
  }
  return { staged, before, after: combat.vitals.health.value, states, telegraphFrame, hitFrame: -1 };
});
console.log('enemy turn:', JSON.stringify(revenge));
check('the enemy telegraphed before swinging', (revenge.states ?? []).includes('telegraph'));
check(
  'the enemy damaged the player',
  revenge.after !== undefined && revenge.after < revenge.before,
  `${revenge.before} -> ${revenge.after}`,
);
check(
  'the telegraph gave the player time to react',
  revenge.hitFrame - revenge.telegraphFrame >= 12,
  `${revenge.hitFrame - revenge.telegraphFrame} frames of warning`,
);
await shot('05-enemy-attack');

writeFileSync(`${OUT}/console.log`, logs.join('\n'));
const errors = logs.filter((line) => line.startsWith('pageerror') || line.startsWith('error'));
check('no page errors', errors.length === 0, errors.slice(0, 4).join(' | '));

console.log(failures.length === 0 ? '\nALL CHECKS PASSED' : `\nFAILED: ${failures.join(', ')}`);
await browser.close();
server.kill();
process.exit(failures.length === 0 ? 0 : 1);
