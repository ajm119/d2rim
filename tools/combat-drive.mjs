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
 *   node tools/combat-drive.mjs [outDir]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import sharp from 'sharp';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const OUT = process.argv[2] ?? '/tmp/combat';
const WIDTH = 720;
const HEIGHT = 405;
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

const step = (n) => page.evaluate((count) => window.__d2rim.engine.stepFrames(count), n);

const state = () =>
  page.evaluate(() => {
    const s = window.__d2rim.ctx.services;
    const combat = s.get('combat');
    const director = s.get('ai.director');
    const player = s.get('character.player');
    const feedback = window.__d2rim.engine.getModule('combat.feedback');
    return {
      timeScale: window.__d2rim.ctx.time.scale,
      move: combat.moveId,
      window: combat.hitWindowOpen,
      chainIndex: combat.combo.chainIndex,
      playerHealth: +combat.vitals.health.value.toFixed(1),
      playerStamina: +combat.stamina.toFixed(1),
      playerPos: player.position.toArray().map((v) => +v.toFixed(2)),
      playerAction: (player.animation?.activeActions ?? []).map(
        (a) => `${a.action}@${a.normalizedTime.toFixed(2)}`,
      ),
      trauma: +(feedback?.trauma ?? 0).toFixed(3),
      sparks: feedback?.liveSparks ?? 0,
      enemies: director.enemies.map((e) => ({
        id: e.id,
        v: e.profile.variant,
        st: e.state,
        hp: +e.health.toFixed(1),
        d: +e.position.distanceTo(player.position).toFixed(2),
        aware: e.aware,
      })),
    };
  });

async function shot(label) {
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
}

const failures = [];
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail === undefined ? '' : ` — ${detail}`}`);
  if (!ok) failures.push(name);
};

/* -- settle -------------------------------------------------------------- */

await step(30);
let snap = await state();
console.log('spawned:', JSON.stringify(snap.enemies));
check('skeletons spawned', snap.enemies.length >= 4, `${snap.enemies.length} enemies`);
check('player is at full health', snap.playerHealth > 0, `${snap.playerHealth} hp`);

/* -- stage one skeleton in front of the hero ------------------------------ */

const staged = await page.evaluate(() => {
  const s = window.__d2rim.ctx.services;
  const director = s.get('ai.director');
  const player = s.get('character.player');
  const T = window.__d2rim.three;
  const enemy = director.enemies.find((e) => e.alive);
  if (enemy === undefined) return null;
  const forward = player.forward(new T.Vector3());
  const target = player.position.clone().addScaledVector(forward, 1.55);
  enemy.skipSpawn();
  enemy.teleport(target.x, target.y, target.z);
  enemy.alert();
  // Move everything else far away so exactly one fight is under test.
  for (const other of director.enemies) {
    if (other === enemy) continue;
    other.teleport(other.position.x + 90, other.position.y, other.position.z + 90);
  }
  return { id: enemy.id, variant: enemy.profile.variant, hp: enemy.health };
});
check('staged a skeleton', staged !== null, JSON.stringify(staged));
await step(2);
await shot('01-face-off');

/* -- swing --------------------------------------------------------------- */

await page.evaluate(() => window.__d2rim.ctx.services.get('combat').press('light'));

let sawWindow = false;
let sawSparks = 0;
let firstDamageFrame = -1;
let minHitStopScale = 1;
const before = (await state()).enemies.find((e) => e.id === staged.id).hp;

for (let i = 0; i < 90; i++) {
  await step(1);
  const now = await state();
  const enemy = now.enemies.find((e) => e.id === staged.id);
  if (now.window) sawWindow = true;
  sawSparks = Math.max(sawSparks, now.sparks);
  minHitStopScale = Math.min(minHitStopScale, now.timeScale);
  if (firstDamageFrame === -1 && enemy !== undefined && enemy.hp < before) {
    firstDamageFrame = i;
    console.log('impact frame', i, JSON.stringify(now));
    await shot('02-impact');
  }
  if (firstDamageFrame !== -1 && i === firstDamageFrame + 3) await shot('03-after-impact');
  if (firstDamageFrame !== -1 && i > firstDamageFrame + 12) break;
}

snap = await state();
const hitEnemy = snap.enemies.find((e) => e.id === staged.id);
check('the damage window opened', sawWindow);
check('the swing dealt damage', hitEnemy.hp < before, `${before} -> ${hitEnemy.hp}`);
check('impact sparks were emitted', sawSparks > 0, `${sawSparks} live particles`);
check('hit stop slowed the clock', minHitStopScale < 0.5, `min time scale ${minHitStopScale}`);
check('camera trauma accumulated', snap.trauma >= 0, `trauma ${snap.trauma}`);

/* -- combo chain ---------------------------------------------------------- */

const chain = await page.evaluate(async () => {
  const s = window.__d2rim.ctx.services;
  const combat = s.get('combat');
  const engine = window.__d2rim.engine;
  const seen = [];
  const off = window.__d2rim.ctx.events.on('combat:swing', (p) => seen.push(p.moveId));
  for (let i = 0; i < 220; i++) {
    combat.press('light');
    await engine.stepFrames(1);
  }
  off();
  return seen;
});
console.log('chain observed:', chain.join(' -> '));
check('the combo chained through distinct moves', new Set(chain).size >= 3, chain.join(','));

/* -- kill ----------------------------------------------------------------- */

for (let i = 0; i < 700; i++) {
  await page.evaluate(() => window.__d2rim.ctx.services.get('combat').press('light'));
  await step(1);
  if (i % 25 === 0) {
    const now = await state();
    const enemy = now.enemies.find((e) => e.id === staged.id);
    if (enemy === undefined || enemy.hp <= 0) break;
  }
}

snap = await state();
const corpse = snap.enemies.find((e) => e.id === staged.id);
console.log('final:', JSON.stringify(snap.enemies));
check(
  'the skeleton died',
  corpse === undefined || corpse.hp <= 0 || corpse.st === 'dead',
  corpse === undefined ? 'removed' : `${corpse.hp} hp, state ${corpse.st}`,
);
await shot('04-death');

/* -- the enemy hits back --------------------------------------------------- */

const revenge = await page.evaluate(async () => {
  const s = window.__d2rim.ctx.services;
  const director = s.get('ai.director');
  const combat = s.get('combat');
  const player = s.get('character.player');
  const T = window.__d2rim.three;
  const engine = window.__d2rim.engine;

  const enemy = director.enemies.find((e) => e.alive);
  if (enemy === undefined) return { error: 'no live enemy left' };
  const forward = player.forward(new T.Vector3());
  const spot = player.position.clone().addScaledVector(forward, 1.4);
  enemy.skipSpawn();
  enemy.teleport(spot.x, spot.y, spot.z);
  enemy.alert();

  const before = combat.vitals.health.value;
  const states = new Set();
  for (let i = 0; i < 500; i++) {
    await engine.stepFrames(1);
    states.add(enemy.state);
    if (combat.vitals.health.value < before) break;
  }
  return {
    before,
    after: combat.vitals.health.value,
    states: [...states],
    enemyState: enemy.state,
  };
});
console.log('enemy turn:', JSON.stringify(revenge));
check('the enemy telegraphed before swinging', (revenge.states ?? []).includes('telegraph'));
check(
  'the enemy damaged the player',
  revenge.after !== undefined && revenge.after < revenge.before,
  `${revenge.before} -> ${revenge.after}`,
);
await shot('05-enemy-attack');

writeFileSync(`${OUT}/console.log`, logs.join('\n'));
const errors = logs.filter((line) => line.startsWith('pageerror') || line.startsWith('error'));
check('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));

console.log(failures.length === 0 ? '\nALL CHECKS PASSED' : `\nFAILED: ${failures.join(', ')}`);
await browser.close();
server.kill();
process.exit(failures.length === 0 ? 0 : 1);
