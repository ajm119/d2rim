/**
 * The kill leg of the combat drive, on its own. Not part of the build.
 *
 * `tools/combat-drive.mjs` takes about ten minutes in this software rasteriser,
 * which is far too slow to iterate on one assertion. This runs only the part
 * that matters for "can the player actually finish a skeleton": stage a duel,
 * bring the target to roughly one swing of health, mash the light attack, and
 * assert it dies from a landed hit.
 *
 *   node tools/combat-kill-check.mjs
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5202', '--strictPort', '--host', '127.0.0.1'],
  { stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((resolve) => {
  server.stdout.on('data', (c) => {
    if (String(c).includes('Local:')) resolve();
  });
  setTimeout(resolve, 10000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: 384, height: 216 } });
const logs = [];
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
await page.goto('http://127.0.0.1:5202/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);

const result = await page.evaluate(async () => {
  const d2 = window.__d2rim;
  const T = d2.three;
  const s = d2.ctx.services;
  const combat = s.get('combat');
  const director = s.get('ai.director');
  const player = s.get('character.player');

  d2.engine.getModule('combat.feedback').setHitStop(false);
  await d2.engine.stepFrames(12);

  const enemy = director.enemies.find((e) => e.alive);
  if (enemy === undefined) return { error: 'no enemy' };
  const forward = player.forward(new T.Vector3());
  const spot = player.position.clone().addScaledVector(forward, 1.5);
  enemy.skipSpawn();
  enemy.teleport(spot.x, spot.y, spot.z);
  enemy.alert();
  for (const other of director.enemies) {
    if (other !== enemy) other.teleport(other.position.x + 120, other.position.y, other.position.z + 120);
  }

  enemy.vitals.applyDamage(Math.max(0, enemy.health - 9));
  const before = enemy.health;
  const staminaBefore = combat.stamina;
  const swings = [];
  const off = d2.ctx.events.on('combat:swing', (p) => swings.push(p.moveId));

  for (let i = 0; i < 150; i++) {
    combat.press('light');
    await d2.engine.stepFrames(1);
    if (!enemy.alive) {
      off();
      return {
        before,
        after: enemy.health,
        state: enemy.state,
        frames: i,
        swings,
        staminaBefore: +staminaBefore.toFixed(1),
        staminaAfter: +combat.stamina.toFixed(1),
      };
    }
  }
  off();
  return {
    before,
    after: enemy.health,
    state: enemy.state,
    frames: -1,
    swings,
    staminaBefore: +staminaBefore.toFixed(1),
    staminaAfter: +combat.stamina.toFixed(1),
  };
});

console.log('kill:', JSON.stringify(result));
console.log('page errors:', logs.length === 0 ? 'none' : logs.join(' | '));
const ok = result.state === 'dead' && result.frames > 0 && logs.length === 0;
console.log(ok ? 'PASS the skeleton died from a landed hit' : 'FAIL');
await browser.close();
server.kill();
process.exit(ok ? 0 : 1);
