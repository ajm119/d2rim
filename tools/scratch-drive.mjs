/**
 * Ad-hoc verification harness for the character work. Not part of the build.
 *
 * Boots the built page with `?autostart=0`, holds real keys, steps the engine a
 * fixed number of frames, reads the frame back and dumps a state snapshot.
 * Everything is deterministic, so consecutive frames are directly comparable.
 *
 * `vite preview` over `dist/`, never the dev server: HMR reloads the page the
 * moment anything under `src/` changes and destroys the execution context
 * mid-run. `?quality=low` because this container rasterises in software and the
 * point here is the character, not the shading.
 *
 *   node tools/scratch-drive.mjs [outDir]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import sharp from 'sharp';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const OUT = process.argv[2] ?? '/tmp/drive';
const WIDTH = 640;
const HEIGHT = 360;
mkdirSync(OUT, { recursive: true });

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', '5199', '--strictPort', '--host', '127.0.0.1'],
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
page.on('console', (message) => logs.push(`${message.type()}: ${message.text()}`));
page.on('pageerror', (error) => logs.push(`pageerror: ${error.message}`));

const started = Date.now();
const mark = (label) => console.log(`[${((Date.now() - started) / 1000).toFixed(1)}s] ${label}`);

await page.goto('http://127.0.0.1:5199/?autostart=0&backend=webgl2&quality=low', {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
mark('ready');

const step = async (n) => {
  await page.evaluate((count) => window.__d2rim.engine.stepFrames(count), n);
  mark(`stepped ${n}`);
};

async function snapshot(label) {
  const state = await page.evaluate(() => {
    const services = window.__d2rim.ctx.services;
    const player = services.get('character.player');
    const graph = player.animation;
    const physics = services.get('physics.world');
    const rig = services.get('character.cameraRig');
    return {
      pos: player.position.toArray().map((v) => +v.toFixed(3)),
      speed: +player.speed.toFixed(3),
      grounded: player.grounded,
      bodyYaw: +player.bodyYaw.toFixed(3),
      state: graph?.state ?? null,
      cycleRate: +(graph?.cycleRate ?? 0).toFixed(3),
      phase: +(graph?.phase ?? 0).toFixed(3),
      weights: graph
        ? Object.fromEntries(
            [...graph.weights].filter(([, w]) => w > 0.002).map(([k, w]) => [k, +w.toFixed(3)]),
          )
        : null,
      strides: graph
        ? Object.fromEntries([...graph.strideTable].map(([k, v]) => [k, +v.toFixed(3)]))
        : null,
      stamina: +player.stamina.toFixed(1),
      colliders: physics.colliders.length,
      actions: graph ? graph.activeActions.map((a) => `${a.action}@${a.normalizedTime.toFixed(2)}`) : [],
      camera: { mode: rig.mode, blend: +rig.blend.toFixed(2), arm: +rig.armLength.toFixed(2) },
      // Negative means the camera sits behind the character's facing direction,
      // which is the whole point of an over-the-shoulder rig.
      behindness: +window.__d2rim.ctx.camera.position
        .clone()
        .sub(player.position)
        .setY(0)
        .normalize()
        .dot(player.forward())
        .toFixed(3),
    };
  });
  const frame = await page.evaluate(
    async ([w, h]) => {
      const ctx = window.__d2rim.ctx;
      const shot = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, w, h);
      let binary = '';
      const bytes = shot.pixels;
      for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
      }
      return { width: shot.width, height: shot.height, data: btoa(binary) };
    },
    [WIDTH, HEIGHT],
  );
  await sharp(Buffer.from(frame.data, 'base64'), {
    raw: { width: frame.width, height: frame.height, channels: 4 },
  })
    .png()
    .toFile(`${OUT}/${label}.png`);
  console.log(`--- ${label} ---\n${JSON.stringify(state)}`);
  return state;
}

await step(16);
await snapshot('01-idle');

await page.keyboard.down('w');
await step(30);
await snapshot('02-run');
await step(8);
await snapshot('03-run-later');

await page.keyboard.down('Shift');
await step(24);
await snapshot('04-sprint');
await page.keyboard.up('Shift');
await page.keyboard.up('w');

await page.keyboard.down('d');
await step(30);
await snapshot('05-strafe-right');
await page.keyboard.up('d');

await step(24);
await page.mouse.down({ button: 'left' });
await step(1);
await page.mouse.up({ button: 'left' });
await step(6);
await snapshot('06-attack');

await step(24);
await page.keyboard.press('f');
await step(24);
await snapshot('07-first-person');

writeFileSync(`${OUT}/console.log`, logs.join('\n'));
console.log(`\nlogged ${logs.length} console lines -> ${OUT}/console.log`);

await browser.close();
server.kill();
process.exit(0);
