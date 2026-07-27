/**
 * tools/verify-pixel-ratio.mjs
 *
 * Confirms the tier's pixel-ratio cap actually reaches the drawing buffer, on a
 * simulated Retina display.
 *
 * The cap is the largest single term in a quality tier's cost — `devicePixelRatio`
 * is squared into the fragment count — and it is also the easiest thing in the
 * renderer to *believe* is applied. A unit test can prove `RenderSettings` asks
 * the engine for the right number; only a browser can prove the canvas came back
 * the right size. Playwright's `deviceScaleFactor` gives a real DPR-2 context
 * without Retina hardware.
 *
 * Usage: `node tools/verify-pixel-ratio.mjs [--dpr=2] [--port=5260]`
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { spawn } from 'node:child_process';

import { chromium } from 'playwright';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';

const DPR = Number((process.argv.find((a) => a.startsWith('--dpr=')) ?? '--dpr=2').split('=')[1]);
const PORT = Number(
  (process.argv.find((a) => a.startsWith('--port=')) ?? '--port=5260').split('=')[1],
);
const WIDTH = 1512;
const HEIGHT = 945;

const failures = [];
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail === undefined ? '' : ` — ${detail}`}`);
  if (!ok) failures.push(name);
};

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

/** Boot at a tier and report what the canvas actually became. */
const measure = async (quality) => {
  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: DPR,
  });
  const page = await context.newPage();
  page.setDefaultNavigationTimeout(300_000);
  page.setDefaultTimeout(300_000);
  const logs = [];
  page.on('console', (m) => logs.push(m.text()));

  await page.goto(
    `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&fade=0&enemies=0&stats=1&quality=${quality}`,
    { waitUntil: 'load' },
  );
  await page.evaluate(() => window.__d2rim.ready);
  // The overlay recomposes on a 0.25 s cadence, and two stepped frames are
  // 33 ms of world time — so it would still be holding its seed text. A resize
  // forces the next update to redraw, which costs one frame instead of fifteen.
  await page.evaluate(async () => {
    await window.__d2rim.engine.stepFrames(2);
    window.__d2rim.engine.resize();
    await window.__d2rim.engine.stepFrames(1);
  });

  const read = await page.evaluate(() => {
    const d2 = window.__d2rim;
    const canvas = document.getElementById('viewport');
    const overlay = document.getElementById('d2rim-debug-overlay');
    return {
      devicePixelRatio: window.devicePixelRatio,
      canvasCss: { width: canvas.clientWidth, height: canvas.clientHeight },
      // The drawing buffer: what the GPU actually fills, and the only number
      // in this file that costs anything.
      buffer: { width: canvas.width, height: canvas.height },
      engineCap: d2.engine.pixelRatioCap,
      tierCap: d2.render.settings.tier.pixelRatioCap,
      overlay: overlay?.textContent ?? '',
    };
  });

  await context.close();
  return { read, logs };
};

for (const [quality, expectedCap] of [
  ['low', 1],
  ['medium', 1.5],
  ['high', 2],
]) {
  const { read, logs } = await measure(quality);
  const expectedRatio = Math.min(read.devicePixelRatio, expectedCap);
  const wantW = Math.round(read.canvasCss.width * expectedRatio);
  const wantH = Math.round(read.canvasCss.height * expectedRatio);
  const mpx = (read.buffer.width * read.buffer.height) / 1e6;

  console.log(
    `\n== ${quality} @ DPR ${read.devicePixelRatio} ==\n` +
      `  css ${read.canvasCss.width}x${read.canvasCss.height}, ` +
      `buffer ${read.buffer.width}x${read.buffer.height} (${mpx.toFixed(2)} Mpx)\n` +
      `  tier cap ${read.tierCap}, engine cap ${read.engineCap}`,
  );

  check(`${quality}: tier declares cap ${expectedCap}`, read.tierCap === expectedCap, `${read.tierCap}`);
  check(`${quality}: engine holds the tier's cap`, read.engineCap === expectedCap, `${read.engineCap}`);
  check(
    `${quality}: drawing buffer is ${wantW}x${wantH}`,
    Math.abs(read.buffer.width - wantW) <= 1 && Math.abs(read.buffer.height - wantH) <= 1,
    `${read.buffer.width}x${read.buffer.height}`,
  );
  check(
    `${quality}: overlay reports the drawing buffer and the device ratio`,
    read.overlay.includes('buffer ') && read.overlay.includes('device '),
    read.overlay.split('\n').find((line) => line.startsWith('buffer')) ?? '(no buffer line)',
  );
  const line = logs.find((entry) => entry.includes('pixel ratio'));
  check(`${quality}: boot log states the ratio`, line !== undefined, line);
}

await browser.close();
server.kill('SIGTERM');
if (failures.length > 0) {
  console.error(`\n${failures.length} check(s) failed: ${failures.join(', ')}`);
  process.exit(1);
}
console.log('\nall pixel-ratio checks passed');
process.exit(0);
