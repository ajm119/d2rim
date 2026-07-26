/**
 * Headless frame capture for d2rim.
 *
 * Serves the production build, boots the engine in a headless Chromium, steps a
 * deterministic number of frames, and writes a PNG.
 *
 * Two capture routes exist because this container's WebGPU implementation
 * renders correctly but never presents to the compositor — a `page.screenshot()`
 * of a WebGPU canvas is pure black even though the GPU produced right pixels.
 * So:
 *
 *   --backend=webgl2 (default)  page.screenshot(), no workarounds
 *   --backend=webgpu            RendererHandle.captureFrame() readback, blitted
 *                               to a 2D canvas that the compositor *can* see
 *
 * Usage:
 *   node scripts/capture.mjs [--backend=webgl2|webgpu] [--frames=90]
 *                            [--width=960] [--height=540] [--out=path.png]
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/**
 * The pre-installed browser bundle in this container does not match the build
 * number the installed Playwright expects, and `playwright install` must never
 * be run here (no network budget for it, and the bundle is provided). Point
 * Playwright at the real binary instead.
 */
function findChromium() {
  const candidates = [
    process.env.CHROMIUM_PATH,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/opt/pw-browsers/chromium/chrome-linux/chrome',
  ].filter(Boolean);
  const found = candidates.find((path) => existsSync(path));
  if (found === undefined) {
    throw new Error(`no Chromium binary found; tried:\n  ${candidates.join('\n  ')}`);
  }
  return found;
}

/** Verified working in this container; see the GPU probe notes in the README. */
const CHROMIUM_ARGS = [
  '--no-sandbox',
  '--disable-dev-shm-usage',
  '--enable-unsafe-webgpu',
  '--use-gl=angle',
  '--use-angle=swiftshader',
];

function parseArgs(argv) {
  const options = {
    backend: 'webgl2',
    frames: 90,
    width: 960,
    height: 540,
    out: null,
    port: 4173,
  };
  for (const arg of argv) {
    const match = /^--([a-z]+)=(.+)$/.exec(arg);
    if (match === null) continue;
    const [, key, value] = match;
    if (key in options) {
      options[key] = /^\d+$/.test(value) ? Number(value) : value;
    }
  }
  options.out ??= resolve(ROOT, `captures/reference-${options.backend}.png`);
  return options;
}

/** Start `vite preview` and resolve once it is actually accepting connections. */
async function startServer(port) {
  const child = spawn(
    process.execPath,
    [resolve(ROOT, 'node_modules/vite/bin/vite.js'), 'preview', '--port', String(port), '--strictPort'],
    { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );

  const url = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return { child, url };
    } catch {
      // Not up yet.
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  child.kill('SIGKILL');
  throw new Error(`preview server did not start on ${url}`);
}

const options = parseArgs(process.argv.slice(2));
const { child, url } = await startServer(options.port);

let exitCode = 0;
let browser;
try {
  browser = await chromium.launch({ args: CHROMIUM_ARGS, executablePath: findChromium() });
  const page = await browser.newPage({
    viewport: { width: options.width, height: options.height },
    deviceScaleFactor: 1,
  });

  page.on('console', (message) => {
    if (['error', 'warning', 'info'].includes(message.type())) {
      console.log(`  [page:${message.type()}] ${message.text()}`);
    }
  });
  page.on('pageerror', (error) => console.log(`  [page:exception] ${error.message}`));

  // `autostart=0`: the harness is the only thing that advances frames, so the
  // captured state does not depend on how long this script took to get here.
  const target = `${url}/?backend=${options.backend}&autostart=0`;
  console.log(`> ${target}`);
  await page.goto(target, { waitUntil: 'load', timeout: 60_000 });

  // SwiftShader renders this scene at roughly 5-20 fps, so every wait here is
  // generous on purpose. This is not a signal about real-GPU performance.
  await page.waitForFunction(() => window.__d2rim !== undefined, null, { timeout: 60_000 });
  await page.evaluate(() => window.__d2rim.ready, null, { timeout: 120_000 });

  const backend = await page.evaluate(() => window.__d2rim.engine.renderer.backend);
  console.log(`  backend: ${backend}`);
  if (backend !== options.backend) {
    throw new Error(`expected backend "${options.backend}" but got "${backend}"`);
  }

  // Measured in this container: presenting to the WebGPU canvas swapchain not
  // only fails to reach the compositor, it eventually loses the device
  // ("A valid external Instance reference no longer exists"), which then takes
  // the readback down with it. Stepping zero frames renders nothing to the
  // canvas, so `captureFrame` gets a healthy device. This is a platform
  // limitation, not an engine one -- the same build steps and presents fine on
  // WebGL2 here, and on real WebGPU hardware.
  if (backend === 'webgpu' && options.frames > 0) {
    console.log(
      '  ! warning: stepping >0 frames on WebGPU presents to the canvas, which\n' +
        '    loses the device in this headless container. Use --frames=0, or\n' +
        '    capture animated frames with --backend=webgl2.',
    );
  }

  console.log(`  stepping ${options.frames} frames...`);
  const started = Date.now();
  await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), options.frames, {
    timeout: 300_000,
  });
  console.log(`  stepped in ${((Date.now() - started) / 1000).toFixed(1)}s`);

  await mkdir(dirname(options.out), { recursive: true });

  // Readback for both backends, not just WebGPU. Two reasons:
  //
  //  - WebGPU has no choice: its swapchain never reaches the headless
  //    compositor, so `page.screenshot()` returns black.
  //  - WebGL2 *can* be screenshotted, but `render()` only queues GPU work.
  //    Under SwiftShader the CPU queues frames far faster than they retire, so
  //    a screenshot waits on a compositor that is dozens of frames behind and
  //    times out. `readRenderTargetPixelsAsync` is a real sync point: it
  //    resolves only once the GPU has actually finished.
  //
  // Using one route for both also means the captured pixels come from the same
  // code path on every backend, which is what makes cross-backend golden-image
  // comparison meaningful.
  const dataUrl = await page.evaluate(
    async ([width, height]) => {
      const { engine } = window.__d2rim;
      const ctx = engine.context;
      const frame = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, width, height);
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      const c2d = canvas.getContext('2d');
      const image = c2d.createImageData(frame.width, frame.height);
      image.data.set(frame.pixels);
      c2d.putImageData(image, 0, 0);
      return canvas.toDataURL('image/png');
    },
    [options.width, options.height],
    { timeout: 300_000 },
  );
  await writeFile(options.out, Buffer.from(dataUrl.split(',')[1], 'base64'));

  console.log(`  wrote ${options.out}`);
} catch (error) {
  console.error(error);
  exitCode = 1;
} finally {
  await browser?.close();
  child.kill('SIGTERM');
}

process.exit(exitCode);
