/**
 * @module tools/capture/capture
 *
 * Deterministic headless screenshot harness for d2rim.
 *
 * This is the backbone of the visual quality loop: it turns a declarative shot
 * list into a directory of PNGs that a critic agent can grade, and it is built
 * so that a broken renderer produces a **failed run**, never a plausible-looking
 * black image.
 *
 * ### Pipeline
 *
 * ```
 * build if stale -> vite preview -> chromium -> per shot:
 *   viewport -> navigate (?autostart=0) -> await __d2rim.ready
 *   -> setup script -> stepFrames(warmup) -> pose script
 *   -> capture -> decode -> BLANK-FRAME GUARD -> write
 * ```
 *
 * ### Why `?autostart=0` and `stepFrames`
 *
 * With the rAF loop running, the world advances between "the page is ready" and
 * "take the picture" by however long the harness happened to take, so two runs
 * of the same shot differ. The engine boots suspended instead, and this tool is
 * the only thing that advances time — `stepFrames(n)` uses a fixed synthetic
 * delta and awaits the renderer each frame, so shot N is byte-comparable across
 * runs and across machines. That is what makes golden-image diffing possible
 * later, and it is why `warmupFrames` is a property of the shot rather than a
 * sleep.
 *
 * ### Why the pixels come back through a readback
 *
 * This container's WebGPU swapchain renders correctly but never reaches the
 * headless compositor, so `page.screenshot()` of a WebGPU canvas is pure black.
 * And even on WebGL2, `render()` only *queues* GPU work: under SwiftShader the
 * CPU queues frames far faster than they retire, so a screenshot waits on a
 * compositor dozens of frames behind. `RendererHandle.captureFrame()` renders
 * to an offscreen target and reads it back, which is a genuine GPU sync point
 * and works identically on both backends. `mode: "screenshot"` remains
 * available for shots that must include DOM/HUD layers.
 *
 * ### Usage
 *
 *   node tools/capture/capture.mjs --shots tools/capture/shots.json --out captures/shots
 *   node tools/capture/capture.mjs --shots shots.json --out shots/ --only wide-establishing
 *   node tools/capture/capture.mjs --shots shots.json --out shots/ --url http://127.0.0.1:5173
 *
 * Exits 0 only if every requested shot was captured *and* passed the guard.
 */

import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, isAbsolute, join, relative, resolve } from 'node:path';

import { chromium } from 'playwright';

import {
  buildBundle,
  CHROMIUM_ARGS,
  findChromium,
  formatHelp,
  isBuildStale,
  parseArgs,
  ROOT,
  since,
  startPreviewServer,
} from './cli.mjs';
import { DEFAULT_THRESHOLDS, evaluateFrame, analyzeImageFile, formatStats } from './frame-guard.mjs';

const OPTIONS = {
  shots: { type: 'string', default: 'tools/capture/shots.json', help: 'shot list JSON' },
  out: { type: 'string', default: 'captures/shots', help: 'output directory for PNGs' },
  width: { type: 'number', help: 'override every shot width' },
  height: { type: 'number', help: 'override every shot height' },
  url: { type: 'string', help: 'use an already-running server instead of vite preview' },
  port: { type: 'number', default: 4173, help: 'port for the preview server' },
  backend: { type: 'string', help: 'force renderer backend: webgl2 | webgpu' },
  only: { type: 'string', help: 'comma-separated shot ids to run' },
  warmup: { type: 'number', help: 'override every shot warmupFrames' },
  build: { type: 'boolean', default: true, help: 'rebuild dist/ when it is stale' },
  'keep-going': { type: 'boolean', default: false, help: 'capture all shots even after a failure' },
  'ignore-page-errors': { type: 'boolean', default: false, help: 'do not fail on uncaught page exceptions' },
  help: { type: 'boolean', default: false, help: 'show this help' },
};

/** Per-shot defaults, overridden by `defaults` in the shot list, then per shot. */
const SHOT_DEFAULTS = Object.freeze({
  url: '/',
  width: 1920,
  height: 1080,
  warmupFrames: 60,
  mode: 'readback',
  backend: 'webgl2',
  setup: null,
  pose: null,
  guard: {},
});

/** Generous because SwiftShader renders this scene at roughly 5-20 fps. */
const TIMEOUTS = Object.freeze({
  navigate: 60_000,
  ready: 180_000,
  script: 60_000,
  step: 600_000,
  capture: 300_000,
});

/**
 * Bound a promise that has no timeout of its own.
 *
 * `page.evaluate()` — unlike `goto`, `waitForFunction` and `screenshot` — takes
 * no options argument and never times out. An engine that deadlocks inside
 * `stepFrames` would otherwise hang the harness indefinitely, which in an
 * automated loop is indistinguishable from a machine that has died. Racing a
 * timer does not cancel the in-page work, but it does let the run fail with a
 * message that names the stage.
 */
function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

// -- shot list ---------------------------------------------------------------

/**
 * Load and normalise a shot list.
 *
 * Accepts either `{ defaults, shots: [...] }` or a bare array. Every shot is
 * validated up front so a typo in shot 9 fails before shot 1 spends two minutes
 * rendering.
 *
 * @param {string} file
 * @returns {Promise<{shots: object[], source: string}>}
 */
async function loadShots(file) {
  const path = isAbsolute(file) ? file : resolve(ROOT, file);
  let parsed;
  try {
    parsed = JSON.parse(await readFile(path, 'utf8'));
  } catch (error) {
    throw new Error(`could not read shot list ${path}: ${error.message}`);
  }

  const list = Array.isArray(parsed) ? parsed : parsed.shots;
  const fileDefaults = Array.isArray(parsed) ? {} : (parsed.defaults ?? {});
  if (!Array.isArray(list)) {
    throw new Error(`${path} must be an array of shots, or an object with a "shots" array`);
  }

  const seen = new Set();
  const shots = list.map((shot, index) => {
    if (typeof shot?.id !== 'string' || shot.id.length === 0) {
      throw new Error(`shot #${index} has no "id"`);
    }
    if (!/^[a-z0-9][a-z0-9-]*$/i.test(shot.id)) {
      throw new Error(`shot "${shot.id}" is not a safe filename (use letters, digits and dashes)`);
    }
    if (seen.has(shot.id)) throw new Error(`duplicate shot id "${shot.id}"`);
    seen.add(shot.id);

    const merged = {
      ...SHOT_DEFAULTS,
      ...fileDefaults,
      ...shot,
      guard: { ...(fileDefaults.guard ?? {}), ...(shot.guard ?? {}) },
    };

    if (merged.mode !== 'readback' && merged.mode !== 'screenshot') {
      throw new Error(`shot "${shot.id}" has invalid mode "${merged.mode}"`);
    }
    if (merged.backend !== 'webgl2' && merged.backend !== 'webgpu') {
      throw new Error(`shot "${shot.id}" has invalid backend "${merged.backend}"`);
    }
    for (const key of ['width', 'height', 'warmupFrames']) {
      if (!Number.isFinite(merged[key]) || merged[key] < 0) {
        throw new Error(`shot "${shot.id}" has invalid ${key}: ${merged[key]}`);
      }
    }
    // JSON has no multi-line strings, so a script may also be given as an array
    // of lines. Without this every shot script would be one unreadable line
    // full of `\n`, and nobody would ever edit them.
    for (const key of ['setup', 'pose']) {
      const value = merged[key];
      if (Array.isArray(value)) {
        if (!value.every((line) => typeof line === 'string')) {
          throw new Error(`shot "${shot.id}" has a non-string line in its ${key} script`);
        }
        merged[key] = value.join('\n');
      } else if (value !== null && typeof value !== 'string') {
        throw new Error(`shot "${shot.id}" has a non-string ${key} script`);
      }
    }
    for (const key of Object.keys(merged.guard)) {
      if (!(key in DEFAULT_THRESHOLDS)) {
        throw new Error(`shot "${shot.id}" overrides unknown guard threshold "${key}"`);
      }
    }
    return merged;
  });

  return { shots, source: path };
}

/**
 * Resolve a shot's page URL against the server origin, forcing the parameters
 * the harness depends on.
 *
 * `autostart=0` is non-negotiable — a shot that let the rAF loop run would not
 * be reproducible — so it is applied after the shot's own query string.
 */
function resolveShotUrl(origin, shot) {
  const url = new URL(shot.url, origin);
  url.searchParams.set('backend', shot.backend);
  url.searchParams.set('autostart', '0');
  return url.toString();
}

// -- in-page routines --------------------------------------------------------
//
// These run inside the browser. They are written as standalone functions passed
// to `page.evaluate` so their source stays readable and lint-visible here,
// rather than living in template strings.

/**
 * Evaluate a shot script with the game's moving parts bound as named arguments.
 *
 * Shot scripts are strings (they come from JSON), so they cannot import. Every
 * handle they could reasonably need is injected instead, including the live
 * three.js namespace via `window.__d2rim.three`.
 */
function pageRunScript({ source, label }) {
  const d2rim = window.__d2rim;
  const engine = d2rim.engine;
  const ctx = engine.context;
  const factory = new Function(
    'd2rim',
    'engine',
    'ctx',
    'scene',
    'camera',
    'renderer',
    'time',
    'services',
    'events',
    'THREE',
    `return (async () => {\n${source}\n})();`,
  );
  return Promise.resolve(
    factory(
      d2rim,
      engine,
      ctx,
      ctx.scene,
      ctx.camera,
      ctx.renderer,
      ctx.time,
      ctx.services,
      ctx.events,
      d2rim.three,
    ),
  ).then(
    // Swallow the return value: a script that ends with an expression could
    // hand back an Object3D, and serialising one of those across CDP either
    // explodes on a cycle or ships a megabyte of nothing.
    () => undefined,
    (error) => {
      throw new Error(`${label} script failed: ${error?.message ?? error}`);
    },
  );
}

/**
 * Render one frame into an offscreen target, read it back, and re-encode it as
 * a PNG data URL via a 2D canvas.
 *
 * The 2D canvas round-trip is not decoration: it is the only surface in this
 * container that both receives GPU-produced pixels and composites, and
 * `toDataURL` gives a lossless PNG without shipping raw bytes over CDP.
 */
async function pageCaptureReadback({ width, height }) {
  const ctx = window.__d2rim.engine.context;
  if (typeof ctx.renderer.captureFrame !== 'function') {
    throw new Error('RendererHandle.captureFrame() is unavailable; cannot capture in readback mode');
  }

  const frame = await ctx.renderer.captureFrame(ctx.scene, ctx.camera, width, height);
  const canvas = document.createElement('canvas');
  canvas.width = frame.width;
  canvas.height = frame.height;
  const c2d = canvas.getContext('2d');
  const image = c2d.createImageData(frame.width, frame.height);
  image.data.set(frame.pixels);
  c2d.putImageData(image, 0, 0);
  return canvas.toDataURL('image/png');
}

/**
 * Push a frame to the visible canvas and wait for the GPU to actually finish it.
 *
 * `render()` only queues work, so screenshotting straight after it captures
 * whatever the compositor last saw. The tiny `captureFrame` that follows forces
 * a real readback on the same device, which cannot complete until the queued
 * canvas work ahead of it has retired.
 */
async function pageFlushToCanvas() {
  const ctx = window.__d2rim.engine.context;
  await ctx.renderer.render(ctx.scene, ctx.camera);
  if (typeof ctx.renderer.captureFrame === 'function') {
    await ctx.renderer.captureFrame(ctx.scene, ctx.camera, 16, 16);
  }
}

// -- capture -----------------------------------------------------------------

/**
 * Capture one shot end to end and judge the result.
 *
 * @returns {Promise<object>} the report entry for this shot
 */
async function captureShot(page, shot, { origin, outDir, ignorePageErrors }) {
  const startedAt = Date.now();
  const file = join(outDir, `${shot.id}.png`);
  const entry = {
    id: shot.id,
    description: shot.description ?? '',
    file: relative(ROOT, file),
    width: shot.width,
    height: shot.height,
    warmupFrames: shot.warmupFrames,
    mode: shot.mode,
    requestedBackend: shot.backend,
    ok: false,
    failures: [],
    pageErrors: [],
    stats: null,
    capturedAt: new Date().toISOString(),
    durationMs: 0,
  };

  const pageErrors = [];
  const onPageError = (error) => pageErrors.push(error.message);
  page.on('pageerror', onPageError);

  try {
    // Viewport first: the engine reads the canvas size at boot, so resizing
    // afterwards would render the first frames at the wrong aspect ratio.
    await page.setViewportSize({ width: shot.width, height: shot.height });

    const target = resolveShotUrl(origin, shot);
    process.stdout.write(`    navigate ${target}\n`);
    await page.goto(target, { waitUntil: 'load', timeout: TIMEOUTS.navigate });

    await page.waitForFunction(() => window.__d2rim !== undefined, null, {
      timeout: TIMEOUTS.navigate,
    });
    await withTimeout(
      page.evaluate(() => window.__d2rim.ready),
      TIMEOUTS.ready,
      'engine boot (__d2rim.ready)',
    );

    const actualBackend = await page.evaluate(() => window.__d2rim.engine.renderer.backend);
    entry.backend = actualBackend;
    if (actualBackend !== shot.backend) {
      throw new Error(
        `requested backend "${shot.backend}" but the page came up on "${actualBackend}"`,
      );
    }

    // Belt and braces: the ResizeObserver has usually fired by now, but an
    // explicit resize makes the camera aspect deterministic rather than
    // dependent on observer timing.
    await page.evaluate(() => window.__d2rim.engine.resize());

    if (shot.setup !== null) {
      process.stdout.write('    setup\n');
      await withTimeout(
        page.evaluate(pageRunScript, { source: shot.setup, label: 'setup' }),
        TIMEOUTS.script,
        'setup script',
      );
    }

    let warmupFrames = shot.warmupFrames;
    // Presenting to the WebGPU canvas swapchain in this container eventually
    // loses the device ("A valid external Instance reference no longer exists"),
    // which takes the readback down with it. Stepping zero frames renders
    // nothing to the canvas, so the device stays healthy. Animated WebGPU shots
    // are a platform limitation here, not an engine one.
    if (actualBackend === 'webgpu' && warmupFrames > 0) {
      process.stdout.write(
        `    ! webgpu: clamping warmupFrames ${warmupFrames} -> 0 (canvas presentation loses\n` +
          '      the device in this container; use backend "webgl2" for animated shots)\n',
      );
      warmupFrames = 0;
      entry.warmupFrames = 0;
      entry.warmupClamped = true;
    }

    if (warmupFrames > 0) {
      const steppedAt = Date.now();
      process.stdout.write(`    step ${warmupFrames} frames`);
      await withTimeout(
        page.evaluate((n) => window.__d2rim.engine.stepFrames(n), warmupFrames),
        TIMEOUTS.step,
        `stepFrames(${warmupFrames})`,
      );
      process.stdout.write(` (${since(steppedAt)})\n`);
    }

    // Pose runs *after* the warmup on purpose. Modules drive the camera from
    // `time.elapsed` every update, so a camera set before stepping would simply
    // be overwritten. Nothing updates between here and the capture render, so a
    // pose sticks exactly as written.
    if (shot.pose !== null) {
      process.stdout.write('    pose\n');
      await withTimeout(
        page.evaluate(pageRunScript, { source: shot.pose, label: 'pose' }),
        TIMEOUTS.script,
        'pose script',
      );
    }

    const capturedAt = Date.now();
    let buffer;
    if (shot.mode === 'screenshot') {
      await withTimeout(page.evaluate(pageFlushToCanvas), TIMEOUTS.capture, 'canvas flush');
      buffer = await page.screenshot({ timeout: TIMEOUTS.capture });
    } else {
      const dataUrl = await withTimeout(
        page.evaluate(pageCaptureReadback, { width: shot.width, height: shot.height }),
        TIMEOUTS.capture,
        'captureFrame readback',
      );
      const comma = dataUrl.indexOf(',');
      if (!dataUrl.startsWith('data:image/png;base64,') || comma === -1) {
        throw new Error('capture did not produce a PNG data URL');
      }
      buffer = Buffer.from(dataUrl.slice(comma + 1), 'base64');
    }

    await writeFile(file, buffer);
    entry.bytes = buffer.length;
    process.stdout.write(`    captured ${(buffer.length / 1024).toFixed(0)} KiB in ${since(capturedAt)}\n`);

    // The guard decodes from disk, not from the buffer above, so a bad encode
    // or a short write is caught too.
    const stats = await analyzeImageFile(file, shot.guard.edgeThreshold);
    const verdict = evaluateFrame(stats, {
      ...shot.guard,
      expectWidth: shot.width,
      expectHeight: shot.height,
    });
    entry.stats = stats;
    entry.failures = verdict.failures;
    entry.ok = verdict.ok;

    if (!ignorePageErrors && pageErrors.length > 0) {
      entry.ok = false;
      entry.failures = [...entry.failures, `page threw: ${pageErrors[0]}`];
    }
  } catch (error) {
    entry.failures = [...entry.failures, error.message];
    entry.ok = false;
  } finally {
    page.off('pageerror', onPageError);
    entry.pageErrors = pageErrors;
    entry.durationMs = Date.now() - startedAt;
  }

  return entry;
}

/**
 * Persist the run report, merged with whatever a previous run left behind.
 *
 * `--only one-shot` must not erase the other six shots' entries: the normal
 * workflow is a full run, then iterate on a single shot, then rebuild the
 * contact sheet — and the sheet reads this file for ordering and labels. So the
 * file carries the latest known result *per shot*, ordered by the shot list,
 * while `lastRun` records what this particular invocation did.
 *
 * Entries whose PNG has since been deleted are dropped, so the report can never
 * advertise a capture that is not on disk.
 */
async function writeReport(path, report, allShots, outDir) {
  const merged = new Map();

  try {
    const previous = JSON.parse(await readFile(path, 'utf8'));
    for (const entry of previous.shots ?? []) merged.set(entry.id, entry);
  } catch {
    // No previous report, or an unreadable one. Either way this run is the
    // whole truth.
  }

  for (const entry of report.shots) merged.set(entry.id, entry);

  const order = allShots.map((shot) => shot.id);
  const ordered = [...merged.values()].sort(
    (a, b) => indexOrLast(order, a.id) - indexOrLast(order, b.id),
  );

  const output = {
    ...report,
    lastRun: {
      generatedAt: report.generatedAt,
      ok: report.ok,
      shots: report.shots.map((entry) => entry.id),
    },
    shots: ordered.filter((entry) => existsSync(join(outDir, `${entry.id}.png`))),
  };

  await writeFile(path, `${JSON.stringify(output, null, 2)}\n`);
}

/**
 * Pairs of shots that photograph the *same* scene and must therefore agree.
 *
 * Every entry here exists because the harness once shipped a divergence it had
 * no way to notice. `exposure` is locked in `FrameGraph` and `autoExposure` is
 * off, so two shots of one scene from one camera cannot legitimately differ in
 * mean luminance — if they do, something in the backend path or the quality
 * ladder is changing the picture rather than the cost of the picture.
 *
 * `meanLuminance` catches "two different times of day". `nearBlackShare`
 * catches the other half of the same class of bug: a screen-space effect that
 * is only enabled at some tiers and is punching black holes in the frame does
 * not necessarily move the mean at all, because it removes light from a small
 * fraction of pixels and the tone curve hides the rest.
 */
const PARITY_PAIRS = [
  {
    reference: 'wide-establishing',
    subject: 'webgpu-backend-check',
    why: 'both backends, identical pose, locked exposure',
    meanLuminance: 0.03,
    nearBlackShare: 0.01,
  },
  {
    reference: 'quality-low',
    subject: 'quality-ultra',
    why: 'quality tiers must change cost, not art direction',
    meanLuminance: 0.03,
    // Looser than the backend pair, and for a real reason rather than to make
    // it pass: the tiers legitimately change `scatterDensity`, so `ultra` draws
    // several times as many props, rocks and trees as `low` (62 k unique
    // colours against 18 k). More small dark geometry means more genuinely dark
    // pixels, and that is the tier doing its job. What must NOT differ is the
    // *tone* — hence the tight mean — and what must not happen is a
    // screen-space effect punching holes, which at the scale it was doing so
    // (3.55% against 0.00%) this still catches.
    nearBlackShare: 0.025,
  },
];

/**
 * Compare every {@link PARITY_PAIRS} entry whose two shots are both present.
 *
 * Silently skipped when only one side was captured — `--only` runs are the
 * normal iteration workflow and must not fail for photographing one shot.
 *
 * @returns {{ id: string, message: string }[]}
 */
function checkParity(entries) {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const violations = [];

  for (const pair of PARITY_PAIRS) {
    const reference = byId.get(pair.reference);
    const subject = byId.get(pair.subject);
    if (reference?.stats == null || subject?.stats == null) continue;

    for (const metric of ['meanLuminance', 'nearBlackShare']) {
      const tolerance = pair[metric];
      const delta = Math.abs(subject.stats[metric] - reference.stats[metric]);
      if (delta <= tolerance) continue;
      violations.push({
        id: pair.subject,
        message:
          `parity: ${metric} differs from ${pair.reference} by ${delta.toFixed(4)} ` +
          `(tolerance ${tolerance}; ${pair.why})`,
      });
    }
  }

  return violations;
}

/** Position in `order`, or one past the end for ids no longer in the shot list. */
function indexOrLast(order, id) {
  const index = order.indexOf(id);
  return index === -1 ? order.length : index;
}

// -- main --------------------------------------------------------------------

async function main() {
  const { options } = parseArgs(process.argv.slice(2), OPTIONS);
  if (options.help) {
    console.log('Deterministic headless capture for d2rim.\n');
    console.log('Usage: node tools/capture/capture.mjs --shots shots.json --out shots/\n');
    console.log(formatHelp(OPTIONS));
    return 0;
  }

  const { shots: allShots, source } = await loadShots(options.shots);

  let shots = allShots;
  if (options.only !== undefined) {
    const wanted = new Set(options.only.split(',').map((id) => id.trim()).filter(Boolean));
    const known = new Set(allShots.map((shot) => shot.id));
    const unknown = [...wanted].filter((id) => !known.has(id));
    if (unknown.length > 0) {
      throw new Error(
        `--only names unknown shot(s): ${unknown.join(', ')}\nAvailable: ${[...known].join(', ')}`,
      );
    }
    shots = allShots.filter((shot) => wanted.has(shot.id));
  }

  // Global overrides are applied after the shot list is validated so that
  // `--width 640` re-renders every shot at draft resolution without editing JSON.
  shots = shots.map((shot) => ({
    ...shot,
    width: options.width ?? shot.width,
    height: options.height ?? shot.height,
    warmupFrames: options.warmup ?? shot.warmupFrames,
    backend: options.backend ?? shot.backend,
  }));

  if (shots.length === 0) {
    throw new Error(`${relative(ROOT, source)} contains no shots to capture`);
  }

  const outDir = isAbsolute(options.out) ? options.out : resolve(ROOT, options.out);
  await mkdir(outDir, { recursive: true });

  console.log(`d2rim capture`);
  console.log(`  shots   ${relative(ROOT, source)} (${shots.length} of ${allShots.length})`);
  console.log(`  out     ${relative(ROOT, outDir)}`);

  let server = null;
  let origin = options.url;

  if (origin === undefined) {
    if (options.build) {
      const { stale, reason } = await isBuildStale();
      if (stale) {
        console.log(`  build   rebuilding: ${reason}`);
        await buildBundle();
      } else {
        console.log(`  build   ${reason}`);
      }
    }
    server = await startPreviewServer(options.port);
    origin = server.url;
    console.log(`  server  ${origin} (vite preview)`);
  } else {
    console.log(`  server  ${origin} (external)`);
  }

  const executablePath = findChromium();
  console.log(`  browser ${executablePath}\n`);

  const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath });
  const report = {
    generatedAt: new Date().toISOString(),
    origin,
    shotList: relative(ROOT, source),
    outDir: relative(ROOT, outDir),
    chromium: executablePath,
    ok: false,
    shots: [],
  };

  try {
    const page = await browser.newPage({
      viewport: { width: shots[0]?.width ?? 1920, height: shots[0]?.height ?? 1080 },
      deviceScaleFactor: 1,
    });

    // Console output is echoed rather than swallowed: a WebGL shader-compile
    // warning is often the only clue why a frame degraded but still passed.
    page.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning') {
        console.log(`    [page:${message.type()}] ${message.text()}`);
      }
    });

    for (const [index, shot] of shots.entries()) {
      console.log(`[${index + 1}/${shots.length}] ${shot.id} - ${shot.description ?? ''}`);
      const entry = await captureShot(page, shot, {
        origin,
        outDir,
        ignorePageErrors: options['ignore-page-errors'],
      });
      report.shots.push(entry);

      if (entry.ok) {
        console.log(`    PASS  ${formatStats(entry.stats)}\n`);
      } else {
        console.log(`    FAIL`);
        for (const reason of entry.failures) console.log(`      ! ${reason}`);
        console.log('');
        if (!options['keep-going']) break;
      }
    }
  } finally {
    await browser.close();
    server?.stop();
  }

  const parity = checkParity(report.shots);
  for (const violation of parity) {
    const entry = report.shots.find((shot) => shot.id === violation.id);
    if (entry === undefined) continue;
    entry.ok = false;
    entry.failures.push(violation.message);
  }

  const passed = report.shots.filter((shot) => shot.ok);
  const failed = report.shots.filter((shot) => !shot.ok);
  const skipped = shots.length - report.shots.length;
  report.ok = failed.length === 0 && skipped === 0;

  await writeReport(join(outDir, 'report.json'), report, allShots, outDir);

  console.log('-'.repeat(78));
  for (const entry of report.shots) {
    const mark = entry.ok ? 'PASS' : 'FAIL';
    const detail = entry.stats === null ? entry.failures[0] : formatStats(entry.stats);
    console.log(`${mark}  ${entry.id.padEnd(26)} ${detail}`);
  }
  if (skipped > 0) console.log(`SKIP  ${skipped} shot(s) not attempted (stopped on first failure)`);
  console.log('-'.repeat(78));
  console.log(
    `${passed.length} passed, ${failed.length} failed${skipped > 0 ? `, ${skipped} skipped` : ''}` +
      `  ->  ${relative(ROOT, join(outDir, 'report.json'))}`,
  );

  if (!report.ok) {
    console.error(
      '\nCAPTURE FAILED. A degenerate frame means the render path is broken, not that the\n' +
        'thresholds are wrong - inspect the PNG before touching tools/capture/frame-guard.mjs.',
    );
    return 1;
  }
  return 0;
}

try {
  process.exitCode = await main();
} catch (error) {
  console.error(`\n${basename(process.argv[1])}: ${error.message}`);
  if (process.env.DEBUG) console.error(error.stack);
  process.exitCode = 1;
}
