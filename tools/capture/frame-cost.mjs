/**
 * @module tools/capture/frame-cost
 *
 * Measure what a frame *costs to submit*, headlessly, with no GPU required.
 *
 * This container rasterises in software on four cores, so it cannot say
 * anything true about frame time. It can, however, say exactly how many draw
 * calls and triangles the renderer submits, how many render targets the post
 * stack has allocated and how many bytes of texture are resident — all of which
 * are device-independent properties of the scene and the tier, and all of which
 * are the numbers an optimisation pass actually moves.
 *
 * Run it before and after a change and diff the output. That is the only honest
 * performance claim available from here.
 *
 *   node tools/capture/frame-cost.mjs
 *   node tools/capture/frame-cost.mjs --zones encampment,bloodMoor --tiers low,medium
 *   node tools/capture/frame-cost.mjs --out captures/frame-cost.json
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, resolve } from 'node:path';

import { chromium } from 'playwright';

import {
  buildBundle,
  CHROMIUM_ARGS,
  findChromium,
  formatHelp,
  isBuildStale,
  parseArgs,
  ROOT,
  startPreviewServer,
} from './cli.mjs';

const SPEC = {
  zones: { type: 'string', default: 'encampment,bloodMoor', help: 'comma-separated zone ids' },
  tiers: { type: 'string', default: 'low,medium', help: 'comma-separated ?quality= values' },
  flags: {
    type: 'string',
    default: '',
    help:
      'comma-separated render-flag sets to compare, each an & -joined query fragment; ' +
      'use "-" for the unmodified baseline (e.g. "-,shadows=off,fog=off,minimal=1")',
  },
  backend: { type: 'string', default: 'webgl2', help: 'webgl2 | webgpu' },
  width: { type: 'number', default: 1528, help: 'CSS viewport width' },
  height: { type: 'number', default: 794, help: 'CSS viewport height' },
  frames: { type: 'number', default: 12, help: 'deterministic frames to step before reading' },
  port: { type: 'number', default: 4319, help: 'preview server port' },
  out: { type: 'string', default: '', help: 'write JSON here as well as stdout' },
  help: { type: 'boolean', default: false, help: 'show this' },
};

/**
 * Read the counters, in the page.
 *
 * `renderer.info.render` is reset by `Engine` at the top of every frame and is
 * complete only after the render has been awaited, so this runs after
 * `stepFrames` rather than from inside a module update.
 */
const PROBE = `
  const d2rim = window.__d2rim;
  await d2rim.ready;
  await d2rim.engine.stepFrames(FRAMES);

  const three = d2rim.ctx.renderer.three;
  const render = three.info?.render ?? {};
  const memory = three.info?.memory ?? {};

  const assets = d2rim.ctx.services.tryGet('assets');
  const post = d2rim.render?.post;
  const stats = post?.stats ?? null;

  return {
    drawCalls: render.drawCalls ?? render.calls ?? 0,
    triangles: Math.round(render.triangles ?? 0),
    geometries: memory.geometries ?? 0,
    textures: memory.textures ?? 0,
    flags: window.__d2rimFlags ?? null,
    programs: d2rim.ctx.renderer.programCount?.() ?? null,
    postQuality: post?.quality ?? null,
    postPasses: stats ? stats.active.slice() : null,
    postDraws: stats ? stats.passDraws : null,
    postBytes: stats ? stats.bytes : null,
    shadowCascades: d2rim.render?.settings?.tier?.shadowCascades ?? null,
    shadowMapSize: d2rim.render?.settings?.tier?.shadowMapSize ?? null,
    assetBytes: assets?.stats?.().bytes ?? null,
    compressedFormat: assets?.compressedFormat?.format ?? null,
    compressedLoads: assets?.compressedLoadCount ?? null,
    pixelRatio: three.getPixelRatio?.() ?? null,
  };
`;

const mb = (bytes) => (bytes === null || bytes === undefined ? '—' : (bytes / 1048576).toFixed(1));

async function main() {
  const { options: args } = parseArgs(process.argv.slice(2), SPEC);
  if (args.help) {
    console.log(formatHelp(SPEC));
    return 0;
  }

  if (await isBuildStale()) await buildBundle();
  const server = await startPreviewServer(args.port);
  const executablePath = findChromium();
  const browser = await chromium.launch({
    args: CHROMIUM_ARGS,
    ...(executablePath ? { executablePath } : {}),
  });

  const rows = [];
  try {
    const page = await browser.newPage({
      viewport: { width: args.width, height: args.height },
    });

    // `-` is the baseline: no extra query at all. Spelled explicitly rather
    // than as an empty string so a trailing comma in `--flags` cannot silently
    // add a duplicate baseline row and make a comparison look like a no-op.
    const flagSets = args.flags === '' ? ['-'] : args.flags.split(',').filter(Boolean);

    for (const tier of args.tiers.split(',').filter(Boolean)) {
      for (const zone of args.zones.split(',').filter(Boolean)) {
        for (const flagSet of flagSets) {
          const extra = flagSet === '-' ? '' : `&${flagSet}`;
          const url =
            `${server.url}/?autostart=0&enemies=0&fade=0&stats=1` +
            `&backend=${args.backend}&quality=${tier}&zone=${zone}${extra}`;
          process.stdout.write(
            `  ${tier.padEnd(7)} ${zone.padEnd(12)} ${flagSet.padEnd(22)} ... `,
          );
          await page.goto(url, { waitUntil: 'load', timeout: 180_000 });
          const result = await page.evaluate(
            new Function(`return (async () => { ${PROBE.replace('FRAMES', args.frames)} })();`),
          );
          rows.push({ tier, zone, flagSet, ...result });
          process.stdout.write(
            `${String(result.drawCalls).padStart(5)} draws  ` +
              `${result.triangles.toLocaleString('en-US').padStart(9)} tris  ` +
              `${mb(result.assetBytes).padStart(6)} MB tex  ` +
              `${mb(result.postBytes).padStart(6)} MB post-rt\n`,
          );
        }
      }
    }
  } finally {
    await browser.close();
    server.stop();
  }

  console.log('');
  console.log(
    'tier    zone           flags                   draws     triangles  post-rt MB  passes',
  );
  console.log('-'.repeat(104));
  const baseline = rows.find((r) => r.flagSet === '-');
  for (const r of rows) {
    // The delta against the unmodified run is the whole point of `--flags`: a
    // kill switch that does not move the draw count has not removed any work,
    // whatever the frame time does.
    const delta =
      baseline && r !== baseline && r.tier === baseline.tier && r.zone === baseline.zone
        ? ` (${r.drawCalls - baseline.drawCalls >= 0 ? '+' : ''}${r.drawCalls - baseline.drawCalls})`
        : '';
    console.log(
      `${r.tier.padEnd(7)} ${r.zone.padEnd(14)} ${String(r.flagSet ?? '-').padEnd(22)} ` +
        `${String(r.drawCalls).padStart(5)}${delta.padEnd(8)}  ` +
        `${r.triangles.toLocaleString('en-US').padStart(11)}  ${mb(r.postBytes).padStart(10)}  ` +
        `${(r.postPasses ?? []).join(' ')}`,
    );
  }
  const format = rows.find((r) => r.compressedFormat)?.compressedFormat;
  if (format) console.log(`\ncompressed texture format: ${format}`);

  if (args.out) {
    const path = isAbsolute(args.out) ? args.out : resolve(ROOT, args.out);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, `${JSON.stringify(rows, null, 2)}\n`);
    console.log(`\nwrote ${path}`);
  }
  return 0;
}

process.exitCode = await main();
