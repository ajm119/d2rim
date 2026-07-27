/**
 * @module tools/capture/frame-guard
 *
 * The blank-frame guard: decodes a captured PNG and proves it contains a real
 * rendered image.
 *
 * ### Why this exists
 *
 * The worst failure mode in an automated visual-quality loop is not a crash —
 * it is a capture harness that quietly writes a 1.2 MB black PNG. Every
 * downstream consumer (contact sheets, critic agents, golden-image diffs) then
 * grades a frame that was never rendered, and the loop reports progress while
 * the renderer is dead. So every capture is decoded back off disk and put
 * through the checks below before the harness is allowed to succeed. Anything
 * degenerate is a hard, loud, nonzero-exit failure.
 *
 * Reading the PNG back from disk (rather than checking the pixel buffer in
 * memory) is deliberate: it also validates the encode and the write, so a
 * truncated file or a bad `toDataURL` cannot slip through.
 *
 * ### The checks
 *
 * No single statistic catches every degenerate frame, so five cheap and
 * near-orthogonal ones are combined:
 *
 * | statistic             | catches                                            |
 * |-----------------------|----------------------------------------------------|
 * | `uniqueColors`        | solid fills, 2-tone clear-colour-only frames       |
 * | `meanLuminance`       | all-black (device lost) and blown-out white        |
 * | `luminanceStdDev`     | flat fills of *any* brightness, incl. mid grey     |
 * | `edgeDensity`         | smooth gradients with no geometry in them          |
 * | `dominantBucketShare` | a frame that is 99 % sky with the scene missing    |
 * | `opaqueShare`         | a fully transparent PNG (alpha never written)      |
 *
 * A clear sky gradient with nothing in front of it passes the first three and
 * fails the last two — which is exactly the case "the renderer booted but the
 * scene never loaded", and exactly what a naive black-pixel check misses.
 *
 * ### Standalone use
 *
 *   node tools/capture/frame-guard.mjs captures/shots/*.png
 *
 * Prints a statistics table and exits nonzero if any frame is degenerate.
 */

import { basename } from 'node:path';

import sharp from 'sharp';

/**
 * Default acceptance thresholds.
 *
 * Measured against the reference scene at 1600x900: its shots land at 59k-126k
 * unique colours, luminance sigma 0.115-0.193, and edge density 0.197-0.392.
 * Every threshold below sits one to two orders of magnitude away from those
 * values, so the guard flags catastrophes and never bikesheds a frame that
 * merely got darker.
 */
export const DEFAULT_THRESHOLDS = Object.freeze({
  /** Distinct 24-bit RGB triples. A solid fill has 1; dithered noise has ~thousands. */
  minUniqueColors: 512,
  /** Mean luminance, 0–1. Below this the frame is black in every practical sense. */
  minMeanLuminance: 0.02,
  /** Above this the frame is blown out — usually a broken tone-mapping path. */
  maxMeanLuminance: 0.97,
  /** Standard deviation of luminance. A uniform frame of any colour scores 0. */
  minLuminanceStdDev: 0.015,
  /** Share of pixels sitting on a local luminance step > `edgeThreshold`. */
  minEdgeDensity: 0.002,
  /** Share of pixels in the single most populated 5-bit-per-channel colour bucket. */
  maxDominantBucketShare: 0.92,
  /** Share of pixels with alpha >= 250. Catches never-written alpha. */
  minOpaqueShare: 0.98,
  /** Local luminance delta (0–1) that counts as an edge. */
  edgeThreshold: 0.02,
});

/** Human-readable label and comparison direction for each threshold. */
const CHECKS = [
  ['uniqueColors', 'minUniqueColors', 'atLeast', 'unique colours'],
  ['meanLuminance', 'minMeanLuminance', 'atLeast', 'mean luminance'],
  ['meanLuminance', 'maxMeanLuminance', 'atMost', 'mean luminance'],
  ['luminanceStdDev', 'minLuminanceStdDev', 'atLeast', 'luminance std-dev'],
  ['edgeDensity', 'minEdgeDensity', 'atLeast', 'edge density'],
  ['dominantBucketShare', 'maxDominantBucketShare', 'atMost', 'dominant colour share'],
  ['opaqueShare', 'minOpaqueShare', 'atLeast', 'opaque pixel share'],
];

/**
 * @typedef {object} FrameStats
 * @property {number} width
 * @property {number} height
 * @property {number} pixels             total pixel count
 * @property {number} uniqueColors       distinct 24-bit RGB values
 * @property {number} meanLuminance      0–1
 * @property {number} luminanceStdDev    0–1
 * @property {number} edgeDensity        0–1, share of pixels on a luminance step
 * @property {number} dominantBucketShare 0–1, largest 32x32x32 colour bucket
 * @property {number} nearBlackShare     0–1, luminance < 0.02
 * @property {number} nearWhiteShare     0–1, luminance > 0.98
 * @property {number} opaqueShare        0–1, alpha >= 250
 */

/**
 * Compute statistics for a tightly packed RGBA8 buffer.
 *
 * Single pass for the per-pixel accumulators plus one pass over the interior
 * for edges; both are O(n) with no allocation per pixel, so a 1920x1080 frame
 * costs a few tens of milliseconds.
 *
 * @param {Uint8Array | Buffer} rgba `width * height * 4` bytes, row-major.
 * @param {number} width
 * @param {number} height
 * @param {number} [edgeThreshold]
 * @returns {FrameStats}
 */
export function analyzeRgba(rgba, width, height, edgeThreshold = DEFAULT_THRESHOLDS.edgeThreshold) {
  const count = width * height;
  if (rgba.length < count * 4) {
    throw new Error(
      `pixel buffer too small: got ${rgba.length} bytes, need ${count * 4} for ${width}x${height}`,
    );
  }

  // Bit-packed presence set over the 24-bit RGB space: 2 MiB instead of the
  // 16 MiB a byte-per-colour flag array would cost, and far cheaper than a Set
  // holding two million boxed numbers.
  const seen = new Uint8Array(1 << 21);
  // 5 bits per channel. Coarse enough that near-identical pixels of a flat fill
  // land in one bucket, fine enough that a real gradient spreads across many.
  const buckets = new Uint32Array(1 << 15);
  // Luminance is cached so the edge pass does not redo the dot product.
  const luma = new Float32Array(count);

  let sum = 0;
  let sumSquares = 0;
  let nearBlack = 0;
  let nearWhite = 0;
  let opaque = 0;
  let uniqueColors = 0;
  let dominantBucket = 0;

  for (let i = 0, p = 0; i < count; i++, p += 4) {
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];

    // Rec. 709 luma on the sRGB-encoded values. Perceptual rather than
    // physical on purpose: "is this frame black to a viewer" is the question.
    const l = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    luma[i] = l;
    sum += l;
    sumSquares += l * l;
    if (l < 0.02) nearBlack++;
    else if (l > 0.98) nearWhite++;
    if (rgba[p + 3] >= 250) opaque++;

    const key = (r << 16) | (g << 8) | b;
    const byte = key >>> 3;
    const mask = 1 << (key & 7);
    if ((seen[byte] & mask) === 0) {
      seen[byte] |= mask;
      uniqueColors++;
    }

    const bucket = ((r >>> 3) << 10) | ((g >>> 3) << 5) | (b >>> 3);
    const bucketCount = ++buckets[bucket];
    if (bucketCount > dominantBucket) dominantBucket = bucketCount;
  }

  // Forward differences against the right and lower neighbour. Cheaper than a
  // Sobel and, for a "does this frame contain geometry" test, just as decisive.
  let edges = 0;
  let interior = 0;
  for (let y = 0; y < height - 1; y++) {
    const row = y * width;
    for (let x = 0; x < width - 1; x++) {
      const i = row + x;
      const here = luma[i];
      const gradient = Math.abs(luma[i + 1] - here) + Math.abs(luma[i + width] - here);
      if (gradient > edgeThreshold) edges++;
      interior++;
    }
  }

  const mean = sum / count;
  // max(0, ...) guards the tiny negative that float error can produce when the
  // frame really is uniform.
  const variance = Math.max(0, sumSquares / count - mean * mean);

  return {
    width,
    height,
    pixels: count,
    uniqueColors,
    meanLuminance: mean,
    luminanceStdDev: Math.sqrt(variance),
    edgeDensity: interior === 0 ? 0 : edges / interior,
    dominantBucketShare: dominantBucket / count,
    nearBlackShare: nearBlack / count,
    nearWhiteShare: nearWhite / count,
    opaqueShare: opaque / count,
  };
}

/**
 * Decode a PNG (or any sharp-readable image) from disk and analyse it.
 *
 * @param {string} filePath
 * @param {number} [edgeThreshold]
 * @returns {Promise<FrameStats>}
 */
export async function analyzeImageFile(filePath, edgeThreshold) {
  const { data, info } = await sharp(filePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  if (info.channels !== 4) {
    throw new Error(`expected 4 channels after ensureAlpha(), got ${info.channels}`);
  }
  if (info.width === 0 || info.height === 0) {
    throw new Error(`decoded image has zero extent (${info.width}x${info.height})`);
  }
  return analyzeRgba(data, info.width, info.height, edgeThreshold);
}

/**
 * @typedef {object} GuardResult
 * @property {boolean} ok
 * @property {FrameStats} stats
 * @property {string[]} failures Human-readable reasons, empty when `ok`.
 */

/**
 * Test statistics against thresholds.
 *
 * @param {FrameStats} stats
 * @param {Partial<typeof DEFAULT_THRESHOLDS> & {expectWidth?: number, expectHeight?: number}} [overrides]
 * @returns {GuardResult}
 */
export function evaluateFrame(stats, overrides = {}) {
  const thresholds = { ...DEFAULT_THRESHOLDS, ...overrides };
  const failures = [];

  for (const [statKey, thresholdKey, direction, label] of CHECKS) {
    const actual = stats[statKey];
    const limit = thresholds[thresholdKey];
    if (typeof limit !== 'number') continue;
    const failed = direction === 'atLeast' ? actual < limit : actual > limit;
    if (failed) {
      const relation = direction === 'atLeast' ? '>=' : '<=';
      failures.push(
        `${label} ${formatNumber(actual)} violates ${thresholdKey} (${relation} ${formatNumber(limit)})`,
      );
    }
  }

  // Dimension mismatch means the harness captured something other than what it
  // was asked for — a stale render target, or a viewport that never resized.
  const { expectWidth, expectHeight } = overrides;
  if (
    (typeof expectWidth === 'number' && stats.width !== expectWidth) ||
    (typeof expectHeight === 'number' && stats.height !== expectHeight)
  ) {
    failures.push(
      `dimensions ${stats.width}x${stats.height} do not match the requested ` +
        `${expectWidth ?? '?'}x${expectHeight ?? '?'}`,
    );
  }

  return { ok: failures.length === 0, stats, failures };
}

/**
 * Decode, analyse and judge an image file in one call.
 *
 * @param {string} filePath
 * @param {Parameters<typeof evaluateFrame>[1]} [overrides]
 * @returns {Promise<GuardResult>}
 */
export async function guardImageFile(filePath, overrides = {}) {
  const stats = await analyzeImageFile(filePath, overrides.edgeThreshold);
  return evaluateFrame(stats, overrides);
}

/** Fixed-width-ish number formatting that keeps small shares readable. */
function formatNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  if (Number.isInteger(value)) return value.toLocaleString('en-US');
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(5);
}

/** One-line summary of a frame's statistics, for console output and reports. */
export function formatStats(stats) {
  return [
    `${stats.width}x${stats.height}`,
    `colours=${formatNumber(stats.uniqueColors)}`,
    `luma=${stats.meanLuminance.toFixed(3)}+/-${stats.luminanceStdDev.toFixed(3)}`,
    `edges=${stats.edgeDensity.toFixed(4)}`,
    `dominant=${(stats.dominantBucketShare * 100).toFixed(1)}%`,
  ].join('  ');
}

// -- CLI ---------------------------------------------------------------------

if (import.meta.url === `file://${process.argv[1]}`) {
  const files = process.argv.slice(2).filter((arg) => !arg.startsWith('-'));
  if (files.length === 0) {
    console.error('usage: node tools/capture/frame-guard.mjs <image.png> [more.png ...]');
    process.exit(2);
  }

  let failed = 0;
  for (const file of files) {
    try {
      const { ok, stats, failures } = await guardImageFile(file);
      const mark = ok ? 'PASS' : 'FAIL';
      console.log(`${mark}  ${basename(file)}  ${formatStats(stats)}`);
      for (const reason of failures) console.log(`        ! ${reason}`);
      if (!ok) failed++;
    } catch (error) {
      console.log(`FAIL  ${basename(file)}  could not be decoded: ${error.message}`);
      failed++;
    }
  }

  if (failed > 0) {
    console.error(`\n${failed} of ${files.length} frame(s) are degenerate.`);
    process.exit(1);
  }
  console.log(`\nAll ${files.length} frame(s) look like real renders.`);
}
