/**
 * tools/capture/exposure-report.mjs
 *
 * Numbers to argue with while looking at the picture — never instead of it.
 *
 * Reports, per image: the luma histogram at nine percentiles, the share of the
 * frame that is crushed or clipped, and — the number this project kept getting
 * wrong — the **warm/cold separation**: how far apart the mean hue of the
 * frame's warmest decile and its coldest decile actually are. A frame can have
 * a textbook histogram and still read as monotone brown, and that is precisely
 * what "no cold/warm separation" means.
 *
 * Usage: `node tools/capture/exposure-report.mjs <image...>`
 */

import sharp from 'sharp';

/** sRGB byte -> linear, for honest luminance. */
const toLinear = (v) => {
  const s = v / 255;
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
};

const percentile = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];

export async function reportImage(file) {
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const n = info.width * info.height;
  const luma = new Float32Array(n);
  // Blue-minus-red in display space: positive is cold, negative is warm. Cheap,
  // stable, and it is exactly the axis the split-tone grade works on.
  const warmCold = new Float32Array(n);
  const sat = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    luma[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    warmCold[i] = (b - r) / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    sat[i] = max === 0 ? 0 : (max - min) / max;
  }

  const sorted = Float32Array.from(luma).sort();
  const p = (q) => +percentile(sorted, q).toFixed(3);

  // Warm/cold separation: mean of the warm-cold axis over the warmest 10% of
  // pixels against the coldest 10%, so a frame with a real fire pool and a real
  // blue surround scores high and a monotone brown frame scores near zero.
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => warmCold[a] - warmCold[b]);
  const decile = Math.max(1, Math.floor(n / 10));
  let warmSum = 0;
  let coldSum = 0;
  let warmLuma = 0;
  let coldLuma = 0;
  for (let i = 0; i < decile; i++) {
    warmSum += warmCold[order[i]];
    warmLuma += luma[order[i]];
    coldSum += warmCold[order[n - 1 - i]];
    coldLuma += luma[order[n - 1 - i]];
  }

  let crushed = 0;
  let clipped = 0;
  let satSum = 0;
  for (let i = 0; i < n; i++) {
    if (luma[i] < 0.02) crushed++;
    if (luma[i] > 0.98) clipped++;
    satSum += sat[i];
  }

  return {
    file,
    size: `${info.width}x${info.height}`,
    p01: p(0.01),
    p05: p(0.05),
    p10: p(0.1),
    p25: p(0.25),
    p50: p(0.5),
    p75: p(0.75),
    p90: p(0.9),
    p95: p(0.95),
    p99: p(0.99),
    mean: +(sorted.reduce((a, x) => a + x, 0) / n).toFixed(3),
    crushedShare: +(crushed / n).toFixed(4),
    clippedShare: +(clipped / n).toFixed(4),
    meanSaturation: +(satSum / n).toFixed(3),
    warmestDecile: +(warmSum / decile).toFixed(3),
    coldestDecile: +(coldSum / decile).toFixed(3),
    warmColdSeparation: +((coldSum - warmSum) / decile).toFixed(3),
    warmDecileLuma: +(warmLuma / decile).toFixed(3),
    coldDecileLuma: +(coldLuma / decile).toFixed(3),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const files = process.argv.slice(2);
  if (files.length === 0) {
    console.error('usage: node tools/capture/exposure-report.mjs <image...>');
    process.exit(2);
  }
  for (const file of files) {
    const r = await reportImage(file);
    console.log(
      `${r.file} ${r.size}\n` +
        `  luma  p01=${r.p01} p05=${r.p05} p10=${r.p10} p25=${r.p25} p50=${r.p50} ` +
        `p75=${r.p75} p90=${r.p90} p95=${r.p95} p99=${r.p99} mean=${r.mean}\n` +
        `  crushed<0.02 ${(r.crushedShare * 100).toFixed(2)}%  clipped>0.98 ${(r.clippedShare * 100).toFixed(2)}%  ` +
        `mean sat ${r.meanSaturation}\n` +
        `  warm/cold: warmest decile ${r.warmestDecile} (luma ${r.warmDecileLuma}), ` +
        `coldest ${r.coldestDecile} (luma ${r.coldDecileLuma}), separation ${r.warmColdSeparation}`,
    );
  }
}
