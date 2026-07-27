/**
 * Tests for the shared denoiser's filter math.
 *
 * These are the functions that decide *what counts as the same surface*. Get
 * them wrong in the permissive direction and the filter blurs across
 * silhouettes, which is the halo every cheap half-resolution effect has; get
 * them wrong in the restrictive direction and it stops denoising at all and the
 * result crawls with sampling noise.
 */

import { describe, expect, it } from 'vitest';

import {
  B3_SPLINE_1D,
  atrousStepSize,
  atrousSupportRadius,
  atrousTapWeight,
  atrousWeight,
  bilateralUpsampleWeights,
  clampToNeighbourhood,
  depthEdgeWeight,
  interleavedGradientNoise,
  mipChainLevels,
  normalEdgeWeight,
  stdDevFromMoments,
  temporalAlpha,
} from '../src/render/post/Denoise';

describe('a-trous kernel', () => {
  it('uses the B3 spline and sums to one in 1D', () => {
    expect(B3_SPLINE_1D).toEqual([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16]);
    expect(B3_SPLINE_1D.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('sums to one over the full 5x5 support', () => {
    let total = 0;
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) total += atrousTapWeight(dx, dy, 2);
    }
    expect(total).toBeCloseTo(1, 12);
  });

  it('sums to one over the 3x3 tent used by the low tiers', () => {
    let total = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) total += atrousTapWeight(dx, dy, 1);
    }
    expect(total).toBeCloseTo(1, 12);
  });

  it('is separable and symmetric', () => {
    for (const [dx, dy] of [
      [1, 2],
      [-2, 1],
      [0, 2],
    ] as Array<[number, number]>) {
      expect(atrousTapWeight(dx, dy, 2)).toBeCloseTo(atrousTapWeight(dy, dx, 2), 12);
      expect(atrousTapWeight(dx, dy, 2)).toBeCloseTo(atrousTapWeight(-dx, -dy, 2), 12);
    }
  });

  it('returns zero outside the support instead of reading past the array', () => {
    expect(atrousTapWeight(3, 0, 2)).toBe(0);
    expect(atrousTapWeight(0, -3, 2)).toBe(0);
    expect(atrousTapWeight(2, 0, 1)).toBe(0);
  });

  it('doubles its tap spacing every iteration', () => {
    expect([0, 1, 2, 3, 4].map(atrousStepSize)).toEqual([1, 2, 4, 8, 16]);
  });

  it('reaches a wide support for a linear number of taps', () => {
    // Three 5x5 iterations span 14 texels either side for 75 taps; a single
    // filter of the same reach would be 29x29 = 841.
    expect(atrousSupportRadius(3, 2)).toBe(14);
    expect(atrousSupportRadius(5, 2)).toBe(62);
  });
});

describe('edge-stopping weights', () => {
  it('is exactly 1 for an identical depth', () => {
    expect(depthEdgeWeight(10, 10, 1, 0.05)).toBeCloseTo(1, 12);
  });

  it('falls off with depth difference', () => {
    const near = depthEdgeWeight(10, 10.01, 1, 0.05);
    const far = depthEdgeWeight(10, 11, 1, 0.05);
    expect(near).toBeGreaterThan(far);
    expect(far).toBeLessThan(0.01);
  });

  it('tolerates more depth difference on a steep surface', () => {
    // The gradient term is what stops a sloped floor being mistaken for a
    // silhouette; without it, AO stops being filtered on every ramp.
    const flat = depthEdgeWeight(10, 10.2, 0.05, 1);
    const steep = depthEdgeWeight(10, 10.2, 2, 1);
    expect(steep).toBeGreaterThan(flat);
  });

  it('never returns a negative or greater-than-one weight', () => {
    for (const d of [-5, 0, 0.001, 3, 1000]) {
      const w = depthEdgeWeight(10, 10 + d, 1, 0.05);
      expect(w).toBeGreaterThanOrEqual(0);
      expect(w).toBeLessThanOrEqual(1);
    }
  });

  it('rejects taps facing away and keeps aligned ones', () => {
    expect(normalEdgeWeight(1, 64)).toBeCloseTo(1, 12);
    expect(normalEdgeWeight(0, 64)).toBe(0);
    expect(normalEdgeWeight(-0.5, 64)).toBe(0);
  });

  it('makes the normal term a crease detector, not a soft cosine', () => {
    // A 10 degree difference should already have cut the weight substantially,
    // otherwise the filter rounds off every hard edge in the geometry.
    const tenDegrees = Math.cos((10 * Math.PI) / 180);
    expect(normalEdgeWeight(tenDegrees, 64)).toBeLessThan(0.4);
    const twoDegrees = Math.cos((2 * Math.PI) / 180);
    expect(normalEdgeWeight(twoDegrees, 64)).toBeGreaterThan(0.9);
  });

  it('composes the kernel and both edge terms', () => {
    const composed = atrousWeight(1, 0, 10, 10, 1, { depthSigma: 0.05, normalSigma: 64 });
    expect(composed).toBeCloseTo(atrousTapWeight(1, 0, 2), 12);

    const rejected = atrousWeight(1, 0, 10, 10, 0, { depthSigma: 0.05, normalSigma: 64 });
    expect(rejected).toBe(0);
  });
});

describe('joint bilateral upsample weights', () => {
  const bilinear = [0.25, 0.25, 0.25, 0.25];

  it('reduces to bilinear when every tap agrees', () => {
    const weights = bilateralUpsampleWeights(bilinear, [5, 5, 5, 5], 5, [1, 1, 1, 1], {
      depthSigma: 0.05,
      normalSigma: 32,
    });
    for (const w of weights) expect(w).toBeCloseTo(0.25, 6);
  });

  it('always sums to one', () => {
    const weights = bilateralUpsampleWeights(bilinear, [5, 5, 40, 5], 5, [1, 1, 1, 0.2], {
      depthSigma: 0.05,
      normalSigma: 32,
    });
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it('discards a tap from a different surface', () => {
    // Tap 2 is 8x further away: it belongs to whatever is behind the silhouette.
    const weights = bilateralUpsampleWeights(bilinear, [5, 5, 40, 5], 5, [1, 1, 1, 1], {
      depthSigma: 0.05,
      normalSigma: 32,
    });
    expect(weights[2]).toBeLessThan(1e-6);
    expect(weights[0]).toBeCloseTo(1 / 3, 4);
  });

  it('discards a tap facing the other way', () => {
    const weights = bilateralUpsampleWeights(bilinear, [5, 5, 5, 5], 5, [1, 1, -1, 1], {
      depthSigma: 0.05,
      normalSigma: 32,
    });
    expect(weights[2]).toBe(0);
  });

  it('falls back to bilinear rather than black when every tap is rejected', () => {
    // A one-pixel silhouette where no half-resolution sample belongs to this
    // surface. A slightly wrong value beats a hole in the image.
    const weights = bilateralUpsampleWeights(bilinear, [500, 500, 500, 500], 5, [-1, -1, -1, -1], {
      depthSigma: 0.05,
      normalSigma: 32,
    });
    expect(weights).toEqual(bilinear);
  });
});

describe('temporal accumulation', () => {
  it('takes the new sample outright on the first frame', () => {
    expect(temporalAlpha(0, 0.1)).toBeCloseTo(1, 12);
  });

  it('behaves as an arithmetic mean while the history is short', () => {
    expect(temporalAlpha(1, 0.05)).toBeCloseTo(1 / 2, 12);
    expect(temporalAlpha(3, 0.05)).toBeCloseTo(1 / 4, 12);
    expect(temporalAlpha(9, 0.05)).toBeCloseTo(1 / 10, 12);
  });

  it('settles into a fixed-weight filter once the history is long', () => {
    expect(temporalAlpha(100, 0.1)).toBeCloseTo(0.1, 12);
    expect(temporalAlpha(1e6, 0.1)).toBeCloseTo(0.1, 12);
  });

  it('never returns a weight outside (0, 1]', () => {
    for (const n of [-5, 0, 1, 7, 64, 1e9]) {
      const alpha = temporalAlpha(n, 0.1);
      expect(alpha).toBeGreaterThan(0);
      expect(alpha).toBeLessThanOrEqual(1);
    }
  });

  it('converges to the true mean of a constant signal', () => {
    let value = 0;
    for (let n = 0; n < 40; n++) value += (0.7 - value) * temporalAlpha(n, 0.1);
    expect(value).toBeCloseTo(0.7, 4);
  });

  it('clamps a stale history into the current neighbourhood', () => {
    // 0.9 is far outside a neighbourhood centred on 0.2 with sigma 0.05.
    expect(clampToNeighbourhood(0.9, 0.2, 0.05, 1.5)).toBeCloseTo(0.275, 6);
    // ... and leaves a plausible history alone.
    expect(clampToNeighbourhood(0.22, 0.2, 0.05, 1.5)).toBeCloseTo(0.22, 6);
  });

  it('produces a real standard deviation even for a uniform neighbourhood', () => {
    // Catastrophic cancellation makes E[x^2] - E[x]^2 slightly negative here;
    // an unguarded sqrt would return NaN and poison the history forever.
    expect(stdDevFromMoments(0.5, 0.25 - 1e-18)).toBe(0);
    expect(stdDevFromMoments(0, 0.25)).toBeCloseTo(0.5, 12);
  });
});

describe('mip chain levels', () => {
  it('counts level 0 and stops before a zero-sized level', () => {
    expect(mipChainLevels(960, 540)).toBe(10);
    expect(mipChainLevels(1, 1)).toBe(1);
    expect(mipChainLevels(8, 8)).toBe(4);
  });

  it('is driven by the smaller dimension', () => {
    expect(mipChainLevels(4096, 16)).toBe(5);
  });

  it('honours the cap', () => {
    expect(mipChainLevels(4096, 4096, 6)).toBe(6);
  });

  it('never returns zero for a degenerate size', () => {
    expect(mipChainLevels(0, 0)).toBe(1);
  });
});

describe('interleaved gradient noise', () => {
  it('stays inside [0, 1)', () => {
    for (let x = 0; x < 64; x++) {
      for (let y = 0; y < 64; y++) {
        const n = interleavedGradientNoise(x, y);
        expect(n).toBeGreaterThanOrEqual(0);
        expect(n).toBeLessThan(1);
      }
    }
  });

  it('averages close to 0.5 over a tile', () => {
    let total = 0;
    let count = 0;
    for (let x = 0; x < 128; x++) {
      for (let y = 0; y < 128; y++) {
        total += interleavedGradientNoise(x, y);
        count++;
      }
    }
    expect(total / count).toBeCloseTo(0.5, 2);
  });

  it('is well spread across its range rather than clumped', () => {
    // Low discrepancy is the entire reason for choosing IGN over a hash: every
    // decile of the range should be populated roughly equally, so a small number
    // of samples per pixel already covers the domain.
    const buckets = new Array<number>(10).fill(0);
    for (let x = 0; x < 100; x++) {
      for (let y = 0; y < 100; y++) {
        const index = Math.min(9, Math.floor(interleavedGradientNoise(x, y) * 10));
        buckets[index] = (buckets[index] ?? 0) + 1;
      }
    }
    for (const count of buckets) {
      expect(count).toBeGreaterThan(700);
      expect(count).toBeLessThan(1300);
    }
  });

  it('decorrelates across frames', () => {
    // If the frame offset did nothing, temporal accumulation would average the
    // same sample over and over and never converge.
    let identical = 0;
    for (let x = 0; x < 32; x++) {
      for (let y = 0; y < 32; y++) {
        if (Math.abs(interleavedGradientNoise(x, y, 0) - interleavedGradientNoise(x, y, 1)) < 1e-6) {
          identical++;
        }
      }
    }
    expect(identical).toBeLessThan(32);
  });

  it('is deterministic', () => {
    expect(interleavedGradientNoise(17, 42, 3)).toBe(interleavedGradientNoise(17, 42, 3));
  });
});
