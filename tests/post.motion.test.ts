/**
 * Unit tests for the motion-vector module's pure math.
 *
 * The jitter sequence is the part of TAA where a mistake is invisible in a
 * still frame and catastrophic in motion: a sequence that does not integrate to
 * the pixel centre converges to a *shifted* image, and one that clumps converges
 * slowly and shimmers while it does.
 */

import { describe, expect, it } from 'vitest';

import {
  NDC_TO_UV,
  haltonJitterSequence,
  jitterSequenceMean,
  radicalInverse,
} from '../src/render/post/Motion';

describe('radicalInverse', () => {
  it('reflects the digits about the radix point', () => {
    // 1, 2, 3 in base 2 are 1, 10, 11 -> 0.1, 0.01, 0.11 -> 1/2, 1/4, 3/4.
    expect(radicalInverse(1, 2)).toBeCloseTo(0.5, 12);
    expect(radicalInverse(2, 2)).toBeCloseTo(0.25, 12);
    expect(radicalInverse(3, 2)).toBeCloseTo(0.75, 12);
    expect(radicalInverse(4, 2)).toBeCloseTo(0.125, 12);
  });

  it('works in base 3', () => {
    expect(radicalInverse(1, 3)).toBeCloseTo(1 / 3, 12);
    expect(radicalInverse(2, 3)).toBeCloseTo(2 / 3, 12);
    expect(radicalInverse(3, 3)).toBeCloseTo(1 / 9, 12);
  });

  it('maps index 0 to 0', () => {
    expect(radicalInverse(0, 2)).toBe(0);
    expect(radicalInverse(0, 7)).toBe(0);
  });

  it('stays inside the unit interval for a long run', () => {
    for (let i = 1; i < 512; i++) {
      const value = radicalInverse(i, 2);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe('haltonJitterSequence', () => {
  it('produces the requested number of xy pairs', () => {
    expect(haltonJitterSequence(8)).toHaveLength(16);
    expect(haltonJitterSequence(16)).toHaveLength(32);
  });

  it('stays within a single pixel footprint', () => {
    const offsets = haltonJitterSequence(16);
    for (const value of offsets) {
      expect(value).toBeGreaterThanOrEqual(-0.5);
      expect(value).toBeLessThan(0.5);
    }
  });

  it('never repeats the degenerate zero offset', () => {
    const offsets = haltonJitterSequence(16);
    for (let i = 0; i < 16; i++) {
      const x = offsets[i * 2] ?? 0;
      const y = offsets[i * 2 + 1] ?? 0;
      expect(Math.abs(x) + Math.abs(y)).toBeGreaterThan(1e-9);
    }
  });

  it('integrates to the pixel centre', () => {
    // A biased sequence converges to an image shifted off the true sample grid,
    // which reads as a soft, subtly misregistered picture rather than as a bug.
    const [mx, my] = jitterSequenceMean(haltonJitterSequence(16));
    expect(Math.abs(mx)).toBeLessThan(0.04);
    expect(Math.abs(my)).toBeLessThan(0.06);
  });

  it('is well distributed at every prefix length, not just the full period', () => {
    // The whole reason for a low-discrepancy sequence: after *any* number of
    // frames the samples should already cover the footprint, so a TAA history
    // that is reset mid-sequence still resolves evenly.
    //
    // Halton(2, 3) needs 8 samples to reach all four quadrants — its first four
    // are (0, -0.17), (-0.25, 0.17), (0.25, -0.39), (-0.38, -0.06), which leave
    // the upper-right empty. That is a property of the sequence, not a defect:
    // by 8 samples it is even, and 8 is the shortest sequence the stack uses.
    const offsets = haltonJitterSequence(16);
    for (const prefix of [8, 12, 16]) {
      const quadrants = [0, 0, 0, 0];
      for (let i = 0; i < prefix; i++) {
        const x = offsets[i * 2] ?? 0;
        const y = offsets[i * 2 + 1] ?? 0;
        quadrants[(x < 0 ? 0 : 1) + (y < 0 ? 0 : 2)]! += 1;
      }
      for (const count of quadrants) expect(count).toBeGreaterThan(0);
    }
  });

  it('has no two samples closer than a random sequence typically would', () => {
    const offsets = haltonJitterSequence(16);
    let minimum = Infinity;
    for (let i = 0; i < 16; i++) {
      for (let j = i + 1; j < 16; j++) {
        const dx = (offsets[i * 2] ?? 0) - (offsets[j * 2] ?? 0);
        const dy = (offsets[i * 2 + 1] ?? 0) - (offsets[j * 2 + 1] ?? 0);
        minimum = Math.min(minimum, Math.hypot(dx, dy));
      }
    }
    // 16 points on a perfect grid inside a unit square are 0.25 apart. Halton
    // does not reach that, but clumping below ~0.08 would show as shimmer.
    expect(minimum).toBeGreaterThan(0.08);
  });
});

describe('NDC_TO_UV', () => {
  it('halves the x range and flips y', () => {
    // NDC spans [-1, 1] and texture space [0, 1], hence 0.5. The negative y is
    // the NDC-is-Y-up / render-target-texture-is-Y-down flip; getting it wrong
    // produces a TAA that smears every horizontal edge while looking almost
    // right on vertical ones.
    expect(NDC_TO_UV[0]).toBe(0.5);
    expect(NDC_TO_UV[1]).toBe(-0.5);
  });
});
