/**
 * Tests for the blending, projection, detail and macro-variation math.
 *
 * These are the functions where a wrong implementation still *looks* like a
 * material — it is dark in the creases and light on the peaks — while quietly
 * changing the average brightness of every large surface in the game. The
 * properties pinned here are the ones a plausible-but-wrong version violates:
 *
 * - partition of unity, so a blend never darkens or blows out;
 * - degeneration of the height blend to a plain lerp as the band widens, so the
 *   sharpness knob is continuous and has no special cases;
 * - exclusion of zero-coverage layers, so a material at zero weight can never
 *   poke through on height alone;
 * - unit length and correct identities for normal composition, since a
 *   non-unit shading normal silently rescales the whole BRDF;
 * - mean preservation for macro variation, because the archetype table's
 *   authored albedos are only meaningful if the variation is energy neutral.
 */

import { describe, expect, it } from 'vitest';

import {
  heightBlend2,
  heightBlendWeights,
  reorientNormal,
  scaleNormal,
  sharpenWeights,
  variancePreservingWeights,
  type Vec3Tuple,
} from '../src/render/materials/Blending';
import { detailDensityFade, detailDistanceFade, detailFade } from '../src/render/materials/Detail';
import {
  macroAlbedoGain,
  macroAlbedoGainMean,
  macroRoughnessDelta,
  macroTintWeight,
} from '../src/render/materials/MacroVariation';
import { triplanarUVs, triplanarWeights } from '../src/render/materials/Triplanar';

const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0);

describe('height blending', () => {
  it('produces weights that sum to one', () => {
    for (const depth of [0.02, 0.1, 0.5, 2]) {
      for (let w = 0; w <= 1.0001; w += 0.1) {
        const weights = heightBlendWeights([0.2, 0.8], [1 - w, w], depth);
        expect(sum(weights)).toBeCloseTo(1, 10);
      }
    }
  });

  it('never returns a negative weight', () => {
    const weights = heightBlendWeights([0.9, 0.1, 0.5], [0.2, 0.7, 0.1], 0.05);
    for (const w of weights) expect(w).toBeGreaterThanOrEqual(0);
  });

  it('lets the taller material win inside the transition band', () => {
    // Equal coverage, but layer 1 stands much higher: it should dominate.
    const weights = heightBlendWeights([0.1, 0.9], [0.5, 0.5], 0.2);
    expect(weights[1]).toBeGreaterThan(0.99);
  });

  it('degenerates to a linear lerp as the band widens', () => {
    // With a band far wider than the height range, both biases are dominated by
    // the coverage terms and the operator reduces to `mix`.
    const w = heightBlend2(0.2, 0.8, 0.3, 1000);
    expect(w).toBeCloseTo(0.3, 4);
  });

  it('excludes a layer at exactly zero coverage however tall it is', () => {
    // The classic failure: rock at zero splat weight poking through the grass
    // because its height map happens to be higher everywhere.
    const w = heightBlend2(0.05, 1.0, 0, 0.1);
    expect(w).toBe(0);
  });

  it('is symmetric under swapping the two layers', () => {
    const forward = heightBlend2(0.3, 0.7, 0.4, 0.15);
    const backward = heightBlend2(0.7, 0.3, 0.6, 0.15);
    expect(forward).toBeCloseTo(1 - backward, 10);
  });

  it('falls back to an even split when every layer has zero coverage', () => {
    const weights = heightBlendWeights([0.4, 0.6], [0, 0], 0.1);
    expect(weights).toEqual([0.5, 0.5]);
  });
});

describe('variance-preserving weights', () => {
  it('leaves a one-hot weight vector alone', () => {
    expect(variancePreservingWeights([1, 0, 0])).toEqual([1, 0, 0]);
  });

  it('scales equal weights by sqrt(n), restoring the deviation averaging lost', () => {
    const w = variancePreservingWeights([1 / 3, 1 / 3, 1 / 3]);
    // Each weight becomes 1/sqrt(3), so the sum is sqrt(3): exactly the factor
    // by which averaging three independent samples shrank the std deviation.
    expect(sum(w)).toBeCloseTo(Math.sqrt(3), 10);
  });

  it('always produces a unit-L2 weight vector', () => {
    const w = variancePreservingWeights([0.6, 0.3, 0.1]);
    expect(Math.hypot(...w)).toBeCloseTo(1, 10);
  });
});

describe('weight sharpening', () => {
  it('preserves the L1 normalisation', () => {
    expect(sum(sharpenWeights([0.5, 0.3, 0.2], 7))).toBeCloseTo(1, 10);
  });

  it('concentrates weight on the largest entry', () => {
    const before: [number, number, number] = [0.5, 0.3, 0.2];
    const after = sharpenWeights(before, 7);
    expect(after[0]).toBeGreaterThan(before[0] ?? 0);
    expect(after[2]).toBeLessThan(before[2] ?? 1);
  });

  it('degrades gracefully when every weight is zero', () => {
    expect(sum(sharpenWeights([0, 0, 0], 7))).toBeCloseTo(1, 10);
  });
});

describe('reoriented normal mapping', () => {
  const FLAT: Vec3Tuple = [0, 0, 1];

  it('is the identity when the detail normal is flat', () => {
    const base: Vec3Tuple = [0.3, -0.4, Math.sqrt(1 - 0.09 - 0.16)];
    const out = reorientNormal(base, FLAT);
    expect(out[0]).toBeCloseTo(base[0], 6);
    expect(out[1]).toBeCloseTo(base[1], 6);
    expect(out[2]).toBeCloseTo(base[2], 6);
  });

  it('returns the detail normal when the base is flat', () => {
    const detail: Vec3Tuple = [-0.5, 0.2, Math.sqrt(1 - 0.25 - 0.04)];
    const out = reorientNormal(FLAT, detail);
    expect(out[0]).toBeCloseTo(detail[0], 6);
    expect(out[1]).toBeCloseTo(detail[1], 6);
    expect(out[2]).toBeCloseTo(detail[2], 6);
  });

  it('always returns a unit vector', () => {
    for (const base of [
      [0.7, 0.1, Math.sqrt(1 - 0.49 - 0.01)],
      [-0.2, -0.6, Math.sqrt(1 - 0.04 - 0.36)],
    ] as Vec3Tuple[]) {
      for (const detail of [
        [0.4, 0.4, Math.sqrt(1 - 0.32)],
        [-0.9, 0.1, Math.sqrt(1 - 0.81 - 0.01)],
      ] as Vec3Tuple[]) {
        const out = reorientNormal(base, detail);
        expect(Math.hypot(...out)).toBeCloseTo(1, 6);
      }
    }
  });

  it('keeps detail slope on a steep base, unlike a naive linear blend', () => {
    // The whole reason to prefer RNM: on a steep base normal, linear blending
    // and normalising crushes the detail's contribution towards nothing.
    const steep: Vec3Tuple = [0.9, 0, Math.sqrt(1 - 0.81)];
    const detail: Vec3Tuple = [0, 0.5, Math.sqrt(1 - 0.25)];
    const rnm = reorientNormal(steep, detail);
    const linear = ((): Vec3Tuple => {
      const v: Vec3Tuple = [steep[0] + detail[0], steep[1] + detail[1], steep[2] + detail[2]];
      const len = Math.hypot(...v);
      return [v[0] / len, v[1] / len, v[2] / len];
    })();
    expect(Math.abs(rnm[1])).toBeGreaterThan(Math.abs(linear[1]));
  });
});

describe('normal strength scaling', () => {
  it('returns a unit vector for any strength', () => {
    const n: Vec3Tuple = [0.3, 0.2, Math.sqrt(1 - 0.09 - 0.04)];
    for (const s of [0, 0.5, 1, 2, 8]) {
      expect(Math.hypot(...scaleNormal(n, s))).toBeCloseTo(1, 6);
    }
  });

  it('flattens to +Z at strength zero', () => {
    const out = scaleNormal([0.6, -0.3, Math.sqrt(1 - 0.36 - 0.09)], 0);
    expect(out[2]).toBeCloseTo(1, 6);
  });

  it('increases slope monotonically with strength', () => {
    const n: Vec3Tuple = [0.3, 0, Math.sqrt(1 - 0.09)];
    const slopes = [0.5, 1, 2, 4].map((s) => {
      const o = scaleNormal(n, s);
      return Math.hypot(o[0], o[1]) / o[2];
    });
    for (let i = 1; i < slopes.length; i++) {
      expect(slopes[i]).toBeGreaterThan(slopes[i - 1] ?? 0);
    }
  });
});

describe('triplanar weights', () => {
  it('sum to one for every normal', () => {
    for (const n of [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0.577, 0.577, 0.577],
      [-0.3, 0.8, 0.5],
    ] as Vec3Tuple[]) {
      expect(sum(triplanarWeights(n, 6))).toBeCloseTo(1, 10);
    }
  });

  it('is exactly one-hot for an axis-aligned normal', () => {
    // Flat ground is the common case, and cross-fading it with two stretched
    // projections would cost contrast everywhere for nothing.
    expect(triplanarWeights([0, 1, 0], 4)).toEqual([0, 1, 0]);
  });

  it('is symmetric in the sign of the normal', () => {
    const a = triplanarWeights([0.4, -0.8, 0.2], 6);
    const b = triplanarWeights([-0.4, 0.8, -0.2], 6);
    expect(a).toEqual(b);
  });

  it('narrows the transition band as sharpness rises', () => {
    const n: Vec3Tuple = [0.6, 0.8, 0];
    const soft = triplanarWeights(n, 2);
    const hard = triplanarWeights(n, 16);
    expect(hard[1]).toBeGreaterThan(soft[1] ?? 0);
  });

  it('degrades to the Y projection for a degenerate normal', () => {
    expect(triplanarWeights([0, 0, 0], 6)).toEqual([0, 1, 0]);
  });
});

describe('triplanar UVs', () => {
  it('mirrors the U axis with the sign of the projection normal', () => {
    const pos: Vec3Tuple = [1, 2, 3];
    const plus = triplanarUVs(pos, [1, 0, 0]);
    const minus = triplanarUVs(pos, [-1, 0, 0]);
    // Without this flip, the +X and -X faces of a rock sample mirror-image
    // texture and any directional detail reverses across the silhouette.
    expect(plus.uvX[0]).toBeCloseTo(-minus.uvX[0], 10);
    expect(plus.uvX[1]).toBeCloseTo(minus.uvX[1], 10);
  });

  it('projects each plane along the two axes it spans', () => {
    const { uvX, uvY, uvZ } = triplanarUVs([1, 2, 3], [1, 1, 1]);
    expect(uvX).toEqual([3, 2]);
    expect(uvY).toEqual([1, 3]);
    expect(uvZ).toEqual([-1, 2]);
  });
});

describe('detail fades', () => {
  it('is fully on at the camera and fully off past the end', () => {
    expect(detailDistanceFade(0, 4, 12)).toBe(1);
    expect(detailDistanceFade(4, 4, 12)).toBe(1);
    expect(detailDistanceFade(12, 4, 12)).toBe(0);
    expect(detailDistanceFade(40, 4, 12)).toBe(0);
  });

  it('is monotonically decreasing with distance', () => {
    let previous = 1;
    for (let d = 0; d <= 20; d += 0.5) {
      const value = detailDistanceFade(d, 4, 12);
      expect(value).toBeLessThanOrEqual(previous + 1e-12);
      previous = value;
    }
  });

  it('has zero derivative at both ends, so no ring appears on the ground', () => {
    const eps = 1e-4;
    const nearStart = (detailDistanceFade(4 + eps, 4, 12) - detailDistanceFade(4, 4, 12)) / eps;
    const nearEnd = (detailDistanceFade(12, 4, 12) - detailDistanceFade(12 - eps, 4, 12)) / eps;
    expect(Math.abs(nearStart)).toBeLessThan(1e-3);
    expect(Math.abs(nearEnd)).toBeLessThan(1e-3);
  });

  it('treats a degenerate range as "no detail"', () => {
    expect(detailDistanceFade(1, 8, 8)).toBe(0);
    expect(detailDistanceFade(1, 12, 4)).toBe(0);
  });

  it('removes detail before it reaches one cycle per pixel', () => {
    expect(detailDensityFade(0.1)).toBe(1);
    expect(detailDensityFade(0.5)).toBe(0);
    // Nyquist is half a cycle per pixel; the layer must already be gone there.
    expect(detailDensityFade(0.5)).toBeLessThan(detailDensityFade(0.3));
  });

  it('compounds the two terms multiplicatively', () => {
    const combined = detailFade(6, 0.3, 4, 12);
    expect(combined).toBeCloseTo(detailDistanceFade(6, 4, 12) * detailDensityFade(0.3), 12);
  });
});

describe('macro variation', () => {
  it('is energy neutral over a symmetric noise distribution', () => {
    // The archetype table's albedo tints are authored against a reference look;
    // a variation term with a mean other than 1 would make every one of them
    // wrong by the same hidden factor.
    const samples = 4001;
    let total = 0;
    for (let i = 0; i < samples; i++) {
      const s = (i / (samples - 1)) * 2 - 1;
      total += macroAlbedoGain(s, 0.3);
    }
    expect(total / samples).toBeCloseTo(macroAlbedoGainMean(), 6);
  });

  it('spans exactly the requested peak-to-peak range', () => {
    expect(macroAlbedoGain(1, 0.3) - macroAlbedoGain(-1, 0.3)).toBeCloseTo(0.3, 10);
  });

  it('never produces a negative gain, however extreme the parameters', () => {
    expect(macroAlbedoGain(-1, 8)).toBeGreaterThan(0);
  });

  it('offsets roughness symmetrically around zero', () => {
    expect(macroRoughnessDelta(0, 0.2)).toBe(0);
    expect(macroRoughnessDelta(1, 0.2)).toBeCloseTo(-macroRoughnessDelta(-1, 0.2), 12);
  });

  it('tints only on the positive lobe, so moss reads as patches', () => {
    expect(macroTintWeight(-1, 0.5)).toBe(0);
    expect(macroTintWeight(-0.2, 0.5)).toBe(0);
    expect(macroTintWeight(1, 0.5)).toBeCloseTo(0.5, 12);
  });

  it('concentrates the tint quadratically rather than hazing everything', () => {
    expect(macroTintWeight(0.5, 1)).toBeCloseTo(0.25, 12);
  });
});
