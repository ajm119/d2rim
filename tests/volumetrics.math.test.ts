/**
 * Pins the closed-form maths behind the volumetric fog.
 *
 * Everything here is the CPU twin of a line of TSL. The shaders cannot be run
 * in CI (there is no GPU in the container and no WebGPU adapter in vitest), so
 * the strategy is to keep every non-trivial formula in an exported pure
 * function, test *that*, and keep the shader a transliteration of it. Where the
 * two forms genuinely differ — the ray-march versus the closed-form optical
 * depth — the test checks that they converge, which is the property that
 * matters.
 */

import { describe, expect, it } from 'vitest';

import {
  VOLUMETRIC_TIERS,
  WORKGROUP_SIZE,
  bakeFogNoiseTexture,
  dualLobePhase,
  fogVolumeWeight,
  froxelDepthToSlice,
  froxelSliceDepth,
  haltonJitter3D,
  heightFogDensity,
  heightFogOpticalDepth,
  henyeyGreenstein,
  integrateScatteringSlice,
  radicalInverse,
  temporalBlendWeight,
  tilingFbm3D,
  tilingValueNoise3D,
} from '../src/render/Volumetrics';

/* ------------------------------------------------------------------------- *
 * Phase function
 * ------------------------------------------------------------------------- */

/**
 * Numerically integrate `p(cosθ)` over the sphere with the substitution
 * `∫_S p dω = 2π ∫_{-1}^{1} p(μ) dμ`, using the midpoint rule.
 */
function integrateOverSphere(phase: (cosTheta: number) => number, samples = 20000): number {
  let total = 0;
  for (let i = 0; i < samples; i++) {
    const mu = -1 + ((i + 0.5) / samples) * 2;
    total += phase(mu);
  }
  return (total / samples) * 2 * 2 * Math.PI;
}

describe('henyeyGreenstein', () => {
  it('is isotropic at g = 0', () => {
    for (const mu of [-1, -0.5, 0, 0.5, 1]) {
      expect(henyeyGreenstein(mu, 0)).toBeCloseTo(1 / (4 * Math.PI), 12);
    }
  });

  it('normalises to unity over the sphere', () => {
    for (const g of [-0.6, -0.2, 0, 0.3, 0.7]) {
      expect(integrateOverSphere((mu) => henyeyGreenstein(mu, g))).toBeCloseTo(1, 2);
    }
  });

  it('peaks forward for positive g and backward for negative g', () => {
    expect(henyeyGreenstein(1, 0.7)).toBeGreaterThan(henyeyGreenstein(-1, 0.7));
    expect(henyeyGreenstein(-1, -0.7)).toBeGreaterThan(henyeyGreenstein(1, -0.7));
  });

  it('stays finite at the degenerate g -> 1, cos -> 1 corner', () => {
    const value = henyeyGreenstein(1, 0.999999);
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeGreaterThan(0);
  });
});

describe('dualLobePhase', () => {
  it('reduces to a single lobe at the blend extremes', () => {
    expect(dualLobePhase(0.3, 0.7, -0.3, 0)).toBeCloseTo(henyeyGreenstein(0.3, 0.7), 12);
    expect(dualLobePhase(0.3, 0.7, -0.3, 1)).toBeCloseTo(henyeyGreenstein(0.3, -0.3), 12);
  });

  it('is still normalised, because a convex combination of unit integrals is one', () => {
    expect(integrateOverSphere((mu) => dualLobePhase(mu, 0.72, -0.28, 0.22))).toBeCloseTo(1, 2);
  });

  it('produces a backward halo the single forward lobe does not', () => {
    const single = henyeyGreenstein(-1, 0.72);
    const dual = dualLobePhase(-1, 0.72, -0.28, 0.22);
    expect(dual).toBeGreaterThan(single);
  });

  it('clamps the blend into [0, 1]', () => {
    expect(dualLobePhase(0.5, 0.7, -0.3, -5)).toBeCloseTo(dualLobePhase(0.5, 0.7, -0.3, 0), 12);
    expect(dualLobePhase(0.5, 0.7, -0.3, 5)).toBeCloseTo(dualLobePhase(0.5, 0.7, -0.3, 1), 12);
  });
});

/* ------------------------------------------------------------------------- *
 * Froxel depth distribution
 * ------------------------------------------------------------------------- */

describe('froxel depth distribution', () => {
  const near = 0.1;
  const far = 64;

  for (const distribution of ['exponential', 'quadratic'] as const) {
    describe(distribution, () => {
      it('spans exactly [near, far]', () => {
        expect(froxelSliceDepth(0, 64, near, far, distribution)).toBeCloseTo(near, 6);
        expect(froxelSliceDepth(64, 64, near, far, distribution)).toBeCloseTo(far, 4);
      });

      it('is strictly increasing', () => {
        let previous = -Infinity;
        for (let slice = 0; slice <= 64; slice++) {
          const depth = froxelSliceDepth(slice, 64, near, far, distribution);
          expect(depth).toBeGreaterThan(previous);
          previous = depth;
        }
      });

      it('round-trips through froxelDepthToSlice', () => {
        for (let slice = 0; slice <= 64; slice++) {
          const depth = froxelSliceDepth(slice, 64, near, far, distribution);
          expect(froxelDepthToSlice(depth, 64, near, far, distribution)).toBeCloseTo(slice, 4);
        }
      });

      it('clamps out-of-range depths rather than extrapolating', () => {
        expect(froxelDepthToSlice(-10, 64, near, far, distribution)).toBe(0);
        expect(froxelDepthToSlice(1e6, 64, near, far, distribution)).toBeCloseTo(64, 4);
      });
    });
  }

  it('exponential keeps a constant relative slice thickness', () => {
    // The defining property: d(k+1)/d(k) is the same for every k, which is what
    // makes froxels a roughly constant fraction of their own distance.
    const ratios: number[] = [];
    for (let slice = 0; slice < 32; slice++) {
      const a = froxelSliceDepth(slice, 32, near, far, 'exponential');
      const b = froxelSliceDepth(slice + 1, 32, near, far, 'exponential');
      ratios.push(b / a);
    }
    for (const ratio of ratios) expect(ratio).toBeCloseTo(ratios[0] as number, 10);
  });

  it('quadratic spends less of the volume on the first metre than exponential', () => {
    const quadratic = froxelSliceDepth(1, 64, near, far, 'quadratic');
    const exponential = froxelSliceDepth(1, 64, near, far, 'exponential');
    expect(quadratic).toBeGreaterThan(exponential);
  });
});

describe('froxel tier budgets', () => {
  it('keeps every shipped grid an exact multiple of the workgroup size', () => {
    // An exact multiple means no thread in the dispatch is out of range, which
    // is what lets the injection kernel skip its bounds check.
    for (const tier of Object.values(VOLUMETRIC_TIERS)) {
      const [x, y, z] = tier.froxels;
      expect((x * y * z) % WORKGROUP_SIZE).toBe(0);
      // The integration kernel dispatches one thread per column.
      expect((x * y) % WORKGROUP_SIZE).toBe(0);
    }
  });

  it('orders the tiers by cost', () => {
    const counts = (['low', 'medium', 'high', 'ultra'] as const).map((tier) => {
      const [x, y, z] = VOLUMETRIC_TIERS[tier].froxels;
      return x * y * z;
    });
    for (let i = 1; i < counts.length; i++) {
      expect(counts[i] as number).toBeGreaterThan(counts[i - 1] as number);
    }
  });
});

/* ------------------------------------------------------------------------- *
 * Height fog
 * ------------------------------------------------------------------------- */

describe('heightFogDensity', () => {
  it('is flat at and below the reference height', () => {
    expect(heightFogDensity(-100, 1.5, 0.085)).toBe(1);
    expect(heightFogDensity(1.5, 1.5, 0.085)).toBe(1);
  });

  it('decays exponentially above it', () => {
    const k = 0.085;
    expect(heightFogDensity(1.5 + 10, 1.5, k)).toBeCloseTo(Math.exp(-k * 10), 12);
  });

  it('never explodes as the camera descends', () => {
    // The whole point of the flat region: a pure exponential profile would give
    // exp(+8.5) here, and the Den of Evil would be opaque white.
    expect(heightFogDensity(-100, 1.5, 0.085)).toBeLessThanOrEqual(1);
  });
});

describe('heightFogOpticalDepth', () => {
  /** Brute-force reference: the same integral by the midpoint rule. */
  function reference(
    originY: number,
    dirY: number,
    distance: number,
    referenceHeight: number,
    falloff: number,
    steps = 200000,
  ): number {
    let total = 0;
    for (let i = 0; i < steps; i++) {
      const t = ((i + 0.5) / steps) * distance;
      total += heightFogDensity(originY + dirY * t, referenceHeight, falloff);
    }
    return (total / steps) * distance;
  }

  const cases: Array<[string, number, number, number]> = [
    ['horizontal below the layer', 0, 0, 120],
    ['horizontal above the layer', 40, 0, 120],
    ['rising, entirely above', 20, 0.8, 60],
    ['falling, entirely above', 60, -0.5, 60],
    ['rising through the reference height', -5, 0.6, 90],
    ['falling through the reference height', 30, -0.9, 90],
    ['steeply up from the ground', 0.5, 0.99, 200],
  ];

  for (const [name, originY, dirY, distance] of cases) {
    it(`matches numerical integration: ${name}`, () => {
      const analytic = heightFogOpticalDepth(originY, dirY, distance, 1.5, 0.085);
      const numeric = reference(originY, dirY, distance, 1.5, 0.085);
      expect(analytic).toBeCloseTo(numeric, 2);
    });
  }

  it('is zero for zero distance and never negative', () => {
    expect(heightFogOpticalDepth(10, 0.5, 0, 1.5, 0.085)).toBe(0);
    expect(heightFogOpticalDepth(10, 0.5, -5, 1.5, 0.085)).toBe(0);
  });

  it('degenerates to the ray length when the falloff is zero', () => {
    expect(heightFogOpticalDepth(50, 0.7, 33, 1.5, 0)).toBeCloseTo(33, 10);
  });

  it('is additive along a ray, which is what makes the tail term valid', () => {
    // τ(0, a+b) == τ(0, a) + τ(a, a+b): the property the analytic continuation
    // past the froxel volume relies on.
    const originY = 4;
    const dirY = 0.35;
    const whole = heightFogOpticalDepth(originY, dirY, 100, 1.5, 0.085);
    const first = heightFogOpticalDepth(originY, dirY, 40, 1.5, 0.085);
    const second = heightFogOpticalDepth(originY + dirY * 40, dirY, 60, 1.5, 0.085);
    expect(first + second).toBeCloseTo(whole, 8);
  });

  it('grows monotonically with distance', () => {
    let previous = -1;
    for (let d = 0; d <= 200; d += 5) {
      const value = heightFogOpticalDepth(0, 0.4, d, 1.5, 0.085);
      expect(value).toBeGreaterThanOrEqual(previous);
      previous = value;
    }
  });
});

/* ------------------------------------------------------------------------- *
 * Scattering integration
 * ------------------------------------------------------------------------- */

describe('integrateScatteringSlice', () => {
  it('tends to S·d in the optically thin limit', () => {
    const { scattering } = integrateScatteringSlice(3, 1e-9, 0.4);
    expect(scattering).toBeCloseTo(3 * 0.4, 6);
  });

  it('is continuous across the thin-limit branch', () => {
    const below = integrateScatteringSlice(3, 9.99e-7, 0.4).scattering;
    const above = integrateScatteringSlice(3, 1.01e-6, 0.4).scattering;
    expect(Math.abs(above - below)).toBeLessThan(1e-6);
  });

  it('matches the midpoint rule when the slice is optically thin', () => {
    const d = 0.05;
    const sigma = 0.01;
    const { scattering } = integrateScatteringSlice(2, sigma, d);
    expect(scattering).toBeCloseTo(2 * d, 4);
  });

  it('saturates at S/σ for an optically thick slice', () => {
    const { scattering, transmittance } = integrateScatteringSlice(5, 2, 50);
    expect(transmittance).toBeCloseTo(0, 10);
    expect(scattering).toBeCloseTo(5 / 2, 8);
  });

  it('conserves energy against a fine subdivision of the same slice', () => {
    // Splitting a homogeneous slice into N sub-slices and composing them
    // front-to-back must give exactly the same answer. A midpoint rule fails
    // this; the analytic form passes it, which is why banding disappears.
    const sigma = 0.6;
    const scatteringCoefficient = 1.7;
    const length = 4;

    const whole = integrateScatteringSlice(scatteringCoefficient, sigma, length);

    let accumulated = 0;
    let transmittance = 1;
    const parts = 37;
    for (let i = 0; i < parts; i++) {
      const slice = integrateScatteringSlice(scatteringCoefficient, sigma, length / parts);
      accumulated += transmittance * slice.scattering;
      transmittance *= slice.transmittance;
    }

    expect(accumulated).toBeCloseTo(whole.scattering, 10);
    expect(transmittance).toBeCloseTo(whole.transmittance, 10);
  });

  it('reports Beer-Lambert transmittance', () => {
    expect(integrateScatteringSlice(0, 0.25, 8).transmittance).toBeCloseTo(Math.exp(-2), 12);
  });
});

/* ------------------------------------------------------------------------- *
 * Local fog volumes
 * ------------------------------------------------------------------------- */

describe('fogVolumeWeight', () => {
  const center = { x: 10, y: 2, z: -4 };
  const extent = { x: 6, y: 3, z: 6 };

  it('is 1 at the centre of a box and 0 outside it', () => {
    expect(fogVolumeWeight('box', center, center, extent, 0.4)).toBeCloseTo(1, 10);
    expect(fogVolumeWeight('box', { x: 30, y: 2, z: -4 }, center, extent, 0.4)).toBe(0);
  });

  it('is 1 at the centre of a sphere and 0 outside it', () => {
    expect(fogVolumeWeight('sphere', center, center, extent, 0.4)).toBeCloseTo(1, 10);
    expect(fogVolumeWeight('sphere', { x: 30, y: 2, z: -4 }, center, extent, 0.4)).toBe(0);
  });

  it('reaches exactly zero at the boundary, so there is no hard edge', () => {
    const onFace = { x: center.x + extent.x, y: center.y, z: center.z };
    expect(fogVolumeWeight('box', onFace, center, extent, 0.4)).toBe(0);
  });

  it('is monotonically decreasing outward', () => {
    let previous = Infinity;
    for (let dx = 0; dx <= extent.x; dx += 0.25) {
      const weight = fogVolumeWeight(
        'box',
        { x: center.x + dx, y: center.y, z: center.z },
        center,
        extent,
        0.4,
      );
      expect(weight).toBeLessThanOrEqual(previous + 1e-12);
      previous = weight;
    }
  });

  it('has a zero derivative at both ends of the ramp', () => {
    // Smoothstep, not a linear ramp: a linear falloff leaves a Mach band at
    // both ends of the fade that reads as a visible box edge.
    const feather = 0.5;
    const half = extent.x;
    const start = half * (1 - feather);
    const epsilon = 1e-4;
    const sampleAt = (dx: number): number =>
      fogVolumeWeight('box', { x: center.x + dx, y: center.y, z: center.z }, center, extent, feather);

    const slopeAtOuterEdge = (sampleAt(half - epsilon) - sampleAt(half - 2 * epsilon)) / epsilon;
    const slopeAtInnerEdge = (sampleAt(start + 2 * epsilon) - sampleAt(start + epsilon)) / epsilon;
    expect(Math.abs(slopeAtOuterEdge)).toBeLessThan(1e-2);
    expect(Math.abs(slopeAtInnerEdge)).toBeLessThan(1e-2);
  });

  it('produces rounded box corners, because the axis ramps multiply', () => {
    const corner = fogVolumeWeight(
      'box',
      { x: center.x + extent.x * 0.9, y: center.y + extent.y * 0.9, z: center.z + extent.z * 0.9 },
      center,
      extent,
      0.4,
    );
    const face = fogVolumeWeight(
      'box',
      { x: center.x + extent.x * 0.9, y: center.y, z: center.z },
      center,
      extent,
      0.4,
    );
    expect(corner).toBeLessThan(face);
  });

  it('tolerates a degenerate feather without dividing by zero', () => {
    const weight = fogVolumeWeight('box', center, center, extent, 0);
    expect(Number.isFinite(weight)).toBe(true);
  });
});

/* ------------------------------------------------------------------------- *
 * Temporal sampling
 * ------------------------------------------------------------------------- */

describe('radicalInverse / haltonJitter3D', () => {
  it('reproduces the base-2 van der Corput sequence', () => {
    expect(radicalInverse(1, 2)).toBeCloseTo(0.5, 12);
    expect(radicalInverse(2, 2)).toBeCloseTo(0.25, 12);
    expect(radicalInverse(3, 2)).toBeCloseTo(0.75, 12);
    expect(radicalInverse(4, 2)).toBeCloseTo(0.125, 12);
  });

  it('stays inside [0, 1)', () => {
    for (let i = 0; i < 512; i++) {
      for (const base of [2, 3, 5]) {
        const value = radicalInverse(i, base);
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThan(1);
      }
    }
  });

  it('gives well-distributed jitter over one 16-frame cycle', () => {
    // The froxel jitter cycles every 16 frames; over that cycle each axis must
    // cover its range reasonably evenly, or the temporal average is biased and
    // the fog sits slightly off from where the analytic model puts it.
    const buckets = [0, 0, 0, 0];
    for (let frame = 1; frame <= 16; frame++) {
      const [x] = haltonJitter3D(frame);
      buckets[Math.min(3, Math.floor(x * 4))]!++;
    }
    for (const count of buckets) expect(count).toBeGreaterThanOrEqual(3);
  });

  it('never repeats a point within a cycle', () => {
    const seen = new Set<string>();
    for (let frame = 1; frame <= 16; frame++) {
      const key = haltonJitter3D(frame).map((v) => v.toFixed(6)).join(',');
      expect(seen.has(key)).toBe(false);
      seen.add(key);
    }
  });
});

describe('temporalBlendWeight', () => {
  it('takes the new sample whole when there is no history', () => {
    expect(temporalBlendWeight(0.06, 0, false)).toBe(1);
    expect(temporalBlendWeight(0.06, 1, false)).toBe(1);
  });

  it('settles to minAlpha when the signal is stable', () => {
    expect(temporalBlendWeight(0.06, 0, true)).toBeCloseTo(0.06, 12);
  });

  it('accelerates towards 1 as the disagreement grows', () => {
    const weights = [0, 0.25, 0.5, 0.75, 1].map((change) =>
      temporalBlendWeight(0.06, change, true),
    );
    for (let i = 1; i < weights.length; i++) {
      expect(weights[i] as number).toBeGreaterThan(weights[i - 1] as number);
    }
    expect(weights.at(-1)).toBeCloseTo(1, 12);
  });

  it('responds quadratically, so sensor noise does not defeat the filter', () => {
    // A 10% disagreement must move alpha by far less than 10% of its range,
    // otherwise the march's own noise keeps resetting the history.
    const small = temporalBlendWeight(0.06, 0.1, true);
    expect(small).toBeLessThan(0.07);
  });

  it('clamps out-of-range inputs', () => {
    expect(temporalBlendWeight(-1, 5, true)).toBeLessThanOrEqual(1);
    expect(temporalBlendWeight(2, -5, true)).toBeLessThanOrEqual(1);
  });
});

/* ------------------------------------------------------------------------- *
 * Noise
 * ------------------------------------------------------------------------- */

describe('tilingValueNoise3D', () => {
  it('is deterministic for a given seed', () => {
    expect(tilingValueNoise3D(1.3, 2.7, 0.4, 4, 12)).toBe(
      tilingValueNoise3D(1.3, 2.7, 0.4, 4, 12),
    );
  });

  it('differs between seeds', () => {
    expect(tilingValueNoise3D(1.3, 2.7, 0.4, 4, 12)).not.toBe(
      tilingValueNoise3D(1.3, 2.7, 0.4, 4, 13),
    );
  });

  it('tiles exactly over the lattice period', () => {
    const period = 4;
    for (const [x, y, z] of [
      [0.3, 1.7, 2.2],
      [3.9, 0.05, 1.5],
      [2.5, 2.5, 2.5],
    ]) {
      const base = tilingValueNoise3D(x!, y!, z!, period, 7);
      expect(tilingValueNoise3D(x! + period, y!, z!, period, 7)).toBeCloseTo(base, 12);
      expect(tilingValueNoise3D(x!, y! + period, z!, period, 7)).toBeCloseTo(base, 12);
      expect(tilingValueNoise3D(x!, y!, z! + period, period, 7)).toBeCloseTo(base, 12);
      expect(tilingValueNoise3D(x! - period, y! - period, z! - period, period, 7)).toBeCloseTo(
        base,
        12,
      );
    }
  });

  it('stays inside [0, 1]', () => {
    for (let i = 0; i < 4000; i++) {
      const value = tilingValueNoise3D(i * 0.137, i * 0.311, i * 0.079, 4, 3);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('is continuous: neighbouring samples differ by little', () => {
    let maxDelta = 0;
    for (let i = 0; i < 2000; i++) {
      const x = i * 0.011;
      const a = tilingValueNoise3D(x, 0.5, 0.5, 4, 3);
      const b = tilingValueNoise3D(x + 0.001, 0.5, 0.5, 4, 3);
      maxDelta = Math.max(maxDelta, Math.abs(a - b));
    }
    expect(maxDelta).toBeLessThan(0.02);
  });

  it('is exactly the lattice value at integer coordinates', () => {
    // The quintic fade is 0 at t=0 and 1 at t=1, so a lattice point must
    // interpolate to itself. This is what makes the tiling test above exact
    // rather than approximate.
    const atLattice = tilingValueNoise3D(2, 3, 1, 4, 9);
    const nearLattice = tilingValueNoise3D(2 + 1e-9, 3, 1, 4, 9);
    expect(nearLattice).toBeCloseTo(atLattice, 9);
  });
});

describe('tilingFbm3D', () => {
  it('stays inside [0, 1] and tiles over the base period', () => {
    const period = 4;
    for (let i = 0; i < 500; i++) {
      const x = i * 0.173;
      const y = i * 0.091;
      const z = i * 0.233;
      const value = tilingFbm3D(x, y, z, period, 3, 5);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
      expect(tilingFbm3D(x + period, y, z, period, 3, 5)).toBeCloseTo(value, 10);
    }
  });

  it('has a mean near 0.5, so the density remap is unbiased', () => {
    // `mediaAt` remaps the noise as `1 + strength·(2n − 1)`, which preserves
    // the calibrated mean density only if `E[n] ≈ 0.5`.
    let total = 0;
    const samples = 20000;
    for (let i = 0; i < samples; i++) {
      total += tilingFbm3D(i * 0.0173, i * 0.0291, i * 0.0433, 4, 3, 11);
    }
    expect(total / samples).toBeCloseTo(0.5, 1);
  });

  it('adds detail with each octave without changing the range', () => {
    const one = tilingFbm3D(1.1, 2.2, 3.3, 4, 1, 2);
    const three = tilingFbm3D(1.1, 2.2, 3.3, 4, 3, 2);
    expect(one).not.toBeCloseTo(three, 6);
    expect(three).toBeGreaterThanOrEqual(0);
    expect(three).toBeLessThanOrEqual(1);
  });
});

describe('bakeFogNoiseTexture', () => {
  it('produces a wrapping, linearly filtered RGBA volume of the requested size', () => {
    const texture = bakeFogNoiseTexture(8, 'test');
    expect(texture.image.width).toBe(8);
    expect(texture.image.height).toBe(8);
    expect(texture.image.depth).toBe(8);
    expect((texture.image.data as Uint8Array).length).toBe(8 * 8 * 8 * 4);
    expect(texture.wrapS).toBe(1000); // THREE.RepeatWrapping
    expect(texture.wrapT).toBe(1000);
    expect(texture.wrapR).toBe(1000);
    texture.dispose();
  });

  it('fills every channel with real signal rather than a constant', () => {
    const texture = bakeFogNoiseTexture(16, 'd2rim.fog');
    const data = texture.image.data as Uint8Array;
    for (let channel = 0; channel < 4; channel++) {
      let low = 255;
      let high = 0;
      for (let i = channel; i < data.length; i += 4) {
        const value = data[i] as number;
        low = Math.min(low, value);
        high = Math.max(high, value);
      }
      expect(high - low).toBeGreaterThan(40);
    }
    texture.dispose();
  });

  it('is deterministic for a given seed', () => {
    const a = bakeFogNoiseTexture(8, 'seed-a');
    const b = bakeFogNoiseTexture(8, 'seed-a');
    const c = bakeFogNoiseTexture(8, 'seed-b');
    expect(Array.from(a.image.data as Uint8Array)).toEqual(Array.from(b.image.data as Uint8Array));
    expect(Array.from(a.image.data as Uint8Array)).not.toEqual(
      Array.from(c.image.data as Uint8Array),
    );
    a.dispose();
    b.dispose();
    c.dispose();
  });

  it('refuses to bake a degenerate volume', () => {
    const texture = bakeFogNoiseTexture(1, 'tiny');
    expect(texture.image.width).toBeGreaterThanOrEqual(4);
    texture.dispose();
  });
});

/* ------------------------------------------------------------------------- *
 * Cross-checks between the two sampling strategies
 * ------------------------------------------------------------------------- */

describe('ray march versus closed form', () => {
  /**
   * March the height-fog profile exactly the way the shader does — the
   * exponential slice distribution, the analytic slice integral — and check
   * that the transmittance converges on the closed form.
   *
   * This is the property that makes the froxel path and the WebGL2 path look
   * the same: they use different sampling, but both converge on the same
   * integral, and they converge fast enough that 32 steps is already within a
   * fraction of a percent.
   */
  function marchTransmittance(
    originY: number,
    dirY: number,
    distance: number,
    density: number,
    steps: number,
    offset = 0.5,
  ): number {
    const near = 0.1;
    let transmittance = 1;
    let previous = near;
    for (let i = 0; i < steps; i++) {
      const sliceFar = froxelSliceDepth(i + 1, steps, near, distance, 'exponential');
      const stepLength = sliceFar - previous;
      previous = sliceFar;
      const sampleDepth = froxelSliceDepth(i + offset, steps, near, distance, 'exponential');
      const sigma = density * heightFogDensity(originY + dirY * sampleDepth, 1.5, 0.085);
      transmittance *= Math.exp(-sigma * stepLength);
    }
    return transmittance;
  }

  /**
   * The march covers `[near, far]`, so the reference integral has to start at
   * the near plane too. The 10 cm in front of the camera that the march skips
   * is worth `exp(-0.0014) ≈ 0.14%` of transmittance at the default density —
   * enough to swamp a convergence test, and correctly excluded here because it
   * is also excluded from the frame (nothing is drawn in front of the near
   * plane).
   */
  function analyticTransmittance(
    originY: number,
    dirY: number,
    near: number,
    far: number,
    density: number,
  ): number {
    const tau = heightFogOpticalDepth(originY + dirY * near, dirY, far - near, 1.5, 0.085);
    return Math.exp(-density * tau);
  }

  it('converges on the analytic transmittance as steps increase', () => {
    const density = 0.014;
    const analytic = analyticTransmittance(2, 0.25, 0.1, 64, density);

    const coarse = marchTransmittance(2, 0.25, 64, density, 8);
    const fine = marchTransmittance(2, 0.25, 64, density, 256);

    expect(Math.abs(fine - analytic)).toBeLessThan(Math.abs(coarse - analytic) + 1e-9);
    expect(fine).toBeCloseTo(analytic, 4);
  });

  it('is already within a tenth of a percent at the shipping step count', () => {
    const density = 0.014;
    const analytic = analyticTransmittance(2, 0.25, 0.1, 64, density);
    const marched = marchTransmittance(2, 0.25, 64, density, VOLUMETRIC_TIERS.high.marchSteps);
    expect(Math.abs(marched - analytic) / analytic).toBeLessThan(0.001);
  });

  it('is not systematically biased by the dither offset', () => {
    // The offset is a per-pixel, per-frame random in [0, 1); the temporal
    // filter averages over it, so what matters is that the *average over the
    // offset* is unbiased rather than any single offset being exact.
    const density = 0.014;
    const analytic = analyticTransmittance(2, 0.25, 0.1, 64, density);
    let total = 0;
    const trials = 64;
    for (let i = 0; i < trials; i++) {
      total += marchTransmittance(2, 0.25, 64, density, 32, (i + 0.5) / trials);
    }
    expect(Math.abs(total / trials - analytic) / analytic).toBeLessThan(0.002);
  });
});
