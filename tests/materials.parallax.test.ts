/**
 * Tests for the parallax march and the hex-tiling lattice.
 *
 * Neither ray march can be run without a GPU, but everything that decides what
 * the march *means* can be, and those are the parts that produce silently wrong
 * images: an adaptive step count that goes the wrong way shimmers at grazing
 * angles, an offset limiter that fails to bound the step smears the surface
 * edge-on, and a crossing solve that is off by an interval terraces the relief.
 *
 * For hex-tiling, the lattice is the whole technique. Weights that do not
 * partition unity change the surface brightness along every triangle edge;
 * weights that are not one-hot at the lattice vertices leave the texture in a
 * permanent three-way blend and destroy its contrast everywhere.
 */

import { describe, expect, it } from 'vitest';

import {
  HEX_SCALE_X,
  hexHashOffset,
  hexTriangleGrid,
  sharpenHexWeights,
  varianceBlend,
  varianceRestoreFactor,
} from '../src/render/materials/AntiTile';
import {
  parallaxCrossing,
  parallaxStepCount,
  parallaxStepUV,
  scaledParallaxSteps,
} from '../src/render/materials/Parallax';

describe('parallax step count', () => {
  it('uses the minimum head-on and the maximum edge-on', () => {
    expect(parallaxStepCount(1, 8, 24)).toBeCloseTo(8, 12);
    expect(parallaxStepCount(0, 8, 24)).toBeCloseTo(24, 12);
  });

  it('interpolates monotonically', () => {
    let previous = parallaxStepCount(0, 8, 24);
    for (let c = 0.1; c <= 1.0001; c += 0.1) {
      const value = parallaxStepCount(c, 8, 24);
      expect(value).toBeLessThanOrEqual(previous + 1e-12);
      previous = value;
    }
  });

  it('is symmetric in the sign of N·V, so back faces behave', () => {
    expect(parallaxStepCount(-0.4, 8, 24)).toBeCloseTo(parallaxStepCount(0.4, 8, 24), 12);
  });
});

describe('parallax offset limiting', () => {
  it('scales the step with the height volume depth', () => {
    const a = parallaxStepUV([0.5, 0, 0.866], 0.04, 16);
    const b = parallaxStepUV([0.5, 0, 0.866], 0.08, 16);
    expect(b[0]).toBeCloseTo(a[0] * 2, 12);
  });

  it('bounds the total displacement at grazing incidence', () => {
    // Without the limiter the step diverges as V.z -> 0 and the surface smears.
    const layers = 16;
    const scale = 0.05;
    const grazing = parallaxStepUV([1, 0, 1e-6], scale, layers);
    const total = Math.abs(grazing[0]) * layers;
    // 1 / 0.35 is the clamp, so the whole march can travel at most ~2.9x scale.
    expect(total).toBeLessThanOrEqual((scale / 0.35) * 1.0001);
  });

  it('is unaffected by the limiter well away from grazing', () => {
    const [dx] = parallaxStepUV([0.3, 0, 0.9], 0.04, 10);
    expect(dx).toBeCloseTo((0.3 / 0.9) * (0.04 / 10), 12);
  });

  it('shrinks the step as the layer count rises', () => {
    const coarse = parallaxStepUV([0.4, 0.2, 0.8], 0.04, 8);
    const fine = parallaxStepUV([0.4, 0.2, 0.8], 0.04, 32);
    expect(Math.abs(fine[0])).toBeCloseTo(Math.abs(coarse[0]) / 4, 12);
  });
});

describe('parallax crossing solve', () => {
  it('recovers the exact crossing of two linear segments', () => {
    // Ray depth goes 0.4 -> 0.6 while the surface goes 0.6 -> 0.4: they cross
    // exactly halfway, i.e. 0.5 of the way back from the current sample.
    expect(parallaxCrossing(0.4, 0.6, 0.6, 0.4)).toBeCloseTo(0.5, 12);
    // And an asymmetric case solved by hand: ray 0.4 -> 0.6, surface 0.7 -> 0.5
    // meet at 0.75 of the way forward, so 0.25 of the way back.
    expect(parallaxCrossing(0.4, 0.7, 0.6, 0.5)).toBeCloseTo(0.25, 12);
  });

  it('returns 0 when the crossing is exactly at the current sample', () => {
    expect(parallaxCrossing(0.4, 0.7, 0.6, 0.6)).toBeCloseTo(0, 12);
  });

  it('returns 1 when the crossing is exactly at the previous sample', () => {
    expect(parallaxCrossing(0.4, 0.4, 0.6, 0.2)).toBeCloseTo(1, 12);
  });

  it('stays inside [0,1]', () => {
    for (const [pr, ph, cr, ch] of [
      [0, 1, 0.1, 0.9],
      [0.9, 0.95, 1.0, 0.1],
      [0.2, 0.21, 0.4, 0.39],
    ] as [number, number, number, number][]) {
      const t = parallaxCrossing(pr, ph, cr, ch);
      expect(t).toBeGreaterThanOrEqual(0);
      expect(t).toBeLessThanOrEqual(1);
    }
  });

  it('degrades safely when the two samples are identical', () => {
    expect(parallaxCrossing(0, 0.5, 0, 0.5)).toBe(0);
  });
});

describe('parallax quality scaling', () => {
  it('never returns fewer than one step', () => {
    expect(scaledParallaxSteps(24, 0)).toBe(1);
    expect(scaledParallaxSteps(1, 0.01)).toBe(1);
  });

  it('caps the march so a bad setting cannot melt the shader', () => {
    expect(scaledParallaxSteps(24, 100)).toBe(64);
  });

  it('scales linearly in between', () => {
    expect(scaledParallaxSteps(24, 0.5)).toBe(12);
    expect(scaledParallaxSteps(24, 1.5)).toBe(36);
  });
});

describe('hex-tiling lattice', () => {
  const samples: [number, number][] = [];
  for (let i = 0; i < 37; i++) {
    for (let j = 0; j < 41; j++) {
      samples.push([(i * 0.317) % 7, (j * 0.211) % 5]);
    }
  }

  it('produces weights that sum to one everywhere', () => {
    for (const [s, t] of samples) {
      const grid = hexTriangleGrid(s, t);
      const total = grid.weights[0] + grid.weights[1] + grid.weights[2];
      expect(total).toBeCloseTo(1, 10);
    }
  });

  it('never produces a negative weight', () => {
    for (const [s, t] of samples) {
      for (const w of hexTriangleGrid(s, t).weights) {
        expect(w).toBeGreaterThanOrEqual(-1e-12);
      }
    }
  });

  it('is one-hot at a lattice vertex', () => {
    // Invert the skew to land exactly on the lattice point (0, 0): a point on a
    // vertex must be a single un-blended sample, or the texture is permanently
    // averaged with two neighbours and loses its contrast.
    const grid = hexTriangleGrid(0, 0);
    const sorted = [...grid.weights].sort((a, b) => b - a);
    expect(sorted[0]).toBeCloseTo(1, 10);
    expect(sorted[1]).toBeCloseTo(0, 10);
  });

  it('returns three distinct lattice vertices', () => {
    for (const [s, t] of samples.slice(0, 200)) {
      const { vertices } = hexTriangleGrid(s, t);
      const keys = new Set(vertices.map((v) => `${v[0]},${v[1]}`));
      expect(keys.size).toBe(3);
    }
  });

  it('is continuous across a triangle boundary', () => {
    // The two triangles of a rhombus meet along the anti-diagonal; the weights
    // must agree in the limit or a hard seam appears along every lattice edge.
    const eps = 1e-7;
    // A point straddling the diagonal in skewed space.
    const s = 0.5 / HEX_SCALE_X;
    const t = 0.5 / (2 / 3) / 1.1547005383792515;
    const a = hexTriangleGrid(s - eps, t - eps);
    const b = hexTriangleGrid(s + eps, t + eps);
    const maxA = Math.max(...a.weights);
    const maxB = Math.max(...b.weights);
    expect(Math.abs(maxA - maxB)).toBeLessThan(1e-4);
  });

  it('moves through the lattice as the input grows', () => {
    const near = hexTriangleGrid(0.1, 0.1).vertices[0];
    const far = hexTriangleGrid(20, 20).vertices[0];
    expect(`${far[0]},${far[1]}`).not.toBe(`${near[0]},${near[1]}`);
  });
});

describe('hex-tiling hash', () => {
  it('is deterministic', () => {
    expect(hexHashOffset([3, -7])).toEqual(hexHashOffset([3, -7]));
  });

  it('stays inside the unit square', () => {
    for (let x = -40; x <= 40; x += 7) {
      for (let y = -40; y <= 40; y += 3) {
        const [a, b] = hexHashOffset([x, y]);
        expect(a).toBeGreaterThanOrEqual(0);
        expect(a).toBeLessThan(1);
        expect(b).toBeGreaterThanOrEqual(0);
        expect(b).toBeLessThan(1);
      }
    }
  });

  it('decorrelates neighbouring lattice cells', () => {
    // Adjacent cells must not land on nearly the same offset, or the lattice
    // reintroduces exactly the repetition it exists to remove.
    let collisions = 0;
    for (let x = -20; x <= 20; x++) {
      for (let y = -20; y <= 20; y++) {
        const a = hexHashOffset([x, y]);
        const b = hexHashOffset([x + 1, y]);
        if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.05) collisions++;
      }
    }
    expect(collisions).toBeLessThan(20);
  });

  it('spreads roughly uniformly over the unit square', () => {
    const bins = new Array<number>(16).fill(0);
    let count = 0;
    for (let x = 0; x < 64; x++) {
      for (let y = 0; y < 64; y++) {
        const [a, b] = hexHashOffset([x, y]);
        const bin = Math.min(3, Math.floor(a * 4)) * 4 + Math.min(3, Math.floor(b * 4));
        bins[bin] = (bins[bin] ?? 0) + 1;
        count++;
      }
    }
    const expected = count / 16;
    for (const n of bins) {
      expect(n).toBeGreaterThan(expected * 0.7);
      expect(n).toBeLessThan(expected * 1.3);
    }
  });
});

describe('variance-preserving hex blend', () => {
  it('is neutral when one sample dominates', () => {
    expect(varianceRestoreFactor([1, 0, 0])).toBeCloseTo(1, 12);
  });

  it('restores exactly the deviation that averaging n samples destroyed', () => {
    expect(varianceRestoreFactor([1 / 3, 1 / 3, 1 / 3])).toBeCloseTo(Math.sqrt(3), 12);
    expect(varianceRestoreFactor([0.5, 0.5])).toBeCloseTo(Math.sqrt(2), 12);
  });

  it('leaves the mean exactly where it was', () => {
    const mean = 0.37;
    expect(varianceBlend([mean, mean, mean], [1 / 3, 1 / 3, 1 / 3], mean)).toBeCloseTo(mean, 12);
  });

  it('keeps contrast that a plain average would flatten', () => {
    const samples = [0.9, 0.3, 0.45];
    const weights = [1 / 3, 1 / 3, 1 / 3];
    const mean = 0.5;
    const plain = samples.reduce((a, s, i) => a + s * (weights[i] ?? 0), 0);
    const restored = varianceBlend(samples, weights, mean);
    expect(Math.abs(restored - mean)).toBeGreaterThan(Math.abs(plain - mean));
  });

  it('reduces to the sample itself at a lattice vertex', () => {
    expect(varianceBlend([0.9, 0.2, 0.4], [1, 0, 0], 0.5)).toBeCloseTo(0.9, 12);
  });
});

describe('hex weight sharpening', () => {
  it('preserves normalisation', () => {
    const w = sharpenHexWeights([0.2, 0.5, 0.3], 7);
    expect(w[0] + w[1] + w[2]).toBeCloseTo(1, 12);
  });

  it('narrows the three-way blend region as the exponent rises', () => {
    const soft = sharpenHexWeights([0.4, 0.35, 0.25], 1);
    const hard = sharpenHexWeights([0.4, 0.35, 0.25], 7);
    expect(hard[0]).toBeGreaterThan(soft[0] ?? 0);
    expect(hard[2]).toBeLessThan(soft[2] ?? 1);
  });
});
