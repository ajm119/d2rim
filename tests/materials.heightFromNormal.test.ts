/**
 * Tests for the normal-map height integrator.
 *
 * The whole point of this module is that it recovers a height field a normal
 * map only implies, and "it looks about right" is not a usable check — a solver
 * that converged to half the correct amplitude, or to the correct shape with an
 * inverted sign, would produce parallax that is merely shallow or that moves the
 * wrong way, and both are easy to mistake for a tuning problem.
 *
 * So the solver is checked against analytic surfaces whose gradients are known
 * exactly. Two properties are pinned:
 *
 * 1. **Adjointness.** The divergence operator must be the exact negative adjoint
 *    of the gradient operator on the periodic grid, `⟨∇f, g⟩ = −⟨f, ∇·g⟩`. If it
 *    is not, the linear system being solved is not the normal equations of the
 *    least-squares problem and the answer is systematically wrong in a way no
 *    amount of extra relaxation fixes.
 * 2. **Reconstruction.** Given the exact discrete gradients of a known periodic
 *    field, the solver must return that field back, up to the constant its null
 *    space cannot determine.
 */

import { describe, expect, it } from 'vitest';

import {
  channelMeans,
  divergence,
  heightFromNormalMap,
  integrateGradients,
  normalMapToGradients,
  relax,
  type GradientField,
} from '../src/render/materials/HeightFromNormal';

/** Exact forward-difference gradients of a periodic field on a torus. */
function gradientsOf(field: Float32Array, width: number, height: number): GradientField {
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      gx[i] = (field[y * width + ((x + 1) % width)] ?? 0) - (field[i] ?? 0);
      gy[i] = (field[((y + 1) % height) * width + x] ?? 0) - (field[i] ?? 0);
    }
  }
  return { width, height, gx, gy };
}

/** A smooth, band-limited, exactly periodic test surface. */
function analyticSurface(size: number): Float32Array {
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (2 * Math.PI * x) / size;
      const v = (2 * Math.PI * y) / size;
      out[y * size + x] =
        0.35 * Math.sin(u) + 0.22 * Math.cos(2 * v) + 0.12 * Math.sin(3 * u + v);
    }
  }
  return out;
}

function normalise(field: Float32Array): Float32Array {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of field) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const range = hi - lo || 1;
  const out = new Float32Array(field.length);
  for (let i = 0; i < field.length; i++) out[i] = ((field[i] ?? 0) - lo) / range;
  return out;
}

function rms(a: Float32Array, b: Float32Array): number {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    total += d * d;
  }
  return Math.sqrt(total / a.length);
}

describe('divergence operator', () => {
  it('is the exact negative adjoint of the gradient on the torus', () => {
    const size = 16;
    const f = analyticSurface(size);
    // An arbitrary, unrelated vector field.
    const gx = new Float32Array(size * size);
    const gy = new Float32Array(size * size);
    for (let i = 0; i < gx.length; i++) {
      gx[i] = Math.sin(i * 0.37);
      gy[i] = Math.cos(i * 0.11);
    }
    const g: GradientField = { width: size, height: size, gx, gy };

    const gradF = gradientsOf(f, size, size);
    let lhs = 0;
    for (let i = 0; i < gx.length; i++) {
      lhs += (gradF.gx[i] ?? 0) * (gx[i] ?? 0) + (gradF.gy[i] ?? 0) * (gy[i] ?? 0);
    }

    const div = divergence(g);
    let rhs = 0;
    for (let i = 0; i < div.length; i++) rhs += (f[i] ?? 0) * (div[i] ?? 0);

    expect(lhs).toBeCloseTo(-rhs, 6);
  });

  it('is exactly the discrete Laplacian of the field it came from', () => {
    const size = 8;
    const f = analyticSurface(size);
    const div = divergence(gradientsOf(f, size, size));
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const laplacian =
          (f[y * size + ((x + 1) % size)] ?? 0) +
          (f[y * size + ((x + size - 1) % size)] ?? 0) +
          (f[((y + 1) % size) * size + x] ?? 0) +
          (f[((y + size - 1) % size) * size + x] ?? 0) -
          4 * (f[y * size + x] ?? 0);
        expect(div[y * size + x] ?? 0).toBeCloseTo(laplacian, 6);
      }
    }
  });

  it('vanishes on a constant field', () => {
    const size = 8;
    const constant = new Float32Array(size * size).fill(0.37);
    const div = divergence(gradientsOf(constant, size, size));
    for (const v of div) expect(v).toBeCloseTo(0, 10);
  });
});

describe('Poisson relaxation', () => {
  it('leaves the exact solution alone', () => {
    // The fixed point of Gauss-Seidel is the solution, so applying it to a
    // field that already solves the system must not move it (up to the
    // constant, which the discrete operator does not pin).
    const size = 16;
    const f = analyticSurface(size);
    const div = divergence(gradientsOf(f, size, size));
    const working = Float32Array.from(f);
    relax(working, div, size, size, 4);
    // Compare after removing the mean, since the null space is the constants.
    const meanOf = (a: Float32Array): number => a.reduce((s, v) => s + v, 0) / a.length;
    const shift = meanOf(working) - meanOf(f);
    for (let i = 0; i < f.length; i++) {
      expect((working[i] ?? 0) - shift).toBeCloseTo(f[i] ?? 0, 4);
    }
  });

  it('is independent of traversal order thanks to red-black ordering', () => {
    // Reproducibility matters: the derived height field ends up in golden-image
    // captures, so two runs must agree bit for bit.
    const size = 16;
    const f = analyticSurface(size);
    const div = divergence(gradientsOf(f, size, size));
    const a = new Float32Array(size * size);
    const b = new Float32Array(size * size);
    relax(a, div, size, size, 6);
    relax(b, div, size, size, 6);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

describe('gradient integration', () => {
  it('reconstructs a known periodic surface', () => {
    const size = 64;
    const truth = analyticSurface(size);
    const recovered = integrateGradients(gradientsOf(truth, size, size), { sweeps: 12 });
    // Both normalised, because the absolute scale and offset of an integrated
    // normal map are not determined by the data.
    expect(rms(normalise(truth), recovered)).toBeLessThan(0.02);
  });

  it('preserves the *shape* well enough for parallax, not just the average', () => {
    const size = 64;
    const truth = normalise(analyticSurface(size));
    const recovered = integrateGradients(gradientsOf(analyticSurface(size), size, size), {
      sweeps: 12,
    });
    // Pearson correlation: a solver that converged to a plausible but wrong
    // low-frequency field would still pass an RMS test on a normalised pair if
    // the amplitude happened to match, but it cannot pass this.
    const mean = (a: Float32Array): number => a.reduce((s, v) => s + v, 0) / a.length;
    const mt = mean(truth);
    const mr = mean(recovered);
    let cov = 0;
    let vt = 0;
    let vr = 0;
    for (let i = 0; i < truth.length; i++) {
      const dt = (truth[i] ?? 0) - mt;
      const dr = (recovered[i] ?? 0) - mr;
      cov += dt * dr;
      vt += dt * dt;
      vr += dr * dr;
    }
    expect(cov / Math.sqrt(vt * vr)).toBeGreaterThan(0.99);
  });

  it('is stable on a flat field', () => {
    const size = 32;
    const flat = new Float32Array(size * size);
    const recovered = integrateGradients({
      width: size,
      height: size,
      gx: flat,
      gy: Float32Array.from(flat),
    });
    for (const v of recovered) expect(v).toBeCloseTo(0.5, 6);
  });

  it('always returns values inside [0,1] when normalising', () => {
    const size = 32;
    const recovered = integrateGradients(gradientsOf(analyticSurface(size), size, size));
    for (const v of recovered) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('handles a non-power-of-two grid without coarsening past it', () => {
    const size = 30;
    const recovered = integrateGradients(gradientsOf(analyticSurface(size), size, size));
    expect(recovered.length).toBe(size * size);
    for (const v of recovered) expect(Number.isFinite(v)).toBe(true);
  });
});

describe('normal map decoding', () => {
  const flatMap = (count: number): Uint8Array => {
    const data = new Uint8Array(count * 4);
    for (let i = 0; i < count; i++) {
      data[i * 4] = 128;
      data[i * 4 + 1] = 128;
      data[i * 4 + 2] = 255;
      data[i * 4 + 3] = 255;
    }
    return data;
  };

  it('produces zero gradients for a flat normal map', () => {
    const field = normalMapToGradients(flatMap(64), 8, 8);
    for (let i = 0; i < 64; i++) {
      expect(Math.abs(field.gx[i] ?? 1)).toBeLessThan(0.01);
      expect(Math.abs(field.gy[i] ?? 1)).toBeLessThan(0.01);
    }
  });

  it('recovers the slope implied by a tilted normal', () => {
    // n = (-0.6, 0, 0.8) implies dh/dx = 0.6 / 0.8 = 0.75.
    const data = new Uint8Array(4);
    data[0] = Math.round((-0.6 * 0.5 + 0.5) * 255);
    data[1] = 128;
    data[2] = 255;
    data[3] = 255;
    const field = normalMapToGradients(data, 1, 1);
    expect(field.gx[0] ?? 0).toBeCloseTo(0.75, 1);
  });

  it('inverts the Y slope when the green channel convention is flipped', () => {
    const data = new Uint8Array([128, 40, 255, 255]);
    const gl = normalMapToGradients(data, 1, 1);
    const dx = normalMapToGradients(data, 1, 1, { flipGreen: true });
    expect(gl.gy[0] ?? 0).toBeCloseTo(-(dx.gy[0] ?? 0), 6);
  });

  it('scales the recovered slopes linearly with strength', () => {
    const data = new Uint8Array([60, 200, 255, 255]);
    const one = normalMapToGradients(data, 1, 1, { strength: 1 });
    const two = normalMapToGradients(data, 1, 1, { strength: 2 });
    expect(two.gx[0] ?? 0).toBeCloseTo((one.gx[0] ?? 0) * 2, 6);
  });

  it('bounds the slope implied by a degenerate normal', () => {
    // A JPEG block can easily produce a texel whose xy is at the unit circle;
    // without the z floor that single texel injects an infinite slope and the
    // Poisson solve smears it across the whole tile.
    const data = new Uint8Array([255, 255, 0, 255]);
    const field = normalMapToGradients(data, 1, 1, { minZ: 0.05 });
    expect(Number.isFinite(field.gx[0] ?? Infinity)).toBe(true);
    expect(Math.abs(field.gx[0] ?? 0)).toBeLessThanOrEqual(1 / 0.05);
  });

  it('runs the full normal-map pipeline end to end', () => {
    const size = 16;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const o = (y * size + x) * 4;
        const nx = 0.4 * Math.sin((2 * Math.PI * x) / size);
        data[o] = Math.round((nx * 0.5 + 0.5) * 255);
        data[o + 1] = 128;
        data[o + 2] = 255;
        data[o + 3] = 255;
      }
    }
    const field = heightFromNormalMap(data, size, size, { sweeps: 10 });
    expect(field.length).toBe(size * size);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of field) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    expect(lo).toBeCloseTo(0, 6);
    expect(hi).toBeCloseTo(1, 6);
  });
});

describe('channel means', () => {
  it('averages each channel independently', () => {
    const data = new Uint8Array([0, 255, 128, 255, 255, 0, 128, 255]);
    const [r, g, b, a] = channelMeans(data, 2, 1);
    expect(r).toBeCloseTo(0.5, 6);
    expect(g).toBeCloseTo(0.5, 6);
    expect(b).toBeCloseTo(128 / 255, 6);
    expect(a).toBeCloseTo(1, 6);
  });

  it('does not divide by zero on an empty image', () => {
    const [r] = channelMeans(new Uint8Array(0), 0, 0);
    expect(Number.isFinite(r)).toBe(true);
  });
});
