/**
 * Unit tests for the cascaded-shadow-map geometry.
 *
 * These cover the parts of CSM that are pure math and therefore the parts where
 * a subtle mistake is invisible in a screenshot but catastrophic in motion: the
 * split scheme, the rotation-invariant slice bounding sphere, and the texel
 * snapping that stops shadow edges from swimming.
 */

import { describe, expect, it } from 'vitest';

import {
  frustumSliceSphere,
  practicalSplits,
  shadowArrayLayers,
  snapToTexelGrid,
  vogelDiskSample,
} from '../src/render/CascadedShadowMaps';

describe('practicalSplits', () => {
  it('returns one far distance per cascade, ending exactly at the far plane', () => {
    const splits = practicalSplits(0.1, 180, 4, 0.55);
    expect(splits).toHaveLength(4);
    expect(splits[3]).toBeCloseTo(180, 10);
  });

  it('is strictly increasing', () => {
    const splits = practicalSplits(0.1, 200, 4, 0.55);
    for (let i = 1; i < splits.length; i++) {
      expect(splits[i]!).toBeGreaterThan(splits[i - 1]!);
    }
  });

  it('reduces to the uniform scheme at lambda = 0', () => {
    const near = 1;
    const far = 101;
    const splits = practicalSplits(near, far, 4, 0);
    expect(splits[0]!).toBeCloseTo(26, 6);
    expect(splits[1]!).toBeCloseTo(51, 6);
    expect(splits[2]!).toBeCloseTo(76, 6);
    expect(splits[3]!).toBeCloseTo(101, 6);
  });

  it('reduces to the logarithmic scheme at lambda = 1', () => {
    const near = 1;
    const far = 16;
    const splits = practicalSplits(near, far, 4, 1);
    // near * (far/near)^(i/4) = 2, 4, 8, 16
    expect(splits[0]!).toBeCloseTo(2, 6);
    expect(splits[1]!).toBeCloseTo(4, 6);
    expect(splits[2]!).toBeCloseTo(8, 6);
    expect(splits[3]!).toBeCloseTo(16, 6);
  });

  it('places more resolution near the camera than the uniform scheme does', () => {
    const uniform = practicalSplits(0.1, 180, 4, 0);
    const practical = practicalSplits(0.1, 180, 4, 0.55);
    // The first cascade must be tighter, which is the entire point of the
    // logarithmic contribution.
    expect(practical[0]!).toBeLessThan(uniform[0]!);
  });

  it('survives degenerate input without producing NaN', () => {
    for (const splits of [
      practicalSplits(0, 10, 3, 0.5),
      practicalSplits(10, 10, 3, 0.5),
      practicalSplits(5, 1, 3, 0.5),
    ]) {
      for (const value of splits) expect(Number.isFinite(value)).toBe(true);
    }
  });
});

/** The eight corners of a symmetric perspective frustum slice, in view space. */
function sliceCorners(
  near: number,
  far: number,
  fovY: number,
  aspect: number,
): Array<[number, number, number]> {
  const corners: Array<[number, number, number]> = [];
  for (const z of [near, far]) {
    const h = Math.tan(fovY * 0.5) * z;
    const w = h * aspect;
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        corners.push([sx * w, sy * h, -z]);
      }
    }
  }
  return corners;
}

describe('frustumSliceSphere', () => {
  const cases: Array<[number, number, number, number]> = [
    [0.1, 10, Math.PI / 3, 16 / 9],
    [10, 40, Math.PI / 3, 16 / 9],
    [40, 180, Math.PI / 3, 16 / 9],
    [0.1, 0.5, Math.PI / 3, 16 / 9], // very shallow slice
    [1, 2, Math.PI / 6, 1], // narrow fov, square aspect
    [1, 400, Math.PI / 2, 21 / 9], // extreme
  ];

  it('contains every corner of the slice', () => {
    for (const [near, far, fovY, aspect] of cases) {
      const sphere = frustumSliceSphere(near, far, fovY, aspect);
      for (const [x, y, z] of sliceCorners(near, far, fovY, aspect)) {
        const dz = z + sphere.distance; // centre sits at -distance on the view axis
        const d = Math.sqrt(x * x + y * y + dz * dz);
        expect(d).toBeLessThanOrEqual(sphere.radius * (1 + 1e-9));
      }
    }
  });

  it('is tight: at least one corner touches the sphere', () => {
    for (const [near, far, fovY, aspect] of cases) {
      const sphere = frustumSliceSphere(near, far, fovY, aspect);
      let maxDistance = 0;
      for (const [x, y, z] of sliceCorners(near, far, fovY, aspect)) {
        const dz = z + sphere.distance;
        maxDistance = Math.max(maxDistance, Math.sqrt(x * x + y * y + dz * dz));
      }
      expect(maxDistance).toBeCloseTo(sphere.radius, 6);
    }
  });

  it('is invariant to camera rotation by construction', () => {
    // A sphere has no orientation, so the only thing a rotation can change is
    // where the centre lands in world space — never the radius. This is what
    // keeps the cascade extent, and therefore the shadow texel density,
    // constant as the player looks around.
    const a = frustumSliceSphere(10, 40, Math.PI / 3, 16 / 9);
    const b = frustumSliceSphere(10, 40, Math.PI / 3, 16 / 9);
    expect(a.radius).toBe(b.radius);
    expect(a.distance).toBe(b.distance);
  });

  it('produces a positive radius for a degenerate slice', () => {
    const sphere = frustumSliceSphere(5, 5, Math.PI / 3, 16 / 9);
    expect(sphere.radius).toBeGreaterThan(0);
    expect(Number.isFinite(sphere.distance)).toBe(true);
  });
});

describe('snapToTexelGrid', () => {
  const texel = 0.25;

  it('quantises both axes to multiples of the texel size', () => {
    for (const [x, y] of [
      [0, 0],
      [1.37, -2.61],
      [-0.001, 0.001],
      [1234.5678, -8765.4321],
    ] as Array<[number, number]>) {
      const [sx, sy] = snapToTexelGrid(x, y, texel);
      expect(Math.abs(sx / texel - Math.round(sx / texel))).toBeLessThan(1e-9);
      expect(Math.abs(sy / texel - Math.round(sy / texel))).toBeLessThan(1e-9);
    }
  });

  it('holds the snapped position steady across sub-texel camera motion', () => {
    // This is the property that actually stops shadows from swimming: sweeping
    // the camera continuously must produce a *staircase*, never a slide. Over a
    // 40-texel sweep sampled 4000 times, the snapped value may only ever change
    // by exactly one texel at a time.
    let previous = snapToTexelGrid(0, 0, texel)[0];
    let steps = 0;
    for (let i = 1; i <= 4000; i++) {
      const x = (i / 4000) * 40 * texel;
      const [snapped] = snapToTexelGrid(x, 0, texel);
      const delta = snapped - previous;
      expect(Math.abs(delta) < 1e-9 || Math.abs(delta - texel) < 1e-9).toBe(true);
      if (Math.abs(delta - texel) < 1e-9) steps++;
      previous = snapped;
    }
    expect(steps).toBe(40);
  });

  it('is monotonic and never overshoots by more than one texel', () => {
    for (let i = 0; i < 500; i++) {
      const x = (i - 250) * 0.017;
      const [snapped] = snapToTexelGrid(x, 0, texel);
      expect(snapped).toBeLessThanOrEqual(x + 1e-9);
      expect(x - snapped).toBeLessThan(texel + 1e-9);
    }
  });

  it('passes values through unchanged for a non-positive texel size', () => {
    expect(snapToTexelGrid(3.5, -1.25, 0)).toEqual([3.5, -1.25]);
  });
});

describe('vogelDiskSample', () => {
  const count = 16;

  it('stays inside the unit disc', () => {
    for (let i = 0; i < count; i++) {
      const [x, y] = vogelDiskSample(i, count);
      expect(Math.hypot(x, y)).toBeLessThanOrEqual(1);
    }
  });

  it('distributes radius as sqrt(i), giving equal-area rings', () => {
    // Equal-area spacing is what makes a Vogel disc an unbiased estimator of
    // the filter footprint; sampling radius linearly would over-weight the
    // centre and produce a soft, mushy penumbra.
    for (let i = 0; i < count; i++) {
      const [x, y] = vogelDiskSample(i, count);
      expect(Math.hypot(x, y)).toBeCloseTo(Math.sqrt((i + 0.5) / count), 12);
    }
  });

  it('never repeats an angle', () => {
    const angles = new Set<number>();
    for (let i = 0; i < 64; i++) {
      const [x, y] = vogelDiskSample(i, 64);
      const angle = Math.round(Math.atan2(y, x) * 1e6);
      expect(angles.has(angle)).toBe(false);
      angles.add(angle);
    }
  });

  it('has a near-zero centroid, so the kernel does not bias the shadow', () => {
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < count; i++) {
      const [x, y] = vogelDiskSample(i, count);
      sx += x;
      sy += y;
    }
    expect(Math.hypot(sx / count, sy / count)).toBeLessThan(0.06);
  });
});

/**
 * The array-layer floor.
 *
 * This is not an optimisation, it is the fix for a real captured defect. The
 * tier table carried `shadowCascades: 2` for several rounds with a note that
 * one "was tried and does not work" — it rendered the sky away and replaced the
 * ground with grey blocks. The cause is that three derives
 * `Texture.isArrayTexture` from `image.depth > 1`, so a one-layer array is not
 * an array and both the backend's array-target path and this module's
 * `.depth(layer)` sampling silently take the wrong branch. Allocating a spare
 * layer is what makes the single-cascade configuration correct, and the single
 * cascade is worth 77 of 222 draws in the Rogue Encampment.
 */
describe('shadowArrayLayers', () => {
  it('never returns fewer than two, because a one-layer array is not an array', () => {
    expect(shadowArrayLayers(1)).toBe(2);
  });

  it('allocates exactly one layer per cascade above that', () => {
    expect(shadowArrayLayers(2)).toBe(2);
    expect(shadowArrayLayers(3)).toBe(3);
    expect(shadowArrayLayers(4)).toBe(4);
  });
});
