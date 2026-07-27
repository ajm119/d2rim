/**
 * Tests for the GTAO integrand and its supporting math.
 *
 * The visibility integral is the whole effect. If it is wrong the AO still
 * *looks* like AO — dark in corners, light in the open — which is exactly why
 * it needs to be pinned numerically rather than by eye. The properties asserted
 * here are the ones a plausible-looking but wrong implementation violates:
 *
 * - the closed form agrees with a quadrature of its own integrand, derived
 *   independently by differentiating the published expression, so a
 *   transcription error cannot hide;
 * - a half-occluded slice integrates to exactly 1/2;
 * - and — the one that matters most — the *slice-weighted average* over an
 *   unoccluded hemisphere is exactly 1 for any surface tilt. A flipped cross
 *   product or a lost sign in the slice frame still produces AO that darkens
 *   creases and looks broadly right, while multiplying every ambient term on
 *   open ground by something other than 1 and silently rebalancing the whole
 *   lighting rig.
 *
 * Note that a *single* slice is deliberately not normalised to 1; that is
 * asserted explicitly below so nobody "fixes" it.
 */

import { describe, expect, it } from 'vitest';

import {
  GTAO_TEMPORAL_OFFSETS,
  GTAO_TEMPORAL_PERIOD,
  GTAO_TEMPORAL_ROTATIONS,
  GTAO_TIERS,
  gtaoArcIntegral,
  gtaoMultiBounce,
  gtaoOpenHorizons,
  gtaoScreenRadius,
  gtaoSliceFrame,
  gtaoSliceVisibility,
  gtaoTemporalOffset,
  gtaoTemporalRotation,
  type Vec3Tuple,
} from '../src/render/post/GTAO';

const HALF_PI = Math.PI / 2;

/**
 * Numerically integrate the quantity `gtaoArcIntegral` is a closed form of.
 *
 * Differentiating the published expression gives `sin θ · cos(θ − n)`: the
 * cosine weight against the slice-projected normal times the `sin θ` Jacobian of
 * sweeping the slice through the hemisphere. Integrating that from 0 out to the
 * horizon — separately for each side, which is why the two horizons are summed
 * rather than integrated between — is exactly what the closed form must equal.
 */
function numericArcIntegral(h: number, n: number, samples = 400000): number {
  let total = 0;
  const step = h / samples;
  for (let i = 0; i < samples; i++) {
    const theta = (i + 0.5) * step;
    total += Math.sin(theta) * Math.cos(theta - n) * step;
  }
  return total;
}

/**
 * Slice-weighted visibility of a completely unoccluded hemisphere, computed
 * exactly the way the shader does: build each slice's frame, take its fully
 * open horizons, evaluate the arc integral, weight by the projected normal
 * length, average.
 */
function unoccludedVisibility(tiltRadians: number, slices = 512): number {
  const view: Vec3Tuple = [0, 0, 1];
  const normal: Vec3Tuple = [Math.sin(tiltRadians), 0, Math.cos(tiltRadians)];

  let total = 0;
  for (let i = 0; i < slices; i++) {
    const phi = ((i + 0.5) * Math.PI) / slices;
    const frame = gtaoSliceFrame(normal, view, phi);
    const [h1, h2] = gtaoOpenHorizons(frame.angle);
    total += gtaoSliceVisibility(h1, h2, frame.angle, frame.projectedNormalLength);
  }
  return total / slices;
}

describe('GTAO arc integral', () => {
  it('gives full visibility for an unoccluded hemisphere', () => {
    // Normal facing the camera, horizons at ±90°: nothing is blocking.
    const v = gtaoArcIntegral(-HALF_PI, 0) + gtaoArcIntegral(HALF_PI, 0);
    expect(v).toBeCloseTo(1, 12);
  });

  it('is NOT individually normalised for a tilted normal', () => {
    // Documented here because it looks like a bug and is not: a single slice
    // integrates to cos(n) + n·sin(n), which exceeds 1. Only the projected-
    // normal-weighted average over slices is normalised — see the suite below.
    for (const n of [0.3, 0.9, 1.2]) {
      const [h1, h2] = gtaoOpenHorizons(n);
      const v = gtaoArcIntegral(h1, n) + gtaoArcIntegral(h2, n);
      expect(v).toBeCloseTo(Math.cos(n) + n * Math.sin(n), 10);
      expect(v).toBeGreaterThan(1);
    }
  });

  it('gives exactly half visibility when one side is fully blocked', () => {
    // A wall rising exactly along the view vector: one horizon at 0.
    const v = gtaoArcIntegral(0, 0) + gtaoArcIntegral(HALF_PI, 0);
    expect(v).toBeCloseTo(0.5, 12);
  });

  it('agrees with numerical integration of sin(theta)*cos(theta - n)', () => {
    // The strongest available check on the transcription of the formula: it is
    // compared against a quadrature of its own integrand, derived independently
    // by differentiating the published closed form.
    for (const n of [0, 0.4, -0.7, 1.1]) {
      const reference = gtaoArcIntegral(0, n);
      for (const h of [-1.4, -1.0, -0.3, 0.3, 0.9, 1.5]) {
        const closed = gtaoArcIntegral(h, n) - reference;
        expect(closed).toBeCloseTo(numericArcIntegral(h, n), 4);
      }
    }
  });

  it('opens monotonically on each side of the view vector', () => {
    // Raising the horizon on the positive side can only let more light in.
    let previous = -Infinity;
    for (let h = 0; h <= HALF_PI + 1e-9; h += 0.05) {
      const v = gtaoArcIntegral(h, 0);
      expect(v).toBeGreaterThanOrEqual(previous - 1e-12);
      previous = v;
    }
    // ... and symmetrically on the negative side.
    previous = -Infinity;
    for (let h = 0; h >= -HALF_PI - 1e-9; h -= 0.05) {
      const v = gtaoArcIntegral(h, 0);
      expect(v).toBeGreaterThanOrEqual(previous - 1e-12);
      previous = v;
    }
  });
});

describe('GTAO slice frame', () => {
  it('integrates an unoccluded hemisphere to exactly 1 at any surface tilt', () => {
    // This is the load-bearing test for the whole trace. A flipped cross
    // product, a lost `sign`, or a missing projected-normal weight all leave AO
    // that still darkens creases and still looks broadly plausible, but that
    // silently multiplies every ambient term by something other than 1 on open
    // ground — which quietly rebalances the entire lighting rig.
    for (const tilt of [0, 0.2, 0.5, 0.9, 1.2, 1.4]) {
      expect(unoccludedVisibility(tilt)).toBeCloseTo(1, 5);
    }
  });

  it('gives a full-length projected normal for the slice containing the normal', () => {
    const view: Vec3Tuple = [0, 0, 1];
    const normal: Vec3Tuple = [Math.sin(0.6), 0, Math.cos(0.6)];
    // phi = 0 -> slice direction along +X, which is the plane the normal tilts in.
    const frame = gtaoSliceFrame(normal, view, 0);
    expect(frame.projectedNormalLength).toBeCloseTo(1, 6);
    expect(Math.abs(frame.angle)).toBeCloseTo(0.6, 6);
  });

  it('gives a shortened projected normal for the perpendicular slice', () => {
    const view: Vec3Tuple = [0, 0, 1];
    const normal: Vec3Tuple = [Math.sin(0.6), 0, Math.cos(0.6)];
    const frame = gtaoSliceFrame(normal, view, Math.PI / 2);
    expect(frame.projectedNormalLength).toBeCloseTo(Math.cos(0.6), 6);
    // The projected normal now lies along the view vector: zero angle.
    expect(frame.angle).toBeCloseTo(0, 6);
  });

  it('reports a normal facing the camera as zero angle in every slice', () => {
    const view: Vec3Tuple = [0, 0, 1];
    for (let i = 0; i < 16; i++) {
      const frame = gtaoSliceFrame([0, 0, 1], view, (i * Math.PI) / 16);
      expect(frame.angle).toBeCloseTo(0, 6);
      expect(frame.projectedNormalLength).toBeCloseTo(1, 6);
    }
  });
});

describe('GTAO slice visibility', () => {
  it('scales with the projected normal length', () => {
    const full = gtaoSliceVisibility(-HALF_PI, HALF_PI, 0, 1);
    const half = gtaoSliceVisibility(-HALF_PI, HALF_PI, 0, 0.5);
    expect(full).toBeCloseTo(1, 12);
    expect(half).toBeCloseTo(0.5, 12);
  });
});

describe('GTAO multi-bounce', () => {
  it('is the identity at zero and one', () => {
    for (const albedo of [0, 0.2, 0.5, 0.9]) {
      expect(gtaoMultiBounce(0, albedo)).toBeCloseTo(0, 6);
      expect(gtaoMultiBounce(1, albedo)).toBeCloseTo(1, 6);
    }
  });

  it('never darkens below the raw visibility', () => {
    for (let v = 0; v <= 1; v += 0.05) {
      for (const albedo of [0, 0.05, 0.25, 0.55, 0.9]) {
        expect(gtaoMultiBounce(v, albedo)).toBeGreaterThanOrEqual(v - 1e-9);
      }
    }
  });

  it('lifts occlusion more as albedo rises', () => {
    // This is the property that keeps pale stone from reading like dark mud.
    const dark = gtaoMultiBounce(0.5, 0.06);
    const grass = gtaoMultiBounce(0.5, 0.25);
    const stone = gtaoMultiBounce(0.5, 0.55);
    expect(dark).toBeLessThan(grass);
    expect(grass).toBeLessThan(stone);
    // Dark wet mud should barely move at all.
    expect(dark).toBeLessThan(0.55);
    // Pale stone should lift substantially.
    expect(stone).toBeGreaterThan(0.65);
  });

  it('stays inside [0, 1] for every input', () => {
    for (let v = -0.2; v <= 1.2; v += 0.05) {
      for (let albedo = 0; albedo <= 1; albedo += 0.05) {
        const result = gtaoMultiBounce(v, albedo);
        expect(result).toBeGreaterThanOrEqual(0);
        expect(result).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe('GTAO screen radius', () => {
  it('is inversely proportional to depth', () => {
    const near = gtaoScreenRadius(1, 2, 1.73, 1080, 1e9);
    const far = gtaoScreenRadius(1, 4, 1.73, 1080, 1e9);
    expect(near / far).toBeCloseTo(2, 6);
  });

  it('matches the projection identity it is derived from', () => {
    // A 1 m sphere at 10 m with a 60° vertical FOV (P11 = 1/tan(30°)).
    const p11 = 1 / Math.tan((30 * Math.PI) / 180);
    const expected = (1 * p11 * 1080) / (2 * 10);
    expect(gtaoScreenRadius(1, 10, p11, 1080, 1e9)).toBeCloseTo(expected, 6);
  });

  it('clamps at both ends so the march cost stays bounded', () => {
    // Right against the near plane the unclamped radius is enormous.
    expect(gtaoScreenRadius(1, 0.01, 1.73, 1080, 96)).toBe(96);
    // Very far away it would fall below a texel and produce a hard cut-off.
    expect(gtaoScreenRadius(0.8, 5000, 1.73, 1080, 96)).toBe(1);
  });
});

describe('GTAO temporal sequence', () => {
  it('visits every rotation before repeating', () => {
    const seen = new Set<number>();
    for (let frame = 0; frame < 6; frame++) seen.add(gtaoTemporalRotation(frame));
    expect(seen.size).toBe(GTAO_TEMPORAL_ROTATIONS.length);
  });

  it('spreads rotations over the half turn', () => {
    // Slices span [0, π), so the rotations are turns of that half turn. Any
    // three consecutive frames should already cover it reasonably evenly, which
    // is what makes the one-slice `low` tier converge quickly.
    const first3 = [0, 1, 2].map((f) => gtaoTemporalRotation(f)).sort((a, b) => a - b);
    const gaps = first3.map((value, index) =>
      index === 0 ? value : value - (first3[index - 1] ?? 0),
    );
    for (const gap of gaps) expect(gap).toBeLessThanOrEqual(0.7);
  });

  it('advances the step offset only after a full rotation cycle', () => {
    for (let frame = 0; frame < 6; frame++) {
      expect(gtaoTemporalOffset(frame)).toBe(GTAO_TEMPORAL_OFFSETS[0]);
    }
    expect(gtaoTemporalOffset(6)).toBe(GTAO_TEMPORAL_OFFSETS[1]);
    expect(gtaoTemporalOffset(12)).toBe(GTAO_TEMPORAL_OFFSETS[2]);
    // ... and wraps after the full 24-frame period.
    expect(gtaoTemporalOffset(GTAO_TEMPORAL_PERIOD)).toBe(GTAO_TEMPORAL_OFFSETS[0]);
  });

  it('handles negative and large frame indices without producing NaN', () => {
    for (const frame of [-1, -7, 1e6, 1e6 + 3]) {
      expect(Number.isFinite(gtaoTemporalRotation(frame))).toBe(true);
      expect(Number.isFinite(gtaoTemporalOffset(frame))).toBe(true);
    }
  });
});

describe('GTAO quality tiers', () => {
  it('increase sample counts monotonically', () => {
    const order = ['low', 'medium', 'high', 'ultra'] as const;
    for (let i = 1; i < order.length; i++) {
      const previous = GTAO_TIERS[order[i - 1]!];
      const current = GTAO_TIERS[order[i]!];
      expect(current.slices).toBeGreaterThanOrEqual(previous.slices);
      expect(current.steps).toBeGreaterThanOrEqual(previous.steps);
      expect(current.slices * current.steps).toBeGreaterThan(previous.slices * previous.steps);
    }
  });

  it('keeps every tier inside its stated tap budget', () => {
    // Taps per half-resolution pixel = slices x steps x 2 sides. The `ultra`
    // ceiling of 64 is what the module's cost table is priced against.
    for (const tier of Object.values(GTAO_TIERS)) {
      expect(tier.slices * tier.steps * 2).toBeLessThanOrEqual(64);
    }
  });
});
