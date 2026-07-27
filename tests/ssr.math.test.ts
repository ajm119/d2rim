/**
 * Tests for the SSR cone-tracing and Fresnel math.
 *
 * The ray march itself cannot be unit tested without a GPU, but everything that
 * decides *what the march means* can be, and those are the parts that produce
 * silently wrong images:
 *
 * - the roughness → cone-angle → mip chain, which is what makes a rough surface
 *   blurry rather than a mirror;
 * - the environment-BRDF fit, which decides how strong the reflection is at
 *   every angle and is the difference between "wet" and "chrome";
 * - the screen-border fade, which is the mechanism that makes the hand-off to
 *   the probe seamless.
 */

import { describe, expect, it } from 'vitest';

import {
  SSR_TIERS,
  coneRadiusToMip,
  envBRDFApprox,
  roughnessToConeAngle,
  roughnessToSpecularPower,
  screenEdgeFade,
  specularPowerToConeAngle,
  thicknessAt,
  vogelDisk,
} from '../src/render/post/SSR';

describe('roughness to specular power', () => {
  it('is the published 2/alpha^4 - 2 relation', () => {
    const roughness = 0.4;
    const alpha = roughness * roughness;
    expect(roughnessToSpecularPower(roughness)).toBeCloseTo(2 / (alpha * alpha) - 2, 6);
  });

  it('falls monotonically as roughness rises', () => {
    let previous = Infinity;
    for (let r = 0.05; r <= 1; r += 0.05) {
      const power = roughnessToSpecularPower(r);
      expect(power).toBeLessThan(previous);
      previous = power;
    }
  });

  it('does not divide by zero at roughness 0', () => {
    expect(Number.isFinite(roughnessToSpecularPower(0))).toBe(true);
  });
});

describe('cone angle', () => {
  it('is (near) zero for a mirror and wide for a rough surface', () => {
    expect(roughnessToConeAngle(0.01)).toBeLessThan(0.02);
    expect(roughnessToConeAngle(1)).toBeGreaterThan(0.6);
  });

  it('grows monotonically with roughness', () => {
    let previous = -1;
    for (let r = 0.02; r <= 1; r += 0.02) {
      const angle = roughnessToConeAngle(r);
      expect(angle).toBeGreaterThanOrEqual(previous);
      previous = angle;
    }
  });

  it('never exceeds a hemisphere', () => {
    for (let r = 0; r <= 1; r += 0.05) {
      expect(roughnessToConeAngle(r)).toBeLessThanOrEqual(Math.PI / 2 + 1e-9);
    }
  });

  it('collapses to zero for an effectively infinite specular power', () => {
    expect(specularPowerToConeAngle(Number.POSITIVE_INFINITY)).toBe(0);
    expect(specularPowerToConeAngle(1e9)).toBe(0);
  });

  it('spans a plausible range for this scene s materials', () => {
    // Wet mud and rain-slicked stone are the cases the effect exists for. Their
    // cones should be wide enough to visibly blur but not so wide that the trace
    // is pointless.
    const wetMud = roughnessToConeAngle(0.3);
    const wetStone = roughnessToConeAngle(0.15);
    const polishedSteel = roughnessToConeAngle(0.05);
    expect(polishedSteel).toBeLessThan(wetStone);
    expect(wetStone).toBeLessThan(wetMud);
    expect(wetMud).toBeGreaterThan(0.1);
    expect(wetMud).toBeLessThan(0.6);
  });
});

describe('cone radius to mip', () => {
  it('is level 0 for a sub-texel footprint', () => {
    expect(coneRadiusToMip(0, 8)).toBe(0);
    expect(coneRadiusToMip(0.4, 8)).toBe(0);
  });

  it('advances one level per doubling of the footprint', () => {
    // Level L of a box pyramid averages a 2^L-wide block, so a footprint of
    // diameter d belongs at log2(d).
    expect(coneRadiusToMip(1, 8)).toBeCloseTo(1, 6);
    expect(coneRadiusToMip(2, 8)).toBeCloseTo(2, 6);
    expect(coneRadiusToMip(4, 8)).toBeCloseTo(3, 6);
  });

  it('clamps at the top of the pyramid', () => {
    expect(coneRadiusToMip(100000, 6)).toBe(6);
  });
});

describe('environment BRDF approximation', () => {
  /**
   * Reference values from Karis's split-sum integration, as tabulated in
   * "Real Shading in Unreal Engine 4" and reproduced in the Lazarov talk this
   * fit comes from. The fit is documented as accurate to well under 1 %.
   */
  it('gives near-unity scale for a smooth surface viewed head-on', () => {
    const [a, b] = envBRDFApprox(0, 1);
    expect(a + b).toBeCloseTo(1, 2);
  });

  it('drives F_env towards 1 at grazing incidence for any roughness', () => {
    // Fresnel: everything is a mirror at 90 degrees. This is what makes a wet
    // road blinding along the view direction and matte underfoot, and it is the
    // single most important behaviour of the term.
    for (const roughness of [0.05, 0.3, 0.6]) {
      const [aGrazing, bGrazing] = envBRDFApprox(roughness, 0.02);
      const [aHead, bHead] = envBRDFApprox(roughness, 1);
      const f0 = 0.04;
      const grazing = f0 * aGrazing + bGrazing;
      const head = f0 * aHead + bHead;
      expect(grazing).toBeGreaterThan(head);
    }
  });

  it('produces a dielectric response in a physically sane range', () => {
    // F0 = 0.04 is a typical dielectric; head-on reflectance should sit near it
    // and never exceed 1.
    for (let r = 0; r <= 1; r += 0.1) {
      for (let nv = 0.02; nv <= 1; nv += 0.1) {
        const [a, b] = envBRDFApprox(r, nv);
        const f = 0.04 * a + b;
        expect(f).toBeGreaterThanOrEqual(0);
        expect(f).toBeLessThanOrEqual(1.01);
      }
    }
  });

  it('weakens as roughness rises at a fixed angle', () => {
    const f0 = 0.9; // a metal, where the effect is most visible
    let previous = Infinity;
    for (let r = 0.05; r <= 0.95; r += 0.1) {
      const [a, b] = envBRDFApprox(r, 0.7);
      const f = f0 * a + b;
      expect(f).toBeLessThan(previous);
      previous = f;
    }
  });
});

describe('screen edge fade', () => {
  it('is fully confident in the middle of the screen', () => {
    expect(screenEdgeFade(0.5, 0.5, 0.12)).toBeCloseTo(1, 6);
  });

  it('reaches exactly zero at every border', () => {
    // Not "close to zero": a residual contribution at the border is precisely
    // the hard cut-off this whole design exists to avoid.
    expect(screenEdgeFade(0, 0.5, 0.12)).toBe(0);
    expect(screenEdgeFade(1, 0.5, 0.12)).toBe(0);
    expect(screenEdgeFade(0.5, 0, 0.12)).toBe(0);
    expect(screenEdgeFade(0.5, 1, 0.12)).toBe(0);
  });

  it('is continuous and monotonic approaching a border', () => {
    let previous = 1.0001;
    for (let u = 0.5; u >= 0; u -= 0.01) {
      const fade = screenEdgeFade(u, 0.5, 0.2);
      expect(fade).toBeLessThanOrEqual(previous + 1e-9);
      previous = fade;
    }
  });

  it('is symmetric in both axes', () => {
    for (const u of [0.03, 0.12, 0.4]) {
      expect(screenEdgeFade(u, 0.5, 0.15)).toBeCloseTo(screenEdgeFade(1 - u, 0.5, 0.15), 12);
      expect(screenEdgeFade(0.5, u, 0.15)).toBeCloseTo(screenEdgeFade(0.5, 1 - u, 0.15), 12);
    }
  });

  it('has zero derivative at both ends of the ramp', () => {
    // Smoothstep, not linear: a C0-only fade shows as a visible band where the
    // reflection starts disappearing.
    const fade = 0.2;
    const justInside = screenEdgeFade(fade / 2 - 0.001, 0.5, fade);
    const atRampEnd = screenEdgeFade(fade / 2, 0.5, fade);
    expect(Math.abs(atRampEnd - justInside)).toBeLessThan(1e-3);
  });
});

describe('thickness', () => {
  it('grows with depth but far more slowly than depth itself', () => {
    const near = thicknessAt(1, 0.12);
    const far = thicknessAt(100, 0.12);
    expect(far).toBeGreaterThan(near);
    // A linear-in-depth window would be 100x here and would accept nonsense.
    expect(far / near).toBeLessThan(10);
  });

  it('stays close to the authored value at contact distance', () => {
    expect(thicknessAt(0, 0.12)).toBeCloseTo(0.12, 6);
    expect(thicknessAt(1, 0.12)).toBeCloseTo(0.126, 6);
  });

  it('never returns a negative window for a degenerate depth', () => {
    expect(thicknessAt(-5, 0.12)).toBeGreaterThan(0);
  });
});

describe('Vogel disk', () => {
  it('stays inside the unit disk', () => {
    for (const count of [1, 4, 8, 16]) {
      for (const [x, y] of vogelDisk(count)) {
        expect(Math.hypot(x, y)).toBeLessThanOrEqual(1 + 1e-12);
      }
    }
  });

  it('has a centre of mass near the origin for any count', () => {
    // A clumped pattern biases the cone average towards one side of the
    // footprint, which reads as structured noise in a rough reflection.
    for (const count of [4, 8, 16]) {
      const points = vogelDisk(count);
      const cx = points.reduce((sum, p) => sum + p[0], 0) / count;
      const cy = points.reduce((sum, p) => sum + p[1], 0) / count;
      expect(Math.hypot(cx, cy)).toBeLessThan(0.35);
    }
  });

  it('produces the requested number of distinct points', () => {
    const points = vogelDisk(8);
    expect(points).toHaveLength(8);
    const keys = new Set(points.map(([x, y]) => `${x.toFixed(6)},${y.toFixed(6)}`));
    expect(keys.size).toBe(8);
  });
});

describe('SSR quality tiers', () => {
  it('increase cost monotonically', () => {
    const order = ['low', 'medium', 'high', 'ultra'] as const;
    for (let i = 1; i < order.length; i++) {
      const previous = SSR_TIERS[order[i - 1]!];
      const current = SSR_TIERS[order[i]!];
      expect(current.maxSteps).toBeGreaterThan(previous.maxSteps);
      expect(current.coneTaps).toBeGreaterThanOrEqual(previous.coneTaps);
      expect(current.maxHiZLevel).toBeGreaterThanOrEqual(previous.maxHiZLevel);
    }
  });

  it('gives the low tier a linear march by capping the hierarchy at level 0', () => {
    // This is how the tier degrades without a second code path.
    expect(SSR_TIERS.low.maxHiZLevel).toBe(0);
  });

  it('keeps every tier inside its stated step budget', () => {
    for (const tier of Object.values(SSR_TIERS)) {
      expect(tier.maxSteps).toBeLessThanOrEqual(80);
      expect(tier.coneTaps).toBeLessThanOrEqual(8);
    }
  });
});
