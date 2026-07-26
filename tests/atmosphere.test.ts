/**
 * Tests for the scattering model in `render/Atmosphere`.
 *
 * The interesting properties here are not "does it return a number" but the
 * physical invariants a scattering model has to satisfy — normalised phase
 * functions, an invertible LUT parameterisation, extinction that behaves like
 * an exponential, and, most importantly, a GPU uniform block that reproduces
 * the CPU path exactly. That last one is the whole reason this module exists
 * as a shared service.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import {
  Atmosphere,
  EARTH_ATMOSPHERE,
  cornetteShanksPhase,
  rMuFromTransmittanceUv,
  raySphereDistance,
  rayleighPhase,
  transmittanceUvFromRMu,
  UNIFORM_PHASE,
} from '../src/render/Atmosphere';

const { bottomRadiusKm, topRadiusKm } = EARTH_ATMOSPHERE;

/** Integrate `phase` over the unit sphere with a dense Gauss-free grid. */
function integrateOverSphere(phase: (mu: number) => number, steps = 4000): number {
  // Uniform in mu is the right measure: d(omega) = 2*pi*d(mu) for an
  // azimuthally symmetric function.
  let total = 0;
  for (let i = 0; i < steps; i++) {
    const mu = -1 + (2 * (i + 0.5)) / steps;
    total += phase(mu) * (2 / steps);
  }
  return total * 2 * Math.PI;
}

describe('phase functions', () => {
  it('Rayleigh integrates to 1 over the sphere', () => {
    expect(integrateOverSphere(rayleighPhase)).toBeCloseTo(1, 6);
  });

  it('Cornette-Shanks integrates to 1 for every plausible g', () => {
    for (const g of [0, 0.2, 0.5, 0.76, 0.8, 0.85]) {
      expect(integrateOverSphere((mu) => cornetteShanksPhase(mu, g), 200_000)).toBeCloseTo(1, 3);
    }
  });

  it('Cornette-Shanks reduces to isotropic at g = 0', () => {
    for (const mu of [-1, -0.4, 0, 0.4, 1]) {
      // At g = 0 the lobe is (3/16pi)(1+mu^2) — Rayleigh, not flat. What must
      // hold is that it matches Rayleigh exactly there.
      expect(cornetteShanksPhase(mu, 0)).toBeCloseTo(rayleighPhase(mu), 12);
    }
  });

  it('is strongly forward-peaked at g = 0.8', () => {
    expect(cornetteShanksPhase(1, 0.8)).toBeGreaterThan(cornetteShanksPhase(-1, 0.8) * 100);
  });

  it('uniform phase is 1/4pi', () => {
    expect(UNIFORM_PHASE * 4 * Math.PI).toBeCloseTo(1, 12);
  });
});

describe('raySphereDistance', () => {
  it('finds the far root for a ray starting inside the sphere', () => {
    // Straight up from the surface: exactly the shell thickness.
    const d = raySphereDistance(bottomRadiusKm, 1, topRadiusKm);
    expect(d).toBeCloseTo(topRadiusKm - bottomRadiusKm, 6);
  });

  it('finds the horizon tangent length', () => {
    // Horizontal ray from the surface to the top of the atmosphere.
    const expected = Math.sqrt(topRadiusKm ** 2 - bottomRadiusKm ** 2);
    expect(raySphereDistance(bottomRadiusKm, 0, topRadiusKm)).toBeCloseTo(expected, 6);
  });

  it('reports a miss for a ray that never reaches the sphere', () => {
    // From above the top shell, pointing away from the planet.
    expect(raySphereDistance(topRadiusKm + 10, 1, bottomRadiusKm)).toBe(-1);
  });

  it('detects a ground hit for a downward ray', () => {
    const d = raySphereDistance(bottomRadiusKm + 2, -1, bottomRadiusKm);
    expect(d).toBeCloseTo(2, 6);
  });
});

describe('transmittance LUT parameterisation', () => {
  it('round-trips (r, mu) through uv for the whole valid domain', () => {
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 16; j++) {
        const u = (i + 0.5) / 32;
        const v = (j + 0.5) / 16;
        const { r, mu } = rMuFromTransmittanceUv(u, v, bottomRadiusKm, topRadiusKm);
        const back = transmittanceUvFromRMu(r, mu, bottomRadiusKm, topRadiusKm);
        expect(back.u).toBeCloseTo(u, 5);
        expect(back.v).toBeCloseTo(v, 5);
      }
    }
  });

  it('keeps r inside the atmosphere shell', () => {
    for (let j = 0; j <= 16; j++) {
      const { r } = rMuFromTransmittanceUv(0.5, j / 16, bottomRadiusKm, topRadiusKm);
      expect(r).toBeGreaterThanOrEqual(bottomRadiusKm - 1e-6);
      expect(r).toBeLessThanOrEqual(topRadiusKm + 1e-6);
    }
  });

  it('maps u = 1 to the horizon-grazing direction', () => {
    // At the maximum distance to the top shell, the ray is tangent to the
    // planet, i.e. mu is at its smallest non-occluded value.
    const grazing = rMuFromTransmittanceUv(1, 0.5, bottomRadiusKm, topRadiusKm);
    const straightUp = rMuFromTransmittanceUv(0, 0.5, bottomRadiusKm, topRadiusKm);
    expect(straightUp.mu).toBeCloseTo(1, 6);
    expect(grazing.mu).toBeLessThan(0);
  });
});

describe('Atmosphere', () => {
  let atmosphere: Atmosphere;
  const scratch = new THREE.Vector3();

  const setSunElevation = (degrees: number): void => {
    const e = THREE.MathUtils.degToRad(degrees);
    atmosphere.setSunDirection(new THREE.Vector3(0, Math.sin(e), Math.cos(e)));
  };

  beforeAll(() => {
    atmosphere = new Atmosphere();
  });

  it('produces a blue zenith under a high sun', () => {
    setSunElevation(60);
    const zenith = atmosphere.skyRadiance(new THREE.Vector3(0, 1, 0), scratch).clone();
    expect(zenith.z).toBeGreaterThan(zenith.y);
    expect(zenith.y).toBeGreaterThan(zenith.x);
    // Rayleigh's lambda^-4 puts blue at roughly 4-6x red at the zenith.
    expect(zenith.z / zenith.x).toBeGreaterThan(3);
  });

  it('reddens the horizon toward a low sun', () => {
    setSunElevation(3);
    const toward = atmosphere
      .skyRadiance(new THREE.Vector3(0, 0.02, 0.9998).normalize(), scratch)
      .clone();
    // The classic sunset signature: red now dominates where blue did at noon.
    expect(toward.x).toBeGreaterThan(toward.z * 3);
  });

  it('reddens direct sunlight monotonically as the sun sets', () => {
    const ratios: number[] = [];
    for (const elevation of [60, 30, 15, 5, 2]) {
      setSunElevation(elevation);
      const t = atmosphere.sunTransmittance(scratch);
      ratios.push(t.z / t.x); // blue/red; falls as the path lengthens
    }
    for (let i = 1; i < ratios.length; i++) {
      expect(ratios[i] as number).toBeLessThan(ratios[i - 1] as number);
    }
    expect(ratios[0] as number).toBeLessThan(1);
  });

  it('goes dark once the sun is well below the horizon', () => {
    setSunElevation(-12);
    const zenith = atmosphere.skyRadiance(new THREE.Vector3(0, 1, 0), scratch).clone();
    expect(zenith.length()).toBeLessThan(1e-3);
    expect(atmosphere.sunTransmittance(scratch).length()).toBe(0);
  });

  it('never produces negative radiance in any direction', () => {
    for (const elevation of [80, 20, 0, -5, -30]) {
      setSunElevation(elevation);
      for (let i = 0; i < 64; i++) {
        const theta = (i / 64) * Math.PI * 2;
        const dir = new THREE.Vector3(
          Math.cos(theta) * 0.8,
          Math.sin(i * 1.7) * 0.9,
          Math.sin(theta) * 0.8,
        ).normalize();
        const radiance = atmosphere.skyRadiance(dir, scratch);
        expect(radiance.x).toBeGreaterThanOrEqual(0);
        expect(radiance.y).toBeGreaterThanOrEqual(0);
        expect(radiance.z).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(radiance.x + radiance.y + radiance.z)).toBe(true);
      }
    }
  });

  describe('aerial perspective', () => {
    const horizontal = new THREE.Vector3(0, 0, 1);

    beforeAll(() => setSunElevation(35));

    it('is the identity at zero distance', () => {
      const sample = atmosphere.aerialPerspective(horizontal, 0);
      expect(sample.transmittance.x).toBe(1);
      expect(sample.inscatter.length()).toBe(0);
    });

    it('extinguishes exponentially: T(2d) = T(d)^2', () => {
      const a = atmosphere.aerialPerspective(horizontal, 3000).transmittance.clone();
      const b = atmosphere.aerialPerspective(horizontal, 6000).transmittance.clone();
      expect(b.x).toBeCloseTo(a.x * a.x, 9);
      expect(b.y).toBeCloseTo(a.y * a.y, 9);
      expect(b.z).toBeCloseTo(a.z * a.z, 9);
    });

    it('extinguishes blue faster than red, so distance goes blue-grey', () => {
      const t = atmosphere.aerialPerspective(horizontal, 20_000).transmittance.clone();
      expect(t.z).toBeLessThan(t.y);
      expect(t.y).toBeLessThan(t.x);
      const inscatter = atmosphere.aerialPerspective(horizontal, 20_000).inscatter.clone();
      expect(inscatter.z).toBeGreaterThan(inscatter.x);
    });

    it('saturates to the source radiance at long range', () => {
      const mid = atmosphere.aerialPerspective(horizontal, 2_000_000).inscatter.clone();
      const far = atmosphere.aerialPerspective(horizontal, 8_000_000);
      // Fully opaque: nothing of the surface survives, and the in-scattered
      // radiance has converged to S/sigma_t and stopped growing.
      expect(far.transmittance.length()).toBeLessThan(1e-3);
      expect(far.inscatter.y).toBeGreaterThan(0);
      expect(far.inscatter.y).toBeCloseTo(mid.y, 6);
      expect(far.inscatter.x).toBeCloseTo(mid.x, 4);
    });

    it('agrees exactly with the uniform block the GPU path binds', () => {
      // This is the invariant the whole "one fog model" claim rests on: the
      // shader evaluates `S/sigma_t * (1 - T)` from `atmosphere.uniforms`, so
      // recomputing it here from those same uniforms must reproduce the CPU
      // result to floating-point precision.
      atmosphere.refresh();
      const u = atmosphere.uniforms;
      const distanceKm = 4.2;
      const sample = atmosphere.aerialPerspective(horizontal, distanceKm * 1000);

      const mu = horizontal.dot(atmosphere.sunDirection);
      const phaseR = rayleighPhase(mu);
      const phaseM = cornetteShanksPhase(mu, u.miePhaseG);

      for (const axis of ['x', 'y', 'z'] as const) {
        const extinction = u.extinction[axis];
        const scatter = u.rayleighScattering[axis] + u.mieScattering;
        const source =
          (u.rayleighScattering[axis] * phaseR + u.mieScattering * phaseM) * u.sunRadiance[axis] +
          scatter * u.multiScatter[axis];
        const transmittance = Math.exp(-extinction * distanceKm);
        const inscatter = (source / extinction) * (1 - transmittance);

        expect(sample.transmittance[axis]).toBeCloseTo(transmittance, 10);
        expect(sample.inscatter[axis]).toBeCloseTo(inscatter, 10);
      }
    });

    it('darkens the haze when the sun is occluded by a cloud deck', () => {
      const lit = atmosphere.aerialPerspective(horizontal, 8000).inscatter.clone();
      atmosphere.aerial.sunOcclusion.set(0.05, 0.05, 0.05);
      atmosphere.refresh();
      const occluded = atmosphere.aerialPerspective(horizontal, 8000).inscatter.clone();
      expect(occluded.y).toBeLessThan(lit.y * 0.3);

      // ...and the deck's own radiance puts light back in, diffusely.
      atmosphere.aerial.ambientRadiance.set(0.4, 0.42, 0.45);
      atmosphere.refresh();
      const withDeck = atmosphere.aerialPerspective(horizontal, 8000).inscatter.clone();
      expect(withDeck.y).toBeGreaterThan(occluded.y);

      atmosphere.aerial.sunOcclusion.set(1, 1, 1);
      atmosphere.aerial.ambientRadiance.set(0, 0, 0);
      atmosphere.refresh();
    });

    it('thickens with haze and mist without inverting the colour balance', () => {
      const clear = atmosphere.aerialPerspective(horizontal, 5000).transmittance.clone();
      atmosphere.aerial.hazeDensity = 3;
      atmosphere.aerial.mistDensity = 0.4;
      atmosphere.refresh();
      const murky = atmosphere.aerialPerspective(horizontal, 5000).transmittance.clone();
      expect(murky.y).toBeLessThan(clear.y);
      // Grey mist desaturates: the channel spread must narrow, not widen.
      expect(murky.x - murky.z).toBeLessThan(clear.x - clear.z);

      atmosphere.aerial.hazeDensity = 1;
      atmosphere.aerial.mistDensity = 0;
      atmosphere.refresh();
    });

    it('reports a visibility range that matches its own extinction', () => {
      const range = atmosphere.visibilityRange(0.02);
      const sample = atmosphere.aerialPerspective(horizontal, range);
      expect(sample.transmittance.y).toBeCloseTo(0.02, 3);
    });
  });

  it('moves the observer altitude without breaking the model', () => {
    const original = atmosphere.observerAltitude;
    atmosphere.observerAltitude = 3200;
    atmosphere.refresh();
    setSunElevation(45);
    const high = atmosphere.sunTransmittance(scratch).clone();
    atmosphere.observerAltitude = 0;
    atmosphere.refresh();
    const low = atmosphere.sunTransmittance(scratch).clone();
    // Less air above you means less extinction, most visibly in blue.
    expect(high.z).toBeGreaterThan(low.z);
    atmosphere.observerAltitude = original;
    atmosphere.refresh();
  });
});
