/**
 * Tests for the wetness model.
 *
 * Wetness is the single most consequential shading decision in this project —
 * rain-soaked is the Blood Moor's resting state — and it is also the easiest
 * one to get subtly, permanently wrong. The properties pinned here are the ones
 * that separate a physical model from "multiply the albedo by 0.6 and drop the
 * roughness":
 *
 * - the two-interface reflectance is *lower* than the dry value for a typical
 *   dielectric, so anything that makes wet surfaces look shinier by raising F₀
 *   is compensating for a roughness term it failed to drop;
 * - absorption follows Beer-Lambert, so wet surfaces get more saturated rather
 *   than merely darker — which is precisely how the art direction avoids grey
 *   mush;
 * - porosity, not wetness, decides the split between darkening and gloss, so
 *   mud and iron respond in opposite ways to the same rain.
 */

import { describe, expect, it } from 'vitest';

import {
  F0_WATER,
  IOR_WATER,
  MAX_PATH_EXTENSION,
  WATER_ROUGHNESS,
  fresnelF0,
  iorFromF0,
  puddleMask,
  wetAlbedoChannel,
  wetRoughness,
  wetSpecularF0,
  wetnessSplit,
} from '../src/render/materials/Wetness';

describe('Fresnel helpers', () => {
  it('round-trips F0 through the implied IOR', () => {
    for (const f0 of [0.02, 0.035, 0.04, 0.05, 0.08]) {
      expect(fresnelF0(1, iorFromF0(f0))).toBeCloseTo(f0, 10);
    }
  });

  it('places water where physics does', () => {
    expect(F0_WATER).toBeCloseTo(((IOR_WATER - 1) / (IOR_WATER + 1)) ** 2, 12);
    expect(F0_WATER).toBeCloseTo(0.0204, 3);
  });

  it('recovers IOR 1.5 from the standard dielectric F0', () => {
    expect(iorFromF0(0.04)).toBeCloseTo(1.5, 6);
  });
});

describe('two-interface wet reflectance', () => {
  it('is the dry value at zero coverage', () => {
    expect(wetSpecularF0(0.04, 0)).toBeCloseTo(0.04, 12);
  });

  it('is *lower* than dry for a typical dielectric under a full film', () => {
    // The counter-intuitive but correct result. A wet dielectric does not
    // reflect more light at normal incidence; it reflects it in a far tighter
    // lobe. Anything that "fixes" this by raising F0 is papering over a
    // roughness term that was not dropped far enough.
    const wet = wetSpecularF0(0.04, 1);
    expect(wet).toBeLessThan(0.04);
    expect(wet).toBeCloseTo(0.0236, 3);
  });

  it('raises reflectance for a substrate below water', () => {
    // Skin sits at 0.028, below water's 0.0204 plus the substrate term, so its
    // wet reflectance rises rather than falls.
    expect(wetSpecularF0(0.02, 1)).toBeGreaterThan(0.02);
  });

  it('is monotonic in coverage', () => {
    let previous = wetSpecularF0(0.05, 0);
    for (let c = 0.1; c <= 1.0001; c += 0.1) {
      const value = wetSpecularF0(0.05, c);
      expect(value).toBeLessThanOrEqual(previous + 1e-12);
      previous = value;
    }
  });

  it('clamps coverage outside [0,1]', () => {
    expect(wetSpecularF0(0.04, -3)).toBeCloseTo(wetSpecularF0(0.04, 0), 12);
    expect(wetSpecularF0(0.04, 7)).toBeCloseTo(wetSpecularF0(0.04, 1), 12);
  });
});

describe('the porosity split', () => {
  it('sends everything to absorption for a fully porous substance', () => {
    const { soak, film } = wetnessSplit(1, 1, 1, 0);
    expect(soak).toBeCloseTo(1, 12);
    expect(film).toBeCloseTo(0, 12);
  });

  it('sends everything to film for a fully non-porous substance', () => {
    const { soak, film } = wetnessSplit(1, 1, 0, 0);
    expect(soak).toBeCloseTo(0, 12);
    expect(film).toBeCloseTo(1, 12);
  });

  it('lets standing water form a film even on porous ground', () => {
    // A puddle in mud is still a puddle. But it must not *add* to the film
    // term, or mud would become a mirror wherever it is also soaked.
    const { film } = wetnessSplit(1, 1, 0.95, 1);
    expect(film).toBeCloseTo(1, 12);
    const dry = wetnessSplit(1, 1, 0.95, 0);
    expect(dry.film).toBeCloseTo(0.05, 12);
  });

  it('scales both terms by exposure', () => {
    const full = wetnessSplit(1, 1, 0.5, 0);
    const half = wetnessSplit(1, 0.5, 0.5, 0);
    expect(half.soak).toBeCloseTo(full.soak * 0.5, 12);
    expect(half.film).toBeCloseTo(full.film * 0.5, 12);
  });

  it('never exceeds unity for out-of-range input', () => {
    const { soak, film } = wetnessSplit(4, 3, 1.5, 9);
    expect(soak).toBeLessThanOrEqual(1);
    expect(film).toBeLessThanOrEqual(1);
  });
});

describe('absorption', () => {
  it('leaves a dry surface untouched', () => {
    expect(wetAlbedoChannel(0.42, 0)).toBeCloseTo(0.42, 12);
  });

  it('darkens monotonically with soak', () => {
    let previous = wetAlbedoChannel(0.4, 0);
    for (let s = 0.1; s <= 1.0001; s += 0.1) {
      const value = wetAlbedoChannel(0.4, s);
      expect(value).toBeLessThan(previous);
      previous = value;
    }
  });

  it('follows the Beer-Lambert path extension exactly', () => {
    expect(wetAlbedoChannel(0.25, 0.5)).toBeCloseTo(
      Math.pow(0.25, 1 + 0.5 * MAX_PATH_EXTENSION),
      12,
    );
  });

  it('increases saturation rather than washing colour towards grey', () => {
    // This is the property the art direction depends on: a multiplicative
    // darkening preserves channel ratios and produces grey mush, while the
    // exponential form pushes the absorbing channels down faster and separates
    // hue. Wet mud must go *browner*, not merely darker.
    const dry: [number, number, number] = [0.3, 0.2, 0.1];
    const wet = dry.map((c) => wetAlbedoChannel(c, 0.6)) as [number, number, number];
    const dryRatio = dry[0] / dry[2];
    const wetRatio = wet[0] / wet[2];
    expect(wetRatio).toBeGreaterThan(dryRatio);
  });

  it('keeps black black and white white', () => {
    expect(wetAlbedoChannel(0, 1)).toBe(0);
    expect(wetAlbedoChannel(1, 1)).toBeCloseTo(1, 12);
  });
});

describe('roughness response', () => {
  it('is unchanged when bone dry', () => {
    expect(wetRoughness(0.8, 0, 0)).toBeCloseTo(0.8, 12);
  });

  it('collapses to the water film under full coverage', () => {
    expect(wetRoughness(0.9, 0, 1)).toBeCloseTo(WATER_ROUGHNESS, 12);
  });

  it('barely changes for a soaked but film-free surface', () => {
    // Soaked mud is darker and richer, not glossy. A model that glosses porous
    // surfaces up is the one that makes rain look like varnish.
    const soaked = wetRoughness(0.9, 1, 0);
    expect(soaked).toBeGreaterThan(0.6);
    expect(soaked).toBeLessThan(0.9);
  });

  it('separates mud from iron under identical rain', () => {
    const rain = 1;
    const mud = wetnessSplit(rain, 1, 0.95, 0);
    const iron = wetnessSplit(rain, 1, 0.02, 0);

    const mudRough = wetRoughness(0.85, mud.soak, mud.film);
    const ironRough = wetRoughness(0.3, iron.soak, iron.film);
    // Iron becomes a mirror; mud stays matte.
    expect(ironRough).toBeLessThan(0.06);
    expect(mudRough).toBeGreaterThan(0.6);

    const mudAlbedo = wetAlbedoChannel(0.25, mud.soak);
    const ironAlbedo = wetAlbedoChannel(0.25, iron.soak);
    // And the darkening goes the other way round: mud loses half its
    // reflectance, iron barely 3%.
    expect(mudAlbedo).toBeLessThan(0.13);
    expect(ironAlbedo).toBeGreaterThan(0.24);
    expect(ironAlbedo).toBeLessThan(0.25);
  });
});

describe('puddle mask', () => {
  it('is empty above the waterline and full below it', () => {
    expect(puddleMask(0, 0.5, 0.05)).toBe(0);
    expect(puddleMask(1, 0.5, 0.05)).toBe(1);
  });

  it('is exactly half at the waterline', () => {
    expect(puddleMask(0.5, 0.5, 0.1)).toBeCloseTo(0.5, 12);
  });

  it('has zero derivative at both edges, so the waterline does not shimmer', () => {
    const eps = 1e-5;
    const lo = 0.4;
    const hi = 0.6;
    const dLo = (puddleMask(lo + eps, 0.5, 0.1) - puddleMask(lo, 0.5, 0.1)) / eps;
    const dHi = (puddleMask(hi, 0.5, 0.1) - puddleMask(hi - eps, 0.5, 0.1)) / eps;
    expect(Math.abs(dLo)).toBeLessThan(1e-3);
    expect(Math.abs(dHi)).toBeLessThan(1e-3);
  });

  it('floors the softness so the edge can never alias to a hard step', () => {
    const value = puddleMask(0.5 + 0.01, 0.5, 0);
    expect(value).toBeGreaterThan(0);
    expect(value).toBeLessThan(1);
  });

  it('rises monotonically with cavity depth', () => {
    let previous = -1;
    for (let c = 0; c <= 1.0001; c += 0.05) {
      const value = puddleMask(c, 0.5, 0.1);
      expect(value).toBeGreaterThanOrEqual(previous);
      previous = value;
    }
  });
});
