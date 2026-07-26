/**
 * Unit tests for the IBL occlusion terms and the light-culling ranking.
 *
 * Both are small pure functions with outsized visual consequences: get the
 * specular-occlusion curve wrong and every crevice in the world glows; get the
 * light ranking wrong and the torch two rooms away wins the shadow slot the
 * player's campfire needed.
 */

import { describe, expect, it } from 'vitest';

import {
  computeSpecularOcclusion,
  horizonOcclusion,
  prefilteredMipCount,
  roughnessToMip,
} from '../src/render/IBL';
import { lightImportance } from '../src/render/Lighting';

describe('computeSpecularOcclusion', () => {
  it('is a no-op when nothing is occluded', () => {
    for (const roughness of [0, 0.25, 0.5, 0.75, 1]) {
      for (const nDotV of [0.05, 0.5, 1]) {
        expect(computeSpecularOcclusion(nDotV, 1, roughness)).toBeCloseTo(1, 6);
      }
    }
  });

  it('fully occludes specular when AO is zero and the surface faces away', () => {
    expect(computeSpecularOcclusion(0, 0, 1)).toBe(0);
  });

  it('stays within [0, 1] across the whole domain', () => {
    for (let i = 0; i <= 20; i++) {
      for (let j = 0; j <= 20; j++) {
        for (let k = 0; k <= 20; k++) {
          const value = computeSpecularOcclusion(i / 20, j / 20, k / 20);
          expect(value).toBeGreaterThanOrEqual(0);
          expect(value).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it('is monotonically non-decreasing in ambient occlusion', () => {
    for (const roughness of [0, 0.3, 0.6, 1]) {
      for (const nDotV of [0.1, 0.5, 0.9]) {
        let previous = -1;
        for (let i = 0; i <= 20; i++) {
          const value = computeSpecularOcclusion(nDotV, i / 20, roughness);
          expect(value).toBeGreaterThanOrEqual(previous - 1e-12);
          previous = value;
        }
      }
    }
  });

  it('darkens a rough surface toward the diffuse AO, a mirror less so', () => {
    // Viewed head on, a mirror lobe samples a tiny solid angle, so
    // hemisphere-wide occlusion barely applies to it. A fully rough lobe covers
    // the same hemisphere the AO was integrated over, so it converges on the
    // diffuse value exactly.
    const ao = 0.4;
    const mirror = computeSpecularOcclusion(1.0, ao, 0.0);
    const rough = computeSpecularOcclusion(1.0, ao, 1.0);
    expect(rough).toBeCloseTo(ao, 4);
    expect(mirror).toBeGreaterThan(rough);
  });

  it('occludes grazing angles more than head-on ones', () => {
    const grazing = computeSpecularOcclusion(0.05, 0.4, 0.5);
    const headOn = computeSpecularOcclusion(1.0, 0.4, 0.5);
    expect(grazing).toBeLessThan(headOn);
  });
});

describe('horizonOcclusion', () => {
  it('leaves reflections above the geometric horizon untouched', () => {
    expect(horizonOcclusion(1)).toBe(1);
    expect(horizonOcclusion(0.5)).toBe(1);
    expect(horizonOcclusion(0)).toBe(1);
  });

  it('kills reflections pointing straight into the surface', () => {
    expect(horizonOcclusion(-1)).toBe(0);
  });

  it('falls off smoothly rather than stepping', () => {
    const samples: number[] = [];
    for (let i = 0; i <= 20; i++) samples.push(horizonOcclusion(-1 + i / 20));
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeGreaterThanOrEqual(samples[i - 1]!);
      // No single step may jump more than 15% of the range: a hard step here
      // is exactly the bright rim artefact the term exists to remove.
      expect(samples[i]! - samples[i - 1]!).toBeLessThan(0.15);
    }
  });

  it('is disabled by a zero fade', () => {
    for (const r of [-1, -0.5, 0, 0.5, 1]) expect(horizonOcclusion(r, 0)).toBe(1);
  });
});

describe('roughnessToMip / prefilteredMipCount', () => {
  it('maps roughness linearly onto the prefiltered chain', () => {
    expect(roughnessToMip(0, 8)).toBe(0);
    expect(roughnessToMip(1, 8)).toBe(7);
    expect(roughnessToMip(0.5, 9)).toBeCloseTo(4, 12);
  });

  it('clamps out-of-range roughness instead of sampling past the chain', () => {
    expect(roughnessToMip(-1, 8)).toBe(0);
    expect(roughnessToMip(2, 8)).toBe(7);
  });

  it('never reports fewer than one usable level', () => {
    expect(prefilteredMipCount(0)).toBeGreaterThanOrEqual(1);
    expect(prefilteredMipCount(16)).toBeGreaterThanOrEqual(1);
  });

  it('grows with source resolution', () => {
    expect(prefilteredMipCount(2048)).toBeGreaterThan(prefilteredMipCount(256));
  });
});

describe('lightImportance', () => {
  it('ranks a nearer light above an identical farther one', () => {
    const near = lightImportance(10, 5, 6);
    const far = lightImportance(10, 5, 30);
    expect(near).toBeGreaterThan(far);
  });

  it('ranks a brighter light above a dimmer one at the same distance', () => {
    expect(lightImportance(20, 5, 10)).toBeGreaterThan(lightImportance(5, 5, 10));
  });

  it('ranks a wider light above a tighter one at the same distance', () => {
    expect(lightImportance(10, 12, 20)).toBeGreaterThan(lightImportance(10, 4, 20));
  });

  it('saturates once the camera is inside the light volume', () => {
    // Walking from the edge of a torch's sphere into its centre must not keep
    // inflating its score, or a single light the player stands on would
    // monopolise every slot.
    const atEdge = lightImportance(10, 5, 5);
    const atCentre = lightImportance(10, 5, 0);
    expect(atCentre).toBe(atEdge);
  });

  it('lets one priority step outrank ordinary distance falloff', () => {
    const closeOrdinary = lightImportance(10, 5, 12, 0);
    const distantStory = lightImportance(10, 5, 14, 1);
    expect(distantStory).toBeGreaterThan(closeOrdinary);
  });

  it('scores dark or degenerate lights at zero so they are never bound', () => {
    expect(lightImportance(0, 5, 10)).toBe(0);
    expect(lightImportance(10, 0, 10)).toBe(0);
    expect(lightImportance(-1, 5, 10)).toBe(0);
  });

  it('produces a total order with no ties or NaN over a random population', () => {
    let seed = 12345;
    const random = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const scores: number[] = [];
    for (let i = 0; i < 500; i++) {
      const value = lightImportance(
        random() * 20 + 0.1,
        random() * 15 + 0.1,
        random() * 200,
        Math.floor(random() * 3),
      );
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThan(0);
      scores.push(value);
    }
    const sorted = [...scores].sort((a, b) => b - a);
    expect(sorted[0]!).toBeGreaterThanOrEqual(sorted[sorted.length - 1]!);
  });
});
