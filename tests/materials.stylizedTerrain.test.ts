/**
 * The stylized terrain material's CPU-side reference math and its palette.
 *
 * The TSL graph itself cannot be evaluated without a GPU, so what is pinned
 * here is (a) the shaping functions the graph mirrors, and (b) the authored
 * palette — because the palette *is* the art direction now that there are no
 * textures, and a value drifting out of range is the difference between
 * "deliberately stylized" and "somebody typed a number".
 */

import { describe, expect, it } from 'vitest';

import {
  crispBand,
  crispWeight,
  GRIMDARK_GROUND,
  rockCoverage,
  slopeShade,
  type TerrainLayer,
} from '../src/render/materials/StylizedTerrain';

const LAYERS: ReadonlyArray<readonly [string, TerrainLayer]> = [
  ['earth', GRIMDARK_GROUND.earth],
  ['cover', GRIMDARK_GROUND.cover],
  ['rock', GRIMDARK_GROUND.rock],
];

describe('crispBand', () => {
  it('narrows monotonically as crispness rises', () => {
    let previous = Number.POSITIVE_INFINITY;
    for (let c = 0; c <= 1.0001; c += 0.1) {
      const band = crispBand(c);
      expect(band).toBeLessThan(previous);
      previous = band;
    }
  });

  it('is never zero, so the boundary never aliases into a hard step', () => {
    // A `smoothstep` with a zero-width band is a `step`, and a step edge across
    // a ground plane crawls under any camera motion no matter what the AA does.
    expect(crispBand(1)).toBeGreaterThan(0);
    expect(crispBand(2)).toBeGreaterThan(0);
  });

  it('spans a full linear ramp at zero crispness', () => {
    expect(crispBand(0)).toBeCloseTo(0.52, 6);
  });

  it('clamps out-of-range input rather than producing a negative band', () => {
    expect(crispBand(-1)).toBe(crispBand(0));
    expect(crispBand(5)).toBe(crispBand(1));
  });
});

describe('crispWeight', () => {
  it('is a partition: 0 below the band, 1 above it', () => {
    const c = GRIMDARK_GROUND.crispness;
    expect(crispWeight(0, c)).toBe(0);
    expect(crispWeight(1, c)).toBe(1);
    expect(crispWeight(0.5, c)).toBeCloseTo(0.5, 6);
  });

  it('is monotonic across the whole range', () => {
    let previous = -1;
    for (let w = 0; w <= 1.0001; w += 0.02) {
      const value = crispWeight(w, 0.72);
      expect(value).toBeGreaterThanOrEqual(previous - 1e-12);
      previous = value;
    }
  });

  it('actually sharpens: the transition occupies less of the range', () => {
    // The whole trick. A soft cross-fade between two desaturated browns is mud;
    // a narrow one is two materials meeting. Measure the width of the region
    // that is neither fully one layer nor fully the other.
    const width = (crispness: number): number => {
      let count = 0;
      for (let w = 0; w <= 1.0001; w += 0.001) {
        const v = crispWeight(w, crispness);
        if (v > 0.02 && v < 0.98) count++;
      }
      return count;
    };
    expect(width(0.9)).toBeLessThan(width(0.72));
    expect(width(0.72)).toBeLessThan(width(0.2));
  });

  it('degenerates to a near-linear ramp at zero crispness', () => {
    // Not exactly linear — it is still a smoothstep — but it must cover the
    // whole range rather than clipping, which is what "0 = cross-fade" means.
    expect(crispWeight(0.1, 0)).toBeGreaterThan(0);
    expect(crispWeight(0.9, 0)).toBeLessThan(1);
  });
});

describe('rockCoverage', () => {
  const { rockSlopeStart: start, rockSlopeEnd: end } = GRIMDARK_GROUND;

  it('shows no rock on a flat and full rock on a wall', () => {
    expect(rockCoverage(1, start, end)).toBe(0);
    expect(rockCoverage(0, start, end)).toBe(1);
  });

  it('rises monotonically as the surface tips over', () => {
    let previous = -1;
    for (let y = 1; y >= 0; y -= 0.02) {
      const value = rockCoverage(y, start, end);
      expect(value).toBeGreaterThanOrEqual(previous - 1e-12);
      previous = value;
    }
  });

  it('leaves gentle undulation alone', () => {
    // The moor is displaced terrain; if rock crept onto every rise the whole
    // field would go grey. cos(20 deg) is a noticeable slope and must be clean.
    expect(rockCoverage(Math.cos((20 * Math.PI) / 180), start, end)).toBe(0);
  });

  it('handles a degenerate range without dividing by zero', () => {
    expect(Number.isFinite(rockCoverage(0.7, 0.5, 0.5))).toBe(true);
  });
});

describe('slopeShade', () => {
  it('leaves a flat surface completely untouched', () => {
    expect(slopeShade(1, GRIMDARK_GROUND.slopeShading)).toBe(1);
  });

  it('darkens a vertical face by exactly the authored strength', () => {
    expect(slopeShade(0, 0.55)).toBeCloseTo(0.45, 6);
  });

  it('is quadratic, so gentle ground barely moves', () => {
    // The point of squaring: without it the whole field greys out, which is the
    // "grey mush" the art direction explicitly forbids. A 20 degree slope must
    // lose only a few percent.
    const gentle = slopeShade(Math.cos((20 * Math.PI) / 180), 0.55);
    expect(gentle).toBeGreaterThan(0.98);
  });

  it('never goes negative or above one', () => {
    for (let y = -0.5; y <= 1.5; y += 0.1) {
      const value = slopeShade(y, 1);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });
});

describe('GRIMDARK_GROUND palette', () => {
  it('keeps every layer dark enough for firelight to read against it', () => {
    // The encampment's only warm light is a bonfire. A ground already carrying
    // its own bright warm brown gives the fire nowhere to go. sRGB luma stays
    // well under half.
    for (const [name, layer] of LAYERS) {
      const r = ((layer.color >> 16) & 0xff) / 255;
      const g = ((layer.color >> 8) & 0xff) / 255;
      const b = (layer.color & 0xff) / 255;
      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      expect(luma, name).toBeGreaterThan(0.08);
      expect(luma, name).toBeLessThan(0.42);
    }
  });

  it('keeps every layer desaturated', () => {
    // Grimdark, not autumnal. Max-minus-min channel over max is the HSV
    // saturation; anything above ~0.4 starts reading as a colour rather than as
    // a tone.
    for (const [name, layer] of LAYERS) {
      const r = (layer.color >> 16) & 0xff;
      const g = (layer.color >> 8) & 0xff;
      const b = layer.color & 0xff;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      expect((max - min) / max, name).toBeLessThan(0.4);
    }
  });

  it('separates the layers enough to read as different materials', () => {
    // Two colours a boundary cannot be seen between are one colour with extra
    // instructions. Require a real perceptual gap between earth and cover.
    const luma = (hex: number): number =>
      (0.2126 * ((hex >> 16) & 0xff) + 0.7152 * ((hex >> 8) & 0xff) + 0.0722 * (hex & 0xff)) / 255;
    expect(Math.abs(luma(GRIMDARK_GROUND.cover.color) - luma(GRIMDARK_GROUND.earth.color)))
      .toBeGreaterThan(0.04);
    expect(Math.abs(luma(GRIMDARK_GROUND.rock.color) - luma(GRIMDARK_GROUND.earth.color)))
      .toBeGreaterThan(0.04);
  });

  it('keeps terrain rough, and rock the least porous', () => {
    for (const [name, layer] of LAYERS) {
      expect(layer.roughness, name).toBeGreaterThan(0.7);
      expect(layer.roughness, name).toBeLessThanOrEqual(1);
      expect(layer.porosity, name).toBeGreaterThanOrEqual(0);
      expect(layer.porosity, name).toBeLessThanOrEqual(1);
      expect(layer.variation, name).toBeGreaterThan(0);
      expect(layer.variation, name).toBeLessThan(0.5);
    }
    // Earth soaks and darkens; rock glosses without darkening. Same ordering as
    // the archetype table's porosity, which this palette replaces for terrain.
    expect(GRIMDARK_GROUND.earth.porosity).toBeGreaterThan(GRIMDARK_GROUND.rock.porosity);
  });

  it('keeps the noise wavelengths separated so they cannot beat', () => {
    // Two bands an octave apart produce a visible interference pattern. These
    // are more than 5x apart, which is far outside anything the eye locks onto.
    expect(GRIMDARK_GROUND.macroMetres / GRIMDARK_GROUND.boundaryMetres).toBeGreaterThan(4);
    expect(GRIMDARK_GROUND.boundaryMetres).toBeGreaterThan(1);
  });

  it('keeps the rock slope thresholds ordered', () => {
    // `start` is the flatter of the two: coverage rises as the normal tips.
    expect(GRIMDARK_GROUND.rockSlopeStart).toBeGreaterThan(GRIMDARK_GROUND.rockSlopeEnd);
    expect(GRIMDARK_GROUND.rockSlopeStart).toBeLessThanOrEqual(1);
    expect(GRIMDARK_GROUND.rockSlopeEnd).toBeGreaterThanOrEqual(0);
  });

  it('keeps boundary perturbation from swamping the mask entirely', () => {
    // The coverage mask the scenes pass in is the art direction. Noise is
    // allowed to make its edge ragged, not to overrule it.
    expect(GRIMDARK_GROUND.boundaryNoise).toBeLessThan(0.5);
    expect(GRIMDARK_GROUND.mottle).toBeLessThan(0.35);
  });
});
