/**
 * Unit tests for the colour-grading math.
 *
 * Three things here are worth guarding with real assertions rather than a
 * screenshot: the curve interpolant must not overshoot (an overshooting tone
 * curve inverts contrast in the toe), the white-balance matrix must move the
 * image in the direction the artist asked for (the sign was wrong first time,
 * and a warm "cool" grade looks plausible until you A/B it), and the `.cube`
 * parser must reject a file it cannot interpret rather than produce a
 * confidently wrong LUT.
 */

import { describe, expect, it } from 'vitest';

import {
  GRIMDARK_GRADE,
  NEUTRAL_GRADE,
  evaluateCurve,
  identityLUT,
  kelvinToChromaticity,
  parseCubeLUT,
  whiteBalanceMatrix,
} from '../src/render/post/ColorGrade';

describe('evaluateCurve', () => {
  it('is the identity for an empty control set', () => {
    for (const x of [0, 0.25, 0.5, 0.9, 1]) expect(evaluateCurve([], x)).toBe(x);
  });

  it('passes exactly through its control points', () => {
    const points: ReadonlyArray<readonly [number, number]> = [
      [0, 0.05],
      [0.5, 0.4],
      [1, 0.95],
    ];
    for (const [x, y] of points) expect(evaluateCurve(points, x)).toBeCloseTo(y, 10);
  });

  it('clamps outside the control range', () => {
    const points: ReadonlyArray<readonly [number, number]> = [
      [0.2, 0.1],
      [0.8, 0.9],
    ];
    expect(evaluateCurve(points, 0)).toBeCloseTo(0.1, 12);
    expect(evaluateCurve(points, 1)).toBeCloseTo(0.9, 12);
  });

  it('never overshoots a monotone control set', () => {
    // The reason for Fritsch-Carlson rather than a natural cubic spline. A
    // classic overshoot case: a long flat run followed by a sharp rise.
    const points: ReadonlyArray<readonly [number, number]> = [
      [0, 0],
      [0.4, 0.02],
      [0.5, 0.5],
      [0.6, 0.98],
      [1, 1],
    ];
    for (let i = 0; i <= 1000; i++) {
      const y = evaluateCurve(points, i / 1000);
      expect(y).toBeGreaterThanOrEqual(-1e-9);
      expect(y).toBeLessThanOrEqual(1 + 1e-9);
    }
  });

  it('stays monotone when the control points are monotone', () => {
    const points: ReadonlyArray<readonly [number, number]> = [
      [0, 0],
      [0.25, 0.05],
      [0.5, 0.5],
      [0.75, 0.95],
      [1, 1],
    ];
    let previous = -Infinity;
    for (let i = 0; i <= 2000; i++) {
      const y = evaluateCurve(points, i / 2000);
      expect(y).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = y;
    }
  });

  it('reproduces the shipping green toe', () => {
    expect(evaluateCurve(GRIMDARK_GRADE.curveG, 0)).toBeCloseTo(0.008, 10);
    expect(evaluateCurve(GRIMDARK_GRADE.curveG, 1)).toBeCloseTo(1, 10);
    // The lift must decay: by the quarter tone it is already gone.
    expect(evaluateCurve(GRIMDARK_GRADE.curveG, 0.25)).toBeCloseTo(0.25, 6);
  });
});

describe('kelvinToChromaticity', () => {
  it('puts 6500 K close to D65', () => {
    // Not exactly: D65 is a *daylight* illuminant and sits a little off the
    // Planckian locus this fit describes, so the reference-white round trip is
    // accurate to about 0.006 in xy. That is well inside the tolerance the
    // white-balance matrix needs, and it is why `whiteBalanceMatrix` special-
    // cases a zero shift to the exact identity rather than to "nearly".
    const [x, y] = kelvinToChromaticity(6500);
    expect(Math.abs(x - 0.3127)).toBeLessThan(0.01);
    expect(Math.abs(y - 0.329)).toBeLessThan(0.01);
  });

  it('moves toward red as the temperature drops', () => {
    const cool = kelvinToChromaticity(9000);
    const warm = kelvinToChromaticity(3000);
    expect(warm[0]).toBeGreaterThan(cool[0]);
  });

  it('clamps rather than diverging outside its fit range', () => {
    for (const k of [1, 500, 100000]) {
      const [x, y] = kelvinToChromaticity(k);
      expect(Number.isFinite(x)).toBe(true);
      expect(Number.isFinite(y)).toBe(true);
      expect(x).toBeGreaterThan(0);
      expect(y).toBeGreaterThan(0);
    }
  });
});

describe('whiteBalanceMatrix', () => {
  /** `M · (1,1,1)`, column-major input as three's `Matrix3.fromArray` wants. */
  function applyToWhite(m: number[]): [number, number, number] {
    return [
      (m[0] ?? 0) + (m[3] ?? 0) + (m[6] ?? 0),
      (m[1] ?? 0) + (m[4] ?? 0) + (m[7] ?? 0),
      (m[2] ?? 0) + (m[5] ?? 0) + (m[8] ?? 0),
    ];
  }

  it('is the identity with no shift', () => {
    expect(whiteBalanceMatrix(0, 0)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('makes the image cooler for a negative temperature shift', () => {
    // The sign convention that was wrong first time. Negative must mean bluer:
    // blue gain up, red gain down.
    const [r, , b] = applyToWhite(whiteBalanceMatrix(-1000, 0));
    expect(b).toBeGreaterThan(1);
    expect(r).toBeLessThan(b);
  });

  it('makes the image warmer for a positive temperature shift', () => {
    const [r, , b] = applyToWhite(whiteBalanceMatrix(1000, 0));
    expect(b).toBeLessThan(1);
    expect(r).toBeGreaterThan(b);
  });

  it('scales monotonically with the size of the shift', () => {
    const small = applyToWhite(whiteBalanceMatrix(-300, 0))[2];
    const large = applyToWhite(whiteBalanceMatrix(-1500, 0))[2];
    expect(large).toBeGreaterThan(small);
  });

  it('pushes green for a positive tint', () => {
    const [r, g, b] = applyToWhite(whiteBalanceMatrix(0, 0.5));
    expect(g).toBeGreaterThan((r + b) / 2);
  });

  it('keeps the shipping grade a small correction', () => {
    // A white balance that changes any channel by more than ~20% is a colour
    // filter, not a balance.
    const gains = applyToWhite(
      whiteBalanceMatrix(GRIMDARK_GRADE.temperature, GRIMDARK_GRADE.tint),
    );
    for (const gain of gains) {
      expect(gain).toBeGreaterThan(0.8);
      expect(gain).toBeLessThan(1.2);
    }
  });
});

describe('parseCubeLUT', () => {
  const tiny = [
    '# a comment',
    'TITLE "tiny"',
    'LUT_3D_SIZE 2',
    '',
    '0 0 0',
    '1 0 0',
    '0 1 0',
    '1 1 0',
    '0 0 1',
    '1 0 1',
    '0 1 1',
    '1 1 1',
  ].join('\n');

  it('parses size, title and data in red-fastest order', () => {
    const lut = parseCubeLUT(tiny);
    expect(lut.size).toBe(2);
    expect(lut.title).toBe('tiny');
    expect(lut.data).toHaveLength(24);
    expect(Array.from(lut.data.slice(0, 6))).toEqual([0, 0, 0, 1, 0, 0]);
  });

  it('normalises a non-unit domain', () => {
    const scaled = tiny.replace('LUT_3D_SIZE 2', 'LUT_3D_SIZE 2\nDOMAIN_MAX 2 2 2');
    const lut = parseCubeLUT(scaled);
    expect(lut.data[3]).toBeCloseTo(0.5, 10);
  });

  it('rejects a 1D LUT rather than misreading it as 3D', () => {
    expect(() => parseCubeLUT('LUT_1D_SIZE 4\n0 0 0\n1 1 1\n')).toThrow(/1D/);
  });

  it('rejects a truncated file', () => {
    expect(() => parseCubeLUT('LUT_3D_SIZE 2\n0 0 0\n1 1 1\n')).toThrow(/expected/);
  });

  it('rejects a file with no size declaration', () => {
    expect(() => parseCubeLUT('0 0 0\n1 1 1\n')).toThrow(/LUT_3D_SIZE/);
  });
});

describe('identityLUT', () => {
  it('maps every lattice point to itself', () => {
    const size = 5;
    const { data } = identityLUT(size);
    const step = 1 / (size - 1);
    for (let b = 0; b < size; b++) {
      for (let g = 0; g < size; g++) {
        for (let r = 0; r < size; r++) {
          const i = ((b * size + g) * size + r) * 3;
          expect(data[i]).toBeCloseTo(r * step, 6);
          expect(data[i + 1]).toBeCloseTo(g * step, 6);
          expect(data[i + 2]).toBeCloseTo(b * step, 6);
        }
      }
    }
  });

  it('round-trips through the .cube parser', () => {
    const size = 3;
    const { data } = identityLUT(size);
    const lines = ['LUT_3D_SIZE 3'];
    for (let i = 0; i < data.length; i += 3) {
      lines.push(`${data[i]} ${data[i + 1]} ${data[i + 2]}`);
    }
    const parsed = parseCubeLUT(lines.join('\n'));
    expect(parsed.size).toBe(size);
    for (let i = 0; i < data.length; i++) {
      expect(parsed.data[i]).toBeCloseTo(data[i] ?? 0, 6);
    }
  });
});

describe('presets', () => {
  it('NEUTRAL_GRADE is a genuine pass-through', () => {
    expect(NEUTRAL_GRADE.temperature).toBe(0);
    expect(NEUTRAL_GRADE.saturation).toBe(1);
    expect(NEUTRAL_GRADE.contrast).toBe(1);
    expect(NEUTRAL_GRADE.vignette).toBe(0);
    expect(NEUTRAL_GRADE.grain).toBe(0);
    expect(NEUTRAL_GRADE.chromaticAberration).toBe(0);
    expect(NEUTRAL_GRADE.gain).toEqual([1, 1, 1]);
    expect(NEUTRAL_GRADE.gamma).toEqual([1, 1, 1]);
    expect(NEUTRAL_GRADE.lift).toEqual([0, 0, 0]);
  });

  it('GRIMDARK_GRADE stays inside restrained bounds', () => {
    // Every number in the shipping look must be small enough that the player
    // does not see the grade, only the mood.
    for (const value of GRIMDARK_GRADE.lift) expect(Math.abs(value)).toBeLessThan(0.05);
    for (const value of GRIMDARK_GRADE.gain) expect(Math.abs(value - 1)).toBeLessThan(0.06);
    for (const value of GRIMDARK_GRADE.gamma) expect(Math.abs(value - 1)).toBeLessThan(0.06);
    expect(Math.abs(GRIMDARK_GRADE.saturation - 1)).toBeLessThan(0.2);
    expect(Math.abs(GRIMDARK_GRADE.contrast - 1)).toBeLessThan(0.15);
    expect(GRIMDARK_GRADE.vignette).toBeLessThan(0.45);
    expect(GRIMDARK_GRADE.grain).toBeLessThan(0.03);
    expect(GRIMDARK_GRADE.chromaticAberration).toBeLessThan(1.5);
  });

  it('keeps the contrast pivot near display middle grey', () => {
    // The grade runs on display-*encoded* values, where middle grey (linear
    // 0.18) sits at about 0.47. A pivot authored for linear values instead —
    // anywhere near 0.18 — crushes the shadows; one near 1.0 lifts them.
    for (const grade of [NEUTRAL_GRADE, GRIMDARK_GRADE]) {
      expect(grade.contrastPivot).toBeGreaterThan(0.35);
      expect(grade.contrastPivot).toBeLessThan(0.6);
    }
  });
});
