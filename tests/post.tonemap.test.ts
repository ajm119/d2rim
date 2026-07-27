/**
 * Unit tests for the exposure model.
 *
 * These are the numbers that decide whether the game is visible at all, and
 * they are the ones that were wrong first time: the module originally quoted
 * photographic EV100 values (overcast ≈ 12) against a renderer whose lighting
 * is in relative units where overcast ≈ 1, a 4000x error that renders black.
 */

import { describe, expect, it } from 'vitest';

import {
  AGX_LOOKS,
  adaptationWeight,
  ev100FromCamera,
  ev100FromLuminance,
  exposureFromEV100,
} from '../src/render/post/Tonemap';

describe('exposureFromEV100', () => {
  it('is the reciprocal of the saturation-based maximum luminance', () => {
    expect(exposureFromEV100(0)).toBeCloseTo(1 / 1.2, 12);
    expect(exposureFromEV100(1)).toBeCloseTo(1 / 2.4, 12);
    expect(exposureFromEV100(-1)).toBeCloseTo(1 / 0.6, 12);
  });

  it('halves for every stop', () => {
    for (let ev = -4; ev <= 16; ev++) {
      expect(exposureFromEV100(ev + 1)).toBeCloseTo(exposureFromEV100(ev) / 2, 12);
    }
  });
});

describe('ev100FromLuminance', () => {
  it('uses the K = 12.5 reflected-light calibration', () => {
    // L · S/K = 1 exactly when L = 0.125, which must therefore be EV100 0.
    expect(ev100FromLuminance(0.125)).toBeCloseTo(0, 12);
  });

  it('gains a stop per doubling of luminance', () => {
    expect(ev100FromLuminance(2) - ev100FromLuminance(1)).toBeCloseTo(1, 12);
  });

  it('does not return -Infinity for a black frame', () => {
    expect(Number.isFinite(ev100FromLuminance(0))).toBe(true);
  });
});

describe('ev100FromCamera', () => {
  it('puts the sunny-16 rule where it belongs', () => {
    // f/16 at 1/100 s, ISO 100 is the canonical bright-daylight exposure and
    // should land at EV100 ~15.
    expect(ev100FromCamera(16, 1 / 100, 100)).toBeCloseTo(14.64, 2);
  });

  it('loses a stop per doubling of ISO', () => {
    const base = ev100FromCamera(2.8, 1 / 60, 100);
    expect(ev100FromCamera(2.8, 1 / 60, 200)).toBeCloseTo(base - 1, 12);
  });

  it('gains a stop per stop of aperture', () => {
    const base = ev100FromCamera(2, 1 / 60, 100);
    expect(ev100FromCamera(2 * Math.SQRT2, 1 / 60, 100)).toBeCloseTo(base + 1, 10);
  });

  it('agrees with the luminance form for a correctly metered scene', () => {
    // Metering L to middle grey and then exposing at that EV must return the
    // grey card to 0.18 through the 1/(1.2·2^EV) relation, within the constant
    // ratio 0.18·1.2·(100/12.5)/1 — i.e. exposure·L is invariant.
    const products = [0.02, 0.2, 2, 20].map(
      (l) => exposureFromEV100(ev100FromLuminance(l)) * l,
    );
    for (const p of products) expect(p).toBeCloseTo(products[0]!, 12);
  });
});

describe('adaptationWeight', () => {
  it('is 0 at dt = 0 and approaches 1 for long steps', () => {
    expect(adaptationWeight(0, 1)).toBeCloseTo(0, 12);
    expect(adaptationWeight(100, 1)).toBeCloseTo(1, 12);
  });

  it('reaches 1 - 1/e after exactly one time constant', () => {
    expect(adaptationWeight(0.5, 0.5)).toBeCloseTo(1 - Math.exp(-1), 12);
  });

  it('is frame-rate independent', () => {
    // Two half-steps must leave the same residual as one whole step. The naive
    // `lerp(a, b, k)` with a constant k fails this and silently adapts twice as
    // fast at 120 Hz as at 60.
    const tau = 0.8;
    const whole = 1 - adaptationWeight(1 / 30, tau);
    const halves = (1 - adaptationWeight(1 / 60, tau)) ** 2;
    expect(halves).toBeCloseTo(whole, 12);
  });

  it('degenerates to an instant snap when the time constant is zero', () => {
    expect(adaptationWeight(1 / 60, 0)).toBe(1);
  });
});

describe('AGX_LOOKS', () => {
  it('has a genuinely neutral "none"', () => {
    const look = AGX_LOOKS.none;
    expect(look.slope).toEqual([1, 1, 1]);
    expect(look.offset).toEqual([0, 0, 0]);
    expect(look.power).toEqual([1, 1, 1]);
    expect(look.saturation).toBe(1);
  });

  it('keeps the shipping look restrained', () => {
    // The project brief is restraint. A look with a power much above ~1.3 or a
    // saturation much above ~1.2 reads as a filter rather than as a grade.
    const look = AGX_LOOKS.grimdark;
    for (const p of look.power) {
      expect(p).toBeGreaterThan(1);
      expect(p).toBeLessThan(1.3);
    }
    expect(look.saturation).toBeGreaterThan(1);
    expect(look.saturation).toBeLessThan(1.2);
  });
});
