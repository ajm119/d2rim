import { describe, expect, it } from 'vitest';

import { Clock, FixedStepAccumulator, createTimeState } from '../src/core/Time';

const FIXED = 1 / 60;

/**
 * The fixed-timestep accumulator is the piece of the engine that determinism
 * rests on: physics, replay and golden-image capture all assume the same input
 * sequence yields the same step counts. These tests pin that behaviour along
 * with the two failure modes that bite in practice — the spiral of death, and
 * multi-second deltas from a backgrounded tab.
 */
describe('FixedStepAccumulator', () => {
  it('runs exactly one step for one exact fixed delta', () => {
    const acc = new FixedStepAccumulator();
    expect(acc.begin(FIXED)).toBe(1);
    expect(acc.pending).toBeCloseTo(0, 10);
  });

  it('runs no steps until enough time has accumulated', () => {
    const acc = new FixedStepAccumulator();

    expect(acc.begin(0.009)).toBe(0); // 0.009 < 1/60 (0.01666...)
    expect(acc.begin(0.009)).toBe(1); // 0.018 >= 1/60
  });

  it('carries the remainder across frames instead of dropping it', () => {
    const acc = new FixedStepAccumulator();
    // 100 frames at 10 ms is 1.0 s of simulation == 60 steps, exactly.
    let steps = 0;
    for (let i = 0; i < 100; i++) steps += acc.begin(0.01);

    expect(steps).toBe(60);
    expect(acc.pending).toBeLessThan(FIXED);
  });

  it('runs several steps when a frame is long', () => {
    const acc = new FixedStepAccumulator();
    // 50 ms is three whole 16.67 ms slices with a remainder.
    expect(acc.begin(0.05)).toBe(3);
    expect(acc.pending).toBeCloseTo(0.05 - 3 * FIXED, 10);
  });

  it('never exceeds maxSubSteps in a single frame', () => {
    const acc = new FixedStepAccumulator({ maxSubSteps: 3 });
    expect(acc.begin(1.0)).toBe(3);
    expect(acc.wasStarved).toBe(true);
    // Backlog is discarded, so the next normal frame behaves normally.
    expect(acc.begin(FIXED)).toBe(1);
  });

  it('clamps an absurd delta from a resumed background tab', () => {
    const acc = new FixedStepAccumulator({ maxDelta: 0.25, maxSubSteps: 60 });
    const steps = acc.begin(30);

    expect(acc.wasClamped).toBe(true);
    // 0.25 s clamped, not 30 s: 15 steps, not 1800.
    expect(steps).toBe(15);
  });

  it('does not spiral: a permanently overloaded loop stays bounded', () => {
    const acc = new FixedStepAccumulator({ maxSubSteps: 5 });
    for (let i = 0; i < 200; i++) {
      expect(acc.begin(0.5)).toBeLessThanOrEqual(5);
    }
    expect(acc.pending).toBeLessThan(FIXED);
  });

  it('treats NaN, Infinity and negative deltas as zero', () => {
    const acc = new FixedStepAccumulator();

    expect(acc.begin(Number.NaN)).toBe(0);
    expect(acc.begin(Number.POSITIVE_INFINITY)).toBe(0);
    expect(acc.begin(-1)).toBe(0);
    expect(acc.pending).toBe(0);
  });

  it('reports alpha as the fraction of a step still pending', () => {
    const acc = new FixedStepAccumulator();

    acc.begin(FIXED * 0.5);
    expect(acc.alpha).toBeCloseTo(0.5, 6);

    acc.begin(FIXED * 0.25);
    expect(acc.alpha).toBeCloseTo(0.75, 6);

    // Crossing a whole step wraps alpha back into [0, 1).
    acc.begin(FIXED * 0.5);
    expect(acc.alpha).toBeGreaterThanOrEqual(0);
    expect(acc.alpha).toBeLessThan(1);
  });

  it('keeps alpha within [0, 1) for a long run of irregular frames', () => {
    const acc = new FixedStepAccumulator();
    let t = 0;
    for (let i = 0; i < 500; i++) {
      // Deterministic pseudo-jitter between roughly 4 and 28 ms.
      t += 1;
      const delta = 0.016 + Math.sin(t * 12.9898) * 0.012;
      acc.begin(delta);
      expect(acc.alpha).toBeGreaterThanOrEqual(0);
      expect(acc.alpha).toBeLessThan(1);
    }
  });

  it('applies time scale, including a full freeze', () => {
    const acc = new FixedStepAccumulator();

    expect(acc.begin(FIXED, 0)).toBe(0);
    expect(acc.pending).toBe(0);

    // Half speed: two frames of real time yield one simulation step.
    expect(acc.begin(FIXED, 0.5)).toBe(0);
    expect(acc.begin(FIXED, 0.5)).toBe(1);
  });

  it('clamps before scaling, so slow-motion cannot smuggle in extra steps', () => {
    const acc = new FixedStepAccumulator({ maxDelta: 0.25, maxSubSteps: 100 });
    // 10 s raw clamps to 0.25 s, then 2x scaling gives 0.5 s == 30 steps.
    expect(acc.begin(10, 2)).toBe(30);
  });

  it('is deterministic: identical input sequences give identical output', () => {
    const run = (): number[] => {
      const acc = new FixedStepAccumulator();
      const out: number[] = [];
      for (let i = 0; i < 240; i++) out.push(acc.begin(0.0137));
      return out;
    };
    expect(run()).toEqual(run());
  });

  it('reset() discards pending time and alpha', () => {
    const acc = new FixedStepAccumulator();
    acc.begin(FIXED * 0.75);
    expect(acc.pending).toBeGreaterThan(0);

    acc.reset();
    expect(acc.pending).toBe(0);
    expect(acc.alpha).toBe(0);
    expect(acc.begin(FIXED * 0.5)).toBe(0);
  });

  it('honours a non-60 Hz simulation rate', () => {
    const acc = new FixedStepAccumulator({ hz: 120 });
    expect(acc.fixedDelta).toBeCloseTo(1 / 120, 10);
    expect(acc.begin(1 / 60)).toBe(2);
  });

  it('rejects a non-positive rate', () => {
    expect(() => new FixedStepAccumulator({ hz: 0 })).toThrow(RangeError);
    expect(() => new FixedStepAccumulator({ hz: -60 })).toThrow(RangeError);
  });
});

describe('Clock', () => {
  it('returns 0 on the first tick and deltas in seconds thereafter', () => {
    let now = 1000;
    const clock = new Clock(() => now);

    expect(clock.tick()).toBe(0);

    now = 1016;
    expect(clock.tick()).toBeCloseTo(0.016, 10);

    now = 1032;
    expect(clock.tick()).toBeCloseTo(0.016, 10);
  });

  it('reports 0 rather than a negative delta if the source goes backwards', () => {
    let now = 5000;
    const clock = new Clock(() => now);
    clock.tick();

    now = 4000;
    expect(clock.tick()).toBe(0);
  });

  it('reset() discards time spent paused', () => {
    let now = 0;
    const clock = new Clock(() => now);
    clock.tick();

    now = 60_000;
    clock.reset();
    now = 60_016;

    expect(clock.tick()).toBeCloseTo(0.016, 10);
  });
});

describe('createTimeState', () => {
  it('starts zeroed at unit time scale', () => {
    expect(createTimeState()).toEqual({ elapsed: 0, delta: 0, frame: 0, scale: 1 });
  });
});
