import { describe, expect, it } from 'vitest';

import {
  BLEND_DIRECTIONS,
  DEFAULT_BLEND_PARAMS,
  blendContributions,
  blendedStride,
  cycleRate,
  directionWeights,
  sampleBlendSpace,
  tierWeights,
  type BlendContribution,
} from '../src/character/BlendSpace';

const sum = (values: Record<string, number>): number =>
  Object.values(values).reduce((total, value) => total + value, 0);

describe('directionWeights', () => {
  it('puts all the weight on one clip for an axis-aligned heading', () => {
    expect(directionWeights(0, 1).forward).toBeCloseTo(1);
    expect(directionWeights(0, -1).back).toBeCloseTo(1);
    expect(directionWeights(1, 0).right).toBeCloseTo(1);
    expect(directionWeights(-1, 0).left).toBeCloseTo(1);
  });

  it('splits a 45-degree heading evenly between exactly two clips', () => {
    const weights = directionWeights(1, 1);
    expect(weights.forward).toBeCloseTo(0.5);
    expect(weights.right).toBeCloseTo(0.5);
    expect(weights.back).toBe(0);
    expect(weights.left).toBe(0);
  });

  it('never activates opposing clips at once', () => {
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 32) {
      const weights = directionWeights(Math.sin(angle), Math.cos(angle));
      expect(weights.forward * weights.back).toBe(0);
      expect(weights.left * weights.right).toBe(0);
    }
  });

  it('sums to one at every heading and is scale invariant', () => {
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 24) {
      const slow = directionWeights(Math.sin(angle) * 0.1, Math.cos(angle) * 0.1);
      const fast = directionWeights(Math.sin(angle) * 9, Math.cos(angle) * 9);
      expect(sum(slow)).toBeCloseTo(1);
      for (const direction of BLEND_DIRECTIONS) {
        expect(slow[direction]).toBeCloseTo(fast[direction]);
      }
    }
  });

  it('returns nothing for a zero velocity rather than dividing by zero', () => {
    expect(sum(directionWeights(0, 0))).toBe(0);
  });
});

describe('tierWeights', () => {
  const params = DEFAULT_BLEND_PARAMS;

  it('is pure idle below the threshold', () => {
    expect(tierWeights(0, params)).toEqual({ idle: 1, walk: 0, run: 0 });
    expect(tierWeights(params.idleThreshold, params).idle).toBe(1);
  });

  it('is pure walk at the walk speed and pure run at the run speed', () => {
    const walk = tierWeights(params.walkSpeed, params);
    expect(walk.walk).toBeCloseTo(1);
    expect(walk.idle).toBeCloseTo(0);
    const run = tierWeights(params.runSpeed, params);
    expect(run.run).toBeCloseTo(1);
    expect(run.walk).toBeCloseTo(0);
  });

  it('never leaks idle into the run band', () => {
    for (let speed = params.walkSpeed; speed < 12; speed += 0.25) {
      expect(tierWeights(speed, params).idle).toBe(0);
    }
  });

  it('always sums to one and stays monotonic in run weight', () => {
    let previous = -1;
    for (let speed = 0; speed < 12; speed += 0.1) {
      const weights = tierWeights(speed, params);
      expect(sum(weights)).toBeCloseTo(1);
      expect(weights.run).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = weights.run;
    }
  });

  it('pins run at one above the run speed instead of over-weighting', () => {
    expect(tierWeights(20, params).run).toBe(1);
  });
});

describe('sampleBlendSpace and contributions', () => {
  it('produces a single contribution for a straight run', () => {
    const contributions = blendContributions(sampleBlendSpace(0, 6));
    expect(contributions).toHaveLength(1);
    expect(contributions[0]?.direction).toBe('forward');
    expect(contributions[0]?.tier).toBe('run');
    expect(contributions[0]?.weight).toBeCloseTo(1);
  });

  it('produces at most four contributions anywhere in the space', () => {
    for (let x = -8; x <= 8; x += 0.5) {
      for (let z = -8; z <= 8; z += 0.5) {
        const contributions = blendContributions(sampleBlendSpace(x, z));
        expect(contributions.length).toBeLessThanOrEqual(4);
      }
    }
  });

  it('conserves total weight together with the idle tier', () => {
    for (let speed = 0; speed < 9; speed += 0.3) {
      const sample = sampleBlendSpace(speed * 0.6, speed * 0.8);
      const total =
        sample.tier.idle +
        blendContributions(sample, 0).reduce((acc, entry) => acc + entry.weight, 0);
      expect(total).toBeCloseTo(1);
    }
  });

  it('is idle when standing still', () => {
    const sample = sampleBlendSpace(0, 0);
    expect(sample.tier.idle).toBe(1);
    expect(blendContributions(sample)).toHaveLength(0);
  });
});

describe('stride blending and cycle rate', () => {
  const strides: Record<string, number> = { 'walk.forward': 1.5, 'run.forward': 3.4 };
  const strideOf = (contribution: BlendContribution): number =>
    strides[`${contribution.tier}.${contribution.direction}`] ?? 0;

  it('returns the fallback when nothing is weighted', () => {
    expect(blendedStride([], strideOf, 1.25)).toBe(1.25);
  });

  it('interpolates between tiers by weight', () => {
    const half: BlendContribution[] = [
      { direction: 'forward', tier: 'walk', weight: 0.5 },
      { direction: 'forward', tier: 'run', weight: 0.5 },
    ];
    expect(blendedStride(half, strideOf, 1)).toBeCloseTo((1.5 + 3.4) / 2);
  });

  it('ignores clips with no measured stride rather than dragging the mean to zero', () => {
    const mixed: BlendContribution[] = [
      { direction: 'forward', tier: 'run', weight: 0.5 },
      { direction: 'left', tier: 'run', weight: 0.5 },
    ];
    expect(blendedStride(mixed, strideOf, 9)).toBeCloseTo(3.4);
  });

  it('matches cadence to speed so a planted foot does not slide', () => {
    // 4.2 m/s over a 3.4 m stride is 1.235 cycles per second.
    expect(cycleRate(4.2, 3.4)).toBeCloseTo(4.2 / 3.4);
    // Doubling the speed doubles the cadence, up to the clamp.
    expect(cycleRate(2, 4)).toBeCloseTo(0.5);
    expect(cycleRate(4, 4)).toBeCloseTo(1);
  });

  it('clamps the cadence at both ends instead of freezing or shredding', () => {
    expect(cycleRate(0, 3.4)).toBe(0.4);
    expect(cycleRate(100, 3.4)).toBe(1.75);
    expect(cycleRate(1, 0)).toBe(0.4);
  });
});
