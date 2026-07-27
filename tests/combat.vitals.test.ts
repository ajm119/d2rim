import { describe, expect, it } from 'vitest';

import { DEFAULT_VITALS, ResourcePool, Vitals } from '../src/combat/Vitals';
import {
  HIT_STOP_MAX,
  HIT_STOP_MIN,
  decayTrauma,
  hitStopDuration,
  traumaForHit,
  traumaToShake,
  valueNoise,
} from '../src/combat/Feedback';

describe('ResourcePool', () => {
  it('starts full unless told otherwise', () => {
    expect(new ResourcePool({ max: 50 }).value).toBe(50);
    expect(new ResourcePool({ max: 50, value: 10 }).value).toBe(10);
  });

  it('clamps a starting value into range', () => {
    expect(new ResourcePool({ max: 50, value: 500 }).value).toBe(50);
    expect(new ResourcePool({ max: 50, value: -5 }).value).toBe(0);
  });

  it('reports a zero fraction for a zero maximum instead of NaN', () => {
    expect(new ResourcePool({ max: 0 }).fraction).toBe(0);
  });

  it('drains no more than it holds and reports what it removed', () => {
    const pool = new ResourcePool({ max: 10 });
    expect(pool.drain(4)).toBe(4);
    expect(pool.drain(100)).toBe(6);
    expect(pool.empty).toBe(true);
  });

  it('restores no more than the headroom', () => {
    const pool = new ResourcePool({ max: 10, value: 8 });
    expect(pool.restore(5)).toBe(2);
    expect(pool.full).toBe(true);
  });

  it('spends all-or-nothing', () => {
    const pool = new ResourcePool({ max: 10, value: 5 });
    expect(pool.spend(6)).toBe(false);
    expect(pool.value).toBe(5);
    expect(pool.spend(5)).toBe(true);
    expect(pool.value).toBe(0);
  });

  it('does not regenerate before the delay expires', () => {
    const pool = new ResourcePool({ max: 100, value: 0, regen: 50, regenDelay: 1, regenRamp: 0 });
    pool.drain(1);
    pool.update(0.5);
    expect(pool.value).toBe(0);
    pool.update(0.6);
    expect(pool.value).toBeGreaterThan(0);
  });

  it('restarts the delay every time it is spent', () => {
    const pool = new ResourcePool({ max: 100, value: 50, regen: 50, regenDelay: 1, regenRamp: 0 });
    for (let i = 0; i < 10; i++) {
      pool.spend(1);
      pool.update(0.5);
    }
    // Never idle for a full second, so it never recovered anything.
    expect(pool.value).toBe(40);
  });

  it('ramps regeneration in rather than stepping', () => {
    const gentle = new ResourcePool({ max: 100, value: 0, regen: 100, regenDelay: 0, regenRamp: 1 });
    const abrupt = new ResourcePool({ max: 100, value: 0, regen: 100, regenDelay: 0, regenRamp: 0 });
    // Both must have been touched: an untouched pool is already warmed up, by
    // design — nothing has interrupted it.
    gentle.interrupt();
    abrupt.interrupt();
    gentle.update(0.1);
    abrupt.update(0.1);
    expect(gentle.value).toBeLessThan(abrupt.value);
  });

  it('recovers the full rate once the ramp completes', () => {
    const pool = new ResourcePool({ max: 1000, value: 0, regen: 10, regenDelay: 0, regenRamp: 0.5 });
    pool.update(1);
    const before = pool.value;
    pool.update(1);
    expect(pool.value - before).toBeCloseTo(10, 6);
  });

  it('keeps the fraction when the maximum changes', () => {
    const pool = new ResourcePool({ max: 100, value: 50 });
    pool.setMax(200);
    expect(pool.value).toBe(100);
  });

  it('refills and clears the delay', () => {
    const pool = new ResourcePool({ max: 10, value: 0 });
    pool.refill();
    expect(pool.full).toBe(true);
    expect(pool.idleTime).toBe(Infinity);
  });

  it('ignores a non-positive delta', () => {
    const pool = new ResourcePool({ max: 10, value: 0, regen: 100, regenDelay: 0, regenRamp: 0 });
    pool.update(0);
    pool.update(-1);
    expect(pool.value).toBe(0);
  });
});

describe('Vitals', () => {
  it('starts alive and full', () => {
    const vitals = new Vitals(DEFAULT_VITALS);
    expect(vitals.alive).toBe(true);
    expect(vitals.health.full).toBe(true);
  });

  it('reports the kill exactly once', () => {
    const vitals = new Vitals({
      health: { max: 10 },
      mana: { max: 0 },
      stamina: { max: 0 },
    });
    expect(vitals.applyDamage(4).killed).toBe(false);
    const killing = vitals.applyDamage(50);
    expect(killing.killed).toBe(true);
    expect(killing.removed).toBe(6);
    expect(vitals.applyDamage(50)).toEqual({ removed: 0, killed: false });
  });

  it('stalls stamina recovery when damage lands', () => {
    const vitals = new Vitals({
      health: { max: 100 },
      mana: { max: 10 },
      stamina: { max: 100, value: 0, regen: 100, regenDelay: 1, regenRamp: 0 },
    });
    vitals.update(0.9);
    const before = vitals.stamina.value;
    expect(before).toBeGreaterThan(0);
    vitals.applyDamage(1);
    vitals.update(0.9);
    expect(vitals.stamina.value).toBe(before);
  });

  it('does not regenerate a corpse', () => {
    const vitals = new Vitals({
      health: { max: 100, regen: 100, regenDelay: 0, regenRamp: 0 },
      mana: { max: 10 },
      stamina: { max: 10 },
    });
    vitals.kill();
    vitals.update(5);
    expect(vitals.health.value).toBe(0);
    expect(vitals.alive).toBe(false);
  });

  it('tracks how long it has been dead', () => {
    const vitals = new Vitals(DEFAULT_VITALS);
    expect(vitals.timeDead).toBe(0);
    vitals.update(1);
    vitals.kill();
    vitals.update(2);
    expect(vitals.timeDead).toBeCloseTo(2, 6);
  });

  it('revives to full', () => {
    const vitals = new Vitals(DEFAULT_VITALS);
    vitals.kill();
    vitals.revive();
    expect(vitals.alive).toBe(true);
    expect(vitals.health.full).toBe(true);
    expect(vitals.stamina.full).toBe(true);
  });

  it('does not come back to life just because the pool was refilled', () => {
    const vitals = new Vitals(DEFAULT_VITALS);
    vitals.kill();
    vitals.health.refill();
    expect(vitals.alive).toBe(false);
  });
});

describe('feel maths', () => {
  it('keeps hit stop inside the readable band for ordinary hits', () => {
    expect(hitStopDuration(0, 1)).toBeCloseTo(HIT_STOP_MIN, 6);
    expect(hitStopDuration(1, 1)).toBeCloseTo(HIT_STOP_MAX, 6);
  });

  it('scales hit stop with damage', () => {
    expect(hitStopDuration(0.3, 1)).toBeGreaterThan(hitStopDuration(0.05, 1));
  });

  it('lets a finisher hang longer, but not indefinitely', () => {
    expect(hitStopDuration(1, 1.6)).toBeGreaterThan(hitStopDuration(1, 1));
    expect(hitStopDuration(1, 100)).toBeLessThanOrEqual(HIT_STOP_MAX * 1.8);
  });

  it('squares trauma so small shakes stay small', () => {
    expect(traumaToShake(0.5)).toBeCloseTo(0.25, 6);
    expect(traumaToShake(1)).toBe(1);
    expect(traumaToShake(-3)).toBe(0);
    expect(traumaToShake(3)).toBe(1);
  });

  it('decays trauma linearly and never below zero', () => {
    expect(decayTrauma(1, 0.5, 1)).toBeCloseTo(0.5, 6);
    expect(decayTrauma(0.1, 10, 1)).toBe(0);
  });

  it('adds more trauma for a bigger hit and more again for a crit', () => {
    expect(traumaForHit(0.4, false)).toBeGreaterThan(traumaForHit(0.05, false));
    expect(traumaForHit(0.2, true)).toBeGreaterThan(traumaForHit(0.2, false));
    expect(traumaForHit(5, true)).toBeLessThanOrEqual(1);
  });

  it('produces smooth, bounded, seed-stable noise', () => {
    for (let i = 0; i < 500; i++) {
      const value = valueNoise(7, i * 0.137);
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(valueNoise(7, 3.21)).toBe(valueNoise(7, 3.21));
    expect(valueNoise(7, 3.21)).not.toBe(valueNoise(8, 3.21));
  });

  it('is continuous: adjacent samples never jump the whole range', () => {
    let previous = valueNoise(3, 0);
    for (let t = 0.01; t < 20; t += 0.01) {
      const value = valueNoise(3, t);
      // A step of 0.01 across a unit lattice can move at most ~1.5% of the
      // 2-unit range; white noise would routinely move the whole range.
      expect(Math.abs(value - previous)).toBeLessThan(0.1);
      previous = value;
    }
  });

  it('is not constant', () => {
    const samples = new Set<number>();
    for (let i = 0; i < 50; i++) samples.add(Math.round(valueNoise(11, i * 0.7) * 1000));
    expect(samples.size).toBeGreaterThan(20);
  });
});
