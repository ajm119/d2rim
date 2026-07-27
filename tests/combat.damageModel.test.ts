import { describe, expect, it } from 'vitest';

import {
  DAMAGE_TYPES,
  DEFAULT_MAX_RESISTANCE,
  MAX_BLOCK_CHANCE,
  MAX_HIT_CHANCE,
  MIN_HIT_CHANCE,
  applyResistance,
  blockChance,
  clampResistance,
  hitChance,
  knockbackSpeed,
  mitigatePhysical,
  mulberry32,
  poisonTickDamage,
  resolveAttack,
  rollDamage,
  shouldStagger,
  zeroTally,
  type DefenseStats,
  type OffenseStats,
  type Rng,
} from '../src/combat/DamageModel';

/** A scripted RNG: hands back the given values in order, then repeats the last. */
function scripted(...values: number[]): Rng {
  let index = 0;
  return () => values[Math.min(index++, values.length - 1)] ?? 0;
}

const OFFENSE: OffenseStats = {
  level: 5,
  attackRating: 200,
  damage: { physical: { min: 10, max: 20 } },
  criticalChance: 0.1,
  criticalMultiplier: 2,
};

const DEFENSE: DefenseStats = {
  level: 5,
  defense: 100,
  resistances: {},
  blockChance: 0.5,
  blocking: true,
  maxHealth: 100,
  poise: 0,
};

describe('hitChance', () => {
  it('implements the Diablo II formula for equal levels', () => {
    // AR/(AR+DEF) * 2*alvl/(alvl+dlvl) = 200/300 * 1 = 0.6667
    expect(hitChance(200, 100, 5, 5)).toBeCloseTo(2 / 3, 6);
  });

  it('applies the level term independently of the gear term', () => {
    // A level 1 attacker against a level 9 defender: 2*1/(1+9) = 0.2
    expect(hitChance(200, 200, 1, 9)).toBeCloseTo(0.5 * 0.2, 6);
  });

  it('clamps to the 5% floor', () => {
    expect(hitChance(1, 100000, 1, 99)).toBe(MIN_HIT_CHANCE);
  });

  it('clamps to the 95% ceiling', () => {
    expect(hitChance(100000, 1, 99, 1)).toBe(MAX_HIT_CHANCE);
  });

  it('degrades to the level term when neither side has a rating', () => {
    expect(hitChance(0, 0, 5, 5)).toBeCloseTo(0.95, 6);
  });

  it('treats negative inputs as zero rather than producing nonsense', () => {
    expect(hitChance(-50, -50, -3, -3)).toBeGreaterThanOrEqual(MIN_HIT_CHANCE);
    expect(hitChance(-50, -50, -3, -3)).toBeLessThanOrEqual(MAX_HIT_CHANCE);
  });

  it('is monotonic in attack rating', () => {
    let previous = 0;
    for (const ar of [0, 25, 50, 100, 200, 400, 800]) {
      const chance = hitChance(ar, 200, 10, 10);
      expect(chance).toBeGreaterThanOrEqual(previous);
      previous = chance;
    }
  });
});

describe('resistances', () => {
  it('caps at 75 by default', () => {
    expect(clampResistance(90)).toBe(DEFAULT_MAX_RESISTANCE);
  });

  it('honours a raised cap', () => {
    expect(clampResistance(90, 95)).toBe(90);
    expect(clampResistance(100, 95)).toBe(95);
  });

  it('floors at -100', () => {
    expect(clampResistance(-400)).toBe(-100);
  });

  it('halves damage at 50 resist', () => {
    expect(applyResistance(100, 50)).toBe(50);
  });

  it('doubles damage at -100 resist', () => {
    expect(applyResistance(100, -100)).toBe(200);
  });

  it('cannot reduce below the cap even at 100 resist', () => {
    expect(applyResistance(100, 100)).toBeCloseTo(25, 6);
  });

  it('immunity is expressible by raising the cap', () => {
    expect(applyResistance(100, 100, 100)).toBe(0);
  });

  it('never returns negative damage', () => {
    expect(applyResistance(-5, 0)).toBe(0);
  });
});

describe('rollDamage', () => {
  it('returns the minimum at rng 0', () => {
    expect(rollDamage({ min: 7, max: 19 }, () => 0)).toBe(7);
  });

  it('approaches the maximum at rng just below 1', () => {
    expect(rollDamage({ min: 7, max: 19 }, () => 0.9999999)).toBe(19);
  });

  it('is integral', () => {
    const rng = mulberry32(99);
    for (let i = 0; i < 200; i++) {
      expect(Number.isInteger(rollDamage({ min: 3, max: 12 }, rng))).toBe(true);
    }
  });

  it('stays inside the range for every draw', () => {
    const rng = mulberry32(4);
    for (let i = 0; i < 500; i++) {
      const value = rollDamage({ min: 3, max: 12 }, rng);
      expect(value).toBeGreaterThanOrEqual(3);
      expect(value).toBeLessThanOrEqual(12);
    }
  });

  it('collapses an inverted range to its minimum rather than throwing', () => {
    expect(rollDamage({ min: 10, max: 2 }, () => 0.5)).toBe(10);
  });

  it('never returns a negative value for a negative range', () => {
    expect(rollDamage({ min: -5, max: -1 }, () => 0.5)).toBe(0);
  });
});

describe('blockChance', () => {
  it('is zero when the blow comes from behind', () => {
    expect(blockChance(0.6, -1)).toBe(0);
  });

  it('reaches the base value dead ahead', () => {
    expect(blockChance(0.6, 1)).toBeCloseTo(0.6, 6);
  });

  it('tapers through the flanks rather than snapping', () => {
    const flank = blockChance(0.6, 0.35);
    expect(flank).toBeGreaterThan(0);
    expect(flank).toBeLessThan(0.6);
  });

  it('caps at 75%', () => {
    expect(blockChance(0.99, 1)).toBe(MAX_BLOCK_CHANCE);
  });

  it('is zero with no base chance', () => {
    expect(blockChance(0, 1)).toBe(0);
  });
});

describe('mitigatePhysical', () => {
  it('applies percentage before flat', () => {
    // 100 -> 50% -> 50 -> minus 10 -> 40
    expect(mitigatePhysical(100, 0.5, 10)).toBe(40);
  });

  it('never mitigates below 1', () => {
    expect(mitigatePhysical(5, 0.9, 1000)).toBe(1);
  });

  it('caps percentage reduction at 95%', () => {
    expect(mitigatePhysical(1000, 5, 0)).toBeCloseTo(50, 6);
  });

  it('passes zero through as zero', () => {
    expect(mitigatePhysical(0, 0.5, 5)).toBe(0);
  });
});

describe('knockback and stagger', () => {
  it('scales knockback with the fraction of health removed', () => {
    const light = knockbackSpeed(2, 100, 3);
    const heavy = knockbackSpeed(18, 100, 3);
    expect(heavy).toBeGreaterThan(light);
  });

  it('bounds knockback at both ends', () => {
    expect(knockbackSpeed(0.01, 100, 3)).toBeCloseTo(3 * 0.3, 6);
    expect(knockbackSpeed(9999, 100, 3)).toBeCloseTo(3 * 1.75, 6);
  });

  it('is zero for zero damage', () => {
    expect(knockbackSpeed(0, 100, 3)).toBe(0);
  });

  it('staggers only above poise', () => {
    expect(shouldStagger(10, 9)).toBe(true);
    expect(shouldStagger(9, 9)).toBe(false);
  });

  it('lets a move scale its way through poise', () => {
    expect(shouldStagger(10, 14)).toBe(false);
    expect(shouldStagger(10, 14, 1.8)).toBe(true);
  });
});

describe('resolveAttack', () => {
  it('misses when the hit roll fails, and reports the chance it used', () => {
    const outcome = resolveAttack(OFFENSE, DEFENSE, scripted(0.99));
    expect(outcome.result).toBe('miss');
    expect(outcome.total).toBe(0);
    expect(outcome.hitChance).toBeCloseTo(2 / 3, 6);
    expect(outcome.rolled).toEqual(zeroTally());
  });

  it('always hits when the attacker says so', () => {
    const outcome = resolveAttack({ ...OFFENSE, alwaysHits: true }, DEFENSE, scripted(0.999));
    expect(outcome.result).not.toBe('miss');
    expect(outcome.hitChance).toBe(1);
  });

  it('lands physical damage inside the weapon range', () => {
    // hit, no block, no crit, damage roll 0.5 -> 15
    const outcome = resolveAttack(OFFENSE, { ...DEFENSE, blockChance: 0 }, scripted(0, 1, 1, 0.5));
    expect(outcome.result).toBe('hit');
    expect(outcome.critical).toBe(false);
    expect(outcome.rolled.physical).toBe(15);
    // 5% reduction is not on this defender; mitigation floor is 1.
    expect(outcome.applied.physical).toBe(15);
    expect(outcome.total).toBe(15);
  });

  it('doubles only physical damage on a critical strike', () => {
    const offense: OffenseStats = {
      ...OFFENSE,
      damage: { physical: { min: 10, max: 10 }, fire: { min: 10, max: 10 } },
      criticalChance: 1,
    };
    const outcome = resolveAttack(offense, { ...DEFENSE, blockChance: 0 }, scripted(0, 1, 0, 0.5));
    expect(outcome.critical).toBe(true);
    expect(outcome.rolled.physical).toBe(20);
    expect(outcome.rolled.fire).toBe(10);
  });

  it('negates damage on a successful block', () => {
    // hit, block roll succeeds (0 < blockChance)
    const outcome = resolveAttack(OFFENSE, DEFENSE, scripted(0, 0, 1, 0.5));
    expect(outcome.result).toBe('blocked');
    expect(outcome.total).toBe(0);
    expect(outcome.knockback).toBe(0);
    expect(outcome.staggered).toBe(false);
  });

  it('partially absorbs with a lesser shield', () => {
    const outcome = resolveAttack(
      { ...OFFENSE, damage: { physical: { min: 20, max: 20 } } },
      { ...DEFENSE, blockAbsorb: 0.5 },
      scripted(0, 0, 1, 0.5),
    );
    expect(outcome.result).toBe('blocked');
    expect(outcome.applied.physical).toBe(10);
  });

  it('cannot be blocked from behind', () => {
    const outcome = resolveAttack(OFFENSE, DEFENSE, scripted(0, 0, 1, 0.5), { facing: -1 });
    expect(outcome.result).toBe('hit');
    expect(outcome.blockChance).toBe(0);
  });

  it('cannot be blocked when the defender is not blocking', () => {
    const outcome = resolveAttack(OFFENSE, { ...DEFENSE, blocking: false }, scripted(0, 0, 1, 0.5));
    expect(outcome.result).toBe('hit');
  });

  it('honours an unblockable move even against a raised shield', () => {
    const outcome = resolveAttack(OFFENSE, { ...DEFENSE, blocking: true }, scripted(0, 0, 1, 0.5), {
      move: { unblockable: true },
    });
    expect(outcome.result).toBe('hit');
    expect(outcome.blockChance).toBe(0);
  });

  it('blocks when the defender is blocking and facing', () => {
    const outcome = resolveAttack(OFFENSE, { ...DEFENSE, blocking: true }, scripted(0, 0, 1, 0.5));
    expect(outcome.result).toBe('blocked');
  });

  it('applies a move damage multiplier', () => {
    const base = resolveAttack(
      { ...OFFENSE, damage: { physical: { min: 10, max: 10 } } },
      { ...DEFENSE, blockChance: 0 },
      scripted(0, 1, 1, 0),
    );
    const scaled = resolveAttack(
      { ...OFFENSE, damage: { physical: { min: 10, max: 10 } } },
      { ...DEFENSE, blockChance: 0 },
      scripted(0, 1, 1, 0),
      { move: { damageScale: 1.85 } },
    );
    expect(base.total).toBe(10);
    expect(scaled.total).toBe(19);
  });

  it('routes each damage type through its own resistance', () => {
    const offense: OffenseStats = {
      ...OFFENSE,
      damage: {
        physical: { min: 10, max: 10 },
        fire: { min: 40, max: 40 },
        cold: { min: 40, max: 40 },
      },
      criticalChance: 0,
    };
    const defense: DefenseStats = {
      ...DEFENSE,
      blockChance: 0,
      resistances: { fire: 75, cold: -50 },
    };
    const outcome = resolveAttack(offense, defense, scripted(0, 1, 1, 0.5));
    expect(outcome.applied.fire).toBe(10);
    expect(outcome.applied.cold).toBe(60);
    expect(outcome.applied.physical).toBe(10);
    expect(outcome.total).toBe(80);
  });

  it('separates poison out of the impact total', () => {
    const offense: OffenseStats = {
      ...OFFENSE,
      damage: { physical: { min: 10, max: 10 }, poison: { min: 30, max: 30 } },
      criticalChance: 0,
      poisonDuration: 5,
    };
    const outcome = resolveAttack(offense, { ...DEFENSE, blockChance: 0 }, scripted(0, 1, 1, 0.5));
    expect(outcome.applied.poison).toBe(0);
    expect(outcome.total).toBe(10);
    expect(outcome.poison).toEqual({ total: 30, duration: 5 });
  });

  it('lets an immune defender ignore poison entirely', () => {
    const offense: OffenseStats = {
      ...OFFENSE,
      damage: { poison: { min: 30, max: 30 } },
      criticalChance: 0,
    };
    const outcome = resolveAttack(
      offense,
      {
        ...DEFENSE,
        blockChance: 0,
        resistances: { poison: 100 },
        maxResistances: { poison: 100 },
      },
      scripted(0, 1, 1, 0.5),
    );
    expect(outcome.poison).toBeNull();
  });

  it('produces a fully populated tally for every damage type', () => {
    const outcome = resolveAttack(OFFENSE, { ...DEFENSE, blockChance: 0 }, scripted(0, 1, 1, 0.5));
    for (const type of DAMAGE_TYPES) {
      expect(outcome.applied[type]).toBeTypeOf('number');
      expect(outcome.rolled[type]).toBeTypeOf('number');
    }
  });

  it('is reproducible from a seed', () => {
    const a = mulberry32(1234);
    const b = mulberry32(1234);
    for (let i = 0; i < 50; i++) {
      expect(resolveAttack(OFFENSE, DEFENSE, a)).toEqual(resolveAttack(OFFENSE, DEFENSE, b));
    }
  });

  it('never applies negative damage across a long random sweep', () => {
    const rng = mulberry32(777);
    for (let i = 0; i < 2000; i++) {
      const outcome = resolveAttack(OFFENSE, DEFENSE, rng);
      expect(outcome.total).toBeGreaterThanOrEqual(0);
      for (const type of DAMAGE_TYPES) expect(outcome.applied[type]).toBeGreaterThanOrEqual(0);
    }
  });

  it('converges on the analytic hit rate over many rolls', () => {
    const rng = mulberry32(2024);
    let hits = 0;
    const trials = 20000;
    for (let i = 0; i < trials; i++) {
      if (resolveAttack(OFFENSE, { ...DEFENSE, blockChance: 0 }, rng).result !== 'miss') hits++;
    }
    // Expected 2/3; three sigma on 20k trials is well inside 2 points.
    expect(hits / trials).toBeGreaterThan(0.65);
    expect(hits / trials).toBeLessThan(0.685);
  });
});

describe('poison ticking', () => {
  it('spreads the total evenly across the duration', () => {
    expect(poisonTickDamage({ total: 40, duration: 4 }, 1)).toBeCloseTo(10, 6);
  });

  it('sums to the total over the whole duration', () => {
    const effect = { total: 40, duration: 4 };
    let sum = 0;
    for (let i = 0; i < 240; i++) sum += poisonTickDamage(effect, 1 / 60);
    expect(sum).toBeCloseTo(40, 4);
  });

  it('delivers everything at once for a zero duration', () => {
    expect(poisonTickDamage({ total: 40, duration: 0 }, 0.1)).toBe(40);
  });
});

describe('mulberry32', () => {
  it('produces values in [0, 1)', () => {
    const rng = mulberry32(5);
    for (let i = 0; i < 1000; i++) {
      const value = rng();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('is deterministic per seed and different across seeds', () => {
    expect(mulberry32(1)()).toBe(mulberry32(1)());
    expect(mulberry32(1)()).not.toBe(mulberry32(2)());
  });
});
