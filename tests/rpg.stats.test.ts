/**
 * Character statistics: the XP curve, point grants, the derivation, and the
 * bridge into the damage model.
 *
 * The derivation tests deliberately use *asymmetric* fixtures — different
 * numbers in every attribute, gear that touches only some of them — because a
 * derivation that confuses strength with dexterity agrees with a correct one on
 * any input where the two are equal, and a fixture built from round equal
 * numbers cannot tell them apart.
 */

import { describe, expect, it } from 'vitest';

import { hitChance } from '../src/combat/DamageModel';
import {
  BARBARIAN,
  CharacterStats,
  MAX_LEVEL,
  STAT_POINTS_PER_LEVEL,
  SKILL_POINTS_PER_LEVEL,
  XP_THRESHOLDS,
  defenseFrom,
  deriveStats,
  describeModifier,
  experienceForKill,
  experienceForLevel,
  levelForExperience,
  levelProgress,
  offenseFrom,
  sumModifiers,
  zeroModifiers,
  type Attributes,
} from '../src/rpg/Stats';

const NONE: Attributes = { strength: 0, dexterity: 0, vitality: 0, energy: 0 };

describe('experience curve', () => {
  it('uses Diablo II thresholds for levels 1-8', () => {
    expect(XP_THRESHOLDS).toEqual([0, 500, 1500, 3750, 7875, 14175, 22275, 32175]);
  });

  it('is a step function with the step exactly on the threshold', () => {
    // The off-by-one that matters: 499 must not be level 2 and 500 must be.
    expect(levelForExperience(499)).toBe(1);
    expect(levelForExperience(500)).toBe(2);
    expect(levelForExperience(1499)).toBe(2);
    expect(levelForExperience(1500)).toBe(3);
    expect(levelForExperience(32174)).toBe(7);
    expect(levelForExperience(32175)).toBe(8);
  });

  it('caps at the maximum level however much experience is thrown at it', () => {
    expect(levelForExperience(1e9)).toBe(MAX_LEVEL);
    expect(experienceForLevel(99)).toBe(32175);
  });

  it('reports progress as a fraction of the current band, not of the total', () => {
    // Half way from level 2 (500) to level 3 (1500) is 1000.
    expect(levelProgress(1000)).toBeCloseTo(0.5, 6);
    expect(levelProgress(500)).toBeCloseTo(0, 6);
    expect(levelProgress(1499)).toBeCloseTo(0.999, 3);
    expect(levelProgress(32175)).toBe(1);
  });

  it('tapers the kill award once the character out-levels the monster', () => {
    expect(experienceForKill(3, 3, 100)).toBe(100);
    expect(experienceForKill(3, 5, 100)).toBe(100); // two levels is still full
    expect(experienceForKill(3, 6, 100)).toBe(82);
    expect(experienceForKill(3, 8, 100)).toBe(46);
    // And never reaches zero, so backtracking is not actively punished.
    expect(experienceForKill(1, 8, 100)).toBeGreaterThan(0);
  });
});

describe('CharacterStats', () => {
  it('grants five stat points and one skill point per level', () => {
    const stats = new CharacterStats();
    const result = stats.addExperience(500);
    expect(result.levelsGained).toBe(1);
    expect(result.level).toBe(2);
    expect(stats.statPoints).toBe(STAT_POINTS_PER_LEVEL);
    expect(stats.skillPoints).toBe(SKILL_POINTS_PER_LEVEL);
  });

  it('handles crossing several thresholds in one award', () => {
    const stats = new CharacterStats();
    const result = stats.addExperience(3750);
    expect(result.levelsGained).toBe(3);
    expect(stats.level).toBe(4);
    expect(stats.statPoints).toBe(3 * STAT_POINTS_PER_LEVEL);
    expect(stats.skillPoints).toBe(3 * SKILL_POINTS_PER_LEVEL);
  });

  it('spends stat points all-or-nothing', () => {
    const stats = new CharacterStats();
    stats.addExperience(500);
    expect(stats.spendStatPoint('vitality', 6)).toBe(false);
    expect(stats.statPoints).toBe(5);
    expect(stats.spent.vitality).toBe(0);

    expect(stats.spendStatPoint('vitality', 5)).toBe(true);
    expect(stats.statPoints).toBe(0);
    expect(stats.spent.vitality).toBe(5);
  });

  it('refuses to award experience past the cap', () => {
    const stats = new CharacterStats();
    stats.addExperience(1e9);
    expect(stats.level).toBe(MAX_LEVEL);
    expect(stats.experience).toBe(32175);
    expect(stats.addExperience(500).levelsGained).toBe(0);
  });

  it('round-trips through a snapshot and clamps hostile values', () => {
    const stats = new CharacterStats();
    stats.addExperience(3750);
    stats.spendStatPoint('strength', 4);
    const snapshot = stats.toJSON();

    const restored = new CharacterStats();
    restored.load(snapshot);
    expect(restored.toJSON()).toEqual(snapshot);

    restored.load({ ...snapshot, level: 4000, experience: -5, statPoints: -12 });
    expect(restored.level).toBe(MAX_LEVEL);
    expect(restored.experience).toBe(0);
    expect(restored.statPoints).toBe(0);
  });
});

describe('deriveStats', () => {
  it('opens on exactly the pools the combat system was balanced around', () => {
    const derived = deriveStats({ level: 1, spent: NONE });
    expect(derived.maxLife).toBe(120);
    expect(derived.maxMana).toBe(40);
    expect(derived.maxStamina).toBe(100);
  });

  it("uses D2's attack rating and defence formulas", () => {
    const derived = deriveStats({ level: 1, spent: NONE });
    // AR = dexterity * 5 - 35, with a base dexterity of 20.
    expect(derived.attackRating).toBe(65);
    // Defence = floor(dexterity / 4).
    expect(derived.defense).toBe(5);
  });

  it('routes each attribute to its own pool and nowhere else', () => {
    // Asymmetric on purpose: an implementation that reads vitality where it
    // should read energy agrees with a correct one only when they are equal.
    const base = deriveStats({ level: 1, spent: NONE });
    const vit = deriveStats({ level: 1, spent: { ...NONE, vitality: 10 } });
    const enr = deriveStats({ level: 1, spent: { ...NONE, energy: 7 } });
    const dex = deriveStats({ level: 1, spent: { ...NONE, dexterity: 3 } });

    expect(vit.maxLife - base.maxLife).toBe(10 * BARBARIAN.lifePerVitality);
    expect(vit.maxMana).toBe(base.maxMana);
    expect(enr.maxMana - base.maxMana).toBe(7 * BARBARIAN.manaPerEnergy);
    expect(enr.maxLife).toBe(base.maxLife);
    expect(dex.attackRating - base.attackRating).toBe(15);
    expect(dex.maxLife).toBe(base.maxLife);
  });

  it('grows every pool with character level', () => {
    const one = deriveStats({ level: 1, spent: NONE });
    const eight = deriveStats({ level: 8, spent: NONE });
    expect(eight.maxLife - one.maxLife).toBe(7 * BARBARIAN.lifePerLevel);
    expect(eight.maxMana - one.maxMana).toBe(7 * BARBARIAN.manaPerLevel);
  });

  it('applies enhanced damage and the strength bonus multiplicatively over flat mods', () => {
    const weapon = { min: 10, max: 20 };
    const plain = deriveStats({ level: 1, spent: NONE, weaponDamage: weapon });
    expect(plain.damage).toEqual({ min: 10, max: 20 });

    const flat = deriveStats({
      level: 1,
      spent: NONE,
      weaponDamage: weapon,
      modifiers: { minDamage: 5, maxDamage: 5 },
    });
    expect(flat.damage).toEqual({ min: 15, max: 25 });

    // 50% enhanced damage multiplies the *flat-boosted* range, not the base.
    const enhanced = deriveStats({
      level: 1,
      spent: NONE,
      weaponDamage: weapon,
      modifiers: { minDamage: 5, maxDamage: 5, enhancedDamage: 50 },
    });
    expect(enhanced.damage).toEqual({ min: 23, max: 38 });

    // 20 points of strength over the base is +20% on the Barbarian.
    const strong = deriveStats({
      level: 1,
      spent: { ...NONE, strength: 20 },
      weaponDamage: weapon,
    });
    expect(strong.damage).toEqual({ min: 12, max: 24 });
  });

  it('falls back to unarmed damage with no weapon', () => {
    const derived = deriveStats({ level: 1, spent: NONE });
    expect(derived.damage).toEqual({ min: 1, max: 3 });
  });

  it('scales armour defence by enhanced defence before adding flat defence', () => {
    const derived = deriveStats({
      level: 1,
      spent: NONE,
      armourDefense: 40,
      modifiers: { enhancedDefense: 50, defense: 7 },
    });
    // floor(20/4) + round(40 * 1.5) + 7
    expect(derived.defense).toBe(5 + 60 + 7);
  });

  it('folds resistAll into every individual resistance', () => {
    const derived = deriveStats({
      level: 1,
      spent: NONE,
      modifiers: { resistAll: 10, resistFire: 15 },
    });
    expect(derived.resistances.fire).toBe(25);
    expect(derived.resistances.cold).toBe(10);
    expect(derived.resistances.lightning).toBe(10);
    expect(derived.resistances.poison).toBe(10);
  });

  it('turns gear attributes into pools exactly as spent points do', () => {
    const spentPoints = deriveStats({ level: 1, spent: { ...NONE, vitality: 6 } });
    const fromGear = deriveStats({ level: 1, spent: NONE, modifiers: { vitality: 6 } });
    expect(fromGear.maxLife).toBe(spentPoints.maxLife);
    // But the sheet still distinguishes them, so the UI can colour the bonus.
    expect(fromGear.baseAttributes.vitality).toBe(BARBARIAN.attributes.vitality);
    expect(fromGear.attributes.vitality).toBe(BARBARIAN.attributes.vitality + 6);
  });

  it('exposes elemental damage only when it is actually present', () => {
    expect(deriveStats({ level: 1, spent: NONE }).elemental).toEqual({});
    const fiery = deriveStats({ level: 1, spent: NONE, modifiers: { fireMin: 2, fireMax: 9 } });
    expect(fiery.elemental).toEqual({ fire: { min: 2, max: 9 } });
  });
});

describe('bridging into the damage model', () => {
  it('produces an OffenseStats the hit formula accepts', () => {
    const derived = deriveStats({
      level: 4,
      spent: { ...NONE, dexterity: 20 },
      weaponDamage: { min: 8, max: 14 },
      modifiers: { criticalChance: 12, fireMin: 3, fireMax: 7 },
    });
    const offense = offenseFrom(derived, 0.09);

    expect(offense.level).toBe(4);
    expect(offense.attackRating).toBe(derived.attackRating);
    expect(offense.damage.physical).toEqual(derived.damage);
    expect(offense.damage.fire).toEqual({ min: 3, max: 7 });
    expect(offense.criticalChance).toBeCloseTo(0.09 + 0.12, 6);

    // And the rating is a real improvement against a real defender.
    const weak = offenseFrom(deriveStats({ level: 4, spent: NONE }));
    expect(hitChance(offense.attackRating, 46, 4, 3)).toBeGreaterThan(
      hitChance(weak.attackRating, 46, 4, 3),
    );
  });

  it('produces a DefenseStats with the shield base folded in', () => {
    const derived = deriveStats({
      level: 3,
      spent: NONE,
      modifiers: { blockChance: 10, damageReduction: 4, damageReducedPercent: 8 },
    });
    const defense = defenseFrom(derived, { baseBlock: 0.3, poise: 3 });
    expect(defense.blockChance).toBeCloseTo(0.4, 6);
    expect(defense.flatReduction).toBe(4);
    expect(defense.physicalReduction).toBeCloseTo(0.08, 6);
    expect(defense.maxHealth).toBe(derived.maxLife);
    expect(defense.poise).toBe(3);
  });

  it('caps block chance at the game maximum', () => {
    const derived = deriveStats({ level: 3, spent: NONE, modifiers: { blockChance: 90 } });
    expect(defenseFrom(derived, { baseBlock: 0.35 }).blockChance).toBe(0.75);
  });
});

describe('modifier arithmetic', () => {
  it('sums sparse bags into dense totals', () => {
    const totals = sumModifiers({ strength: 3 }, { strength: 4, life: 10 }, {});
    expect(totals.strength).toBe(7);
    expect(totals.life).toBe(10);
    expect(totals.dexterity).toBe(0);
  });

  it('starts from all zeroes', () => {
    const zero = zeroModifiers();
    expect(Object.values(zero).every((value) => value === 0)).toBe(true);
  });

  it('formats modifiers the way an item tooltip should read', () => {
    expect(describeModifier('strength', 5)).toBe('+5 to Strength');
    expect(describeModifier('enhancedDamage', 42)).toBe('+42% Enhanced Damage');
    expect(describeModifier('damageReduction', 3)).toBe('3 Damage Reduced by');
  });
});
