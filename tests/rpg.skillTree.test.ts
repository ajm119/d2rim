/**
 * The skill tree: point spending, level gates, prerequisites, and the claim
 * that a point invested changes a real combat number.
 *
 * That last one is the point of the file. A tree that hands out points and
 * records them is easy; a tree whose points reach `resolveAttack` and make a
 * swing hit harder is the thing being tested, so the effect tests go through
 * the actual damage model rather than reading the effect struct back.
 */

import { describe, expect, it } from 'vitest';

import { mulberry32, resolveAttack, type DefenseStats } from '../src/combat/DamageModel';
import {
  BARBARIAN_SKILLS,
  MAX_SKILL_POINTS,
  NO_SKILL_EFFECT,
  SkillTree,
  findSkill,
} from '../src/rpg/SkillTree';
import { CharacterStats, deriveStats, offenseFrom } from '../src/rpg/Stats';

const NONE = { strength: 0, dexterity: 0, vitality: 0, energy: 0 };

const SKELETON: DefenseStats = {
  level: 3,
  defense: 46,
  resistances: { poison: 100, cold: -25 },
  blockChance: 0.2,
  blockAbsorb: 0.6,
  physicalReduction: 0.08,
  maxHealth: 84,
  poise: 14,
};

/** A character with `levels - 1` level-ups' worth of points. */
function statsAtLevel(level: number): CharacterStats {
  const stats = new CharacterStats();
  const thresholds = [0, 500, 1500, 3750, 7875, 14175, 22275, 32175];
  stats.addExperience(thresholds[level - 1] ?? 0);
  return stats;
}

describe('the tree definition', () => {
  it('has unique ids and resolvable prerequisites', () => {
    const ids = new Set(BARBARIAN_SKILLS.map((entry) => entry.id));
    expect(ids.size).toBe(BARBARIAN_SKILLS.length);
    for (const skill of BARBARIAN_SKILLS) {
      for (const prerequisite of skill.prerequisites) {
        expect(ids.has(prerequisite), `${skill.id} needs ${prerequisite}`).toBe(true);
      }
    }
  });

  it('contains the three skills the act calls for', () => {
    expect(findSkill('bash')).not.toBeNull();
    expect(findSkill('double-swing')).not.toBeNull();
    expect(findSkill('howl')).not.toBeNull();
  });

  it('gates Double Swing behind Bash', () => {
    expect(findSkill('double-swing')?.prerequisites).toEqual(['bash']);
  });

  it('summarises every skill at zero and at one point', () => {
    for (const skill of BARBARIAN_SKILLS) {
      expect(skill.summaryAt(0).length).toBeGreaterThan(0);
      expect(skill.summaryAt(1).length).toBeGreaterThan(0);
      expect(skill.summaryAt(1)).not.toBe(skill.summaryAt(0));
    }
  });
});

describe('spending points', () => {
  it('refuses without an available point', () => {
    const stats = new CharacterStats();
    const tree = new SkillTree(stats);
    expect(tree.canInvest('bash')).toEqual({ allowed: false, reason: 'no-points' });
    expect(tree.invest('bash')).toBe(false);
    expect(tree.pointsIn('bash')).toBe(0);
  });

  it('spends a point and consumes it from the pool', () => {
    const stats = statsAtLevel(2);
    const tree = new SkillTree(stats);
    expect(tree.available).toBe(1);
    expect(tree.invest('bash')).toBe(true);
    expect(tree.pointsIn('bash')).toBe(1);
    expect(stats.skillPoints).toBe(0);
    expect(tree.totalInvested).toBe(1);
  });

  it('refuses an unknown skill', () => {
    const tree = new SkillTree(statsAtLevel(4));
    expect(tree.canInvest('whirlwind').reason).toBe('unknown-skill');
    expect(tree.invest('whirlwind')).toBe(false);
  });

  it('enforces the character level gate before the point check', () => {
    const tree = new SkillTree(statsAtLevel(4));
    // Axe Mastery needs level 6; the character is 4 but does have points.
    expect(tree.available).toBeGreaterThan(0);
    expect(tree.canInvest('axe-mastery')).toEqual({ allowed: false, reason: 'level' });
    expect(tree.invest('axe-mastery')).toBe(false);
  });

  it('enforces prerequisites, and opens once they are met', () => {
    const tree = new SkillTree(statsAtLevel(6));
    const refusal = tree.canInvest('double-swing');
    expect(refusal.allowed).toBe(false);
    expect(refusal.reason).toBe('prerequisite');
    expect(refusal.missing).toBe('bash');

    expect(tree.invest('bash')).toBe(true);
    expect(tree.canInvest('double-swing').allowed).toBe(true);
    expect(tree.invest('double-swing')).toBe(true);
    expect(tree.pointsIn('double-swing')).toBe(1);
  });

  it('caps a skill at its maximum', () => {
    const stats = statsAtLevel(2);
    stats.grantSkillPoints(MAX_SKILL_POINTS + 5);
    const tree = new SkillTree(stats);
    for (let i = 0; i < MAX_SKILL_POINTS; i++) expect(tree.invest('bash')).toBe(true);
    expect(tree.canInvest('bash').reason).toBe('maxed');
    expect(tree.invest('bash')).toBe(false);
    expect(tree.pointsIn('bash')).toBe(MAX_SKILL_POINTS);
  });
});

describe('the active skill', () => {
  it('selects the first active skill invested in', () => {
    const tree = new SkillTree(statsAtLevel(2));
    expect(tree.active).toBeNull();
    tree.invest('bash');
    expect(tree.active).toBe('bash');
  });

  it('refuses to select a skill with no points in it', () => {
    const tree = new SkillTree(statsAtLevel(2));
    expect(tree.setActive('howl')).toBe(false);
    expect(tree.active).toBeNull();
  });

  it('never selects a passive', () => {
    const stats = statsAtLevel(6);
    stats.grantSkillPoints(4);
    const tree = new SkillTree(stats);
    tree.invest('axe-mastery');
    expect(tree.active).toBeNull();
    expect(tree.setActive('axe-mastery')).toBe(false);
  });

  it('cycles through usable actives and wraps', () => {
    const stats = statsAtLevel(3);
    stats.grantSkillPoints(2);
    const tree = new SkillTree(stats);
    tree.invest('bash');
    tree.invest('howl');
    expect(tree.usableActives().map((entry) => entry.id)).toEqual(['bash', 'howl']);
    expect(tree.cycleActive()).toBe('howl');
    expect(tree.cycleActive()).toBe('bash');
  });

  it('returns the identity effect when nothing is selected', () => {
    const tree = new SkillTree(statsAtLevel(2));
    expect(tree.activeEffect()).toEqual(NO_SKILL_EFFECT);
  });
});

describe('points change real combat numbers', () => {
  it('makes Bash hit measurably harder with each point', () => {
    const stats = statsAtLevel(8);
    const tree = new SkillTree(stats);
    tree.invest('bash');
    const one = tree.activeEffect();
    tree.invest('bash');
    const two = tree.activeEffect();

    expect(two.damageScale).toBeGreaterThan(one.damageScale);
    expect(two.attackRatingBonus).toBeGreaterThan(one.attackRatingBonus);

    // Through the real damage model: same seed, same rolls, more damage.
    const derived = deriveStats({ level: 8, spent: NONE, weaponDamage: { min: 9, max: 17 } });
    // `alwaysHits` isolates the damage roll from the hit roll: the point of
    // this assertion is the size of the blow, not whether it landed. The draw
    // order is unchanged, so both resolutions still consume the same randomness.
    const base = { ...offenseFrom(derived, 0.09), alwaysHits: true };
    const plain = resolveAttack(base, SKELETON, mulberry32(4), { move: { damageScale: 1 } });
    const bashed = resolveAttack(base, SKELETON, mulberry32(4), {
      move: { damageScale: two.damageScale },
    });
    expect(plain.result).toBe('hit');
    expect(bashed.total).toBeGreaterThan(plain.total);
  });

  it('makes Bash land more often through the hit formula', () => {
    const stats = statsAtLevel(8);
    const tree = new SkillTree(stats);
    tree.invest('bash');
    const effect = tree.activeEffect();

    const derived = deriveStats({ level: 8, spent: NONE, weaponDamage: { min: 9, max: 17 } });
    // A deliberately weak attack rating, so the clamp at 95% does not hide the
    // improvement the way it would at the Barbarian's tuned rating.
    const weak = { ...offenseFrom(derived, 0), attackRating: 40 };
    const plain = resolveAttack(weak, SKELETON, mulberry32(1), {});
    const bashed = resolveAttack(weak, SKELETON, mulberry32(1), {
      move: { attackRatingBonus: effect.attackRatingBonus },
    });
    expect(bashed.hitChance).toBeGreaterThan(plain.hitChance);
  });

  it('turns Axe Mastery into sheet modifiers that change the swing', () => {
    const stats = statsAtLevel(8);
    const tree = new SkillTree(stats);
    expect(tree.modifiers()).toEqual({});
    tree.invest('axe-mastery');

    const mods = tree.modifiers();
    expect(mods.attackRating).toBe(25);
    expect(mods.criticalChance).toBe(3);
    expect(mods.enhancedDamage).toBe(8);

    const without = deriveStats({ level: 8, spent: NONE, weaponDamage: { min: 10, max: 20 } });
    const with_ = deriveStats({
      level: 8,
      spent: NONE,
      weaponDamage: { min: 10, max: 20 },
      modifiers: mods,
    });
    expect(with_.attackRating).toBe(without.attackRating + 25);
    expect(with_.damage.max).toBeGreaterThan(without.damage.max);
    expect(with_.criticalChance).toBeGreaterThan(without.criticalChance);
  });

  it('gives Double Swing a second strike rather than a bigger one', () => {
    const stats = statsAtLevel(8);
    const tree = new SkillTree(stats);
    tree.invest('bash');
    tree.invest('double-swing');
    tree.setActive('double-swing');
    const effect = tree.activeEffect();
    expect(effect.extraHits).toBe(1);
    // Each individual strike is *weaker* than a plain swing at one point.
    expect(effect.damageScale).toBeLessThan(1);
    // But two of them beat one plain swing.
    expect(effect.damageScale * 2).toBeGreaterThan(1);
  });

  it('gives Howl a radius that grows and a damage scale that does not', () => {
    const stats = statsAtLevel(8);
    stats.grantSkillPoints(4);
    const tree = new SkillTree(stats);
    tree.invest('howl');
    const one = tree.effectOf('howl');
    tree.invest('howl');
    const two = tree.effectOf('howl');

    expect(two.radius).toBeGreaterThan(one.radius);
    expect(two.staggerScale).toBeGreaterThan(one.staggerScale);
    expect(two.damageScale).toBe(one.damageScale);
    expect(two.damageScale).toBeLessThan(1);
    expect(one.manaCost).toBeGreaterThan(0);
  });
});

describe('persistence', () => {
  it('round-trips points and the active skill', () => {
    const stats = statsAtLevel(8);
    const tree = new SkillTree(stats);
    tree.invest('bash');
    tree.invest('bash');
    tree.invest('howl');
    tree.setActive('howl');
    const snapshot = tree.toJSON();

    const restored = new SkillTree(statsAtLevel(8));
    restored.load(snapshot);
    expect(restored.toJSON()).toEqual(snapshot);
    expect(restored.pointsIn('bash')).toBe(2);
    expect(restored.active).toBe('howl');
  });

  it('drops unknown skills and clamps hostile values on load', () => {
    const tree = new SkillTree(statsAtLevel(8));
    tree.load({ points: { bash: 9999, 'not-a-skill': 5 }, active: 'not-a-skill' });
    expect(tree.pointsIn('bash')).toBe(MAX_SKILL_POINTS);
    expect(tree.pointsIn('not-a-skill')).toBe(0);
    expect(tree.active).toBeNull();
  });
});
