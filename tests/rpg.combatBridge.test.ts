/**
 * The bridge between the character sheet and the combat system.
 *
 * `RpgSystem` cannot edit `CombatSystem`, so it delivers the difference between
 * the character's real offence and combat's baseline as a second damage packet.
 * The arithmetic of that difference is pure and lives here; the tests check the
 * two properties that decide whether the bridge is honest:
 *
 * 1. A character with no gear produces **no packet at all**. The damage model
 *    floors a landed physical blow at 1, so an all-zero packet would silently
 *    hand a bare-handed level 1 character a free point of damage per swing.
 * 2. A character with gear produces a packet that, added to the baseline,
 *    equals what the sheet says they should be dealing.
 */

import { describe, expect, it } from 'vitest';

import { PLAYER_OFFENSE } from '../src/combat/CombatSystem';
import { mulberry32, resolveAttack, type DefenseStats, type DamageSpread } from '../src/combat/DamageModel';
import { Character } from '../src/rpg/Character';
import { generateItem } from '../src/rpg/ItemGenerator';
import { BARBARIAN_ARMS, addToPhysical, differenceSpread, scaleSpread } from '../src/rpg/RpgSystem';

const SKELETON: DefenseStats = {
  level: 3,
  defense: 46,
  resistances: {},
  physicalReduction: 0,
  maxHealth: 84,
  poise: 14,
};

describe('scaleSpread', () => {
  it('returns the same object at scale 1, so nothing is allocated needlessly', () => {
    const spread: DamageSpread = { physical: { min: 3, max: 9 } };
    expect(scaleSpread(spread, 1)).toBe(spread);
  });

  it('scales every damage type it finds and invents none', () => {
    const scaled = scaleSpread({ physical: { min: 10, max: 20 }, fire: { min: 2, max: 4 } }, 1.5);
    expect(scaled.physical).toEqual({ min: 15, max: 30 });
    expect(scaled.fire).toEqual({ min: 3, max: 6 });
    expect(scaled.cold).toBeUndefined();
  });

  it('never produces a negative range', () => {
    const scaled = scaleSpread({ physical: { min: 10, max: 20 } }, 0);
    expect(scaled.physical).toEqual({ min: 0, max: 0 });
  });
});

describe('differenceSpread', () => {
  it('is null when the character is no better than the baseline', () => {
    expect(differenceSpread(PLAYER_OFFENSE.damage, PLAYER_OFFENSE.damage)).toBeNull();
    // And also when the character is strictly *worse* — an unarmed level 1
    // character must not be penalised into a negative packet.
    expect(differenceSpread({ physical: { min: 1, max: 3 } }, PLAYER_OFFENSE.damage)).toBeNull();
    expect(differenceSpread({}, PLAYER_OFFENSE.damage)).toBeNull();
  });

  it('is the honest difference when the character is better', () => {
    // Read the baseline rather than hard-coding it: `PLAYER_OFFENSE` is combat's
    // tuning knob and is expected to move, and a test that pins its value would
    // fail every time someone rebalanced the Barbarian's opening damage.
    const base = PLAYER_OFFENSE.damage.physical as { min: number; max: number };
    const mine = { physical: { min: base.min + 10, max: base.max + 20 } };
    const bonus = differenceSpread(mine, PLAYER_OFFENSE.damage);
    expect(bonus?.physical).toEqual({ min: 10, max: 20 });
  });

  it('carries a whole elemental type through, since the baseline has none', () => {
    const base = PLAYER_OFFENSE.damage.physical as { min: number; max: number };
    const bonus = differenceSpread(
      { physical: { ...base }, fire: { min: 4, max: 11 } },
      PLAYER_OFFENSE.damage,
    );
    expect(bonus?.fire).toEqual({ min: 4, max: 11 });
    expect(bonus?.physical).toBeUndefined();
  });

  it('never lets the minimum exceed the maximum', () => {
    const bonus = differenceSpread(
      { physical: { min: 30, max: 18 } },
      { physical: { min: 9, max: 17 } },
    );
    expect((bonus?.physical?.min ?? 0) <= (bonus?.physical?.max ?? 0)).toBe(true);
  });

  it('adds up: baseline plus packet is what the character should deal', () => {
    const base = PLAYER_OFFENSE.damage.physical as { min: number; max: number };
    const mine: DamageSpread = { physical: { min: base.min * 2, max: base.max * 2 } };
    const bonus = differenceSpread(mine, PLAYER_OFFENSE.damage);
    expect(bonus).not.toBeNull();

    const baseHit = resolveAttack(
      { ...PLAYER_OFFENSE, alwaysHits: true, criticalChance: 0 },
      SKELETON,
      mulberry32(3),
    );
    const bonusHit = resolveAttack(
      { ...PLAYER_OFFENSE, alwaysHits: true, criticalChance: 0, damage: bonus as DamageSpread },
      SKELETON,
      mulberry32(3),
    );
    const wholeHit = resolveAttack(
      { ...PLAYER_OFFENSE, alwaysHits: true, criticalChance: 0, damage: mine },
      SKELETON,
      mulberry32(3),
    );
    // The two packets between them cover the whole swing, to within the
    // damage model's own rounding.
    expect(Math.abs(baseHit.total + bonusHit.total - wholeHit.total)).toBeLessThanOrEqual(1);
  });
});

describe('gear changes a real combat number', () => {
  it('produces no augmentation for a bare-handed level 1 character', () => {
    const character = new Character();
    const bonus = differenceSpread(character.offense().damage, PLAYER_OFFENSE.damage);
    expect(bonus).toBeNull();
  });

  it('produces one the moment a decent weapon is equipped', () => {
    const character = new Character();
    const axe = generateItem({ seed: 4, itemLevel: 1, baseId: 'hand-axe', quality: 'normal' });
    character.acquire(axe);
    character.equip(axe);
    // A plain hand axe is *worse* than the combat system's tuned baseline, so
    // there is still nothing to add — which is correct, and is why the next
    // assertion uses a magic one.
    expect(character.derived.damage.max).toBeLessThan(PLAYER_OFFENSE.damage.physical?.max ?? 0);

    const better = generateItem({
      seed: 4,
      itemLevel: 8,
      baseId: 'broad-sword',
      quality: 'magic',
    });
    // A level 8 character, so the requirement is met and the equip happens.
    character.addExperience(32_175);
    character.spendStatPoint('strength', 35);
    character.acquire(better);
    expect(character.equip(better).equipped).toBe(true);

    const bonus = differenceSpread(character.offense().damage, PLAYER_OFFENSE.damage);
    expect(bonus).not.toBeNull();
    expect(bonus?.physical?.max ?? 0).toBeGreaterThan(0);
  });

  it('raises maximum life, which combat reads directly', () => {
    const character = new Character();
    const before = character.derived.maxLife;
    // Seed 8 is a "Hale Ring" — `+6 to Life`. Pinned rather than searched for,
    // and asserted, so that a change to the affix table turns this into a
    // failure rather than into a test that quietly stops testing anything.
    const ring = generateItem({ seed: 8, itemLevel: 1, baseId: 'ring', quality: 'magic' });
    expect(ring.mods.life ?? 0).toBeGreaterThan(0);
    character.acquire(ring);
    character.equip(ring);
    expect(character.derived.maxLife).toBe(before + (ring.mods.life ?? 0));
    expect(character.defense().maxHealth).toBe(character.derived.maxLife);
  });

  it('makes an equipped item show up in the offence handed to resolveAttack', () => {
    const character = new Character();
    character.addExperience(32_175);
    character.spendStatPoint('strength', 35);
    const before = character.offense(0.09);

    const sword = generateItem({ seed: 4, itemLevel: 8, baseId: 'broad-sword', quality: 'magic' });
    character.acquire(sword);
    expect(character.equip(sword).equipped).toBe(true);
    const after = character.offense(0.09);

    expect(after.damage.physical?.max ?? 0).toBeGreaterThan(before.damage.physical?.max ?? 0);

    const weak = resolveAttack({ ...before, alwaysHits: true }, SKELETON, mulberry32(9));
    const strong = resolveAttack({ ...after, alwaysHits: true }, SKELETON, mulberry32(9));
    expect(strong.total).toBeGreaterThan(weak.total);
  });
});

/* -------------------------------------------------------------------------- */
/* The class combat floor                                                      */
/* -------------------------------------------------------------------------- */

/**
 * The regression these guard is not "the arithmetic is wrong". It is the much
 * quieter one that shipped: `CombatSystem` adopted `rpg.offense` / `rpg.defense`
 * as the authority, which replaced every tuned combat constant with a level-1
 * character sheet, and **not one of 1370 tests noticed** — because every test
 * either exercised `deriveStats` (correct in isolation) or `CombatSystem` with
 * no RPG layer registered (also correct in isolation). The seam between them
 * was the only place the defect lived.
 *
 * So these assert the composed value, and they assert it against the constants
 * the encounter was actually balanced against, which is the property that has
 * to hold for the fight to feel the way it was tuned to feel.
 */
describe('BARBARIAN_ARMS, the class combat floor', () => {
  /** The offence a level-1 Barbarian swings with, exactly as `RpgSystem` builds it. */
  const composed = (character: Character) => {
    const base = character.offense(PLAYER_OFFENSE.criticalChance);
    return {
      attackRating: base.attackRating + BARBARIAN_ARMS.attackRating,
      damage: addToPhysical(base.damage, BARBARIAN_ARMS.damage).physical,
    };
  };

  const withStartingKit = (): Character => {
    const character = new Character();
    const axe = generateItem({ seed: 0x1a2b3c, itemLevel: 1, baseId: 'hand-axe', quality: 'normal' });
    character.acquire(axe);
    character.equip(axe);
    return character;
  };

  it('reproduces exactly the constants the encounter was tuned against', () => {
    const swing = composed(withStartingKit());
    expect(swing.attackRating).toBe(PLAYER_OFFENSE.attackRating);
    expect(swing.damage).toEqual(PLAYER_OFFENSE.damage.physical);
  });

  it('is a floor and not a replacement: better gear still beats it', () => {
    const kit = withStartingKit();
    const bare = composed(kit);

    const better = withStartingKit();
    const ring = generateItem({ seed: 0x515abc, itemLevel: 12, baseId: 'ring', quality: 'magic' });
    better.acquire(ring);
    better.equip(ring);

    // Whatever the ring rolled, the floor is unchanged and the sheet is added
    // to it — so the composed rating can only move the way the ring moved it.
    const geared = composed(better);
    expect(geared.attackRating - bare.attackRating).toBe(
      better.derived.attackRating - kit.derived.attackRating,
    );
  });

  it('leaves the D2 formulas in `deriveStats` alone', () => {
    // The floor lives in the provider precisely so that the sheet stays a
    // faithful D2 sheet. If this ever fails, the floor has leaked downwards.
    expect(withStartingKit().derived.attackRating).toBe(65);
  });

  it('adds physical damage without inventing an elemental type', () => {
    const spread = addToPhysical({ physical: { min: 3, max: 6 }, fire: { min: 1, max: 2 } }, {
      min: 7,
      max: 13,
    });
    expect(spread.physical).toEqual({ min: 10, max: 19 });
    expect(spread.fire).toEqual({ min: 1, max: 2 });
    expect(spread.cold).toBeUndefined();
  });

  it('adds physical damage to a character carrying no weapon at all', () => {
    const spread = addToPhysical({}, { min: 7, max: 13 });
    expect(spread.physical).toEqual({ min: 7, max: 13 });
  });
});
