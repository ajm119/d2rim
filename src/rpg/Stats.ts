/**
 * @module rpg/Stats
 *
 * Diablo II character statistics: the four primary attributes, the derived
 * pools and combat ratings they feed, and the level/experience progression that
 * hands out points to spend on them.
 *
 * ### What is faithful and what is scaled
 *
 * The *shapes* are D2's, verbatim:
 *
 * - Four primaries — Strength, Dexterity, Vitality, Energy — raised one point
 *   at a time from a pool granted on level up ({@link STAT_POINTS_PER_LEVEL}).
 * - Life, mana and stamina are **derived**, never stored: a base, plus a
 *   per-level term, plus a per-point term on the attribute that owns the pool.
 *   Storing them is the classic bug — a `+20 life` ring equipped and unequipped
 *   twice leaves the character permanently richer.
 * - Attack rating is `5 * dexterity - 35`, defence is `dexterity / 4`, and
 *   strength adds `1%` enhanced physical damage per point, which is the
 *   Barbarian's own bonus.
 * - The experience thresholds for levels 1-8 are D2's actual numbers
 *   (see {@link XP_THRESHOLDS}).
 *
 * The one number that is *scaled* rather than copied is the life base. D2's
 * Barbarian starts on 55 life; this project's combat system was balanced around
 * `DEFAULT_VITALS.health.max === 120` before an RPG layer existed, and a stat
 * system that silently halves the player's health the moment it is wired in is
 * a stat system that broke the game. So {@link BARBARIAN_BASE} starts at exactly
 * the pools combat already uses, and only the *growth* is D2's. A level 1
 * character with this module installed is numerically identical to one without.
 *
 * ### How this reaches combat
 *
 * Nothing here talks to combat. It produces {@link OffenseStats} and
 * {@link DefenseStats} — the exact structures `combat/DamageModel` consumes —
 * so the arithmetic of a landed blow stays in one place and this module only
 * decides what numbers go into it. That is the difference between wiring into
 * the damage model and reimplementing it beside it.
 */

import type {
  DamageRange,
  DamageSpread,
  DefenseStats,
  OffenseStats,
  Resistances,
} from '../combat/DamageModel';

/* -------------------------------------------------------------------------- */
/* Primary attributes                                                          */
/* -------------------------------------------------------------------------- */

export type PrimaryStat = 'strength' | 'dexterity' | 'vitality' | 'energy';

/** Display order, and the order every `Record<PrimaryStat, …>` iterates in. */
export const PRIMARY_STATS: readonly PrimaryStat[] = [
  'strength',
  'dexterity',
  'vitality',
  'energy',
];

export type Attributes = Readonly<Record<PrimaryStat, number>>;

/* -------------------------------------------------------------------------- */
/* Modifiers                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Every statistic an affix, a skill or a piece of gear is allowed to touch.
 *
 * This union is the contract between {@link module:rpg/ItemGenerator}'s affix
 * table and the derivation below: an affix may only name a key that appears
 * here, and every key that appears here is read by {@link deriveStats}. Adding
 * a modifier that nothing reads is the failure mode this list exists to make
 * impossible — the compiler rejects the affix, rather than the affix rolling
 * for the rest of the project's life and doing nothing.
 */
export type ModifierKey =
  // primaries
  | 'strength'
  | 'dexterity'
  | 'vitality'
  | 'energy'
  // pools
  | 'life'
  | 'mana'
  | 'stamina'
  // ratings
  | 'attackRating'
  | 'defense'
  | 'enhancedDefense'
  // physical damage
  | 'minDamage'
  | 'maxDamage'
  | 'enhancedDamage'
  // elemental damage
  | 'fireMin'
  | 'fireMax'
  | 'coldMin'
  | 'coldMax'
  | 'lightningMin'
  | 'lightningMax'
  | 'poisonMin'
  | 'poisonMax'
  // resistances, in percentage points
  | 'resistFire'
  | 'resistCold'
  | 'resistLightning'
  | 'resistPoison'
  | 'resistAll'
  // defensive utility
  | 'blockChance'
  | 'damageReduction'
  | 'damageReducedPercent'
  // offensive utility
  | 'criticalChance';

/** All modifier keys, in a stable order. Drives tooltips and serialisation. */
export const MODIFIER_KEYS: readonly ModifierKey[] = [
  'strength',
  'dexterity',
  'vitality',
  'energy',
  'life',
  'mana',
  'stamina',
  'attackRating',
  'defense',
  'enhancedDefense',
  'minDamage',
  'maxDamage',
  'enhancedDamage',
  'fireMin',
  'fireMax',
  'coldMin',
  'coldMax',
  'lightningMin',
  'lightningMax',
  'poisonMin',
  'poisonMax',
  'resistFire',
  'resistCold',
  'resistLightning',
  'resistPoison',
  'resistAll',
  'blockChance',
  'damageReduction',
  'damageReducedPercent',
  'criticalChance',
];

/** A sparse bag of modifiers. Absent keys are zero. */
export type Modifiers = Partial<Readonly<Record<ModifierKey, number>>>;

/** A dense, mutable bag. What accumulation produces. */
export type ModifierTotals = Record<ModifierKey, number>;

/** The all-zero totals. */
export function zeroModifiers(): ModifierTotals {
  const out = {} as ModifierTotals;
  for (const key of MODIFIER_KEYS) out[key] = 0;
  return out;
}

/** Add `source` into `target` in place. @returns `target`. */
export function addModifiers(target: ModifierTotals, source: Modifiers): ModifierTotals {
  for (const key of MODIFIER_KEYS) {
    const value = source[key];
    if (value !== undefined) target[key] += value;
  }
  return target;
}

/** Sum any number of sparse modifier bags into fresh totals. */
export function sumModifiers(...sources: readonly Modifiers[]): ModifierTotals {
  const out = zeroModifiers();
  for (const source of sources) addModifiers(out, source);
  return out;
}

/**
 * Human-readable name and formatting for one modifier, in D2's own phrasing.
 *
 * `signed` controls whether a positive value prints a leading `+`; `suffix` is
 * what follows the number. Both exist so a tooltip can be rendered from data
 * rather than from a switch statement in the UI layer.
 */
export interface ModifierDisplay {
  readonly label: string;
  readonly suffix: string;
  readonly signed: boolean;
}

const MODIFIER_DISPLAY: Readonly<Record<ModifierKey, ModifierDisplay>> = {
  strength: { label: 'to Strength', suffix: '', signed: true },
  dexterity: { label: 'to Dexterity', suffix: '', signed: true },
  vitality: { label: 'to Vitality', suffix: '', signed: true },
  energy: { label: 'to Energy', suffix: '', signed: true },
  life: { label: 'to Life', suffix: '', signed: true },
  mana: { label: 'to Mana', suffix: '', signed: true },
  stamina: { label: 'to Stamina', suffix: '', signed: true },
  attackRating: { label: 'to Attack Rating', suffix: '', signed: true },
  defense: { label: 'Defense', suffix: '', signed: true },
  enhancedDefense: { label: 'Enhanced Defense', suffix: '%', signed: true },
  minDamage: { label: 'to Minimum Damage', suffix: '', signed: true },
  maxDamage: { label: 'to Maximum Damage', suffix: '', signed: true },
  enhancedDamage: { label: 'Enhanced Damage', suffix: '%', signed: true },
  fireMin: { label: 'Minimum Fire Damage', suffix: '', signed: true },
  fireMax: { label: 'Maximum Fire Damage', suffix: '', signed: true },
  coldMin: { label: 'Minimum Cold Damage', suffix: '', signed: true },
  coldMax: { label: 'Maximum Cold Damage', suffix: '', signed: true },
  lightningMin: { label: 'Minimum Lightning Damage', suffix: '', signed: true },
  lightningMax: { label: 'Maximum Lightning Damage', suffix: '', signed: true },
  poisonMin: { label: 'Minimum Poison Damage', suffix: '', signed: true },
  poisonMax: { label: 'Maximum Poison Damage', suffix: '', signed: true },
  resistFire: { label: 'Fire Resist', suffix: '%', signed: true },
  resistCold: { label: 'Cold Resist', suffix: '%', signed: true },
  resistLightning: { label: 'Lightning Resist', suffix: '%', signed: true },
  resistPoison: { label: 'Poison Resist', suffix: '%', signed: true },
  resistAll: { label: 'to All Resistances', suffix: '', signed: true },
  blockChance: { label: 'Increased Chance of Blocking', suffix: '%', signed: true },
  damageReduction: { label: 'Damage Reduced by', suffix: '', signed: false },
  damageReducedPercent: { label: 'Damage Reduced by', suffix: '%', signed: false },
  criticalChance: { label: 'Deadly Strike', suffix: '%', signed: true },
};

/** Format one modifier the way an item tooltip should show it. */
export function describeModifier(key: ModifierKey, value: number): string {
  const display = MODIFIER_DISPLAY[key];
  const rounded = Math.round(value * 100) / 100;
  const sign = display.signed && rounded > 0 ? '+' : '';
  return `${sign}${rounded}${display.suffix} ${display.label}`.trim();
}

/* -------------------------------------------------------------------------- */
/* Progression                                                                 */
/* -------------------------------------------------------------------------- */

/** Highest level this act supports. Act I to the Den is levels 1-8 territory. */
export const MAX_LEVEL = 8;

/**
 * Total experience required to *have reached* each level, index 0 = level 1.
 *
 * These are Diablo II's real thresholds. They matter more than they look: the
 * gap from level 1 to 2 is 500 and from 7 to 8 is 9 900, so the curve is what
 * makes the Den of Evil a level-4-ish experience rather than a level-8 one, and
 * inventing a smooth curve here would quietly rebalance the whole act.
 */
export const XP_THRESHOLDS: readonly number[] = [
  0, // 1
  500, // 2
  1500, // 3
  3750, // 4
  7875, // 5
  14175, // 6
  22275, // 7
  32175, // 8
];

/** Stat points granted per level, as in D2. */
export const STAT_POINTS_PER_LEVEL = 5;

/** Skill points granted per level, as in D2. */
export const SKILL_POINTS_PER_LEVEL = 1;

/** Total experience needed to be `level`. Clamped to the table. */
export function experienceForLevel(level: number): number {
  const index = Math.max(0, Math.min(XP_THRESHOLDS.length - 1, Math.round(level) - 1));
  return XP_THRESHOLDS[index] ?? 0;
}

/** The level a given total experience corresponds to, `1..MAX_LEVEL`. */
export function levelForExperience(experience: number): number {
  let level = 1;
  for (let i = 1; i < XP_THRESHOLDS.length; i++) {
    if (experience >= (XP_THRESHOLDS[i] ?? Infinity)) level = i + 1;
    else break;
  }
  return Math.min(MAX_LEVEL, level);
}

/**
 * Progress toward the next level as a fraction in `[0, 1]`.
 *
 * At the cap this returns 1 rather than dividing by a zero-width band, so the
 * XP bar reads "full" instead of `NaN`.
 */
export function levelProgress(experience: number): number {
  const level = levelForExperience(experience);
  if (level >= MAX_LEVEL) return 1;
  const floor = experienceForLevel(level);
  const ceiling = experienceForLevel(level + 1);
  const span = ceiling - floor;
  if (span <= 0) return 1;
  return Math.max(0, Math.min(1, (experience - floor) / span));
}

/**
 * Experience awarded for killing a monster of `monsterLevel` by a character of
 * `characterLevel`.
 *
 * D2 applies a penalty when the character out-levels the monster by more than
 * a few levels; without it, farming the Den at level 8 is as good as at level
 * 3 and the player has no reason to move on. The base rate is tuned so that
 * clearing the Den's full spawn count at level 1 lands the character around
 * level 4, which is where D2 puts them.
 */
export function experienceForKill(
  monsterLevel: number,
  characterLevel: number,
  baseExperience: number,
): number {
  const base = Math.max(0, baseExperience);
  const gap = Math.max(0, Math.round(characterLevel) - Math.round(monsterLevel));
  // A gentle taper rather than D2's cliff: five levels of overshoot still pays
  // something, which keeps a returning player's backtracking from feeling
  // actively punished.
  const penalty = gap <= 2 ? 1 : Math.max(0.1, 1 - (gap - 2) * 0.18);
  return Math.max(1, Math.round(base * penalty));
}

/* -------------------------------------------------------------------------- */
/* Class definition                                                            */
/* -------------------------------------------------------------------------- */

/** The per-class constants the derivations read. */
export interface ClassDefinition {
  readonly id: string;
  readonly displayName: string;
  readonly attributes: Attributes;
  /** Pool bases at level 1 with no points spent. */
  readonly life: number;
  readonly mana: number;
  readonly stamina: number;
  /** Pool growth per character level. */
  readonly lifePerLevel: number;
  readonly manaPerLevel: number;
  readonly staminaPerLevel: number;
  /** Pool growth per point spent in the owning attribute. */
  readonly lifePerVitality: number;
  readonly staminaPerVitality: number;
  readonly manaPerEnergy: number;
  /** Percentage enhanced physical damage per point of strength. */
  readonly damagePerStrength: number;
}

/**
 * The Barbarian.
 *
 * Attributes are D2's exactly (30/20/25/10). The three pool bases are this
 * project's `DEFAULT_VITALS`, for the reason in the module header; the growth
 * rates are scaled from D2's by the same factor the bases were, so the *shape*
 * of levelling — vitality being far and away the best life-per-point, strength
 * being the damage stat — is unchanged.
 */
export const BARBARIAN: ClassDefinition = {
  id: 'barbarian',
  displayName: 'Barbarian',
  attributes: { strength: 30, dexterity: 20, vitality: 25, energy: 10 },
  life: 120,
  mana: 40,
  stamina: 100,
  lifePerLevel: 6,
  manaPerLevel: 2,
  staminaPerLevel: 2,
  lifePerVitality: 5,
  staminaPerVitality: 2,
  manaPerEnergy: 2,
  damagePerStrength: 0.01,
};

/* -------------------------------------------------------------------------- */
/* Derivation                                                                  */
/* -------------------------------------------------------------------------- */

/** Everything the rest of the game reads off a character sheet. */
export interface DerivedStats {
  readonly level: number;
  /** Attributes after gear, i.e. what the character sheet's white numbers show. */
  readonly attributes: Attributes;
  /** Points spent by the player, before gear. Used by the "+" spend buttons. */
  readonly baseAttributes: Attributes;
  readonly maxLife: number;
  readonly maxMana: number;
  readonly maxStamina: number;
  readonly attackRating: number;
  readonly defense: number;
  /** Physical damage range including weapon, flat mods and enhanced damage. */
  readonly damage: DamageRange;
  /** Non-physical damage contributed by gear, per type. */
  readonly elemental: DamageSpread;
  readonly criticalChance: number;
  readonly resistances: Resistances;
  readonly blockChance: number;
  readonly damageReduction: number;
  readonly damageReducedPercent: number;
}

/** The bare-handed damage of a character with no weapon equipped. */
export const UNARMED_DAMAGE: DamageRange = { min: 1, max: 3 };

export interface DeriveOptions {
  readonly definition?: ClassDefinition;
  readonly level: number;
  /** Points the player has actually spent, excluding the class base. */
  readonly spent: Attributes;
  /** Everything gear and skills contribute. */
  readonly modifiers?: Modifiers;
  /** Base damage of the equipped weapon, before any modifier. */
  readonly weaponDamage?: DamageRange;
  /** Base defence of equipped armour, before `enhancedDefense`. */
  readonly armourDefense?: number;
}

function elementalRange(
  totals: ModifierTotals,
  minKey: ModifierKey,
  maxKey: ModifierKey,
): DamageRange | null {
  const min = Math.max(0, Math.round(totals[minKey]));
  const max = Math.max(min, Math.round(totals[maxKey]));
  if (max <= 0) return null;
  return { min, max };
}

/**
 * Turn a level, spent points and a bag of modifiers into the character sheet.
 *
 * Order of operations is deliberate and matters:
 *   1. attributes = class base + spent + gear,
 *   2. pools from the *post-gear* attributes, so `+10 vitality` on a ring is
 *      worth the same 50 life whether it came from a ring or from a level up,
 *   3. attack rating and defence from post-gear dexterity,
 *   4. physical damage: `(weapon + flat) * (1 + enhanced% + strength%)`, which
 *      is D2's ordering — enhanced damage multiplies the flat additions, which
 *      is why `+min/+max` jewellery is worth so much on a high-ED weapon.
 */
export function deriveStats(options: DeriveOptions): DerivedStats {
  const definition = options.definition ?? BARBARIAN;
  const level = Math.max(1, Math.min(MAX_LEVEL, Math.round(options.level)));
  const totals = sumModifiers(options.modifiers ?? {});

  const baseAttributes: Attributes = {
    strength: definition.attributes.strength + Math.max(0, options.spent.strength),
    dexterity: definition.attributes.dexterity + Math.max(0, options.spent.dexterity),
    vitality: definition.attributes.vitality + Math.max(0, options.spent.vitality),
    energy: definition.attributes.energy + Math.max(0, options.spent.energy),
  };

  const attributes: Attributes = {
    strength: baseAttributes.strength + totals.strength,
    dexterity: baseAttributes.dexterity + totals.dexterity,
    vitality: baseAttributes.vitality + totals.vitality,
    energy: baseAttributes.energy + totals.energy,
  };

  const vitalityPoints = attributes.vitality - definition.attributes.vitality;
  const energyPoints = attributes.energy - definition.attributes.energy;
  const levels = level - 1;

  const maxLife =
    definition.life +
    definition.lifePerLevel * levels +
    definition.lifePerVitality * vitalityPoints +
    totals.life;
  const maxMana =
    definition.mana +
    definition.manaPerLevel * levels +
    definition.manaPerEnergy * energyPoints +
    totals.mana;
  const maxStamina =
    definition.stamina +
    definition.staminaPerLevel * levels +
    definition.staminaPerVitality * vitalityPoints +
    totals.stamina;

  // D2: AR = dexterity * 5 - 35. A level 1 Barbarian therefore opens on 65.
  const attackRating = Math.max(0, attributes.dexterity * 5 - 35 + totals.attackRating);

  const armour = Math.max(0, options.armourDefense ?? 0);
  const defense = Math.max(
    0,
    Math.floor(attributes.dexterity / 4) +
      Math.round(armour * (1 + totals.enhancedDefense / 100)) +
      totals.defense,
  );

  const weapon = options.weaponDamage ?? UNARMED_DAMAGE;
  const strengthBonus =
    (attributes.strength - definition.attributes.strength) * definition.damagePerStrength;
  const damageScale = Math.max(0, 1 + totals.enhancedDamage / 100 + strengthBonus);
  const rawMin = Math.max(0, weapon.min + totals.minDamage);
  const rawMax = Math.max(rawMin, weapon.max + totals.maxDamage);
  const damage: DamageRange = {
    min: Math.max(1, Math.round(rawMin * damageScale)),
    max: Math.max(1, Math.round(rawMax * damageScale)),
  };

  const elemental: Record<string, DamageRange> = {};
  const fire = elementalRange(totals, 'fireMin', 'fireMax');
  if (fire !== null) elemental['fire'] = fire;
  const cold = elementalRange(totals, 'coldMin', 'coldMax');
  if (cold !== null) elemental['cold'] = cold;
  const lightning = elementalRange(totals, 'lightningMin', 'lightningMax');
  if (lightning !== null) elemental['lightning'] = lightning;
  const poison = elementalRange(totals, 'poisonMin', 'poisonMax');
  if (poison !== null) elemental['poison'] = poison;

  const all = totals.resistAll;
  return {
    level,
    attributes,
    baseAttributes,
    maxLife: Math.max(1, Math.round(maxLife)),
    maxMana: Math.max(0, Math.round(maxMana)),
    maxStamina: Math.max(1, Math.round(maxStamina)),
    attackRating: Math.round(attackRating),
    defense: Math.round(defense),
    damage,
    elemental: elemental as DamageSpread,
    criticalChance: Math.max(0, Math.min(1, totals.criticalChance / 100)),
    resistances: {
      fire: totals.resistFire + all,
      cold: totals.resistCold + all,
      lightning: totals.resistLightning + all,
      poison: totals.resistPoison + all,
    },
    blockChance: Math.max(0, totals.blockChance / 100),
    damageReduction: Math.max(0, totals.damageReduction),
    damageReducedPercent: Math.max(0, Math.min(95, totals.damageReducedPercent)) / 100,
  };
}

/* -------------------------------------------------------------------------- */
/* Bridging into the damage model                                              */
/* -------------------------------------------------------------------------- */

/**
 * The character's swing, in the exact structure `resolveAttack` consumes.
 *
 * `baseCritical` is the class's innate deadly-strike chance, which lives in
 * combat's own tuning rather than on the sheet; gear adds to it.
 */
export function offenseFrom(derived: DerivedStats, baseCritical = 0): OffenseStats {
  const damage: Record<string, DamageRange> = { physical: derived.damage };
  for (const [type, range] of Object.entries(derived.elemental)) {
    if (range !== undefined) damage[type] = range;
  }
  return {
    level: derived.level,
    attackRating: derived.attackRating,
    damage: damage as DamageSpread,
    criticalChance: Math.max(0, Math.min(1, baseCritical + derived.criticalChance)),
    criticalMultiplier: 2,
  };
}

/**
 * The character's guard, in the structure `resolveAttack` consumes.
 *
 * `baseBlock` is what the equipped shield or the class contributes before
 * `blockChance` modifiers; passing 0 for a shieldless character is correct and
 * makes `+% chance of blocking` on a ring do nothing, exactly as in D2.
 */
export function defenseFrom(
  derived: DerivedStats,
  options: { readonly baseBlock?: number; readonly poise?: number } = {},
): DefenseStats {
  return {
    level: derived.level,
    defense: derived.defense,
    resistances: derived.resistances,
    blockChance: Math.max(0, Math.min(0.75, (options.baseBlock ?? 0) + derived.blockChance)),
    blockAbsorb: 1,
    physicalReduction: derived.damageReducedPercent,
    flatReduction: derived.damageReduction,
    maxHealth: derived.maxLife,
    poise: options.poise ?? 0,
  };
}

/* -------------------------------------------------------------------------- */
/* The character sheet                                                         */
/* -------------------------------------------------------------------------- */

/** Serialisable state of a {@link CharacterStats}. */
export interface CharacterStatsSnapshot {
  readonly classId: string;
  readonly level: number;
  readonly experience: number;
  readonly spent: Attributes;
  readonly statPoints: number;
  readonly skillPoints: number;
}

/** Result of {@link CharacterStats.addExperience}. */
export interface LevelUpResult {
  readonly levelsGained: number;
  readonly level: number;
  readonly statPointsGained: number;
  readonly skillPointsGained: number;
}

const ZERO_SPENT: Attributes = { strength: 0, dexterity: 0, vitality: 0, energy: 0 };

/**
 * The mutable half of the sheet: level, experience, spent points and the two
 * unspent pools.
 *
 * It deliberately owns *no* gear. Derivation takes modifiers as an argument, so
 * this object can be tested — and serialised — without an inventory existing.
 */
export class CharacterStats {
  readonly definition: ClassDefinition;

  #level = 1;
  #experience = 0;
  #spent: Record<PrimaryStat, number> = { ...ZERO_SPENT };
  #statPoints = 0;
  #skillPoints = 0;

  constructor(definition: ClassDefinition = BARBARIAN) {
    this.definition = definition;
  }

  get level(): number {
    return this.#level;
  }

  get experience(): number {
    return this.#experience;
  }

  get statPoints(): number {
    return this.#statPoints;
  }

  get skillPoints(): number {
    return this.#skillPoints;
  }

  /** Points the player has spent, excluding the class base. */
  get spent(): Attributes {
    return { ...this.#spent };
  }

  /** `[0, 1]` toward the next level. 1 at the cap. */
  get progress(): number {
    return levelProgress(this.#experience);
  }

  /** Total experience still needed for the next level. 0 at the cap. */
  get experienceToNextLevel(): number {
    if (this.#level >= MAX_LEVEL) return 0;
    return Math.max(0, experienceForLevel(this.#level + 1) - this.#experience);
  }

  /**
   * Award experience and level up as many times as it warrants.
   *
   * Multiple levels in one award is a real case — a quest reward at low level
   * can cross two thresholds — and handling it in a loop rather than assuming
   * one is the difference between a working reward and a character stuck one
   * level below where the numbers say they should be.
   */
  addExperience(amount: number): LevelUpResult {
    if (!Number.isFinite(amount) || amount <= 0 || this.#level >= MAX_LEVEL) {
      return {
        levelsGained: 0,
        level: this.#level,
        statPointsGained: 0,
        skillPointsGained: 0,
      };
    }
    this.#experience += Math.round(amount);
    const cap = experienceForLevel(MAX_LEVEL);
    if (this.#experience > cap) this.#experience = cap;

    const target = levelForExperience(this.#experience);
    const gained = Math.max(0, target - this.#level);
    if (gained > 0) {
      this.#level = target;
      this.#statPoints += gained * STAT_POINTS_PER_LEVEL;
      this.#skillPoints += gained * SKILL_POINTS_PER_LEVEL;
    }
    return {
      levelsGained: gained,
      level: this.#level,
      statPointsGained: gained * STAT_POINTS_PER_LEVEL,
      skillPointsGained: gained * SKILL_POINTS_PER_LEVEL,
    };
  }

  /**
   * Spend `count` unspent points on `stat`.
   *
   * All-or-nothing: spending three points when two remain does nothing rather
   * than spending two, because a UI that silently spends fewer points than the
   * player asked for is a UI the player stops trusting.
   */
  spendStatPoint(stat: PrimaryStat, count = 1): boolean {
    const n = Math.round(count);
    if (n <= 0 || n > this.#statPoints) return false;
    this.#spent[stat] += n;
    this.#statPoints -= n;
    return true;
  }

  /** Award skill points directly. Quest rewards use this. */
  grantSkillPoints(count: number): void {
    const n = Math.round(count);
    if (n > 0) this.#skillPoints += n;
  }

  /** Award stat points directly. Reserved for shrines and quest rewards. */
  grantStatPoints(count: number): void {
    const n = Math.round(count);
    if (n > 0) this.#statPoints += n;
  }

  /**
   * Consume skill points. Called by {@link module:rpg/SkillTree}, which owns
   * the prerequisite rules; this only owns the pool.
   */
  consumeSkillPoints(count: number): boolean {
    const n = Math.round(count);
    if (n <= 0 || n > this.#skillPoints) return false;
    this.#skillPoints -= n;
    return true;
  }

  /** Derive the sheet against a bag of gear/skill modifiers. */
  derive(options: Omit<DeriveOptions, 'level' | 'spent' | 'definition'> = {}): DerivedStats {
    return deriveStats({
      definition: this.definition,
      level: this.#level,
      spent: this.spent,
      ...options,
    });
  }

  toJSON(): CharacterStatsSnapshot {
    return {
      classId: this.definition.id,
      level: this.#level,
      experience: this.#experience,
      spent: this.spent,
      statPoints: this.#statPoints,
      skillPoints: this.#skillPoints,
    };
  }

  /**
   * Restore from a snapshot.
   *
   * Values are clamped rather than trusted: a save file is user-editable data,
   * and a level of 4 000 crashes the derivation tables rather than making a
   * strong character.
   */
  load(snapshot: CharacterStatsSnapshot): void {
    this.#level = Math.max(1, Math.min(MAX_LEVEL, Math.round(snapshot.level)));
    this.#experience = Math.max(0, Math.round(snapshot.experience));
    this.#spent = {
      strength: Math.max(0, Math.round(snapshot.spent.strength)),
      dexterity: Math.max(0, Math.round(snapshot.spent.dexterity)),
      vitality: Math.max(0, Math.round(snapshot.spent.vitality)),
      energy: Math.max(0, Math.round(snapshot.spent.energy)),
    };
    this.#statPoints = Math.max(0, Math.round(snapshot.statPoints));
    this.#skillPoints = Math.max(0, Math.round(snapshot.skillPoints));
  }
}
