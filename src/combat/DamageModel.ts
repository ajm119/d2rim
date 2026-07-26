/**
 * @module combat/DamageModel
 *
 * The arithmetic of a swing landing, in the Diablo II idiom. Every function
 * here is pure and deterministic given its `rng` argument, which is the whole
 * point: this is the layer that decides whether the game is fair, and a system
 * you can only inspect by playing it is a system you cannot balance.
 *
 * ### The Diablo II bits, and where they are deliberately simplified
 *
 * - **Attack rating vs defence.** D2's chance-to-hit is
 *   `100 * AR/(AR+DR) * 2*alvl/(alvl+dlvl)`, clamped to `[5, 95]`. Both halves
 *   matter: the first is the gear check, the second is the level check, and
 *   dropping either one produces a game where either stats or levels are the
 *   only thing that counts. Kept verbatim — see {@link hitChance}.
 * - **Damage rolls.** A min-max range rolled uniformly, per damage type. D2
 *   rolls integers; so does this.
 * - **Critical strike** is D2's "deadly strike": it multiplies *physical*
 *   damage only. Elemental damage is unaffected, which is why a pure-elemental
 *   build does not want crit — a real build decision that falls straight out of
 *   the formula.
 * - **Resistances** are percentages with a cap (75 by default, D2's) and a
 *   floor of -100. Fire/cold/lightning/poison each carry their own.
 * - **Physical mitigation** applies percentage reduction first, then flat
 *   reduction, then a floor of 1: in D2 a landed hit always hurts a little.
 * - **Blocking** is a roll, not a state. A successful block removes
 *   `blockAbsorb` of the incoming damage (1 = negated outright) and cannot
 *   happen at all if the defender is not facing the blow, which is what makes
 *   a shield a positioning decision rather than a passive stat.
 * - **Poison** does not apply on impact. It is returned as a separate effect
 *   with a total and a duration, because a poison number that pops on the hit
 *   frame reads as physical damage and teaches the player nothing.
 *
 * Deliberately *not* modelled: stamina drain on block, life/mana steal, open
 * wounds, crushing blow, magic find. They are stat plumbing on top of these
 * formulas rather than changes to them.
 */

/* -------------------------------------------------------------------------- */
/* Damage types                                                               */
/* -------------------------------------------------------------------------- */

export type DamageType = 'physical' | 'fire' | 'cold' | 'lightning' | 'poison';

/** Iteration order for every `Record<DamageType, …>` this module produces. */
export const DAMAGE_TYPES: readonly DamageType[] = [
  'physical',
  'fire',
  'cold',
  'lightning',
  'poison',
];

/** An inclusive integer damage range. `max < min` is treated as `min`. */
export interface DamageRange {
  readonly min: number;
  readonly max: number;
}

/** Per-type damage ranges. Absent types contribute nothing. */
export type DamageSpread = Partial<Readonly<Record<DamageType, DamageRange>>>;

/** Per-type resistance percentages. Absent types are 0. */
export type Resistances = Partial<Readonly<Record<DamageType, number>>>;

/** A fully populated per-type tally. */
export type DamageTally = Readonly<Record<DamageType, number>>;

/** The all-zero tally. Reused as the base of every result. */
export function zeroTally(): Record<DamageType, number> {
  return { physical: 0, fire: 0, cold: 0, lightning: 0, poison: 0 };
}

/* -------------------------------------------------------------------------- */
/* Combatant statistics                                                        */
/* -------------------------------------------------------------------------- */

export interface OffenseStats {
  /** Character level. Feeds the level half of the hit formula. */
  readonly level: number;
  /** Attack rating. Feeds the gear half of the hit formula. */
  readonly attackRating: number;
  /** Base weapon damage, before the move's multiplier. */
  readonly damage: DamageSpread;
  /** Chance in `[0, 1]` of a deadly strike. */
  readonly criticalChance: number;
  /** Physical multiplier on a deadly strike. Default 2. */
  readonly criticalMultiplier?: number;
  /** Skips the defender's block roll entirely. Reserved for finishers. */
  readonly unblockable?: boolean;
  /** Never misses. Used by scripted or environmental damage. */
  readonly alwaysHits?: boolean;
  /** Poison duration in seconds when the spread carries poison. Default 4. */
  readonly poisonDuration?: number;
}

export interface DefenseStats {
  readonly level: number;
  /** Defence rating. */
  readonly defense: number;
  readonly resistances?: Resistances;
  /** Per-type resistance caps. Defaults to 75 for every type. */
  readonly maxResistances?: Resistances;
  /** Base block chance in `[0, 1]`, before facing and the cap. */
  readonly blockChance?: number;
  /** Fraction of a blocked hit removed. 1 negates it. Default 1. */
  readonly blockAbsorb?: number;
  /** Whether the shield is actually up right now. */
  readonly blocking?: boolean;
  /** Percentage physical damage reduction in `[0, 1]`. */
  readonly physicalReduction?: number;
  /** Flat physical damage removed after the percentage. */
  readonly flatReduction?: number;
  /** Total health, used to scale stagger, knockback and hit stop. */
  readonly maxHealth: number;
  /**
   * Resistance to being staggered, in the same units as damage. A hit staggers
   * when its applied damage exceeds it. Default 0 — everything staggers.
   */
  readonly poise?: number;
}

/** How a move modifies the attacker's base statistics. */
export interface MoveModifiers {
  /** Multiplier on every rolled damage type. Default 1. */
  readonly damageScale?: number;
  /** Added to the attacker's attack rating for this swing. Default 0. */
  readonly attackRatingBonus?: number;
  /** Added to the critical chance for this swing. Default 0. */
  readonly criticalBonus?: number;
  /** Knockback impulse in m/s at reference damage. Default 3. */
  readonly knockback?: number;
  /** Multiplier on the stagger comparison. Default 1. */
  readonly staggerScale?: number;
  /** Bypasses the block roll. */
  readonly unblockable?: boolean;
}

/* -------------------------------------------------------------------------- */
/* Results                                                                     */
/* -------------------------------------------------------------------------- */

export type AttackResult = 'miss' | 'blocked' | 'hit';

/** A damage-over-time effect handed to the defender to tick down itself. */
export interface PoisonEffect {
  /** Total damage delivered across the whole duration, post-resistance. */
  readonly total: number;
  readonly duration: number;
}

export interface AttackOutcome {
  readonly result: AttackResult;
  readonly critical: boolean;
  /** Rolled damage before any mitigation, per type. */
  readonly rolled: DamageTally;
  /** Damage actually delivered on impact, per type. Excludes poison. */
  readonly applied: DamageTally;
  /** Sum of {@link applied}. This is what comes off the health pool. */
  readonly total: number;
  readonly poison: PoisonEffect | null;
  /** The chance the hit roll was made against, for tuning readouts. */
  readonly hitChance: number;
  /** The chance the block roll was made against. 0 when not blocking. */
  readonly blockChance: number;
  /** Whether the hit should interrupt the defender's current action. */
  readonly staggered: boolean;
  /** Knockback speed in m/s along the swing direction. */
  readonly knockback: number;
}

/** A miss, shared so callers can compare against it cheaply. */
function missOutcome(hitChanceValue: number): AttackOutcome {
  return {
    result: 'miss',
    critical: false,
    rolled: zeroTally(),
    applied: zeroTally(),
    total: 0,
    poison: null,
    hitChance: hitChanceValue,
    blockChance: 0,
    staggered: false,
    knockback: 0,
  };
}

/* -------------------------------------------------------------------------- */
/* Random numbers                                                              */
/* -------------------------------------------------------------------------- */

/** A source of uniform values in `[0, 1)`. */
export type Rng = () => number;

/**
 * Mulberry32: a 32-bit PRNG that is small, fast, and — the reason it is here —
 * seedable. Combat that cannot be replayed cannot be debugged, so every roll in
 * this module takes its randomness as an argument and the game seeds one of
 * these per encounter rather than reaching for `Math.random`.
 */
export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* -------------------------------------------------------------------------- */
/* Primitive formulas                                                          */
/* -------------------------------------------------------------------------- */

export function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return value < min ? min : value > max ? max : value;
}

/** D2's floor and ceiling on chance to hit, as fractions. */
export const MIN_HIT_CHANCE = 0.05;
export const MAX_HIT_CHANCE = 0.95;

/**
 * Diablo II chance to hit.
 *
 * `100 * AR/(AR+DR) * 2*alvl/(alvl+dlvl)`, clamped to `[5%, 95%]`. Both factors
 * are clamped into sanity first so that a zeroed-out stat block degrades to a
 * coin flip rather than to `NaN`.
 */
export function hitChance(
  attackRating: number,
  defense: number,
  attackerLevel: number,
  defenderLevel: number,
): number {
  const ar = Math.max(0, attackRating);
  const dr = Math.max(0, defense);
  const alvl = Math.max(1, attackerLevel);
  const dlvl = Math.max(1, defenderLevel);

  const gear = ar + dr <= 0 ? 1 : ar / (ar + dr);
  const levels = (2 * alvl) / (alvl + dlvl);
  return clamp(gear * levels, MIN_HIT_CHANCE, MAX_HIT_CHANCE);
}

/** Default resistance cap, matching D2's 75%. */
export const DEFAULT_MAX_RESISTANCE = 75;

/** Clamp a resistance percentage to `[-100, cap]`. */
export function clampResistance(resist: number, cap: number = DEFAULT_MAX_RESISTANCE): number {
  const ceiling = Number.isFinite(cap) ? cap : DEFAULT_MAX_RESISTANCE;
  return clamp(resist, -100, ceiling);
}

/**
 * Apply a resistance percentage to an amount.
 *
 * Negative resistance amplifies, which is what makes a Conviction aura or a
 * "-fire resist" curse worth carrying.
 */
export function applyResistance(
  amount: number,
  resist: number,
  cap: number = DEFAULT_MAX_RESISTANCE,
): number {
  if (amount <= 0) return 0;
  return amount * (1 - clampResistance(resist, cap) / 100);
}

/**
 * Roll a min-max range.
 *
 * Rounded, because D2 damage is integral and a floating "7.3 damage" number
 * floating off an enemy's head is a tell that nothing underneath is a real
 * simulation. `max < min` collapses to `min` rather than throwing: a content
 * typo should cost damage, not crash the encounter.
 */
export function rollDamage(range: DamageRange, rng: Rng): number {
  const min = Math.max(0, range.min);
  const max = Math.max(min, range.max);
  if (max === min) return Math.round(min);
  return Math.round(min + rng() * (max - min));
}

/** D2's cap on block chance. */
export const MAX_BLOCK_CHANCE = 0.75;

/**
 * Effective block chance for one incoming blow.
 *
 * `facing` is `cos(angle)` between the defender's forward vector and the
 * direction the blow arrives *from*. Blows from behind cannot be blocked at
 * all, and the chance tapers through the flanks rather than snapping off, so
 * that a player turning into an attack is rewarded continuously.
 */
export function blockChance(
  base: number,
  facing: number,
  options: { readonly minFacing?: number; readonly fullFacing?: number } = {},
): number {
  const minFacing = options.minFacing ?? 0.1;
  const fullFacing = options.fullFacing ?? 0.6;
  if (base <= 0) return 0;
  if (facing <= minFacing) return 0;
  const span = Math.max(1e-6, fullFacing - minFacing);
  const taper = clamp((facing - minFacing) / span, 0, 1);
  return clamp(base * taper, 0, MAX_BLOCK_CHANCE);
}

/**
 * Physical mitigation: percentage first, then flat, then a floor of 1.
 *
 * The floor is the interesting part. Without it, stacking flat reduction makes
 * a character literally immune to a whole class of attacker, and the player on
 * the receiving end has no feedback distinguishing "immune" from "the hitbox
 * missed".
 */
export function mitigatePhysical(
  amount: number,
  percentReduction = 0,
  flatReduction = 0,
): number {
  if (amount <= 0) return 0;
  const afterPercent = amount * (1 - clamp(percentReduction, 0, 0.95));
  return Math.max(1, afterPercent - Math.max(0, flatReduction));
}

/**
 * Knockback speed for a landed hit.
 *
 * Scaled by the fraction of the defender's health the blow removed, so a chip
 * hit nudges and a finisher throws. Bounded at both ends: zero knockback reads
 * as no impact at all, and unbounded knockback launches skeletons over the
 * horizon on a crit.
 */
export function knockbackSpeed(damage: number, maxHealth: number, base = 3): number {
  if (damage <= 0 || base <= 0) return 0;
  const fraction = maxHealth > 0 ? damage / (maxHealth * 0.18) : 1;
  return base * clamp(fraction, 0.3, 1.75);
}

/**
 * Whether a hit interrupts what the defender was doing.
 *
 * Poise is a flat damage threshold rather than a probability: an enemy the
 * player can reliably stagger-lock with light hits is a bad enemy, but an
 * enemy whose interrupt is a dice roll is an *unreadable* enemy, and unreadable
 * is worse.
 */
export function shouldStagger(appliedDamage: number, poise: number, scale = 1): boolean {
  return appliedDamage * Math.max(0, scale) > Math.max(0, poise);
}

/* -------------------------------------------------------------------------- */
/* The whole swing                                                            */
/* -------------------------------------------------------------------------- */

export interface ResolveOptions {
  readonly move?: MoveModifiers;
  /**
   * `cos(angle)` between the defender's forward vector and the direction the
   * blow arrives from. 1 is dead ahead. Defaults to 1, i.e. the defender is
   * assumed to be facing the attack unless the caller says otherwise.
   */
  readonly facing?: number;
}

/**
 * Resolve one attack against one defender.
 *
 * The order of operations is fixed and load-bearing:
 *   1. hit roll (attack rating vs defence),
 *   2. block roll (only if the defender is blocking and facing),
 *   3. damage rolls per type, scaled by the move,
 *   4. critical strike, physical only,
 *   5. resistances per type,
 *   6. physical mitigation,
 *   7. block absorption,
 *   8. poison split out into its own effect.
 *
 * Consuming exactly three `rng` draws before the damage rolls (hit, block,
 * crit) keeps a seeded replay stable when only the *damage* numbers change,
 * which is what makes balance passes reproducible.
 */
export function resolveAttack(
  offense: OffenseStats,
  defense: DefenseStats,
  rng: Rng,
  options: ResolveOptions = {},
): AttackOutcome {
  const move = options.move ?? {};
  const facing = options.facing ?? 1;

  const rating = Math.max(0, offense.attackRating + (move.attackRatingBonus ?? 0));
  const chance = offense.alwaysHits === true
    ? 1
    : hitChance(rating, defense.defense, offense.level, defense.level);

  if (rng() >= chance) return missOutcome(chance);

  const unblockable = offense.unblockable === true || move.unblockable === true;
  const blockBase = defense.blocking === true && !unblockable ? (defense.blockChance ?? 0) : 0;
  const block = blockChance(blockBase, facing);
  const blocked = block > 0 && rng() < block;

  const criticalChance = clamp(offense.criticalChance + (move.criticalBonus ?? 0), 0, 1);
  const critical = rng() < criticalChance;
  const criticalMultiplier = critical ? Math.max(1, offense.criticalMultiplier ?? 2) : 1;

  const scale = Math.max(0, move.damageScale ?? 1);
  const rolled = zeroTally();
  for (const type of DAMAGE_TYPES) {
    const range = offense.damage[type];
    if (range === undefined) continue;
    const base = rollDamage(range, rng) * scale;
    rolled[type] = Math.round(type === 'physical' ? base * criticalMultiplier : base);
  }

  const absorb = blocked ? clamp(defense.blockAbsorb ?? 1, 0, 1) : 0;
  const applied = zeroTally();
  const resistances = defense.resistances ?? {};
  const caps = defense.maxResistances ?? {};

  let poisonTotal = 0;
  for (const type of DAMAGE_TYPES) {
    const raw = rolled[type];
    if (raw <= 0) continue;
    let value = applyResistance(
      raw,
      resistances[type] ?? 0,
      caps[type] ?? DEFAULT_MAX_RESISTANCE,
    );
    if (type === 'physical') {
      value = mitigatePhysical(value, defense.physicalReduction ?? 0, defense.flatReduction ?? 0);
    }
    value *= 1 - absorb;
    if (type === 'poison') {
      poisonTotal = Math.round(value);
      continue;
    }
    applied[type] = Math.round(value);
  }

  const total = DAMAGE_TYPES.reduce((sum, type) => sum + applied[type], 0);
  const poison =
    poisonTotal > 0
      ? { total: poisonTotal, duration: Math.max(0.1, offense.poisonDuration ?? 4) }
      : null;

  return {
    result: blocked ? 'blocked' : 'hit',
    critical,
    rolled,
    applied,
    total,
    poison,
    hitChance: chance,
    blockChance: block,
    staggered:
      !blocked && shouldStagger(total, defense.poise ?? 0, move.staggerScale ?? 1),
    knockback: blocked
      ? 0
      : knockbackSpeed(total, defense.maxHealth, move.knockback ?? 3),
  };
}

/**
 * Damage per second for a poison effect, for the tick loop.
 *
 * Split out so the defender does not have to know that `total / duration` is
 * the contract; if poison ever gains a ramp this is the only place that
 * changes.
 */
export function poisonTickDamage(effect: PoisonEffect, dt: number): number {
  if (effect.duration <= 0) return effect.total;
  return (effect.total / effect.duration) * Math.max(0, dt);
}
