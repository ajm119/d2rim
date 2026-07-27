/**
 * @module rpg/SkillTree
 *
 * A trimmed Barbarian skill tree: Bash, Double Swing, Howl and Axe Mastery,
 * with D2's prerequisite and level rules and a point pool fed by levelling.
 *
 * ### What a skill point actually does here
 *
 * The rule this module is built around is that **a point must change a number
 * the player can feel**. Every skill therefore resolves to a
 * {@link SkillEffect}, which is expressed in the same vocabulary as
 * `combat/DamageModel`'s `MoveModifiers` — `damageScale`, `attackRatingBonus`,
 * `criticalBonus`, `staggerScale`. A passive folds into the character sheet as
 * {@link Modifiers}; an active is applied to the swing that is happening.
 * Neither path invents its own arithmetic.
 *
 * ### The tree
 *
 * ```
 * tier 1   Bash ............... +damage, +attack rating       (clvl 1)
 *          Howl ............... staggers nearby enemies       (clvl 1)
 * tier 2   Axe Mastery ........ passive +AR, +deadly strike   (clvl 6)
 *          Double Swing ....... two strikes per swing         (clvl 6, needs Bash)
 * ```
 *
 * Double Swing requiring Bash is the whole point of having prerequisites at
 * all: it means the player's first point is a commitment rather than a free
 * choice, which is what makes a build a build.
 *
 * ### Diminishing returns
 *
 * Every per-point bonus below is *linear* and every skill caps at
 * {@link MAX_SKILL_POINTS}. D2's own synergy curves are deliberately not
 * reproduced: with a level cap of 8 the player will place at most seven points
 * in total, so a curve would be indistinguishable from a line and would only
 * make the tooltip lie.
 */

import type { CharacterStats } from './Stats';
import type { ModifierKey, Modifiers } from './Stats';

/* -------------------------------------------------------------------------- */
/* Effects                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * What investing in a skill does to a swing.
 *
 * Field-for-field compatible with the parts of `MoveModifiers` that matter, so
 * the combat bridge can hand it straight through without a translation table.
 */
export interface SkillEffect {
  /** Multiplier on rolled damage. 1 means "no change". */
  readonly damageScale: number;
  /** Flat attack rating added for this swing. */
  readonly attackRatingBonus: number;
  /** Added deadly-strike chance, as a fraction. */
  readonly criticalBonus: number;
  /** Multiplier on the stagger comparison. */
  readonly staggerScale: number;
  /** Additional strikes the move delivers beyond the first. */
  readonly extraHits: number;
  /** Effect radius in metres, for area skills. 0 for single-target. */
  readonly radius: number;
  /** Mana spent per activation. */
  readonly manaCost: number;
}

/** The identity effect: a skill with no points invested. */
export const NO_SKILL_EFFECT: SkillEffect = {
  damageScale: 1,
  attackRatingBonus: 0,
  criticalBonus: 0,
  staggerScale: 1,
  extraHits: 0,
  radius: 0,
  manaCost: 0,
};

export type SkillKind = 'active' | 'passive';

export interface SkillDefinition {
  readonly id: string;
  readonly name: string;
  /** One line, shown under the skill on the tree screen. */
  readonly description: string;
  readonly kind: SkillKind;
  /** Tree row, 1-based. Drives layout and reads as the tier. */
  readonly tier: number;
  /** Column within the tier, 1-based. */
  readonly column: number;
  /** Character level required before the first point may be spent. */
  readonly requiredLevel: number;
  /** Skills that need at least one point before this one may be invested in. */
  readonly prerequisites: readonly string[];
  readonly maxPoints: number;
  /** The swing modifiers at `points`. Called with `points >= 1`. */
  readonly effectAt: (points: number) => SkillEffect;
  /** Passive contributions to the character sheet at `points`. */
  readonly modifiersAt?: (points: number) => Modifiers;
  /** One line describing the numbers at `points`, for the tooltip. */
  readonly summaryAt: (points: number) => string;
}

/** Hard cap on points in any one skill. */
export const MAX_SKILL_POINTS = 20;

function effect(partial: Partial<SkillEffect>): SkillEffect {
  return { ...NO_SKILL_EFFECT, ...partial };
}

/**
 * The tree.
 *
 * The numbers are tuned against `PLAYER_OFFENSE` (9-17 physical, 150 attack
 * rating): one point of Bash is a visible ~25% damage step, which is the
 * smallest increment a player will actually notice mid-fight, and five points
 * roughly doubles a swing — a real but not absurd payoff for the whole of this
 * act's skill budget.
 */
export const BARBARIAN_SKILLS: readonly SkillDefinition[] = [
  {
    id: 'bash',
    name: 'Bash',
    description: 'A committed overhand blow. More damage, and far more likely to land.',
    kind: 'active',
    tier: 1,
    column: 1,
    requiredLevel: 1,
    prerequisites: [],
    maxPoints: MAX_SKILL_POINTS,
    effectAt: (points) =>
      effect({
        damageScale: 1 + 0.25 * points,
        attackRatingBonus: 30 * points,
        staggerScale: 1 + 0.2 * points,
        manaCost: 2,
      }),
    summaryAt: (points) =>
      points <= 0
        ? '+25% damage and +30 attack rating per point'
        : `+${Math.round(points * 25)}% damage, +${points * 30} attack rating, 2 mana`,
  },
  {
    id: 'howl',
    name: 'Howl',
    description: 'A war cry that rocks every skeleton within earshot back on its heels.',
    kind: 'active',
    tier: 1,
    column: 2,
    requiredLevel: 1,
    prerequisites: [],
    maxPoints: MAX_SKILL_POINTS,
    effectAt: (points) =>
      effect({
        // Howl is not a damage skill. Its numbers live in radius and stagger,
        // and its `damageScale` stays low on purpose so that a player who
        // leaves it selected is trading damage for control rather than getting
        // both.
        damageScale: 0.25,
        staggerScale: 4 + points,
        radius: 3.5 + 0.6 * points,
        manaCost: 4,
      }),
    summaryAt: (points) =>
      points <= 0
        ? 'Staggers enemies within 4.1 m; +0.6 m per point'
        : `Staggers enemies within ${(3.5 + 0.6 * points).toFixed(1)} m, 4 mana`,
  },
  {
    id: 'axe-mastery',
    name: 'Axe Mastery',
    description: 'Years with an axe. Always in effect, whatever skill is drawn.',
    kind: 'passive',
    tier: 2,
    column: 1,
    requiredLevel: 6,
    prerequisites: [],
    maxPoints: MAX_SKILL_POINTS,
    effectAt: () => NO_SKILL_EFFECT,
    modifiersAt: (points) => ({
      attackRating: 25 * points,
      criticalChance: 3 * points,
      enhancedDamage: 8 * points,
    }),
    summaryAt: (points) =>
      points <= 0
        ? '+25 attack rating, +8% damage and +3% deadly strike per point'
        : `+${25 * points} attack rating, +${8 * points}% damage, +${3 * points}% deadly strike`,
  },
  {
    id: 'double-swing',
    name: 'Double Swing',
    description: 'Two blows in the space of one. Requires Bash.',
    kind: 'active',
    tier: 2,
    column: 2,
    requiredLevel: 6,
    prerequisites: ['bash'],
    maxPoints: MAX_SKILL_POINTS,
    effectAt: (points) =>
      effect({
        damageScale: 0.85 + 0.1 * points,
        attackRatingBonus: 20 * points,
        extraHits: 1,
        manaCost: 3,
      }),
    summaryAt: (points) =>
      points <= 0
        ? 'Two strikes per swing at 95% damage each; +10% per point'
        : `Two strikes at ${Math.round((0.85 + 0.1 * points) * 100)}% damage, 3 mana`,
  },
];

const SKILLS_BY_ID = new Map(BARBARIAN_SKILLS.map((entry) => [entry.id, entry]));

export function findSkill(id: string): SkillDefinition | null {
  return SKILLS_BY_ID.get(id) ?? null;
}

/* -------------------------------------------------------------------------- */
/* The tree instance                                                           */
/* -------------------------------------------------------------------------- */

/** Why a point cannot be spent. `'ok'` means it can. */
export type InvestRefusal =
  | 'ok'
  | 'unknown-skill'
  | 'no-points'
  | 'level'
  | 'prerequisite'
  | 'maxed';

export interface InvestCheck {
  readonly allowed: boolean;
  readonly reason: InvestRefusal;
  /** Which prerequisite is missing, when `reason === 'prerequisite'`. */
  readonly missing?: string;
}

export interface SkillTreeSnapshot {
  readonly points: Readonly<Record<string, number>>;
  readonly active: string | null;
}

/**
 * Points invested, the active skill, and the rules that govern both.
 *
 * Holds a reference to {@link CharacterStats} rather than its own point pool:
 * the pool is granted by levelling and spent here, and duplicating it is how a
 * character ends up able to spend the same point twice.
 */
export class SkillTree {
  readonly #stats: CharacterStats;
  readonly #points = new Map<string, number>();
  #active: string | null = null;

  constructor(stats: CharacterStats) {
    this.#stats = stats;
  }

  /** Every skill definition, in tree order. */
  get skills(): readonly SkillDefinition[] {
    return BARBARIAN_SKILLS;
  }

  get available(): number {
    return this.#stats.skillPoints;
  }

  /** The skill whose effect applies to the player's next swing, if any. */
  get active(): string | null {
    return this.#active;
  }

  pointsIn(id: string): number {
    return this.#points.get(id) ?? 0;
  }

  get totalInvested(): number {
    let total = 0;
    for (const value of this.#points.values()) total += value;
    return total;
  }

  /**
   * Whether the next point in `id` may be spent, and why not.
   *
   * Checks are ordered by what the player most needs to be told: an unmet
   * character level is actionable ("come back at 6"), a missing prerequisite is
   * actionable ("put one in Bash"), and "no points" is the least informative of
   * the three, so it is checked last among the blocking conditions.
   */
  canInvest(id: string): InvestCheck {
    const definition = SKILLS_BY_ID.get(id);
    if (definition === undefined) return { allowed: false, reason: 'unknown-skill' };
    if (this.pointsIn(id) >= definition.maxPoints) return { allowed: false, reason: 'maxed' };
    if (this.#stats.level < definition.requiredLevel) return { allowed: false, reason: 'level' };
    for (const prerequisite of definition.prerequisites) {
      if (this.pointsIn(prerequisite) <= 0) {
        return { allowed: false, reason: 'prerequisite', missing: prerequisite };
      }
    }
    if (this.#stats.skillPoints <= 0) return { allowed: false, reason: 'no-points' };
    return { allowed: true, reason: 'ok' };
  }

  /**
   * Spend one point.
   *
   * The first point spent in an *active* skill also selects it, because a
   * player who has just invested in Bash and then swings without it having
   * changed anything concludes — correctly — that the skill did nothing.
   */
  invest(id: string): boolean {
    const check = this.canInvest(id);
    if (!check.allowed) return false;
    if (!this.#stats.consumeSkillPoints(1)) return false;
    const next = this.pointsIn(id) + 1;
    this.#points.set(id, next);
    if (next === 1 && SKILLS_BY_ID.get(id)?.kind === 'active' && this.#active === null) {
      this.#active = id;
    }
    return true;
  }

  /** Select the active skill. Only skills with a point in them may be chosen. */
  setActive(id: string | null): boolean {
    if (id === null) {
      this.#active = null;
      return true;
    }
    const definition = SKILLS_BY_ID.get(id);
    if (definition === undefined || definition.kind !== 'active') return false;
    if (this.pointsIn(id) <= 0) return false;
    this.#active = id;
    return true;
  }

  /** Active skills with at least one point, in tree order. For the hotbar. */
  usableActives(): SkillDefinition[] {
    return BARBARIAN_SKILLS.filter(
      (entry) => entry.kind === 'active' && this.pointsIn(entry.id) > 0,
    );
  }

  /** Cycle to the next usable active skill, wrapping. @returns the new active. */
  cycleActive(): string | null {
    const usable = this.usableActives();
    if (usable.length === 0) return null;
    const index = usable.findIndex((entry) => entry.id === this.#active);
    const next = usable[(index + 1) % usable.length];
    this.#active = next?.id ?? null;
    return this.#active;
  }

  /** The active skill's effect, or the identity effect when none is selected. */
  activeEffect(): SkillEffect {
    if (this.#active === null) return NO_SKILL_EFFECT;
    const definition = SKILLS_BY_ID.get(this.#active);
    const points = this.pointsIn(this.#active);
    if (definition === undefined || points <= 0) return NO_SKILL_EFFECT;
    return definition.effectAt(points);
  }

  /** The effect of any skill at its current investment. */
  effectOf(id: string): SkillEffect {
    const definition = SKILLS_BY_ID.get(id);
    const points = this.pointsIn(id);
    if (definition === undefined || points <= 0) return NO_SKILL_EFFECT;
    return definition.effectAt(points);
  }

  /**
   * Passive contributions to the character sheet.
   *
   * Summed across every passive with points in it, in the {@link Modifiers}
   * shape so gear and skills reach `deriveStats` through the same door.
   */
  modifiers(): Modifiers {
    const out: Partial<Record<ModifierKey, number>> = {};
    for (const definition of BARBARIAN_SKILLS) {
      const points = this.pointsIn(definition.id);
      if (points <= 0 || definition.modifiersAt === undefined) continue;
      const contribution = definition.modifiersAt(points);
      for (const [key, value] of Object.entries(contribution)) {
        if (value === undefined) continue;
        const stat = key as ModifierKey;
        out[stat] = (out[stat] ?? 0) + value;
      }
    }
    return out;
  }

  /** Wipe every investment. The points are *not* refunded — respec is not a thing. */
  clear(): void {
    this.#points.clear();
    this.#active = null;
  }

  toJSON(): SkillTreeSnapshot {
    const points: Record<string, number> = {};
    for (const [id, value] of this.#points) points[id] = value;
    return { points, active: this.#active };
  }

  /**
   * Restore from a snapshot.
   *
   * Unknown skill ids are dropped and values are clamped: a save written by an
   * older build with a skill that no longer exists must load, not throw.
   */
  load(snapshot: SkillTreeSnapshot): void {
    this.#points.clear();
    for (const [id, value] of Object.entries(snapshot.points)) {
      const definition = SKILLS_BY_ID.get(id);
      if (definition === undefined) continue;
      const clamped = Math.max(0, Math.min(definition.maxPoints, Math.round(value)));
      if (clamped > 0) this.#points.set(id, clamped);
    }
    this.#active =
      snapshot.active !== null && this.pointsIn(snapshot.active) > 0 ? snapshot.active : null;
  }
}
