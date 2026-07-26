/**
 * @module combat/Combatant
 *
 * The one interface everything that can be hit implements, and the registry the
 * hit-detection sweep queries.
 *
 * Keeping this separate from both the player and the AI is what lets a swing be
 * written once: `WeaponHitbox` sweeps against `Combatant`s, and neither it nor
 * the damage model knows or cares whether the thing it just connected with is a
 * skeleton, the hero, or a barrel that grew a health bar later.
 *
 * Every combatant is approximated as an **upright capsule** for hit purposes.
 * That is not the same shape as its physics collider on purpose: hit volumes
 * want to be forgiving (a swing that visually clips a shoulder should land) and
 * collision volumes want to be tight (a character that catches on scenery feels
 * broken). Tying them together makes one of the two feel wrong.
 */

import type * as THREE from 'three/webgpu';

import { serviceKey } from '../core/ServiceLocator';
import type { AttackOutcome, DefenseStats, OffenseStats } from './DamageModel';

export type Faction = 'player' | 'enemy';

/** Where a blow landed and how it was travelling, for feedback and knockback. */
export interface IncomingHit {
  /** Who swung. Null for environmental or scripted damage. */
  readonly source: Combatant | null;
  readonly offense: OffenseStats;
  /** Modifiers from the move that produced this hit. */
  readonly move: {
    readonly id: string;
    readonly damageScale?: number;
    readonly attackRatingBonus?: number;
    readonly criticalBonus?: number;
    readonly knockback?: number;
    readonly staggerScale?: number;
    readonly unblockable?: boolean;
  };
  /** World-space contact point. */
  readonly point: THREE.Vector3;
  /** Surface normal at the contact, pointing back toward the attacker. */
  readonly normal: THREE.Vector3;
  /** Unit direction the weapon was travelling. Drives knockback. */
  readonly direction: THREE.Vector3;
}

/**
 * Anything that can be swung at.
 *
 * `receiveHit` returns the resolved outcome — including a miss — so the caller
 * can drive feedback from one place instead of every implementer duplicating
 * the sparks-and-numbers code.
 */
export interface Combatant {
  readonly id: number;
  readonly faction: Faction;
  /** Diagnostic label. Shows up in the debug readout and in tests. */
  readonly label: string;
  /** The visible object, when there is one. */
  readonly object: THREE.Object3D | null;
  readonly alive: boolean;
  /** Live defensive statistics, sampled per hit. */
  readonly defense: DefenseStats;
  /** Capsule radius, metres. */
  readonly hitRadius: number;
  /** Capsule height from the feet, metres. */
  readonly hitHeight: number;
  /** Feet position in world space. Written into `out`. */
  footPosition(out: THREE.Vector3): THREE.Vector3;
  /** Unit forward vector in world space. Written into `out`. */
  facing(out: THREE.Vector3): THREE.Vector3;
  receiveHit(hit: IncomingHit): AttackOutcome;
}

export const CombatantsKey = serviceKey<CombatantRegistry>('combat.targets');

/**
 * Every live combatant, in registration order.
 *
 * An array rather than a spatial index: a Blood Moor encounter is a handful of
 * skeletons, and an O(n) scan over eight entries beats a broadphase that has to
 * be kept coherent with a set of characters that move every frame.
 */
export class CombatantRegistry {
  readonly #entries: Combatant[] = [];
  #nextId = 1;

  /** Mint an id. Combatants take theirs from here at construction. */
  nextId(): number {
    return this.#nextId++;
  }

  add(combatant: Combatant): Combatant {
    if (!this.#entries.includes(combatant)) this.#entries.push(combatant);
    return combatant;
  }

  remove(combatant: Combatant): boolean {
    const index = this.#entries.indexOf(combatant);
    if (index === -1) return false;
    this.#entries.splice(index, 1);
    return true;
  }

  get all(): readonly Combatant[] {
    return this.#entries;
  }

  /** Live combatants not on `faction`. The set a swing is allowed to hit. */
  hostileTo(faction: Faction): Combatant[] {
    return this.#entries.filter((entry) => entry.faction !== faction && entry.alive);
  }

  byFaction(faction: Faction): Combatant[] {
    return this.#entries.filter((entry) => entry.faction === faction);
  }

  /** The first live combatant on `faction`, or null. Used to find the player. */
  first(faction: Faction): Combatant | null {
    return this.#entries.find((entry) => entry.faction === faction && entry.alive) ?? null;
  }

  clear(): void {
    this.#entries.length = 0;
  }
}
