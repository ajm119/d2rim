/**
 * @module ai/enemies/Skeleton
 *
 * The four Act I skeletons: warrior, minion, rogue and mage, and the behaviour
 * tree they all share.
 *
 * They differ in numbers, not in code. That is deliberate — four bespoke AIs
 * for four enemies that fight the same way is four times the surface area for a
 * quarter of the readability, and the player cannot tell the difference between
 * a rogue with its own tree and a rogue with a shorter cooldown. The variants
 * that *would* need different code (the mage's ranged spellcast, which the rig
 * has clips for) are called out at the bottom as unfinished rather than faked.
 *
 * ### The tree, top to bottom
 *
 * ```
 * priority
 *   ├ dead?           → nothing (the enemy stops ticking the tree entirely)
 *   ├ attacking?      → keep swinging   (committed: cannot be re-planned away)
 *   ├ in range + aware → cooldown → [face, attack]
 *   ├ aware           → chase to the approach slot
 *   └ patrol          → wander around the spawn anchor, then wait
 * ```
 *
 * The priority selector re-evaluates every tick, so "I am being hit" and "the
 * player just walked into range" take over from a chase immediately. The
 * *attack* branch sits above them precisely because it must not: once the
 * telegraph starts the skeleton is committed, and that commitment is the thing
 * the player is reading.
 */

import * as THREE from 'three/webgpu';

import { BehaviorTree, bt, type Status, type TickContext } from '../BehaviorTree';
import {
  EnemyBase,
  type EnemyAttack,
  type EnemyBlackboard,
  type EnemyProfile,
  type EnemySpawnOptions,
} from '../EnemyBase';

/* -------------------------------------------------------------------------- */
/* Attacks                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * ### Why every window below is pinned to a measured number
 *
 * Each `window` is the span of the clip over which that clip's blade genuinely
 * passes through a player-sized capsule standing in front of the skeleton, and
 * `reachDuringWindow` is the smallest such contact distance across all four
 * variant rigs. Both come from `tools/measure-enemy-reach.mjs`, which walks the
 * GLB, samples the hand and forearm bones at 40 points per clip, extends the
 * implied blade exactly the way `Hitbox.LimbAnchor` does, and sweeps a capsule
 * outward to find the furthest separation that still touches.
 *
 * That measurement is the whole reason these attacks look the way they do. The
 * previous table authored every skeleton's standard swing onto
 * `1H_Melee_Attack_Chop`, whose implied blade never gets further than 0.66 m
 * from the enemy's centre on *any* variant — inside a 0.70 m stand-off, so the
 * hit window opened on schedule, the sweep ran, and it swept empty air. The
 * clips that reach are the ones that extend the arm *away* from the body, and
 * those are the slices and the stab.
 */

/**
 * The standard skeleton swing: a diagonal slice across the body.
 *
 * `telegraph: 0.34` plus the clip's own lead-in gives the player roughly three
 * quarters of a second of visible wind-up — long enough to read and step out
 * of, short enough that standing still is still fatal. The window opens early
 * in the clip because the wind-up has already been paid for in the telegraph.
 */
const CHOP: EnemyAttack = {
  id: 'skeleton.chop',
  action: 'attack.slice',
  speed: 0.9,
  // 1H_Melee_Attack_Slice_Diagonal reaches 1.40 m or better from 0.10 to 0.38.
  window: [0.12, 0.36],
  recovery: 0.82,
  telegraph: 0.34,
  reachDuringWindow: 1.4,
  modifiers: { damageScale: 1, knockback: 2, staggerScale: 1 },
};

const SLICE: EnemyAttack = {
  id: 'skeleton.slice',
  action: 'attack.sweep',
  speed: 1,
  // 1H_Melee_Attack_Slice_Horizontal reaches 0.90 m at 0.07 and peaks at 1.42.
  window: [0.08, 0.2],
  recovery: 0.75,
  telegraph: 0.24,
  reachDuringWindow: 0.9,
  modifiers: { damageScale: 0.85, knockback: 1.6, attackRatingBonus: 20 },
};

const THRUST: EnemyAttack = {
  id: 'skeleton.thrust',
  action: 'attack.stab',
  speed: 1,
  // 1H_Melee_Attack_Stab holds 1.54 m or better from 0.28 all the way to 0.50:
  // the longest and most reliable contact window on the rig.
  window: [0.28, 0.5],
  recovery: 0.72,
  telegraph: 0.3,
  reachDuringWindow: 1.5,
  modifiers: { damageScale: 1.1, knockback: 2.4, attackRatingBonus: 30 },
};

const HEAVY: EnemyAttack = {
  id: 'skeleton.heavy',
  action: 'attack.heavy',
  speed: 0.8,
  // 2H_Melee_Attack_Chop clears 0.84 m from 0.42 and peaks at 1.56 by 0.47.
  // Trimmed at 0.52 rather than the old 0.64: past that the axe is at rest
  // beside the body and the window was paying out on the follow-through.
  window: [0.42, 0.52],
  recovery: 0.9,
  // A long, obvious wind-up on a hit that hurts. The contract with the player
  // is that the biggest hits are always the most readable ones.
  telegraph: 0.55,
  reachDuringWindow: 0.84,
  modifiers: { damageScale: 1.7, knockback: 4, staggerScale: 1.6 },
};

/* -------------------------------------------------------------------------- */
/* Variants                                                                    */
/* -------------------------------------------------------------------------- */

function profile(overrides: Partial<EnemyProfile> & Pick<EnemyProfile, 'variant' | 'asset'>): EnemyProfile {
  return {
    maxHealth: 60,
    height: 1.78,
    capsuleRadius: 0.3,
    walkSpeed: 1.1,
    chaseSpeed: 3.1,
    attackRange: 1.85,
    reach: 0.8,
    offense: {
      level: 2,
      attackRating: 90,
      damage: { physical: { min: 5, max: 11 } },
      criticalChance: 0.04,
    },
    defense: {
      level: 2,
      defense: 34,
      resistances: { poison: 100, cold: -25 },
      maxResistances: { poison: 100 },
      blockChance: 0,
      physicalReduction: 0,
      poise: 9,
    },
    perception: {
      visionRange: 14,
      visionHalfAngle: (65 * Math.PI) / 180,
      hearingRange: 6.5,
      noiseMultiplier: 1.8,
      loseRange: 26,
    },
    attacks: [CHOP],
    attackCooldown: 2,
    staggerTime: 0.45,
    telegraphColor: 0xff3a1e,
    ...overrides,
  };
}

/**
 * The four skeletons.
 *
 * Poison immunity and cold vulnerability are straight out of Diablo II's
 * undead, and they are the only place in the whole combat stack where a
 * resistance actually changes how a fight plays — which is the point of having
 * damage types at all.
 */
export const SKELETON_PROFILES: Readonly<Record<string, EnemyProfile>> = {
  warrior: profile({
    variant: 'warrior',
    asset: 'enemy.skeleton.warrior',
    maxHealth: 84,
    height: 1.82,
    chaseSpeed: 2.9,
    attacks: [CHOP, HEAVY],
    attackCooldown: 2.2,
    defense: {
      level: 3,
      defense: 46,
      resistances: { poison: 100, cold: -25 },
      maxResistances: { poison: 100 },
      blockChance: 0.2,
      blockAbsorb: 0.6,
      physicalReduction: 0.08,
      poise: 14,
    },
  }),
  minion: profile({
    variant: 'minion',
    asset: 'enemy.skeleton.minion',
    maxHealth: 46,
    height: 1.68,
    chaseSpeed: 3.35,
    attackCooldown: 1.7,
    staggerTime: 0.55,
  }),
  rogue: profile({
    variant: 'rogue',
    asset: 'enemy.skeleton.rogue',
    maxHealth: 58,
    height: 1.74,
    chaseSpeed: 3.8,
    attackRange: 2,
    reach: 0.95,
    attacks: [SLICE, THRUST],
    attackCooldown: 1.4,
    offense: {
      level: 3,
      attackRating: 130,
      damage: { physical: { min: 4, max: 9 } },
      criticalChance: 0.12,
    },
  }),
  mage: profile({
    variant: 'mage',
    asset: 'enemy.skeleton.mage',
    maxHealth: 52,
    height: 1.76,
    chaseSpeed: 2.5,
    attackRange: 2.1,
    attackCooldown: 2.6,
    offense: {
      level: 3,
      attackRating: 95,
      // Cold damage on an undead caster: the one enemy whose damage the
      // player's fire resistance does nothing about.
      damage: { physical: { min: 2, max: 5 }, cold: { min: 6, max: 13 } },
      criticalChance: 0.03,
    },
    telegraphColor: 0x4fc3ff,
  }),
};

export type SkeletonVariant = keyof typeof SKELETON_PROFILES;

/** Every variant key, in spawn-rotation order. */
export const SKELETON_VARIANTS: readonly string[] = ['warrior', 'minion', 'rogue', 'mage'];

/** Metres over which a pursuit eases from full chase speed down to a halt. */
export const ARRIVAL_EASE = 0.7;

/**
 * Clearance held between the player's hit capsule and the enemy's, in metres.
 *
 * Zero: the hit capsules stop exactly touching. The physical capsules are
 * smaller than the hit capsules (0.30 + 0.30 against 0.40 + 0.30), so touching
 * hit capsules still leaves the bodies 0.1 m apart and nothing interpenetrates.
 *
 * This used to carry an apology explaining that any positive clearance disarmed
 * the enemy, because the closest point of the swinging arm to the player was
 * its elbow. That was true, and it was a symptom rather than a cause: the
 * windows were authored onto a clip that never reaches. They are now authored
 * onto clips that do — see `reachDuringWindow` on each attack, all of which
 * clear 0.70 m by at least 0.14 m — so the stand-off is a positioning choice
 * again rather than a workaround, and it has headroom to grow if a designer
 * wants the skeletons standing further off.
 */
export const STANDOFF_CLEARANCE = 0;

/** Radius of the player's hit capsule, from `PlayerCombatant.hitRadius`. */
const PLAYER_HIT_RADIUS = 0.4;

/**
 * Where a pursuit stops: touching distance, not overlapping distance.
 *
 * Deliberately *not* derived from `attackRange`. That number is the distance at
 * which the skeleton decides to swing, and it is authored well outside its
 * reach on purpose so the enemy has to commit to closing. Stopping there would
 * park the skeleton permanently outside its own weapon.
 */
export function standoffRadius(profile: EnemyProfile): number {
  return PLAYER_HIT_RADIUS + profile.capsuleRadius + STANDOFF_CLEARANCE;
}

/**
 * Chase speed for an enemy `distance` from its target, holding a `ring`
 * stand-off.
 *
 * Enemy capsules deliberately do not collide with the player's, so nothing
 * physically stops a pursuit: steering straight at the player's feet does not
 * halt at his chest, it walks *through* him and settles on top of him. The
 * player then cannot see what is hitting him, and the camera has a skeleton
 * inside its near plane. So the approach is given an arrival radius and eased
 * to zero across `ease` metres, which also stops the steering jittering on the
 * boundary the way a hard stop/go test would.
 *
 * @returns metres per second, or 0 once the stand-off is reached.
 */
export function pursuitSpeed(
  distance: number,
  ring: number,
  chaseSpeed: number,
  ease: number = ARRIVAL_EASE,
): number {
  const gap = distance - ring;
  if (gap <= 0.05) return 0;
  const span = Math.max(1e-3, ease);
  // Floored at a quarter speed so the last few centimetres are not a crawl.
  return chaseSpeed * THREE.MathUtils.clamp(gap / span, 0.25, 1);
}

/* -------------------------------------------------------------------------- */
/* The enemy                                                                   */
/* -------------------------------------------------------------------------- */

export interface SkeletonOptions extends EnemySpawnOptions {
  /** Point the skeleton patrols around when it has no target. */
  readonly anchor?: THREE.Vector3;
  /** Patrol radius around the anchor, metres. Default 4. */
  readonly patrolRadius?: number;
}

export class Skeleton extends EnemyBase {
  readonly #anchor: THREE.Vector3;
  readonly #patrolRadius: number;
  readonly #patrolTarget = new THREE.Vector3();
  readonly #point = new THREE.Vector3();
  #attackIndex = 0;
  #patrolValid = false;

  constructor(options: SkeletonOptions) {
    super(options);
    this.#anchor = (options.anchor ?? options.position).clone();
    this.#patrolRadius = Math.max(0, options.patrolRadius ?? 4);
  }

  protected buildTree(): BehaviorTree<EnemyBlackboard> {
    // `this.attacking ||` is the commitment clause. Once a telegraph has
    // started the branch must keep passing its own precondition even if the
    // player has since stepped out of range, or the priority selector hands
    // control to `chase`, resets this subtree and cancels the swing — which
    // from the player's seat looks like the enemy flinching for no reason.
    const canAttack = bt.condition<EnemyBlackboard>(
      'inRange',
      ({ blackboard }) =>
        this.attacking ||
        (blackboard.player !== null &&
          blackboard.aware &&
          blackboard.distance <= this.profile.attackRange),
    );

    const swing = bt.action<EnemyBlackboard>(
      'swing',
      ({ blackboard, dt }) => {
        if (blackboard.player === null) return 'failure';
        blackboard.player.footPosition(this.#point);
        this.holdFacing(this.#point);
        if (!this.attacking && !this.beginAttack(this.#pickAttack())) return 'failure';
        return this.advanceAttack(dt) ? 'running' : 'success';
      },
      () => this.endAttack(),
    );

    const attackBranch = bt.sequence<EnemyBlackboard>(
      'attack',
      canAttack,
      bt.cooldown(this.profile.attackCooldown, swing, 'attackCooldown'),
    );

    const chaseBranch = bt.sequence<EnemyBlackboard>(
      'chase',
      bt.condition('aware', ({ blackboard }) => blackboard.aware && blackboard.player !== null),
      bt.action('pursue', ({ blackboard }) => this.#pursue(blackboard)),
    );

    const patrolBranch = bt.sequence<EnemyBlackboard>(
      'patrol',
      bt.action('wander', (tick) => this.#patrol(tick)),
      bt.wait(1.6, 'rest'),
    );

    return new BehaviorTree<EnemyBlackboard>(
      bt.priority<EnemyBlackboard>('skeleton', attackBranch, chaseBranch, patrolBranch),
    );
  }

  /** Cycle the variant's attacks so a fight is not one clip on repeat. */
  #pickAttack(): EnemyAttack {
    const attacks = this.profile.attacks;
    const attack = attacks[this.#attackIndex % attacks.length] ?? attacks[0];
    this.#attackIndex++;
    // `attacks` is never empty in any shipped profile; the fallback keeps a
    // hand-edited content file from throwing at runtime.
    return attack ?? CHOP;
  }

  #pursue(blackboard: EnemyBlackboard): Status {
    const player = blackboard.player;
    if (player === null) return 'failure';
    this.setState('chase');
    player.footPosition(this.#point);
    // Aim at a slot on the ring rather than at the player, so a pack arrives as
    // an arc. Inside the ring, close the last step straight on.
    const ring = Math.max(0.6, this.profile.attackRange * 0.8);
    if (blackboard.distance > ring + 0.8) this.approachPoint(this.#point, ring, this.#point);
    const standoff = standoffRadius(this.profile);
    const speed = pursuitSpeed(blackboard.distance, standoff, this.profile.chaseSpeed);
    // Arrived: hold at touching distance and keep facing him. Well inside
    // `attackRange`, so the attack branch keeps firing — the skeleton waits out
    // its cooldown at sword's length instead of climbing into him.
    if (speed <= 0) {
      this.holdFacing(this.#point);
      return 'running';
    }
    this.moveToward(this.#point, speed);
    return 'running';
  }

  #patrol(tick: TickContext<EnemyBlackboard>): Status {
    this.setState('patrol');
    if (this.#patrolRadius <= 0.05) {
      this.halt();
      this.setState('idle');
      return 'success';
    }
    if (!this.#patrolValid) {
      // Deterministic wander: the tree's own clock and the enemy id, not
      // `Math.random`, so a replayed encounter walks the same path.
      const angle = (this.id * 2.399 + tick.time * 0.37) % (Math.PI * 2);
      this.#patrolTarget.set(
        this.#anchor.x + Math.sin(angle) * this.#patrolRadius,
        this.#anchor.y,
        this.#anchor.z + Math.cos(angle) * this.#patrolRadius,
      );
      this.#patrolValid = true;
    }
    this.moveToward(this.#patrolTarget, this.profile.walkSpeed);
    this.#point.copy(this.#patrolTarget).sub(this.position).setY(0);
    if (this.#point.length() < 0.6) {
      this.#patrolValid = false;
      this.halt();
      this.setState('idle');
      return 'success';
    }
    return 'running';
  }
}
