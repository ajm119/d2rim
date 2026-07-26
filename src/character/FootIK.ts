/**
 * @module character/FootIK
 *
 * Two-bone IK that plants the Barbarian's feet on whatever he is actually
 * standing on.
 *
 * ### What it fixes
 *
 * The capsule is a capsule. It reports one ground height for a character whose
 * feet are 30 cm apart, so on the basin rim or on the wall's fallen rubble the
 * authored pose puts one boot through the mud and the other in the air. That is
 * the second-loudest tell of amateur character work after foot sliding, and it
 * is only visible in exactly the terrain this zone is made of.
 *
 * ### How
 *
 * Per foot, per frame, after the animation has posed the skeleton:
 *
 * 1. Ray down from the animated foot to find the real surface.
 * 2. Take the *lower* of the two required corrections and drop the hips by it,
 *    so the character crouches into a slope instead of stretching one leg.
 *    Without this step the downhill leg simply runs out of length.
 * 3. Solve each leg to its target with an analytic two-bone solve.
 *
 * Every correction is smoothed and clamped. An IK system that can move a foot
 * arbitrarily far will, the first time a ray hits the underside of a bridge.
 *
 * The solve is exposed as a pure function so the geometry can be tested without
 * a skeleton, a scene, or a physics world.
 */

import * as THREE from 'three/webgpu';

import type { GameContext, GameModule } from '../core/types';
import { findBone } from './BoneNames';
import { CollisionLayer, layerMask } from '../physics/Layers';
import { PhysicsWorldKey, type PhysicsWorld } from '../physics/PhysicsWorld';
import { PlayerKey, type PlayerController } from './PlayerController';

/* -------------------------------------------------------------------------- */
/* Solver                                                                     */
/* -------------------------------------------------------------------------- */

/** The two interior angles a two-bone chain needs to reach a distance. */
export interface TwoBoneAngles {
  /** Angle at the root between the upper bone and the root→target line. */
  readonly root: number;
  /** Interior angle at the joint between the two bones. π is straight. */
  readonly joint: number;
  /** Whether the target was out of reach and the chain was straightened. */
  readonly clamped: boolean;
}

/**
 * Solve a two-bone chain by the law of cosines.
 *
 * `upper` and `lower` are the bone lengths, `reach` the distance from the root
 * to the target. Out of range, the chain straightens and reports it rather than
 * producing `NaN` from an out-of-domain `acos` — which is how this class of bug
 * usually manifests: a limb that vanishes for one frame.
 */
export function solveTwoBoneAngles(
  upper: number,
  lower: number,
  reach: number,
): TwoBoneAngles {
  const total = upper + lower;
  const shortest = Math.abs(upper - lower);
  if (upper <= 1e-6 || lower <= 1e-6) return { root: 0, joint: Math.PI, clamped: true };
  if (reach >= total) return { root: 0, joint: Math.PI, clamped: true };
  if (reach <= shortest) {
    const clampedReach = Math.max(shortest + 1e-5, 1e-5);
    return {
      root: Math.acos(
        THREE.MathUtils.clamp(
          (upper * upper + clampedReach * clampedReach - lower * lower) /
            (2 * upper * clampedReach),
          -1,
          1,
        ),
      ),
      joint: Math.acos(
        THREE.MathUtils.clamp(
          (upper * upper + lower * lower - clampedReach * clampedReach) / (2 * upper * lower),
          -1,
          1,
        ),
      ),
      clamped: true,
    };
  }
  const root = Math.acos(
    THREE.MathUtils.clamp(
      (upper * upper + reach * reach - lower * lower) / (2 * upper * reach),
      -1,
      1,
    ),
  );
  const joint = Math.acos(
    THREE.MathUtils.clamp(
      (upper * upper + lower * lower - reach * reach) / (2 * upper * lower),
      -1,
      1,
    ),
  );
  return { root, joint, clamped: false };
}

/**
 * Rotate a bone by a world-space quaternion, preserving its parent transform.
 *
 * `bone.quaternion` is parent-relative, so a world rotation has to be conjugated
 * into parent space before it can be assigned. Doing it any other way works
 * perfectly for a bone whose parent is the identity and silently shears every
 * other bone in the rig.
 */
function rotateBoneWorld(bone: THREE.Object3D, rotation: THREE.Quaternion): void {
  const parentWorld = new THREE.Quaternion();
  bone.parent?.getWorldQuaternion(parentWorld);
  const world = new THREE.Quaternion();
  bone.getWorldQuaternion(world);
  world.premultiply(rotation);
  bone.quaternion.copy(parentWorld.invert().multiply(world));
  bone.updateMatrixWorld(true);
}

/** One leg the module drives. */
interface Leg {
  readonly upper: THREE.Object3D;
  readonly lower: THREE.Object3D;
  readonly foot: THREE.Object3D;
  /** Height of the foot bone above the sole in the bind pose, metres. */
  readonly soleOffset: number;
  /** Smoothed vertical correction currently applied, metres. */
  offset: number;
}

export interface FootIKOptions {
  /** Bone name prefixes for the legs. Default `upperleg`/`lowerleg`/`foot`. */
  readonly bones?: { upper: string; lower: string; foot: string };
  /** Largest correction a single foot may receive, metres. Default 0.42. */
  readonly maxLift?: number;
  /** Largest distance the hips may be dropped, metres. Default 0.3. */
  readonly maxHipDrop?: number;
  /** Smoothing rate for corrections. Higher is snappier. Default 12. */
  readonly damping?: number;
  /** Run at all. Default true. */
  readonly enabled?: boolean;
}

export class FootIK implements GameModule {
  readonly name = 'character.footIK';

  readonly #options: Required<FootIKOptions>;
  readonly #legs: Leg[] = [];

  #player: PlayerController | null = null;
  #physics: PhysicsWorld | null = null;
  #hips: THREE.Object3D | null = null;
  #hipDrop = 0;
  #weight = 0;

  constructor(options: FootIKOptions = {}) {
    this.#options = {
      bones: options.bones ?? { upper: 'upperleg', lower: 'lowerleg', foot: 'foot' },
      maxLift: options.maxLift ?? 0.42,
      maxHipDrop: options.maxHipDrop ?? 0.3,
      damping: options.damping ?? 12,
      enabled: options.enabled ?? true,
    };
  }

  /** Whether the solver found a rig it can drive. */
  get active(): boolean {
    return this.#options.enabled && this.#legs.length === 2;
  }

  init(ctx: GameContext): void {
    this.#player = ctx.services.tryGet<PlayerController>(PlayerKey) ?? null;
    this.#physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey) ?? null;
    const object = this.#player?.object ?? null;
    if (object === null || this.#physics === null) return;

    const names = this.#options.bones;
    this.#hips = findBone(object, 'hips');

    object.updateMatrixWorld(true);
    const rootY = object.getWorldPosition(new THREE.Vector3()).y;
    for (const side of ['l', 'r'] as const) {
      const upper = findBone(object, `${names.upper}.${side}`);
      const lower = findBone(object, `${names.lower}.${side}`);
      const foot = findBone(object, `${names.foot}.${side}`);
      if (upper === null || lower === null || foot === null) continue;
      const footY = foot.getWorldPosition(new THREE.Vector3()).y;
      this.#legs.push({ upper, lower, foot, soleOffset: Math.max(0.02, footY - rootY), offset: 0 });
    }
    if (this.#legs.length !== 2) {
      console.warn('[FootIK] could not find both legs; foot planting is disabled');
    } else {
      console.info(
        `[FootIK] driving ${this.#legs.length} legs, sole offsets ` +
          this.#legs.map((leg) => leg.soleOffset.toFixed(3)).join(' / '),
      );
    }
  }

  lateUpdate(_ctx: GameContext, dt: number): void {
    if (!this.active) return;
    const player = this.#player;
    const physics = this.#physics;
    const object = player?.object ?? null;
    if (player === null || physics === null || object === null || !physics.ready) return;

    // Off in the air, and eased rather than switched: a foot that snaps to the
    // ground on the landing frame looks like a glitch, not like a landing.
    const target = player.grounded ? 1 : 0;
    this.#weight += (target - this.#weight) * Math.min(1, dt * 10);
    if (this.#weight < 1e-3) {
      this.#resetHips();
      return;
    }

    object.updateMatrixWorld(true);

    const desired: number[] = [];
    const world = new THREE.Vector3();
    for (const leg of this.#legs) {
      leg.foot.getWorldPosition(world);
      const hit = physics.raycast(
        new THREE.Vector3(world.x, world.y + 0.6, world.z),
        new THREE.Vector3(0, -1, 0),
        1.6,
        { layers: layerMask(CollisionLayer.Terrain, CollisionLayer.Prop) },
      );
      if (hit === null) {
        desired.push(0);
        continue;
      }
      const wanted = hit.point.y + leg.soleOffset - world.y;
      desired.push(THREE.MathUtils.clamp(wanted, -this.#options.maxLift, this.#options.maxLift));
    }

    // Hips follow the lower foot so the higher leg never has to over-extend.
    const lowest = Math.min(...desired, 0);
    const hipTarget = THREE.MathUtils.clamp(lowest, -this.#options.maxHipDrop, 0) * this.#weight;
    this.#hipDrop += (hipTarget - this.#hipDrop) * Math.min(1, dt * this.#options.damping);
    this.#applyHips();

    object.updateMatrixWorld(true);

    for (let i = 0; i < this.#legs.length; i++) {
      const leg = this.#legs[i];
      const wanted = desired[i];
      if (leg === undefined || wanted === undefined) continue;
      leg.offset += (wanted * this.#weight - leg.offset) * Math.min(1, dt * this.#options.damping);
      if (Math.abs(leg.offset) < 1e-4) continue;
      this.#solveLeg(leg, leg.offset - this.#hipDrop);
    }
  }

  dispose(): void {
    this.#resetHips();
    this.#legs.length = 0;
  }

  /* -- internals ---------------------------------------------------------- */

  /**
   * Offset the hips downward, *additively*.
   *
   * Additive and not absolute, because every locomotion clip animates
   * `hips.position` — that vertical bob is most of what sells a stride — and
   * assigning an absolute height here would flatten it. The mixer rewrites the
   * authored value every frame before this runs, so adding the correction on
   * top preserves the bob and the correction both.
   */
  #applyHips(): void {
    const hips = this.#hips;
    if (hips === null || Math.abs(this.#hipDrop) < 1e-5) return;
    // The hips track is authored in model units; the drop is in metres, so it
    // has to be divided back out by the character's world scale.
    const scale = hips.parent?.getWorldScale(new THREE.Vector3()).y ?? 1;
    hips.position.y += this.#hipDrop / Math.max(scale, 1e-4);
    hips.updateMatrixWorld(true);
  }

  #resetHips(): void {
    this.#hipDrop = 0;
    for (const leg of this.#legs) leg.offset = 0;
  }

  /**
   * Aim one leg at a foot target `lift` metres above the animated foot.
   *
   * Two rotations, both applied in world space: first the whole chain is swung
   * so it points at the target, then the knee is opened or closed to make the
   * chain the right *length*. The bend axis is taken from the existing pose, so
   * the knee keeps whatever direction the animator gave it and never inverts.
   */
  #solveLeg(leg: Leg, lift: number): void {
    const root = leg.upper.getWorldPosition(new THREE.Vector3());
    const knee = leg.lower.getWorldPosition(new THREE.Vector3());
    const foot = leg.foot.getWorldPosition(new THREE.Vector3());

    const upperLength = root.distanceTo(knee);
    const lowerLength = knee.distanceTo(foot);
    if (upperLength < 1e-4 || lowerLength < 1e-4) return;

    const target = foot.clone();
    target.y += lift;

    const toFoot = foot.clone().sub(root);
    const toTarget = target.clone().sub(root);
    const reach = toTarget.length();
    if (reach < 1e-4) return;

    // Bend axis from the current pose. Degenerate on a perfectly straight leg,
    // in which case the character's right is a safe substitute — a knee bends
    // forwards.
    let axis = knee.clone().sub(root).cross(foot.clone().sub(knee));
    if (axis.lengthSq() < 1e-8) axis = new THREE.Vector3(1, 0, 0);
    axis.normalize();

    // 1. Swing the chain onto the target line.
    const swing = new THREE.Quaternion().setFromUnitVectors(
      toFoot.clone().normalize(),
      toTarget.clone().normalize(),
    );
    rotateBoneWorld(leg.upper, swing);

    // 2. Re-derive the pose and open the joint to the required angle.
    const knee2 = leg.lower.getWorldPosition(new THREE.Vector3());
    const foot2 = leg.foot.getWorldPosition(new THREE.Vector3());
    const currentRoot = angleBetween(knee2.clone().sub(root), toTarget);
    const currentJoint = angleBetween(root.clone().sub(knee2), foot2.clone().sub(knee2));
    const solved = solveTwoBoneAngles(upperLength, lowerLength, reach);

    rotateBoneWorld(
      leg.upper,
      new THREE.Quaternion().setFromAxisAngle(axis, currentRoot - solved.root),
    );
    rotateBoneWorld(
      leg.lower,
      new THREE.Quaternion().setFromAxisAngle(axis, currentJoint - solved.joint),
    );
  }
}

function angleBetween(a: THREE.Vector3, b: THREE.Vector3): number {
  const denominator = a.length() * b.length();
  if (denominator < 1e-8) return 0;
  return Math.acos(THREE.MathUtils.clamp(a.dot(b) / denominator, -1, 1));
}
