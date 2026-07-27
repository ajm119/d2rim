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

/* -------------------------------------------------------------------------- */
/* Stance detection                                                           */
/* -------------------------------------------------------------------------- */

/** What the stance test looks at, for one foot, on one frame. */
export interface StanceSample {
  /** Height of the sole above the surface under it, metres. */
  readonly clearance: number;
  /**
   * Lowest clearance this foot has reached recently, metres.
   *
   * The reason the test is not simply "clearance is small": `clearance` is
   * measured from a sole offset taken in the bind pose, and no rig's bind pose
   * is exactly its contact pose. On this one the offset is out by a few
   * centimetres, which is the same order as the whole walk cycle's foot lift —
   * so an absolute threshold either plants nothing or plants everything. The
   * floor cancels that bias out: what matters is how far *above its own
   * contact height* the foot currently is.
   */
  readonly stanceFloor: number;
  /** Largest clearance this foot has reached recently, metres. */
  readonly swingPeak: number;
  /** World speed of the *animated* foot in the ground plane, m/s. */
  readonly animatedSpeed: number;
  /** World speed of the character, m/s. */
  readonly bodySpeed: number;
}

export interface StanceThresholds {
  readonly plantClearanceRatio: number;
  readonly releaseClearanceRatio: number;
  readonly plantClearanceFloor: number;
  readonly plantSpeedRatio: number;
  readonly plantSpeedFloor: number;
}

/**
 * Is this foot in contact with the ground and not travelling?
 *
 * Pure, and exported, because it is the single judgement the whole foot lock
 * rests on and it is far easier to argue about against a table of numbers than
 * against a video of a character walking.
 *
 * Three properties, and each one is there because leaving it out was tried and
 * measured:
 *
 * - **Relative, not absolute.** See `stanceFloor` above. The first version used
 *   a fraction of the swing peak alone; at walking pace the foot barely lifts,
 *   the fraction collapsed onto its floor, and the bind-pose bias then decided
 *   everything. Plants lasted six frames where a walk stance is twenty-four.
 * - **Hysteretic.** A separate, looser threshold to *stay* planted than to
 *   become planted. Without it the decision chatters across its own boundary
 *   and the lock spends most of its time blending in and out — which is worse
 *   than no lock at all, because a blend is motion.
 * - **Speed only on the way in.** The swing foot passes low over the ground at
 *   both ends of its arc; the speed test is what stops it being mistaken for a
 *   plant there. Once a foot *is* pinned its animated speed is not evidence of
 *   anything, because the pin is what the foot is following. Applying the test
 *   in both directions releases every pin the moment the animation accelerates
 *   underneath it, which is exactly what the lock exists to override.
 */
export function isFootPlanted(
  sample: StanceSample,
  thresholds: StanceThresholds,
  alreadyPlanted = false,
): boolean {
  if (!Number.isFinite(sample.clearance)) return false;
  // Amplitude of this foot's clearance cycle, floored so a character standing
  // perfectly still does not divide the world by zero.
  const amplitude = Math.max(0.02, sample.swingPeak - sample.stanceFloor);
  const ratio = alreadyPlanted ? thresholds.releaseClearanceRatio : thresholds.plantClearanceRatio;
  const limit = sample.stanceFloor + Math.max(thresholds.plantClearanceFloor, amplitude * ratio);
  if (sample.clearance > limit) return false;
  if (alreadyPlanted) return true;
  const speedLimit = Math.max(
    thresholds.plantSpeedFloor,
    sample.bodySpeed * thresholds.plantSpeedRatio,
  );
  return sample.animatedSpeed <= speedLimit;
}

/**
 * Should a live pin be let go?
 *
 * Three reasons, and they are the three ways a foot lock turns into its own
 * bug if it is left out:
 *
 * 1. the foot has lifted — the ordinary end of a stance;
 * 2. the animation has walked away from the pin by more than `maxDrift`. This
 *    is what makes turning safe: a character pivoting on the spot swings the
 *    pinned foot's animated pose away from the pin, and rather than the leg
 *    fighting the rotation the pin is dropped and re-taken;
 * 3. the pin is further from the hip than the leg is long. Holding it would
 *    straighten the chain and pop, so it goes before it can.
 */
export function shouldReleasePin(
  contact: boolean,
  drift: number,
  maxDrift: number,
  hipToPin: number,
  legLength: number,
): boolean {
  return !contact || drift > maxDrift || hipToPin > legLength * 0.98;
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

  /* -- foot lock ---------------------------------------------------------- */

  /** Previous animated foot position, world space. Drives the speed test. */
  readonly previous: THREE.Vector3;
  hasPrevious: boolean;
  /**
   * Peak clearance this foot has reached recently, metres.
   *
   * The self-calibrating half of the stance test. A fixed "the foot is down
   * when it is within N centimetres of the ground" threshold has to be re-tuned
   * for every rig, every gait and every playback rate, and is wrong for all of
   * them at once; the *shape* of a gait is that clearance oscillates between
   * roughly zero and a peak, so a fraction of the observed peak is the same
   * question asked in a way that answers itself. Decays so a character that
   * stops walking does not keep a stale peak from its last stride.
   */
  swingPeak: number;
  /** Lowest clearance seen recently. Cancels the bind-pose sole-offset bias. */
  stanceFloor: number;
  /** Consecutive frames the plant test has passed. Debounces a one-frame dip. */
  contactFrames: number;
  /** World position the sole is pinned to while planted, or null. */
  pin: THREE.Vector3 | null;
  /** How much of the pin is blended in, `0..1`. */
  lock: number;
  /** Diagnostics: clearance and animated world speed from the last frame. */
  clearance: number;
  animatedSpeed: number;
}

/** What {@link FootIK.debug} reports for one leg. */
export interface FootIKLegDebug {
  readonly clearance: number;
  readonly animatedSpeed: number;
  readonly swingPeak: number;
  readonly stanceFloor: number;
  readonly locked: boolean;
  readonly lock: number;
  /** How far the pin has drifted from the animated foot, metres. */
  readonly drift: number;
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

  /* -- foot lock ---------------------------------------------------------- */

  /** Pin the stance foot in world space. Default true. */
  readonly footLock?: boolean;
  /**
   * Fraction of a foot's recent clearance *amplitude*, above its own contact
   * height, under which it counts as newly down. Default 0.22.
   */
  readonly plantClearanceRatio?: number;
  /**
   * The same fraction for a foot that is already planted. Larger than
   * `plantClearanceRatio` on purpose: the gap between the two is the
   * hysteresis, and without it the decision chatters. Default 0.6.
   */
  readonly releaseClearanceRatio?: number;
  /** Absolute clearance slack for both tests, metres. Default 0.012. */
  readonly plantClearanceFloor?: number;
  /**
   * Animated foot speed under which a foot may be planted, as a multiple of
   * body speed. Default 1 — the swing foot runs at well over 1.5x body speed
   * while the stance foot runs at about 0.6x, so this separates them with a
   * wide margin without needing to know either number.
   */
  readonly plantSpeedRatio?: number;
  /** Speed floor for the same test, m/s. Default 0.45. */
  readonly plantSpeedFloor?: number;
  /**
   * Furthest a pin may drift from the animated foot before it is released,
   * metres. Default 0.42.
   *
   * Sized off the error being corrected, not off a feeling. The drift *is* the
   * animation's sliding error accumulated over one stance: at a run the body
   * covers about 0.68 m during contact while the animated stance foot travels
   * about 0.45 m of that with it, so anything under 0.45 releases the pin
   * mid-stance and re-plants, which shows up in the harness as a short
   * `lockedFraction` and a high plant count.
   *
   * This is the safety valve that makes the lock unable to misbehave: a pin
   * that the animation has walked away from, a character that turned on the
   * spot, a slope that moved the ground out from under the plant — all of them
   * end the same way, with the pin released and the foot back on the animated
   * pose within a blend.
   */
  readonly maxPinDrift?: number;
  /** Seconds the pin blends in and out over. Default 0.05. */
  readonly lockBlend?: number;
  /**
   * Frames the plant test must pass consecutively before a pin is taken.
   * Default 2. A swing foot can dip under the clearance threshold for a single
   * frame at either end of its arc; two frames of agreement costs 33 ms of
   * stance and removes that whole class of false plant.
   */
  readonly plantDebounce?: number;
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
  #plants = 0;

  constructor(options: FootIKOptions = {}) {
    this.#options = {
      bones: options.bones ?? { upper: 'upperleg', lower: 'lowerleg', foot: 'foot' },
      maxLift: options.maxLift ?? 0.42,
      maxHipDrop: options.maxHipDrop ?? 0.3,
      damping: options.damping ?? 12,
      enabled: options.enabled ?? true,
      footLock: options.footLock ?? true,
      plantClearanceRatio: options.plantClearanceRatio ?? 0.22,
      releaseClearanceRatio: options.releaseClearanceRatio ?? 0.6,
      plantClearanceFloor: options.plantClearanceFloor ?? 0.012,
      plantSpeedRatio: options.plantSpeedRatio ?? 0.85,
      plantSpeedFloor: options.plantSpeedFloor ?? 0.4,
      maxPinDrift: options.maxPinDrift ?? 0.42,
      lockBlend: options.lockBlend ?? 0.05,
      plantDebounce: options.plantDebounce ?? 2,
    };
  }

  /**
   * Per-leg solver state, for `tools/verify-footplant.mjs` and the debug
   * overlay. Reading it is free; nothing here is computed on demand.
   */
  get debug(): readonly FootIKLegDebug[] {
    return this.#legs.map((leg) => ({
      clearance: leg.clearance,
      animatedSpeed: leg.animatedSpeed,
      swingPeak: leg.swingPeak,
      stanceFloor: leg.stanceFloor,
      locked: leg.pin !== null,
      lock: leg.lock,
      drift:
        leg.pin === null
          ? 0
          : Math.hypot(leg.pin.x - leg.previous.x, leg.pin.z - leg.previous.z),
    }));
  }

  /** Plants made and released since boot. A cadence sanity check. */
  get plants(): number {
    return this.#plants;
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
      this.#legs.push({
        upper,
        lower,
        foot,
        soleOffset: Math.max(0.02, footY - rootY),
        offset: 0,
        previous: new THREE.Vector3(),
        hasPrevious: false,
        swingPeak: 0,
        stanceFloor: 0,
        contactFrames: 0,
        pin: null,
        lock: 0,
        clearance: 0,
        animatedSpeed: 0,
      });
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

  /**
   * ### The foot lock, and why the vertical solve alone was never going to work
   *
   * Matching the animation's playback rate to the character's ground speed gets
   * the *average* stance-phase foot velocity to zero and no more. This rig's
   * stance foot does not travel at a constant rate: over `Running_A`'s contact
   * the foot covers between 1.97 and 2.37 m per cycle, so whatever body speed
   * the rate is matched at, the foot is too fast for part of the stance and too
   * slow for the rest. Measured, that floor is a slower foot at 0.60–0.67 of
   * body speed and a best plant of 0.33 m/s — plainly sliding.
   *
   * The only way to hold a foot still is to *hold it still*: decide when it is
   * in contact, record where it is at that instant, and solve the leg to that
   * world point until it lifts. What follows is that, plus the two guards that
   * keep it from becoming its own bug — a drift limit, so a pin the animation
   * or the character's rotation has walked away from is released rather than
   * dragged, and a reach limit, so a leg is never asked to stretch past the
   * length it has.
   */
  lateUpdate(_ctx: GameContext, dt: number): void {
    if (!this.active) return;
    const player = this.#player;
    const physics = this.#physics;
    const object = player?.object ?? null;
    if (player === null || physics === null || object === null || !physics.ready) return;
    const step = Math.max(1e-4, Math.min(dt, 0.1));

    // Off in the air, and eased rather than switched: a foot that snaps to the
    // ground on the landing frame looks like a glitch, not like a landing.
    const target = player.grounded ? 1 : 0;
    this.#weight += (target - this.#weight) * Math.min(1, step * 10);
    if (this.#weight < 1e-3) {
      this.#resetHips();
      return;
    }

    object.updateMatrixWorld(true);

    /* -- 1. sample the animated pose and the ground under each foot -------- */

    const bodySpeed = player.speed;
    const desired: number[] = [];
    const groundY: (number | null)[] = [];
    const animated: THREE.Vector3[] = [];
    const world = new THREE.Vector3();
    for (const leg of this.#legs) {
      leg.foot.getWorldPosition(world);
      animated.push(world.clone());
      const hit = physics.raycast(
        new THREE.Vector3(world.x, world.y + 0.6, world.z),
        new THREE.Vector3(0, -1, 0),
        1.6,
        { layers: layerMask(CollisionLayer.Terrain, CollisionLayer.Prop) },
      );
      if (hit === null) {
        desired.push(0);
        groundY.push(null);
        leg.clearance = Number.POSITIVE_INFINITY;
        continue;
      }
      groundY.push(hit.point.y);
      // Clearance of the *sole*, which is the surface that is or is not on the
      // ground. The foot bone sits `soleOffset` above it.
      leg.clearance = Math.max(0, world.y - leg.soleOffset - hit.point.y);
      const wanted = hit.point.y + leg.soleOffset - world.y;
      desired.push(THREE.MathUtils.clamp(wanted, -this.#options.maxLift, this.#options.maxLift));
    }

    /* -- 2. hips ----------------------------------------------------------- */

    // Hips follow the lower foot so the higher leg never has to over-extend.
    const lowest = Math.min(...desired, 0);
    const hipTarget = THREE.MathUtils.clamp(lowest, -this.#options.maxHipDrop, 0) * this.#weight;
    this.#hipDrop += (hipTarget - this.#hipDrop) * Math.min(1, step * this.#options.damping);
    this.#applyHips();

    object.updateMatrixWorld(true);

    /* -- 3. stance detection, then solve ----------------------------------- */

    const now = new THREE.Vector3();
    const free = new THREE.Vector3();
    const aim = new THREE.Vector3();
    for (let i = 0; i < this.#legs.length; i++) {
      const leg = this.#legs[i];
      const wanted = desired[i];
      const animatedPose = animated[i];
      if (leg === undefined || wanted === undefined || animatedPose === undefined) continue;

      // Animated world speed, measured on the *pre-IK* pose. Measuring it after
      // the solve would feed the lock its own output and latch it on.
      leg.animatedSpeed = leg.hasPrevious
        ? Math.hypot(animatedPose.x - leg.previous.x, animatedPose.z - leg.previous.z) / step
        : 0;
      leg.previous.copy(animatedPose);
      leg.hasPrevious = true;
      // Envelope trackers for this foot's clearance cycle. The peak decays
      // downward and the floor creeps upward, both slowly enough to survive a
      // whole stride and fast enough to follow a change of gait or of ground.
      if (Number.isFinite(leg.clearance)) {
        leg.swingPeak = Math.max(leg.swingPeak - step * 0.5, leg.clearance);
        leg.stanceFloor = Math.min(leg.stanceFloor + step * 0.25, leg.clearance);
      }

      leg.offset += (wanted * this.#weight - leg.offset) * Math.min(1, step * this.#options.damping);

      // The pose this leg would hold with no lock: the animation, corrected
      // vertically onto the surface under it.
      leg.foot.getWorldPosition(now);
      free.set(now.x, now.y + leg.offset - this.#hipDrop, now.z);

      this.#updatePin(leg, free, groundY[i] ?? null, bodySpeed, step);

      if (leg.pin === null && Math.abs(leg.offset) < 1e-4 && leg.lock < 1e-3) continue;
      // `lock` is the only thing that decides how much of the pin is used, and
      // it is scaled by the airborne weight so a jump releases the feet the
      // same way it releases the vertical correction.
      const blend = leg.pin === null ? 0 : leg.lock * this.#weight;
      aim.copy(free);
      if (blend > 0 && leg.pin !== null) aim.lerp(leg.pin, blend);
      this.#solveLeg(leg, aim);
    }
  }

  dispose(): void {
    this.#resetHips();
    this.#legs.length = 0;
  }

  /* -- foot lock ----------------------------------------------------------- */

  /**
   * Decide whether this foot is in contact, and maintain its pin.
   *
   * @param free  where the foot bone would sit with no lock, world space
   * @param ground surface height under the foot, or null if the ray missed
   */
  #updatePin(
    leg: Leg,
    free: THREE.Vector3,
    ground: number | null,
    bodySpeed: number,
    dt: number,
  ): void {
    const options = this.#options;
    const rate = Math.min(1, dt / Math.max(1e-3, options.lockBlend));

    if (!options.footLock || ground === null) {
      leg.pin = null;
      leg.lock = Math.max(0, leg.lock - rate);
      return;
    }

    const sample = {
      clearance: leg.clearance,
      stanceFloor: leg.stanceFloor,
      swingPeak: leg.swingPeak,
      animatedSpeed: leg.animatedSpeed,
      bodySpeed,
    };
    const planted = leg.pin !== null;
    const contact = isFootPlanted(sample, options, planted);
    leg.contactFrames = isFootPlanted(sample, options, false) ? leg.contactFrames + 1 : 0;

    if (leg.pin !== null) {
      // The bone lengths of the rig, not of the target: how far this leg can
      // reach is a property of the skeleton and must not be read off wherever
      // the foot happens to be being asked to go.
      const hip = leg.upper.getWorldPosition(new THREE.Vector3());
      const knee = leg.lower.getWorldPosition(new THREE.Vector3());
      const ankle = leg.foot.getWorldPosition(new THREE.Vector3());
      const legLength = hip.distanceTo(knee) + knee.distanceTo(ankle);
      const released = shouldReleasePin(
        contact,
        Math.hypot(leg.pin.x - free.x, leg.pin.z - free.z),
        options.maxPinDrift,
        hip.distanceTo(leg.pin),
        legLength,
      );
      if (released) leg.pin = null;
    }

    if (leg.pin === null && leg.contactFrames >= options.plantDebounce) {
      // Plant. The pin is taken from the *corrected* pose, so the foot is
      // pinned where it is standing rather than where the clip put it.
      leg.pin = free.clone();
      leg.lock = 0;
      this.#plants++;
    }

    leg.lock =
      leg.pin === null
        ? Math.max(0, leg.lock - rate)
        : Math.min(1, leg.lock + rate);
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
    for (const leg of this.#legs) {
      leg.offset = 0;
      leg.pin = null;
      leg.lock = 0;
      leg.hasPrevious = false;
      leg.swingPeak = 0;
      leg.stanceFloor = 0;
      leg.contactFrames = 0;
    }
  }

  /**
   * Aim one leg at an absolute world-space foot target.
   *
   * Absolute rather than "a lift above the animated foot", which is what it
   * used to take: a lock has to be able to hold a foot at a fixed point in the
   * world while the character walks past it, and a target expressed relative to
   * the animated pose moves with the thing it is supposed to be independent of.
   *
   * Two rotations, both applied in world space: first the whole chain is swung
   * so it points at the target, then the knee is opened or closed to make the
   * chain the right *length*. The bend axis is taken from the existing pose, so
   * the knee keeps whatever direction the animator gave it and never inverts.
   */
  #solveLeg(leg: Leg, target: THREE.Vector3): void {
    const root = leg.upper.getWorldPosition(new THREE.Vector3());
    const knee = leg.lower.getWorldPosition(new THREE.Vector3());
    const foot = leg.foot.getWorldPosition(new THREE.Vector3());

    const upperLength = root.distanceTo(knee);
    const lowerLength = knee.distanceTo(foot);
    if (upperLength < 1e-4 || lowerLength < 1e-4) return;

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
