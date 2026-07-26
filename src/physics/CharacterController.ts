/**
 * @module physics/CharacterController
 *
 * A kinematic capsule controller built on Rapier's
 * `KinematicCharacterController`, plus the small amount of policy Rapier
 * deliberately leaves to the game: how gravity is applied while grounded, what
 * counts as a jump, and what to do on a slope too steep to stand on.
 *
 * ### Why kinematic
 *
 * A dynamic rigid body with a locked rotation is the lazy way to move a
 * character and it never feels right: the player's acceleration is fought by
 * the solver, contacts bounce, and standing still on a 20-degree slope means
 * creeping downhill forever. A kinematic capsule moved by explicit sweeps has
 * exactly the motion the input asked for, minus whatever the world took away.
 *
 * ### The four things that make it feel solid
 *
 * 1. **Ground stick, not gravity.** While grounded the vertical velocity is
 *    pinned to a small negative constant instead of integrating gravity. An
 *    integrated fall speed grows every frame the character is on the ground and
 *    is spent the instant it steps off a lip, which reads as being yanked down.
 * 2. **Horizontal intent, handed to Rapier unmodified.** It is tempting to
 *    rotate the movement into the slope plane before the sweep, and it is
 *    wrong: `computeColliderMovement` already climbs, slides and steps for
 *    itself, and pre-tilting the input fights it. Uphill, the added `+y` is
 *    swamped by the ground-stick bias and the net vector drives *into* the
 *    hillside — the character walks at a few centimetres per second up a
 *    13-degree bank and appears to be stuck on nothing. The projection helper
 *    is still exported below, because animation and camera code legitimately
 *    want "this velocity, laid onto that surface"; the controller just does not
 *    use it on the way in.
 *
 *    Because the vertical is never derived from horizontal intent, cresting a
 *    ramp cannot fling the character into the air.
 * 3. **Snap to ground.** A short downward search after each move keeps the
 *    capsule welded to the surface across the sub-centimetre gaps a heightfield
 *    leaves between triangles, which is where slope jitter comes from.
 * 4. **Autostep.** Rubble and the ruined wall's fallen blocks are 20–30 cm
 *    obstacles. Without autostep the player stops dead on them, which reads as
 *    a bug in the geometry rather than as a decision.
 *
 * The slope, step and jump policy is written as pure functions above the class
 * so it can be tested without booting the WASM world.
 */

import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three/webgpu';

import { COLLISION_GROUPS, CollisionLayer, layerMask } from './Layers';
import type { PhysicsWorld } from './PhysicsWorld';

/* -------------------------------------------------------------------------- */
/* Pure policy                                                                */
/* -------------------------------------------------------------------------- */

/** How the controller should treat the surface it is standing on. */
export interface SlopeClassification {
  /** Angle between the surface normal and up, in radians. */
  readonly angle: number;
  /** Shallow enough to walk up. */
  readonly walkable: boolean;
  /** Steep enough that the character slides down it under its own weight. */
  readonly sliding: boolean;
}

/**
 * Classify a ground normal against the climb and slide limits.
 *
 * The two thresholds are independent on purpose and `slideAngle` is normally
 * the *smaller* of the two, which gives the band that makes slopes feel real:
 * below `slideAngle` you stand still, between the two you keep your footing but
 * drift, and above `climbAngle` you cannot make upward progress at all.
 */
export function classifySlope(
  normalY: number,
  climbAngle: number,
  slideAngle: number,
): SlopeClassification {
  const angle = Math.acos(THREE.MathUtils.clamp(normalY, -1, 1));
  return {
    angle,
    walkable: angle <= climbAngle + 1e-6,
    sliding: angle >= slideAngle - 1e-6,
  };
}

/**
 * Rotate a horizontal motion vector into the plane of a slope, preserving its
 * length.
 *
 * Preserving length matters: the naive `v - n * dot(v, n)` shortens the vector
 * by `cos(slope)`, so a body on a 30-degree bank silently loses 13% of its
 * speed. Here the direction comes from the projection and the magnitude from
 * the input.
 *
 * Not used by the sweep — see the module docs — but exported because it is the
 * right way to ask "how fast is this body travelling *along the ground*", which
 * is what a slope-aware camera or a stride-matched animation wants.
 */
export function projectOntoGroundPlane(
  motion: THREE.Vector3,
  normal: THREE.Vector3,
  out = new THREE.Vector3(),
): THREE.Vector3 {
  const speed = Math.hypot(motion.x, motion.z);
  if (speed < 1e-9) return out.set(0, 0, 0);
  const dot = motion.x * normal.x + motion.y * normal.y + motion.z * normal.z;
  out.set(motion.x - normal.x * dot, motion.y - normal.y * dot, motion.z - normal.z * dot);
  const length = out.length();
  if (length < 1e-9) return out.set(0, 0, 0);
  return out.multiplyScalar(speed / length);
}

/**
 * The downhill direction of a slope, as a unit vector. Zero on flat ground.
 * Used to make a too-steep surface shed the character rather than hold them.
 */
export function downhillDirection(
  normal: THREE.Vector3,
  out = new THREE.Vector3(),
): THREE.Vector3 {
  out.set(normal.x, 0, normal.z);
  const length = out.length();
  return length < 1e-6 ? out.set(0, 0, 0) : out.multiplyScalar(1 / length);
}

/** The jump gate's view of the character. */
export interface JumpState {
  /** Seconds since the character was last grounded. 0 while grounded. */
  readonly timeSinceGrounded: number;
  /** Whether a jump has already been spent since the last ground contact. */
  readonly jumpSpent: boolean;
  /** Seconds of buffered jump input remaining; 0 means no request pending. */
  readonly bufferRemaining: number;
}

/**
 * Whether a jump should fire this step.
 *
 * Two forgivenesses, both standard and both invisible when they work:
 * *coyote time* accepts a jump pressed shortly after walking off a ledge, and
 * the *input buffer* accepts a jump pressed shortly before landing. Without the
 * first, running off a slope and pressing jump does nothing about a third of
 * the time; without the second, chained jumps require frame-perfect timing.
 */
export function shouldJump(state: JumpState, coyoteTime: number): boolean {
  if (state.bufferRemaining <= 0) return false;
  if (state.jumpSpent) return false;
  return state.timeSinceGrounded <= coyoteTime;
}

/**
 * Whether an obstacle should be stepped over rather than collided with.
 *
 * Mirrors the configuration handed to Rapier's autostep so the intent is
 * testable: an obstacle is stepped when its lip is within the step offset *and*
 * there is enough clear tread beyond it to put a foot on. The width test is
 * what stops the character from mantling onto a fence rail.
 */
export function canAutostep(
  obstacleHeight: number,
  clearWidthBeyond: number,
  stepOffset: number,
  minWidth: number,
): boolean {
  if (obstacleHeight <= 0) return false;
  if (obstacleHeight > stepOffset) return false;
  return clearWidthBeyond >= minWidth;
}

/* -------------------------------------------------------------------------- */
/* Controller                                                                 */
/* -------------------------------------------------------------------------- */

export interface CharacterControllerOptions {
  /** Capsule radius, metres. Default 0.32 — a broad-shouldered Barbarian. */
  readonly radius?: number;
  /** Total standing height including both caps, metres. Default 1.85. */
  readonly height?: number;
  /** Rapier's collision gap. Default 0.02. Too small and contacts jitter. */
  readonly skinWidth?: number;
  /** Steepest slope the character can walk up, degrees. Default 50. */
  readonly maxClimbAngle?: number;
  /** Shallowest slope the character slides down, degrees. Default 55. */
  readonly minSlideAngle?: number;
  /** Tallest lip that is stepped over instead of blocking. Default 0.42 m. */
  readonly stepOffset?: number;
  /** Clear tread required beyond a step. Default 0.18 m. */
  readonly stepMinWidth?: number;
  /** Downward search distance used to stay welded to the ground. Default 0.4 m. */
  readonly snapDistance?: number;
  /** Gravity, m/s². Default −24 (game gravity: falls read as heavy). */
  readonly gravity?: number;
  /** Terminal velocity, m/s. Default 55. */
  readonly maxFallSpeed?: number;
  /** Jump take-off speed, m/s. Default 6.2 — about a 0.8 m hop under −24. */
  readonly jumpSpeed?: number;
  /** Grace period after leaving the ground in which a jump still fires. */
  readonly coyoteTime?: number;
  /** How long a jump press is remembered while airborne. Default 0.15 s. */
  readonly jumpBuffer?: number;
  /** Downward speed held while grounded. Default 2 m/s. */
  readonly groundStickSpeed?: number;
  /** Downhill acceleration applied on unwalkable slopes, m/s². Default 12. */
  readonly slideAcceleration?: number;
}

/** What one {@link CharacterController.move} did. */
export interface MoveResult {
  /** Feet position after the move. The same live vector as `position`. */
  readonly position: THREE.Vector3;
  readonly grounded: boolean;
  /** True on the frame the character regained the ground. */
  readonly landed: boolean;
  /** True on the frame a jump left the ground. */
  readonly jumped: boolean;
  readonly groundNormal: THREE.Vector3;
  readonly slope: SlopeClassification;
  /** Vertical velocity after the move, m/s. */
  readonly verticalVelocity: number;
  /** Fraction of the requested horizontal motion actually achieved, 0..1. */
  readonly motionRatio: number;
  /** Whether the sweep collided with anything at all. */
  readonly blocked: boolean;
  /**
   * Outward normal of the steepest near-vertical surface hit this step, or
   * `null` when nothing wall-like was touched.
   *
   * Ground contacts are excluded, because the caller uses this to cancel the
   * component of its velocity that is driving into a wall, and doing that
   * against the floor normal would cancel gravity.
   */
  readonly wallNormal: THREE.Vector3 | null;
}

const UP = new THREE.Vector3(0, 1, 0);

export class CharacterController {
  readonly #physics: PhysicsWorld;
  readonly #options: Required<CharacterControllerOptions>;

  readonly #position = new THREE.Vector3();
  readonly #previous = new THREE.Vector3();
  readonly #groundNormal = new THREE.Vector3(0, 1, 0);

  #controller: RAPIER.KinematicCharacterController;
  #body: RAPIER.RigidBody;
  #collider: RAPIER.Collider;

  #verticalVelocity = 0;
  #grounded = false;
  #timeSinceGrounded = Number.POSITIVE_INFINITY;
  #jumpSpent = false;
  #jumpBuffer = 0;
  #slope: SlopeClassification = { angle: 0, walkable: true, sliding: false };
  #disposed = false;

  constructor(physics: PhysicsWorld, options: CharacterControllerOptions = {}) {
    this.#physics = physics;
    this.#options = {
      radius: options.radius ?? 0.32,
      height: options.height ?? 1.85,
      skinWidth: options.skinWidth ?? 0.02,
      maxClimbAngle: options.maxClimbAngle ?? 50,
      minSlideAngle: options.minSlideAngle ?? 55,
      stepOffset: options.stepOffset ?? 0.42,
      stepMinWidth: options.stepMinWidth ?? 0.18,
      snapDistance: options.snapDistance ?? 0.4,
      gravity: options.gravity ?? -24,
      maxFallSpeed: options.maxFallSpeed ?? 55,
      jumpSpeed: options.jumpSpeed ?? 6.2,
      coyoteTime: options.coyoteTime ?? 0.14,
      jumpBuffer: options.jumpBuffer ?? 0.15,
      groundStickSpeed: options.groundStickSpeed ?? 2,
      slideAcceleration: options.slideAcceleration ?? 12,
    };

    const world = physics.world;
    const radius = this.#options.radius;
    const halfHeight = Math.max(0.05, this.#options.height * 0.5 - radius);

    this.#body = world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(0, 0, 0),
    );
    const record = physics.addCollider(
      RAPIER.ColliderDesc.capsule(halfHeight, radius)
        // The body origin is the character's feet, so the capsule is pushed up
        // by half its height. Everything outside this class talks in feet
        // positions, because that is where the model's root goes and where the
        // ground is.
        .setTranslation(0, radius + halfHeight, 0)
        .setCollisionGroups(COLLISION_GROUPS.player)
        .setFriction(0)
        .setRestitution(0),
      { kind: 'character', label: 'character.capsule' },
      this.#body,
    );
    this.#collider = record.collider;

    this.#controller = world.createCharacterController(this.#options.skinWidth);
    this.#controller.setUp({ x: 0, y: 1, z: 0 });
    this.#controller.setSlideEnabled(true);
    this.#controller.setMaxSlopeClimbAngle(THREE.MathUtils.degToRad(this.#options.maxClimbAngle));
    this.#controller.setMinSlopeSlideAngle(THREE.MathUtils.degToRad(this.#options.minSlideAngle));
    this.#controller.enableAutostep(this.#options.stepOffset, this.#options.stepMinWidth, true);
    this.#controller.enableSnapToGround(this.#options.snapDistance);
    // Nudging the character out along contact normals is what stops it burying
    // itself in a heightfield seam and then popping free a frame later.
    this.#controller.setNormalNudgeFactor(1e-4);
    this.#controller.setApplyImpulsesToDynamicBodies(true);
    this.#controller.setCharacterMass(80);
  }

  /* -- state -------------------------------------------------------------- */

  /** Live feet position. Mutating it directly is a bug; use {@link setPosition}. */
  get position(): THREE.Vector3 {
    return this.#position;
  }

  /** Feet position at the start of the current fixed step. */
  get previousPosition(): THREE.Vector3 {
    return this.#previous;
  }

  get grounded(): boolean {
    return this.#grounded;
  }

  get groundNormal(): THREE.Vector3 {
    return this.#groundNormal;
  }

  get slope(): SlopeClassification {
    return this.#slope;
  }

  get verticalVelocity(): number {
    return this.#verticalVelocity;
  }

  get timeSinceGrounded(): number {
    return this.#timeSinceGrounded;
  }

  get collider(): RAPIER.Collider {
    return this.#collider;
  }

  get body(): RAPIER.RigidBody {
    return this.#body;
  }

  /** Capsule dimensions, for the camera and animation to reason about. */
  get dimensions(): { radius: number; height: number } {
    return { radius: this.#options.radius, height: this.#options.height };
  }

  /**
   * Feet position interpolated between the last two fixed steps.
   *
   * Simulation runs at 60 Hz and rendering does not, so drawing the character
   * at the raw simulation position stutters on any other refresh rate. `alpha`
   * comes from {@link Engine.alpha}.
   */
  interpolatedPosition(alpha: number, out = new THREE.Vector3()): THREE.Vector3 {
    return out.lerpVectors(this.#previous, this.#position, THREE.MathUtils.clamp(alpha, 0, 1));
  }

  /** Hard-set the position. Teleports; clears momentum and ground state. */
  setPosition(position: THREE.Vector3): void {
    this.#position.copy(position);
    this.#previous.copy(position);
    this.#verticalVelocity = 0;
    this.#grounded = false;
    this.#timeSinceGrounded = Number.POSITIVE_INFINITY;
    this.#body.setTranslation({ x: position.x, y: position.y, z: position.z }, true);
  }

  /** Queue a jump. Honoured for {@link CharacterControllerOptions.jumpBuffer}. */
  requestJump(): void {
    this.#jumpBuffer = this.#options.jumpBuffer;
  }

  /** Cancel any buffered jump — used when a menu opens mid-press. */
  clearJump(): void {
    this.#jumpBuffer = 0;
  }

  /* -- integration -------------------------------------------------------- */

  /**
   * Advance the character by one fixed step.
   *
   * `desiredVelocity` is the *horizontal* velocity the locomotion layer wants,
   * in world space, metres per second. Vertical motion belongs to the
   * controller: gravity, jumping and ground stick are policy, not input.
   */
  move(desiredVelocity: THREE.Vector3, dt: number): MoveResult {
    if (this.#disposed || !this.#physics.ready) {
      return this.#result(false, false, 0, false);
    }

    this.#previous.copy(this.#position);
    const wasGrounded = this.#grounded;
    const options = this.#options;

    this.#jumpBuffer = Math.max(0, this.#jumpBuffer - dt);

    // -- vertical ---------------------------------------------------------
    let jumped = false;
    const jumpNow = shouldJump(
      {
        timeSinceGrounded: this.#timeSinceGrounded,
        jumpSpent: this.#jumpSpent,
        bufferRemaining: this.#jumpBuffer,
      },
      options.coyoteTime,
    );

    if (jumpNow) {
      this.#verticalVelocity = options.jumpSpeed;
      this.#jumpSpent = true;
      this.#jumpBuffer = 0;
      this.#grounded = false;
      jumped = true;
    } else if (this.#grounded && this.#verticalVelocity <= 0) {
      // Ground stick. Constant, not integrated — see the module docs.
      this.#verticalVelocity = -options.groundStickSpeed;
    } else {
      this.#verticalVelocity = Math.max(
        -options.maxFallSpeed,
        this.#verticalVelocity + options.gravity * dt,
      );
    }

    // -- horizontal -------------------------------------------------------
    const motion = new THREE.Vector3(desiredVelocity.x, 0, desiredVelocity.z);
    const requested = Math.hypot(motion.x, motion.z);

    if (this.#grounded && !jumped) {
      if (!this.#slope.walkable) {
        // Too steep to stand on: bias the input downhill so a failed climb
        // becomes a slide rather than a stall.
        const downhill = downhillDirection(this.#groundNormal);
        motion.addScaledVector(downhill, options.slideAcceleration * dt);
      }
    }

    const desired = {
      x: motion.x * dt,
      y: motion.y * dt + this.#verticalVelocity * dt,
      z: motion.z * dt,
    };

    this.#controller.computeColliderMovement(
      this.#collider,
      desired,
      RAPIER.QueryFilterFlags.EXCLUDE_SENSORS,
      COLLISION_GROUPS.player,
    );
    const applied = this.#controller.computedMovement();

    this.#position.x += applied.x;
    this.#position.y += applied.y;
    this.#position.z += applied.z;
    this.#body.setTranslation(
      { x: this.#position.x, y: this.#position.y, z: this.#position.z },
      true,
    );

    // -- ground state -----------------------------------------------------
    this.#grounded = this.#controller.computedGrounded();
    if (this.#grounded) {
      this.#timeSinceGrounded = 0;
      this.#jumpSpent = false;
      if (this.#verticalVelocity < 0) this.#verticalVelocity = 0;
    } else {
      this.#timeSinceGrounded += dt;
    }

    // Rapier reports *that* the character is grounded but not the surface it is
    // standing on, and the slope response needs the normal. One short ray from
    // inside the capsule is cheaper than inspecting every computed collision.
    this.#sampleGroundNormal();
    this.#slope = classifySlope(
      this.#groundNormal.y,
      THREE.MathUtils.degToRad(options.maxClimbAngle),
      THREE.MathUtils.degToRad(options.minSlideAngle),
    );

    const achieved = Math.hypot(applied.x, applied.z);
    const ratio = requested * dt < 1e-9 ? 1 : THREE.MathUtils.clamp(achieved / (requested * dt), 0, 1);

    return this.#result(
      this.#grounded && !wasGrounded,
      jumped,
      ratio,
      this.#controller.numComputedCollisions() > 0,
      this.#findWallNormal(),
    );
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const world = this.#physics.ready ? this.#physics.world : null;
    if (world !== null) {
      world.removeCharacterController(this.#controller);
      world.removeRigidBody(this.#body);
    }
  }

  /* -- internals ---------------------------------------------------------- */

  #sampleGroundNormal(): void {
    const radius = this.#options.radius;
    const origin = new THREE.Vector3(
      this.#position.x,
      this.#position.y + radius + 0.05,
      this.#position.z,
    );
    const hit = this.#physics.raycast(
      origin,
      new THREE.Vector3(0, -1, 0),
      radius + 0.05 + this.#options.snapDistance,
      {
        layers: layerMask(CollisionLayer.Terrain, CollisionLayer.Prop),
        exclude: this.#collider,
      },
    );
    if (hit === null || hit.normal.lengthSq() < 1e-8) {
      this.#groundNormal.copy(UP);
      return;
    }
    // A heightfield ray can return the back face when the origin is a hair
    // below the surface; flip it rather than reporting an upside-down floor.
    this.#groundNormal.copy(hit.normal);
    if (this.#groundNormal.y < 0) this.#groundNormal.negate();

    // A single downward ray from inside the capsule is a cheap probe, not a
    // ground-truth contact: standing at the foot of a crate it happily returns
    // the crate's vertical *side*. Taking that as the floor makes the character
    // believe it is on an unclimbable slope and stops it dead a step away from
    // an obstacle it never touched.
    //
    // Rapier's `computedGrounded()` is the authority on whether there is ground;
    // this probe only supplies a normal to lean the movement into. So when the
    // probe returns something the character could not possibly be standing on,
    // discard it and use up. The worst case is a step of unprojected movement
    // on a slope, which the sweep corrects anyway.
    const limit = Math.cos(THREE.MathUtils.degToRad(this.#options.maxClimbAngle));
    if (this.#groundNormal.y < limit) this.#groundNormal.copy(UP);
  }

  /**
   * The most wall-like contact from the last sweep.
   *
   * "Wall-like" is `|normal.y| < 0.55`, i.e. steeper than about 57 degrees —
   * past the climb limit, so anything that qualifies is a surface the character
   * cannot walk up and must slide along instead.
   */
  #findWallNormal(): THREE.Vector3 | null {
    const count = this.#controller.numComputedCollisions();
    let best: THREE.Vector3 | null = null;
    let flattest = 0.55;
    for (let i = 0; i < count; i++) {
      const collision = this.#controller.computedCollision(i);
      if (collision === null) continue;
      const normal = collision.normal1;
      const vertical = Math.abs(normal.y);
      if (vertical >= flattest) continue;
      flattest = vertical;
      best = new THREE.Vector3(normal.x, normal.y, normal.z);
    }
    return best;
  }

  #result(
    landed: boolean,
    jumped: boolean,
    motionRatio: number,
    blocked: boolean,
    wallNormal: THREE.Vector3 | null = null,
  ): MoveResult {
    return {
      position: this.#position,
      grounded: this.#grounded,
      landed,
      jumped,
      groundNormal: this.#groundNormal,
      slope: this.#slope,
      verticalVelocity: this.#verticalVelocity,
      motionRatio,
      blocked,
      wallNormal,
    };
  }
}
