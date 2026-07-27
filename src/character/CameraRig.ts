/**
 * @module character/CameraRig
 *
 * Third-person spring arm with a first-person mode, and a smooth blend between
 * them.
 *
 * ### Why one rig and not two cameras
 *
 * The naive implementation is two camera poses and a switch. It cuts, and a cut
 * between two views of the same character is the most disorienting thing a
 * third/first-person game can do. Instead there is a single continuous
 * parameterisation: both modes produce an eye position, both share the *same*
 * look direction (derived from the player's yaw/pitch), and the mode blend
 * interpolates position only. Because the orientation never changes across the
 * transition, the blend reads as the camera flying to the character's head
 * rather than as a cut.
 *
 * ### The arm
 *
 * A ray would thread between a fence rail and the ground and put the lens in
 * the mud, so the arm is a **sphere cast** at the near-plane radius. Contact
 * pulls the camera in *immediately* — a damped pull-in means one frame with the
 * lens inside the ruined wall, which is one frame too many — and lets it back
 * out at a limited, damped rate, so brushing past a tree does not fire the
 * camera backwards and the recovery does not end in a jolt.
 *
 * The arm also has a **floor**, and the floor is the more important of the two
 * rules. A cold boot into the Rogue Encampment used to put the camera 0.38 m
 * behind the Barbarian, because one merged decorative mesh — six torch posts on
 * a 17.5 m ring, welded into a single `Mesh` — derived a single 35 m box
 * collider spanning the entire camp at chest height, and the arm dutifully
 * pulled in against it from the first frame. That defect is fixed at the
 * source, but the lesson generalises: an arm that any prop can collapse to
 * nothing will do it again somewhere else, and a camera the character
 * completely occludes has stopped being a camera. So the arm refuses to go
 * below {@link CameraRigOptions.minDistance} and accepts clipping a corner
 * instead, which is the smaller and the self-correcting failure. The probe also
 * refuses hits from characters, triggers and projectiles by name as well as by
 * layer mask — see {@link NON_BLOCKING_KINDS}.
 *
 * ### Order
 *
 * Runs in `lateUpdate`, after `PlayerController.update` has placed the model at
 * its interpolated position. Anything camera-shaped that runs in `update` is
 * one frame behind the thing it is following, which is exactly the wobble
 * players describe as "the camera feels laggy".
 */

import * as THREE from 'three/webgpu';

import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { findBone } from './BoneNames';
import { CollisionLayer, layerMask } from '../physics/Layers';
import { PhysicsWorldKey, type PhysicsWorld } from '../physics/PhysicsWorld';
import { PlayerKey, type PlayerController } from './PlayerController';

declare module '../core/EventBus' {
  interface GameEvents {
    'camera:mode': { mode: CameraMode };
  }
}

export const CameraRigKey = serviceKey<CameraRig>('character.cameraRig');

export type CameraMode = 'third' | 'first';

export interface CameraRigOptions {
  /** Mode at boot. Default `third`. */
  readonly mode?: CameraMode;
  /** Vertical FOV in degrees for gameplay. Default 55. */
  readonly fov?: number;
  /**
   * Arm length in metres. Defaults to 2.1 character-heights, so the framing is
   * the same whatever size the character turns out to be.
   */
  readonly distance?: number;
  /**
   * Pivot height above the feet, metres. Defaults to 0.86 of the player's
   * measured height — shoulder level on whatever the character actually is,
   * rather than a constant that only suits one model.
   */
  readonly pivotHeight?: number;
  /** Lateral shoulder offset, metres. Positive is the character's right. */
  readonly shoulder?: number;
  /** Sphere-cast radius for the arm, metres. Default 0.24. */
  readonly probeRadius?: number;
  /**
   * The shortest the arm is ever allowed to become, metres.
   *
   * Defaults to 0.34 of the rest length — about 1.2 m on the Barbarian, which
   * is the distance at which a shoulder still reads as a shoulder rather than
   * as a wall of pauldron.
   *
   * This is a *floor*, and it is deliberately preferred over correctness at the
   * lens: below roughly a metre the character occludes the frame completely, so
   * the camera has stopped being a camera. Clipping the corner of a tent for a
   * few frames is a far smaller failure than the player being unable to see the
   * game, and the failure is self-correcting the moment they walk on. Everything
   * within the floor is handled instead by hiding the head (see
   * {@link CameraRig.#applyHeadVisibility}) and by the near plane.
   */
  readonly minDistance?: number;
  /** Follow damping. Higher is stiffer. Default 14. */
  readonly followDamping?: number;
  /** Metres the pivot leads the character at full speed. Default 0.55. */
  readonly lookAhead?: number;
  /** Seconds the third/first transition takes. Default 0.38. */
  readonly transitionTime?: number;
  /** Rate the arm is allowed to extend back out, m/s. Default 5. */
  readonly extendRate?: number;
  /**
   * Damping on the arm extending back out. Higher settles sooner. Default 6.
   *
   * Works with {@link CameraRigOptions.extendRate}, not instead of it: the rate
   * caps how fast the recovery may start, the damping decides how it *stops*. A
   * pure rate limit runs at a constant speed and then halts dead at the rest
   * length, and that terminal discontinuity is visible — it reads as the camera
   * being yanked to a stop.
   */
  readonly extendDamping?: number;
  /** Take control of the camera at all. Default true. */
  readonly enabled?: boolean;
}

const DEFAULTS = {
  mode: 'third' as CameraMode,
  fov: 55,
  distance: 3.9,
  pivotHeight: 1.45,
  shoulder: 0.42,
  probeRadius: 0.24,
  minDistance: 1.33,
  followDamping: 14,
  lookAhead: 0.55,
  transitionTime: 0.38,
  extendRate: 5,
  extendDamping: 6,
  enabled: true,
} satisfies Required<CameraRigOptions>;

/**
 * The arm floor as a fraction of the rest length, when nobody has said.
 *
 * 0.34 rather than a constant, so a character of a different size is framed the
 * same way: the floor and the rest length are the same judgement about how much
 * of the frame the character may own, made at two distances.
 */
export const MIN_ARM_FRACTION = 0.34;

/**
 * Collider kinds the camera arm must never be stopped by.
 *
 * The layer mask already excludes all of these — `Player` and `Trigger` are not
 * members of `Camera`, so Rapier's symmetric test rejects them before the query
 * ever sees them. This is the second line, and it exists because the first line
 * is a bitmask: the project has already shipped one bug where an interaction
 * group was assembled as `(layers, layers)` and silently matched nothing, and
 * the failure mode in the other direction — the arm colliding with the body it
 * is anchored to — costs the player the entire frame. A named check that reads
 * as an English sentence is worth the four lines.
 */
const NON_BLOCKING_KINDS: ReadonlySet<string> = new Set(['character', 'trigger', 'projectile']);

/**
 * Frame-rate independent exponential smoothing.
 *
 * `lerp(a, b, k * dt)` is the version everybody writes first and it is wrong:
 * the effective smoothing changes with frame rate, so the camera behaves
 * differently at 30 and 144 fps. This is the closed form.
 */
export function damp(current: number, target: number, lambda: number, dt: number): number {
  return target + (current - target) * Math.exp(-lambda * dt);
}

/** Smootherstep. C² at both ends, which a mode transition needs to not tick. */
export function smootherstep(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * x * (x * (x * 6 - 15) + 10);
}

export class CameraRig implements GameModule {
  readonly name = 'character.cameraRig';

  readonly #options: Required<CameraRigOptions>;
  /** The caller's options, unresolved — `undefined` means "derive it". */
  readonly #given: CameraRigOptions;
  readonly #pivot = new THREE.Vector3();
  readonly #smoothPivot = new THREE.Vector3();
  readonly #lookAhead = new THREE.Vector3();
  readonly #desired = new THREE.Vector3();
  readonly #eyeThird = new THREE.Vector3();
  readonly #eyeFirst = new THREE.Vector3();
  readonly #forward = new THREE.Vector3();
  readonly #right = new THREE.Vector3();
  readonly #scratch = new THREE.Vector3();
  readonly #armDir = new THREE.Vector3();

  #player: PlayerController | null = null;
  #physics: PhysicsWorld | null = null;
  #ctx: GameContext | null = null;
  #head: THREE.Object3D | null = null;
  #headScale: THREE.Vector3 | null = null;

  #mode: CameraMode;
  #blend = 0;
  #arm: number;
  /** Resolved from the player's measured height unless explicitly configured. */
  #pivotHeight: number;
  #distance: number;
  #minDistance: number;
  #initialised = false;
  #toggleFrame = -1;
  /** What last shortened the arm, for diagnostics. Null when nothing did. */
  #obstruction: string | null = null;

  constructor(options: CameraRigOptions = {}) {
    this.#given = options;
    this.#options = { ...DEFAULTS, ...options };
    this.#mode = this.#options.mode;
    this.#blend = this.#mode === 'first' ? 1 : 0;
    this.#pivotHeight = this.#options.pivotHeight;
    this.#distance = this.#options.distance;
    this.#minDistance = Math.min(this.#options.minDistance, this.#distance);
    this.#arm = this.#distance;
  }

  /* -- public ------------------------------------------------------------- */

  get mode(): CameraMode {
    return this.#mode;
  }

  /** 0 = fully third person, 1 = fully first person. */
  get blend(): number {
    return this.#blend;
  }

  /** Current arm length after collision pull-in, metres. */
  get armLength(): number {
    return this.#arm;
  }

  /** The arm's uncollapsed length — what it returns to in the open, metres. */
  get restLength(): number {
    return this.#distance;
  }

  /** The floor {@link armLength} is never allowed below, metres. */
  get minLength(): number {
    return this.#minDistance;
  }

  /** Pivot height above the feet, metres. Resolved from the character. */
  get pivotHeight(): number {
    return this.#pivotHeight;
  }

  /**
   * Label of the collider currently shortening the arm, or `null`.
   *
   * Exposed because "the camera is too close" and "the camera is too close
   * *because of that tent*" are a diagnosis apart, and the second one is only
   * answerable from inside the rig — by the time a capture is on disk the
   * obstacle is whatever is filling the frame.
   */
  get obstruction(): string | null {
    return this.#obstruction;
  }

  setMode(mode: CameraMode): void {
    if (this.#mode === mode) return;
    this.#mode = mode;
    this.#ctx?.events.emit('camera:mode', { mode });
  }

  toggleMode(): void {
    this.setMode(this.#mode === 'third' ? 'first' : 'third');
  }

  init(ctx: GameContext): void {
    const options = this.#given;
    this.#ctx = ctx;
    ctx.services.register(CameraRigKey, this);
    this.#player = ctx.services.tryGet<PlayerController>(PlayerKey) ?? null;
    this.#physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey) ?? null;
    if (this.#player === null) {
      console.error('[CameraRig] no player service; the camera has nothing to follow');
      return;
    }
    if (this.#options.enabled) {
      ctx.camera.fov = this.#options.fov;
      ctx.camera.updateProjectionMatrix();
    }
    const object = this.#player.object;
    this.#head = object === null ? null : findBone(object, 'head');
    if (this.#head !== null) this.#headScale = this.#head.scale.clone();

    // Frame the character that actually loaded, not the one the defaults
    // assumed. The Barbarian measures well under the nominal 1.85 m, and a
    // pivot fixed at shoulder height for a 1.85 m figure floats above his head
    // and pushes him into the bottom of frame.
    const height = this.#player.height;
    if (options.pivotHeight === undefined) this.#pivotHeight = height * 0.86;
    if (options.distance === undefined) this.#distance = height * 2.1;
    this.#minDistance =
      options.minDistance === undefined
        ? this.#distance * MIN_ARM_FRACTION
        : Math.min(options.minDistance, this.#distance);
    this.#arm = this.#distance;
    console.info(
      `[CameraRig] third person: pivot ${this.#pivotHeight.toFixed(2)} m, ` +
        `arm ${this.#distance.toFixed(2)} m (floor ${this.#minDistance.toFixed(2)} m), ` +
        `fov ${this.#options.fov}`,
    );
  }

  lateUpdate(ctx: GameContext, dt: number): void {
    const player = this.#player;
    if (player === null || !this.#options.enabled) return;

    if (ctx.input.wasPressed('ToggleCamera') && ctx.time.frame !== this.#toggleFrame) {
      this.#toggleFrame = ctx.time.frame;
      this.toggleMode();
    }

    const target = this.#mode === 'first' ? 1 : 0;
    const rate = this.#options.transitionTime <= 0 ? 1 : dt / this.#options.transitionTime;
    this.#blend += Math.sign(target - this.#blend) * Math.min(rate, Math.abs(target - this.#blend));
    const eased = smootherstep(this.#blend);

    this.#updateBasis(player);
    this.#updatePivot(player, dt);
    this.#updateThirdPerson(dt);
    this.#updateFirstPerson(player);

    ctx.camera.position.lerpVectors(this.#eyeThird, this.#eyeFirst, eased);
    // Both modes look along the same vector, so there is nothing to interpolate
    // here — which is precisely what makes the transition read as a move rather
    // than as a cut.
    ctx.camera.lookAt(
      this.#scratch.copy(ctx.camera.position).addScaledVector(this.#forward, 10),
    );

    this.#applyHeadVisibility(eased);
  }

  dispose(): void {
    if (this.#head !== null && this.#headScale !== null) this.#head.scale.copy(this.#headScale);
    this.#ctx?.services.unregister(CameraRigKey);
    this.#ctx = null;
  }

  /* -- internals ---------------------------------------------------------- */

  #updateBasis(player: PlayerController): void {
    const yaw = player.yaw;
    const pitch = player.pitch;
    const cosPitch = Math.cos(pitch);
    this.#forward.set(-Math.sin(yaw) * cosPitch, Math.sin(pitch), -Math.cos(yaw) * cosPitch);
    this.#right.set(Math.cos(yaw), 0, -Math.sin(yaw));
  }

  #updatePivot(player: PlayerController, dt: number): void {
    const position = player.object?.position ?? player.position;
    this.#pivot.set(position.x, position.y + this.#pivotHeight, position.z);

    // Look-ahead. The pivot leads the character slightly in the direction of
    // travel, which opens up the space they are moving into instead of framing
    // the space they just left. Scaled by speed so it vanishes when standing.
    const velocity = player.velocity;
    const speed = Math.hypot(velocity.x, velocity.z);
    const amount = speed < 0.1 ? 0 : Math.min(1, speed / 5) * this.#options.lookAhead;
    this.#scratch.set(velocity.x, 0, velocity.z);
    if (this.#scratch.lengthSq() > 1e-8) this.#scratch.normalize().multiplyScalar(amount);
    else this.#scratch.set(0, 0, 0);
    this.#lookAhead.x = damp(this.#lookAhead.x, this.#scratch.x, 5, dt);
    this.#lookAhead.z = damp(this.#lookAhead.z, this.#scratch.z, 5, dt);
    this.#pivot.add(this.#lookAhead);

    // A teleport is not a movement, and following it is wrong twice over.
    //
    // Zone travel puts the player down somewhere else entirely, and a pivot that
    // damps toward the new position spends the next half-second in the air
    // between the two — sweeping the arm through the whole world on the way, so
    // the probe reports a contact every frame and the camera arrives collapsed.
    // The measured symptom on a Blood Moor arrival was an arm of 0.33 m on the
    // first frame recovering over the following second, which is precisely the
    // window a player is looking hardest at a new area.
    //
    // Detected by distance rather than by an event, so it also covers a debug
    // warp, a future respawn, and anything else that moves the character
    // without asking the camera first.
    const jumped = this.#initialised && this.#smoothPivot.distanceToSquared(this.#pivot) > 9;
    if (!this.#initialised || jumped) {
      this.#smoothPivot.copy(this.#pivot);
      this.#initialised = true;
      // Arriving at rest length rather than at whatever the last zone left
      // behind: the new area gets a clean frame, and the probe below is still
      // free to pull the arm in if the new area really is tight.
      this.#arm = this.#distance;
      return;
    }
    const lambda = this.#options.followDamping;
    this.#smoothPivot.set(
      damp(this.#smoothPivot.x, this.#pivot.x, lambda, dt),
      // Vertical is stiffer than horizontal: a camera that lags the character's
      // Y bobs against every step and every kerb, and the eye reads vertical
      // instability far more harshly than horizontal.
      damp(this.#smoothPivot.y, this.#pivot.y, lambda * 1.6, dt),
      damp(this.#smoothPivot.z, this.#pivot.z, lambda, dt),
    );
  }

  #updateThirdPerson(dt: number): void {
    const options = this.#options;
    const origin = this.#scratch
      .copy(this.#smoothPivot)
      .addScaledVector(this.#right, options.shoulder);

    this.#desired.copy(origin).addScaledVector(this.#forward, -this.#distance);

    let wanted = this.#distance;
    this.#obstruction = null;
    const physics = this.#physics;
    if (physics !== null && physics.ready) {
      const direction = this.#armDir.copy(this.#desired).sub(origin);
      const length = direction.length();
      if (length > 1e-4) {
        direction.multiplyScalar(1 / length);
        const hit = physics.sphereCast(origin, direction, options.probeRadius, length, {
          layers: layerMask(CollisionLayer.Camera),
          solidOnly: true,
          // The arm is anchored inside the body it is following, so the body's
          // own collider is *always* overlapping the sweep's start. The layer
          // mask already rejects it; naming it as well means the arm survives
          // a future collider being added to the player on the wrong layer.
          exclude: this.#player?.controller?.collider ?? null,
        });
        if (hit !== null && !NON_BLOCKING_KINDS.has(hit.record?.kind ?? 'prop')) {
          this.#obstruction = hit.record?.label ?? 'unknown';
          wanted = Math.max(this.#minDistance, hit.distance - 0.06);
        }
      }
    }

    // Asymmetric: snap in, ease out.
    //
    // Pulling *in* is instantaneous, and that is not laziness — a damped pull-in
    // spends its settling time with the lens inside the masonry. Letting the arm
    // back *out* is both rate-limited and damped: the rate stops the camera
    // being fired backwards when the player brushes past a tree, and the damping
    // is what removes the hard stop at the far end, which a rate limit on its
    // own leaves as a visible jolt exactly when the shot has finished recovering.
    if (wanted < this.#arm) {
      this.#arm = wanted;
    } else {
      const eased = damp(this.#arm, wanted, options.extendDamping, dt);
      this.#arm = Math.min(eased, this.#arm + options.extendRate * dt);
    }
    this.#eyeThird.copy(origin).addScaledVector(this.#forward, -this.#arm);
  }

  #updateFirstPerson(player: PlayerController): void {
    const head = this.#head;
    if (head !== null) {
      head.getWorldPosition(this.#eyeFirst);
      // Slightly forward of the skull's centre and a touch up, so the near
      // plane clears the face rather than clipping through the jaw.
      this.#eyeFirst.addScaledVector(this.#forward, 0.16).y += 0.06;
      return;
    }
    const position = player.object?.position ?? player.position;
    this.#eyeFirst
      .set(position.x, position.y + player.eyeHeight, position.z)
      .addScaledVector(this.#forward, 0.12);
  }

  /**
   * Hide the head once the camera is close enough to be inside it.
   *
   * Scaling the bone rather than hiding the whole character keeps the body and
   * arms visible in first person — looking down and seeing your own chest is
   * most of what makes a first-person mode feel embodied, and it costs nothing
   * because the mesh is already being skinned.
   */
  #applyHeadVisibility(blend: number): void {
    const head = this.#head;
    const original = this.#headScale;
    if (head === null || original === null) return;
    const hidden = blend > 0.82;
    const scale = hidden ? 1e-4 : 1;
    head.scale.set(original.x * scale, original.y * scale, original.z * scale);
  }
}
