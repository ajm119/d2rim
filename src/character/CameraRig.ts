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
 * out at a limited rate, so brushing past a tree does not fire the camera
 * backwards.
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
  /** Arm length in metres. Default 3.9. */
  readonly distance?: number;
  /** Pivot height above the feet, metres. Default 1.45 (shoulder). */
  readonly pivotHeight?: number;
  /** Lateral shoulder offset, metres. Positive is the character's right. */
  readonly shoulder?: number;
  /** Sphere-cast radius for the arm, metres. Default 0.24. */
  readonly probeRadius?: number;
  /** Follow damping. Higher is stiffer. Default 14. */
  readonly followDamping?: number;
  /** Metres the pivot leads the character at full speed. Default 0.55. */
  readonly lookAhead?: number;
  /** Seconds the third/first transition takes. Default 0.38. */
  readonly transitionTime?: number;
  /** Rate the arm is allowed to extend back out, m/s. Default 5. */
  readonly extendRate?: number;
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
  followDamping: 14,
  lookAhead: 0.55,
  transitionTime: 0.38,
  extendRate: 5,
  enabled: true,
} satisfies Required<CameraRigOptions>;

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
  #initialised = false;
  #toggleFrame = -1;

  constructor(options: CameraRigOptions = {}) {
    this.#options = { ...DEFAULTS, ...options };
    this.#mode = this.#options.mode;
    this.#blend = this.#mode === 'first' ? 1 : 0;
    this.#arm = this.#options.distance;
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

  setMode(mode: CameraMode): void {
    if (this.#mode === mode) return;
    this.#mode = mode;
    this.#ctx?.events.emit('camera:mode', { mode });
  }

  toggleMode(): void {
    this.setMode(this.#mode === 'third' ? 'first' : 'third');
  }

  init(ctx: GameContext): void {
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
    this.#pivot.set(position.x, position.y + this.#options.pivotHeight, position.z);

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

    if (!this.#initialised) {
      this.#smoothPivot.copy(this.#pivot);
      this.#initialised = true;
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

    this.#desired.copy(origin).addScaledVector(this.#forward, -options.distance);

    let wanted = options.distance;
    const physics = this.#physics;
    if (physics !== null && physics.ready) {
      const direction = this.#armDir.copy(this.#desired).sub(origin);
      const length = direction.length();
      if (length > 1e-4) {
        direction.multiplyScalar(1 / length);
        const hit = physics.sphereCast(origin, direction, options.probeRadius, length, {
          layers: layerMask(CollisionLayer.Camera),
          solidOnly: true,
        });
        if (hit !== null) wanted = Math.max(0.25, hit.distance - 0.06);
      }
    }

    // Asymmetric: snap in, ease out.
    this.#arm =
      wanted < this.#arm ? wanted : Math.min(wanted, this.#arm + options.extendRate * dt);
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
