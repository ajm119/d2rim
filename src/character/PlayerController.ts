/**
 * @module character/PlayerController
 *
 * The Barbarian: input, momentum, and the glue between the capsule and the
 * animation graph.
 *
 * ### The feel this is aiming at
 *
 * Skyrim, not Quake. The character has mass. Direction changes cost time, a
 * standing start ramps through walk speed on its way to a run, and letting go
 * of the stick coasts rather than stops. Concretely:
 *
 * - Velocity is integrated toward a target with separate acceleration and
 *   deceleration rates, so stopping is slower than starting.
 * - There is a hard turn-rate on the *body*, not just on the camera: the model
 *   rotates toward its heading at a finite speed, so a 180 is a pivot rather
 *   than a teleport.
 * - Airborne input is heavily attenuated. Full air control is the single
 *   loudest tell of a twitch shooter.
 * - Everything the animation graph sees is *measured* — the actual delta the
 *   capsule achieved, divided by the timestep — not what the input asked for.
 *   Walk into a wall and the legs stop, because the character really did.
 *
 * ### Ownership
 *
 * This module owns the look angles (`yaw`/`pitch`) even though the camera is
 * what visibly uses them, because movement is camera-relative and the
 * dependency has to point one way. `CameraRig` reads them. That also means
 * mouse input is consumed here, in `fixedUpdate`, before anything reads it.
 *
 * Combat is deliberately *not* here. The action-input block below plays the
 * right clip on the right layer and nothing else — no damage, no hitboxes, no
 * cancel windows — and stands down entirely once a `combat` service registers
 * itself, so the combat module can take the input over without an edit here.
 */

import * as THREE from 'three/webgpu';

import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { CharacterController, type MoveResult } from '../physics/CharacterController';
import { PhysicsWorldKey, type PhysicsWorld } from '../physics/PhysicsWorld';
import type { BloodMoor } from '../scene/BloodMoor';
import { AssetManagerKey, type AssetManager } from '../assets/AssetManager';
import { AnimationGraph } from './AnimationGraph';

declare module '../core/EventBus' {
  interface GameEvents {
    'player:spawn': { position: THREE.Vector3 };
    'player:jump': { position: THREE.Vector3 };
    'player:land': { position: THREE.Vector3; impact: number };
    'player:footstep': { position: THREE.Vector3; speed: number };
    'player:stamina': { value: number; max: number };
  }
}

/** Service key for the player. Everything that needs "where is the hero". */
export const PlayerKey = serviceKey<PlayerController>('character.player');

export interface PlayerControllerOptions {
  /** Spawn point in world XZ. Y is resolved against the ground. */
  readonly spawn?: { x: number; z: number };
  /**
   * Capsule height in metres. Defaults to the *measured* height of the loaded
   * model's skinned meshes, which is the only number that keeps the collision
   * volume and the thing the player can see in agreement.
   */
  readonly height?: number;
  /**
   * Measure the model at load and hand the animation graph the blend and
   * cadence policy that matches the gait speeds below. Default true — see
   * {@link PlayerController}.
   *
   * This used to mean the opposite: derive the gait speeds *from* the clips.
   * See `#calibrate` for why that was backwards.
   */
  readonly calibrateToAnimation?: boolean;
  /** Default ground speed with no modifier held, m/s. Default 4.4. */
  readonly runSpeed?: number;
  /** Speed while `Block` is held, m/s. Default 1.25. */
  readonly walkSpeed?: number;
  /** Speed while `Sprint` is held and stamina remains, m/s. Default 6.0. */
  readonly sprintSpeed?: number;
  /** Backpedal speed as a fraction of the current gait. Default 0.55. */
  readonly backFactor?: number;
  /** Strafe speed as a fraction of the current gait. Default 0.82. */
  readonly strafeFactor?: number;
  /** Ground acceleration, m/s². Default 26. */
  readonly acceleration?: number;
  /** Ground deceleration, m/s². Default 18 — coasting, not braking. */
  readonly deceleration?: number;
  /** Airborne acceleration, m/s². Default 5. */
  readonly airAcceleration?: number;
  /** Body turn rate, radians per second. Default 11. */
  readonly turnRate?: number;
  /** Radians of look per pixel of mouse movement. Default 0.0022. */
  readonly mouseSensitivity?: number;
  /** Pitch clamp in degrees, `[down, up]`. Default `[-72, 68]`. */
  readonly pitchLimits?: readonly [number, number];
  readonly staminaMax?: number;
  /** Stamina per second while sprinting. Default 18. */
  readonly sprintDrain?: number;
  /** Stamina per second recovered. Default 22. */
  readonly staminaRegen?: number;
  /** Delay before regeneration resumes, seconds. Default 0.8. */
  readonly staminaRegenDelay?: number;
  /** Stamina spent on a jump. Default 12. */
  readonly jumpCost?: number;
}

const DEFAULTS = {
  spawn: { x: 1.1, z: 0.4 },
  height: 0,
  calibrateToAnimation: true,
  // Chosen for feel and then imposed on the animation, not read off it. See
  // `#calibrate`. 4.4 m/s is a committed third-person action-RPG jog; 6.0 is a
  // sprint that outruns a skeleton (chase 2.5–3.8) without turning the camera
  // into a blur; 1.25 is the shuffle while a block is up, deliberately slow
  // enough that holding block is a decision.
  runSpeed: 4.4,
  walkSpeed: 1.25,
  sprintSpeed: 6.0,
  backFactor: 0.55,
  strafeFactor: 0.82,
  acceleration: 26,
  deceleration: 18,
  airAcceleration: 5,
  turnRate: 11,
  mouseSensitivity: 0.0022,
  pitchLimits: [-72, 68] as readonly [number, number],
  staminaMax: 100,
  sprintDrain: 18,
  staminaRegen: 22,
  staminaRegenDelay: 0.8,
  jumpCost: 12,
} satisfies Required<PlayerControllerOptions>;

export class PlayerController implements GameModule {
  readonly name = 'character.player';

  readonly #options: Required<PlayerControllerOptions>;
  /** Gait speeds in force, after calibration. */
  readonly #gait = { walk: 0, run: 0, sprint: 0 };
  #height = 1.85;
  readonly #velocity = new THREE.Vector3();
  readonly #measured = new THREE.Vector3();
  readonly #wish = new THREE.Vector3();
  readonly #scratch = new THREE.Vector3();
  readonly #localVelocity = new THREE.Vector3();

  #ctx: GameContext | null = null;
  #physics: PhysicsWorld | null = null;
  #controller: CharacterController | null = null;
  #animation: AnimationGraph | null = null;
  #object: THREE.Object3D | null = null;

  #yaw = 0;
  #pitch = -0.06;
  #bodyYaw = 0;
  #renderYaw = 0;
  #stamina = DEFAULTS.staminaMax;
  #staminaIdle = 0;
  #sprinting = false;
  #grounded = false;
  #enabled = true;
  #blocking = false;
  #lookFrame = -1;
  #jumpFrame = -1;
  #fallStart = 0;
  #combatOwned = false;

  constructor(options: PlayerControllerOptions = {}) {
    this.#options = { ...DEFAULTS, ...options };
    this.#gait.walk = this.#options.walkSpeed;
    this.#gait.run = this.#options.runSpeed;
    this.#gait.sprint = this.#options.sprintSpeed;
  }

  /* -- public surface ----------------------------------------------------- */

  /** The visible character. Null until the model has loaded. */
  get object(): THREE.Object3D | null {
    return this.#object;
  }

  /** The animation graph. Combat drives this. Null until the model loaded. */
  get animation(): AnimationGraph | null {
    return this.#animation;
  }

  /** The capsule controller. Null until physics is up. */
  get controller(): CharacterController | null {
    return this.#controller;
  }

  /** Live feet position. Do not mutate; use {@link teleport}. */
  get position(): THREE.Vector3 {
    return this.#controller?.position ?? this.#scratch.set(0, 0, 0);
  }

  /** Measured world velocity, m/s. Reflects walls and slopes, not intent. */
  get velocity(): THREE.Vector3 {
    return this.#measured;
  }

  /** Planar ground speed, m/s. */
  get speed(): number {
    return Math.hypot(this.#measured.x, this.#measured.z);
  }

  get grounded(): boolean {
    return this.#grounded;
  }

  /**
   * Camera yaw in radians, in three.js camera convention: the look direction is
   * `(-sin(yaw), 0, -cos(yaw))`, so 0 looks down −Z.
   *
   * Note that this is the *opposite* convention to {@link bodyYaw}, and the
   * difference is not a mistake — a mesh at `rotation.y = θ` points its local +Z
   * at `(sin θ, 0, cos θ)`, while a camera at the same rotation looks the other
   * way. The two are related by exactly π, and {@link CAMERA_TO_BODY_YAW} is the
   * only place that conversion is written down.
   */
  get yaw(): number {
    return this.#yaw;
  }

  /** Camera pitch in radians, clamped. Positive looks up. */
  get pitch(): number {
    return this.#pitch;
  }

  /**
   * The direction the body is facing, radians, in *mesh* convention: the model's
   * forward is `(sin(bodyYaw), 0, cos(bodyYaw))`. Assigned straight to
   * `object.rotation.y`. Lags {@link yaw} while the character turns.
   */
  get bodyYaw(): number {
    return this.#bodyYaw;
  }

  /** Unit forward vector of the body, in world space. */
  forward(out = new THREE.Vector3()): THREE.Vector3 {
    return out.set(Math.sin(this.#bodyYaw), 0, Math.cos(this.#bodyYaw));
  }

  get stamina(): number {
    return this.#stamina;
  }

  get staminaMax(): number {
    return this.#options.staminaMax;
  }

  get sprinting(): boolean {
    return this.#sprinting;
  }

  /** Measured standing height of the visible character, metres. */
  get height(): number {
    return this.#height;
  }

  /** Eye height above the feet, metres. The camera anchors to it. */
  get eyeHeight(): number {
    return this.#height * 0.92;
  }

  /** The gait speeds in force, after calibration against the animation. */
  get gait(): Readonly<{ walk: number; run: number; sprint: number }> {
    return this.#gait;
  }

  /** Suspend movement and look without tearing anything down (menus, cutscenes). */
  setEnabled(enabled: boolean): void {
    this.#enabled = enabled;
    if (!enabled) {
      this.#velocity.set(0, 0, 0);
      this.#controller?.clearJump();
    }
  }

  /** Move the character instantly. Clears momentum. */
  teleport(x: number, y: number, z: number): void {
    this.#controller?.setPosition(this.#scratch.set(x, y, z));
    this.#velocity.set(0, 0, 0);
    this.#measured.set(0, 0, 0);
  }

  /**
   * Spend stamina. Returns false and spends nothing when there is not enough,
   * which is the contract combat wants for a dodge or a power attack.
   */
  consumeStamina(amount: number): boolean {
    if (amount <= 0) return true;
    if (this.#stamina < amount) return false;
    this.#stamina -= amount;
    this.#staminaIdle = 0;
    this.#ctx?.events.emit('player:stamina', {
      value: this.#stamina,
      max: this.#options.staminaMax,
    });
    return true;
  }

  /* -- lifecycle ---------------------------------------------------------- */

  async init(ctx: GameContext): Promise<void> {
    this.#ctx = ctx;
    ctx.services.register(PlayerKey, this);
    this.#combatOwned = ctx.services.has('combat');

    const physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey);
    if (physics === undefined || !physics.ready) {
      console.error('[PlayerController] no physics world; the Barbarian cannot be controlled');
      return;
    }
    this.#physics = physics;

    await this.#acquireModel(ctx);
    this.#calibrate();

    // The capsule is sized *after* the model is measured. A collision volume
    // that disagrees with the mesh is the difference between "I bumped into
    // that" and "I bumped into nothing", and it is invisible until someone
    // turns the debug draw on.
    this.#controller = new CharacterController(physics, {
      height: this.#height,
      radius: THREE.MathUtils.clamp(this.#height * 0.18, 0.16, 0.45),
      stepOffset: this.#height * 0.23,
    });
    this.#spawn(ctx);
  }

  fixedUpdate(ctx: GameContext, dt: number): void {
    const controller = this.#controller;
    if (controller === null) return;

    this.#consumeLook(ctx);

    if (!this.#enabled) {
      const idle = controller.move(this.#scratch.set(0, 0, 0), dt);
      this.#afterMove(ctx, idle, dt);
      return;
    }

    this.#updateIntent(ctx, dt);
    this.#updateStamina(dt);
    this.#consumeJump(ctx);

    const result = controller.move(this.#velocity, dt);
    this.#afterMove(ctx, result, dt);
  }

  update(ctx: GameContext, dt: number): void {
    const object = this.#object;
    const controller = this.#controller;
    if (object === null || controller === null) return;

    // Interpolated between fixed steps: the simulation is 60 Hz and the display
    // is not, and drawing the raw simulation position judders on every other
    // refresh rate.
    controller.interpolatedPosition(ctx.engine.alpha, this.#scratch);
    object.position.copy(this.#scratch);

    // Body yaw is smoothed again at render rate. The fixed-step turn is already
    // rate-limited, but at 60 Hz simulation and 144 Hz display the difference
    // between "rate-limited" and "smooth" is visible on a fast pivot.
    this.#renderYaw = approachAngle(this.#renderYaw, this.#bodyYaw, dt * 24);
    object.rotation.set(0, this.#renderYaw, 0);

    if (!this.#combatOwned && this.#enabled) this.#updateActionInput(ctx);
    this.#animation?.update(dt);
  }

  dispose(): void {
    this.#animation?.dispose();
    this.#animation = null;
    this.#controller?.dispose();
    this.#controller = null;
    this.#ctx?.services.unregister(PlayerKey);
    this.#ctx = null;
  }

  /* -- setup -------------------------------------------------------------- */

  /**
   * Take the Barbarian the scene already built, or load one.
   *
   * The scene's copy is preferred because it has been measured to 1.85 m,
   * weathered into the palette and hooked into the occlusion pass — rebuilding
   * all of that here would be a second, silently diverging art path.
   */
  async #acquireModel(ctx: GameContext): Promise<void> {
    const scene = ctx.engine.getModule<BloodMoor & GameModule>('scene.bloodMoor');
    let object = scene?.hero ?? null;
    let clips: readonly THREE.AnimationClip[] = scene?.heroClips ?? [];

    if (object === null) {
      const assets = ctx.services.tryGet<AssetManager>(AssetManagerKey);
      if (assets === undefined) {
        console.error('[PlayerController] no AssetManager and no scene hero; nothing to control');
        return;
      }
      try {
        const gltf = await assets.loadGLTF('character.barbarian');
        assets.pin('character.barbarian');
        object = gltf.scene;
        clips = gltf.animations;
        const measured = new THREE.Box3().setFromObject(object);
        const height = measured.max.y - measured.min.y;
        if (Number.isFinite(height) && height > 1e-3) object.scale.setScalar(1.85 / height);
        object.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            child.frustumCulled = false;
          }
        });
        ctx.scene.add(object);
      } catch (error) {
        console.error('[PlayerController] could not load the Barbarian:', error);
        return;
      }
    }

    this.#object = object;
    if (clips.length === 0) {
      console.warn('[PlayerController] the Barbarian has no clips; he will stand in bind pose');
      return;
    }
    this.#animation = new AnimationGraph(object, clips);
    this.#animation.onAnimationEvent((event) => {
      if (event.name !== 'footstep') return;
      ctx.events.emit('player:footstep', {
        position: this.position.clone(),
        speed: this.speed,
      });
    });
  }

  /**
   * Size the character and its gaits to the asset.
   *
   * Two measurements, both taken from the model rather than assumed:
   *
   * - **Height** comes from the skinned meshes only. The GLB parents a loose
   *   axe, a shield and a mug to the hand sockets, so a `Box3` over the whole
   *   scene — which is what the scene module uses to normalise the figure —
   *   measures the props, not the Barbarian.
   * - **Blend thresholds** are handed *to* the graph from the gait speeds, so
   *   the walk ring is fully in at the walk speed and the run ring at the run
   *   speed, whatever those speeds happen to be.
   *
   * ### Which way round this dependency goes
   *
   * It used to run the other way: the gait speeds were computed from the
   * measured stride (`walkStride * 1.55`, `runStride * 1.9`). The reasoning was
   * that clips are fixed and config numbers are not, so the clip should win.
   *
   * That produced walk 0.95, run 1.76, sprint 2.46 m/s — a sprint slower than a
   * human jog — because this rig's clips are authored tiny (its run cycle
   * carries the body 1.97 m at an authored 1.25 cycles/s, i.e. 2.46 m/s). The
   * game was asset-locked to a plod, and no amount of level design or camera
   * work fixes a character who cannot move.
   *
   * Game feel is the requirement; the animation is the material. So the speeds
   * are authored, and the graph is told to *time-scale* the clips to meet them:
   * at 4.4 m/s the run clip runs at 2.23 cycles per second against an authored
   * 1.25, i.e. 1.8× rate. A faster cadence is the correct and normal cost of
   * this trade — a character with this rig's short legs genuinely does take
   * more steps per metre — and the cadence cap below is what stops it becoming
   * a cartoon at the top of the range.
   *
   * Pass explicit speeds to override all of this.
   */
  #calibrate(): void {
    const graph = this.#animation;
    const object = this.#object;
    if (object !== null) this.#height = this.#measureHeight(object);
    if (this.#options.height > 0) this.#height = this.#options.height;

    if (graph === null || !this.#options.calibrateToAnimation) return;

    // The run *ring* reaches full weight at the speed the run clip natively
    // covers, not at the gait's top speed. Those are different numbers and
    // conflating them was costing plants: a ramp from 1.25 to 4.4 m/s left the
    // character in a walk-and-run blend for three quarters of his speed range,
    // and the two clips plant on different beats with different duty factors,
    // so the blended stance is smeared and neither clip's plant survives. Ramp
    // to the run clip's own 2.46 m/s instead and everything above it is one
    // clean pose, time-scaled.
    const runNatural = graph.naturalSpeedForState('run.forward');
    const runRing =
      runNatural > 0
        ? THREE.MathUtils.clamp(runNatural, this.#gait.walk * 1.5, this.#gait.run)
        : this.#gait.run;
    graph.setBlendParams({
      idleThreshold: Math.min(0.12, this.#gait.walk * 0.12),
      walkSpeed: this.#gait.walk,
      runSpeed: runRing,
    });

    const walkStride = graph.strideForState('walk.forward');
    const runStride = graph.strideForState('run.forward');
    const cadence = (speed: number, stride: number): string =>
      stride > 0 ? `${(speed / stride).toFixed(2)} c/s` : 'unmeasured';
    console.info(
      `[PlayerController] height ${this.#height.toFixed(2)} m; gaits walk ` +
        `${this.#gait.walk.toFixed(2)} / run ${this.#gait.run.toFixed(2)} / sprint ` +
        `${this.#gait.sprint.toFixed(2)} m/s against measured strides walk ` +
        `${walkStride.toFixed(2)} / run ${runStride.toFixed(2)} m per cycle -> cadence ` +
        `${cadence(this.#gait.walk, walkStride)} walking, ` +
        `${cadence(this.#gait.run, runStride)} running, ` +
        `${cadence(this.#gait.sprint, runStride)} sprinting`,
    );
  }

  /** Standing height of the skinned body, ignoring anything hung off a socket. */
  #measureHeight(object: THREE.Object3D): number {
    const bounds = new THREE.Box3();
    let found = false;
    object.updateMatrixWorld(true);
    object.traverse((child) => {
      if (!(child instanceof THREE.SkinnedMesh)) return;
      const geometry = child.geometry;
      if (geometry.boundingBox === null) geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      if (box === null) return;
      bounds.union(box.clone().applyMatrix4(child.matrixWorld));
      found = true;
    });
    if (!found) return 1.85;
    const base = object.getWorldPosition(new THREE.Vector3()).y;
    return THREE.MathUtils.clamp(bounds.max.y - base, 0.6, 3);
  }

  #spawn(ctx: GameContext): void {
    const controller = this.#controller;
    if (controller === null) return;
    const { x, z } = this.#options.spawn;
    // A few centimetres above the sampled ground, so the first `move` snaps
    // down onto the real collider rather than starting inside it. The sample
    // only works because `PhysicsWorld.syncQueries` has already been run: an
    // unstepped query pipeline reports empty space everywhere, and the
    // character spawns at y=0 and free-falls into the moor.
    const radius = controller.dimensions.radius;
    const spot = this.#physics?.findClearSpot(x, z, radius, this.#height) ?? null;
    if (spot === null) {
      const ground = this.#physics?.groundHeight(x, z);
      console.warn(
        `[PlayerController] no clear ground within 4 m of ${x}, ${z}; spawning anyway`,
      );
      controller.setPosition(this.#scratch.set(x, (ground ?? 2) + 0.05, z));
    } else {
      if (Math.hypot(spot.x - x, spot.z - z) > 0.05) {
        console.info(
          `[PlayerController] spawn point ${x}, ${z} is occupied; moved to ` +
            `${spot.x.toFixed(2)}, ${spot.z.toFixed(2)}`,
        );
      }
      controller.setPosition(this.#scratch.set(spot.x, spot.y + 0.05, spot.z));
    }
    this.#fallStart = controller.position.y;

    // Facing the campfire, which is where the composition wants him looking and
    // therefore where a player's first frame should point. The camera starts
    // behind him, which is a half turn away in its own convention.
    this.#bodyYaw = Math.atan2(3.1 - x, -5.2 - z);
    this.#renderYaw = this.#bodyYaw;
    this.#yaw = wrapAngle(this.#bodyYaw + CAMERA_TO_BODY_YAW);
    this.#object?.position.copy(controller.position);
    this.#object?.rotation.set(0, this.#renderYaw, 0);
    ctx.events.emit('player:spawn', { position: controller.position.clone() });
    console.info(
      `[PlayerController] spawned at ${x.toFixed(2)}, ${controller.position.y.toFixed(2)}, ` +
        `${z.toFixed(2)}`,
    );
  }

  /* -- per-step ----------------------------------------------------------- */

  /**
   * Fold this frame's mouse movement into the look angles.
   *
   * Latched by frame number: `fixedUpdate` can run several times per frame and
   * the mouse delta is per *frame*, so consuming it once per step would make
   * look sensitivity depend on the display rate.
   */
  #consumeLook(ctx: GameContext): void {
    if (ctx.time.frame === this.#lookFrame) return;
    this.#lookFrame = ctx.time.frame;
    if (!this.#enabled) return;

    const delta = ctx.input.mouseDelta;
    const sensitivity = this.#options.mouseSensitivity;
    this.#yaw -= delta.x * sensitivity;
    this.#pitch -= delta.y * sensitivity;
    const limits = this.#options.pitchLimits;
    this.#pitch = THREE.MathUtils.clamp(
      this.#pitch,
      THREE.MathUtils.degToRad(limits[0]),
      THREE.MathUtils.degToRad(limits[1]),
    );
    // Keeping yaw bounded matters more than it looks: unbounded, a long session
    // accumulates enough magnitude that float precision starts to quantise the
    // camera in visible steps.
    this.#yaw = wrapAngle(this.#yaw);
  }

  #updateIntent(ctx: GameContext, dt: number): void {
    const move = ctx.input.moveVector();
    const blocking = ctx.input.isDown('Block');
    this.#blocking = blocking;

    const wantsSprint = ctx.input.isDown('Sprint') && move.y > 0.1 && !blocking;
    this.#sprinting = wantsSprint && this.#stamina > 1;

    const gait = blocking
      ? this.#gait.walk
      : this.#sprinting
        ? this.#gait.sprint
        : this.#gait.run;

    // Camera-relative: forward is where the player is looking, flattened.
    const sin = Math.sin(this.#yaw);
    const cos = Math.cos(this.#yaw);
    const forwardX = -sin;
    const forwardZ = -cos;
    const rightX = cos;
    const rightZ = -sin;

    this.#wish.set(
      forwardX * move.y + rightX * move.x,
      0,
      forwardZ * move.y + rightZ * move.x,
    );

    // Directional speed limits are applied to the *intent*, in local terms,
    // before the vector goes back to world space: backpedalling and strafing
    // are slower than advancing, which is what makes a fight read as footwork
    // rather than as an omnidirectional hover.
    const directional =
      move.y < -0.05
        ? this.#options.backFactor
        : Math.abs(move.x) > Math.abs(move.y)
          ? this.#options.strafeFactor
          : 1;
    const target = this.#wish.lengthSq() > 1e-8 ? gait * directional : 0;
    if (target > 0) this.#wish.normalize().multiplyScalar(target);

    const grounded = this.#grounded;
    const rate = grounded
      ? target > 0
        ? this.#options.acceleration
        : this.#options.deceleration
      : this.#options.airAcceleration;

    // Integrate toward the target with a hard per-step cap. An exponential
    // approach would be smoother on paper and mushier in the hand: it never
    // quite arrives, so the top speed is never the number in the config.
    this.#scratch.subVectors(this.#wish, this.#velocity);
    this.#scratch.y = 0;
    const distance = this.#scratch.length();
    const step = rate * dt;
    if (distance > 1e-6) {
      this.#velocity.addScaledVector(this.#scratch, Math.min(1, step / distance));
    }

    // Body facing. Grounded and moving, the character turns toward its heading;
    // blocking or acting, it faces the camera so the player can aim a swing.
    if (this.#blocking || this.#animation?.isActionPlaying() === true) {
      const facing = wrapAngle(this.#yaw + CAMERA_TO_BODY_YAW);
      this.#bodyYaw = approachAngle(this.#bodyYaw, facing, this.#options.turnRate * dt);
    } else if (Math.hypot(this.#velocity.x, this.#velocity.z) > 0.35) {
      const heading = Math.atan2(this.#velocity.x, this.#velocity.z);
      this.#bodyYaw = approachAngle(this.#bodyYaw, heading, this.#options.turnRate * dt);
    }
  }

  #updateStamina(dt: number): void {
    const options = this.#options;
    if (this.#sprinting && this.speed > 0.5) {
      this.#stamina = Math.max(0, this.#stamina - options.sprintDrain * dt);
      this.#staminaIdle = 0;
      if (this.#stamina <= 0) this.#sprinting = false;
    } else {
      this.#staminaIdle += dt;
      if (this.#staminaIdle >= options.staminaRegenDelay) {
        this.#stamina = Math.min(options.staminaMax, this.#stamina + options.staminaRegen * dt);
      }
    }
  }

  #consumeJump(ctx: GameContext): void {
    if (ctx.time.frame === this.#jumpFrame) return;
    if (!ctx.input.wasPressed('Jump')) return;
    this.#jumpFrame = ctx.time.frame;
    if (this.#stamina < this.#options.jumpCost) return;
    this.#controller?.requestJump();
  }

  #afterMove(ctx: GameContext, result: MoveResult, dt: number): void {
    const controller = this.#controller;
    if (controller === null) return;

    // Measured, not requested. Everything downstream — the animation rate, the
    // camera look-ahead, the UI speedometer — reads the truth.
    this.#measured
      .subVectors(controller.position, controller.previousPosition)
      .multiplyScalar(1 / Math.max(dt, 1e-6));

    // Cancel the component of velocity that is pushing into a wall, so running
    // along one is a slide rather than a series of stop-start collisions.
    if (result.wallNormal !== null) {
      const into = this.#velocity.dot(result.wallNormal);
      if (into < 0) this.#velocity.addScaledVector(result.wallNormal, -into);
    }

    const wasGrounded = this.#grounded;
    this.#grounded = result.grounded;

    if (result.jumped) {
      this.consumeStamina(this.#options.jumpCost);
      this.#animation?.playAction('jump.start', { layer: 'full', fadeIn: 0.05, fadeOut: 0.1 });
      ctx.events.emit('player:jump', { position: controller.position.clone() });
    }
    if (!result.grounded && wasGrounded) this.#fallStart = controller.position.y;
    if (result.landed) {
      const impact = Math.max(0, this.#fallStart - controller.position.y);
      if (impact > 0.8) {
        this.#animation?.playAction('jump.land', { layer: 'full', fadeIn: 0.05, fadeOut: 0.14 });
      }
      ctx.events.emit('player:land', { position: controller.position.clone(), impact });
    }

    // Into body space for the blend space: x right, z forward.
    const sin = Math.sin(this.#bodyYaw);
    const cos = Math.cos(this.#bodyYaw);
    this.#localVelocity.set(
      this.#measured.x * cos - this.#measured.z * sin,
      0,
      this.#measured.x * sin + this.#measured.z * cos,
    );
    this.#animation?.setLocomotion(this.#localVelocity, result.grounded);
  }

  /**
   * Placeholder combat input.
   *
   * Present so the character is playable and so the upper-body layer is
   * exercised from day one; it plays clips and does nothing else. Registering a
   * `combat` service disables it wholesale — see the module docs.
   */
  #updateActionInput(ctx: GameContext): void {
    const animation = this.#animation;
    if (animation === null) return;

    if (ctx.input.wasPressed('Attack')) animation.playAction('attack');
    else if (ctx.input.wasPressed('HeavyAttack')) animation.playAction('attack.heavy');
    else if (ctx.input.wasPressed('Interact')) animation.playAction('interact');

    if (ctx.input.wasPressed('Block')) {
      animation.playAction('block', { loop: true, layer: 'upper', fadeIn: 0.12 });
    } else if (ctx.input.wasReleased('Block')) {
      for (const active of animation.activeActions) {
        if (active.action === 'block') active.cancel(0.16);
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Angle helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * The constant offset between the camera's yaw convention and a mesh's.
 *
 * A `PerspectiveCamera` at `rotation.y = θ` looks along `(-sin θ, 0, -cos θ)`;
 * a mesh at the same rotation points its forward (+Z, for this character set) at
 * `(sin θ, 0, cos θ)`. Getting this wrong is invisible in a static screenshot
 * and unmistakable the moment anything moves: the character runs backwards
 * through the world with the camera in his face.
 */
export const CAMERA_TO_BODY_YAW = Math.PI;

/** Wrap to `(-π, π]`. */
export function wrapAngle(angle: number): number {
  const wrapped = (angle + Math.PI) % (Math.PI * 2);
  return (wrapped < 0 ? wrapped + Math.PI * 2 : wrapped) - Math.PI;
}

/**
 * Step `from` toward `to` by at most `maxDelta`, taking the short way round.
 *
 * The short-way-round part is what stops a character who turns past ±180°
 * from spinning all the way back the other way — the single most common
 * rotation bug in third-person movement.
 */
export function approachAngle(from: number, to: number, maxDelta: number): number {
  const delta = wrapAngle(to - from);
  if (Math.abs(delta) <= maxDelta) return wrapAngle(to);
  return wrapAngle(from + Math.sign(delta) * maxDelta);
}
