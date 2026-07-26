/**
 * @module ai/EnemyBase
 *
 * Everything an enemy is that is not its behaviour tree: the model, the
 * capsule, the animation graph, the swept weapon hitbox, its vitals, and the
 * {@link Combatant} face it presents to the rest of combat.
 *
 * ### Retargeting is free, so use it
 *
 * The skeletons share the Barbarian's 41-joint rig exactly. That means the same
 * `AnimationGraph`, the same semantic clip map, the same stride measurement and
 * the same upper-body mask all work unchanged — an enemy is a `PlayerController`
 * with a behaviour tree instead of a keyboard. Nothing in this file is skeleton
 * specific; `enemies/Skeleton.ts` supplies stats and a tree.
 *
 * ### Two deliberate divergences from the player
 *
 * 1. **Capsule collision groups.** `CharacterController` builds its capsule in
 *    the `player` group, which by the layer table does not collide with itself.
 *    Enemies therefore pass through one another as far as Rapier is concerned,
 *    and are kept apart by steering separation instead. That is the right trade
 *    for a crowd of melee attackers: physical push-out between kinematic
 *    controllers produces jitter, and the player reads a stack of skeletons as
 *    a bug regardless of which system caused it. Fixing it "properly" would
 *    mean editing `src/physics`, which is not this module's to edit.
 * 2. **No weapon mesh.** The skeleton GLBs carry no weapon geometry at all
 *    (verified by decoding the GLB node list), so their blade is the forearm
 *    extended past the fist. Reach is a stat rather than a measurement.
 */

import * as THREE from 'three/webgpu';

import { AnimationGraph } from '../character/AnimationGraph';
import type { Combatant, IncomingHit } from '../combat/Combatant';
import type { CombatSystem } from '../combat/CombatSystem';
import {
  resolveAttack,
  type AttackOutcome,
  type DefenseStats,
  type MoveModifiers,
  type OffenseStats,
} from '../combat/DamageModel';
import { WeaponHitbox, resolveWeaponAnchor } from '../combat/Hitbox';
import { Vitals } from '../combat/Vitals';
import type { GameContext } from '../core/types';
import { CharacterController } from '../physics/CharacterController';
import type { PhysicsWorld } from '../physics/PhysicsWorld';
import type { BehaviorTree } from './BehaviorTree';

/* -------------------------------------------------------------------------- */
/* Profile                                                                     */
/* -------------------------------------------------------------------------- */

/** One enemy attack. Same shape as a player move, minus the chain plumbing. */
export interface EnemyAttack {
  readonly id: string;
  readonly action: string;
  readonly speed: number;
  /** Damage window in normalised clip time. */
  readonly window: readonly [number, number];
  /** Normalised time at which the enemy may act again. */
  readonly recovery: number;
  /**
   * Seconds of motionless, glowing wind-up before the clip starts.
   *
   * This is the number the whole fight is balanced around. It is the window in
   * which the player sees the attack coming and can back out of range, and if
   * it is too short the enemy is not difficult, it is unfair.
   */
  readonly telegraph: number;
  readonly modifiers: MoveModifiers;
}

export interface EnemyPerception {
  /** How far the enemy can see, metres. */
  readonly visionRange: number;
  /** Half-angle of the vision cone, radians. */
  readonly visionHalfAngle: number;
  /** Radius within which the enemy notices the player regardless of facing. */
  readonly hearingRange: number;
  /** Multiplier on the hearing radius while the player is sprinting or swinging. */
  readonly noiseMultiplier: number;
  /** Distance past `visionRange` at which an alerted enemy gives up. */
  readonly loseRange: number;
}

export interface EnemyProfile {
  readonly variant: string;
  /** Asset key of the GLB. */
  readonly asset: string;
  readonly maxHealth: number;
  /** Metres the model is scaled to, measured across its skinned meshes. */
  readonly height: number;
  readonly capsuleRadius: number;
  readonly walkSpeed: number;
  readonly chaseSpeed: number;
  /** Distance from the enemy's centre at which it will commit to an attack. */
  readonly attackRange: number;
  /** Reach of the implied weapon past the fist, metres. */
  readonly reach: number;
  readonly offense: OffenseStats;
  readonly defense: Omit<DefenseStats, 'maxHealth'>;
  readonly perception: EnemyPerception;
  readonly attacks: readonly EnemyAttack[];
  /** Seconds between attacks. */
  readonly attackCooldown: number;
  /** Seconds of stagger on a hit that beats poise. */
  readonly staggerTime: number;
  /** Tint applied while winding up. */
  readonly telegraphColor: number;
}

export interface EnemySpawnOptions {
  readonly profile: EnemyProfile;
  readonly root: THREE.Object3D;
  readonly clips: readonly THREE.AnimationClip[];
  readonly position: THREE.Vector3;
  readonly yaw?: number;
  readonly physics: PhysicsWorld;
  readonly combat: CombatSystem;
  readonly ctx: GameContext;
  readonly id: number;
  /** Other enemies, for separation steering. Live, not a snapshot. */
  readonly neighbours: () => readonly EnemyBase[];
}

/** The coarse state the enemy is in. Drives animation and the debug readout. */
export type EnemyState =
  | 'spawning'
  | 'idle'
  | 'patrol'
  | 'alert'
  | 'chase'
  | 'telegraph'
  | 'attack'
  | 'stagger'
  | 'dead';

/** What every behaviour-tree node on an enemy sees. */
export interface EnemyBlackboard {
  readonly self: EnemyBase;
  /** The player, when one is registered and alive. */
  readonly player: Combatant | null;
  /** Horizontal distance between capsule centres, metres. */
  readonly distance: number;
  /** Unit horizontal vector from the enemy toward the player. */
  readonly toPlayer: THREE.Vector3;
  /** Whether the player is currently perceived (cone, hearing, or memory). */
  readonly aware: boolean;
}

/* -------------------------------------------------------------------------- */
/* Enemy                                                                       */
/* -------------------------------------------------------------------------- */

const UP = new THREE.Vector3(0, 1, 0);

export abstract class EnemyBase implements Combatant {
  readonly id: number;
  readonly faction = 'enemy' as const;
  readonly label: string;
  readonly profile: EnemyProfile;
  readonly vitals: Vitals;

  /** The visible model. Public because {@link Combatant} requires it. */
  readonly object: THREE.Object3D;

  protected readonly ctx: GameContext;
  protected readonly combat: CombatSystem;
  protected readonly graph: AnimationGraph | null;
  protected readonly controller: CharacterController;
  protected readonly neighbours: () => readonly EnemyBase[];

  readonly #hitbox: WeaponHitbox | null;
  readonly #tintables: Array<{
    material: THREE.Material & { emissive?: THREE.Color; emissiveIntensity?: number };
    baseColor: THREE.Color;
    baseIntensity: number;
  }> = [];

  readonly #desired = new THREE.Vector3();
  readonly #knockback = new THREE.Vector3();
  readonly #velocity = new THREE.Vector3();
  readonly #scratch = new THREE.Vector3();
  readonly #scratchB = new THREE.Vector3();
  readonly #tintColor = new THREE.Color();
  readonly #telegraphCache = new THREE.Color();

  #tree: BehaviorTree<EnemyBlackboard> | null = null;
  #state: EnemyState = 'spawning';
  #yaw = 0;
  #renderYaw = 0;

  #spawnTimer = 0;
  #staggerTimer = 0;
  #deathTimer = 0;
  #awareTimer = 0;
  #aware = false;
  #attack: EnemyApproachedAttack | null = null;
  #tintAmount = 0;
  #tintApplied = 0;
  #flashAmount = 0;
  #flashDecay = 8;
  #sink = 0;
  #removed = false;
  #detachEvents: (() => void) | null = null;

  /** Approach slot: a stable angular offset so a pack rings the player. */
  #slotAngle = 0;

  constructor(options: EnemySpawnOptions) {
    this.id = options.id;
    this.profile = options.profile;
    this.label = `${options.profile.variant}#${options.id}`;
    this.ctx = options.ctx;
    this.combat = options.combat;
    this.object = options.root;
    this.neighbours = options.neighbours;
    this.vitals = new Vitals({
      health: { max: options.profile.maxHealth, regen: 0 },
      mana: { max: 0 },
      stamina: { max: 0 },
    });

    this.#prepareMaterials();

    this.graph =
      options.clips.length > 0
        ? new AnimationGraph(options.root, options.clips, { verbose: false })
        : null;

    this.controller = new CharacterController(options.physics, {
      radius: options.profile.capsuleRadius,
      height: options.profile.height,
      maxClimbAngle: 55,
    });
    this.controller.setPosition(options.position);
    this.#yaw = options.yaw ?? 0;
    this.#renderYaw = this.#yaw;
    this.#slotAngle = ((options.id * 137.508) % 360) * (Math.PI / 180);

    options.root.position.copy(options.position);
    options.root.rotation.set(0, this.#yaw, 0);

    const anchor = resolveWeaponAnchor(options.root, { reach: options.profile.reach });
    this.#hitbox = anchor === null ? null : new WeaponHitbox(anchor, { radius: 0.13 });
    if (anchor === null) {
      console.warn(`[ai] ${this.label} has no arm bones; it will never land a hit`);
    }

    if (this.graph !== null) {
      this.#detachEvents = this.graph.onAnimationEvent((event) => this.#onAnimationEvent(event));
    }
    this.#beginSpawn();
  }

  /* -- Combatant ---------------------------------------------------------- */

  get alive(): boolean {
    return this.vitals.alive;
  }

  get hitRadius(): number {
    return this.profile.capsuleRadius + 0.12;
  }

  get hitHeight(): number {
    return this.profile.height;
  }

  get defense(): DefenseStats {
    return { ...this.profile.defense, maxHealth: this.profile.maxHealth };
  }

  footPosition(out: THREE.Vector3): THREE.Vector3 {
    return out.copy(this.controller.position);
  }

  facing(out: THREE.Vector3): THREE.Vector3 {
    return out.set(Math.sin(this.#yaw), 0, Math.cos(this.#yaw));
  }

  receiveHit(hit: IncomingHit): AttackOutcome {
    const facing = this.#facingTowards(hit.point);
    const outcome = resolveAttack(hit.offense, this.defense, this.combat.rng, {
      move: hit.move,
      facing,
    });
    // The hit flash itself lives in `combat/Feedback`, which reacts to the
    // `combat:hit` event: one place owns "what an impact looks like".
    if (outcome.result === 'miss' || !this.alive || outcome.result === 'blocked') return outcome;

    const applied = this.vitals.applyDamage(outcome.total);
    if (outcome.knockback > 0) {
      this.#knockback.copy(hit.direction).setY(0);
      if (this.#knockback.lengthSq() < 1e-8) this.#knockback.set(0, 0, 1);
      this.#knockback.normalize().multiplyScalar(outcome.knockback);
    }

    if (applied.killed) {
      this.#die();
    } else if (outcome.staggered) {
      this.#stagger();
      // Being hit is information: a skeleton clubbed from behind knows where
      // the player is even if it never saw them.
      this.#aware = true;
      this.#awareTimer = 6;
    }
    return outcome;
  }

  /* -- public surface ----------------------------------------------------- */

  get state(): EnemyState {
    return this.#state;
  }

  get position(): THREE.Vector3 {
    return this.controller.position;
  }

  get yaw(): number {
    return this.#yaw;
  }

  get removable(): boolean {
    return this.#removed;
  }

  get aware(): boolean {
    return this.#aware;
  }

  get health(): number {
    return this.vitals.health.value;
  }

  get healthMax(): number {
    return this.vitals.health.max;
  }

  /**
   * Move the enemy instantly, clearing its knockback.
   *
   * Exists for encounter scripting and the drive harness: placing a skeleton at
   * a known offset from the player is the only way to make a combat test
   * deterministic without also scripting the pathfinding.
   */
  teleport(x: number, y: number, z: number): void {
    if (this.#state === 'dead') return;
    this.#scratch.set(x, y, z);
    this.controller.setPosition(this.#scratch);
    this.object.position.copy(this.#scratch);
    this.#knockback.set(0, 0, 0);
    this.#velocity.set(0, 0, 0);
  }

  /** Force the enemy to know where the player is. Encounter scripting. */
  alert(): void {
    this.#aware = true;
    this.#awareTimer = 6;
    if (this.#state === 'idle' || this.#state === 'patrol') this.setState('chase');
  }

  /** Skip the rise-from-the-ground animation. Used by tests and captures. */
  skipSpawn(): void {
    if (this.#state !== 'spawning') return;
    this.#spawnTimer = 0;
    this.graph?.cancelActions('full', 0.05);
    this.setState('idle');
  }

  /** Centre of the capsule in world space. The point AI reasons about. */
  centre(out: THREE.Vector3): THREE.Vector3 {
    return out.copy(this.controller.position).addScaledVector(UP, this.profile.height * 0.5);
  }

  /** Briefly tint the whole model. Used by combat feedback on impact. */
  flash(color: number, seconds: number): void {
    this.#tintColor.setHex(color);
    this.#flashAmount = 1;
    this.#flashDecay = 1 / Math.max(0.02, seconds);
  }

  /* -- lifecycle ---------------------------------------------------------- */

  /** Build the behaviour tree. Called once, lazily, on the first tick. */
  protected abstract buildTree(): BehaviorTree<EnemyBlackboard>;

  fixedUpdate(dt: number): void {
    this.vitals.update(dt);
    this.#updateTimers(dt);

    if (this.#state === 'dead') {
      this.#updateCorpse(dt);
      return;
    }

    const player = this.combat.self;
    this.#updatePerception(player, dt);

    if (this.#state === 'spawning') {
      this.#desired.set(0, 0, 0);
    } else if (this.#state === 'stagger') {
      this.#desired.set(0, 0, 0);
    } else {
      if (this.#tree === null) this.#tree = this.buildTree();
      this.#desired.set(0, 0, 0);
      this.#tree.tick(this.#blackboard(player), dt);
    }

    this.#applyKnockback(dt);
    const result = this.controller.move(this.#desired, dt);
    // Measured, not requested: an enemy pinned against the ruined wall must
    // stop its legs, exactly as the player's does.
    this.#velocity
      .copy(result.position)
      .sub(this.controller.previousPosition)
      .setY(0)
      .divideScalar(Math.max(1e-5, dt));
  }

  update(dt: number): void {
    this.controller.interpolatedPosition(this.ctx.engine.alpha, this.#scratch);
    this.object.position.set(this.#scratch.x, this.#scratch.y - this.#sink, this.#scratch.z);
    this.#renderYaw = approachAngle(this.#renderYaw, this.#yaw, dt * 14);
    this.object.rotation.set(0, this.#renderYaw, 0);

    const graph = this.graph;
    if (graph !== null) {
      // Local-space velocity for the blend space: +z forward, +x right.
      const forward = this.facing(this.#scratch);
      const right = this.#scratchB.set(forward.z, 0, -forward.x);
      const frozen = this.#state === 'dead' || this.#state === 'spawning';
      graph.setLocomotion(
        frozen
          ? { x: 0, z: 0 }
          : { x: right.dot(this.#velocity), z: forward.dot(this.#velocity) },
        true,
      );
      graph.update(dt);
    }
    this.#updateTint(dt);
  }

  /**
   * Sweep the enemy's weapon. Runs after the mixer, exactly as the player's
   * does, and for the same reason.
   */
  lateUpdate(): void {
    const hitbox = this.#hitbox;
    if (hitbox === null) return;
    const player = this.combat.self;
    const attack = this.#attack;
    // Track unconditionally: the sweep needs a truthful previous pose even on
    // the frames where nothing can be hit.
    const targets = player !== null && player.alive && attack !== null ? [player] : [];
    for (const contact of hitbox.track(targets)) {
      if (attack === null) break;
      this.combat.resolve(this, contact.target, {
        source: this,
        offense: this.profile.offense,
        move: { id: attack.definition.id, ...attack.definition.modifiers },
        point: contact.point,
        normal: contact.normal,
        direction: contact.travel,
      });
    }
  }

  dispose(): void {
    this.#detachEvents?.();
    this.#detachEvents = null;
    this.graph?.dispose();
    this.controller.dispose();
    this.object.removeFromParent();
    this.object.traverse((child) => {
      if (child instanceof THREE.Mesh) child.geometry?.dispose?.();
    });
    for (const entry of this.#tintables) entry.material.dispose();
  }

  /* -- behaviour-tree API -------------------------------------------------- */

  /** Steer toward a world point at `speed`. Called from tree actions. */
  moveToward(point: THREE.Vector3, speed: number): void {
    this.#desired.copy(point).sub(this.controller.position).setY(0);
    if (this.#desired.lengthSq() > 1e-8) this.#desired.normalize().multiplyScalar(speed);
    this.#desired.add(this.#separation());
    this.#faceDirection(this.#desired);
  }

  /** Stop moving, but keep facing `point`. */
  holdFacing(point: THREE.Vector3): void {
    this.#desired.set(0, 0, 0);
    this.#scratch.copy(point).sub(this.controller.position).setY(0);
    this.#faceDirection(this.#scratch);
  }

  /** Stand still and stop steering. */
  halt(): void {
    this.#desired.set(0, 0, 0);
  }

  setState(state: EnemyState): void {
    if (this.#state !== 'dead') this.#state = state;
  }

  /**
   * The ring position this enemy should approach from.
   *
   * Every enemy owns a fixed angular slot around its target, so a pack forms an
   * arc rather than a queue converging on one point. Cheap, stateless, and it
   * is the difference between "three skeletons" and "one skeleton with a
   * flickering z-fight".
   */
  approachPoint(target: THREE.Vector3, radius: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(
      target.x + Math.sin(this.#slotAngle) * radius,
      target.y,
      target.z + Math.cos(this.#slotAngle) * radius,
    );
  }

  /** Whether an attack is currently mid-swing (telegraph or clip). */
  get attacking(): boolean {
    return this.#attack !== null;
  }

  /**
   * Start an attack. Returns false when one is already running.
   *
   * The telegraph is implemented here rather than in the tree so that the tree
   * node stays a two-line "am I in range, swing" and the timing lives with the
   * animation it belongs to.
   */
  beginAttack(definition: EnemyAttack): boolean {
    if (this.#attack !== null || !this.alive) return false;
    this.#attack = { definition, phase: 'telegraph', timer: definition.telegraph };
    this.setState('telegraph');
    this.#hitbox?.beginSwing();
    return true;
  }

  /** Advance the attack. @returns true while it is still running. */
  advanceAttack(dt: number): boolean {
    const attack = this.#attack;
    if (attack === null) return false;

    if (attack.phase === 'telegraph') {
      attack.timer -= dt;
      this.#tintAmount = Math.min(
        1,
        1 - Math.max(0, attack.timer) / Math.max(0.01, attack.definition.telegraph),
      );
      if (attack.timer > 0) return true;
      const graph = this.graph;
      const handle = graph?.playAction(attack.definition.action, {
        layer: 'full',
        speed: attack.definition.speed,
        fadeIn: 0.07,
        events: [
          { name: 'hit.open', at: attack.definition.window[0] },
          { name: 'hit.close', at: attack.definition.window[1] },
        ],
      });
      attack.phase = 'swing';
      attack.timer =
        ((handle?.duration ?? 1) * attack.definition.recovery) /
        Math.max(0.05, attack.definition.speed);
      this.setState('attack');
      return true;
    }

    attack.timer -= dt;
    if (attack.timer > 0) return true;
    this.endAttack();
    return false;
  }

  endAttack(): void {
    this.#attack = null;
    this.#hitbox?.closeWindow();
    this.#tintAmount = 0;
    if (this.#state === 'attack' || this.#state === 'telegraph') this.setState('chase');
  }

  /* -- internals ----------------------------------------------------------- */

  #blackboard(player: Combatant | null): EnemyBlackboard {
    const toPlayer = new THREE.Vector3();
    let distance = Infinity;
    if (player !== null) {
      player.footPosition(toPlayer);
      toPlayer.sub(this.controller.position).setY(0);
      distance = toPlayer.length();
      if (distance > 1e-6) toPlayer.divideScalar(distance);
    }
    return { self: this, player, distance, toPlayer, aware: this.#aware };
  }

  /**
   * Vision cone plus hearing, with a memory timer.
   *
   * The memory is what stops an enemy oscillating between chase and idle every
   * time the player steps behind a rock: perception is instantaneous, awareness
   * decays. Line of sight is a real raycast against the world so the ruined
   * wall genuinely hides the player.
   */
  #updatePerception(player: Combatant | null, dt: number): void {
    if (player === null || !player.alive) {
      this.#aware = false;
      this.#awareTimer = 0;
      return;
    }
    const perception = this.profile.perception;
    player.footPosition(this.#scratch);
    this.#scratch.sub(this.controller.position);
    const distance = Math.hypot(this.#scratch.x, this.#scratch.z);

    let perceived = false;
    if (distance <= perception.hearingRange * this.#noiseFactor()) {
      perceived = true;
    } else if (distance <= perception.visionRange) {
      this.#scratchB.set(this.#scratch.x, 0, this.#scratch.z);
      if (this.#scratchB.lengthSq() > 1e-8) {
        this.#scratchB.normalize();
        const cone = this.facing(this.#scratch).dot(this.#scratchB);
        if (cone >= Math.cos(perception.visionHalfAngle)) perceived = this.#hasLineOfSight(player);
      }
    }

    if (perceived) {
      this.#aware = true;
      this.#awareTimer = 6;
    } else if (this.#aware) {
      this.#awareTimer -= dt;
      if (this.#awareTimer <= 0 || distance > perception.loseRange) this.#aware = false;
    }
  }

  #noiseFactor(): number {
    return this.combat.moveId !== null ? this.profile.perception.noiseMultiplier : 1;
  }

  #hasLineOfSight(player: Combatant): boolean {
    const physics = this.ctx.services.tryGet<PhysicsWorld>('physics.world') ?? null;
    if (physics === null) return true;
    this.centre(this.#scratch);
    player.footPosition(this.#scratchB);
    this.#scratchB.y += player.hitHeight * 0.55;
    this.#scratchB.sub(this.#scratch);
    const distance = this.#scratchB.length();
    if (distance < 1e-4) return true;
    this.#scratchB.divideScalar(distance);
    const hit = physics.raycast(this.#scratch, this.#scratchB, distance, {
      exclude: this.controller.collider,
    });
    return hit === null;
  }

  /**
   * Push away from neighbours that are too close.
   *
   * Inverse-distance weighted, capped, and *added* to the steering vector
   * rather than replacing it, so a skeleton being crowded still makes progress
   * toward the player instead of standing off politely.
   */
  #separation(): THREE.Vector3 {
    const out = this.#scratchB.set(0, 0, 0);
    const radius = this.profile.capsuleRadius * 2 + 0.35;
    for (const other of this.neighbours()) {
      if (other === this || !other.alive) continue;
      this.#scratch.copy(this.controller.position).sub(other.controller.position).setY(0);
      const distance = this.#scratch.length();
      if (distance >= radius || distance < 1e-4) continue;
      this.#scratch.divideScalar(distance).multiplyScalar((radius - distance) / radius);
      out.add(this.#scratch);
    }
    return out.multiplyScalar(this.profile.chaseSpeed * 0.9);
  }

  #faceDirection(direction: THREE.Vector3): void {
    if (direction.lengthSq() < 1e-8) return;
    this.#yaw = Math.atan2(direction.x, direction.z);
  }

  #applyKnockback(dt: number): void {
    if (this.#knockback.lengthSq() < 1e-6) return;
    this.#desired.add(this.#knockback);
    // 12 m/s² of drag: a shove that reads as an impact and is over in a beat.
    const decay = Math.max(0, 1 - 12 * dt);
    this.#knockback.multiplyScalar(decay);
    if (this.#knockback.lengthSq() < 0.01) this.#knockback.set(0, 0, 0);
  }

  #updateTimers(dt: number): void {
    if (this.#spawnTimer > 0) {
      this.#spawnTimer -= dt;
      if (this.#spawnTimer <= 0 && this.#state === 'spawning') this.setState('idle');
    }
    if (this.#staggerTimer > 0) {
      this.#staggerTimer -= dt;
      if (this.#staggerTimer <= 0 && this.#state === 'stagger') this.setState('chase');
    }
  }

  #updateCorpse(dt: number): void {
    this.#deathTimer += dt;
    // Lie still, then sink. Sinking rather than fading because the material
    // stack in this project is opaque by design and turning a skeleton
    // transparent for two seconds costs a whole extra render pass.
    if (this.#deathTimer > 3) {
      this.#sink = Math.min(1, (this.#deathTimer - 3) / 1.6) * 1.4;
      if (this.#sink >= 1.4) this.#removed = true;
    }
  }

  #beginSpawn(): void {
    const graph = this.graph;
    this.#state = 'spawning';
    this.#spawnTimer = 1.2;
    if (graph === null) return;
    // Act I flavour, and free: the rig is shared, so the skeletons' own
    // `Spawn_Ground` clip plays on the same graph the player uses.
    const clip = graph.clipNames.includes('Spawn_Ground')
      ? 'Spawn_Ground'
      : graph.clipNames.includes('Skeletons_Awaken_Floor')
        ? 'Skeletons_Awaken_Floor'
        : null;
    if (clip === null) {
      this.#spawnTimer = 0.1;
      return;
    }
    const handle = graph.playAction('spawn', { clip, layer: 'full', fadeIn: 0.01 });
    this.#spawnTimer = Math.max(0.3, handle.duration * 0.92);
  }

  #stagger(): void {
    this.endAttack();
    this.#tree?.reset();
    this.#staggerTimer = this.profile.staggerTime;
    this.setState('stagger');
    this.graph?.playAction('hit', { layer: 'upper', fadeIn: 0.03 });
  }

  #die(): void {
    this.endAttack();
    this.#state = 'dead';
    this.#deathTimer = 0;
    this.#desired.set(0, 0, 0);
    this.#knockback.set(0, 0, 0);
    this.graph?.cancelActions(undefined, 0.08);
    this.graph?.playAction('death', { layer: 'full', hold: true, fadeIn: 0.08 });
    // The capsule is deliberately *not* torn down here. It is in the `player`
    // collision group, which by the layer table does not collide with the
    // player or with other characters, so a corpse already blocks nothing —
    // and disposing it leaves a live `CharacterController` whose rigid body is
    // gone, so the next call to `setPosition` or `move` traps inside the Rapier
    // WASM with `RuntimeError: unreachable`. That is a crash, in the middle of
    // combat, triggered by anything that touches a corpse. The collider goes
    // away with the rest of the enemy in `dispose`, at cull time.
    this.ctx.events.emit('combat:death', {
      combatant: this.id,
      faction: 'enemy',
      label: this.label,
    });
  }

  #onAnimationEvent(event: { action: string; name: string }): void {
    const attack = this.#attack;
    if (attack === null || event.action !== attack.definition.action) return;
    if (event.name === 'hit.open') this.#hitbox?.openWindow();
    else if (event.name === 'hit.close') this.#hitbox?.closeWindow();
  }

  #facingTowards(point: THREE.Vector3): number {
    this.facing(this.#scratch);
    this.#scratchB.copy(point).sub(this.controller.position).setY(0);
    if (this.#scratchB.lengthSq() < 1e-8) return 1;
    return this.#scratch.dot(this.#scratchB.normalize());
  }

  /* -- materials ----------------------------------------------------------- */

  /**
   * Clone every material so this instance can be tinted independently.
   *
   * Four skeleton variants share materials across their instances, and a hit
   * flash on a shared material lights up every skeleton in the moor at once —
   * which looks like a bug in the renderer rather than a bug in combat, and
   * costs an afternoon to find.
   */
  #prepareMaterials(): void {
    this.object.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      child.castShadow = true;
      child.receiveShadow = true;
      child.frustumCulled = false;
      const source = child.material;
      const materials = Array.isArray(source) ? source : [source];
      const cloned = materials.map((material) => {
        const copy = material.clone() as THREE.Material & {
          emissive?: THREE.Color;
          emissiveIntensity?: number;
        };
        if (copy.emissive instanceof THREE.Color) {
          this.#tintables.push({
            material: copy,
            baseColor: copy.emissive.clone(),
            baseIntensity: copy.emissiveIntensity ?? 1,
          });
        }
        return copy;
      });
      child.material = Array.isArray(source) ? cloned : (cloned[0] ?? source);
    });
  }

  /**
   * Blend the telegraph glow and the hit flash into the emissive channel.
   *
   * One channel, two sources, flash wins: an enemy that is hit mid-wind-up
   * should read as *hit*, because that is the information the player needs.
   */
  #updateTint(dt: number): void {
    if (this.#flashAmount > 0) this.#flashAmount = Math.max(0, this.#flashAmount - this.#flashDecay * dt);
    const telegraph = this.#tintAmount;
    const flash = this.#flashAmount;
    if (telegraph <= 0 && flash <= 0 && this.#tintApplied === 0) return;

    const amount = Math.max(telegraph * 0.85, flash);
    this.#tintApplied = amount;
    const colour = flash >= telegraph * 0.85 ? this.#tintColor : this.#telegraphColour();
    for (const entry of this.#tintables) {
      const emissive = entry.material.emissive;
      if (emissive === undefined) continue;
      // Capped well short of a full white-out. The tint has to read as "that
      // one, right now" at a glance without erasing the silhouette — a fully
      // blown-out skeleton is less legible than a tinted one, not more.
      // Kept well short of a white-out. Emissive is *added* to the lit result
      // and then bloomed by the post stack, so radiance much above ~0.8 stops
      // reading as "that one, hit, now" and starts reading as a rendering
      // fault: the silhouette disappears exactly when the player needs it.
      emissive.copy(entry.baseColor).lerp(colour, Math.min(0.5, amount * 0.5));
      entry.material.emissiveIntensity = entry.baseIntensity + amount * 0.55;
    }
  }

  #telegraphColour(): THREE.Color {
    return this.#telegraphCache.setHex(this.profile.telegraphColor);
  }
}

interface EnemyApproachedAttack {
  readonly definition: EnemyAttack;
  phase: 'telegraph' | 'swing';
  timer: number;
}

/** Shortest-path angular approach, matching the player's turn behaviour. */
export function approachAngle(current: number, target: number, maxStep: number): number {
  let delta = target - current;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  if (Math.abs(delta) <= maxStep) return target;
  return current + Math.sign(delta) * maxStep;
}
