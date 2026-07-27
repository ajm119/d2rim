/**
 * @module ai/EnemyDirector
 *
 * Loads the skeleton models, populates the Blood Moor, and owns the per-frame
 * loop for every live enemy.
 *
 * One module rather than one module per enemy, because the engine's registry is
 * keyed by name and a spawn would otherwise have to mutate it mid-frame. It
 * also gives the enemies a single place to be culled from: an enemy is removed
 * when its corpse has finished sinking, and removal is a splice plus a dispose,
 * with no dangling Rapier bodies left behind.
 *
 * ### Cloning
 *
 * `AssetManager` caches by semantic key, so every skeleton of a given variant
 * would otherwise share one skinned mesh and one skeleton — and animate as one
 * creature. `SkeletonUtils.clone` rebuilds the bone hierarchy and rebinds the
 * skinning, which is the only correct way to duplicate a rigged GLTF.
 */

import * as THREE from 'three/webgpu';
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js';

import { AssetManagerKey, type AssetManager } from '../assets/AssetManager';
import { CombatKey, type CombatSystem } from '../combat/CombatSystem';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { PhysicsWorldKey, type PhysicsWorld } from '../physics/PhysicsWorld';
import type { EnemyBase } from './EnemyBase';
import { SKELETON_PROFILES, SKELETON_VARIANTS, Skeleton } from './enemies/Skeleton';

declare module '../core/EventBus' {
  interface GameEvents {
    'ai:spawn': { id: number; variant: string; position: THREE.Vector3 };
    'ai:despawn': { id: number };
  }
}

export const EnemyDirectorKey = serviceKey<EnemyDirector>('ai.director');

/** Where a skeleton starts, in world XZ, relative to the player's spawn. */
export interface SpawnPoint {
  readonly variant: string;
  readonly x: number;
  readonly z: number;
  /** Patrol radius. 0 makes a sentry. */
  readonly patrol?: number;
}

/**
 * The Blood Moor encounter.
 *
 * Placed as a rough arc ahead of and around the player's spawn rather than a
 * ring: the player should be able to walk forward and pick a fight, or hold the
 * high ground and fight two at a time. Every one of these is far enough out
 * that nothing is already swinging when the game finishes loading.
 */
export const BLOOD_MOOR_SPAWNS: readonly SpawnPoint[] = [
  { variant: 'minion', x: 6.5, z: -7.5, patrol: 3 },
  { variant: 'minion', x: 9.5, z: -5.0, patrol: 3 },
  { variant: 'warrior', x: 12.5, z: -9.5, patrol: 2 },
  { variant: 'rogue', x: -7.0, z: -10.5, patrol: 5 },
  { variant: 'warrior', x: -11.0, z: 4.5, patrol: 2 },
  { variant: 'mage', x: 3.0, z: -15.0, patrol: 0 },
];

export interface EnemyDirectorOptions {
  readonly spawns?: readonly SpawnPoint[];
  /** Skip spawning entirely. Used by capture shots that want an empty moor. */
  readonly enabled?: boolean;
}

export class EnemyDirector implements GameModule {
  readonly name = 'ai.director';

  readonly #options: Required<EnemyDirectorOptions>;
  readonly #enemies: EnemyBase[] = [];
  readonly #neighbours = (): readonly EnemyBase[] => this.#enemies;
  /** One loaded model per variant, cloned per spawn. Populated by `init`. */
  readonly #sources = new Map<string, { scene: THREE.Object3D; clips: THREE.AnimationClip[] }>();

  #ctx: GameContext | null = null;
  #loaded = false;
  #placed = 0;
  #ready = false;

  constructor(options: EnemyDirectorOptions = {}) {
    this.#options = {
      spawns: options.spawns ?? BLOOD_MOOR_SPAWNS,
      enabled: options.enabled ?? true,
    };
  }

  get enemies(): readonly EnemyBase[] {
    return this.#enemies;
  }

  /**
   * Whether the encounter table has been fully placed.
   *
   * This used to mean "the model load resolved", which is not the same thing
   * and is the weaker of the two: on the boot path the load resolves *before*
   * `CombatSystem` exists, so a director could be `ready` with an empty moor.
   * It now means what a caller actually wants to know — the fight is there.
   */
  get ready(): boolean {
    return this.#ready;
  }

  /** Whether the skeleton models have finished downloading. */
  get loaded(): boolean {
    return this.#loaded;
  }

  /** Declared spawns that have not been placed yet. */
  get pending(): number {
    return Math.max(0, this.#options.spawns.length - this.#placed);
  }

  /** Live enemies, for the debug readout and the drive harness. */
  get alive(): number {
    return this.#enemies.filter((enemy) => enemy.alive).length;
  }

  /**
   * Download the models, then place the encounter.
   *
   * Split in two on purpose. The download needs only the {@link AssetManager};
   * placing needs `PhysicsWorld` *and* `CombatSystem`, and on the boot path the
   * combat system has not registered yet when the start zone loads. Fusing the
   * two meant the boot path silently produced an empty zone — the load
   * completed, `ready` went true, and nothing was there. Now the download runs
   * as early as it can and {@link populate} is retried until it takes.
   */
  async init(ctx: GameContext): Promise<void> {
    this.#ctx = ctx;
    ctx.services.register(EnemyDirectorKey, this);
    if (!this.#options.enabled) {
      this.#loaded = true;
      this.#ready = true;
      return;
    }

    const assets = ctx.services.tryGet<AssetManager>(AssetManagerKey);
    if (assets === undefined) {
      console.warn('[ai] no AssetManager; the zone will be empty');
      this.#loaded = true;
      this.#ready = true;
      return;
    }

    // Load each distinct variant once, then clone per spawn.
    const wanted = new Set(this.#options.spawns.map((spawn) => spawn.variant));
    for (const variant of wanted) {
      const key = SKELETON_PROFILES[variant]?.asset;
      if (key === undefined || !assets.has(key)) {
        console.warn(`[ai] unknown skeleton variant "${variant}"`);
        continue;
      }
      try {
        const gltf = await assets.loadGLTF(key);
        assets.pin(key);
        this.#sources.set(variant, { scene: gltf.scene, clips: [...gltf.animations] });
      } catch (error) {
        console.error(`[ai] could not load ${key}:`, error);
      }
    }
    // The director may have been disposed while the GLBs were in flight — a
    // player who walks straight back out of the moor does exactly that.
    if (this.#ctx === null) return;
    this.#loaded = true;
    this.populate();
  }

  /**
   * Place every declared spawn whose model has arrived.
   *
   * Idempotent and cheap to call every frame: it returns immediately until the
   * models are down and the services it needs exist, and once it has placed the
   * table it never places it again.
   *
   * @returns true once the encounter is fully placed.
   */
  populate(): boolean {
    if (this.#ready) return true;
    const ctx = this.#ctx;
    if (!this.#loaded || ctx === null) return false;
    // Both are required by `spawn`; without them it would refuse per enemy and
    // burn the whole table on the first frame.
    if (!ctx.services.has(PhysicsWorldKey) || !ctx.services.has(CombatKey)) return false;

    const spawns = this.#options.spawns;
    for (; this.#placed < spawns.length; this.#placed++) {
      const point = spawns[this.#placed];
      if (point === undefined) continue;
      const source = this.#sources.get(point.variant);
      if (source === undefined) continue;
      this.spawn(point, source.scene, source.clips);
    }
    this.#ready = true;
    console.info(
      `[ai] zone populated: ${this.#enemies.length} of ${spawns.length} skeletons ` +
        `(${SKELETON_VARIANTS.filter((v) => this.#sources.has(v)).join(', ')})`,
    );
    return true;
  }

  fixedUpdate(_ctx: GameContext, dt: number): void {
    for (const enemy of this.#enemies) enemy.fixedUpdate(dt);
  }

  update(_ctx: GameContext, dt: number): void {
    for (const enemy of this.#enemies) enemy.update(dt);
    this.#cull();
  }

  lateUpdate(): void {
    for (const enemy of this.#enemies) enemy.lateUpdate();
  }

  dispose(): void {
    for (const enemy of this.#enemies) this.#detach(enemy);
    this.#enemies.length = 0;
    this.#sources.clear();
    this.#ctx?.services.unregister(EnemyDirectorKey);
    this.#ctx = null;
  }

  /* -- spawning ----------------------------------------------------------- */

  /**
   * Place one skeleton.
   *
   * Public so a test or a capture shot can drop a single enemy in front of the
   * player without going through the encounter table.
   */
  spawn(
    point: SpawnPoint,
    source: THREE.Object3D,
    clips: readonly THREE.AnimationClip[],
  ): EnemyBase | null {
    const ctx = this.#ctx;
    if (ctx === null) return null;
    const profile = SKELETON_PROFILES[point.variant];
    if (profile === undefined) return null;

    const physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey);
    const combat = ctx.services.tryGet<CombatSystem>(CombatKey);
    if (physics === undefined || combat === undefined) {
      console.warn('[ai] physics or combat is missing; cannot spawn');
      return null;
    }

    const root = cloneSkinned(source);
    root.name = `enemy.skeleton.${point.variant}`;
    normaliseHeight(root, profile.height);

    // Stand it on the ground, and nowhere it would be born inside a rock.
    const clear = physics.findClearSpot(point.x, point.z, profile.capsuleRadius, profile.height, 5);
    const ground = clear ?? new THREE.Vector3(point.x, physics.groundHeight(point.x, point.z) ?? 0, point.z);

    const enemy = new Skeleton({
      profile,
      root,
      clips,
      position: ground,
      // Facing *away* from the player's spawn, not toward it. With a 14 m
      // vision cone, skeletons pointed at the origin all notice the hero the
      // instant the game finishes loading and the moor opens with six enemies
      // already charging. Backs turned, the player gets to choose the fight.
      yaw: Math.atan2(ground.x, ground.z),
      physics,
      combat,
      ctx,
      id: combat.targets.nextId(),
      neighbours: this.#neighbours,
      ...(point.patrol === undefined ? {} : { patrolRadius: point.patrol }),
    });

    ctx.scene.add(root);
    combat.targets.add(enemy);
    this.#enemies.push(enemy);
    ctx.events.emit('ai:spawn', {
      id: enemy.id,
      variant: point.variant,
      position: ground.clone(),
    });
    return enemy;
  }

  /* -- removal ------------------------------------------------------------- */

  #cull(): void {
    for (let i = this.#enemies.length - 1; i >= 0; i--) {
      const enemy = this.#enemies[i];
      if (enemy === undefined || !enemy.removable) continue;
      this.#enemies.splice(i, 1);
      this.#detach(enemy);
      this.#ctx?.events.emit('ai:despawn', { id: enemy.id });
    }
  }

  #detach(enemy: EnemyBase): void {
    const combat = this.#ctx?.services.tryGet<CombatSystem>(CombatKey);
    combat?.targets.remove(enemy);
    enemy.dispose();
  }
}

/**
 * Scale a loaded model so its *skinned body* is `target` metres tall.
 *
 * Measured over skinned meshes only, for the same reason `PlayerController`
 * does it: these GLBs park loose props on hand sockets, and a `Box3` over the
 * whole scene measures the props.
 */
export function normaliseHeight(root: THREE.Object3D, target: number): number {
  const box = new THREE.Box3();
  let found = false;
  root.updateMatrixWorld(true);
  root.traverse((child) => {
    if (!(child instanceof THREE.SkinnedMesh)) return;
    const geometry = child.geometry;
    if (geometry.boundingBox === null) geometry.computeBoundingBox();
    const local = geometry.boundingBox;
    if (local === null) return;
    box.union(local.clone().applyMatrix4(child.matrixWorld));
    found = true;
  });
  if (!found) return 1;
  const height = box.max.y - box.min.y;
  if (!Number.isFinite(height) || height <= 1e-3) return 1;
  const scale = target / height;
  root.scale.multiplyScalar(scale);
  return scale;
}
