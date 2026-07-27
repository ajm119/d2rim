/**
 * @module physics/WorldColliders
 *
 * Turns the Blood Moor into something solid.
 *
 * A separate module from {@link PhysicsWorld} because the two have genuinely
 * different lifetimes: the world exists before any scene does and outlives a
 * zone transition, while this is per-zone content. When Act I grows a second
 * zone, it gets its own instance of this and the world is untouched.
 *
 * It must be registered *after* the scene it reads, since the engine
 * initialises modules in registration order and prop placement happens inside
 * `BloodMoor.init`.
 */

import * as THREE from 'three/webgpu';

import type { GameContext, GameModule } from '../core/types';
import type { BloodMoor } from '../scene/BloodMoor';
import { PhysicsWorldKey, type PhysicsWorld, type SceneColliderReport } from './PhysicsWorld';

export interface WorldCollidersOptions {
  /** Name of the scene module to read. Default `scene.bloodMoor`. */
  readonly sceneModule?: string;
  /**
   * Terrain collider resolution. Defaults to the visual mesh's own segment
   * count, which is the only value that makes the collider surface and the
   * rendered surface the same surface.
   */
  readonly terrainSegments?: number;
}

export class WorldColliders implements GameModule {
  readonly name = 'physics.worldColliders';

  readonly #options: WorldCollidersOptions;
  #report: SceneColliderReport | null = null;

  constructor(options: WorldCollidersOptions = {}) {
    this.#options = options;
  }

  /** What the last build produced. Null until `init` has run. */
  get report(): SceneColliderReport | null {
    return this.#report;
  }

  init(ctx: GameContext): void {
    const physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey);
    if (physics === undefined || !physics.ready) {
      console.error('[WorldColliders] no physics world; the moor will not be solid');
      return;
    }

    const moduleName = this.#options.sceneModule ?? 'scene.bloodMoor';
    const scene = ctx.engine.getModule<BloodMoor & GameModule>(moduleName);
    if (scene === undefined) {
      console.error(`[WorldColliders] no "${moduleName}" module; nothing to build colliders from`);
      return;
    }

    physics.buildTerrain(
      scene.field,
      scene.terrainSize,
      this.#options.terrainSegments ?? scene.terrainSegments,
    );

    const root = ctx.scene.getObjectByName('BloodMoor');
    if (root instanceof THREE.Object3D) this.#report = physics.buildSceneColliders(root);

    // Nothing built above is visible to a raycast until the query pipeline is
    // rebuilt, and the very next module to initialise asks the world where the
    // ground is so it can put the player on it.
    physics.syncQueries();
  }
}
