/**
 * @module physics/PhysicsWorld
 *
 * The Rapier world, wrapped as a {@link GameModule}.
 *
 * ### Determinism
 *
 * Rapier is deterministic given identical inputs and an identical timestep, so
 * the world is stepped exactly once per {@link GameModule.fixedUpdate} at the
 * engine's constant 60 Hz slice and never from `update`. Nothing in this module
 * reads the wall clock. `Engine.stepFrames(n)` therefore reproduces the same
 * simulation byte-for-byte across runs, which is what lets a capture of the
 * character mid-stride be a regression test rather than a photograph.
 *
 * Registration order matters: this module must be added to the engine *before*
 * anything that resolves {@link PhysicsWorldKey} in `init`, and its
 * `fixedUpdate` then runs first in the slice. Characters move after the step,
 * against a query pipeline that was refreshed by it.
 *
 * ### Colliders
 *
 * The world is built from two sources, both static:
 *
 * - the terrain, as a Rapier heightfield sampled from the *same*
 *   {@link HeightSampler} the visual mesh is displaced by, at the same grid
 *   resolution. Sampling the same function at a different resolution is the
 *   classic way to end up with a character whose feet sink into a ridge.
 * - the props, derived by walking the scene graph and fitting an oriented box
 *   or an upright cylinder to each renderable's bounds. See
 *   {@link PhysicsWorld.buildSceneColliders}.
 *
 * Nothing here creates dynamic bodies yet. The pipeline supports them — the
 * step is a full `world.step()` — but Act I's props are scenery, and a barrel
 * that can be knocked over is a gameplay decision, not a physics one.
 */

import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three/webgpu';

import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import {
  COLLISION_GROUPS,
  CollisionLayer,
  interactionGroups,
  layerMask,
  type LayerMask,
} from './Layers';

/** Service key for the physics world. */
export const PhysicsWorldKey = serviceKey<PhysicsWorld>('physics.world');

/** The height function a terrain collider is built from. */
export interface HeightSampler {
  heightAt(x: number, z: number): number;
}

/** What kind of thing a registered collider stands for. */
export type ColliderKind = 'terrain' | 'prop' | 'character' | 'trigger' | 'projectile';

/** Bookkeeping for one collider the world owns. */
export interface ColliderRecord {
  readonly id: number;
  readonly kind: ColliderKind;
  /** Diagnostic label — usually the source `Object3D.name`. */
  readonly label: string;
  readonly collider: RAPIER.Collider;
  /** The rigid body the collider hangs off, when it has one. */
  readonly body: RAPIER.RigidBody | null;
  /** The scene object the collider was derived from, when there was one. */
  readonly source: THREE.Object3D | null;
}

/** A hit returned by {@link PhysicsWorld.raycast}. */
export interface RaycastHit {
  /** Distance along the ray direction, in metres. */
  readonly distance: number;
  readonly point: THREE.Vector3;
  /** World-space surface normal at the hit. Zero-length when unavailable. */
  readonly normal: THREE.Vector3;
  readonly collider: RAPIER.Collider;
  readonly record: ColliderRecord | null;
}

/** A hit returned by {@link PhysicsWorld.sphereCast}. */
export interface ShapeCastHit {
  /** Distance travelled along `direction` before contact, in metres. */
  readonly distance: number;
  /** World-space contact point on the obstacle. */
  readonly point: THREE.Vector3;
  readonly normal: THREE.Vector3;
  readonly collider: RAPIER.Collider;
  readonly record: ColliderRecord | null;
}

export interface QueryOptions {
  /** Layers the query is willing to hit. Defaults to terrain + props. */
  readonly layers?: LayerMask;
  /** Collider to ignore — almost always the caster's own. */
  readonly exclude?: RAPIER.Collider | null;
  /** Skip sensors (triggers). Default `true`. */
  readonly solidOnly?: boolean;
}

export interface PhysicsWorldOptions {
  /** Metres per second squared. Default `-24`: game gravity, not Earth's. */
  readonly gravity?: number;
  /** Draw the collider wireframe from boot. Default `false`. */
  readonly debugDraw?: boolean;
}

/** Tuning for {@link PhysicsWorld.buildSceneColliders}. */
export interface SceneColliderOptions {
  /** Objects whose name matches are skipped entirely. */
  readonly exclude?: RegExp;
  /** Bounds smaller than this in every axis are ignored (pebbles, embers). */
  readonly minExtent?: number;
  /** Bounds larger than this in any axis are ignored (ground planes, skirts). */
  readonly maxExtent?: number;
  /**
   * Above this height-to-footprint ratio an object is fitted with an upright
   * cylinder instead of a box. Tree trunks and fence posts are tall and thin
   * with a wide branch spread, and a box around one blocks a corridor of empty
   * air the player can see straight through.
   */
  readonly slenderRatio?: number;
  /** Cylinder radius as a fraction of the object's smaller horizontal half-extent. */
  readonly trunkRadiusFactor?: number;
}

const DEFAULT_SCENE_COLLIDERS: Required<SceneColliderOptions> = {
  // Emissive cards, fog volumes, the painted backdrop and the light rig helpers
  // are all renderables with bounds and none of them is a physical object.
  exclude: /flame|ember|scatter|glow|silhouette|fog|decal|rim\.|ground|barbarian|player/i,
  minExtent: 0.14,
  maxExtent: 90,
  // 1.6, not 2: a dead tree is 8 m of trunk inside a 4 m spread of branches, so
  // its bounding box has a ratio near 2 and a box collider would seal off a
  // 4-metre square of ground the player can see straight through.
  slenderRatio: 1.6,
  trunkRadiusFactor: 0.34,
};

/** Result summary from a collider build pass, for logging and tests. */
export interface SceneColliderReport {
  readonly boxes: number;
  readonly cylinders: number;
  readonly skipped: number;
}

/**
 * Choose between a box and an upright cylinder for a set of half-extents.
 *
 * Pure, and exported, because it is the one judgement call in the whole
 * collider-derivation pass and it is much easier to argue about against a table
 * of numbers than against a screenshot of a character stuck on nothing.
 */
export function fitShapeKind(
  half: { x: number; y: number; z: number },
  slenderRatio: number,
): 'box' | 'cylinder' {
  const footprint = Math.max(half.x, half.z);
  if (footprint <= 1e-4) return 'box';
  return half.y / footprint >= slenderRatio ? 'cylinder' : 'box';
}

export class PhysicsWorld implements GameModule {
  readonly name = 'physics.world';

  readonly #options: PhysicsWorldOptions;
  readonly #records = new Map<number, ColliderRecord>();

  #world: RAPIER.World | null = null;
  #ctx: GameContext | null = null;
  #nextId = 1;
  #debugDraw = false;
  #debugLines: THREE.LineSegments | null = null;
  #steps = 0;

  constructor(options: PhysicsWorldOptions = {}) {
    this.#options = options;
  }

  /**
   * The Rapier namespace, so callers can build `ColliderDesc`s without adding a
   * second import of the WASM module (which would boot a second copy).
   */
  get rapier(): typeof RAPIER {
    return RAPIER;
  }

  /** The live world. Throws before `init` has resolved. */
  get world(): RAPIER.World {
    if (this.#world === null) {
      throw new Error('[PhysicsWorld] the world does not exist until init() has resolved');
    }
    return this.#world;
  }

  /** Whether the world is built. Cheaper than catching from {@link world}. */
  get ready(): boolean {
    return this.#world !== null;
  }

  /** Simulation steps executed since boot. Diagnostics only. */
  get steps(): number {
    return this.#steps;
  }

  /** Every registered collider, in creation order. */
  get colliders(): readonly ColliderRecord[] {
    return Array.from(this.#records.values());
  }

  async init(ctx: GameContext): Promise<void> {
    this.#ctx = ctx;
    // Idempotent in rapier3d-compat: repeated calls resolve against the same
    // instantiation promise, so HMR does not re-inflate the WASM blob.
    await RAPIER.init();

    const gravity = this.#options.gravity ?? -24;
    this.#world = new RAPIER.World({ x: 0, y: gravity, z: 0 });
    // The character controller does its own sub-stepping through
    // `computeColliderMovement`, so a single solver iteration profile tuned for
    // stacked rigid bodies would be wasted here. Rapier's defaults are kept;
    // only the timestep is ours, and it is set per step from the engine slice.
    ctx.services.register(PhysicsWorldKey, this);

    if (this.#options.debugDraw === true) this.setDebugDraw(true);
  }

  fixedUpdate(_ctx: GameContext, fixedDt: number): void {
    const world = this.#world;
    if (world === null) return;
    // Assigning every step rather than once at boot: the engine's slice is
    // configurable, and a world whose timestep silently disagrees with the
    // caller's dt drifts in a way that is almost impossible to see.
    world.timestep = fixedDt;
    world.step();
    this.#steps++;
  }

  lateUpdate(): void {
    if (this.#debugDraw) this.#updateDebugLines();
  }

  dispose(): void {
    this.#records.clear();
    if (this.#debugLines !== null) {
      this.#debugLines.removeFromParent();
      this.#debugLines.geometry.dispose();
      (this.#debugLines.material as THREE.Material).dispose();
      this.#debugLines = null;
    }
    this.#world?.free();
    this.#world = null;
    this.#ctx = null;
  }

  /* -- registry ----------------------------------------------------------- */

  /**
   * Create a collider and register it.
   *
   * Everything that puts geometry into the world goes through here, so that a
   * raycast hit can always be turned back into "what did I actually hit" —
   * without which every physics bug is debugged by counting colliders.
   */
  addCollider(
    desc: RAPIER.ColliderDesc,
    meta: { kind: ColliderKind; label: string; source?: THREE.Object3D | null },
    body?: RAPIER.RigidBody,
  ): ColliderRecord {
    const collider = this.world.createCollider(desc, body);
    const record: ColliderRecord = {
      id: this.#nextId++,
      kind: meta.kind,
      label: meta.label,
      collider,
      body: body ?? null,
      source: meta.source ?? null,
    };
    this.#records.set(collider.handle, record);
    return record;
  }

  /** Look up the record for a Rapier collider handle. */
  recordFor(collider: RAPIER.Collider | null | undefined): ColliderRecord | null {
    if (collider === null || collider === undefined) return null;
    return this.#records.get(collider.handle) ?? null;
  }

  /** Remove a collider (and its record) from the world. */
  removeCollider(record: ColliderRecord): void {
    this.#records.delete(record.collider.handle);
    this.world.removeCollider(record.collider, false);
  }

  /* -- world construction ------------------------------------------------- */

  /**
   * Build the terrain collider from a height function.
   *
   * Rapier's heightfield is a `(rows+1) x (cols+1)` grid stored with the row
   * index varying fastest. Empirically — and this is worth stating because
   * getting it wrong produces terrain rotated 90 degrees from the mesh, which
   * looks *almost* right — the fast index runs along **+z** and the slow index
   * along **+x**, both centred on the collider's origin:
   *
   * ```
   * x = (j / cols - 0.5) * size    z = (i / rows - 0.5) * size
   * heights[i + j * (rows + 1)] = heightAt(x, z)
   * ```
   *
   * `segments` should match the visual mesh's segment count. The mesh
   * interpolates linearly between the same grid points, so at equal resolution
   * the collider surface and the rendered surface are the same surface.
   */
  buildTerrain(field: HeightSampler, size: number, segments: number): ColliderRecord {
    const n = Math.max(2, Math.round(segments));
    const heights = new Float32Array((n + 1) * (n + 1));
    for (let j = 0; j <= n; j++) {
      const x = (j / n - 0.5) * size;
      for (let i = 0; i <= n; i++) {
        const z = (i / n - 0.5) * size;
        heights[i + j * (n + 1)] = field.heightAt(x, z);
      }
    }

    const desc = RAPIER.ColliderDesc.heightfield(n, n, heights, { x: size, y: 1, z: size })
      .setCollisionGroups(COLLISION_GROUPS.terrain)
      // Friction is irrelevant to a kinematic character but matters the moment
      // a dropped item or a ragdoll lands on the moor.
      .setFriction(0.9)
      .setRestitution(0);

    return this.addCollider(desc, { kind: 'terrain', label: 'terrain.heightfield' });
  }

  /**
   * Derive static colliders for every renderable under `root`.
   *
   * The scene is authored for the camera, not for the simulation: it has no
   * collision meshes and no physics metadata, and adding some would mean
   * touching a frozen art pipeline. So the shapes are fitted to bounds instead.
   * That is a deliberate approximation, and the trade is stated plainly: a box
   * around a boulder is slightly too generous at the corners, which the player
   * reads as "I brushed the rock", where a triangle mesh of every prop would be
   * exact, an order of magnitude more expensive to query, and would catch the
   * capsule on every authored crack.
   *
   * `InstancedMesh` is expanded per instance — the scattered rocks and fences
   * are all instanced, and a single collider around the whole batch would seal
   * off half the moor.
   *
   * Opt out per object with `object.userData.noCollide = true`.
   */
  buildSceneColliders(
    root: THREE.Object3D,
    options: SceneColliderOptions = {},
  ): SceneColliderReport {
    const config = { ...DEFAULT_SCENE_COLLIDERS, ...options };
    let boxes = 0;
    let cylinders = 0;
    let skipped = 0;

    const bounds = new THREE.Box3();
    const center = new THREE.Vector3();
    const half = new THREE.Vector3();
    const world = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const instanceMatrix = new THREE.Matrix4();

    root.updateMatrixWorld(true);

    const emit = (mesh: THREE.Mesh, matrix: THREE.Matrix4, label: string): void => {
      const geometry = mesh.geometry;
      if (geometry.boundingBox === null) geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      if (box === null) {
        skipped++;
        return;
      }
      bounds.copy(box);
      bounds.getCenter(center);
      bounds.getSize(half).multiplyScalar(0.5);

      matrix.decompose(position, quaternion, scale);
      const hx = Math.abs(half.x * scale.x);
      const hy = Math.abs(half.y * scale.y);
      const hz = Math.abs(half.z * scale.z);

      const largest = Math.max(hx, hy, hz) * 2;
      if (largest < config.minExtent || largest > config.maxExtent) {
        skipped++;
        return;
      }

      // The geometry's bounding-box centre is rarely the origin (props are
      // authored standing on y=0), so the collider has to be offset by the
      // rotated, scaled centre or every prop sinks by half its height.
      const offset = center.clone().multiply(scale).applyQuaternion(quaternion);
      const cx = position.x + offset.x;
      const cy = position.y + offset.y;
      const cz = position.z + offset.z;

      const kind = fitShapeKind({ x: hx, y: hy, z: hz }, config.slenderRatio);
      let desc: RAPIER.ColliderDesc;
      if (kind === 'cylinder') {
        const radius = THREE.MathUtils.clamp(
          Math.min(hx, hz) * config.trunkRadiusFactor,
          0.12,
          0.7,
        );
        desc = RAPIER.ColliderDesc.cylinder(hy, radius);
        cylinders++;
      } else {
        desc = RAPIER.ColliderDesc.cuboid(hx, hy, hz);
        boxes++;
      }

      desc
        .setTranslation(cx, cy, cz)
        // A cylinder stays upright regardless of the source object's yaw, so
        // only the box takes the rotation. Tilting a trunk collider by a
        // scatter-authored lean would open a gap at the base.
        .setRotation(
          kind === 'cylinder'
            ? { x: 0, y: 0, z: 0, w: 1 }
            : { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w },
        )
        .setCollisionGroups(COLLISION_GROUPS.prop)
        .setFriction(0.8)
        .setRestitution(0);

      this.addCollider(desc, { kind: 'prop', label, source: mesh });
    };

    root.traverse((object) => {
      if (object.userData['noCollide'] === true) return;
      if (config.exclude.test(object.name)) return;
      // Skinned meshes are characters. They get capsules, not bounds boxes.
      if (object instanceof THREE.SkinnedMesh) return;

      if (object instanceof THREE.InstancedMesh) {
        for (let i = 0; i < object.count; i++) {
          object.getMatrixAt(i, instanceMatrix);
          world.multiplyMatrices(object.matrixWorld, instanceMatrix);
          emit(object, world, `${object.name}#${i}`);
        }
        return;
      }

      if (object instanceof THREE.Mesh) {
        emit(object, object.matrixWorld, object.name || 'mesh');
      }
    });

    console.info(
      `[PhysicsWorld] scene colliders: ${boxes} boxes, ${cylinders} cylinders, ${skipped} skipped`,
    );
    return { boxes, cylinders, skipped };
  }

  /* -- queries ------------------------------------------------------------ */

  /**
   * Cast a ray. `direction` need not be normalised; `maxDistance` is measured
   * in units of `direction`'s length, so pass a unit vector unless you mean it.
   */
  raycast(
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    maxDistance: number,
    options: QueryOptions = {},
  ): RaycastHit | null {
    const world = this.#world;
    if (world === null) return null;
    const groups = this.#queryGroups(options);
    const ray = new RAPIER.Ray(
      { x: origin.x, y: origin.y, z: origin.z },
      { x: direction.x, y: direction.y, z: direction.z },
    );
    const hit = world.castRayAndGetNormal(
      ray,
      maxDistance,
      true,
      this.#queryFlags(options),
      groups,
      options.exclude ?? undefined,
    );
    if (hit === null) return null;
    const distance = hit.timeOfImpact;
    return {
      distance,
      point: new THREE.Vector3(
        origin.x + direction.x * distance,
        origin.y + direction.y * distance,
        origin.z + direction.z * distance,
      ),
      normal: new THREE.Vector3(hit.normal.x, hit.normal.y, hit.normal.z),
      collider: hit.collider,
      record: this.recordFor(hit.collider),
    };
  }

  /**
   * Sweep a sphere. This is what the camera arm uses: a ray would thread
   * between a fence rail and the ground and put the lens inside the mud.
   */
  sphereCast(
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    radius: number,
    maxDistance: number,
    options: QueryOptions = {},
  ): ShapeCastHit | null {
    const world = this.#world;
    if (world === null) return null;
    const shape = new RAPIER.Ball(radius);
    const hit = world.castShape(
      { x: origin.x, y: origin.y, z: origin.z },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: direction.x, y: direction.y, z: direction.z },
      shape,
      0,
      maxDistance,
      // Starting the sweep already overlapping something (the camera pivot
      // inside a wall) must still report a hit at distance 0 rather than
      // tunnelling straight through it.
      true,
      this.#queryFlags(options),
      this.#queryGroups(options),
      options.exclude ?? undefined,
    );
    if (hit === null) return null;
    return {
      distance: hit.time_of_impact,
      point: new THREE.Vector3(hit.witness1.x, hit.witness1.y, hit.witness1.z),
      normal: new THREE.Vector3(hit.normal1.x, hit.normal1.y, hit.normal1.z),
      collider: hit.collider,
      record: this.recordFor(hit.collider),
    };
  }

  /**
   * Refresh the query acceleration structure.
   *
   * Rapier only rebuilds it inside `step()`, so every raycast issued before the
   * first simulation step reports empty space — including the one that decides
   * where the player spawns. Colliders built during `init` are therefore
   * invisible to queries until this is called, and the failure is silent: no
   * error, just a character who spawns at y=0 and falls into the moor.
   */
  syncQueries(): void {
    this.#world?.updateSceneQueries();
  }

  /** Ground height under a point, or `null` when there is nothing below it. */
  groundHeight(x: number, z: number, from = 200, range = 400): number | null {
    const hit = this.raycast(
      new THREE.Vector3(x, from, z),
      new THREE.Vector3(0, -1, 0),
      range,
      { layers: layerMask(CollisionLayer.Terrain, CollisionLayer.Prop) },
    );
    return hit === null ? null : hit.point.y;
  }

  #queryGroups(options: QueryOptions): number {
    const layers = options.layers ?? layerMask(CollisionLayer.Terrain, CollisionLayer.Prop);
    // A query has no membership of its own, so it borrows one broad enough to
    // satisfy the symmetric rule against anything it is allowed to hit.
    return interactionGroups(layers, layers);
  }

  #queryFlags(options: QueryOptions): RAPIER.QueryFilterFlags | undefined {
    return options.solidOnly === false ? undefined : RAPIER.QueryFilterFlags.EXCLUDE_SENSORS;
  }

  /* -- debug draw --------------------------------------------------------- */

  /** Whether the collider wireframe is being drawn. */
  get debugDraw(): boolean {
    return this.#debugDraw;
  }

  /**
   * Toggle the collider wireframe.
   *
   * `world.debugRender()` rebuilds the whole vertex buffer every call, which is
   * far too expensive to leave on — hence a toggle rather than a permanent
   * overlay. It is the only way to see the difference between "the collider is
   * in the wrong place" and "the controller is misbehaving", and those two look
   * identical from outside.
   */
  setDebugDraw(enabled: boolean): void {
    this.#debugDraw = enabled;
    const ctx = this.#ctx;
    if (!enabled) {
      if (this.#debugLines !== null) this.#debugLines.visible = false;
      return;
    }
    if (ctx === null) return;
    if (this.#debugLines === null) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 4));
      const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        depthTest: false,
        toneMapped: false,
        transparent: true,
      });
      const lines = new THREE.LineSegments(geometry, material);
      lines.name = 'physics.debug';
      lines.frustumCulled = false;
      lines.renderOrder = 999;
      ctx.scene.add(lines);
      this.#debugLines = lines;
    }
    this.#debugLines.visible = true;
    this.#updateDebugLines();
  }

  #updateDebugLines(): void {
    const world = this.#world;
    const lines = this.#debugLines;
    if (world === null || lines === null) return;
    const buffers = world.debugRender();
    lines.geometry.setAttribute('position', new THREE.BufferAttribute(buffers.vertices, 3));
    lines.geometry.setAttribute('color', new THREE.BufferAttribute(buffers.colors, 4));
    lines.geometry.setDrawRange(0, buffers.vertices.length / 3);
  }
}
