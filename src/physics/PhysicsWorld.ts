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
  ALL_LAYERS,
  COLLISION_GROUPS,
  CollisionLayer,
  interactionGroups,
  layerMask,
  type LayerMask,
} from './Layers';

/**
 * Every object belonging to a skinned character under `root`.
 *
 * For each `SkinnedMesh`, the highest ancestor still inside `root` is treated
 * as that character's root and its whole subtree is marked. That catches the
 * skinned body, the armature, and anything parented to a bone socket.
 */
function collectCharacterSubtrees(root: THREE.Object3D): Set<THREE.Object3D> {
  const roots = new Set<THREE.Object3D>();
  root.traverse((object) => {
    if (!(object instanceof THREE.SkinnedMesh)) return;
    let top: THREE.Object3D = object;
    while (top.parent !== null && top.parent !== root) top = top.parent;
    roots.add(top);
  });
  const parts = new Set<THREE.Object3D>();
  for (const characterRoot of roots) characterRoot.traverse((object) => parts.add(object));
  return parts;
}

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

/**
 * When a derived box collider is suspicious enough to say so out loud.
 *
 * Two conditions, because either one alone has honest counter-examples: a
 * lintel really is 11 m long, and a doorstep really is 2 m square. What is
 * almost never a real prop is something *large in both horizontal axes* — that
 * is the signature of several separate objects merged into one mesh, whose
 * bounding box has swallowed the gaps between them. See `buildSceneColliders`.
 */
export const OVERSIZE_BOX_AREA = 25;
export const OVERSIZE_BOX_MIN_SIDE = 1.5;

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

  /**
   * Drop a record for a collider Rapier has *already* freed.
   *
   * Removing a rigid body removes the colliders attached to it, inside Rapier,
   * without going through {@link removeCollider} — so the owner of that body
   * has no way to keep the registry honest. It cannot call `removeCollider`:
   * that would ask Rapier to remove a collider whose handle is already dead,
   * which in a WASM build is a trap rather than an exception. And leaving the
   * record is not harmless: `#records` is what turns a raycast hit back into
   * "what did I hit", it is what the zone-leak assertions count, and a record
   * whose handle has since been *reissued* to a different collider makes
   * `recordFor` return the wrong answer rather than no answer.
   *
   * Measured before this existed: +32 stale `character` records across one
   * encampment → moor → den → moor → encampment lap, one per despawned enemy.
   *
   * @returns whether a record was actually dropped.
   */
  forgetCollider(record: ColliderRecord | null | undefined): boolean {
    if (record === null || record === undefined) return false;
    const known = this.#records.get(record.collider.handle);
    // Handle equality is not enough on its own — Rapier reuses handles — so the
    // record identity is checked too. A reissued handle belongs to somebody
    // else and must not be dropped on this caller's behalf.
    if (known !== record) return false;
    this.#records.delete(record.collider.handle);
    return true;
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

      // A box this big in both horizontal axes is almost never a prop; it is
      // several props merged into one mesh, whose bounding box now spans the
      // gaps between them. The camp shipped with six torch posts on a 17.5 m
      // ring welded into a single `Mesh`, which derived one invisible 35 m box
      // over the whole enclosure at chest height — the player stood inside it,
      // the camera arm sphere-cast started inside it, and nothing about the
      // picture said so.
      //
      // This warns rather than skips on purpose. A legitimately huge collider
      // (a cliff face, a bridge span) does exist, and silently dropping one
      // would put the player through the world; a line in the console is enough
      // to make the mistake findable, which is the whole thing it was missing.
      if (
        kind === 'box' &&
        hx * hz * 4 > OVERSIZE_BOX_AREA &&
        Math.min(hx, hz) * 2 > OVERSIZE_BOX_MIN_SIDE
      ) {
        console.warn(
          `[PhysicsWorld] "${label}" derives a ${(hx * 2).toFixed(1)} x ${(hz * 2).toFixed(1)} m ` +
            'box collider. If that is several objects merged into one mesh, split them: ' +
            'a bounding box cannot tell them apart, and the gap between them becomes solid.',
        );
      }

      this.addCollider(desc, { kind: 'prop', label, source: mesh });
    };

    // Everything belonging to a character, found before anything is emitted.
    //
    // Skipping `SkinnedMesh` alone is not enough, and the way it fails is
    // memorable: the Barbarian GLB parents a `1H_Axe`, a `2H_Axe`, a shield and
    // a `Mug` to his hand sockets as ordinary meshes, unused by the game and
    // invisible in the frame. As world geometry they become four static boxes
    // at chest height, arranged in a ring around wherever he spawned — a cage
    // he cannot walk out of, made of props nobody can see. Anything under a
    // character's root is a character's problem, and characters get capsules.
    const characterParts = collectCharacterSubtrees(root);

    root.traverse((object) => {
      if (object.userData['noCollide'] === true) return;
      if (config.exclude.test(object.name)) return;
      if (characterParts.has(object)) return;

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

  /**
   * Is anything on `layers` overlapping a sphere at `point`?
   *
   * Used to answer "can a body stand here" without moving one there first.
   * Returns the offending collider's record so the caller can say *what* is in
   * the way, which turns "the player cannot move" into "the player spawned
   * inside prop.barrel.large#3".
   */
  overlapSphere(
    point: THREE.Vector3,
    radius: number,
    layers: LayerMask = CollisionLayer.Prop,
  ): ColliderRecord | null {
    const world = this.#world;
    if (world === null) return null;
    const hit = world.intersectionWithShape(
      { x: point.x, y: point.y, z: point.z },
      { x: 0, y: 0, z: 0, w: 1 },
      new RAPIER.Ball(radius),
      RAPIER.QueryFilterFlags.EXCLUDE_SENSORS,
      interactionGroups(ALL_LAYERS, layers),
    );
    return hit === null ? null : (this.recordFor(hit) ?? null);
  }

  /**
   * Every collider on `layers` overlapping a sphere at `point`.
   *
   * The single-hit {@link overlapSphere} answers "is this blocked"; this
   * answers "by what", which is the question you have at 2am when a character
   * will not move and the scene has 120 derived colliders in it.
   */
  overlapAll(
    point: THREE.Vector3,
    radius: number,
    layers: LayerMask = CollisionLayer.Prop,
    exclude?: RAPIER.Collider,
  ): ColliderRecord[] {
    const world = this.#world;
    if (world === null) return [];
    const out: ColliderRecord[] = [];
    world.intersectionsWithShape(
      { x: point.x, y: point.y, z: point.z },
      { x: 0, y: 0, z: 0, w: 1 },
      new RAPIER.Ball(radius),
      (collider) => {
        const record = this.recordFor(collider);
        if (record !== null) out.push(record);
        return true;
      },
      RAPIER.QueryFilterFlags.EXCLUDE_SENSORS,
      interactionGroups(ALL_LAYERS, layers),
      exclude,
    );
    return out;
  }

  /**
   * Find somewhere near `x, z` a body of `radius` can actually stand.
   *
   * The scene is composed for a camera, so the spot the composition wants the
   * hero to occupy is not necessarily a spot he can walk out of — a barrel
   * placed for the frame sits half a metre in front of him and he is pinned on
   * his first step, which reads as broken controls rather than as a prop.
   *
   * A ring search, widening outward, keeping the requested point when it is
   * already clear. Returns `null` when nothing within `maxRadius` works, which
   * the caller should treat as a content bug rather than paper over.
   */
  findClearSpot(
    x: number,
    z: number,
    radius: number,
    height: number,
    maxRadius = 4,
  ): THREE.Vector3 | null {
    const probe = new THREE.Vector3();
    const rings = [0, 0.7, 1.4, 2.1, 2.8, 3.5].filter((r) => r <= maxRadius);
    for (const ring of rings) {
      const steps = ring === 0 ? 1 : 12;
      for (let i = 0; i < steps; i++) {
        const angle = (i / steps) * Math.PI * 2;
        const px = x + Math.cos(angle) * ring;
        const pz = z + Math.sin(angle) * ring;
        const ground = this.groundHeight(px, pz);
        if (ground === null) continue;
        // Two probes up the body: ankles clear of rubble, chest clear of a rail.
        probe.set(px, ground + radius * 1.1, pz);
        if (this.overlapSphere(probe, radius) !== null) continue;
        probe.set(px, ground + height * 0.6, pz);
        if (this.overlapSphere(probe, radius) !== null) continue;
        return new THREE.Vector3(px, ground, pz);
      }
    }
    return null;
  }

  /**
   * Height of the highest surface under a point, or `null` when there is
   * nothing below it.
   *
   * `layers` defaults to the terrain alone, not terrain + props: the usual
   * caller is asking "where does something stand here", and answering with the
   * lid of a crate the ray happened to pass through spawns the player on top of
   * the scenery.
   */
  groundHeight(
    x: number,
    z: number,
    layers: LayerMask = CollisionLayer.Terrain,
    from = 200,
    range = 400,
  ): number | null {
    const hit = this.raycast(new THREE.Vector3(x, from, z), new THREE.Vector3(0, -1, 0), range, {
      layers,
    });
    return hit === null ? null : hit.point.y;
  }

  /**
   * Interaction groups for a query.
   *
   * The membership half must be `ALL_LAYERS`, and this is the single most
   * error-prone line in the whole physics layer. Rapier's rule is symmetric:
   * a hit requires `query.membership & collider.filter` *and*
   * `collider.membership & query.filter`. The terrain's filter lists the things
   * that may stand on it — players, enemies, projectiles, the camera — and
   * deliberately does not list "terrain". So a query that declares itself a
   * member of the layers it wants to hit fails the first half of the test
   * against every one of them and silently returns `null` forever.
   *
   * The symptom of getting this wrong is not an error. It is a character who
   * has no ground normal (so slopes stop working), a camera arm that never
   * finds a wall (so it clips through the ruined masonry) and a foot IK solver
   * that never finds a floor — three unrelated-looking failures from one line.
   */
  #queryGroups(options: QueryOptions): number {
    const layers = options.layers ?? layerMask(CollisionLayer.Terrain, CollisionLayer.Prop);
    return interactionGroups(ALL_LAYERS, layers);
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
