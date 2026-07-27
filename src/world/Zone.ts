/**
 * @module world/Zone
 *
 * The contract every playable area implements, and the small value types the
 * zone system passes around.
 *
 * A {@link Zone} is a {@link GameModule} with three additions, and each one
 * exists because zone transitions are the moment all three become necessary:
 *
 * - a **root** object. Zones build into a subtree rather than straight into
 *   `ctx.scene`, so that unloading is "detach one node and walk it" rather than
 *   "remember everything you added". The player, his camera and the HUD live
 *   outside it deliberately: they survive travel.
 * - **entry points**, so travel can name a destination without the caller
 *   knowing zone-local coordinates. `travelTo('denOfEvil', 'cave-mouth')` is a
 *   sentence; `travelTo('denOfEvil', new Vector3(0.5, 3.1, 38.4))` is a
 *   coordinate that goes stale the first time the cave is regenerated.
 * - **colliders**, built by the zone against the shared physics world and torn
 *   down wholesale by {@link ZoneManager} on unload. A zone that leaked its
 *   colliders would leave the player walking into walls that are not there any
 *   more, which is the single most confusing failure a transition can produce.
 *
 * Coordinates inside a zone are **zone-local and origin-centred**: every zone
 * builds around `(0, ?, 0)`. Zones are never loaded simultaneously, so there is
 * no need for a global coordinate space, and an origin-centred zone keeps
 * floating-point precision, terrain-heightfield centring and camera-far
 * reasoning all in the range they were designed for.
 */

import * as THREE from 'three/webgpu';

import type { GameContext, GameModule } from '../core/types';
import type { PhysicsWorld } from '../physics/PhysicsWorld';
import type { SpawnPoint } from '../ai/EnemyDirector';

/**
 * A named arrival point.
 *
 * `position.y` is advisory: {@link ZoneManager} re-resolves the standing height
 * against the physics world before it teleports the player, because a hand-authored
 * y and a procedurally generated floor disagree eventually and the failure is a
 * player who spawns inside the ground.
 */
export interface ZoneEntryPoint {
  readonly id: string;
  /** Zone-local position. */
  readonly position: { readonly x: number; readonly y: number; readonly z: number };
  /** Facing, in radians, measured the way `PlayerController.yaw` is. */
  readonly yaw?: number;
}

/**
 * A marked, documented spot for a character phase 5 will add.
 *
 * The whole point of an anchor is that the *scene* owns the staging — which tent
 * a vendor stands outside, which way she faces, how much clear ground she needs
 * — and the quest/NPC system owns the character. Phase 5 asks for an anchor by
 * id and puts a figure on it; it never needs to know where the tents ended up.
 */
export interface NpcAnchor {
  /** Stable lowercase id, e.g. `'akara'`. */
  readonly id: string;
  readonly displayName: string;
  /** One line of who they are, so the anchor is self-documenting in a debugger. */
  readonly role: string;
  /** Zone-local ground position. `y` is the walkable surface height. */
  readonly position: { readonly x: number; readonly y: number; readonly z: number };
  /** Facing, in radians. Every camp anchor faces roughly inward, toward the fire. */
  readonly yaw: number;
  /** Radius kept free of props, in metres. A 1.85 m figure fits in 0.6. */
  readonly clearRadius: number;
  /** What the anchor is staged against, for whoever writes phase 5. */
  readonly note: string;
}

/** A transition volume, declared by the zone that contains it. */
export interface PortalSpec {
  /** Unique within the zone. */
  readonly id: string;
  /** Zone id to travel to. */
  readonly targetZone: string;
  /** Entry point id within the target zone. */
  readonly targetEntry: string;
  /** Zone-local centre of the trigger volume, at the player's feet. */
  readonly position: { readonly x: number; readonly y: number; readonly z: number };
  /** Horizontal radius of the volume, in metres. Default 2.2. */
  readonly radius?: number;
  /** Vertical extent above `position.y`, in metres. Default 3. */
  readonly height?: number;
  /**
   * The destination as the prompt should name it: "the Den of Evil" becomes
   * "Press E to enter the Den of Evil".
   */
  readonly label: string;
  /** Verb for the prompt. Default `'enter'`. */
  readonly verb?: string;
}

/**
 * A playable area.
 *
 * Zones are hosted by {@link ZoneManager}, not by the {@link Engine}: the manager
 * forwards the update phases to the active zone. That is what makes unloading a
 * zone a normal operation instead of a mid-frame mutation of the engine's module
 * registry.
 */
export interface Zone extends GameModule {
  /** Stable id used by `travelTo`. Distinct from `name`, the module id. */
  readonly zoneId: string;
  /** What a loading screen would call it. */
  readonly displayName: string;
  /** The subtree this zone owns. Added to `ctx.scene` by the zone's own `init`. */
  readonly root: THREE.Object3D;
  readonly entryPoints: readonly ZoneEntryPoint[];
  readonly portals: readonly PortalSpec[];
  /** Marked spots for phase 5's NPCs. Empty for zones with no inhabitants. */
  readonly npcAnchors?: readonly NpcAnchor[];
  /** Enemy encounter table. Omit or leave empty for a safe zone. */
  readonly enemySpawns?: readonly SpawnPoint[];
  /**
   * Build this zone's static physics.
   *
   * Called by {@link ZoneManager} after `init` resolves and before
   * `syncQueries()`. Everything created here is removed automatically on
   * unload — the manager tracks the colliders that appear across the call — so
   * a zone must not hold on to collider handles past its own `dispose`.
   */
  buildColliders?(physics: PhysicsWorld, ctx: GameContext): void;
}

/** Find an entry point by id, falling back to the first declared one. */
export function resolveEntryPoint(zone: Zone, id: string | null): ZoneEntryPoint | null {
  if (id !== null) {
    const named = zone.entryPoints.find((entry) => entry.id === id);
    if (named !== undefined) return named;
    console.warn(
      `[Zone] "${zone.zoneId}" has no entry point "${id}" ` +
        `(has: ${zone.entryPoints.map((e) => e.id).join(', ') || 'none'})`,
    );
  }
  return zone.entryPoints[0] ?? null;
}

/** Look an NPC anchor up by id. */
export function findAnchor(zone: Zone, id: string): NpcAnchor | null {
  return zone.npcAnchors?.find((anchor) => anchor.id === id) ?? null;
}

/* -------------------------------------------------------------------------- */
/* Resource ownership                                                         */
/* -------------------------------------------------------------------------- */

/**
 * Mark a material, geometry or texture as *shared* so that unloading a zone
 * leaves it alone.
 *
 * Anything that came out of {@link AssetManager}'s cache is shared by
 * definition: the cache hands the same `GLTF` to every caller, and disposing its
 * materials from one zone's teardown corrupts it for the next. Zones stamp the
 * things they cloned rather than created.
 */
export function markShared<T extends { userData?: Record<string, unknown> }>(resource: T): T {
  const target = resource as { userData?: Record<string, unknown> };
  target.userData = { ...(target.userData ?? {}), shared: true };
  return resource;
}

function isShared(resource: { userData?: Record<string, unknown> } | null | undefined): boolean {
  return resource?.userData?.['shared'] === true;
}

/** Every texture slot three exposes on a material, as a flat list. */
function disposeMaterialTextures(material: THREE.Material, disposed: Set<unknown>): number {
  let count = 0;
  // A material's texture slots are just properties holding `THREE.Texture`s, and
  // the set of slots differs per material class (and per node material). Walking
  // the own enumerable properties finds all of them without a hard-coded list
  // that would silently miss `iridescenceMap` the day someone uses it.
  for (const value of Object.values(material as unknown as Record<string, unknown>)) {
    if (value === null || typeof value !== 'object') continue;
    const texture = value as Partial<THREE.Texture> & { isTexture?: boolean };
    if (texture.isTexture !== true) continue;
    if (disposed.has(texture) || isShared(texture as { userData?: Record<string, unknown> })) {
      continue;
    }
    disposed.add(texture);
    (texture as THREE.Texture).dispose();
    count++;
  }
  return count;
}

/**
 * Detach a subtree and reclaim every GPU resource it owns.
 *
 * Returns the number of disposed resources so a test can assert that a
 * transition actually freed something rather than silently walking an already
 * empty tree.
 */
export function disposeZoneTree(root: THREE.Object3D): number {
  const disposed = new Set<unknown>();
  let count = 0;

  root.traverse((object) => {
    if (object instanceof THREE.Light) {
      // Lights own a shadow map render target that `Object3D.clear()` cannot see.
      object.dispose();
      count++;
    }
    const holder = object as unknown as {
      geometry?: THREE.BufferGeometry;
      material?: THREE.Material | THREE.Material[];
    };

    const geometry = holder.geometry;
    if (geometry !== undefined && !disposed.has(geometry) && !isShared(geometry)) {
      disposed.add(geometry);
      geometry.dispose();
      count++;
    }

    const material = holder.material;
    if (material === undefined) return;
    const materials = Array.isArray(material) ? material : [material];
    for (const entry of materials) {
      if (disposed.has(entry) || isShared(entry)) continue;
      disposed.add(entry);
      count += disposeMaterialTextures(entry, disposed);
      entry.dispose();
      count++;
    }
  });

  root.removeFromParent();
  root.clear();
  return count;
}

/** Meshes and triangles under a subtree. The analytical cost of a zone. */
export function measureZone(root: THREE.Object3D): { renderables: number; triangles: number } {
  let renderables = 0;
  let triangles = 0;
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    renderables++;
    const index = object.geometry.getIndex();
    const position = object.geometry.getAttribute('position');
    const verts = index !== null ? index.count : (position?.count ?? 0);
    const instances = object instanceof THREE.InstancedMesh ? object.count : 1;
    triangles += (verts / 3) * instances;
  });
  return { renderables, triangles: Math.round(triangles) };
}
