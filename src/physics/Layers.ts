/**
 * @module physics/Layers
 *
 * Collision layers and the interaction groups built from them.
 *
 * Rapier encodes filtering in a single 32-bit `InteractionGroups`: the high 16
 * bits are the *membership* (which layers this collider belongs to) and the low
 * 16 bits are the *filter* (which layers it is willing to interact with). Two
 * colliders interact only when each one's membership intersects the other's
 * filter, which makes the relation symmetric whether you like it or not — a
 * player that ignores triggers is also invisible to them.
 *
 * The constants live in their own module, free of any Rapier import, so that
 * the filtering rules can be unit-tested without instantiating a physics world
 * (and without paying for the WASM boot in the test runner).
 */

/**
 * Every collision layer in the game. Values are single bits in the low 16, so
 * they can be OR-ed into a mask and shifted into a membership field.
 */
export const CollisionLayer = {
  /** The terrain heightfield. Always solid, never moves. */
  Terrain: 1 << 0,
  /** Static world geometry: the ruined wall, boulders, fences, crates. */
  Prop: 1 << 1,
  /** The player's capsule. */
  Player: 1 << 2,
  /** Enemy capsules. */
  Enemy: 1 << 3,
  /** Non-solid volumes that report overlap: loot radius, zone borders, traps. */
  Trigger: 1 << 4,
  /** Arrows, bolts, thrown axes. */
  Projectile: 1 << 5,
  /**
   * Geometry the third-person camera arm must not pass through.
   *
   * Distinct from {@link CollisionLayer.Prop} because the two sets are not the
   * same: a waist-high crate blocks the body but the camera happily flies over
   * it, and a low fog card blocks neither. Everything solid joins this layer by
   * default; the separation exists so that individual props can opt out.
   */
  Camera: 1 << 6,
} as const;

export type CollisionLayerName = keyof typeof CollisionLayer;

/** All layer names in a stable order. For debug UI and diagnostics. */
export const COLLISION_LAYER_NAMES: readonly CollisionLayerName[] = [
  'Terrain',
  'Prop',
  'Player',
  'Enemy',
  'Trigger',
  'Projectile',
  'Camera',
];

/** A bitmask of {@link CollisionLayer} values. */
export type LayerMask = number;

/** Every layer at once. Useful as a "collide with anything" filter. */
export const ALL_LAYERS: LayerMask = 0xffff;

/** Nothing. A collider with this filter is inert. */
export const NO_LAYERS: LayerMask = 0;

/** OR a list of layers into a mask. */
export function layerMask(...layers: readonly number[]): LayerMask {
  let mask = 0;
  for (const layer of layers) mask |= layer;
  return mask >>> 0;
}

/**
 * Pack a membership and a filter into Rapier's `InteractionGroups` word.
 *
 * Both halves are masked to 16 bits first: an accidental overflow would
 * silently corrupt the other half, which produces the worst class of physics
 * bug — geometry that is solid for some things and not others, with no error.
 */
export function interactionGroups(membership: LayerMask, filter: LayerMask): number {
  return (((membership & 0xffff) << 16) | (filter & 0xffff)) >>> 0;
}

/** Extract the membership half of an interaction-groups word. */
export function membershipOf(groups: number): LayerMask {
  return (groups >>> 16) & 0xffff;
}

/** Extract the filter half of an interaction-groups word. */
export function filterOf(groups: number): LayerMask {
  return groups & 0xffff;
}

/**
 * Whether two colliders may interact, per Rapier's rule.
 *
 * Exported because it is the single sentence the whole layer scheme rests on,
 * and the only way to be sure a change to {@link COLLISION_GROUPS} did not
 * quietly make the player fall through the world is to assert it directly.
 */
export function groupsInteract(a: number, b: number): boolean {
  return (membershipOf(a) & filterOf(b)) !== 0 && (membershipOf(b) & filterOf(a)) !== 0;
}

/** Everything a body is expected to stand on or bump into. */
export const SOLID_WORLD: LayerMask = layerMask(CollisionLayer.Terrain, CollisionLayer.Prop);

/**
 * The canonical interaction-groups word for each kind of collider.
 *
 * Read the pairs as sentences:
 * - `Terrain` and `Prop` are members of the world and of the camera-blocking
 *   set, and accept characters, projectiles and the camera.
 * - `Player` collides with the world and with enemies, and is seen by triggers
 *   and projectiles, but is *not* in the camera set — the arm must be free to
 *   pass through the body it is anchored to.
 * - `Trigger` sees characters only. It never blocks anything, because it is
 *   built as a sensor; the groups just keep it out of every other query.
 */
export const COLLISION_GROUPS = {
  terrain: interactionGroups(
    layerMask(CollisionLayer.Terrain, CollisionLayer.Camera),
    layerMask(
      CollisionLayer.Player,
      CollisionLayer.Enemy,
      CollisionLayer.Projectile,
      CollisionLayer.Camera,
    ),
  ),
  prop: interactionGroups(
    layerMask(CollisionLayer.Prop, CollisionLayer.Camera),
    layerMask(
      CollisionLayer.Player,
      CollisionLayer.Enemy,
      CollisionLayer.Projectile,
      CollisionLayer.Camera,
    ),
  ),
  player: interactionGroups(
    CollisionLayer.Player,
    layerMask(
      CollisionLayer.Terrain,
      CollisionLayer.Prop,
      CollisionLayer.Enemy,
      CollisionLayer.Trigger,
      CollisionLayer.Projectile,
    ),
  ),
  enemy: interactionGroups(
    CollisionLayer.Enemy,
    layerMask(
      CollisionLayer.Terrain,
      CollisionLayer.Prop,
      CollisionLayer.Player,
      CollisionLayer.Enemy,
      CollisionLayer.Trigger,
      CollisionLayer.Projectile,
    ),
  ),
  trigger: interactionGroups(
    CollisionLayer.Trigger,
    layerMask(CollisionLayer.Player, CollisionLayer.Enemy),
  ),
  projectile: interactionGroups(
    CollisionLayer.Projectile,
    layerMask(
      CollisionLayer.Terrain,
      CollisionLayer.Prop,
      CollisionLayer.Player,
      CollisionLayer.Enemy,
    ),
  ),
  /**
   * The query groups the camera arm sphere-cast uses. It is a query, not a
   * collider, so only its filter half does any work — but the membership half
   * must still be set or nothing will ever match it.
   */
  cameraProbe: interactionGroups(CollisionLayer.Camera, CollisionLayer.Camera),
} as const satisfies Record<string, number>;

export type CollisionGroupName = keyof typeof COLLISION_GROUPS;
