/**
 * @module combat/Hitbox
 *
 * Animation-driven, swept melee hit detection.
 *
 * ### Why swept, and not an overlap test
 *
 * A one-handed chop moves the axe head roughly 2.5 m in about 130 ms. At 60 Hz
 * that is ~4 cm short of half a metre of travel *per frame*, and a skeleton is
 * 0.35 m wide. A single-frame overlap test therefore misses roughly half of all
 * genuinely connecting swings, non-deterministically, depending on where the
 * frame boundary happened to fall. The player experiences this as "the game
 * ate my hit", which is the single most damaging thing melee combat can do.
 *
 * So the blade is treated as a **moving line segment**: its endpoints are
 * sampled every rendered frame, and the test is against the volume swept
 * between the previous and current segment, discretised into substeps. Five
 * substeps at 60 Hz puts the sampling interval at ~3 ms of clip time, which is
 * finer than the fastest authored swing by an order of magnitude.
 *
 * ### Where the damage window comes from
 *
 * Nowhere in this file. The window is *authored*, as normalised times on the
 * clip, and delivered by `AnimationGraph`'s event callback — see
 * `combat/CombatSystem.ts`, which passes `hit.open` / `hit.close` markers into
 * `playAction`. The hitbox only knows "open" and "closed". That separation is
 * what lets a swing's timing be retuned by editing one number in a move table
 * without touching any geometry code.
 *
 * ### One hit per target per swing
 *
 * Held by the hitbox, cleared on `beginSwing`. Without it a slow horizontal
 * slice registers on every substep it overlaps and deletes an enemy in one
 * button press.
 */

import * as THREE from 'three/webgpu';

import { findBone } from '../character/BoneNames';

/* -------------------------------------------------------------------------- */
/* Segment geometry (pure)                                                     */
/* -------------------------------------------------------------------------- */

/** Closest approach between two line segments. */
export interface SegmentClosest {
  /** Parameter along `p1 -> q1`, in `[0, 1]`. */
  readonly s: number;
  /** Parameter along `p2 -> q2`, in `[0, 1]`. */
  readonly t: number;
  readonly distance: number;
  readonly pointA: THREE.Vector3;
  readonly pointB: THREE.Vector3;
}

/**
 * Closest points between segments `[p1, q1]` and `[p2, q2]`.
 *
 * Ericson, *Real-Time Collision Detection*, §5.1.9. Degenerate segments (a
 * zero-length "segment", which a weapon anchor legitimately produces on the
 * frame a model finishes loading) are handled rather than divided by, because
 * the failure mode of the naive version is a `NaN` distance that compares false
 * against every threshold and silently disables the whole hitbox.
 */
export function closestPointsOnSegments(
  p1: THREE.Vector3,
  q1: THREE.Vector3,
  p2: THREE.Vector3,
  q2: THREE.Vector3,
): SegmentClosest {
  const d1 = new THREE.Vector3().subVectors(q1, p1);
  const d2 = new THREE.Vector3().subVectors(q2, p2);
  const r = new THREE.Vector3().subVectors(p1, p2);
  const a = d1.dot(d1);
  const e = d2.dot(d2);
  const f = d2.dot(r);
  const epsilon = 1e-10;

  let s: number;
  let t: number;

  if (a <= epsilon && e <= epsilon) {
    s = 0;
    t = 0;
  } else if (a <= epsilon) {
    s = 0;
    t = clamp01(f / e);
  } else {
    const c = d1.dot(r);
    if (e <= epsilon) {
      t = 0;
      s = clamp01(-c / a);
    } else {
      const b = d1.dot(d2);
      const denom = a * e - b * b;
      s = denom > epsilon ? clamp01((b * f - c * e) / denom) : 0;
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = clamp01(-c / a);
      } else if (t > 1) {
        t = 1;
        s = clamp01((b - c) / a);
      }
    }
  }

  const pointA = d1.multiplyScalar(s).add(p1);
  const pointB = d2.multiplyScalar(t).add(p2);
  return { s, t, distance: pointA.distanceTo(pointB), pointA, pointB };
}

function clamp01(value: number): number {
  return Number.isFinite(value) ? (value < 0 ? 0 : value > 1 ? 1 : value) : 0;
}

/** A contact produced by a swept test. */
export interface SweepContact {
  /** World-space contact point on the target's surface. */
  readonly point: THREE.Vector3;
  /** Unit surface normal, pointing from the target toward the blade. */
  readonly normal: THREE.Vector3;
  /** Fraction of the sweep interval at which contact happened, `[0, 1]`. */
  readonly time: number;
  /** How deep the blade was inside the target capsule, metres. */
  readonly depth: number;
}

/**
 * Sweep a blade segment from its previous pose to its current one and test it
 * against an upright capsule.
 *
 * `substeps` is the discretisation; the first substep that overlaps wins, so
 * the returned contact is the *earliest* one in the interval rather than the
 * deepest. That matters for the impact point: a deep contact is a point inside
 * the enemy, and the sparks belong on its skin.
 */
export function sweepSegmentAgainstCapsule(
  prevA: THREE.Vector3,
  prevB: THREE.Vector3,
  currA: THREE.Vector3,
  currB: THREE.Vector3,
  capsuleBase: THREE.Vector3,
  capsuleTop: THREE.Vector3,
  capsuleRadius: number,
  bladeRadius = 0,
  substeps = 5,
): SweepContact | null {
  const steps = Math.max(1, Math.floor(substeps));
  const reach = capsuleRadius + bladeRadius;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();

  for (let i = 0; i <= steps; i++) {
    const time = i / steps;
    a.lerpVectors(prevA, currA, time);
    b.lerpVectors(prevB, currB, time);
    const closest = closestPointsOnSegments(a, b, capsuleBase, capsuleTop);
    if (closest.distance > reach) continue;

    const normal = new THREE.Vector3().subVectors(closest.pointA, closest.pointB);
    if (normal.lengthSq() < 1e-12) {
      // The blade passed exactly through the capsule axis. Any lateral normal
      // is as good as any other; pick the horizontal swing direction so the
      // sparks still fly sideways rather than collapsing to a point.
      normal.subVectors(currB, prevB).setY(0);
      if (normal.lengthSq() < 1e-12) normal.set(0, 0, 1);
    }
    normal.normalize();

    return {
      point: closest.pointB.clone().addScaledVector(normal, capsuleRadius),
      normal,
      time,
      depth: reach - closest.distance,
    };
  }
  return null;
}

/* -------------------------------------------------------------------------- */
/* Weapon anchors                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Something that can report where the blade is, in world space, this frame.
 *
 * Two implementations exist because the two rigs in the game differ: the
 * Barbarian GLB parents real `1H_Axe` / `2H_Axe` meshes to `handslot.r`, and
 * the skeletons carry no weapon meshes at all. Rather than fake a weapon onto
 * the skeletons, their "blade" is their forearm extended past the fist, which
 * is both honest about what the art shows and produces a sensible reach.
 */
export interface WeaponAnchor {
  readonly label: string;
  /** Approximate blade length, metres. Diagnostics and reach checks. */
  readonly length: number;
  /** Write the hilt and tip world positions. @returns false if unavailable. */
  sample(hilt: THREE.Vector3, tip: THREE.Vector3): boolean;
}

/** A rigid weapon mesh parented to a hand socket. */
class MeshAnchor implements WeaponAnchor {
  readonly label: string;
  readonly length: number;
  readonly #object: THREE.Object3D;
  readonly #localHilt: THREE.Vector3;
  readonly #localTip: THREE.Vector3;

  constructor(object: THREE.Object3D, localHilt: THREE.Vector3, localTip: THREE.Vector3) {
    this.#object = object;
    this.#localHilt = localHilt;
    this.#localTip = localTip;
    this.label = object.name;
    this.length = localHilt.distanceTo(localTip);
  }

  sample(hilt: THREE.Vector3, tip: THREE.Vector3): boolean {
    const matrix = this.#object.matrixWorld;
    hilt.copy(this.#localHilt).applyMatrix4(matrix);
    tip.copy(this.#localTip).applyMatrix4(matrix);
    return true;
  }
}

/** The forearm extended past the fist: a punch, a claw, or an implied weapon. */
class LimbAnchor implements WeaponAnchor {
  readonly label: string;
  readonly length: number;
  readonly #hand: THREE.Object3D;
  readonly #elbow: THREE.Object3D;
  readonly #reach: number;

  constructor(hand: THREE.Object3D, elbow: THREE.Object3D, reach: number, label: string) {
    this.#hand = hand;
    this.#elbow = elbow;
    this.#reach = reach;
    this.label = label;
    this.length = reach;
  }

  sample(hilt: THREE.Vector3, tip: THREE.Vector3): boolean {
    this.#hand.getWorldPosition(hilt);
    this.#elbow.getWorldPosition(tip);
    // Direction elbow -> hand, continued past the hand by `reach`.
    tip.subVectors(hilt, tip);
    if (tip.lengthSq() < 1e-10) tip.set(0, 0, 1);
    tip.normalize().multiplyScalar(this.#reach).add(hilt);
    return true;
  }
}

export interface WeaponAnchorOptions {
  /** Mesh names to prefer, best first. Matched case-insensitively. */
  readonly meshNames?: readonly string[];
  /** Hand bone to fall back to. Default `hand.r`. */
  readonly handBone?: string;
  /** Elbow bone that gives the forearm direction. Default `lowerarm.r`. */
  readonly elbowBone?: string;
  /** Fallback reach past the fist, metres. Default 0.55. */
  readonly reach?: number;
}

/**
 * Resolve the best available blade for a character.
 *
 * Tries the named weapon meshes in order, then the limb fallback. Returns null
 * only when neither the meshes nor the arm bones exist, which means the model
 * is not the rig this game runs on and the caller should say so loudly.
 */
export function resolveWeaponAnchor(
  root: THREE.Object3D,
  options: WeaponAnchorOptions = {},
): WeaponAnchor | null {
  for (const name of options.meshNames ?? []) {
    const object = findWeaponMesh(root, name);
    if (object === null) continue;
    const extents = localBladeExtents(object);
    if (extents === null) continue;
    return new MeshAnchor(object, extents.hilt, extents.tip);
  }

  const handName = options.handBone ?? 'hand.r';
  const elbowName = options.elbowBone ?? 'lowerarm.r';
  const hand = findBone(root, handName);
  const elbow = findBone(root, elbowName);
  if (hand === null || elbow === null) return null;
  return new LimbAnchor(hand, elbow, Math.max(0.05, options.reach ?? 0.55), `${handName}+reach`);
}

function findWeaponMesh(root: THREE.Object3D, name: string): THREE.Object3D | null {
  const wanted = name.toLowerCase();
  let found: THREE.Object3D | null = null;
  root.traverse((object) => {
    if (found !== null) return;
    if (object.name.toLowerCase() === wanted) found = object;
  });
  return found;
}

/**
 * The two ends of a weapon mesh, in its own local space.
 *
 * Taken from the geometry's bounding box along its longest axis. The mesh is a
 * rigid child of a hand socket, so this is computed once and then only ever
 * transformed — no per-frame bounds recomputation, which would be the obvious
 * and much more expensive way to get the same numbers.
 */
export function localBladeExtents(
  object: THREE.Object3D,
): { hilt: THREE.Vector3; tip: THREE.Vector3 } | null {
  const mesh = object as THREE.Mesh;
  const geometry = mesh.geometry as THREE.BufferGeometry | undefined;
  if (geometry === undefined || typeof geometry.computeBoundingBox !== 'function') return null;
  if (geometry.boundingBox === null) geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  if (box === null) return null;

  const size = new THREE.Vector3().subVectors(box.max, box.min);
  const centre = new THREE.Vector3().addVectors(box.min, box.max).multiplyScalar(0.5);
  const axis =
    size.x >= size.y && size.x >= size.z ? 'x' : size.y >= size.z ? 'y' : 'z';

  const hilt = centre.clone();
  const tip = centre.clone();
  hilt[axis] = box.min[axis];
  tip[axis] = box.max[axis];
  return { hilt, tip };
}

/* -------------------------------------------------------------------------- */
/* The hitbox                                                                  */
/* -------------------------------------------------------------------------- */

/** The capsule a hitbox tests against, in world space. */
export interface CapsuleTarget {
  readonly id: number;
  footPosition(out: THREE.Vector3): THREE.Vector3;
  readonly hitRadius: number;
  readonly hitHeight: number;
}

export interface WeaponHitboxOptions {
  /** Blade thickness, metres. Forgiveness, not realism. Default 0.14. */
  readonly radius?: number;
  /** Sweep discretisation. Default 5. */
  readonly substeps?: number;
  /**
   * Extra reach added past the tip along the blade, metres. Default 0.
   * Melee that requires pixel-accurate spacing reads as unresponsive; a few
   * centimetres of slack costs nothing and is invisible.
   */
  readonly overreach?: number;
}

/** What a landed sweep reports back. */
export interface HitboxContact<T extends CapsuleTarget = CapsuleTarget> extends SweepContact {
  readonly target: T;
  /** Unit direction the tip travelled over the sweep interval. */
  readonly travel: THREE.Vector3;
}

export class WeaponHitbox {
  readonly #anchor: WeaponAnchor;
  readonly #radius: number;
  readonly #substeps: number;
  readonly #overreach: number;

  readonly #prevHilt = new THREE.Vector3();
  readonly #prevTip = new THREE.Vector3();
  readonly #hilt = new THREE.Vector3();
  readonly #tip = new THREE.Vector3();
  readonly #base = new THREE.Vector3();
  readonly #top = new THREE.Vector3();
  readonly #foot = new THREE.Vector3();

  readonly #hitThisSwing = new Set<number>();

  #primed = false;
  #open = false;
  /** Set when the window opened and closed inside one frame. */
  #forceSweep = false;

  constructor(anchor: WeaponAnchor, options: WeaponHitboxOptions = {}) {
    this.#anchor = anchor;
    this.#radius = Math.max(0.01, options.radius ?? 0.14);
    this.#substeps = Math.max(1, Math.floor(options.substeps ?? 5));
    this.#overreach = Math.max(0, options.overreach ?? 0);
  }

  get anchor(): WeaponAnchor {
    return this.#anchor;
  }

  get isOpen(): boolean {
    return this.#open;
  }

  /** Targets already hit by the swing in progress. Diagnostics and tests. */
  get hitCount(): number {
    return this.#hitThisSwing.size;
  }

  /** Current blade segment, for debug drawing. */
  segment(hilt: THREE.Vector3, tip: THREE.Vector3): void {
    hilt.copy(this.#hilt);
    tip.copy(this.#tip);
  }

  /** Start a new swing: clear the per-swing hit set and close the window. */
  beginSwing(): void {
    this.#hitThisSwing.clear();
    this.#open = false;
    this.#forceSweep = false;
  }

  openWindow(): void {
    this.#open = true;
    this.#forceSweep = true;
  }

  /**
   * Close the window.
   *
   * The window stays *logically* open for one more `track` call when it was
   * opened and closed inside the same frame — a real possibility when a clip is
   * short or the frame is long — so a swing can never be silently skipped by a
   * frame boundary.
   */
  closeWindow(): void {
    this.#open = false;
  }

  /** Abandon the swing entirely (interrupted, staggered, died). */
  cancel(): void {
    this.#open = false;
    this.#forceSweep = false;
    this.#hitThisSwing.clear();
  }

  /**
   * Sample the blade and, if the window is open, test it against `targets`.
   *
   * Must be called *after* the animation mixer has advanced for the frame, or
   * the segment sampled is one frame stale and every hit lands late.
   *
   * @returns the contacts found this frame, at most one per target.
   */
  track<T extends CapsuleTarget>(targets: readonly T[]): HitboxContact<T>[] {
    this.#prevHilt.copy(this.#hilt);
    this.#prevTip.copy(this.#tip);
    if (!this.#anchor.sample(this.#hilt, this.#tip)) return [];

    if (this.#overreach > 0) {
      const direction = this.#tip.clone().sub(this.#hilt);
      if (direction.lengthSq() > 1e-10) {
        this.#tip.addScaledVector(direction.normalize(), this.#overreach);
      }
    }

    if (!this.#primed) {
      // First sample of this character's life: there is no previous pose, and
      // sweeping from the origin would carve a hitbox across the whole level.
      this.#prevHilt.copy(this.#hilt);
      this.#prevTip.copy(this.#tip);
      this.#primed = true;
    }

    const shouldSweep = this.#open || this.#forceSweep;
    this.#forceSweep = false;
    if (!shouldSweep) return [];

    const travel = new THREE.Vector3().subVectors(this.#tip, this.#prevTip);
    if (travel.lengthSq() < 1e-12) travel.subVectors(this.#tip, this.#hilt);
    travel.normalize();

    const contacts: HitboxContact<T>[] = [];
    for (const target of targets) {
      if (this.#hitThisSwing.has(target.id)) continue;
      target.footPosition(this.#foot);
      const radius = Math.max(0.05, target.hitRadius);
      const height = Math.max(radius * 2, target.hitHeight);
      this.#base.set(this.#foot.x, this.#foot.y + radius, this.#foot.z);
      this.#top.set(this.#foot.x, this.#foot.y + height - radius, this.#foot.z);

      const contact = sweepSegmentAgainstCapsule(
        this.#prevHilt,
        this.#prevTip,
        this.#hilt,
        this.#tip,
        this.#base,
        this.#top,
        radius,
        this.#radius,
        this.#substeps,
      );
      if (contact === null) continue;
      this.#hitThisSwing.add(target.id);
      contacts.push({ ...contact, target, travel: travel.clone() });
    }
    return contacts;
  }
}
