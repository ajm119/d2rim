/**
 * @module scene/DeadTree
 *
 * Procedural bare, wind-killed trees — the Blood Moor's only vegetation above
 * knee height.
 *
 * ## Why this exists rather than a tree model
 *
 * The reachable CC0 nature kit ships two conifers. They are competent models
 * and they are wrong for this brief in a way no material override can fix: a
 * conifer is a solid green cone, and a solid cone has no silhouette. Stylized
 * art direction lives on silhouette — the reason Torchlight's and Fable's
 * skylines read at a glance is that every tree on them is a *shape*, a set of
 * dark negative-space gaps against the sky, not a filled triangle. Retinting a
 * cone to brown produces a brown cone.
 *
 * So the trees are generated. That buys three things the kit cannot:
 *
 * 1. **Silhouette.** Limbs that fork, thin and claw outward give the ridge a
 *    broken, legible skyline with sky visible *through* it, which is what makes
 *    the aerial-perspective gradient behind it readable at all.
 * 2. **Palette control at the source.** There is no foliage, so there is no
 *    green. The moor's rule — nothing saturated but the fire — is enforced by
 *    the geometry existing, not by a grade fighting an atlas.
 * 3. **Cost control.** Four variant geometries are authored once and drawn as
 *    `InstancedMesh` batches, so an eighty-tree treeline is four draw calls.
 *
 * ## The generator
 *
 * A recursive tapered-tube skeleton. Each limb is swept as a ring of
 * `radialSegments` vertices per node, advancing along a direction that is
 * re-perturbed at every node so limbs *curve* instead of being straight cones —
 * straightness is the single loudest tell of a procedural tree. At each split
 * the limb forks into two or three children whose directions are the parent's
 * bent away from it and biased upward by {@link BranchSpec.upBias}, which is
 * what produces the characteristic dead-hardwood shape: heavy low limbs
 * reaching sideways, fine twigs clawing up.
 *
 * Radius follows the classic da Vinci area-preserving rule (a parent's
 * cross-section equals the sum of its children's) softened by `taper`, so limbs
 * thin believably instead of stepping down.
 *
 * Vertex colours carry a neutral (grey) luminance modulation, not a hue: dark
 * at the trunk base where mud splashes and rot collect, lightening toward the
 * twigs, times a per-limb random. Written grey specifically so it can be
 * consumed by `NodeMaterial`'s stock `vertexColors` path — which multiplies
 * *after* `colorNode` and therefore composes cleanly on top of whatever the
 * material library built — without tinting the bark toward a channel. It is the
 * cheapest available defence against eighty identically-toned trees reading as
 * wallpaper.
 *
 * Everything is seeded through `Procedural.createRng`, so a given seed always
 * produces the same forest — `Engine.stepFrames(n)` reproducibility depends on
 * it.
 */

import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

import { createRng, type Rng } from '../assets/Procedural';

/* -------------------------------------------------------------------------- */
/* Parameters                                                                 */
/* -------------------------------------------------------------------------- */

export interface DeadTreeOptions {
  readonly seed?: string | number;
  /** Overall height in metres, before instance scale. Default 6.2. */
  readonly height?: number;
  /** Trunk radius at the root, in metres. Default 0.19. */
  readonly radius?: number;
  /** How many times a limb may fork. Default 4. 5 is dense, 3 is a stump. */
  readonly depth?: number;
  /** Ring vertices on the trunk. Halves each level, floored at 3. Default 7. */
  readonly radialSegments?: number;
  /**
   * How hard children lean away from the parent, radians. Default 0.62 — a
   * wide, clawed hardwood. Below ~0.35 the tree reads as a broom.
   */
  readonly spread?: number;
  /**
   * How hard children are pulled back toward vertical, 0–1. Default 0.28. This
   * is the parameter that decides "willow" (0) versus "oak" (0.5).
   */
  readonly upBias?: number;
  /** Lean of the whole tree from vertical, radians. Default 0.12. */
  readonly lean?: number;
}

interface BranchSpec {
  readonly origin: THREE.Vector3;
  readonly direction: THREE.Vector3;
  readonly length: number;
  readonly radius: number;
  readonly depth: number;
  readonly upBias: number;
  readonly spread: number;
  /** 0 at the root of the tree, 1 at the tips. Baked to vertex colour. */
  readonly heightFraction: number;
}

/* -------------------------------------------------------------------------- */
/* Generation                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Build one dead tree as a single indexed `BufferGeometry`.
 *
 * Origin is at the base of the trunk, +Y up. Attributes: `position`, `normal`,
 * `uv` (u around the limb, v along it, so a bark set tiles without a seam
 * running up the trunk) and `color` (a neutral grey shading ramp — see the
 * module note).
 */
export function generateDeadTreeGeometry(options: DeadTreeOptions = {}): THREE.BufferGeometry {
  const {
    seed = 'd2rim.deadtree',
    height = 6.2,
    radius = 0.19,
    depth = 4,
    radialSegments = 7,
    spread = 0.62,
    upBias = 0.28,
    lean = 0.12,
  } = options;

  const rng = createRng(seed);
  const parts: THREE.BufferGeometry[] = [];

  // The trunk. Leaned in a random compass direction so a stand of trees does
  // not all list the same way — which, on a windswept moor, they very nearly
  // would, but "very nearly" is the difference between weather and a bug.
  const leanAzimuth = rng.next() * Math.PI * 2;
  const trunkDirection = new THREE.Vector3(
    Math.sin(lean) * Math.cos(leanAzimuth),
    Math.cos(lean),
    Math.sin(lean) * Math.sin(leanAzimuth),
  ).normalize();

  growBranch(
    {
      origin: new THREE.Vector3(0, 0, 0),
      direction: trunkDirection,
      // The trunk is 38% of the tree; the crown is built out of the remaining
      // 62% by the recursion. A trunk that is half the height reads as a
      // telegraph pole with sticks on it.
      length: height * 0.38,
      radius,
      depth,
      upBias,
      spread,
      heightFraction: 0,
    },
    rng,
    radialSegments,
    height,
    parts,
  );

  const merged = mergeGeometries(parts, false);
  for (const part of parts) part.dispose();
  if (merged === null) {
    // mergeGeometries only returns null on attribute mismatch, which cannot
    // happen here — every part comes out of the same builder. Degrading to an
    // empty geometry rather than throwing keeps one bad tree from costing the
    // scene.
    console.warn('[DeadTree] limb merge failed; emitting an empty tree.');
    return new THREE.BufferGeometry();
  }
  merged.computeBoundingSphere();
  return merged;
}

/**
 * Sweep one limb and recurse into its children.
 *
 * `parts` is appended to rather than returned so the recursion allocates one
 * array for the whole tree.
 */
function growBranch(
  spec: BranchSpec,
  rng: Rng,
  radialSegments: number,
  totalHeight: number,
  parts: THREE.BufferGeometry[],
): void {
  // More nodes low in the tree: the trunk is where curvature is visible at
  // silhouette scale, and a twig bending in four places is four times the
  // vertices for a shape nobody can resolve.
  const nodes = Math.max(2, spec.depth + 1);
  const segments = Math.max(3, radialSegments);
  const limbRandom = rng.next();

  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];

  const point = spec.origin.clone();
  const direction = spec.direction.clone().normalize();
  const tangent = new THREE.Vector3();
  const bitangent = new THREE.Vector3();
  const step = spec.length / nodes;

  // Taper along the limb. The tip is never zero-radius: a cone that closes to
  // a point aliases into a shimmering needle at distance, and a 12 mm twig end
  // is both cheaper to shade and closer to what a dead branch looks like.
  // 30 mm floor, up from 12. A 12 mm twig on the ridge is well under a pixel
  // wide at 47 m, and a sub-pixel bright-on-dark edge is where every temporal
  // and spatial artefact in the stack shows up at once: it crawls under TAA,
  // it aliases without it, and the grade's lateral aberration paints it
  // magenta. Stylized silhouettes want to be *bold* anyway — thin is not the
  // same as delicate.
  const tipRadius = Math.max(0.03, spec.radius * 0.34);

  for (let node = 0; node <= nodes; node++) {
    const t = node / nodes;
    const r = THREE.MathUtils.lerp(spec.radius, tipRadius, t * t * 0.6 + t * 0.4);
    basisFor(direction, tangent, bitangent);

    for (let s = 0; s <= segments; s++) {
      const angle = (s / segments) * Math.PI * 2;
      // Non-circular cross-section: dead hardwood limbs are lobed, and a
      // perfect cylinder is the second-loudest procedural tell after
      // straightness. The lobing is phase-locked to the limb random so the
      // whole limb twists as one instead of rippling.
      const lobe = 1 + 0.14 * Math.sin(angle * 3 + limbRandom * 6.283) + 0.07 * Math.sin(angle * 5);
      const cos = Math.cos(angle) * r * lobe;
      const sin = Math.sin(angle) * r * lobe;
      const nx = tangent.x * Math.cos(angle) + bitangent.x * Math.sin(angle);
      const ny = tangent.y * Math.cos(angle) + bitangent.y * Math.sin(angle);
      const nz = tangent.z * Math.cos(angle) + bitangent.z * Math.sin(angle);

      positions.push(
        point.x + tangent.x * cos + bitangent.x * sin,
        point.y + tangent.y * cos + bitangent.y * sin,
        point.z + tangent.z * cos + bitangent.z * sin,
      );
      normals.push(nx, ny, nz);
      // v advances in metres, not in [0,1], so bark tiles at a constant real
      // scale whether it is on a 3 m trunk or a 0.4 m twig.
      uvs.push(s / segments, (spec.heightFraction * totalHeight + t * spec.length) * 0.55);
      const heightRamp = THREE.MathUtils.clamp(
        (spec.heightFraction * totalHeight + t * spec.length) / totalHeight,
        0,
        1,
      );
      // Grey, so this reads as shading rather than as a tint. Base 0.58 →
      // tip 0.92, modulated ±12% per limb.
      const shade = THREE.MathUtils.clamp(
        (0.58 + 0.34 * heightRamp) * (0.88 + 0.24 * limbRandom),
        0.35,
        1.05,
      );
      colors.push(shade, shade, shade);
    }

    if (node < nodes) {
      const ring = segments + 1;
      for (let s = 0; s < segments; s++) {
        const a = node * ring + s;
        const b = a + ring;
        indices.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }

    // Advance, then bend. Bending after the ring is emitted is what makes the
    // limb curve smoothly rather than kink at each node.
    point.addScaledVector(direction, step);
    perturb(direction, rng, 0.16 + 0.1 * spec.upBias);
    // Gravity on the heavy low limbs, reach on the fine high ones. Sign flip
    // at depth 2 is the shape of a dead oak in one line.
    direction.y += (spec.depth >= 3 ? -0.035 : 0.05) * (1 - spec.upBias);
    direction.normalize();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  parts.push(geometry);

  if (spec.depth <= 0) return;

  // Three-way forks everywhere but the last level.
  //
  // Two-way forks below depth 3 were what made the crowns *sparse* — the first
  // build's trees read as scattered sticks against the cliff rather than as
  // trees, because a binary tree of long limbs puts almost all of its mass in
  // a handful of spars with nothing between them. A dead hardwood's crown is
  // dense at the outside and open at the inside, which is what three-way
  // forking plus a faster length decay produces.
  const children = spec.depth >= 1 ? 3 : 2;
  const tipHeight = spec.heightFraction + spec.length / totalHeight;
  // da Vinci: sum of child cross-sections equals the parent's, softened.
  const childRadius = spec.radius * Math.pow(1 / children, 0.42);
  const azimuthOffset = rng.next() * Math.PI * 2;

  for (let i = 0; i < children; i++) {
    const azimuth = azimuthOffset + (i / children) * Math.PI * 2 + (rng.next() - 0.5) * 0.7;
    basisFor(direction, tangent, bitangent);
    const angle = spec.spread * (0.65 + rng.next() * 0.7);
    const childDirection = direction
      .clone()
      .multiplyScalar(Math.cos(angle))
      .addScaledVector(tangent, Math.sin(angle) * Math.cos(azimuth))
      .addScaledVector(bitangent, Math.sin(angle) * Math.sin(azimuth));
    childDirection.y += spec.upBias;
    childDirection.normalize();

    growBranch(
      {
        origin: point.clone(),
        direction: childDirection,
        // Faster decay, from (0.58-0.80). Long children are the other half of
        // the sparseness problem: a limb that keeps 80% of its parent's length
        // for five levels ends up nearly as long as the trunk, and the crown
        // becomes a starburst instead of a canopy.
        length: spec.length * (0.5 + rng.next() * 0.17),
        radius: childRadius,
        depth: spec.depth - 1,
        upBias: spec.upBias,
        // Children spread barely wider than their parent. At 1.12 per level
        // the spread compounded to nearly 1.6x over five levels and the crown
        // blew open; the sky between the limbs has to come from the *gaps*
        // between fine twigs, not from the whole tree being splayed.
        spread: spec.spread * 1.03,
        heightFraction: tipHeight,
      },
      rng,
      Math.max(3, Math.round(radialSegments * 0.72)),
      totalHeight,
      parts,
    );
  }
}

/** An orthonormal basis perpendicular to `direction`, written in place. */
function basisFor(direction: THREE.Vector3, tangent: THREE.Vector3, bitangent: THREE.Vector3): void {
  // Pick the reference axis least parallel to the direction, or the basis
  // degenerates on a vertical trunk — which is, of course, every trunk.
  const reference =
    Math.abs(direction.y) < 0.9 ? UP : SIDE;
  tangent.copy(reference).cross(direction).normalize();
  bitangent.copy(direction).cross(tangent).normalize();
}

const UP = new THREE.Vector3(0, 1, 0);
const SIDE = new THREE.Vector3(1, 0, 0);

/** Nudge a direction by a bounded random rotation, in place. */
function perturb(direction: THREE.Vector3, rng: Rng, amount: number): void {
  direction.x += (rng.next() - 0.5) * amount;
  direction.y += (rng.next() - 0.5) * amount * 0.5;
  direction.z += (rng.next() - 0.5) * amount;
  direction.normalize();
}

/* -------------------------------------------------------------------------- */
/* Variant pool                                                               */
/* -------------------------------------------------------------------------- */

/**
 * A small pool of distinct silhouettes.
 *
 * Four is the number that matters: with one variant a treeline is a repeating
 * stamp and the eye finds it in under a second; with four plus random yaw and a
 * 0.7–1.5 scale range it does not. Twenty variants would be twenty geometry
 * buffers and twenty draw calls for a difference nobody can see.
 */
export const DEAD_TREE_VARIANTS: readonly DeadTreeOptions[] = [
  // The hero: tall, heavy-limbed, the one that carries the skyline. Depth 5,
  // not 4: at 4 the crown was a handful of long bare spars and the tree read as
  // a television aerial. The last level is what supplies the fine twig mass
  // that makes a bare crown look like a *crown* — it is roughly 60% of the
  // triangles and 100% of whether the silhouette is convincing.
  { height: 6.6, radius: 0.22, depth: 5, spread: 0.56, upBias: 0.3, lean: 0.08 },
  // Wind-bent, wide and clawed. Reads strongly in profile.
  { height: 5.4, radius: 0.185, depth: 5, spread: 0.74, upBias: 0.12, lean: 0.26 },
  // Younger, narrower, more upright — fills between the heroes.
  { height: 4.6, radius: 0.14, depth: 4, spread: 0.5, upBias: 0.42, lean: 0.14 },
  // A broken snag: two forks and a blunt top. Every treeline needs a stump.
  { height: 3.0, radius: 0.24, depth: 2, spread: 0.9, upBias: 0.05, lean: 0.19 },
] as const;
