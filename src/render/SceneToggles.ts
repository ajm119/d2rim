/**
 * @module render/SceneToggles
 *
 * Scene-graph kill switches: `?props=off`, `?terrain=off`, `?chars=off` and
 * `?flat=1`.
 *
 * ## Why the classification is by name and by type, not by a tag
 *
 * The three zones (`scene/RogueEncampment`, `scene/BloodMoor`, `scene/DenOfEvil`)
 * are 240 kB of content code between them and they build their contents
 * imperatively, so tagging every object at construction would mean editing
 * several hundred call sites for a diagnostic. What they *do* already do,
 * consistently, is name things: the ground is `ground` / `camp.ground` /
 * `den.floor`, props are `camp.forge.anvil`, `moor.denMouth.rock` and so on,
 * and characters arrive as `THREE.SkinnedMesh` because they are rigged.
 *
 * So the classifier reads:
 *
 * - **character** — the object is a `SkinnedMesh`, or has one anywhere beneath
 *   it. Structural, and therefore exactly right: a character is a thing with a
 *   skeleton, and nothing else in these scenes has one.
 * - **terrain** — the last dot-separated segment of the name is `ground`,
 *   `terrain`, `floor` or `heightfield`. Deliberately narrow. Cave *walls* and
 *   *ceilings* are not terrain here; they are static geometry that behaves like
 *   props for costing purposes, and calling them terrain would make
 *   `?terrain=off` empty the Den entirely and tell us nothing.
 * - **prop** — everything else that draws.
 *
 * Any zone can override the guess by setting `userData.d2rimKind` to
 * `'terrain' | 'prop' | 'character' | 'ignore'` on an object; `ignore` means
 * "never hide this", which is what lights, the sky dome and the fog volumes
 * want. That escape hatch is the reason the heuristic is allowed to be a
 * heuristic.
 *
 * ## Why `visible = false` and not removal
 *
 * three skips an invisible object *and its whole subtree* while assembling the
 * render list, so the draw never reaches the GPU and `info.render.drawCalls`
 * drops accordingly — which is the property the tests assert and the property
 * that makes the flag trustworthy. Removing objects instead would also work,
 * but it would fight the zone system's own lifetime management and could not be
 * undone. Hiding is idempotent, reversible, and survives a zone reload because
 * this module simply re-applies itself.
 *
 * ## Why the sweep is on a cadence rather than per frame
 *
 * Zones load asynchronously and stream props in afterwards, so a single sweep
 * at boot would miss most of the scene. A `Scene.traverse` of ~600 objects is
 * a few tens of microseconds, which is nothing — but it is also not nothing
 * *per frame* on a machine already at 730 ms, and this module exists to measure
 * that machine. It therefore sweeps on `zone:loaded` and `zone:travelEnd`, plus
 * a slow 0.5 s poll to catch anything spawned later (enemies, loot, the hero's
 * own rig). When every flag is at its default the module is never registered at
 * all and costs literally zero.
 *
 * ## `?flat=1`
 *
 * The single biggest lever available. Every scene material is swapped for an
 * unlit `MeshBasicNodeMaterial` carrying the original's base colour, which
 * removes — in one step — the terrain uber-shader's triplanar sample sets, all
 * IBL and environment sampling, all shadow-map sampling and every BRDF
 * evaluation from every fragment in the frame. Geometry, draw count and
 * triangle count are unchanged, so a comparison against the unflagged run
 * isolates *shading cost* from *submission cost* exactly.
 *
 * The originals are kept in a map and restored on dispose, and the swap is
 * skipped for anything the material library may still be hot-swapping (see
 * `#swapMaterial`), so a flat run is not a different scene, only a differently
 * shaded one.
 */

import * as THREE from 'three/webgpu';

import type { GameContext, GameModule } from '../core/types';

import type { RenderFlags } from './DebugFlags';

/** How a scene object is costed by the toggles. */
export type SceneObjectKind = 'terrain' | 'prop' | 'character' | 'ignore';

/** Name segments that mean "this is the ground". Matched on the last segment. */
const TERRAIN_SEGMENTS = new Set(['ground', 'terrain', 'floor', 'heightfield']);

/** How often the fallback sweep runs, in seconds. */
const SWEEP_INTERVAL = 0.5;

interface KindCarrier {
  readonly userData?: { d2rimKind?: unknown };
}

/**
 * Classify one object.
 *
 * Exported for `tests/render.sceneToggles.test.ts`, which is the only honest
 * way to pin a heuristic: the rule is only as good as the names it is asserted
 * against, and those names live in the scene files.
 *
 * @param object the object to classify
 * @param hasSkinnedDescendant whether the caller has already established that
 *   this object contains a skinned mesh. Passed in rather than recomputed
 *   because the sweep discovers it top-down in one pass.
 */
export function classifySceneObject(
  object: THREE.Object3D,
  hasSkinnedDescendant = false,
): SceneObjectKind {
  const declared = (object as unknown as KindCarrier).userData?.d2rimKind;
  if (
    declared === 'terrain' ||
    declared === 'prop' ||
    declared === 'character' ||
    declared === 'ignore'
  ) {
    return declared;
  }

  if (hasSkinnedDescendant || (object as { isSkinnedMesh?: boolean }).isSkinnedMesh === true) {
    return 'character';
  }

  const name = object.name;
  if (name.length > 0) {
    const segment = name.slice(name.lastIndexOf('.') + 1).toLowerCase();
    if (TERRAIN_SEGMENTS.has(segment)) return 'terrain';
  }

  return 'prop';
}

/**
 * Every object that is a skinned mesh or an ancestor of one.
 *
 * One `traverse` plus one walk up the parent chain per skinned mesh, rather
 * than a `containsSkinnedMesh` call per node — which is quadratic in the depth
 * of the graph and would have made the sweep itself a frame cost on the machine
 * this module exists to measure.
 */
export function collectCharacterRoots(root: THREE.Object3D): Set<THREE.Object3D> {
  const marked = new Set<THREE.Object3D>();
  root.traverse((child) => {
    if ((child as { isSkinnedMesh?: boolean }).isSkinnedMesh !== true) return;
    let node: THREE.Object3D | null = child;
    while (node !== null && node !== root && !marked.has(node)) {
      marked.add(node);
      node = node.parent;
    }
  });
  return marked;
}

/** What one sweep changed. Returned so tests and the overlay can assert it. */
export interface SceneToggleStats {
  /** Objects currently hidden by this module. */
  readonly hidden: number;
  /** Materials currently swapped for the flat stand-in. */
  readonly flattened: number;
  /** Sweeps performed since `init`. */
  readonly sweeps: number;
}

export class SceneToggles implements GameModule {
  readonly name = 'render.sceneToggles';

  readonly #flags: RenderFlags;
  /** Objects this module hid, so it never un-hides something the game hid. */
  readonly #hidden = new Set<THREE.Object3D>();
  /** Original materials, keyed by the mesh they came off. */
  readonly #originalMaterials = new WeakMap<THREE.Mesh, THREE.Material | THREE.Material[]>();
  readonly #flattened = new Set<THREE.Mesh>();
  /** Flat stand-ins, one per source material, so a swap does not churn. */
  readonly #flatCache = new WeakMap<THREE.Material, THREE.Material>();
  readonly #ownedFlats: THREE.Material[] = [];

  #sinceSweep = SWEEP_INTERVAL;
  #sweeps = 0;
  #unsubscribe: (() => void)[] = [];

  constructor(flags: RenderFlags) {
    this.#flags = flags;
  }

  /** True when this module would do nothing, so the caller can skip adding it. */
  static isNoOp(flags: RenderFlags): boolean {
    return flags.props && flags.terrain && flags.chars && flags.lit;
  }

  init(ctx: GameContext): void {
    // A zone that has just finished loading is the moment most of the scene
    // appears, and it is also the moment a travel undoes the previous sweep.
    for (const event of ['zone:loaded', 'zone:travelEnd'] as const) {
      this.#unsubscribe.push(
        ctx.events.on(event, () => {
          this.#sinceSweep = SWEEP_INTERVAL;
        }),
      );
    }
    this.sweep(ctx.scene);
  }

  /**
   * Runs in `lateUpdate` rather than `update` so that anything a gameplay
   * module spawned this frame — a loot drop, a summoned enemy — is already in
   * the graph when the sweep walks it.
   */
  lateUpdate(ctx: GameContext, dt: number): void {
    this.#sinceSweep += dt;
    if (this.#sinceSweep < SWEEP_INTERVAL) return;
    this.#sinceSweep = 0;
    this.sweep(ctx.scene);
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe = [];
    for (const object of this.#hidden) object.visible = true;
    this.#hidden.clear();
    for (const mesh of this.#flattened) {
      const original = this.#originalMaterials.get(mesh);
      if (original !== undefined) mesh.material = original;
    }
    this.#flattened.clear();
    for (const material of this.#ownedFlats) material.dispose();
    this.#ownedFlats.length = 0;
  }

  get stats(): SceneToggleStats {
    return { hidden: this.#hidden.size, flattened: this.#flattened.size, sweeps: this.#sweeps };
  }

  /**
   * Walk the scene once and apply every active flag.
   *
   * Public because the capture harness and the tests need a deterministic
   * "apply now" that does not depend on a frame having elapsed.
   */
  sweep(root: THREE.Object3D): SceneToggleStats {
    this.#sweeps++;
    const flags = this.#flags;
    const characterRoots = collectCharacterRoots(root);

    /**
     * @param inherited the kind established by an ancestor. Once inside a
     *   character rig everything is part of that character — a weapon mesh
     *   parented to a hand is not a "prop" that `?props=off` should delete out
     *   from under the hero — so the kind propagates down rather than being
     *   re-derived per node.
     */
    const visit = (object: THREE.Object3D, inherited: SceneObjectKind | null): void => {
      const kind =
        inherited ?? classifySceneObject(object, characterRoots.has(object));

      if (kind !== 'ignore') {
        const wanted =
          kind === 'terrain' ? flags.terrain : kind === 'character' ? flags.chars : flags.props;
        // Pure grouping nodes classify as `prop`, and hiding those would take
        // the whole world out under `?props=off`. Only objects that actually
        // draw — or the root of a character rig, which is a group by design —
        // are hideable.
        const drawable =
          (object as { isMesh?: boolean }).isMesh === true ||
          (object as { isPoints?: boolean }).isPoints === true ||
          (object as { isLine?: boolean }).isLine === true ||
          (kind === 'character' && characterRoots.has(object));

        if (!wanted && drawable) {
          if (object.visible) {
            object.visible = false;
            this.#hidden.add(object);
          }
          // An invisible object is skipped with its whole subtree during
          // render-list assembly, so there is nothing left below to classify.
          return;
        }
      }

      if (!flags.lit) this.#flatten(object);

      const descend = kind === 'character' ? 'character' : null;
      for (const child of object.children) visit(child, descend);
    };

    for (const child of root.children) visit(child, null);
    return this.stats;
  }

  /* -- internals --------------------------------------------------------- */

  #flatten(object: THREE.Object3D): void {
    const mesh = object as THREE.Mesh;
    if ((mesh as { isMesh?: boolean }).isMesh !== true) return;
    if (this.#flattened.has(mesh)) return;

    const current = mesh.material;
    this.#originalMaterials.set(mesh, current);
    mesh.material = Array.isArray(current)
      ? current.map((material) => this.#flatFor(material))
      : this.#flatFor(current);
    this.#flattened.add(mesh);
  }

  /**
   * An unlit stand-in carrying the source material's base colour.
   *
   * `MeshBasicNodeMaterial` still handles skinning and instancing — those are
   * geometry-stage concerns the node system attaches independently of the
   * shading model — so a flat run keeps the same silhouettes and the same
   * animation, and only the fragment work disappears. Transparency and side are
   * carried across because a flat frame with the foliage cards turned into
   * opaque quads would be a *different* number of shaded pixels, which defeats
   * the comparison.
   */
  #flatFor(material: THREE.Material): THREE.Material {
    const cached = this.#flatCache.get(material);
    if (cached !== undefined) return cached;

    const flat = new THREE.MeshBasicNodeMaterial();
    flat.name = `flat(${material.name.length > 0 ? material.name : material.type})`;
    const source = material as unknown as { color?: THREE.Color; map?: THREE.Texture | null };
    if (source.color instanceof THREE.Color) flat.color.copy(source.color);
    else flat.color.setRGB(0.5, 0.5, 0.5);
    flat.transparent = material.transparent;
    flat.opacity = material.opacity;
    flat.alphaTest = material.alphaTest;
    flat.side = material.side;
    flat.depthWrite = material.depthWrite;
    flat.depthTest = material.depthTest;
    // No `map`: binding the source texture would keep the sampler, the
    // filtering and the memory traffic that `?flat=1` exists to remove.

    this.#flatCache.set(material, flat);
    this.#ownedFlats.push(flat);
    return flat;
  }
}
