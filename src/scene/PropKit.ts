/**
 * @module scene/PropKit
 *
 * Placing loaded prop models into a zone, correctly, without every zone
 * re-deriving the same four fixes.
 *
 * The four are worth stating, because each one is a bug that took a screenshot
 * to notice:
 *
 * 1. **Clone, never reuse.** `AssetManager.loadGLTF` returns the *cached* GLTF.
 *    Adding `gltf.scene` to the graph moves it — the second placement steals it
 *    from the first, and the zone ends up with one barrel where it asked for
 *    twelve.
 * 2. **Normalise scale from the model, not from a guess.** These packs are
 *    authored at wildly different scales (the hexkit tents are in tile units,
 *    the KayKit barrels in metres). Measuring the bounding box and solving for a
 *    target size is the only way a tent and a barrel end up the same world.
 * 3. **Weather the materials.** A raw kit atlas is saturated and bright, which
 *    breaks the act's palette rule on contact. {@link weatherMaterial} is the
 *    project's one grading path; the *grade* is the caller's decision.
 * 4. **Mark what is shared.** A clone shares its geometry and its source
 *    textures with the asset cache. {@link disposeZoneTree} would dispose them
 *    on unload and corrupt the cache for the next zone, so they are stamped and
 *    skipped. The weathered materials are freshly constructed, so they are *not*
 *    stamped: those genuinely are the zone's to free.
 *
 * The weathered-material cache is per-kit and therefore per-zone: fifty crates
 * cloned from one source share one weathered material and one draw-call setup,
 * and the whole set is freed when the zone is.
 */

import * as THREE from 'three/webgpu';

import type { AssetKey, AssetManager } from '../assets/AssetManager';
import { markShared } from '../world/Zone';
import { weatherMaterial, type WeatheringGrade } from './Weathering';

/** Where and how big a prop goes. All fields are world units and radians. */
export interface PropPlacement {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly yaw?: number;
  /** Extra multiplier on top of the normalising scale. Default 1. */
  readonly scale?: number;
  /** Small random-looking lean, in radians, applied about x and z. Default 0. */
  readonly tilt?: number;
  readonly name?: string;
}

export interface PropKitOptions {
  /** Weathering grade applied to every placed prop. */
  readonly grade: WeatheringGrade;
  /** Cast shadows from placed props. Default true. */
  readonly castShadow?: boolean;
}

/**
 * A loaded set of prop models, ready to be stamped into a zone.
 *
 * Construct one per zone, `await load(keys)` once, then `place()` as often as
 * needed. `dispose()` frees the weathered materials the kit created and nothing
 * else — the models themselves belong to the asset cache.
 */
export class PropKit {
  readonly #assets: AssetManager;
  readonly #grade: WeatheringGrade;
  readonly #castShadow: boolean;
  readonly #sources = new Map<AssetKey, THREE.Object3D>();
  readonly #scales = new Map<string, number>();
  readonly #materials = new Map<THREE.Material, THREE.Material>();

  constructor(assets: AssetManager, options: PropKitOptions) {
    this.#assets = assets;
    this.#grade = options.grade;
    this.#castShadow = options.castShadow ?? true;
  }

  /** Asset keys that actually loaded. */
  get available(): AssetKey[] {
    return Array.from(this.#sources.keys());
  }

  has(key: AssetKey): boolean {
    return this.#sources.has(key);
  }

  /**
   * Fetch a batch of models.
   *
   * Never throws and never rejects a whole zone because one model is missing:
   * failures are reported and the placements that needed them are skipped. A
   * camp with no barrels is a worse camp; a camp that fails to load is no camp.
   */
  async load(keys: readonly AssetKey[]): Promise<{ loaded: AssetKey[]; failed: AssetKey[] }> {
    const wanted = keys.filter((key) => this.#assets.has(key));
    const missing = keys.filter((key) => !this.#assets.has(key));
    const result = await this.#assets.preload(wanted);
    for (const key of result.loaded) {
      const gltf = this.#assets.peek(key);
      if (gltf === undefined || !isGltfLike(gltf)) continue;
      this.#assets.pin(key);
      this.#sources.set(key, gltf.scene);
    }
    const failed = [...result.failed, ...missing];
    if (failed.length > 0) {
      console.warn(
        `[PropKit] ${failed.length}/${keys.length} models unavailable ` +
          `(${failed.join(', ')}); those placements are skipped. Run \`npm run assets\`.`,
      );
    }
    return { loaded: result.loaded, failed };
  }

  /**
   * Uniform scale that makes this model `metres` tall (or wide, per `axis`).
   *
   * Cached per key+axis+target, because a `Box3.setFromObject` walks and
   * transforms every vertex of the model and a camp places forty props.
   */
  fitScale(key: AssetKey, metres: number, axis: 'y' | 'xz' = 'y'): number {
    const cacheKey = `${key}|${axis}|${metres}`;
    const cached = this.#scales.get(cacheKey);
    if (cached !== undefined) return cached;

    const source = this.#sources.get(key);
    if (source === undefined) return 1;
    source.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(source);
    const size = box.getSize(new THREE.Vector3());
    const measured = axis === 'y' ? size.y : Math.max(size.x, size.z);
    const scale = Number.isFinite(measured) && measured > 1e-4 ? metres / measured : 1;
    this.#scales.set(cacheKey, scale);
    return scale;
  }

  /**
   * Clone a prop into `parent`.
   *
   * @returns the placed object, or `null` when the model failed to load.
   */
  place(key: AssetKey, placement: PropPlacement, parent: THREE.Object3D): THREE.Object3D | null {
    const source = this.#sources.get(key);
    if (source === undefined) return null;

    const object = source.clone(true);
    object.name = placement.name ?? key;
    object.position.set(placement.x, placement.y, placement.z);
    object.rotation.set(placement.tilt ?? 0, placement.yaw ?? 0, (placement.tilt ?? 0) * 0.6);
    object.scale.setScalar(placement.scale ?? 1);
    this.#dress(object);
    parent.add(object);
    return object;
  }

  /**
   * Place a prop sized to a target height, in one call.
   *
   * The common case: "put a barrel here and make it 0.9 m tall" rather than
   * "put a barrel here at scale 0.0137", which is a number nobody can review.
   */
  placeSized(
    key: AssetKey,
    metres: number,
    placement: PropPlacement,
    parent: THREE.Object3D,
    axis: 'y' | 'xz' = 'y',
  ): THREE.Object3D | null {
    const fit = this.fitScale(key, metres, axis);
    return this.place(key, { ...placement, scale: fit * (placement.scale ?? 1) }, parent);
  }

  dispose(): void {
    for (const material of this.#materials.values()) material.dispose();
    this.#materials.clear();
    this.#sources.clear();
    this.#scales.clear();
  }

  /** Weather the clone's materials, tag shared resources, enable shadows. */
  #dress(root: THREE.Object3D): void {
    root.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      child.castShadow = this.#castShadow;
      child.receiveShadow = true;
      // The geometry is the cache's, shared with every other clone of this model.
      markShared(child.geometry);

      const source = child.material;
      const list = Array.isArray(source) ? source : [source];
      const replaced = list.map((material) => {
        const existing = this.#materials.get(material);
        if (existing !== undefined) return existing;
        const graded = weatherMaterial(material, this.#grade);
        // The graded material is new and owned. Its *textures* are not: the
        // node material copies `map`, `normalMap` and friends straight off the
        // source, and those came out of the asset cache.
        for (const value of Object.values(graded as unknown as Record<string, unknown>)) {
          if (value !== null && typeof value === 'object') {
            const texture = value as { isTexture?: boolean; userData?: Record<string, unknown> };
            if (texture.isTexture === true) markShared(texture);
          }
        }
        this.#materials.set(material, graded);
        return graded;
      });
      child.material = Array.isArray(source) ? replaced : (replaced[0] as THREE.Material);
    });
  }
}

/** Structural test: the loader's GLTF is a plain object, not a class instance. */
function isGltfLike(value: unknown): value is { scene: THREE.Object3D } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'scene' in value &&
    (value as { scene: unknown }).scene instanceof THREE.Object3D
  );
}
