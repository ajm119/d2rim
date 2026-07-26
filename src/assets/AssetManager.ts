/**
 * @module assets/AssetManager
 *
 * The single door every runtime asset comes through.
 *
 * ### Semantic keys, not paths
 *
 * Game code asks for `'env.overcast'`, never for
 * `'/assets/hdri/overcast.exr'`. The key space is generated from
 * `tools/assets/manifest.json` into `registry.generated.ts`, so it is typed:
 * a typo is a compile error, and re-pointing an asset at a different file is a
 * manifest edit that no gameplay module has to know about. It also means the
 * licence metadata for anything on screen is always one lookup away, which is
 * what makes the attribution guarantee enforceable rather than aspirational.
 *
 * ### Colour space is decided by role
 *
 * Every texture is tagged from its declared {@link AssetRole}: `albedo` and
 * `emissive` are sRGB, everything else is linear. This is not a detail. A
 * normal or roughness map decoded as sRGB produces lighting that is wrong in a
 * way that looks *almost* right, and it is nearly impossible to diagnose by eye
 * three modules downstream. Centralising the decision here means it can only be
 * made once, and correctly.
 *
 * ### Caching
 *
 * An LRU with a byte budget. Entries can be pinned, because the active
 * environment map must never be evicted mid-frame by a burst of texture loads.
 * Concurrent requests for the same key share one in-flight promise, so a
 * hundred rocks asking for the same albedo produce one network request.
 *
 * ### Progress
 *
 * Loading reports through the {@link EventBus} rather than through callbacks,
 * so a loading screen can subscribe without the asset layer knowing a loading
 * screen exists.
 */

import * as THREE from 'three/webgpu';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule, RendererHandle } from '../core/types';
import {
  GENERATED_ASSETS,
  type GeneratedAssetEntry,
  type GeneratedAssetKey,
} from './registry.generated';
import { isColorRole, type AssetRole } from './types';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** Byte-level progress for a single asset. Only fires for XHR-backed loaders. */
    'assets:progress': {
      key: AssetKey;
      loaded: number;
      total: number;
      /** `[0,1]`, or `-1` when the server sent no content length. */
      fraction: number;
    };
    /** One asset finished successfully. */
    'assets:loaded': {
      key: AssetKey;
      role: AssetRole;
      elapsedMs: number;
      /** True when served from the LRU without touching the network. */
      fromCache: boolean;
    };
    /** One asset failed. The load promise rejects as well; this is for UI. */
    'assets:error': { key: AssetKey; message: string };
    /** Aggregate progress across a {@link AssetManager.preload} batch. */
    'assets:batch': { completed: number; total: number; fraction: number; done: boolean };
  }
}

/* -------------------------------------------------------------------------- */
/* Public types                                                                */
/* -------------------------------------------------------------------------- */

/** Every semantic key the manifest defines. */
export type AssetKey = GeneratedAssetKey;

/** Anything the manager can hand back. */
export type LoadedAsset = THREE.Texture | GLTF | ArrayBuffer;

/** Service key so other modules can find the manager without importing it. */
export const AssetManagerKey = serviceKey<AssetManager>('assets');

export interface AssetManagerOptions {
  /**
   * Root the registry's relative paths resolve against. Defaults to Vite's
   * `BASE_URL`, so a build deployed under a sub-path still finds its assets.
   */
  baseUrl?: string;
  /**
   * LRU budget in bytes. Default 256 MB. This is an estimate of GPU-side
   * footprint, not of download size — it is what actually constrains us.
   */
  cacheBudgetBytes?: number;
  /**
   * Anisotropic filtering level. Defaults to the renderer's maximum, which is
   * the right default for a project where ground planes are viewed at grazing
   * angles almost constantly.
   */
  anisotropy?: number;
}

export interface TextureLoadOptions {
  /** Override the role-derived colour space. Rarely correct; think first. */
  colorSpace?: THREE.ColorSpace;
  /** Wrap mode for both axes. Default `RepeatWrapping` — we tile nearly everything. */
  wrap?: THREE.Wrapping;
  /** Override anisotropy for this texture. */
  anisotropy?: number;
  /** Texture-space repeat, applied to `texture.repeat`. */
  repeat?: number | readonly [number, number];
  /** Force `flipY`. Defaults to the loader's own convention. */
  flipY?: boolean;
}

/** A cache entry plus the bookkeeping the LRU needs. */
interface CacheEntry {
  readonly key: AssetKey;
  readonly value: LoadedAsset;
  readonly bytes: number;
  pinned: boolean;
}

export interface AssetStats {
  readonly entries: number;
  readonly bytes: number;
  readonly budgetBytes: number;
  readonly pinned: number;
  readonly hits: number;
  readonly misses: number;
  readonly evictions: number;
  readonly inFlight: number;
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

/** Extension in lower case, without the dot. */
function extensionOf(path: string): string {
  const dot = path.lastIndexOf('.');
  return dot < 0 ? '' : path.slice(dot + 1).toLowerCase();
}

/**
 * Best-effort GPU footprint of a texture, in bytes.
 *
 * Deliberately an estimate. The exact number depends on the driver's internal
 * format choice, but a cache budget only needs to be right to within a factor
 * that keeps eviction sane, and a wrong-by-30% estimate still evicts the right
 * things in the right order. The 4/3 factor accounts for the mip chain.
 */
function estimateTextureBytes(texture: THREE.Texture): number {
  const image = texture.image as { width?: number; height?: number } | null;
  const width = image?.width ?? 0;
  const height = image?.height ?? 0;
  if (width === 0 || height === 0) return 0;

  const bytesPerChannel =
    texture.type === THREE.FloatType ? 4 : texture.type === THREE.HalfFloatType ? 2 : 1;
  const mipFactor = texture.generateMipmaps ? 4 / 3 : 1;
  return Math.round(width * height * 4 * bytesPerChannel * mipFactor);
}

/**
 * Read the renderer's maximum anisotropy across both backends.
 *
 * The node renderer exposes `getMaxAnisotropy()` directly; the classic
 * `WebGLRenderer` hides it behind `capabilities`. Probed structurally so this
 * stays true if the fallback tier is ever switched to a real `WebGLRenderer`.
 */
function maxAnisotropyOf(renderer: RendererHandle): number {
  const candidate = renderer.three as unknown as {
    getMaxAnisotropy?: () => number;
    capabilities?: { getMaxAnisotropy?: () => number };
  };
  const value = candidate.getMaxAnisotropy?.() ?? candidate.capabilities?.getMaxAnisotropy?.() ?? 1;
  return Number.isFinite(value) && value > 0 ? value : 1;
}

/**
 * Flip an equirectangular texture's rows in place, so row 0 becomes the zenith.
 *
 * ### Why this is necessary
 *
 * `scene.environment` interprets an equirect DataTexture with row 0 at the
 * zenith — that is the orientation the project's procedural sky uses and the
 * one whose reflections were verified correct against the metal spheres in the
 * reference scene.
 *
 * The EXR plates decode the other way up. This is not a guess: each shipped
 * plate was decoded and inspected, and every one has the ground at row 0 and
 * the sky at the last row. Measured on `overcast.exr`, the solid-angle-weighted
 * mean radiance of the row-0 half is 0.159 against 0.818 for the other half —
 * the bright hemisphere is at the bottom of the buffer.
 *
 * Left uncorrected the consequence is not subtle but it is easy to misread: the
 * scene gets lit from below, so contact shadows fill in, the key-to-fill ratio
 * collapses, and the whole image goes flat and slightly uplit. It looks like a
 * tone-mapping or exposure problem, which is where the time gets lost.
 *
 * Rows are swapped rather than setting `flipY`, because `flipY` is ignored for
 * data textures on both backends.
 */
/** The subset of the TypedArray surface {@link flipEquirectVertically} needs. */
interface MutableTypedArray extends ArrayLike<number> {
  set(array: ArrayLike<number>, offset?: number): void;
  subarray(begin: number, end: number): MutableTypedArray;
  slice(begin: number, end: number): MutableTypedArray;
  copyWithin(target: number, start: number, end: number): void;
}

export function flipEquirectVertically(texture: THREE.DataTexture): void {
  // Boundary cast: three types `image` loosely, but for a DataTexture it is
  // always `{ data: TypedArray, width, height }`. `MutableTypedArray` is the
  // minimal shape every TypedArray satisfies, which keeps this working for the
  // half-float (`Uint16Array`) and float (`Float32Array`) cases alike.
  const image = texture.image as unknown as {
    data: MutableTypedArray;
    width: number;
    height: number;
  };

  const { data, width, height } = image;
  const channels = data.length / (width * height);
  const rowLength = width * channels;
  const scratch = data.slice(0, rowLength);

  for (let y = 0; y < height >> 1; y++) {
    const top = y * rowLength;
    const bottom = (height - 1 - y) * rowLength;
    scratch.set(data.subarray(top, top + rowLength));
    data.copyWithin(top, bottom, bottom + rowLength);
    data.set(scratch, bottom);
  }
  texture.needsUpdate = true;
}

/** Dispose anything the cache can hold, releasing GPU memory deterministically. */
function disposeAsset(value: LoadedAsset): void {
  if (value instanceof THREE.Texture) {
    value.dispose();
    return;
  }
  if (value instanceof ArrayBuffer) return;

  // A GLTF: walk the scene and release geometries, materials and their maps.
  const seenMaterials = new Set<THREE.Material>();
  value.scene.traverse((object) => {
    const mesh = object as Partial<THREE.Mesh>;
    mesh.geometry?.dispose();
    const material = mesh.material;
    if (material === undefined) return;
    for (const entry of Array.isArray(material) ? material : [material]) {
      if (seenMaterials.has(entry)) continue;
      seenMaterials.add(entry);
      for (const property of Object.values(entry as unknown as Record<string, unknown>)) {
        if (property instanceof THREE.Texture) property.dispose();
      }
      entry.dispose();
    }
  });
}

/* -------------------------------------------------------------------------- */
/* AssetManager                                                                */
/* -------------------------------------------------------------------------- */

export class AssetManager implements GameModule {
  readonly name = 'AssetManager';

  readonly #options: Required<AssetManagerOptions>;

  /** Insertion order is LRU order: least recently used first. */
  readonly #cache = new Map<AssetKey, CacheEntry>();
  /** Deduplicates concurrent requests for the same key. */
  readonly #inFlight = new Map<AssetKey, Promise<LoadedAsset>>();

  #ctx: GameContext | null = null;
  #cacheBytes = 0;
  #hits = 0;
  #misses = 0;
  #evictions = 0;
  #disposed = false;

  #gltfLoader: GLTFLoader | null = null;
  #ktx2Loader: KTX2Loader | null = null;
  readonly #textureLoader = new THREE.TextureLoader();
  readonly #exrLoader = new EXRLoader();
  readonly #rgbeLoader = new RGBELoader();

  /**
   * Override for where the KTX2 Basis transcoder is served from.
   *
   * Leave it `null` — the default — and three resolves its own transcoder
   * through `new URL('../libs/basis/...', import.meta.url)`, which Vite rewrites
   * at build time into a hashed emitted asset. That is the path that actually
   * works, and setting a transcoder path unconditionally *breaks* it by pointing
   * the loader at a directory the build never creates.
   *
   * Set this only when self-hosting the transcoder somewhere specific, e.g.
   * behind a CDN or an offline mirror.
   *
   * No reachable asset source in this environment ships KTX2, so nothing
   * requests it today; the wiring is here so that adding a `.ktx2` entry to the
   * manifest is all it takes.
   */
  ktx2TranscoderPath: string | null = null;

  constructor(options: AssetManagerOptions = {}) {
    this.#options = {
      baseUrl: options.baseUrl ?? import.meta.env.BASE_URL,
      cacheBudgetBytes: options.cacheBudgetBytes ?? 256 * 1024 * 1024,
      // Resolved properly in `init`, once a renderer exists.
      anisotropy: options.anisotropy ?? 0,
    };
  }

  /* -- GameModule -------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;

    // Self-registering, so that modules initialised after this one find the
    // manager through the service locator rather than importing it. Registered
    // in `init` rather than in the constructor because the locator only exists
    // once the engine has built a context.
    ctx.services.register(AssetManagerKey, this);

    if (this.#options.anisotropy <= 0) {
      this.#options.anisotropy = maxAnisotropyOf(ctx.renderer);
    }

    this.#gltfLoader = new GLTFLoader();

    console.info(
      `[AssetManager] ready — ${Object.keys(GENERATED_ASSETS).length} registered assets, ` +
        `anisotropy ${this.#options.anisotropy}, ` +
        `cache budget ${(this.#options.cacheBudgetBytes / (1024 * 1024)).toFixed(0)} MB`,
    );
  }

  dispose(): void {
    this.#disposed = true;
    this.clear({ includePinned: true });
    this.#ktx2Loader?.dispose();
    this.#ktx2Loader = null;
    this.#gltfLoader = null;
    this.#ctx = null;
  }

  /* -- Registry ---------------------------------------------------------- */

  /** Whether `key` names a registered asset. Cheap; no I/O. */
  has(key: string): key is AssetKey {
    return Object.hasOwn(GENERATED_ASSETS, key);
  }

  /**
   * Registry metadata for an asset: path, role, tier, licence, size, hash.
   *
   * The return type is deliberately the *wide* {@link GeneratedAssetEntry}
   * rather than the generated literal type. The generated object is `as const`,
   * so `role` would otherwise narrow to just the roles the manifest happens to
   * use today — and then perfectly correct code like `if (role === 'binary')`
   * becomes a compile error until someone adds a binary asset. Widening once,
   * here, keeps the key space precisely typed while leaving the value space
   * open for growth.
   */
  entry(key: AssetKey): GeneratedAssetEntry {
    return GENERATED_ASSETS[key];
  }

  /** Fully resolved URL an asset loads from. */
  url(key: AssetKey): string {
    const base = this.#options.baseUrl;
    const path = this.entry(key).path;
    return `${base.endsWith('/') ? base : `${base}/`}${path}`;
  }

  /** Every registered key. Diagnostics and tooling. */
  keys(): AssetKey[] {
    return Object.keys(GENERATED_ASSETS) as AssetKey[];
  }

  /* -- Loading ----------------------------------------------------------- */

  /**
   * Load by key, dispatching on the asset's declared role.
   *
   * Prefer the typed variants ({@link loadTexture}, {@link loadGLTF},
   * {@link loadEnvironment}) at call sites that know what they want; this
   * exists for generic paths like {@link preload}.
   */
  async load(key: AssetKey): Promise<LoadedAsset> {
    const role = this.entry(key).role;
    if (role === 'gltf') return this.loadGLTF(key);
    if (role === 'hdri') return this.loadEnvironment(key);
    if (role === 'binary') return this.loadBinary(key);
    return this.loadTexture(key);
  }

  /**
   * Load a texture with role-correct colour space and full anisotropy.
   *
   * @throws if `key`'s role is not a texture role, so a GLB requested as a
   *         texture fails at the call site instead of producing a broken map.
   */
  async loadTexture(key: AssetKey, options: TextureLoadOptions = {}): Promise<THREE.Texture> {
    const entry = this.entry(key);
    if (entry.role === 'gltf' || entry.role === 'binary') {
      throw new Error(`[AssetManager] "${key}" has role "${entry.role}" and is not a texture`);
    }

    const texture = (await this.#acquire(key, async () => {
      const loaded = await this.#loadTextureFile(key);
      this.#configureTexture(loaded, entry.role, options);
      return loaded;
    })) as THREE.Texture;

    // Per-request presentation overrides are applied even on a cache hit, since
    // two call sites may legitimately want different tiling of the same image.
    this.#applyPresentation(texture, options);
    return texture;
  }

  /**
   * Load an equirectangular HDR environment map, ready for `scene.environment`.
   *
   * The texture is returned unfiltered with `EquirectangularReflectionMapping`
   * and the renderer builds its own pre-filtered chain. That is deliberate:
   * on the node renderer, running `PMREMGenerator` by hand and assigning the
   * result is both redundant and a known source of orientation and range bugs.
   * Assign the result straight to `scene.environment`.
   */
  async loadEnvironment(key: AssetKey): Promise<THREE.DataTexture> {
    const entry = this.entry(key);
    if (entry.role !== 'hdri') {
      throw new Error(`[AssetManager] "${key}" has role "${entry.role}", expected "hdri"`);
    }

    const texture = (await this.#acquire(key, async () => {
      const url = this.url(key);
      const extension = extensionOf(entry.path);

      let loaded: THREE.DataTexture;
      if (extension === 'exr') {
        loaded = await this.#loadWithProgress(this.#exrLoader, url, key);
        // See flipEquirectVertically: EXR plates decode ground-up, and an
        // upside-down IBL lights the scene from below.
        flipEquirectVertically(loaded);
      } else if (extension === 'hdr') {
        loaded = await this.#loadWithProgress(this.#rgbeLoader, url, key);
      } else {
        throw new Error(
          `[AssetManager] "${key}" is role "hdri" but has extension ".${extension}" ` +
            `(expected .exr or .hdr)`,
        );
      }

      loaded.mapping = THREE.EquirectangularReflectionMapping;
      // Linear, not sRGB: these are radiance values, and several plates carry
      // highlights well above 1 that ACES needs intact to roll off.
      loaded.colorSpace = THREE.LinearSRGBColorSpace;
      loaded.needsUpdate = true;
      return loaded;
    })) as THREE.DataTexture;

    return texture;
  }

  /** Load a glTF/GLB scene. The returned object is shared — clone before mutating. */
  async loadGLTF(key: AssetKey): Promise<GLTF> {
    const entry = this.entry(key);
    if (entry.role !== 'gltf') {
      throw new Error(`[AssetManager] "${key}" has role "${entry.role}", expected "gltf"`);
    }

    return (await this.#acquire(key, async () => {
      const loader = this.#gltfLoader;
      if (loader === null) throw new Error('[AssetManager] init() has not run yet');

      const gltf = await this.#loadWithProgress(loader, this.url(key), key);

      // glTF ships colour textures already tagged sRGB by the loader, but
      // anisotropy is ours to set and matters as much here as anywhere.
      gltf.scene.traverse((object) => {
        const mesh = object as Partial<THREE.Mesh>;
        const material = mesh.material;
        if (material === undefined) return;
        for (const entry_ of Array.isArray(material) ? material : [material]) {
          for (const property of Object.values(entry_ as unknown as Record<string, unknown>)) {
            if (property instanceof THREE.Texture) {
              property.anisotropy = this.#options.anisotropy;
              property.needsUpdate = true;
            }
          }
        }
      });
      return gltf;
    })) as GLTF;
  }

  /** Load raw bytes. For anything the engine parses itself. */
  async loadBinary(key: AssetKey): Promise<ArrayBuffer> {
    return (await this.#acquire(key, async () => {
      const response = await fetch(this.url(key));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }
      return response.arrayBuffer();
    })) as ArrayBuffer;
  }

  /**
   * Load many assets, emitting aggregate `assets:batch` progress.
   *
   * Failures are collected rather than short-circuiting: a loading screen
   * should get through the whole set and then report what is missing, not stall
   * on the first 404.
   */
  async preload(keys: readonly AssetKey[]): Promise<{ loaded: AssetKey[]; failed: AssetKey[] }> {
    const total = keys.length;
    const loaded: AssetKey[] = [];
    const failed: AssetKey[] = [];
    let completed = 0;

    const emit = (done: boolean): void => {
      this.#ctx?.events.emit('assets:batch', {
        completed,
        total,
        fraction: total === 0 ? 1 : completed / total,
        done,
      });
    };
    emit(total === 0);

    await Promise.all(
      keys.map(async (key) => {
        try {
          await this.load(key);
          loaded.push(key);
        } catch {
          failed.push(key);
        } finally {
          completed++;
          emit(completed === total);
        }
      }),
    );

    return { loaded, failed };
  }

  /* -- Cache ------------------------------------------------------------- */

  /** Synchronous peek. Returns `undefined` unless the asset is already resident. */
  peek(key: AssetKey): LoadedAsset | undefined {
    const entry = this.#cache.get(key);
    if (entry === undefined) return undefined;
    this.#touch(key, entry);
    return entry.value;
  }

  /**
   * Protect an asset from eviction.
   *
   * The active environment map is the motivating case: evicting it during a
   * texture burst would blank every reflection in the frame.
   */
  pin(key: AssetKey): void {
    const entry = this.#cache.get(key);
    if (entry !== undefined) entry.pinned = true;
  }

  /** Release a pin, making the asset evictable again. */
  unpin(key: AssetKey): void {
    const entry = this.#cache.get(key);
    if (entry !== undefined) entry.pinned = false;
  }

  /** Evict and dispose one asset, pinned or not. */
  unload(key: AssetKey): boolean {
    const entry = this.#cache.get(key);
    if (entry === undefined) return false;
    this.#cache.delete(key);
    this.#cacheBytes -= entry.bytes;
    disposeAsset(entry.value);
    return true;
  }

  /** Evict everything. Pinned entries survive unless `includePinned` is set. */
  clear(options: { includePinned?: boolean } = {}): void {
    for (const [key, entry] of [...this.#cache]) {
      if (entry.pinned && options.includePinned !== true) continue;
      this.#cache.delete(key);
      this.#cacheBytes -= entry.bytes;
      disposeAsset(entry.value);
    }
  }

  stats(): AssetStats {
    let pinned = 0;
    for (const entry of this.#cache.values()) if (entry.pinned) pinned++;
    return {
      entries: this.#cache.size,
      bytes: this.#cacheBytes,
      budgetBytes: this.#options.cacheBudgetBytes,
      pinned,
      hits: this.#hits,
      misses: this.#misses,
      evictions: this.#evictions,
      inFlight: this.#inFlight.size,
    };
  }

  /* -- Internals --------------------------------------------------------- */

  /**
   * Cache-aware, dedup-aware load.
   *
   * Three states are possible: resident (return it), in flight (join the
   * existing promise), or cold (start one). The in-flight map is what stops a
   * scene full of identical props from issuing a hundred parallel fetches for
   * the same file.
   */
  async #acquire(key: AssetKey, produce: () => Promise<LoadedAsset>): Promise<LoadedAsset> {
    if (this.#disposed) throw new Error('[AssetManager] used after dispose()');
    if (!Object.hasOwn(GENERATED_ASSETS, key)) {
      throw new Error(`[AssetManager] unknown asset key "${key}"`);
    }

    const cached = this.#cache.get(key);
    if (cached !== undefined) {
      this.#hits++;
      this.#touch(key, cached);
      this.#ctx?.events.emit('assets:loaded', {
        key,
        role: this.entry(key).role,
        elapsedMs: 0,
        fromCache: true,
      });
      return cached.value;
    }

    const pending = this.#inFlight.get(key);
    if (pending !== undefined) return pending;

    this.#misses++;
    this.#guardTier(key);

    const started = performance.now();
    const promise = produce()
      .then((value) => {
        const bytes =
          value instanceof THREE.Texture
            ? estimateTextureBytes(value)
            : (this.entry(key).bytes ?? 0);
        this.#cache.set(key, { key, value, bytes, pinned: false });
        this.#cacheBytes += bytes;
        this.#evictToBudget();

        this.#ctx?.events.emit('assets:loaded', {
          key,
          role: this.entry(key).role,
          elapsedMs: performance.now() - started,
          fromCache: false,
        });
        return value;
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        this.#ctx?.events.emit('assets:error', { key, message });
        throw new Error(`[AssetManager] failed to load "${key}" (${this.url(key)}): ${message}`, {
          cause: error,
        });
      })
      .finally(() => {
        this.#inFlight.delete(key);
      });

    this.#inFlight.set(key, promise);
    return promise;
  }

  /**
   * Fail fast, and explain, when a `review-required` asset is requested.
   *
   * Without this the failure surfaces as a plain 404 much later, and the reason
   * — that the asset was deliberately withheld on licensing grounds — is
   * completely invisible.
   */
  #guardTier(key: AssetKey): void {
    const entry = this.entry(key);
    if (entry.tier === 'review-required' && entry.sha256 === null) {
      throw new Error(
        `[AssetManager] "${key}" is licence-review-required and has not been fetched. ` +
          `Its redistribution rights are unresolved (see public/ATTRIBUTIONS.md). ` +
          `For local evaluation only: ` +
          `node tools/assets/fetch-assets.mjs --include-review-required`,
      );
    }
  }

  /** Move an entry to the most-recently-used end of the LRU. */
  #touch(key: AssetKey, entry: CacheEntry): void {
    this.#cache.delete(key);
    this.#cache.set(key, entry);
  }

  /** Drop least-recently-used unpinned entries until back under budget. */
  #evictToBudget(): void {
    if (this.#cacheBytes <= this.#options.cacheBudgetBytes) return;
    for (const [key, entry] of this.#cache) {
      if (this.#cacheBytes <= this.#options.cacheBudgetBytes) break;
      if (entry.pinned) continue;
      this.#cache.delete(key);
      this.#cacheBytes -= entry.bytes;
      this.#evictions++;
      disposeAsset(entry.value);
    }
  }

  /** Pick a loader by extension and load the image. */
  async #loadTextureFile(key: AssetKey): Promise<THREE.Texture> {
    const url = this.url(key);
    const extension = extensionOf(this.entry(key).path);

    if (extension === 'ktx2') {
      return this.#loadWithProgress(this.#ktx2(), url, key);
    }
    if (extension === 'exr') {
      return this.#loadWithProgress(this.#exrLoader, url, key);
    }
    if (extension === 'hdr') {
      return this.#loadWithProgress(this.#rgbeLoader, url, key);
    }
    return this.#loadWithProgress(this.#textureLoader, url, key);
  }

  /**
   * Lazily build the KTX2 loader.
   *
   * Deferred because it needs a renderer to detect supported GPU formats and
   * because instantiating the Basis transcoder costs a WASM fetch that most
   * sessions never need.
   */
  #ktx2(): KTX2Loader {
    if (this.#ktx2Loader !== null) return this.#ktx2Loader;
    const ctx = this.#ctx;
    if (ctx === null) throw new Error('[AssetManager] init() has not run yet');

    const loader = new KTX2Loader();
    if (this.ktx2TranscoderPath !== null) {
      const base = this.#options.baseUrl;
      loader.setTranscoderPath(
        `${base.endsWith('/') ? base : `${base}/`}${this.ktx2TranscoderPath}`,
      );
    }
    // `detectSupport` takes the concrete renderer; the handle's union type is
    // wider than its signature, so this is a genuine boundary cast.
    loader.detectSupport(ctx.renderer.three as unknown as THREE.WebGPURenderer);
    this.#ktx2Loader = loader;
    return loader;
  }

  /** Promisified `Loader.load` that republishes byte progress on the bus. */
  #loadWithProgress<T>(
    loader: {
      load(
        url: string,
        onLoad: (result: T) => void,
        onProgress?: (event: ProgressEvent) => void,
        onError?: (error: unknown) => void,
      ): void;
    },
    url: string,
    key: AssetKey,
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      loader.load(
        url,
        resolve,
        (event) => {
          this.#ctx?.events.emit('assets:progress', {
            key,
            loaded: event.loaded,
            total: event.total,
            fraction: event.lengthComputable && event.total > 0 ? event.loaded / event.total : -1,
          });
        },
        (error) => {
          reject(error instanceof Error ? error : new Error(String(error)));
        },
      );
    });
  }

  /** Role-derived colour space, filtering and anisotropy. Applied once, on load. */
  #configureTexture(
    texture: THREE.Texture,
    role: AssetRole,
    options: TextureLoadOptions,
  ): void {
    texture.colorSpace =
      options.colorSpace ?? (isColorRole(role) ? THREE.SRGBColorSpace : THREE.NoColorSpace);

    texture.anisotropy = options.anisotropy ?? this.#options.anisotropy;
    texture.wrapS = options.wrap ?? THREE.RepeatWrapping;
    texture.wrapT = options.wrap ?? THREE.RepeatWrapping;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = true;
    if (options.flipY !== undefined) texture.flipY = options.flipY;
    texture.needsUpdate = true;
  }

  /** Per-request tiling. Safe to re-apply to a cached texture. */
  #applyPresentation(texture: THREE.Texture, options: TextureLoadOptions): void {
    if (options.repeat === undefined) return;
    const [u, v] = typeof options.repeat === 'number'
      ? [options.repeat, options.repeat]
      : options.repeat;
    texture.repeat.set(u, v);
    texture.needsUpdate = true;
  }
}
