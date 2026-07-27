/**
 * @module render/materials/TextureSets
 *
 * Resolves an archetype's texture slots, through {@link AssetManager} semantic
 * keys only, with a procedural fallback for every slot.
 *
 * ### The rule this module exists to enforce
 *
 * No file path ever appears in the material system. An archetype names
 * `'rock.cliff.albedo'`; the manifest decides what file that is, what colour
 * space it decodes in, and what licence it ships under. When a better CC0 rock
 * set becomes reachable, it is a manifest edit and a re-fetch — no material
 * code changes, no re-authoring, no risk of a normal map quietly being decoded
 * as sRGB by whoever wired the new set up.
 *
 * ### Placeholders and hot-swapping
 *
 * Materials must be constructible synchronously: the integrator needs
 * `library.get('wetMud')` to return a material it can put on a mesh right now,
 * not a promise. So every slot is created immediately bound to a 1×1
 * placeholder and the real texture is swapped into the same `TextureNode` when
 * it lands. Rebinding a node's `value` is the same mechanism the post stack
 * uses to re-point its passes, and it does not recompile the shader — the
 * binding is a uniform, not a define.
 *
 * ### Derived data
 *
 * Two things are computed from the decoded pixels, once, at load:
 *
 * - the **height field**, integrated from the normal map (see
 *   {@link module:render/materials/HeightFromNormal}), because no reachable
 *   CC0 set ships one and parallax, height blending and puddles all need it;
 * - the **channel means**, needed by the variance-preserving hex-tiling blend.
 *
 * Decoding needs a canvas, so it only happens in a browser. In Node — tests,
 * tooling — decoding is skipped and the consumers degrade: the height field
 * falls back to the AO map or to a constant, and hex-tiling falls back to a
 * neutral mean, which shifts where contrast is restored slightly but never
 * breaks the image.
 */

import * as THREE from 'three/webgpu';
import { texture } from 'three/tsl';

import type { AssetKey, AssetManager } from '../../assets/AssetManager';
import { generateMaterialSet, type MaterialSet } from '../../assets/Procedural';

import { channelMeans, heightFromNormalMap, heightFromOcclusion } from './HeightFromNormal';
import type { TextureSampler } from './Triplanar';
import type { SurfaceSpec } from './types';

/* ------------------------------------------------------------------------- *
 * Placeholders
 * ------------------------------------------------------------------------- */

/**
 * A 1×1 texture of a constant colour, used until the real one arrives.
 *
 * Cached per colour so that thirteen archetypes waiting on their albedo share
 * one placeholder rather than allocating thirteen GPU textures that exist for
 * two hundred milliseconds.
 */
const placeholders = new Map<string, THREE.DataTexture>();

function placeholder(r: number, g: number, b: number, a = 255): THREE.DataTexture {
  const key = `${r},${g},${b},${a}`;
  const existing = placeholders.get(key);
  if (existing !== undefined) return existing;
  const data = new Uint8Array([r, g, b, a]);
  const tex = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  placeholders.set(key, tex);
  return tex;
}

/** Neutral values per slot: flat normal, mid grey, fully unoccluded, and so on. */
const SLOT_PLACEHOLDER: Record<TextureSlotName, [number, number, number]> = {
  // White, so that an archetype with no albedo map shades at exactly its
  // authored tint rather than at the tint times an arbitrary grey — and so that
  // a surface waiting on its texture shows the right colour rather than a flash
  // of something darker.
  albedo: [255, 255, 255],
  normal: [128, 128, 255],
  roughness: [180, 180, 180],
  metalness: [0, 0, 0],
  ao: [255, 255, 255],
  height: [128, 128, 128],
  orm: [255, 180, 128],
};

/* ------------------------------------------------------------------------- *
 * Slots
 * ------------------------------------------------------------------------- */

export type TextureSlotName =
  | 'albedo'
  | 'normal'
  | 'roughness'
  | 'metalness'
  | 'ao'
  | 'height'
  /**
   * The packed surface texture: `R = AO`, `G = roughness`, `B = height`,
   * `A = metalness`.
   *
   * Built at load time from the individual maps. This is the single largest
   * cost saving in the whole material system: a triplanar surface samples every
   * map three times, so folding four scalar maps into one turns twelve fetches
   * into three. It also *reduces* texture memory, because four 2K greyscale
   * JPEGs decompress to four full RGBA8 surfaces on the GPU (three of whose
   * channels are redundant) while the packed version is one.
   *
   * Not an asset key: it has no meaning outside this system and there is no
   * point manifesting it. If a future art pipeline ships pre-packed ORM maps,
   * they slot in as a normal declared key and the packing step is skipped.
   */
  | 'orm';

/** One bound texture channel. */
export interface TextureSlot {
  readonly name: TextureSlotName;
  /** The sampler node materials reference. Stable across hot-swaps. */
  readonly node: TextureSampler;
  /** Whether a real (non-placeholder) texture is currently bound. */
  readonly resolved: boolean;
  /** Where the current texture came from, for diagnostics and the debug UI. */
  readonly source: 'asset' | 'procedural' | 'derived' | 'placeholder';
}

interface MutableSlot {
  name: TextureSlotName;
  node: TextureSampler;
  resolved: boolean;
  source: 'asset' | 'procedural' | 'derived' | 'placeholder';
}

function createSlot(name: TextureSlotName): MutableSlot {
  const rgb = SLOT_PLACEHOLDER[name];
  const node = texture(placeholder(rgb[0], rgb[1], rgb[2]));
  return { name, node, resolved: false, source: 'placeholder' };
}

/** Every channel an archetype can bind, plus the data derived from them. */
export interface MaterialTextureSet {
  readonly albedo: TextureSlot;
  readonly normal: TextureSlot;
  readonly roughness: TextureSlot;
  readonly metalness: TextureSlot;
  readonly ao: TextureSlot;
  readonly height: TextureSlot;
  /** Packed `AO / roughness / height / metalness`. See {@link TextureSlotName}. */
  readonly orm: TextureSlot;
  /**
   * Mean of the albedo texture's channels in `[0, 1]`, for the
   * variance-preserving hex blend. `[0.5, 0.5, 0.5, 1]` when undecodable.
   */
  readonly albedoMean: readonly [number, number, number, number];
  /** True once every declared slot has finished resolving, successfully or not. */
  readonly settled: boolean;
  dispose(): void;
}

interface MutableTextureSet {
  albedo: MutableSlot;
  normal: MutableSlot;
  roughness: MutableSlot;
  metalness: MutableSlot;
  ao: MutableSlot;
  height: MutableSlot;
  orm: MutableSlot;
  albedoMean: [number, number, number, number];
  settled: boolean;
  owned: THREE.Texture[];
  proceduralSet: MaterialSet | null;
  dispose(): void;
}

function createEmptySet(): MutableTextureSet {
  const owned: THREE.Texture[] = [];
  return {
    albedo: createSlot('albedo'),
    normal: createSlot('normal'),
    roughness: createSlot('roughness'),
    metalness: createSlot('metalness'),
    ao: createSlot('ao'),
    height: createSlot('height'),
    orm: createSlot('orm'),
    albedoMean: [0.5, 0.5, 0.5, 1],
    settled: false,
    owned,
    proceduralSet: null,
    dispose(): void {
      for (const t of owned) t.dispose();
      owned.length = 0;
      this.proceduralSet?.dispose();
      this.proceduralSet = null;
    },
  };
}

function bind(
  slot: MutableSlot,
  tex: THREE.Texture,
  source: 'asset' | 'procedural' | 'derived',
): void {
  slot.node.value = tex;
  slot.resolved = true;
  slot.source = source;
}

/* ------------------------------------------------------------------------- *
 * Pixel decoding
 * ------------------------------------------------------------------------- */

/** Anything that can be drawn onto a 2D canvas. */
type DrawableImage = CanvasImageSource & { width?: number; height?: number };

function canDecode(): boolean {
  return typeof document !== 'undefined' || typeof OffscreenCanvas !== 'undefined';
}

function createCanvas(size: number): { ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D } | null {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    return ctx === null ? null : { ctx };
  }
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  return ctx === null ? null : { ctx };
}

/**
 * Decode a texture's image to RGBA at `size × size`.
 *
 * The image is drawn **vertically flipped**. That is not cosmetic: three
 * uploads image-backed textures with `flipY = true`, so GPU row 0 is the
 * image's *bottom* row, while `getImageData` returns the top row first. Any
 * data derived here and handed back to the GPU as a `DataTexture` (which is
 * uploaded unflipped) would otherwise be mirrored against the map it was
 * derived from — and a height field mirrored against its own normal map
 * produces parallax that moves the wrong way, which is subtle enough to waste
 * an afternoon.
 *
 * @returns `null` in environments with no canvas, or if the image is not yet
 *          decodable.
 */
export function decodeTextureRGBA(tex: THREE.Texture, size: number): Uint8ClampedArray | null {
  if (!canDecode()) return null;
  const image = tex.image as DrawableImage | null | undefined;
  if (image === null || image === undefined) return null;

  const surface = createCanvas(size);
  if (surface === null) return null;
  const { ctx } = surface;

  try {
    ctx.save();
    ctx.translate(0, size);
    ctx.scale(1, -1);
    ctx.drawImage(image, 0, 0, size, size);
    ctx.restore();
    return ctx.getImageData(0, 0, size, size).data;
  } catch {
    // Tainted canvas, or an image that is not yet complete. Neither is fatal:
    // the consumers all have a documented degraded path.
    return null;
  }
}

/**
 * Wrap a `[0, 1]` height field in a single-channel texture.
 *
 * `RedFormat` rather than RGBA: the field is scalar, and at 256² that is 64 KB
 * against 256 KB. It is filterable on both backends.
 */
export function heightFieldToTexture(field: Float32Array, size: number): THREE.DataTexture {
  const bytes = new Uint8Array(field.length);
  for (let i = 0; i < field.length; i++) {
    bytes[i] = Math.max(0, Math.min(255, Math.round((field[i] ?? 0) * 255)));
  }
  const tex = new THREE.DataTexture(bytes, size, size, THREE.RedFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  // The decoded source was already flipped into GPU row order, so this must not
  // be flipped again. See `decodeTextureRGBA`.
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Bilinear resample of a `[0,1]` scalar field onto a square grid, wrapping at
 * the edges because these fields tile.
 */
function resampleField(
  field: Float32Array,
  fromSize: number,
  toSize: number,
): Float32Array {
  if (fromSize === toSize) return field;
  const out = new Float32Array(toSize * toSize);
  const ratio = fromSize / toSize;
  for (let y = 0; y < toSize; y++) {
    const sy = (y + 0.5) * ratio - 0.5;
    const y0 = Math.floor(sy);
    const fy = sy - y0;
    const r0 = (((y0 % fromSize) + fromSize) % fromSize) * fromSize;
    const r1 = ((((y0 + 1) % fromSize) + fromSize) % fromSize) * fromSize;
    for (let x = 0; x < toSize; x++) {
      const sx = (x + 0.5) * ratio - 0.5;
      const x0 = Math.floor(sx);
      const fx = sx - x0;
      const c0 = (((x0 % fromSize) + fromSize) % fromSize);
      const c1 = ((((x0 + 1) % fromSize) + fromSize) % fromSize);
      const a = (field[r0 + c0] ?? 0) * (1 - fx) + (field[r0 + c1] ?? 0) * fx;
      const b = (field[r1 + c0] ?? 0) * (1 - fx) + (field[r1 + c1] ?? 0) * fx;
      out[y * toSize + x] = a * (1 - fy) + b * fy;
    }
  }
  return out;
}

/**
 * Build the packed `AO / roughness / height / metalness` texture.
 *
 * @returns `null` when nothing could be decoded, in which case the caller keeps
 *          sampling the individual slots.
 *
 * Channel assignment follows the glTF `occlusionRoughnessMetallic` convention
 * for R and G so that a pre-packed ORM map from a real art pipeline drops
 * straight in; height takes B (glTF puts metalness there, which is moved to A)
 * because height is the channel this project always has and metalness is the
 * one it almost never does.
 */
export function packSurfaceTexture(
  sources: {
    readonly ao: THREE.Texture | null;
    readonly roughness: THREE.Texture | null;
    readonly metalness: THREE.Texture | null;
    readonly height: Float32Array | null;
    readonly heightSize: number;
  },
  size: number,
): THREE.DataTexture | null {
  const ao = sources.ao === null ? null : decodeTextureRGBA(sources.ao, size);
  const rough = sources.roughness === null ? null : decodeTextureRGBA(sources.roughness, size);
  const metal = sources.metalness === null ? null : decodeTextureRGBA(sources.metalness, size);
  const height =
    sources.height === null ? null : resampleField(sources.height, sources.heightSize, size);

  if (ao === null && rough === null && metal === null && height === null) return null;

  const count = size * size;
  const data = new Uint8Array(count * 4);
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    data[o] = ao === null ? 255 : (ao[o] ?? 255);
    data[o + 1] = rough === null ? 180 : (rough[o + 1] ?? 180);
    data[o + 2] =
      height === null ? 128 : Math.max(0, Math.min(255, Math.round((height[i] ?? 0.5) * 255)));
    data[o + 3] = metal === null ? 0 : (metal[o + 2] ?? 0);
  }

  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  // Sources were decoded in GPU row order; do not flip again.
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/* ------------------------------------------------------------------------- *
 * Resolution
 * ------------------------------------------------------------------------- */

export interface ResolveOptions {
  readonly anisotropy?: number;
  /** Edge length of generated procedural fallbacks. */
  readonly proceduralSize?: number;
  /** Edge length of the derived height field. */
  readonly derivedHeightSize?: number;
  /** Edge length of the packed ORM texture. */
  readonly packedSize?: number;
  /** Skip normal-map integration; the height slot then falls back to AO. */
  readonly disableHeightDerivation?: boolean;
  /** Skip ORM packing; materials then sample the individual scalar maps. */
  readonly disablePacking?: boolean;
}

const SLOT_KEYS: readonly (keyof SurfaceSpec['textures'])[] = [
  'albedo',
  'normal',
  'roughness',
  'metalness',
  'ao',
  'height',
];

/**
 * Load every declared slot, fill the gaps, and derive what is missing.
 *
 * Order of preference per slot:
 *   1. the archetype's declared asset key;
 *   2. the procedural fallback set, if the archetype asked for one or if a
 *      declared key failed to load;
 *   3. the neutral placeholder, which is a valid material — a flat normal, mid
 *      roughness and white AO shade correctly, they are just featureless.
 *
 * Failures are logged and swallowed. A missing texture must never prevent the
 * world from rendering: the whole point of the fallback chain is that the game
 * still boots on a broken asset checkout.
 */
export async function fillTextureSet(
  target: MaterialTextureSet,
  assets: AssetManager | null,
  spec: SurfaceSpec,
  options: ResolveOptions = {},
): Promise<void> {
  const set = target as MutableTextureSet;
  const proceduralSize = options.proceduralSize ?? 256;
  const derivedSize = options.derivedHeightSize ?? 256;

  const packedSize = options.packedSize ?? 512;

  const slots: Record<TextureSlotName, MutableSlot> = {
    albedo: set.albedo,
    normal: set.normal,
    roughness: set.roughness,
    metalness: set.metalness,
    ao: set.ao,
    height: set.height,
    orm: set.orm,
  };

  const declared = spec.textures;
  const wanted: Array<[keyof SurfaceSpec['textures'], AssetKey]> = [];
  for (const name of SLOT_KEYS) {
    const key = declared[name];
    if (key !== undefined) wanted.push([name, key]);
  }

  const loadOptions =
    options.anisotropy === undefined
      ? { wrap: THREE.RepeatWrapping as THREE.Wrapping }
      : { wrap: THREE.RepeatWrapping as THREE.Wrapping, anisotropy: options.anisotropy };

  const results = await Promise.all(
    wanted.map(async ([name, key]) => {
      if (assets === null || !assets.has(key)) return { name, texture: null };
      try {
        return { name, texture: await assets.loadTexture(key, loadOptions) };
      } catch (error) {
        console.warn(`[MaterialLibrary] "${spec.archetype}" slot "${name}" (${key}) failed:`, error);
        return { name, texture: null };
      }
    }),
  );

  let heightField: Float32Array | null = null;
  let heightFieldSize = derivedSize;

  let anyFailed = false;
  for (const { name, texture: tex } of results) {
    if (tex === null) {
      anyFailed = true;
      continue;
    }
    const slot = slots[name];
    bind(slot, tex, 'asset');
  }

  // A procedural set is generated when the archetype asked for one outright, or
  // to repair a partially failed load. Generating it is a few hundred
  // milliseconds of main-thread work at 256², so it is never done speculatively.
  const needProcedural = (wanted.length === 0 && spec.useProceduralBase) || anyFailed;
  if (needProcedural) {
    try {
      const generated = generateMaterialSet(spec.proceduralFallback, {
        size: proceduralSize,
        seed: `d2rim.${spec.archetype}`,
      });
      set.proceduralSet = generated;
      if (!slots.albedo.resolved) bind(slots.albedo, generated.map, 'procedural');
      if (!slots.normal.resolved) bind(slots.normal, generated.normalMap, 'procedural');
      if (!slots.roughness.resolved) bind(slots.roughness, generated.roughnessMap, 'procedural');
      if (!slots.ao.resolved) bind(slots.ao, generated.aoMap, 'procedural');
      if (!slots.height.resolved) {
        heightField = generated.height;
        heightFieldSize = generated.size;
        const tex = heightFieldToTexture(generated.height, generated.size);
        set.owned.push(tex);
        bind(slots.height, tex, 'derived');
      }
    } catch (error) {
      console.warn(`[MaterialLibrary] procedural fallback for "${spec.archetype}" failed:`, error);
    }
  }

  // Derive the height field from the normal map when nothing has supplied one.
  if (!slots.height.resolved && slots.normal.resolved && options.disableHeightDerivation !== true) {
    const decoded = decodeTextureRGBA(slots.normal.node.value, derivedSize);
    if (decoded !== null) {
      heightField = heightFromNormalMap(decoded, derivedSize, derivedSize, { sweeps: 8 });
      heightFieldSize = derivedSize;
      const tex = heightFieldToTexture(heightField, derivedSize);
      set.owned.push(tex);
      bind(slots.height, tex, 'derived');
    }
  }

  // Last resort for the height slot: invert AO. Documented as a poor proxy in
  // HeightFromNormal, but strictly better than a constant for the puddle mask.
  if (!slots.height.resolved && slots.ao.resolved) {
    const decoded = decodeTextureRGBA(slots.ao.node.value, derivedSize);
    if (decoded !== null) {
      heightField = heightFromOcclusion(decoded, derivedSize, derivedSize);
      heightFieldSize = derivedSize;
      const tex = heightFieldToTexture(heightField, derivedSize);
      set.owned.push(tex);
      bind(slots.height, tex, 'derived');
    }
  }

  // Fold the scalar maps into one RGBA texture. Failure is fine: the builder
  // falls back to sampling the individual slots, which is correct but costs
  // three extra fetches per projection axis.
  if (options.disablePacking !== true) {
    const packed = packSurfaceTexture(
      {
        ao: slots.ao.resolved ? slots.ao.node.value : null,
        roughness: slots.roughness.resolved ? slots.roughness.node.value : null,
        metalness: slots.metalness.resolved ? slots.metalness.node.value : null,
        height: heightField,
        heightSize: heightFieldSize,
      },
      packedSize,
    );
    if (packed !== null) {
      set.owned.push(packed);
      bind(slots.orm, packed, 'derived');
    }
  }

  if (slots.albedo.resolved) {
    const decoded = decodeTextureRGBA(slots.albedo.node.value, 64);
    if (decoded !== null) set.albedoMean = channelMeans(decoded, 64, 64);
  }

  set.settled = true;
}

/**
 * A set bound entirely to placeholders.
 *
 * Materials are built against this immediately and {@link fillTextureSet}
 * swaps the real textures into the same nodes when they arrive, which is what
 * lets `MaterialLibrary.get()` stay synchronous.
 */
export function createTextureSet(): MaterialTextureSet {
  return createEmptySet();
}

/** Create and fill in one step, for callers that can await. */
export async function resolveTextureSet(
  assets: AssetManager | null,
  spec: SurfaceSpec,
  options: ResolveOptions = {},
): Promise<MaterialTextureSet> {
  const set = createTextureSet();
  await fillTextureSet(set, assets, spec, options);
  return set;
}
