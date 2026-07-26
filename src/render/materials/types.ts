/**
 * @module render/materials/types
 *
 * The vocabulary of the material system: what a *substance* is, what knobs it
 * exposes, and the contracts this subsystem asks other subsystems for.
 *
 * Nothing here imports another material module, so every file in
 * `src/render/materials/` can depend on this one without creating a cycle.
 *
 * ### Why archetypes rather than "make me a material"
 *
 * Diablo II Act I is built from a small, closed set of substances: churned wet
 * mud, half-dead grass, wet cliff rock, mossy rock, rain-blackened masonry,
 * bark, weathered plank, iron (clean and rusted), leather, cloth and skin. A
 * fixed archetype table means the *art direction* — reflectance ranges, how
 * porous each substance is, how much moss it grows, how it responds to rain —
 * lives in one reviewable place instead of being re-guessed at every call site.
 * It is also what makes the wetness system tractable: "wet" means something
 * different for mud (soaks, darkens a lot, glosses a little) than for iron
 * (soaks nothing, darkens nothing, glosses a lot), and that difference is a
 * single authored porosity number per archetype.
 */

import type * as THREE from 'three/webgpu';

import type { AssetKey } from '../../assets/AssetManager';
import type { MaterialPreset } from '../../assets/Procedural';
import { serviceKey } from '../../core/ServiceLocator';

/* ------------------------------------------------------------------------- *
 * Node type aliases
 * ------------------------------------------------------------------------- */

export type FloatNode = THREE.Node<'float'>;
export type Vec2Node = THREE.Node<'vec2'>;
export type Vec3Node = THREE.Node<'vec3'>;
export type Vec4Node = THREE.Node<'vec4'>;

/* ------------------------------------------------------------------------- *
 * Contracts with modules owned by other agents
 * ------------------------------------------------------------------------- */

/**
 * Global weather state, published by whichever module owns rain and
 * time-of-day.
 *
 * The material system polls this every frame and pushes the values into its
 * shared uniforms; it never writes back. Every field is optional except
 * `wetness`, so a weather module that only models "how wet is the world"
 * satisfies the contract without inventing the rest.
 *
 * - `wetness` — 0 bone dry, 1 fully rain-soaked. Drives the whole wetness
 *   response described in {@link module:render/materials/Wetness}.
 * - `puddleLevel` — how far standing water has risen into surface cavities,
 *   in the same normalised height units as the mesoscale height map. Defaults
 *   to `wetness` when absent, which is the physically sensible coupling.
 * - `rainIntensity` — reserved for ripple animation. Currently only used to
 *   keep puddles from looking like static varnish; safe to omit.
 *
 * If no provider is registered the library falls back to
 * {@link MaterialLibraryOptions.wetness}, so the Blood Moor still reads
 * rain-soaked with no weather system present at all.
 */
export interface WeatherStateProvider {
  readonly wetness: number;
  readonly puddleLevel?: number;
  readonly rainIntensity?: number;
}

/** Service id the weather/time-of-day module should register itself under. */
export const WeatherStateKey = serviceKey<WeatherStateProvider>('world.weather');

/**
 * Optional quality governor.
 *
 * A settings or scalability module can publish a tier here and the library will
 * follow it, so the parallax step counts and anti-tiling tap counts drop
 * together with the rest of the renderer instead of being tuned separately.
 */
export interface RenderQualityProvider {
  readonly materialQuality?: MaterialQuality;
}

/** Service id for {@link RenderQualityProvider}. */
export const RenderQualityKey = serviceKey<RenderQualityProvider>('render.quality');

/* ------------------------------------------------------------------------- *
 * Archetypes
 * ------------------------------------------------------------------------- */

/**
 * The closed set of substances Act I is built from.
 *
 * `wetMud` and `wetStone` are named for their *default* state rather than being
 * separate substances from dry mud and dry stone: the wetness parameter is
 * global and continuous, and these names record what the Blood Moor looks like
 * at rest.
 */
export type MaterialArchetype =
  | 'wetMud'
  | 'deadGrass'
  | 'rock'
  | 'mossyRock'
  | 'wetStone'
  | 'bark'
  | 'plank'
  | 'woodFloor'
  | 'ironRusted'
  | 'ironClean'
  | 'leather'
  | 'cloth'
  | 'skin';

/** Every archetype, in a stable order. Iterating a union needs a real array. */
export const MATERIAL_ARCHETYPES: readonly MaterialArchetype[] = [
  'wetMud',
  'deadGrass',
  'rock',
  'mossyRock',
  'wetStone',
  'bark',
  'plank',
  'woodFloor',
  'ironRusted',
  'ironClean',
  'leather',
  'cloth',
  'skin',
] as const;

/**
 * How texture coordinates are generated.
 *
 * - `uv` — the mesh's own UV set, scaled by {@link SurfaceSpec.tiling}. Correct
 *   for anything authored with UVs: props, planks, cloth, skin.
 * - `triplanar` — world-space projection along the three axes, blended by the
 *   geometric normal. Correct for terrain and cliffs, which have no sane UV
 *   parameterisation and would otherwise show metres of vertical smear on every
 *   steep face.
 */
export type ProjectionMode = 'uv' | 'triplanar';

/**
 * How hard the system works to hide the fact that one texture tiles.
 *
 * - `off` — plain tiling. Only correct for surfaces smaller than one tile.
 * - `macro` — low-frequency world-space modulation of albedo and roughness.
 *   One noise evaluation, no extra texture fetches. Removes *tonal* repetition
 *   (the "quilt" you see across a large field) but not structural repetition.
 * - `hex` — Mikkelsen's hex-tiling on top of `macro`: three randomly offset
 *   samples on a triangular lattice, blended with a variance-preserving
 *   height blend. Removes structural repetition outright at 3x the fetches.
 *
 * `macro` is always applied; `hex` is `macro` plus the lattice.
 */
export type AntiTileMode = 'off' | 'macro' | 'hex';

/** Coarse quality tier; scales loop counts and tap counts. */
export type MaterialQuality = 'low' | 'medium' | 'high' | 'ultra';

/** Which asset keys supply each PBR channel for an archetype. */
export interface TextureSlotKeys {
  readonly albedo?: AssetKey;
  readonly normal?: AssetKey;
  readonly roughness?: AssetKey;
  readonly metalness?: AssetKey;
  readonly ao?: AssetKey;
  /**
   * Mesoscale height, used for parallax, height blending and puddle
   * accumulation. No CC0 set we can reach ships one, so it is normally derived
   * from the normal map at load time — see
   * {@link module:render/materials/HeightFromNormal}. Declared here so a real
   * set that *does* ship a height map drops in with no code change.
   */
  readonly height?: AssetKey;
}

/** Detail-layer configuration: a second UV scale that survives close inspection. */
export interface DetailSpec {
  /** Detail normal map. One of the three CC0 detail normals we ship. */
  readonly normalKey: AssetKey;
  /**
   * Optional true detail albedo, sampled at the same second UV scale.
   *
   * No CC0 detail-albedo set is reachable today, so when this is absent the
   * detail albedo grain is *derived* from the detail normal instead — see
   * {@link module:render/materials/Detail}. Declared here so that dropping a
   * real one in later is a table edit rather than a code change.
   */
  readonly albedoKey?: AssetKey;
  /**
   * Detail tiling rate, in the same units as {@link SurfaceSpec.tiling}:
   * repeats per UV unit for `uv` projection, repeats per metre for
   * `triplanar`. Typically 10-30x the base rate.
   */
  readonly tiling: number;
  /** Detail normal strength at full fade-in. */
  readonly normalStrength: number;
  /**
   * How much the detail layer perturbs albedo, as a +/- multiplier around 1.
   * Small values only: this is grain, not a second material.
   */
  readonly albedoStrength: number;
  /** Metres at which the detail layer begins to fade out. */
  readonly fadeStart: number;
  /** Metres at which it is fully gone. */
  readonly fadeEnd: number;
}

/** Low-frequency variation that breaks up large surfaces. */
export interface MacroSpec {
  /** Wavelength of the primary variation, in metres. 20-50 for terrain. */
  readonly metres: number;
  /** Peak-to-peak albedo brightness modulation, as a fraction. 0.25 = +/-12.5%. */
  readonly albedoAmount: number;
  /** Peak-to-peak roughness modulation, absolute. */
  readonly roughnessAmount: number;
  /**
   * Linear-space colour the bright lobe of the macro noise drifts towards.
   * This is where the art direction's colour separation is bought: mud drifts
   * to a rust brown, rock to a moss green, and the surface stops being one flat
   * hue without ever becoming saturated.
   */
  readonly tint: readonly [number, number, number];
  /** How far the tint is allowed to push, 0-1. */
  readonly tintAmount: number;
}

/** Parallax occlusion mapping configuration. */
export interface ParallaxSpec {
  /** Depth of the height volume in UV-space units of the *tile*. 0.02-0.08. */
  readonly scale: number;
  /** Steps at grazing incidence, before quality scaling. */
  readonly maxSteps: number;
  /** Steps at normal incidence, before quality scaling. */
  readonly minSteps: number;
  /** Metres beyond which parallax is skipped entirely. */
  readonly fadeEnd: number;
  /**
   * Discard fragments whose marched UV leaves the `[0, 1]` tile. Only correct
   * for a bounded, non-tiling patch; on tiling ground it would punch holes.
   */
  readonly clipSilhouette: boolean;
}

/**
 * A complete authored substance.
 *
 * Every number here is art direction expressed as data. The defaults live in
 * {@link module:render/materials/Archetypes}; per-instance overrides are merged
 * on top at creation time.
 */
export interface SurfaceSpec {
  readonly archetype: MaterialArchetype;
  readonly textures: TextureSlotKeys;
  /** Procedural preset used when the authored textures are unavailable. */
  readonly proceduralFallback: MaterialPreset;
  /**
   * Synthesise the base maps from {@link proceduralFallback} even when no asset
   * keys are declared.
   *
   * True for the substances no reachable CC0 set covers — leather above all.
   * False for substances that are better served by a flat authored colour plus
   * a detail normal than by a procedural approximation of something they are
   * not; skin is the example, and its real maps arrive with the character mesh.
   */
  readonly useProceduralBase: boolean;
  /** Linear-space multiplier on the sampled albedo. Grades a set into the palette. */
  readonly albedoTint: readonly [number, number, number];

  /**
   * Chroma retained in the sampled albedo, `0` grey to `1` untouched. Default 1.
   *
   * `albedoTint` cannot do this job and it is important to understand why:
   * multiplying by a cold tint darkens a warm texture but leaves it warm — a
   * limestone set that samples at (0.80, 0.70, 0.50) times a deliberately blue
   * (0.29, 0.32, 0.35) still comes out warm-dominant, because a multiply
   * preserves channel *ratios*. That is exactly how a ruined wall ends up
   * reading as warm cream in a scene whose entire palette rule is that the
   * campfire is the only warm thing in it.
   *
   * Pulling the albedo toward its own luminance first, and tinting afterwards,
   * gives the tint authority over hue instead of only over value. One `mix` and
   * one dot product per fragment.
   */
  readonly albedoSaturation?: number;
  /** Remap of the sampled roughness into `[min, max]`. */
  readonly roughnessRange: readonly [number, number];
  /** Constant metalness when no metalness map is bound. */
  readonly metalness: number;
  /**
   * Dielectric normal-incidence reflectance, *not* three's `specularIntensity`.
   * 0.04 is the everyday default; rock and stone sit slightly below, wet-prone
   * organics slightly above. Ignored when `metalness` is 1.
   */
  readonly reflectance: number;
  /** Ambient-occlusion strength, 0 = ignore the AO map. */
  readonly aoStrength: number;
  /** Strength of the base normal map. */
  readonly normalStrength: number;
  readonly projection: ProjectionMode;
  /**
   * Tiling rate. For `uv` projection: repeats per UV unit. For `triplanar`:
   * repeats per metre of world space. Keeping one field for both means the
   * archetype table reads as "how big is one tile" either way.
   */
  readonly tiling: number;
  /** Triplanar blend exponent. Higher = harder transitions between axes. */
  readonly triplanarSharpness: number;
  readonly antiTile: AntiTileMode;
  readonly detail: DetailSpec | null;
  readonly macro: MacroSpec | null;
  readonly parallax: ParallaxSpec | null;
  /**
   * How much water the substance absorbs, 0-1.
   *
   * Porosity is the single number that makes wetness read correctly per
   * substance. High porosity (mud 0.95, cloth 0.85) means water soaks in: the
   * albedo darkens a lot and very little smooth film forms, so gloss barely
   * changes. Low porosity (iron 0.02, skin 0.15) means water beads on the
   * surface: almost no darkening, but a mirror-smooth film. Getting this
   * backwards is what makes "wet" look like "varnished".
   */
  readonly porosity: number;
  /**
   * Scales how much of the global wetness this surface sees at all. Interiors,
   * undersides and cloth worn under armour should be below 1.
   */
  readonly wetnessExposure: number;
  /** Optional sheen (retroreflective fuzz) for cloth and worn leather. */
  readonly sheen: number;
  readonly sheenRoughness: number;
  /**
   * Peak clearcoat applied by the wetness system on non-porous surfaces, 0-1.
   *
   * A water film on iron or polished leather is a genuine second specular
   * interface, and clearcoat is exactly the right model for it. It is
   * *scaled by wetness*, never applied dry, so a dry surface pays nothing:
   * three only compiles the clearcoat branch when the node exists, so this is
   * `0` for every archetype that should never look lacquered.
   */
  readonly clearcoat: number;
}

/** Per-instance overrides. Deep-merged over the archetype defaults. */
export type SurfaceOverrides = {
  -readonly [K in keyof SurfaceSpec]?: SurfaceSpec[K];
};

/* ------------------------------------------------------------------------- *
 * Library options and public service shape
 * ------------------------------------------------------------------------- */

export interface MaterialLibraryOptions {
  /** Starting global wetness when no weather provider is registered. Default 0.65. */
  readonly wetness?: number;
  /** Starting quality tier. Default `'high'`. */
  readonly quality?: MaterialQuality;
  /**
   * Anisotropy for material textures. Ground planes are viewed at grazing
   * angles constantly, so this matters more here than anywhere else.
   * Defaults to the renderer maximum via the AssetManager.
   */
  readonly anisotropy?: number;
  /**
   * Archetypes to load textures for during `init`. Everything else loads
   * lazily and hot-swaps its placeholder when it arrives. Defaults to the
   * Blood Moor set.
   */
  readonly preload?: readonly MaterialArchetype[];
  /**
   * Edge length of the procedurally generated fallback textures. 256 is enough
   * to stand in convincingly at the distances a fallback is ever seen at, and
   * generating 512s for thirteen archetypes on the main thread is a visible
   * hitch. Default 256.
   */
  readonly proceduralSize?: number;
  /**
   * Resolution of the height field derived from each normal map. The derived
   * field is only ever used for parallax, blending and puddles, none of which
   * need the full 2K of the source normal. Default 256.
   */
  readonly derivedHeightSize?: number;
  /** Skip normal-map height derivation entirely (falls back to AO). Default false. */
  readonly disableHeightDerivation?: boolean;
}

/** Loop and tap counts for each quality tier. */
export interface QualityTier {
  /** Multiplier on the archetype's parallax step counts. */
  readonly parallaxStepScale: number;
  /** Whether parallax runs at all. */
  readonly parallax: boolean;
  /** Whether hex-tiling runs, or degrades to `macro`. */
  readonly hexTiling: boolean;
  /** Multiplier on detail fade distances; lower tiers fade detail out sooner. */
  readonly detailRangeScale: number;
}

export const QUALITY_TIERS: Readonly<Record<MaterialQuality, QualityTier>> = {
  low: { parallaxStepScale: 0, parallax: false, hexTiling: false, detailRangeScale: 0.5 },
  medium: { parallaxStepScale: 0.5, parallax: true, hexTiling: false, detailRangeScale: 0.75 },
  high: { parallaxStepScale: 1, parallax: true, hexTiling: true, detailRangeScale: 1 },
  ultra: { parallaxStepScale: 1.5, parallax: true, hexTiling: true, detailRangeScale: 1.35 },
};
