/**
 * @module render/materials/StylizedTerrain
 *
 * A deliberately stylized ground material: flat authored colour zones, crisp
 * noise-broken boundaries, geometric normals, **zero texture fetches**.
 *
 * ## Why the realistic terrain was replaced
 *
 * The ground was a two-archetype height-blend of `wetMud` and `deadGrass`, each
 * built as a full triplanar PBR surface. Per fragment that is:
 *
 * | | per surface | blended |
 * |---|---|---|
 * | albedo (3 axes) | 3 | 6 |
 * | packed ORM (3 axes) | 3 | 6 |
 * | base normal (3 axes) | 3 | 6 |
 * | detail normal (3 axes) | 3 | 6 |
 * | **texture fetches** | **12** | **24** |
 *
 * plus two macro-variation fBm evaluations per surface, two full wetness
 * models, two tangent-frame reconstructions and two whiteout normal blends.
 * Twenty-four dependent fetches across most of the screen, on an integrated
 * GPU, at 1.9 megapixels. That was the frame.
 *
 * And it did not even look right. The reported symptom was "a blurry maze-like
 * smear that does not read as a material", and the terrain shader had at least
 * three independent ways to produce exactly that:
 *
 * 1. **The detail normal ran at 14 repeats per metre.** `GROUND_DETAIL.tiling`
 *    is 14 against a base tiling of 0.45, and the detail frame is derived as
 *    `detail.tiling / spec.tiling` — 31× the base frame, i.e. a full texture
 *    period every 7 cm of world space. At a third-person camera distance that
 *    is far below one period per pixel, so every pixel samples an essentially
 *    random point of the normal map. The mip chain cannot help because the
 *    tangent-space normals are being *renormalised* after filtering. The result
 *    is high-frequency directional hash — which is what a "maze" looks like.
 * 2. **Block-compressed normal maps.** ETC1S is a chroma-subsampled,
 *    4×4-block, RGB codec designed for photographs. A tangent-space normal map
 *    is not a photograph: its three channels are geometrically coupled, and the
 *    codec's per-block endpoint fitting introduces exactly the blocky,
 *    directionally-biased artefacts that survive renormalisation and read as a
 *    lattice.
 * 3. **Triplanar at 0.45 tiles/m over a near-flat plane.** Nearly all the
 *    weight sits on the Y plane, so two of the three sample sets are computed
 *    and multiplied by ~0, and the surviving one is a 2.2 m tile stretched over
 *    a 260 m field — the "quilt" that macro variation exists to hide and can
 *    only partly hide.
 *
 * Fixing all three would have produced a slightly-less-broken realistic
 * material that still cost 24 fetches while sitting under low-poly stylized
 * geometry that cannot carry photoscanned detail anyway. So this replaces the
 * approach rather than patching it.
 *
 * ## What this does instead
 *
 * Three authored layers — bare earth, dead cover, exposed rock — selected by
 * *slope* and by a caller-supplied coverage mask, with the boundaries broken up
 * and then **re-sharpened** by noise. That last step is the whole look:
 *
 * ```
 * w' = smoothstep(0.5 - k, 0.5 + k, w + noise * amplitude)
 * ```
 *
 * Perturbing the weight by noise makes the boundary wander; running it back
 * through a narrow `smoothstep` makes the boundary *crisp*. A soft cross-fade
 * between two brown-greys is mud — literally and visually. A hard, irregular
 * edge between two brown-greys reads as two materials meeting, which is what
 * hand-painted terrain in this genre actually does. Torchlight and the Diablo
 * III cinematics both lean on this: the colour count is tiny and the shapes do
 * the work.
 *
 * On top of the layer selection there are exactly two more terms, both free:
 *
 * - **Macro tone drift** at ~30 m, so a 260 m field is not one flat colour.
 * - **Slope cavity darkening**, so the terrain's own silhouette reads without
 *   any normal map at all. Steep faces get darker and rougher; this is the
 *   cheapest possible substitute for ambient occlusion and it is the term that
 *   makes flat colour look sculpted rather than painted on.
 *
 * Wetness is kept, because the act's tone depends on it, but reduced to a
 * two-line response — darken by porosity, drop roughness — instead of the full
 * two-interface Fresnel water model. On a surface with no mesoscale height
 * there is nothing for the puddle model to pool into anyway.
 *
 * ## Cost
 *
 * | | old (blended triplanar) | new (stylized) |
 * |---|---|---|
 * | texture fetches | 24 | **0** |
 * | fBm evaluations | 4 (2 per surface, 3+2 octaves) | **2** (2 octaves each) |
 * | tangent frames | 2 | 0 |
 * | normal map decodes + whiteout blends | 4 + 2 | 0 |
 * | wetness models | 2 (full) | 1 (reduced) |
 * | varyings needed | world pos, geometric normal, tangent | world pos, geometric normal |
 *
 * Zero dependent texture reads is the number that matters: on a tiled
 * integrated GPU the terrain fragment shader stops touching the texture unit
 * entirely, and the 189 MB of resident terrain texture stops being sampled
 * every frame across most of the screen.
 *
 * ## Colour authoring
 *
 * Layer colours are authored as sRGB hex, the way an artist would pick them,
 * and converted once on the CPU. Under three's colour management a hex literal
 * handed to TSL is decoded to the linear working space, so the authored value
 * is what a colour picker shows — not a number that has to be pre-divided by
 * 2.2 by hand and is therefore never reviewed.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  float,
  mix,
  mx_fractal_noise_float,
  normalWorldGeometry,
  positionWorld,
  saturate,
  smoothstep,
  vec3,
} from 'three/tsl';

import type { FloatNode, Vec3Node } from './types';
import type { WetnessUniforms } from './Wetness';

/* ------------------------------------------------------------------------- *
 * Palette
 * ------------------------------------------------------------------------- */

/** One flat colour zone. */
export interface TerrainLayer {
  /** sRGB hex, as a colour picker shows it. Converted to linear once, on the CPU. */
  readonly color: number;
  /** Flat roughness for the layer. Terrain is never smooth; keep this high. */
  readonly roughness: number;
  /**
   * How far macro noise is allowed to drift this layer's brightness, ±, as a
   * fraction. 0.12 is a visible but unobtrusive tonal wander over ~30 m.
   */
  readonly variation: number;
  /**
   * How much this layer soaks up water, 0-1. Earth soaks and darkens a lot;
   * rock barely darkens but glosses. Same meaning as `SurfaceSpec.porosity`.
   */
  readonly porosity: number;
}

/** The full authored description of a stylized ground surface. */
export interface StylizedTerrainSpec {
  /** Bare earth: trodden paths, churned mud, the default surface. */
  readonly earth: TerrainLayer;
  /** Ground cover: dead grass, scrub, whatever grows where feet do not go. */
  readonly cover: TerrainLayer;
  /** Exposed rock, selected by slope. */
  readonly rock: TerrainLayer;
  /**
   * Boundary crispness, 0-1. 0 is a linear cross-fade (mud); 1 is a hard
   * two-pixel edge (a cel-shaded decal). ~0.75 is the Torchlight look: clearly
   * a boundary, still soft enough to survive a moving camera without crawling.
   */
  readonly crispness: number;
  /** How far noise is allowed to push a boundary around, in weight units. */
  readonly boundaryNoise: number;
  /** Wavelength of the boundary-breaking noise, in metres. */
  readonly boundaryMetres: number;
  /** Wavelength of the large-scale tone drift, in metres. */
  readonly macroMetres: number;
  /**
   * Mid-scale mottling depth, ±, as a fraction of albedo.
   *
   * Reuses the boundary noise that has already been evaluated, so it is two
   * instructions. Without it a layer that covers the whole visible ground —
   * the inside of the palisade is all bare earth — renders as one flat colour
   * over the entire floor of the frame, which reads as untextured rather than
   * as stylized. This is the term that says "mud", at 5 m scale, for free.
   */
  readonly mottle: number;
  /** Surface normal Y above which no rock shows through. */
  readonly rockSlopeStart: number;
  /** Surface normal Y below which the surface is entirely rock. */
  readonly rockSlopeEnd: number;
  /**
   * Strength of slope-based darkening, 0-1.
   *
   * The single term that makes flat colour read as sculpted terrain. Without a
   * normal map, a hillside and a flat both return the same albedo and only the
   * direct lighting distinguishes them — which, under an overcast sky that is
   * one enormous ambient source, is almost nothing.
   */
  readonly slopeShading: number;
}

/**
 * The act's ground: cold, rained-on, mostly dead.
 *
 * Every value is desaturated and dark — reflectances between 0.04 and 0.09
 * linear. That is deliberate headroom: the encampment is lit by bonfires, and a
 * warm point light landing on a cold near-neutral albedo reads as firelight. A
 * ground already carrying its own warm brown has nowhere to go and the fires
 * stop reading as light sources at all.
 */
export const GRIMDARK_GROUND: StylizedTerrainSpec = {
  // Cold umber. Brown enough to be earth, blue enough to sit under an overcast
  // sky without looking like it is lit by a sunset.
  earth: { color: 0x38322a, roughness: 0.93, variation: 0.2, porosity: 0.95 },
  // Dead khaki-olive. The green is almost entirely gone; what is left is the
  // yellow of dry stalks, knocked back hard.
  cover: { color: 0x524c33, roughness: 0.96, variation: 0.22, porosity: 0.85 },
  // Wet slate. Cooler and lighter than the earth so banks and cuts separate
  // from the flats they sit in.
  rock: { color: 0x4b4d51, roughness: 0.82, variation: 0.12, porosity: 0.25 },
  crispness: 0.72,
  boundaryNoise: 0.42,
  boundaryMetres: 5.5,
  macroMetres: 31,
  mottle: 0.17,
  rockSlopeStart: 0.82,
  rockSlopeEnd: 0.55,
  slopeShading: 0.55,
};

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/**
 * The crisping curve, on the CPU, so it can be tested without a GPU.
 *
 * `crispness` maps to the half-width of the `smoothstep` band: 0 gives a band
 * of half-width 0.5 (a straight linear ramp across the whole weight range) and
 * 1 gives 0.02 (two pixels of gradient). The mapping is deliberately non-linear
 * at the top so that the interesting range — 0.6 to 0.9 — has resolution.
 *
 * @param weight raw coverage in `[0, 1]`
 * @param crispness `[0, 1]`
 * @returns the sharpened weight, in `[0, 1]`
 */
export function crispWeight(weight: number, crispness: number): number {
  const k = crispBand(crispness);
  const lo = 0.5 - k;
  const hi = 0.5 + k;
  if (weight <= lo) return 0;
  if (weight >= hi) return 1;
  const t = (weight - lo) / (hi - lo);
  return t * t * (3 - 2 * t);
}

/** Half-width of the crisping band for a given crispness. Never zero. */
export function crispBand(crispness: number): number {
  const c = Math.min(1, Math.max(0, crispness));
  return 0.5 * (1 - c) ** 1.6 + 0.02;
}

/**
 * Slope-driven rock coverage, on the CPU.
 *
 * `normalY` is the world-space geometric normal's Y component: 1 on a flat,
 * 0 on a vertical face. Returns 1 where the surface is entirely rock.
 */
export function rockCoverage(normalY: number, start: number, end: number): number {
  // `start` is the *flatter* threshold and therefore the upper bound, so the
  // smoothstep runs backwards: coverage rises as the normal tips over.
  if (start <= end) return normalY <= end ? 1 : 0;
  const t = Math.min(1, Math.max(0, (start - normalY) / (start - end)));
  return t * t * (3 - 2 * t);
}

/**
 * Slope darkening factor, on the CPU. 1 on a flat, less on a slope.
 *
 * Squared rather than linear in the slope so that gentle undulation is barely
 * touched and only real banks darken. A linear falloff greys out the whole
 * field, which is the "grey mush" the art direction forbids.
 */
export function slopeShade(normalY: number, strength: number): number {
  const tip = Math.min(1, Math.max(0, 1 - Math.min(1, Math.max(0, normalY))));
  return 1 - tip * tip * Math.min(1, Math.max(0, strength));
}

/* ------------------------------------------------------------------------- *
 * TSL
 * ------------------------------------------------------------------------- */

/** Convert an authored sRGB hex to a linear-working-space TSL constant. */
function linearColor(hex: number): Vec3Node {
  const c = new THREE.Color();
  c.setHex(hex, THREE.SRGBColorSpace);
  return vec3(c.r, c.g, c.b);
}

/**
 * The crisping curve as a node. Mirrors {@link crispWeight}.
 *
 * A `Fn` with a declared layout so it compiles to one function call rather than
 * being inlined three times.
 */
const crispNode = /*@__PURE__*/ Fn(([weight, band]: [FloatNode, FloatNode]) =>
  smoothstep(float(0.5).sub(band), float(0.5).add(band), weight),
).setLayout({
  name: 'terrainCrisp',
  type: 'float',
  inputs: [
    { name: 'weight', type: 'float' },
    { name: 'band', type: 'float' },
  ],
});

/** What {@link stylizedTerrainSurface} produces. */
export interface StylizedTerrainNodes {
  readonly albedo: Vec3Node;
  readonly roughness: FloatNode;
  readonly ao: FloatNode;
  /** Dielectric normal-incidence reflectance after wetness. */
  readonly reflectance: FloatNode;
}

export interface StylizedTerrainOptions {
  readonly spec: StylizedTerrainSpec;
  /**
   * Coverage of the `cover` layer over `earth`, in `[0, 1]`.
   *
   * This is where the scene's own art direction enters: the encampment passes a
   * radial mask so the ground is trodden bare inside the palisade, and the moor
   * passes a slope/height/track mask. Omitted, the whole field is `cover`,
   * which is a valid look on its own.
   */
  readonly coverage?: FloatNode;
  /** Shared wetness uniforms. Omitted, the surface is authored-dry. */
  readonly wetness?: WetnessUniforms;
}

/**
 * Build the stylized ground surface.
 *
 * No texture fetch, no tangent frame, no normal map. The mesh's own geometric
 * normal does all the shading, which is why this composes with low-poly
 * displaced terrain rather than fighting it.
 */
export function stylizedTerrainSurface(
  options: StylizedTerrainOptions,
): StylizedTerrainNodes {
  const { spec, coverage, wetness } = options;

  const band = float(crispBand(spec.crispness));
  const normalY = normalWorldGeometry.y as unknown as FloatNode;

  // -- noise ---------------------------------------------------------------
  //
  // Two fBm evaluations, total, for the whole material. The first breaks
  // boundaries, the second drifts tone. They are deliberately an order of
  // magnitude apart in wavelength so neither can beat against the other.
  const boundary = mx_fractal_noise_float(
    positionWorld.mul(1 / Math.max(spec.boundaryMetres, 0.5)),
    2,
    2.0,
    0.5,
  ).toVar('terrainBoundaryNoise') as FloatNode;

  const macro = mx_fractal_noise_float(
    positionWorld.mul(1 / Math.max(spec.macroMetres, 1)).add(vec3(11.7, 3.1, 23.9)),
    2,
    2.0,
    0.55,
  ).toVar('terrainMacroNoise') as FloatNode;

  // -- layer selection -----------------------------------------------------

  // Cover over earth. Perturb, then re-sharpen — see the module header; this
  // pair of operations is the entire difference between "two colours smeared
  // together" and "two materials meeting".
  const coverRaw = (coverage ?? float(1)).add(boundary.mul(spec.boundaryNoise));
  const coverW = crispNode(saturate(coverRaw), band).toVar('terrainCoverWeight');

  // Rock by slope. Its boundary is broken by the same noise at a different
  // amplitude, so cliff edges are ragged rather than following an iso-slope
  // contour, which is the tell-tale of procedural terrain.
  const rockRaw = smoothstep(
    float(spec.rockSlopeStart),
    float(spec.rockSlopeEnd),
    normalY,
  ).add(boundary.mul(spec.boundaryNoise * 0.5));
  const rockW = crispNode(saturate(rockRaw), band).toVar('terrainRockWeight');

  // -- albedo --------------------------------------------------------------

  const earthColor = linearColor(spec.earth.color);
  const coverColor = linearColor(spec.cover.color);
  const rockColor = linearColor(spec.rock.color);

  // Per-layer tone drift, folded into one multiply. The variation amounts
  // differ per layer (rock varies least; it is stone, not soil), so the gain is
  // itself a mix before it is applied.
  const variation = mix(
    mix(float(spec.earth.variation), float(spec.cover.variation), coverW),
    float(spec.rock.variation),
    rockW,
  );
  // Two scales of tonal drift folded into one gain: ~31 m from the macro band,
  // ~5.5 m from the boundary band. The second is the one that makes a single
  // uniform layer read as a material rather than as a fill colour, and it is
  // free because the boundary noise was already evaluated for the layer edges.
  const tone = float(1)
    .add(macro.mul(variation))
    .add(boundary.mul(spec.mottle))
    .max(0.15);

  const layered = mix(mix(earthColor, coverColor, coverW), rockColor, rockW);

  // Slope darkening: the substitute for an AO map and a normal map at once.
  const tip = saturate(float(1).sub(normalY));
  const shade = float(1).sub(tip.mul(tip).mul(spec.slopeShading)).toVar('terrainSlopeShade');

  const dryAlbedo = layered.mul(tone).mul(shade).toVar('terrainAlbedo');

  // -- roughness -----------------------------------------------------------

  const dryRoughness = mix(
    mix(float(spec.earth.roughness), float(spec.cover.roughness), coverW),
    float(spec.rock.roughness),
    rockW,
  )
    // A little noise in roughness is what stops a flat-coloured surface from
    // reading as plastic under a moving light. It costs one multiply-add.
    .add(macro.mul(0.05))
    .clamp(0.08, 1);

  const porosity = mix(
    mix(float(spec.earth.porosity), float(spec.cover.porosity), coverW),
    float(spec.rock.porosity),
    rockW,
  );

  // -- wetness (reduced) ---------------------------------------------------
  //
  // Not the two-interface Fresnel model in `Wetness.ts`. That model earns its
  // cost on a surface with mesoscale height to pool water into; this surface
  // has none, so the whole puddle branch would evaluate to a constant. What
  // survives is the part that actually carries the tone: soaked ground is
  // darker and glossier.
  if (wetness === undefined) {
    return {
      albedo: dryAlbedo,
      roughness: dryRoughness,
      ao: shade,
      reflectance: float(0.04),
    };
  }

  const soak = saturate(wetness.wetness.mul(porosity)).toVar('terrainSoak');
  return {
    // Beer-Lambert, linearised: a soaked surface absorbs more on the way back
    // out, so it darkens towards its own colour cubed rather than towards grey.
    albedo: dryAlbedo.mul(mix(float(1), float(0.62), soak)),
    // Water fills the microstructure. 0.12 rather than the physical 0.045 of
    // flat water, because a stylized surface with a mirror-sharp lobe picks up
    // exactly the specular speckle the weather defaults were tuned to avoid.
    roughness: mix(dryRoughness, float(0.12), soak.mul(0.65)).clamp(0.08, 1),
    ao: shade,
    // Wet dielectric surfaces reflect more at normal incidence. 0.04 dry to
    // 0.055 soaked is the visible range; beyond that the ground reads as vinyl.
    reflectance: mix(float(0.04), float(0.055), soak),
  };
}
