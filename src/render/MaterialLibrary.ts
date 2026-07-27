/**
 * @module render/MaterialLibrary
 *
 * The material system for d2rim: a TSL node-material library that turns the
 * authored archetype table in `src/render/materials/Archetypes.ts` into
 * `MeshPhysicalNodeMaterial` instances, and keeps them consistent with the
 * world's global wetness.
 *
 * ### Backends
 *
 * Everything here is TSL, and TSL runs on both of this project's backends: the
 * WebGL2 tier is three's own `WebGLBackend` behind `WebGPURenderer`, not the
 * classic `WebGLRenderer` (see `src/render/RendererFactory.ts`). There is
 * therefore exactly one shader implementation, not two, and no risk of the two
 * paths drifting. The places where the backends genuinely differ are called out
 * where they matter — most importantly the parallax loop bound, which is a
 * compile-time literal because several WebGL2 drivers will not compile a
 * dynamically bounded loop.
 *
 * ### What a surface is made of
 *
 * Every material runs the same pipeline, with stages switched off per
 * archetype:
 *
 * ```
 *   projection        uv · tiling      or  triplanar(worldPos, geoNormal)
 *   parallax          POM march, uv projection only
 *   anti-tiling       macro noise, plus hex-tiling on hero surfaces
 *   base sample       albedo + normal + packed AO/rough/height/metal
 *   detail            second UV scale, RNM-composed, distance + density faded
 *   macro variation   low-frequency albedo/roughness/tint drift
 *   wetness           absorption, film, cavity puddles, porosity-correct F0
 *   output            colour / normal / roughness / metalness / AO / specular
 * ```
 *
 * Each stage lives in its own module under `src/render/materials/`, documented
 * with the paper it comes from. This file is the assembly.
 *
 * ### Analytical performance budget
 *
 * There is no usable GPU in the development container — software rasterisation
 * on four cores — so the budget below is reasoned from first principles rather
 * than measured. Target: 60 fps at 1920×1080 on a mid-range 2020 discrete GPU
 * (RX 5600 XT / GTX 1660 Super class, ~250 GB/s, ~5 TFLOP/s fp32).
 *
 * Texture fetches per fragment, at `high` quality:
 *
 * | archetype class          | base | detail | anti-tile | total |
 * |--------------------------|------|--------|-----------|-------|
 * | UV, no POM (`cloth`)     | 3    | 1      | 0         | 4     |
 * | UV + POM (`wetStone`)    | 3    | 1      | 0         | 4 + march |
 * | UV + hex (not used)      | 3    | 1      | +6        | 10    |
 * | triplanar (`wetMud`)     | 9    | 3      | +18       | 30    |
 *
 * The packed AO/roughness/height/metalness texture is what keeps the triplanar
 * base at 9 rather than 18. Hex-tiling on a triplanar surface is the expensive
 * corner: it is applied to albedo and normal only (never to the packed scalar
 * map, whose repetition is invisible under an overcast key), which is 6 taps
 * per axis instead of 12.
 *
 * At 1080p with 2× overdraw that is ~2.1 M ground fragments × 30 fetches ≈
 * 63 M fetches per frame. All of these textures are ≤ 2K and mip-sampled with
 * high locality; on a card with ~200 GB/s of usable bandwidth and 8-16 texture
 * units per CU this is roughly 3-4 ms of the 16.6 ms budget. The POM march adds
 * up to 24 dependent fetches on masonry, but masonry is a small screen fraction
 * and the march is distance-faded to nothing past 14 m.
 *
 * ALU: the wetness stage is ~35 instructions, macro variation ~60, triplanar
 * blending ~40, hex lattice ~40 per hexed map. A triplanar hexed surface is
 * therefore ~250 ALU plus fetches — well inside the ~1500 ALU/fragment a 5 TFLOP
 * card affords at 1080p60 with 2× overdraw.
 *
 * Memory: eleven archetypes × (2K albedo 16 MB + 2K normal 16 MB + 512² packed
 * ORM 1 MB) ≈ 360 MB if every archetype is resident at 2K. The Blood Moor's
 * preload set is seven archetypes ≈ 230 MB, which is why the AssetManager's LRU
 * budget matters and why `preload` is an explicit list rather than "everything".
 * The packing step *reduces* this: without it the four scalar maps would occupy
 * 64 MB per archetype instead of 1.
 *
 * ### What this module asks of other systems
 *
 * Both are resolved through the `ServiceLocator` at runtime and both are
 * optional — see {@link WeatherStateProvider} and {@link RenderQualityProvider}
 * in `src/render/materials/types.ts`. With neither present the library runs at
 * its configured defaults and the Blood Moor still looks rain-soaked.
 */

import * as THREE from 'three/webgpu';
import {
  cameraViewMatrix,
  float,
  luminance,
  mix,
  normalMap,
  normalWorldGeometry,
  parallaxDirection,
  positionWorld,
  saturate,
  smoothstep,
  transformDirection,
  uv,
  vec3,
  vec4,
} from 'three/tsl';

import { AssetManagerKey, type AssetManager } from '../assets/AssetManager';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';

import { ARCHETYPE_SPECS, resolveSpec } from './materials/Archetypes';
import { hexFrame, hexSample, hexSampleLinear } from './materials/AntiTile';
import {
  decodeNormalNode,
  heightBlend2Node,
  reorientNormalNode,
  scaleNormalNode,
} from './materials/Blending';
import {
  detailAlbedoGainNode,
  detailFadeNode,
  detailTangentNormal,
  detailTriplanarTangentNormals,
} from './materials/Detail';
import { macroVariation, applyMacroAlbedoNode } from './materials/MacroVariation';
import { parallaxOcclusion, scaledParallaxSteps } from './materials/Parallax';
import {
  createTextureSet,
  fillTextureSet,
  type MaterialTextureSet,
  type TextureSlot,
} from './materials/TextureSets';
import {
  blendTriplanarNormals,
  triplanarFrame,
  triplanarSample,
  triplanarTangentNormals,
  withTiling,
} from './materials/Triplanar';
import {
  GRIMDARK_GROUND,
  stylizedTerrainSurface,
  type StylizedTerrainSpec,
} from './materials/StylizedTerrain';
import { applyWetness, createWetnessUniforms, type WetnessUniforms } from './materials/Wetness';
import {
  QUALITY_TIERS,
  RenderQualityKey,
  WeatherStateKey,
  type FloatNode,
  type MaterialArchetype,
  type MaterialLibraryOptions,
  type MaterialQuality,
  type SurfaceSpec,
  type Vec2Node,
  type Vec3Node,
  type Vec4Node,
} from './materials/types';

export type {
  MaterialArchetype,
  MaterialLibraryOptions,
  MaterialQuality,
  SurfaceSpec,
  RenderQualityProvider,
  WeatherStateProvider,
} from './materials/types';
export { MATERIAL_ARCHETYPES, RenderQualityKey, WeatherStateKey } from './materials/types';
export { ARCHETYPE_SPECS } from './materials/Archetypes';
export { GRIMDARK_GROUND } from './materials/StylizedTerrain';
export type { StylizedTerrainSpec, TerrainLayer } from './materials/StylizedTerrain';

/* ------------------------------------------------------------------------- *
 * Public service
 * ------------------------------------------------------------------------- */

/** A material produced by this library. */
export type SurfaceMaterial = THREE.MeshPhysicalNodeMaterial;

/** Options for a height-blended two-archetype material. */
export interface BlendedMaterialOptions {
  /** The substance underneath. */
  readonly base: MaterialArchetype;
  /** The substance layered on top. */
  readonly overlay: MaterialArchetype;
  /**
   * Coverage of the overlay in `[0, 1]`.
   *
   * Defaults to a slope mask: full overlay on ground flatter than ~25°, none on
   * anything steeper than ~50°. That default alone gives a usable terrain
   * material — grass on the flats, rock on the banks — with no terrain system
   * present, which is what lets this be tested and captured in isolation.
   * A terrain module that has real splat weights passes its own node here.
   */
  readonly weight?: FloatNode;
  /**
   * Transition band, in normalised height units. Small values (0.05) interlock
   * tightly along the mesostructure; large values (0.5) approach a linear
   * cross-fade. Default 0.15.
   */
  readonly depth?: number;
  readonly baseOverrides?: Partial<SurfaceSpec>;
  readonly overlayOverrides?: Partial<SurfaceSpec>;
}

/** Options for {@link MaterialLibraryService.createStylizedTerrain}. */
export interface StylizedTerrainOptions {
  /** Palette and blending behaviour. Defaults to {@link GRIMDARK_GROUND}. */
  readonly spec?: StylizedTerrainSpec;
  /**
   * Coverage of the `cover` layer over bare `earth`, in `[0, 1]`.
   *
   * The scenes pass their existing masks here unchanged — the encampment's
   * trodden-ground radial falloff, the moor's slope/height/track mask — so the
   * art direction survives the material swap intact.
   */
  readonly coverage?: FloatNode;
  /** Material name, for the debug inspector. */
  readonly name?: string;
}

export interface MaterialLibraryStats {
  readonly quality: MaterialQuality;
  readonly wetness: number;
  readonly puddleLevel: number;
  readonly materials: number;
  readonly textureSets: number;
  readonly settledSets: number;
  readonly weatherProvider: boolean;
}

/** The service this module registers under {@link MaterialLibraryKey}. */
export interface MaterialLibraryService {
  /** A shared, cached material for an archetype. Never mutate the result. */
  get(archetype: MaterialArchetype): SurfaceMaterial;
  /** A fresh, uncached material, optionally with overrides. Caller owns it. */
  create(archetype: MaterialArchetype, overrides?: Partial<SurfaceSpec>): SurfaceMaterial;
  /** A material that height-blends two archetypes. Caller owns it. */
  createBlended(options: BlendedMaterialOptions): SurfaceMaterial;
  /** A stylized, texture-free terrain material. Caller owns it. */
  createStylizedTerrain(options?: StylizedTerrainOptions): SurfaceMaterial;
  /** Resolves once the given archetypes' textures have settled. */
  ready(archetypes?: readonly MaterialArchetype[]): Promise<void>;
  /** The archetype table entry, after any library-level overrides. */
  spec(archetype: MaterialArchetype): SurfaceSpec;
  readonly wetness: number;
  setWetness(value: number): void;
  readonly quality: MaterialQuality;
  setQuality(value: MaterialQuality): void;
  stats(): MaterialLibraryStats;
}

/** Service id other modules resolve the library under. */
export const MaterialLibraryKey = serviceKey<MaterialLibraryService>('render.materials');

/* ------------------------------------------------------------------------- *
 * Surface construction
 * ------------------------------------------------------------------------- */

/** The shading inputs a surface produces, before they are attached to a material. */
interface SurfaceNodes {
  readonly albedo: Vec3Node;
  /** Final view-space shading normal. */
  readonly normalView: Vec3Node;
  readonly roughness: FloatNode;
  readonly metalness: FloatNode;
  readonly ao: FloatNode;
  /** Dielectric normal-incidence reflectance, after wetness. */
  readonly reflectance: FloatNode;
  readonly clearcoat: FloatNode | null;
  readonly clearcoatRoughness: FloatNode | null;
  /** Mesoscale height in `[0, 1]`, used by the two-archetype blend. */
  readonly height: FloatNode;
}

/** The scalar channels, however they happen to be stored. */
interface ScalarChannels {
  readonly ao: FloatNode;
  readonly roughnessRaw: FloatNode;
  readonly height: FloatNode;
  readonly metalnessRaw: FloatNode;
}

/**
 * Read AO, roughness, height and metalness.
 *
 * Prefers the packed texture — one fetch per projection axis instead of four —
 * and falls back to the individual slots when packing was unavailable (no
 * canvas, i.e. Node). The fallback is correct but three times the fetches on a
 * triplanar surface, which is why packing is not optional in the browser.
 */
function readScalars(
  set: MaterialTextureSet,
  sample: (slot: TextureSlot, uvNode: Vec2Node) => Vec4Node,
  uvNode: Vec2Node,
): ScalarChannels {
  if (set.orm.resolved) {
    const packed = sample(set.orm, uvNode).toVar('surfaceOrm');
    return {
      ao: packed.r,
      roughnessRaw: packed.g,
      height: packed.b,
      metalnessRaw: set.metalness.resolved ? packed.a : float(1),
    };
  }
  return {
    ao: set.ao.resolved ? sample(set.ao, uvNode).r : float(1),
    // Roughness conventionally lives in green; for the greyscale maps every
    // free set ships, all three channels agree anyway.
    roughnessRaw: set.roughness.resolved ? sample(set.roughness, uvNode).g : float(0.5),
    height: set.height.resolved ? sample(set.height, uvNode).r : float(0.5),
    metalnessRaw: set.metalness.resolved ? sample(set.metalness, uvNode).b : float(1),
  };
}

/** Build a surface whose texture coordinates come from the mesh's UV set. */
function buildUvSurface(
  spec: SurfaceSpec,
  set: MaterialTextureSet,
  detailNormal: TextureSlot | null,
  uniforms: WetnessUniforms,
  quality: MaterialQuality,
): SurfaceNodes {
  const tier = QUALITY_TIERS[quality];
  const baseUv = uv().mul(spec.tiling).toVar('surfaceUv');

  // --- parallax ----------------------------------------------------------
  const heightAvailable = set.orm.resolved || set.height.resolved;
  const doParallax = spec.parallax !== null && tier.parallax && heightAvailable;

  const sampleHeightAt = (at: Vec2Node): FloatNode =>
    set.orm.resolved ? set.orm.node.sample(at).b : set.height.node.sample(at).r;

  let sampleUv = baseUv;
  let parallaxHeight: FloatNode | null = null;
  if (doParallax && spec.parallax !== null) {
    const march = parallaxOcclusion({
      sampleHeight: sampleHeightAt,
      uv: baseUv,
      // `parallaxDirection` is `positionViewDirection * TBNViewMatrix`, i.e. the
      // view vector expressed in the tangent frame — exactly what the march
      // needs, and computed by three from the same TBN the normal map uses.
      viewTangent: parallaxDirection as Vec3Node,
      scale: float(spec.parallax.scale),
      maxSteps: scaledParallaxSteps(spec.parallax.maxSteps, tier.parallaxStepScale),
      minSteps: scaledParallaxSteps(spec.parallax.minSteps, tier.parallaxStepScale),
      fadeEnd: float(spec.parallax.fadeEnd),
      clipSilhouette: spec.parallax.clipSilhouette,
    });
    sampleUv = march.uv.toVar('surfaceUvPom');
    parallaxHeight = march.height;
  }

  // --- base sample -------------------------------------------------------
  const useHex = spec.antiTile === 'hex' && tier.hexTiling;
  const hex = useHex ? hexFrame(sampleUv, 1, 7) : null;
  const mean = vec4(
    set.albedoMean[0],
    set.albedoMean[1],
    set.albedoMean[2],
    set.albedoMean[3],
  );

  const albedoRaw =
    hex === null
      ? set.albedo.node.sample(sampleUv).xyz
      : hexSample(set.albedo.node, hex, mean).xyz;

  const baseNormalRaw =
    hex === null
      ? set.normal.node.sample(sampleUv).xyz
      : hexSampleLinear(set.normal.node, hex).xyz;

  const scalars = readScalars(set, (slot, at) => slot.node.sample(at), sampleUv);
  const heightValue = (parallaxHeight ?? scalars.height).toVar('surfaceHeight');

  // --- detail ------------------------------------------------------------
  let tangentNormal = scaleNormalNode(
    decodeNormalNode(baseNormalRaw),
    float(spec.normalStrength),
  ).toVar('surfaceTn');
  let detailGain: FloatNode = float(1);

  if (spec.detail !== null && detailNormal !== null && tier.detailRangeScale > 0) {
    const detailUv = uv().mul(spec.detail.tiling).toVar('detailUv');
    const fade = detailFadeNode(
      detailUv,
      float(spec.detail.fadeStart * tier.detailRangeScale),
      float(spec.detail.fadeEnd * tier.detailRangeScale),
    ).toVar('detailFadeAmount');
    const detailTn = detailTangentNormal(
      detailNormal.node,
      detailUv,
      float(spec.detail.normalStrength),
      fade,
    ).toVar('detailTn');
    tangentNormal = reorientNormalNode(tangentNormal, detailTn).toVar('surfaceTnDetail');
    detailGain = detailAlbedoGainNode(detailTn, float(spec.detail.albedoStrength).mul(fade));
  }

  return finishSurface({
    spec,
    uniforms,
    albedoRaw,
    detailGain,
    scalars,
    heightValue,
    normal: tangentNormal,
    flatNormal: vec3(0, 0, 1),
    // Routed through three's own `normalMap` node rather than multiplying by
    // `TBNViewMatrix` by hand. It applies the same tangent frame — derived from
    // vertex tangents when the geometry has them and from screen-space
    // derivatives when it does not — and it also handles flat shading and
    // back-face negation, which a manual multiply would silently drop. The
    // re-encode to `[0,1]` is one multiply-add and is what the node expects.
    toView: (n) => normalMap(n.mul(0.5).add(0.5)) as unknown as Vec3Node,
  });
}

/** Build a surface whose texture coordinates come from world-space projection. */
function buildTriplanarSurface(
  spec: SurfaceSpec,
  set: MaterialTextureSet,
  detailNormal: TextureSlot | null,
  uniforms: WetnessUniforms,
  quality: MaterialQuality,
): SurfaceNodes {
  const tier = QUALITY_TIERS[quality];
  const frame = triplanarFrame(
    positionWorld,
    normalWorldGeometry,
    float(spec.tiling),
    float(spec.triplanarSharpness),
  );

  const useHex = spec.antiTile === 'hex' && tier.hexTiling;
  const mean = vec4(
    set.albedoMean[0],
    set.albedoMean[1],
    set.albedoMean[2],
    set.albedoMean[3],
  );

  // Hex-tiling is applied per projection axis to albedo and normal only. The
  // packed scalar map is left on plain tiling: AO, roughness and height have no
  // memorable features for the eye to lock a lattice onto, and hexing them
  // would add nine fetches for something invisible under an overcast key.
  const hexed = (slot: TextureSlot, at: Vec2Node, variance: boolean): Vec4Node => {
    if (!useHex) return slot.node.sample(at);
    const f = hexFrame(at, 1, 7);
    return variance ? hexSample(slot.node, f, mean) : hexSampleLinear(slot.node, f);
  };

  const albedoRaw = vec3(
    hexed(set.albedo, frame.uvX, true)
      .xyz.mul(frame.weights.x)
      .add(hexed(set.albedo, frame.uvY, true).xyz.mul(frame.weights.y))
      .add(hexed(set.albedo, frame.uvZ, true).xyz.mul(frame.weights.z)),
  ).toVar('surfaceAlbedoTp');

  const scalars = readScalars(
    set,
    (slot) => triplanarSample(slot.node, frame),
    frame.uvX, // unused by the triplanar sampler; the frame carries all three
  );
  const heightValue = scalars.height.toVar('surfaceHeightTp');

  // --- normals, composed inside each projection's own tangent frame -------
  const baseTns = useHex
    ? ([
        scaleNormalNode(decodeNormalNode(hexed(set.normal, frame.uvX, false).xyz), float(spec.normalStrength)),
        scaleNormalNode(decodeNormalNode(hexed(set.normal, frame.uvY, false).xyz), float(spec.normalStrength)),
        scaleNormalNode(decodeNormalNode(hexed(set.normal, frame.uvZ, false).xyz), float(spec.normalStrength)),
      ] as [Vec3Node, Vec3Node, Vec3Node])
    : triplanarTangentNormals(set.normal.node, frame, float(spec.normalStrength));

  let composed = baseTns;
  let detailGain: FloatNode = float(1);

  if (spec.detail !== null && detailNormal !== null && tier.detailRangeScale > 0) {
    const detailFrame = withTiling(frame, spec.detail.tiling / Math.max(spec.tiling, 1e-4));
    const fade = detailFadeNode(
      detailFrame.uvY,
      float(spec.detail.fadeStart * tier.detailRangeScale),
      float(spec.detail.fadeEnd * tier.detailRangeScale),
    ).toVar('detailFadeAmountTp');
    const detailTns = detailTriplanarTangentNormals(
      detailNormal.node,
      detailFrame,
      float(spec.detail.normalStrength),
      fade,
    );
    composed = [
      reorientNormalNode(baseTns[0], detailTns[0]),
      reorientNormalNode(baseTns[1], detailTns[1]),
      reorientNormalNode(baseTns[2], detailTns[2]),
    ];
    // The detail albedo grain is taken from the dominant projection only. All
    // three agree to within the blend weights and this is a scalar multiplier,
    // so blending it would cost two multiply-adds for no visible difference.
    detailGain = detailAlbedoGainNode(
      detailTns[1],
      float(spec.detail.albedoStrength).mul(fade),
    );
  }

  const normalWorld = blendTriplanarNormals(composed, frame).toVar('surfaceNormalWorld');

  return finishSurface({
    spec,
    uniforms,
    albedoRaw,
    detailGain,
    scalars,
    heightValue,
    normal: normalWorld,
    flatNormal: normalWorldGeometry,
    toView: (n) => transformDirection(n, cameraViewMatrix).normalize(),
  });
}

interface FinishInput {
  readonly spec: SurfaceSpec;
  readonly uniforms: WetnessUniforms;
  readonly albedoRaw: Vec3Node;
  readonly detailGain: FloatNode;
  readonly scalars: ScalarChannels;
  readonly heightValue: FloatNode;
  /** Shading normal in tangent space (uv) or world space (triplanar). */
  readonly normal: Vec3Node;
  /** The un-perturbed normal in the same space, for wetness flattening. */
  readonly flatNormal: Vec3Node;
  readonly toView: (n: Vec3Node) => Vec3Node;
}

/**
 * The stages both projections share: grading, macro variation, wetness, output.
 *
 * Order matters and is the same as the physical order of events. The albedo is
 * graded and varied first, because that is the *dry* surface; wetness then acts
 * on a finished dry material, which is what it does in the world.
 */
function finishSurface(input: FinishInput): SurfaceNodes {
  const { spec, uniforms, scalars } = input;

  const tint = vec3(spec.albedoTint[0], spec.albedoTint[1], spec.albedoTint[2]);
  // Desaturate *before* the tint, so the tint decides hue rather than merely
  // scaling whatever hue the scan happened to have. See `SurfaceSpec.albedoSaturation`.
  const saturation = spec.albedoSaturation ?? 1;
  const graded =
    saturation >= 1
      ? input.albedoRaw
      : mix(vec3(luminance(input.albedoRaw)), input.albedoRaw, float(saturation));
  let albedo = graded.mul(tint).mul(input.detailGain).toVar('surfaceAlbedoDry');

  let roughness = mix(
    float(spec.roughnessRange[0]),
    float(spec.roughnessRange[1]),
    saturate(scalars.roughnessRaw),
  ).toVar('surfaceRoughnessDry');

  // Reused by the wetness stage to decide where water pools, so a surface with
  // no macro variation still gets an even waterline rather than a broken one.
  let puddleBias: FloatNode = float(0);

  if (spec.macro !== null) {
    const macro = macroVariation({
      positionWorld,
      metres: spec.macro.metres,
      albedoAmount: spec.macro.albedoAmount,
      roughnessAmount: spec.macro.roughnessAmount,
      tintAmount: spec.macro.tintAmount,
    });
    albedo = applyMacroAlbedoNode(
      albedo,
      macro.albedoGain,
      vec3(spec.macro.tint[0], spec.macro.tint[1], spec.macro.tint[2]),
      macro.tintWeight,
    ).toVar('surfaceAlbedoMacro');
    roughness = saturate(roughness.add(macro.roughnessDelta)).toVar('surfaceRoughnessMacro');
    puddleBias = macro.noise;
  }

  const ao = mix(float(1), saturate(scalars.ao), float(spec.aoStrength)).toVar('surfaceAo');

  const wet = applyWetness(
    {
      albedo,
      roughness,
      reflectance: float(spec.reflectance),
      normal: input.normal,
      flatNormal: input.flatNormal,
      // Cavity is the complement of height: 1 in the pits, where water pools.
      cavity: input.heightValue.oneMinus(),
      ao,
      porosity: float(spec.porosity),
      exposure: float(spec.wetnessExposure),
      puddleBias,
      clearcoat: spec.clearcoat,
    },
    uniforms,
  );

  return {
    albedo: wet.albedo,
    normalView: input.toView(wet.normal),
    // Roughness has a floor: a perfectly smooth GGX lobe renders a punctual
    // light as a single blown-out pixel that no amount of TAA will resolve.
    roughness: wet.roughness.max(0.02),
    metalness: float(spec.metalness).mul(saturate(scalars.metalnessRaw)),
    ao: wet.ao,
    reflectance: wet.reflectance,
    clearcoat: wet.clearcoat,
    clearcoatRoughness: wet.clearcoatRoughness,
    height: input.heightValue,
  };
}

/**
 * Attach a surface's nodes to a material.
 *
 * `specularIntensityNode` rather than `iorNode`: three derives a dielectric's
 * `F₀` as `((ior−1)/(ior+1))² · specularColor · specularIntensity`, and with the
 * material's `ior` left at its default 1.5 that first factor is exactly 0.04.
 * Dividing the wetness system's absolute reflectance by 0.04 therefore sets
 * `F₀` to precisely the value the two-interface water model computed, with no
 * inverse-Fresnel round trip and no risk of the two disagreeing.
 */
function applySurface(material: SurfaceMaterial, surface: SurfaceNodes, spec: SurfaceSpec): void {
  material.colorNode = surface.albedo;
  material.normalNode = surface.normalView;
  material.roughnessNode = surface.roughness;
  material.metalnessNode = surface.metalness;
  material.aoNode = surface.ao;
  material.ior = 1.5;
  material.specularIntensityNode = surface.reflectance.div(0.04);

  if (surface.clearcoat !== null && surface.clearcoatRoughness !== null) {
    material.clearcoatNode = surface.clearcoat;
    material.clearcoatRoughnessNode = surface.clearcoatRoughness;
  }
  if (spec.sheen > 0) {
    // Sheen is a retroreflective fuzz lobe; on cloth and worn leather it is the
    // difference between fabric and painted cardboard at grazing angles.
    material.sheenNode = vec3(spec.sheen).mul(surface.albedo.add(vec3(0.1)));
    material.sheenRoughnessNode = float(spec.sheenRoughness);
  }
}

/* ------------------------------------------------------------------------- *
 * The module
 * ------------------------------------------------------------------------- */

/** Archetypes the Blood Moor opening actually needs on screen at boot. */
const DEFAULT_PRELOAD: readonly MaterialArchetype[] = [
  'wetMud',
  'deadGrass',
  'rock',
  'mossyRock',
  'wetStone',
  'bark',
  'plank',
];

/** Detail normal keys used across the table, preloaded together. */
const DETAIL_KEYS = ['detail.normal.coarse', 'detail.normal.fine', 'detail.normal.organic'] as const;

export class MaterialLibrary implements GameModule, MaterialLibraryService {
  readonly name = 'render.materials';

  readonly #options: MaterialLibraryOptions;
  readonly #uniforms: WetnessUniforms;

  readonly #sets = new Map<MaterialArchetype, MaterialTextureSet>();
  readonly #setPromises = new Map<MaterialArchetype, Promise<void>>();
  readonly #cache = new Map<MaterialArchetype, SurfaceMaterial>();
  readonly #owned: SurfaceMaterial[] = [];
  readonly #detailSlots = new Map<string, TextureSlot>();

  #assets: AssetManager | null = null;
  #ctx: GameContext | null = null;
  #quality: MaterialQuality;
  #wetness: number;
  #puddle: number;

  constructor(options: MaterialLibraryOptions = {}) {
    this.#options = options;
    this.#quality = options.quality ?? 'high';
    this.#wetness = THREE.MathUtils.clamp(options.wetness ?? 0.65, 0, 1);
    this.#puddle = this.#wetness * 0.6;
    this.#uniforms = createWetnessUniforms(this.#wetness);
  }

  /* -- Lifecycle --------------------------------------------------------- */

  async init(ctx: GameContext): Promise<void> {
    this.#ctx = ctx;
    this.#assets = ctx.services.tryGet(AssetManagerKey) ?? null;
    ctx.services.register(MaterialLibraryKey, this);

    // Detail normals are shared by every archetype, so they are loaded once and
    // held in their own slots rather than duplicated per texture set.
    await Promise.all(DETAIL_KEYS.map(async (key) => this.#detailSlot(key)));

    const preload = this.#options.preload ?? DEFAULT_PRELOAD;
    await Promise.all(preload.map(async (archetype) => this.#ensureSet(archetype)));
  }

  update(): void {
    const ctx = this.#ctx;
    if (ctx === null) return;

    const weather = ctx.services.tryGet(WeatherStateKey);
    if (weather !== undefined) {
      this.#wetness = THREE.MathUtils.clamp(weather.wetness, 0, 1);
      this.#puddle = THREE.MathUtils.clamp(weather.puddleLevel ?? this.#wetness * 0.6, 0, 1);
      this.#uniforms.rain.value = THREE.MathUtils.clamp(weather.rainIntensity ?? 0, 0, 1);
    }
    // Written unconditionally: a uniform write is a few bytes and this keeps a
    // manual `setWetness` and a weather provider from fighting over the value.
    this.#uniforms.wetness.value = this.#wetness;
    this.#uniforms.puddleLevel.value = this.#puddle;

    const quality = ctx.services.tryGet(RenderQualityKey)?.materialQuality;
    if (quality !== undefined && quality !== this.#quality) this.setQuality(quality);
  }

  dispose(): void {
    for (const material of this.#cache.values()) material.dispose();
    for (const material of this.#owned) material.dispose();
    for (const set of this.#sets.values()) set.dispose();
    this.#cache.clear();
    this.#owned.length = 0;
    this.#sets.clear();
    this.#setPromises.clear();
    this.#detailSlots.clear();
    this.#ctx?.services.unregister(MaterialLibraryKey);
    this.#ctx = null;
    this.#assets = null;
  }

  /* -- Service ----------------------------------------------------------- */

  get(archetype: MaterialArchetype): SurfaceMaterial {
    const existing = this.#cache.get(archetype);
    if (existing !== undefined) return existing;
    const material = this.#build(resolveSpec(archetype));
    this.#cache.set(archetype, material);
    return material;
  }

  create(archetype: MaterialArchetype, overrides?: Partial<SurfaceSpec>): SurfaceMaterial {
    const material = this.#build(resolveSpec(archetype, overrides));
    this.#owned.push(material);
    return material;
  }

  /**
   * A material that height-blends two archetypes.
   *
   * Both surfaces are built in full and blended per fragment, so this costs the
   * sum of the two — which is why it exists as an explicit call rather than
   * being the default. On terrain it is the right trade: one blended material
   * removes the geometry seam, the double-draw and the sorting problem that a
   * two-layer decal approach would introduce.
   */
  createBlended(options: BlendedMaterialOptions): SurfaceMaterial {
    const baseSpec = resolveSpec(options.base, options.baseOverrides);
    const overlaySpec = resolveSpec(options.overlay, options.overlayOverrides);
    const material = new THREE.MeshPhysicalNodeMaterial();
    material.name = `d2rim.${options.base}+${options.overlay}`;

    const base = this.#surface(baseSpec);
    const overlay = this.#surface(overlaySpec);

    // Default mask: how flat the ground is. `normalWorldGeometry.y` is the
    // cosine of the slope, so 0.9 ≈ 25° and 0.64 ≈ 50°.
    const weight = (
      options.weight ?? smoothstep(float(0.64), float(0.9), normalWorldGeometry.y)
    ).toVar('blendWeight');
    const depth = float(options.depth ?? 0.15);
    const w = heightBlend2Node(base.height, overlay.height, saturate(weight), depth).toVar(
      'blendW',
    );

    material.colorNode = mix(base.albedo, overlay.albedo, w);
    material.normalNode = mix(base.normalView, overlay.normalView, w).normalize();
    material.roughnessNode = mix(base.roughness, overlay.roughness, w);
    material.metalnessNode = mix(base.metalness, overlay.metalness, w);
    material.aoNode = mix(base.ao, overlay.ao, w);
    material.ior = 1.5;
    material.specularIntensityNode = mix(base.reflectance, overlay.reflectance, w).div(0.04);

    const baseCoat = base.clearcoat ?? float(0);
    const overlayCoat = overlay.clearcoat ?? float(0);
    if (base.clearcoat !== null || overlay.clearcoat !== null) {
      material.clearcoatNode = mix(baseCoat, overlayCoat, w);
      material.clearcoatRoughnessNode = float(0.045);
    }

    this.#owned.push(material);
    return material;
  }

  /**
   * A stylized terrain material: flat colour zones, crisp boundaries, **no
   * texture fetches at all**.
   *
   * This is the ground path now. `createBlended` remains for surfaces that
   * genuinely want two photographic substances interlocking — a mossy rock face
   * against wet stone — but the 260 m ground planes are not that, and were
   * paying 24 dependent texture reads per fragment across most of the screen
   * for a result the report described as "a blurry maze-like smear".
   *
   * See `materials/StylizedTerrain` for the full diagnosis and the cost table.
   * The material still tracks the shared wetness uniforms, so weather continues
   * to move the ground, and it still responds to the tier through nothing at
   * all — there is no tier-dependent branch left in it, which is the point.
   */
  createStylizedTerrain(options: StylizedTerrainOptions = {}): SurfaceMaterial {
    const spec = options.spec ?? GRIMDARK_GROUND;
    const material = new THREE.MeshPhysicalNodeMaterial();
    material.name = options.name ?? 'd2rim.terrain.stylized';

    const surface = stylizedTerrainSurface({
      spec,
      ...(options.coverage === undefined ? {} : { coverage: options.coverage }),
      wetness: this.#uniforms,
    });

    material.colorNode = surface.albedo;
    material.roughnessNode = surface.roughness;
    material.metalnessNode = float(0);
    material.aoNode = surface.ao;
    material.ior = 1.5;
    material.specularIntensityNode = surface.reflectance.div(0.04);
    // `normalNode` is deliberately left alone. Without it three uses the
    // interpolated geometric normal, which is exactly what a stylized surface
    // wants: no tangent frame, no normal-map decode, no renormalise, and no
    // 14-repeats-per-metre detail map aliasing into a lattice.

    this.#owned.push(material);
    return material;
  }

  async ready(archetypes?: readonly MaterialArchetype[]): Promise<void> {
    const list = archetypes ?? Array.from(this.#setPromises.keys());
    await Promise.all(list.map(async (a) => this.#ensureSet(a)));
  }

  spec(archetype: MaterialArchetype): SurfaceSpec {
    return ARCHETYPE_SPECS[archetype];
  }

  get wetness(): number {
    return this.#wetness;
  }

  setWetness(value: number): void {
    this.#wetness = THREE.MathUtils.clamp(value, 0, 1);
    this.#puddle = this.#wetness * 0.6;
    this.#uniforms.wetness.value = this.#wetness;
    this.#uniforms.puddleLevel.value = this.#puddle;
  }

  get quality(): MaterialQuality {
    return this.#quality;
  }

  /**
   * Change the quality tier.
   *
   * Every cached material is rebuilt, because the tier changes the *structure*
   * of the shader — parallax on or off, hex-tiling on or off — not a uniform.
   * Rebuilding in place keeps every mesh that already references a material
   * pointing at the same object, so callers never see a stale material.
   */
  setQuality(value: MaterialQuality): void {
    if (value === this.#quality) return;
    this.#quality = value;
    for (const [archetype, material] of this.#cache) {
      this.#configure(material, resolveSpec(archetype));
      material.needsUpdate = true;
    }
  }

  stats(): MaterialLibraryStats {
    let settled = 0;
    for (const set of this.#sets.values()) if (set.settled) settled++;
    return {
      quality: this.#quality,
      wetness: this.#wetness,
      puddleLevel: this.#puddle,
      materials: this.#cache.size + this.#owned.length,
      textureSets: this.#sets.size,
      settledSets: settled,
      weatherProvider: this.#ctx?.services.has(WeatherStateKey) ?? false,
    };
  }

  /* -- Internals --------------------------------------------------------- */

  #build(spec: SurfaceSpec): SurfaceMaterial {
    const material = new THREE.MeshPhysicalNodeMaterial();
    material.name = `d2rim.${spec.archetype}`;
    this.#configure(material, spec);
    return material;
  }

  #configure(material: SurfaceMaterial, spec: SurfaceSpec): void {
    applySurface(material, this.#surface(spec), spec);
  }

  #surface(spec: SurfaceSpec): SurfaceNodes {
    const set = this.#setFor(spec.archetype);
    const detail =
      spec.detail === null ? null : (this.#detailSlots.get(spec.detail.normalKey) ?? null);
    return spec.projection === 'triplanar'
      ? buildTriplanarSurface(spec, set, detail, this.#uniforms, this.#quality)
      : buildUvSurface(spec, set, detail, this.#uniforms, this.#quality);
  }

  /**
   * The texture set for an archetype, created synchronously and filled
   * asynchronously.
   *
   * The returned set is bound to placeholders on the first call; the real
   * textures swap into the same nodes when loading completes, so a material
   * built now is correct later without being rebuilt.
   */
  #setFor(archetype: MaterialArchetype): MaterialTextureSet {
    const existing = this.#sets.get(archetype);
    if (existing !== undefined) {
      if (!this.#setPromises.has(archetype)) void this.#ensureSet(archetype);
      return existing;
    }
    const set = createTextureSet();
    this.#sets.set(archetype, set);
    void this.#ensureSet(archetype);
    return set;
  }

  async #ensureSet(archetype: MaterialArchetype): Promise<void> {
    const pending = this.#setPromises.get(archetype);
    if (pending !== undefined) return pending;

    let set = this.#sets.get(archetype);
    if (set === undefined) {
      set = createTextureSet();
      this.#sets.set(archetype, set);
    }

    const resolveOptions = {
      ...(this.#options.anisotropy === undefined ? {} : { anisotropy: this.#options.anisotropy }),
      ...(this.#options.proceduralSize === undefined
        ? {}
        : { proceduralSize: this.#options.proceduralSize }),
      ...(this.#options.derivedHeightSize === undefined
        ? {}
        : { derivedHeightSize: this.#options.derivedHeightSize }),
      ...(this.#options.disableHeightDerivation === undefined
        ? {}
        : { disableHeightDerivation: this.#options.disableHeightDerivation }),
    };

    const promise = fillTextureSet(set, this.#assets, ARCHETYPE_SPECS[archetype], resolveOptions);
    this.#setPromises.set(archetype, promise);
    return promise;
  }

  async #detailSlot(key: (typeof DETAIL_KEYS)[number]): Promise<TextureSlot | null> {
    const existing = this.#detailSlots.get(key);
    if (existing !== undefined) return existing;

    // Detail normals reuse the texture-set machinery for one slot only, which
    // keeps the placeholder-and-hot-swap behaviour identical to every other map.
    const set = createTextureSet();
    this.#detailSlots.set(key, set.normal);
    const assets = this.#assets;
    if (assets !== null && assets.has(key)) {
      try {
        const tex = await assets.loadTexture(key, { wrap: THREE.RepeatWrapping });
        (set.normal.node as { value: THREE.Texture }).value = tex;
        (set.normal as { resolved: boolean }).resolved = true;
      } catch (error) {
        console.warn(`[MaterialLibrary] detail normal "${key}" failed to load:`, error);
      }
    }
    return set.normal;
  }
}

/* ------------------------------------------------------------------------- *
 * Entry point
 * ------------------------------------------------------------------------- */

/**
 * Create the material library module.
 *
 * ```ts
 * engine.add(createMaterialLibrary({ wetness: 0.7 }));
 * // later, in any module:
 * const materials = ctx.services.get(MaterialLibraryKey);
 * mesh.material = materials.get('wetMud');
 * ```
 *
 * Register it **after** the {@link AssetManager}, whose service it resolves
 * during `init`, and before any module that asks for a material.
 */
export function createMaterialLibrary(options: MaterialLibraryOptions = {}): MaterialLibrary {
  return new MaterialLibrary(options);
}

/** Create, register with the engine, and return the library. */
export function registerMaterialLibrary(
  ctx: GameContext,
  options: MaterialLibraryOptions = {},
): MaterialLibrary {
  const library = createMaterialLibrary(options);
  ctx.engine.add(library);
  return library;
}
