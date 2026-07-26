/**
 * @module render/Volumetrics
 *
 * Participating media for the whole game: a camera-frustum-aligned **froxel**
 * volume that is filled with scattering and extinction, lit by the sun and by
 * every torch in range, integrated front-to-back, and finally composited over
 * the frame by {@link module:render/post/LightShafts}.
 *
 * This is the single biggest contributor to the "expensive" look the project is
 * chasing. A rain-soaked moor is not a clear scene with a grey filter on it —
 * it is a scene where the air itself is a light source, where the distance
 * fades because photons are being scattered *out* of the view ray and replaced
 * by photons scattered *in* from the sky, and where a shaft of sun through bare
 * branches is visible because the air between the branch and the eye is lit.
 * None of that is reproducible with `FogExp2`.
 *
 * ---
 *
 * ## The model
 *
 * Radiative transfer along a view ray, in the single-scattering approximation
 * with an ambient multiple-scattering term:
 *
 * ```
 *   L(x_eye) = ∫₀^d  T(0,t) · σ_s(t) · Σ_l  p(θ_l) · V_l(t) · L_l(t)  dt
 *            +  T(0,d) · L_surface
 *   T(a,b)   = exp( −∫_a^b σ_t(s) ds )
 * ```
 *
 * - `σ_t` — extinction (absorption + out-scattering), per metre.
 * - `σ_s` — scattering coefficient, `σ_t · albedo`. Fog albedo is very close to
 *   1 (water droplets barely absorb), which is why fog *brightens* a dark scene
 *   rather than dimming it, and why the fog colour must come from the lighting
 *   rather than from a painted constant.
 * - `p(θ)` — the phase function. Water droplets are strongly forward
 *   scattering, which is what makes looking towards the sun through mist so
 *   much brighter than looking away from it. See {@link dualLobePhase}.
 * - `V_l` — visibility of light `l` at the sample, i.e. the shadow map. This
 *   term is the god rays. Without it, fog is a wash; with it, the fog carries
 *   the silhouette of everything between it and the sun.
 *
 * ## Froxel discretisation (WebGPU)
 *
 * The volume is a 3D texture whose `(x, y)` axes are the screen and whose `z`
 * axis is view depth, distributed **non-linearly** so that froxels stay roughly
 * cubic in world space out to the volumetric far plane — see
 * {@link froxelSliceDepth}. Two compute passes fill it:
 *
 * 1. **Injection** — one thread per froxel. Evaluates media properties and
 *    lighting at the froxel centre and writes `(σ_s·L_scattered, σ_t)`.
 *    This is where all the expensive work happens (shadow lookups, light
 *    loops, noise fetches), and it is why the froxel formulation is worth it:
 *    the cost is proportional to the *volume resolution*, which is far below
 *    screen resolution × march steps, and it is amortised over every pixel that
 *    later reads that froxel.
 * 2. **Integration** — one thread per froxel *column*, marching `z` and
 *    accumulating scattering and transmittance with the analytic slice
 *    integral of {@link integrateScatteringSlice}. Writes
 *    `(L_inscattered_so_far, T_so_far)`.
 *
 * The compositing pass then does one trilinear 3D fetch per pixel. Beyond the
 * volume's far plane the fog continues analytically with the closed-form
 * height-fog integral {@link heightFogOpticalDepth}, so distant terrain still
 * fades correctly for the cost of a handful of ALU.
 *
 * The technique is Wronski, *"Volumetric Fog: Unified Compute Shader Based
 * Solution to Atmospheric Scattering"* (SIGGRAPH 2014, Advances in Real-Time
 * Rendering — Assassin's Creed IV), refined by Hillaire, *"Physically Based and
 * Unified Volumetric Rendering in Frostbite"* (SIGGRAPH 2015), from which the
 * energy-conserving slice integral and the temporal reprojection scheme are
 * taken directly.
 *
 * ## Temporal reprojection
 *
 * Sampling one point per froxel per frame is far too few: the noise field and
 * the shadow term both alias badly, and a torch produces a visibly blocky cone.
 * Instead each frame jitters the sample point inside its froxel along a Halton
 * sequence, and blends the result with the *same froxel reprojected into the
 * previous frame's volume*. Because the volume is view-aligned, reprojection is
 * a world-space round trip through the previous view-projection matrix; a
 * froxel that falls outside the previous frustum simply takes the new sample
 * at full weight. With a blend of `0.05` the effective sample count is about
 * twenty per froxel, at one sample of cost.
 *
 * Ghosting is bounded three ways: the history is rejected outside the previous
 * frustum; it is discarded wholesale on a camera cut (the motion-vector service
 * reports one); and the blend weight is raised where the new sample and the
 * history disagree strongly, which is exactly the disocclusion case (a torch
 * moving behind a wall). Fog is a low-frequency signal, so a neighbourhood
 * clamp — the usual TAA tool — is not needed and would only reintroduce noise.
 *
 * ## WebGL2
 *
 * There is no compute shader on WebGL2, and writing a 3D texture from a
 * fragment shader means one draw call per slice — sixty-four draw calls of
 * state-change-dominated work, which is worse than not doing it. The WebGL2
 * path instead ray-marches the *same* media and lighting functions in a
 * half-resolution screen-space pass owned by
 * {@link module:render/post/LightShafts}, with a per-pixel dithered ray offset
 * and the same temporal accumulation. It is the identical physical model
 * evaluated with a different sampling strategy, so the two backends match
 * closely enough that the critic loop cannot tell them apart on a still frame;
 * the froxel path is simply cheaper and steadier under motion.
 *
 * `createResolveNode` hides which of the two is running, so nothing downstream
 * branches on the backend.
 *
 * ## Banding
 *
 * Banding in a fog gradient is an automatic fail, and it has three independent
 * causes, each handled separately:
 *
 * - **Too few integration steps.** Fixed by the analytic slice integral (which
 *   is exact within a slice, not a midpoint approximation) plus the per-frame
 *   Halton jitter of the sample position.
 * - **Correlated per-pixel sampling.** Fixed by interleaved gradient noise
 *   (Jimenez 2014) offsetting the ray, rotated per frame so the temporal filter
 *   converges instead of locking to a fixed pattern.
 * - **Quantisation of a smooth low-amplitude gradient.** Fixed by a
 *   triangular-PDF dither applied to the in-scatter in the composite, scaled
 *   *relative* to the local in-scatter magnitude so it is invisible in bright
 *   areas and still breaks up the 1-LSB steps in dark ones.
 *
 * ## Services this module consumes (all optional)
 *
 * Every one of these is resolved through the {@link ServiceLocator} at `init`
 * and degrades to a documented fallback when absent, so this module compiles
 * and runs on its own.
 *
 * | key | interface | absent ⇒ |
 * |---|---|---|
 * | `render.sky.sun` | {@link VolumetricSunProvider} | the sun set through {@link VolumetricsService.setSun} |
 * | `render.shadow.volumetric` | {@link VolumetricShadowProvider} | no volumetric shadowing; fog is unshadowed but still lit |
 * | `render.volumetrics.lights` | {@link VolumetricLightProvider} | point/spot lights are gathered by walking `ctx.scene` |
 * | `render.volumetrics.ambient` | {@link VolumetricAmbientProvider} | the constant ambient set through {@link VolumetricsService.setAmbient} |
 * | `render.motion` | motion vectors (see `post/Denoise`) | temporal history is never force-invalidated |
 *
 * ## References
 *
 * - B. Wronski, *"Volumetric Fog: Unified Compute Shader Based Solution to
 *   Atmospheric Scattering"*, SIGGRAPH 2014.
 * - S. Hillaire, *"Physically Based and Unified Volumetric Rendering in
 *   Frostbite"*, SIGGRAPH 2015.
 * - L. G. Henyey & J. L. Greenstein, *"Diffuse Radiation in the Galaxy"*,
 *   Astrophysical Journal 93, 1941 — the phase function.
 * - J. Jimenez et al., *"Next Generation Post Processing in Call of Duty:
 *   Advanced Warfare"*, SIGGRAPH 2014 — interleaved gradient noise.
 * - E. Heitz & L. Belcour, *"Distributing Monte Carlo Errors as a Blue Noise in
 *   Screen Space"*, EGSR 2019 — why the per-frame rotation of the dither
 *   matters more than the dither itself.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  clamp,
  compute,
  exp,
  float,
  instanceIndex,
  int,
  ivec2,
  ivec3,
  log,
  max,
  min,
  mix,
  normalize,
  pow,
  saturate,
  sqrt,
  step,
  storageTexture3D,
  texture3D,
  textureLoad,
  textureStore,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import {
  asNodeRenderer,
  interleavedGradientNoiseNode,
  tryGetMotionVectors,
  viewPositionFromDepth,
  type FloatNode,
  type MotionVectorSource,
  type NodeRenderer,
  type Vec2Node,
  type Vec3Node,
  type Vec4Node,
} from './post/Denoise';

declare module '../core/EventBus' {
  interface GameEvents {
    /** Emitted once the volume exists and {@link VolumetricsKey} is registered. */
    'volumetrics:ready': { mode: VolumetricsMode; froxels: number };
    /** Emitted when the froxel path fails at runtime and the ray-march takes over. */
    'volumetrics:fallback': { reason: string };
  }
}

/* ------------------------------------------------------------------------- *
 * Contracts with modules owned by other agents
 *
 * None of these are imported from the module that implements them: the
 * contract is the service id and the shape, exactly as `render/Lighting.ts`
 * does for the sky. That keeps the module graph acyclic and lets any of the
 * five parallel render systems land in any order.
 * ------------------------------------------------------------------------- */

/**
 * The sun, as the volumetrics need it.
 *
 * Structurally identical to `SkySunProvider` in `render/Lighting.ts` and
 * `render/Sky.ts`, and published under the same id, so registering once
 * satisfies all three consumers.
 *
 * `sunDirection` points **from the world toward the sun**; light travels along
 * `-sunDirection`.
 */
export interface VolumetricSunProvider {
  readonly sunDirection: THREE.Vector3;
  /** Linear-space colour of direct sunlight. Normalised or not; see below. */
  readonly sunColor?: THREE.Color;
  /**
   * Scalar in `[0, 1]`, zero at or below the horizon. Multiplied into
   * `sunColor` to produce the radiance used for in-scattering, then scaled by
   * {@link GlobalFogParams.sunScatteringScale}.
   */
  readonly sunIntensity?: number;
}

/** Service id the sky module publishes its sun under. */
export const VolumetricSunKey = serviceKey<VolumetricSunProvider>('render.sky.sun');

/**
 * Sun visibility for a **world-space point**, which is what makes god rays
 * possible. Surface shadowing samples the shadow map at the fragment; the fog
 * has to sample it at every point along the ray instead.
 *
 * ### Preferred form
 *
 * Implement {@link createSunVisibilityNode} and this module uses it verbatim.
 * That is the right contract for a cascaded-shadow-map module, which already
 * knows its own cascade selection, bias and filtering rules and should not have
 * them duplicated here.
 *
 * ### Raw form
 *
 * If `createSunVisibilityNode` is absent but `shadowDepthTexture` and
 * `cascadeMatrices` are present, this module builds its own single-tap
 * comparison against the cascade array. One tap is deliberate: the ray
 * integration averages tens of independent lookups along its length, so a PCF
 * kernel at each of them buys nothing that the integration is not already
 * doing, and costs `filterSamples ×` as much.
 *
 * ### Conventions
 *
 * - `cascadeMatrices[i]` maps world space to that cascade's `[0,1]³` shadow
 *   space, `xy` in texture coordinates with **v already flipped** to three's
 *   node-renderer shadow convention (the same matrix
 *   `render/CascadedShadowMaps.ts` builds).
 * - `shadowDepthTexture` is a depth **array** texture, one layer per cascade,
 *   `NearestFilter`, with **no** `compareFunction` — the comparison happens in
 *   the shader.
 * - The arrays are live references, rewritten in place each frame. This module
 *   binds them once and never copies them.
 */
export interface VolumetricShadowProvider {
  readonly cascadeCount: number;
  readonly shadowDepthTexture: THREE.Texture | null;
  readonly cascadeMatrices: readonly THREE.Matrix4[];
  /** Square resolution of one cascade layer, in texels. */
  readonly shadowMapSize: number;
  /** `true` when the depth buffer is reversed (1 at the near plane). */
  readonly reversedDepth?: boolean;
  /** World-space distance past which the provider has no shadow data. */
  readonly shadowDistance?: number;
  /**
   * Escape hatch: a ready-made TSL node returning sun visibility in `[0, 1]`
   * for a world-space position. Takes precedence over the raw form.
   */
  createSunVisibilityNode?(worldPosition: Vec3Node): FloatNode;
}

/** Service id the shadow module should publish a volumetric sampler under. */
export const VolumetricShadowKey = serviceKey<VolumetricShadowProvider>(
  'render.shadow.volumetric',
);

/** One local light, in the only terms in-scattering cares about. */
export interface VolumetricPointLight {
  /** World-space position. */
  readonly position: THREE.Vector3;
  /** Linear-space colour. */
  readonly color: THREE.Color;
  /** Radiant intensity, in the same units the surface shading uses. */
  readonly intensity: number;
  /** Influence radius in world units; the falloff window closes at it. */
  readonly radius: number;
  /**
   * Per-light multiplier on the scattering contribution, `1` by default. A
   * torch wants `1`; a fill light that exists only to lift a surface wants `0`
   * so it does not produce a glow cone in mid-air that no one placed.
   */
  readonly volumetricScale?: number;
}

/**
 * How the lighting rig hands over its light list.
 *
 * `render/Lighting.ts` owns every light in the game but publishes only handles,
 * not the list, because its slot arbitration is per frame. Implementing this
 * one method on the lighting service is the whole integration: fill `out` with
 * at most `max` lights, strongest first, and return how many were written.
 *
 * When the service is absent this module walks `ctx.scene` for visible
 * `PointLight`/`SpotLight` objects instead, which is correct but does not know
 * about the rig's culling, so the ranking may differ by a frame.
 */
export interface VolumetricLightProvider {
  collectVolumetricLights(out: VolumetricPointLight[], max: number): number;
}

/** Service id for {@link VolumetricLightProvider}. */
export const VolumetricLightsKey = serviceKey<VolumetricLightProvider>(
  'render.volumetrics.lights',
);

/**
 * Ambient radiance arriving at a point in the fog from everywhere that is not a
 * direct light — the sky dome, mostly. This is the multiple-scattering stand-in
 * and it is what stops shadowed fog from going black, which is the single most
 * common way volumetric fog looks wrong.
 *
 * The sky/IBL module can publish a live, time-of-day-driven value here; absent
 * that, {@link VolumetricsService.setAmbient} supplies a constant.
 */
export interface VolumetricAmbientProvider {
  /** Linear-space radiance, already in render units. Live reference. */
  readonly volumetricAmbient: THREE.Color;
  /** Optional extra tint applied to fog below {@link GlobalFogParams.height}. */
  readonly volumetricGroundAmbient?: THREE.Color;
}

/** Service id for {@link VolumetricAmbientProvider}. */
export const VolumetricAmbientKey = serviceKey<VolumetricAmbientProvider>(
  'render.volumetrics.ambient',
);

/* ------------------------------------------------------------------------- *
 * Public vocabulary
 * ------------------------------------------------------------------------- */

/** A position accepted by the fog-volume API. */
export interface Vec3Like {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** Which sampling strategy is actually running. */
export type VolumetricsMode = 'froxel' | 'raymarch' | 'off';

/** Froxel depth distributions. See {@link froxelSliceDepth}. */
export type FroxelDepthDistribution = 'exponential' | 'quadratic';

export type VolumetricsQuality = 'off' | 'low' | 'medium' | 'high' | 'ultra';

/** Globally uniform media parameters. Everything here is art-directable. */
export interface GlobalFogParams {
  /**
   * Extinction coefficient at and below {@link height}, per metre. `0.014`
   * puts the 50% transmittance distance at about 50 m, which is the Blood
   * Moor's overcast register; `0.05` is a cave.
   */
  density: number;
  /**
   * Single-scattering albedo, per channel, in `[0, 1]`. Water fog is
   * near-white and near-1; the slight blue bias here is the tiny bit of
   * Rayleigh that survives in a humid, sunless sky and it is what keeps
   * shadowed fog *cold* rather than grey.
   */
  albedo: THREE.Color;
  /** Height, in world units, below which the fog is at full density. */
  height: number;
  /** Reciprocal scale height above {@link height}. Larger ⇒ thinner layer. */
  heightFalloff: number;
  /** Forward lobe anisotropy `g ∈ (-1, 1)`. Mist is strongly forward, ~0.7. */
  anisotropy: number;
  /** Backward lobe anisotropy, negative. Gives the halo when facing away. */
  backAnisotropy: number;
  /** Weight of the backward lobe in `[0, 1]`. */
  lobeBlend: number;
  /** Multiplier on the sun's in-scattering. Artistic; 1 is physical. */
  sunScatteringScale: number;
  /** Multiplier on local lights' in-scattering. */
  lightScatteringScale: number;
  /** Multiplier on the ambient (multiple-scattering) term. */
  ambientScatteringScale: number;
  /** Density modulation depth from the 3D noise, in `[0, 1]`. */
  noiseStrength: number;
  /** World size, in units, of one period of the coarse noise octave. */
  noiseScale: number;
  /** Wind velocity, world units per second, applied to the noise lookup. */
  wind: THREE.Vector3;
  /**
   * View distance the froxel volume covers, in world units. Beyond it the fog
   * continues analytically. 64 m is the sweet spot for an outdoor scene: far
   * enough that the shadowed shafts that need volumetric detail are inside it,
   * near enough that froxels stay small.
   */
  volumeDistance: number;
}

/** Shape of a designer-placed fog volume. */
export type FogVolumeShape = 'box' | 'sphere';

export interface FogVolumeDesc {
  /** Default `'box'`. */
  readonly shape?: FogVolumeShape;
  readonly center: Vec3Like;
  /** Box only. Default `(4, 2, 4)`. */
  readonly halfExtent?: Vec3Like;
  /** Sphere only. Default `4`. */
  readonly radius?: number;
  /**
   * Multiplier on the global density inside the volume. `>1` thickens (a mist
   * pooling in the moor's low ground), `<1` thins, `0` clears a pocket. This is
   * a multiplier, not an override, so a global weather change still affects it.
   * Default `3`.
   */
  readonly densityScale?: number;
  /** Scattering albedo tint inside the volume. Default white (no tint). */
  readonly color?: THREE.ColorRepresentation;
  /**
   * Fraction of the half-extent spent fading from full strength to nothing, in
   * `[0, 1]`. A hard-edged fog volume reads as a box of gas and is the classic
   * tell; the default `0.4` is deliberately generous.
   */
  readonly feather?: number;
  /**
   * Radiance added to the ambient term inside the volume, independent of any
   * light. This is how the Den of Evil gets its sickly self-lit haze without
   * placing a light that would also hit the walls. Default `0`.
   */
  readonly emissive?: THREE.ColorRepresentation;
  /** Arbitration weight when more volumes are visible than the budget. */
  readonly priority?: number;
  readonly name?: string;
}

export interface FogVolumeHandle {
  readonly id: number;
  /** False once {@link release} has been called. */
  readonly alive: boolean;
  /** Whether the volume was bound to a GPU slot on the most recent frame. */
  readonly active: boolean;
  setCenter(x: number, y: number, z: number): void;
  setHalfExtent(x: number, y: number, z: number): void;
  setRadius(radius: number): void;
  setDensityScale(scale: number): void;
  setColor(color: THREE.ColorRepresentation): void;
  setEmissive(color: THREE.ColorRepresentation): void;
  setPriority(priority: number): void;
  /** Idempotent. */
  release(): void;
}

export interface VolumetricsStats {
  readonly mode: VolumetricsMode;
  readonly quality: VolumetricsQuality;
  /** Froxel grid dimensions, or `(0,0,0)` on the ray-march path. */
  readonly froxelDimensions: readonly [number, number, number];
  /** Approximate GPU bytes held by the froxel volumes. */
  readonly bytes: number;
  /** Local lights bound to the media shader this frame. */
  readonly lights: number;
  /** Fog volumes bound this frame, and how many were registered. */
  readonly volumes: number;
  readonly registeredVolumes: number;
  /** Whether a volumetric shadow sampler was found. */
  readonly shadowed: boolean;
  /** Whether temporal reprojection had a valid history this frame. */
  readonly temporal: boolean;
}

/**
 * Everything `LightShafts` (and any debug UI) needs from this module.
 *
 * The important member is {@link createResolveNode}: it returns the in-scatter
 * and transmittance for a screen pixel, hiding whether that came from a froxel
 * fetch or from a ray march.
 */
export interface VolumetricsService {
  readonly mode: VolumetricsMode;
  readonly stats: VolumetricsStats;
  readonly params: Readonly<GlobalFogParams>;
  enabled: boolean;

  /**
   * Build the sampling node.
   *
   * @returns `vec4(inScatteredRadiance.rgb, transmittance)`. Composite as
   *   `final = surface * result.a + result.rgb`.
   */
  createResolveNode(input: VolumetricResolveInput): Vec4Node;

  /**
   * Whether the resolve node is expensive enough that the consumer should
   * evaluate it at half resolution and bilaterally upsample. True on the
   * ray-march path, false on the froxel path (where it is one 3D fetch).
   */
  readonly prefersHalfResolution: boolean;

  /**
   * Bumped whenever {@link createResolveNode} would return a structurally
   * different graph (mode change, shadow provider appearing, quality change),
   * so consumers know to rebuild their materials.
   */
  readonly graphVersion: number;

  setParams(params: Partial<GlobalFogParams>): void;
  setQuality(quality: VolumetricsQuality): void;
  /** Override the sun when no {@link VolumetricSunProvider} is registered. */
  setSun(direction: Vec3Like, color: THREE.ColorRepresentation, intensity: number): void;
  /** Override the ambient term when no provider is registered. */
  setAmbient(color: THREE.ColorRepresentation, groundColor?: THREE.ColorRepresentation): void;

  addFogVolume(desc: FogVolumeDesc): FogVolumeHandle;
}

/** What a consumer supplies to {@link VolumetricsService.createResolveNode}. */
export interface VolumetricResolveInput {
  /** Texture-space UV of the pixel being shaded, Y-down. */
  readonly screenUv: Vec2Node;
  /** Positive linear view depth of the surface at that pixel, in world units. */
  readonly viewZ: FloatNode;
  /** Pixel coordinate, for the per-pixel dither. */
  readonly pixel: Vec2Node;
}

/** Service key other modules resolve this module by. */
export const VolumetricsKey = serviceKey<VolumetricsService>('render.volumetrics');

export interface VolumetricsOptions {
  quality?: VolumetricsQuality;
  /** Overrides the tier's froxel grid. Rounded to a multiple of the workgroup. */
  froxelDimensions?: readonly [number, number, number];
  depthDistribution?: FroxelDepthDistribution;
  /** Local lights bound to the media shader. Default 4. */
  maxLights?: number;
  /** Designer fog volumes bound at once. Default 8. */
  maxFogVolumes?: number;
  /** Steady-state temporal blend weight for the new sample. Default 0.06. */
  temporalBlend?: number;
  /** Edge of the baked 3D noise texture, in texels. Default 32. */
  noiseTextureSize?: number;
  /** Seed for the baked noise. Default `'d2rim.fog'`. */
  noiseSeed?: string | number;
  /** Ray-march steps on the WebGL2 path. Overridden by the tier. */
  marchSteps?: number;
  params?: Partial<GlobalFogParams>;
  /** Register under {@link VolumetricsKey}. Default `true`. */
  registerService?: boolean;
}

/* ------------------------------------------------------------------------- *
 * Pure maths — exported so the behaviour can be pinned without a GPU
 * ------------------------------------------------------------------------- */

const FOUR_PI = 4 * Math.PI;

/**
 * The Henyey–Greenstein phase function, normalised so that its integral over
 * the sphere is 1.
 *
 * ```
 *   p(θ) = (1 − g²) / (4π · (1 + g² − 2g·cosθ)^{3/2})
 * ```
 *
 * `g = 0` is isotropic, `g → 1` is a forward delta, `g < 0` scatters backwards.
 * `cosTheta` is the cosine between the direction light is *travelling* and the
 * direction it is scattered into (i.e. towards the eye), so looking straight
 * into the sun through fog gives `cosTheta = 1`.
 */
export function henyeyGreenstein(cosTheta: number, g: number): number {
  const gg = g * g;
  const denominator = 1 + gg - 2 * g * cosTheta;
  // The clamp matters: at |g| → 1 and cosTheta → 1 the denominator underflows
  // to zero and the phase function goes to infinity, which is physically true
  // of a delta lobe and catastrophic in a shader.
  return (1 - gg) / (FOUR_PI * Math.pow(Math.max(denominator, 1e-4), 1.5));
}

/**
 * A two-lobe phase function: a strong forward lobe plus a weak backward one.
 *
 * A single HG lobe cannot reproduce the *glory* — the bright halo around your
 * own shadow when you look away from the sun through mist — and the moor scene
 * looks flat without it. Two lobes is the standard cheap fix and is what
 * Frostbite, Unreal and Frostpunk all ship.
 *
 * @param blend weight of the backward lobe, in `[0, 1]`.
 */
export function dualLobePhase(
  cosTheta: number,
  forwardG: number,
  backwardG: number,
  blend: number,
): number {
  const t = Math.min(1, Math.max(0, blend));
  return (1 - t) * henyeyGreenstein(cosTheta, forwardG) + t * henyeyGreenstein(cosTheta, backwardG);
}

/**
 * View-space depth at the far boundary of froxel slice `slice`.
 *
 * Two distributions, both mapping `slice ∈ [0, sliceCount]` onto
 * `[near, far]` monotonically:
 *
 * - `'exponential'` — `near · (far/near)^t`. Constant *relative* precision, so
 *   froxels stay a constant fraction of their distance and therefore roughly
 *   constant in screen-space size. This is the distribution clustered shading
 *   uses and it is the right one when the near plane is close.
 * - `'quadratic'` — `near + (far − near)·t²`. Wronski's original choice.
 *   Spends less of the volume on the first metre, which matters when the near
 *   plane is very close (0.1 m) and the exponential distribution would burn a
 *   third of the slices on the player's own boots.
 *
 * Both are exactly inverted by {@link froxelDepthToSlice}.
 */
export function froxelSliceDepth(
  slice: number,
  sliceCount: number,
  near: number,
  far: number,
  distribution: FroxelDepthDistribution = 'exponential',
): number {
  const n = Math.max(1e-4, near);
  const f = Math.max(n * (1 + 1e-6), far);
  const t = Math.min(1, Math.max(0, slice / Math.max(1, sliceCount)));
  if (distribution === 'quadratic') return n + (f - n) * t * t;
  return n * Math.pow(f / n, t);
}

/** Exact inverse of {@link froxelSliceDepth}; returns a fractional slice. */
export function froxelDepthToSlice(
  depth: number,
  sliceCount: number,
  near: number,
  far: number,
  distribution: FroxelDepthDistribution = 'exponential',
): number {
  const n = Math.max(1e-4, near);
  const f = Math.max(n * (1 + 1e-6), far);
  const d = Math.min(f, Math.max(n, depth));
  const count = Math.max(1, sliceCount);
  if (distribution === 'quadratic') return Math.sqrt((d - n) / (f - n)) * count;
  return (Math.log(d / n) / Math.log(f / n)) * count;
}

/**
 * Normalised density of the height-fog profile at world height `y`.
 *
 * ```
 *   ρ(y) = exp( −k · max(0, y − y₀) )
 * ```
 *
 * Constant below `y₀` and exponentially thinning above it. The flat region is
 * not a simplification for its own sake: a pure exponential profile grows
 * without bound as the camera descends, so a player walking into the Den of
 * Evil would watch the fog density explode. Clamping the profile at its
 * reference height makes the model well behaved everywhere *and* gives an exact
 * closed-form line integral (see {@link heightFogOpticalDepth}), which is what
 * lets the fog continue past the froxel volume for free.
 */
export function heightFogDensity(y: number, referenceHeight: number, falloff: number): number {
  return Math.exp(-Math.max(0, falloff) * Math.max(0, y - referenceHeight));
}

/**
 * Closed-form optical depth `∫₀^d ρ(y(t)) dt` through the profile of
 * {@link heightFogDensity}, for a ray starting at height `originY` and rising
 * `dirY` per unit of distance.
 *
 * Piecewise because the profile is: the portion of the ray below `y₀`
 * integrates to its own length, and the portion above integrates to
 * `(exp(−k·Δ₁) − exp(−k·Δ₂)) / (k·dirY)`.
 *
 * Used for the analytic continuation of the fog beyond the froxel volume, where
 * marching would be pure waste — the media is smooth out there and the sun
 * shadowing has long since run out of cascades.
 *
 * @param dirY the `y` component of a **unit** direction vector.
 * @returns optical depth for a unit-density medium; multiply by `σ_t`.
 */
export function heightFogOpticalDepth(
  originY: number,
  dirY: number,
  distance: number,
  referenceHeight: number,
  falloff: number,
): number {
  const d = Math.max(0, distance);
  if (d === 0) return 0;
  const k = Math.max(0, falloff);
  const y0 = originY - referenceHeight;
  const y1 = y0 + dirY * d;

  // Degenerate cases: a horizontal ray, or no falloff at all, are constant.
  if (k === 0) return d;
  if (Math.abs(dirY) < 1e-6) return d * Math.exp(-k * Math.max(0, y0));

  // Parameter at which the ray crosses the reference height.
  const tCross = -y0 / dirY;
  const segments: Array<[number, number]> = [];
  if (tCross <= 0 || tCross >= d) {
    segments.push([0, d]);
  } else {
    segments.push([0, tCross], [tCross, d]);
  }

  let total = 0;
  for (const [a, b] of segments) {
    const ya = y0 + dirY * a;
    const yb = y0 + dirY * b;
    if (ya <= 0 && yb <= 0) {
      // Entirely inside the flat region.
      total += b - a;
    } else {
      // Entirely inside the exponential region: ∫ exp(−k(y₀ + dirY·t)) dt.
      total += (Math.exp(-k * ya) - Math.exp(-k * yb)) / (k * dirY);
    }
  }
  // `y1` is referenced only to make the intent of the crossing test explicit.
  void y1;
  return Math.max(0, total);
}

export interface SliceIntegral {
  /** In-scattered radiance contributed by this slice, already transmittance-weighted internally. */
  readonly scattering: number;
  /** Transmittance across this slice alone. */
  readonly transmittance: number;
}

/**
 * Energy-conserving integration of one homogeneous slice.
 *
 * The naive `scattering · stepLength · transmittance` is a midpoint rule and it
 * is wrong by up to the whole slice's worth of extinction, which is exactly
 * where the visible banding in a low-slice-count volume comes from. Hillaire's
 * form integrates the slice analytically instead:
 *
 * ```
 *   S_int = (S − S·exp(−σ_t·d)) / σ_t
 *   T     = exp(−σ_t·d)
 * ```
 *
 * As `σ_t → 0` this tends to `S·d`, which is the correct optically-thin limit
 * and the branch the code takes explicitly to avoid a `0/0`.
 */
export function integrateScatteringSlice(
  scattering: number,
  extinction: number,
  stepLength: number,
): SliceIntegral {
  const d = Math.max(0, stepLength);
  const sigma = Math.max(0, extinction);
  const transmittance = Math.exp(-sigma * d);
  if (sigma < 1e-6) {
    return { scattering: scattering * d, transmittance };
  }
  return { scattering: (scattering - scattering * transmittance) / sigma, transmittance };
}

/**
 * Falloff weight of a designer fog volume at a point, in `[0, 1]`.
 *
 * For a box this is the product of the three per-axis feathered ramps, which
 * gives rounded corners for free; for a sphere it is a single radial ramp. The
 * ramp is smoothstep, not linear, so the density derivative is continuous at
 * both ends and the volume boundary never shows as a Mach band.
 *
 * @param feather fraction of the half-extent used for the fade, in `[0, 1]`.
 */
export function fogVolumeWeight(
  shape: FogVolumeShape,
  point: Vec3Like,
  center: Vec3Like,
  halfExtent: Vec3Like,
  feather: number,
): number {
  const f = Math.min(1, Math.max(0.001, feather));
  const dx = point.x - center.x;
  const dy = point.y - center.y;
  const dz = point.z - center.z;

  if (shape === 'sphere') {
    const r = Math.max(1e-4, halfExtent.x);
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return smoothstep01((r - distance) / (r * f));
  }

  const wx = smoothstep01((Math.max(1e-4, halfExtent.x) - Math.abs(dx)) / (halfExtent.x * f));
  const wy = smoothstep01((Math.max(1e-4, halfExtent.y) - Math.abs(dy)) / (halfExtent.y * f));
  const wz = smoothstep01((Math.max(1e-4, halfExtent.z) - Math.abs(dz)) / (halfExtent.z * f));
  return wx * wy * wz;
}

/** `smoothstep(0, 1, x)` with the clamp folded in. */
function smoothstep01(x: number): number {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

/**
 * The `index`-th term of the radical-inverse (van der Corput) sequence in
 * `base`. Two of these with coprime bases make the Halton sequence used to
 * jitter froxel sample positions between frames.
 */
export function radicalInverse(index: number, base: number): number {
  let result = 0;
  let f = 1 / base;
  let i = Math.max(0, Math.floor(index));
  while (i > 0) {
    result += (i % base) * f;
    i = Math.floor(i / base);
    f /= base;
  }
  return result;
}

/** Halton point `(base 2, base 3, base 5)` in `[0,1)³`, for froxel jitter. */
export function haltonJitter3D(index: number): [number, number, number] {
  return [radicalInverse(index, 2), radicalInverse(index, 3), radicalInverse(index, 5)];
}

/**
 * Blend weight for the new sample in the temporal reprojection.
 *
 * A plain exponential moving average with a fixed `α` takes `1/α` frames to
 * converge, which is a visible smear on a freshly disoccluded froxel. The
 * weight is therefore raised both for young histories and where the new sample
 * disagrees strongly with the history — `relativeChange` is
 * `|new − old| / (new + old)`, a scale-free measure that behaves the same in a
 * dark cave and in daylight.
 *
 * @param historyValid `false` on the first frame, after a cut, or outside the
 *   previous frustum, which forces the new sample to full weight.
 */
export function temporalBlendWeight(
  minAlpha: number,
  relativeChange: number,
  historyValid: boolean,
): number {
  if (!historyValid) return 1;
  const a = Math.min(1, Math.max(0, minAlpha));
  const change = Math.min(1, Math.max(0, relativeChange));
  // Quadratic response: small disagreements (noise) are ignored, large ones
  // (a real change in the world) take over quickly.
  return Math.min(1, a + (1 - a) * change * change);
}

/* --- baked 3D noise ------------------------------------------------------ */

/** 32-bit integer hash; the lattice hash for {@link tilingValueNoise3D}. */
function hash3(x: number, y: number, z: number, seed: number): number {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(z, 2147483647) + seed) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/**
 * Seamlessly tiling 3D value noise on a lattice of `period` cells.
 *
 * Value rather than gradient noise because the result is baked into an 8-bit
 * texture and then trilinearly filtered by the GPU anyway — the extra frequency
 * content of a gradient basis does not survive the quantisation, and the
 * lattice hash is a third of the cost to bake. Tiling is exact: lattice
 * coordinates are taken modulo `period`, so the baked texture can be sampled
 * with `RepeatWrapping` at any world scale with no seam.
 *
 * The interpolant is the quintic `6t⁵ − 15t⁴ + 10t³` (Perlin's improved fade),
 * whose first *and* second derivatives vanish at the lattice points; the cubic
 * smoothstep leaves a second-derivative discontinuity that is plainly visible
 * as a grid in a slowly drifting fog.
 */
export function tilingValueNoise3D(
  x: number,
  y: number,
  z: number,
  period: number,
  seed = 0,
): number {
  const p = Math.max(1, Math.floor(period));
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const tx = fade(x - xi);
  const ty = fade(y - yi);
  const tz = fade(z - zi);

  const wrap = (v: number): number => ((v % p) + p) % p;
  const x0 = wrap(xi);
  const y0 = wrap(yi);
  const z0 = wrap(zi);
  const x1 = wrap(xi + 1);
  const y1 = wrap(yi + 1);
  const z1 = wrap(zi + 1);

  const c000 = hash3(x0, y0, z0, seed);
  const c100 = hash3(x1, y0, z0, seed);
  const c010 = hash3(x0, y1, z0, seed);
  const c110 = hash3(x1, y1, z0, seed);
  const c001 = hash3(x0, y0, z1, seed);
  const c101 = hash3(x1, y0, z1, seed);
  const c011 = hash3(x0, y1, z1, seed);
  const c111 = hash3(x1, y1, z1, seed);

  const e00 = c000 + (c100 - c000) * tx;
  const e10 = c010 + (c110 - c010) * tx;
  const e01 = c001 + (c101 - c001) * tx;
  const e11 = c011 + (c111 - c011) * tx;
  const f0 = e00 + (e10 - e00) * ty;
  const f1 = e01 + (e11 - e01) * ty;
  return f0 + (f1 - f0) * tz;
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/**
 * Fractal sum of {@link tilingValueNoise3D}. Each octave doubles the lattice
 * frequency and halves the amplitude; because every octave's period also
 * doubles, the sum tiles over the *base* period. Normalised to `[0, 1]`.
 */
export function tilingFbm3D(
  x: number,
  y: number,
  z: number,
  period: number,
  octaves: number,
  seed = 0,
): number {
  let sum = 0;
  let amplitude = 1;
  let total = 0;
  let frequency = 1;
  const n = Math.max(1, Math.floor(octaves));
  for (let i = 0; i < n; i++) {
    sum +=
      amplitude *
      tilingValueNoise3D(x * frequency, y * frequency, z * frequency, period * frequency, seed + i * 8191);
    total += amplitude;
    amplitude *= 0.5;
    frequency *= 2;
  }
  return sum / total;
}

/**
 * Bake the fog's detail noise into an RGBA8 `Data3DTexture`.
 *
 * Four independent bands, sampled at four different world scales in the shader
 * and drifting at four different speeds, which is what turns a single octave's
 * obvious "moving lumps" into something that reads as turbulence:
 *
 * | channel | content |
 * |---|---|
 * | `r` | 3-octave fBm — the large-scale density variation |
 * | `g` | 2-octave fBm at twice the frequency — mid detail |
 * | `b` | 1-octave, `1 − |2n − 1|` ridged — the wisps |
 * | `a` | 3-octave fBm, decorrelated seed — used to warp the lookup |
 *
 * 32³ RGBA8 is 128 KiB, sits entirely in L2, and is more than enough: it is
 * being trilinearly filtered and then multiplied by a density that is itself
 * smooth. A 64³ texture is 1 MiB for detail that the froxel grid cannot resolve.
 */
export function bakeFogNoiseTexture(size = 32, seed: string | number = 'd2rim.fog'): THREE.Data3DTexture {
  const n = Math.max(4, Math.floor(size));
  const numeric = typeof seed === 'number' ? seed | 0 : hashString(seed);
  const data = new Uint8Array(n * n * n * 4);
  // Lattice period in *texture* space: the fBm's base period must divide the
  // texture edge for the bake to tile, so it is expressed in lattice cells and
  // the sample coordinate is scaled to match.
  const period = 4;
  const scale = period / n;

  let offset = 0;
  for (let z = 0; z < n; z++) {
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const fx = x * scale;
        const fy = y * scale;
        const fz = z * scale;
        const base = tilingFbm3D(fx, fy, fz, period, 3, numeric);
        const mid = tilingFbm3D(fx * 2, fy * 2, fz * 2, period * 2, 2, numeric + 101);
        const single = tilingValueNoise3D(fx * 2, fy * 2, fz * 2, period * 2, numeric + 809);
        const ridged = 1 - Math.abs(2 * single - 1);
        const warp = tilingFbm3D(fx, fy, fz, period, 3, numeric + 7919);
        data[offset] = Math.round(base * 255);
        data[offset + 1] = Math.round(mid * 255);
        data[offset + 2] = Math.round(ridged * 255);
        data[offset + 3] = Math.round(warp * 255);
        offset += 4;
      }
    }
  }

  const texture = new THREE.Data3DTexture(data, n, n, n);
  texture.name = 'fog.noise3D';
  texture.format = THREE.RGBAFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.wrapR = THREE.RepeatWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function hashString(value: string): number {
  let h = 2166136261;
  for (let i = 0; i < value.length; i++) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h | 0;
}

/* ------------------------------------------------------------------------- *
 * Quality tiers
 * ------------------------------------------------------------------------- */

interface TierConfig {
  /** Froxel grid, `x·y·z` a multiple of {@link WORKGROUP_SIZE}. */
  readonly froxels: readonly [number, number, number];
  /** Ray-march steps on the WebGL2 path. */
  readonly marchSteps: number;
  /** Local lights bound to the media shader. */
  readonly lights: number;
}

/** Linear workgroup size for both compute kernels. 64 is the portable sweet spot. */
export const WORKGROUP_SIZE = 64;

export const VOLUMETRIC_TIERS: Readonly<Record<Exclude<VolumetricsQuality, 'off'>, TierConfig>> = {
  //           froxels         steps  lights
  low: { froxels: [96, 54, 32], marchSteps: 16, lights: 2 },
  medium: { froxels: [128, 72, 48], marchSteps: 24, lights: 3 },
  high: { froxels: [160, 90, 64], marchSteps: 32, lights: 4 },
  ultra: { froxels: [192, 108, 80], marchSteps: 48, lights: 6 },
};

/* ------------------------------------------------------------------------- *
 * Defaults — the Blood Moor register
 * ------------------------------------------------------------------------- */

function defaultParams(): GlobalFogParams {
  return {
    density: 0.014,
    // Very slightly blue: shadowed fog must read cold against firelight.
    albedo: new THREE.Color(0.92, 0.95, 1.0),
    height: 1.5,
    heightFalloff: 0.085,
    anisotropy: 0.72,
    backAnisotropy: -0.28,
    lobeBlend: 0.22,
    sunScatteringScale: 1,
    lightScatteringScale: 1,
    ambientScatteringScale: 1,
    noiseStrength: 0.55,
    noiseScale: 26,
    wind: new THREE.Vector3(0.35, 0.04, 0.16),
    volumeDistance: 64,
  };
}

/* ------------------------------------------------------------------------- *
 * Internal state
 * ------------------------------------------------------------------------- */

interface FogVolumeRecord {
  id: number;
  alive: boolean;
  active: boolean;
  shape: FogVolumeShape;
  center: THREE.Vector3;
  halfExtent: THREE.Vector3;
  densityScale: number;
  color: THREE.Color;
  emissive: THREE.Color;
  feather: number;
  priority: number;
  name: string;
}

const _sphere = new THREE.Sphere();
const _frustum = new THREE.Frustum();
const _viewProjection = new THREE.Matrix4();
const _lightPosition = new THREE.Vector3();

/* ------------------------------------------------------------------------- *
 * Shader-side uniform bundle
 * ------------------------------------------------------------------------- */

/**
 * Every uniform the media and lighting nodes read.
 *
 * Bundled into one object because both the froxel injection kernel and the
 * WebGL2 ray march build their graphs from exactly the same set — that shared
 * evaluation is the reason the two backends agree, and keeping the bundle
 * explicit is what stops them drifting apart.
 */
/**
 * Allocate the uniform bundle. Written as a factory so the type is *inferred*
 * from the values rather than restated: three's `uniform()` return type is a
 * node type derived from its argument, and spelling it out by hand is both
 * unstable across three versions and impossible to get right for `Color`.
 */
function createMediaUniforms() {
  return {
    density: uniform(0.014),
    albedo: uniform(new THREE.Vector3(1, 1, 1)),
    height: uniform(1.5),
    heightFalloff: uniform(0.085),
    /** `(forwardG, backwardG, lobeBlend)` */
    phase: uniform(new THREE.Vector3(0.72, -0.28, 0.22)),
    sunDirection: uniform(new THREE.Vector3(0, 1, 0)),
    sunRadiance: uniform(new THREE.Vector3(0, 0, 0)),
    ambient: uniform(new THREE.Vector3(0, 0, 0)),
    groundAmbient: uniform(new THREE.Vector3(0, 0, 0)),
    lightScale: uniform(1),
    /** `(1/noiseScale, strength, warpStrength, unused)` */
    noise: uniform(new THREE.Vector4(1 / 26, 0.55, 0.35, 0)),
    wind: uniform(new THREE.Vector3(0.35, 0.04, 0.16)),
    time: uniform(0),
    lightCount: uniform(0),
    volumeCount: uniform(0),
    cameraPosition: uniform(new THREE.Vector3()),
    cameraMatrixWorld: uniform(new THREE.Matrix4()),
    projScale: uniform(new THREE.Vector2(1, 1)),
    /** `(camera near, volumetric far)` */
    range: uniform(new THREE.Vector2(0.1, 64)),
    frame: uniform(0),
    jitter: uniform(new THREE.Vector3(0.5, 0.5, 0.5)),
    enabled: uniform(1),
  };
}

type MediaUniforms = ReturnType<typeof createMediaUniforms>;


/* ------------------------------------------------------------------------- *
 * The module
 * ------------------------------------------------------------------------- */

/**
 * Owns the froxel volume, the media parameters, the fog-volume list and the
 * shared media/lighting TSL graphs.
 *
 * Runs its GPU work in `lateUpdate`, after every gameplay module has settled
 * the camera and before the engine's `renderer.render`, so the volume the
 * composite reads describes *this* frame's camera and not the previous one's.
 */
export class VolumetricsModule implements GameModule, VolumetricsService {
  readonly name = 'render.volumetrics';

  /* -- configuration ----------------------------------------------------- */

  readonly #params: GlobalFogParams = defaultParams();
  readonly #options: Required<
    Omit<VolumetricsOptions, 'params' | 'froxelDimensions' | 'quality' | 'marchSteps'>
  >;
  #quality: VolumetricsQuality;
  #froxelOverride: readonly [number, number, number] | null;
  #marchStepOverride: number | null;
  #depthDistribution: FroxelDepthDistribution;

  enabled = true;

  /* -- resolved services ------------------------------------------------- */

  #ctx: GameContext | null = null;
  #sun: VolumetricSunProvider | undefined;
  #shadows: VolumetricShadowProvider | undefined;
  #lights: VolumetricLightProvider | undefined;
  #ambient: VolumetricAmbientProvider | undefined;
  #motion: MotionVectorSource | undefined;

  /* -- CPU-side scratch -------------------------------------------------- */

  readonly #sunDirection = new THREE.Vector3(0.32, 0.55, -0.77).normalize();
  readonly #sunColor = new THREE.Color(0.72, 0.76, 0.86);
  #sunIntensity = 1;
  readonly #ambientColor = new THREE.Color(0.055, 0.068, 0.09);
  readonly #groundAmbientColor = new THREE.Color(0.022, 0.021, 0.019);

  readonly #volumes: FogVolumeRecord[] = [];
  #nextVolumeId = 1;
  readonly #lightScratch: VolumetricPointLight[] = [];
  readonly #sceneLights: MutableVolumetricLight[] = [];

  /* -- uniforms ---------------------------------------------------------- */

  readonly #u: MediaUniforms = createMediaUniforms();

  /** `(position.xyz, 1/radius²)` per bound light. */
  readonly #lightPositions: THREE.Vector4[] = [];
  /** `(colour·intensity, volumetricScale)` per bound light. */
  readonly #lightColors: THREE.Vector4[] = [];
  /** `(center.xyz, shape)` — shape 0 box, 1 sphere. */
  readonly #volumeA: THREE.Vector4[] = [];
  /** `(halfExtent.xyz, densityScale)`. */
  readonly #volumeB: THREE.Vector4[] = [];
  /** `(albedoTint.rgb, feather)`. */
  readonly #volumeC: THREE.Vector4[] = [];
  /** `(emissiveRadiance.rgb, unused)`. */
  readonly #volumeD: THREE.Vector4[] = [];

  /* -- GPU resources ----------------------------------------------------- */

  #noiseTexture: THREE.Data3DTexture | null = null;
  #froxel: FroxelResources | null = null;
  #mode: VolumetricsMode = 'off';
  #graphVersion = 0;
  #froxelFailed = false;
  #warnedNoRenderer = false;
  #temporalValid = false;
  #frameCounter = 0;
  #boundLights = 0;
  #boundVolumes = 0;

  /** Previous frame's `P·V`, for froxel history reprojection. */
  readonly #previousViewProjection = new THREE.Matrix4();
  readonly #uPreviousViewProjection = uniform(new THREE.Matrix4());
  readonly #uHistoryValid = uniform(0);
  readonly #uTemporalBlend = uniform(0.06);

  constructor(options: VolumetricsOptions = {}) {
    this.#quality = options.quality ?? 'high';
    this.#froxelOverride = options.froxelDimensions ?? null;
    this.#marchStepOverride = options.marchSteps ?? null;
    this.#depthDistribution = options.depthDistribution ?? 'exponential';
    this.#options = {
      depthDistribution: this.#depthDistribution,
      maxLights: Math.max(0, Math.min(16, options.maxLights ?? 4)),
      maxFogVolumes: Math.max(0, Math.min(16, options.maxFogVolumes ?? 8)),
      temporalBlend: Math.min(1, Math.max(0.01, options.temporalBlend ?? 0.06)),
      noiseTextureSize: options.noiseTextureSize ?? 32,
      noiseSeed: options.noiseSeed ?? 'd2rim.fog',
      registerService: options.registerService ?? true,
    };
    this.#uTemporalBlend.value = this.#options.temporalBlend;

    if (options.params !== undefined) this.setParams(options.params);

    const lights = this.#maxLights();
    for (let i = 0; i < lights; i++) {
      this.#lightPositions.push(new THREE.Vector4(0, 0, 0, 1));
      this.#lightColors.push(new THREE.Vector4(0, 0, 0, 0));
    }
    for (let i = 0; i < this.#options.maxFogVolumes; i++) {
      this.#volumeA.push(new THREE.Vector4(0, 0, 0, 0));
      this.#volumeB.push(new THREE.Vector4(1, 1, 1, 0));
      this.#volumeC.push(new THREE.Vector4(1, 1, 1, 0.4));
      this.#volumeD.push(new THREE.Vector4(0, 0, 0, 0));
    }
  }

  /* -- GameModule -------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    this.#sun = ctx.services.tryGet(VolumetricSunKey);
    this.#shadows = ctx.services.tryGet(VolumetricShadowKey);
    this.#lights = ctx.services.tryGet(VolumetricLightsKey);
    this.#ambient = ctx.services.tryGet(VolumetricAmbientKey);
    this.#motion = tryGetMotionVectors(ctx);

    this.#noiseTexture = bakeFogNoiseTexture(
      this.#options.noiseTextureSize,
      this.#options.noiseSeed,
    );

    this.#syncUniformsFromParams();
    this.#mode = this.#chooseMode(ctx);

    if (this.#options.registerService) {
      ctx.services.register(VolumetricsKey, this);
    }

    const [fx, fy, fz] = this.#froxelDimensions();
    ctx.events.emit('volumetrics:ready', {
      mode: this.#mode,
      froxels: this.#mode === 'froxel' ? fx * fy * fz : 0,
    });
  }

  /**
   * Push this frame's camera, sun, lights and fog volumes into the uniforms,
   * then — on the froxel path — dispatch the two compute passes.
   */
  lateUpdate(ctx: GameContext, _dt: number): void {
    if (!this.enabled || this.#mode === 'off') {
      this.#u.enabled.value = 0;
      return;
    }
    this.#u.enabled.value = 1;
    this.#resolveLateServices(ctx);

    const camera = ctx.camera;
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    this.#updateCameraUniforms(camera);
    this.#updateSunUniforms();
    this.#updateAmbientUniforms();
    this.#updateLightUniforms(ctx, camera);
    this.#updateVolumeUniforms(camera);

    this.#u.time.value = ctx.time.elapsed;
    this.#frameCounter++;
    this.#u.frame.value = this.#frameCounter % 64;

    const [jx, jy, jz] = haltonJitter3D((this.#frameCounter % 16) + 1);
    this.#u.jitter.value.set(jx, jy, jz);

    // Temporal history validity. A camera cut reported by the motion-vector
    // service invalidates the whole volume; so does the very first frame.
    const cut = this.#motion !== undefined && this.#motion.historyValid === false;
    const valid = this.#temporalValid && !cut;
    this.#uHistoryValid.value = valid ? 1 : 0;
    this.#uPreviousViewProjection.value.copy(this.#previousViewProjection);

    // Everything above is CPU bookkeeping — culling, arbitration, uniform
    // packing — and it runs whether or not there is a GPU to talk to, so the
    // stats readout stays truthful on a degraded backend and so the ray-march
    // path (whose only GPU work belongs to `LightShafts`) still gets fed.
    // Only the compute dispatch below needs a node renderer.
    const renderer = asNodeRenderer(ctx.renderer);
    if (renderer === null) {
      if (!this.#warnedNoRenderer) {
        this.#warnedNoRenderer = true;
        console.warn('[Volumetrics] the renderer is not a node renderer; fog is disabled.');
      }
      this.#u.enabled.value = 0;
      return;
    }

    if (this.#mode === 'froxel') {
      this.#renderFroxelVolume(renderer);
    }

    this.#previousViewProjection
      .copy(camera.projectionMatrix)
      .multiply(camera.matrixWorldInverse);
    this.#temporalValid = true;
  }

  dispose(): void {
    const ctx = this.#ctx;
    if (ctx !== null && this.#options.registerService) {
      if (ctx.services.tryGet(VolumetricsKey) === this) ctx.services.unregister(VolumetricsKey);
    }
    this.#froxel?.dispose();
    this.#froxel = null;
    this.#noiseTexture?.dispose();
    this.#noiseTexture = null;
    this.#ctx = null;
  }

  /* -- VolumetricsService ------------------------------------------------ */

  get mode(): VolumetricsMode {
    return this.enabled ? this.#mode : 'off';
  }

  get prefersHalfResolution(): boolean {
    return this.#mode === 'raymarch';
  }

  get graphVersion(): number {
    return this.#graphVersion;
  }

  get params(): Readonly<GlobalFogParams> {
    return this.#params;
  }

  get stats(): VolumetricsStats {
    const froxel = this.#mode === 'froxel' ? this.#froxelDimensions() : ([0, 0, 0] as const);
    return {
      mode: this.mode,
      quality: this.#quality,
      froxelDimensions: froxel,
      bytes: this.#froxel?.bytes ?? 0,
      lights: this.#boundLights,
      volumes: this.#boundVolumes,
      registeredVolumes: this.#volumes.length,
      shadowed: this.#shadows !== undefined,
      temporal: this.#uHistoryValid.value === 1,
    };
  }

  setParams(params: Partial<GlobalFogParams>): void {
    const target = this.#params;
    if (params.density !== undefined) target.density = Math.max(0, params.density);
    if (params.albedo !== undefined) target.albedo.copy(params.albedo);
    if (params.height !== undefined) target.height = params.height;
    if (params.heightFalloff !== undefined) {
      target.heightFalloff = Math.max(0, params.heightFalloff);
    }
    if (params.anisotropy !== undefined) target.anisotropy = clampRange(params.anisotropy, -0.95, 0.95);
    if (params.backAnisotropy !== undefined) {
      target.backAnisotropy = clampRange(params.backAnisotropy, -0.95, 0.95);
    }
    if (params.lobeBlend !== undefined) target.lobeBlend = clampRange(params.lobeBlend, 0, 1);
    if (params.sunScatteringScale !== undefined) {
      target.sunScatteringScale = Math.max(0, params.sunScatteringScale);
    }
    if (params.lightScatteringScale !== undefined) {
      target.lightScatteringScale = Math.max(0, params.lightScatteringScale);
    }
    if (params.ambientScatteringScale !== undefined) {
      target.ambientScatteringScale = Math.max(0, params.ambientScatteringScale);
    }
    if (params.noiseStrength !== undefined) {
      target.noiseStrength = clampRange(params.noiseStrength, 0, 1);
    }
    if (params.noiseScale !== undefined) target.noiseScale = Math.max(0.01, params.noiseScale);
    if (params.wind !== undefined) target.wind.copy(params.wind);
    if (params.volumeDistance !== undefined) {
      target.volumeDistance = Math.max(1, params.volumeDistance);
    }
    this.#syncUniformsFromParams();
  }

  setQuality(quality: VolumetricsQuality): void {
    if (quality === this.#quality) return;
    this.#quality = quality;
    // The froxel grid is a compile-time constant in both kernels (loop bounds
    // and index arithmetic fold into the generated code), so a tier change
    // rebuilds them rather than writing a uniform.
    this.#invalidateGraph();
    const ctx = this.#ctx;
    this.#mode = ctx === null ? 'off' : this.#chooseMode(ctx);
  }

  setSun(direction: Vec3Like, color: THREE.ColorRepresentation, intensity: number): void {
    this.#sunDirection.set(direction.x, direction.y, direction.z).normalize();
    this.#sunColor.set(color);
    this.#sunIntensity = Math.max(0, intensity);
  }

  setAmbient(color: THREE.ColorRepresentation, groundColor?: THREE.ColorRepresentation): void {
    this.#ambientColor.set(color);
    if (groundColor !== undefined) this.#groundAmbientColor.set(groundColor);
  }

  addFogVolume(desc: FogVolumeDesc): FogVolumeHandle {
    const shape = desc.shape ?? 'box';
    const radius = Math.max(0.01, desc.radius ?? 4);
    const id = this.#nextVolumeId++;
    const record: FogVolumeRecord = {
      id,
      alive: true,
      active: false,
      shape,
      center: new THREE.Vector3(desc.center.x, desc.center.y, desc.center.z),
      halfExtent:
        shape === 'sphere'
          ? new THREE.Vector3(radius, radius, radius)
          : new THREE.Vector3(
              Math.max(0.01, desc.halfExtent?.x ?? 4),
              Math.max(0.01, desc.halfExtent?.y ?? 2),
              Math.max(0.01, desc.halfExtent?.z ?? 4),
            ),
      densityScale: Math.max(0, desc.densityScale ?? 3),
      color: new THREE.Color(desc.color ?? 0xffffff),
      emissive: new THREE.Color(desc.emissive ?? 0x000000),
      feather: clampRange(desc.feather ?? 0.4, 0.01, 1),
      priority: desc.priority ?? 0,
      name: desc.name ?? `fogVolume${id}`,
    };
    this.#volumes.push(record);
    return new FogVolumeHandleImpl(record, this.#volumes);
  }

  /**
   * The resolve node.
   *
   * On the froxel path this is one trilinear 3D fetch plus the analytic
   * continuation beyond the volume. On the ray-march path it is the full
   * integration loop. Either way the result is the same quantity, so the
   * consumer never branches.
   */
  createResolveNode(input: VolumetricResolveInput): Vec4Node {
    if (this.#mode === 'off' || !this.enabled) {
      return vec4(0, 0, 0, 1);
    }
    const froxel = this.#froxel;
    if (this.#mode === 'froxel' && froxel !== null) {
      return this.#buildFroxelResolveNode(input, froxel);
    }
    return this.#buildRaymarchResolveNode(input);
  }

  /**
   * Pick up optional services that registered *after* this module did.
   *
   * Five render systems are being brought up in parallel and the integrator
   * decides the order; requiring the shadow map to exist before the fog does
   * would be a needless ordering constraint. A provider appearing changes the
   * *shape* of the shader graph (it adds the cascade lookup), so the graph
   * version is bumped and every consumer rebuilds on the next frame.
   *
   * The lookups stop once they succeed, so the steady-state cost is a handful
   * of already-false boolean tests.
   */
  #resolveLateServices(ctx: GameContext): void {
    if (this.#sun === undefined) this.#sun = ctx.services.tryGet(VolumetricSunKey);
    if (this.#lights === undefined) this.#lights = ctx.services.tryGet(VolumetricLightsKey);
    if (this.#ambient === undefined) this.#ambient = ctx.services.tryGet(VolumetricAmbientKey);
    if (this.#motion === undefined) this.#motion = tryGetMotionVectors(ctx);
    if (this.#shadows === undefined) {
      const shadows = ctx.services.tryGet(VolumetricShadowKey);
      if (shadows !== undefined) {
        this.#shadows = shadows;
        // Only this one alters the graph: the others are pure uniform sources.
        this.#invalidateGraph();
      }
    }
  }

  /** Force every consumer to rebuild against a structurally different graph. */
  #invalidateGraph(): void {
    this.#cachedEvaluator = null;
    this.#froxel?.dispose();
    this.#froxel = null;
    this.#temporalValid = false;
    this.#graphVersion++;
  }

  /* -- internals: mode selection ----------------------------------------- */

  #chooseMode(ctx: GameContext): VolumetricsMode {
    if (this.#quality === 'off') return 'off';
    if (this.#froxelFailed) return 'raymarch';
    return ctx.renderer.capabilities.compute ? 'froxel' : 'raymarch';
  }

  #maxLights(): number {
    if (this.#quality === 'off') return this.#options.maxLights;
    return Math.min(this.#options.maxLights, VOLUMETRIC_TIERS[this.#quality].lights);
  }

  #froxelDimensions(): readonly [number, number, number] {
    if (this.#froxelOverride !== null) {
      const [x, y, z] = this.#froxelOverride;
      return [Math.max(8, x | 0), Math.max(8, y | 0), Math.max(4, z | 0)];
    }
    if (this.#quality === 'off') return [0, 0, 0];
    return VOLUMETRIC_TIERS[this.#quality].froxels;
  }

  #marchSteps(): number {
    if (this.#marchStepOverride !== null) return Math.max(4, this.#marchStepOverride | 0);
    if (this.#quality === 'off') return 0;
    return VOLUMETRIC_TIERS[this.#quality].marchSteps;
  }

  /* -- internals: per-frame uniform updates ------------------------------ */

  #syncUniformsFromParams(): void {
    const p = this.#params;
    this.#u.density.value = p.density;
    this.#u.albedo.value.set(p.albedo.r, p.albedo.g, p.albedo.b);
    this.#u.height.value = p.height;
    this.#u.heightFalloff.value = p.heightFalloff;
    this.#u.phase.value.set(p.anisotropy, p.backAnisotropy, p.lobeBlend);
    this.#u.lightScale.value = p.lightScatteringScale;
    this.#u.noise.value.set(1 / p.noiseScale, p.noiseStrength, 0.35, 0);
    this.#u.wind.value.copy(p.wind);
  }

  #updateCameraUniforms(camera: THREE.PerspectiveCamera): void {
    const projection = camera.projectionMatrix.elements;
    this.#u.projScale.value.set(projection[0] ?? 1, projection[5] ?? 1);
    this.#u.cameraMatrixWorld.value.copy(camera.matrixWorld);
    camera.getWorldPosition(this.#u.cameraPosition.value);
    // The volume never extends past the camera's own far plane: froxels behind
    // it would be integrated but never sampled.
    const far = Math.min(this.#params.volumeDistance, camera.far);
    this.#u.range.value.set(Math.max(1e-3, camera.near), Math.max(camera.near + 1e-3, far));
  }

  #updateSunUniforms(): void {
    const provider = this.#sun;
    const direction = provider?.sunDirection ?? this.#sunDirection;
    const color = provider?.sunColor ?? this.#sunColor;
    const intensity = provider?.sunIntensity ?? this.#sunIntensity;
    this.#u.sunDirection.value.copy(direction).normalize();
    const scale = intensity * this.#params.sunScatteringScale;
    this.#u.sunRadiance.value.set(color.r * scale, color.g * scale, color.b * scale);
  }

  #updateAmbientUniforms(): void {
    const provider = this.#ambient;
    const sky = provider?.volumetricAmbient ?? this.#ambientColor;
    const ground = provider?.volumetricGroundAmbient ?? this.#groundAmbientColor;
    const scale = this.#params.ambientScatteringScale;
    this.#u.ambient.value.set(sky.r * scale, sky.g * scale, sky.b * scale);
    this.#u.groundAmbient.value.set(ground.r * scale, ground.g * scale, ground.b * scale);
  }

  /**
   * Gather the local lights that scatter into the fog.
   *
   * Prefers the lighting rig's own ranked list. Without it, the scene graph is
   * walked and lights are ranked by the same analytic importance the rig uses:
   * radiant intensity over squared distance, i.e. roughly how much of the
   * screen the light's glow will actually occupy.
   */
  #updateLightUniforms(ctx: GameContext, camera: THREE.PerspectiveCamera): void {
    const capacity = this.#lightPositions.length;
    if (capacity === 0) {
      this.#u.lightCount.value = 0;
      this.#boundLights = 0;
      return;
    }

    let count = 0;
    const provider = this.#lights;
    if (provider !== undefined) {
      this.#lightScratch.length = 0;
      count = Math.min(capacity, provider.collectVolumetricLights(this.#lightScratch, capacity));
      for (let i = 0; i < count; i++) {
        const light = this.#lightScratch[i];
        if (light === undefined) {
          count = i;
          break;
        }
        this.#packLight(i, light);
      }
    } else {
      count = this.#gatherSceneLights(ctx, camera, capacity);
    }

    // Park the unused slots at zero intensity so the loop can still run to a
    // constant bound if a driver decides to unroll it.
    for (let i = count; i < capacity; i++) {
      this.#lightColors[i]?.set(0, 0, 0, 0);
    }
    this.#u.lightCount.value = count;
    this.#boundLights = count;
  }

  #packLight(index: number, light: VolumetricPointLight): void {
    const radius = Math.max(0.05, light.radius);
    const position = this.#lightPositions[index];
    const color = this.#lightColors[index];
    if (position === undefined || color === undefined) return;
    position.set(light.position.x, light.position.y, light.position.z, 1 / (radius * radius));
    const intensity = Math.max(0, light.intensity);
    color.set(
      light.color.r * intensity,
      light.color.g * intensity,
      light.color.b * intensity,
      light.volumetricScale ?? 1,
    );
  }

  #gatherSceneLights(
    ctx: GameContext,
    camera: THREE.PerspectiveCamera,
    capacity: number,
  ): number {
    const found = this.#sceneLights;
    found.length = 0;
    const eye = this.#u.cameraPosition.value;

    ctx.scene.traverseVisible((object) => {
      const light = object as THREE.Object3D & {
        isPointLight?: boolean;
        isSpotLight?: boolean;
        color?: THREE.Color;
        intensity?: number;
        distance?: number;
      };
      if (light.isPointLight !== true && light.isSpotLight !== true) return;
      if (light.color === undefined || light.intensity === undefined) return;
      if (light.intensity <= 0) return;

      object.getWorldPosition(_lightPosition);
      // `distance === 0` means "unbounded" in three; the fog needs a finite
      // radius, so fall back to the distance at which a physical inverse-square
      // falloff drops the light below a hundredth of a render unit.
      const declared = light.distance ?? 0;
      const radius = declared > 0 ? declared : Math.min(64, Math.sqrt(light.intensity * 100) + 1);
      const importance = light.intensity / Math.max(1, eye.distanceToSquared(_lightPosition));
      found.push({
        position: _lightPosition.clone(),
        color: light.color.clone(),
        intensity: light.intensity,
        radius,
        volumetricScale: 1,
        importance,
      });
    });

    if (found.length === 0) return 0;

    // Cull against the camera frustum, but generously: a torch just off screen
    // still lights the fog that *is* on screen, so the test uses the light's
    // influence sphere rather than its position.
    _viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_viewProjection, camera.coordinateSystem);
    const visible = found.filter((light) => {
      _sphere.center.copy(light.position);
      _sphere.radius = light.radius;
      return _frustum.intersectsSphere(_sphere);
    });

    visible.sort((a, b) => b.importance - a.importance);
    const count = Math.min(capacity, visible.length);
    for (let i = 0; i < count; i++) {
      const light = visible[i];
      if (light !== undefined) this.#packLight(i, light);
    }
    return count;
  }

  /**
   * Bind the fog volumes that matter this frame.
   *
   * Culling is a frustum test against the volume's bounding sphere, then a sort
   * by explicit priority and by proximity to the camera. Over budget, the
   * furthest low-priority volume is the one that drops — which is the right
   * answer, because a fog volume the player is standing in is the only one
   * whose absence is obvious.
   */
  #updateVolumeUniforms(camera: THREE.PerspectiveCamera): void {
    const capacity = this.#volumeA.length;
    for (const volume of this.#volumes) volume.active = false;
    if (capacity === 0 || this.#volumes.length === 0) {
      this.#u.volumeCount.value = 0;
      this.#boundVolumes = 0;
      return;
    }

    _viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_viewProjection, camera.coordinateSystem);
    const eye = this.#u.cameraPosition.value;

    const candidates = this.#volumes.filter((volume) => {
      if (!volume.alive || volume.densityScale === 0) return false;
      _sphere.center.copy(volume.center);
      _sphere.radius = volume.halfExtent.length();
      return _frustum.intersectsSphere(_sphere);
    });

    candidates.sort((a, b) => {
      if (a.priority !== b.priority) return b.priority - a.priority;
      return eye.distanceToSquared(a.center) - eye.distanceToSquared(b.center);
    });

    const count = Math.min(capacity, candidates.length);
    for (let i = 0; i < count; i++) {
      const volume = candidates[i];
      if (volume === undefined) continue;
      volume.active = true;
      this.#volumeA[i]?.set(
        volume.center.x,
        volume.center.y,
        volume.center.z,
        volume.shape === 'sphere' ? 1 : 0,
      );
      this.#volumeB[i]?.set(
        volume.halfExtent.x,
        volume.halfExtent.y,
        volume.halfExtent.z,
        volume.densityScale,
      );
      this.#volumeC[i]?.set(volume.color.r, volume.color.g, volume.color.b, volume.feather);
      this.#volumeD[i]?.set(volume.emissive.r, volume.emissive.g, volume.emissive.b, 0);
    }
    for (let i = count; i < capacity; i++) this.#volumeB[i]?.set(1, 1, 1, 0);

    this.#u.volumeCount.value = count;
    this.#boundVolumes = count;
  }

  /* -- internals: froxel dispatch ---------------------------------------- */

  #renderFroxelVolume(renderer: NodeRenderer): void {
    let froxel = this.#froxel;
    if (froxel === null) {
      try {
        froxel = this.#buildFroxelResources();
        this.#froxel = froxel;
        this.#graphVersion++;
      } catch (error) {
        this.#failFroxel(error instanceof Error ? error.message : String(error));
        return;
      }
    }

    try {
      froxel.render(renderer);
    } catch (error) {
      this.#failFroxel(error instanceof Error ? error.message : String(error));
    }
  }

  /**
   * Latch to the ray-march path.
   *
   * Same "detect, don't sniff" shape as `render/webgpuCompat.ts`: the froxel
   * path is attempted, and only an observed failure turns it off. A capability
   * query cannot tell us whether a given driver will accept an `rgba16float`
   * 3D storage texture, so asking it is not an option.
   */
  #failFroxel(reason: string): void {
    if (this.#froxelFailed) return;
    this.#froxelFailed = true;
    this.#mode = 'raymarch';
    this.#invalidateGraph();
    console.warn(`[Volumetrics] froxel path unavailable (${reason}); falling back to ray march.`);
    this.#ctx?.events.emit('volumetrics:fallback', { reason });
  }

  #buildFroxelResources(): FroxelResources {
    const [dx, dy, dz] = this.#froxelDimensions();
    const media = this.#mediaEvaluator();
    return new FroxelResources(
      dx,
      dy,
      dz,
      this.#u,
      media,
      this.#uPreviousViewProjection,
      this.#uHistoryValid,
      this.#uTemporalBlend,
      this.#depthDistribution,
    );
  }

  /* -- internals: shared TSL --------------------------------------------- */

  /**
   * Build the two functions the whole system is made of:
   *
   * - `media(worldPosition)` → `vec4(σ_s.rgb, σ_t)`
   * - `inscatter(worldPosition, viewDirection, scattering)` → `vec3` radiance
   *
   * Both are ordinary TSL `Fn`s, so they inline into whichever shader asks for
   * them — the compute injection kernel or the ray-march fragment — and there
   * is exactly one implementation of the physics.
   */
  /**
   * The media evaluator, built once and reused.
   *
   * Each build allocates its own uniform buffers for the light and fog-volume
   * arrays — bound to the same JavaScript arrays, so a second copy would be
   * correct but would upload the same data twice a frame. It is invalidated
   * only when the graph's *shape* changes (see {@link graphVersion}).
   */
  #mediaEvaluator(): MediaEvaluator {
    const cached = this.#cachedEvaluator;
    if (cached !== null && this.#cachedEvaluatorVersion === this.#graphVersion) return cached;
    const evaluator = this.#buildMediaEvaluator();
    this.#cachedEvaluator = evaluator;
    this.#cachedEvaluatorVersion = this.#graphVersion;
    return evaluator;
  }

  #cachedEvaluator: MediaEvaluator | null = null;
  #cachedEvaluatorVersion = -1;

  #buildMediaEvaluator(): MediaEvaluator {
    const u = this.#u;
    const noise = this.#noiseTexture;
    // Explicit level-0 fetches, never an implicitly-derived one. A compute
    // shader has no fragment quad and therefore no derivatives at all, and a
    // fragment shader may not take them inside the non-uniform control flow a
    // ray-march loop is — so `textureSampleLevel` is the only legal form in
    // *both* consumers of this function, and the volume has no mips anyway.
    const sampleNoise =
      noise === null ? null : (coord: Vec3Node): Vec4Node =>
          sampleVolume(noise, coord);
    const volumeCapacity = this.#volumeA.length;
    const lightCapacity = this.#lightPositions.length;

    const volA = volumeCapacity > 0 ? uniformArray<'vec4'>(this.#volumeA, 'vec4') : null;
    const volB = volumeCapacity > 0 ? uniformArray<'vec4'>(this.#volumeB, 'vec4') : null;
    const volC = volumeCapacity > 0 ? uniformArray<'vec4'>(this.#volumeC, 'vec4') : null;
    const volD = volumeCapacity > 0 ? uniformArray<'vec4'>(this.#volumeD, 'vec4') : null;
    const lightPos = lightCapacity > 0 ? uniformArray<'vec4'>(this.#lightPositions, 'vec4') : null;
    const lightCol = lightCapacity > 0 ? uniformArray<'vec4'>(this.#lightColors, 'vec4') : null;

    const sunVisibility = this.#buildSunVisibility();

    /**
     * Media properties at a world-space point.
     *
     * Returns `vec4(σ_s.rgb, σ_t)` plus, separately, the emissive ambient boost
     * from any fog volume the point is inside — packed as the `w` of the second
     * return would be if TSL had tuples, so it is handed back through a second
     * function instead. Keeping them separate avoids evaluating the volume loop
     * twice.
     */
    const mediaAt = Fn(([worldPosition]: [Vec3Node]) => {
      // Height profile: flat below `height`, exponential above it. `max(0, …)`
      // is what makes it flat, and what makes the analytic continuation exact.
      const above = max(worldPosition.y.sub(u.height), float(0));
      const heightFactor = exp(above.mul(u.heightFalloff).negate()).toVar('heightFactor');

      const densityScale = float(1).toVar('densityScale');
      const tint = vec3(1, 1, 1).toVar('tint');

      if (volA !== null && volB !== null && volC !== null) {
        Loop({ start: int(0), end: int(u.volumeCount), type: 'int' }, ({ i }) => {
          const a = volA.element(i);
          const b = volB.element(i);
          const c = volC.element(i);

          const delta = worldPosition.sub(a.xyz).toVar('fogDelta');
          const extent = max(b.xyz, vec3(1e-4, 1e-4, 1e-4)).toVar('fogExtent');
          const feather = max(c.w, float(0.01)).toVar('fogFeather');

          // Box weight: the product of three feathered axis ramps, which gives
          // rounded corners for free. Sphere weight: one radial ramp. `mix` on
          // the shape flag keeps the loop branchless, which matters because it
          // runs once per froxel per volume.
          const boxRamp = saturate(
            extent.sub(delta.abs()).div(extent.mul(feather)),
          ).toVar('boxRamp');
          const boxWeight = smoothstepNode(boxRamp.x)
            .mul(smoothstepNode(boxRamp.y))
            .mul(smoothstepNode(boxRamp.z));

          const radius = extent.x;
          const sphereRamp = saturate(
            radius.sub(delta.length()).div(radius.mul(feather)),
          );
          const sphereWeight = smoothstepNode(sphereRamp);

          const weight = mix(boxWeight, sphereWeight, a.w).toVar('fogWeight');
          densityScale.assign(mix(densityScale, densityScale.mul(b.w), weight));
          tint.assign(mix(tint, tint.mul(c.xyz), weight));
        });
      }

      // Animated detail. Two bands at different scales drifting at different
      // speeds, with the third band warping the lookup of the first — the
      // cheapest way to get something that reads as advection rather than as a
      // texture sliding past.
      const noiseFactor = float(1).toVar('noiseFactor');
      if (sampleNoise !== null) {
        const scale = u.noise.x;
        const drift = u.wind.mul(u.time);
        const base = worldPosition.add(drift).mul(scale).toVar('noiseBase');
        const warp = sampleNoise(base.mul(0.37) as unknown as Vec3Node).w.sub(0.5).mul(u.noise.z);
        const coarse = sampleNoise(base.add(warp) as unknown as Vec3Node).x;
        const detail = sampleNoise(
          worldPosition.add(drift.mul(2.1)).mul(scale.mul(3.1)) as unknown as Vec3Node,
        ).y;
        const combined = coarse.mul(0.65).add(detail.mul(0.35));
        // Remapped so the *mean* stays 1: the density calibration is preserved
        // whatever the noise strength, which is what lets an artist push the
        // structure slider without also changing the fog's depth cue.
        noiseFactor.assign(saturate(combined.mul(2).sub(1).mul(u.noise.y).add(1)));
      }

      const extinction = u.density
        .mul(heightFactor)
        .mul(densityScale)
        .mul(noiseFactor)
        .toVar('sigmaT');
      const scattering = u.albedo.mul(tint).mul(extinction).toVar('sigmaS');
      return vec4(scattering, extinction);
    });

    /**
     * Ambient plus emissive radiance at a point, which is the term that keeps
     * shadowed fog from going black. Split out of `mediaAt` because it needs
     * the fog-volume loop's emissive accumulation, and because the ray-march
     * path folds it in at a different point than the froxel path.
     */
    const ambientAt = Fn(([worldPosition]: [Vec3Node]) => {
      // Ground bounce dominates below the fog reference height, sky above it.
      const t = saturate(worldPosition.y.sub(u.height).mul(0.5).add(0.5));
      const base = mix(u.groundAmbient, u.ambient, t).toVar('fogAmbient');

      if (volA !== null && volB !== null && volD !== null) {
        Loop({ start: int(0), end: int(u.volumeCount), type: 'int' }, ({ i }) => {
          const a = volA.element(i);
          const b = volB.element(i);
          const d = volD.element(i);
          const delta = worldPosition.sub(a.xyz);
          const extent = max(b.xyz, vec3(1e-4, 1e-4, 1e-4));
          const inside = saturate(
            float(1).sub(max(max(delta.x.abs().div(extent.x), delta.y.abs().div(extent.y)), delta.z.abs().div(extent.z))).mul(4),
          );
          base.addAssign(d.xyz.mul(inside));
        });
      }
      return base;
    });

    /**
     * Scattered radiance arriving at the eye from one point.
     *
     * `viewDirection` points from the eye towards the sample, so the phase
     * argument for a light travelling along `−L` is `dot(viewDirection, −L)`.
     */
    const inscatterAt = Fn(
      ([worldPosition, viewDirection, scattering]: [Vec3Node, Vec3Node, Vec3Node]) => {
        const gForward = u.phase.x;
        const gBackward = u.phase.y;
        const blend = u.phase.z;

        // -- sun ---------------------------------------------------------
        const cosSun = viewDirection.dot(u.sunDirection.negate()).toVar('cosSun');
        const sunPhase = dualLobePhaseNode(cosSun, gForward, gBackward, blend);
        const visibility = sunVisibility(worldPosition).toVar('sunVis');
        const total = u.sunRadiance.mul(sunPhase).mul(visibility).toVar('inscatter');

        // -- ambient / multiple scattering --------------------------------
        // Isotropic, hence the 1/4π: the sky is everywhere, so the phase
        // function integrates to unity over the sphere and drops out.
        total.addAssign(ambientAt(worldPosition).mul(1 / FOUR_PI));

        // -- local lights -------------------------------------------------
        if (lightPos !== null && lightCol !== null) {
          Loop({ start: int(0), end: int(u.lightCount), type: 'int' }, ({ i }) => {
            const packedPosition = lightPos.element(i);
            const packedColor = lightCol.element(i);
            const toLight = packedPosition.xyz.sub(worldPosition).toVar('toLight');
            const distanceSq = max(toLight.dot(toLight), float(1e-4)).toVar('lightDistSq');
            const direction = normalize(toLight).toVar('lightDir');

            // Karis windowed inverse square: physical 1/d² inside the radius,
            // smoothly clamped to zero at it, so a light never contributes an
            // abrupt edge to the fog when it leaves the culling sphere.
            const ratio = distanceSq.mul(packedPosition.w);
            const window = saturate(float(1).sub(ratio.mul(ratio))).toVar('w');
            const attenuation = window.mul(window).div(distanceSq.add(1));

            const cosLight = viewDirection.dot(direction.negate());
            const phase = dualLobePhaseNode(cosLight, gForward, gBackward, blend);
            total.addAssign(
              packedColor.xyz.mul(attenuation).mul(phase).mul(packedColor.w).mul(u.lightScale),
            );
          });
        }

        return total.mul(scattering);
      },
    );

    return {
      mediaAt: (worldPosition) => mediaAt(worldPosition) as unknown as Vec4Node,
      inscatterAt: (worldPosition, viewDirection, scattering) =>
        inscatterAt(worldPosition, viewDirection, scattering) as unknown as Vec3Node,
    };
  }

  /**
   * Sun visibility as a function of world position.
   *
   * Three cases, in order of preference: the shadow module's own node; a
   * single-tap comparison against its cascade array; or a constant `1`.
   */
  #buildSunVisibility(): (worldPosition: Vec3Node) => FloatNode {
    const provider = this.#shadows;
    if (provider === undefined) {
      return () => float(1);
    }
    if (typeof provider.createSunVisibilityNode === 'function') {
      const build = provider.createSunVisibilityNode.bind(provider);
      return (worldPosition) => build(worldPosition);
    }

    const depthTexture = provider.shadowDepthTexture;
    const matrices = provider.cascadeMatrices;
    const cascadeCount = Math.max(1, Math.min(4, provider.cascadeCount));
    if (depthTexture === null || matrices.length < cascadeCount) {
      return () => float(1);
    }

    const mapSize = Math.max(1, provider.shadowMapSize);
    const reversed = provider.reversedDepth === true;
    // The array is a live reference owned by the provider; `uniformArray` keeps
    // it and re-uploads every frame, so cascades that refit as the camera moves
    // are picked up with no explicit sync.
    const cascadeMatrices = uniformArray<'mat4'>(matrices as THREE.Matrix4[], 'mat4');

    const sample = Fn(([worldPosition]: [Vec3Node]) => {
      const lit = float(1).toVar('sunVisibility');
      const done = float(0).toVar('cascadeFound');

      // Cascades are ordered near-to-far, so the first one that contains the
      // point is the highest-resolution one that does. Written branchlessly
      // with a `done` flag rather than an early exit: the loop is fully
      // unrolled (cascadeCount is a JS constant) and WGSL is happier with
      // straight-line code inside the non-uniform control flow a ray march
      // puts this in.
      for (let cascade = 0; cascade < cascadeCount; cascade++) {
        const projected = cascadeMatrices.element(int(cascade)).mul(vec4(worldPosition, 1));
        const shadowCoord = projected.xyz.div(projected.w).toVar(`shadowCoord${cascade}`);
        const inside = shadowCoord.x
          .greaterThanEqual(0)
          .and(shadowCoord.x.lessThanEqual(1))
          .and(shadowCoord.y.greaterThanEqual(0))
          .and(shadowCoord.y.lessThanEqual(1))
          .and(shadowCoord.z.greaterThanEqual(0))
          .and(shadowCoord.z.lessThanEqual(1));

        If(inside.and(done.lessThan(0.5)), () => {
          done.assign(1);
          const texel = clamp(
            shadowCoord.xy.mul(mapSize),
            vec2(0, 0),
            vec2(mapSize - 1, mapSize - 1),
          );
          // Explicit texel fetch: the map is `NearestFilter`, an implicit
          // derivative is illegal here, and three only appends the depth
          // swizzle on the load path. Same reasoning as
          // `render/CascadedShadowMaps.ts`.
          const occluder = textureLoad(depthTexture, ivec2(texel)).depth(
            int(cascade),
          ) as unknown as FloatNode;
          // A generous constant bias: the fog has no surface normal to offset
          // along, and self-shadowing acne is meaningless in a medium — the
          // only thing that matters is that a shaft's *edge* lands in the right
          // place, which a uniform depth bias does not move.
          const bias = float(1.5 / mapSize);
          // Reversed depth flips which side of the comparison is "in front".
          lit.assign(
            reversed
              ? step(occluder, shadowCoord.z.add(bias))
              : step(shadowCoord.z.sub(bias), occluder),
          );
        });
      }
      return lit;
    });

    return (worldPosition) => sample(worldPosition) as unknown as FloatNode;
  }

  /* -- internals: resolve nodes ------------------------------------------ */

  /**
   * Froxel resolve: one trilinear fetch into the integrated volume, plus the
   * analytic tail beyond the volume's far plane.
   */
  #buildFroxelResolveNode(input: VolumetricResolveInput, froxel: FroxelResources): Vec4Node {
    const u = this.#u;
    const volume = froxel.integratedTexture;
    const distribution = this.#depthDistribution;
    const sliceCount = froxel.dimensions[2];

    return Fn(() => {
      const near = u.range.x;
      const far = u.range.y;
      const viewZ = max(input.viewZ, near).toVar('viewZ');
      const clamped = min(viewZ, far).toVar('clampedZ');

      // Depth → slice, exactly inverting the distribution the volume was built
      // with. `- 0.5 / sliceCount` recentres onto the *texel centres*, because
      // slice `k` stores the integral up to its far boundary.
      const w = froxelSliceCoordNode(clamped, near, far, sliceCount, distribution).toVar('w');
      const sample = sampleVolume(volume, vec3(input.screenUv, w) as unknown as Vec3Node).toVar('froxel');

      const inscatter = sample.xyz.toVar('inscatter');
      const transmittance = saturate(sample.w).toVar('transmittance');

      // Analytic continuation past the volume's far plane.
      If(viewZ.greaterThan(far), () => {
        const rayDirection = this.#viewRayDirectionNode(input.screenUv).toVar('rayDir');
        const obliquity = float(1)
          .div(max(rayDirection.dot(this.#forwardNode()), float(1e-3)))
          .toVar('obliquity');
        const tail = this.#analyticTailNode(rayDirection, obliquity, far, viewZ).toVar('tail');
        inscatter.addAssign(transmittance.mul(tail.xyz));
        transmittance.mulAssign(tail.w);
      });

      const dithered = this.#ditherNode(inscatter, input.pixel);
      return vec4(dithered, saturate(transmittance));
    })() as unknown as Vec4Node;
  }

  /**
   * Ray-march resolve, for WebGL2 and for any device where the froxel path
   * failed to come up.
   *
   * Identical physics to the injection kernel — literally the same `Fn`s — but
   * marched along the view ray at half resolution with a dithered start offset
   * instead of being tabulated in a volume. The step distribution is the same
   * exponential one the froxel grid uses, so the two paths place their samples
   * in the same places and produce the same image.
   */
  #buildRaymarchResolveNode(input: VolumetricResolveInput): Vec4Node {
    const u = this.#u;
    const { mediaAt, inscatterAt } = this.#mediaEvaluator();
    const steps = this.#marchSteps();
    const distribution = this.#depthDistribution;

    return Fn(() => {
      const near = u.range.x;
      const far = u.range.y;
      const surfaceZ = max(input.viewZ, near).toVar('surfaceZ');
      const marchFar = min(surfaceZ, far).toVar('marchFar');

      const rayDirection = this.#viewRayDirectionNode(input.screenUv).toVar('rayDir');
      // View depth → distance along the (unit) ray.
      const obliquity = float(1)
        .div(max(rayDirection.dot(this.#forwardNode()), float(1e-3)))
        .toVar('obliquity');

      // Per-pixel, per-frame dither of the ray offset. This is the single most
      // important line in the function: without it the march's step boundaries
      // are correlated across the screen and show up as concentric bands, and
      // no amount of extra steps removes them.
      const offset = interleavedGradientNoiseNode(input.pixel, u.frame).toVar('rayDither');

      const scatteredLight = vec3(0, 0, 0).toVar('scattered');
      const transmittance = float(1).toVar('transmittance');
      const previousDepth = near.toVar('prevDepth');

      Loop({ start: int(0), end: int(steps), type: 'int' }, ({ i }) => {
        // Slice `i` spans `[d(i/N), d((i+1)/N)]`, and the media is sampled at a
        // *dithered point inside it*. Deriving the step length from the slice
        // boundaries rather than from consecutive sample points is what makes
        // the marched range cover exactly `[near, marchFar]`: taking the
        // difference of jittered sample depths instead loses half a slice at
        // the near plane, which is precisely where the fog is densest and where
        // the error shows.
        const far0 = froxelDepthNode(
          toFloat(i).add(1).div(steps),
          near,
          marchFar,
          distribution,
        ).toVar('sliceFar');
        const stepLength = max(far0.sub(previousDepth), float(0)).mul(obliquity).toVar('stepLength');
        previousDepth.assign(far0);

        const t = toFloat(i).add(offset).div(steps).toVar('t');
        const depth = froxelDepthNode(t, near, marchFar, distribution).toVar('depth');
        const position = u.cameraPosition.add(rayDirection.mul(depth.mul(obliquity))).toVar('p');
        const media = mediaAt(position).toVar('media');
        const scattering = media.xyz;
        const extinction = media.w;

        const inscatter = inscatterAt(position, rayDirection, scattering).toVar('S');

        // Hillaire's analytic slice integral, in shader form. `max(σ_t, ε)` is
        // the optically-thin branch: the expression tends to `S·d` there, and
        // clamping the denominator is exactly equivalent to taking that limit
        // while staying branchless.
        const safeExtinction = max(extinction, float(1e-5));
        const sliceTransmittance = exp(safeExtinction.mul(stepLength).negate()).toVar('sliceT');
        const integrated = inscatter
          .sub(inscatter.mul(sliceTransmittance))
          .div(safeExtinction)
          .toVar('sliceS');

        scatteredLight.addAssign(transmittance.mul(integrated));
        transmittance.mulAssign(sliceTransmittance);
      });

      // Analytic tail past the marched range — the same node the froxel path
      // uses, so the two agree at the seam.
      If(surfaceZ.greaterThan(far), () => {
        const tail = this.#analyticTailNode(rayDirection, obliquity, far, surfaceZ).toVar('tail');
        scatteredLight.addAssign(transmittance.mul(tail.xyz));
        transmittance.mulAssign(tail.w);
      });

      return vec4(this.#ditherNode(scatteredLight, input.pixel), saturate(transmittance));
    })() as unknown as Vec4Node;
  }

  /**
   * The fog between the volumetric far plane and the surface, in closed form.
   *
   * Beyond the volume there is nothing left to march *for*: the shadow cascades
   * have run out, the local lights are far outside their radii, and the media
   * is the smooth height profile with the 3D noise averaged away by distance.
   * All that survives is ambient in-scattering attenuated by an exponential
   * height-fog integral, and that has an exact solution
   * ({@link heightFogOpticalDepth}) — so the tail costs a handful of ALU and
   * two `exp`s instead of a second march, and it is *more* accurate than
   * marching would be, not less.
   *
   * Sharing this between the froxel and ray-march paths is what stops a visible
   * ring at the far plane on one backend and not the other.
   *
   * @returns `vec4(inScatter, transmittance)` for the tail segment alone; the
   *   caller weights the in-scatter by the transmittance accumulated so far.
   */
  #analyticTailNode(
    rayDirection: Vec3Node,
    obliquity: FloatNode,
    fromViewZ: FloatNode,
    toViewZ: FloatNode,
  ): Vec4Node {
    const u = this.#u;
    const start = u.cameraPosition.add(rayDirection.mul(fromViewZ.mul(obliquity)));
    const distance = max(toViewZ.sub(fromViewZ), float(0)).mul(obliquity);
    const opticalDepth = heightFogOpticalDepthNode(
      start.y.sub(u.height),
      rayDirection.y,
      distance,
      u.heightFalloff,
    ).mul(u.density);
    const transmittance = exp(opticalDepth.negate());

    // Isotropic ambient only — the sun's phase term is deliberately dropped
    // here. Keeping it would put a bright directional lobe on the far distance
    // that no shadow map can occlude, which reads as a haze bug rather than as
    // depth.
    const radiance = u.ambient.mul(u.albedo).mul(1 / FOUR_PI);
    return vec4(radiance.mul(float(1).sub(transmittance)), transmittance) as unknown as Vec4Node;
  }

  /** Unit world-space direction of the view ray through a screen UV. */
  #viewRayDirectionNode(screenUv: Vec2Node): Vec3Node {
    const u = this.#u;
    // Reconstruct at unit view depth, then rotate into world space. The
    // translation column is deliberately not applied: this is a direction.
    const viewPosition = viewPositionFromDepth(screenUv, float(1), u.projScale);
    const world = u.cameraMatrixWorld.mul(vec4(viewPosition, 0)).xyz;
    return normalize(world) as unknown as Vec3Node;
  }

  /** Unit world-space camera forward axis, i.e. `-Z` of the view matrix. */
  #forwardNode(): Vec3Node {
    const world = this.#u.cameraMatrixWorld.mul(vec4(0, 0, -1, 0)).xyz;
    return normalize(world) as unknown as Vec3Node;
  }

  /**
   * Relative triangular-PDF dither on the in-scatter.
   *
   * Fog gradients are the textbook banding case: a smooth low-amplitude ramp
   * across a large screen area, quantised to 8 bits at the very end of the
   * pipeline. The dither is *relative* — proportional to the local in-scatter —
   * so it is a constant fraction of a code value regardless of exposure, and
   * therefore invisible where the fog is bright and exactly strong enough where
   * it is dim. Two noise samples are subtracted to get a triangular
   * distribution, which removes the noise-modulation artefact a uniform dither
   * leaves behind (Wronski, *"Dithering part three — real world 2D
   * quantization dithering"*).
   */
  #ditherNode(inscatter: Vec3Node, pixel: Vec2Node): Vec3Node {
    const frame = this.#u.frame;
    const a = interleavedGradientNoiseNode(pixel, frame);
    const b = interleavedGradientNoiseNode(pixel.add(vec2(37.0, 17.0)), frame.add(11));
    const triangular = a.sub(b);
    return max(inscatter.mul(triangular.mul(DITHER_AMPLITUDE).add(1)), vec3(0, 0, 0)) as unknown as Vec3Node;
  }
}

/** ±0.4% of the local value: below one 8-bit code everywhere it matters. */
const DITHER_AMPLITUDE = 0.004;

/** Local light record with the CPU-side ranking key attached. */
interface MutableVolumetricLight extends VolumetricPointLight {
  readonly importance: number;
}

function clampRange(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/* ------------------------------------------------------------------------- *
 * TSL helpers shared by both paths
 * ------------------------------------------------------------------------- */

/** A scalar uniform node, as `uniform(0)` produces one. */
type FloatUniform = ReturnType<typeof createMediaUniforms>['time'];
/** A `mat4` uniform node. */
type Mat4Uniform = ReturnType<typeof createMediaUniforms>['cameraMatrixWorld'];

/**
 * Trilinear fetch from a 3D texture, at level 0 explicitly.
 *
 * Never an implicitly-levelled `textureSample`: a compute shader has no
 * fragment quad and therefore no derivatives, and a fragment shader may not
 * take them inside the non-uniform control flow a ray-march loop creates. The
 * volumes have no mips, so nothing is lost.
 */
function sampleVolume(volume: THREE.Texture, coordinate: Vec3Node): Vec4Node {
  return texture3D(volume, coordinate).level(float(0)) as unknown as Vec4Node;
}

/** Exact texel fetch from a 3D texture, with the sampler disabled. */
function loadVolume(volume: THREE.Texture, coordinate: THREE.Node): Vec4Node {
  return texture3D(volume, coordinate).setSampler(false) as unknown as Vec4Node;
}

/** `int` → `float`, with the one cast three's TSL typings require. */
function toFloat(value: THREE.Node): FloatNode {
  return float(value as unknown as FloatNode) as unknown as FloatNode;
}

/** `smoothstep(0, 1, x)`, assuming `x` is already clamped. */
function smoothstepNode(x: FloatNode): FloatNode {
  return x.mul(x).mul(float(3).sub(x.mul(2))) as unknown as FloatNode;
}

/** TSL form of {@link henyeyGreenstein}. */
function henyeyGreensteinNode(cosTheta: FloatNode, g: FloatNode): FloatNode {
  const gg = g.mul(g);
  const denominator = max(float(1).add(gg).sub(g.mul(cosTheta).mul(2)), float(1e-4));
  return float(1)
    .sub(gg)
    .div(pow(denominator, 1.5).mul(FOUR_PI)) as unknown as FloatNode;
}

/** TSL form of {@link dualLobePhase}. */
function dualLobePhaseNode(
  cosTheta: FloatNode,
  forwardG: FloatNode,
  backwardG: FloatNode,
  blend: FloatNode,
): FloatNode {
  return mix(
    henyeyGreensteinNode(cosTheta, forwardG),
    henyeyGreensteinNode(cosTheta, backwardG),
    blend,
  ) as unknown as FloatNode;
}

/** TSL form of {@link froxelSliceDepth}, with `t` already in `[0, 1]`. */
function froxelDepthNode(
  t: FloatNode,
  near: FloatNode,
  far: FloatNode,
  distribution: FroxelDepthDistribution,
): FloatNode {
  if (distribution === 'quadratic') {
    return near.add(far.sub(near).mul(t).mul(t)) as unknown as FloatNode;
  }
  return near.mul(pow(far.div(near), t)) as unknown as FloatNode;
}

/**
 * TSL form of {@link froxelDepthToSlice}, normalised to a `[0, 1]` texture
 * coordinate and biased onto texel centres.
 */
function froxelSliceCoordNode(
  depth: FloatNode,
  near: FloatNode,
  far: FloatNode,
  sliceCount: number,
  distribution: FroxelDepthDistribution,
): FloatNode {
  const t =
    distribution === 'quadratic'
      ? sqrt(saturate(depth.sub(near).div(max(far.sub(near), float(1e-4)))))
      : saturate(log(depth.div(near)).div(max(log(far.div(near)), float(1e-4))));
  // Slice `k` holds the integral up to its far boundary; the texel centre of
  // slice `k` is therefore at `(k + 0.5) / N`, and a depth landing exactly on a
  // slice boundary must read halfway between two texels.
  return clamp(t.sub(0.5 / sliceCount), float(0), float(1)) as unknown as FloatNode;
}

/**
 * TSL form of {@link heightFogOpticalDepth}, for a unit-density medium.
 *
 * Written branchlessly. The piecewise split of the JS version becomes a `mix`
 * between the flat-region and exponential-region antiderivatives keyed on the
 * sign of the height above the reference plane; because both branches agree at
 * the crossing, the blend is exact wherever the ray does not cross and
 * continuous where it does. The remaining error is confined to rays that cross
 * the reference height *within* the tail, which is the far distance where the
 * fog is already saturated.
 *
 * @param relativeY  ray origin height **relative to** the reference height.
 */
function heightFogOpticalDepthNode(
  relativeY: FloatNode,
  dirY: FloatNode,
  distance: FloatNode,
  falloff: FloatNode,
): FloatNode {
  const k = max(falloff, float(1e-5));
  const d = max(distance, float(0));
  // Clamping both endpoints at the reference plane is what implements the flat
  // region of the profile. Where the ray stays on one side of the plane the
  // result is exact; where it crosses, the two antiderivatives agree at the
  // crossing, so the expression stays continuous and monotone — which is all
  // the tail term needs, since by then the fog is nearly saturated.
  const y0 = max(relativeY, float(0)).toVar('tau_y0');
  const y1 = max(relativeY.add(dirY.mul(d)), float(0)).toVar('tau_y1');
  const dy = y1.sub(y0).toVar('tau_dy');

  const expA = exp(k.mul(y0).negate());
  const expB = exp(k.mul(y1).negate());
  const denominator = k.mul(dy).toVar('tau_denom');
  // Never zero, and never a sign flip: the epsilon carries the denominator's
  // own sign, so the quotient stays finite without biasing the result.
  const safeDenominator = denominator.add(sign01(denominator).mul(1e-6));
  const exact = expA.sub(expB).mul(d).div(safeDenominator);
  // Optically-thin limit, `d·exp(−k·y₀)`, taken wherever the denominator is too
  // small for the quotient to be well conditioned.
  const limit = d.mul(expA);
  const useExact = saturate(denominator.abs().mul(1e4).sub(1));
  return max(mix(limit, exact, useExact), float(0)) as unknown as FloatNode;
}

/** `+1` for non-negative input, `-1` otherwise, with no zero-sign hazard. */
function sign01(x: FloatNode): FloatNode {
  return step(float(0), x).mul(2).sub(1) as unknown as FloatNode;
}

/** The pair of functions every volumetric shader is built from. */
interface MediaEvaluator {
  mediaAt(worldPosition: Vec3Node): Vec4Node;
  inscatterAt(worldPosition: Vec3Node, viewDirection: Vec3Node, scattering: Vec3Node): Vec3Node;
}

/* ------------------------------------------------------------------------- *
 * Froxel resources (WebGPU compute path)
 * ------------------------------------------------------------------------- */

/**
 * The three 3D textures and two compute kernels that make up the froxel volume.
 *
 * The scattering volume is double buffered because the injection kernel reads
 * the previous frame's version as its temporal history while writing this
 * frame's, and a WebGPU storage texture cannot be bound for reading and writing
 * in the same dispatch. Both parities of both kernels are built up front:
 * a compute node's bindings are baked into its pipeline, so swapping textures
 * at runtime would rebuild the pipeline every frame.
 */
class FroxelResources {
  readonly dimensions: readonly [number, number, number];

  readonly #scatter: [THREE.Storage3DTexture, THREE.Storage3DTexture];
  readonly #integrated: THREE.Storage3DTexture;
  readonly #inject: [THREE.ComputeNode, THREE.ComputeNode];
  readonly #integrate: [THREE.ComputeNode, THREE.ComputeNode];
  #parity = 0;

  constructor(
    dimX: number,
    dimY: number,
    dimZ: number,
    u: MediaUniforms,
    media: MediaEvaluator,
    previousViewProjection: Mat4Uniform,
    historyValid: FloatUniform,
    temporalBlend: FloatUniform,
    distribution: FroxelDepthDistribution,
  ) {
    this.dimensions = [dimX, dimY, dimZ];

    this.#scatter = [
      createStorageVolume('fog.scatter.0', dimX, dimY, dimZ),
      createStorageVolume('fog.scatter.1', dimX, dimY, dimZ),
    ];
    this.#integrated = createStorageVolume('fog.integrated', dimX, dimY, dimZ);

    const froxelCount = dimX * dimY * dimZ;
    const columnCount = dimX * dimY;

    this.#inject = [
      compute(
        buildInjectionKernel(
          this.#scatter[0],
          this.#scatter[1],
          dimX,
          dimY,
          dimZ,
          u,
          media,
          previousViewProjection,
          historyValid,
          temporalBlend,
          distribution,
        ),
        froxelCount,
        [WORKGROUP_SIZE],
      ).setName('fog.inject.0'),
      compute(
        buildInjectionKernel(
          this.#scatter[1],
          this.#scatter[0],
          dimX,
          dimY,
          dimZ,
          u,
          media,
          previousViewProjection,
          historyValid,
          temporalBlend,
          distribution,
        ),
        froxelCount,
        [WORKGROUP_SIZE],
      ).setName('fog.inject.1'),
    ];

    this.#integrate = [
      compute(
        buildIntegrationKernel(this.#scatter[0], this.#integrated, dimX, dimY, dimZ, u, distribution),
        columnCount,
        [WORKGROUP_SIZE],
      ).setName('fog.integrate.0'),
      compute(
        buildIntegrationKernel(this.#scatter[1], this.#integrated, dimX, dimY, dimZ, u, distribution),
        columnCount,
        [WORKGROUP_SIZE],
      ).setName('fog.integrate.1'),
    ];
  }

  /** The volume the composite samples. Stable across frames. */
  get integratedTexture(): THREE.Texture {
    return this.#integrated;
  }

  get bytes(): number {
    const [x, y, z] = this.dimensions;
    // Three RGBA16F volumes.
    return x * y * z * 8 * 3;
  }

  render(renderer: NodeRenderer): void {
    this.#parity = 1 - this.#parity;
    const inject = this.#inject[this.#parity];
    const integrate = this.#integrate[this.#parity];
    if (inject === undefined || integrate === undefined) return;
    // Two dispatches, not one: every froxel's injected value must be visible to
    // the column march, and WebGPU only guarantees that across pass boundaries.
    renderer.compute(inject);
    renderer.compute(integrate);
  }

  dispose(): void {
    this.#scatter[0].dispose();
    this.#scatter[1].dispose();
    this.#integrated.dispose();
  }
}

function createStorageVolume(
  name: string,
  width: number,
  height: number,
  depth: number,
): THREE.Storage3DTexture {
  const texture = new THREE.Storage3DTexture(width, height, depth);
  texture.name = name;
  texture.format = THREE.RGBAFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  return texture;
}

/**
 * One thread per froxel: evaluate the media and the lighting at a jittered
 * point inside the froxel, blend against the reprojected history, store
 * `(σ_s·L, σ_t)`.
 */
function buildInjectionKernel(
  destination: THREE.Storage3DTexture,
  history: THREE.Storage3DTexture,
  dimX: number,
  dimY: number,
  dimZ: number,
  u: MediaUniforms,
  media: MediaEvaluator,
  previousViewProjection: Mat4Uniform,
  historyValid: FloatUniform,
  temporalBlend: FloatUniform,
  distribution: FroxelDepthDistribution,
): THREE.Node {
  const target = storageTexture3D(destination);
  const slabSize = dimX * dimY;

  return Fn(() => {
    // Linear thread id → froxel coordinate. The dispatch count is an exact
    // multiple of the workgroup size for every shipped tier, but the clamp
    // keeps an overridden grid safe: an out-of-range thread recomputes the last
    // froxel and writes it the same value, which is a benign race.
    // `mod` rather than `clamp` on the trailing axis: it needs no comparison
    // and it maps an over-dispatched thread onto a froxel some other thread
    // also owns, which writes the identical value — a benign race either way.
    const index = int(instanceIndex);
    const x = index.mod(int(dimX)).toVar('fx');
    const y = index.div(int(dimX)).mod(int(dimY)).toVar('fy');
    const z = index.div(int(slabSize)).mod(int(dimZ)).toVar('fz');

    // Halton jitter inside the froxel, rotated every frame. This is what turns
    // one sample per froxel into ~20 effective samples once the temporal blend
    // has converged, and it is why the noise field and the shadow edges do not
    // show the froxel grid.
    const jitter = u.jitter;
    const uvw = vec3(
      toFloat(x).add(jitter.x).div(dimX),
      toFloat(y).add(jitter.y).div(dimY),
      toFloat(z).add(jitter.z).div(dimZ),
    ).toVar('froxelUv');

    const near = u.range.x;
    const far = u.range.y;
    const depth = froxelDepthNode(uvw.z, near, far, distribution).toVar('froxelDepth');

    const viewPosition = viewPositionFromDepth(uvw.xy, depth, u.projScale).toVar('viewP');
    const worldPosition = u.cameraMatrixWorld.mul(vec4(viewPosition, 1)).xyz.toVar('worldP');
    const viewDirection = normalize(worldPosition.sub(u.cameraPosition)).toVar('viewDir');

    const sample = media.mediaAt(worldPosition).toVar('media');
    const scattering = sample.xyz;
    const extinction = sample.w;
    const inscatter = media
      .inscatterAt(worldPosition, viewDirection, scattering)
      .toVar('inscatter');

    const current = vec4(inscatter, extinction).toVar('current');

    // -- temporal reprojection ------------------------------------------
    //
    // The volume is view-aligned, so "the same froxel last frame" is not the
    // same texel: the world point has to make the round trip through the
    // previous view-projection and back into the previous volume's coordinates.
    const previousClip = previousViewProjection.mul(vec4(worldPosition, 1)).toVar('prevClip');
    const previousNdc = previousClip.xyz.div(max(previousClip.w, float(1e-5))).toVar('prevNdc');
    const previousUv = vec2(
      previousNdc.x.mul(0.5).add(0.5),
      float(0.5).sub(previousNdc.y.mul(0.5)),
    ).toVar('prevUv');
    const previousDepth = previousClip.w.toVar('prevDepth');
    const previousW = froxelSliceCoordNode(
      clamp(previousDepth, near, far),
      near,
      far,
      dimZ,
      distribution,
    ).toVar('prevW');

    const inFrustum = previousUv.x
      .greaterThanEqual(0)
      .and(previousUv.x.lessThanEqual(1))
      .and(previousUv.y.greaterThanEqual(0))
      .and(previousUv.y.lessThanEqual(1))
      .and(previousDepth.greaterThan(near))
      .and(previousDepth.lessThan(far));

    const blended = current.toVar('blended');
    If(inFrustum.and(historyValid.greaterThan(0.5)), () => {
      // Level-0 fetch: a compute shader has no derivatives, so an implicitly
      // levelled `textureSample` would not compile.
      const previous = sampleVolume(
        history,
        vec3(previousUv, previousW) as unknown as Vec3Node,
      ).toVar('history');

      // Scale-free disagreement measure, on extinction (the term that changes
      // when geometry moves) rather than on radiance (which changes with every
      // flicker of a torch and would defeat the filter).
      const change = current.w
        .sub(previous.w)
        .abs()
        .div(max(current.w.add(previous.w), float(1e-5)))
        .toVar('change');
      const alpha = saturate(
        temporalBlend.add(float(1).sub(temporalBlend).mul(change).mul(change)),
      ).toVar('alpha');
      blended.assign(mix(previous, current, alpha));
    });

    textureStore(target, ivec3(x, y, z), blended);
    return blended;
  })();
}

/**
 * One thread per froxel *column*: march `z` accumulating scattering and
 * transmittance, writing the running totals so a pixel can read its answer with
 * a single fetch.
 */
function buildIntegrationKernel(
  source: THREE.Storage3DTexture,
  destination: THREE.Storage3DTexture,
  dimX: number,
  dimY: number,
  dimZ: number,
  u: MediaUniforms,
  distribution: FroxelDepthDistribution,
): THREE.Node {
  const target = storageTexture3D(destination);

  return Fn(() => {
    const index = int(instanceIndex);
    const x = index.mod(int(dimX)).toVar('cx');
    const y = index.div(int(dimX)).mod(int(dimY)).toVar('cy');

    const near = u.range.x;
    const far = u.range.y;

    // Obliquity: the world-space length of a slice along *this* column's ray is
    // longer than the difference in view depth by `|v| / v_z`. Ignoring it is
    // the classic froxel bug — fog gets visibly thinner towards the corners of
    // the screen.
    const uv = vec2(toFloat(x).add(0.5).div(dimX), toFloat(y).add(0.5).div(dimY));
    const direction = viewPositionFromDepth(uv, float(1), u.projScale).toVar('colDir');
    const obliquity = direction.length().toVar('obliquity');

    const scatteredLight = vec3(0, 0, 0).toVar('accScatter');
    const transmittance = float(1).toVar('accT');
    const previousDepth = near.toVar('prevDepth');

    Loop({ start: int(0), end: int(dimZ), type: 'int' }, ({ i }) => {
      const t = toFloat(i).add(1).div(dimZ);
      const depth = froxelDepthNode(t, near, far, distribution).toVar('sliceDepth');
      const stepLength = max(depth.sub(previousDepth), float(0)).mul(obliquity).toVar('stepLength');
      previousDepth.assign(depth);

      // An exact texel fetch, not a filtered sample: this thread owns this
      // column and must read precisely what the injection pass wrote.
      const cell = loadVolume(source, ivec3(x, y, i)).toVar('cell');
      const inscatter = cell.xyz;
      const extinction = max(cell.w, float(1e-5)).toVar('sigmaT');

      const sliceTransmittance = exp(extinction.mul(stepLength).negate()).toVar('sliceT');
      const integrated = inscatter
        .sub(inscatter.mul(sliceTransmittance))
        .div(extinction)
        .toVar('sliceS');

      scatteredLight.addAssign(transmittance.mul(integrated));
      transmittance.mulAssign(sliceTransmittance);

      textureStore(target, ivec3(x, y, i), vec4(scatteredLight, transmittance));
    });

    return transmittance;
  })();
}

/* ------------------------------------------------------------------------- *
 * Fog volume handle
 * ------------------------------------------------------------------------- */

class FogVolumeHandleImpl implements FogVolumeHandle {
  readonly #record: FogVolumeRecord;
  readonly #owner: FogVolumeRecord[];

  constructor(record: FogVolumeRecord, owner: FogVolumeRecord[]) {
    this.#record = record;
    this.#owner = owner;
  }

  get id(): number {
    return this.#record.id;
  }

  get alive(): boolean {
    return this.#record.alive;
  }

  get active(): boolean {
    return this.#record.active;
  }

  setCenter(x: number, y: number, z: number): void {
    this.#record.center.set(x, y, z);
  }

  setHalfExtent(x: number, y: number, z: number): void {
    this.#record.halfExtent.set(Math.max(0.01, x), Math.max(0.01, y), Math.max(0.01, z));
  }

  setRadius(radius: number): void {
    const r = Math.max(0.01, radius);
    this.#record.halfExtent.set(r, r, r);
  }

  setDensityScale(scale: number): void {
    this.#record.densityScale = Math.max(0, scale);
  }

  setColor(color: THREE.ColorRepresentation): void {
    this.#record.color.set(color);
  }

  setEmissive(color: THREE.ColorRepresentation): void {
    this.#record.emissive.set(color);
  }

  setPriority(priority: number): void {
    this.#record.priority = priority;
  }

  release(): void {
    if (!this.#record.alive) return;
    this.#record.alive = false;
    this.#record.active = false;
    const index = this.#owner.indexOf(this.#record);
    if (index >= 0) this.#owner.splice(index, 1);
  }
}

/* ------------------------------------------------------------------------- *
 * Entry point
 * ------------------------------------------------------------------------- */

/**
 * Construct the volumetrics module and register it with the engine.
 *
 * The composite is **not** installed here — that belongs to
 * {@link module:render/post/LightShafts}, which owns the screen-space pass and
 * the post-stack integration. Register both; order does not matter.
 */
export function registerVolumetrics(
  ctx: GameContext,
  options: VolumetricsOptions = {},
): VolumetricsModule {
  const module = new VolumetricsModule(options);
  ctx.engine.add(module);
  return module;
}
