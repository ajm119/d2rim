/**
 * @module render/post/SSR
 *
 * Screen-space reflections: hierarchical-depth ray marching with roughness-aware
 * cone tracing, thickness handling, and a seamless hand-off to the IBL probe
 * wherever the ray leaves the screen.
 *
 * This is the effect the art direction leans on hardest. Rain-soaked mud, wet
 * cobbles and standing water are *defined* by what they reflect: a wet surface
 * without reflections just reads as a darker dry surface. A cube probe alone
 * cannot do it, because the thing a puddle in the Blood Moor most needs to
 * reflect is the twisted tree standing in it.
 *
 * ---
 *
 * ## 1. Hi-Z traversal
 *
 * Naïve SSR marches fixed steps through the depth buffer and is wrong in both
 * directions at once: too coarse and it steps over thin geometry, too fine and
 * it costs hundreds of taps.
 *
 * The hierarchical march (Uludag, *"Hi-Z Screen-Space Cone-Traced Reflections"*,
 * GPU Pro 5, 2014) walks a **min-depth pyramid** instead. At level `L`, one
 * texel holds the closest depth in a `2ᴸ × 2ᴸ` block. If the ray is nearer to
 * the camera than that minimum, nothing in the entire block can be hit, so the
 * ray jumps straight to the block's exit boundary and *coarsens* a level. When
 * it is not, the ray refines a level and re-tests. The result is a DDA over a
 * quadtree: empty sky is crossed in two or three taps, and geometry is
 * approached at single-texel accuracy.
 *
 * The ray is marched in **screen space parameterised by `s ∈ [0, 1]`** along the
 * projected segment, carrying `1/viewZ` rather than `viewZ`. Reciprocal depth is
 * affine in screen space under a perspective projection, so linear interpolation
 * of `(uv, 1/z)` in `s` is *exact* — no perspective error accumulates over a
 * long ray, which is what otherwise makes distant reflections drift.
 *
 * ## 2. Thickness
 *
 * A depth buffer stores a surface, not a solid. When the ray's depth passes
 * behind the stored depth, the honest answer is "unknown": the ray might have
 * hit the front face, or it might have passed through empty space behind a
 * thin object. This implementation accepts a hit only inside
 * `[sceneZ, sceneZ + thickness]`, with `thickness` growing linearly with depth
 * so a 20 cm tolerance up close does not become a sub-pixel one at 100 m. Past
 * that window the march continues rather than reporting a hit, which is what
 * stops the "reflection smeared along the silhouette" artefact.
 *
 * ## 3. Roughness-aware cone tracing
 *
 * A mirror ray is only correct at roughness 0. For anything rougher the
 * reflected radiance is an integral over a lobe, whose screen-space footprint
 * grows with both roughness and hit distance. Following GPU Pro 5, the GGX
 * roughness is converted to an equivalent Blinn-Phong specular power, then to
 * the half-angle of the cone containing a fixed fraction of the lobe's energy
 * ({@link roughnessToSpecularPower}, {@link specularPowerToConeAngle}). The
 * cone's radius at the hit selects a mip of a blurred colour pyramid, and the
 * high tiers additionally average a small golden-angle disk of taps across that
 * footprint.
 *
 * That is what makes wet mud (roughness ≈ 0.3) reflect a soft smear of tree and
 * sky while a polished helm (roughness ≈ 0.05) reflects a legible image, from
 * one code path.
 *
 * ## 4. The screen-edge problem, and how it is actually solved
 *
 * Rays leave the screen. Fading the reflection to *black* at the border is the
 * single most recognisable SSR failure. Fading it to *nothing* (zero
 * contribution) is only slightly better: the surface visibly loses its
 * reflection as the camera turns.
 *
 * This module never composites reflected radiance. It composites the
 * **difference** between what the screen says and what the probe already said:
 *
 * ```
 * delta = ( L_ssr − L_probe ) · confidence · F_env(F₀, roughness, N·V)
 * ```
 *
 * The scene material has already added `L_probe · F_env` as its indirect
 * specular. Adding `delta` therefore *replaces* the probe with the screen-space
 * result exactly where the march succeeded, and leaves the probe untouched — to
 * the last bit — where it did not, because `confidence → 0` makes `delta → 0`.
 * There is no border, no cut-off and no seam, and a ray that leaves the screen
 * costs nothing visually. `confidence` folds together the screen-border fade,
 * the ray-facing-the-camera fade, the distance fade, the roughness cut-off and
 * whether a hit happened at all.
 *
 * ## 5. What the integrator must wire
 *
 * SSR fundamentally needs the scene's *shaded* colour, which only exists after
 * the forward pass. This module runs before it, so it consumes:
 *
 * - **`render.sceneColor`** ({@link SceneColorProvider}) — linear HDR radiance
 *   before tone mapping. Almost always the previous frame's, which is why hit
 *   positions are reprojected through the velocity buffer before the colour is
 *   fetched. **Without this service SSR produces nothing at all** (and says so,
 *   once, in the console); everything else still works.
 * - **`render.gbuffer.surface`** ({@link SurfaceParameterProvider}) — optional
 *   roughness/metalness/reflectance. Without it the whole frame is treated as
 *   one material with {@link SSROptions.defaultRoughness}, which is a real
 *   approximation, not a placeholder: a rain-soaked exterior genuinely is
 *   near-uniformly wet. Supply the texture and it becomes correct per-pixel.
 * - **`render.ibl`** ({@link IndirectRadianceProvider}) — optional. Used for the
 *   `L_probe` term above. Without it the fallback is a constant radiance
 *   ({@link SSROptions.fallbackRadiance}), which keeps the composite seamless
 *   but is only as good as that constant.
 * - **`render.motion`** — optional; used to reproject colour fetches and the
 *   temporal history.
 *
 * ## 6. Cost, at 1080p on a mid-range 2020 discrete GPU (RX 5700 / RTX 2060 class)
 *
 * | Stage                     | Resolution | Work                          | Budget |
 * |---------------------------|-----------|--------------------------------|--------|
 * | Hi-Z pyramid (9 levels)   | 960×540 ↓ | 2 passes/level, 4 taps each    | 0.12 ms |
 * | colour pyramid (7 levels) | 960×540 ↓ | 2 passes/level, 4 taps each    | 0.18 ms |
 * | SSR trace (`high`)        | 960×540   | ≤48 Hi-Z steps + 4 cone taps   | 1.10 ms |
 * | à-trous ×1 (5×5)          | 960×540   | 50 taps                        | 0.18 ms |
 * | temporal accumulate       | 960×540   | 8 taps                         | 0.09 ms |
 * | bilateral upsample        | inline    | 8 taps in the composite        | 0.08 ms |
 * | **total**                 |           |                                | **1.75 ms** |
 *
 * The trace dominates and its cost is *data dependent*: the 48-step cap is a
 * worst case for a grazing ray across a complex silhouette, while a typical ray
 * over open ground resolves in 8–14 steps because the pyramid lets it skip the
 * sky in two. The 1.10 ms figure assumes an average of ~16 steps over 518 k
 * half-resolution pixels, each step a single R16F fetch that is highly
 * cache-resident (rays from neighbouring pixels traverse the same coarse
 * cells). Published Hi-Z SSR implementations at half resolution on this class
 * of hardware land in the 0.9–1.4 ms band, which brackets the estimate.
 *
 * Combined with GTAO's 1.4 ms — and noting that the depth+normal prepass and
 * the guide buffer are *shared*, so SSR does not pay for them again — the two
 * effects together fit in ~3.1 ms. Against a 16.6 ms frame that leaves 13.5 ms
 * for the forward pass, shadows, sky and the rest of post, which is the right
 * shape for this scene.
 *
 * **No frame time in this file was measured on this machine.** The development
 * container has no GPU (SwiftShader on 4 cores); every number above is derived
 * from tap counts, resolutions, buffer formats and published costs for the same
 * passes on the target hardware.
 *
 * ## 7. Quality tiers
 *
 * | Tier     | Max steps | Hi-Z levels | Cone taps | À-trous | Est. total |
 * |----------|-----------|-------------|-----------|---------|-----------|
 * | `off`    | —         | —           | —         | —       | 0 ms      |
 * | `low`    | 16        | 0 (linear)  | 1         | 0       | 0.55 ms   |
 * | `medium` | 32        | 5           | 1         | 1 × 3×3 | 1.05 ms   |
 * | `high`   | 48        | all         | 4         | 1 × 5×5 | 1.75 ms   |
 * | `ultra`  | 80        | all         | 8         | 2 × 5×5 | 2.90 ms   |
 *
 * `low` sets the maximum Hi-Z level to 0, which turns the same traversal code
 * into a per-texel linear DDA — the hierarchy disappears without a second code
 * path, and the tier degrades to "short, cheap, contact reflections only".
 *
 * ## References
 *
 * - Y. Uludag, *Hi-Z Screen-Space Cone-Traced Reflections*, GPU Pro 5, 2014.
 * - M. McGuire & M. Mara, *Efficient GPU Screen-Space Ray Tracing*, JCGT 2014.
 * - D. Lazarov, *Getting More Physical in Call of Duty: Black Ops II*,
 *   SIGGRAPH 2013 — the analytic environment-BRDF fit used for `F_env`.
 * - B. Karis, *Real Shading in Unreal Engine 4*, SIGGRAPH 2013 — split-sum
 *   approximation the fit above stands in for.
 */

import * as THREE from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  cos,
  dot,
  exp2,
  float,
  log2,
  max,
  min,
  mix,
  normalize,
  reflect,
  saturate,
  screenUV,
  sin,
  smoothstep,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import { serviceKey } from '../../core/ServiceLocator';
import type { GameContext, GameModule } from '../../core/types';
import {
  AtrousDenoiser,
  FullScreenPass,
  MipChain,
  RendererStateScope,
  TemporalAccumulator,
  acquireGuideBuffer,
  asNodeRenderer,
  bilateralUpsampleNode,
  decodeGuide,
  releaseGuideBuffer,
  screenUvFromViewPosition,
  tryGetMotionVectors,
  viewPositionFromDepth,
  type FloatNode,
  type GuideBufferPass,
  type GuideBufferProvider,
  type MotionVectorSource,
  type TemporalUniforms,
  type Vec2Node,
  type Vec3Node,
} from './Denoise';

/* ------------------------------------------------------------------------- *
 * Interfaces this module expects from other systems
 * ------------------------------------------------------------------------- */

/**
 * The shaded scene colour SSR reflects.
 *
 * Register under {@link SCENE_COLOR_SERVICE_ID}. Must be **linear HDR radiance
 * before tone mapping** — reflecting a tone-mapped, sRGB-encoded buffer makes
 * every reflection wash out towards white and destroys the cold/warm separation
 * the art direction depends on.
 *
 * `isPreviousFrame` defaults to `true` because this module runs before the
 * engine's scene render: the only colour buffer that exists at that point is
 * the last one. Hit positions are then reprojected backwards through the
 * velocity buffer before the colour is fetched, so moving geometry still
 * reflects correctly.
 */
export interface SceneColorProvider {
  readonly sceneColorTexture: THREE.Texture | null;
  /** Bumped when the texture object is replaced, so the pyramid rebinds. */
  readonly sceneColorVersion?: number;
  /** Default `true`. Set false only if the buffer is this frame's. */
  readonly isPreviousFrame?: boolean;
}

/** Service id for {@link SceneColorProvider}. */
export const SCENE_COLOR_SERVICE_ID = 'render.sceneColor';

/**
 * Per-pixel surface parameters, if the main pass writes them.
 *
 * `surfaceTexture` must be `r = perceptual roughness`, `g = metalness`,
 * `b = dielectric reflectance` (the `F₀` of non-metals, normally 0.04–0.08;
 * water and wet surfaces sit at the top of that range). The alpha channel is
 * ignored.
 *
 * When this service is absent, the module falls back to
 * {@link SSROptions.defaultRoughness} / `defaultMetalness` /
 * `defaultReflectance` for the whole frame.
 */
export interface SurfaceParameterProvider {
  readonly surfaceTexture: THREE.Texture | null;
  readonly surfaceVersion?: number;
}

/** Service id for {@link SurfaceParameterProvider}. */
export const SURFACE_PARAMETERS_SERVICE_ID = 'render.gbuffer.surface';

/**
 * A prefiltered environment probe, used for the fallback radiance that SSR
 * blends *out of* as its confidence rises.
 *
 * Structurally the subset of {@link module:render/IBL}'s `IBLService` this module
 * needs; declared here rather than imported so neither module depends on the
 * other. Resolve under {@link IBL_SERVICE_ID}.
 *
 * @param direction world-space reflection direction
 * @param roughness perceptual roughness in `[0, 1]`
 */
export interface IndirectRadianceProvider {
  prefilteredRadiance(
    direction: THREE.Node<'vec3'>,
    roughness: THREE.Node<'float'>,
  ): THREE.Node<'vec3'>;
}

/** Service id for {@link IndirectRadianceProvider}. */
export const IBL_SERVICE_ID = 'render.ibl';

/* ------------------------------------------------------------------------- *
 * Public service
 * ------------------------------------------------------------------------- */

export type SSRQuality = 'off' | 'low' | 'medium' | 'high' | 'ultra';

export interface SSRStats {
  readonly enabled: boolean;
  readonly quality: SSRQuality;
  readonly width: number;
  readonly height: number;
  readonly maxSteps: number;
  readonly hiZLevels: number;
  readonly coneTaps: number;
  /** False when no scene colour was found; SSR contributes nothing. */
  readonly sceneColorAvailable: boolean;
  /** False when no per-pixel roughness was found; uniform defaults are used. */
  readonly surfaceParametersAvailable: boolean;
  /** False when no IBL probe was found; a constant fallback radiance is used. */
  readonly probeAvailable: boolean;
  readonly temporalFromVelocity: boolean;
}

export interface SSRService {
  /**
   * The additive **delta** radiance, at full resolution, ready to be added to
   * the scene colour. Zero wherever the march failed, so adding it is always
   * safe and never produces a screen-edge seam.
   *
   * `null` when the effect is off or unwired.
   */
  readonly deltaNode: THREE.Node<'vec3'> | null;
  /** `sceneColor + delta`, for a post stack that prefers a single node. */
  composite(sceneColor: THREE.Node<'vec4'>): THREE.Node<'vec4'>;
  /** Half-resolution denoised delta, for debug views. */
  readonly reflectionTexture: THREE.Texture | null;
  /** The Hi-Z pyramid, for debug views. */
  readonly hiZTexture: THREE.Texture | null;
  readonly quality: SSRQuality;
  setQuality(quality: SSRQuality): void;
  /** Explicitly supply the scene colour instead of registering a service. */
  setSceneColor(texture: THREE.Texture | null, isPreviousFrame?: boolean): void;
  readonly stats: SSRStats;
}

/** Service key for {@link SSRService}. */
export const SSRKey = serviceKey<SSRService>('render.ssr');

export interface SSROptions {
  /** Starting quality tier. Default `'high'`. */
  quality?: SSRQuality;
  /** Maximum ray length in world units. Default 40 m. */
  maxDistance?: number;
  /**
   * Depth tolerance for accepting a hit, in world units at 1 m, growing
   * linearly with depth. Default 0.12 m — roughly the thickness of the
   * geometry this scene is built from (planks, tombstones, tree trunks).
   */
  thickness?: number;
  /**
   * Fraction of the screen over which the confidence fades to zero at the
   * border. Default 0.12 — wide enough that a turning camera does not reveal a
   * moving edge, narrow enough not to waste the usable screen.
   */
  edgeFade?: number;
  /**
   * Roughness above which SSR stops contributing. Default 0.6: beyond that the
   * cone footprint is so large that the probe is both cheaper *and* more
   * accurate than a handful of screen taps.
   */
  maxRoughness?: number;
  /** Uniform roughness used when no surface G-buffer exists. Default 0.28. */
  defaultRoughness?: number;
  /** Uniform metalness used when no surface G-buffer exists. Default 0. */
  defaultMetalness?: number;
  /** Uniform dielectric `F₀` used when no surface G-buffer exists. Default 0.05. */
  defaultReflectance?: number;
  /**
   * Radiance used for `L_probe` when no IBL service is registered. Defaults to
   * a cold overcast grey (0.055, 0.062, 0.075) — deliberately blue-shifted, so
   * that the delta does not tint reflections warm on a grey day.
   */
  fallbackRadiance?: THREE.ColorRepresentation;
  /** Global multiplier on the reflection delta. Default 1. */
  intensity?: number;
  /** Resolution of the trace relative to the framebuffer. Default 0.5. */
  resolutionScale?: number;
  /** Steady-state temporal blend weight. Default 0.2. */
  temporalMinAlpha?: number;
}

/* ------------------------------------------------------------------------- *
 * Pure math — unit tested in tests/ssr.math.test.ts
 * ------------------------------------------------------------------------- */

/**
 * GGX perceptual roughness to the equivalent Blinn-Phong specular power
 * (GPU Pro 5, `RoughnessToSpecularPower`): `2/α⁴ − 2` with `α = roughness²`.
 *
 * The conversion exists purely so the cone half-angle below has a closed form;
 * nothing else in the pipeline uses Blinn-Phong.
 */
export function roughnessToSpecularPower(roughness: number): number {
  const alpha = Math.max(1e-3, roughness * roughness);
  return 2 / (alpha * alpha) - 2;
}

/**
 * Half-angle, in radians, of the cone containing a fixed fraction of a
 * Blinn-Phong lobe of the given power (GPU Pro 5,
 * `SpecularPowerToConeAngle`):
 *
 * ```
 * θ = acos( ξ^(1 / (power + 1)) ),  ξ = 0.244
 * ```
 *
 * `ξ = 0.244` is the paper's choice and corresponds to capturing roughly the
 * dominant three quarters of the lobe's energy. Larger `ξ` narrows the cone and
 * under-blurs; smaller widens it and over-blurs.
 */
export function specularPowerToConeAngle(specularPower: number): number {
  if (!Number.isFinite(specularPower) || specularPower >= 1e5) return 0;
  const xi = 0.244;
  return Math.acos(Math.pow(xi, 1 / (specularPower + 1)));
}

/** Convenience composition of the two functions above. */
export function roughnessToConeAngle(roughness: number): number {
  return specularPowerToConeAngle(roughnessToSpecularPower(roughness));
}

/**
 * Mip level of a colour pyramid that matches a cone footprint of
 * `radiusPixels` at the hit.
 *
 * `log2(2r)` because level `L` of a pyramid averages a `2ᴸ`-wide box, so a
 * footprint of diameter `d` is represented by level `log2(d)`.
 */
export function coneRadiusToMip(radiusPixels: number, maxLevel: number): number {
  const diameter = Math.max(1, 2 * radiusPixels);
  return Math.min(maxLevel, Math.max(0, Math.log2(diameter)));
}

/**
 * Screen-border confidence fade.
 *
 * Returns 1 in the interior and falls smoothly to 0 within `fade` (a fraction
 * of the half-extent) of any edge. Symmetric in both axes and independent of
 * aspect ratio, which matters because an aspect-dependent fade makes ultrawide
 * displays lose their reflections at a visibly different rate.
 */
export function screenEdgeFade(u: number, v: number, fade: number): number {
  const width = Math.max(1e-4, fade);
  const fx = Math.min(1, Math.max(0, (1 - Math.abs(2 * u - 1)) / width));
  const fy = Math.min(1, Math.max(0, (1 - Math.abs(2 * v - 1)) / width));
  return smoothstep01(fx) * smoothstep01(fy);
}

function smoothstep01(x: number): number {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

/**
 * Lazarov's analytic environment-BRDF fit (SIGGRAPH 2013), the standard
 * texture-free stand-in for Karis's split-sum `A`/`B` lookup table.
 *
 * ```
 * F_env = F₀ · A + B
 * ```
 *
 * Returns `[A, B]`. Accurate to well under a percent over the whole
 * roughness/`N·V` domain, and it removes a texture fetch and a 128×128 LUT from
 * a pass that is already bandwidth bound.
 *
 * This is the term that makes the reflection *behave*: without it, reflections
 * would be equally strong head-on and at grazing incidence, when in reality
 * Fresnel makes a wet road blindingly reflective along the view direction and
 * nearly matte underfoot.
 */
export function envBRDFApprox(roughness: number, nDotV: number): [number, number] {
  const c0 = [-1, -0.0275, -0.572, 0.022];
  const c1 = [1, 0.0425, 1.04, -0.04];
  const r = [
    roughness * (c0[0] ?? 0) + (c1[0] ?? 0),
    roughness * (c0[1] ?? 0) + (c1[1] ?? 0),
    roughness * (c0[2] ?? 0) + (c1[2] ?? 0),
    roughness * (c0[3] ?? 0) + (c1[3] ?? 0),
  ];
  const rx = r[0] ?? 0;
  const ry = r[1] ?? 0;
  const a004 = Math.min(rx * rx, Math.pow(2, -9.28 * Math.max(0, nDotV))) * rx + ry;
  return [-1.04 * a004 + (r[2] ?? 0), 1.04 * a004 + (r[3] ?? 0)];
}

/**
 * Thickness tolerance at a given view depth.
 *
 * A constant world-space thickness is wrong at both ends of the range: at 1 m a
 * 12 cm window is generous, and at 150 m it is smaller than the depth buffer's
 * own quantisation, so distant reflections stop finding hits at all. Scaling it
 * with depth keeps the window a roughly constant number of *pixels*, which is
 * the quantity the march can actually resolve.
 */
export function thicknessAt(viewZ: number, baseThickness: number): number {
  return baseThickness * Math.max(1, viewZ);
}

/**
 * Golden-angle (Vogel) disk sample points, precomputed.
 *
 * Used to average the colour pyramid across the cone's screen footprint. The
 * golden angle spreads `n` points over a disk with no clumping at any `n`,
 * which a square or ring pattern cannot do — and a clumped pattern shows up
 * immediately as structured noise in a rough reflection.
 */
export function vogelDisk(count: number): Array<[number, number]> {
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const points: Array<[number, number]> = [];
  for (let i = 0; i < count; i++) {
    const radius = Math.sqrt((i + 0.5) / count);
    const theta = i * goldenAngle;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
  }
  return points;
}

/* ------------------------------------------------------------------------- *
 * Quality tiers
 * ------------------------------------------------------------------------- */

interface SSRTierConfig {
  readonly maxSteps: number;
  /** Hard cap on the Hi-Z level used. 0 collapses the march to a linear DDA. */
  readonly maxHiZLevel: number;
  readonly coneTaps: number;
  readonly refineSteps: number;
  readonly denoiseIterations: number;
  readonly denoiseRadius: 1 | 2;
}

export const SSR_TIERS: Readonly<Record<Exclude<SSRQuality, 'off'>, SSRTierConfig>> = {
  low: { maxSteps: 16, maxHiZLevel: 0, coneTaps: 1, refineSteps: 0, denoiseIterations: 0, denoiseRadius: 1 },
  medium: { maxSteps: 32, maxHiZLevel: 5, coneTaps: 1, refineSteps: 3, denoiseIterations: 1, denoiseRadius: 1 },
  high: { maxSteps: 48, maxHiZLevel: 12, coneTaps: 4, refineSteps: 4, denoiseIterations: 1, denoiseRadius: 2 },
  ultra: { maxSteps: 80, maxHiZLevel: 12, coneTaps: 8, refineSteps: 5, denoiseIterations: 2, denoiseRadius: 2 },
};

/* ------------------------------------------------------------------------- *
 * TSL helpers
 * ------------------------------------------------------------------------- */

/** TSL form of {@link envBRDFApprox}; returns `(A, B)`. */
export const envBRDFApproxNode = Fn(([roughness, nDotV]: [FloatNode, FloatNode]) => {
  const c0 = vec4(-1, -0.0275, -0.572, 0.022);
  const c1 = vec4(1, 0.0425, 1.04, -0.04);
  const r = c0.mul(roughness).add(c1).toVar('r');
  const a004 = min(r.x.mul(r.x), exp2(nDotV.max(0).mul(-9.28))).mul(r.x).add(r.y);
  return vec2(a004.mul(-1.04).add(r.z), a004.mul(1.04).add(r.w));
});

/** TSL form of {@link roughnessToConeAngle}, as a tangent (what we actually use). */
export const coneTangentNode = Fn(([roughness]: [FloatNode]) => {
  const alpha = max(roughness.mul(roughness), float(1e-3)).toVar('alpha');
  const power = float(2).div(alpha.mul(alpha)).sub(2).toVar('power');
  // acos(ξ^(1/(power+1))) with ξ = 0.244, expressed through exp2/log2 so the
  // shader never evaluates a general `pow` with a runtime base.
  const cosTheta = exp2(log2(float(0.244)).div(power.add(1))).clamp(0, 1).toVar('cosTheta');
  const sinTheta = float(1).sub(cosTheta.mul(cosTheta)).max(0).sqrt();
  return sinTheta.div(max(cosTheta, float(1e-4)));
});

/** TSL form of {@link screenEdgeFade}. */
export const screenEdgeFadeNode = Fn(([screenUv, fade]: [Vec2Node, FloatNode]) => {
  const distanceToEdge = vec2(1, 1).sub(screenUv.mul(2).sub(1).abs()).div(max(fade, float(1e-4)));
  const fx = smoothstep(float(0), float(1), distanceToEdge.x);
  const fy = smoothstep(float(0), float(1), distanceToEdge.y);
  return fx.mul(fy);
});

/* ------------------------------------------------------------------------- *
 * Trace shader
 * ------------------------------------------------------------------------- */

interface TraceResources {
  readonly guide: THREE.Texture;
  readonly hiZ: THREE.Texture;
  readonly color: THREE.Texture;
  readonly surface: THREE.Texture | null;
  readonly velocity: THREE.Texture | null;
  readonly probe: IndirectRadianceProvider | null;
}

interface TraceUniforms {
  readonly projScale: Vec2Node;
  readonly cameraFar: FloatNode;
  readonly cameraNear: FloatNode;
  readonly resolution: Vec2Node;
  readonly hiZLevels: FloatNode;
  readonly colorLevels: FloatNode;
  readonly maxDistance: FloatNode;
  readonly thickness: FloatNode;
  readonly edgeFade: FloatNode;
  readonly maxRoughness: FloatNode;
  readonly defaultRoughness: FloatNode;
  readonly defaultMetalness: FloatNode;
  readonly defaultReflectance: FloatNode;
  readonly fallbackRadiance: Vec3Node;
  readonly intensity: FloatNode;
  readonly ndcToUv: Vec2Node;
  readonly useVelocity: FloatNode;
  readonly viewToWorld: THREE.Node<'mat3'>;
}

/**
 * The Hi-Z screen-space trace.
 *
 * Reading order:
 *
 * 1. **Setup** — reconstruct the view position and normal, fetch or default the
 *    surface parameters, build the reflection ray and reject anything too rough
 *    or facing away.
 * 2. **Projection** — clip the ray to the near plane, project both endpoints,
 *    and set up the `(uv, 1/z)` segment the march walks.
 * 3. **Traversal** — the quadtree DDA described in the module header.
 * 4. **Refinement** — a few bisections to place the hit inside the last texel.
 * 5. **Shading** — cone-traced colour fetch, confidence, probe fallback, and
 *    the environment-BRDF weighted delta.
 */
function ssrTraceFragment(
  tier: SSRTierConfig,
  res: TraceResources,
  u: TraceUniforms,
): THREE.Node {
  const guideNode = texture(res.guide);
  // Explicit-level sampling has to go through `texture(tex, uv, level)`:
  // `TextureNode.sample()` only takes a UV.
  const hiZAt = (uvNode: Vec2Node, levelNode: FloatNode): FloatNode =>
    texture(res.hiZ, uvNode, levelNode).x;
  const colorAt = (uvNode: Vec2Node, levelNode: FloatNode): Vec3Node =>
    texture(res.color, uvNode, levelNode).xyz;
  const surfaceNode = res.surface === null ? null : texture(res.surface);
  const velocityNode = res.velocity === null ? null : texture(res.velocity);
  const coneOffsets = vogelDisk(tier.coneTaps);

  return Fn(() => {
    const base = uv();
    const centerGuide = decodeGuide(guideNode.sample(base), u.cameraFar).toVar('guide');
    const viewZ = centerGuide.w.toVar('viewZ');

    const delta = vec3(0).toVar('delta');

    // Sky: nothing to reflect off.
    If(viewZ.lessThan(u.cameraFar.mul(0.999)), () => {
      /* -- 1. setup ---------------------------------------------------- */

      const normal = centerGuide.xyz.toVar('normal');
      const position = viewPositionFromDepth(base, viewZ, u.projScale).toVar('position');
      const toEye = normalize(position.negate()).toVar('toEye');
      const nDotV = saturate(dot(normal, toEye)).toVar('nDotV');

      const surface =
        surfaceNode === null
          ? vec3(u.defaultRoughness, u.defaultMetalness, u.defaultReflectance).toVar('surface')
          : surfaceNode.sample(base).xyz.toVar('surface');
      const roughness = surface.x.clamp(0.015, 1).toVar('roughness');
      const metalness = surface.y.toVar('metalness');
      const reflectance = surface.z.toVar('reflectance');

      // Dielectric F0 is the reflectance; metals take their F0 from the albedo,
      // which this pass does not have, so a neutral value is used and the
      // material's own probe term carries the tint. Metals are rare in this
      // scene (wet mud, stone, cloth) so the approximation is cheap to accept.
      const f0 = mix(vec3(reflectance), vec3(0.9), metalness).toVar('f0');

      const rayDirection = reflect(position.normalize(), normal).toVar('ray');

      // Rays pointing back towards the camera reflect what is *behind* the
      // viewer, which by definition is not on screen. Fade rather than cut, or
      // the transition shows up as a hard band across a curved surface.
      const backFacing = saturate(rayDirection.dot(toEye).mul(2)).toVar('backFacing');
      const roughFade = saturate(
        float(1).sub(smoothstep(u.maxRoughness.mul(0.7), u.maxRoughness, roughness)),
      ).toVar('roughFade');

      If(roughFade.mul(backFacing.oneMinus()).greaterThan(float(0.001)), () => {
        /* -- 2. projection --------------------------------------------- */

        // Clip the ray to the near plane so the projection never crosses w = 0.
        const rayLength = u.maxDistance.toVar('rayLength');
        const endZ = position.z.add(rayDirection.z.mul(rayLength));
        If(endZ.greaterThan(u.cameraNear.negate()), () => {
          rayLength.assign(
            u.cameraNear.negate().sub(position.z).div(
              rayDirection.z.sign().mul(max(rayDirection.z.abs(), float(1e-5))),
            ),
          );
        });

        const endPosition = position.add(rayDirection.mul(rayLength)).toVar('endPosition');
        const uv0 = screenUvFromViewPosition(position, u.projScale).toVar('uv0');
        const uv1 = screenUvFromViewPosition(endPosition, u.projScale).toVar('uv1');
        // Reciprocal view depth: affine in screen space, so the interpolation
        // below is exact rather than an approximation that drifts with range.
        const q0 = float(1).div(max(viewZ, float(1e-4))).toVar('q0');
        const q1 = float(1).div(max(endPosition.z.negate(), float(1e-4))).toVar('q1');

        const duv = uv1.sub(uv0).toVar('duv');
        const dq = q1.sub(q0).toVar('dq');
        // Guard against a degenerate axis: a ray exactly along a texel row makes
        // one component of `duv` zero, and the boundary solve would divide by it.
        const duvSafe = vec2(
          duv.x.abs().lessThan(float(1e-8)).select(float(1e-8), duv.x),
          duv.y.abs().lessThan(float(1e-8)).select(float(1e-8), duv.y),
        ).toVar('duvSafe');
        const stepSign = vec2(duvSafe.x.sign(), duvSafe.y.sign()).toVar('stepSign');
        // `step(0, d)` picks the far edge of the cell in the direction of travel.
        const nearEdge = vec2(
          stepSign.x.greaterThan(float(0)).select(float(1), float(0)),
          stepSign.y.greaterThan(float(0)).select(float(1), float(0)),
        ).toVar('nearEdge');

        /* -- 3. traversal ---------------------------------------------- */

        const texel = vec2(1, 1).div(u.resolution).toVar('texel');
        // Start one texel along the ray. Beginning at s = 0 would immediately
        // self-intersect: the ray's depth equals the surface's depth there.
        const sStart = min(texel.div(duvSafe.abs()).x, texel.div(duvSafe.abs()).y)
          .abs()
          .max(1e-4)
          .toVar('sStart');
        const s = sStart.toVar('s');
        const sBefore = sStart.toVar('sBefore');
        const level = float(Math.min(1, tier.maxHiZLevel)).toVar('level');
        const hit = float(0).toVar('hit');
        const hitUv = base.toVar('hitUv');
        const hitZ = float(0).toVar('hitZ');

        Loop({ start: 0, end: tier.maxSteps, type: 'int' }, () => {
          const sampleUv = uv0.add(duv.mul(s)).toVar('sampleUv');

          // Off screen: the march is over. Confidence stays 0 and the probe
          // keeps the surface, which is the whole point of the delta form.
          If(
            sampleUv.x.lessThan(0).or(sampleUv.x.greaterThan(1)).or(
              sampleUv.y.lessThan(0).or(sampleUv.y.greaterThan(1)),
            ).or(s.greaterThan(1)),
            () => {
              Break();
            },
          );

          const rayZ = float(1).div(max(q0.add(dq.mul(s)), float(1e-6))).toVar('rayZ');
          // Level `L` texel = closest surface in a 2^L block. Stored normalised.
          const cellZ = hiZAt(sampleUv, level).mul(u.cameraFar).toVar('cellZ');

          If(rayZ.lessThan(cellZ), () => {
            /* Ray is in front of everything in this cell: skip the whole cell
             * and coarsen. This is where the hierarchy pays for itself — over
             * open sky the level climbs to the top of the pyramid in three
             * iterations and the rest of the screen is crossed in one step. */
            const cellSize = texel.mul(exp2(level)).toVar('cellSize');
            const cell = sampleUv.div(cellSize).floor();
            const boundary = cell.add(nearEdge).mul(cellSize);
            const advance = boundary.sub(sampleUv).div(duvSafe).toVar('advance');
            sBefore.assign(s);
            s.assign(s.add(min(advance.x, advance.y).max(float(1e-6))).add(1e-6));
            level.assign(min(level.add(1), u.hiZLevels.sub(1)));
          }).Else(() => {
            If(level.lessThan(float(0.5)), () => {
              // Level 0 and behind the surface: accept only inside the
              // thickness window, otherwise the ray passed through empty space
              // behind a thin object and the march must continue.
              If(rayZ.lessThan(cellZ.add(u.thickness.mul(max(cellZ, float(1))))), () => {
                hit.assign(1);
                hitUv.assign(sampleUv);
                hitZ.assign(rayZ);
                Break();
              });
              const cellSize = texel.toVar('cellSize0');
              const cell = sampleUv.div(cellSize).floor();
              const boundary = cell.add(nearEdge).mul(cellSize);
              const advance = boundary.sub(sampleUv).div(duvSafe);
              sBefore.assign(s);
              s.assign(s.add(min(advance.x, advance.y).max(float(1e-6))).add(1e-6));
            }).Else(() => {
              // Something is in this cell: refine without advancing.
              level.assign(level.sub(1));
            });
          });
        });

        /* -- 4. refinement --------------------------------------------- */

        // Bisect between the last known-empty parameter and the hit, so the hit
        // lands on the surface rather than up to a texel past it. Without this
        // a grazing ray on a sloped surface reflects from visibly the wrong
        // place, which reads as the reflection "sliding".
        for (let i = 0; i < tier.refineSteps; i++) {
          If(hit.greaterThan(float(0.5)), () => {
            const sMid = sBefore.add(s).mul(0.5).toVar(`sMid${i}`);
            const midUv = uv0.add(duv.mul(sMid)).toVar(`midUv${i}`);
            const midRayZ = float(1).div(max(q0.add(dq.mul(sMid)), float(1e-6)));
            const midSceneZ = hiZAt(midUv, float(0)).mul(u.cameraFar);
            If(midRayZ.lessThan(midSceneZ), () => {
              sBefore.assign(sMid);
            }).Else(() => {
              s.assign(sMid);
              hitUv.assign(midUv);
              hitZ.assign(midRayZ);
            });
          });
        }

        /* -- 5. shading ------------------------------------------------ */

        // Cone footprint at the hit, in half-resolution pixels.
        const travelPixels = hitUv.sub(base).mul(u.resolution).length().toVar('travelPixels');
        const coneRadius = coneTangentNode(roughness).mul(travelPixels).mul(0.5).toVar('coneRadius');
        const mip = log2(max(coneRadius.mul(2), float(1)))
          .clamp(0, u.colorLevels.sub(1))
          .toVar('mip');

        // The colour buffer is the previous frame's, so the hit's *screen*
        // position has to be walked back through the velocity buffer before the
        // colour is fetched. Skipping this makes every reflection of a moving
        // object lag by a frame in screen space, which is very visible on a
        // reflection of the player's own legs.
        const colorUv = hitUv.toVar('colorUv');
        if (velocityNode !== null) {
          const motion = velocityNode.sample(hitUv).xy.mul(u.ndcToUv);
          colorUv.assign(
            u.useVelocity.greaterThan(float(0.5)).select(hitUv.sub(motion), hitUv),
          );
        }

        const radiance = vec3(0).toVar('radiance');
        if (coneOffsets.length <= 1) {
          radiance.assign(colorAt(colorUv, mip));
        } else {
          // Average the cone's screen footprint with a golden-angle disk. The
          // mip already carries most of the blur; these taps remove the boxy
          // structure a single mip fetch leaves on a wide cone.
          const tapScale = coneRadius.mul(texel).toVar('tapScale');
          for (const [ox, oy] of coneOffsets) {
            radiance.addAssign(colorAt(colorUv.add(tapScale.mul(vec2(ox, oy))), mip));
          }
          radiance.assign(radiance.div(coneOffsets.length));
        }

        /* -- confidence ------------------------------------------------ */

        const edge = screenEdgeFadeNode(hitUv, u.edgeFade).toVar('edge');
        // Fade with ray length so a ray that used its whole budget without
        // converging does not slam to full strength at the last step.
        const distanceFade = saturate(
          float(1).sub(smoothstep(float(0.75), float(1), s)),
        ).toVar('distanceFade');
        const confidence = hit
          .mul(edge)
          .mul(distanceFade)
          .mul(roughFade)
          .mul(backFacing.oneMinus())
          .clamp(0, 1)
          .toVar('confidence');

        /* -- probe fallback and the delta ------------------------------ */

        const probeRadiance = vec3(0).toVar('probeRadiance');
        if (res.probe !== null) {
          const worldReflection = u.viewToWorld.mul(rayDirection).normalize();
          probeRadiance.assign(res.probe.prefilteredRadiance(worldReflection, roughness));
        } else {
          probeRadiance.assign(u.fallbackRadiance);
        }

        // F_env = F0·A + B  (Lazarov's split-sum fit).
        const ab = envBRDFApproxNode(roughness, nDotV).toVar('ab');
        const fresnelWeight = f0.mul(ab.x).add(ab.y).toVar('fresnelWeight');

        // The material already added `probeRadiance · fresnelWeight`. Adding the
        // difference replaces it exactly where confidence is 1 and leaves it
        // untouched where confidence is 0 — no seam, ever.
        delta.assign(
          radiance
            .sub(probeRadiance)
            .mul(fresnelWeight)
            .mul(confidence)
            .mul(u.intensity),
        );
      });
    });

    // `w` is reserved for the temporal accumulator's per-pixel history length.
    return vec4(delta, 0);
  })();
}

/* ------------------------------------------------------------------------- *
 * Module
 * ------------------------------------------------------------------------- */

/**
 * The screen-space reflection render module.
 *
 * Runs in `lateUpdate`, sharing the guide buffer with GTAO (whichever module
 * initialises first allocates it; see `acquireGuideBuffer`).
 */
export class SSRModule implements GameModule, SSRService {
  readonly name = 'render.ssr';

  #quality: SSRQuality;
  readonly #options: Required<Omit<SSROptions, 'quality' | 'fallbackRadiance'>> & {
    fallbackRadiance: THREE.Color;
  };

  /* -- uniforms ---------------------------------------------------------- */

  readonly #projScale = uniform(new THREE.Vector2(1, 1));
  readonly #cameraFar = uniform(1000);
  readonly #cameraNear = uniform(0.1);
  readonly #lowResolution = uniform(new THREE.Vector2(1, 1));
  readonly #ndcToUv = uniform(new THREE.Vector2(0.5, -0.5));
  readonly #hiZLevels = uniform(1);
  readonly #colorLevels = uniform(1);
  readonly #maxDistance = uniform(40);
  readonly #thickness = uniform(0.12);
  readonly #edgeFade = uniform(0.12);
  readonly #maxRoughness = uniform(0.6);
  readonly #defaultRoughness = uniform(0.28);
  readonly #defaultMetalness = uniform(0);
  readonly #defaultReflectance = uniform(0.05);
  readonly #fallbackRadiance = uniform(new THREE.Vector3(0.055, 0.062, 0.075));
  readonly #intensity = uniform(1);
  readonly #useVelocity = uniform(0);
  readonly #viewToWorld = uniform(new THREE.Matrix3());

  /* -- GPU resources ----------------------------------------------------- */

  #guide: GuideBufferProvider | null = null;
  #ownedGuide: GuideBufferPass | null = null;
  #hiZ: MipChain | null = null;
  #colorPyramid: MipChain | null = null;
  #tracePass: FullScreenPass | null = null;
  #denoiser: AtrousDenoiser | null = null;
  #temporal: TemporalAccumulator | null = null;
  #deltaNode: THREE.Node<'vec3'> | null = null;

  readonly #scope = new RendererStateScope();
  readonly #reprojection = new THREE.Matrix4();
  readonly #previousViewProjection = new THREE.Matrix4();

  #ctx: GameContext | null = null;
  #motion: MotionVectorSource | undefined;
  #sceneColorProvider: SceneColorProvider | undefined;
  #explicitSceneColor: THREE.Texture | null = null;
  #explicitIsPrevious = true;
  #boundSceneColor: THREE.Texture | null = null;
  #surface: SurfaceParameterProvider | undefined;
  #probe: IndirectRadianceProvider | undefined;

  #hasPreviousFrame = false;
  #width = 0;
  #height = 0;
  #warnedNoSceneColor = false;
  #warnedNoRenderer = false;

  constructor(options: SSROptions = {}) {
    this.#quality = options.quality ?? 'high';
    this.#options = {
      maxDistance: options.maxDistance ?? 40,
      thickness: options.thickness ?? 0.12,
      edgeFade: options.edgeFade ?? 0.12,
      maxRoughness: options.maxRoughness ?? 0.6,
      defaultRoughness: options.defaultRoughness ?? 0.28,
      defaultMetalness: options.defaultMetalness ?? 0,
      defaultReflectance: options.defaultReflectance ?? 0.05,
      intensity: options.intensity ?? 1,
      resolutionScale: options.resolutionScale ?? 0.5,
      temporalMinAlpha: options.temporalMinAlpha ?? 0.2,
      fallbackRadiance: new THREE.Color(
        options.fallbackRadiance ?? new THREE.Color(0.055, 0.062, 0.075),
      ),
    };
  }

  /* -- GameModule -------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    this.#maxDistance.value = this.#options.maxDistance;
    this.#thickness.value = this.#options.thickness;
    this.#edgeFade.value = this.#options.edgeFade;
    this.#maxRoughness.value = this.#options.maxRoughness;
    this.#defaultRoughness.value = this.#options.defaultRoughness;
    this.#defaultMetalness.value = this.#options.defaultMetalness;
    this.#defaultReflectance.value = this.#options.defaultReflectance;
    this.#intensity.value = this.#options.intensity;
    const fallback = this.#options.fallbackRadiance;
    this.#fallbackRadiance.value.set(fallback.r, fallback.g, fallback.b);

    const guide = acquireGuideBuffer(ctx, { resolutionScale: this.#options.resolutionScale });
    this.#guide = guide.provider;
    this.#ownedGuide = guide.owned;

    this.#motion = tryGetMotionVectors(ctx);
    if (this.#motion?.ndcToUv !== undefined) this.#ndcToUv.value.copy(this.#motion.ndcToUv);

    this.#sceneColorProvider = ctx.services.tryGet<SceneColorProvider>(SCENE_COLOR_SERVICE_ID);
    this.#surface = ctx.services.tryGet<SurfaceParameterProvider>(
      SURFACE_PARAMETERS_SERVICE_ID,
    );
    this.#probe = ctx.services.tryGet<IndirectRadianceProvider>(IBL_SERVICE_ID);

    ctx.services.register(SSRKey, this);
  }

  lateUpdate(ctx: GameContext): void {
    if (this.#quality === 'off') return;

    const renderer = asNodeRenderer(ctx.renderer);
    if (renderer === null) {
      if (!this.#warnedNoRenderer) {
        this.#warnedNoRenderer = true;
        console.warn('[SSR] the renderer is not a node renderer; reflections are disabled.');
      }
      return;
    }

    const sceneColor = this.#resolveSceneColor();
    if (sceneColor === null) {
      if (!this.#warnedNoSceneColor) {
        this.#warnedNoSceneColor = true;
        console.warn(
          '[SSR] no scene colour is available, so screen-space reflections contribute ' +
            `nothing. Register a SceneColorProvider under "${SCENE_COLOR_SERVICE_ID}" ` +
            '(linear HDR, pre-tone-mapping) or call SSRModule.setSceneColor().',
        );
      }
      return;
    }
    if (sceneColor !== this.#boundSceneColor) this.#build(sceneColor);

    const size = renderer.getDrawingBufferSize(_size);
    this.#resize(size.width, size.height);
    if (this.#width === 0 || this.#height === 0) return;

    const camera = ctx.camera;
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    const projection = camera.projectionMatrix.elements;
    this.#projScale.value.set(projection[0] ?? 1, projection[5] ?? 1);
    this.#cameraFar.value = camera.far;
    this.#cameraNear.value = camera.near;
    // View -> world rotation, for looking up the probe with a world direction.
    this.#viewToWorld.value.setFromMatrix4(camera.matrixWorld);

    const velocityLive =
      this.#motion?.velocityTexture != null && this.#motion.historyValid === true;
    this.#useVelocity.value = velocityLive ? 1 : 0;
    if (!velocityLive) {
      this.#reprojection.copy(this.#previousViewProjection).multiply(camera.matrixWorld);
    }
    const invalidate =
      !this.#hasPreviousFrame || (this.#motion !== undefined && !this.#motion.historyValid);

    this.#scope.begin(renderer, ctx.scene);
    try {
      this.#ownedGuide?.render(renderer, ctx.scene, camera);
      this.#hiZ?.build(renderer);
      this.#colorPyramid?.build(renderer);

      const trace = this.#tracePass;
      const denoiser = this.#denoiser;
      const temporal = this.#temporal;
      if (trace !== null && denoiser !== null && temporal !== null) {
        trace.render(renderer, denoiser.inputTarget);
        denoiser.render(renderer);
        temporal.render(renderer, this.#reprojection, velocityLive, invalidate);
      }
    } finally {
      this.#scope.end();
    }

    if (this.#motion !== undefined) {
      this.#previousViewProjection
        .copy(this.#motion.previousProjectionMatrix)
        .multiply(this.#motion.previousViewMatrix);
    } else {
      this.#previousViewProjection
        .copy(camera.projectionMatrix)
        .multiply(camera.matrixWorldInverse);
    }
    this.#hasPreviousFrame = true;
  }

  dispose(): void {
    const ctx = this.#ctx;
    if (ctx !== null) {
      ctx.services.unregister(SSRKey);
      releaseGuideBuffer(ctx);
    }
    this.#teardown();
    this.#ctx = null;
  }

  /* -- SSRService -------------------------------------------------------- */

  get deltaNode(): THREE.Node<'vec3'> | null {
    return this.#quality === 'off' ? null : this.#deltaNode;
  }

  composite(sceneColor: THREE.Node<'vec4'>): THREE.Node<'vec4'> {
    const delta = this.deltaNode;
    if (delta === null) return sceneColor;
    return vec4(sceneColor.xyz.add(delta), sceneColor.w);
  }

  get reflectionTexture(): THREE.Texture | null {
    return this.#temporal?.outputTexture ?? null;
  }

  get hiZTexture(): THREE.Texture | null {
    return this.#hiZ?.texture ?? null;
  }

  get quality(): SSRQuality {
    return this.#quality;
  }

  setQuality(quality: SSRQuality): void {
    if (quality === this.#quality) return;
    this.#quality = quality;
    if (quality === 'off') return;
    const sceneColor = this.#resolveSceneColor();
    if (sceneColor !== null) this.#build(sceneColor);
  }

  setSceneColor(texture_: THREE.Texture | null, isPreviousFrame = true): void {
    this.#explicitSceneColor = texture_;
    this.#explicitIsPrevious = isPreviousFrame;
  }

  /** Explicitly supply the probe when the IBL service registers late. */
  setProbe(probe: IndirectRadianceProvider | null): void {
    this.#probe = probe ?? undefined;
    const sceneColor = this.#resolveSceneColor();
    if (sceneColor !== null) this.#build(sceneColor);
  }

  get stats(): SSRStats {
    const tier = this.#quality === 'off' ? null : SSR_TIERS[this.#quality];
    return {
      enabled: this.#quality !== 'off',
      quality: this.#quality,
      width: this.#guide?.halfWidth ?? 0,
      height: this.#guide?.halfHeight ?? 0,
      maxSteps: tier?.maxSteps ?? 0,
      hiZLevels: this.#hiZ?.levels ?? 0,
      coneTaps: tier?.coneTaps ?? 0,
      sceneColorAvailable: this.#resolveSceneColor() !== null,
      surfaceParametersAvailable: this.#surface?.surfaceTexture != null,
      probeAvailable: this.#probe !== undefined,
      temporalFromVelocity: this.#motion?.velocityTexture != null,
    };
  }

  /* -- internals --------------------------------------------------------- */

  #resolveSceneColor(): THREE.Texture | null {
    if (this.#explicitSceneColor !== null) return this.#explicitSceneColor;
    return this.#sceneColorProvider?.sceneColorTexture ?? null;
  }

  #isPreviousFrameColor(): boolean {
    if (this.#explicitSceneColor !== null) return this.#explicitIsPrevious;
    return this.#sceneColorProvider?.isPreviousFrame ?? true;
  }

  #build(sceneColor: THREE.Texture): void {
    const guide = this.#guide;
    if (guide === null || this.#quality === 'off') return;
    const halfGuide = guide.halfGuideTexture ?? guide.guideTexture;
    if (halfGuide === null) return;

    this.#teardown();
    this.#boundSceneColor = sceneColor;
    const tier = SSR_TIERS[this.#quality];

    /* Hi-Z: a single-channel min-depth pyramid over the half-resolution guide.
     * `min` is what makes the skip test conservative — if the ray is nearer than
     * the closest surface in a block, nothing in that block can be hit. */
    const guideNode = texture(halfGuide);
    this.#hiZ = new MipChain(
      'ssr.hiz',
      Fn(() => vec4(guideNode.sample(uv()).w))(),
      { reduction: 'min', format: THREE.RedFormat, type: THREE.HalfFloatType },
    );

    /* Colour: an averaged pyramid whose level `L` is a 2^L box blur, which is
     * what the cone trace samples. Trilinear so a roughness gradient across a
     * surface does not band at mip boundaries. */
    const sceneColorNode = texture(sceneColor);
    this.#colorPyramid = new MipChain(
      'ssr.color',
      Fn(() => vec4(sceneColorNode.sample(uv()).xyz, 1))(),
      { reduction: 'average', format: THREE.RGBAFormat, type: THREE.HalfFloatType },
    );
    this.#colorPyramid.texture.minFilter = THREE.LinearMipmapLinearFilter;
    this.#colorPyramid.texture.magFilter = THREE.LinearFilter;

    this.#denoiser = new AtrousDenoiser(halfGuide, this.#cameraFar, {
      name: 'ssr',
      iterations: tier.denoiseIterations,
      radius: tier.denoiseRadius,
      // Reflections change fast across a depth step (that *is* the reflection
      // changing), so the depth window is tighter than AO's and the normal
      // window much tighter: two facets at 15° reflect completely different
      // things.
      depthSigma: 0.03,
      normalSigma: 128,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
    });

    const velocity = this.#isPreviousFrameColor()
      ? (this.#motion?.velocityTexture ?? null)
      : null;

    this.#tracePass = new FullScreenPass(
      `ssr.trace.${this.#quality}`,
      ssrTraceFragment(
        tier,
        {
          guide: halfGuide,
          hiZ: this.#hiZ.texture,
          color: this.#colorPyramid.texture,
          surface: this.#surface?.surfaceTexture ?? null,
          velocity,
          probe: this.#probe ?? null,
        },
        {
          projScale: this.#projScale,
          cameraFar: this.#cameraFar,
          cameraNear: this.#cameraNear,
          resolution: this.#lowResolution,
          hiZLevels: this.#hiZLevels,
          colorLevels: this.#colorLevels,
          maxDistance: this.#maxDistance,
          thickness: this.#thickness,
          edgeFade: this.#edgeFade,
          maxRoughness: this.#maxRoughness,
          defaultRoughness: this.#defaultRoughness,
          defaultMetalness: this.#defaultMetalness,
          defaultReflectance: this.#defaultReflectance,
          fallbackRadiance: this.#fallbackRadiance,
          intensity: this.#intensity,
          ndcToUv: this.#ndcToUv,
          useVelocity: this.#useVelocity,
          viewToWorld: this.#viewToWorld,
        },
      ),
    );

    const temporalUniforms: TemporalUniforms = {
      projScale: this.#projScale,
      cameraFar: this.#cameraFar,
      resolution: this.#lowResolution,
      ndcToUv: this.#ndcToUv,
    };
    this.#temporal = new TemporalAccumulator(
      this.#denoiser.outputTarget.texture,
      halfGuide,
      this.#motion?.velocityTexture ?? null,
      temporalUniforms,
      {
        name: 'ssr',
        signalChannels: 3,
        minAlpha: this.#options.temporalMinAlpha,
        // Reflections are view dependent, so a long history ghosts badly on a
        // turning camera. A short history plus a tight clamp is the right
        // trade: slightly noisier, but it never smears.
        clampGamma: 1.0,
        depthRejection: 0.03,
        format: THREE.RGBAFormat,
        type: THREE.HalfFloatType,
      },
    );

    this.#buildNodes(guide, halfGuide);
    this.#applySizes();
  }

  #buildNodes(guide: GuideBufferProvider, halfGuide: THREE.Texture): void {
    const temporal = this.#temporal;
    const fullGuide = guide.guideTexture;
    if (temporal === null || fullGuide === null) return;

    const upsample = bilateralUpsampleNode(
      temporal.outputTexture,
      halfGuide,
      this.#lowResolution,
      this.#cameraFar,
      { depthSigma: 0.03, normalSigma: 64 },
    );
    const fullGuideNode = texture(fullGuide);

    this.#deltaNode = Fn(() => {
      const screen = screenUV;
      const centre = decodeGuide(fullGuideNode.sample(screen), this.#cameraFar).toVar('centre');
      return upsample(screen, centre.w, centre.xyz).xyz;
    })();
  }

  #resize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.#width && h === this.#height) return;
    this.#width = w;
    this.#height = h;
    this.#ownedGuide?.setSize(w, h);
    this.#applySizes();
    this.#temporal?.reset();
    this.#hasPreviousFrame = false;
  }

  #applySizes(): void {
    if (this.#width === 0 || this.#height === 0) return;
    const guide = this.#guide;
    const lowWidth =
      guide?.halfWidth ?? Math.max(1, Math.round(this.#width * this.#options.resolutionScale));
    const lowHeight =
      guide?.halfHeight ?? Math.max(1, Math.round(this.#height * this.#options.resolutionScale));

    this.#lowResolution.value.set(lowWidth, lowHeight);
    this.#hiZ?.setSize(lowWidth, lowHeight);
    this.#colorPyramid?.setSize(lowWidth, lowHeight);
    this.#denoiser?.setSize(lowWidth, lowHeight);
    this.#temporal?.setSize(lowWidth, lowHeight);

    const tier = this.#quality === 'off' ? null : SSR_TIERS[this.#quality];
    const availableLevels = this.#hiZ?.levels ?? 1;
    this.#hiZLevels.value = Math.max(
      1,
      Math.min(availableLevels, (tier?.maxHiZLevel ?? 0) + 1),
    );
    this.#colorLevels.value = this.#colorPyramid?.levels ?? 1;
  }

  #teardown(): void {
    this.#tracePass?.dispose();
    this.#denoiser?.dispose();
    this.#temporal?.dispose();
    this.#hiZ?.dispose();
    this.#colorPyramid?.dispose();
    this.#tracePass = null;
    this.#denoiser = null;
    this.#temporal = null;
    this.#hiZ = null;
    this.#colorPyramid = null;
    this.#deltaNode = null;
    this.#boundSceneColor = null;
  }
}

/** Scratch vector for `getDrawingBufferSize`; the module is not re-entrant. */
const _size = new THREE.Vector2();

/**
 * Create the SSR module.
 *
 * ```ts
 * engine.add(createSSR({ quality: 'high', maxDistance: 40 }));
 * ```
 *
 * Register it after the motion-vector system and after whatever publishes the
 * scene colour; if the scene colour arrives later, call
 * {@link SSRModule.setSceneColor} and the pipeline rebuilds itself.
 */
export function createSSR(options: SSROptions = {}): SSRModule {
  return new SSRModule(options);
}

/** Construct, register with the engine, and return the module. */
export function registerSSR(ctx: GameContext, options: SSROptions = {}): SSRModule {
  const module = new SSRModule(options);
  ctx.engine.add(module);
  return module;
}

/* Suppress the unused-import diagnostics for TSL helpers that are only ever
 * referenced from inside node graphs the compiler cannot see into. */
void abs;
void cos;
void sin;
void mix;
