/**
 * @module render/post/Denoise
 *
 * The shared screen-space infrastructure that {@link module:render/post/GTAO}
 * and {@link module:render/post/SSR} are both built on: a geometry guide
 * buffer, hierarchical min/average pyramids, an edge-avoiding à-trous
 * denoiser, temporal accumulation, and a joint bilateral upsample.
 *
 * Nothing here is specific to occlusion or to reflections. Both effects are
 * *stochastic estimators sampled at half resolution*, and both therefore need
 * exactly the same four things, so they live together and are written once.
 *
 * ---
 *
 * ## 1. The guide buffer
 *
 * Every edge-stopping filter in this file needs to know, per pixel, "what
 * surface is this?". That is stored in a single RGBA16F attachment:
 *
 * ```
 * rgb = view-space shading normal, unit length, signed
 * a   = linear view depth / camera.far, in [0, 1]
 * ```
 *
 * Two decisions in that layout are load bearing.
 *
 * **Linear depth, normalised by far.** Storing `viewZ / far` rather than a
 * hardware depth value removes every difference between the two backends: WebGPU
 * clips to `z ∈ [0, 1]` and WebGL2 to `z ∈ [-1, 1]`, and a shader that
 * reconstructs view position through `projectionMatrixInverse` has to know
 * which. Reconstruction here uses only the two projection scale factors
 * (see {@link viewPositionFromDepth}), so the same TSL runs bit-identically on
 * both. Half-float has constant *relative* precision (~4.9e-4), so normalising
 * by `far` costs nothing: at 60 m with a 2 km far plane the depth error is
 * ~3 cm, which is far below the AO radius and the SSR thickness threshold.
 *
 * **Clear to `(0, 0, 1, 1)`.** Sky pixels then decode as "normal facing the
 * camera at the far plane" instead of a zero-length normal that `normalize()`
 * turns into NaN and a zero depth that reads as "touching the lens".
 *
 * The buffer is filled by a depth+normal prepass (`scene.overrideMaterial`),
 * *or* adopted wholesale from an existing G-buffer if one is registered under
 * {@link GuideBufferKey} — see {@link acquireGuideBuffer}.
 *
 * ## 2. Pyramids
 *
 * {@link MipChain} builds a mip pyramid by repeated 2×2 reduction: `min` for
 * the Hi-Z depth pyramid SSR marches against, `average` for the blurred colour
 * pyramid SSR cone-traces against.
 *
 * It does *not* render level `i` while sampling level `i-1` of the same
 * texture. WebGPU binds a sampled texture as a view over its whole mip range,
 * which overlaps the render attachment subresource, and that is a validation
 * error — it would work on the WebGL2 fallback and hard-fail on the backend
 * real users get. Instead two mipped targets are kept in lockstep: level `i` of
 * A is reduced from level `i-1` of B, then level `i` of B from level `i-1` of A.
 * Two tiny passes per level, no aliasing hazard, no copies, and both chains end
 * up complete.
 *
 * ## 3. À-trous denoise
 *
 * Dammertz, Sewtz, Hanika & Lensch, *"Edge-Avoiding À-Trous Wavelet Transform
 * for fast Global Illumination Filtering"*, HPG 2010. A fixed 5×5 B3-spline
 * kernel is applied `N` times with the tap spacing doubling each iteration, so
 * `N` passes reach the support of a `(2^N · 4 + 1)`-wide filter at `N · 25`
 * taps instead of `(2^N·4+1)²`. Each tap is weighted by an edge-stopping
 * function of depth and normal (Schied et al., *"Spatiotemporal Variance-Guided
 * Filtering"*, HPG 2017, §4.2), so the filter never blurs across a silhouette
 * or a crease — which is the difference between "denoised" and "smeared".
 *
 * ## 4. Temporal accumulation
 *
 * History is reprojected with the velocity buffer published by
 * {@link module:render/post/Motion} when it exists, and by camera-only
 * reprojection through this module's own snapshotted matrices when it does not.
 * It is rejected on depth mismatch and clamped to the local neighbourhood's
 * mean ± γ·σ (Karis, *"High Quality Temporal Supersampling"*, SIGGRAPH 2014) so
 * that a wrong reprojection produces one soft frame rather than a permanent
 * smear.
 *
 * ## 5. Joint bilateral upsample
 *
 * Kopf, Cohen, Lischinski & Uyttendaele, *"Joint Bilateral Upsampling"*,
 * SIGGRAPH 2007. The four half-resolution taps around a full-resolution pixel
 * are weighted by bilinear position *and* by how well their guide (depth,
 * normal) matches the full-resolution guide. Without this the half-resolution
 * signal leaks a two-pixel halo across every silhouette, which is the single
 * most recognisable "cheap AO" artefact.
 *
 * ---
 *
 * ## References
 *
 * - H. Dammertz et al., *Edge-Avoiding À-Trous Wavelet Transform*, HPG 2010.
 * - C. Schied et al., *Spatiotemporal Variance-Guided Filtering*, HPG 2017.
 * - J. Kopf et al., *Joint Bilateral Upsampling*, SIGGRAPH 2007.
 * - B. Karis, *High Quality Temporal Supersampling*, SIGGRAPH 2014.
 * - J. Jimenez, *Next Generation Post Processing in Call of Duty: Advanced
 *   Warfare*, SIGGRAPH 2014 (interleaved gradient noise).
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  cameraFar,
  float,
  max,
  min,
  normalView,
  normalize,
  positionView,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import { serviceKey } from '../../core/ServiceLocator';
import type { GameContext, RendererHandle } from '../../core/types';

/* ------------------------------------------------------------------------- *
 * Local aliases for the node types this module passes around
 * ------------------------------------------------------------------------- */

/** A TSL node of a known GLSL/WGSL type. */
export type FloatNode = THREE.Node<'float'>;
export type Vec2Node = THREE.Node<'vec2'>;
export type Vec3Node = THREE.Node<'vec3'>;
export type Vec4Node = THREE.Node<'vec4'>;

/**
 * three's node `Renderer` (the common base of `WebGPURenderer`). The engine's
 * {@link RendererHandle.three} is typed as a union with the classic
 * `WebGLRenderer` purely to satisfy the frozen core contract; in this project
 * the factory always constructs a `WebGPURenderer` (with `forceWebGL` for the
 * WebGL2 tier), so the narrowing below always succeeds at runtime.
 */
export type NodeRenderer = THREE.Renderer;

/**
 * Narrow a {@link RendererHandle} to the node renderer, or `null` if the handle
 * somehow wraps a classic `WebGLRenderer` (which cannot render TSL at all).
 *
 * The discriminator is `.backend`, which only the node renderer has.
 */
export function asNodeRenderer(handle: RendererHandle): NodeRenderer | null {
  const candidate = handle.three as unknown as { backend?: unknown };
  return candidate.backend === undefined ? null : (handle.three as NodeRenderer);
}

/* ------------------------------------------------------------------------- *
 * Pure math — exported so tests can pin the behaviour without a GPU
 * ------------------------------------------------------------------------- */

/**
 * The B3 cubic-spline kernel used by the à-trous transform, as its separable
 * 1D factor. `[1, 4, 6, 4, 1] / 16`.
 *
 * Dammertz et al. use exactly this kernel because repeatedly convolving it with
 * itself at doubled spacing converges to a Gaussian, so `N` iterations behave
 * like one wide Gaussian rather than like `N` visible box passes.
 */
export const B3_SPLINE_1D: readonly number[] = [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16];

/**
 * Separable 2D weight of the tap at `(dx, dy)` for a kernel of the given
 * radius. Radius 2 is the full 5×5 B3 spline; radius 1 collapses to the 3×3
 * `[1, 2, 1] / 4` tent, which is the medium/low quality tier.
 */
export function atrousTapWeight(dx: number, dy: number, radius = 2): number {
  const kernel = radius >= 2 ? B3_SPLINE_1D : [0.25, 0.5, 0.25];
  const wx = kernel[dx + radius];
  const wy = kernel[dy + radius];
  if (wx === undefined || wy === undefined) return 0;
  return wx * wy;
}

/**
 * Tap spacing for à-trous iteration `i`, in texels: `2^i`.
 *
 * "À trous" is French for "with holes": the kernel keeps its 5×5 shape but its
 * taps spread apart, so the filter support grows geometrically while the tap
 * count stays constant.
 */
export function atrousStepSize(iteration: number): number {
  return 1 << Math.max(0, Math.floor(iteration));
}

/**
 * Total filter support, in texels, after `iterations` à-trous passes with the
 * given kernel radius. Used to sanity-check that the configured iteration count
 * actually covers the sampling noise it is meant to hide.
 */
export function atrousSupportRadius(iterations: number, radius = 2): number {
  let support = 0;
  for (let i = 0; i < iterations; i++) support += radius * atrousStepSize(i);
  return support;
}

/**
 * Depth edge-stopping weight (SVGF §4.2).
 *
 * `exp(-|z_c - z_s| / (σ_z · |∇z · Δp| + ε))`. The denominator is the depth
 * change the *surface plane itself* would produce over the tap offset, so a
 * steeply sloped floor is not mistaken for a depth discontinuity. `gradient` is
 * that expected change; pass 1 when no gradient estimate is available and the
 * weight degenerates to a plain exponential falloff.
 *
 * @param centerDepth linear view depth of the centre tap
 * @param sampleDepth linear view depth of the neighbour tap
 * @param gradient    expected depth change across the offset, ≥ 0
 * @param sigma       tolerance multiplier; larger blurs more across depth
 */
export function depthEdgeWeight(
  centerDepth: number,
  sampleDepth: number,
  gradient: number,
  sigma: number,
): number {
  const denominator = sigma * Math.abs(gradient) + 1e-4;
  return Math.exp(-Math.abs(centerDepth - sampleDepth) / denominator);
}

/**
 * Normal edge-stopping weight (SVGF §4.2): `max(0, n_c · n_s)^σ_n`.
 *
 * The exponent is what makes it a *crease* detector rather than a soft cosine
 * falloff — at σ_n = 64 a 10° normal difference already halves the weight.
 */
export function normalEdgeWeight(dotNormals: number, sigma: number): number {
  return Math.pow(Math.max(0, dotNormals), sigma);
}

/**
 * Combined edge-stopping weight for one à-trous tap.
 *
 * Exported as one function (rather than leaving callers to multiply the parts)
 * so the test suite pins the *composition*, which is where sign and ordering
 * mistakes actually happen.
 */
export function atrousWeight(
  dx: number,
  dy: number,
  centerDepth: number,
  sampleDepth: number,
  dotNormals: number,
  options: { depthSigma: number; normalSigma: number; gradient?: number; radius?: number },
): number {
  const radius = options.radius ?? 2;
  const gradient = options.gradient ?? 1;
  return (
    atrousTapWeight(dx, dy, radius) *
    depthEdgeWeight(centerDepth, sampleDepth, gradient, options.depthSigma) *
    normalEdgeWeight(dotNormals, options.normalSigma)
  );
}

/**
 * Weights for a joint bilateral upsample of a half-resolution signal.
 *
 * `bilinear` are the four standard bilinear weights of the low-resolution taps
 * around the high-resolution sample point; `lowDepths` and `lowDotNormals` are
 * the guide values at those taps compared against the high-resolution pixel.
 *
 * Returns four weights summing to 1. If every tap is rejected — which happens
 * on a one-pixel-wide silhouette where no half-resolution sample belongs to the
 * same surface — the function falls back to plain bilinear rather than
 * returning zeros, because a slightly wrong value beats a black pixel.
 */
export function bilateralUpsampleWeights(
  bilinear: readonly number[],
  lowDepths: readonly number[],
  highDepth: number,
  lowDotNormals: readonly number[],
  options: { depthSigma: number; normalSigma: number },
): number[] {
  const weights: number[] = [];
  let total = 0;
  for (let i = 0; i < 4; i++) {
    const b = bilinear[i] ?? 0;
    const d = lowDepths[i] ?? highDepth;
    const n = lowDotNormals[i] ?? 1;
    // The depth tolerance is relative: a 1 % depth mismatch at 2 m and at 200 m
    // are the same surface-relative error, and an absolute epsilon would reject
    // everything in the distance.
    const w =
      b *
      depthEdgeWeight(highDepth, d, Math.max(highDepth, 1e-4), options.depthSigma) *
      normalEdgeWeight(n, options.normalSigma);
    weights.push(w);
    total += w;
  }
  if (total <= 1e-6) return [...bilinear.slice(0, 4)].map((w) => w ?? 0);
  return weights.map((w) => w / total);
}

/**
 * Temporal blend factor for a pixel whose history is `historyLength` frames
 * long.
 *
 * `max(minAlpha, 1 / (n + 1))` is an exponential moving average that behaves as
 * an exact arithmetic mean for the first `1/minAlpha` frames and then settles
 * into a fixed-weight filter. That matters: a fresh pixel converges in a couple
 * of frames instead of crawling towards the answer at the steady-state rate,
 * which is what makes disocclusions stop shimmering quickly.
 *
 * @param historyLength frames already accumulated, ≥ 0
 * @param minAlpha      steady-state weight of the new sample (0.05–0.15)
 */
export function temporalAlpha(historyLength: number, minAlpha: number): number {
  return Math.max(minAlpha, 1 / (Math.max(0, historyLength) + 1));
}

/**
 * Clamp `history` into `mean ± gamma·stdDev` (Karis 2014, "variance clipping").
 *
 * A reprojection that lands on a different surface returns a plausible-looking
 * but wrong colour. Clamping it to the statistics of the *current* frame's
 * neighbourhood converts that error from a persistent ghost into at most one
 * frame of extra noise.
 */
export function clampToNeighbourhood(
  history: number,
  mean: number,
  stdDev: number,
  gamma: number,
): number {
  const spread = Math.abs(stdDev) * gamma;
  return Math.min(Math.max(history, mean - spread), mean + spread);
}

/**
 * Standard deviation from the first two raw moments, guarded against the small
 * negative values that catastrophic cancellation produces when the neighbourhood
 * is uniform.
 */
export function stdDevFromMoments(mean: number, meanOfSquares: number): number {
  return Math.sqrt(Math.max(0, meanOfSquares - mean * mean));
}

/**
 * Number of usable levels in a `width × height` pyramid, counting level 0 and
 * stopping before any dimension would reach zero.
 */
export function mipChainLevels(width: number, height: number, maxLevels = 16): number {
  const smallest = Math.max(1, Math.min(width, height));
  return Math.max(1, Math.min(maxLevels, Math.floor(Math.log2(smallest)) + 1));
}

/**
 * Interleaved gradient noise (Jimenez, *Next Generation Post Processing in Call
 * of Duty: Advanced Warfare*, SIGGRAPH 2014).
 *
 * A low-discrepancy per-pixel dither that is *cheap* (one dot, two fracts) and,
 * unlike a blue-noise texture, needs no bandwidth. Offsetting the input by a
 * per-frame constant decorrelates it in time so the temporal filter has genuine
 * new information to integrate each frame.
 *
 * Exported in plain JS so the sequence's distribution can be asserted on.
 */
export function interleavedGradientNoise(x: number, y: number, frame = 0): number {
  // The frame offset is the golden-ratio-conjugate walk from the same talk; it
  // moves the whole pattern by an irrational fraction of its period so no two
  // nearby frames repeat.
  const fx = x + 5.588238 * (frame % 64);
  const fy = y + 5.588238 * (frame % 64);
  const magic = fx * 0.06711056 + fy * 0.00583715;
  return fract(52.9829189 * fract(magic));
}

function fract(value: number): number {
  return value - Math.floor(value);
}

/* ------------------------------------------------------------------------- *
 * Shared TSL helpers
 * ------------------------------------------------------------------------- */

/**
 * View-space position from a screen UV and a *linear* view depth.
 *
 * ```
 * x_view =  ndc.x · viewZ / P00
 * y_view =  ndc.y · viewZ / P11
 * z_view = -viewZ
 * ```
 *
 * derived directly from the perspective projection, with
 * `ndc = (2u - 1, 1 - 2v)` because three's render-target texture space is
 * Y-down while NDC is Y-up.
 *
 * Using the two projection scale factors instead of `projectionMatrixInverse`
 * is not a micro-optimisation: it removes the WebGPU (`z ∈ [0,1]`) versus
 * WebGL2 (`z ∈ [-1,1]`) clip-space difference from every shader in this
 * pipeline, so both backends run identical code.
 *
 * @param screenUv  texture-space UV, Y-down
 * @param viewZ     positive distance along the view axis, in world units
 * @param projScale `(P[0][0], P[1][1])` of the projection matrix
 */
export const viewPositionFromDepth = Fn(
  ([screenUv, viewZ, projScale]: [Vec2Node, FloatNode, Vec2Node]) => {
    const ndc = vec2(screenUv.x.mul(2).sub(1), float(1).sub(screenUv.y.mul(2)));
    return vec3(ndc.div(projScale).mul(viewZ), viewZ.negate());
  },
);

/**
 * Inverse of {@link viewPositionFromDepth}: project a view-space point back to
 * a texture-space UV. Returns `(u, v)`; the caller is responsible for rejecting
 * points behind the camera (`p.z >= 0`), which project to nonsense.
 */
export const screenUvFromViewPosition = Fn(
  ([viewPosition, projScale]: [Vec3Node, Vec2Node]) => {
    const invW = float(-1).div(min(viewPosition.z, float(-1e-4)));
    const ndc = viewPosition.xy.mul(projScale).mul(invW);
    return vec2(ndc.x.mul(0.5).add(0.5), float(0.5).sub(ndc.y.mul(0.5)));
  },
);

/** TSL form of {@link interleavedGradientNoise}. `pixel` is in pixels. */
export const interleavedGradientNoiseNode = Fn(
  ([pixel, frame]: [Vec2Node, FloatNode]) => {
    const shifted = pixel.add(float(5.588238).mul(frame.mod(64)));
    return shifted.dot(vec2(0.06711056, 0.00583715)).fract().mul(52.9829189).fract();
  },
);

/**
 * Decode a guide texel into `(normal, viewZ)`.
 *
 * The normal is renormalised because bilinear-free point sampling still leaves
 * half-float rounding, and an off-unit normal quietly biases every dot product
 * downstream.
 */
export const decodeGuide = Fn(([guide, cameraFar]: [Vec4Node, FloatNode]) => {
  return vec4(normalize(guide.xyz), guide.w.mul(cameraFar));
});

/**
 * Combined depth+normal edge-stopping weight, the TSL twin of
 * {@link atrousWeight} minus the constant kernel factor.
 */
export const edgeStoppingWeightNode = Fn(
  ([centerDepth, sampleDepth, dotNormals, depthSigma, normalSigma]: [
    FloatNode,
    FloatNode,
    FloatNode,
    FloatNode,
    FloatNode,
  ]) => {
    // Relative depth tolerance: an absolute epsilon rejects everything in the
    // distance and nothing up close.
    const scale = max(centerDepth, float(1e-3)).mul(depthSigma).add(1e-4);
    const depthWeight = centerDepth.sub(sampleDepth).abs().div(scale).negate().exp();
    const normalWeight = max(dotNormals, float(0)).pow(normalSigma);
    return depthWeight.mul(normalWeight);
  },
);

/* ------------------------------------------------------------------------- *
 * Full-screen pass plumbing
 * ------------------------------------------------------------------------- */

/** One `QuadMesh` shared by every pass in the process. */
const sharedQuad = new THREE.QuadMesh();

/**
 * Save/reset/restore of the renderer (and optionally scene) state around a
 * block of off-screen passes.
 *
 * Every effect in this pipeline renders several full-screen quads and possibly
 * a scene prepass into its own targets, in the middle of someone else's frame.
 * Leaving the render target, clear colour, MRT declaration or pixel ratio
 * changed afterwards corrupts the main pass in ways that are extremely annoying
 * to trace, so all of it is saved once and restored in a `finally`.
 *
 * This wraps `THREE.RendererUtils` rather than calling its combined
 * `resetRendererAndSceneState` helper because that helper's `@types/three`
 * signature is missing its `scene` parameter (r185), and calling it through a
 * cast would silently break when the typings are fixed.
 */
export class RendererStateScope {
  #renderer: NodeRenderer | null = null;
  #scene: THREE.Scene | null = null;
  #rendererState: THREE.RendererUtils.RendererState | undefined;
  #sceneState: THREE.RendererUtils.SceneState | undefined;

  /**
   * @param clearColor clear colour applied after the reset. Defaults to black.
   * @param clearAlpha clear alpha applied after the reset. Defaults to 1.
   */
  begin(
    renderer: NodeRenderer,
    scene: THREE.Scene | null,
    clearColor?: THREE.Color,
    clearAlpha = 1,
  ): void {
    this.#renderer = renderer;
    this.#scene = scene;
    this.#rendererState = THREE.RendererUtils.saveRendererState(renderer, this.#rendererState);

    renderer.setMRT(null);
    renderer.setRenderObjectFunction(null);
    renderer.setClearColor(clearColor ?? BLACK, clearAlpha);
    renderer.autoClear = true;
    // Off-screen targets are already sized in device pixels; leaving the pixel
    // ratio at its display value would scale every viewport a second time.
    renderer.setPixelRatio(1);

    if (scene !== null) {
      this.#sceneState = THREE.RendererUtils.saveSceneState(scene, this.#sceneState);
      scene.background = null;
      scene.backgroundNode = null;
      scene.overrideMaterial = null;
    }
  }

  end(): void {
    const renderer = this.#renderer;
    if (renderer !== null && this.#rendererState !== undefined) {
      THREE.RendererUtils.restoreRendererState(renderer, this.#rendererState);
    }
    if (this.#scene !== null && this.#sceneState !== undefined) {
      THREE.RendererUtils.restoreSceneState(this.#scene, this.#sceneState);
    }
    this.#renderer = null;
    this.#scene = null;
  }
}

const BLACK = new THREE.Color(0, 0, 0);

/**
 * A single full-screen shader invocation.
 *
 * Deliberately thin: it owns a material and nothing else. Render targets belong
 * to the effect that allocates them, because only the effect knows the
 * ping-pong topology.
 */
export class FullScreenPass {
  readonly material: THREE.NodeMaterial;

  constructor(name: string, fragmentNode: THREE.Node) {
    const material = new THREE.NodeMaterial();
    material.name = name;
    material.fragmentNode = fragmentNode;
    material.depthTest = false;
    material.depthWrite = false;
    material.transparent = false;
    this.material = material;
  }

  /**
   * Draw into `target` (or the current target when `null`).
   *
   * @param mipLevel mip level of `target` to render into. The caller must not
   *   have `target`'s texture bound for sampling in the same pass — see the
   *   module header for why {@link MipChain} keeps two targets.
   */
  render(renderer: NodeRenderer, target: THREE.RenderTarget | null, mipLevel = 0): void {
    renderer.setRenderTarget(target, 0, mipLevel);
    sharedQuad.material = this.material;
    sharedQuad.render(renderer);
  }

  dispose(): void {
    this.material.dispose();
  }
}

/** Point or bilinear filtering; the only two modes any buffer here wants. */
export type BufferFilter = typeof THREE.NearestFilter | typeof THREE.LinearFilter;

export interface TargetOptions {
  width?: number;
  height?: number;
  format?: THREE.PixelFormat;
  type?: THREE.TextureDataType;
  filter?: BufferFilter;
  mipmaps?: boolean;
  depthBuffer?: boolean;
}

/** Allocate a colour-only render target with sane post-processing defaults. */
export function createTarget(name: string, options: TargetOptions = {}): THREE.RenderTarget {
  const filter = options.filter ?? THREE.LinearFilter;
  const target = new THREE.RenderTarget(options.width ?? 1, options.height ?? 1, {
    format: options.format ?? THREE.RGBAFormat,
    type: options.type ?? THREE.HalfFloatType,
    minFilter: options.mipmaps === true ? THREE.NearestMipmapNearestFilter : filter,
    magFilter: filter,
    depthBuffer: options.depthBuffer ?? false,
    generateMipmaps: options.mipmaps ?? false,
    // Post-processing data is never colour-managed; these buffers hold
    // radiance, occlusion and geometry, not display-referred colour.
    colorSpace: THREE.NoColorSpace,
  });
  target.texture.name = name;
  target.texture.wrapS = THREE.ClampToEdgeWrapping;
  target.texture.wrapT = THREE.ClampToEdgeWrapping;
  return target;
}

/* ------------------------------------------------------------------------- *
 * Geometry guide buffer
 * ------------------------------------------------------------------------- */

/**
 * The geometry buffer every screen-space effect in this pipeline reads.
 *
 * ### For the integrator
 *
 * If the main scene pass already writes view-space normals and linear depth
 * (for example through `PostStack`'s MRT), register an object of this shape
 * under {@link GuideBufferKey} **before** GTAO or SSR initialise and the
 * standalone prepass is skipped entirely — that is worth roughly 0.3 ms of
 * vertex work per frame at 1080p on the target GPU. The contract is exactly:
 *
 * - `guideTexture` — RGBA16F, `rgb` = view-space unit normal, `a` = linear view
 *   depth divided by `camera.far`, cleared to `(0, 0, 1, 1)` where there is no
 *   geometry.
 * - `halfGuideTexture` — the same, at {@link GuideBufferOptions.resolutionScale}
 *   resolution, produced by *selecting* the nearest of each 2×2 block rather
 *   than averaging (an averaged normal or depth belongs to no real surface and
 *   breaks the bilateral upsample). May be `null`, in which case the effects
 *   sample the full-resolution buffer with point filtering.
 * - Both textures must use `NearestFilter`.
 */
export interface GuideBufferProvider {
  readonly guideTexture: THREE.Texture | null;
  readonly halfGuideTexture: THREE.Texture | null;
  readonly width: number;
  readonly height: number;
  readonly halfWidth: number;
  readonly halfHeight: number;
  /** Bumped whenever the texture *objects* are replaced, so consumers rebind. */
  readonly version: number;
}

/** Service key for {@link GuideBufferProvider}. */
export const GuideBufferKey = serviceKey<GuideBufferProvider>('render.guideBuffer');

export interface GuideBufferOptions {
  /**
   * Resolution of the half-resolution guide relative to the full one. 0.5 is
   * the default and the only value that keeps the 2×2 nearest-depth selection
   * exact; smaller values are supported for a "low" tier but the upsample gets
   * correspondingly softer.
   */
  resolutionScale?: number;
}

/**
 * Renders the guide buffer with a depth+normal prepass over the whole scene.
 *
 * The prepass is not free — it is a second traversal of the scene's geometry —
 * but it buys three things that a same-frame G-buffer read cannot:
 *
 * 1. AO and reflections are computed from *this* frame's geometry, not last
 *    frame's reprojected into it. One-frame-late AO detaches visibly from
 *    fast-moving objects, and it is one of the classic tells.
 * 2. It writes depth, so the main forward pass gets early-Z rejection and pays
 *    part of the prepass back in avoided overdraw.
 * 3. It is independent of whatever MRT layout the rest of the post stack
 *    settles on, so this module can be developed and shipped on its own.
 */
export class GuideBufferPass implements GuideBufferProvider {
  readonly #full: THREE.RenderTarget;
  readonly #half: THREE.RenderTarget;
  readonly #prepassMaterial: THREE.NodeMaterial;
  readonly #downsample: FullScreenPass;
  readonly #sourceTexel = uniform(new THREE.Vector2(1, 1));
  readonly #clearColor = new THREE.Color(0, 0, 1);
  readonly #resolutionScale: number;
  readonly #scope = new RendererStateScope();

  #version = 0;

  constructor(options: GuideBufferOptions = {}) {
    this.#resolutionScale = clamp01(options.resolutionScale ?? 0.5, 0.25, 1);

    this.#full = createTarget('guide.full', {
      type: THREE.HalfFloatType,
      filter: THREE.NearestFilter,
      depthBuffer: true,
    });
    this.#full.depthTexture = new THREE.DepthTexture(1, 1);

    this.#half = createTarget('guide.half', {
      type: THREE.HalfFloatType,
      filter: THREE.NearestFilter,
    });

    this.#prepassMaterial = createGuidePrepassMaterial();
    this.#downsample = new FullScreenPass(
      'guide.downsample',
      nearestDepthDownsampleNode(this.#full.texture, this.#sourceTexel),
    );
  }

  get guideTexture(): THREE.Texture {
    return this.#full.texture;
  }

  get halfGuideTexture(): THREE.Texture {
    return this.#half.texture;
  }

  /** Hardware depth attachment of the prepass, for anyone who wants early-Z. */
  get depthTexture(): THREE.DepthTexture | null {
    return this.#full.depthTexture;
  }

  get width(): number {
    return this.#full.width;
  }

  get height(): number {
    return this.#full.height;
  }

  get halfWidth(): number {
    return this.#half.width;
  }

  get halfHeight(): number {
    return this.#half.height;
  }

  get version(): number {
    return this.#version;
  }

  setSize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (this.#full.width === w && this.#full.height === h) return;

    this.#full.setSize(w, h);
    this.#half.setSize(
      Math.max(1, Math.round(w * this.#resolutionScale)),
      Math.max(1, Math.round(h * this.#resolutionScale)),
    );
    this.#sourceTexel.value.set(1 / w, 1 / h);
    this.#version++;
  }

  /**
   * Draw the prepass and the nearest-depth downsample.
   *
   * Saves and restores the full renderer *and* scene state, so it is safe to
   * call from a module's `lateUpdate` immediately before the engine's own
   * scene render.
   */
  render(renderer: NodeRenderer, scene: THREE.Scene, camera: THREE.Camera): void {
    this.#scope.begin(renderer, scene, this.#clearColor, 1);
    try {
      scene.overrideMaterial = this.#prepassMaterial;

      renderer.setRenderTarget(this.#full);
      renderer.clear();
      renderer.render(scene, camera);

      scene.overrideMaterial = null;
      this.#downsample.render(renderer, this.#half);
    } finally {
      this.#scope.end();
    }
  }

  dispose(): void {
    this.#full.dispose();
    this.#half.dispose();
    this.#prepassMaterial.dispose();
    this.#downsample.dispose();
  }
}

/**
 * `scene.overrideMaterial` for the prepass: view normal in `rgb`, normalised
 * linear view depth in `a`.
 *
 * Skinning, morphing and instancing are applied by `NodeMaterial.setupPosition`
 * from the *object*, not the material, so the Barbarian and any instanced
 * scatter land in the guide buffer at their animated positions without this
 * material knowing anything about them.
 *
 * Geometric (interpolated vertex) normals are used rather than normal-mapped
 * ones on purpose. Ambient occlusion integrates visibility over the hemisphere
 * of the *surface*; feeding it a tangent-space perturbation makes the AO
 * shimmer with the normal map's mip level and double-darkens detail the normal
 * map already shades.
 */
function createGuidePrepassMaterial(): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial();
  material.name = 'guide.prepass';
  // Lazily imported here rather than at module scope to keep the accessor
  // imports next to the one place that needs them.
  material.fragmentNode = guidePrepassFragment();
  return material;
}

/**
 * Fragment graph for {@link createGuidePrepassMaterial}.
 *
 * This is the only place in the module that reaches into three's
 * scene-derived node accessors; everything else is a pure function of the
 * guide buffer, which is what makes the rest of the file testable and
 * backend-independent.
 */
const guidePrepassFragment = Fn(() => {
  // `normalView` is the interpolated, renormalised view-space normal;
  // `positionView.z` is negative in front of the camera.
  const normal = normalize(normalView);
  const viewZ = positionView.z.negate();
  return vec4(normal, viewZ.div(cameraFar));
});

/**
 * Nearest-depth 2×2 downsample.
 *
 * Picks the *closest* of the four source texels and copies its whole guide
 * texel — normal included — rather than averaging. An averaged normal is not a
 * unit vector and an averaged depth sits between two surfaces, so a bilateral
 * filter guided by them matches nothing and degenerates into a box blur across
 * exactly the silhouettes it exists to protect.
 */
function nearestDepthDownsampleNode(
  source: THREE.Texture,
  sourceTexel: THREE.Node<'vec2'>,
): THREE.Node {
  const sourceNode = texture(source);
  return Fn(() => {
    const base = uv();
    // The four full-resolution texels covered by this half-resolution texel.
    const offsets: Array<[number, number]> = [
      [-0.5, -0.5],
      [0.5, -0.5],
      [-0.5, 0.5],
      [0.5, 0.5],
    ];
    const best = sourceNode.sample(base.add(sourceTexel.mul(vec2(offsets[0]![0], offsets[0]![1])))).toVar('best');
    for (let i = 1; i < offsets.length; i++) {
      const offset = offsets[i]!;
      const candidate = sourceNode
        .sample(base.add(sourceTexel.mul(vec2(offset[0], offset[1]))))
        .toVar(`guideTap${i}`);
      If(candidate.w.lessThan(best.w), () => {
        best.assign(candidate);
      });
    }
    return best;
  })();
}

/* ------------------------------------------------------------------------- *
 * Shared guide-buffer ownership
 * ------------------------------------------------------------------------- */

interface GuideRefCount {
  provider: GuideBufferProvider;
  owned: GuideBufferPass | null;
  count: number;
}

const guideRefs = new WeakMap<GameContext, GuideRefCount>();

/**
 * Obtain the guide buffer, creating and registering one if this is the first
 * caller.
 *
 * GTAO and SSR both need it and either may be enabled alone, so ownership is
 * reference counted rather than assigned to one of them. An externally
 * registered {@link GuideBufferProvider} always wins — see the interface docs.
 */
export function acquireGuideBuffer(
  ctx: GameContext,
  options: GuideBufferOptions = {},
): { provider: GuideBufferProvider; owned: GuideBufferPass | null } {
  const existing = guideRefs.get(ctx);
  if (existing !== undefined) {
    existing.count++;
    return { provider: existing.provider, owned: existing.owned };
  }

  const external = ctx.services.tryGet(GuideBufferKey);
  if (external !== undefined) {
    const ref: GuideRefCount = { provider: external, owned: null, count: 1 };
    guideRefs.set(ctx, ref);
    return { provider: external, owned: null };
  }

  const pass = new GuideBufferPass(options);
  ctx.services.register(GuideBufferKey, pass);
  const ref: GuideRefCount = { provider: pass, owned: pass, count: 1 };
  guideRefs.set(ctx, ref);
  return { provider: pass, owned: pass };
}

/** Release a reference taken by {@link acquireGuideBuffer}. */
export function releaseGuideBuffer(ctx: GameContext): void {
  const ref = guideRefs.get(ctx);
  if (ref === undefined) return;
  ref.count--;
  if (ref.count > 0) return;

  guideRefs.delete(ctx);
  if (ref.owned !== null) {
    ctx.services.unregister(GuideBufferKey);
    ref.owned.dispose();
  }
}

/* ------------------------------------------------------------------------- *
 * Mip pyramids
 * ------------------------------------------------------------------------- */

/** How a {@link MipChain} folds a 2×2 block into one texel. */
export type MipReduction = 'min' | 'average';

export interface MipChainOptions {
  /** `min` for a Hi-Z depth pyramid, `average` for a blurred colour pyramid. */
  reduction: MipReduction;
  /** Texture format. Default `RGBAFormat`. */
  format?: THREE.PixelFormat;
  /** Texture type. Default `HalfFloatType`. */
  type?: THREE.TextureDataType;
  /** Hard cap on levels. Default 16 (i.e. whatever the resolution allows). */
  maxLevels?: number;
}

/**
 * A GPU mip pyramid built by repeated 2×2 reduction, safe on both backends.
 *
 * Two mipped targets are kept identical: level `i` of A reduces level `i-1` of
 * B, and level `i` of B reduces level `i-1` of A. Neither pass ever samples the
 * texture it is writing, which is what makes this legal under WebGPU's
 * subresource aliasing rules (three binds sampled textures as a view over the
 * *whole* mip range, so "different mip level" is not enough on its own).
 *
 * The last level of B is skipped: nothing reduces from it.
 */
export class MipChain {
  readonly #a: THREE.RenderTarget;
  readonly #b: THREE.RenderTarget;
  readonly #reduceAtoB: FullScreenPass;
  readonly #reduceBtoA: FullScreenPass;
  readonly #level0: FullScreenPass;
  readonly #sourceLevel = uniform(0);
  readonly #sourceTexel = uniform(new THREE.Vector2(1, 1));
  readonly #maxLevels: number;

  #levels = 1;

  /**
   * @param name  debug name; becomes the texture name
   * @param level0 fragment graph that fills level 0 from whatever source the
   *   owner has. It is rendered with `uv()` spanning the level-0 target.
   */
  constructor(name: string, level0: THREE.Node, options: MipChainOptions) {
    const shared = {
      format: options.format ?? THREE.RGBAFormat,
      type: options.type ?? THREE.HalfFloatType,
      filter: THREE.NearestFilter,
      mipmaps: true,
    } as const;

    this.#maxLevels = options.maxLevels ?? 16;
    this.#a = createTarget(`${name}.a`, shared);
    this.#b = createTarget(`${name}.b`, shared);

    this.#level0 = new FullScreenPass(`${name}.level0`, level0);
    this.#reduceAtoB = new FullScreenPass(
      `${name}.reduce.a`,
      reduceNode(this.#a.texture, this.#sourceLevel, this.#sourceTexel, options.reduction),
    );
    this.#reduceBtoA = new FullScreenPass(
      `${name}.reduce.b`,
      reduceNode(this.#b.texture, this.#sourceLevel, this.#sourceTexel, options.reduction),
    );
  }

  /** The complete pyramid. Sample with an explicit level; filtering is point. */
  get texture(): THREE.Texture {
    return this.#a.texture;
  }

  get levels(): number {
    return this.#levels;
  }

  get width(): number {
    return this.#a.width;
  }

  get height(): number {
    return this.#a.height;
  }

  setSize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (this.#a.width === w && this.#a.height === h) return;
    this.#a.setSize(w, h);
    this.#b.setSize(w, h);
    this.#levels = mipChainLevels(w, h, this.#maxLevels);
  }

  /**
   * Build every level. The caller must already have reset the renderer state.
   */
  build(renderer: NodeRenderer): void {
    // Level 0 is generated twice rather than copied: the source pass is a
    // single texture fetch, and a `copyTextureToTexture` between two render
    // target textures has meaningfully different support across the two
    // backends. Both chains must start identical for the alternation below.
    this.#level0.render(renderer, this.#a, 0);
    this.#level0.render(renderer, this.#b, 0);

    for (let level = 1; level < this.#levels; level++) {
      const sourceWidth = Math.max(1, this.#a.width >> (level - 1));
      const sourceHeight = Math.max(1, this.#a.height >> (level - 1));
      this.#sourceLevel.value = level - 1;
      this.#sourceTexel.value.set(1 / sourceWidth, 1 / sourceHeight);

      // A(level) <- B(level-1), then B(level) <- A(level-1). Neither pass ever
      // samples the texture it is writing, which is the whole point.
      this.#reduceBtoA.render(renderer, this.#a, level);
      if (level < this.#levels - 1) {
        this.#reduceAtoB.render(renderer, this.#b, level);
      }
    }
  }

  dispose(): void {
    this.#a.dispose();
    this.#b.dispose();
    this.#level0.dispose();
    this.#reduceAtoB.dispose();
    this.#reduceBtoA.dispose();
  }
}

/** 2×2 reduction fragment for {@link MipChain}. */
function reduceNode(
  source: THREE.Texture,
  sourceLevel: THREE.Node<'float'>,
  sourceTexel: THREE.Node<'vec2'>,
  reduction: MipReduction,
): THREE.Node {
  return Fn(() => {
    const base = uv();
    const level = sourceLevel;
    const offsets: Array<[number, number]> = [
      [-0.5, -0.5],
      [0.5, -0.5],
      [-0.5, 0.5],
      [0.5, 0.5],
    ];
    const taps = offsets.map((offset, index) =>
      texture(source, base.add(sourceTexel.mul(vec2(offset[0], offset[1]))), level).toVar(
        `mipTap${index}`,
      ),
    );
    const t0 = taps[0]!;
    const t1 = taps[1]!;
    const t2 = taps[2]!;
    const t3 = taps[3]!;
    if (reduction === 'min') {
      return min(min(t0, t1), min(t2, t3));
    }
    return t0.add(t1).add(t2).add(t3).mul(0.25);
  })();
}

/* ------------------------------------------------------------------------- *
 * À-trous denoiser
 * ------------------------------------------------------------------------- */

export interface AtrousDenoiserOptions {
  /** Debug name; becomes the render target and material names. */
  name: string;
  /** Number of à-trous passes. 0 disables the filter entirely. Default 3. */
  iterations?: number;
  /** Kernel radius in taps: 2 = 5×5 B3 spline, 1 = 3×3 tent. Default 2. */
  radius?: number;
  /** Depth tolerance, *relative* to the centre depth. Default 0.05. */
  depthSigma?: number;
  /** Normal tolerance exponent. Default 64. */
  normalSigma?: number;
  /** Texture format for the intermediate targets. Default `RGBAFormat`. */
  format?: THREE.PixelFormat;
  /** Texture type. Default `HalfFloatType`. */
  type?: THREE.TextureDataType;
}

/**
 * Edge-avoiding à-trous wavelet denoiser (Dammertz et al. 2010), shared by
 * GTAO and SSR.
 *
 * Ping-pongs between two targets. Iteration `i` uses a tap spacing of `2^i`, so
 * three iterations of a 5×5 kernel span 29 texels for the cost of 75 taps
 * rather than 841.
 *
 * Two materials are built — even iterations read target A and write B, odd read
 * B and write A — so no `TextureNode.value` is ever mutated between draws. Only
 * the scalar `stepSize` uniform changes, and three re-uploads uniform buffers
 * per `render()` call, each of which is its own command submission.
 */
export class AtrousDenoiser {
  readonly #a: THREE.RenderTarget;
  readonly #b: THREE.RenderTarget;
  readonly #evenPass: FullScreenPass;
  readonly #oddPass: FullScreenPass;
  readonly #stepSize = uniform(1);
  readonly #texel = uniform(new THREE.Vector2(1, 1));
  readonly #iterations: number;

  #output: THREE.RenderTarget;

  /**
   * @param guide  the half-resolution guide buffer the filter edge-stops on.
   * @param farNode `camera.far` as a uniform, for decoding guide depth.
   */
  constructor(
    guide: THREE.Texture,
    farNode: THREE.Node<'float'>,
    options: AtrousDenoiserOptions,
  ) {
    this.#iterations = Math.max(0, Math.floor(options.iterations ?? 3));
    const shared = {
      format: options.format ?? THREE.RGBAFormat,
      type: options.type ?? THREE.HalfFloatType,
      filter: THREE.LinearFilter,
    } as const;

    this.#a = createTarget(`${options.name}.atrous.a`, shared);
    this.#b = createTarget(`${options.name}.atrous.b`, shared);
    this.#output = this.#a;

    const params = {
      radius: Math.max(1, Math.min(2, Math.floor(options.radius ?? 2))),
      depthSigma: uniform(options.depthSigma ?? 0.05),
      normalSigma: uniform(options.normalSigma ?? 64),
    };

    // Iteration 0 reads target A directly: the producing pass renders into
    // `inputTarget`, so the raw signal never needs a copy.
    this.#evenPass = new FullScreenPass(
      `${options.name}.atrous.even`,
      atrousFragment(this.#a.texture, guide, farNode, this.#stepSize, this.#texel, params),
    );
    this.#oddPass = new FullScreenPass(
      `${options.name}.atrous.odd`,
      atrousFragment(this.#b.texture, guide, farNode, this.#stepSize, this.#texel, params),
    );
  }

  /**
   * The target the *producer* of the signal must render into. Iteration 0 reads
   * it, which avoids an otherwise-pointless full-resolution copy.
   */
  get inputTarget(): THREE.RenderTarget {
    return this.#a;
  }

  /** The target holding the filtered result after the last {@link render}. */
  get outputTexture(): THREE.Texture {
    return this.#output.texture;
  }

  get outputTarget(): THREE.RenderTarget {
    return this.#output;
  }

  setSize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (this.#a.width === w && this.#a.height === h) return;
    this.#a.setSize(w, h);
    this.#b.setSize(w, h);
    this.#texel.value.set(1 / w, 1 / h);
  }

  /** Run every iteration. The caller must already have reset renderer state. */
  render(renderer: NodeRenderer): THREE.Texture {
    let source = this.#a;
    for (let i = 0; i < this.#iterations; i++) {
      this.#stepSize.value = atrousStepSize(i);
      const even = source === this.#a;
      const destination = even ? this.#b : this.#a;
      (even ? this.#evenPass : this.#oddPass).render(renderer, destination);
      source = destination;
    }
    this.#output = source;
    return source.texture;
  }

  dispose(): void {
    this.#a.dispose();
    this.#b.dispose();
    this.#evenPass.dispose();
    this.#oddPass.dispose();
  }
}

interface AtrousParams {
  radius: number;
  depthSigma: THREE.Node<'float'>;
  normalSigma: THREE.Node<'float'>;
}

/**
 * One à-trous iteration.
 *
 * The tap loop is unrolled in TypeScript because the kernel weights are
 * compile-time constants; a runtime loop would have to fetch them from a
 * uniform array and would defeat the compiler's ability to fold the 25
 * multiply-adds.
 */
function atrousFragment(
  signal: THREE.Texture,
  guide: THREE.Texture,
  farNode: THREE.Node<'float'>,
  stepSize: THREE.Node<'float'>,
  texel: THREE.Node<'vec2'>,
  params: AtrousParams,
): THREE.Node {
  const signalNode = texture(signal);
  const guideNode = texture(guide);

  return Fn(() => {
    const base = uv();
    const centerGuide = decodeGuide(guideNode.sample(base), farNode).toVar('centerGuide');
    const centerDepth = centerGuide.w.toVar('centerDepth');
    const centerNormal = centerGuide.xyz.toVar('centerNormal');
    const center = signalNode.sample(base).toVar('center');

    const sum = center.mul(atrousTapWeight(0, 0, params.radius)).toVar('sum');
    const weightSum = float(atrousTapWeight(0, 0, params.radius)).toVar('weightSum');

    const stride = texel.mul(stepSize);
    const radius = params.radius;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx === 0 && dy === 0) continue;
        const kernel = atrousTapWeight(dx, dy, radius);
        if (kernel === 0) continue;

        const tapUv = base.add(stride.mul(vec2(dx, dy)));
        const tapGuide = decodeGuide(guideNode.sample(tapUv), farNode);
        const weight = edgeStoppingWeightNode(
          centerDepth,
          tapGuide.w,
          centerNormal.dot(tapGuide.xyz),
          params.depthSigma,
          params.normalSigma,
        ).mul(kernel);

        sum.addAssign(signalNode.sample(tapUv).mul(weight));
        weightSum.addAssign(weight);
      }
    }

    return sum.div(max(weightSum, float(1e-5)));
  })();
}

/* ------------------------------------------------------------------------- *
 * Joint bilateral upsample
 * ------------------------------------------------------------------------- */

export interface BilateralUpsampleParams {
  /** Relative depth tolerance. Default 0.05. */
  depthSigma?: number;
  /** Normal tolerance exponent. Default 32. */
  normalSigma?: number;
}

/**
 * Joint bilateral upsample of a half-resolution signal, evaluated at an
 * arbitrary full-resolution UV (Kopf et al. 2007).
 *
 * Returns the upsampled `vec4`. The four low-resolution taps are weighted by
 * bilinear position and by guide agreement; if every tap disagrees — a
 * one-pixel silhouette where no half-resolution sample belongs to this surface
 * — the result falls back to the bilinear average rather than to zero.
 *
 * @param signal          half-resolution signal texture (linear filtering)
 * @param lowGuide        half-resolution guide texture (nearest filtering)
 * @param lowResolution   `(width, height)` of the half-resolution buffers
 */
export function bilateralUpsampleNode(
  signal: THREE.Texture,
  lowGuide: THREE.Texture,
  lowResolution: THREE.Node<'vec2'>,
  farNode: THREE.Node<'float'>,
  params: BilateralUpsampleParams = {},
): (screenUv: Vec2Node, highDepth: FloatNode, highNormal: Vec3Node) => Vec4Node {
  const signalNode = texture(signal);
  const guideNode = texture(lowGuide);
  const depthSigma = uniform(params.depthSigma ?? 0.05);
  const normalSigma = uniform(params.normalSigma ?? 32);

  const upsample = Fn(
    ([screenUv, highDepth, highNormal]: [Vec2Node, FloatNode, Vec3Node]) => {
      const texel = vec2(1, 1).div(lowResolution);
      // Position of this pixel in low-resolution texel space, offset so that
      // `floor` lands on the top-left of the 2x2 bilinear footprint.
      const coord = screenUv.mul(lowResolution).sub(0.5).toVar('coord');
      const baseTexel = coord.floor().toVar('baseTexel');
      const fractional = coord.sub(baseTexel).toVar('frac');

      const sum = vec4(0).toVar('sum');
      const weightSum = float(0).toVar('weightSum');
      const bilinearSum = vec4(0).toVar('bilinearSum');

      const corners: Array<[number, number]> = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ];
      for (const [ox, oy] of corners) {
        const tapUv = baseTexel.add(vec2(ox + 0.5, oy + 0.5)).mul(texel);
        const bilinear = float(ox === 0 ? 1 : 0)
          .sub(fractional.x.mul(ox === 0 ? 1 : -1))
          .mul(float(oy === 0 ? 1 : 0).sub(fractional.y.mul(oy === 0 ? 1 : -1)));

        const tapGuide = decodeGuide(guideNode.sample(tapUv), farNode);
        const value = signalNode.sample(tapUv);
        const weight = bilinear.mul(
          edgeStoppingWeightNode(
            highDepth,
            tapGuide.w,
            highNormal.dot(tapGuide.xyz),
            depthSigma,
            normalSigma,
          ),
        );

        sum.addAssign(value.mul(weight));
        weightSum.addAssign(weight);
        bilinearSum.addAssign(value.mul(bilinear));
      }

      return weightSum.greaterThan(float(1e-4)).select(sum.div(weightSum), bilinearSum);
    },
  );

  return (screenUv, highDepth, highNormal) => upsample(screenUv, highDepth, highNormal);
}

/* ------------------------------------------------------------------------- *
 * Temporal accumulation
 * ------------------------------------------------------------------------- */

/**
 * The subset of {@link module:render/post/Motion}'s `MotionVectorProvider` that
 * the temporal filters here actually consume.
 *
 * Declared structurally rather than imported so that this module compiles and
 * runs whether or not the motion-vector system is present. Resolve it at
 * runtime under the service id `render.motion`.
 */
export interface MotionVectorSource {
  /** NDC-space `p_now - p_prev`, or `null` when velocity is not rendered. */
  readonly velocityTexture: THREE.Texture | null;
  /** Multiply a sampled velocity by this to get a texture-space offset. */
  readonly ndcToUv: THREE.Vector2;
  /** View matrix used for the previous frame's scene draw. */
  readonly previousViewMatrix: THREE.Matrix4;
  /** Unjittered projection matrix used for the previous frame's scene draw. */
  readonly previousProjectionMatrix: THREE.Matrix4;
  /** False for exactly one frame after a camera cut or resize. */
  readonly historyValid: boolean;
  /** Monotonic count of scene draws. */
  readonly frameIndex: number;
}

/** Service id the motion-vector system publishes itself under. */
export const MOTION_VECTORS_SERVICE_ID = 'render.motion';

/** Resolve the motion-vector provider, or `undefined` when it is absent. */
export function tryGetMotionVectors(ctx: GameContext): MotionVectorSource | undefined {
  return ctx.services.tryGet<MotionVectorSource>(MOTION_VECTORS_SERVICE_ID);
}

export interface TemporalAccumulatorOptions {
  name: string;
  /**
   * How many channels of the input actually carry signal.
   *
   * The per-pixel history length has to live *somewhere*, and it must survive
   * the render target's format. With `1` the layout is `(signal, n)` and an
   * `RG` target is enough — which halves the bandwidth of the AO history, the
   * single busiest buffer in the pipeline. With `3` it is `(signal.rgb, n)` and
   * the target must be `RGBA`. Default 3.
   */
  signalChannels?: 1 | 3;
  /** Steady-state weight of the new sample. Default 0.1 (≈10-frame history). */
  minAlpha?: number;
  /** Neighbourhood clamp width in standard deviations. Default 1.5. */
  clampGamma?: number;
  /** Relative depth mismatch above which history is rejected. Default 0.05. */
  depthRejection?: number;
  format?: THREE.PixelFormat;
  type?: THREE.TextureDataType;
}

/**
 * Exponential-moving-average temporal filter with reprojection, depth
 * rejection and neighbourhood variance clipping.
 *
 * ### Reprojection
 *
 * With a velocity buffer, history is fetched at `uv - v · ndcToUv`, which
 * handles moving objects and skinned characters. Without one, the pixel's view
 * position is reconstructed from the guide buffer, transformed by
 * `P_prev · V_prev · V_now⁻¹` and reprojected — correct for camera motion,
 * which is the dominant term, and wrong for moving objects, which then rely on
 * the neighbourhood clamp to avoid ghosting. The degraded path is *not* a
 * placeholder: for a mostly static outdoor scene it is very nearly as good, and
 * it is what keeps this module independent of the motion-vector system.
 *
 * ### Why the history stores frame count
 *
 * Alpha is `max(minAlpha, 1/(n+1))`. Storing `n` per pixel is what lets a
 * freshly disoccluded pixel converge in three frames instead of thirty while
 * long-lived pixels keep the full temporal denoise. The counter lives in the
 * history target's alpha channel, which the signal does not use.
 */
export class TemporalAccumulator {
  readonly #history: [THREE.RenderTarget, THREE.RenderTarget];
  readonly #passes: [FullScreenPass, FullScreenPass];
  readonly #reset = uniform(1);
  readonly #reprojection = uniform(new THREE.Matrix4());
  readonly #useVelocity = uniform(0);

  #index = 0;

  constructor(
    current: THREE.Texture,
    guide: THREE.Texture,
    velocity: THREE.Texture | null,
    uniforms: TemporalUniforms,
    options: TemporalAccumulatorOptions,
  ) {
    const shared = {
      format: options.format ?? THREE.RGBAFormat,
      type: options.type ?? THREE.HalfFloatType,
      filter: THREE.LinearFilter,
    } as const;

    this.#history = [
      createTarget(`${options.name}.history.0`, shared),
      createTarget(`${options.name}.history.1`, shared),
    ];

    const params: TemporalParams = {
      minAlpha: uniform(options.minAlpha ?? 0.1),
      clampGamma: uniform(options.clampGamma ?? 1.5),
      depthRejection: uniform(options.depthRejection ?? 0.05),
      reset: this.#reset,
      reprojection: this.#reprojection,
      useVelocity: this.#useVelocity,
    };

    const channels = options.signalChannels ?? 3;
    this.#passes = [
      new FullScreenPass(
        `${options.name}.temporal.0`,
        temporalFragment(
          current,
          this.#history[1].texture,
          guide,
          velocity,
          uniforms,
          params,
          channels,
        ),
      ),
      new FullScreenPass(
        `${options.name}.temporal.1`,
        temporalFragment(
          current,
          this.#history[0].texture,
          guide,
          velocity,
          uniforms,
          params,
          channels,
        ),
      ),
    ];
  }

  /** The accumulated result of the most recent {@link render}. */
  get outputTexture(): THREE.Texture {
    return this.#history[this.#index]!.texture;
  }

  setSize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    for (const target of this.#history) {
      if (target.width !== w || target.height !== h) target.setSize(w, h);
    }
  }

  /** Force the next frame to discard history (resize, camera cut, tier swap). */
  reset(): void {
    this.#reset.value = 1;
  }

  /**
   * Accumulate this frame's signal into the history.
   *
   * @param reprojection `P_prev · V_prev · V_now⁻¹`, used when no velocity
   *   buffer is available. Ignored otherwise.
   * @param useVelocity whether the velocity texture is live this frame.
   */
  render(
    renderer: NodeRenderer,
    reprojection: THREE.Matrix4,
    useVelocity: boolean,
    invalidate: boolean,
  ): THREE.Texture {
    this.#reprojection.value.copy(reprojection);
    this.#useVelocity.value = useVelocity ? 1 : 0;
    if (invalidate) this.#reset.value = 1;

    this.#index = 1 - this.#index;
    this.#passes[this.#index]!.render(renderer, this.#history[this.#index]!);
    this.#reset.value = 0;
    return this.#history[this.#index]!.texture;
  }

  dispose(): void {
    for (const target of this.#history) target.dispose();
    for (const pass of this.#passes) pass.dispose();
  }
}

/** Camera/​resolution uniforms shared by the temporal pass and its callers. */
export interface TemporalUniforms {
  /** `(P[0][0], P[1][1])` of the current projection matrix. */
  readonly projScale: THREE.Node<'vec2'>;
  /** `camera.far`. */
  readonly cameraFar: THREE.Node<'float'>;
  /** Resolution of the accumulated buffer, in texels. */
  readonly resolution: THREE.Node<'vec2'>;
  /** NDC-to-UV scale for the velocity buffer, `(0.5, -0.5)`. */
  readonly ndcToUv: THREE.Node<'vec2'>;
}

interface TemporalParams {
  minAlpha: THREE.Node<'float'>;
  clampGamma: THREE.Node<'float'>;
  depthRejection: THREE.Node<'float'>;
  reset: THREE.Node<'float'>;
  reprojection: THREE.Node<'mat4'>;
  useVelocity: THREE.Node<'float'>;
}

function temporalFragment(
  current: THREE.Texture,
  history: THREE.Texture,
  guide: THREE.Texture,
  velocity: THREE.Texture | null,
  uniforms: TemporalUniforms,
  params: TemporalParams,
  channels: 1 | 3,
): THREE.Node {
  const currentNode = texture(current);
  const historyNode = texture(history);
  const guideNode = texture(guide);
  const velocityNode = velocity === null ? null : texture(velocity);

  /** The four cross neighbours used for the neighbourhood statistics. */
  const neighbours: Array<[number, number]> = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  const inverseCount = 1 / (neighbours.length + 1);

  return Fn(() => {
    const base = uv();
    const centerGuide = decodeGuide(guideNode.sample(base), uniforms.cameraFar).toVar('guide');
    const centerDepth = centerGuide.w.toVar('depth');

    /* -- reprojection -------------------------------------------------- */

    const historyUv = base.toVar('historyUv');
    if (velocityNode !== null) {
      // NDC delta -> texture-space offset. The Y component of `ndcToUv` is
      // negative because NDC is Y-up and render-target texture space is Y-down.
      const motion = velocityNode.sample(base).xy.mul(uniforms.ndcToUv);
      historyUv.assign(base.sub(motion));
    }
    // Camera-only fallback: reconstruct the view position and push it through
    // `P_prev · V_prev · V_now⁻¹`.
    const viewPosition = viewPositionFromDepth(base, centerDepth, uniforms.projScale);
    const previousClip = params.reprojection.mul(vec4(viewPosition, 1));
    const previousNdc = previousClip.xy.div(max(previousClip.w.abs(), float(1e-6)));
    const cameraUv = vec2(
      previousNdc.x.mul(0.5).add(0.5),
      float(0.5).sub(previousNdc.y.mul(0.5)),
    );
    historyUv.assign(params.useVelocity.greaterThan(float(0.5)).select(historyUv, cameraUv));

    /* -- validity ------------------------------------------------------ */

    const inside = historyUv
      .greaterThanEqual(vec2(0))
      .all()
      .and(historyUv.lessThanEqual(vec2(1)).all());
    // The guide buffer is from *this* frame, so this compares the reprojected
    // pixel's depth against the depth of whatever now occupies that texel. A
    // mismatch means the surface was occluded or revealed: reject.
    const historyDepth = decodeGuide(guideNode.sample(historyUv), uniforms.cameraFar).w;
    const depthOk = centerDepth
      .sub(historyDepth)
      .abs()
      .lessThan(max(centerDepth, float(1e-3)).mul(params.depthRejection));
    const valid = inside.and(depthOk).and(params.reset.lessThan(float(0.5))).toVar('valid');

    const texel = vec2(1, 1).div(uniforms.resolution);
    const historySample = historyNode.sample(historyUv).toVar('history');

    if (channels === 1) {
      const value = currentNode.sample(base).x.toVar('value');
      const moment1 = value.toVar('moment1');
      const moment2 = value.mul(value).toVar('moment2');
      for (const [dx, dy] of neighbours) {
        const tap = currentNode.sample(base.add(texel.mul(vec2(dx, dy)))).x;
        moment1.addAssign(tap);
        moment2.addAssign(tap.mul(tap));
      }
      const mean = moment1.mul(inverseCount);
      const spread = max(moment2.mul(inverseCount).sub(mean.mul(mean)), float(0))
        .sqrt()
        .mul(params.clampGamma);

      const length = valid
        .select(min(historySample.y.add(1), float(64)), float(0))
        .toVar('n');
      const alpha = max(params.minAlpha, float(1).div(length.add(1)));
      const clamped = historySample.x.clamp(mean.sub(spread), mean.add(spread));
      const blended = valid.select(clamped.mul(alpha.oneMinus()).add(value.mul(alpha)), value);
      return vec4(blended, length, 0, 0);
    }

    const value = currentNode.sample(base).xyz.toVar('value');
    const moment1 = value.toVar('moment1');
    const moment2 = value.mul(value).toVar('moment2');
    for (const [dx, dy] of neighbours) {
      const tap = currentNode.sample(base.add(texel.mul(vec2(dx, dy)))).xyz;
      moment1.addAssign(tap);
      moment2.addAssign(tap.mul(tap));
    }
    const mean = moment1.mul(inverseCount);
    const spread = max(moment2.mul(inverseCount).sub(mean.mul(mean)), vec3(0))
      .sqrt()
      .mul(params.clampGamma);

    const length = valid.select(min(historySample.w.add(1), float(64)), float(0)).toVar('n');
    const alpha = max(params.minAlpha, float(1).div(length.add(1)));
    const clamped = historySample.xyz.clamp(mean.sub(spread), mean.add(spread));
    const blended = valid.select(clamped.mul(alpha.oneMinus()).add(value.mul(alpha)), value);
    return vec4(blended, length);
  })();
}

/* ------------------------------------------------------------------------- *
 * Small helpers
 * ------------------------------------------------------------------------- */

function clamp01(value: number, low = 0, high = 1): number {
  return Math.min(high, Math.max(low, value));
}
