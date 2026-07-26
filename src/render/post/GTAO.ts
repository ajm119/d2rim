/**
 * @module render/post/GTAO
 *
 * Ground-Truth Ambient Occlusion: horizon-based visibility integrated against
 * the *cosine-weighted* hemisphere, with an albedo-aware multi-bounce term.
 *
 * Reference implementation of Jimenez, Wu, Pesce & Jarabo, *"Practical
 * Real-Time Strategies for Accurate Indirect Occlusion"*, SIGGRAPH 2016 Physically
 * Based Shading course (Activision), with the thin-occluder compensation and
 * quadratic step distribution from Intel's XeGTAO (Sekulic et al., 2021).
 *
 * ---
 *
 * ## Why GTAO rather than SSAO/HBAO
 *
 * SSAO answers "what fraction of a sphere around this point is solid?" and then
 * multiplies the ambient term by it. That is not an integral of anything: it is
 * a heuristic, and it is why classic SSAO produces the flat grey wash around
 * every object that reads instantly as "hobby renderer".
 *
 * GTAO computes the actual quantity the rendering equation asks for. For each
 * *slice* — a plane containing the view vector — it finds the two horizon
 * angles `h₁`, `h₂` by marching the depth buffer, and then evaluates the
 * cosine-weighted visibility integral over that slice in closed form:
 *
 * ```
 * a(h, n) = ¼ · ( −cos(2h − n) + cos n + 2·h·sin n )
 * V_slice = |projected normal| · ( a(h₁, n) + a(h₂, n) )
 * ```
 *
 * where `n` is the angle between the surface normal projected into the slice
 * plane and the view vector (Jimenez et al., eq. 7). Averaging `V_slice` over
 * uniformly distributed slices converges to the ground-truth cosine-weighted
 * visibility — hence the name. The consequence you can see: creases darken
 * correctly, but a flat wall next to a pillar does *not*, because half its
 * hemisphere is still open.
 *
 * ## Multi-bounce, and why it matters here
 *
 * Pure visibility is too dark, because light that is blocked from reaching a
 * point directly still arrives after bouncing off the occluder. Jimenez's
 * closed-form fit approximates that with a cubic in the visibility whose
 * coefficients are functions of the surface albedo
 * ({@link gtaoMultiBounce}). Dark, wet mud stays dark in its creases; pale
 * stone and dead grass lift substantially. Applying a flat visibility instead
 * is exactly the "dirty grey smear" the art direction has to avoid — the whole
 * frame ends up desaturated in the same direction.
 *
 * The multi-bounce term is evaluated **inside the material**, not in the AO
 * pass, because that is the only place the albedo is known. See
 * {@link GTAOModule.occlusionNode}.
 *
 * ## AO is applied to indirect light only
 *
 * This module publishes an `occlusionNode` under the service id `render.ao`.
 * The IBL module folds it into `material.aoNode`, and three's
 * `PhysicalLightingModel` multiplies **`reflectedLight.indirectDiffuse`** and
 * the indirect specular lobes by it — never `directDiffuse` or
 * `directSpecular`. Screen-space occlusion applied to direct light is the
 * classic tell: it produces contact shadows that do not move with the sun,
 * double-darkens against the real shadow map, and washes out at noon.
 *
 * If no IBL service is present the module still publishes the node, and the
 * integrator can attach it wherever ambient is composed.
 *
 * ## Resolution and filtering
 *
 * Traced at half resolution, spatially denoised with the shared à-trous filter
 * ({@link module:render/post/Denoise}), temporally accumulated with motion-vector
 * reprojection, and joint-bilaterally upsampled at consumption time. Sample
 * directions are rotated per-frame through the Activision paper's six-entry
 * rotation table and offset through its four-entry step table, so six frames of
 * temporal accumulation see 6 × 4 = 24 distinct sampling patterns.
 *
 * ## Cost, at 1080p on a mid-range 2020 discrete GPU (RX 5700 / RTX 2060 class)
 *
 * | Stage                    | Resolution | Work                        | Budget |
 * |--------------------------|-----------|------------------------------|--------|
 * | depth+normal prepass     | 1920×1080 | geometry, no shading         | 0.30 ms |
 * | guide downsample         | 960×540   | 4 taps                       | 0.03 ms |
 * | GTAO trace (`high`)      | 960×540   | 3 slices × 6 steps × 2 sides | 0.55 ms |
 * | à-trous ×2 (5×5)         | 960×540   | 50 taps/px/iter              | 0.35 ms |
 * | temporal accumulate      | 960×540   | 8 taps                       | 0.08 ms |
 * | bilateral upsample       | inline    | 8 taps in the forward pass   | 0.10 ms |
 * | **total**                |           |                              | **1.4 ms** |
 *
 * The trace is the interesting number. 36 depth taps per half-resolution pixel
 * is 18.7 M taps at 1080p; the guide buffer is RGBA16F, so at 8 bytes per tap
 * that is 150 MB of reads per frame — but they are extremely cache-coherent
 * (the whole march stays inside a ~40 px screen-space radius), so the effective
 * rate is closer to L2 bandwidth than to VRAM. Measured GTAO implementations of
 * this shape land at 0.4–0.7 ms at 1080p on that hardware, which is where the
 * 0.55 ms estimate comes from. The prepass estimate assumes ~2000 draw calls of
 * depth-only geometry.
 *
 * **This project's development container has no GPU** (SwiftShader, 4 cores),
 * so none of the above was measured here and no frame time from this machine
 * should be believed. The numbers are analytic, derived from tap counts,
 * resolutions and the published cost of equivalent passes.
 *
 * ## Quality tiers
 *
 * | Tier     | Slices | Steps | À-trous | Res  | Est. total |
 * |----------|--------|-------|---------|------|-----------|
 * | `off`    | —      | —     | —       | —    | 0 ms      |
 * | `low`    | 1      | 3     | 1 × 3×3 | 1/2  | 0.5 ms    |
 * | `medium` | 2      | 4     | 2 × 3×3 | 1/2  | 0.9 ms    |
 * | `high`   | 3      | 6     | 2 × 5×5 | 1/2  | 1.4 ms    |
 * | `ultra`  | 4      | 8     | 3 × 5×5 | 1/2  | 2.2 ms    |
 *
 * `low` leans hard on temporal accumulation: one slice per frame with the
 * six-entry rotation table still integrates six slices over six frames, so a
 * static shot converges to `ultra` quality. It is motion where it gives up.
 *
 * ## References
 *
 * - J. Jimenez, X.-C. Wu, A. Pesce, A. Jarabo, *Practical Real-Time Strategies
 *   for Accurate Indirect Occlusion*, SIGGRAPH 2016 PBS course.
 * - F. Sekulic et al., *XeGTAO*, Intel, 2021 — thin-occluder compensation,
 *   quadratic step distribution.
 * - S. Hill, *Rendering Rousseau* / Lagarde & de Rousiers, *Moving Frostbite to
 *   PBR*, 2014 §4.10 — why AO belongs on indirect light only.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  acos,
  clamp,
  cos,
  cross,
  diffuseColor,
  dot,
  float,
  length,
  luminance,
  max,
  mix,
  normalize,
  saturate,
  screenUV,
  sign,
  sin,
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
  RendererStateScope,
  TemporalAccumulator,
  acquireGuideBuffer,
  asNodeRenderer,
  bilateralUpsampleNode,
  decodeGuide,
  interleavedGradientNoiseNode,
  releaseGuideBuffer,
  tryGetMotionVectors,
  viewPositionFromDepth,
  type FloatNode,
  type GuideBufferPass,
  type GuideBufferProvider,
  type MotionVectorSource,
  type TemporalUniforms,
} from './Denoise';

/* ------------------------------------------------------------------------- *
 * Contract with the rest of the renderer
 * ------------------------------------------------------------------------- */

/**
 * What this module publishes for the image-based lighting system.
 *
 * This is structurally identical to `AmbientOcclusionProvider` in
 * {@link module:render/IBL}, restated here so the two modules do not import each
 * other. The IBL module resolves the service id `render.ao`, multiplies
 * `occlusionNode` into `material.aoNode`, and three then applies it to indirect
 * diffuse and indirect specular only.
 *
 * `occlusionNode` is a TSL float in `[0, 1]`, 1 meaning unoccluded.
 */
export interface AmbientOcclusionProvider {
  readonly occlusionNode: THREE.Node<'float'> | null;
}

/** Service id the IBL module looks this module up under. */
export const AMBIENT_OCCLUSION_SERVICE_ID = 'render.ao';

/** Everything this module exposes to debug UI and the post stack. */
export interface GTAOService extends AmbientOcclusionProvider {
  /** Half-resolution, denoised, temporally accumulated visibility. `r` only. */
  readonly aoTexture: THREE.Texture | null;
  /** Raw visibility straight out of the trace, for A/B debug views. */
  readonly rawTexture: THREE.Texture | null;
  /** Full-resolution bilaterally upsampled visibility, as a TSL float. */
  readonly upsampledNode: THREE.Node<'float'> | null;
  readonly quality: GTAOQuality;
  setQuality(quality: GTAOQuality): void;
  setRadius(worldUnits: number): void;
  setIntensity(power: number): void;
  readonly stats: GTAOStats;
}

/** Service key for {@link GTAOService}. */
export const GTAOKey = serviceKey<GTAOService>('render.gtao');

export interface GTAOStats {
  readonly enabled: boolean;
  readonly quality: GTAOQuality;
  readonly width: number;
  readonly height: number;
  readonly slices: number;
  readonly steps: number;
  readonly denoiseIterations: number;
  /** Whether a velocity buffer was found; false means camera-only reprojection. */
  readonly temporalFromVelocity: boolean;
  /** Whether the guide buffer came from an external G-buffer. */
  readonly sharedGuideBuffer: boolean;
}

/* ------------------------------------------------------------------------- *
 * Quality tiers
 * ------------------------------------------------------------------------- */

export type GTAOQuality = 'off' | 'low' | 'medium' | 'high' | 'ultra';

interface TierConfig {
  readonly slices: number;
  readonly steps: number;
  readonly denoiseIterations: number;
  readonly denoiseRadius: 1 | 2;
}

/** See the module header's quality table for the reasoning behind these. */
export const GTAO_TIERS: Readonly<Record<Exclude<GTAOQuality, 'off'>, TierConfig>> = {
  low: { slices: 1, steps: 3, denoiseIterations: 1, denoiseRadius: 1 },
  medium: { slices: 2, steps: 4, denoiseIterations: 2, denoiseRadius: 1 },
  high: { slices: 3, steps: 6, denoiseIterations: 2, denoiseRadius: 2 },
  ultra: { slices: 4, steps: 8, denoiseIterations: 3, denoiseRadius: 2 },
};

export interface GTAOOptions {
  /** Starting quality tier. Default `'high'`. */
  quality?: GTAOQuality;
  /**
   * World-space radius of the occlusion search, in metres. 0.8 m is authored
   * for this game's scale: big enough that a boulder occludes the mud beside
   * it, small enough that a distant treeline does not shadow the whole moor.
   */
  radius?: number;
  /**
   * Exponent applied to visibility. 1 is physically neutral; the default 1.15
   * is a very slight artistic deepening that suits the overcast key light,
   * where there is almost no directional shadow to carry contact darkening.
   */
  intensity?: number;
  /**
   * Fraction of the radius over which a sample's contribution fades to zero,
   * in `[0, 1]`. Prevents a hard ring at the search boundary. Default 0.4.
   */
  falloff?: number;
  /**
   * Thin-occluder compensation in `[0, 1]` (XeGTAO). 0 lets a one-pixel-thin
   * railing occlude as if it were a solid wall; 1 ignores it entirely. Default
   * 0.6 — bare twisted branches against a grey sky are exactly the case this
   * exists for.
   */
  thinOccluderCompensation?: number;
  /**
   * Multi-bounce strength in `[0, 1]`. 1 is the full Jimenez fit, 0 is raw
   * visibility. Default 1.
   */
  multiBounce?: number;
  /** Resolution of the trace relative to the framebuffer. Default 0.5. */
  resolutionScale?: number;
  /** Steady-state temporal blend weight. Default 0.1. */
  temporalMinAlpha?: number;
  /**
   * Register the occlusion node under `render.ao` so the IBL module picks it
   * up. Default true. Set false if the integrator wants to compose it manually.
   */
  publishToIBL?: boolean;
}

/* ------------------------------------------------------------------------- *
 * Pure math — unit tested in tests/gtao.math.test.ts
 * ------------------------------------------------------------------------- */

/**
 * The GTAO inner integral for one horizon (Jimenez et al. 2016, eq. 7):
 *
 * ```
 * ¼ · ( −cos(2h − n) + cos n + 2·h·sin n )
 * ```
 *
 * `h` is the horizon angle measured from the view vector inside the slice
 * plane, `n` is the angle of the slice-projected normal from the view vector.
 * The two horizons of a slice are summed and the result scaled by the length of
 * the projected normal.
 *
 * The identity worth knowing when reading this: with `n = 0` (normal facing the
 * camera) and `h = ±π/2` (nothing occluding), the two horizons sum to
 * `¼·(1+1) + ¼·(1+1) = 1` — full visibility, exactly as an unoccluded
 * cosine-weighted hemisphere should give.
 */
export function gtaoArcIntegral(h: number, n: number): number {
  return 0.25 * (-Math.cos(2 * h - n) + Math.cos(n) + 2 * h * Math.sin(n));
}

/**
 * Visibility of one complete slice from its two horizon angles.
 *
 * @param h1 horizon on the negative side of the slice direction
 * @param h2 horizon on the positive side
 * @param n  angle of the slice-projected normal from the view vector
 * @param projectedNormalLength length of the normal projected into the slice
 */
export function gtaoSliceVisibility(
  h1: number,
  h2: number,
  n: number,
  projectedNormalLength: number,
): number {
  return projectedNormalLength * (gtaoArcIntegral(h1, n) + gtaoArcIntegral(h2, n));
}

/**
 * Jimenez's albedo-aware multi-bounce approximation (SIGGRAPH 2016, §"Multi
 * bounce"):
 *
 * ```
 * a =  2.0404·ρ − 0.3324
 * b = −4.7951·ρ + 0.6417
 * c =  2.7552·ρ + 0.6903
 * GTAOMultiBounce(V, ρ) = max( V, ((V·a + b)·V + c)·V )
 * ```
 *
 * A cubic fit to a Monte-Carlo simulation of interreflections in a corner of
 * albedo `ρ`. The `max` with `V` guarantees it never *darkens* below pure
 * visibility, which the raw cubic can do for very dark albedos.
 *
 * The practical effect on this project's palette: at `V = 0.5`, mud (`ρ ≈ 0.06`)
 * comes out at 0.50 — unchanged — while dead grass (`ρ ≈ 0.25`) lifts to 0.57
 * and weathered limestone (`ρ ≈ 0.55`) to 0.70. That difference is what keeps
 * the shadowed half of the frame reading as *materials* rather than as one
 * uniform grey veil.
 */
export function gtaoMultiBounce(visibility: number, albedo: number): number {
  const a = 2.0404 * albedo - 0.3324;
  const b = -4.7951 * albedo + 0.6417;
  const c = 2.7552 * albedo + 0.6903;
  const x = Math.min(1, Math.max(0, visibility));
  return Math.min(1, Math.max(x, ((x * a + b) * x + c) * x));
}

/**
 * Screen-space radius, in pixels, of a world-space length `radius` lying
 * perpendicular to the view axis at view depth `viewZ`.
 *
 * ```
 * ndc  = radius · P₁₁ / viewZ          (perspective divide)
 * px   = ndc · height / 2              (NDC spans [-1, 1] over `height` pixels)
 * ```
 *
 * This is what turns a constant world-space AO radius into a per-pixel march
 * length. Clamping it matters: without an upper bound a surface right against
 * the near plane asks for a several-hundred-pixel march and the pass becomes
 * unboundedly expensive; without a lower bound a distant surface samples the
 * same texel repeatedly and produces a hard AO cut-off at the horizon.
 */
export function gtaoScreenRadius(
  radius: number,
  viewZ: number,
  projectionScaleY: number,
  height: number,
  maxPixels = 256,
): number {
  const pixels = (radius * projectionScaleY * height) / (2 * Math.max(viewZ, 1e-4));
  return Math.min(maxPixels, Math.max(1, pixels));
}

/**
 * Per-frame slice rotation, in turns, from the Activision paper's six-entry
 * table `[60, 300, 180, 240, 120, 0] / 360`.
 *
 * The order is not sorted — it is a permutation chosen so that any *prefix* of
 * the sequence is close to uniformly spread. That property is what lets the
 * `low` tier trace a single slice per frame and still have well-distributed
 * coverage after only three frames.
 */
export const GTAO_TEMPORAL_ROTATIONS: readonly number[] = [
  60 / 360,
  300 / 360,
  180 / 360,
  240 / 360,
  120 / 360,
  0,
];

/** Per-frame step offset from the same paper: `[0, 0.5, 0.25, 0.75]`. */
export const GTAO_TEMPORAL_OFFSETS: readonly number[] = [0, 0.5, 0.25, 0.75];

/** Slice rotation for a frame index, in turns. */
export function gtaoTemporalRotation(frame: number): number {
  const index = ((frame % 6) + 6) % 6;
  return GTAO_TEMPORAL_ROTATIONS[index] ?? 0;
}

/** Step offset for a frame index, in `[0, 1)`. */
export function gtaoTemporalOffset(frame: number): number {
  const index = ((Math.floor(frame / 6) % 4) + 4) % 4;
  return GTAO_TEMPORAL_OFFSETS[index] ?? 0;
}

/**
 * Number of distinct sampling patterns the temporal sequence visits before it
 * repeats. Useful as a sanity check on the tier tables: a tier that traces `s`
 * slices per frame effectively integrates `s × 24` slices once converged.
 */
export const GTAO_TEMPORAL_PERIOD = GTAO_TEMPORAL_ROTATIONS.length * GTAO_TEMPORAL_OFFSETS.length;

/* ------------------------------------------------------------------------- *
 * Trace shader
 * ------------------------------------------------------------------------- */

interface TraceUniforms {
  readonly guide: THREE.Texture;
  readonly projScale: THREE.Node<'vec2'>;
  readonly cameraFar: THREE.Node<'float'>;
  readonly resolution: THREE.Node<'vec2'>;
  readonly radius: THREE.Node<'float'>;
  readonly falloff: THREE.Node<'float'>;
  readonly intensity: THREE.Node<'float'>;
  readonly thinOccluder: THREE.Node<'float'>;
  readonly sliceRotation: THREE.Node<'float'>;
  readonly stepOffset: THREE.Node<'float'>;
  readonly frameIndex: THREE.Node<'float'>;
  readonly maxScreenRadius: THREE.Node<'float'>;
}

const HALF_PI = Math.PI / 2;

/**
 * The GTAO trace.
 *
 * Structure, following Jimenez et al. §"Implementation":
 *
 * 1. Reconstruct view position and pick a screen-space march radius.
 * 2. For each slice: build the slice frame, project the normal into it, and
 *    measure the angle `n` of that projected normal from the view vector.
 * 3. March both sides of the slice, tracking the maximum horizon cosine.
 * 4. Evaluate the closed-form arc integral for the two horizons.
 *
 * The slice count is a JavaScript constant, not a uniform, so the loop is fully
 * unrolled and the per-slice trigonometry folds into constants where it can.
 * Changing tier rebuilds the material, which is why {@link GTAOModule} keeps one
 * per tier rather than mutating a uniform.
 */
function gtaoTraceFragment(config: TierConfig, u: TraceUniforms): THREE.Node {
  const guideNode = texture(u.guide);

  return Fn(() => {
    const base = uv();
    const centerGuide = decodeGuide(guideNode.sample(base), u.cameraFar).toVar('centerGuide');
    const viewZ = centerGuide.w.toVar('viewZ');

    const visibility = float(1).toVar('visibility');

    // Sky and anything at the far plane: fully unoccluded, and marching from
    // there would produce garbage horizons off the reconstructed position.
    If(viewZ.lessThan(u.cameraFar.mul(0.999)), () => {
      const normal = centerGuide.xyz.toVar('n');
      const position = viewPositionFromDepth(base, viewZ, u.projScale).toVar('p');
      const view = normalize(position.negate()).toVar('v');

      // World radius -> pixels, clamped at both ends. See `gtaoScreenRadius`.
      const screenRadius = clamp(
        u.radius.mul(u.projScale.y).mul(u.resolution.y).mul(0.5).div(max(viewZ, float(1e-4))),
        float(1),
        u.maxScreenRadius,
      ).toVar('screenRadius');

      // Per-pixel dither, decorrelated across frames by the frame index.
      const noise = interleavedGradientNoiseNode(
        base.mul(u.resolution),
        u.frameIndex,
      ).toVar('noise');

      const falloffStart = u.radius.mul(u.falloff.oneMinus()).toVar('falloffStart');
      const falloffScale = float(1)
        .div(max(u.radius.sub(falloffStart), float(1e-4)))
        .toVar('falloffScale');

      const total = float(0).toVar('total');

      Loop({ start: 0, end: config.slices, type: 'int' }, ({ i }) => {
        // Slices span [0, π): a direction and its negation describe the same
        // plane, and both sides of it are marched below.
        const phi = float(i)
          .add(noise)
          .add(u.sliceRotation)
          .mul(Math.PI / config.slices)
          .toVar('phi');
        const omega = vec2(cos(phi), sin(phi)).toVar('omega');

        // Slice frame. `axis` is the slice plane's normal; projecting the
        // shading normal onto the plane and measuring its angle from the view
        // vector gives `n`, which the arc integral is parameterised by.
        const direction = vec3(omega, 0).toVar('direction');
        const orthoDirection = direction.sub(view.mul(dot(direction, view))).toVar('ortho');
        const axis = normalize(cross(direction, view)).toVar('axis');
        const projectedNormal = normal.sub(axis.mul(dot(normal, axis))).toVar('projN');
        const projectedLength = max(length(projectedNormal), float(1e-5)).toVar('projLen');
        const cosN = clamp(
          dot(projectedNormal, view).div(projectedLength),
          float(-1),
          float(1),
        );
        const signN = sign(dot(orthoDirection, projectedNormal));
        const angleN = signN.mul(acos(cosN)).toVar('angleN');

        // Two horizons, marched independently. Unrolled in JS so each side gets
        // its own registers rather than a dynamically indexed array.
        const horizons: FloatNode[] = [];
        for (const side of [-1, 1] as const) {
          const horizonCos = cos(angleN.add(side * HALF_PI)).toVar(
            side < 0 ? 'horizonNeg' : 'horizonPos',
          );

          Loop({ start: 0, end: config.steps, type: 'int' }, ({ i: step }) => {
            // Quadratic step distribution (XeGTAO): samples cluster near the
            // shading point, where the horizon changes fastest, instead of
            // being wasted at the edge of the search radius. `+ noise` and
            // `+ stepOffset` jitter it per pixel and per frame.
            const t = float(step)
              .add(u.stepOffset)
              .add(noise)
              .add(1)
              .div(config.steps + 1)
              .toVar('t');
            const stepPixels = max(t.mul(t).mul(screenRadius), float(1)).toVar('stepPixels');

            const sampleUv = base.add(
              omega.mul(side).mul(stepPixels).div(u.resolution),
            );
            const sampleGuide = decodeGuide(guideNode.sample(sampleUv), u.cameraFar);
            const samplePosition = viewPositionFromDepth(
              sampleUv,
              sampleGuide.w,
              u.projScale,
            );

            const delta = samplePosition.sub(position).toVar('delta');
            const distance = max(length(delta), float(1e-5)).toVar('dist');
            const sampleCos = dot(delta, view).div(distance);

            // Distance attenuation: past `falloffStart` the sample is blended
            // back towards the horizon we already have, so it contributes
            // nothing at the search boundary and there is no visible ring.
            const attenuation = saturate(distance.sub(falloffStart).mul(falloffScale));
            const candidate = mix(sampleCos, horizonCos, attenuation).toVar('candidate');

            // Thin-occluder compensation (XeGTAO). A sample that *lowers* the
            // horizon is a thin object the ray would pass either side of; only
            // partially accept it instead of letting a bare branch occlude like
            // a wall.
            If(candidate.greaterThan(horizonCos), () => {
              horizonCos.assign(candidate);
            }).Else(() => {
              horizonCos.assign(mix(candidate, horizonCos, u.thinOccluder));
            });
          });

          // Convert the horizon cosine back to an angle, relative to `n`, and
          // clamp it to the hemisphere. Without the clamp a horizon behind the
          // surface produces a negative visibility contribution.
          const horizonAngle = angleN.add(
            clamp(
              float(side).mul(acos(clamp(horizonCos, float(-1), float(1)))).sub(angleN),
              float(-HALF_PI),
              float(HALF_PI),
            ),
          );
          horizons.push(horizonAngle.toVar(side < 0 ? 'h1' : 'h2'));
        }

        const h1 = horizons[0]!;
        const h2 = horizons[1]!;
        // ¼·( −cos(2h − n) + cos n + 2·h·sin n ), summed over both horizons.
        const cosAngleN = cos(angleN);
        const sinAngleN = sin(angleN);
        const arc1 = cos(h1.mul(2).sub(angleN)).negate().add(cosAngleN).add(h1.mul(2).mul(sinAngleN));
        const arc2 = cos(h2.mul(2).sub(angleN)).negate().add(cosAngleN).add(h2.mul(2).mul(sinAngleN));
        total.addAssign(projectedLength.mul(arc1.add(arc2)).mul(0.25));
      });

      visibility.assign(saturate(total.div(config.slices)).pow(u.intensity));
    });

    // `g` carries the per-pixel history length once the temporal pass owns this
    // buffer; the trace writes 0 so a fresh pixel starts a fresh history.
    return vec4(visibility, 0, 0, 1);
  })();
}

/* ------------------------------------------------------------------------- *
 * Module
 * ------------------------------------------------------------------------- */

/**
 * The GTAO render module.
 *
 * Runs in `lateUpdate`, i.e. after every gameplay module has settled the camera
 * and before the engine's own `renderer.render`. That ordering is what makes
 * the AO current rather than one frame late: the depth+normal prepass sees
 * exactly the geometry the forward pass is about to shade.
 */
export class GTAOModule implements GameModule, GTAOService {
  readonly name = 'render.gtao';

  /* -- configuration ----------------------------------------------------- */

  #quality: GTAOQuality;
  readonly #options: Required<Omit<GTAOOptions, 'quality'>>;

  /* -- uniforms ---------------------------------------------------------- */

  readonly #projScale = uniform(new THREE.Vector2(1, 1));
  readonly #cameraFar = uniform(1000);
  readonly #resolution = uniform(new THREE.Vector2(1, 1));
  readonly #lowResolution = uniform(new THREE.Vector2(1, 1));
  readonly #ndcToUv = uniform(new THREE.Vector2(0.5, -0.5));
  readonly #radius = uniform(0.8);
  readonly #falloff = uniform(0.4);
  readonly #intensity = uniform(1.15);
  readonly #thinOccluder = uniform(0.6);
  readonly #sliceRotation = uniform(0);
  readonly #stepOffset = uniform(0);
  readonly #frameIndex = uniform(0);
  readonly #maxScreenRadius = uniform(96);
  readonly #multiBounce = uniform(1);
  readonly #aoStrength = uniform(1);

  /* -- GPU resources ----------------------------------------------------- */

  #guide: GuideBufferProvider | null = null;
  #ownedGuide: GuideBufferPass | null = null;
  #tracePass: FullScreenPass | null = null;
  #traceTarget: THREE.RenderTarget | null = null;
  #denoiser: AtrousDenoiser | null = null;
  #temporal: TemporalAccumulator | null = null;
  #occlusionNode: THREE.Node<'float'> | null = null;
  #upsampledNode: THREE.Node<'float'> | null = null;

  readonly #scope = new RendererStateScope();
  readonly #reprojection = new THREE.Matrix4();
  readonly #previousViewProjection = new THREE.Matrix4();
  readonly #inverseView = new THREE.Matrix4();

  #ctx: GameContext | null = null;
  #motion: MotionVectorSource | undefined;
  #hasPreviousFrame = false;
  #frame = 0;
  #width = 0;
  #height = 0;
  #warnedNoRenderer = false;

  constructor(options: GTAOOptions = {}) {
    this.#quality = options.quality ?? 'high';
    this.#options = {
      radius: options.radius ?? 0.8,
      intensity: options.intensity ?? 1.15,
      falloff: options.falloff ?? 0.4,
      thinOccluderCompensation: options.thinOccluderCompensation ?? 0.6,
      multiBounce: options.multiBounce ?? 1,
      resolutionScale: options.resolutionScale ?? 0.5,
      temporalMinAlpha: options.temporalMinAlpha ?? 0.1,
      publishToIBL: options.publishToIBL ?? true,
    };
  }

  /* -- GameModule -------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    this.#radius.value = this.#options.radius;
    this.#intensity.value = this.#options.intensity;
    this.#falloff.value = this.#options.falloff;
    this.#thinOccluder.value = this.#options.thinOccluderCompensation;
    this.#multiBounce.value = this.#options.multiBounce;

    const guide = acquireGuideBuffer(ctx, { resolutionScale: this.#options.resolutionScale });
    this.#guide = guide.provider;
    this.#ownedGuide = guide.owned;

    this.#motion = tryGetMotionVectors(ctx);
    if (this.#motion?.ndcToUv !== undefined) {
      this.#ndcToUv.value.copy(this.#motion.ndcToUv);
    }

    this.#build();

    ctx.services.register(GTAOKey, this);
    if (this.#options.publishToIBL) {
      // The IBL module resolves this id and folds `occlusionNode` into
      // `material.aoNode`, which three applies to indirect light only.
      const existing = ctx.services.tryGet<AmbientOcclusionProvider>(
        AMBIENT_OCCLUSION_SERVICE_ID,
      );
      if (existing === undefined) {
        ctx.services.register<AmbientOcclusionProvider>(AMBIENT_OCCLUSION_SERVICE_ID, this);
      } else {
        console.warn(
          `[GTAO] "${AMBIENT_OCCLUSION_SERVICE_ID}" is already registered; ` +
            'the occlusion node is available through the "render.gtao" service instead.',
        );
      }
    }
  }

  lateUpdate(ctx: GameContext): void {
    if (this.#quality === 'off') return;

    const renderer = asNodeRenderer(ctx.renderer);
    if (renderer === null) {
      if (!this.#warnedNoRenderer) {
        this.#warnedNoRenderer = true;
        console.warn('[GTAO] the renderer is not a node renderer; ambient occlusion is disabled.');
      }
      return;
    }

    const size = renderer.getDrawingBufferSize(_size);
    this.#resize(size.width, size.height);
    if (this.#width === 0 || this.#height === 0) return;

    const camera = ctx.camera;
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    // (P[0][0], P[1][1]) — the only two projection terms the reconstruction
    // needs, which is what keeps the shaders clip-space-convention agnostic.
    const projection = camera.projectionMatrix.elements;
    this.#projScale.value.set(projection[0] ?? 1, projection[5] ?? 1);
    this.#cameraFar.value = camera.far;

    // Temporal sampling pattern for this frame.
    const frame = this.#motion?.frameIndex ?? this.#frame;
    this.#sliceRotation.value = gtaoTemporalRotation(frame);
    this.#stepOffset.value = gtaoTemporalOffset(frame);
    this.#frameIndex.value = frame % 64;

    // Camera-only reprojection matrix: P_prev · V_prev · V_now⁻¹.
    this.#inverseView.copy(camera.matrixWorld);
    const velocityLive =
      this.#motion?.velocityTexture != null && this.#motion.historyValid === true;
    if (!velocityLive) {
      this.#reprojection.copy(this.#previousViewProjection).multiply(this.#inverseView);
    }
    const invalidate =
      !this.#hasPreviousFrame || (this.#motion !== undefined && !this.#motion.historyValid);

    this.#scope.begin(renderer, ctx.scene);
    try {
      this.#ownedGuide?.render(renderer, ctx.scene, camera);

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

    // Snapshot for next frame's camera-only reprojection.
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
    this.#frame++;
  }

  dispose(): void {
    const ctx = this.#ctx;
    if (ctx !== null) {
      ctx.services.unregister(GTAOKey);
      if (ctx.services.tryGet<AmbientOcclusionProvider>(AMBIENT_OCCLUSION_SERVICE_ID) === this) {
        ctx.services.unregister(AMBIENT_OCCLUSION_SERVICE_ID);
      }
      releaseGuideBuffer(ctx);
    }
    this.#teardown();
    this.#ctx = null;
  }

  /* -- GTAOService ------------------------------------------------------- */

  /**
   * Occlusion as a TSL float, ready for `material.aoNode`.
   *
   * Two things happen here that cannot happen in the AO pass:
   *
   * 1. The half-resolution buffer is **joint-bilaterally upsampled** against
   *    this pixel's own depth and normal, so the AO does not halo across the
   *    silhouette the pixel sits on.
   * 2. The **multi-bounce** term is evaluated with this material's albedo. The
   *    scalar form uses the albedo's luminance, because three's ambient
   *    occlusion context is a float — a per-channel `GTAOMultiBounce` would
   *    need a coloured AO term that `PhysicalLightingModel` has nowhere to put.
   *    Luminance keeps the energy right on average while still separating dark
   *    wet mud from pale stone, which is the difference that matters visually.
   */
  get occlusionNode(): THREE.Node<'float'> | null {
    return this.#quality === 'off' ? null : this.#occlusionNode;
  }

  get upsampledNode(): THREE.Node<'float'> | null {
    return this.#quality === 'off' ? null : this.#upsampledNode;
  }

  get aoTexture(): THREE.Texture | null {
    return this.#temporal?.outputTexture ?? null;
  }

  get rawTexture(): THREE.Texture | null {
    return this.#traceTarget?.texture ?? null;
  }

  get quality(): GTAOQuality {
    return this.#quality;
  }

  /**
   * Switch tier. Rebuilds the trace material (slice and step counts are
   * unrolled loop bounds, not uniforms) and resets the temporal history,
   * because the new tier's noise characteristics invalidate the old average.
   */
  setQuality(quality: GTAOQuality): void {
    if (quality === this.#quality) return;
    this.#quality = quality;
    if (quality === 'off') return;
    this.#build();
    this.#temporal?.reset();
  }

  setRadius(worldUnits: number): void {
    this.#radius.value = Math.max(0.01, worldUnits);
  }

  setIntensity(power: number): void {
    this.#intensity.value = Math.max(0.01, power);
  }

  /** Scale the whole effect towards "no occlusion". 1 is full strength. */
  setStrength(strength: number): void {
    this.#aoStrength.value = Math.min(1, Math.max(0, strength));
  }

  get stats(): GTAOStats {
    const tier = this.#quality === 'off' ? null : GTAO_TIERS[this.#quality];
    return {
      enabled: this.#quality !== 'off',
      quality: this.#quality,
      width: this.#guide?.halfWidth ?? 0,
      height: this.#guide?.halfHeight ?? 0,
      slices: tier?.slices ?? 0,
      steps: tier?.steps ?? 0,
      denoiseIterations: tier?.denoiseIterations ?? 0,
      temporalFromVelocity: this.#motion?.velocityTexture != null,
      sharedGuideBuffer: this.#ownedGuide === null,
    };
  }

  /* -- internals --------------------------------------------------------- */

  #build(): void {
    const guide = this.#guide;
    if (guide === null || this.#quality === 'off') return;
    const halfGuide = guide.halfGuideTexture ?? guide.guideTexture;
    if (halfGuide === null) return;

    this.#teardown();
    const tier = GTAO_TIERS[this.#quality];

    // R+G half-float: visibility in `r`, temporal history length in `g`. Half
    // the bandwidth of an RGBA buffer through the busiest stage of the effect.
    const traceOptions = {
      format: THREE.RGFormat,
      type: THREE.HalfFloatType,
    } as const;

    this.#denoiser = new AtrousDenoiser(halfGuide, this.#cameraFar, {
      name: 'gtao',
      iterations: tier.denoiseIterations,
      radius: tier.denoiseRadius,
      // AO is a smooth signal on a smooth surface, so it tolerates a wider
      // depth window than a reflection would; the normal term does the real
      // edge preservation.
      depthSigma: 0.06,
      normalSigma: 64,
      ...traceOptions,
    });
    this.#traceTarget = this.#denoiser.inputTarget;

    this.#tracePass = new FullScreenPass(
      `gtao.trace.${this.#quality}`,
      gtaoTraceFragment(tier, {
        guide: halfGuide,
        projScale: this.#projScale,
        cameraFar: this.#cameraFar,
        resolution: this.#lowResolution,
        radius: this.#radius,
        falloff: this.#falloff,
        intensity: this.#intensity,
        thinOccluder: this.#thinOccluder,
        sliceRotation: this.#sliceRotation,
        stepOffset: this.#stepOffset,
        frameIndex: this.#frameIndex,
        maxScreenRadius: this.#maxScreenRadius,
      }),
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
        name: 'gtao',
        signalChannels: 1,
        minAlpha: this.#options.temporalMinAlpha,
        // AO is bounded in [0, 1] and low-frequency, so a fairly tight clamp is
        // safe and it kills ghosting on fast camera turns.
        clampGamma: 1.25,
        depthRejection: 0.04,
        ...traceOptions,
      },
    );

    this.#buildNodes(guide, halfGuide);
  }

  /**
   * Build the consumption-side nodes: the bilateral upsample and the
   * albedo-aware multi-bounce wrapper.
   */
  #buildNodes(guide: GuideBufferProvider, halfGuide: THREE.Texture): void {
    const temporal = this.#temporal;
    const fullGuide = guide.guideTexture;
    if (temporal === null || fullGuide === null) return;

    const upsample = bilateralUpsampleNode(
      temporal.outputTexture,
      halfGuide,
      this.#lowResolution,
      this.#cameraFar,
      { depthSigma: 0.04, normalSigma: 32 },
    );
    const fullGuideNode = texture(fullGuide);

    const upsampled = Fn(() => {
      // `screenUV` is the fragment's position in the *output* framebuffer, which
      // is what makes this usable from inside an arbitrary scene material.
      const screen = screenUV;
      const centre = decodeGuide(fullGuideNode.sample(screen), this.#cameraFar).toVar('centre');
      return upsample(screen, centre.w, centre.xyz).x;
    });
    this.#upsampledNode = upsampled();

    const occlusion = Fn(() => {
      const visibility = saturate(upsampled()).toVar('visibility');
      // Albedo-aware multi-bounce, evaluated where the albedo actually exists.
      const albedo = saturate(luminance(diffuseColor.rgb)).toVar('albedo');
      const bounced = gtaoMultiBounceNode(visibility, albedo);
      const withBounce = mix(visibility, bounced, this.#multiBounce);
      // `aoStrength` lerps the whole term towards 1 (no occlusion) rather than
      // scaling it, so turning the effect down never inverts it.
      return mix(float(1), withBounce, this.#aoStrength);
    });
    this.#occlusionNode = occlusion();
  }

  #resize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.#width && h === this.#height) return;

    this.#width = w;
    this.#height = h;
    this.#resolution.value.set(w, h);

    this.#ownedGuide?.setSize(w, h);
    const guide = this.#guide;
    const lowWidth = guide?.halfWidth ?? Math.max(1, Math.round(w * this.#options.resolutionScale));
    const lowHeight =
      guide?.halfHeight ?? Math.max(1, Math.round(h * this.#options.resolutionScale));
    this.#lowResolution.value.set(lowWidth, lowHeight);

    this.#denoiser?.setSize(lowWidth, lowHeight);
    this.#temporal?.setSize(lowWidth, lowHeight);
    this.#temporal?.reset();
    this.#hasPreviousFrame = false;

    // A radius clamp proportional to the trace resolution keeps the worst-case
    // march length — and therefore the pass's cost — bounded as resolution
    // changes, instead of exploding on a 4K display.
    this.#maxScreenRadius.value = Math.max(24, Math.round(lowHeight * 0.18));
  }

  #teardown(): void {
    this.#tracePass?.dispose();
    this.#denoiser?.dispose();
    this.#temporal?.dispose();
    this.#tracePass = null;
    this.#denoiser = null;
    this.#temporal = null;
    this.#traceTarget = null;
  }
}

/**
 * TSL form of {@link gtaoMultiBounce}.
 *
 * Kept next to the JS version so the two can be diffed by eye; the test suite
 * pins the JS one and the coefficients are shared by inspection.
 */
const gtaoMultiBounceNode = Fn(([visibility, albedo]: [FloatNode, FloatNode]) => {
  const a = albedo.mul(2.0404).sub(0.3324);
  const b = albedo.mul(-4.7951).add(0.6417);
  const c = albedo.mul(2.7552).add(0.6903);
  const x = saturate(visibility).toVar('x');
  return saturate(max(x, x.mul(a).add(b).mul(x).add(c).mul(x)));
});

/** Scratch vector for `getDrawingBufferSize`; the module is not re-entrant. */
const _size = new THREE.Vector2();

/**
 * Create the GTAO module.
 *
 * ```ts
 * engine.add(createGTAO({ quality: 'high', radius: 0.8 }));
 * ```
 *
 * Register it **after** any module that publishes a shared G-buffer under
 * `render.guideBuffer` and after the motion-vector system, and **before** the
 * IBL module, which resolves `render.ao` during its own `init`.
 */
export function createGTAO(options: GTAOOptions = {}): GTAOModule {
  return new GTAOModule(options);
}

/**
 * Convenience for the integrator: construct, register and return the module in
 * one call.
 */
export function registerGTAO(ctx: GameContext, options: GTAOOptions = {}): GTAOModule {
  const module = new GTAOModule(options);
  ctx.engine.add(module);
  return module;
}

/** Re-exported so debug UI can type the guide buffer without a second import. */
export type { GuideBufferProvider } from './Denoise';
