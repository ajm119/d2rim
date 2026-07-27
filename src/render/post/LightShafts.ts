/**
 * @module render/post/LightShafts
 *
 * The screen-space half of the volumetric system: it evaluates the participating
 * media that {@link module:render/Volumetrics} describes, filters it in time,
 * optionally stretches it along the sun's screen-space rays, and composites the
 * result over the frame.
 *
 * `Volumetrics` owns the *physics* — the froxel volume, the media parameters,
 * the phase function, the light list, the fog volumes. This module owns the
 * *pixels*: resolution, temporal stability, upsampling, and the one line that
 * actually joins fog to image,
 *
 * ```
 *   final = surface · T  +  L_inscattered
 * ```
 *
 * ---
 *
 * ## Why half resolution
 *
 * Volumetric scattering is, by construction, the lowest-frequency signal in the
 * frame: it is an integral along a ray through a smooth medium, and the only
 * high frequencies in it come from shadow-map edges, which are themselves
 * soft. Evaluating it per pixel is therefore paying full price for information
 * that is not there. The march runs at half resolution — a quarter of the
 * pixels — and is brought back up with a **joint bilateral upsample** (Kopf et
 * al., *"Joint Bilateral Upsampling"*, SIGGRAPH 2007) keyed on the geometry
 * buffer's depth and normal, so the one place where the signal *does* have a
 * high frequency — the silhouette of a foreground object against distant fog —
 * stays crisp instead of haloing.
 *
 * The same buffer feeds the temporal filter and the radial pass, so the choice
 * pays for itself three times.
 *
 * ## Temporal filtering
 *
 * Even at 32 march steps the shadowed part of the integral is noisy, because
 * each step is a point sample of a binary visibility function. Rather than
 * raise the step count, the ray offset is dithered per pixel and rotated per
 * frame, and consecutive frames are accumulated with camera reprojection. Eight
 * frames of history at `α = 0.12` is worth roughly `8×` the sample count for
 * `1×` the cost.
 *
 * Reprojection is camera-only: the fog is a property of the *air*, which does
 * not move with the geometry, so a velocity buffer would actually reproject it
 * wrongly — a wall sliding across the screen does not drag the fog in front of
 * it. What does need handling is disocclusion, and that is done by rejecting
 * history whose reprojected depth disagrees with the current pixel's, plus
 * raising the blend where transmittance changes sharply.
 *
 * ## Radial light shafts
 *
 * The volumetric integration already produces true god rays wherever the
 * shadow map reaches — that is the whole point of sampling visibility along the
 * ray. But the shadow cascades end at 100-ish metres, and the froxel volume
 * ends sooner, so the *long* shafts through a cave mouth or a distant treeline
 * are cut off exactly where they would be most dramatic.
 *
 * The optional radial pass extends them, following Mitchell, *"Volumetric Light
 * Scattering as a Post-Process"* (GPU Gems 3, chapter 13): from each pixel,
 * step towards the sun's screen position accumulating the scattering buffer
 * with exponential decay. Applied to the *scattering* buffer only — never to
 * the scene colour — so it can only stretch light that the physical model
 * already put there. It cannot invent a shaft where there is no fog, and it
 * cannot produce the streaked-highlight look that gives cheap post-process god
 * rays away.
 *
 * It is off when the sun is behind the camera and fades in over the last 20° of
 * the frustum, because a radial blur centred behind the viewer produces
 * shafts pointing the wrong way. Intensity defaults to a deliberately modest
 * `0.22`: the brief for this project is restraint, and this is the control that
 * most easily ruins it.
 *
 * ## Where this runs
 *
 * Two pieces, because they belong in different parts of the frame:
 *
 * - {@link LightShaftsModule} is a `GameModule`. In `lateUpdate` — after the
 *   camera has settled, before the engine's `render` — it renders the
 *   half-resolution scattering buffer, the temporal blend and the radial pass.
 * - {@link LightShaftsPass} is a `PostPass`. It runs inside the post chain,
 *   where the scene colour actually exists, and does the upsample and the
 *   composite.
 *
 * The module exposes the pass through {@link LightShaftsService.pass}; see
 * "For the integrator" below.
 *
 * ## For the integrator
 *
 * `PostStack` currently builds a fixed pass list. To run the composite, splice
 * the pass in **before** the composite/tonemap pass and after TAA:
 *
 * ```ts
 * const shafts = registerLightShafts(ctx);
 * // inside PostStack's pass-list construction:
 * this.#passes = [this.taa, shafts.pass, this.bloom, this.composite, this.fxaa];
 * ```
 *
 * Order matters in one direction only: the fog must be composited *before*
 * bloom, or the in-scattered light will not bloom, and a shaft that does not
 * bloom at its bright end reads as a decal. It must be after TAA because the
 * fog is already temporally filtered with its own, gentler history and does not
 * want to be clamped by TAA's neighbourhood test.
 *
 * If the pass is never added, the module still runs and the service still
 * reports correctly; the frame simply has no fog composited into it. Nothing
 * throws and nothing leaks.
 *
 * ## References
 *
 * - K. Mitchell, *"Volumetric Light Scattering as a Post-Process"*, GPU Gems 3
 *   ch. 13, 2007 — the radial pass.
 * - J. Kopf, M. Cohen, D. Lischinski & M. Uyttendaele, *"Joint Bilateral
 *   Upsampling"*, SIGGRAPH 2007 — the upsample.
 * - S. Hillaire, *"Physically Based and Unified Volumetric Rendering in
 *   Frostbite"*, SIGGRAPH 2015 — half-resolution evaluation with bilateral
 *   upsample as the shipping configuration.
 * - B. Karis, *"High Quality Temporal Supersampling"*, SIGGRAPH 2014 — the
 *   reprojection scheme the temporal pass follows, minus the neighbourhood
 *   clamp, which is wrong for a signal this smooth.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  float,
  max,
  mix,
  saturate,
  texture as textureNode,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import { serviceKey } from '../../core/ServiceLocator';
import type { GameContext, GameModule } from '../../core/types';
import {
  VolumetricSunKey,
  VolumetricsKey,
  type VolumetricSunProvider,
  type VolumetricsService,
} from '../Volumetrics';
import {
  FullScreenPass,
  RendererStateScope,
  acquireGuideBuffer,
  asNodeRenderer,
  bilateralUpsampleNode,
  createTarget,
  decodeGuide,
  interleavedGradientNoiseNode,
  releaseGuideBuffer,
  tryGetMotionVectors,
  viewPositionFromDepth,
  type FloatNode,
  type GuideBufferPass,
  type GuideBufferProvider,
  type MotionVectorSource,
  type NodeRenderer,
  type Vec2Node,
} from './Denoise';
import type { PostCapabilities, PostFrame, PostPass, QualityTier } from './PostStack';

declare module '../../core/EventBus' {
  interface GameEvents {
    /** Emitted once the buffers exist and {@link LightShaftsKey} is registered. */
    'lightShafts:ready': { halfWidth: number; halfHeight: number; radial: boolean };
  }
}

/* ------------------------------------------------------------------------- *
 * Public vocabulary
 * ------------------------------------------------------------------------- */

export interface LightShaftsOptions {
  /**
   * Fraction of the geometry buffer's resolution the scattering march runs at.
   * `0.5` is the shipping value and the only one at which the bilateral
   * upsample's 2×2 footprint is exact. Clamped to `[0.25, 1]`.
   */
  resolutionScale?: number;
  /**
   * Steady-state weight of the new frame in the temporal blend, in `(0, 1]`.
   * `0.12` is about an 8-frame history — long enough to clean up the march,
   * short enough that a torch being lit does not take half a second to appear.
   */
  temporalBlend?: number;
  /**
   * Relative depth mismatch above which reprojected history is rejected
   * outright. `0.06` accepts a 6% depth change, which covers normal camera
   * motion and rejects a genuine disocclusion.
   */
  depthRejection?: number;
  /** Master multiplier on the composited in-scatter. `1` is physical. */
  intensity?: number;
  /** Enable the radial (Mitchell) shaft extension. Default `true`. */
  radialShafts?: boolean;
  /**
   * Strength of the radial extension. This is the single easiest control to
   * overdo; `0.22` is a deliberate, restrained default and `0.6` already looks
   * like a stock lens effect.
   */
  radialIntensity?: number;
  /** Taps along each radial ray. Overridden downwards by the quality tier. */
  radialSamples?: number;
  /**
   * Fraction of the distance to the sun covered by the radial march, in
   * `(0, 1]`. `0.5` keeps the shafts anchored near their source instead of
   * smearing the whole screen.
   */
  radialLength?: number;
  /** Per-tap attenuation of the radial march. */
  radialDecay?: number;
  /** Register under {@link LightShaftsKey}. Default `true`. */
  registerService?: boolean;
}

export interface LightShaftsStats {
  readonly enabled: boolean;
  readonly width: number;
  readonly height: number;
  /** Whether the radial extension ran on the last frame. */
  readonly radial: boolean;
  /** `0` when the sun is behind the camera, `1` when it is well inside frame. */
  readonly sunOnScreen: number;
  /** Whether temporal history was accepted on the last frame. */
  readonly temporal: boolean;
  /** Approximate GPU bytes held by this module's targets. */
  readonly bytes: number;
  /** Whether the geometry buffer is shared with GTAO/SSR rather than owned. */
  readonly sharedGuideBuffer: boolean;
}

export interface LightShaftsService {
  /**
   * The composite pass, for splicing into the post chain. Its `enabled` flag
   * is the artist toggle; disabling it leaves the module's buffers allocated
   * so that re-enabling is instant.
   */
  readonly pass: PostPass;
  readonly stats: LightShaftsStats;
  /** Master multiplier on the composited in-scatter. */
  setIntensity(intensity: number): void;
  setRadialIntensity(intensity: number): void;
  setEnabled(enabled: boolean): void;
  readonly enabled: boolean;
  /**
   * The half-resolution scattering buffer, for a debug view.
   * `rgb` is in-scattered radiance, `a` is transmittance.
   */
  readonly scatteringTexture: THREE.Texture | null;
}

/** Service key for {@link LightShaftsService}. */
export const LightShaftsKey = serviceKey<LightShaftsService>('render.lightShafts');

/** Radial tap counts per tier. `low` disables the extension entirely. */
const RADIAL_TAPS: Readonly<Record<QualityTier, number>> = {
  low: 0,
  medium: 8,
  high: 12,
  ultra: 16,
};

/* ------------------------------------------------------------------------- *
 * The composite pass
 * ------------------------------------------------------------------------- */

/**
 * Upsample the scattering buffer and composite it over the scene.
 *
 * A `chain` pass: it consumes the running colour texture and produces a new
 * one, in HDR. It owns no buffer that has to survive between frames — the
 * scattering history belongs to {@link LightShaftsModule} — so it costs exactly
 * one pooled target, and zero when it happens to be last in the chain.
 */
export class LightShaftsPass implements PostPass {
  readonly id = 'lightshafts';
  readonly kind = 'chain' as const;
  readonly outputDomain = 'hdr' as const;
  enabled = true;

  readonly #scene = makeSourceNode('lightShafts.scene');
  readonly #intensity = uniform(1);
  readonly #frame = uniform(0);
  readonly #cameraFar = uniform(1000);
  readonly #lowResolution = uniform(new THREE.Vector2(1, 1));
  readonly #resolution = uniform(new THREE.Vector2(1, 1));

  #pass: FullScreenPass | null = null;
  readonly #copy = makeSourceNode('lightShafts.copy');
  readonly #copyPass = new FullScreenPass(
    'lightShafts.passthrough',
    Fn(() => this.#copy.sample(uv()))(),
  );
  #guide: GuideBufferProvider | null = null;
  #scattering: THREE.Texture | null = null;
  #signature = '';

  /* -- PostPass ---------------------------------------------------------- */

  isAvailable(_quality: QualityTier, _capabilities: PostCapabilities): boolean {
    // Fog is not a tier-gated flourish in this project; it is the art
    // direction. What the tier changes is how many samples it costs, which is
    // decided in `LightShaftsModule`, not here.
    return true;
  }

  configure(_quality: QualityTier, _capabilities: PostCapabilities): void {
    // The composite's graph does not depend on the tier: the upsample footprint
    // and the composite are the same at every quality level.
  }

  setSize(width: number, height: number): void {
    this.#resolution.value.set(Math.max(1, width), Math.max(1, height));
  }

  render(frame: PostFrame): THREE.Texture | null | void {
    const pass = this.#pass;
    if (pass === null || this.#scattering === null) {
      // Nothing to composite yet: the module has not produced a buffer, which
      // is the state of the world for exactly one frame after a resize. Copy
      // the colour through rather than returning it, because this pass may be
      // the last one in the chain, in which case whatever it fails to write is
      // what reaches the screen.
      this.#copy.value = frame.input;
      this.#copyPass.render(frame.renderer as NodeRenderer, frame.output);
      return null;
    }
    this.#scene.value = frame.input;
    this.#frame.value = frame.frameIndex % 64;
    this.#cameraFar.value = frame.camera.far;
    pass.render(frame.renderer as NodeRenderer, frame.output);
    return null;
  }

  dispose(): void {
    this.#pass?.dispose();
    this.#pass = null;
    this.#copyPass.dispose();
  }

  /* -- driven by LightShaftsModule --------------------------------------- */

  setIntensity(intensity: number): void {
    this.#intensity.value = Math.max(0, intensity);
  }

  /**
   * (Re)build the composite graph.
   *
   * The graph binds specific texture *objects*, so it must be rebuilt whenever
   * the module replaces one — which happens on a resize and when the guide
   * buffer's owner changes. The signature check makes that idempotent, so
   * calling this every frame is free.
   */
  bind(
    guide: GuideBufferProvider,
    scattering: THREE.Texture,
    lowWidth: number,
    lowHeight: number,
  ): void {
    this.#lowResolution.value.set(Math.max(1, lowWidth), Math.max(1, lowHeight));
    const guideTexture = guide.guideTexture;
    const lowGuide = guide.halfGuideTexture ?? guideTexture;
    if (guideTexture === null || lowGuide === null) return;

    const signature = `${guide.version}:${scattering.uuid}:${guideTexture.uuid}:${lowGuide.uuid}`;
    if (signature === this.#signature && this.#pass !== null) return;
    this.#signature = signature;
    this.#guide = guide;
    this.#scattering = scattering;

    this.#pass?.dispose();
    this.#pass = new FullScreenPass(
      'lightShafts.composite',
      this.#buildComposite(guideTexture, lowGuide, scattering),
    );
  }

  #buildComposite(
    guideTexture: THREE.Texture,
    lowGuide: THREE.Texture,
    scattering: THREE.Texture,
  ): THREE.Node {
    const guideNode = textureNode(guideTexture);
    const upsample = bilateralUpsampleNode(
      scattering,
      lowGuide,
      this.#lowResolution,
      this.#cameraFar,
      // Fog tolerates a much looser depth test than AO does: neighbouring
      // depths that differ by a few percent still carry almost the same
      // integral, and tightening this only starves the filter of taps.
      { depthSigma: 0.12, normalSigma: 4 },
    );

    return Fn(() => {
      const base = uv();
      const guide = decodeGuide(guideNode.sample(base), this.#cameraFar).toVar('guide');
      const highDepth = guide.w.toVar('viewZ');
      const highNormal = guide.xyz.toVar('normal');

      const resolved = upsample(base, highDepth, highNormal).toVar('fog');
      const inscatter = max(resolved.xyz.mul(this.#intensity), vec3(0, 0, 0)).toVar('inscatter');
      const transmittance = saturate(resolved.w).toVar('transmittance');

      const scene = this.#scene.sample(base).toVar('scene');

      // The compositing equation, and the reason the transmittance has to be
      // carried through every stage above rather than being recomputed here:
      // `T` is the product of every slice's transmittance along the ray, and
      // there is no closed form for it once local fog volumes and 3D noise are
      // in play.
      const composited = scene.xyz.mul(transmittance).add(inscatter).toVar('composited');

      // Final ordered dither. The upsample and the composite are both linear
      // operators, so the smooth gradient that entered them is still smooth —
      // and still one quantisation step away from banding once the tonemapper
      // reduces it to 8 bits. Triangular PDF, and rotated per frame so the
      // pattern averages out rather than sitting still on screen.
      const noiseA = interleavedGradientNoiseNode(base.mul(this.#resolution), this.#frame);
      const noiseB = interleavedGradientNoiseNode(
        base.mul(this.#resolution).add(vec2(23.0, 71.0)),
        this.#frame.add(7),
      );
      const dither = noiseA.sub(noiseB).mul(COMPOSITE_DITHER);

      return vec4(max(composited.mul(dither.add(1)), vec3(0, 0, 0)), scene.w);
    })();
  }

  /** For diagnostics; `null` until {@link bind} has run. */
  get guide(): GuideBufferProvider | null {
    return this.#guide;
  }
}

/** ±0.3% of the local value: sub-LSB after the tone curve, and dither is cheap. */
const COMPOSITE_DITHER = 0.003;

/* ------------------------------------------------------------------------- *
 * The module
 * ------------------------------------------------------------------------- */

const _drawingBuffer = new THREE.Vector2();
const _sunWorld = new THREE.Vector3();
const _cameraForward = new THREE.Vector3();
const _cameraPosition = new THREE.Vector3();

/**
 * Owns the half-resolution scattering buffer, its temporal history and the
 * radial extension, and drives {@link LightShaftsPass}.
 */
export class LightShaftsModule implements GameModule, LightShaftsService {
  readonly name = 'render.lightShafts';

  readonly pass = new LightShaftsPass();

  readonly #options: Required<Omit<LightShaftsOptions, 'registerService'>>;
  readonly #registerService: boolean;

  #ctx: GameContext | null = null;
  #volumetrics: VolumetricsService | undefined;
  #sun: VolumetricSunProvider | undefined;
  #motion: MotionVectorSource | undefined;
  #guide: GuideBufferProvider | null = null;
  #ownedGuide: GuideBufferPass | null = null;

  /* -- uniforms ---------------------------------------------------------- */

  readonly #projScale = uniform(new THREE.Vector2(1, 1));
  readonly #cameraFar = uniform(1000);
  readonly #lowResolution = uniform(new THREE.Vector2(1, 1));
  readonly #frame = uniform(0);
  readonly #reprojection = uniform(new THREE.Matrix4());
  readonly #reset = uniform(1);
  readonly #minAlpha = uniform(0.12);
  readonly #depthRejection = uniform(0.06);
  readonly #sunUv = uniform(new THREE.Vector2(0.5, 0.5));
  readonly #radial = uniform(new THREE.Vector3(0.22, 0.5, 0.96));
  readonly #sunOnScreen = uniform(0);

  /* -- GPU resources ----------------------------------------------------- */

  #scatterTarget: THREE.RenderTarget | null = null;
  #history: [THREE.RenderTarget, THREE.RenderTarget] | null = null;
  #shaftTarget: THREE.RenderTarget | null = null;
  #marchPass: FullScreenPass | null = null;
  #temporalPasses: [FullScreenPass, FullScreenPass] | null = null;
  #radialPasses: [FullScreenPass, FullScreenPass] | null = null;
  #historyIndex = 0;

  readonly #scope = new RendererStateScope();
  readonly #previousViewProjection = new THREE.Matrix4();
  readonly #inverseView = new THREE.Matrix4();

  #width = 0;
  #height = 0;
  #graphVersion = -1;
  #guideVersion = -1;
  #radialTaps = 12;
  #radialActive = false;
  #hasPreviousFrame = false;
  #temporalAccepted = false;
  #frameCounter = 0;
  #warnedNoRenderer = false;
  #warnedNoVolumetrics = false;
  #enabled = true;
  #announced = false;

  constructor(options: LightShaftsOptions = {}) {
    this.#options = {
      resolutionScale: clamp(options.resolutionScale ?? 0.5, 0.25, 1),
      temporalBlend: clamp(options.temporalBlend ?? 0.12, 0.01, 1),
      depthRejection: clamp(options.depthRejection ?? 0.06, 0.001, 1),
      intensity: Math.max(0, options.intensity ?? 1),
      radialShafts: options.radialShafts ?? true,
      radialIntensity: Math.max(0, options.radialIntensity ?? 0.22),
      radialSamples: Math.max(2, Math.round(options.radialSamples ?? 12)),
      radialLength: clamp(options.radialLength ?? 0.5, 0.05, 1),
      radialDecay: clamp(options.radialDecay ?? 0.96, 0.5, 1),
    };
    this.#registerService = options.registerService ?? true;
    this.#minAlpha.value = this.#options.temporalBlend;
    this.#depthRejection.value = this.#options.depthRejection;
    this.#radialTaps = this.#options.radialSamples;
    this.#radial.value.set(
      this.#options.radialIntensity,
      this.#options.radialLength,
      this.#options.radialDecay,
    );
    this.pass.setIntensity(this.#options.intensity);
  }

  /* -- GameModule -------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    this.#volumetrics = ctx.services.tryGet(VolumetricsKey);
    this.#sun = ctx.services.tryGet(VolumetricSunKey);
    this.#motion = tryGetMotionVectors(ctx);

    const guide = acquireGuideBuffer(ctx, { resolutionScale: this.#options.resolutionScale });
    this.#guide = guide.provider;
    this.#ownedGuide = guide.owned;

    if (this.#registerService) ctx.services.register(LightShaftsKey, this);
  }

  /** Emitted once, when the half-resolution buffers first exist. */
  #announce(ctx: GameContext): void {
    if (this.#announced) return;
    this.#announced = true;
    ctx.events.emit('lightShafts:ready', {
      halfWidth: this.#width,
      halfHeight: this.#height,
      radial: this.#options.radialShafts && this.#radialTaps >= 2,
    });
  }

  lateUpdate(ctx: GameContext, _dt: number): void {
    if (!this.#enabled) return;

    const volumetrics = this.#volumetrics ?? ctx.services.tryGet(VolumetricsKey);
    this.#volumetrics = volumetrics;
    if (volumetrics === undefined) {
      if (!this.#warnedNoVolumetrics) {
        this.#warnedNoVolumetrics = true;
        console.warn(
          '[LightShafts] no "render.volumetrics" service; register `Volumetrics` for fog.',
        );
      }
      return;
    }
    if (volumetrics.mode === 'off') return;

    const renderer = asNodeRenderer(ctx.renderer);
    if (renderer === null) {
      if (!this.#warnedNoRenderer) {
        this.#warnedNoRenderer = true;
        console.warn('[LightShafts] the renderer is not a node renderer; fog is disabled.');
      }
      return;
    }

    const guide = this.#guide;
    if (guide === null) return;

    renderer.getDrawingBufferSize(_drawingBuffer);
    if (_drawingBuffer.width === 0 || _drawingBuffer.height === 0) return;

    const camera = ctx.camera;
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    const projection = camera.projectionMatrix.elements;
    this.#projScale.value.set(projection[0] ?? 1, projection[5] ?? 1);
    this.#cameraFar.value = camera.far;

    this.#frameCounter++;
    this.#frame.value = this.#frameCounter % 64;

    this.#updateSunScreenPosition(camera);

    // Rebuild whenever the volumetrics graph changed shape (mode swap, quality
    // change, a shadow provider appearing) or the guide buffer's textures were
    // replaced.
    const resized = this.#resize(guide);
    if (
      resized ||
      volumetrics.graphVersion !== this.#graphVersion ||
      guide.version !== this.#guideVersion
    ) {
      this.#graphVersion = volumetrics.graphVersion;
      this.#guideVersion = guide.version;
      this.#build(volumetrics, guide);
      this.#reset.value = 1;
      this.#announce(ctx);
    }

    // Camera-only reprojection, `P_prev · V_prev · V_now⁻¹`. See the module
    // header for why the velocity buffer is deliberately not used here.
    this.#inverseView.copy(camera.matrixWorld);
    this.#reprojection.value.copy(this.#previousViewProjection).multiply(this.#inverseView);
    const cut = this.#motion !== undefined && this.#motion.historyValid === false;
    if (!this.#hasPreviousFrame || cut) this.#reset.value = 1;
    this.#temporalAccepted = this.#reset.value === 0;

    this.#scope.begin(renderer, ctx.scene);
    try {
      this.#ownedGuide?.render(renderer, ctx.scene, camera, ctx.time.frame);
      this.#renderChain(renderer);
    } finally {
      this.#scope.end();
      this.#reset.value = 0;
    }

    this.#previousViewProjection
      .copy(camera.projectionMatrix)
      .multiply(camera.matrixWorldInverse);
    this.#hasPreviousFrame = true;

    // Hand the composite the buffer it should read this frame.
    const source = this.#compositeSource();
    if (source !== null) {
      this.pass.bind(guide, source, this.#width, this.#height);
    }
  }

  dispose(): void {
    const ctx = this.#ctx;
    if (ctx !== null) {
      if (this.#registerService && ctx.services.tryGet(LightShaftsKey) === this) {
        ctx.services.unregister(LightShaftsKey);
      }
      releaseGuideBuffer(ctx);
    }
    this.#teardown();
    this.pass.dispose();
    this.#ctx = null;
    this.#guide = null;
    this.#ownedGuide = null;
  }

  /* -- LightShaftsService ------------------------------------------------ */

  get enabled(): boolean {
    return this.#enabled;
  }

  setEnabled(enabled: boolean): void {
    this.#enabled = enabled;
    this.pass.enabled = enabled;
    if (!enabled) this.#reset.value = 1;
  }

  setIntensity(intensity: number): void {
    this.#options.intensity = Math.max(0, intensity);
    this.pass.setIntensity(this.#options.intensity);
  }

  setRadialIntensity(intensity: number): void {
    this.#options.radialIntensity = Math.max(0, intensity);
    this.#radial.value.x = this.#options.radialIntensity;
  }

  get scatteringTexture(): THREE.Texture | null {
    return this.#compositeSource();
  }

  get stats(): LightShaftsStats {
    const targets = [
      this.#scatterTarget,
      this.#history?.[0] ?? null,
      this.#history?.[1] ?? null,
      this.#shaftTarget,
    ];
    let bytes = 0;
    for (const target of targets) {
      if (target !== null) bytes += target.width * target.height * 8;
    }
    return {
      enabled: this.#enabled,
      width: this.#width,
      height: this.#height,
      radial: this.#radialActive,
      sunOnScreen: this.#sunOnScreen.value,
      temporal: this.#temporalAccepted,
      bytes,
      sharedGuideBuffer: this.#ownedGuide === null,
    };
  }

  /* -- internals --------------------------------------------------------- */

  /**
   * Project the sun onto the screen and decide how much the radial pass is
   * allowed to contribute.
   *
   * A radial blur centred behind the camera sweeps its rays the wrong way, so
   * the contribution is faded out over the last 20° before the sun leaves the
   * frustum rather than being switched off — a hard cut on a sun crossing the
   * screen edge is a very visible pop.
   */
  #updateSunScreenPosition(camera: THREE.PerspectiveCamera): void {
    const direction = this.#sun?.sunDirection;
    if (direction === undefined || !this.#options.radialShafts || this.#radialTaps < 2) {
      this.#sunOnScreen.value = 0;
      return;
    }

    camera.getWorldPosition(_cameraPosition);
    camera.getWorldDirection(_cameraForward);
    _sunWorld.copy(direction).normalize();

    const alignment = _cameraForward.dot(_sunWorld);
    // cos(70°) ≈ 0.342: fully on below that, fading to zero at the horizon of
    // the view direction.
    const onScreen = smoothstep(0, 0.342, alignment);
    this.#sunOnScreen.value = onScreen;
    if (onScreen <= 0) return;

    // A point far along the sun direction; `camera.far` keeps it inside the
    // projection's numerically sane range.
    _sunWorld.multiplyScalar(camera.far * 0.9).add(_cameraPosition);
    _sunWorld.project(camera);
    this.#sunUv.value.set(_sunWorld.x * 0.5 + 0.5, 0.5 - _sunWorld.y * 0.5);
  }

  /** @returns whether the targets were reallocated. */
  #resize(guide: GuideBufferProvider): boolean {
    const width = Math.max(1, guide.halfGuideTexture === null ? guide.width : guide.halfWidth);
    const height = Math.max(1, guide.halfGuideTexture === null ? guide.height : guide.halfHeight);
    if (width === this.#width && height === this.#height && this.#scatterTarget !== null) {
      return false;
    }
    this.#width = width;
    this.#height = height;
    this.#lowResolution.value.set(width, height);

    if (this.#scatterTarget === null) {
      this.#scatterTarget = createTarget('lightShafts.scatter', { type: THREE.HalfFloatType });
      this.#history = [
        createTarget('lightShafts.history.0', { type: THREE.HalfFloatType }),
        createTarget('lightShafts.history.1', { type: THREE.HalfFloatType }),
      ];
      this.#shaftTarget = createTarget('lightShafts.radial', { type: THREE.HalfFloatType });
    }
    this.#scatterTarget.setSize(width, height);
    this.#history?.[0].setSize(width, height);
    this.#history?.[1].setSize(width, height);
    this.#shaftTarget?.setSize(width, height);
    return true;
  }

  #build(volumetrics: VolumetricsService, guide: GuideBufferProvider): void {
    const lowGuide = guide.halfGuideTexture ?? guide.guideTexture;
    const history = this.#history;
    const scatter = this.#scatterTarget;
    if (lowGuide === null || history === null || scatter === null) return;

    this.#marchPass?.dispose();
    this.#temporalPasses?.[0].dispose();
    this.#temporalPasses?.[1].dispose();
    this.#radialPasses?.[0].dispose();
    this.#radialPasses?.[1].dispose();

    this.#marchPass = new FullScreenPass(
      'lightShafts.march',
      this.#buildMarchFragment(volumetrics, lowGuide),
    );
    this.#temporalPasses = [
      new FullScreenPass(
        'lightShafts.temporal.0',
        this.#buildTemporalFragment(scatter.texture, history[1].texture, lowGuide),
      ),
      new FullScreenPass(
        'lightShafts.temporal.1',
        this.#buildTemporalFragment(scatter.texture, history[0].texture, lowGuide),
      ),
    ];
    // One radial material per history parity, for the same reason the temporal
    // pass has two: it reads whichever history was just written. A single
    // material selecting between the two with a uniform would work, but it
    // would issue *both* texture fetches at every tap — twice the bandwidth of
    // the pass's dominant cost, to save one material.
    this.#radialPasses = [
      new FullScreenPass('lightShafts.radial.0', this.#buildRadialFragment(history[0].texture)),
      new FullScreenPass('lightShafts.radial.1', this.#buildRadialFragment(history[1].texture)),
    ];
  }

  /**
   * The half-resolution scattering march.
   *
   * Almost all of this function is supplied by `Volumetrics.createResolveNode`;
   * what lives here is only the screen-space plumbing — which pixel, at what
   * depth, with which dither index.
   */
  #buildMarchFragment(volumetrics: VolumetricsService, lowGuide: THREE.Texture): THREE.Node {
    const guideNode = textureNode(lowGuide);
    return Fn(() => {
      const base = uv();
      const guide = decodeGuide(guideNode.sample(base), this.#cameraFar).toVar('guide');
      // Where there is no geometry the guide holds `camera.far`, which is
      // exactly right: the ray integrates the whole volume and the sky behind
      // it is attenuated by the resulting transmittance.
      const viewZ = guide.w.toVar('viewZ');
      const pixel = base.mul(this.#lowResolution).toVar('pixel');
      return volumetrics.createResolveNode({
        screenUv: base as unknown as Vec2Node,
        viewZ: viewZ as unknown as FloatNode,
        pixel: pixel as unknown as Vec2Node,
      });
    })();
  }

  /**
   * Exponential moving average with camera reprojection and depth rejection.
   *
   * No neighbourhood clamp. The usual TAA variance clip exists to stop a
   * high-frequency signal from smearing, and volumetric scattering has no high
   * frequencies to protect — applying it would clamp the filter back towards
   * this frame's noise, which is exactly what the filter is there to remove.
   * What replaces it is a depth test (a reprojected sample that lands on
   * different geometry is discarded) plus an adaptive term on transmittance,
   * which is the channel that actually changes when the world changes.
   */
  #buildTemporalFragment(
    current: THREE.Texture,
    history: THREE.Texture,
    lowGuide: THREE.Texture,
  ): THREE.Node {
    const currentNode = textureNode(current);
    const historyNode = textureNode(history);
    const guideNode = textureNode(lowGuide);

    return Fn(() => {
      const base = uv();
      const guide = decodeGuide(guideNode.sample(base), this.#cameraFar).toVar('guide');
      const viewZ = guide.w.toVar('viewZ');
      const now = currentNode.sample(base).toVar('current');

      const viewPosition = viewPositionFromDepth(base, viewZ, this.#projScale).toVar('viewP');
      const reprojected = this.#reprojection.mul(vec4(viewPosition, 1)).toVar('prevClip');
      const previousUv = vec2(
        reprojected.x.div(reprojected.w).mul(0.5).add(0.5),
        float(0.5).sub(reprojected.y.div(reprojected.w).mul(0.5)),
      ).toVar('prevUv');

      const inside = previousUv.x
        .greaterThanEqual(0)
        .and(previousUv.x.lessThanEqual(1))
        .and(previousUv.y.greaterThanEqual(0))
        .and(previousUv.y.lessThanEqual(1))
        .and(reprojected.w.greaterThan(0));

      const blended = now.toVar('blended');
      If(inside.and(this.#reset.lessThan(0.5)), () => {
        const previous = historyNode.sample(previousUv).toVar('history');
        const previousGuide = decodeGuide(guideNode.sample(previousUv), this.#cameraFar);

        // Relative depth test. An absolute epsilon rejects everything at range
        // and nothing up close, which is the wrong way round for fog.
        const depthDelta = viewZ
          .sub(previousGuide.w)
          .abs()
          .div(max(viewZ, float(1e-3)))
          .toVar('depthDelta');
        const depthValid = saturate(
          float(1).sub(depthDelta.div(this.#depthRejection)),
        ).toVar('depthValid');

        // Scale-free disagreement on transmittance: the channel that moves when
        // a shaft is occluded or a torch is lit.
        const change = now.w
          .sub(previous.w)
          .abs()
          .div(max(now.w.add(previous.w), float(1e-4)))
          .toVar('change');

        const alpha = saturate(
          mix(
            float(1),
            this.#minAlpha.add(float(1).sub(this.#minAlpha).mul(change).mul(change)),
            depthValid,
          ),
        ).toVar('alpha');
        blended.assign(mix(previous, now, alpha));
      });

      return blended;
    })();
  }

  /**
   * Mitchell's radial accumulation, applied to the scattering buffer.
   *
   * The march walks from the pixel towards the sun's screen position with a
   * geometrically decaying weight, so the contribution of a bright, unshadowed
   * patch of fog is smeared *away* from the sun — the direction light actually
   * travels. Because the input is scattering and not scene colour, an occluder
   * that is dark in the scattering buffer (because the shadow map says the fog
   * behind it is unlit) blocks the shaft for free; that is the mechanism
   * Mitchell's occlusion prepass exists to fake, and here it is already true.
   *
   * The output replaces the scattering buffer for compositing, carrying
   * transmittance through untouched — a radial blur of transmittance would
   * leak fog across silhouettes.
   */
  #buildRadialFragment(history: THREE.Texture): THREE.Node {
    const historyNode = textureNode(history);
    const taps = Math.max(2, this.#radialTaps);

    return Fn(() => {
      const base = uv();
      const source = historyNode.sample(base).toVar('source');

      const accumulated = vec3(0, 0, 0).toVar('radial');
      If(this.#sunOnScreen.greaterThan(0.001), () => {
        // Step vector towards the sun, covering `radialLength` of the distance
        // in `taps` steps, with a per-pixel dither on the start so the taps do
        // not land on the same texels for every pixel (which shows as rings).
        const toSun = this.#sunUv.sub(base).mul(this.#radial.y).div(taps).toVar('step');
        const dither = interleavedGradientNoiseNode(
          base.mul(this.#lowResolution),
          this.#frame,
        ).toVar('dither');
        const cursor = base.add(toSun.mul(dither)).toVar('cursor');
        const weight = float(1).toVar('weight');
        const weightSum = float(0).toVar('weightSum');

        Loop({ start: 0, end: taps, type: 'int' }, () => {
          cursor.addAssign(toSun);
          const clamped = cursor.clamp(vec2(0, 0), vec2(1, 1));
          accumulated.addAssign(historyNode.sample(clamped).xyz.mul(weight));
          weightSum.addAssign(weight);
          weight.mulAssign(this.#radial.z);
        });

        accumulated.assign(accumulated.div(max(weightSum, float(1e-4))));
      });

      // Normalised so the radial term is an *addition of the same order* as the
      // signal it came from, scaled by the artist control and by how far inside
      // the frustum the sun is.
      const gain = this.#radial.x.mul(this.#sunOnScreen);
      return vec4(source.xyz.add(accumulated.mul(gain)), source.w);
    })();
  }

  #renderChain(renderer: NodeRenderer): void {
    const march = this.#marchPass;
    const temporal = this.#temporalPasses;
    const scatter = this.#scatterTarget;
    const history = this.#history;
    if (march === null || temporal === null || scatter === null || history === null) return;

    march.render(renderer, scatter);

    this.#historyIndex = 1 - this.#historyIndex;
    const historyTarget = history[this.#historyIndex];
    if (historyTarget !== undefined) {
      temporal[this.#historyIndex]?.render(renderer, historyTarget);
    }

    const radial = this.#radialPasses?.[this.#historyIndex] ?? null;
    const shaft = this.#shaftTarget;
    this.#radialActive =
      this.#options.radialShafts &&
      this.#radialTaps >= 2 &&
      this.#sunOnScreen.value > 0.001 &&
      radial !== null &&
      shaft !== null;
    if (this.#radialActive && radial !== null && shaft !== null) {
      radial.render(renderer, shaft);
    }
  }

  #compositeSource(): THREE.Texture | null {
    if (this.#radialActive && this.#shaftTarget !== null) return this.#shaftTarget.texture;
    return this.#history?.[this.#historyIndex]?.texture ?? null;
  }

  #teardown(): void {
    this.#marchPass?.dispose();
    this.#temporalPasses?.[0].dispose();
    this.#temporalPasses?.[1].dispose();
    this.#radialPasses?.[0].dispose();
    this.#radialPasses?.[1].dispose();
    this.#marchPass = null;
    this.#temporalPasses = null;
    this.#radialPasses = null;

    this.#scatterTarget?.dispose();
    this.#history?.[0].dispose();
    this.#history?.[1].dispose();
    this.#shaftTarget?.dispose();
    this.#scatterTarget = null;
    this.#history = null;
    this.#shaftTarget = null;
    this.#width = 0;
    this.#height = 0;
  }

  /**
   * Apply a quality tier. Called by the integrator when `PostStack` changes
   * tier; the tier only chooses the radial tap count here, because the march's
   * step count belongs to `Volumetrics`.
   */
  setQuality(quality: QualityTier): void {
    const taps = Math.min(this.#options.radialSamples, RADIAL_TAPS[quality]);
    if (taps === this.#radialTaps) return;
    this.#radialTaps = taps;
    // The tap count is an unrolled loop bound, so the material has to be
    // rebuilt; forcing a graph-version mismatch does that on the next frame.
    this.#graphVersion = -1;
  }
}

/* ------------------------------------------------------------------------- *
 * Small helpers
 * ------------------------------------------------------------------------- */

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** Scalar `smoothstep`, matching the GLSL definition. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0 || 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * A texture node whose `value` is rebound every frame.
 *
 * The post chain hands a *different* texture object to each pass on each frame
 * (the pool ping-pongs), so a node built against one of them at construction
 * time would sample a stale buffer. The placeholder keeps the graph valid
 * before the first frame.
 */
function makeSourceNode(name: string): THREE.TextureNode {
  const placeholder = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  placeholder.needsUpdate = true;
  const node = textureNode(placeholder);
  node.name = name;
  return node;
}

/* ------------------------------------------------------------------------- *
 * Entry point
 * ------------------------------------------------------------------------- */

/**
 * Construct the light-shaft module and register it with the engine.
 *
 * Register {@link module:render/Volumetrics} too — this module resolves it
 * through the service locator and, without it, does nothing but warn once.
 * Order does not matter: the lookup is retried until it succeeds.
 *
 * @returns the module, whose `.pass` must be spliced into `PostStack`'s chain
 *   for the fog to reach the frame. See "For the integrator" in the module
 *   header.
 */
export function registerLightShafts(
  ctx: GameContext,
  options: LightShaftsOptions = {},
): LightShaftsModule {
  const module = new LightShaftsModule(options);
  ctx.engine.add(module);
  return module;
}

/** Re-exported so a consumer needs one import for the whole volumetric system. */
export { VolumetricsKey } from '../Volumetrics';
export type { VolumetricsService } from '../Volumetrics';
