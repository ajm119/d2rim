/**
 * @module render/post/PostStack
 *
 * The post-processing backbone: an ordered chain of passes, explicit render
 * target ownership, and one place where HDR becomes LDR.
 *
 * ---
 *
 * ## Where this sits in the frame
 *
 * `Engine` ends every frame with `renderer.render(scene, camera)` through the
 * backend-agnostic {@link RendererHandle}. `core/types.ts` is frozen and
 * `core/Engine.ts` belongs to nobody in this phase, so `PostStack` inserts
 * itself by *replacing* `RendererHandle.render` (and `captureFrame`) with a
 * function that drives the chain, keeping the originals for delegation and
 * restoring them on `dispose()`. `render` and `captureFrame` are the only two
 * non-`readonly` members of `RendererHandle`, which makes this a legal
 * implementation of the contract rather than a hole punched through it.
 *
 * Delegation matters: the wrapper only takes over when it is handed *the*
 * scene and *the* camera it was initialised with. Every other `render` call —
 * PMREM prefiltering, shadow atlas debug views, an env-map bake — passes
 * straight through untouched.
 *
 * If the integrator prefers an explicit call site, construct with
 * `{ install: false }` and call {@link PostStack.render} directly.
 *
 * ## Chain
 *
 * ```
 *  scene ──► sceneTarget  (MRT: colour RGBA16F + velocity RG16F, + depth)
 *              │
 *              ├── TAA          chain   HDR → HDR   (history is private)
 *              ├── Bloom        producer            (pyramid is private)
 *              ├── Composite    chain   HDR → LDR   (exposure, AgX, grade)
 *              └── FXAA         chain   LDR → LDR
 *                                        │
 *                                        ▼  screen or capture target
 * ```
 *
 * A **chain** pass consumes the running colour texture and produces a new one.
 * A **producer** pass reads the running colour but writes only into buffers it
 * owns, so it does not consume a ping-pong slot — that is why bloom, which is
 * six render targets of work, costs zero full-resolution scratch memory here.
 *
 * ## Ping-pong, and why there is a pool at all
 *
 * The naive implementation gives each effect its own full-resolution target.
 * At 1080p in RGBA16F that is 16.6 MB *per effect*, and every one of them is
 * live for the whole frame. Instead, {@link RenderTargetPool} hands out targets
 * from a per-format free list: a chain pass borrows an output, and as soon as
 * the next pass has consumed it the buffer goes back and is immediately reused.
 * The pool therefore holds a bounded number of buffers no matter how many
 * passes are enabled — and in practice a very small one, because the *last*
 * chain pass writes straight to the destination rather than through a redundant
 * copy-to-screen blit:
 *
 * - `high`/`ultra` (TAA + bloom + composite): **zero** pooled buffers. TAA
 *   supplies its own output (its history), bloom is a producer, and the
 *   composite is last so it writes to the screen.
 * - `low`/`medium` (bloom + composite + FXAA): **one**, the LDR buffer the
 *   composite hands to FXAA.
 *
 * A chain twice as long would still need two — one per format in flight —
 * which is the property the pool exists to guarantee.
 *
 * ## Colour management
 *
 * The scene is drawn into a linear half-float target, so `Renderer` applies
 * neither tone mapping nor an output transfer function (`Renderer.isOutputTarget`
 * is false for any target that is not the designated output). Everything up to
 * the composite pass is scene-referred linear. The composite pass applies
 * exposure and the tone curve, and emits display-referred linear in `[0, 1]`;
 * the renderer's `outputColorSpace` then applies the sRGB OETF on the final
 * write. `renderer.toneMapping` is forced to `NoToneMapping` while the stack is
 * installed — leaving three's ACES in place would tone-map an already
 * tone-mapped image.
 *
 * The one wrinkle is FXAA, whose edge detection is defined on gamma-encoded
 * luma. When an AA pass is last in the chain the composite is
 * told to emit sRGB-encoded values into a target tagged
 * `LinearSRGBColorSpace` (so sampling does not undo the encode), and the
 * renderer's `outputColorSpace` is switched to linear for that frame so the
 * final write passes the encoded bytes through unchanged. The AA therefore
 * operates in the space it was designed for and the frame is encoded exactly
 * once, with no transfer-function round trip.
 *
 * ## Quality tiers
 *
 * | | low | medium | high | ultra |
 * |---|---|---|---|---|
 * | anti-aliasing | FXAA | FXAA | TAA (8) | TAA (16) + sharpen |
 * | motion vectors | off | on | on | on |
 * | bloom mips | 4 | 5 | 6 | 7 |
 * | auto exposure | off | on | on | on |
 * | grain / vignette / CA | vignette only | + grain | all | all |
 * | render scale | 0.75 | 1.0 | 1.0 | 1.0 |
 *
 * The tier system is not decoration: `low` removes the velocity attachment
 * entirely, which also removes the previous-position vertex work from every
 * skinned material, and it drops the internal resolution — the two changes that
 * actually move the needle on a weak GPU. WebGL2 is not itself a lower tier
 * (three's `WebGLBackend` runs the same TSL graphs), but it is capped at `high`
 * because the 16-tap variance clip plus sharpen at `ultra` costs more on a
 * driver that cannot re-use the WebGPU pipeline cache.
 *
 * ## References
 *
 * - J. Jimenez, *"Next Generation Post Processing in Call of Duty: Advanced
 *   Warfare"*, SIGGRAPH 2014 — pyramid bloom, and the general "one pass per
 *   concept, one target per lifetime" structure.
 * - B. Karis, *"High Quality Temporal Supersampling"*, SIGGRAPH 2014 — TAA.
 * - three.js `PassNode` / `PostProcessing` — the MRT and renderer-state
 *   save/restore protocol this module follows.
 */

import * as THREE from 'three/webgpu';

import { trackRenderTarget } from '../MemoryReport';
import { texture as textureNodeFactory } from 'three/tsl';

import { serviceKey } from '../../core/ServiceLocator';
import type {
  CapturedFrame,
  GameContext,
  GameModule,
  RendererBackend,
  RendererHandle,
} from '../../core/types';
import { unpackRows } from '../RendererFactory';

import { BloomPass, type BloomOptions } from './Bloom';
import { ColorGrade, type ColorGradeOptions } from './ColorGrade';
import { MotionVectors, MotionVectorsKey, type MotionVectorsOptions } from './Motion';
import { FxaaPass, TAAPass, type TAAOptions } from './TAA';
import { CompositePass, type TonemapOptions } from './Tonemap';

/* ------------------------------------------------------------------------- *
 * Public vocabulary
 * ------------------------------------------------------------------------- */

export type QualityTier = 'low' | 'medium' | 'high' | 'ultra';

export type AntiAliasMode = 'taa' | 'fxaa' | 'none';

/** Ordinal comparison helper — tiers are ordered, not just named. */
const TIER_ORDER: Readonly<Record<QualityTier, number>> = {
  low: 0,
  medium: 1,
  high: 2,
  ultra: 3,
};

/** `true` when `tier` is at least `minimum`. */
export function tierAtLeast(tier: QualityTier, minimum: QualityTier): boolean {
  return TIER_ORDER[tier] >= TIER_ORDER[minimum];
}

/** Static description of what the current device can actually do. */
export interface PostCapabilities {
  readonly backend: RendererBackend;
  /** Two-channel half-float render targets are colour-renderable. */
  readonly rgHalfFloat: boolean;
  /** Half-float render targets are colour-renderable at all. */
  readonly halfFloat: boolean;
  /** `Renderer.reversedDepthBuffer` — flips the sense of the depth attachment. */
  readonly reversedDepth: boolean;
}

/**
 * Everything a pass is handed for one invocation.
 *
 * The object identity is stable across passes and frames; the fields are
 * rewritten before each `render`. Passes must not retain it.
 */
export interface PostFrame {
  readonly renderer: THREE.Renderer;
  readonly capabilities: PostCapabilities;
  readonly camera: THREE.PerspectiveCamera;
  readonly quality: QualityTier;
  /** Chain resolution, which may be below the destination resolution. */
  readonly width: number;
  readonly height: number;
  /** Seconds since the previous rendered frame, clamped to something sane. */
  readonly deltaTime: number;
  /** Seconds since the stack was installed. Drives grain and adaptation. */
  readonly elapsed: number;
  readonly frameIndex: number;
  readonly motion: MotionVectors;
  readonly depthTexture: THREE.DepthTexture | null;
  /** Colour entering this pass. */
  readonly input: THREE.Texture;
  /** Where this pass must write. `null` means the renderer's output surface. */
  readonly output: THREE.RenderTarget | null;
  /** Draw a full-screen triangle with `material` into `target`. */
  blit(material: THREE.Material, target: THREE.RenderTarget | null, label: string): void;
}

export type PostPassKind = 'chain' | 'producer';

/**
 * One effect in the stack.
 *
 * Implementations own any buffer whose contents must survive between frames
 * (TAA history, bloom pyramid). Everything else comes from the pool.
 */
export interface PostPass {
  readonly id: string;
  readonly kind: PostPassKind;
  /**
   * The pass wants display-encoded (sRGB) input rather than display-linear.
   * Only meaningful for a pass that runs after the composite.
   */
  readonly prefersEncodedInput?: boolean;
  /**
   * The pass's output is already display-encoded, so the renderer must not
   * apply its output transfer function again.
   */
  readonly outputsEncoded?: boolean;
  /** Format of the target `PostStack` allocates for a `chain` pass. */
  readonly outputDomain: 'hdr' | 'ldr';
  /**
   * The pass writes into a buffer it owns and returns that texture, so no pool
   * target is allocated for it. TAA sets this: its resolve target *is* its
   * history, and copying between the two would be a full-resolution round trip
   * for nothing. A pass that owns its output must still honour `frame.output`
   * when it happens to be last in the chain.
   */
  readonly ownsOutput?: boolean;
  /** Artist/user toggle. */
  enabled: boolean;
  /** Whether the pass can run at all on this tier and device. */
  isAvailable(quality: QualityTier, capabilities: PostCapabilities): boolean;
  /** Chain resolution changed. */
  setSize(width: number, height: number): void;
  /** Tier or device changed; rebuild node graphs if their shape depends on it. */
  configure(quality: QualityTier, capabilities: PostCapabilities): void;
  /**
   * Run the pass.
   *
   * @returns the texture holding the result when the pass supplied its own
   *   buffer, or `null`/`void` when it wrote into `frame.output`.
   */
  render(frame: PostFrame): THREE.Texture | null | void;
  dispose(): void;
}

export interface PostStackStats {
  /** Full-screen draws issued by the chain, excluding the scene draw. */
  readonly passDraws: number;
  /** Render targets currently allocated by the pool and by passes. */
  readonly targets: number;
  /** Approximate GPU bytes held by every target this module owns. */
  readonly bytes: number;
  /** Ids of the passes that ran on the last frame, in order. */
  readonly active: readonly string[];
}

export interface PostStackOptions {
  /** Starting tier, or `'auto'` to derive one from the backend. Default `'auto'`. */
  quality?: QualityTier | 'auto';
  /** Anti-aliasing technique. `'auto'` follows the tier table. Default `'auto'`. */
  antiAlias?: AntiAliasMode | 'auto';
  /**
   * Internal resolution multiplier applied on top of the tier default. The
   * chain runs at this scale and the final pass upscales on the way to the
   * destination. Clamped to `[0.5, 1]`.
   */
  renderScale?: number;
  /**
   * Replace `RendererHandle.render`/`captureFrame` so that the existing engine
   * loop routes through the stack. Default `true`. Set `false` to drive
   * {@link PostStack.render} yourself.
   */
  install?: boolean;
  /** Register under {@link PostStackKey}. Default `true`. */
  registerService?: boolean;
  motion?: MotionVectorsOptions;
  taa?: TAAOptions;
  bloom?: BloomOptions;
  tonemap?: TonemapOptions;
  grade?: ColorGradeOptions;
}

/** Service key for the stack itself, so UI/settings code can retune it. */
export const PostStackKey = serviceKey<PostStack>('render.post');

/** `RendererHandle.captureFrame` with the optionality stripped. */
type CaptureFrameFn = NonNullable<RendererHandle['captureFrame']>;

/* ------------------------------------------------------------------------- *
 * Render target pool
 * ------------------------------------------------------------------------- */

type TargetDomain = 'hdr' | 'ldr';

/**
 * A free list of full-resolution scratch targets, one list per format.
 *
 * Targets are never destroyed between frames — only handed back — so the
 * steady state allocates nothing and the GPU allocator is never touched inside
 * the frame. `releaseAll()` at the end of the frame is a safety net for a pass
 * that threw before releasing.
 */
export class RenderTargetPool {
  #width = 1;
  #height = 1;

  readonly #free: Record<TargetDomain, THREE.RenderTarget[]> = { hdr: [], ldr: [] };
  readonly #all: THREE.RenderTarget[] = [];
  readonly #inUse = new Set<THREE.RenderTarget>();

  #halfFloat = true;

  constructor(width: number, height: number, halfFloat = true) {
    this.#width = Math.max(1, Math.floor(width));
    this.#height = Math.max(1, Math.floor(height));
    this.#halfFloat = halfFloat;
  }

  get width(): number {
    return this.#width;
  }

  get height(): number {
    return this.#height;
  }

  /** Number of targets ever allocated and still alive. */
  get count(): number {
    return this.#all.length;
  }

  /** Approximate GPU bytes held. */
  get bytes(): number {
    let total = 0;
    for (const target of this.#all) {
      const bytesPerPixel = target.texture.type === THREE.HalfFloatType ? 8 : 4;
      total += target.width * target.height * bytesPerPixel;
    }
    return total;
  }

  setSize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.#width && h === this.#height) return;
    this.#width = w;
    this.#height = h;
    for (const target of this.#all) target.setSize(w, h);
  }

  acquire(domain: TargetDomain): THREE.RenderTarget {
    const free = this.#free[domain];
    const reused = free.pop();
    if (reused !== undefined) {
      this.#inUse.add(reused);
      return reused;
    }

    const hdr = domain === 'hdr' && this.#halfFloat;
    const target = new THREE.RenderTarget(this.#width, this.#height, {
      format: THREE.RGBAFormat,
      type: hdr ? THREE.HalfFloatType : THREE.UnsignedByteType,
      // Every intermediate buffer is scene- or display-referred *linear data*,
      // never sRGB-encoded pixels that need decoding on sample. The one
      // exception (the composite's sRGB hand-off to FXAA) relies on exactly
      // this: `LinearSRGBColorSpace` means "sample the bytes as stored".
      colorSpace: THREE.LinearSRGBColorSpace,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      samples: 0,
    });
    target.texture.name = `post.pool.${domain}.${this.#all.length}`;
    trackRenderTarget(target);
    target.texture.wrapS = THREE.ClampToEdgeWrapping;
    target.texture.wrapT = THREE.ClampToEdgeWrapping;

    this.#all.push(target);
    this.#inUse.add(target);
    return target;
  }

  release(target: THREE.RenderTarget | null): void {
    if (target === null || !this.#inUse.delete(target)) return;
    const domain: TargetDomain = target.texture.type === THREE.HalfFloatType ? 'hdr' : 'ldr';
    this.#free[domain].push(target);
  }

  /** Return every outstanding target. Called once per frame. */
  releaseAll(): void {
    for (const target of this.#inUse) {
      const domain: TargetDomain = target.texture.type === THREE.HalfFloatType ? 'hdr' : 'ldr';
      this.#free[domain].push(target);
    }
    this.#inUse.clear();
  }

  dispose(): void {
    for (const target of this.#all) target.dispose();
    this.#all.length = 0;
    this.#free.hdr.length = 0;
    this.#free.ldr.length = 0;
    this.#inUse.clear();
  }
}

/* ------------------------------------------------------------------------- *
 * Tier table
 * ------------------------------------------------------------------------- */

interface TierProfile {
  readonly antiAlias: AntiAliasMode;
  readonly velocity: boolean;
  readonly jitterSamples: number;
  readonly bloomMips: number;
  readonly autoExposure: boolean;
  readonly grain: boolean;
  readonly chromaticAberration: boolean;
  readonly sharpen: boolean;
  readonly renderScale: number;
}

const TIERS: Readonly<Record<QualityTier, TierProfile>> = {
  low: {
    antiAlias: 'fxaa',
    velocity: false,
    jitterSamples: 8,
    bloomMips: 4,
    autoExposure: false,
    grain: false,
    chromaticAberration: false,
    sharpen: false,
    renderScale: 0.75,
  },
  medium: {
    antiAlias: 'fxaa',
    velocity: true,
    jitterSamples: 8,
    bloomMips: 5,
    autoExposure: true,
    grain: true,
    chromaticAberration: false,
    sharpen: false,
    renderScale: 1,
  },
  high: {
    antiAlias: 'taa',
    velocity: true,
    jitterSamples: 8,
    bloomMips: 6,
    autoExposure: true,
    grain: true,
    chromaticAberration: true,
    sharpen: true,
    renderScale: 1,
  },
  ultra: {
    antiAlias: 'taa',
    velocity: true,
    jitterSamples: 16,
    bloomMips: 7,
    autoExposure: true,
    grain: true,
    chromaticAberration: true,
    sharpen: true,
    renderScale: 1,
  },
};

/* ------------------------------------------------------------------------- *
 * PostStack
 * ------------------------------------------------------------------------- */

/** Reused for every full-screen draw; three's own passes do the same. */
const quad = new THREE.QuadMesh();

/**
 * Narrow structural view of the renderer surfaces `@types/three` does not
 * describe on the common `Renderer` base. One cast, at the boundary.
 */
interface RendererInternals {
  setMRT(mrt: THREE.Node | null): void;
  getMRT(): THREE.Node | null;
  setOutputRenderTarget(target: THREE.RenderTarget | null): void;
  getOutputRenderTarget(): THREE.RenderTarget | null;
  reversedDepthBuffer?: boolean;
  autoClear: boolean;
}

export class PostStack implements GameModule {
  readonly name = 'render.post';

  readonly motion: MotionVectors;
  readonly taa: TAAPass;
  readonly bloom: BloomPass;
  readonly composite: CompositePass;
  readonly grade: ColorGrade;
  readonly fxaa: FxaaPass;

  /** Master switch. When false the stack blits the raw scene to the screen. */
  enabled = true;

  readonly #options: PostStackOptions;
  #quality: QualityTier = 'high';
  /** Explicit auto-exposure setting from the constructor, or null to follow the tier. */
  readonly #autoExposureOverride: boolean | null = null;
  #antiAlias: AntiAliasMode | 'auto';
  #renderScaleOverride: number | null;

  #passes: PostPass[] = [];
  /**
   * Passes inserted through {@link PostStack.addPass}. They are owned by
   * whoever added them, so `dispose()` unlinks them instead of destroying
   * them — double-disposing a pass whose owning module also disposes it
   * leaves a freed material bound to a live node graph.
   */
  readonly #externalPasses = new Set<PostPass>();
  #pool: RenderTargetPool | null = null;
  #sceneTarget: THREE.RenderTarget | null = null;

  #renderer: THREE.Renderer | null = null;
  #handle: RendererHandle | null = null;
  #scene: THREE.Scene | null = null;
  #camera: THREE.PerspectiveCamera | null = null;
  #capabilities: PostCapabilities | null = null;

  #installed = false;
  #originalRender: RendererHandle['render'] | null = null;
  #originalCapture: CaptureFrameFn | null = null;
  #originalToneMapping: THREE.ToneMapping | null = null;

  #width = 1;
  #height = 1;
  #deltaTime = 1 / 60;
  #elapsed = 0;
  #frameIndex = 0;
  #passDraws = 0;
  #active: string[] = [];
  #inRender = false;
  /** Offscreen presentation size, or `null` to present to the canvas. */
  #headless: THREE.Vector2 | null = null;

  /** Blit material for the degenerate "nothing is enabled" path. */
  #copyMaterial: THREE.NodeMaterial | null = null;
  #copyTexture: THREE.Node | null = null;

  /** Mutable per-pass frame descriptor; identity is stable. */
  readonly #frame: {
    renderer: THREE.Renderer;
    capabilities: PostCapabilities;
    camera: THREE.PerspectiveCamera;
    quality: QualityTier;
    width: number;
    height: number;
    deltaTime: number;
    elapsed: number;
    frameIndex: number;
    motion: MotionVectors;
    depthTexture: THREE.DepthTexture | null;
    input: THREE.Texture;
    output: THREE.RenderTarget | null;
    blit: PostFrame['blit'];
  };

  constructor(options: PostStackOptions = {}) {
    this.#options = options;
    this.#antiAlias = options.antiAlias ?? 'auto';
    this.#renderScaleOverride =
      options.renderScale === undefined
        ? null
        : THREE.MathUtils.clamp(options.renderScale, 0.5, 1);

    this.motion = new MotionVectors(options.motion ?? {});
    this.grade = new ColorGrade(options.grade ?? {});
    this.taa = new TAAPass(this.motion, options.taa ?? {});
    this.bloom = new BloomPass(options.bloom ?? {});
    this.composite = new CompositePass(this.bloom, this.grade, options.tonemap ?? {});
    // An explicitly configured auto-exposure setting outranks the tier table.
    //
    // Without this, `#applyTier` re-derived auto exposure from the quality tier
    // on every tier change and on init, which meant a scene that had
    // deliberately locked its exposure got it silently unlocked at `low` (off)
    // versus `high` (on) — and a *locked* key is the whole reason the same
    // scene reads the same in every shot. See `Tonemap`'s note on why a
    // composed frame should not meter itself.
    this.#autoExposureOverride = options.tonemap?.autoExposure ?? null;
    this.fxaa = new FxaaPass();

    // Placeholder values; every field is rewritten before the first pass runs.
    this.#frame = {
      renderer: null as unknown as THREE.Renderer,
      capabilities: null as unknown as PostCapabilities,
      camera: null as unknown as THREE.PerspectiveCamera,
      quality: 'high',
      width: 1,
      height: 1,
      deltaTime: 1 / 60,
      elapsed: 0,
      frameIndex: 0,
      motion: this.motion,
      depthTexture: null,
      input: null as unknown as THREE.Texture,
      output: null,
      blit: (material, target, label) => this.#blit(material, target, label),
    };
  }

  /* -- GameModule -------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#handle = ctx.renderer;
    this.#renderer = ctx.renderer.three as THREE.Renderer;
    this.#scene = ctx.scene;
    this.#camera = ctx.camera;
    this.#capabilities = probeCapabilities(ctx.renderer);

    const requested = this.#options.quality ?? 'auto';
    this.#quality = requested === 'auto' ? defaultTier(this.#capabilities) : requested;

    this.#passes = [this.taa, this.bloom, this.composite, this.fxaa];

    if (this.#options.registerService !== false) {
      ctx.services.register(PostStackKey, this);
      ctx.services.register(MotionVectorsKey, this.motion);
    }

    this.#applyTier();

    const size = this.#drawingBufferSize();
    this.#resize(size.x, size.y);

    if (this.#options.install !== false) this.#install();

    ctx.events.emit('module:added', { name: `${this.name}:${this.#quality}` });
  }

  update(_ctx: GameContext, dt: number): void {
    // Clamped: a tab-switch hitch must not make auto-exposure jump a stop.
    this.#deltaTime = THREE.MathUtils.clamp(dt, 1 / 480, 1 / 15);
    this.#elapsed += this.#deltaTime;
  }

  dispose(): void {
    this.#uninstall();
    for (const pass of this.#passes) {
      if (!this.#externalPasses.has(pass)) pass.dispose();
    }
    this.#externalPasses.clear();
    this.#passes = [];
    this.motion.dispose();
    this.grade.dispose();
    this.#pool?.dispose();
    this.#pool = null;
    this.#disposeSceneTarget();
    this.#copyMaterial?.dispose();
    this.#copyMaterial = null;
    this.#renderer = null;
    this.#handle = null;
  }

  /* -- public surface ---------------------------------------------------- */

  get quality(): QualityTier {
    return this.#quality;
  }

  /**
   * Switch tier. Reconfigures every pass, resizes the chain, and rejects the
   * temporal history (the sample sequence and resolution both changed).
   */
  setQuality(quality: QualityTier): void {
    if (quality === this.#quality) return;
    this.#quality = quality;
    this.#applyTier();
    const size = this.#drawingBufferSize();
    this.#resize(size.x, size.y);
    this.motion.reset();
    this.taa.resetHistory();
  }

  /** Force an anti-aliasing technique, or `'auto'` to follow the tier. */
  setAntiAlias(mode: AntiAliasMode | 'auto'): void {
    this.#antiAlias = mode;
    this.#applyTier();
  }

  get antiAlias(): AntiAliasMode {
    return this.#antiAlias === 'auto' ? TIERS[this.#quality].antiAlias : this.#antiAlias;
  }

  /** Toggle a single pass by id (`'post.taa'`, `'post.bloom'`, ...). */
  setPassEnabled(id: string, enabled: boolean): boolean {
    const pass = this.#passes.find((candidate) => candidate.id === id);
    if (pass === undefined) return false;
    pass.enabled = enabled;
    return true;
  }

  isPassEnabled(id: string): boolean {
    return this.#passes.find((candidate) => candidate.id === id)?.enabled ?? false;
  }

  get passes(): readonly PostPass[] {
    return this.#passes;
  }

  /**
   * Splice an externally-owned pass into the chain.
   *
   * The built-in chain is fixed (`taa → bloom → composite → fxaa`) because
   * those four have a hard ordering relationship with the one place HDR
   * becomes LDR. Everything else that wants to run inside the chain — light
   * shafts compositing the volumetric buffer, a screen-space reflection
   * resolve — is authored in its own module and inserted here, which is what
   * keeps `PostStack` from importing half the renderer.
   *
   * Insertion is by id rather than by index so that a caller does not have to
   * know how many passes the current tier happens to have enabled.
   *
   * @param before id of the pass to insert in front of. Appends when omitted
   *   or unknown. HDR-domain passes must go before `'post.composite'`.
   */
  addPass(pass: PostPass, options: { readonly before?: string } = {}): void {
    if (this.#passes.some((candidate) => candidate.id === pass.id)) {
      throw new Error(`[PostStack] a pass with id "${pass.id}" is already in the chain`);
    }
    const at =
      options.before === undefined
        ? -1
        : this.#passes.findIndex((candidate) => candidate.id === options.before);
    this.#passes.splice(at < 0 ? this.#passes.length : at, 0, pass);
    this.#externalPasses.add(pass);

    // Bring the newcomer up to the state every other pass is already in. A
    // pass added after `init` would otherwise run its first frame at 1x1 with
    // an unconfigured graph.
    if (this.#capabilities !== null) pass.configure(this.#quality, this.#capabilities);
    pass.setSize(this.#width, this.#height);
  }

  /** Remove a pass added by {@link PostStack.addPass}. Does not dispose it. */
  removePass(id: string): boolean {
    const index = this.#passes.findIndex((candidate) => candidate.id === id);
    if (index < 0) return false;
    const removed = this.#passes.splice(index, 1);
    const pass = removed[0];
    if (pass !== undefined) this.#externalPasses.delete(pass);
    return true;
  }

  /**
   * The HDR scene colour attachment, before any chain pass has run.
   *
   * Exposed for screen-space effects that trace against last frame's radiance:
   * they run in `lateUpdate`, i.e. *before* this frame's scene draw, so what
   * they read here is the previous frame's fully-lit colour. That is exactly
   * the contract `SSR`'s `SceneColorProvider` describes with
   * `isPreviousFrame: true`. `null` until the first frame has been drawn.
   */
  get sceneColorTexture(): THREE.Texture | null {
    // `textures[0]`, not `.texture`: the target is MRT whenever the tier keeps
    // a velocity attachment, and `.texture` on an MRT target is the array.
    return this.#sceneTarget?.textures[0] ?? null;
  }

  get stats(): PostStackStats {
    const sceneBytes = this.#sceneTargetBytes();
    return {
      passDraws: this.#passDraws,
      targets: (this.#pool?.count ?? 0) + (this.#sceneTarget === null ? 0 : 1),
      bytes: (this.#pool?.bytes ?? 0) + sceneBytes + this.bloom.bytes + this.taa.bytes,
      active: this.#active,
    };
  }

  /**
   * Draw `scene` through the whole stack.
   *
   * @param destination `null` renders to the canvas. A render target is used
   *   as-is, and is temporarily designated the renderer's output target so that
   *   the final pass still receives the output colour-space conversion.
   */
  render(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    destination: THREE.RenderTarget | null = null,
  ): void {
    const renderer = this.#renderer;
    const capabilities = this.#capabilities;
    if (renderer === null || capabilities === null) return;

    // Re-entrancy would corrupt the pool's in-use set and the renderer state
    // stack. It should be impossible, but a pass that accidentally calls the
    // wrapped handle would otherwise fail in a very confusing way.
    if (this.#inRender) return;
    this.#inRender = true;

    // A caller asking for the canvas gets the offscreen surface instead while
    // headless presentation is on. See `setHeadlessPresentation`.
    const headless = this.#headless;
    const target =
      destination ?? (headless === null ? null : this.#ensureCaptureTarget(headless.x, headless.y));

    const internals = renderer as unknown as RendererInternals;
    const previousOutput = internals.getOutputRenderTarget();
    const previousColorSpace = renderer.outputColorSpace;

    try {
      const targetWidth = target === null ? this.#drawingBufferSize().x : target.width;
      const targetHeight = target === null ? this.#drawingBufferSize().y : target.height;
      this.#resize(targetWidth, targetHeight);

      if (target !== null) internals.setOutputRenderTarget(target);

      const sceneTarget = this.#ensureSceneTarget();
      this.#renderScene(renderer, internals, scene, camera, sceneTarget);

      if (!this.enabled) {
        this.#copyToDestination(sceneTarget.textures[0] ?? sceneTarget.texture, target);
        return;
      }

      this.#runChain(renderer, capabilities, camera, sceneTarget, target);
    } finally {
      internals.setOutputRenderTarget(previousOutput);
      renderer.outputColorSpace = previousColorSpace;
      renderer.setRenderTarget(null);
      this.#pool?.releaseAll();
      this.#frameIndex++;
      this.#inRender = false;
    }
  }

  /**
   * Send frames that would go to the canvas into the capture target instead.
   *
   * This exists for one reason, and it is the same reason {@link captureFrame}
   * exists: in the headless container this project is graded in, *presenting* a
   * WebGPU swapchain kills the device after a handful of frames ("A valid
   * external Instance reference no longer exists"), which takes the readback
   * down with it. The capture harness worked around that by refusing to run any
   * warmup frames on WebGPU — and a warmup is not optional here. TAA, the
   * froxel volume and the environment probe all converge over the first dozen
   * frames, so an unwarmed frame is a different picture, roughly 0.12 of mean
   * luminance and a whole colour temperature away from the converged one. The
   * "WebGPU/WebGL2 backend divergence" was that: a converged frame compared
   * against an unconverged one.
   *
   * Rendering the warmup into a render target exercises the identical chain at
   * the identical resolution and never touches the swapchain, so both backends
   * can now converge the same number of frames. `null` restores the canvas.
   *
   * @param width  offscreen width, normally the capture width
   * @param height offscreen height
   */
  setHeadlessPresentation(width: number | null, height = 0): void {
    if (width === null) {
      this.#headless = null;
      return;
    }
    this.#headless = new THREE.Vector2(Math.max(1, Math.floor(width)), Math.max(1, Math.floor(height)));
  }

  /** Whether {@link setHeadlessPresentation} is currently diverting the canvas. */
  get headlessPresentation(): boolean {
    return this.#headless !== null;
  }

  /**
   * Render one frame through the stack and read it back as tightly packed
   * top-row-first RGBA8, matching {@link RendererHandle.captureFrame}.
   *
   * This exists because the headless container cannot present a WebGPU
   * swapchain (see `RendererFactory`), so every capture goes through a render
   * target. Post-processing must be *in* that path or the critic loop grades an
   * image the player never sees.
   */
  async captureFrame(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    width: number,
    height: number,
  ): Promise<CapturedFrame> {
    const renderer = this.#renderer;
    if (renderer === null) throw new Error('[PostStack] captureFrame before init');

    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    const target = this.#ensureCaptureTarget(w, h);

    this.render(scene, camera, target);

    const data = await renderer.readRenderTargetPixelsAsync(target, 0, 0, w, h);
    const raw = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const pixels = unpackRows(raw, w, h);

    return {
      width: w,
      height: h,
      pixels: this.#capabilities?.backend === 'webgl2' ? flipRows(pixels, w, h) : pixels,
    };
  }

  /* -- installation ------------------------------------------------------ */

  #install(): void {
    const handle = this.#handle;
    const renderer = this.#renderer;
    if (handle === null || renderer === null || this.#installed) return;

    this.#originalRender = handle.render.bind(handle);
    this.#originalCapture = handle.captureFrame?.bind(handle) ?? null;

    // Three's own tone mapping would compress an image this stack already
    // tone-mapped. The output transfer function stays where it is: the
    // composite deliberately emits display-referred *linear*.
    this.#originalToneMapping = renderer.toneMapping;
    renderer.toneMapping = THREE.NoToneMapping;

    handle.render = (scene: THREE.Scene, camera: THREE.Camera): void => {
      // Only *the* scene and *the* camera go through the stack. PMREM bakes,
      // shadow debug views and env-map captures pass straight through.
      if (scene === this.#scene && camera === this.#camera) {
        this.render(scene, camera as THREE.PerspectiveCamera, null);
        return;
      }
      this.#originalRender?.(scene, camera);
    };

    handle.captureFrame = async (
      scene: THREE.Scene,
      camera: THREE.Camera,
      captureWidth?: number,
      captureHeight?: number,
    ): Promise<CapturedFrame> => {
      const delegate = this.#originalCapture;
      if (scene !== this.#scene || camera !== this.#camera) {
        if (delegate === null) {
          throw new Error('[PostStack] the wrapped renderer has no captureFrame');
        }
        return delegate(scene, camera, captureWidth, captureHeight);
      }
      const size = this.#drawingBufferSize();
      return this.captureFrame(
        scene,
        camera as THREE.PerspectiveCamera,
        captureWidth ?? size.x,
        captureHeight ?? size.y,
      );
    };

    this.#installed = true;
  }

  #uninstall(): void {
    const handle = this.#handle;
    const renderer = this.#renderer;
    if (handle === null || !this.#installed) return;

    const restoreRender = this.#originalRender;
    const restoreCapture = this.#originalCapture;
    if (restoreRender !== null) handle.render = restoreRender;
    if (restoreCapture !== null) handle.captureFrame = restoreCapture;
    if (renderer !== null && this.#originalToneMapping !== null) {
      renderer.toneMapping = this.#originalToneMapping;
    }

    this.#originalRender = null;
    this.#originalCapture = null;
    this.#originalToneMapping = null;
    this.#installed = false;
  }

  /* -- frame ------------------------------------------------------------- */

  /**
   * Draw the scene into the MRT target.
   *
   * Renderer state is saved and restored around the draw exactly the way
   * `PassNode` does it, because everything between here and the final blit
   * depends on `setMRT`/`setRenderTarget` being where it was left.
   */
  #renderScene(
    renderer: THREE.Renderer,
    internals: RendererInternals,
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    sceneTarget: THREE.RenderTarget,
  ): void {
    const previousTarget = renderer.getRenderTarget();
    const previousMRT = internals.getMRT();
    const previousAutoClear = internals.autoClear;

    this.motion.beginSceneDraw(camera, this.#width, this.#height);
    try {
      internals.setMRT(this.motion.mrtNode);
      internals.autoClear = true;
      renderer.setRenderTarget(sceneTarget);
      renderer.render(scene, camera);
    } finally {
      this.motion.endSceneDraw(camera);
      renderer.setRenderTarget(previousTarget);
      internals.setMRT(previousMRT);
      internals.autoClear = previousAutoClear;
    }

    const depth = sceneTarget.depthTexture;
    this.motion.setTargets(sceneTarget.textures[1] ?? null, depth);
  }

  /** Walk the pass list, ping-ponging pool targets between chain passes. */
  #runChain(
    renderer: THREE.Renderer,
    capabilities: PostCapabilities,
    camera: THREE.PerspectiveCamera,
    sceneTarget: THREE.RenderTarget,
    destination: THREE.RenderTarget | null,
  ): void {
    const pool = this.#ensurePool();
    const active = this.#passes.filter(
      (pass) => pass.enabled && pass.isAvailable(this.#quality, capabilities),
    );

    this.#active = active.map((pass) => pass.id);
    this.#passDraws = 0;

    let lastChain = -1;
    for (let i = 0; i < active.length; i++) {
      if (active[i]?.kind === 'chain') lastChain = i;
    }

    // The composite is the only pass that knows how to turn HDR into something
    // displayable, so an AA pass that follows it must receive sRGB-encoded
    // values. Tell it before anything runs.
    const lastPass = lastChain >= 0 ? active[lastChain] : undefined;
    const trailing = lastPass !== undefined && lastPass !== this.composite ? lastPass : undefined;

    this.composite.setEncodeOutput(trailing?.prefersEncodedInput === true);

    // When the chain already emits display-encoded values, the renderer must
    // not encode them a second time. Switching the output colour space is the
    // whole mechanism — it is cheaper than an EOTF/OETF round trip and, unlike
    // wrapping the trailing pass's result in an inverse transform, it cannot be
    // defeated by an addon that returns a pass texture rather than an
    // expression.
    renderer.outputColorSpace =
      trailing?.outputsEncoded === true ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace;

    const frame = this.#frame;
    frame.renderer = renderer;
    frame.capabilities = capabilities;
    frame.camera = camera;
    frame.quality = this.#quality;
    frame.width = this.#width;
    frame.height = this.#height;
    frame.deltaTime = this.#deltaTime;
    frame.elapsed = this.#elapsed;
    frame.frameIndex = this.#frameIndex;
    frame.depthTexture = sceneTarget.depthTexture;

    let current = sceneTarget.textures[0] ?? sceneTarget.texture;
    let currentBorrowed: THREE.RenderTarget | null = null;

    for (let i = 0; i < active.length; i++) {
      const pass = active[i];
      if (pass === undefined) continue;

      frame.input = current;

      if (pass.kind === 'producer') {
        frame.output = null;
        pass.render(frame);
        continue;
      }

      const isLast = i === lastChain;
      const suppliesOwn = pass.ownsOutput === true && !isLast;
      const output = isLast ? destination : suppliesOwn ? null : pool.acquire(pass.outputDomain);
      frame.output = output;

      const produced = pass.render(frame) ?? null;

      if (produced !== null && produced !== current) {
        // The pass handed back a buffer of its own; the one it consumed is free.
        pool.release(currentBorrowed);
        currentBorrowed = null;
        current = produced;
      } else if (produced === null) {
        // Release *after* the pass has consumed it, so a buffer is never handed
        // to a second pass while it is still being read.
        pool.release(currentBorrowed);
        currentBorrowed = isLast ? null : output;
        current = isLast ? current : (output?.texture ?? current);
      }
      // produced === current: a pass-through. Keep the borrow exactly as it is.
    }

    if (lastChain === -1) {
      // Producers only (or nothing enabled): the scene still has to reach the
      // screen, and it is still HDR, so this is a tone-mapped copy at worst.
      this.#copyToDestination(current, destination);
    }

    pool.release(currentBorrowed);
  }

  #copyToDestination(texture: THREE.Texture, destination: THREE.RenderTarget | null): void {
    const material = this.#ensureCopyMaterial();
    const source = this.#copyTexture as THREE.TextureNode | null;
    if (source !== null) source.value = texture;
    this.#blit(material, destination, 'post.copy');
  }

  #blit(material: THREE.Material, target: THREE.RenderTarget | null, label: string): void {
    const renderer = this.#renderer;
    if (renderer === null) return;
    renderer.setRenderTarget(target);
    quad.material = material;
    quad.name = label;
    quad.render(renderer);
    this.#passDraws++;
  }

  /* -- resources --------------------------------------------------------- */

  #applyTier(): void {
    const profile = TIERS[this.#quality];
    const capabilities = this.#capabilities;
    const mode = this.antiAlias;

    this.motion.setVelocityEnabled(profile.velocity);
    this.motion.setSequenceLength(profile.jitterSamples);
    this.motion.setJitterEnabled(mode === 'taa' && profile.velocity);

    this.taa.enabled = mode === 'taa' && profile.velocity;
    this.taa.setSharpenEnabled(profile.sharpen);
    this.fxaa.enabled = mode === 'fxaa';

    this.bloom.setMipCount(profile.bloomMips);
    this.composite.setAutoExposureEnabled(this.#autoExposureOverride ?? profile.autoExposure);
    this.grade.setGrainEnabled(profile.grain);
    this.grade.setChromaticAberrationEnabled(profile.chromaticAberration);

    if (capabilities !== null) {
      for (const pass of this.#passes) pass.configure(this.#quality, capabilities);
    }
  }

  #renderScale(): number {
    return this.#renderScaleOverride ?? TIERS[this.#quality].renderScale;
  }

  #resize(targetWidth: number, targetHeight: number): void {
    const scale = this.#renderScale();
    const width = Math.max(1, Math.floor(targetWidth * scale));
    const height = Math.max(1, Math.floor(targetHeight * scale));
    if (width === this.#width && height === this.#height) return;

    this.#width = width;
    this.#height = height;

    this.#pool?.setSize(width, height);
    this.#sceneTarget?.setSize(width, height);
    for (const pass of this.#passes) pass.setSize(width, height);
    this.motion.reset();
  }

  #ensurePool(): RenderTargetPool {
    if (this.#pool === null) {
      this.#pool = new RenderTargetPool(
        this.#width,
        this.#height,
        this.#capabilities?.halfFloat ?? true,
      );
    }
    return this.#pool;
  }

  /**
   * The scene target: linear half-float colour, plus a velocity attachment when
   * the tier asks for one, plus depth.
   *
   * `samples: 0` is deliberate. MSAA and TAA do not compose (resolving before
   * the temporal filter destroys the sub-pixel information TAA is integrating),
   * and at the FXAA tiers MSAA on a multi-attachment half-float target is the
   * most expensive way available to buy a small amount of edge quality.
   */
  #ensureSceneTarget(): THREE.RenderTarget {
    const capabilities = this.#capabilities;
    const wantVelocity = this.motion.velocityEnabled;
    const existing = this.#sceneTarget;

    if (existing !== null) {
      const hasVelocity = existing.textures.length > 1;
      if (hasVelocity === wantVelocity) {
        if (existing.width !== this.#width || existing.height !== this.#height) {
          existing.setSize(this.#width, this.#height);
        }
        return existing;
      }
      this.#disposeSceneTarget();
    }

    const halfFloat = capabilities?.halfFloat ?? true;
    const target = new THREE.RenderTarget(this.#width, this.#height, {
      format: THREE.RGBAFormat,
      type: halfFloat ? THREE.HalfFloatType : THREE.UnsignedByteType,
      colorSpace: THREE.LinearSRGBColorSpace,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      generateMipmaps: false,
      depthBuffer: true,
      stencilBuffer: false,
      samples: 0,
      count: wantVelocity ? 2 : 1,
    });

    trackRenderTarget(target, 'post.scene');

    // `MRTNode` binds outputs to attachments by *name*, so these must match the
    // keys used in `mrt({ output, velocity })`.
    const colour = target.textures[0];
    if (colour !== undefined) {
      colour.name = 'output';
      colour.wrapS = THREE.ClampToEdgeWrapping;
      colour.wrapT = THREE.ClampToEdgeWrapping;
    }

    const velocityTexture = target.textures[1];
    if (velocityTexture !== undefined) {
      velocityTexture.name = 'velocity';
      velocityTexture.type = THREE.HalfFloatType;
      // Two channels when the device can render to them, four otherwise. The
      // WebGL2 fallback needs EXT_color_buffer_float for RG16F specifically;
      // RGBA16F is the safer universal format at twice the bandwidth.
      velocityTexture.format = capabilities?.rgHalfFloat === true ? THREE.RGFormat : THREE.RGBAFormat;
      velocityTexture.minFilter = THREE.NearestFilter;
      velocityTexture.magFilter = THREE.NearestFilter;
      velocityTexture.generateMipmaps = false;
      velocityTexture.wrapS = THREE.ClampToEdgeWrapping;
      velocityTexture.wrapT = THREE.ClampToEdgeWrapping;
    }

    const depth = new THREE.DepthTexture(this.#width, this.#height);
    depth.name = 'post.sceneDepth';
    if (capabilities?.reversedDepth === true) depth.type = THREE.FloatType;
    target.depthTexture = depth;

    this.#sceneTarget = target;
    return target;
  }

  #disposeSceneTarget(): void {
    const target = this.#sceneTarget;
    if (target === null) return;
    target.depthTexture?.dispose();
    target.dispose();
    this.#sceneTarget = null;
  }

  #sceneTargetBytes(): number {
    const target = this.#sceneTarget;
    if (target === null) return 0;
    let bytes = 0;
    for (const texture of target.textures) {
      const channels = texture.format === THREE.RGFormat ? 2 : 4;
      const size = texture.type === THREE.HalfFloatType ? 2 : 1;
      bytes += target.width * target.height * channels * size;
    }
    // Depth attachment: 32-bit is the pessimistic assumption.
    bytes += target.width * target.height * 4;
    return bytes;
  }

  #captureTarget: THREE.RenderTarget | null = null;

  /**
   * The target every readback capture is rendered into.
   *
   * `colorSpace` is **`LinearSRGBColorSpace`, and that is not a typo** — it is
   * the difference between a capture that matches the screen and one that does
   * not. The tag does not describe the values; it selects the texture *format*,
   * and `SRGBColorSpace` gets an `rgba8unorm-srgb` (GL: `SRGB8_ALPHA8`) surface
   * whose hardware writes apply the sRGB OETF. `render()` designates this
   * target as the renderer's output target, which makes `Renderer.currentColorSpace`
   * report `outputColorSpace`, so three's own output pass *already* encodes.
   * Tagged sRGB the frame is therefore encoded twice — measured on both
   * backends: display-linear 0.5 came back as byte 223 where the canvas, and a
   * PNG, want 188. Every readback capture in the project was reading roughly
   * one stop brighter than the image a player sees, which is exactly how
   * `hud-composite` (a real screenshot of the canvas) ended up "disagreeing"
   * with every other shot. Linear-tagged storage stores what the output pass
   * wrote, i.e. exactly one encode.
   */
  #ensureCaptureTarget(width: number, height: number): THREE.RenderTarget {
    let target = this.#captureTarget;
    if (target === null) {
      target = new THREE.RenderTarget(width, height, {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        colorSpace: THREE.LinearSRGBColorSpace,
        depthBuffer: false,
        stencilBuffer: false,
        generateMipmaps: false,
        samples: 0,
      });
      target.texture.name = 'post.capture';
      this.#captureTarget = target;
    } else if (target.width !== width || target.height !== height) {
      target.setSize(width, height);
    }
    return target;
  }

  #ensureCopyMaterial(): THREE.NodeMaterial {
    let material = this.#copyMaterial;
    if (material === null) {
      const source = makeTextureNode('post.copy.source');
      material = new THREE.NodeMaterial();
      material.name = 'post.copy';
      material.depthTest = false;
      material.depthWrite = false;
      material.fragmentNode = source as unknown as THREE.Node;
      this.#copyMaterial = material;
      this.#copyTexture = source as unknown as THREE.Node;
    }
    return material;
  }

  #drawingBufferSize(): THREE.Vector2 {
    const renderer = this.#renderer;
    if (renderer === null) return new THREE.Vector2(1, 1);
    return renderer.getDrawingBufferSize(new THREE.Vector2());
  }
}

/* ------------------------------------------------------------------------- *
 * Entry point
 * ------------------------------------------------------------------------- */

/**
 * Construct a {@link PostStack}, register it with the engine, and hand it back.
 *
 * Register it **last**, after every module that draws into the scene: the stack
 * reads `ctx.renderer` at `init` time to install its wrapper, and a module that
 * wraps `render` afterwards would end up outside the post chain.
 */
export function registerPostStack(ctx: GameContext, options: PostStackOptions = {}): PostStack {
  const stack = new PostStack(options);
  ctx.engine.add(stack);
  return stack;
}

/* ------------------------------------------------------------------------- *
 * Helpers
 * ------------------------------------------------------------------------- */

/**
 * A `TextureNode` with no texture bound yet.
 *
 * Every pass builds its node graph once, at construction, and swaps the bound
 * texture per frame by assigning `.value`. Rebuilding the graph instead would
 * recompile the shader on the first frame after every ping-pong swap.
 */
export function makeTextureNode(name: string): THREE.TextureNode {
  // `texture()` requires a Texture; a 1x1 placeholder keeps the node valid
  // until the first real assignment and costs 4 bytes.
  const placeholder = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  placeholder.name = `${name}.placeholder`;
  placeholder.needsUpdate = true;
  const node = textureNodeFactory(placeholder);
  node.name = name;
  return node;
}

/**
 * Starting tier when the caller says `'auto'`.
 *
 * Both backends run the same TSL graphs, so this is not a correctness
 * distinction. `high` is the default everywhere: it is the lowest tier at which
 * TAA runs, and the project's art direction — thin bare branches, rain, distant
 * fog — aliases badly enough that spatial AA alone looks broken. The integrator
 * is expected to expose the tier in a settings menu and drop it on request.
 */
function defaultTier(_capabilities: PostCapabilities): QualityTier {
  return 'high';
}

function probeCapabilities(handle: RendererHandle): PostCapabilities {
  const renderer = handle.three as unknown as RendererInternals;
  const reversedDepth = renderer.reversedDepthBuffer === true;

  if (handle.backend === 'webgpu') {
    return { backend: 'webgpu', rgHalfFloat: true, halfFloat: true, reversedDepth };
  }

  // Boundary cast: `getContext()` is `unknown` on the common Renderer because
  // the concrete type depends on the backend.
  const gl = (handle.three as THREE.Renderer).getContext() as WebGL2RenderingContext | null;
  if (gl === null || typeof gl.getExtension !== 'function') {
    return { backend: 'webgl2', rgHalfFloat: false, halfFloat: true, reversedDepth };
  }

  // EXT_color_buffer_float makes RG16F/RGBA16F colour-renderable;
  // EXT_color_buffer_half_float only guarantees the RGBA variants.
  const full = gl.getExtension('EXT_color_buffer_float') !== null;
  const half = full || gl.getExtension('EXT_color_buffer_half_float') !== null;
  return { backend: 'webgl2', rgHalfFloat: full, halfFloat: half, reversedDepth };
}

/** Reverse row order of a tightly packed RGBA8 buffer, in place. */
function flipRows(pixels: Uint8Array, width: number, height: number): Uint8Array {
  const stride = width * 4;
  const row = new Uint8Array(stride);
  for (let y = 0; y < height >> 1; y++) {
    const top = y * stride;
    const bottom = (height - 1 - y) * stride;
    row.set(pixels.subarray(top, top + stride));
    pixels.copyWithin(top, bottom, bottom + stride);
    pixels.set(row, bottom);
  }
  return pixels;
}
