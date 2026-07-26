/**
 * @module render/post/TAA
 *
 * Temporal anti-aliasing, plus the two spatial fallbacks (FXAA, SMAA) that the
 * lower quality tiers use instead.
 *
 * ---
 *
 * ## The technique
 *
 * The camera is jittered by a sub-pixel offset every frame (see
 * {@link module:render/post/Motion}) and the results are integrated over time.
 * Left alone, that integration is wrong the instant anything moves, so the
 * whole art of TAA is in deciding how much of the accumulated history to keep.
 * This implementation follows Karis, *"High Quality Temporal Supersampling"*
 * (SIGGRAPH 2014, Advances in Real-Time Rendering), with the refinements that
 * have become standard since:
 *
 * 1. **Reprojection with dilated velocity.** The velocity is fetched not at the
 *    pixel itself but at whichever of its 3x3 neighbours is *closest to the
 *    camera*. Without this, a silhouette pixel picks up the background's
 *    velocity and drags a smear of foreground colour behind every moving edge —
 *    the single most recognisable TAA artefact.
 *
 * 2. **Neighbourhood clamping in YCoCg.** The history is constrained to the
 *    convex bounds of the current frame's 3x3 neighbourhood. YCoCg rather than
 *    RGB because the clamp is fundamentally a statement about *luminance*
 *    plausibility, and an RGB axis-aligned box is a poor fit to a luminance-
 *    dominated distribution: it is simultaneously too loose (letting ghosts
 *    through on chroma) and too tight (clipping legitimate saturated colour).
 *    Karis §4.2 measured a substantial artefact reduction from this change
 *    alone.
 *
 * 3. **Variance clipping instead of a min/max box.** The bounds come from the
 *    neighbourhood's mean ± γ·σ (Salvi, *"An Excursion in Temporal
 *    Supersampling"*, GDC 2016). A min/max box is dominated by its two most
 *    extreme samples, so one firefly inflates it until it stops rejecting
 *    anything. Clipping — moving the history point along the ray toward the box
 *    centre until it lands on the surface — rather than clamping per-channel
 *    keeps the hue of the corrected sample.
 *
 * 4. **Luminance-weighted blending (Karis anti-flicker).** The current and
 *    history samples are weighted by `1/(1+luma)` before mixing. A bright,
 *    temporally unstable sample therefore contributes less, which is what stops
 *    specular highlights on wet stone from crawling.
 *
 * 5. **Catmull-Rom history reconstruction.** Reprojection lands between texels,
 *    and a bilinear fetch there is a low-pass filter applied *every frame*, so
 *    a bilinear history is exponentially blurry under sustained motion. The
 *    5-tap Catmull-Rom approximation (Karis, and MJP's `SampleTextureCatmullRom`)
 *    costs four extra bilinear fetches and removes essentially all of it.
 *
 * 6. **Post-resolve sharpening.** Even with (5), integrating sub-pixel samples
 *    is a mild low-pass. A single unsharp-mask tap against the neighbourhood
 *    mean — which is already in registers from (3) — restores the edge acuity,
 *    clamped to the neighbourhood bounds so it cannot ring.
 *
 * ## History rejection
 *
 * The history is discarded outright, not merely clamped, when:
 *
 * - the reprojected coordinate leaves the frame (nothing to reproject from);
 * - {@link MotionVectors.historyValid} is false — a camera cut or a resize;
 * - the pixel's velocity exceeds `maxVelocity` texels, where reprojection is
 *   too inaccurate to trust regardless of what the clamp says.
 *
 * Between "keep" and "discard" the blend weight ramps with velocity, so fast-
 * moving objects converge toward the current frame and slow ones keep the
 * accumulated detail. This is the mechanism that makes ghosting on a moving
 * character a *tuning* question rather than a structural one.
 *
 * ## Fallbacks
 *
 * FXAA 3.11 (Lottes, NVIDIA 2011) and SMAA 1x (Jimenez et al., 2012) are wired
 * to three.js's own TSL implementations. Both are spatial, so neither needs
 * velocity or jitter, which is what makes them the right choice for the `low`
 * and `medium` tiers where the velocity attachment is not even allocated. Both
 * operate on gamma-encoded luma, so the composite hands them sRGB-encoded
 * values and they convert back to linear on the way out — see the colour
 * management note in {@link module:render/post/PostStack}.
 */

import * as THREE from 'three/webgpu';
import {
  float,
  ivec2,
  max as maxNode,
  min as minNode,
  mix,
  sRGBTransferEOTF,
  texture as textureNode,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { smaa } from 'three/addons/tsl/display/SMAANode.js';

import type { MotionVectors } from './Motion';
import type { PostCapabilities, PostFrame, PostPass, QualityTier } from './PostStack';

/* ------------------------------------------------------------------------- *
 * Options
 * ------------------------------------------------------------------------- */

export interface TAAOptions {
  /**
   * Minimum weight given to the current frame, in `[0, 1]`. 1/16 is the
   * classic value: it caps the effective history length at ~16 frames, which is
   * long enough to converge an 8- or 16-sample jitter sequence and short enough
   * that a mis-rejected ghost decays in a quarter of a second.
   */
  currentWeight?: number;
  /**
   * Variance clipping γ. Larger accepts more history (smoother, more prone to
   * ghosting); smaller rejects more (crisper, more prone to flicker). 1.0 is
   * the usual starting point.
   */
  varianceGamma?: number;
  /** Velocity, in texels per frame, at which the history is fully rejected. */
  maxVelocity?: number;
  /** Unsharp-mask amount applied after the resolve. 0 disables. */
  sharpness?: number;
  /** Catmull-Rom history reconstruction. Costs 4 extra fetches. Default true. */
  bicubicHistory?: boolean;
}

/* ------------------------------------------------------------------------- *
 * Colour space helpers
 * ------------------------------------------------------------------------- */

/**
 * RGB to YCoCg-R.
 *
 * A lifting-scheme integer-reversible transform in its original form; used here
 * for its ordinary real-valued version because all that matters is that the
 * first axis carries essentially all the luminance and the other two carry
 * chroma differences with roughly equal, much smaller variance.
 */
function rgbToYCoCg(color: THREE.Node<'vec3'>): THREE.Node<'vec3'> {
  return vec3(
    color.dot(vec3(0.25, 0.5, 0.25)),
    color.dot(vec3(0.5, 0.0, -0.5)),
    color.dot(vec3(-0.25, 0.5, -0.25)),
  ) as unknown as THREE.Node<'vec3'>;
}

function yCoCgToRgb(color: THREE.Node<'vec3'>): THREE.Node<'vec3'> {
  const y = color.x;
  const co = color.y;
  const cg = color.z;
  return vec3(y.add(co).sub(cg), y.add(cg), y.sub(co).sub(cg)) as unknown as THREE.Node<'vec3'>;
}

/**
 * Playdead's optimised AABB clip.
 *
 * Rather than clamping each channel independently — which moves the sample to a
 * box corner and changes its hue — this finds the intersection of the segment
 * from the box centre to the history point with the box surface. Reference:
 * https://github.com/playdeadgames/temporal (`clip_aabb`).
 */
function clipToAABB(
  history: THREE.Node<'vec3'>,
  minimum: THREE.Node<'vec3'>,
  maximum: THREE.Node<'vec3'>,
): THREE.Node<'vec3'> {
  const centre = maximum.add(minimum).mul(0.5);
  const extent = maximum.sub(minimum).mul(0.5).add(1e-7);
  const offset = history.sub(centre);
  const unit = offset.div(extent).abs();
  const longest = maxNode(maxNode(unit.x, unit.y), unit.z);
  return longest
    .greaterThan(1)
    .select(centre.add(offset.div(longest)), history) as unknown as THREE.Node<'vec3'>;
}

/* ------------------------------------------------------------------------- *
 * TAAPass
 * ------------------------------------------------------------------------- */

export class TAAPass implements PostPass {
  readonly id = 'post.taa';
  readonly kind = 'chain' as const;
  readonly outputDomain = 'hdr' as const;
  readonly ownsOutput = true;

  enabled = true;

  readonly #motion: MotionVectors;

  #currentWeight: number;
  #varianceGamma: number;
  #maxVelocity: number;
  #sharpness: number;
  #bicubic: boolean;
  #sharpenEnabled = true;

  #width = 1;
  #height = 1;

  #history: THREE.RenderTarget[] = [];
  #index = 0;
  #needsReset = true;

  #material: THREE.NodeMaterial | null = null;
  #copyMaterial: THREE.NodeMaterial | null = null;
  #copySource: THREE.TextureNode | null = null;
  #structure = '';

  readonly #uCurrent = makeSourceNode('taa.current');
  readonly #uHistory = makeSourceNode('taa.history');
  readonly #uVelocity = makeSourceNode('taa.velocity');
  readonly #uDepth = makeSourceNode('taa.depth');
  readonly #uTexel = uniform(new THREE.Vector2(1, 1));
  readonly #uSize = uniform(new THREE.Vector2(1, 1));
  /** `(currentWeight, varianceGamma, maxVelocity, sharpness)` */
  readonly #uParams = uniform(new THREE.Vector4(1 / 16, 1, 64, 0.35));
  /** 1 when the history may be used at all, 0 on a cut or the first frame. */
  readonly #uHistoryValid = uniform(0);
  /** 1 when the depth buffer is reversed (near = 1). */
  readonly #uReversedDepth = uniform(0);

  constructor(motion: MotionVectors, options: TAAOptions = {}) {
    this.#motion = motion;
    this.#currentWeight = options.currentWeight ?? 1 / 16;
    this.#varianceGamma = options.varianceGamma ?? 1.0;
    this.#maxVelocity = options.maxVelocity ?? 64;
    this.#sharpness = options.sharpness ?? 0.35;
    this.#bicubic = options.bicubicHistory ?? true;
    this.#uploadParams();
  }

  /* -- configuration ----------------------------------------------------- */

  setCurrentWeight(weight: number): void {
    this.#currentWeight = THREE.MathUtils.clamp(weight, 1 / 64, 1);
    this.#uploadParams();
  }

  setVarianceGamma(gamma: number): void {
    this.#varianceGamma = Math.max(0.1, gamma);
    this.#uploadParams();
  }

  setSharpness(sharpness: number): void {
    this.#sharpness = Math.max(0, sharpness);
    this.#uploadParams();
  }

  setSharpenEnabled(enabled: boolean): void {
    if (enabled === this.#sharpenEnabled) return;
    this.#sharpenEnabled = enabled;
    this.#uploadParams();
  }

  setBicubicHistory(enabled: boolean): void {
    if (enabled === this.#bicubic) return;
    this.#bicubic = enabled;
    this.#invalidate();
  }

  /** Throw away the accumulated history on the next frame. */
  resetHistory(): void {
    this.#needsReset = true;
  }

  get bytes(): number {
    let total = 0;
    for (const target of this.#history) total += target.width * target.height * 8;
    return total;
  }

  /* -- PostPass ---------------------------------------------------------- */

  isAvailable(_quality: QualityTier, capabilities: PostCapabilities): boolean {
    // No velocity buffer, no reprojection. Falling back to a naive
    // accumulate-everything TAA would ghost catastrophically, so the honest
    // answer is to be unavailable and let FXAA take over.
    return capabilities.halfFloat && this.#motion.velocityEnabled;
  }

  configure(_quality: QualityTier, capabilities: PostCapabilities): void {
    this.#uReversedDepth.value = capabilities.reversedDepth ? 1 : 0;
  }

  setSize(width: number, height: number): void {
    if (width === this.#width && height === this.#height) return;
    this.#width = width;
    this.#height = height;
    this.#uTexel.value.set(1 / width, 1 / height);
    this.#uSize.value.set(width, height);
    for (const target of this.#history) target.setSize(width, height);
    this.#needsReset = true;
  }

  /**
   * Resolve into the current history buffer and hand that texture on.
   *
   * The resolve target *is* the history — writing somewhere else and copying
   * would cost a full-resolution read plus write (about 33 MB at 1080p) to
   * accomplish a pointer swap.
   */
  render(frame: PostFrame): THREE.Texture | null {
    this.#ensureHistory();

    const current = this.#history[this.#index];
    const previous = this.#history[1 - this.#index];
    if (current === undefined || previous === undefined) return null;

    const velocityTexture = this.#motion.velocityTexture;
    const depthTexture = frame.depthTexture;
    if (velocityTexture === null || depthTexture === null) {
      // Without velocity or depth the resolve is meaningless; pass through.
      return frame.input;
    }

    const material = this.#ensureMaterial();

    this.#uCurrent.value = frame.input;
    this.#uHistory.value = previous.texture;
    this.#uVelocity.value = velocityTexture;
    this.#uDepth.value = depthTexture;
    this.#uHistoryValid.value =
      this.#needsReset || !this.#motion.historyValid ? 0 : 1;

    frame.blit(material, current, 'post.taa');

    this.#needsReset = false;
    this.#index = 1 - this.#index;

    // Degenerate configuration: TAA is the final chain pass, so it also has to
    // land in the destination. Never happens with the composite enabled.
    if (frame.output !== null) {
      const copy = this.#ensureCopyMaterial();
      if (this.#copySource !== null) this.#copySource.value = current.texture;
      frame.blit(copy, frame.output, 'post.taa.copy');
      return null;
    }

    return current.texture;
  }

  dispose(): void {
    for (const target of this.#history) target.dispose();
    this.#history = [];
    this.#material?.dispose();
    this.#material = null;
    this.#copyMaterial?.dispose();
    this.#copyMaterial = null;
  }

  /* -- internals --------------------------------------------------------- */

  #uploadParams(): void {
    this.#uParams.value.set(
      this.#currentWeight,
      this.#varianceGamma,
      this.#maxVelocity,
      this.#sharpenEnabled ? this.#sharpness : 0,
    );
  }

  #invalidate(): void {
    this.#material?.dispose();
    this.#material = null;
  }

  #ensureHistory(): void {
    if (this.#history.length === 2) {
      for (const target of this.#history) {
        if (target.width !== this.#width || target.height !== this.#height) {
          target.setSize(this.#width, this.#height);
          this.#needsReset = true;
        }
      }
      return;
    }

    for (let i = 0; i < 2; i++) {
      const target = new THREE.RenderTarget(this.#width, this.#height, {
        format: THREE.RGBAFormat,
        type: THREE.HalfFloatType,
        colorSpace: THREE.LinearSRGBColorSpace,
        depthBuffer: false,
        stencilBuffer: false,
        generateMipmaps: false,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        samples: 0,
      });
      target.texture.name = `post.taa.history.${i}`;
      target.texture.wrapS = THREE.ClampToEdgeWrapping;
      target.texture.wrapT = THREE.ClampToEdgeWrapping;
      this.#history.push(target);
    }
    this.#needsReset = true;
  }

  #ensureCopyMaterial(): THREE.NodeMaterial {
    let material = this.#copyMaterial;
    if (material === null) {
      const source = makeSourceNode('taa.copy');
      material = new THREE.NodeMaterial();
      material.name = 'post.taa.copy';
      material.depthTest = false;
      material.depthWrite = false;
      material.fragmentNode = source as unknown as THREE.Node;
      this.#copyMaterial = material;
      this.#copySource = source;
    }
    return material;
  }

  #ensureMaterial(): THREE.NodeMaterial {
    const key = this.#bicubic ? 'bicubic' : 'bilinear';
    if (this.#material !== null && key === this.#structure) return this.#material;

    this.#material?.dispose();
    this.#structure = key;

    const material = new THREE.NodeMaterial();
    material.name = 'post.taa.resolve';
    material.depthTest = false;
    material.depthWrite = false;
    material.fragmentNode = this.#buildResolve() as unknown as THREE.Node;
    this.#material = material;
    return material;
  }

  #buildResolve(): THREE.Node<'vec4'> {
    const uvNode = uv();
    const texel = this.#uTexel;
    const size = this.#uSize;
    const params = this.#uParams;

    const pixel = uvNode.mul(size);
    const centreTexel = pixel.sub(0.5).floor();

    /* -- 1. dilate velocity toward the closest surface ------------------- */

    // `reversedDepth` flips the sense of "closest"; folding it into the sample
    // rather than branching keeps the loop uniform.
    const reversed = this.#uReversedDepth;
    const depthAt = (dx: number, dy: number): THREE.Node<'float'> => {
      const coord = ivec2(centreTexel.add(vec2(dx, dy)));
      const raw = this.#uDepth.load(coord).r;
      return mix(raw, raw.oneMinus(), reversed) as unknown as THREE.Node<'float'>;
    };

    let closestDepth: THREE.Node<'float'> = depthAt(0, 0);
    let closestOffset: THREE.Node<'vec2'> = vec2(0, 0) as unknown as THREE.Node<'vec2'>;
    for (let y = -1; y <= 1; y++) {
      for (let x = -1; x <= 1; x++) {
        if (x === 0 && y === 0) continue;
        const candidate = depthAt(x, y);
        const isCloser = candidate.lessThan(closestDepth);
        closestDepth = isCloser.select(candidate, closestDepth) as unknown as THREE.Node<'float'>;
        closestOffset = isCloser.select(
          vec2(x, y),
          closestOffset,
        ) as unknown as THREE.Node<'vec2'>;
      }
    }

    const velocityTexel = ivec2(centreTexel.add(closestOffset));
    // NDC delta -> texture-space delta. The Y flip is the whole reason this is
    // a named constant in Motion.ts.
    const offsetUV = this.#uVelocity.load(velocityTexel).xy.mul(vec2(0.5, -0.5));
    const historyUV = uvNode.sub(offsetUV) as unknown as THREE.Node<'vec2'>;

    /* -- 2. current frame neighbourhood, in YCoCg ------------------------ */

    const centre = rgbToYCoCg(this.#uCurrent.sample(uvNode).rgb.max(0) as unknown as THREE.Node<'vec3'>);
    let moment1: THREE.Node<'vec3'> = centre;
    let moment2: THREE.Node<'vec3'> = centre.mul(centre) as unknown as THREE.Node<'vec3'>;
    let neighbourMin: THREE.Node<'vec3'> = centre;
    let neighbourMax: THREE.Node<'vec3'> = centre;

    for (let y = -1; y <= 1; y++) {
      for (let x = -1; x <= 1; x++) {
        if (x === 0 && y === 0) continue;
        const offset = vec2(texel.x.mul(x), texel.y.mul(y));
        // `max(0)` guards against a NaN or negative propagating from a
        // misbehaving material; one bad texel would otherwise poison the whole
        // neighbourhood and, through the history, persist for seconds.
        const sample = rgbToYCoCg(
          this.#uCurrent.sample(uvNode.add(offset)).rgb.max(0) as unknown as THREE.Node<'vec3'>,
        );
        moment1 = moment1.add(sample) as unknown as THREE.Node<'vec3'>;
        moment2 = moment2.add(sample.mul(sample)) as unknown as THREE.Node<'vec3'>;
        neighbourMin = minNode(neighbourMin, sample) as unknown as THREE.Node<'vec3'>;
        neighbourMax = maxNode(neighbourMax, sample) as unknown as THREE.Node<'vec3'>;
      }
    }

    const mean = moment1.div(9) as unknown as THREE.Node<'vec3'>;
    const sigma = moment2
      .div(9)
      .sub(mean.mul(mean))
      .max(0)
      .sqrt() as unknown as THREE.Node<'vec3'>;

    /* -- 3. history --------------------------------------------------- */

    const history = this.#bicubic
      ? this.#sampleCatmullRom(historyUV)
      : (this.#uHistory.sample(historyUV).rgb.max(0) as unknown as THREE.Node<'vec3'>);
    const historyYCoCg = rgbToYCoCg(history);

    // Variance clipping bounds, further constrained by the hard neighbourhood
    // box so that a degenerate (zero-variance) neighbourhood still rejects.
    const gamma = params.y;
    const spread = sigma.mul(gamma);
    const clipMin = maxNode(mean.sub(spread), neighbourMin) as unknown as THREE.Node<'vec3'>;
    const clipMax = minNode(mean.add(spread), neighbourMax) as unknown as THREE.Node<'vec3'>;
    const clipped = clipToAABB(historyYCoCg, clipMin, clipMax);

    /* -- 4. blend weight ------------------------------------------------- */

    const velocityTexels = offsetUV.mul(size).length();
    const motionFactor = velocityTexels.div(params.z).saturate();

    const inBounds = historyUV
      .greaterThanEqual(vec2(0, 0))
      .all()
      .and(historyUV.lessThanEqual(vec2(1, 1)).all());
    const usable = inBounds.and(this.#uHistoryValid.greaterThan(0.5));

    // Karis anti-flicker: weight by 1/(1+luma) so an unstable bright sample
    // cannot dominate the integration.
    const baseCurrent = params.x.add(motionFactor).saturate();
    const currentWeight = baseCurrent.div(centre.x.max(0).add(1));
    const historyWeight = baseCurrent.oneMinus().div(clipped.x.max(0).add(1));

    const blended = centre
      .mul(currentWeight)
      .add(clipped.mul(historyWeight))
      .div(maxNode(currentWeight.add(historyWeight), float(1e-5))) as unknown as THREE.Node<'vec3'>;

    const resolvedYCoCg = usable.select(blended, centre) as unknown as THREE.Node<'vec3'>;

    /* -- 5. sharpen ------------------------------------------------------ */

    // Unsharp mask against the neighbourhood mean, which is already computed.
    // Clamping to the neighbourhood box afterwards is what keeps it from
    // ringing into a halo on high-contrast edges.
    const sharpened = resolvedYCoCg
      .add(resolvedYCoCg.sub(mean).mul(params.w))
      .clamp(neighbourMin, neighbourMax) as unknown as THREE.Node<'vec3'>;

    const output = yCoCgToRgb(sharpened).max(0) as unknown as THREE.Node<'vec3'>;
    return vec4(output, 1) as unknown as THREE.Node<'vec4'>;
  }

  /**
   * 5-tap Catmull-Rom reconstruction of the history.
   *
   * The separable 4x4 Catmull-Rom kernel is collapsed to five bilinear fetches
   * by exploiting the fact that the two inner weights per axis can be fetched as
   * one weighted bilinear sample, and that the four extreme corners contribute
   * little enough to drop. Karis, SIGGRAPH 2014; the compact form is MJP's.
   */
  #sampleCatmullRom(uvNode: THREE.Node<'vec2'>): THREE.Node<'vec3'> {
    const size = this.#uSize;
    const texel = this.#uTexel;

    const samplePos = uvNode.mul(size);
    const texPos1 = samplePos.sub(0.5).floor().add(0.5);
    const f = samplePos.sub(texPos1);

    const w0 = f.mul(f.mul(f.mul(-0.5).add(1)).add(-0.5));
    const w1 = f.mul(f).mul(f.mul(1.5).add(-2.5)).add(1);
    const w2 = f.mul(f.mul(f.mul(-1.5).add(2)).add(0.5));
    const w3 = f.mul(f).mul(f.mul(0.5).add(-0.5));

    const w12 = w1.add(w2);
    const offset12 = w2.div(w12.add(1e-6));

    const texPos0 = texPos1.sub(1).mul(texel);
    const texPos3 = texPos1.add(2).mul(texel);
    const texPos12 = texPos1.add(offset12).mul(texel);

    const fetch = (coord: THREE.Node<'vec2'>): THREE.Node<'vec3'> =>
      this.#uHistory.sample(coord).rgb.max(0) as unknown as THREE.Node<'vec3'>;

    const weightA = w12.x.mul(w0.y);
    const weightB = w0.x.mul(w12.y);
    const weightC = w12.x.mul(w12.y);
    const weightD = w3.x.mul(w12.y);
    const weightE = w12.x.mul(w3.y);

    const sum = fetch(vec2(texPos12.x, texPos0.y))
      .mul(weightA)
      .add(fetch(vec2(texPos0.x, texPos12.y)).mul(weightB))
      .add(fetch(vec2(texPos12.x, texPos12.y)).mul(weightC))
      .add(fetch(vec2(texPos3.x, texPos12.y)).mul(weightD))
      .add(fetch(vec2(texPos12.x, texPos3.y)).mul(weightE));

    const total = weightA.add(weightB).add(weightC).add(weightD).add(weightE);
    return sum.div(maxNode(total, float(1e-5))).max(0) as unknown as THREE.Node<'vec3'>;
  }
}

/* ------------------------------------------------------------------------- *
 * Spatial fallbacks
 * ------------------------------------------------------------------------- */

/**
 * Shared plumbing for the two spatial AA passes.
 *
 * Both receive sRGB-encoded values (see `CompositePass.setEncodeOutput`) and
 * must hand back linear light, because the renderer applies the output transfer
 * function on the final write.
 */
abstract class SpatialAAPass implements PostPass {
  abstract readonly id: string;
  readonly kind = 'chain' as const;
  readonly outputDomain = 'ldr' as const;

  enabled = false;

  protected readonly source = makeSourceNode('aa.source');
  protected material: THREE.NodeMaterial | null = null;

  isAvailable(_quality: QualityTier, _capabilities: PostCapabilities): boolean {
    return true;
  }

  configure(_quality: QualityTier, _capabilities: PostCapabilities): void {
    /* no tier-dependent structure */
  }

  setSize(_width: number, _height: number): void {
    /* the node reads its own texture size */
  }

  render(frame: PostFrame): THREE.Texture | null {
    const material = this.ensureMaterial();
    if (material === null) return frame.input;
    this.source.value = frame.input;
    frame.blit(material, frame.output, this.id);
    return null;
  }

  dispose(): void {
    this.material?.dispose();
    this.material = null;
  }

  protected abstract build(): THREE.Node<'vec4'>;

  protected ensureMaterial(): THREE.NodeMaterial | null {
    if (this.material !== null) return this.material;
    try {
      const material = new THREE.NodeMaterial();
      material.name = this.id;
      material.depthTest = false;
      material.depthWrite = false;
      material.fragmentNode = this.build() as unknown as THREE.Node;
      this.material = material;
      return material;
    } catch (error) {
      // A failure here is a shader-graph problem in an addon, not something the
      // player should experience as a black screen. Degrade to no AA.
      console.error(`[${this.id}] failed to build; falling back to no AA:`, error);
      return null;
    }
  }
}

/** FXAA 3.11 (Lottes, NVIDIA 2011) via three.js's TSL implementation. */
export class FxaaPass extends SpatialAAPass {
  override readonly id = 'post.fxaa';

  protected override build(): THREE.Node<'vec4'> {
    return toLinear(fxaa(this.source) as unknown as THREE.Node<'vec4'>);
  }
}

/**
 * SMAA 1x (Jimenez et al., 2012) via three.js's TSL implementation.
 *
 * Higher quality than FXAA on near-horizontal and near-vertical edges, at the
 * cost of two extra render targets and two extra passes that three manages
 * internally. Its area/search lookup textures are base64 data URIs inside the
 * addon, so it works with no network access — which in this project's build
 * environment is a hard requirement, not a convenience.
 */
export class SmaaPass extends SpatialAAPass {
  override readonly id = 'post.smaa';

  protected override build(): THREE.Node<'vec4'> {
    return toLinear(smaa(this.source) as unknown as THREE.Node<'vec4'>);
  }
}

/* ------------------------------------------------------------------------- *
 * Helpers
 * ------------------------------------------------------------------------- */

/**
 * Undo the sRGB encode the composite applied for the AA pass's benefit, so the
 * renderer's own output transfer function is the only encode in the frame.
 */
function toLinear(color: THREE.Node<'vec4'>): THREE.Node<'vec4'> {
  const linear = sRGBTransferEOTF(
    color.rgb as unknown as THREE.Node<'vec3'>,
  ) as unknown as THREE.Node<'vec3'>;
  return vec4(linear, 1) as unknown as THREE.Node<'vec4'>;
}

function makeSourceNode(name: string): THREE.TextureNode {
  const placeholder = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  placeholder.needsUpdate = true;
  const node = textureNode(placeholder);
  node.name = name;
  return node;
}
