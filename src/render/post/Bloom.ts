/**
 * @module render/post/Bloom
 *
 * Progressive dual-filter bloom: a downsample pyramid built with a 13-tap
 * partial Karis average, then an upsample pyramid built with a 9-tap tent,
 * blended level by level so the total energy is preserved.
 *
 * Technique is Jimenez, *"Next Generation Post Processing in Call of Duty:
 * Advanced Warfare"*, SIGGRAPH 2014 (Advances in Real-Time Rendering), which is
 * itself a refinement of Kawase's dual filter. It is the standard because it
 * gets a very wide, very smooth kernel for a bounded cost: each level halves
 * the resolution, so the whole pyramid costs at most `4/3` of one half-
 * resolution pass on the way down and the same on the way up, regardless of how
 * wide the final glow is.
 *
 * ---
 *
 * ## Why there is no threshold by default
 *
 * The usual bloom implementation subtracts a threshold, blurs what is left, and
 * adds it back. That has two failure modes and this project cannot afford
 * either: the subtraction **clips** — a pixel just under the threshold
 * contributes nothing and one just over contributes almost nothing, so a slowly
 * brightening surface pops into bloom — and the additive composite is not
 * energy conserving, so raising the bloom slider raises the exposure of the
 * whole image.
 *
 * The scheme here instead blurs the *whole* image and composites with
 * `mix(color, bloom, intensity)`. Because a blur preserves the local mean, the
 * mix is an identity wherever the image is locally flat: a uniformly dark scene
 * comes out exactly as dark as it went in, and the effect only becomes visible
 * where there is local contrast — which is exactly "glow around bright things"
 * and never "global haze". Energy is conserved by construction, since the
 * downsample weights sum to 1, the tent weights sum to 1, and the level blend
 * is a convex combination.
 *
 * `threshold`/`knee` remain available as an artistic override. They apply a
 * *soft* quadratic knee (Unity/Karis) rather than a hard subtract, so the
 * response is C1-continuous through the knee and nothing pops. Using them
 * trades away the energy-conservation argument above; the default is 0.
 *
 * ## Firefly control
 *
 * A single very bright pixel — a specular hit on a wet cobble, a spark — has
 * enough energy to dominate an entire mip level once it is spread over 64
 * pixels, and because it is temporally unstable it flickers. The first
 * downsample therefore weights each 2x2 group by `1 / (1 + luma)` before
 * averaging (Karis, *"Graphic Rants: Tone Mapping"* / the partial Karis average
 * used in the COD talk), which caps the contribution of any single sample
 * without touching the rest of the image. It is applied only at the first
 * level: after that the outliers are already averaged away, and applying it
 * again would visibly darken large bright areas.
 *
 * ## Upsample blending
 *
 * The upsample writes `mip[i] = mix(mip[i], tent(mip[i+1]), spread)` using
 * fixed-function alpha blending with `autoClear` disabled, rather than a second
 * set of ping-pong targets. That keeps the pyramid at exactly one render target
 * per level — `4/3 * (w/2) * (h/2) * 8` bytes total — and, because `mix` with a
 * constant is a convex combination, keeps the mean intact at every level. The
 * resulting per-level weights are `(1-s), s(1-s), s²(1-s), ...`, a geometric
 * series summing to 1.
 */

import * as THREE from 'three/webgpu';

import { trackRenderTarget } from '../MemoryReport';
import { float, texture as textureNode, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';

import type { PostCapabilities, PostFrame, PostPass, QualityTier } from './PostStack';

export interface BloomOptions {
  /**
   * Fraction of the image replaced by its own blur, in `[0, 1]`. This is
   * physically the fraction of incident light a lens scatters rather than
   * focuses; real lenses are around 1-3%, and film/anamorphic glass more.
   * Default 0.055.
   */
  intensity?: number;
  /**
   * Level-to-level blend in `(0, 1)`. Higher pushes energy into the wider,
   * lower-frequency levels — a bigger, softer halo. Default 0.62.
   */
  spread?: number;
  /** Tent kernel radius in source texels. 1 is the textbook filter. Default 1. */
  radius?: number;
  /** Pyramid depth. Overridden by the quality tier. Default 6. */
  mipCount?: number;
  /**
   * Optional soft-knee threshold in scene-referred luminance. 0 disables it and
   * keeps the energy-conserving behaviour described above.
   */
  threshold?: number;
  /** Width of the soft knee around the threshold. Default 0.6. */
  knee?: number;
}

/**
 * The 13-tap downsample footprint, in source texels.
 *
 * Five overlapping 2x2 boxes: the centre box is weighted 0.5 and the four
 * corner boxes 0.125 each, which reproduces a 4x4 tent while costing 13 bilinear
 * fetches instead of 16 point fetches and — crucially — has no aliasing holes
 * when the source is exactly twice the destination resolution.
 */
const DOWNSAMPLE_TAPS: ReadonlyArray<readonly [number, number]> = [
  [-2, -2], [0, -2], [2, -2],
  [-2, 0], [0, 0], [2, 0],
  [-2, 2], [0, 2], [2, 2],
  [-1, -1], [1, -1], [-1, 1], [1, 1],
];

/** Tent (Bartlett) 3x3 weights, normalised. */
const TENT_TAPS: ReadonlyArray<readonly [number, number, number]> = [
  [-1, -1, 1 / 16], [0, -1, 2 / 16], [1, -1, 1 / 16],
  [-1, 0, 2 / 16], [0, 0, 4 / 16], [1, 0, 2 / 16],
  [-1, 1, 1 / 16], [0, 1, 2 / 16], [1, 1, 1 / 16],
];

const REC709_LUMA: readonly [number, number, number] = [0.2126, 0.7152, 0.0722];

/** `uniform(new Vector2())`, named once so the pass signatures stay readable. */
type Vec2Uniform = ReturnType<typeof uniform<'vec2'>>;

interface MipLevel {
  target: THREE.RenderTarget;
  downMaterial: THREE.NodeMaterial;
  downSource: THREE.TextureNode;
  downTexel: Vec2Uniform;
  upMaterial: THREE.NodeMaterial;
  upSource: THREE.TextureNode;
  upTexel: Vec2Uniform;
}

export class BloomPass implements PostPass {
  readonly id = 'post.bloom';
  readonly kind = 'producer' as const;
  readonly outputDomain = 'hdr' as const;

  enabled = true;

  #intensity: number;
  #spread: number;
  #radius: number;
  #requestedMips: number;
  #threshold: number;
  #knee: number;

  #width = 1;
  #height = 1;
  #levels: MipLevel[] = [];
  #dirty = true;

  readonly #uThreshold = uniform(new THREE.Vector3(0, 0, 0));
  readonly #uSpread = uniform(0.62);

  constructor(options: BloomOptions = {}) {
    this.#intensity = options.intensity ?? 0.055;
    this.#spread = THREE.MathUtils.clamp(options.spread ?? 0.62, 0.05, 0.95);
    this.#radius = options.radius ?? 1;
    this.#requestedMips = options.mipCount ?? 6;
    this.#threshold = options.threshold ?? 0;
    this.#knee = options.knee ?? 0.6;
    this.#uSpread.value = this.#spread;
    this.#uploadThreshold();
  }

  /* -- configuration ----------------------------------------------------- */

  /** Fraction of the image replaced by its blur. Read by the composite pass. */
  get intensity(): number {
    return this.#intensity;
  }

  setIntensity(value: number): void {
    this.#intensity = THREE.MathUtils.clamp(value, 0, 1);
  }

  setSpread(value: number): void {
    this.#spread = THREE.MathUtils.clamp(value, 0.05, 0.95);
    this.#uSpread.value = this.#spread;
  }

  setThreshold(threshold: number, knee = this.#knee): void {
    this.#threshold = Math.max(0, threshold);
    this.#knee = Math.max(1e-4, knee);
    this.#uploadThreshold();
  }

  setMipCount(count: number): void {
    const next = THREE.MathUtils.clamp(Math.floor(count), 1, 9);
    if (next === this.#requestedMips) return;
    this.#requestedMips = next;
    this.#dirty = true;
  }

  /** Half-resolution texture holding the finished pyramid. */
  get texture(): THREE.Texture | null {
    return this.#levels[0]?.target.texture ?? null;
  }

  get bytes(): number {
    let total = 0;
    for (const level of this.#levels) {
      total += level.target.width * level.target.height * 8;
    }
    return total;
  }

  /* -- PostPass ---------------------------------------------------------- */

  isAvailable(_quality: QualityTier, capabilities: PostCapabilities): boolean {
    // Without a float-renderable target the pyramid would quantise a
    // scene-referred signal to 8 bits and band catastrophically.
    return capabilities.halfFloat;
  }

  configure(_quality: QualityTier, _capabilities: PostCapabilities): void {
    // Mip count is driven through `setMipCount` by the tier table.
  }

  setSize(width: number, height: number): void {
    if (width === this.#width && height === this.#height) return;
    this.#width = width;
    this.#height = height;
    this.#dirty = true;
  }

  render(frame: PostFrame): void {
    this.#ensureLevels();
    const levels = this.#levels;
    if (levels.length === 0) return;

    const renderer = frame.renderer;
    const previousAutoClear = (renderer as unknown as { autoClear: boolean }).autoClear;

    // -- downsample ------------------------------------------------------
    // Level 0 reads the scene at full resolution; every other level reads the
    // level above it.
    for (let i = 0; i < levels.length; i++) {
      const level = levels[i];
      if (level === undefined) continue;
      const previous = levels[i - 1];
      const source = i === 0 ? frame.input : previous?.target.texture;
      if (source === undefined) continue;
      const sourceWidth = i === 0 ? frame.width : (previous?.target.width ?? 1);
      const sourceHeight = i === 0 ? frame.height : (previous?.target.height ?? 1);
      level.downSource.value = source;
      level.downTexel.value.set(1 / Math.max(1, sourceWidth), 1 / Math.max(1, sourceHeight));
      frame.blit(level.downMaterial, level.target, `bloom.down.${i}`);
    }

    // -- upsample --------------------------------------------------------
    // Walk back up, blending each level into the one above with fixed-function
    // alpha so no extra targets are needed. `autoClear` must be off or the
    // destination's own content — the thing being blended into — is erased.
    (renderer as unknown as { autoClear: boolean }).autoClear = false;
    try {
      for (let i = levels.length - 1; i >= 1; i--) {
        const source = levels[i];
        const destination = levels[i - 1];
        if (source === undefined || destination === undefined) continue;
        destination.upSource.value = source.target.texture;
        destination.upTexel.value.set(1 / source.target.width, 1 / source.target.height);
        frame.blit(destination.upMaterial, destination.target, `bloom.up.${i}`);
      }
    } finally {
      (renderer as unknown as { autoClear: boolean }).autoClear = previousAutoClear;
    }
  }

  dispose(): void {
    this.#disposeLevels();
  }

  /* -- internals --------------------------------------------------------- */

  #uploadThreshold(): void {
    // Karis/Unity soft knee, packed as (threshold, threshold - knee, 4 * knee).
    // The shader forms `max(0, luma - t + knee)^2 / (4*knee)` and picks the
    // larger of that and `luma - t`, which is C1 across the knee.
    this.#uThreshold.value.set(this.#threshold, this.#threshold - this.#knee, 4 * this.#knee);
  }

  #maxMips(): number {
    // Stop before a level would be smaller than 2 texels on either axis: a 1x1
    // tent tap set degenerates and the bilinear fetch starts clamping.
    const smallest = Math.min(this.#width, this.#height);
    let levels = 0;
    let size = smallest;
    while (size >= 4 && levels < 9) {
      size = Math.floor(size / 2);
      levels++;
    }
    return Math.max(1, levels);
  }

  #ensureLevels(): void {
    if (!this.#dirty) return;
    this.#dirty = false;

    const count = Math.min(this.#requestedMips, this.#maxMips());
    if (this.#levels.length !== count) {
      this.#disposeLevels();
      this.#levels = [];
      for (let i = 0; i < count; i++) this.#levels.push(this.#createLevel(i));
      return;
    }

    for (let i = 0; i < count; i++) {
      const level = this.#levels[i];
      if (level === undefined) continue;
      const { width, height } = this.#levelSize(i);
      level.target.setSize(width, height);
    }
  }

  #levelSize(index: number): { width: number; height: number } {
    const divisor = 2 ** (index + 1);
    return {
      width: Math.max(1, Math.floor(this.#width / divisor)),
      height: Math.max(1, Math.floor(this.#height / divisor)),
    };
  }

  #createLevel(index: number): MipLevel {
    const { width, height } = this.#levelSize(index);

    const target = new THREE.RenderTarget(width, height, {
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
    target.texture.name = `post.bloom.mip${index}`;
    trackRenderTarget(target);
    target.texture.wrapS = THREE.ClampToEdgeWrapping;
    target.texture.wrapT = THREE.ClampToEdgeWrapping;

    const downSource = makeSourceNode(`bloom.down.${index}`);
    const downTexel = uniform(new THREE.Vector2(1 / Math.max(1, width), 1 / Math.max(1, height)));
    const downMaterial = new THREE.NodeMaterial();
    downMaterial.name = `post.bloom.down.${index}`;
    downMaterial.depthTest = false;
    downMaterial.depthWrite = false;
    downMaterial.fragmentNode = this.#buildDownsample(
      downSource,
      downTexel,
      index === 0,
    ) as unknown as THREE.Node;

    const upSource = makeSourceNode(`bloom.up.${index}`);
    const upTexel = uniform(new THREE.Vector2(1, 1));
    const upMaterial = new THREE.NodeMaterial();
    upMaterial.name = `post.bloom.up.${index}`;
    upMaterial.depthTest = false;
    upMaterial.depthWrite = false;
    upMaterial.transparent = true;
    upMaterial.blending = THREE.NormalBlending;
    upMaterial.premultipliedAlpha = false;
    upMaterial.fragmentNode = this.#buildUpsample(upSource, upTexel) as unknown as THREE.Node;

    return { target, downMaterial, downSource, downTexel, upMaterial, upSource, upTexel };
  }

  /**
   * 13-tap downsample.
   *
   * @param karis apply the partial Karis average and the optional soft-knee
   *   threshold. Only true for the first level, where the source is the
   *   full-resolution scene and therefore the only place fireflies exist.
   */
  #buildDownsample(
    source: THREE.TextureNode,
    texel: Vec2Uniform,
    karis: boolean,
  ): THREE.Node<'vec4'> {
    const uvNode = screenUv();
    const taps: THREE.Node<'vec3'>[] = DOWNSAMPLE_TAPS.map(([x, y]) => {
      const offset = vec2(texel.x.mul(x), texel.y.mul(y));
      return source.sample(uvNode.add(offset)).rgb.max(0) as unknown as THREE.Node<'vec3'>;
    });

    const [a, b, c, d, e, f, g, h, i, j, k, l, m] = taps as [
      THREE.Node<'vec3'>, THREE.Node<'vec3'>, THREE.Node<'vec3'>,
      THREE.Node<'vec3'>, THREE.Node<'vec3'>, THREE.Node<'vec3'>,
      THREE.Node<'vec3'>, THREE.Node<'vec3'>, THREE.Node<'vec3'>,
      THREE.Node<'vec3'>, THREE.Node<'vec3'>, THREE.Node<'vec3'>,
      THREE.Node<'vec3'>,
    ];

    const boxes: THREE.Node<'vec3'>[] = [
      average4(j, k, l, m), // centre
      average4(a, b, d, e),
      average4(b, c, e, f),
      average4(d, e, g, h),
      average4(e, f, h, i),
    ];
    const weights = [0.5, 0.125, 0.125, 0.125, 0.125];

    let accumulated: THREE.Node<'vec3'> = vec3(0) as unknown as THREE.Node<'vec3'>;

    if (karis) {
      // Weight each box by 1/(1+luma) *and* by its kernel weight, then divide
      // by the summed weight. This is an average, not a sum, so a box with a
      // firefly in it contributes proportionally less without darkening the
      // result where no firefly exists.
      let weightSum: THREE.Node<'float'> = float(1e-6) as unknown as THREE.Node<'float'>;
      for (let index = 0; index < boxes.length; index++) {
        const box = boxes[index];
        if (box === undefined) continue;
        const luma = box.dot(vec3(...REC709_LUMA));
        const karisWeight = float(1).div(luma.add(1));
        const weight = karisWeight.mul(weights[index] ?? 0);
        accumulated = accumulated.add(box.mul(weight)) as unknown as THREE.Node<'vec3'>;
        weightSum = weightSum.add(weight) as unknown as THREE.Node<'float'>;
      }
      accumulated = accumulated.div(weightSum) as unknown as THREE.Node<'vec3'>;
      accumulated = this.#applySoftKnee(accumulated);
    } else {
      for (let index = 0; index < boxes.length; index++) {
        const box = boxes[index];
        if (box === undefined) continue;
        accumulated = accumulated.add(box.mul(weights[index] ?? 0)) as unknown as THREE.Node<'vec3'>;
      }
    }

    return vec4(accumulated, 1) as unknown as THREE.Node<'vec4'>;
  }

  /**
   * Soft-knee highlight isolation.
   *
   * `max(0, luma - t + knee)^2 / (4 * knee)` is the quadratic that joins the
   * "everything passes" and "nothing passes" branches with matching value and
   * derivative, so a surface brightening through the threshold ramps in instead
   * of popping. Scaling the *colour* by `contribution / luma` rather than
   * subtracting keeps hue and saturation intact, which a straight subtract does
   * not.
   */
  #applySoftKnee(color: THREE.Node<'vec3'>): THREE.Node<'vec3'> {
    if (this.#threshold <= 0) return color;
    const luma = color.dot(vec3(...REC709_LUMA)).max(1e-5);
    const soft = luma.sub(this.#uThreshold.y).max(0);
    const softContribution = soft.mul(soft).div(this.#uThreshold.z.add(1e-5));
    const contribution = softContribution.max(luma.sub(this.#uThreshold.x)).max(0);
    return color.mul(contribution.div(luma)) as unknown as THREE.Node<'vec3'>;
  }

  /**
   * 9-tap tent upsample.
   *
   * Alpha carries the blend weight, so the fixed-function blender performs
   * `dst = tent * spread + dst * (1 - spread)` without a read-modify-write in
   * the shader.
   */
  #buildUpsample(
    source: THREE.TextureNode,
    texel: Vec2Uniform,
  ): THREE.Node<'vec4'> {
    const uvNode = screenUv();
    let accumulated: THREE.Node<'vec3'> = vec3(0) as unknown as THREE.Node<'vec3'>;
    for (const [x, y, weight] of TENT_TAPS) {
      const offset = vec2(texel.x.mul(x * this.#radius), texel.y.mul(y * this.#radius));
      const tap = source.sample(uvNode.add(offset)).rgb.max(0);
      accumulated = accumulated.add(tap.mul(weight)) as unknown as THREE.Node<'vec3'>;
    }
    return vec4(accumulated, this.#uSpread) as unknown as THREE.Node<'vec4'>;
  }

  #disposeLevels(): void {
    for (const level of this.#levels) {
      level.target.dispose();
      level.downMaterial.dispose();
      level.upMaterial.dispose();
    }
    this.#levels = [];
  }
}

/* ------------------------------------------------------------------------- *
 * Local helpers
 * ------------------------------------------------------------------------- */

function average4(
  a: THREE.Node<'vec3'>,
  b: THREE.Node<'vec3'>,
  c: THREE.Node<'vec3'>,
  d: THREE.Node<'vec3'>,
): THREE.Node<'vec3'> {
  return a.add(b).add(c).add(d).mul(0.25) as unknown as THREE.Node<'vec3'>;
}

function screenUv(): THREE.Node<'vec2'> {
  return uv() as unknown as THREE.Node<'vec2'>;
}

function makeSourceNode(name: string): THREE.TextureNode {
  const placeholder = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  placeholder.needsUpdate = true;
  const node = textureNode(placeholder);
  node.name = name;
  return node;
}
