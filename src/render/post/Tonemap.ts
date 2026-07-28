/**
 * @module render/post/Tonemap
 *
 * Exposure, the tone curve, and the single full-resolution draw that turns a
 * scene-referred HDR buffer into a display-referred image.
 *
 * ---
 *
 * ## Exposure is measured in stops
 *
 * Exposure is a linear multiplier on scene radiance, but it is *authored* in
 * stops, because that is the only scale on which "a bit brighter" means the
 * same thing in a cellar and on a hilltop.
 *
 * A word on units, because it determines every default in this file. The
 * project's lighting rig (`render/Lighting.ts`, `render/Sky.ts`) works in
 * **relative render units**, not cd/m²: the key light sits at intensity ~3 and
 * a mid-grey surface under overcast sky lands near 0.18. So the neutral
 * exposure is 1.0 and the useful window is roughly ±5 stops around it. The
 * photographic conversions are still here — {@link exposureFromEV100},
 * {@link ev100FromLuminance}, {@link ev100FromCamera} — and
 * {@link TonemapOptions.ev100} routes through them, so the day the lighting is
 * recalibrated to lux nothing in this module has to change. What must not
 * happen is quoting EV100 12 for "overcast" while the renderer's overcast is
 * numerically 1.0; that is a 4000x error and it renders black.
 *
 * ## Auto exposure
 *
 * Metering is a GPU-side geometric mean of luminance, targeting middle grey
 * (0.18) — the classic key-value scheme from Reinhard et al., *"Photographic
 * Tone Reproduction for Digital Images"* (SIGGRAPH 2002, §3.1), which is
 * unit-agnostic by construction: it asks only "how much do I have to scale the
 * image so its average lands on grey".
 *
 * The HDR buffer is reduced to 64x64 with a 16-tap box (so a single bright
 * pixel cannot dominate a bin), then 64 → 16 → 4 → 1 with further 16-tap boxes,
 * accumulating `log2(luminance)` because the *geometric* mean is the
 * perceptually meaningful average — an arithmetic mean is dragged around by
 * specular highlights and produces the classic "camera hunts every time the sun
 * catches a puddle" behaviour.
 *
 * Adaptation is an exponential approach evaluated in a 1x1 ping-pong target:
 *
 * ```
 * L(t+dt) = L_target + (L(t) − L_target) · e^(−dt / τ)
 * ```
 *
 * with separate time constants for brightening and darkening (the human visual
 * system adapts to darkness far more slowly than to light: rod adaptation runs
 * to minutes, cone adaptation a few seconds — τ = 0.5 s / 1.6 s here, which is
 * a game-legible compression of that asymmetry). The result is clamped to an
 * exposure window so that walking into a pitch-dark cellar does not gain the
 * image up to a wall of noise, which is the single most common auto-exposure
 * failure, and so that a cave stays *dark* — an auto-exposure that always
 * normalises to grey destroys the difference between places, which for this
 * game is most of the mood.
 *
 * There is **no CPU readback anywhere in this chain.** A `readPixels` of even
 * one texel forces a pipeline flush and costs more than every other pass in
 * this module combined.
 *
 * ## The curve: AgX
 *
 * AgX (Troy Sobotka, 2022; the default view transform in Blender 4.x) rather
 * than Reinhard or a bare ACES fit. Its property that matters here is
 * **hue-preserving highlight rolloff**: as a channel saturates, AgX walks the
 * chromaticity toward white along a path that keeps the hue, so a torch flame
 * goes orange → pale orange → white. Reinhard and the common ACES fits both
 * shift hue as they clip — ACES notoriously rotates saturated reds toward
 * orange and saturated blues toward purple — which for a game whose entire
 * palette rests on warm firelight against cold blue-grey is exactly the wrong
 * failure.
 *
 * The implementation follows the standard chain: Rec.709 → Rec.2020 → AgX
 * inset, log2 encode over `[-12.47, +4.03]` EV, the 6th-order polynomial fit of
 * the AgX sigmoid, an ASC CDL *look* applied in that display-log space, the
 * outset matrix, and back to Rec.709. The look stage is where the project's
 * contrast lives; putting it before the outset is what Blender does and is why
 * an AgX look can add contrast without the hue skew a post-hoc contrast
 * adjustment produces.
 *
 * A Hill ACES fit and a plain clamp are available for A/B comparison.
 *
 * ## References
 *
 * - S. Lagarde, C. de Rousiers, *"Moving Frostbite to PBR"*, SIGGRAPH 2014.
 * - T. Sobotka, *AgX*, https://github.com/sobotka/AgX (and Blender's port).
 * - S. Hill, ACES fit, from *"Real-Time Rendering"* 4th ed. supplements.
 * - E. Reinhard et al., *"Photographic Tone Reproduction for Digital Images"*,
 *   SIGGRAPH 2002 — log-average key value.
 * - K. Pettineo, *"Average Luminance Calculation Using A Compute Shader"*, 2012
 *   — the log-average metering scheme, adapted here to a raster reduction so
 *   the WebGL2 fallback runs the same code.
 */

import * as THREE from 'three/webgpu';
import {
  float,
  mat3,
  mix,
  sRGBTransferEOTF,
  sRGBTransferOETF,
  smoothstep,
  texture as textureNode,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import type { BloomPass } from './Bloom';
import type { ColorGrade } from './ColorGrade';
import { safeNodeName } from './nodeNames';
import type { PostCapabilities, PostFrame, PostPass, QualityTier } from './PostStack';

/* ------------------------------------------------------------------------- *
 * Public vocabulary
 * ------------------------------------------------------------------------- */

export type ToneCurve = 'agx' | 'aces' | 'none';

/** ASC CDL look applied inside AgX's display-log space. */
export interface AgXLook {
  readonly slope: readonly [number, number, number];
  readonly offset: readonly [number, number, number];
  readonly power: readonly [number, number, number];
  readonly saturation: number;
}

/** Sobotka's reference looks, plus the one this project ships. */
export const AGX_LOOKS: Readonly<Record<'none' | 'punchy' | 'golden' | 'grimdark', AgXLook>> = {
  none: { slope: [1, 1, 1], offset: [0, 0, 0], power: [1, 1, 1], saturation: 1 },
  punchy: { slope: [1, 1, 1], offset: [0, 0, 0], power: [1.35, 1.35, 1.35], saturation: 1.4 },
  golden: {
    slope: [1.0, 0.9, 0.5],
    offset: [0, 0, 0],
    power: [0.8, 0.8, 0.8],
    saturation: 0.8,
  },
  // Restrained: enough contrast that the image is not milky, not so much that
  // the shadows crush and take the colour separation with them.
  grimdark: {
    slope: [1.0, 1.0, 1.0],
    offset: [0, 0, 0],
    power: [1.16, 1.16, 1.2],
    saturation: 1.08,
  },
};

export interface TonemapOptions {
  curve?: ToneCurve;
  /** AgX look, by name or as an explicit CDL. Default `'grimdark'`. */
  look?: keyof typeof AGX_LOOKS | AgXLook;
  /**
   * Manual exposure as a linear multiplier on scene radiance. Default 1, which
   * is the value the project's lighting and sky are authored against.
   */
  exposure?: number;
  /**
   * Manual exposure expressed photographically instead. Overrides `exposure`
   * when supplied. Only meaningful if the lighting rig is calibrated in cd/m².
   */
  ev100?: number;
  /** Additional stops applied on top of manual *or* metered exposure. */
  exposureCompensation?: number;
  autoExposure?: boolean;
  /**
   * Display-referred value the metered average is driven toward. 0.18 is the
   * photographic grey card and the right answer almost always.
   */
  middleGrey?: number;
  /**
   * Metered exposure is clamped to this multiplier window. The default
   * `[0.25, 6]` is ±2.6 stops around neutral: enough to keep a cellar readable
   * without erasing the difference between a cellar and a hilltop.
   */
  autoExposureRange?: readonly [number, number];
  /** Seconds to adapt to a brighter scene. Default 0.5. */
  adaptationUp?: number;
  /** Seconds to adapt to a darker scene. Default 1.6. */
  adaptationDown?: number;
}

/* ------------------------------------------------------------------------- *
 * Exposure math — exported for tests
 * ------------------------------------------------------------------------- */

/**
 * Linear exposure multiplier for an exposure value at ISO 100.
 *
 * `1 / (1.2 · 2^EV100)`. The `1.2` is the ISO saturation-based sensitivity
 * constant; it is what makes an 18% grey card land on 0.18 after the transform
 * when the scene is metered correctly.
 */
export function exposureFromEV100(ev100: number): number {
  return 1 / (1.2 * 2 ** ev100);
}

/**
 * EV100 that would meter the given average luminance to middle grey.
 *
 * `EV100 = log2(L · S / K)` with S = 100 (ISO) and K = 12.5 (the reflected-light
 * calibration constant Canon and Nikon both use).
 */
export function ev100FromLuminance(averageLuminance: number): number {
  return Math.log2(Math.max(averageLuminance, 1e-6) * (100 / 12.5));
}

/**
 * EV100 for a physical camera setting. Not used by the renderer, but it is the
 * sanity check that the two functions above are mutually consistent, and it is
 * how a cinematic camera would be authored.
 */
export function ev100FromCamera(aperture: number, shutterSeconds: number, iso: number): number {
  return Math.log2((aperture * aperture) / shutterSeconds) - Math.log2(iso / 100);
}

/**
 * Exponential adaptation step: how far to move toward the target in `dt`.
 *
 * Frame-rate independent by construction — `1 - e^(-dt/τ)` composes correctly
 * over sub-steps, unlike the `lerp(a, b, k)` with a constant `k` that most
 * engines ship and that silently adapts twice as fast at 120 Hz.
 */
export function adaptationWeight(dt: number, tau: number): number {
  if (tau <= 0) return 1;
  return 1 - Math.exp(-dt / tau);
}

/* ------------------------------------------------------------------------- *
 * Tone curves, as TSL
 * ------------------------------------------------------------------------- */

const LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
  vec3(0.6274, 0.0691, 0.0164),
  vec3(0.3293, 0.9195, 0.088),
  vec3(0.0433, 0.0113, 0.8956),
);

const LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
  vec3(1.6605, -0.1246, -0.0182),
  vec3(-0.5876, 1.1329, -0.1006),
  vec3(-0.0728, -0.0083, 1.1187),
);

const AGX_INSET = mat3(
  vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
  vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
  vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859),
);

const AGX_OUTSET = mat3(
  vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
  vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
  vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
);

const AGX_MIN_EV = -12.47393;
const AGX_MAX_EV = 4.026069;

const REC709_LUMA = vec3(0.2126, 0.7152, 0.0722);

/**
 * Meter weight at the corner of the frame, relative to 1.0 at its centre.
 *
 * 0.22 is roughly the classic centre-weighted-average distribution (Nikon's
 * 60/40 within a 12 mm circle, extended smoothly). It is enough to stop a
 * sky band along the top of frame from owning the exposure, and not so
 * aggressive that walking toward a bright doorway leaves the exposure unmoved.
 */
const METER_EDGE_WEIGHT = 0.22;

/**
 * Scene luminance above which a meter tap starts losing weight, in the same
 * relative render units as the lighting rig (mid-grey ≈ 0.18).
 *
 * 2.0 is roughly 3.5 stops over mid-grey: bright sky and bright metal still
 * count, a flame core and a specular glint do not.
 */
const METER_HIGHLIGHT_CEILING = 2.0;

/**
 * Floor under the combined weight, so a frame that is *entirely* sky or
 * entirely fire still meters something rather than dividing by zero and
 * pinning the exposure to a clamp.
 */
const METER_MIN_WEIGHT = 0.04;

/**
 * Uniform hash of a pixel coordinate into `[0, 1)`. The classic sin-fract hash;
 * good enough for dither, where the only requirement is that adjacent pixels
 * are uncorrelated.
 */
function hash21(coord: THREE.Node<'vec2'>): THREE.Node<'float'> {
  return coord.dot(vec2(12.9898, 78.233)).sin().mul(43758.5453).fract() as THREE.Node<'float'>;
}

/**
 * Sixth-order polynomial fit of the AgX sigmoid on `[0, 1]`.
 *
 * This is the fit shipped by Filament and three.js and originally posted by
 * Benjamin Wrensch; it is accurate to well under a display code value against
 * the reference LUT and costs six multiply-adds instead of a texture fetch.
 */
function agxSigmoid(x: THREE.Node<'vec3'>): THREE.Node<'vec3'> {
  const x2 = x.mul(x);
  const x4 = x2.mul(x2);
  return x4
    .mul(x2)
    .mul(15.5)
    .sub(x4.mul(x).mul(40.14))
    .add(x4.mul(31.96))
    .sub(x2.mul(x).mul(6.868))
    .add(x2.mul(0.4298))
    .add(x.mul(0.1191))
    .sub(0.00232) as unknown as THREE.Node<'vec3'>;
}

/**
 * AgX, with an ASC CDL look applied in display-log space.
 *
 * `color` is scene-referred linear sRGB *after* exposure. The return is
 * display-referred linear sRGB in `[0, 1]`.
 */
export function agx(
  color: THREE.Node<'vec3'>,
  slope: THREE.Node<'vec3'>,
  offset: THREE.Node<'vec3'>,
  power: THREE.Node<'vec3'>,
  saturation: THREE.Node<'float'>,
): THREE.Node<'vec3'> {
  const rec2020 = LINEAR_SRGB_TO_LINEAR_REC2020.mul(color.max(0));
  const inset = AGX_INSET.mul(rec2020).max(1e-10);

  // Log encode and normalise the working range to [0, 1].
  const logged = inset.log2();
  const normalised = logged
    .sub(AGX_MIN_EV)
    .div(AGX_MAX_EV - AGX_MIN_EV)
    .clamp(0, 1) as unknown as THREE.Node<'vec3'>;

  const sigmoid = agxSigmoid(normalised);

  // ASC CDL: (x · slope + offset) ^ power, then saturation about Rec.709 luma.
  // Applied here — between the sigmoid and the outset — because this is the
  // perceptually uniform space AgX creates, and contrast applied here does not
  // skew hue the way the same operation on linear light does.
  const cdl = sigmoid.mul(slope).add(offset).max(0).pow(power) as unknown as THREE.Node<'vec3'>;
  const luma = cdl.dot(REC709_LUMA);
  const looked = mix(vec3(luma), cdl, saturation) as unknown as THREE.Node<'vec3'>;

  const outset = AGX_OUTSET.mul(looked.max(0));
  // The sigmoid produces a ~2.2-gamma display encoding; undo it so the result
  // is linear light again and the renderer's own OETF is the only encode.
  const linearised = outset.max(0).pow(vec3(2.2));
  return LINEAR_REC2020_TO_LINEAR_SRGB.mul(linearised).clamp(0, 1) as unknown as THREE.Node<'vec3'>;
}

/** Stephen Hill's ACES fit, for comparison. Hue-skews; that is the point. */
export function acesHill(color: THREE.Node<'vec3'>): THREE.Node<'vec3'> {
  const inputMatrix = mat3(
    vec3(0.59719, 0.07600, 0.02840),
    vec3(0.35458, 0.90834, 0.13383),
    vec3(0.04823, 0.01566, 0.83777),
  );
  const outputMatrix = mat3(
    vec3(1.60475, -0.10208, -0.00327),
    vec3(-0.53108, 1.10813, -0.07276),
    vec3(-0.07367, -0.00605, 1.07602),
  );
  const v = inputMatrix.mul(color.max(0).div(0.6));
  const a = v.mul(v.add(0.0245786)).sub(0.000090537);
  const b = v.mul(v.mul(0.983729).add(0.4329510)).add(0.238081);
  return outputMatrix.mul(a.div(b)).clamp(0, 1) as unknown as THREE.Node<'vec3'>;
}

/* ------------------------------------------------------------------------- *
 * CompositePass
 * ------------------------------------------------------------------------- */

/** Sizes of the luminance reduction chain, largest first. */
const METER_SIZES: readonly number[] = [64, 16, 4, 1];

/** `uniform(new Vector2())`, named once so the pass signatures stay readable. */
type Vec2Uniform = ReturnType<typeof uniform<'vec2'>>;

interface MeterStage {
  target: THREE.RenderTarget;
  material: THREE.NodeMaterial;
  source: THREE.TextureNode;
  texel: Vec2Uniform;
}

/**
 * The one full-resolution pass that leaves HDR behind.
 *
 * Chromatic aberration, bloom composite, white balance, exposure, tone curve,
 * primary grade, LUT, vignette and grain all happen in this single draw. Every
 * one of them is per-pixel arithmetic on data already in registers, so splitting
 * them into separate passes would buy nothing and cost a full-resolution
 * round-trip each — roughly 33 MB of read+write bandwidth per split at 1080p.
 */
export class CompositePass implements PostPass {
  readonly id = 'post.composite';
  readonly kind = 'chain' as const;
  readonly outputDomain = 'ldr' as const;

  enabled = true;

  readonly #bloom: BloomPass;
  readonly #grade: ColorGrade;

  #curve: ToneCurve;
  #look: AgXLook;
  #exposure: number;
  #compensation: number;
  #autoExposure: boolean;
  #middleGrey: number;
  #autoRange: readonly [number, number];
  #adaptUp: number;
  #adaptDown: number;

  #encodeOutput = false;

  #material: THREE.NodeMaterial | null = null;
  #gradeRevision = -1;
  #structure = '';

  // -- composite uniforms --------------------------------------------------
  readonly #uScene = makeSourceNode('composite.scene');
  readonly #uBloom = makeSourceNode('composite.bloom');
  readonly #uAdapted = makeSourceNode('composite.adapted');
  readonly #uBloomIntensity = uniform(0);
  readonly #uExposure = uniform(1);
  readonly #uResolution = uniform(new THREE.Vector2(1, 1));
  readonly #uElapsed = uniform(0);
  readonly #uSlope = uniform(new THREE.Vector3(1, 1, 1));
  readonly #uOffset = uniform(new THREE.Vector3(0, 0, 0));
  readonly #uPower = uniform(new THREE.Vector3(1, 1, 1));
  readonly #uSaturation = uniform(1);
  /** `(log2(middleGrey) + compensationStops, minLogLuminance, maxLogLuminance)` */
  readonly #uAutoParams = uniform(new THREE.Vector3(Math.log2(0.18), -4.5, 1.5));

  // -- metering ------------------------------------------------------------
  #meter: MeterStage[] = [];
  #adaptTargets: THREE.RenderTarget[] = [];
  #adaptMaterial: THREE.NodeMaterial | null = null;
  #adaptPrevious: THREE.TextureNode | null = null;
  #adaptCurrent: THREE.TextureNode | null = null;
  readonly #uAdaptParams = uniform(new THREE.Vector3(1 / 60, 0.5, 1.6));
  /** Log2-luminance window the adapted value is confined to. */
  readonly #uAdaptRange = uniform(new THREE.Vector2(-4.5, 1.5));
  #adaptIndex = 0;
  #meterReady = false;

  constructor(bloom: BloomPass, grade: ColorGrade, options: TonemapOptions = {}) {
    this.#bloom = bloom;
    this.#grade = grade;
    this.#curve = options.curve ?? 'agx';
    this.#look =
      typeof options.look === 'string'
        ? (AGX_LOOKS[options.look] ?? AGX_LOOKS.grimdark)
        : (options.look ?? AGX_LOOKS.grimdark);
    this.#exposure =
      options.ev100 === undefined ? (options.exposure ?? 1) : exposureFromEV100(options.ev100);
    this.#compensation = options.exposureCompensation ?? 0;
    this.#autoExposure = options.autoExposure ?? true;
    this.#middleGrey = options.middleGrey ?? 0.18;
    this.#autoRange = options.autoExposureRange ?? [0.25, 6];
    this.#adaptUp = options.adaptationUp ?? 0.5;
    this.#adaptDown = options.adaptationDown ?? 1.6;
    this.#uploadUniforms();
  }

  /* -- configuration ----------------------------------------------------- */

  /** Manual exposure multiplier. Ignored while auto exposure is on. */
  get exposure(): number {
    return this.#exposure;
  }

  setExposure(multiplier: number): void {
    this.#exposure = Math.max(1e-6, multiplier);
    this.#uploadUniforms();
  }

  /** Manual exposure expressed photographically. See the units note above. */
  setEV100(ev100: number): void {
    this.setExposure(exposureFromEV100(ev100));
  }

  /**
   * Extra stops on top of the manual or metered value.
   *
   * This is the player's brightness trim (see `render/DisplaySettings`), which
   * is why it is readable: the pause menu has to be able to show what is
   * currently in force, including a value that came from `?exposure=` rather
   * than from the slider.
   */
  get exposureCompensation(): number {
    return this.#compensation;
  }

  /** Extra stops applied on top of the manual or metered value. */
  setExposureCompensation(stops: number): void {
    this.#compensation = stops;
    this.#uploadUniforms();
  }

  setAutoExposureEnabled(enabled: boolean): void {
    if (enabled === this.#autoExposure) return;
    this.#autoExposure = enabled;
    this.#uploadUniforms();
    this.#invalidate();
  }

  get autoExposureEnabled(): boolean {
    return this.#autoExposure;
  }

  /** Clamp the metered exposure to `[minExposure, maxExposure]`. */
  setAutoExposureRange(minExposure: number, maxExposure: number): void {
    const lo = Math.max(1e-6, Math.min(minExposure, maxExposure));
    const hi = Math.max(lo, Math.max(minExposure, maxExposure));
    this.#autoRange = [lo, hi];
    this.#uploadUniforms();
  }

  /** Display-referred target for the metered average. */
  setMiddleGrey(value: number): void {
    this.#middleGrey = THREE.MathUtils.clamp(value, 0.01, 0.9);
    this.#uploadUniforms();
  }

  setAdaptation(upSeconds: number, downSeconds: number): void {
    this.#adaptUp = Math.max(0, upSeconds);
    this.#adaptDown = Math.max(0, downSeconds);
  }

  setCurve(curve: ToneCurve): void {
    if (curve === this.#curve) return;
    this.#curve = curve;
    this.#invalidate();
  }

  setLook(look: keyof typeof AGX_LOOKS | AgXLook): void {
    this.#look = typeof look === 'string' ? (AGX_LOOKS[look] ?? AGX_LOOKS.grimdark) : look;
    this.#uploadUniforms();
  }

  /** Whether the pass currently emits sRGB-encoded values. */
  get encodeOutput(): boolean {
    return this.#encodeOutput;
  }

  /**
   * Emit sRGB-encoded rather than linear values.
   *
   * Set by `PostStack` when a spatial AA pass follows, because those operate on
   * gamma-encoded luma. See the colour-management note in `PostStack`.
   */
  setEncodeOutput(encode: boolean): void {
    if (encode === this.#encodeOutput) return;
    this.#encodeOutput = encode;
    this.#invalidate();
  }

  /* -- PostPass ---------------------------------------------------------- */

  isAvailable(_quality: QualityTier, _capabilities: PostCapabilities): boolean {
    // The composite is the only path from HDR to a displayable image; without
    // it the frame would be a raw linear buffer clipped at 1.0.
    return true;
  }

  configure(_quality: QualityTier, _capabilities: PostCapabilities): void {
    /* Tier only affects auto-exposure, which PostStack toggles directly. */
  }

  setSize(width: number, height: number): void {
    this.#uResolution.value.set(width, height);
    this.#grade.setSize(width, height);
  }

  render(frame: PostFrame): void {
    if (this.#autoExposure) this.#renderMetering(frame);

    const material = this.#ensureMaterial();

    this.#uScene.value = frame.input;
    this.#uElapsed.value = frame.elapsed;
    this.#uResolution.value.set(frame.width, frame.height);

    const bloomTexture = this.#bloom.enabled ? this.#bloom.texture : null;
    if (bloomTexture !== null) {
      this.#uBloom.value = bloomTexture;
      this.#uBloomIntensity.value = this.#bloom.intensity;
    } else {
      this.#uBloomIntensity.value = 0;
    }

    frame.blit(material, frame.output, 'post.composite');
  }

  dispose(): void {
    this.#material?.dispose();
    this.#material = null;
    for (const stage of this.#meter) {
      stage.target.dispose();
      stage.material.dispose();
    }
    this.#meter = [];
    for (const target of this.#adaptTargets) target.dispose();
    this.#adaptTargets = [];
    this.#adaptMaterial?.dispose();
    this.#adaptMaterial = null;
  }

  /* -- metering ---------------------------------------------------------- */

  /**
   * Reduce the HDR buffer to a single adapted log-luminance texel.
   *
   * Five draws totalling roughly 70 000 texture fetches — about 0.03 % of the
   * fetch budget of one full-resolution pass at 1080p.
   */
  #renderMetering(frame: PostFrame): void {
    this.#ensureMeter();

    for (let i = 0; i < this.#meter.length; i++) {
      const stage = this.#meter[i];
      if (stage === undefined) continue;
      const previous = this.#meter[i - 1];
      const source = i === 0 ? frame.input : previous?.target.texture;
      if (source === undefined) continue;
      const sourceWidth = i === 0 ? frame.width : (previous?.target.width ?? 1);
      const sourceHeight = i === 0 ? frame.height : (previous?.target.height ?? 1);
      stage.source.value = source;
      stage.texel.value.set(1 / Math.max(1, sourceWidth), 1 / Math.max(1, sourceHeight));
      frame.blit(stage.material, stage.target, `post.meter.${i}`);
    }

    // Adaptation: ping-pong so the shader can read last frame's value.
    const previous = this.#adaptTargets[this.#adaptIndex];
    const current = this.#adaptTargets[1 - this.#adaptIndex];
    const adaptMaterial = this.#adaptMaterial;
    const meterResult = this.#meter[this.#meter.length - 1]?.target.texture;
    if (previous === undefined || current === undefined || adaptMaterial === null) return;

    this.#uAdaptParams.value.set(frame.deltaTime, this.#adaptUp, this.#adaptDown);
    this.#uAdaptRange.value.set(this.#autoRange[0], this.#autoRange[1]);
    if (this.#adaptPrevious !== null) {
      // On the very first frame there is no history; seeding from the metered
      // value directly avoids a one-second fade up from black at level load.
      this.#adaptPrevious.value = this.#meterReady
        ? previous.texture
        : (meterResult ?? previous.texture);
    }
    if (this.#adaptCurrent !== null && meterResult !== undefined) {
      this.#adaptCurrent.value = meterResult;
    }

    frame.blit(adaptMaterial, current, 'post.adapt');

    this.#uAdapted.value = current.texture;
    this.#adaptIndex = 1 - this.#adaptIndex;
    this.#meterReady = true;
  }

  #ensureMeter(): void {
    if (this.#meter.length === METER_SIZES.length) return;

    for (let i = 0; i < METER_SIZES.length; i++) {
      const size = METER_SIZES[i] ?? 1;
      const target = new THREE.RenderTarget(size, size, {
        format: THREE.RGBAFormat,
        type: THREE.HalfFloatType,
        colorSpace: THREE.LinearSRGBColorSpace,
        depthBuffer: false,
        stencilBuffer: false,
        generateMipmaps: false,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
      });
      target.texture.name = `post.meter.${size}`;

      const source = makeSourceNode(`meter.${i}`);
      const texel = uniform(new THREE.Vector2(1, 1));
      const material = new THREE.NodeMaterial();
      material.name = `post.meter.${i}`;
      material.depthTest = false;
      material.depthWrite = false;
      material.fragmentNode = this.#buildMeterNode(source, texel, i === 0) as unknown as THREE.Node;

      this.#meter.push({ target, material, source, texel });
    }

    for (let i = 0; i < 2; i++) {
      const target = new THREE.RenderTarget(1, 1, {
        format: THREE.RGBAFormat,
        type: THREE.HalfFloatType,
        colorSpace: THREE.LinearSRGBColorSpace,
        depthBuffer: false,
        stencilBuffer: false,
        generateMipmaps: false,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
      });
      target.texture.name = `post.adapt.${i}`;
      this.#adaptTargets.push(target);
    }

    this.#adaptPrevious = makeSourceNode('adapt.previous');
    this.#adaptCurrent = makeSourceNode('adapt.current');
    this.#adaptMaterial = new THREE.NodeMaterial();
    this.#adaptMaterial.name = 'post.adapt';
    this.#adaptMaterial.depthTest = false;
    this.#adaptMaterial.depthWrite = false;
    this.#adaptMaterial.fragmentNode = this.#buildAdaptNode() as unknown as THREE.Node;
    this.#uAdapted.value = this.#adaptTargets[0]?.texture ?? this.#uAdapted.value;
  }

  /**
   * One reduction step: a 4x4 box of bilinear taps.
   *
   * The first stage converts to `log2(luminance)`; later stages average an
   * already-logarithmic signal, which is what makes the end result a geometric
   * mean rather than an arithmetic one.
   *
   * ## Why the mean is weighted
   *
   * An unweighted full-frame mean is metering the *picture*, and the picture is
   * mostly sky. Under a full overcast the sky sits two to three stops above
   * anything on the ground and can cover half the frame, so an unweighted meter
   * stops the world down until the ground has no midtones left — the exact
   * failure this scene had. Two weights fix it, and they are the two a real
   * camera has:
   *
   * 1. **Centre weighting** (`METER_EDGE_WEIGHT`). A cosine-ish radial falloff
   *    from 1.0 at the frame centre to 0.22 at the corners. The subject is in
   *    the middle of the frame; a band of sky along the top is not what the
   *    exposure is for. This is the classic centre-weighted average meter.
   *
   * 2. **Highlight rejection** (`METER_HIGHLIGHT_CEILING`). Taps brighter than
   *    the ceiling have their weight rolled off smoothly to zero, so a campfire
   *    ember, a specular hit or a hole in the cloud cannot drag the average.
   *    Rolled off rather than clipped: a hard cut makes the meter pop as a
   *    bright object crosses the threshold.
   *
   * The pair is carried down the reduction chain as a premultiplied
   * `(w·log2 L, w)` in `.rg`, and the final division happens once, in
   * {@link #buildAdaptNode}. Averaging `w·x` and `w` separately and dividing at
   * the end is exact; normalising per stage is not.
   */
  #buildMeterNode(
    source: THREE.TextureNode,
    texel: Vec2Uniform,
    encodeLog: boolean,
  ): THREE.Node<'vec4'> {
    const base = uv();
    let weightedSum: THREE.Node<'float'> = float(0) as unknown as THREE.Node<'float'>;
    let weightSum: THREE.Node<'float'> = float(0) as unknown as THREE.Node<'float'>;

    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        // Sample on the half-texel grid of a 4x4 footprint scaled to cover the
        // whole source region this destination texel represents.
        const offsetX = (x - 1.5) * 2;
        const offsetY = (y - 1.5) * 2;
        const offset = vec2(texel.x.mul(offsetX), texel.y.mul(offsetY));
        const coord = base.add(offset);
        const sample = source.sample(coord);

        if (encodeLog) {
          const luminance = sample.rgb.dot(REC709_LUMA).max(1e-5);

          // Radial centre weighting. `r2` is 0 at the centre and 0.5 at a
          // corner of a square frame, so the scale puts the corner exactly at
          // METER_EDGE_WEIGHT.
          const centred = coord.sub(vec2(0.5, 0.5));
          const r2 = centred.dot(centred).mul(2).clamp(0, 1);
          const spatial = float(1).sub(r2.mul(1 - METER_EDGE_WEIGHT));

          // Highlight rejection: full weight up to the ceiling, smoothly to
          // zero an octave above it.
          const highlight = float(1).sub(
            smoothstep(
              float(METER_HIGHLIGHT_CEILING),
              float(METER_HIGHLIGHT_CEILING * 4),
              luminance,
            ),
          );

          const weight = spatial.mul(highlight).max(METER_MIN_WEIGHT);
          weightedSum = weightedSum.add(weight.mul(luminance.log2())) as THREE.Node<'float'>;
          weightSum = weightSum.add(weight) as THREE.Node<'float'>;
        } else {
          weightedSum = weightedSum.add(sample.r) as THREE.Node<'float'>;
          weightSum = weightSum.add(sample.g) as THREE.Node<'float'>;
        }
      }
    }

    return vec4(weightedSum.div(16), weightSum.div(16), 0, 1) as unknown as THREE.Node<'vec4'>;
  }

  /**
   * Exponential adaptation with asymmetric time constants, evaluated on a
   * single texel.
   */
  #buildAdaptNode(): THREE.Node<'vec4'> {
    const previous = this.#adaptPrevious;
    const current = this.#adaptCurrent;
    if (previous === null || current === null) {
      return vec4(0, 0, 0, 1) as unknown as THREE.Node<'vec4'>;
    }

    const centre = vec2(0.5, 0.5);
    // `.r` is the weight-premultiplied sum of log2 luminance, `.g` the sum of
    // the weights. The division that turns them into a weighted geometric mean
    // happens exactly once, here, at 1x1.
    const meter = current.sample(centre);
    const target = meter.r.div(meter.g.max(1e-4));
    const history = previous.sample(centre).r;

    // The exposure window is authored as a multiplier range and stored here as
    // the equivalent log2-luminance range, so the clamp is one instruction and
    // the adaptation state stays in the space it is filtered in.
    const clampedTarget = target.clamp(this.#uAdaptRange.x, this.#uAdaptRange.y);

    const dt = this.#uAdaptParams.x;
    const tau = clampedTarget.greaterThan(history).select(
      this.#uAdaptParams.y,
      this.#uAdaptParams.z,
    );
    // 1 - e^(-dt/tau): frame-rate independent, unlike a constant lerp factor.
    const weight = float(1).sub(dt.div(tau.max(1e-4)).negate().exp());
    const adapted = mix(history, clampedTarget, weight.clamp(0, 1));

    return vec4(adapted, adapted, adapted, 1) as unknown as THREE.Node<'vec4'>;
  }

  /* -- composite graph --------------------------------------------------- */

  #invalidate(): void {
    this.#material?.dispose();
    this.#material = null;
  }

  #structureKey(): string {
    return [
      this.#curve,
      this.#encodeOutput ? 'srgb' : 'linear',
      this.#autoExposure ? 'auto' : 'manual',
      // Bloom is part of the *structure*, not just a uniform.
      //
      // It used to be neither: `#buildComposite` sampled `#uBloom`
      // unconditionally and `render()` set the intensity to 0 when the pyramid
      // was off, so a disabled bloom still cost a full-resolution texture fetch
      // in the frame's one full-resolution pass — and the tier table carried a
      // comment forbidding anyone from turning bloom off at all, on the
      // strength of a black frame that turned out to belong to a different bug.
      // (The sampler was never actually unbound: `makeSourceNode` seeds every
      // source with a 1x1 opaque black `DataTexture`.) Putting it in the key
      // means `?bloom=off` recompiles the composite without the fetch, which is
      // what "off" has to mean for a kill switch to be worth measuring with.
      this.#bloom.enabled ? 'bloom' : '-',
      this.#grade.aberrationActive ? 'ca' : '-',
      this.#grade.vignetteActive ? 'vig' : '-',
      this.#grade.grainActive ? 'grain' : '-',
    ].join('|');
  }

  #ensureMaterial(): THREE.NodeMaterial {
    const key = this.#structureKey();
    if (this.#material !== null && key === this.#structure && this.#gradeRevision === this.#grade.revision) {
      return this.#material;
    }

    this.#material?.dispose();
    this.#structure = key;
    this.#gradeRevision = this.#grade.revision;

    const material = new THREE.NodeMaterial();
    material.name = 'post.composite';
    material.depthTest = false;
    material.depthWrite = false;
    material.fragmentNode = this.#buildComposite() as unknown as THREE.Node;
    this.#material = material;
    return material;
  }

  #buildComposite(): THREE.Node<'vec4'> {
    const base = uv();
    const centred = base.sub(0.5) as unknown as THREE.Node<'vec2'>;

    // -- scene, with optional lateral chromatic aberration ----------------
    let scene: THREE.Node<'vec3'>;
    if (this.#grade.aberrationActive) {
      const { red, blue } = this.#grade.aberrationScales(centred);
      scene = vec3(
        this.#uScene.sample(red.add(0.5)).r,
        this.#uScene.sample(base).g,
        this.#uScene.sample(blue.add(0.5)).b,
      ) as unknown as THREE.Node<'vec3'>;
    } else {
      scene = this.#uScene.sample(base).rgb as unknown as THREE.Node<'vec3'>;
    }
    scene = scene.max(0) as unknown as THREE.Node<'vec3'>;

    // -- bloom: additive, because the pyramid is thresholded ---------------
    //
    // This was `mix(scene, bloom, intensity)`, with a comment claiming the
    // convex combination preserved the mean of the image. It does — but only
    // when the second operand carries the same mean as the first, and this
    // one cannot: `BloomPass` thresholds at 1.0 so that *only* the fire and the
    // brightest sky bloom, which is exactly the art direction. Everywhere else
    // in the frame the pyramid is black, so the lerp was not adding a glow, it
    // was deleting `intensity` (0.55) of the entire scene's light and replacing
    // it with nothing.
    //
    // That is what made tone a function of the quality tier. The tier table
    // changes `bloomMips` (4 at low → 7 at ultra); more levels spread the
    // thresholded energy across more of the frame, so how much of the image
    // survived the lerp depended on the tier. Measured on this scene at 1280×720
    // with the lerp in place: `ultra` put **40.5%** of the frame below luminance
    // 0.02 against `low`'s 5.5%, and switching bloom off at `ultra` moved it to
    // 3.5% and lifted mean luminance from 0.129 to 0.214. A "quality" setting
    // was costing most of a stop.
    //
    // Adding is the composite a thresholded pyramid actually wants: the scene
    // passes through untouched, the pyramid contributes only where it has
    // energy, and the mip count changes the *radius* of the glow rather than
    // the exposure of the frame. Tier-invariance then holds by construction
    // rather than by luck.
    // ...and when the pyramid is off the term is not multiplied by zero, it is
    // absent. See `#structureKey`.
    let color = scene;
    if (this.#bloom.enabled) {
      const bloomColor = this.#uBloom.sample(base).rgb.max(0);
      color = scene.add(bloomColor.mul(this.#uBloomIntensity)) as unknown as THREE.Node<'vec3'>;
    }

    // -- white balance, still scene-referred ------------------------------
    color = this.#grade.whiteBalance(color);

    // -- exposure ---------------------------------------------------------
    let exposure: THREE.Node<'float'>;
    if (this.#autoExposure) {
      // exposure = middleGrey / L_avg, in log2: 2^(log2(grey) + comp - log2(L)).
      // uAutoParams = (log2(middleGrey) + compensation, minLogL, maxLogL).
      const metered = this.#uAdapted
        .sample(vec2(0.5, 0.5))
        .r.clamp(this.#uAutoParams.y, this.#uAutoParams.z);
      exposure = this.#uAutoParams.x.sub(metered).exp2() as unknown as THREE.Node<'float'>;
    } else {
      exposure = this.#uExposure as unknown as THREE.Node<'float'>;
    }
    color = color.mul(exposure) as unknown as THREE.Node<'vec3'>;

    // -- tone curve -------------------------------------------------------
    let display: THREE.Node<'vec3'>;
    switch (this.#curve) {
      case 'agx':
        display = agx(
          color,
          this.#uSlope as unknown as THREE.Node<'vec3'>,
          this.#uOffset as unknown as THREE.Node<'vec3'>,
          this.#uPower as unknown as THREE.Node<'vec3'>,
          this.#uSaturation as unknown as THREE.Node<'float'>,
        );
        break;
      case 'aces':
        display = acesHill(color);
        break;
      default:
        display = color.clamp(0, 1) as unknown as THREE.Node<'vec3'>;
        break;
    }

    // -- vignette, while the signal is still linear light ------------------
    // A vignette is an intensity falloff in the lens, so it multiplies photons,
    // not code values. Doing it after the encode would bend the shape of the
    // falloff by the transfer function.
    if (this.#grade.vignetteActive) display = this.#grade.vignette(display, centred);

    // -- primary grade, in display-encoded space ---------------------------
    // See the encoding note at the top of ColorGrade.ts: every grading
    // primitive, and every .cube LUT in existence, is defined on code values.
    let encoded = sRGBTransferOETF(
      display.clamp(0, 1) as unknown as THREE.Node<'vec3'>,
    ) as unknown as THREE.Node<'vec3'>;

    encoded = this.#grade.gradeDisplay(encoded);
    if (this.#grade.grainActive) {
      const fragCoord = base.mul(this.#uResolution) as unknown as THREE.Node<'vec2'>;
      encoded = this.#grade.grain(
        encoded,
        fragCoord,
        this.#uElapsed as unknown as THREE.Node<'float'>,
      );
    }
    // -- output dither ------------------------------------------------------
    // A triangular-PDF dither of ±1 code value, applied in display space as the
    // last operation before the write. The frame is a wide smooth sky gradient
    // over a wide smooth firelight falloff — the two shapes that band worst in
    // 8-bit — and TPDF (two independent uniforms differenced) is the standard
    // fix: it decorrelates the quantisation error from the signal, so the
    // contour becomes noise rather than a visible step. Amplitude is one code
    // value, which is below the film grain already present and invisible on
    // anything that is not a gradient.
    //
    // Cheaper and more correct than raising the intermediate targets: the HDR
    // chain is already RGBA16F, so the banding was purely in the final encode.
    {
      const fragCoord = base.mul(this.#uResolution) as unknown as THREE.Node<'vec2'>;
      const n1 = hash21(fragCoord);
      const n2 = hash21(fragCoord.add(vec2(17.13, 91.77)));
      encoded = encoded.add(n1.sub(n2).mul(1 / 255)) as unknown as THREE.Node<'vec3'>;
    }

    encoded = encoded.clamp(0, 1) as unknown as THREE.Node<'vec3'>;

    // The renderer applies the sRGB OETF on the final write, so decode back to
    // linear — unless an AA pass follows, in which case it wants the encoded
    // values and `PostStack` has switched the renderer's output colour space to
    // linear so the bytes pass straight through.
    const result = this.#encodeOutput
      ? encoded
      : (sRGBTransferEOTF(encoded) as unknown as THREE.Node<'vec3'>);

    return vec4(result, 1) as unknown as THREE.Node<'vec4'>;
  }

  #uploadUniforms(): void {
    this.#uExposure.value = this.#exposure * 2 ** this.#compensation;
    this.#uSlope.value.set(this.#look.slope[0], this.#look.slope[1], this.#look.slope[2]);
    this.#uOffset.value.set(this.#look.offset[0], this.#look.offset[1], this.#look.offset[2]);
    this.#uPower.value.set(this.#look.power[0], this.#look.power[1], this.#look.power[2]);
    this.#uSaturation.value = this.#look.saturation;
    // exposure = grey / L  =>  L_bound = grey / exposure_bound. The larger
    // exposure bound therefore produces the *smaller* luminance bound.
    const greyLog = Math.log2(this.#middleGrey);
    const minLogL = Math.log2(this.#middleGrey / this.#autoRange[1]);
    const maxLogL = Math.log2(this.#middleGrey / this.#autoRange[0]);
    this.#uAutoParams.value.set(greyLog + this.#compensation, minLogL, maxLogL);
    this.#uAdaptRange.value.set(minLogL, maxLogL);
  }
}

/* ------------------------------------------------------------------------- *
 * Helpers
 * ------------------------------------------------------------------------- */

function makeSourceNode(name: string): THREE.TextureNode {
  const placeholder = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  placeholder.needsUpdate = true;
  const node = textureNode(placeholder);
  // Sanitised for the same reason as `PostStack.makeTextureNode` — three emits
  // a named uniform node's name straight into the shader source, and a dot in
  // an identifier is a compile error that presents as a black frame.
  node.name = safeNodeName(name);
  return node;
}
