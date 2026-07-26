/**
 * @module render/post/ColorGrade
 *
 * The look. Everything here runs *after* the tone curve, on **display-encoded**
 * (sRGB, gamma) values in `[0, 1]`, except white balance and chromatic
 * aberration which are scene-referred and therefore run before it, and the
 * vignette which is an optical intensity falloff and therefore runs on display
 * *linear*.
 *
 * The encoding matters more than it sounds. Every grading primitive below —
 * lift/gamma/gain, per-channel curves, a `.cube` LUT, saturation, contrast about
 * a pivot — is defined by the grading tools on display-encoded code values, and
 * every LUT you will ever be handed was authored that way. Applying a contrast
 * of 1.07 about a pivot of 0.435 to display-*linear* values instead crushes
 * everything below linear 0.029 — sRGB 0.19 — straight to black, which looks
 * exactly like "a moody grade" until you compare it against the ungraded frame
 * and discover you have thrown away the bottom fifth of the tonal range.
 *
 * This module is deliberately not a `PostPass`. Grading is pure per-pixel
 * arithmetic with no neighbourhood and no extra buffer, so giving it its own
 * full-resolution render target would cost 16 MB of bandwidth per frame to
 * accomplish nothing. Instead it exposes TSL sub-graphs that
 * {@link module:render/post/Tonemap}'s composite pass splices into a single
 * full-screen draw.
 *
 * ---
 *
 * ## Pipeline
 *
 * ```
 *   scene-referred (linear, HDR)
 *     chromatic aberration     radial RGB sample offsets
 *     white balance            von Kries chromatic adaptation, CAT02
 *   ── tone curve (Tonemap.ts) ──
 *   display-referred (linear, [0,1])
 *     vignette                 optical falloff, so it must be linear
 *   ── sRGB OETF ──
 *   display-encoded ([0,1] code values)
 *     lift / gamma / gain      ASC CDL slope-offset-power, in that order
 *     per-channel curves       monotone-cubic 1D LUT, 1024 entries, half float
 *     3D LUT                   optional .cube, trilinear
 *     saturation + contrast    Rec.709 luma preserving
 *     film grain               luminance-weighted, temporally animated
 * ```
 *
 * Ordering is not arbitrary. Lift/gamma/gain before the curves means the curves
 * see a signal that already has its black and white points where the colourist
 * put them, which is how a DaVinci node graph is built. The 3D LUT comes last
 * of the colour operations because a `.cube` from a film emulation expects a
 * finished, primary-graded image.
 *
 * ## The default look
 *
 * `GRIMDARK_GRADE` is the Blood Moor: cold, wet, overcast. It works by
 * *separating* colour rather than removing it — the standard failure mode of a
 * "dark fantasy" grade is to pull global saturation down until everything is
 * the same grey. Instead:
 *
 * - White balance is pushed ~700 K cool, so open sky and wet stone go blue-grey
 *   while anything lit by fire stays emphatically orange.
 * - Lift carries a small positive blue and negative red, which tints only the
 *   shadows. Gain carries the opposite, which tints only the highlights. The
 *   result is complementary shadow/highlight separation with global saturation
 *   almost untouched (0.9), so moss, rust and blood survive.
 * - The green channel gets a slightly lifted toe via its own curve, which is
 *   what makes wet vegetation read as *dead* rather than *dark green*.
 *
 * ## References
 *
 * - ASC CDL v1.2 — the slope/offset/power primitive grade.
 * - CIECAM02 CAT02 chromatic adaptation matrix, for white balance.
 * - Fritsch & Carlson, *"Monotone Piecewise Cubic Interpolation"*, SIAM J.
 *   Numer. Anal. 17(2), 1980 — the curve interpolant, chosen because a natural
 *   cubic spline overshoots and an overshooting tone curve inverts contrast.
 * - Kang et al., *"Design of Advanced Color Temperature Control System for HDTV
 *   Applications"*, 2002 — Kelvin to CIE xy.
 */

import * as THREE from 'three/webgpu';
import {
  clamp,
  float,
  fract,
  max,
  mix,
  pow,
  sin,
  smoothstep,
  texture as textureNode,
  texture3D,
  uniform,
  vec2,
  vec3,
} from 'three/tsl';

/* ------------------------------------------------------------------------- *
 * Settings
 * ------------------------------------------------------------------------- */

/** Control points of a per-channel tone curve, as `[input, output]` pairs. */
export type CurvePoints = ReadonlyArray<readonly [number, number]>;

export interface ColorGradeSettings {
  /** Kelvin shift relative to the 6500 K working white. Negative is cooler. */
  temperature: number;
  /** Green/magenta shift. Positive is greener. Range roughly `[-1, 1]`. */
  tint: number;

  /** Additive shadow offset, per channel. Small values: `[-0.05, 0.05]`. */
  lift: readonly [number, number, number];
  /** Midtone power, per channel. 1 is neutral; below 1 brightens. */
  gamma: readonly [number, number, number];
  /** Multiplicative highlight scale, per channel. 1 is neutral. */
  gain: readonly [number, number, number];

  /** Global saturation. 1 is neutral. */
  saturation: number;
  /** Contrast about {@link contrastPivot}. 1 is neutral. */
  contrast: number;
  /** Display-referred pivot the contrast rotates about. */
  contrastPivot: number;

  /** Per-channel curves. An empty array means "identity, skip the LUT". */
  curveR: CurvePoints;
  curveG: CurvePoints;
  curveB: CurvePoints;

  /** Blend toward the bound 3D LUT, in `[0, 1]`. */
  lutMix: number;

  /** Vignette darkening at the frame corners, in `[0, 1]`. 0 disables. */
  vignette: number;
  /** Where the vignette starts, as a fraction of the corner radius. */
  vignetteStart: number;
  /** Roundness: 0 follows the frame aspect, 1 is circular. */
  vignetteRoundness: number;

  /** Lateral chromatic aberration at the frame corner, in pixels. */
  chromaticAberration: number;

  /** Grain amplitude on a mid-grey patch, in display-referred units. */
  grain: number;
  /** Grain cell size in pixels. Larger reads as coarser film stock. */
  grainSize: number;
  /** Grain updates per second. 0 freezes it (useful for golden images). */
  grainRate: number;
}

export interface ColorGradeOptions extends Partial<ColorGradeSettings> {
  /** Start from this preset instead of {@link GRIMDARK_GRADE}. */
  preset?: ColorGradeSettings;
}

/**
 * Neutral pass-through. Useful as a base for a custom look, and as the
 * reference the tests compare against.
 */
export const NEUTRAL_GRADE: ColorGradeSettings = {
  temperature: 0,
  tint: 0,
  lift: [0, 0, 0],
  gamma: [1, 1, 1],
  gain: [1, 1, 1],
  saturation: 1,
  contrast: 1,
  contrastPivot: 0.435,
  curveR: [],
  curveG: [],
  curveB: [],
  lutMix: 0,
  vignette: 0,
  vignetteStart: 0.45,
  vignetteRoundness: 0.6,
  chromaticAberration: 0,
  grain: 0,
  grainSize: 1.4,
  grainRate: 24,
};

/**
 * The shipping look: cold overcast medieval grimdark.
 *
 * Every number here is small. A grade that reads on a histogram is a grade the
 * player notices, and the brief for this project is restraint.
 */
export const GRIMDARK_GRADE: ColorGradeSettings = {
  temperature: -700,
  tint: 0.04,

  // Cold shadows, faintly warm highlights: complementary separation.
  lift: [-0.004, 0.0, 0.012],
  gamma: [1.0, 0.99, 1.02],
  gain: [1.015, 1.0, 0.972],

  saturation: 0.9,
  contrast: 1.07,
  contrastPivot: 0.435,

  // A raised toe on green only. Dead grass and wet moss stop reading as black
  // and start reading as the colour of something that used to be alive.
  curveR: [],
  curveG: [
    [0, 0.008],
    [0.25, 0.25],
    [0.6, 0.6],
    [1, 1],
  ],
  curveB: [],

  lutMix: 0,

  vignette: 0.32,
  vignetteStart: 0.42,
  vignetteRoundness: 0.55,

  chromaticAberration: 0.7,

  grain: 0.016,
  grainSize: 1.5,
  grainRate: 24,
};

/* ------------------------------------------------------------------------- *
 * Pure math — exported for tests
 * ------------------------------------------------------------------------- */

/**
 * Evaluate a monotone piecewise-cubic Hermite interpolant through `points`.
 *
 * Fritsch-Carlson tangent limiting guarantees the result never overshoots the
 * control points, which for a tone curve is the difference between "an S-curve"
 * and "an S-curve with a contrast inversion in the toe".
 *
 * Points must be sorted by input and lie in `[0, 1]`; outside the first and
 * last point the curve is clamped, which is the correct behaviour for a LUT.
 */
export function evaluateCurve(points: CurvePoints, x: number): number {
  const n = points.length;
  if (n === 0) return x;
  const first = points[0];
  const last = points[n - 1];
  if (first === undefined || last === undefined) return x;
  if (n === 1) return first[1];
  if (x <= first[0]) return first[1];
  if (x >= last[0]) return last[1];

  // Secants and Fritsch-Carlson limited tangents.
  const slope = new Array<number>(n - 1);
  for (let i = 0; i < n - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    if (a === undefined || b === undefined) return x;
    const h = b[0] - a[0];
    slope[i] = h === 0 ? 0 : (b[1] - a[1]) / h;
  }

  const tangent = new Array<number>(n);
  tangent[0] = slope[0] ?? 0;
  tangent[n - 1] = slope[n - 2] ?? 0;
  for (let i = 1; i < n - 1; i++) {
    const previous = slope[i - 1] ?? 0;
    const next = slope[i] ?? 0;
    tangent[i] = previous * next <= 0 ? 0 : (previous + next) / 2;
  }
  for (let i = 0; i < n - 1; i++) {
    const s = slope[i] ?? 0;
    if (s === 0) {
      tangent[i] = 0;
      tangent[i + 1] = 0;
      continue;
    }
    const alpha = (tangent[i] ?? 0) / s;
    const beta = (tangent[i + 1] ?? 0) / s;
    const magnitude = Math.hypot(alpha, beta);
    if (magnitude > 3) {
      const scale = 3 / magnitude;
      tangent[i] = scale * alpha * s;
      tangent[i + 1] = scale * beta * s;
    }
  }

  let segment = 0;
  for (let i = 0; i < n - 1; i++) {
    const b = points[i + 1];
    if (b !== undefined && x <= b[0]) {
      segment = i;
      break;
    }
    segment = i;
  }

  const a = points[segment];
  const b = points[segment + 1];
  if (a === undefined || b === undefined) return x;

  const h = b[0] - a[0];
  if (h === 0) return b[1];
  const t = (x - a[0]) / h;
  const t2 = t * t;
  const t3 = t2 * t;

  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;

  return h00 * a[1] + h10 * h * (tangent[segment] ?? 0) + h01 * b[1] + h11 * h * (tangent[segment + 1] ?? 0);
}

/**
 * CIE 1931 chromaticity of a blackbody at `kelvin`, using Kang et al.'s
 * piecewise cubic fit. Valid for 1667-25000 K.
 */
export function kelvinToChromaticity(kelvin: number): [number, number] {
  const t = THREE.MathUtils.clamp(kelvin, 1667, 25000);
  const t2 = t * t;
  const t3 = t2 * t;

  let x: number;
  if (t <= 4000) {
    x = -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.17991;
  } else {
    x = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.24039;
  }

  const x2 = x * x;
  const x3 = x2 * x;

  let y: number;
  if (t <= 2222) {
    y = -1.1063814 * x3 - 1.3481102 * x2 + 2.18555832 * x - 0.20219683;
  } else if (t <= 4000) {
    y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
  } else {
    y = 3.081758 * x3 - 5.8733867 * x2 + 3.75112997 * x - 0.37001483;
  }

  return [x, y];
}

/**
 * A linear-sRGB white-balance matrix.
 *
 * Converts the working white (D65, 6500 K) and the requested white to LMS via
 * CAT02, forms the von Kries diagonal ratio, and folds the whole chain back to
 * linear sRGB. Returned column-major, ready for `THREE.Matrix3.fromArray`.
 *
 * `temperatureShift` is in Kelvin relative to 6500 K; `tintShift` is a
 * green/magenta offset applied to the target chromaticity's `y`.
 */
export function whiteBalanceMatrix(temperatureShift: number, tintShift: number): number[] {
  if (temperatureShift === 0 && tintShift === 0) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }

  // Sign convention matches a camera's white-balance control: a *negative*
  // temperature shift must make the image cooler. Adapting the render toward a
  // warmer illuminant is what does that — the von Kries ratio then boosts the
  // short-wavelength cone response. Getting this backwards is the classic
  // white-balance bug, and it is invisible until you compare two grades.
  const target = kelvinToChromaticity(6500 + temperatureShift);
  const targetXyz = chromaticityToXyz(target[0], target[1] - tintShift * 0.05);
  const sourceXyz = chromaticityToXyz(0.31271, 0.32902); // D65

  const sourceLms = multiplyMatrixVector(CAT02, sourceXyz);
  const targetLms = multiplyMatrixVector(CAT02, targetXyz);

  const ratio: [number, number, number] = [
    safeRatio(sourceLms[0], targetLms[0]),
    safeRatio(sourceLms[1], targetLms[1]),
    safeRatio(sourceLms[2], targetLms[2]),
  ];

  // M = RGB_to_XYZ^-1 · CAT02^-1 · diag(ratio) · CAT02 · RGB_to_XYZ
  const adaptation = multiplyMatrices(
    CAT02_INVERSE,
    multiplyMatrices(diagonal(ratio), CAT02),
  );
  const rowMajor = multiplyMatrices(
    XYZ_TO_LINEAR_SRGB,
    multiplyMatrices(adaptation, LINEAR_SRGB_TO_XYZ),
  );

  // three's Matrix3.fromArray reads column-major.
  return [
    rowMajor[0] ?? 1,
    rowMajor[3] ?? 0,
    rowMajor[6] ?? 0,
    rowMajor[1] ?? 0,
    rowMajor[4] ?? 1,
    rowMajor[7] ?? 0,
    rowMajor[2] ?? 0,
    rowMajor[5] ?? 0,
    rowMajor[8] ?? 1,
  ];
}

/**
 * Parse an Adobe/Resolve `.cube` 3D LUT.
 *
 * Supports `LUT_3D_SIZE`, `TITLE`, `DOMAIN_MIN`/`DOMAIN_MAX` and `#` comments.
 * 1D LUTs are rejected rather than silently misinterpreted, because a 1D `.cube`
 * loaded as 3D produces a plausible-looking but completely wrong image.
 *
 * Data is returned in the order a `Data3DTexture` wants: R fastest, then G,
 * then B — the same order `.cube` files store it in.
 */
export function parseCubeLUT(text: string): { size: number; data: Float32Array; title: string } {
  let size = 0;
  let title = '';
  const domainMin: [number, number, number] = [0, 0, 0];
  const domainMax: [number, number, number] = [1, 1, 1];
  const values: number[] = [];

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line.length === 0 || line.startsWith('#')) continue;

    if (line.startsWith('TITLE')) {
      title = line.slice(5).trim().replace(/^"|"$/g, '');
      continue;
    }
    if (line.startsWith('LUT_1D_SIZE')) {
      throw new Error('[ColorGrade] parseCubeLUT: 1D .cube LUTs are not supported');
    }
    if (line.startsWith('LUT_3D_SIZE')) {
      size = Number.parseInt(line.slice(11).trim(), 10);
      continue;
    }
    if (line.startsWith('DOMAIN_MIN') || line.startsWith('DOMAIN_MAX')) {
      const target = line.startsWith('DOMAIN_MIN') ? domainMin : domainMax;
      const parts = line.slice(10).trim().split(/\s+/).map(Number);
      target[0] = parts[0] ?? target[0];
      target[1] = parts[1] ?? target[1];
      target[2] = parts[2] ?? target[2];
      continue;
    }

    const parts = line.split(/\s+/);
    if (parts.length < 3) continue;
    const r = Number(parts[0]);
    const g = Number(parts[1]);
    const b = Number(parts[2]);
    if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) continue;
    values.push(r, g, b);
  }

  if (size <= 1) throw new Error('[ColorGrade] parseCubeLUT: missing or invalid LUT_3D_SIZE');
  const expected = size * size * size * 3;
  if (values.length !== expected) {
    throw new Error(
      `[ColorGrade] parseCubeLUT: expected ${expected} samples for size ${size}, got ${values.length}`,
    );
  }

  // Normalise out of the declared domain so the sampler can assume [0, 1].
  const data = new Float32Array(expected);
  for (let i = 0; i < expected; i += 3) {
    for (let c = 0; c < 3; c++) {
      const lo = domainMin[c] ?? 0;
      const hi = domainMax[c] ?? 1;
      const span = hi - lo === 0 ? 1 : hi - lo;
      data[i + c] = ((values[i + c] ?? 0) - lo) / span;
    }
  }

  return { size, data, title };
}

/** Build an identity 3D LUT of `size` entries per axis. */
export function identityLUT(size: number): { size: number; data: Float32Array } {
  const data = new Float32Array(size * size * size * 3);
  const scale = size === 1 ? 0 : 1 / (size - 1);
  let i = 0;
  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        data[i++] = r * scale;
        data[i++] = g * scale;
        data[i++] = b * scale;
      }
    }
  }
  return { size, data };
}

/**
 * Upload a parsed LUT as a `Data3DTexture`.
 *
 * Half float rather than 8-bit: an 8-bit LUT quantises the *output* of the
 * grade, and because a LUT is usually contrast-increasing that quantisation is
 * amplified into visible banding in skies and fog — exactly where this game
 * spends most of its screen area.
 */
export function createLUTTexture(lut: { size: number; data: Float32Array }): THREE.Data3DTexture {
  const { size, data } = lut;
  const packed = new Uint16Array(size * size * size * 4);
  for (let i = 0, o = 0; i < data.length; i += 3, o += 4) {
    packed[o] = THREE.DataUtils.toHalfFloat(data[i] ?? 0);
    packed[o + 1] = THREE.DataUtils.toHalfFloat(data[i + 1] ?? 0);
    packed[o + 2] = THREE.DataUtils.toHalfFloat(data[i + 2] ?? 0);
    packed[o + 3] = THREE.DataUtils.toHalfFloat(1);
  }

  const texture = new THREE.Data3DTexture(packed, size, size, size);
  texture.format = THREE.RGBAFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Bake three curves into one 1D texture, one channel each.
 *
 * `width` samples with linear interpolation; half float so the curve does not
 * become the precision floor of the whole grade.
 */
export function createCurveTexture(
  curveR: CurvePoints,
  curveG: CurvePoints,
  curveB: CurvePoints,
  width = 1024,
): THREE.DataTexture {
  const data = new Uint16Array(width * 4);
  for (let i = 0; i < width; i++) {
    const x = i / (width - 1);
    data[i * 4] = THREE.DataUtils.toHalfFloat(evaluateCurve(curveR, x));
    data[i * 4 + 1] = THREE.DataUtils.toHalfFloat(evaluateCurve(curveG, x));
    data[i * 4 + 2] = THREE.DataUtils.toHalfFloat(evaluateCurve(curveB, x));
    data[i * 4 + 3] = THREE.DataUtils.toHalfFloat(1);
  }

  const texture = new THREE.DataTexture(data, width, 1, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.name = 'post.grade.curves';
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/* ------------------------------------------------------------------------- *
 * ColorGrade
 * ------------------------------------------------------------------------- */

/**
 * Uniform bank plus TSL sub-graph builders for the look.
 *
 * Scalar changes (any of the {@link ColorGradeSettings} numbers) are uniform
 * writes and cost nothing. *Structural* changes — enabling grain, binding a
 * LUT, turning chromatic aberration off — change the shape of the graph and
 * bump {@link revision}; the composite pass watches that and rebuilds its
 * material, which is a shader compile and must not happen per frame.
 */
export class ColorGrade {
  #settings: ColorGradeSettings;
  #revision = 0;

  #grainEnabled = true;
  #aberrationEnabled = true;

  #curveTexture: THREE.DataTexture | null = null;
  #lutTexture: THREE.Data3DTexture | null = null;
  #lutSize = 0;

  // -- uniforms ------------------------------------------------------------

  readonly #uWhiteBalance = uniform(new THREE.Matrix3());
  readonly #uLift = uniform(new THREE.Vector3());
  readonly #uInvGamma = uniform(new THREE.Vector3(1, 1, 1));
  readonly #uGain = uniform(new THREE.Vector3(1, 1, 1));
  readonly #uSaturation = uniform(1);
  readonly #uContrast = uniform(1);
  readonly #uContrastPivot = uniform(0.435);
  readonly #uLutMix = uniform(0);
  readonly #uLutScaleOffset = uniform(new THREE.Vector2(1, 0));
  /** `(strength, start)` — the shape lives in {@link #uVignetteScale}. */
  readonly #uVignette = uniform(new THREE.Vector2(0, 0.45));
  /**
   * Per-axis scale that maps centred UV onto a radius of exactly 1 at the frame
   * corner. Precomputed on the CPU so the shader neither branches on roundness
   * nor carries the aspect ratio around.
   */
  readonly #uVignetteScale = uniform(new THREE.Vector2(1.41421356, 1.41421356));
  /**
   * Radial UV scale per unit of `radius²`, already converted from "pixels at
   * the corner" so the authored number means what it says at any resolution.
   */
  readonly #uAberration = uniform(0);
  readonly #uGrain = uniform(new THREE.Vector3(0, 1.5, 24));
  #width = 1;
  #height = 1;

  constructor(options: ColorGradeOptions = {}) {
    const { preset, ...overrides } = options;
    this.#settings = { ...(preset ?? GRIMDARK_GRADE), ...overrides };
    this.#rebuildCurves();
    this.#uploadUniforms();
  }

  /* -- settings ---------------------------------------------------------- */

  get settings(): Readonly<ColorGradeSettings> {
    return this.#settings;
  }

  /** Bumped whenever the *shape* of the node graph changes. */
  get revision(): number {
    return this.#revision;
  }

  /** Merge `changes` into the current look. Curves are rebaked when supplied. */
  set(changes: Partial<ColorGradeSettings>): void {
    const curvesChanged =
      changes.curveR !== undefined || changes.curveG !== undefined || changes.curveB !== undefined;
    const hadCurves = this.#hasCurves();

    this.#settings = { ...this.#settings, ...changes };
    if (curvesChanged) this.#rebuildCurves();
    this.#uploadUniforms();

    if (curvesChanged && hadCurves !== this.#hasCurves()) this.#revision++;
  }

  /** Replace the whole look. */
  setPreset(preset: ColorGradeSettings): void {
    const hadCurves = this.#hasCurves();
    this.#settings = { ...preset };
    this.#rebuildCurves();
    this.#uploadUniforms();
    if (hadCurves !== this.#hasCurves()) this.#revision++;
  }

  setGrainEnabled(enabled: boolean): void {
    if (enabled === this.#grainEnabled) return;
    this.#grainEnabled = enabled;
    this.#revision++;
  }

  setChromaticAberrationEnabled(enabled: boolean): void {
    if (enabled === this.#aberrationEnabled) return;
    this.#aberrationEnabled = enabled;
    this.#revision++;
  }

  /** Whether the graph currently includes the aberration taps. */
  get aberrationActive(): boolean {
    return this.#aberrationEnabled && this.#settings.chromaticAberration > 0;
  }

  /**
   * Bind a 3D LUT.
   *
   * Pass `null` to unbind. The texture is *not* owned — dispose it yourself if
   * you created it — because a LUT is usually a shared asset.
   */
  setLUT(lut: THREE.Data3DTexture | null, mix = 1): void {
    const had = this.#lutTexture !== null;
    this.#lutTexture = lut;
    this.#lutSize = lut === null ? 0 : lut.image.width;
    this.#settings = { ...this.#settings, lutMix: THREE.MathUtils.clamp(mix, 0, 1) };
    this.#uploadUniforms();
    if (had !== (lut !== null)) this.#revision++;
  }

  /** Convenience: parse, upload and bind a `.cube` file's contents. */
  setLUTFromCube(text: string, mix = 1): THREE.Data3DTexture {
    const texture = createLUTTexture(parseCubeLUT(text));
    this.setLUT(texture, mix);
    return texture;
  }

  /**
   * Frame size. Both the vignette shape and the aberration strength are
   * resolution-dependent and are resolved on the CPU here rather than in the
   * shader.
   */
  setSize(width: number, height: number): void {
    this.#width = Math.max(1, width);
    this.#height = Math.max(1, height);
    this.#uploadUniforms();
  }

  dispose(): void {
    this.#curveTexture?.dispose();
    this.#curveTexture = null;
    this.#lutTexture = null;
  }

  /* -- node builders ----------------------------------------------------- */

  /**
   * Radial UV offsets for lateral chromatic aberration.
   *
   * Real transverse CA is a *scale* difference between wavelengths: the image
   * formed by short wavelengths is very slightly larger than the one formed by
   * long ones. So the offset must be radial and must grow with distance from
   * the optical axis, not be a constant per-channel shift — a constant shift
   * gives every edge in the frame the same fringe, including the ones in the
   * middle, which is the tell-tale sign of a fake.
   *
   * The `r^2` growth (rather than `r`) keeps the centre third of the frame
   * visibly clean, which is where the player is looking.
   *
   * @returns scale factors to apply to the centred UV for R and B; G is 1.
   */
  aberrationScales(uvCentred: THREE.Node<'vec2'>): {
    red: THREE.Node<'vec2'>;
    blue: THREE.Node<'vec2'>;
  } {
    const radiusSq = uvCentred.dot(uvCentred);
    // `uAberration` is already `4 * strengthPixels / width`, so at the frame
    // corner (radius² = 0.5, |uv.x| = 0.5) the red tap lands exactly
    // `strengthPixels` away from the green one.
    const magnitude = radiusSq.mul(this.#uAberration);
    return {
      red: uvCentred.mul(float(1).add(magnitude)) as THREE.Node<'vec2'>,
      blue: uvCentred.mul(float(1).sub(magnitude)) as THREE.Node<'vec2'>,
    };
  }

  /** White balance, applied to scene-referred linear light. */
  whiteBalance(color: THREE.Node<'vec3'>): THREE.Node<'vec3'> {
    return this.#uWhiteBalance.mul(color) as unknown as THREE.Node<'vec3'>;
  }

  /**
   * The primary grade, on display-referred values.
   *
   * Slope-offset-power in ASC CDL order, then curves, then the 3D LUT, then
   * saturation and contrast.
   */
  gradeDisplay(color: THREE.Node<'vec3'>): THREE.Node<'vec3'> {
    // ASC CDL: out = (in * slope + offset) ^ power. `lift` is folded into the
    // offset with a (1 - in) weight so that it acts on shadows only, which is
    // the colourist's "lift" rather than a flat pedestal.
    const gained = color.mul(this.#uGain);
    const lifted = gained.add(this.#uLift.mul(gained.oneMinus().max(0)));
    let graded = pow(lifted.max(0), this.#uInvGamma) as unknown as THREE.Node<'vec3'>;

    const curves = this.#curveTexture;
    if (curves !== null) {
      const node = textureNode(curves);
      // One fetch per channel: each channel indexes its *own* curve.
      graded = vec3(
        node.sample(vec2(graded.r, 0.5)).r,
        node.sample(vec2(graded.g, 0.5)).g,
        node.sample(vec2(graded.b, 0.5)).b,
      ) as unknown as THREE.Node<'vec3'>;
    }

    const lut = this.#lutTexture;
    if (lut !== null) {
      // Half-texel inset: sampling a 3D LUT at the exact [0,1] corners reads
      // half a texel outside the data and clamps, which crushes the extremes.
      const coord = graded.saturate().mul(this.#uLutScaleOffset.x).add(this.#uLutScaleOffset.y);
      const sampled = texture3D(lut, coord).rgb as unknown as THREE.Node<'vec3'>;
      graded = mix(graded, sampled, this.#uLutMix) as unknown as THREE.Node<'vec3'>;
    }

    const luma = graded.dot(vec3(0.2126, 0.7152, 0.0722));
    const saturated = mix(vec3(luma), graded, this.#uSaturation) as unknown as THREE.Node<'vec3'>;

    const contrasted = saturated
      .sub(this.#uContrastPivot)
      .mul(this.#uContrast)
      .add(this.#uContrastPivot);

    return contrasted.max(0) as unknown as THREE.Node<'vec3'>;
  }

  /**
   * Optical vignette.
   *
   * Modelled on the `cos^4` natural falloff of an ideal lens but reshaped with
   * a smoothstep so the artist controls where it starts. Roundness blends
   * between an aspect-correct ellipse (0) and a circle (1); at 1 a wide frame
   * gets darker at the left and right edges than at the top and bottom, which
   * is what an anamorphic front element actually does.
   */
  vignette(color: THREE.Node<'vec3'>, uvCentred: THREE.Node<'vec2'>): THREE.Node<'vec3'> {
    const strength = this.#uVignette.x;
    const start = this.#uVignette.y;

    const radius = uvCentred.mul(this.#uVignetteScale).length();
    const falloff = smoothstep(start, float(1.0), radius);
    const attenuation = float(1).sub(falloff.mul(strength));
    return color.mul(attenuation.saturate()) as unknown as THREE.Node<'vec3'>;
  }

  /**
   * Film grain.
   *
   * Two properties make grain read as film rather than as video noise:
   *
   * 1. It is strongest in the midtones and vanishes in the highlights. Silver
   *    halide grain is a *density* fluctuation, so a fully exposed area has
   *    nothing left to fluctuate. `4 * L * (1 - L)` is the standard cheap
   *    stand-in for that response curve.
   * 2. It has a spatial scale independent of resolution. Hashing on
   *    `pixelCoord / grainSize` keeps the grain the same apparent size at 720p
   *    and 4K instead of turning into invisible per-pixel dither.
   *
   * Temporal animation is quantised to `grainRate` steps per second so the
   * grain does not shimmer at the display refresh rate — real projected film
   * changes grain 24 times a second.
   */
  grain(
    color: THREE.Node<'vec3'>,
    fragCoord: THREE.Node<'vec2'>,
    elapsed: THREE.Node<'float'>,
  ): THREE.Node<'vec3'> {
    const amount = this.#uGrain.x;
    const size = this.#uGrain.y;
    const rate = this.#uGrain.z;

    const cell = fragCoord.div(max(size, float(0.25)));
    const frame = elapsed.mul(rate).floor();

    // Interleaved-gradient-style hash: cheap, no texture, and its spectrum is
    // flat enough that the result does not show structure at low amplitude.
    const seeded = vec2(cell.x.add(frame.mul(17.13)), cell.y.add(frame.mul(29.71)));
    const noise = fract(sin(seeded.dot(vec2(12.9898, 78.233))).mul(43758.5453)).sub(0.5);

    const luma = color.dot(vec3(0.2126, 0.7152, 0.0722)).saturate();
    const response = luma.mul(luma.oneMinus()).mul(4);

    return color.add(noise.mul(amount).mul(response)).max(0) as unknown as THREE.Node<'vec3'>;
  }

  /** Whether the graph should include the grain nodes. */
  get grainActive(): boolean {
    return this.#grainEnabled && this.#settings.grain > 0;
  }

  /** Whether the graph should include the vignette nodes. */
  get vignetteActive(): boolean {
    return this.#settings.vignette > 0;
  }

  /* -- internals --------------------------------------------------------- */

  #hasCurves(): boolean {
    return this.#curveTexture !== null;
  }

  #rebuildCurves(): void {
    const { curveR, curveG, curveB } = this.#settings;
    const identity = curveR.length === 0 && curveG.length === 0 && curveB.length === 0;

    this.#curveTexture?.dispose();
    this.#curveTexture = identity ? null : createCurveTexture(curveR, curveG, curveB);
  }

  #uploadUniforms(): void {
    const s = this.#settings;

    this.#uWhiteBalance.value.fromArray(whiteBalanceMatrix(s.temperature, s.tint));
    this.#uLift.value.set(s.lift[0], s.lift[1], s.lift[2]);
    this.#uInvGamma.value.set(1 / safePositive(s.gamma[0]), 1 / safePositive(s.gamma[1]), 1 / safePositive(s.gamma[2]));
    this.#uGain.value.set(s.gain[0], s.gain[1], s.gain[2]);
    this.#uSaturation.value = s.saturation;
    this.#uContrast.value = s.contrast;
    this.#uContrastPivot.value = s.contrastPivot;
    this.#uLutMix.value = this.#lutTexture === null ? 0 : s.lutMix;

    this.#uVignette.value.set(s.vignette, s.vignetteStart);
    // roundness 0 -> the vignette follows the frame (an ellipse); 1 -> a circle
    // in screen space, so a wide frame darkens at the sides first. Normalising
    // by the corner radius keeps `vignetteStart` meaning the same thing either
    // way.
    const aspect = this.#width / this.#height;
    const scaleX = 1 + (aspect - 1) * THREE.MathUtils.clamp(s.vignetteRoundness, 0, 1);
    const cornerRadius = Math.hypot(0.5 * scaleX, 0.5);
    this.#uVignetteScale.value.set(scaleX / cornerRadius, 1 / cornerRadius);

    this.#uAberration.value = this.#aberrationEnabled
      ? (4 * s.chromaticAberration) / this.#width
      : 0;
    this.#uGrain.value.set(this.#grainEnabled ? s.grain : 0, s.grainSize, s.grainRate);

    if (this.#lutSize > 1) {
      const size = this.#lutSize;
      this.#uLutScaleOffset.value.set((size - 1) / size, 0.5 / size);
    } else {
      this.#uLutScaleOffset.value.set(1, 0);
    }
  }
}

/* ------------------------------------------------------------------------- *
 * Colour science constants and small matrix helpers
 * ------------------------------------------------------------------------- */

/** CIECAM02 CAT02 sharpened cone response, row-major. */
const CAT02: readonly number[] = [
  0.7328, 0.4296, -0.1624, -0.7036, 1.6975, 0.0061, 0.003, 0.0136, 0.9834,
];

const CAT02_INVERSE: readonly number[] = [
  1.096124, -0.278869, 0.182745, 0.454369, 0.473533, 0.072098, -0.009628, -0.005698, 1.015326,
];

/** Rec.709 primaries with a D65 white, row-major. */
const LINEAR_SRGB_TO_XYZ: readonly number[] = [
  0.4123908, 0.3575843, 0.1804808, 0.2126390, 0.7151687, 0.0721923, 0.0193308, 0.1191948, 0.9505322,
];

const XYZ_TO_LINEAR_SRGB: readonly number[] = [
  3.2409699, -1.5373832, -0.4986108, -0.9692436, 1.8759675, 0.0415551, 0.0556301, -0.2039770,
  1.0569715,
];

function chromaticityToXyz(x: number, y: number): [number, number, number] {
  const safeY = Math.abs(y) < 1e-6 ? 1e-6 : y;
  return [x / safeY, 1, (1 - x - safeY) / safeY];
}

function multiplyMatrixVector(
  m: readonly number[],
  v: readonly [number, number, number],
): [number, number, number] {
  return [
    (m[0] ?? 0) * v[0] + (m[1] ?? 0) * v[1] + (m[2] ?? 0) * v[2],
    (m[3] ?? 0) * v[0] + (m[4] ?? 0) * v[1] + (m[5] ?? 0) * v[2],
    (m[6] ?? 0) * v[0] + (m[7] ?? 0) * v[1] + (m[8] ?? 0) * v[2],
  ];
}

function multiplyMatrices(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(9).fill(0);
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) sum += (a[row * 3 + k] ?? 0) * (b[k * 3 + col] ?? 0);
      out[row * 3 + col] = sum;
    }
  }
  return out;
}

function diagonal(v: readonly [number, number, number]): number[] {
  return [v[0], 0, 0, 0, v[1], 0, 0, 0, v[2]];
}

function safeRatio(numerator: number, denominator: number): number {
  return Math.abs(denominator) < 1e-9 ? 1 : numerator / denominator;
}

function safePositive(value: number): number {
  return Math.abs(value) < 1e-4 ? 1e-4 : value;
}

/** Re-exported so the composite pass can clamp without importing tsl twice. */
export const clampDisplay = (color: THREE.Node<'vec3'>): THREE.Node<'vec3'> =>
  clamp(color, vec3(0), vec3(1)) as unknown as THREE.Node<'vec3'>;
