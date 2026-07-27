/**
 * @module render/materials/AntiTile
 *
 * Hex-tiling: removes *structural* texture repetition, not just tonal
 * repetition.
 *
 * ### The problem macro variation does not solve
 *
 * {@link module:render/materials/MacroVariation} breaks the periodicity of the
 * envelope, which is enough for a texture with no memorable features. It is not
 * enough for one that has them. If the rock albedo contains one distinctive
 * pale streak, that streak appears on a perfect lattice across the entire cliff
 * and no amount of brightness modulation hides it. The only fix is to stop
 * sampling the texture at the same offset every tile.
 *
 * ### Technique
 *
 * Morten S. Mikkelsen, "Practical Real-Time Hex-Tiling", Journal of Computer
 * Graphics Techniques 11(2), 2022. The lineage runs back through Heitz &
 * Neyret, "High-Performance By-Example Noise using a Histogram-Preserving
 * Blending Operator" (HPG 2018), which introduced the variance-preserving
 * blend, and Mikkelsen's contribution is doing it on a triangular lattice with
 * only three taps and without the histogram transform's precomputed LUT.
 *
 * 1. Skew the UV into a triangular (simplex) lattice.
 * 2. Take the three lattice vertices of the containing triangle and the
 *    barycentric weights.
 * 3. Offset the UV by a per-vertex hash. Three differently offset copies of the
 *    same texture.
 * 4. Sample all three **with explicit gradients**. This is not optional: the
 *    UV offset is discontinuous at every triangle edge, so the implicit
 *    derivative would spike there and the hardware would pick the lowest mip,
 *    drawing a blurred grey line along every lattice edge. Passing the
 *    *undisplaced* derivatives fixes the mip selection everywhere.
 * 5. Blend, preserving both mean and variance.
 *
 * ### Why the blend needs both corrections
 *
 * Averaging N samples of a random field divides its standard deviation by
 * roughly √N. Blending three copies of a rock texture with plain barycentric
 * weights therefore produces a visibly washed-out band through the middle of
 * every triangle — the classic "hex-tiling ghosting" that makes people abandon
 * the technique. The fix (Heitz & Neyret §3, Mikkelsen §3.3) is to rescale the
 * deviation from the mean:
 *
 * ```
 * out = μ + (Σ wᵢ cᵢ − μ) · (Σ wᵢ) / ‖w‖₂
 * ```
 *
 * The mean μ is the texture's own average, computed once at load time. The
 * factor is exactly 1 when one weight dominates and √3 at the triangle centre,
 * which is precisely the contrast the averaging destroyed.
 *
 * ### Cost
 *
 * Three taps per map instead of one, plus ~40 ALU for the lattice. It is gated
 * to `high`/`ultra` quality and to UV-projected surfaces: combining it with
 * triplanar would be nine taps per map, which no budget here survives.
 */

import {
  Fn,
  dFdx,
  dFdy,
  float,
  floor,
  fract,
  max,
  pow,
  step,
  uint,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import type { TextureSampler } from './Triplanar';
import type { Vec2Node, Vec3Node, Vec4Node } from './types';

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/** Integer lattice coordinate. */
export type LatticeVertex = readonly [number, number];

/** Result of the triangular-lattice lookup. */
export interface TriangleGrid {
  /** Barycentric weights of the three vertices. Non-negative, sum to 1. */
  readonly weights: [number, number, number];
  readonly vertices: [LatticeVertex, LatticeVertex, LatticeVertex];
}

/**
 * Lattice basis constants, from Mikkelsen 2022 listing 1.
 *
 * `HEX_SCALE_*` map the UV into a space where the skewed cells are equilateral;
 * `HEX_SKEW` and `HEX_UNSKEW` are the two non-trivial entries of the
 * grid-to-skewed-grid matrix. `2/√3 = 1.1547…`, `1/√3 = 0.5773…`.
 */
export const HEX_SCALE_X = 1.1547005383792515;
export const HEX_SCALE_Y = 2 / 3;
export const HEX_SKEW = 0.5773502691896258;
export const HEX_UNSKEW = 1.1547005383792515;

/**
 * Locate a UV in the triangular lattice.
 *
 * Mikkelsen 2022, listing 1. The lattice is the standard simplex grid: skew
 * into a rhombic basis, take the integer cell, and the fractional part's
 * position relative to the anti-diagonal selects which of the cell's two
 * triangles contains the point.
 *
 * The branchless `s = step(0, -z)` formulation handles both triangles with one
 * arithmetic path, which matters on the GPU and is preserved here so the
 * reference and the shader are the same algorithm rather than two.
 *
 * @param s UV x, already scaled by the lattice density.
 * @param t UV y, likewise.
 */
export function hexTriangleGrid(s: number, t: number): TriangleGrid {
  // Scale so that lattice cells are equilateral in the original UV space.
  const x = s * HEX_SCALE_X;
  const y = t * HEX_SCALE_Y;

  // Skew into the rhombic basis.
  const skewedX = x - y * HEX_SKEW;
  const skewedY = y * HEX_UNSKEW;

  const baseX = Math.floor(skewedX);
  const baseY = Math.floor(skewedY);
  const fx = skewedX - baseX;
  const fy = skewedY - baseY;
  const fz = 1 - fx - fy;

  // Upper or lower triangle of the rhombus. `<= 0` rather than `< 0` so this
  // matches the shader's `step(fz, 0)` exactly on the measure-zero boundary.
  const upper = fz <= 0 ? 1 : 0;
  const sign = 2 * upper - 1;

  const w1 = -fz * sign;
  const w2 = upper - fy * sign;
  const w3 = upper - fx * sign;

  return {
    weights: [w1, w2, w3],
    vertices: [
      [baseX + upper, baseY + upper],
      [baseX + upper, baseY + 1 - upper],
      [baseX + 1 - upper, baseY + upper],
    ],
  };
}

/**
 * Deterministic 2D hash of a lattice vertex to a UV offset in `[0, 1)²`.
 *
 * Integer arithmetic, not `fract(sin(...))`. A sine hash is fine in isolation
 * but its low bits depend on the transcendental implementation, so two backends
 * disagree about where the tiles land and a WebGPU capture stops matching a
 * WebGL2 one pixel for pixel — which would make the whole capture-diff workflow
 * useless for this material. A 32-bit integer mix wraps identically in JS
 * (`Math.imul`), WGSL and GLSL ES 3.0.
 */
export function hexHashOffset(vertex: LatticeVertex): [number, number] {
  // Bias into the positive range: negative-to-unsigned conversion is the one
  // integer operation whose semantics genuinely differ between the two shading
  // languages, so it is avoided rather than relied on.
  const x = (vertex[0] + 0x8000) | 0;
  const y = (vertex[1] + 0x8000) | 0;

  let h = (Math.imul(x, 0x8da6b343) + Math.imul(y, 0xd8163841)) | 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0x5bd1e995);
  h ^= h >>> 15;
  const a = (h >>> 0) / 4294967296;

  let g = (Math.imul(y, 0x27d4eb2d) + Math.imul(x, 0x165667b1)) | 0;
  g ^= g >>> 12;
  g = Math.imul(g, 0x2545f491);
  g ^= g >>> 16;
  const b = (g >>> 0) / 4294967296;

  return [a, b];
}

/**
 * Sharpen barycentric weights so the three-way blend region is narrow.
 *
 * @param exponent Mikkelsen's `g_exp`; 7 is his recommended default. Higher
 *                 narrows the ghosting band but sharpens the seams; below ~4
 *                 the whole surface is a permanent three-way blend.
 */
export function sharpenHexWeights(
  weights: readonly [number, number, number],
  exponent: number,
): [number, number, number] {
  const p: [number, number, number] = [
    Math.pow(Math.max(weights[0], 0), exponent),
    Math.pow(Math.max(weights[1], 0), exponent),
    Math.pow(Math.max(weights[2], 0), exponent),
  ];
  const total = p[0] + p[1] + p[2];
  if (!(total > 0)) return [1, 0, 0];
  return [p[0] / total, p[1] / total, p[2] / total];
}

/**
 * The variance-preserving contrast restoration factor.
 *
 * `(Σ wᵢ) / ‖w‖₂`. Equals 1 when one weight dominates and `√n` when `n` weights
 * are equal, which is exactly the factor by which averaging `n` independent
 * samples shrank the standard deviation.
 */
export function varianceRestoreFactor(weights: readonly number[]): number {
  let sum = 0;
  let sumSq = 0;
  for (const w of weights) {
    sum += w;
    sumSq += w * w;
  }
  if (sumSq <= 0) return 1;
  return sum / Math.sqrt(sumSq);
}

/**
 * Blend samples of one texture, preserving mean and variance.
 *
 * @param samples per-tap values of one channel.
 * @param weights normalised weights.
 * @param mean    the texture's own average for that channel.
 */
export function varianceBlend(
  samples: readonly number[],
  weights: readonly number[],
  mean: number,
): number {
  let acc = 0;
  for (let i = 0; i < samples.length; i++) acc += (samples[i] ?? 0) * (weights[i] ?? 0);
  return mean + (acc - mean) * varianceRestoreFactor(weights);
}

/* ------------------------------------------------------------------------- *
 * TSL
 * ------------------------------------------------------------------------- */

/** The three offset UVs, their weights, and the shared derivatives. */
export interface HexFrame {
  readonly uv1: Vec2Node;
  readonly uv2: Vec2Node;
  readonly uv3: Vec2Node;
  /** Sharpened, L1-normalised barycentric weights. */
  readonly weights: Vec3Node;
  /** Derivatives of the *undisplaced* UV, for explicit-gradient sampling. */
  readonly ddx: Vec2Node;
  readonly ddy: Vec2Node;
}

/** Integer hash mirroring {@link hexHashOffset}, evaluated on the GPU. */
const hexHashNode = /*@__PURE__*/ Fn(([vertex]: [Vec2Node]) => {
  const x = uint(vertex.x.add(32768)).toVar('hexHx');
  const y = uint(vertex.y.add(32768)).toVar('hexHy');

  const h = x.mul(uint(0x8da6b343)).add(y.mul(uint(0xd8163841))).toVar('hexH');
  h.assign(h.bitXor(h.shiftRight(uint(13))));
  h.assign(h.mul(uint(0x5bd1e995)));
  h.assign(h.bitXor(h.shiftRight(uint(15))));

  const g = y.mul(uint(0x27d4eb2d)).add(x.mul(uint(0x165667b1))).toVar('hexG');
  g.assign(g.bitXor(g.shiftRight(uint(12))));
  g.assign(g.mul(uint(0x2545f491)));
  g.assign(g.bitXor(g.shiftRight(uint(16))));

  // 1 / 2^32, applied after converting to float. Dividing in integer space
  // would quantise every offset to 0.
  const inv = float(1 / 4294967296);
  return vec2(float(h).mul(inv), float(g).mul(inv));
}).setLayout({
  name: 'hexHash',
  type: 'vec2',
  inputs: [{ name: 'vertex', type: 'vec2' }],
});

/**
 * Build the hex-tiling frame for a UV.
 *
 * @param uv        the tiled UV, in texture repeats.
 * @param density   lattice cells per texture repeat. Below ~0.5 the lattice is
 *                  coarser than the texture and the offsets stop hiding
 *                  anything; above ~2 the three-way blend regions overlap the
 *                  features they are meant to hide. 1 is the sane default.
 * @param sharpness weight exponent; see {@link sharpenHexWeights}.
 */
export function hexFrame(uv: Vec2Node, density: number, sharpness: number): HexFrame {
  const ddx = dFdx(uv).toVar('hexDdx');
  const ddy = dFdy(uv).toVar('hexDdy');

  const scaled = uv.mul(density).toVar('hexScaled');
  const x = scaled.x.mul(HEX_SCALE_X).toVar('hexX');
  const y = scaled.y.mul(HEX_SCALE_Y).toVar('hexY');

  const skewed = vec2(x.sub(y.mul(HEX_SKEW)), y.mul(HEX_UNSKEW)).toVar('hexSkew');
  const base = floor(skewed).toVar('hexBase');
  const f = fract(skewed).toVar('hexFrac');
  const fz = float(1).sub(f.x).sub(f.y).toVar('hexFz');

  // Branchless triangle selection, exactly as in `hexTriangleGrid`.
  const upper = step(fz, 0).toVar('hexUpper');
  const sign = upper.mul(2).sub(1).toVar('hexSign');

  const raw = vec3(
    fz.negate().mul(sign),
    upper.sub(f.y.mul(sign)),
    upper.sub(f.x.mul(sign)),
  ).toVar('hexRaw');

  const v1 = base.add(vec2(upper, upper)).toVar('hexV1');
  const v2 = base.add(vec2(upper, float(1).sub(upper))).toVar('hexV2');
  const v3 = base.add(vec2(float(1).sub(upper), upper)).toVar('hexV3');

  const sharpened = pow(max(raw, vec3(0)), vec3(sharpness)).toVar('hexSharp');
  const weights = sharpened
    .div(max(sharpened.x.add(sharpened.y).add(sharpened.z), 1e-6))
    .toVar('hexWeights');

  return {
    uv1: uv.add(hexHashNode(v1)).toVar('hexUv1'),
    uv2: uv.add(hexHashNode(v2)).toVar('hexUv2'),
    uv3: uv.add(hexHashNode(v3)).toVar('hexUv3'),
    weights,
    ddx,
    ddy,
  };
}

/** The variance restoration factor on the GPU. Mirrors {@link varianceRestoreFactor}. */
export const varianceRestoreNode = /*@__PURE__*/ Fn(([weights]: [Vec3Node]) => {
  const sum = weights.x.add(weights.y).add(weights.z).toVar('vrSum');
  return sum.div(max(weights.dot(weights), 1e-8).sqrt());
}).setLayout({
  name: 'varianceRestore',
  type: 'float',
  inputs: [{ name: 'weights', type: 'vec3' }],
});

/**
 * Sample a texture through the hex lattice, preserving mean and variance.
 *
 * @param mean the texture's average value, as a `vec4`. Supplying a wrong mean
 *             does not break the image, it just shifts where contrast is
 *             restored around; supplying `0.5` everywhere is a usable fallback
 *             when the texture's pixels were never decoded.
 */
export function hexSample(tex: TextureSampler, frame: HexFrame, mean: Vec4Node): Vec4Node {
  const grad = (uv: Vec2Node): Vec4Node => tex.sample(uv).grad(frame.ddx, frame.ddy);
  const blended = grad(frame.uv1)
    .mul(frame.weights.x)
    .add(grad(frame.uv2).mul(frame.weights.y))
    .add(grad(frame.uv3).mul(frame.weights.z))
    .toVar('hexBlend');
  return mean.add(blended.sub(mean).mul(varianceRestoreNode(frame.weights)));
}

/**
 * Sample through the hex lattice with plain mean-preserving weights.
 *
 * Correct for normal maps: the variance restoration would push the tangent
 * normal's XY past the unit circle at triangle centres, and the subsequent
 * renormalisation would then quietly flatten the normal exactly where the
 * restoration was meant to help. Normals get the L1 blend and are renormalised
 * by the caller, which is both cheaper and correct.
 */
export function hexSampleLinear(tex: TextureSampler, frame: HexFrame): Vec4Node {
  const grad = (uv: Vec2Node): Vec4Node => tex.sample(uv).grad(frame.ddx, frame.ddy);
  return grad(frame.uv1)
    .mul(frame.weights.x)
    .add(grad(frame.uv2).mul(frame.weights.y))
    .add(grad(frame.uv3).mul(frame.weights.z));
}

/** Convenience for building the `mean` argument from a colour and a scalar set. */
export function meanVec4(r: number, g: number, b: number, a: number): Vec4Node {
  return vec4(r, g, b, a);
}

/** A neutral mean, for textures whose pixels were never decoded. */
export function neutralMean(): Vec4Node {
  return vec4(0.5, 0.5, 0.5, 0.5);
}
