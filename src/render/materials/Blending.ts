/**
 * @module render/materials/Blending
 *
 * The two blends every layered surface depends on: height-aware material
 * transitions, and normal-map composition.
 *
 * ### Height blending, and why linear lerp is the amateur tell
 *
 * `mix(grass, rock, w)` cross-fades two *images*. Real material boundaries do
 * not cross-fade — gravel sits in the gaps between grass tufts, mortar shows
 * between bricks, mud fills the low points of a cobble. What you want is a
 * *displacement-ordered* union: at every texel the material whose mesostructure
 * is physically higher wins, and the transition happens over a narrow band
 * where the two heights are comparable.
 *
 * The implementation is the standard height-map splat operator (Andrey
 * Mishkinis, "Advanced Terrain Texture Splatting", GameDev.net 2013; the same
 * operator is UE4's `HeightLerp` and Unity HDRP's `BlendLayeredFloat`):
 *
 * ```
 * ma  = max_i(h_i + w_i) - depth
 * b_i = w_i · max(h_i + w_i - ma, 0)
 * out = Σ b_i · c_i / Σ b_i
 * ```
 *
 * `depth` is the width of the transition band. The `w_i` factor in front of the
 * bias is a small departure from the published operator and it buys two things.
 * First, a layer at exactly zero coverage is excluded outright, so a rock
 * texture at zero splat weight can never poke through the grass because its
 * height map happens to be taller — which the unweighted form allows and which
 * is a real and confusing artefact. Second, as `depth → ∞` the biases become
 * `w_i · depth` and the operator degenerates *exactly* to a linear lerp by `w`.
 * That makes the sharpness knob continuous over its whole range with no special
 * case for "soft", where the unweighted form instead collapses to an even split
 * and silently ignores the coverage it was given.
 *
 * ### Variance-preserving weights
 *
 * When several *randomly offset copies of the same texture* are blended — which
 * is what hex-tiling does — a plain weighted average destroys contrast, because
 * averaging N independent samples of a random field divides its standard
 * deviation by √N. The result is the characteristic washed-out mush. The fix is
 * Mikkelsen's variance-preserving blend (Morten S. Mikkelsen, "Practical
 * Real-Time Hex-Tiling", JCGT 11(2), 2022, §3.3): normalise by the L2 norm of
 * the weight vector rather than its L1 norm, which keeps the variance of the
 * blended field equal to that of a single sample.
 *
 * ### Normal composition
 *
 * Detail normals are combined with Reoriented Normal Mapping (Colin
 * Barré-Brisebois & Stephen Hill, "Blending in Detail", 2012). RNM treats the
 * detail normal as living in the *base normal's* tangent frame and rotates it
 * accordingly, which is the only common method that is correct when the base
 * normal is steep. Whiteout and partial-derivative blending both flatten
 * detail on steep faces; linear blend-and-normalise is simply wrong.
 *
 * Every function below has a CPU reference implementation that the TSL node
 * mirrors exactly, so `tests/materials.blending.test.ts` can pin the behaviour
 * without a GPU.
 */

import { Fn, float, max, mix, normalize, pow, vec3 } from 'three/tsl';

import type { FloatNode, Vec3Node } from './types';

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/** A 3-component vector as a plain tuple. */
export type Vec3Tuple = readonly [number, number, number];

/**
 * Height-aware blend weights for `n` layers.
 *
 * @param heights   mesoscale height of each layer at this texel, any range.
 * @param weights   the coverage each layer would have under a linear blend.
 *                  Need not sum to 1; only ratios matter.
 * @param depth     width of the transition band in height units. Must be > 0.
 *                  Small values (0.02) give a crisp, interlocking boundary;
 *                  large values (10) degenerate to a linear lerp by `weights`.
 * @returns weights summing to exactly 1 (up to float error).
 *
 * Layers whose weight is 0 are excluded outright: a layer that is not present
 * must never win on height alone, or a rock texture at zero coverage would
 * still poke through the grass.
 */
export function heightBlendWeights(
  heights: readonly number[],
  weights: readonly number[],
  depth: number,
): number[] {
  const n = Math.min(heights.length, weights.length);
  const band = Math.max(depth, 1e-6);
  const out = new Array<number>(n).fill(0);

  let peak = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i++) {
    const w = weights[i] ?? 0;
    if (w <= 0) continue;
    const s = (heights[i] ?? 0) + w;
    if (s > peak) peak = s;
  }
  // Every layer had zero coverage: fall back to a uniform split rather than
  // returning zeros, which would blank the surface.
  if (peak === Number.NEGATIVE_INFINITY) return out.map(() => 1 / Math.max(n, 1));

  const cut = peak - band;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const w = weights[i] ?? 0;
    if (w <= 0) continue;
    const b = w * Math.max((heights[i] ?? 0) + w - cut, 0);
    out[i] = b;
    total += b;
  }
  if (total <= 0) return out.map(() => 1 / Math.max(n, 1));
  for (let i = 0; i < n; i++) out[i] = (out[i] ?? 0) / total;
  return out;
}

/**
 * Two-layer height blend, returning the weight of the *second* layer.
 *
 * The common case, and cheaper than the general form because the normalisation
 * collapses. Equivalent to `heightBlendWeights([h0, h1], [1 - w, w], depth)[1]`.
 */
export function heightBlend2(h0: number, h1: number, w: number, depth: number): number {
  const weights = heightBlendWeights([h0, h1], [1 - w, w], depth);
  return weights[1] ?? 0;
}

/**
 * Variance-preserving normalisation of a weight vector.
 *
 * Mikkelsen 2022 §3.3. Given weights `w_i` applied to N samples of the *same*
 * random field, dividing by `‖w‖₂` instead of `Σw` keeps the variance of the
 * blend equal to the variance of one sample, so contrast survives.
 *
 * Note this is only correct when the samples are drawn from one field. Blending
 * two genuinely different materials must use {@link heightBlendWeights}, whose
 * L1 normalisation is what preserves the *mean* (and therefore energy).
 */
export function variancePreservingWeights(weights: readonly number[]): number[] {
  let sumSq = 0;
  for (const w of weights) sumSq += w * w;
  const norm = Math.sqrt(Math.max(sumSq, 1e-12));
  return weights.map((w) => w / norm);
}

/**
 * Sharpen a weight vector by exponentiation, then renormalise (L1).
 *
 * Used by hex-tiling to push the barycentric weights towards a single sample so
 * that the blend region — where contrast loss is unavoidable — is as narrow as
 * possible.
 */
export function sharpenWeights(weights: readonly number[], exponent: number): number[] {
  const raised = weights.map((w) => Math.pow(Math.max(w, 0), exponent));
  let total = 0;
  for (const w of raised) total += w;
  if (total <= 0) return weights.map(() => 1 / Math.max(weights.length, 1));
  return raised.map((w) => w / total);
}

/**
 * Reoriented Normal Mapping (Barré-Brisebois & Hill 2012).
 *
 * @param base   base tangent-space normal, unit length, z > 0.
 * @param detail detail tangent-space normal, unit length, z > 0.
 * @returns the composed unit normal.
 *
 * Derivation: build the rotation that takes `+Z` to `base`, apply it to
 * `detail`. The published closed form avoids constructing the matrix:
 *
 * ```
 * t = base  + (0, 0, 1)
 * u = detail * (-1, -1, 1)
 * r = t * dot(t, u) / t.z - u
 * ```
 */
export function reorientNormal(base: Vec3Tuple, detail: Vec3Tuple): Vec3Tuple {
  const t: Vec3Tuple = [base[0], base[1], base[2] + 1];
  const u: Vec3Tuple = [-detail[0], -detail[1], detail[2]];
  const d = t[0] * u[0] + t[1] * u[1] + t[2] * u[2];
  const s = d / Math.max(t[2], 1e-6);
  const r: Vec3Tuple = [t[0] * s - u[0], t[1] * s - u[1], t[2] * s - u[2]];
  const len = Math.hypot(r[0], r[1], r[2]);
  if (len < 1e-9) return [0, 0, 1];
  return [r[0] / len, r[1] / len, r[2] / len];
}

/**
 * Scale a tangent-space normal's XY, renormalising to keep it unit length.
 *
 * The naive `vec3(n.xy * s, n.z)` is what three's `NormalMapNode` does and it
 * is fine for `s <= 1`, but it silently produces non-unit normals for `s > 1`,
 * which then get normalised anyway — so the effective strength saturates. Doing
 * the renormalisation explicitly makes the knob behave linearly in slope.
 */
export function scaleNormal(n: Vec3Tuple, strength: number): Vec3Tuple {
  const x = n[0] * strength;
  const y = n[1] * strength;
  const z = Math.max(n[2], 1e-6);
  const len = Math.hypot(x, y, z);
  return [x / len, y / len, z / len];
}

/* ------------------------------------------------------------------------- *
 * TSL mirrors
 * ------------------------------------------------------------------------- */

/**
 * Two-layer height blend on the GPU, returning the second layer's weight.
 *
 * Mirrors {@link heightBlend2}. Kept branch-free: `max(x, 0)` on both terms is
 * cheaper than any conditional and the degenerate `total == 0` case is handled
 * by the epsilon rather than by an `If`, because a divergent branch here would
 * cost more than the epsilon ever does.
 */
export const heightBlend2Node = /*@__PURE__*/ Fn(
  ([h0, h1, w, depth]: [FloatNode, FloatNode, FloatNode, FloatNode]) => {
    const w0 = float(1).sub(w).toVar('hbW0');
    const w1 = w.toVar('hbW1');
    const s0 = h0.add(w0).toVar('hbS0');
    const s1 = h1.add(w1).toVar('hbS1');
    const cut = max(s0, s1).sub(depth).toVar('hbCut');
    // The coverage factor both excludes zero-coverage layers exactly and makes
    // the wide-band limit a true linear lerp. Branch-free by construction.
    const b0 = max(s0.sub(cut), 0).mul(w0).toVar('hbB0');
    const b1 = max(s1.sub(cut), 0).mul(w1).toVar('hbB1');
    return b1.div(max(b0.add(b1), 1e-6));
  },
).setLayout({
  name: 'heightBlend2',
  type: 'float',
  inputs: [
    { name: 'h0', type: 'float' },
    { name: 'h1', type: 'float' },
    { name: 'w', type: 'float' },
    { name: 'depth', type: 'float' },
  ],
});

/**
 * Three-way height blend weights, returned packed into a vec3.
 *
 * Used by triplanar projection and by hex-tiling, both of which are inherently
 * three-way. Normalised with L1 so the mean is preserved.
 */
export const heightBlend3Node = /*@__PURE__*/ Fn(
  ([heights, weights, depth]: [Vec3Node, Vec3Node, FloatNode]) => {
    const s = heights.add(weights).toVar('hb3S');
    const cut = max(max(s.x, s.y), s.z).sub(depth).toVar('hb3Cut');
    const b = max(s.sub(cut), 0).mul(weights).toVar('hb3B');
    const total = b.x.add(b.y).add(b.z).toVar('hb3Total');
    return b.div(max(total, 1e-6));
  },
).setLayout({
  name: 'heightBlend3',
  type: 'vec3',
  inputs: [
    { name: 'heights', type: 'vec3' },
    { name: 'weights', type: 'vec3' },
    { name: 'depth', type: 'float' },
  ],
});

/** Variance-preserving normalisation of three weights. Mirrors {@link variancePreservingWeights}. */
export const variancePreserving3Node = /*@__PURE__*/ Fn(([weights]: [Vec3Node]) => {
  const sumSq = weights.dot(weights).toVar('vpSumSq');
  return weights.div(max(sumSq, 1e-12).sqrt());
}).setLayout({
  name: 'variancePreserving3',
  type: 'vec3',
  inputs: [{ name: 'weights', type: 'vec3' }],
});

/** Sharpen and L1-renormalise three weights. Mirrors {@link sharpenWeights}. */
export const sharpen3Node = /*@__PURE__*/ Fn(([weights, exponent]: [Vec3Node, FloatNode]) => {
  const raised = pow(max(weights, vec3(0)), vec3(exponent)).toVar('sharpRaised');
  return raised.div(max(raised.x.add(raised.y).add(raised.z), 1e-6));
}).setLayout({
  name: 'sharpen3',
  type: 'vec3',
  inputs: [
    { name: 'weights', type: 'vec3' },
    { name: 'exponent', type: 'float' },
  ],
});

/** Reoriented Normal Mapping on the GPU. Mirrors {@link reorientNormal}. */
export const reorientNormalNode = /*@__PURE__*/ Fn(([base, detail]: [Vec3Node, Vec3Node]) => {
  const t = base.add(vec3(0, 0, 1)).toVar('rnmT');
  const u = detail.mul(vec3(-1, -1, 1)).toVar('rnmU');
  return normalize(t.mul(t.dot(u).div(max(t.z, 1e-6))).sub(u));
}).setLayout({
  name: 'reorientNormal',
  type: 'vec3',
  inputs: [
    { name: 'base', type: 'vec3' },
    { name: 'detail', type: 'vec3' },
  ],
});

/** Renormalising XY scale of a tangent-space normal. Mirrors {@link scaleNormal}. */
export const scaleNormalNode = /*@__PURE__*/ Fn(([n, strength]: [Vec3Node, FloatNode]) => {
  return normalize(vec3(n.xy.mul(strength), max(n.z, 1e-6)));
}).setLayout({
  name: 'scaleNormalTs',
  type: 'vec3',
  inputs: [
    { name: 'n', type: 'vec3' },
    { name: 'strength', type: 'float' },
  ],
});

/**
 * Decode an RGB normal-map sample into a unit tangent-space normal.
 *
 * `z` is reconstructed rather than taken from the texture. Two-channel
 * reconstruction is robust to the BC5/JPEG chroma damage that leaves the blue
 * channel of a compressed normal map noticeably wrong, and it costs one
 * multiply-add plus a square root.
 */
export const decodeNormalNode = /*@__PURE__*/ Fn(([rgb]: [Vec3Node]) => {
  const xy = rgb.xy.mul(2).sub(1).toVar('nmXY');
  const z = max(float(1).sub(xy.dot(xy)), 0).sqrt();
  return vec3(xy, z);
}).setLayout({
  name: 'decodeNormalTs',
  type: 'vec3',
  inputs: [{ name: 'rgb', type: 'vec3' }],
});

/**
 * Blend a normal towards flat (`+Z`) by `amount`, renormalising.
 *
 * Used by the wetness system: standing water fills mesoscale relief, so a
 * flooded texel's shading normal must relax towards the geometric normal or
 * puddles look like shiny gravel instead of water.
 */
export const flattenNormalNode = /*@__PURE__*/ Fn(([n, amount]: [Vec3Node, FloatNode]) => {
  return normalize(mix(n, vec3(0, 0, 1), amount));
}).setLayout({
  name: 'flattenNormalTs',
  type: 'vec3',
  inputs: [
    { name: 'n', type: 'vec3' },
    { name: 'amount', type: 'float' },
  ],
});
