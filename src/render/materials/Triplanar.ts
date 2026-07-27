/**
 * @module render/materials/Triplanar
 *
 * World-space triplanar projection for terrain and cliffs.
 *
 * ### Why
 *
 * A heightfield terrain or a sculpted cliff has no usable UV parameterisation.
 * Planar-projecting a texture down the world Y axis is the usual fallback, and
 * it is exactly why so many games have cliffs covered in metre-long vertical
 * smears: at 90 degrees of slope the projected texel density along the vertical
 * axis goes to zero. Triplanar projection samples the texture three times —
 * once per world axis — and blends by the *geometric* normal, so a texel is
 * always sampled from the plane most nearly parallel to the surface.
 *
 * ### Blend weights
 *
 * `w = normalize(|n|^k)`. The exponent `k` controls how wide the transition
 * band is; k≈4-8 is the usable range. Below ~2 the three projections are
 * visibly superimposed everywhere (a triple-exposure look); above ~16 the
 * transition becomes a hard, straight seam that reads as a modelling error.
 *
 * Optionally the linear blend is replaced by the height-aware blend from
 * {@link module:render/materials/Blending}, which makes the axis transition
 * interlock along the mesostructure instead of cross-fading. It costs three
 * extra taps of a small single-channel height texture and is worth it on hero
 * cliffs, which is why it is opt-in per material rather than global.
 *
 * ### Normal mapping
 *
 * Naively blending three tangent-space normals and calling the result a world
 * normal is wrong — each was authored in a different tangent frame. This module
 * uses the "whiteout" blend from Ben Golus, "Normal Mapping for a Triplanar
 * Shader" (2017): swizzle each projection's tangent normal into world space,
 * add the geometric normal's corresponding components to the XY, take |n| for
 * the Z, then blend and normalise. Golus's own comparison puts whiteout within
 * a hair of full Reoriented Normal Mapping at roughly a third of the cost, and
 * unlike UDN it does not flatten detail on steep faces.
 *
 * Mirrored faces are handled with the sign correction from the same article:
 * without it, the +X and -X faces of a rock sample mirror-image texture, which
 * is instantly obvious on anything with directional detail such as strata.
 *
 * ### Cost
 *
 * Three taps per map. A triplanar material with albedo + normal + ORM is
 * 9 texture fetches per fragment against 3 for a UV-mapped one. That is the
 * price of admission for terrain and it is budgeted for explicitly in
 * {@link module:render/MaterialLibrary}: triplanar is used for ground and
 * cliffs only, never for props.
 */

import { Fn, abs, max, normalize, pow, step, texture, vec2, vec3 } from 'three/tsl';

import { decodeNormalNode, heightBlend3Node, scaleNormalNode } from './Blending';
import type { FloatNode, Vec2Node, Vec3Node, Vec4Node } from './types';

/** A bound texture sampler node, as produced by TSL's `texture()`. */
export type TextureSampler = ReturnType<typeof texture>;

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/** A 3-component vector as a plain tuple. */
export type Vec3Tuple = readonly [number, number, number];

/**
 * Triplanar blend weights for a world-space normal.
 *
 * @param n         world-space normal. Need not be unit length.
 * @param sharpness blend exponent; 1 is a plain `|n|` blend, higher narrows the
 *                  transition band. Clamped to at least 1.
 * @returns three non-negative weights summing to 1.
 *
 * Properties the tests pin, because a plausible-looking mistake violates them:
 * partition of unity for every normal (otherwise the surface darkens or blows
 * out along transitions), and exact one-hot behaviour for an axis-aligned
 * normal (otherwise flat ground is quietly cross-faded with two stretched
 * projections and loses contrast).
 */
export function triplanarWeights(n: Vec3Tuple, sharpness: number): [number, number, number] {
  const k = Math.max(sharpness, 1);
  const w: [number, number, number] = [
    Math.pow(Math.abs(n[0]), k),
    Math.pow(Math.abs(n[1]), k),
    Math.pow(Math.abs(n[2]), k),
  ];
  const total = w[0] + w[1] + w[2];
  if (!(total > 0)) return [0, 1, 0];
  return [w[0] / total, w[1] / total, w[2] / total];
}

/**
 * The per-axis UV pairs a world position projects to, before tiling.
 *
 * Returned in the order X-plane, Y-plane, Z-plane. The sign corrections are the
 * ones described in the module header; `axisSign` is `+1`/`-1` per component of
 * the normal, never 0.
 */
export function triplanarUVs(
  position: Vec3Tuple,
  n: Vec3Tuple,
): { uvX: [number, number]; uvY: [number, number]; uvZ: [number, number] } {
  const sx = n[0] < 0 ? -1 : 1;
  const sy = n[1] < 0 ? -1 : 1;
  const sz = n[2] < 0 ? -1 : 1;
  return {
    uvX: [position[2] * sx, position[1]],
    uvY: [position[0] * sy, position[2]],
    uvZ: [position[0] * -sz, position[1]],
  };
}

/* ------------------------------------------------------------------------- *
 * TSL
 * ------------------------------------------------------------------------- */

/** Everything the three projections share, computed once per fragment. */
export interface TriplanarFrame {
  /** Normalised blend weights, `x`/`y`/`z` for the X-, Y- and Z-plane. */
  readonly weights: Vec3Node;
  readonly uvX: Vec2Node;
  readonly uvY: Vec2Node;
  readonly uvZ: Vec2Node;
  /** `+1`/`-1` per axis of the geometric normal. Never 0. */
  readonly axisSign: Vec3Node;
  /** The geometric world normal the frame was built from. */
  readonly normal: Vec3Node;
}

/**
 * Build the shared triplanar frame.
 *
 * @param positionWorld world position of the fragment, in metres.
 * @param normalWorld   *geometric* world normal. Passing the normal-mapped one
 *                      is a classic bug: the projection would then swim as the
 *                      normal map perturbs it.
 * @param tiling        tiling rate in repeats per metre. One texture tile
 *                      covers `1 / tiling` metres.
 * @param sharpness     blend exponent.
 */
export function triplanarFrame(
  positionWorld: Vec3Node,
  normalWorld: Vec3Node,
  tiling: FloatNode,
  sharpness: FloatNode,
): TriplanarFrame {
  const n = normalWorld.toVar('tpNormal');
  const scaled = positionWorld.mul(tiling).toVar('tpPos');

  // `sign()` returns 0 at exactly 0, which would collapse a UV axis on a
  // perfectly axis-aligned normal — and axis-aligned normals are precisely what
  // flat ground produces, so that degenerate case is the common one. `step`
  // maps to {0,1} with no zero-valued middle, and the affine remap turns it
  // into a clean +/-1.
  const axisSign = step(vec3(0), n).mul(2).sub(1).toVar('tpAxisSign');

  const w = pow(abs(n), vec3(sharpness)).toVar('tpWeightsRaw');
  const weights = w.div(max(w.x.add(w.y).add(w.z), 1e-6)).toVar('tpWeights');

  return {
    weights,
    uvX: vec2(scaled.z.mul(axisSign.x), scaled.y).toVar('tpUvX'),
    uvY: vec2(scaled.x.mul(axisSign.y), scaled.z).toVar('tpUvY'),
    uvZ: vec2(scaled.x.mul(axisSign.z.negate()), scaled.y).toVar('tpUvZ'),
    axisSign,
    normal: n,
  };
}

/**
 * A frame at a different tiling rate, sharing this one's blend weights.
 *
 * The detail layer samples at ten to twenty times the base rate but must blend
 * across exactly the same axis transitions, and recomputing the weights would
 * cost three `pow`s for an identical answer.
 */
export function withTiling(frame: TriplanarFrame, factor: number): TriplanarFrame {
  return {
    ...frame,
    uvX: frame.uvX.mul(factor).toVar('tpUvXDetail'),
    uvY: frame.uvY.mul(factor).toVar('tpUvYDetail'),
    uvZ: frame.uvZ.mul(factor).toVar('tpUvZDetail'),
  };
}

/**
 * Replace the frame's linear weights with height-aware ones.
 *
 * @param frame  a frame from {@link triplanarFrame}.
 * @param height single-channel mesoscale height texture, matching the material.
 * @param depth  transition band width; see {@link heightBlend3Node}.
 * @returns a new frame; the original is left untouched so a caller can keep
 *          both (linear weights for cheap channels, height weights for albedo).
 *
 * Costs three extra taps. Only worth binding on surfaces where the axis seam is
 * actually visible — cliff faces near the camera.
 */
export function withHeightBlend(
  frame: TriplanarFrame,
  height: TextureSampler,
  depth: FloatNode,
): TriplanarFrame {
  const heights = vec3(
    height.sample(frame.uvX).r,
    height.sample(frame.uvY).r,
    height.sample(frame.uvZ).r,
  ).toVar('tpHeights');
  return { ...frame, weights: heightBlend3Node(heights, frame.weights, depth).toVar('tpHbWeights') };
}

/** Blend three per-axis samples with the frame's weights. */
export function triplanarBlend(x: Vec4Node, y: Vec4Node, z: Vec4Node, frame: TriplanarFrame): Vec4Node {
  return x.mul(frame.weights.x).add(y.mul(frame.weights.y)).add(z.mul(frame.weights.z));
}

/**
 * Sample a texture triplanarly.
 *
 * The three taps are unconditional. Branching on the weights to skip a
 * near-zero projection is a false economy on a GPU: the branch is almost always
 * divergent across a quad, so both sides execute anyway, and the derivative
 * chain inside a non-uniform branch is undefined.
 */
export function triplanarSample(tex: TextureSampler, frame: TriplanarFrame): Vec4Node {
  return triplanarBlend(
    tex.sample(frame.uvX),
    tex.sample(frame.uvY),
    tex.sample(frame.uvZ),
    frame,
  );
}

/**
 * Whiteout-blend three tangent-space normal samples into a world-space normal.
 *
 * Golus 2017. `strength` scales the tangent-space XY before the swizzle, so it
 * behaves the same way as a normal map scale on a UV-mapped surface.
 */
export const whiteoutBlendNode = /*@__PURE__*/ Fn(
  ([tnX, tnY, tnZ, normal, weights]: [Vec3Node, Vec3Node, Vec3Node, Vec3Node, Vec3Node]) => {
    const absN = abs(normal).toVar('woAbsN');
    // Swizzle each projection's tangent normal so its XY lines up with the two
    // world axes that plane spans, then add the geometric normal's components.
    // Taking |n| for Z is what keeps the sum from cancelling on back faces.
    const nx = vec3(tnX.xy.add(normal.zy), absN.x);
    const ny = vec3(tnY.xy.add(normal.xz), absN.y);
    const nz = vec3(tnZ.xy.add(normal.xy), absN.z);
    return normalize(
      nx.zyx.mul(weights.x).add(ny.xzy.mul(weights.y)).add(nz.xyz.mul(weights.z)),
    );
  },
).setLayout({
  name: 'triplanarWhiteout',
  type: 'vec3',
  inputs: [
    { name: 'tnX', type: 'vec3' },
    { name: 'tnY', type: 'vec3' },
    { name: 'tnZ', type: 'vec3' },
    { name: 'normal', type: 'vec3' },
    { name: 'weights', type: 'vec3' },
  ],
});

/**
 * The three per-axis tangent-space normals, *before* the mirroring correction.
 *
 * Exposed separately from {@link blendTriplanarNormals} so that the detail
 * layer can compose with the base normal *inside each projection's own tangent
 * frame*, which is the only place where the two normals are actually
 * comparable. Composing after the world-space blend would need a synthesised
 * tangent basis and detail would visibly swim through every axis transition.
 */
export function triplanarTangentNormals(
  tex: TextureSampler,
  frame: TriplanarFrame,
  strength: FloatNode,
): [Vec3Node, Vec3Node, Vec3Node] {
  return [
    scaleNormalNode(decodeNormalNode(tex.sample(frame.uvX).xyz), strength),
    scaleNormalNode(decodeNormalNode(tex.sample(frame.uvY).xyz), strength),
    scaleNormalNode(decodeNormalNode(tex.sample(frame.uvZ).xyz), strength),
  ];
}

/**
 * Apply the mirroring correction and whiteout-blend into a world-space normal.
 *
 * The per-axis X flips undo the UV mirroring applied in {@link triplanarFrame};
 * skip them and opposite faces of a rock get mirror-image strata.
 */
export function blendTriplanarNormals(
  tangentNormals: readonly [Vec3Node, Vec3Node, Vec3Node],
  frame: TriplanarFrame,
): Vec3Node {
  const flip = (n: Vec3Node, sign: FloatNode): Vec3Node => vec3(n.x.mul(sign), n.y, n.z);
  return whiteoutBlendNode(
    flip(tangentNormals[0], frame.axisSign.x),
    flip(tangentNormals[1], frame.axisSign.y),
    flip(tangentNormals[2], frame.axisSign.z.negate()),
    frame.normal,
    frame.weights,
  );
}

/** Convenience: sample a normal map triplanarly and return the world normal. */
export function triplanarNormal(
  tex: TextureSampler,
  frame: TriplanarFrame,
  strength: FloatNode,
): Vec3Node {
  return blendTriplanarNormals(triplanarTangentNormals(tex, frame, strength), frame);
}
