/**
 * @module render/materials/Parallax
 *
 * Parallax occlusion mapping for deep-relief surfaces: masonry, cobbles,
 * plank floors.
 *
 * ### Technique
 *
 * Ray-march the view vector through the height volume in tangent space, find
 * the first layer where the ray falls below the height field, then refine the
 * crossing by linear interpolation between the last two samples. This is
 * Brawley & Tatarchuk, "Parallax Occlusion Mapping: Self-Shadowing, Perceptually
 * Accurate Bump Mapping" (ShaderX³, 2004), refined in Tatarchuk, "Dynamic
 * Parallax Occlusion Mapping with Approximate Soft Shadows" (I3D 2006).
 *
 * Three details separate a POM that reads as real geometry from one that reads
 * as a wobbling texture:
 *
 * 1. **Adaptive step count.** The number of layers is interpolated by `N·V`.
 *    At grazing incidence the ray crosses many layers of the height field per
 *    unit of UV travel and needs the steps; head-on it needs almost none.
 *    A fixed step count either shimmers at grazing angles or wastes half the
 *    shader on flat ground.
 * 2. **Offset limiting.** The naive step `V.xy / V.z` diverges as `V.z → 0`, so
 *    a surface seen edge-on samples arbitrarily far away and smears. Clamping
 *    the divisor (Welsh, "Parallax Mapping with Offset Limiting", Infiscape
 *    2004) bounds the total displacement at the cost of under-parallaxing the
 *    most grazing angles, which is the right trade: an under-displaced surface
 *    looks flat, an over-displaced one looks broken.
 * 3. **Interpolated crossing.** Taking the last sampled layer as the answer
 *    quantises the surface into visible terraces. One linear solve between the
 *    two bracketing samples removes them entirely and costs nothing.
 *
 * ### Silhouette clipping
 *
 * POM displaces the *sampled* texel, not the geometry, so a raised brick at the
 * edge of a wall does not actually stick out past the polygon — the wall's
 * outline stays flat. For a bounded, non-tiling patch the honest fix is to
 * discard fragments whose marched UV leaves the `[0,1]` domain: the ray has
 * walked off the end of the height volume and there is nothing there. That
 * yields a correctly ragged silhouette along the patch border.
 *
 * This is only correct for a bounded patch. On tiling ground the marched UV
 * leaves `[0,1]` constantly and clipping would punch holes in the world, which
 * is why {@link ParallaxSpec.clipSilhouette} is per-archetype and off by
 * default.
 *
 * ### Deliberately not implemented
 *
 * Tatarchuk's approximate soft self-shadowing needs a second march towards the
 * light per light. At the light counts this project runs, that is a worse use
 * of the budget than the contact shadows the AO pass already provides, so it is
 * left out. The height field is exposed on {@link ParallaxResult} so a future
 * pass can add it without touching this one.
 */

import {
  Break,
  Discard,
  Fn,
  If,
  Loop,
  abs,
  float,
  max,
  min,
  mix,
  normalize,
  positionView,
  saturate,
  smoothstep,
  vec2,
  vec4,
} from 'three/tsl';

import type { FloatNode, Vec2Node, Vec3Node } from './types';

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/**
 * Layer count for a given incidence.
 *
 * @param nDotV  `|dot(N, V)|` in `[0, 1]`; 1 is head-on.
 * @returns the interpolated step count, not rounded — the shader uses it as a
 *          continuous layer depth so the transition between step counts is
 *          smooth rather than popping.
 */
export function parallaxStepCount(nDotV: number, minSteps: number, maxSteps: number): number {
  const t = Math.min(Math.max(Math.abs(nDotV), 0), 1);
  return maxSteps + (minSteps - maxSteps) * t;
}

/**
 * The per-layer UV step, with offset limiting.
 *
 * @param viewTangent unit view direction in tangent space, pointing *away* from
 *                    the surface (towards the eye).
 * @param scale       height-volume depth in UV units.
 * @param layers      layer count from {@link parallaxStepCount}.
 * @param minZ        floor applied to `V.z`. 0.35 bounds the maximum
 *                    displacement at ~2.9x the scale.
 */
export function parallaxStepUV(
  viewTangent: readonly [number, number, number],
  scale: number,
  layers: number,
  minZ = 0.35,
): [number, number] {
  const z = Math.max(Math.abs(viewTangent[2]), minZ);
  const n = Math.max(layers, 1);
  return [
    (viewTangent[0] / z) * (scale / n),
    (viewTangent[1] / z) * (scale / n),
  ];
}

/**
 * Refine a bracketed height-field crossing by linear interpolation.
 *
 * At the moment the march terminates we hold two samples: the previous layer,
 * where the ray was still above the surface, and the current one, where it is
 * below. Both the ray depth and the height field are treated as linear across
 * that interval, and the crossing solves in closed form.
 *
 * @param prevRay    ray depth at the previous layer.
 * @param prevHeight surface depth sampled at the previous layer.
 * @param currRay    ray depth at the current layer.
 * @param currHeight surface depth sampled at the current layer.
 * @returns the fraction in `[0, 1]` of the way from the current sample *back
 *          towards* the previous one at which the crossing occurs.
 *
 * Both quantities are expressed as *depth below the reference plane*, so the
 * ray starts at 0 and increases, and the crossing is where `ray == height`.
 */
export function parallaxCrossing(
  prevRay: number,
  prevHeight: number,
  currRay: number,
  currHeight: number,
): number {
  const after = currHeight - currRay;
  const before = prevHeight - prevRay;
  const denom = after - before;
  if (Math.abs(denom) < 1e-9) return 0;
  return Math.min(Math.max(after / denom, 0), 1);
}

/**
 * Effective step count after quality scaling, clamped to a sane range.
 *
 * The maximum is a hard 64: past that the march costs more than the surface is
 * worth and the height fields we derive are only 256² anyway, so extra steps
 * resolve nothing that is actually in the data.
 */
export function scaledParallaxSteps(steps: number, scale: number): number {
  return Math.max(1, Math.min(64, Math.round(steps * scale)));
}

/* ------------------------------------------------------------------------- *
 * TSL
 * ------------------------------------------------------------------------- */

/** What a parallax march produced. */
export interface ParallaxResult {
  /** The displaced UV every other map must be sampled with. */
  readonly uv: Vec2Node;
  /** Height at the hit, in `[0, 1]`. Reused as the cavity term downstream. */
  readonly height: FloatNode;
  /** 1 where parallax ran at full strength, 0 where it faded out by distance. */
  readonly strength: FloatNode;
}

export interface ParallaxOptions {
  /**
   * Samples the mesoscale height at a UV, in `[0, 1]` with 1 = highest.
   *
   * A function rather than a texture, because the height may live in any
   * channel: it is the blue channel of the packed surface texture when packing
   * succeeded and the red channel of a standalone map when it did not, and the
   * march must not have to know which.
   */
  readonly sampleHeight: (uv: Vec2Node) => FloatNode;
  /** Base UV before displacement. */
  readonly uv: Vec2Node;
  /** Unit view direction in tangent space, pointing towards the eye. */
  readonly viewTangent: Vec3Node;
  /** Height-volume depth, in UV units of the tiled texture. */
  readonly scale: FloatNode;
  /** Steps at grazing incidence. Compile-time constant: it bounds the loop. */
  readonly maxSteps: number;
  /** Steps at normal incidence. */
  readonly minSteps: number;
  /** Metres at which parallax has faded to nothing. */
  readonly fadeEnd: FloatNode;
  /** Discard fragments that march off the `[0,1]` tile. */
  readonly clipSilhouette: boolean;
}

/**
 * March the height field and return the displaced UV.
 *
 * The whole march lives inside a single `Fn` and is evaluated *once* per
 * fragment: the result is packed into a `vec4` and stored in a variable, and
 * every downstream slot (albedo, normal, roughness, AO) reads that variable.
 * This is not an optimisation detail, it is a correctness one — building the
 * march separately per material slot would run it three or four times per
 * fragment, and a POM material would silently cost four times what its budget
 * says.
 *
 * The loop bound is a JavaScript number, not a node, so it becomes a literal in
 * the generated shader. That matters on the WebGL2 backend, where several
 * drivers refuse to unroll or even to compile a loop with a non-constant bound,
 * and it matters on WebGPU because a uniform-bounded loop defeats the
 * compiler's ability to hoist the texture fetches. The *effective* step count
 * still varies per fragment — it is expressed as the layer depth, with an early
 * `Break` — so head-on fragments exit after a couple of iterations.
 */
export function parallaxOcclusion(options: ParallaxOptions): ParallaxResult {
  const { sampleHeight, uv, viewTangent, scale, maxSteps, minSteps, fadeEnd, clipSilhouette } =
    options;
  const steps = Math.max(1, Math.round(maxSteps));
  const floor = Math.max(1, Math.min(minSteps, steps));

  const marched = Fn(() => {
    const v = normalize(viewTangent).toVar('pomView');

    // Fade parallax out with distance. Past a few metres the displacement is
    // sub-pixel and the march is pure cost; worse, the mip level rises with
    // distance until the height field the march reads no longer matches the
    // one the normal map was built from, and the surface starts to swim.
    const distance = positionView.length().toVar('pomDist');
    const strength = smoothstep(fadeEnd.mul(0.6), fadeEnd, distance)
      .oneMinus()
      .toVar('pomStrength');

    const layers = mix(float(steps), float(floor), abs(v.z)).toVar('pomLayers');
    const layerDepth = float(1).div(max(layers, 1)).toVar('pomLayerDepth');

    // Offset limiting: clamp |V.z| so the per-layer step cannot diverge at
    // grazing incidence. See the module header.
    const limitedZ = max(abs(v.z), 0.35).toVar('pomLimitedZ');
    const deltaUv = v.xy.div(limitedZ).mul(scale.mul(strength)).mul(layerDepth).toVar('pomDeltaUv');

    const currentUv = uv.toVar('pomUv');
    const currentRay = float(0).toVar('pomRayDepth');
    // `1 - height` converts the height map's "up" convention into depth below
    // the reference plane, which is the space the march is expressed in.
    const currentDepth = sampleHeight(currentUv).oneMinus().toVar('pomSurfaceDepth');
    const prevUv = currentUv.toVar('pomPrevUv');
    const prevRay = float(0).toVar('pomPrevRay');
    const prevDepth = currentDepth.toVar('pomPrevDepth');

    Loop({ start: 0, end: steps, type: 'int' }, () => {
      If(currentRay.greaterThanEqual(currentDepth), () => {
        Break();
      });
      prevUv.assign(currentUv);
      prevRay.assign(currentRay);
      prevDepth.assign(currentDepth);

      currentUv.subAssign(deltaUv);
      currentRay.addAssign(layerDepth);
      currentDepth.assign(sampleHeight(currentUv).oneMinus());
    });

    // Linear solve for the crossing; mirrors `parallaxCrossing`.
    //
    // At loop exit `after <= 0` (the ray has passed below the surface) and
    // `before > 0` (at the previous layer it had not), so `denom` is negative
    // by construction. Clamping it to at most -1e-6 handles the one degenerate
    // case — a height of exactly 1 at the very first sample, where the loop
    // breaks before stepping — without a branch.
    const after = currentDepth.sub(currentRay).toVar('pomAfter');
    const before = prevDepth.sub(prevRay).toVar('pomBefore');
    const denom = min(after.sub(before), -1e-6).toVar('pomDenom');
    const t = saturate(after.div(denom)).toVar('pomT');

    const hitUv = mix(currentUv, prevUv, t).toVar('pomHitUv');
    const hitDepth = mix(currentDepth, prevDepth, t).toVar('pomHitDepth');

    if (clipSilhouette) {
      // The ray walked off the end of the height volume: there is no surface
      // here, so the fragment is genuinely not part of the object.
      Discard(
        hitUv.x
          .lessThan(0)
          .or(hitUv.x.greaterThan(1))
          .or(hitUv.y.lessThan(0))
          .or(hitUv.y.greaterThan(1)),
      );
    }

    // Blend back towards the undisplaced UV as the effect fades, so there is no
    // discontinuity at the fade boundary.
    return vec4(mix(uv, hitUv, strength), hitDepth.oneMinus(), strength);
  })().toVar('pomResult');

  return { uv: marched.xy, height: marched.z, strength: marched.w };
}

/**
 * Clamp a UV into the unit tile.
 *
 * Used on non-clipping parallax surfaces that are nonetheless UV-bounded (an
 * atlas region, say) so the march cannot bleed into a neighbouring chart.
 */
export const clampTileUvNode = /*@__PURE__*/ Fn(([uv]: [Vec2Node]) => {
  return min(max(uv, vec2(0)), vec2(1));
}).setLayout({
  name: 'clampTileUv',
  type: 'vec2',
  inputs: [{ name: 'uv', type: 'vec2' }],
});
