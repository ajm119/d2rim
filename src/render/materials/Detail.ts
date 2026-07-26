/**
 * @module render/materials/Detail
 *
 * The second UV scale: high-frequency normal and albedo detail that fades in as
 * the camera approaches, so surfaces do not go flat and plastic at arm's length.
 *
 * ### The problem
 *
 * A 2K albedo tiled at one repeat per two metres gives ~1 mm texels. That is
 * plenty at three metres away and completely inadequate at thirty centimetres,
 * which is where a first-person camera spends a lot of its time in a game about
 * hitting things. Raising the base tiling rate is not the answer: it makes the
 * repetition period shorter and the tiling *more* obvious at distance. The
 * standard solution — used by essentially every open-world renderer since
 * Crysis — is a second, much finer, low-contrast layer that only exists near
 * the camera.
 *
 * ### Normal composition
 *
 * Detail normals compose with the base normal via Reoriented Normal Mapping
 * (Barré-Brisebois & Hill, "Blending in Detail", 2012) — see
 * {@link module:render/materials/Blending}. Doing it in *tangent* space is
 * important: for triplanar surfaces that means composing inside each
 * projection's frame before the world-space blend, which is why
 * {@link detailTriplanarTangentNormals} returns three normals rather than one.
 *
 * ### Detail albedo without a detail albedo texture
 *
 * No CC0 detail-albedo set is reachable from this build, so the albedo grain is
 * derived from the detail normal's own Z component. This is not a hack for its
 * own sake: `z` of a unit tangent normal is `cos θ` of the micro-facet tilt, and
 * a tilted micro-facet is on average more occluded by its neighbours than a flat
 * one. Using `mix(1, z, k)` as a multiplicative albedo term is therefore a
 * first-order micro-occlusion, and it correlates the darkening with the same
 * bumps the normal is bending around — which is exactly the correlation a real
 * detail albedo would have and a random noise texture would not.
 * {@link DetailSpec.albedoKey} takes over the moment a real map exists.
 *
 * ### Two fades, not one
 *
 * Distance fade alone is not enough. At grazing angles a fine detail layer
 * exceeds one texel per pixel long before it exceeds the distance threshold, and
 * the result is a shimmering moiré that no amount of anisotropy fixes because
 * the *content* is aliasing, not the filtering. So the fade is the product of a
 * distance term and a screen-space texel-density term measured with `fwidth`.
 * The density term is what lets the distance fade be generous.
 */

import { Fn, float, fwidth, max, mix, positionView, saturate, smoothstep, vec3 } from 'three/tsl';

import { decodeNormalNode, reorientNormalNode, scaleNormalNode } from './Blending';
import type { TextureSampler, TriplanarFrame } from './Triplanar';
import type { FloatNode, Vec2Node, Vec3Node } from './types';

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/**
 * Distance fade for the detail layer: 1 up close, 0 past `fadeEnd`.
 *
 * Smoothstep rather than a linear ramp because a linear fade has a slope
 * discontinuity at both ends, and a discontinuity in the *derivative* of a
 * surface property is visible as a ring on the ground when the camera moves.
 *
 * `fadeStart >= fadeEnd` is treated as "no detail at all", which is how the
 * quality tiers switch the layer off without a separate flag.
 */
export function detailDistanceFade(distance: number, fadeStart: number, fadeEnd: number): number {
  if (!(fadeEnd > fadeStart)) return 0;
  const t = (distance - fadeStart) / (fadeEnd - fadeStart);
  const c = Math.min(Math.max(t, 0), 1);
  return 1 - c * c * (3 - 2 * c);
}

/**
 * Texel-density fade: 1 while the detail layer is comfortably resolved, falling
 * to 0 as it approaches one cycle per pixel.
 *
 * @param texelWidth UV-space width of one pixel footprint, i.e. `fwidth(uv)`,
 *                   in units of the detail texture's own UV space.
 *
 * Nyquist says a signal is unrecoverable at half a cycle per pixel. Detail maps
 * are broadband, so the practical threshold sits lower; fading between 0.25 and
 * 0.5 texels per pixel keeps the layer alive as long as it carries information
 * and removes it before it turns into noise.
 */
export function detailDensityFade(texelWidth: number): number {
  const t = (texelWidth - 0.25) / 0.25;
  const c = Math.min(Math.max(t, 0), 1);
  return 1 - c * c * (3 - 2 * c);
}

/**
 * Combined fade. The product, not the minimum: two independent reasons to fade
 * should compound, and `min` would produce a visible crease where the active
 * term changes over.
 */
export function detailFade(distance: number, texelWidth: number, start: number, end: number): number {
  return detailDistanceFade(distance, start, end) * detailDensityFade(texelWidth);
}

/* ------------------------------------------------------------------------- *
 * TSL
 * ------------------------------------------------------------------------- */

/**
 * The detail fade, evaluated per fragment. Mirrors {@link detailFade}.
 *
 * @param detailUv the *detail-scale* UV. The density term is measured on this,
 *                 not on the base UV, because they differ by an order of
 *                 magnitude and it is the fine layer that aliases.
 */
export const detailFadeNode = /*@__PURE__*/ Fn(
  ([detailUv, fadeStart, fadeEnd]: [Vec2Node, FloatNode, FloatNode]) => {
    const distance = positionView.length().toVar('detailDist');
    const byDistance = smoothstep(fadeStart, max(fadeEnd, fadeStart.add(1e-3)), distance)
      .oneMinus()
      .toVar('detailByDistance');
    // `fwidth` of the UV is the screen-space footprint of one pixel measured in
    // texture space; comparing it against a fixed threshold is a per-fragment
    // Nyquist test that costs two derivative instructions.
    const width = fwidth(detailUv).length().toVar('detailWidth');
    const byDensity = smoothstep(float(0.25), float(0.5), width).oneMinus();
    return saturate(byDistance.mul(byDensity));
  },
).setLayout({
  name: 'detailFade',
  type: 'float',
  inputs: [
    { name: 'detailUv', type: 'vec2' },
    { name: 'fadeStart', type: 'float' },
    { name: 'fadeEnd', type: 'float' },
  ],
});

/**
 * Sample the detail normal map and scale it by strength and fade in one step.
 *
 * Scaling by `strength * fade` before the RNM composition — rather than lerping
 * the composed normal afterwards — is deliberate: it keeps the detail normal a
 * unit vector at every fade value, so the composition never has to renormalise
 * a shortened vector and the fade is perceptually linear in *slope*.
 */
export function detailTangentNormal(
  tex: TextureSampler,
  uv: Vec2Node,
  strength: FloatNode,
  fade: FloatNode,
): Vec3Node {
  return scaleNormalNode(decodeNormalNode(tex.sample(uv).xyz), strength.mul(fade));
}

/** The same, sampled triplanarly: one detail normal per projection plane. */
export function detailTriplanarTangentNormals(
  tex: TextureSampler,
  frame: TriplanarFrame,
  strength: FloatNode,
  fade: FloatNode,
): [Vec3Node, Vec3Node, Vec3Node] {
  const s = strength.mul(fade);
  return [
    scaleNormalNode(decodeNormalNode(tex.sample(frame.uvX).xyz), s),
    scaleNormalNode(decodeNormalNode(tex.sample(frame.uvY).xyz), s),
    scaleNormalNode(decodeNormalNode(tex.sample(frame.uvZ).xyz), s),
  ];
}

/** Compose a detail tangent normal onto a base tangent normal (RNM). */
export function composeDetailNormal(base: Vec3Node, detail: Vec3Node): Vec3Node {
  return reorientNormalNode(base, detail);
}

/**
 * Albedo grain derived from a detail tangent normal.
 *
 * See the module header for why `z` is the right signal. Returns a multiplier
 * centred on 1 so it can be applied to any base albedo without shifting its
 * average brightness much: `mix(1, z, k)` has mean `1 - k(1 - E[z])`, and for
 * the detail maps we ship `E[z] ≈ 0.95`, so a strength of 0.4 costs about 2%
 * brightness. That is small enough to leave the archetype's authored albedo
 * meaning what it says.
 */
export const detailAlbedoGainNode = /*@__PURE__*/ Fn(
  ([detailNormal, strength]: [Vec3Node, FloatNode]) => {
    return mix(float(1), max(detailNormal.z, 0), saturate(strength));
  },
).setLayout({
  name: 'detailAlbedoGain',
  type: 'float',
  inputs: [
    { name: 'detailNormal', type: 'vec3' },
    { name: 'strength', type: 'float' },
  ],
});

/**
 * Albedo modulation from a real detail albedo texture, when one is bound.
 *
 * Detail albedo is an *overlay*, not a replacement: it carries only the
 * high-frequency departure from the base colour. Treating the texture's
 * mid-grey as "no change" and scaling around it is the standard convention and
 * is what makes a detail map reusable across substances.
 */
export const detailAlbedoOverlayNode = /*@__PURE__*/ Fn(
  ([base, detail, strength]: [Vec3Node, Vec3Node, FloatNode]) => {
    const delta = detail.sub(vec3(0.5)).mul(2).toVar('detailDelta');
    return max(base.mul(float(1).add(delta.mul(strength))), vec3(0));
  },
).setLayout({
  name: 'detailAlbedoOverlay',
  type: 'vec3',
  inputs: [
    { name: 'base', type: 'vec3' },
    { name: 'detail', type: 'vec3' },
    { name: 'strength', type: 'float' },
  ],
});
