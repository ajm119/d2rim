/**
 * @module render/materials/Wetness
 *
 * The global wetness system. Rain-soaked is the Blood Moor's resting state, so
 * this is not an effect bolted on top of the materials — it is part of what
 * every material *is*.
 *
 * ### The physical model
 *
 * Water interacts with a surface in two distinct ways, and conflating them is
 * why so many "wet" shaders look like varnish:
 *
 * 1. **Absorption.** Water soaks into the pore structure and replaces the air
 *    between the grains. Because water's refractive index (1.33) is much closer
 *    to the substrate's (~1.5) than air's is, light entering the material is
 *    refracted less at each grain boundary, scatters *forward* more, and takes
 *    a longer path before it escapes. A longer path through an absorbing medium
 *    means more absorption — the surface darkens.
 *
 *    Beer-Lambert gives the exact shape of that darkening: transmittance is
 *    `exp(-α·d)`, so multiplying the optical path length by `(1 + s)` maps
 *    reflectance `c` to `c^(1+s)`. That is the model used here, and it is worth
 *    a paragraph because it is *also* what buys the art direction its colour
 *    separation. A plain multiplicative darkening (`albedo *= 0.6`) preserves
 *    hue ratios and turns everything towards grey mush. The exponential form
 *    darkens the absorbing channels faster than the reflecting ones, so wet mud
 *    goes *browner*, wet moss goes *greener*, and dried blood goes *redder*
 *    instead of all three converging on the same dark grey.
 *
 * 2. **Film formation.** Water that does not soak in forms a smooth layer on
 *    top. That layer is a new, much smoother specular interface. It barely
 *    changes the normal-incidence reflectance — 0.020 for air/water against
 *    0.04 for a typical dielectric — but it collapses the roughness, which
 *    concentrates the same energy into a far narrower lobe. That concentration,
 *    not a reflectance increase, is what "shiny when wet" actually is.
 *
 * **Porosity decides the split.** A porous substance (mud, cloth, unglazed
 * brick) absorbs nearly all the water: large darkening, almost no film, so it
 * gets darker and richer but stays matte. A non-porous one (iron, polished
 * stone, skin) absorbs almost none: no darkening at all, but a mirror-smooth
 * film. Getting this backwards produces the classic error of glossy mud and
 * matte wet metal.
 *
 * ### Cavity accumulation
 *
 * Water runs downhill. Standing water accumulates in mesoscale cavities long
 * before it covers the peaks, so the film term is gated by a puddle mask
 * derived from the same height/AO field the rest of the material uses. This is
 * what makes wetness read as *water on a surface* rather than a uniform
 * varnish, and it is also what gives screen-space reflections something
 * spatially varying to bite on.
 *
 * ### References
 *
 * - Sébastien Lagarde, "Water drop 2b — Dynamic rain and its effects" (2013),
 *   and "Water drop 3b — Physically based wet surfaces" (2013).
 * - Lagarde & de Rousiers, "Moving Frostbite to Physically Based Rendering 3.0"
 *   (SIGGRAPH 2014 course), §5.4 on wet surface parameters.
 * - Jensen et al., "A Physically-Based Night Sky Model" is *not* the source
 *   here; the two-interface reflectance below is the standard incoherent
 *   thin-film series, e.g. Stokes' formula for a stack of two interfaces.
 */

import * as THREE from 'three/webgpu';
import { Fn, clamp, float, max, min, mix, pow, saturate, smoothstep, uniform, vec3 } from 'three/tsl';

import { flattenNormalNode } from './Blending';
import type { FloatNode, Vec3Node } from './types';

/* ------------------------------------------------------------------------- *
 * Physical constants
 * ------------------------------------------------------------------------- */

/** Refractive index of water at visible wavelengths. */
export const IOR_WATER = 1.333;

/**
 * Normal-incidence reflectance of the air/water interface.
 *
 * `((n - 1) / (n + 1))²` with `n = 1.333`.
 */
export const F0_WATER = ((IOR_WATER - 1) / (IOR_WATER + 1)) ** 2;

/**
 * Perceptual roughness of a still water film.
 *
 * Not zero. A perfectly smooth surface produces a delta-function specular that
 * punctual lights render as a single blown-out pixel, and real standing water
 * always carries some sub-millimetre disturbance. 0.045 is smooth enough to
 * mirror the sky and rough enough to give a torch a highlight with a shape.
 */
export const WATER_ROUGHNESS = 0.045;

/**
 * Maximum multiplier on the optical path length at full saturation.
 *
 * `albedo^(1 + s)` with `s` up to this value. 1.1 puts a mid-grey 0.18 albedo
 * at 0.05 when fully soaked, which matches how dramatically dry sand darkens
 * when a wave runs over it.
 */
export const MAX_PATH_EXTENSION = 1.1;

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/** Substrate IOR implied by a dielectric normal-incidence reflectance. */
export function iorFromF0(f0: number): number {
  const s = Math.sqrt(Math.max(Math.min(f0, 0.99), 0));
  return (1 + s) / (1 - s);
}

/** Normal-incidence reflectance of an interface between two media. */
export function fresnelF0(n1: number, n2: number): number {
  return ((n2 - n1) / (n2 + n1)) ** 2;
}

/**
 * Normal-incidence reflectance of a water film over a dielectric substrate.
 *
 * Two interfaces in series — air/water and water/substrate — summed
 * incoherently over the infinite internal reflections (Stokes):
 *
 * ```
 * R = R₁ + (1 - R₁)² R₂ / (1 - R₁R₂)
 * ```
 *
 * The result for a typical `F₀ = 0.04` substrate is ≈ 0.0236, i.e. *lower* than
 * the dry surface. That is the physically correct and frequently surprising
 * answer: a wet dielectric does not reflect more light at normal incidence, it
 * reflects it in a much tighter lobe. Any implementation that raises `F₀` to
 * make things "look wetter" is compensating for a roughness term it failed to
 * drop far enough.
 *
 * @param dryF0    substrate normal-incidence reflectance.
 * @param coverage fraction of the surface covered by a continuous film, 0-1.
 */
export function wetSpecularF0(dryF0: number, coverage: number): number {
  const c = Math.min(Math.max(coverage, 0), 1);
  const nSub = iorFromF0(dryF0);
  const r1 = F0_WATER;
  const r2 = fresnelF0(IOR_WATER, nSub);
  const stacked = r1 + ((1 - r1) ** 2 * r2) / (1 - r1 * r2);
  return dryF0 + (stacked - dryF0) * c;
}

/**
 * How the global wetness splits into absorbed water and surface film.
 *
 * @param wetness  global wetness, 0-1.
 * @param exposure how much of it this surface sees (interiors, undersides).
 * @param porosity how much of the water the substance absorbs.
 * @param puddle   cavity puddle mask, 0-1; raises film coverage locally.
 */
export function wetnessSplit(
  wetness: number,
  exposure: number,
  porosity: number,
  puddle: number,
): { soak: number; film: number } {
  const w = Math.min(Math.max(wetness * exposure, 0), 1);
  const p = Math.min(Math.max(porosity, 0), 1);
  const soak = w * p;
  // Film coverage from the non-absorbed fraction, raised wherever standing
  // water has pooled. `max` rather than a sum: a puddle in a porous surface is
  // still a puddle, and adding the two terms would let mud become a mirror.
  const film = Math.max(w * (1 - p), Math.min(Math.max(puddle, 0), 1) * w);
  return { soak, film };
}

/**
 * Albedo after absorption. Beer-Lambert path extension; see the module header.
 *
 * @param albedo linear reflectance of one channel, 0-1.
 * @param soak   absorbed-water fraction from {@link wetnessSplit}.
 */
export function wetAlbedoChannel(albedo: number, soak: number): number {
  const s = Math.min(Math.max(soak, 0), 1) * MAX_PATH_EXTENSION;
  return Math.pow(Math.min(Math.max(albedo, 0), 1), 1 + s);
}

/**
 * Roughness after wetting.
 *
 * Two effects compose. The film drives roughness towards
 * {@link WATER_ROUGHNESS} over the fraction of the surface it covers. Absorbed
 * water separately smooths the substrate a little, because it fills the finest
 * micro-relief — but only a little, which is what keeps soaked mud matte.
 */
export function wetRoughness(dryRoughness: number, soak: number, film: number): number {
  const damped = dryRoughness * (1 - 0.25 * Math.min(Math.max(soak, 0), 1));
  const f = Math.min(Math.max(film, 0), 1);
  return damped + (WATER_ROUGHNESS - damped) * f;
}

/**
 * Puddle mask from a cavity term.
 *
 * @param cavity   0 at the top of the mesostructure, 1 at the bottom.
 * @param level    the cavity depth the waterline sits at. Note the inversion:
 *                 a *rising* water level corresponds to a *falling* threshold,
 *                 so callers pass `1 - puddleLevel`.
 * @param softness width of the waterline. Small values give a crisp edge; too
 *                 crisp and the waterline aliases, hence a floor.
 *
 * The waterline is a *hard* physical boundary, so this is a smoothstep and not
 * a linear ramp: the derivative is zero on both sides of it, which is what
 * keeps the transition from shimmering as the camera moves.
 */
export function puddleMask(cavity: number, level: number, softness: number): number {
  const s = Math.max(softness, 0.02);
  const lo = level - s;
  const hi = level + s;
  if (hi <= lo) return cavity >= level ? 1 : 0;
  const t = Math.min(Math.max((cavity - lo) / (hi - lo), 0), 1);
  return t * t * (3 - 2 * t);
}

/* ------------------------------------------------------------------------- *
 * Shared uniforms
 * ------------------------------------------------------------------------- */

/**
 * The uniforms every material in the library shares.
 *
 * One set for the whole world, created once and referenced by every material.
 * Sharing matters: a per-material copy would mean a uniform upload per material
 * per frame and, worse, would make it possible for two materials to disagree
 * about how wet the world is.
 */
/** A scalar uniform node. Named so the shape does not have to be spelled out. */
export type FloatUniform = ReturnType<typeof floatUniform>;

function floatUniform(value: number) {
  return uniform(value, 'float');
}

export interface WetnessUniforms {
  /** Global wetness, 0-1. */
  readonly wetness: FloatUniform;
  /** How far standing water has risen into cavities, 0-1. */
  readonly puddleLevel: FloatUniform;
  /** Rain intensity, 0-1. Reserved for ripples; currently unused in shading. */
  readonly rain: FloatUniform;
}

/** Create the shared wetness uniform set. */
export function createWetnessUniforms(wetness = 0.65): WetnessUniforms {
  const w = THREE.MathUtils.clamp(wetness, 0, 1);
  return {
    wetness: floatUniform(w),
    puddleLevel: floatUniform(w * 0.6),
    rain: floatUniform(0),
  };
}

/* ------------------------------------------------------------------------- *
 * TSL
 * ------------------------------------------------------------------------- */

/** Puddle mask on the GPU. Mirrors {@link puddleMask}. */
export const puddleMaskNode = /*@__PURE__*/ Fn(
  ([cavity, level, softness]: [FloatNode, FloatNode, FloatNode]) => {
    const s = max(softness, 0.02).toVar('puddleSoft');
    return smoothstep(level.sub(s), level.add(s), cavity);
  },
).setLayout({
  name: 'puddleMask',
  type: 'float',
  inputs: [
    { name: 'cavity', type: 'float' },
    { name: 'level', type: 'float' },
    { name: 'softness', type: 'float' },
  ],
});

/** Albedo absorption on the GPU. Mirrors {@link wetAlbedoChannel}, per channel. */
export const wetAlbedoNode = /*@__PURE__*/ Fn(([albedo, soak]: [Vec3Node, FloatNode]) => {
  const exponent = float(1).add(saturate(soak).mul(MAX_PATH_EXTENSION)).toVar('wetExp');
  return pow(saturate(albedo), vec3(exponent));
}).setLayout({
  name: 'wetAlbedo',
  type: 'vec3',
  inputs: [
    { name: 'albedo', type: 'vec3' },
    { name: 'soak', type: 'float' },
  ],
});

/** Roughness response on the GPU. Mirrors {@link wetRoughness}. */
export const wetRoughnessNode = /*@__PURE__*/ Fn(
  ([roughness, soak, film]: [FloatNode, FloatNode, FloatNode]) => {
    const damped = roughness.mul(float(1).sub(saturate(soak).mul(0.25))).toVar('wetDamped');
    return mix(damped, float(WATER_ROUGHNESS), saturate(film));
  },
).setLayout({
  name: 'wetRoughness',
  type: 'float',
  inputs: [
    { name: 'roughness', type: 'float' },
    { name: 'soak', type: 'float' },
    { name: 'film', type: 'float' },
  ],
});

/**
 * Two-interface reflectance on the GPU. Mirrors {@link wetSpecularF0}.
 *
 * The substrate IOR is recovered from `dryF0` rather than passed in, so the
 * archetype table only ever has to author one reflectance number.
 */
export const wetSpecularF0Node = /*@__PURE__*/ Fn(
  ([dryF0, coverage]: [FloatNode, FloatNode]) => {
    const s = clamp(dryF0, 0, 0.99).sqrt().toVar('wf0Sqrt');
    const nSub = float(1).add(s).div(max(float(1).sub(s), 1e-4)).toVar('wf0Nsub');
    const r1 = float(F0_WATER);
    const r2 = pow(nSub.sub(IOR_WATER).div(nSub.add(IOR_WATER)), 2).toVar('wf0R2');
    const stacked = r1.add(
      float((1 - F0_WATER) ** 2).mul(r2).div(max(float(1).sub(r1.mul(r2)), 1e-4)),
    );
    return mix(dryF0, stacked, saturate(coverage));
  },
).setLayout({
  name: 'wetSpecularF0',
  type: 'float',
  inputs: [
    { name: 'dryF0', type: 'float' },
    { name: 'coverage', type: 'float' },
  ],
});

/** Everything the wetness stage needs to know about a dry surface. */
export interface WetnessInput {
  /** Dry linear albedo. */
  readonly albedo: Vec3Node;
  /** Dry perceptual roughness. */
  readonly roughness: FloatNode;
  /** Dry dielectric normal-incidence reflectance. */
  readonly reflectance: FloatNode;
  /** Shading normal, in whatever space the caller is working in. */
  readonly normal: Vec3Node;
  /**
   * The normal to relax towards where water pools. Tangent space: `(0,0,1)`.
   * Triplanar/world space: the geometric world normal.
   */
  readonly flatNormal: Vec3Node;
  /**
   * Cavity term: 0 on the peaks of the mesostructure, 1 in the pits. Normally
   * `1 - height`, falling back to `1 - ao` when no height field exists.
   */
  readonly cavity: FloatNode;
  /** Ambient occlusion, so pooled water can darken its own crevice further. */
  readonly ao: FloatNode;
  readonly porosity: FloatNode;
  readonly exposure: FloatNode;
  /** Peak clearcoat for the film, from the archetype. 0 disables it entirely. */
  readonly clearcoat: number;
}

/** The wet surface. Fields the wetness stage does not touch are passed through. */
export interface WetnessOutput {
  readonly albedo: Vec3Node;
  readonly roughness: FloatNode;
  readonly reflectance: FloatNode;
  readonly normal: Vec3Node;
  readonly ao: FloatNode;
  /** Clearcoat strength, or `null` when the archetype disables it. */
  readonly clearcoat: FloatNode | null;
  readonly clearcoatRoughness: FloatNode | null;
  /** Film coverage, exported so SSR-facing code can weight its confidence. */
  readonly film: FloatNode;
}

/**
 * Apply the wetness model to a dry surface.
 *
 * Ordering is deliberate. Absorption is applied to the albedo *before* the film
 * term touches the roughness, because a wet surface's diffuse colour is the
 * colour of the soaked substrate seen *through* the film, not the film's own.
 */
export function applyWetness(input: WetnessInput, uniforms: WetnessUniforms): WetnessOutput {
  const wet = saturate(uniforms.wetness.mul(input.exposure)).toVar('wetLevel');

  // Standing water gates the film term by cavity depth. The softness scales
  // with wetness so the waterline is crisp in a shallow puddle and diffuse as
  // the whole surface floods.
  const puddle = puddleMaskNode(
    input.cavity,
    uniforms.puddleLevel.oneMinus(),
    float(0.08).add(wet.mul(0.12)),
  ).mul(wet).toVar('wetPuddle');

  const soak = wet.mul(saturate(input.porosity)).toVar('wetSoak');
  const film = max(wet.mul(saturate(input.porosity).oneMinus()), puddle).toVar('wetFilm');

  const albedo = wetAlbedoNode(input.albedo, soak).toVar('wetAlbedoOut');
  const roughness = wetRoughnessNode(input.roughness, soak, film).toVar('wetRoughnessOut');
  const reflectance = wetSpecularF0Node(input.reflectance, film).toVar('wetF0Out');

  // Pooled water fills the mesoscale relief it sits in, so the shading normal
  // must relax towards flat exactly where the puddle mask says there is water.
  // Without this, puddles read as shiny gravel.
  const normal = flattenNormalNode(input.normal, puddle.mul(0.85)).toVar('wetNormalOut');

  // Water in a crevice absorbs on the way in and on the way out, so occlusion
  // deepens where it pools. Bounded so a fully flooded surface does not go
  // black.
  const ao = min(input.ao, float(1).sub(puddle.mul(0.35))).toVar('wetAoOut');

  const clearcoat =
    input.clearcoat > 0 ? saturate(film.mul(input.clearcoat)).toVar('wetClearcoat') : null;

  return {
    albedo,
    roughness,
    reflectance,
    normal,
    ao,
    clearcoat,
    clearcoatRoughness: clearcoat === null ? null : float(WATER_ROUGHNESS),
    film,
  };
}
