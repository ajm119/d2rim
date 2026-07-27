/**
 * @module render/materials/MacroVariation
 *
 * Low-frequency, world-space modulation of albedo and roughness at 20-50 metre
 * scale. This is the cheapest and most effective single defence against the
 * "quilt" — the instantly readable grid you get when one 2 m texture tile
 * covers a 200 m field.
 *
 * ### Why it works
 *
 * Tiling repetition is detected by the visual system as a *periodic* signal.
 * The eye is extremely good at spotting periodicity and almost blind to
 * low-frequency drift. Multiplying the tiled texture by an aperiodic field an
 * order of magnitude larger than the tile does not remove the repetition — the
 * high-frequency content is still identical from tile to tile — but it destroys
 * the periodicity of the *envelope*, which is what the eye was locking onto. It
 * is the same principle as dithering a banded gradient.
 *
 * Structural repetition (the same distinctive rock appearing on a lattice) needs
 * the stronger medicine in {@link module:render/materials/AntiTile}. The two
 * compose: macro variation is always on, hex-tiling is added on top for hero
 * surfaces.
 *
 * ### Energy preservation
 *
 * The albedo gain is centred on 1 and symmetric, so the *mean* albedo of a large
 * surface is unchanged. That matters because the archetype table authors an
 * albedo tint against a reference look, and a variation term that quietly
 * darkened everything by 8% would make every one of those numbers a lie.
 * `tests/materials.macro.test.ts` pins the mean.
 *
 * ### Colour, not just brightness
 *
 * The bright lobe of the noise also drifts the albedo towards an authored tint —
 * moss green on rock, rust on iron, a warmer ochre on mud. Only the *positive*
 * lobe tints, so the result reads as patches of something growing on the
 * surface rather than as a global hue rotation. This is where the art
 * direction's "desaturated but not grey mush" requirement is actually satisfied
 * on large surfaces: without it, a 200 m field of one desaturated brown is
 * exactly the grey mush the brief forbids.
 *
 * ### Cost
 *
 * Two evaluations of MaterialX's gradient-noise fBm, three octaves each, and no
 * texture fetches at all. Roughly 60 ALU. That is under a tenth the cost of one
 * extra texture tap, which is why this is on for every archetype including the
 * ones that also run hex-tiling.
 */

import { Fn, float, mix, mx_fractal_noise_float, saturate, vec3 } from 'three/tsl';

import type { FloatNode, Vec3Node } from './types';

/* ------------------------------------------------------------------------- *
 * CPU reference math
 * ------------------------------------------------------------------------- */

/**
 * Albedo brightness gain from a signed noise sample.
 *
 * @param signedNoise noise in `[-1, 1]`.
 * @param amount      peak-to-peak modulation as a fraction. 0.25 means the
 *                    brightest patch is 12.5% up and the darkest 12.5% down.
 * @returns a multiplier centred on 1.
 *
 * Clamped at the bottom so an extreme noise value with a large amount cannot
 * produce a negative gain, which would flip the surface to black.
 */
export function macroAlbedoGain(signedNoise: number, amount: number): number {
  return Math.max(1 + signedNoise * amount * 0.5, 0.05);
}

/**
 * Roughness offset from a signed noise sample.
 *
 * Absolute, not multiplicative: perceptual roughness is already a perceptually
 * uniform parameterisation, so a constant offset is a constant apparent change
 * whether the surface started at 0.3 or 0.8.
 */
export function macroRoughnessDelta(signedNoise: number, amount: number): number {
  return signedNoise * amount * 0.5;
}

/**
 * Weight of the authored tint, from a signed noise sample.
 *
 * Only the positive lobe tints — see the module header. Squared so the tint
 * concentrates into distinct patches instead of hazing the whole surface.
 */
export function macroTintWeight(signedNoise: number, amount: number): number {
  const positive = Math.min(Math.max(signedNoise, 0), 1);
  return positive * positive * Math.min(Math.max(amount, 0), 1);
}

/**
 * The mean of {@link macroAlbedoGain} over a symmetric noise distribution.
 *
 * Exists so the test can state the energy-preservation property directly rather
 * than rediscovering it, and so a future change to the gain curve has to update
 * one obvious place.
 */
export function macroAlbedoGainMean(): number {
  return 1;
}

/* ------------------------------------------------------------------------- *
 * TSL
 * ------------------------------------------------------------------------- */

/** Everything the macro stage produces. */
export interface MacroVariation {
  /** Multiplier on albedo, centred on 1. */
  readonly albedoGain: FloatNode;
  /** Additive roughness offset. */
  readonly roughnessDelta: FloatNode;
  /** Weight of the archetype's tint colour, 0-1. */
  readonly tintWeight: FloatNode;
  /** The raw signed noise, for callers that want to drive something else. */
  readonly noise: FloatNode;
}

export interface MacroOptions {
  /** World position in metres. */
  readonly positionWorld: Vec3Node;
  /** Wavelength of the primary band, in metres. */
  readonly metres: number;
  readonly albedoAmount: number;
  readonly roughnessAmount: number;
  readonly tintAmount: number;
}

/**
 * Evaluate the macro variation field.
 *
 * Two bands rather than one wide fBm. A single fBm spanning 40 m down to 2 m
 * spends most of its octaves in the range where the base texture already has
 * content, so it contributes nothing but noise; two well-separated bands put
 * energy exactly where the tiling artefact lives — at the tile period and at a
 * few times the tile period — and nowhere else.
 *
 * The second band is offset in world space by an irrational-ish vector so the
 * two bands cannot phase-align into a visible beat pattern.
 */
export function macroVariation(options: MacroOptions): MacroVariation {
  const { positionWorld, metres, albedoAmount, roughnessAmount, tintAmount } = options;
  const scale = 1 / Math.max(metres, 0.5);

  // `mx_fractal_noise_float` returns roughly [-1, 1] for the default amplitude.
  const broad = mx_fractal_noise_float(positionWorld.mul(scale), 3, 2.0, 0.55) as FloatNode;
  const mid = mx_fractal_noise_float(
    positionWorld.mul(scale * 3.7).add(vec3(17.31, 5.77, 29.13)),
    2,
    2.0,
    0.5,
  ) as FloatNode;

  // 70/30 towards the broad band: the broad band is what kills the quilt, the
  // mid band only stops the broad band itself from reading as smooth blobs.
  const noise = broad.mul(0.7).add(mid.mul(0.3)).clamp(-1, 1).toVar('macroNoise');

  return {
    noise,
    albedoGain: float(1).add(noise.mul(albedoAmount * 0.5)).max(0.05).toVar('macroAlbedoGain'),
    roughnessDelta: noise.mul(roughnessAmount * 0.5).toVar('macroRoughnessDelta'),
    tintWeight: saturate(noise).pow(2).mul(tintAmount).toVar('macroTintWeight'),
  };
}

/**
 * Apply the macro albedo terms.
 *
 * Gain first, then tint. The other order would let the tint be scaled by the
 * gain and the patches would fade out exactly where the surface is darkest,
 * which is the opposite of how lichen and rust actually distribute.
 */
export const applyMacroAlbedoNode = /*@__PURE__*/ Fn(
  ([albedo, gain, tint, tintWeight]: [Vec3Node, FloatNode, Vec3Node, FloatNode]) => {
    return mix(albedo.mul(gain), tint, saturate(tintWeight));
  },
).setLayout({
  name: 'applyMacroAlbedo',
  type: 'vec3',
  inputs: [
    { name: 'albedo', type: 'vec3' },
    { name: 'gain', type: 'float' },
    { name: 'tint', type: 'vec3' },
    { name: 'tintWeight', type: 'float' },
  ],
});
