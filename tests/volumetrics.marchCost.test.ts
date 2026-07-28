/**
 * How much the volumetric ray march costs, and whether it runs at all.
 *
 * ## Why this file exists
 *
 * The deployed build reports 1.4 fps with 630 ms per frame unaccounted for by
 * any CPU timer, and the overlay line
 *
 * ```
 * fog    raymarch noise=on dispatch=0 scope=4
 * ```
 *
 * made the fog the leading suspect: a per-pixel loop sampling a 3D noise volume
 * three times per step, over 1.9 megapixels, is exactly the shape of a
 * pathological cost. It is a good hypothesis and it is wrong, and the way it is
 * wrong is instructive — that overlay line reports the mode the module *would*
 * select on a backend with no compute shader, not whether anything is being
 * rendered with it.
 *
 * `VolumetricsService.createResolveNode` has exactly one caller in the whole
 * project: `LightShaftsPass.#buildMarchFragment`. `FrameGraph.#installPasses`
 * only installs that pass when `RenderTier.lightShafts` is true, and that field
 * is `false` at `low` and `medium`. The machine under investigation is running
 * `medium`. So at `medium` the ray-march node is never built, never compiled and
 * never executed, and switching the fog off cannot recover a single millisecond.
 *
 * These tests pin both halves of that: the cost of the march where it does run,
 * and the wiring fact that decides where it runs.
 */

import { describe, expect, it } from 'vitest';

import { RENDER_TIERS } from '../src/render/RenderSettings';
import { VOLUMETRIC_TIERS, raymarchCost } from '../src/render/Volumetrics';

describe('VOLUMETRIC_TIERS march steps', () => {
  it('are the numbers the cost analysis was done against', () => {
    // Pinned so that a future retune has to come back through this file and
    // re-read the reasoning below rather than silently invalidating it.
    expect(VOLUMETRIC_TIERS.low.marchSteps).toBe(8);
    expect(VOLUMETRIC_TIERS.medium.marchSteps).toBe(14);
    expect(VOLUMETRIC_TIERS.high.marchSteps).toBe(32);
    expect(VOLUMETRIC_TIERS.ultra.marchSteps).toBe(48);
  });

  it('rise monotonically with the tier', () => {
    expect(VOLUMETRIC_TIERS.low.marchSteps).toBeLessThan(VOLUMETRIC_TIERS.medium.marchSteps);
    expect(VOLUMETRIC_TIERS.medium.marchSteps).toBeLessThan(VOLUMETRIC_TIERS.high.marchSteps);
    expect(VOLUMETRIC_TIERS.high.marchSteps).toBeLessThan(VOLUMETRIC_TIERS.ultra.marchSteps);
  });
});

describe('raymarchCost', () => {
  it('counts three noise fetches per step, which is what the shader does', () => {
    // `mediaAt`: a warp lookup, the warped coarse band, and a detail band at
    // 3.1x the frequency. All three are trilinear fetches into the 32^3
    // `Data3DTexture`, and all three are per step, not per pixel.
    const cost = raymarchCost('medium', { detailNoise: true, shadowCascades: 0 });
    expect(cost.noiseFetchesPerStep).toBe(3);
    expect(cost.fetches).toBe(14 * 3);
  });

  it('adds one shadow fetch per cascade per step', () => {
    // `#buildSunVisibility` unrolls the cascade selection — `cascadeCount` is a
    // JS constant — so every step pays every cascade's `mat4` transform and
    // `textureLoad`, not just the one it lands in.
    const cost = raymarchCost('high', { detailNoise: true, shadowCascades: 4 });
    expect(cost.shadowFetchesPerStep).toBe(4);
    expect(cost.fetches).toBe(32 * (3 + 4));
    expect(cost.fetches).toBe(224);
  });

  it('drops the noise fetches entirely when detail noise is off', () => {
    // `?fog=off` forces `detailNoise: 'off'`, and the WebGPU-compat path
    // disables it too. Two thirds of the march's texture traffic at zero
    // cascades, and it must not survive as a multiply by zero.
    const cost = raymarchCost('high', { detailNoise: false, shadowCascades: 4 });
    expect(cost.noiseFetchesPerStep).toBe(0);
    expect(cost.fetches).toBe(32 * 4);
  });

  it('counts the local-light loop, which is the other per-step multiplier', () => {
    expect(raymarchCost('ultra', {}).lightIterations).toBe(48 * 6);
    expect(raymarchCost('medium', {}).lightIterations).toBe(14 * 3);
  });

  it('puts the worst case within reach of a modern GPU, which is the point', () => {
    // `ultra`: 48 steps x (3 noise + 4 cascades) = 336 fetches per marched
    // pixel. At the `ultra` tier's 0.5 shaft scale that is a quarter of the
    // frame's pixels, so at 1.9 Mpx it is ~0.475 Mpx x 336 = 160M fetches.
    // Large, and nowhere near able to explain 630 ms on an M4 — which is the
    // number this whole exercise is trying to account for.
    const cost = raymarchCost('ultra', { detailNoise: true, shadowCascades: 4 });
    expect(cost.fetches).toBe(336);
    const marchedPixels = 1.9e6 * 0.5 * 0.5;
    expect(Math.round((marchedPixels * cost.fetches) / 1e6)).toBe(160);
  });
});

describe('where the ray march actually runs', () => {
  it('has no consumer pass below the high tier', () => {
    // The finding. `LightShaftsPass` is the sole caller of
    // `createResolveNode`, and `FrameGraph.#installPasses` gates it on this
    // flag — so at `low` and `medium` the fog resolve node is never built and
    // the march never executes, whatever `VolumetricsStats.mode` says.
    expect(RENDER_TIERS.low.lightShafts).toBe(false);
    expect(RENDER_TIERS.medium.lightShafts).toBe(false);
    expect(RENDER_TIERS.high.lightShafts).toBe(true);
    expect(RENDER_TIERS.ultra.lightShafts).toBe(true);
  });

  it('is marched at reduced resolution wherever it does run', () => {
    // Half per axis, i.e. a quarter of the pixels. The march reads a half-res
    // guide buffer and writes a half-res scatter target; the composite
    // upsamples. This is the single largest reason the cost above is bearable.
    expect(RENDER_TIERS.high.lightShaftScale).toBeLessThanOrEqual(0.5);
    expect(RENDER_TIERS.ultra.lightShaftScale).toBeLessThanOrEqual(0.5);
  });

  it('leaves the medium tier — the auto-detected one — with no fog cost at all', () => {
    // Automatic detection only ever chooses `low` or `medium` (see
    // `detectQuality`), so *every* machine that has not typed a `?quality=`
    // override is in the no-fog-pass configuration. That is the whole reason
    // `?fog=off` is expected to change nothing on the reporter's machine, and
    // why a bisection must not stop when it doesn't.
    for (const tier of ['low', 'medium'] as const) {
      expect(RENDER_TIERS[tier].lightShafts).toBe(false);
    }
  });
});
