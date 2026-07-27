/**
 * @module render/RenderSettings
 *
 * The single authority for "how expensive is this frame" and "how wet is the
 * world", and the module that publishes both to everyone else.
 *
 * ## Why this exists
 *
 * Six subsystems were authored independently, and every one of them grew its
 * own quality enum: `QualityTier` in the post stack, `GTAOQuality`,
 * `SSRQuality`, `VolumetricsQuality`, `MaterialQuality`, plus a shadow
 * cascade/map-size pair. Left alone that is six knobs the player cannot see
 * and the integrator has to keep in agreement by hand. {@link RENDER_TIERS}
 * collapses them into one four-value ladder, so `?quality=low` is a single
 * decision that propagates coherently: fewer cascades *and* a cheaper AO
 * *and* no parallax, rather than a low-tier AO under ultra-tier shadows.
 *
 * The tier ladder is calibrated against the project's stated target — 60 fps
 * at 1080p on a mid-range 2020 discrete GPU (think RTX 2060 / RX 5600 XT):
 *
 * | | low | medium | high | ultra |
 * |---|---|---|---|---|
 * | post / AA | FXAA, 0.75 scale | FXAA | TAA 8 | TAA 16 + sharpen |
 * | shadow cascades × size | 2 × 1024 | 3 × 1536 | 4 × 2048 | 4 × 3072 |
 * | GTAO | off | low (½ res) | medium (½ res) | high (full res) |
 * | SSR | off | off | medium (½ res) | high (½ res) |
 * | volumetrics | low | medium | high | ultra |
 * | materials | low (no parallax) | medium | high | ultra |
 *
 * `low` deletes work rather than shrinking it — no AO pass, no SSR pass, no
 * velocity attachment — because on a weak GPU the win comes from the render
 * targets that stop being allocated at all, not from a smaller radius.
 *
 * ## Weather
 *
 * `src/render/materials/*` reads wetness through a `WeatherStateProvider`
 * under `'world.weather'`, and nothing in the project registered one, so every
 * material silently fell back to its authored default and the wet-mud look was
 * unreachable. This module owns that state too: it is the same kind of global,
 * it has the same single-authority requirement, and phase 3's weather system
 * can replace the instance without any material knowing.
 *
 * Blood Moor's default is `wetness 0.45` with puddles in the hollows — see
 * {@link BLOOD_MOOR_WEATHER} for why that number is much lower than
 * "rain-soaked" sounds.
 */

import type { GameContext, GameModule } from '../core/types';

import type { GTAOQuality } from './post/GTAO';
import type { QualityTier } from './post/PostStack';
import type { SSRQuality } from './post/SSR';
import type { VolumetricsQuality } from './Volumetrics';
import {
  RenderQualityKey,
  WeatherStateKey,
  type MaterialQuality,
  type RenderQualityProvider,
  type WeatherStateProvider,
} from './materials/types';

/** The one user-facing quality ladder. Every subsystem tier derives from it. */
export type RenderQuality = 'low' | 'medium' | 'high' | 'ultra';

export const RENDER_QUALITIES: readonly RenderQuality[] = ['low', 'medium', 'high', 'ultra'];

/** Every subsystem setting a tier decides, in one place. */
export interface RenderTier {
  readonly post: QualityTier;
  readonly materials: MaterialQuality;
  readonly gtao: GTAOQuality;
  readonly gtaoScale: number;
  readonly ssr: SSRQuality;
  readonly ssrScale: number;
  readonly volumetrics: VolumetricsQuality;
  readonly shadowCascades: number;
  readonly shadowMapSize: number;
  readonly shadowDistance: number;
  /** Sky-view LUT width. The environment probe derives from it. */
  readonly skyViewWidth: number;
  readonly environmentWidth: number;
  /** Froxel grid. `null` lets the volumetrics module pick from its own tier. */
  readonly froxelDimensions: readonly [number, number, number] | null;
  /** Light-shaft buffer scale relative to the chain resolution. */
  readonly lightShaftScale: number;
  /**
   * Upper bound on `devicePixelRatio` for this tier.
   *
   * ### Why this belongs to the tier and not to the Engine
   *
   * It used to be a flat 2 on `Engine`, and a flat cap is the most expensive
   * constant in the project. `devicePixelRatio` is *squared* into the pixel
   * count: on a Retina laptop a cap of 2 makes every fragment shader in the
   * frame run four times as often as it would on the same window at DPR 1. The
   * frame's dominant cost here is the terrain uber-shader — three triplanar
   * sample sets plus detail, macro variation and wetness, over most of the
   * screen — so that factor of four lands almost entirely on the thing that is
   * already the most expensive.
   *
   * Everything else the tier controls is chosen to fit a machine's budget, and
   * this is the largest single term in that budget, so it has to be chosen in
   * the same place and travel with the same knob. A tier that carefully halves
   * its shadow map and then renders at four times the pixels is not a quality
   * setting, it is two settings arguing.
   *
   * The ladder is deliberately not linear:
   * - `low` = 1: a machine that needed `low` cannot afford Retina fill.
   * - `medium` = 1.5: 2.25x the pixels of DPR 1, not 4x. Half a DPR step is
   *   most of the visible sharpness — text and high-contrast edges — for a
   *   little over half the cost, and 1.5 is the ratio at which a modern laptop
   *   GPU still has thermal headroom left over. On a fanless machine, headroom
   *   is the difference between a frame rate and a frame rate that decays.
   * - `high`/`ultra` = 2: explicit opt-in, so the player has said they want it.
   */
  readonly pixelRatioCap: number;
  /**
   * Scatter multiplier for prop/vegetation instancing in the showcase scene.
   *
   * **1 at every tier, deliberately.** This used to run 0.45 → 1.25 across the
   * ladder, which meant `ultra` drew nearly three times as many props, rocks
   * and trees as `low`: not a more expensive picture of the same world, a
   * different world. It showed up as a tone divergence the tier gate could not
   * be made to pass honestly — more small dark geometry is more genuinely dark
   * pixels, so `ultra` measured 2× the unique colours and a materially lower
   * mean luminance no matter what the renderer did — and it would have shipped
   * as two players standing on the same hill seeing a different number of
   * trees. A quality setting may change what a frame *costs*; it may not change
   * what is in it. The field is kept (rather than deleted) because a future
   * streaming/LOD system is the right owner of scatter counts, and it will want
   * a knob here that is driven by distance rather than by a menu.
   */
  readonly scatterDensity: number;
}

export const RENDER_TIERS: Readonly<Record<RenderQuality, RenderTier>> = {
  low: {
    post: 'low',
    materials: 'low',
    gtao: 'off',
    gtaoScale: 0.5,
    ssr: 'off',
    ssrScale: 0.5,
    volumetrics: 'low',
    shadowCascades: 2,
    shadowMapSize: 1024,
    shadowDistance: 90,
    skyViewWidth: 128,
    environmentWidth: 128,
    froxelDimensions: [96, 54, 48],
    lightShaftScale: 0.35,
    pixelRatioCap: 1,
    scatterDensity: 1,
  },
  medium: {
    post: 'medium',
    materials: 'medium',
    gtao: 'low',
    gtaoScale: 0.5,
    ssr: 'off',
    ssrScale: 0.5,
    volumetrics: 'medium',
    shadowCascades: 3,
    shadowMapSize: 1536,
    shadowDistance: 130,
    skyViewWidth: 192,
    environmentWidth: 192,
    froxelDimensions: [128, 72, 64],
    lightShaftScale: 0.4,
    pixelRatioCap: 1.5,
    scatterDensity: 1,
  },
  high: {
    post: 'high',
    materials: 'high',
    gtao: 'medium',
    gtaoScale: 0.5,
    ssr: 'medium',
    ssrScale: 0.5,
    volumetrics: 'high',
    shadowCascades: 4,
    shadowMapSize: 2048,
    shadowDistance: 180,
    skyViewWidth: 256,
    environmentWidth: 256,
    froxelDimensions: null,
    lightShaftScale: 0.5,
    pixelRatioCap: 2,
    scatterDensity: 1,
  },
  ultra: {
    post: 'ultra',
    materials: 'ultra',
    gtao: 'high',
    gtaoScale: 1,
    ssr: 'high',
    ssrScale: 0.5,
    volumetrics: 'ultra',
    shadowCascades: 4,
    shadowMapSize: 3072,
    shadowDistance: 240,
    skyViewWidth: 320,
    environmentWidth: 384,
    froxelDimensions: null,
    lightShaftScale: 0.5,
    pixelRatioCap: 2,
    scatterDensity: 1,
  },
};

/** Blood Moor's default weather: cold, rained-on, water still in the hollows. */
export interface WeatherState {
  wetness: number;
  puddleLevel: number;
  rainIntensity: number;
}

export const BLOOD_MOOR_WEATHER: Readonly<WeatherState> = {
  // 0.45, not the 0.8+ that "rain-soaked" suggests. Wetness in this material
  // model is a *specular* term: it darkens albedo a little and raises
  // reflectance a lot, and under a fully overcast sky — which is one enormous
  // area light — anything above about half turns every upward-facing surface
  // into a mirror of the sky and blows the ground out to white. The read the
  // brief asks for (dark, saturated, wet-looking mud) comes from the darkening,
  // and the darkening saturates well before the specular does.
  wetness: 0.38,
  // Near zero, and this one is not an art call so much as a rendering one.
  // Puddles are modelled as mirror-flat patches: under an overcast sky — a
  // single enormous bright source — each patch becomes a hard-edged specular
  // highlight, and across a 260 m ground plane at a grazing angle those
  // highlights alias into a field of white speckles that no amount of TAA
  // resolves. Standing water in this scene comes from the mud archetype's own
  // darkening in the hollows instead, which costs nothing and cannot sparkle.
  puddleLevel: 0.06,
  rainIntensity: 0,
};

/** What {@link detectQuality} looked at, so the choice can be logged and tested. */
export interface DeviceProfile {
  /** `navigator.deviceMemory` in GB, or `null` where the browser withholds it. */
  readonly deviceMemoryGb: number | null;
  /** `navigator.hardwareConcurrency`, or `null`. */
  readonly cores: number | null;
  /** Backing-store pixels the renderer would have to fill at native ratio. */
  readonly backingStorePixels: number;
  /** Coarse pointer / no hover: a phone or tablet. */
  readonly mobile: boolean;
}

/** Read a device profile from the current environment. Safe outside a browser. */
export function readDeviceProfile(): DeviceProfile {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return { deviceMemoryGb: null, cores: null, backingStorePixels: 0, mobile: false };
  }
  const nav = navigator as Navigator & { deviceMemory?: number };
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  return {
    deviceMemoryGb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
    cores: typeof nav.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    backingStorePixels: Math.round(
      (window.innerWidth || 1920) * (window.innerHeight || 1080) * ratio * ratio,
    ),
    mobile:
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(pointer: coarse)').matches,
  };
}

/**
 * Choose a starting tier from what the device is willing to admit about itself.
 *
 * ### Why this never returns `high` or `ultra`
 *
 * It used to. The default was a hardcoded `high` — 4x2048 shadow cascades, TAA,
 * GTAO and SSR — handed to every first-time visitor regardless of what they were
 * running. On an unknown machine that is a guess, and the cost of guessing
 * wrong is not a slow frame but a dead tab: the deployed build was killed by
 * Chromium's out-of-memory handler ("Error code: 5") on a mid-range laptop.
 *
 * So the ladder is deliberately asymmetric. Automatic detection picks between
 * `low` and `medium` only, and `high`/`ultra` are reachable exclusively by
 * explicit opt-in through `?quality=`. A player who finds the game running
 * conservatively can turn it up in one URL edit and see the result immediately.
 * A player whose tab is killed on load cannot do anything at all — they just
 * leave. Those two failure modes are not symmetric and the default should not
 * pretend they are.
 *
 * The signals are weak individually and that is fine, because they are only
 * being asked to separate "probably fine" from "probably not":
 * `deviceMemory` is clamped by browsers to 8 GB and rounded to a power of two,
 * `hardwareConcurrency` counts logical cores and says nothing about the GPU,
 * and neither is present in every browser. Any one of them looking bad is
 * enough to drop a tier, because the downside is a slightly softer picture.
 */
export function detectQuality(profile: DeviceProfile = readDeviceProfile()): RenderQuality {
  const reasons: string[] = [];

  // A phone or tablet, whatever else it reports. The thermal envelope decides
  // this long before the numbers do.
  if (profile.mobile) reasons.push('coarse pointer (mobile/tablet)');
  // 4 GB is the point below which the whole tab budget is a few hundred MB.
  if (profile.deviceMemoryGb !== null && profile.deviceMemoryGb <= 4) {
    reasons.push(`deviceMemory ${profile.deviceMemoryGb} GB`);
  }
  // Four logical cores usually means two physical ones and integrated graphics.
  if (profile.cores !== null && profile.cores <= 4) reasons.push(`${profile.cores} cores`);
  // Above 1440p-with-DPR the render targets alone start to dominate, and a
  // machine driving that many pixels is not necessarily equipped to shade them.
  if (profile.backingStorePixels > 5_000_000) {
    reasons.push(`${(profile.backingStorePixels / 1e6).toFixed(1)} Mpx backing store`);
  }

  const quality: RenderQuality = reasons.length > 0 ? 'low' : 'medium';
  if (typeof console !== 'undefined') {
    console.info(
      `[RenderSettings] auto-detected quality "${quality}"` +
        (reasons.length > 0 ? ` — ${reasons.join(', ')}` : ' — no constraints detected') +
        `. Override with ?quality=${RENDER_QUALITIES.join('|')}.`,
    );
  }
  return quality;
}

/**
 * Read `?quality=` from the current URL, falling back to {@link detectQuality}.
 *
 * Unknown values fall back *and say so*: a typo that silently changes every
 * capture is exactly the class of bug this whole module is meant to eliminate.
 *
 * @param fallback used when the URL says nothing. Defaults to detection rather
 *   than to a fixed tier — see {@link detectQuality}.
 */
export function qualityFromUrl(fallback?: RenderQuality): RenderQuality {
  const resolve = (): RenderQuality => fallback ?? detectQuality();
  if (typeof window === 'undefined') return fallback ?? 'medium';
  const raw = new URLSearchParams(window.location.search).get('quality');
  if (raw === null) return resolve();
  if ((RENDER_QUALITIES as readonly string[]).includes(raw)) return raw as RenderQuality;
  const chosen = resolve();
  console.warn(
    `[RenderSettings] unknown ?quality=${raw}; expected one of ` +
      `${RENDER_QUALITIES.join(' | ')}. Falling back to "${chosen}".`,
  );
  return chosen;
}

export interface RenderSettingsOptions {
  readonly quality?: RenderQuality;
  readonly weather?: Partial<WeatherState>;
}

/**
 * Publishes `'render.quality'` and `'world.weather'`.
 *
 * Registered first in the frame graph, before anything that reads them. Both
 * services are plain live views onto this module's fields, so a later change
 * through {@link RenderSettings.setWetness} is seen by every holder without
 * re-registration.
 */
export class RenderSettings implements GameModule, RenderQualityProvider, WeatherStateProvider {
  readonly name = 'render.settings';

  #quality: RenderQuality;
  readonly #weather: WeatherState;
  #ctx: GameContext | null = null;

  constructor(options: RenderSettingsOptions = {}) {
    this.#quality = options.quality ?? qualityFromUrl();
    this.#weather = { ...BLOOD_MOOR_WEATHER, ...options.weather };
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    ctx.services.register<RenderQualityProvider>(RenderQualityKey, this);
    ctx.services.register<WeatherStateProvider>(WeatherStateKey, this);
    // The engine boots with a default cap; this is the tier taking ownership of
    // it. Done in `init` rather than left to the first resize so the very first
    // frame is already the right size — a boot that renders one 4x frame and
    // then shrinks is a visible hitch on exactly the machines this protects.
    this.#applyPixelRatioCap();

    const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
    const css = this.#cssSize();
    const buffer = this.bufferSize(css.width, css.height, dpr);
    console.info(
      `[RenderSettings] quality="${this.#quality}" ` +
        `(post=${this.tier.post} gtao=${this.tier.gtao} ssr=${this.tier.ssr} ` +
        `volumetrics=${this.tier.volumetrics} shadows=${this.tier.shadowCascades}×` +
        `${this.tier.shadowMapSize}), wetness=${this.#weather.wetness.toFixed(2)}`,
    );
    console.info(
      `[RenderSettings] pixel ratio: device ${dpr.toFixed(2)}x, tier cap ` +
        `${this.tier.pixelRatioCap}x, using ${buffer.ratio.toFixed(2)}x — ` +
        `drawing buffer ${buffer.width}x${buffer.height} ` +
        `(${buffer.megapixels.toFixed(2)} Mpx before the post stack's render scale)`,
    );
  }

  dispose(): void {
    this.#ctx = null;
    // Services are unregistered by `ServiceLocator.clear()` on engine teardown.
  }

  #cssSize(): { width: number; height: number } {
    if (typeof window === 'undefined') return { width: 1920, height: 1080 };
    return { width: window.innerWidth || 1920, height: window.innerHeight || 1080 };
  }

  get quality(): RenderQuality {
    return this.#quality;
  }

  get tier(): RenderTier {
    return RENDER_TIERS[this.#quality];
  }

  /**
   * Change tier at runtime and push the consequences that cannot wait.
   *
   * Most of a tier is read through `this.tier` by whoever needs it, so it
   * follows automatically. The pixel-ratio cap is the exception: it lives on
   * the renderer's drawing buffer, which only changes when something asks it
   * to. Applying it here rather than leaving it to the next window resize is
   * the difference between `?quality=` being a setting and being a reload.
   */
  setQuality(quality: RenderQuality): void {
    if (this.#quality === quality) return;
    this.#quality = quality;
    this.#applyPixelRatioCap();
  }

  /**
   * The drawing buffer this tier would produce for a given CSS size and DPR.
   *
   * Exported as a function rather than left implicit because the number nobody
   * computes is the number that surprises everybody: a "0.75 render scale" tier
   * on a Retina display still fills more pixels than a full-scale tier on a
   * DPR-1 one, and stating it is the only way that comparison ever gets made.
   */
  bufferSize(cssWidth: number, cssHeight: number, devicePixelRatio: number): {
    ratio: number;
    width: number;
    height: number;
    megapixels: number;
  } {
    const ratio = Math.min(devicePixelRatio || 1, this.tier.pixelRatioCap);
    const width = Math.round(cssWidth * ratio);
    const height = Math.round(cssHeight * ratio);
    return { ratio, width, height, megapixels: (width * height) / 1e6 };
  }

  #applyPixelRatioCap(): void {
    const cap = this.tier.pixelRatioCap;
    this.#ctx?.engine.setPixelRatioCap(cap);
  }

  /* -- RenderQualityProvider --------------------------------------------- */

  get materialQuality(): MaterialQuality {
    return this.tier.materials;
  }

  /* -- WeatherStateProvider ---------------------------------------------- */

  get wetness(): number {
    return this.#weather.wetness;
  }

  get puddleLevel(): number {
    return this.#weather.puddleLevel;
  }

  get rainIntensity(): number {
    return this.#weather.rainIntensity;
  }

  setWetness(value: number): void {
    this.#weather.wetness = Math.min(1, Math.max(0, value));
  }

  setWeather(state: Partial<WeatherState>): void {
    Object.assign(this.#weather, state);
  }
}
