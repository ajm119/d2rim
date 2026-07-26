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
  /** Scatter multiplier for prop/vegetation instancing in the showcase scene. */
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
    scatterDensity: 0.45,
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
    scatterDensity: 0.7,
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
    scatterDensity: 1.25,
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

/**
 * Read `?quality=` from the current URL.
 *
 * Unknown values fall back to `high` *and say so*: a typo that silently
 * downgrades every capture is exactly the class of bug this whole module is
 * meant to eliminate.
 */
export function qualityFromUrl(fallback: RenderQuality = 'high'): RenderQuality {
  if (typeof window === 'undefined') return fallback;
  const raw = new URLSearchParams(window.location.search).get('quality');
  if (raw === null) return fallback;
  if ((RENDER_QUALITIES as readonly string[]).includes(raw)) return raw as RenderQuality;
  console.warn(
    `[RenderSettings] unknown ?quality=${raw}; expected one of ` +
      `${RENDER_QUALITIES.join(' | ')}. Falling back to "${fallback}".`,
  );
  return fallback;
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

  constructor(options: RenderSettingsOptions = {}) {
    this.#quality = options.quality ?? qualityFromUrl();
    this.#weather = { ...BLOOD_MOOR_WEATHER, ...options.weather };
  }

  init(ctx: GameContext): void {
    ctx.services.register<RenderQualityProvider>(RenderQualityKey, this);
    ctx.services.register<WeatherStateProvider>(WeatherStateKey, this);
    console.info(
      `[RenderSettings] quality="${this.#quality}" ` +
        `(post=${this.tier.post} gtao=${this.tier.gtao} ssr=${this.tier.ssr} ` +
        `volumetrics=${this.tier.volumetrics} shadows=${this.tier.shadowCascades}×` +
        `${this.tier.shadowMapSize}), wetness=${this.#weather.wetness.toFixed(2)}`,
    );
  }

  dispose(): void {
    // Services are unregistered by `ServiceLocator.clear()` on engine teardown.
  }

  get quality(): RenderQuality {
    return this.#quality;
  }

  get tier(): RenderTier {
    return RENDER_TIERS[this.#quality];
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
