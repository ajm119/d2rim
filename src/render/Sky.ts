/**
 * @module render/Sky
 *
 * The visible sky, the cloud deck, and the image-based lighting they produce.
 *
 * This module owns nothing about *how light scatters* — that lives in
 * {@link module:render/Atmosphere} — and nothing about *where the sun is* —
 * that lives in {@link module:render/TimeOfDay}. What it owns is the bridge:
 * turning the scattering model into pixels, into an environment map, and into
 * numbers a lighting rig can drive a directional light with, such that all
 * three can never disagree.
 *
 * ---
 *
 * ## What is drawn
 *
 * `scene.backgroundNode` is a TSL graph evaluated once per screen pixel:
 *
 * ```
 * L = mix( skyView + sunDisc + moonDisc + stars,   aerialPerspective(cloudDeck),   cloudAlpha )
 * ```
 *
 * - **skyView** — one bilinear fetch from a small equirectangular radiance
 *   texture whose contents are `Atmosphere.skyRadiance` evaluated on the CPU.
 *   Vertical parameterisation is `v = 0.5 - 0.5*sign(theta)*sqrt(|theta|/(pi/2))`,
 *   which is Hillaire's horizon-concentrating warp: the sky is a very smooth
 *   function *except* within a few degrees of the horizon, so a uniform
 *   parameterisation wastes half its rows on the featureless upper hemisphere
 *   and still bands at the horizon. With the warp, 160x80 is indistinguishable
 *   from a per-pixel ray march.
 * - **sunDisc** — analytic, at full screen resolution, with `fwidth`-based
 *   antialiasing and Hestroffer & Magnan limb darkening (see below). Baking the
 *   disc into the texture instead would smear a 0.53-degree feature across a
 *   2.25-degree texel.
 * - **cloudDeck** — layered 2.5D, described below.
 * - **aerialPerspective** — the *same* function distant geometry uses, so the
 *   cloud deck at the horizon converges exactly onto the haze in front of it.
 *
 * ## Clouds: why layered 2.5D and not a raymarch
 *
 * A production volumetric cloud raymarch (Schneider & Vos, *"The Real-Time
 * Volumetric Cloudscapes of Horizon Zero Dawn"*, SIGGRAPH 2015) costs 64-128
 * primary steps with 4-6 secondary light steps each, against two 3D textures.
 * Call it 250 texture fetches per sky pixel, plus reprojection machinery to
 * make a half-resolution buffer temporally stable. At 1080p with the sky
 * covering even 40% of the frame that is ~200 M fetches per frame; a mid-range
 * 2020 discrete GPU retires on the order of 90-170 Gtexel/s, so that is 1.5-2 ms
 * *at half resolution with TAA*, and it brings a temporal-stability problem with
 * it that interacts badly with every other pass.
 *
 * The layered model here samples **four** density slabs whose positions come
 * from real ray/sphere intersections at four altitudes, plus **two** shadow
 * probes offset along the sun direction. Six fetches. It gets genuine parallax
 * (clouds compress toward the horizon and spread overhead, because the geometry
 * is actually spherical), genuine self-shadowing, and genuine silhouette detail
 * from the erosion channels. What it cannot do is let the camera fly *into* a
 * cloud — which this game never does.
 *
 * Two details make the difference between that model working and it looking
 * like corrugated iron near the horizon, and both are commented at their call
 * sites: the inter-layer shear is clamped (a single 2D field has no vertical
 * correlation, so far-apart layers sample unrelated weather), and the shaped
 * density converges to its analytic mean once the screen-space filter footprint
 * exceeds the cloud feature size (mip filtering prefilters the *sample*, but the
 * coverage threshold re-sharpens the average straight back into blocks).
 *
 * Cloud radiative transfer uses the two-stream solution for a scattering slab
 * (see {@link CLOUD_TWO_STREAM}), evaluated per RGB channel against the
 * single-scattering albedo of liquid water. That is what makes an overcast sky
 * here get *brighter as the deck thins*, *darker at its base as it thickens*,
 * and — because water absorbs red about forty times more strongly than blue —
 * *colder the thicker it gets*, instead of being a flat grey the artist has to
 * keep re-tuning.
 *
 * ## Sun disc
 *
 * Limb darkening from Hestroffer & Magnan, *"Wavelength dependency of the
 * Solar limb darkening"*, A&A 333, 1998, in its single-exponent form
 * `I(mu)/I(1) = mu^alpha` with `alpha = (0.397, 0.503, 0.652)` for R, G, B —
 * blue falls off fastest, which is why the limb of the sun is measurably redder
 * than its centre.
 *
 * ## Image-based lighting
 *
 * A second, smaller equirectangular texture is baked with the cloud deck and an
 * **energy-conserving** sun disc (radiance scaled by the ratio of the sun's
 * solid angle to the texel's, so the total flux is right even though the disc is
 * far smaller than a texel) and assigned to `scene.environment`. Because it is
 * generated from the same `Atmosphere` instance as the background, the IBL and
 * the visible sky are the same function sampled twice. They cannot drift.
 *
 * Hemispherical irradiance is integrated from that same bake and published in
 * {@link CelestialLightState}, so an ambient/fill term is also guaranteed
 * consistent rather than hand-dialled.
 *
 * ## Update budget
 *
 * The expensive part is the CPU ray march that fills the sky-view grid: 12 800
 * marches of 24 steps for the default 160x80, doubled at night when the moon is
 * up and gets a second pass. Measured at ~53 ms on one core of this container's
 * CPU. It runs **synchronously once at `init`** (so a zero-warmup capture is
 * still correct) and thereafter only when the sun has moved past
 * {@link SkyOptions.sunMoveThresholdDeg} or the weather changed — and then it is
 * sliced across frames against a wall-clock budget
 * ({@link SkyOptions.rebuildBudgetMs}, default 1.2 ms) into a *back buffer*, so
 * a half-finished rebuild is never visible. PMREM re-filtering of the
 * environment map happens once per completed rebuild, not per frame.
 *
 * Everything that can track the clock at full frame rate without a rebuild
 * does: the sun and moon discs, the cloud drift and the whole aerial-perspective
 * uniform block are plain uniform writes.
 *
 * ---
 *
 * ## Interfaces this module expects from other systems
 *
 * Everything below is resolved lazily through the {@link ServiceLocator} and is
 * **optional**; the sky is fully functional with none of them present.
 *
 * - `'lighting.celestial'` -> {@link CelestialLightSink}. If a lighting module
 *   registers one, this module pushes {@link CelestialLightState} into it
 *   whenever the sky changes. Otherwise the same payload is emitted on the
 *   event bus as `'sky:celestial'`, and can also be pulled at any time via
 *   {@link Sky.celestialLight}. The integrator only needs to wire *one* of the
 *   three.
 * - `'render.timeOfDay'` -> {@link TimeOfDay}. If the integrator has already
 *   registered a clock, this module drives that one. If not, it creates and
 *   registers its own, so `engine.add(new Sky())` alone is a complete,
 *   working system.
 *
 * ## Services this module registers
 *
 * - `'render.atmosphere'` -> {@link AtmosphereService} — the fog/aerial
 *   perspective model. **Any pass that needs fog must use this one.**
 * - `'render.sky'` -> {@link Sky}. Also satisfies the `SkyEnvironmentProvider`
 *   shape `render/IBL.ts` looks for under the same id — `environmentTexture`
 *   plus `environmentVersion` — so no adapter is needed there.
 * - `'render.sky.sun'` -> {@link SkySunProvider}, the poll-side view of the sun
 *   that `render/Lighting.ts` declares. Only registered if absent.
 * - `'render.timeOfDay'` -> {@link TimeOfDay} (only if absent)
 *
 * ## What the integrator has to do
 *
 * ```ts
 * engine.add(new Sky());          // after AssetManager, before scene modules
 * ```
 *
 * That is the whole wiring. `scene.backgroundNode`, `scene.environment` and
 * `scene.fogNode` are installed by `init`, and the lighting rig is fed through
 * whichever of the three routes above it registered for. Register `Sky` *after*
 * any module that also assigns `scene.backgroundNode` (`scene/ReferenceScene`
 * does) so this one wins.
 */

import * as THREE from 'three/webgpu';
import {
  abs,
  cameraPosition,
  acos,
  asin,
  atan,
  clamp,
  dot,
  exp,
  float,
  floor,
  fract,
  fwidth,
  hash,
  length,
  max,
  min,
  mix,
  normalize,
  oneMinus,
  output,
  positionWorld,
  positionWorldDirection,
  pow,
  remapClamp,
  saturate,
  sign,
  smoothstep,
  sqrt,
  step,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import { SimplexNoise, WorleyNoise } from '../assets/Procedural';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import {
  Atmosphere,
  AtmosphereKey,
  cornetteShanksPhase,
  type AtmosphereOptions,
  type AtmosphereService,
} from './Atmosphere';
import {
  MOON_ANGULAR_RADIUS_DEG,
  SUN_ANGULAR_RADIUS_DEG,
  TimeOfDay,
  TimeOfDayKey,
  type TimeOfDayPresetName,
} from './TimeOfDay';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** Sun/moon/ambient state changed. The payload object is reused; copy it. */
    'sky:celestial': CelestialLightState;
    /** A sky-view rebuild finished and the environment map was re-filtered. */
    'sky:rebuilt': { hours: number; elapsedMs: number; sunElevationDeg: number };
  }
}

/* -------------------------------------------------------------------------- */
/* Contract with the lighting system                                           */
/* -------------------------------------------------------------------------- */

/** One celestial light, in a form a `THREE.DirectionalLight` takes directly. */
export interface CelestialLightSample {
  /** World-space unit vector from the scene *toward* the body. */
  readonly direction: THREE.Vector3;
  /** Hue of the light, normalised so the largest channel is 1. */
  readonly color: THREE.Color;
  /**
   * Illuminance on a surface facing the body, in the engine's linear render
   * units. Already includes atmospheric extinction and cloud attenuation, and
   * already zero when the body is below the horizon.
   */
  readonly illuminance: number;
  /** `0` fully set, `1` fully risen. */
  readonly visibility: number;
}

/**
 * Everything the lighting rig needs from the sky, in one object.
 *
 * The irradiance terms are the cosine-weighted integrals of the *actual*
 * environment map that `scene.environment` is filtered from, so a fill light
 * driven from them agrees with the IBL by construction.
 */
export interface CelestialLightState {
  readonly sun: CelestialLightSample;
  readonly moon: CelestialLightSample;
  /** Irradiance on an upward-facing surface from the sky hemisphere. */
  readonly skyIrradiance: THREE.Color;
  /** Irradiance on a downward-facing surface (ground bounce). */
  readonly groundIrradiance: THREE.Color;
  /** `0` full daylight, `1` full night. */
  readonly nightFactor: number;
}

/**
 * Implemented by the lighting module. Register under
 * {@link CelestialLightSinkKey} and the sky will push state into it.
 */
export interface CelestialLightSink {
  setCelestialLight(state: CelestialLightState): void;
}

export const CelestialLightSinkKey = serviceKey<CelestialLightSink>('lighting.celestial');

/**
 * Service key for this module.
 *
 * Deliberately the same id `render/IBL.ts` looks for its `SkyEnvironmentProvider`
 * under: the {@link Sky} class already satisfies that interface structurally
 * ({@link Sky.environmentTexture} plus {@link Sky.environmentVersion}), so the
 * IBL module needs no adapter and no import.
 */
export const SkyKey = serviceKey<Sky>('render.sky');

/**
 * Poll-side view of the sun, for consumers that would rather read than be
 * pushed to.
 *
 * `render/Lighting.ts` declares this shape as its `SkySunProvider` and looks
 * for it under `'render.sky.sun'`; a small adapter is registered there at init
 * so that a lighting rig which prefers polling works without any wiring. The
 * push route ({@link CelestialLightSink}) carries strictly more information —
 * the moon, the ambient irradiances, the night factor — and should be preferred.
 */
export interface SkySunProvider {
  /** Unit vector from the origin *toward* the sun; light travels along `-this`. */
  readonly sunDirection: THREE.Vector3;
  /** Linear-space hue of direct sunlight, peak-normalised. */
  readonly sunColor: THREE.Color;
  /** Relative intensity in `[0, 1]`; 0 at or below the horizon. */
  readonly sunIntensity: number;
}

export const SkySunKey = serviceKey<SkySunProvider>('render.sky.sun');

/* -------------------------------------------------------------------------- */
/* Options                                                                     */
/* -------------------------------------------------------------------------- */

export interface SkyOptions {
  /** Use this clock instead of resolving/creating one. */
  timeOfDay?: TimeOfDay;
  /** Forwarded to the {@link Atmosphere} constructor. */
  atmosphere?: AtmosphereOptions;
  /** Preset applied at `init` when no clock was supplied. Default `'bloodMoor'`. */
  preset?: TimeOfDayPresetName;

  /** Sky-view texture width; height is half. Default 160. */
  skyViewWidth?: number;
  /** Environment (IBL) texture width; height is half. Default 128. */
  environmentWidth?: number;
  /** Cloud density texture edge length. Default 512. */
  cloudTextureSize?: number;
  /** Deterministic seed for the cloud noise. Default 20241. */
  seed?: number;

  /** Wall-clock budget per frame for incremental sky-view rebuilds. Default 1.2 ms. */
  rebuildBudgetMs?: number;
  /** Sun movement, in degrees, that triggers a rebuild. Default 0.35. */
  sunMoveThresholdDeg?: number;

  /** Install `scene.backgroundNode`. Default true. */
  installBackground?: boolean;
  /** Install `scene.environment`. Default true. */
  installEnvironment?: boolean;
  /**
   * Install `scene.fogNode`, giving every material aerial perspective from this
   * model with no per-material work. Default true.
   */
  installFog?: boolean;
  /** Draw a procedural star field at night. Default true. */
  stars?: boolean;
  /** Cloud tiling scale: kilometres per texture repeat. Default 9. */
  cloudTileKm?: number;
  /** Cloud drift in km/s, `[east, north]`. Default `[0.010, 0.004]`. */
  cloudWind?: readonly [number, number];
}

/* -------------------------------------------------------------------------- */
/* Constants                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Density slabs through the deck. Four is the knee of the quality curve: three
 * shows visible layer banding on the underside of thick cloud, five costs a
 * fetch for a difference nobody sees.
 */
const CLOUD_LAYERS = 4;

/**
 * Asymmetry of cloud droplet scattering. Water droplets are strongly forward
 * scattering; 0.85 is the usual fit for the Mie solution of a 10 um droplet
 * distribution and is what produces the silver lining.
 */
const CLOUD_PHASE_G = 0.78;

/**
 * Maximum separation, in texture repeats, between the deck's lowest and highest
 * density slab. See the clamp in `Sky.#buildSkyNode`.
 */
const CLOUD_MAX_SHEAR = 0.15;

/** Asymmetry parameter used by the two-stream slab solution. */
const CLOUD_SLAB_G = 0.85;

/**
 * Single-scattering albedo of cloud droplets, per RGB channel.
 *
 * Liquid water is very nearly transparent in the visible, but *not equally* so:
 * its absorption coefficient rises by almost two orders of magnitude from blue
 * to red (Hale & Querry, *"Optical constants of water in the 200 nm to 200 um
 * wavelength region"*, Applied Optics 12(3), 1973 — roughly 0.009, 0.057 and
 * 0.34 per metre at 450, 550 and 650 nm). For a stratus deck at 0.3 g/m^3 those
 * give the single-scattering albedos below.
 *
 * The difference looks negligible per scattering event and is not: a thick deck
 * scatters light tens of times before it emerges, and `omega_0^N` compounds it
 * into a real, visible shift. This is why the underside of heavy overcast reads
 * *cold* blue-grey rather than warm grey even when the light arriving on top of
 * it is a low, warm sun — and it is the single physical effect that makes the
 * Blood Moor mood come out of the model instead of out of a colour grade.
 */
const CLOUD_SINGLE_SCATTER_ALBEDO: readonly [number, number, number] = [
  0.99887, 0.99981, 0.99997,
];

/**
 * Two-stream slab coefficients derived from
 * {@link CLOUD_SINGLE_SCATTER_ALBEDO}, per channel.
 *
 * Standard two-stream solution for a homogeneous scattering slab (Meador &
 * Weaver, *"Two-stream approximations to radiative transfer in planetary
 * atmospheres"*, J. Atmos. Sci. 37, 1980; the accessible derivation is in
 * Bohren, *"Multiple scattering of light and some of its observable
 * consequences"*, Am. J. Phys. 55(6), 1987):
 *
 * ```
 * gamma = sqrt(3 (1 - w0)(1 - w0 g))       a = sqrt((1 - w0) / (1 - w0 g))
 * T_diffuse(tau) = 4a / ( (1+a)^2 e^{gamma tau} - (1-a)^2 e^{-gamma tau} )
 * ```
 *
 * As `w0 -> 1` this reduces to the conservative slab result
 * `1 / (1 + (sqrt(3)/2)(1-g) tau)` (expand both exponentials to first order in
 * `a`), so nothing is lost relative to the simpler non-absorbing model — the
 * droplet absorption is a strictly additive correction on top of it.
 */
const CLOUD_TWO_STREAM = (() => {
  const g = CLOUD_SLAB_G;
  const gamma: [number, number, number] = [0, 0, 0];
  const asym: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const w0 = CLOUD_SINGLE_SCATTER_ALBEDO[c] as number;
    gamma[c] = Math.sqrt(3 * (1 - w0) * (1 - w0 * g));
    asym[c] = Math.sqrt((1 - w0) / (1 - w0 * g));
  }
  return { gamma, asym };
})();

/**
 * Diffuse transmittance of the cloud slab for one channel. See
 * {@link CLOUD_TWO_STREAM}.
 */
export function cloudSlabTransmittance(tau: number, channel: 0 | 1 | 2): number {
  const gamma = CLOUD_TWO_STREAM.gamma[channel];
  const a = CLOUD_TWO_STREAM.asym[channel];
  const e = Math.exp(gamma * tau);
  return (4 * a) / ((1 + a) * (1 + a) * e - ((1 - a) * (1 - a)) / e);
}

/**
 * Fraction of the flux transmitted through the deck that is still travelling
 * close enough to the original solar direction to shape lighting.
 *
 * With `g = 0.85` each scattering event deflects a photon by only ~30 degrees on
 * average, so treating *all* transmitted light as isotropic under-shapes the
 * scene badly. This fraction is reported as directional illuminance in
 * {@link CelestialLightState} in addition to the unscattered beam. It does
 * knowingly overlap with the IBL by that amount; set it to 0 for a strictly
 * energy-exact split and accept a completely shadowless overcast.
 */
const CLOUD_FORWARD_MEMORY = 0.25;

/**
 * Upper bound on sun-disc radiance in render units.
 *
 * The physical value is `E_sun / omega_sun`, roughly 1.5e5 with this module's
 * default irradiance, which overflows a half-float HDR target and turns any
 * downstream bloom into `Inf`. Clamping to 12 000 still tone-maps to a hard
 * white disc under ACES while keeping every buffer finite. This is a numerical
 * guard, not an art decision.
 */
const SUN_DISC_MAX_RADIANCE = 12000;

const DEG = Math.PI / 180;

/**
 * Spectral tint of moonlight relative to sunlight.
 *
 * Reflected sunlight off a slightly reddish regolith is, radiometrically,
 * *warmer* than sunlight. Night reads blue to a human because rod-dominated
 * scotopic vision is more sensitive at short wavelengths — the Purkinje shift.
 * A renderer with no eye model has to bake that in, and this is where.
 */
const MOON_TINT: readonly [number, number, number] = [0.72, 0.84, 1.0];

/**
 * Moon irradiance as a fraction of the solar constant.
 *
 * The physical figure is about 1/400 000 (full moon ~0.25 lux against ~127 klx),
 * built here from the lunar geometric albedo (0.12) and the illuminated
 * fraction. It is then lifted by {@link MOON_EXPOSURE_LIFT}, because a renderer
 * with a fixed exposure and no dark adaptation would otherwise show an
 * unplayable black screen. One constant, in one place, clearly labelled as the
 * cheat it is — rather than a separate "night lighting" system that quietly
 * disagrees with the sky.
 */
const MOON_EXPOSURE_LIFT = 380;

function moonIrradianceScale(illuminatedFraction: number, visibility: number): number {
  return 0.12 * illuminatedFraction * visibility * MOON_EXPOSURE_LIFT * 1e-3;
}

/* -------------------------------------------------------------------------- */
/* Sky                                                                         */
/* -------------------------------------------------------------------------- */

type FloatNode = THREE.Node<'float'>;
type Vec2Node = THREE.Node<'vec2'>;
type Vec3Node = THREE.Node<'vec3'>;

/**
 * `Scene.backgroundNode` is honoured by the node renderer but missing from the
 * `Scene` class declaration in @types/three r185. One narrow structural
 * interface is the honest bridge; see `render/ProceduralSky` for the same
 * workaround.
 */
interface SceneWithBackgroundNode {
  backgroundNode: THREE.Node | null;
}

/** Same story for `Scene.fogNode`, which the node renderer reads but the
 * @types/three r185 `Scene` declaration omits. */
interface SceneWithFogNode {
  fogNode: THREE.Node | null;
}

export class Sky implements GameModule {
  readonly name = 'Sky';

  readonly #options: Required<Omit<SkyOptions, 'timeOfDay' | 'atmosphere' | 'cloudWind'>> & {
    cloudWind: readonly [number, number];
  };
  readonly #atmosphereOptions: AtmosphereOptions;
  readonly #providedClock: TimeOfDay | undefined;

  #atmosphere: Atmosphere | null = null;
  #timeOfDay: TimeOfDay | null = null;
  #ctx: GameContext | null = null;

  /* -- sky-view grid ------------------------------------------------------ */

  readonly #skyW: number;
  readonly #skyH: number;
  /** Completed radiance grid, RGB float, row 0 = zenith. */
  readonly #skyGrid: Float32Array;
  /** Rebuild target. Swapped in only once complete, so no half-updated frame. */
  readonly #skyGridBack: Float32Array;
  /** Next row to march during an incremental rebuild; `#skyH` when idle. */
  #rebuildRow: number;
  #rebuildStartMs = 0;

  #skyTexture: THREE.DataTexture | null = null;
  #envTexture: THREE.DataTexture | null = null;
  #cloudTexture: THREE.DataTexture | null = null;

  /* -- change tracking ---------------------------------------------------- */

  /** True when this module created and registered the clock itself. */
  #ownsClock = false;
  #environmentVersion = 0;

  /** Poll-side adapter registered under {@link SkySunKey}. Built in the ctor. */
  readonly #sunProvider: SkySunProvider;
  #lastClockRevision = -1;
  readonly #lastBuiltSun = new THREE.Vector3(0, -1, 0);
  #lastMoodHash = '';

  /* -- published state ---------------------------------------------------- */

  readonly #celestial: {
    sun: { direction: THREE.Vector3; color: THREE.Color; illuminance: number; visibility: number };
    moon: { direction: THREE.Vector3; color: THREE.Color; illuminance: number; visibility: number };
    skyIrradiance: THREE.Color;
    groundIrradiance: THREE.Color;
    nightFactor: number;
  } = {
    sun: {
      direction: new THREE.Vector3(0, 1, 0),
      color: new THREE.Color(1, 1, 1),
      illuminance: 0,
      visibility: 0,
    },
    moon: {
      direction: new THREE.Vector3(0, -1, 0),
      color: new THREE.Color(0.72, 0.8, 1),
      illuminance: 0,
      visibility: 0,
    },
    skyIrradiance: new THREE.Color(0, 0, 0),
    groundIrradiance: new THREE.Color(0, 0, 0),
    nightFactor: 0,
  };

  /* -- shader uniforms ---------------------------------------------------- */

  readonly #u = {
    sunDirection: uniform(new THREE.Vector3(0, 1, 0)),
    moonDirection: uniform(new THREE.Vector3(0, -1, 0)),
    /** `E_sun * T_sun / omega_sun`, clamped. */
    sunDiscRadiance: uniform(new THREE.Vector3()),
    moonDiscRadiance: uniform(new THREE.Vector3()),
    /** `E_sun * T_sun`: irradiance on a surface facing the sun, above the deck. */
    sunRadianceTop: uniform(new THREE.Vector3()),
    /** Direct solar irradiance reaching the *ground*, i.e. through the deck. */
    sunRadianceGround: uniform(new THREE.Vector3()),
    /** Downward clear-sky *irradiance* reaching the top of the deck. */
    skyAmbient: uniform(new THREE.Vector3()),
    /** Upward radiance from the ground onto the base of the deck. */
    groundBounce: uniform(new THREE.Vector3()),

    extinction: uniform(new THREE.Vector3()),
    scatterRayleigh: uniform(new THREE.Vector3()),
    scatterMie: uniform(0),
    multiScatter: uniform(new THREE.Vector3()),
    miePhaseG: uniform(0.8),

    cloudBaseKm: uniform(0.9),
    cloudThicknessKm: uniform(0.7),
    cloudCoverage: uniform(0.94),
    cloudOpticalDepth: uniform(16),
    cloudScale: uniform(1 / 9),
    cloudWind: uniform(new THREE.Vector2()),
    /** Mean of {@link shapeCloudDensity} at the current coverage. */
    cloudMeanDensity: uniform(0),

    planetRadiusKm: uniform(6360),
    observerRadiusKm: uniform(6360.002),

    nightFactor: uniform(0),
    starIntensity: uniform(0),
    skyIntensity: uniform(1),
  };

  /* -- scratch ------------------------------------------------------------ */

  /** Cosine-weighted hemispherical irradiance of the clear sky, from the grid. */
  readonly #skyIrradianceClear = new THREE.Vector3();
  /** Moonlight as a fraction of solar irradiance, per channel. */
  readonly #moonSkyWeight = new THREE.Vector3();
  readonly #dir = new THREE.Vector3();
  readonly #dir2 = new THREE.Vector3();
  readonly #rgb = new THREE.Vector3();
  readonly #sunT = new THREE.Vector3();

  constructor(options: SkyOptions = {}) {
    this.#providedClock = options.timeOfDay;
    this.#atmosphereOptions = options.atmosphere ?? {};
    this.#options = {
      preset: options.preset ?? 'bloodMoor',
      skyViewWidth: options.skyViewWidth ?? 160,
      environmentWidth: options.environmentWidth ?? 128,
      cloudTextureSize: options.cloudTextureSize ?? 512,
      seed: options.seed ?? 20241,
      rebuildBudgetMs: options.rebuildBudgetMs ?? 1.2,
      sunMoveThresholdDeg: options.sunMoveThresholdDeg ?? 0.35,
      installBackground: options.installBackground ?? true,
      installEnvironment: options.installEnvironment ?? true,
      installFog: options.installFog ?? true,
      stars: options.stars ?? true,
      cloudTileKm: options.cloudTileKm ?? 9,
      cloudWind: options.cloudWind ?? [0.01, 0.004],
    };

    this.#skyW = Math.max(32, this.#options.skyViewWidth);
    this.#skyH = Math.max(16, this.#skyW >> 1);
    this.#skyGrid = new Float32Array(this.#skyW * this.#skyH * 3);
    this.#skyGridBack = new Float32Array(this.#skyGrid.length);
    this.#rebuildRow = this.#skyH;

    // Live view rather than a snapshot: the underlying objects are mutated in
    // place every rebuild, so a consumer that grabbed this once still sees the
    // current sun.
    const self = this;
    this.#sunProvider = {
      get sunDirection(): THREE.Vector3 {
        return self.#celestial.sun.direction;
      },
      get sunColor(): THREE.Color {
        return self.#celestial.sun.color;
      },
      get sunIntensity(): number {
        return self.keyLightFraction;
      },
    };
  }

  /* ---------------------------------------------------------------------- */
  /* Lifecycle                                                              */
  /* ---------------------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;

    this.#atmosphere = new Atmosphere(this.#atmosphereOptions);
    ctx.services.register(AtmosphereKey, this.#atmosphere);
    ctx.services.register(SkyKey, this);
    if (!ctx.services.has(SkySunKey)) ctx.services.register(SkySunKey, this.#sunProvider);

    // Use the integrator's clock if there is one; otherwise own one, so that
    // `engine.add(new Sky())` on its own is a complete system.
    const clock =
      this.#providedClock ??
      ctx.services.tryGet(TimeOfDayKey) ??
      new TimeOfDay({ preset: this.#options.preset });
    this.#timeOfDay = clock;
    if (!ctx.services.has(TimeOfDayKey)) {
      ctx.services.register(TimeOfDayKey, clock);
      this.#ownsClock = true;
    }
    // The clock needs a context to emit events even when the integrator never
    // registered it as a module in its own right. `init` is idempotent.
    clock.init(ctx);

    this.#cloudTexture = createCloudTexture(
      this.#options.cloudTextureSize,
      this.#options.seed,
    );

    this.#skyTexture = createRadianceTexture(this.#skyW, this.#skyH);
    this.#envTexture = createRadianceTexture(
      this.#options.environmentWidth,
      Math.max(8, this.#options.environmentWidth >> 1),
    );
    this.#envTexture.mapping = THREE.EquirectangularReflectionMapping;

    // Full synchronous build. Captures run with zero warmup frames on the
    // WebGPU path, so frame 1 must already be correct.
    this.#syncSunToAtmosphere();
    this.#marchSkyView(0, this.#skyH);
    this.#skyGrid.set(this.#skyGridBack);
    this.#uploadSkyView();
    this.#bakeEnvironment();
    this.#publishCelestial();
    this.#markBuilt();

    if (this.#options.installBackground) {
      (ctx.scene as unknown as SceneWithBackgroundNode).backgroundNode = this.#buildSkyNode();
    }
    if (this.#options.installEnvironment) {
      ctx.scene.environment = this.#envTexture;
    }
    if (this.#options.installFog) {
      (ctx.scene as unknown as SceneWithFogNode).fogNode = this.fogNode();
    }
  }

  update(ctx: GameContext, dt: number): void {
    const clock = this.#timeOfDay;
    const atmosphere = this.#atmosphere;
    if (clock === null || atmosphere === null) return;

    clock.advance(dt, ctx);

    // Cloud drift is a pure uniform update: no rebuild, no upload.
    const wind = this.#options.cloudWind;
    const scale = 1 / this.#options.cloudTileKm;
    this.#u.cloudWind.value.set(
      ctx.time.elapsed * wind[0] * scale,
      ctx.time.elapsed * wind[1] * scale,
    );

    if (this.#rebuildRow < this.#skyH) {
      this.#continueRebuild();
      return;
    }

    if (clock.revision !== this.#lastClockRevision && this.#needsRebuild()) {
      this.#beginRebuild();
    }
  }

  dispose(): void {
    const ctx = this.#ctx;
    if (ctx !== null) {
      if (ctx.scene.environment === this.#envTexture) ctx.scene.environment = null;
      const scene = ctx.scene as unknown as SceneWithBackgroundNode & SceneWithFogNode;
      if (scene.backgroundNode !== null) scene.backgroundNode = null;
      if (scene.fogNode !== null) scene.fogNode = null;
      ctx.services.unregister(AtmosphereKey);
      ctx.services.unregister(SkyKey);
      if (ctx.services.tryGet(SkySunKey) === this.#sunProvider) ctx.services.unregister(SkySunKey);
      // Only tear down the clock registration if this module was the one that
      // put it there; the integrator's own clock outlives the sky.
      if (this.#ownsClock) ctx.services.unregister(TimeOfDayKey);
    }
    this.#skyTexture?.dispose();
    this.#envTexture?.dispose();
    this.#cloudTexture?.dispose();
    this.#skyTexture = null;
    this.#envTexture = null;
    this.#cloudTexture = null;
    this.#ctx = null;
  }

  /* ---------------------------------------------------------------------- */
  /* Public API                                                             */
  /* ---------------------------------------------------------------------- */

  /** The scattering model. Also available as `services.get(AtmosphereKey)`. */
  get atmosphere(): AtmosphereService {
    if (this.#atmosphere === null) throw new Error('[Sky] not initialised');
    return this.#atmosphere;
  }

  /** The clock driving the sun. Also available as `services.get(TimeOfDayKey)`. */
  get timeOfDay(): TimeOfDay {
    if (this.#timeOfDay === null) throw new Error('[Sky] not initialised');
    return this.#timeOfDay;
  }

  /** The equirectangular HDR environment map assigned to `scene.environment`. */
  get environmentTexture(): THREE.DataTexture | null {
    return this.#envTexture;
  }

  /**
   * Bumped every time the environment map's pixels change.
   *
   * The texture object identity is stable across a time-of-day step — the bake
   * writes into the same buffer — so a consumer that caches anything derived
   * from it (a prefiltered chain, spherical harmonics) needs this to know it
   * has gone stale.
   */
  get environmentVersion(): number {
    return this.#environmentVersion;
  }

  /**
   * The most intense of the two celestial lights, as an intensity ratio in
   * `[0, 1]` against a clear-sky sun. Convenience for exposure and for LOD
   * decisions that only care "how much light is there".
   */
  get keyLightFraction(): number {
    const reference = Math.max(1e-6, this.#atmosphere?.solarIlluminance.y ?? 1);
    return THREE.MathUtils.clamp(
      Math.max(this.#celestial.sun.illuminance, this.#celestial.moon.illuminance) / reference,
      0,
      1,
    );
  }

  /** Whether an incremental rebuild is in flight. */
  get rebuilding(): boolean {
    return this.#rebuildRow < this.#skyH;
  }

  /**
   * Current sun/moon/ambient state.
   *
   * The returned object is reused between calls — copy anything you intend to
   * keep. Prefer registering a {@link CelestialLightSink} so you are told when
   * it changes instead of polling.
   */
  celestialLight(): CelestialLightState {
    return this.#celestial;
  }

  /** Apply a named mood and rebuild immediately rather than incrementally. */
  setPreset(name: TimeOfDayPresetName): void {
    this.timeOfDay.setPreset(name);
    this.rebuildNow();
  }

  /** Jump the clock and rebuild immediately. Use for cutscenes and captures. */
  setHours(hours: number): void {
    this.timeOfDay.setHours(hours);
    this.rebuildNow();
  }

  /**
   * Force a complete, synchronous rebuild.
   *
   * ~53 ms at the default resolution — acceptable for a loading screen, a
   * cutscene cut or a headless capture, and never acceptable inside gameplay.
   * Normal time-of-day motion goes through the incremental path automatically.
   */
  rebuildNow(): void {
    if (this.#atmosphere === null) return;
    const started = now();
    this.#syncSunToAtmosphere();
    this.#marchSkyView(0, this.#skyH);
    this.#skyGrid.set(this.#skyGridBack);
    this.#rebuildRow = this.#skyH;
    this.#uploadSkyView();
    this.#bakeEnvironment();
    this.#publishCelestial();
    this.#markBuilt();
    this.#ctx?.events.emit('sky:rebuilt', {
      hours: this.timeOfDay.hours,
      elapsedMs: now() - started,
      sunElevationDeg: this.timeOfDay.sun.elevationDeg,
    });
  }

  /* ---------------------------------------------------------------------- */
  /* Rebuild scheduling                                                     */
  /* ---------------------------------------------------------------------- */

  #moodHash(): string {
    const m = this.timeOfDay.mood;
    return `${m.cloudCoverage}|${m.cloudOpticalDepth}|${m.cloudBaseAltitude}|${m.cloudThickness}|${m.hazeDensity}|${m.mistDensity}|${m.skyIntensity}`;
  }

  #needsRebuild(): boolean {
    const sun = this.timeOfDay.sun.direction;
    const cosMoved = THREE.MathUtils.clamp(this.#lastBuiltSun.dot(sun), -1, 1);
    const movedDeg = Math.acos(cosMoved) / DEG;
    return movedDeg >= this.#options.sunMoveThresholdDeg || this.#moodHash() !== this.#lastMoodHash;
  }

  #markBuilt(): void {
    this.#lastBuiltSun.copy(this.timeOfDay.sun.direction);
    this.#lastMoodHash = this.#moodHash();
    this.#lastClockRevision = this.timeOfDay.revision;
  }

  #beginRebuild(): void {
    this.#syncSunToAtmosphere();
    this.#rebuildRow = 0;
    this.#rebuildStartMs = now();
    // Uniforms that do not depend on the grid can update straight away, so the
    // sun disc and the aerial perspective track the clock at full frame rate
    // even while the low-frequency sky texture catches up.
    this.#updateUniforms();
    this.#continueRebuild();
  }

  #continueRebuild(): void {
    const deadline = now() + this.#options.rebuildBudgetMs;
    while (this.#rebuildRow < this.#skyH && now() < deadline) {
      const end = Math.min(this.#skyH, this.#rebuildRow + 4);
      this.#marchSkyView(this.#rebuildRow, end);
      this.#rebuildRow = end;
    }

    if (this.#rebuildRow >= this.#skyH) {
      this.#skyGrid.set(this.#skyGridBack);
      this.#uploadSkyView();
      this.#bakeEnvironment();
      this.#publishCelestial();
      this.#markBuilt();
      this.#ctx?.events.emit('sky:rebuilt', {
        hours: this.timeOfDay.hours,
        elapsedMs: now() - this.#rebuildStartMs,
        sunElevationDeg: this.timeOfDay.sun.elevationDeg,
      });
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Sky-view grid                                                          */
  /* ---------------------------------------------------------------------- */

  #syncSunToAtmosphere(): void {
    const atmosphere = this.#atmosphere;
    if (atmosphere === null) return;
    const clock = this.timeOfDay;
    atmosphere.aerial.hazeDensity = clock.mood.hazeDensity;
    atmosphere.aerial.mistDensity = clock.mood.mistDensity;
    atmosphere.setSunDirection(clock.sun.direction);

    // Moonlight, expressed as a fraction of the solar irradiance the scattering
    // model is normalised against. Radiative transfer is linear in the source,
    // so the moonlit sky is literally the same ray march with the moon in the
    // sun's place, scaled by this. See `#marchSkyView`.
    //
    // Gated on the sun actually being down and the moon actually being up: at
    // any other time the moon's contribution is a rounding error against
    // daylight and not worth a second ray march.
    const worthMarching = clock.sun.elevationDeg <= 2 && clock.moon.elevationDeg >= -1;
    const scale = worthMarching
      ? moonIrradianceScale(clock.moonIlluminatedFraction, clock.moon.visibility)
      : 0;
    this.#moonSkyWeight.set(
      scale * MOON_TINT[0],
      scale * MOON_TINT[1],
      scale * MOON_TINT[2],
    );
  }

  /**
   * March rows `[rowStart, rowEnd)` of the sky-view grid into the back buffer.
   *
   * Rows are the warped elevation axis, so a partial rebuild covers a
   * contiguous band of the sky — which is why slicing by row (rather than by
   * an interleaved pattern) is the right unit even though the result is never
   * displayed until complete.
   */
  #marchSkyView(rowStart: number, rowEnd: number): void {
    const atmosphere = this.#atmosphere;
    if (atmosphere === null) return;
    const dir = this.#dir;
    const rgb = this.#rgb;
    const grid = this.#skyGridBack;

    for (let y = rowStart; y < rowEnd; y++) {
      const v = (y + 0.5) / this.#skyH;
      const elevation = elevationFromV(v);
      const cosEl = Math.cos(elevation);
      const sinEl = Math.sin(elevation);

      for (let x = 0; x < this.#skyW; x++) {
        const u = (x + 0.5) / this.#skyW;
        const azimuth = (u - 0.5) * Math.PI * 2;
        dir.set(cosEl * Math.sin(azimuth), sinEl, -cosEl * Math.cos(azimuth));
        atmosphere.skyRadiance(dir, rgb);
        const o = (y * this.#skyW + x) * 3;
        grid[o] = rgb.x;
        grid[o + 1] = rgb.y;
        grid[o + 2] = rgb.z;
      }
    }

    // Second pass for moonlight. Without it a moonlit night is a black dome
    // with a bright disc pasted on, which is exactly what it does not look
    // like: the moon lights the whole sky, faintly and blue.
    const weight = this.#moonSkyWeight;
    if (weight.x + weight.y + weight.z <= 1e-5) return;

    const sunDirection = this.#dir2.copy(atmosphere.sunDirection);
    atmosphere.setSunDirection(this.timeOfDay.moon.direction);
    for (let y = rowStart; y < rowEnd; y++) {
      const v = (y + 0.5) / this.#skyH;
      const elevation = elevationFromV(v);
      const cosEl = Math.cos(elevation);
      const sinEl = Math.sin(elevation);
      for (let x = 0; x < this.#skyW; x++) {
        const azimuth = ((x + 0.5) / this.#skyW - 0.5) * Math.PI * 2;
        dir.set(cosEl * Math.sin(azimuth), sinEl, -cosEl * Math.cos(azimuth));
        atmosphere.skyRadiance(dir, rgb);
        const o = (y * this.#skyW + x) * 3;
        grid[o] = (grid[o] as number) + rgb.x * weight.x;
        grid[o + 1] = (grid[o + 1] as number) + rgb.y * weight.y;
        grid[o + 2] = (grid[o + 2] as number) + rgb.z * weight.z;
      }
    }
    atmosphere.setSunDirection(sunDirection);
  }

  /**
   * Cosine-weighted hemispherical irradiance of the clear sky, integrated over
   * the sky-view grid.
   *
   * This is what actually illuminates the top of the cloud deck, and using it
   * instead of `pi * L_zenith` matters: the sky is several times brighter near
   * the horizon and near the sun than at the zenith, and it is far bluer than
   * the direct beam. Getting it wrong is most of the difference between an
   * overcast sky that reads cold and one that reads sepia.
   *
   * The solid angle of a texel in the warped parameterisation is
   * `cos(el) * (2pi/W) * |d el/dv| * (1/H)` with `|d el/dv| = 2 pi |s|`,
   * `s = 1 - 2v` — the Jacobian of {@link elevationFromV}.
   */
  #integrateSkyIrradiance(): void {
    const grid = this.#skyGrid;
    let r = 0;
    let g = 0;
    let b = 0;
    for (let y = 0; y < this.#skyH; y++) {
      const v = (y + 0.5) / this.#skyH;
      const s = 1 - 2 * v;
      if (s <= 0) continue; // lower hemisphere contributes nothing downward
      const elevation = elevationFromV(v);
      const sinEl = Math.sin(elevation);
      const cosEl = Math.cos(elevation);
      const dOmega =
        cosEl * ((2 * Math.PI) / this.#skyW) * ((2 * Math.PI * Math.abs(s)) / this.#skyH);
      const weight = sinEl * dOmega;
      if (weight <= 0) continue;
      for (let x = 0; x < this.#skyW; x++) {
        const o = (y * this.#skyW + x) * 3;
        r += (grid[o] as number) * weight;
        g += (grid[o + 1] as number) * weight;
        b += (grid[o + 2] as number) * weight;
      }
    }
    this.#skyIrradianceClear.set(r, g, b);
  }

  #uploadSkyView(): void {
    const tex = this.#skyTexture;
    if (tex === null) return;
    this.#integrateSkyIrradiance();
    const data = tex.image.data as Uint16Array;
    const grid = this.#skyGrid;
    const toHalf = THREE.DataUtils.toHalfFloat;
    const intensity = this.timeOfDay.mood.skyIntensity;
    for (let i = 0, o = 0; i < grid.length; i += 3, o += 4) {
      data[o] = toHalf((grid[i] as number) * intensity);
      data[o + 1] = toHalf((grid[i + 1] as number) * intensity);
      data[o + 2] = toHalf((grid[i + 2] as number) * intensity);
      data[o + 3] = toHalf(1);
    }
    tex.needsUpdate = true;
    this.#updateUniforms();
  }

  /** Bilinear sample of the completed sky grid, in the warped parameterisation. */
  #sampleSkyGrid(direction: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
    const u = Math.atan2(direction.x, -direction.z) / (Math.PI * 2) + 0.5;
    const v = vFromElevation(Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1)));

    const fx = u * this.#skyW - 0.5;
    const fy = v * this.#skyH - 0.5;
    const x0 = Math.floor(fx);
    const y0 = THREE.MathUtils.clamp(Math.floor(fy), 0, this.#skyH - 1);
    const tx = fx - x0;
    const ty = THREE.MathUtils.clamp(fy - y0, 0, 1);
    const wrap = (x: number): number => ((x % this.#skyW) + this.#skyW) % this.#skyW;
    const xa = wrap(x0);
    const xb = wrap(x0 + 1);
    const yb = Math.min(y0 + 1, this.#skyH - 1);

    const grid = this.#skyGrid;
    const i00 = (y0 * this.#skyW + xa) * 3;
    const i10 = (y0 * this.#skyW + xb) * 3;
    const i01 = (yb * this.#skyW + xa) * 3;
    const i11 = (yb * this.#skyW + xb) * 3;

    const lerp = (c: number): number => {
      const a = (grid[i00 + c] as number) * (1 - tx) + (grid[i10 + c] as number) * tx;
      const b = (grid[i01 + c] as number) * (1 - tx) + (grid[i11 + c] as number) * tx;
      return a * (1 - ty) + b * ty;
    };
    return out.set(lerp(0), lerp(1), lerp(2));
  }

  /* ---------------------------------------------------------------------- */
  /* Environment bake                                                        */
  /* ---------------------------------------------------------------------- */

  /**
   * Bake `scene.environment` from the sky grid, the cloud slab model and an
   * energy-conserving sun disc, and integrate the hemispherical irradiances
   * that {@link CelestialLightState} publishes.
   *
   * Orientation matches `render/ProceduralSky` exactly — row 0 is the zenith,
   * `u = 0` faces `+Z` — because that is the layout three's PMREM already
   * interprets correctly for `EquirectangularReflectionMapping`, and it is
   * covered by `tests/hdri.orientation.test.ts`.
   */
  #bakeEnvironment(): void {
    const tex = this.#envTexture;
    const atmosphere = this.#atmosphere;
    if (tex === null || atmosphere === null) return;

    const width = tex.image.width;
    const height = tex.image.height;
    const data = tex.image.data as Uint16Array;
    const toHalf = THREE.DataUtils.toHalfFloat;

    const clock = this.timeOfDay;
    const mood = clock.mood;
    const sun = clock.sun.direction;
    const intensity = mood.skyIntensity;

    atmosphere.sunTransmittance(this.#sunT);
    const eSun = atmosphere.solarIlluminance;
    const sunIrradiance = new THREE.Vector3(
      this.#sunT.x * eSun.x,
      this.#sunT.y * eSun.y,
      this.#sunT.z * eSun.z,
    );

    // Downward irradiance reaching the top of the deck: direct sun on a
    // horizontal surface, plus the whole clear sky above it.
    const skyIrradiance = this.#skyIrradianceClear;
    const cosSun = Math.max(0, sun.y);

    // Sun solid angle, for the energy-conserving disc contribution below.
    const sunSolidAngle = 2 * Math.PI * (1 - Math.cos(SUN_ANGULAR_RADIUS_DEG * DEG));

    const meanDensity = this.#meanCloudDensity(mood.cloudCoverage);
    const tauVertical = meanDensity * mood.cloudOpticalDepth;

    const dir = this.#dir;
    const rgb = this.#rgb;
    const cloud = new THREE.Vector3();

    let skyIrrR = 0;
    let skyIrrG = 0;
    let skyIrrB = 0;
    let groundIrrR = 0;
    let groundIrrG = 0;
    let groundIrrB = 0;

    // Radiance of the ground hemisphere, filled in as soon as the loop crosses
    // the horizon. See the note where it is computed.
    const groundRadiance = new THREE.Vector3();
    let groundResolved = false;

    for (let y = 0; y < height; y++) {
      // Row 0 is the zenith: phi runs from 0 (up) to pi (down).
      const phi = ((y + 0.5) / height) * Math.PI;
      const sinPhi = Math.sin(phi);
      const dirY = Math.cos(phi);
      // Solid angle of one texel in this row.
      const texelSolidAngle = ((2 * Math.PI) / width) * (Math.PI / height) * sinPhi;

      // The sky-view grid's own lower hemisphere is the atmosphere model's
      // ground bounce, and that ground is lit by an *unobstructed* sun — which
      // is exactly wrong under a deck. Rows above the horizon are already
      // accumulated by the time the loop gets here, so the honest answer is
      // available: a Lambertian ground under the downwelling irradiance this
      // very texture describes. Without this the IBL lights the underside of
      // everything with sunshine the sky says is not there.
      if (!groundResolved && dirY <= 0) {
        groundResolved = true;
        const albedo = atmosphere.params.groundAlbedo;
        groundRadiance.set(
          (skyIrrR * albedo[0]) / Math.PI,
          (skyIrrG * albedo[1]) / Math.PI,
          (skyIrrB * albedo[2]) / Math.PI,
        );
      }

      for (let x = 0; x < width; x++) {
        const theta = ((x + 0.5) / width) * Math.PI * 2;
        dir.set(sinPhi * Math.sin(theta), dirY, sinPhi * Math.cos(theta));

        if (groundResolved) {
          rgb.copy(groundRadiance);
        } else {
          this.#sampleSkyGrid(dir, rgb);
          rgb.multiplyScalar(intensity);
        }

        if (dirY > 0.0) {
          // Uniform slab at the *measured* mean of the shader's density field,
          // so the IBL sits at the same brightness as the sky on screen.
          const viewSlant = Math.min(1 / Math.max(0.16, dirY), 5);
          const alpha = 1 - Math.exp(-tauVertical * viewSlant * 0.9);
          if (alpha > 1e-3) {
            cloudRadiance(
              cloud,
              tauVertical,
              viewSlant,
              meanDensity,
              dir.dot(sun),
              sunIrradiance,
              cosSun,
              skyIrradiance,
              atmosphere.params.groundAlbedo,
            );
            rgb.lerp(cloud, alpha);
          }

          // Energy-conserving sun disc: the disc is far smaller than a texel,
          // so spread its flux across the texel it falls in rather than writing
          // its (enormous) true radiance.
          const cosToSun = dir.dot(sun);
          if (cosToSun > Math.cos(SUN_ANGULAR_RADIUS_DEG * DEG * 1.5) && texelSolidAngle > 0) {
            const through = Math.exp(-tauVertical * 0.9);
            const k = (sunSolidAngle / texelSolidAngle) * through;
            rgb.x += sunIrradiance.x * k;
            rgb.y += sunIrradiance.y * k;
            rgb.z += sunIrradiance.z * k;
          }
        }

        const o = (y * width + x) * 4;
        data[o] = toHalf(rgb.x);
        data[o + 1] = toHalf(rgb.y);
        data[o + 2] = toHalf(rgb.z);
        data[o + 3] = toHalf(1);

        // Cosine-weighted hemispherical irradiance, split by hemisphere.
        const weight = texelSolidAngle * Math.abs(dirY);
        if (dirY > 0) {
          skyIrrR += rgb.x * weight;
          skyIrrG += rgb.y * weight;
          skyIrrB += rgb.z * weight;
        } else {
          groundIrrR += rgb.x * weight;
          groundIrrG += rgb.y * weight;
          groundIrrB += rgb.z * weight;
        }
      }
    }

    this.#celestial.skyIrradiance.setRGB(skyIrrR, skyIrrG, skyIrrB);
    this.#celestial.groundIrradiance.setRGB(groundIrrR, groundIrrG, groundIrrB);

    this.#environmentVersion++;
    tex.needsUpdate = true;
    // Invalidate three's cached PMREM chain; without this the pre-filtered
    // radiance keeps the sky from whenever the texture was first uploaded and
    // the lighting silently stops tracking the time of day.
    tex.needsPMREMUpdate = true;
  }

  /* ---------------------------------------------------------------------- */
  /* Uniforms and published lighting state                                   */
  /* ---------------------------------------------------------------------- */

  #updateUniforms(): void {
    const atmosphere = this.#atmosphere;
    if (atmosphere === null) return;
    const clock = this.timeOfDay;
    const mood = clock.mood;
    const u = this.#u;
    const uni = atmosphere.uniforms;

    u.sunDirection.value.copy(clock.sun.direction);
    u.moonDirection.value.copy(clock.moon.direction);

    atmosphere.sunTransmittance(this.#sunT);
    const eSun = atmosphere.solarIlluminance;
    const sunR = this.#sunT.x * eSun.x * clock.sun.visibility;
    const sunG = this.#sunT.y * eSun.y * clock.sun.visibility;
    const sunB = this.#sunT.z * eSun.z * clock.sun.visibility;
    u.sunRadianceTop.value.set(sunR, sunG, sunB);

    const sunSolidAngle = 2 * Math.PI * (1 - Math.cos(SUN_ANGULAR_RADIUS_DEG * DEG));
    const discScale = Math.min(1 / sunSolidAngle, SUN_DISC_MAX_RADIANCE / Math.max(1e-4, sunG));
    u.sunDiscRadiance.value.set(sunR * discScale, sunG * discScale, sunB * discScale);

    // The moon is sunlit rock: illuminance is the solar constant times the
    // lunar geometric albedo (0.12) times the illuminated fraction, times the
    // atmospheric transmittance along the moon's own slant path. Then it is
    // lifted by a fixed art-direction factor, because a physically exposed
    // night is unplayable in a game with no eye adaptation. The lift is one
    // constant in one place rather than a separate "night lighting" system.
    const moonScale = moonIrradianceScale(
      clock.moonIlluminatedFraction,
      clock.moon.visibility,
    );
    const moonR = eSun.x * moonScale * MOON_TINT[0];
    const moonG = eSun.y * moonScale * MOON_TINT[1];
    const moonB = eSun.z * moonScale * MOON_TINT[2];
    const moonSolidAngle = 2 * Math.PI * (1 - Math.cos(MOON_ANGULAR_RADIUS_DEG * DEG));
    const moonDisc = Math.min(1 / moonSolidAngle, SUN_DISC_MAX_RADIANCE / Math.max(1e-4, moonG));
    u.moonDiscRadiance.value.set(moonR * moonDisc, moonG * moonDisc, moonB * moonDisc);

    u.skyAmbient.value.copy(this.#skyIrradianceClear).multiplyScalar(mood.skyIntensity);

    const albedo = atmosphere.params.groundAlbedo;
    const irr = this.#skyIrradianceClear;
    u.groundBounce.value.set(
      (irr.x * albedo[0]) / Math.PI,
      (irr.y * albedo[1]) / Math.PI,
      (irr.z * albedo[2]) / Math.PI,
    );

    // --- how the deck changes the air underneath it ---------------------
    //
    // Air below an overcast deck is not lit by the sun. Feeding the unoccluded
    // solar irradiance into the aerial-perspective source term is what makes a
    // grey sky sit on top of a glowing haze, and it is the single most visible
    // way for the fog and the sky to disagree. So the deck's transmittance and
    // its own base radiance are pushed into the atmosphere model, which then
    // recomputes the shared uniforms both the CPU and GPU fog paths read.
    u.cloudMeanDensity.value = this.#meanCloudDensity(u.cloudCoverage.value);
    const tauMean = u.cloudMeanDensity.value * Math.max(0, mood.cloudOpticalDepth);
    const alphaMean = 1 - Math.exp(-tauMean * 0.9);
    const sunSlant = 1 / Math.max(0.12, clock.sun.direction.y);
    const beamThrough = Math.exp(-tauMean * sunSlant);
    const shadeMean = 0.32 + 0.68 * Math.exp(-u.cloudMeanDensity.value * 2.2);
    const cosSunZenith = Math.max(0, clock.sun.direction.y);

    const occlusion = atmosphere.aerial.sunOcclusion;
    const ambient = atmosphere.aerial.ambientRadiance;
    const skyE = this.#skyIrradianceClear;
    const topIrr = [
      sunR * cosSunZenith + skyE.x,
      sunG * cosSunZenith + skyE.y,
      sunB * cosSunZenith + skyE.z,
    ] as const;
    const bounce = u.groundBounce.value;
    for (let c = 0; c < 3; c++) {
      const deckT = cloudSlabTransmittance(tauMean, c as 0 | 1 | 2);
      const through = beamThrough + deckT * CLOUD_FORWARD_MEMORY;
      const cloudBase =
        ((topIrr[c] as number) * deckT * shadeMean) / Math.PI + bounce.getComponent(c);
      occlusion.setComponent(c, 1 + (through - 1) * alphaMean);
      ambient.setComponent(c, cloudBase * alphaMean);
    }
    atmosphere.refresh();

    u.sunRadianceGround.value.copy(uni.sunRadiance);
    u.extinction.value.copy(uni.extinction);
    u.scatterRayleigh.value.copy(uni.rayleighScattering);
    u.scatterMie.value = uni.mieScattering;
    u.multiScatter.value.copy(uni.multiScatter);
    u.miePhaseG.value = uni.miePhaseG;

    u.cloudBaseKm.value = mood.cloudBaseAltitude / 1000;
    u.cloudThicknessKm.value = Math.max(0.05, mood.cloudThickness / 1000);
    u.cloudCoverage.value = THREE.MathUtils.clamp(mood.cloudCoverage, 0, 1);
    u.cloudOpticalDepth.value = Math.max(0, mood.cloudOpticalDepth);
    u.cloudScale.value = 1 / Math.max(0.5, this.#options.cloudTileKm);

    u.planetRadiusKm.value = atmosphere.params.bottomRadiusKm;
    u.observerRadiusKm.value =
      atmosphere.params.bottomRadiusKm + atmosphere.observerAltitude / 1000;

    u.nightFactor.value = clock.nightFactor;
    // Stars are hidden by cloud as well as by daylight; both are multiplicative.
    u.starIntensity.value = this.#options.stars
      ? clock.nightFactor * (1 - THREE.MathUtils.clamp(mood.cloudCoverage, 0, 1) * 0.9)
      : 0;
    u.skyIntensity.value = mood.skyIntensity;
  }

  #publishCelestial(): void {
    const atmosphere = this.#atmosphere;
    if (atmosphere === null) return;
    this.#updateUniforms();

    const clock = this.timeOfDay;
    const mood = clock.mood;
    const state = this.#celestial;

    // Direct light that survives the deck. The two-flux slab transmittance is
    // the same expression the shader and the IBL bake use.
    const tau = this.#u.cloudMeanDensity.value * mood.cloudOpticalDepth;
    const sunSlant = 1 / Math.max(0.12, clock.sun.direction.y);
    // Unscattered beam plus the forward-peaked memory of the scattered flux.
    const beam = Math.exp(-tau * sunSlant);
    const through = (c: 0 | 1 | 2): number =>
      beam + cloudSlabTransmittance(tau, c) * CLOUD_FORWARD_MEMORY;

    const sunRadiance = this.#u.sunRadianceTop.value;
    writeSample(
      state.sun,
      clock.sun.direction,
      sunRadiance.x * through(0),
      sunRadiance.y * through(1),
      sunRadiance.z * through(2),
      clock.sun.visibility,
    );

    const moonRadiance = this.#u.moonDiscRadiance.value;
    const moonSolidAngle = 2 * Math.PI * (1 - Math.cos(MOON_ANGULAR_RADIUS_DEG * DEG));
    writeSample(
      state.moon,
      clock.moon.direction,
      moonRadiance.x * moonSolidAngle * through(0),
      moonRadiance.y * moonSolidAngle * through(1),
      moonRadiance.z * moonSolidAngle * through(2),
      clock.moon.visibility,
    );

    state.nightFactor = clock.nightFactor;

    const ctx = this.#ctx;
    if (ctx !== null) {
      ctx.services.tryGet(CelestialLightSinkKey)?.setCelestialLight(state);
      ctx.events.emit('sky:celestial', state);
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Shader graph                                                            */
  /* ---------------------------------------------------------------------- */

  /**
   * Sample the tiling cloud density texture and shape it into a density.
   *
   * Coverage remap, height profile and detail erosion follow Schneider & Vos
   * (SIGGRAPH 2015): coverage selects *where* cloud exists, the height profile
   * gives the deck rounded bottoms and flattened tops, and the erosion pass
   * subtracts high-frequency noise more strongly near the base so silhouettes
   * are wispy rather than uniformly blobby.
   */
  #cloudDensity(uv: Vec2Node, heightFraction: FloatNode): FloatNode {
    const tex = this.#cloudTexture;
    if (tex === null) return float(0);
    const n = texture(tex, uv);
    const cov = this.#u.cloudCoverage;

    // Node-for-node translation of `shapeCloudDensity`. Any change belongs
    // there first, then here.
    const weatherAmount = oneMinus(cov).mul(0.75);
    const effective = cov.mul(oneMinus(weatherAmount.mul(oneMinus(n.w))));
    const base = remapClamp(n.x, oneMinus(effective), float(1.0), float(0.0), float(1.0));

    const bottom = remapClamp(heightFraction, float(-0.3), float(0.25), float(0.0), float(1.0));
    const top = remapClamp(heightFraction, float(0.7), float(1.55), float(1.0), float(0.0));
    const shaped = base.mul(bottom).mul(top);

    const erosion = n.y.mul(0.6).add(n.z.mul(0.4));
    const strength = erosion
      .mul(0.75)
      .mul(oneMinus(heightFraction.mul(0.4)))
      .mul(oneMinus(cov));
    return remapClamp(shaped, strength, float(1.0), float(0.0), float(1.0));
  }

  /**
   * Mean of {@link shapeCloudDensity} over the whole cloud texture and all four
   * layers, at the current coverage.
   *
   * The IBL bake models the deck as a uniform slab, so it needs the *mean*
   * density the shader will actually produce — not the nominal coverage, which
   * is 2-4x higher once the height profile and erosion have taken their cut.
   * Measuring it rather than guessing is what keeps `scene.environment` at the
   * same brightness as the sky the player is looking at.
   *
   * Subsampled on a 48x48 grid: the mean of a stationary noise field converges
   * long before the full 512x512.
   */
  #meanCloudDensity(coverage: number): number {
    const tex = this.#cloudTexture;
    if (tex === null) return 0;
    const size = tex.image.width;
    const data = tex.image.data as Uint8Array;
    const samples = Math.min(48, size);
    const stride = Math.max(1, Math.floor(size / samples));
    const n = { x: 0, y: 0, z: 0, w: 0 };

    let total = 0;
    let count = 0;
    for (let y = 0; y < size; y += stride) {
      for (let x = 0; x < size; x += stride) {
        const o = (y * size + x) * 4;
        n.x = (data[o] as number) / 255;
        n.y = (data[o + 1] as number) / 255;
        n.z = (data[o + 2] as number) / 255;
        n.w = (data[o + 3] as number) / 255;
        for (let layer = 0; layer < CLOUD_LAYERS; layer++) {
          total += shapeCloudDensity(coverage, n, (layer + 0.5) / CLOUD_LAYERS);
          count++;
        }
      }
    }
    return count > 0 ? total / count : 0;
  }

  /** Cornette-Shanks phase function, in TSL. Mirrors the CPU implementation. */
  #phase(mu: FloatNode, g: number): FloatNode {
    const g2 = g * g;
    const numerator = float(3 * (1 - g2)).mul(float(1.0).add(mu.mul(mu)));
    const denominator = float(8 * Math.PI * (2 + g2)).mul(
      pow(max(float(1 + g2).sub(mu.mul(2 * g)), float(1e-4)), float(1.5)),
    );
    return numerator.div(denominator);
  }

  /**
   * Distance along a ray from the observer to a concentric shell of radius
   * `shellRadius`, in km.
   *
   * The observer is always inside the shell, so only the far root exists and
   * the discriminant is unconditionally positive — no branch needed.
   */
  #shellDistance(mu: FloatNode, shellRadius: FloatNode): FloatNode {
    const r = this.#u.observerRadiusKm;
    const disc = r.mul(r).mul(mu.mul(mu).sub(1.0)).add(shellRadius.mul(shellRadius));
    return r.negate().mul(mu).add(sqrt(max(disc, float(0.0))));
  }

  /** Build the complete background node graph. */
  #buildSkyNode(): Vec3Node {
    const u = this.#u;
    const dir = positionWorldDirection.normalize();
    const mu = dir.y;

    /* -- clear sky ------------------------------------------------------- */

    const azimuth = atan(dir.x, dir.z.negate());
    const elevation = asin(clamp(mu, float(-1.0), float(1.0)));
    // v = 0.5 - 0.5 * sign(el) * sqrt(|el| / (pi/2)) — horizon-concentrating.
    const s = elevation.div(Math.PI / 2);
    const skyV = float(0.5).sub(sign(s).mul(sqrt(abs(s))).mul(0.5));
    const skyU = azimuth.div(Math.PI * 2).add(0.5);

    const skyTex = this.#skyTexture;
    let sky: Vec3Node =
      skyTex === null ? vec3(0.0, 0.0, 0.0) : texture(skyTex, vec2(skyU, skyV)).rgb;

    /* -- sun disc -------------------------------------------------------- */

    const cosSun = dot(dir, u.sunDirection);
    const sunAngle = acos(clamp(cosSun, float(-1.0), float(1.0)));
    const sunR = sunAngle.div(SUN_ANGULAR_RADIUS_DEG * DEG);
    // Analytic antialiasing: the disc is about 9 pixels across at 1080p/60deg,
    // so its edge has to be feathered by exactly one pixel or it crawls.
    const sunFeather = max(fwidth(sunR), float(1e-4));
    const sunMask = oneMinus(smoothstep(float(1.0).sub(sunFeather), float(1.0), sunR));
    // Hestroffer & Magnan limb darkening, single-exponent form.
    const sunMu = sqrt(max(oneMinus(sunR.mul(sunR)), float(0.0)));
    const limb = pow(max(sunMu, float(1e-3)), vec3(0.397, 0.503, 0.652));
    sky = sky.add(u.sunDiscRadiance.mul(limb).mul(sunMask));

    /* -- moon ------------------------------------------------------------ */

    const cosMoon = dot(dir, u.moonDirection);
    const moonAngle = acos(clamp(cosMoon, float(-1.0), float(1.0)));
    const moonRadius = MOON_ANGULAR_RADIUS_DEG * DEG;
    const moonR = moonAngle.div(moonRadius);
    const moonFeather = max(fwidth(moonR), float(1e-4));
    const moonMask = oneMinus(smoothstep(float(1.0).sub(moonFeather), float(1.0), moonR));
    // Reconstruct the sphere normal across the disc so the terminator is a real
    // curve rather than a painted crescent.
    const tangential = dir.sub(u.moonDirection.mul(cosMoon)).div(moonRadius);
    const nz = sqrt(max(oneMinus(dot(tangential, tangential)), float(0.0)));
    const moonNormal = tangential.add(u.moonDirection.mul(nz)).normalize();
    // The lunar regolith is markedly non-Lambertian; sqrt() flattens the
    // terminator toward the Lommel-Seeliger falloff the real moon shows.
    const moonLit = sqrt(saturate(dot(moonNormal, u.sunDirection)));
    sky = sky.add(u.moonDiscRadiance.mul(moonLit).mul(moonMask));

    /* -- stars ----------------------------------------------------------- */

    if (this.#options.stars) {
      const cell = dir.mul(190.0);
      const id = floor(cell);
      const h0 = hash(id.x.mul(157.1).add(id.y.mul(113.7)).add(id.z.mul(271.9)));
      const h1 = hash(h0.mul(97.3).add(11.7));
      const h2 = hash(h1.mul(53.1).add(3.3));
      const centre = vec3(h0, h1, h2).mul(0.7).add(0.15);
      const d = length(fract(cell).sub(centre));
      // step() keeps roughly one cell in 40 populated; magnitude varies as the
      // cube of a uniform so the field is dominated by a few bright stars.
      const populated = step(float(0.975), hash(h2.mul(31.7).add(7.1)));
      const magnitude = h1.mul(h1).mul(h1).mul(0.9).add(0.1);
      const twinkle = oneMinus(smoothstep(float(0.0), float(0.08), d));
      sky = sky.add(
        vec3(0.86, 0.9, 1.0)
          .mul(twinkle.mul(populated).mul(magnitude))
          .mul(u.starIntensity)
          .mul(3.0),
      );
    }

    /* -- cloud deck ------------------------------------------------------ */

    const planet = u.planetRadiusKm;
    const baseKm = u.cloudBaseKm;
    const thickness = u.cloudThicknessKm;
    const scale = u.cloudScale;

    // Clouds cover everything above the horizon — the deck's shell is hit for
    // *any* upward ray, however shallow. Only below the horizon does the ground
    // come first and the shell intersection become the one behind the planet.
    // Gating any higher than this leaves a band of un-clouded sky along the
    // horizon, and because that band is where the clear-sky aureole is
    // brightest it reads as a blazing hole under an otherwise leaden deck.
    const aboveHorizon = smoothstep(float(-0.004), float(0.006), mu);

    // Anchor on the middle of the deck, then displace each layer along the view
    // ray by its own shell intersection. That displacement *is* the parallax:
    // overhead the layers barely separate, toward the horizon they slide apart
    // and the deck is seen edge-on.
    const midDistance = this.#shellDistance(mu, planet.add(baseKm).add(thickness.mul(0.5)));
    const midUv = vec2(dir.x, dir.z).mul(midDistance).mul(scale).add(u.cloudWind);
    const uvPerKm = vec2(dir.x, dir.z).mul(scale);

    let densitySum: FloatNode = float(0.0);
    for (let i = 0; i < CLOUD_LAYERS; i++) {
      const heightFraction = (i + 0.5) / CLOUD_LAYERS;
      const shell = planet.add(baseKm).add(thickness.mul(heightFraction));
      const offset = uvPerKm.mul(this.#shellDistance(mu, shell).sub(midDistance));
      // Clamped, because a single 2D noise field has no vertical correlation:
      // once the layers are more than about a cloud apart in the texture they
      // are sampling unrelated weather, and the deck breaks into hard
      // horizontal stripes near the horizon. Limiting the shear degrades the
      // parallax to a single coherent sheet exactly where its vertical
      // structure stops being resolvable anyway.
      const shear = length(offset);
      const limited = offset.mul(min(float(1.0), float(CLOUD_MAX_SHEAR).div(max(shear, float(1e-5)))));
      densitySum = densitySum.add(this.#cloudDensity(midUv.add(limited), float(heightFraction)));
    }

    // Toward the horizon the deck is compressed so hard that a single pixel
    // covers tens of texels. Mip filtering prefilters the *density*, but the
    // coverage remap immediately re-thresholds that average back into a hard
    // edge, so the deck breaks into blocks the size of a high mip's texels.
    //
    // The fix has to be applied to the shaped density, not the sample: once the
    // filter footprint exceeds the feature size, converge to the analytic mean
    // — which is what an ideal filter would return anyway. Driving the blend
    // off the actual screen-space footprint rather than off distance makes it
    // correct at any resolution, field of view and cloud scale.
    const footprint = length(fwidth(midUv));
    const meanDensity = u.cloudMeanDensity;
    const density = mix(
      densitySum.div(CLOUD_LAYERS),
      meanDensity,
      smoothstep(float(0.005), float(0.035), footprint),
    );

    // Vertical optical depth drives how much light gets through the deck; the
    // view-slanted depth drives how much of the sky behind it is hidden. They
    // are different quantities and conflating them makes a grazing deck both
    // wrongly opaque and wrongly dark.
    const tau = density.mul(u.cloudOpticalDepth);
    const viewSlant = min(float(1.0).div(max(mu, float(0.16))), float(5.0));
    const alpha = oneMinus(exp(tau.mul(viewSlant).mul(-0.9))).mul(aboveHorizon);

    // Two shadow probes offset along the sun's horizontal projection. The
    // offset per unit of vertical travel is tan(sunZenith), clamped so a sun on
    // the horizon does not produce an unbounded offset.
    const sunHoriz = vec2(u.sunDirection.x, u.sunDirection.z);
    const sunHorizLen = max(length(sunHoriz), float(1e-3));
    const slant = float(1.0).div(max(u.sunDirection.y, float(0.12)));
    const sunStep = sunHoriz.div(sunHorizLen).mul(thickness).mul(slant).mul(scale);
    const shadowA = this.#cloudDensity(midUv.add(sunStep.mul(0.35)), float(0.62));
    const shadowB = this.#cloudDensity(midUv.add(sunStep.mul(0.85)), float(0.9));
    // Deliberately in *density*, not optical depth: this is a normalised
    // shading term for form, and tying it to the optical depth would flatten
    // the whole deck to the darkest value as soon as the cloud got thick.
    const shadowDensity = shadowA.add(shadowB).mul(0.5);

    // Two-stream diffuse transmittance of the slab, per channel. Three exp()
    // calls buy the entire cold cast of an overcast sky; see
    // CLOUD_SINGLE_SCATTER_ALBEDO.
    const gammaTau = vec3(
      CLOUD_TWO_STREAM.gamma[0],
      CLOUD_TWO_STREAM.gamma[1],
      CLOUD_TWO_STREAM.gamma[2],
    ).mul(tau);
    const eUp = exp(gammaTau);
    const eDown = exp(gammaTau.negate());
    const aVec = vec3(
      CLOUD_TWO_STREAM.asym[0],
      CLOUD_TWO_STREAM.asym[1],
      CLOUD_TWO_STREAM.asym[2],
    );
    const onePlus = aVec.add(1.0);
    const oneMinusA = float(1.0).sub(aVec);
    const diffuseT = aVec
      .mul(4.0)
      .div(max(onePlus.mul(onePlus).mul(eUp).sub(oneMinusA.mul(oneMinusA).mul(eDown)), vec3(1e-5, 1e-5, 1e-5)));

    // Irradiance arriving on the top of the deck: direct sun plus skylight.
    const topIrradiance = u.sunRadianceTop.mul(saturate(u.sunDirection.y)).add(u.skyAmbient);
    // Lambertian base radiance, shaped by the self-shadow probes so the deck
    // has form instead of reading as a flat ceiling.
    const shade = mix(float(0.32), float(1.0), exp(shadowDensity.mul(-2.2)));
    let cloudColor: Vec3Node = topIrradiance
      .mul(diffuseT)
      .div(Math.PI)
      .mul(shade)
      .add(u.groundBounce);


    // Single-scattered direct beam. Both limits are the physical ones: it
    // vanishes as the deck thins to nothing (no scatterers) *and* as it
    // thickens (the beam is extinguished before it reaches the base), peaking
    // on thin edges — which is precisely where a silver lining appears, and why
    // an overcast sky has a diffuse hot spot rather than a visible sun.
    const beamPhase = this.#phase(cosSun, CLOUD_PHASE_G);
    const beam = u.sunRadianceTop
      .mul(beamPhase)
      .mul(oneMinus(exp(tau.mul(viewSlant).negate())))
      .mul(exp(tau.mul(slant).negate()));
    cloudColor = cloudColor.add(beam);

    // Aerial perspective between camera and deck, using the shared model, so
    // the deck at the horizon converges onto the haze in front of it.
    cloudColor = this.#applyAerialPerspective(cloudColor, dir, midDistance);

    const result = mix(sky, cloudColor, saturate(alpha));
    return vec3(max(result, vec3(0.0, 0.0, 0.0)));
  }

  /**
   * The shader half of {@link AtmosphereService.aerialPerspective}.
   *
   * Same closed-form solution, same uniforms, same coefficients as the CPU
   * path. Exposed publicly so terrain, water and any other pass composites
   * against exactly this fog and no other.
   */
  #applyAerialPerspective(color: Vec3Node, direction: Vec3Node, distanceKm: FloatNode): Vec3Node {
    const u = this.#u;
    const d = max(distanceKm, float(0.0));
    const extinction = max(u.extinction, vec3(1e-7, 1e-7, 1e-7));
    const transmittance = exp(extinction.mul(d).negate());

    const mu = dot(direction, u.sunDirection);
    const phaseR = float(3 / (16 * Math.PI)).mul(float(1.0).add(mu.mul(mu)));
    const phaseM = this.#phase(mu, 0.8);

    const scatter = u.scatterRayleigh.add(u.scatterMie);
    const source = u.scatterRayleigh
      .mul(phaseR)
      .add(u.scatterMie.mul(phaseM))
      .mul(u.sunRadianceGround)
      .add(scatter.mul(u.multiScatter));

    const inscatter = source.div(extinction).mul(oneMinus(transmittance));
    return color.mul(transmittance).add(inscatter);
  }

  /**
   * Composite `color` (a surface radiance) against the atmosphere, for a
   * surface `distance` **metres** away along `direction`.
   *
   * This is the entry point other render modules should call. It is the
   * shader-side twin of {@link AtmosphereService.aerialPerspective}; both are
   * driven by the same uniforms, so they cannot disagree.
   *
   * ```ts
   * const sky = ctx.services.tryGet(SkyKey);
   * material.colorNode = sky
   *   ? sky.aerialPerspectiveNode(litColor, viewDirWorld, distanceNode)
   *   : litColor;
   * ```
   */
  aerialPerspectiveNode(color: Vec3Node, direction: Vec3Node, distance: FloatNode): Vec3Node {
    return this.#applyAerialPerspective(color, direction, distance.mul(0.001));
  }

  /**
   * A node for `scene.fogNode`, applying this module's aerial perspective to
   * every material's final colour.
   *
   * This is the intended integration point, and it is installed automatically
   * unless {@link SkyOptions.installFog} is false. Three's node renderer hands
   * a fog node the already-lit output and takes its result as the final colour,
   * so one assignment gives terrain, props, characters and vegetation the same
   * extinction and in-scattering as the sky behind them — without any of those
   * systems knowing this module exists, and without anyone being tempted to add
   * a second `FogExp2` that disagrees with it.
   *
   * ```ts
   * scene.fogNode = sky.fogNode();   // done automatically at init
   * ```
   */
  fogNode(): THREE.Node<'vec4'> {
    const toFragment = positionWorld.sub(cameraPosition);
    const distanceKm = length(toFragment).mul(0.001);
    const direction = normalize(toFragment);
    return vec4(
      this.#applyAerialPerspective(output.rgb, direction, distanceKm),
      output.a,
    );
  }

  /** The live sky-view radiance texture, for passes that want the sky colour. */
  get skyViewTexture(): THREE.DataTexture | null {
    return this.#skyTexture;
  }
}

/* -------------------------------------------------------------------------- */
/* Convenience entry point                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Create the sky module.
 *
 * ```ts
 * engine.add(createSky());                       // Blood Moor overcast
 * engine.add(createSky({ preset: 'dusk' }));
 * ```
 */
export function createSky(options: SkyOptions = {}): Sky {
  return new Sky(options);
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/**
 * Elevation, in radians, for a row coordinate in the horizon-warped sky-view
 * parameterisation. `v = 0` is the zenith, `v = 1` the nadir.
 */
export function elevationFromV(v: number): number {
  const s = 1 - 2 * v;
  return Math.sign(s) * s * s * (Math.PI / 2);
}

/** Exact inverse of {@link elevationFromV}. */
export function vFromElevation(elevationRad: number): number {
  const s = elevationRad / (Math.PI / 2);
  return 0.5 - 0.5 * Math.sign(s) * Math.sqrt(Math.abs(s));
}

/**
 * Shape a raw cloud-noise sample into a density.
 *
 * **This is the reference implementation.** `Sky.#cloudDensity` is its TSL
 * translation, node for node; `Sky.#meanCloudDensity` runs it over the texture
 * to derive the mean the IBL bake needs. Keeping one authored version and two
 * derived ones is what stops the visible sky and the lighting from drifting
 * apart as this gets tuned.
 *
 * Coverage remap, height profile and detail erosion follow Schneider & Vos,
 * *"The Real-Time Volumetric Cloudscapes of Horizon Zero Dawn"* (SIGGRAPH
 * 2015): coverage decides where cloud exists, the profile gives the deck
 * rounded bottoms and flattened tops, and erosion subtracts high-frequency
 * detail more strongly near the base so silhouettes are wispy rather than
 * uniformly blobby.
 */
export function shapeCloudDensity(
  coverage: number,
  /** R low-frequency coverage, G detail, B billow, A weather mask, all `[0,1]`. */
  n: { x: number; y: number; z: number; w: number },
  heightFraction: number,
): number {
  // The weather mask must not punch holes in a genuinely overcast deck, so its
  // authority shrinks as coverage approaches 1.
  const weatherAmount = (1 - coverage) * 0.75;
  const effective = coverage * (1 - weatherAmount * (1 - n.w));
  const base = remap01(n.x, 1 - effective, 1);
  // A gentle bell rather than a cumulus tower: this deck is stratiform, and a
  // steep profile makes the outer layers so sparse that the nominal coverage
  // stops meaning anything.
  const bottom = remap01(heightFraction, -0.3, 0.25);
  const top = 1 - remap01(heightFraction, 0.7, 1.55);
  const shaped = base * bottom * top;
  // Erosion authority falls to exactly zero at full coverage. That is both the
  // physical truth — an unbroken stratus deck has no edges to erode — and the
  // property that makes `coverage` mean what it says: with the coverage channel
  // histogram-equalised (see `createCloudTexture`) and erosion out of the way,
  // the covered fraction of the sky *is* `coverage`.
  const erosion = n.y * 0.6 + n.z * 0.4;
  const strength = erosion * 0.75 * (1 - heightFraction * 0.4) * (1 - coverage);
  return remap01(shaped, strength, 1);
}

function remap01(value: number, low: number, high: number): number {
  const t = (value - low) / Math.max(1e-5, high - low);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/**
 * Radiance leaving the base of a cloud slab of optical depth `tau`.
 *
 * The two-flux solution for conservative scattering under the similarity
 * relation (van de Hulst 1980) plus the forward-scattered direct beam. Shared
 * between the IBL bake and — in its TSL translation — the background shader, so
 * a change here changes both.
 */
function cloudRadiance(
  out: THREE.Vector3,
  /** Vertical optical depth of the deck. */
  tauVertical: number,
  /** `1 / cos(view zenith)`, clamped — the extra path a slanted view sees. */
  viewSlant: number,
  /** Mean density, for the normalised self-shadow term. */
  meanDensity: number,
  cosSun: number,
  sunIrradiance: THREE.Vector3,
  cosSunZenith: number,
  /** Clear-sky hemispherical irradiance on the top of the deck. */
  skyIrradiance: THREE.Vector3,
  groundAlbedo: readonly [number, number, number],
): THREE.Vector3 {
  const shade = 0.32 + 0.68 * Math.exp(-meanDensity * 2.2);
  const diffR = cloudSlabTransmittance(tauVertical, 0);
  const diffG = cloudSlabTransmittance(tauVertical, 1);
  const diffB = cloudSlabTransmittance(tauVertical, 2);
  const sunSlant = 1 / Math.max(0.12, cosSunZenith);
  const beamFactor =
    cornetteShanksPhase(cosSun, CLOUD_PHASE_G) *
    (1 - Math.exp(-tauVertical * viewSlant)) *
    Math.exp(-tauVertical * sunSlant);

  const topR = sunIrradiance.x * cosSunZenith + skyIrradiance.x;
  const topG = sunIrradiance.y * cosSunZenith + skyIrradiance.y;
  const topB = sunIrradiance.z * cosSunZenith + skyIrradiance.z;

  // Ground bounce onto the underside, from the same skylight.
  const bounceR = (skyIrradiance.x * groundAlbedo[0]) / Math.PI;
  const bounceG = (skyIrradiance.y * groundAlbedo[1]) / Math.PI;
  const bounceB = (skyIrradiance.z * groundAlbedo[2]) / Math.PI;

  return out.set(
    (topR * diffR * shade) / Math.PI + sunIrradiance.x * beamFactor + bounceR,
    (topG * diffG * shade) / Math.PI + sunIrradiance.y * beamFactor + bounceG,
    (topB * diffB * shade) / Math.PI + sunIrradiance.z * beamFactor + bounceB,
  );
}

/** Normalise an illuminance triple into a hue plus a scalar. */
function writeSample(
  target: {
    direction: THREE.Vector3;
    color: THREE.Color;
    illuminance: number;
    visibility: number;
  },
  direction: THREE.Vector3,
  r: number,
  g: number,
  b: number,
  visibility: number,
): void {
  target.direction.copy(direction);
  const peak = Math.max(r, g, b);
  if (peak <= 1e-6) {
    target.color.setRGB(1, 1, 1);
    target.illuminance = 0;
  } else {
    target.color.setRGB(r / peak, g / peak, b / peak);
    target.illuminance = peak;
  }
  target.visibility = visibility;
}

/** An empty half-float RGBA equirect texture in linear space. */
function createRadianceTexture(width: number, height: number): THREE.DataTexture {
  const texture = new THREE.DataTexture(
    new Uint16Array(width * height * 4),
    width,
    height,
    THREE.RGBAFormat,
    THREE.HalfFloatType,
  );
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  // Horizontal wrap is what makes the azimuth seam invisible; vertical must
  // clamp or the zenith row bleeds into the nadir row.
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Build the tiling cloud density texture.
 *
 * Four channels, chosen so the shader needs exactly one fetch per sample:
 *
 * - **R** low-frequency coverage — decides where cloud is at all.
 * - **G** mid-frequency fBm — the first erosion pass.
 * - **B** inverted Worley (billow) — the second erosion pass. Cellular noise is
 *   what gives cumuliform edges their cauliflower silhouette; fBm alone reads
 *   as smoke.
 * - **A** a very low-frequency weather mask, so coverage varies across the map
 *   and a fully overcast sky still has structure.
 *
 * Every octave uses the tiling variants in `assets/Procedural`, so the texture
 * wraps exactly and the horizon shows no repeat seam.
 */
export function createCloudTexture(size: number, seed: number): THREE.DataTexture {
  const simplex = new SimplexNoise(seed);
  const weather = new SimplexNoise(seed + 977);
  const worleyA = new WorleyNoise(seed + 31);
  const worleyB = new WorleyNoise(seed + 67);

  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const v = (y + 0.5) / size;
    for (let x = 0; x < size; x++) {
      const u = (x + 0.5) / size;

      const coverage = simplex.tileableFbm(u, v, 3, 5, 2, 0.55);
      const detail = simplex.tileableFbm(u, v, 11, 4);
      const billowA = 1 - Math.min(1, worleyA.sample(u, v, 9).f1 * 1.7);
      const billowB = 1 - Math.min(1, worleyB.sample(u, v, 21).f1 * 1.7);
      const billow = billowA * 0.62 + billowB * 0.38;
      const mask = weather.tileableFbm(u, v, 1.5, 3);

      const o = (y * size + x) * 4;
      data[o] = Math.round(255 * clamp01(coverage));
      data[o + 1] = Math.round(255 * clamp01(detail));
      data[o + 2] = Math.round(255 * clamp01(billow));
      data[o + 3] = Math.round(255 * clamp01(mask));
    }
  }

  // Coverage and weather are *thresholded* by the shader, so their absolute
  // distribution is what `coverage` means. fBm is roughly Gaussian, which makes
  // "0.6 coverage" cover far less than 60% of the sky and "0.2" cover almost
  // nothing. Equalising their histograms to uniform makes the parameter
  // linear and honest; the detail and billow channels are *subtracted*, not
  // thresholded, so they are left alone.
  equalizeChannel(data, 0);
  equalizeChannel(data, 3);

  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  // Mask data, not colour: decoding it as sRGB would bend every remap.
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  // Mipmaps are not optional here: near the horizon a single pixel covers many
  // tiles of this texture, and without them the deck boils into noise.
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = 4;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Remap one channel of an RGBA8 buffer so its values are uniformly distributed
 * over `[0, 255]`, via a 256-bin cumulative histogram. Order-preserving, so the
 * spatial structure of the noise is untouched — only its distribution changes.
 */
function equalizeChannel(data: Uint8Array, channel: 0 | 1 | 2 | 3): void {
  const histogram = new Uint32Array(256);
  const pixels = data.length >> 2;
  for (let i = 0; i < pixels; i++) histogram[data[(i << 2) + channel] as number]!++;

  const lut = new Uint8Array(256);
  let cumulative = 0;
  for (let v = 0; v < 256; v++) {
    // Midpoint of the bin's cumulative range, so neither end saturates.
    const count = histogram[v] as number;
    lut[v] = Math.round((255 * (cumulative + count * 0.5)) / Math.max(1, pixels));
    cumulative += count;
  }
  for (let i = 0; i < pixels; i++) {
    const index = (i << 2) + channel;
    data[index] = lut[data[index] as number] as number;
  }
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}
