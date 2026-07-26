/**
 * @module render/Atmosphere
 *
 * The single source of truth for how light behaves in the air, for **every**
 * pass in the renderer.
 *
 * ### Technique
 *
 * This is a CPU-side implementation of Sébastien Hillaire's
 * *"A Scalable and Production Ready Sky and Atmosphere Rendering Technique"*
 * (EGSR 2020), which is itself a re-parameterisation of Bruneton & Neyret's
 * *"Precomputed Atmospheric Scattering"* (EGSR 2008) built for realtime
 * dynamic time-of-day. The three pieces that matter are all here:
 *
 * 1. **Transmittance LUT** — `T(r, mu)`, the extinction between a point at
 *    radius `r` and the top of the atmosphere along a ray with cosine `mu`.
 *    Independent of the sun, so it is built exactly once. Hillaire §4,
 *    parameterisation from Bruneton's `GetTransmittanceTextureUvFromRMu`.
 * 2. **Multiple-scattering LUT** — `Psi_ms(r, mu_s)`. Hillaire's key
 *    contribution (§5): compute *second*-order scattering `L_2nd` and the
 *    single-bounce transfer function `f_ms` over a uniformly sampled sphere,
 *    then close the infinite series analytically as
 *    `Psi_ms = L_2nd / (1 - f_ms)`. This replaces Bruneton's 4D scattering
 *    texture and iterative accumulation with one small 2D table, which is the
 *    whole reason a dynamic sky is affordable.
 * 3. **Ray march** — energy-conserving analytic integration of the in-scatter
 *    over each segment (`(S - S*T_seg) / sigma_t`, Hillaire §4), rather than a
 *    midpoint rectangle rule that loses energy at high optical depth.
 *
 * Media model follows Bruneton's Earth fit: exponential Rayleigh (scale height
 * 8 km), exponential Mie aerosol (1.2 km), and a tent-shaped ozone absorption
 * layer centred at 25 km. Ozone is not cosmetic — without it the zenith at
 * twilight goes muddy brown instead of the deep blue that actually happens,
 * because ozone's Chappuis band is what removes the yellow-green from a long
 * slant path.
 *
 * ### Why CPU
 *
 * Hillaire computes these in compute/fullscreen passes. Here they are computed
 * in TypeScript and consumed as (a) a small equirect radiance texture built by
 * {@link module:render/Sky} and (b) plain uniforms. Reasons, in order:
 *
 * - The aerial-perspective model must be queryable by *gameplay-adjacent* code
 *   (what colour is that distant landmark, how far can the player see) and by
 *   any pass that has no render-target budget. A GPU-only model cannot answer
 *   that without a readback.
 * - Both backends (WebGPU and the WebGL2 fallback) get bit-identical results,
 *   because the expensive part never touches a shader.
 * - The per-frame cost is bounded and amortisable (see {@link module:render/Sky}),
 *   whereas a per-frame LUT chain is a fixed tax whether the sun moved or not.
 *
 * The parameterisation helpers below (`transmittanceUvFromRMu` and its inverse)
 * are written to be portable verbatim to TSL, so moving the LUT generation onto
 * the GPU later is a mechanical translation, not a redesign.
 *
 * ### Units
 *
 * Lengths are **kilometres** throughout (scattering coefficients are 1/km, as
 * published). The public API takes and returns metres where it touches world
 * space, because that is what the rest of the engine uses.
 *
 * Radiometry is normalised: `solarIlluminance` is the top-of-atmosphere solar
 * irradiance in the engine's own linear render units. The default is calibrated
 * so a clear-noon zenith sits near 1.5 and sunlit ground near 10 under ACES at
 * exposure 1.0, matching the range the rest of the project is authored against.
 */

import * as THREE from 'three/webgpu';

import { serviceKey } from '../core/ServiceLocator';

/* -------------------------------------------------------------------------- */
/* Medium parameters                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Static description of the participating medium.
 *
 * Changing any of these invalidates both LUTs, so they are treated as
 * construction-time constants rather than per-frame state. Weather varies
 * through {@link AerialPerspectiveSettings.hazeDensity} instead, which scales
 * the near-ground aerosol without a rebuild.
 */
export interface AtmosphereParams {
  /** Planet radius in km. Earth: 6360. */
  readonly bottomRadiusKm: number;
  /** Top-of-atmosphere radius in km. Earth: 6460. */
  readonly topRadiusKm: number;

  /** Rayleigh scattering coefficient at sea level, 1/km, per RGB channel. */
  readonly rayleighScattering: readonly [number, number, number];
  /** Rayleigh density scale height, km. */
  readonly rayleighScaleHeightKm: number;

  /** Mie scattering coefficient at sea level, 1/km (grey). */
  readonly mieScattering: number;
  /** Mie extinction (scattering + absorption) at sea level, 1/km. */
  readonly mieExtinction: number;
  /** Mie density scale height, km. */
  readonly mieScaleHeightKm: number;
  /** Cornette-Shanks asymmetry parameter. 0.8 for haze, higher for fog. */
  readonly miePhaseG: number;

  /** Ozone absorption coefficient at the layer peak, 1/km, per RGB channel. */
  readonly ozoneAbsorption: readonly [number, number, number];
  /** Altitude of the ozone tent's peak, km. */
  readonly ozoneCenterKm: number;
  /** Half-width of the ozone tent, km. Density falls to 0 at center +/- width. */
  readonly ozoneWidthKm: number;

  /** Lambertian albedo of the planet surface, used for the ground bounce. */
  readonly groundAlbedo: readonly [number, number, number];
}

/**
 * Earth, per Bruneton's `demo/model.cc` reference fit, converted to 1/km.
 *
 * `groundAlbedo` deliberately departs from Bruneton's 0.1 grey: Act I is
 * rain-soaked mud and dead grass, and the ground bounce is a real part of why
 * the lower hemisphere of an overcast sky reads warm-brown rather than neutral.
 * A wet-earth albedo is both physically defensible and on-mood.
 */
export const EARTH_ATMOSPHERE: AtmosphereParams = {
  bottomRadiusKm: 6360,
  topRadiusKm: 6460,

  rayleighScattering: [5.802e-3, 13.558e-3, 33.1e-3],
  rayleighScaleHeightKm: 8.0,

  mieScattering: 3.996e-3,
  mieExtinction: 4.44e-3,
  mieScaleHeightKm: 1.2,
  miePhaseG: 0.8,

  ozoneAbsorption: [0.65e-3, 1.881e-3, 0.085e-3],
  ozoneCenterKm: 25.0,
  ozoneWidthKm: 15.0,

  groundAlbedo: [0.085, 0.075, 0.058],
};

/* -------------------------------------------------------------------------- */
/* Pure math — exported because it is what the unit tests can actually pin      */
/* -------------------------------------------------------------------------- */

/**
 * Rayleigh phase function, normalised so that it integrates to 1 over the
 * sphere. `mu` is `dot(viewDir, lightDir)`.
 */
export function rayleighPhase(mu: number): number {
  return (3 / (16 * Math.PI)) * (1 + mu * mu);
}

/**
 * Cornette-Shanks phase function — the Mie approximation Hillaire uses.
 *
 * Preferred over plain Henyey-Greenstein because HG is singular in its
 * derivative at `mu = 1` and visibly wrong in the backscatter hemisphere, which
 * is exactly where an overcast sky spends most of its solid angle.
 *
 * Cornette & Shanks, *"Physically reasonable analytic expression for the
 * single-scattering phase function"*, Applied Optics 31(16), 1992.
 */
export function cornetteShanksPhase(mu: number, g: number): number {
  const g2 = g * g;
  const num = 3 * (1 - g2) * (1 + mu * mu);
  const den = 8 * Math.PI * (2 + g2) * Math.pow(1 + g2 - 2 * g * mu, 1.5);
  return num / den;
}

/** Isotropic phase, `1 / 4pi`. Used for the multiple-scattering term. */
export const UNIFORM_PHASE = 1 / (4 * Math.PI);

/**
 * Distance from `r0` (with view cosine `mu`) to a sphere of radius `radius`,
 * for a ray starting inside it.
 *
 * Returns the *nearest positive* root, or `-1` when the ray misses. The
 * quadratic is written in the `b^2 - 4ac` form with `b = r*mu` because that is
 * the numerically stable arrangement when `r` is ~6360 and the discriminant is
 * a difference of two very large numbers.
 */
export function raySphereDistance(r: number, mu: number, radius: number): number {
  const discriminant = r * r * (mu * mu - 1) + radius * radius;
  if (discriminant < 0) return -1;
  const sqrtDisc = Math.sqrt(discriminant);
  const near = -r * mu - sqrtDisc;
  const far = -r * mu + sqrtDisc;
  if (far < 0) return -1;
  return near < 0 ? far : near;
}

/**
 * Bruneton's transmittance-LUT parameterisation, `(r, mu) -> (u, v)`.
 *
 * The mapping distributes texels along the *distance to the top of the
 * atmosphere* rather than along `mu`, which is what keeps the horizon — where
 * transmittance changes by two orders of magnitude across a fraction of a
 * degree — properly resolved at only 64 rows.
 */
export function transmittanceUvFromRMu(
  r: number,
  mu: number,
  bottomRadius: number,
  topRadius: number,
): { u: number; v: number } {
  const h = Math.sqrt(Math.max(0, topRadius * topRadius - bottomRadius * bottomRadius));
  const rho = Math.sqrt(Math.max(0, r * r - bottomRadius * bottomRadius));
  const d = raySphereDistance(r, mu, topRadius);
  const dMin = topRadius - r;
  const dMax = rho + h;
  const xMu = dMax - dMin > 0 ? (d - dMin) / (dMax - dMin) : 0;
  const xR = h > 0 ? rho / h : 0;
  return { u: THREE.MathUtils.clamp(xMu, 0, 1), v: THREE.MathUtils.clamp(xR, 0, 1) };
}

/** Exact inverse of {@link transmittanceUvFromRMu}. */
export function rMuFromTransmittanceUv(
  u: number,
  v: number,
  bottomRadius: number,
  topRadius: number,
): { r: number; mu: number } {
  const h = Math.sqrt(Math.max(0, topRadius * topRadius - bottomRadius * bottomRadius));
  const rho = h * v;
  const r = Math.sqrt(rho * rho + bottomRadius * bottomRadius);
  const dMin = topRadius - r;
  const dMax = rho + h;
  const d = dMin + u * (dMax - dMin);
  const mu = d === 0 ? 1 : (h * h - rho * rho - d * d) / (2 * r * d);
  return { r, mu: THREE.MathUtils.clamp(mu, -1, 1) };
}

/* -------------------------------------------------------------------------- */
/* Sampled medium                                                              */
/* -------------------------------------------------------------------------- */

/** Scattering and extinction coefficients at one altitude. */
interface MediumSample {
  /** Rayleigh scattering, 1/km, per channel. */
  rayleigh: [number, number, number];
  /** Mie scattering, 1/km (grey). */
  mie: number;
  /** Total extinction, 1/km, per channel. */
  extinction: [number, number, number];
}

function makeMediumSample(): MediumSample {
  return { rayleigh: [0, 0, 0], mie: 0, extinction: [0, 0, 0] };
}

/* -------------------------------------------------------------------------- */
/* Aerial perspective                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Knobs that vary with weather and art direction without invalidating the LUTs.
 *
 * `hazeDensity` scales the near-ground aerosol for the aerial-perspective
 * evaluation only. It multiplies scattering and extinction *together*, so it
 * behaves like genuinely denser haze (colour shifts toward the inscattered
 * radiance, contrast falls) rather than like the "add grey" fog that a naive
 * multiplier produces.
 */
export interface AerialPerspectiveSettings {
  /** Aerosol multiplier for the near-ground layer. 1 = clear, 2-3 = wet murk. */
  hazeDensity: number;
  /**
   * Extra grey extinction, 1/km, for genuine ground mist that is not part of
   * the atmospheric model (river fog, marsh vapour). Scatters the same source
   * function, so it still cannot disagree with the sky.
   */
  mistDensity: number;
  /** Altitude in km over which {@link mistDensity} falls to 1/e. */
  mistScaleHeightKm: number;
}

export const DEFAULT_AERIAL_PERSPECTIVE: AerialPerspectiveSettings = {
  hazeDensity: 1,
  mistDensity: 0,
  mistScaleHeightKm: 0.09,
};

/** Result of an aerial-perspective query along one view ray segment. */
export interface AerialPerspectiveSample {
  /** Per-channel transmittance from the camera to the sample point, `[0,1]`. */
  readonly transmittance: THREE.Vector3;
  /** Radiance added along the segment, in engine linear render units. */
  readonly inscatter: THREE.Vector3;
}

/**
 * The uniform block every GPU pass that wants fog must bind.
 *
 * These are `THREE.UniformNode`-backed values owned by the {@link Atmosphere}
 * instance and refreshed whenever the sun or weather changes, so a pass that
 * binds them once stays in sync forever. See
 * {@link AtmosphereService.aerialPerspectiveNode} for the matching evaluation.
 */
export interface AerialPerspectiveUniforms {
  /** Per-channel extinction, 1/km, at the observer's altitude. */
  readonly extinction: THREE.Vector3;
  /** Rayleigh scattering, 1/km, at the observer's altitude. */
  readonly rayleighScattering: THREE.Vector3;
  /** Mie scattering, 1/km, at the observer's altitude. */
  readonly mieScattering: number;
  /** Sun radiance already multiplied by transmittance to the observer. */
  readonly sunRadiance: THREE.Vector3;
  /** Multiple-scattering term at the observer's altitude. */
  readonly multiScatter: THREE.Vector3;
  /** Cornette-Shanks `g`. */
  readonly miePhaseG: number;
  /** World-space unit vector toward the sun. */
  readonly sunDirection: THREE.Vector3;
}

/* -------------------------------------------------------------------------- */
/* Service contract                                                            */
/* -------------------------------------------------------------------------- */

/**
 * What other render modules resolve from the {@link ServiceLocator}.
 *
 * There is exactly one of these. Any module that wants fog, sky colour, or the
 * sun's radiance asks here; nobody is allowed a second, private fog model. That
 * is the entire point — a `FogExp2` sitting next to a scattering sky is how
 * distant geometry ends up a different colour from the sky behind it.
 */
export interface AtmosphereService {
  /** The static medium description. */
  readonly params: AtmosphereParams;
  /** Mutable weather knobs. Call {@link refresh} after changing them. */
  readonly aerial: AerialPerspectiveSettings;
  /** Live uniform values for GPU passes; do not mutate. */
  readonly uniforms: AerialPerspectiveUniforms;

  /** World-space unit vector toward the sun (apparent, refraction included). */
  readonly sunDirection: THREE.Vector3;
  /** Observer altitude above sea level, in metres. */
  readonly observerAltitude: number;

  /**
   * Sky radiance arriving from `direction` (world space, unit length), with no
   * sun disc. This is the function the sky-view texture is a discretisation of.
   */
  skyRadiance(direction: THREE.Vector3, target?: THREE.Vector3): THREE.Vector3;

  /**
   * Transmittance from the observer to the top of the atmosphere toward the
   * sun. Multiply by {@link solarIlluminance} to get the sunlight actually
   * reaching the observer.
   */
  sunTransmittance(target?: THREE.Vector3): THREE.Vector3;

  /** Top-of-atmosphere solar irradiance in engine render units. */
  readonly solarIlluminance: THREE.Vector3;

  /**
   * Extinction and in-scattering between the observer and a point `distance`
   * metres away along `direction`.
   *
   * Composite as `final = surfaceColour * transmittance + inscatter`.
   */
  aerialPerspective(
    direction: THREE.Vector3,
    distance: number,
    target?: AerialPerspectiveSample,
  ): AerialPerspectiveSample;

  /**
   * Distance in metres at which transmittance falls to `threshold` for a
   * horizontal view ray. Useful for picking a camera far plane or an LOD fade
   * that is guaranteed to be invisible.
   */
  visibilityRange(threshold?: number): number;

  /** Recompute cached observer-dependent terms. Cheap; call after any change. */
  refresh(): void;
}

/** Service key. Registered by {@link module:render/Sky}. */
export const AtmosphereKey = serviceKey<AtmosphereService>('render.atmosphere');

/* -------------------------------------------------------------------------- */
/* LUT dimensions                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Hillaire uses 256x64 for transmittance and 32x32 for multiple scattering.
 * Both are kept: the transmittance table is the accuracy-critical one (it is
 * sampled in the innermost loop of everything else) and 32x32 is genuinely
 * enough for `Psi_ms`, which is a very smooth function of both axes.
 */
const TRANSMITTANCE_W = 256;
const TRANSMITTANCE_H = 64;
const TRANSMITTANCE_STEPS = 40;

const MULTISCATTER_SIZE = 32;
/** Directions on the sphere for the `f_ms` / `L_2nd` integral. */
const MULTISCATTER_DIRECTIONS = 32;
const MULTISCATTER_STEPS = 20;

/** Golden-angle increment for spherical Fibonacci direction sampling. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/* -------------------------------------------------------------------------- */
/* Atmosphere                                                                  */
/* -------------------------------------------------------------------------- */

export interface AtmosphereOptions {
  params?: Partial<AtmosphereParams>;
  /** Top-of-atmosphere solar irradiance in render units. Default 22. */
  solarIlluminance?: number | THREE.Vector3;
  /** Observer altitude above sea level in metres. Default 2. */
  observerAltitude?: number;
  /** Ray-march steps for {@link Atmosphere.skyRadiance}. Default 24. */
  skySteps?: number;
}

/**
 * The scattering model. Owns both precomputed tables and answers every query
 * the rest of the renderer makes about air.
 *
 * Construction cost is dominated by the two LUTs; see the module header and the
 * budget notes in {@link module:render/Sky}.
 */
export class Atmosphere implements AtmosphereService {
  readonly params: AtmosphereParams;
  readonly aerial: AerialPerspectiveSettings = { ...DEFAULT_AERIAL_PERSPECTIVE };
  readonly solarIlluminance: THREE.Vector3;
  readonly sunDirection = new THREE.Vector3(0, 1, 0);

  readonly uniforms: {
    extinction: THREE.Vector3;
    rayleighScattering: THREE.Vector3;
    mieScattering: number;
    sunRadiance: THREE.Vector3;
    multiScatter: THREE.Vector3;
    miePhaseG: number;
    sunDirection: THREE.Vector3;
  };

  /** `TRANSMITTANCE_W * TRANSMITTANCE_H * 3` floats, row-major, v-major. */
  readonly #transmittance: Float32Array;
  /** `MULTISCATTER_SIZE^2 * 3` floats. */
  readonly #multiScatter: Float32Array;

  #observerAltitudeKm: number;
  readonly #skySteps: number;

  /* Scratch — every hot path is allocation-free. */
  readonly #medium = makeMediumSample();
  readonly #mediumB = makeMediumSample();
  readonly #tmpA = new THREE.Vector3();
  readonly #scratchSample: { transmittance: THREE.Vector3; inscatter: THREE.Vector3 } = {
    transmittance: new THREE.Vector3(1, 1, 1),
    inscatter: new THREE.Vector3(),
  };

  constructor(options: AtmosphereOptions = {}) {
    this.params = { ...EARTH_ATMOSPHERE, ...options.params };
    this.#observerAltitudeKm = (options.observerAltitude ?? 2) / 1000;
    this.#skySteps = options.skySteps ?? 24;

    const illuminance = options.solarIlluminance ?? 22;
    this.solarIlluminance =
      illuminance instanceof THREE.Vector3
        ? illuminance.clone()
        : new THREE.Vector3(illuminance, illuminance, illuminance);

    this.#transmittance = new Float32Array(TRANSMITTANCE_W * TRANSMITTANCE_H * 3);
    this.#multiScatter = new Float32Array(MULTISCATTER_SIZE * MULTISCATTER_SIZE * 3);

    this.#buildTransmittanceLut();
    this.#buildMultiScatterLut();

    this.uniforms = {
      extinction: new THREE.Vector3(),
      rayleighScattering: new THREE.Vector3(),
      mieScattering: 0,
      sunRadiance: new THREE.Vector3(),
      multiScatter: new THREE.Vector3(),
      miePhaseG: this.params.miePhaseG,
      sunDirection: new THREE.Vector3(0, 1, 0),
    };
    this.refresh();
  }

  /** Observer altitude above sea level, metres. */
  get observerAltitude(): number {
    return this.#observerAltitudeKm * 1000;
  }

  set observerAltitude(metres: number) {
    // Clamped just above the surface: a ray origin exactly on (or below) the
    // planet makes `raySphereDistance` return the wrong root and the whole sky
    // goes black, which is a spectacularly confusing failure to debug.
    this.#observerAltitudeKm = Math.max(0.0005, metres / 1000);
  }

  /** Set the sun direction. Normalises in place; refresh is automatic. */
  setSunDirection(direction: THREE.Vector3): void {
    this.sunDirection.copy(direction).normalize();
    this.refresh();
  }

  /* ---------------------------------------------------------------------- */
  /* Medium                                                                  */
  /* ---------------------------------------------------------------------- */

  /** Sample the medium at altitude `h` km into `out`. */
  #sampleMedium(h: number, out: MediumSample): void {
    const p = this.params;
    const rayleighDensity = Math.exp(-h / p.rayleighScaleHeightKm);
    const mieDensity = Math.exp(-h / p.mieScaleHeightKm);
    // Tent profile: 1 at the centre, 0 at centre +/- width. Bruneton models this
    // as two clipped linear segments; the tent is the same function written in
    // a form that is obviously non-negative.
    const ozoneDensity = Math.max(0, 1 - Math.abs(h - p.ozoneCenterKm) / p.ozoneWidthKm);

    const r = out.rayleigh;
    r[0] = p.rayleighScattering[0] * rayleighDensity;
    r[1] = p.rayleighScattering[1] * rayleighDensity;
    r[2] = p.rayleighScattering[2] * rayleighDensity;

    out.mie = p.mieScattering * mieDensity;

    const mieExt = p.mieExtinction * mieDensity;
    const e = out.extinction;
    e[0] = r[0] + mieExt + p.ozoneAbsorption[0] * ozoneDensity;
    e[1] = r[1] + mieExt + p.ozoneAbsorption[1] * ozoneDensity;
    e[2] = r[2] + mieExt + p.ozoneAbsorption[2] * ozoneDensity;
  }

  /* ---------------------------------------------------------------------- */
  /* Transmittance LUT                                                       */
  /* ---------------------------------------------------------------------- */

  #buildTransmittanceLut(): void {
    const { bottomRadiusKm, topRadiusKm } = this.params;
    const medium = this.#medium;
    const lut = this.#transmittance;

    for (let j = 0; j < TRANSMITTANCE_H; j++) {
      const v = (j + 0.5) / TRANSMITTANCE_H;
      for (let i = 0; i < TRANSMITTANCE_W; i++) {
        const u = (i + 0.5) / TRANSMITTANCE_W;
        const { r, mu } = rMuFromTransmittanceUv(u, v, bottomRadiusKm, topRadiusKm);

        // Distance to the top of the atmosphere. `mu` from the inverse mapping
        // never points below the horizon by construction, so the top-sphere
        // root always exists.
        const tMax = Math.max(0, raySphereDistance(r, mu, topRadiusKm));
        const dt = tMax / TRANSMITTANCE_STEPS;

        let odR = 0;
        let odG = 0;
        let odB = 0;
        for (let s = 0; s < TRANSMITTANCE_STEPS; s++) {
          const t = (s + 0.5) * dt;
          // Law of cosines: radius at parameter t along a ray with cosine mu.
          const rt = Math.sqrt(Math.max(0, t * t + 2 * r * mu * t + r * r));
          this.#sampleMedium(rt - bottomRadiusKm, medium);
          odR += medium.extinction[0] * dt;
          odG += medium.extinction[1] * dt;
          odB += medium.extinction[2] * dt;
        }

        const o = (j * TRANSMITTANCE_W + i) * 3;
        lut[o] = Math.exp(-odR);
        lut[o + 1] = Math.exp(-odG);
        lut[o + 2] = Math.exp(-odB);
      }
    }
  }

  /**
   * Bilinear fetch from the transmittance LUT.
   *
   * `mu` below the horizon is *not* clamped to the table: a ray that enters the
   * planet has zero transmittance to the sun, and clamping instead would light
   * the ground from below at sunset.
   */
  #transmittanceToTop(r: number, mu: number, out: [number, number, number]): void {
    const { bottomRadiusKm, topRadiusKm } = this.params;
    if (raySphereDistance(r, mu, bottomRadiusKm) >= 0) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      return;
    }

    const { u, v } = transmittanceUvFromRMu(r, mu, bottomRadiusKm, topRadiusKm);
    const x = u * TRANSMITTANCE_W - 0.5;
    const y = v * TRANSMITTANCE_H - 0.5;
    const x0 = THREE.MathUtils.clamp(Math.floor(x), 0, TRANSMITTANCE_W - 1);
    const y0 = THREE.MathUtils.clamp(Math.floor(y), 0, TRANSMITTANCE_H - 1);
    const x1 = Math.min(x0 + 1, TRANSMITTANCE_W - 1);
    const y1 = Math.min(y0 + 1, TRANSMITTANCE_H - 1);
    const fx = THREE.MathUtils.clamp(x - x0, 0, 1);
    const fy = THREE.MathUtils.clamp(y - y0, 0, 1);

    const lut = this.#transmittance;
    const i00 = (y0 * TRANSMITTANCE_W + x0) * 3;
    const i10 = (y0 * TRANSMITTANCE_W + x1) * 3;
    const i01 = (y1 * TRANSMITTANCE_W + x0) * 3;
    const i11 = (y1 * TRANSMITTANCE_W + x1) * 3;

    for (let c = 0; c < 3; c++) {
      const a = (lut[i00 + c] as number) * (1 - fx) + (lut[i10 + c] as number) * fx;
      const b = (lut[i01 + c] as number) * (1 - fx) + (lut[i11 + c] as number) * fx;
      out[c] = a * (1 - fy) + b * fy;
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Multiple-scattering LUT                                                 */
  /* ---------------------------------------------------------------------- */

  /**
   * Hillaire §5.
   *
   * For each `(altitude, sun zenith)` cell, integrate over a uniformly sampled
   * sphere of view directions:
   *
   * - `L_2nd` — radiance from light that scattered exactly twice, assuming an
   *   isotropic phase function for the second event.
   * - `f_ms`  — the fraction of light that scatters *again* per bounce.
   *
   * Then close the series: `Psi_ms = L_2nd * (1 / (1 - f_ms))`. That single
   * division is what a Bruneton-style iterative solve is being replaced by, and
   * the error is small precisely because higher orders of scattering are close
   * to isotropic.
   *
   * Directions are spherical Fibonacci rather than Hillaire's 8x8 `sqrt`
   * lattice: it has lower discrepancy for the same count, which matters at 32
   * samples where the lattice still shows banding near the poles.
   */
  #buildMultiScatterLut(): void {
    const { bottomRadiusKm, topRadiusKm, groundAlbedo } = this.params;
    const medium = this.#medium;
    const sunT: [number, number, number] = [0, 0, 0];

    for (let j = 0; j < MULTISCATTER_SIZE; j++) {
      const vAlt = (j + 0.5) / MULTISCATTER_SIZE;
      const r = bottomRadiusKm + vAlt * (topRadiusKm - bottomRadiusKm);

      for (let i = 0; i < MULTISCATTER_SIZE; i++) {
        const uSun = (i + 0.5) / MULTISCATTER_SIZE;
        const muS = uSun * 2 - 1;
        // Sun direction in the local frame where +Z is up.
        const sunZ = muS;
        const sunX = Math.sqrt(Math.max(0, 1 - muS * muS));

        let l2r = 0;
        let l2g = 0;
        let l2b = 0;
        let fmsR = 0;
        let fmsG = 0;
        let fmsB = 0;

        for (let d = 0; d < MULTISCATTER_DIRECTIONS; d++) {
          // Spherical Fibonacci: uniform in cos(theta), golden-angle azimuth.
          const cosTheta = 1 - (2 * (d + 0.5)) / MULTISCATTER_DIRECTIONS;
          const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
          const phi = d * GOLDEN_ANGLE;
          const dirX = sinTheta * Math.cos(phi);
          const dirY = sinTheta * Math.sin(phi);
          const dirZ = cosTheta;

          const hitsGround = raySphereDistance(r, dirZ, bottomRadiusKm) >= 0;
          const tMax = hitsGround
            ? raySphereDistance(r, dirZ, bottomRadiusKm)
            : raySphereDistance(r, dirZ, topRadiusKm);
          if (tMax <= 0) continue;
          const dt = tMax / MULTISCATTER_STEPS;

          let tR = 1;
          let tG = 1;
          let tB = 1;
          let sR = 0;
          let sG = 0;
          let sB = 0;
          let fR = 0;
          let fG = 0;
          let fB = 0;

          for (let s = 0; s < MULTISCATTER_STEPS; s++) {
            const t = (s + 0.5) * dt;
            const rt = Math.sqrt(Math.max(0, t * t + 2 * r * dirZ * t + r * r));
            this.#sampleMedium(rt - bottomRadiusKm, medium);

            // Sun cosine at the sample point. The sample sits at
            // (r*0 + t*dirX, t*dirY, r + t*dirZ) in the local frame; its up
            // vector is that position normalised.
            // The sun lies in the local XZ plane (sunY = 0 by construction),
            // so the Y component of the sample position cannot contribute.
            const px = t * dirX;
            const pz = r + t * dirZ;
            const muSample = (px * sunX + pz * sunZ) / Math.max(1e-6, rt);

            this.#transmittanceToTop(rt, muSample, sunT);

            const sctR = medium.rayleigh[0] + medium.mie;
            const sctG = medium.rayleigh[1] + medium.mie;
            const sctB = medium.rayleigh[2] + medium.mie;

            // Analytic in-scatter integration over the segment (Hillaire §4):
            // exact for a piecewise-constant medium, unlike a midpoint sum.
            const segTR = Math.exp(-medium.extinction[0] * dt);
            const segTG = Math.exp(-medium.extinction[1] * dt);
            const segTB = Math.exp(-medium.extinction[2] * dt);
            const invER = 1 / Math.max(1e-9, medium.extinction[0]);
            const invEG = 1 / Math.max(1e-9, medium.extinction[1]);
            const invEB = 1 / Math.max(1e-9, medium.extinction[2]);

            // Second-order source: sunlight, scattered once here, isotropically.
            const srcR = sctR * sunT[0] * UNIFORM_PHASE;
            const srcG = sctG * sunT[1] * UNIFORM_PHASE;
            const srcB = sctB * sunT[2] * UNIFORM_PHASE;
            sR += tR * (srcR - srcR * segTR) * invER;
            sG += tG * (srcG - srcG * segTG) * invEG;
            sB += tB * (srcB - srcB * segTB) * invEB;

            // Transfer function: throughput of one further isotropic bounce.
            fR += tR * (sctR - sctR * segTR) * invER;
            fG += tG * (sctG - sctG * segTG) * invEG;
            fB += tB * (sctB - sctB * segTB) * invEB;

            tR *= segTR;
            tG *= segTG;
            tB *= segTB;
          }

          if (hitsGround) {
            // Lambertian ground bounce, folded into the same series.
            const groundR = bottomRadiusKm;
            const px = tMax * dirX;
            const py = tMax * dirY;
            const pz = r + tMax * dirZ;
            const inv = 1 / Math.max(1e-6, Math.sqrt(px * px + py * py + pz * pz));
            const nDotL = Math.max(0, (px * sunX + pz * sunZ) * inv);
            this.#transmittanceToTop(groundR, (px * sunX + pz * sunZ) * inv, sunT);
            sR += tR * groundAlbedo[0] * nDotL * sunT[0] / Math.PI;
            sG += tG * groundAlbedo[1] * nDotL * sunT[1] / Math.PI;
            sB += tB * groundAlbedo[2] * nDotL * sunT[2] / Math.PI;
          }

          l2r += sR;
          l2g += sG;
          l2b += sB;
          fmsR += fR;
          fmsG += fG;
          fmsB += fB;
        }

        // Sphere integral weight 4pi/N times the isotropic phase 1/4pi = 1/N.
        const inv = 1 / MULTISCATTER_DIRECTIONS;
        l2r *= inv;
        l2g *= inv;
        l2b *= inv;
        fmsR *= inv;
        fmsG *= inv;
        fmsB *= inv;

        const o = (j * MULTISCATTER_SIZE + i) * 3;
        this.#multiScatter[o] = l2r / Math.max(1e-6, 1 - fmsR);
        this.#multiScatter[o + 1] = l2g / Math.max(1e-6, 1 - fmsG);
        this.#multiScatter[o + 2] = l2b / Math.max(1e-6, 1 - fmsB);
      }
    }
  }

  /** Bilinear fetch of `Psi_ms` at radius `r` and sun cosine `muS`. */
  #sampleMultiScatter(r: number, muS: number, out: [number, number, number]): void {
    const { bottomRadiusKm, topRadiusKm } = this.params;
    const u = THREE.MathUtils.clamp(muS * 0.5 + 0.5, 0, 1);
    const v = THREE.MathUtils.clamp(
      (r - bottomRadiusKm) / (topRadiusKm - bottomRadiusKm),
      0,
      1,
    );

    const x = u * MULTISCATTER_SIZE - 0.5;
    const y = v * MULTISCATTER_SIZE - 0.5;
    const x0 = THREE.MathUtils.clamp(Math.floor(x), 0, MULTISCATTER_SIZE - 1);
    const y0 = THREE.MathUtils.clamp(Math.floor(y), 0, MULTISCATTER_SIZE - 1);
    const x1 = Math.min(x0 + 1, MULTISCATTER_SIZE - 1);
    const y1 = Math.min(y0 + 1, MULTISCATTER_SIZE - 1);
    const fx = THREE.MathUtils.clamp(x - x0, 0, 1);
    const fy = THREE.MathUtils.clamp(y - y0, 0, 1);

    const lut = this.#multiScatter;
    const i00 = (y0 * MULTISCATTER_SIZE + x0) * 3;
    const i10 = (y0 * MULTISCATTER_SIZE + x1) * 3;
    const i01 = (y1 * MULTISCATTER_SIZE + x0) * 3;
    const i11 = (y1 * MULTISCATTER_SIZE + x1) * 3;

    for (let c = 0; c < 3; c++) {
      const a = (lut[i00 + c] as number) * (1 - fx) + (lut[i10 + c] as number) * fx;
      const b = (lut[i01 + c] as number) * (1 - fx) + (lut[i11 + c] as number) * fx;
      out[c] = a * (1 - fy) + b * fy;
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Queries                                                                 */
  /* ---------------------------------------------------------------------- */

  /**
   * Full single + multiple scattering ray march from the observer along
   * `direction`. No sun disc: the disc is added analytically by the sky shader,
   * where it can be resolved at full screen resolution instead of at LUT
   * resolution.
   */
  skyRadiance(direction: THREE.Vector3, target?: THREE.Vector3): THREE.Vector3 {
    const out = target ?? new THREE.Vector3();
    const { bottomRadiusKm, topRadiusKm, groundAlbedo, miePhaseG } = this.params;

    const r0 = bottomRadiusKm + this.#observerAltitudeKm;
    const mu = direction.y; // world +Y is up, so this is cos(view zenith).
    const muSunView = direction.dot(this.sunDirection);

    const groundT = raySphereDistance(r0, mu, bottomRadiusKm);
    const hitsGround = groundT >= 0;
    const tMax = hitsGround ? groundT : raySphereDistance(r0, mu, topRadiusKm);
    if (tMax <= 0) return out.set(0, 0, 0);

    const steps = this.#skySteps;
    const dt = tMax / steps;

    const phaseR = rayleighPhase(muSunView);
    const phaseM = cornetteShanksPhase(muSunView, miePhaseG);

    const medium = this.#medium;
    const sunT: [number, number, number] = [0, 0, 0];
    const ms: [number, number, number] = [0, 0, 0];

    // Sun direction expressed in the ray's own frame: the sun cosine at a
    // sample depends on how the local up vector has rotated along the ray.
    const sun = this.sunDirection;
    let tr = 1;
    let tg = 1;
    let tb = 1;
    let lr = 0;
    let lg = 0;
    let lb = 0;

    for (let s = 0; s < steps; s++) {
      const t = (s + 0.5) * dt;
      const rt = Math.sqrt(Math.max(0, t * t + 2 * r0 * mu * t + r0 * r0));

      // Sample position in world-aligned coordinates with the observer on the
      // +Y axis at radius r0.
      const px = direction.x * t;
      const py = r0 + direction.y * t;
      const pz = direction.z * t;
      const invR = 1 / Math.max(1e-6, rt);
      const muSample = (px * sun.x + py * sun.y + pz * sun.z) * invR;

      this.#sampleMedium(rt - bottomRadiusKm, medium);
      this.#transmittanceToTop(rt, muSample, sunT);
      this.#sampleMultiScatter(rt, muSample, ms);

      // Planet shadow: a sample can be above the horizon relative to its own
      // local up and still be in the planet's umbra.
      const shadow = raySphereDistance(rt, muSample, bottomRadiusKm) >= 0 ? 0 : 1;

      const e = medium.extinction;
      const segR = Math.exp(-e[0] * dt);
      const segG = Math.exp(-e[1] * dt);
      const segB = Math.exp(-e[2] * dt);

      const scR = medium.rayleigh[0] + medium.mie;
      const scG = medium.rayleigh[1] + medium.mie;
      const scB = medium.rayleigh[2] + medium.mie;

      // Single scattering uses the real phase functions; multiple scattering is
      // isotropic by construction (Hillaire's approximation) and so multiplies
      // total scattering without a phase term.
      const srcR =
        (medium.rayleigh[0] * phaseR + medium.mie * phaseM) * sunT[0] * shadow + scR * ms[0];
      const srcG =
        (medium.rayleigh[1] * phaseR + medium.mie * phaseM) * sunT[1] * shadow + scG * ms[1];
      const srcB =
        (medium.rayleigh[2] * phaseR + medium.mie * phaseM) * sunT[2] * shadow + scB * ms[2];

      lr += tr * (srcR - srcR * segR) / Math.max(1e-9, e[0]);
      lg += tg * (srcG - srcG * segG) / Math.max(1e-9, e[1]);
      lb += tb * (srcB - srcB * segB) / Math.max(1e-9, e[2]);

      tr *= segR;
      tg *= segG;
      tb *= segB;
    }

    if (hitsGround) {
      const px = direction.x * tMax;
      const py = r0 + direction.y * tMax;
      const pz = direction.z * tMax;
      const inv = 1 / Math.max(1e-6, Math.sqrt(px * px + py * py + pz * pz));
      const nDotL = (px * sun.x + py * sun.y + pz * sun.z) * inv;
      this.#transmittanceToTop(bottomRadiusKm, nDotL, sunT);
      const lambert = Math.max(0, nDotL) / Math.PI;
      this.#sampleMultiScatter(bottomRadiusKm, nDotL, ms);
      lr += tr * groundAlbedo[0] * (lambert * sunT[0] + ms[0]);
      lg += tg * groundAlbedo[1] * (lambert * sunT[1] + ms[1]);
      lb += tb * groundAlbedo[2] * (lambert * sunT[2] + ms[2]);
    }

    return out.set(
      lr * this.solarIlluminance.x,
      lg * this.solarIlluminance.y,
      lb * this.solarIlluminance.z,
    );
  }

  sunTransmittance(target?: THREE.Vector3): THREE.Vector3 {
    const out = target ?? new THREE.Vector3();
    const r0 = this.params.bottomRadiusKm + this.#observerAltitudeKm;
    const t: [number, number, number] = [0, 0, 0];
    this.#transmittanceToTop(r0, this.sunDirection.y, t);
    return out.set(t[0], t[1], t[2]);
  }

  /**
   * Aerial perspective for a segment of length `distance` metres.
   *
   * Modelled as a homogeneous slab at the observer's altitude, integrated
   * analytically:
   *
   * ```
   * T      = exp(-sigma_t * d)
   * L_in   = (S / sigma_t) * (1 - T)
   * S      = (sigma_R*P_R(mu_s) + sigma_M*P_M(mu_s)) * T_sun * E_sun
   *        + (sigma_R + sigma_M) * Psi_ms * E_sun
   * ```
   *
   * `L_in = S/sigma_t * (1 - T)` is the exact solution of the radiative
   * transfer equation for constant `S` and `sigma_t`, so it is correct, not a
   * fit — the only approximation is holding the medium constant over the
   * segment. Over the few kilometres a third-person camera can see, and with
   * less than a kilometre of altitude change, the error against the full march
   * is under a percent (see `tests/atmosphere.test.ts`).
   *
   * Critically, `S` here is built from the *same* coefficients, the *same*
   * `T_sun` and the *same* `Psi_ms` as {@link skyRadiance}. Fog at the horizon
   * therefore converges exactly onto the sky behind it, which is the property a
   * separate `FogExp2` can never have.
   */
  aerialPerspective(
    direction: THREE.Vector3,
    distance: number,
    target?: AerialPerspectiveSample,
  ): AerialPerspectiveSample {
    const out = (target ?? this.#scratchSample) as {
      transmittance: THREE.Vector3;
      inscatter: THREE.Vector3;
    };

    const dKm = Math.max(0, distance) / 1000;
    if (dKm <= 0) {
      out.transmittance.set(1, 1, 1);
      out.inscatter.set(0, 0, 0);
      return out;
    }

    const { miePhaseG, bottomRadiusKm } = this.params;
    const haze = Math.max(0, this.aerial.hazeDensity);
    const medium = this.#medium;

    // Evaluate the medium at the midpoint altitude of the segment rather than
    // at the camera: for a ray climbing toward a mountain top this is a much
    // better constant than either endpoint, at zero extra cost.
    const midAltitude = this.#observerAltitudeKm + direction.y * dKm * 0.5;
    this.#sampleMedium(Math.max(0, midAltitude), medium);

    // Extra ground mist, exponential in altitude. Grey, conservative-ish
    // scattering: it both extinguishes and inscatters, so it desaturates
    // distance instead of tinting it.
    const mist =
      this.aerial.mistDensity > 0
        ? this.aerial.mistDensity *
          Math.exp(-Math.max(0, this.#observerAltitudeKm) / this.aerial.mistScaleHeightKm)
        : 0;

    const scR = medium.rayleigh[0] + medium.mie * haze + mist;
    const scG = medium.rayleigh[1] + medium.mie * haze + mist;
    const scB = medium.rayleigh[2] + medium.mie * haze + mist;
    const extR = medium.extinction[0] + medium.mie * (haze - 1) + mist;
    const extG = medium.extinction[1] + medium.mie * (haze - 1) + mist;
    const extB = medium.extinction[2] + medium.mie * (haze - 1) + mist;

    const muS = direction.dot(this.sunDirection);
    const phaseR = rayleighPhase(muS);
    const phaseM = cornetteShanksPhase(muS, miePhaseG);

    const r0 = bottomRadiusKm + this.#observerAltitudeKm;
    const sunT: [number, number, number] = [0, 0, 0];
    const ms: [number, number, number] = [0, 0, 0];
    this.#transmittanceToTop(r0, this.sunDirection.y, sunT);
    this.#sampleMultiScatter(r0, this.sunDirection.y, ms);

    const eR = Math.max(1e-9, extR);
    const eG = Math.max(1e-9, extG);
    const eB = Math.max(1e-9, extB);

    const tR = Math.exp(-eR * dKm);
    const tG = Math.exp(-eG * dKm);
    const tB = Math.exp(-eB * dKm);

    // Mist is isotropic; Rayleigh and Mie carry their phase functions.
    const srcR =
      ((medium.rayleigh[0] * phaseR + medium.mie * haze * phaseM + mist * UNIFORM_PHASE) *
        sunT[0] +
        scR * ms[0]) *
      this.solarIlluminance.x;
    const srcG =
      ((medium.rayleigh[1] * phaseR + medium.mie * haze * phaseM + mist * UNIFORM_PHASE) *
        sunT[1] +
        scG * ms[1]) *
      this.solarIlluminance.y;
    const srcB =
      ((medium.rayleigh[2] * phaseR + medium.mie * haze * phaseM + mist * UNIFORM_PHASE) *
        sunT[2] +
        scB * ms[2]) *
      this.solarIlluminance.z;

    out.transmittance.set(tR, tG, tB);
    out.inscatter.set(
      (srcR / eR) * (1 - tR),
      (srcG / eG) * (1 - tG),
      (srcB / eB) * (1 - tB),
    );
    return out;
  }

  visibilityRange(threshold = 0.02): number {
    const horizon = this.#tmpA.set(1, 0, 0);
    const sample = this.aerialPerspective(horizon, 1000, this.#scratchSample);
    // Transmittance is exponential, so one probe at 1 km gives the coefficient.
    const perKm = -Math.log(Math.max(1e-6, sample.transmittance.y));
    return (Math.log(1 / Math.max(1e-6, threshold)) / Math.max(1e-6, perKm)) * 1000;
  }

  /**
   * Refresh the observer-dependent uniform block.
   *
   * Everything here depends only on the sun direction, the observer altitude
   * and the weather knobs, so a GPU pass can bind {@link uniforms} once and
   * never think about it again.
   */
  refresh(): void {
    const r0 = this.params.bottomRadiusKm + this.#observerAltitudeKm;
    const medium = this.#mediumB;
    this.#sampleMedium(Math.max(0, this.#observerAltitudeKm), medium);

    const haze = Math.max(0, this.aerial.hazeDensity);
    const mist =
      this.aerial.mistDensity > 0
        ? this.aerial.mistDensity *
          Math.exp(-Math.max(0, this.#observerAltitudeKm) / this.aerial.mistScaleHeightKm)
        : 0;

    const sunT: [number, number, number] = [0, 0, 0];
    const ms: [number, number, number] = [0, 0, 0];
    this.#transmittanceToTop(r0, this.sunDirection.y, sunT);
    this.#sampleMultiScatter(r0, this.sunDirection.y, ms);

    const u = this.uniforms;
    u.rayleighScattering.set(
      medium.rayleigh[0],
      medium.rayleigh[1],
      medium.rayleigh[2],
    );
    u.mieScattering = medium.mie * haze;
    u.extinction.set(
      medium.extinction[0] + medium.mie * (haze - 1) + mist,
      medium.extinction[1] + medium.mie * (haze - 1) + mist,
      medium.extinction[2] + medium.mie * (haze - 1) + mist,
    );
    u.sunRadiance.set(
      sunT[0] * this.solarIlluminance.x,
      sunT[1] * this.solarIlluminance.y,
      sunT[2] * this.solarIlluminance.z,
    );
    u.multiScatter.set(
      ms[0] * this.solarIlluminance.x,
      ms[1] * this.solarIlluminance.y,
      ms[2] * this.solarIlluminance.z,
    );
    u.miePhaseG = this.params.miePhaseG;
    u.sunDirection.copy(this.sunDirection);
  }
}
