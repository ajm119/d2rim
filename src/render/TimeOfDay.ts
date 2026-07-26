/**
 * @module render/TimeOfDay
 *
 * A driveable 0-24 h clock that owns where the sun and moon actually are, and
 * what mood the world is in at that hour.
 *
 * ### Why a real solar model
 *
 * It would be shorter to lerp the sun around a fixed arc. It would also be
 * wrong in a way players feel: the arc's tilt, how long twilight lasts, and how
 * far north of west the sun sets are all functions of latitude and season, and
 * they are exactly what makes an outdoor scene read as *a place* rather than as
 * a lighting rig. So the elevation/azimuth come from the standard solar
 * position equations:
 *
 * - Declination from Spencer's Fourier fit (J. W. Spencer, *"Fourier series
 *   representation of the position of the sun"*, Search 2(5), 1971). Accurate
 *   to about 0.01 rad, which is far below what anyone can see in a sky.
 * - Hour angle, elevation and azimuth from the standard spherical-astronomy
 *   relations as given by Michalsky (*"The Astronomical Almanac's algorithm for
 *   approximate solar position"*, Solar Energy 40(3), 1988).
 * - Atmospheric refraction from Bennett (*"The calculation of astronomical
 *   refraction in marine navigation"*, Journal of Navigation 35, 1982). This is
 *   the reason a setting sun is still fully visible when it is geometrically
 *   already below the horizon — about 0.57 deg of lift, slightly more than the
 *   sun's own diameter.
 *
 * The equation of time and the observer's longitude are deliberately *not*
 * modelled: `hours` is local apparent solar time by definition, because a game
 * clock that disagrees with the sun by 15 minutes is a bug report, not realism.
 *
 * ### World convention
 *
 * Right-handed, `+Y` up, **`+X` east and `-Z` north** (so `+Z` is south, which
 * is where the noon sun sits at northern latitudes). {@link northOffsetDeg}
 * rotates the compass if the level's art was laid out against a different one.
 *
 * ### Moon
 *
 * First-order: the moon rides the ecliptic opposite the sun, offset in hour
 * angle by the phase, with declination interpolated between `+delta_sun` at new
 * moon and `-delta_sun` at full. The 5.1 deg lunar orbital inclination and the
 * 27.3 d nodal drift are not modelled. The visible consequence of the model
 * that *is* here — a full moon rides high in winter and low in summer, and
 * rises as the sun sets — is the part that matters for lighting.
 *
 * ### Coordination
 *
 * This is a {@link GameModule}, but {@link module:render/Sky} will create and
 * register one if the integrator has not. {@link advance} is idempotent per
 * engine frame, so being driven by both is harmless.
 */

import * as THREE from 'three/webgpu';

import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** The clock moved. Fires at most once per frame, and only on real change. */
    'timeofday:changed': {
      hours: number;
      previousHours: number;
      /** True when the change crossed midnight. */
      dayRolled: boolean;
    };
    /** A named mood was applied. Weather-driving modules should listen. */
    'timeofday:preset': { preset: TimeOfDayPresetName; mood: TimeOfDayMood };
  }
}

/* -------------------------------------------------------------------------- */
/* Public types                                                                */
/* -------------------------------------------------------------------------- */

/** Where a celestial body is, and how much of it is actually up. */
export interface CelestialBody {
  /** World-space unit vector from the observer toward the body (refracted). */
  readonly direction: THREE.Vector3;
  /** Apparent elevation above the horizon, degrees. Refraction included. */
  readonly elevationDeg: number;
  /** Compass azimuth, degrees clockwise from north. */
  readonly azimuthDeg: number;
  /**
   * `0` fully set, `1` fully risen — a smooth ramp across the disc's own
   * angular diameter plus a degree of slop, so lights fade rather than pop.
   */
  readonly visibility: number;
}

/**
 * The art-direction half of a preset: everything about the *weather*, as
 * opposed to the astronomy.
 *
 * {@link module:render/Sky} consumes this. It lives here so that "dusk" is one
 * concept — an hour *and* the sky that belongs to it — instead of two settings
 * a level designer has to remember to change together.
 */
export interface TimeOfDayMood {
  /** Fractional sky covered by the cloud deck, `[0,1]`. */
  cloudCoverage: number;
  /**
   * Optical depth of the deck. Real stratus is 10-60; cumulus cores reach 100.
   * Drives both how dark the base reads and how much sun gets through.
   */
  cloudOpticalDepth: number;
  /** Base of the cloud deck above the observer, metres. */
  cloudBaseAltitude: number;
  /** Vertical extent of the deck, metres. */
  cloudThickness: number;
  /** Aerosol multiplier for aerial perspective. 1 clear, 2-3 wet murk. */
  hazeDensity: number;
  /** Extra grey ground-mist extinction, 1/km. */
  mistDensity: number;
  /** Multiplier on the whole sky's radiance. Art escape hatch; keep near 1. */
  skyIntensity: number;
}

/** The named moods the game ships with. */
export type TimeOfDayPresetName = 'bloodMoor' | 'dawn' | 'clearNoon' | 'dusk' | 'night' | 'storm';

export interface TimeOfDayPreset {
  readonly hours: number;
  readonly mood: TimeOfDayMood;
  /** Optional astronomy override; falls back to the current configuration. */
  readonly latitudeDeg?: number;
  readonly dayOfYear?: number;
  readonly moonPhase?: number;
}

/**
 * The shipping presets.
 *
 * `bloodMoor` is the default mood and the one the whole art direction is
 * calibrated against: mid-morning, near-total stratus at low altitude, thick
 * wet aerosol, ground mist. Cold, flat, and *bright in the sky while dark on
 * the ground*, which is the specific overcast signature — not a uniform grey
 * wash. The deck is unbroken (`coverage` 1.0) — Act I's sky is stratus, not
 * scattered cumulus — but its *optical depth* still varies across the map, so
 * there is a soft bright region where the sun is behind the thinnest cloud.
 * That gradient is what stops an overcast sky reading as a painted dome, and it
 * comes out of the density field rather than from holes punched in the deck.
 */
export const TIME_OF_DAY_PRESETS: Readonly<Record<TimeOfDayPresetName, TimeOfDayPreset>> = {
  bloodMoor: {
    // 12.15, moved here from `buildFrameGraph`, which had been overriding the
    // preset's 10.30 on every construction. A preset named after a scene should
    // be the scene's actual time of day; two places holding two different
    // answers is how a test ends up grading a lighting condition that never
    // ships. At latitude 51 in early November, 10.30 puts the sun at ~15° and
    // 15° of atmosphere reddens the beam into gold no matter how much cloud is
    // in front of it — which is what was warming the skylight.
    hours: 12.15,
    latitudeDeg: 51,
    dayOfYear: 305, // early November: low sun, long shadows, no summer warmth.
    mood: {
      cloudCoverage: 1.0,
      // 40. A two-stream slab at `g = 0.85` transmits ~0.11 of the beam at
      // depth 62, ~0.155 at 40 and ~0.27 at 20. 62 extinguished the key to ~3%
      // of clear sky, which is a shadowless, formless, directionless image; 20
      // fixed the form and overcorrected the *colour*, because a deck that thin
      // passes enough of the low-sun beam to warm the skylight itself, and a
      // warm ambient is the one thing this scene's palette cannot have. 40 is
      // the value that keeps ~40% more key than the original — enough for the
      // cascades to ground objects and for terrain to hold its gradient — while
      // leaving the sky hemisphere cold. Taken together with
      // `Sky.CLOUD_FORWARD_MEMORY`, now 0.55, the directional term is still
      // more than double what it was.
      cloudOpticalDepth: 40,
      cloudBaseAltitude: 900,
      cloudThickness: 700,
      hazeDensity: 2.6,
      mistDensity: 0.34,
      skyIntensity: 1,
    },
  },
  dawn: {
    hours: 7.6,
    mood: {
      cloudCoverage: 0.66,
      cloudOpticalDepth: 30,
      cloudBaseAltitude: 1500,
      cloudThickness: 900,
      hazeDensity: 2.2,
      mistDensity: 0.5,
      skyIntensity: 1,
    },
  },
  clearNoon: {
    hours: 12.4,
    mood: {
      cloudCoverage: 0.22,
      cloudOpticalDepth: 22,
      cloudBaseAltitude: 2200,
      cloudThickness: 1100,
      hazeDensity: 1,
      mistDensity: 0,
      skyIntensity: 1,
    },
  },
  dusk: {
    hours: 16.4,
    latitudeDeg: 51,
    dayOfYear: 305,
    mood: {
      cloudCoverage: 0.6,
      cloudOpticalDepth: 30,
      cloudBaseAltitude: 1700,
      cloudThickness: 1000,
      hazeDensity: 2.1,
      mistDensity: 0.22,
      skyIntensity: 1,
    },
  },
  night: {
    hours: 1.4,
    moonPhase: 0.55,
    mood: {
      cloudCoverage: 0.42,
      cloudOpticalDepth: 26,
      cloudBaseAltitude: 1400,
      cloudThickness: 800,
      hazeDensity: 1.6,
      mistDensity: 0.26,
      skyIntensity: 1,
    },
  },
  storm: {
    hours: 15.0,
    mood: {
      cloudCoverage: 1.0,
      cloudOpticalDepth: 130,
      cloudBaseAltitude: 620,
      cloudThickness: 1600,
      hazeDensity: 3.4,
      mistDensity: 0.62,
      skyIntensity: 1,
    },
  },
};

export interface TimeOfDayOptions {
  /** Starting preset. Default `'bloodMoor'`. */
  preset?: TimeOfDayPresetName;
  /** Starting hour, overriding the preset's. */
  hours?: number;
  /** Observer latitude, degrees. Default 51 (a cold northern Europe analogue). */
  latitudeDeg?: number;
  /** Day of year `1..365`. Default 305 — early November. */
  dayOfYear?: number;
  /** Compass rotation applied to both bodies, degrees. Default 0. */
  northOffsetDeg?: number;
  /** Real seconds for one in-game day. Default 0 (clock is frozen). */
  dayLengthSeconds?: number;
  /** Lunar phase `[0,1)`; 0 new, 0.5 full. Default 0.72. */
  moonPhase?: number;
}

/** Service key. Registered by {@link module:render/Sky} if nothing else has. */
export const TimeOfDayKey = serviceKey<TimeOfDay>('render.timeOfDay');

/* -------------------------------------------------------------------------- */
/* Solar geometry — free functions so the tests can pin them                   */
/* -------------------------------------------------------------------------- */

const DEG = Math.PI / 180;

/**
 * Solar declination in radians for a day of year, per Spencer (1971).
 *
 * Peaks at about +23.44 deg near day 172 and bottoms out near day 355.
 */
export function solarDeclination(dayOfYear: number): number {
  const g = (2 * Math.PI * (dayOfYear - 1)) / 365;
  return (
    0.006918 -
    0.399912 * Math.cos(g) +
    0.070257 * Math.sin(g) -
    0.006758 * Math.cos(2 * g) +
    0.000907 * Math.sin(2 * g) -
    0.002697 * Math.cos(3 * g) +
    0.00148 * Math.sin(3 * g)
  );
}

/**
 * Atmospheric refraction lift in degrees for a true elevation, per Bennett
 * (1982). About 0.57 deg at the horizon, falling below 0.02 deg by 45 deg.
 *
 * Returns 0 well below the horizon, where the formula is not defined and the
 * body is not being drawn anyway.
 */
export function refractionLiftDeg(trueElevationDeg: number): number {
  if (trueElevationDeg < -1.5) return 0;
  const h = trueElevationDeg;
  const arcminutes = 1 / Math.tan((h + 7.31 / (h + 4.4)) * DEG);
  return Math.max(0, arcminutes / 60);
}

/** Horizon geometry for one body. Angles out in **degrees**. */
export interface HorizonAngles {
  elevationDeg: number;
  azimuthDeg: number;
}

/**
 * Convert (declination, hour angle, latitude) to apparent horizon coordinates.
 *
 * `hourAngleDeg` is 0 at upper transit and increases westward at 15 deg/h.
 * The returned azimuth is measured clockwise from north.
 */
export function horizonFromEquatorial(
  declinationRad: number,
  hourAngleDeg: number,
  latitudeDeg: number,
): HorizonAngles {
  const dec = declinationRad;
  const ha = hourAngleDeg * DEG;
  const lat = latitudeDeg * DEG;

  const sinEl = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(ha);
  const trueElevation = Math.asin(THREE.MathUtils.clamp(sinEl, -1, 1)) / DEG;

  // Azimuth measured from *south*, positive toward west. Written with atan2 so
  // it is continuous through the poles of the naive arccos form.
  const azSouth = Math.atan2(
    Math.sin(ha),
    Math.cos(ha) * Math.sin(lat) - Math.tan(dec) * Math.cos(lat),
  );

  return {
    elevationDeg: trueElevation + refractionLiftDeg(trueElevation),
    azimuthDeg: (azSouth / DEG + 180 + 360) % 360,
  };
}

/**
 * Horizon angles to a world-space unit vector.
 *
 * `+X` east, `+Y` up, `-Z` north.
 */
export function directionFromHorizon(
  elevationDeg: number,
  azimuthDeg: number,
  target?: THREE.Vector3,
): THREE.Vector3 {
  const out = target ?? new THREE.Vector3();
  const el = elevationDeg * DEG;
  const az = azimuthDeg * DEG;
  const cosEl = Math.cos(el);
  return out.set(cosEl * Math.sin(az), Math.sin(el), -cosEl * Math.cos(az)).normalize();
}

/* -------------------------------------------------------------------------- */
/* TimeOfDay                                                                   */
/* -------------------------------------------------------------------------- */

/** Angular radius of the sun's disc, degrees. */
export const SUN_ANGULAR_RADIUS_DEG = 0.2666;
/** Angular radius of the moon's disc, degrees. Very close to the sun's. */
export const MOON_ANGULAR_RADIUS_DEG = 0.2593;

interface MutableBody {
  direction: THREE.Vector3;
  elevationDeg: number;
  azimuthDeg: number;
  visibility: number;
}

export class TimeOfDay implements GameModule {
  readonly name = 'TimeOfDay';

  /** Observer latitude in degrees. Positive north. */
  latitudeDeg: number;
  /** Day of year, `1..365`. Drives the seasonal tilt of the sun's arc. */
  dayOfYear: number;
  /** Compass rotation applied to both bodies, degrees clockwise. */
  northOffsetDeg: number;
  /** Real seconds per in-game day. `0` freezes the clock. */
  dayLengthSeconds: number;
  /** Lunar phase `[0,1)`. 0 new, 0.5 full. */
  moonPhase: number;
  /** When true, {@link advance} does nothing. */
  paused = false;

  /** Current weather mood. Mutate freely; {@link module:render/Sky} polls it. */
  readonly mood: TimeOfDayMood;

  readonly #sun: MutableBody = {
    direction: new THREE.Vector3(0, 1, 0),
    elevationDeg: 90,
    azimuthDeg: 180,
    visibility: 1,
  };
  readonly #moon: MutableBody = {
    direction: new THREE.Vector3(0, -1, 0),
    elevationDeg: -90,
    azimuthDeg: 0,
    visibility: 0,
  };

  #hours: number;
  #preset: TimeOfDayPresetName;
  /** Manual sun override; when set, astronomy is bypassed. */
  #manualSun: HorizonAngles | null = null;
  #ctx: GameContext | null = null;
  /** Frame number of the last {@link advance}, so double-driving is a no-op. */
  #lastAdvancedFrame = -1;
  /** Bumped on every state change so {@link module:render/Sky} can poll cheaply. */
  #revision = 0;

  constructor(options: TimeOfDayOptions = {}) {
    const presetName = options.preset ?? 'bloodMoor';
    const preset = TIME_OF_DAY_PRESETS[presetName];

    this.#preset = presetName;
    this.mood = { ...preset.mood };
    this.latitudeDeg = options.latitudeDeg ?? preset.latitudeDeg ?? 51;
    this.dayOfYear = options.dayOfYear ?? preset.dayOfYear ?? 305;
    this.northOffsetDeg = options.northOffsetDeg ?? 0;
    this.dayLengthSeconds = options.dayLengthSeconds ?? 0;
    this.moonPhase = options.moonPhase ?? preset.moonPhase ?? 0.72;
    this.#hours = wrapHours(options.hours ?? preset.hours);

    this.#recompute();
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    if (!ctx.services.has(TimeOfDayKey)) ctx.services.register(TimeOfDayKey, this);
  }

  update(ctx: GameContext, dt: number): void {
    this.advance(dt, ctx);
  }

  dispose(): void {
    this.#ctx = null;
  }

  /* -- state ------------------------------------------------------------- */

  /** Local apparent solar time, `[0, 24)`. */
  get hours(): number {
    return this.#hours;
  }

  set hours(value: number) {
    this.setHours(value);
  }

  /** The preset most recently applied. Not invalidated by manual edits. */
  get preset(): TimeOfDayPresetName {
    return this.#preset;
  }

  /** Sun position. The object identity is stable; read the fields per frame. */
  get sun(): CelestialBody {
    return this.#sun;
  }

  /** Moon position. The object identity is stable. */
  get moon(): CelestialBody {
    return this.#moon;
  }

  /**
   * Monotonic counter bumped whenever anything that affects the sky changes.
   *
   * This is how {@link module:render/Sky} decides whether its LUTs are stale
   * without diffing a dozen fields or subscribing to an event that might be
   * emitted mid-frame.
   */
  get revision(): number {
    return this.#revision;
  }

  /**
   * `0` in full daylight, `1` in full night, smooth across civil twilight.
   *
   * Defined on the sun's *true* geometric elevation between +2 deg and -8 deg
   * (roughly the end of civil twilight), which is the window in which artificial
   * light, stars and the moon actually have to fade in.
   */
  get nightFactor(): number {
    return THREE.MathUtils.smoothstep(-this.#sun.elevationDeg, -2, 8);
  }

  /** Fraction of the moon's disc that is lit, `[0,1]`. */
  get moonIlluminatedFraction(): number {
    return (1 - Math.cos(2 * Math.PI * this.moonPhase)) * 0.5;
  }

  /* -- mutation ---------------------------------------------------------- */

  /**
   * Set the clock. Wraps into `[0,24)` and clears any manual sun override,
   * because a caller asking for 14:00 means "put the sun where 14:00 puts it".
   */
  setHours(value: number, emit = true): void {
    const previous = this.#hours;
    const next = wrapHours(value);
    if (next === previous) return;
    this.#hours = next;
    this.#manualSun = null;
    this.#recompute();
    if (emit) {
      this.#ctx?.events.emit('timeofday:changed', {
        hours: next,
        previousHours: previous,
        dayRolled: next < previous,
      });
    }
  }

  /** Apply a named mood: hour, astronomy overrides and weather together. */
  setPreset(name: TimeOfDayPresetName): void {
    const preset = TIME_OF_DAY_PRESETS[name];
    this.#preset = name;
    Object.assign(this.mood, preset.mood);
    if (preset.latitudeDeg !== undefined) this.latitudeDeg = preset.latitudeDeg;
    if (preset.dayOfYear !== undefined) this.dayOfYear = preset.dayOfYear;
    if (preset.moonPhase !== undefined) this.moonPhase = preset.moonPhase;
    this.#hours = wrapHours(preset.hours);
    this.#manualSun = null;
    this.#recompute();
    this.#ctx?.events.emit('timeofday:preset', { preset: name, mood: this.mood });
  }

  /**
   * Pin the sun to an exact elevation and azimuth, ignoring the clock.
   *
   * This exists because a director framing a shot wants the sun *there*, not at
   * whatever latitude and hour would produce it. The clock keeps running and
   * the moon keeps moving; only the sun is pinned, until the next
   * {@link setHours} or {@link setPreset}.
   */
  setSunAngles(elevationDeg: number, azimuthDeg: number): void {
    this.#manualSun = { elevationDeg, azimuthDeg };
    this.#recompute();
  }

  /** Whether {@link setSunAngles} is currently overriding the astronomy. */
  get hasManualSun(): boolean {
    return this.#manualSun !== null;
  }

  /**
   * Advance the clock by `dt` real seconds.
   *
   * Idempotent within an engine frame: whichever of `TimeOfDay.update` and
   * `Sky.update` runs first does the work, and the other is a no-op. That is
   * what lets `Sky` guarantee a working clock without forbidding the integrator
   * from registering `TimeOfDay` as a module in its own right.
   */
  advance(dt: number, ctx?: GameContext): void {
    const context = ctx ?? this.#ctx;
    const frame = context?.time.frame ?? -1;
    if (frame >= 0) {
      if (frame === this.#lastAdvancedFrame) return;
      this.#lastAdvancedFrame = frame;
    }
    if (this.paused || this.dayLengthSeconds <= 0 || dt <= 0) return;
    this.setHours(this.#hours + (dt / this.dayLengthSeconds) * 24);
  }

  /** Force a recomputation after mutating a public field directly. */
  refresh(): void {
    this.#recompute();
  }

  /* -- internals --------------------------------------------------------- */

  #recompute(): void {
    const declination = solarDeclination(this.dayOfYear);
    const sunHourAngle = (this.#hours - 12) * 15;

    const sunAngles =
      this.#manualSun ?? horizonFromEquatorial(declination, sunHourAngle, this.latitudeDeg);
    applyBody(this.#sun, sunAngles, this.northOffsetDeg, SUN_ANGULAR_RADIUS_DEG);

    // Moon: coincident with the sun at new, opposite it at full. It *lags* the
    // sun by one full turn per synodic cycle — which is exactly the statement
    // that a full moon transits at local midnight, a first-quarter moon at
    // 18:00, and a new moon at noon.
    const moonHourAngle = sunHourAngle - this.moonPhase * 360;
    const moonDeclination = -declination * Math.cos(2 * Math.PI * (this.moonPhase - 0.5));
    const moonAngles = horizonFromEquatorial(moonDeclination, moonHourAngle, this.latitudeDeg);
    applyBody(this.#moon, moonAngles, this.northOffsetDeg, MOON_ANGULAR_RADIUS_DEG);

    this.#revision++;
  }
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

function wrapHours(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return ((value % 24) + 24) % 24;
}

/**
 * Write horizon angles into a body, applying the compass offset and computing
 * the rise/set visibility ramp.
 *
 * The ramp spans the body's own angular diameter plus 0.35 deg of slop, so a
 * light driven from `visibility` fades over roughly the two minutes the disc
 * actually takes to clear the horizon rather than switching on in one frame.
 */
function applyBody(
  body: MutableBody,
  angles: HorizonAngles,
  northOffsetDeg: number,
  angularRadiusDeg: number,
): void {
  const azimuth = (((angles.azimuthDeg + northOffsetDeg) % 360) + 360) % 360;
  body.elevationDeg = angles.elevationDeg;
  body.azimuthDeg = azimuth;
  directionFromHorizon(angles.elevationDeg, azimuth, body.direction);
  const fade = angularRadiusDeg + 0.35;
  body.visibility = THREE.MathUtils.smoothstep(angles.elevationDeg, -fade, fade);
}
