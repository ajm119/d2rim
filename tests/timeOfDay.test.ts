/**
 * Tests for `render/TimeOfDay`.
 *
 * Solar position is one of the few parts of a renderer with an externally
 * checkable ground truth: the sun really is due south at local noon in the
 * northern hemisphere, it really does rise north of east in June, and the
 * declination really does peak near the solstice. Those are the assertions
 * here — not "the number did not change".
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import {
  MOON_ANGULAR_RADIUS_DEG,
  SUN_ANGULAR_RADIUS_DEG,
  TIME_OF_DAY_PRESETS,
  TimeOfDay,
  directionFromHorizon,
  horizonFromEquatorial,
  refractionLiftDeg,
  solarDeclination,
} from '../src/render/TimeOfDay';

const SUMMER_SOLSTICE = 172;
const WINTER_SOLSTICE = 355;
const SPRING_EQUINOX = 80;

describe('solarDeclination', () => {
  it('peaks near +23.44 deg at the summer solstice', () => {
    const degrees = (solarDeclination(SUMMER_SOLSTICE) * 180) / Math.PI;
    expect(degrees).toBeGreaterThan(23.0);
    expect(degrees).toBeLessThan(23.6);
  });

  it('bottoms out near -23.44 deg at the winter solstice', () => {
    const degrees = (solarDeclination(WINTER_SOLSTICE) * 180) / Math.PI;
    expect(degrees).toBeLessThan(-23.0);
    expect(degrees).toBeGreaterThan(-23.6);
  });

  it('crosses zero at the equinoxes', () => {
    expect(Math.abs((solarDeclination(SPRING_EQUINOX) * 180) / Math.PI)).toBeLessThan(0.6);
  });

  it('stays within the obliquity of the ecliptic all year', () => {
    for (let day = 1; day <= 365; day++) {
      expect(Math.abs((solarDeclination(day) * 180) / Math.PI)).toBeLessThanOrEqual(23.6);
    }
  });
});

describe('refractionLiftDeg', () => {
  it('lifts a body on the horizon by about 0.57 deg', () => {
    const lift = refractionLiftDeg(0);
    expect(lift).toBeGreaterThan(0.5);
    expect(lift).toBeLessThan(0.62);
  });

  it('is negligible near the zenith', () => {
    expect(refractionLiftDeg(85)).toBeLessThan(0.005);
  });

  it('decreases monotonically with elevation', () => {
    let previous = Infinity;
    for (let e = 0; e <= 90; e += 2) {
      const lift = refractionLiftDeg(e);
      expect(lift).toBeLessThanOrEqual(previous + 1e-9);
      previous = lift;
    }
  });
});

describe('horizonFromEquatorial', () => {
  it('puts the noon sun due south in the northern hemisphere', () => {
    const angles = horizonFromEquatorial(solarDeclination(SPRING_EQUINOX), 0, 51);
    expect(angles.azimuthDeg).toBeCloseTo(180, 4);
    // Equinox at 51N: noon elevation = 90 - latitude.
    expect(angles.elevationDeg).toBeGreaterThan(38.5);
    expect(angles.elevationDeg).toBeLessThan(39.6);
  });

  it('puts the noon sun due north in the southern hemisphere', () => {
    const angles = horizonFromEquatorial(solarDeclination(SPRING_EQUINOX), 0, -35);
    expect(angles.azimuthDeg % 360).toBeCloseTo(0, 3);
  });

  it('puts the sun east in the morning and west in the afternoon', () => {
    const declination = solarDeclination(SPRING_EQUINOX);
    const morning = horizonFromEquatorial(declination, -60, 51);
    const afternoon = horizonFromEquatorial(declination, 60, 51);
    expect(morning.azimuthDeg).toBeLessThan(180);
    expect(afternoon.azimuthDeg).toBeGreaterThan(180);
    // Symmetric about the meridian.
    expect(morning.azimuthDeg + afternoon.azimuthDeg).toBeCloseTo(360, 6);
    expect(morning.elevationDeg).toBeCloseTo(afternoon.elevationDeg, 9);
  });

  it('makes midsummer sunrise happen north of east at high latitude', () => {
    // At 51N in June the sun rises well north of due east — the reason a
    // fixed sun arc always looks wrong in a game with seasons.
    const declination = solarDeclination(SUMMER_SOLSTICE);
    let sunriseAzimuth = 90;
    for (let ha = -180; ha < 0; ha += 0.05) {
      const angles = horizonFromEquatorial(declination, ha, 51);
      if (angles.elevationDeg > 0) {
        sunriseAzimuth = angles.azimuthDeg;
        break;
      }
    }
    expect(sunriseAzimuth).toBeLessThan(55);
    expect(sunriseAzimuth).toBeGreaterThan(40);
  });

  it('gives a midnight sun above the horizon inside the Arctic Circle', () => {
    const angles = horizonFromEquatorial(solarDeclination(SUMMER_SOLSTICE), 180, 70);
    expect(angles.elevationDeg).toBeGreaterThan(0);
  });
});

describe('directionFromHorizon', () => {
  const near = (v: THREE.Vector3, x: number, y: number, z: number): void => {
    expect(v.x).toBeCloseTo(x, 9);
    expect(v.y).toBeCloseTo(y, 9);
    expect(v.z).toBeCloseTo(z, 9);
  };

  it('maps the compass onto the engine axes: +X east, -Z north', () => {
    near(directionFromHorizon(0, 0), 0, 0, -1); // north
    near(directionFromHorizon(0, 90), 1, 0, 0); // east
    near(directionFromHorizon(0, 180), 0, 0, 1); // south
    near(directionFromHorizon(0, 270), -1, 0, 0); // west
  });

  it('maps 90 deg elevation to straight up', () => {
    near(directionFromHorizon(90, 137), 0, 1, 0);
  });

  it('always returns a unit vector', () => {
    for (let e = -90; e <= 90; e += 17) {
      for (let a = 0; a < 360; a += 23) {
        expect(directionFromHorizon(e, a).length()).toBeCloseTo(1, 12);
      }
    }
  });
});

describe('TimeOfDay', () => {
  it('wraps the clock into [0, 24)', () => {
    const clock = new TimeOfDay();
    clock.setHours(26.5);
    expect(clock.hours).toBeCloseTo(2.5, 9);
    clock.setHours(-3);
    expect(clock.hours).toBeCloseTo(21, 9);
  });

  it('bumps its revision only when something actually changed', () => {
    const clock = new TimeOfDay();
    const before = clock.revision;
    clock.setHours(clock.hours);
    expect(clock.revision).toBe(before);
    clock.setHours(clock.hours + 1);
    expect(clock.revision).toBeGreaterThan(before);
  });

  it('agrees with the free functions it is built from', () => {
    const clock = new TimeOfDay({ hours: 15, latitudeDeg: 51, dayOfYear: 200 });
    const expected = horizonFromEquatorial(solarDeclination(200), (15 - 12) * 15, 51);
    expect(clock.sun.elevationDeg).toBeCloseTo(expected.elevationDeg, 9);
    expect(clock.sun.azimuthDeg).toBeCloseTo(expected.azimuthDeg, 9);
    expect(clock.sun.direction.length()).toBeCloseTo(1, 12);
  });

  it('rotates the compass with northOffsetDeg without touching elevation', () => {
    const a = new TimeOfDay({ hours: 9, northOffsetDeg: 0 });
    const b = new TimeOfDay({ hours: 9, northOffsetDeg: 90 });
    expect(b.sun.elevationDeg).toBeCloseTo(a.sun.elevationDeg, 9);
    expect(((b.sun.azimuthDeg - a.sun.azimuthDeg + 360) % 360)).toBeCloseTo(90, 6);
  });

  it('puts a full moon high at local midnight in winter', () => {
    // Full moon (phase 0.5) transits at midnight, and in winter it rides the
    // summer sun's high arc — the single most recognisable consequence of
    // modelling the moon as anti-solar rather than as a second sun.
    const clock = new TimeOfDay({ hours: 0, dayOfYear: WINTER_SOLSTICE, latitudeDeg: 51 });
    clock.moonPhase = 0.5;
    clock.refresh();
    expect(clock.moon.elevationDeg).toBeGreaterThan(55);
    expect(clock.moon.azimuthDeg).toBeCloseTo(180, 1);
    expect(clock.moonIlluminatedFraction).toBeCloseTo(1, 9);
  });

  it('puts a new moon with the sun and leaves it unlit', () => {
    const clock = new TimeOfDay({ hours: 12, dayOfYear: 200, latitudeDeg: 51 });
    clock.moonPhase = 0;
    clock.refresh();
    expect(clock.moon.direction.dot(clock.sun.direction)).toBeGreaterThan(0.99);
    expect(clock.moonIlluminatedFraction).toBeCloseTo(0, 9);
  });

  it('puts a first-quarter moon on the meridian at 18:00', () => {
    const clock = new TimeOfDay({ hours: 18, dayOfYear: SPRING_EQUINOX, latitudeDeg: 51 });
    clock.moonPhase = 0.25;
    clock.refresh();
    expect(clock.moon.azimuthDeg).toBeCloseTo(180, 1);
    expect(clock.moonIlluminatedFraction).toBeCloseTo(0.5, 6);
  });

  it('fades visibility across the horizon rather than switching it', () => {
    const clock = new TimeOfDay({ latitudeDeg: 51, dayOfYear: SPRING_EQUINOX });
    // Sweep the morning and find the window where visibility is partial.
    let partial = 0;
    for (let h = 5; h < 8; h += 0.002) {
      clock.setHours(h);
      const v = clock.sun.visibility;
      if (v > 0.02 && v < 0.98) partial++;
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    expect(partial).toBeGreaterThan(10);
  });

  it('reports night only after the sun is genuinely down', () => {
    const clock = new TimeOfDay({ latitudeDeg: 51, dayOfYear: SPRING_EQUINOX });
    clock.setSunAngles(30, 180);
    expect(clock.nightFactor).toBe(0);
    clock.setSunAngles(-2, 180);
    expect(clock.nightFactor).toBeGreaterThan(0);
    expect(clock.nightFactor).toBeLessThan(1);
    clock.setSunAngles(-20, 180);
    expect(clock.nightFactor).toBe(1);
  });

  it('honours a manual sun override until the clock is set again', () => {
    const clock = new TimeOfDay();
    clock.setSunAngles(12.5, 275);
    expect(clock.hasManualSun).toBe(true);
    expect(clock.sun.elevationDeg).toBeCloseTo(12.5, 9);
    expect(clock.sun.azimuthDeg).toBeCloseTo(275, 9);
    clock.setHours(9);
    expect(clock.hasManualSun).toBe(false);
  });

  it('advances at the configured day length', () => {
    const clock = new TimeOfDay({ hours: 6, dayLengthSeconds: 1200 });
    clock.advance(60); // one twentieth of a day
    expect(clock.hours).toBeCloseTo(7.2, 6);
    clock.paused = true;
    clock.advance(60);
    expect(clock.hours).toBeCloseTo(7.2, 6);
  });

  it('advances at most once per engine frame', () => {
    const clock = new TimeOfDay({ hours: 6, dayLengthSeconds: 1200 });
    const ctx = { time: { frame: 7, elapsed: 0, delta: 0, scale: 1 } };
    clock.advance(60, ctx as never);
    clock.advance(60, ctx as never); // same frame: must be a no-op
    expect(clock.hours).toBeCloseTo(7.2, 6);
    ctx.time.frame = 8;
    clock.advance(60, ctx as never);
    expect(clock.hours).toBeCloseTo(8.4, 6);
  });

  it('applies both halves of a preset — the hour and the weather', () => {
    const clock = new TimeOfDay({ preset: 'clearNoon' });
    clock.setPreset('storm');
    const storm = TIME_OF_DAY_PRESETS.storm;
    expect(clock.hours).toBeCloseTo(storm.hours, 9);
    expect(clock.mood.cloudOpticalDepth).toBe(storm.mood.cloudOpticalDepth);
    expect(clock.preset).toBe('storm');
    // Mutating the live mood must not corrupt the shared preset table.
    clock.mood.cloudOpticalDepth = 1;
    expect(TIME_OF_DAY_PRESETS.storm.mood.cloudOpticalDepth).toBe(storm.mood.cloudOpticalDepth);
  });

  it('ships presets whose key light is actually in the sky', () => {
    // A preset whose sun is below the horizon by accident is the kind of bug
    // that only shows up as "why is this shot black".
    for (const name of ['bloodMoor', 'dawn', 'clearNoon', 'dusk'] as const) {
      const clock = new TimeOfDay({ preset: name });
      expect(clock.sun.elevationDeg).toBeGreaterThan(0);
    }
    const night = new TimeOfDay({ preset: 'night' });
    expect(night.sun.elevationDeg).toBeLessThan(-10);
    expect(night.moon.elevationDeg).toBeGreaterThan(10);
    expect(night.nightFactor).toBe(1);
  });

  it('uses realistic angular radii for both discs', () => {
    // Sun and moon subtend almost exactly the same angle — which is why total
    // eclipses are possible, and why a disc shader can share its constants.
    expect(SUN_ANGULAR_RADIUS_DEG).toBeCloseTo(0.2666, 4);
    expect(MOON_ANGULAR_RADIUS_DEG).toBeCloseTo(0.2593, 4);
    expect(Math.abs(SUN_ANGULAR_RADIUS_DEG - MOON_ANGULAR_RADIUS_DEG)).toBeLessThan(0.02);
  });
});
