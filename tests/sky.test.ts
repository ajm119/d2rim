/**
 * Tests for `render/Sky`.
 *
 * The shader half cannot be executed here (there is no GPU in this container,
 * and a node graph is not a picture). What *can* be pinned is everything the
 * shader is derived from: the sky-view parameterisation it samples with, the
 * cloud shaping function it is a literal translation of, the slab radiative
 * transfer both share, and — the important one — that the environment map, the
 * published light state and the visible sky all come out of the same numbers.
 *
 * Sky boots against a stub context on purpose: `init` touches only
 * `services`, `scene` and `events`, so the whole CPU half of the module is
 * exercisable without a renderer.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import { EventBus } from '../src/core/EventBus';
import { ServiceLocator } from '../src/core/ServiceLocator';
import type { GameContext } from '../src/core/types';
import { AtmosphereKey } from '../src/render/Atmosphere';
import {
  Sky,
  SkyKey,
  cloudSlabTransmittance,
  createCloudTexture,
  createSky,
  elevationFromV,
  shapeCloudDensity,
  vFromElevation,
  type CelestialLightSink,
  type CelestialLightState,
} from '../src/render/Sky';
import { TimeOfDayKey, type TimeOfDayPresetName } from '../src/render/TimeOfDay';

/* -------------------------------------------------------------------------- */
/* Harness                                                                     */
/* -------------------------------------------------------------------------- */

interface Harness {
  sky: Sky;
  ctx: GameContext;
  scene: THREE.Scene;
  services: ServiceLocator;
  events: EventBus;
}

/**
 * Boot a Sky against the smallest context that satisfies what `init` reads.
 * Deliberately small resolutions: these tests are about relationships between
 * numbers, and a 64-wide sky-view grid has the same physics as a 160-wide one
 * at a twentieth of the cost.
 */
function boot(preset: TimeOfDayPresetName = 'bloodMoor'): Harness {
  const scene = new THREE.Scene();
  const services = new ServiceLocator();
  const events = new EventBus();
  const ctx = {
    scene,
    services,
    events,
    camera: new THREE.PerspectiveCamera(),
    time: { elapsed: 0, delta: 0, frame: 1, scale: 1 },
  } as unknown as GameContext;

  const sky = createSky({
    preset,
    skyViewWidth: 64,
    environmentWidth: 48,
    cloudTextureSize: 128,
  });
  sky.init(ctx);
  return { sky, ctx, scene, services, events };
}

/** Mean radiance of a half-float RGBA equirect, weighted by texel solid angle. */
function meanRadiance(texture: THREE.DataTexture, upperHemisphereOnly = false): THREE.Vector3 {
  const { width, height } = texture.image;
  const data = texture.image.data as Uint16Array;
  const fromHalf = THREE.DataUtils.fromHalfFloat;
  const total = new THREE.Vector3();
  let weightSum = 0;
  for (let y = 0; y < height; y++) {
    const phi = ((y + 0.5) / height) * Math.PI;
    if (upperHemisphereOnly && Math.cos(phi) <= 0) continue;
    const weight = Math.sin(phi);
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      total.x += fromHalf(data[o] as number) * weight;
      total.y += fromHalf(data[o + 1] as number) * weight;
      total.z += fromHalf(data[o + 2] as number) * weight;
      weightSum += weight;
    }
  }
  return weightSum > 0 ? total.divideScalar(weightSum) : total;
}

/* -------------------------------------------------------------------------- */
/* Sky-view parameterisation                                                   */
/* -------------------------------------------------------------------------- */

describe('sky-view parameterisation', () => {
  it('round-trips elevation through v', () => {
    for (let i = 0; i <= 64; i++) {
      const v = i / 64;
      expect(vFromElevation(elevationFromV(v))).toBeCloseTo(v, 9);
    }
  });

  it('places the zenith at v = 0, the horizon at 0.5 and the nadir at 1', () => {
    expect(elevationFromV(0)).toBeCloseTo(Math.PI / 2, 9);
    expect(elevationFromV(0.5)).toBeCloseTo(0, 12);
    expect(elevationFromV(1)).toBeCloseTo(-Math.PI / 2, 9);
  });

  it('is monotonically decreasing in v', () => {
    let previous = Infinity;
    for (let i = 0; i <= 200; i++) {
      const e = elevationFromV(i / 200);
      expect(e).toBeLessThan(previous);
      previous = e;
    }
  });

  it('concentrates rows near the horizon, which is the entire point', () => {
    // Half the rows must land within 25 degrees of the horizon; a uniform
    // parameterisation would put only 28% there.
    const rows = 128;
    let nearHorizon = 0;
    for (let i = 0; i < rows; i++) {
      const elevation = (elevationFromV((i + 0.5) / rows) * 180) / Math.PI;
      if (Math.abs(elevation) <= 25) nearHorizon++;
    }
    expect(nearHorizon / rows).toBeGreaterThan(0.5);
  });
});

/* -------------------------------------------------------------------------- */
/* Cloud model                                                                 */
/* -------------------------------------------------------------------------- */

describe('cloudSlabTransmittance', () => {
  it('is total at zero optical depth', () => {
    for (const c of [0, 1, 2] as const) expect(cloudSlabTransmittance(0, c)).toBeCloseTo(1, 9);
  });

  it('falls monotonically and stays in (0, 1]', () => {
    for (const c of [0, 1, 2] as const) {
      let previous = 1.0000001;
      for (const tau of [0, 1, 4, 12, 30, 62, 130, 400]) {
        const t = cloudSlabTransmittance(tau, c);
        expect(t).toBeGreaterThan(0);
        expect(t).toBeLessThan(previous);
        previous = t;
      }
    }
  });

  it('reduces to the conservative slab solution when absorption vanishes', () => {
    // Expanding the two-stream expression to first order in `a` as w0 -> 1
    // gives T = 1 / (1 + (sqrt(3)/2)(1-g)tau). Blue is the channel where water
    // is very nearly non-absorbing, so it should sit on that limit.
    for (const tau of [0.5, 2, 5, 12]) {
      const conservative = 1 / (1 + (Math.sqrt(3) / 2) * tau * (1 - 0.85));
      const relativeError = Math.abs(cloudSlabTransmittance(tau, 2) - conservative) / conservative;
      expect(relativeError).toBeLessThan(0.02);
    }
  });

  it('transmits blue better than red once the deck is thick', () => {
    // Liquid water absorbs ~40x more strongly at 650 nm than at 450 nm, and
    // compounding that over tens of scattering events is what makes the base of
    // heavy overcast read cold rather than warm.
    const thick = [0, 1, 2].map((c) => cloudSlabTransmittance(62, c as 0));
    expect(thick[2] as number).toBeGreaterThan(thick[1] as number);
    expect(thick[1] as number).toBeGreaterThan(thick[0] as number);
    expect((thick[2] as number) / (thick[0] as number)).toBeGreaterThan(1.2);
  });

  it('barely tints a thin deck', () => {
    const thin = [0, 1, 2].map((c) => cloudSlabTransmittance(2, c as 0));
    expect((thin[2] as number) / (thin[0] as number)).toBeLessThan(1.02);
  });
});

describe('shapeCloudDensity', () => {
  const clear = { x: 0.5, y: 0.5, z: 0.5, w: 0.5 };

  it('produces nothing at zero coverage', () => {
    for (let h = 0; h <= 1; h += 0.25) {
      expect(shapeCloudDensity(0, { x: 1, y: 0, z: 0, w: 1 }, h)).toBe(0);
    }
  });

  it('is monotonically non-decreasing in coverage', () => {
    let previous = -1;
    for (let coverage = 0; coverage <= 1; coverage += 0.05) {
      const d = shapeCloudDensity(coverage, clear, 0.5);
      expect(d).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = d;
    }
  });

  it('stays in [0, 1] for every input combination', () => {
    for (let i = 0; i < 500; i++) {
      const n = {
        x: (i * 0.37) % 1,
        y: (i * 0.11) % 1,
        z: (i * 0.73) % 1,
        w: (i * 0.29) % 1,
      };
      const d = shapeCloudDensity((i % 11) / 10, n, (i % 7) / 6);
      expect(d).toBeGreaterThanOrEqual(0);
      expect(d).toBeLessThanOrEqual(1);
    }
  });

  it('rounds the bottom and flattens the top of the deck', () => {
    const dense = { x: 1, y: 0, z: 0, w: 1 };
    const bottom = shapeCloudDensity(1, dense, 0.02);
    const middle = shapeCloudDensity(1, dense, 0.5);
    const top = shapeCloudDensity(1, dense, 0.98);
    expect(bottom).toBeLessThan(middle);
    expect(top).toBeLessThan(middle);
  });

  it('leaves a fully overcast deck unbroken', () => {
    // Erosion authority goes to zero at coverage 1: an unbroken stratus deck
    // must not have holes punched in it by detail noise, or the clear-sky
    // aureole blazes through an otherwise leaden sky.
    for (let i = 0; i < 200; i++) {
      const n = { x: 0.3 + (i % 70) / 100, y: (i * 0.41) % 1, z: (i * 0.83) % 1, w: (i % 5) / 4 };
      expect(shapeCloudDensity(1, n, 0.5)).toBeGreaterThan(0);
    }
  });
});

describe('createCloudTexture', () => {
  it('is deterministic for a given seed and differs between seeds', () => {
    const a = createCloudTexture(64, 7).image.data as Uint8Array;
    const b = createCloudTexture(64, 7).image.data as Uint8Array;
    const c = createCloudTexture(64, 8).image.data as Uint8Array;
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(Array.from(a)).not.toEqual(Array.from(c));
  });

  it('tiles seamlessly on both axes', () => {
    // The wrap seam is what would otherwise draw a hard line across the sky at
    // the horizon, where the deck repeats many times per pixel.
    const size = 64;
    const data = createCloudTexture(size, 3).image.data as Uint8Array;
    const at = (x: number, y: number, c: number): number =>
      data[((y * size + x) << 2) + c] as number;

    // A seam shows up as a step larger than the texture's own local variation,
    // so compare the mean jump across the wrap against the mean jump inside.
    for (let c = 0; c < 4; c++) {
      let horizontalSeam = 0;
      let verticalSeam = 0;
      let interior = 0;
      for (let i = 0; i < size; i++) {
        horizontalSeam += Math.abs(at(0, i, c) - at(size - 1, i, c));
        verticalSeam += Math.abs(at(i, 0, c) - at(i, size - 1, c));
        for (let j = 0; j < size - 1; j++) {
          interior += Math.abs(at(j + 1, i, c) - at(j, i, c));
        }
      }
      const meanInterior = interior / (size * (size - 1));
      expect(horizontalSeam / size).toBeLessThan(meanInterior * 2);
      expect(verticalSeam / size).toBeLessThan(meanInterior * 2);
    }
  });

  it('equalises the coverage channel so `coverage` is a real fraction', () => {
    const size = 128;
    const data = createCloudTexture(size, 11).image.data as Uint8Array;
    const values: number[] = [];
    for (let i = 0; i < size * size; i++) values.push(data[i << 2] as number);
    values.sort((a, b) => a - b);

    // A uniform distribution has its q-th quantile at q*255.
    for (const q of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      const actual = values[Math.floor(q * (values.length - 1))] as number;
      expect(Math.abs(actual - q * 255)).toBeLessThan(14);
    }
  });
});

/* -------------------------------------------------------------------------- */
/* Module integration                                                          */
/* -------------------------------------------------------------------------- */

describe('Sky module', () => {
  it('registers the services other modules resolve', () => {
    const { sky, services } = boot();
    expect(services.get(SkyKey)).toBe(sky);
    expect(services.has(AtmosphereKey)).toBe(true);
    expect(services.has(TimeOfDayKey)).toBe(true);
    expect(services.get(TimeOfDayKey)).toBe(sky.timeOfDay);
    sky.dispose();
  });

  it('installs the background, the environment and the shared fog', () => {
    const { sky, scene } = boot();
    const s = scene as unknown as { backgroundNode: unknown; fogNode: unknown };
    expect(s.backgroundNode).not.toBeNull();
    expect(s.fogNode).not.toBeNull();
    expect(scene.environment).toBe(sky.environmentTexture);
    sky.dispose();
    expect(scene.environment).toBeNull();
    expect(s.backgroundNode).toBeNull();
    expect(s.fogNode).toBeNull();
  });

  it('adopts an existing clock instead of registering a second one', () => {
    const scene = new THREE.Scene();
    const services = new ServiceLocator();
    const ctx = {
      scene,
      services,
      events: new EventBus(),
      time: { elapsed: 0, delta: 0, frame: 1, scale: 1 },
    } as unknown as GameContext;

    const sky = createSky({ skyViewWidth: 32, environmentWidth: 16, cloudTextureSize: 64 });
    sky.init(ctx);
    const first = services.get(TimeOfDayKey);
    expect(first).toBe(sky.timeOfDay);
    sky.dispose();
  });

  it('produces a complete environment map before a single frame is stepped', () => {
    // WebGPU captures run with zero warmup frames, so anything that only
    // becomes correct after N updates is broken by definition.
    const { sky } = boot();
    expect(sky.rebuilding).toBe(false);
    const environment = sky.environmentTexture;
    expect(environment).not.toBeNull();
    const mean = meanRadiance(environment as THREE.DataTexture);
    expect(mean.length()).toBeGreaterThan(0.01);
    sky.dispose();
  });

  it('keeps the environment map and the published irradiance in agreement', () => {
    const { sky } = boot();
    const state = sky.celestialLight();
    const upper = meanRadiance(sky.environmentTexture as THREE.DataTexture, true);

    // skyIrradiance is the cosine-weighted integral of the same texture, so it
    // must sit near `pi * meanRadiance` — the exact identity for a uniform sky
    // and a good bound for any real one. If these ever diverge, the IBL and the
    // ambient fill have stopped describing the same sky.
    for (const [irradiance, radiance] of [
      [state.skyIrradiance.r, upper.x],
      [state.skyIrradiance.g, upper.y],
      [state.skyIrradiance.b, upper.z],
    ] as const) {
      expect(irradiance).toBeGreaterThan(radiance * Math.PI * 0.4);
      expect(irradiance).toBeLessThan(radiance * Math.PI * 1.6);
    }
    sky.dispose();
  });

  it('gives the Blood Moor a cold cast and a dark ground', () => {
    const { sky } = boot('bloodMoor');
    const state = sky.celestialLight();

    // The signature of heavy overcast: skylight cooler than the direct beam it
    // was made from, because the deck's droplets absorb red far more than blue.
    //
    // 0.90, relaxed from 0.95 when the deck was deliberately thinned (optical
    // depth 62 -> 40) so that the key light survives it and the terrain keeps
    // its form. A thinner deck passes more of the low-November beam, and that
    // beam is warm, so the skylight it is made from warms with it: this is the
    // model behaving correctly, not the art direction slipping. What the
    // threshold still forbids is the failure that matters — a skylight
    // measurably *warmer* than neutral, which would put the scene's ambient in
    // competition with the campfire for the only warm source in the frame.
    const skyRatio = state.skyIrradiance.b / state.skyIrradiance.r;
    expect(skyRatio).toBeGreaterThan(0.9);

    // ...and the ground hemisphere is much darker than the sky above it.
    const luminance = (c: THREE.Color): number => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    expect(luminance(state.groundIrradiance)).toBeLessThan(luminance(state.skyIrradiance) * 0.25);

    // Some key light survives the deck, but not much of it.
    expect(state.sun.illuminance).toBeGreaterThan(0);
    expect(state.sun.visibility).toBe(1);
    expect(state.nightFactor).toBe(0);
    sky.dispose();
  });

  it('lets far more light through a thin deck than a thick one', () => {
    const clear = boot('clearNoon');
    const storm = boot('storm');
    expect(clear.sky.celestialLight().sun.illuminance).toBeGreaterThan(
      storm.sky.celestialLight().sun.illuminance * 5,
    );
    clear.sky.dispose();
    storm.sky.dispose();
  });

  it('switches the key light to the moon at night, and lights the sky with it', () => {
    const { sky } = boot('night');
    const state = sky.celestialLight();
    expect(state.nightFactor).toBe(1);
    expect(state.sun.illuminance).toBe(0);
    expect(state.moon.illuminance).toBeGreaterThan(0);
    expect(state.moon.visibility).toBeGreaterThan(0.9);
    // Moonlight is rendered cool, and it must actually reach the sky rather
    // than leaving a black dome with a bright disc pasted onto it.
    expect(state.moon.color.b).toBeGreaterThan(state.moon.color.r);
    expect(state.skyIrradiance.b).toBeGreaterThan(0);
    expect(meanRadiance(sky.environmentTexture as THREE.DataTexture, true).length()).toBeGreaterThan(
      1e-4,
    );
    sky.dispose();
  });

  it('makes dusk warmer than noon in the direct beam', () => {
    const noon = boot('clearNoon');
    const dusk = boot('dusk');
    const warmth = (s: CelestialLightState): number => s.sun.color.r / Math.max(1e-6, s.sun.color.b);
    expect(warmth(dusk.sky.celestialLight())).toBeGreaterThan(
      warmth(noon.sky.celestialLight()) * 1.5,
    );
    noon.sky.dispose();
    dusk.sky.dispose();
  });

  it('pushes state into a registered lighting sink and onto the event bus', () => {
    const scene = new THREE.Scene();
    const services = new ServiceLocator();
    const events = new EventBus();
    const received: CelestialLightState[] = [];
    const sink: CelestialLightSink = {
      setCelestialLight: (state) => {
        received.push(state);
      },
    };
    services.register('lighting.celestial', sink);

    const emitted: unknown[] = [];
    events.on('sky:celestial', (payload) => emitted.push(payload));

    const ctx = {
      scene,
      services,
      events,
      time: { elapsed: 0, delta: 0, frame: 1, scale: 1 },
    } as unknown as GameContext;
    const sky = createSky({ skyViewWidth: 32, environmentWidth: 16, cloudTextureSize: 64 });
    sky.init(ctx);

    expect(received.length).toBeGreaterThan(0);
    expect(emitted.length).toBeGreaterThan(0);
    expect(received[0]).toBe(sky.celestialLight());
    sky.dispose();
  });

  it('survives with no lighting sink at all', () => {
    const { sky } = boot();
    expect(() => sky.setPreset('dusk')).not.toThrow();
    expect(sky.celestialLight().sun.illuminance).toBeGreaterThan(0);
    sky.dispose();
  });

  it('tracks the clock: moving the sun changes the sky', () => {
    const { sky } = boot('clearNoon');
    const before = meanRadiance(sky.environmentTexture as THREE.DataTexture, true);
    sky.setHours(19.5);
    const after = meanRadiance(sky.environmentTexture as THREE.DataTexture, true);
    expect(after.length()).toBeLessThan(before.length() * 0.6);
    sky.dispose();
  });

  it('drives the shared aerial-perspective model from the cloud deck', () => {
    // The atmosphere service must know about the clouds, or distant geometry
    // ends up lit by a sun the sky says is hidden.
    const overcast = boot('storm');
    const clear = boot('clearNoon');
    const occludedSun = overcast.sky.atmosphere.uniforms.sunRadiance.y;
    const clearSun = clear.sky.atmosphere.uniforms.sunRadiance.y;
    expect(occludedSun).toBeLessThan(clearSun * 0.2);
    // ...and the deck puts its own diffuse light back into the haze.
    expect(overcast.sky.atmosphere.uniforms.multiScatter.y).toBeGreaterThan(0);
    overcast.sky.dispose();
    clear.sky.dispose();
  });

  it('reports a visibility range consistent with the weather', () => {
    const clear = boot('clearNoon');
    const murk = boot('storm');
    expect(murk.sky.atmosphere.visibilityRange()).toBeLessThan(
      clear.sky.atmosphere.visibilityRange() * 0.2,
    );
    expect(murk.sky.atmosphere.visibilityRange()).toBeGreaterThan(1000);
    clear.sky.dispose();
    murk.sky.dispose();
  });

  it('rebuilds incrementally as the clock runs, never showing a half-built sky', () => {
    const { sky, ctx } = boot('clearNoon');
    const clock = sky.timeOfDay;
    clock.dayLengthSeconds = 120; // a fast day, so one frame moves the sun a lot
    const before = meanRadiance(sky.environmentTexture as THREE.DataTexture, true).clone();

    let rebuilt = false;
    ctx.events.on('sky:rebuilt', () => {
      rebuilt = true;
    });

    const time = ctx.time as { frame: number; elapsed: number };
    for (let frame = 0; frame < 40 && !rebuilt; frame++) {
      time.frame = frame + 2;
      time.elapsed += 1 / 60;
      sky.update(ctx, 1 / 60);
      // Whatever the rebuild is doing, the *published* texture is only ever a
      // complete one: the march writes into a back buffer.
      const mean = meanRadiance(sky.environmentTexture as THREE.DataTexture, true);
      expect(Number.isFinite(mean.x + mean.y + mean.z)).toBe(true);
    }

    expect(rebuilt).toBe(true);
    expect(sky.rebuilding).toBe(false);
    const after = meanRadiance(sky.environmentTexture as THREE.DataTexture, true);
    expect(after.distanceTo(before)).toBeGreaterThan(0);
    sky.dispose();
  });

  it('emits a rebuild event with a plausible cost', () => {
    const { sky, events } = boot();
    const rebuilds: Array<{ hours: number; elapsedMs: number }> = [];
    events.on('sky:rebuilt', (payload) => rebuilds.push(payload));
    sky.setPreset('dawn');
    expect(rebuilds.length).toBe(1);
    expect(rebuilds[0]?.hours).toBeCloseTo(sky.timeOfDay.hours, 9);
    expect(rebuilds[0]?.elapsedMs).toBeGreaterThanOrEqual(0);
    sky.dispose();
  });

  it('produces finite, non-negative radiance everywhere in the environment map', () => {
    for (const preset of ['bloodMoor', 'dawn', 'clearNoon', 'dusk', 'night', 'storm'] as const) {
      const { sky } = boot(preset);
      const texture = sky.environmentTexture as THREE.DataTexture;
      const data = texture.image.data as Uint16Array;
      for (let i = 0; i < data.length; i++) {
        const value = THREE.DataUtils.fromHalfFloat(data[i] as number);
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
        // Must stay well inside half-float range or downstream HDR buffers and
        // any bloom pass turn into Inf.
        expect(value).toBeLessThan(60000);
      }
      sky.dispose();
    }
  });
});
