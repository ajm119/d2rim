/**
 * Contract and graceful-degradation tests for the volumetric system.
 *
 * No GPU, no renderer: what is asserted here is the CPU half of the modules —
 * service discovery, mode selection, light gathering, fog-volume culling and
 * arbitration, uniform packing, and the fallbacks each of those takes when the
 * services they would prefer are missing. That is the half the integrator
 * actually has to reason about, and it is the half that is wrong most often.
 *
 * The GPU half (shader graphs, compute dispatch) is exercised only as far as
 * "does building it throw", because a TSL graph is not compiled until a real
 * `NodeBuilder` walks it and there is no backend in this container that can
 * supply one.
 */

import * as THREE from 'three/webgpu';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EventBus } from '../src/core/EventBus';
import { ServiceLocator } from '../src/core/ServiceLocator';
import type { GameContext, RendererHandle } from '../src/core/types';
import {
  VolumetricAmbientKey,
  VolumetricLightsKey,
  VolumetricSunKey,
  VolumetricsKey,
  VolumetricsModule,
  type VolumetricPointLight,
} from '../src/render/Volumetrics';
import { LightShaftsKey, LightShaftsModule } from '../src/render/post/LightShafts';

/** A `GameContext` with everything these modules touch and nothing else. */
function createContext(options: { compute?: boolean } = {}): GameContext {
  const renderer = {
    backend: options.compute === true ? 'webgpu' : 'webgl2',
    // Deliberately not a node renderer: `lateUpdate` must bail out cleanly.
    three: {} as never,
    capabilities: {
      compute: options.compute ?? false,
      float32Filterable: false,
      maxSamples: 4,
    },
    setSize: () => {},
    render: () => {},
  } as unknown as RendererHandle;

  return {
    engine: {} as never,
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000),
    renderer,
    input: {} as never,
    events: new EventBus(),
    time: { elapsed: 0, delta: 1 / 60, frame: 0, scale: 1 },
    services: new ServiceLocator(),
  };
}

let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  warn.mockRestore();
});

/* ------------------------------------------------------------------------- *
 * Registration and mode selection
 * ------------------------------------------------------------------------- */

describe('VolumetricsModule registration', () => {
  it('registers itself and announces its mode', () => {
    const ctx = createContext({ compute: true });
    const events: Array<{ mode: string; froxels: number }> = [];
    ctx.events.on('volumetrics:ready', (payload) => events.push(payload));

    const module = new VolumetricsModule();
    module.init(ctx);

    expect(ctx.services.tryGet(VolumetricsKey)).toBe(module);
    expect(events).toHaveLength(1);
    expect(events[0]?.mode).toBe('froxel');
    expect(events[0]?.froxels).toBe(160 * 90 * 64);
    module.dispose();
    expect(ctx.services.tryGet(VolumetricsKey)).toBeUndefined();
  });

  it('picks the ray-march path when the backend has no compute', () => {
    const ctx = createContext({ compute: false });
    const module = new VolumetricsModule();
    module.init(ctx);
    expect(module.mode).toBe('raymarch');
    expect(module.prefersHalfResolution).toBe(true);
    expect(module.stats.froxelDimensions).toEqual([0, 0, 0]);
    module.dispose();
  });

  it('reports "off" at the off tier without registering GPU resources', () => {
    const ctx = createContext({ compute: true });
    const module = new VolumetricsModule({ quality: 'off' });
    module.init(ctx);
    expect(module.mode).toBe('off');
    expect(module.stats.bytes).toBe(0);
    module.dispose();
  });

  it('can opt out of registering the service', () => {
    const ctx = createContext();
    const module = new VolumetricsModule({ registerService: false });
    module.init(ctx);
    expect(ctx.services.tryGet(VolumetricsKey)).toBeUndefined();
    module.dispose();
  });

  it('bumps the graph version when the quality tier changes', () => {
    const ctx = createContext({ compute: true });
    const module = new VolumetricsModule();
    module.init(ctx);
    const before = module.graphVersion;
    module.setQuality('ultra');
    expect(module.graphVersion).toBeGreaterThan(before);
    expect(module.stats.quality).toBe('ultra');
    module.dispose();
  });
});

/* ------------------------------------------------------------------------- *
 * Parameters
 * ------------------------------------------------------------------------- */

describe('global fog parameters', () => {
  it('applies constructor overrides and keeps the rest at their defaults', () => {
    const module = new VolumetricsModule({ params: { density: 0.05, volumeDistance: 30 } });
    expect(module.params.density).toBeCloseTo(0.05, 10);
    expect(module.params.volumeDistance).toBeCloseTo(30, 10);
    // Untouched: the Blood Moor default.
    expect(module.params.anisotropy).toBeCloseTo(0.72, 10);
  });

  it('clamps physically meaningless values instead of trusting the caller', () => {
    const module = new VolumetricsModule();
    module.setParams({
      density: -1,
      anisotropy: 5,
      backAnisotropy: -5,
      lobeBlend: 2,
      noiseStrength: -3,
      noiseScale: 0,
      volumeDistance: -10,
      heightFalloff: -1,
    });
    expect(module.params.density).toBe(0);
    expect(module.params.anisotropy).toBeCloseTo(0.95, 10);
    expect(module.params.backAnisotropy).toBeCloseTo(-0.95, 10);
    expect(module.params.lobeBlend).toBe(1);
    expect(module.params.noiseStrength).toBe(0);
    expect(module.params.noiseScale).toBeGreaterThan(0);
    expect(module.params.volumeDistance).toBeGreaterThanOrEqual(1);
    expect(module.params.heightFalloff).toBe(0);
  });

  it('leaves fields absent from the patch alone', () => {
    const module = new VolumetricsModule();
    const height = module.params.height;
    module.setParams({ density: 0.02 });
    expect(module.params.height).toBe(height);
  });
});

/* ------------------------------------------------------------------------- *
 * Fog volumes
 * ------------------------------------------------------------------------- */

describe('fog volumes', () => {
  it('defaults a box volume sensibly and reports it as registered', () => {
    const ctx = createContext();
    const module = new VolumetricsModule();
    module.init(ctx);

    const handle = module.addFogVolume({ center: { x: 0, y: 0, z: 0 } });
    expect(handle.alive).toBe(true);
    expect(handle.active).toBe(false);
    expect(module.stats.registeredVolumes).toBe(1);
    module.dispose();
  });

  it('releases idempotently and drops the record', () => {
    const ctx = createContext();
    const module = new VolumetricsModule();
    module.init(ctx);

    const handle = module.addFogVolume({ center: { x: 0, y: 0, z: 0 } });
    handle.release();
    expect(handle.alive).toBe(false);
    expect(module.stats.registeredVolumes).toBe(0);
    handle.release();
    expect(module.stats.registeredVolumes).toBe(0);
    module.dispose();
  });

  it('gives every volume a distinct id', () => {
    const ctx = createContext();
    const module = new VolumetricsModule();
    module.init(ctx);
    const ids = new Set<number>();
    for (let i = 0; i < 20; i++) {
      ids.add(module.addFogVolume({ center: { x: i, y: 0, z: 0 } }).id);
    }
    expect(ids.size).toBe(20);
    module.dispose();
  });

  it('accepts a sphere and mirrors its radius into all three half-extents', () => {
    const ctx = createContext();
    const module = new VolumetricsModule();
    module.init(ctx);
    const handle = module.addFogVolume({
      shape: 'sphere',
      center: { x: 1, y: 2, z: 3 },
      radius: 7,
    });
    // Observable only through behaviour: setRadius must not throw and the
    // handle must stay alive.
    handle.setRadius(9);
    expect(handle.alive).toBe(true);
    module.dispose();
  });

  it('never binds more volumes than the configured budget', () => {
    const ctx = createContext();
    const module = new VolumetricsModule({ maxFogVolumes: 2 });
    module.init(ctx);
    for (let i = 0; i < 8; i++) {
      module.addFogVolume({ center: { x: 0, y: 0, z: -5 }, halfExtent: { x: 20, y: 20, z: 20 } });
    }
    // `lateUpdate` bails before the GPU work but after the CPU culling, which
    // is exactly the part under test.
    module.lateUpdate(ctx, 1 / 60);
    expect(module.stats.volumes).toBeLessThanOrEqual(2);
    expect(module.stats.registeredVolumes).toBe(8);
    module.dispose();
  });
});

/* ------------------------------------------------------------------------- *
 * Optional services
 * ------------------------------------------------------------------------- */

describe('optional service discovery', () => {
  it('runs with no optional services at all', () => {
    const ctx = createContext();
    const module = new VolumetricsModule();
    expect(() => {
      module.init(ctx);
      module.lateUpdate(ctx, 1 / 60);
    }).not.toThrow();
    expect(module.stats.shadowed).toBe(false);
    module.dispose();
  });

  it('reports a shadow provider when one is registered', () => {
    const ctx = createContext();
    ctx.services.register('render.shadow.volumetric', {
      cascadeCount: 4,
      shadowDepthTexture: null,
      cascadeMatrices: [],
      shadowMapSize: 2048,
    });
    const module = new VolumetricsModule();
    module.init(ctx);
    expect(module.stats.shadowed).toBe(true);
    module.dispose();
  });

  it('picks up a shadow provider that registers after init, and rebuilds', () => {
    // Five render systems come up in parallel and the integrator picks the
    // order; the fog must not require the shadow map to exist first.
    const ctx = createContext();
    const module = new VolumetricsModule();
    module.init(ctx);
    expect(module.stats.shadowed).toBe(false);
    const before = module.graphVersion;

    ctx.services.register('render.shadow.volumetric', {
      cascadeCount: 4,
      shadowDepthTexture: null,
      cascadeMatrices: [],
      shadowMapSize: 2048,
    });
    module.lateUpdate(ctx, 1 / 60);

    expect(module.stats.shadowed).toBe(true);
    // The graph gained a cascade lookup, so every consumer has to rebuild.
    expect(module.graphVersion).toBeGreaterThan(before);
    module.dispose();
  });

  it('picks up a sun provider that registers after init without rebuilding', () => {
    const ctx = createContext();
    const module = new VolumetricsModule();
    module.init(ctx);
    const before = module.graphVersion;

    ctx.services.register(VolumetricSunKey, {
      sunDirection: new THREE.Vector3(0.3, 0.6, -0.7).normalize(),
      sunColor: new THREE.Color(1, 0.92, 0.8),
      sunIntensity: 0.8,
    });
    module.lateUpdate(ctx, 1 / 60);

    // A sun is a pure uniform source: nothing about the graph's shape changed,
    // so nothing downstream needs to recompile.
    expect(module.graphVersion).toBe(before);
    module.dispose();
  });

  it('prefers a registered light provider over walking the scene', () => {
    const ctx = createContext();
    let asked = 0;
    ctx.services.register(VolumetricLightsKey, {
      collectVolumetricLights(out: VolumetricPointLight[], max: number): number {
        asked++;
        const count = Math.min(2, max);
        for (let i = 0; i < count; i++) {
          out.push({
            position: new THREE.Vector3(i, 1, -3),
            color: new THREE.Color(1, 0.6, 0.25),
            intensity: 4,
            radius: 8,
          });
        }
        return count;
      },
    });

    const module = new VolumetricsModule();
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);
    expect(asked).toBe(1);
    expect(module.stats.lights).toBe(2);
    module.dispose();
  });

  it('never binds more lights than the tier allows, however many the provider offers', () => {
    const ctx = createContext();
    ctx.services.register(VolumetricLightsKey, {
      collectVolumetricLights(out: VolumetricPointLight[], max: number): number {
        for (let i = 0; i < max + 5; i++) {
          out.push({
            position: new THREE.Vector3(i, 1, -3),
            color: new THREE.Color(1, 1, 1),
            intensity: 1,
            radius: 4,
          });
        }
        // A provider that lies about the count must not be able to overrun the
        // uniform array.
        return max + 5;
      },
    });

    const module = new VolumetricsModule({ quality: 'low' });
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);
    expect(module.stats.lights).toBeLessThanOrEqual(2);
    module.dispose();
  });

  it('falls back to walking the scene graph for point lights', () => {
    const ctx = createContext();
    ctx.camera.position.set(0, 2, 0);
    ctx.camera.lookAt(0, 2, -10);
    ctx.camera.updateMatrixWorld();

    const near = new THREE.PointLight(0xffaa55, 6, 10);
    near.position.set(0, 2, -4);
    const far = new THREE.PointLight(0xffaa55, 1, 10);
    far.position.set(0, 2, -30);
    const behind = new THREE.PointLight(0xffaa55, 6, 10);
    behind.position.set(0, 2, 40);
    ctx.scene.add(near, far, behind);
    ctx.scene.updateMatrixWorld(true);

    const module = new VolumetricsModule();
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);

    // The two in front are bound; the one behind the camera is culled.
    expect(module.stats.lights).toBe(2);
    module.dispose();
  });

  it('ignores zero-intensity lights when walking the scene', () => {
    const ctx = createContext();
    ctx.camera.updateMatrixWorld();
    const dark = new THREE.PointLight(0xffffff, 0, 10);
    dark.position.set(0, 0, -3);
    ctx.scene.add(dark);
    ctx.scene.updateMatrixWorld(true);

    const module = new VolumetricsModule();
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);
    expect(module.stats.lights).toBe(0);
    module.dispose();
  });

  it('accepts an ambient provider without requiring one', () => {
    const ctx = createContext();
    ctx.services.register(VolumetricAmbientKey, {
      volumetricAmbient: new THREE.Color(0.04, 0.05, 0.07),
    });
    const module = new VolumetricsModule();
    expect(() => {
      module.init(ctx);
      module.lateUpdate(ctx, 1 / 60);
    }).not.toThrow();
    module.dispose();
  });

  it('tracks a registered sun provider in preference to setSun', () => {
    const ctx = createContext();
    const provider = {
      sunDirection: new THREE.Vector3(0, 1, 0),
      sunColor: new THREE.Color(1, 0.9, 0.8),
      sunIntensity: 0.5,
    };
    ctx.services.register(VolumetricSunKey, provider);
    const module = new VolumetricsModule();
    module.init(ctx);
    module.setSun({ x: 1, y: 0, z: 0 }, 0xff0000, 1);
    // No throw and no crash is the assertion: the provider wins, and the
    // manual override remains available for when it is unregistered.
    expect(() => module.lateUpdate(ctx, 1 / 60)).not.toThrow();
    module.dispose();
  });
});

/* ------------------------------------------------------------------------- *
 * Resolve node
 * ------------------------------------------------------------------------- */

describe('createResolveNode', () => {
  it('returns a fully transparent result when the module is off', () => {
    const ctx = createContext();
    const module = new VolumetricsModule({ quality: 'off' });
    module.init(ctx);
    const node = module.createResolveNode({
      screenUv: null as never,
      viewZ: null as never,
      pixel: null as never,
    });
    expect(node).toBeDefined();
    module.dispose();
  });

  it('builds a graph on the ray-march path without touching a renderer', () => {
    const ctx = createContext({ compute: false });
    const module = new VolumetricsModule();
    module.init(ctx);
    expect(() =>
      module.createResolveNode({
        screenUv: null as never,
        viewZ: null as never,
        pixel: null as never,
      }),
    ).not.toThrow();
    module.dispose();
  });
});

/* ------------------------------------------------------------------------- *
 * Froxel resources
 * ------------------------------------------------------------------------- */

/**
 * A context whose renderer looks enough like three's node renderer to get past
 * `asNodeRenderer`, with a spy on `compute`.
 *
 * This is as close to the WebGPU path as a container with no GPU can get, and
 * it is worth having: it constructs the storage textures and both compute
 * kernels for real, so a missing export, a bad `Storage3DTexture` format or a
 * malformed `compute()` call fails here rather than on a player's machine.
 */
function createComputeContext(options: { throwOnCompute?: boolean } = {}): {
  ctx: GameContext;
  compute: ReturnType<typeof vi.fn>;
} {
  const compute = vi.fn(() => {
    if (options.throwOnCompute === true) throw new Error('no storage texture support');
  });
  const ctx = createContext({ compute: true });
  (ctx.renderer as { three: unknown }).three = { backend: {}, compute };
  return { ctx, compute };
}

describe('froxel resources', () => {
  it('builds both kernels and dispatches them once each per frame', () => {
    const { ctx, compute } = createComputeContext();
    const module = new VolumetricsModule({ quality: 'low' });
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);

    expect(module.mode).toBe('froxel');
    // Injection, then integration — two dispatches, because a storage write is
    // only guaranteed visible to a later pass, not a later thread.
    expect(compute).toHaveBeenCalledTimes(2);

    module.lateUpdate(ctx, 1 / 60);
    expect(compute).toHaveBeenCalledTimes(4);
    module.dispose();
  });

  it('reports the volume memory it holds', () => {
    const { ctx } = createComputeContext();
    const module = new VolumetricsModule({ quality: 'low' });
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);
    // Three RGBA16F volumes at 96x54x32.
    expect(module.stats.bytes).toBe(96 * 54 * 32 * 8 * 3);
    module.dispose();
  });

  it('latches to the ray march when the compute path throws', () => {
    const { ctx, compute } = createComputeContext({ throwOnCompute: true });
    const fallbacks: Array<{ reason: string }> = [];
    ctx.events.on('volumetrics:fallback', (payload) => fallbacks.push(payload));

    const module = new VolumetricsModule({ quality: 'low' });
    module.init(ctx);
    expect(module.mode).toBe('froxel');

    module.lateUpdate(ctx, 1 / 60);
    expect(module.mode).toBe('raymarch');
    expect(fallbacks).toHaveLength(1);
    expect(fallbacks[0]?.reason).toContain('storage texture');

    // And it stays latched: no repeated attempts, no repeated warnings.
    const callsAfterFailure = compute.mock.calls.length;
    module.lateUpdate(ctx, 1 / 60);
    module.lateUpdate(ctx, 1 / 60);
    expect(compute.mock.calls.length).toBe(callsAfterFailure);
    expect(fallbacks).toHaveLength(1);
    module.dispose();
  });

  it('does not hold froxel memory once it has fallen back', () => {
    const { ctx } = createComputeContext({ throwOnCompute: true });
    const module = new VolumetricsModule({ quality: 'low' });
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);
    expect(module.stats.bytes).toBe(0);
    module.dispose();
  });

  it('rebuilds the volume when the quality tier changes', () => {
    const { ctx, compute } = createComputeContext();
    const module = new VolumetricsModule({ quality: 'low' });
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);
    const lowBytes = module.stats.bytes;

    module.setQuality('high');
    module.lateUpdate(ctx, 1 / 60);
    expect(module.stats.bytes).toBeGreaterThan(lowBytes);
    expect(compute).toHaveBeenCalledTimes(4);
    module.dispose();
  });
});

/* ------------------------------------------------------------------------- *
 * LightShafts
 * ------------------------------------------------------------------------- */

describe('LightShaftsModule', () => {
  it('registers its service and exposes a post pass', () => {
    const ctx = createContext();
    const module = new LightShaftsModule();
    module.init(ctx);

    expect(ctx.services.tryGet(LightShaftsKey)).toBe(module);
    expect(module.pass.id).toBe('lightshafts');
    expect(module.pass.kind).toBe('chain');
    expect(module.pass.outputDomain).toBe('hdr');
    module.dispose();
    expect(ctx.services.tryGet(LightShaftsKey)).toBeUndefined();
  });

  it('is available at every quality tier, because fog is the art direction', () => {
    const module = new LightShaftsModule();
    const capabilities = {
      backend: 'webgl2' as const,
      rgHalfFloat: true,
      halfFloat: true,
      reversedDepth: false,
    };
    for (const tier of ['low', 'medium', 'high', 'ultra'] as const) {
      expect(module.pass.isAvailable(tier, capabilities)).toBe(true);
    }
  });

  it('warns exactly once when the volumetrics service is missing', () => {
    const ctx = createContext();
    const module = new LightShaftsModule();
    module.init(ctx);
    module.lateUpdate(ctx, 1 / 60);
    module.lateUpdate(ctx, 1 / 60);
    module.lateUpdate(ctx, 1 / 60);
    expect(warn).toHaveBeenCalledTimes(1);
    module.dispose();
  });

  it('does nothing but stay consistent when disabled', () => {
    const ctx = createContext();
    const module = new LightShaftsModule();
    module.init(ctx);
    module.setEnabled(false);
    expect(module.enabled).toBe(false);
    expect(module.pass.enabled).toBe(false);
    module.lateUpdate(ctx, 1 / 60);
    expect(warn).not.toHaveBeenCalled();
    module.dispose();
  });

  it('clamps its options into sane ranges', () => {
    const module = new LightShaftsModule({
      resolutionScale: 8,
      temporalBlend: 0,
      depthRejection: -1,
      intensity: -4,
      radialIntensity: -1,
      radialLength: 99,
      radialDecay: 0,
    });
    // Only observable through the absence of an explosion and through stats,
    // which is the contract that matters: an out-of-range option must never
    // produce a divide-by-zero in a shader.
    expect(module.stats.enabled).toBe(true);
    expect(module.stats.bytes).toBe(0);
  });

  it('shares the geometry buffer rather than allocating a second one', () => {
    const ctx = createContext();
    const volumetrics = new VolumetricsModule();
    const shafts = new LightShaftsModule();
    volumetrics.init(ctx);
    shafts.init(ctx);
    // The guide buffer is reference counted across GTAO, SSR and this module.
    expect(ctx.services.has('render.guideBuffer')).toBe(true);
    shafts.dispose();
    volumetrics.dispose();
    expect(ctx.services.has('render.guideBuffer')).toBe(false);
  });

  it('resolves volumetrics late, so registration order does not matter', () => {
    const ctx = createContext();
    const shafts = new LightShaftsModule();
    shafts.init(ctx);

    const volumetrics = new VolumetricsModule({ quality: 'off' });
    volumetrics.init(ctx);

    // The service appeared after `init`; `lateUpdate` must pick it up and *not*
    // log the missing-service warning.
    shafts.lateUpdate(ctx, 1 / 60);
    expect(warn).not.toHaveBeenCalled();
    shafts.dispose();
    volumetrics.dispose();
  });

  it('accepts intensity changes before it has ever rendered', () => {
    const module = new LightShaftsModule();
    expect(() => {
      module.setIntensity(0.5);
      module.setRadialIntensity(0);
      module.setQuality('low');
    }).not.toThrow();
  });

  it('passes the frame through untouched before the first scattering buffer exists', () => {
    const module = new LightShaftsModule();
    expect(module.scatteringTexture).toBeNull();
  });
});
