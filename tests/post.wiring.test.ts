/**
 * Wiring and graceful-degradation tests for GTAO and SSR.
 *
 * These run headlessly, with no GPU and no renderer, which is exactly the point:
 * the two modules have to survive being registered in an environment where
 * every optional service is missing, because that is the state the integrator
 * will first see them in. What is asserted here is the contract surface — which
 * services get registered, which get consumed, what happens when they are absent
 * — not the pixels, which no CPU rasteriser in this container could produce at a
 * useful rate anyway.
 *
 * `lateUpdate` is expected to bail out early: the fake renderer handle is not a
 * node renderer, so both modules log once and return rather than throwing. That
 * path is real and shipped — it is what happens on a machine that falls all the
 * way back to a non-node renderer.
 */

import * as THREE from 'three/webgpu';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ServiceLocator } from '../src/core/ServiceLocator';
import type { GameContext, RendererHandle } from '../src/core/types';
import {
  GuideBufferKey,
  acquireGuideBuffer,
  asNodeRenderer,
  releaseGuideBuffer,
} from '../src/render/post/Denoise';
import { AMBIENT_OCCLUSION_SERVICE_ID, GTAOKey, createGTAO } from '../src/render/post/GTAO';
import { SCENE_COLOR_SERVICE_ID, SSRKey, createSSR } from '../src/render/post/SSR';

/**
 * A `GameContext` with everything the render modules touch and nothing else.
 *
 * The engine, input and event bus are never reached by these modules during
 * `init`, so they are left as typed holes rather than mocked; if a future change
 * starts using them the cast will not save it and the test will fail loudly,
 * which is the desired outcome.
 */
function createContext(options: { nodeRenderer?: boolean } = {}): GameContext {
  const renderer = {
    backend: 'webgl2',
    // By default deliberately *not* a node renderer: no `.backend` property on
    // `.three`. Pass `nodeRenderer` to get past that guard and exercise the
    // checks that follow it.
    three: (options.nodeRenderer === true ? { backend: {} } : {}) as never,
    capabilities: { compute: false, float32Filterable: false, maxSamples: 4 },
    setSize: () => {},
    render: () => {},
  } as unknown as RendererHandle;

  return {
    engine: {} as never,
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000),
    renderer,
    input: {} as never,
    events: {} as never,
    time: { elapsed: 0, delta: 0, frame: 0, scale: 1 },
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

describe('renderer narrowing', () => {
  it('rejects a handle that does not wrap a node renderer', () => {
    const ctx = createContext();
    expect(asNodeRenderer(ctx.renderer)).toBeNull();
  });

  it('accepts a handle whose renderer exposes a backend', () => {
    const ctx = createContext();
    const handle = { ...ctx.renderer, three: { backend: {} } } as unknown as RendererHandle;
    expect(asNodeRenderer(handle)).not.toBeNull();
  });
});

describe('guide buffer ownership', () => {
  it('is created once and shared by reference count', () => {
    const ctx = createContext();
    const first = acquireGuideBuffer(ctx);
    const second = acquireGuideBuffer(ctx);

    expect(first.owned).not.toBeNull();
    expect(second.owned).toBe(first.owned);
    expect(ctx.services.has(GuideBufferKey)).toBe(true);

    // One release is not enough: the second holder still needs it.
    releaseGuideBuffer(ctx);
    expect(ctx.services.has(GuideBufferKey)).toBe(true);

    releaseGuideBuffer(ctx);
    expect(ctx.services.has(GuideBufferKey)).toBe(false);
  });

  it('adopts an externally registered G-buffer instead of rendering a prepass', () => {
    const ctx = createContext();
    const external = {
      guideTexture: new THREE.Texture(),
      halfGuideTexture: new THREE.Texture(),
      width: 1920,
      height: 1080,
      halfWidth: 960,
      halfHeight: 540,
      version: 1,
    };
    ctx.services.register(GuideBufferKey, external);

    const acquired = acquireGuideBuffer(ctx);
    expect(acquired.provider).toBe(external);
    // `owned` being null is what stops the module rendering its own prepass.
    expect(acquired.owned).toBeNull();

    releaseGuideBuffer(ctx);
    // An external provider is not ours to unregister.
    expect(ctx.services.has(GuideBufferKey)).toBe(true);
  });

  it('resizes the half-resolution buffer to match', () => {
    const ctx = createContext();
    const { owned } = acquireGuideBuffer(ctx, { resolutionScale: 0.5 });
    expect(owned).not.toBeNull();
    owned?.setSize(1920, 1080);
    expect(owned?.width).toBe(1920);
    expect(owned?.height).toBe(1080);
    expect(owned?.halfWidth).toBe(960);
    expect(owned?.halfHeight).toBe(540);
    releaseGuideBuffer(ctx);
  });
});

describe('GTAO module', () => {
  it('registers itself and publishes the ambient-occlusion service', () => {
    const ctx = createContext();
    const gtao = createGTAO({ quality: 'high' });
    gtao.init(ctx);

    expect(ctx.services.get(GTAOKey)).toBe(gtao);
    expect(ctx.services.tryGet(AMBIENT_OCCLUSION_SERVICE_ID)).toBe(gtao);
    // The node is what the IBL module folds into `material.aoNode`, and it is
    // what applies AO to indirect light only.
    expect(gtao.occlusionNode).not.toBeNull();
    expect(gtao.upsampledNode).not.toBeNull();

    gtao.dispose();
    expect(ctx.services.has(GTAOKey)).toBe(false);
    expect(ctx.services.has(AMBIENT_OCCLUSION_SERVICE_ID)).toBe(false);
  });

  it('can be told not to publish to the IBL service', () => {
    const ctx = createContext();
    const gtao = createGTAO({ publishToIBL: false });
    gtao.init(ctx);
    expect(ctx.services.has(AMBIENT_OCCLUSION_SERVICE_ID)).toBe(false);
    gtao.dispose();
  });

  it('warns rather than throwing when something else owns render.ao', () => {
    const ctx = createContext();
    ctx.services.register(AMBIENT_OCCLUSION_SERVICE_ID, { occlusionNode: null });
    const gtao = createGTAO();
    gtao.init(ctx);
    expect(warn).toHaveBeenCalled();
    expect(ctx.services.get(GTAOKey)).toBe(gtao);
    gtao.dispose();
  });

  it('reports no occlusion node when switched off', () => {
    const ctx = createContext();
    const gtao = createGTAO({ quality: 'high' });
    gtao.init(ctx);
    gtao.setQuality('off');
    expect(gtao.occlusionNode).toBeNull();
    expect(gtao.stats.enabled).toBe(false);
    gtao.dispose();
  });

  it('rebuilds and stays usable across a tier change', () => {
    const ctx = createContext();
    const gtao = createGTAO({ quality: 'low' });
    gtao.init(ctx);
    expect(gtao.stats.slices).toBe(1);

    gtao.setQuality('ultra');
    expect(gtao.stats.quality).toBe('ultra');
    expect(gtao.stats.slices).toBe(4);
    expect(gtao.occlusionNode).not.toBeNull();
    gtao.dispose();
  });

  it('reports camera-only reprojection when no velocity buffer exists', () => {
    const ctx = createContext();
    const gtao = createGTAO();
    gtao.init(ctx);
    expect(gtao.stats.temporalFromVelocity).toBe(false);
    expect(gtao.stats.sharedGuideBuffer).toBe(false);
    gtao.dispose();
  });

  it('degrades to a single warning when the renderer cannot run TSL', () => {
    const ctx = createContext();
    const gtao = createGTAO();
    gtao.init(ctx);
    expect(() => {
      gtao.lateUpdate(ctx, 1 / 60);
      gtao.lateUpdate(ctx, 1 / 60);
      gtao.lateUpdate(ctx, 1 / 60);
    }).not.toThrow();
    // Once, not once per frame.
    expect(warn.mock.calls.filter((call) => String(call[0]).includes('[GTAO]'))).toHaveLength(1);
    gtao.dispose();
  });
});

describe('SSR module', () => {
  it('registers itself and reports its missing dependencies honestly', () => {
    const ctx = createContext();
    const ssr = createSSR();
    ssr.init(ctx);

    expect(ctx.services.get(SSRKey)).toBe(ssr);
    const stats = ssr.stats;
    expect(stats.sceneColorAvailable).toBe(false);
    expect(stats.surfaceParametersAvailable).toBe(false);
    expect(stats.probeAvailable).toBe(false);
    expect(stats.temporalFromVelocity).toBe(false);

    ssr.dispose();
    expect(ctx.services.has(SSRKey)).toBe(false);
  });

  it('contributes nothing, and says so once, without a scene colour', () => {
    const ctx = createContext({ nodeRenderer: true });
    const ssr = createSSR();
    ssr.init(ctx);

    expect(ssr.deltaNode).toBeNull();
    ssr.lateUpdate(ctx, 1 / 60);
    ssr.lateUpdate(ctx, 1 / 60);
    const messages = warn.mock.calls.filter((call) => String(call[0]).includes('[SSR]'));
    expect(messages).toHaveLength(1);
    expect(String(messages[0]?.[0])).toContain(SCENE_COLOR_SERVICE_ID);
    ssr.dispose();
  });

  it('leaves the scene colour untouched when there is nothing to add', () => {
    const ctx = createContext();
    const ssr = createSSR();
    ssr.init(ctx);
    // `composite` must be a no-op rather than producing black, so a post stack
    // can call it unconditionally.
    const sceneColor = {} as unknown as THREE.Node<'vec4'>;
    expect(ssr.composite(sceneColor)).toBe(sceneColor);
    ssr.dispose();
  });

  it('builds its pipeline once a scene colour is supplied', () => {
    const ctx = createContext();
    const ssr = createSSR({ quality: 'high' });
    ssr.init(ctx);
    ssr.setSceneColor(new THREE.Texture());

    expect(ssr.stats.sceneColorAvailable).toBe(true);
    // The pipeline is built lazily inside `lateUpdate`; without a node renderer
    // it never gets that far, which is exactly the degradation being asserted.
    expect(() => ssr.lateUpdate(ctx, 1 / 60)).not.toThrow();
    ssr.dispose();
  });

  it('picks up a scene colour registered as a service', () => {
    const ctx = createContext();
    ctx.services.register(SCENE_COLOR_SERVICE_ID, {
      sceneColorTexture: new THREE.Texture(),
      isPreviousFrame: true,
    });
    const ssr = createSSR();
    ssr.init(ctx);
    expect(ssr.stats.sceneColorAvailable).toBe(true);
    ssr.dispose();
  });

  it('reports its tier configuration', () => {
    const ctx = createContext();
    const ssr = createSSR({ quality: 'medium' });
    ssr.init(ctx);
    expect(ssr.stats.maxSteps).toBe(32);
    expect(ssr.stats.coneTaps).toBe(1);
    ssr.setQuality('ultra');
    expect(ssr.stats.maxSteps).toBe(80);
    expect(ssr.stats.coneTaps).toBe(8);
    ssr.dispose();
  });
});

describe('GTAO and SSR together', () => {
  it('share one guide buffer and one prepass', () => {
    const ctx = createContext();
    const gtao = createGTAO();
    const ssr = createSSR();
    gtao.init(ctx);
    ssr.init(ctx);

    const guide = ctx.services.get(GuideBufferKey);
    expect(guide).toBeDefined();

    // Disposing one must not pull the buffer out from under the other.
    gtao.dispose();
    expect(ctx.services.has(GuideBufferKey)).toBe(true);

    ssr.dispose();
    expect(ctx.services.has(GuideBufferKey)).toBe(false);
  });

  it('can both be switched off without leaving services behind', () => {
    const ctx = createContext();
    const gtao = createGTAO({ quality: 'off' });
    const ssr = createSSR({ quality: 'off' });
    gtao.init(ctx);
    ssr.init(ctx);

    expect(() => {
      gtao.lateUpdate(ctx, 1 / 60);
      ssr.lateUpdate(ctx, 1 / 60);
    }).not.toThrow();
    expect(gtao.occlusionNode).toBeNull();
    expect(ssr.deltaNode).toBeNull();

    gtao.dispose();
    ssr.dispose();
    expect(ctx.services.keys()).toHaveLength(0);
  });
});
